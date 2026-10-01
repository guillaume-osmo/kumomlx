"""Double-single ("48-bit") matmul in MLX, and what it actually buys on this model.

Apple GPUs have no float64, so the standard way to exceed float32 is to carry each value as an
unevaluated sum of two float32s. Splitting each operand at the mantissa midpoint,

    a = a_hi + a_lo,  b = b_hi + b_lo,   a@b ~ a_hi@b_hi + (a_hi@b_lo + a_lo@b_hi)

makes every PRODUCT exact (11+11 = 22 bits fits in float32's 24), at 3x the matmuls. The
a_lo@b_lo term contributes ~2^-48 and is dropped. Note what this does NOT fix: the summation
over the contraction axis still accumulates in float32, so the gain is bounded by how much of
the error was product rounding rather than accumulation -- which is why this is measured, not
assumed.

The split is done by masking mantissa bits rather than by Dekker's `c - (c - a)`, which a
compiler is free to reassociate into nothing.
"""
from __future__ import annotations
import sys, time
import numpy as np, torch
from safetensors.torch import load_file
import os

# Repository root; override with KUMOMLX_ROOT. Weights and goldens are expected under
# $KUMOMLX_ROOT/export (see the README -- they are not distributed with this repo).
HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isdir(f"{HERE}/site"):            # optional vendored install of the recipe package
    sys.path.insert(0, f"{HERE}/site")
sys.path.insert(0, HERE)
import mlx.core as mx
import sdm.models as SM
from sdm.task import Task
import kumo_mlx as K

_MASK = mx.array(0xFFFFE000, mx.uint32)   # keep sign + exponent + top 11 mantissa bits


def split32(a: mx.array):
    """Exact split a = hi + lo. hi keeps 11 mantissa bits, so hi*hi is exact in float32."""
    hi = (a.view(mx.uint32) & _MASK).view(mx.float32)
    return hi, a - hi


def matmul48(a: mx.array, b: mx.array) -> mx.array:
    if a.dtype != mx.float32:
        return a @ b
    ah, al = split32(a); bh, bl = split32(b)
    return ah @ bh + (ah @ bl + al @ bh)


def linear48(x, weight, bias=None):
    y = matmul48(x, weight.T)
    return y if bias is None else y + bias


def sdpa48(q, k, v, scale):
    """Attention with the two matmuls in double-single; softmax stays float32."""
    attn = matmul48(q * scale, mx.swapaxes(k, -1, -2))
    return matmul48(mx.softmax(attn, axis=-1, precise=True), v)


if __name__ == "__main__":
    torch.set_num_threads(1)
    t = np.load(f"{HERE}/export/local_taps.npz"); x_np, y_np = t["icl_in_x"], t["icl_in_y"]
    SD = load_file(f"{HERE}/export/kumo_large_reg.safetensors")

    # exactness of the split itself
    rng = np.random.default_rng(0); probe = mx.array(rng.standard_normal(10000).astype(np.float32))
    hi, lo = split32(probe); mx.eval(hi, lo)
    print(f"split exact (hi+lo == a): {bool(mx.all((hi + lo) == probe).item())}")

    m = SM.KumoTabular(task=Task.regression, size="large", pretrained=False,
                       device="cpu").eval().to(torch.float64)
    m.load_state_dict({k: v.to(torch.float64) for k, v in SD.items()}, strict=True)
    with torch.no_grad():
        o = m.models[Task.regression].icl_block(
            torch.tensor(x_np, dtype=torch.float64), torch.tensor(y_np, dtype=torch.float64))
    truth = torch.as_tensor(getattr(o, "numerical", o)).double().numpy()

    def run(label, use_lin48, use_sdpa48):
        o_lin, o_sdpa = K._linear, K._sdpa
        if use_lin48: K._linear = linear48
        if use_sdpa48: K._sdpa = lambda q, k, v, s: sdpa48(q, k, v, s)
        try:
            w = {k[len("models.regression."):]: v for k, v in
                 mx.load(f"{HERE}/export/kumo_large_reg.safetensors").items()
                 if k.startswith("models.regression.")}
            mx.eval(list(w.values()))
            blk = K.ICLBlock(w, "icl_block")
            xs, ys = mx.array(x_np), mx.array(y_np)
            out = blk(xs, ys); mx.eval(out)                       # warm up
            t0 = time.perf_counter()
            for _ in range(3): out = blk(xs, ys); mx.eval(out)
            dt = (time.perf_counter() - t0) / 3
            e = np.abs(np.array(out).astype(np.float64) - truth).max()
            print(f"  {label:34s} max abs {e:.3e}   {dt*1000:7.1f} ms")
            return e
        finally:
            K._linear, K._sdpa = o_lin, o_sdpa

    print("\nagainst the float64 reference (torch, CPU):")
    base = run("float32 (as shipped)", False, False)
    a = run("48-bit linears", True, False)
    b = run("48-bit linears + attention", True, True)
    print(f"\n  torch's own float32 for comparison: 5.074e-06")
    print(f"  improvement: linears {base/a:.2f}x, linears+attention {base/b:.2f}x")
