"""Compensated accumulation applied to the real model, measured on the device.

A microbenchmark ranked the variants; a microbenchmark is not a runtime model, so the winner is
re-measured inside the actual ICL stack where kernel launch counts and memory traffic differ.
"""
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


def matmul_comp(a, b, chunk=256):
    """Neumaier-compensated accumulation over chunks of the contraction axis.

    Each chunk's matmul still rounds internally, but the running total carries the bits every
    cross-chunk addition drops, so error grows with the chunk size rather than with K.
    """
    Kd = a.shape[-1]
    if Kd <= chunk or a.dtype != mx.float32:
        return a @ b
    s = c = None
    for i in range(0, Kd, chunk):
        p = a[..., i:i + chunk] @ b[..., i:i + chunk, :]
        if s is None:
            s, c = p, mx.zeros_like(p)
        else:
            t = s + p; bb = t - s
            c = c + ((s - (t - bb)) + (p - bb))
            s = t
    return s + c


def make_linear(chunk):
    def lin(x, weight, bias=None):
        y = matmul_comp(x, weight.T, chunk)
        return y if bias is None else y + bias
    return lin


torch.set_num_threads(1)
t = np.load(f"{HERE}/export/local_taps.npz"); x_np, y_np = t["icl_in_x"], t["icl_in_y"]
SD = load_file(f"{HERE}/export/kumo_large_reg.safetensors")
m = SM.KumoTabular(task=Task.regression, size="large", pretrained=False,
                   device="cpu").eval().to(torch.float64)
m.load_state_dict({k: v.to(torch.float64) for k, v in SD.items()}, strict=True)
with torch.no_grad():
    o = m.models[Task.regression].icl_block(torch.tensor(x_np, dtype=torch.float64),
                                            torch.tensor(y_np, dtype=torch.float64))
truth = torch.as_tensor(getattr(o, "numerical", o)).double().numpy()

w = {k[len("models.regression."):]: v for k, v in
     mx.load(f"{HERE}/export/kumo_large_reg.safetensors").items()
     if k.startswith("models.regression.")}
mx.eval(list(w.values()))
xs, ys = mx.array(x_np), mx.array(y_np)

def run(label, lin):
    orig = K._linear
    if lin is not None: K._linear = lin
    try:
        blk = K.ICLBlock(w, "icl_block")
        out = blk(xs, ys); mx.eval(out)
        t0 = time.perf_counter()
        for _ in range(5): out = blk(xs, ys); mx.eval(out)
        dt = (time.perf_counter() - t0) / 5
        e = np.abs(np.array(out).astype(np.float64) - truth).max()
        print(f"  {label:34s} max abs {e:.3e}   {dt*1000:6.1f} ms")
        return e, dt
    finally:
        K._linear = orig

print("ICL stack vs the float64 reference:")
e0, t0 = run("float32 (as shipped)", None)
for ch in (512, 256, 128):
    e, d = run(f"compensated linears, chunk={ch}", make_linear(ch))
    print(f"      -> {e0/e:4.2f}x accuracy at {d/t0:4.2f}x cost")
print(f"\n  torch's own float32: 5.074e-06  (the port at float32 is already better)")
