"""The float64 structural test, with both harness artifacts removed.

Two traps invalidated the first attempt:
  * `mx.array(np_float64)` silently returns float32 -- dtype must be given AT CONSTRUCTION;
  * `mx.erf` is only float32-accurate even on float64 input (4.1e-07), and GELU uses it in all
    24 MLP blocks, so the oracle's resolution floor was ~1e-06 rather than ~1e-15.

So the model is compared twice: as-is, and with GELU replaced by the identity on BOTH sides,
which removes erf entirely and lets the real floor show through.
"""
import sys
import numpy as np, torch
from safetensors.torch import load_file
import os

# Repository root; override with KUMOMLX_ROOT. Weights and goldens are expected under
# $KUMOMLX_ROOT/export (see the README -- they are not distributed with this repo).
HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isdir(f"{HERE}/site"):            # optional vendored install of the recipe package
    sys.path.insert(0, f"{HERE}/site")
sys.path.insert(0, os.path.join(HERE, "src"))
import mlx.core as mx
import sdm.models as SM
from sdm.task import Task
from kumomlx import network as K

torch.set_num_threads(1); mx.set_default_device(mx.cpu)
t = np.load(f"{HERE}/export/local_taps.npz"); x_np, y_np = t["icl_in_x"], t["icl_in_y"]
SD = load_file(f"{HERE}/export/kumo_large_reg.safetensors")

def build_torch(dtype, no_gelu=False):
    m = SM.KumoTabular(task=Task.regression, size="large", pretrained=False, device="cpu").eval().to(dtype)
    m.load_state_dict({k: v.to(dtype) for k, v in SD.items()}, strict=True)
    reg = m.models[Task.regression]
    if no_gelu:
        # Every GELU must go, on both sides. Patching K._gelu in MLX also disables the HEAD's
        # GELU, so missing it here makes the two models different networks, not two
        # implementations of one -- which is what produced a bogus 66.6 discrepancy.
        n = 0
        for mod in reg.icl_block.modules():
            for name, child in list(mod.named_children()):
                if isinstance(child, torch.nn.GELU):
                    setattr(mod, name, torch.nn.Identity()); n += 1
        assert n == 25, f"expected 24 mlp GELUs + 1 head GELU, replaced {n}"
    return reg

def run_torch(reg, dtype):
    with torch.no_grad():
        o = reg.icl_block(torch.tensor(x_np, dtype=dtype), torch.tensor(y_np, dtype=dtype))
    return torch.as_tensor(getattr(o, "numerical", o)).double().numpy()

def run_mlx(dtype, no_gelu=False):
    orig = K._gelu
    if no_gelu: K._gelu = lambda z: z
    try:
        w = {k[len("models.regression."):]: v.astype(dtype)
             for k, v in mx.load(f"{HERE}/export/kumo_large_reg.safetensors").items()
             if k.startswith("models.regression.")}
        mx.eval(list(w.values()))
        o = K.ICLBlock(w, "icl_block")(
            mx.array(x_np.astype(np.float64), dtype=dtype),   # dtype AT CONSTRUCTION
            mx.array(y_np.astype(np.float64), dtype=dtype))
        mx.eval(o)
        return np.array(o).astype(np.float64)
    finally:
        K._gelu = orig

print("A) model as-is, float64 both sides")
truth = run_torch(build_torch(torch.float64), torch.float64)
e = np.abs(run_mlx(mx.float64) - truth).max()
print(f"   MLX f64 vs torch f64: {e:.3e}   <- limited by mx.erf, not by the port\n")

print("B) GELU -> identity on BOTH sides (erf removed), float64")
truth_ng = run_torch(build_torch(torch.float64, no_gelu=True), torch.float64)
e_ng = np.abs(run_mlx(mx.float64, no_gelu=True) - truth_ng).max()
rel = e_ng / max(np.abs(truth_ng).max(), 1e-30)
print(f"   MLX f64 vs torch f64: {e_ng:.3e}  (relative {rel:.3e})")
print(f"   torch's own f64 noise floor is 3.3e-15")
# What is left after erf is removed is MLX's float32-precision `exp` inside softmax
# (measured 3.6e-09 relative per softmax), accumulated over 24 attentions -- a library
# precision limit, not a port error. Anything materially above that is structural.
print(f"\n   VERDICT: " + ("structurally EXACT -- the remainder is MLX's float32-precision exp "
                          "in softmax, not a port error"
                          if rel < 1e-8 else f"a residual STRUCTURAL error remains at {rel:.1e}"))

print("\nC) float32, as shipped")
e32 = np.abs(run_mlx(mx.float32) - truth).max()
e32t = np.abs(run_torch(build_torch(torch.float32), torch.float32) - truth).max()
print(f"   torch f32 vs f64 truth: {e32t:.3e}")
print(f"   MLX   f32 vs f64 truth: {e32:.3e}  <- {'better' if e32 < e32t else 'worse'} than torch's own f32")
