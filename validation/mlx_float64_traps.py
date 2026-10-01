"""numpy float64 as the impartial referee, plus a check that MLX float64 is really float64."""
import sys
import numpy as np, torch
import os

# Repository root; override with KUMOMLX_ROOT. Weights and goldens are expected under
# $KUMOMLX_ROOT/export (see the README -- they are not distributed with this repo).
HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isdir(f"{HERE}/site"):            # optional vendored install of the recipe package
    sys.path.insert(0, f"{HERE}/site")
sys.path.insert(0, os.path.join(HERE, "src"))
import mlx.core as mx
mx.set_default_device(mx.cpu)

print("=== is MLX float64 actually float64 on CPU? ===")
# a sum that float32 cannot represent: 1 + 2^-30 summed many times
a = np.full(1 << 12, 1.0 + 2.0**-30, dtype=np.float64)
exact = np.float64(1 << 12) * (1.0 + 2.0**-30)
mlx_sum = float(mx.sum(mx.array(a)).item())
print(f"  sum in MLX f64 err {abs(mlx_sum-exact):.3e}   numpy f64 err {abs(a.sum()-exact):.3e}")
b = np.random.default_rng(0).standard_normal((512, 512))
mm_mlx = np.array(mx.array(b) @ mx.array(b))
print(f"  512x512 matmul MLX f64 vs numpy f64: {np.abs(mm_mlx - b@b).max():.3e}")
print(f"  (float32 would give ~1e-5 here)")
x32 = mx.array(np.array([1.0], dtype=np.float64))
print(f"  mx.array(np.float64) keeps dtype: {x32.dtype}")

print("\n=== RMSNorm: who is right? ===")
rng = np.random.default_rng(1)
xr = rng.standard_normal((8, 1024)); wr = rng.standard_normal(1024) * 0.1 + 1.0
for eps_name, eps in [("finfo f64 (2.2e-16)", 2.220446049250313e-16), ("1e-6", 1e-6)]:
    ref = xr / np.sqrt((xr**2).mean(-1, keepdims=True) + eps) * wr
    tt = torch.nn.RMSNorm(1024, eps=None, elementwise_affine=True).double()
    with torch.no_grad(): tt.weight.copy_(torch.tensor(wr))
    to = tt(torch.tensor(xr)).detach().numpy()
    mo = np.array(mx.array(xr) * mx.rsqrt(mx.mean(mx.square(mx.array(xr)), axis=-1, keepdims=True)
                                          + eps) * mx.array(wr))
    print(f"  eps={eps_name:22s} numpy-vs-torch {np.abs(ref-to).max():.3e}   "
          f"numpy-vs-mlx {np.abs(ref-mo).max():.3e}")
