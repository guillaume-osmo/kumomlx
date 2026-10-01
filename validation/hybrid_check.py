"""The shipped hybrid against pure torch, after the eps fix."""
import sys, time
import numpy as np, torch
import os

# Repository root; override with KUMOMLX_ROOT. Weights and goldens are expected under
# $KUMOMLX_ROOT/export (see the README -- they are not distributed with this repo).
HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isdir(f"{HERE}/site"):            # optional vendored install of the recipe package
    sys.path.insert(0, f"{HERE}/site")
sys.path.insert(0, os.path.join(HERE, "src"))
import mlx.core as mx
import sdm.models as sdm_models
from safetensors.torch import load_file
from sdm.task import Task
from kumomlx import KumoMLX

g = np.load(f"{HERE}/export/golden.npz")
xc, yc, xq = g["x_context"], g["y_context"], g["x_query"]

torch.set_num_threads(1)
tm = sdm_models.KumoTabular(task=Task.regression, size="large", pretrained=False, device="cpu").eval()
tm.load_state_dict(load_file(f"{HERE}/export/kumo_large_reg.safetensors"), strict=True)
def torch_pred():
    with torch.no_grad():
        out = tm(x_context=torch.tensor(xc), y_context=torch.tensor(yc).unsqueeze(-1),
                 x_query=torch.tensor(xq))
    return torch.as_tensor(getattr(out, "numerical", out)).float().numpy().reshape(len(xq), -1).mean(1)
t0 = time.perf_counter(); ref = torch_pred(); t_torch = time.perf_counter() - t0

model = KumoMLX(weights=f"{HERE}/export/kumo_large_reg.safetensors")
got = model.predict(xc, yc, xq)
t0 = time.perf_counter(); got = model.predict(xc, yc, xq); t_mlx = time.perf_counter() - t0

d = np.abs(got - ref)
print(f"hybrid vs torch: max abs {d.max():.3e}   rms {np.sqrt((d**2).mean()):.3e}")
print(f"prediction spread: {ref.min():.3f} to {ref.max():.3f}  (sd {ref.std():.3f})")
print(f"error as a fraction of the spread: {d.max()/ref.std():.2%}")
print(f"speed: MLX {t_mlx:.3f}s vs torch {t_torch:.3f}s = {t_torch/t_mlx:.1f}x")
torch.set_num_threads(8)
print(f"\ntorch 1-thread vs 8-thread (its own reduction-order spread): "
      f"{np.abs(torch_pred()-ref).max():.3e}")
