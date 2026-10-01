"""The shipped hybrid against pure torch.

The comparison only means anything with the recipe's generator SEEDED. The recipe draws a latin
column shuffle and a per-column transform rotation on every call, so unseeded, torch disagrees
with ITSELF by ~2e-02 -- larger than any port error, and enough to make a float32 explanation
look convincing when nothing is being measured but noise.
"""
import os
import sys

import numpy as np
import torch

# Repository root; override with KUMOMLX_ROOT. Weights and goldens are expected under
# $KUMOMLX_ROOT/export (see the README -- they are not distributed with this repo).
HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isdir(f"{HERE}/site"):            # optional vendored install of the recipe package
    sys.path.insert(0, f"{HERE}/site")
sys.path.insert(0, os.path.join(HERE, "src"))

import sdm.models as sdm_models  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from sdm.task import Task  # noqa: E402

from kumomlx import KumoMLX  # noqa: E402
from kumomlx import weights as W  # noqa: E402

torch.set_num_threads(1)
g = np.load(f"{HERE}/export/golden.npz")
xc, yc, xq = g["x_context"], g["y_context"], g["x_query"]
path = W.resolve()

tm = sdm_models.KumoTabular(task=Task.regression, size="large",
                            pretrained=False, device="cpu").eval()
sd = load_file(str(path))
if not any(k.startswith("models.") for k in sd):
    sd = {f"models.regression.{k}": v for k, v in sd.items()}
tm.load_state_dict(sd, strict=False)


def torch_pred(seed):
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    with torch.no_grad():
        out = tm(x_context=torch.tensor(xc), y_context=torch.tensor(yc).unsqueeze(-1),
                 x_query=torch.tensor(xq), generator=gen)
    return torch.as_tensor(getattr(out, "numerical", out)).float().numpy(
        ).reshape(len(xq), -1).mean(1)


ref = torch_pred(0)
model = KumoMLX(weights=path)
got = model.predict(xc, yc, xq, seed=0)

print(f"torch vs torch, seeded      : {np.abs(torch_pred(0) - ref).max():.3e}  (deterministic)")
print(f"torch vs torch, UNSEEDED    : {np.abs(torch_pred(None) - torch_pred(None)).max():.3e}"
      f"  <- the recipe's own randomness")
print(f"MLX hybrid vs torch, seeded : {np.abs(got - ref).max():.3e}")
print(f"prediction spread (sd)      : {ref.std():.3f}")
print(f"port error / spread         : {np.abs(got - ref).max() / ref.std():.4%}")
