"""Are the two sides even holding the same numbers?

Every comparison so far assumed load_state_dict(strict=False) actually loaded the checkpoint.
strict=False silently leaves any unmatched parameter at its random init, and RMSNorm inits to
ones -- which would look like a small, structural, precision-independent error. Exactly what we see.
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

sd = load_file(f"{HERE}/export/kumo_large_reg.safetensors")
m = SM.KumoTabular(task=Task.regression, size="large", pretrained=False, device="cpu").eval()
res = m.load_state_dict(sd, strict=False)
print(f"checkpoint tensors: {len(sd)}")
print(f"MISSING keys (left at random init): {len(res.missing_keys)}")
for k in res.missing_keys[:12]: print("   ", k)
print(f"UNEXPECTED keys (in file, unused): {len(res.unexpected_keys)}")
for k in res.unexpected_keys[:12]: print("   ", k)

# do the loaded torch params equal the MLX ones, bit for bit?
w = K.load_weights(f"{HERE}/export/kumo_large_reg.safetensors")
reg = m.models[Task.regression]
tp = dict(reg.named_parameters())
bad = []
for name, p in tp.items():
    if name in w:
        d = np.abs(p.detach().numpy() - np.array(w[name])).max()
        if d > 0: bad.append((d, name, float(p.detach().abs().mean())))
    else:
        bad.append((float("inf"), name + "  [ABSENT from MLX dict]", float(p.detach().abs().mean())))
bad.sort(reverse=True)
print(f"\ntorch params not matching MLX: {len(bad)} of {len(tp)}")
for d, name, mag in bad[:15]:
    print(f"   {d:10.3e}  {name:55s} mean|w| {mag:.4f}")
