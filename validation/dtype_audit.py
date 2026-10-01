"""Before trusting the f64 oracle: confirm both sides really compute in float64.

A disagreement of 1.9e-6 between two float64 evaluations is implausible for structurally
identical code -- it is the magnitude of float32 rounding, so the first hypothesis is that one
path silently stayed in float32.
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
sys.path.insert(0, HERE)
import mlx.core as mx
import sdm.models as SM
from sdm.task import Task
import kumo_mlx as K

t = np.load(f"{HERE}/export/local_taps.npz"); x_np, y_np = t["icl_in_x"], t["icl_in_y"]

m = SM.KumoTabular(task=Task.regression, size="large", pretrained=False, device="cpu").eval()
m.load_state_dict(load_file(f"{HERE}/export/kumo_large_reg.safetensors"), strict=False)
m = m.to(torch.float64)
reg = m.models[Task.regression]
pd = {p.dtype for p in reg.parameters()}; bd = {b.dtype for b in reg.buffers()}
print("torch param dtypes:", pd, " buffer dtypes:", bd)

seen = {}
def hook(mod, inp, out):
    o = out[0] if isinstance(out, tuple) else out
    o = getattr(o, "numerical", o)
    if torch.is_tensor(o):
        seen.setdefault(type(mod).__name__, set()).add(str(o.dtype))
hs = [mod.register_forward_hook(hook) for mod in reg.modules()]
with torch.no_grad():
    o = reg.icl_block(torch.tensor(x_np).double(), torch.tensor(y_np).double())
for h in hs: h.remove()
f32mods = {k: v for k, v in seen.items() if "torch.float32" in v}
print("torch output dtype:", torch.as_tensor(getattr(o,'numerical',o)).dtype)
print("torch modules emitting float32:", f32mods or "NONE -- torch is fully float64")

with mx.stream(mx.cpu):
    w = {k: v.astype(mx.float64) for k, v in
         K.load_weights(f"{HERE}/export/kumo_large_reg.safetensors").items()}
    mx.eval(list(w.values()))
    om = K.ICLBlock(w, "icl_block")(mx.array(x_np).astype(mx.float64),
                                    mx.array(y_np).astype(mx.float64))
    mx.eval(om)
print("MLX weight dtypes:", {str(v.dtype) for v in w.values()}, " output dtype:", om.dtype)

# determinism of each side in f64
with mx.stream(mx.cpu):
    om2 = K.ICLBlock(w, "icl_block")(mx.array(x_np).astype(mx.float64),
                                     mx.array(y_np).astype(mx.float64)); mx.eval(om2)
print("MLX f64 run-to-run:", np.abs(np.array(om)-np.array(om2)).max())
torch.set_num_threads(8)
with torch.no_grad():
    o8 = reg.icl_block(torch.tensor(x_np).double(), torch.tensor(y_np).double())
a=torch.as_tensor(getattr(o,'numerical',o)).numpy(); b=torch.as_tensor(getattr(o8,'numerical',o8)).numpy()
print("torch f64 1-thread vs 8-thread:", np.abs(a-b).max())
