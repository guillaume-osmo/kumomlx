"""Capture reference tensors at the INNER model boundary, i.e. after the recipe has run.

The recipe stays in Python; only the network is ported. So the contract the MLX side must satisfy
is not "raw dataframe in, prediction out" but "recipe-processed tensors in, quantiles out". This
script records exactly that boundary.

Every tap is CLONED. The ICL block mutates its input in place, so an un-cloned hook records the
post-mutation tensor, and the symptom is that torch cannot reproduce its own output from its own
tap. That cost an hour the first time.
"""

import sys

import numpy as np
import torch
from safetensors.torch import load_file

import os

# Repository root; override with KUMOMLX_ROOT. Weights and goldens are expected under
# $KUMOMLX_ROOT/export (see the README -- they are not distributed with this repo).
HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{HERE}/site")

import sdm.models as M  # noqa: E402
from sdm.task import Task  # noqa: E402

torch.manual_seed(0)
model = M.KumoTabular(task=Task.regression, size="large", pretrained=False, device="cpu").eval()
missing, unexpected = model.load_state_dict(
    load_file(f"{HERE}/export/kumo_large_reg.safetensors"), strict=False)
print(f"weights loaded (missing {len(missing)}, unexpected {len(unexpected)})")

reg = model.models[Task.regression]
taps = {}
handles = []


def clone(t):
    return torch.as_tensor(getattr(t, "numerical", t)).detach().float().clone().numpy()


def pre_tap(name, module):
    def hook(_m, args, kwargs):
        # icl_block is called with KEYWORDS, so args is empty there; scan both.
        for i, a in enumerate(args):
            if torch.is_tensor(a) or hasattr(a, "numerical"):
                taps[f"{name}_in{i}"] = clone(a)
        for k, a in kwargs.items():
            if torch.is_tensor(a) or hasattr(a, "numerical"):
                taps[f"{name}_in_{k}"] = clone(a)
    handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))


def post_tap(name, module):
    def hook(_m, _i, out):
        taps[name] = clone(out[0] if isinstance(out, tuple) else out)
    handles.append(module.register_forward_hook(hook))


pre_tap("inner", reg)
pre_tap("icl", reg.icl_block)
post_tap("row_embedding", reg.row_embedding)
post_tap("cell_embedding", reg.row_embedding.cell_embedding)
post_tap("icl_block", reg.icl_block)

g = torch.Generator().manual_seed(7)
n_ctx, n_q, p = 256, 64, 12
X = torch.randn(n_ctx + n_q, p, generator=g)
y = X[:, :3] @ torch.randn(3, generator=g) + 0.1 * torch.randn(n_ctx + n_q, generator=g)
X[5, 2] = float("nan")

with torch.no_grad():
    final = model(x_context=X[:n_ctx], y_context=y[:n_ctx].unsqueeze(-1), x_query=X[n_ctx:])
taps["final"] = clone(final)
for h in handles:
    h.remove()

np.savez(f"{HERE}/export/local_taps.npz",
         x_raw=X.numpy(), y_raw=y.numpy(), n_ctx=np.array(n_ctx), **taps)
for k, v in taps.items():
    print(f"  {k:18s} {v.shape}")

# The check that failed before: can torch reproduce its own ICL output from its own tap?
icl_in = {k: v for k, v in taps.items() if k.startswith("icl_in")}
print("\nicl inputs captured:", {k: v.shape for k, v in icl_in.items()})
