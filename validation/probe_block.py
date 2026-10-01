"""One block, float64, component by component. The float64 floor is ~1e-15, so the first
component above it is the fault -- no judgement call required."""
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

torch.set_num_threads(1); mx.set_default_device(mx.cpu); D = torch.float64
m = SM.KumoTabular(task=Task.regression, size="large", pretrained=False, device="cpu").eval().to(D)
m.load_state_dict({k: v.to(D) for k, v in
                   load_file(f"{HERE}/export/kumo_large_reg.safetensors").items()}, strict=False)
reg = m.models[Task.regression]; layer = reg.icl_block.layers[0]
w = {k: v.astype(mx.float64) for k, v in
     K.load_weights(f"{HERE}/export/kumo_large_reg.safetensors").items()}
mx.eval(list(w.values()))
blk = K.Block(w, "icl_block.layers.0", 16, 64)

t = np.load(f"{HERE}/export/local_taps.npz")
x = torch.tensor(t["icl_in_x"]).to(D); y = torch.tensor(t["icl_in_y"]).to(D); n = y.shape[0]
with torch.no_grad():
    y_emb = reg.icl_block.y_lin(y.unsqueeze(-1))
    x0 = torch.cat([x[:n] + y_emb, x[n:]], 0)
x0m = mx.array(x0.numpy())

def cmp(name, a, b):
    a = a.detach().numpy() if torch.is_tensor(a) else a
    b = np.array(b)
    if a.shape != b.shape: print(f"  {name:26s} SHAPE {a.shape} vs {b.shape}"); return
    e = np.abs(a-b).max()
    print(f"  {name:26s} {e:.3e}  {'<-- ABOVE FLOOR' if e > 1e-12 else ''}")

print("float64, layer 0, context pass (query = key_value = context):")
ctx, ctxm = x0[:n], x0m[:n]
with torch.no_grad():
    qn = layer.query_norm(ctx); kn = layer.key_value_norm(ctx)
cmp("query_norm", qn, K._rms_norm(ctxm, blk.qn))
cmp("key_value_norm", kn, K._rms_norm(ctxm, blk.kn))

a = layer.attn
with torch.no_grad():
    qkv = a.query_key_value_projection if hasattr(a, "query_key_value_projection") else None
print(f"  [attn module: {type(a).__name__}, children {[c for c,_ in a.named_children()]}]")
with torch.no_grad():
    at = a(query=qn, key_value=kn); at = at[0] if isinstance(at, tuple) else at
cmp("attn", at, blk.attn(K._rms_norm(ctxm, blk.qn), K._rms_norm(ctxm, blk.kn), kv_heads=None))

with torch.no_grad():
    h = ctx + at
    mo = layer.mlp(h)
hm = ctxm + mx.array(at.numpy())
mlp_m = K._linear(K._gelu(K._linear(K._rms_norm(hm, blk.mlp_norm), blk.mlp_w1, blk.mlp_b1)),
                  blk.mlp_w2, blk.mlp_b2)
cmp("mlp", mo, mlp_m)
cmp("mlp_norm only", layer.mlp[0](h), K._rms_norm(hm, blk.mlp_norm))
cmp("gelu(lin1(norm))", layer.mlp[2](layer.mlp[1](layer.mlp[0](h))),
    K._gelu(K._linear(K._rms_norm(hm, blk.mlp_norm), blk.mlp_w1, blk.mlp_b1)))

with torch.no_grad():
    bt = layer(query=ctx.clone(), key_value=ctx.clone()); bt = bt[0] if isinstance(bt, tuple) else bt
cmp("full block (context)", bt, blk(ctxm, ctxm, kv_heads=None))
