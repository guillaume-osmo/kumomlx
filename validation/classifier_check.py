"""The classification head against pure torch, before any real classifier weights are involved.

The classifier is the regressor's network with two swaps: a class lookup (`y_emb`) where the
regressor maps the label value linearly (`y_lin`), and a 10-logit head instead of 999 quantiles.
So parity can be established on a RANDOMLY initialised classifier: every weight the port reads
is exercised, and no checkpoint has to be downloaded. A random network is a stricter test than a
trained one here, not a weaker one -- its logits are not pulled towards anything, so a wrong
label path cannot hide behind a confident, mostly-right prediction.

Checks, all with the recipe's generator SEEDED (unseeded, torch disagrees with itself):

  1. inner seam    torch inner forward vs MLX network on the SAME recipe-processed tensors,
                   compared on raw logits, before softmax can compress a difference;
  2. end to end    probabilities through the shipped hybrid, binary and three-class;
  3. ECOC          twelve classes, which the upstream wrapper splits into 10-class tasks and
                   feeds to the network as a leading task dimension;
  4. guard         a regressor checkpoint handed to the classifier must be refused;
  5. regressor     the shared code path still reproduces torch with the real regressor weights.

    python validation/classifier_check.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "src"))

import mlx.core as mx  # noqa: E402
import sdm.models as sdm_models  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from sdm.stype import Stype  # noqa: E402
from sdm.task import Task  # noqa: E402

from kumomlx import network as K  # noqa: E402
from kumomlx import weights as W  # noqa: E402
from kumomlx.hybrid import KumoMLX, KumoMLXClassifier  # noqa: E402

torch.set_num_threads(1)
failures = []


def check(name, err, bar):
    ok = err <= bar
    print(f"  {'PASS' if ok else 'FAIL'}  {name:46s} {err:.3e}  (bar {bar:.0e})")
    if not ok:
        failures.append(name)


def table(n_ctx, n_q, p, n_classes, seed):
    g = np.random.default_rng(seed)
    x = g.normal(size=(n_ctx + n_q, p)).astype(np.float32)
    score = x[:, :3] @ g.normal(size=3)
    y = np.digitize(score, np.quantile(score, np.linspace(0, 1, n_classes + 1)[1:-1]))
    # NaN only AFTER the labels exist: a NaN score turns every quantile bin into NaN and
    # silently collapses the labels to one or two classes.
    x[3, 1] = np.nan                                  # exercise context-only imputation
    assert len(np.unique(y[:n_ctx])) == n_classes, "every class must reach the context"
    return x[:n_ctx], y[:n_ctx], x[n_ctx:], y[n_ctx:]


# ----------------------------------------------------------------- a random classifier
torch.manual_seed(0)
ref = sdm_models.KumoTabular(task=Task.classification, size="large",
                             pretrained=False, device="cpu").eval()
inner = ref.models[Task.classification]
tmp = tempfile.mkdtemp(prefix="kumomlx_cls_")
path = os.path.join(tmp, "random_classifier.safetensors")
# clone(): the RoPE inv_freq buffers alias one another and safetensors rejects shared storage
save_file({k: v.detach().clone().contiguous() for k, v in inner.state_dict().items()}, path)
print(f"random classifier: {sum(v.numel() for v in inner.state_dict().values()):,} values -> {path}")


def torch_proba(xc, yc, xq, seed):
    gen = None if seed is None else torch.Generator().manual_seed(seed)
    with torch.no_grad():
        out = ref(x_context=torch.tensor(xc), y_context=torch.tensor(yc).long().unsqueeze(-1),
                  x_query=torch.tensor(xq), generator=gen)
    order = np.argsort([int(c) for c in out.columns[Stype.numerical]])
    return out.numerical.float().numpy()[:, order]


# 1. inner seam: capture what the recipe hands the network, replay it on both sides
print("\n1. inner seam (raw logits)")
xc, yc, xq, _ = table(200, 64, 8, 2, seed=1)
taps = []
orig = inner.forward


def spy(x, y, categorical_mask, *, cache=None):
    taps.append((x.clone(), y.clone(), categorical_mask.clone()))
    return orig(x, y, categorical_mask, cache=cache)


inner.forward = spy
torch_proba(xc, yc, xq, seed=0)
inner.forward = orig
x_t, y_t, m_t = taps[0]
with torch.no_grad():
    logits_t = orig(x_t.clone(), y_t, m_t).numpy()   # clone: the ICL block writes into x
w = K.load_weights(path)
logits_m = np.array(K.ICLBlock(w, "icl_block")(
    K.RowEmbedding(w)(mx.array(x_t.numpy()), mx.array(y_t.numpy()),
                      mx.array(m_t.numpy()).astype(mx.bool_)),
    mx.array(y_t.numpy())))
print(f"  recipe output: x {tuple(x_t.shape)} {x_t.dtype}, y {tuple(y_t.shape)} {y_t.dtype}"
      f" codes {sorted(set(y_t.tolist()))}; logits {logits_t.shape}, sd {logits_t.std():.3f}")
check("logits, MLX vs torch (max abs)", np.abs(logits_m - logits_t).max(), 1e-3)
check("logits, relative to their spread", np.abs(logits_m - logits_t).max() / logits_t.std(), 1e-4)
blind = K.LabelEmbedding({"p.y_emb.weight": w["icl_block.y_emb.weight"]}, "p")
check("label lookup is a row gather", float(np.abs(
    np.array(blind(mx.array(y_t.numpy()))) - np.array(w["icl_block.y_emb.weight"])[y_t.numpy()]
).max()), 0.0)

# 2. end to end through the hybrid
print("\n2. end to end (probabilities)")
clf = KumoMLXClassifier(weights=path)
for k in (2, 3):
    xc, yc, xq, _ = table(200, 64, 8, k, seed=k)
    p_t = torch_proba(xc, yc, xq, seed=0)
    p_m = clf.predict_proba(xc, yc, xq, seed=0)
    assert p_m.shape == (64, k), p_m.shape
    check(f"{k} classes, MLX vs torch, seeded", np.abs(p_m - p_t).max(), 1e-4)
    check(f"{k} classes, rows sum to one", np.abs(p_m.sum(1) - 1).max(), 1e-5)
    noise = np.abs(torch_proba(xc, yc, xq, None) - torch_proba(xc, yc, xq, None)).max()
    print(f"        torch vs torch UNSEEDED (the recipe's own randomness): {noise:.3e}")
labels = np.array(["ready", "not_ready"])[yc % 2]
p_lab = clf.predict_proba(xc, labels, xq, seed=0)
check("string labels give the same columns", np.abs(
    p_lab - clf.predict_proba(xc, (labels == "ready").astype(int), xq, seed=0)).max(), 0.0)
assert list(clf.classes_) == [0, 1]

# 3. ECOC: more than ten classes become a leading task dimension
print("\n3. ECOC (12 classes)")
xc, yc, xq, _ = table(240, 48, 8, 12, seed=12)
p_t = torch_proba(xc, yc, xq, seed=0)
p_m = clf.predict_proba(xc, yc, xq, seed=0)
assert p_m.shape == (48, 12), p_m.shape
check("12 classes, MLX vs torch, seeded", np.abs(p_m - p_t).max(), 1e-4)

# 4. the guard
print("\n4. a regressor checkpoint is refused by the classifier")
reg_path = W.resolve(local_files_only=True)
try:
    KumoMLXClassifier(weights=reg_path)
    failures.append("regressor weights accepted by the classifier")
    print("  FAIL  regressor weights were accepted")
except ValueError as e:
    print(f"  PASS  {e}")

# 5. the regressor through the shared, refactored path, with its real weights
print("\n5. regressor unchanged (real weights)")
treg = sdm_models.KumoTabular(task=Task.regression, size="large",
                              pretrained=False, device="cpu").eval()
from safetensors.torch import load_file  # noqa: E402

sd = load_file(str(reg_path))
treg.load_state_dict({f"models.regression.{k}": v for k, v in sd.items()}
                     if not any(k.startswith("models.") for k in sd) else sd, strict=False)
g = np.random.default_rng(7)
xr = g.normal(size=(320, 12)).astype(np.float32)
yr = (xr[:, :3] @ g.normal(size=3) + 0.1 * g.normal(size=320)).astype(np.float32)
with torch.no_grad():
    out = treg(x_context=torch.tensor(xr[:256]), y_context=torch.tensor(yr[:256]).unsqueeze(-1),
               x_query=torch.tensor(xr[256:]), generator=torch.Generator().manual_seed(0))
ref_pred = out.numerical.float().numpy().reshape(64, -1).mean(1)
got = KumoMLX(weights=reg_path).predict(xr[:256], yr[:256], xr[256:], seed=0)
check("regressor, MLX vs torch, seeded", np.abs(got - ref_pred).max(), 1e-4)

print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + ', '.join(failures)}")
sys.exit(1 if failures else 0)
