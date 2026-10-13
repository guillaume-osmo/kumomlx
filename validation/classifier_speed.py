"""Time the classifier alone, one backend per process, so neither pays for the other's memory.

`classifier_check.py` holds three models at once and is a correctness script; its timings are
not to be quoted. This one loads exactly one network and times it at two context sizes.

    python validation/classifier_speed.py mlx
    python validation/classifier_speed.py torch
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "src"))

from kumomlx import load_boiling_point  # noqa: E402
from kumomlx import weights as W  # noqa: E402

backend = sys.argv[1]
path = W.cache_dir() / "kumo_large_classification.safetensors"
data = load_boiling_point()
(xc_all, bc_all), (xq_all, _), _ = data.split(n_context=800, seed=0)
yc_all = (bc_all > np.median(bc_all)).astype(int)

if backend == "mlx":
    from kumomlx import KumoMLXClassifier

    clf = KumoMLXClassifier(weights=path)

    def run(xc, yc, xq):
        return clf.predict_proba(xc, yc, xq, seed=0)
else:
    import sdm.models as sdm_models
    from safetensors.torch import load_file
    from sdm.task import Task

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))
    tm = sdm_models.KumoTabular(task=Task.classification, size="large",
                                pretrained=False, device="cpu").eval()
    tm.load_state_dict({f"models.classification.{k}": v
                        for k, v in load_file(str(path)).items()}, strict=False)

    def run(xc, yc, xq):
        with torch.no_grad():
            return tm(x_context=torch.tensor(xc), y_context=torch.tensor(yc).long().unsqueeze(-1),
                      x_query=torch.tensor(xq), generator=torch.Generator().manual_seed(0))

run(xc_all[:50], yc_all[:50], xq_all[:5])                       # warm-up
for n_ctx, n_q in ((256, 64), (400, 100), (800, 200)):
    ts = []
    for _ in range(2):
        t0 = time.perf_counter()
        run(xc_all[:n_ctx], yc_all[:n_ctx], xq_all[:n_q])
        ts.append(time.perf_counter() - t0)
    print(f"{backend:5s} context {n_ctx:4d} + query {n_q:3d} x 217 cols: "
          f"{min(ts):6.2f} s (best of 2)", flush=True)
