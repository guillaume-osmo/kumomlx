"""Where the MLX network's memory goes, by stage and by context size.

MLX evaluates lazily: unless something forces evaluation, a whole forward pass is one graph and
every intermediate that graph still references stays allocated. This script reports MLX's own
peak allocation (not process RSS, which also counts the torch recipe and the buffer cache) for
the row embedding and the ICL stack separately.

    python validation/memory_profile.py
"""
import os
import sys
import time

import numpy as np

HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, "src"))

import mlx.core as mx  # noqa: E402

from kumomlx import network as K  # noqa: E402
from kumomlx import weights as W  # noqa: E402

w = K.load_weights(W.cache_dir() / "kumo_large_classification.safetensors")
mx.eval(list(w.values()))
row, icl = K.RowEmbedding(w), K.ICLBlock(w, "icl_block")
base = mx.get_active_memory()
print(f"weights resident: {base / 1e9:.2f} GB")

g = np.random.default_rng(0)
for n_ctx, n_q, n_col in ((256, 64, 217), (800, 200, 217), (1600, 400, 217), (800, 200, 64)):
    x = mx.array(g.normal(size=(n_ctx + n_q, n_col)).astype(np.float32))
    y = mx.array((g.random(n_ctx) > 0.5).astype(np.int64))
    cat = mx.zeros((n_col,), dtype=mx.bool_)

    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    h = row(x, y, cat)
    mx.eval(h)
    t_row, p_row = time.perf_counter() - t0, mx.get_peak_memory() - base

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    out = icl(h, y)
    mx.eval(out)
    t_icl, p_icl = time.perf_counter() - t0, mx.get_peak_memory() - base
    print(f"rows {n_ctx + n_q:5d} x cols {n_col:3d}:  row embedding {p_row / 1e9:6.2f} GB "
          f"{t_row:5.2f} s   ICL {p_icl / 1e9:5.2f} GB {t_icl:5.2f} s", flush=True)
    del h, out
