"""Kumo-Tabular with the recipe in Python and the network in MLX.

The split follows the cost: the recipe is per-column statistics and costs almost nothing, while
the 215.7 M-parameter network is the whole runtime. So `sdm` keeps doing the feature engineering
(rank-Gaussian, power transform, robust scaling, latin column shuffling, target standardisation
and its inverse) on CPU, and the network runs on the Apple GPU.

The seam is the inner model's forward: `reg(x, y, categorical_mask) -> quantiles`. Replacing just
that leaves every pre- and post-processing step, and the public API, untouched.

    from kumo_hybrid import KumoMLX
    model = KumoMLX(weights="export/kumo_large_reg.safetensors")
    pred = model.predict(x_context, y_context, x_query)   # numpy in, numpy out
"""

from __future__ import annotations

import sys

import numpy as np
import torch

import os

# Repository root; override with KUMOMLX_ROOT. Weights and goldens are expected under
# $KUMOMLX_ROOT/export (see the README -- they are not distributed with this repo).
HERE = os.environ.get("KUMOMLX_ROOT", os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(f"{HERE}/site"):
    sys.path.insert(0, f"{HERE}/site")
sys.path.insert(0, HERE)

import mlx.core as mx  # noqa: E402
import sdm.models as sdm_models  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from sdm.task import Task  # noqa: E402

import kumo_mlx as K  # noqa: E402


class KumoMLX:
    """Drop-in for `sdm.models.KumoTabular` whose network evaluates in MLX."""

    def __init__(self, weights: str, size: str = "large"):
        # The torch model is kept for its recipe and its API, not for its arithmetic: the inner
        # forward below never calls the torch network.
        self.torch_model = sdm_models.KumoTabular(
            task=Task.regression, size=size, pretrained=False, device="cpu").eval()
        self.torch_model.load_state_dict(load_file(weights), strict=False)

        w = K.load_weights(weights)
        mx.eval(list(w.values()))
        self.row_embedding = K.RowEmbedding(w)
        self.icl = K.ICLBlock(w, "icl_block")

        inner = self.torch_model.models[Task.regression]
        outer = self

        def mlx_forward(x, y, categorical_mask, *, cache=None):
            quantiles = outer.icl(
                outer.row_embedding(
                    mx.array(np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)),
                    mx.array(np.ascontiguousarray(y.detach().cpu().numpy(), dtype=np.float32)),
                    mx.array(np.ascontiguousarray(
                        categorical_mask.detach().cpu().numpy())).astype(mx.bool_),
                ),
                mx.array(np.ascontiguousarray(y.detach().cpu().numpy(), dtype=np.float32)),
            )
            mx.eval(quantiles)
            return torch.from_numpy(np.array(quantiles))

        inner.forward = mlx_forward

    def predict(self, x_context, y_context, x_query) -> np.ndarray:
        """Point predictions: the mean over the quantile grid, which is the Bayes estimator
        under squared error."""
        with torch.no_grad():
            out = self.torch_model(
                x_context=torch.as_tensor(np.asarray(x_context), dtype=torch.float32),
                y_context=torch.as_tensor(np.asarray(y_context), dtype=torch.float32).unsqueeze(-1),
                x_query=torch.as_tensor(np.asarray(x_query), dtype=torch.float32),
            )
        grid = torch.as_tensor(getattr(out, "numerical", out)).float().numpy()
        return grid.reshape(len(np.asarray(x_query)), -1).mean(axis=1)
