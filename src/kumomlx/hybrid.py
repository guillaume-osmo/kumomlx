"""Kumo-Tabular with the preprocessing recipe in Python and the network in MLX.

The split follows the cost. The recipe is per-column statistics and costs almost nothing, while
the 215.7 M-parameter network is the whole runtime. So the upstream package keeps doing the
feature engineering (rank-Gaussian, power transform, robust scaling, latin column shuffling,
target standardisation and its inverse) on CPU, and the network runs on the Apple GPU.

The seam is the inner model's ``forward``: ``reg(x, y, categorical_mask) -> quantiles``.
Replacing only that leaves every pre- and post-processing step, and the public API, untouched.

    from kumomlx import KumoMLX

    model = KumoMLX()                                   # weights resolved + cached on first use
    pred = model.predict(x_context, y_context, x_query)  # numpy in, numpy out
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kumomlx import network as K
from kumomlx import weights as W

__all__ = ["KumoMLX"]


class KumoMLX:
    """Drop-in for ``sdm.models.KumoTabular`` whose network evaluates in MLX.

    Args:
        weights: path to MLX-ready safetensors. ``None`` resolves (and converts) the upstream
            checkpoint via :func:`kumomlx.weights.resolve`.
        size: only ``"large"`` is implemented by the port.
    """

    def __init__(self, weights: str | Path | None = None, size: str = "large"):
        import mlx.core as mx                           # noqa: PLC0415
        import sdm.models as sdm_models                 # noqa: PLC0415
        import torch                                    # noqa: PLC0415
        from safetensors.torch import load_file         # noqa: PLC0415
        from sdm.task import Task                       # noqa: PLC0415

        path = Path(weights) if weights is not None else W.resolve(size=size)

        # The torch model is kept for its recipe and its API, not for its arithmetic: the inner
        # forward installed below never calls the torch network.
        self._torch = sdm_models.KumoTabular(
            task=Task.regression, size=size, pretrained=False, device="cpu").eval()
        sd = load_file(str(path))
        if not any(k.startswith("models.") for k in sd):
            sd = {f"models.regression.{k}": v for k, v in sd.items()}
        self._torch.load_state_dict(sd, strict=False)

        w = K.load_weights(path)
        mx.eval(list(w.values()))
        self._row_embedding = K.RowEmbedding(w)
        self._icl = K.ICLBlock(w, "icl_block")

        inner = self._torch.models[Task.regression]
        outer, _mx, _torch = self, mx, torch

        def mlx_forward(x, y, categorical_mask, *, cache=None):
            def arr(t, dtype=np.float32):
                return _mx.array(np.ascontiguousarray(t.detach().cpu().numpy(), dtype=dtype))
            quantiles = outer._icl(
                outer._row_embedding(arr(x), arr(y),
                                     arr(categorical_mask, None).astype(_mx.bool_)),
                arr(y))
            _mx.eval(quantiles)
            return _torch.from_numpy(np.array(quantiles))

        inner.forward = mlx_forward

    def quantiles(self, x_context, y_context, x_query) -> np.ndarray:
        """The full predictive distribution: ``[n_query, 999]`` quantiles per row."""
        import torch                                    # noqa: PLC0415

        with torch.no_grad():
            out = self._torch(
                x_context=torch.as_tensor(np.asarray(x_context), dtype=torch.float32),
                y_context=torch.as_tensor(np.asarray(y_context),
                                          dtype=torch.float32).unsqueeze(-1),
                x_query=torch.as_tensor(np.asarray(x_query), dtype=torch.float32))
        grid = torch.as_tensor(getattr(out, "numerical", out)).float().numpy()
        return grid.reshape(len(np.asarray(x_query)), -1)

    def predict(self, x_context, y_context, x_query) -> np.ndarray:
        """Point predictions: the mean over the quantile grid, the Bayes estimator under
        squared error."""
        return self.quantiles(x_context, y_context, x_query).mean(axis=1)

    def predict_interval(self, x_context, y_context, x_query, level: float = 0.9):
        """Central prediction interval at ``level`` coverage, read off the quantile grid."""
        q = self.quantiles(x_context, y_context, x_query)
        lo = int(round((1 - level) / 2 * (q.shape[1] - 1)))
        hi = q.shape[1] - 1 - lo
        qs = np.sort(q, axis=1)
        return qs[:, lo], qs[:, hi]
