"""Kumo-Tabular with the preprocessing recipe in Python and the network in MLX.

The split follows the cost. The recipe is per-column statistics and costs almost nothing, while
the 215.7 M-parameter network is the whole runtime. So the upstream package keeps doing the
feature engineering (rank-Gaussian, power transform, robust scaling, latin column shuffling,
target standardisation or class shuffling, and their inverses) on CPU, and the network runs on
the Apple GPU.

The seam is the inner model's ``forward``: ``model(x, y, categorical_mask) -> outputs``.
Replacing only that leaves every pre- and post-processing step, and the public API, untouched.

    from kumomlx import KumoMLX, KumoMLXClassifier

    pred = KumoMLX().predict(x_context, y_context, x_query)                    # regression
    proba = KumoMLXClassifier().predict_proba(x_context, labels, x_query)      # classification
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kumomlx import network as K
from kumomlx import weights as W

__all__ = ["KumoMLX", "KumoMLXClassifier"]


class _KumoHybrid:
    """Builds the upstream wrapper for one task and swaps its inner forward for the MLX network."""

    task: str = ""

    def __init__(self, weights: str | Path | None = None, size: str = "large",
                 cache_limit_gb: float | None = 1.0):
        import mlx.core as mx                           # noqa: PLC0415
        import sdm.models as sdm_models                 # noqa: PLC0415
        import torch                                    # noqa: PLC0415
        from safetensors.torch import load_file         # noqa: PLC0415
        from sdm.task import Task                       # noqa: PLC0415

        task = Task.regression if self.task == "regression" else Task.classification
        path = Path(weights) if weights is not None else W.resolve(size=size, task=self.task)

        w = K.load_weights(path)
        mx.eval(list(w.values()))
        self._row_embedding = K.RowEmbedding(w)
        self._icl = K.ICLBlock(w, "icl_block")
        # A regressor checkpoint loads into the classifier's modules without a single error --
        # the shapes that matter are shared -- and then answers with quantiles read as logits.
        # The weights decide which head this is, so check them against the task asked for.
        if self._icl.label.is_categorical != (self.task == "classification"):
            kind = "classifier" if self._icl.label.is_categorical else "regressor"
            raise ValueError(f"{path} holds a {kind} checkpoint, not a {self.task} one")

        # The torch model is kept for its recipe and its API, not for its arithmetic: the inner
        # forward installed below never calls the torch network.
        self._torch = sdm_models.KumoTabular(
            task=task, size=size, pretrained=False, device="cpu").eval()
        sd = load_file(str(path))
        if not any(k.startswith("models.") for k in sd):
            sd = {f"models.{task.value}.{k}": v for k, v in sd.items()}
        self._torch.load_state_dict(sd, strict=False)

        inner = self._torch.models[task]
        outer, _mx, _torch = self, mx, torch
        # MLX keeps every freed buffer for reuse, and a forward pass frees buffers of many
        # different shapes, so the pool only grows: at 1,000 rows x 217 columns it reached
        # 17 GB around 4 GB of live arrays, and the process swapped. Bounding it cost nothing
        # (1.76 s vs 2.00 s). The limit is process-global in MLX, so it is applied around each
        # forward and the caller's setting restored afterwards.
        limit = None if cache_limit_gb is None else int(cache_limit_gb * 1e9)

        def one(x, y, categorical):
            return outer._icl(outer._row_embedding(x, y, categorical), y)

        def mlx_forward(x, y, categorical_mask, *, cache=None):
            if cache is not None:
                # Replay feeds query rows against a context cached on an earlier call; the port
                # recomputes the context every call and has nothing to replay from.
                raise NotImplementedError("the KV cache (fit/predict) is not ported; "
                                          "call the model on context and query together")

            def arr(t, dtype=np.float32):
                return _mx.array(np.ascontiguousarray(t.detach().cpu().numpy(), dtype=dtype))

            # Leading dimensions are estimators batched together, or ECOC tasks (more than ten
            # classes): [*lead, R, C] features with [*lead, R_train] labels. The network is
            # written for a single table, so run each one and restack.
            lead = tuple(x.shape[:-2])
            n_rows, n_cols = x.shape[-2:]
            xs = x.reshape(-1, n_rows, n_cols)
            ys = y.expand(*lead, y.shape[-1]).reshape(len(xs), -1)
            ms = categorical_mask.expand(*lead, n_cols).reshape(len(xs), n_cols)
            y_dtype = np.int64 if outer.task == "classification" else np.float32
            previous = None if limit is None else _mx.set_cache_limit(limit)
            try:
                outs = []
                for xi, yi, mi in zip(xs, ys, ms):
                    o = one(arr(xi), arr(yi, y_dtype), arr(mi, None).astype(_mx.bool_))
                    _mx.eval(o)                         # one table's graph at a time
                    outs.append(o)
                out = _mx.stack(outs).reshape(*lead, *outs[0].shape)
                result = _torch.from_numpy(np.array(out))
                del outs, out
            finally:
                if previous is not None:
                    _mx.set_cache_limit(previous)
            return result

        inner.forward = mlx_forward

    def _call(self, x_context, y_context, x_query, *, seed, num_estimators):
        import torch                                    # noqa: PLC0415

        gen = None if seed is None else torch.Generator().manual_seed(seed)
        with torch.no_grad():
            return self._torch(
                x_context=torch.as_tensor(np.asarray(x_context), dtype=torch.float32),
                y_context=y_context,
                x_query=torch.as_tensor(np.asarray(x_query), dtype=torch.float32),
                num_estimators=num_estimators,
                generator=gen)


class KumoMLX(_KumoHybrid):
    """Drop-in for ``sdm.models.KumoTabular`` (regression) whose network evaluates in MLX.

    Args:
        weights: path to MLX-ready safetensors. ``None`` resolves (and converts) the upstream
            checkpoint via :func:`kumomlx.weights.resolve`.
        size: only ``"large"`` is implemented by the port.
        cache_limit_gb: MLX buffer-cache bound during each forward; ``None`` leaves MLX's
            process-wide setting alone (unbounded by default, which swaps at ~1,000 rows).
    """

    task = "regression"

    def quantiles(self, x_context, y_context, x_query, *, seed: int | None = 0,
                  num_estimators: int | None = None) -> np.ndarray:
        """The full predictive distribution: ``[n_query, 999]`` quantiles per row.

        The recipe shuffles columns (a latin square) and rotates the per-column transform, and it
        draws that randomness fresh on every call. Unseeded, the SAME model on the SAME inputs
        moves by ~2e-02 between calls -- enough to swamp any comparison and to make results
        irreproducible. ``seed`` is therefore fixed by default; pass ``seed=None`` for the
        upstream stochastic behaviour, e.g. to average several draws.
        """
        import torch                                    # noqa: PLC0415

        y = torch.as_tensor(np.asarray(y_context), dtype=torch.float32).unsqueeze(-1)
        out = self._call(x_context, y, x_query, seed=seed, num_estimators=num_estimators)
        grid = torch.as_tensor(getattr(out, "numerical", out)).float().numpy()
        return grid.reshape(len(np.asarray(x_query)), -1)

    def predict(self, x_context, y_context, x_query, *, seed: int | None = 0,
                num_estimators: int | None = None) -> np.ndarray:
        """Point predictions: the mean over the quantile grid, the Bayes estimator under
        squared error."""
        return self.quantiles(x_context, y_context, x_query, seed=seed,
                              num_estimators=num_estimators).mean(axis=1)

    def predict_interval(self, x_context, y_context, x_query, level: float = 0.9,
                         *, seed: int | None = 0, num_estimators: int | None = None):
        """Central prediction interval at ``level`` coverage, read off the quantile grid."""
        q = self.quantiles(x_context, y_context, x_query, seed=seed,
                           num_estimators=num_estimators)
        lo = int(round((1 - level) / 2 * (q.shape[1] - 1)))
        hi = q.shape[1] - 1 - lo
        qs = np.sort(q, axis=1)
        return qs[:, lo], qs[:, hi]


class KumoMLXClassifier(_KumoHybrid):
    """Drop-in for ``sdm.models.KumoTabular`` (classification) whose network evaluates in MLX.

    Labels may be any sortable values (ints, strings, bools). Up to ten classes are scored
    natively; more go through the upstream ECOC wrapper, which this port supports.

    Args:
        weights: path to MLX-ready classifier safetensors. ``None`` resolves (and converts)
            ``large/classifier.pt`` via :func:`kumomlx.weights.resolve`.
        size: only ``"large"`` is implemented by the port.
        cache_limit_gb: as for :class:`KumoMLX`.
    """

    task = "classification"

    def predict_proba(self, x_context, y_context, x_query, *, seed: int | None = 0,
                      num_estimators: int | None = None) -> np.ndarray:
        """Class probabilities, ``[n_query, n_classes]``, columns in the order of ``classes_``.

        ``classes_`` is set from the labels of THIS call's context: a class absent from the
        context cannot be predicted, and gets no column. The recipe also shifts the class codes
        at random per estimator, so ``seed`` is fixed by default for the same reason as in
        :meth:`KumoMLX.quantiles`.
        """
        import torch                                    # noqa: PLC0415
        from sdm.stype import Stype                     # noqa: PLC0415

        classes, codes = np.unique(np.asarray(y_context), return_inverse=True)
        if len(classes) < 2:
            raise ValueError("the context holds a single class; there is nothing to separate")
        self.classes_ = classes
        y = torch.as_tensor(codes.reshape(-1), dtype=torch.int64).unsqueeze(-1)
        out = self._call(x_context, y, x_query, seed=seed, num_estimators=num_estimators)
        proba = torch.as_tensor(out.numerical).float().numpy().reshape(
            len(np.asarray(x_query)), -1)
        # Output columns are named by class CODE as a string; order them 0..K-1 so they line up
        # with classes_, rather than trusting the order the recipe happened to emit.
        order = np.argsort([int(c) for c in out.columns[Stype.numerical]])
        return proba[:, order]

    def predict(self, x_context, y_context, x_query, *, seed: int | None = 0,
                num_estimators: int | None = None) -> np.ndarray:
        """Most probable class per query row, as the original label values."""
        proba = self.predict_proba(x_context, y_context, x_query, seed=seed,
                                   num_estimators=num_estimators)
        return self.classes_[proba.argmax(axis=1)]
