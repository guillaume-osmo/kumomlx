"""Classify molecules by boiling point from 800 labelled examples, with no training.

The same bundled molecules as `boiling_point.py`, turned into two classification tasks so the
example needs no extra dataset: does a molecule boil above the context median (binary), and in
which tercile does it fall (three classes). The context rows are the fit, exactly as in
regression; only the checkpoint and the label type change.

Run:
    python examples/boiling_point_classes.py
"""

from __future__ import annotations

import time

import numpy as np

from kumomlx import KumoMLXClassifier, load_boiling_point


def ridge_floor(xc, yc, xq, n_classes: int, alpha: float = 1.0) -> np.ndarray:
    """One-vs-rest closed-form ridge on standardised features: a floor, not a competitor."""
    mu, sd = np.nanmean(xc, 0), np.nanstd(xc, 0) + 1e-9
    fill = lambda v: np.where(np.isnan(v), 0.0, (v - mu) / sd)  # noqa: E731
    a, b = np.c_[fill(xc), np.ones(len(xc))], np.c_[fill(xq), np.ones(len(xq))]
    reg = alpha * np.eye(a.shape[1])
    reg[-1, -1] = 0.0
    onehot = np.eye(n_classes)[yc]
    return b @ np.linalg.solve(a.T @ a + reg, a.T @ onehot)


def auc(score, positive) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):                         # average ranks over ties
        tie = score == v
        ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main() -> None:
    data = load_boiling_point()
    (xc, bc), (xq, bq), _ = data.split(n_context=800, seed=0)
    clf = KumoMLXClassifier()

    for name, edges in (("above the median", [50]), ("tercile", [100 / 3, 200 / 3])):
        cut = np.percentile(bc, edges)                 # cut points from the CONTEXT only
        yc, yq = np.digitize(bc, cut), np.digitize(bq, cut)
        k = len(edges) + 1

        t0 = time.perf_counter()
        proba = clf.predict_proba(xc, yc, xq)
        dt = time.perf_counter() - t0
        floor = ridge_floor(xc, yc, xq, k)

        print(f"\n{name}: {k} classes, context {len(yc)} / query {len(yq)}")
        print(f"  {'':8s} {'accuracy':>9} {'AUC':>7}")
        for label, s in (("Kumo", proba), ("ridge", floor)):
            acc = float((s.argmax(1) == yq).mean())
            # one-vs-rest AUC, averaged over classes
            a = np.mean([auc(s[:, c], yq == c) for c in range(k)])
            print(f"  {label:8s} {acc:9.3f} {a:7.3f}")
        print(f"  in-context prediction: {dt:.2f} s, no training")


if __name__ == "__main__":
    main()
