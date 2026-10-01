"""Predict boiling points for 200 molecules from 800 examples, with no training.

Kumo-Tabular is an in-context model: the context rows ARE the fit. There is no optimiser, no
epochs and no saved model per dataset -- the 800 labelled molecules are fed in alongside the 200
unlabelled ones, and the forward pass returns a predictive distribution for each query.

Run:
    python examples/boiling_point.py
"""

from __future__ import annotations

import time

import numpy as np

from kumomlx import KumoMLX, load_boiling_point


def ridge_baseline(xc, yc, xq, alpha: float = 1.0) -> np.ndarray:
    """Closed-form ridge on standardised features -- a floor, not a competitor.

    Included because an impressive-looking error means nothing without something to compare it
    to: boiling point is largely driven by molecular size, so a linear model already does well.
    """
    mu, sd = xc.mean(0), xc.std(0) + 1e-9
    a = np.c_[(xc - mu) / sd, np.ones(len(xc))]
    b = np.c_[(xq - mu) / sd, np.ones(len(xq))]
    reg = alpha * np.eye(a.shape[1])
    reg[-1, -1] = 0.0                      # never penalise the intercept
    w = np.linalg.solve(a.T @ a + reg, a.T @ (yc - yc.mean()))
    return b @ w + yc.mean()


def metrics(pred, true) -> dict[str, float]:
    err = pred - true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": 1.0 - ss_res / ss_tot,
        "Spearman": float(np.corrcoef(rank(pred), rank(true))[0, 1]),
    }


def main() -> None:
    data = load_boiling_point()
    (xc, yc), (xq, yq), _ = data.split(n_context=800, seed=0)
    print(f"{len(data)} molecules, {len(data.feature_names)} RDKit descriptors")
    print(f"boiling point {data.y.min():.0f} to {data.y.max():.0f} C "
          f"(sd {data.y.std():.1f})")
    print(f"context {len(yc)} / query {len(yq)}\n")

    model = KumoMLX()
    t0 = time.perf_counter()
    pred = model.predict(xc, yc, xq)
    dt = time.perf_counter() - t0

    base = ridge_baseline(xc, yc, xq)
    mk, mb = metrics(pred, yq), metrics(base, yq)
    print(f"{'':10s} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'Spearman':>9}")
    for name, m in (("Kumo", mk), ("ridge", mb)):
        print(f"{name:10s} {m['MAE']:8.2f} {m['RMSE']:8.2f} {m['R2']:8.4f} {m['Spearman']:9.4f}")
    print(f"\nin-context prediction of {len(yq)} molecules: {dt:.2f} s "
          f"({dt / len(yq) * 1000:.1f} ms each), no training")

    # The model returns 999 quantiles, so an interval costs nothing extra.
    lo, hi = model.predict_interval(xc, yc, xq, level=0.9)
    cov = float(((yq >= lo) & (yq <= hi)).mean())
    print(f"90% interval covers {cov:.1%} of held-out molecules "
          f"(median width {np.median(hi - lo):.0f} C)")

    worst = np.argsort(-np.abs(pred - yq))[:3]
    print("\nlargest errors:")
    for i in worst:
        print(f"  {data.smiles[_[1][i]]:<28s} true {yq[i]:7.1f}  pred {pred[i]:7.1f}")


if __name__ == "__main__":
    main()
