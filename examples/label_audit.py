"""Find suspect boiling points by asking the model how surprised it is.

Kumo returns 999 quantiles per query, so for each molecule we can compute the PIT (probability
integral transform): the fraction of the predictive distribution lying below the recorded value.
For a calibrated model and a correct label, PIT is uniform on [0, 1]. A label error shows up as a
PIT pinned near 0 or 1 -- the recorded value sits in the tail of what the other 800 molecules
imply.

Each molecule is held out in many random splits, so the verdict does not depend on one draw, and
the recipe is re-seeded per draw so its own randomness is averaged over rather than mistaken for
uncertainty.

Multiplicity is handled explicitly: 1000 molecules tested at a 1% tail would throw ~10 false
flags by chance, so the two-sided PIT p-values go through Benjamini-Hochberg at FDR 5%.

    python examples/label_audit.py --draws 50

A flag is a REVIEW QUEUE, not a verdict: the model is imperfect, so chemically unusual molecules
flag alongside genuinely bad labels. The output separates the two where the data can.
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from kumomlx import KumoMLX, load_boiling_point


def collect_pit(model, data, draws: int, n_context: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean PIT per molecule and how many times each was held out."""
    n = len(data)
    tot = np.zeros(n)
    cnt = np.zeros(n, dtype=int)
    for d in range(draws):
        idx = np.random.default_rng(1000 + d).permutation(n)
        c, q = idx[:n_context], idx[n_context:]
        grid = model.quantiles(data.x[c], data.y[c], data.x[q], seed=d)
        grid = np.sort(grid, axis=1)
        # PIT: where the recorded value falls inside the predicted distribution
        pit = (grid < data.y[q][:, None]).mean(axis=1)
        tot[q] += pit
        cnt[q] += 1
        print(f"  draw {d + 1}/{draws}", end="\r", flush=True)
    print(" " * 30, end="\r")
    return tot / np.maximum(cnt, 1), cnt


def benjamini_hochberg(p: np.ndarray, fdr: float) -> np.ndarray:
    """Indices rejected at the given false-discovery rate."""
    order = np.argsort(p)
    m = len(p)
    thresh = fdr * (np.arange(1, m + 1) / m)
    passing = p[order] <= thresh
    if not passing.any():
        return np.array([], dtype=int)
    k = np.max(np.nonzero(passing)[0])
    return order[:k + 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--n-context", type=int, default=800)
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "..", "label_audit.csv"))
    args = ap.parse_args()

    data = load_boiling_point()
    model = KumoMLX()
    print(f"{len(data)} molecules, {args.draws} random splits of {args.n_context} context rows")
    pit, cnt = collect_pit(model, data, args.draws, args.n_context)
    print(f"each molecule held out {cnt.min()}-{cnt.max()} times (mean {cnt.mean():.1f})\n")

    # two-sided tail probability, floored at the grid resolution (999 quantiles)
    p = np.maximum(2 * np.minimum(pit, 1 - pit), 1 / 999)
    flagged = benjamini_hochberg(p, args.fdr)
    order = flagged[np.argsort(p[flagged])]

    # A point prediction for context, from one clean split per flagged molecule.
    print(f"PIT uniformity: mean {pit.mean():.3f} (0.5 if calibrated), "
          f"frac in [0.05,0.95] {np.mean((pit > .05) & (pit < .95)):.3f} (0.90 if calibrated)")
    print(f"flagged at FDR {args.fdr:.0%}: {len(flagged)} of {len(data)}\n")

    rng = np.random.default_rng(7)
    idx = rng.permutation(len(data))
    ctx = np.setdiff1d(idx, order)[:args.n_context]
    pred = model.predict(data.x[ctx], data.y[ctx], data.x[order], seed=0)

    rows = []
    print(f"{'PIT':>6} {'recorded':>9} {'predicted':>10} {'gap':>8}  smiles")
    for rank, i in enumerate(order):
        gap = data.y[i] - pred[rank]
        rows.append({"smiles": data.smiles[i], "recorded_bp": float(data.y[i]),
                     "predicted_bp": round(float(pred[rank]), 1), "gap": round(float(gap), 1),
                     "mean_pit": round(float(pit[i]), 4), "times_held_out": int(cnt[i])})
        if rank < 25:
            print(f"{pit[i]:6.3f} {data.y[i]:9.1f} {pred[rank]:10.1f} {gap:+8.1f}  "
                  f"{data.smiles[i][:54]}")

    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote {len(rows)} flagged molecules -> {os.path.normpath(args.out)}")


if __name__ == "__main__":
    main()
