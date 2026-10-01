"""The bundled boiling-point dataset.

1000 molecules with a measured boiling point in degrees Celsius and the full RDKit descriptor
block (217 columns, `Descriptors.descList` for RDKit 2025.09.x) each.

One column is not the stock descriptor: `Ipc` is replaced by `Ipc_log2`, because raw `Ipc` grows
super-exponentially with molecule size and leaves float32 range near 170 atoms. Both Ipc-derived
columns are computed in log space and emitted as NaN above 100 atoms, where RDKit's float64
characteristic polynomial stops being reliable. See `examples/make_bp_features.py` for the
derivation; that script also regenerates this file from SMILES.

Missing cells are kept as NaN rather than filled: Kumo imputes from the context rows, and a
wrong number is worse than no number.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

__all__ = ["load_boiling_point", "BoilingPoint"]

_CSV = Path(__file__).parent / "data" / "bp_1000.csv"


class BoilingPoint:
    """Columns of the bundled dataset.

    Attributes:
        x: ``[n, 217]`` descriptor matrix, float32, NaN where missing.
        y: ``[n]`` boiling point in degrees Celsius, float32.
        smiles: the molecules, for inspection and error analysis.
        feature_names: column names of ``x``.
    """

    def __init__(self, x, y, smiles, feature_names):
        self.x, self.y, self.smiles, self.feature_names = x, y, smiles, feature_names

    def __len__(self) -> int:
        return len(self.y)

    def split(self, n_context: int = 800, seed: int = 0):
        """Shuffle once and cut into context and query halves.

        In-context learning does no training: the context rows ARE the fit, so this is a plain
        random partition rather than a train loop.
        """
        idx = np.random.default_rng(seed).permutation(len(self))
        c, q = idx[:n_context], idx[n_context:]
        return (self.x[c], self.y[c]), (self.x[q], self.y[q]), (c, q)


def load_boiling_point() -> BoilingPoint:
    with open(_CSV) as fh:
        rows = list(csv.DictReader(fh))
    names = [k for k in rows[0] if k not in ("smiles", "boiling_point")]
    x = np.array([[float(r[k]) for k in names] for r in rows], dtype=np.float32)
    y = np.array([float(r["boiling_point"]) for r in rows], dtype=np.float32)
    return BoilingPoint(x, y, [r["smiles"] for r in rows], names)
