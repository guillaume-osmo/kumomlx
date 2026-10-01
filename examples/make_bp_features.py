"""Build the bundled boiling-point feature table: the full RDKit descriptor block, made finite.

Kumo-Tabular is a TABULAR model, so molecules are presented as columns. This uses the complete
`rdkit.Chem.Descriptors.descList` (217 descriptors in RDKit 2025.09.x) rather than a hand-picked
subset, with one descriptor repaired -- see below.

The result is checked in as `src/kumomlx/data/bp_1000.csv` so the example needs no RDKit; rerun
this only if the descriptor set or the RDKit version changes.

----------------------------------------------------------------------------------------------
Why `Ipc` is replaced
----------------------------------------------------------------------------------------------
`Ipc` is the TOTAL information content of the characteristic polynomial of the adjacency matrix:

    c      = |characteristic polynomial coefficients|       (n_atoms + 1 of them)
    S      = sum(c)
    H      = -sum (c_i/S) log2(c_i/S)      Shannon entropy, bounded by log2(n_atoms+1)
    Ipc    = S * H                          EXTENSIVE
    AvgIpc = H                              INTENSIVE

S grows super-exponentially with molecule size (measured: 8.0 at 5 atoms, 2.7e26 at 120), so
`Ipc` leaves float32 range near 170 atoms. A pipeline that casts to float32 and maps non-finite
values to zero then gives the MOST complex molecules the same value as a null feature -- a silent
sign reversal, worse than dropping the column.

The right normalisation is the denominator that is already there: divide by S, which is exactly
H = AvgIpc. It is dimensionless, bounded by construction, and measures how unevenly the
structural information is distributed rather than how big the molecule is -- and size is already
carried by a dozen other descriptors. Taking log(Ipc) also tames the range, but log(Ipc) still
grows with size, so it largely duplicates atom count; it is useful only as the SEPARATE
extensive term, since log2(Ipc) = log2(S) + log2(H) decomposes cleanly.

So this script ships both halves, each computed in log space so neither can overflow:
  * `AvgIpc`   -- H, the intensive information content (kept from the stock descriptor list)
  * `Ipc_log2` -- log2(S) + log2(H), replacing the raw `Ipc` column

One further limit, found by checking RDKit against an exact integer characteristic polynomial:
RDKit's float64 coefficients suffer catastrophic cancellation, and `AvgIpc` stops being monotone
in n from about 110 atoms (3.775 at 105, then 3.577, 2.891, 1.401 ...), while the exact entropy
is monotone and bounded. Both Ipc columns are therefore emitted as NaN above
`IPC_MAX_ATOMS`; Kumo handles missing values natively, which is far better than a wrong number.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import warnings

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Graphs

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

#: Above this, RDKit's float64 characteristic polynomial loses monotonicity (measured at 110).
IPC_MAX_ATOMS = 100


def ipc_terms(mol) -> tuple[float, float]:
    """``(AvgIpc, log2(Ipc))``, both in log space, or ``(nan, nan)`` past the validity limit."""
    if mol.GetNumAtoms() > IPC_MAX_ATOMS:
        return math.nan, math.nan
    c = np.abs(np.asarray(Graphs.CharacteristicPolynomial(mol, Chem.GetAdjacencyMatrix(mol)),
                          dtype=np.float64))
    c = c[c > 0]
    if c.size == 0:
        return math.nan, math.nan
    # log-sum-exp: log2(S) without ever forming S, which is what overflows.
    L = np.log2(c)
    m = L.max()
    log2_S = m + math.log2(float(np.exp2(L - m).sum()))
    H = float((np.exp2(L - log2_S) * (log2_S - L)).sum())
    if not (H > 0 and math.isfinite(H) and math.isfinite(log2_S)):
        return math.nan, math.nan
    return H, log2_S + math.log2(H)


def build_columns():
    """The stock descriptor list, with `Ipc` swapped for its log-space form."""
    cols = []
    for name, fn in Descriptors.descList:
        if name == "Ipc":
            cols.append(("Ipc_log2", lambda m: ipc_terms(m)[1]))
        elif name == "AvgIpc":
            cols.append(("AvgIpc", lambda m: ipc_terms(m)[0]))
        else:
            cols.append((name, fn))
    return cols


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the bundled BP feature table.")
    ap.add_argument("--source", required=True, help="CSV with 'smiles' and 'measured' columns")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "kumomlx", "data", "bp_1000.csv"))
    args = ap.parse_args()

    cols = build_columns()
    print(f"RDKit {Chem.rdBase.rdkitVersion}, {len(cols)} descriptors")

    rows, skipped = [], 0
    for rec in csv.DictReader(open(args.source)):
        mol = Chem.MolFromSmiles(rec["smiles"])
        if mol is None:
            skipped += 1
            continue
        feats = {}
        for name, fn in cols:
            try:
                v = float(fn(mol))
            except Exception:  # noqa: BLE001
                v = math.nan
            # Non-finite values are emitted as empty cells (NaN), never as 0: Kumo imputes
            # missing values from the context, and a wrong number is worse than no number.
            feats[name] = v if math.isfinite(v) else math.nan
        rows.append({"smiles": rec["smiles"], **feats, "boiling_point": float(rec["measured"])})

    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), restval="")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v)
                         for k, v in r.items()})

    arr = np.array([[r[n] for n, _ in cols] for r in rows], dtype=np.float64)
    n_nan = int(np.isnan(arr).sum())
    const = int((np.nanstd(arr, axis=0) == 0).sum())
    print(f"wrote {len(rows)} rows x {len(cols)} descriptors -> {args.out}")
    print(f"  unparseable SMILES skipped : {skipped}")
    print(f"  non-finite cells (left NaN): {n_nan}")
    print(f"  constant columns           : {const} (the recipe drops these)")
    print(f"  max |value|                : {np.nanmax(np.abs(arr)):.3e} "
          f"(float32 max is 3.4e38)")


if __name__ == "__main__":
    main()
