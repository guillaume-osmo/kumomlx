# The RDKit 217 block, and why `Ipc` cannot be used as shipped

The bundled dataset uses the full `rdkit.Chem.Descriptors.descList` — 217 descriptors in RDKit
2025.09.x — with exactly one column replaced. This is the reasoning.

## What `Ipc` actually is

`Ipc` is the Bonchev–Trinajstić *total* information content of the characteristic polynomial of
the hydrogen-suppressed adjacency matrix:

```
c      = |characteristic polynomial coefficients|        (n_atoms + 1 of them)
S      = sum(c)
H      = -sum (c_i/S) log2(c_i/S)        Shannon entropy, bounded by log2(n_atoms + 1)
Ipc    = S * H                           EXTENSIVE
AvgIpc = H                               INTENSIVE
```

So `Ipc` is a bounded information measure multiplied by a size factor. The size factor is the
problem.

## S grows super-exponentially

Measured on linear alkanes:

| n_atoms | S = sum\|c\| | H = AvgIpc | bound log2(n+1) | Ipc = S·H |
|---|---|---|---|---|
| 5 | 8.00e+00 | 1.4056 | 2.5850 | 1.13e+01 |
| 20 | 1.10e+04 | 2.4672 | 4.3923 | 2.70e+04 |
| 50 | 2.04e+10 | 3.1279 | 5.6724 | 6.37e+10 |
| 80 | 3.79e+16 | 3.4669 | 6.3399 | 1.31e+17 |
| 100 | 5.76e+20 | 3.6561 | 6.6582 | 2.10e+21 |
| 120 | 2.74e+26 | *1.4007* | 6.9189 | 3.84e+26 |

`Ipc` leaves **float32** range (3.4e38) near 170 atoms. A pipeline that casts to float32 and maps
non-finite values to zero then hands the *most* complex molecules the same value as a null
feature — a silent sign reversal, strictly worse than dropping the column.

## `AvgIpc` is not safe either

Note the italicised cell: at 120 atoms RDKit's `AvgIpc` reports **1.4007**, *below* its value at
20 atoms, when an entropy bounded by log2(n+1) = 6.92 should still be rising.

Checked against an exact integer characteristic polynomial (Faddeev–LeVerrier over
`fractions.Fraction`; the adjacency matrix is integer, so the coefficients are integers and the
computation is exact), RDKit matches to five decimals up to n = 60 and then collapses:

```
n=100  AvgIpc 3.65611
n=105  AvgIpc 3.77501
n=110  AvgIpc 3.57666   <-- decreased
n=115  AvgIpc 2.89106   <-- decreased
n=120  AvgIpc 1.40068   <-- decreased
```

The exact entropy is monotone and bounded, so this is float64 catastrophic cancellation inside
the characteristic polynomial, not chemistry. **Both** Ipc columns are unreliable above ~110
atoms, and the normalised one fails earlier and far more quietly than the overflowing one.

## The fix

The "denominator normalisation" the descriptor needs is already present in its own definition:
divide by `S`. That is exactly `H = AvgIpc`. It is dimensionless, bounded by construction, and
measures *how unevenly* structural information is distributed rather than how large the molecule
is — and size is already carried by a dozen other descriptors in the block.

Taking `log(Ipc)` also tames the range, but it is the weaker idea on its own: `log(Ipc)` still
grows with molecular size, so it largely duplicates atom count. It earns its place only as the
*separate* extensive term, because the product decomposes cleanly:

```
log2(Ipc) = log2(S) + log2(H)
```

So this repository ships **both halves, each computed in log space** so neither can overflow at
any molecular size:

| column | meaning | behaviour |
|---|---|---|
| `AvgIpc` | H, intensive information content | bounded by log2(n+1) |
| `Ipc_log2` | log2(S) + log2(H), replacing raw `Ipc` | ≈ linear in n |

`log2(S)` is obtained by log-sum-exp over the coefficients, so `S` is never formed. Above
`IPC_MAX_ATOMS = 100` both columns are emitted as **NaN** rather than as a wrong number — Kumo
imputes missing values from the context rows, which is a far better failure mode.

## On the bundled dataset

All 1000 boiling-point molecules are small, so nothing overflows in practice: **0 non-finite
cells**, max |value| 2.0e3. Twelve columns are constant on this set (`NumRadicalElectrons`,
`SMR_VSA8`, `SlogP_VSA9`, and nine `fr_*` substructure counts) and the recipe drops them.

The repair matters for the general case, not for this example — which is the honest reason to
state the measurement rather than the folklore.

## A note on pre-transforming

Applying a squashing transform (`arcsinh`, `log1p`) to descriptors *before* Kumo is mostly
redundant: the recipe already applies, per column, a round-robin choice of identity, power
transform, robust scaling with soft clipping, or rank-Gaussian. Rank-Gaussian is invariant to any
monotone transform, so a pre-applied `arcsinh` is a no-op for those columns and changes only the
identity/power-transform ones. The example therefore feeds raw descriptor values.
