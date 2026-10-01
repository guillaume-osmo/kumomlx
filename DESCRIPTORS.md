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

---

# What is `Ipc`, actually? A reformulation

Fixing the arithmetic leaves a prior question: for a 250-carbon chain the corrected `Ipc` is
**5.48e+52**. What is that number?

## S is the Hosoya index

`S = sum |c_i|` is not an abstract normaliser. For an **acyclic** molecule it is exactly the
**Hosoya index Z** — the total number of matchings of the graph. Verified for linear alkanes,
where it is also the Fibonacci number:

| n | S = Σ\|c\| | Fibonacci(n+1) | Hosoya Z |
|---|---|---|---|
| 5 | 8 | 8 | 8 |
| 12 | 233 | 233 | 233 |
| 20 | 10946 | 10946 | 10946 |

For cyclic molecules the cycle terms make S the weighted Sachs subgraph count instead, slightly
above Z (benzene: S = 20, Z = 18). And it grows geometrically: `log2(S)/n` converges to
**log2(φ) = 0.6942**, the golden ratio, measured 0.648 at n=10 rising to 0.692 at n=250.

So `Ipc` for C₂₅₀ is `Fibonacci(251) × 4.2886`.

## Why the formula is malformed

Bonchev and Trinajstić's total information content, `I = N · H`, is well posed when `N` counts the
**elements of a population** that `H` partitions into classes: N elements × H bits per element =
bits. `Ipc = S · H` does not have that structure:

* `H` is an entropy over the **n+1 coefficient classes**, bounded by `log2(n+1)`;
* `S` counts **matchings**, and grows like φⁿ.

These are two different populations. Multiplying them yields a quantity with neither units nor an
interpretation — it is not "information" in any sense.

## And empirically it is vacuous

On 400 molecules of 3–49 atoms:

| | |
|---|---|
| variance of log2(Ipc) explained by log2(S) alone | **99.79%** |
| range contributed by log2(S) | 32.3 bits |
| range contributed by log2(H) | 2.24 bits |
| correlation of log2(S) with atom count | r = 0.993 |

`Ipc` is the Hosoya index wearing an information-theoretic hat, and the Hosoya index is largely
molecular size. The entropy factor is decoration worth one part in fourteen.

## The reformulation

The information content of a set of Z matchings is **log2 Z** bits — the bits needed to name one
of them. That is the well-posed "total information", and it is extensive (linear in n) as an
extensive quantity should be. Three dimensionally coherent quantities replace the one malformed
product:

| quantity | formula | units | range on the BP set |
|---|---|---|---|
| `matching_information` | log2 Z | bits, extensive | 1.6 – 33.9 |
| `information_density` | log2(Z) / n | bits/atom, intensive | 0.464 – 0.862 |
| `evenness` | J = H / log2(n+1) | dimensionless, [0,1] | 0.279 – 0.808 |

`information_density` is the statistical-mechanics entropy per site of the monomer–dimer model on
the graph, and converges along a homologous series — to log2(φ) for alkanes.

**`AvgIpc` is not the right intensive quantity either.** It does not explode, but it is not
size-free: its own bound `log2(n+1)` grows with the molecule, and raw H still correlates with atom
count at **r = +0.80**. Dividing by the bound gives Pielou evenness `J`, which is genuinely
size-free (**r = +0.15**) — a pure shape descriptor: how evenly structural information is spread
across matching sizes, regardless of molecular size.

| molecule | atoms | log2 Z | bits/atom | J | AvgIpc |
|---|---|---|---|---|---|
| ethanol | 3 | 1.585 | 0.528 | 0.459 | 0.918 |
| benzene | 6 | 4.322 | 0.720 | 0.613 | 1.720 |
| caffeine | 14 | 9.529 | 0.681 | 0.650 | 2.539 |
| cholesterol | 28 | 19.463 | 0.695 | 0.615 | 2.990 |
| C₅₀ chain | 50 | 34.245 | 0.685 | 0.551 | 3.128 |
| C₂₅₀ chain | 250 | 173.094 | 0.692 | 0.538 | 4.289 |

Read the last column against the others: `AvgIpc` climbs monotonically with size and says little
else, while `J` separates benzene and caffeine (dense, even matching structure) from the long
chains, independently of how many atoms they have.

Available as `kumomlx.ipc.matching_information`, `.information_density` and `.evenness`.

## Should the density be per atom, per surface, or per volume?

"We live in 3-D space, so a density should be per volume (n³) or per surface (n²), not per
atom." The physical instinct is right; it does not survive contact with this chemical space, for
three measured reasons.

**1. `n` is already volume-like, not a length.** Atoms pack at roughly constant density, so
n ∝ V. "Per atom" *is* "per unit volume" up to a constant. The n² / n³ exponents would apply if
n were a linear extent, which it is not.

**2. Small molecules are not compact 3-D objects.** Measured over the 1000 boiling-point
molecules (2–63 heavy atoms):

| | empirical exponent | compact sphere would give |
|---|---|---|
| surface area (LabuteASA) ~ n^k | **0.913** | 0.667 |
| volume (MolMR) ~ n^k | **0.926** | 1.000 |

Both are nearly linear. These molecules are chain- and branch-like — effectively
low-dimensional — so surface ≈ volume ≈ n and the geometric distinction evaporates. It would
matter for proteins or nanoparticles, where compactness is real.

**3. The exponent carries no independent information.** `log2(Z)/n^k` is a deterministic
function of the pair `(log2 Z, n)`, so it cannot know anything they do not. Partial correlation
with the boiling-point residual, as the basis is enriched:

| basis | /n^(1/3) | /n^(2/3) | /n |
|---|---|---|---|
| size only (n, MW, ASA), linear | +0.444 | +0.426 | +0.360 |
| size + log2(Z) | +0.247 | +0.220 | +0.060 |
| size + log2(Z) + nonlinear n | **−0.087** | **−0.092** | **−0.096** |

Once `log2(Z)` and nonlinear powers of `n` are both present, every exponent adds nothing and all
three converge. The apparent advantage of `n^(1/3)` in the first row is a nonlinear reshaping the
linear basis could not absorb — not evidence about the dimensionality of space.

**So the exponent is a question of functional form, not of physics.** For a model that learns
nonlinear interactions, ship `log2(Z)` and `n` as separate columns and let it find the
relationship; do not pre-divide and discard a degree of freedom.

`information_density` is kept because it is the right quantity for the *physics* of a homologous
series — it is the monomer–dimer entropy per site and converges to log2(φ) — not because it is
the right feature for a learned model.
