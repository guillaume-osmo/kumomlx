# Fixing RDKit's characteristic polynomial

`rdkit.Chem.Graphs.CharacteristicPolynomial` uses the Le Verrier–Faddeev–Frame recursion in
float64. It is a **sequential** method: coefficient *k* is produced after *k* matrix products have
grown the intermediates to the magnitude of the largest coefficient, and it is obtained as a
difference of quantities of that size. Once that size exceeds ~1e16, float64 has no significant
digits left, and because each wrong coefficient feeds the next matrix, the error amplifies.

For a 120-atom chain the true final coefficient (the determinant) is **+1**; RDKit returns
**−1.97e26** — wrong sign, 26 orders of magnitude out. **Nothing overflows**, so the failure is
completely silent.

## Consequences

`Ipc` and `AvgIpc` are computed from these coefficients. The spurious tail values exceed the
genuine ones, so they dominate the normalised distribution and the entropy collapses:

| n_atoms | RDKit AvgIpc | exact | coefficients wrong by >1% |
|---|---|---|---|
| 60 | 3.2594 | 3.2594 | 0.0% |
| 100 | 3.6561 | 3.6278 | 19.6% |
| 120 | **1.4007** | 3.7593 | 23.0% |
| 250 | **1.1084** | 4.2886 | — |

`AvgIpc` is an entropy bounded by log2(n+1); it must not *decrease* in n. It starts decreasing at
about 110 atoms. Note that `AvgIpc` being bounded does **not** make it safe — it is computed from
the same corrupted coefficients, and fails earlier and more quietly than the extensive `Ipc`.

## The fix

Expand the polynomial from its roots instead of by recursion, with the roots ordered by
increasing magnitude so that every partial product stays near the scale of the final
coefficients. For a symmetric adjacency matrix the eigenvalues are computed by a backward-stable
symmetric solver.

Measured against exact integer arithmetic (Faddeev–LeVerrier over `fractions.Fraction`, and a
modular-CRT implementation — see `kumomlx/ipc.py`):

| n | current, max rel. error | **proposed** | current | **proposed** |
|---|---|---|---|---|
| 80 | 4.58e-07 | **1.6e-15** | 0.71 ms | — |
| 120 | 1.87e+02 | **3.8e-16** | 2.20 ms | **0.62 ms** |
| 200 | 5.36e+15 | **2.3e-15** | 12.18 ms | **1.45 ms** |

Machine precision throughout, and **faster** than the recursion from ~120 atoms. It agrees with
the current implementation to 3.9e-13 on small molecules, where the current one is still correct,
so the change is backward compatible in the regime anyone has been relying on.

Ordering the roots is what does the work: `numpy.poly(eigvalsh(A))`, which multiplies them in
arbitrary order, still reaches 4.8e+04 relative error at 160 atoms.

## Files

    graphs.py.patch      the upstream Python fix (rdkit/Chem/Graphs.py)
    charpoly_eigen.cpp   the same algorithm in C++ using Eigen, as RDKit's C++ does not link
                         LAPACKE; also includes ipcTerms() returning (AvgIpc, log2(Ipc))
    test_charpoly.cpp    checks the C++ output against exact integer values

Build the C++ check with:

    g++ -std=c++17 -O2 -I/path/to/eigen3 charpoly_eigen.cpp test_charpoly.cpp -o test_charpoly

The last coefficient should print as ±1.000000 at every size.

## A caveat for anyone reimplementing the descriptor

`Ipc` remains *extensive* even once computed correctly: S reaches 2.7e26 at 120 atoms and leaves
float32 range near 170. That is a representation problem rather than a numerical one, and is
addressed separately by reporting `log2(Ipc) = log2(S) + log2(H)` and the intensive `AvgIpc`.
See [../DESCRIPTORS.md](../DESCRIPTORS.md).

**Fixing this changes descriptor values**, so any model trained on the old numbers sees feature
drift. On molecules below ~60 atoms the two agree to 1e-15 and nothing changes; above that they
diverge. Verify against your own training distribution before swapping it in.
