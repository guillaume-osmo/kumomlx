# Exhaustive Z_max for chemical graphs

`dIpc` measures against the n-alkane, which is Z_max for the *acyclic* class. The other end of
the scale is Z_max over all chemical graphs on n atoms: connected, max degree ≤ 4 (the carbon
valence bound). Adding an edge can only add matchings, so the maximum lives on 4-regular graphs —
confirmed at n=10, where the maximum over *all* Δ≤4 connected graphs (89,402 of them) and over
the 4-regular ones is the same, **780**.

## Method

`geng` from [nauty](https://pallini.di.uniroma1.it/) enumerates the graphs; `zmax.c` counts
matchings by subset DP (`f[S]` = matchings induced on `S`, split on the lowest set bit). Python
was far too slow for this — n=14 alone is 88,168 graphs.

```bash
brew install nauty
gcc -O3 -march=native -o zmax zmax.c
geng -c -d4 -D4 -q 14 | ./zmax        # prints: count  Z_max  graph6
```

Generated counts match OEIS A006820 exactly (6 at n=8, 59 at n=10, 1544 at n=12, 88168 at n=14,
805491 at n=15), which is the check that the enumeration is complete.

## Results

| n | graphs | Z_max | log2(Z)/n |
|---|---|---|---|
| 5 | 1 | 26 | 0.94009 |
| 8 | 6 | **209** (K₄,₄) | **0.96342** |
| 10 | 59 | 780 | 0.96073 |
| 12 | 1,544 | 2,921 | 0.95935 |
| 14 | 88,168 | 11,032 | 0.95924 |
| 15 | 805,491 | 21,482 | 0.95939 |

This covers the terpene range outright — monoterpene C₁₀, sesquiterpene C₁₅ — which is where
most natural-product skeletons live (cf. COCONUT).

Two things worth noting. The density **peaks at K₄,₄** and then falls, with an odd/even
oscillation that damps toward ≈0.9593: a *connected* 4-regular graph cannot reach the 0.96342 of
disjoint K₄,₄ blocks, because joining blocks costs edges. And `log2 Z_max` is linear in n to
within **0.028 bits** over the whole enumerated range, so

    log2 Z_max(n) ≈ 0.96196 n − 0.04003

extrapolates cleanly past where enumeration dies (8.0M graphs at n=16, ~10³⁰ by n=40).

## Use

`kumomlx.ipc.z_max_log2(n)` returns the exact value for n ≤ 15 and the fit beyond.
`kumomlx.ipc.matching_saturation(mol)` places a molecule in [0, 1] between the n-alkane and this
ceiling.

The ceiling is a *bound*, not a molecule: a 4-regular carbon graph has no hydrogens at all. Real
molecules occupy the bottom of the range (median saturation −0.01, 95% below 0.21 on the
boiling-point set), so `delta_matching_information` keeps more resolution and remains the better
default descriptor.
