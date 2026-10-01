---
title: "Ipc is the Hosoya index: a numerically stable, size-free reformulation of molecular information content"
author:
  - Claude Opus 5 (Anthropic)
  - Guillaume Godin
date: "2026-10-01"
geometry: margin=2.4cm
fontsize: 10pt
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
abstract: |
  The information content index `Ipc`, available in standard cheminformatics toolkits, is shown
  to be both definitionally redundant and numerically unsound. Over 400 molecules, 99.79% of the
  variance of $\log_2\mathrm{Ipc}$ is explained by $\log_2 S$ alone, where $S$ is the Hosoya
  index; the entropy factor contributes 2.24 bits against 32.3. Separately, the Le
  Verrier–Faddeev–Frame recursion used to obtain the characteristic polynomial loses all
  significant digits above approximately 80 atoms: for a 120-atom chain the exact final
  coefficient is $+1$ and the computed value is $-1.97\times10^{26}$, without overflow or
  warning. We give an eigenvalue-based algorithm with magnitude-ordered root expansion, accurate
  to $2.3\times10^{-15}$ at $n=200$ and eight times faster. We then define
  $\mathrm{dIpc} = \log_2 Z(G) - \log_2 F(n{+}1)$, matching information relative to the
  $n$-alkane of equal size, whose correlation with atom count is $+0.049$ against $+0.993$ for
  $\log_2 Z$. The upper reference $Z_{\max}$ over chemical graphs is determined exhaustively to
  $n=16$, by local search to $n=24$, and by a fitted law accurate to $0.022$ bits thereafter.
  Applied to 6,000 structurally defined terpenes from COCONUT, spanning C10 to C40, against
  skeleton-size-matched natural products from other pathways, terpene carbon skeletons occupy a
  quarter of the topological variance (ratio 4.07, Cohen's $d=-1.16$), consistently across
  skeleton sizes.
---

# Introduction

Molecular descriptor blocks are computed on every molecule in every model that uses them, so a
defective descriptor propagates silently and at scale. RDKit's `descList` contains 217 entries,
two of which — `Ipc` and `AvgIpc` — are information-theoretic indices over the characteristic
polynomial of the adjacency matrix [1,2].

This work asks whether those two indices measure what they are intended to measure, and whether
they are computed correctly. Both answers are negative. We identify the quantity `Ipc` actually
tracks, give a corrected algorithm for the underlying polynomial, propose a reformulation with
analytic references at both ends of its range, and apply it to natural product chemical space.

# Methods

**Exact characteristic polynomial.** The adjacency matrix of a molecular graph is integer, so
its characteristic polynomial has integer coefficients. We compute them exactly by running the
Faddeev–LeVerrier recursion modulo several small primes and reconstructing by the Chinese
remainder theorem, with primes chosen so that every modular matrix product remains exactly
representable in double precision (S2.1). Results were verified against exact rational
arithmetic.

**Information quantities.** With coefficient magnitudes $c$, let $S=\sum|c_i|$ and
$H = -\sum (c_i/S)\log_2(c_i/S)$. The standard descriptors are $\mathrm{Ipc}=S\cdot H$ and
$\mathrm{AvgIpc}=H$. We evaluate both by log-sum-exp so that $S$ is never formed explicitly.

**The dIpc index.** For an acyclic graph, $S$ equals the Hosoya index $Z$, the number of
matchings [1]. Among trees on $n$ vertices the path maximises $Z$, with $Z=F(n{+}1)$, the
Fibonacci number. We therefore define

$$\mathrm{dIpc}(G) \;=\; \log_2 Z(G) \;-\; \log_2 F(n{+}1),$$

which is zero for an unbranched chain, negative for branched acyclic graphs, and positive when
rings contribute additional matchings.

**Upper reference.** Since adding an edge cannot reduce the matching count, the maximum over
chemical graphs (connected, maximum degree $\le 4$) is attained on 4-regular graphs. These were
enumerated exhaustively with `geng` from *nauty* [5] and their matchings counted by subset
dynamic programming. Beyond $n=16$ we used local search over 2-opt edge swaps with an exact
objective, which yields lower bounds (S2.2); beyond $n=24$ the dynamic program exceeds memory
and we use a fitted law bracketed by an explicit construction below and a density ceiling above.

**Natural product data.** COCONUT 2022.01.01 structures with NPClassifier pathway labels
[6,7,8]. The largest fragment is used throughout, as $Z$ is multiplicative over components.

**Structural definition of a terpene.** Terpene identity is a property of the carbon backbone,
not of the decorated molecule: enzymes add oxidation, epoxidation, acetylation and
glycosylation after the skeleton is assembled. We therefore extract the carbon framework —
deleting all non-carbon atoms and retaining the largest connected carbon fragment — and require
it to satisfy the isoprene rule: a skeleton of 10, 15, 20, 25, 30 or 40 carbons — mono- through
tetraterpene (carotenoid) — carrying at least $C/5 - 1$ pendant methyl carbons. This is a structural criterion, independent of the
classifier, and dIpc is computed on the skeleton rather than on the parent molecule.

Controls are single-pathway non-terpenoids, matched one-for-one on skeleton carbon count. They
are *not* filtered by the isoprene rule: excluding structurally isoprenoid controls leaves an
artificially homogeneous residue at small carbon counts (s.d. 0.07, almost entirely linear
chains) and reverses the comparison in those bands.

# Results

## Ipc is dominated by the Hosoya index

![**Left:** `AvgIpc` from RDKit 2025.09.4 compared with exact integer arithmetic for linear carbon chains, $n=10$–160; dotted line is the bound $\log_2(n{+}1)$. **Right:** $\log_2\mathrm{Ipc}$ against $\log_2 S$ for 400 molecules of 3–49 heavy atoms; dashed line is $y=x$.](fig_ipc.png)

The Bonchev–Trinajstić total information content $I = N\cdot H$ is defined for a population of
$N$ elements partitioned into classes by $H$ [2]. In $\mathrm{Ipc}=S\cdot H$, however, $H$ is an
entropy over the $n{+}1$ polynomial coefficients, bounded by $\log_2(n{+}1)$, while $S$ counts
matchings and grows as $\varphi^n$. The two factors refer to different populations.

Empirically the product is dominated by $S$. Across 400 molecules, $\log_2 S$ spans 32.3 bits
and $\log_2 H$ spans 2.24, with $R^2 = 0.9979$ between $\log_2\mathrm{Ipc}$ and $\log_2 S$
(Figure 1, right). As $\log_2 S$ correlates with heavy-atom count at $r=0.993$, `Ipc` is
approximately a measure of molecular size.

## The characteristic polynomial is computed incorrectly above 80 atoms

The Le Verrier–Faddeev–Frame recursion is sequential: coefficient $k$ is formed after $k$ matrix
products have grown the intermediates to the magnitude of the largest coefficient, then obtained
as a difference of quantities of that magnitude, with each coefficient entering the next matrix.

| $n$ | `AvgIpc`, RDKit | exact | coefficients in error by $>1\%$ |
|---|---|---|---|
| 60 | 3.2594 | 3.2594 | 0.0% |
| 80 | 3.46687 | 3.46686 | — |
| 100 | 3.6561 | 3.6278 | 19.6% |
| 120 | 1.4007 | 3.7593 | 23.0% |
| 250 | 1.1084 | 4.2886 | — |

`AvgIpc` is a Shannon entropy bounded by $\log_2(n{+}1)$ and cannot decrease as a chain is
extended; it begins decreasing near 110 atoms (Figure 1, left). Boundedness does not confer
safety: the quantity is computed from the same corrupted coefficients and degrades earlier and
less visibly than the unbounded `Ipc`. No overflow occurs — at $n=250$, $S=1.3\times10^{52}$
against a double-precision ceiling of $1.8\times10^{308}$ — so the returned values are finite.

Expanding the polynomial from its eigenvalues, with root factors multiplied in order of
increasing magnitude, keeps each partial product near the scale of the final coefficients.
Maximum relative error falls from $5.4\times10^{15}$ to $2.3\times10^{-15}$ at $n=200$, with an
eight-fold speed-up. The ordering is essential: unordered expansion still reaches
$4.8\times10^{4}$ relative error at $n=160$. The correction has been submitted upstream
(rdkit/rdkit#9657).

## dIpc removes the size dependence that other normalisations do not

| quantity | correlation with heavy-atom count |
|---|---|
| $\log_2 Z$ | $+0.993$ |
| $\log_2(Z)/n$ | $+0.409$ |
| $\mathrm{dIpc}$ | $+0.049$ |

Subtracting $\log_2 F(n{+}1)$ removes the size term exactly, where division by a power of $n$
removes it only approximately. The resulting index is signed and interpretable: zero for an
$n$-alkane, negative with branching, positive with ring fusion. Across 478 acyclic molecules no
value exceeded zero, consistent with the extremal property of the path.

## Z_max is exact to n = 16 and bounded to 0.022 bits beyond

![$\log_2 Z$ against $n$. Blue: exhaustive enumeration, exact. Orange: local search, lower bounds. Green: an explicit $K_{4,4}$-chain construction. Shaded: the interval between the $n$-alkane reference and $Z_{\max}$. Lower panel: residuals of the two approximations.](zmax_fit.png)

| $n$ | $Z_{\max}$ | $\log_2 Z/n$ | basis |
|---|---|---|---|
| 12 | 2,921 | 0.95935 | exhaustive |
| 14 | 11,032 | 0.95924 | exhaustive |
| 16 | 42,439 | 0.96082 | exhaustive |
| 20 | 598,516 | 0.95957 | local search |
| 24 | 8,659,688 | 0.96025 | local search |
| 30 | $4.69\times10^{8}$ | — | fitted |
| 40 | $3.66\times10^{11}$ | — | fitted |
| 50 | $2.86\times10^{14}$ | — | fitted |

The fitted law $\log_2 Z_{\max}(n) = 0.96092\,n - 0.02313$ holds to $\pm0.022$ bits over
$n=12$–24. It is bounded below by an explicit construction — a chain of $K_{4,4}$ blocks, each
with one internal edge removed so that connector vertices remain within degree 4 — whose exact
$Z$ is obtained by transfer matrix at any size and which lies a constant 0.22 bits below the fit
up to $n=248$. It is bounded above by the $K_{4,4}$ density ceiling $0.96342\,n$. At $n=50$ the
two bounds differ by 0.25 bits.

The density is maximal at $K_{4,4}$ and the sequence shows period-8 structure: a connected
4-regular graph cannot attain the density of disjoint $K_{4,4}$ blocks, as joining two blocks
costs 0.0416 bits.

## Terpene skeletons occupy a quarter of the topological variance of other pathways

![**Left:** dIpc of the carbon skeleton for 6,000 structurally defined terpenes and 6,000 non-terpenoid natural products matched on skeleton carbon count. **Right:** standard deviation by skeleton size, variance ratio annotated.](fig_terpene.png)

The isoprene rule is satisfied by 77.2% of NPClassifier-labelled terpenoids and by 11.2% of
other pathways, a seven-fold enrichment that provides independent structural support for the
labels (S3.3). Carbon counts agree exactly with the predicted class for 71.3% of 11,871 labelled
terpenoids and to within three carbons for 87.7%, the remainder being degraded or rearranged
skeletons; carotenoids agree best (87.3% exact). Decoration is substantial: the median labelled
terpenoid carries seven non-carbon heavy atoms, and monoterpenoids carry more decoration than
backbone.

| | terpene | non-terpenoid | test |
|---|---|---|---|
| $n$ | 6,000 | 6,000 | — |
| mean skeleton carbons | 20.73 | 20.73 | matched exactly |
| mean dIpc | $+0.054$ | $+0.648$ | Cohen's $d = -1.16$ |
| s.d. | 0.322 | 0.650 | Levene $p < 10^{-300}$ |
| variance ratio | — | 4.065 | — |

Terpene skeletons occupy approximately a quarter of the topological variance of size-matched
natural products from other pathways, and lie close to the $n$-alkane reference while other
pathways lie well to the ring-fused side. Variance ratios by skeleton size are 1.37, 2.50, 2.46,
8.90 and 4.88 over 6–14, 15–19, 20–24, 25–32 and 33–60 carbons, covering the full terpene range
from monoterpene to carotenoid.

Class means separate within the terpenes. Carotenoids are the extreme: a C40 skeleton carrying
eight methyl branches gives mean dIpc $-1.10$ with s.d. 0.218 ($n=81$), the most branched and the
most topologically uniform of any class (S3.3).

The effect strengthens as the measured object approaches the biosynthetic one:

| definition | variance ratio | Cohen's $d$ |
|---|---|---|
| whole molecule, classifier labels | 2.045 | $-0.86$ |
| carbon skeleton, classifier labels | 3.223 | $-0.96$ |
| carbon skeleton, isoprene rule | 4.065 | $-1.16$ |

Removing enzymatic decoration and restricting to skeletons that satisfy the isoprene rule each
increase the separation, which is the behaviour expected if the constraint acts on the backbone
rather than on the decorated product.

# Discussion

`Ipc` should be removed from descriptor blocks. It is approximately a function of molecular
size, which is already represented by numerous other descriptors, and it is computed from a
polynomial that is unreliable above 80 atoms. Models trained on molecules below about 60 atoms
are unaffected by the correction, since old and corrected values agree to $10^{-15}$ in that
range; above it, feature distributions change and should be re-examined rather than simply
recomputed.

`AvgIpc` should be retained once the underlying polynomial is corrected, and `dIpc` added
alongside it. The cost of dIpc is approximately 10 ms per molecule at $n=50$ using the exact
path. It contributes no additional information to a model already provided with $\log_2 Z$ and
$n$ and sufficient flexibility to combine them (partial $r=+0.0095$); its value lies in
interpretability, in extrapolation beyond the training size range, and in settings where the
model cannot learn the interaction. For a feature matrix, $\log_2 Z$ and $n$ are better supplied
as separate columns than pre-combined: division by $n^{1/3}$, $n^{2/3}$ or $n$ adds nothing once
both are present (partial $r$ of $-0.087$, $-0.092$, $-0.096$).

Terpene biosynthesis leaves a measurable signature in graph topology. Assembly from a single
branched C5 unit confines terpene carbon skeletons to a quarter of the topological variance of
skeleton-size-matched natural products from other pathways, holding them near the unbranched
chain reference while other pathways extend to the ring-fused side. The effect is large
($d=-1.16$) and uniform in direction across skeleton sizes from C10 to C40.

That it strengthens monotonically as decoration is removed and as the terpene definition is
tightened from a predicted label to a structural criterion indicates that the constraint acts on
the backbone, as the biosynthesis implies, rather than on the elaborated product. dIpc is
consequently usable as a size-independent descriptor of biosynthetic class, not only as a repair
of `Ipc`.

# Limitations

The isoprene rule applied here is a necessary condition, not a decision procedure: a carbon
count divisible by five with the expected methyl density admits some non-terpenoid skeletons and
excludes rearranged or degraded terpenes, including steroids and apocarotenoids. The 23% of
labelled terpenoids it rejects are not a random sample. Carotenoids and sesterterpenes are
represented by 81 and 72 skeletons respectively, so their class means are the least precise.

Control composition is consequential. Filtering controls to exclude structurally isoprenoid
molecules inverts the comparison below 20 carbons, because what survives at those sizes is
almost entirely linear chains (s.d. 0.07); controls are therefore defined by pathway alone.
COCONUT 2022.01.01 is a literature aggregation and no deduplication was applied beyond fragment
selection, so related scaffolds may be over-represented in both groups.

$Z_{\max}$ values for $n=17$–24 are lower bounds rather than proven maxima, with a validated
worst-case shortfall of 0.48%; the fitted law is correspondingly a slight underestimate. The
optimality of $K_{4,4}$ is confirmed exhaustively at $n=8$ and $n=16$ but its asymptotic form is
taken from the literature.

dIpc contributes no additional information to a model already supplied with $\log_2 Z$ and $n$
and able to combine them nonlinearly; its value is interpretability and extrapolation, not
predictive gain. Its components are established prior results [1,4]; a literature search over
OpenAlex returned no previous definition of $\log Z - \log F(n{+}1)$ as a descriptor, but this
is not an exhaustive novelty search.

# References

1. H. Hosoya. Topological Index. A Newly Proposed Quantity Characterizing the Topological Nature
   of Structural Isomers of Saturated Hydrocarbons. *Bull. Chem. Soc. Jpn.* **44**, 2332 (1971).
   [10.1246/bcsj.44.2332](https://doi.org/10.1246/bcsj.44.2332)
2. D. Bonchev, N. Trinajstić. Information theory, distance matrix, and molecular branching.
   *J. Chem. Phys.* **67**, 4517 (1977). [10.1063/1.434593](https://doi.org/10.1063/1.434593)
3. S. H. Bertz. Branching in graphs and molecules. *Discrete Appl. Math.* **19**, 65 (1988).
   [10.1016/0166-218X(88)90006-6](https://doi.org/10.1016/0166-218X(88)90006-6)
4. H. Hosoya. Chemical meaning of octane number analyzed by topological indices (2002).
5. B. D. McKay, A. Piperno. Practical graph isomorphism, II. *J. Symb. Comput.* **60**, 94 (2014).
   [10.1016/j.jsc.2013.09.003](https://doi.org/10.1016/j.jsc.2013.09.003)
6. M. Sorokina, P. Merseburger, K. Rajan, M. A. Yirik, C. Steinbeck. COCONUT online: Collection
   of Open Natural Products database. *J. Cheminform.* **13**, 2 (2021).
   [10.1186/s13321-020-00478-9](https://doi.org/10.1186/s13321-020-00478-9)
7. H. W. Kim *et al.* NPClassifier: A Deep Neural Network-Based Structural Classification Tool
   for Natural Products. *J. Nat. Prod.* **84**, 2795 (2021).
   [10.1021/acs.jnatprod.1c00399](https://doi.org/10.1021/acs.jnatprod.1c00399)
8. NPClassifier predictions of COCONUT compounds.
   [10.5281/zenodo.10629838](https://doi.org/10.5281/zenodo.10629838)

\newpage

# Supplementary information

## S1. Data

| source | count | range | use |
|---|---|---|---|
| boiling-point set | 1,000 molecules | 2–63 heavy atoms | general chemical space |
| — variance decomposition subset | 400 | 3–49 | Figure 1 |
| COCONUT 2022.01.01, NPClassifier-labelled | 407,029 | — | terpenoid pool |
| — labelled terpenoids, skeleton validation | 11,871 | — | S3.3 |
| — terpenes passing the isoprene rule | 6,000 | 10–40 skeleton C | Figure 3 |
| — pathway controls, skeleton-size-matched | 6,000 | 10–40 skeleton C | Figure 3 |
| connected 4-regular graphs | 8,037,418 at $n{=}16$ | $n=5$–16 | $Z_{\max}$ |

Terpenoid superclass counts in COCONUT: diterpenoids 23,082; triterpenoids 22,280;
sesquiterpenoids 20,418; steroids 16,768; monoterpenoids 6,207; meroterpenoids 3,607;
sesterterpenoids 937; carotenoids 953.

## S2. Algorithms

### S2.1 Modular characteristic polynomial

Primes are selected such that $n(p-1)^2 < 2^{53}$, which keeps each modular matrix product
exactly representable in double precision and allows the inner loop to use a floating-point
matrix multiply rather than arbitrary-precision integer arithmetic. The number of primes is set
from a Hadamard bound on the coefficient magnitudes. Output was verified identical to
Faddeev–LeVerrier over exact rationals on random sparse symmetric graphs up to $n=30$.

### S2.2 Local search over 4-regular graphs

Initialisation is a circulant graph $C_n(1,2)$, which is 4-regular and connected. Each step
proposes a 2-opt edge swap: edges $(a,b)$ and $(c,d)$ are deleted and $(a,c)$, $(b,d)$ inserted,
leaving all vertex degrees unchanged. Swaps that disconnect the graph are rejected; the exact
matching count is recomputed and the swap accepted when it does not decrease. Multiple restarts
from independent random states are used. Every reported value is the exact $Z$ of a realised
graph and is therefore a lower bound on $Z_{\max}$.

Validation against exhaustively known maxima: recovered at $n=10$, 12, 13 and 15; at $n=14$ the
best of 24 restarts fell 0.48% short.

### S2.3 Block-chain transfer matrix

For blocks joined in a path by single edges, with per-block counts $N$, $N_l$, $N_r$, $N_{lr}$
(total matchings, and those leaving the left, right, or both connectors unmatched),

$$u_{i+1} = u_i N_{lr} + (u_i+m_i)N_r, \qquad m_{i+1} = u_i(N_l-N_{lr}) + (u_i+m_i)(N-N_r),$$

where $u$ and $m$ count configurations with the right connector unmatched and matched. This is
exact and linear in the number of blocks. An initial version used intact $K_{4,4}$ blocks, which
raises connector degree to 5 and violates the valence constraint; one internal edge is therefore
removed per block.

## S3. Supplementary results

### S3.1 Timings

| step | $n$ | wall time |
|---|---|---|
| exhaustive enumeration and exact $Z$ | 10 | 0.03 s |
| | 12 | 0.07 s |
| | 14 | 4.3 s |
| | 15 | 52 s |
| | 16 | 1,140 s |
| local search, 2,000 moves | 20 | 2.1 s |
| | 24 | 42 s |
| block-chain transfer matrix | $\le 248$ | 0.17 s |

Exhaustive enumeration scales at roughly 10–20× per added atom; $n=17$ would require
approximately six hours and $n=18$ approximately three days.

### S3.2 Exhaustive $Z_{\max}$

| $n$ | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $Z_{\max}$ | 26 | 51 | 100 | 209 | 388 | 780 | 1482 | 2921 | 5600 | 11032 | 21482 | 42439 |

Generated graph counts match OEIS A006820. The maximum at $n=8$ is $K_{4,4}$. At $n=10$ the
maximum over all connected graphs of maximum degree $\le4$ (89,402 graphs) equals the maximum
over 4-regular graphs, consistent with edge monotonicity.

### S3.3 Structural validation and class breakdown

Carbon-skeleton size against the size implied by the predicted class, 11,871 labelled terpenoids:

| class | expected C | $n$ | exact | within 1 | within 3 | median decoration |
|---|---|---|---|---|---|---|
| Monoterpenoid | 10 | 2,482 | 58.0% | 78.1% | 85.5% | 11 |
| Sesquiterpenoid | 15 | 2,500 | 83.5% | 89.7% | 92.3% | 5 |
| Diterpenoid | 20 | 2,500 | 74.3% | 83.5% | 87.2% | 7 |
| Sesterterpenoid | 25 | 936 | 60.8% | 76.6% | 87.1% | 5 |
| Triterpenoid | 30 | 2,500 | 67.0% | 78.8% | 82.6% | 14 |
| Carotenoid | 40 | 953 | 87.3% | 90.8% | 96.5% | 4 |
| all | | 11,871 | 71.3% | 82.7% | 87.7% | 7 |

Decoration is the count of heavy atoms outside the carbon framework; 2.1% of labelled terpenoids
are undecorated and the 90th percentile is 30.

SMARTS substructure frequencies, labelled terpenoids against other pathways ($n\approx6{,}000$ each):

| pattern | SMARTS | terpenoid | other | enrichment |
|---|---|---|---|---|
| isopropenyl | `[CH3][CX3](=[CH2])[#6]` | 5.6% | 0.7% | 7.9x |
| trisubstituted alkene | `[CH3][CX3](=[CX3])[#6]` | 37.2% | 7.0% | 5.3x |
| gem-dimethyl | `[CH3][CX4]([CH3])[#6]` | 56.7% | 11.5% | 4.9x |
| isoprene C5 frame | `[CH3][#6]([#6])[#6][#6]` | 92.7% | 28.6% | 3.2x |
| isopropyl | `[CH3][CHX4]([CH3])[#6]` | 12.7% | 6.2% | 2.1x |

The composite rule — skeleton carbons in $\{10,15,20,25,30,40\}$ with at least $C/5-1$ pendant
methyls — is satisfied by 77.2% of labelled terpenoids and 11.2% of other pathways.

dIpc of the carbon skeleton, by class, within the 6,000 terpenes passing the rule:

| class | $n$ | mean skeleton C | mean dIpc | s.d. |
|---|---|---|---|---|
| Monoterpene C10 | 441 | 10.7 | $-0.099$ | 0.302 |
| Sesquiterpene C15 | 1,886 | 15.4 | $+0.110$ | 0.261 |
| Diterpene C20 | 1,963 | 20.1 | $+0.098$ | 0.339 |
| Sesterterpene C25 | 72 | 25.1 | $-0.329$ | 0.451 |
| Triterpene C30 | 1,557 | 29.7 | $+0.054$ | 0.217 |
| Carotenoid C40 | 81 | 39.9 | $-1.101$ | 0.218 |
| all terpenes | 6,000 | 20.7 | $+0.054$ | 0.322 |

### S3.4 Normalisation exponents

Partial correlation with the boiling-point residual for $\log_2(Z)/n^k$, as the regression basis
is enriched:

| basis | $k=1/3$ | $k=2/3$ | $k=1$ |
|---|---|---|---|
| size terms only, linear | $+0.444$ | $+0.426$ | $+0.360$ |
| size $+\log_2 Z$ | $+0.247$ | $+0.220$ | $+0.060$ |
| size $+\log_2 Z+$ nonlinear $n$ | $-0.087$ | $-0.092$ | $-0.096$ |

## S4. Software and reproducibility

Code, data and figures: **github.com/guillaume-osmo/kumomlx**

| component | path |
|---|---|
| exact characteristic polynomial, dIpc, $Z_{\max}$ | `src/kumomlx/ipc.py` |
| derivation and measurements | `DESCRIPTORS.md` |
| upstream patch and Eigen C++ implementation | `rdkit-fix/` |
| $Z_{\max}$ enumeration | `rdkit-fix/zmax/` |
| COCONUT analysis | `paper/coconut.py`, `paper/coconut_stats.py` |
| figures | `paper/fig_*.py` |

RDKit 2025.09.4; nauty 2.9.3, invoked as `geng -c -d4 -D4 n`.
