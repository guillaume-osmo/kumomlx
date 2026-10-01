---
title: "Ipc is the Hosoya index, computed wrongly"
subtitle: "A bounded reformulation of molecular information content, and what it measures in terpene space"
author: "Guillaume Godin"
date: "2026-10-01"
geometry: margin=2.4cm
fontsize: 10pt
colorlinks: true
linkcolor: NavyBlue
urlcolor: NavyBlue
header-includes:
  - \usepackage{xcolor}
  - \definecolor{tldrbg}{HTML}{EEF4FB}
  - \definecolor{warnbg}{HTML}{FDF3F3}
  - \newsavebox{\cbox}
  - \newenvironment{tldr}{\par\medskip\begin{lrbox}{\cbox}\begin{minipage}{\dimexpr\linewidth-2em}}{\end{minipage}\end{lrbox}\noindent\colorbox{tldrbg}{\hspace{0.5em}\usebox{\cbox}\hspace{0.5em}}\par\medskip}
  - \newenvironment{warn}{\par\medskip\begin{lrbox}{\cbox}\begin{minipage}{\dimexpr\linewidth-2em}}{\end{minipage}\end{lrbox}\noindent\colorbox{warnbg}{\hspace{0.5em}\usebox{\cbox}\hspace{0.5em}}\par\medskip}
---

\begin{tldr}
\textbf{Verdict.} RDKit's \texttt{Ipc} descriptor should not be used as shipped, for two
independent reasons. \emph{Definitionally}, it is the Hosoya index wearing an
information-theoretic hat: $99.79\%$ of the variance of $\log_2\mathrm{Ipc}$ is explained by
$\log_2 S$ alone ($n=400$ molecules), where $S$ is simply the number of matchings of the
molecular graph. \emph{Numerically}, RDKit computes it with a recursion that loses every
significant digit above about 80 atoms --- for a 120-atom chain the true final polynomial
coefficient is $+1$ and RDKit returns $-1.97\times10^{26}$, silently, with nothing overflowing.

\medskip
\textbf{Replacement.} Measure matching information against the $n$-alkane of the same size,
$\mathrm{dIpc} = \log_2 Z(G) - \log_2 F(n{+}1)$. The baseline is analytic (a Fibonacci number),
the quantity is signed and interpretable (negative = branched, positive = ring-fused), and it is
the only form that is genuinely size-free: correlation with atom count $+0.049$, against
$+0.993$ for raw $\log_2 Z$ and $+0.409$ for $\log_2(Z)/n$.

\medskip
\textbf{Scale.} The upper reference $Z_{\max}$ is now known exactly to $n=16$ by exhaustive
enumeration (8{,}037{,}418 graphs), by validated search to $n=24$, and to within $0.022$ bits by
a fitted law beyond: $\log_2 Z_{\max}(n) = 0.96092\,n - 0.02313$.

\medskip
\textbf{Application.} On COCONUT ($n=6{,}121$ terpenoids against exactly size-matched
non-terpenoids), terpenoid topological space is \textbf{half as variable} as the rest of natural
product chemistry --- variance ratio $2.05$, $p = 1.4\times10^{-146}$, and the gap holds in every
size stratum from 6 to 60 heavy atoms. Terpenoids are also substantially more branched: mean
$\mathrm{dIpc}$ $-0.058$ against $+0.541$, Cohen's $d = -0.86$.
\end{tldr}

## The question

A molecular descriptor block is a fixed cost paid on every molecule in every model. RDKit's
`descList` has 217 entries and two of them, `Ipc` and `AvgIpc`, are information-theoretic indices
over the characteristic polynomial of the adjacency matrix. The decision is whether to keep them,
and if not, what to put in their place that measures the same intended thing — how structurally
"rich" a molecular graph is — without the defects.

This matters beyond tidiness. A descriptor that is silently wrong above a size threshold corrupts
exactly the large natural products where structure is most interesting, and a descriptor that is
99% redundant with molecular size wastes a column and inflates apparent feature counts.

## Data census

| source | count | range | role |
|---|---|---|---|
| boiling-point set (`bpsubset.csv`) | 1,000 molecules | 2–63 heavy atoms | general chemical space |
| — subset used for variance decomposition | 400 molecules | 3–49 heavy atoms | Finding 1 |
| — size-matched to terpenes | 609 molecules | 9–35 heavy atoms | Finding 3 control |
| COCONUT 2022.01.01, NPClassifier-labelled | 407,029 compounds | — | Finding 3 pool |
| — terpenoids sampled, 5 superclasses | 6,121 | 6–60 heavy atoms | Finding 3 |
| — non-terpenoid controls, exactly size-matched | 6,121 | 6–60 heavy atoms | Finding 3 control |
| curated terpenes (pilot) | 43 (of 46 proposed) | 10–30 carbons | Finding 3 pilot |
| connected 4-regular graphs, `geng` | 8,037,418 at $n{=}16$ | $n = 5\ldots16$ | $Z_{\max}$, exact |
| local-search runs | 8 sizes × 3–16 restarts | $n = 17\ldots24$ | $Z_{\max}$, lower bound |

Finding 3 was first run on a curated set, then repeated on COCONUT; both are reported, because
the disagreement between them is itself informative. For the curated set, every entry was checked
against the carbon count its class requires — C10 monoterpene, C15
sesquiterpene, C20 diterpene, C30 triterpene. **Three of 46 failed and were dropped**
(fenchone at 9 C, alpha-humulene and valencene at 14 C), then corrected and re-validated; the check
exists because a mistranscribed SMILES almost always has the wrong carbon count.

For COCONUT, terpenoids are those with NPClassifier pathway exactly `Terpenoids` and one of five
terpene superclasses; the 4,706 compounds labelled `Alkaloids,Terpenoids` are excluded rather
than assigned. Controls are single-pathway non-terpenoids, **matched one-for-one on heavy-atom
count** (mean size agrees to $10^{-4}$ atoms). A range restriction would have left a size
gradient inside the window, and the spread of dIpc grows with size, which would have manufactured
the effect under test. The largest fragment is taken, since COCONUT contains salts and $Z$ is
multiplicative over components, so a counter-ion would read as a large negative dIpc.

## Method

**Exact characteristic polynomial.** The adjacency matrix is integer, so its characteristic
polynomial has integer coefficients. They are computed exactly by running the
Faddeev–LeVerrier recursion modulo several small primes and reconstructing with the Chinese
remainder theorem. Primes are chosen so that $n(p-1)^2 < 2^{53}$, which keeps every modular
matrix product exactly representable in float64 — so the inner loop is a BLAS matmul rather than
Python big-integer arithmetic. Verified identical to exact rational arithmetic
(`fractions.Fraction`) on random molecular-like graphs up to $n=30$.

**The entropy, in log space.** With $c$ the coefficient magnitudes, $S=\sum|c_i|$ and
$H = -\sum (c_i/S)\log_2(c_i/S)$, the stock descriptors are $\mathrm{Ipc}=S\cdot H$ and
$\mathrm{AvgIpc}=H$. Both are evaluated here via log-sum-exp, so $S$ is never formed and nothing
overflows at any molecular size.

**dIpc.** For an acyclic molecule (a *tree*, in graph terms), $S$ is exactly the **Hosoya index**
$Z$ — the total number of matchings, i.e. ways to choose a set of bonds no two of which share an
atom. Among trees on $n$ vertices the path *maximises* $Z$, and for a path $Z = F(n{+}1)$, the
Fibonacci number. So

$$\mathrm{dIpc}(G) \;=\; \log_2 Z(G) \;-\; \log_2 F(n{+}1)$$

is zero for an unbranched chain, negative for any branched acyclic, and positive only when rings
contribute extra matchings.

**$Z_{\max}$ by three methods, because each one runs out.** Adding an edge can only add
matchings, so the maximum over chemical graphs (connected, maximum degree $\le 4$, the carbon
valence bound) is attained on 4-regular graphs. These were enumerated exhaustively with `geng`
from *nauty*, with matchings counted by subset dynamic programming in C. Beyond $n=16$ that is
infeasible, so a **local search** was used: start from a circulant graph (4-regular and
connected), repeatedly propose a 2-opt edge swap — delete edges $(a,b)$ and $(c,d)$, add $(a,c)$
and $(b,d)$, which leaves every vertex degree unchanged — accept the swap when the exact matching
count does not decrease, and restart from a new random starting point many times. It is "local"
because every step is a small perturbation of the current graph rather than a fresh draw, and it
returns a *lower bound*, since each value is the exact $Z$ of a real graph. Beyond $n=24$ the
subset DP exceeds memory ($2^{24}$ states), so the remaining sizes are a fitted law, bracketed
below by an explicit construction and above by a density ceiling.

\begin{warn}
\textbf{What this method cannot capture.} Four things.
\emph{(i)} The local search is not a proof of optimality. Validated against the known exact
maxima it recovered them at 4 of 5 tested sizes, missing $n=14$ by $0.48\%$, so the $n=17\ldots24$
values may be marginally low.
\emph{(ii)} $Z_{\max}$ is a bound, not a molecule: a 4-regular carbon graph has no hydrogens at
all and does not exist. Real molecules occupy the bottom of that range.
\emph{(iii)} dIpc is a deterministic function of the pair $(\log_2 Z, n)$, so against a model
already given both with enough flexibility to combine them it adds no new information — measured
partial correlation $+0.0095$ against the boiling-point residual. Its value is as a
\emph{parameterisation}, not as extra signal.
\emph{(iv)} 43 curated terpenes is a small, hand-assembled sample, not a draw from a natural
product database. Finding 3 should be read as a hypothesis worth testing on COCONUT, not as an
established property of natural product space.
\end{warn}

## Finding 1: Ipc is the Hosoya index, and RDKit computes it wrongly above 80 atoms

![**Left:** RDKit's `AvgIpc` against exact integer arithmetic, linear carbon chains, $n=10$–160. The dotted line is the theoretical bound $\log_2(n{+}1)$. **Right:** $\log_2\mathrm{Ipc}$ against $\log_2 S$ for 400 molecules of 3–49 heavy atoms; dashed line is $y=x$.](fig_ipc.png)

Bonchev and Trinajstić's total information content $I = N\cdot H$ is well posed when $N$ counts
the *elements* that $H$ partitions into classes: $N$ elements × $H$ bits per element = bits.
$\mathrm{Ipc} = S\cdot H$ does not have that structure. $H$ is an entropy over the $n{+}1$
**coefficient classes**, bounded by $\log_2(n{+}1)$; $S$ counts **matchings** and grows like
$\varphi^n$. The product multiplies two different populations and carries neither units nor an
interpretation.

Empirically it is also vacuous. Over 400 molecules, $\log_2 S$ spans 32.3 bits while $\log_2 H$
spans 2.24 — a ratio of 14 to 1 — and $R^2 = 0.9979$ between $\log_2 \mathrm{Ipc}$ and
$\log_2 S$. Since $\log_2 S$ correlates with atom count at $r = 0.993$, `Ipc` is, to a very good
approximation, molecular size.

Separately, the implementation is broken. RDKit uses the Le Verrier–Faddeev–Frame recursion in
float64. It is *sequential*: coefficient $k$ is produced after $k$ matrix products have grown the
intermediates to the magnitude of the largest coefficient, then obtained as a *difference* of
quantities of that size, and each wrong coefficient feeds the next matrix.

| $n$ | RDKit `AvgIpc` | exact | coefficients wrong by $>1\%$ |
|---|---|---|---|
| 60 | 3.2594 | 3.2594 | 0.0% |
| 80 | 3.46687 | 3.46686 | — |
| 100 | 3.6561 | 3.6278 | 19.6% |
| 120 | **1.4007** | 3.7593 | 23.0% |
| 250 | **1.1084** | 4.2886 | — |

`AvgIpc` is a Shannon entropy and cannot decrease as a chain is extended; it starts decreasing at
about 110 atoms. Note that being *bounded* does not make it safe — it is computed from the same
corrupted coefficients and fails earlier and far more quietly than the unbounded `Ipc`.

**Nothing overflows.** At $n=250$, $S = 1.3\times10^{52}$ against a float64 ceiling of
$1.8\times10^{308}$. The failure is pure cancellation, and the returned value is finite and
plausible-looking.

The fix is to expand the polynomial from its eigenvalues with the roots multiplied in order of
increasing magnitude, which keeps every partial product near the scale of the final coefficients.
Maximum relative error falls from $5.4\times10^{15}$ to $2.3\times10^{-15}$ at $n=200$, and it is
8× *faster*. Ordering is what does the work: `numpy.poly(eigvalsh(A))`, which multiplies in
arbitrary order, still reaches $4.8\times10^{4}$ relative error at $n=160$. Submitted upstream as
[rdkit/rdkit#9657](https://github.com/rdkit/rdkit/pull/9657).

## Finding 2: the scale has two computable ends, exact to $n=16$ and bounded to $n=50$

![$\log_2 Z$ against $n$. Blue: exhaustive enumeration with `geng`, exact. Orange: local search, lower bounds. Green: an explicit $K_{4,4}$-chain construction. Shaded: the window between the $n$-alkane reference and $Z_{\max}$, which is the range dIpc spans. Lower panel: approximation residuals.](zmax_fit.png)

| $n$ | $Z_{\max}$ | $\log_2 Z/n$ | basis | wall time |
|---|---|---|---|---|
| 12 | 2,921 | 0.95935 | exhaustive, 1,544 graphs | 0.07 s |
| 14 | 11,032 | 0.95924 | exhaustive, 88,168 graphs | 4.3 s |
| 16 | 42,439 | 0.96082 | exhaustive, 8,037,418 graphs | 1,140 s |
| 20 | 598,516 | 0.95957 | local search | 2.1 s / 2k moves |
| 24 | 8,659,688 | 0.96025 | local search | 42 s / 2k moves |
| 30 | $4.69\times10^{8}$ | — | fitted law | — |
| 40 | $3.66\times10^{11}$ | — | fitted law | — |
| 50 | $2.86\times10^{14}$ | — | fitted law | — |

Exhaustive enumeration costs roughly 10–20× per added atom — $n=17$ would take about six hours
and $n=18$ about three days — which is why it stops at 16. The fitted law

$$\log_2 Z_{\max}(n) \;=\; 0.96092\,n - 0.02313$$

holds to $\pm 0.022$ bits over $n = 12\ldots24$. It is bracketed from below by an explicit
construction — a chain of $K_{4,4}$ blocks, each with one internal edge deleted so the connectors
stay within degree 4 — whose exact $Z$ is computable at any size by transfer matrix in 0.17 s,
and which sits a constant $0.22$ bits under the fit all the way to $n=248$. It is bracketed from
above by the $K_{4,4}$ density ceiling, $0.96342\,n$. At $n=50$ those two brackets are $0.25$
bits apart.

The density *peaks* at $K_{4,4}$ ($n=8$) and the sequence has period-8 structure, because a
connected 4-regular graph cannot reach the density of disjoint $K_{4,4}$ blocks: joining two
blocks costs exactly $0.0416$ bits.

## Finding 3: terpenoid space is half as variable as the rest of natural product chemistry

![**Left:** dIpc distributions, 6,121 COCONUT terpenoids against 6,121 non-terpenoids matched one-for-one on heavy-atom count. Dashed lines are group means. **Right:** standard deviation by size band, with the variance ratio annotated.](fig_terpene.png)

| | terpenoid | non-terpenoid (size-matched) | test |
|---|---|---|---|
| n | 6,121 | 6,121 | — |
| mean heavy atoms | 30.72 | 30.72 | matched to $10^{-4}$ |
| mean dIpc | $-0.058$ | $+0.541$ | Cohen's $d = -0.86$ |
| s.d. | 0.562 | 0.803 | Levene $p = 1.4\times10^{-146}$ |
| variance ratio | — | 2.045 | Fligner $p = 2.8\times10^{-146}$ |

**Terpenoid skeletons explore roughly half the topological variance** of size-matched natural
products, and they sit well to the branched side of the chain reference while everything else
sits to the ring-fused side. The effect is not a size artefact: it holds in every stratum from 6
to 60 heavy atoms, with variance ratios of 1.52, 1.95, 1.78, 2.09 and 2.06 (all $p < 0.01$, four
of five below $10^{-22}$).

This is what a biosynthetic constraint should look like. Terpenoids are assembled by head-to-tail
condensation of a single branched C5 unit, so their skeletons cannot explore graph topology
freely; polyketides, alkaloids and shikimates are not built that way and do not show the
restriction.

### The pilot got two of three answers wrong

The curated 43-molecule pilot is reported here because the comparison is instructive:

| claim | curated pilot ($n=43$) | COCONUT ($n=6{,}121$) | verdict |
|---|---|---|---|
| variance ratio | 2.120, $p=0.038$ | 2.045, $p=1.4\times10^{-146}$ | **replicates**, almost exactly |
| mean difference | $-0.141$, $p=0.106$ | $-0.599$, $d=-0.86$ | **null overturned** |
| isoprene slope | $-0.118$/unit, $r=-0.351$ | $-0.046$/unit, $r=-0.110$ | **effect size halved** |

The variance result replicated to within 4% on a sample 140 times larger. The pilot's "means do
not differ" was a **power failure, not a null** — the true effect is large. And the isoprene
trend, which the pilot put at $-0.118$ bits per unit, is $-0.046$ on COCONUT ($p=4.9\times
10^{-18}$, $n=6{,}121$): still real, but the small sample inflated it by a factor of 2.5, and
$r$ fell from $-0.351$ to $-0.110$.

Class means are not monotone in isoprene count on the full data — monoterpenoid $-0.030$
($n=509$), sesquiterpenoid $+0.031$ ($n=1{,}680$), diterpenoid $-0.055$ ($n=1{,}898$),
sesterterpenoid $-0.108$ ($n=200$), triterpenoid $-0.146$ ($n=1{,}834$). Sesquiterpenoids break
the trend, which the pilot's four classes could not have revealed.

## Recommendation

**Drop `Ipc`; keep a repaired `AvgIpc`; add `dIpc`.** Costed:

- *Drop raw `Ipc`* — zero cost, removes a column that is 99.8% molecular size and that goes
  silently wrong above 80 atoms. Any model retrained without it should be unaffected; models
  trained *with* it on molecules under ~60 atoms are also unaffected, since old and new values
  agree to $10^{-15}$ there.
- *Take the upstream fix* — no cost once [rdkit/rdkit#9657](https://github.com/rdkit/rdkit/pull/9657)
  merges; it is also 8× faster. Before then, the patch is three lines.
  **This changes descriptor values above ~60 atoms**, so any model trained on the old numbers
  needs its feature distribution checked, not just a rebuild.
- *Add `dIpc`* — one descriptor, ~10 ms per molecule at $n=50$ using the exact path (0.14 s at
  $n=100$). Buys a signed, size-free, interpretable axis. It adds no information to a flexible
  model already given $\log_2 Z$ and $n$ ($r_{\text{partial}} = +0.0095$), so the case for it is
  interpretability, extrapolation beyond the training size range, and usability in linear or
  small-data settings — not predictive lift.
- *Second-order move*: for a feature matrix, ship $\log_2 Z$ and $n$ as separate columns and let
  the model learn the interaction. Pre-dividing by any power of $n$ discards a degree of freedom
  and, measured, gains nothing (partial $r$ of $-0.087$, $-0.092$, $-0.096$ for $n^{1/3}$,
  $n^{2/3}$, $n^{1}$ once $\log_2 Z$ and nonlinear $n$ terms are present).

## Caveats

- **The isoprene trend is weak.** On COCONUT it is $-0.046$ bits per unit with $r = -0.110$ —
  highly significant at $n=6{,}121$, but explaining about 1% of the variance. The pilot's
  $r=-0.351$ was small-sample inflation. Class means are not monotone: sesquiterpenoids sit
  above monoterpenoids.
- **NPClassifier labels are predictions, not ground truth.** Terpenoid assignment comes from a
  neural classifier applied to COCONUT, not from curated biosynthetic provenance. Its errors are
  unlikely to correlate with dIpc, but the class boundaries are soft — meroterpenoids and
  steroids were excluded as ambiguous, which is a judgement call that was not sensitivity-tested.
- **COCONUT is a literature aggregation.** Version 2022.01.01 is assembled from many sources and
  is known to contain duplicates and structures of varying curation quality. No deduplication was
  applied here beyond taking the largest fragment, so near-identical scaffolds may be
  over-represented in both groups.
- **Flattering limitation:** the pilot reported a *larger* isoprene slope and a cleaner monotone
  class trend than the full data support. Had the pilot been the only study, Finding 3 would
  have read as stronger and tidier than it is.
- **Another flattering one:** the local-search values at $n = 17\ldots24$ are lower bounds. If
  they are slightly low, the fitted law in Finding 2 is a slight *underestimate*, which would
  make it look more accurate than it is rather than less.
- **dIpc adds no information to a flexible model** already given $\log_2 Z$ and $n$ (partial
  $r = +0.0095$). The case for it is interpretability, extrapolation and small-data usability,
  not predictive lift. A report that omitted this would be advocacy.
- **$K_{4,4}$ optimality is taken from the literature, not proved here.** It is exhaustively
  confirmed at $n=8$ and $n=16$ and was checked against $K_5$, the octahedron and circulants,
  but the asymptotic claim rests on results that were not re-derived.
- **The block-and-link model is not an upper bound.** It appeared to be one over $n\le16$ and was
  described as such in working notes; the local search beats it at $n=23$ and $n=24$. The fitted
  law is used instead.
- **dIpc is not new in its ingredients.** The Hosoya index is Hosoya (1971); "branched implies
  small $Z$" is Hosoya's own later result; the path maximising $Z$ among trees is classical
  extremal graph theory. A literature search over OpenAlex found no prior definition of
  $\log Z - \log F(n{+}1)$ as a descriptor, and none for a normalised $Z/Z_{\max}$ — but that is
  not an exhaustive novelty search, and mathematical-chemistry venues are thin in that index.

## Reproducibility

All code, data and figures: **[github.com/guillaume-osmo/kumomlx](https://github.com/guillaume-osmo/kumomlx)**

| artefact | path |
|---|---|
| exact integer characteristic polynomial, dIpc, $Z_{\max}$ | `src/kumomlx/ipc.py` |
| derivation and measurements | `DESCRIPTORS.md` |
| upstream RDKit patch and Eigen C++ port | `rdkit-fix/` |
| $Z_{\max}$ enumeration (C, reads `geng` output) | `rdkit-fix/zmax/zmax.c` |
| bundled boiling-point data, 217 descriptors | `src/kumomlx/data/bp_1000.csv` |
| COCONUT validation of Finding 3 | `paper/coconut.py`, `paper/coconut_stats.py` |

Upstream pull request: [rdkit/rdkit#9657](https://github.com/rdkit/rdkit/pull/9657).
Graph enumeration used `geng` from nauty 2.9.3 (`geng -c -d4 -D4 n`); generated counts match
OEIS A006820, which is the check that the enumeration is complete. RDKit 2025.09.4 throughout. Natural product structures and labels are COCONUT 2022.01.01 with NPClassifier predictions (refs 6–8); the terpenoid/control split, the exact size-matching and every statistic in Finding 3 regenerate from `paper/coconut.py` and `paper/coconut_stats.py`.

Validation performed: the modular-CRT polynomial against exact rational arithmetic; the terpene
set against the isoprene rule; the local search against known exact maxima; the $K_{4,4}$-chain
construction against the exhaustive maximum at $n=16$ (an earlier version of that construction
exceeded it, which revealed it was violating the degree bound).

## References

1. H. Hosoya. Topological Index. A Newly Proposed Quantity Characterizing the Topological Nature
   of Structural Isomers of Saturated Hydrocarbons. *Bull. Chem. Soc. Jpn.* **44**, 2332 (1971).
   [10.1246/bcsj.44.2332](https://doi.org/10.1246/bcsj.44.2332)
2. D. Bonchev, N. Trinajstić. Information theory, distance matrix, and molecular branching.
   *J. Chem. Phys.* **67**, 4517 (1977). [10.1063/1.434593](https://doi.org/10.1063/1.434593)
3. S. H. Bertz. Branching in graphs and molecules. *Discrete Appl. Math.* **19**, 65 (1988).
   [10.1016/0166-218X(88)90006-6](https://doi.org/10.1016/0166-218X(88)90006-6)
4. H. Hosoya. Chemical meaning of octane number analyzed by topological indices (2002) — the
   source of "a high-octane isomer should be highly branched (small $Z$)".
5. B. D. McKay, A. Piperno. Practical graph isomorphism, II. *J. Symb. Comput.* **60**, 94 (2014).
   [10.1016/j.jsc.2013.09.003](https://doi.org/10.1016/j.jsc.2013.09.003) — `nauty` / `geng`.
6. M. Sorokina, P. Merseburger, K. Rajan, M. A. Yirik, C. Steinbeck. COCONUT online: Collection
   of Open Natural Products database. *J. Cheminform.* **13**, 2 (2021).
   [10.1186/s13321-020-00478-9](https://doi.org/10.1186/s13321-020-00478-9)
7. H. W. Kim *et al.* NPClassifier: A Deep Neural Network-Based Structural Classification Tool
   for Natural Products. *J. Nat. Prod.* **84**, 2795 (2021).
   [10.1021/acs.jnatprod.1c00399](https://doi.org/10.1021/acs.jnatprod.1c00399)
8. NPClassifier predictions of COCONUT compounds.
   [10.5281/zenodo.10629838](https://doi.org/10.5281/zenodo.10629838) — the labelled dump used here.
