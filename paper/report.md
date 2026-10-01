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
\textbf{Application.} Terpene topological space is narrower than size-matched general chemistry
(variance ratio $2.12$, Levene $p=0.038$), and $\mathrm{dIpc}$ falls by $0.118$ bits per isoprene
unit ($r=-0.351$, $p=0.021$). Mean position does \emph{not} differ ($p=0.106$).
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
| curated terpenes | 43 (of 46 proposed) | 10–30 carbons | Finding 3 |
| connected 4-regular graphs, `geng` | 8,037,418 at $n{=}16$ | $n = 5\ldots16$ | $Z_{\max}$, exact |
| local-search runs | 8 sizes × 3–16 restarts | $n = 17\ldots24$ | $Z_{\max}$, lower bound |

Terpenes were curated rather than drawn from COCONUT, which was not available locally. Every
entry was checked against the carbon count its class requires — C10 monoterpene, C15
sesquiterpene, C20 diterpene, C30 triterpene. **Three of 46 failed and were dropped**
(fenchone at 9 C, alpha-humulene and valencene at 14 C), then corrected and re-validated; the check
exists because a mistranscribed SMILES almost always has the wrong carbon count.

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

## Finding 3: terpene topological space is narrower, and branches with each isoprene unit

![**Left:** dIpc distributions, 43 curated terpenes against 609 size-matched general molecules (9–35 heavy atoms). **Right:** dIpc against isoprene unit count; band is the fit $\pm 1$ s.e.](fig_terpene.png)

| comparison | terpenes | general (size-matched) | test |
|---|---|---|---|
| n | 43 | 609 | — |
| mean dIpc | $-0.115$ | $+0.026$ | Mann–Whitney $p = 0.106$ |
| s.d. | 0.378 | 0.550 | Levene $p = 0.038$ |
| variance ratio | — | 2.120 | Fligner $p = 0.055$ |

The **means do not differ**. What differs is the spread: terpene dIpc has less than half the
variance of size-matched general chemistry. Stratifying by size to remove any residual size
confound, the effect is carried by the smallest band — 9–13 atoms, $p = 0.028$ ($n=26$ terpenes
against 442 general) — while the 14–18 and 19–35 bands are not significant ($p = 0.24$, $p =
0.52$) with 10 and 7 terpenes respectively.

Within terpenes, dIpc falls by $0.118 \pm 0.049$ bits per isoprene unit ($r = -0.351$,
$p = 0.021$, $n=43$). That is chemically what one expects: isoprene is a branched C5 unit, so
each addition contributes methyl branching, and branching reduces the matching count. Class means
run monotonically from $-0.012$ (monoterpene, $n=26$) through $-0.172$ (sesquiterpene, $n=10$) to
$-0.462$ (diterpene, $n=4$), with triterpenes at $-0.357$ ($n=3$) breaking the monotonicity on
three molecules.

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

- **The terpene result is the weakest finding here.** 43 curated molecules, not a database draw.
  The variance effect is marginal ($p=0.038$ Levene, $p=0.055$ Fligner) and survives in only one
  of three size strata. It should be replicated on COCONUT before being relied on.
- **Flattering limitation:** had the terpene set been larger, the *mean* difference ($-0.141$
  bits, $p=0.106$) might well have reached significance, which would have made the result look
  stronger and cleaner than "the spread differs but the location does not".
- **Another flattering one:** the local search values at $n=17\ldots24$ are lower bounds. If they
  are slightly low, the true $Z_{\max}$ curve is slightly steeper, and the fitted law — which is
  the headline of Finding 2 — would be *more* accurate as an underestimate, not less.
- **dIpc is not new in its ingredients.** The Hosoya index is Hosoya 1971; "branched implies small
  $Z$" is Hosoya's own 2002 result; the path maximising $Z$ among trees is classical extremal
  graph theory. Seven OpenAlex queries found no prior definition of $\log Z - \log F(n{+}1)$ as a
  descriptor, and none for a normalised $Z/Z_{\max}$ — but that is not an exhaustive novelty
  search, and mathematical-chemistry venues are thin in that index.
- **$K_{4,4}$ optimality is taken from the literature, not proved here.** It was verified against
  $K_5$, the octahedron and circulants, and is exhaustively confirmed at $n = 8$ and $n = 16$,
  but the asymptotic claim rests on Friedland-type results that were not re-derived.
- **The block-and-link model is not an upper bound.** It appeared to be one over $n\le16$ and
  was described as such in working notes; the search beats it at $n=23$ and $n=24$. The fitted
  law is used instead.

## Reproducibility

All code, data and figures: **[github.com/guillaume-osmo/kumomlx](https://github.com/guillaume-osmo/kumomlx)**

| artefact | path |
|---|---|
| exact integer characteristic polynomial, dIpc, $Z_{\max}$ | `src/kumomlx/ipc.py` |
| derivation and measurements | `DESCRIPTORS.md` |
| upstream RDKit patch and Eigen C++ port | `rdkit-fix/` |
| $Z_{\max}$ enumeration (C, reads `geng` output) | `rdkit-fix/zmax/zmax.c` |
| bundled boiling-point data, 217 descriptors | `src/kumomlx/data/bp_1000.csv` |

Upstream pull request: [rdkit/rdkit#9657](https://github.com/rdkit/rdkit/pull/9657).
Graph enumeration used `geng` from nauty 2.9.3 (`geng -c -d4 -D4 n`); generated counts match
OEIS A006820, which is the check that the enumeration is complete. RDKit 2025.09.4 throughout.

Validation performed: the modular-CRT polynomial against exact rational arithmetic; the terpene
set against the isoprene rule; the local search against known exact maxima; the $K_{4,4}$-chain
construction against the exhaustive maximum at $n=16$ (an earlier version of that construction
exceeded it, which revealed it was violating the degree bound).
