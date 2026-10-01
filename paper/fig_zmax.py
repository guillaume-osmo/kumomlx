"""Z_max and its approximations on a log2 axis, with residuals."""
import functools, math
import numpy as np
import networkx as nx
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXACT = {5:26,6:51,7:100,8:209,9:388,10:780,11:1482,12:2921,13:5600,14:11032,15:21482,16:42439}
SEARCH = {17:81725,18:159310,19:306152,20:598516,21:1154562,22:2268536,23:4445701,24:8659688}
BLK = math.log2(209); FIT_C, FIT_B = 0.96092, -0.02313

def bc(G, l, r):
    e = tuple(sorted(tuple(sorted(x)) for x in G.edges()))
    @functools.lru_cache(maxsize=None)
    def rec(es, ban):
        if not es: return 1
        (u, v), rest = es[0], es[1:]
        t = rec(rest, ban)
        if u not in ban and v not in ban:
            t += rec(tuple(x for x in rest if u not in x and v not in x), ban)
        return t
    return rec(e, frozenset()), rec(e, frozenset([l])), rec(e, frozenset([r])), rec(e, frozenset([l, r]))

G = nx.complete_bipartite_graph(4, 4); G.remove_edge(0, 4)
C44 = bc(G, 0, 4)
def chain(q):
    N, Nl, Nr, Nlr = C44; u, m = Nr, N - Nr
    for _ in range(q - 1):
        u, m = u*Nlr + (u+m)*Nr, u*(Nl-Nlr) + (u+m)*(N-Nr)
    return math.log2(u + m)

_f = [0, 1]
def lf(k):
    while len(_f) <= k: _f.append(_f[-1]+_f[-2])
    return math.log2(_f[k])

ne = np.array(sorted(EXACT)); le = np.array([math.log2(EXACT[n]) for n in ne])
ns = np.array(sorted(SEARCH)); ls = np.array([math.log2(SEARCH[n]) for n in ns])
alln = np.concatenate([ne, ns]); alll = np.concatenate([le, ls])
grid = np.arange(5, 51)
fit = FIT_C*grid + FIT_B
ceil = (BLK/8)*grid
chainn = np.array([8*q for q in range(2, 7)]); chainl = np.array([chain(q) for q in range(2, 7)])
alk = np.array([lf(n+1) for n in grid])
extra = np.array([30, 40, 50]); extral = FIT_C*extra + FIT_B

fig, (ax, bx) = plt.subplots(2, 1, figsize=(9.5, 8.6), height_ratios=[2.2, 1], sharex=True)
ax.fill_between(grid, alk, fit, color="#4C78A8", alpha=0.10, label="accessible window (dIpc range)")
ax.plot(grid, ceil, color="#999999", ls=":", lw=1.6, label=r"$K_{4,4}$ density ceiling $0.96342\,n$")
ax.plot(grid, fit, color="#E45756", lw=1.9, label=r"fit  $0.96092\,n-0.02313$")
ax.plot(chainn, chainl, "d-", color="#54A24B", ms=7, lw=1.5, mec="white",
        label=r"valid $K_{4,4}$-chain construction (lower bound)")
ax.plot(grid, alk, color="black", lw=1.6, label=r"$n$-alkane reference $\log_2 F(n{+}1)$")
ax.plot(ne, le, "o", ms=7, color="#4C78A8", mec="white", mew=1.1, zorder=5, label="exhaustive (geng), exact")
ax.plot(ns, ls, "s", ms=7, color="#F58518", mec="white", mew=1.1, zorder=5, label="local search, lower bound")
ax.plot(extra, extral, "*", ms=15, color="#B279A2", mec="white", mew=1.0, zorder=6,
        label="extrapolated (n = 30, 40, 50)")
for n, v in zip(extra, extral):
    ax.annotate(f"$2^{{{v:.1f}}}$", (n, v), textcoords="offset points", xytext=(6, -13),
                fontsize=8.5, color="#7B4F91")
for x, lab in ((16.5, "enumeration ends"), (24.5, "exact $Z$ ends")):
    ax.axvline(x, color="#CCCCCC", lw=1, ls="-.")
    ax.text(x+0.4, 6.5, lab, fontsize=8.5, color="#777777", rotation=90, va="bottom")
ax.set_ylabel(r"$\log_2 Z$   (bits)")
ax.set_title("$Z_{\\max}$ for chemical graphs ($\\Delta \\leq 4$) and its approximations", fontsize=12)
ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95); ax.grid(alpha=0.25)

rf = (FIT_C*alln + FIT_B) - alll
bx.axhline(0, color="black", lw=1)
bx.fill_between([11.5, 24.5], -0.0216, 0.0216, color="#E45756", alpha=0.13,
                label=r"fit residual band $\pm0.022$ bits (n$\geq$12)")
bx.plot(alln, rf, "o-", color="#E45756", ms=5.5, lw=1.4, label="fit $-$ actual")
bx.plot(chainn, (FIT_C*chainn + FIT_B) - chainl, "d--", color="#54A24B", ms=6, lw=1.3,
        label="fit $-$ chain construction")
bx.set_xlabel("n   (heavy atoms)")
bx.set_ylabel("approximation $-$ actual\n(bits)")
bx.legend(fontsize=8.5, loc="upper left", framealpha=0.95); bx.grid(alpha=0.25); bx.set_xlim(4, 51)
fig.tight_layout()
out = "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/zmax_fit.png"
fig.savefig(out, dpi=165)
print("wrote", out)
print(f"fit residual max |.| over 20 measured points: {np.abs(rf).max():.4f} bits")
