"""Figure 1: Ipc is the Hosoya index, and RDKit computes it wrongly above ~80 atoms."""
import sys, csv, math, warnings
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/kumomlx-repo/src")
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
from kumomlx.ipc import ipc_terms, charpoly_exact

ns = list(range(10, 161, 10))
rk, ex, bound = [], [], []
for n in ns:
    m = Chem.MolFromSmiles("C" * n)
    rk.append(Descriptors.AvgIpc(m)); ex.append(ipc_terms(m)[0]); bound.append(math.log2(n + 1))

# variance decomposition on real molecules
log2S, H = [], []
for r in list(csv.DictReader(open("/Users/guillaume-osmo/Github/dmpnn-cpp-fp16/examples/bpsubset.csv")))[:400]:
    m = Chem.MolFromSmiles(r["smiles"])
    c = [abs(int(x)) for x in charpoly_exact(np.asarray(Chem.GetAdjacencyMatrix(m), dtype=np.int64)) if x]
    if len(c) < 2: continue
    L = [math.log2(v) for v in c]; mx = max(L)
    ls = mx + math.log2(sum(2.0**(l-mx) for l in L))
    h = sum(2.0**(l-ls)*(ls-l) for l in L)
    if h > 0: log2S.append(ls); H.append(h)
log2S, H = np.array(log2S), np.array(H)
log2Ipc = log2S + np.log2(H)
r2 = np.corrcoef(log2Ipc, log2S)[0, 1] ** 2

fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))

a.plot(ns, bound, color="#999999", ls=":", lw=1.6, label=r"theoretical bound $\log_2(n{+}1)$")
a.plot(ns, ex, "o-", color="#4C78A8", ms=5, lw=1.8, label="exact (integer characteristic polynomial)")
a.plot(ns, rk, "s-", color="#E45756", ms=5, lw=1.8, label="RDKit 2025.09.4 (float64)")
a.axvspan(110, 160, color="#E45756", alpha=0.08)
a.annotate("RDKit collapses:\nan entropy cannot decrease", xy=(120, rk[11]), xytext=(60, 1.0),
           fontsize=9, color="#A33", arrowprops=dict(arrowstyle="->", color="#A33", lw=1.2))
a.set_xlabel("n   (carbon chain length)"); a.set_ylabel(r"AvgIpc   (bits)")
a.set_title("RDKit's AvgIpc stops being monotone above ~110 atoms", fontsize=11)
a.legend(fontsize=8.5, loc="upper left"); a.grid(alpha=0.25)

b.scatter(log2S, log2Ipc, s=12, color="#4C78A8", alpha=0.55, edgecolors="none")
lim = [min(log2S.min(), log2Ipc.min()) - 1, max(log2S.max(), log2Ipc.max()) + 1]
b.plot(lim, lim, color="#999999", ls="--", lw=1.2)
b.set_xlabel(r"$\log_2 S$  ($S$ = Hosoya index, the matching count)")
b.set_ylabel(r"$\log_2 \mathrm{Ipc}$")
b.set_title(f"Ipc is the Hosoya index: $R^2$ = {r2:.4f}", fontsize=11)
b.text(0.04, 0.93, f"n = {len(log2S)} molecules, 3–49 atoms\n"
       r"$\log_2 S$ spans 32.3 bits," "\n" r"$\log_2 H$ only 2.24",
       transform=b.transAxes, fontsize=9, va="top",
       bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC"))
b.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/fig_ipc.png", dpi=165)
print(f"wrote fig_ipc.png   R2={r2:.4f}  n={len(log2S)}")
