"""Figure 3: terpene topological variance, on structurally defined terpenes and their skeletons."""
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/"
T = np.load(P + "rule_terp.npy"); C = np.load(P + "rule_ctrl.npy")
tn, td = T[:, 0], T[:, 1]; cn, cd = C[:, 0], C[:, 1]
o = np.isfinite(td); td, tn = td[o], tn[o]
o = np.isfinite(cd); cd, cn = cd[o], cn[o]
vr = cd.var(ddof=1) / td.var(ddof=1)

fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.5))
bins = np.linspace(-1.6, 3.2, 70)
a.hist(cd, bins=bins, density=True, color="#BBBBBB", alpha=0.9,
       label=f"non-terpenoid pathways (n={len(cd):,})\nsd {cd.std(ddof=1):.3f}")
a.hist(td, bins=bins, density=True, histtype="step", lw=2.2, color="#54A24B",
       label=f"terpenes, isoprene rule (n={len(td):,})\nsd {td.std(ddof=1):.3f}")
a.axvline(0, color="black", lw=1.3)
a.annotate(f"variance ratio {vr:.2f}\nCohen's $d$ = $-1.16$",
           xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top", fontsize=9.5,
           bbox=dict(boxstyle="round,pad=0.4", fc="#F3F8F2", ec="#54A24B"))
a.text(0.015, 0.55, "0 = $n$-alkane", transform=a.transAxes, fontsize=8.5, color="#444444")
caro = td[T[o2, 2] == 40] if (o2 := np.isfinite(T[:, 1])).any() else np.array([])
if len(caro):
    a.axvline(caro.mean(), color="#B279A2", lw=1.5, ls=":")
    # label to the RIGHT of the line: to its left is the y-axis
    a.annotate(f"carotenoids C40 (n={len(caro)})\nmean {caro.mean():+.2f}",
               xy=(caro.mean(), 0.72), xycoords=("data", "axes fraction"),
               xytext=(7, 0), textcoords="offset points", ha="left", va="center",
               fontsize=8.5, color="#8A5F96",
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#B279A2", alpha=0.92))
a.set_xlabel("dIpc of the carbon skeleton   (bits)"); a.set_ylabel("density")
a.set_title("Carbon-skeleton dIpc, terpene vs other pathways", fontsize=11)
a.legend(fontsize=8.5, loc="upper left"); a.grid(alpha=0.2)

bands = [(6, 14), (15, 19), (20, 24), (25, 32), (33, 60)]
xs, st, sc = [], [], []
for lo, hi in bands:
    x = td[(tn >= lo) & (tn <= hi)]; y = cd[(cn >= lo) & (cn <= hi)]
    xs.append(f"{lo}–{hi}"); st.append(x.std(ddof=1)); sc.append(y.std(ddof=1))
i = np.arange(len(bands)); w = 0.38
b.bar(i - w/2, sc, w, color="#BBBBBB", label="non-terpenoid")
b.bar(i + w/2, st, w, color="#54A24B", label="terpene")
for k in i:
    b.text(k, max(st[k], sc[k]) + 0.03, f"{sc[k]**2/st[k]**2:.1f}×", ha="center", fontsize=9)
b.set_xticks(i); b.set_xticklabels(xs)
b.set_xlabel("skeleton carbons"); b.set_ylabel("s.d. of dIpc   (bits)")
b.set_title("Variance ratio by skeleton size", fontsize=11)
b.legend(fontsize=9, loc="upper left"); b.grid(alpha=0.2, axis="y")
b.set_ylim(0, max(sc) * 1.25)
fig.tight_layout(); fig.savefig(P + "fig_terpene.png", dpi=165)
print(f"wrote fig_terpene.png  vr={vr:.3f}")
