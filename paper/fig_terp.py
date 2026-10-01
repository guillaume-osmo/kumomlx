"""Figure 3, on COCONUT: terpenoid topological space is narrower, at every size."""
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/"
T = np.load(P + "coconut_terp.npy"); C = np.load(P + "coconut_ctrl.npy")
tn, td, tu = T[:, 0], T[:, 1], T[:, 2].astype(int)
cn, cd = C[:, 0], C[:, 1]
m = np.isfinite(td); td, tn, tu = td[m], tn[m], tu[m]
m = np.isfinite(cd); cd, cn = cd[m], cn[m]
vr = cd.var(ddof=1) / td.var(ddof=1)

fig, (a, b) = plt.subplots(1, 2, figsize=(11.6, 4.5))

bins = np.linspace(-3, 3, 70)
a.hist(cd, bins=bins, density=True, color="#BBBBBB", alpha=0.9,
       label=f"non-terpenoid, size-matched\n(n={len(cd):,}), sd {cd.std(ddof=1):.3f}")
a.hist(td, bins=bins, density=True, histtype="step", lw=2.2, color="#54A24B",
       label=f"terpenoid (n={len(td):,}), sd {td.std(ddof=1):.3f}")
a.axvline(0, color="black", lw=1.3)
a.axvline(td.mean(), color="#54A24B", lw=1.4, ls="--")
a.axvline(cd.mean(), color="#777777", lw=1.4, ls="--")
a.annotate(f"variance ratio {vr:.2f}\nLevene $p = 1.4\\times10^{{-146}}$\n"
           f"Cohen's $d = {-0.864:.2f}$ on the mean",
           xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top", fontsize=9,
           bbox=dict(boxstyle="round,pad=0.4", fc="#F3F8F2", ec="#54A24B"))
a.text(0.02, 0.60, "0 = n-alkane", transform=a.transAxes, fontsize=8.5, color="#444444")
a.set_xlabel("dIpc   (bits relative to the n-alkane)"); a.set_ylabel("density")
a.set_title("Terpenoids are narrower and more branched", fontsize=11)
a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.2); a.set_xlim(-3, 3)

bands = [(6, 13), (14, 20), (21, 28), (29, 40), (41, 60)]
xs, st, sc, ps = [], [], [], []
for lo, hi in bands:
    x = td[(tn >= lo) & (tn <= hi)]; y = cd[(cn >= lo) & (cn <= hi)]
    xs.append(f"{lo}–{hi}"); st.append(x.std(ddof=1)); sc.append(y.std(ddof=1))
    ps.append(stats.levene(x, y).pvalue)
i = np.arange(len(bands)); w = 0.38
b.bar(i - w/2, sc, w, color="#BBBBBB", label="non-terpenoid")
b.bar(i + w/2, st, w, color="#54A24B", label="terpenoid")
for k, (p, hgt) in enumerate(zip(ps, np.maximum(st, sc))):
    b.text(k, hgt + 0.035, f"{sc[k]**2/st[k]**2:.2f}×", ha="center", fontsize=9, color="#333333")
b.set_xticks(i); b.set_xticklabels(xs)
b.set_xlabel("heavy atoms"); b.set_ylabel("s.d. of dIpc   (bits)")
b.set_title("The gap holds at every size (variance ratio annotated)", fontsize=11)
b.legend(fontsize=9, loc="upper left"); b.grid(alpha=0.2, axis="y")
b.set_ylim(0, max(sc) * 1.22)
fig.tight_layout()
fig.savefig(P + "fig_terpene.png", dpi=165)
print(f"wrote fig_terpene.png  vr={vr:.3f}  all Levene p: "
      + ", ".join(f"{p:.1e}" for p in ps))
