"""Figure 3: terpene topological space is narrower than general chemistry."""
import json, math
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/data.json"))
td = np.array([r["dIpc"] for r in d["terpenes"]]); tn = np.array([r["n"] for r in d["terpenes"]])
gd = np.array([r["dIpc"] for r in d["general"]]);  gn = np.array([r["n"] for r in d["general"]])
m = (gn >= tn.min()) & (gn <= tn.max()); g = gd[m]
UNITS = {"monoterpene (C10)": 2, "sesquiterpene (C15)": 3, "diterpene (C20)": 4, "triterpene (C30)": 6}

fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4))

bins = np.linspace(-1.6, 1.6, 33)
a.hist(g, bins=bins, density=True, color="#BBBBBB", alpha=0.85,
       label=f"general, size-matched (n={len(g)}), sd {g.std(ddof=1):.3f}")
a.hist(td, bins=bins, density=True, histtype="step", lw=2.2, color="#54A24B",
       label=f"terpenes (n={len(td)}), sd {td.std(ddof=1):.3f}")
a.axvline(0, color="black", lw=1.4)
a.text(0.02, 0.97, "0 = n-alkane\n(unbranched chain)", transform=a.transAxes, fontsize=8.5,
       va="top", color="#444444")
a.annotate(f"variance ratio 2.12\nLevene p = 0.038", xy=(0.97, 0.80), xycoords="axes fraction",
           ha="right", fontsize=9.5,
           bbox=dict(boxstyle="round,pad=0.4", fc="#F3F8F2", ec="#54A24B"))
a.set_xlabel("dIpc   (bits relative to the n-alkane)"); a.set_ylabel("density")
a.set_title("Terpene topological variance is less than half", fontsize=11)
a.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.80)); a.grid(alpha=0.2)

x = np.array([UNITS[r["cls"]] for r in d["terpenes"]])
jitter = x + np.random.default_rng(0).normal(0, 0.055, len(x))
colors = {2: "#4C78A8", 3: "#54A24B", 4: "#F58518", 6: "#B279A2"}
for u in sorted(set(x)):
    s = x == u
    b.scatter(jitter[s], td[s], s=46, color=colors[u], alpha=0.85, edgecolors="white", lw=0.8)
sl, ic, r, p, se = stats.linregress(x, td)
xs = np.linspace(1.7, 6.3, 50)
b.plot(xs, sl * xs + ic, color="#E45756", lw=2)
b.fill_between(xs, (sl - se) * xs + ic, (sl + se) * xs + ic, color="#E45756", alpha=0.15)
b.axhline(0, color="black", lw=1.2, ls="--")
# class labels INSIDE the axes, under the points, so they cannot collide with the title
ylo = td.min() - 0.30
for u, lab in ((2, "mono C10"), (3, "sesqui C15"), (4, "di C20"), (6, "tri C30")):
    b.text(u, ylo, lab, ha="center", fontsize=8.5, color=colors[u])
b.set_ylim(ylo - 0.10, td.max() + 0.46)
b.annotate(f"slope {sl:+.3f} bits per isoprene unit\n"
           f"r = {r:+.3f},  p = {p:.3f}   (band: fit $\\pm$1 s.e.)",
           xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top", fontsize=9.5,
           bbox=dict(boxstyle="round,pad=0.4", fc="#FDF3F3", ec="#E45756"))
b.set_xlabel("isoprene units"); b.set_ylabel("dIpc   (bits)")
b.set_title("More isoprene units, more branching", fontsize=11)
b.set_xticks([2, 3, 4, 6]); b.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/fig_terpene.png", dpi=165)
print(f"wrote fig_terpene.png  terpenes n={len(td)} sd={td.std(ddof=1):.4f}, "
      f"general n={len(g)} sd={g.std(ddof=1):.4f}, slope={sl:+.4f} p={p:.4f}")
