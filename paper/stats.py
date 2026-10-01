"""Is terpene topological space NARROWER than general chemical space of the same size?

The means barely differ (p=0.106). The spreads do not look alike, and a narrower spread is the
signature one would expect from a biosynthetic constraint: terpenes are assembled from isoprene
units, so their skeletons cannot explore topology freely.
"""
import json, sys
import numpy as np
from scipy import stats

d = json.load(open("/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/data.json"))
td = np.array([r["dIpc"] for r in d["terpenes"]]); tn = np.array([r["n"] for r in d["terpenes"]])
gd = np.array([r["dIpc"] for r in d["general"]]);  gn = np.array([r["n"] for r in d["general"]])
m = (gn >= tn.min()) & (gn <= tn.max())
g = gd[m]

print(f"terpenes n={len(td)}  sd={td.std(ddof=1):.4f}   general (size-matched) n={len(g)}  "
      f"sd={g.std(ddof=1):.4f}")
print(f"variance ratio (general/terpene): {g.var(ddof=1)/td.var(ddof=1):.3f}")
print(f"  Levene  (robust to non-normality): W={stats.levene(td,g).statistic:.3f}, "
      f"p={stats.levene(td,g).pvalue:.4f}")
print(f"  Fligner (robust, rank-based)     : X2={stats.fligner(td,g).statistic:.3f}, "
      f"p={stats.fligner(td,g).pvalue:.4f}")
print(f"  Bartlett (assumes normality)     : p={stats.bartlett(td,g).pvalue:.4f}")

print(f"\nlocation:")
print(f"  Mann-Whitney p = {stats.mannwhitneyu(td,g).pvalue:.4f}  "
      f"(means {td.mean():+.4f} vs {g.mean():+.4f})")

# size-stratified, since terpene sizes are clustered
print(f"\nsize-stratified (narrow bands, to remove any size confound):")
print(f"  {'band':>12} {'n_terp':>7} {'n_gen':>6} {'sd_terp':>8} {'sd_gen':>7} {'Levene p':>9}")
for lo, hi in ((9, 13), (14, 18), (19, 35)):
    a = td[(tn >= lo) & (tn <= hi)]; b = gd[(gn >= lo) & (gn <= hi)]
    if len(a) < 4 or len(b) < 10: continue
    p = stats.levene(a, b).pvalue
    print(f"  {f'{lo}-{hi} atoms':>12} {len(a):7d} {len(b):6d} {a.std(ddof=1):8.4f} "
          f"{b.std(ddof=1):7.4f} {p:9.4f}")

print(f"\ntrend across terpene classes (isoprene units):")
for cls in ("monoterpene (C10)", "sesquiterpene (C15)", "diterpene (C20)", "triterpene (C30)"):
    v = np.array([r["dIpc"] for r in d["terpenes"] if r["cls"] == cls])
    print(f"  {cls:22s} N={len(v):2d}  mean dIpc {v.mean():+.4f}")
units = {"monoterpene (C10)": 2, "sesquiterpene (C15)": 3, "diterpene (C20)": 4,
         "triterpene (C30)": 6}
x = np.array([units[r["cls"]] for r in d["terpenes"]]); y = td
sl, ic, r, p, se = stats.linregress(x, y)
print(f"  dIpc vs isoprene units: slope {sl:+.4f}/unit, r={r:+.3f}, p={p:.4f}")
