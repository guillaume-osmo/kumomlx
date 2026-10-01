"""Does the curated Finding 3 survive on COCONUT?"""
import numpy as np
from scipy import stats

P = "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/"
T = np.load(P + "coconut_terp.npy"); C = np.load(P + "coconut_ctrl.npy")
tn, td, tu = T[:, 0], T[:, 1], T[:, 2].astype(int)
cn, cd = C[:, 0], C[:, 1]
ok = np.isfinite(td); td, tn, tu = td[ok], tn[ok], tu[ok]
ok = np.isfinite(cd); cd, cn = cd[ok], cn[ok]

print(f"terpenoids n={len(td):,}   controls n={len(cd):,}   (exactly size-matched)")
print(f"size: terpenoid mean {tn.mean():.2f}, control mean {cn.mean():.2f} heavy atoms "
      f"-> matched to {abs(tn.mean()-cn.mean()):.4f}\n")

print("SPREAD  (the curated claim: terpene space is narrower)")
print(f"  terpenoid sd {td.std(ddof=1):.4f}   control sd {cd.std(ddof=1):.4f}")
vr = cd.var(ddof=1) / td.var(ddof=1)
print(f"  variance ratio (control/terpenoid) = {vr:.3f}   "
      f"[curated value was 2.120]")
print(f"  Levene   W={stats.levene(td, cd).statistic:.2f}  p={stats.levene(td, cd).pvalue:.3e}")
print(f"  Fligner  X2={stats.fligner(td, cd).statistic:.2f}  p={stats.fligner(td, cd).pvalue:.3e}")

print("\nLOCATION  (curated: not significant, p=0.106)")
u = stats.mannwhitneyu(td, cd)
print(f"  mean dIpc terpenoid {td.mean():+.4f}   control {cd.mean():+.4f}   "
      f"difference {td.mean()-cd.mean():+.4f}")
print(f"  Mann-Whitney U={u.statistic:.3e}  p={u.pvalue:.3e}")
# effect size, since n is large enough that anything will be 'significant'
pooled = np.sqrt((td.var(ddof=1) + cd.var(ddof=1)) / 2)
print(f"  Cohen's d = {(td.mean()-cd.mean())/pooled:+.4f}  (|d|<0.2 is negligible)")

print("\nISOPRENE TREND  (curated: -0.118 bits/unit, p=0.021)")
sl, ic, r, p, se = stats.linregress(tu, td)
print(f"  slope {sl:+.4f} +/- {se:.4f} bits per isoprene unit   r={r:+.4f}  p={p:.3e}")
names = {2: "Monoterpenoid", 3: "Sesquiterpenoid", 4: "Diterpenoid",
         5: "Sesterterpenoid", 6: "Triterpenoid"}
for u_ in sorted(set(tu)):
    v = td[tu == u_]
    print(f"    {names[u_]:17s} n={len(v):5d}  mean {v.mean():+.4f}  sd {v.std(ddof=1):.4f}")

print("\nSIZE-STRATIFIED spread (curated: carried only by the 9-13 band)")
print(f"  {'band':>12} {'n_terp':>7} {'n_ctrl':>7} {'sd_t':>7} {'sd_c':>7} {'ratio':>6} {'Levene p':>11}")
for lo, hi in ((6, 13), (14, 20), (21, 28), (29, 40), (41, 60)):
    a = td[(tn >= lo) & (tn <= hi)]; b = cd[(cn >= lo) & (cn <= hi)]
    if len(a) < 30 or len(b) < 30: continue
    print(f"  {f'{lo}-{hi}':>12} {len(a):7d} {len(b):7d} {a.std(ddof=1):7.4f} "
          f"{b.std(ddof=1):7.4f} {b.var(ddof=1)/a.var(ddof=1):6.2f} "
          f"{stats.levene(a,b).pvalue:11.2e}")
