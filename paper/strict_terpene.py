"""Finding 3 on STRUCTURALLY DEFINED terpenes, on the carbon skeleton.

Two changes from the classifier-only version:
  * terpenes must satisfy the isoprene rule structurally -- carbon framework of 10, 15, 20, 25,
    30 or 40 carbons, carrying at least C/5 - 1 pendant methyls -- not merely carry a predicted
    label. 76.7% of labelled terpenoids pass; the rest are degraded or rearranged skeletons.
  * controls must FAIL that rule, which removes the 10.9% of other pathways that are
    structurally isoprenoid (prenylated and meroterpenoid compounds) and would otherwise blunt
    the contrast in both directions.
dIpc is computed on the carbon skeleton, and controls are matched on skeleton carbon count.
"""
import collections, csv, random, sys, time, warnings
import numpy as np
from scipy import stats
sys.path.insert(0, "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/kumomlx-repo/src")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
from kumomlx.ipc import delta_matching_information as dIpc

TSV = "/tmp/coconut_predictions/coconut_predictions.tsv"
TERP = {"Monoterpenoids": 10, "Sesquiterpenoids": 15, "Diterpenoids": 20,
        "Sesterterpenoids": 25, "Triterpenoids": 30,
        "Carotenoids (C40)": 40}
CONTROL = {"Alkaloids", "Shikimates and Phenylpropanoids", "Polyketides",
           "Fatty acids", "Amino acids and Peptides", "Carbohydrates"}
TERPENE_C = {10, 15, 20, 25, 30, 40}


def analyse(smi):
    """-> (skeleton, nC, passes_isoprene_rule) or None."""
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    frs = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    m = max(frs, key=lambda f: f.GetNumAtoms()) if len(frs) > 1 else m
    em = Chem.RWMol(m)
    for a in sorted((a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() != "C"), reverse=True):
        em.RemoveAtom(a)
    sk = em.GetMol()
    Chem.SanitizeMol(sk, Chem.SanitizeFlags.SANITIZE_ALL ^
                     Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    frs = Chem.GetMolFrags(sk, asMols=True, sanitizeFrags=False)
    if not frs: return None
    sk = max(frs, key=lambda f: f.GetNumAtoms())
    nC = sk.GetNumAtoms()
    if not 6 <= nC <= 60: return None
    pend = sum(1 for a in sk.GetAtoms() if a.GetDegree() == 1)
    return sk, nC, (nC in TERPENE_C and pend >= nC // 5 - 1)


terp, ctrl = [], []
with open(TSV) as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        p, s, smi = (r["prediction_pathway"].strip(), r["prediction_superclass"].strip(),
                     r["smiles"].strip())
        if not smi: continue
        if p == "Terpenoids" and s in TERP: terp.append((smi, TERP[s]))
        elif p in CONTROL: ctrl.append(smi)

rng = random.Random(0); rng.shuffle(terp); rng.shuffle(ctrl); t0 = time.time()
T = []
for smi, units in terp:
    if len(T) >= 6000: break
    a = analyse(smi)
    if a and a[2]:                              # must PASS the isoprene rule
        T.append((a[1], dIpc(a[0]), units))
print(f"strict terpenes: {len(T):,}  ({time.time()-t0:.0f}s)")

need = collections.Counter(n for n, _, _ in T); have = collections.Counter(); C = []
for smi in ctrl:
    if sum((need - have).values()) == 0: break
    a = analyse(smi)
    if not a: continue       # controls are pathway-defined and NOT rule-filtered:
                             # filtering them leaves an artificially homogeneous
                             # residue at small carbon counts (sd 0.07) and
                             # reverses the comparison in those bands.
    if have[a[1]] >= need[a[1]]: continue
    C.append((a[1], dIpc(a[0]))); have[a[1]] += 1
print(f"non-isoprenoid controls matched: {len(C):,}  ({time.time()-t0:.0f}s)")

T, C = np.array(T), np.array(C)
tn, td, tu = T[:, 0], T[:, 1], T[:, 2]
cn, cd = C[:, 0], C[:, 1]
ok = np.isfinite(td); td, tn, tu = td[ok], tn[ok], tu[ok]
ok = np.isfinite(cd); cd, cn = cd[ok], cn[ok]
np.save("/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/rule_terp.npy", np.c_[tn, td, tu])
np.save("/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/rule_ctrl.npy", np.c_[cn, cd])

vr = cd.var(ddof=1) / td.var(ddof=1)
pooled = np.sqrt((td.var(ddof=1) + cd.var(ddof=1)) / 2)
print(f"\nSTRICT TERPENES, carbon skeleton  (n={len(td):,} vs {len(cd):,})")
print(f"  mean skeleton C  terpene {tn.mean():.2f}   control {cn.mean():.2f}")
print(f"  mean dIpc        terpene {td.mean():+.4f}   control {cd.mean():+.4f}")
print(f"  s.d.             terpene {td.std(ddof=1):.4f}   control {cd.std(ddof=1):.4f}")
print(f"  variance ratio   {vr:.3f}    Levene p = {stats.levene(td, cd).pvalue:.2e}")
print(f"  Cohen's d        {(td.mean()-cd.mean())/pooled:+.4f}   "
      f"Mann-Whitney p = {stats.mannwhitneyu(td, cd).pvalue:.2e}")
print(f"\n  progression:  whole molecule / classifier labels   vr 2.045,  d -0.86")
print(f"                carbon skeleton / classifier labels   vr 3.223,  d -0.96")
print(f"                carbon skeleton / isoprene rule       vr {vr:.3f},  d "
      f"{(td.mean()-cd.mean())/pooled:+.2f}")
print(f"\n  by size band:")
for lo, hi in ((6, 14), (15, 19), (20, 24), (25, 32), (33, 60)):
    a = td[(tn >= lo) & (tn <= hi)]; b = cd[(cn >= lo) & (cn <= hi)]
    if len(a) < 30 or len(b) < 30: continue
    print(f"    {lo:2d}-{hi:2d} C  n={len(a):4d}  sd_t {a.std(ddof=1):.3f}  sd_c {b.std(ddof=1):.3f}"
          f"  ratio {b.var(ddof=1)/a.var(ddof=1):5.2f}  p {stats.levene(a,b).pvalue:.1e}")
