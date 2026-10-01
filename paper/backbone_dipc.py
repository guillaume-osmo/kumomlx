"""dIpc on the carbon skeleton: does the terpenoid signal survive stripping the decoration?

dIpc on the whole molecule mixes the biosynthetic backbone with enzymatic decoration. Since
biosynthesis constrains the skeleton, the test is repeated on the carbon framework alone, with
controls matched on SKELETON size rather than molecule size.
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
        "Sesterterpenoids": 25, "Triterpenoids": 30}
CONTROL = {"Alkaloids", "Shikimates and Phenylpropanoids", "Polyketides",
           "Fatty acids", "Amino acids and Peptides", "Carbohydrates"}
MIN_C, MAX_C = 6, 60


def skeleton(smi):
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
    n = sk.GetNumAtoms()
    return (sk, n) if MIN_C <= n <= MAX_C else None


terp, ctrl = collections.defaultdict(list), []
with open(TSV) as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        p, s, smi = (r["prediction_pathway"].strip(), r["prediction_superclass"].strip(),
                     r["smiles"].strip())
        if not smi: continue
        if p == "Terpenoids" and s in TERP: terp[s].append(smi)
        elif p in CONTROL: ctrl.append(smi)

rng = random.Random(0); t0 = time.time()
per = {k: max(200, int(6000 * len(v) / sum(len(x) for x in terp.values()))) for k, v in terp.items()}
T = []
for s, smis in terp.items():
    rng.shuffle(smis); got = 0
    for smi in smis:
        if got >= per[s]: break
        sk = skeleton(smi)
        if sk is None: continue
        T.append((sk[1], dIpc(sk[0]), TERP[s])); got += 1
    print(f"  {s:18s} {got:5d}  ({time.time()-t0:.0f}s)", flush=True)

need = collections.Counter(n for n, _, _ in T); have = collections.Counter(); C = []
rng.shuffle(ctrl)
for smi in ctrl:
    if sum((need - have).values()) == 0: break
    sk = skeleton(smi)
    if sk is None or have[sk[1]] >= need[sk[1]]: continue
    C.append((sk[1], dIpc(sk[0]))); have[sk[1]] += 1
print(f"controls matched on SKELETON size: {len(C):,}/{len(T):,}  ({time.time()-t0:.0f}s)")

T = np.array(T); C = np.array(C)
tn, td = T[:, 0], T[:, 1]; cn, cd = C[:, 0], C[:, 1]
ok = np.isfinite(td); td, tn = td[ok], tn[ok]
ok = np.isfinite(cd); cd, cn = cd[ok], cn[ok]
np.save(f"{S}/paper/skel_terp.npy" if (S := "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad") else "", np.c_[tn, td, T[ok, 2] if len(T)==len(ok) else tn*0])
np.save(f"{S}/paper/skel_ctrl.npy", np.c_[cn, cd])

print(f"\nCARBON SKELETON  (n={len(td):,} vs {len(cd):,}, matched on skeleton carbons)")
print(f"  mean skeleton C: terpenoid {tn.mean():.2f}  control {cn.mean():.2f}")
print(f"  mean dIpc:  terpenoid {td.mean():+.4f}   control {cd.mean():+.4f}")
print(f"  s.d.:       terpenoid {td.std(ddof=1):.4f}   control {cd.std(ddof=1):.4f}")
vr = cd.var(ddof=1)/td.var(ddof=1)
pooled = np.sqrt((td.var(ddof=1)+cd.var(ddof=1))/2)
print(f"  variance ratio {vr:.3f}   Levene p={stats.levene(td,cd).pvalue:.2e}")
print(f"  Cohen's d {(td.mean()-cd.mean())/pooled:+.4f}   Mann-Whitney p={stats.mannwhitneyu(td,cd).pvalue:.2e}")
print(f"\n  [whole molecule, for comparison: variance ratio 2.045, Cohen's d -0.86]")
