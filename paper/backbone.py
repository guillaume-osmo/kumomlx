"""Validate NPClassifier terpenoid labels against the carbon skeleton, and recompute dIpc on it.

Terpene identity is a property of the CARBON BACKBONE: isoprene units condensed into C10, C15,
C20, C25, C30 or C40 skeletons. Enzymes then decorate -- oxidation, epoxidation, acetylation,
glycosylation -- and can also remove carbons (cholesterol is a C27 degraded triterpene). So the
label can be checked against chemistry rather than taken from a classifier, and dIpc can be
computed on the skeleton, which is what biosynthesis actually constrains.
"""
import collections, csv, random, sys, warnings
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/kumomlx-repo/src")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")

TSV = "/tmp/coconut_predictions/coconut_predictions.tsv"
EXPECT = {"Monoterpenoids": 10, "Sesquiterpenoids": 15, "Diterpenoids": 20,
          "Sesterterpenoids": 25, "Triterpenoids": 30,
        "Carotenoids (C40)": 40}


def carbon_skeleton(mol):
    """The connected carbon framework: delete every non-carbon atom, keep the largest fragment."""
    em = Chem.RWMol(mol)
    for a in sorted((a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != "C"), reverse=True):
        em.RemoveAtom(a)
    sk = em.GetMol()
    try:
        Chem.SanitizeMol(sk, Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    except Exception:
        return None
    frs = Chem.GetMolFrags(sk, asMols=True, sanitizeFrags=False)
    if not frs: return None
    return max(frs, key=lambda f: f.GetNumAtoms())


rows = collections.defaultdict(list)
with open(TSV) as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        if r["prediction_pathway"].strip() == "Terpenoids":
            s = r["prediction_superclass"].strip()
            if s in EXPECT and r["smiles"].strip():
                rows[s].append(r["smiles"].strip())

rng = random.Random(0)
print(f"{'class':18s} {'expected C':>10} {'n':>6} {'exact':>7} {'within 1':>9} {'within 3':>9} "
      f"{'median C':>9} {'median decor':>13}")
keep = collections.defaultdict(list)
for s, smis in rows.items():
    rng.shuffle(smis)
    cs, dec = [], []
    for smi in smis[:2500]:
        m = Chem.MolFromSmiles(smi)
        if m is None: continue
        frs = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
        m = max(frs, key=lambda f: f.GetNumAtoms()) if len(frs) > 1 else m
        sk = carbon_skeleton(m)
        if sk is None or sk.GetNumAtoms() < 5: continue
        nC = sk.GetNumAtoms()
        cs.append(nC)
        dec.append(m.GetNumAtoms() - nC)          # heavy atoms outside the carbon framework
        keep[s].append((smi, nC, m.GetNumAtoms() - nC))
    cs = np.array(cs); e = EXPECT[s]
    print(f"{s:18s} {e:10d} {len(cs):6d} {np.mean(cs==e):7.1%} {np.mean(abs(cs-e)<=1):9.1%} "
          f"{np.mean(abs(cs-e)<=3):9.1%} {int(np.median(cs)):9d} {int(np.median(dec)):13d}")

allc = np.array([c for v in keep.values() for _, c, _ in v])
alle = np.array([EXPECT[s] for s, v in keep.items() for _ in v])
alld = np.array([d for v in keep.values() for _, _, d in v])
print(f"\noverall: {np.mean(allc==alle):.1%} exact, {np.mean(abs(allc-alle)<=1):.1%} within 1, "
      f"{np.mean(abs(allc-alle)<=3):.1%} within 3   (n={len(allc):,})")
print(f"decoration: median {int(np.median(alld))} non-carbon heavy atoms per molecule, "
      f"{np.mean(alld==0):.1%} undecorated, 90th pct {int(np.percentile(alld,90))}")
np.save(f"{sys.path[0]}/../paper/_skel_check.npy", np.c_[allc, alle, alld])
