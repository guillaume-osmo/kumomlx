"""Define a terpene structurally with SMARTS, instead of trusting a classifier.

The isoprene rule says a terpene skeleton is built from C5 units, so a true terpene carbon
framework should (a) have a carbon count that is a multiple of 5 in the terpene range, and
(b) carry the methyl branching those units leave behind -- one pendant methyl per isoprene unit,
modulo cyclisation. Both are computable. This measures how far NPClassifier's labels agree with
that structural definition, in both directions.
"""
import collections, csv, random, sys, warnings
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/kumomlx-repo/src")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")

TSV = "/tmp/coconut_predictions/coconut_predictions.tsv"
TERP = {"Monoterpenoids": 10, "Sesquiterpenoids": 15, "Diterpenoids": 20,
        "Sesterterpenoids": 25, "Triterpenoids": 30,
        "Carotenoids (C40)": 40}
CONTROL = {"Alkaloids", "Shikimates and Phenylpropanoids", "Polyketides",
           "Fatty acids", "Amino acids and Peptides", "Carbohydrates"}

# Structural marks of isoprenoid assembly, as SMARTS on the parent molecule.
SMARTS = {
    "gem_dimethyl":   Chem.MolFromSmarts("[CH3][CX4]([CH3])[#6]"),
    "isopropyl":      Chem.MolFromSmarts("[CH3][CHX4]([CH3])[#6]"),
    "isopropenyl":    Chem.MolFromSmarts("[CH3][CX3](=[CH2])[#6]"),
    "trisub_alkene":  Chem.MolFromSmarts("[CH3][CX3](=[CX3])[#6]"),   # head-to-tail C=C
    "isoprene_C5":    Chem.MolFromSmarts("[CH3][#6]([#6])[#6][#6]"),  # 2-methylbutane frame
}


def skeleton(m):
    em = Chem.RWMol(m)
    for a in sorted((a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() != "C"), reverse=True):
        em.RemoveAtom(a)
    sk = em.GetMol()
    Chem.SanitizeMol(sk, Chem.SanitizeFlags.SANITIZE_ALL ^
                     Chem.SanitizeFlags.SANITIZE_PROPERTIES, catchErrors=True)
    frs = Chem.GetMolFrags(sk, asMols=True, sanitizeFrags=False)
    return max(frs, key=lambda f: f.GetNumAtoms()) if frs else None


def features(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    frs = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    m = max(frs, key=lambda f: f.GetNumAtoms()) if len(frs) > 1 else m
    sk = skeleton(m)
    if sk is None or sk.GetNumAtoms() < 5: return None
    nC = sk.GetNumAtoms()
    # pendant methyls of the BACKBONE = degree-1 carbons in the carbon framework
    pend = sum(1 for a in sk.GetAtoms() if a.GetDegree() == 1)
    hits = {k: len(m.GetSubstructMatches(q)) for k, q in SMARTS.items()}
    return nC, pend, hits


rng = random.Random(0)
groups = {"terpenoid": [], "control": []}
with open(TSV) as fh:
    for r in csv.DictReader(fh, delimiter="\t"):
        p, s, smi = (r["prediction_pathway"].strip(), r["prediction_superclass"].strip(),
                     r["smiles"].strip())
        if not smi: continue
        if p == "Terpenoids" and s in TERP: groups["terpenoid"].append(smi)
        elif p in CONTROL: groups["control"].append(smi)

data = {}
for g, smis in groups.items():
    rng.shuffle(smis)
    rows = [f for smi in smis[:6000] if (f := features(smi))]
    data[g] = rows
    nC = np.array([r[0] for r in rows]); pend = np.array([r[1] for r in rows])
    print(f"{g:10s} n={len(rows):5d}   C%5==0: {np.mean(nC % 5 == 0):6.1%}   "
          f"C in terpene set: {np.mean(np.isin(nC, [10,15,20,25,30,40])):6.1%}   "
          f"median pendant methyls {int(np.median(pend))}")

print(f"\n{'SMARTS':16s} {'terpenoid hit':>14} {'control hit':>12} {'enrichment':>11}")
for k in SMARTS:
    a = np.mean([r[2][k] > 0 for r in data["terpenoid"]])
    b = np.mean([r[2][k] > 0 for r in data["control"]])
    print(f"{k:16s} {a:14.1%} {b:12.1%} {a/max(b,1e-9):11.2f}x")

print(f"\nSTRICT STRUCTURAL RULE: skeleton C in {{10,15,20,25,30,40}} AND pendant methyls >= C/5 - 1")
for g in ("terpenoid", "control"):
    rows = data[g]
    ok = [r for r in rows if r[0] in (10,15,20,25,30,40) and r[1] >= r[0]//5 - 1]
    print(f"  {g:10s} {len(ok):5d} / {len(rows):5d} = {len(ok)/len(rows):6.1%}")
