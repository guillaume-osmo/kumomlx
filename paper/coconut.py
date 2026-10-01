"""Validate Finding 3 on COCONUT: is terpenoid topological space really narrower?

The curated test used 43 hand-assembled molecules, which is why the paper called it the weakest
finding. This repeats it on COCONUT 2022.01.01 with NPClassifier pathway labels (Zenodo
10.5281/zenodo.10629838), ~102k terpenoids against ~245k non-terpenoids.

Design decisions that matter:
  * terpenoids are pathway == "Terpenoids" exactly; the 4,706 mixed "Alkaloids,Terpenoids" are
    excluded rather than assigned;
  * the control is EXACTLY size-matched -- for each sampled terpenoid, a non-terpenoid with the
    same heavy-atom count. A range restriction would leave a size gradient inside the window,
    and dIpc's spread grows with size, which would manufacture the effect being tested;
  * the largest fragment is taken (COCONUT contains salts), since Z is multiplicative over
    components and a counter-ion would read as a huge negative dIpc.
"""
import collections, csv, math, random, sys, time, warnings
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/kumomlx-repo/src")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*"); warnings.filterwarnings("ignore")
from kumomlx.ipc import delta_matching_information as dIpc

TSV = "/tmp/coconut_predictions/coconut_predictions.tsv"
TERP_SUPER = {"Monoterpenoids": 2, "Sesquiterpenoids": 3, "Diterpenoids": 4,
              "Sesterterpenoids": 5, "Triterpenoids": 6}
CONTROL = {"Alkaloids", "Shikimates and Phenylpropanoids", "Polyketides",
           "Fatty acids", "Amino acids and Peptides", "Carbohydrates"}
MIN_N, MAX_N = 6, 60
N_PER_GROUP = int(sys.argv[1]) if len(sys.argv) > 1 else 6000

terp, ctrl = collections.defaultdict(list), collections.defaultdict(list)
with open(TSV) as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        p, s = row["prediction_pathway"].strip(), row["prediction_superclass"].strip()
        smi = row["smiles"].strip()
        if not smi: continue
        if p == "Terpenoids" and s in TERP_SUPER:
            terp[s].append(smi)
        elif p in CONTROL:
            ctrl["x"].append(smi)
print(f"pool: {sum(len(v) for v in terp.values()):,} classified terpenoids, "
      f"{len(ctrl['x']):,} single-pathway controls")

rng = random.Random(0)

def prep(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    m = max(frags, key=lambda f: f.GetNumAtoms()) if len(frags) > 1 else m
    n = m.GetNumAtoms()
    return (m, n) if MIN_N <= n <= MAX_N else None

# stratified sample of terpenoids, proportional to the real class sizes
per = {k: max(200, int(N_PER_GROUP * len(v) / sum(len(x) for x in terp.values())))
       for k, v in terp.items()}
t0 = time.time()
trows = []
for s, smis in terp.items():
    rng.shuffle(smis)
    got = 0
    for smi in smis:
        if got >= per[s]: break
        pm = prep(smi)
        if pm is None: continue
        m, n = pm
        trows.append({"n": n, "d": dIpc(m), "units": TERP_SUPER[s], "cls": s})
        got += 1
    print(f"  {s:20s} {got:5d} sampled   ({time.time()-t0:.0f}s)", flush=True)

# EXACT size matching: one control per terpenoid at the same heavy-atom count
need = collections.Counter(r["n"] for r in trows)
rng.shuffle(ctrl["x"])
crows, have = [], collections.Counter()
for smi in ctrl["x"]:
    if sum((need - have).values()) == 0: break
    pm = prep(smi)
    if pm is None: continue
    m, n = pm
    if have[n] >= need[n]: continue
    crows.append({"n": n, "d": dIpc(m)})
    have[n] += 1
print(f"controls matched: {len(crows):,} of {len(trows):,} required "
      f"({time.time()-t0:.0f}s total)")

np.save("/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/coconut_terp.npy",
        np.array([(r["n"], r["d"], r["units"]) for r in trows]))
np.save("/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper/coconut_ctrl.npy",
        np.array([(r["n"], r["d"]) for r in crows]))
print("saved arrays")
