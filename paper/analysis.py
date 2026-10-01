"""Where terpenes sit on the matching-information scale, against general molecules."""
import sys, csv, math, json
import numpy as np
sys.path.insert(0, "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/kumomlx-repo/src")
sys.path.insert(0, "/private/tmp/claude-501/-Users-guillaume-osmo-Github-osmo-main/d61ca1be-f3e0-442e-80ae-010a819887cd/scratchpad/paper")
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
from kumomlx.ipc import delta_matching_information as dIpc, matching_information as logZ
from terpenes import validated

ok, bad = validated()
rows = []
for cls, members in ok.items():
    for name, (smi, m) in members.items():
        rows.append({"cls": cls, "name": name, "smiles": smi, "n": m.GetNumAtoms(),
                     "rings": rdMolDescriptors.CalcNumRings(m),
                     "logZ": logZ(m), "dIpc": dIpc(m)})

print(f"{'class':22s} {'n':>4} {'dIpc mean':>10} {'sd':>7} {'range':>16} {'rings':>6}")
for cls in ok:
    sub = [r for r in rows if r["cls"] == cls]
    d = np.array([r["dIpc"] for r in sub]); rg = np.mean([r["rings"] for r in sub])
    print(f"{cls:22s} {len(sub):4d} {d.mean():10.4f} {d.std():7.4f} "
          f"{d.min():+7.3f}..{d.max():+6.3f} {rg:6.2f}")

# general molecules for contrast
gen = []
for r in csv.DictReader(open("/Users/guillaume-osmo/Github/dmpnn-cpp-fp16/examples/bpsubset.csv")):
    m = Chem.MolFromSmiles(r["smiles"])
    if m is None or m.GetNumAtoms() < 4: continue
    gen.append({"n": m.GetNumAtoms(), "dIpc": dIpc(m),
                "rings": rdMolDescriptors.CalcNumRings(m)})
gd = np.array([g["dIpc"] for g in gen]); gn = np.array([g["n"] for g in gen])
td = np.array([r["dIpc"] for r in rows]); tn = np.array([r["n"] for r in rows])

print(f"\n{'set':22s} {'N':>5} {'dIpc mean':>10} {'sd':>7} {'median':>8} {'frac>0':>7}")
print(f"{'terpenes (curated)':22s} {len(td):5d} {td.mean():10.4f} {td.std():7.4f} "
      f"{np.median(td):8.4f} {np.mean(td>0):7.2%}")
print(f"{'general (BP set)':22s} {len(gd):5d} {gd.mean():10.4f} {gd.std():7.4f} "
      f"{np.median(gd):8.4f} {np.mean(gd>0):7.2%}")

# size-matched comparison: general molecules in the terpene size range
mask = (gn >= tn.min()) & (gn <= tn.max())
print(f"{'general, size-matched':22s} {mask.sum():5d} {gd[mask].mean():10.4f} "
      f"{gd[mask].std():7.4f} {np.median(gd[mask]):8.4f} {np.mean(gd[mask]>0):7.2%}")

from scipy import stats
u = stats.mannwhitneyu(td, gd[mask], alternative="two-sided")
print(f"\nMann-Whitney terpenes vs size-matched general: U={u.statistic:.0f}, p={u.pvalue:.3e}")
print(f"terpene dIpc is {'HIGHER' if td.mean()>gd[mask].mean() else 'LOWER'} "
      f"(more ring-fused) by {td.mean()-gd[mask].mean():+.4f} bits on average")

json.dump({"terpenes": rows,
           "general": gen,
           "zmax_exact": {5:26,6:51,7:100,8:209,9:388,10:780,11:1482,12:2921,13:5600,
                          14:11032,15:21482,16:42439}},
          open(f"{sys.path[1]}/data.json", "w"))
print(f"\nwrote data.json ({len(rows)} terpenes, {len(gen)} general)")
