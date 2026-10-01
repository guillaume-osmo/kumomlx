"""A curated terpene reference set, validated by the isoprene rule.

No COCONUT dump was available locally, so the set is curated. Every entry is checked against the
carbon count its class requires -- C10 monoterpene, C15 sesquiterpene, C20 diterpene, C30
triterpene -- which catches a mistranscribed SMILES, since a wrong skeleton almost always has
the wrong number of carbons. Entries failing the check are reported and dropped rather than used.
"""
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

TERPENES = {
    "monoterpene (C10)": {
        "limonene": "CC1=CCC(CC1)C(=C)C", "alpha-pinene": "CC1=CCC2CC1C2(C)C",
        "beta-pinene": "CC1(C)C2CCC(=C)C1C2", "myrcene": "CC(=CCCC(=C)C=C)C",
        "camphene": "CC1(C)C2CCC(=C)C1C2", "3-carene": "CC1=CCC2C(C1)C2(C)C",
        "sabinene": "CC(C)C12CCC(=C)C1C2", "alpha-terpinene": "CC1=CC=C(CC1)C(C)C",
        "gamma-terpinene": "CC1=CCC(=CC1)C(C)C", "terpinolene": "CC1=CCC(=C(C)C)CC1",
        "ocimene": "CC(=CCC=C(C)C=C)C", "alpha-phellandrene": "CC(C)C1CC=C(C)C=C1",
        "menthol": "CC(C)C1CCC(C)CC1O", "carvone": "CC(=C)C1CC=C(C)C(=O)C1",
        "camphor": "CC1(C)C2CCC1(C)C(=O)C2", "borneol": "CC1(C)C2CCC1(C)C(O)C2",
        "linalool": "CC(=CCCC(C)(C=C)O)C", "geraniol": "CC(=CCCC(=CCO)C)C",
        "citronellol": "CC(CCC=C(C)C)CCO", "citral": "CC(=CCCC(=CC=O)C)C",
        "thymol": "CC(C)c1ccc(C)cc1O", "eucalyptol": "CC1(C)C2CCC(C)(O2)CC1",
        "alpha-terpineol": "CC1=CCC(CC1)C(C)(C)O", "fenchone": "CC1(C)C2CCC(C)(C2)C1=O",
        "pulegone": "CC1CCC(=C(C)C)C(=O)C1", "menthone": "CC(C)C1CCC(C)CC1=O",
    },
    "sesquiterpene (C15)": {
        "beta-caryophyllene": "CC1=CCCC(=C)C2CC(C)(C)C2CC1",
        "alpha-humulene": "CC1=CCCC(C)(C)C=CCC(C)=CC1",
        "alpha-farnesene": "CC(=CCCC(=CCCC(=C)C=C)C)C",
        "beta-bisabolene": "CC(=CCCC(=C)C1CCC(C)=CC1)C",
        "valencene": "CC1=CC2CC(C(=C)C)CCC2(C)CC1",
        "longifolene": "CC1(C)CCCC2(C)C1CC3CC2C3=C",
        "nerolidol": "CC(=CCCC(=CCCC(C)(C=C)O)C)C",
        "farnesol": "CC(=CCCC(=CCCC(=CCO)C)C)C",
        "zingiberene": "CC(=CCCC(C)C1CC=C(C)C=C1)C",
        "patchoulol": "CC1CCC2(C)CCC3(C)C(O)CCC1C23",
    },
    "diterpene (C20)": {
        "phytol": "CC(C)CCCC(C)CCCC(C)CCCC(=CCO)C",
        "sclareol": "CC1(C)CCCC2(C)C1CCC(C)(O)C2CCC(C)(O)C=C",
        "manool": "CC1(C)CCCC2(C)C1CCC(=C)C2CCC(C)(O)C=C",
        "abietic acid": "CC(C)C1=CC2=CCC3C(C)(C(=O)O)CCCC3(C)C2CC1",
    },
    "triterpene (C30)": {
        "squalene": "CC(=CCCC(=CCCC(=CCCC=C(C)CCC=C(C)CCC=C(C)C)C)C)C",
        "lanosterol": "CC(CCC=C(C)C)C1CCC2(C)C1CCC3C2(C)CCC4C3(C)CCC(O)C4(C)C",
        "lupeol": "CC(=C)C1CCC2(C)CCC3(C)C(CCC4C5(C)CCC(O)C(C)(C)C5CCC34C)C12",
    },
}
EXPECTED_C = {"monoterpene (C10)": 10, "sesquiterpene (C15)": 15,
              "diterpene (C20)": 20, "triterpene (C30)": 30}


def validated():
    """Return {class: {name: (smiles, mol)}} keeping only entries passing the isoprene check."""
    out, bad = {}, []
    for cls, members in TERPENES.items():
        want = EXPECTED_C[cls]
        keep = {}
        for name, smi in members.items():
            m = Chem.MolFromSmiles(smi)
            if m is None:
                bad.append((cls, name, "unparseable"))
                continue
            nc = sum(a.GetSymbol() == "C" for a in m.GetAtoms())
            if nc != want:
                bad.append((cls, name, f"{nc} carbons, expected {want}"))
                continue
            keep[name] = (smi, m)
        out[cls] = keep
    return out, bad


if __name__ == "__main__":
    ok, bad = validated()
    for cls, members in ok.items():
        print(f"{cls:22s} {len(members):3d} validated / {len(TERPENES[cls])}")
    if bad:
        print(f"\nREJECTED ({len(bad)}) -- dropped, not used:")
        for cls, name, why in bad:
            print(f"   {cls:22s} {name:22s} {why}")
