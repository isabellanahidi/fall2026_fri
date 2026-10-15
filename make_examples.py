"""Build a small set of example slab relaxations to test soft_kdb.py.

The query is CO on top of a Pt atom on Pt(111). The "database" holds
variations that a strict KDB-style match would mostly reject, ranging from
nearly identical to clearly different.
"""
import os
from ase.build import fcc111, add_adsorbate, molecule

OUT = "examples"


def slab(metal, size, mol, site, height, tilt=0.0):
    s = fcc111(metal, size=size, vacuum=10.0)
    m = molecule(mol)
    if mol == "CO":
        m.rotate(180, "x")          # carbon down, oxygen up
    if tilt:
        m.rotate(tilt, "y", center=m[0].position)
    add_adsorbate(s, m, height, site)
    return s


cases = {
    "query/CONTCAR":                     slab("Pt", (3, 3, 4), "CO", "ontop", 1.85),
    "db/Pt_CO_top_bigcell/CONTCAR":      slab("Pt", (4, 4, 3), "CO", "ontop", 1.85),
    "db/Pt_CO_top_higher/CONTCAR":       slab("Pt", (3, 3, 4), "CO", "ontop", 2.05),
    "db/Pt_CO_top_tilted/CONTCAR":       slab("Pt", (3, 3, 4), "CO", "ontop", 1.85, tilt=25),
    "db/Pt_CO_fcc_hollow/CONTCAR":       slab("Pt", (3, 3, 4), "CO", "fcc", 1.30), # bcc would make more sense than fcc
    "db/Pt_O_fcc_hollow/CONTCAR":        slab("Pt", (3, 3, 4), "O", "fcc", 1.20), # ontop would be more stable than fcc
    "db/Cu_CO_top/CONTCAR":              slab("Cu", (3, 3, 4), "CO", "ontop", 1.85),
    "db/Pt_clean/CONTCAR":               fcc111("Pt", size=(3, 3, 4), vacuum=10.0),
}

for path, atoms in cases.items():
    full = os.path.join(OUT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    atoms.write(full, format="vasp")
print(f"wrote {len(cases)} structures to {OUT}/")
