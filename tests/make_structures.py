"""Build the static test structures in tests/structures/.

  h2o.vasp          Ray's water molecule (byte copy of ray_structures/h2o.vasp)
  h2o_copy.vasp     exact copy of h2o.vasp
  h2o_rotated.vasp  h2o rigidly rotated 5 degrees about its centroid
                    (axis = x, the normal of the molecule's plane)
  h2o_stretched.vasp  h2o with the first O-H bond lengthened by 0.5 A along
                    its own direction (other H unchanged)
  h2s.vasp          H2S in the same box: S-H 1.34 A, H-S-H 92 degrees, S on
                    water's O site, H's in water's plane on the same bisector

Run from anywhere:  python tests/make_structures.py
"""
import os
import shutil

import numpy as np
from ase import Atoms
from ase.io import read

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "ray_structures", "h2o.vasp")
OUT = os.path.join(HERE, "structures")

ROTATION_DEG = 5.0
SH_BOND, HSH_ANGLE = 1.34, 92.0   # Å, degrees
STRETCH = 0.5                     # Å added to the first O-H bond


def write(atoms, name):
    atoms.write(os.path.join(OUT, name), format="vasp", direct=True)


os.makedirs(OUT, exist_ok=True)
shutil.copyfile(SRC, os.path.join(OUT, "h2o.vasp"))
shutil.copyfile(SRC, os.path.join(OUT, "h2o_copy.vasp"))

water = read(SRC, format="vasp")       # order: O, H, H

rotated = water.copy()
rotated.rotate(ROTATION_DEG, "x", center=rotated.positions.mean(axis=0))
write(rotated, "h2o_rotated.vasp")

o, h1, h2 = water.positions

stretched = water.copy()
oh = (h1 - o) / np.linalg.norm(h1 - o)
stretched.positions[1] = h1 + STRETCH * oh
write(stretched, "h2o_stretched.vasp")

bisector = (h1 - o) / np.linalg.norm(h1 - o) + (h2 - o) / np.linalg.norm(h2 - o)
bisector /= np.linalg.norm(bisector)
normal = np.cross(h1 - o, h2 - o)
across = np.cross(normal, bisector)
across /= np.linalg.norm(across)
half = np.radians(HSH_ANGLE / 2)
h_dirs = [np.cos(half) * bisector + s * np.sin(half) * across for s in (1, -1)]
h2s = Atoms("SH2", positions=[o, o + SH_BOND * h_dirs[0], o + SH_BOND * h_dirs[1]],
            cell=water.cell, pbc=True)
write(h2s, "h2s.vasp")

print(f"wrote 5 structures to {OUT}/")
