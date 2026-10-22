#!/usr/bin/env python3

import argparse
import os

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.io import read

from soft_kdb import Entry, compare

HERE = os.path.dirname(os.path.abspath(__file__))
RAY_DIR = os.path.join(HERE, "ray_structures")
OUT_DIR = os.path.join(HERE, "simple_cases")
BOX = (15.0, 9.0, 9.0)   # same box as Ray's h2o CONTCAR

SAME, SIMILAR = 0.95, 0.50


# ------------------------------------------------------------ structures ----
def load(name, fallback):
    """Read ray_structures/<name>.vasp, or build a stand-in if it's missing."""
    path = os.path.join(RAY_DIR, f"{name}.vasp")
    if os.path.exists(path):
        return read(path, format="vasp")
    print(f"  WARNING: {path} not found, using an ASE-built stand-in for {name}")
    atoms = fallback()
    atoms.set_cell(BOX)
    atoms.set_pbc(True)
    atoms.center()
    return atoms


def stand_in_linear_h2o():
    return Atoms("OH2", positions=[(0, 0, 0), (0.97, 0, 0), (-0.97, 0, 0)])


def stand_in_hoh():
    w = molecule("H2O")                      # ASE order is O, H, H
    return w[[1, 0, 2]]                      # reorder to H, O, H


def center_of(atoms):
    return atoms.positions.mean(axis=0)


# ------------------------------------------------- simple modifications ----
# Each takes Atoms and returns a NEW Atoms; the original is never changed.
def translate(atoms, shift=(1.0, 0.5, -0.5)):
    a = atoms.copy()
    a.translate(shift)
    return a


def rotate(atoms, degrees, axis="z"):
    """Rigid rotation about the molecule's own center."""
    a = atoms.copy()
    a.rotate(degrees, axis, center=center_of(a))
    return a


def mirror(atoms, axis=0):
    """Reflect through a plane through the molecule's center (a symmetry op)."""
    a = atoms.copy()
    c = center_of(a)
    pos = a.positions.copy()
    pos[:, axis] = 2 * c[axis] - pos[:, axis]
    a.positions = pos
    return a


def shuffle_order(atoms):
    """Same atoms listed in reverse order in the file."""
    return atoms[list(range(len(atoms)))[::-1]]


def stretch_bond(atoms, i, j, delta):
    """Move atom j along the i->j bond by delta Å (positive = longer)."""
    a = atoms.copy()
    v = a.get_distance(i, j, mic=True, vector=True)
    a.positions[j] += delta * v / np.linalg.norm(v)
    return a


def duplicate(atoms, shift=(5.0, 0.0, 0.0)):
    """Two copies of the molecule in the same box (Ray's 1 vs 2 question)."""
    a = atoms.copy()
    b = atoms.copy()
    b.translate(shift)
    return a + b


def first_index(atoms, symbol):
    return atoms.get_chemical_symbols().index(symbol)


def h_bonded_to_o(atoms):
    """Index of one H in a water molecule (needed for the bond-stretch cases)."""
    syms = atoms.get_chemical_symbols()
    return [k for k, s in enumerate(syms) if s == "H"][0]


# ----------------------------------------------------------------- cases ----
def build_cases():
    h2o = load("h2o", lambda: molecule("H2O"))
    hoh = load("hoh", stand_in_hoh)
    lin = load("linear_h2o", stand_in_linear_h2o)
    o2 = load("o2", lambda: molecule("O2"))

    o, h = first_index(h2o, "O"), h_bonded_to_o(h2o)

    # (name, query, other, expected, why)
    return [
        ("copy",              h2o, h2o.copy(),                "same",
         "identical file copied"),
        ("translated",        h2o, translate(h2o),            "same",
         "whole molecule moved in the box"),
        ("rotated 5 deg",     h2o, rotate(h2o, 5),            "same",
         "small rigid rotation"),
        ("rotated 30 deg",    h2o, rotate(h2o, 30),           "same",
         "rigid rotation, no internal change"),
        ("rotated 90 deg x",  h2o, rotate(h2o, 90, "x"),      "same",
         "rigid rotation about another axis"),
        ("mirrored",          h2o, mirror(h2o),               "same",
         "reflection (symmetry operation)"),
        ("atom order",        h2o, shuffle_order(h2o),        "same",
         "same atoms, listed in a different order"),
        ("hoh file",          h2o, hoh,                       "same",
         "Ray's hoh calc -- check what it actually is"),
        ("O-H +0.05 A",       h2o, stretch_bond(h2o, o, h, 0.05), "same",
         "tiny bond change, well under 0.3 A"),
        ("O-H +0.30 A",       h2o, stretch_bond(h2o, o, h, 0.30), "similar",
         "bond change equal to KDB's 0.3 A cutoff"),
        ("linear water",      h2o, lin,                       "different",
         "same bonds, 180 deg angle instead of ~104"),
        ("two waters",        h2o, duplicate(h2o),            "similar",
         "OPEN QUESTION (Ray): one molecule vs two"),
        ("O2",                h2o, o2,                        "different",
         "different molecule"),
        ("O2 copy",           o2,  o2.copy(),                 "same",
         "identical file copied (second molecule)"),
        ("O2 rotated 45 deg", o2,  rotate(o2, 45, "y"),       "same",
         "rigid rotation"),
    ]


# ------------------------------------------------------------------- run ----
def verdict(score):
    if score >= SAME:
        return "same"
    if score >= SIMILAR:
        return "similar"
    return "different"


def as_entry(name, atoms):
    # Isolated molecules: treat EVERY atom as the "adsorbate". soft_kdb's
    # automatic find_adsorbate assumes a slab and would drop atoms here.
    return Entry(name, atoms, adsorbate=list(range(len(atoms))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="save each case's structures to simple_cases/")
    args = ap.parse_args()

    cases = build_cases()
    print()
    hdr = (f"{'case':<18} {'overall':>7} {'topol':>5} {'geom':>5} {'comp':>5} "
           f"{'rms Å':>6}  {'expected':<9} {'got':<9} {'':<4} why")
    print(hdr)
    print("-" * len(hdr))

    n_ok = 0
    for name, query, other, expected, why in cases:
        s = compare(as_entry("query", query), as_entry(name, other))
        got = verdict(s["overall"])
        ok = got == expected
        n_ok += ok
        rms = f"{s['rms']:.3f}" if s["rms"] is not None else "  -"
        print(f"{name:<18} {s['overall']:7.3f} {s['topology']:5.2f} "
              f"{s['geometry']:5.2f} {s['composition']:5.2f} {rms:>6}  "
              f"{expected:<9} {got:<9} {'ok' if ok else '<--':<4} {why}")

        if args.write:
            folder = os.path.join(OUT_DIR, name.replace(" ", "_"))
            os.makedirs(folder, exist_ok=True)
            query.write(os.path.join(folder, "query.vasp"), format="vasp")
            other.write(os.path.join(folder, "other.vasp"), format="vasp")

    print(f"\n{n_ok}/{len(cases)} cases match the expected verdict.")
    print("Rows marked <-- disagree: decide whether the algorithm or the "
          "expectation is wrong.")
    if args.write:
        print(f"Structures written to {OUT_DIR}/")


if __name__ == "__main__":
    main()