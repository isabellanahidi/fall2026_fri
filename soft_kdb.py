#!/usr/bin/env python3
"""
soft_kdb.py — a "grayscale" version of KDB's structure matching.

KDB (Henkelman group) decides match / no-match with hard gates:
  1. the query must contain every element of the stored entry,
  2. the stored bond graph must fit *exactly* inside the query's bond graph,
  3. every mapped distance must agree within 0.3 Å.

This prototype keeps KDB's ideas (covalent-radius bond graphs, local regions,
0.3 Å as the natural distance scale) but replaces each gate with a score:

  composition  – how much the two regions' elements overlap        (0–1)
  topology     – size of the largest shared bonded fragment        (0–1)
  geometry     – how close the shared fragment's distances are     (0–1)
  overall      – composition × topology × geometry, used for ranking

Region of interest (for slabs): the adsorbate atoms plus the surface atoms
bonded to them. Only that region is compared, so two slabs with different
cell sizes, thicknesses or coverages can still score as similar.

Usage
  python soft_kdb.py QUERY_FILE  DB_FOLDER_OR_FILE [...]
  python soft_kdb.py CONTCAR_new  /path/to/old_calcs/      # ranks every match
Options
  --shells N        bonded neighbor shells added around the adsorbate (default 1)
  --fudge F         bond tolerance, like KDB's NEIGHBOR_FUDGE (default 0.2)
  --adsorbate i,j   adsorbate atom indices for the query (default: auto)
  --top K           show only the best K results

Any format ASE reads works (POSCAR/CONTCAR, .vasp, .xyz, .cif, OUTCAR, ...).
A folder is searched recursively for CONTCAR, POSCAR, *.vasp, *.cif, *.xyz.
Requires: ase, networkx, numpy.
"""

import argparse
import math
import os
import sys
from collections import Counter

import networkx as nx
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from networkx.algorithms.isomorphism import ISMAGS

DIST_SCALE = 0.3  # Å — KDB's DISTANCE_CUTOFF, reused as the decay scale
STRUCT_NAMES = ("CONTCAR", "POSCAR")
STRUCT_EXTS = (".vasp", ".cif", ".xyz", ".traj")


# ---------------------------------------------------------------- bonding ----
def bond_graph(atoms, fudge=0.2):
    """Bond graph like KDB's: bonded if d < (r_i + r_j) * (1 + fudge).
    Periodic boundaries are respected. Nodes carry the element symbol."""
    cutoffs = natural_cutoffs(atoms, mult=1.0 + fudge)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)
    g = nx.Graph()
    for i, sym in enumerate(atoms.get_chemical_symbols()):
        g.add_node(i, element=sym)
    for i in range(len(atoms)):
        for j in nl.get_neighbors(i)[0]:
            if i < j:
                g.add_edge(i, int(j))
    return g


# ---------------------------------------------------- region of interest ----
def find_adsorbate(atoms, gap=0.5):
    """Guess the adsorbate on a slab: atoms sitting more than `gap` Å above
    the top layer of the slab's most common element. Falls back to the
    topmost layer if nothing qualifies (e.g. a clean slab)."""
    syms = atoms.get_chemical_symbols()
    z = atoms.positions[:, 2]
    host = Counter(syms).most_common(1)[0][0]
    top_host = max(z[i] for i, s in enumerate(syms) if s == host)
    ads = [i for i in range(len(atoms)) if z[i] > top_host + gap]
    if not ads:
        ads = [i for i in range(len(atoms)) if z[i] > top_host - gap]
    return ads


def region(atoms, graph, adsorbate, shells=1):
    """Adsorbate atoms plus `shells` bonded shells around them."""
    nodes = set(adsorbate)
    frontier = set(adsorbate)
    for _ in range(shells):
        frontier = {n for f in frontier for n in graph.neighbors(f)} - nodes
        nodes |= frontier
    return sorted(nodes)


# --------------------------------------------------------------- scoring ----
def composition_score(syms_a, syms_b):
    """Weighted Jaccard overlap of element counts (1 = same composition)."""
    ca, cb = Counter(syms_a), Counter(syms_b)
    keys = set(ca) | set(cb)
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / union if union else 0.0


def distance_rms(atoms_a, atoms_b, mapping):
    """RMS difference of all pairwise distances among mapped atoms (Å).
    Uses minimum-image distances, so cell size and position don't matter."""
    pairs = list(mapping.items())
    if len(pairs) < 2:
        return 0.0
    diffs = []
    for x in range(len(pairs)):
        for y in range(x + 1, len(pairs)):
            (a1, b1), (a2, b2) = pairs[x], pairs[y]
            da = atoms_a.get_distance(a1, a2, mic=True)
            db = atoms_b.get_distance(b1, b2, mic=True)
            diffs.append(da - db)
    return float(np.sqrt(np.mean(np.square(diffs))))


def best_common_fragment(ga, gb, atoms_a, atoms_b, max_mappings=200):
    """Largest common bonded fragment (element-labelled, induced subgraph).
    Among equally large fragments, keep the one with the closest geometry."""
    same_element = lambda n1, n2: n1["element"] == n2["element"]
    small, big, swapped = (gb, ga, True) if len(gb) < len(ga) else (ga, gb, False)
    ismags = ISMAGS(big, small, node_match=same_element)
    best_map, best_rms = {}, math.inf
    for k, m in enumerate(ismags.largest_common_subgraph(symmetry=False)):
        # m maps big-graph nodes -> small-graph nodes; convert to a -> b
        if swapped:          # big = ga, small = gb
            mapping = dict(m)
        else:                # big = gb, small = ga
            mapping = {v: u for u, v in m.items()}
        rms = distance_rms(atoms_a, atoms_b, mapping)
        if len(mapping) > len(best_map) or (len(mapping) == len(best_map) and rms < best_rms):
            best_map, best_rms = mapping, rms
        if k + 1 >= max_mappings:
            break
    return best_map, (best_rms if best_map else math.inf)


# ------------------------------------------------------------ main logic ----
class Entry:
    """One structure, reduced to its region of interest."""

    def __init__(self, path, atoms, fudge=0.2, shells=1, adsorbate=None):
        self.path = path
        full_graph = bond_graph(atoms, fudge)
        ads = adsorbate if adsorbate is not None else find_adsorbate(atoms)
        self.region_idx = region(atoms, full_graph, ads, shells)
        self.atoms = atoms[self.region_idx]
        # rebuild the graph on the extracted region, keeping original bonds
        keep = {old: new for new, old in enumerate(self.region_idx)}
        self.graph = nx.Graph()
        for old, new in keep.items():
            self.graph.add_node(new, element=atoms[old].symbol)
        for u, v in full_graph.subgraph(self.region_idx).edges():
            self.graph.add_edge(keep[u], keep[v])
        self.adsorbate_formula = atoms[ads].get_chemical_formula() if ads else "-"
        self.formula = atoms.get_chemical_formula()


def compare(q, e):
    """Score entry e against query q. Returns a dict of the breakdown."""
    comp = composition_score(q.atoms.get_chemical_symbols(),
                             e.atoms.get_chemical_symbols())
    if comp == 0:
        return dict(composition=0.0, topology=0.0, geometry=0.0, overall=0.0,
                    matched=0, rms=None)
    mapping, rms = best_common_fragment(q.graph, e.graph, q.atoms, e.atoms)
    n = len(mapping)
    # Dice overlap: 1.0 only if every region atom on both sides is matched
    topo = 2 * n / (len(q.atoms) + len(e.atoms))
    geom = math.exp(-rms / DIST_SCALE) if n > 1 else (1.0 if n == 1 else 0.0)
    return dict(composition=comp, topology=topo, geometry=geom,
                overall=comp * topo * geom, matched=n, rms=rms if n > 1 else None)


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for name in names:
                    if name in STRUCT_NAMES or name.endswith(STRUCT_EXTS):
                        files.append(os.path.join(root, name))
        else:
            files.append(p)
    return sorted(files)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("database", nargs="+")
    ap.add_argument("--shells", type=int, default=1)
    ap.add_argument("--fudge", type=float, default=0.2)
    ap.add_argument("--adsorbate", type=str, default=None)
    ap.add_argument("--top", type=int, default=None)
    args = ap.parse_args(argv)

    ads = [int(i) for i in args.adsorbate.split(",")] if args.adsorbate else None
    q = Entry(args.query, read(args.query), args.fudge, args.shells, ads)
    print(f"Query: {args.query}  ({q.formula}, adsorbate {q.adsorbate_formula}, "
          f"region {len(q.atoms)} atoms: {q.atoms.get_chemical_formula()})\n")

    results = []
    for path in collect(args.database):
        if os.path.abspath(path) == os.path.abspath(args.query):
            continue
        try:
            e = Entry(path, read(path), args.fudge, args.shells)
        except Exception as exc:  # unreadable file: report and move on
            print(f"  skipped {path}: {exc}", file=sys.stderr)
            continue
        results.append((compare(q, e), e))

    results.sort(key=lambda r: (r[0]["overall"], r[0]["composition"]), reverse=True)
    if args.top:
        results = results[: args.top]

    hdr = f"{'rank':>4}  {'overall':>7}  {'topol':>5}  {'geom':>5}  {'comp':>5}  " \
          f"{'matched':>7}  {'rms Å':>6}  {'adsorb':>7}  entry"
    print(hdr)
    print("-" * len(hdr))
    for k, (s, e) in enumerate(results, 1):
        rms = f"{s['rms']:.3f}" if s["rms"] is not None else "  -"
        matched = f"{s['matched']}/{len(e.atoms)}"
        print(f"{k:>4}  {s['overall']:7.3f}  {s['topology']:5.2f}  {s['geometry']:5.2f}  "
              f"{s['composition']:5.2f}  {matched:>7}  {rms:>6}  "
              f"{e.adsorbate_formula:>7}  {e.path}")


if __name__ == "__main__":
    main()
