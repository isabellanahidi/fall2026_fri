#!/usr/bin/env python3
"""
kdb_compare.py — run KDB's structural comparison on two POSCAR files and
print a step-by-step trace of every check.

This is a line-by-line port of the matching part of KDB's query
(kdb_reference/kdb/kdbquery.py, graph.py, isomorphism.py, fortran/common.f90).
Nothing is added to KDB's definition of "same structure".

KDB's comparison is one-directional: a stored entry is searched for inside a
query structure. Here:
  A = the KDB entry   (KDB's "kdbmin",   graph G2)
  B = the reactant    (KDB's "reactant", graph G1)
A matches B when every atom of A can be assigned to a distinct atom of B and
the assigned atoms line up after a rigid rotation.

All atoms of A are treated as mobile (KDB reads entry['mobile'] from its
database; we have no database, so nothing is guessed and every atom counts).
Frozen atoms in B come from the POSCAR's Selective dynamics flags, as in KDB.

Usage
  python kdb_compare.py A.vasp B.vasp
Options
  --nf-bonds    build bonds with mult = 1 + NEIGHBOR_FUDGE, as KDB did before
                commit 1997cff (graph.py:38). The current KDB code uses mult = 0,
                which produces no bonds at all (graph.py:39).
Exit code 0 = match, 1 = no match.
Requires: ase, networkx, numpy.
"""

import argparse
import sys
from collections import Counter

import networkx as nx
import numpy as np
from ase.data import covalent_radii
from ase.io import read
from ase.neighborlist import NeighborList, PrimitiveNeighborList

# KDB parameters — kdb_reference/kdb/config.py
NEIGHBOR_FUDGE = 0.2         # config.py:5  (also the nf default, kdbquery.py:472)
DISTANCE_CUTOFF = 0.3        # config.py:6  (also the dc default, kdbquery.py:472)
USE_GROUP_SIMILARITY = False  # config.py:12 (compare elements by symbol)
USE_GRAPH = True             # config.py:15 (graph method, not brute force)
# MOBILE_ATOM_CUTOFF (config.py:4) and PBC_MAPPING_CHECK (config.py:7) are not
# used by the graph method's comparison.

MAX_LISTED = 10  # how many mappings to print individually in Step 6


# ------------------------------------------------------------------ trace ----
class NoMatch(Exception):
    pass


def header(n, title, source):
    print(f"\nStep {n}: {title}")
    print(f"  [KDB: {source}]")


def info(text):
    print(f"  {text}")


def passed(reason):
    print(f"  PASS — {reason}")


def failed(step, reason):
    print(f"  FAIL — {reason}")
    raise NoMatch(f"failed at Step {step}: {reason}")


def fmt_mapping(mapping, limit=12):
    """A->B pairs sorted by A index; same-index pairs are summarised."""
    moved = [(a, b) for a, b in sorted(mapping.items()) if a != b]
    if not moved:
        return f"A_i->B_i for all {len(mapping)} atoms"
    text = " ".join(f"A{a}->B{b}" for a, b in moved[:limit])
    if len(moved) > limit:
        text += f" ... (+{len(moved) - limit} more)"
    same = len(mapping) - len(moved)
    return text + (f"; other {same} atoms A_i->B_i" if same else "")


def fmt_counts(counts):
    return ", ".join(f"{el}:{n}" for el, n in sorted(counts.items())) or "none"


# ---------------------------------------------------- KDB geometry helpers ----
def pbc(v, cell, icell):
    """fortran/common.f90:109-129 (pbc). Works on one vector or an (n, 3) array."""
    f = v @ icell
    f = np.mod(f + 0.5, 1.0) - 0.5
    return f @ cell


def perm_distances(pos, cell, icell):
    """fortran/common.f90:17-56 (atom_atom_perm_distance), called through
    common.py:133-146. For every pair (a, b): distances from a to b's image in
    the 8 periodic cells nearest to b's wrapped position. Returns (n, n, 8)."""
    v = pos[None, :, :] - pos[:, None, :]          # v = pos[b] - pos[a]
    vdir = v @ icell
    vdir = np.mod(vdir, 1.0) + 1.5
    vdir = np.mod(vdir, 1.0) - 0.5
    vpbc = vdir @ cell
    sgn = -np.where(vdir >= 0, 1.0, -1.0)            # -1 * sign(1, vdir)
    d = np.empty(v.shape[:2] + (8,))
    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                shift = (i * sgn[..., 0:1] * cell[0] + j * sgn[..., 1:2] * cell[1]
                         + k * sgn[..., 2:3] * cell[2])
                d[..., 4 * i + 2 * j + k] = np.linalg.norm(vpbc + shift, axis=-1)
    return d


def find_clump_order(pos):
    """fortran/common.f90:162-217 (find_clump_order). Greedy closest-pair order,
    no PBC: [source_1, target_1, source_2, target_2, ...]."""
    n = len(pos)
    done, todo, order = [0], list(range(1, n)), []
    while todo:
        best = None
        for i in todo:
            for j in done:
                d2 = np.sum((pos[i] - pos[j]) ** 2)
                if best is None or d2 < best[0]:
                    best = (d2, i, j)
        _, i, j = best
        order += [j, i]
        done.append(i)
        todo.remove(i)
    return order


def ordered_clump(pos, order, cell, icell):
    """fortran/common.f90:220-258 (ordered_clump). Moves each successor atom to
    the periodic image closest to its source atom."""
    out = pos.copy()
    for s, t in zip(order[0::2], order[1::2]):
        out[t] = out[s] + pbc(out[t] - out[s], cell, icell)
    return out


# ----------------------------------------------------------------- graphs ----
def build_graph(atoms, mult, self_interaction):
    """graph.py:21-56 (graph_reactant) and graph.py:81-116 (graph_kdbentry).
    Nodes are labelled by element; edges come from a neighbor list with
    cutoff = covalent_radius * mult and skin = 0."""
    cutoffs = [covalent_radii[z] * mult for z in atoms.numbers]
    nl = NeighborList(cutoffs, skin=0.0, self_interaction=self_interaction,
                      bothways=False, primitive=PrimitiveNeighborList)
    nl.update(atoms)
    g = nx.Graph()
    for i, sym in enumerate(atoms.get_chemical_symbols()):
        g.add_node(i, symbol=sym)
    for i in range(len(atoms)):
        for j in nl.get_neighbors(i)[0]:
            if i != j:
                g.add_edge(i, int(j))
    return g


# ----------------------------------------------------- VF2 search order ----
def argmax(nodes, key, reverse=False, use_label=False, graph=None):
    """isomorphism.py:191-208."""
    if len(nodes) == 1:
        return set(nodes)
    best, out = None, set()
    for node in nodes:
        val = key[graph.nodes[node]["symbol"]] if use_label else key[node]
        if reverse:
            val = -val
        if best is None or val > best:
            best, out = val, {node}
        elif val == best:
            out.add(node)
    return out


def node_order(G1, G2):
    """isomorphism.py:124-189 (nodeOrder_plusplus) and 210-235 (process_bfsTree),
    without the partialMapping branch (KDB's query never passes one)."""
    order, remaining = {}, set(G2.nodes)
    rarity = Counter(G1.nodes[n]["symbol"] for n in G1.nodes)
    degree = G2.degree
    n = 0

    def process(at_depth, parents, n):
        new_parents = at_depth.copy()
        conn = {node: len(parents & set(G2.adj[node])) for node in at_depth}
        while at_depth:
            sel = argmax(at_depth, conn)
            sel = argmax(sel, degree)
            sel = argmax(sel, rarity, reverse=True, use_label=True, graph=G2).pop()
            remaining.remove(sel)
            at_depth.remove(sel)
            order[sel] = n
            n += 1
            rarity[G2.nodes[sel]["symbol"]] -= 1
            for node in at_depth:
                if node in G2.adj[sel]:
                    conn[node] += 1
        return new_parents, n

    while remaining:
        root = max(argmax(remaining, rarity, reverse=True, use_label=True, graph=G2),
                   key=degree)
        order[root] = n
        n += 1
        remaining.remove(root)
        rarity[G2.nodes[root]["symbol"]] -= 1
        parents, at_depth = {root}, set()
        for u, v in nx.bfs_edges(G2, source=root):
            if u in parents:
                at_depth.add(v)
            else:
                parents, n = process(at_depth, parents, n)
                at_depth = {v}
        if at_depth:
            parents, n = process(at_depth, parents, n)
    return order


# ------------------------------------------------------- VF2 subgraph match ----
class Matcher:
    """isomorphism.py GraphMatcher with test = "subgraph", as called from
    kdbquery.py:197 and kdbquery.py:201 (subgraph_isomorphisms_iter)."""

    CHECKS = {
        "element": "same element (isomorphism.py:454-455)",
        "frozen": "mobile atom not on a frozen atom (isomorphism.py:459-462)",
        "degree": "mobile atom has same bond count (isomorphism.py:464-467)",
        "distance": "pair distances agree within 2*dc (isomorphism.py:470-479)",
        "selfloop": "same self-loop count (isomorphism.py:486-490)",
        "bond": "bonds to mapped atoms agree (isomorphism.py:497-520)",
        "lookahead": "neighbor element look-ahead (isomorphism.py:522-585)",
    }

    def __init__(self, G1, G2, d_A, perm_B, symbols_A, symbols_B, mobile, frozen, dc):
        self.G1, self.G2 = G1, G2
        self.d_A, self.perm_B = d_A, perm_B
        self.sA, self.sB = symbols_A, symbols_B
        self.mobile, self.frozen, self.dc = mobile, frozen, dc
        self.order = node_order(G1, G2)
        self.core_1, self.core_2, self.inout_1, self.inout_2 = {}, {}, {}, {}
        self.rejects = Counter()
        self.deepest = (-1, {}, None, None)  # (depth, partial A->B, reason, check)
        # isomorphism.py:89-94
        sys.setrecursionlimit(max(sys.getrecursionlimit(), int(1.5 * len(G2)) + 100))

    def candidate_pairs(self):
        """isomorphism.py:250-279."""
        T1 = [n for n in self.inout_1 if n not in self.core_1]
        T2 = [n for n in self.inout_2 if n not in self.core_2]
        if T1 and T2:
            node_2 = min(T2, key=self.order.__getitem__)
            for node_1 in T1:
                yield node_1, node_2
        else:
            node_2 = min(set(self.G2) - set(self.core_2), key=self.order.__getitem__)
            for node_1 in self.G1:
                if node_1 not in self.core_1:
                    yield node_1, node_2

    def reject(self, kind, detail):
        self.rejects[kind] += 1
        depth = len(self.core_1)
        # report the deepest rejection; at equal depth, an element mismatch
        # (expected for most candidate pairs) never hides a more telling check
        old_depth, _, _, old_kind = self.deepest
        if depth > old_depth or (depth == old_depth and
                                 (kind != "element" or old_kind == "element")):
            self.deepest = (depth, dict(self.core_2), detail, kind)
        return False

    def feasible(self, n1, n2):
        """isomorphism.py:412-588 (feasibility), subgraph branch."""
        G1, G2 = self.G1, self.G2
        if G1.nodes[n1]["symbol"] != G2.nodes[n2]["symbol"]:
            return self.reject("element", f"A{n2} ({self.sA[n2]}) cannot map to "
                                          f"B{n1} ({self.sB[n1]}): different element")
        if self.frozen and n2 in self.mobile and n1 in self.frozen:
            return self.reject("frozen", f"A{n2} is mobile but B{n1} is frozen")
        if n2 in self.mobile and G2.degree[n2] != G1.degree[n1]:
            return self.reject("degree", f"A{n2} has {G2.degree[n2]} bonds, "
                                         f"B{n1} has {G1.degree[n1]}")
        for o1, o2 in self.core_1.items():
            d2 = self.d_A[n2, o2]
            d1s = self.perm_B[n1, o1]
            if not np.any(np.abs(d1s - d2) < 2 * self.dc):
                best = d1s[np.argmin(np.abs(d1s - d2))]
                return self.reject(
                    "distance",
                    f"A{n2}->B{n1}: d_A(A{n2},A{o2}) = {d2:.3f} Å but closest "
                    f"d_B(B{n1},B{o1}) = {best:.3f} Å, |diff| = {abs(best - d2):.3f} "
                    f"≥ 2*dc = {2 * self.dc:.2f} Å")
        if G1.number_of_edges(n1, n1) != G2.number_of_edges(n2, n2):
            return self.reject("selfloop", f"A{n2} and B{n1} differ in self-loops")
        for nb in G1[n1]:
            if nb in self.core_1 and self.core_1[nb] not in G2[n2]:
                return self.reject("bond", f"B{n1}-B{nb} bonded but "
                                           f"A{n2}-A{self.core_1[nb]} not")
        for nb in G2[n2]:
            if nb in self.core_2 and self.core_2[nb] not in G1[n1]:
                return self.reject("bond", f"A{n2}-A{nb} bonded but "
                                           f"B{n1}-B{self.core_2[nb]} not")
        # R_terminout and R_new: B's neighbor elements must cover A's
        for test in ((lambda nb, io, core: nb in io and nb not in core),
                             (lambda nb, io, core: nb not in io)):
            counts = Counter(G1.nodes[nb]["symbol"] for nb in G1[n1]
                             if test(nb, self.inout_1, self.core_1))
            for nb in G2[n2]:
                if test(nb, self.inout_2, self.core_2):
                    sym = G2.nodes[nb]["symbol"]
                    counts[sym] -= 1
                    if counts[sym] < 0:
                        return self.reject("lookahead", f"A{n2} has more unmapped {sym} "
                                                        f"neighbors than B{n1}")
        return True

    def add(self, n1, n2):
        """isomorphism.py:600-664 (GMState.__init__)."""
        depth = len(self.core_1) + 1
        self.core_1[n1], self.core_2[n2] = n2, n1
        self.inout_1.setdefault(n1, depth)
        self.inout_2.setdefault(n2, depth)
        for core, io, G in ((self.core_1, self.inout_1, self.G1),
                            (self.core_2, self.inout_2, self.G2)):
            for node in list(core):
                for nb in G[node]:
                    if nb not in core:
                        io.setdefault(nb, depth)
        return depth

    def remove(self, n1, n2, depth):
        """isomorphism.py:683-696 (GMState.restore)."""
        del self.core_1[n1], self.core_2[n2]
        for io in (self.inout_1, self.inout_2):
            for node in [k for k, v in io.items() if v == depth]:
                del io[node]

    def mappings(self):
        """isomorphism.py:347-375 (match). Yields A index -> B index
        (swap_mapping=True, kdbquery.py:197 and isomorphism.py:358-359)."""
        if len(self.core_1) == len(self.G2):
            yield dict(self.core_2)
            return
        for n1, n2 in self.candidate_pairs():
            if self.feasible(n1, n2):
                depth = self.add(n1, n2)
                yield from self.mappings()
                self.remove(n1, n2, depth)


# --------------------------------------------------------------- compare ----
def compare(path_a, path_b, nf_bonds=False):
    nf, dc = NEIGHBOR_FUDGE, DISTANCE_CUTOFF
    A = read(path_a, format="vasp")
    B = read(path_b, format="vasp")
    sA, sB = A.get_chemical_symbols(), B.get_chemical_symbols()
    print(f"A (KDB entry):    {path_a}  — {len(A)} atoms  {A.get_chemical_formula()}")
    print(f"B (reactant):     {path_b}  — {len(B)} atoms  {B.get_chemical_formula()}")
    print(f"Parameters: dc = {dc} Å (config.py:6), nf = {nf} (config.py:5), "
          f"USE_GRAPH = {USE_GRAPH}, USE_GROUP_SIMILARITY = {USE_GROUP_SIMILARITY}")

    # Step 1 — kdbquery.py:169-181 (name count check)
    header(1, "B has at least as many atoms of each element as A",
           "kdbquery.py:169-181, query_kdb_entry")
    cA, cB = Counter(sA), Counter(sB)
    info(f"A: {fmt_counts(cA)}")
    info(f"B: {fmt_counts(cB)}")
    for el in sorted(cA):
        if el not in cB:
            failed(1, f"B has no {el} atoms (A has {cA[el]})")
        if cA[el] > cB[el]:
            failed(1, f"A has {cA[el]} {el} but B only has {cB[el]}")
    passed("every element in A is present in B in equal or greater number")

    # Step 2 — graph.py:21-56 (reactant) and graph.py:81-116 (entry)
    mult = 1 + nf if nf_bonds else 0
    header(2, "build bond graphs (nodes = atoms labelled by element)",
           "graph.py:21-56 graph_reactant, graph.py:81-116 graph_kdbentry, "
           "called at kdbquery.py:528 and kdbquery.py:196")
    if nf_bonds:
        info(f"bond if d < (r_cov,i + r_cov,j) * {mult:.1f}  "
             f"[--nf-bonds: graph.py:38, mult = 1 + nf, pre-1997cff KDB]")
    else:
        info("cutoff = r_cov * 0  [graph.py:39 and graph.py:103, mult=0 in current KDB]"
             " -> no pair can be closer than 0 Å, so no bonds are made")
    A.cell, A.pbc = B.cell, (True, True, True)       # graph.py:100-101
    G1 = build_graph(B, mult, self_interaction=True)    # graph.py:40
    G2 = build_graph(A, mult, self_interaction=False)   # graph.py:104
    info(f"G2 (A): {G2.number_of_nodes()} nodes, {G2.number_of_edges()} bonds, "
         f"{nx.number_connected_components(G2)} connected pieces")
    info(f"G1 (B): {G1.number_of_nodes()} nodes, {G1.number_of_edges()} bonds")
    passed("graphs built (this step has no pass/fail condition in KDB)")

    # Step 3 — isomorphism.py:106-113 (distance tables)
    header(3, "precompute pair distances",
           "isomorphism.py:106-113; common.py:56 atomAtomDistance, "
           "common.py:133 atomAtomPermDistance, fortran/common.f90:17-56")
    cell, icell = B.cell.array, np.linalg.inv(B.cell.array)
    d_A = np.linalg.norm(A.positions[None] - A.positions[:, None], axis=-1)
    perm_B = perm_distances(B.positions, cell, icell)
    info("A: plain Cartesian distances, no periodic wrapping")
    info("B: distances to 8 nearby periodic images of each partner atom")
    if len(A) > 1:
        iu = np.triu_indices(len(A), 1)
        info(f"A distances: min {d_A[iu].min():.3f} Å, max {d_A[iu].max():.3f} Å")
        mic_A = perm_distances(A.positions, cell, icell).min(axis=-1)
        wrapped = np.abs(d_A - mic_A)[iu] > dc
        if wrapped.any():
            i, j = iu[0][wrapped][0], iu[1][wrapped][0]
            info(f"NOTE: some A pairs are closer through the periodic boundary than "
                 f"directly (e.g. A{i}-A{j}: {d_A[i, j]:.3f} Å direct vs {mic_A[i, j]:.3f} Å "
                 "through the boundary). KDB uses only the direct distance for the entry "
                 "(common.py:56); real KDB entries are stored unwrapped (kdbinsert.py).")
    passed("distance tables built (no pass/fail condition)")

    # Step 4 — isomorphism.py:124-189 (search order)
    mobile = set(range(len(A)))   # all of A's atoms; KDB reads entry['mobile'] (kdbquery.py:188)
    frozen = set()
    if B.constraints:             # isomorphism.py:459-461 uses constraints[0].index
        frozen = set(int(i) for i in np.atleast_1d(B.constraints[0].index))
    header(4, "choose the order A's atoms are matched in (speed only)",
           "isomorphism.py:124-235, nodeOrder_plusplus")
    matcher = Matcher(G1, G2, d_A, perm_B, sA, sB, mobile, frozen, dc)
    order = sorted(matcher.order, key=matcher.order.get)
    info("order: " + " ".join(f"A{i}({sA[i]})" for i in order))
    info(f"mobile atoms in A: all {len(mobile)};  frozen atoms in B: "
         f"{len(frozen) if frozen else 'none'}")
    passed("order fixed (no pass/fail condition)")

    # Step 5 — kdbquery.py:195-202, isomorphism.py:347-588 (VF2 subgraph match)
    header(5, "assign every atom of A to a distinct atom of B (VF2 subgraph search)",
           "kdbquery.py:195-202; isomorphism.py:347-375 match, 412-588 feasibility")
    info(f"each assignment must pass, in order: element, frozen, degree, distance "
         f"(|d_A - d_B| < 2*dc = {2 * dc:.2f} Å for every already-assigned pair), bonds")
    mappings = list(matcher.mappings())
    if matcher.rejects:
        info("rejected candidate assignments by check:")
        for kind, n in matcher.rejects.most_common():
            info(f"    {n:8d}  {Matcher.CHECKS[kind]}")
    else:
        info("no candidate assignment was rejected")
    info(f"complete mappings found: {len(mappings)}")
    if not mappings:
        depth, partial, reason, _ = matcher.deepest
        if partial:
            info(f"furthest partial mapping ({depth}/{len(A)} atoms): "
                 + fmt_mapping(partial))
        failed(5, f"no way to assign all {len(A)} atoms of A; "
                  f"last blocking check: {reason}")
    passed(f"{len(mappings)} mapping(s) satisfy every check")

    # Step 6 — kdbquery.py:305-363 (rigid alignment score)
    header(6, "rotate A onto each mapping; every atom must land within dc",
           "kdbquery.py:305-363; SVD fit at 343-350, score at 355, cut at 359")
    clump_order = find_clump_order(A.positions)       # fortran/common.f90:162-217
    kdb_shifted = A.positions - A.positions.mean(axis=0)   # kdbquery.py:305-306
    H = kdb_shifted.T @ kdb_shifted                    # kdbquery.py:314-317
    S = np.linalg.svd(H, compute_uv=False)
    coplanar = S[2] / len(A) < dc * dc
    info(f"coplanar test: smallest singular value / N = {S[2] / len(A):.4f} "
         f"{'<' if coplanar else '≥'} dc² = {dc * dc:.2f} -> "
         f"{'coplanar: mirror image also tried' if coplanar else 'not coplanar'}")
    info(f"pass if score = max over atoms of |rotated A - B|² ≤ dc² = {dc * dc:.2f} Å² "
         f"(i.e. every atom within {dc} Å)")
    best = None
    for k, mapping in enumerate(mappings):
        a_idx = list(mapping.keys())                   # kdbquery.py:333-334
        b_idx = list(mapping.values())
        mapped_order = [mapping[i] for i in clump_order]     # kdbquery.py:337
        b_rot = ordered_clump(B.positions, mapped_order, cell, icell)  # kdbquery.py:341
        b_shift = b_rot[b_idx] - b_rot[b_idx].mean(axis=0)   # kdbquery.py:344-345
        U, _, V = np.linalg.svd(kdb_shifted[a_idx].T @ b_shift)  # kdbquery.py:346-347
        for mirror in range(1, -1 * coplanar - 1, -2):        # kdbquery.py:348
            V[2] *= mirror
            R = U @ V
            diff = kdb_shifted[a_idx] @ R - b_shift          # kdbquery.py:353-354
            per_atom = np.sum(diff * diff, axis=1)
            score = per_atom.max()                           # kdbquery.py:355
            ok = score <= dc * dc                            # kdbquery.py:359
            worst = a_idx[int(per_atom.argmax())]
            if k < MAX_LISTED:
                if mirror == 1:
                    info(f"mapping {k + 1}: {fmt_mapping(mapping)}")
                info(f"    {'mirror try, ' if mirror == -1 else ''}det(R) = {np.linalg.det(R):+.0f}, worst atom A{worst} off by "
                     f"{np.sqrt(score):.3f} Å (score {score:.4f}) -> "
                     f"{'within' if ok else 'exceeds'} {dc} Å")
            if best is None or score < best[0] - 1e-12:   # first mapping wins ties
                best = (score, mapping, worst, np.linalg.det(R))
    if len(mappings) > MAX_LISTED:
        info(f"... {len(mappings) - MAX_LISTED} more mappings not listed")
    score, mapping, worst, det = best
    if score > dc * dc:
        failed(6, f"best mapping still has atom A{worst} {np.sqrt(score):.3f} Å from "
                  f"its partner, more than dc = {dc} Å")
    note = " (via a mirror image, det(R) = -1)" if det < 0 else ""
    passed(f"best mapping puts every atom within {np.sqrt(score):.3f} Å{note}")
    return mapping, np.sqrt(score), note


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("A", help="KDB entry structure (POSCAR/.vasp)")
    ap.add_argument("B", help="reactant structure to search in (POSCAR/.vasp)")
    ap.add_argument("--nf-bonds", action="store_true",
                    help="bond graph with mult = 1 + nf (pre-1997cff KDB)")
    args = ap.parse_args()
    try:
        mapping, dev, note = compare(args.A, args.B, args.nf_bonds)
    except NoMatch as e:
        print(f"\nVERDICT: NO MATCH — {e}")
        sys.exit(1)
    print(f"\nVERDICT: MATCH — all {len(mapping)} atoms of A map onto B and line up "
          f"within {dev:.3f} Å ≤ dc = {DISTANCE_CUTOFF} Å{note}")


if __name__ == "__main__":
    main()
