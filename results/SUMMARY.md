# KDB structural comparison: summary

Source: Henkelman group KDB (`kdb_reference/kdb/`), ported to `kdb_compare.py`.
Parameters from `config.py`: `DISTANCE_CUTOFF` dc = 0.3 Å, `NEIGHBOR_FUDGE` nf = 0.2.

## 1. The algorithm in simple steps

KDB checks whether a stored structure **A** (the database entry) can be found
inside a query structure **B** (the reactant). The check only goes one way.

1. **Element count** (`kdbquery.py:169-181`): B must have at least as many
   atoms of each element as A. If not, stop: no match.
2. **Build bond graphs** (`graph.py:21-116`): each atom becomes a node
   labelled with its element, and bonded atoms are joined by an edge.
   *In the current KDB code the bond cutoff is multiplied by 0
   (`graph.py:39`), so no bonds are ever made.*
3. **Measure distances** (`isomorphism.py:106-113`): every atom-atom distance
   in A (direct), and in B (checking 8 nearby periodic images).
4. **Choose a search order** (`isomorphism.py:124-235`): rarest elements
   first. This only affects speed, not the answer.
5. **Assign atoms** (`isomorphism.py:347-588`, the VF2 graph search): try to
   pair every atom of A with a different atom of B. Each pair must:
   - be the same element,
   - not put a mobile atom of A on a frozen atom of B,
   - have the same number of bonds (for mobile atoms),
   - keep every distance to the atoms already paired within **2·dc = 0.6 Å**,
   - agree on bonds to the atoms already paired.

   If no complete assignment exists, stop: no match.
6. **Align and score** (`kdbquery.py:305-363`): for each complete assignment,
   find the best rigid rotation of A onto B (SVD fit). The score is the
   largest squared distance between any atom and its partner. It is a match
   if **every atom lands within dc = 0.3 Å** (score ≤ dc²). Flat structures
   are also tried as their mirror image.

## 2. Pseudocode

```text
function kdb_match(A, B, dc = 0.3):
    # Step 1: element count
    for each element e in A:
        if count(B, e) < count(A, e): return NO MATCH (step 1)

    # Steps 2-3: graphs and distances
    GA, GB = bond_graph(A), bond_graph(B)        # currently no bonds (mult = 0)
    dA[i,j]  = |A_i - A_j|                       # direct distance
    dB[k,l]  = { |B_k - B_l + periodic shift| for 8 shifts }

    # Step 5: VF2 search for every complete assignment A -> B
    mappings = []
    search(partial = {}):
        if partial covers all of A: mappings.append(partial); return
        a = next atom of A in search order        # Step 4
        for each unused atom b in B:
            if element(a) != element(b):              continue
            if a mobile and b frozen:                 continue
            if a mobile and degree(a) != degree(b):   continue
            if any assigned (a', b'): no d in dB[b,b'] with |dA[a,a'] - d| < 2*dc:
                                                      continue
            if bonds to assigned atoms disagree:      continue
            search(partial + {a: b})
    if mappings is empty: return NO MATCH (step 5)

    # Step 6: rigid alignment
    for each mapping m:
        unwrap B's mapped atoms across periodic boundaries
        R = best rotation (SVD) of centred A onto centred B[m]
        score = max over atoms of |R·A_i - B_m(i)|^2
        if score <= dc^2: return MATCH
    return NO MATCH (step 6)
```

## 3. Test results

All A = `tests/structures/h2o.vasp` (Ray's water). Traces are in `results/`.

| System A | System B | Expected | Got | Decided at | Key numbers |
|---|---|---|---|---|---|
| H₂O | `h2o_copy` (exact copy) | same | **MATCH** | Step 6 | worst atom 0.000 Å ≤ 0.3 Å |
| H₂O | `h2o_rotated` (rotated 5°) | same | **MATCH** | Step 6 | worst atom 0.000 Å ≤ 0.3 Å; rotation removed by the SVD fit |
| H₂O | `h2s` (S–H 1.34 Å, 92°) | different | **NO MATCH** | Step 1 | B has no O atom |
| H₂O | `h2o_stretched` (one O–H +0.5 Å) | different | **NO MATCH** | Step 6 | passes Step 5 (0.50 Å and 0.42 Å errors < 0.6 Å); worst atom 0.315 Å > 0.3 Å |

All four results match expectations. The two "different" cases fail at
different steps, so they are failing for different reasons.

The stretched case is close to the line. From a sweep of stretch sizes:
+0.40 Å and +0.45 Å still **match**, +0.48 to +0.60 Å fail at Step 6, and
+0.65 Å fails at Step 5. So a 0.45 Å stretch, which no longer counts as a
real bond, is still accepted as "the same structure". That happens because
current KDB makes no bonds. With KDB's older bond rule (`--nf-bonds`), the
stretched case fails at Step 5 instead: O has 2 bonds in A but only 1 in B.

## 4. What we adapted from KDB, and why

| Adaptation | Why |
|---|---|
| **A is treated as the database entry, B as the query.** | KDB never compares two equal files; it looks for a stored entry inside a query. The direction matters: swapping A and B can change the result. |
| **Every atom of A is treated as mobile.** | KDB reads the mobile atoms from its database, which we don't have. Treating all atoms as mobile avoids guessing (e.g. "which atoms are the adsorbate"). |
| **Frozen atoms in B come from the POSCAR's Selective dynamics flags.** | This is how KDB gets them (`constraints[0].index`). Our test files have none. |
| **The clump order is computed from A on the fly** with KDB's own `find_clump_order` rule. | KDB normally stores this order when the entry is inserted into the database. |
| **KDB's Fortran helpers are re-written in numpy** (`pbc`, periodic-image distances, `find_clump_order`, `ordered_clump`). | KDB's Fortran module has to be compiled before it can be imported. The numpy versions follow the Fortran line by line. |
| **ASE's `PrimitiveNeighborList` replaces KDB's `aselite` copy.** | `aselite.py` is a copy of that same ASE class, and importing KDB would pull in the Fortran module. |
| **The `mult = 0` bond cutoff is kept as the default**, with an optional `--nf-bonds` flag for the older `mult = 1 + nf` rule (`graph.py:38`). | Following current KDB exactly. The flag lets us show what the bond checks do when bonds actually exist. |
| **Debug plotting removed** (`isomorphism.py:47-77`). | It saves PNG files for every comparison and has hard-coded atom numbers; it doesn't affect the result. |
| **The verdict uses the best-scoring mapping.** | KDB keeps every passing mapping because it builds a saddle/product suggestion from each. For a yes/no answer, one passing mapping is enough. |
| **Steps after alignment are left out** (overlap check, saddle/product, duplicates, database filtering). | They build suggestions; they don't decide whether two structures match. |

Notes that apply to plain molecules:

- **Every 3-atom molecule is flat**, so Step 6 always also tries the mirror
  image. For a flat molecule the mirror image is the same shape, so this is
  harmless here.
- **A's distances are direct, with no periodic wrapping** (`common.py:56`).
  This is fine for a molecule sitting whole in its box. Real KDB entries are
  stored already unwrapped. A molecule split across the cell boundary, or a
  whole periodic slab, would not compare correctly, and `kdb_compare.py`
  prints a NOTE when it detects this.
