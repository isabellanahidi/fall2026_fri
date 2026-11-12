#!/usr/bin/env python3
"""
scan_fri.py -- build a SQLite database of VASP calculations found under a folder.

For every directory that holds a structure file (POSCAR or CONTCAR):
  * no OUTCAR / OUTCAR.gz      -> never run, NOT added (only counted)
  * OUTCAR without normal end  -> status 'unfinished'
  * OUTCAR ended normally      -> status 'ended_not_converged'
  * ... and relaxation settled -> status 'converged'

Every calculation that ran is stored, with its status as a column, so the
admission rule can be changed later with a WHERE clause instead of a rescan.

Read-only on the scanned folders. Safe to stop and restart: directories that
are already in the database are skipped.

Usage:
  python3 scan_fri.py /home/wc5879/gases --db test.db
  python3 scan_fri.py /home --db fri.db --exclude in3387
"""

import argparse
import gzip
import hashlib
import os
import re
import sqlite3
import sys
import time

from ase.io import read

SCHEMA = """
CREATE TABLE IF NOT EXISTS calculations (
    id                INTEGER PRIMARY KEY,
    path              TEXT UNIQUE,   -- calculation directory
    owner             TEXT,          -- cluster username the folder belongs to
    status            TEXT,          -- converged / ended_not_converged / unfinished
    ended_normally    INTEGER,       -- 1 if VASP wrote its normal ending
    converged         INTEGER,       -- 1 if 'reached required accuracy'
    ibrion            INTEGER,       -- VASP setting: how ions are moved
    nsw               INTEGER,       -- VASP setting: max ionic steps (0 = single point)
    encut             REAL,          -- plane-wave cutoff (eV)
    ionic_steps       INTEGER,       -- ionic steps actually taken
    energy            REAL,          -- last total energy, TOTEN (eV)
    formula           TEXT,
    natoms            INTEGER,
    struct_hash       TEXT,          -- same hash = exactly the same final structure
    initial_structure TEXT,          -- POSCAR text
    final_structure   TEXT,          -- CONTCAR text (falls back to POSCAR)
    final_source      TEXT,          -- which file final_structure came from
    outcar_file       TEXT,          -- OUTCAR or OUTCAR.gz
    outcar_mtime      REAL,          -- last-modified time of the OUTCAR
    error             TEXT           -- anything that went wrong reading this folder
);
CREATE INDEX IF NOT EXISTS idx_status  ON calculations(status);
CREATE INDEX IF NOT EXISTS idx_formula ON calculations(formula);
CREATE INDEX IF NOT EXISTS idx_hash    ON calculations(struct_hash);
"""

RE_NUM = r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)"
RE_TOTEN = re.compile(r"TOTEN\s*=\s*" + RE_NUM)
RE_NSW = re.compile(r"^\s*NSW\s*=\s*(-?\d+)")
RE_IBRION = re.compile(r"^\s*IBRION\s*=\s*(-?\d+)")
RE_ENCUT = re.compile(r"^\s*ENCUT\s*=\s*" + RE_NUM)


def open_text(path):
    """Open a plain or gzipped text file."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def pick(files, name):
    """Return 'name' or 'name.gz' if present and not empty, else None."""
    for candidate in (name, name + ".gz"):
        if candidate in files:
            return candidate
    return None


def parse_outcar(path):
    """One pass through the OUTCAR, collecting everything we need."""
    info = dict(ended_normally=0, converged=0, ibrion=None, nsw=None,
                encut=None, ionic_steps=0, energy=None)
    with open_text(path) as fh:
        for line in fh:
            if "TOTEN" in line:
                # two spaces after 'free' = the once-per-ionic-step line
                if "free  energy" in line:
                    m = RE_TOTEN.search(line)
                    if m:
                        info["energy"] = float(m.group(1))
                        info["ionic_steps"] += 1
            elif "reached required accuracy" in line:
                info["converged"] = 1
            elif "General timing" in line:
                info["ended_normally"] = 1
            elif info["nsw"] is None and "NSW" in line:
                m = RE_NSW.match(line)
                if m:
                    info["nsw"] = int(m.group(1))
            elif info["ibrion"] is None and "IBRION" in line:
                m = RE_IBRION.match(line)
                if m:
                    info["ibrion"] = int(m.group(1))
            elif info["encut"] is None and "ENCUT" in line:
                m = RE_ENCUT.match(line)
                if m:
                    info["encut"] = float(m.group(1))
    return info


def structure_hash(atoms):
    """Hash that is identical only for exactly matching structures."""
    cell = [round(float(x), 3) for x in atoms.cell.array.flatten()]
    scaled = atoms.get_scaled_positions(wrap=True)
    rows = sorted(
        (s, round(float(p[0]), 3) % 1.0, round(float(p[1]), 3) % 1.0,
         round(float(p[2]), 3) % 1.0)
        for s, p in zip(atoms.get_chemical_symbols(), scaled))
    return hashlib.sha1(repr((cell, rows)).encode()).hexdigest()


def read_text(path, limit=5_000_000):
    with open_text(path) as fh:
        return fh.read(limit)


def owner_of(path, root):
    """Username = first folder under /home (or under the scan root)."""
    parts = os.path.abspath(path).split(os.sep)
    if len(parts) > 2 and parts[1] == "home":
        return parts[2]
    rel = os.path.relpath(path, root).split(os.sep)
    return rel[0] if rel and rel[0] != "." else os.path.basename(root)


def process(dirpath, files, root):
    """Build one database row for a calculation directory."""
    row = dict(path=dirpath, owner=owner_of(dirpath, root), error=None,
               formula=None, natoms=None, struct_hash=None,
               initial_structure=None, final_structure=None, final_source=None)
    errors = []

    outcar = pick(files, "OUTCAR")
    opath = os.path.join(dirpath, outcar)
    row["outcar_file"] = outcar
    try:
        row["outcar_mtime"] = os.path.getmtime(opath)
        row.update(parse_outcar(opath))
    except Exception as exc:
        row.update(ended_normally=0, converged=0, ibrion=None, nsw=None,
                   encut=None, ionic_steps=0, energy=None)
        row.setdefault("outcar_mtime", None)
        errors.append(f"OUTCAR: {exc}")

    if row["converged"] and row["ended_normally"]:
        row["status"] = "converged"
    elif row["ended_normally"]:
        row["status"] = "ended_not_converged"
    else:
        row["status"] = "unfinished"

    poscar = pick(files, "POSCAR")
    if poscar:
        try:
            row["initial_structure"] = read_text(os.path.join(dirpath, poscar))
        except Exception as exc:
            errors.append(f"POSCAR: {exc}")

    # final structure: CONTCAR if it has content, otherwise POSCAR
    for name in ("CONTCAR", "POSCAR"):
        fname = pick(files, name)
        if not fname:
            continue
        fpath = os.path.join(dirpath, fname)
        try:
            if os.path.getsize(fpath) == 0:
                continue
            atoms = read(fpath, format="vasp")
            row["final_structure"] = read_text(fpath)
            row["final_source"] = fname
            row["formula"] = atoms.get_chemical_formula()
            row["natoms"] = len(atoms)
            row["struct_hash"] = structure_hash(atoms)
            break
        except Exception as exc:
            errors.append(f"{fname}: {exc}")

    if errors:
        row["error"] = "; ".join(errors)[:500]
    return row


COLUMNS = ["path", "owner", "status", "ended_normally", "converged", "ibrion",
           "nsw", "encut", "ionic_steps", "energy", "formula", "natoms",
           "struct_hash", "initial_structure", "final_structure",
           "final_source", "outcar_file", "outcar_mtime", "error"]
INSERT = (f"INSERT OR IGNORE INTO calculations ({', '.join(COLUMNS)}) "
          f"VALUES ({', '.join(':' + c for c in COLUMNS)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder to scan, e.g. /home or /home/wc5879")
    ap.add_argument("--db", default="fri.db", help="SQLite file to write (default fri.db)")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="usernames (top-level folders under the root) to skip")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after adding this many calculations (for testing)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    db = sqlite3.connect(args.db)
    db.executescript(SCHEMA)
    done = {r[0] for r in db.execute("SELECT path FROM calculations")}
    if done:
        print(f"resuming: {len(done)} calculations already in {args.db}", flush=True)

    counts = dict(not_run=0, added=0, skipped_existing=0)
    by_status = {}
    start = time.time()

    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        if dirpath == root and args.exclude:
            dirnames[:] = [d for d in dirnames if d not in args.exclude]
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        files = set(filenames)
        if not (pick(files, "POSCAR") or pick(files, "CONTCAR")):
            continue
        if not pick(files, "OUTCAR"):
            counts["not_run"] += 1
            continue
        if dirpath in done:
            counts["skipped_existing"] += 1
            continue

        try:
            row = process(dirpath, files, root)
        except Exception as exc:  # never let one bad folder stop the scan
            print(f"  failed {dirpath}: {exc}", file=sys.stderr, flush=True)
            continue
        db.execute(INSERT, row)
        counts["added"] += 1
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1

        if counts["added"] % 100 == 0:
            db.commit()
            mins = (time.time() - start) / 60
            print(f"  {counts['added']} added  ({mins:.1f} min)  last: {dirpath}",
                  flush=True)
        if args.limit and counts["added"] >= args.limit:
            break

    db.commit()
    total = db.execute("SELECT COUNT(*) FROM calculations").fetchone()[0]
    unique = db.execute(
        "SELECT COUNT(DISTINCT struct_hash) FROM calculations").fetchone()[0]
    db.close()

    print(f"\ndone in {(time.time() - start) / 60:.1f} min")
    print(f"  never run (no OUTCAR, not added): {counts['not_run']}")
    print(f"  already in database, skipped:     {counts['skipped_existing']}")
    print(f"  added this run:                   {counts['added']}")
    for status in ("converged", "ended_not_converged", "unfinished"):
        print(f"      {status:<22}{by_status.get(status, 0)}")
    print(f"  total rows in {args.db}: {total}  (unique structures: {unique})")


if __name__ == "__main__":
    main()
