"""Stage the 54 diaphyseal radius / tibia measurements whose delivered evaluation carries no operator correction, at the
evaluation run that is AUTOMATIC, with the cortical pore cascade's inputs added, as normal single-version folders the harness reads unchanged.

Source   <IPLDT_LAB_ROOT>/radius_tibia/delivery/<Group>/<Sub>/<meas>/     (the local delivery, read-only)
         <IPLDT_LAB_ROOT>/radius_tibia/scanner_versions_set2/<Tag>/              (every stored version, read-only)
Decode   the internal decode of the two-run measurements (IPLDT_TWO_RUNS; not distributed)
Target   <IPLDT_LAB_ROOT>/radius_tibia/set2/<Group>/<Sub>/<meas>/
The inputs are not public; the script documents how the validation set was assembled.

What the decode established, and what this script does with it:
  51 measurements  the local delivery IS the automatic run (every product follows the run's own STEP 1 contours,
                   the second stored version of TRAB_MASK.GOBJ and the first of CORT_MASK.GOBJ, 0 voxels on 51/51;
                   the first stored trabecular contour of 47 of them is a 2022 contour-editor save the run never
                   read).  Staged: every local file, hardlinked, plus the first stored version of the fetched
                   <base>_PORE.AIM, <base>_CORT_SEG.AIM and <base>_TRAB_SEG.AIM (single version; pore_cascade of
                   those reproduces that PORE with 0 voxels on 51/51).
  353308           as the 51, except that the local <base>_CT.AIM was rendered from the second stored version of the
                   periosteal contour, an operator save 31 min into the run; IPL's evaluation read the first.
                   <base>_CT.AIM is re-rendered from the first stored version of <base>.GOBJ on the same grid.
  610892 581203 433045   the local delivery is the Script 34 REDO (run 2); run 1 is the automatic, complete,
                   internally consistent evaluation.  Staged from run 1 exactly as the internal decode lists, with the
                   two compartment renderings generated from run 1's contour objects (the fetched and local
                   _CORT_MASK_CT.AIM render the REDO's second stored version of CORT_MASK.GOBJ).  No *_DECOMPRESSED / *_COMPRESSED export
                   is staged: they are run-2 products and the oslh layout reads those names first.

Products are HARDLINKED (same NTFS volume); only the rendered rasters are new bytes.

Run:  python validation/stage_noedit_set.py [--force]
"""
import argparse, csv, io, json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datapaths import lab_path  # noqa: E402  (non-public data roots: validation/datapaths.py)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from ipldt.io import read_aim, write_aim                                   # noqa: E402
from stage_automatic_set import render_gobj                                # noqa: E402

LOCAL = os.environ.get("OSLH_ROOT") or lab_path("radius_tibia/delivery")
FETCH = lab_path("radius_tibia/scanner_versions_set2")
DST = lab_path("radius_tibia/set2")
MANIFEST = os.environ.get("IPLDT_NOEDIT_MANIFEST") or os.path.join(FETCH, "manifest.csv")   # the fetch manifest
TWO_RUNS = os.environ.get("IPLDT_TWO_RUNS") or os.path.join(FETCH, "run_decode.json")                # internal decode
PERIOSTEAL_V1 = {"Diaphyseal/REPRO/353308": 1}       # IPL's evaluation read version 1 of <base>.GOBJ; the local _CT.AIM renders version 2


def vms_name(name, version):
    """<name> with the scanner's version suffix appended (the fetch keeps every stored version of a file)."""
    return f"{name};{version}"


def link(src, dst, force):
    if os.path.exists(dst):
        if not force:
            return "kept"
        os.remove(dst)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def render_onto(gobj, grid_aim, out):
    """Render a contour object on the grid of an existing AIM, writing it with that AIM's header."""
    ref = read_aim(grid_aim)
    G = render_gobj(gobj, tuple(ref["dim"]), tuple(ref["pos"]))
    write_aim(out, G.astype(np.uint8) * 127, ref["header"])
    return int(G.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dst", default=DST)
    a = ap.parse_args()

    rows = list(csv.DictReader(io.open(MANIFEST, encoding="utf-8")))
    two = json.load(io.open(TWO_RUNS, encoding="utf-8"))
    stats = {"linked": 0, "copied": 0, "kept": 0, "rendered": 0}
    staged, problems = [], []

    for r in rows:
        mid, tag, base = r["id"], r["tag"], r["base"]
        dst = os.path.join(a.dst, *mid.split("/"))
        os.makedirs(dst, exist_ok=True)
        src_f = os.path.join(FETCH, tag)
        entry = dict(id=mid, tag=tag, base=base, dst=dst.replace("\\", "/"))

        if mid in two:                                          # run 1 from the fetch, per the decode
            e = two[mid]
            for f in e["files_to_stage"]:
                s = os.path.join(src_f, f["source"])
                t = os.path.join(dst, f["target"])
                if f["action"] == "hardlink":
                    if not os.path.exists(s):
                        problems.append(f"{mid}: missing {f['source']}"); continue
                    stats[link(s, t, a.force)] += 1
                else:                                           # render run 1's contour on the periosteal grid
                    if os.path.exists(t) and not a.force:
                        stats["kept"] += 1; continue
                    grid = os.path.join(dst, f"{base}_CT.AIM")
                    n = render_onto(s, grid, t)
                    stats["rendered"] += 1
                    entry.setdefault("rendered", {})[f["target"]] = dict(source=f["source"], voxels=n)
            entry["run"] = e["chosen_run"]
            entry["source"] = "fetch run %s (the local delivery is the Script 34 REDO)" % e["chosen_run"]
        else:                                                   # the local delivery IS the automatic run
            src_l = os.path.join(LOCAL, *mid.split("/"))
            for name in sorted(os.listdir(src_l)):
                p = os.path.join(src_l, name)
                if os.path.isfile(p):
                    stats[link(p, os.path.join(dst, name), a.force)] += 1
            for prod in ("PORE", "CORT_SEG", "TRAB_SEG"):   # TRAB_SEG: configuration A's Tb.Th object (2026-09-25)
                s = os.path.join(src_f, vms_name(f"{base}_{prod}.AIM", 1))
                if not os.path.exists(s):
                    problems.append(f"{mid}: missing {os.path.basename(s)}"); continue
                stats[link(s, os.path.join(dst, f"{base}_{prod}.AIM"), a.force)] += 1
            entry["source"] = "local delivery (automatic) + fetched PORE, CORT_SEG (version 1)"
            if mid in PERIOSTEAL_V1:
                ct = os.path.join(dst, f"{base}_CT.AIM")
                tmp = ct + ".v1"
                n = render_onto(os.path.join(src_f, vms_name(f"{base}.GOBJ", PERIOSTEAL_V1[mid])), ct, tmp)
                os.remove(ct)                                   # drop the hardlink to the version-2 rendering
                os.replace(tmp, ct)
                stats["rendered"] += 1
                entry["source"] += "; <base>_CT.AIM re-rendered from version 1 of <base>.GOBJ (%d voxels)" % n
        staged.append(entry)

    out = os.path.join(a.dst, "staged_manifest.json")
    json.dump(staged, open(out, "w", encoding="utf-8"), indent=1)
    print(f"staged {len(staged)} measurements -> {a.dst}")
    print(f"  files: {stats['linked']} hardlinked, {stats['copied']} copied, {stats['kept']} already there, "
          f"{stats['rendered']} rasters rendered")
    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for p in problems:
            print("   ", p)
        return 1
    print("  no problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
