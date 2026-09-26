"""patella_bvtv_A -- configuration-A BV/TV of the 21 patellae, with IPL's printed value beside it.

Configuration A takes IPL's own SEG and IPL's own rendered trabecular contour as inputs, so the only thing
ipldt contributes to BV/TV is the ratio itself:

    BV/TV = |SEG AND rendered trabecular contour| / |rendered trabecular contour|

computed here with validation/validate_dataset.py's bvtv(...) -- the function the radius/tibia records use
(bvtv(A_frame['seg'], ipl_G['trab'])) -- on the patella files validate_against_ipl.py names:
  SEG       <base>_SEG_decompressed.AIM (IPL's segmentation, labels 127 / 126)
  contour   Subject.gobj('trab'): IPL's TRAB_MASK rendered with /togobj_from_aim + /gobj_to_aim semantics on the
            SEG grid (validate_against_ipl.py, cached under validation/cache; identical to IPL's rendering on 21/21)

Three references are stored per subject:
  1. the same ratio taken by validate_from_ipl_contour.ipl_bvtv on the same files (the 'bvtv.ipl' entry of
     validation/results/from_ipl_contour_ceil_dt/records/<subject>.json) -- must be bitwise equal;
  2. IPL's PRINTED BV/TV from the evaluation result sheet (a PDF) in the subject's folder (three
     decimals, as IPL prints it) -- agreement is tested after rounding the ratio to three decimals;
  3. the other printed structure values of the sheet (Tb.N, Tb.Th, Tb.Sp, Ct.Th, Tb.1/N.SD, slices) for later use.  No patient identifiers are copied from the sheet.

Run:  set PYTHONUTF8=1 && python validation/patella_bvtv_A.py [--root <patella folder>] [--results <folder>]
Outputs: <results>/records.json and summary.md (default validation/results/patella_bvtv_A_run/; the published
validation/results/patella_bvtv_A/ is written only with --allow-overwrite-shipped)
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import validate_against_ipl as vai  # noqa: E402
from validate_dataset import bvtv  # noqa: E402

OUT_DIR = os.path.join(HERE, "results", "patella_bvtv_A")          # the published folder
RUN_DIR = os.path.join(HERE, "results", "patella_bvtv_A_run")      # default output of a new run
CEIL_DT = os.path.join(HERE, "results", "from_ipl_contour_ceil_dt", "records")
PRINT_KEYS = (("BV/TV", "BVTV"), ("Tb.N", "TbN"), ("Tb.Th", "TbTh"), ("Tb.Sp", "TbSp"), ("Ct.Th", "CtTh"), ("Tb.1/N.SD", "Tb1NSD"))


def pdf_text(path):
    """Text of the evaluation sheet: pypdf first, pdftotext -layout as the fallback."""
    try:
        from pypdf import PdfReader
        t = "\n".join((pg.extract_text() or "") for pg in PdfReader(path).pages)
        if "BV/TV" in t:
            return t, "pypdf"
    except Exception:
        pass
    r = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout, "pdftotext -layout"


def printed_values(folder, base):
    hits = glob.glob(os.path.join(folder, f"{base}_*.PDF")) + glob.glob(os.path.join(folder, f"{base}_*.pdf"))
    if not hits:
        return None
    t, tool = pdf_text(hits[0])
    d = dict(extracted_with=tool)
    for lab, key in PRINT_KEYS:
        m = re.search(re.escape(lab) + r"\s+(-?[0-9]+\.[0-9]+)\s*\[", t)
        d[key] = float(m.group(1)) if m else None
        d[key + "_decimals"] = (len(m.group(1).split(".")[1]) if m else None)
    m = re.search(r"Number of Slices:\s+(\d+)", t)
    d["slices"] = int(m.group(1)) if m else None
    return d


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=vai.DATA_ROOT, help="the patella data folder (non-public; validation/datapaths.py)")
    ap.add_argument("--results", default=RUN_DIR, help="output folder (default results/patella_bvtv_A_run/)")
    ap.add_argument("--allow-overwrite-shipped", action="store_true",
                    help="allow --results to be the published results/patella_bvtv_A/ (or another published folder)")
    a = ap.parse_args(argv)
    if vai.refuse_published(a.results, a.allow_overwrite_shipped):
        return 2
    out_dir = a.results
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    subjects = vai.list_subjects(a.root)
    recs = {}
    for s in subjects:
        subj = vai.Subject(s, a.root)
        seg = subj.aim("SEG")
        seg_v = dict(data=(np.asarray(seg["data"]) != 0), dim=seg["dim"], pos=seg["pos"])
        G = subj.gobj("trab")                                   # bool, on the SEG grid
        G_v = dict(data=G, dim=seg["dim"], pos=seg["pos"])
        r = bvtv(seg_v, G_v)
        rec = dict(subject=s, base=subj.base, seg_grid=[list(map(int, seg["dim"])), list(map(int, seg["pos"]))],
                   bv_voxels=r["bv"], tv_voxels=r["tv"], bvtv=r["bvtv"], bvtv_rounded_3=round(r["bvtv"], 3),
                   el_size_mm=[float(e) for e in seg["el_size_mm"]])
        # reference 1: the ceil_dt record's IPL-side ratio (same files, validate_from_ipl_contour.ipl_bvtv)
        p = os.path.join(CEIL_DT, f"{s}.json")
        if os.path.exists(p):
            b = json.load(open(p, encoding="utf-8"))["bvtv"]
            rec["ceil_dt_record"] = dict(bv_ipl=b["bv_ipl"], tv_ipl=b["tv_ipl"], ipl=b["ipl"], ours_B=b["ours"], bv_ours_B=b["bv_ours"], tv_ours_B=b["tv_ours"])
            rec["equals_ceil_dt_ipl"] = bool(b["bv_ipl"] == r["bv"] and b["tv_ipl"] == r["tv"] and b["ipl"] == r["bvtv"])
        # reference 2 / 3: the printed sheet
        pr = printed_values(subj.folder, subj.base)
        rec["printed"] = pr
        if pr and pr.get("BVTV") is not None:
            rec["printed_agrees_after_rounding"] = bool(abs(rec["bvtv_rounded_3"] - pr["BVTV"]) < 1e-9)
            rec["diff_to_printed"] = r["bvtv"] - pr["BVTV"]
            if "ceil_dt_record" in rec:
                rec["printed_agrees_after_rounding_B"] = bool(abs(round(rec["ceil_dt_record"]["ours_B"], 3) - pr["BVTV"]) < 1e-9)
        recs[s] = rec
        subj.release()
        print(f"[{time.time() - t0:6.1f}s] {s} {subj.base}: BV {r['bv']:,d} / TV {r['tv']:,d} = {r['bvtv']:.9f}"
              f"  ceil_dt {'==' if rec.get('equals_ceil_dt_ipl') else '!='}  printed {pr and pr.get('BVTV')} -> "
              f"{'agree' if rec.get('printed_agrees_after_rounding') else 'DIFFER'}", flush=True)
    n = len(recs)
    meta = dict(built=time.strftime("%Y-%m-%d %H:%M"), n=n,
                formula="BV/TV = |SEG AND rendered trabecular contour| / |rendered trabecular contour| (validation/validate_dataset.py bvtv)",
                inputs="IPL's <base>_SEG_decompressed.AIM and IPL's TRAB_MASK rendered on the SEG grid (validate_against_ipl.Subject.gobj('trab'))",
                printed_source="IPL's evaluation result sheet of each patella (three decimals; not distributed)",
                equals_ceil_dt_ipl=sum(1 for r in recs.values() if r.get("equals_ceil_dt_ipl")),
                printed_available=sum(1 for r in recs.values() if r.get("printed") and r["printed"].get("BVTV") is not None),
                printed_agree_after_rounding=sum(1 for r in recs.values() if r.get("printed_agrees_after_rounding")),
                printed_agree_after_rounding_B=sum(1 for r in recs.values() if r.get("printed_agrees_after_rounding_B")),
                max_abs_diff_to_printed=max(abs(r["diff_to_printed"]) for r in recs.values() if "diff_to_printed" in r))
    json.dump(dict(meta=meta, records=recs), open(os.path.join(out_dir, "records.json"), "w", encoding="utf-8"), indent=1)
    L = ["# Configuration-A BV/TV of the 21 patellae", "", f"Built {meta['built']}. {meta['formula']}. Inputs: {meta['inputs']}.", "",
         f"- ratio equals the ceil_dt record's IPL-side ratio (same files) on {meta['equals_ceil_dt_ipl']}/{n}",
         f"- IPL's printed BV/TV available on {meta['printed_available']}/{n}; agrees with the ratio rounded to three decimals on "
         f"{meta['printed_agree_after_rounding']}/{n} (configuration B's ratio: {meta['printed_agree_after_rounding_B']}/{n}); "
         f"max |ratio - printed| = {meta['max_abs_diff_to_printed']:.6f} (rounding bound 0.0005)",
         "", "| subject | base | BV voxels | TV voxels | BV/TV | rounded | printed | agrees | printed Tb.N | printed Tb.Th | printed Tb.Sp | printed Ct.Th | sheet slices |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in recs.items():
        pr = r.get("printed") or {}
        L.append(f"| {s} | {r['base']} | {r['bv_voxels']:,d} | {r['tv_voxels']:,d} | {r['bvtv']:.9f} | {r['bvtv_rounded_3']:.3f} | {pr.get('BVTV')} | "
                 f"{r.get('printed_agrees_after_rounding')} | {pr.get('TbN')} | {pr.get('TbTh')} | {pr.get('TbSp')} | {pr.get('CtTh')} | {pr.get('slices')} |")
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
