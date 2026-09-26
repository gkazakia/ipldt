# -*- coding: utf-8 -*-
"""Aggregate the cortical-pore results of validate_dataset over both cohorts, in both configurations.

Reads the records of the two radius / tibia sets named in validation/result_sets.py and of
validation/results/porosity_AB_patella/ (each written by `validate_dataset.py`, configurations A and B), and
reports, against IPL's own exported
<base>_PORE.AIM:

  configuration A   IPL's cortical segmentation and cortical contour in
  configuration B   the whole chain from IPL's periosteal contour: our compartment separation, our
                    rendered cortical contour and our cortical segmentation

Ct.Po is |PORE and rendered CORT_MASK| / |rendered CORT_MASK| taken on each side's own products, so the
configuration-B comparison is ipldt's Ct.Po against IPL's Ct.Po, not against a printed number.

  python validation/porosity_AB_stats.py [--set 137] [--json out.json]
"""
import argparse, glob, io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from result_sets import OSLH_AUTO, OSLH_NOEDIT                     # noqa: E402
# n = 137: the two radius / tibia sets (62 counted + 54) and the 21 patellae
DIRS_137 = [os.path.join(HERE, "results", d) for d in (OSLH_AUTO, "porosity_AB_patella", OSLH_NOEDIT)]
from validate_dataset import exclusion_reason                      # noqa: E402

def cohort(r):
    """patella / ultradistal radius / ultradistal tibia / diaphyseal, from the greyscale processing log."""
    m = r.get("meta") or {}
    bone = (m.get("bone") or "").strip()
    region = (m.get("region") or "").strip().upper()
    if r.get("dataset") == "patella" or bone.lower() == "patella":
        return "patella"
    if region == "UD":
        return "ultradistal %s" % bone.lower()
    return "diaphyseal"


def load(dirs=None):
    out, skipped = [], []
    for d in (dirs or DIRS_137):
        for f in sorted(glob.glob(os.path.join(d, "records", "*.json"))):
            r = json.load(io.open(f, encoding="utf-8"))
            if exclusion_reason(r):
                skipped.append(r.get("id")); continue
            p = r.get("porosity")
            if not p or not p.get("available"):
                skipped.append("%s (no IPL PORE)" % r.get("id")); continue
            out.append(r)
    return out, skipped


def block(recs, cfg):
    rows = [(r, r["porosity"].get(cfg)) for r in recs]
    rows = [(r, e) for r, e in rows if e]
    mism = np.array([e["mismatch"] for _, e in rows])
    ours = np.array([e["ours_only"] for _, e in rows])
    ipl = np.array([e["ipl_only"] for _, e in rows])
    ref = np.array([r["porosity"]["A"]["ct_po_ipl"] for r, _ in rows])        # IPL's own Ct.Po, always on IPL's products
    val = np.array([e["ct_po"] for _, e in rows])
    pore_ipl = np.array([r["porosity"]["ipl_pore_voxels"] for r, _ in rows])
    pore_our = np.array([e["pore_voxels"] for _, e in rows])
    d = val - ref
    out = dict(config=cfg, n=len(rows), identical=int((mism == 0).sum()),
               mismatch_total=int(mism.sum()), ours_only=int(ours.sum()), ipl_only=int(ipl.sum()),
               max_mismatch=int(mism.max()) if len(mism) else 0,
               median_mismatch=float(np.median(mism)) if len(mism) else 0.0,
               ipl_pore_voxels=int(pore_ipl.sum()), our_pore_voxels=int(pore_our.sum()),
               compared_voxels=int(sum(e.get("compared_voxels", 0) for _, e in rows)),
               dice_min=float(min(e["dice"] for _, e in rows)),
               ctpo_ipl_mean=float(ref.mean()), ctpo_ipl_sd=float(ref.std(ddof=1)),
               ctpo_mean=float(val.mean()), ctpo_sd=float(val.std(ddof=1)),
               bias=float(d.mean()), sd_diff=float(d.std(ddof=1)),
               max_abs=float(np.abs(d).max()), identical_ctpo=int((d == 0).sum()),
               max_abs_rel_pct=float(np.abs(100 * d / ref).max()))
    out["loa"] = [out["bias"] - 1.96 * out["sd_diff"], out["bias"] + 1.96 * out["sd_diff"]]
    if len(d) > 2 and ref.std() > 0:
        sl, ic, rr, _, _ = stats.linregress(ref, val)
        m = np.column_stack([ref, val]); n, k = m.shape; gm = m.mean()
        msr = k * ((m.mean(1) - gm) ** 2).sum() / (n - 1)
        msc = n * ((m.mean(0) - gm) ** 2).sum() / (k - 1)
        mse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
        out.update(slope=float(sl), intercept=float(ic), r2=float(rr ** 2),
                   icc=float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)))
        if np.any(d != 0):
            out["wilcoxon_p"] = float(stats.wilcoxon(val, ref).pvalue)
    out["by_cohort"] = {}
    for c in sorted({cohort(r) for r, _ in rows}):
        sub = [(r, e) for r, e in rows if cohort(r) == c]
        mm = np.array([e["mismatch"] for _, e in sub])
        vv = np.array([e["ct_po"] for _, e in sub])
        out["by_cohort"][c] = dict(n=len(sub), identical=int((mm == 0).sum()), mismatch=int(mm.sum()),
                                   ctpo_mean=float(vv.mean()), ctpo_sd=float(vv.std(ddof=1)) if len(vv) > 1 else 0.0,
                                   ctpo_min=float(vv.min()), ctpo_max=float(vv.max()))
    return out


def show(b):
    print("\nconfiguration %s   n = %d" % (b["config"], b["n"]))
    print("  pore map identical on %d/%d scans; %d differing voxels (%d ours-only, %d IPL-only), "
          "worst scan %d, median %g; min Dice %.6f"
          % (b["identical"], b["n"], b["mismatch_total"], b["ours_only"], b["ipl_only"],
             b["max_mismatch"], b["median_mismatch"], b["dice_min"]))
    print("  pore voxels: ipldt %s, IPL %s; voxels compared %s"
          % (format(b["our_pore_voxels"], ","), format(b["ipl_pore_voxels"], ","), format(b["compared_voxels"], ",")))
    print("  Ct.Po  IPL %.4f +- %.4f   ipldt %.4f +- %.4f   bias %+.3g   LoA [%+.3g, %+.3g]   max|d| %.3g"
          % (b["ctpo_ipl_mean"], b["ctpo_ipl_sd"], b["ctpo_mean"], b["ctpo_sd"], b["bias"],
             b["loa"][0], b["loa"][1], b["max_abs"]))
    if "slope" in b:
        print("         slope %.6f  R2 %.7f  ICC %.7f  max|rel| %.4f %%  identical %d/%d%s"
              % (b["slope"], b["r2"], b["icc"], b["max_abs_rel_pct"], b["identical_ctpo"], b["n"],
                 ("  Wilcoxon p = %.3g" % b["wilcoxon_p"]) if "wilcoxon_p" in b else ""))
    for c, v in b["by_cohort"].items():
        print("    %-20s n=%2d  identical %2d  differing %6d  Ct.Po %.4f +- %.4f  [%.4f, %.4f]"
              % (c, v["n"], v["identical"], v["mismatch"], v["ctpo_mean"], v["ctpo_sd"], v["ctpo_min"], v["ctpo_max"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "results", "porosity_AB_summary.json"))
    ap.add_argument("--set", dest="vset", choices=("137",), default="137",
                    help="137 (the default and only set): the two radius/tibia sets of validation/result_sets.py + "
                         "porosity_AB_patella (writes porosity_AB_summary_n137.json unless --json)")
    ap.add_argument("--also", nargs="*", default=[],
                    help="further result directories (validate_dataset --configs A,B) to pool with the paper's two, "
                         "e.g. validation/results/<another validate_dataset run>")
    a = ap.parse_args()
    extra = [os.path.abspath(d) for d in a.also]
    if a.vset == "137":
        if a.json == os.path.join(HERE, "results", "porosity_AB_summary.json"):
            a.json = os.path.join(HERE, "results", "porosity_AB_summary_n137.json")
        recs, skipped = load(DIRS_137 + extra)
    print("records used: %d   skipped: %d %s" % (len(recs), len(skipped), skipped if skipped else ""))
    res = {c: block(recs, c) for c in ("A", "B") if any((r["porosity"].get(c)) for r in recs)}
    for b in res.values():
        show(b)
    json.dump(res, io.open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\nwrote %s" % a.json)


if __name__ == "__main__":
    main()
