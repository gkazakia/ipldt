# -*- coding: utf-8 -*-
"""The pooled statistics of the paper's 137 scans, computed independently of build_facts.py as its cross-check.

The 137 scans are the 21 patellae, the 62 counted radius / tibia measurements of validation/results/<OSLH_AUTO>
and the 54 diaphyseal measurements of validation/results/<OSLH_NOEDIT> (validation/result_sets.py); for every
radius / tibia scan the reference is IPL's automatic evaluation run.  The headline statistics (configuration-A
map agreement, configuration-B SEG agreement, the configuration-B metrics) are recomputed with fig5_agreement's own
loader and estimators and written to validation/results/pooled_137_comparison.json (block n137), which
manuscript/facts/build_facts.py must reproduce.

One of the 54, Diaphyseal/BMAT/610892, ships no IPL SEG (IPL's evaluation command errored), so it enters only
the comparisons that do not need one: Ct.Th in both configurations and the configuration-B trabecular
metrics.  Every statistic below states its own n.

  python manuscript/figures/pooled_statistics.py
"""
import collections, glob, io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, HERE)
import fig5_agreement as F                                              # noqa: E402

NOEDIT = os.path.join(REPO, "validation", "results", OSLH_NOEDIT, "records")
METRICS = [("BVTV", "BV/TV", ""), ("TbTh", "Tb.Th", "mm"), ("TbSp", "Tb.Sp", "mm"),
           ("TbN", "Tb.N", "1/mm"), ("CtTh", "Ct.Th", "mm")]


def load_noedit():
    """The 54 records of the second radius / tibia set, read the way fig5_agreement.load_scans reads the first, except that a
    quantity IPL did not deliver is None instead of an error."""
    scans = []
    for p in sorted(glob.glob(os.path.join(NOEDIT, "*.json"))):
        name = os.path.basename(p)
        if not F.OSLH_GLOB.match(name):
            continue
        r = F.read_json(p)
        A, B = r.get("A") or {}, r.get("B") or {}
        mi, mo = B.get("metrics_ipl") or {}, B.get("metrics_ours") or {}

        def pair(k):
            return (float(mi[k]), float(mo[k])) if k in mi and k in mo else None

        am = A.get("maps") or {}

        def mm(k, f):
            return int(am[k][f]) if k in am else None

        seg = (B.get("seg") or {}).get("SEG")
        scans.append(dict(
            id=name[:-5], cohort=F.site_of(r["meta"]["bone_key"]),
            metrics=dict(BVTV=pair("BVTV"), TbTh=pair("TbTh_tseg"), TbSp=pair("TbSp"), TbN=pair("TbN"),
                         CtTh=pair("CtTh")),
            vox_a=dict(TbTh=mm("TRAB_TH_tseg", "mismatches"), TbSp=mm("TRAB_SP", "mismatches"),
                       TbN=mm("TRAB_1N", "mismatches"), CtTh=mm("CORT_TH", "mismatches")),
            nvox_a=dict(TbTh=mm("TRAB_TH_tseg", "n_voxels"), TbSp=mm("TRAB_SP", "n_voxels"),
                        TbN=mm("TRAB_1N", "n_voxels"), CtTh=mm("CORT_TH", "n_voxels")),
            seg_mism=(int(seg["mismatches"]) if seg else None),
            seg_dice=(float(seg["dice"]) if seg else None),
            seg_union=(int(seg["union_voxels"]) if seg else None)))
    return scans


def block(scans):
    out = {"n": len(scans), "cohorts": collections.Counter(s["cohort"] for s in scans)}
    # configuration A: differing voxels over the four maps (a map IPL did not deliver is not compared)
    pairs = [(s["vox_a"][k], s["nvox_a"][k]) for s in scans for k in s["vox_a"] if s["vox_a"][k] is not None]
    out["A_mismatch"] = sum(m for m, _ in pairs)
    out["A_compared"] = sum(n for _, n in pairs)
    out["A_exact_comparisons"] = sum(1 for m, _ in pairs if m == 0)
    out["A_comparisons"] = len(pairs)
    for k in ("TbTh", "TbSp", "TbN", "CtTh"):
        v = [s["vox_a"][k] for s in scans if s["vox_a"][k] is not None]
        out["A_exact_" + k] = sum(1 for x in v if x == 0)
        out["A_n_" + k] = len(v)
    # configuration B: SEG
    seg = [s for s in scans if s["seg_mism"] is not None]
    sm = np.array([s["seg_mism"] for s in seg], dtype=np.int64)       # voxel totals exceed 2**31: int64 on every
    su = np.array([s["seg_union"] for s in seg], dtype=np.int64)      # platform (NumPy 1.x on Windows defaults to int32)
    dc = np.array([s["seg_dice"] for s in seg])
    out["B_seg_n"] = len(seg)
    out["B_seg_mismatch"], out["B_seg_compared"] = int(sm.sum()), int(su.sum())
    out["B_seg_ppm"] = 1e6 * sm.sum() / su.sum()
    out["B_seg_exact"] = int((sm == 0).sum())
    out["B_seg_median"] = float(np.median(sm))
    out["B_dice_min"], out["B_dice_median"] = float(dc.min()), float(np.median(dc))
    out["B_dice_ge_9999"] = int((dc >= 0.9999).sum())
    # configuration B: the scalar metrics
    out["metrics"] = {}
    for k, lab, unit in METRICS:
        v = [s["metrics"][k] for s in scans if s["metrics"][k] is not None]
        ipl = np.array([a for a, _ in v])
        our = np.array([b for _, b in v])
        st = F.stats_pair(ipl, our)
        st["n"] = len(v)
        st.pop("x"); st.pop("y"); st.pop("d")
        st["ipl_mean"] = float(ipl.mean())
        st["ipl_sd"] = float(ipl.std(ddof=1))
        out["metrics"][k] = st
    return out


def show(name, b):
    print("\n%s   n = %d   %s" % (name, b["n"], dict(sorted(b["cohorts"].items()))))
    print("  configuration A, four maps: %s differing of %s compared; %d of %d scan-map comparisons exact"
          % (format(b["A_mismatch"], ","), format(b["A_compared"], ","), b["A_exact_comparisons"], b["A_comparisons"]))
    print("     exact scans per map: Tb.Th %d/%d, Tb.Sp %d/%d, 1/Tb.N %d/%d, Ct.Th %d/%d"
          % (b["A_exact_TbTh"], b["A_n_TbTh"], b["A_exact_TbSp"], b["A_n_TbSp"], b["A_exact_TbN"], b["A_n_TbN"],
             b["A_exact_CtTh"], b["A_n_CtTh"]))
    print("  configuration B, SEG: %s of %s differ (%.1f per million); identical on %d/%d; median %g per scan"
          % (format(b["B_seg_mismatch"], ","), format(b["B_seg_compared"], ","), b["B_seg_ppm"],
             b["B_seg_exact"], b["B_seg_n"], b["B_seg_median"]))
    print("     Dice min %.6f, median %.7f, >= 0.9999 on %d/%d"
          % (b["B_dice_min"], b["B_dice_median"], b["B_dice_ge_9999"], b["B_seg_n"]))
    print("  configuration B, metrics (ipldt on IPL):")
    print("     %-8s %-22s %-10s %-11s %-11s %-10s %s" % ("metric", "IPL mean +- SD", "slope", "R2", "ICC(2,1)",
                                                         "max|rel|%", "identical"))
    for k, lab, unit in METRICS:
        st = b["metrics"][k]
        print("     %-8s %9.4f +- %-9.4f %-10.6f %-11.7f %-11.7f %-10.4f %d/%d"
              % (lab, st["ipl_mean"], st["ipl_sd"], st["slope"], st["r2"], st["icc21"],
                 st["max_rel_pct"], st["identical"], st["n"]))


def main():
    first, _ = F.load_scans(oslh_dirs=(F.OSLH_DIR,))     # the 21 patellae and the first radius / tibia set
    second = load_noedit()
    ids = {s["id"] for s in first}
    dup = [s["id"] for s in second if s["id"] in ids]
    if dup:
        raise SystemExit("records in both sets: %s" % dup)
    b137 = block(first + second)
    show("POOLED, THE 137 SCANS OF THE PAPER", b137)
    json.dump(dict(n137=b137),
              io.open(os.path.join(REPO, "validation", "results", "pooled_137_comparison.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1, default=str)
    print("\nwrote validation/results/pooled_137_comparison.json")


if __name__ == "__main__":
    main()
