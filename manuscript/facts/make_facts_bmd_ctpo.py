# -*- coding: utf-8 -*-
"""Write manuscript/facts/FACTS_BMD_CTPO.{md,json}: every cortical-porosity number the paper quotes, with the file
it was computed from, in the same key/value/source shape as facts.json (n = 137).  (The file name is kept for the
scripts that read it; the BMD validation is not part of the paper and not part of this public build.)

Inputs (nothing is re-run; every number is read or recomputed from these files):
  validation/results/porosity_AB_summary_n137.json
        the pooled pore-map and Ct.Po statistics of configurations A and B (validation/porosity_AB_stats.py
        --set 137), over the records of the two radius/tibia sets (validation/result_sets.py) and porosity_AB_patella.
        Every pooled number is recomputed here from those records by independent code and must agree with the
        summary, or the script stops.
  validation/results/<the same three sets>/records/*.json
        per-scan Ct.Po (full precision), the cortical-segmentation comparison and the cascade's run time
  validation/results/ipl_printed_values_137.csv   IPL's printed Ct.Po per scan (id, cohort, site, excluded, sheet_ok,
        sheet_type, Ct_Po; transcribed from IPL's result sheets by a non-public parser); a printed number is used
        only from a single-measurement sheet of the evaluation run the paper uses (sheet_ok == 'yes')

  python manuscript/facts/make_facts_bmd_ctpo.py
"""
import argparse, csv, glob, io, json, math, os, sys
from collections import Counter, OrderedDict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
R = os.path.join(REPO, "validation", "results")
sys.path.insert(0, os.path.join(REPO, "validation"))
from validate_dataset import exclusion_reason                       # noqa: E402  (the aggregators' own rule)

_ap = argparse.ArgumentParser(description="FACTS_BMD_CTPO.{md,json}")
_ap.add_argument("--set", dest="vset", choices=("137",), default="137")
_ap.add_argument("--out", default=None, help="output directory (default manuscript/facts)")
_ap.add_argument("--facts", default=None, help="the facts.json of the same set (default manuscript/facts/facts.json)")
ARGS = _ap.parse_args()
OUT_DIR = os.path.abspath(ARGS.out) if ARGS.out else HERE
os.makedirs(OUT_DIR, exist_ok=True)
FACTS = os.path.abspath(ARGS.facts) if ARGS.facts else os.path.join(HERE, "facts.json")

if ARGS.vset == "137":
    SUMMARY = os.path.join(R, "porosity_AB_summary_n137.json")
    REC_DIRS = [os.path.join(R, d) for d in (OSLH_AUTO, "porosity_AB_patella", OSLH_NOEDIT)]
    PRINTED = os.path.join(R, "ipl_printed_values_137.csv")
    XCHECK = None
    SRC_SUM = ("validation/results/porosity_AB_summary_n137.json (validation/porosity_AB_stats.py --set 137; "
               f"records of validation/results/{{{OSLH_AUTO},porosity_AB_patella,{OSLH_NOEDIT}}})")
    SRC_REC = f"validation/results/{{{OSLH_AUTO},porosity_AB_patella,{OSLH_NOEDIT}}}/records/*.json (porosity block)"
    SRC_PRI = "validation/results/ipl_printed_values_137.csv (IPL's result sheets, non-public parser), single-measurement sheets only"

F = OrderedDict()
CHECKS = []                     # (what, worst relative deviation or 'exact', ok)


def put(key, value, source, note=""):
    F[key] = OrderedDict(value=value, source=source, **({"note": note} if note else {}))


def load_csv(p, key="id"):
    return {r[key]: r for r in csv.DictReader(io.open(p, encoding="utf-8"))}


def fl(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float("nan")


# ================================================================================================ records
def cohort(r):
    """patella / ultradistal radius / ultradistal tibia / diaphyseal -- porosity_AB_stats.cohort's rule."""
    m = r.get("meta") or {}
    bone = (m.get("bone") or "").strip()
    region = (m.get("region") or "").strip().upper()
    if r.get("dataset") == "patella" or bone.lower() == "patella":
        return "patella"
    if region == "UD":
        return "ultradistal %s" % bone.lower()
    return "diaphyseal"


def site(r):
    c = cohort(r)
    return c if c != "diaphyseal" else "diaphyseal %s" % ((r.get("meta") or {}).get("bone") or "").strip().lower()


recs, skipped = [], []
for d in REC_DIRS:
    for f in sorted(glob.glob(os.path.join(d, "records", "*.json"))):
        r = json.load(io.open(f, encoding="utf-8"))
        if exclusion_reason(r):
            skipped.append(r.get("id"))
            continue
        p = r.get("porosity")
        if not p or not p.get("available"):
            skipped.append("%s (no IPL PORE)" % r.get("id"))
            continue
        recs.append(r)
ids = [r["id"] for r in recs]
assert len(ids) == len(set(ids)), "a scan in two result directories"
BY = {r["id"]: r for r in recs}
N = len(recs)
facts_json = json.load(io.open(FACTS, encoding="utf-8"))
assert N == facts_json["facts"]["cohort.n_total"]["value"], (N, facts_json["facts"]["cohort.n_total"]["value"])
assert all(r["porosity"].get(c) for r in recs for c in ("A", "B")), "a record without both configurations"


def block(cfg):
    """The pooled statistics of one configuration, recomputed from the records (independent of porosity_AB_stats.py)."""
    e = [BY[i]["porosity"][cfg] for i in ids]
    mism = np.array([x["mismatch"] for x in e], dtype=np.int64)
    ref = np.array([BY[i]["porosity"]["A"]["ct_po_ipl"] for i in ids], dtype=np.float64)   # IPL's own Ct.Po, on IPL's products
    val = np.array([x["ct_po"] for x in e], dtype=np.float64)
    d = val - ref
    out = OrderedDict(config=cfg, n=len(e), identical=int((mism == 0).sum()), mismatch_total=int(mism.sum()),
                      ours_only=int(sum(x["ours_only"] for x in e)), ipl_only=int(sum(x["ipl_only"] for x in e)),
                      max_mismatch=int(mism.max()), median_mismatch=float(np.median(mism)),
                      ipl_pore_voxels=int(sum(BY[i]["porosity"]["ipl_pore_voxels"] for i in ids)),
                      our_pore_voxels=int(sum(x["pore_voxels"] for x in e)),
                      compared_voxels=int(sum(x.get("compared_voxels", 0) for x in e)),
                      dice_min=float(min(x["dice"] for x in e)),
                      ctpo_ipl_mean=float(ref.mean()), ctpo_ipl_sd=float(ref.std(ddof=1)),
                      ctpo_mean=float(val.mean()), ctpo_sd=float(val.std(ddof=1)),
                      bias=float(d.mean()), sd_diff=float(d.std(ddof=1)), max_abs=float(np.abs(d).max()),
                      identical_ctpo=int((d == 0).sum()), max_abs_rel_pct=float(np.abs(100.0 * d / ref).max()),
                      identical_ctpo_12sf=int((np.abs(d) <= 1e-12 * np.maximum(1.0, np.abs(ref))).sum()))
    out["loa"] = [out["bias"] - 1.96 * out["sd_diff"], out["bias"] + 1.96 * out["sd_diff"]]
    X = np.column_stack([np.ones_like(ref), ref])
    coef, *_ = np.linalg.lstsq(X, val, rcond=None)
    out["intercept"], out["slope"] = float(coef[0]), float(coef[1])
    out["r2"] = float(np.corrcoef(ref, val)[0, 1] ** 2) if np.any(val != ref) else 1.0
    out["icc"] = icc21(ref, val)
    if np.any(d != 0):
        out["wilcoxon_p"] = float(stats.wilcoxon(val, ref).pvalue)
    return out


def icc21(x, y):
    m = np.column_stack([x, y])
    n, k = m.shape
    gm = m.mean()
    msr = k * ((m.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den != 0 else 1.0


def compare_block(mine, ref, what, float_tol=1e-9):
    """Every field of `ref` that `mine` also has: integers exactly, floats to float_tol relative (1e-15 absolute)."""
    worst, bad = 0.0, []
    for k, v in ref.items():
        if k in ("config", "by_cohort") or k not in mine:
            continue
        a, b = mine[k], v
        if isinstance(b, list):
            pairs = list(zip(a, b))
        else:
            pairs = [(a, b)]
        for x, y in pairs:
            if isinstance(y, int) and not isinstance(y, bool):
                if int(x) != y:
                    bad.append((k, x, y))
            else:
                x, y = float(x), float(y)
                dev = 0.0 if x == y else abs(x - y) / max(abs(x), abs(y))
                if dev > float_tol and abs(x - y) > 1e-15:
                    bad.append((k, x, y))
                elif abs(x - y) > 1e-15:
                    worst = max(worst, dev)
    CHECKS.append((what, worst, not bad))
    return bad, worst


summary = json.load(io.open(SUMMARY, encoding="utf-8"))
BLK = {c: block(c) for c in ("A", "B")}
for c in ("A", "B"):
    bad, worst = compare_block(BLK[c], summary[c], "pore.%s pooled vs %s" % (c, os.path.basename(SUMMARY)))
    if bad:
        raise SystemExit("porosity summary disagrees with the records (configuration %s): %s" % (c, bad))
    # by cohort: n, identical, mismatch, Ct.Po mean / SD / min / max
    for coh, v in summary[c]["by_cohort"].items():
        sub = [i for i in ids if cohort(BY[i]) == coh]
        mm = np.array([BY[i]["porosity"][c]["mismatch"] for i in sub])
        vv = np.array([BY[i]["porosity"][c]["ct_po"] for i in sub])
        mine = OrderedDict(n=len(sub), identical=int((mm == 0).sum()), mismatch=int(mm.sum()), ctpo_mean=float(vv.mean()),
                           ctpo_sd=float(vv.std(ddof=1)), ctpo_min=float(vv.min()), ctpo_max=float(vv.max()))
        bad, _ = compare_block(mine, v, "pore.%s.by_cohort.%s" % (c, coh))
        if bad:
            raise SystemExit("porosity summary disagrees with the records (configuration %s, %s): %s" % (c, coh, bad))

# ================================================================================================ pore map, configuration A (IPL's CORT_SEG + IPL's contour)
A = summary["A"]
put("porosity.scans", A["n"], SRC_SUM, "configuration A: IPL's cortical segmentation and IPL's rendered cortical contour into ipldt's pore cascade")
put("porosity.mismatch_total", A["mismatch_total"], SRC_SUM)
put("porosity.identical_scans", A["identical"], SRC_SUM)
put("porosity.compared_voxels", A["compared_voxels"], SRC_SUM, "summed compare.compared_voxels over the counted scans")
put("porosity.ipl_pore_voxels", A["ipl_pore_voxels"], SRC_SUM)
put("porosity.our_pore_voxels", A["our_pore_voxels"], SRC_SUM)
put("porosity.min_dice", A["dice_min"], SRC_SUM)

# ---- Ct.Po by cohort (configuration A = ipldt's value, identical to IPL's on every scan) -------------------------------------
NAME = OrderedDict([("patella", "patella"), ("ultradistal radius", "ud_radius"), ("ultradistal tibia", "ud_tibia"), ("diaphyseal", "diaphyseal")])
for coh, slug in NAME.items():
    v = A["by_cohort"][coh]
    put("ctpo.%s.n" % slug, v["n"], SRC_SUM)
    put("ctpo.%s.mean" % slug, round(v["ctpo_mean"], 4), SRC_SUM)
    put("ctpo.%s.sd" % slug, round(v["ctpo_sd"], 4), SRC_SUM)
    put("ctpo.%s.min" % slug, round(v["ctpo_min"], 4), SRC_SUM)
    put("ctpo.%s.max" % slug, round(v["ctpo_max"], 4), SRC_SUM)
for sname, slug in (("diaphyseal radius", "d_radius"), ("diaphyseal tibia", "d_tibia")):
    vv = np.array([BY[i]["porosity"]["A"]["ct_po"] for i in ids if site(BY[i]) == sname])
    put("ctpo.%s.n" % slug, int(vv.size), SRC_REC, "the diaphyseal cohort split by bone")
    put("ctpo.%s.mean" % slug, round(float(vv.mean()), 4), SRC_REC)
    put("ctpo.%s.sd" % slug, round(float(vv.std(ddof=1)), 4), SRC_REC)
    put("ctpo.%s.min" % slug, round(float(vv.min()), 4), SRC_REC)
    put("ctpo.%s.max" % slug, round(float(vv.max()), 4), SRC_REC)
allv = np.array([BY[i]["porosity"]["A"]["ct_po"] for i in ids])
put("ctpo.all.n", A["n"], SRC_SUM)
put("ctpo.all.mean", round(A["ctpo_mean"], 4), SRC_SUM)
put("ctpo.all.sd", round(A["ctpo_sd"], 4), SRC_SUM)
put("ctpo.all.min", round(float(allv.min()), 4), SRC_REC)
put("ctpo.all.max", round(float(allv.max()), 4), SRC_REC)
put("ctpo.vs_ipl_pore_map.identical_scans", A["identical_ctpo"], SRC_SUM,
    "Ct.Po on ipldt's configuration-A pore map vs the same ratio on IPL's own PORE.AIM, compared as exact floats")

# ================================================================================================ the printed sheet
pri = load_csv(PRINTED)
assert set(ids) <= set(pri), sorted(set(ids) - set(pri))
assert all(pri[i]["excluded"] in ("False", "0", "") for i in ids)
census = Counter(pri[i]["sheet_ok"] for i in ids)
single = [i for i in ids if pri[i]["sheet_ok"] == "yes"]
followup = [i for i in ids if pri[i]["sheet_ok"] == "followup"]
put("sheet.census", OrderedDict(sorted(census.items())), SRC_PRI.split(",")[0],
    "the result sheet of the evaluation run the paper uses, per counted scan: 'yes' = readable single-measurement sheet, "
    "'blank' = a blank Ghostscript stub, 'followup' = a longitudinal sheet over a matched common region of two measurements")
put("sheet.followup_ids", followup, SRC_PRI.split(",")[0], "never compared: their values belong to the common region, not to the scan")
IS_PAT = lambda i: cohort(BY[i]) == "patella"                                   # noqa: E731
hp = [i for i in single if not math.isnan(fl(pri[i], "Ct_Po"))]


def agree(o, p, unit=""):
    o, p = np.asarray(o, float), np.asarray(p, float)
    d = o - p
    out = OrderedDict(n=int(len(d)), ipl_mean=float(p.mean()), ipl_sd=float(p.std(ddof=1)) if len(d) > 1 else 0.0,
                      ours_mean=float(o.mean()), ours_sd=float(o.std(ddof=1)) if len(d) > 1 else 0.0,
                      bias=float(d.mean()), sd_diff=float(d.std(ddof=1)) if len(d) > 1 else 0.0)
    out["loa"] = [out["bias"] - 1.96 * out["sd_diff"], out["bias"] + 1.96 * out["sd_diff"]]
    out.update(max_abs=float(np.abs(d).max()), mean_rel_pct=float((100 * d / p).mean()), max_abs_rel_pct=float(np.abs(100 * d / p).max()))
    if len(d) > 2 and p.std() > 0:
        sl, ic, rr, _, _ = stats.linregress(p, o)
        out.update(slope=float(sl), intercept=float(ic), r2=float(rr ** 2), icc=icc21(p, o))
        if np.any(d != 0):
            out["wilcoxon_p"] = float(stats.wilcoxon(o, p).pvalue)
    if unit:
        out["unit"] = unit
    return out


# ---- Ct.Po against the printed sheet (configurations A and B; ipldt's full-precision value from the records) ---------------
ctpo = {c: {i: BY[i]["porosity"][c]["ct_po"] for i in ids} for c in ("A", "B")}
ok3 = {c: [i for i in hp if "%.3f" % ctpo[c][i] == pri[i]["Ct_Po"].strip()] for c in ("A", "B")}
put("ctpo.printed.sheets_with_values", len(hp), SRC_PRI)
put("ctpo.printed.sheets_patellae", sum(1 for i in hp if IS_PAT(i)), SRC_PRI)
put("ctpo.printed.reproduced_3dp", len(ok3["A"]), SRC_PRI + " + " + SRC_REC, "configuration A Ct.Po printed with three decimals equals the sheet's string")
put("ctpo.printed.reproduced_3dp_patellae", sum(1 for i in ok3["A"] if IS_PAT(i)), SRC_PRI + " + " + SRC_REC)
put("ctpo.printed.disagreeing", [OrderedDict(id=i, printed=fl(pri[i], "Ct_Po"), ours=ctpo["A"][i], ipl_map=BY[i]["porosity"]["A"]["ct_po_ipl"])
                                 for i in hp if i not in ok3["A"]], SRC_PRI + " + " + SRC_REC)
put("ctpo.printed.agreement_all", agree([ctpo["A"][i] for i in hp], [fl(pri[i], "Ct_Po") for i in hp]), SRC_PRI + " + " + SRC_REC,
    "configuration A Ct.Po (= IPL's own-map value) against the printed Ct.Po, every single sheet that prints one")
put("ctpo.printed.agreement_reproduced", agree([ctpo["A"][i] for i in ok3["A"]], [fl(pri[i], "Ct_Po") for i in ok3["A"]]), SRC_PRI + " + " + SRC_REC,
    "the same over the sheets whose three decimals are reproduced")
put("ctpo.printed.agreement_patellae", agree([ctpo["A"][i] for i in hp if IS_PAT(i)], [fl(pri[i], "Ct_Po") for i in hp if IS_PAT(i)]), SRC_PRI + " + " + SRC_REC)
put("ctpo.printed.agreement_radius_tibia", agree([ctpo["A"][i] for i in hp if not IS_PAT(i)], [fl(pri[i], "Ct_Po") for i in hp if not IS_PAT(i)]), SRC_PRI + " + " + SRC_REC)
put("ctpo.printed.radius_tibia_ids", [i for i in hp if not IS_PAT(i)], SRC_PRI)
put("ctpo.printed.B.reproduced_3dp", len(ok3["B"]), SRC_PRI + " + " + SRC_REC, "configuration B Ct.Po (ipldt's own chain) printed with three decimals equals the sheet's string")
put("ctpo.printed.B.agreement_all", agree([ctpo["B"][i] for i in hp], [fl(pri[i], "Ct_Po") for i in hp]), SRC_PRI + " + " + SRC_REC)

# ================================================================================================ the pore map in both configurations
for cfg in ("A", "B"):
    b = summary[cfg]
    for k in ("n", "identical", "mismatch_total", "ours_only", "ipl_only", "max_mismatch", "median_mismatch", "compared_voxels",
              "ipl_pore_voxels", "our_pore_voxels", "dice_min", "ctpo_ipl_mean", "ctpo_ipl_sd", "ctpo_mean", "ctpo_sd", "bias", "sd_diff",
              "loa", "max_abs", "max_abs_rel_pct", "identical_ctpo", "slope", "intercept", "r2", "icc", "wilcoxon_p"):
        if k in b:
            put("pore.%s.%s" % (cfg, k), b[k], SRC_SUM)
    put("pore.%s.identical_ctpo_12sf" % cfg, BLK[cfg]["identical_ctpo_12sf"], SRC_REC,
        "pairs identical to 12 significant digits (Figure 5 / build_facts convention); identical_ctpo counts exactly equal floats")
    for c, v in b.get("by_cohort", {}).items():
        put("pore.%s.by_cohort.%s" % (cfg, c.replace(" ", "_")), v, SRC_SUM)
    for sname in ("diaphyseal radius", "diaphyseal tibia"):
        sub = [i for i in ids if site(BY[i]) == sname]
        mm = np.array([BY[i]["porosity"][cfg]["mismatch"] for i in sub])
        vv = np.array([BY[i]["porosity"][cfg]["ct_po"] for i in sub])
        put("pore.%s.by_site.%s" % (cfg, sname.replace(" ", "_")),
            OrderedDict(n=len(sub), identical=int((mm == 0).sum()), mismatch=int(mm.sum()), ctpo_mean=float(vv.mean()), ctpo_sd=float(vv.std(ddof=1)),
                        ctpo_min=float(vv.min()), ctpo_max=float(vv.max())), SRC_REC, "the diaphyseal cohort split by bone")

# ---- what a configuration-B difference is made of: the cortical segmentation (fig6_porosity.propagation's rule) --------------
have = [i for i in ids if "CORT_SEG" in ((BY[i].get("B") or {}).get("seg") or {})]
cs = {i: BY[i]["B"]["seg"]["CORT_SEG"] for i in have}
seg_diff = [i for i in have if cs[i]["mismatches"] > 0]
pore_diff = [i for i in have if BY[i]["porosity"]["B"]["mismatch"] > 0]
put("pore.B.cort_seg.n_scans", len(have), SRC_REC + " B.seg.CORT_SEG", "scans whose record compares ipldt's cortical segmentation with IPL's CORT_SEG")
put("pore.B.cort_seg.mismatches", int(sum(cs[i]["mismatches"] for i in have)), SRC_REC + " B.seg.CORT_SEG", "voxels")
put("pore.B.cort_seg.ours_only", int(sum(cs[i]["ours_only"] for i in have)), SRC_REC + " B.seg.CORT_SEG")
put("pore.B.cort_seg.ipl_only", int(sum(cs[i]["ipl_only"] for i in have)), SRC_REC + " B.seg.CORT_SEG")
put("pore.B.cort_seg.voxels_compared", int(sum(cs[i]["union_voxels"] for i in have)), SRC_REC + " B.seg.CORT_SEG", "union grids, summed")
put("pore.B.cort_seg.exact_scans", len(have) - len(seg_diff), SRC_REC + " B.seg.CORT_SEG")
put("pore.B.cort_seg.dice_min", float(min(cs[i]["dice"] for i in have)), SRC_REC + " B.seg.CORT_SEG")
put("pore.B.cort_seg.differs", len(seg_diff), SRC_REC, "scans whose configuration-B cortical segmentation differs from IPL's")
put("pore.B.cort_seg.differs_pore_identical", sum(1 for i in seg_diff if BY[i]["porosity"]["B"]["mismatch"] == 0), SRC_REC,
    "of those, scans on which the cascade still returns IPL's pore map exactly")
put("pore.B.pore_differs_cort_seg_identical", [i for i in pore_diff if cs[i]["mismatches"] == 0], SRC_REC,
    "scans whose pore map differs although the cortical segmentation is identical (empty = a pore difference always sits on a segmentation difference)")
put("pore.B.renderings_all_identical", all(BY[i]["B"]["renderings"]["cort"]["mismatches"] == 0 for i in ids), SRC_REC + " B.renderings.cort")
# the porosity runs' SEG must be the facts sheet's SEG (same engine, same inputs): per-scan check against facts.json tables
seg_tab = {}
for row in facts_json["tables"]["B.oslh.SEG.per_scan"]:
    seg_tab[row["scan"]] = row["mismatches"]
for row in facts_json["tables"]["B.patella.SEG.per_scan"]:
    seg_tab[row["scan"]] = row["mismatches"]
seg_same = sum(1 for i in ids if seg_tab.get(BY[i].get("tag") or i.replace("/", "_"), seg_tab.get(i)) == BY[i]["B"]["seg"]["SEG"]["mismatches"])
put("pore.B.seg_equals_facts_seg", seg_same, SRC_REC + " B.seg.SEG vs manuscript/facts/facts.json tables.*.SEG.per_scan",
    "scans whose porosity-run SEG differs from IPL's by the same voxel count as the SEG of the facts sheet (of %d)" % N)
CHECKS.append(("porosity-run SEG == facts-sheet SEG per scan", 0.0 if seg_same == N else float(N - seg_same), seg_same == N))

# ---- run time of the cascade (both configurations, IPL's PORE / CORT_SEG reads and the comparisons) ---------------------------
for grp, sub in (("radius_tibia", [i for i in ids if not IS_PAT(i)]), ("patella", [i for i in ids if IS_PAT(i)])):
    t = [BY[i]["timings"]["porosity"] for i in sub]
    put("pore.time_s.%s.median" % grp, float(np.median(t)), SRC_REC + " timings.porosity",
        "both configurations' cascades with IPL's PORE and CORT_SEG reads and the comparisons, per scan, %d scans" % len(t))
    put("pore.time_s.%s.min" % grp, float(min(t)), SRC_REC + " timings.porosity")
    put("pore.time_s.%s.max" % grp, float(max(t)), SRC_REC + " timings.porosity")

put("coverage.statement",
    "Ct.Po is compared against IPL's own exported pore map on all %d counted scans and against IPL's printed value on the %d "
    "whose single-measurement result sheet prints one. For the other %d, the sheet IPL saved for the evaluation the paper "
    "uses is a blank stub (%d) or a longitudinal follow-up sheet over a matched common region of two measurements (%d)."
    % (N, len(hp), N - len(hp), census.get("blank", 0), census.get("followup", 0)), SRC_PRI)


failed = [c for c in CHECKS if not c[2]]
if failed:
    for c in failed:
        print("CHECK FAILED:", c)
    raise SystemExit("cross-checks failed: %d" % len(failed))

# ================================================================================================ write
meta = OrderedDict(built="2026-09-23", n=N, set=ARGS.vset, builder="manuscript/facts/make_facts_bmd_ctpo.py",
                   sources=OrderedDict(porosity_summary=SRC_SUM, porosity_records=SRC_REC, printed=SRC_PRI),
                   skipped=skipped,
                   crosschecks=[OrderedDict(check=c[0], worst_relative_deviation=c[1], ok=c[2]) for c in CHECKS])
json.dump(OrderedDict(meta=meta, facts=F), io.open(os.path.join(OUT_DIR, "FACTS_BMD_CTPO.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

L = ["# Facts: cortical porosity (n = %d)" % N, "",
     "Every number the paper quotes for the cortical-pore module, with the record it comes",
     "from. Regenerate with `python manuscript/facts/make_facts_bmd_ctpo.py`.", "",
     "Cross-checks run by the script (it stops on any failure): " + "; ".join(
         "%s: %s" % (c[0], "exact" if c[1] == 0 else "worst relative deviation %.1e" % c[1]) for c in CHECKS) + ".", "",
     "| key | value | source |", "|---|---|---|"]
for k, v in F.items():
    val = v["value"]
    if isinstance(val, (dict, list)):
        val = "`" + json.dumps(val, ensure_ascii=False) + "`"
    L.append("| %s | %s | %s |" % (k, val, v["source"] + ((" -- " + v["note"]) if v.get("note") else "")))
io.open(os.path.join(OUT_DIR, "FACTS_BMD_CTPO.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")

print("wrote %s and %s  (%d keys, n = %d; skipped %s)" % (os.path.join(OUT_DIR, "FACTS_BMD_CTPO.md"),
                                                         os.path.join(OUT_DIR, "FACTS_BMD_CTPO.json"), len(F), N, skipped))
print("cross-checks: %d, all passed" % len(CHECKS))
for c in CHECKS:
    print("   %-95s %s" % (c[0], "exact" if c[1] == 0 else "%.2e" % c[1]))
for k in ("porosity.scans", "porosity.mismatch_total", "porosity.compared_voxels", "porosity.ipl_pore_voxels",
          "ctpo.all.mean", "ctpo.printed.sheets_with_values", "ctpo.printed.reproduced_3dp",
          "pore.B.identical", "pore.B.mismatch_total", "pore.B.cort_seg.differs", "pore.B.cort_seg.differs_pore_identical"):
    print("  %-42s %s" % (k, F[k]["value"]))
