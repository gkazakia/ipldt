# -*- coding: utf-8 -*-
"""fig6_porosity.py -- Figure 6 of the ipldt / ORMIR-BQRL manuscript: the cortical pore cascade (IPL's
Burghardt, Bone 2010 block, which writes <base>_PORE.AIM) and cortical porosity Ct.Po, against IPL V5.42
on the 137 scans of the validation set (21 patellae, 29 ultradistal radii, 29 ultradistal tibiae, 29
diaphyseal radii, 29 diaphyseal tibiae).

Panels
  A  one transverse slice of a diaphyseal tibia (SCAN_A; a diaphysis because the hysteresis step of the
     cascade does nothing on a patella): the cortical compartment, IPL's PORE.AIM as a fill and ipldt's
     pore map as an outline, differing voxels in vermilion.  The counts under the panel are over the whole
     3-D volume, not the slice.
  B  a voxel-level zoom on the largest real cluster of differing voxels of the worst scan of configuration
     B, drawn the same way, so the reader sees what a difference looks like.
  C  per-scan agreement over all 137 scans: differing pore voxels per scan on a symlog axis, scans grouped
     by site and ordered by decreasing count within a site (configuration B, bars), with the
     configuration-A count of every scan in the strip below.  Open circles mark identical maps.  The two
     diaphyseal sites share the cohort color (as in Figures 3 and 5) but are bracketed separately.
  D  Ct.Po, ipldt against IPL, configuration B: one marker per scan, identity line, OLS fit, slope, R^2,
     ICC(2,1) and the number of scans whose two values are equal ('identical' is kept for voxel maps).
  E  Bland-Altman of the same pairs: difference against mean, bias and 95 % limits of agreement.
  F  Ct.Po by site: IPL's value for every scan with ipldt's overlaid, so the reader sees the range the
     agreement holds over (the two diaphyseal sites in their own columns, their Ct.Po differs by 2.5x).
Panel letters are bold capitals in the upper-left corner, as in Figures 2-5.  Every text is set in the
figure's own font (no mathtext): powers of ten with Unicode superscripts.  Slice numbers are printed
1-based, as in Figure 3 (index z is "slice z + 1 of N"); the cache and the sidecar keep the 0-based index.

CONFIGURATIONS.  A: IPL's cortical segmentation and IPL's rendered cortical contour are the input, so only
the cascade itself is tested.  B: IPL's periosteal contour is the only input -- ipldt's compartment
separation, contour rendering and Laplace-Hamming segmentation produce CORT_SEG and the cortical contour,
and the cascade runs on those.  IPL's own exported PORE.AIM is the reference in both.  Ct.Po is
|PORE and rendered CORT_MASK| / |rendered CORT_MASK| taken on each side's own products; IPL's Ct.Po is
therefore always the value on IPL's own pore map and contour (the records' porosity.A.ct_po_ipl), which is
the reference porosity_AB_stats.py uses for both configurations.

DATA (read only, nothing under ipldt/ or validation/ is modified):
  validation/results/oslh_auto_vN/records/*.json          63 radius / tibia measurements (62 counted)
  validation/results/porosity_AB_patella/records/*.json   21 patellae
  validation/results/oslh_noedit_vN/records/*.json        54 diaphyseal radius / tibia measurements
  validation/results/porosity_AB_summary_n137.json        written by validation/porosity_AB_stats.py --also
                                                          validation/results/oslh_noedit_vN; every pooled
                                                          statistic drawn here is recomputed from the records
                                                          and cross-checked against it
  validation/results/ipl_printed_values_137.csv           IPL's printed result-sheet values; a printed Ct.Po is
                                                          used only from a readable single-measurement sheet
                                                          (sheet_ok == "yes"), the rule of
                                                          manuscript/facts/make_facts_bmd_ctpo.py (quoted in the
                                                          legend only)
Diaphyseal/CKD/991161 (validate_dataset.EXCLUDED_IDS) is dropped from every count, as it is everywhere else.

The two displayed scans are recomputed here from IPL's delivered files with the shipped engine -- the
Laplace-Hamming threshold and SEG assembly of validate_dataset, then ipldt.porosity.pore_cascade -- and the
result is asserted to reproduce the validation record of that scan (SEG, CORT_SEG and pore counts) before
anything is drawn.

Run from the repository root in the `ormir` environment:

    set PYTHONUTF8=1
    python manuscript/figures/fig6_porosity.py [--recompute]

Writes manuscript/figures/fig6_porosity.png (300 dpi, 180 mm wide), fig6_porosity.svg and
fig6_porosity_numbers.json (every number the figure prints).  The two slices are cached in
manuscript/figures/cache/fig6_porosity_cache.{npz,json} (about 1 min on the first run: two Laplace-Hamming
filters and four pore cascades); --recompute redoes them.
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy import stats as sstats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "validation"))
from ipldt import ipl_ops as ops  # noqa: E402
from ipldt import porosity  # noqa: E402
from ipldt.io import read_aim  # noqa: E402

REC_DIRS = [os.path.join(REPO, "validation", "results", OSLH_AUTO),
            os.path.join(REPO, "validation", "results", "porosity_AB_patella"),
            os.path.join(REPO, "validation", "results", OSLH_NOEDIT)]
SUMMARY_JSON = os.path.join(REPO, "validation", "results", "porosity_AB_summary_n137.json")
PRINTED_CSV = os.path.join(REPO, "validation", "results", "ipl_printed_values_137.csv")


def scan_folder(r):
    """The folder of IPL's products of a scan under the non-public data roots (validation/datapaths.py): the
    published records carry no local paths, so it is rebuilt from the record's dataset and id (panels A / B only)."""
    if r.get("dataset") == "patella":
        return os.path.join(lab_path("PFJOA/XCT_masks_full_grab"), r["id"]).replace("\\", "/")
    for root in (lab_path("Cross_validation_IPL/OS_LH_AUTO"), lab_path("Cross_validation_IPL/OS_LH_NOEDIT")):
        p = os.path.join(root, *r["id"].split("/")).replace("\\", "/")
        if os.path.isdir(p):
            return p
    return os.path.join(lab_path("Cross_validation_IPL/OS_LH_AUTO"), *r["id"].split("/")).replace("\\", "/")
CACHE = os.path.join(REPO, "manuscript", "figures", "cache", "fig6_porosity_cache")
OUT = os.path.join(HERE, "fig6_porosity")

#: The validation set this figure is drawn for (manuscript/facts/facts.json cohort.site.*.n).
EXPECT_N = 137
EXPECT_SITES = {"patella": 21, "ultradistal radius": 29, "ultradistal tibia": 29, "diaphyseal radius": 29,
                "diaphyseal tibia": 29}

#: The scan of panel A.  A diaphysis is chosen because the hysteresis step of the cascade is a no-op on a
#: patella -- the weak-evidence label it grows into is empty on every patella of the cohort
#: (ipldt/porosity.py) -- and this is the scan Supplementary Figure S9 follows stage by stage, so the two
#: figures show the same measurement.  The panel-B scan is not fixed here: it is whichever scan the records make worst.
SCAN_A = "Diaphyseal/CKD/2422"
ZOOM = 44                       # panel B: the window drawn, in voxels
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def fmt(n):
    """An integer with comma thousands separators, as in the legends and the text."""
    return f"{int(n):,d}"


# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito (color-blind safe), the palette and the cohort assignment of Figures 3 and 5
COHORTS = ["patella", "ultradistal radius", "ultradistal tibia", "diaphyseal"]
COHORT_SHORT = {"patella": "patella", "ultradistal radius": "UD radius", "ultradistal tibia": "UD tibia",
                "diaphyseal": "diaphyseal"}
COHORT_COLOR = {"patella": "#0072B2", "ultradistal radius": "#009E73", "ultradistal tibia": "#E69F00",
                "diaphyseal": "#CC79A7"}
COHORT_MARKER = {"patella": "o", "ultradistal radius": "^", "ultradistal tibia": "s", "diaphyseal": "D"}
COHORT_LEGEND = {"patella": "patella", "ultradistal radius": "ultradistal radius",
                 "ultradistal tibia": "ultradistal tibia", "diaphyseal": "diaphyseal radius / tibia"}
# Panels C and F group by site: the two diaphyseal sites are 29 scans each at n = 137, and their Ct.Po differs
# by a factor of 2.5, so they get their own bracket / column -- in the one cohort color, as in Figure 3.
SITES = ["patella", "ultradistal radius", "ultradistal tibia", "diaphyseal radius", "diaphyseal tibia"]
SITE_COHORT = {"patella": "patella", "ultradistal radius": "ultradistal radius",
               "ultradistal tibia": "ultradistal tibia", "diaphyseal radius": "diaphyseal",
               "diaphyseal tibia": "diaphyseal"}
SITE_SHORT = {"patella": "patella", "ultradistal radius": "UD radius", "ultradistal tibia": "UD tibia",
              "diaphyseal radius": "D radius", "diaphyseal tibia": "D tibia"}
SITE_TICK = {"patella": "patella", "ultradistal radius": "UD\nradius", "ultradistal tibia": "UD\ntibia",
             "diaphyseal radius": "D\nradius", "diaphyseal tibia": "D\ntibia"}
C_BLUE, C_VERM = "#0072B2", "#D55E00"     # Okabe-Ito blue and vermilion, as in Figures 2-5
INK, INK2, INK3, GRID, AXIS = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#e3e3e3", "#c3c2b7"
C_COMPART = "#e9e8e3"           # the cortical compartment
C_BONE = "#b9b8b3"              # cortical bone (CORT_SEG)
C_PORE = "#5d5c58"              # IPL's PORE.AIM

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
    "text.color": INK, "savefig.dpi": 300, "figure.dpi": 100, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.unicode_minus": True,
})
FIG_W = 180.0
FIG_H = 196.0                   # set for real in draw()


# ------------------------------------------------------------------------------------------------ records
def cohort_of(r):
    """patella / ultradistal radius / ultradistal tibia / diaphyseal -- porosity_AB_stats.cohort, verbatim."""
    m = r.get("meta") or {}
    bone = (m.get("bone") or "").strip()
    region = (m.get("region") or "").strip().upper()
    if r.get("dataset") == "patella" or bone.lower() == "patella":
        return "patella"
    if region == "UD":
        return "ultradistal %s" % bone.lower()
    return "diaphyseal"


def load_records():
    """Every record of the two runs that porosity_AB_stats.py counts, with its excluded / unavailable list."""
    from validate_dataset import exclusion_reason
    kept, skipped = [], []
    for d in REC_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "records", "*.json"))):
            r = json.load(io.open(f, encoding="utf-8"))
            if exclusion_reason(r):
                skipped.append(dict(id=r.get("id"), reason="excluded from every aggregate"))
                continue
            p = r.get("porosity")
            if not p or not p.get("available"):
                skipped.append(dict(id=r.get("id"), reason="no IPL PORE.AIM"))
                continue
            kept.append(r)
    return kept, skipped


def site_name(r):
    """The readable site of a scan: 'patella', 'ultradistal radius', 'diaphyseal tibia', ...  The cohort
    pools the two diaphyseal sites, so a panel head that names one scan needs the bone as well."""
    m = r.get("meta") or {}
    bone = (m.get("bone") or "").strip().lower()
    region = (m.get("region") or "").strip().upper()
    if r.get("dataset") == "patella" or bone == "patella":
        return "patella"
    return f"{'ultradistal' if region == 'UD' else 'diaphyseal'} {bone}"


def scan_rows(recs):
    """One row per scan: cohort, both configurations' pore comparison, and the Ct.Po pair."""
    rows = []
    for r in recs:
        p = r["porosity"]
        row = dict(id=r["id"], tag=r.get("tag") or r["id"].replace("/", "_"), base=r["base"],
                   dataset=r["dataset"], folder=scan_folder(r), cohort=cohort_of(r),
                   ipl_pore=int(p["ipl_pore_voxels"]), ctpo_ipl=float(p["A"]["ct_po_ipl"]),
                   site=site_name(r), seg_mismatch=int(r["B"]["seg"]["SEG"]["mismatches"]),
                   cort_seg_mismatch=(int(r["B"]["seg"]["CORT_SEG"]["mismatches"])
                                      if "CORT_SEG" in r["B"]["seg"] else None),
                   rendering_mismatch=int(r["B"]["renderings"]["cort"]["mismatches"]))
        for cfg in ("A", "B"):
            e = p[cfg]
            row[cfg] = dict(mismatch=int(e["mismatch"]), ours_only=int(e["ours_only"]),
                            ipl_only=int(e["ipl_only"]), dice=float(e["dice"]),
                            pore=int(e["pore_voxels"]), ctpo=float(e["ct_po"]),
                            compared=int(e["compared_voxels"]), contour=int(e["contour_voxels"]),
                            same_grid=bool(e["same_grid"]))
        rows.append(row)
    order = {c: i for i, c in enumerate(COHORTS)}
    rows.sort(key=lambda s: (order[s["cohort"]], s["id"]))
    return rows


def icc21(x, y):
    """ICC(2,1), two-way random effects, absolute agreement, single measurement -- the form
    porosity_AB_stats.block uses (and manuscript/facts/build_facts.py::icc21)."""
    m = np.column_stack([x, y])
    n, k = m.shape
    gm = m.mean()
    msr = k * ((m.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def propagation(rows):
    """What a configuration-B difference is made of: ipldt's cortical segmentation is the only input of the
    cascade that configuration B changes (the rendered contour is IPL's own on every scan, asserted in
    rebuild_scan), so a pore map can differ only where CORT_SEG does.  Measured here, never assumed."""
    have = [s for s in rows if s["cort_seg_mismatch"] is not None]
    seg_diff = [s for s in have if s["cort_seg_mismatch"] > 0]
    pore_diff = [s for s in have if s["B"]["mismatch"] > 0]
    unexplained = [s["id"] for s in pore_diff if s["cort_seg_mismatch"] == 0]
    absorbed = [s for s in seg_diff if s["B"]["mismatch"] == 0]
    out = dict(n_with_cort_seg=len(have), cort_seg_differs=len(seg_diff), pore_differs=len(pore_diff),
               cort_seg_differs_pore_identical=len(absorbed), pore_differs_cort_seg_identical=unexplained,
               renderings_all_identical=all(s["rendering_mismatch"] == 0 for s in rows))
    if unexplained:
        out["sentence"] = (f"ipldt's cortical segmentation differs from IPL's on {len(seg_diff)} of "
                           f"{len(have)} scans; {len(unexplained)} scans whose pore map differs have an "
                           f"identical cortical segmentation")
    else:
        out["sentence"] = (f"the pore map differs only on scans whose cortical segmentation (CORT_SEG) differs: "
                           f"{len(seg_diff)} of {len(have)} scans, and on {len(absorbed)} of "
                           f"those the cascade still returns IPL's map exactly")
    return out


def block(rows, cfg):
    """The pooled statistics of one configuration, recomputed from the records exactly as
    validation/porosity_AB_stats.py::block defines them."""
    mism = np.array([s[cfg]["mismatch"] for s in rows])
    ref = np.array([s["ctpo_ipl"] for s in rows])                 # IPL's own Ct.Po, on IPL's own products
    val = np.array([s[cfg]["ctpo"] for s in rows])
    d = val - ref
    out = dict(config=cfg, n=len(rows), identical=int((mism == 0).sum()), mismatch_total=int(mism.sum()),
               ours_only=int(sum(s[cfg]["ours_only"] for s in rows)),
               ipl_only=int(sum(s[cfg]["ipl_only"] for s in rows)),
               max_mismatch=int(mism.max()), median_mismatch=float(np.median(mism)),
               ipl_pore_voxels=int(sum(s["ipl_pore"] for s in rows)),
               our_pore_voxels=int(sum(s[cfg]["pore"] for s in rows)),
               compared_voxels=int(sum(s[cfg]["compared"] for s in rows)),
               dice_min=float(min(s[cfg]["dice"] for s in rows)),
               ctpo_ipl_mean=float(ref.mean()), ctpo_ipl_sd=float(ref.std(ddof=1)),
               ctpo_mean=float(val.mean()), ctpo_sd=float(val.std(ddof=1)),
               bias=float(d.mean()), sd_diff=float(d.std(ddof=1)), max_abs=float(np.abs(d).max()),
               identical_ctpo=int((d == 0).sum()), max_abs_rel_pct=float(np.abs(100 * d / ref).max()),
               # porosity_AB_stats.py counts an exactly zero difference; Figure 5 and build_facts.py call a
               # pair identical when it agrees to 12 significant digits.  Both are reported; the figure
               # prints the exact one.
               identical_ctpo_12sf=int((np.abs(d) <= 1e-12 * np.maximum(1.0, np.abs(ref))).sum()))
    out["loa"] = [out["bias"] - 1.96 * out["sd_diff"], out["bias"] + 1.96 * out["sd_diff"]]
    sl, ic, rr, _, _ = sstats.linregress(ref, val)
    out.update(slope=float(sl), intercept=float(ic), r2=float(rr ** 2), icc=icc21(ref, val))
    if np.any(d != 0):
        out["wilcoxon_p"] = float(sstats.wilcoxon(val, ref).pvalue)
    def group(sub):
        mm = np.array([s[cfg]["mismatch"] for s in sub])
        vv = np.array([s[cfg]["ctpo"] for s in sub])
        ii = np.array([s["ctpo_ipl"] for s in sub])
        return dict(n=len(sub), identical=int((mm == 0).sum()), mismatch=int(mm.sum()),
                    ctpo_mean=float(vv.mean()),
                    ctpo_sd=float(vv.std(ddof=1)) if len(vv) > 1 else 0.0,
                    ctpo_min=float(vv.min()), ctpo_max=float(vv.max()),
                    ctpo_ipl_mean=float(ii.mean()),
                    ctpo_ipl_sd=float(ii.std(ddof=1)) if len(ii) > 1 else 0.0,
                    ctpo_ipl_min=float(ii.min()), ctpo_ipl_max=float(ii.max()))

    out["by_cohort"] = {c: group([s for s in rows if s["cohort"] == c]) for c in COHORTS
                        if any(s["cohort"] == c for s in rows)}
    out["by_site"] = {c: group([s for s in rows if s["site"] == c]) for c in SITES
                      if any(s["site"] == c for s in rows)}
    # the numbers the legend quotes beyond porosity_AB_stats.py's block
    dice = np.array([s[cfg]["dice"] for s in rows])
    out["dice_ge_0_999"] = int((dice >= 0.999).sum())
    top3 = sorted(rows, key=lambda s: (-s[cfg]["mismatch"], s["id"]))[:3]
    out["top3"] = [dict(id=s["id"], mismatch=s[cfg]["mismatch"]) for s in top3]
    out["top3_mismatch"] = int(sum(s[cfg]["mismatch"] for s in top3))
    out["same_grid"] = int(sum(1 for s in rows if s[cfg]["same_grid"]))
    out["bias_pct_of_mean"] = float(100 * out["bias"] / out["ctpo_ipl_mean"])
    out["ctpo_ipl_min_all"] = float(ref.min())
    out["ctpo_ipl_max_all"] = float(ref.max())
    out["ctpo_ipl_min_id"] = rows[int(np.argmin(ref))]["id"]
    out["ctpo_ipl_fold_range"] = float(ref.max() / ref.min())
    return out


def cross_check(stats, path=SUMMARY_JSON):
    """Compare every recomputed pooled statistic with porosity_AB_stats.py's own, and report -- never hide --
    a disagreement."""
    if not os.path.exists(path):
        say(f"  WARNING {os.path.relpath(path, REPO)} is not on disk: no cross-check of the pooled statistics")
        return dict(available=False)
    ref = json.load(io.open(path, encoding="utf-8"))
    keys = ["n", "identical", "mismatch_total", "ours_only", "ipl_only", "max_mismatch", "median_mismatch",
            "ipl_pore_voxels", "our_pore_voxels", "compared_voxels", "dice_min", "ctpo_ipl_mean",
            "ctpo_ipl_sd", "ctpo_mean", "ctpo_sd", "bias", "sd_diff", "max_abs", "identical_ctpo",
            "max_abs_rel_pct", "slope", "intercept", "r2", "icc"]
    out = dict(available=True, file=os.path.relpath(path, REPO).replace("\\", "/"), worst=0.0, disagree=[])
    for cfg, b in stats.items():
        if cfg not in ref:
            out["disagree"].append(dict(config=cfg, key="(whole block)", mine=None, theirs=None))
            continue
        for k in keys:
            if k not in ref[cfg] or k not in b:
                continue
            a, c = float(b[k]), float(ref[cfg][k])
            err = abs(a - c) / max(1e-300, abs(c)) if c != 0 else abs(a - c)
            out["worst"] = max(out["worst"], err)
            if err > 1e-9:
                out["disagree"].append(dict(config=cfg, key=k, mine=a, theirs=c))
                say(f"  DISAGREEMENT configuration {cfg} {k}: this figure {a!r}, porosity_AB_stats.py {c!r}")
    say(f"  cross-check against {out['file']}: worst relative deviation {out['worst']:.2e}, "
        f"{len(out['disagree'])} disagreements")
    return out


def sheet_table(rows):
    """IPL's printed Ct.Po against configuration A, quoted in the legend only.  A printed value is read only
    from a readable single-measurement sheet (sheet_ok == "yes"; a 'followup' sheet prints the common region
    of two measurements, a 'blank' one nothing) -- the rule of manuscript/facts/make_facts_bmd_ctpo.py -- and
    it is reproduced when ipldt's full-precision Ct.Po printed with three decimals equals the sheet's string."""
    if not os.path.exists(PRINTED_CSV):
        return None
    pri = {r["id"]: r for r in csv.DictReader(io.open(PRINTED_CSV, encoding="utf-8"))}
    missing = [s["id"] for s in rows if s["id"] not in pri]
    assert not missing, f"scans without a row in {PRINTED_CSV}: {missing}"
    census = {}
    for s in rows:
        k = pri[s["id"]]["sheet_ok"]
        census[k] = census.get(k, 0) + 1
    with_sheet = [s for s in rows if pri[s["id"]]["sheet_ok"] == "yes" and pri[s["id"]]["Ct_Po"].strip()]
    ok = [s for s in with_sheet if f"{s['A']['ctpo']:.3f}" == pri[s["id"]]["Ct_Po"].strip()]
    return dict(table=os.path.relpath(PRINTED_CSV, REPO).replace("\\", "/"), scans=len(rows),
                census=dict(sorted(census.items())), sheets=len(with_sheet),
                sheets_patellae=sum(1 for s in with_sheet if s["cohort"] == "patella"), matches=len(ok),
                disagreeing=[s["id"] for s in with_sheet if s not in ok])


# ------------------------------------------------------------------------------------- the displayed scans
def load_inputs(row):
    """IPL's delivered files of one measurement: (periosteal raster, cortical contour rendering, trabecular
    contour rendering, CORT_SEG, PORE, grayscale, element size, the grid the cascade runs its contour on).

    OS_LH: the renderings are files (<base>_CT / _CORT_MASK_CT / _TRAB_MASK_CT).  Patella: no rendering is on
    disk, so the raw masks are rendered here exactly as ipldt.porosity's docstring and validate_dataset do,
    onto the CORT_MASK | TRAB_MASK union box."""
    import validate_dataset as vd
    from ipldt.contour import render_volume
    f, b = row["folder"], row["base"]

    def aim(name):
        return read_aim(os.path.join(f, name))

    grey = aim(f"{b}.AIM")
    if row["dataset"] == "oslh":
        per = vd.set_vol(aim(f"{b}_CT.AIM"))
        g_cort = vd.set_vol(aim(f"{b}_CORT_MASK_CT.AIM"))
        tn = f"{b}_TRAB_MASK_CORR_CT.AIM"
        g_trab = vd.set_vol(aim(tn if os.path.exists(os.path.join(f, tn)) else f"{b}_TRAB_MASK_CT.AIM"))
        cort_seg = aim(f"{b}_CORT_SEG.AIM")
        pore = aim(f"{b}_PORE.AIM")
    else:
        raw_c, raw_t = aim(f"{b}_CORT_MASK_decompressed.AIM"), aim(f"{b}_TRAB_MASK_decompressed.AIM")
        import validate_from_ipl_contour as vfc
        per = vd.set_vol(vfc.periosteal_from_ipl(raw_c, raw_t))
        g_cort = vd.bvol(render_volume(np.asarray(raw_c["data"]) != 0), raw_c["dim"], raw_c["pos"])
        g_trab = vd.bvol(render_volume(np.asarray(raw_t["data"]) != 0), raw_t["dim"], raw_t["pos"])
        cort_seg = aim(f"{b}_CORT_SEG_decompressed.AIM")
        pore = aim(f"{b}_PORE_decompressed.AIM")
    return dict(grey=grey, per=per, g_cort=g_cort, g_trab=g_trab, cort_seg=cort_seg, pore=pore,
                el=float(grey["el_size_mm"][0]))


def char_on(v, grid=None):
    """A 0 / non-zero volume as IPL's char mask (0 / 127), optionally pasted onto `grid` -- validate_dataset's
    own _char, which the slice-wise rule of the cascade is sensitive to."""
    d, p = tuple(v["dim"]), tuple(v["pos"])
    c = dict(data=(np.asarray(v["data"]) != 0).astype(np.uint8) * 127, dim=d, pos=p)
    if grid is None:
        return c
    return ops.vol(ops.on_grid(c, tuple(grid[0]), tuple(grid[1])), tuple(grid[0]), tuple(grid[1]))


def rebuild_scan(row, rec):
    """Both configurations of the pore cascade on one measurement, from IPL's delivered files, with the
    shipped engine.  Every count is asserted against that scan's validation record before it is drawn."""
    import validate_dataset as vd
    import validate_from_ipl_contour as vfc
    say(f"{row['id']}: reading IPL's files")
    I = load_inputs(row)
    grid = (tuple(rec["porosity"]["contour_grid"]["dim"]), tuple(rec["porosity"]["contour_grid"]["pos"]))
    gdim, gpos = tuple(int(v) for v in I["grey"]["dim"]), tuple(int(v) for v in I["grey"]["pos"])
    el3 = tuple(float(e) for e in I["grey"]["el_size_mm"])

    # ---- configuration A: IPL's CORT_SEG and IPL's rendered cortical contour
    cr = char_on(I["g_cort"], grid)
    poreA = porosity.pore_cascade(cr, char_on(I["cort_seg"]))["pore"]
    cmpA = porosity.compare_pore(poreA, I["pore"])
    ctpoA = porosity.ct_po(poreA, cr)["ct_po"]
    say(f"  [A] pore {fmt(cmpA['ours'])} voxels, {cmpA['mismatch']} differing, Ct.Po {ctpoA:.6f}")

    # ---- configuration B: ipldt's Laplace-Hamming CORT_SEG, on ipldt's own cortical rendering
    v = rec["variants"]
    bm = vd.laplace_hamming_threshold(I["grey"]["data"], el3, border=v["lh_border"], keep_border=True,
                                      pad_offset=v["lh_pad_offset"], dtype=v["lh_dtype"],
                                      lp_cut_off_freq=v["lp_cut_off_freq"])
    variant = rec["seg_variant_used"]["value"]
    s = vd.assemble_seg(bm, (gdim, gpos), I["per"], I["g_cort"], I["g_trab"], variant,
                        trab_masked=(rec["B"]["seg"]["trab_seg_mask"] == "gobj"))
    del bm
    # ipldt's rendered cortical contour is IPL's on every scan of the set (the record carries the count);
    # assert it
    assert row["rendering_mismatch"] == 0, (row["id"], row["rendering_mismatch"])
    poreB = porosity.pore_cascade(cr, char_on(s["CORT_SEG"]))["pore"]
    cmpB = porosity.compare_pore(poreB, I["pore"])
    ctpoB = porosity.ct_po(poreB, cr)["ct_po"]
    say(f"  [B] pore {fmt(cmpB['ours'])} voxels, {cmpB['mismatch']} differing, Ct.Po {ctpoB:.6f}")

    # ---- the self-check: this must be the run the records describe, voxel for voxel
    seg_name = f"{row['base']}_SEG.AIM" if row["dataset"] == "oslh" else f"{row['base']}_SEG_decompressed.AIM"
    cs = vfc.compare_masks(s["SEG"], read_aim(os.path.join(row["folder"], seg_name)))
    checks = dict(seg_mismatches=(cs["mismatches"], rec["B"]["seg"]["SEG"]["mismatches"]),
                  A_mismatch=(cmpA["mismatch"], rec["porosity"]["A"]["mismatch"]),
                  A_pore=(cmpA["ours"], rec["porosity"]["A"]["pore_voxels"]),
                  B_mismatch=(cmpB["mismatch"], rec["porosity"]["B"]["mismatch"]),
                  B_pore=(cmpB["ours"], rec["porosity"]["B"]["pore_voxels"]))
    for k, (mine, theirs) in checks.items():
        assert int(mine) == int(theirs), f"{row['id']} {k}: recomputed {mine}, record {theirs}"
    for cfg, got in (("A", ctpoA), ("B", ctpoB)):
        want = rec["porosity"][cfg]["ct_po"]
        assert abs(got - want) <= 1e-12 * max(1.0, abs(want)), f"{row['id']} Ct.Po {cfg}: {got} vs {want}"
    say(f"  self-check: SEG, both pore maps and both Ct.Po values reproduce the validation record")

    G = lambda vol: ops.on_grid(vol, grid[0], grid[1]) != 0                       # noqa: E731
    return dict(compartment=G(cr), cort_seg=G(I["cort_seg"]), ipl=G(I["pore"]), ours_A=G(poreA),
                ours_B=G(poreB), el=I["el"], grid=grid,
                counts=dict(A=cmpA, B=cmpB, ct_po_A=ctpoA, ct_po_B=ctpoB,
                            ct_po_ipl=float(rec["porosity"]["A"]["ct_po_ipl"]),
                            seg_mismatches=int(cs["mismatches"]),
                            cort_seg_mismatches=row["cort_seg_mismatch"]))


def slice_crop(vol, margin=4):
    """The (y, x) bounding box of a boolean slice, grown by `margin` and clipped."""
    ys, xs = np.nonzero(vol)
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, vol.shape[0])
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, vol.shape[1])
    return y0, y1, x0, x1


def pick_slice_A(S):
    """The transverse slice of the panel-A scan with the most IPL pore voxels: the most informative one."""
    per_slice = S["ipl"].reshape(S["ipl"].shape[0], -1).sum(1)
    return int(np.argmax(per_slice)), per_slice


def pick_cluster_B(S, win=ZOOM):
    """The largest 3-D cluster of voxels on which ipldt's configuration-B pore map and IPL's disagree, and a
    window of `win` voxels centered on its densest slice."""
    diff = S["ours_B"] != S["ipl"]
    lab, n = ndi.label(diff, structure=np.ones((3, 3, 3), bool))
    cnt = np.bincount(lab.ravel())
    cnt[0] = 0
    k = int(np.argmax(cnt))
    zz, yy, xx = np.nonzero(lab == k)
    per_slice = np.bincount(zz, minlength=diff.shape[0])
    z = int(np.argmax(per_slice))
    sel = zz == z
    cy, cx = int(round(yy[sel].mean())), int(round(xx[sel].mean()))
    y0 = int(np.clip(cy - win // 2, 0, diff.shape[1] - win))
    x0 = int(np.clip(cx - win // 2, 0, diff.shape[2] - win))
    zs = np.nonzero(diff.reshape(diff.shape[0], -1).sum(1))[0]
    return dict(z=z, y0=y0, x0=x0, win=win, clusters=int(n), cluster_voxels=int(cnt[k]),
                cluster_in_slice=int(per_slice[z]), diff_in_slice=int(diff[z].sum()),
                cluster_z_range=[int(zz.min()), int(zz.max())],
                slices_with_a_difference=int(zs.size), diff_z_range=[int(zs.min()), int(zs.max())],
                largest_clusters=[int(c) for c in np.sort(cnt)[::-1][:5]])


def compute_display(rows, recs_by_id, scan_a, scan_b):
    """The two crops the figure draws, plus every count printed beside them."""
    N, A = {}, {}
    for key, sid in (("A", scan_a), ("B", scan_b)):
        row = next(s for s in rows if s["id"] == sid)
        S = rebuild_scan(row, recs_by_id[sid])
        phrase = row["site"]
        if key == "A":
            z, per_slice = pick_slice_A(S)
            y0, y1, x0, x1 = slice_crop(S["compartment"][z], margin=4)
            sl = (z, slice(y0, y1), slice(x0, x1))
            N["A"] = dict(id=sid, base=row["base"], cohort=row["cohort"], site_phrase=phrase,
                          el=S["el"], slice=z,
                          slices=int(S["compartment"].shape[0]), crop=[y0, y1, x0, x1],
                          ipl_pore_in_slice=int(per_slice[z]),
                          compartment_voxels=int(S["compartment"].sum()),
                          cort_seg_voxels=int(S["cort_seg"].sum()), **S["counts"])
        else:
            w = pick_cluster_B(S)
            sl = (w["z"], slice(w["y0"], w["y0"] + w["win"]), slice(w["x0"], w["x0"] + w["win"]))
            N["B"] = dict(id=sid, base=row["base"], cohort=row["cohort"], site_phrase=phrase,
                          el=S["el"], window=w,
                          slices=int(S["compartment"].shape[0]),
                          ipl_pore_in_window=int(S["ipl"][sl].sum()),
                          diff_in_window=int((S["ours_B"] != S["ipl"])[sl].sum()),
                          ours_only_in_window=int((S["ours_B"] & ~S["ipl"])[sl].sum()),
                          ipl_only_in_window=int((~S["ours_B"] & S["ipl"])[sl].sum()), **S["counts"])
        for k in ("compartment", "cort_seg", "ipl", "ours_A", "ours_B"):
            A[f"{key}_{k}"] = np.ascontiguousarray(S[k][sl])
        del S
    return N, A


# ------------------------------------------------------------------------------------------------ drawing
_MEAS = [None]


def measure_mm(s, fontsize, weight="normal"):
    """The drawn width of a string in millimeters, from the renderer (not an estimate)."""
    if _MEAS[0] is None:
        _MEAS[0] = plt.figure(figsize=(2, 2))
    f = _MEAS[0]
    t = f.text(0, 0, s, fontsize=fontsize, fontweight=weight)
    w = t.get_window_extent(renderer=f.canvas.get_renderer()).width / f.dpi * 25.4
    t.remove()
    return w


def wrap_mm(s, width_mm, fontsize, weight="normal"):
    """Greedy word wrap to width_mm at the given font, measured with the renderer; '\\n' is a hard break."""
    out = []
    for para in s.split("\n"):
        line = ""
        for word in para.split(" "):
            cand = (line + " " + word).strip()
            if line and measure_mm(cand, fontsize, weight) > width_mm:
                out.append(line)
                line = word
            else:
                line = cand
        out.append(line)
    return out


SUPERSCRIPT = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")
# Arial has the superscript digits but not the superscript minus (U+207B): a text that carries a power of
# ten names this family list, so that one glyph falls back to DejaVu Sans and everything else stays Arial
FONT_SUP = ["Arial", "DejaVu Sans"]


def pow10(e):
    """-4 -> '10' + superscript '-4' (Unicode superscripts, set in the figure's own font)."""
    return "10" + str(int(e)).translate(SUPERSCRIPT)


def sci(v, digits=2):
    """6.04e-06 -> '6.0 x 10' + superscript '-6' in plain Unicode (the figure's own font, no mathtext);
    0 -> '0'."""
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = f"{v / 10 ** e:.{digits - 1}f}".replace("-", "\u2212")
    return f"{m} \u00d7 {pow10(e)}"


def ax_mm(fig, x, y_top, w, h, **kw):
    """Axes at (x, y_top) millimeters from the top-left corner of the figure, w x h mm."""
    return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H], **kw)


def ftext(fig, x, y_top, s, **kw):
    kw.setdefault("va", "top")
    return fig.text(x / FIG_W, 1 - y_top / FIG_H, s, **kw)


def panel_letter(fig, x, y, s):
    return ftext(fig, x, y, s, fontsize=9, fontweight="bold", ha="left", color=INK)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(AXIS)
        s.set_linewidth(0.4)


def scale_bar(ax, mm, vox_mm, label, frac_x=0.05, frac_y=0.05, color=INK):
    """A horizontal scale bar of `mm` millimeters on an axes drawn in voxel units (origin upper)."""
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()[1], ax.get_ylim()[0]          # origin='upper': ylim is (high, low)
    W = abs(x1 - x0)
    H = abs(y0 - y1)
    nb = mm / vox_mm
    xb = min(x0, x1) + W * frac_x
    yb = max(y0, y1) - H * frac_y
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    ax.plot([xb, xb + nb], [yb, yb], color=color, lw=1.4, solid_capstyle="butt", zorder=8,
            path_effects=halo)
    ax.text(xb + nb / 2, yb - H * 0.022, label, ha="center", va="bottom", fontsize=7, color=color,
            zorder=8, path_effects=halo)


def pore_image(ax, compartment, cort_seg, ipl, ours, el, bar, lw=0.45, grid_lines=False):
    """The drawing shared by panels A and B: the compartment pale, cortical bone gray, IPL's pore map as a
    fill, ipldt's as a blue outline, and the voxels on which the two differ in vermilion.  The image and the
    contour share the array's own index coordinates, so the outline sits on the voxels it belongs to."""
    rgb = np.ones(compartment.shape + (3,))
    rgb[compartment] = hex_rgb(C_COMPART)
    rgb[cort_seg] = hex_rgb(C_BONE)
    rgb[ipl] = hex_rgb(C_PORE)
    diff = ipl != ours
    rgb[diff] = hex_rgb(C_VERM)
    h, w = compartment.shape
    ax.imshow(rgb, interpolation="nearest", origin="upper", extent=(-0.5, w - 0.5, h - 0.5, -0.5))
    if lw:
        ax.contour(np.arange(w), np.arange(h), ours.astype(float), levels=[0.5], colors=[C_BLUE],
                   linewidths=lw)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    if grid_lines:
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(True, which="minor", color="white", lw=0.25)
        ax.tick_params(which="minor", length=0)
    frame(ax)
    if bar:
        scale_bar(ax, bar[0], el, bar[1])
    return int(diff.sum())


def draw(rows, stats, prop, N, A, sheets, check):
    """The whole page, laid out in millimeters from its top-left corner (the geometry of S1-S9)."""
    global FIG_H
    checks = []
    B_, A_ = stats["B"], stats["A"]
    a, b = N["A"], N["B"]
    w = b["window"]

    # ------------------------------------------------------------------ the columns, in millimeters
    x0 = 3.0
    gap = 5.0
    col_w = (FIG_W - 2 * x0 - gap) / 2                      # A and B
    img = 46.0
    leg_w = col_w - img - 2.0
    w3 = (FIG_W - 2 * x0 - 2 * 6.0) / 3                     # D, E, F
    xs3 = [x0 + k * (w3 + 6.0) for k in range(3)]
    xb = x0 + col_w + gap
    c_left = 20.5                                           # left margin of the panel-C axes (room for the 7-pt strip label)
    c_w = FIG_W - c_left - 3.0
    NOTE_DY, DESC_FS, NOTE_FS = 3.0, 7.0, 7.0

    # ------------------------------------------------------------------ every description, wrapped first
    D_ = {
        "A": ("The pore map on one slice",
              f"{a['site_phrase']} {a['id']} ({a['el'] * 1000:.1f} µm voxels), slice {a['slice'] + 1} of "
              f"{a['slices']}: the slice with the most pore voxels in IPL's map"),
        "B": ("What a difference looks like",
              f"the largest cluster of differing voxels of the worst scan of configuration B "
              f"({b['site_phrase']} {b['id']}), {w['win']} × {w['win']} voxels of slice {w['z'] + 1} of "
              f"{b['slices']}; same colors as A, one square per voxel"),
        "C": ("Differing pore voxels per scan, both configurations",
              "every scan, grouped by site and ordered by decreasing configuration-B count within a "
              "site; the strip below gives the configuration-A count of the same scan"),
        "D": ("Ct.Po, ipldt against IPL",
              "configuration B, one marker per scan; IPL's Ct.Po is its own pore map over its own rendered "
              "cortical contour"),
        "E": ("Bland–Altman of Ct.Po",
              "the same pairs: difference against mean, with the bias and the 95% limits of agreement, "
              "bias ± 1.96 SD"),
        "F": ("Ct.Po by site",
              "IPL's value for every scan (filled) with ipldt's overlaid (open ring): the range over which "
              "the agreement holds; UD ultradistal, D diaphyseal"),
    }
    WID = dict(A=col_w, B=col_w, C=FIG_W - 2 * x0, D=w3, E=w3, F=w3)
    DESC = {k: wrap_mm(v[1], WID[k] - 3.6, DESC_FS) for k, v in D_.items()}

    def desc_h(n):
        return 4.3 + n * DESC_FS / 72.0 * 25.4 * 1.15 + 0.7

    h1 = desc_h(max(len(DESC[k]) for k in "AB"))
    h2 = desc_h(len(DESC["C"]))
    h3 = desc_h(max(len(DESC[k]) for k in "DEF"))

    # ------------------------------------------------------------------ every note, wrapped first
    NT = {
        "A": [(f"whole volume: {fmt(a['A']['ipl'])} pore voxels in IPL's map, {fmt(a['B']['ours'])} in "
               f"ipldt's, Ct.Po {a['ct_po_B']:.4f} against IPL's {a['ct_po_ipl']:.4f}", False, INK2),
              (f"{a['A']['mismatch']} differing voxels in configuration A and {a['B']['mismatch']} in "
               f"configuration B, of {fmt(a['A']['compared_voxels'])} compared "
               f"(Dice {a['B']['dice']:.6f})", True, INK if a["B"]["mismatch"] == 0 else C_VERM)],
        "B": [(f"this window: {fmt(b['ipl_pore_in_window'])} of IPL's pore voxels, {b['diff_in_window']} differing "
               f"({b['ours_only_in_window']} ipldt only, {b['ipl_only_in_window']} IPL only)", False, INK2),
              (f"whole volume: {fmt(b['B']['mismatch'])} differing voxels of "
               f"{fmt(b['B']['compared_voxels'])} compared, on {w['slices_with_a_difference']} of "
               f"{b['slices']} slices, in {fmt(w['clusters'])} clusters of which this is the largest "
               f"({fmt(w['cluster_voxels'])} voxels, slices {w['cluster_z_range'][0] + 1}–"
               f"{w['cluster_z_range'][1] + 1}); Dice {b['B']['dice']:.6f}; Ct.Po {b['ct_po_B']:.4f} against "
               f"IPL's {b['ct_po_ipl']:.4f}", True, INK)],
    }
    NT = {k: [(ln, bold, col) for s, bold, col in v for ln in wrap_mm(s, col_w, NOTE_FS,
                                                                     "bold" if bold else "normal")]
          for k, v in NT.items()}
    coh_lines = []
    for c in SITES:
        v = B_["by_site"].get(c)
        if not v:
            continue
        coh_lines.append(f"{c} (n = {v['n']}): IPL Ct.Po {v['ctpo_ipl_mean']:.4f} ± "
                         f"{v['ctpo_ipl_sd']:.4f}, range {v['ctpo_ipl_min']:.4f}–{v['ctpo_ipl_max']:.4f}"
                         f"; configuration-B pore map identical to IPL's on {v['identical']}/{v['n']} scans")

    # ------------------------------------------------------------------ the rows, in millimeters
    y_r1 = 1.0
    y_img1 = y_r1 + h1
    y_note1 = y_img1 + img + 1.8
    y_r2 = y_note1 + max(len(NT["A"]), len(NT["B"])) * NOTE_DY + 3.4
    bar_h, strip_h = 33.0, 4.0
    y_bars = y_r2 + h2
    y_strip = y_bars + bar_h + 1.4
    y_brack = y_strip + strip_h + 1.2
    y_legC = y_brack + 4.6
    y_r3 = y_legC + 5.0 + 2.6
    ax3 = 38.0
    y_ax3 = y_r3 + h3
    y_note3 = y_ax3 + ax3 + 9.2
    FIG_H = y_note3 + len(coh_lines) * NOTE_DY + 2.4
    say(f"layout: rows at {y_r1:.1f} / {y_r2:.1f} / {y_r3:.1f} mm, figure {FIG_W:.0f} × {FIG_H:.1f} mm")

    fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4))

    def head(key, x, yt, wd):
        panel_letter(fig, x, yt, key)
        t1 = ftext(fig, x + 3.6, yt + 0.2, D_[key][0], fontsize=7.5, fontweight="bold", ha="left")
        t2 = ftext(fig, x, yt + 4.3, "\n".join(DESC[key]), fontsize=DESC_FS, color=INK2, ha="left",
                   linespacing=1.15)
        checks.extend([(t1, x + 3.6, x + wd, D_[key][0]), (t2, x, x + wd, " ".join(DESC[key]))])

    def notes(x, yt, wd, lines, fs=NOTE_FS):
        for i, (s, bold, col) in enumerate(lines):
            t = ftext(fig, x, yt + i * NOTE_DY, s, fontsize=fs, ha="left", color=col,
                      fontweight="bold" if bold else "normal")
            checks.append((t, x, x + wd, s))

    def legend_at(x, yt, wd, ht, handles, **kw):
        axl = ax_mm(fig, x, yt, wd, ht)
        axl.axis("off")
        kw.setdefault("frameon", False)
        kw.setdefault("borderaxespad", 0)
        kw.setdefault("handletextpad", 0.5)
        axl.legend(handles=handles, **kw)
        return axl

    # ============================================================================== A: the slice
    head("A", x0, y_r1, col_w)
    ha, wa = A["A_compartment"].shape
    img_a = float(np.clip(img * ha / wa, 0.5 * img, img))        # the crop's own aspect: no white padding
    ax = ax_mm(fig, x0, y_img1 + (img - img_a) / 2, img, img_a)
    d_slice = pore_image(ax, A["A_compartment"], A["A_cort_seg"], A["A_ipl"], A["A_ours_B"], a["el"],
                         (5.0, "5 mm"), lw=0.2)
    legend_at(x0 + img + 2.0, y_img1 + 2.0, leg_w, 24.0,
              [patches.Patch(fc=C_COMPART, ec=AXIS, lw=0.4, label="cortical compartment"),
               patches.Patch(fc=C_BONE, ec="none", label="cortical bone (CORT_SEG)"),
               patches.Patch(fc=C_PORE, ec="none", label="IPL's PORE.AIM"),
               Line2D([], [], color=C_BLUE, lw=0.9, label="ipldt's pore map (outline)"),
               patches.Patch(fc=C_VERM, ec="none", label="differing voxels")],
              loc="upper left", ncol=1, fontsize=7, handlelength=1.1, labelspacing=0.5)
    notes(x0, y_note1, col_w, NT["A"])

    # ============================================================================== B: the zoom
    head("B", xb, y_r1, col_w)
    ax = ax_mm(fig, xb, y_img1, img, img)
    d_zoom = pore_image(ax, A["B_compartment"], A["B_cort_seg"], A["B_ipl"], A["B_ours_B"], b["el"],
                        (0.5, "0.5 mm"), lw=0.45, grid_lines=True)
    legend_at(xb + img + 2.0, y_img1 + 2.0, leg_w, 24.0,
              [patches.Patch(fc=C_PORE, ec="none", label="in both pore maps"),
               patches.Patch(fc=C_VERM, ec="none", label="in one map only"),
               patches.Patch(fc=C_BONE, ec="none", label="cortical bone"),
               patches.Patch(fc=C_COMPART, ec=AXIS, lw=0.4, label="rest of the compartment"),
               Line2D([], [], color=C_BLUE, lw=0.9, label="ipldt's pore map")],
              loc="upper left", ncol=1, fontsize=7, handlelength=1.1, labelspacing=0.5)
    notes(xb, y_note1, col_w, NT["B"])

    # ============================================================================== C: per-scan agreement
    head("C", x0, y_r2, FIG_W - 2 * x0)
    order, xpos, groups = [], [], []
    x = 0.0
    present = [c for c in COHORTS if any(s["cohort"] == c for s in rows)]
    present_sites = [c for c in SITES if any(s["site"] == c for s in rows)]
    for c in present_sites:
        rs = sorted([s for s in rows if s["site"] == c], key=lambda s: (-s["B"]["mismatch"], s["id"]))
        start = x
        for s in rs:
            order.append(s)
            xpos.append(x)
            x += 1.0
        groups.append((c, len(rs), start, x - 1.0))
        x += 2.5 if len(rs) > 4 else 5.0
    xpos = np.array(xpos)
    mb = np.array([s["B"]["mismatch"] for s in order], float)
    cols = [COHORT_COLOR[s["cohort"]] for s in order]
    xlim = (-1.4, groups[-1][3] + 1.4)
    # 137 scans across ~158 mm leave ~1.05 mm per scan: the identical-map marker must stay inside its slot
    slot_mm = c_w / (xlim[1] - xlim[0])
    ms_zero = float(min(3.0, 0.80 * slot_mm / 0.3528))             # points; 0.80 of a slot, at most 3 pt
    say(f"panel C: {len(order)} scans, {slot_mm:.2f} mm per scan, identical-map marker {ms_zero:.2f} pt")

    axc = ax_mm(fig, c_left, y_bars, c_w, bar_h)
    axc.bar(xpos, mb, width=0.8, color=cols, lw=0)
    zero = mb == 0
    axc.plot(xpos[zero], np.zeros(int(zero.sum())), "o", ms=ms_zero, mfc="white", mec=INK2, mew=0.5, zorder=3)
    axc.set_yscale("symlog", linthresh=1.0, linscale=0.45)
    # symlog: a bar of value v sits at (linscale + log10 v) / (linscale + log10 top) of the axes height.
    # Solve that for the top which puts the tallest bar at 0.55, and round it up to a power of ten.
    vmax = max(float(mb.max()), 10.0)
    top = 10.0 ** np.ceil((0.45 + np.log10(vmax)) / 0.55 - 0.45)
    axc.set_ylim(-0.5, top)
    # the axis runs well above the tallest bar so the summary has a clear band, but a tick is drawn only up
    # to one decade above it: the empty band carries no labels that mean nothing
    tick_top = 10.0 ** np.ceil(np.log10(vmax))
    ticks = [t for t in (0, 1, 10, 100, 1e3, 1e4, 1e5, 1e6) if t <= tick_top]
    axc.yaxis.set_major_locator(FixedLocator(ticks))
    axc.yaxis.set_major_formatter(FixedFormatter([fmt(t) for t in ticks]))
    axc.yaxis.set_minor_locator(NullLocator())
    axc.set_xlim(*xlim)
    axc.xaxis.set_major_locator(NullLocator())
    axc.grid(True, axis="y", color=GRID, lw=0.4)
    axc.set_axisbelow(True)
    for sp in ("top", "right"):
        axc.spines[sp].set_visible(False)
    axc.set_ylabel("differing voxels per scan", labelpad=2, fontsize=7)
    summary = (f"configuration B (bars): {fmt(B_['mismatch_total'])} differing voxels "
               f"({fmt(B_['ours_only'])} ipldt only, {fmt(B_['ipl_only'])} IPL only) of "
               f"{fmt(B_['compared_voxels'])} compared, over {fmt(B_['ipl_pore_voxels'])} pore voxels in "
               f"IPL's maps; identical on {B_['identical']}/{B_['n']} scans, median "
               f"{B_['median_mismatch']:.0f} per scan, worst {fmt(B_['max_mismatch'])}, "
               f"smallest Dice {B_['dice_min']:.6f}\n"
               f"configuration A (strip): {fmt(A_['mismatch_total'])} differing voxels of "
               f"{fmt(A_['compared_voxels'])} compared; identical on {A_['identical']}/{A_['n']} scans\n"
               + prop["sentence"])
    lines = []
    for para in summary.split("\n"):
        lines += wrap_mm(para, c_w - 2.0, NOTE_FS)
    t = axc.text(0.004, 0.98, "\n".join(lines), transform=axc.transAxes, fontsize=NOTE_FS, va="top",
                 ha="left", color=INK, linespacing=1.3)
    checks.append((t, c_left, FIG_W - 3.0, " ".join(lines)))

    axs = ax_mm(fig, c_left, y_strip, c_w, strip_h)
    axs.set_xlim(*xlim)
    axs.set_ylim(0, 1)
    axs.set_yticks([])
    axs.xaxis.set_major_locator(NullLocator())
    for sp in axs.spines.values():
        sp.set_visible(False)
    n_a_nonzero = 0
    for i, s in enumerate(order):
        if s["A"]["mismatch"] == 0:
            axs.plot([xpos[i]], [0.5], "o", ms=ms_zero, mfc="white", mec=INK2, mew=0.5)
        else:
            n_a_nonzero += 1
            axs.bar([xpos[i]], [1.0], width=0.8, color=C_VERM, lw=0)
    t = ftext(fig, c_left - 1.2, y_strip + strip_h / 2 + 1.0, "configuration A", fontsize=NOTE_FS,
              ha="right", va="center", color=INK2)
    checks.append((t, 0.5, c_left - 1.0, "configuration A"))
    for c, n_, xa, xb_ in groups:                     # the brackets and their labels, in figure millimeters
        xa_mm = c_left + c_w * (xa - 0.4 - xlim[0]) / (xlim[1] - xlim[0])
        xb_mm = c_left + c_w * (xb_ + 0.4 - xlim[0]) / (xlim[1] - xlim[0])
        fig.add_artist(Line2D([xa_mm / FIG_W, xb_mm / FIG_W], [1 - y_brack / FIG_H] * 2,
                              transform=fig.transFigure, color=INK2, lw=0.6))
        lab = f"{c} ({n_})"
        t = ftext(fig, (xa_mm + xb_mm) / 2, y_brack + 1.0, lab, fontsize=NOTE_FS, ha="center", color=INK)
        checks.append((t, xa_mm - 0.5, xb_mm + 0.5, lab))
    legend_at(x0, y_legC, FIG_W - 2 * x0, 5.0,
              [patches.Patch(fc=COHORT_COLOR[c], ec="none",
                             label=f"{COHORT_LEGEND[c]} (n = {B_['by_cohort'][c]['n']})") for c in present]
              + [Line2D([], [], marker="o", ms=3.0, mfc="white", mec=INK2, mew=0.6, ls="none",
                        label="0 differing voxels (identical)")],
              loc="upper center", ncol=len(present) + 1, fontsize=7, handlelength=1.1, handleheight=0.8,
              columnspacing=1.0)

    # ============================================================================== D: Ct.Po regression
    head("D", xs3[0], y_r3, w3)
    axd = ax_mm(fig, xs3[0] + 9.5, y_ax3, ax3, ax3)
    ref = np.array([s["ctpo_ipl"] for s in rows])
    val = np.array([s["B"]["ctpo"] for s in rows])
    hi = float(max(ref.max(), val.max())) * 1.08
    axd.plot([0, hi], [0, hi], ls="--", lw=0.7, color=INK3, zorder=1)
    if B_["identical_ctpo"] < B_["n"]:
        axd.plot([0, hi], [B_["intercept"], B_["slope"] * hi + B_["intercept"]], lw=0.7, color=INK, zorder=2)
    for c in present[::-1]:           # the 58 diaphyseal markers first, so they do not hide the rest
        idx = [i for i, s in enumerate(rows) if s["cohort"] == c]
        axd.scatter(ref[idx], val[idx], s=11, marker=COHORT_MARKER[c], facecolors=COHORT_COLOR[c],
                    edgecolors="white", linewidths=0.35, zorder=3)
    axd.set_xlim(0, hi)
    axd.set_ylim(0, hi)
    axd.set_box_aspect(1)
    axd.locator_params(nbins=4)
    axd.set_xlabel("IPL Ct.Po", labelpad=2)
    axd.set_ylabel("ipldt Ct.Po", labelpad=2)
    for sp in ("top", "right"):
        axd.spines[sp].set_visible(False)
    # configuration A is drawn only when it is not the identity, which its differing-voxel count decides
    a_identity = A_["identical_ctpo"] == A_["n"]
    if not a_identity:
        axd.scatter(ref, np.array([s["A"]["ctpo"] for s in rows]), s=9, marker="x", c=INK3, linewidths=0.5,
                    zorder=2)
    # R^2 and ICC to 7 decimals, as the facts sheet quotes them (6 would round 0.9999988 up to 0.999999)
    # scalars are 'equal', never 'identical' (the paper keeps that word for voxel maps with 0 differing voxels)
    txt = (f"slope {B_['slope']:.6f}\n$R^2$ {B_['r2']:.7f}\nICC {B_['icc']:.7f}\n"
           f"equal {B_['identical_ctpo']}/{B_['n']}\n"
           + (f"configuration A: equal\non {A_['identical_ctpo']}/{A_['n']} scans" if a_identity else
              f"configuration A (×): equal\non {A_['identical_ctpo']}/{A_['n']}"))
    axd.text(0.98, 0.03, txt, transform=axd.transAxes, ha="right", va="bottom", fontsize=NOTE_FS,
             color=INK, linespacing=1.15)

    # ============================================================================== E: Bland-Altman
    head("E", xs3[1], y_r3, w3)
    axe = ax_mm(fig, xs3[1] + 9.5, y_ax3, ax3, ax3)
    d = val - ref
    m = (val + ref) / 2
    scale = max(abs(B_["loa"][0]), abs(B_["loa"][1]), B_["max_abs"])
    e = int(np.floor(np.log10(scale))) if scale > 0 else -6
    f_ = 10.0 ** e
    axe.axhline(0, lw=0.5, color=INK3, zorder=1)
    axe.axhline(B_["bias"] / f_, lw=0.8, color=INK, zorder=2)
    for y in B_["loa"]:
        axe.axhline(y / f_, lw=0.7, ls="--", color=INK, zorder=2)
    for c in present[::-1]:           # as in D: the diaphyseal markers underneath
        idx = [i for i, s in enumerate(rows) if s["cohort"] == c]
        axe.scatter(m[idx], d[idx] / f_, s=11, marker=COHORT_MARKER[c], facecolors=COHORT_COLOR[c],
                    edgecolors="white", linewidths=0.35, zorder=3)
    lo0 = min(float(d.min()) / f_, B_["loa"][0] / f_)
    hi0 = max(float(d.max()) / f_, B_["loa"][1] / f_)
    pad = 0.12 * (hi0 - lo0)
    lo1, hi1 = lo0 - pad, hi0 + pad
    axe.set_ylim(lo1, lo1 + (hi1 - lo1) / 0.66)          # the data in the bottom two thirds, text above
    axe.set_xlim(0, float(m.max()) * 1.08)
    axe.set_box_aspect(1)
    axe.locator_params(axis="x", nbins=4)
    axe.locator_params(axis="y", nbins=5)
    axe.set_xlabel("mean of IPL and ipldt Ct.Po", labelpad=2)
    axe.set_ylabel(f"ipldt \u2212 IPL Ct.Po (\u00d7{pow10(e)})", labelpad=2, fontfamily=FONT_SUP)
    for sp in ("top", "right"):
        axe.spines[sp].set_visible(False)
    axe.text(0.035, 0.97, f"bias {sci(B_['bias'])}\nLoA {sci(B_['loa'][0])} to {sci(B_['loa'][1])}\n"
                          f"largest |difference| {sci(B_['max_abs'])}\n"
                          f"({B_['max_abs_rel_pct']:.4f}% of IPL's value)",
             transform=axe.transAxes, ha="left", va="top", fontsize=NOTE_FS, color=INK, linespacing=1.3,
             fontfamily=FONT_SUP)

    # ============================================================================== F: Ct.Po by cohort
    head("F", xs3[2], y_r3, w3)
    axf = ax_mm(fig, xs3[2] + 9.5, y_ax3, ax3, ax3)
    rng = np.random.default_rng(6)
    for i, c in enumerate(present_sites):
        sub = [s for s in rows if s["site"] == c]
        coh = SITE_COHORT[c]
        vi = np.array([s["ctpo_ipl"] for s in sub])
        vo = np.array([s["B"]["ctpo"] for s in sub])
        j = (rng.random(vi.size) - 0.5) * 0.56
        axf.scatter(i + j, vo, s=17, marker="o", facecolors="none", edgecolors=INK, linewidths=0.35, zorder=3)
        axf.scatter(i + j, vi, s=7, marker=COHORT_MARKER[coh], facecolors=COHORT_COLOR[coh],
                    edgecolors="white", linewidths=0.25, zorder=4)
        axf.plot([i - 0.38, i + 0.38], [vi.mean(), vi.mean()], lw=0.9, color=INK, zorder=5)
    axf.set_xticks(range(len(present_sites)))
    axf.set_xticklabels([SITE_TICK[c] for c in present_sites])
    axf.set_xlim(-0.6, len(present_sites) - 0.4)
    axf.set_ylim(0, hi)
    axf.set_box_aspect(1)
    axf.set_ylabel("Ct.Po", labelpad=2)
    axf.grid(True, axis="y", color=GRID, lw=0.4)
    axf.set_axisbelow(True)
    for sp in ("top", "right"):
        axf.spines[sp].set_visible(False)
    axf.legend(handles=[Line2D([], [], marker="o", ms=3.2, mfc=INK3, mec="white", mew=0.3, ls="none",
                               label="IPL"),
                        Line2D([], [], marker="o", ms=4.4, mfc="none", mec=INK, mew=0.4, ls="none",
                               label="ipldt"),
                        Line2D([], [], color=INK, lw=0.9, label="site mean (IPL)")],
               loc="upper right", frameon=False, fontsize=7, handlelength=1.0, labelspacing=0.32,
               handletextpad=0.5, borderaxespad=0.15)

    for i, ln in enumerate(coh_lines):
        t = ftext(fig, x0, y_note3 + i * NOTE_DY, ln, fontsize=NOTE_FS, ha="left", color=INK2)
        checks.append((t, x0, FIG_W - x0, ln))

    bad = check_text_widths(fig, checks)
    printed_slices = dict(A_slice=a["slice"] + 1, B_slice=w["z"] + 1,
                          B_cluster_slices=[w["cluster_z_range"][0] + 1, w["cluster_z_range"][1] + 1],
                          note="1-based, as printed; 'slice', 'z' and '*_z_range' under A and B are 0-based indices")
    P = dict(A=a, B=b, printed_slices=printed_slices,
             C=dict(configuration_A=A_, configuration_B=B_, config_A_scans_with_a_difference=n_a_nonzero,
                    propagation=prop,
                    per_scan=[dict(id=s["id"], cohort=s["cohort"], site=s["site"], A=s["A"]["mismatch"],
                                   B=s["B"]["mismatch"], dice_B=s["B"]["dice"], ctpo_ipl=s["ctpo_ipl"],
                                   ctpo_B=s["B"]["ctpo"], seg_mismatch=s["seg_mismatch"],
                                   cort_seg_mismatch=s["cort_seg_mismatch"]) for s in order]),
             D=dict(slope=B_["slope"], intercept=B_["intercept"], r2=B_["r2"], icc=B_["icc"],
                    identical=B_["identical_ctpo"], n=B_["n"], ctpo_ipl_max=float(ref.max()),
                    ctpo_ours_max=float(val.max())),
             E=dict(bias=B_["bias"], sd_diff=B_["sd_diff"], loa=B_["loa"], max_abs=B_["max_abs"],
                    max_abs_rel_pct=B_["max_abs_rel_pct"], exponent=e, wilcoxon_p=B_.get("wilcoxon_p")),
             F=dict(by_site=B_["by_site"], by_cohort=B_["by_cohort"]),
             slice_differing_voxels=dict(A=d_slice, B=d_zoom),
             sheet_cross_check=sheets, stats_cross_check=check)
    return fig, P, bad


def check_text_widths(fig, items):
    """Two checks, as in S1-S9: every registered text inside the span it was given, and every text of the
    figure -- axes texts, tick labels and legends included -- inside the page."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for t, xa, xb, label in items:
        bb = t.get_window_extent(renderer=r)
        lo, hi = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        if lo < xa - 0.3 or hi > xb + 0.3:
            bad += 1
            say(f"  WARNING text overflows its {xb - xa:.1f} mm span by {max(xa - lo, hi - xb):.1f} mm: {label!r}")
    every = list(fig.texts)
    for ax in fig.axes:
        every += list(ax.texts)
        if ax.get_legend() is not None:
            every += ax.get_legend().get_texts()
        if not ax.axison:
            continue
        every += [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    off = 0
    for t in every:
        if not t.get_text() or not t.get_visible():
            continue
        bb = t.get_window_extent(renderer=r)
        lo, hi = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        if lo < 0.2 or hi > FIG_W - 0.2:
            off += 1
            say(f"  WARNING text runs off the page ({lo:.1f} .. {hi:.1f} mm): {t.get_text()[:60]!r}")
    say(f"text-width check: {len(items)} registered texts, {bad} overflow; {len(every)} texts on the page, "
        f"{off} off it")
    return bad + off


# ------------------------------------------------------------------------------------------------ cache
def save_cache(N, A):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE + ".npz", **A)
    json.dump(N, io.open(CACHE + ".json", "w", encoding="utf-8"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_cache():
    z = np.load(CACHE + ".npz")
    return json.load(io.open(CACHE + ".json", encoding="utf-8")), {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recompute", action="store_true", help="ignore the cached slices and recompute them")
    args = ap.parse_args()

    recs, skipped = load_records()
    rows = scan_rows(recs)
    say(f"{len(rows)} scans counted, {len(skipped)} skipped {skipped if skipped else ''}")
    counts = {c: sum(1 for s in rows if s["cohort"] == c) for c in COHORTS}
    site_counts = {c: sum(1 for s in rows if s["site"] == c) for c in SITES}
    say(f"  cohorts: {counts}; sites: {site_counts}")
    assert len(rows) == EXPECT_N, f"{len(rows)} records, expected {EXPECT_N}"
    assert site_counts == EXPECT_SITES, (site_counts, EXPECT_SITES)
    assert len({s["id"] for s in rows}) == len(rows), "a scan in two result directories"

    stats = {cfg: block(rows, cfg) for cfg in ("A", "B")}
    for cfg in ("A", "B"):
        b = stats[cfg]
        say(f"  configuration {cfg}: identical {b['identical']}/{b['n']}, {fmt(b['mismatch_total'])} differing "
            f"voxels ({fmt(b['ours_only'])} ours-only, {fmt(b['ipl_only'])} IPL-only), worst "
            f"{fmt(b['max_mismatch'])}, min Dice {b['dice_min']:.6f}; Ct.Po slope {b['slope']:.6f} "
            f"R2 {b['r2']:.7f} ICC {b['icc']:.7f} bias {b['bias']:+.3e} identical {b['identical_ctpo']}/{b['n']}")
    check = cross_check(stats)
    prop = propagation(rows)
    say(f"  {prop['sentence']}")

    scan_b = max(rows, key=lambda s: (s["B"]["mismatch"], s["id"]))["id"]
    say(f"  panel A scan {SCAN_A}; panel B scan {scan_b} ({fmt(stats['B']['max_mismatch'])} differing voxels)")
    recs_by_id = {r["id"]: r for r in recs}
    if (not args.recompute and os.path.exists(CACHE + ".npz") and os.path.exists(CACHE + ".json")
            and json.load(io.open(CACHE + ".json", encoding="utf-8"))["B"]["id"] == scan_b):
        say(f"loading the cached slices {CACHE}")
        N, A = load_cache()
    else:
        N, A = compute_display(rows, recs_by_id, SCAN_A, scan_b)
        save_cache(N, A)
        say(f"cache written {CACHE}")

    sheets = sheet_table(rows)
    say(f"  printed sheets: {sheets}")
    fig, P, bad = draw(rows, stats, prop, N, A, sheets, check)
    fig.savefig(OUT + ".png", dpi=300, facecolor="white")
    fig.savefig(OUT + ".svg", facecolor="white")
    plt.close(fig)
    P["figure"] = dict(width_mm=FIG_W, height_mm=round(FIG_H, 1), dpi=300, n_scans=len(rows),
                       cohorts=counts, sites=site_counts, text_overflows=bad, skipped=skipped,
                       sources=[os.path.relpath(d, REPO).replace("\\", "/") for d in REC_DIRS],
                       generated=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(P, io.open(OUT + "_numbers.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    say(f"wrote {OUT}.png, {OUT}.svg, {OUT}_numbers.json")
    if bad:
        say(f"  {bad} text problems remain -- fix the layout before using the figure")


if __name__ == "__main__":
    main()
