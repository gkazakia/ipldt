"""fig3_lh.py -- Figure 3 of the ipldt / ORMIR-BQRL manuscript: the Laplace-Hamming segmentation (STEP 2).

Panels
  (a) the transfer function H(|k|) of IPL's /fft_laplace_hamming as ipldt implements it: the Laplacian factor
      (2 pi)^2 [(1 - eps) + eps |k|^2], eps = 0.45, multiplied by the radial Hann window (amp 1) cut at 0.3 / el_z.
  (b) one slice of a real scan (patella, 60.7 um): native greyscale, the normalised filter response (int16), the
      SEG ipldt assembles from it (cortical 127 / trabecular 126), IPL's SEG of the same slice with the voxel that
      differs marked, and a voxel-level zoom of the neighbourhood.  Everything is computed here with the shipped
      engine (ipldt.ormir.lh_filter_core -> threshold -> ipl_seg_assembly); the differing voxels found are asserted
      to be exactly the ones the validation record lists.
  (c) per-scan agreement of the SEG over all 137 scans (configuration B: IPL's periosteal contour in): differing
      voxels per scan (symlog) and Dice (log 1 - Dice), one colour per cohort (the assignment of Figure 5), sites
      bracketed.  Every number is read from the validation records and cross-checked against manuscript/facts/facts.json.
Panel letters are printed as capitals (A, B, C) in the upper-left corner, journal style.

Run from the repository root in the `ormir` environment:

    set PYTHONUTF8=1
    python manuscript/figures/fig3_lh.py            # the figure
    python manuscript/figures/fig3_lh.py --verify oslh     # legend check
    python manuscript/figures/fig3_lh.py --verify patella  # legend check

Writes manuscript/figures/fig3_lh.png (300 dpi, 180 mm wide), fig3_lh.svg and fig3_lh_numbers.json (every number
drawn or quoted).  `--verify` recomputes the normalised filter response at every differing SEG voxel whose global
position the records store (radius/tibia scans with 1-200 differing voxels, 19 patellae) and writes
fig3_lh_verify_<cohort>.json: the evidence behind the legend sentences on where the differing voxels lie (its
"census" block: at the threshold or not, levels above it, 26-connected clusters, the scans without a voxel list).
Validation set (n = 137): 21 patellae (REC_PATELLA) and 116 radius/tibia scans -- the 62 of oslh_auto_vN and the 54
diaphyseal measurements of oslh_noedit_vN (REC_OSLH_DIRS; Diaphyseal_CKD_991161 is not part of it).
Reads only the greyscale AIMs, IPL's compartment rasters / contour renderings / SEG / exported STEP-2 intermediates of the displayed scan,
and the validation records.  Runtime: about 1.5 min for the figure (one 1024 x 512 x 256 FFT), ~10 min for
--verify oslh (116 scans), ~25 min for --verify patella.  Deterministic.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
from ipldt.io import read_aim  # noqa: E402
from ipldt import ormir  # noqa: E402
from ipldt import ipl_ops as ops  # noqa: E402

# ------------------------------------------------------------------------------------------------ inputs
# the displayed scan (a patella of the validation set) and IPL's files of it
DATA = lab_path("PFJOA/XCT_masks_full_grab/PFJ-0be66a_R")
BASE = "X2420448"
IPL_EXPORTS = lab_path("Python/scripts/IPL/probes/p15_gobj_render/aims_and_logs")   # IPL's own STEP-2 intermediates
PATELLA_DATA_ROOT = lab_path("PFJOA/XCT_masks_full_grab")
REC_PATELLA = os.path.join(REPO, "validation", "results", "from_ipl_contour_ceil", "records")
# radius / tibia: the 62 counted measurements of oslh_auto_vN and the 54 diaphyseal measurements of
# oslh_noedit_vN -- 116 scans; the same list as manuscript/facts/facts.json
REC_OSLH_DIRS = [os.path.join(REPO, "validation", "results", OSLH_AUTO, "records"),
                 os.path.join(REPO, "validation", "results", OSLH_NOEDIT, "records")]
REC_OSLH = REC_OSLH_DIRS[0]
OSLH_PATTERN = re.compile(r"^[A-Za-z]+_[A-Za-z]+_\d+\.json$")
OSLH_NOT_USED = {"Diaphyseal_CKD_991161.json"}          # a record that exists but is not part of the validation set
N_TOTAL, N_PATELLA, N_OSLH = 137, 21, 116
SEG_LIST_MAX = 200                                     # the records store voxel coordinates for 1..200 differing voxels


def scan_folder(r):
    """The measurement folder of a radius / tibia record under the non-public data roots (validation/datapaths.py):
    the published records carry no local paths, so the folder is rebuilt from the record id."""
    for root in (lab_path("Cross_validation_IPL/OS_LH_AUTO"), lab_path("Cross_validation_IPL/OS_LH_NOEDIT")):
        p = os.path.join(root, *r["id"].split("/")).replace("\\", "/")
        if os.path.isdir(p):
            return p
    return os.path.join(lab_path("Cross_validation_IPL/OS_LH_AUTO"), *r["id"].split("/")).replace("\\", "/")


def oslh_files():
    """The 116 radius/tibia records, in name order; a tag must not occur in both directories."""
    files = [f for d in REC_OSLH_DIRS for f in glob.glob(os.path.join(d, "*.json"))
             if OSLH_PATTERN.match(os.path.basename(f)) and os.path.basename(f) not in OSLH_NOT_USED]
    names = [os.path.basename(f) for f in files]
    assert len(names) == len(set(names)), "a radius/tibia record occurs in both record directories"
    return sorted(files, key=os.path.basename)


FACTS_JSON = os.path.join(REPO, "manuscript", "facts", "facts.json")
OUT_PNG = os.path.join(HERE, "fig3_lh.png")
OUT_SVG = os.path.join(HERE, "fig3_lh.svg")
OUT_NUM = os.path.join(HERE, "fig3_lh_numbers.json")

# IPL parameters as ipldt carries them
EPS = ormir.LAPLACE_EPS               # 0.45
LP = ormir.LP_CUT_OFF_FREQ            # 0.3
AMP = ormir.HAMMING_AMP               # 1.0 -> Hann
NMAX = ormir.NORM_MAX_VALUE           # 200000
THR = ormir.LH_THRESHOLD              # 15564 = 475 permille of 32767
I16 = ormir.INT16_MAX                 # 32767
SEG_CORT, SEG_TRAB = ormir.SEG_VALUE_CORT, ormir.SEG_VALUE_TRAB

T0 = time.time()
NUM = {}


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def fmt_int(n):
    """Thousands separated by commas, as in the legends and the text (American style)."""
    return f"{int(n):,d}"


# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito, colour-blind safe
C_BLUE, C_ORANGE, C_VERM, C_GREEN, C_SKY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9"
C_CORT, C_TRAB = "#3A3A3A", "#A6A6A6"          # SEG labels 127 / 126 (two greys: one hue, dark -> light)
C_BOTH = "#8C8C8C"
INK, INK2, GRID = "#1A1A1A", "#4D4D4D", "#D9D9D9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "savefig.dpi": 300, "figure.dpi": 100, "svg.fonttype": "none", "pdf.fonttype": 42,
})
FIG_W_MM, FIG_H_MM = 180.0, 156.0


def mm_axes(fig, x, y, w, h, **kw):
    """An axes placed in millimetres from the top-left corner of the figure."""
    return fig.add_axes([x / FIG_W_MM, 1.0 - (y + h) / FIG_H_MM, w / FIG_W_MM, h / FIG_H_MM], **kw)


def panel_letter(fig, x, y, s):
    fig.text(x / FIG_W_MM, 1.0 - y / FIG_H_MM, s, fontsize=9, fontweight="bold", ha="left", va="top", color=INK)


# ================================================================================================= (a) transfer function
def transfer_function(k, el_z):
    """H(|k|) of /fft_laplace_hamming in float64 for plotting (ipldt.ormir._lh_transfer is the float32 volume form)."""
    k_lp = LP / el_z
    win = np.where(k < k_lp, (1.0 - 0.5 * AMP) + 0.5 * AMP * np.cos(np.pi * k / k_lp), 0.0)
    lap = (2.0 * np.pi) ** 2 * ((1.0 - EPS) + EPS * k * k)
    return lap * win, lap, win, k_lp


# ================================================================================================= (b) the slice
def load_slice_data():
    """The shipped engine on the displayed scan: filter response, threshold, SEG assembly, comparison with IPL."""
    say("reading the greyscale and IPL's files of the displayed scan")
    grey = read_aim(os.path.join(DATA, f"{BASE}.AIM"))
    native = grey["data"]
    el = tuple(float(e) for e in grey["el_size_mm"])
    gdim, gpos = tuple(int(v) for v in grey["dim"]), tuple(int(v) for v in grey["pos"])
    NUM["scan"] = dict(dim=gdim, pos=gpos, el_size_mm=el, voxels=int(np.prod(gdim)))

    say("Laplace-Hamming filter (ipldt.ormir.lh_filter_core, the shipped defaults) with Script 32's duplicated border")
    extended = np.pad(native, ((1, 1), (1, 1), (1, 1)), mode="edge").astype(np.float32)
    lh_ext, sh_ext = ormir.lh_filter_core(extended, el)          # pad offset / dtype: the engine's defaults
    del extended, lh_ext
    short = np.ascontiguousarray(sh_ext[1:-1, 1:-1, 1:-1])       # IPL's short on the greyscale grid
    del sh_ext
    bm = (short >= THR) & (short <= I16)                          # /threshold 475..1000 permille
    NUM["lh"] = dict(pad_offset=ormir.LH_PAD_OFFSET, dtype=ormir.LH_DTYPE, threshold=THR, threshold_voxels=int(bm.sum()),
                     pad_dims=[int(x) for x in (np.array([1 if n + 2 <= 1 else 2 ** int(np.ceil(np.log2(n + 2))) for n in gdim]))])

    say("SEG assembly (periosteal raster, component filters 35 / 70, compartment contours)")
    cort_raw = read_aim(os.path.join(DATA, f"{BASE}_CORT_MASK.AIM"))
    trab_raw = read_aim(os.path.join(DATA, f"{BASE}_TRAB_MASK.AIM"))
    prx = (ops.on_grid(cort_raw, gdim, gpos) != 0) | (ops.on_grid(trab_raw, gdim, gpos) != 0)
    # the rendered compartment contours: IPL's renderings, which equal ipldt's own on this scan (0 voxels differ; records)
    g_cort = ops.on_grid(read_aim(os.path.join(IPL_EXPORTS, f"{BASE}_P15_CORT_G2A.AIM")), gdim, gpos) != 0
    g_trab = ops.on_grid(read_aim(os.path.join(IPL_EXPORTS, f"{BASE}_P15_TRAB_G2A.AIM")), gdim, gpos) != 0
    cort_seg, trab_seg = ormir.ipl_seg_assembly(bm, prx, g_cort, g_trab)
    seg = np.zeros(bm.shape, np.uint8)
    seg[trab_seg] = SEG_TRAB
    seg[cort_seg] = SEG_CORT
    del cort_seg, trab_seg, g_cort, g_trab, prx

    seg_ipl = ops.on_grid(read_aim(os.path.join(DATA, f"{BASE}_SEG.AIM")), gdim, gpos)
    mism = (seg != 0) != (seg_ipl != 0)
    zz, yy, xx = np.nonzero(mism)
    found = sorted((int(x + gpos[0]), int(y + gpos[1]), int(z + gpos[2]), 1 if seg[z, y, x] else -1) for z, y, x in zip(zz, yy, xx))
    labels_differ = int(((seg != 0) & (seg_ipl != 0) & (seg != seg_ipl)).sum())

    rec = json.load(open(os.path.join(REC_PATELLA, "PFJ-0be66a_R.json")))["seg"]["SEG"]
    recorded = sorted(tuple(int(v) for v in w) for w in rec["where"])
    assert rec["mismatches"] == len(found) == len(recorded) and found == recorded, \
        f"the engine's SEG differs from IPL's on {found}, the validation record says {recorded}"
    NUM["seg"] = dict(voxels_ours=int((seg != 0).sum()), voxels_ipl=int((seg_ipl != 0).sum()), mismatches=len(found),
                      ours_only=sum(1 for w in found if w[3] > 0), ipl_only=sum(1 for w in found if w[3] < 0),
                      label_mismatches=labels_differ, where_global_xyz_sign=found, record_dice=rec["dice"],
                      record_agrees=True)
    say(f"  SEG {NUM['seg']['voxels_ours']:,d} (IPL {NUM['seg']['voxels_ipl']:,d}); {len(found)} voxels differ "
        f"(+{NUM['seg']['ours_only']} / -{NUM['seg']['ipl_only']}), labels differ {labels_differ}; = the validation record")

    # IPL's own exported intermediates of this scan: the normalised short and the thresholded volume
    say("IPL's exported normalised response and thresholded volume at the differing voxels")
    short_ipl = ops.on_grid(read_aim(os.path.join(IPL_EXPORTS, f"{BASE}_P15_LHNORM.AIM")), gdim, gpos)
    thr_ipl = ops.on_grid(read_aim(os.path.join(IPL_EXPORTS, f"{BASE}_P15_LHSEG.AIM")), gdim, gpos) != 0
    raw_thr_mism = bm != thr_ipl
    vox = []
    for (x, y, z, sgn) in found:
        i = (z - gpos[2], y - gpos[1], x - gpos[0])
        vox.append(dict(global_xyz=[x, y, z], sign=sgn, short_ipldt=int(short[i]), short_ipl=int(short_ipl[i]),
                        threshold_ipldt=bool(bm[i]), threshold_ipl=bool(thr_ipl[i]),
                        raw_threshold_decision_differs=bool(raw_thr_mism[i])))
    NUM["threshold_level"] = dict(raw_threshold_decisions_differ=int(raw_thr_mism.sum()), voxels_compared=int(np.prod(gdim)),
                                  short_differs=int((short != short_ipl).sum()),
                                  short_abs_diff_max=int(np.abs(short.astype(np.int32) - short_ipl.astype(np.int32)).max()),
                                  at_seg_mismatches=vox)
    say(f"  raw threshold decisions differ on {int(raw_thr_mism.sum())} of {np.prod(gdim):,d} voxels; the short differs on "
        f"{NUM['threshold_level']['short_differs']:,d} voxels by at most {NUM['threshold_level']['short_abs_diff_max']} level(s)")
    for v in vox:
        say(f"  differing SEG voxel {v['global_xyz']} ({'ipldt only' if v['sign'] > 0 else 'IPL only'}): short ipldt {v['short_ipldt']}, "
            f"IPL {v['short_ipl']}, threshold {THR}; raw threshold decision differs: {v['raw_threshold_decision_differs']}")
    del short_ipl, thr_ipl, raw_thr_mism

    # the slice: the differing voxel nearest the middle of the stack
    zc = gdim[2] // 2
    pick = min(found, key=lambda w: abs((w[2] - gpos[2]) - zc))
    x_l, y_l, z_l = pick[0] - gpos[0], pick[1] - gpos[1], pick[2] - gpos[2]
    return dict(native=native, short=short, seg=seg, seg_ipl=seg_ipl, mism=mism, el=el, gdim=gdim, gpos=gpos,
                pick=pick, pick_local=(x_l, y_l, z_l), pick_values=next(v for v in vox if v["global_xyz"] == list(pick[:3])))


# ================================================================================================= (c) the cohort
SITE_ORDER = [("patella", "patella"), ("UD Radius", "UD radius"), ("UD Tibia", "UD tibia"), ("D Radius", "D radius"), ("D Tibia", "D tibia")]
# one colour per cohort, the same assignment as Figure 5 (the two diaphyseal sites share the diaphyseal colour)
SITE_COLOUR = {"patella": C_BLUE, "UD Radius": C_GREEN, "UD Tibia": C_ORANGE, "D Radius": "#CC79A7", "D Tibia": "#CC79A7"}


def load_cohort():
    rows = []
    for f in sorted(glob.glob(os.path.join(REC_PATELLA, "*.json"))):
        r = json.load(open(f))
        s = r["seg"]["SEG"]
        rows.append(dict(scan=r["subject"], cohort="patella", site="patella", mism=int(s["mismatches"]), plus=int(s["ours_only"]),
                         minus=int(s["ipl_only"]), dice=float(s["dice"]), union=int(s["union_voxels"]), ipl=int(s["voxels_ipl"]),
                         labels=int(s["label_mismatches"])))
    for f in oslh_files():
        r = json.load(open(f))
        s = r["B"]["seg"]["SEG"]
        rows.append(dict(scan=r["tag"], cohort="radius/tibia", site=r["meta"]["bone_key"], mism=int(s["mismatches"]),
                         plus=int(s["ours_only"]), minus=int(s["ipl_only"]), dice=float(s["dice"]), union=int(s["union_voxels"]),
                         ipl=int(s["voxels_ipl"]), labels=int(s["label_mismatches"])))
    assert len(rows) == N_TOTAL, len(rows)
    assert sum(r["cohort"] == "patella" for r in rows) == N_PATELLA and sum(r["cohort"] != "patella" for r in rows) == N_OSLH

    facts = json.load(open(FACTS_JSON))["facts"]

    def fact(key):
        return facts[key]["value"]

    tot = sum(r["mism"] for r in rows)
    summary = dict(n=len(rows), mismatches=tot, ours_only=sum(r["plus"] for r in rows), ipl_only=sum(r["minus"] for r in rows),
                   voxels_compared=sum(r["union"] for r in rows), voxels_ipl=sum(r["ipl"] for r in rows),
                   exact=sum(r["mism"] == 0 for r in rows), median_per_scan=float(np.median([r["mism"] for r in rows])),
                   max_per_scan=max(r["mism"] for r in rows), dice_min=min(r["dice"] for r in rows),
                   dice_median=float(np.median([r["dice"] for r in rows])), dice_ge_0_9999=sum(r["dice"] >= 0.9999 for r in rows),
                   label_mismatches=sum(r["labels"] for r in rows),
                   per_million=1e6 * tot / sum(r["union"] for r in rows),
                   le_10=sum(r["mism"] <= 10 for r in rows), le_100=sum(r["mism"] <= 100 for r in rows), le_1000=sum(r["mism"] <= 1000 for r in rows))
    # the records must reproduce the facts sheet (the paper's numbers) exactly
    checks = {"B.pooled.SEG.mismatches": summary["mismatches"], "B.pooled.SEG.ours_only": summary["ours_only"],
              "B.pooled.SEG.ipl_only": summary["ipl_only"], "B.pooled.SEG.voxels_compared": summary["voxels_compared"],
              "B.pooled.SEG.exact_scans": summary["exact"], "B.pooled.SEG.n_scans": summary["n"],
              "B.patella.SEG.mismatches": sum(r["mism"] for r in rows if r["cohort"] == "patella"),
              "B.oslh.SEG.mismatches": sum(r["mism"] for r in rows if r["cohort"] != "patella"),
              "B.oslh.SEG.top3_mismatches": sum(sorted((r["mism"] for r in rows if r["cohort"] != "patella"), reverse=True)[:3]),
              "B.pooled.SEG.voxels_ipl": summary["voxels_ipl"], "B.pooled.SEG.dice.ge_0_9999": summary["dice_ge_0_9999"],
              "B.pooled.SEG.per_scan.median": summary["median_per_scan"], "B.pooled.SEG.per_scan.le_10": summary["le_10"],
              "B.pooled.SEG.per_scan.le_100": summary["le_100"], "B.pooled.SEG.per_scan.le_1000": summary["le_1000"],
              "B.oslh.SEG.label_mismatches": sum(r["labels"] for r in rows if r["cohort"] != "patella"),
              "B.patella.SEG.label_mismatches": sum(r["labels"] for r in rows if r["cohort"] == "patella")}
    fsite = {"UD Radius": "UD_radius", "UD Tibia": "UD_tibia", "D Radius": "D_radius", "D Tibia": "D_tibia"}
    for key, fk in fsite.items():
        rs = [r for r in rows if r["site"] == key]
        checks.update({f"B.site.{fk}.SEG.n_scans": len(rs), f"B.site.{fk}.SEG.mismatches": sum(r["mism"] for r in rs),
                       f"B.site.{fk}.SEG.exact_scans": sum(r["mism"] == 0 for r in rs),
                       f"B.site.{fk}.SEG.per_scan.max": max(r["mism"] for r in rs)})
    for k, v in checks.items():
        assert fact(k) == v, (k, fact(k), v)
    for k, v in (("B.pooled.SEG.dice.min", summary["dice_min"]), ("B.pooled.SEG.dice.median", summary["dice_median"]),
                 ("B.pooled.SEG.mismatch_ppm", summary["per_million"])):
        assert abs(fact(k) - v) < 1e-9, (k, fact(k), v)
    summary["facts_checked"] = sorted(checks) + ["B.pooled.SEG.dice.min", "B.pooled.SEG.dice.median", "B.pooled.SEG.mismatch_ppm"]
    per_site = {}
    for key, label in SITE_ORDER:
        rs = [r for r in rows if r["site"] == key]
        per_site[label] = dict(n=len(rs), mismatches=sum(r["mism"] for r in rs), exact=sum(r["mism"] == 0 for r in rs),
                               dice_min=min(r["dice"] for r in rs), median=float(np.median([r["mism"] for r in rs])),
                               max=max(r["mism"] for r in rs))
    top3 = sorted(rows, key=lambda r: -r["mism"])[:3]
    summary["top3"] = [dict(site=r["site"], mismatches=r["mism"], ours_only=r["plus"], ipl_only=r["minus"], dice=r["dice"]) for r in top3]
    summary["top3_mismatches"] = sum(r["mism"] for r in top3)
    summary["without_top3_max"] = max(r["mism"] for r in rows if r not in top3)
    oslh = [r for r in rows if r["cohort"] != "patella"]
    top3o = sorted(oslh, key=lambda r: -r["mism"])[:3]
    rest = [r for r in oslh if r not in top3o]
    summary["oslh"] = dict(n=len(oslh), mismatches=sum(r["mism"] for r in oslh),
                           top3_mismatches=sum(r["mism"] for r in top3o),
                           top3_share_pct=100.0 * sum(r["mism"] for r in top3o) / sum(r["mism"] for r in oslh),
                           top3_sites=[r["site"] for r in top3o], without_top3_n=len(rest),
                           without_top3_mismatches=sum(r["mism"] for r in rest),
                           without_top3_median=float(np.median([r["mism"] for r in rest])),
                           without_top3_max=max(r["mism"] for r in rest))
    for k, v in (("B.oslh.SEG.without_top3.mismatches", summary["oslh"]["without_top3_mismatches"]),
                 ("B.oslh.SEG.without_top3.n_scans", summary["oslh"]["without_top3_n"]),
                 ("B.oslh.SEG.without_top3.per_scan_max", summary["oslh"]["without_top3_max"]),
                 ("B.oslh.SEG.without_top3.per_scan_median", summary["oslh"]["without_top3_median"])):
        assert fact(k) == v, (k, fact(k), v)
    assert abs(fact("B.oslh.SEG.top3_share_pct") - summary["oslh"]["top3_share_pct"]) < 1e-9
    NUM["cohort"] = dict(summary=summary, per_site=per_site, rows=rows)
    say(f"  cohort: {summary['n']} scans, {summary['mismatches']:,d} differing voxels of {summary['voxels_compared']:,d} "
        f"({summary['per_million']:.2f} per million), identical on {summary['exact']}, median {summary['median_per_scan']:.0f} per scan, "
        f"min Dice {summary['dice_min']:.8f}, median Dice {summary['dice_median']:.8f}, Dice >= 0.9999 on {summary['dice_ge_0_9999']}")
    return rows, summary, per_site


# ================================================================================================= the figure
def draw(sl, rows, summary, per_site):
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))

    # ---------------------------------------------------------------------------------------- (a)
    el_z = sl["el"][2]
    nyq = 0.5 / el_z
    k = np.linspace(0.0, nyq, 2000)
    H, lap, win, k_lp = transfer_function(k, el_z)
    NUM["transfer"] = dict(cut_off_lp_mm=k_lp, nyquist_lp_mm=nyq, H_max=float(H.max()), k_at_H_max=float(k[np.argmax(H)]),
                           H_at_zero=float(H[0]), eps=EPS, lp_cut_off=LP, amp=AMP)
    axw = mm_axes(fig, 11, 8, 50, 11)
    axh = mm_axes(fig, 11, 21.5, 50, 40, sharex=axw)
    for ax in (axw, axh):
        ax.set_facecolor("white")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    axw.plot(k, win, color=INK2, lw=1.0)
    axw.fill_between(k, 0, win, color=C_SKY, alpha=0.35, lw=0)
    axw.set_ylim(0, 1.05)
    axw.set_yticks([0, 0.5, 1])
    axw.set_ylabel("W(|k|)", labelpad=2)
    axw.tick_params(labelbottom=False)
    # between the cut-off and the Nyquist line, clear of both
    axw.text(k_lp + 0.35, 0.55, "Hann window,\namplitude 1", fontsize=7, color=INK2, va="center", ha="left", linespacing=1.1)
    axh.plot(k, lap, color=C_ORANGE, lw=1.0, ls="--", label="(2\u03c0)\u00b2[(1 \u2212 \u03b5) + \u03b5|k|\u00b2],  \u03b5 = 0.45")
    axh.plot(k, H, color=C_BLUE, lw=1.6, label="H(|k|) = Laplacian \u00d7 window")
    axh.set_ylim(0, 100)
    axh.set_xlim(0, nyq * 1.02)
    axh.set_yticks([0, 20, 40, 60, 80, 100])
    axh.set_xlabel("spatial frequency |k| (line pairs / mm)", labelpad=2)
    axh.set_ylabel("gain H(|k|)", labelpad=2)
    axh.legend(handles=[Line2D([], [], color=C_BLUE, lw=1.6, label="H(|k|) = Laplacian \u00d7 window"),
                        Line2D([], [], color=C_ORANGE, lw=1.0, ls="--",
                               label="Laplacian factor\n(2\u03c0)\u00b2[(1 \u2212 \u03b5) + \u03b5|k|\u00b2],  \u03b5 = 0.45")],
               loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, borderpad=0.2,
               handlelength=1.8, borderaxespad=0.2, bbox_to_anchor=(1.0, 1.0)).set_zorder(6)
    # the legend (with the formula of the Laplacian factor) sits on a white ground above the cutoff / Nyquist lines
    for ax in (axw, axh):
        ax.axvline(k_lp, color=C_VERM, lw=0.8, ls=(0, (3, 2)))
        ax.axvline(nyq, color=INK2, lw=0.6, ls=":")
    axh.text(k_lp + 0.15, 40, f"cutoff 0.3 / \u0394z\n= {k_lp:.2f} lp/mm", color=C_VERM, fontsize=7, ha="left", va="center")
    axh.text(nyq - 0.12, 14, f"Nyquist\n{nyq:.2f} lp/mm", color=INK2, fontsize=7, ha="right", va="center")
    fig.text(11 / FIG_W_MM, 1.0 - 70.0 / FIG_H_MM,
             "then: scale to \u00b1200,000 \u2192 int16 (\u00b132,767);\n"
             "threshold 475\u20131,000\u2030 (15,564 \u2264 s \u2264 32,767);\n"
             "periosteal mask; components \u2265 35 (cortical)\n"
             "or \u2265 70 (trabecular) voxels, each kept within\n"
             "its compartment contour \u2192 SEG 127 / 126",
             fontsize=7, color=INK2, va="top", ha="left", linespacing=1.25)
    panel_letter(fig, 2, 4, "A")

    # ---------------------------------------------------------------------------------------- (b)
    x_l, y_l, z_l = sl["pick_local"]
    gdim = sl["gdim"]
    CROP = 160
    x0 = int(np.clip(x_l - CROP // 2, 0, gdim[0] - CROP))
    y0 = int(np.clip(y_l - CROP // 2, 0, gdim[1] - CROP))
    x1, y1 = x0 + CROP, y0 + CROP
    ext = [x0, x1, y1, y0]
    nat = sl["native"][z_l, y0:y1, x0:x1].astype(np.float32)
    sho = sl["short"][z_l, y0:y1, x0:x1]
    seg = sl["seg"][z_l, y0:y1, x0:x1]
    segi = sl["seg_ipl"][z_l, y0:y1, x0:x1]
    mis = sl["mism"][z_l, y0:y1, x0:x1]
    n_mis_slice = int(sl["mism"][z_l].sum())
    NUM["slice"] = dict(z_local=z_l, z_global=z_l + sl["gpos"][2], crop_xyxy_local=[x0, y0, x1, y1], crop_mm=CROP * sl["el"][0],
                        seg_mismatches_in_slice=n_mis_slice, marked_voxel=sl["pick_values"])

    def seg_rgb(lab):
        rgb = np.ones(lab.shape + (3,), np.float32)
        rgb[lab == SEG_CORT] = mcolors.to_rgb(C_CORT)
        rgb[lab == SEG_TRAB] = mcolors.to_rgb(C_TRAB)
        return rgb

    S = 29.0                      # image size (mm)
    bx, by, gap = 70.0, 8.0, 2.0
    pos = {"nat": (bx, by), "sho": (bx + S + gap, by), "seg": (bx, by + S + gap + 1.5), "ipl": (bx + S + gap, by + S + gap + 1.5)}
    axes_b = {}
    for key, (px, py) in pos.items():
        ax = mm_axes(fig, px, py, S, S)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color(INK2)
        axes_b[key] = ax
    axes_b["nat"].imshow(nat, cmap="gray", vmin=-1500, vmax=9000, interpolation="nearest", extent=ext)
    im = axes_b["sho"].imshow(sho, cmap="gray", vmin=-20000, vmax=I16, interpolation="nearest", extent=ext)
    axes_b["seg"].imshow(seg_rgb(seg), interpolation="nearest", extent=ext)
    rgb_ipl = seg_rgb(segi)
    ours_only = mis & (seg != 0)
    ipl_only = mis & (segi != 0)
    rgb_ipl[ours_only] = mcolors.to_rgb(C_VERM)
    rgb_ipl[ipl_only] = mcolors.to_rgb(C_BLUE)
    axes_b["ipl"].imshow(rgb_ipl, interpolation="nearest", extent=ext)
    mark_col = C_VERM if sl["pick"][3] > 0 else C_BLUE
    for key in ("seg", "ipl"):
        axes_b[key].add_patch(Circle((x_l + 0.5, y_l + 0.5), 9, fill=False, ec=mark_col, lw=0.9))
    # 1 mm scale bar
    bar = 1.0 / sl["el"][0]
    # inset far enough that the centred '1 mm' label (wider than the bar) stays inside the image
    axes_b["nat"].add_patch(Rectangle((x0 + 14, y1 - 10), bar, 2.2, fc="white", ec="none"))
    axes_b["nat"].text(x0 + 14 + bar / 2, y1 - 12, "1 mm", color="white", fontsize=7, ha="center", va="bottom")
    titles = {"nat": "native grayscale (int16)", "sho": "filter response, normalized int16", "seg": "SEG, ipldt",
              "ipl": "SEG, IPL (differing voxel circled)"}
    for key, t in titles.items():
        px, py = pos[key]
        fig.text((px) / FIG_W_MM, 1.0 - (py - 0.8) / FIG_H_MM, t, fontsize=7.2, ha="left", va="bottom", color=INK)
    # colour bar of the response with the threshold marked
    cax = mm_axes(fig, bx + 2 * S + gap + 1.2, by, 1.6, S)
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0, THR, I16])
    cb.set_ticklabels(["0", f"{fmt_int(THR)}", f"{fmt_int(I16)}"])
    cb.ax.tick_params(labelsize=7, length=2, width=0.5, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.ax.axhline(THR, color=C_VERM, lw=1.2)
    cb.ax.text(1.35, THR - 2200, "threshold", transform=cb.ax.get_yaxis_transform(), fontsize=7, color=C_VERM, ha="left", va="top")
    cb.ax.text(1.35, -20000, "\u221220,000", transform=cb.ax.get_yaxis_transform(), fontsize=7, color=INK2, ha="left", va="bottom")
    # legend of the SEG colours (two rows, under the 2 x 2 block)
    ly = by + 2 * S + gap + 1.5 + 1.0
    handles = [Patch(fc=C_CORT, ec="none", label="cortical (127)"), Patch(fc=C_VERM, ec="none", label="ipldt only"),
               Patch(fc=C_TRAB, ec="none", label="trabecular (126)"), Patch(fc=C_BLUE, ec="none", label="IPL only")]
    lax = mm_axes(fig, bx, ly, 2 * S + gap, 7)
    lax.axis("off")
    lax.legend(handles=handles, loc="upper left", ncol=2, frameon=False, handlelength=1.0, handleheight=0.8, columnspacing=1.5,
               borderaxespad=0, fontsize=7, labelspacing=0.3)
    # voxel-level zoom of the marked voxel: SEG agreement, 15 x 15 voxels
    R = 7
    zx0, zy0 = x_l - R, y_l - R
    zx1, zy1 = x_l + R + 1, y_l + R + 1
    zs = sl["seg"][z_l, zy0:zy1, zx0:zx1] != 0
    zi = sl["seg_ipl"][z_l, zy0:zy1, zx0:zx1] != 0
    zrgb = np.ones(zs.shape + (3,), np.float32)
    zrgb[zs & zi] = mcolors.to_rgb(C_BOTH)
    zrgb[zs & ~zi] = mcolors.to_rgb(C_VERM)
    zrgb[~zs & zi] = mcolors.to_rgb(C_BLUE)
    ZX = 145.0                    # x of the right-hand column (zoom, legend, numbers)
    zax = mm_axes(fig, ZX, by, S, S)
    zax.imshow(zrgb, interpolation="nearest", extent=[zx0, zx1, zy1, zy0])
    zax.set_xticks(np.arange(zx0, zx1 + 1), minor=True)
    zax.set_yticks(np.arange(zy0, zy1 + 1), minor=True)
    zax.grid(True, which="minor", color="white", lw=0.4)
    zax.set_xticks([])
    zax.set_yticks([])
    zax.tick_params(which="minor", length=0)
    for sp in zax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color(INK2)
    zax.add_patch(Rectangle((x_l, y_l), 1, 1, fill=False, ec=INK, lw=0.9))
    fig.text(ZX / FIG_W_MM, 1.0 - (by - 0.8) / FIG_H_MM, f"zoom ({2 * R + 1} \u00d7 {2 * R + 1} voxels)", fontsize=7.2, ha="left", va="bottom")
    zh = [Patch(fc=C_BOTH, ec="none", label="set in both SEGs"), Patch(fc="white", ec=INK2, lw=0.5, label="set in neither"),
          Patch(fc=mark_col, ec="none", label="ipldt only" if sl["pick"][3] > 0 else "IPL only")]
    zlax = mm_axes(fig, ZX, by + S + 1.2, 33, 10)
    zlax.axis("off")
    zlax.legend(handles=zh, loc="upper left", ncol=1, frameon=False, handlelength=1.0, handleheight=0.8, borderaxespad=0, fontsize=7,
                labelspacing=0.25)
    pv = sl["pick_values"]
    side = "ipldt only" if pv["sign"] > 0 else "IPL only"
    mvox = NUM["scan"]["voxels"] / 1e6
    txt = (f"marked voxel ({side}):\n"
           f"ipldt response  {fmt_int(pv['short_ipldt'])}\n"
           f"IPL response    {fmt_int(pv['short_ipl'])}\n"
           f"threshold          {fmt_int(THR)}\n\n"
           f"this scan ({mvox:.1f} M voxels):\n"
           f"threshold decisions differ  {NUM['threshold_level']['raw_threshold_decisions_differ']}\n"
           f"SEG voxels differ  {NUM['seg']['mismatches']} "
           f"(+{NUM['seg']['ours_only']} / \u2212{NUM['seg']['ipl_only']})\n"
           f"Dice {NUM['seg']['record_dice']:.7f}")
    fig.text(ZX / FIG_W_MM, 1.0 - (by + S + 12.0) / FIG_H_MM, txt, fontsize=7, ha="left", va="top", color=INK, linespacing=1.25)
    panel_letter(fig, 65, 4, "B")

    # ---------------------------------------------------------------------------------------- (c)
    order = []
    xpos = []
    x = 0.0
    groups = []
    for key, label in SITE_ORDER:
        rs = sorted([r for r in rows if r["site"] == key], key=lambda r: (-r["mism"], r["scan"]))
        start = x
        for r in rs:
            order.append(r)
            xpos.append(x)
            x += 1.0
        groups.append((label, len(rs), start, x - 1.0))
        x += 2.5
    xpos = np.array(xpos)
    mism = np.array([r["mism"] for r in order], float)
    dice = np.array([r["dice"] for r in order], float)
    cols = [SITE_COLOUR[r["site"]] for r in order]
    cy0 = 90.0
    ax1 = mm_axes(fig, 21, cy0, 155, 30)
    ax2 = mm_axes(fig, 21, cy0 + 32.5, 155, 22, sharex=ax1)
    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, axis="y", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(-1.2, x - 2.0 + 0.2)
        ax.xaxis.set_major_locator(NullLocator())
    ax1.bar(xpos, mism, width=0.8, color=cols, lw=0)
    zero = mism == 0
    ax1.plot(xpos[zero], np.zeros(zero.sum()), "o", ms=2.5, mfc="white", mec=INK2, mew=0.6, zorder=3)
    ax1.set_yscale("symlog", linthresh=1.0, linscale=0.4)
    ax1.set_ylim(-0.45, 4e5)
    ax1.yaxis.set_major_locator(FixedLocator([0, 1, 10, 100, 1e3, 1e4, 1e5]))
    ax1.yaxis.set_major_formatter(FixedFormatter(["0", "1", "10", "100", "1,000", "10,000", "100,000"]))
    ax1.yaxis.set_minor_locator(NullLocator())
    ax1.set_ylabel("differing SEG\nvoxels per scan", labelpad=3, linespacing=1.1)
    ax1.yaxis.set_label_coords(-0.098, 0.5)
    ax1.tick_params(labelbottom=False)
    top3 = list(np.argsort(-mism)[:3])
    ax1.text(xpos[max(top3)] + 0.9, mism[top3[1]], "  ".join(fmt_int(mism[i]) for i in sorted(top3, key=lambda i: xpos[i])),
             fontsize=7, ha="left", va="center", color=INK2)
    s = summary
    # the pooled summary sits over the two diaphyseal groups (at most 4 differing voxels per scan), clear of every bar
    ax1.text(0.995, 0.99,
             f"n = {s['n']}: {fmt_int(s['mismatches'])} differing voxels\nof {fmt_int(s['voxels_compared'])} compared "
             f"({s['per_million']:.1f} per million);\nidentical on {s['exact']} scans; median {s['median_per_scan']:.0f} per scan;\n"
             f"three scans carry {fmt_int(s['top3_mismatches'])}; labels (127 / 126)\nnever differ on a voxel set in both",
             transform=ax1.transAxes, fontsize=7, va="top", ha="right", color=INK, linespacing=1.25)
    n_diaph = per_site["D radius"]["n"] + per_site["D tibia"]["n"]
    leg = [Patch(fc=SITE_COLOUR["patella"], ec="none", label=f"patella (n = {per_site['patella']['n']})"),
           Patch(fc=SITE_COLOUR["UD Radius"], ec="none", label=f"ultradistal radius (n = {per_site['UD radius']['n']})"),
           Patch(fc=SITE_COLOUR["UD Tibia"], ec="none", label=f"ultradistal tibia (n = {per_site['UD tibia']['n']})"),
           Patch(fc=SITE_COLOUR["D Radius"], ec="none", label=f"diaphyseal radius / tibia (n = {n_diaph})"),
           Line2D([], [], marker="o", ms=2.5, mfc="white", mec=INK2, mew=0.6, ls="none", label="0 (identical)")]
    ax1.legend(handles=leg, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=5, handlelength=1.0, handleheight=0.8,
               columnspacing=1.2, borderaxespad=0.0, fontsize=7)
    # Dice on a log(1 - Dice) axis, 1 at the top
    one_minus = 1.0 - dice
    for i, r in enumerate(order):
        ax2.plot([xpos[i]], [one_minus[i]], "o", ms=2.3, mec="none", mfc=cols[i], zorder=3)
    ax2.set_yscale("symlog", linthresh=1e-8, linscale=0.35)
    ax2.set_ylim(3e-3, -3.5e-9)
    ax2.yaxis.set_major_locator(FixedLocator([0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]))
    ax2.yaxis.set_major_formatter(FixedFormatter(["1", "0.9999999", "0.999999", "0.99999", "0.9999", "0.999"]))
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.set_ylabel("Dice\n(log(1 \u2212 Dice) axis)", labelpad=3, linespacing=1.1)
    ax2.yaxis.set_label_coords(-0.098, 0.5)
    ax2.text(0.995, 0.05, f"min {s['dice_min']:.6f}, median {s['dice_median']:.8f}; Dice \u2265 0.9999 on {s['dice_ge_0_9999']} of {s['n']}",
             transform=ax2.transAxes, fontsize=7, va="bottom", ha="right", color=INK)
    # site brackets under the Dice strip
    for label, n, xa, xb in groups:
        ax2.annotate("", xy=(xa - 0.4, -0.06), xytext=(xb + 0.4, -0.06), xycoords=("data", "axes fraction"),
                     textcoords=("data", "axes fraction"), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=0, shrinkB=0),
                     annotation_clip=False)
        text = f"{label} ({n})" if n > 2 else f"{label}\n({n})"
        ax2.text((xa + xb) / 2, -0.09, text, transform=ax2.get_xaxis_transform(), fontsize=7, ha="center", va="top", color=INK,
                 linespacing=1.1)
    ax2.set_xlabel("scans, per site in descending order of differing voxels", labelpad=16)
    panel_letter(fig, 2, 84, "C")
    return fig


# ================================================================================================= --verify
def verify(cohort):
    """Recompute ipldt's normalised response at every differing SEG voxel the records locate, and test the legend's
    claim: an 'ipldt only' voxel has a response >= 15564 and an 'IPL only' voxel a response < 15564, and how far
    from the threshold those responses sit (in int16 levels of a 65 535-level range)."""
    out = dict(cohort=cohort, threshold=THR, engine=dict(pad_offset=ormir.LH_PAD_OFFSET, dtype=ormir.LH_DTYPE), scans=[])
    if cohort == "oslh":
        files = oslh_files()
        assert len(files) == N_OSLH, len(files)
    else:
        files = sorted(glob.glob(os.path.join(REC_PATELLA, "*.json")))
    for f in files:
        r = json.load(open(f))
        if cohort == "oslh":
            s = r["B"]["seg"]["SEG"]
            grey_path = scan_folder(r).rstrip("/") + "/" + r["base"] + ".AIM"      # non-public: IPLDT_LAB_ROOT
            tag = r["tag"]
            lp = LP
        else:
            s = r["seg"]["SEG"]
            grey_path = os.path.join(PATELLA_DATA_ROOT, r["subject"], f"{r['base']}.AIM")
            tag = r["subject"]
            lp = LP
        if "where" not in s:
            out["scans"].append(dict(scan=tag, mismatches=s["mismatches"], checked=0, note="no voxel list in the record"))
            continue
        if not os.path.exists(grey_path):
            out["scans"].append(dict(scan=tag, mismatches=s["mismatches"], checked=0, note="greyscale not on disk"))
            continue
        say(f"{tag}: {len(s['where'])} differing voxels")
        grey = read_aim(grey_path)
        el = tuple(float(e) for e in grey["el_size_mm"])
        gpos = tuple(int(v) for v in grey["pos"])
        extended = np.pad(grey["data"], ((1, 1), (1, 1), (1, 1)), mode="edge").astype(np.float32)
        lh, sh = ormir.lh_filter_core(extended, el, lp_cut_off_freq=lp)
        del extended, lh
        short = sh[1:-1, 1:-1, 1:-1]
        vox = []
        for (x, y, z, sgn) in s["where"]:
            i = (z - gpos[2], y - gpos[1], x - gpos[0])
            v = int(short[i])
            vox.append(dict(xyz=[x, y, z], sign=int(sgn), short=v, offset=v - THR if sgn > 0 else v - (THR - 1),
                            consistent=bool((v >= THR) if sgn > 0 else (v < THR))))
        del short, sh
        out["scans"].append(dict(scan=tag, mismatches=s["mismatches"], checked=len(vox),
                                 consistent=sum(v["consistent"] for v in vox), voxels=vox))
    checked = [v for sc in out["scans"] if sc.get("voxels") for v in sc["voxels"]]
    offs = np.array([v["offset"] for v in checked]) if checked else np.array([])
    hist = {int(o): int((offs == o).sum()) for o in np.unique(offs)} if checked else {}
    out["summary"] = dict(scans_with_voxel_list=sum(1 for sc in out["scans"] if sc.get("checked", 0) > 0),
                          scans_without=sum(1 for sc in out["scans"] if sc.get("checked", 0) == 0),
                          voxels_checked=len(checked), consistent=int(sum(v["consistent"] for v in checked)),
                          within_1_level=int((np.abs(offs) <= 1).sum()) if checked else 0,
                          within_2_levels=int((np.abs(offs) <= 2).sum()) if checked else 0,
                          within_5_levels=int((np.abs(offs) <= 5).sum()) if checked else 0,
                          max_abs_offset=int(np.abs(offs).max()) if checked else None, offset_histogram=hist,
                          note="offset = ipldt short - 15564 for 'ipldt only' voxels (>= 0 expected), ipldt short - 15563 for 'IPL only' voxels (<= 0 expected); one int16 level = 200000/32767 = 6.1 filter units")
    out["census"] = census(out["scans"])
    path = os.path.join(HERE, f"fig3_lh_verify_{cohort}.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    say(f"verify {cohort}: {out['summary']}")
    say(f"written {path}")


def clusters_26(xyz):
    """Sizes of the 26-connected clusters of a list of (x, y, z) voxels (union-find on Chebyshev distance <= 1)."""
    xyz = [tuple(p) for p in xyz]
    parent = list(range(len(xyz)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(xyz)):
        for j in range(i + 1, len(xyz)):
            if max(abs(a - b) for a, b in zip(xyz[i], xyz[j])) <= 1:
                parent[root(i)] = root(j)
    sizes = {}
    for i in range(len(xyz)):
        sizes[root(i)] = sizes.get(root(i), 0) + 1
    return sorted(sizes.values(), reverse=True)


def census(scans):
    """Where the differing SEG voxels lie (the numbers of the Figure 3 legend): 'at the threshold' = ipldt's response is
    exactly 15564 at an ipldt-only voxel or 15563 at an IPL-only voxel; every other checked voxel is described by its
    response in int16 levels above the threshold (short - 15564) and grouped into 26-connected clusters per scan."""
    listed = [sc for sc in scans if sc.get("voxels")]
    at = [v for sc in listed for v in sc["voxels"] if v["offset"] == 0]
    other = [v for sc in listed for v in sc["voxels"] if v["offset"] != 0]
    cl = [c for sc in listed for c in clusters_26([v["xyz"] for v in sc["voxels"] if v["offset"] != 0])
          if any(v["offset"] != 0 for v in sc["voxels"])]
    lv = np.array([v["short"] - THR for v in other]) if other else np.array([0])
    unlisted = [sc for sc in scans if not sc.get("voxels") and sc["mismatches"] > 0]
    return dict(scans=len(scans), scans_exact=sum(sc["mismatches"] == 0 for sc in scans), scans_listed=len(listed),
                listed_mismatch_range=[min(sc["mismatches"] for sc in listed), max(sc["mismatches"] for sc in listed)] if listed else None,
                voxels_listed=sum(len(sc["voxels"]) for sc in listed), at_threshold=len(at),
                at_threshold_ipldt_only=sum(v["sign"] > 0 for v in at), at_threshold_ipl_only=sum(v["sign"] < 0 for v in at),
                scans_all_at_threshold=sum(all(v["offset"] == 0 for v in sc["voxels"]) for sc in listed),
                other=len(other), other_ipldt_only=sum(v["sign"] > 0 for v in other), other_ipl_only=sum(v["sign"] < 0 for v in other),
                other_levels_above_min=int(lv.min()), other_levels_above_median=float(np.median(lv)),
                other_levels_above_max=int(lv.max()), other_saturated=sum(v["short"] == I16 for v in other),
                other_below_threshold=sum(v["short"] < THR for v in other),
                other_clusters=len(cl), other_cluster_size_min=min(cl) if cl else None, other_cluster_size_max=max(cl) if cl else None,
                scans_unlisted=len(unlisted), voxels_unlisted=sum(sc["mismatches"] for sc in unlisted),
                unlisted_min=min((sc["mismatches"] for sc in unlisted), default=None),
                list_rule_holds=all(sc["mismatches"] <= SEG_LIST_MAX for sc in listed)
                and all(sc["mismatches"] > SEG_LIST_MAX for sc in unlisted))


# ================================================================================================= main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", choices=["oslh", "patella"], default=None, help="legend check instead of the figure")
    args = ap.parse_args()
    if args.verify:
        verify(args.verify)
        return
    sl = load_slice_data()
    say("cohort records")
    rows, summary, per_site = load_cohort()
    # the legend's voxel-level census, from the --verify outputs (recomputed from their voxel lists here)
    NUM["verify_census"] = {}
    for cohort in ("oslh", "patella"):
        p = os.path.join(HERE, f"fig3_lh_verify_{cohort}.json")
        if os.path.exists(p):
            vs = json.load(open(p))["scans"]
            c = dict(all=census(vs))
            if cohort == "oslh":
                c["ultradistal"] = census([x for x in vs if x["scan"].startswith("Distal_")])
                c["diaphyseal"] = census([x for x in vs if x["scan"].startswith("Diaphyseal_")])
                assert c["all"]["scans"] == N_OSLH and c["all"]["voxels_listed"] + c["all"]["voxels_unlisted"] == \
                    summary["oslh"]["mismatches"], "the verify file does not cover the current radius/tibia set"
            NUM["verify_census"][cohort] = c
    say("drawing")
    fig = draw(sl, rows, summary, per_site)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_SVG)
    plt.close(fig)
    with open(OUT_NUM, "w") as fh:
        json.dump(NUM, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    say(f"written {OUT_PNG}, {OUT_SVG}, {OUT_NUM}")


if __name__ == "__main__":
    main()
