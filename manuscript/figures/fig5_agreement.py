"""fig5_agreement.py -- Figure 5 of the ipldt / ORMIR-BQRL manuscript: end-to-end agreement with IPL V5.42, n = 137.

Twelve panels (A-L), 180 mm wide (two-column), 300 dpi PNG + SVG:
  A-E  regression of ipldt on IPL in configuration B (the whole chain from IPL's periosteal contour): BV/TV, Tb.Th,
       Tb.Sp, Tb.N, Ct.Th
  F    configuration A (IPL's segmentation and contours in; ipldt's distance-transform engine out): differing voxels
       per scan and map (4 maps x 137 scans) -- 0 everywhere but one cell
  G-K  Bland-Altman of the same five metrics (difference ipldt - IPL against the mean of the two; bias, 95 % limits
       of agreement)
  L    configuration B, Laplace-Hamming segmentation: differing voxels per scan, by cohort

The validation set: n = 137 (patella 21; ultradistal and diaphyseal radius and tibia 29 each).
The figure pools the two diaphyseal sites as one cohort (n = 58), with the same Okabe-Ito color
as Figures 3 and 6.

Every number is regenerated here from the validation records (never from figure sidecars):
  patella, configuration A voxel counts   validation/results/records.json (105 = 21 subjects x 5 maps)
  patella, configuration B (SEG, metrics) validation/results/from_ipl_contour_ceil_dt/records/*.json (the dt-inclusive run
                                          under the shipped engine; the single source of every patella configuration-B number)
  radius/tibia, A and B                   validation/results/{oslh_auto_vN,oslh_noedit_vN}/records/*.json matching
                                          <Group>_<Study>_<n>.json, minus the record(s) listed in
                                          manuscript/facts/facts.json meta.sources.oslh_dropped (Diaphyseal_CKD_991161)
The statistics follow manuscript/facts/build_facts.py::stats_pair exactly (bias, SD, LoA, OLS, R^2, ICC(2,1); a pair whose
difference is below 1e-12 relative is 'equal', printed "equal N/137"; 'identical' is kept for voxel maps with 0 differing
voxels) and are cross-checked against facts.json at the end of the run; every
number the figure prints is also checked against facts.json (fig5_agreement_numbers.json, 'checks').

Tables 2 and 3 (manuscript/tables/table2_modules.*, table3_metrics.*) are written from facts.json and, for the cortical
pore map and Ct.Po rows, from manuscript/facts/FACTS_BMD_CTPO.json (keys named in the last column).
With IPLDT_TABLE3_PROPOSAL=1 a proposal for per-site rows of Table 3 is also written to manuscript/drafts/ (not part of
Table 3).

Run from the repository root with the ormir python:
    set PYTHONUTF8=1
    python manuscript/figures/fig5_agreement.py
Outputs: manuscript/figures/fig5_agreement.png (300 dpi), .svg, fig5_agreement_stats.csv (the regenerated statistics),
         fig5_agreement_pairs.csv (the per-scan pairs drawn), fig5_agreement_numbers.json (every printed number and its
         facts.json check), manuscript/tables/table2_modules.csv/.md, table3_metrics.csv/.md
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from collections import OrderedDict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
FACTS_JSON = os.path.join(REPO, "manuscript", "facts", "facts.json")
FACTS_BMD_JSON = os.path.join(REPO, "manuscript", "facts", "FACTS_BMD_CTPO.json")
TABLES_DIR = os.path.join(REPO, "manuscript", "tables")
DRAFTS_DIR = os.path.join(REPO, "manuscript", "drafts")
OUT_STEM = os.path.join(HERE, "fig5_agreement")

PAT_A_RECORDS = os.path.join(REPO, "validation", "results", "records.json")
PAT_B_SEG_DIR = os.path.join(REPO, "validation", "results", "from_ipl_contour_ceil_dt", "records")
PAT_B_DT_DIR = PAT_B_SEG_DIR          # one run, every stage (SEG and the dt maps / metrics come from the same records)
OSLH_DIR = os.path.join(REPO, "validation", "results", OSLH_AUTO, "records")         # the 62 edited-cohort scans (shipped SEG order)
OSLH_NOEDIT_DIR = os.path.join(REPO, "validation", "results", OSLH_NOEDIT, "records")  # the 54 diaphyses of the second radius / tibia set
OSLH_DIRS = (OSLH_DIR, OSLH_NOEDIT_DIR)
OSLH_GLOB = re.compile(r"^[A-Za-z]+_[A-Za-z]+_\d+\.json$")

# the validation set this figure is drawn for; asserted against the records and against facts.json
EXPECTED_N = 137
EXPECTED_SITES = {"patella": 21, "UD radius": 29, "UD tibia": 29, "D radius": 29, "D tibia": 29}

# ------------------------------------------------------------------------------------------------ style
MM = 1 / 25.4
FIG_W_MM, FIG_H_MM = 180.0, 212.0
# Okabe-Ito (colour-blind safe); every cohort also has its own marker shape, so identity is never colour alone
COHORTS = ["patella", "UD radius", "UD tibia", "diaphyseal"]
COHORT_LABEL = {"patella": "patella", "UD radius": "ultradistal radius", "UD tibia": "ultradistal tibia",
                "diaphyseal": "diaphyseal radius / tibia"}
COHORT_COLOR = {"patella": "#0072B2", "UD radius": "#009E73", "UD tibia": "#E69F00", "diaphyseal": "#CC79A7"}
COHORT_MARKER = {"patella": "o", "UD radius": "^", "UD tibia": "s", "diaphyseal": "D"}
INK, INK2, INK3, GRID = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#e3e3e3"
METRICS = [("BVTV", "BV/TV", ""), ("TbTh", "Tb.Th", "mm"), ("TbSp", "Tb.Sp", "mm"), ("TbN", "Tb.N", "1/mm"), ("CtTh", "Ct.Th", "mm")]
MAPS_A = [("TbTh", "Tb.Th"), ("TbSp", "Tb.Sp"), ("TbN", "1/Tb.N"), ("CtTh", "Ct.Th")]
FACTS_METRIC_KEY = {"BVTV": "BVTV", "TbTh": "TbTh_trabseg", "TbSp": "TbSp", "TbN": "TbN", "CtTh": "CtTh"}
FACTS_MAP_KEY = {"TbTh": "TbTh_trabseg", "TbSp": "TbSp", "TbN": "TbN", "CtTh": "CtTh"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.grid": False, "axes.axisbelow": True,
    "savefig.dpi": 300, "figure.dpi": 100, "svg.fonttype": "none", "svg.hashsalt": "ipldt-fig5",
    "pdf.fonttype": 42, "axes.unicode_minus": True,
})


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------ statistics
def icc21(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement (as in build_facts.py)."""
    data = np.stack([x, y], axis=1)
    n, k = data.shape
    grand = data.mean()
    rm, cm = data.mean(axis=1), data.mean(axis=0)
    msr = k * ((rm - grand) ** 2).sum() / (n - 1)
    msc = n * ((cm - grand) ** 2).sum() / (k - 1)
    sse = ((data - rm[:, None] - cm[None, :] + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den != 0 else 1.0


def stats_pair(ipl, ours):
    """Sample-wise agreement of ipldt against IPL (the reference), identical to build_facts.py::stats_pair."""
    x = np.asarray(ipl, dtype=np.float64)
    y = np.asarray(ours, dtype=np.float64)
    n = int(x.size)
    d = y - x
    noise = np.abs(d) <= 1e-12 * np.maximum(1.0, np.abs(x))   # summation-order noise between identical maps
    d = np.where(noise, 0.0, d)
    y = x + d
    out = OrderedDict(n=n, mean_ipl=float(x.mean()), sd_ipl=float(x.std(ddof=1)), mean_ours=float(y.mean()),
                      sd_ours=float(y.std(ddof=1)), bias=float(d.mean()), sd_diff=float(d.std(ddof=1)))
    out["loa_low"] = out["bias"] - 1.96 * out["sd_diff"]
    out["loa_high"] = out["bias"] + 1.96 * out["sd_diff"]
    out["max_abs_diff"] = float(np.abs(d).max())
    rel = np.abs(d) / np.abs(x) * 100.0
    out["max_rel_pct"] = float(rel.max())
    out["identical"] = int(noise.sum())
    if out["identical"] == n:
        out["slope"], out["intercept"], out["r2"], out["icc21"] = 1.0, 0.0, 1.0, 1.0
    else:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res, ss_tot = float(((y - yhat) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
        out["slope"], out["intercept"] = float(slope), float(intercept)
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        out["icc21"] = icc21(x, y)
    out["x"], out["y"], out["d"] = x, y, d
    return out


# ------------------------------------------------------------------------------------------------ loaders
def read_json(p):
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def site_of(bone_key):
    """'UD Radius' -> 'UD radius'; the two diaphyseal sites are pooled as 'diaphyseal' in the figure."""
    region, bone = str(bone_key).split()
    if region == "D":
        return "diaphyseal"
    return f"{region} {bone.lower()}"


def fine_site_of(bone_key):
    """'D Radius' -> 'D radius' (the five sites of Table 1, used for the counts; the figure pools D radius / D tibia)."""
    region, bone = str(bone_key).split()
    return f"{region} {bone.lower()}"


def load_scans(oslh_dirs=OSLH_DIRS):
    """One row per scan: cohort, configuration-B metric pairs (IPL, ipldt), configuration-A differing voxels per map,
    configuration-B differing SEG voxels and Dice. `oslh_dirs` defaults to the n = 137 set (oslh_auto_vN + oslh_noedit_vN);
    (OSLH_DIR,) reads the first radius / tibia set only (manuscript/figures/pooled_statistics.py)."""
    facts = read_json(FACTS_JSON)
    dropped = set(facts["meta"]["sources"].get("oslh_dropped", []))
    scans = []
    # --- patella
    pat_a = {}
    for rec in read_json(PAT_A_RECORDS):
        pat_a.setdefault(rec["subject"], {})[rec["metric"]] = rec
    pat_seg = {os.path.basename(p)[:-5]: read_json(p) for p in sorted(glob.glob(os.path.join(PAT_B_SEG_DIR, "*.json")))
               if os.path.basename(p).count(".") == 1}
    pat_dt = {os.path.basename(p)[:-5]: read_json(p) for p in sorted(glob.glob(os.path.join(PAT_B_DT_DIR, "*.json")))
              if os.path.basename(p).count(".") == 1}
    for sid in sorted(pat_seg):
        a, s, b = pat_a[sid], pat_seg[sid], pat_dt[sid]
        mo, mp = b["morphometry"], b["maps"]
        # Tb.Th is dt_thickness on TRAB_SEG cropped to the trabecular contour (Script 32): patella IPL file <base>_TRAB_TH_old
        metrics = dict(
            BVTV=(float(b["bvtv"]["ipl"]), float(b["bvtv"]["ours"])),
            TbTh=(float(mp["TRAB_TH_old"]["mean_ipl_mm"]), float(mo["Tb_Th_old_mm"])),
            TbSp=(float(mp["TRAB_SP"]["mean_ipl_mm"]), float(mo["Tb_Sp_mm"])),
            TbN=(1.0 / float(mp["TRAB_1N"]["mean_ipl_mm"]), float(mo["Tb_N_per_mm"])),
            CtTh=(float(mp["CORT_TH"]["mean_ipl_mm"]), float(mo["Ct_Th_mm"])))
        vox_a = dict(TbTh=int(a["TbTh_old"]["mismatches"]), TbSp=int(a["TbSp"]["mismatches"]), TbN=int(a["TbN"]["mismatches"]),
                     CtTh=int(a["CtTh"]["mismatches"]))
        nvox_a = dict(TbTh=int(a["TbTh_old"]["n_voxels"]), TbSp=int(a["TbSp"]["n_voxels"]), TbN=int(a["TbN"]["n_voxels"]),
                      CtTh=int(a["CtTh"]["n_voxels"]))
        seg = s["seg"]["SEG"]
        scans.append(dict(id=sid, cohort="patella", site="patella", source="patella", metrics=metrics, vox_a=vox_a,
                          nvox_a=nvox_a, seg_mism=int(seg["mismatches"]), seg_dice=float(seg["dice"]),
                          seg_union=int(seg["union_voxels"])))
    # --- radius / tibia (every directory of the set; a record id may appear in only one of them)
    paths = []
    for d in oslh_dirs:
        paths += sorted(glob.glob(os.path.join(d, "*.json")))
    seen = {}
    for p in paths:
        name = os.path.basename(p)
        if not OSLH_GLOB.match(name) or name[:-5] in dropped:
            continue
        if name in seen:
            raise SystemExit(f"{name} is in both {seen[name]} and {os.path.dirname(p)}")
        seen[name] = os.path.dirname(p)
        r = read_json(p)
        assert r["trab_th_definition"]["value"] == "trab_seg", name
        A, B = r["A"], r["B"]
        mi, mo = B["metrics_ipl"], B["metrics_ours"]
        metrics = dict(BVTV=(float(mi["BVTV"]), float(mo["BVTV"])), TbTh=(float(mi["TbTh_tseg"]), float(mo["TbTh_tseg"])),
                       TbSp=(float(mi["TbSp"]), float(mo["TbSp"])), TbN=(float(mi["TbN"]), float(mo["TbN"])),
                       CtTh=(float(mi["CtTh"]), float(mo["CtTh"])))
        am = A["maps"]
        vox_a = dict(TbTh=int(am["TRAB_TH_tseg"]["mismatches"]), TbSp=int(am["TRAB_SP"]["mismatches"]),
                     TbN=int(am["TRAB_1N"]["mismatches"]), CtTh=int(am["CORT_TH"]["mismatches"]))
        nvox_a = dict(TbTh=int(am["TRAB_TH_tseg"]["n_voxels"]), TbSp=int(am["TRAB_SP"]["n_voxels"]),
                      TbN=int(am["TRAB_1N"]["n_voxels"]), CtTh=int(am["CORT_TH"]["n_voxels"]))
        seg = B["seg"]["SEG"]
        scans.append(dict(id=name[:-5], cohort=site_of(r["meta"]["bone_key"]), site=fine_site_of(r["meta"]["bone_key"]),
                          source=os.path.basename(os.path.dirname(os.path.dirname(p))), metrics=metrics, vox_a=vox_a,
                          nvox_a=nvox_a, seg_mism=int(seg["mismatches"]), seg_dice=float(seg["dice"]),
                          seg_union=int(seg["union_voxels"])))
    order = {c: i for i, c in enumerate(COHORTS)}
    scans.sort(key=lambda s: (order[s["cohort"]], s["id"]))
    return scans, facts


# ------------------------------------------------------------------------------------------------ formatting
SUP = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")
SUP_FONTS = ["Arial", "DejaVu Sans"]      # Arial has the superscript digits; the superscript minus comes from DejaVu Sans


def sup(e):
    """An integer exponent as Unicode superscripts (a 7-pt glyph, not a 4.9-pt mathtext script): -6 -> '⁻⁶'."""
    return str(int(e)).translate(SUP)


def sci(v, digits=2):
    """3.44e-06 -> '3.4 × 10⁻⁶' (Unicode superscripts); 0 -> '0'."""
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** e
    return f"{m:.{digits - 1}f} \u00d7 10{sup(e)}".replace("-", "\u2212")


def unit_str(u):
    return f" ({u})" if u else ""


def fmt_int(n):
    """Thousands separated by commas, as in the legends and the text (American style)."""
    return f"{int(n):,d}"


def panel_letter(ax, letter):
    ax.text(-0.22, 1.06, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, ha="left", va="bottom")


def nice_limits(vals, pad=0.06):
    lo, hi = float(np.min(vals)), float(np.max(vals))
    span = hi - lo if hi > lo else abs(hi) or 1.0
    return lo - pad * span, hi + pad * span


def scatter_by_cohort(ax, scans, xs, ys, size=8):
    # the largest cohort (58 diaphyses) is drawn first, so the smaller cohorts stay visible on top of it
    for c in sorted(COHORTS, key=lambda c: -sum(1 for s in scans if s["cohort"] == c)):
        idx = [i for i, s in enumerate(scans) if s["cohort"] == c]
        if not idx:
            continue
        ax.scatter(xs[idx], ys[idx], s=size, marker=COHORT_MARKER[c], facecolors=COHORT_COLOR[c], edgecolors="white",
                   linewidths=0.35, zorder=3)


# ------------------------------------------------------------------------------------------------ panels
def draw_regression(ax, scans, key, label, unit, st, letter):
    x, y = st["x"], st["y"]
    lo, hi = nice_limits(np.concatenate([x, y]))
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color=INK3, zorder=1)
    if st["identical"] < st["n"]:
        ax.plot([lo, hi], [st["slope"] * lo + st["intercept"], st["slope"] * hi + st["intercept"]], lw=0.7, color=INK, zorder=2)
    scatter_by_cohort(ax, scans, x, y)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)
    ax.set_xlabel(f"IPL {label}{unit_str(unit)}")
    ax.set_ylabel(f"ipldt {label}{unit_str(unit)}")
    ax.set_title(label, pad=3)
    ax.locator_params(nbins=4)
    # the data lie on the diagonal, so the lower-right triangle is always free of points
    txt = (f"slope {st['slope']:.6f}\nR\u00b2 {st['r2']:.6f}\nICC {st['icc21']:.6f}\n"
           f"equal {st['identical']}/{st['n']}")          # scalars: 'equal' (12 s.f.); 'identical' is for voxel maps
    ax.text(0.96, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=INK, linespacing=1.25)
    panel_letter(ax, letter)


def draw_bland_altman(ax, scans, key, label, unit, st, letter):
    x, y, d = st["x"], st["y"], st["d"]
    m = (x + y) / 2
    scale_ref = max(abs(st["loa_low"]), abs(st["loa_high"]), st["max_abs_diff"])
    e = int(math.floor(math.log10(scale_ref))) if scale_ref > 0 else -6
    f = 10.0 ** e
    top = 1.25 * scale_ref / f if scale_ref > 0 else 1.25
    ax.axhline(0, lw=0.5, color=INK3, zorder=1)
    ax.axhline(st["bias"] / f, lw=0.8, color=INK, zorder=2)
    ax.axhline(st["loa_low"] / f, lw=0.7, ls="--", color=INK, zorder=2)
    ax.axhline(st["loa_high"] / f, lw=0.7, ls="--", color=INK, zorder=2)
    scatter_by_cohort(ax, scans, m, d / f)
    xlo, xhi = nice_limits(m)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-top, 2.15 * top)          # the band above 1.25 x the data range holds the annotation
    ax.set_xlabel(f"mean of IPL and ipldt {label}{unit_str(unit)}")
    ax.set_ylabel(f"ipldt \u2212 IPL {label} (\u00d710{sup(e)}" + (f" {unit})" if unit else ")"), fontfamily=SUP_FONTS)
    ax.set_title(label, pad=3)
    ax.locator_params(axis="x", nbins=4)
    ax.locator_params(axis="y", nbins=5)
    if st["identical"] == st["n"]:
        txt = f"all {st['n']} differences = 0"
    else:
        txt = (f"bias {sci(st['bias'])}\nLoA {sci(st['loa_low'])} to {sci(st['loa_high'])}\n"
               f"max |rel.| {st['max_rel_pct']:.3f}%")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK, linespacing=1.25,
            fontfamily=SUP_FONTS)
    panel_letter(ax, letter)


def draw_config_a_matrix(ax, scans, letter):
    """4 maps x 137 scans, configuration A: the number of differing voxels per (scan, map); a cell is white for 0 and
    filled with its cohort color for anything else. At 137 columns in a one-third-width panel a cell is ~0.33 mm wide,
    so cells carry no outline: each map row is one outlined strip, cohorts are separated by vertical rules, and every
    non-zero cell is ringed and called out."""
    n = len(scans)
    M = np.array([[s["vox_a"][k] for s in scans] for k, _ in MAPS_A], dtype=np.int64)   # (4, n)
    total_vox = sum(sum(s["nvox_a"].values()) for s in scans)
    total_mism = int(M.sum())
    n_nonzero = int((M > 0).sum())
    starts = {}
    for i, s in enumerate(scans):
        starts.setdefault(s["cohort"], [i, i])
        starts[s["cohort"]][1] = i
    # rows: one outlined white strip per map; non-zero cells filled
    for r in range(M.shape[0]):
        ax.add_patch(Rectangle((0, r + 0.08), n, 0.84, facecolor="white", edgecolor=INK3, linewidth=0.5, zorder=2))
    for r, c in zip(*np.nonzero(M)):
        ax.add_patch(Rectangle((c, r + 0.08), 1, 0.84, facecolor=COHORT_COLOR[scans[c]["cohort"]], edgecolor="none", zorder=3))
    # cohort boundaries through the rows
    for c, (i0, i1) in starts.items():
        if i0 > 0:
            ax.plot([i0, i0], [0.08, M.shape[0] - 0.08], lw=0.5, color=GRID, zorder=2.5)
    # cohort bands along the top
    for c, (i0, i1) in starts.items():
        ax.add_patch(Rectangle((i0, M.shape[0] + 0.25), i1 - i0 + 1, 0.45, facecolor=COHORT_COLOR[c], edgecolor="white",
                               linewidth=0.6, zorder=2))
        short = {"patella": "patella", "UD radius": "UD rad.", "UD tibia": "UD tib.", "diaphyseal": "diaphyseal"}[c]
        if c == "patella":   # 21 of 137 columns: too narrow for its label -> one tier higher, left-aligned on the band
            ax.text(i0, M.shape[0] + 1.75, short, ha="left", va="bottom", fontsize=7, color=INK)
            ax.plot([i0 + 0.5, i0 + 0.5], [M.shape[0] + 0.75, M.shape[0] + 1.7], lw=0.5, color=INK3, zorder=1)
        else:
            ax.text((i0 + i1 + 1) / 2, M.shape[0] + 0.85, short, ha="center", va="bottom", fontsize=7, color=INK)
    # callouts for the non-zero cells: an open ring on the cell and a label below the strip
    for r, c in zip(*np.nonzero(M)):
        v = int(M[r, c])
        ax.plot([c + 0.5], [r + 0.5], ls="", marker="o", ms=5.5, mfc="none", mec=INK, mew=0.7, zorder=5)
        ax.annotate(f"{v} voxel" + ("" if v == 1 else "s"), xy=(c + 0.5, r + 0.5), xytext=(max(0.07 * n, c + 0.5 - 0.2 * n), r - 1.3),
                    fontsize=7, color=INK, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=0, shrinkB=3.2), zorder=4)
    ax.set_xlim(0, n)
    ax.set_ylim(-4.6, M.shape[0] + 2.9)
    ax.set_yticks(np.arange(M.shape[0]) + 0.5)
    ax.set_yticklabels([lab + " map" for _, lab in MAPS_A])
    # ticks at the first scan of each cohort and at the last scan
    firsts = sorted(i0 for i0, _ in starts.values())
    ticks = [i0 + 1 for i0 in firsts] + [n]
    ax.set_xticks(np.array(ticks) - 0.5)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel("scan")
    ax.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_bounds(0, n)
    ax.set_title("configuration A (distance transforms):\ndiffering voxels per scan and map", pad=3)
    verb = "differs" if total_mism == 1 else "differ"
    ax.text(n, -1.9, f"{fmt_int(total_vox)} voxels compared\n{fmt_int(total_mism)} {verb}; {M.size - n_nonzero} of {M.size}\n"
                     f"maps identical",
            ha="right", va="top", fontsize=7, color=INK, linespacing=1.25)
    panel_letter(ax, letter)
    return dict(total_vox=total_vox, total_mism=total_mism, n_cells=int(M.size), n_nonzero=n_nonzero, M=M)


def draw_seg_strip(ax, scans, letter, seed=7):
    """Configuration B, Laplace-Hamming SEG: differing voxels per scan by cohort (symlog), median per cohort."""
    rng = np.random.default_rng(seed)
    xs = {c: i for i, c in enumerate(COHORTS)}
    allv = np.array([s["seg_mism"] for s in scans])
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    for c in COHORTS:
        v = np.array([s["seg_mism"] for s in scans if s["cohort"] == c], dtype=float)
        if v.size == 0:
            continue
        jitter = (rng.random(v.size) - 0.5) * 0.62
        ax.scatter(xs[c] + jitter, v, s=8, marker=COHORT_MARKER[c], facecolors=COHORT_COLOR[c], edgecolors="white",
                   linewidths=0.35, zorder=3)
        med = float(np.median(v))
        ax.plot([xs[c] - 0.34, xs[c] + 0.34], [med, med], lw=0.9, color=INK, zorder=4)
    ax.set_xticks(range(len(COHORTS)))
    ax.set_xticklabels(["patella", "UD\nradius", "UD\ntibia", "diaph."])
    ax.set_xlim(-0.6, len(COHORTS) - 0.4)
    ax.set_yticks([0, 1, 10, 100, 1000, 10000, 100000])
    # plain numbers with comma thousands separators, as Figures 3C and 6C print them
    ax.set_yticklabels([fmt_int(t) for t in (0, 1, 10, 100, 1000, 10000, 100000)])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(-0.6, 2e6)
    ax.set_ylabel("differing SEG voxels per scan")
    ax.set_title("configuration B:\ndiffering SEG voxels per scan", pad=3)
    n_exact = int((allv == 0).sum())
    dice_min = min(s["seg_dice"] for s in scans)
    total = int(allv.sum())
    union = sum(s["seg_union"] for s in scans)
    txt = (f"identical {n_exact}/{len(scans)}\nmedian {int(np.median(allv))} voxels\n"
           f"min Dice {dice_min:.4f}")
    ax.text(0.04, 0.96, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7, color=INK, linespacing=1.25)
    panel_letter(ax, letter)
    return dict(n_exact=n_exact, dice_min=dice_min, total=total, union=union, median=float(np.median(allv)))


# ------------------------------------------------------------------------------------------------ tables
MINUS = "−"


def fval_raw(d, key):
    v = d[key]
    return v["value"] if isinstance(v, dict) and "value" in v else v


def load_all_facts(facts):
    """facts.json plus FACTS_BMD_CTPO.json (pore map, Ct.Po) in one key space; the two key sets are disjoint."""
    bmd = read_json(FACTS_BMD_JSON)
    clash = set(facts["facts"]) & set(bmd["facts"])
    assert not clash, sorted(clash)[:5]
    assert bmd["meta"]["n"] == fval_raw(facts["facts"], "cohort.n_total"), (bmd["meta"]["n"], "FACTS_BMD_CTPO is not the same set")
    merged = dict(facts["facts"])
    merged.update(bmd["facts"])
    return {"facts": merged, "meta": facts["meta"], "bmd_meta": bmd["meta"]}


def fval(facts, key):
    return fval_raw(facts["facts"], key)


def um(s):
    """ASCII hyphen-minus in front of a number -> the typographic minus the manuscript tables use."""
    return re.sub(r"(^|[\s\[(,])-(?=\d)", lambda m: m.group(1) + MINUS, s)


def fmt_mean_sd(m, s, dec):
    return um(f"{m:.{dec}f} ± {s:.{dec}f}")


def fmt_e(v):
    if v == 0:
        return "0"
    return um(f"{v:.2e}".replace("e-0", "e-").replace("e+0", "e+"))


def write_table(rows, cols, stem, title, notes=()):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    lines = [f"**{title}**", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join("---" for _ in cols) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    if notes:
        lines.append("")
        for n_ in notes:
            lines.append(n_)
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {os.path.relpath(stem, REPO)}.csv / .md ({len(rows)} rows)")


def table2_modules(facts):
    F = lambda k: fval(facts, k)  # noqa: E731
    N = F("cohort.n_total")
    N_OSLH = F("cohort.oslh.n")
    rows = []

    def row(module, cmd, comparison, n, compared, differing, exact, dice, key):
        rows.append(OrderedDict([("Module", module), ("IPL command(s) reimplemented", cmd), ("Comparison against IPL", comparison),
                                 ("n", n), ("Voxels compared", compared), ("Differing voxels", differing), ("Identical scans", exact),
                                 ("Dice", dice), ("facts key", key)]))

    row("AIM I/O", "AIM read / write (int16 grayscale, char masks, run-length-compressed products, processing log)",
        "input of every comparison below", N, "—", "—", "—", "—", "cohort.n_total")
    row("Periosteal contour", "ORMIR-XCT autocontour (not an IPL reimplementation)",
        "not compared: IPL's own periosteal contour is the input of configurations A and B", "—", "—", "—", "—", "—", "params.autocontour")
    n_step1 = F("B.patella.step1.cort.n_scans") + F("B.oslh.rendering.cort.n_scans")
    row("Cortical / trabecular compartment separation (STEP 1)",
        "/seg_gauss, /gobj_maskaimpeel_ow, /erosion, /dilation, /close, /open, /cl_ow_rank_extract, /cl_nr_extract, "
        "/cl_slicewise_extractow, /subtract_aims, /add_aims, /bounding_box_cut (36 commands; TIBIA / RADIUS presets)",
        "cortical and trabecular compartment masks (patella: mask rasters; radius / tibia: rendered contours of the masks)",
        n_step1,
        f"{F('B.patella.step1.cort.voxels_ipl') + F('B.patella.step1.trab.voxels_ipl') + F('B.oslh.rendering.cort.voxels_ipl') + F('B.oslh.rendering.trab.voxels_ipl'):,} (IPL mask voxels)",
        F("B.patella.step1.cort.mismatches") + F("B.patella.step1.trab.mismatches") + F("B.oslh.rendering.cort.mismatches") + F("B.oslh.rendering.trab.mismatches"),
        f"{F('B.patella.step1.cort.exact_scans') + F('B.oslh.rendering.cort.exact_scans')}/{n_step1} cortical, "
        f"{F('B.patella.step1.trab.exact_scans') + F('B.oslh.rendering.trab.exact_scans')}/{n_step1} trabecular",
        f"{min(F('B.patella.step1.cort.dice.min'), F('B.patella.step1.trab.dice.min'), F('B.oslh.rendering.cort.dice.min'), F('B.oslh.rendering.trab.dice.min')):.4f}",
        "B.patella.step1.*, B.oslh.rendering.*")
    n_ren = F("B.patella.contour.cort.n_scans") + F("B.oslh.rendering.cort.n_scans")
    row("Contour rendering", "/togobj_from_aim -curvature_smooth 1, /gobj_to_aim",
        "rendered cortical and trabecular contours",
        n_ren,
        f"{F('B.patella.contour.cort.voxels_ipl') + F('B.patella.contour.trab.voxels_ipl') + F('B.oslh.rendering.cort.voxels_ipl') + F('B.oslh.rendering.trab.voxels_ipl'):,} (IPL contour voxels)",
        F("B.patella.contour.cort.mismatches") + F("B.patella.contour.trab.mismatches") + F("B.oslh.rendering.cort.mismatches") + F("B.oslh.rendering.trab.mismatches"),
        f"{F('B.patella.contour.cort.exact_scans') + F('B.oslh.rendering.cort.exact_scans')}/{n_ren} cortical, "
        f"{F('B.patella.contour.trab.exact_scans') + F('B.oslh.rendering.trab.exact_scans')}/{n_ren} trabecular",
        f"{min(F('B.oslh.rendering.cort.dice.min'), F('B.oslh.rendering.trab.dice.min')):.4f}",
        "B.patella.contour.*, B.oslh.rendering.*")
    row("Laplace–Hamming segmentation (STEP 2)",
        "/fft_laplace_hamming (ε 0.45, cutoff 0.3, amplitude 1), /norm_max 200000, /threshold 475–1,000‰, /cl_nr_extract (35 / 70), "
        "/gobj_maskaimpeel_ow, SEG assembly (labels 127 / 126)",
        "SEG (configuration B)", F("B.pooled.SEG.n_scans"), f"{F('B.pooled.SEG.voxels_compared'):,}",
        f"{F('B.pooled.SEG.mismatches'):,} ({F('B.pooled.SEG.ours_only'):,} ipldt only, {F('B.pooled.SEG.ipl_only'):,} IPL only)",
        f"{F('B.pooled.SEG.exact_scans')}/{F('B.pooled.SEG.n_scans')}",
        f"min {F('B.pooled.SEG.dice.min'):.6f}, median {F('B.pooled.SEG.dice.median'):.8f}", "B.pooled.SEG.*")
    # cortical pore extraction, configuration A (FACTS_BMD_CTPO.json pore.A.*)
    pore_cmd = ("/cl_slicewise_extractow (0–5%), /cl_rank_extract, /gobj_maskaimpeel_ow, /hysteresis_threshold, /cl_nr_extract (20), "
                "/set_value, /add_aims, /subtract_aims (the Burghardt pore cascade)")
    row("Cortical pore extraction (STEP 2)", pore_cmd,
        "cortical pore map PORE, configuration A (IPL's cortical segmentation and contour in)",
        F("pore.A.n"), f"{F('pore.A.compared_voxels'):,}", f"{F('pore.A.mismatch_total'):,}",
        f"{F('pore.A.identical')}/{F('pore.A.n')}", f"{F('pore.A.dice_min'):.4f}", "FACTS_BMD_CTPO: pore.A.*")
    dt_rows = [("Tb.Th map", "/dt_thickness (trabecular segmentation, trabecular contour)", "TbTh_trabseg"),
               ("Tb.Sp map", "/dt_spacing", "TbSp"), ("1/Tb.N map", "/dt_number", "TbN"),
               ("Ct.Th map", "/dt_thickness (cortical compartment, cortical contour)", "CtTh")]
    for lab, cmd, mk in dt_rows:
        row(f"Distance-transform engine (STEP 3): {lab}", cmd,
            "configuration A (IPL's SEG and contours in)", F(f"A.pooled.map.{mk}.n_scans"), f"{F(f'A.pooled.map.{mk}.voxels_compared'):,}",
            f"{F(f'A.pooled.map.{mk}.mismatches'):,}", f"{F(f'A.pooled.map.{mk}.exact_scans')}/{F(f'A.pooled.map.{mk}.n_scans')}",
            "— (value map)", f"A.pooled.map.{mk}.*")
    # cortical pore map, whole chain (configuration B)
    row("Whole chain: cortical pore map", "the pore cascade, from ipldt's own cortical segmentation and contour",
        "configuration B (IPL's periosteal contour in)", F("pore.B.n"), f"{F('pore.B.compared_voxels'):,}",
        f"{F('pore.B.mismatch_total'):,} ({F('pore.B.ours_only'):,} ipldt only, {F('pore.B.ipl_only'):,} IPL only)",
        f"{F('pore.B.identical')}/{F('pore.B.n')}", f"{F('pore.B.dice_min'):.4f}–1.0000", "FACTS_BMD_CTPO: pore.B.*")
    for lab, cmd, mk in dt_rows:
        row(f"Whole chain: {lab}", cmd, "configuration B (IPL's periosteal contour in)",
            F(f"B.pooled.map.{mk}.n_scans"), f"{F(f'B.pooled.map.{mk}.voxels_compared'):,}", f"{F(f'B.pooled.map.{mk}.mismatches'):,}",
            f"{F(f'B.pooled.map.{mk}.exact_scans')}/{F(f'B.pooled.map.{mk}.n_scans')}", "— (value map)", f"B.pooled.map.{mk}.*")
    row("Metrics (BV/TV, Tb.Th, Tb.Sp, Tb.N, Ct.Th, Ct.Po)", "IPL evaluation statistics (mean of each map inside its contour; Tb.N = 1 / mean of the 1/Tb.N map; "
        "Ct.Po = pore-map voxels / rendered cortical contour voxels)",
        "Table 3 and Figures 5 and 6", F("metrics.B.pooled.BVTV.n"), "—", "—", "—", "—", "metrics.B.pooled.*")
    row("Outputs", "report JSON / CSV, masks and maps as NIfTI or AIM, 3D Slicer segmentation and labelmap, manual-correction re-entry",
        "—", "—", "—", "—", "—", "—", "software.outputs")
    notes = [
        "Voxels are compared by global position over the union of the two grids; a scan is identical when 0 voxels differ. "
        "Dice is given for binary masks and segmentations only. Configuration A: IPL's segmentation and contours in; ipldt's distance-transform "
        "engine and cortical pore cascade out. Configuration B: IPL's periosteal contour in; ipldt's compartment separation, contour rendering, "
        "Laplace–Hamming segmentation, cortical pore cascade and distance transforms out. Radius / tibia compartment masks are compared through their rendered contours "
        f"(the same files verify STEP 1 and the contour rendering for those {N_OSLH} scans).",
    ]
    write_table(rows, list(rows[0].keys()), os.path.join(TABLES_DIR, "table2_modules"),
                f"Table 2. Modules of ipldt, the IPL commands each reimplements and the voxel-level agreement with IPL V5.42 (n = {N}).",
                notes)
    return rows


def metric_row(F, cfg, cohort_label, label, unit, p, dec):
    n = F(p + ".n")
    return OrderedDict([
        ("Config.", cfg), ("Cohort", cohort_label), ("Metric", label + (f" ({unit})" if unit else "")), ("n", n),
        ("IPL mean ± SD", fmt_mean_sd(F(p + ".mean_ipl"), F(p + ".sd_ipl"), dec)),
        ("ipldt mean ± SD", fmt_mean_sd(F(p + ".mean_ours"), F(p + ".sd_ours"), dec)),
        ("Bias (ipldt − IPL)", fmt_e(F(p + ".bias"))),
        ("95% LoA", f"[{fmt_e(F(p + '.loa_low'))}, {fmt_e(F(p + '.loa_high'))}]"),
        ("Slope", f"{F(p + '.slope'):.6f}"), ("R²", f"{F(p + '.r2'):.7f}"), ("ICC(2,1)", f"{F(p + '.icc21'):.7f}"),
        ("Largest relative difference (%)", f"{F(p + '.max_rel_pct'):.4f}"), ("Equal (12 s.f.)", f"{F(p + '.exact')}/{n}"),
        ("facts key", p + ".*")])


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
DEC = {"BVTV": 4, "TbTh": 4, "TbSp": 4, "TbN": 3, "CtTh": 3}


def table3_metrics(facts):
    F = lambda k: fval(facts, k)  # noqa: E731
    N = F("cohort.n_total")
    rows = []
    groups = [("B", "pooled", None), ("B", "patella", "patella"), ("B", "oslh", "radius / tibia"), ("A", "pooled", None)]
    for cfg, grp, grp_label in groups:
        for key, label, unit in METRICS:
            p = f"metrics.{cfg}.{grp}.{FACTS_METRIC_KEY[key]}"
            n = F(p + ".n")
            assert grp != "pooled" or n == N == EXPECTED_N, (p, n)   # every pooled row, configuration A BV/TV included
            rows.append(metric_row(F, cfg, grp_label or f"all (n = {n})", label, unit, p, DEC[key]))
    # Ct.Po, configurations A and B (FACTS_BMD_CTPO.json pore.A.* / pore.B.*): IPL's own exported pore map is the reference
    for cfg in ("A", "B"):
        q = f"pore.{cfg}"
        n = F(q + ".n")
        assert n == N, (q, n)
        loa = F(q + ".loa")
        rows.append(OrderedDict([
            ("Config.", cfg), ("Cohort", f"all (n = {n})"), ("Metric", "Ct.Po"), ("n", n),
            ("IPL mean ± SD", fmt_mean_sd(F(q + ".ctpo_ipl_mean"), F(q + ".ctpo_ipl_sd"), 4)),
            ("ipldt mean ± SD", fmt_mean_sd(F(q + ".ctpo_mean"), F(q + ".ctpo_sd"), 4)),
            ("Bias (ipldt − IPL)", fmt_e(F(q + ".bias"))),
            ("95% LoA", f"[{fmt_e(loa[0])}, {fmt_e(loa[1])}]"),
            ("Slope", f"{F(q + '.slope'):.6f}"), ("R²", f"{F(q + '.r2'):.7f}"), ("ICC(2,1)", f"{F(q + '.icc'):.7f}"),
            ("Largest relative difference (%)", f"{F(q + '.max_abs_rel_pct'):.4f}"),
            ("Equal (12 s.f.)", f"{F(q + '.identical_ctpo')}/{n}"),
            ("facts key", f"FACTS_BMD_CTPO: {q}.*")]))
    # compartmental BMD is not part of the paper's validation (ORMIR-BQRL computes it with ORMIR-XCT bmd_masked)
    pb = "A.patella.BVTV.printed"
    n_pr, n_ok = F(pb + ".n_with_value"), F(pb + ".agree_after_rounding")
    notes = [
        "IPL is the reference (x). Bias = mean(ipldt − IPL); 95% limits of agreement (LoA) = bias ± 1.96 SD of the differences; slope and R² from "
        "ordinary least squares of ipldt on IPL; ICC(2,1) two-way random effects, absolute agreement, single measurement; largest relative "
        "difference = largest |ipldt − IPL| / IPL over the scans; equal (12 s.f.) = scans whose two values agree to 12 significant digits (when every pair is equal the "
        "fit is the identity). Tb.Th is /dt_thickness on the trabecular segmentation inside the trabecular contour, as in IPL's evaluation scripts. "
        "BV/TV = |segmentation ∩ rendered trabecular contour| / |rendered trabecular contour|; IPL's evaluation script writes no "
        f"BV/TV, so IPL's value is that ratio taken on IPL's own segmentation and contour (for the {n_pr} patellae it equals the value "
        f"printed to three decimals on IPL's evaluation sheet after rounding on {n_ok} of {n_pr} scans, the other {WORDS.get(n_pr - n_ok, n_pr - n_ok)} lying within "
        f"{math.ceil(F(pb + '.differing_max_distance_to_boundary') * 1e5):d} × 10⁻⁵ of a rounding boundary); in configuration A both "
        "implementations take the ratio on the same files, so the two values are equal on every scan. "
        "Ct.Po = |pore map ∩ rendered cortical contour| / |rendered cortical contour|; IPL's value is that ratio on IPL's own exported "
        f"pore map and contour, which exist for all {F('pore.A.n')} scans, so the comparison does not depend on the evaluation sheet (which "
        f"prints Ct.Po for {F('ctpo.printed.sheets_with_values')} of the {F('pore.A.n')}; the ratio reproduces its three decimals on "
        f"{F('ctpo.printed.reproduced_3dp')} of those).",
    ]
    write_table(rows, list(rows[0].keys()), os.path.join(TABLES_DIR, "table3_metrics"),
                "Table 3. Scalar bone morphometry metrics of ipldt against IPL V5.42.", notes)
    return rows


def table3_per_site_proposal(facts):
    """A PROPOSAL, not part of Table 3: configuration-B rows for each of the four radius / tibia sites, from the
    facts.json metrics.B.site.* keys, written to manuscript/drafts/ for the authors to decide on."""
    F = lambda k: fval(facts, k)  # noqa: E731
    rows = []
    sites = [("UD_radius", "ultradistal radius"), ("UD_tibia", "ultradistal tibia"), ("D_radius", "diaphyseal radius"),
             ("D_tibia", "diaphyseal tibia")]
    for site, lab_site in sites:
        for key, label, unit in METRICS:
            p = f"metrics.B.site.{site}.{FACTS_METRIC_KEY[key]}"
            rows.append(metric_row(F, "B", lab_site, label, unit, p, DEC[key]))
    cols = list(rows[0].keys())
    stem = os.path.join(DRAFTS_DIR, "table3_per_site_proposal_n137")
    os.makedirs(DRAFTS_DIR, exist_ok=True)
    lines = ["# Proposal: per-site rows for Table 3 (configuration B), n = 137", "",
             "Generated by manuscript/figures/fig5_agreement.py from facts.json (metrics.B.site.*). NOT part of Table 3 as "
             "generated: the table keeps its pooled / patella / radius-tibia rows. These 20 rows would replace (or follow) the "
             "'radius / tibia' block if the authors choose to split it by site.", "",
             "| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {os.path.relpath(stem, REPO)}.md ({len(rows)} rows, proposal)")
    return rows


# ------------------------------------------------------------------------------------------------ main
def check_printed(checks, what, printed, facts_value, key):
    """One number the figure prints against the same number formatted the same way from facts.json."""
    ok = str(printed) == str(facts_value)
    checks.append(OrderedDict(what=what, printed=str(printed), facts=str(facts_value), key=key, ok=ok))
    if not ok:
        log(f"  PRINTED != FACTS  {what}: figure '{printed}' vs facts '{facts_value}' ({key})")


def main():
    scans, facts_raw = load_scans()
    facts = load_all_facts(facts_raw)
    F = lambda k: fval(facts, k)  # noqa: E731
    n = len(scans)
    counts = {c: sum(1 for s in scans if s["cohort"] == c) for c in COHORTS}
    sites = {k: sum(1 for s in scans if s["site"] == k) for k in EXPECTED_SITES}
    sources = {}
    for s in scans:
        sources[s["source"]] = sources.get(s["source"], 0) + 1
    log(f"{n} scans: {counts}; sites {sites}; sources {sources}")
    assert n == EXPECTED_N == F("cohort.n_total"), (n, F("cohort.n_total"))
    assert sites == EXPECTED_SITES, sites
    assert sites == {"patella": F("cohort.patella.n"), "UD radius": F("cohort.site.UD_radius.n"),
                     "UD tibia": F("cohort.site.UD_tibia.n"), "D radius": F("cohort.site.D_radius.n"),
                     "D tibia": F("cohort.site.D_tibia.n")}, "site counts differ from facts.json"
    assert len({s["id"] for s in scans}) == n, "duplicate scan ids"

    stats = {}
    for key, label, unit in METRICS:
        ipl = [s["metrics"][key][0] for s in scans]
        ours = [s["metrics"][key][1] for s in scans]
        stats[key] = stats_pair(ipl, ours)

    # ---- cross-check against facts.json (the phase-1 sheet); a mismatch is reported, not hidden
    worst = 0.0
    for key, label, unit in METRICS:
        p = f"metrics.B.pooled.{FACTS_METRIC_KEY[key]}"
        for k_mine, k_facts in (("bias", "bias"), ("sd_diff", "sd_diff"), ("loa_low", "loa_low"), ("loa_high", "loa_high"),
                                ("slope", "slope"), ("intercept", "intercept"), ("r2", "r2"), ("icc21", "icc21"),
                                ("max_rel_pct", "max_rel_pct"), ("identical", "exact"), ("mean_ipl", "mean_ipl"), ("sd_ipl", "sd_ipl"),
                                ("n", "n")):
            a, b = float(stats[key][k_mine]), float(F(f"{p}.{k_facts}"))
            err = abs(a - b) / max(1e-300, abs(b)) if b != 0 else abs(a - b)
            worst = max(worst, err)
            if err > 1e-9:
                log(f"  CHECK {label} {k_mine}: regenerated {a!r} vs facts.json {b!r}")
    log(f"cross-check of the regenerated pooled statistics against facts.json: worst relative deviation {worst:.2e}")

    # ---- figure
    fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM))
    gs = gridspec.GridSpec(4, 3, figure=fig, left=0.075, right=0.985, top=0.965, bottom=0.082, wspace=0.45, hspace=0.55,
                           height_ratios=[1.05, 1.05, 0.85, 0.85])
    letters = "ABCDEFGHIJKL"          # capital letters, upper-left of each panel (journal style; as in Figure 2)
    axes = {}
    slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
    for i, (r, c) in enumerate(slots):
        axes[letters[i]] = fig.add_subplot(gs[r, c])
    for i, (key, label, unit) in enumerate(METRICS):
        draw_regression(axes[letters[i]], scans, key, label, unit, stats[key], letters[i])
        draw_bland_altman(axes[letters[6 + i]], scans, key, label, unit, stats[key], letters[6 + i])
    info_a = draw_config_a_matrix(axes["F"], scans, "F")
    info_l = draw_seg_strip(axes["L"], scans, "L")

    # row-group captions (left margin) and a shared legend at the bottom
    handles = [Line2D([], [], ls="", marker=COHORT_MARKER[c], mfc=COHORT_COLOR[c], mec="white", mew=0.35, ms=5,
                      label=f"{COHORT_LABEL[c]} (n = {counts[c]})") for c in COHORTS]
    handles += [Line2D([], [], ls="--", lw=0.7, color=INK3, label="identity (A–E)"),
                Line2D([], [], ls="-", lw=0.7, color=INK, label="OLS fit (A–D); bias (G–K); median (L)"),
                Line2D([], [], ls="--", lw=0.7, color=INK, label="95% limits of agreement (G–K)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.5,
               columnspacing=1.4, borderaxespad=0.2)

    for ext in ("png", "svg"):
        kw = {"metadata": {"Date": None}} if ext == "svg" else {}
        fig.savefig(f"{OUT_STEM}.{ext}", dpi=300, facecolor="white", **kw)
        log(f"wrote {os.path.relpath(OUT_STEM, REPO)}.{ext}")
    plt.close(fig)

    # ---- every number the figure prints, against facts.json formatted the same way
    checks = []
    for key, label, unit in METRICS:
        st, p = stats[key], f"metrics.B.pooled.{FACTS_METRIC_KEY[key]}"
        for k_mine, k_facts in (("slope", "slope"), ("r2", "r2"), ("icc21", "icc21")):
            check_printed(checks, f"{label} {k_mine}", f"{st[k_mine]:.6f}", f"{F(f'{p}.{k_facts}'):.6f}", f"{p}.{k_facts}")
        check_printed(checks, f"{label} equal", f"{st['identical']}/{st['n']}", f"{F(p + '.exact')}/{F(p + '.n')}",
                      f"{p}.exact, {p}.n")
        if st["identical"] == st["n"]:
            check_printed(checks, f"{label} all differences 0", f"all {st['n']} differences = 0",
                          f"all {F(p + '.n')} differences = 0" if F(p + ".exact") == F(p + ".n") else "not all equal",
                          f"{p}.exact")
        else:
            for k_mine in ("bias", "loa_low", "loa_high"):
                check_printed(checks, f"{label} {k_mine}", sci(st[k_mine]), sci(F(f"{p}.{k_mine}")), f"{p}.{k_mine}")
            check_printed(checks, f"{label} max_rel_pct", f"{st['max_rel_pct']:.3f}", f"{F(p + '.max_rel_pct'):.3f}",
                          f"{p}.max_rel_pct")
    check_printed(checks, "F voxels compared", fmt_int(info_a["total_vox"]), fmt_int(F("A.pooled.voxels_compared.four_maps")),
                  "A.pooled.voxels_compared.four_maps")
    check_printed(checks, "F differing voxels", fmt_int(info_a["total_mism"]), fmt_int(F("A.pooled.mismatches.four_maps")),
                  "A.pooled.mismatches.four_maps")
    check_printed(checks, "F identical comparisons", f"{info_a['n_cells'] - info_a['n_nonzero']} of {info_a['n_cells']}",
                  f"{F('A.pooled.exact_comparisons.four_maps')} of {4 * F('cohort.n_total')}",
                  "A.pooled.exact_comparisons.four_maps, 4 x cohort.n_total")
    nz = [(MAPS_A[r][1], scans[c]["id"], int(info_a["M"][r, c])) for r, c in zip(*np.nonzero(info_a["M"]))]
    facts_nz = [(dict(MAPS_A)[k], e["id"], int(e["mismatches"])) for k, fk in FACTS_MAP_KEY.items()
                for e in (facts["facts"].get(f"A.oslh.map.{fk}.nonexact_scans", {}) or {}).get("value", [])]
    check_printed(checks, "F non-zero cells (map, scan, voxels)", nz, facts_nz, "A.oslh.map.*.nonexact_scans")
    check_printed(checks, "L identical", f"{info_l['n_exact']}/{n}", f"{F('B.pooled.SEG.exact_scans')}/{F('B.pooled.SEG.n_scans')}",
                  "B.pooled.SEG.exact_scans, n_scans")
    check_printed(checks, "L median", f"{int(info_l['median'])}", f"{int(F('B.pooled.SEG.per_scan.median'))}", "B.pooled.SEG.per_scan.median")
    check_printed(checks, "L min Dice", f"{info_l['dice_min']:.4f}", f"{F('B.pooled.SEG.dice.min'):.4f}", "B.pooled.SEG.dice.min")
    check_printed(checks, "L differing voxels (legend)", f"{info_l['total']:,}", f"{F('B.pooled.SEG.mismatches'):,}", "B.pooled.SEG.mismatches")
    check_printed(checks, "L voxels compared (legend)", f"{info_l['union']:,}", f"{F('B.pooled.SEG.voxels_compared'):,}",
                  "B.pooled.SEG.voxels_compared")
    check_printed(checks, "legend n patella", counts["patella"], F("cohort.patella.n"), "cohort.patella.n")
    check_printed(checks, "legend n UD radius", counts["UD radius"], F("cohort.site.UD_radius.n"), "cohort.site.UD_radius.n")
    check_printed(checks, "legend n UD tibia", counts["UD tibia"], F("cohort.site.UD_tibia.n"), "cohort.site.UD_tibia.n")
    check_printed(checks, "legend n diaphyseal", counts["diaphyseal"], F("cohort.site.D_radius.n") + F("cohort.site.D_tibia.n"),
                  "cohort.site.D_radius.n + cohort.site.D_tibia.n")
    bad = [c for c in checks if not c["ok"]]
    log(f"printed numbers checked against facts.json: {len(checks) - len(bad)}/{len(checks)} agree")

    # per-cohort SEG summaries of panel L (drawn as medians; quoted in the legend draft only if the authors want them)
    per_cohort_l = OrderedDict()
    for c in COHORTS:
        v = np.array([s["seg_mism"] for s in scans if s["cohort"] == c])
        per_cohort_l[c] = dict(n=int(v.size), identical=int((v == 0).sum()), median=float(np.median(v)), max=int(v.max()),
                               total=int(v.sum()))

    numbers = OrderedDict(
        figure="Figure 5", script="manuscript/figures/fig5_agreement.py", n=n, cohorts=counts, sites=sites, sources=sources,
        facts=dict(facts_json=os.path.relpath(FACTS_JSON, REPO).replace(os.sep, "/"),
                   bmd_ctpo_json=os.path.relpath(FACTS_BMD_JSON, REPO).replace(os.sep, "/"),
                   built=facts["meta"].get("built"), set=facts["meta"].get("set")),
        crosscheck_worst_relative_deviation=worst,
        panels_A_E={label: OrderedDict((k, stats[key][k]) for k in ("n", "slope", "intercept", "r2", "icc21", "identical"))
                    for key, label, _ in METRICS},
        panels_G_K={label: OrderedDict((k, stats[key][k]) for k in ("bias", "sd_diff", "loa_low", "loa_high", "max_abs_diff",
                                                                     "max_rel_pct")) for key, label, _ in METRICS},
        panel_F=dict(voxels_compared=info_a["total_vox"], differing=info_a["total_mism"], comparisons=info_a["n_cells"],
                     nonidentical_comparisons=info_a["n_nonzero"], nonzero_cells=nz),
        panel_L=dict(n=n, identical=info_l["n_exact"], median=info_l["median"], dice_min=info_l["dice_min"],
                     differing=info_l["total"], voxels_compared=info_l["union"], per_cohort=per_cohort_l),
        checks=checks, disagreements=bad)
    with open(OUT_STEM + "_numbers.json", "w", encoding="utf-8") as fh:
        json.dump(numbers, fh, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"wrote {os.path.relpath(OUT_STEM, REPO)}_numbers.json")

    # ---- the regenerated statistics and the pairs drawn, beside the figure
    with open(OUT_STEM + "_stats.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = ["n", "mean_ipl", "sd_ipl", "mean_ours", "sd_ours", "bias", "sd_diff", "loa_low", "loa_high", "max_abs_diff", "max_rel_pct",
                "identical", "slope", "intercept", "r2", "icc21"]
        w.writerow(["configuration", "metric", "unit"] + [("equal_12sf" if c == "identical" else c) for c in cols])  # scalars: equal
        for key, label, unit in METRICS:
            w.writerow(["B", label, unit] + [repr(stats[key][c]) if isinstance(stats[key][c], float) else stats[key][c] for c in cols])
        w.writerow([])
        w.writerow(["configuration A, four maps: voxels compared", info_a["total_vox"], "differing", info_a["total_mism"],
                    "scan-map comparisons", info_a["n_cells"], "non-identical comparisons", info_a["n_nonzero"]])
        w.writerow(["configuration B, SEG: scans", n, "identical scans", info_l["n_exact"], "differing voxels", info_l["total"],
                    "voxels compared", info_l["union"], "median per scan", info_l["median"], "min Dice", repr(info_l["dice_min"])])
    with open(OUT_STEM + "_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scan", "cohort", "site", "source"] + [f"{k}_{side}" for k, _, _ in METRICS for side in ("ipl", "ipldt")]
                   + [f"A_{k}_differing_voxels" for k, _ in MAPS_A] + ["B_SEG_differing_voxels", "B_SEG_dice"])
        for s in scans:
            w.writerow([s["id"], s["cohort"], s["site"], s["source"]] + [repr(s["metrics"][k][i]) for k, _, _ in METRICS for i in (0, 1)]
                       + [s["vox_a"][k] for k, _ in MAPS_A] + [s["seg_mism"], repr(s["seg_dice"])])
    log(f"wrote {os.path.relpath(OUT_STEM, REPO)}_stats.csv / _pairs.csv")

    # ---- tables from facts.json (+ FACTS_BMD_CTPO.json)
    table2_modules(facts)
    table3_metrics(facts)
    if os.environ.get("IPLDT_TABLE3_PROPOSAL"):
        table3_per_site_proposal(facts)

    # ---- console summary
    log("")
    for key, label, unit in METRICS:
        st = stats[key]
        log(f"{label:6s} n {st['n']}  slope {st['slope']:.6f}  R2 {st['r2']:.7f}  ICC {st['icc21']:.7f}  bias {st['bias']:+.3e}  "
            f"LoA [{st['loa_low']:+.3e}, {st['loa_high']:+.3e}]  max|rel| {st['max_rel_pct']:.4f} %  equal {st['identical']}/{st['n']}")
    log(f"config A: {info_a['total_vox']:,} voxels compared, {info_a['total_mism']} differ; "
        f"{info_a['n_cells'] - info_a['n_nonzero']}/{info_a['n_cells']} scan-map comparisons identical; non-zero {nz}")
    log(f"config B SEG: identical {info_l['n_exact']}/{n}, {info_l['total']:,} differing voxels in {info_l['union']:,}, "
        f"median {info_l['median']:.0f}, min Dice {info_l['dice_min']:.8f}")
    for c, d in per_cohort_l.items():
        log(f"  L {c:10s} {d}")
    if bad:
        log(f"{len(bad)} printed number(s) disagree with facts.json -- see {os.path.relpath(OUT_STEM, REPO)}_numbers.json")


if __name__ == "__main__":
    main()
