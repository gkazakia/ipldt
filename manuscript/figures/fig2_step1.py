"""fig2_step1.py -- Figure 2 of the ipldt / ORMIR-BQRL manuscript: the cortical / trabecular compartment
separation (the reimplementation of IPL Script 32 STEP 1) stage by stage on one patella scan, and the agreement
of the final compartment masks on all 137 scans.

Panels A-H: one mid-slice of a patella scan (XtremeCT II, 60.7 um).  Fill = the voxels IPL V5.42 exported for
that stage of its own evaluation; blue outline = ipldt's result for the same stage, run end to end from the
greyscale and the same periosteal contour with the standard tibia parameters; orange = voxels that differ (none).
Under each panel: the number of differing voxels of the WHOLE stage volume (compared by global voxel position)
and the voxel count of the stage.
Panel I: the final cortical and trabecular masks of all 137 scans -- the size of IPL's masks and, per scan, the
number of voxels in which ipldt's mask differs (0 on every scan).

Run from the repository root in the `ormir` environment (numpy, scipy, numba, matplotlib; no GPU needed):

    python manuscript/figures/fig2_step1.py [--recompute]

Writes manuscript/figures/fig2_step1.png (300 dpi) and fig2_step1.svg, with the sidecars
fig2_step1_numbers.json (every number drawn) and fig2_step1_scans.csv (the 137 rows of panel I).
The slice crops and stage comparisons of panels A-H are cached in manuscript/figures/cache/ after the first run
(about 2 min: 32 AIM exports read, the ipldt chain run once, the two renderings computed); --recompute redoes it.

Inputs (read only):
  * IPL's stage exports of the patella scan and its greyscale AIM (T16_DIR, GRAB below);
  * validation/results/from_ipl_contour_ceil_dt/records/*.json   (21 patellae, configuration B: the dt-inclusive run
    under the shipped engine, the source of every patella configuration-B number of the manuscript);
  * validation/results/oslh_auto_vN/records/<Group>_<Study>_<n>.json (62 radius / tibia scans, configuration B) and
    validation/results/oslh_noedit_vN/records/<Group>_<Study>_<n>.json (the 54 diaphyseal scans of the second set):
    116 radius / tibia scans, the list of manuscript/facts/facts.json (Diaphyseal_CKD_991161 is not part of it);
  * manuscript/facts/facts.json: every pooled and per-site number of panel I is asserted against it.
Nothing under ipldt/ is modified: every ipldt number comes from calling ipldt.step1 / ipldt.contour on IPL's files.
"""
from __future__ import annotations

import argparse
import csv
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
from matplotlib import patches  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
from ipldt.io import read_aim, align_to  # noqa: E402
from ipldt import ipl_ops as ops  # noqa: E402
from ipldt.step1 import STAGES, TIBIA, cort_trab_separation  # noqa: E402
from ipldt.contour import render_volume  # noqa: E402

# ------------------------------------------------------------------------------------------------ inputs
T16_DIR = lab_path("ipl_test_runs/run15/aims_and_logs")   # IPL's stage exports
EXPORT_BASE = "X2420448_T16_"                                                                  # <base>_<TAG>.AIM;n
GREY_AIM = lab_path("patellae/PFJ-0be66a_R/X2420448.AIM")           # the scan's greyscale
PATELLA_RECORDS = os.path.join(REPO, "validation", "results", "from_ipl_contour_ceil_dt", "records")
OSLH_RECORD_DIRS = [os.path.join(REPO, "validation", "results", OSLH_AUTO, "records"),
                    os.path.join(REPO, "validation", "results", OSLH_NOEDIT, "records")]
FACTS_JSON = os.path.join(REPO, "manuscript", "facts", "facts.json")
OSLH_RX = re.compile(r"^[A-Za-z]+_[A-Za-z]+_\d+\.json$")
OSLH_SKIP = {"Diaphyseal_CKD_991161"}          # a record that exists but is not part of the validation set
CACHE = os.path.join(HERE, "cache", "fig2_step1_cache")
OUT = os.path.join(HERE, "fig2_step1")

VOX_MM = 0.0607
Z_MID = 252                                  # global z of the shown slice (the stage-00 volume spans z 168..335)
TAGS32 = STAGES + ("30_cortgobj", "31_trabgobj")

# the eight stages shown, in pipeline order: (tag, letter, title, what the stage does -- two short lines)
PANELS = [
    ("00_all", "A", "Periosteal contour (input)", "rendered from the periosteal\ncontour file"),
    ("01_seggauss", "B", "Dense bone", "Gaussian σ 2, support 3;\n500–3,000 mg HA/cm³"),
    ("02_trab0", "C", "Trabecular candidate", "periosteal region minus\ndense bone"),
    ("15_open15", "D", "Morphological cleaning", "fill, largest component, erode 3,\ndilate 3, close 15, open 15"),
    ("16_corners", "E", "Corner recovery", "opening residue; components of\n≥ 200,000 voxels are kept"),
    ("25_slicewise", "F", "Trabecular region", "close 50, 6-voxel peel,\nslice-wise ≥ 50% rule"),
    ("28_cortfinal", "G", "Cortical mask", "periosteal region minus\ntrabecular region"),
    ("29_trabfinal", "H", "Trabecular mask", "periosteal region minus\ncortical mask"),
]

# panel I: site groups in drawing order (the order of Figures 3, 5 and 6) and their labels; the n is filled in from the data
SITES = [("patella", "patella"), ("UD radius", "ultradistal radius"), ("UD tibia", "ultradistal tibia"),
         ("D radius", "diaphyseal radius"), ("D tibia", "diaphyseal tibia")]
N_SITE = {"patella": 21, "UD radius": 29, "UD tibia": 29, "D radius": 29, "D tibia": 29}     # n = 137

# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito (colour-blind safe, the palette of every figure): blue = ipldt, vermilion = differing voxels,
# greys = IPL's masks, ink for text
C_BLUE, C_ORANGE = "#0072B2", "#D55E00"
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_FILL = "#b9b8b3"           # IPL's export (panels A-H)
C_CORT, C_TRAB = "#52514e", "#b9b8b3"   # IPL's cortical / trabecular mask (panel I)
C_PERI = "#c3c2b7"           # periosteal outline reference
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.5, "axes.edgecolor": C_AXIS, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.color": C_INK2, "ytick.color": C_INK2, "savefig.dpi": 300,
    "figure.dpi": 100, "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": C_INK, "axes.labelcolor": C_INK2,
})

T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def find_export(tag):
    """The newest VMS version (';n') of <EXPORT_BASE><TAG>.AIM in T16_DIR."""
    rx = re.compile("^" + re.escape(EXPORT_BASE + tag) + r"\.AIM(?:;(\d+))?$", re.I)
    cands = [(int(m.group(1) or 0), f) for f in os.listdir(T16_DIR) for m in [rx.match(f)] if m]
    if not cands:
        raise FileNotFoundError(f"no export for {tag} in {T16_DIR}")
    return os.path.join(T16_DIR, max(cands)[1])


def count(v):
    return int(np.count_nonzero(v["data"]))


def mismatch(a, b):
    """Two volumes on any grids, pasted onto their union grid by global position:
    (differing voxels, a-only, b-only, grids identical)."""
    dim, pos = ops.union_grid(a, b)
    A = ops.on_grid(a, dim, pos) != 0
    B = ops.on_grid(b, dim, pos) != 0
    d = A != B
    same = tuple(a["dim"]) == tuple(b["dim"]) and tuple(a["pos"]) == tuple(b["pos"])
    return int(d.sum()), int((A & ~B).sum()), int((B & ~A).sum()), bool(same)


def slice_at(v, z, frame):
    """The set voxels of volume v on global slice z inside the (x0, y0, W, H) window, as a bool (H, W) array."""
    x0, y0, W, H = frame
    return align_to(v, (int(W), int(H), 1), (int(x0), int(y0), int(z)))[0] != 0


def bbox2d(mask, margin, bounds):
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, xs.min() - margin), min(bounds[0], xs.max() + 1 + margin)
    y0, y1 = max(0, ys.min() - margin), min(bounds[1], ys.max() + 1 + margin)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


# ------------------------------------------------------------------------------------------------ compute
def compute_stages():
    """IPL's 32 exports vs ipldt's chain on the same scan: whole-volume comparison per stage + mid-slice crops."""
    say("reading IPL's 32 stage exports and the greyscale")
    E = {tag: read_aim(find_export(tag)) for tag in TAGS32}
    grey = read_aim(GREY_AIM)
    N = dict(z_mid=Z_MID, vox_mm=VOX_MM, export_counts={t: count(E[t]) for t in TAGS32},
             export_grids={t: [[int(x) for x in E[t]["dim"]], [int(x) for x in E[t]["pos"]]] for t in TAGS32})

    say("ipldt: cort_trab_separation(grey, IPL's periosteal rendering, TIBIA, keep_stages=True)")
    t = time.time()
    res = cort_trab_separation(grey, E["00_all"], TIBIA, keep_stages=True)
    N["ipldt_seconds_step1"] = time.time() - t
    N["thresholds"] = res["info"]["thresholds"]
    N["params"] = res["info"]["params"]
    st = dict(res["stages"])
    st["00_all"] = E["00_all"]
    for tag, src in (("30_cortgobj", "28_cortfinal"), ("31_trabgobj", "29_trabfinal")):
        t = time.time()
        st[tag] = ops.mask_vol(render_volume(st[src]["data"] != 0), st[src]["dim"], st[src]["pos"])
        N[f"ipldt_seconds_{tag}"] = time.time() - t

    say("comparing every stage by global position")
    cmp = {}
    for tag in TAGS32:
        if tag == "00_all":
            cmp[tag] = dict(mismatches=None, ours_only=None, ipl_only=None, same_grid=True, voxels_ours=count(E[tag]),
                            voxels_ipl=count(E[tag]), note="input: IPL's periosteal rendering is the chain's input")
            continue
        d, ao, bo, same = mismatch(st[tag], E[tag])
        cmp[tag] = dict(mismatches=d, ours_only=ao, ipl_only=bo, same_grid=same, voxels_ours=count(st[tag]),
                        voxels_ipl=count(E[tag]))
        say(f"  {tag:15s} {d:>6d} differing   ipldt {count(st[tag]):>11,d}   IPL {count(E[tag]):>11,d}   grid same {same}")
    N["stages"] = cmp

    # the window: union x/y extent of all exports, cropped to the stage-00 mid-slice box + 10 voxels
    xs = [v["pos"][0] for v in E.values()] + [v["pos"][0] + v["dim"][0] for v in E.values()]
    ys = [v["pos"][1] for v in E.values()] + [v["pos"][1] + v["dim"][1] for v in E.values()]
    ux0, ux1, uy0, uy1 = min(xs), max(xs), min(ys), max(ys)
    full = (ux0, uy0, ux1 - ux0, uy1 - uy0)
    bx0, by0, bw, bh = bbox2d(slice_at(E["00_all"], Z_MID, full), 10, (full[2], full[3]))
    frame = (ux0 + bx0, uy0 + by0, bw, bh)
    N["frame"] = list(frame)
    ipl = np.stack([slice_at(E[t], Z_MID, frame) for t in TAGS32])
    ours = np.stack([slice_at(st[t], Z_MID, frame) for t in TAGS32])
    N["slice_differing"] = {t: int((ipl[i] != ours[i]).sum()) for i, t in enumerate(TAGS32)}
    return dict(numbers=N, ipl=ipl, ours=ours)


def load_scans():
    """One row per scan (137): site, IPL's final mask sizes and the differing voxels of ipldt's masks."""
    rows = []
    for f in sorted(glob.glob(os.path.join(PATELLA_RECORDS, "*.json"))):
        d = json.load(open(f))
        s, c = d["step1"], d["contours"]
        rows.append(dict(site="patella", key=os.path.basename(f)[:-5], slices=int(d["grey"]["grid"][0][2]),
                         cort_ipl=s["cort"]["voxels_ipl"], trab_ipl=s["trab"]["voxels_ipl"],
                         cort_ours=s["cort"]["voxels_ours"], trab_ours=s["trab"]["voxels_ours"],
                         cort_diff=s["cort"]["mismatches"], trab_diff=s["trab"]["mismatches"],
                         cort_diff_rendered=c["cort"]["mismatches"], trab_diff_rendered=c["trab"]["mismatches"],
                         compared="STEP 1 rasters", step1_seconds=d["timings"]["step1"]))
    oslh = [f for dd in OSLH_RECORD_DIRS for f in glob.glob(os.path.join(dd, "*.json")) if OSLH_RX.match(os.path.basename(f))]
    assert len({os.path.basename(f) for f in oslh}) == len(oslh), "a record occurs in both radius/tibia directories"
    for f in sorted(oslh, key=os.path.basename):
        d = json.load(open(f))
        if d["tag"] in OSLH_SKIP:
            continue
        r = d["B"]["renderings"]
        site = f"{d['meta']['region']} {d['meta']['bone'].lower()}"
        rows.append(dict(site=site, key=d["tag"], slices=int(d["grey"]["grid"][0][2]),
                         cort_ipl=r["cort"]["voxels_ipl"], trab_ipl=r["trab"]["voxels_ipl"],
                         cort_ours=r["cort"]["voxels_ours"], trab_ours=r["trab"]["voxels_ours"],
                         cort_diff=r["cort"]["mismatches"], trab_diff=r["trab"]["mismatches"],
                         cort_diff_rendered=r["cort"]["mismatches"], trab_diff_rendered=r["trab"]["mismatches"],
                         compared="rendered contours", step1_seconds=d["timings"]["step1"]))
    order = {s: i for i, (s, _) in enumerate(SITES)}
    rows.sort(key=lambda r: (order[r["site"]], r["key"]))
    n_site = {s: sum(r["site"] == s for r in rows) for s, _ in SITES}
    assert len(rows) == sum(N_SITE.values()) == 137 and n_site == N_SITE, n_site
    return rows


def check_facts(scans):
    """Every pooled / per-site number panel I draws must equal manuscript/facts/facts.json (the paper's numbers)."""
    f = json.load(open(FACTS_JSON))["facts"]
    pat = [r for r in scans if r["site"] == "patella"]
    osl = [r for r in scans if r["site"] != "patella"]
    want = {"cohort.n_total": len(scans), "cohort.patella.n": len(pat), "cohort.oslh.n": len(osl)}
    for part in ("cort", "trab"):
        want.update({f"B.patella.step1.{part}.n_scans": len(pat),
                     f"B.patella.step1.{part}.exact_scans": sum(r[f"{part}_diff"] == 0 for r in pat),
                     f"B.patella.step1.{part}.mismatches": sum(r[f"{part}_diff"] for r in pat),
                     f"B.patella.step1.{part}.voxels_ipl": sum(r[f"{part}_ipl"] for r in pat),
                     f"B.oslh.rendering.{part}.n_scans": len(osl),
                     f"B.oslh.rendering.{part}.exact_scans": sum(r[f"{part}_diff"] == 0 for r in osl),
                     f"B.oslh.rendering.{part}.mismatches": sum(r[f"{part}_diff"] for r in osl),
                     f"B.oslh.rendering.{part}.voxels_ipl": sum(r[f"{part}_ipl"] for r in osl)})
        for site, key in (("UD radius", "UD_radius"), ("UD tibia", "UD_tibia"), ("D radius", "D_radius"), ("D tibia", "D_tibia")):
            rs = [r for r in osl if r["site"] == site]
            want.update({f"cohort.site.{key}.n": len(rs),
                         f"B.site.{key}.rendering.{part}.n_scans": len(rs),
                         f"B.site.{key}.rendering.{part}.exact_scans": sum(r[f"{part}_diff"] == 0 for r in rs),
                         f"B.site.{key}.rendering.{part}.mismatches": sum(r[f"{part}_diff"] for r in rs),
                         f"B.site.{key}.rendering.{part}.voxels_ipl": sum(r[f"{part}_ipl"] for r in rs)})
    bad = {k: (f[k]["value"], v) for k, v in want.items() if f[k]["value"] != v}
    assert not bad, f"panel I disagrees with facts.json: {bad}"
    say(f"facts check: {len(want)} keys of facts.json reproduced exactly")
    return sorted(want)


def check_text_widths(fig, items):
    """Warn when a text runs past the horizontal span it was given: items = [(Text, x0_mm, x1_mm, label)]."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for t, x0, x1, label in items:
        bb = t.get_window_extent(renderer=r)
        lo, hi = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        if lo < x0 - 0.3 or hi > x1 + 0.3:
            bad += 1
            say(f"  WARNING text overflows its {x1 - x0:.1f} mm span by {max(x0 - lo, hi - x1):.1f} mm: {label!r}")
    say(f"text-width check: {len(items)} texts, {bad} overflow")
    return bad


def save_cache(R):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE + ".npz", ipl=R["ipl"], ours=R["ours"])
    json.dump(R["numbers"], open(CACHE + ".json", "w"), indent=1)


def load_cache():
    z = np.load(CACHE + ".npz")
    return dict(numbers=json.load(open(CACHE + ".json")), ipl=z["ipl"], ours=z["ours"])


# ------------------------------------------------------------------------------------------------ drawing
FIG_W, FIG_H = 180.0, 133.0          # mm


def ax_mm(fig, x, y_top, w, h):
    """Axes at (x, y_top) mm from the top-left corner of the figure, w x h mm."""
    return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H])


def ftext(fig, x, y_top, s, **kw):
    """fig.text at (x, y_top) in mm from the top-left corner."""
    kw.setdefault("va", "top")
    return fig.text(x / FIG_W, 1 - y_top / FIG_H, s, **kw)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def fmt(n):
    """An integer with comma thousands separators, as in the legends and the text."""
    return f"{n:,d}"


def draw(R, scans):
    N, IPL, OURS = R["numbers"], R["ipl"], R["ours"]
    idx = {t: i for i, t in enumerate(TAGS32)}
    H, W = IPL[0].shape
    fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4))
    checks = []

    # ---------------------------------------------------------------- panels A-H: 2 rows x 4 columns
    ncol, x_left, gap, title_h, note_h = 4, 2.0, 3.6, 9.6, 6.6
    pw = (FIG_W - 2 * x_left - (ncol - 1) * gap) / ncol
    ph = pw * H / W
    row_h = title_h + ph + note_h
    y0 = 0.8
    peri = IPL[idx["00_all"]]
    for k, (tag, letter, title, what) in enumerate(PANELS):
        r, c = divmod(k, ncol)
        x = x_left + c * (pw + gap)
        yt = y0 + r * (row_h + 1.0)
        ax = ax_mm(fig, x, yt + title_h, pw, ph)
        ipl, ours = IPL[idx[tag]], OURS[idx[tag]]
        rgb = np.ones(ipl.shape + (3,))
        rgb[ipl] = hex_rgb(C_FILL)
        rgb[ipl != ours] = hex_rgb(C_ORANGE)
        ax.imshow(rgb, interpolation="antialiased", origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        if tag != "00_all":
            ax.contour(peri.astype(float), levels=[0.5], colors=[C_PERI], linewidths=0.35)
            ax.contour(ours.astype(float), levels=[0.5], colors=[C_BLUE], linewidths=0.42)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(C_AXIS)
            s.set_linewidth(0.4)
        # letter, title and the two-line description above the image
        ftext(fig, x, yt, letter, fontsize=9, fontweight="bold", ha="left")
        t1 = ftext(fig, x + 3.4, yt + 0.25, title, fontsize=7.5, fontweight="bold", ha="left")
        t2 = ftext(fig, x, yt + 3.9, what, fontsize=7, color=C_INK2, ha="left", linespacing=1.15)
        checks += [(t1, x, x + pw, title), (t2, x, x + pw, what)]
        # the whole-volume comparison of the stage, below the image
        st = N["stages"][tag]
        yn = yt + title_h + ph + 0.7
        if tag == "00_all":
            n1 = ftext(fig, x + pw / 2, yn, "input", fontsize=7, fontweight="bold", ha="center")
            n2 = ftext(fig, x + pw / 2, yn + 2.9, f"{fmt(st['voxels_ipl'])} voxels", fontsize=7, color=C_INK2, ha="center")
        else:
            n_ipl, n_our, d = st["voxels_ipl"], st["voxels_ours"], st["mismatches"]
            n1 = ftext(fig, x + pw / 2, yn, f"{fmt(d)} voxels differ", fontsize=7, fontweight="bold",
                       color=C_INK if d == 0 else C_ORANGE, ha="center")
            tail = f"{fmt(n_ipl)} voxels in each" if n_ipl == n_our else f"IPL {fmt(n_ipl)}, ipldt {fmt(n_our)}"
            n2 = ftext(fig, x + pw / 2, yn + 2.9, tail, fontsize=7, color=C_INK2, ha="center")
        checks += [(n1, x, x + pw, "note 1 " + tag), (n2, x, x + pw, "note 2 " + tag)]
        if tag == "00_all":                                  # the one scale bar
            n = 5.0 / VOX_MM
            xb, yb = W * 0.035, H * 0.90
            ax.plot([xb, xb + n], [yb, yb], color=C_INK, lw=1.2, solid_capstyle="butt")
            ax.text(xb + n / 2, yb - H * 0.04, "5 mm", ha="center", va="bottom", fontsize=7, color=C_INK)
        if tag == "16_corners":                              # the branch keeps nothing on this scan
            kept = N["stages"]["21_corners2"]["voxels_ipl"]
            ax.text(0.5, 0.45, f"largest component after erosion:\n{fmt(N['largest_17'])} voxels, below 200,000\n"
                               f"→ {kept} voxels added",
                    transform=ax.transAxes, ha="center", va="center", fontsize=7, color=C_INK2, linespacing=1.25,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        if c < ncol - 1:                                     # flow arrow to the next panel
            xa0, xa1 = (x + pw + 0.5) / FIG_W, (x + pw + gap - 0.5) / FIG_W
            ya = 1 - (yt + title_h + ph / 2) / FIG_H
            fig.add_artist(patches.FancyArrowPatch((xa0, ya), (xa1, ya), transform=fig.transFigure, arrowstyle="-|>",
                                                   mutation_scale=7, lw=0.7, color=C_INK2))
    # legend of panels A-H
    y_leg = y0 + 2 * row_h + 1.0 + 0.2
    handles = [patches.Patch(fc=C_FILL, ec="none", label="IPL's result for the stage (fill)"),
               Line2D([], [], color=C_BLUE, lw=1.0, label="ipldt's result for the stage (outline)"),
               patches.Patch(fc=C_ORANGE, ec="none", label="voxels that differ (none in any panel)"),
               Line2D([], [], color=C_PERI, lw=0.8, label="periosteal contour")]
    axl = ax_mm(fig, x_left, y_leg, FIG_W - 2 * x_left, 3.0)
    axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=4, frameon=False, fontsize=7, handlelength=1.6,
               columnspacing=1.6, handletextpad=0.6, borderaxespad=0)

    # ---------------------------------------------------------------- panel I: the 137 scans
    yI = y_leg + 4.6
    xl, xr = 15.0, FIG_W - 2.0
    n = len(scans)
    ftext(fig, x_left, yI, "I", fontsize=9, fontweight="bold", ha="left")
    ftext(fig, x_left + 3.4, yI + 0.25, f"Final compartment masks, all {n} scans", fontsize=7.5, fontweight="bold", ha="left")
    axh = ax_mm(fig, xr - 60.0, yI - 0.2, 60.0, 3.4)
    axh.axis("off")
    axh.legend(handles=[patches.Patch(fc=C_CORT, label="IPL cortical mask"), patches.Patch(fc=C_TRAB, label="IPL trabecular mask")],
               loc="center right", ncol=2, frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.2,
               handletextpad=0.5, borderaxespad=0)
    lane_top, lane_h = yI + 4.6, 7.0                       # the site labels and their brackets, above the bars
    bar_top, bar_h = lane_top + lane_h, 24.0
    axb = ax_mm(fig, xl, bar_top, xr - xl, bar_h)
    xs = np.arange(n)
    cort = np.array([r["cort_ipl"] for r in scans]) / 1e6
    trab = np.array([r["trab_ipl"] for r in scans]) / 1e6
    axb.bar(xs, cort, width=0.8, color=C_CORT, lw=0)
    axb.bar(xs, trab, bottom=cort + 0.3, width=0.8, color=C_TRAB, lw=0)      # 0.3 M gap = the surface gap
    axb.set_xlim(-0.7, n - 0.3)
    ymax = 60.0
    assert (cort + trab).max() < ymax, (cort + trab).max()
    axb.set_ylim(0, ymax)
    axb.set_yticks(np.arange(0, 61, 10))
    axb.set_ylabel("mask voxels (millions)", fontsize=7, labelpad=3)
    axb.set_xticks([])
    axb.tick_params(axis="y", length=2, pad=1.5, labelsize=7)
    axb.grid(True, axis="y", color=C_GRID, lw=0.4)
    axb.set_axisbelow(True)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.spines["bottom"].set_color(C_AXIS)
    axb.spines["left"].set_color(C_AXIS)
    # site groups: separators + labels along the top of the bar axes
    bounds, i0 = [], 0
    for s, lab in SITES:
        k = sum(r["site"] == s for r in scans)
        bounds.append((s, f"{lab}\n(n = {k})", i0, i0 + k - 1))
        i0 += k
    axg = ax_mm(fig, xl, lane_top, xr - xl, lane_h)          # the label lane shares the bar axes' x scale
    axg.set_xlim(-0.7, n - 0.3)
    axg.set_ylim(0, 1)
    axg.axis("off")
    for j, (s, lab, a, b) in enumerate(bounds):
        if j:
            axb.axvline(a - 0.5, color=C_AXIS, lw=0.5, ls=(0, (2, 2)))
        axg.plot([a - 0.35, a - 0.35, b + 0.35, b + 0.35], [0.22, 0.08, 0.08, 0.22], color=C_AXIS, lw=0.6)
        tg = axg.text((a + b) / 2, 0.3, lab, ha="center", va="bottom", fontsize=7, color=C_INK2, linespacing=1.1)
        checks.append((tg, xl, xr, "site label " + s))
    # the strip: differing voxels per scan, two rows, with its header in the gap under the bars
    strip_top, cell_h = bar_top + bar_h + 4.4, 3.2
    n_exact_c = sum(r["cort_diff"] == 0 for r in scans)
    n_exact_t = sum(r["trab_diff"] == 0 for r in scans)
    hd = ftext(fig, xl, strip_top - 0.9, "Differing voxels per scan, ipldt vs IPL (white cell = 0)", fontsize=7,
               fontweight="bold", ha="left", va="bottom")
    ftext(fig, xr, strip_top - 0.9, f"0 in {n_exact_c} of {n} cortical and {n_exact_t} of {n} trabecular masks",
          fontsize=7, color=C_INK2, ha="right", va="bottom")
    axs = ax_mm(fig, xl, strip_top, xr - xl, 2 * cell_h)
    axs.set_xlim(-0.7, n - 0.3)
    axs.set_ylim(0, 2)
    axs.axis("off")
    # 137 cells of 1.2 mm are too narrow for a 7-pt numeral: a cell is white for 0 and vermilion otherwise, with its
    # count printed above the strip (the convention of Figure 5F)
    for row, key in ((1, "cort_diff"), (0, "trab_diff")):
        for i, r in enumerate(scans):
            d = r[key]
            axs.add_patch(patches.Rectangle((i - 0.5, row + 0.06), 1.0, 0.88, fc="white" if d == 0 else C_ORANGE,
                                            ec=C_AXIS, lw=0.3))
            if d:
                axs.text(i, 2.05, f"{d}", ha="center", va="bottom", fontsize=7, color=C_ORANGE)
    for row, lab in ((1, "cortical"), (0, "trabecular")):
        ftext(fig, xl - 0.9, strip_top + (2 - row - 0.5) * cell_h, lab, fontsize=7, color=C_INK2, ha="right", va="center")
    # the pooled totals and the comparison rule under the strip
    tot_c, tot_t = sum(r["cort_ipl"] for r in scans), sum(r["trab_ipl"] for r in scans)
    y_foot = strip_top + 2 * cell_h + 1.0
    f1 = ftext(fig, xl, y_foot, f"Compared per scan by global voxel position; IPL's masks hold {fmt(tot_c)} cortical "
                                f"and {fmt(tot_t)} trabecular voxels in total.", fontsize=7, color=C_INK2, ha="left")
    f2 = ftext(fig, xl, y_foot + 2.9, "Patellae: the STEP 1 rasters; radius and tibia: the STEP 1 masks as rendered "
                                      "from their contour files.", fontsize=7, color=C_INK2, ha="left")
    checks += [(f1, xl, xr, "footnote 1"), (f2, xl, xr, "footnote 2"), (hd, xl, xr, "strip header")]
    check_text_widths(fig, checks)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true", help="ignore the cache of the stage comparisons and crops")
    a = ap.parse_args()
    if not a.recompute and os.path.exists(CACHE + ".json") and os.path.exists(CACHE + ".npz"):
        say("loading cache", CACHE)
        R = load_cache()
    else:
        R = compute_stages()
        save_cache(R)
    N = R["numbers"]
    # the largest component of the eroded corner candidates (stage 17) on this scan: from IPL's export
    if "largest_17" not in N:
        v17 = read_aim(find_export("17_cornero"))
        sizes = ops.component_sizes(v17)
        N["largest_17"] = int(sizes[0]) if sizes.size else 0
        N["n_components_17"] = int(sizes.size)
        json.dump(N, open(CACHE + ".json", "w"), indent=1)
    scans = load_scans()
    facts_keys = check_facts(scans)
    say(f"{len(scans)} scans; differing voxels cortical {sum(r['cort_diff'] for r in scans)}, "
        f"trabecular {sum(r['trab_diff'] for r in scans)}")
    fig = draw(R, scans)
    fig.savefig(OUT + ".png", dpi=300, facecolor="white")
    fig.savefig(OUT + ".svg", facecolor="white")
    plt.close(fig)
    say("wrote", OUT + ".png", "and .svg")
    # sidecars
    out = dict(figure="fig2_step1", scan_shown="one patella scan, mid-slice", z_mid=N["z_mid"], frame=N["frame"],
               vox_mm=VOX_MM, panels=[dict(tag=t, letter=L, title=ti, what=w, **N["stages"][t],
                                           slice_differing=N["slice_differing"][t]) for t, L, ti, w in PANELS],
               all_stages=N["stages"], slice_differing=N["slice_differing"], thresholds=N["thresholds"],
               params=N["params"], largest_component_stage17=N["largest_17"], n_components_stage17=N["n_components_17"],
               ipldt_seconds_step1_this_scan=N["ipldt_seconds_step1"],
               scans=dict(n=len(scans), n_exact_cort=sum(r["cort_diff"] == 0 for r in scans),
                          n_exact_trab=sum(r["trab_diff"] == 0 for r in scans),
                          differing_cort_total=sum(r["cort_diff"] for r in scans),
                          differing_trab_total=sum(r["trab_diff"] for r in scans),
                          ipl_cort_voxels_total=sum(r["cort_ipl"] for r in scans),
                          ipl_trab_voxels_total=sum(r["trab_ipl"] for r in scans),
                          per_site={s: dict(n=sum(r["site"] == s for r in scans),
                                            exact_cort=sum(r["site"] == s and r["cort_diff"] == 0 for r in scans),
                                            exact_trab=sum(r["site"] == s and r["trab_diff"] == 0 for r in scans))
                                    for s, _ in SITES},
                          step1_seconds_median=float(np.median([r["step1_seconds"] for r in scans]))),
               facts_checked=facts_keys,
               sources=dict(exports=public_path(T16_DIR), greyscale=public_path(GREY_AIM), patella_records=public_path(PATELLA_RECORDS),
                            oslh_records=public_path(OSLH_RECORD_DIRS), facts=public_path(FACTS_JSON)))
    json.dump(out, open(OUT + "_numbers.json", "w"), indent=1)
    with open(OUT + "_scans.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scans[0].keys()))
        w.writeheader()
        w.writerows(scans)
    say("sidecars", OUT + "_numbers.json", OUT + "_scans.csv")


if __name__ == "__main__":
    main()
