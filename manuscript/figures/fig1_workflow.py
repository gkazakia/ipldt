"""fig1_workflow.py -- Figure 1 of the ipldt / ORMIR-BQRL manuscript: the workflow, stage by stage.

Two rows of four stage boxes (AIM in -> periosteal contour -> compartment separation -> contour rendering ->
Laplace-Hamming segmentation -> distance-transform maps -> morphometry and BMD -> outputs), each with the IPL
command group it reimplements in small type, a one-line agreement statement taken from the facts sheet, and a
real mid-stack slice of one patellar scan showing the product of that stage.  The border colour
marks the kind of stage: reimplemented and exact on every scan, reimplemented with differing voxels on some scans,
an ORMIR-XCT component, or input / output.
The class of each IPL-reimplementing stage is computed from the facts sheet (stage_class), never set by hand:
green only when every comparison its italic line reports is exact on all n scans.

Run from the repository root in the `ormir` environment:

    set PYTHONUTF8=1
    python manuscript/figures/fig1_workflow.py [--no-cache]

Writes manuscript/figures/fig1_workflow.png (300 dpi, 180 mm wide) and fig1_workflow.svg, the sidecar
fig1_workflow_numbers.json (every agreement statement printed, with the facts values behind it), and caches the
2-D slice crops in manuscript/figures/cache/fig1_workflow_slices.npz (deleted or bypassed with --no-cache).

Inputs (read only):
  * the patellar scan's greyscale AIM, IPL's cortical / trabecular compartment masks, SEG and Tb.Th map
    (DATA_DIR; the same folder the guide figures use);
  * the ORMIR-XCT periosteal mask of the same scan written by the ORMIR-BQRL run (PRX_NIFTI; SimpleITK), with
    IPL's periosteal raster as the fallback when SimpleITK is unavailable;
  * manuscript/facts/facts.json and manuscript/facts/FACTS_BMD_CTPO.json (Ct.Po, pore map) for every number
    printed on the figure and for the border colour of stages 3-7.
The compartment contours are rendered here with the package's own renderer (ipldt.contour.render_slice) on the
shown slice.  Nothing under ipldt/ is modified.  Deterministic; no GPU needed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors, patches  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
from ipldt.io import read_aim  # noqa: E402
from ipldt.contour import render_slice  # noqa: E402

DATA_DIR = lab_path("PFJOA/XCT_masks_full_grab/PFJ-0be66a_R")
BASE = "X2420448"
PRX_NIFTI = os.environ.get("IPLDT_FIG1_PRX_MASK") or lab_path(f"ormir_run_PFJ-0be66a/{BASE}_PRX_MASK.nii.gz")
FACTS = os.path.join(REPO, "manuscript", "facts", "facts.json")
FACTS_PORE = os.path.join(REPO, "manuscript", "facts", "FACTS_BMD_CTPO.json")
CACHE = os.path.join(HERE, "cache", "fig1_workflow_slices.npz")
OUT_PNG = os.path.join(HERE, "fig1_workflow.png")
OUT_SVG = os.path.join(HERE, "fig1_workflow.svg")
OUT_NUM = os.path.join(HERE, "fig1_workflow_numbers.json")

Z_LOCAL = 84                 # mid-stack slice of the 168-slice scan
VOX_MM = 0.0607              # nominal XtremeCT II voxel size
FRAME_MARGIN = 8             # voxels around the periosteal bounding box of the shown slice
GREY_WINDOW = (-400, 8200)   # native int16 display window

# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito colour-blind-safe palette; one hue per category, assigned in fixed order (never cycled).
GREEN, ORANGE, BLUE, SKY, PURPLE, VERMILION = "#009E73", "#E69F00", "#0072B2", "#56B4E9", "#CC79A7", "#D55E00"
INK, INK2, MUTED, RULE = "#111111", "#333333", "#5A5A5A", "#BDBDBD"
CAT = {
    "identical": dict(ec=GREEN, fc="#E4F3EE", ls="-", label="output identical to IPL's on every scan, from IPL's input (0 differing voxels)"),
    "differs": dict(ec=ORANGE, fc="#FBF1DC", ls="-", label="output differs from IPL's on some scans (differences given in the box)"),
    "ormir": dict(ec=BLUE, fc="#E1ECF5", ls="-", label="ORMIR-XCT component"),
    "io": dict(ec="#7F7F7F", fc="#FFFFFF", ls=(0, (3.0, 1.8)), label="input / output stage, no IPL counterpart"),
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.linewidth": 0.5, "figure.dpi": 300, "savefig.dpi": 300,
    "svg.fonttype": "path", "pdf.fonttype": 42,
})
FS_TITLE, FS_BODY = 8.0, 7.0          # points; nothing on the figure is smaller than 7 pt
LINE = 1.22                            # line spacing (em)

# ------------------------------------------------------------------------------------------------ geometry (mm, y down)
W_MM = 180.0
MARGIN = 2.0
GAP = 4.6
PAD = 1.7
NCOL = 4
BW = (W_MM - 2 * MARGIN - (NCOL - 1) * GAP) / NCOL
CORRIDOR = 7.0                         # between the two rows (the turn of the flow)
CAPTION_SLOT = 5.6                     # under each thumbnail: caption or colour bar
LEGEND_H = 8.0


def pt2mm(pt):
    return pt / 72.0 * 25.4


def lh(fs):
    return pt2mm(fs) * LINE


# ------------------------------------------------------------------------------------------------ facts
def load_facts():
    F = json.load(open(FACTS, encoding="utf-8"))["facts"]      # flat: {'A.pooled.map...': {'value': .., 'source': ..}}
    P = json.load(open(FACTS_PORE, encoding="utf-8"))["facts"]  # same layout; the pore map and Ct.Po

    def fact(key):
        return F[key]["value"]

    def pfact(key):
        return P[key]["value"]

    n = fact("cohort.n_total")
    step1_exact = fact("B.patella.step1.cort.exact_scans") + fact("B.oslh.rendering.cort.exact_scans")
    step1_exact_t = fact("B.patella.step1.trab.exact_scans") + fact("B.oslh.rendering.trab.exact_scans")
    rend_exact = fact("B.patella.contour.cort.exact_scans") + fact("B.oslh.rendering.cort.exact_scans")
    rend_exact_t = fact("B.patella.contour.trab.exact_scans") + fact("B.oslh.rendering.trab.exact_scans")
    seg_ppm = fact("B.pooled.SEG.mismatch_ppm")
    seg_median = fact("B.pooled.SEG.per_scan.median")
    seg_dice_min = fact("B.pooled.SEG.dice.min")
    dt_exact = {m: fact(f"A.pooled.map.{m}.exact_scans") for m in ("TbSp", "TbN", "CtTh", "TbTh_trabseg")}
    dt_mism = fact("A.pooled.mismatches.four_maps")
    dt_vox = fact("A.pooled.voxels_compared.four_maps")
    met_exact_A = {m: fact(f"metrics.A.pooled.{m}.exact") for m in ("TbSp", "TbN", "CtTh")}
    MET5 = ("BVTV", "TbTh_trabseg", "TbSp", "TbN", "CtTh")
    met_maxrel_B = max(fact(f"metrics.B.pooled.{m}.max_rel_pct") for m in MET5)
    ctpo_maxrel_B = pfact("pore.B.max_abs_rel_pct")
    ctpo_bound_B = np.ceil(ctpo_maxrel_B * 100 - 1e-9) / 100     # an upper bound at two decimals
    ormir_ver = fact("software.ormir_xct.version")
    ipl_ver = fact("cohort.ipl_version")
    assert step1_exact == step1_exact_t == rend_exact == rend_exact_t == n, "compartment / contour exactness must be pooled n"
    assert dt_exact["TbSp"] == dt_exact["TbN"] == dt_exact["CtTh"] == n
    # the border class of stages 3-7: the number of scans on which every comparison the stage's italic line reports
    # is exact (0 differing voxels; scalars equal to 12 significant digits).  Stages 3-5 and 7 report configuration B
    # (end to end), stage 6 configuration A (maps from IPL's segmentation).  Green only when that number is n.
    exact_scans = {
        3: min(step1_exact, step1_exact_t),
        4: min(rend_exact, rend_exact_t),
        5: fact("B.pooled.SEG.exact_scans"),
        6: min(list(dt_exact.values()) + [pfact("pore.A.identical")]),     # the four maps and the pore map, configuration A
        7: min([fact(f"metrics.B.pooled.{m}.exact") for m in MET5]
               + [pfact("pore.B.identical"), pfact("pore.B.identical_ctpo_12sf")]),
    }
    stage_class = {k: ("identical" if v == n else "differs") for k, v in exact_scans.items()}
    print("  stage class from the facts:", ", ".join(f"{k}: {v} ({exact_scans[k]}/{n} scans exact)" for k, v in stage_class.items()))
    S = dict(
        n=n, ipl_ver=ipl_ver, ormir_ver=ormir_ver,
        facts_values=dict(step1_exact=step1_exact, render_exact=rend_exact, seg_mismatch_ppm=seg_ppm, seg_dice_min=seg_dice_min,
                          dt_exact=dt_exact, dt_mismatches_four_maps=dt_mism, dt_voxels_compared_four_maps=dt_vox,
                          metrics_A_exact=met_exact_A, metrics_B_max_rel_pct_of_five=met_maxrel_B,
                          ctpo_B_max_abs_rel_pct=ctpo_maxrel_B, stage_exact_scans=exact_scans),
        stage_class=stage_class,
        step1=f"identical on {step1_exact}/{n} scans",
        render=f"identical on {rend_exact}/{n} scans",
        seg=f"{seg_ppm:.1f} voxels per million differ; Dice \u2265 {seg_dice_min:.3f}",
        seg_median=seg_median,
        dt=(f"identical on {n}/{n} scans: all four maps ({dt_mism} of {dt_vox / 1e9:.1f} × 10⁹ voxels differ) and the pore map"
            if min(dt_exact.values()) == n and pfact("pore.A.identical") == n else
            f"identical on {dt_exact['TbSp']}/{n} scans (Tb.Sp, 1/Tb.N, Ct.Th); Tb.Th {dt_exact['TbTh_trabseg']}/{n}, "
            f"{dt_mism} voxel of {dt_vox / 1e9:.1f} \u00d7 10\u2079"),
        metrics=(f"BV/TV, Tb.Th, Tb.Sp, Tb.N, Ct.Th within {met_maxrel_B:.3f}%; "
                 f"Ct.Po within {ctpo_bound_B:.2f}%"),
    )
    for k, v in S.items():
        print(f"  fact -> {k}: {v}")
    return S


# ------------------------------------------------------------------------------------------------ data
def slice_on_grey(v, grey, z):
    """The z-slice of volume `v` pasted onto the greyscale (y, x) grid by global position."""
    out = np.zeros((grey["dim"][1], grey["dim"][0]), v["data"].dtype)
    dx, dy, dz = (int(v["pos"][i] - grey["pos"][i]) for i in range(3))
    zz = z - dz
    if not 0 <= zz < v["dim"][2]:
        raise ValueError("slice outside the volume")
    sl = v["data"][zz]
    x0, y0 = max(dx, 0), max(dy, 0)
    x1, y1 = min(dx + v["dim"][0], out.shape[1]), min(dy + v["dim"][1], out.shape[0])
    out[y0:y1, x0:x1] = sl[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def read_prx(grey):
    """The ORMIR-XCT periosteal mask on the greyscale grid (z, y, x); None if it cannot be read."""
    try:
        import SimpleITK as sitk
    except ImportError:
        print("  SimpleITK not available: falling back to IPL's periosteal raster for the contour thumbnail")
        return None
    im = sitk.ReadImage(PRX_NIFTI)
    a = sitk.GetArrayFromImage(im)
    org, sp = np.array(im.GetOrigin()), np.array(im.GetSpacing())
    pos = np.rint(org / sp).astype(int)
    if tuple(pos) != tuple(int(p) for p in grey["pos"]) or a.shape != grey["data"].shape:
        raise RuntimeError(f"PRX grid {a.shape} @ {pos} differs from the greyscale grid {grey['data'].shape} @ {grey['pos']}")
    return a != 0


def load_slices():
    print("reading the scan and IPL's products")
    grey = read_aim(os.path.join(DATA_DIR, f"{BASE}.AIM"))
    z = Z_LOCAL
    el = float(grey["el_size_mm"][0])
    g2 = grey["data"][z].astype(np.float32)
    cort = slice_on_grey(read_aim(os.path.join(DATA_DIR, f"{BASE}_CORT_MASK.AIM")), grey, z) != 0
    trab = slice_on_grey(read_aim(os.path.join(DATA_DIR, f"{BASE}_TRAB_MASK.AIM")), grey, z) != 0
    seg = slice_on_grey(read_aim(os.path.join(DATA_DIR, f"{BASE}_SEG.AIM")), grey, z)
    tbth = slice_on_grey(read_aim(os.path.join(DATA_DIR, f"{BASE}_TRAB_TH_old_decompressed.AIM")), grey, z).astype(np.int16)
    prx3 = read_prx(grey)
    prx = prx3[z] if prx3 is not None else (cort | trab)
    prx_is_ormir = prx3 is not None
    print("  rendering the compartment contours of the shown slice (ipldt.contour.render_slice)")
    g_cort = render_slice(cort)
    g_trab = render_slice(trab)
    ys, xs = np.nonzero(prx | cort | trab)
    x0, x1 = max(0, xs.min() - FRAME_MARGIN), min(g2.shape[1], xs.max() + 1 + FRAME_MARGIN)
    y0, y1 = max(0, ys.min() - FRAME_MARGIN), min(g2.shape[0], ys.max() + 1 + FRAME_MARGIN)
    crop = (slice(y0, y1), slice(x0, x1))
    D = dict(grey=g2[crop], prx=prx[crop], cort=cort[crop], trab=trab[crop], g_cort=g_cort[crop], g_trab=g_trab[crop],
             seg=seg[crop], tbth=tbth[crop], el=np.float64(el), prx_is_ormir=np.bool_(prx_is_ormir),
             frame=np.array([x0, y0, x1 - x0, y1 - y0]), z_global=np.int64(grey["pos"][2] + z))
    print(f"  slice z = {D['z_global']} (global), window {x1 - x0} x {y1 - y0} voxels, voxel {el:.5f} mm, "
          f"periosteal contour from {'ORMIR-XCT' if prx_is_ormir else 'IPL'}")
    return D


def get_slices(use_cache=True):
    if use_cache and os.path.exists(CACHE):
        z = np.load(CACHE)
        D = {k: z[k] for k in z.files}
        print(f"slice crops read from {CACHE}")
        return D
    D = load_slices()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **D)
    return D


# ------------------------------------------------------------------------------------------------ the boxes
def stage_specs(S):
    """Title, body, IPL command group, status line, category, thumbnail kind -- in pipeline order."""
    return [
        dict(num=1, title="Scan input", cat="io", thumb="grey",
             body="XtremeCT II scan: native int16 grayscale with its calibration log.",
             ipl="AIM files, including IPL's compressed products", status=None, caption="native grayscale"),
        dict(num=2, title="Periosteal contour", cat="ormir", thumb="prx",
             body="ORMIR-XCT autocontour of the outer bone surface (dual-threshold method).",
             ipl="in place of the Scanco autocontour script",
             status=f"ORMIR-XCT {S['ormir_ver']} component; not an IPL reimplementation", caption="periosteal contour"),
        dict(num=3, title="Compartment separation", cat=S["stage_class"][3], thumb="masks",
             body="Cortical / trabecular masks: Gaussian segmentation, chamfer-metric morphology, component and slice-wise rules; site presets.",
             ipl="/seg_gauss, /erosion, /dilation, /open, /close, /cl_*_extract, /gobj_maskaimpeel_ow (STEP 1)",
             status=S["step1"], caption="cortical (dark), trabecular (light)"),
        dict(num=4, title="Contour rendering", cat=S["stage_class"][4], thumb="contours",
             body="Each compartment mask traced, curvature-smoothed and rasterized as IPL's contour object.",
             ipl="/togobj_from_aim -curvature_smooth 1; /gobj_to_aim",
             status=S["render"], caption="cortical (black), trabecular (green)"),
        dict(num=5, title="Segmentation", cat=S["stage_class"][5], thumb="seg",
             body="Laplace\u2013Hamming FFT filter, normalization, 475\u2030 threshold, component filters; labels 127 (cortical), 126 (trabecular).",
             ipl="/fft_laplace_hamming, /norm_max, /threshold, /cl_nr_extract (STEP 2)",
             status=S["seg"], caption="cortical (dark), trabecular (gray)"),
        dict(num=6, title="Distance-transform and pore maps", cat=S["stage_class"][6], thumb="tbth",
             body="Vector distance transform, ridge and sphere fitting: Tb.Th, Tb.Sp, 1/Tb.N and Ct.Th maps; cortical pore cascade; GPU and CPU back ends agree.",
             ipl="/dt_thickness, /dt_spacing, /dt_number (STEP 3); pore cascade (STEP 2)",
             status=S["dt"], caption="Tb.Th (mm)"),
        dict(num=7, title="Morphometry and BMD", cat=S["stage_class"][7], thumb="report",
             body="BV/TV from the segmentation; Tb.Th, Tb.Sp, Tb.N, Ct.Th from the maps; Ct.Po from the pore map; inside the compartment contours.",
             pill="Tb.BMD, Ct.BMD: ORMIR-XCT",
             ipl="map statistics of STEP 3",
             status=S["metrics"], caption=None),
        dict(num=8, title="Outputs", cat="io", thumb="labelmap",
             body="Report (JSON / CSV); masks and maps as NIfTI or AIM; 3D Slicer segmentation and labelmap; re-entry after manual correction of a contour.",
             ipl="re-entry as in Script 34", status=None, caption="labelmap for 3D Slicer"),
    ]


class Page:
    """A figure whose canvas coordinates are millimetres with y pointing down."""

    def __init__(self, w_mm, h_mm):
        self.W, self.H = w_mm, h_mm
        self.fig = plt.figure(figsize=(w_mm / 25.4, h_mm / 25.4))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w_mm)
        self.ax.set_ylim(h_mm, 0)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.renderer = self.fig.canvas.get_renderer()

    def width_mm(self, s, fs, weight="normal", style="normal"):
        t = self.ax.text(0, 0, s, fontsize=fs, fontweight=weight, fontstyle=style)
        bb = t.get_window_extent(self.renderer)
        t.remove()
        return bb.width / self.fig.dpi * 25.4

    def wrap(self, s, width, fs, weight="normal", style="normal", first_width=None):
        """Greedy word wrap by measured width; the first line may be narrower (a tag in front of it).
        Splits on plain spaces only, so a no-break space (U+00A0) keeps a number and its unit on one line."""
        words, lines, cur = s.split(" "), [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            limit = first_width if (first_width is not None and not lines) else width
            if cur and self.width_mm(cand, fs, weight, style) > limit:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            lines.append(cur)
        return lines

    def text(self, x, y, s, fs=FS_BODY, color=INK, **kw):
        kw.setdefault("ha", "left")
        kw.setdefault("va", "top")
        return self.ax.text(x, y, s, fontsize=fs, color=color, zorder=6, **kw)

    def block(self, x, y, s, width, fs=FS_BODY, color=INK, weight="normal", style="normal", draw=True, ha="left"):
        """Wrapped text from (x, y) at the top; returns the y below it."""
        for ln in self.wrap(s, width, fs, weight, style):
            if draw:
                self.text(x, y, ln, fs=fs, color=color, fontweight=weight, fontstyle=style, ha=ha)
            y += lh(fs)
        return y

    def tagged(self, x, y, tag, s, width, fs=FS_BODY, color=MUTED, draw=True):
        """'TAG  text', the tag bold, the text wrapped around it; returns the y below it."""
        tw = self.width_mm(tag, fs, "bold") + 1.3
        lines = self.wrap(s, width, fs, first_width=width - tw)
        for i, ln in enumerate(lines):
            if draw:
                if i == 0:
                    self.text(x, y, tag, fs=fs, color=color, fontweight="bold")
                self.text(x + (tw if i == 0 else 0), y, ln, fs=fs, color=color)
            y += lh(fs)
        return y

    def axes_mm(self, x, y, w, h):
        """A thumbnail axes at (x, y, w, h) in page millimetres (y from the top)."""
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H])

    def arrow(self, p0, p1, color=INK2, lw=0.9):
        self.ax.add_patch(patches.FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=lw, color=color,
                                                  shrinkA=0, shrinkB=0, zorder=5))

    def polyline(self, pts, color=INK2, lw=0.9):
        xs, ys = zip(*pts)
        self.ax.add_line(Line2D(xs, ys, color=color, lw=lw, zorder=5, solid_capstyle="round", solid_joinstyle="round"))
        self.arrow(pts[-2], pts[-1], color=color, lw=lw)


def content_height(page, spec, width):
    """Height (mm) of the text part of a box, measured without drawing."""
    y = 0.0
    y = page.block(0, y, f"{spec['num']}  {spec['title']}", width, FS_TITLE, weight="bold", draw=False) + 0.5
    y = page.block(0, y, spec["body"], width, FS_BODY, draw=False)
    if spec.get("pill"):
        y += 0.6 + lh(FS_BODY) + 1.0
    y += 0.5
    y = page.tagged(0, y, "IPL", spec["ipl"], width, draw=False)
    if spec.get("status"):
        y += 0.5
        y = page.block(0, y, spec["status"], width - 2.6, FS_BODY, style="italic", draw=False)
    return y


def draw_text_part(page, spec, x, y, width):
    st = CAT[spec["cat"]]
    y = page.block(x, y, f"{spec['num']}  {spec['title']}", width, FS_TITLE, weight="bold", draw=True) + 0.5
    y = page.block(x, y, spec["body"], width, FS_BODY)
    if spec.get("pill"):
        y += 0.6
        ph = lh(FS_BODY) + 1.0
        pw = page.width_mm(spec["pill"], FS_BODY) + 3.0
        page.ax.add_patch(patches.FancyBboxPatch((x + 0.2, y), pw, ph, boxstyle="round,pad=0,rounding_size=0.9",
                                                 fc="#FFFFFF", ec=CAT["ormir"]["ec"], lw=0.8, zorder=4))
        page.text(x + 0.2 + pw / 2, y + ph / 2, spec["pill"], fs=FS_BODY, color=INK, ha="center", va="center")
        y += ph
    y += 0.5
    y = page.tagged(x, y, "IPL", spec["ipl"], width)      # the IPL command group, in a muted ink
    if spec.get("status"):
        y += 0.5
        r = 0.75
        page.ax.add_patch(patches.Circle((x + r, y + pt2mm(FS_BODY) * 0.55), r, fc=st["ec"], ec="none", zorder=6))
        y = page.block(x + 2.6, y, spec["status"], width - 2.6, FS_BODY, style="italic")
    return y


# ------------------------------------------------------------------------------------------------ thumbnails
def thumb_image(kind, D):
    """(rgb image, overlays) for a thumbnail kind; overlays = list of (mask, colour, lw) outlines."""
    g = D["grey"]
    lo, hi = GREY_WINDOW
    gn = np.clip((g - lo) / (hi - lo), 0, 1)
    H, W = g.shape
    white = np.ones((H, W, 3))
    if kind == "grey":
        return np.repeat(gn[..., None], 3, -1), []
    if kind == "prx":
        return np.repeat(gn[..., None], 3, -1), [(D["prx"], BLUE, 0.9)]
    if kind == "masks":
        rgb = white.copy()
        rgb[D["trab"]] = 0.80
        rgb[D["cort"]] = 0.38
        return rgb, []
    if kind == "contours":
        rgb = white.copy()
        return rgb, [(D["g_cort"], INK, 0.7), (D["g_trab"], GREEN, 0.7)]
    if kind == "seg":
        rgb = white.copy()
        rgb[D["seg"] == 126] = 0.55
        rgb[D["seg"] == 127] = 0.12
        return rgb, []
    if kind == "labelmap":
        base = 0.35 + 0.65 * gn                       # lifted greyscale, as a viewer shows it under segments
        rgb = np.repeat(base[..., None], 3, -1)
        for m, col in ((D["trab"], SKY), (D["cort"], PURPLE)):
            c = np.array(colors.to_rgb(col))
            rgb[m] = 0.45 * rgb[m] + 0.55 * c
        return rgb, [(D["prx"], INK, 0.7)]
    raise KeyError(kind)


def tbth_cmap(vmax):
    base = plt.get_cmap("viridis", vmax)
    cols = np.vstack([[1, 1, 1, 1], base(np.arange(vmax))])
    cm = colors.ListedColormap(cols)
    norm = colors.BoundaryNorm(np.arange(-0.5, vmax + 1.5, 1), cm.N)
    return cm, norm


def draw_thumbnail(page, spec, D, x, y, w, h, caption_y):
    """The thumbnail of one stage in the (x, y, w, h) slot plus its caption / colour bar in the caption slot."""
    kind = spec["thumb"]
    frame_col = RULE
    if kind == "report":
        h = h + CAPTION_SLOT - 1.2                  # no caption under the report: the table takes the caption slot
        ax = page.axes_mm(x, y, w, h)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#F7F7F7")
        for s in ax.spines.values():
            s.set_edgecolor(frame_col)
            s.set_linewidth(0.5)
        rows = [("BV/TV", "\u2013"), ("Tb.Th", "mm"), ("Tb.Sp", "mm"), ("Tb.N", "1/mm"), ("Ct.Th", "mm"),
                ("Ct.Po", "\u2013"), ("Tb.BMD, Ct.BMD", "mg HA/cm\u00b3")]
        n = len(rows)
        inset = 0.9 / h                             # 0.9 mm clear of the frame at top and bottom (axes fraction)
        for i, (nm, unit) in enumerate(rows):
            yy = 1 - inset - (i + 0.5) * (1 - 2 * inset) / n
            ax.text(0.05, yy, nm, fontsize=FS_BODY, color=INK, ha="left", va="center", transform=ax.transAxes)
            ax.text(0.95, yy, unit, fontsize=FS_BODY, color=MUTED, ha="right", va="center", transform=ax.transAxes)
        ax.text(0.5, 1.0, "report", fontsize=FS_BODY, color=MUTED, ha="center", va="bottom", transform=ax.transAxes)
        return
    ax = page.axes_mm(x, y, w, h)
    Hpx, Wpx = D["grey"].shape
    if kind == "tbth":
        vmax = int(D["tbth"].max())
        cm, norm = tbth_cmap(vmax)
        ax.imshow(D["tbth"], cmap=cm, norm=norm, interpolation="nearest")
        ax.contour(D["g_trab"].astype(float), levels=[0.5], colors=["#9A9A9A"], linewidths=0.5)
        # colour bar in the caption slot: map value = sphere diameter in voxels, labelled in mm
        el = float(D["el"])
        label_w = page.width_mm(spec["caption"], FS_BODY) + 1.5
        cw = w - label_w
        cb_top = caption_y + 1.0                     # the bar's top edge; the label is centred on the bar, clear of the frame
        page.text(x, cb_top + 0.8, spec["caption"], fs=FS_BODY, color=MUTED, va="center")
        cax = page.axes_mm(x + label_w, cb_top, cw, 1.6)
        cax.imshow(np.arange(vmax + 1)[None, :], cmap=cm, norm=norm, aspect="auto", interpolation="nearest",
                   extent=(-0.5, vmax + 0.5, 0, 1))
        cax.set_xlim(-0.5, vmax + 0.5)
        cax.set_ylim(0, 1)
        cax.set_xticks([])
        cax.set_yticks([])
        for s in cax.spines.values():
            s.set_edgecolor(frame_col)
            s.set_linewidth(0.4)
        for t in (0.1, 0.2, 0.3, 0.4):
            v = t / el
            if v <= vmax + 0.5:
                cax.plot([v, v], [0, 0.3], color=INK2, lw=0.5, clip_on=False)
                page.text(x + label_w + cw * (v + 0.5) / (vmax + 1), cb_top + 1.6 + 0.2, f"{t:.1f}",
                          fs=FS_BODY, color=INK2, ha="center", va="top")
    else:
        rgb, overlays = thumb_image(kind, D)
        ax.imshow(rgb, interpolation="antialiased")
        for m, col, lw in overlays:
            ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=lw)
        if kind == "grey":                          # scale bar on the first thumbnail only
            n = 5.0 / float(D["el"])
            x0, y0 = Wpx * 0.03, Hpx * 0.93
            ax.plot([x0, x0 + n], [y0, y0], color="white", lw=1.6, solid_capstyle="butt")
            ax.text(x0 + n / 2, y0 - Hpx * 0.04, "5 mm", color="white", fontsize=FS_BODY, ha="center", va="bottom")
        if spec.get("caption"):
            page.block(x + w / 2, caption_y + 0.4, spec["caption"], w, FS_BODY, color=MUTED, ha="center")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(frame_col)
        s.set_linewidth(0.5)


# ------------------------------------------------------------------------------------------------ the page
def build(D, S):
    specs = stage_specs(S)
    aspect = D["grey"].shape[1] / D["grey"].shape[0]
    tw = BW - 2 * PAD
    th = tw / aspect
    # measure the text parts on a scratch page to fix the box height
    scratch = Page(W_MM, 100.0)
    heights = [content_height(scratch, s, tw) for s in specs]
    text_h = max(heights)
    print("text part per box (mm):", ", ".join(f"{s['num']}: {h:.1f}" for s, h in zip(specs, heights)))
    box_h = PAD + text_h + 1.6 + th + CAPTION_SLOT + PAD
    # the legend band: two columns of swatches whose widths are measured from the labels (left, the two classes of
    # IPL reimplementation; right, the ORMIR-XCT and input/output stages), and the note to the right of them when
    # it fits in three lines there, else under them at full width
    keys = ("identical", "differs", "ormir", "io")
    swatch, gap_col, row_h = 6.3, 3.5, 4.0                                # swatch + gap before the label; column gap; row pitch
    col_w = [max(scratch.width_mm(CAT[k]["label"], FS_BODY) for k in keys[2 * c:2 * c + 2]) + swatch + gap_col for c in (0, 1)]
    col_x = [MARGIN, MARGIN + col_w[0]]
    note = (f"Border color: relation of the stage to IPL {S['ipl_ver']}, the reference (green and orange: stages that "
            f"reimplement IPL); agreement statements are pooled over the n = {S['n']} scans, stages 3–5 and 7 run from IPL's "
            f"periosteal contour (configuration B), stage 6 from IPL's segmentation and contours (configuration A). Thumbnails: "
            f"one patella scan, mid-stack slice ({float(D['el']) * 1e3:.1f} µm voxels).")
    nx = col_x[1] + col_w[1]
    note_lines = scratch.wrap(note, W_MM - MARGIN - nx, FS_BODY) if W_MM - MARGIN - nx > 30 else None
    if note_lines is not None and len(note_lines) <= 3:
        ny_off = -0.2
        legend_h = max(LEGEND_H, 2 * row_h, len(note_lines) * lh(FS_BODY) + 0.4)
    else:                                                                 # the note under the swatches, full width
        nx = MARGIN
        note_lines = scratch.wrap(note, W_MM - 2 * MARGIN, FS_BODY)
        ny_off = 2 * row_h + 0.2
        legend_h = 2 * row_h + 0.2 + len(note_lines) * lh(FS_BODY) + 0.4
    plt.close(scratch.fig)
    print(f"legend columns at {col_x[0]:.1f} / {col_x[1]:.1f} mm, note from {nx:.1f} mm ({W_MM - MARGIN - nx:.1f} mm wide, "
          f"{len(note_lines)} lines); legend band {legend_h:.1f} mm")
    H_MM = MARGIN + box_h + CORRIDOR + box_h + 2.0 + legend_h + MARGIN
    page = Page(W_MM, H_MM)
    print(f"page {W_MM:.0f} x {H_MM:.1f} mm; box {BW:.1f} x {box_h:.1f} mm; text part {text_h:.1f} mm; thumbnail {tw:.1f} x {th:.1f} mm")

    boxes = []
    for i, spec in enumerate(specs):
        r, c = divmod(i, NCOL)
        x = MARGIN + c * (BW + GAP)
        y = MARGIN + r * (box_h + CORRIDOR)
        st = CAT[spec["cat"]]
        page.ax.add_patch(patches.FancyBboxPatch((x, y), BW, box_h, boxstyle="round,pad=0,rounding_size=1.4",
                                                 fc=st["fc"], ec=st["ec"], lw=1.1, ls=st["ls"], zorder=2))
        draw_text_part(page, spec, x + PAD, y + PAD, tw)
        ty = y + box_h - PAD - CAPTION_SLOT - th
        draw_thumbnail(page, spec, D, x + PAD, ty, tw, th, ty + th)
        boxes.append(dict(x=x, y=y, w=BW, h=box_h, cx=x + BW / 2, cy=y + box_h / 2, right=x + BW, bottom=y + box_h))

    # flow arrows: along each row at the title line, and the turn from box 4 to box 5 through the corridor
    ya = MARGIN + PAD + pt2mm(FS_TITLE) * 0.6
    for i in (0, 1, 2, 4, 5, 6):
        a, b = boxes[i], boxes[i + 1]
        yy = a["y"] + PAD + pt2mm(FS_TITLE) * 0.6
        page.arrow((a["right"] + 0.15, yy), (b["x"] - 0.15, yy))
    a, b = boxes[3], boxes[4]
    ymid = a["bottom"] + CORRIDOR / 2
    page.polyline([(a["cx"], a["bottom"] + 0.15), (a["cx"], ymid), (b["cx"], ymid), (b["cx"], b["y"] - 0.15)])

    # legend: the four border colours (2 x 2, measured above), then the note on the right
    ly = boxes[-1]["bottom"] + 2.2
    for i, k in enumerate(keys):
        st = CAT[k]
        xx = col_x[i // 2]
        yy = ly + (i % 2) * row_h
        page.ax.add_patch(patches.FancyBboxPatch((xx, yy), 5.0, 2.8, boxstyle="round,pad=0,rounding_size=0.6",
                                                 fc=st["fc"], ec=st["ec"], lw=1.0, ls=st["ls"], zorder=3))
        page.text(xx + swatch, yy + 1.4, st["label"], fs=FS_BODY, color=INK, va="center")
    y_end = page.block(nx, ly + ny_off, note, W_MM - MARGIN - nx, FS_BODY, color=MUTED)
    assert y_end <= H_MM - MARGIN + 0.05, f"the legend note ends at {y_end:.1f} mm on a {H_MM:.1f} mm page"
    return page


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-cache", action="store_true", help="re-read the volumes instead of the cached slice crops")
    args = ap.parse_args()
    S = load_facts()
    D = get_slices(use_cache=not args.no_cache)
    page = build(D, S)
    page.fig.savefig(OUT_PNG, dpi=300, facecolor="white")
    page.fig.savefig(OUT_SVG, facecolor="white")
    print("wrote", OUT_PNG)
    print("wrote", OUT_SVG)
    json.dump(S, open(OUT_NUM, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", OUT_NUM)


if __name__ == "__main__":
    main()
