"""S4_contour.py -- Supplementary Figure S4 of the ipldt / ORMIR-BQRL manuscript: contour rendering.

What IPL's /togobj_from_aim -curvature_smooth 1 (mask -> contour object) followed by /gobj_to_aim (contour -> raster)
does, rule by rule, and where the rendered contour differs from the mask it came from.  Ten panels, 180 mm wide,
300 dpi PNG + SVG, panel letters A-J upper-left:

  A  one cortical mask slice of a patella: the raw mask, the pixels the rendering drops and adds, ipldt = IPL
  B  zoom of A: the stored chain through the pixel centres and the edge of its raster (identical for IPL and ipldt)
  C  all 21 patellae, both masks (42 volumes): voxels dropped and added per million mask voxels;
     ipldt's rendering vs IPL's: 0 differing voxels in every volume
  D  phase 1 (the run-of-1 pixel rule and its two exceptions) and the counter-clockwise Moore trace    (synthetic)
  E  curvature smoothing, stage A: one sweep of four vertex rules                                       (synthetic)
  F  the gate and stage B: all five rules until a sweep changes nothing; the stored chain               (synthetic)
  G  rasterisation: chain pixels + the pixel centres strictly inside the polygon (even-odd scanline rule) (synthetic)
  H  an inner contour on a real cortical slice: only its strict interior is removed; a lobe enclosed by chain
     pixels but outside the polygon (even number of crossings to its left) is kept, as IPL keeps it
  I  the gate on a real trabecular slice: horizontal notches only, no stage-A event -> the chain is stored raw and
     the slice renders as the mask; unconditional smoothing would differ from IPL
  J  the minimum-vertex rule on the tiny-object phantom IPL evaluated: chains left with fewer than 4 vertices are
     not stored (no contour, nothing rendered); a raw 6-vertex chain is stored (the 4..6 class; 7..8 refuted)

Panels D-G use a 20 x 24 synthetic slice that exercises every rule (a real slice never shows all five in one crop);
every other panel is real data: IPL's compartment masks of the patellae and IPL's own contour renderings (probe-15
exports), IPL's stored contour files, and the probe-19 phantom with IPL's rendering and contour file of it.

Run from the repository root with the ormir python (about 2 min the first time: 42 volumes rendered for panel C and
compared with IPL's exports; cached afterwards in manuscript/figures/cache/S4_contour_cohort.json):

    set PYTHONUTF8=1
    python manuscript/figures/supp/S4_contour.py [--recompute]

Writes manuscript/figures/supp/S4_contour.png (300 dpi), S4_contour.svg and S4_contour_numbers.json (every number drawn).

Inputs (read only):
  * IPL's compartment masks    <IPLDT_LAB_ROOT>/PFJOA/XCT_masks_full_grab/<subj>/<base>_{CORT,TRAB}_MASK.AIM
  * IPL's contour renderings   <IPLDT_LAB_ROOT>/Python/scripts/IPL/probes/p15_gobj_render/aims_and_logs/<base>_P15_<kind>_G2A.AIM
    and stored contour files   .../<base>_<kind>_MASK.GOBJ;n  (P5MASK.GOBJ / P7MASK.GOBJ for two trabecular contours)
  * the probe-19 phantom       <IPLDT_PROBE19_MIRROR>/{phantom19_manifest.json, oracle/X2420448_P19_TINY.AIM, oracle/P19TINY.GOBJ}
Nothing under ipldt/ is modified; every ipldt result comes from calling ipldt.contour on those files.  The only
re-implementation is an event-logging copy of ipldt.contour.smooth.sweep for the markers of panels E and F, asserted
equal to the package's sweep on every chain it annotates.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
import ipldt  # noqa: E402
from ipldt.contour.render import (MIN_VERTICES, phase1, outer_components, raster_first, holes, polygon_fill,  # noqa: E402
                                  render_slice, render_volume, slice_chains)
from ipldt.contour.moore import moore_trace  # noqa: E402
from ipldt.contour import smooth as SM  # noqa: E402
from ipldt.contour.gobj_file import read_gobj, path as gobj_path  # noqa: E402

# ------------------------------------------------------------------------------------------------ inputs
GRAB = lab_path("PFJOA/XCT_masks_full_grab")
P15 = lab_path("Python/scripts/IPL/probes/p15_gobj_render/aims_and_logs")
P19 = os.environ.get("IPLDT_PROBE19_MIRROR") or lab_path("ipl_probes/p19_open_halves")
CACHE = os.path.join(REPO, "manuscript", "figures", "cache", "S4_contour_cohort.json")
OUT = os.path.join(HERE, "S4_contour")
VOX_MM = 0.0607

# the 21 patellae (subject, scanner id) and the two trabecular contour files IPL wrote under working names
COHORT = [("PFJ-ab6af9_R", "X3931708"), ("PFJ-0be66a_R", "X2420448"), ("PFJ-351dc7_R", "X9463122"), ("PFJ-8bcf88_L", "X4121991"),
          ("PFJ-8bcf88_R", "X4891377"), ("PFJ-d81140_L", "X5366335"), ("PFJ-d81140_R", "X1130319"), ("PFJ-0fb201_R", "X5578364"),
          ("PFJ-a5ee4c_R", "X3744142"), ("PFJ-5f00b4_R", "X5974634"), ("PFJ-a91200_L", "X3186488"), ("PFJ-b5d5d9_L", "X7775442"),
          ("PFJ-42293d_L", "X3623103"), ("PFJ-a56aae_R", "X5606590"), ("PFJ-6f5538_R", "X5492058"), ("PFJ-6b714e_L", "X5484780"),
          ("PFJ-6b714e_R", "X2663245"), ("PFJ-69bcb0_L", "X1183449"), ("PFJ-69bcb0_R", "X1346001"), ("PFJ-411dfd_L", "X3334670"),
          ("PFJ-411dfd_R", "X5143651")]
BASE = dict(COHORT)
TRAB_GOBJ_SPECIAL = {"PFJ-0be66a_R": "P5MASK.GOBJ", "PFJ-8bcf88_R": "P7MASK.GOBJ"}

# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito (colour-blind safe): blue = ipldt / chain, vermilion = dropped, green = added, orange = start vertex,
# purple = spur / neck, sky = strict interior, yellow = kept lobe; greys for masks and text
C_BLUE, C_VERM, C_GREEN, C_ORANGE, C_PURPLE, C_SKY, C_YELLOW = ("#0072B2", "#D55E00", "#009E73", "#E69F00",
                                                                "#CC79A7", "#56B4E9", "#F0E442")
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MASK_GREY, MASK_GREY_LIGHT, RAW_GREY = "#bdbdbd", "#d9d9d9", "#9a9a9a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.3, "axes.linewidth": 0.5, "axes.edgecolor": C_AXIS, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.color": C_INK2,
    "ytick.color": C_INK2, "savefig.dpi": 300, "figure.dpi": 100, "pdf.fonttype": 42, "svg.fonttype": "none",
    "text.color": C_INK, "axes.labelcolor": C_INK2, "legend.frameon": False, "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5, "legend.columnspacing": 1.0, "legend.borderaxespad": 0.0, "legend.labelspacing": 0.35,
})
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def fmt(n):
    return f"{n:,d}"


# ------------------------------------------------------------------------------------------------ data access
def read_mask(subj, kind):
    return ipldt.read_aim(f"{GRAB}/{subj}/{BASE[subj]}_{kind}_MASK.AIM")


def read_export(subj, kind, tag="G2A"):
    """IPL's probe-15 export <base>_P15_<kind>_<tag>.AIM: /gobj_to_aim of the mask's /togobj_from_aim contour."""
    return ipldt.read_aim(f"{P15}/{BASE[subj]}_P15_{kind}_{tag}.AIM")


def ipl_on_grid(subj, kind, mask):
    """IPL's rendering pasted onto the mask's own grid by global position (bool volume)."""
    return ipldt.align_to(read_export(subj, kind), mask["dim"], mask["pos"]) > 0


def gobj_name(subj, kind):
    if kind == "TRAB" and subj in TRAB_GOBJ_SPECIAL:
        return TRAB_GOBJ_SPECIAL[subj]
    return f"{BASE[subj]}_{kind}_MASK.GOBJ"


def newest_gobj(name):
    """Newest VMS version (name;N) of a stored contour file that reads and holds contours: (basename, slices)."""
    files = sorted(glob.glob(f"{P15}/{name};*"), key=lambda p: int(p.rsplit(";", 1)[1]))
    for gp in reversed(files):
        try:
            _, sl = read_gobj(gp)
        except Exception:
            continue
        sl = [s for s in sl if s["contours"]]
        if sl:
            return os.path.basename(gp), sorted(sl, key=lambda s: s["z"])
    raise FileNotFoundError(name)


def gobj_slice_chains(slices, pos, zloc):
    """IPL's stored chains of one slice as vertex lists in local slice coordinates (stored order)."""
    out = []
    for s in slices:
        if s["z"] != zloc + pos[2]:
            continue
        for c in s["contours"]:
            x, y = gobj_path(c["codes"], c["start"], 8)
            out.append([(int(a) - pos[0], int(b) - pos[1]) for a, b in zip(x, y)])
    return out


def ipl_chains(subj, kind, mask, zloc):
    gname, gsl = newest_gobj(gobj_name(subj, kind))
    return gname, gobj_slice_chains(gsl, mask["pos"], zloc)


# ------------------------------------------------------------------------------------------------ drawing helpers
FIG_W, FIG_H = 180.0, 261.0          # mm


def ax_mm(fig, x, y_top, w, h):
    return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H])


def ftext(fig, x, y_top, s, **kw):
    kw.setdefault("va", "top")
    return fig.text(x / FIG_W, 1 - y_top / FIG_H, s, **kw)


def show_mask(ax, img, x0=0, y0=0, grey=MASK_GREY):
    h, w = img.shape
    rgb = np.ones(img.shape + (3,))
    rgb[img] = matplotlib.colors.to_rgb(grey)
    ax.imshow(rgb, interpolation="nearest", extent=(x0 - 0.5, x0 + w - 0.5, y0 + h - 0.5, y0 - 0.5))
    ax.set_xlim(x0 - 0.5, x0 + w - 0.5)
    ax.set_ylim(y0 + h - 0.5, y0 - 0.5)
    ax.set_aspect("equal")


def pixel_grid(ax, x0, y0, w, h, color="#e4e4e4", lw=0.3):
    segs = [[(x0 + i - 0.5, y0 - 0.5), (x0 + i - 0.5, y0 + h - 0.5)] for i in range(w + 1)]
    segs += [[(x0 - 0.5, y0 + j - 0.5), (x0 + w - 0.5, y0 + j - 0.5)] for j in range(h + 1)]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=1.5))


def raster_outline(ax, img, x0=0, y0=0, **kw):
    """The pixel-edge boundary of a bool raster (stair-step outline)."""
    segs = []
    v = img[:, 1:] != img[:, :-1]
    for y, x in zip(*np.nonzero(v)):
        segs.append([(x0 + x + 0.5, y0 + y - 0.5), (x0 + x + 0.5, y0 + y + 0.5)])
    hz = img[1:] != img[:-1]
    for y, x in zip(*np.nonzero(hz)):
        segs.append([(x0 + x - 0.5, y0 + y + 0.5), (x0 + x + 0.5, y0 + y + 0.5)])
    # the raster's own frame edges (object touching the crop border)
    for x in range(img.shape[1]):
        if img[0, x]:
            segs.append([(x0 + x - 0.5, y0 - 0.5), (x0 + x + 0.5, y0 - 0.5)])
        if img[-1, x]:
            segs.append([(x0 + x - 0.5, y0 + img.shape[0] - 0.5), (x0 + x + 0.5, y0 + img.shape[0] - 0.5)])
    for y in range(img.shape[0]):
        if img[y, 0]:
            segs.append([(x0 - 0.5, y0 + y - 0.5), (x0 - 0.5, y0 + y + 0.5)])
        if img[y, -1]:
            segs.append([(x0 + img.shape[1] - 0.5, y0 + y - 0.5), (x0 + img.shape[1] - 0.5, y0 + y + 0.5)])
    kw.setdefault("linewidths", 0.9)
    ax.add_collection(LineCollection(segs, **kw))


def squares(ax, img, x0=0, y0=0, color=C_VERM, ms=4.0, alpha=1.0, zorder=3, **kw):
    yy, xx = np.nonzero(img)
    if len(xx):
        ax.plot(xx + x0, yy + y0, "s", ms=ms, color=color, mec="none", alpha=alpha, zorder=zorder, ls="none", **kw)


def chain_line(ax, pts, color=C_BLUE, lw=0.9, ms=1.8, zorder=4, closed=True, **kw):
    px = [p[0] for p in pts] + ([pts[0][0]] if closed else [])
    py = [p[1] for p in pts] + ([pts[0][1]] if closed else [])
    ax.plot(px, py, "-", color=color, lw=lw, zorder=zorder, solid_joinstyle="round", **kw)
    if ms:
        ax.plot(px[:-1] if closed else px, py[:-1] if closed else py, "o", color=color, ms=ms, mec="none",
                zorder=zorder + 0.1)


def in_crop_runs(pts, x0, y0, w, h):
    runs, cur = [], []
    for x, y in pts:
        if x0 - 1 <= x < x0 + w + 1 and y0 - 1 <= y < y0 + h + 1:
            cur.append((x, y))
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


def draw_runs(ax, pts, x0, y0, w, h, color=C_BLUE, lw=0.9, ms=1.8, zorder=6, **kw):
    for r in in_crop_runs(pts, x0, y0, w, h):
        ax.plot([p[0] for p in r], [p[1] for p in r], "-o", color=color, lw=lw, ms=ms, mec="none", zorder=zorder, **kw)


def grid_axes(ax, x0, y0, w, h, step=4, px_labels=True):
    ax.set_xlim(x0 - 0.5, x0 + w - 0.5)
    ax.set_ylim(y0 + h - 0.5, y0 - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(x0 - x0 % step + (step if x0 % step else 0), x0 + w, step))
    ax.set_yticks(range(y0 - y0 % step + (step if y0 % step else 0), y0 + h, step))
    ax.tick_params(length=1.5, pad=1.2)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    if not px_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def check_text_widths(fig, items):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for t, x0, x1, label in items:
        bb = t.get_window_extent(renderer=r)
        lo, hi = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        if lo < x0 - 0.3 or hi > x1 + 0.3:
            bad += 1
            say(f"  WARNING text overflows its {x1 - x0:.1f} mm span by {max(x0 - lo, hi - x1):.1f} mm: {label[:70]!r}")
    say(f"text-width check: {len(items)} texts, {bad} overflow")
    return bad


# ------------------------------------------------------------------------------------------------ the sweep with events
def sweep_events(pts, rules):
    """ipldt.contour.smooth.sweep with an event log [(rule, old vertex, new vertex or None, partner or None)];
    the caller asserts it equal to the package's sweep."""
    n = len(pts)
    if n < 4:
        return list(pts), False, []
    new = list(pts)
    keep = [True] * n
    changed = False
    ev = []
    for i in list(range(1, n)) + [0]:
        if not keep[i]:
            continue
        ip = (i - 1) % n
        while not keep[ip]:
            ip = (ip - 1) % n
        inx = (i + 1) % n
        while not keep[inx]:
            inx = (inx + 1) % n
        if ip == i or inx == i:
            break
        (px, py), (nx, ny), here = new[ip], new[inx], new[i]
        dx, dy = nx - px, ny - py
        ad = (abs(dx), abs(dy))
        if ad == (1, 1):
            if SM.CORNER in rules:
                keep[i] = False; changed = True; ev.append((SM.CORNER, here, None, None))
        elif ad == (0, 0):
            if SM.SPUR in rules:
                keep[i] = False; keep[inx] = False; changed = True; ev.append((SM.SPUR, here, None, new[inx]))
        elif ad in ((1, 0), (0, 1)):
            if SM.SHARP in rules:
                keep[i] = False; changed = True; ev.append((SM.SHARP, here, None, None))
        elif ad == (0, 2):
            if SM.EXCURSION_V in rules:
                mid = (px, py + dy // 2)
                if mid != here:
                    new[i] = mid; changed = True; ev.append((SM.EXCURSION_V, here, mid, None))
        elif ad == (2, 0):
            if SM.EXCURSION_H in rules:
                mid = (px + dx // 2, py)
                if mid != here:
                    new[i] = mid; changed = True; ev.append((SM.EXCURSION_H, here, mid, None))
    survivors = [new[k] for k in range(n) if keep[k]]
    if not keep[0] and survivors:
        survivors = survivors[-1:] + survivors[:-1]
    return survivors, changed, ev


EV_STYLE = {SM.CORNER: ("x", C_VERM, "corner, |d| = (1, 1): vertex deleted"),
            SM.SPUR: ("X", C_PURPLE, "spur tip, d = (0, 0): vertex and its successor deleted"),
            SM.SHARP: ("P", C_ORANGE, "135° turn, |d| = (1, 0) or (0, 1): vertex deleted"),
            SM.EXCURSION_V: (">", C_GREEN, "excursion on vertical travel, |d| = (0, 2): moved to the midpoint"),
            SM.EXCURSION_H: ("v", C_BLUE, "excursion on horizontal travel, |d| = (2, 0): moved to the midpoint")}


def draw_events(ax, events, ms=5.5):
    for rule, old, new, partner in events:
        mk, col, _ = EV_STYLE[rule]
        if new is None:
            ax.plot(old[0], old[1], mk, color=col, ms=ms, mew=1.3, mec=col, mfc="none" if mk == "P" else col, zorder=8)
            if partner is not None:
                ax.plot(partner[0], partner[1], mk, color=col, ms=ms, mew=1.3, zorder=8)
        else:
            ax.annotate("", xy=new, xytext=old, arrowprops=dict(arrowstyle="->", color=col, lw=1.1, shrinkA=0,
                                                                  shrinkB=1.5), zorder=8)
            ax.plot(old[0], old[1], "o", color=col, ms=ms * 0.75, mfc="none", mew=1.1, zorder=8)


def event_handles(rules):
    hs = []
    for rule in (SM.CORNER, SM.SPUR, SM.SHARP, SM.EXCURSION_V, SM.EXCURSION_H):
        if rule not in rules:
            continue
        mk, col, lab = EV_STYLE[rule]
        if rule in (SM.EXCURSION_V, SM.EXCURSION_H):
            hs.append(Line2D([], [], marker="o", mfc="none", mec=col, color=col, ms=4, mew=1.1, ls="-", lw=1.0, label=lab))
        else:
            hs.append(Line2D([], [], marker=mk, color=col, mec=col, mfc="none" if mk == "P" else col, ms=5, mew=1.3,
                             ls="none", label=lab))
    return hs


# ------------------------------------------------------------------------------------------------ synthetic slices
def synthetic_outer():
    """A 20 x 24 slice exercising every outer-contour rule."""
    m = np.zeros((20, 24), bool)
    m[6:14, 4:18] = True        # the body
    m[5, 4] = True              # bump: a run of 1 that is the raster-first pixel -> kept (trace start)
    m[6, 3] = True              # bump on the left (vertical) edge -> vertical-travel excursion
    m[5, 12] = True             # bump on the top edge: a run of 1 -> dropped by phase 1 (activates stage B)
    m[9, 17] = False            # notch in the right (vertical) edge -> vertical-travel excursion (stage A)
    m[6, 14] = False            # notch in the top (horizontal) edge -> horizontal-travel excursion (stage B only)
    m[10, 18:20] = True         # a 2-pixel horizontal spur -> spur tip
    m[14, 8] = True             # a vertical neck: run of 1 with mask above and below -> kept
    m[15:17, 7:10] = True       # the lobe it holds
    return m


def synthetic_chain_data():
    sl = synthetic_outer()
    comps = outer_components(sl)
    assert len(comps) == 1
    comp, pre = comps[0]
    dropped = sl & ~comp
    start = raster_first(sl)
    Wn = np.zeros_like(sl); En = np.zeros_like(sl); Nn = np.zeros_like(sl); Sn = np.zeros_like(sl)
    Wn[:, 1:] = sl[:, :-1]; En[:, :-1] = sl[:, 1:]; Nn[1:] = sl[:-1]; Sn[:-1] = sl[1:]
    run1 = sl & ~Wn & ~En
    neck = run1 & Nn & Sn
    raw = moore_trace(comp, raster_first(comp), backtrack=(-1, 0), ccw=True)
    a_pts, a_changed, a_ev = sweep_events(raw, SM.STAGE_A_RULES)
    a_ref, a_changed_ref = SM.sweep(raw, SM.STAGE_A_RULES)
    assert a_pts == a_ref and a_changed == a_changed_ref
    b_pts, b_ev, nsweeps = list(a_pts), [], 0
    while True:
        nxt, ch, ev = sweep_events(b_pts, SM.ALL_RULES)
        ref, ch_ref = SM.sweep(b_pts, SM.ALL_RULES)
        assert nxt == ref and ch == ch_ref
        nsweeps += 1
        if not ch:
            break
        b_pts, b_ev = nxt, b_ev + ev
    st = {}
    final = SM.smooth_chain(raw, pre, st)
    assert final == b_pts and st["activated"] and st["sweeps"] == nsweeps
    stored = slice_chains(sl)
    assert len(stored) == 1 and stored[0]["vertices"] == final
    chain_px, interior = polygon_fill(final, sl.shape)
    G = render_slice(sl)
    assert np.array_equal(G, chain_px | interior)
    return dict(sl=sl, comp=comp, pre=pre, dropped=dropped, start=start, neck=neck, raw=raw, a_pts=a_pts, a_ev=a_ev,
                final=final, b_ev=b_ev, nsweeps=nsweeps, chain_px=chain_px, interior=interior, G=G)


# ------------------------------------------------------------------------------------------------ real data
def slice_overlay_data(subj="PFJ-0be66a_R", kind="CORT", n=28, margin=3, z_margin=8):
    """The n x n window of one compartment mask volume that shows the most rendering differences of both kinds
    (dropped and added) at least `margin` pixels from its border, and the slice it lies on (a deterministic
    argmax over the volume without its first and last `z_margin` slices)."""
    mask = read_mask(subj, kind)
    M = mask["data"] > 0
    Gi = ipl_on_grid(subj, kind, mask)
    drop, add = M & ~Gi, ~M & Gi
    m = n - 2 * margin

    def box_sums(v):                                   # sums over every m x m window of every slice
        c = np.pad(v.astype(np.int32), ((0, 0), (1, 0), (1, 0))).cumsum(1).cumsum(2)
        return c[:, m:, m:] - c[:, :-m, m:] - c[:, m:, :-m] + c[:, :-m, :-m]

    dd, aa = box_sums(drop), box_sums(add)
    score = np.minimum(dd, aa) * 100 + dd + aa
    score[:z_margin] = -1
    score[M.shape[0] - z_margin:] = -1
    z = int(np.unravel_index(int(np.argmax(score)), score.shape)[0])
    # the equally scored windows of that slice form a block (every window holding the same pixels): take its centre
    ys_, xs_ = np.nonzero(score[z] == score[z].max())
    yi, xi = int(np.median(ys_)), int(np.median(xs_))
    x0 = int(max(0, min(M.shape[2] - n, xi - margin)))
    y0 = int(max(0, min(M.shape[1] - n, yi - margin)))
    sl, gi = M[z], Gi[z]
    ours = render_slice(sl)
    chains = slice_chains(sl)
    gname, ipl = ipl_chains(subj, kind, mask, z)
    assert [c["vertices"] for c in chains] == ipl, "stored chains differ from IPL's"
    ndiff_slice = int((ours != gi).sum())
    d2, a2 = drop[z], add[z]
    return dict(subj=subj, kind=kind, base=BASE[subj], z=z, z_global=z + int(mask["pos"][2]), sl=sl, gi=gi, ours=ours,
                d2=d2, a2=a2, chains=chains, gobj=gname, crop=(x0, y0, n), ndiff_slice=ndiff_slice,
                mask_px=int(sl.sum()), ipl_px=int(gi.sum()), dropped=int(d2.sum()), added=int(a2.sum()),
                slices=int(mask["dim"][2]), crop_dropped=int(d2[y0:y0 + n, x0:x0 + n].sum()),
                crop_added=int(a2[y0:y0 + n, x0:x0 + n].sum()))


def cohort_rows(recompute=False):
    """Per volume (21 patellae x cortical / trabecular): mask voxels, IPL's rendered voxels, dropped, added, and
    the voxels on which ipldt's rendering differs from IPL's (compared on the union of the two grids)."""
    if not recompute and os.path.exists(CACHE):
        rows = json.load(open(CACHE))
        say(f"cohort: {len(rows)} volumes loaded from {CACHE}")
        return rows
    rows = []
    for subj, base in COHORT:
        for kind in ("CORT", "TRAB"):
            mask = read_mask(subj, kind)
            M = mask["data"] > 0
            ours = render_volume(M)
            e = read_export(subj, kind)
            lo = tuple(min(mask["pos"][i], e["pos"][i]) for i in range(3))
            hi = tuple(max(mask["pos"][i] + mask["dim"][i], e["pos"][i] + e["dim"][i]) for i in range(3))
            dim = tuple(hi[i] - lo[i] for i in range(3))
            A = ipldt.align_to(dict(data=ours.astype(np.uint8), dim=mask["dim"], pos=mask["pos"]), dim, lo) > 0
            B = ipldt.align_to(e, dim, lo) > 0
            Mu = ipldt.align_to(dict(data=M.astype(np.uint8), dim=mask["dim"], pos=mask["pos"]), dim, lo) > 0
            rows.append(dict(subj=subj, base=base, kind=kind, slices=int(mask["dim"][2]), mask=int(M.sum()),
                             ours=int(ours.sum()), ipl=int(B.sum()), dropped=int((Mu & ~B).sum()),
                             added=int((~Mu & B).sum()), mismatch=int((A != B).sum())))
            r = rows[-1]
            say(f"cohort {subj} {kind}: mask {r['mask']:,d} IPL {r['ipl']:,d} ours {r['ours']:,d} dropped {r['dropped']} "
                f"added {r['added']} ipldt-vs-IPL {r['mismatch']}")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(rows, open(CACHE, "w"), indent=1)
    return rows


def finger_data(subj="PFJ-a56aae_R", kind="CORT", zg=258, w=32, h=24):
    """A real cortical slice whose inner chain wraps a finger of cortex through a 1-pixel neck."""
    mask = read_mask(subj, kind)
    z = zg - int(mask["pos"][2])
    sl = mask["data"][z] > 0
    gi = ipl_on_grid(subj, kind, mask)[z]
    ours = slice_chains(sl)
    gname, ipl = ipl_chains(subj, kind, mask, z)
    assert [c["vertices"] for c in ours] == ipl
    inner = [c for c in ours if c["kind"] == "inner"][0]["vertices"]
    ich, iint = polygon_fill(inner, sl.shape)
    enclosed = ndi.binary_fill_holes(ich)
    lobe = enclosed & ~iint & ~ich
    G = render_slice(sl)
    alt = G.copy(); alt[enclosed & ~ich] = False
    alt_diff = int((alt != gi).sum())
    ndiff = int((G != gi).sum())
    ys_l, xs_l = np.nonzero(lobe)
    cx, cy = (xs_l.min() + xs_l.max()) / 2, (ys_l.min() + ys_l.max()) / 2
    x0 = max(0, min(sl.shape[1] - w, int(round(cx - w / 2)))); y0 = max(0, min(sl.shape[0] - h, int(round(cy - h / 2))))
    # the scanline with the most lobe pixels, and the chain crossings on it (even-odd rule)
    row = int(np.bincount(ys_l).argmax())
    xs = np.array([p[0] for p in inner]); ys = np.array([p[1] for p in inner])
    x1, y1 = np.roll(xs, -1), np.roll(ys, -1)
    dy = y1 - ys
    cxs = np.where(dy > 0, xs, x1)[dy != 0]
    cys = np.where(dy > 0, ys, y1)[dy != 0]
    cross = sorted(cxs[cys == row].tolist())
    return dict(subj=subj, kind=kind, z=z, z_global=zg, sl=sl, gi=gi, G=G, inner=inner, ich=ich, iint=iint, lobe=lobe,
                alt_diff=alt_diff, ndiff=ndiff, crop=(x0, y0, w, h), row=row, cross=cross, gobj=gname,
                n_inner=len(inner), lobe_px=int(lobe.sum()))


def stored_raw_data(subj="PFJ-d81140_R", kind="TRAB", z=0, w=40, h=28):
    """A real trabecular slice whose chain has horizontal notches only: stored raw by the gate."""
    mask = read_mask(subj, kind)
    sl = mask["data"][z] > 0
    gi = ipl_on_grid(subj, kind, mask)[z]
    gname, ipl = ipl_chains(subj, kind, mask, z)
    (comp, pre), = outer_components(sl)
    raw = moore_trace(comp, raster_first(comp), backtrack=(-1, 0), ccw=True)
    st = {}
    stored = SM.smooth_chain(raw, pre, st)
    ours = slice_chains(sl)
    assert ours[0]["vertices"] == stored == ipl[0] and not st["activated"] and stored == raw
    n = len(raw)
    exc = []
    for i in range(n):
        (px, py), (nx, ny), here = raw[i - 1], raw[(i + 1) % n], raw[i]
        if (abs(nx - px), abs(ny - py)) == (2, 0) and (px + (nx - px) // 2, py) != here:
            exc.append((here, (px + (nx - px) // 2, py)))
    pts, _ = SM.sweep(raw, SM.STAGE_A_RULES)
    for _ in range(200):
        pts, ch = SM.sweep(pts, SM.ALL_RULES)
        if not ch:
            break
    unc = polygon_fill(pts, sl.shape)
    unc_raster = unc[0] | unc[1]
    unc_diff = int((unc_raster != gi).sum())
    ndiff = int((render_slice(sl) != gi).sum())
    near = sorted(exc, key=lambda e: abs(e[0][0] - raw[0][0]) + abs(e[0][1] - raw[0][1]))[:2]
    pts_c = [raw[0]] + [e[0] for e in near]
    xs = [p[0] for p in pts_c]; ys = [p[1] for p in pts_c]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    x0 = max(0, min(sl.shape[1] - w, int(round(cx - w / 2)))); y0 = max(0, min(sl.shape[0] - h, int(round(cy - h / 2))))
    return dict(subj=subj, kind=kind, z=z, z_global=z + int(mask["pos"][2]), sl=sl, gi=gi, raw=raw, exc=exc,
                unc_raster=unc_raster, unc_diff=unc_diff, ndiff=ndiff, crop=(x0, y0, w, h), gobj=gname, n=n,
                mask_px=int(sl.sum()), ipl_px=int(gi.sum()))


# ------------------------------------------------------------------------------------------------ the probe-19 phantom
PH_SHOW = [  # (test, caption)
    ("h_v12", "vertical 1 × 2 hole"),
    ("h_h12", "horizontal 1 × 2 hole"),
    ("h_11", "1 × 1 hole"),
    ("c_2x2", "2 × 2 block"),
    ("c_hex8", "8-pixel hexagon"),
    ("c_4x4", "4 × 4 block"),
    ("c_disc12", "12-pixel disc"),
]
PH_CROP = dict(hole=(104, 105, 12, 10), component=(297, 147, 12, 10))    # (x0, y0, w, h) in phantom pixels


def phantom_data():
    man = json.load(open(os.path.join(P19, "phantom19_manifest.json")))
    dim, pos = tuple(man["dim"]), tuple(man["pos"])
    a = np.zeros(dim[::-1], bool)
    for t in man["tests"].values():
        z0, z1 = t["z"]
        if t["kind"] == "hole":
            b = t["block"]
            a[z0:z1 + 1, b["y"][0]:b["y"][1] + 1, b["x"][0]:b["x"][1] + 1] = True
            for x, y in t["hole_pixels"]:
                a[z0:z1 + 1, y, x] = False
        else:
            for x, y in t["pixels"]:
                a[z0:z1 + 1, y, x] = True
    tiny = ipldt.read_aim(os.path.join(P19, "oracle", "X2420448_P19_TINY.AIM"))
    T = ipldt.align_to(tiny, dim, pos) > 0
    _, gsl = read_gobj(os.path.join(P19, "oracle", "P19TINY.GOBJ"))
    G = render_volume(a)
    assert np.array_equal(G, T), "phantom rendering differs from IPL's"
    tests = {}
    for name, cap in PH_SHOW:
        t = man["tests"][name]
        z = t["z"][0]
        sl = a[z]
        ipl = gobj_slice_chains(gsl, pos, z)
        ours = slice_chains(sl)
        assert [c["vertices"] for c in ours] == ipl, name
        if t["kind"] == "hole":
            lab, ids = holes(sl)
            assert len(ids) == 1
            hole = lab == ids[0]
            hx, hy = raster_first(hole)
            xe = hx
            while xe + 1 < hole.shape[1] and hole[hy, xe + 1]:
                xe += 1
            raw = moore_trace(~hole, (xe + 1, hy), backtrack=(-1, 0), ccw=True)
            st = {}
            fin = SM.smooth_chain(raw, False, st)
            shown = [c["vertices"] for c in ours if c["kind"] == "inner"]
        else:
            comps = outer_components(sl)
            assert len(comps) == 1
            comp, pre = comps[0]
            raw = moore_trace(comp, raster_first(comp), backtrack=(-1, 0), ccw=True)
            st = {}
            fin = SM.smooth_chain(raw, pre, st)
            shown = [c["vertices"] for c in ours if c["kind"] == "outer"]
        stored = len(fin) >= MIN_VERTICES
        assert stored == (len(shown) == 1)
        x0, y0, w, h = PH_CROP[t["kind"]]
        crop = sl[y0:y0 + h, x0:x0 + w]
        ipl_crop = T[z, y0:y0 + h, x0:x0 + w]
        tests[name] = dict(kind=t["kind"], z=z, z_global=z + pos[2], crop=(x0, y0, w, h), mask=crop, ipl=ipl_crop,
                           raw=raw, final=fin, stored=stored, activated=bool(st["activated"]), sweeps=int(st["sweeps"]),
                           n_raw=len(raw), n_final=len(fin), n_ipl=(len(shown[0]) if shown else 0),
                           rendered_px=int(T[z].sum()), mask_px=int(sl.sum()),
                           hole_kept=(bool((~T[z] & ~sl & ndi.binary_fill_holes(sl)).any()) if t["kind"] == "hole" else None))
        say(f"phantom {name}: raw {len(raw)} -> {len(fin)} (activated {st['activated']}, {st['sweeps']} sweeps) "
            f"stored {stored} IPL chain {tests[name]['n_ipl']} rendered {tests[name]['rendered_px']} of {tests[name]['mask_px']} px")
    # the alternative reading, MIN_VERTICES = 8, differs from IPL's rendering on h_v12's hole voxels only
    from ipldt.contour import render as R
    R.MIN_VERTICES = 8
    try:
        G8 = render_volume(a)
    finally:
        R.MIN_VERTICES = MIN_VERTICES
    d8 = int((G8 != T).sum())
    return dict(tests=tests, total_rendered=int(T.sum()), total_mask=int(a.sum()), min8_diff=d8,
                n_chains=sum(len(s["contours"]) for s in gsl))


# ------------------------------------------------------------------------------------------------ draw
def panel_title(fig, x, y, letter, title, sub, checks, span):
    ftext(fig, x, y, letter, fontsize=9, fontweight="bold", ha="left")
    t1 = ftext(fig, x + 3.8, y + 0.35, title, fontsize=7.5, fontweight="bold", ha="left")
    checks.append((t1, x, x + span, title))
    if sub:
        t2 = ftext(fig, x, y + 4.1, sub, fontsize=6.4, color=C_INK2, ha="left", linespacing=1.18)
        checks.append((t2, x, x + span, sub))


def legend_strip(fig, x, y_top, w, h, handles, ncol, fontsize=6.3):
    ax = ax_mm(fig, x, y_top, w, h)
    ax.axis("off")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=ncol, fontsize=fontsize, frameon=False,
              handlelength=1.5, handletextpad=0.5, columnspacing=1.6, labelspacing=0.32, borderaxespad=0.0)
    return ax


def H_sq(color, label, ms=4.5, mec="none", mew=0.0):
    return Line2D([], [], marker="s", ls="none", color=color, mec=mec, mew=mew, ms=ms, label=label)


def H_line(color, label, ls="-", lw=1.0, marker=None, ms=2.2):
    return Line2D([], [], color=color, ls=ls, lw=lw, marker=marker, ms=ms, mec="none", label=label)


def H_circ(color, label, mec=C_INK, mew=0.6, ms=5, fill=True):
    return Line2D([], [], marker="o", ls="none", color=color, mfc=color if fill else "none", mec=mec, mew=mew, ms=ms, label=label)


def draw(S, A, rows, F, I, PH):
    fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4))
    checks = []
    N = dict(A={}, C={}, D={}, E={}, F={}, G={}, H={}, I={}, J={})
    pw, gap = 56.0, 4.0
    xcol = [2.0, 2.0 + pw + gap, 2.0 + 2 * (pw + gap)]

    # ===================================================================================== row 1: the real slice
    y1, ax_y1, hB = 0.0, 13.0, 40.0
    # --- A: the whole slice
    xA, wA = 2.0, 74.0
    Hs, Ws = A["sl"].shape
    hA = wA * Hs / Ws
    x0, y0, n = A["crop"]
    panel_title(fig, xA, y1, "A", "A cortical mask slice and its rendered contour",
                f"one patella, slice {A['z'] + 1} of {A['slices']}; gray = IPL's cortical mask; squares: pixels\n"
                f"dropped by the rendering (vermilion, {A['dropped']}) or added (green, {A['added']}); box = panel B",
                checks, span=wA + 6)
    ax = ax_mm(fig, xA, ax_y1, wA, hA)
    show_mask(ax, A["sl"], grey=MASK_GREY_LIGHT)
    squares(ax, A["d2"], color=C_VERM, ms=2.2, zorder=4)
    squares(ax, A["a2"], color=C_GREEN, ms=2.2, zorder=4)
    ax.add_patch(patches.Rectangle((x0 - 0.5, y0 - 0.5), n, n, fc="none", ec=C_INK, lw=0.8, zorder=5))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    nb = 5.0 / VOX_MM
    xb, yb = Ws * 0.03, Hs * 0.93
    ax.plot([xb, xb + nb], [yb, yb], color=C_INK, lw=1.2, solid_capstyle="butt")
    ax.text(xb + nb / 2, yb - Hs * 0.035, "5 mm", ha="center", va="bottom", fontsize=6.5)
    tA = ftext(fig, xA, ax_y1 + hA + 1.4,
               f"{fmt(A['mask_px'])} mask pixels → {fmt(A['ipl_px'])} rendered ({A['dropped']} dropped, {A['added']} added).\n"
               f"ipldt's rendering = IPL's: {A['ndiff_slice']} differing pixels on this slice and\n"
               f"{A['vol_mismatch']} on the whole volume ({fmt(A['vol_ipl'])} rendered voxels).",
               fontsize=6.5, color=C_INK2, ha="left", linespacing=1.2)
    checks.append((tA, xA, xA + wA + 6, "A note"))
    N["A"] = dict(subject=A["subj"], mask=A["kind"], slice_local=A["z"], slice_global=A["z_global"], slices=A["slices"],
                  mask_px=A["mask_px"], rendered_px=A["ipl_px"], dropped=A["dropped"], added=A["added"],
                  ipldt_vs_ipl_slice=A["ndiff_slice"], ipldt_vs_ipl_volume=A["vol_mismatch"], crop=list(A["crop"]),
                  crop_dropped=A["crop_dropped"], crop_added=A["crop_added"],
                  chains=[(c["kind"], len(c["vertices"])) for c in A["chains"]], gobj=A["gobj"])

    # --- B: the zoom
    xB, wB = 86.0, hB
    panel_title(fig, xB, y1, "B", "The chain and its raster",
                "the stored contour is a closed polygon\nthrough pixel centers; the raster it renders\nis what every masked IPL command sees",
                checks, span=wB + 6)
    ax = ax_mm(fig, xB, ax_y1, wB, hB)
    crop = A["sl"][y0:y0 + n, x0:x0 + n]
    show_mask(ax, crop, x0, y0)
    pixel_grid(ax, x0, y0, n, n)
    squares(ax, A["d2"][y0:y0 + n, x0:x0 + n], x0, y0, color=C_VERM, ms=4.0)
    squares(ax, A["a2"][y0:y0 + n, x0:x0 + n], x0, y0, color=C_GREEN, ms=4.0)
    raster_outline(ax, A["gi"][y0:y0 + n, x0:x0 + n], x0, y0, colors=C_INK, linewidths=1.3, zorder=4)
    raster_outline(ax, A["ours"][y0:y0 + n, x0:x0 + n], x0, y0, colors=C_SKY, linewidths=1.3, linestyles=(0, (2.2, 2.2)),
                   zorder=4.5)
    for c in A["chains"]:
        draw_runs(ax, c["vertices"], x0, y0, n, n, color=C_BLUE, lw=0.8, ms=1.7, zorder=6)
    grid_axes(ax, x0, y0, n, n, step=5)

    # --- C: the cohort
    xC, wC = 138.0, 40.0
    panel_title(fig, xC, y1, "C", "21 patellae, 42 volumes",
                "voxels the rendering changes, per\nmillion mask voxels; each marker one\ncortical or trabecular mask volume",
                checks, span=wC + 2)
    ax = ax_mm(fig, xC + 6.5, ax_y1, wC - 6.5, hB)
    dr = np.array([r["dropped"] / r["mask"] * 1e6 for r in rows])
    ad = np.array([r["added"] / r["mask"] * 1e6 for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    top = float(max(dr.max(), ad.max())) * 1.1
    ax.plot([0, top], [0, top], color=C_GRID, lw=0.6, zorder=1)
    for kind, mk, col, lab in (("CORT", "o", C_INK2, "cortical"), ("TRAB", "s", C_MUTED, "trabecular")):
        m = kinds == kind
        ax.plot(dr[m], ad[m], mk, ms=3.2, color=col, mec="white", mew=0.4, ls="none", label=lab, zorder=3)
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.set_xticks([0, 200, 400]); ax.set_yticks([0, 200, 400])
    ax.set_xlabel("dropped per 10⁶", labelpad=1.5)
    ax.set_ylabel("added per 10⁶", labelpad=1.5)
    ax.tick_params(length=1.5, pad=1.2)
    ax.grid(True, color=C_GRID, lw=0.3)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    mism = sum(r["mismatch"] for r in rows)
    n_exact = sum(r["mismatch"] == 0 for r in rows)
    ax.legend(loc="upper left", fontsize=6.3, frameon=False, handletextpad=0.3, borderaxespad=0.3)
    tC = ftext(fig, xC, ax_y1 + hB + 6.0,
               f"dropped {dr.min():.0f}–{dr.max():.0f}, added {ad.min():.0f}–{ad.max():.0f} per 10⁶\n"
               f"(medians {np.median(dr):.0f} and {np.median(ad):.0f}). ipldt vs IPL:\n"
               f"{mism} differing voxels in all {len(rows)} volumes.",
               fontsize=6.5, color=C_INK2, ha="left", linespacing=1.2)
    checks.append((tC, xC, xC + wC + 2, "C note"))
    N["C"] = dict(volumes=len(rows), mismatch_total=mism, volumes_identical=n_exact,
                  dropped_ppm=dict(min=float(dr.min()), max=float(dr.max()), median=float(np.median(dr))),
                  added_ppm=dict(min=float(ad.min()), max=float(ad.max()), median=float(np.median(ad))),
                  mask_voxels=sum(r["mask"] for r in rows), rendered_voxels=sum(r["ipl"] for r in rows),
                  dropped_voxels=sum(r["dropped"] for r in rows), added_voxels=sum(r["added"] for r in rows), rows=rows)
    # the legend strip of row 1 (panels A and B)
    h1 = [H_sq(C_VERM, "in the mask, not rendered (dropped)"), H_sq(C_GREEN, "rendered, not in the mask (added)"),
          H_line(C_BLUE, "stored chain, vertices at pixel centers (= IPL's)", marker="o"),
          H_line(C_INK, "IPL's rendered raster edge", lw=1.3), H_line(C_SKY, "ipldt's rendered raster edge (coincident)", ls=(0, (2.2, 2.2)), lw=1.3)]
    legend_strip(fig, 2.0, ax_y1 + hB + 5.0, 130.0, 9.0, h1, ncol=2)

    # ===================================================================================== row 2: the synthetic rule
    y2 = 69.0
    ax_y2 = y2 + 13.0
    sl, comp = S["sl"], S["comp"]
    win = (1, 3, 22, 16)                                  # (x0, y0, w, h) of the synthetic slice shown
    ph2 = pw * win[3] / win[2]

    def base(ax, img, grey=MASK_GREY):
        show_mask(ax, img, grey=grey)
        pixel_grid(ax, 0, 0, img.shape[1], img.shape[0], color="#d8d8d8", lw=0.35)
        grid_axes(ax, *win, step=4)

    # --- D: phase 1 + trace
    raw = S["raw"]
    panel_title(fig, xcol[0], y2, "D", "Pre-trace rule, Moore trace (synthetic)",
                "a pixel with background W and E is dropped unless\nit is the raster-first pixel or has mask directly\n"
                "N and S; then a counterclockwise Moore trace", checks, span=pw)
    ax = ax_mm(fig, xcol[0], ax_y2, pw, ph2)
    base(ax, sl)
    squares(ax, S["dropped"], color=C_VERM, ms=6.4, zorder=3)
    ax.plot(S["start"][0], S["start"][1], "s", color=C_ORANGE, ms=6.4, mec=C_INK, mew=0.6, zorder=5)
    squares(ax, S["neck"], color=C_PURPLE, ms=6.4, zorder=3)
    chain_line(ax, raw, color=C_BLUE, lw=0.9, ms=1.9)
    ax.annotate("", xy=raw[2], xytext=raw[0], arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.6, shrinkA=3, shrinkB=0), zorder=8)
    for k in range(0, len(raw), 10):
        ax.annotate(str(k), raw[k], xytext=(2.5, 2.5), textcoords="offset points", fontsize=5.5, color=C_BLUE, zorder=9)
    ax.text(0.02, 0.03, f"{int(sl.sum())} mask px, {int(S['dropped'].sum())} dropped; raw chain {len(raw)} vertices", transform=ax.transAxes,
            fontsize=6.2, color=C_INK2, ha="left", va="bottom")
    N["D"] = dict(mask_px=int(sl.sum()), dropped=int(S["dropped"].sum()), pre_activated=bool(S["pre"]), raw_vertices=len(raw),
                  start=list(S["start"]))

    # --- E: stage A
    a_ev = S["a_ev"]
    panel_title(fig, xcol[1], y2, "E", "Curvature smoothing, stage A (synthetic)",
                "one in-place sweep, start vertex last, of four\nrules on d = p[i+1] − p[i−1], neighbors as already\n"
                f"updated; here {len(a_ev)} vertices change: the gate opens", checks, span=pw)
    ax = ax_mm(fig, xcol[1], ax_y2, pw, ph2)
    base(ax, comp)
    chain_line(ax, raw, color=RAW_GREY, lw=0.8, ms=0)
    chain_line(ax, S["a_pts"], color=C_BLUE, lw=0.9, ms=1.8)
    draw_events(ax, a_ev)
    ax.plot(S["a_pts"][0][0], S["a_pts"][0][1], "o", color=C_ORANGE, ms=6, mec=C_INK, mew=0.6, zorder=7)
    ax.text(0.02, 0.03, f"{len(raw)} → {len(S['a_pts'])} vertices; the raw start is deleted,\nits predecessor becomes the start",
            transform=ax.transAxes, fontsize=6.2, color=C_INK2, ha="left", va="bottom", linespacing=1.15)
    N["E"] = dict(events=[(e[0], list(e[1])) for e in a_ev], vertices_after=len(S["a_pts"]), start_after=list(S["a_pts"][0]))

    # --- F: gate + stage B
    b_ev, final = S["b_ev"], S["final"]
    panel_title(fig, xcol[2], y2, "F", "The gate and stage B (synthetic)",
                "stage B needs a stage-A change or a pre-trace drop,\nelse the raw chain is stored; it sweeps all five\n"
                f"rules until nothing changes ({S['nsweeps']} sweeps here)", checks, span=pw)
    ax = ax_mm(fig, xcol[2], ax_y2, pw, ph2)
    base(ax, comp)
    chain_line(ax, S["a_pts"], color=RAW_GREY, lw=0.8, ms=0)
    chain_line(ax, final, color=C_BLUE, lw=0.9, ms=1.8)
    draw_events(ax, b_ev)
    ax.plot(final[0][0], final[0][1], "o", color=C_ORANGE, ms=6, mec=C_INK, mew=0.6, zorder=7)
    ax.text(0.02, 0.03, f"{len(S['a_pts'])} → {len(final)} vertices: the stored chain\n(= IPL's, -curvature_smooth 1)",
            transform=ax.transAxes, fontsize=6.2, color=C_INK2, ha="left", va="bottom", linespacing=1.15)
    N["F"] = dict(events=[(e[0], list(e[1])) for e in b_ev], sweeps=S["nsweeps"], vertices_stored=len(final), start=list(final[0]))
    # the legend strip of row 2
    h2 = [H_line(RAW_GREY, "chain entering the stage", lw=0.8), H_line(C_BLUE, "chain leaving the stage", marker="o"),
          H_circ(C_ORANGE, "start vertex p[0]"),
          H_sq(C_VERM, "run of 1: dropped (activates stage B)"), H_sq(C_ORANGE, "run of 1, raster-first pixel: kept", mec=C_INK, mew=0.6),
          H_sq(C_PURPLE, "run of 1 with mask N and S: kept")] + event_handles(SM.ALL_RULES)
    legend_strip(fig, 2.0, ax_y2 + ph2 + 4.2, 176.0, 12.0, h2, ncol=3)

    # ===================================================================================== row 3
    y3 = 141.0
    ax_y3 = y3 + 13.0
    # --- G: even-odd fill (synthetic)
    ys_row = 10
    xs = np.array([p[0] for p in final]); ys = np.array([p[1] for p in final])
    x1, yy1 = np.roll(xs, -1), np.roll(ys, -1)
    dyy = yy1 - ys
    cxs = np.where(dyy > 0, xs, x1)[dyy != 0]
    cys = np.where(dyy > 0, ys, yy1)[dyy != 0]
    cr = sorted(cxs[cys == ys_row].tolist())
    G = S["G"]
    panel_title(fig, xcol[0], y3, "G", "Rasterization of a contour (synthetic)",
                "rendered = chain pixels + centers strictly inside\nthe polygon; an edge crosses the scanline of its\n"
                "lower end; inside = odd crossings to its left", checks, span=pw)
    ax = ax_mm(fig, xcol[0], ax_y3, pw, ph2)
    base(ax, sl)
    squares(ax, S["interior"], color=C_SKY, ms=6.4, alpha=0.9, zorder=2.5)
    squares(ax, S["chain_px"], color=C_BLUE, ms=6.4, alpha=0.9, zorder=2.6)
    squares(ax, sl & ~G, color=C_VERM, ms=3.4, zorder=6)
    squares(ax, ~sl & G, color=C_GREEN, ms=3.4, zorder=6)
    chain_line(ax, final, color=C_INK, lw=0.6, ms=0, zorder=5)
    ax.axhline(ys_row, color=C_ORANGE, lw=1.0, ls="--", zorder=6)
    for x in cr:
        ax.plot(x, ys_row, "|", color=C_ORANGE, ms=9, mew=1.6, zorder=9)
    ax.text(0.02, 0.03, f"scanline y = {ys_row}: crossings at x = {cr[0]} and {cr[1]};\n{int(G.sum())} px rendered of {int(sl.sum())} "
            f"({int((sl & ~G).sum())} dropped, {int((~sl & G).sum())} added)", transform=ax.transAxes, fontsize=6.2, color=C_INK2,
            ha="left", va="bottom", linespacing=1.15)
    N["G"] = dict(rendered_px=int(G.sum()), mask_px=int(sl.sum()), dropped=int((sl & ~G).sum()), added=int((~sl & G).sum()),
                  scanline=ys_row, crossings=cr)

    # --- H: the inner contour on the real finger slice
    x0, y0, w, h = F["crop"]
    ncr_left = sum(1 for x in F["cross"] if x < min(np.nonzero(F["lobe"][F["row"]])[0]))
    panel_title(fig, xcol[1], y3, "H", "An inner contour on a real slice",
                "a hole's chain keeps its pixels; only its strict\ninterior is removed; the finger it wraps through\n"
                f"a 1-px neck has {ncr_left} crossings to its left: kept", checks, span=pw)
    ax = ax_mm(fig, xcol[1], ax_y3, pw, pw * h / w)
    show_mask(ax, F["sl"][y0:y0 + h, x0:x0 + w], x0, y0)
    pixel_grid(ax, x0, y0, w, h)
    raster_outline(ax, F["gi"][y0:y0 + h, x0:x0 + w], x0, y0, colors=C_INK, linewidths=1.2, zorder=4)
    squares(ax, F["iint"][y0:y0 + h, x0:x0 + w], x0, y0, color=C_SKY, ms=4.2, alpha=0.6, zorder=2.5)
    squares(ax, F["lobe"][y0:y0 + h, x0:x0 + w], x0, y0, color=C_YELLOW, ms=4.2, zorder=3)
    squares(ax, (F["sl"] & ~F["G"])[y0:y0 + h, x0:x0 + w], x0, y0, color=C_VERM, ms=2.6, zorder=6)
    squares(ax, (~F["sl"] & F["G"])[y0:y0 + h, x0:x0 + w], x0, y0, color=C_GREEN, ms=2.6, zorder=6)
    draw_runs(ax, F["inner"], x0, y0, w, h, color=C_BLUE, lw=0.8, ms=1.5)
    ax.axhline(F["row"], color=C_ORANGE, lw=0.9, ls="--", zorder=6)
    for x in F["cross"]:
        if x0 - 1 <= x < x0 + w + 1:
            ax.plot(x, F["row"], "|", color=C_ORANGE, ms=8, mew=1.5, zorder=9)
    grid_axes(ax, x0, y0, w, h, step=5)
    ax.text(0.02, 0.97, f"cortical mask, slice {F['z'] + 1}: inner chain {fmt(F['n_inner'])} vertices;\nfinger {F['lobe_px']} px; "
            f"filling by connectivity would differ by {F['alt_diff']} px", transform=ax.transAxes, fontsize=6.2, color=C_INK2,
            ha="left", va="top", linespacing=1.15, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    N["H"] = dict(subject=F["subj"], mask=F["kind"], slice_local=F["z"], slice_global=F["z_global"], inner_vertices=F["n_inner"],
                  lobe_px=F["lobe_px"], fill_by_connectivity_vs_ipl=F["alt_diff"], ipldt_vs_ipl=F["ndiff"], crop=list(F["crop"]),
                  scanline=F["row"], crossings=F["cross"], crossings_left_of_finger=ncr_left, gobj=F["gobj"])

    # --- I: the gate on a real slice (stored raw)
    x0, y0, w, h = I["crop"]
    panel_title(fig, xcol[2], y3, "I", "The gate on a real slice: stored raw",
                f"trabecular mask, slice {I['z'] + 1}: {len(I['exc'])} one-pixel notches in\nhorizontal runs, no other event; "
                "stage A changes\nnothing: the raw chain is stored; raster = mask", checks, span=pw)
    ax = ax_mm(fig, xcol[2], ax_y3, pw, pw * h / w)
    show_mask(ax, I["sl"][y0:y0 + h, x0:x0 + w], x0, y0)
    pixel_grid(ax, x0, y0, w, h)
    raster_outline(ax, I["gi"][y0:y0 + h, x0:x0 + w], x0, y0, colors=C_INK, linewidths=1.2, zorder=4)
    squares(ax, (I["unc_raster"] != I["gi"])[y0:y0 + h, x0:x0 + w], x0, y0, color=C_PURPLE, ms=4.0, zorder=3)
    draw_runs(ax, I["raw"], x0, y0, w, h, color=C_BLUE, lw=0.8, ms=1.5)
    for (hx, hy), (mx, my) in I["exc"]:
        if x0 <= hx < x0 + w and y0 <= hy < y0 + h:
            ax.plot(hx, hy, "o", color=C_VERM, ms=5.5, mfc="none", mew=1.3, zorder=8)
            ax.annotate("", xy=(mx, my), xytext=(hx, hy), arrowprops=dict(arrowstyle="->", color=C_VERM, lw=1.0, ls="--",
                                                                          shrinkA=0, shrinkB=1.5), zorder=8)
    ax.plot(I["raw"][0][0], I["raw"][0][1], "o", color=C_ORANGE, ms=6, mec=C_INK, mew=0.6, zorder=9)
    grid_axes(ax, x0, y0, w, h, step=5)
    ax.text(0.02, 0.03, f"stored chain = raw trace, {fmt(I['n'])} vertices (= IPL's);\nunconditional smoothing would differ from IPL "
            f"on {I['unc_diff']} px", transform=ax.transAxes, fontsize=6.2, color=C_INK2, ha="left", va="bottom", linespacing=1.15,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    N["I"] = dict(subject=I["subj"], mask=I["kind"], slice_local=I["z"], slice_global=I["z_global"], vertices=I["n"], notches=len(I["exc"]),
                  unconditional_smoothing_vs_ipl=I["unc_diff"], ipldt_vs_ipl=I["ndiff"], mask_px=I["mask_px"], rendered_px=I["ipl_px"],
                  crop=list(I["crop"]), gobj=I["gobj"])
    # the legend strip of row 3
    h3 = [H_line(C_BLUE, "stored chain (= IPL's)", marker="o"), H_line(C_INK, "IPL's rendered raster edge (= ipldt's)", lw=1.2),
          H_circ(C_ORANGE, "trace start (= IPL's stored start)"),
          H_sq(C_BLUE, "chain pixels: rendered"), H_sq(C_SKY, "strict interior of the polygon (odd crossings to the left)"),
          H_sq(C_YELLOW, "enclosed by chain pixels, outside the polygon: kept"),
          H_sq(C_VERM, "mask pixel not rendered", ms=3.5), H_sq(C_GREEN, "rendered pixel not in the mask", ms=3.5),
          H_line(C_ORANGE, "scanline and its chain crossings", ls="--", marker="|", ms=7),
          Line2D([], [], marker="o", mfc="none", mec=C_VERM, color=C_VERM, ms=4.5, mew=1.3, ls="--", lw=1.0,
                 label="notch, |d| = (2, 0): its stage-B move is never reached"),
          H_sq(C_PURPLE, "where unconditional smoothing would differ from IPL", ms=4)]
    ph3 = max(ph2, pw * F["crop"][3] / F["crop"][2], pw * I["crop"][3] / I["crop"][2])
    legend_strip(fig, 2.0, ax_y3 + ph3 + 4.2, 176.0, 12.0, h3, ncol=3)

    # ===================================================================================== row 4: the minimum-vertex rule
    y4 = 214.0
    xJ = 2.0
    panel_title(fig, xJ, y4, "J", "The minimum-vertex rule on the tiny-object phantom evaluated by IPL",
                "a chain left with fewer than 4 vertices is not stored: no contour, nothing rendered (the sweep has no fixed "
                "point below 8 vertices,\nso a chain still shrinking at 3 is collapsing); a raw chain the gate leaves untouched has at least 4 "
                "vertices and is therefore always stored.\nThresholds 4–6 render identically; 7 or 8 would also drop the 6-vertex raw chain "
                "of the vertical 1 × 2 hole and fill the hole that IPL keeps.\nGray = phantom pixels; "
                "thin gray = raw trace; blue = stored chain (ipldt's = IPL's); black = IPL's rendering.", checks, span=176.0)
    ax_y4 = y4 + 16.5
    nJ = len(PH_SHOW)
    mg = 3.0
    mw = (176.0 - (nJ - 1) * mg) / nJ
    tests = PH["tests"]
    for k, (name, cap) in enumerate(PH_SHOW):
        t = tests[name]
        x0, y0, w, h = t["crop"]
        xk = xJ + k * (mw + mg)
        ax = ax_mm(fig, xk, ax_y4, mw, mw * h / w)
        show_mask(ax, t["mask"], x0, y0)
        pixel_grid(ax, x0, y0, w, h, color="#d8d8d8", lw=0.35)
        raster_outline(ax, t["ipl"], x0, y0, colors=C_INK, linewidths=1.1, zorder=4)
        draw_runs(ax, t["raw"], x0, y0, w, h, color=RAW_GREY, lw=0.6, ms=0, zorder=5)
        if t["stored"]:
            draw_runs(ax, t["final"], x0, y0, w, h, color=C_BLUE, lw=0.9, ms=1.6, zorder=6)
        grid_axes(ax, x0, y0, w, h, step=4, px_labels=False)
        ax.tick_params(length=0)
        if t["kind"] == "hole":
            what = "hole kept" if t["hole_kept"] else "hole filled"
        else:
            what = f"{t['rendered_px']} of {t['mask_px']} px" if t["stored"] else "0 px"
        act = "gate closed" if not t["activated"] else f"{t['sweeps']} sweep{'s' if t['sweeps'] != 1 else ''}"
        verdict = "stored" if t["stored"] else "not stored"
        cap2 = f"{cap}\nraw {t['n_raw']} → {t['n_final']}, {act}\n{verdict}; {what}"
        tj = ftext(fig, xk, ax_y4 + mw * h / w + 1.2, cap2, fontsize=6.2, color=C_INK2, ha="left", linespacing=1.18)
        checks.append((tj, xk, xk + mw + mg, "J cap " + name))
        N["J"][name] = dict(kind=t["kind"], slice_global=t["z_global"], raw_vertices=t["n_raw"], final_vertices=t["n_final"],
                            activated=t["activated"], sweeps=t["sweeps"], stored=t["stored"], ipl_chain_vertices=t["n_ipl"],
                            rendered_px=t["rendered_px"], mask_px=t["mask_px"], hole_kept=t["hole_kept"])
    N["J"]["_phantom"] = dict(total_mask_voxels=PH["total_mask"], total_rendered_voxels=PH["total_rendered"],
                              ipldt_vs_ipl_min4=0, ipldt_vs_ipl_min8=PH["min8_diff"], stored_chains_ipl=PH["n_chains"])
    check_text_widths(fig, checks)
    return fig, N


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true", help="recompute the 42-volume cohort comparison of panel C")
    a = ap.parse_args()
    say("synthetic slice")
    S = synthetic_chain_data()
    say(f"  raw {len(S['raw'])} -> stage A {len(S['a_pts'])} -> stored {len(S['final'])} vertices, {S['nsweeps']} stage-B sweeps, "
        f"rendered {int(S['G'].sum())} of {int(S['sl'].sum())} px")
    say("panel A/B: the real slice")
    A = slice_overlay_data()
    say(f"  {A['subj']} {A['kind']} z={A['z']}: mask {A['mask_px']:,d} IPL {A['ipl_px']:,d} dropped {A['dropped']} added {A['added']} "
        f"ipldt-vs-IPL {A['ndiff_slice']}; crop {A['crop']}; chains {[(c['kind'], len(c['vertices'])) for c in A['chains']]}")
    say("panel C: the cohort")
    rows = cohort_rows(a.recompute)
    assert len(rows) == 42
    vol = [r for r in rows if r["subj"] == A["subj"] and r["kind"] == A["kind"]][0]
    A["vol_mismatch"], A["vol_ipl"] = vol["mismatch"], vol["ipl"]
    say("panel H: the finger slice")
    F = finger_data()
    say(f"  {F['subj']} z={F['z_global']}: inner {F['n_inner']} vertices, lobe {F['lobe_px']} px, scanline {F['row']} crossings {F['cross']}, "
        f"fill-by-connectivity vs IPL {F['alt_diff']}, ipldt vs IPL {F['ndiff']}")
    say("panel I: the stored-raw slice")
    I = stored_raw_data()
    say(f"  {I['subj']} z={I['z']}: {I['n']} vertices, {len(I['exc'])} notches, unconditional smoothing vs IPL {I['unc_diff']}, ipldt vs IPL {I['ndiff']}")
    say("panel J: the phantom")
    PH = phantom_data()
    say(f"  phantom rendered {PH['total_rendered']} of {PH['total_mask']} voxels; MIN 8 would differ on {PH['min8_diff']}")
    fig, N = draw(S, A, rows, F, I, PH)
    fig.savefig(OUT + ".png", dpi=300, facecolor="white")
    fig.savefig(OUT + ".svg", facecolor="white")
    plt.close(fig)
    say("wrote", OUT + ".png", "and .svg")
    N["figure"] = dict(width_mm=FIG_W, height_mm=FIG_H, dpi=300, vox_mm=VOX_MM)
    N["sources"] = dict(masks=public_path(GRAB), ipl_renderings_and_contours=public_path(P15), phantom=public_path(P19),
                        cohort_cache=public_path(CACHE))
    json.dump(N, open(OUT + "_numbers.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    say("sidecar", OUT + "_numbers.json")


if __name__ == "__main__":
    main()
