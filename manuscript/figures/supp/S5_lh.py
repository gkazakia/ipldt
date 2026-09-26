"""S5_lh.py -- Supplementary Figure S5 of the ipldt / ORMIR-BQRL manuscript: the Laplace-Hamming segmentation,
one panel per parameter or rule, every panel on real data except the padding phantom (D).

Panels (letters A-F in the upper-left corner)
  A  laplace_eps (Laplacian weight epsilon): the transfer function H(|k|) for epsilon = 0 / 0.45 (IPL) / 0.9 and the
     thresholded response of one slice of a real scan under each value.
  B  lp_cut_off_freq (radial cut-off, in cycles per voxel along z): 0.2 / 0.3 (IPL) / 0.4, transfer function + slice.
  C  hamming_amp (window amplitude): 0 (rectangular) / 0.5 / 1 (Hann, IPL), transfer function + slice.
  D  the power-of-two padding: IPL's own filter output on a 63-voxel impulse phantom (one padding voxel per axis)
     against the reimplementation with the padding voxel before the data (the rule adopted) and after it.
  E  norm_max and the threshold: the distribution of the float filter output of the real scan, the scale to int16
     (x 32767 / 200000, truncated, clipped at +-200 000) and the lower threshold 475 permille = 15 564.
  F  the component filters: 6-connected components of the masked threshold set smaller than 35 voxels (removed from
     both compartments) and of 35-69 voxels (removed from the trabecular compartment only); the SEG that results.

Run from the repository root in the `ormir` environment:

    set PYTHONUTF8=1
    python manuscript/figures/supp/S5_lh.py              # draw (cached data)
    python manuscript/figures/supp/S5_lh.py --recompute  # recompute everything

Writes manuscript/figures/supp/S5_lh.png (300 dpi, 180 mm wide), S5_lh.svg and S5_lh_numbers.json (every number
drawn or quoted).  The parameter variants are computed here with the engine's own arithmetic (float32 transfer
function, scipy FFT, IPL's padding plan) with epsilon / cut-off / amplitude as arguments; at IPL's values the
result is asserted bit-identical to ipldt.ormir.lh_filter_core, and its threshold decisions are counted against
IPL's exported thresholded volume of the same scan.  Reads only the greyscale AIM, IPL's compartment rasters,
contour renderings and exported STEP-2 intermediates of the displayed scan, and IPL's exports of the impulse
phantom.  Runtime about 1 min (one forward FFT of 1024 x 512 x 256 and seven inverse ones); the intermediate
results are cached in manuscript/figures/cache/S5_lh_cache.npz.  Deterministic.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.fft import fftn, ifftn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
from ipldt.io import read_aim  # noqa: E402
from ipldt import ormir  # noqa: E402
from ipldt import ipl_ops as ops  # noqa: E402

# ------------------------------------------------------------------------------------------------ inputs
DATA = lab_path("patellae/PFJ-0be66a_R")          # the displayed scan (a patella)
BASE = "X2420448"
IPL_EXPORTS = lab_path("ipl_test_runs/run15/aims_and_logs")   # IPL's STEP-2 exports
PHANTOM_DIR = lab_path("ipl_test_runs/run21")                # the impulse phantom
PHANTOM = "x63i01"                                                                                # 63 voxels along x, impulse at x = 1
CACHE_DIR = os.path.join(REPO, "manuscript", "figures", "cache")
CACHE_NPZ = os.path.join(CACHE_DIR, "S5_lh_cache.npz")
CACHE_JSON = os.path.join(CACHE_DIR, "S5_lh_cache.json")
OUT_PNG = os.path.join(HERE, "S5_lh.png")
OUT_SVG = os.path.join(HERE, "S5_lh.svg")
OUT_NUM = os.path.join(HERE, "S5_lh_numbers.json")

# IPL parameters as ipldt carries them
EPS = ormir.LAPLACE_EPS               # 0.45
LP = ormir.LP_CUT_OFF_FREQ            # 0.3
AMP = ormir.HAMMING_AMP               # 1.0 -> Hann
NMAX = ormir.NORM_MAX_VALUE           # 200000
THR = ormir.LH_THRESHOLD              # 15564 = 475 permille of 32767
I16 = ormir.INT16_MAX                 # 32767
CC_CORT, CC_TRAB = ormir.CC_MIN_VOXELS_CORT, ormir.CC_MIN_VOXELS_TRAB   # 35, 70
SEG_CORT, SEG_TRAB = ormir.SEG_VALUE_CORT, ormir.SEG_VALUE_TRAB         # 127, 126
F_THR = THR * NMAX / I16              # the threshold in filter units (94 997.8)

# the variants: (key, eps, cut-off, amp)
VARIANTS = [("std", EPS, LP, AMP),
            ("eps0", 0.0, LP, AMP), ("eps09", 0.9, LP, AMP),
            ("lp02", EPS, 0.2, AMP), ("lp04", EPS, 0.4, AMP),
            ("amp0", EPS, LP, 0.0), ("amp05", EPS, LP, 0.5)]
CROP = 150                            # voxels (9.1 mm) shown in every tile

T0 = time.time()
NUM = {}


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def fmt_int(n):
    """Thousands separated by a thin space, journal style."""
    return f"{int(n):,d}"


# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito, colour-blind safe
C_BLUE, C_ORANGE, C_VERM, C_GREEN, C_SKY, C_PURPLE = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9", "#CC79A7"
C_CORT, C_TRAB, C_SET = "#3A3A3A", "#A6A6A6", "#3A3A3A"
INK, INK2, GRID = "#1A1A1A", "#4D4D4D", "#D9D9D9"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "savefig.dpi": 300, "figure.dpi": 100, "svg.fonttype": "none", "pdf.fonttype": 42,
})
FIG_W_MM, FIG_H_MM = 180.0, 247.0


def mm_axes(fig, x, y, w, h, **kw):
    """An axes placed in millimetres from the top-left corner of the figure."""
    return fig.add_axes([x / FIG_W_MM, 1.0 - (y + h) / FIG_H_MM, w / FIG_W_MM, h / FIG_H_MM], **kw)


def mm_text(fig, x, y, s, **kw):
    kw.setdefault("fontsize", 7.2)
    kw.setdefault("ha", "left")
    kw.setdefault("va", "top")
    kw.setdefault("color", INK)
    return fig.text(x / FIG_W_MM, 1.0 - y / FIG_H_MM, s, **kw)


def panel_letter(fig, x, y, s):
    fig.text(x / FIG_W_MM, 1.0 - y / FIG_H_MM, s, fontsize=9, fontweight="bold", ha="left", va="top", color=INK)


# ================================================================================================= the filter
def transfer(shape, el, eps, lp, amp):
    """H(k) of /fft_laplace_hamming on the padded box `shape` (array order (z, y, x)), float32, with the three
    parameters as arguments: (2 pi)^2 [(1 - eps) + eps |k|^2] x W(|k|), W = (1 - amp/2) + (amp/2) cos(pi |k| / k_c)
    below the radial cut-off k_c = lp / el_z and 0 above it.  Operation for operation the engine's arithmetic."""
    f32 = np.float32
    nz, ny, nx = shape
    ex, ey, ez = (f32(e) for e in el)
    kx = np.fft.fftfreq(nx, d=float(ex)).astype(f32)
    ky = np.fft.fftfreq(ny, d=float(ey)).astype(f32)
    kz = np.fft.fftfreq(nz, d=float(ez)).astype(f32)
    K2 = (kx * kx)[None, None, :] + (ky * ky)[None, :, None]
    K2 = K2 + (kz * kz)[:, None, None]
    Kmag = np.sqrt(K2)
    k_lp = f32(lp) / ez
    half_amp = f32(amp) * f32(0.5)
    win = np.where(Kmag < k_lp, (f32(1.0) - half_amp) + half_amp * np.cos(np.pi * Kmag / k_lp), f32(0.0)).astype(f32)
    del Kmag
    H = f32((2.0 * np.pi) ** 2) * ((f32(1.0) - f32(eps)) + f32(eps) * K2) * win
    return H


def transfer_radial(k, el_z, eps, lp, amp):
    """The same H as a function of |k| (float64, for the curves)."""
    k_lp = lp / el_z
    win = np.where(k < k_lp, (1.0 - 0.5 * amp) + 0.5 * amp * np.cos(np.pi * k / k_lp), 0.0)
    return (2.0 * np.pi) ** 2 * ((1.0 - eps) + eps * k * k) * win


def to_short(lh):
    """IPL /norm_max -max 200000 -type_out short: float32 product, clipped, truncated toward zero."""
    scaled = lh * (np.float32(I16) / np.float32(NMAX))
    np.clip(scaled, -I16, I16, out=scaled)
    return np.trunc(scaled).astype(np.int16)


# ================================================================================================= compute
def choose_crop(seg0, comp_size, cort, trab, gdim):
    """The CROP x CROP window of a slice near the middle of the stack that contains the most distinct components
    smaller than 70 voxels while holding cortex and trabecular region (deterministic grid search)."""
    zc = gdim[2] // 2
    h = CROP // 2
    best = None
    for zz in range(zc - 30, zc + 31, 5):
        sl = seg0[zz]
        small = sl & (comp_size[zz] < CC_TRAB)
        if not small.any():
            continue
        for cx in range(h, gdim[0] - h, 10):
            for cy in range(h, gdim[1] - h, 10):
                win = (slice(cy - h, cy + h), slice(cx - h, cx + h))
                nc, nt = int(cort[zz][win].sum()), int(trab[zz][win].sum())
                if nc < 800 or nt < 6000:
                    continue
                score = int(len(np.unique(comp_size[zz][win][small[win]])))
                if best is None or score > best[0]:
                    best = (score, zz, cx, cy)
    _, zz, cx, cy = best
    return zz, cx - h, cy - h


def compute():
    say("reading the greyscale of the displayed scan")
    grey = read_aim(os.path.join(DATA, f"{BASE}.AIM"))
    native = grey["data"]
    el = tuple(float(e) for e in grey["el_size_mm"])
    gdim, gpos = tuple(int(v) for v in grey["dim"]), tuple(int(v) for v in grey["pos"])
    NUM["scan"] = dict(dim=gdim, pos=gpos, el_size_mm=el, voxels=int(np.prod(gdim)))

    say("one forward FFT of the border-duplicated, mirror-padded volume")
    extended = np.pad(native, ((1, 1), (1, 1), (1, 1)), mode="edge").astype(np.float32)   # Script 32's 1-voxel border
    pad_widths, inner = ormir.lh_pad_plan(extended.shape)                                  # IPL's power-of-two plan
    padded = np.pad(extended, pad_widths, mode="reflect")
    NUM["padding"] = dict(extended_dim_xyz=[int(v) for v in extended.shape[::-1]], padded_dim_xyz=[int(v) for v in padded.shape[::-1]],
                          pad_before_after_xyz=[[int(a), int(b)] for a, b in pad_widths[::-1]], pad_offset=ormir.LH_PAD_OFFSET)
    F = fftn(padded, workers=-1)
    del padded

    bms, shorts_ext = {}, {}
    lh_std = None
    for key, eps, lp, amp in VARIANTS:
        say(f"variant {key}: eps {eps}, cut-off {lp}, amp {amp}")
        H = transfer(F.shape, el, eps, lp, amp)
        G = F * H
        del H
        lh = np.real(ifftn(G, workers=-1, overwrite_x=True)).astype(np.float32)
        del G
        lh = np.ascontiguousarray(lh[inner])
        short = to_short(lh)
        bm_ext = (short >= THR) & (short <= I16)
        bms[key] = np.ascontiguousarray(bm_ext[1:-1, 1:-1, 1:-1])
        if key == "std":
            lh_std, short_std_ext = lh, short
        del lh, short, bm_ext
    del F

    say("self-check: the standard variant against the engine and against IPL's exported thresholded volume")
    lh_e, sh_e = ormir.lh_filter_core(extended, el)
    NUM["engine_check"] = dict(float_differs=int((lh_e != lh_std).sum()), short_differs=int((sh_e != short_std_ext).sum()))
    assert NUM["engine_check"]["float_differs"] == 0 and NUM["engine_check"]["short_differs"] == 0, NUM["engine_check"]
    del lh_e, sh_e, extended
    short_std = np.ascontiguousarray(short_std_ext[1:-1, 1:-1, 1:-1])
    del short_std_ext
    lh_grey = np.ascontiguousarray(lh_std[1:-1, 1:-1, 1:-1])
    del lh_std
    thr_ipl = ops.on_grid(read_aim(os.path.join(IPL_EXPORTS, f"{BASE}_T15_LHSEG.AIM")), gdim, gpos) != 0
    short_ipl = ops.on_grid(read_aim(os.path.join(IPL_EXPORTS, f"{BASE}_T15_LHNORM.AIM")), gdim, gpos)
    NUM["vs_ipl"] = dict(threshold_decisions_differ=int((bms["std"] != thr_ipl).sum()), voxels=int(np.prod(gdim)),
                         threshold_voxels_ipl=int(thr_ipl.sum()), short_differs=int((short_std != short_ipl).sum()),
                         short_abs_diff_max=int(np.abs(short_std.astype(np.int32) - short_ipl.astype(np.int32)).max()))
    say(f"  engine: 0 differ; IPL: threshold decisions differ on {NUM['vs_ipl']['threshold_decisions_differ']} of "
        f"{NUM['vs_ipl']['voxels']:,d} voxels, the short on {NUM['vs_ipl']['short_differs']:,d} by <= {NUM['vs_ipl']['short_abs_diff_max']}")
    del thr_ipl, short_ipl

    say("histogram of the float output and the scale to int16")
    edges = np.arange(-160000.0, 420000.0 + 1, 2000.0)
    hist, _ = np.histogram(lh_grey.ravel(), bins=edges)
    n_all = lh_grey.size
    NUM["norm"] = dict(float_min=float(lh_grey.min()), float_max=float(lh_grey.max()), threshold_float_units=F_THR,
                       scale_units_per_level=NMAX / I16, ge_threshold=int(bms["std"].sum()), ge_threshold_pct=100.0 * bms["std"].sum() / n_all,
                       above_clip=int((lh_grey > NMAX).sum()), above_clip_pct=100.0 * (lh_grey > NMAX).sum() / n_all,
                       below_neg_clip=int((lh_grey < -NMAX).sum()), at_short_max=int((short_std == I16).sum()),
                       hist_edges=edges.tolist(), hist_counts=hist.tolist())
    say(f"  float {NUM['norm']['float_min']:.1f} .. {NUM['norm']['float_max']:.1f}; >= threshold {NUM['norm']['ge_threshold_pct']:.2f} %; "
        f"above +200000 {NUM['norm']['above_clip']:,d} ({NUM['norm']['above_clip_pct']:.3f} %)")
    del lh_grey

    say("SEG assembly: periosteal raster, components, compartments")
    cort_raw = read_aim(os.path.join(DATA, f"{BASE}_CORT_MASK.AIM"))
    trab_raw = read_aim(os.path.join(DATA, f"{BASE}_TRAB_MASK.AIM"))
    cort = ops.on_grid(cort_raw, gdim, gpos) != 0
    trab = ops.on_grid(trab_raw, gdim, gpos) != 0
    prx = cort | trab
    g_cort = ops.on_grid(read_aim(os.path.join(IPL_EXPORTS, f"{BASE}_T15_CORT_G2A.AIM")), gdim, gpos) != 0
    g_trab = ops.on_grid(read_aim(os.path.join(IPL_EXPORTS, f"{BASE}_T15_TRAB_G2A.AIM")), gdim, gpos) != 0
    bm = bms["std"]
    seg0 = bm & prx
    lab, ncomp, sizes = ops.label6(seg0)
    s = sizes[1:]
    comp_size = sizes[lab].astype(np.int32)
    del lab
    cort_seg, trab_seg = ormir.ipl_seg_assembly(bm, prx, g_cort, g_trab)
    seg = np.zeros(bm.shape, np.uint8)
    seg[trab_seg] = SEG_TRAB
    seg[cort_seg] = SEG_CORT
    seg_ipl = ops.on_grid(read_aim(os.path.join(DATA, f"{BASE}_SEG.AIM")), gdim, gpos)
    NUM["components"] = dict(threshold_voxels=int(bm.sum()), periosteal_voxels=int(prx.sum()), masked_threshold_voxels=int(seg0.sum()),
                             removed_by_periosteal=int((bm & ~prx).sum()), n_components=int(ncomp), largest=int(s.max()),
                             five_largest=[int(v) for v in np.sort(s)[::-1][:5]],
                             n_lt_35=int((s < CC_CORT).sum()), voxels_lt_35=int(s[s < CC_CORT].sum()),
                             n_35_69=int(((s >= CC_CORT) & (s < CC_TRAB)).sum()), voxels_35_69=int(s[(s >= CC_CORT) & (s < CC_TRAB)].sum()),
                             n_ge_70=int((s >= CC_TRAB).sum()),
                             cort_seg=int(cort_seg.sum()), trab_seg=int(trab_seg.sum()), seg=int((seg != 0).sum()),
                             seg_ipl=int((seg_ipl != 0).sum()), seg_differs_from_ipl=int(((seg != 0) != (seg_ipl != 0)).sum()),
                             sizes=[int(v) for v in s])
    c = NUM["components"]
    say(f"  masked threshold {c['masked_threshold_voxels']:,d}; {c['n_components']:,d} components: {c['n_lt_35']:,d} < 35 "
        f"({c['voxels_lt_35']:,d} voxels), {c['n_35_69']} in 35-69 ({c['voxels_35_69']:,d}), {c['n_ge_70']} >= 70, largest {c['largest']:,d}; "
        f"SEG {c['seg']:,d} (IPL {c['seg_ipl']:,d}, {c['seg_differs_from_ipl']} differ)")

    say("the displayed window")
    z, x0, y0 = choose_crop(seg0, comp_size, cort, trab, gdim)
    x1, y1 = x0 + CROP, y0 + CROP
    win = (z, slice(y0, y1), slice(x0, x1))
    NUM["window"] = dict(z_local=int(z), z_global=int(z + gpos[2]), x0=int(x0), y0=int(y0), size=CROP, size_mm=CROP * el[0])
    tiles = {key: bms[key][win].copy() for key in bms}
    counts = {key: int(bms[key].sum()) for key in bms}
    NUM["variants"] = {key: dict(eps=eps, lp=lp, amp=amp, threshold_voxels=counts[key],
                                 rel_to_ipl_pct=100.0 * (counts[key] - counts["std"]) / counts["std"],
                                 in_window=int(tiles[key].sum())) for key, eps, lp, amp in VARIANTS}
    for key in NUM["variants"]:
        v = NUM["variants"][key]
        say(f"  {key:6s} >= threshold {v['threshold_voxels']:,d} ({v['rel_to_ipl_pct']:+.1f} %)")
    arrays = dict(native_win=native[win].astype(np.int16), short_win=short_std[win].copy(), seg0_win=seg0[win].copy(),
                  comp_win=comp_size[win].copy(), gcort_win=g_cort[win].copy(), gtrab_win=g_trab[win].copy(),
                  cort_win=cort[win].copy(), prx_win=prx[win].copy(), seg_win=seg[win].copy(), segipl_win=seg_ipl[win].copy(),
                  sizes=s.astype(np.int64), hist_counts=hist.astype(np.int64), hist_edges=edges)
    for key in tiles:
        arrays[f"tile_{key}"] = tiles[key]
    del bms, seg0, comp_size, cort_seg, trab_seg, seg, seg_ipl, prx, cort, trab, g_cort, g_trab, short_std

    say("the impulse phantom: IPL's export against the two placements of the padding voxel")
    ph = np.load(os.path.join(PHANTOM_DIR, "prediction_volumes", f"{PHANTOM}.npz"))
    inp, el_ph = ph["input"], tuple(float(e) for e in ph["el"])
    ipl_lh = read_aim(os.path.join(PHANTOM_DIR, "aims_and_logs", f"{BASE}_T21_{PHANTOM.upper()}_LH.AIM"))
    ipl_nm = read_aim(os.path.join(PHANTOM_DIR, "aims_and_logs", f"{BASE}_T21_{PHANTOM.upper()}_NM.AIM"))
    ipl_sg = read_aim(os.path.join(PHANTOM_DIR, "aims_and_logs", f"{BASE}_T21_{PHANTOM.upper()}_SG.AIM"))
    zi, yi, xi = (int(v) for v in np.argwhere(inp != 0)[0])
    res = dict(dim_xyz=[int(v) for v in inp.shape[::-1]], impulse_xyz=[xi, yi, zi], amplitude=int(inp.max()), el_size_mm=el_ph,
               ipl_max=float(ipl_lh["data"].max()), ipl_argmax_x=int(np.argmax(ipl_lh["data"][zi, yi])),
               ipl_threshold_voxels=int((ipl_sg["data"] != 0).sum()))
    for po in ("ceil", "floor"):
        lh_p, sh_p = ormir.lh_filter_core(inp.astype(np.float32), el_ph, pad_offset=po)
        d = lh_p.astype(np.float64) - ipl_lh["data"].astype(np.float64)
        res[po] = dict(max_abs_diff=float(np.abs(d).max()), short_differs=int((sh_p != ipl_nm["data"]).sum()),
                       threshold_differs=int(((sh_p >= THR) != (ipl_sg["data"] != 0)).sum()))
        arrays[f"prof_{po}"] = lh_p[zi, yi, :].astype(np.float32)
    arrays["prof_ipl"] = ipl_lh["data"][zi, yi, :].astype(np.float32)
    arrays["ipl_slice"] = ipl_lh["data"][zi].astype(np.float32)
    NUM["phantom"] = res
    say(f"  IPL max {res['ipl_max']:.1f} at x = {res['ipl_argmax_x']}; before-the-data: max |diff| {res['ceil']['max_abs_diff']:.4f}, "
        f"threshold differs {res['ceil']['threshold_differs']}; after-the-data: max |diff| {res['floor']['max_abs_diff']:.1f}, "
        f"threshold differs {res['floor']['threshold_differs']}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(CACHE_NPZ, **arrays)
    with open(CACHE_JSON, "w") as f:
        json.dump(NUM, f, indent=1)
    say("cached")
    return arrays


def load_cache():
    global NUM
    arrays = dict(np.load(CACHE_NPZ))
    NUM = json.load(open(CACHE_JSON))
    return arrays


# ================================================================================================= draw
def curve_panel(fig, x, y, w, h, el_z, rows, ymax, cut_marks, legend_loc="upper right", legend_title=None,
                legend_box=False):
    ax = mm_axes(fig, x, y, w, h)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    nyq = 0.5 / el_z
    k = np.linspace(0.0, nyq, 3000)
    for (eps, lp, amp, col, lw, label) in rows:
        ax.plot(k, transfer_radial(k, el_z, eps, lp, amp), color=col, lw=lw, label=label, solid_capstyle="butt")
    for kc, col in cut_marks:
        ax.axvline(kc, color=col, lw=0.7, ls=(0, (3, 2)))
    ax.axvline(nyq, color=INK2, lw=0.5, ls=":")
    ax.set_xlim(0, nyq * 1.02)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("spatial frequency |k| (line pairs / mm)", labelpad=1.5)
    ax.set_ylabel("gain H(|k|)", labelpad=2)
    # legend_box: an opaque, borderless box, so the cut-off lines pass behind the entries instead of through them
    # (panels A and B, where the box sits clear of every curve)
    box = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, borderpad=0.25) if legend_box         else dict(frameon=False)
    leg = ax.legend(loc=legend_loc, handlelength=1.6, borderaxespad=0.2, labelspacing=0.3, title=legend_title,
                    title_fontsize=6.6, alignment="left", **box)
    if legend_title is not None:
        leg.get_title().set_color(INK2)
    return ax, nyq


def tile(fig, x, y, s, mask, title, sub, ipl=False, scalebar_mm=None, el=None):
    ax = mm_axes(fig, x, y, s, s)
    rgb = np.ones(mask.shape + (3,), np.float32)
    rgb[mask] = mcolors.to_rgb(C_SET)
    ax.imshow(rgb, interpolation="nearest", extent=[0, mask.shape[1], mask.shape[0], 0])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.2 if ipl else 0.5)
        sp.set_color(C_BLUE if ipl else INK2)
    mm_text(fig, x, y - 0.6, title, va="bottom", color=C_BLUE if ipl else INK, fontweight="bold" if ipl else "normal")
    mm_text(fig, x, y + s + 0.7, sub, fontsize=6.8, color=INK2, linespacing=1.15)
    if scalebar_mm is not None:
        bar = scalebar_mm / el
        ax.add_patch(Rectangle((5, mask.shape[0] - 9), bar, 2.6, fc=INK, ec="white", lw=0.6))
        ax.text(5 + bar / 2, mask.shape[0] - 11, f"{scalebar_mm:g} mm", color=INK, fontsize=6.6, ha="center", va="bottom",
                bbox=dict(fc="white", ec="none", pad=0.6))
    return ax


def draw(A):
    from scipy import ndimage as ndi
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    el = NUM["scan"]["el_size_mm"]
    el_z = el[2]
    V = NUM["variants"]

    def sub(key):
        v = V[key]
        if key == "std":
            return f"{fmt_int(v['threshold_voxels'])} voxels ≥ threshold\n(reference)"
        return f"{fmt_int(v['threshold_voxels'])} voxels ≥ threshold\n({v['rel_to_ipl_pct']:+.1f}% vs IPL)".replace("-", "−")

    # ---------------------------------------------------------------------------------------- rows A, B, C
    ROW_Y = [4.0, 48.0, 92.0]
    CX, CW, CH = 13.0, 46.0, 30.0
    TX = [66.0, 103.0, 140.0]
    TS = 33.0
    k_ipl = LP / el_z
    rows_spec = [
        ("A", "Laplacian weight ε  (IPL option laplace_eps)",
         [(0.0, LP, AMP, C_ORANGE, 1.0, "ε = 0"), (EPS, LP, AMP, C_BLUE, 1.6, "ε = 0.45 (IPL)"), (0.9, LP, AMP, C_GREEN, 1.0, "ε = 0.9")],
         150, [(k_ipl, C_VERM)], "upper right", None, [("eps0", "ε = 0"), ("std", "ε = 0.45 (IPL)"), ("eps09", "ε = 0.9")]),
        ("B", "radial cutoff  (IPL option lp_cut_off_freq)",
         [(EPS, 0.2, AMP, C_ORANGE, 1.0, "0.2"), (EPS, LP, AMP, C_BLUE, 1.6, "0.3 (IPL)"), (EPS, 0.4, AMP, C_GREEN, 1.0, "0.4")],
         200, [(0.2 / el_z, C_ORANGE), (k_ipl, C_BLUE), (0.4 / el_z, C_GREEN)], "upper right", "k_c = value / Δz (lp/mm)",
         [("lp02", "cutoff 0.2"), ("std", "cutoff 0.3 (IPL)"), ("lp04", "cutoff 0.4")]),
        ("C", "window amplitude A  (IPL option hamming_amp)",
         [(EPS, LP, 0.0, C_ORANGE, 1.0, "A = 0 (rectangular)"), (EPS, LP, 0.5, C_GREEN, 1.0, "A = 0.5"), (EPS, LP, AMP, C_BLUE, 1.6, "A = 1 (Hann, IPL)")],
         800, [(k_ipl, C_VERM)], "upper left", "W = (1 − A/2) + (A/2)·cos(π|k| / k_c)\nfor |k| < k_c, 0 above",
         [("amp0", "A = 0"), ("amp05", "A = 0.5"), ("std", "A = 1 (IPL)")]),
    ]
    for (letter, title, curves, ymax, cut_marks, legend_loc, legend_title, tiles), ry in zip(rows_spec, ROW_Y):
        panel_letter(fig, 2.0, ry, letter)
        mm_text(fig, 8.0, ry + 0.2, title, fontsize=7.5)
        ax, nyq = curve_panel(fig, CX, ry + 4.5, CW, CH, el_z, curves, ymax, cut_marks, legend_loc, legend_title,
                              legend_box=(letter in ("A", "B")))
        if letter == "A":
            ax.text(k_ipl + 0.12, ymax * 0.32, f"cutoff\n0.3 / Δz\n= {k_ipl:.2f} lp/mm", color=C_VERM, fontsize=6.6, ha="left", va="center")
            ax.text(nyq - 0.1, ymax * 0.10, "Nyquist", color=INK2, fontsize=6.6, ha="right", va="center")
        if letter == "B":
            for kc, col, lab in ((0.2 / el_z, C_ORANGE, "3.30"), (k_ipl, C_BLUE, "4.94"), (0.4 / el_z, C_GREEN, "6.59")):
                ax.text(kc + 0.1, ymax * 0.07, lab, color=col, fontsize=6.6, ha="left", va="center")
        if letter == "C":
            ax.text(k_ipl + 0.12, ymax * 0.16, "cutoff", color=C_VERM, fontsize=6.6, ha="left", va="center")
        for (key, ttl), tx in zip(tiles, TX):
            tile(fig, tx, ry + 4.5, TS, A[f"tile_{key}"], ttl, sub(key), ipl=(key == "std"),
                 scalebar_mm=2.0 if (letter == "A" and key == "std") else None, el=el[0])

    # ---------------------------------------------------------------------------------------- D padding
    DY = 140.0
    panel_letter(fig, 2.0, DY, "D")
    mm_text(fig, 8.0, DY + 0.2, "power-of-two padding  (impulse phantom, 63 voxels along x)", fontsize=7.5)
    P = NUM["phantom"]
    n = len(A["prof_ipl"])
    xs = np.arange(n)
    axs = mm_axes(fig, 13.0, DY + 5.8, 72.0, 3.2)          # the strip: padding voxel + data
    axp = mm_axes(fig, 13.0, DY + 9.4, 72.0, 25.8, sharex=axs)
    axs.set_xlim(-2.0, n + 0.5)
    axs.set_ylim(0, 1)
    axs.axis("off")
    for i in range(n):
        axs.add_patch(Rectangle((i - 0.5, 0.15), 1.0, 0.7, fc="#EEEEEE", ec=INK2, lw=0.3))
    axs.add_patch(Rectangle((-1.5, 0.15), 1.0, 0.7, fc=C_ORANGE, ec=INK2, lw=0.3, alpha=0.75))
    axs.add_patch(Rectangle((P["impulse_xyz"][0] - 0.5, 0.15), 1.0, 0.7, fc=C_BLUE, ec=INK2, lw=0.3))
    axs.text(-1.6, 1.1, "pad", fontsize=6.6, ha="center", va="bottom", color=C_ORANGE)
    axs.text(2.2, 1.1, "impulse at x = 1", fontsize=6.6, ha="left", va="bottom", color=C_BLUE)
    axs.text(n * 0.62, 1.1, "data voxels x = 0 … 62", fontsize=6.6, ha="center", va="bottom", color=INK2)
    for sp in ("top", "right"):
        axp.spines[sp].set_visible(False)
    axp.grid(True, color=GRID, lw=0.4)
    axp.set_axisbelow(True)
    sc = 1e-3
    axp.plot(xs, A["prof_floor"] * sc, color=C_ORANGE, lw=1.0, ls=(0, (4, 2)), label="reimplementation, padding voxel after the data")
    axp.plot(xs, A["prof_ceil"] * sc, color=C_BLUE, lw=1.3, label="reimplementation, padding voxel before the data (adopted)")
    axp.plot(xs, A["prof_ipl"] * sc, "o", ms=3.0, mfc="none", mec=INK, mew=0.7, label="IPL output")
    axp.set_xlim(-2.0, n + 0.5)
    axp.set_ylim(-25, 165)
    axp.set_xticks([0, 10, 20, 30, 40, 50, 62])
    axp.set_yticks([0, 50, 100, 150])
    axp.set_xlabel("x (voxel)", labelpad=1.5)
    axp.set_ylabel("filter output (×10³)", labelpad=2)
    axp.legend(loc="upper center", bbox_to_anchor=(0.56, 1.02), frameon=False, handlelength=1.8, borderaxespad=0.0, labelspacing=0.3, fontsize=6.6)
    axp.text(0.16, 0.40,                  # right of the curves at x = 0..3, below the key
             f"IPL vs before-the-data: max |Δ| {P['ceil']['max_abs_diff']:.3f} units,\nthreshold decisions differ on {P['ceil']['threshold_differs']}\n"
             f"IPL vs after-the-data: max |Δ| {fmt_int(round(P['floor']['max_abs_diff']))} units,\nthreshold decisions differ on {P['floor']['threshold_differs']}",
             transform=axp.transAxes, fontsize=6.6, va="center", ha="left", color=INK, linespacing=1.25)
    mm_text(fig, 13.0, DY + 42.7,
            "the mirror is edge-exclusive: the padding voxel copies voxel 1 (not voxel 0),\n"
            "so the impulse is duplicated at x = −1, the response peaks at x = 0 and\n"
            "reappears at the far end of the axis, the circular neighbor of the padding\n"
            "voxel.  The scan itself first receives a 1-voxel duplicated border on every\n"
            "face, then this padding to 1024 × 512 × 256.",
            fontsize=6.6, color=INK2, linespacing=1.2)

    # ---------------------------------------------------------------------------------------- E normalisation + threshold
    EX = 96.0
    panel_letter(fig, EX - 5.0, DY, "E")
    mm_text(fig, EX + 1.0, DY + 0.2, "scale to int16 (norm_max 200,000) and the threshold 475‰", fontsize=7.5)
    N = NUM["norm"]
    edges = np.asarray(A["hist_edges"])
    cnt = np.asarray(A["hist_counts"]).astype(float)
    axe = mm_axes(fig, EX + 6.0, DY + 9.2, 71.0, 26.0)
    for sp in ("top", "right"):
        axe.spines[sp].set_visible(False)
    axe.grid(True, color=GRID, lw=0.4)
    axe.set_axisbelow(True)
    ctr = 0.5 * (edges[:-1] + edges[1:]) * 1e-3
    axe.fill_between(ctr, 0.5, np.maximum(cnt, 0.5), step="mid", color=C_TRAB, lw=0)
    sel = ctr >= F_THR * 1e-3
    axe.fill_between(ctr[sel], 0.5, np.maximum(cnt[sel], 0.5), step="mid", color=C_SET, lw=0)
    axe.set_yscale("log")
    axe.set_ylim(0.5, cnt.max() * 60)
    axe.set_xlim(-225, 420)
    axe.axvline(F_THR * 1e-3, color=C_VERM, lw=1.0)
    axe.axvline(NMAX * 1e-3, color=INK2, lw=0.7, ls=(0, (3, 2)))
    axe.axvline(-NMAX * 1e-3, color=INK2, lw=0.7, ls=(0, (3, 2)))
    axe.set_xlabel("filter output (×10³ units)", labelpad=1.5)
    axe.set_ylabel("voxels", labelpad=2)
    axe.set_xticks([-200, -100, 0, 100, 200, 300, 400])
    axe.set_yticks([1, 1e2, 1e4, 1e6])
    axe.set_yticklabels(["10⁰", "10²", "10⁴", "10⁶"])      # Unicode powers: no 4.9-pt mathtext exponents
    axe.minorticks_off()
    top = axe.secondary_xaxis("top", functions=(lambda f: f * 1e3 * I16 / NMAX, lambda s: s * NMAX / I16 * 1e-3))
    top.set_xticks([-I16, 0, THR, I16])
    top.set_xticklabels(["−32,767", "0", "15,564", "32,767"])
    top.tick_params(labelsize=6.6, length=2, width=0.5, pad=1.5, colors=INK2)
    top.spines["top"].set_visible(False)
    top.set_xlabel("int16 level = trunc(output × 32,767 / 200,000)", labelpad=2, fontsize=6.6, color=INK2)
    axe.text(F_THR * 1e-3 + 8, cnt.max() * 1.3, f"threshold: 475‰ of 32,767\n= 15,564 = {F_THR / 1e3:.1f} × 10³ units",
             color=C_VERM, fontsize=6.6, ha="left", va="bottom", linespacing=1.2, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
    axe.text(NMAX * 1e-3 + 6, 1.2, "clip →\n32,767", color=INK2, fontsize=6.6, ha="left", va="bottom", linespacing=1.15,
             bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))
    axe.text(-NMAX * 1e-3 + 6, 1.2, "clip →\n−32,767", color=INK2, fontsize=6.6, ha="left", va="bottom", linespacing=1.15,
             bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))
    mm_text(fig, EX + 6.0, DY + 42.7,
            f"{N['ge_threshold_pct']:.1f}% of the scan's voxels reach the threshold; {fmt_int(N['above_clip'])}\n"
            f"voxels ({N['above_clip_pct']:.2f}%) exceed +200,000 and clip to 32,767, all of them\n"
            f"above the threshold; none fall below −200,000.  The float → int16\n"
            f"conversion truncates toward zero; the segmentation keeps\n"
            f"15,564 ≤ level ≤ 32,767.",
            fontsize=6.6, color=INK2, linespacing=1.2)

    # ---------------------------------------------------------------------------------------- F component filters
    FY = 197.0
    panel_letter(fig, 2.0, FY, "F")
    mm_text(fig, 8.0, FY + 0.2, "component filters  (IPL cl_nr_extract: components < 35 voxels leave the cortical, < 70 the trabecular compartment)",
            fontsize=7.5)
    C = NUM["components"]
    seg0 = A["seg0_win"]
    cs = A["comp_win"]
    gcort = A["gcort_win"]
    S2 = 28.0
    TY = FY + 9.5
    ax1 = mm_axes(fig, 13.0, TY, S2, S2)
    rgb = np.ones(seg0.shape + (3,), np.float32)
    rgb[seg0] = mcolors.to_rgb("#8C8C8C")
    rgb[seg0 & (cs >= CC_CORT) & (cs < CC_TRAB)] = mcolors.to_rgb(C_ORANGE)
    rgb[seg0 & (cs < CC_CORT)] = mcolors.to_rgb(C_VERM)
    ax1.imshow(rgb, interpolation="nearest", extent=[0, CROP, CROP, 0])
    ax1.contour(np.arange(CROP) + 0.5, np.arange(CROP) + 0.5, gcort.astype(float), levels=[0.5], colors=[C_GREEN], linewidths=0.7)
    # ring every small component so that single voxels are visible at print size
    for cls, col in (((cs < CC_CORT), C_VERM), (((cs >= CC_CORT) & (cs < CC_TRAB)), C_ORANGE)):
        m = seg0 & cls
        lab2, n2 = ndi.label(m, structure=np.ones((3, 3)))
        for i in range(1, n2 + 1):
            yy, xx = np.nonzero(lab2 == i)
            ax1.add_patch(Circle((xx.mean() + 0.5, yy.mean() + 0.5), 5.0, fill=False, ec=col, lw=0.6))
    n_small_win = len(np.unique(cs[seg0 & (cs < CC_CORT)]))
    n_mid_win = len(np.unique(cs[seg0 & (cs >= CC_CORT) & (cs < CC_TRAB)]))
    NUM["window"].update(components_lt_35_in_window=int(n_small_win), components_35_69_in_window=int(n_mid_win))
    for sp in ax1.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color(INK2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    mm_text(fig, 13.0, TY - 0.6, "masked threshold set\n(components by size)", va="bottom", fontsize=6.8, linespacing=1.15)
    ax2 = mm_axes(fig, 13.0 + S2 + 3.0, TY, S2, S2)
    seg = A["seg_win"]
    rgb2 = np.ones(seg.shape + (3,), np.float32)
    rgb2[seg == SEG_CORT] = mcolors.to_rgb(C_CORT)
    rgb2[seg == SEG_TRAB] = mcolors.to_rgb(C_TRAB)
    removed = seg0 & (seg == 0)
    rgb2[removed & (cs < CC_CORT)] = mcolors.to_rgb(C_VERM)
    rgb2[removed & (cs >= CC_CORT) & (cs < CC_TRAB)] = mcolors.to_rgb(C_ORANGE)
    ax2.imshow(rgb2, interpolation="nearest", extent=[0, CROP, CROP, 0])
    ax2.contour(np.arange(CROP) + 0.5, np.arange(CROP) + 0.5, gcort.astype(float), levels=[0.5], colors=[C_GREEN], linewidths=0.7)
    for sp in ax2.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color(INK2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    mm_text(fig, 13.0 + S2 + 3.0, TY - 0.6, "SEG after the filters\n(and the compartments)", va="bottom", fontsize=6.8, linespacing=1.15)
    n_removed_win = int(removed.sum())
    handles = [Patch(fc="#8C8C8C", ec="none", label="≥ 70 voxels (kept)"),
               Patch(fc=C_ORANGE, ec="none", label="35–69 voxels"),
               Patch(fc=C_VERM, ec="none", label="< 35 voxels"),
               Line2D([], [], color=C_GREEN, lw=0.8, label="cortical compartment"),
               Patch(fc=C_CORT, ec="none", label="cortical SEG (127)"),
               Patch(fc=C_TRAB, ec="none", label="trabecular SEG (126)")]
    lax = mm_axes(fig, 13.0, TY + S2 + 0.8, 2 * S2 + 3.0, 8.0)
    lax.axis("off")
    lax.legend(handles=handles, loc="upper left", ncol=2, frameon=False, handlelength=1.0, handleheight=0.8, columnspacing=1.2,
               borderaxespad=0, fontsize=6.6, labelspacing=0.25)
    # the size distribution
    sizes = np.asarray(A["sizes"], float)
    HX = 13.0 + 2 * S2 + 3.0 + 12.0
    HH = S2 - 4.0
    axh = mm_axes(fig, HX, TY, 177.0 - HX, HH)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
    axh.grid(True, color=GRID, lw=0.4)
    axh.set_axisbelow(True)
    bins = np.logspace(0, np.log10(sizes.max()) + 0.05, 60)
    h, _ = np.histogram(sizes, bins=bins)
    ctr = np.sqrt(bins[:-1] * bins[1:])
    cols = [C_VERM if b1 <= CC_CORT else (C_ORANGE if b1 <= CC_TRAB else "#8C8C8C") for b1 in bins[1:]]
    axh.bar(ctr, np.maximum(h, 0), width=np.diff(bins) * 0.9, color=cols, lw=0, align="center")
    axh.set_xscale("log")
    axh.set_yscale("log")
    axh.set_ylim(0.6, max(h) * 6)
    axh.set_xlim(0.8, sizes.max() * 3)
    axh.axvline(CC_CORT, color=C_VERM, lw=0.8, ls=(0, (3, 2)))
    axh.axvline(CC_TRAB, color=C_ORANGE, lw=0.8, ls=(0, (3, 2)))
    axh.text(CC_CORT * 0.9, max(h) * 2.5, "35", color=C_VERM, fontsize=6.8, ha="right", va="center")
    axh.text(CC_TRAB * 1.1, max(h) * 2.5, "70", color=C_ORANGE, fontsize=6.8, ha="left", va="center")
    axh.set_xlabel("component size (voxels)", labelpad=1.5)
    axh.set_ylabel("components", labelpad=2)
    axh.set_xticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6, 1e7])
    axh.set_xticklabels(["1", "10", "100", "10³", "10⁴", "10⁵", "10⁶", "10⁷"])
    axh.set_yticks([1, 10, 100, 1000])
    axh.set_yticklabels(["10⁰", "10¹", "10²", "10³"])
    axh.text(0.98, 0.95,
             f"{fmt_int(C['n_components'])} components; largest {fmt_int(C['largest'])} voxels\n"
             f"< 35: {fmt_int(C['n_lt_35'])} components, {fmt_int(C['voxels_lt_35'])} voxels\n"
             f"35–69: {C['n_35_69']} components, {fmt_int(C['voxels_35_69'])} voxels\n"
             f"≥ 70: {C['n_ge_70']} components",
             transform=axh.transAxes, fontsize=6.6, ha="right", va="top", color=INK, linespacing=1.2)
    mm_text(fig, HX, TY + HH + 8.2,
            f"this scan: SEG {fmt_int(C['seg'])} voxels (cortical {fmt_int(C['cort_seg'])}, trabecular\n"
            f"{fmt_int(C['trab_seg'])}; {C['cort_seg'] + C['trab_seg'] - C['seg']} in both carry 127); IPL's SEG: {fmt_int(C['seg_ipl'])}, {C['seg_differs_from_ipl']} differ.\n"
            f"Window: {n_small_win} components < 35 and {n_mid_win} of 35–69 voxels (ringed); {n_removed_win} removed.",
            fontsize=6.6, color=INK2, linespacing=1.2)
    NUM["window"]["voxels_removed_in_window"] = n_removed_win

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_SVG)
    plt.close(fig)
    say(f"wrote {OUT_PNG} and {OUT_SVG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="ignore the cache and recompute every intermediate")
    args = ap.parse_args()
    if args.recompute or not (os.path.exists(CACHE_NPZ) and os.path.exists(CACHE_JSON)):
        A = compute()
    else:
        A = load_cache()
        say("loaded the cache")
    draw(A)
    NUM["outputs"] = dict(png=public_path(OUT_PNG), svg=public_path(OUT_SVG), width_mm=FIG_W_MM, height_mm=FIG_H_MM, dpi=300)
    out = json.loads(json.dumps(NUM, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    out["components"].pop("sizes", None)                 # the arrays live in the cache; the sheet keeps the scalars
    out["norm"].pop("hist_edges", None)
    out["norm"].pop("hist_counts", None)
    with open(OUT_NUM, "w") as f:
        json.dump(out, f, indent=1)
    say("done")


if __name__ == "__main__":
    main()
