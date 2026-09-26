"""S7_spacing_number.py -- Supplementary Figure S7 of the ipldt / ORMIR-BQRL manuscript: dt_spacing and dt_number.

How dt_spacing and dt_number, which work on complements, differ from dt_thickness, on real data (one 64 x 64-voxel
window of one slice of one XtremeCT II patella scan, the window of Figure 4 enlarged) with every parameter at IPL's
standard value, and two rule variations with IPL's value marked:

  row 1  dt_spacing (Tb.Sp): A the object: marrow space, i.e. every voxel of the image that is not segmented bone
         (the trabecular contour does not bound it); B its containment ridge (ridge_epsilon 0.9), centres kept
         inside the contour (peel_iter -1) and discarded outside; C the Tb.Sp map, clipped to the contour, identical to
         IPL's; D the map that treating the contour as a boundary would give (object = marrow within the contour), which is
         not IPL's rule: the voxels that then differ from IPL's map are marked.
  row 2  dt_number (1/Tb.N): E the first ridge ('mid-axis'): the containment ridge of segmented bone at ridge_epsilon
         0.9, over the whole bone including the cortex, with no contour masking; F the second ridge: the containment
         ridge of the mid-axis complement, ridge_epsilon 0.9 again, with centres restricted to the contour; G the 1/Tb.N map, identical to IPL's; H the first ridge at ridge_epsilon 0.5 instead of IPL's 0.9
         (the voxels it adds to the mid-axis are marked) and, in the caption, how many voxels of the map then differ
         from IPL's.
  row 3  the statistics: I the distribution of the Tb.Sp map's values over the whole scan with its mean, which is Tb.Sp;
         J the same for the 1/Tb.N map, whose mean is 1/Tb.N and whose reciprocal is Tb.N.
  row 4  the objects of the two thickness maps: K the object of Tb.Th, the trabecular label of the segmentation inside
         the trabecular contour (the evaluation script's object; the cortical label is not part of it); L its Tb.Th map,
         identical to IPL's, with the value the whole segmentation would give as the object instead; M the object of
         Ct.Th, the cortical compartment mask inside the cortical contour, pores included (a window twice the size, at
         the cortex), whose Ct.Th map is identical to IPL's.

Every panel is computed here with the package's own stage functions (ipldt.field.sir_quad, ipldt.core ridge /
peel_gobj / diameters / draw_spheres; GPU kernels when CuPy is available); the maps of C and G are the public API's
(ipldt.dt_spacing / ipldt.dt_number), asserted equal to the stage chain, the maps of L and M are ipldt.dt_thickness on
IPL's trabecular segmentation and cortical mask, and the 'differing voxels' counts are taken on the whole 3-D grid
against IPL's exported maps.  Parameters: ridge_epsilon 0.9, assign_epsilon 0.5, peel_iter -1, version 3 (IPL's
standard values, Script 32).

Run from the repository root in the `ormir` environment:

    python manuscript/figures/supp/S7_spacing_number.py
        [--data <IPLDT_LAB_ROOT>/patellae/PFJ-0be66a_R] [--base X2420448]
        [--slice 84] [--window 30 422] [--size 64] [--recompute] [--out manuscript/figures/supp]

Writes S7_spacing_number.png (300 dpi, 180 mm wide), S7_spacing_number.svg and S7_spacing_number_numbers.json (every
number drawn, for the legend).  The whole-volume results are cached in manuscript/figures/cache/S7_spacing_number.npz
so that the layout can be redrawn without recomputing (--recompute forces the computation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
warnings.filterwarnings("ignore", message="CUDA path could not be detected")
import ipldt  # noqa: E402
from ipldt.core import ridge, surface_distance, diameters, draw_spheres, peel_gobj  # noqa: E402
from ipldt.field import sir_quad  # noqa: E402
from ipldt.gpu import cupy_available  # noqa: E402

DATA_DEFAULT = lab_path("patellae/PFJ-0be66a_R")
BASE_DEFAULT = "X2420448"
CACHE_DEFAULT = os.path.join(REPO, "manuscript", "figures", "cache", "S7_spacing_number.npz")
T0 = time.time()

RIDGE_EPS, ASSIGN_EPS, PEEL_ITER, VERSION = 0.9, 0.5, -1, 3      # IPL's standard values (Script 32)
MIDAXIS_EPS_VARIANT = 0.5                                         # the varied value of panel H


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------ style
# Okabe-Ito colour-blind-safe palette; viridis for the diameter maps (as in Figure 4).
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", blue="#0072B2", verm="#D55E00", purple="#CC79A7")
BG = np.array([1.0, 1.0, 1.0])
OBJ = np.array([0.86, 0.86, 0.86])        # object voxels without a value (light grey)
BONE = np.array([0.55, 0.55, 0.55])       # bone when it is not the object (dark grey)
CONTOUR = "#000000"
INK, INK2 = "#1a1a1a", "#4d4d4d"
PT = 7.0                                  # smallest font at print size (points)
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
                     "font.size": PT, "axes.linewidth": 0.5, "svg.fonttype": "none", "figure.dpi": 100,
                     "savefig.dpi": 300, "pdf.fonttype": 42, "axes.unicode_minus": True,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold",
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0,
                     "ytick.major.size": 2.0, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5})

W_MM = 180.0
MM = 1.0 / 25.4


def rgb(c):
    return np.array(colors.to_rgb(c))


def fmt(n):
    """Thousands separated by commas, as in the legends and the text."""
    return f"{int(n):,d}"


def diam_cmap(vmax):
    """Discrete colour scale for integer diameters 1..vmax (viridis); 0 is handled by the caller."""
    base = plt.get_cmap("viridis", vmax)
    cm = colors.ListedColormap(base(np.arange(vmax)))
    norm = colors.BoundaryNorm(np.arange(0.5, vmax + 1.5, 1.0), cm.N)
    return cm, norm


# ------------------------------------------------------------------------------------------ data
def load(data, base):
    p = lambda nm: os.path.join(data, f"{base}_{nm}_decompressed.AIM")  # noqa: E731
    A = ipldt.read_aim(p("SEG"))
    dim, pos, el = A["dim"], A["pos"], float(A["el_size_mm"][0])
    seg = np.asarray(A["data"])
    L = lambda nm: np.asarray(ipldt.align_to(ipldt.read_aim(p(nm)), dim, pos))  # noqa: E731
    maps = {nm: L(nm).astype(np.int16) for nm in ("TRAB_SP", "TRAB_1N", "TRAB_TH_old", "TRAB_TH")}
    mraw = ipldt.read_aim(p("TRAB_MASK"))
    G_own = ipldt.render_volume(np.asarray(mraw["data"]) > 0)
    G = np.asarray(ipldt.align_to(dict(data=G_own.astype(np.uint8), dim=mraw["dim"], pos=mraw["pos"]), dim, pos)) > 0
    trabseg = L("TRAB_SEG") > 0                          # IPL's trabecular segmentation: the object of Tb.Th
    craw = ipldt.read_aim(p("CORT_MASK"))                # the cortical compartment on its own grid (the grid of Ct.Th)
    cm = np.asarray(craw["data"]) > 0
    Gc = ipldt.render_volume(cm)
    ct_ipl = np.asarray(ipldt.align_to(ipldt.read_aim(p("CORT_TH")), craw["dim"], craw["pos"])).astype(np.int16)
    say(f"grid {tuple(seg.shape)} (z, y, x), voxel {el:.6f} mm; bone {int((seg > 0).sum()):,d} voxels "
        f"(cortical {int((seg == 127).sum()):,d}, trabecular {int((seg == 126).sum()):,d}); marrow "
        f"{int((seg == 0).sum()):,d}; rendered trabecular contour {int(G.sum()):,d} voxels; trabecular segmentation "
        f"{int(trabseg.sum()):,d}; cortical compartment {int(cm.sum()):,d} voxels on a {tuple(cm.shape)} grid")
    return dict(seg=seg, el=el, G=G, maps=maps, dim=dim, pos=pos, trabseg=trabseg,
                cort=dict(mask=cm, G=Gc, ipl=ct_ipl, dim=craw["dim"], pos=craw["pos"]))


def ridge_any(obj, V, eps):
    if cupy_available():
        from ipldt.gpu import ridge_gpu
        return ridge_gpu(obj, V, eps)
    return ridge(obj, V, eps)


def draw_any(z, y, x, D, shape):
    if cupy_available():
        from ipldt.gpu import draw_spheres_gpu
        return draw_spheres_gpu(z, y, x, D, shape, ASSIGN_EPS)
    return draw_spheres(z, y, x, D, shape, ASSIGN_EPS)


def run_stages(obj, G, public_map, public_centres, label):
    """The stage chain on the whole volume; `public_map` / `public_centres` are the public API's, asserted equal."""
    obj = np.ascontiguousarray(obj, dtype=bool)
    V = sir_quad(obj)
    s = surface_distance(V)
    s[~obj] = 0.0
    cen_all = ridge_any(obj, V, RIDGE_EPS)
    cen = cen_all & peel_gobj(G, PEEL_ITER)
    z, y, x = np.nonzero(cen)
    D = diameters(obj, V, z, y, x, version=VERSION)
    m = draw_any(z, y, x, D, obj.shape)
    m[~obj] = 0
    m[~G] = 0
    n_diff_api = int((m != public_map).sum())
    n_diff_cen = int((cen != public_centres).sum())
    assert n_diff_api == 0, f"{label}: stage chain and public API maps differ on {n_diff_api} voxels"
    assert n_diff_cen == 0, f"{label}: stage chain and public API centres differ on {n_diff_cen} voxels"
    say(f"{label}: {int(cen_all.sum()):,d} ridge centres, {int(cen.sum()):,d} inside the contour, diameters "
        f"{int(D.min())}..{int(D.max())}, map support {int((m > 0).sum()):,d} voxels; stage chain == public API")
    return dict(obj=obj, s=s, cen_all=cen_all, cen=cen, map=m, n_cen_all=int(cen_all.sum()), n_cen=int(cen.sum()),
                n_removed=int((cen_all & ~cen).sum()), d_min=int(D.min()), d_max=int(D.max()),
                support=int((m > 0).sum()))


def compute(data, base, z, y0, x0, n):
    """Everything the figure needs, on the whole volume; returns the crops and the numbers."""
    P = load(data, base)
    seg, G, el = P["seg"], P["G"], P["el"]
    bone = seg > 0
    ipl_sp, ipl_1n = P["maps"]["TRAB_SP"], P["maps"]["TRAB_1N"]
    sl = (z, slice(y0, y0 + n), slice(x0, x0 + n))
    kw = dict(voxel_size_mm=el, ridge_epsilon=RIDGE_EPS, assign_epsilon=ASSIGN_EPS, peel_iter=PEEL_ITER, version=VERSION)
    N = dict(parameters=dict(ridge_epsilon=RIDGE_EPS, assign_epsilon=ASSIGN_EPS, peel_iter=PEEL_ITER, version=VERSION,
                             midaxis_ridge_epsilon_variant=MIDAXIS_EPS_VARIANT,
                             backend="gpu" if cupy_available() else "cpu"),
             scan=dict(base=base, grid_zyx=[int(v) for v in seg.shape], voxel_mm=el, slice_index=z,
                       window_origin_yx=[int(y0), int(x0)], window_voxels=n,
                       n_bone=int(bone.sum()), n_marrow=int((~bone).sum()), n_contour=int(G.sum()),
                       n_voxels=int(seg.size)))
    C = dict(seg=seg[sl].copy(), G=G[sl].copy())

    # ---------------------------------------------------------------- row 1: dt_spacing (the marrow object)
    r_sp = ipldt.dt_spacing(bone, {"rendered": G}, **kw)
    S = run_stages(~bone, G, r_sp.map, r_sp.centres, "spacing")
    n_diff_sp = int((r_sp.map != ipl_sp).sum())
    say(f"spacing: Tb.Sp {r_sp.report['Sp_mm']:.6f} mm over {r_sp.report['Sp_n_voxels']:,d} voxels; differing voxels "
        f"vs IPL {n_diff_sp} of {ipl_sp.size:,d}")
    # the variation: the contour treated as a surface (object = marrow inside the contour only)
    r_sp_alt = ipldt.dt_thickness((~bone) & G, {"rendered": G}, **kw)
    diff_sp_alt = r_sp_alt.map != ipl_sp
    say(f"spacing, contour as surface: map differs from IPL on {int(diff_sp_alt.sum()):,d} voxels "
        f"(support {int((r_sp_alt.map > 0).sum()):,d}; mean {float(r_sp_alt.map[r_sp_alt.map > 0].mean()) * el:.6f} mm)")
    C.update(sp_cen_all=S["cen_all"][sl].copy(), sp_cen=S["cen"][sl].copy(), sp_map=S["map"][sl].copy(),
             sp_ipl=ipl_sp[sl].copy(), sp_alt=r_sp_alt.map[sl].copy(), sp_alt_diff=diff_sp_alt[sl].copy())
    N["spacing"] = dict(volume=dict(centres=S["n_cen_all"], kept=S["n_cen"], removed=S["n_removed"],
                                    diameters=[S["d_min"], S["d_max"]], support=S["support"],
                                    ipl_support=int((ipl_sp > 0).sum()), differing_voxels=n_diff_sp,
                                    voxels_compared=int(ipl_sp.size), report=dict(r_sp.report)),
                        contour_as_surface=dict(differing_voxels=int(diff_sp_alt.sum()),
                                                support=int((r_sp_alt.map > 0).sum()),
                                                mean_mm=float(r_sp_alt.map[r_sp_alt.map > 0].mean()) * el,
                                                changed_lower=int((diff_sp_alt & (r_sp_alt.map < ipl_sp)).sum()),
                                                changed_higher=int((diff_sp_alt & (r_sp_alt.map > ipl_sp)).sum())))
    hist_sp = np.bincount(r_sp.map[r_sp.map > 0].astype(np.int64))
    hist_sp_ipl = np.bincount(ipl_sp[ipl_sp > 0].astype(np.int64))
    del S, r_sp_alt, diff_sp_alt

    # ---------------------------------------------------------------- row 2: dt_number (two ridges)
    r_n = ipldt.dt_number(bone, {"rendered": G}, **kw)
    Vb = sir_quad(bone)
    mid = ridge_any(bone, Vb, RIDGE_EPS)                # the first ridge: the mid-axis, whole bone, unmasked
    mid_var = ridge_any(bone, Vb, MIDAXIS_EPS_VARIANT)  # the varied first ridge (panel H)
    del Vb
    say(f"mid-axis (ridge of the bone at {RIDGE_EPS}): {int(mid.sum()):,d} voxels, {int((mid & G).sum()):,d} inside the "
        f"contour, {int((mid & (seg == 127)).sum()):,d} in the cortical label; at {MIDAXIS_EPS_VARIANT}: {int(mid_var.sum()):,d}")
    S = run_stages(~mid, G, r_n.map, r_n.centres, "number")
    n_diff_n = int((r_n.map != ipl_1n).sum())
    say(f"number: 1/Tb.N {r_n.report['inv_N_mm']:.6f} mm over {r_n.report['inv_N_n_voxels']:,d} voxels, Tb.N "
        f"{r_n.report['Tb_N_per_mm']:.6f} /mm; differing voxels vs IPL {n_diff_n} of {ipl_1n.size:,d}")
    r_n_alt = ipldt.dt_thickness(~mid_var, {"rendered": G}, **kw)
    diff_n_alt = r_n_alt.map != ipl_1n
    mean_alt = float(r_n_alt.map[r_n_alt.map > 0].mean()) * el
    say(f"number, mid-axis at {MIDAXIS_EPS_VARIANT}: map differs from IPL on {int(diff_n_alt.sum()):,d} voxels; 1/Tb.N "
        f"{mean_alt:.6f} mm, Tb.N {1 / mean_alt:.6f} /mm")
    C.update(mid=mid[sl].copy(), mid_var=mid_var[sl].copy(), n_cen_all=S["cen_all"][sl].copy(), n_cen=S["cen"][sl].copy(),
             n_map=S["map"][sl].copy(), n_ipl=ipl_1n[sl].copy(), n_alt=r_n_alt.map[sl].copy(),
             n_alt_diff=diff_n_alt[sl].copy())
    N["number"] = dict(volume=dict(midaxis=int(mid.sum()), midaxis_inside_contour=int((mid & G).sum()),
                                   midaxis_cortical=int((mid & (seg == 127)).sum()),
                                   centres=S["n_cen_all"], kept=S["n_cen"], removed=S["n_removed"],
                                   diameters=[S["d_min"], S["d_max"]], support=S["support"],
                                   ipl_support=int((ipl_1n > 0).sum()), differing_voxels=n_diff_n,
                                   voxels_compared=int(ipl_1n.size), report=dict(r_n.report)),
                       midaxis_variant=dict(ridge_epsilon=MIDAXIS_EPS_VARIANT, midaxis=int(mid_var.sum()),
                                            added=int((mid_var & ~mid).sum()), removed=int((mid & ~mid_var).sum()),
                                            differing_voxels=int(diff_n_alt.sum()),
                                            support=int((r_n_alt.map > 0).sum()), inv_N_mm=mean_alt,
                                            Tb_N_per_mm=1 / mean_alt,
                                            changed_lower=int((diff_n_alt & (r_n_alt.map < ipl_1n)).sum()),
                                            changed_higher=int((diff_n_alt & (r_n_alt.map > ipl_1n)).sum())))
    hist_n = np.bincount(r_n.map[r_n.map > 0].astype(np.int64))
    hist_n_ipl = np.bincount(ipl_1n[ipl_1n > 0].astype(np.int64))
    C.update(hist_sp=hist_sp, hist_sp_ipl=hist_sp_ipl, hist_n=hist_n, hist_n_ipl=hist_n_ipl)
    del S, r_n, r_n_alt, diff_n_alt, mid_var

    # ---------------------------------------------------------------- row 3: the objects of Tb.Th and Ct.Th
    trabseg = P["trabseg"]
    ipl_th_old, ipl_th = P["maps"]["TRAB_TH_old"], P["maps"]["TRAB_TH"]
    r_th = ipldt.dt_thickness(trabseg, {"rendered": G}, **kw)
    n_diff_th = int((r_th.map != ipl_th_old).sum())
    say(f"thickness, trabecular label: Tb.Th {r_th.report['Th_mm']:.6f} mm over {r_th.report['Th_n_voxels']:,d} voxels; "
        f"differing voxels vs IPL {n_diff_th} of {ipl_th_old.size:,d}")
    r_th_all = ipldt.dt_thickness(bone, {"rendered": G}, **kw)
    n_diff_th_all = int((r_th_all.map != ipl_th).sum())
    say(f"thickness, whole segmentation as the object: {r_th_all.report['Th_mm']:.6f} mm; differing voxels vs IPL "
        f"{n_diff_th_all} of {ipl_th.size:,d}")
    cort = P["cort"]
    r_ct = ipldt.dt_thickness(cort["mask"], {"rendered": cort["G"]}, **kw)
    n_diff_ct = int((r_ct.map != cort["ipl"]).sum())
    say(f"cortical: Ct.Th {r_ct.report['Th_mm']:.6f} mm over the compartment ({int(cort['mask'].sum()):,d} voxels); "
        f"differing voxels vs IPL {n_diff_ct} of {cort['ipl'].size:,d}")
    to_seg = lambda a: np.asarray(ipldt.align_to(dict(data=a.astype(np.uint8), dim=cort["dim"], pos=cort["pos"]),  # noqa: E731
                                                 P["dim"], P["pos"])) > 0
    cm_s, Gc_s = to_seg(cort["mask"]), to_seg(cort["G"])
    pores = cm_s & ~bone
    nc = 2 * n                                           # the cortical window: twice the size, the same x centre
    xc0 = int(x0 + n // 2 - nc // 2)
    rows_c = np.nonzero(cm_s[z, :, xc0:xc0 + nc].any(axis=1))[0]
    yc0 = int(max(0, rows_c.min() - 6)) if rows_c.size else int(max(0, y0 + n // 2 - nc // 2))
    slc = (z, slice(yc0, yc0 + nc), slice(xc0, xc0 + nc))
    C.update(trabseg=trabseg[sl].copy(), th_map=r_th.map[sl].copy(), th_ipl=ipl_th_old[sl].copy(),
             ct_mask=cm_s[slc].copy(), ct_G=Gc_s[slc].copy(), ct_bone=bone[slc].copy(),
             ct_window=np.array([yc0, xc0, nc]))
    N["thickness"] = dict(
        trabecular_label=dict(n_object=int(trabseg.sum()), differing_voxels=n_diff_th, voxels_compared=int(ipl_th_old.size),
                              support=int((r_th.map > 0).sum()), report=dict(r_th.report)),
        whole_segmentation=dict(n_object=int(bone.sum()), differing_voxels=n_diff_th_all, voxels_compared=int(ipl_th.size),
                                support=int((r_th_all.map > 0).sum()), report=dict(r_th_all.report)))
    N["cortical"] = dict(n_object=int(cort["mask"].sum()), n_pores=int(pores.sum()), n_contour=int(cort["G"].sum()),
                         differing_voxels=n_diff_ct, voxels_compared=int(cort["ipl"].size),
                         grid_zyx=[int(v) for v in cort["mask"].shape], window_origin_yx=[yc0, xc0], window_voxels=nc,
                         report=dict(r_ct.report))
    return C, N


# ------------------------------------------------------------------------------------------ figure
def crop_numbers(C):
    """Counts inside the window, for the captions."""
    g = C["G"]
    out = {}
    for key in ("sp", "n"):
        ca, ck = C[f"{key}_cen_all"], C[f"{key}_cen"]
        out[key] = dict(centres=int(ca.sum()), kept=int(ck.sum()), removed=int((ca & ~ck).sum()),
                        map_max=int(C[f"{key}_map"].max()), ipl_max=int(C[f"{key}_ipl"].max()),
                        alt_max=int(C[f"{key}_alt"].max()), alt_diff=int(C[f"{key}_alt_diff"].sum()),
                        differing=int((C[f"{key}_map"] != C[f"{key}_ipl"]).sum()))
    out["midaxis"] = int(C["mid"].sum())
    out["midaxis_inside_contour"] = int((C["mid"] & g).sum())
    out["midaxis_variant"] = int(C["mid_var"].sum())
    out["midaxis_variant_added"] = int((C["mid_var"] & ~C["mid"]).sum())
    out["midaxis_variant_removed"] = int((C["mid"] & ~C["mid_var"]).sum())
    out["marrow"] = int((C["seg"] == 0).sum())
    out["marrow_inside_contour"] = int(((C["seg"] == 0) & g).sum())
    out["th"] = dict(object=int(C["trabseg"].sum()), map_max=int(C["th_map"].max()), ipl_max=int(C["th_ipl"].max()),
                     differing=int((C["th_map"] != C["th_ipl"]).sum()))
    out["ct"] = dict(object=int(C["ct_mask"].sum()), pores=int((C["ct_mask"] & ~C["ct_bone"]).sum()),
                     bone_outside=int((C["ct_bone"] & ~C["ct_mask"]).sum()))
    return out


def make_figure(C, N, out_png, out_svg):
    seg, G, el = C["seg"], C["G"], N["scan"]["voxel_mm"]
    n = seg.shape[0]
    y0, x0 = N["scan"]["window_origin_yx"]
    bone = seg > 0
    marrow = ~bone
    cn = crop_numbers(C)
    V = N["spacing"]["volume"], N["number"]["volume"]
    nvox = N["scan"]["n_voxels"]

    # ---- colour scales per row (the diameters present in the window, as in Figure 4)
    vmax_sp = int(max(C["sp_map"].max(), C["sp_ipl"].max(), C["sp_alt"].max()))
    vmax_n = int(max(C["n_map"].max(), C["n_ipl"].max(), C["n_alt"].max()))

    def map_img(m, obj, vmax):
        cm, norm = diam_cmap(vmax)
        im = np.where(obj[..., None], OBJ, BG)
        has = m > 0
        im[has] = cm(norm(m[has]))[:, :3]
        return im

    # row 1 images
    img_a = np.where(marrow[..., None], OBJ, BONE)                     # marrow light grey = the object; bone dark
    img_b = np.where(marrow[..., None], OBJ, BG)
    img_b[C["sp_cen_all"] & ~C["sp_cen"]] = rgb(OI["blue"])
    img_b[C["sp_cen"]] = rgb(OI["verm"])
    img_c = map_img(C["sp_map"], marrow, vmax_sp)
    img_d = map_img(C["sp_alt"], marrow, vmax_sp)
    img_d[C["sp_alt_diff"]] = rgb(OI["verm"])
    # row 2 images
    img_e = np.where(bone[..., None], OBJ, BG)
    img_e[C["mid"]] = rgb(OI["verm"])
    img_f = np.where(C["mid"][..., None], BONE, OBJ)                   # object = NOT mid-axis (light); mid-axis dark
    img_f[C["n_cen_all"] & ~C["n_cen"]] = rgb(OI["blue"])
    img_f[C["n_cen"]] = rgb(OI["verm"])
    img_g = map_img(C["n_map"], ~C["mid"], vmax_n)
    img_h = np.where(bone[..., None], OBJ, BG)                         # the first ridge at the varied epsilon
    img_h[C["mid"]] = rgb(OI["verm"])
    img_h[C["mid_var"] & ~C["mid"]] = rgb(OI["blue"])
    # row 3 images: the objects of Tb.Th and Ct.Th
    trabseg = C["trabseg"]
    vmax_th = int(max(C["th_map"].max(), C["th_ipl"].max()))
    img_i = np.where(trabseg[..., None], OBJ, np.where(bone[..., None], BONE, BG))   # object = trabecular label
    img_j = map_img(C["th_map"], trabseg, vmax_th)
    ct_mask, ct_G, ct_bone = C["ct_mask"], C["ct_G"], C["ct_bone"]
    img_k = np.where(ct_mask[..., None], OBJ, np.where(ct_bone[..., None], BONE, BG))  # object = the compartment
    img_k[ct_mask & ~ct_bone] = rgb(OI["verm"])                        # its pores are object voxels too
    yc0, xc0, nc = (int(v) for v in C["ct_window"])
    ext = (x0 - 0.5, x0 + n - 0.5, y0 + n - 0.5, y0 - 0.5)              # the 64-voxel window (rows 1, 2 and K, L)
    ext_c = (xc0 - 0.5, xc0 + nc - 0.5, yc0 + nc - 0.5, yc0 - 0.5)      # the cortical window (M)
    T, Tw, Tc = N["thickness"]["trabecular_label"], N["thickness"]["whole_segmentation"], N["cortical"]

    rows = [
        dict(letters="ABCD", images=[img_a, img_b, img_c, img_d], vmax=vmax_sp, cbar_under=(2, 3),
             headers=["Object: the marrow space", f"Centers, ridge_epsilon {RIDGE_EPS:g}",
                      "Tb.Sp map (ipldt = IPL's)", "Contour treated as a surface"],
             captions=["complement of the segmentation\nover the whole image (light gray);\nbone dark gray",
                       f"{cn['sp']['kept']} kept inside the contour,\n{cn['sp']['removed']} discarded outside it",
                       f"differing voxels: {V[0]['differing_voxels']} of {fmt(nvox)}",
                       f"not IPL's rule: the map differs\nfrom IPL's on "
                       f"{fmt(N['spacing']['contour_as_surface']['differing_voxels'])} voxels\n"
                       f"(vermilion; {cn['sp']['alt_diff']} in the window)"],
             contour_color=[CONTOUR] * 4),
        dict(letters="EFGH", images=[img_e, img_f, img_g, img_h], vmax=vmax_n, cbar_under=(2, 3),
             headers=[f"Mid-axis: ridge_epsilon {RIDGE_EPS:g}",
                      f"Centers: ridge_epsilon {RIDGE_EPS:g}",
                      "1/Tb.N map (ipldt = IPL's)",
                      f"Mid-axis: ridge_epsilon {MIDAXIS_EPS_VARIANT:g}"],
             captions=[f"first ridge: containment ridge of\nthe whole bone, not masked by the\n"
                       f"contour ({cn['midaxis']} voxels in the window)",
                       f"second ridge: on the complement of\nthe mid-axis (dark gray); {cn['n']['kept']} kept,\n"
                       f"{cn['n']['removed']} discarded",
                       f"differing voxels: {V[1]['differing_voxels']} of {fmt(nvox)}",
                       f"IPL's value: {RIDGE_EPS:g}. Blue: voxels added\nat {MIDAXIS_EPS_VARIANT:g}; the map then differs from\n"
                       f"IPL's on {fmt(N['number']['midaxis_variant']['differing_voxels'])} voxels"],
             contour_color=[CONTOUR] * 4),
        dict(letters="KLM", images=[img_i, img_j, img_k], vmax=vmax_th, cbar_under=(1, 1),
             headers=["Object of Tb.Th", "Tb.Th map (ipldt = IPL's)", "Object of Ct.Th"],
             captions=["the trabecular label inside the\ntrabecular contour (light gray);\ncortical label (dark) not included",
                       f"differing voxels: {T['differing_voxels']} of {fmt(T['voxels_compared'])}\n"
                       f"Tb.Th {T['report']['Th_mm']:.4f} mm; with the whole\nSEG as the object: {Tw['report']['Th_mm']:.4f} mm",
                       f"the compartment is the object;\nnon-bone voxels (vermilion) count,\nbone outside it (dark gray) not\n"
                       f"Ct.Th map: {Tc['differing_voxels']} of {fmt(Tc['voxels_compared'])} differ"],
             contour_color=[CONTOUR] * 3, contours=[G, G, ct_G], exts=[ext, ext, ext_c], scalebar={2: 2.0}),
    ]

    # ---- geometry in mm
    left, right, gap = 5.0, 1.0, 1.6
    P = (W_MM - left - right - 3 * gap) / 4.0          # panel width = height
    hdr_h, cap_h, cb_h = 3.6, 9.0, 1.9
    below = 0.8 + cap_h + 0.4 + cb_h + 2.6 + 2.8       # caption + colour bar + tick labels + colour-bar label
    row_h = hdr_h + P + below
    hist_h = 30.0
    hist_below = 7.5                                    # x tick labels + x label
    top, bottom, row_gap = 1.0, 1.0, 2.5
    H_MM = top + len(rows) * (row_h + row_gap) + hdr_h + hist_h + hist_below + bottom
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))

    def ax_mm(x, y_top, w, h):
        return fig.add_axes([x / W_MM, 1.0 - (y_top + h) / H_MM, w / W_MM, h / H_MM])

    def text_mm(x, y_top, s, **k):
        return fig.text(x / W_MM, 1.0 - y_top / H_MM, s, **k)

    checks = []
    # rows 1 and 2 (A-H), then the statistics (I, J), then the objects row (K-M) under them
    y_hist = top + 2 * (row_h + row_gap)
    y_tops = [top, top + (row_h + row_gap), y_hist + hdr_h + hist_h + hist_below + row_gap]
    for r, row in enumerate(rows):
        py = y_tops[r] + hdr_h                              # top of this row's panels
        cm, norm = diam_cmap(row["vmax"])
        exts = row.get("exts", [ext] * len(row["images"]))
        conts = row.get("contours", [G] * len(row["images"]))
        bars = row.get("scalebar", {0: 1.0} if r == 0 else {})   # panel index -> scale-bar length (mm)
        for i, img in enumerate(row["images"]):
            px = left + i * (P + gap)
            ax = ax_mm(px, py, P, P)
            e, g = exts[i], conts[i]
            gx = np.arange(e[0] + 0.5, e[0] + 0.5 + g.shape[1])
            gy = np.arange(e[3] + 0.5, e[3] + 0.5 + g.shape[0])
            ax.imshow(img, interpolation="nearest", extent=e)
            ax.contour(gx, gy, g.astype(float), levels=[0.5], colors=[row["contour_color"][i]], linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#9a9a9a")
            # letter (upper-left, above the panel) and header
            text_mm(px - 0.2, py - 0.9, row["letters"][i], ha="left", va="bottom", fontsize=PT + 2, fontweight="bold")
            t = text_mm(px + 4.2, py - 1.1, row["headers"][i], ha="left", va="bottom", fontsize=PT, fontweight="bold")
            checks.append((t, P - 4.2, "header"))
            # scale bar (lower left; its offsets scale with the window so that it looks the same in every window)
            if i in bars:
                wv = g.shape[0]
                bar = bars[i] / el
                bx, by = e[0] + 0.5 + 2.0 * wv / 64, e[3] + 0.5 + wv - 4.0 * wv / 64
                ax.plot([bx, bx + bar], [by, by], color="black", lw=1.6, solid_capstyle="butt")
                ax.text(bx + bar / 2, by - 1.8 * wv / 64, f"{bars[i]:g} mm", ha="center", va="bottom", fontsize=PT,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
            cy = py + P + 0.8
            t = text_mm(px + P / 2, cy + 0.2, row["captions"][i], ha="center", va="top", fontsize=PT, color="#222222",
                        linespacing=1.15)
            checks.append((t, P + 0.6, "caption"))
        # one colour bar for the two map panels of the row
        i0, i1 = row["cbar_under"]
        cx0 = left + i0 * (P + gap) + P * 0.15
        cx1 = left + i1 * (P + gap) + P * 0.85
        cax = ax_mm(cx0, py + P + 0.8 + cap_h + 0.4, cx1 - cx0, cb_h)
        sm = ScalarMappable(norm=norm, cmap=cm)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        vmax = row["vmax"]
        step = 1 if vmax <= 10 else (2 if vmax <= 20 else 4)
        ticks = list(range(1, vmax + 1, step))
        if ticks[-1] != vmax:
            ticks.append(vmax)
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])
        cb.ax.tick_params(labelsize=PT, length=1.5, pad=1.2, width=0.5)
        cb.outline.set_linewidth(0.5)
        cb.set_label("sphere diameter (voxels)", fontsize=PT, labelpad=1.5)

    # ---- row 3: the statistics (whole-scan histograms of the two maps)
    py = y_hist + hdr_h
    hist_w = 2 * P + gap
    specs = [
        ("I", "Tb.Sp = mean of the Tb.Sp map", C["hist_sp"], C["hist_sp_ipl"], N["spacing"]["volume"]["report"]["Sp_mm"],
         N["spacing"]["volume"]["report"]["Sp_n_voxels"], "Tb.Sp map value (voxels)", OI["blue"], None),
        ("J", "Tb.N = 1 / mean of the 1/Tb.N map", C["hist_n"], C["hist_n_ipl"], N["number"]["volume"]["report"]["inv_N_mm"],
         N["number"]["volume"]["report"]["inv_N_n_voxels"], "1/Tb.N map value (voxels)", OI["green"],
         N["number"]["volume"]["report"]["Tb_N_per_mm"]),
    ]
    hist_numbers = {}
    for k, (letter, header, h_ours, h_ipl, mean_mm, n_vox, xlabel, col, tbn) in enumerate(specs):
        px = left + k * (hist_w + gap)
        ax = ax_mm(px + 7.0, py, hist_w - 8.0, hist_h)
        d = np.arange(1, len(h_ours))
        counts = h_ours[1:] / 1e6
        assert len(h_ours) == len(h_ipl) and int((h_ours != h_ipl).sum()) == 0, "histograms of ipldt and IPL differ"
        ax.bar(d, counts, width=0.8, color=col, edgecolor="none", zorder=2)
        mean_vox = mean_mm / el
        ax.axvline(mean_vox, color=INK, lw=0.9, ls="--", zorder=3)
        ymax = counts.max() * (1.30 if tbn is not None else 1.18)   # room for the annotation box
        ax.set_ylim(0, ymax)
        ax.set_xlim(0.3, len(h_ours) - 0.3)
        ax.set_xticks(np.arange(1, len(h_ours), 2 if len(h_ours) > 12 else 1))
        ax.set_xlabel(f"{xlabel[:-1]}; 1 voxel = {el:.4f} mm)", fontsize=PT, labelpad=1.5)
        ax.set_ylabel("voxels (millions)", fontsize=PT, labelpad=2)
        ax.tick_params(labelsize=PT)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if tbn is None:
            lab = (f"mean {mean_vox:.3f} voxels = {mean_mm:.4f} mm = Tb.Sp\n{fmt(n_vox)} voxels with a value")
        else:
            lab = (f"mean {mean_vox:.3f} voxels = {mean_mm:.4f} mm = 1/Tb.N\n"
                   f"Tb.N = 1 / {mean_mm:.4f} mm = {tbn:.4f} 1/mm\n{fmt(n_vox)} voxels with a value")
        ax.text(0.98, 0.96, lab, transform=ax.transAxes, ha="right", va="top", fontsize=PT, color=INK, linespacing=1.25,
                bbox=dict(boxstyle="square,pad=0.25", fc="white", ec="none", alpha=0.9))
        text_mm(px - 0.2, py - 0.9, letter, ha="left", va="bottom", fontsize=PT + 2, fontweight="bold")
        text_mm(px + 4.2, py - 1.1, header, ha="left", va="bottom", fontsize=PT, fontweight="bold")
        hist_numbers[letter] = dict(mean_voxels=mean_vox, mean_mm=mean_mm, n_voxels=int(n_vox), max_value=int(len(h_ours) - 1),
                                    ipl_histogram_identical=True, **({"Tb_N_per_mm": tbn} if tbn else {}))

    # text widths at print size (headers and captions must stay inside their panel)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for t, pw, kind in checks:
        w_mm = t.get_window_extent(rend).width / fig.dpi * 25.4
        flag = "" if w_mm <= pw else "   <-- WIDER THAN PANEL"
        say(f"{kind} '{t.get_text().splitlines()[0]}': {w_mm:.1f} mm of {pw:.1f} mm{flag}")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)
    N["figure"] = dict(width_mm=W_MM, height_mm=round(H_MM, 1), panel_mm=round(P, 2), font_pt=PT,
                       colour_scale_max=dict(spacing=vmax_sp, number=vmax_n, thickness=vmax_th), window=cn,
                       cortical_window=dict(origin_yx=[yc0, xc0], voxels=nc), histograms=hist_numbers)
    return N


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--slice", type=int, default=84)
    ap.add_argument("--window", type=int, nargs=2, default=(30, 422), metavar=("Y0", "X0"),
                    help="window origin (y, x); the default is Figure 4's 48-voxel window enlarged to 64")
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(a.cache), exist_ok=True)
    y0, x0 = a.window
    key = f"{a.base}_z{a.slice}_y{y0}_x{x0}_n{a.size}_objects"      # '_objects': the cache also holds row 3
    C = N = None
    if not a.recompute and os.path.exists(a.cache):
        with np.load(a.cache, allow_pickle=False) as Z:
            if str(Z["key"]) == key:
                C = {k: Z[k] for k in Z.files if k not in ("key", "numbers")}
                N = json.loads(str(Z["numbers"]))
                say("loaded the cached whole-volume results:", a.cache)
    if C is None:
        say("backend:", "gpu" if cupy_available() else "cpu")
        C, N = compute(a.data, a.base, a.slice, y0, x0, a.size)
        np.savez_compressed(a.cache, key=np.array(key), numbers=np.array(json.dumps(N)), **C)
        say("cached:", a.cache)
    out_png = os.path.join(a.out, "S7_spacing_number.png")
    out_svg = os.path.join(a.out, "S7_spacing_number.svg")
    N = make_figure(C, N, out_png, out_svg)
    with open(os.path.join(a.out, "S7_spacing_number_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(N, f, indent=1)
    say("written:", out_png, out_svg)


if __name__ == "__main__":
    main()
