"""S6_dt_params.py -- Supplementary Figure S6 of the ipldt / ORMIR-BQRL manuscript: the parameters of the
distance-transform engine (IPL /dt_thickness, /dt_spacing, /dt_number), one panel per parameter or rule.

Row 1  A  the sweep order of the vector distance transform (schematic): two passes over the slices, four
          boustrophedon sub-scans per slice, the mirrored order in the second pass
       B  the half-voxel surface distance s = |p(v)| (schematic, one vector)
       C  the vectors v and the surface distance s on the scan (real data, a 16 x 16-voxel window); IPL's
          exported distance maps (metric 31 = floor(s + 1/2), metric 32 = rint(|v|)) compared on every object voxel
Row 2  D  ridge_epsilon: the containment test (schematic), the centres kept at 0 / 0.5 / 0.9 on the scan, and the
          number of centres against ridge_epsilon (ipldt and IPL's own export at each value)
Row 3  E  peel_iter: the rendered contour eroded 0 (= -1, automatic) / 1 / 3 times, the centres discarded by the
          peel, and the number of centres kept against peel_iter (ipldt and IPL's export)
       F  suppress_boundary: voxels of IPL's own outputs that differ between 0, 1, 2 and 3 (none)
Row 4  G  the diameter rule (-version) on a synthetic phantom: version 1 (round 2s), version 3 accepting the
          pinned sphere (|M| < 1), version 3 declining it (|M| >= 1) with version 2's cone gate
       H  version 3 against version 1 at the centres of the scan, and the three versions against IPL's export
Row 5  I  assign_epsilon: the drawing tolerance (schematic), the map at 0 / 0.5 / 1.0 on the scan, and the map
          support against assign_epsilon (ipldt and IPL's export)

Real-data panels use one XtremeCT II patella scan (PFJ-0be66a_R, X2420448): the object is IPL's trabecular
segmentation inside IPL's trabecular contour (rendered by ipldt), in the window of Figure 4 (slice 84, 48 x 48 voxels from (y, x) = (38, 430)).  Every computation is the package's own stage
function (ipldt.field.sir_quad, ipldt.core surface_distance / ridge / peel_gobj / diameters / draw_spheres, GPU
kernels when CuPy is available); the standard chain is asserted equal to the public API (ipldt.dt_thickness)
and to IPL's delivered map.  IPL's own exports at each parameter value (dt_ridge / dt_thickness /
distance_map outputs of IPL V5.42 on the same inputs) are read from --ipl-exports when present and compared
voxel for voxel; the counts drawn are whole-scan counts.

Run from the repository root in the `ormir` environment:

    set PYTHONUTF8=1
    python manuscript/figures/supp/S6_dt_params.py
        [--data <IPLDT_LAB_ROOT>/PFJOA/XCT_masks_full_grab/PFJ-0be66a_R] [--base X2420448]
        [--ipl-exports <IPLDT_LAB_ROOT>/Python/scripts/IPL/decompressed] [--slice 84] [--window 38 430]
        [--recompute] [--out manuscript/figures/supp]

Writes S6_dt_params.png (300 dpi, 180 mm wide), S6_dt_params.svg and S6_dt_params_numbers.json (every number
drawn); the computed arrays are cached in manuscript/figures/cache/S6_dt_params_cache.npz (--recompute rebuilds).
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
from matplotlib import colors, patches  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
warnings.filterwarnings("ignore", message="CUDA path could not be detected")
import ipldt  # noqa: E402
from ipldt.core import ridge, surface_distance, surface_vector, diameters, draw_spheres, peel_gobj  # noqa: E402
from ipldt.field import sir_quad, SNAKE_SUBS, SNAKE_SUBS2  # noqa: E402
from ipldt.gpu import cupy_available  # noqa: E402

DATA_DEFAULT = lab_path("PFJOA/XCT_masks_full_grab/PFJ-0be66a_R")
BASE_DEFAULT = "X2420448"
IPL_EXPORTS_DEFAULT = lab_path("Python/scripts/IPL/decompressed")
CACHE = os.path.join(REPO, "manuscript", "figures", "cache", "S6_dt_params_cache.npz")
T0 = time.time()

RIDGE_EPS, ASSIGN_EPS, PEEL_ITER, VERSION, SUPPRESS = 0.9, 0.5, -1, 3, 2       # IPL's standard values
RIDGE_SWEEP = (0.0, 0.25, 0.5, 0.9)
ASSIGN_SWEEP = (0.0, 0.25, 0.5, 1.0)
PEEL_SWEEP = (0, 1, 3)
# IPL's own exports on the same inputs (file stem suffixes under --ipl-exports): dt_ridge outputs hold the diameter
# at every centre (1..254) and 255 on the other object voxels; dt_thickness outputs are the map; distance_map
# outputs are the metric-31 / metric-32 distance maps.
IPL_RIDGE = {0.0: "RGE00", 0.25: "RGE25", 0.5: "RIDGE05", 0.9: "RGV3"}
IPL_VERSION = {1: "RGV1", 2: "RGV2", 3: "RGV3"}
IPL_PEEL = {0: "P12PI0", 1: "P12PI1", 3: "P12PI3"}
IPL_ASSIGN = {0.0: "P12AE00", 0.25: "P12AE25", 0.5: "P12CTRL", 1.0: "P12AE10"}
IPL_SUPPRESS_RIDGE = {0: "P12SB0", 1: "P12SB1", 2: "P12PI0", 3: "P12SB3"}
IPL_SUPPRESS_MAP = {0: "P12TH0", 2: "P12CTRL"}
IPL_DM = {31: "DM31", 32: "DM32"}


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def fmt(n):
    """Thousands separated by a thin space, as in the other figures."""
    return f"{int(n):,d}"


# ------------------------------------------------------------------------------------------ style
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442", blue="#0072B2", verm="#D55E00",
          purple="#CC79A7", black="#000000")
BG = np.array([1.0, 1.0, 1.0])
OBJ = np.array([0.86, 0.86, 0.86])            # object voxels
OBJ_DARK = np.array([0.62, 0.62, 0.62])
CONTOUR = "#000000"
INK, INK2, GRID = "#1a1a1a", "#4d4d4d", "#e3e3e3"
PT = 7.0                                      # smallest font at print size (points)
plt.rcParams.update({"font.family": "Arial", "font.size": PT, "axes.linewidth": 0.5, "svg.fonttype": "none",
                     "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42,
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                     "mathtext.bf": "Arial:bold", "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.labelsize": PT, "ytick.labelsize": PT, "axes.labelsize": PT})
W_MM = 180.0
MM = 1.0 / 25.4


def rgb(c):
    return np.array(colors.to_rgb(c))


def diam_cmap(vmax):
    base = plt.get_cmap("viridis", vmax)
    cm = colors.ListedColormap(base(np.arange(vmax)))
    norm = colors.BoundaryNorm(np.arange(0.5, vmax + 1.5, 1.0), cm.N)
    return cm, norm


# ------------------------------------------------------------------------------------------ engine helpers
def ridge_any(obj, V, eps):
    if cupy_available():
        from ipldt.gpu import ridge_gpu
        return ridge_gpu(obj, V, eps)
    return ridge(obj, V, eps)


def draw_any(z, y, x, D, shape, ae):
    if cupy_available():
        from ipldt.gpu import draw_spheres_gpu
        return draw_spheres_gpu(z, y, x, D, shape, ae)
    return draw_spheres(z, y, x, D, shape, ae)


def read_export(folder, base, stem, dim, pos):
    """One IPL export aligned to the working grid, or None when the file is absent."""
    p = os.path.join(folder, f"{base}_{stem}_decompressed.AIM")
    if not os.path.exists(p):
        return None
    return np.asarray(ipldt.align_to(ipldt.read_aim(p), dim, pos))


def ridge_export(E):
    """Centres and diameters of a dt_ridge export: values 1..254 are diameters at centres."""
    cen = (E >= 1) & (E <= 254)
    return cen, E.astype(np.int64)


def compare_centres(cen_ours, D_ours, E):
    """(missing, extra, diameters differing) between ipldt's centres/diameters and a dt_ridge export."""
    if E is None:
        return None
    cen_ipl, val = ridge_export(E)
    both = cen_ours & cen_ipl
    d_diff = int((D_ours[both] != val[both]).sum())
    return dict(ipl_centres=int(cen_ipl.sum()), missing=int((cen_ipl & ~cen_ours).sum()),
                extra=int((cen_ours & ~cen_ipl).sum()), diameters_differing=d_diff)


# ------------------------------------------------------------------------------------------ compute
def compute(data, base, ipl_dir, z, y0, x0, n, n_vec):
    p = lambda nm: os.path.join(data, f"{base}_{nm}_decompressed.AIM")  # noqa: E731
    A = ipldt.read_aim(p("SEG"))
    dim, pos, el = A["dim"], A["pos"], float(A["el_size_mm"][0])
    L = lambda nm: np.asarray(ipldt.align_to(ipldt.read_aim(p(nm)), dim, pos))  # noqa: E731
    tseg = np.ascontiguousarray(L("TRAB_SEG") > 0)
    ipl_map_std = L("TRAB_TH_old").astype(np.int16)
    mraw = ipldt.read_aim(p("TRAB_MASK"))
    G_own = ipldt.render_volume(np.asarray(mraw["data"]) > 0)
    G = np.asarray(ipldt.align_to(dict(data=G_own.astype(np.uint8), dim=mraw["dim"], pos=mraw["pos"]), dim, pos)) > 0
    say(f"grid {tuple(tseg.shape)} (z, y, x), voxel {el:.4f} mm; trabecular segmentation {fmt(tseg.sum())} voxels, "
        f"{fmt((tseg & ~G).sum())} outside the rendered contour ({fmt(G.sum())} voxels)")
    have_ipl = os.path.isdir(ipl_dir)
    E = (lambda stem: read_export(ipl_dir, base, stem, dim, pos)) if have_ipl else (lambda stem: None)
    num = dict(voxel_mm=el, grid_zyx=list(tseg.shape), voxels=int(tseg.size), object_voxels=int(tseg.sum()),
               contour_voxels=int(G.sum()), object_outside_contour=int((tseg & ~G).sum()),
               backend="gpu" if cupy_available() else "cpu", ipl_exports_present=have_ipl)
    sl = (z, slice(y0, y0 + n), slice(x0, x0 + n))
    arrays = dict(tseg=tseg[sl], G=G[sl])

    # ---- stages 1-2: the field and the surface distance
    V = sir_quad(tseg)
    s = surface_distance(V)
    s[~tseg] = 0.0
    say(f"field done; s max {float(s.max()):.3f} voxels")
    arrays["s"] = s[sl]
    arrays["V"] = V[(slice(None),) + sl]
    dm = {}
    for metric, stem in IPL_DM.items():
        Em = E(stem)
        if Em is None:
            continue
        ours = np.floor(s + 0.5).astype(np.int64) if metric == 31 else \
            np.rint(np.sqrt((V.astype(np.int64) ** 2).sum(axis=0))).astype(np.int64)
        ours[~tseg] = 0
        dm[str(metric)] = dict(differing=int((ours[tseg] != Em.astype(np.int64)[tseg]).sum()), compared=int(tseg.sum()),
                               rule="floor(s + 1/2)" if metric == 31 else "rint(|v|)")
        say(f"distance_map metric {metric} vs IPL: {dm[str(metric)]['differing']} of {fmt(tseg.sum())} object voxels differ")
    num["distance_map"] = dm

    # ---- stage 3: the ridge at several epsilons (centres inside the contour, peel 0)
    ridge_rows = {}
    cen09 = None
    for eps in RIDGE_SWEEP:
        cen = ridge_any(tseg, V, eps) & G
        zc, yc, xc = np.nonzero(cen)
        D3 = diameters(tseg, V, zc, yc, xc, version=3)
        Dmap = np.zeros(tseg.shape, np.int64)
        Dmap[zc, yc, xc] = D3
        cmp_ = compare_centres(cen, Dmap, E(IPL_RIDGE[eps]))
        ridge_rows[eps] = dict(centres=int(cen.sum()), ipl=cmp_)
        arrays[f"cen_{eps}"] = cen[sl]
        say(f"ridge_epsilon {eps}: {fmt(cen.sum())} centres; IPL export {cmp_}")
        if eps == RIDGE_EPS:
            cen09, D3_09, Dmap09 = cen, D3, Dmap
            zc09, yc09, xc09 = zc, yc, xc
    num["ridge"] = {str(k): v for k, v in ridge_rows.items()}

    # ---- stage 4: the peel
    peel_rows = {}
    for k in PEEL_SWEEP:
        Gk = peel_gobj(G, k)
        cenk = cen09 & Gk
        Dk = np.where(cenk, Dmap09, 0)
        cmp_ = compare_centres(cenk, Dk, E(IPL_PEEL[k]))
        peel_rows[k] = dict(centres=int(cenk.sum()), removed=int((cen09 & ~cenk).sum()), ipl=cmp_)
        arrays[f"Gpeel_{k}"] = Gk[sl]
        say(f"peel_iter {k}: {fmt(cenk.sum())} centres kept, {fmt((cen09 & ~cenk).sum())} removed; IPL export {cmp_}")
    num["peel"] = {str(k): v for k, v in peel_rows.items()}
    r_auto = ipldt.dt_thickness(tseg, {"rendered": G}, voxel_size_mm=el)        # peel_iter -1 (automatic)
    num["peel_auto_equals_0"] = bool((r_auto.centres == cen09).all())
    say(f"peel_iter -1 (automatic) gives the peel-0 centres: {num['peel_auto_equals_0']}")

    # ---- stage 5: the diameter versions at the standard centres
    ver_rows = {}
    Dv = {ver: diameters(tseg, V, zc09, yc09, xc09, version=ver) for ver in (1, 2, 3)}
    for ver in (1, 2, 3):
        Dm = np.zeros(tseg.shape, np.int64)
        Dm[zc09, yc09, xc09] = Dv[ver]
        arrays[f"D_v{ver}"] = Dm[sl]
        cmp_ = compare_centres(cen09, Dm, E(IPL_VERSION[ver]))
        m = draw_any(zc09, yc09, xc09, Dv[ver], tseg.shape, ASSIGN_EPS)
        m[~tseg] = 0
        m[~G] = 0
        vals = m[m > 0]
        ver_rows[ver] = dict(mean_mm=float(vals.mean() * el), support=int(vals.size), max_diameter=int(m.max()),
                             differs_from_v3=int((Dv[ver] != Dv[3]).sum()), ipl=cmp_)
        say(f"version {ver}: Tb.Th {ver_rows[ver]['mean_mm']:.6f} mm, {fmt(vals.size)} voxels; differs from version 3 at "
            f"{fmt((Dv[ver] != Dv[3]).sum())} of {fmt(cen09.sum())} centres; IPL export {cmp_}")
    d31 = (Dv[3] - Dv[1]).astype(np.int64)
    num["version"] = {str(k): v for k, v in ver_rows.items()}
    num["v3_minus_v1"] = {str(int(u)): int((d31 == u).sum()) for u in np.unique(d31)}
    num["centres_standard"] = int(cen09.sum())

    # ---- stage 6: the drawing tolerance
    ae_rows = {}
    for ae in ASSIGN_SWEEP:
        m = draw_any(zc09, yc09, xc09, Dv[3], tseg.shape, ae)
        m[~tseg] = 0
        m[~G] = 0
        Em = E(IPL_ASSIGN[ae])
        vals = m[m > 0]
        ae_rows[ae] = dict(support=int(vals.size), mean_mm=float(vals.mean() * el), max_diameter=int(m.max()),
                           ipl=None if Em is None else dict(differing=int((m != Em.astype(np.int16)).sum()),
                                                            ipl_support=int((Em > 0).sum())))
        arrays[f"map_{ae}"] = m[sl]
        say(f"assign_epsilon {ae}: support {fmt(vals.size)}, Tb.Th {ae_rows[ae]['mean_mm']:.6f} mm; IPL {ae_rows[ae]['ipl']}")
        if ae == ASSIGN_EPS:
            m_std = m
    num["assign"] = {str(k): v for k, v in ae_rows.items()}
    num["standard_map_vs_delivered"] = int((m_std != ipl_map_std).sum())
    num["standard_map_vs_public_api"] = int((m_std != r_auto.map).sum())
    say(f"standard map vs IPL's delivered map: {num['standard_map_vs_delivered']} differing voxels; vs public API: "
        f"{num['standard_map_vs_public_api']}")
    arrays["map_ipl"] = ipl_map_std[sl]

    # ---- suppress_boundary: IPL's own outputs against each other and against ipldt
    sb = {}
    ref_r = E(IPL_SUPPRESS_RIDGE[2])
    for k, stem in IPL_SUPPRESS_RIDGE.items():
        Ek = E(stem)
        if Ek is None or ref_r is None:
            continue
        sb[f"ridge_{k}"] = dict(vs_standard_ipl=int((Ek != ref_r).sum()),
                                vs_ipldt=compare_centres(cen09, Dmap09, Ek))
    ref_m = E(IPL_SUPPRESS_MAP[2])
    for k, stem in IPL_SUPPRESS_MAP.items():
        Ek = E(stem)
        if Ek is None or ref_m is None:
            continue
        sb[f"map_{k}"] = dict(vs_standard_ipl=int((Ek != ref_m).sum()), vs_ipldt=int((Ek.astype(np.int16) != m_std).sum()))
    num["suppress_boundary"] = sb
    say(f"suppress_boundary: {sb}")

    num["window"] = dict(slice_index=z, origin_yx=[y0, x0], size=n, vector_window=n_vec)
    return num, arrays


# ------------------------------------------------------------------------------------------ synthetic phantom (diameter rule)
def phantom_2d(seed, H=44, W=44):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W), bool)
    for _ in range(5):
        cx, cy = rng.uniform(6, W - 6), rng.uniform(6, H - 6)
        ang = rng.uniform(0, np.pi)
        Lr, t = rng.uniform(12, 24), rng.uniform(1.0, 3.2)
        u = np.array([np.cos(ang), np.sin(ang)])
        d = (xx - cx) * u[0] + (yy - cy) * u[1]
        e = -(xx - cx) * u[1] + (yy - cy) * u[0]
        img |= (np.abs(d) <= Lr / 2) & (np.abs(e) <= t)
    for _ in range(3):
        cx, cy, r = rng.uniform(6, W - 6), rng.uniform(6, H - 6), rng.uniform(2.0, 4.0)
        img |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return img


def diameter_quantities(obj, V, z, y, x):
    """The intermediate quantities of the diameter rule at the listed voxels (for drawing only; the
    resulting V1 / V2 / V3 are asserted equal to ipldt.diameters)."""
    N = np.array(obj.shape)
    v = np.stack([V[i][z, y, x].astype(np.float64) for i in range(3)], axis=1)
    P1 = surface_vector(v)
    s = np.linalg.norm(P1, axis=1)
    V1 = np.floor(2 * s + 0.5 + 1e-9).astype(int)
    n1 = s.copy()
    n1[n1 == 0] = 1
    step = -np.rint(P1 / n1[:, None])
    q = np.stack([z, y, x], axis=1) + step.astype(int)
    inside = ((q >= 0) & (q < N)).all(axis=1)
    qc = tuple(np.clip(q[:, i], 0, N[i] - 1) for i in range(3))
    bg_y = ~obj[qc] | ~inside
    vy = np.stack([V[i][qc].astype(np.float64) for i in range(3)], axis=1)
    vy[bg_y] = 0
    Py = surface_vector(vy)
    P2 = step + Py
    D = np.linalg.norm(P1 - P2, axis=1)
    M = (P1 + P2) / 2
    R = np.floor(D + 0.5 + 1e-9).astype(int)
    nPy = np.linalg.norm(Py, axis=1)
    cosg = np.where(nPy > 0, -(Py * P1).sum(1) / (np.maximum(nPy, 1e-12) * np.maximum(s, 1e-12)), -2.0)
    V2 = np.where(inside & ~bg_y & (cosg >= 0.5 - 1e-9), R, V1)
    V3 = np.where(inside & (np.linalg.norm(M, axis=1) < 1 - 1e-9), R, V1)
    return dict(v=v, P1=P1, s=s, V1=V1, step=step, y_obj=~bg_y, vy=vy, Py=Py, P2=P2, D=D, M=M, R=R, cosg=cosg, V2=V2, V3=V3)


def diameter_cases():
    """The phantom (first seed that provides all three cases) and one voxel per panel."""
    for seed in range(3, 60):
        img = phantom_2d(seed)
        obj = img[None].copy()
        V = sir_quad(obj)
        cen = ridge(obj, V, 0.0)
        z, y, x = np.nonzero(cen)
        Q = diameter_quantities(obj, V, z, y, x)
        for ver, key in ((1, "V1"), (2, "V2"), (3, "V3")):
            assert (ipldt.diameters(obj, V, z, y, x, version=ver) == Q[key]).all()
        Mn = np.linalg.norm(Q["M"], axis=1)
        interior = (y >= 6) & (y < img.shape[0] - 6) & (x >= 6) & (x < img.shape[1] - 6)
        cosP = (Q["P1"] * Q["P2"]).sum(1) / np.maximum(np.linalg.norm(Q["P1"], axis=1) * np.linalg.norm(Q["P2"], axis=1), 1e-9)
        oblique = np.abs(cosP + 1) > 0.05

        def pick(mask, key):
            idx = np.flatnonzero(mask)
            return None if idx.size == 0 else int(idx[np.argmax(key[idx])])

        ia = pick(interior & Q["y_obj"] & (Mn < 1) & (Q["V3"] != Q["V1"]) & oblique & (Q["s"] >= 1.0) & (Q["s"] <= 2.6), Mn)
        ib = pick(interior & Q["y_obj"] & (Mn >= 1) & (Q["R"] != Q["V1"]) & (Q["cosg"] < 0.5) & (Q["s"] >= 1.0) & (Q["s"] <= 2.6),
                  -np.abs(Mn - 1.3))
        if ia is not None and ib is not None:
            return dict(seed=seed, img=img, y=y, x=x, Q=Q, Mn=Mn, cases=(ia, ib))
    raise RuntimeError("no phantom seed provides the diameter cases")


# ------------------------------------------------------------------------------------------ drawing helpers
class Sheet:
    """Millimetre placement on a fixed-width figure."""

    def __init__(self, w_mm, h_mm):
        self.W, self.H = w_mm, h_mm
        self.fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))

    def ax(self, x, y_top, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1.0 - (y_top + h) / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y_top, s, **kw):
        kw.setdefault("va", "top")
        kw.setdefault("ha", "left")
        return self.fig.text(x / self.W, 1.0 - y_top / self.H, s, **kw)

    def letter(self, x, y_top, s):
        self.fig.text(x / self.W, 1.0 - y_top / self.H, s, ha="left", va="top", fontsize=9, fontweight="bold", color=INK)


def crop_axes(sh, x, y, w, ext, frame_lw=0.5, frame_color="#9a9a9a"):
    ax = sh.ax(x, y, w, w)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(frame_color)
        sp.set_linewidth(frame_lw)
    return ax


def standard_badge(ax):
    ax.text(0.97, 0.96, "IPL standard", transform=ax.transAxes, ha="right", va="top", fontsize=PT - 0.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=INK, lw=0.5))
    for sp in ax.spines.values():
        sp.set_edgecolor(INK)
        sp.set_linewidth(1.1)


def contour_line(ax, mask2d, ext, color=CONTOUR, lw=0.6, ls="-"):
    n = mask2d.shape[0]
    xs = np.arange(ext[0] + 0.5, ext[0] + 0.5 + n)
    ys = np.arange(ext[3] + 0.5, ext[3] + 0.5 + n)
    ax.contour(xs, ys, mask2d.astype(float), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


CAP_PT = 6.2


def caption(sh, x, y_top, w, lines, color="#222222"):
    sh.text(x + w / 2, y_top, "\n".join(lines), ha="center", va="top", fontsize=CAP_PT, color=color, linespacing=1.15)


def header(sh, x, y_top, w, s, indent=0.0):
    sh.text(x + indent + (w - indent) / 2, y_top, s, ha="center", va="top", fontsize=PT, fontweight="bold", color=INK)


def chart_axes(sh, x, y, w, h):
    ax = sh.ax(x + 9.0, y + 1.0, w - 9.6, h - 8.5)
    ax.tick_params(length=1.5, width=0.5, pad=1.5, labelsize=PT - 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    return ax


def sweep_grid(ax, yd, xd, n=5):
    ax.set_xlim(-0.9, n - 0.1)
    ax.set_ylim(n - 0.1, -0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    for yy in range(n):
        for xx in range(n):
            ax.add_patch(patches.Rectangle((xx - 0.5, yy - 0.5), 1, 1, fc="#f4f4f4", ec="#c8c8c8", lw=0.4))
    rows = range(n) if yd > 0 else range(n - 1, -1, -1)
    for i, yy in enumerate(rows):
        x0, x1 = (-0.35, n - 0.65) if xd > 0 else (n - 0.65, -0.35)
        ax.annotate("", xy=(x1, yy), xytext=(x0, yy), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=OI["blue"], mutation_scale=5))
        ax.text(-0.75 if xd > 0 else n - 0.25, yy, str(i + 1), fontsize=PT - 2, va="center", ha="center", color=OI["blue"])
    ax.annotate("", xy=(n - 0.15, n - 0.75 if yd > 0 else -0.25), xytext=(n - 0.15, -0.25 if yd > 0 else n - 0.75),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=OI["verm"], mutation_scale=6))


def voxel_grid(ax, img2d, x0, x1, y0, y1, obj_color=OBJ, ec="#d0d0d0"):
    for yy in range(y0, y1):
        for xx in range(x0, x1):
            inside = 0 <= yy < img2d.shape[0] and 0 <= xx < img2d.shape[1]
            on = inside and bool(img2d[yy, xx])
            ax.add_patch(patches.Rectangle((xx - 0.5, yy - 0.5), 1, 1, fc=obj_color if on else "white", ec=ec, lw=0.4))


# ------------------------------------------------------------------------------------------ the figure
def make_figure(num, arr, cases, out_png, out_svg):
    el = num["voxel_mm"]
    z, (y0, x0), n = num["window"]["slice_index"], num["window"]["origin_yx"], num["window"]["size"]
    ext = (x0 - 0.5, x0 + n - 0.5, y0 + n - 0.5, y0 - 0.5)
    tseg, G = arr["tseg"], arr["G"]
    base_img = np.where(tseg[..., None], OBJ, BG)
    n_vox = num["voxels"]

    # ---- geometry (mm)
    left, right, gap = 5.5, 1.5, 2.2
    ncol = 5
    w = (W_MM - left - right - (ncol - 1) * gap) / ncol
    xs = [left + i * (w + gap) for i in range(ncol)]
    hdr, cap, rgap = 3.4, 7.2, 1.2
    top = 1.5
    row1_h = 40.0
    rows_y = {1: top}
    y = top + hdr + row1_h + rgap
    for r in (2, 3, 4, 5):
        rows_y[r] = y
        y += hdr + w + cap + rgap
    cbar_y = y - rgap + 0.4                      # the diameter colour bar strip under row 5
    H_MM = cbar_y + 8.0
    sh = Sheet(W_MM, H_MM)
    drawn = {}
    ipl_line = lambda d: f"IPL export: {d} differ"  # noqa: E731

    def crop_caption(x, y_top, lines):
        caption(sh, x, y_top + hdr + w + 0.6, w, lines)

    def std_label(ax, xv, where, text="IPL standard", pad=0.03):
        """Rotated label beside the dotted line at the parameter's standard value."""
        lo, hi = ax.get_ylim()
        xlo, xhi = ax.get_xlim()
        dx = (xhi - xlo) * pad
        if where == "top-left":
            ax.text(xv - dx, hi - (hi - lo) * 0.03, text, rotation=90, fontsize=PT - 1.0, color=INK2, ha="right", va="top")
        elif where == "top-right":
            ax.text(xv + dx, hi - (hi - lo) * 0.03, text, rotation=90, fontsize=PT - 1.0, color=INK2, ha="left", va="top")
        elif where == "bottom-left":
            ax.text(xv - dx, lo + (hi - lo) * 0.03, text, rotation=90, fontsize=PT - 1.0, color=INK2, ha="right", va="bottom")
        else:
            ax.text(xv + dx, lo + (hi - lo) * 0.03, text, rotation=90, fontsize=PT - 1.0, color=INK2, ha="left", va="bottom")

    # =========================================================== row 1: A sweep order, B surface distance, C real field
    yA = rows_y[1]
    xA, wA = xs[0], 3 * w + 2 * gap
    sh.letter(xA - 4.0, yA - 0.6, "A")
    header(sh, xA, yA, wA, "Sweep order of the vector distance transform")
    gy = yA + hdr + 0.3
    mini = 11.0
    for r, (label, subs) in enumerate((("pass 1: slices z ascending", SNAKE_SUBS), ("pass 2: slices z descending", SNAKE_SUBS2))):
        ry = gy + r * (mini + 3.4)
        sh.text(xA, ry + 0.3, label, fontsize=PT - 0.5, color=INK)
        for k, (yd, xd) in enumerate(subs):
            gx = xA + 35.0 + k * (mini + 5.6)
            ax = sh.ax(gx, ry, mini, mini)
            sweep_grid(ax, yd, xd)
            sh.text(gx + mini / 2, ry + mini + 0.3, f"{k + 1}: y {'↑' if yd > 0 else '↓'}, x {'→' if xd > 0 else '←'}",
                    ha="center", fontsize=PT - 1.0, color=INK2)
    sh.text(xA, gy + 2 * (mini + 3.4) + 0.4,
            "In each sub-scan every object voxel takes the shortest of the 7 candidates behind the scan\n"
            "(a neighbor's vector plus the offset to it), replacing its own vector only when the candidate is\n"
            "strictly shorter. Outside the image is object. Two passes; the sub-scan order of pass 2 mirrors pass 1.",
            fontsize=PT - 1.0, color=INK2, linespacing=1.2)
    drawn["A"] = dict(pass1=SNAKE_SUBS, pass2=SNAKE_SUBS2)

    # B: surface distance schematic, v = (2, 1)
    xB = xs[3]
    sh.letter(xB - 1.2, yA - 0.6, "B")
    header(sh, xB, yA, w, "Surface distance s = |p(v)|", indent=2.5)
    ax = sh.ax(xB, yA + hdr, w, 29.0)
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 3.0)
    ax.set_ylim(2.15, -1.0)
    ax.axis("off")
    vx, vy = 2, 1
    for yy in range(-1, 3):
        for xx in range(-1, 3):
            on = not (xx == vx and yy == vy)
            ax.add_patch(patches.Rectangle((xx - 0.5, yy - 0.5), 1, 1, fc=OBJ if on else "white", ec="#c8c8c8", lw=0.4))
    ax.add_patch(patches.Rectangle((vx - 0.5, vy - 0.5), 1, 1, fc="white", ec=OI["verm"], lw=1.0, hatch="////"))
    ax.add_patch(patches.Rectangle((-0.5, -0.5), 1, 1, fc="#cfe3f3", ec=OI["blue"], lw=0.9))
    ax.plot(0, 0, "o", color=OI["blue"], ms=3)
    ax.annotate("", xy=(vx, vy), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#777777", ls="--", mutation_scale=6))
    pv = surface_vector(np.array([vx, vy], float))
    ax.annotate("", xy=(pv[0], pv[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.3, color=OI["blue"], mutation_scale=7))
    ax.plot(pv[0], pv[1], "o", color=OI["green"], ms=4, mec="black", mew=0.4)
    ax.text(-0.42, -0.62, "x", fontsize=PT, color=OI["blue"], fontweight="bold")
    ax.text(vx, vy + 0.72, "c = x + v", fontsize=PT - 0.5, color=OI["verm"], ha="center", va="top")
    ax.text(pv[0] - 0.05, pv[1] + 0.28, "p(v)", fontsize=PT - 0.5, color=OI["green"], ha="right", va="top")
    ax.text(0.9, 0.05, "v = (2, 1)\n|v| = 2.24", fontsize=PT - 1, color="#555555", va="bottom", ha="left")
    ax.text(-0.9, 1.75, f"s = |p(v)| = {np.linalg.norm(pv):.2f}", fontsize=PT - 0.5, color=OI["blue"], fontweight="bold", va="top")
    caption(sh, xB, yA + hdr + 32.6, w, ["p(u) = sign(u)·max(|u| − ½, 0)",
                                          "per component; s = distance",
                                          "to the contact voxel's cube"])
    drawn["B"] = dict(v=[vx, vy], p=[float(pv[0]), float(pv[1])], s=float(np.linalg.norm(pv)), v_norm=float(np.hypot(vx, vy)))

    # C: real vectors + s on a small window inside the crop
    xC = xs[4]
    sh.letter(xC - 1.2, yA - 0.6, "C")
    header(sh, xC, yA, w, "v and s on the scan", indent=2.5)
    nv = num["window"]["vector_window"]
    best = None
    for yy in range(0, n - nv + 1, 2):
        for xx in range(0, n - nv + 1, 2):
            f = tseg[yy:yy + nv, xx:xx + nv].mean()
            sc = -abs(f - 0.5)
            if best is None or sc > best[0]:
                best = (sc, yy, xx)
    _, vy0, vx0 = best
    subw = (slice(vy0, vy0 + nv), slice(vx0, vx0 + nv))
    ext_v = (x0 + vx0 - 0.5, x0 + vx0 + nv - 0.5, y0 + vy0 + nv - 0.5, y0 + vy0 - 0.5)
    wc = 25.0
    ax = crop_axes(sh, xC + (w - wc) / 2, yA + hdr, wc, ext_v)
    s_sub = arr["s"][subw]
    smax = float(arr["s"].max())
    img = np.where(tseg[subw][..., None], OBJ, BG).copy()
    objm = tseg[subw]
    img[objm] = plt.get_cmap("viridis")(s_sub[objm] / smax)[:, :3]
    ax.imshow(img, interpolation="nearest", extent=ext_v)
    Vs = arr["V"][:, vy0:vy0 + nv, vx0:vx0 + nv]
    yy, xx = np.nonzero(objm)
    ax.quiver(xx + x0 + vx0, yy + y0 + vy0, Vs[2][yy, xx], Vs[1][yy, xx], angles="xy", scale_units="xy", scale=1,
              width=0.007, headwidth=3.5, headlength=4, headaxislength=3.5, color="white", edgecolor="black", linewidth=0.3)
    ax.set_xlim(ext_v[0], ext_v[1])
    ax.set_ylim(ext_v[2], ext_v[3])
    cax = sh.ax(xC + (w - wc) / 2 + 3.0, yA + hdr + wc + 0.8, wc - 6.0, 1.5)
    cb = sh.fig.colorbar(ScalarMappable(norm=colors.Normalize(0, smax), cmap="viridis"), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 1, smax])
    cb.set_ticklabels(["0", "1", f"{smax:.1f}"])
    cb.ax.tick_params(labelsize=PT - 1.0, length=1.2, pad=0.8, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label("s (voxels)", fontsize=PT - 1.0, labelpad=0.8)
    dm = num.get("distance_map", {})
    cl = [f"{nv} × {nv} voxels; arrows = v"]
    if dm:
        cl += ["IPL distance maps 31 / 32:",
               f"{dm['31']['differing']} / {dm['32']['differing']} of {fmt(dm['31']['compared'])} differ"]
    caption(sh, xC, yA + hdr + 32.6, w, cl)
    drawn["C"] = dict(sub_window_origin_yx=[y0 + vy0, x0 + vx0], size=nv, s_max_window=smax, distance_map=dm)

    # =========================================================== row 2: D ridge_epsilon
    yD = rows_y[2]
    sh.letter(xs[0] - 4.0, yD - 0.6, "D")
    header(sh, xs[0], yD, w, "Containment test")
    ax = sh.ax(xs[0], yD + hdr, w, w)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-3.4, 4.6)
    ax.set_ylim(4.0, -4.0)
    sx, sy_, sep, eps = 1.5, 2.5, 1.0, 0.9
    X = np.array([-0.6, 0.0])
    Y = X + np.array([sep, 0.0])
    ax.add_patch(patches.Circle(Y, sy_ + eps, fc="none", ec=OI["orange"], lw=0.9, ls=":"))
    ax.add_patch(patches.Circle(Y, sy_, fc="#fbe9cc", ec=OI["orange"], lw=1.0))
    ax.add_patch(patches.Circle(X, sx, fc="#cfe3f3", ec=OI["blue"], lw=1.0, alpha=0.9))
    ax.plot(*X, "o", color=OI["blue"], ms=3.5)
    ax.plot(*Y, "s", color=OI["orange"], ms=3.5)
    ax.text(X[0] - 0.15, X[1] - 0.15, "x", fontsize=PT, color=OI["blue"], fontweight="bold", ha="right", va="bottom")
    ax.text(Y[0], Y[1] - 0.3, "y", fontsize=PT, color=OI["orange"], fontweight="bold", ha="center", va="bottom")
    ax.annotate("", xy=(X[0], X[1] - sx), xytext=(X[0], X[1]), arrowprops=dict(arrowstyle="-", lw=0.7, color=OI["blue"]))
    ax.text(X[0] + 0.12, X[1] - sx / 2 - 0.15, "$s_x$", fontsize=PT - 0.5, color=OI["blue"], va="center", ha="left")
    ax.annotate("", xy=(Y[0] + sy_, Y[1]), xytext=(Y[0], Y[1]), arrowprops=dict(arrowstyle="-", lw=0.7, color=OI["orange"]))
    ax.text(Y[0] + sy_ / 2, Y[1] + 0.35, "$s_y$", fontsize=PT - 0.5, color=OI["orange"], ha="center", va="top")
    ax.annotate("", xy=(Y[0] + sy_ + eps, Y[1]), xytext=(Y[0] + sy_, Y[1]), arrowprops=dict(arrowstyle="-", lw=0.7, color=OI["orange"], ls=":"))
    ax.text(Y[0] + sy_ + eps / 2, Y[1] - 0.25, "ε", fontsize=PT - 0.5, color=OI["orange"], ha="center", va="bottom")
    ax.text(X[0] + sep / 2, X[1] + 0.35, "|x − y|", fontsize=PT - 1.0, color=INK2, ha="center", va="top")
    crop_caption(xs[0], yD, ["x discarded when a neighbor y",
                             "has |x − y| + $s_x$ − $s_y$ ≤ ε",
                             "(1 + 1.5 − 2.5 = 0 ≤ 0.9 here)"])
    for k, eps in enumerate((0.0, 0.5, 0.9)):
        x = xs[k + 1]
        header(sh, x, yD, w, f"ridge_epsilon {eps:g}")
        ax = crop_axes(sh, x, yD + hdr, w, ext)
        img = base_img.copy()
        img[arr[f"cen_{eps}"]] = rgb(OI["verm"])
        ax.imshow(img, interpolation="nearest", extent=ext)
        contour_line(ax, G, ext)
        if eps == RIDGE_EPS:
            standard_badge(ax)
        if k == 0:
            bar = 1.0 / el
            bx, by = x0 + 2.0, y0 + n - 3.5
            ax.plot([bx, bx + bar], [by, by], color="black", lw=1.4, solid_capstyle="butt")
            ax.text(bx + bar / 2, by - 1.4, "1 mm", ha="center", va="bottom", fontsize=PT - 0.5,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
        rr = num["ridge"][str(eps)]
        lines = [f"{fmt(rr['centres'])} centers"]
        if rr["ipl"] is not None:
            lines.append(ipl_line(rr["ipl"]["missing"] + rr["ipl"]["extra"] + rr["ipl"]["diameters_differing"]))
        crop_caption(x, yD, lines)
    header(sh, xs[4], yD, w, "Centers vs ridge_epsilon")
    ax = chart_axes(sh, xs[4], yD + hdr, w, w)
    e_ = list(RIDGE_SWEEP)
    c_ = [num["ridge"][str(e)]["centres"] / 1e6 for e in e_]
    ax.plot(e_, c_, "-o", color=OI["blue"], ms=3.5, lw=0.9, label="ipldt", zorder=3)
    if all(num["ridge"][str(e)]["ipl"] is not None for e in e_):
        ci = [num["ridge"][str(e)]["ipl"]["ipl_centres"] / 1e6 for e in e_]
        ax.plot(e_, ci, "s", mfc="none", mec=OI["verm"], ms=5.5, mew=0.8, label="IPL export", zorder=4)
    ax.set_xlabel("ridge_epsilon")
    ax.set_ylabel("centers (millions)")
    ax.set_xticks(e_)
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0, 6.2)
    ax.axvline(RIDGE_EPS, color=INK2, lw=0.6, ls=":")
    std_label(ax, RIDGE_EPS, "top-left")
    ax.legend(frameon=False, fontsize=PT - 1.0, loc="lower left", handletextpad=0.4, borderaxespad=0.2)
    crop_caption(xs[4], yD, [fmt(num["object_voxels"]) + " object voxels", "IPL export: 0 centers and",
                             "0 diameters differ, all values"])
    drawn["D"] = num["ridge"]

    # =========================================================== row 3: E peel_iter, F suppress_boundary
    yE = rows_y[3]
    sh.letter(xs[0] - 4.0, yE - 0.6, "E")
    for k, pk in enumerate(PEEL_SWEEP):
        x = xs[k]
        header(sh, x, yE, w, f"peel_iter {pk}" + (" (= −1)" if pk == 0 else ""))
        ax = crop_axes(sh, x, yE + hdr, w, ext)
        Gk = arr[f"Gpeel_{pk}"]
        img = base_img.copy()
        rim = G & ~Gk
        img[rim & ~tseg] = rgb("#fbe9cc")
        img[rim & tseg] = rgb("#e6c98a")
        kept = arr["cen_0.9"] & Gk
        removed = arr["cen_0.9"] & ~Gk
        img[kept] = rgb(OI["verm"])
        img[removed] = rgb(OI["blue"])
        ax.imshow(img, interpolation="nearest", extent=ext)
        contour_line(ax, G, ext)
        if pk > 0:
            contour_line(ax, Gk, ext, color=INK, lw=0.6, ls="--")
        if pk == 0:
            standard_badge(ax)
        pr = num["peel"][str(pk)]
        lines = [f"{fmt(pr['centres'])} centers kept", f"{fmt(pr['removed'])} discarded (blue)"]
        if pr["ipl"] is not None:
            lines.append(ipl_line(pr["ipl"]["missing"] + pr["ipl"]["extra"] + pr["ipl"]["diameters_differing"]))
        crop_caption(x, yE, lines)
    header(sh, xs[3], yE, w, "Centers vs peel_iter")
    ax = chart_axes(sh, xs[3], yE + hdr, w, w)
    p_ = list(PEEL_SWEEP)
    c_ = [num["peel"][str(k)]["centres"] / 1e6 for k in p_]
    ax.plot(p_, c_, "-o", color=OI["blue"], ms=3.5, lw=0.9, label="ipldt", zorder=3)
    if all(num["peel"][str(k)]["ipl"] is not None for k in p_):
        ci = [num["peel"][str(k)]["ipl"]["ipl_centres"] / 1e6 for k in p_]
        ax.plot(p_, ci, "s", mfc="none", mec=OI["verm"], ms=5.5, mew=0.8, label="IPL export", zorder=4)
    lo, hi = min(c_) * 0.985, max(c_) * 1.008
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.35, 3.35)
    ax.axvline(0, color=INK2, lw=0.6, ls=":")
    std_label(ax, 0, "bottom-right", "standard (−1 → 0)")
    ax.set_xlabel("peel_iter")
    ax.set_ylabel("centers (millions)")
    ax.set_xticks(p_)
    ax.legend(frameon=False, fontsize=PT - 1.0, loc="upper right", handletextpad=0.4, borderaxespad=0.2)
    crop_caption(xs[3], yE, ["−1 (automatic) resolves to 0", "slice-wise 4-connected erosion",
                             "IPL export: 0 differ, all values"])
    drawn["E"] = dict(num["peel"], automatic_equals_0=num["peel_auto_equals_0"])
    # F: suppress_boundary
    sh.letter(xs[4] - 1.2, yE - 0.6, "F")
    header(sh, xs[4], yE, w, "suppress_boundary", indent=2.5)
    ax = chart_axes(sh, xs[4], yE + hdr, w, w)
    sb = num["suppress_boundary"]
    vals = [0, 1, 3]
    yr = [sb.get(f"ridge_{v}", {}).get("vs_standard_ipl") for v in vals]
    ym = [sb.get(f"map_{v}", {}).get("vs_standard_ipl") for v in vals]
    ax.plot([v for v, q in zip(vals, yr) if q is not None], [q for q in yr if q is not None], "o", color=OI["blue"], ms=4.0,
            label="centers, diameters", zorder=3)
    ax.plot([v + 0.15 for v, q in zip(vals, ym) if q is not None], [q for q in ym if q is not None], "^", color=OI["green"], ms=4.0,
            label="thickness map", zorder=3)
    ax.set_xlim(-0.4, 3.4)
    ax.set_ylim(-0.08, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 1, 2, 3])
    ax.axvline(SUPPRESS, color=INK2, lw=0.6, ls=":")
    std_label(ax, SUPPRESS, "bottom-right")
    ax.set_xlabel("suppress_boundary")
    ax.set_ylabel("differing voxels")
    ax.legend(frameon=False, fontsize=PT - 1.0, loc="upper left", handletextpad=0.4, borderaxespad=0.2)
    for v, q in zip(vals, yr):
        if q is not None:
            ax.text(v, 0.1, str(q), ha="center", va="bottom", fontsize=PT - 0.5, color=INK)
    crop_caption(xs[4], yE, ["IPL's outputs at 0, 1, 3 vs 2:", f"0 of {fmt(n_vox)} voxels differ",
                             "ipldt: 0 differ from each"])
    drawn["F"] = sb

    # =========================================================== row 4: G diameter rule (synthetic), H real
    yG = rows_y[4]
    sh.letter(xs[0] - 4.0, yG - 0.6, "G")
    img_ph, yc, xc, Q, Mn = cases["img"], cases["y"], cases["x"], cases["Q"], cases["Mn"]
    ia, ib = cases["cases"]

    def draw_case(ax, i, mode):
        cy, cx = int(yc[i]), int(xc[i])
        r = 4
        ax.set_aspect("equal")
        ax.set_xlim(cx - r - 0.5, cx + r + 0.5)
        ax.set_ylim(cy + r + 0.5, cy - r - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        voxel_grid(ax, img_ph, cx - r, cx + r + 1, cy - r, cy + r + 1)
        v, P1, step, vy_, Py, P2, M, D, s = (Q[k][i] for k in ("v", "P1", "step", "vy", "Py", "P2", "M", "D", "s"))
        vX, P1X, stX, P2X, MX, PyX = (np.array([q[2], q[1]]) for q in (v, P1, step, P2, M, Py))
        Xp = np.array([cx, cy], float)
        Yp = Xp + stX
        ax.add_patch(patches.Rectangle((Xp[0] + vX[0] - 0.5, Xp[1] + vX[1] - 0.5), 1, 1, fc="none", ec=OI["verm"], lw=0.9, hatch="////"))
        if mode != "v1" and Q["y_obj"][i]:
            vyX = np.array([vy_[2], vy_[1]])
            ax.add_patch(patches.Rectangle((Yp[0] + vyX[0] - 0.5, Yp[1] + vyX[1] - 0.5), 1, 1, fc="none", ec=OI["blue"], lw=0.9, hatch="\\\\\\\\"))
        ax.add_patch(patches.Rectangle((Xp[0] - 0.5, Xp[1] - 0.5), 1, 1, fc="#ffe9a8", ec="black", lw=0.7))
        ax.plot(*Xp, "o", color="black", ms=2.5)
        ax.text(Xp[0] + 0.12, Xp[1] - 0.45, "x", fontsize=PT, fontweight="bold")
        ax.annotate("", xy=Xp + P1X, xytext=Xp, arrowprops=dict(arrowstyle="-|>", lw=1.0, color=OI["verm"], mutation_scale=6))
        ax.plot(*(Xp + P1X), "o", color=OI["verm"], ms=3.5, mec="black", mew=0.3)
        ax.text(*(Xp + P1X + [0.1, 0.6]), "$P_1$", fontsize=PT - 0.5, color=OI["verm"], va="top")
        ax.add_patch(patches.Circle(Xp, s, fc="none", ec=OI["orange"], lw=1.0, ls="--"))
        if mode != "v1":
            ax.add_patch(patches.Rectangle((Yp[0] - 0.5, Yp[1] - 0.5), 1, 1, fc="#cfe8ff", ec="black", lw=0.7))
            ax.plot(*Yp, "o", color="black", ms=2.0)
            ax.text(Yp[0] + 0.12, Yp[1] - 0.45, "y", fontsize=PT, fontweight="bold")
            ax.annotate("", xy=Xp + P2X, xytext=Yp, arrowprops=dict(arrowstyle="-|>", lw=1.0, color=OI["blue"], mutation_scale=6))
            ax.plot(*(Xp + P2X), "o", color=OI["blue"], ms=3.5, mec="black", mew=0.3)
            ax.text(*(Xp + P2X + [0.1, -0.35]), "$P_2$", fontsize=PT - 0.5, color=OI["blue"], va="bottom")
            ax.add_patch(patches.Circle(Xp + MX, D / 2, fc="none", ec=OI["purple"], lw=1.3))
            ax.plot(*(Xp + MX), "x", color=OI["purple"], ms=5, mew=1.2)
            ax.add_patch(patches.Circle(Xp, 1.0, fc="none", ec=OI["purple"], lw=0.6, ls=":"))
            ax.text(Xp[0] + MX[0] + 0.15, Xp[1] + MX[1] + 0.45, "M", fontsize=PT - 0.5, color=OI["purple"], fontweight="bold")
        if mode == "decline":
            d = -P1X / max(np.linalg.norm(P1X), 1e-9)
            ang0 = np.degrees(np.arctan2(d[1], d[0]))
            ax.add_patch(patches.Wedge(Yp, 2.0, ang0 - 60, ang0 + 60, fc=OI["green"], alpha=0.18, ec="none"))
        for sp in ax.spines.values():
            sp.set_edgecolor("#9a9a9a")
            sp.set_linewidth(0.5)
        ax.text(0.97, 0.03, "synthetic", transform=ax.transAxes, ha="right", va="bottom", fontsize=PT - 1.0, color=INK,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))

    specs = [(ia, "v1", "version 1: round(2s)"), (ia, "accept", "version 3, accepted"), (ib, "decline", "version 3, declined")]
    for k, (i, mode, title) in enumerate(specs):
        x = xs[k]
        header(sh, x, yG, w, title)
        ax = sh.ax(x, yG + hdr, w, w)
        draw_case(ax, i, mode)
        if mode == "v1":
            lines = [f"s = {Q['s'][i]:.2f}; V1 = round(2s) = {Q['V1'][i]}", "sphere centered on x, touching",
                     "its nearest surface point"]
        elif mode == "accept":
            lines = [f"D = |$P_1$ − $P_2$| = {Q['D'][i]:.2f}, |M| = {Mn[i]:.2f}",
                     f"|M| < 1: V3 = round(D) = {Q['V3'][i]}", f"(V1 = {Q['V1'][i]}); sphere pinned at M"]
        else:
            lines = [f"|M| = {Mn[i]:.2f} ≥ 1: V3 = V1 = {Q['V3'][i]}", f"(round(D) = {Q['R'][i]} is not used)",
                     f"v2 gate: cos = {Q['cosg'][i]:.2f} < ½ → V1".replace("-", "−")]
        crop_caption(x, yG, lines)
    drawn["G"] = dict(seed=cases["seed"], accept=dict(s=float(Q["s"][ia]), V1=int(Q["V1"][ia]), D=float(Q["D"][ia]),
                                                      M=float(Mn[ia]), V3=int(Q["V3"][ia]), V2=int(Q["V2"][ia]), cos=float(Q["cosg"][ia])),
                      decline=dict(s=float(Q["s"][ib]), V1=int(Q["V1"][ib]), D=float(Q["D"][ib]), R=int(Q["R"][ib]),
                                   M=float(Mn[ib]), V3=int(Q["V3"][ib]), V2=int(Q["V2"][ib]), cos=float(Q["cosg"][ib])))
    # H: real data: V3 - V1 at the centres of the window; chart of the versions
    xH = xs[3]
    sh.letter(xH - 1.2, yG - 0.6, "H")
    header(sh, xH, yG, w, "V3 − V1 on the scan", indent=2.5)
    ax = crop_axes(sh, xH, yG + hdr, w, ext)
    cen = arr["cen_0.9"]
    dd = arr["D_v3"] - arr["D_v1"]
    img = base_img.copy()
    img[cen & (dd == 0)] = rgb("#4d4d4d")
    img[cen & (dd > 0)] = rgb(OI["verm"])
    img[cen & (dd < 0)] = rgb(OI["blue"])
    ax.imshow(img, interpolation="nearest", extent=ext)
    contour_line(ax, G, ext)
    d31 = {int(k): v for k, v in num["v3_minus_v1"].items()}
    n_cen = num["centres_standard"]
    n_plus = sum(v for k, v in d31.items() if k > 0)
    n_minus = sum(v for k, v in d31.items() if k < 0)
    n_zero = d31.get(0, 0)
    handles = [patches.Patch(fc="#4d4d4d", label=f"V3 = V1 ({100 * n_zero / n_cen:.0f}%)"),
               patches.Patch(fc=OI["verm"], label=f"V3 > V1 ({100 * n_plus / n_cen:.0f}%)"),
               patches.Patch(fc=OI["blue"], label=f"V3 < V1 ({100 * n_minus / n_cen:.1f}%)")]
    ax.legend(handles=handles, loc="upper left", fontsize=PT - 1.2, frameon=True, framealpha=0.9, edgecolor="none",
              handlelength=0.9, handleheight=0.9, borderpad=0.3, labelspacing=0.25, handletextpad=0.4)
    crop_caption(xH, yG, [f"{fmt(n_cen)} centers", f"V3 − V1 from {min(d31):+d} to {max(d31):+d} voxels".replace("-", "−"),
                          f"at ridge_epsilon 0.9, peel 0"])
    header(sh, xs[4], yG, w, "Tb.Th vs version")
    ax = chart_axes(sh, xs[4], yG + hdr, w, w)
    vv = [1, 2, 3]
    tt = [num["version"][str(k)]["mean_mm"] for k in vv]
    ax.plot(vv, tt, "-o", color=OI["blue"], ms=3.5, lw=0.9, zorder=3)
    lo, hi = min(tt) - 0.006, max(tt) + 0.006
    ax.set_ylim(lo, hi)
    ax.set_xlim(0.6, 3.4)
    ax.axvline(VERSION, color=INK2, lw=0.6, ls=":")
    std_label(ax, VERSION, "bottom-left")
    for k, t in zip(vv, tt):
        ax.annotate(f"{t:.4f}", (k, t), xytext=(0, 5) if k == 1 else ((0, -5) if k == 2 else (-3, 4)), textcoords="offset points",
                    ha="center" if k != 3 else "right", va="top" if k == 2 else "bottom", fontsize=PT - 1.0, color=INK2)
    ax.set_xticks(vv)
    ax.set_xlabel("diameter rule (version)")
    ax.set_ylabel("Tb.Th (mm)")
    vr = num["version"]
    lines = ["map mean, assign_epsilon 0.5"]
    if all(vr[str(k)]["ipl"] is not None for k in vv):
        lines += ["IPL export: 0 centers and", "0 diameters differ, 3 versions"]
    crop_caption(xs[4], yG, lines)
    drawn["H"] = dict(versions=num["version"], v3_minus_v1=num["v3_minus_v1"], centres=n_cen)

    # =========================================================== row 5: I assign_epsilon
    yI = rows_y[5]
    sh.letter(xs[0] - 4.0, yI - 0.6, "I")
    header(sh, xs[0], yI, w, "Drawing tolerance")
    ax = sh.ax(xs[0], yI + hdr, w, w)
    ax.set_aspect("equal")
    ax.axis("off")
    Dd = 3
    r = 3
    ax.set_xlim(-r - 0.5, r + 0.5)
    ax.set_ylim(r + 0.5, -r - 0.5)
    fills = {0.0: "#4d4d4d", 0.5: OI["verm"], 1.0: "#f2b78a"}
    nn = {0.0: 0, 0.5: 0, 1.0: 0}
    for yv in range(-r, r + 1):
        for xv in range(-r, r + 1):
            dv = np.hypot(xv, yv)
            fc = "white"
            for ae in (1.0, 0.5, 0.0):
                if dv <= Dd / 2 + ae + 1e-9:
                    fc = fills[ae]
                    nn[ae] += 1
            ax.add_patch(patches.Rectangle((xv - 0.5, yv - 0.5), 1, 1, fc=fc, ec="#c8c8c8", lw=0.4))
    for ae, col in ((0.0, "#4d4d4d"), (0.5, OI["verm"]), (1.0, "#c07a3a")):
        ax.add_patch(patches.Circle((0, 0), Dd / 2 + ae, fc="none", ec=col, lw=0.8))
    ax.plot(0, 0, "o", color="white", ms=3, mec="black", mew=0.5)
    handles = [patches.Patch(fc=fills[0.0], label="ε = 0"), patches.Patch(fc=fills[0.5], label="ε = 0.5"),
               patches.Patch(fc=fills[1.0], label="ε = 1.0")]
    ax.legend(handles=handles, loc="lower right", fontsize=PT - 1.2, frameon=True, framealpha=0.9, edgecolor="none",
              handlelength=0.9, handleheight=0.9, borderpad=0.3, labelspacing=0.25, handletextpad=0.4)
    ax.text(0.03, 0.97, f"one sphere, D = {Dd}", transform=ax.transAxes, fontsize=PT - 1.0, color=INK2, va="top",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.9))
    crop_caption(xs[0], yI, ["c is reached by the sphere at x", "when |c − x| ≤ D/2 + ε; c keeps",
                             "the largest D reaching it"])
    drawn["I_schematic"] = dict(D=Dd, voxels_reached_2d={str(k): v for k, v in nn.items()})
    vmax = int(max(arr[f"map_{ae}"].max() for ae in (0.0, 0.5, 1.0)))
    cm, norm = diam_cmap(vmax)
    for k, ae in enumerate((0.0, 0.5, 1.0)):
        x = xs[k + 1]
        header(sh, x, yI, w, f"assign_epsilon {ae:g}")
        ax = crop_axes(sh, x, yI + hdr, w, ext)
        m = arr[f"map_{ae}"]
        img = base_img.copy()
        has = m > 0
        img[has] = cm(norm(m[has]))[:, :3]
        ax.imshow(img, interpolation="nearest", extent=ext)
        contour_line(ax, G, ext)
        if ae == ASSIGN_EPS:
            standard_badge(ax)
        ar = num["assign"][str(ae)]
        lines = [f"{fmt(ar['support'])} voxels with a value", f"Tb.Th {ar['mean_mm']:.4f} mm"]
        if ar["ipl"] is not None:
            lines.append(ipl_line(f"{ar['ipl']['differing']} voxels"))
        crop_caption(x, yI, lines)
    cax = sh.ax(xs[2] + 4.0, cbar_y, w - 8.0, 1.8)
    cb = sh.fig.colorbar(ScalarMappable(norm=norm, cmap=cm), cax=cax, orientation="horizontal")
    ticks = list(range(1, vmax + 1))
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) for t in ticks])
    cb.ax.tick_params(labelsize=PT - 0.5, length=1.5, pad=1.0, width=0.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label("sphere diameter of the maps (voxels)", fontsize=PT - 0.5, labelpad=1.0)
    header(sh, xs[4], yI, w, "Support vs assign_epsilon")
    ax = chart_axes(sh, xs[4], yI + hdr, w, w)
    a_ = list(ASSIGN_SWEEP)
    su = [num["assign"][str(a)]["support"] / 1e6 for a in a_]
    ax.plot(a_, su, "-o", color=OI["blue"], ms=3.5, lw=0.9, label="ipldt", zorder=3)
    if all(num["assign"][str(a)]["ipl"] is not None for a in a_):
        si = [num["assign"][str(a)]["ipl"]["ipl_support"] / 1e6 for a in a_]
        ax.plot(a_, si, "s", mfc="none", mec=OI["verm"], ms=5.5, mew=0.8, label="IPL export", zorder=4)
    lo, hi = min(su) * 0.97, max(su) * 1.04
    ax.set_ylim(lo, hi)
    ax.set_xlim(-0.07, 1.07)
    ax.axvline(ASSIGN_EPS, color=INK2, lw=0.6, ls=":")
    std_label(ax, ASSIGN_EPS, "bottom-right")
    ax.set_xticks(a_)
    ax.set_xlabel("assign_epsilon")
    ax.set_ylabel("map voxels (millions)")
    ax.legend(frameon=False, fontsize=PT - 1.0, loc="upper left", handletextpad=0.4, borderaxespad=0.2)
    crop_caption(xs[4], yI, ["IPL export: 0 voxels differ", "at 0, 0.25, 0.5 and 1.0"])
    drawn["I"] = num["assign"]

    # ---- print-size check of headers and captions
    sh.fig.canvas.draw()
    rend = sh.fig.canvas.get_renderer()
    for t in sh.fig.texts:
        wmm = t.get_window_extent(rend).width / sh.fig.dpi * 25.4
        limit = wA + 0.5 if t.get_text().startswith(("Sweep order", "In each sub-scan", "pass ")) else w + 1.0
        if wmm > limit:
            say(f"text '{t.get_text()[:40]!r}': {wmm:.1f} mm  <-- WIDER THAN {limit:.1f} mm")
    sh.fig.savefig(out_png, dpi=300)
    sh.fig.savefig(out_svg)
    plt.close(sh.fig)
    drawn["figure"] = dict(width_mm=W_MM, height_mm=round(H_MM, 1), panel_mm=round(w, 2), font_pt=PT,
                           window=num["window"], voxel_mm=el, backend=num["backend"])
    return drawn


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--ipl-exports", default=IPL_EXPORTS_DEFAULT)
    ap.add_argument("--slice", type=int, default=84)
    ap.add_argument("--window", type=int, nargs=2, default=(38, 430), metavar=("Y0", "X0"))
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--vector-window", type=int, default=16)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    y0, x0 = a.window
    key = dict(data=a.data, base=a.base, slice=a.slice, window=[y0, x0], size=a.size, vector_window=a.vector_window)
    if os.path.exists(CACHE) and not a.recompute:
        C = np.load(CACHE, allow_pickle=True)
        if json.loads(str(C["key"])) == key:
            num = json.loads(str(C["numbers"]))
            arr = {k: C[k] for k in C.files if k not in ("key", "numbers")}
            say("cache loaded:", CACHE)
        else:
            num = None
    else:
        num = None
    if num is None:
        say("backend:", "gpu" if cupy_available() else "cpu")
        num, arr = compute(a.data, a.base, a.ipl_exports, a.slice, y0, x0, a.size, a.vector_window)
        np.savez_compressed(CACHE, key=json.dumps(key), numbers=json.dumps(num), **arr)
        say("cache written:", CACHE)
    cases = diameter_cases()
    say(f"diameter phantom: seed {cases['seed']}, {cases['y'].size} centres")
    out_png = os.path.join(a.out, "S6_dt_params.png")
    out_svg = os.path.join(a.out, "S6_dt_params.svg")
    drawn = make_figure(num, arr, cases, out_png, out_svg)
    drawn["computed"] = num
    with open(os.path.join(a.out, "S6_dt_params_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(drawn, f, indent=1)
    say("written:", out_png, out_svg)


if __name__ == "__main__":
    main()
