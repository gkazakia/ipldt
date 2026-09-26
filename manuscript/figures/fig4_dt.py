"""fig4_dt.py -- Figure 4 of the ipldt / ORMIR-BQRL manuscript: the distance-transform engine on real data.

Two rows of seven crops from one XtremeCT II patella scan (one 48 x 48-voxel window of one slice, the same
window in both rows), read left to right as the engine runs:

    segmentation with contour -> surface distance -> containment ridge -> centres inside the contour
    -> integer diameters -> the map drawn by ipldt -> the map computed by IPL (differing voxels: 0)

  row (a)  thickness: the object is the bone segmentation, the region of interest is the trabecular contour
           (IPL /dt_thickness; the map IPL delivers as TRAB_TH for this cohort).  With --object trabseg the
           object is the trabecular segmentation instead (Script 32's Tb.Th; IPL file TRAB_TH_old).
  row (b)  spacing: the object is the marrow space (NOT segmentation), same contour (IPL /dt_spacing, TRAB_SP).

Every panel is computed here with the package's own stage functions (ipldt.field.sir_quad, ipldt.core
surface_distance / ridge / peel_gobj / diameters / draw_spheres, GPU kernels when CuPy is available); the map of
panel 6 is the public API's result (ipldt.dt_thickness / ipldt.dt_spacing), asserted equal to the stage chain,
and the count on panel 7 is the number of voxels of the whole 3-D grid on which that map differs from IPL's map.
Parameters are IPL's standard values (ridge_epsilon 0.9, assign_epsilon 0.5, peel_iter -1, version 3).

Run from the repository root in the `ormir` environment:

    python manuscript/figures/fig4_dt.py
        [--data <IPLDT_LAB_ROOT>/patellae/PFJ-0be66a_R] [--base X2420448]
        [--slice 84] [--object seg|trabseg] [--one-row] [--out manuscript/figures]

Writes fig4_dt.png (300 dpi), fig4_dt.svg and fig4_dt_numbers.json (every number drawn, for the legend).
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
REPO = os.path.dirname(os.path.dirname(HERE))
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
T0 = time.time()

RIDGE_EPS, ASSIGN_EPS, PEEL_ITER, VERSION = 0.9, 0.5, -1, 3      # IPL's standard values


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------ style
# Okabe-Ito colour-blind-safe palette; viridis for the scalar fields.
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", blue="#0072B2", verm="#D55E00", purple="#CC79A7")
BG = np.array([1.0, 1.0, 1.0])
OBJ = np.array([0.86, 0.86, 0.86])        # object voxels without a value
CORT = np.array([0.55, 0.55, 0.55])       # cortical label, panel 1 of row (a)
CONTOUR = "#000000"
PT = 7.0                                  # smallest font at print size (points)
plt.rcParams.update({"font.family": "Arial", "font.size": PT, "axes.linewidth": 0.5, "svg.fonttype": "none",
                     "figure.dpi": 100, "savefig.dpi": 300, "pdf.fonttype": 42})

W_MM = 180.0
MM = 1.0 / 25.4


def rgb(c):
    return np.array(colors.to_rgb(c))


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
    maps = {nm: L(nm).astype(np.int16) for nm in ("TRAB_TH", "TRAB_TH_old", "TRAB_SP")}
    trab_seg = L("TRAB_SEG") > 0
    mraw = ipldt.read_aim(p("TRAB_MASK"))
    G_own = ipldt.render_volume(np.asarray(mraw["data"]) > 0)
    G = np.asarray(ipldt.align_to(dict(data=G_own.astype(np.uint8), dim=mraw["dim"], pos=mraw["pos"]), dim, pos)) > 0
    say(f"grid {tuple(seg.shape)} (z, y, x), voxel {el:.4f} mm; SEG {int((seg > 0).sum()):,d} voxels "
        f"(cortical {int((seg == 127).sum()):,d}, trabecular {int((seg == 126).sum()):,d}); rendered trabecular contour "
        f"{int(G.sum()):,d} voxels; trabecular segmentation {int(trab_seg.sum()):,d} voxels, "
        f"{int((trab_seg & ~G).sum()):,d} of them outside the contour")
    return dict(seg=seg, el=el, G=G, trab_seg=trab_seg, maps=maps)


def ridge_any(obj, V):
    if cupy_available():
        from ipldt.gpu import ridge_gpu
        return ridge_gpu(obj, V, RIDGE_EPS)
    return ridge(obj, V, RIDGE_EPS)


def draw_any(z, y, x, D, shape):
    if cupy_available():
        from ipldt.gpu import draw_spheres_gpu
        return draw_spheres_gpu(z, y, x, D, shape, ASSIGN_EPS)
    return draw_spheres(z, y, x, D, shape, ASSIGN_EPS)


def run_stages(obj, G, public_map, label):
    """The stage chain on the whole volume; `public_map` is the public API's map, asserted equal."""
    obj = np.ascontiguousarray(obj, dtype=bool)
    V = sir_quad(obj)
    s = surface_distance(V)
    s[~obj] = 0.0
    cen_all = ridge_any(obj, V)
    cen = cen_all & peel_gobj(G, PEEL_ITER)
    z, y, x = np.nonzero(cen)
    D = diameters(obj, V, z, y, x, version=VERSION)
    m = draw_any(z, y, x, D, obj.shape)
    m[~obj] = 0
    m[~G] = 0
    n_diff_api = int((m != public_map).sum())
    assert n_diff_api == 0, f"{label}: stage chain and public API differ on {n_diff_api} voxels"
    Dmap = np.zeros(obj.shape, np.int16)
    Dmap[z, y, x] = D
    say(f"{label}: {int(cen_all.sum()):,d} ridge centres, {int(cen.sum()):,d} inside the contour, diameters "
        f"{int(D.min())}..{int(D.max())}, map support {int((m > 0).sum()):,d} voxels; stage chain == public API")
    return dict(obj=obj, s=s, cen_all=cen_all, cen=cen, Dmap=Dmap, map=m,
                n_cen_all=int(cen_all.sum()), n_cen=int(cen.sum()), n_removed=int((cen_all & ~cen).sum()),
                d_min=int(D.min()), d_max=int(D.max()), support=int((m > 0).sum()))


def choose_window(obj2d, g2d, cen2d, n=48, stride=4):
    """The n x n window of the slice that contains the contour (45-75 % inside), a moderate object fraction
    and the most balanced count of ridge centres inside / outside the contour."""
    H, W = obj2d.shape
    best = None
    for y0 in range(2, H - n - 2, stride):
        for x0 in range(2, W - n - 2, stride):
            g = g2d[y0:y0 + n, x0:x0 + n]
            gf = g.mean()
            if gf < 0.45 or gf > 0.75:
                continue
            of = obj2d[y0:y0 + n, x0:x0 + n].mean()
            if of < 0.30 or of > 0.65:
                continue
            c = cen2d[y0:y0 + n, x0:x0 + n]
            sc = min(int((c & g).sum()), int((c & ~g).sum()))
            if best is None or sc > best[0]:
                best = (sc, y0, x0)
    if best is None:
        raise RuntimeError("no window found")
    return best[1], best[2]


# ------------------------------------------------------------------------------------------ figure
def panel_images(S, seg, G, sl, dark2d=None, vmax=None):
    """The seven RGB crops of one row.  dark2d: voxels of panel 1 drawn in the darker grey (the cortical
    label in row (a); the whole bone in row (b), whose object is the marrow)."""
    obj = S["obj"][sl]
    base = np.where(obj[..., None], OBJ, BG)
    # 1 object with contour (the contour line is drawn on top)
    img1 = base.copy()
    if dark2d is not None:
        img1[dark2d] = CORT
    # 2 surface distance
    s = S["s"][sl]
    smax = float(s.max())
    img2 = base.copy()
    img2[obj] = plt.get_cmap("viridis")(s[obj] / smax)[:, :3]
    # 3 ridge centres
    img3 = base.copy()
    img3[S["cen_all"][sl]] = rgb(OI["verm"])
    # 4 centres inside the contour
    img4 = base.copy()
    kept = S["cen"][sl]
    removed = S["cen_all"][sl] & ~kept
    img4[removed] = rgb(OI["blue"])
    img4[kept] = rgb(OI["verm"])
    # 5 diameters at the kept centres; 6 our map; 7 IPL's map
    cm, norm = diam_cmap(vmax)
    img5 = base.copy()
    Dc = S["Dmap"][sl]
    img5[kept] = cm(norm(Dc[kept]))[:, :3]

    def map_img(m):
        im = base.copy()
        has = m > 0
        im[has] = cm(norm(m[has]))[:, :3]
        return im

    return [img1, img2, img3, img4, img5, map_img(S["map"][sl]), map_img(S["ipl"][sl])], smax, (cm, norm), \
        dict(centres=int(S["cen_all"][sl].sum()), kept=int(kept.sum()), removed=int(removed.sum()),
             d_min=int(Dc[kept].min()) if kept.any() else 0, d_max=int(Dc[kept].max()) if kept.any() else 0,
             s_max=smax, map_max=int(S["map"][sl].max()), ipl_max=int(S["ipl"][sl].max()))


def make_figure(rows, seg, G, z, y0, x0, n, el, out_png, out_svg):
    headers = ["Object + contour", "Surface distance", "Containment ridge", "Centers in contour",
               "Integer diameters", "Map (ipldt)", "Map (IPL)"]
    # ---- geometry in mm
    left, right, gap = 7.0, 1.0, 1.3
    P = (W_MM - left - right - 6 * gap) / 7.0
    hdr_h, cap_h, cb_h = 3.4, 3.0, 1.9
    below = 1.0 + cb_h + 2.7 + 2.9          # colour bar + tick labels + colour-bar label
    row_h = P + below
    top, bottom, row_gap = 1.0, 0.5, 1.5
    H_MM = top + hdr_h + len(rows) * row_h + (len(rows) - 1) * row_gap + bottom
    fig = plt.figure(figsize=(W_MM * MM, H_MM * MM))

    def ax_mm(x, y_top, w, h):
        return fig.add_axes([x / W_MM, 1.0 - (y_top + h) / H_MM, w / W_MM, h / H_MM])

    sl = (z, slice(y0, y0 + n), slice(x0, x0 + n))
    xs = np.arange(x0, x0 + n)
    ys = np.arange(y0, y0 + n)
    ext = (x0 - 0.5, x0 + n - 0.5, y0 + n - 0.5, y0 - 0.5)
    numbers = {}
    header_texts = []
    for r, row in enumerate(rows):
        py = top + hdr_h + r * (row_h + row_gap)          # top of this row's panels
        S = row["S"]
        dark2d = {"cortical": seg[sl] == 127, "bone": seg[sl] > 0}.get(row.get("dark"))
        vmax = int(max(S["map"][sl].max(), S["ipl"][sl].max()))
        imgs, smax, (cm, norm), crop = panel_images(S, seg, G, sl, dark2d=dark2d, vmax=vmax)
        n_diff = int((S["map"] != S["ipl"]).sum())
        numbers[row["key"]] = dict(crop=crop, differing_voxels_whole_grid=n_diff, voxels_compared=int(S["map"].size),
                                   n_centres_volume=S["n_cen_all"], n_centres_inside_contour=S["n_cen"],
                                   n_centres_removed=S["n_removed"], diameters_volume=[S["d_min"], S["d_max"]],
                                   map_support_volume=S["support"], ipl_support_volume=int((S["ipl"] > 0).sum()),
                                   colour_scale_max=vmax)
        captions = [row["object_caption"], None, f"{crop['centres']} centers",
                    f"{crop['kept']} kept, {crop['removed']} removed",
                    f"diameters {crop['d_min']}\u2013{crop['d_max']} voxels", None,
                    f"differing voxels: {n_diff}"]
        # row label
        fig.text((left - 3.6) / W_MM, 1.0 - (py + P / 2) / H_MM, row["label"], rotation=90,
                 ha="center", va="center", fontsize=PT + 1, fontweight="bold")
        fig.text((left - 6.6) / W_MM, 1.0 - (py - 0.3) / H_MM, row["letter"], ha="left", va="top",
                 fontsize=PT + 1.5, fontweight="bold")
        for i, img in enumerate(imgs):
            px = left + i * (P + gap)
            ax = ax_mm(px, py, P, P)
            ax.imshow(img, interpolation="nearest", extent=ext)
            ax.contour(xs, ys, G[sl].astype(float), levels=[0.5], colors=[CONTOUR], linewidths=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#9a9a9a")
            if r == 0:
                t = fig.text((px + P / 2) / W_MM, 1.0 - (py - hdr_h / 2 - 0.2) / H_MM, headers[i], ha="center",
                             va="center", fontsize=PT, fontweight="bold")
                header_texts.append((t, P))
            # scale bar on the first panel of the first row
            if r == 0 and i == 0:
                bar = 1.0 / el                                     # 1 mm in voxels
                bx, by = x0 + 2.0, y0 + n - 3.5
                ax.plot([bx, bx + bar], [by, by], color="black", lw=1.6, solid_capstyle="butt")
                ax.text(bx + bar / 2, by - 1.6, "1 mm", ha="center", va="bottom", fontsize=PT,
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.85))
            cy = py + P + 1.0
            if captions[i]:
                fig.text((px + P / 2) / W_MM, 1.0 - (cy + cap_h / 2 - 0.3) / H_MM, captions[i], ha="center",
                         va="center", fontsize=PT, color="#222222")
            if i == 1:                                           # surface-distance colour bar
                cax = ax_mm(px + 0.5, cy, P - 1.0, cb_h)
                sm = ScalarMappable(norm=colors.Normalize(0, smax), cmap="viridis")
                cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
                ticks = [0, smax / 2, smax] if smax < 4 else list(np.arange(0, smax + 1e-9, 2))
                cb.set_ticks(ticks)
                cb.set_ticklabels([f"{t:g}" if abs(t - round(t)) < 1e-9 else f"{t:.1f}" for t in ticks])
                cb.ax.tick_params(labelsize=PT, length=1.5, pad=1.2, width=0.5)
                cb.outline.set_linewidth(0.5)
                cb.set_label("distance to surface (voxels)", fontsize=PT, labelpad=1.5)
            if i == 5:                                           # diameter colour bar (panels 5-7)
                cax = ax_mm(px + 0.5, cy, P - 1.0, cb_h)
                sm = ScalarMappable(norm=norm, cmap=cm)
                cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
                step = 1 if vmax <= 8 else (2 if vmax <= 16 else 4)
                ticks = list(range(1, vmax + 1, step))
                if ticks[-1] != vmax:
                    ticks.append(vmax)
                cb.set_ticks(ticks)
                cb.set_ticklabels([str(t) for t in ticks])
                cb.ax.tick_params(labelsize=PT, length=1.5, pad=1.2, width=0.5)
                cb.outline.set_linewidth(0.5)
                cb.set_label("sphere diameter (voxels)", fontsize=PT, labelpad=1.5)
    # header widths at print size (labels must stay one line and inside their panel)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for t, pw in header_texts:
        w_mm = t.get_window_extent(rend).width / fig.dpi * 25.4
        flag = "" if w_mm <= pw + 0.8 else "   <-- WIDER THAN PANEL"
        say(f"header '{t.get_text()}': {w_mm:.1f} mm of {pw:.1f} mm{flag}")
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)
    numbers["figure"] = dict(width_mm=W_MM, height_mm=round(H_MM, 1), panel_mm=round(P, 2), crop_voxels=n,
                             slice_index=z, window_origin_yx=[int(y0), int(x0)], voxel_mm=el, font_pt=PT)
    return numbers


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--slice", type=int, default=84)
    ap.add_argument("--object", choices=("seg", "trabseg"), default="seg",
                    help="row (a) object: the bone segmentation (IPL's TRAB_TH of this cohort) or the trabecular "
                         "segmentation (Script 32's Tb.Th, IPL's TRAB_TH_old)")
    ap.add_argument("--one-row", action="store_true", help="thickness row only")
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    say("backend:", "gpu" if cupy_available() else "cpu")
    P = load(a.data, a.base)
    seg, G, el = P["seg"], P["G"], P["el"]
    bone = seg > 0

    # row (a): thickness
    if a.object == "seg":
        obj_th, ipl_th = bone, P["maps"]["TRAB_TH"]
        caption, dark = "object: bone", "cortical"
    else:
        obj_th, ipl_th = P["trab_seg"], P["maps"]["TRAB_TH_old"]
        caption, dark = "object: trabecular bone", None
    r_th = ipldt.dt_thickness(obj_th, {"rendered": G}, voxel_size_mm=el, ridge_epsilon=RIDGE_EPS,
                              assign_epsilon=ASSIGN_EPS, peel_iter=PEEL_ITER, version=VERSION)
    S_th = run_stages(obj_th, G, r_th.map, "thickness")
    S_th["ipl"] = ipl_th
    say(f"thickness report: mean {r_th.report['Th_mm']:.6f} mm over {r_th.report['Th_n_voxels']:,d} voxels; "
        f"IPL's map: mean {float(ipl_th[ipl_th > 0].mean()) * el:.6f} mm over {int((ipl_th > 0).sum()):,d} voxels; "
        f"differing voxels {int((S_th['map'] != ipl_th).sum())} of {ipl_th.size:,d}")
    rows = [dict(key="thickness", S=S_th, label="Thickness", letter="A", object_caption=caption, dark=dark)]

    # row (b): spacing
    if not a.one_row:
        r_sp = ipldt.dt_spacing(bone, {"rendered": G}, voxel_size_mm=el, ridge_epsilon=RIDGE_EPS,
                                assign_epsilon=ASSIGN_EPS, peel_iter=PEEL_ITER, version=VERSION)
        S_sp = run_stages(~bone, G, r_sp.map, "spacing")
        S_sp["ipl"] = P["maps"]["TRAB_SP"]
        say(f"spacing report: mean {r_sp.report['Sp_mm']:.6f} mm over {r_sp.report['Sp_n_voxels']:,d} voxels; "
            f"IPL's map: mean {float(S_sp['ipl'][S_sp['ipl'] > 0].mean()) * el:.6f} mm over "
            f"{int((S_sp['ipl'] > 0).sum()):,d} voxels; differing voxels {int((S_sp['map'] != S_sp['ipl']).sum())} "
            f"of {S_sp['ipl'].size:,d}")
        rows.append(dict(key="spacing", S=S_sp, label="Spacing", letter="B", object_caption="object: marrow space",
                         dark="bone"))

    z = a.slice
    n = 48
    y0, x0 = choose_window(S_th["obj"][z], G[z], S_th["cen_all"][z], n)
    say(f"crop: slice {z}, window origin (y, x) = ({y0}, {x0}), {n} x {n} voxels = {n * el:.2f} mm")
    out_png = os.path.join(a.out, "fig4_dt.png")
    out_svg = os.path.join(a.out, "fig4_dt.svg")
    numbers = make_figure(rows, seg, G, z, y0, x0, n, el, out_png, out_svg)
    numbers["parameters"] = dict(ridge_epsilon=RIDGE_EPS, assign_epsilon=ASSIGN_EPS, peel_iter=PEEL_ITER,
                                 version=VERSION, row_a_object=a.object, backend="gpu" if cupy_available() else "cpu")
    numbers["reports"] = dict(thickness={k: v for k, v in r_th.report.items()})
    if not a.one_row:
        numbers["reports"]["spacing"] = {k: v for k, v in r_sp.report.items()}
    with open(os.path.join(a.out, "fig4_dt_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(numbers, f, indent=1)
    say("written:", out_png, out_svg)


if __name__ == "__main__":
    main()
