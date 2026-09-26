"""S3_step1_presets.py -- Supplementary Figure S3 of the ipldt / ORMIR-BQRL manuscript: each parameter of the
compartment chain (STEP 1) on real data, with the tibia and radius presets compared.

One ultradistal tibia and one ultradistal radius of the validation set, each run with its site's standard preset
(tibia: corner components >= 200,000 voxels, second closing 50; radius: >= 800, closing 30).  All panels show real
slices; the voxel counts printed under them count the whole volume, in the run the slices come from.
  A-D  the morphological steps of the trabecular candidate on the tibia scan: erosion 3 / largest component /
       dilation 3, closing 15, the 6-voxel peel, opening 15 (grey = unchanged, vermilion = removed, blue = added)
  E    the corner-recovery branch on the tibia scan (standard 200,000 / 50): the opening residue, its erosion by 3,
       the component threshold, the dilation and the bounded re-addition, the second closing
  F    the same branch on the radius scan (standard 800 / 30)
  G    single-parameter swaps: how the final cortical mask changes when corner_min or close2 takes the other
       site's value (every image states the site's standard value)
  H    the slice-wise 50 % rule on the radius scan, on a slice where it matters: the 4-connected components of
       the trabecular candidate, each with its fraction of the slice, kept or removed

Run from the repository root in the `ormir` environment (numpy, scipy, numba, matplotlib; no GPU needed):

    python manuscript/figures/supp/S3_step1_presets.py [--recompute]

Writes manuscript/figures/supp/S3_step1_presets.png (300 dpi, 180 mm wide) and .svg, plus
S3_step1_presets_numbers.json (every number drawn).  The chain runs (6 runs of ipldt.step1, about 2 min) are
cached in manuscript/figures/cache/S3_step1_presets_cache.{npz,json}; --recompute redoes them.

Inputs (read only): the greyscale AIM and IPL's rendered periosteal contour (<base>_CT.AIM) of the two scans, as
named in validation/results/oslh_auto_vN/records/<tag>.json; IPL's rendered cortical / trabecular masks of the same
scans (the verification that the standard runs reproduce IPL's masks).  Nothing under ipldt/ is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
from ipldt.io import read_aim, align_to  # noqa: E402
from ipldt import ipl_ops as ops  # noqa: E402
from ipldt.step1 import TIBIA, RADIUS, Step1Params, cort_trab_separation  # noqa: E402
from ipldt.contour import render_volume  # noqa: E402

# ------------------------------------------------------------------------------------------------ inputs
RECORDS = os.path.join(REPO, "validation", "results", OSLH_AUTO, "records")
SCANS = {
    "T": dict(tag="Distal_REPRO_476576", site="ultradistal tibia", preset="tibia", params=TIBIA),
    "R": dict(tag="Distal_REPRO_809980", site="ultradistal radius", preset="radius", params=RADIUS),
}
# the parameter varied in panel G: (label, field, the other site's value)
VARIANTS = {"T": [("cm", "corner_min", RADIUS.corner_min), ("c2", "close2", RADIUS.close2)],
            "R": [("cm", "corner_min", TIBIA.corner_min), ("c2", "close2", TIBIA.close2)]}
# the steps of panels A-D and the other single-step comparisons, (before stage, after stage)
STEPS = {"A": ("02_trab0", "03_peel6"), "B": ("08_trabrank", "11_dil3"), "C": ("11_dil3", "12_close15"),
         "D": ("14_bbc", "15_open15"), "peel13": ("12_close15", "13_close15peel"), "close2": ("22_trabadd", "23_close50"),
         "peel24": ("23_close50", "24_close50peel"), "sw25": ("24_close50peel", "25_slicewise"), "sw27": ("26_cort", "27_cortslice")}
CACHE = os.path.join(os.path.dirname(HERE), "cache", "S3_step1_presets_cache")
OUT = os.path.join(HERE, "S3_step1_presets")
VOX_MM = 0.0607

# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito (colour-blind safe, the palette of every figure): blue = added / kept, vermilion = removed,
# orange = candidate under test, greys = masks, ink for text
C_BLUE, C_VERM, C_ORANGE = "#0072B2", "#D55E00", "#E69F00"
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_FILL, C_LIGHT, C_PERI = "#b9b8b3", "#e1e0d9", "#898781"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.5, "axes.edgecolor": C_AXIS, "savefig.dpi": 300, "figure.dpi": 100,
    "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": C_INK, "axes.labelcolor": C_INK2,
})
T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def record_paths(tag):
    d = json.load(open(os.path.join(RECORDS, tag + ".json")))
    # the published records carry no local paths: the folder is rebuilt from the id under the non-public data root
    folder, b = os.path.join(lab_path("Cross_validation_IPL/OS_LH_AUTO"), *d["id"].split("/")), d["base"]
    trab = next((os.path.join(folder, b + s) for s in ("_TRAB_MASK_CORR_CT.AIM", "_TRAB_MASK_CT.AIM")
                 if os.path.exists(os.path.join(folder, b + s))), os.path.join(folder, b + "_TRAB_MASK_CT.AIM"))
    return dict(grey=os.path.normpath(os.path.join(folder, b + ".AIM")), per=os.path.normpath(os.path.join(folder, b + "_CT.AIM")),
                cort=os.path.normpath(os.path.join(folder, b + "_CORT_MASK_CT.AIM")), trab=os.path.normpath(trab),
                preset=d["B"]["step1"]["preset"], region=d["meta"]["region"], bone=d["meta"]["bone"])


def count(v):
    return int(np.count_nonzero(v["data"]))


def slice_at(v, z, frame):
    """The set voxels of volume v on global slice z inside the (x0, y0, W, H) window, as a bool (H, W) array
    (positive values: the -127 that /subtract_aims stores where the dense-bone stage reaches outside the periosteal
    rendering is not part of the candidate)."""
    x0, y0, W, H = frame
    return align_to(v, (int(W), int(H), 1), (int(x0), int(y0), int(z)))[0] > 0


def bbox2d(mask, margin, bounds):
    ys, xs = np.nonzero(mask)
    x0, x1 = max(0, xs.min() - margin), min(bounds[0], xs.max() + 1 + margin)
    y0, y1 = max(0, ys.min() - margin), min(bounds[1], ys.max() + 1 + margin)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def xor_volumes(a, b, on_a=False, z_range=None):
    """(a-only voxels, b-only voxels, per-slice count of differing voxels, z0), compared by global position on the
    union grid of the two volumes or, with on_a, on a's own grid (a step's growth into a dilation margin ignored).
    z_range = (z_lo, z_hi) restricts the counts and the profile to the global slices z_lo <= z < z_hi (the scan's
    stack; the extra slices a dilation margin adds are left out)."""
    if on_a:
        dim, pos = tuple(a["dim"]), tuple(a["pos"])
    else:
        dim, pos = ops.union_grid(a, b)
    A = ops.on_grid(a, dim, pos) > 0            # positive: the -127 of /subtract_aims outside the rendering is not object
    B = ops.on_grid(b, dim, pos) > 0
    if z_range is not None:
        keep = np.zeros(dim[2], bool)
        lo, hi = max(0, z_range[0] - pos[2]), min(dim[2], z_range[1] - pos[2])
        keep[lo:hi] = True
        A, B = A & keep[:, None, None], B & keep[:, None, None]
    X = A != B
    return int((A & ~B).sum()), int((B & ~A).sum()), X.reshape(dim[2], -1).sum(1).astype(np.int64), int(pos[2])


def z_profile(v):
    d = v["data"] != 0
    return d.reshape(d.shape[0], -1).sum(1), int(v["pos"][2])


def count_in_stack(v, z_range):
    """The positive voxels of v on the global slices z_range[0] <= z < z_range[1]."""
    lo, hi = max(0, z_range[0] - int(v["pos"][2])), min(int(v["dim"][2]), z_range[1] - int(v["pos"][2]))
    return int(np.count_nonzero(v["data"][lo:hi] > 0))


def square_window(mask2d, margin, min_side, bounds):
    """A square (x0, y0, side) around the set pixels of a 2-D mask, clipped to bounds = (W, H)."""
    x0, y0, w, h = bbox2d(mask2d, margin, bounds)
    side = int(min(max(w, h, min_side), min(bounds)))
    x0 = int(min(max(0, x0 + w // 2 - side // 2), bounds[0] - side))
    y0 = int(min(max(0, y0 + h // 2 - side // 2), bounds[1] - side))
    return x0, y0, side


def peel2d(mask2d, n):
    """The slice-wise 4-connected peel of a rendered contour slice (ipldt.core.peel_gobj on one slice)."""
    return ndi.binary_erosion(mask2d, structure=ndi.generate_binary_structure(2, 1), iterations=int(n), border_value=0)


# ------------------------------------------------------------------------------------------------ compute
def compute():
    A, N = {}, dict(scans={}, vox_mm=VOX_MM)
    S4 = ndi.generate_binary_structure(2, 1)
    S6 = ndi.generate_binary_structure(3, 1)
    for key, sc in SCANS.items():
        P = record_paths(sc["tag"])
        assert P["preset"] == sc["preset"], (sc["tag"], P["preset"])
        say(f"{key} {sc['tag']} ({P['region']} {P['bone']}, {sc['preset']} preset): reading")
        grey, per = read_aim(P["grey"]), read_aim(P["per"])
        prm = sc["params"]
        n = dict(tag=sc["tag"], site=sc["site"], preset=sc["preset"], params=dict(prm.__dict__),
                 grey_grid=[list(grey["dim"]), list(grey["pos"])], per_grid=[list(per["dim"]), list(per["pos"])])
        say(f"  chain, standard preset (corner_min {prm.corner_min:,d}, close2 {prm.close2}), keep_stages")
        res = cort_trab_separation(grey, per, prm, keep_stages=True)
        st, info = res["stages"], res["info"]
        n["counts"] = {k: v for k, v in info["counts"].items() if k != "peel"}
        n["peel_counts"] = {str(k): v for k, v in info["counts"]["peel"].items()}
        n["thresholds"] = info["thresholds"]
        n["seconds_step1"] = info["timings"]["total"]
        n["grids"] = {k: [list(g[0]), list(g[1])] for k, g in info["grids"].items()}
        # verification: our final masks, rendered as contours, against IPL's rendered masks of the evaluation
        ver = {}
        for name, src, ipl_path in (("cort", "28_cortfinal", P["cort"]), ("trab", "29_trabfinal", P["trab"])):
            ours = ops.mask_vol(render_volume(st[src]["data"] != 0), st[src]["dim"], st[src]["pos"])
            ipl = read_aim(ipl_path)
            a_only, b_only, _, _ = xor_volumes(ours, ipl)
            ver[name] = dict(voxels_ours=count(ours), voxels_ipl=count(ipl), ours_only=a_only, ipl_only=b_only,
                             mismatches=a_only + b_only)
            say(f"  {name}: rendered {count(ours):,d} vs IPL {count(ipl):,d}: {a_only + b_only} differing")
            del ours, ipl
        n["verification"] = ver
        # the display frame: the periosteal rendering's z-projection box + margin (global coordinates), so that
        # every slice of the stack fits
        z_mid = int(per["pos"][2] + per["dim"][2] // 2)
        z_range = (int(per["pos"][2]), int(per["pos"][2] + per["dim"][2]))
        proj = (per["data"] != 0).any(axis=0)
        ys, xs = np.nonzero(proj)
        frame = (int(per["pos"][0] + xs.min() - 8), int(per["pos"][1] + ys.min() - 8),
                 int(xs.max() - xs.min() + 17), int(ys.max() - ys.min() + 17))
        n["frame"], n["z_mid"], n["z_range"] = list(frame), z_mid, list(z_range)
        # component structure of the corner branch
        c17 = ops.component_sizes(st["17_cornero"])
        c18 = ops.component_sizes(st["18_corncl"])
        c21 = ops.component_sizes(st["21_corners2"])
        n["components"] = dict(stage17=[int(x) for x in c17[:8]], n17=int(c17.size), stage18=[int(x) for x in c18[:8]],
                               n18=int(c18.size), stage21=[int(x) for x in c21[:8]], n21=int(c21.size))
        # the slice shown for the branch: where the stage-18 (kept) or, when empty, the largest stage-17 component is largest
        if c18.size:
            prof, z0 = z_profile(st["18_corncl"])
        else:
            lab, nl = ndi.label(st["17_cornero"]["data"] != 0, structure=S6)
            sizes = ndi.sum(np.ones_like(lab, dtype=np.int32), lab, range(1, nl + 1))
            big = lab == (int(np.argmax(sizes)) + 1)
            prof, z0 = big.reshape(big.shape[0], -1).sum(1), int(st["17_cornero"]["pos"][2])
        z_show = int(z0 + int(np.argmax(prof)))
        n["z_show"] = z_show
        n["branch_profile_top"] = [(int(z0 + i), int(prof[i])) for i in np.argsort(prof)[::-1][:5]]
        say(f"  branch: 17 {c17.size} comps {c17[:5].tolist()}, 18 {c18[:5].tolist()}, 21 {c21[:5].tolist()}; z_show {z_show}")
        # crops: the mid-slice of every stage used by A-D, the z_show slice of the branch stages
        for tag in ("00_all", "02_trab0", "03_peel6", "08_trabrank", "11_dil3", "12_close15", "13_close15peel", "14_bbc", "15_open15"):
            A[f"{key}_mid_{tag}"] = slice_at(st[tag], z_mid, frame)
        for tag in ("00_all", "14_bbc", "15_open15", "16_corners", "17_cornero", "18_corncl", "21_corners2",
                    "22_trabadd", "23_close50", "28_cortfinal"):
            A[f"{key}_show_{tag}"] = slice_at(st[tag], z_show, frame)
        # the whole-volume effect of each single step (removed / added on the input stage's grid) and the slice
        # where the step changes most
        # (counts within the scan's stack of slices; the whole-grid totals, which include the slices a dilation
        # margin adds and the peel clears again, are recorded beside them)
        steps = {}
        for name, (a, b) in STEPS.items():
            rem, add, prof, z0 = xor_volumes(st[a], st[b], on_a=True, z_range=z_range)
            rem_t, add_t, _, _ = xor_volumes(st[a], st[b], on_a=True)
            z_max = int(z0 + int(np.argmax(prof))) if prof.max() > 0 else z_mid
            steps[name] = dict(before=a, after=b, removed=rem, added=add, removed_total=rem_t, added_total=add_t,
                               voxels_before=count(st[a]), voxels_after=count(st[b]),
                               voxels_before_in_stack=count_in_stack(st[a], z_range), voxels_after_in_stack=count_in_stack(st[b], z_range),
                               z_max=z_max, max_per_slice=int(prof.max()), slices_changed=int((prof > 0).sum()),
                               z_profile_top=[(int(z0 + i), int(prof[i])) for i in np.argsort(prof)[::-1][:5]])
            A[f"{key}_step_{name}_before"] = slice_at(st[a], z_max, frame)
            A[f"{key}_step_{name}_after"] = slice_at(st[b], z_max, frame)
            A[f"{key}_step_{name}_00"] = slice_at(st["00_all"], z_max, frame)
            say(f"  step {name:7s} {a} -> {b}: -{rem:,d} +{add:,d} within the stack (whole grid -{rem_t:,d} +{add_t:,d}); "
                f"max {int(prof.max()):,d} on slice {z_max}")
        n["steps"] = steps
        # the slice-wise rule: the slice where 24 -> 25 removes most, and that slice's 4-connected components
        a24, a25 = st["24_close50peel"]["data"] != 0, st["25_slicewise"]["data"] != 0
        rem = a24 & ~a25
        zr = rem.reshape(rem.shape[0], -1).sum(1)
        z0 = int(st["24_close50peel"]["pos"][2])
        n["slicewise"] = dict(removed_total=int(rem.sum()), slices=[(int(z0 + i), int(zr[i])) for i in np.nonzero(zr)[0]])
        if zr.max() > 0:
            iz = int(np.argmax(zr))
            z_sw = z0 + iz
            lab, nl = ndi.label(a24[iz], structure=S4)
            tot = int(a24[iz].sum())
            comps = []
            for k in range(1, nl + 1):
                m = lab == k
                cy, cx = ndi.center_of_mass(m)
                comps.append(dict(voxels=int(m.sum()), share=100.0 * m.sum() / tot, kept=bool((a25[iz] & m).any()),
                                  cx=float(st["24_close50peel"]["pos"][0] + cx), cy=float(st["24_close50peel"]["pos"][1] + cy)))
            comps.sort(key=lambda c: -c["voxels"])
            n["slicewise"].update(z=int(z_sw), slice_total=tot, n_components=int(nl), components=comps)
            A[f"{key}_sw_24"] = slice_at(st["24_close50peel"], z_sw, frame)
            A[f"{key}_sw_25"] = slice_at(st["25_slicewise"], z_sw, frame)
            A[f"{key}_sw_00"] = slice_at(st["00_all"], z_sw, frame)
            say(f"  slicewise: {int(rem.sum())} voxels removed on {int((zr > 0).sum())} slices; z {z_sw}: total {tot}, "
                f"{nl} components " + ", ".join(f"{c['voxels']} ({c['share']:.2f}%{'' if c['kept'] else ', removed'})" for c in comps))
        del a24, a25, rem
        cort_std = res["cort"]
        del res, st
        # panel G: one parameter varied at a time
        n["variants"] = {}
        for label, field, value in VARIANTS[key]:
            alt = Step1Params(**{**prm.__dict__, field: value})
            say(f"  chain, {field} = {value:,d} instead of {getattr(prm, field):,d}")
            r2 = cort_trab_separation(grey, per, alt)
            c2 = r2["info"]["counts"]
            std_only, alt_only, prof, z0 = xor_volumes(cort_std, r2["cort"])
            z_max = int(z0 + int(np.argmax(prof)))
            n["variants"][label] = dict(field=field, standard=getattr(prm, field), varied=value, cort_std_only=std_only,
                                        cort_alt_only=alt_only, cort_differs=std_only + alt_only, z_maxdiff=z_max,
                                        slices_with_diff=int((prof > 0).sum()), max_per_slice=int(prof.max()),
                                        counts=dict(s17=c2["17_cornero"], s18=c2["18_corncl"], s21=c2["21_corners2"],
                                                    s22=c2["22_trabadd"], s23=c2["23_close50"], s28=c2["28_cortfinal"]),
                                        cort_voxels_std=count(cort_std), cort_voxels_alt=count(r2["cort"]))
            A[f"{key}_var_{label}_std"] = slice_at(cort_std, z_max, frame)
            A[f"{key}_var_{label}_alt"] = slice_at(r2["cort"], z_max, frame)
            A[f"{key}_var_{label}_00"] = slice_at(per, z_max, frame)
            say(f"    cortical mask: {std_only + alt_only:,d} voxels differ ({std_only:,d} standard-only, {alt_only:,d} varied-only), "
                f"max {int(prof.max()):,d} on slice {z_max}; 18 = {c2['18_corncl']:,d}, 21 = {c2['21_corners2']:,d}")
            del r2
        N["scans"][key] = n
        del grey, per, cort_std
    return dict(arrays=A, numbers=N)


def save_cache(R):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE + ".npz", **R["arrays"])
    json.dump(R["numbers"], open(CACHE + ".json", "w"), indent=1)


def load_cache():
    z = np.load(CACHE + ".npz")
    return dict(arrays={k: z[k] for k in z.files}, numbers=json.load(open(CACHE + ".json")))


# ------------------------------------------------------------------------------------------------ drawing
FIG_W = 180.0
FIG_H = 213.0


def ax_mm(fig, x, y_top, w, h):
    return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H])


def ftext(fig, x, y_top, s, **kw):
    kw.setdefault("va", "top")
    return fig.text(x / FIG_W, 1 - y_top / FIG_H, s, **kw)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def fmt(n):
    return f"{int(n):,d}"


def paint(layers, shape):
    """An RGB image from [(mask, colour), ...] painted in order on white."""
    rgb = np.ones(shape + (3,))
    for m, c in layers:
        rgb[m] = hex_rgb(c)
    return rgb


def show(ax, rgb, outline=None, outline_color=C_PERI, lw=0.35):
    H, W = rgb.shape[:2]
    ax.imshow(rgb, interpolation="antialiased", origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    if outline is not None and outline.any():
        ax.contour(outline.astype(float), levels=[0.5], colors=[outline_color], linewidths=lw)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(C_AXIS)
        s.set_linewidth(0.4)


def scale_bar(ax, W, H, mm, fs=6.5):
    n = mm / VOX_MM
    xb, yb = W * 0.05, H * 0.93
    ax.plot([xb, xb + n], [yb, yb], color=C_INK, lw=1.1, solid_capstyle="butt")
    ax.text(xb + n / 2, yb - H * 0.025, f"{mm:g} mm", ha="center", va="bottom", fontsize=fs, color=C_INK)


def arrow(fig, x0, x1, y_mid):
    fig.add_artist(patches.FancyArrowPatch((x0 / FIG_W, 1 - y_mid / FIG_H), (x1 / FIG_W, 1 - y_mid / FIG_H),
                                           transform=fig.transFigure, arrowstyle="-|>", mutation_scale=6, lw=0.6, color=C_INK2))


def check_text_widths(fig, items):
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


def draw(R):
    A, N = R["arrays"], R["numbers"]
    T, Rr = N["scans"]["T"], N["scans"]["R"]
    fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4))
    checks = []
    x_left, x_right = 2.0, FIG_W - 2.0
    span = x_right - x_left

    def header(x, y, letter, title, sub, x1, sub_lines=2):
        ftext(fig, x, y, letter, fontsize=9, fontweight="bold", ha="left")
        t = ftext(fig, x + 3.6, y + 0.3, title, fontsize=7.5, fontweight="bold", ha="left")
        s = ftext(fig, x, y + 3.7, sub, fontsize=6.8, color=C_INK2, ha="left", linespacing=1.15)
        checks.extend([(t, x, x1, title), (s, x, x1, sub)])
        return 3.7 + 2.9 * sub_lines + 0.7

    # ------------------------------------------------------------ row 1: panels A-D on the tibia scan
    H, W = A["T_mid_00_all"].shape
    gap = 3.0
    pw = (span - 3 * gap) / 4
    ph = pw * H / W
    y = 1.0
    steps = T["steps"]
    # (letter, title, description, step key, which slice: 'mid' or 'max' (the slice where the step changes most))
    row1 = [
        ("A", "6-voxel peel", "candidate kept at least 6 voxels\n(0.36 mm) inside the contour", "A", "mid"),
        ("B", "Erosion 3, dilation 3", "structures thinner than 2 × 3 voxels\nremoved; largest component kept", "B", "mid"),
        ("C", "Closing 15", "bridges gaps of up to 2 × 15 voxels\nbetween parts of the candidate", "C", "mid"),
        ("D", "Opening 15", "removes protrusions narrower than\n2 × 15 voxels: the corner candidates", "D", "mid"),
    ]
    head_h = 10.6
    for k, (letter, title, sub, skey, which) in enumerate(row1):
        x = x_left + k * (pw + gap)
        header(x, y, letter, title, sub, x + pw + gap * 0.5)
        ax = ax_mm(fig, x, y + head_h, pw, ph)
        a, b = STEPS[skey]
        if which == "mid":
            before, after, per = A[f"T_mid_{a}"], A[f"T_mid_{b}"], A["T_mid_00_all"]
        else:
            before, after, per = A[f"T_step_{skey}_before"], A[f"T_step_{skey}_after"], A[f"T_step_{skey}_00"]
        layers = []
        if skey == "A":                                   # the 6-voxel shell inside the periosteal contour, shaded
            layers.append((per & ~peel2d(per, T["params"]["peel0"]), C_LIGHT))
        layers += [(before & after, C_FILL), (before & ~after, C_VERM), (after & ~before, C_BLUE)]
        show(ax, paint(layers, before.shape), outline=per)
        if k == 1:
            scale_bar(ax, W, H, 5.0)
        if skey == "A":                                   # inset: the boundary window where the peel removes most
            rem = before & ~after
            side = 72
            dens = ndi.uniform_filter(rem.astype(float), size=side, mode="constant")
            cy, cx = np.unravel_index(int(np.argmax(dens)), dens.shape)
            wx0 = int(min(max(0, cx - side // 2), W - side))
            wy0 = int(min(max(0, cy - side // 2), H - side))
            Zw = (slice(wy0, wy0 + side), slice(wx0, wx0 + side))
            ax.add_patch(patches.Rectangle((wx0 - 0.5, wy0 - 0.5), side, side, fc="none", ec=C_INK, lw=0.6))
            fw = 0.36
            axi = ax.inset_axes([0.015, 0.015, fw, fw * W / H])
            show(axi, paint([(m[Zw], c) for m, c in layers], (side, side)), outline=per[Zw], lw=0.5)
            axi.contour(peel2d(per, T["params"]["peel0"])[Zw].astype(float), levels=[0.5], colors=[C_PERI], linewidths=0.5,
                        linestyles="dashed")
            for sp in axi.spines.values():
                sp.set_edgecolor(C_INK)
                sp.set_linewidth(0.6)
            T["steps"]["A"]["inset_window"] = [int(N["scans"]["T"]["frame"][0] + wx0), int(N["scans"]["T"]["frame"][1] + wy0), side]
        s = steps[skey]
        parts = []
        if s["removed"]:
            parts.append(f"− {fmt(s['removed'])} voxels")
        if s["added"]:
            parts.append(f"+ {fmt(s['added'])} voxels")
        note = "  ".join(parts) if parts else "no voxel changes"
        pct = 100.0 * (s["removed"] + s["added"]) / s["voxels_before_in_stack"]
        n1 = ftext(fig, x + pw / 2, y + head_h + ph + 0.8, note, fontsize=7, fontweight="bold", ha="center")
        n2 = ftext(fig, x + pw / 2, y + head_h + ph + 3.7, f"{pct:.1f}% of {fmt(s['voxels_before_in_stack'])} voxels",
                   fontsize=6.8, color=C_INK2, ha="center")
        checks += [(n1, x, x + pw, note), (n2, x, x + pw, "note2 " + letter)]
    y_row2 = y + head_h + ph + 7.6

    # ------------------------------------------------------------ rows 2-3: the corner branch under each preset
    h2 = 30.0

    def branch_row(key, letter, y):
        n = N["scans"][key]
        p = n["params"]
        per_s = A[f"{key}_show_00_all"]
        Hs, Ws = per_s.shape
        s15, s16 = A[f"{key}_show_15_open15"], A[f"{key}_show_16_corners"]
        s17, s18, s21 = A[f"{key}_show_17_cornero"], A[f"{key}_show_18_corncl"], A[f"{key}_show_21_corners2"]
        s22, s23 = A[f"{key}_show_22_trabadd"], A[f"{key}_show_23_close50"]
        ref = s21 if s21.any() else (s17 if s17.any() else s16)
        zx0, zy0, side = square_window(ref, 30, 170, (Ws, Hs))
        Z = (slice(zy0, zy0 + side), slice(zx0, zx0 + side))
        w_whole = h2 * Ws / Hs
        g = (span - w_whole - 4 * h2) / 4
        c = n["components"]
        title = (f"Corner-recovery branch, {n['site']} scan, {n['preset']} preset: components of at least "
                 f"{fmt(p['corner_min'])} voxels, second closing {p['close2']}")
        if c["n18"]:
            sub = (f"{c['n18']} of the {c['n17']} eroded candidates reach {fmt(p['corner_min'])} voxels "
                   f"({', '.join(fmt(v) for v in c['stage18'])}) and are restored to the trabecular candidate")
        else:
            sub = (f"none of the {c['n17']} eroded candidates reaches {fmt(p['corner_min'])} voxels "
                   f"(largest {fmt(c['stage17'][0])}): nothing is restored")
        yi = y + header(x_left, y, letter, title, sub, x_right, sub_lines=1)
        cols = [
            ("Opening residue (D)\ncorner candidates", f"{fmt(n['counts']['16_corners'])} voxels",
             paint([(s15, C_FILL), (s16, C_ORANGE)], per_s.shape), per_s, None),
            ("Erosion 3\nof the residue", f"{fmt(n['counts']['17_cornero'])} voxels, {c['n17']} components",
             paint([(s16[Z], C_LIGHT), (s17[Z], C_ORANGE)], (side, side)), per_s[Z], "zoom"),
            (f"Components\n≥ {fmt(p['corner_min'])} voxels", f"{fmt(n['counts']['18_corncl'])} voxels kept" + ("" if c["n18"] else " (empty)"),
             paint([(s17[Z] & ~s18[Z], C_VERM), (s18[Z], C_BLUE)], (side, side)), per_s[Z], "zoom"),
            (f"Dilation 3, again ≥ {fmt(p['corner_min'])}\nand ≤ {fmt(p['corner_max'])} voxels", f"{fmt(n['counts']['21_corners2'])} voxels restored",
             paint([(s18[Z], C_LIGHT), (s21[Z], C_BLUE)], (side, side)), per_s[Z], "zoom"),
            (f"Restored + candidate,\nclosing {p['close2']}", f"+ {fmt(n['steps']['close2']['added'])} voxels",
             paint([(s22[Z], C_FILL), (s23[Z] & ~s22[Z], C_BLUE)], (side, side)), per_s[Z], "zoom"),
        ]
        x = x_left
        title_h = 6.4
        for j, (ttl, note, rgb, outline, kind) in enumerate(cols):
            w = w_whole if kind is None else h2
            ax = ax_mm(fig, x, yi + title_h, w, h2)
            show(ax, rgb, outline=outline)
            if kind is None:
                ax.add_patch(patches.Rectangle((zx0 - 0.5, zy0 - 0.5), side, side, fc="none", ec=C_INK, lw=0.6))
                scale_bar(ax, Ws, Hs, 5.0)
            elif j == 1:
                scale_bar(ax, side, side, 2.0)
            t1 = ftext(fig, x + w / 2, yi, ttl, fontsize=6.8, fontweight="bold", ha="center", linespacing=1.15)
            t2 = ftext(fig, x + w / 2, yi + title_h + h2 + 0.7, note, fontsize=6.8, color=C_INK2, ha="center")
            lo, hi = (x - g / 2, x + w + g / 2) if j else (x, x + w + g / 2)
            if j == len(cols) - 1:
                hi = x_right
            checks.extend([(t1, lo, hi, ttl), (t2, lo, hi, note)])
            if j < len(cols) - 1:
                arrow(fig, x + w + 0.4, x + w + g - 0.4, yi + title_h + h2 / 2)
            x += w + g
        return yi + title_h + h2 + 4.4

    y_row3 = branch_row("T", "E", y_row2)
    y_row4 = branch_row("R", "F", y_row3)

    # ------------------------------------------------------------ row 4: panel G (one parameter varied) and H (slicewise)
    h4 = 33.0
    y = y_row4
    imgs = []
    for key in ("T", "R"):
        n = N["scans"][key]
        for label, _, _ in VARIANTS[key]:
            v = n["variants"][label]
            std, alt, per_v = A[f"{key}_var_{label}_std"], A[f"{key}_var_{label}_alt"], A[f"{key}_var_{label}_00"]
            rgb = paint([(std & alt, C_FILL), (std & ~alt, C_VERM), (alt & ~std, C_BLUE)], std.shape)
            what = {"corner_min": "components \u2265", "close2": "second closing"}[v["field"]]
            ttl = f"{n['preset']} scan\n{what} {fmt(v['varied'])}\nstandard: {fmt(v['standard'])}"
            note = f"{fmt(v['cort_differs'])} voxels change" if v["cort_differs"] else "no voxel changes"
            imgs.append((rgb, per_v, ttl, note, std.shape))
    sw = Rr["slicewise"]
    s24, s25, per_sw = A["R_sw_24"], A["R_sw_25"], A["R_sw_00"]
    rem = s24 & ~s25
    rgb_sw = paint([(s25, C_FILL), (rem, C_VERM)], s24.shape)
    hx0, hy0, hside = square_window(rem, 30, 130, (s24.shape[1], s24.shape[0]))
    ZH = (slice(hy0, hy0 + hside), slice(hx0, hx0 + hside))
    widths = [h4 * s[1] / s[0] for _, _, _, _, s in imgs] + [h4]
    gH = 4.0                                            # the gap before panel H (its letter sits in it)
    g4 = (span - sum(widths) - gH) / (len(widths) - 2)
    ftext(fig, x_left, y, "G", fontsize=9, fontweight="bold", ha="left")
    tG = ftext(fig, x_left + 3.6, y + 0.3, "One parameter varied at a time: the final cortical mask under the other site's value",
               fontsize=7.5, fontweight="bold", ha="left")
    xH = x_left + sum(widths[:-1]) + 3 * g4 + gH
    checks.append((tG, x_left, xH - 1, "G title"))
    yi = y + 4.2
    title_h = 9.8
    x = x_left
    for k, (rgb, outline, ttl, note, shp) in enumerate(imgs):
        w = widths[k]
        ax = ax_mm(fig, x, yi + title_h, w, h4)
        show(ax, rgb, outline=outline)
        if k == 0:
            scale_bar(ax, shp[1], shp[0], 5.0)
        t1 = ftext(fig, x + w / 2, yi, ttl, fontsize=6.8, fontweight="bold", ha="center", linespacing=1.15)
        t2 = ftext(fig, x + w / 2, yi + title_h + h4 + 0.7, note, fontsize=6.8, color=C_INK2, ha="center")
        lo, hi = (x - g4 / 2, x + w + g4 / 2) if k else (x, x + w + g4 / 2)
        checks.extend([(t1, lo, hi, ttl), (t2, lo, hi, note)])
        x += w + g4
    # H: the zoom on the slice where the rule acts, with the whole slice inset
    ftext(fig, xH - 3.8, y, "H", fontsize=9, fontweight="bold", ha="left")
    tH = ftext(fig, xH, y + 0.3, "Slice-wise 50% rule", fontsize=7.5, fontweight="bold", ha="left")
    checks.append((tH, xH - 4, x_right, "H title"))
    wH = widths[-1]
    ttl = f"{Rr['preset']} scan, the slice where\nit acts: 4-connected parts\nof the trabecular candidate"
    t1 = ftext(fig, xH + wH / 2, yi, ttl, fontsize=6.8, fontweight="bold", ha="center", linespacing=1.15)
    checks.append((t1, xH - 4, x_right, ttl))
    ax2 = ax_mm(fig, xH, yi + title_h, wH, h4)
    show(ax2, rgb_sw[ZH], outline=per_sw[ZH])
    scale_bar(ax2, hside, hside, 1.0)
    fx0, fy0 = Rr["frame"][0], Rr["frame"][1]
    k_rem = 0
    for c in sw["components"]:
        cx, cy = c["cx"] - fx0 - hx0, c["cy"] - fy0 - hy0
        if c["kept"]:                                  # the main part's centroid lies outside the zoom: label its most interior
            dt = ndi.distance_transform_edt(s25[ZH])   # pixel inside the zoom, kept clear of the zoom's edges
            cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
            near_right = cx > 0.6 * hside
            ax2.text(min(cx, hside - 3) if near_right else cx, cy, f"{c['share']:.1f}%\nkept", ha="right" if near_right else "center",
                     va="center", fontsize=6.5, color=C_INK, linespacing=1.1, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        elif 0 <= cx < hside and 0 <= cy < hside:
            dy = 12 if k_rem % 2 == 0 else -12         # labels alternately above and below the parts
            ax2.annotate(f"{c['share']:.2f}%\nremoved", (cx, cy), xytext=(0, dy), textcoords="offset points", ha="center",
                         va="bottom" if dy > 0 else "top", fontsize=6.5, color=C_INK, linespacing=1.1,
                         bbox=dict(fc="white", ec=C_VERM, lw=0.5, pad=1.0), arrowprops=dict(arrowstyle="-", color=C_VERM, lw=0.5))
            k_rem += 1
    # inset: the whole slice with the zoom window, in the emptiest corner of the zoom
    ins_w = wH * 0.40
    ins_h = ins_w * s24.shape[0] / s24.shape[1]
    q = hside // 2
    occ = {"tl": rgb_sw[ZH][:q, :q], "tr": rgb_sw[ZH][:q, q:], "bl": rgb_sw[ZH][q:, :q], "br": rgb_sw[ZH][q:, q:]}
    corner = min(occ, key=lambda k: float((occ[k] < 0.999).any(axis=2).mean()))
    xi = xH + 0.6 if corner.endswith("l") else xH + wH - ins_w - 0.6
    yi_ins = yi + title_h + 0.6 if corner.startswith("t") else yi + title_h + h4 - ins_h - 0.6
    axi = ax_mm(fig, xi, yi_ins, ins_w, ins_h)
    show(axi, rgb_sw, outline=per_sw, lw=0.25)
    axi.add_patch(patches.Rectangle((hx0 - 0.5, hy0 - 0.5), hside, hside, fc="none", ec=C_INK, lw=0.5))
    for s in axi.spines.values():
        s.set_edgecolor(C_MUTED)
    rem_c = [c for c in sw["components"] if not c["kept"]]
    note = (f"{fmt(sw['slice_total'])} voxels in the slice;\n"
            f"{' and '.join(fmt(c['voxels']) for c in rem_c)} removed (< 50%)")
    t2 = ftext(fig, xH + wH / 2, yi + title_h + h4 + 0.7, note, fontsize=6.8, color=C_INK2, ha="center", linespacing=1.15)
    checks.append((t2, xH - 4, x_right, note))
    y_leg = yi + title_h + h4 + 7.4

    # ------------------------------------------------------------ legend strip
    handles = [patches.Patch(fc=C_FILL, ec="none", label="unchanged by the step"),
               patches.Patch(fc=C_VERM, ec="none", label="removed / not kept"),
               patches.Patch(fc=C_BLUE, ec="none", label="added / kept"),
               patches.Patch(fc=C_ORANGE, ec="none", label="corner candidates"),
               patches.Patch(fc=C_LIGHT, ec="none", label="step input (A: the 6-voxel shell)"),
               Line2D([], [], color=C_PERI, lw=0.8, label="periosteal contour")]
    axl = ax_mm(fig, x_left, y_leg, span, 3.2)
    axl.axis("off")
    leg = axl.legend(handles=handles, loc="center", ncol=6, frameon=False, fontsize=6.5, handlelength=1.4, columnspacing=1.0,
                     handletextpad=0.45, borderaxespad=0)
    fig.canvas.draw()
    bb = leg.get_window_extent(fig.canvas.get_renderer())
    say(f"legend strip: {bb.x0 / fig.dpi * 25.4:.1f} .. {bb.x1 / fig.dpi * 25.4:.1f} mm of {FIG_W}")
    if bb.x0 / fig.dpi * 25.4 < x_left - 0.3 or bb.x1 / fig.dpi * 25.4 > x_right + 0.3:
        say("  WARNING: the legend strip runs past the figure's side margins")
    check_text_widths(fig, checks)
    used = y_leg + 3.2 + 1.0
    say(f"figure height used: {used:.1f} of {FIG_H} mm")
    if used > FIG_H + 0.05:
        say("  WARNING: the drawing runs past the bottom of the figure")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute", action="store_true", help="ignore the cache of the chain runs and crops")
    a = ap.parse_args()
    if not a.recompute and os.path.exists(CACHE + ".json") and os.path.exists(CACHE + ".npz"):
        say("loading cache", CACHE)
        R = load_cache()
    else:
        R = compute()
        save_cache(R)
    fig = draw(R)
    fig.savefig(OUT + ".png", dpi=300, facecolor="white")
    fig.savefig(OUT + ".svg", facecolor="white")
    plt.close(fig)
    say("wrote", OUT + ".png", "and .svg")
    N = dict(figure="S3_step1_presets", width_mm=FIG_W, height_mm=FIG_H, **R["numbers"])
    json.dump(N, open(OUT + "_numbers.json", "w"), indent=1)
    say("numbers", OUT + "_numbers.json")


if __name__ == "__main__":
    main()
