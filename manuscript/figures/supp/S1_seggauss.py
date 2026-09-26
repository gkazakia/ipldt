"""S1_seggauss.py -- Supplementary Figure S1 of the ipldt / ORMIR-BQRL manuscript: IPL's /seg_gauss, which
opens the compartment separation with a dense-bone estimate, and the effect of each of its parameters.

Panels (every panel drawn from real data; the scan is the patella scan of Figure 2, XtremeCT II, 60.7 um):
  A  the smoothing kernel: the taps of the separable Gaussian for sigma 1, 2 (IPL's standard) and 3 voxels at
     support 3, the offsets beyond the support that are not used, and the three 1-D passes stored as int16
  B  the per-scan conversion of the density thresholds (500 and 3,000 mg HA/cm3) to native units through the
     calibration in each scan's header (all 137 scans; the three calibrations of the cohort)
  C  one greyscale row through the cortex: the native values, the smoothed values at sigma 1 / 2 / 3, the native
     thresholds, and the voxels IPL's own export of this stage sets
  D-F  the same slice at sigma 1 / 2 (IPL's standard; identical to IPL's export) / 3 voxels
  G-I  the same slice with the lower threshold at 300 / 700 mg HA/cm3 and the upper threshold at 1,000 mg HA/cm3
       (IPL's standard is 500-3,000; panel E)
  In D-I: grey = voxels set at the shown setting and at IPL's standard setting; blue = set only at the shown
  setting; vermilion = set only at the standard setting.

Run from the repository root in the `ormir` environment (numpy, matplotlib; no GPU needed):

    python manuscript/figures/supp/S1_seggauss.py [--recompute]

Writes manuscript/figures/supp/S1_seggauss.png (300 dpi, 180 mm wide) and S1_seggauss.svg, with the sidecar
S1_seggauss_numbers.json (every number drawn).  The smoothed rows and slices are cached in
manuscript/figures/cache/S1_seggauss_cache.{npz,json} after the first run (about 2-3 min: five smoothings of
the 42.5 M-voxel box); --recompute redoes it.

Inputs (read only):
  * the scan's greyscale AIM (GREY_AIM) and IPL's exports of the periosteal rendering (stage 00) and of the
    /seg_gauss stage (stage 01) of its own evaluation (P16_DIR);
  * validation/results/from_ipl_contour_ceil_dt/records/*.json     (21 patellae: header calibrations);
  * validation/results/oslh_auto_vN/records/<Group>_<Study>_<n>.json  (62 radius / tibia scans: header calibrations);
  * validation/results/oslh_noedit_vN/records/<Group>_<Study>_<n>.json  (54 diaphyseal radius / tibia scans: the same).
Nothing under ipldt/ is modified: every ipldt number comes from calling ipldt.ipl_ops on IPL's files.
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
from matplotlib import patches  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
from ipldt.io import read_aim  # noqa: E402
from ipldt import ipl_ops as ops  # noqa: E402

# ------------------------------------------------------------------------------------------------ inputs
P16_DIR = lab_path("Python/scripts/IPL/probes/p15_gobj_render/aims_and_logs")   # IPL's stage exports
EXPORT_BASE = "X2420448_P16_"                                                                  # <base>_<TAG>.AIM;n
GREY_AIM = lab_path("PFJOA/XCT_masks_full_grab/PFJ-0be66a_R/X2420448.AIM")           # the scan's greyscale
PATELLA_RECORDS = os.path.join(REPO, "validation", "results", "from_ipl_contour_ceil_dt", "records")
OSLH_RECORDS = [os.path.join(REPO, "validation", "results", d, "records") for d in (OSLH_AUTO, OSLH_NOEDIT)]
OSLH_RX = re.compile(r"^[A-Za-z]+_[A-Za-z]+_\d+\.json$")
OSLH_SKIP = {"Diaphyseal_CKD_991161"}          # a record that exists but is not part of the validation set
CACHE = os.path.join(os.path.dirname(HERE), "cache", "S1_seggauss_cache")
OUT = os.path.join(HERE, "S1_seggauss")

VOX_MM = 0.0607
Z_GLOBAL, Y_GLOBAL, X0_GLOBAL, X1_GLOBAL = 252, 197, 851, 919      # the slice and the row of panel C (global voxel indices)
SIGMA_STD, SUPPORT_STD, LOWER_STD, UPPER_STD = 2.0, 3, 500.0, 3000.0   # IPL's /seg_gauss setting (Scripts 32 / 33)

# the eight settings computed: key, sigma, support, lower / upper thresholds (mg HA/cm3); the standard first,
# then the settings that share its smoothing, then the other kernels (one smoothed volume is held at a time)
VARIANTS = [
    ("std", 2.0, 3, 500.0, 3000.0),
    ("lo300", 2.0, 3, 300.0, 3000.0),
    ("lo700", 2.0, 3, 700.0, 3000.0),
    ("up1000", 2.0, 3, 500.0, 1000.0),
    ("s1", 1.0, 3, 500.0, 3000.0),
    ("s3", 3.0, 3, 500.0, 3000.0),
    ("sup2", 2.0, 2, 500.0, 3000.0),
    ("sup4", 2.0, 4, 500.0, 3000.0),
]
KERNELS = [(1.0, 3), (2.0, 3), (3.0, 3), (2.0, 2), (2.0, 4)]
SITES = ["patella", "UD radius", "UD tibia", "D radius", "D tibia"]
SITE_LABEL = {"patella": "patella", "UD radius": "UD radius", "UD tibia": "UD tibia", "D radius": "D radius", "D tibia": "D tibia"}
N_SCANS = 137
N_SITE = {"patella": 21, "UD radius": 29, "UD tibia": 29, "D radius": 29, "D tibia": 29}

# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito (colour-blind safe, the palette of every figure)
C_BLUE, C_VERM, C_GREEN, C_PURPLE, C_ORANGE = "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_FILL = "#b9b8b3"           # voxels set at both settings (D-I)
C_SHADE = "#e8e7e2"          # the voxels IPL's export sets (C) / the offsets beyond the support (A)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.5, "axes.edgecolor": C_AXIS, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2, "xtick.color": C_INK2,
    "ytick.color": C_INK2, "savefig.dpi": 300, "figure.dpi": 100, "pdf.fonttype": 42, "svg.fonttype": "none",
    "text.color": C_INK, "axes.labelcolor": C_INK2, "legend.frameon": False,
})

T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def find_export(tag):
    """The newest VMS version (';n') of <EXPORT_BASE><TAG>.AIM in P16_DIR."""
    rx = re.compile("^" + re.escape(EXPORT_BASE + tag) + r"\.AIM(?:;(\d+))?$", re.I)
    cands = [(int(m.group(1) or 0), f) for f in os.listdir(P16_DIR) for m in [rx.match(f)] if m]
    if not cands:
        raise FileNotFoundError(f"no export for {tag} in {P16_DIR}")
    return os.path.join(P16_DIR, max(cands)[1])


def native(mgha, cal):
    return ops.mgha_to_native(mgha, cal["slope"], cal["intercept"], cal["mu_scaling"])


def exact_native(mgha, cal):
    return (float(mgha) - cal["intercept"]) / cal["slope"] * cal["mu_scaling"]


def fmt(n):
    """An integer with comma thousands separators, as in the legends and the text."""
    return f"{int(n):,d}"


def fmt2(x):
    """A number with two decimals and comma thousands separators."""
    return f"{x:,.2f}".replace("-", "\u2212")


def signed(n):
    return ("+" if n >= 0 else "\u2212") + fmt(abs(n))


# ------------------------------------------------------------------------------------------------ compute
def compute():
    """The eight settings of /seg_gauss on the scan: whole-volume counts, the slice and the row of each."""
    say("reading the greyscale and IPL's stage 00 / 01 exports")
    grey = read_aim(GREY_AIM)
    all_ = read_aim(find_export("00_ALL"))
    s01 = read_aim(find_export("01_SEGGAUSS"))
    cal = ops.calibration_from_proclog(grey)
    bbc = ops.bounding_box_cut(ops.mask_by_gobj(grey, ops.set_value(all_, ops.SET, 0)))
    del grey, all_
    box = np.asarray(bbc["data"])
    px, py, pz = (int(v) for v in bbc["pos"])
    qx, qy, qz = (int(v) for v in s01["pos"])
    assert (qx, qy, qz) == (px + 3, py + 3, pz + 3) and tuple(s01["dim"]) == tuple(d - 6 for d in bbc["dim"])
    ipl = np.asarray(s01["data"]) != 0                       # IPL's stage 01, on the box shrunk by 3
    N = dict(calibration=cal, box_dim=[int(d) for d in bbc["dim"]], box_pos=[px, py, pz],
             stage_dim=[int(d) for d in s01["dim"]], stage_pos=[qx, qy, qz], voxels_compared=int(ipl.size),
             z_global=Z_GLOBAL, y_global=Y_GLOBAL, x_global=[X0_GLOBAL, X1_GLOBAL], vox_mm=VOX_MM,
             thresholds_native={}, thresholds_exact={}, kernels={}, variants={})
    for mg in sorted({v[3] for v in VARIANTS} | {v[4] for v in VARIANTS}):
        N["thresholds_native"][str(int(mg))] = native(mg, cal)
        N["thresholds_exact"][str(int(mg))] = exact_native(mg, cal)
    say(f"calibration {cal}; native thresholds {N['thresholds_native']}")
    for s, r in KERNELS:
        w = ops.ipl_gauss_weights(s, r)
        k = np.arange(-r, r + 1)
        g = np.exp(-k.astype(float) ** 2 / (2 * s * s))
        full = np.exp(-np.arange(-30, 31).astype(float) ** 2 / (2 * s * s)).sum()
        N["kernels"][f"{s:g}_{r}"] = dict(sigma=s, support=r, taps=[float(x) for x in w],
                                          mass_beyond_support=float(1 - g.sum() / full))
    zs, ys = Z_GLOBAL - qz, Y_GLOBAL - qy                    # the slice and the row on the stage grid
    xs = slice(X0_GLOBAL - qx, X1_GLOBAL - qx)
    row_native = box[Z_GLOBAL - pz, Y_GLOBAL - py, X0_GLOBAL - px:X1_GLOBAL - px].astype(np.int32)
    slices, rows = {}, {}
    std = None
    smoothed = {}
    for key, s, r, lo_mg, up_mg in VARIANTS:
        if (s, r) not in smoothed:
            say(f"smoothing sigma {s:g} support {r}")
            smoothed = {(s, r): ops.gauss_lp_ipl(box, s, r)}    # one smoothed volume at a time
        sm = smoothed[(s, r)][3:-3, 3:-3, 3:-3]                  # on the stage grid
        lo, up = native(lo_mg, cal), native(up_mg, cal)
        m = (sm >= np.float32(lo)) & (sm <= np.float32(up))
        c = max(r, 3) - 3                                    # the region where the setting is valid, on the stage grid
        valid = (slice(c, m.shape[0] - c), slice(c, m.shape[1] - c), slice(c, m.shape[2] - c))
        rec = dict(sigma=s, support=r, lower_mgha=lo_mg, upper_mgha=up_mg, lower_native=lo, upper_native=up,
                   voxels_set=int(m[valid].sum()), valid_margin=c)
        if key == "std":
            std = m
            d = m != ipl
            rec.update(ipl_voxels_set=int(ipl.sum()), differing_from_ipl=int(d.sum()),
                       differing_on_slice=int(d[zs].sum()), smoothed_max=int(sm.max()),
                       above_upper=int((sm > np.float32(up)).sum()), slice_added=0, slice_removed=0)
            say(f"  standard: {rec['voxels_set']:,d} set, IPL {rec['ipl_voxels_set']:,d}, differing {rec['differing_from_ipl']}"
                f" of {ipl.size:,d}; smoothed max {rec['smoothed_max']}, above the upper bound {rec['above_upper']}")
            assert rec["differing_from_ipl"] == 0, "the standard setting must reproduce IPL's export"
        else:
            a, b = m[valid] & ~std[valid], std[valid] & ~m[valid]
            rec.update(added_vs_std=int(a.sum()), removed_vs_std=int(b.sum()), std_voxels_in_region=int(std[valid].sum()),
                       slice_added=int((m[zs] & ~std[zs]).sum()), slice_removed=int((std[zs] & ~m[zs]).sum()))
            say(f"  {key:7s} sigma {s:g} support {r} {lo_mg:g}-{up_mg:g}: {rec['voxels_set']:,d} set, "
                f"+{rec['added_vs_std']:,d} / -{rec['removed_vs_std']:,d} vs the standard")
        rec["slice_set"] = int(m[zs].sum())
        slices[key] = m[zs].copy()
        rows[key] = sm[zs, ys, xs].astype(np.int32)
        N["variants"][key] = rec
        del m
    ipl_slice, ipl_row = ipl[zs].copy(), ipl[zs, ys, xs].copy()
    N["variants"]["std"]["ipl_slice_set"] = int(ipl_slice.sum())
    N["variants"]["std"]["ipl_row_set"] = int(ipl_row.sum())
    return dict(numbers=N, slices=slices, rows=rows, row_native=row_native, ipl_slice=ipl_slice, ipl_row=ipl_row)


def save_cache(R):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE + ".npz", row_native=R["row_native"], ipl_slice=R["ipl_slice"], ipl_row=R["ipl_row"],
                        **{f"slice_{k}": v for k, v in R["slices"].items()}, **{f"row_{k}": v for k, v in R["rows"].items()})
    json.dump(R["numbers"], open(CACHE + ".json", "w"), indent=1)


def load_cache():
    z = np.load(CACHE + ".npz")
    keys = [v[0] for v in VARIANTS]
    return dict(numbers=json.load(open(CACHE + ".json")), row_native=z["row_native"], ipl_slice=z["ipl_slice"],
                ipl_row=z["ipl_row"], slices={k: z[f"slice_{k}"] for k in keys}, rows={k: z[f"row_{k}"] for k in keys})


def load_calibrations():
    """One row per scan (137): site, the header calibration and the native thresholds recorded by the runs."""
    rows = []
    for f in sorted(glob.glob(os.path.join(PATELLA_RECORDS, "*.json"))):
        d = json.load(open(f))
        c, t = d["step1"]["calibration"], d["step1"]["thresholds"]
        rows.append(dict(site="patella", key=d["subject"], slope=c["slope"], intercept=c["intercept"],
                         mu=c["mu_scaling"], lower=int(t["lower_native"]), upper=int(t["upper_native"])))
    seen = set()
    for f in sorted(f for d_ in OSLH_RECORDS for f in glob.glob(os.path.join(d_, "*.json"))):
        if not OSLH_RX.match(os.path.basename(f)):
            continue
        d = json.load(open(f))
        if d["tag"] in OSLH_SKIP:
            continue
        assert d["tag"] not in seen, f"{d['tag']} is in two record directories"
        seen.add(d["tag"])
        c, t = d["B"]["step1"]["calibration"], d["B"]["step1"]["thresholds"]
        rows.append(dict(site=f"{d['meta']['region']} {d['meta']['bone'].lower()}", key=d["tag"], slope=c["slope"],
                         intercept=c["intercept"], mu=c["mu_scaling"], lower=int(t["lower_native"]),
                         upper=int(t["upper_native"])))
    order = {s: i for i, s in enumerate(SITES)}
    rows.sort(key=lambda r: (order[r["site"]], r["key"]))
    n_site = {s: sum(r["site"] == s for r in rows) for s in SITES}
    assert len(rows) == N_SCANS and n_site == N_SITE, n_site
    for r in rows:      # the recorded native numbers are the rounded conversions of the recorded calibration
        cal = dict(slope=r["slope"], intercept=r["intercept"], mu_scaling=r["mu"])
        assert r["lower"] == native(LOWER_STD, cal) and r["upper"] == native(UPPER_STD, cal), r
    return rows


# ------------------------------------------------------------------------------------------------ drawing
FIG_W, FIG_H = 180.0, 139.0          # mm


def ax_mm(fig, x, y_top, w, h):
    """Axes at (x, y_top) mm from the top-left corner of the figure, w x h mm."""
    return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H])


def ftext(fig, x, y_top, s, **kw):
    """fig.text at (x, y_top) in mm from the top-left corner."""
    kw.setdefault("va", "top")
    return fig.text(x / FIG_W, 1 - y_top / FIG_H, s, **kw)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


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


def panel_head(fig, x, y, letter, title, note, w, checks):
    """Panel letter (bold, upper-left), title beside it, a note of up to three lines under it."""
    ftext(fig, x, y, letter, fontsize=9, fontweight="bold", ha="left")
    t1 = ftext(fig, x + 3.4, y + 0.25, title, fontsize=7.5, fontweight="bold", ha="left")
    checks.append((t1, x, x + w, title))
    if note:
        t2 = ftext(fig, x, y + 3.9, note, fontsize=6.3, color=C_INK2, ha="left", linespacing=1.12)
        checks.append((t2, x, x + w, note))


def draw(R, scans):
    N, S, ROWS = R["numbers"], R["slices"], R["rows"]
    V, TN = N["variants"], N["thresholds_native"]
    H, W = S["std"].shape
    fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4))
    checks = []
    y1, h1, head1 = 0.8, 47.0, 11.0              # row 1: letters at y1, axes from y1 + head1 to y1 + h1

    # ------------------------------------------------------------------------------- A: the kernel
    xA, wA = 8.5, 47.0
    n_sup2 = V["sup2"]["added_vs_std"] + V["sup2"]["removed_vs_std"]
    n_sup4 = V["sup4"]["added_vs_std"] + V["sup4"]["removed_vs_std"]
    panel_head(fig, 1.5, y1, "A", "Smoothing kernel: \u03c3 and support",
               f"one 1D pass along x, then y, then z, each\ntruncated to int16; support 2 / 4 instead of 3\n"
               f"changes {fmt(n_sup2)} / {fmt(n_sup4)} voxels", wA + 7, checks)
    ax = ax_mm(fig, xA, y1 + head1, wA, h1 - head1)
    ax.add_patch(patches.Rectangle((3.5, 0), 1.5, 1, transform=ax.get_xaxis_transform(), color=C_SHADE, lw=0, zorder=0))
    ax.add_patch(patches.Rectangle((-5, 0), 1.5, 1, transform=ax.get_xaxis_transform(), color=C_SHADE, lw=0, zorder=0))
    for kx_ in (4.25, -4.25):
        ax.text(kx_, 0.012, "beyond\nsupport 3:\nnot used", ha="center", va="bottom", fontsize=6.0, color=C_INK2, linespacing=1.05)
    kk = np.linspace(-5, 5, 401)
    for (s, r), col, lw, lab in (((1.0, 3), C_GREEN, 0.8, "\u03c3 = 1"), ((2.0, 3), C_BLUE, 1.5, "\u03c3 = 2 (IPL)"),
                                 ((3.0, 3), C_PURPLE, 0.8, "\u03c3 = 3")):
        w = np.array(N["kernels"][f"{s:g}_{r}"]["taps"])
        k = np.arange(-r, r + 1)
        ax.plot(kk, w[r] * np.exp(-kk ** 2 / (2 * s * s)), color=col, lw=0.6, alpha=0.5, zorder=2)
        ax.vlines(k, 0, w, color=col, lw=lw, zorder=3)
        ax.plot(k, w, "o", color=col, ms=3.2 if s == 2 else 2.4, zorder=4, label=lab)
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.62)
    ax.set_xticks(range(-4, 5))
    ax.set_xlabel("tap offset (voxels)", labelpad=1.5)
    ax.set_ylabel("tap weight", labelpad=1.5)
    ax.tick_params(axis="both", labelsize=6.0, pad=1.5)
    ax.grid(axis="y", lw=0.3, color=C_GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.2, handletextpad=0.5, borderaxespad=0.3, labelspacing=0.25)
    w2 = N["kernels"]["2_3"]["taps"]
    m2 = N["kernels"]["2_3"]["mass_beyond_support"] * 100
    tA = ax.text(0.985, 0.975, f"\u03c3 = 2, support 3:\ntaps at offset\n0, \u00b11, \u00b12, \u00b13 =\n"
                 f"{w2[3]:.4f}, {w2[4]:.4f},\n{w2[5]:.4f}, {w2[6]:.4f};\n{m2:.1f}% of this\nGaussian lies\nbeyond support 3\n(taps renormalized)",
                 transform=ax.transAxes, fontsize=6.0, color=C_INK2, ha="right", va="top", linespacing=1.12)

    # ------------------------------------------------------------------------------- B: the conversion
    xB, wB = 69.0, 41.0                            # room left of the strips for the tick labels and the rotated labels
    nS = len(scans)
    panel_head(fig, 59.5, y1, "B", "Thresholds in native units, per scan",
               "native = round((\u03c1 \u2212 b) / a \u00d7 8192); a, b read from\n"
               f"each scan's header; one dot per scan ({nS}), by site",
               57.0, checks)
    groups = {}
    for r in scans:
        groups.setdefault((r["slope"], r["intercept"], r["mu"]), []).append(r)
    bounds, xb = [], 0
    for s in SITES:
        n = sum(r["site"] == s for r in scans)
        bounds.append((s, xb, xb + n))
        xb += n
    hB, gapB, foot = 11.2, 2.6, 7.2
    for j, (which, mg) in enumerate((("lower", 500), ("upper", 3000))):
        ax = ax_mm(fig, xB, y1 + head1 + 3.4 + j * (hB + gapB), wB, hB)
        vals = np.array([r[which] for r in scans])
        ax.scatter(np.arange(nS), vals, s=2.2, color=C_INK2, zorder=3, lw=0)
        levels = sorted({int(v) for v in vals})
        lo_, hi_ = min(levels), max(levels)
        pad = (hi_ - lo_) * 0.5
        ax.set_ylim(lo_ - pad, hi_ + pad)
        for lev in levels:
            n = int((vals == lev).sum())
            ax.axhline(lev, color=C_GRID, lw=0.4, zorder=1)
            ax.text(nS + 1.5, lev, f"n = {n}", fontsize=5.6, color=C_INK2, va="center", ha="left")
        for s, a, b in bounds:
            if a > 0:
                ax.axvline(a - 0.5, color=C_GRID, lw=0.4, zorder=1)
        ax.set_xlim(-1, nS)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=5.6, pad=1.5)
        ax.yaxis.set_major_locator(plt.FixedLocator(levels))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: fmt(v)))
        ax.set_ylabel(f"{fmt(mg)}\nmg HA/cm\u00b3", labelpad=3.0, fontsize=5.6, linespacing=1.15)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if j == 0:
            for s, a, b in bounds:
                if a > 0:           # the site boundary carried up between the labels
                    ax.plot([a - 0.5, a - 0.5], [1.0, 1.24], transform=ax.get_xaxis_transform(), color=C_AXIS, lw=0.4,
                            clip_on=False)
                if SITE_LABEL[s]:
                    t = ax.text((a + b - 1) / 2, 1.05, SITE_LABEL[s], transform=ax.get_xaxis_transform(), fontsize=5.4,
                                color=C_INK2, ha="center", va="bottom")
                    checks.append((t, xB + wB * (a - 0.5 + 1) / (nS + 1) - 0.6, xB + wB * (b - 0.5 + 1) / (nS + 1) + 0.6,
                                   f"site label {s}"))
    cal_lines = ["the three calibrations (a, b) and the exact values before rounding:"]
    for (a, b, mu), rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        calg = dict(slope=a, intercept=b, mu_scaling=mu)
        cal_lines.append(f"n = {len(rs)}:  a = {fmt2(a)}, b = {fmt2(b)}  \u2192  {fmt2(exact_native(500, calg))}, "
                         f"{fmt2(exact_native(3000, calg))}")
    tB = ftext(fig, 59.5, y1 + h1 - foot + 0.8, "\n".join(cal_lines), fontsize=5.4, color=C_INK2, ha="left", va="top",
               linespacing=1.12)
    checks.append((tB, 59.5, xB + wB + 12, "calibration list"))

    # ------------------------------------------------------------------------------- C: the profile
    xC, wC = 125.5, 53.0
    panel_head(fig, 118.0, y1, "C", "One row through the cortex",
               f"native and smoothed values (\u03c3 = 1 / 2 / 3) and this scan's\nnative thresholds; row marked in E. 3,000 mg HA/cm\u00b3 =\n"
               f"{fmt(TN['3000'])} is off scale (largest smoothed value here: {fmt(V['std']['smoothed_max'])})",
               62.0, checks)
    ax = ax_mm(fig, xC, y1 + head1, wC, h1 - head1)
    gx = np.arange(X0_GLOBAL, X1_GLOBAL)
    nat = R["row_native"]
    kept = R["ipl_row"]
    for x_ in gx[kept]:
        ax.add_patch(patches.Rectangle((x_ - 0.5, 0), 1, 1, transform=ax.get_xaxis_transform(), color=C_SHADE, lw=0, zorder=0))
    ax.step(gx, nat, where="mid", color=C_MUTED, lw=0.6, zorder=2)
    for key, col, lw in (("s1", C_GREEN, 0.8), ("std", C_BLUE, 1.5), ("s3", C_PURPLE, 0.8)):
        ax.plot(gx, ROWS[key], color=col, lw=lw, zorder=3)
    for mg, ls, lw in ((500, "--", 0.8), (300, ":", 0.6), (700, ":", 0.6)):
        ax.axhline(TN[str(mg)], color=C_INK, lw=lw, ls=ls, zorder=2)
    ymax = 12500                                   # head-room: the legend sits above the profile and the 700 line
    ax.set_ylim(0, ymax)
    ax.set_yticks(range(0, 8001, 2000))
    ax.set_xlim(gx[0] - 0.5, gx[-1] + 0.5)
    ax.set_xlabel("voxel along the row (0.0607 mm per voxel)", labelpad=1.5)
    ax.set_ylabel("value (native units)", labelpad=1.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: fmt(v)))
    ax.tick_params(axis="both", labelsize=6.0, pad=1.5)
    ax.grid(axis="y", lw=0.3, color=C_GRID)
    ax.set_axisbelow(True)
    hC = [Line2D([], [], color=C_MUTED, lw=0.6, label="native grayscale"),
          Line2D([], [], color=C_GREEN, lw=0.8, label="smoothed, \u03c3 = 1"),
          Line2D([], [], color=C_BLUE, lw=1.5, label="smoothed, \u03c3 = 2 (IPL)"),
          Line2D([], [], color=C_PURPLE, lw=0.8, label="smoothed, \u03c3 = 3"),
          patches.Patch(fc=C_SHADE, ec=C_MUTED, lw=0.5, label="set in IPL's export"),
          Line2D([], [], color=C_INK, lw=0.8, ls="--", label=f"500 mg HA/cm\u00b3 = {fmt(TN['500'])} (IPL)"),
          Line2D([], [], color=C_INK, lw=0.6, ls=":", label=f"300 / 700 mg HA/cm\u00b3 = {fmt(TN['300'])} / {fmt(TN['700'])}")]
    ax.legend(handles=hC, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=5.0, handlelength=1.4, handletextpad=0.5,
              borderaxespad=0.8, labelspacing=0.15, frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
              borderpad=0.3).set_zorder(5)

    # ------------------------------------------------------------------------------- D-I: the slices
    ncol, x_left, gap, title_h, row_gap = 3, 1.5, 3.6, 12.0, 3.0
    pw = (FIG_W - 2 * x_left - (ncol - 1) * gap) / ncol
    ph = pw * H / W
    y0 = y1 + h1 + 7.0
    std = S["std"]
    s = V["std"]
    lo_up = f"native {fmt(TN['500'])}\u2013{fmt(TN['3000'])}"
    panels = [
        ("s1", "D", "\u03c3 = 1 voxel", "support 3, 500\u20133,000 mg HA/cm\u00b3 (as E)"),
        ("std", "E", "IPL standard: \u03c3 = 2, support 3", f"500\u20133,000 mg HA/cm\u00b3 ({lo_up})"),
        ("s3", "F", "\u03c3 = 3 voxels", "support 3, 500\u20133,000 mg HA/cm\u00b3 (as E)"),
        ("lo300", "G", "Lower threshold 300 mg HA/cm\u00b3", f"\u03c3 = 2, support 3; native {fmt(TN['300'])}\u2013{fmt(TN['3000'])}"),
        ("lo700", "H", "Lower threshold 700 mg HA/cm\u00b3", f"\u03c3 = 2, support 3; native {fmt(TN['700'])}\u2013{fmt(TN['3000'])}"),
        ("up1000", "I", "Upper threshold 1,000 mg HA/cm\u00b3", f"\u03c3 = 2, support 3; native {fmt(TN['500'])}\u2013{fmt(TN['1000'])}"),
    ]
    for k, (key, letter, title, line1) in enumerate(panels):
        r_, c_ = divmod(k, ncol)
        x = x_left + c_ * (pw + gap)
        yt = y0 + r_ * (title_h + ph + row_gap)
        v = V[key]
        if key == "std":
            note = (f"{line1}\nslice: {fmt(v['slice_set'])} voxels, 0 differ from IPL's export\n"
                    f"whole volume: 0 of {fmt(N['voxels_compared'])} voxels differ")
        else:
            note = (f"{line1}\nslice: {signed(v['slice_added'])} / {signed(-v['slice_removed'])} voxels vs E\n"
                    f"whole volume: {signed(v['added_vs_std'])} / {signed(-v['removed_vs_std'])} voxels vs E")
        panel_head(fig, x, yt, letter, title, note, pw, checks)
        ax = ax_mm(fig, x, yt + title_h, pw, ph)
        m = S[key]
        rgb = np.ones(m.shape + (3,))
        rgb[m & std] = hex_rgb(C_FILL)
        rgb[m & ~std] = hex_rgb(C_BLUE)
        rgb[~m & std] = hex_rgb(C_VERM)
        ax.imshow(rgb, interpolation="antialiased", origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(C_AXIS)
            sp.set_linewidth(0.4)
        if key == "std":
            qx, qy = N["stage_pos"][0], N["stage_pos"][1]
            ax.plot([X0_GLOBAL - qx - 0.5, X1_GLOBAL - qx - 0.5], [Y_GLOBAL - qy] * 2, color=C_ORANGE, lw=1.0,
                    solid_capstyle="butt", zorder=5)
            ax.text(X0_GLOBAL - qx - 2, Y_GLOBAL - qy - 6, "row of C", fontsize=6.0, color=C_ORANGE, ha="left", va="bottom")
            n = 5.0 / VOX_MM
            ax.plot([W - 14 - n, W - 14], [H - 16, H - 16], color=C_INK, lw=1.2, solid_capstyle="butt")
            ax.text(W - 14 - n / 2, H - 21, "5 mm", fontsize=6, color=C_INK, ha="center", va="bottom")
    # colour key for D-I
    yk = y0 + 2 * (title_h + ph + row_gap) - 1.0
    kx = x_left
    for col, lab in ((C_FILL, "set at the shown setting and at the standard (E)"), (C_BLUE, "set only at the shown setting"),
                     (C_VERM, "set only at the standard setting")):
        fig.patches.append(patches.Rectangle((kx / FIG_W, 1 - (yk + 2.3) / FIG_H), 2.6 / FIG_W, 2.3 / FIG_H,
                                             transform=fig.transFigure, color=col, lw=0))
        t = ftext(fig, kx + 3.3, yk - 0.15, lab, fontsize=6.5, color=C_INK2, ha="left")
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        kx = bb.x1 / fig.dpi * 25.4 + 4.5

    bad = check_text_widths(fig, checks + [(tA, xA, xA + wA, "note A")])
    os.makedirs(HERE, exist_ok=True)
    fig.savefig(OUT + ".png", dpi=300, facecolor="white")
    fig.savefig(OUT + ".svg", facecolor="white")
    plt.close(fig)
    say(f"wrote {OUT}.png / .svg ({FIG_W:.0f} x {FIG_H:.0f} mm, 300 dpi); {bad} text overflow warnings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    if a.recompute or not (os.path.exists(CACHE + ".npz") and os.path.exists(CACHE + ".json")):
        R = compute()
        save_cache(R)
    else:
        R = load_cache()
        say("cache loaded")
    scans = load_calibrations()
    groups = {}
    for r in scans:
        groups.setdefault((r["slope"], r["intercept"], r["mu"], r["lower"], r["upper"]), []).append(r["key"])
    R["numbers"]["cohort_calibrations"] = [
        dict(slope=k[0], intercept=k[1], mu_scaling=k[2], lower_native=k[3], upper_native=k[4], n=len(v), scans=v)
        for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
    draw(R, scans)
    json.dump(R["numbers"], open(OUT + "_numbers.json", "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
