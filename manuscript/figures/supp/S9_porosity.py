"""S9_porosity.py -- Supplementary Figure S9 of the ipldt / ORMIR-BQRL manuscript: IPL's cortical pore
cascade (the Burghardt, Bone 2010 block of the evaluation script that writes <base>_PORE.AIM) as
ipldt.porosity reimplements it, stage by stage, with the reading of /hysteresis_threshold and its refuted
alternatives measured against IPL's own export.

Panels
  A  the two inputs on one slice of a diaphyseal tibia: IPL's rendered cortical compartment
     (/gobj_to_aim of CORT_MASK.GOBJ) and the cortical segmentation CORT_SEG inside it; the window of D and
     F-H is marked;
  B  the first pore estimate: the inverted CORT_SEG, its 4-connected components in this slice and the
     slice-wise 0-5 % rule (/cl_slicewise_extractow) that keeps the small ones at value 1 (pores_E);
  C  the marrow component: /cl_rank_extract -first_rank 1 -last_rank 1 -connect_boundary false of "everything
     that is not cortical bone", and the part of it the gobj mask keeps inside the compartment (pores_M);
  D  the labeled intermediate cortseg_CDE on the window, one color per value (0, 1, 2, 3, 125, 127), with
     the two real z-columns of E marked;
  E  the value encoding and the rule of /hysteresis_threshold: what each value means, how it arises, its
     count on this scan, whether it is a seed / weak / neither voxel, and two real columns of the scan that
     show the +-z growth of -grow_axes 0 0 1 accepting one run and declining another;
  F  the hysteresis result: pores_F0 inside the compartment, the marrow part /subtract_aims removes, and
     pores_F, the voxels the hysteresis contributes;
  G  the final estimate: the second slice-wise pass (pores_DF), pores_G and the 20-voxel component filter
     (/cl_nr_extract -min_number 20) that leaves pores_H;
  H  IPL's own PORE.AIM of the same scan with ipldt's result as an outline, and the differing voxels;
  I  the reading of /hysteresis_threshold: the mismatching voxels against IPL's PORE.AIM for the accepted
     reading and for 15 refuted candidates, on two scans (the scan of A-H and the ultradistal tibia that pins
     the inclusive low bound);
  J  Ct.Po over the cohort: ipldt's |PORE and CORT_MASK| / |CORT_MASK| against the value IPL's result sheet
     prints, for every scan of the validation set that has a sheet carrying values, with the refuted reading
     Ct.Po.V / (Ct.Po.V + Ct.BV) shown for the same scans.

The scan of A-H is Diaphyseal/CKD/2422 (X6435585, XtremeCT II diaphyseal tibia, 60.7 um): a diaphysis is
chosen because the hysteresis does nothing on a patella (value 2, the only thing it can grow into, is empty
on every patella of the cohort), so a patella would not exercise the panel that the figure is about.  That
claim is measured, not assumed: the cascade is run on all 21 patellae (IPL's contour rendering and CORT_SEG
in, 0 voxels differing from IPL's PORE.AIM asserted on each) and the value counts of cortseg_CDE are written
to S9_porosity_numbers.json under "patella_labels" (cached in cache/S9_porosity_patella.json).
Slice numbers in the figure are 1-based (slice 1 is z = 0), as in Figure 3; x, y are 0-based voxel indices.

Run from the repository root in the `ormir` environment (numpy, scipy, matplotlib; no GPU needed):

    set PYTHONUTF8=1
    python manuscript/figures/supp/S9_porosity.py [--recompute]

Writes manuscript/figures/supp/S9_porosity.png (300 dpi, 180 mm wide), S9_porosity.svg and
S9_porosity_numbers.json (every number drawn).  The crops and counts are cached in
manuscript/figures/cache/S9_porosity_cache.{npz,json} (about 2 min on the first run: 16 candidate readings of
the cascade on each of two scans); --recompute redoes everything.

Inputs (read only): the delivered IPL products of the two OS_LH measurements (CORT_MASK_CT.AIM, CORT_SEG.AIM,
PORE.AIM); for panel J, the porosity records of the 137-scan validation set
(validation/results/{oslh_auto_vN,porosity_AB_patella,oslh_noedit_vN}/records/*.json, Diaphyseal/CKD/991161
excluded as everywhere else) and IPL's printed result-sheet values validation/results/ipl_printed_values_137.csv.
A printed Ct.Po is used only from a readable single-measurement sheet (sheet_ok == "yes"): a 'followup' sheet
prints the common region of two measurements, not the scan -- the rule of
manuscript/facts/make_facts_bmd_ctpo.py.  Every ipldt number of A-I is computed here by calling ipldt.porosity
and ipldt.ipl_ops on those files; nothing under ipldt/ or validation/ is modified.
"""
from __future__ import annotations

import argparse
import csv
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
from ipldt import ipl_ops as ops  # noqa: E402
from ipldt import porosity  # noqa: E402
from ipldt.io import read_aim  # noqa: E402

# ------------------------------------------------------------------------------------------------ inputs
OSLH = lab_path("Cross_validation_IPL/OS_LH_AUTO")
MAIN = dict(id="Diaphyseal/CKD/2422", folder=f"{OSLH}/Diaphyseal/CKD/2422", base="X6435585", site="diaphyseal tibia")
SECOND = dict(id="Distal/REPRO/2051", folder=f"{OSLH}/Distal/REPRO/2051", base="X4829835", site="ultradistal tibia")
REC_DIRS = [os.path.join(REPO, "validation", "results", d)
            for d in (OSLH_AUTO, "porosity_AB_patella", OSLH_NOEDIT)]
PRINTED_CSV = os.path.join(REPO, "validation", "results", "ipl_printed_values_137.csv")
EXPECT_N = 137                   # manuscript/facts/facts.json cohort.n_total
EXPECT_SITES = {"patella": 21, "ultradistal radius": 29, "ultradistal tibia": 29, "diaphyseal radius": 29,
                "diaphyseal tibia": 29}
CACHE = os.path.join(REPO, "manuscript", "figures", "cache", "S9_porosity_cache")
CACHE_PATELLA = os.path.join(REPO, "manuscript", "figures", "cache", "S9_porosity_patella.json")
OUT = os.path.join(HERE, "S9_porosity")

Z_SLICE = 101                    # the slice of A-H (global z, 0-based; both volumes start at z = 0); printed 1-based
WIN = 100                        # the window of D and F-H (voxels)
WIN_Y0, WIN_X0 = 306, 186        # its origin in the contour grid of the main scan
SLICE_MARGIN = 6                 # voxels kept around the compartment's bounding box in A-C
LOW, HIGH = porosity.SCRIPT32_HYSTERESIS["low_thresh"], porosity.SCRIPT32_HYSTERESIS["high_thresh"]
LABEL_VALUES = (0, 1, 2, 3, 125, 127)

# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito (colour-blind safe, the palette of every figure of the paper)
C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_PURPLE, C_SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_FILL = "#b9b8b3"               # the compartment / the object
C_BONE = "#6f6e6a"               # cortical bone (CORT_SEG)
C_PALE = "#f1f0ec"               # value 0
C_VERM_L = "#f4d5c2"             # the large slice-wise component that the 0-5 % rule drops
C_SKY_L = "#cfe6f7"              # the rank-1 component outside the compartment
VALUE_COLOUR = {0: C_PALE, 1: C_SKY, 2: C_ORANGE, 3: C_GREEN, 125: C_PURPLE, 127: C_FILL}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.5, "axes.edgecolor": C_AXIS, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.color": C_INK2, "ytick.color": C_INK2, "savefig.dpi": 300,
    "figure.dpi": 100, "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": C_INK, "axes.labelcolor": C_INK2,
    "hatch.linewidth": 0.4, "hatch.color": "#7f8b94",
})

T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def fmt(n):
    """An integer with thin-space thousands separators, as printed in the figure."""
    return f"{int(n):,d}"


# ------------------------------------------------------------------------------------------------ helpers
def load_scan(spec):
    """(contour, cort_seg, ipl_pore, el_size_mm) of one OS_LH measurement, the way the validation harness reads
    them: <base>_CORT_MASK_CT.AIM is IPL's own rendering on IPL's own grid and is used as it is."""
    cr = read_aim(os.path.join(spec["folder"], f"{spec['base']}_CORT_MASK_CT.AIM"))
    if "D3P_Concatenate" in (cr.get("proclog") or ""):
        raise RuntimeError(f"{spec['id']}: CORT_MASK_CT is an evaluation-written raster, not a rendering")
    contour = ops.mask_vol(np.asarray(cr["data"]) != 0, cr["dim"], cr["pos"])
    cort_seg = read_aim(os.path.join(spec["folder"], f"{spec['base']}_CORT_SEG.AIM"))
    ipl_pore = read_aim(os.path.join(spec["folder"], f"{spec['base']}_PORE.AIM"))
    return contour, cort_seg, ipl_pore, float(cr["el_size_mm"][0])


def cascade_tail(F0, pores_M, cort_render, cort_seg, lo=0.0, up=5.0, min_pore=20):
    """Everything Script 32 does after /hysteresis_threshold, so a candidate reading can be scored on the
    map IPL actually writes.  Identical to the tail of ipldt.porosity.pore_cascade."""
    F0 = ops.gobj_maskaimpeel_ow(F0, cort_render, 0)
    F = ops.subtract_aims(F0, ops.set_value(pores_M, 127, 0))
    DF = ops.set_value(ops.add_aims(F, cort_seg), 0, 127)
    DF = ops.cl_slicewise_extractow(DF, lo, up, value_in_range=porosity.SET)
    return ops.cl_nr_extract(ops.add_aims(F, DF), int(min_pore), 0, value_in_range=porosity.SET)


def grow_structure(axes):
    s = np.zeros((3, 3, 3), bool)
    s[1, 1, 1] = True
    gx, gy, gz = axes
    if gz:
        s[0, 1, 1] = s[2, 1, 1] = True
    if gy:
        s[1, 0, 1] = s[1, 2, 1] = True
    if gx:
        s[1, 1, 0] = s[1, 1, 2] = True
    return s


#: the candidate readings of /hysteresis_threshold (label, short label drawn in I, seed(v), weak(v), grow
#: axes, output rule).  The first is the reading ipldt implements; the other fifteen are the alternatives
#: refuted in ipldt/porosity.py, each scored here by running the rest of the cascade unchanged and comparing
#: the map it produces with IPL's own PORE.AIM.
READINGS = [
    ("mode 0, both bounds inclusive: seed v \u2264 1, weak v \u2264 3 (ipldt)",
     "seed v \u2264 1, weak v \u2264 3",
     lambda v: v <= LOW, lambda v: v <= HIGH, (0, 0, 1), "seeded"),
    ("seed v \u2265 3, weak v \u2265 1 (mode ignored)", "seed v \u2265 3, weak v \u2265 1  (mode off)",
     lambda v: v >= HIGH, lambda v: v >= LOW, (0, 0, 1), "seeded"),
    ("seed v > 3, weak v > 1 (both bounds strict)", "seed v > 3, weak v > 1",
     lambda v: v > HIGH, lambda v: v > LOW, (0, 0, 1), "seeded"),
    ("weak = the band 1 \u2264 v \u2264 3, seed v \u2265 3", "seed v \u2265 3, weak 1 \u2264 v \u2264 3",
     lambda v: v >= HIGH, lambda v: (v >= LOW) & (v <= HIGH), (0, 0, 1), "seeded"),
    ("weak = the band 1 \u2264 v \u2264 3, seed v \u2264 1", "seed v \u2264 1, weak 1 \u2264 v \u2264 3",
     lambda v: v <= LOW, lambda v: (v >= LOW) & (v <= HIGH), (0, 0, 1), "seeded"),
    ("weak v \u2264 3, seed v \u2265 3", "seed v \u2265 3, weak v \u2264 3",
     lambda v: v >= HIGH, lambda v: v <= HIGH, (0, 0, 1), "seeded"),
    ("weak v \u2265 1, seed v \u2264 1", "seed v \u2264 1, weak v \u2265 1",
     lambda v: v <= LOW, lambda v: v >= LOW, (0, 0, 1), "seeded"),
    ("weak v \u2264 1 (one threshold only)", "weak v \u2264 1  (one threshold)",
     lambda v: v <= LOW, lambda v: v <= LOW, (0, 0, 1), "seeded"),
    ("weak = every voxel (no upper bound)", "weak = every voxel",
     lambda v: v <= LOW, lambda v: np.ones_like(v, bool), (0, 0, 1), "seeded"),
    ("both bounds strict: seed v < 1, weak v < 3", "seed v < 1, weak v < 3",
     lambda v: v < LOW, lambda v: v < HIGH, (0, 0, 1), "seeded"),
    ("seed v < 1 (strict), weak v \u2264 3", "seed v < 1, weak v \u2264 3",
     lambda v: v < LOW, lambda v: v <= HIGH, (0, 0, 1), "seeded"),
    ("seed v \u2264 1, weak v < 3 (strict)", "seed v \u2264 1, weak v < 3",
     lambda v: v <= LOW, lambda v: v < HIGH, (0, 0, 1), "seeded"),
    ("output = the grown set minus the seeds", "grown set minus the seeds",
     lambda v: v <= LOW, lambda v: v <= HIGH, (0, 0, 1), "minus_seeds"),
    ("no seeding at all (output = the whole weak set)", "no seeding: the whole weak set",
     lambda v: v <= LOW, lambda v: v <= HIGH, (0, 0, 1), "all_weak"),
    ("grow_axes ignored, full 6-connectivity", "grow_axes ignored: 6-connectivity",
     lambda v: v <= LOW, lambda v: v <= HIGH, (1, 1, 1), "seeded"),
    ("grow_axes read as the axes NOT to grow along (x, y)", "grow_axes = the axes NOT grown",
     lambda v: v <= LOW, lambda v: v <= HIGH, (1, 1, 0), "seeded"),
]
IPL_READING = READINGS[0][0]


def reading_output(cde, seed_fn, weak_fn, axes, rule):
    """One candidate reading of /hysteresis_threshold applied to the label volume cortseg_CDE."""
    d = np.asarray(cde["data"])
    weak = weak_fn(d)
    if rule == "all_weak":
        return ops.mask_vol(weak, cde["dim"], cde["pos"], porosity.SET)
    seed = seed_fn(d)
    lab, n = ndi.label(weak, structure=grow_structure(axes))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[seed])] = True
    keep[0] = False
    out = keep[lab]
    if rule == "minus_seeds":
        out &= ~seed
    return ops.mask_vol(out, cde["dim"], cde["pos"], porosity.SET)


# ------------------------------------------------------------------------------------------------ compute
def compute_scan(spec, want_arrays):
    """The cascade on one measurement: the stage counts, the candidate readings, and (want_arrays) the
    crops the figure draws."""
    say(f"{spec['id']}: reading")
    contour, cort_seg, ipl_pore, el = load_scan(spec)
    st = porosity.pore_cascade(contour, cort_seg, keep_stages=True)
    dim, pos = tuple(contour["dim"]), tuple(contour["pos"])
    G = lambda v: ops.on_grid(v, dim, pos)                                          # noqa: E731
    cmp_ = porosity.compare_pore(st["pores_H"], ipl_pore)
    po = porosity.ct_po(st["pores_H"], contour)
    cde_vol = st["cortseg_CDE"]
    cde = np.asarray(cde_vol["data"])
    counts = {int(v): int((cde == v).sum()) for v in LABEL_VALUES}
    assert int(cde.size) == sum(counts.values()), "cortseg_CDE holds a value outside 0/1/2/3/125/127"
    n = dict(id=spec["id"], base=spec["base"], site=spec["site"], voxel_mm=el,
             grid=dict(dim=list(dim), pos=list(pos)), voxels=int(cde.size),
             contour_voxels=int((np.asarray(contour["data"]) != 0).sum()),
             cort_seg_voxels=int((np.asarray(cort_seg["data"]) != 0).sum()),
             cort_seg_outside_compartment=int(((G(cort_seg) != 0) & (G(contour) == 0)).sum()),
             label_counts=counts, compare=cmp_, ct_po=po,
             stage_voxels={k: int((np.asarray(st[k]["data"]) != 0).sum()) for k in
                           ("pores_E", "pores_M", "pores_M0", "cortring_C", "cortring_C3",
                            "pores_F0", "pores_F", "pores_DF", "pores_G", "pores_H")})
    say(f"{spec['id']}: pore {fmt(cmp_['ours'])} voxels, {cmp_['mismatch']} differing, Dice {cmp_['dice']:.6f}; "
        f"labels {counts}")

    # ---- the candidate readings, scored on the map IPL writes
    rows = []
    for lab, short, seed_fn, weak_fn, axes, rule in READINGS:
        t = time.time()
        H = cascade_tail(reading_output(cde_vol, seed_fn, weak_fn, axes, rule), st["pores_M"], contour, cort_seg)
        c = porosity.compare_pore(H, ipl_pore)
        rows.append(dict(label=lab, short=short, mismatch=c["mismatch"], ours=c["ours"],
                         ours_only=c["ours_only"], ipl_only=c["ipl_only"]))
        say(f"   {lab[:54]:<54s} {c['mismatch']:>10d} differing  [{time.time() - t:.1f}s]")
    n["readings"] = rows
    if not want_arrays:
        return n, {}

    # ---- the arrays the figure draws
    comp = G(contour) != 0
    seg = G(cort_seg) != 0
    pore_ipl = G(ipl_pore) != 0
    stg = {k: G(st[k]) != 0 for k in ("pores_E", "pores_M", "pores_F0", "pores_F", "pores_DF", "pores_G", "pores_H")}
    cde_g = G(cde_vol)
    z = Z_SLICE
    ys, xs = np.nonzero(comp[z])
    y0, y1 = max(int(ys.min()) - SLICE_MARGIN, 0), min(int(ys.max()) + SLICE_MARGIN + 1, comp.shape[1])
    x0, x1 = max(int(xs.min()) - SLICE_MARGIN, 0), min(int(xs.max()) + SLICE_MARGIN + 1, comp.shape[2])
    S = (z, slice(y0, y1), slice(x0, x1))
    n["slice"] = dict(index=z, crop=[y0, y1, x0, x1])

    # B: the slice-wise 0-5 % rule, measured on the CORT_SEG grid where the command actually runs
    inv = ops.set_value(cort_seg, 0, 127)
    sl = np.asarray(inv["data"])[z] != 0
    labs, nlab = ndi.label(sl, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
    cnt = np.bincount(labs.ravel(), minlength=nlab + 1).astype(float)
    frac = 100.0 * cnt / max(sl.sum(), 1)
    n["slicewise"] = dict(slice_set_voxels=int(sl.sum()), components=int(nlab),
                          kept=int(((frac >= 0.0) & (frac <= 5.0))[1:].sum()),
                          dropped=int((frac > 5.0)[1:].sum()),
                          largest_share=float(frac[1:].max()) if nlab else 0.0,
                          largest_voxels=int(cnt[1:].max()) if nlab else 0,
                          second_share=float(np.sort(frac[1:])[-2]) if nlab > 1 else 0.0,
                          kept_voxels_in_slice=int(stg["pores_E"][z].sum()))
    inv_g = G(inv) != 0

    # C: the rank-1 component before the gobj mask
    rank1 = ops.cl_ow_rank_extract(st["pores_M0"], 1, 1, connect_boundary=False, value_in_range=2)
    rank1_g = G(rank1) != 0
    lab0, n0, c0 = ops.label6(np.asarray(st["pores_M0"]["data"]) != 0)
    n["rank"] = dict(pores_M0_voxels=int(n["stage_voxels"]["pores_M0"]), components=int(n0),
                     rank1_voxels=int(rank1_g.sum()), kept_in_compartment=int(n["stage_voxels"]["pores_M"]),
                     second_largest=int(np.sort(c0[1:])[-2]) if n0 > 1 else 0,
                     rank1_holds_grid_corner=bool(rank1_g[0, 0, 0]),
                     rank1_outside_compartment=int((rank1_g & (G(contour) == 0)).sum()))

    # G: the 20-voxel component filter
    lab_g, ng, cg = ops.label6(stg["pores_G"])
    n["nr_extract"] = dict(components=int(ng), below_min=int((cg[1:] < 20).sum()),
                           voxels_dropped=int(cg[1:][cg[1:] < 20].sum()), kept=int(n["stage_voxels"]["pores_H"]))

    # F: what the hysteresis contributes, by label value
    n["pores_F_by_value"] = {int(v): int(((cde_g == v) & stg["pores_F"]).sum()) for v in LABEL_VALUES}
    n["value2_in_pore"] = int(((cde_g == 2) & stg["pores_H"]).sum())
    n["value3_in_pore"] = int(((cde_g == 3) & stg["pores_H"]).sum())

    W = (z, slice(WIN_Y0, WIN_Y0 + WIN), slice(WIN_X0, WIN_X0 + WIN))
    n["window"] = dict(slice_index=z, origin_yx=[WIN_Y0, WIN_X0], size=WIN,
                       counts=dict(compartment=int(comp[W].sum()), cort_seg=int(seg[W].sum()),
                                   pores_E=int(stg["pores_E"][W].sum()), pores_M=int(stg["pores_M"][W].sum()),
                                   pores_F0=int(stg["pores_F0"][W].sum()), pores_F=int(stg["pores_F"][W].sum()),
                                   pores_DF=int(stg["pores_DF"][W].sum()), pores_G=int(stg["pores_G"][W].sum()),
                                   pores_H=int(stg["pores_H"][W].sum()), ipl_pore=int(pore_ipl[W].sum()),
                                   dropped=int((stg["pores_G"] & ~stg["pores_H"])[W].sum())))

    # E: two real z-columns of the window, one accepted by the hysteresis and one declined.  The strips are
    # outlined with what /hysteresis_threshold itself writes, BEFORE the /gobj_maskaimpeel_ow that follows it.
    f0_raw = porosity.hysteresis_threshold(cde_vol, **porosity.SCRIPT32_HYSTERESIS)
    f0_raw_g = G(f0_raw) != 0
    n["pores_F0_before_mask"] = int(f0_raw_g.sum())
    col_ok, col_no = pick_columns(cde_g, stg["pores_F0"])
    n["columns"] = dict(accepted=col_ok, declined=col_no)

    A = dict(comp=comp[S], seg=seg[S], inv=inv_g[S], pores_E=stg["pores_E"][S], rank1=rank1_g[S],
             pores_M=stg["pores_M"][S],
             w_cde=cde_g[W], w_comp=comp[W], w_seg=seg[W], w_pores_F0=stg["pores_F0"][W],
             w_pores_M=stg["pores_M"][W], w_pores_F=stg["pores_F"][W], w_pores_DF=stg["pores_DF"][W],
             w_pores_G=stg["pores_G"][W], w_pores_H=stg["pores_H"][W], w_ipl=pore_ipl[W],
             col_ok=cde_g[:, col_ok["y"], col_ok["x"]], col_no=cde_g[:, col_no["y"], col_no["x"]],
             col_ok_sel=f0_raw_g[:, col_ok["y"], col_ok["x"]],
             col_no_sel=f0_raw_g[:, col_no["y"], col_no["x"]],
             col_ok_in=(G(contour) != 0)[:, col_ok["y"], col_ok["x"]],
             col_no_in=(G(contour) != 0)[:, col_no["y"], col_no["x"]])
    return n, {k: np.ascontiguousarray(v) for k, v in A.items()}


def pick_columns(cde, sel, max_run=13):
    """Two (x, y) columns of the window, each carrying a value-2 voxel in the shown slice: one whose weak run
    (the maximal run of v <= 3 down that column, the component /hysteresis_threshold labels with
    -grow_axes 0 0 1) contains a seed voxel and is therefore written, and one whose run does not.  The run
    must be short enough to be drawn whole; among those the longest is taken, deterministically."""
    ys, xs = np.nonzero(cde[Z_SLICE, WIN_Y0:WIN_Y0 + WIN, WIN_X0:WIN_X0 + WIN] == 2)
    best = {True: None, False: None}
    for yy, xx in zip(ys.tolist(), xs.tolist()):
        y, x = yy + WIN_Y0, xx + WIN_X0
        col = cde[:, y, x]
        weak = col <= HIGH
        z0 = Z_SLICE
        while z0 > 0 and weak[z0 - 1]:
            z0 -= 1
        z1 = Z_SLICE
        while z1 + 1 < col.size and weak[z1 + 1]:
            z1 += 1
        run = z1 - z0 + 1
        if run > max_run:
            continue
        has_seed = bool((col[z0:z1 + 1] <= LOW).any())
        assert has_seed == bool(sel[Z_SLICE, y, x]), "the run's seed and IPL's selection disagree"
        key = (run, -y, -x)
        if best[has_seed] is None or key > best[has_seed][0]:
            best[has_seed] = (key, dict(x=int(x), y=int(y), z=int((z0 + z1) // 2), z_run=[int(z0), int(z1)],
                                        run=int(run), seeds=int((col[z0:z1 + 1] <= LOW).sum()),
                                        value2=int((col[z0:z1 + 1] == 2).sum()),
                                        value3=int((col[z0:z1 + 1] == 3).sum()), selected=bool(has_seed)))
    for k in (True, False):
        if best[k] is None:
            raise RuntimeError(f"no column of the window with a weak run of at most {max_run} voxels and "
                               f"seed={k} through slice {Z_SLICE}")
    return best[True][1], best[False][1]


def site_of(r):
    """patella / ultradistal radius / ... / diaphyseal tibia, from the record's greyscale-log meta (the rule of
    validation/porosity_AB_stats.cohort, with the diaphyseal bone kept)."""
    m = r.get("meta") or {}
    bone = (m.get("bone") or "").strip().lower()
    region = (m.get("region") or "").strip().upper()
    if r.get("dataset") == "patella" or bone == "patella":
        return "patella"
    return f"{'ultradistal' if region == 'UD' else 'diaphyseal'} {bone}"


def compute_ctpo():
    """Ct.Po over the cohort: configuration A of the 137 porosity records (IPL's CORT_SEG and rendered cortical
    contour in, IPL's PORE.AIM the reference) against the Ct.Po IPL's single-measurement result sheet prints.
    ipldt's Ct.Po is taken at full precision from the record and printed with three decimals to compare it with
    the sheet's string, as make_facts_bmd_ctpo.py does.  The refuted reading Ct.Po.V / (Ct.Po.V + Ct.BV) takes
    Ct.Po.V = |IPL's PORE.AIM| and Ct.BV = |IPL's CORT_SEG| (the record's B.seg.CORT_SEG.voxels_ipl)."""
    sys.path.insert(0, os.path.join(REPO, "validation"))
    from validate_dataset import exclusion_reason
    import glob
    recs, excluded = [], []
    for d in REC_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "records", "*.json"))):
            r = json.load(open(f, encoding="utf-8"))
            if exclusion_reason(r):
                excluded.append(r.get("id"))
                continue
            assert r.get("porosity", {}).get("available"), f"{r.get('id')}: no IPL PORE.AIM"
            recs.append(r)
    ids = [r["id"] for r in recs]
    assert len(ids) == len(set(ids)), "a scan in two result directories"
    sites = {s: sum(1 for r in recs if site_of(r) == s) for s in EXPECT_SITES}
    assert len(recs) == EXPECT_N and sites == EXPECT_SITES, (len(recs), sites)
    pri = {r["id"]: r for r in csv.DictReader(open(PRINTED_CSV, encoding="utf-8"))}
    assert set(ids) <= set(pri), sorted(set(ids) - set(pri))
    census = {}
    for i in ids:
        census[pri[i]["sheet_ok"]] = census.get(pri[i]["sheet_ok"], 0) + 1
    pts = []
    for r in recs:
        p = pri[r["id"]]
        if p["sheet_ok"] != "yes" or not p["Ct_Po"].strip():
            continue
        po = r["porosity"]
        ours = float(po["A"]["ct_po"])
        bv = po["ipl_pore_voxels"] / (po["ipl_pore_voxels"] + r["B"]["seg"]["CORT_SEG"]["voxels_ipl"])
        s = p["Ct_Po"].strip()
        pts.append(dict(id=r["id"], dataset=r["dataset"], site=site_of(r), ours=ours, printed=float(s),
                        printed_str=s, bv=float(bv), match=f"{ours:.3f}" == s, match_bv=f"{bv:.3f}" == s))
    A = [r["porosity"]["A"] for r in recs]
    n = dict(table=os.path.relpath(PRINTED_CSV, REPO).replace("\\", "/"),
             records=[os.path.relpath(d, REPO).replace("\\", "/") for d in REC_DIRS],
             scans=len(recs), excluded=excluded, sites=sites, census=dict(sorted(census.items())),
             exact=sum(1 for e in A if int(e["mismatch"]) == 0),
             mismatch_total=sum(int(e["mismatch"]) for e in A),
             ipl_pore_voxels=sum(int(r["porosity"]["ipl_pore_voxels"]) for r in recs),
             same_grid=sum(1 for e in A if e["same_grid"]),
             patellae=sum(1 for r in recs if r["dataset"] == "patella"),
             oslh=sum(1 for r in recs if r["dataset"] == "oslh"),
             sheets=len(pts), sheets_patellae=sum(1 for q in pts if q["site"] == "patella"),
             matches=sum(1 for q in pts if q["match"]), matches_bv=sum(1 for q in pts if q["match_bv"]),
             points=pts)
    say(f"Ct.Po: {n['exact']}/{n['scans']} scans with 0 differing voxels, {fmt(n['ipl_pore_voxels'])} IPL pore "
        f"voxels; sheets {n['census']}; {n['matches']}/{n['sheets']} printed values reproduced "
        f"({n['matches_bv']} by the Ct.BV reading)")
    return n


def compute_patella_labels():
    """The value census of cortseg_CDE on the 21 patellae of the validation set: the evidence for the legend's
    reason to show a diaphysis (value 2, the only label the hysteresis can grow into, is empty on every patella,
    so pores_F is empty).  It was computed on the configuration-A inputs of the validation harness (IPL's rendered
    cortical contour and IPL's CORT_SEG of each patella, every cascade asserted to reproduce IPL's PORE.AIM
    exactly) by an internal script that is not distributed; its result is kept in S9_porosity_numbers.json under
    'patella_labels' and in cache/S9_porosity_patella.json."""
    raise SystemExit("the patella label census is not recomputed by the public copy of this script: "
                     "S9_porosity_numbers.json 'patella_labels' holds it (see compute_patella_labels)")


def save_cache(nums, arrays):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE + ".npz", **arrays)
    json.dump(nums, open(CACHE + ".json", "w"), indent=1)


def load_cache():
    z = np.load(CACHE + ".npz")
    return json.load(open(CACHE + ".json")), {k: z[k] for k in z.files}


# ------------------------------------------------------------------------------------------------ drawing
FIG_W = 180.0
FIG_H = 240.0
NOTE_DY = 2.9
DESC_FS = 7.0                        # the panel description, as in S1-S8
NOTE_FS = 6.6                        # the note lines under a panel
_MEAS = [None]


def measure_mm(s, fontsize, weight="normal"):
    """The drawn width of a string in millimetres, from the renderer (not an estimate)."""
    if _MEAS[0] is None:
        _MEAS[0] = plt.figure(figsize=(2, 2))
    fig = _MEAS[0]
    t = fig.text(0, 0, s, fontsize=fontsize, fontweight=weight)
    w = t.get_window_extent(renderer=fig.canvas.get_renderer()).width / fig.dpi * 25.4
    t.remove()
    return w


def wrap_mm(s, width_mm, fontsize, weight="normal"):
    """Greedy word wrap of s to width_mm at the given font, measured with the renderer; '\\n' is a hard break."""
    out = []
    for para in s.split("\n"):
        line = ""
        for word in para.split(" "):
            cand = (line + " " + word).strip()
            if line and measure_mm(cand, fontsize, weight) > width_mm:
                out.append(line)
                line = word
            else:
                line = cand
        out.append(line)
    return out


def desc_h(nlines, fs=DESC_FS):
    """Height of a panel head whose description wraps to nlines lines (mm)."""
    return 3.9 + nlines * fs / 72.0 * 25.4 * 1.15 + 0.6


def wrap_notes(items, width_mm, fs=NOTE_FS):
    """[(text, bold, colour)] -> the same with every text wrapped to width_mm, one entry per drawn line."""
    out = []
    for s, bold, col in items:
        for ln in wrap_mm(s, width_mm, fs, "bold" if bold else "normal"):
            out.append((ln, bold, col))
    return out


def ax_mm(fig, x, y_top, w, h):
    """Axes at (x, y_top) mm from the top-left corner of the figure, w x h mm."""
    return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H])


def ftext(fig, x, y_top, s, **kw):
    kw.setdefault("va", "top")
    return fig.text(x / FIG_W, 1 - y_top / FIG_H, s, **kw)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def head(fig, x, yt, w, letter, title, desc_lines, checks, fs=DESC_FS):
    """The panel head: bold letter upper-left, bold title beside it, the wrapped description under them."""
    ftext(fig, x, yt, letter, fontsize=9, fontweight="bold", ha="left")
    t1 = ftext(fig, x + 3.4, yt + 0.25, title, fontsize=7.5, fontweight="bold", ha="left")
    t2 = ftext(fig, x, yt + 3.9, "\n".join(desc_lines), fontsize=fs, color=C_INK2, ha="left", linespacing=1.15)
    checks += [(t1, x + 3.4, x + w, title), (t2, x, x + w, " ".join(desc_lines))]


def notes(fig, x, yt, w, lines, checks, fs=NOTE_FS):
    for i, (s, bold, col) in enumerate(lines):
        t = ftext(fig, x, yt + i * NOTE_DY, s, fontsize=fs, fontweight="bold" if bold else "normal", color=col, ha="left")
        checks.append((t, x, x + w, s))


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(C_AXIS)
        s.set_linewidth(0.4)


def tick_style(ax):
    ax.tick_params(labelsize=6.5, length=1.8, pad=1.2, width=0.4)
    for s in ax.spines.values():
        s.set_edgecolor(C_AXIS)
        s.set_linewidth(0.4)


def legend_axes(fig, x, yt, w, h, handles, **kw):
    axl = ax_mm(fig, x, yt, w, h)
    axl.axis("off")
    kw.setdefault("frameon", False)
    kw.setdefault("borderaxespad", 0)
    kw.setdefault("handletextpad", 0.5)
    axl.legend(handles=handles, **kw)
    return axl


def scale_bar(ax, mm, vox_mm, label, frac_x=0.05, frac_y=0.94):
    """A horizontal scale bar of `mm` millimetres on an axes drawn in voxel units."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    W, H = abs(x1 - x0), abs(y0 - y1)
    nb = mm / vox_mm
    xb = min(x0, x1) + W * frac_x
    yb = max(y0, y1) - H * (1 - frac_y) if y0 > y1 else min(y0, y1) + H * frac_y
    ax.plot([xb, xb + nb], [yb, yb], color=C_INK, lw=1.3, solid_capstyle="butt", zorder=6)
    ax.text(xb + nb / 2, yb - H * 0.025, label, ha="center", va="bottom", fontsize=6.5, color=C_INK, zorder=6)


def show(ax, rgb, vox_mm=None, bar=None):
    H, W = rgb.shape[:2]
    ax.imshow(rgb, interpolation="nearest", origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    frame(ax)
    if bar:
        scale_bar(ax, bar[0], vox_mm, bar[1])
    return ax


def check_text_widths(fig, items):
    """Two checks: every registered text inside the span it was given, and EVERY text of the figure --
    axes texts, tick labels and legends included -- inside the page."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for t, x0, x1, label in items:
        bb = t.get_window_extent(renderer=r)
        lo, hi = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        if lo < x0 - 0.3 or hi > x1 + 0.3:
            bad += 1
            say(f"  WARNING text overflows its {x1 - x0:.1f} mm span by {max(x0 - lo, hi - x1):.1f} mm: {label!r}")
    every = list(fig.texts)
    for ax in fig.axes:
        every += list(ax.texts)
        if ax.get_legend() is not None:
            every += ax.get_legend().get_texts()
        if not ax.axison:                                   # ax.axis('off'): its tick labels are not drawn
            continue
        every += [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
    off = 0
    for t in every:
        if not t.get_text() or not t.get_visible():
            continue
        bb = t.get_window_extent(renderer=r)
        lo, hi = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        if lo < 0.2 or hi > FIG_W - 0.2:
            off += 1
            say(f"  WARNING text runs off the page ({lo:.1f} .. {hi:.1f} mm): {t.get_text()[:60]!r}")
    say(f"text-width check: {len(items)} registered texts, {bad} overflow; {len(every)} texts on the page, "
        f"{off} off it")
    return bad + off


def draw(N, A, CT):
    global FIG_H
    M = N["main"]
    S2nd = N["second"]
    el = M["voxel_mm"]
    sv = M["stage_voxels"]
    lc = {int(k): v for k, v in M["label_counts"].items()}
    wc = M["window"]["counts"]
    sw = M["slicewise"]
    rk = M["rank"]
    ne = M["nr_extract"]
    cm = M["compare"]
    fbv = {int(k): v for k, v in M["pores_F_by_value"].items()}
    cols = M["columns"]
    nvox = M["grid"]["dim"]

    # ------------------------------------------------------------------ the columns, in mm from the left
    x_left = 2.0
    gap1, gap3 = 3.6, 4.4
    w1 = (FIG_W - 2 * x_left - 3 * gap1) / 4                    # rows 1: A-D
    w3 = (FIG_W - 2 * x_left - 2 * gap3) / 3                    # row 3: F-H
    wE = FIG_W - 2 * x_left                                     # row 2: E
    xI, wI = x_left, 110.0                                      # row 4: I, J
    xJ = xI + wI + 5.0
    wJ = FIG_W - x_left - xJ
    xs1 = [x_left + k * (w1 + gap1) for k in range(4)]
    xs3 = [x_left + k * (w3 + gap3) for k in range(3)]

    # ------------------------------------------------------------------ every text, wrapped to its column
    D_ = {
        "A": ("Inputs", f"the compartment and CORT_SEG (slice {M['slice']['index'] + 1} of {nvox[2]})"),
        "B": ("First pore estimate", "/cl_slicewise_extractow 0–5%, per slice"),
        "C": ("Rank-1 non-bone component", "/cl_rank_extract rank 1 of the non-bone"),
        "D": ("Labeled intermediate", "cortseg_CDE on the window of A"),
        "E": ("The value encoding, and what /hysteresis_threshold does with it",
              "cortseg_CDE = cortring_C3 + (pores_E + cortseg_D2) can hold only these six values. With -low_thresh "
              "1, -high_thresh 3, -unit 5 (native) and -mode 0 the seed set is v ≤ 1 and the weak set v ≤ 3, "
              "both bounds inclusive; -grow_axes 0 0 1 labels the weak set along ±z only, so a component is a "
              "maximal run of weak voxels down one (x, y) column, and every run that holds a seed voxel is written "
              "at -value_in_range 127."),
        "F": ("The hysteresis result", "pores_F0, the rank-1 part (pores_M) removed, and pores_F"),
        "G": ("The final estimate", "the second 0–5% pass, then /cl_nr_extract 20"),
        "H": ("Against IPL's PORE.AIM", "IPL's export, ipldt's result as an outline"),
        "I": ("The reading of /hysteresis_threshold",
              "voxels differing from IPL's PORE.AIM when only the reading of this one command is varied; each "
              f"pair of bars is the scan of A–H (upper, dark) and {S2nd['id']}, "
              f"{'an' if S2nd['site'][0] in 'aeiou' else 'a'} {S2nd['site']} (lower, "
              "light), which is the scan that pins the inclusive low bound"),
        "J": ("Ct.Po over the cohort",
              f"ipldt's |PORE ∩ CORT_MASK| / |CORT_MASK| against the value IPL's single-measurement evaluation "
              f"sheet prints ({CT['sheets']} scans)"),
    }
    WID = dict(A=w1, B=w1, C=w1, D=w1, E=wE, F=w3, G=w3, H=w3, I=wI, J=wJ)
    DESC = {k: wrap_mm(v[1], WID[k], DESC_FS) for k, v in D_.items()}
    TITLE = {k: v[0] for k, v in D_.items()}

    win_counts = {int(v): int((A["w_cde"] == v).sum()) for v in LABEL_VALUES}
    NT = {
        "A": wrap_notes([(f"compartment {fmt(M['contour_voxels'])} voxels,", False, C_INK2),
                         (f"CORT_SEG {fmt(M['cort_seg_voxels'])}, none of it outside", False, C_INK2),
                         ("blue: the window of D and F–H", False, C_BLUE)], w1),
        "B": wrap_notes([(f"this slice: {fmt(sw['components'])} components of", False, C_INK2),
                         (f"{fmt(sw['slice_set_voxels'])} set voxels; {sw['dropped']} above 5%",
                          False, C_INK2),
                         (f"({sw['largest_share']:.1f}, {sw['second_share']:.1f}%) dropped, "
                          f"{fmt(sw['kept'])} kept", False, C_INK2)], w1),
        "C": wrap_notes([(f"{fmt(rk['components'])} components; rank 1 is the", False, C_INK2),
                         (f"background ({fmt(rk['rank1_voxels'])} voxels, next {fmt(rk['second_largest'])}),",
                          False, C_INK2),
                         (f"{fmt(rk['kept_in_compartment'])} of it in the compartment", False, C_INK2)], w1),
        "D": wrap_notes([(f"this window: {fmt(win_counts[2])} of value 2,", False, C_INK2),
                         (f"{fmt(win_counts[3])} of value 3, {fmt(win_counts[127])} of 127;", False, C_INK2),
                         ("colors and whole-volume counts in E", False, C_INK2)], w1),
        "F": wrap_notes([(f"pores_F0 {fmt(sv['pores_F0'])}, of which pores_M {fmt(sv['pores_M'])}",
                          False, C_INK2),
                         (f"pores_F {fmt(sv['pores_F'])} = {fmt(fbv[2])} value 2 + {fmt(fbv[3])} value 3",
                          True, C_INK),
                         (f"this window: {fmt(wc['pores_M'])} + {fmt(wc['pores_F'])} of "
                          f"{fmt(wc['pores_F0'])}", False, C_INK2)], w3),
        "G": wrap_notes([(f"pores_G {fmt(sv['pores_G'])} voxels in {fmt(ne['components'])} components",
                          False, C_INK2),
                         (f"{fmt(ne['below_min'])} below 20 voxels dropped ({fmt(ne['voxels_dropped'])} voxels)",
                          False, C_ORANGE),
                         (f"pores_H {fmt(ne['kept'])} — IPL's PORE.AIM", True, C_INK)], w3),
        "H": wrap_notes([(f"{fmt(cm['mismatch'])} of {fmt(cm['compared_voxels'])} voxels differ (whole volume)",
                          True, C_INK if cm["mismatch"] == 0 else C_VERM),
                         (f"{fmt(cm['ipl'])} voxels in each map, Dice {cm['dice']:.6f}", False, C_INK2),
                         (f"Ct.Po {M['ct_po']['ct_po']:.4f}; the sheet prints {CT['printed_main']}",
                          False, C_INK2)], w3),
        "J": wrap_notes([(f"{CT['matches']} of {CT['sheets']} printed values reproduced (3 decimals); "
                          f"Ct.BV reading {CT['matches_bv']} of {CT['sheets']}", True, C_INK),
                         (f"all {CT['scans']} scans: {fmt(CT['mismatch_total'])} voxels differ from IPL's "
                          f"PORE.AIM over {fmt(CT['ipl_pore_voxels'])} IPL pore voxels", False, C_INK2)], wJ),
    }

    # ------------------------------------------------------------------ the rows, in mm from the top
    hs, ws = A["comp"].shape
    img1_h = min(w1 * hs / ws, 38.0)
    img1_w = img1_h * ws / hs
    img3 = 46.0
    axes4_h = 41.0
    LEG1_H = 5.6
    h_head1 = desc_h(max(len(DESC[k]) for k in "ABCD"))
    h_head3 = desc_h(max(len(DESC[k]) for k in "FGH"))
    h_head4 = desc_h(max(len(DESC[k]) for k in "IJ"))
    n_notes1 = max(len(NT[k]) for k in "ABCD")
    n_notes3 = max(len(NT[k]) for k in "FGH")

    y1 = 0.8
    yn1 = y1 + h_head1 + img1_h + 1.0
    y_leg1 = yn1 + n_notes1 * NOTE_DY + 0.3
    row1_end = y_leg1 + LEG1_H
    y2 = row1_end + 2.0
    tbl_h = 38.5
    row2_end = y2 + desc_h(len(DESC["E"])) + tbl_h + 1.0
    y3 = row2_end + 2.0
    yn3 = y3 + h_head3 + img3 + 1.0
    y_leg3 = yn3 + n_notes3 * NOTE_DY + 0.3
    row3_end = y_leg3 + 4.6
    y4 = row3_end + 2.4
    FIG_H = y4 + h_head4 + 1.0 + axes4_h + 6.0
    say(f"layout: rows at {y1:.1f} / {y2:.1f} / {y3:.1f} / {y4:.1f} mm, figure {FIG_W:.0f} x {FIG_H:.1f} mm")

    fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4))
    checks = []
    P = {}

    def panel(key, x, yt, w, yimg=None):
        head(fig, x, yt, w, key, TITLE[key], DESC[key], checks)
        return None if yimg is None else yimg

    # ============================================================ row 1: A inputs, B pores_E, C pores_M, D CDE
    y_img1 = y1 + h_head1

    # ---- A: the two inputs
    panel("A", xs1[0], y1, w1)
    ax = ax_mm(fig, xs1[0] + (w1 - img1_w) / 2, y_img1, img1_w, img1_h)
    rgb = np.ones(A["comp"].shape + (3,))
    rgb[A["comp"]] = hex_rgb(C_FILL)
    rgb[A["seg"]] = hex_rgb(C_BONE)
    show(ax, rgb, el, (5.0, "5 mm"))
    y0c, _, x0c, _ = M["slice"]["crop"]
    ax.add_patch(patches.Rectangle((WIN_X0 - x0c - 0.5, WIN_Y0 - y0c - 0.5), WIN, WIN, fill=False, ec=C_BLUE, lw=0.8))
    notes(fig, xs1[0], yn1, w1, NT["A"], checks)
    legend_axes(fig, xs1[0], y_leg1, w1, LEG1_H,
                [patches.Patch(fc=C_FILL, ec="none", label="cortical compartment"),
                 patches.Patch(fc=C_BONE, ec="none", label="CORT_SEG")],
                loc="upper left", ncol=1, fontsize=6.0, handlelength=1.1, labelspacing=0.3)
    P["A"] = dict(slice=M["slice"], compartment=M["contour_voxels"], cort_seg=M["cort_seg_voxels"],
                  cort_seg_outside=M["cort_seg_outside_compartment"], window=[WIN_Y0, WIN_X0, WIN])

    # ---- B: the slice-wise 0-5 % rule
    panel("B", xs1[1], y1, w1)
    ax = ax_mm(fig, xs1[1] + (w1 - img1_w) / 2, y_img1, img1_w, img1_h)
    rgb = np.ones(A["comp"].shape + (3,))
    rgb[A["seg"]] = hex_rgb(C_FILL)
    rgb[A["inv"]] = hex_rgb(C_VERM_L)
    rgb[A["pores_E"]] = hex_rgb(C_GREEN)
    show(ax, rgb, el, (5.0, "5 mm"))
    notes(fig, xs1[1], yn1, w1, NT["B"], checks)
    legend_axes(fig, xs1[1], y_leg1, w1, LEG1_H,
                [patches.Patch(fc=C_VERM_L, ec="none", label="above 5%: dropped"),
                 patches.Patch(fc=C_GREEN, ec="none", label="kept at value 1: pores_E")],
                loc="upper left", ncol=1, fontsize=6.0, handlelength=1.1, labelspacing=0.3)
    P["B"] = dict(sw, pores_E_voxels=sv["pores_E"])

    # ---- C: the marrow component
    panel("C", xs1[2], y1, w1)
    ax = ax_mm(fig, xs1[2] + (w1 - img1_w) / 2, y_img1, img1_w, img1_h)
    rgb = np.ones(A["comp"].shape + (3,))
    rgb[A["rank1"]] = hex_rgb(C_SKY_L)
    rgb[A["comp"] & ~A["rank1"]] = hex_rgb(C_PALE)
    rgb[A["seg"]] = hex_rgb(C_FILL)
    rgb[A["pores_M"]] = hex_rgb(C_VERM)
    show(ax, rgb, el, (5.0, "5 mm"))
    notes(fig, xs1[2], yn1, w1, NT["C"], checks)
    legend_axes(fig, xs1[2], y_leg1, w1, LEG1_H,
                [patches.Patch(fc=C_SKY_L, ec="none", label="rank-1 component"),
                 patches.Patch(fc=C_VERM, ec="none", label="pores_M: its part inside")],
                loc="upper left", ncol=1, fontsize=6.0, handlelength=1.1, labelspacing=0.3)
    P["C"] = rk

    # ---- D: the label image
    panel("D", xs1[3], y1, w1)
    ax = ax_mm(fig, xs1[3] + (w1 - img1_h) / 2, y_img1, img1_h, img1_h)
    cde = A["w_cde"]
    rgb = np.ones(cde.shape + (3,))
    for v in LABEL_VALUES:
        rgb[cde == v] = hex_rgb(VALUE_COLOUR[v])
    show(ax, rgb, el, (1.0, "1 mm"))
    for key, mk, col in (("accepted", "o", C_BLUE), ("declined", "s", C_VERM)):
        c = cols[key]
        ax.plot(c["x"] - WIN_X0, c["y"] - WIN_Y0, marker=mk, ms=4.2, mfc="none", mec=col, mew=1.0, zorder=5)
    notes(fig, xs1[3], yn1, w1, NT["D"], checks)
    legend_axes(fig, xs1[3], y_leg1, w1, LEG1_H,
                [Line2D([], [], marker="o", ms=3.6, mfc="none", mec=C_BLUE, mew=0.9, ls="none",
                        label="column written at 127 (E)"),
                 Line2D([], [], marker="s", ms=3.6, mfc="none", mec=C_VERM, mew=0.9, ls="none",
                        label="column left at 0 (E)")],
                loc="upper left", ncol=1, fontsize=6.0, handlelength=1.1, labelspacing=0.3)
    P["D"] = dict(window_counts=win_counts, volume_counts=lc, columns=cols)

    # ============================================================ row 2: E the value encoding and the rule
    panel("E", x_left, y2, wE)
    y_tbl = y2 + desc_h(len(DESC["E"]))
    xt, wt = x_left, 112.0
    ax = ax_mm(fig, xt, y_tbl, wt, tbl_h)
    ax.set_xlim(0, wt)                                          # the table is laid out in millimetres
    ax.set_ylim(tbl_h, 0)
    ax.axis("off")
    rows_tbl = [
        (0, "0", "marrow, or anything outside the compartment", "cortring_C3 = 0", lc[0], "seed"),
        (1, "1", "a slice-wise small void in that region", "0 + pores_E", lc[1], "seed"),
        (2, "2", "void in the compartment, not marrow-connected, large in its slice",
         "cortring_C3 = 2", lc[2], "weak"),
        (3, "3", "the same, but small in its slice", "2 + pores_E", lc[3], "weak"),
        (125, "125", "bone outside the compartment", "cortseg_D2 = 125", lc[125], "neither"),
        (127, "127", "cortical bone inside the compartment", "125 + cortring_C3", lc[127], "neither"),
    ]
    TFS = 6.2
    CX = dict(swatch=0.4, value=4.4, mean=9.5, arise=63.0, count=95.0, role=97.5)
    W_MEAN, W_ARISE = 52.0, 20.0
    for k, h in (("swatch", "value"), ("mean", "what it means"), ("arise", "how it arises"),
                 ("count", "voxels"), ("role", "hysteresis")):
        ax.text(CX[k], 1.6, h, fontsize=6.5, fontweight="bold", va="center",
                ha="right" if k == "count" else "left", color=C_INK)
    ax.plot([0.2, wt - 0.2], [3.0, 3.0], color=C_AXIS, lw=0.5)
    ROLE = {"seed": (C_BLUE, "seed  (v ≤ 1)"), "weak": (C_GREEN, "weak  (v ≤ 3)"), "neither": (C_MUTED, "neither")}
    pitch, y_first = 5.3, 6.0
    for i, (v, lab, mean, arise, cnt, role) in enumerate(rows_tbl):
        yy = y_first + i * pitch
        ax.add_patch(patches.Rectangle((CX["swatch"], yy - 1.1), 2.6, 2.2, fc=VALUE_COLOUR[v], ec=C_AXIS, lw=0.4))
        ax.text(CX["value"], yy, lab, fontsize=6.6, va="center", ha="left", fontweight="bold", color=C_INK)
        ax.text(CX["mean"], yy, "\n".join(wrap_mm(mean, W_MEAN, TFS)), fontsize=TFS, va="center", ha="left",
                color=C_INK2, linespacing=1.15)
        ax.text(CX["arise"], yy, "\n".join(wrap_mm(arise, W_ARISE, TFS)), fontsize=TFS, va="center", ha="left",
                color=C_INK2, linespacing=1.15)
        ax.text(CX["count"], yy, fmt(cnt), fontsize=TFS, va="center", ha="right", color=C_INK)
        ax.text(CX["role"], yy, ROLE[role][1], fontsize=TFS, va="center", ha="left", color=ROLE[role][0])
    foot = (f"whole volume, {fmt(M['voxels'])} voxels of the {nvox[0]} × {nvox[1]} × {nvox[2]} grid. Value 125 "
            f"does not occur on this scan: none of its CORT_SEG voxels lies outside the compartment (A).")
    for i, ln in enumerate(wrap_mm(foot, wt - 1.0, TFS)):
        ax.text(CX["swatch"], y_first + 6 * pitch - 0.7 + i * 2.5, ln, fontsize=TFS, va="center",
                ha="left", color=C_INK2)

    # the two real columns of the scan, drawn as z-strips
    xc = xt + wt + 4.0
    wc_ = FIG_W - x_left - xc
    axc = ax_mm(fig, xc, y_tbl, wc_, tbl_h)
    nz_show = 17
    axc.set_xlim(-0.5, nz_show - 0.5)
    axc.set_ylim(nz_show * tbl_h / wc_ - 1.8, -1.8)
    axc.axis("off")
    SFS = 6.0
    cell_mm = wc_ / nz_show                                  # one voxel of a strip, in mm
    strip = lambda s: "\n".join(wrap_mm(s, wc_ - 3.0, SFS))  # noqa: E731
    axc.annotate("", xy=(nz_show - 0.7, -1.20), xytext=(-0.3, -1.20),
                 arrowprops=dict(arrowstyle="-|>", lw=0.6, color=C_INK2, mutation_scale=5))
    axc.text(-0.5, -1.70, f"one (x, y) column along z, {nz_show} of the {nvox[2]} slices",
             fontsize=SFS, ha="left", va="center", color=C_INK2)
    for k, (key, colr, title) in enumerate((("accepted", C_BLUE, "the run holds a seed: the whole run is written"),
                                            ("declined", C_VERM, "no seed in the run: nothing is written"))):
        c = cols[key]
        col = A["col_ok" if key == "accepted" else "col_no"]
        sel = A["col_ok_sel" if key == "accepted" else "col_no_sel"]
        inn = A["col_ok_in" if key == "accepted" else "col_no_in"]
        z0 = int(np.clip(c["z"] - nz_show // 2, 0, col.size - nz_show))
        yb = 0.9 + k * 3.9
        axc.text(-0.5, yb - 1.15, strip(f"column ({c['x']}, {c['y']}): {c['run']} weak voxels "
                 f"(slices {c['z_run'][0] + 1}–{c['z_run'][1] + 1}), {c['seeds']} seeds"),
                 fontsize=SFS, ha="left", va="center", color=C_INK2)
        for i in range(nz_show):
            v = int(col[z0 + i])
            axc.add_patch(patches.Rectangle((i - 0.5, yb - 0.5), 1, 1, fc=VALUE_COLOUR[v], ec="#c9c8c4", lw=0.35))
            # dark digits, 6.2 pt (5.5 pt for the three-digit 127, which must fit its 3.5 mm cell): >= 4.8 pt
            # at the 158.9 mm embedded width
            axc.text(i, yb, str(v), fontsize=6.0 if v >= 100 else 6.2, ha="center", va="center", color=C_INK)
            # only the run the caption names is outlined, so the outlines count exactly its written voxels
            # (a neighbouring run of the same column, with seeds of its own, is written too but not outlined)
            if sel[z0 + i] and c["z_run"][0] <= z0 + i <= c["z_run"][1]:
                axc.add_patch(patches.Rectangle((i - 0.5, yb - 0.5), 1, 1, fill=False, ec=colr,
                                                ls="-" if inn[z0 + i] else (0, (1.2, 0.9)), lw=0.9, zorder=4))
        axc.text(-0.5, yb + 1.15, title, fontsize=SFS, ha="left", va="center", color=colr)
    # wrapped 6 mm narrower than the other strip texts: at wc_ - 3 its first line ran past the page edge
    # (the "/gobj_maskaimpeel_ow" token was clipped at 180 mm in an earlier render as well)
    axc.text(-0.5, 6.7, "\n".join(wrap_mm("solid outline: written at 127 and kept by the /gobj_maskaimpeel_ow "
                                          "that follows; dashed: written, then cleared for lying outside the "
                                          "compartment", wc_ - 9.0, SFS)),
             fontsize=SFS, ha="left", va="top", color=C_INK2, linespacing=1.3)
    P["E"] = dict(rows=[dict(value=v, meaning=m.replace("\n", " "), arises=a, voxels=cnt, role=r)
                        for v, _l, m, a, cnt, r in rows_tbl],
                  hysteresis=dict(porosity.SCRIPT32_HYSTERESIS,
                                  grow_axes=list(porosity.SCRIPT32_HYSTERESIS["grow_axes"])),
                  columns=cols)

    # ============================================================ row 3: F the hysteresis, G the filter, H IPL
    y_img3 = y3 + h_head3
    base_w = np.ones((WIN, WIN, 3))
    base_w[A["w_comp"]] = hex_rgb(C_PALE)
    base_w[A["w_seg"]] = hex_rgb(C_FILL)

    # ---- F
    panel("F", xs3[0], y3, w3)
    ax = ax_mm(fig, xs3[0] + (w3 - img3) / 2, y_img3, img3, img3)
    rgb = base_w.copy()
    rgb[A["w_pores_M"]] = hex_rgb(C_VERM)
    rgb[A["w_pores_F"]] = hex_rgb(C_BLUE)
    show(ax, rgb, el, (1.0, "1 mm"))
    notes(fig, xs3[0], yn3, w3, NT["F"], checks)
    legend_axes(fig, xs3[0], y_leg3, w3, 4.6,
                [patches.Patch(fc=C_VERM, ec="none", label="pores_M (removed)"),
                 patches.Patch(fc=C_BLUE, ec="none", label="pores_F"),
                 patches.Patch(fc=C_FILL, ec="none", label="CORT_SEG"),
                 patches.Patch(fc=C_PALE, ec=C_AXIS, lw=0.4, label="rest of the compartment")],
                loc="upper left", ncol=2, fontsize=6.0, handlelength=1.1, columnspacing=0.8, labelspacing=0.28)
    P["F"] = dict(pores_F0=sv["pores_F0"], pores_M=sv["pores_M"], pores_F=sv["pores_F"],
                  pores_F_by_value=fbv, window=wc)

    # ---- G
    panel("G", xs3[1], y3, w3)
    ax = ax_mm(fig, xs3[1] + (w3 - img3) / 2, y_img3, img3, img3)
    rgb = base_w.copy()
    rgb[A["w_pores_H"]] = hex_rgb(C_GREEN)
    rgb[A["w_pores_G"] & ~A["w_pores_H"]] = hex_rgb(C_ORANGE)
    rgb[A["w_pores_F"]] = hex_rgb(C_BLUE)
    show(ax, rgb, el, (1.0, "1 mm"))
    notes(fig, xs3[1], yn3, w3, NT["G"], checks)
    legend_axes(fig, xs3[1], y_leg3, w3, 4.6,
                [patches.Patch(fc=C_GREEN, ec="none", label="pores_H"),
                 patches.Patch(fc=C_ORANGE, ec="none", label="dropped (< 20 voxels)"),
                 patches.Patch(fc=C_BLUE, ec="none", label="pores_F, from the hysteresis")],
                loc="upper left", ncol=2, fontsize=6.0, handlelength=1.1, columnspacing=0.8, labelspacing=0.28)
    P["G"] = dict(ne, pores_DF=sv["pores_DF"], pores_G=sv["pores_G"])

    # ---- H
    panel("H", xs3[2], y3, w3)
    ax = ax_mm(fig, xs3[2] + (w3 - img3) / 2, y_img3, img3, img3)
    rgb = base_w.copy()
    rgb[A["w_ipl"]] = hex_rgb(C_BONE)
    rgb[A["w_ipl"] != A["w_pores_H"]] = hex_rgb(C_VERM)
    show(ax, rgb, el, (1.0, "1 mm"))
    ax.contour(A["w_pores_H"].astype(float), levels=[0.5], colors=[C_BLUE], linewidths=0.45)
    notes(fig, xs3[2], yn3, w3, NT["H"], checks)
    legend_axes(fig, xs3[2], y_leg3, w3, 4.6,
                [patches.Patch(fc=C_BONE, ec="none", label="IPL's PORE.AIM"),
                 Line2D([], [], color=C_BLUE, lw=0.9, label="ipldt's pores_H (outline)"),
                 patches.Patch(fc=C_VERM, ec="none", label="differing voxels (none)")],
                loc="upper left", ncol=2, fontsize=6.0, handlelength=1.1, columnspacing=0.8, labelspacing=0.28)
    P["H"] = dict(cm, ct_po=M["ct_po"], printed=CT["printed_main"])

    # ============================================================ row 4: I the readings, J Ct.Po
    m_rows = {r["label"]: r for r in M["readings"]}
    s_rows = {r["label"]: r for r in S2nd["readings"]}
    panel("I", xI, y4, wI)
    ax_x, ax_w, ax_hI = xI + 42.5, 44.0, axes4_h - 8.0
    ax_yI = y4 + h_head4 + 2.5                                   # 2 mm lower (same bottom): room for the column header
    ax = ax_mm(fig, ax_x, ax_yI, ax_w, ax_hI)
    labs = [r[0] for r in READINGS]
    short = {r[0]: r[1] for r in READINGS}
    ypos = np.arange(len(labs))[::-1]
    floor = 0.55
    vm = [m_rows[l]["mismatch"] for l in labs]
    vs = [s_rows[l]["mismatch"] for l in labs]
    ax.barh(ypos + 0.20, [max(v, floor) for v in vm], height=0.36, lw=0,
            color=[C_BLUE if l == IPL_READING else C_MUTED for l in labs])
    ax.barh(ypos - 0.20, [max(v, floor) for v in vs], height=0.36, lw=0,
            color=[C_SKY if l == IPL_READING else "#cdccc7" for l in labs])
    ax.set_xscale("log")
    ax.set_xlim(floor, max(vm + vs) * 3)
    ax.set_ylim(-0.7, len(labs) - 0.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([("IPL: " if l == IPL_READING else "") + short[l] for l in labs], fontsize=6.0)
    ax.set_xticks([1, 100, 10000, 1000000])
    ax.set_xticklabels(["1", "100", "10,000", "1,000,000"])
    ax.minorticks_off()
    tick_style(ax)
    ax.tick_params(axis="y", length=0, pad=2.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color=C_GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("voxels differing from IPL's PORE.AIM (log scale)", fontsize=6.5, labelpad=1.5)
    # the two counts as their own right-aligned column, so nothing collides with a bar
    pitch_I = ax_hI / len(labs)
    ftext(fig, ax_x + ax_w + 1.5, ax_yI - 3.2, "2422 / 2051", fontsize=6.0, fontweight="bold", ha="left",
          color=C_INK2)
    for i, (a, b) in enumerate(zip(vm, vs)):
        t = ftext(fig, xI + wI, ax_yI + (i + 0.5) * pitch_I, "0 / 0 (identical)" if a == b == 0 else
                  f"{fmt(a)} / {fmt(b)}", fontsize=6.0, ha="right", va="center",
                  fontweight="bold" if a == b == 0 else "normal", color=C_INK if a == b == 0 else C_INK2)
        checks.append((t, ax_x + ax_w + 1.0, xI + wI, f"{a} / {b}"))
    P["I"] = dict(scans=[M["id"], S2nd["id"]],
                  rows=[dict(label=l, short=short[l], main=m_rows[l]["mismatch"],
                             second=s_rows[l]["mismatch"]) for l in labs])

    # ---- J: Ct.Po over the cohort
    panel("J", xJ, y4, wJ)
    n_notes4 = len(NT["J"])
    ax_h = min(axes4_h - n_notes4 * NOTE_DY - 7.0, wJ - 10.5)       # a square box: the identity line is 45 deg
    ax = ax_mm(fig, xJ + 9.5, y4 + h_head4 + 0.8, ax_h, ax_h)
    pts = CT["points"]
    lim = max(max(p["printed"] for p in pts), max(p["ours"] for p in pts), max(p["bv"] for p in pts)) * 1.10
    ax.plot([0, lim], [0, lim], color=C_AXIS, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot([p["printed"] for p in pts], [p["bv"] for p in pts], ls="none", marker="o", ms=2.8, mfc="none",
            mec=C_MUTED, mew=0.55, zorder=2,
            label="Ct.Po.V/(Ct.Po.V+Ct.BV)")
    # the cohort colors and markers of Figures 3, 5 and 6 (Okabe-Ito); the two diaphyseal sites share one
    cohorts = [("patella", ("patella",), "o", C_BLUE, "patella"),
               ("UD radius", ("ultradistal radius",), "^", C_GREEN, "UD radius"),
               ("UD tibia", ("ultradistal tibia",), "s", C_ORANGE, "UD tibia"),
               ("diaphyseal", ("diaphyseal radius", "diaphyseal tibia"), "D", C_PURPLE, "diaph. radius / tibia")]
    n_series = 0
    for _, sites, mk, col, name in cohorts:
        q = [p for p in pts if p["site"] in sites]
        if not q:
            continue
        n_series += 1
        ax.plot([p["printed"] for p in q], [p["ours"] for p in q], ls="none", marker=mk,
                ms=3.2 if mk != "D" else 2.8, mfc=col, mec="white", mew=0.3, zorder=4, label=f"{name} ({len(q)})")
    miss = [p for p in pts if not p["match"]]
    if miss:
        n_series += 1
        ax.plot([p["printed"] for p in miss], [p["ours"] for p in miss], ls="none", marker="o", ms=6.0,
                mfc="none", mec=C_VERM, mew=0.8, zorder=5, label=f"3rd decimal differs ({len(miss)})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    tick_style(ax)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color=C_GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("Ct.Po printed by IPL's sheet", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel("ipldt's Ct.Po", fontsize=6.5, labelpad=1.5)
    h_, l_ = ax.get_legend_handles_labels()
    legend_axes(fig, xJ + 9.5 + ax_h + 1.5, y4 + h_head4 + 0.8, wJ - 11.0 - ax_h, ax_h,
                h_[1:] + h_[:1], labels=l_[1:] + l_[:1],
                loc="upper left", ncol=1, fontsize=6.0, handlelength=0.9, labelspacing=0.45,
                handletextpad=0.4)
    notes(fig, xJ, y4 + h_head4 + 0.8 + ax_h + 6.2, wJ, NT["J"], checks)
    P["J"] = dict({k: CT[k] for k in ("scans", "sheets", "matches", "matches_bv", "exact", "mismatch_total",
                                      "ipl_pore_voxels", "same_grid", "patellae", "oslh", "excluded", "table",
                                      "records", "sites", "census", "sheets_patellae")},
                  disagreeing=[dict(id=p["id"], ours=p["ours"], printed=p["printed"]) for p in miss],
                  points=pts)

    bad = check_text_widths(fig, checks)
    return fig, P, bad


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="ignore the cache and recompute everything")
    args = ap.parse_args()
    if not args.recompute and os.path.exists(CACHE + ".npz") and os.path.exists(CACHE + ".json"):
        say("loading the cache", CACHE)
        N, A = load_cache()
        N["main"]["site"], N["second"]["site"] = MAIN["site"], SECOND["site"]
    else:
        main_n, A = compute_scan(MAIN, True)
        second_n, _ = compute_scan(SECOND, False)
        N = json.loads(json.dumps(dict(main=main_n, second=second_n)))     # same key types as a cache load
        save_cache(N, A)
        say("cache written", CACHE)
    if not args.recompute and os.path.exists(CACHE_PATELLA):
        PL = json.load(open(CACHE_PATELLA))
    else:
        say("the value census of cortseg_CDE on the 21 patellae")
        PL = compute_patella_labels()
        os.makedirs(os.path.dirname(CACHE_PATELLA), exist_ok=True)
        json.dump(PL, open(CACHE_PATELLA, "w"), indent=1)
    say(f"patellae: value 3 {PL['value3_min']}..{PL['value3_max']}, value 2 at most {PL['value2_max']}, "
        f"pores_F at most {PL['pores_F_max']} ({PL['scans']} scans)")
    CT = compute_ctpo()
    CT["printed_main"] = next((p['printed_str'] for p in CT["points"] if p["id"] == N["main"]["id"]), "no value")
    fig, P, bad = draw(N, A, CT)
    os.makedirs(HERE, exist_ok=True)
    fig.savefig(OUT + ".png", dpi=300, facecolor="white")
    fig.savefig(OUT + ".svg", facecolor="white")
    plt.close(fig)
    P["figure"] = dict(width_mm=FIG_W, height_mm=round(FIG_H, 1), dpi=300, voxel_mm=N["main"]["voxel_mm"],
                       scan=N["main"]["id"], base=N["main"]["base"], site=N["main"]["site"],
                       slice_index=Z_SLICE, slice_number_1based=Z_SLICE + 1, window=[WIN_Y0, WIN_X0, WIN],
                       text_overflows=bad,
                       generated=time.strftime("%Y-%m-%d %H:%M"))
    P["computed"] = N
    P["patella_labels"] = PL
    json.dump(P, open(OUT + "_numbers.json", "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    say("wrote", OUT + ".png", OUT + ".svg", OUT + "_numbers.json")


if __name__ == "__main__":
    main()
