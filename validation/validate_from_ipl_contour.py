"""validate_from_ipl_contour -- the complete ipldt downstream composed from IPL's periosteal contour,
checked stage by stage against what Scanco IPL V5.42 wrote, on the 21-patella HR-pQCT cohort.

THE QUESTION (PI): given the same bone contour (IPL's periosteal contour), how close are ipldt's
compartment masks, segmentations and metrics to IPL's?  validate_against_ipl.py answers the last stage
only (the dt maps recomputed on IPL's own SEG and masks); this script runs every stage AFTER the contour
with ipldt and compares each intermediate with the file IPL wrote for it, aligned by global voxel
position (never by shape).  Every mask, SEG, map and metric gets a mismatch count, and the summary says
on how many subjects each one is EXACT.

Per subject (volumes are dict(data (z, y, x), dim (x, y, z), pos (x, y, z)) as ipldt.io.read_aim returns):
  input     <base>.AIM (native int16 greyscale; density calibration from its processing log, per-axis
            element sizes from its header) and IPL's raw <base>_CORT_MASK.AIM / _TRAB_MASK.AIM (Script 32
            STEP 1 outputs).  periosteal = CORT_MASK | TRAB_MASK on the TRAB_MASK grid: this IS IPL's
            stage 00, the /gobj_to_aim rendering of the bone contour, because Script 32 partitions that
            rendering into the two masks (29 = 00 - 28; 0 voxels differ on PFJ-0be66a).
  step 1    ipldt.step1.cort_trab_separation(grey, periosteal, TIBIA)  ->  CORT (28) / TRAB (29)
            vs IPL's raw CORT_MASK / TRAB_MASK: mismatches (ours-only / IPL-only), Dice, both grids.
  contours  ipldt.contour.render_volume of OUR two masks, each on its own grid (IPL's /togobj_from_aim
            + rasterisation).  When a mask equals IPL's on the same grid its rendering equals IPL's by
            construction (and IPL's renderings are exact on all 21 subjects, validate_against_ipl.py);
            otherwise IPL's raw mask is rendered too and the two contours are compared by position.
  SEG       ipldt.ormir.laplace_hamming_threshold(native, header element sizes) & periosteal raster,
            /cl_nr_extract 35 & cortical contour, 70 & trabecular contour (ipldt.ormir.ipl_seg_assembly,
            Script 32 lines 407-472); CORT_SEG / TRAB_SEG on the box of the masked threshold (IPL's
            seg_box), SEG = 127 cortical + 126 trabecular on its own tight box
            vs IPL's SEG / TRAB_SEG / CORT_SEG (support mismatches, label agreement, Dice, grids).
  dt        ipldt.ormir.ipl_morphometry(SEG, trabecular contour, CORT_MASK, cortical contour), Script 32
            parameters, voxel size VOXEL_MM of validate_against_ipl.py (0.0607 mm, what the IPL column of
            validation/results/sample_means.csv uses)  ->  BV/TV, BOTH Tb.Th definitions, Tb.Sp,
            1/Tb.N and Tb.N, Ct.Th; every map vs IPL's TRAB_TH / TRAB_SP / TRAB_1N / CORT_TH /
            TRAB_TH_old voxel for voxel (joint histogram on the union grid), every value vs the IPL value
            derived from IPL's map by the same formula (= sample_means.csv; the file is re-read and the
            two IPL columns are checked against each other).  BV/TV: BV = SEG & trabecular contour,
            TV = trabecular contour, on each side from its own SEG and contour.

WHICH Tb.Th IS REPORTED, AND THE TRAP IN IPL'S FILENAMES.  Scripts 32, 33 (byte-identical command files)
and 34 all write TRAB_SEG as IPL_FNAME5 and then run /dt_thickness on it, cropped to the trabecular gobj
(32/33 lines 452 / 708 / 710; 34 lines 190 / 446 / 448).  So the Tb.Th the scripts compute is
dt_thickness(TRAB_SEG) -- and for this cohort that is IPL's file <base>_TRAB_TH_old, NOT <base>_TRAB_TH.
The filenames are the reverse of the intuitive reading: TRAB_TH_old holds the Script-32 (TRAB_SEG) map and
TRAB_TH holds the whole-SEG one.  Reading them the other way round is exactly what misled this project, so
the tables here never say 'old' or 'new': they say 'Tb.Th (whole SEG, as ORMIR-BQRL ships)' -- the Tb.Th
reported first here, because it is compared with IPL's delivered <base>_TRAB_TH and is what the shipped
pipeline computes -- and 'Tb.Th (TRAB_SEG, Scripts 32/33/34)', the scripts' own definition, reported beside
it and compared with IPL's <base>_TRAB_TH_old.  Both values are computed and stored under the historical
record keys TbTh (whole SEG) and TbTh_old (TRAB_SEG).  Each cohort is deliberately compared against the
definition IPL's own delivered file used rather than having one designation forced on both.
(Tb.Sp and Tb.N read the whole SEG in the scripts too -- IPL_SEGAIM -- and Ct.Th the cortical compartment;
those are unaffected.)

Options: --ipl-log-values FILE adds IPL's PRINTED log values (a JSON {subject or base: {metric: value}})
as a second reference column, at the header's x element size (what IPL prints with); --ipl-logs DIR
cross-checks the /seg_gauss native thresholds and IPL's printed Tb.N against the evaluation logs.

Outputs (results/from_ipl_contour/): table.csv (one row per subject), table.md (per-stage tables),
summary.md (cohort totals, exact counts, metric differences, open issues), records.json and
records/<subject>.json (everything, incl. the step-1 stage counts / grids / thresholds for diagnosis).

RESULT (2026-09-14 19:36, 21 subjects, GPU, 1,667 s on a shared machine; results/from_ipl_contour/; computed under the
FLOOR padding rule, so the SEG / TRAB_SEG / CORT_SEG numbers of this paragraph are superseded by the ceil run in the
next paragraph and are kept here as the provenance of the shipped directory): CORT_MASK /
TRAB_MASK exact on 21/21 with identical grids (20/21 in the run of 2026-09-13, kept in
results/from_ipl_contour_pre_mirror/: PFJ-411dfd_R differed by 3 voxels of its last slice, z = 167, that the
pre-mechanism /open 15 removed -- located at /open 15 by test run 17, inside /open's internal chaining by test run 19,
explained the same evening by the erosion's edge-inclusive mirror margin that /open's dilation half reuses, and
implemented in ipl_ops.open_ that night; test run 20, its prospective confirmation, ran on 2026-09-15 and CONFIRMED the
mechanism -- IPL matched its own prediction on 46 of 46 exports carrying readings, 12 of them separating it from every
alternative, 11 contrast hypotheses refuted); cort and trab contours
exact on 21/21; SEG differs by 138 voxels in total (0 .. 40 per subject, 64 ipldt-only / 74 IPL-only,
Laplace-Hamming float32-FFT rounding at the threshold, PFJ-d81140_L exact at every stage; 140 = 64 / 76 before the
mechanism, PFJ-411dfd_R's 5 becoming 3), SEG grids identical on 21/21, labels identical, min Dice 0.9999988 (PFJ-69bcb0_L);
Ct.Th map exact on 21/21 (20/21 before: 1 voxel on PFJ-411dfd_R); metrics: max |ours - IPL| = 4.0e-7 mm Tb.Th,
1.8e-6 mm Tb.Sp, 1.1e-5 /mm Tb.N, 0 mm Ct.Th (3.2e-7 before), 1.9e-7 BV/TV (max relative 0.0005 %); at the
header's element size the dt metrics reproduce IPL's printed 6-decimal values to 1.1e-5 (Tb.N) or better.

RESULT (2026-09-17 15:31, 21 subjects, --skip-dt --lh-pad-offset ceil, 569 s; results/from_ipl_contour_ceil/): with
IPL's padding offset (test run 21: the extra power-of-two padding voxel goes BEFORE the data on an odd-padded axis; the
run above used 'floor', one voxel off on every such axis) SEG differs by 50 voxels (0 .. 5 per subject, 23 ipldt-only
/ 27 IPL-only; PFJ-8bcf88_L and PFJ-d81140_R exact; 9 subjects better, 3 worse by 1-2 voxels (PFJ-42293d_L, PFJ-d81140_L, PFJ-6f5538_R),
9 unchanged, the 6 with no odd-padded axis identical voxel for voxel), TRAB_SEG 33, CORT_SEG 17; step-1 masks,
contours and SEG labels 0 on 21/21; every remaining voxel has our short at exactly 15564 or 15563, the FFT rounding
floor (2026-09-17). In numbers, floor -> ceil: patella SEG 138 -> 50 (64 / 74 -> 23 / 27), TRAB_SEG
67 -> 33, CORT_SEG 71 -> 17 (results/from_ipl_contour/ -> results/from_ipl_contour_ceil/); radius / tibia configuration B
(validate_dataset.py; internal runs, not distributed) on 52 radius / tibia measurements: 620 -> 39 (max 4,
Diaphyseal/CKD/315196), on 53 others: 648 -> 39; pooled 758 -> 89 over 73 measurements and 786 -> 89 over 74; four
measurements worse, none by more than +2 (Diaphyseal/REPRO/229741 0 -> 1, PFJ-d81140_L 0 -> 1, PFJ-42293d_L 1 -> 3, PFJ-6f5538_R
1 -> 2); all 89 remaining voxels are noise-floor class (our short exactly 15564 on the 43 ipldt-only and 15563 on the
46 IPL-only, scaled float within 0.0098 of the threshold, isolated single voxels, distributed like the bone). No dt map or metric has been computed under ceil yet.

SHIPPED-DIRECTORY GUARD (--allow-overwrite-shipped). The published result directories under validation/results/ (the
records the paper's numbers are built from: from_ipl_contour_ceil_dt/, from_ipl_contour_ceil/, porosity_AB_patella/ and
the two radius / tibia sets of result_sets.py) are never overwritten by accident: a run whose --results is one of them,
or lies under one, is REFUSED before anything is written (exit status 2) unless --allow-overwrite-shipped is given.
The default --results is results/from_ipl_contour_run/.

CLI
    python validate_from_ipl_contour.py                          # all 21 subjects
    python validate_from_ipl_contour.py --skip-dt --lh-pad-offset ceil --results validation/results/my_ceil_run
    python validate_from_ipl_contour.py --subjects PFJ-0be66a_R  # one subject (-> results/from_ipl_contour_run/subset_PFJ-0be66a_R/)
    python validate_from_ipl_contour.py --skip-dt --backend cpu  # masks, contours and SEGs only
    python validate_from_ipl_contour.py --resume                 # keep the per-subject records already written
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import ipldt  # noqa: E402
from ipldt import ipl_ops as ops  # noqa: E402
from ipldt import ormir  # noqa: E402
from ipldt.contour import render_volume  # noqa: E402
from ipldt.step1 import RADIUS, TIBIA, cort_trab_separation  # noqa: E402
import validate_against_ipl as vai  # noqa: E402  (Subject, VOXEL_MM, joint_histogram, summarize_joint, log)

RESULTS_DIR = os.path.join(vai.RESULTS_DIR, "from_ipl_contour_run")          # a new run never lands in a published folder
# the published result directories (the records the paper's numbers are built from): no run may write into them or
# below them without --allow-overwrite-shipped; validate_dataset.py uses the same guard
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
SHIPPED_RESULTS_DIRS = tuple(os.path.join(vai.RESULTS_DIR, d) for d in ("from_ipl_contour_ceil_dt", "from_ipl_contour_ceil",
                                                                     "porosity_AB_patella", "patella_bvtv_A", OSLH_AUTO, OSLH_NOEDIT))
SITE_PARAMS = {"tibia": TIBIA, "radius": RADIUS}


def _is_under(path, root):
    """True when `path` is `root` or lies inside it (case-normalised, absolute; symlinks are not resolved)."""
    p, r = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def guard_shipped_results(results, lh_pad_offset, allow_overwrite, shipped_dirs=SHIPPED_RESULTS_DIRS, log_fn=print,
                          lh_dtype=ormir.LH_DTYPE):
    """Refuse (exit status 2, nothing written) a run whose results directory is one of the published result
    directories or lies inside one, unless allow_overwrite; with it, log a warning naming the run's padding offset
    and dtype and go ahead.  Returns the published directory hit, if any."""
    hit = next((d for d in shipped_dirs if _is_under(results, d)), None)
    if hit is None and os.path.normcase(os.path.abspath(results)) == os.path.normcase(os.path.abspath(vai.RESULTS_DIR)):
        hit = vai.RESULTS_DIR          # validation/results/ itself: the published records.json and sample_means.csv
    if hit is None:
        return None
    if not allow_overwrite:
        print(f"REFUSED: --results {results} is (or lies under) the published results directory {hit}; this run\n"
              f"(--lh-pad-offset '{lh_pad_offset}' --lh-dtype '{lh_dtype}'; the published runs used 'ceil' / 'float32')\n"
              "would overwrite the records the paper's numbers are built from. Pass --results <another folder>, or\n"
              "--allow-overwrite-shipped if replacing them is intended. Nothing was written.", file=sys.stderr)
        raise SystemExit(2)
    log_fn(f"WARNING: --allow-overwrite-shipped given: writing '{lh_pad_offset}' / '{lh_dtype}' numbers into the "
           f"published directory {hit}")
    return hit
MASKS = ("cort", "trab")
SEGS = ("SEG", "TRAB_SEG", "CORT_SEG")
MAPS = ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH", "TRAB_TH_old")
# The Tb.Th reported first here is the whole-SEG one, compared with IPL's delivered file <base>_TRAB_TH -- what the
# shipped ORMIR-BQRL pipeline computes.  The definition Scripts 32 / 33 / 34 compute (/dt_thickness on TRAB_SEG, which
# for this cohort is IPL's file <base>_TRAB_TH_old -- the filenames are the REVERSE of the intuitive reading, see the
# module docstring) is reported beside it.  The dict KEYS 'TbTh' / 'TbTh_old' are historical identifiers kept so that the
# stored records, sample_means.csv and the downstream scripts keep working; only the labels say what each one is.
METRICS = (("BVTV", "BV/TV", "-", None),
           ("TbTh", "Tb.Th (whole SEG, as ORMIR-BQRL ships)", "mm", "TRAB_TH"),
           ("TbSp", "Tb.Sp", "mm", "TRAB_SP"), ("TbN", "Tb.N", "1/mm", "TRAB_1N"), ("CtTh", "Ct.Th", "mm", "CORT_TH"),
           ("TbTh_old", "Tb.Th (TRAB_SEG, Scripts 32/33/34)", "mm", "TRAB_TH_old"))
LOG_KEYS = {"bvtv": "BVTV", "bv_tv": "BVTV", "bv/tv": "BVTV", "tbth": "TbTh", "tbth_mm": "TbTh", "tb_th_mm": "TbTh", "tb.th": "TbTh",
            "tbsp": "TbSp", "tbsp_mm": "TbSp", "tb_sp_mm": "TbSp", "tb.sp": "TbSp", "tbn": "TbN", "tbn_per_mm": "TbN", "tb_n_per_mm": "TbN",
            "tb.n": "TbN", "ctth": "CtTh", "ctth_mm": "CtTh", "ct_th_mm": "CtTh", "ct.th": "CtTh", "tbth_old": "TbTh_old", "tbth_old_mm": "TbTh_old",
            "tb_th_old_mm": "TbTh_old"}                     # names IPL's printed values may carry -> METRICS keys
TBTH_LEGEND = (
    "Tb.Th is reported under BOTH definitions.  **Tb.Th (whole SEG, as ORMIR-BQRL ships)** is dt_thickness of the whole SEG "
    "cropped to the trabecular gobj, what the shipped pipeline computes, compared with IPL's delivered file "
    "`<base>_TRAB_TH`; it is the one reported first here.  **Tb.Th (TRAB_SEG, Scripts 32/33/34)** is what Scanco's "
    "evaluation scripts compute -- /dt_thickness on TRAB_SEG (IPL_FNAME5) cropped to the same gobj -- and it is compared "
    "with IPL's file `<base>_TRAB_TH_old`.  NOTE the trap: for this cohort IPL's filenames are the reverse of the intuitive "
    "reading -- TRAB_TH_old is the Script-32 (TRAB_SEG) map and TRAB_TH is the whole-SEG one.  The record / CSV keys stay "
    "`TbTh_old` (TRAB_SEG) and `TbTh` (whole SEG) for continuity."
)
log = vai.log


# --------------------------------------------------------------------------------------- volumes and comparisons
def grid_of(v):
    return tuple(int(x) for x in v["dim"]), tuple(int(x) for x in v["pos"])


def grid_str(g):
    (dx, dy, dz), (px, py, pz) = g
    return f"{dx}x{dy}x{dz} @ {px},{py},{pz}"


def compare_masks(ours, ipl, locate=200):
    """Two 0 / non-zero volumes on any grids, pasted onto their union grid by global position:
    mismatches = |ours xor IPL|, split into ours-only / IPL-only, Dice, both grids; up to `locate`
    mismatching voxels as global (x, y, z, side) with side +1 = ours only, -1 = IPL only."""
    dim, pos = ops.union_grid(ours, ipl)
    a = ops.on_grid(ours, dim, pos) != 0
    b = ops.on_grid(ipl, dim, pos) != 0
    na, nb, both = int(a.sum()), int(b.sum()), int((a & b).sum())
    out = dict(voxels_ours=na, voxels_ipl=nb, mismatches=na + nb - 2 * both, ours_only=na - both, ipl_only=nb - both,
               dice=(2.0 * both / (na + nb)) if na + nb else 1.0, grid_ours=grid_of(ours), grid_ipl=grid_of(ipl),
               grids_equal=grid_of(ours) == grid_of(ipl), union_voxels=int(a.size))
    if 0 < out["mismatches"] <= locate:
        zyx = np.argwhere(a != b)
        out["where"] = [[int(x + pos[0]), int(y + pos[1]), int(z + pos[2]), 1 if a[z, y, x] else -1] for z, y, x in zyx]
    elif out["mismatches"]:
        out["slices"] = [int(z + pos[2]) for z in np.unique(np.nonzero(a != b)[0])[:60]]
    return out


def compare_labels(ours, ipl):
    """Voxels set on both sides whose values differ (SEG: 127 cortical vs 126 trabecular)."""
    dim, pos = ops.union_grid(ours, ipl)
    a = ops.on_grid(ours, dim, pos)
    b = ops.on_grid(ipl, dim, pos)
    return int(((a != 0) & (b != 0) & (a != b)).sum())


def compare_maps(ours, ipl, voxel_mm=vai.VOXEL_MM):
    """Two integer maps on any grids -> validate_against_ipl.summarize_joint of their joint histogram on the
    union grid (mismatches split ours-only / IPL-only / both-non-zero-differ, supports, means in vox and mm),
    plus the (ours, IPL) value pairs and slices of the mismatching voxels."""
    dim, pos = ops.union_grid(ours, ipl)
    a = np.asarray(ops.on_grid(ours, dim, pos)).astype(np.int16)
    b = np.asarray(ops.on_grid(ipl, dim, pos)).astype(np.int16)
    d = vai.summarize_joint(vai.joint_histogram(a, b), voxel_mm)
    d.update(grid_ours=grid_of(ours), grid_ipl=grid_of(ipl), grids_equal=grid_of(ours) == grid_of(ipl))
    if d["mismatches"]:
        m = a != b
        pairs, counts = np.unique(np.stack([a[m], b[m]], 1), axis=0, return_counts=True)
        order = np.argsort(-counts)[:12]
        d["value_pairs"] = [[int(pairs[i, 0]), int(pairs[i, 1]), int(counts[i])] for i in order]
        d["slices"] = [int(z) for z in np.unique(np.nonzero(m)[0])[:60]]
    return d


def metric_values(rec, side, voxel_mm):
    """The six metrics of one side ('ours' / 'ipl') from the map comparisons, at `voxel_mm`; BV/TV as stored."""
    out = {"BVTV": rec["bvtv"][side]}
    for key, _, _, name in METRICS[1:]:
        mean_vox = rec["maps"][name][f"mean_{side}_vox"]
        out[key] = (1.0 / (mean_vox * voxel_mm) if mean_vox else 0.0) if key == "TbN" else mean_vox * voxel_mm
    return out


# --------------------------------------------------------------------------------------- one subject
class Case:
    """One subject: the greyscale, IPL's raw masks, and (through validate_against_ipl.Subject) IPL's
    decompressed SEGs and maps plus the cached renderings of IPL's own masks."""

    def __init__(self, subject, root=vai.DATA_ROOT, cache_dir=vai.CACHE_DIR):
        self.ipl = vai.Subject(subject, root, cache_dir)
        self.subject, self.base, self.folder = subject, self.ipl.base, self.ipl.folder

    def grey(self):
        return ipldt.read_aim(os.path.join(self.folder, f"{self.base}.AIM"))

    def raw_mask(self, which):
        """IPL's raw (run-length) <base>_CORT_MASK.AIM / _TRAB_MASK.AIM."""
        return ipldt.read_aim(os.path.join(self.folder, f"{self.base}_{which.upper()}_MASK.AIM"))

    def ipl_volume(self, suffix):
        return self.ipl.aim(suffix)


def periosteal_from_ipl(cort_raw, trab_raw):
    """CORT_MASK | TRAB_MASK on the TRAB_MASK grid = IPL's stage 00 (the rendered periosteal contour).
    The two masks must be disjoint and the cortical one inside the trabecular grid (29 = 00 - 28)."""
    dim, pos = grid_of(trab_raw)
    u = (ops.on_grid(cort_raw, dim, pos) != 0) | (np.asarray(trab_raw["data"]) != 0)
    n_c, n_t, n_u = int((cort_raw["data"] != 0).sum()), int((trab_raw["data"] != 0).sum()), int(u.sum())
    if n_u != n_c + n_t:
        raise ValueError(f"CORT_MASK ({n_c:,d}) and TRAB_MASK ({n_t:,d}) overlap or the cortical mask leaves the "
                         f"trabecular grid: union {n_u:,d}")
    return ops.mask_vol(u, dim, pos)


def render_contour(mask):
    """IPL's /togobj_from_aim + rasterisation of a mask, on the mask's own grid (bool volume)."""
    return ops.vol(render_volume(np.asarray(mask["data"]) != 0), mask["dim"], mask["pos"])


def build_seg(grey, periosteal, g_cort, g_trab, pad_offset=ormir.LH_PAD_OFFSET, dtype=ormir.LH_DTYPE):
    """Script 32 STEP 2 on our contours: Laplace-Hamming threshold of the native volume (header element
    sizes), masked by the periosteal raster, cleaned and split by the cortical / trabecular contours.
    pad_offset: the /fft_laplace_hamming power-of-two padding offset (ormir.lh_pad_plan): 'ceil' = IPL's rule
    (test run 21), 'floor' = the rule the shipped results/from_ipl_contour numbers were computed with (before 2026-09-17).
    dtype: the FFT / filter arithmetic, 'float32' (the shipped engine, ormir.LH_DTYPE) or 'float64' (the opt-in under
    study since 2026-09-17, see ormir.lh_filter_core).
    Returns dict(SEG (127/126 on its tight box), TRAB_SEG, CORT_SEG (0/127 on IPL's seg_box = the box of the
    masked threshold), lh_voxels, overlap (voxels in both compartments; SEG keeps 127 there, as IPL's does), el_size_mm,
    lh_pad_offset, lh_dtype)."""
    dim, pos = grid_of(grey)
    el = tuple(float(e) for e in grey["el_size_mm"])
    bm = ormir.laplace_hamming_threshold(grey["data"], el, pad_offset=pad_offset, dtype=dtype)
    lh_voxels = int(bm.sum())
    prx = ops.on_grid(periosteal, dim, pos) != 0
    seg_box = grid_of(ormir.bbox_cut(bm & prx, pos))
    cort_seg, trab_seg = ormir.ipl_seg_assembly(bm, prx, ops.on_grid(g_cort, dim, pos) != 0, ops.on_grid(g_trab, dim, pos) != 0)
    del bm, prx
    seg = np.zeros(cort_seg.shape, np.uint8)
    seg[trab_seg] = ormir.SEG_VALUE_TRAB
    seg[cort_seg] = ormir.SEG_VALUE_CORT       # IPL's SEG carries 127 where the two compartments overlap (PFJ-0be66a: all 15 such voxels)
    box = ormir.bbox_cut(seg, pos)
    out = dict(SEG=ops.vol(np.ascontiguousarray(box["data"]), box["dim"], box["pos"]), lh_voxels=lh_voxels,
               overlap=int((cort_seg & trab_seg).sum()), seg_box=seg_box, el_size_mm=el, lh_pad_offset=pad_offset, lh_dtype=dtype)
    for name, arr in (("TRAB_SEG", trab_seg), ("CORT_SEG", cort_seg)):
        data = ops.on_grid(ops.vol(arr, dim, pos), *seg_box)
        out[name] = ops.mask_vol(data, *seg_box)
    return out


def run_dt(seg, trab_seg, g_trab, cort, g_cort, backend, voxel_mm=vai.VOXEL_MM):
    """Script 32's dt stage on OUR volumes (ipldt.ormir.ipl_morphometry): the trabecular maps on our SEG grid
    with our trabecular contour, Ct.Th on our CORT_MASK grid with our cortical contour."""
    dim, pos = grid_of(seg)
    m = ormir.ipl_morphometry(np.asarray(seg["data"]) != 0, ops.on_grid(g_trab, dim, pos) != 0,
                              cort_mask=np.asarray(cort["data"]) != 0, cort_gobj=np.asarray(g_cort["data"]) != 0,
                              voxel_size_mm=voxel_mm, backend=backend, which=MAPS,
                              trab_seg=ops.on_grid(trab_seg, dim, pos) != 0, log=lambda s: log("   " + s))
    maps = {name: ops.vol(res.map.astype(np.int16), *(grid_of(cort) if name == "CORT_TH" else (dim, pos)))
            for name, res in m["results"].items()}
    return m, maps


def ipl_bvtv(case):
    """IPL's BV/TV from IPL's own SEG and its rendered trabecular contour (the cached rendering on the SEG grid)."""
    G = case.ipl.gobj("trab")
    bv, tv = int((case.ipl.seg() & G).sum()), int(G.sum())
    return dict(ipl=bv / tv if tv else 0.0, bv_ipl=bv, tv_ipl=tv, g_trab_ipl_on_seg_grid=tv)


def process_subject(subject, params, backend="auto", skip_dt=False, root=vai.DATA_ROOT, cache_dir=vai.CACHE_DIR,
                    lh_pad_offset=ormir.LH_PAD_OFFSET, lh_dtype=ormir.LH_DTYPE):
    """The whole chain for one subject; returns the record dict (numbers only, JSON-ready).
    lh_pad_offset, lh_dtype: see build_seg ('floor' / 'float32' reproduce the shipped results/from_ipl_contour numbers)."""
    t0 = time.time()
    case = Case(subject, root, cache_dir)
    rec = dict(subject=subject, base=case.base, timings={})
    log(f"{subject} ({case.base}) ----------------------------------------------------------------")

    # (1) inputs: greyscale, IPL's raw masks, periosteal = CORT | TRAB on the TRAB_MASK grid
    t = time.time()
    grey = case.grey()
    ipl_masks = {k: case.raw_mask(k) for k in MASKS}
    periosteal = periosteal_from_ipl(ipl_masks["cort"], ipl_masks["trab"])
    rec["grey"] = dict(grid=grid_of(grey), el_size_mm=[float(e) for e in grey["el_size_mm"]])
    rec["periosteal"] = dict(voxels=int(np.count_nonzero(periosteal["data"])), grid=grid_of(periosteal))
    rec["timings"]["read"] = time.time() - t
    log(f"  grey {grid_str(rec['grey']['grid'])} el {' / '.join(f'{e:.7f}' for e in rec['grey']['el_size_mm'])} mm; "
        f"periosteal = CORT | TRAB {rec['periosteal']['voxels']:,d} voxels on {grid_str(rec['periosteal']['grid'])} [{rec['timings']['read']:.1f}s]")

    # (2) step 1: cortical / trabecular separation vs IPL's raw masks
    t = time.time()
    r = cort_trab_separation(grey, periosteal, params, log=lambda s: log("    " + s))
    ours = {"cort": r["cort"], "trab": r["trab"]}
    info = r["info"]
    rec["step1"] = dict(calibration=info["calibration"], thresholds=info["thresholds"], seg_gauss_box=info["box"],
                        counts={k: v for k, v in info["counts"].items() if k != "peel"}, peel_counts=info["counts"]["peel"],
                        grids=info["grids"], params=info["params"])
    for k in MASKS:
        rec["step1"][k] = compare_masks(ours[k], ipl_masks[k])
    rec["timings"]["step1"] = time.time() - t
    for k in MASKS:
        c = rec["step1"][k]
        log(f"  {k.upper()}_MASK ours {c['voxels_ours']:,d} on {grid_str(c['grid_ours'])} | IPL {c['voxels_ipl']:,d} on {grid_str(c['grid_ipl'])} "
            f"| mismatches {c['mismatches']:,d} (+{c['ours_only']:,d} / -{c['ipl_only']:,d}) Dice {c['dice']:.6f}")
    log(f"  step 1 done [{rec['timings']['step1']:.1f}s]")

    # (3) contours of OUR masks (IPL's rendered when a mask differs)
    t = time.time()
    G = {k: render_contour(ours[k]) for k in MASKS}
    rec["contours"] = {}
    for k in MASKS:
        c = rec["step1"][k]
        entry = dict(voxels_ours=int(np.count_nonzero(G[k]["data"])), grid=grid_of(G[k]))
        if c["mismatches"] == 0 and c["grids_equal"]:
            entry.update(voxels_ipl=entry["voxels_ours"], mismatches=0, identical_by_construction=True)
        else:
            entry.update(compare_masks(G[k], render_contour(ipl_masks[k])), identical_by_construction=False)
        rec["contours"][k] = entry
    rec["timings"]["render"] = time.time() - t
    log("  contours: " + "; ".join(f"{k} {rec['contours'][k]['voxels_ours']:,d} voxels"
                                  + ("" if rec["contours"][k]["identical_by_construction"] else f" (vs IPL's rendering: {rec['contours'][k]['mismatches']:,d} mismatches)")
                                  for k in MASKS) + f" [{rec['timings']['render']:.1f}s]")
    del ipl_masks

    # (4) Laplace-Hamming threshold + SEG assembly vs IPL's SEG / TRAB_SEG / CORT_SEG
    t = time.time()
    s = build_seg(grey, periosteal, G["cort"], G["trab"], pad_offset=lh_pad_offset, dtype=lh_dtype)
    del grey
    rec["seg"] = dict(lh_voxels=s["lh_voxels"], overlap=s["overlap"], seg_box=s["seg_box"], el_size_mm=list(s["el_size_mm"]),
                      lh_pad_offset=s["lh_pad_offset"], lh_dtype=s["lh_dtype"])
    for name in SEGS:
        ipl_v = case.ipl_volume(name)
        rec["seg"][name] = compare_masks(s[name], ipl_v)
        if name == "SEG":
            rec["seg"][name]["label_mismatches"] = compare_labels(s[name], ipl_v)
    rec["timings"]["seg"] = time.time() - t
    log(f"  LH threshold {s['lh_voxels']:,d} voxels; seg_box {grid_str(s['seg_box'])}; cort/trab overlap {s['overlap']:,d}")
    for name in SEGS:
        c = rec["seg"][name]
        extra = f", labels differ {c['label_mismatches']:,d}" if name == "SEG" else ""
        log(f"  {name:<8s} ours {c['voxels_ours']:,d} on {grid_str(c['grid_ours'])} | IPL {c['voxels_ipl']:,d} on {grid_str(c['grid_ipl'])} "
            f"| mismatches {c['mismatches']:,d} (+{c['ours_only']:,d} / -{c['ipl_only']:,d}) Dice {c['dice']:.6f}{extra}")
    log(f"  SEG stage done [{rec['timings']['seg']:.1f}s]")

    # (5) dt maps and metrics vs IPL's maps
    if not skip_dt:
        t = time.time()
        m, maps = run_dt(s["SEG"], s["TRAB_SEG"], G["trab"], ours["cort"], G["cort"], backend)
        rec["maps"] = {name: compare_maps(maps[name], case.ipl_volume(name)) for name in MAPS}
        rec["morphometry"] = m["metrics"]
        rec["bvtv"] = dict(ours=m["metrics"]["BV_TV"], bv_ours=m["metrics"]["BV_voxels"], tv_ours=m["metrics"]["TV_voxels"], **ipl_bvtv(case))
        rec["dt_backend"] = ipldt.gpu.resolve_backend(backend)
        rec["timings"]["dt"] = time.time() - t
        for name in MAPS:
            c = rec["maps"][name]
            log(f"  {name:<12s} vs IPL: {c['mismatches']:,d} mismatches of {c['n_voxels']:,d} (+{c['ours_only']:,d} / -{c['ipl_only']:,d} / both {c['both_nonzero_differ']:,d}); "
                f"mean ours {c['mean_ours_vox']:.6f} IPL {c['mean_ipl_vox']:.6f} vox; grids {'equal' if c['grids_equal'] else 'DIFFER'}")
        vo, vi = metric_values(rec, "ours", vai.VOXEL_MM), metric_values(rec, "ipl", vai.VOXEL_MM)
        log("  metrics ours | IPL: " + "  ".join(f"{lab} {vo[k]:.6f} | {vi[k]:.6f}" for k, lab, _, _ in METRICS) + f" [{rec['timings']['dt']:.1f}s]")
    case.ipl.release()
    rec["timings"]["total"] = time.time() - t0
    log(f"  {subject} total {rec['timings']['total']:.1f}s")
    return rec


# --------------------------------------------------------------------------------------- references
def read_sample_means(path=os.path.join(vai.RESULTS_DIR, "sample_means.csv")):
    """{subject: {metric key: IPL value}} of validate_against_ipl.py's cohort table (its IPL column)."""
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["subject"]] = {k: float(row[f"{k}_ipl"]) for k, _, _, _ in METRICS if f"{k}_ipl" in row}
    return out


def read_log_values(path):
    """IPL's printed values -> {subject: {METRICS key: value}}.  Accepts {subjects: {subject: {metrics_ipl: {name:
    {value, ...}}}}} (the downstream reference file built from the evaluation logs and the printed sheets) or a plain
    {subject or base: {name: value}}; names are matched case-insensitively against LOG_KEYS."""
    if not path or not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    raw = raw.get("subjects", raw) if isinstance(raw, dict) else {}
    out = {}
    for name, vals in raw.items():
        if not isinstance(vals, dict):
            continue
        entry = {}
        for k, v in vals.get("metrics_ipl", vals).items():
            key = LOG_KEYS.get(str(k).lower())
            if key is None:
                continue
            v = v.get("value") if isinstance(v, dict) else v
            try:
                entry[key] = float(v)
            except (TypeError, ValueError):
                pass
        if entry:
            out[str(name)] = entry
    return out


def read_ipl_log(logs_dir, base):
    """The /seg_gauss native thresholds and the printed Tb.N of the newest evaluation log of `base`."""
    if not logs_dir:
        return None
    hits = []
    for f in sorted(glob.glob(os.path.join(logs_dir, "EVAL_LH_EFF_*.LOG"))):
        txt = open(f, encoding="latin-1", errors="replace").read()
        if f"]{base}.AIM" in txt:
            date = re.search(r"^\s*(\d{1,2}-[A-Z]{3}-\d{4} [\d:.]+)", txt, re.M)
            hits.append((date.group(1) if date else "", f, txt))
    if not hits:
        return None
    date, f, txt = sorted(hits, key=lambda h: h[1].count("__"))[0]   # the single-underscore file = the current evaluation
    thr = re.search(r"Thresholds correspond to native numbers:\s+(\d+)\s+(\d+)", txt)
    tbn = re.search(r"MAT N \(1/Th\)\s*=\s*([-\d.]+)", txt)
    return dict(file=os.path.basename(f), date=date, thresholds=[int(thr.group(1)), int(thr.group(2))] if thr else None,
                TbN=float(tbn.group(1)) if tbn else None)


# --------------------------------------------------------------------------------------- tables
def _fmt(v, spec=None):
    if v is None:
        return "-"
    if spec:
        return spec.format(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,d}"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _md(rows, cols):
    """rows: dicts; cols: (key, header[, format spec]) -> a markdown table."""
    cols = [(c[0], c[1], c[2] if len(c) > 2 else None) for c in cols]
    lines = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(k), spec) for k, _, spec in cols) + " |")
    return "\n".join(lines)


def flat_row(rec, sample_means, log_values, voxel_mm=vai.VOXEL_MM):
    """One flat dict per subject for table.csv (and the per-stage markdown tables)."""
    row = dict(subject=rec["subject"], base=rec["base"], grey_grid=grid_str(rec["grey"]["grid"]),
               el_x=rec["grey"]["el_size_mm"][0], el_y=rec["grey"]["el_size_mm"][1], el_z=rec["grey"]["el_size_mm"][2],
               periosteal_voxels=rec["periosteal"]["voxels"], thr_lower=rec["step1"]["thresholds"]["lower_native"],
               thr_upper=rec["step1"]["thresholds"]["upper_native"])
    for k in MASKS:
        c = rec["step1"][k]
        row.update({f"{k}_ours": c["voxels_ours"], f"{k}_ipl": c["voxels_ipl"], f"{k}_mism": c["mismatches"], f"{k}_plus": c["ours_only"],
                    f"{k}_minus": c["ipl_only"], f"{k}_dice": c["dice"], f"{k}_grid_ours": grid_str(c["grid_ours"]),
                    f"{k}_grid_ipl": grid_str(c["grid_ipl"]), f"{k}_grids_equal": c["grids_equal"],
                    f"G_{k}_voxels": rec["contours"][k]["voxels_ours"], f"G_{k}_mism": rec["contours"][k]["mismatches"]})
    row.update(lh_voxels=rec["seg"]["lh_voxels"], seg_overlap=rec["seg"]["overlap"], seg_box=grid_str(rec["seg"]["seg_box"]))
    for name in SEGS:
        c = rec["seg"][name]
        row.update({f"{name}_ours": c["voxels_ours"], f"{name}_ipl": c["voxels_ipl"], f"{name}_mism": c["mismatches"], f"{name}_plus": c["ours_only"],
                    f"{name}_minus": c["ipl_only"], f"{name}_dice": c["dice"], f"{name}_grid_ours": grid_str(c["grid_ours"]),
                    f"{name}_grid_ipl": grid_str(c["grid_ipl"]), f"{name}_grids_equal": c["grids_equal"]})
    row["SEG_label_mism"] = rec["seg"]["SEG"]["label_mismatches"]
    if "maps" in rec:
        for name in MAPS:
            c = rec["maps"][name]
            row.update({f"{name}_mism": c["mismatches"], f"{name}_plus": c["ours_only"], f"{name}_minus": c["ipl_only"],
                        f"{name}_both": c["both_nonzero_differ"], f"{name}_voxels": c["n_voxels"], f"{name}_grids_equal": c["grids_equal"]})
        vo, vi = metric_values(rec, "ours", voxel_mm), metric_values(rec, "ipl", voxel_mm)
        el = rec["grey"]["el_size_mm"][0]                       # what IPL prints its statistics with
        vo_el, vi_el = metric_values(rec, "ours", el), metric_values(rec, "ipl", el)
        sm = sample_means.get(rec["subject"], {})
        lv = log_values.get(rec["subject"]) or log_values.get(rec["base"]) or {}
        for k, _, _, _ in METRICS:
            d = vo[k] - vi[k]
            row.update({f"{k}_ours": vo[k], f"{k}_ipl": vi[k], f"{k}_diff": d, f"{k}_rel_pct": 100.0 * d / vi[k] if vi[k] else 0.0,
                        f"{k}_ours_el": vo_el[k], f"{k}_ipl_el": vi_el[k]})
            if k in sm:
                row[f"{k}_sample_means_delta"] = vi[k] - sm[k]
            if k in lv:
                row.update({f"{k}_log": lv[k], f"{k}_diff_log": vo_el[k] - lv[k]})
    if rec.get("ipl_log"):
        L = rec["ipl_log"]
        row.update(log_file=L["file"], log_thr_lower=L["thresholds"][0] if L["thresholds"] else None,
                   log_thr_upper=L["thresholds"][1] if L["thresholds"] else None, log_TbN=L["TbN"])
    row.update({f"t_{k}": round(v, 1) for k, v in rec["timings"].items()})
    return row


def write_table_csv(rows, path):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def write_table_md(rows, records, path, have_dt):
    P = ["# ipldt from IPL's periosteal contour: per-subject tables", "",
         f"{len(rows)} subjects.  Every comparison is by global voxel position on the union grid of the two files; "
         "'+' = voxels set by ipldt only, '-' = by IPL only.  Grids as dim @ pos (x, y, z).", "",
         "## Step 1: cortical / trabecular masks vs IPL's raw CORT_MASK / TRAB_MASK", "",
         _md(rows, [("subject", "subject"), ("base", "base"), ("thr_lower", "seg_gauss lower"), ("thr_upper", "upper"),
                    ("cort_ours", "CORT ours"), ("cort_ipl", "CORT IPL"), ("cort_mism", "mismatches"), ("cort_plus", "+"), ("cort_minus", "-"),
                    ("cort_dice", "Dice"), ("cort_grid_ours", "CORT grid ours"), ("cort_grid_ipl", "CORT grid IPL"),
                    ("trab_ours", "TRAB ours"), ("trab_ipl", "TRAB IPL"), ("trab_mism", "mismatches"), ("trab_plus", "+"), ("trab_minus", "-"),
                    ("trab_dice", "Dice"), ("trab_grid_ours", "TRAB grid ours"), ("trab_grid_ipl", "TRAB grid IPL")]), "",
         "## Contours rendered from our masks (mismatches vs IPL's rendering; 0 by construction when the mask is exact)", "",
         _md(rows, [("subject", "subject"), ("G_cort_voxels", "cortical contour voxels"), ("G_cort_mism", "mismatches"),
                    ("G_trab_voxels", "trabecular contour voxels"), ("G_trab_mism", "mismatches")]), "",
         "## SEG / TRAB_SEG / CORT_SEG vs IPL's files", "",
         _md(rows, [("subject", "subject"), ("lh_voxels", "LH threshold"), ("seg_box", "seg_box"), ("seg_overlap", "cort & trab overlap"),
                    ("SEG_ours", "SEG ours"), ("SEG_ipl", "SEG IPL"), ("SEG_mism", "mismatches"), ("SEG_plus", "+"), ("SEG_minus", "-"),
                    ("SEG_label_mism", "labels differ"), ("SEG_dice", "Dice"), ("SEG_grid_ours", "SEG grid ours"), ("SEG_grid_ipl", "SEG grid IPL"),
                    ("TRAB_SEG_ours", "TRAB_SEG ours"), ("TRAB_SEG_ipl", "IPL"), ("TRAB_SEG_mism", "mismatches"), ("TRAB_SEG_plus", "+"), ("TRAB_SEG_minus", "-"), ("TRAB_SEG_dice", "Dice"),
                    ("CORT_SEG_ours", "CORT_SEG ours"), ("CORT_SEG_ipl", "IPL"), ("CORT_SEG_mism", "mismatches"), ("CORT_SEG_plus", "+"), ("CORT_SEG_minus", "-"), ("CORT_SEG_dice", "Dice"),
                    ("TRAB_SEG_grid_ours", "TRAB/CORT_SEG grid ours"), ("TRAB_SEG_grid_ipl", "grid IPL")]), ""]
    if have_dt:
        P += [TBTH_LEGEND, "",
              "## Maps vs IPL's TRAB_TH / TRAB_SP / TRAB_1N / CORT_TH / TRAB_TH_old (voxels compared on the union grid)", "",
              _md(rows, [("subject", "subject")] + [c for name in MAPS for c in
                                                    ((f"{name}_mism", f"{name} mismatches"), (f"{name}_plus", "+"), (f"{name}_minus", "-"), (f"{name}_both", "both>0 differ"))]), "",
              f"## Metrics: ipldt (from IPL's contour) vs IPL (from IPL's maps), voxel size {vai.VOXEL_MM} mm; diff = ours - IPL, rel = diff / IPL", "",
              _md(rows, [("subject", "subject")] + [c for k, lab, unit, _ in METRICS for c in
                                                    ((f"{k}_ours", f"{lab} ours"), (f"{k}_ipl", f"{lab} IPL"), (f"{k}_diff", "diff", "{:+.6f}"), (f"{k}_rel_pct", "rel %", "{:+.4f}"))]), ""]
        P += ["## The same metrics at the header's x element size (what IPL prints its statistics with)", "",
              _md(rows, [("subject", "subject"), ("el_x", "el_x (mm)", "{:.7f}")] + [c for k, lab, unit, _ in METRICS for c in
                                                                                 ((f"{k}_ours_el", f"{lab} ours"), (f"{k}_ipl_el", "IPL"))]), ""]
        if any(f"{k}_log" in r for r in rows for k, _, _, _ in METRICS):
            P += ["## Metrics vs IPL's printed log values (ours at the header's x element size)", "",
                  _md(rows, [("subject", "subject")] + [c for k, lab, unit, _ in METRICS for c in
                                                        ((f"{k}_ours_el", f"{lab} ours"), (f"{k}_log", "log"), (f"{k}_diff_log", "diff", "{:+.6f}"))]), ""]
        where = [(r["subject"], name, r["seg"][name]) for r in records for name in SEGS if r["seg"][name].get("where")]
        if where:
            P += ["## Where the SEG voxels differ (global x, y, z; + = ipldt only, - = IPL only)", "",
                  "| subject | file | mismatches | voxels |", "|---|---|---|---|"]
            for s, name, c in where:
                P.append(f"| {s} | {name} | {c['mismatches']} | " + ", ".join(f"({x},{y},{z}){'+' if side > 0 else '-'}" for x, y, z, side in c["where"]) + " |")
            P.append("")
    P += ["## Timing (s)", "", _md(rows, [("subject", "subject"), ("t_read", "read"), ("t_step1", "step 1"), ("t_render", "contours"), ("t_seg", "LH + SEG"),
                                         ("t_dt", "dt"), ("t_total", "total")]), ""]
    open(path, "w", encoding="utf-8").write("\n".join(P))


def cohort_summary(records, rows, have_dt):
    """Totals over the cohort: exact counts per stage, mismatch totals, metric differences."""
    n = len(records)
    stages = []
    for k in MASKS:
        cs = [r["step1"][k] for r in records]
        stages.append(dict(stage=f"{k.upper()}_MASK (step 1)", exact=sum(c["mismatches"] == 0 for c in cs), n=n,
                           mismatches=sum(c["mismatches"] for c in cs), plus=sum(c["ours_only"] for c in cs), minus=sum(c["ipl_only"] for c in cs),
                           min_dice=min(c["dice"] for c in cs), grids_equal=sum(c["grids_equal"] for c in cs)))
    for k in MASKS:
        cs = [r["contours"][k] for r in records]
        stages.append(dict(stage=f"{k} contour", exact=sum(c["mismatches"] == 0 for c in cs), n=n, mismatches=sum(c["mismatches"] for c in cs),
                           plus=sum(c.get("ours_only", 0) for c in cs), minus=sum(c.get("ipl_only", 0) for c in cs), min_dice=min(c.get("dice", 1.0) for c in cs),
                           grids_equal=sum(c.get("grids_equal", True) for c in cs)))
    for name in SEGS:
        cs = [r["seg"][name] for r in records]
        stages.append(dict(stage=name, exact=sum(c["mismatches"] == 0 for c in cs), n=n, mismatches=sum(c["mismatches"] for c in cs),
                           plus=sum(c["ours_only"] for c in cs), minus=sum(c["ipl_only"] for c in cs), min_dice=min(c["dice"] for c in cs),
                           grids_equal=sum(c["grids_equal"] for c in cs)))
    if have_dt:
        for name in MAPS:
            cs = [r["maps"][name] for r in records]
            stages.append(dict(stage=f"{name} map", exact=sum(c["mismatches"] == 0 for c in cs), n=n, mismatches=sum(c["mismatches"] for c in cs),
                               plus=sum(c["ours_only"] for c in cs), minus=sum(c["ipl_only"] for c in cs), min_dice=None,
                               grids_equal=sum(c["grids_equal"] for c in cs)))
    metrics = []
    if have_dt:
        for k, lab, unit, _ in METRICS:
            o = np.array([r[f"{k}_ours"] for r in rows])
            i = np.array([r[f"{k}_ipl"] for r in rows])
            d = o - i
            rel = 100.0 * d / np.where(i != 0, i, 1.0)
            metrics.append(dict(metric=lab, unit=unit, mean_ipl=float(i.mean()), mean_ours=float(o.mean()), mean_diff=float(d.mean()),
                                mean_abs=float(np.abs(d).mean()), max_abs=float(np.abs(d).max()), mean_rel=float(np.abs(rel).mean()),
                                max_rel=float(np.abs(rel).max()), exact=int((d == 0).sum()), n=n))
    return stages, metrics


def write_summary(records, rows, path, args, have_dt, sample_means, log_values):
    stages, metrics = cohort_summary(records, rows, have_dt)
    n = len(records)
    p = records[0]["step1"]["params"]
    L = ["# ipldt from IPL's periosteal contour vs Scanco IPL V5.42: cohort summary", "",
         f"{n} subjects; site preset '{args.site}' (Step1Params corner_min {p['corner_min']:,d}, close2 {p['close2']}, peel0 {p['peel0']}); "
         f"dt parameters {vai.IPL_PARAMS}; voxel size {vai.VOXEL_MM} mm; backend {records[0].get('dt_backend', 'n/a')}; ipldt {ipldt.__version__}; "
         f"{time.strftime('%Y-%m-%d %H:%M')}.", "",
         "Input per subject: IPL's periosteal contour (raw CORT_MASK | TRAB_MASK = IPL's /gobj_to_aim rendering) and the native greyscale; "
         "every later stage is ipldt's and is compared with the file IPL wrote, by global voxel position.", "",
         "## Exactness per stage (subjects with 0 mismatching voxels) and cohort totals", "",
         _md(stages, [("stage", "stage"), ("exact", "subjects exact"), ("n", "of"), ("mismatches", "mismatching voxels (cohort)"), ("plus", "ipldt only"),
                      ("minus", "IPL only"), ("min_dice", "min Dice"), ("grids_equal", "subjects with identical grids")]), ""]
    if have_dt:
        L += [TBTH_LEGEND, "",
              "## Metrics (ours - IPL over the cohort; IPL values derived from IPL's maps as in validation/results/sample_means.csv)", "",
              _md(metrics, [("metric", "metric"), ("unit", "unit"), ("mean_ipl", "IPL mean"), ("mean_ours", "ipldt mean"), ("mean_diff", "mean diff", "{:+.3e}"),
                            ("mean_abs", "mean |diff|", "{:.3e}"), ("max_abs", "max |diff|", "{:.3e}"), ("mean_rel", "mean |rel| %", "{:.5f}"),
                            ("max_rel", "max |rel| %", "{:.5f}"), ("exact", "subjects exact"), ("n", "of")]), ""]
        deltas = [abs(r[f"{k}_sample_means_delta"]) for r in rows for k, _, _, _ in METRICS if f"{k}_sample_means_delta" in r]
        if deltas:
            L += [f"IPL column check: max |IPL value here - sample_means.csv| = {max(deltas):.3e} over {len(deltas)} values "
                  f"(same formula, same maps; {'identical' if max(deltas) == 0 else 'floating-point order only' if max(deltas) < 1e-12 else 'DIFFERENT -- inspect'}).", ""]
        if log_values:
            dl = [(k, r["subject"], r[f"{k}_diff_log"]) for r in rows for k, _, _, _ in METRICS if f"{k}_diff_log" in r]
            if dl:
                worst = max(dl, key=lambda x: abs(x[2]))
                dt_only = [x for x in dl if x[0] != "BVTV"]
                worst_dt = max(dt_only, key=lambda x: abs(x[2])) if dt_only else None
                L += [f"IPL's printed values ({len(dl)} values, {len({s for _, s, _ in dl})} subjects; ours at the header's x element size, which is "
                      f"what IPL prints with): max |ours - printed| = {abs(worst[2]):.3e} ({worst[0]} on {worst[1]})"
                      + (f"; dt metrics only (6 printed decimals, rounding floor 5e-7): {abs(worst_dt[2]):.3e} ({worst_dt[0]} on {worst_dt[1]})" if worst_dt else "")
                      + "; BV/TV comes from the printed sheet at 3 decimals (rounding floor 5e-4).", ""]
        if any(r.get("ipl_log") for r in records):
            bad = [r["subject"] for r in records if r.get("ipl_log") and r["ipl_log"]["thresholds"] and
                   r["ipl_log"]["thresholds"] != [r["step1"]["thresholds"]["lower_native"], r["step1"]["thresholds"]["upper_native"]]]
            L += [f"/seg_gauss native thresholds vs the IPL logs: {n - len(bad)}/{n} subjects agree" + (f"; DIFFER on {bad}" if bad else "") + ".", ""]
    L += ["## Per-subject overview", "",
          _md(rows, [("subject", "subject"), ("base", "base"), ("cort_mism", "CORT_MASK mism."), ("trab_mism", "TRAB_MASK mism."), ("SEG_mism", "SEG mism."),
                     ("SEG_plus", "+"), ("SEG_minus", "-"), ("TRAB_SEG_mism", "TRAB_SEG mism."), ("CORT_SEG_mism", "CORT_SEG mism."), ("SEG_dice", "SEG Dice")]
              + ([(f"{name}_mism", f"{name} mism.") for name in MAPS] if have_dt else []) + [("t_total", "time (s)", "{:.1f}")]), ""]
    if have_dt:
        L += ["## Per-subject metrics (ours | IPL | diff)", "",
              _md(rows, [("subject", "subject")] + [c for k, lab, unit, _ in METRICS for c in
                                                    ((f"{k}_ours", f"{lab} ours"), (f"{k}_ipl", "IPL"), (f"{k}_diff", "diff", "{:+.2e}"))]), ""]
    # open issues: every subject that is not exact somewhere
    issues = []
    for r, row in zip(records, rows):
        bits = [f"{k.upper()}_MASK {r['step1'][k]['mismatches']:,d}" for k in MASKS if r["step1"][k]["mismatches"]]
        bits += [f"{k} contour {r['contours'][k]['mismatches']:,d}" for k in MASKS if r["contours"][k]["mismatches"]]
        bits += [f"{s} {r['seg'][s]['mismatches']:,d} (+{r['seg'][s]['ours_only']:,d}/-{r['seg'][s]['ipl_only']:,d})" for s in SEGS if r["seg"][s]["mismatches"]]
        if r["seg"]["SEG"]["label_mismatches"]:
            bits.append(f"SEG labels {r['seg']['SEG']['label_mismatches']:,d}")
        if have_dt:
            bits += [f"{m} map {r['maps'][m]['mismatches']:,d}" for m in MAPS if r["maps"][m]["mismatches"]]
            bits += [f"{lab} {row[f'{k}_diff']:+.2e}" for k, lab, _, _ in METRICS if row[f"{k}_diff"] != 0]
        if bits:
            issues.append(f"- {r['subject']} ({r['base']}): " + ", ".join(bits))
    L += ["## Subjects not exact at some stage", ""] + (issues or ["- none"]) + [""]
    open(path, "w", encoding="utf-8").write("\n".join(L))
    return stages, metrics


# --------------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=None, help="subject folders (default: all)")
    ap.add_argument("--results", default=RESULTS_DIR)
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--site", default="tibia", choices=sorted(SITE_PARAMS), help="Step1Params preset (tibia = Script 32)")
    ap.add_argument("--skip-dt", action="store_true", help="stop after the SEG comparison")
    ap.add_argument("--lh-pad-offset", default=ormir.LH_PAD_OFFSET, choices=list(ormir.LH_PAD_OFFSETS),
                    help="power-of-two padding offset of /fft_laplace_hamming: ceil (IPL's rule, default) or floor "
                         "(the rule used before 2026-09-17)")
    ap.add_argument("--lh-dtype", default=ormir.LH_DTYPE, choices=list(ormir.LH_DTYPES),
                    help="FFT / filter arithmetic of /fft_laplace_hamming: float32 (the shipped engine, default; every shipped "
                         "result) or float64 (opt-in under study since 2026-09-17: the same operations in double precision, "
                         "rounded to float32 before IPL's float -> short conversion)")
    ap.add_argument("--allow-overwrite-shipped", action="store_true",
                    help="write into (or under) one of the published result directories under validation/results/ (the "
                         "records the paper's numbers are built from); refused otherwise, exit status 2")
    ap.add_argument("--resume", action="store_true", help="reuse results/records/<subject>.json where present")
    ap.add_argument("--root", default=vai.DATA_ROOT)
    ap.add_argument("--cache", default=vai.CACHE_DIR, help="validate_against_ipl's cache (renderings of IPL's masks)")
    ap.add_argument("--ipl-log-values", default=None, help="JSON with IPL's printed metric values per subject (second reference column)")
    ap.add_argument("--ipl-logs", default=None, help="folder of IPL's EVAL_LH_EFF_*.LOG files (threshold / Tb.N cross-check)")
    a = ap.parse_args(argv)
    subjects = a.subjects or vai.list_subjects(a.root)
    if a.subjects and a.results == RESULTS_DIR:
        tag = "_".join(a.subjects) if len("_".join(a.subjects)) <= 60 else f"{len(a.subjects)}subjects"
        a.results = os.path.join(RESULTS_DIR, f"subset_{tag}")
        log(f"partial run: results go to {a.results} (pass --results to override)")
    guard_shipped_results(a.results, a.lh_pad_offset, a.allow_overwrite_shipped, log_fn=log,
                          lh_dtype=a.lh_dtype)                                              # before anything is written
    rec_dir = os.path.join(a.results, "records")
    os.makedirs(rec_dir, exist_ok=True)
    log(f"ipldt {ipldt.__version__}; backend {ipldt.gpu.resolve_backend(a.backend)}; site {a.site}; {len(subjects)} subjects; "
        f"dt {'skipped' if a.skip_dt else 'on'}; LH pad offset {a.lh_pad_offset}; LH dtype {a.lh_dtype}; results {a.results}")
    records = []
    for s in subjects:
        f = os.path.join(rec_dir, f"{s}.json")
        if a.resume and os.path.exists(f):
            rec = json.load(open(f))
            if a.skip_dt or "maps" in rec:
                log(f"{s}: reusing {f}")
                if a.ipl_logs and not rec.get("ipl_log"):
                    rec["ipl_log"] = read_ipl_log(a.ipl_logs, rec["base"])
                records.append(rec)
                continue
        rec = process_subject(s, SITE_PARAMS[a.site], a.backend, a.skip_dt, a.root, a.cache, lh_pad_offset=a.lh_pad_offset,
                              lh_dtype=a.lh_dtype)
        rec["ipl_log"] = read_ipl_log(a.ipl_logs, rec["base"])
        json.dump(rec, open(f, "w"), indent=1, default=ormir._json_default)
        records.append(rec)
    have_dt = all("maps" in r for r in records)
    sample_means = read_sample_means()
    log_values = read_log_values(a.ipl_log_values)
    if a.ipl_log_values:
        log(f"IPL log values: {len(log_values)} subjects read from {a.ipl_log_values}" if log_values else f"IPL log values: none usable at {a.ipl_log_values}")
    rows = [flat_row(r, sample_means, log_values) for r in records]
    json.dump(records, open(os.path.join(a.results, "records.json"), "w"), indent=1, default=ormir._json_default)
    write_table_csv(rows, os.path.join(a.results, "table.csv"))
    write_table_md(rows, records, os.path.join(a.results, "table.md"), have_dt)
    stages, metrics = write_summary(records, rows, os.path.join(a.results, "summary.md"), a, have_dt, sample_means, log_values)
    log("cohort: " + "; ".join(f"{st['stage']} exact {st['exact']}/{st['n']} (mism {st['mismatches']:,d})" for st in stages))
    for m in metrics:
        log(f"  {m['metric']:<12s} mean |diff| {m['mean_abs']:.3e} {m['unit']}  max |diff| {m['max_abs']:.3e}  max |rel| {m['max_rel']:.5f} %  exact {m['exact']}/{m['n']}")
    log(f"wrote {a.results}/table.csv, table.md, summary.md, records.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
