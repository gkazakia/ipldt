"""validate_dataset -- dataset-agnostic, stage-by-stage validation of ipldt against Scanco IPL V5.42, driven
by a LAYOUT description (one for the radius / tibia export -- 'OS_LH': ultradistal and diaphyseal tibia and radius
measurements, from which the paper's 116 are drawn -- and one for the 21-patella cohort), computing THREE
configurations per measurement:

  A  'same IPL segs and contours'   dt on IPL's SEG with IPL's contour renderings -> maps vs IPL's maps
                                    (voxel mismatches, supports, means) and map-derived metrics vs IPL's.
                                    Expected exact when the renderings used here are the ones the maps used.
  B  'same bone contour'            ipldt step 1 (Script 32 / 33 STEP 1) from IPL's periosteal rendering ->
                                    our cort / trab masks -> our renderings (Dice / mismatches vs IPL's renderings;
                                    exact expected unless the trabecular contour was manually corrected, flag CORR)
                                    -> our Laplace-Hamming SEG with OUR renderings vs IPL's SEG -> our dt maps
                                    and metrics vs IPL's.
  C  'same bone contour, IPL's cort/trab contours'   our LH SEG assembled with IPL's renderings, then dt, so the
                                    manually corrected cases still get a like-for-like SEG / map / metric comparison.
  A2 (CORR measurements only)       A's trabecular maps recomputed with OUR (uncorrected, step-1) trabecular
                                    rendering as the gobj: together with the support tests it tells which contour
                                    IPL's maps actually used (the maps' processing logs name 'trab_mask.gobj').

WHAT THE OS_LH LAYOUT ENCODES (from the six IPL logs and the AIM processing logs of the exports; see docstrings)
  * inputs per measurement: <base>.AIM (native int16 greyscale, calibration from its processing log), IPL's contour
    RENDERINGS <base>_CT.AIM (periosteal gobj = Script 32 stage 00), <base>_CORT_MASK_CT.AIM, <base>_TRAB_MASK_CT.AIM
    or <base>_TRAB_MASK_CORR_CT.AIM (manually corrected trabecular contour, flag CORR), IPL's SEG (127 cort + 126
    trab; _SEG_DECOMPRESSED.AIM else _SEG.AIM) and the maps <base>_{TRAB_TH,TRAB_SP,TRAB_1N,CORT_TH}_DECOMPRESSED.AIM
    (or _COMPRESSED.AIM).  No raw CORT_MASK / TRAB_MASK / TRAB_SEG / CORT_SEG rasters exist.
  * Ct.Th object, decided PER MEASUREMENT (cort_object 'auto'): 'periosteal_minus_trab' when render(periosteal - IPL's
    trab rendering) == CORT_MASK_CT (IPL's STEP 2/3 re-evaluation derived cort_mask.aim = gobj_to_aim(periosteal) -
    gobj_to_aim(trab_mask.gobj) and CORT_MASK.GOBJ = togobj_from_aim of it: the 2022 distal and 2024 re-evaluations, 345857 /
    487451 / 2422 / 581203); else 'raw_cort' = STEP 1's own stage 28 when render(our 28) fits CORT_MASK_CT better than
    render(periosteal - our trab rendering) (the 51 single-run 2026 diaphyseal evaluations: 876976 render(28) == CORT_MASK_CT
    exactly and CORT_TH == dt_thickness(our 28) exactly, whereas periosteal - trab gives 485 rendering / 335,740 map
    mismatches).  A's object follows the rule (our stage 28 stands in for the unexported CORT_MASK.AIM, flagged when its
    rendering is not exact); the runner records both candidates' mismatches.
  * Tb.Th: Scripts 32, 33 (byte-identical command files) and 34 all write TRAB_SEG as IPL_FNAME5 and then run
    /dt_thickness on it cropped to the trabecular gobj (32/33 lines 452 / 708 / 710; 34 lines 190 / 446 / 448), so the
    scripts' Tb.Th is dt_thickness(TRAB_SEG) = TRAB_TH_tseg.  Both definitions are computed for every measurement and
    both are reported side by side: TRAB_TH_seg = dt_thickness(whole SEG) (what the shipped ORMIR-BQRL pipeline
    computes) and TRAB_TH_tseg.  ONE DEFINITION, NOT TWO (settled 2026-09-16).  Eleven exported TRAB_TH files used to
    be reproduced by the whole-SEG object and not at all by our TRAB_SEG (2422: 0 mismatches with the whole SEG,
    85,187 with TRAB_SEG), which was recorded as a second definition.  It was not one: those eleven are exactly the
    measurements whose TRAB_SEG was never cropped to the trabecular contour, because Script 32 STEP 2's
    '/gobj_maskaimpeel_ow -input_output trab_gauss -gobj <base>_trab_mask.gobj' did not run -- ten SEG processing logs
    carry no D3P_GobjOrAimMaskAimPeel_OW after D3P_Cl_ExtractNumber_CPP (and no D3P_FillOffsetDuplicate either), and
    610892, which ships no SEG, has its evaluation log showing both commands erroring.  An uncropped TRAB_SEG differs
    from the whole SEG only by the 35..69-voxel components inside the cortical contour, which a map cropped to the
    TRABECULAR contour barely sees, so the whole SEG was simply the nearest object the harness could build for them.
    Modelling the failure (trab_seg_mask 'none', now DETECTED from the same logs) reproduces those files far better
    than the whole-SEG substitute did -- pooled over the ten, Tb.Th 37,119 -> 41 mismatching voxels (0..13 each, the
    cohort's ordinary residual) and max |Tb.Th| 4.83e-04 -> 6.99e-06 mm, with SEG 2,244 -> 90, Tb.Sp 853 -> 250 and
    1/Tb.N 176,890 -> 1,434 alongside -- and it is corroborated by IPL's own text three ways: the DT input-object
    count recovered from each delivered TRAB_TH's processing log matches the uncropped object to 0..8 voxels of
    2.8-15.1 M and never the cropped one (off by 12-94 %), 581203's two evaluation logs print 0.207404 mm / 99.1 % valid
    for the run whose mask succeeded and 0.233023 mm / 88.5 % for the run where it errored while the delivered map's
    own proclog says 0.23302, and every delivered map's printed statistics follow suit.  The same model is WRONG on
    the other 106 (950609: TRAB_SEG exact, unmasked 2,483 Tb.Th mismatches), so the detection is measurement-specific.
    The per-measurement --trab-th auto therefore no longer finds a second definition anywhere: with the failure
    modelled the two candidate objects COINCIDE on the eleven, and all 117 resolve to the scripts' TRAB_SEG.  The
    mechanism is kept for provenance and for anyone who wants to force one reading (--trab-th trab_seg / seg).
    OS_LH exports no TRAB_SEG raster, so TRAB_SEG is reconstructed in A as SEG & trabecular rendering, or -- where
    IPL's log says the crop never ran -- as the delivered SEG's own support.  TRAB_SP / TRAB_1N on the whole SEG
    (Script 32 reads IPL_SEGAIM for them), Ct.Th on the cortical compartment.  All dt_* with IPL's Script 32 parameters.
  * two versions of the evaluation script exist in the cohort and are detected per measurement from IPL's SEG
    processing log: (2022, distal) /fill_offset_duplicate on the greyscale border (LH border 'duplicate';
    Distal/CKD/345857 with IPL's renderings: 1 SEG mismatch, vs 17,787 with 'none' and 110,542 with 'zero') and a bounding_box_cut of TRAB_SEG
    (its TRAB_TH map is on that tight grid); (2024, diaphyseal) '/fill_offset_duplicate cort' errors in the log and
    IPL's FFT mirror-pads the data region directly (LH border 'none': 2422 121 SEG mismatches vs 27,669 with
    'duplicate' and 177,608 with 'zero').  The proclog lists Cl_Label before the gobj-mask entry in the 2022 version,
    but the periosteal mask DOES come first there too (Distal/CKD/345857 with IPL's renderings: periosteal_first 1 mismatch,
    gobj_first 262; patella PFJ-0be66a_R: 5 vs 1,481): both assembly orders are computed in B and C; --seg-variant
    periosteal_first (the default, the shipped engine's order) uses that order, 'auto' the proclog's, and 'best' the
    one whose SEG fits IPL's better (a diagnostic, not used for the published records).  --lh-border overrides the border;
    --lh-both computes the other border variants' SEGs.
  * a measurement that ships NO SEG has no SEG processing log to read, and Diaphyseal/BMAT/610892 is the only one in
    the cohort.  Its variants come instead from IPL's own EVALUATION LOG in the measurement folder (detect_variants ->
    find_eval_logs / choose_eval_log / parse_eval_log), which is IPL's transcript of the run that wrote the delivered
    files.  610892 was evaluated twice and ships both logs; the delivered products come from the SECOND run
    (the log whose name carries the extra underscore), which is identified from
    the products themselves rather than assumed -- X1469636_CORT_TH_COMPRESSED.AIM's own processing log is stamped
    inside that run's time window and not inside the first run's, and the three trabecular maps' logs
    carry neither D3P_FillOffsetDuplicate nor a D3P_GobjOrAimMaskAimPeel_OW after D3P_Cl_ExtractNumber_CPP, which is
    the second run's behaviour and not the first's.  That run began at STEP 2 in a fresh IPL session, so two commands
    addressed objects STEP 1 would have left behind and errored: '/fill_offset_duplicate -input cort' ->
    'IPL_Ol_GetAim: Undefined object name (cort)' (log line 221 ff., so LH border 'none', not 'duplicate') and
    '/gobj_maskaimpeel_ow -input_output trab_gauss -gobj ..._trab_mask.gobj' -> 'Undefined object name (trab_gauss)'
    (line 434 ff., so TRAB_SEG was never cropped to the trabecular contour; the FIRST run's same command succeeded,
    '-> Set 4460987 of total 15679104', line 1465).  The third variant, trab_seg_mask ('gobj' / 'none'), is threaded
    through assemble_seg; --trab-seg-mask overrides it.  Where a SEG exists the same fact is read from the SEG's own
    processing log -- a D3P_GobjOrAimMaskAimPeel_OW after D3P_Cl_ExtractNumber_CPP means 'gobj', its absence 'none' --
    which gives 'gobj' for 111 of the 122 OS_LH SEGs and 'none' for 11 (the ten above plus rerun/581203), with no mixed
    case and nothing else changed on the 111.  The log also NAMES the gobj of each masking block ('Gobj File:
    <disk>:[<dir>]<base>_trab_mask.gobj' for the crop, '<base>.gobj' for the periosteal mask), and the name agrees with the
    position on all 122: the 111 all name a _trab_mask.gobj and no _trab_mask.gobj appears anywhere in the 11 that
    have none.  The names are recorded as evidence and a disagreement raises a note (there is none), which closes the
    positional rule's one ambiguity -- a log whose only late masking entry were a periosteal mask.
  * site preset: 'Patient Name ... <UD|D> <Tibia|Radius>' -> Tibia (UD and D) = TIBIA (corner_min 200000 / close2 50),
    Radius = RADIUS (800 / 30).  Evidence: the 2026 diaphyseal D Tibia 876976 / 312922 reproduce IPL's CORT_MASK_CT and
    TRAB_MASK_CT exactly with TIBIA (0 / 0 mismatches) and not with RADIUS (2,871 / 2,883 and 3,157 / 3,146); the D Radius
    220364 is exact with RADIUS only, 655328 with either (empty corner branch); the 581203 D Radius log ran 800 / 30 and ipldt
    replays it stage for stage.  The BMAT D Tibia 610892 log ran 800 / 30: the fallback covers it.
    --site-override forces one; the other preset is tried when the renderings of an uncorrected measurement do not match
    and is adopted only when it at least halves the mismatch (--no-preset-fallback disables that).  The plain 2022
    distal evaluations (950636 / 113739 / 114224 / 779503 / 379176) match NEITHER preset (100k - 400k voxels): their trab gobj was
    re-saved 10 - 23 days after ISQ_TO_AIM and CORT_MASK.GOBJ re-derived minutes later, i.e. contours corrected by
    hand after the automatic run (the validation sets use each measurement's automatic run instead:
    validation/stage_automatic_set.py).
  * quirks handled: gobj-derived AIMs may carry garbage element sizes (the greyscale's are used everywhere);
    misspelled duplicate map files (the correct spelling is read, the duplicate hashed and reported); a measurement
    without SEG (A's trabecular maps and every SEG comparison are skipped and flagged); compressed maps; rerun/ as
    its own group (hashes show whether it duplicates the original folder); the CORR trab (used as IPL's trabecular
    rendering, flagged, A2 added).

PATELLA LAYOUT: XCT_masks_full_grab/<subject>/<base>_*_decompressed.AIM with raw CORT_MASK / TRAB_MASK (periosteal =
their union = stage 00; the renderings are ipldt's render_volume of the raw masks), SEG, TRAB_SEG, CORT_SEG, maps
TRAB_TH (full-SEG definition), TRAB_TH_old (TRAB_SEG), TRAB_SP, TRAB_1N, CORT_TH (object = raw CORT_MASK); TIBIA preset.

EXCLUDED MEASUREMENTS (EXCLUDED_IDS, shared with the figure scripts)
  A measurement whose own files were not produced by the standard pipeline is not a like-for-like comparison against a
  reimplementation of that pipeline, so it is dropped from every AGGREGATE -- exact counts, mismatch and voxel totals, Dice
  ranges, metric differences, the sample-wise regressions and every per-group / per-cohort / per-flag breakdown --
  while its per-measurement row SURVIVES in table.csv / table.md (column 'excluded', carrying the reason), its record
  stays in records.json and in regression_data.json's per-measurement section, and summary.md names it at the top.
  The one entry is Diaphyseal/CKD/991161, whose delivered trabecular contour is not the one IPL's own evaluation used
  (the evidence is in EXCLUDED_IDS; it is also the only measurement segmented with fft.lp_cut_off_freq 0.20000
  rather than 0.30000).  Nothing is ever excluded for disagreeing with ipldt.
  --include-excluded counts every record again, so the pre-exclusion numbers remain reproducible on demand.

SHIPPED-DIRECTORY GUARD (--allow-overwrite-shipped)
  The published result directories under validation/results/ (the records the paper's numbers are built from) are
  never overwritten by accident: a run whose --results is one of them, or lies under one, is REFUSED before anything
  is written (exit status 2) unless --allow-overwrite-shipped is given.  The default --results
  (results/<dataset>/validate/) is not a published directory (validate_from_ipl_contour.guard_shipped_results).

Metrics are reported at the header's x element size AND at 0.0607 mm (validate_from_ipl_contour convention).
Outputs (results/<dataset>/validate/ or results/<dataset>/validate/subset_<tag>/ for a partial run; results/oslh/ itself
holds the inventory, whose records/ use the same <group>_<id>.json names): records/<id>.json (everything,
including sparse joint histograms ours x IPL per map and configuration), table.csv, table.md, summary.md (per-group
and per-bone totals, exact counts, Dice ranges, metric differences, sample-wise regressions per metric and
configuration), regression_data.json (sample-wise metric pairs and pooled voxel-wise joint histograms for plots).

CLI
    python validate_dataset.py --dataset oslh --subjects Distal/CKD/345857 Distal/CKD/487451 Diaphyseal/CKD/2422
    python validate_dataset.py --dataset oslh --groups Distal/CKD rerun --resume
    python validate_dataset.py --dataset patella --subjects PFJ-0be66a_R --backend cpu --skip-dt
    python validate_dataset.py --dataset oslh --resume --include-excluded   # aggregates over every record again
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
import sys
import time
import traceback

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datapaths import lab_path  # noqa: E402  (non-public data roots: validation/datapaths.py)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import ipldt  # noqa: E402
from ipldt import ipl_ops as ops  # noqa: E402
from ipldt import ormir  # noqa: E402
from ipldt.contour import render_volume  # noqa: E402
from ipldt import porosity  # noqa: E402
from ipldt.step1 import RADIUS, TIBIA, cort_trab_separation  # noqa: E402
import validate_against_ipl as vai  # noqa: E402  (VOXEL_MM, IPL_PARAMS, joint_histogram, summarize_joint, ols, icc)
import validate_from_ipl_contour as vfc  # noqa: E402  (compare_masks, compare_labels, grid_of, grid_str, _md, tables)

VOXEL_MM = vai.VOXEL_MM
# our maps: Tb.Th is computed under BOTH definitions IPL has used -- TRAB_TH_seg = dt_thickness(whole SEG) (the patella cohort's
# corrected TRAB_TH, and what the 2026 OS_LH diaphyseal exports contain) and TRAB_TH_tseg = dt_thickness(TRAB_SEG) (Script 32 STEP 3 as
# written: /read trab_seg.aim; the patella's TRAB_TH_old, the 2022 OS_LH distal exports); the layout says which IPL file each is compared with
MAP_NAMES = ("TRAB_TH_seg", "TRAB_TH_tseg", "TRAB_SP", "TRAB_1N", "CORT_TH")
MAP_OBJECT = {"TRAB_TH_seg": "seg", "TRAB_TH_tseg": "trab_seg", "TRAB_SP": "seg", "TRAB_1N": "seg", "CORT_TH": "cort"}
MAP_GOBJ = {"TRAB_TH_seg": "trab", "TRAB_TH_tseg": "trab", "TRAB_SP": "trab", "TRAB_1N": "trab", "CORT_TH": "cort"}
METRICS = (("BVTV", "BV/TV", "-", None), ("TbTh", "Tb.Th (reported)", "mm", "TRAB_TH"),
           ("TbTh_seg", "Tb.Th (whole SEG, as ORMIR-BQRL ships)", "mm", "TRAB_TH_seg"),
           ("TbTh_tseg", "Tb.Th (TRAB_SEG, Scripts 32/33/34)", "mm", "TRAB_TH_tseg"), ("TbSp", "Tb.Sp", "mm", "TRAB_SP"),
           ("TbN", "Tb.N", "1/mm", "TRAB_1N"), ("CtTh", "Ct.Th", "mm", "CORT_TH"))
TRAB_MAPS = ("TRAB_TH_seg", "TRAB_TH_tseg", "TRAB_SP", "TRAB_1N")
CONFIGS = ("A", "B", "C", "A2")
SITE_PARAMS = {"tibia": TIBIA, "radius": RADIUS}
DEFAULT_LOG_DIR = os.path.join(__import__("tempfile").gettempdir(), "ipldt_validate_dataset")

# ------------------------------------------------------------------------------------------- excluded measurements
# Measurements dropped from every AGGREGATE in summary.md and in the tables (exact counts, mismatch and voxel totals,
# Dice ranges, metric differences, sample-wise regressions, the per-cohort / per-group breakdowns and the per-block
# counters).  They are dropped on a property of their own FILES -- a segmentation parameter that differs from the rest
# of the cohort -- and never because they disagree with ipldt.  Their per-measurement row SURVIVES in table.csv /
# table.md with the reason in the column 'excluded', they stay in records.json and in regression_data.json's
# per-measurement section, and summary.md names them at the top; nothing disappears silently.  --include-excluded
# restores the old behaviour (every record counted), so the earlier numbers stay reproducible.
# The same mapping, with the same reason in the same order, drives the figure scripts.
EXCLUDED_IDS = {
    "Diaphyseal/CKD/991161":
        "the delivered trabecular contour is not the one IPL's own evaluation used, so no result for this "
        "measurement is interpretable. The contour file was overwritten about 19.5 min after STEP 1 had already "
        "derived the cortical compartment from the earlier version, and the evaluation was never re-run: the shipped "
        "SEG, maps and cortical mask all belong to a trabecular contour 196,389 voxels LARGER than the shipped "
        "trabecular rendering (IPL's own SEG log prints its relative volume, 4,242,576 voxels, against 4,046,187 "
        "delivered; the same product lands exactly on the delivered rendering for 50 of 51 other measurements). "
        "Recovering that contour -- the bone contour minus the cortical one, rendered -- reproduces IPL's three "
        "trabecular maps with 0 mismatching voxels, against 495,295 with the delivered contour, which is what proves "
        "the mechanism. The two structural anomalies ARE that difference exactly: 214,715 bone-contour voxels in "
        "neither compartment and 17,548 in both, against about 1,122 and 48 typical. Ct.Th is unaffected because it "
        "is the only map whose chain never touches a trabecular contour. It is also the only measurement in the "
        "cohort segmented with fft.lp_cut_off_freq 0.20000 rather than 0.30000, a further sign of a hand "
        "re-evaluation -- but its SEG and its maps agree on that value, so the cut-off is NOT why the numbers "
        "disagree",
}


def exclusion_reason(rec):
    """The reason this record (or id) is excluded from the aggregates, or None when it is counted."""
    rid = rec.get("id") if isinstance(rec, dict) else rec
    return EXCLUDED_IDS.get(rid)


def split_excluded(records, rows=None, include_excluded=False):
    """(kept_records, kept_rows, excluded_records) -- the aggregate set and what was taken out of it.

    'records' and 'rows' are parallel (rows may be None).  With include_excluded nothing is taken out, which is what
    --include-excluded asks for.  Records are never deleted anywhere: this only decides what a statistic is over."""
    if include_excluded:
        return list(records), (list(rows) if rows is not None else None), []
    pairs = list(zip(records, rows)) if rows is not None else [(r, None) for r in records]
    kept = [(r, w) for r, w in pairs if not exclusion_reason(r)]
    excl = [r for r, _ in pairs if exclusion_reason(r)]
    return [r for r, _ in kept], ([w for _, w in kept] if rows is not None else None), excl


# ------------------------------------------------------------------------------------------------ layouts
LAYOUTS = {
    "oslh": dict(
        root=os.environ.get("OSLH_ROOT") or lab_path("Cross_validation_IPL/OS_LH"),
        groups=["Distal/CKD", "Distal/REPRO", "Diaphyseal/BMAT", "Diaphyseal/CKD", "Diaphyseal/REPRO", "rerun"],
        grey=r"^(?P<base>[A-Z]\d{7})\.AIM$",
        periosteal=["{base}_CT.AIM"],                                   # IPL's rendering of the periosteal gobj (stage 00)
        cort_render=["{base}_CORT_MASK_CT.AIM"],
        trab_render=["{base}_TRAB_MASK_CORR_CT.AIM", "{base}_TRAB_MASK_CT.AIM"],  # first hit wins; CORR flagged
        trab_render_corr_pattern="_CORR_",
        raw_cort=None, raw_trab=None,
        trab_seg=["{base}_TRAB_SEG.AIM"],                                # IPL's own TRAB_SEG, configuration A's Tb.Th object
                                                    # (2026-09-25: the SEG & trabecular-rendering rebuild used when no file is
                                                    # staged adds rendering-overlap voxels of 35..69-voxel components that IPL's
                                                    # cl_nr_extract 70 removed -- one voxel on Distal/REPRO/319997 in the cohort)
        cort_seg=["{base}_CORT_SEG.AIM"],                                # staged: the pore cascade's input
        pore=["{base}_PORE.AIM"],                                        # IPL's own cortical pore map
        seg=["{base}_SEG_DECOMPRESSED.AIM", "{base}_SEG.AIM"],
        maps={m: [f"{{base}}_{m}_DECOMPRESSED.AIM", f"{{base}}_{m}_COMPRESSED.AIM"] for m in ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH")},
        ipl_map_for={"TRAB_TH_seg": "TRAB_TH", "TRAB_TH_tseg": "TRAB_TH", "TRAB_SP": "TRAB_SP", "TRAB_1N": "TRAB_1N", "CORT_TH": "CORT_TH"},
        periosteal_is_rendering=True,
        cort_object="auto",                         # per measurement: 'periosteal_minus_trab' (IPL's STEP 2/3 re-derivation,
                                                    # cort_mask.aim = gobj_to_aim(all) - gobj_to_aim(trab)) or 'raw_cort' (STEP 1's
                                                    # own stage 28, the 2026 single-run evaluations); see the module docstring
        trab_th_definition="auto",                  # which Tb.Th definition IPL's TRAB_TH follows: decided per measurement in A
        lh_border="auto", seg_variant="auto", trab_th_grid="auto",
        trab_seg_mask="auto",                       # was /gobj_maskaimpeel_ow with the trabecular contour applied to
                                                    # TRAB_SEG?  'auto': 'gobj' everywhere a SEG exists, and from IPL's
                                                    # evaluation log for the one measurement that ships none (610892,
                                                    # where the command errored); see detect_variants
        site="auto",
        bone_key="region_bone",
    ),
    "patella": dict(
        root=vai.DATA_ROOT,
        groups=None,
        grey=r"^(?P<base>[A-Z]\d{7})\.AIM$",
        periosteal=None,                                                # CORT_MASK | TRAB_MASK (= stage 00)
        cort_render=None, trab_render=None, trab_render_corr_pattern=None,
        raw_cort=["{base}_CORT_MASK_decompressed.AIM"], raw_trab=["{base}_TRAB_MASK_decompressed.AIM"],
        trab_seg=["{base}_TRAB_SEG_decompressed.AIM"], cort_seg=["{base}_CORT_SEG_decompressed.AIM"],
        pore=["{base}_PORE_decompressed.AIM"],
        seg=["{base}_SEG_decompressed.AIM"],
        maps={m: [f"{{base}}_{m}_decompressed.AIM"] for m in ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH", "TRAB_TH_old")},
        ipl_map_for={"TRAB_TH_seg": "TRAB_TH", "TRAB_TH_tseg": "TRAB_TH_old", "TRAB_SP": "TRAB_SP", "TRAB_1N": "TRAB_1N", "CORT_TH": "CORT_TH"},
        periosteal_is_rendering=True,
        cort_object="raw_cort",
        trab_th_definition="seg",
        lh_border="duplicate", seg_variant="periosteal_first", trab_th_grid="seg", trab_seg_mask="gobj",
        site="tibia",
        bone_key="patella",
    ),
}

_T0 = time.time()
_LOG_FH = [None]


def log(*a):
    line = f"[{time.time() - _T0:8.1f}s] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    if _LOG_FH[0] is not None:
        try:
            _LOG_FH[0].write(line + "\n")
            _LOG_FH[0].flush()
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------ small helpers
def grid_of(v):
    return vfc.grid_of(v)


def grid_str(g):
    return vfc.grid_str(g)


def aim_header(path):
    """(type, dim, pos) of an AIM v020 without reading its data."""
    with open(path, "rb") as fh:
        raw = fh.read(4096)
    if raw[:12] == b"AIMDATA_V030":
        return None
    pre = struct.unpack("<5i", raw[:20])
    hsize = pre[1]
    ints = struct.unpack("<" + "i" * (hsize // 4), raw[20:20 + hsize])
    return dict(type=ints[5], dim=tuple(ints[9:12]), pos=tuple(ints[6:9]))


def md5_of(path, block=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def bvol(data_bool, dim, pos):
    """A bool volume dict (data (z, y, x), dim, pos)."""
    return dict(data=np.asarray(data_bool, bool), dim=tuple(int(x) for x in dim), pos=tuple(int(x) for x in pos))


def set_vol(v):
    return bvol(np.asarray(v["data"]) != 0, v["dim"], v["pos"])


def on(v, dim, pos):
    """v pasted onto (dim, pos) by global position, as a bool array."""
    return np.asarray(ops.on_grid(v, dim, pos)) != 0


def bbox_of(v):
    """(dim, pos) of the tight box of a bool volume (None when empty)."""
    d = np.asarray(v["data"])
    nz = np.nonzero(d)
    if nz[0].size == 0:
        return None
    lo = [int(nz[2].min()), int(nz[1].min()), int(nz[0].min())]
    hi = [int(nz[2].max()) + 1, int(nz[1].max()) + 1, int(nz[0].max()) + 1]
    return tuple(hi[i] - lo[i] for i in range(3)), tuple(int(v["pos"][i]) + lo[i] for i in range(3))


def cut(v, grid):
    dim, pos = grid
    return bvol(on(v, dim, pos), dim, pos)


def count(v):
    return int(np.count_nonzero(v["data"]))


def sparse_hist(J):
    """The joint histogram as [[ours, ipl, count], ...] over its non-zero cells (the (0, 0) cell included)."""
    o, i = np.nonzero(J)
    return [[int(a), int(b), int(J[a, b])] for a, b in zip(o, i)]


def compare_maps_full(ours, ipl, voxel_mm=VOXEL_MM):
    """Two integer maps on any grids -> summarize_joint of the joint histogram on the union grid, both grids, the
    sparse joint histogram (for the voxel-wise regression plots) and the most frequent mismatching value pairs."""
    dim, pos = ops.union_grid(ours, ipl)
    a = np.asarray(ops.on_grid(ours, dim, pos)).astype(np.int16)
    b = np.asarray(ops.on_grid(ipl, dim, pos)).astype(np.int16)
    J = vai.joint_histogram(a, b)
    d = vai.summarize_joint(J, voxel_mm)
    d.update(grid_ours=grid_of(ours), grid_ipl=grid_of(ipl), grids_equal=grid_of(ours) == grid_of(ipl), union_grid=(tuple(dim), tuple(pos)),
             hist=sparse_hist(J))
    if d["mismatches"]:
        m = a != b
        pairs, counts = np.unique(np.stack([a[m], b[m]], 1), axis=0, return_counts=True)
        order = np.argsort(-counts)[:12]
        d["value_pairs"] = [[int(pairs[k, 0]), int(pairs[k, 1]), int(counts[k])] for k in order]
        zs = np.unique(np.nonzero(m)[0])
        d["slices"] = [int(z + pos[2]) for z in zs[:60]]
        d["n_slices"] = int(zs.size)
    return d


def support_outside(map_vol, region_vol):
    """Voxels with a non-zero map value that lie outside `region` (both pasted on the union grid)."""
    dim, pos = ops.union_grid(map_vol, region_vol)
    m = np.asarray(ops.on_grid(map_vol, dim, pos)) != 0
    r = on(region_vol, dim, pos)
    return int((m & ~r).sum())


_MONTHS = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def vms_time(text):
    """A VMS date and time ('d-MON-yyyy hh:mm:ss.cc') -> seconds since the epoch (None when unparsable)."""
    m = re.search(r"(\d{1,2})-([A-Z]{3})-(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})", text or "")
    if not m:
        return None
    import calendar
    d, mo, y, hh, mm, ss = int(m.group(1)), _MONTHS.get(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))
    if not mo:
        return None
    return calendar.timegm((y, mo, d, hh, mm, ss, 0, 0, 0))


def proclog_field(vol, key):
    m = re.search(r"^" + re.escape(key) + r"\s+(.*?)\s*$", (vol or {}).get("proclog", "") or "", re.M)
    return m.group(1).strip() if m else None


def gobj_timeline(grey, inp, layout):
    """Creation dates of the contour files behind the renderings (the 'Original Creation-Date' of a
    D3P_GobjCreateAimPeel export) against the evaluation start (the greyscale's ISQ_TO_AIM 'Time' = STEP 1).  A
    trabecular contour saved long after STEP 1 under the standard name is a later manual correction (2422: TRAB_MASK.GOBJ
    saved about eight weeks after STEP 1, CORT_MASK.GOBJ 27 min after that).  Threshold 15 min: a single run writes both gobjs
    within 1.5 min of ISQ_TO_AIM (876976 / 312922 / 655328 / 581203), whereas the re-saved contours of the plain distal evaluations
    came 32 min (802246), 55 min (386723) and 10 - 23 days (950636 / 969383 / 113739 / 379176 / 114224 / 779503) later and none of those
    is reproduced by STEP 1 with either preset (100k - 720k voxels)."""
    out = {}                                                   # intervals only: no dates or file names are recorded
    t0 = vms_time(proclog_field(grey, "Time"))
    for k, role in (("periosteal", "periosteal"), ("cort", "cort_render"), ("trab", "trab_render")):
        v = inp.vol(role) if layout.get(role) else None
        if v is None:
            continue
        t = vms_time(proclog_field(v, "Original Creation-Date"))
        if t is not None and t0 is not None:
            out[f"{k}_gobj_minus_step1_h"] = (t - t0) / 3600.0
    h = out.get("trab_gobj_minus_step1_h")
    out["trab_gobj_saved_after_step1"] = bool(h is not None and h > 0.25)
    return out


def mem_info():
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_info()
        out = dict(rss_gb=mi.rss / 1e9, peak_rss_gb=getattr(mi, "peak_wset", mi.rss) / 1e9)
    except Exception:
        out = {}
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        out.update(gpu_pool_used_gb=pool.used_bytes() / 1e9, gpu_pool_total_gb=pool.total_bytes() / 1e9)
    except Exception:
        pass
    return out


def gpu_release():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------ discovery
class Measurement:
    """One measurement of a layout: its folder, base, group, the files by role and the metadata parsed from the
    greyscale processing log (patient name, bone, region, site index, calibration)."""

    def __init__(self, layout_name, layout, group, folder, base):
        self.layout_name, self.L, self.group, self.folder, self.base = layout_name, layout, group, folder, base
        self.id = (f"{group}/{os.path.basename(folder)}" if group else os.path.basename(folder)).replace("\\", "/")
        self.tag = self.id.replace("/", "_")
        self.files = {}
        self.notes = []
        self.duplicates = []
        self.corr = False
        files = sorted(os.listdir(folder))
        self.files["grey"] = os.path.join(folder, f"{base}.AIM")
        for role in ("periosteal", "cort_render", "trab_render", "raw_cort", "raw_trab", "trab_seg", "cort_seg", "seg", "pore"):
            pats = layout.get(role)
            self.files[role] = self._first(pats, files)
        if self.files.get("trab_render") and layout.get("trab_render_corr_pattern"):
            self.corr = layout["trab_render_corr_pattern"] in os.path.basename(self.files["trab_render"])
        self.map_files = {}
        for m, pats in layout["maps"].items():
            f = self._first(pats, files)
            if f:
                self.map_files[m] = f
        # misspelled duplicates: files that start with <base>_<MAP> but are not the expected spelling
        expected = {os.path.basename(f) for f in list(self.files.values()) + list(self.map_files.values()) if f}
        for f in files:
            if f.startswith(base + "_") and f.upper().endswith(".AIM") and f not in expected:
                self.duplicates.append(os.path.join(folder, f))
        self.meta = {}

    def _first(self, pats, files):
        if not pats:
            return None
        for pat in pats:
            name = pat.format(base=self.base)
            if name in files:
                return os.path.join(self.folder, name)
        return None

    def parse_grey_meta(self, grey):
        pl = grey.get("proclog", "")
        m = re.search(r"Patient Name\s+(.*)", pl)
        name = m.group(1).strip() if m else ""
        site = re.search(r"^Site\s+(\d+)", pl, re.M)
        region, bone = None, None
        mm = re.search(r"\b(UD|D)\s+(Tibia|Radius)\b", name, re.I)
        if mm:
            region, bone = mm.group(1).upper(), mm.group(2).capitalize()
        elif self.L["bone_key"] == "patella":
            region, bone = "", "Patella"
        st = re.search(r";\s*([A-Za-z]+study)\b", name, re.I)          # the study token only; the name is not kept
        self.meta = dict(study_label=(st.group(1) if st else None), site_index=int(site.group(1)) if site else None,
                         region=region, bone=bone,
                         bone_key=("Patella" if self.L["bone_key"] == "patella" else f"{region or '?'} {bone or '?'}"))
        return self.meta

    def preset_name(self, override="auto"):
        """TIBIA for a tibia (UD and D), RADIUS for a radius (see the module docstring for the evidence)."""
        if override and override != "auto":
            return override
        if self.L["site"] != "auto":
            return self.L["site"]
        if self.meta.get("bone") == "Tibia":
            return "tibia"
        return "radius"


def discover(layout_name, root=None, groups=None, subjects=None):
    L = dict(LAYOUTS[layout_name])
    root = root or L["root"]
    grey_re = re.compile(L["grey"])
    out = []
    if L["groups"]:
        wanted_groups = [g.replace("\\", "/") for g in (groups or L["groups"])]
        for g in wanted_groups:
            gdir = os.path.join(root, g)
            if not os.path.isdir(gdir):
                log(f"group folder missing: {gdir}")
                continue
            for meas in sorted(os.listdir(gdir)):
                folder = os.path.join(gdir, meas)
                if not os.path.isdir(folder):
                    continue
                bases = [grey_re.match(f).group("base") for f in os.listdir(folder) if grey_re.match(f)]
                for base in sorted(bases):
                    out.append(Measurement(layout_name, L, g, folder, base))
    else:
        for subj in sorted(os.listdir(root)):
            folder = os.path.join(root, subj)
            if not os.path.isdir(folder):
                continue
            bases = [grey_re.match(f).group("base") for f in os.listdir(folder) if grey_re.match(f)]
            if len(bases) != 1:
                # the patella folders hold exactly one <base>.AIM; take the one that has a SEG
                bases = [b for b in bases if os.path.exists(os.path.join(folder, f"{b}_SEG_decompressed.AIM"))]
            for base in sorted(bases):
                out.append(Measurement(layout_name, L, None, folder, base))
    if subjects:
        keep = []
        for m in out:
            keys = {m.id, m.tag, os.path.basename(m.folder), m.base, m.id.replace("/", "\\")}
            if any(s.replace("\\", "/") in keys or s == m.id.split("/")[-1] for s in subjects):
                keep.append(m)
        missing = [s for s in subjects if not any(s.replace("\\", "/") in {m.id, m.tag, os.path.basename(m.folder), m.base} or s == m.id.split("/")[-1] for m in keep)]
        if missing:
            log(f"WARNING: --subjects not found in the layout: {missing}")
        out = keep
    return out


def read_inventory(results_root):
    """The inventory agent's validation/results/oslh/inventory.json, when present (never required)."""
    for cand in (os.path.join(results_root, "inventory.json"), os.path.join(HERE, "results", "oslh", "inventory.json")):
        if os.path.exists(cand):
            try:
                inv = json.load(open(cand))
                log(f"inventory: {cand} read")
                return inv
            except Exception as e:
                log(f"inventory: {cand} unreadable ({e})")
    return None


def inventory_entry(inv, meas):
    """Best-effort lookup of a measurement in the inventory (id / folder / base / measurement number)."""
    if not inv:
        return None
    items = inv.get("records", inv.get("measurements", inv)) if isinstance(inv, dict) else inv
    if isinstance(items, list):
        # the inventory's records carry 'group' and 'measurement': match the id exactly (rerun/581203 and Diaphyseal/CKD/581203 share a base)
        for v in items:
            if isinstance(v, dict) and "group" in v and "measurement" in v and f"{v['group']}/{v['measurement']}".replace("\\", "/") == meas.id:
                return v
    if isinstance(items, dict):
        for k, v in items.items():
            kk = str(k).replace("\\", "/")
            if kk in (meas.id, meas.tag, meas.base) or kk.endswith("/" + meas.id):
                return v
        return None
    if isinstance(items, list):
        for v in items:
            if not isinstance(v, dict):
                continue
            vals = {str(x).replace("\\", "/") for x in v.values() if isinstance(x, (str, int))}
            if meas.id in vals or meas.base in vals or meas.tag in vals or any(str(x).endswith(meas.id) for x in vals):
                return v
    return None


# ------------------------------------------------------------------------------------------------ Laplace-Hamming
def laplace_hamming_threshold(native_int16, el_size_mm, border="duplicate", keep_border=False, pad_offset=ormir.LH_PAD_OFFSET,
                              dtype=ormir.LH_DTYPE, lp_cut_off_freq=ormir.LP_CUT_OFF_FREQ):
    """ipldt.ormir.laplace_hamming_threshold with a selectable treatment of the 1-voxel border that IPL's
    /bounding_box_cut -border 1 1 1 + /offset_add put around the greyscale before /fft_laplace_hamming:
      'duplicate'  the border filled by /fill_offset_duplicate (np.pad edge, then the power-of-2 mirror padding) --
                   the 2022 script and the patella evaluation (= the ipldt function);
      'none'       no border at all: the 2024 script's '/fill_offset_duplicate cort' fails on an undefined object and
                   IPL's FFT mirror-pads the data region directly (2422: 121 SEG mismatches with 'none' vs 27,669 with
                   'duplicate' and 177,608 with 'zero', IPL's own renderings);
      'zero'       the border left at zero (refuted on 2422, kept for --lh-border).
    The padding, FFT, filter, scaling and truncation are ipldt.ormir.lh_filter_core ITSELF (delegated since
    2026-09-17, so the harness and the engine cannot drift); pad_offset is its power-of-two padding offset -- 'ceil'
    (IPL's rule, probe 21) or 'floor' (the rule every result under validation/results was computed with before
    2026-09-17; --lh-pad-offset floor reproduces them); dtype is its FFT / filter arithmetic -- 'float32' (the shipped
    engine, ormir.LH_DTYPE) or 'float64' (--lh-dtype float64, the opt-in under study since 2026-09-17: the same
    operations in double precision, rounded to float32 before IPL's float -> short conversion).  keep_border returns
    the thresholded volume on the greyscale grid grown by one voxel per side (pos - 1) for the 2022 component
    labelling (zeros there for 'none'); otherwise the native grid."""
    el = tuple(float(e) for e in el_size_mm)
    native = np.asarray(native_int16)
    if border == "duplicate":
        extended = np.pad(native, ((1, 1), (1, 1), (1, 1)), mode="edge").astype(np.float32)
    elif border == "zero":
        extended = np.pad(native, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=0).astype(np.float32)
    elif border == "none":
        extended = native.astype(np.float32)
    else:
        raise ValueError(f"border must be 'duplicate', 'none' or 'zero' (got {border!r})")
    lh, lh_int16 = ormir.lh_filter_core(extended, el, pad_offset, dtype, lp_cut_off_freq)
    del extended, lh
    bm = (lh_int16 >= ormir.LH_THRESHOLD) & (lh_int16 <= ormir.INT16_MAX)
    del lh_int16
    if border == "none":
        return np.pad(bm, ((1, 1), (1, 1), (1, 1)), mode="constant", constant_values=False) if keep_border else bm
    return bm if keep_border else bm[1:-1, 1:-1, 1:-1]


# --------------------------------------------------------------------------------- IPL's own evaluation log (.LOG)
# Read ONLY when a measurement ships no SEG, whose processing log is otherwise the source of these variants.  An IPL
# job log is a flat transcript: a command is a line '/name' at column 0, its '  -option value' arguments follow, then
# the operator's output until '!%  /name completed'.  A command that could not find its input object prints
# 'IPL_Ol_GetAim: Undefined object name (x)' inside its own block and still 'completes', so a failure is only visible
# there -- which is exactly what happened twice in Diaphyseal/BMAT/610892's second run (see parse_eval_log).
_IPL_CMD_RE = re.compile(r"^/([a-z0-9_]+)\s*$")
_IPL_OPT_RE = re.compile(r"^ {2,}-(\S+)\s+(.*?)\s*$")
_IPL_DONE_RE = re.compile(r"^!%\s+/([a-z0-9_]+) completed")
_IPL_ERR_RE = re.compile(r"IPL_Ol_GetAim: Undefined object name|Can't open file .* for reading|Error reading file:")


def ipl_log_blocks(lines):
    """The command blocks of an IPL job log, in order: dict(name, args, line (1-based), run, ok, error, body).
    'run' counts the '/IPL Processing Starts' banners, so one sub-run = one IPL session = one chain of objects."""
    blocks, cur, run = [], None, 0

    def close():
        if cur is not None:
            body = "\n".join(cur["body"])
            err = _IPL_ERR_RE.search(body)
            cur["ok"] = err is None
            cur["error"] = next((ln.strip() for ln in cur["body"] if _IPL_ERR_RE.search(ln)), None)
            blocks.append(cur)

    for i, ln in enumerate(lines):
        if "IPL Processing Starts" in ln:
            run += 1
        m = _IPL_CMD_RE.match(ln)
        if m:
            close()
            cur = dict(name=m.group(1), args={}, line=i + 1, run=run, body=[])
            continue
        if cur is None:
            continue
        mo = _IPL_OPT_RE.match(ln)
        if mo and not cur["body"]:
            cur["args"][mo.group(1)] = mo.group(2)
            continue
        cur["body"].append(ln)
        md = _IPL_DONE_RE.match(ln)
        if md and md.group(1) == cur["name"]:
            close()
            cur = None
    close()
    return blocks


def parse_eval_log(path):
    """The Script 32 STEP 2 facts one IPL evaluation log records, with the line that carries each.

    The segmentation is the sub-run that contains /fft_laplace_hamming; inside it,
      lh_border       'duplicate' when /fill_offset_duplicate ran on the very object /fft_laplace_hamming then read
                      (the 2022 / first-run behaviour), 'none' when it errored or ran on something else -- in
                      610892's and 581203's SECOND runs the script starts at STEP 2 in a fresh IPL session, so its
                      '-input cort' names an object STEP 1 would have left behind and the command reports
                      'IPL_Ol_GetAim: Undefined object name (cort)': the Laplace-Hamming input has no border.
      seg_variant     'periosteal_first' when the first /gobj_maskaimpeel_ow that succeeded precedes the first
                      /cl_nr_extract, 'gobj_first' when the labelling comes first; None when the log cannot say.
      trab_seg_mask   'gobj' when the /gobj_maskaimpeel_ow with the trabecular contour that follows the labelling
                      succeeded, 'none' when it errored (610892 / 581203 run 2: '-input_output trab_gauss' ->
                      'Undefined object name (trab_gauss)', so TRAB_SEG is the raw /cl_nr_extract output and the
                      trabecular compartment of SEG is never cropped to the trabecular contour), None when absent.
    Nothing here is inferred from our own numbers: every field points at a line of IPL's transcript."""
    with open(path, "r", encoding="latin-1", errors="replace") as fh:
        lines = fh.read().splitlines()
    out = dict(path=path, file=os.path.basename(path), job_start=None, job_end=None, segmentation_run=None,
               lh_border=None, seg_variant=None, trab_seg_mask=None, evidence={})
    for ln in lines[:12]:
        m = re.search(r"\d{1,2}-[A-Z]{3}-\d{4}\s+\d{1,2}:\d{2}:\d{2}\.\d+", ln)
        if m:
            out["job_start"] = m.group(0)
            break
    m = re.search(r"job terminated at\s+(\d{1,2}-[A-Z]{3}-\d{4}\s+\d{1,2}:\d{2}:\d{2}\.\d+)", "\n".join(lines))
    out["job_end"] = m.group(1) if m else None
    out["job_start_epoch"], out["job_end_epoch"] = vms_time(out["job_start"]), vms_time(out["job_end"])
    blocks = ipl_log_blocks(lines)
    out["n_commands"] = len(blocks)
    out["n_runs"] = max([b["run"] for b in blocks], default=0)
    lh = [b for b in blocks if b["name"] == "fft_laplace_hamming"]
    if not lh:
        out["evidence"]["no_fft_laplace_hamming"] = True
        return out
    lh = lh[-1]
    out["segmentation_run"] = lh["run"]
    sub = [b for b in blocks if b["run"] == lh["run"]]
    lh_input = lh["args"].get("input")
    out["evidence"]["fft_laplace_hamming"] = dict(line=lh["line"], input=lh_input)
    reads = [b for b in sub if b["name"] == "read" and b["line"] < lh["line"]]
    out["greyscale_read"] = reads[0]["args"].get("filename") if reads else None
    out["evidence"]["greyscale_read"] = dict(line=reads[0]["line"], filename=out["greyscale_read"]) if reads else dict(present=False)
    fills = [b for b in sub if b["name"] == "fill_offset_duplicate" and b["line"] < lh["line"]]
    f = fills[-1] if fills else None
    out["lh_border"] = "duplicate" if (f is not None and f["ok"] and f["args"].get("input") == lh_input) else "none"
    out["evidence"]["fill_offset_duplicate"] = (dict(line=f["line"], input=f["args"].get("input"), ok=f["ok"], error=f["error"],
                                                     applied_to_lh_input=(f["args"].get("input") == lh_input))
                                                if f is not None else dict(present=False))
    masks = [b for b in sub if b["name"] == "gobj_maskaimpeel_ow"]
    extracts = [b for b in sub if b["name"] == "cl_nr_extract"]
    first_ok_mask = next((b for b in masks if b["ok"]), None)
    if first_ok_mask is not None and extracts:
        out["seg_variant"] = "periosteal_first" if first_ok_mask["line"] < extracts[0]["line"] else "gobj_first"
        out["evidence"]["seg_variant"] = dict(first_gobj_mask_line=first_ok_mask["line"], first_gobj_mask_gobj=first_ok_mask["args"].get("gobj_filename"),
                                              first_cl_nr_extract_line=extracts[0]["line"])
    else:
        out["evidence"]["seg_variant"] = dict(undecidable=True, n_gobj_masks=len(masks), n_cl_nr_extract=len(extracts))
    after = extracts[0]["line"] if extracts else -1
    tm = [b for b in masks if b["line"] > after and str(b["args"].get("gobj_filename", "")).lower().endswith("_trab_mask.gobj")]
    if tm:
        b = tm[0]
        out["trab_seg_mask"] = "gobj" if b["ok"] else "none"
        set_line = next((ln.strip() for ln in b["body"] if "-> Set " in ln), None)
        out["evidence"]["trab_gobj_mask"] = dict(line=b["line"], input_output=b["args"].get("input_output"), gobj=b["args"].get("gobj_filename"),
                                                 ok=b["ok"], error=b["error"], set_line=set_line)
    else:
        out["evidence"]["trab_gobj_mask"] = dict(present=False, n_gobj_masks_after_labelling=0)
    return out


def find_eval_logs(folder):
    """IPL's evaluation logs shipped next to a measurement's AIMs (EVAL_<script>_<patient>_<measurement>.LOG; a
    measurement that was evaluated twice ships one per run, the names differing only by an extra underscore)."""
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [os.path.join(folder, f) for f in names if f.upper().startswith("EVAL_") and f.upper().endswith(".LOG")]


def product_chain_facts(proclog):
    """The two STEP 2 facts a DELIVERED file's own processing log carries, or None when its chain is not a SEG chain.
    A map written from SEG / TRAB_SEG inherits the whole segmentation chain, so it records whether
    D3P_FillOffsetDuplicate ran and whether a D3P_GobjOrAimMaskAimPeel_OW followed the component extraction."""
    pl = proclog or ""
    if "D3P_SupThreshold" not in pl:
        return None
    i_ex = pl.find("D3P_Cl_ExtractNumber_CPP")
    return dict(fill_offset_duplicate=("D3P_FillOffsetDuplicate" in pl),
                gobj_mask_after_extract=bool(i_ex >= 0 and pl.find("D3P_GobjOrAimMaskAimPeel_OW", i_ex) >= 0),
                gobj_mask_entries=pl.count("D3P_GobjOrAimMaskAimPeel_OW"))


def _latest_vms(text):
    ts = re.findall(r"\d{1,2}-[A-Z]{3}-\d{4}\s+\d{1,2}:\d{2}:\d{2}\.\d+", text or "")
    best = max((vms_time(t) or -1, t) for t in ts) if ts else None
    return (best[1], best[0]) if best and best[0] > 0 else (None, None)


def choose_eval_log(paths, products=None):
    """Which of a measurement's evaluation runs produced the files it ships, and why.

    A measurement evaluated twice ships two logs but ONE set of products, so the run must be identified from the
    products themselves, never assumed.  Two independent tests, both recorded:
      chain  every delivered file whose processing log still carries the segmentation chain votes on the two facts
             parse_eval_log reads out of the log (FillOffsetDuplicate present, gobj mask after the labelling);
      date   a job that had already terminated when a delivered file recorded its newest processing step cannot have
             written that file, so any product stamped after a run's 'job terminated at' REFUTES that run; a product
             stamped inside a run's window corroborates it.  (A map inherits the greyscale's ISQ_TO_AIM stamp from
             the FIRST run, so the date test only ever refutes or corroborates -- it never counts votes.)
    Returns (parsed, report) with the full per-log table in `report`."""
    parsed = [parse_eval_log(p) for p in paths]
    facts = {k: product_chain_facts(v) for k, v in (products or {}).items()}
    stamps = {k: _latest_vms(v) for k, v in (products or {}).items()}
    rows = []
    for pr in parsed:
        agree = disagree = 0
        votes = {}
        for k, f in facts.items():
            if not f:
                continue
            v = [(f["fill_offset_duplicate"], pr["lh_border"] == "duplicate"),
                 (f["gobj_mask_after_extract"], pr["trab_seg_mask"] == "gobj")]
            ok = [a == b for a, b in v]
            agree += sum(ok)
            disagree += len(ok) - sum(ok)
            votes[k] = dict(fill_offset_duplicate=f["fill_offset_duplicate"], gobj_mask_after_extract=f["gobj_mask_after_extract"], agrees=sum(ok), of=len(ok))
        s, e0 = pr["job_start_epoch"], pr["job_end_epoch"]
        after = [k for k, (t, e) in stamps.items() if e and e0 is not None and e > e0]
        inside = [k for k, (t, e) in stamps.items() if e and s is not None and e0 is not None and s <= e <= e0]
        rows.append(dict(file=pr["file"], job_start=pr["job_start"], job_end=pr["job_end"], lh_border=pr["lh_border"],
                         seg_variant=pr["seg_variant"], trab_seg_mask=pr["trab_seg_mask"], chain_agreements=agree,
                         chain_disagreements=disagree, chain_votes=votes, product_timestamps={k: t for k, (t, e) in stamps.items()},
                         products_stamped_after_job_end=after, products_stamped_inside_job_window=inside,
                         refuted_by_date=bool(after)))
    if not parsed:
        return None, dict(candidates=[])
    order = sorted(range(len(parsed)), key=lambda i: (not rows[i]["refuted_by_date"],
                                                      rows[i]["chain_agreements"] - rows[i]["chain_disagreements"],
                                                      len(rows[i]["products_stamped_inside_job_window"]), parsed[i]["job_start_epoch"] or 0), reverse=True)
    best, b = order[0], rows[order[0]]
    if len(parsed) == 1:
        why = "the only evaluation log in the folder" + (f"; {b['chain_agreements']} of {b['chain_agreements'] + b['chain_disagreements']} chain facts in the "
                                                         f"delivered files' own processing logs agree with it" if products else "")
    else:
        beaten = [r for i, r in enumerate(rows) if i != best]
        others = ", ".join("{} {}/{}".format(r["file"], r["chain_agreements"], r["chain_agreements"] + r["chain_disagreements"]) for r in beaten)
        why = (f"{b['chain_agreements']} of {b['chain_agreements'] + b['chain_disagreements']} chain facts in the delivered files' own processing logs agree with this run "
               f"(the other candidate{'s' if len(beaten) > 1 else ''}: {others}); "
               f"{', '.join(b['products_stamped_inside_job_window']) or 'no product'} stamped inside its [{b['job_start']} .. {b['job_end']}] window and none stamped after it"
               + (f"; {', '.join(sorted({p for r in beaten for p in r['products_stamped_after_job_end']}))} stamped after the other run{'s' if len(beaten) > 1 else ''} had terminated"
                  if any(r["products_stamped_after_job_end"] for r in beaten) else ""))
    return parsed[best], dict(chosen=parsed[best]["file"], reason=why, candidates=rows)


def detect_variants(seg_vol, layout, folder=None, base=None, products=None):
    """The script variants of one measurement, each read off a record IPL itself wrote.

    PREFERRED SOURCE -- IPL's SEG processing log: the greyscale border of the Laplace-Hamming input ('duplicate' when
    D3P_FillOffsetDuplicate is logged = the 2022 script, else 'none' = the 2024 script), the SEG assembly order
    ('periosteal_first' when the gobj mask entry precedes the component labelling, 'gobj_first' when the labelling
    comes first) and whether TRAB_SEG was cropped to the trabecular contour ('gobj' when a D3P_GobjOrAimMaskAimPeel_OW
    follows D3P_Cl_ExtractNumber_CPP, 'none' when none does).

    FALLBACK -- IPL's EVALUATION LOG, used only when the measurement ships no SEG (one measurement in the OS_LH
    cohort, Diaphyseal/BMAT/610892), where there is no SEG processing log to read and the previous code fell back to
    the cohort's usual values.  The evaluation log is IPL's own transcript of the run, so the same three facts are
    read from the commands themselves -- including the two that ERRORED in 610892's second run and left the delivered
    products without a duplicated Laplace-Hamming border and with an unmasked TRAB_SEG.  When several runs exist the
    run is identified from the delivered files' own processing logs (choose_eval_log), never assumed.

    trab_seg_mask ('gobj' = TRAB_SEG cropped to the trabecular contour, the normal behaviour; 'none' = the
    /gobj_maskaimpeel_ow did not run and TRAB_SEG is the raw component extraction) is read on BOTH paths from the same
    fact -- whether the masking command appears after the component extraction -- in the SEG's processing log where one
    exists, and in IPL's transcript of the run where it does not.  The two paths agree wherever both records exist:
    610892's evaluation log shows the command erroring, and the ten SEG proclogs that lack the entry lack
    D3P_FillOffsetDuplicate with it, which is the same failure's other half.  The SEG path also records the gobj FILE
    NAME each masking block prints (evidence 'gobj_mask_files' / 'gobj_mask_names_trab_mask'): a '_trab_mask.gobj'
    identifies the crop independently of where it sits in the chain, the two readings agree on all 122 OS_LH SEG
    logs, and a disagreement is recorded as a note for a human rather than silently deciding.

    Everything is recorded under 'evidence' and 'source'; --lh-border / --seg-variant / --trab-seg-mask override."""
    def _gobj_mask_files(pl):
        """The gobj file each D3P_GobjOrAimMaskAimPeel_OW block of a processing log masked with, in log order.
        IPL prints it as 'Gobj File:  <disk>:[<directory>]<base>[_trab_mask].gobj' inside the block."""
        out, i = [], pl.find("D3P_GobjOrAimMaskAimPeel_OW")
        while i >= 0:
            nxt = pl.find("Procedure:", i)
            g = re.search(r"Gobj File:\s+(\S+)", pl[i:nxt if nxt > i else len(pl)])
            out.append(g.group(1).strip().lower() if g else "?")
            i = pl.find("D3P_GobjOrAimMaskAimPeel_OW", i + 1)
        return out

    out = dict(lh_border=layout["lh_border"], seg_variant=layout["seg_variant"],
               trab_seg_mask=layout.get("trab_seg_mask", "gobj"), source=None, evidence={}, notes=[])
    pl = (seg_vol or {}).get("proclog", "") if seg_vol else ""
    if pl:
        out["source"] = "seg_proclog"
        fill = "D3P_FillOffsetDuplicate" in pl
        i_mask = pl.find("D3P_GobjOrAimMaskAimPeel_OW")
        i_lab = pl.find("D3P_Cl_Label_CPP")
        i_ex = pl.find("D3P_Cl_ExtractNumber_CPP")
        mask_after_extract = bool(i_ex >= 0 and pl.find("D3P_GobjOrAimMaskAimPeel_OW", i_ex) >= 0)
        out["evidence"] = dict(fill_offset_duplicate=fill, gobj_mask_index=i_mask, cl_label_index=i_lab,
                               laplace_hamming=("D3P_FFT_LaplaceHamming" in pl),
                               gobj_mask_entries=pl.count("D3P_GobjOrAimMaskAimPeel_OW"),
                               gobj_mask_after_extract=mask_after_extract)
        if layout["lh_border"] == "auto":
            out["lh_border"] = "duplicate" if fill else "none"
        if layout["seg_variant"] == "auto":
            out["seg_variant"] = "periosteal_first" if (i_mask >= 0 and i_lab >= 0 and i_mask < i_lab) else "gobj_first"
        if out["trab_seg_mask"] == "auto":
            # READ FROM THE LOG, exactly as the evaluation-log branch reads it.  Script 32 STEP 2 crops the trabecular
            # component extraction to the trabecular contour with '/gobj_maskaimpeel_ow -input_output trab_gauss -gobj
            # <base>_trab_mask.gobj', which the SEG's processing log records as a D3P_GobjOrAimMaskAimPeel_OW AFTER the
            # D3P_Cl_ExtractNumber_CPP of the labelling.  Its presence means TRAB_SEG was cropped ('gobj'); its
            # absence means the command did not run, so TRAB_SEG is the raw extraction ('none') -- the same absence
            # that IPL's evaluation log shows being CAUSED, by that command erroring on an undefined object, in the one
            # measurement that ships no SEG (610892, see parse_eval_log).  Of the 122 OS_LH SEG processing logs 111 carry
            # the entry and 11 carry none, with no mixed case, and those 11 are exactly the measurements whose proclog
            # also lacks D3P_FillOffsetDuplicate -- the second command that errors in the same failed run.
            out["trab_seg_mask"] = "gobj" if mask_after_extract else "none"
            out["evidence"]["trab_gobj_mask_absent_from_seg_proclog"] = not mask_after_extract
            out["evidence"]["cl_extract_index"] = i_ex
            out["evidence"]["gobj_mask_after_extract_index"] = pl.find("D3P_GobjOrAimMaskAimPeel_OW", i_ex) if i_ex >= 0 else -1
            # CORROBORATION FROM THE SAME LOG, added 2026-09-16 by the adversarial review: every
            # D3P_GobjOrAimMaskAimPeel_OW block prints the gobj it masked with ('Gobj File: <disk>:[<dir>]<base>_trab_mask.gobj'
            # for the trabecular crop, '<base>.gobj' for the periosteal mask), so the entry can be identified by NAME and
            # not only by its position after the extraction.  The two readings agree on all 122 OS_LH SEG logs -- the 111
            # with a post-extract entry all name a _trab_mask.gobj, and no _trab_mask.gobj appears anywhere in the 11 that
            # have none -- which closes the only ambiguity the positional rule had (a log whose sole late entry were a
            # periosteal mask; no such log exists here).  The names are RECORDED, and a disagreement is flagged as a
            # warning-worthy note rather than silently deciding, because the positional rule is the one IPL's evaluation
            # log pins for 610892 and the two must not diverge without someone looking.
            out["evidence"]["gobj_mask_files"] = _gobj_mask_files(pl)
            names_trab_mask = any("_trab_mask.gobj" in g for g in out["evidence"]["gobj_mask_files"])
            out["evidence"]["gobj_mask_names_trab_mask"] = names_trab_mask
            if names_trab_mask != mask_after_extract:
                out["notes"].append(
                    f"IPL's SEG processing log disagrees with itself about the trabecular crop: a "
                    f"D3P_GobjOrAimMaskAimPeel_OW after D3P_Cl_ExtractNumber_CPP is "
                    f"{'present' if mask_after_extract else 'absent'} while a '_trab_mask.gobj' is "
                    f"{'named' if names_trab_mask else 'not named'} in {out['evidence']['gobj_mask_files']}; "
                    f"the position is used (trab_seg_mask '{out['trab_seg_mask']}') and this is worth checking by hand")
            if not mask_after_extract:
                out["notes"].append(
                    "IPL's SEG processing log carries no D3P_GobjOrAimMaskAimPeel_OW after D3P_Cl_ExtractNumber_CPP"
                    + (" and no D3P_FillOffsetDuplicate" if not fill else "")
                    + ": Script 32 STEP 2's '/gobj_maskaimpeel_ow -input_output trab_gauss -gobj <base>_trab_mask.gobj' "
                      "did not run, so TRAB_SEG is the raw component extraction (trab_seg_mask 'none')")
        return out
    if seg_vol is not None:
        # a SEG file that carries no processing log at all (none in either cohort).  The evaluation-log branch exists
        # ONLY for a measurement that ships no SEG, so it is not entered here: a measurement WITH a SEG keeps exactly
        # the behaviour it had before that branch existed -- the cohort defaults -- and nothing can change silently.
        for key, dflt in (("lh_border", "duplicate"), ("seg_variant", "periosteal_first"), ("trab_seg_mask", "gobj")):
            if out[key] == "auto":
                out[key] = dflt
        out["source"] = "default"
        out["evidence"] = dict(seg_without_proclog=True)
        out["notes"].append("this measurement's SEG carries no processing log: the cohort defaults are kept and no evaluation log is read")
        return out
    logs = find_eval_logs(folder) if folder else []
    if logs:
        pr, report = choose_eval_log(logs, products=(products() if callable(products) else products))
        # a log in this folder must be a log OF this measurement: its segmentation reads <base>.aim
        grey_read = (pr.get("greyscale_read") or "").lower()
        if base and grey_read and base.lower() not in grey_read:
            out["source"] = "default"
            out["evidence"] = dict(no_seg_proclog=True, eval_log=report,
                                   rejected=f"{pr['file']} segments {pr['greyscale_read']}, not {base}.aim")
            out["notes"].append(f"IPL's evaluation log {pr['file']} is not this measurement's ({pr['greyscale_read']} rather than {base}.aim): "
                                "the cohort defaults are kept")
            for key, dflt in (("lh_border", "duplicate"), ("seg_variant", "periosteal_first"), ("trab_seg_mask", "gobj")):
                if out[key] == "auto":
                    out[key] = dflt
            return out
        out["source"] = "evaluation_log"
        out["evidence"] = dict(no_seg_proclog=True, eval_log=report, **{k: pr["evidence"][k] for k in pr["evidence"]})
        out["evidence"]["eval_log_file"] = pr["file"]
        out["evidence"]["eval_log_path"] = os.path.basename(pr["path"])
        out["evidence"]["job_start"] = pr["job_start"]
        out["evidence"]["job_end"] = pr["job_end"]
        defaults = dict(lh_border="duplicate", seg_variant="periosteal_first", trab_seg_mask="gobj")
        for key in defaults:
            if out[key] != "auto":
                continue                                          # the layout pins it: the log is recorded, not obeyed
            if pr[key] is not None:
                out[key] = pr[key]
            else:
                out[key] = defaults[key]
                out["notes"].append(f"IPL's evaluation log {pr['file']} does not decide {key}: the cohort default {out[key]} is kept")
        out["notes"].append(f"no SEG file: the script variants come from IPL's own evaluation log {pr['file']} ({report['reason']})")
        return out
    if layout["lh_border"] == "auto":
        out["lh_border"] = "duplicate"
    if layout["seg_variant"] == "auto":
        out["seg_variant"] = "periosteal_first"
    if out["trab_seg_mask"] == "auto":
        out["trab_seg_mask"] = "gobj"
    out["source"] = "default"
    out["evidence"] = dict(no_seg_proclog=True, no_evaluation_log=True)
    return out


# ------------------------------------------------------------------------------------------------ SEG assembly
def assemble_seg(bm_ext, grey_grid, periosteal, cort_gobj, trab_gobj, variant, trab_masked=True):
    """Script 32 STEP 2 on the thresholded Laplace-Hamming volume `bm_ext` (bool, on the greyscale grid grown by the
    1-voxel border: pos - 1, dim + 2) with the periosteal raster and the cortical / trabecular contour renderings:
      periosteal_first (2024 script, patella): seg = bm & periosteal, seg_box = its box, cort_seg = cl_nr(seg, 35) &
          cort gobj, trab_seg = cl_nr(seg, 70) & trab gobj, both on seg_box; SEG = bbox(127 cort | 126 trab).
      gobj_first (2022 script): components labelled on the whole thresholded volume (border included), cort_seg =
          bbox(cl_nr(bm, 35) & cort gobj), trab_seg = bbox(cl_nr(bm, 70) & trab gobj), SEG = bbox(add on the union).
    trab_masked=False drops the '& trab gobj' from the trabecular branch, which is what IPL's own log shows happening
    when its '/gobj_maskaimpeel_ow -input_output trab_gauss' errors on an undefined object (Diaphyseal/BMAT/610892's
    second run): TRAB_SEG is then the raw /cl_nr_extract output and SEG's 126 label is everything /cl_nr_extract kept
    that the 127 cortical label did not already claim.  The cortical branch is untouched either way.
    Returns dict(SEG (uint8 127/126 volume), TRAB_SEG, CORT_SEG (bool volumes), seg_box, lh_voxels, overlap, variant,
    trab_seg_mask)."""
    dim_g, pos_g = grey_grid
    dim_e = tuple(d + 2 for d in dim_g)
    pos_e = tuple(p - 1 for p in pos_g)
    bm_ext = np.asarray(bm_ext, bool)
    assert bm_ext.shape == dim_e[::-1], (bm_ext.shape, dim_e)
    out = dict(variant=variant, trab_seg_mask=("gobj" if trab_masked else "none"),
               lh_voxels=int(bm_ext[1:-1, 1:-1, 1:-1].sum()), lh_voxels_with_border=int(bm_ext.sum()))
    per_e = on(periosteal, dim_e, pos_e)
    if variant == "periosteal_first":
        seg0 = bm_ext & per_e
        del per_e
        box = ormir.bbox_cut(seg0, pos_e)
        seg_box = (tuple(box["dim"]), tuple(box["pos"]))
        s = np.ascontiguousarray(box["data"])
        del seg0
        cl35 = ormir.cl_nr_extract(s, ormir.CC_MIN_VOXELS_CORT)
        cl70 = ormir.cl_nr_extract(s, ormir.CC_MIN_VOXELS_TRAB)
        cort_seg = cl35 & on(cort_gobj, *seg_box)
        trab_seg = (cl70 & on(trab_gobj, *seg_box)) if trab_masked else cl70
        del cl35, cl70, s
        seg = np.zeros(cort_seg.shape, np.uint8)
        seg[trab_seg] = ormir.SEG_VALUE_TRAB
        seg[cort_seg] = ormir.SEG_VALUE_CORT
        out["overlap"] = int((cort_seg & trab_seg).sum())
        sb = ormir.bbox_cut(seg, seg_box[1])
        out["SEG"] = dict(data=np.ascontiguousarray(sb["data"]), dim=tuple(sb["dim"]), pos=tuple(sb["pos"]))
        out["TRAB_SEG"] = bvol(trab_seg, *seg_box)
        out["CORT_SEG"] = bvol(cort_seg, *seg_box)
        out["seg_box"] = seg_box
    elif variant == "gobj_first":
        cl35 = ormir.cl_nr_extract(bm_ext, ormir.CC_MIN_VOXELS_CORT)
        cort_seg = cl35 & on(cort_gobj, dim_e, pos_e)
        del cl35
        cl70 = ormir.cl_nr_extract(bm_ext, ormir.CC_MIN_VOXELS_TRAB)
        trab_seg = (cl70 & on(trab_gobj, dim_e, pos_e)) if trab_masked else cl70
        del cl70
        out["overlap"] = int((cort_seg & trab_seg).sum())
        cb = bbox_of(bvol(cort_seg, dim_e, pos_e))
        tb = bbox_of(bvol(trab_seg, dim_e, pos_e))
        cort_v = cut(bvol(cort_seg, dim_e, pos_e), cb) if cb else bvol(np.zeros((1, 1, 1), bool), (1, 1, 1), pos_e)
        trab_v = cut(bvol(trab_seg, dim_e, pos_e), tb) if tb else bvol(np.zeros((1, 1, 1), bool), (1, 1, 1), pos_e)
        # /add_aims on the union grid (127 + 126 saturates at 127 in ipldt.ipl_ops.add_aims; IPL keeps 127 there)
        dim_u, pos_u = ops.union_grid(cort_v, trab_v)
        seg = np.zeros(dim_u[::-1], np.uint8)
        seg[on(trab_v, dim_u, pos_u)] = ormir.SEG_VALUE_TRAB
        seg[on(cort_v, dim_u, pos_u)] = ormir.SEG_VALUE_CORT
        sb = ormir.bbox_cut(seg, pos_u)
        out["SEG"] = dict(data=np.ascontiguousarray(sb["data"]), dim=tuple(sb["dim"]), pos=tuple(sb["pos"]))
        out["TRAB_SEG"] = trab_v
        out["CORT_SEG"] = cort_v
        out["seg_box"] = (tuple(dim_u), tuple(pos_u))
        # the periosteal-masked support, for the record
        out["lh_voxels_in_periosteal"] = int((bm_ext & per_e).sum())
        del per_e
    else:
        raise ValueError(variant)
    return out


# ------------------------------------------------------------------------------------------------ dt maps
def dt_map(name, obj_bool, gobj_bool, backend):
    p = vai.IPL_PARAMS
    if name in ("TRAB_TH_seg", "TRAB_TH_tseg", "CORT_TH"):
        r = ipldt.dt_thickness(obj_bool, gobj={"rendered": gobj_bool}, voxel_size_mm=VOXEL_MM, backend=backend, **p)
    elif name == "TRAB_SP":
        r = ipldt.dt_spacing(obj_bool, gobj={"rendered": gobj_bool}, voxel_size_mm=VOXEL_MM, backend=backend, **p)
    elif name == "TRAB_1N":
        r = ipldt.dt_number(obj_bool, gobj={"rendered": gobj_bool}, voxel_size_mm=VOXEL_MM, backend=backend, **p)
    else:
        raise ValueError(name)
    rep = dict(r.report)
    rep["n_centres"] = int(r.centres.sum())
    return r.map.astype(np.int16), rep


def compute_maps(frame, which, grids, backend, log_prefix="   "):
    """frame: dict(seg, trab_seg, cort, trab_gobj, cort_gobj) bool volumes (any grids); grids: {map: (dim, pos)} the
    grid each dt runs on (object and gobj pasted there by position).  Returns ({map: int16 volume}, {map: report},
    {map: seconds})."""
    maps, reports, timing = {}, {}, {}
    for name in which:
        obj_v = frame.get(MAP_OBJECT[name])
        gobj_v = frame.get(MAP_GOBJ[name] + "_gobj")
        if obj_v is None or gobj_v is None or name not in grids:
            continue
        dim, pos = grids[name]
        t = time.time()
        obj = on(obj_v, dim, pos)
        G = on(gobj_v, dim, pos)
        m, rep = dt_map(name, obj, G, backend)
        del obj, G
        maps[name] = dict(data=m, dim=tuple(dim), pos=tuple(pos))
        reports[name] = rep
        timing[name] = time.time() - t
        log(f"{log_prefix}{name:<12s} on {grid_str((dim, pos))}: support {int((m != 0).sum()):,d}, mean {float(m[m != 0].mean()) if (m != 0).any() else 0.0:.6f} vox, "
            f"centres {rep['n_centres']:,d} [{timing[name]:.1f}s]")
    return maps, reports, timing


def bvtv(seg_v, trab_gobj_v):
    dim, pos = ops.union_grid(seg_v, trab_gobj_v)
    s = on(seg_v, dim, pos)
    G = on(trab_gobj_v, dim, pos)
    bv, tv = int((s & G).sum()), int(G.sum())
    return dict(bv=bv, tv=tv, bvtv=(bv / tv if tv else 0.0))


def metrics_from(map_recs, bv, side, voxel_mm, trab_th_definition="seg"):
    """The metric values of one side from the map comparisons: mean of the map over its non-zero voxels x voxel size
    (Tb.N = 1 / (mean x voxel size)); BV/TV as given.  'TbTh' follows the Tb.Th definition IPL's TRAB_TH map uses
    on this measurement (TRAB_TH_seg or TRAB_TH_tseg); both definitions are also reported separately."""
    out = {}
    if bv is not None:
        out["BVTV"] = bv
    for key, _, _, name in METRICS[1:]:
        if key == "TbTh":
            name = "TRAB_TH_seg" if trab_th_definition == "seg" else "TRAB_TH_tseg"
        if name in map_recs:
            mean_vox = map_recs[name][f"mean_{side}_vox"]
            out[key] = (1.0 / (mean_vox * voxel_mm) if mean_vox else 0.0) if key == "TbN" else mean_vox * voxel_mm
    return out


# ------------------------------------------------------------------------------------------------ per measurement
class Inputs:
    """The volumes of one measurement, read lazily; the greyscale's element sizes are the reference (gobj-derived
    AIMs may carry garbage element sizes)."""

    def __init__(self, meas):
        self.m = meas
        self._cache = {}

    def vol(self, role):
        if role in self._cache:
            return self._cache[role]
        path = self.m.files.get(role)
        if not path:
            self._cache[role] = None
            return None
        v = ipldt.read_aim(path)
        self._cache[role] = v
        return v

    def map(self, name):
        key = f"map:{name}"
        if key in self._cache:
            return self._cache[key]
        path = self.m.map_files.get(name)
        self._cache[key] = ipldt.read_aim(path) if path else None
        return self._cache[key]

    def release(self):
        self._cache.clear()


def build_periosteal(inp, layout):
    """IPL's stage 00: the periosteal rendering file (OS_LH) or CORT_MASK | TRAB_MASK of the raw masks (patella)."""
    if layout["periosteal"]:
        return set_vol(inp.vol("periosteal"))
    cort, trab = inp.vol("raw_cort"), inp.vol("raw_trab")
    return set_vol(vfc.periosteal_from_ipl(cort, trab))


def ipl_renderings(inp, layout, cache_dir, meas):
    """IPL's cortical / trabecular contour renderings: files (OS_LH) or render_volume of the raw masks (patella,
    cached as bool .npy on the mask's own grid).  A cortical file that is an evaluation-written RASTER (581203: periosteal
    rendering minus trab rendering, D3P_Concatenate in its log) is rendered here, since the contour IPL's dt used is
    togobj_from_aim of that raster.  Returns (renderings, notes)."""
    out, notes = {}, []
    for k, role, raw in (("cort", "cort_render", "raw_cort"), ("trab", "trab_render", "raw_trab")):
        if layout.get(role):
            v = inp.vol(role)
            if k == "cort" and "D3P_Concatenate" in (v.get("proclog") or ""):
                G = render_volume(np.asarray(v["data"]) != 0)
                out[k] = bvol(G, v["dim"], v["pos"])
                notes.append(f"IPL's CORT_MASK_CT is an evaluation-written raster (D3P_Concatenate in its log): its ipldt.contour rendering is used as the cortical "
                             f"contour ({int((np.asarray(v['data']) != 0).sum()):,d} raster voxels -> {int(G.sum()):,d} rendered)")
            else:
                out[k] = set_vol(v)
        else:
            mv = inp.vol(raw)
            f = os.path.join(cache_dir, f"{meas.tag}_{meas.base}_render_{k}_own.npy")
            if os.path.exists(f):
                G = np.load(f)
            else:
                os.makedirs(cache_dir, exist_ok=True)
                G = render_volume(np.asarray(mv["data"]) != 0)
                np.save(f, G)
            out[k] = bvol(G, mv["dim"], mv["pos"])
    return out, notes


def cort_object_pm(periosteal, trab_gobj):
    """IPL's STEP 2/3 cortical raster: cort_mask.aim = gobj_to_aim(periosteal) - gobj_to_aim(trab_mask.gobj), on the
    periosteal grid."""
    dim, pos = periosteal["dim"], periosteal["pos"]
    return bvol(np.asarray(periosteal["data"]) & ~on(trab_gobj, dim, pos), dim, pos)


def frame_grids(frame, layout_rule, ipl_maps=None, which=MAP_NAMES, ipl_map_for=None):
    """The compute grid of every map.  Configuration A (`ipl_maps` given): the grid of the IPL file the map is compared
    with (the grid IPL ran the dt on), except that the whole-SEG object maps run on the SEG grid.  Otherwise our grids:
    SEG's box for the SEG-object maps, the tight box of TRAB_SEG for TRAB_TH_tseg under the 'tight' rule (else the SEG
    grid), the cortical object's grid for CORT_TH."""
    grids = {}
    for name in which:
        obj = frame.get(MAP_OBJECT[name])
        if obj is None:
            continue
        if ipl_maps is not None:
            ref = ipl_maps.get((ipl_map_for or {}).get(name, name))
            if MAP_OBJECT[name] == "seg":
                grids[name] = grid_of(frame["seg"])
            elif ref is not None:
                grids[name] = grid_of(ref)
            continue
        if MAP_OBJECT[name] == "seg":
            grids[name] = grid_of(frame["seg"])
        elif MAP_OBJECT[name] == "trab_seg":
            grids[name] = (bbox_of(frame["trab_seg"]) or grid_of(frame["trab_seg"])) if layout_rule == "tight" else grid_of(frame["seg"])
        elif MAP_OBJECT[name] == "cort":
            grids[name] = grid_of(frame["cort"])
    return grids


def map_stage(frame, which, grids, ipl_maps, backend, voxel_mm_el, prefix, ipl_map_for):
    """Run the dt maps of a frame and compare each with the IPL file the layout names for it (ipl_map_for): returns
    dict(maps={name: comparison}, reports, timing)."""
    maps, reports, timing = compute_maps(frame, which, grids, backend, prefix)
    recs = {}
    for name, mv in maps.items():
        ipl_name = ipl_map_for.get(name, name)
        ipl = ipl_maps.get(ipl_name)
        if ipl is None:
            continue
        recs[name] = compare_maps_full(mv, ipl)
        recs[name]["ipl_file"] = ipl_name
        c = recs[name]
        log(f"{prefix}{name:<12s} vs IPL {ipl_name}: {c['mismatches']:,d} mismatches of {c['n_voxels']:,d} (+{c['ours_only']:,d} / -{c['ipl_only']:,d} / both {c['both_nonzero_differ']:,d}); "
            f"support ours {c['support_ours']:,d} IPL {c['support_ipl']:,d}; mean ours {c['mean_ours_vox']:.6f} IPL {c['mean_ipl_vox']:.6f} vox; "
            f"grids {'equal' if c['grids_equal'] else 'DIFFER ' + grid_str(c['grid_ours']) + ' vs ' + grid_str(c['grid_ipl'])}")
    return dict(maps=recs, reports=reports, timing=timing), maps


def finish_metrics(stage, bv_ours, bv_ipl, el_x, tdef="seg"):
    stage["bvtv"] = dict(ours=bv_ours, ipl=bv_ipl)
    stage["trab_th_definition"] = tdef
    stage["metrics_ours"] = metrics_from(stage["maps"], bv_ours["bvtv"] if bv_ours else None, "ours", VOXEL_MM, tdef)
    stage["metrics_ipl"] = metrics_from(stage["maps"], bv_ipl["bvtv"] if bv_ipl else None, "ipl", VOXEL_MM, tdef)
    stage["metrics_ours_el"] = metrics_from(stage["maps"], bv_ours["bvtv"] if bv_ours else None, "ours", el_x, tdef)
    stage["metrics_ipl_el"] = metrics_from(stage["maps"], bv_ipl["bvtv"] if bv_ipl else None, "ipl", el_x, tdef)
    stage["metric_diff"] = {k: stage["metrics_ours"][k] - stage["metrics_ipl"][k] for k in stage["metrics_ours"] if k in stage["metrics_ipl"]}
    return stage


def process_measurement(meas, args, cache_dir, inventory=None):
    L = meas.L
    IMF = L["ipl_map_for"]
    t0 = time.time()
    rec = dict(id=meas.id, tag=meas.tag, group=meas.group, base=meas.base, dataset=meas.layout_name,
               corr=meas.corr, timings={}, notes=[], warnings=[])
    log(f"{meas.id} ({meas.base}) {'CORR ' if meas.corr else ''}=========================================================")
    inp = Inputs(meas)

    # ---- inputs, metadata, hashes
    t = time.time()
    grey = inp.vol("grey")
    meta = meas.parse_grey_meta(grey)
    el = tuple(float(e) for e in grey["el_size_mm"])
    rec["meta"] = meta
    rec["grey"] = dict(grid=grid_of(grey), el_size_mm=list(el))
    rec["inputs"] = {}
    for role, path in list(meas.files.items()) + [(f"map:{k}", v) for k, v in meas.map_files.items()]:
        if not path:
            rec["inputs"][role] = None
            continue
        h = aim_header(path)
        entry = dict(file=os.path.basename(path), size=os.path.getsize(path),
                     type=(f"0x{h['type']:08x}" if h else "v030"), dim=(list(h["dim"]) if h else None), pos=(list(h["pos"]) if h else None))
        if not args.no_hash:
            entry["md5"] = md5_of(path)
        rec["inputs"][role] = entry
    rec["duplicates"] = []
    sizes = {v["size"]: k for k, v in rec["inputs"].items() if v}
    for d in meas.duplicates:
        entry = dict(file=os.path.basename(d), size=os.path.getsize(d))
        if not args.no_hash and entry["size"] in sizes:          # a size twin of an input: hash it to tell a duplicate
            entry["md5"] = md5_of(d)
            twin = [k for k, v in rec["inputs"].items() if v and v.get("md5") == entry["md5"]]
            entry["identical_to"] = twin[0] if twin else None
        rec["duplicates"].append(entry)
        if entry.get("identical_to") or re.search(r"DECOMP|COMPRESS", os.path.basename(d), re.I):
            rec["notes"].append(f"extra file not used: {os.path.basename(d)}" + (f" (byte-identical to {entry['identical_to']})" if entry.get("identical_to") else ""))
    inv_entry = inventory_entry(inventory, meas)
    if inv_entry is not None:                                   # identity whitelist only (no names, dates or paths)
        idn = inv_entry.get("identity") or {}
        rec["inventory"] = dict(identity={k: idn[k] for k in ("study", "subject", "region", "bone", "site", "index_measurement",
                                                          "name_notes") if k in idn},
                                calibration=inv_entry.get("calibration"), plan=dict(status=(inv_entry.get("plan") or {}).get("status")))
    preset = meas.preset_name(args.site_override)
    rec["preset"] = dict(name=preset, params=dict(SITE_PARAMS[preset].__dict__), source=("override" if args.site_override not in (None, "auto") else "layout rule"))
    seg_ipl = inp.vol("seg")
    rec["seg_missing"] = seg_ipl is None
    if seg_ipl is None:
        rec["warnings"].append("no SEG file: configuration A's trabecular maps and every SEG comparison are skipped")
    # the script variants: from IPL's SEG processing log, or -- with no SEG to read -- from IPL's own evaluation log
    # for this measurement.  The delivered maps' processing logs are handed in so that a measurement evaluated twice
    # has its RUN identified from its own products; they are read lazily, so a measurement with a SEG never opens them
    # here (they are read a few lines below anyway, from the same cache).
    var = detect_variants(seg_ipl, L, folder=meas.folder, base=meas.base,
                          products=lambda: {f"map:{n}": (inp.map(n) or {}).get("proclog", "") for n in L["maps"] if meas.map_files.get(n)})
    if args.lh_border != "auto":
        var["lh_border"] = args.lh_border
    var["lh_pad_offset"] = getattr(args, "lh_pad_offset", ormir.LH_PAD_OFFSET)     # probe 21: 'ceil'; 'floor' = the pre-2026-09-17 results
    var["lh_dtype"] = getattr(args, "lh_dtype", ormir.LH_DTYPE)                    # 'float32' = the shipped engine; 'float64' = the opt-in
    # the Laplace-Hamming cut-off IPL actually used, from its own SEG processing log (it prints
    # "fft.lp_cut_off_freq").  116 of the 117 OS_LH measurements use 0.30000; Diaphyseal/CKD/991161's
    # automatic run used 0.20000, and computing it at 0.3 puts ~1.28 M voxels into configuration B.
    _lp = proclog_field(seg_ipl, "fft.lp_cut_off_freq")
    try:
        var["lp_cut_off_freq"] = float(_lp) if _lp is not None else ormir.LP_CUT_OFF_FREQ
    except ValueError:
        var["lp_cut_off_freq"] = ormir.LP_CUT_OFF_FREQ
    if var["lp_cut_off_freq"] != ormir.LP_CUT_OFF_FREQ:
        rec["notes"].append(f"IPL segmented this measurement with fft.lp_cut_off_freq {var['lp_cut_off_freq']:.5f}, "
                            f"not the cohort's {ormir.LP_CUT_OFF_FREQ}; the Laplace-Hamming filter uses its value")
    if args.seg_variant not in ("auto", "best"):
        var["seg_variant"] = args.seg_variant
    if getattr(args, "trab_seg_mask", "auto") != "auto":
        var["trab_seg_mask"] = args.trab_seg_mask
        var.setdefault("notes", []).append(f"--trab-seg-mask {args.trab_seg_mask} overrides the detected value")
    rec["notes"].extend(var.pop("notes", []))
    rec["variants"] = var
    trab_masked = var["trab_seg_mask"] != "none"
    # the periosteal (stage 00) and IPL's renderings
    per = build_periosteal(inp, L)
    ipl_G, notes_G = ipl_renderings(inp, L, cache_dir, meas)
    rec["notes"].extend(notes_G)
    rec["periosteal"] = dict(voxels=count(per), grid=grid_of(per))
    rec["ipl_renderings"] = {k: dict(voxels=count(v), grid=grid_of(v), corr=(meas.corr if k == "trab" else False)) for k, v in ipl_G.items()}
    rec["gobj_timeline"] = gobj_timeline(grey, inp, L)
    if rec["gobj_timeline"].get("trab_gobj_saved_after_step1") and not meas.corr:
        rec["warnings"].append(f"trabecular contour file saved {60 * rec['gobj_timeline']['trab_gobj_minus_step1_h']:.0f} min after ISQ_TO_AIM under the standard name: a manual correction after the automatic run suspected")
    ipl_maps = {name: inp.map(name) for name in L["maps"]}
    ipl_maps = {k: v for k, v in ipl_maps.items() if v is not None}
    rec["ipl_maps"] = {k: dict(grid=grid_of(v), support=int((np.asarray(v["data"]) != 0).sum()), max=int(np.asarray(v["data"]).max()),
                               dtype=str(np.asarray(v["data"]).dtype)) for k, v in ipl_maps.items()}
    # the TRAB_TH grid rule of this measurement: IPL's TRAB_TH grid equal to its SEG grid ('seg') or tighter ('tight')
    rule = L["trab_th_grid"]
    if rule == "auto":
        ref = grid_of(seg_ipl) if seg_ipl is not None else (grid_of(ipl_maps["TRAB_SP"]) if "TRAB_SP" in ipl_maps else None)
        rule = "seg" if ("TRAB_TH" in ipl_maps and ref is not None and grid_of(ipl_maps["TRAB_TH"]) == ref) else "tight"
    rec["trab_th_grid_rule"] = rule
    which_maps = [m for m in MAP_NAMES if IMF.get(m) in L["maps"]]
    rec["timings"]["read"] = time.time() - t
    log(f"  {meta.get('study_label')} | site {meta.get('site_index')} | {meta.get('bone_key')} | preset {preset} | grey {grid_str(rec['grey']['grid'])} el "
        f"{' / '.join(f'{e:.7f}' for e in el)} | periosteal {rec['periosteal']['voxels']:,d} on {grid_str(rec['periosteal']['grid'])} | "
        f"LH border {var['lh_border']}, SEG variant {var['seg_variant']}, TRAB_SEG mask {var['trab_seg_mask']} (from {var['source']}), TRAB_TH grid rule {rule} | "
        f"SEG {'missing' if seg_ipl is None else grid_str(grid_of(seg_ipl))} [{rec['timings']['read']:.1f}s]")
    if var.get("source") == "evaluation_log":
        ev = var["evidence"]
        log(f"  variants read from IPL's evaluation log {ev.get('eval_log_file')} ({ev.get('job_start')} .. {ev.get('job_end')}): {ev['eval_log']['reason']}")
        log(f"    /fill_offset_duplicate line {ev['fill_offset_duplicate'].get('line')}: input {ev['fill_offset_duplicate'].get('input')!r} ok "
            f"{ev['fill_offset_duplicate'].get('ok')} -> LH border {var['lh_border']}" + (f"   [{ev['fill_offset_duplicate'].get('error')}]" if ev['fill_offset_duplicate'].get('error') else ""))
        log(f"    /gobj_maskaimpeel_ow (trab) line {ev['trab_gobj_mask'].get('line')}: input_output {ev['trab_gobj_mask'].get('input_output')!r} ok "
            f"{ev['trab_gobj_mask'].get('ok')} -> TRAB_SEG mask {var['trab_seg_mask']}" + (f"   [{ev['trab_gobj_mask'].get('error')}]" if ev['trab_gobj_mask'].get('error') else ""))
        log(f"    first gobj mask line {ev.get('seg_variant', {}).get('first_gobj_mask_line')} vs first /cl_nr_extract line "
            f"{ev.get('seg_variant', {}).get('first_cl_nr_extract_line')} -> SEG variant {var['seg_variant']}")
    elif var.get("source") == "seg_proclog" and var.get("trab_seg_mask") == "none":
        ev = var["evidence"]
        log(f"  IPL's SEG processing log has D3P_Cl_ExtractNumber_CPP at offset {ev.get('cl_extract_index')} and NO "
            f"D3P_GobjOrAimMaskAimPeel_OW after it ({ev.get('gobj_mask_entries')} in the whole log; D3P_FillOffsetDuplicate "
            f"{ev.get('fill_offset_duplicate')}) -> TRAB_SEG mask none: the trabecular extraction was never cropped to the trabecular contour")
    for k, v in rec["ipl_renderings"].items():
        log(f"  IPL {k} rendering {v['voxels']:,d} on {grid_str(v['grid'])}{' (CORR)' if v['corr'] else ''}")

    # ---- support diagnostics of IPL's maps against IPL's renderings (do the maps live inside the renderings?)
    rec["ipl_support"] = {}
    for name in ipl_maps:
        G = ipl_G["cort" if name == "CORT_TH" else "trab"]
        rec["ipl_support"][name] = dict(outside_ipl_rendering=support_outside(ipl_maps[name], G))
    if seg_ipl is not None:
        for name in ipl_maps:
            if name != "CORT_TH":
                rec["ipl_support"][name]["outside_ipl_seg_grid"] = support_outside(ipl_maps[name], bvol(np.ones(np.asarray(seg_ipl["data"]).shape, bool), seg_ipl["dim"], seg_ipl["pos"]))
    log("  IPL map support outside IPL's own rendering: " + ", ".join(f"{k} {v['outside_ipl_rendering']:,d}" for k, v in rec["ipl_support"].items()))

    # ================================================================================ configuration B: step 1 -- before A, because the
    # cortical object rule of this measurement is decided from IPL's CORT_MASK_CT against BOTH candidates, render(our stage 28)
    # and render(periosteal - our trab rendering), together with render(periosteal - IPL's trab rendering)
    t = time.time()
    log(f"  [B] ipldt step 1 from IPL's periosteal rendering, preset {preset}")

    def step1_info(info, preset):
        return dict(preset=preset, calibration=info["calibration"], thresholds=info["thresholds"], seg_gauss_box=info["box"],
                    counts={k: v for k, v in info["counts"].items() if k != "peel"}, peel_counts=info["counts"]["peel"],
                    grids={k: [list(g[0]), list(g[1])] for k, g in info["grids"].items()}, time_s=info["timings"]["total"])

    def our_renderings(masks):
        """Our trabecular rendering render(29) and both cortical candidates: 'raw_cort' = render(28) (STEP 1's own cortical
        contour: CORT_MASK.GOBJ = togobj_from_aim(cort_mask.aim) of a single-run evaluation) and 'periosteal_minus_trab' =
        render(all - render(29)) (IPL's STEP 2/3 re-derivation cort_mask.aim = gobj_to_aim(periosteal) - gobj_to_aim(trab))."""
        G = {"trab": bvol(render_volume(masks["trab"]["data"]), masks["trab"]["dim"], masks["trab"]["pos"])}
        cands = {}
        if L["cort_object"] in ("periosteal_minus_trab", "auto"):
            cr = cort_object_pm(per, G["trab"])
            cands["periosteal_minus_trab"] = (cr, bvol(render_volume(cr["data"]), cr["dim"], cr["pos"]))
        if L["cort_object"] in ("raw_cort", "auto"):
            cands["raw_cort"] = (masks["cort"], bvol(render_volume(masks["cort"]["data"]), masks["cort"]["dim"], masks["cort"]["pos"]))
        return G, cands

    def trial(pname):
        r = cort_trab_separation(grey, per, SITE_PARAMS[pname], log=lambda s: log("      " + s) if args.verbose_step1 else None)
        masks = {"cort": set_vol(r["cort"]), "trab": set_vol(r["trab"])}
        tr = time.time()
        G, cands = our_renderings(masks)
        comp_c = {rule: vfc.compare_masks(cands[rule][1], ipl_G["cort"]) for rule in cands}
        comp_t = vfc.compare_masks(G["trab"], ipl_G["trab"])
        return dict(preset=pname, info=r["info"], masks=masks, G=G, cands=cands, comp_cort=comp_c, comp_trab=comp_t, render_s=time.time() - tr,
                    score=min(c["mismatches"] for c in comp_c.values()) + comp_t["mismatches"])

    def log_trial(T):
        info = T["info"]
        log(f"     [{T['preset']}] thresholds {info['thresholds']['lower_native']} / {info['thresholds']['upper_native']}; CORT 28 {count(T['masks']['cort']):,d} on "
            f"{grid_str(grid_of(T['masks']['cort']))}; TRAB 29 {count(T['masks']['trab']):,d} on {grid_str(grid_of(T['masks']['trab']))} [{info['timings']['total']:.1f}s]")
        for rule, c in T["comp_cort"].items():
            log(f"     cort rendering [{rule}] ours {c['voxels_ours']:,d} on {grid_str(c['grid_ours'])} | IPL {c['voxels_ipl']:,d} on {grid_str(c['grid_ipl'])} | "
                f"mismatches {c['mismatches']:,d} (+{c['ours_only']:,d} / -{c['ipl_only']:,d}) Dice {c['dice']:.6f}")
        c = T["comp_trab"]
        log(f"     trab rendering ours {c['voxels_ours']:,d} on {grid_str(c['grid_ours'])} | IPL {c['voxels_ipl']:,d} on {grid_str(c['grid_ipl'])} | "
            f"mismatches {c['mismatches']:,d} (+{c['ours_only']:,d} / -{c['ipl_only']:,d}) Dice {c['dice']:.6f}{' [IPL trab is CORR]' if meas.corr else ''}")

    def trial_record(T):
        return dict(cort_mismatches={rule: c["mismatches"] for rule, c in T["comp_cort"].items()}, cort_dice={rule: c["dice"] for rule, c in T["comp_cort"].items()},
                    trab_mismatches=T["comp_trab"]["mismatches"], trab_dice=T["comp_trab"]["dice"], score=T["score"])

    T = trial(preset)
    log_trial(T)
    rec["B"] = dict(step1=step1_info(T["info"], preset), preset_trials={preset: trial_record(T)})
    # preset fallback: an uncorrected measurement whose renderings do not match (with either cortical formula) tries the other preset
    if (not args.no_preset_fallback) and (not meas.corr) and T["score"] > 0 and args.site_override in (None, "auto"):
        other = "radius" if preset == "tibia" else "tibia"
        log(f"     renderings differ with preset {preset} (score {T['score']:,d}): trying {other}")
        T2 = trial(other)
        log_trial(T2)
        rec["B"]["preset_trials"][other] = trial_record(T2)
        # switch only when the other preset is exact or at least halves the mismatch (a few per cent either way is a
        # hand-edited contour, not a preset question: 2422 radius 71,174 vs tibia 70,557)
        if T2["score"] < 0.5 * T["score"]:
            preset, T = other, T2
            rec["B"]["step1"] = step1_info(T["info"], preset)
            rec["preset"].update(name=preset, params=dict(SITE_PARAMS[preset].__dict__), source="fallback (better rendering match)")
            rec["notes"].append(f"preset fallback: {other} matches IPL's renderings better than the layout rule's choice")
        del T2
    rec["timings"]["render_B"] = T["render_s"]
    ours, info = T["masks"], T["info"]
    # was IPL's cortical contour derived from the trabecular contour we have?  render(all - IPL trab) vs CORT_MASK_CT
    tr = time.time()
    c_ipl_raster = cort_object_pm(per, ipl_G["trab"])
    G_c_from_ipl_trab = bvol(render_volume(c_ipl_raster["data"]), c_ipl_raster["dim"], c_ipl_raster["pos"])
    rec["B"]["cort_gobj_from_ipl_trab"] = vfc.compare_masks(G_c_from_ipl_trab, ipl_G["cort"])
    rec["B"]["cort_gobj_from_ipl_trab"]["time_s"] = time.time() - tr
    c = rec["B"]["cort_gobj_from_ipl_trab"]
    e_mism = c["mismatches"]
    log(f"     render(periosteal - IPL's trab rendering) vs IPL's CORT_MASK_CT: mismatches {c['mismatches']:,d} (+{c['ours_only']:,d} / -{c['ipl_only']:,d}) Dice {c['dice']:.6f}"
        f"  [0 = IPL's cortical contour was derived from this trabecular contour]")
    del G_c_from_ipl_trab, c_ipl_raster
    # the cortical object rule of this measurement
    cort_rule, rule_src = L["cort_object"], "layout"
    c_pm = T["comp_cort"].get("periosteal_minus_trab", {}).get("mismatches")
    c_raw = T["comp_cort"].get("raw_cort", {}).get("mismatches")
    if cort_rule == "auto":
        if e_mism == 0:
            cort_rule, rule_src = "periosteal_minus_trab", "render(periosteal - IPL's trab rendering) == CORT_MASK_CT: IPL's cortical contour was re-derived from the trabecular one"
        elif c_raw is not None and c_pm is not None and c_raw < c_pm:
            cort_rule, rule_src = "raw_cort", f"render(our stage 28) vs CORT_MASK_CT {c_raw:,d} < render(periosteal - our trab rendering) {c_pm:,d}; render(periosteal - IPL's trab) {e_mism:,d}"
        else:
            cort_rule, rule_src = "periosteal_minus_trab", (f"render(periosteal - our trab rendering) vs CORT_MASK_CT {c_pm:,d} <= render(our stage 28) {c_raw:,d}; "
                                                            f"render(periosteal - IPL's trab) {e_mism:,d}: neither formula exact")
    cort_raster_ours, G_cort_ours = T["cands"][cort_rule]
    G_ours = {"trab": T["G"]["trab"], "cort": G_cort_ours}
    comp = {"cort": T["comp_cort"][cort_rule], "trab": T["comp_trab"]}
    rec["cort_object_rule"] = dict(value=cort_rule, source=rule_src, candidates={rule: cc["mismatches"] for rule, cc in T["comp_cort"].items()}, from_ipl_trab=e_mism)
    log(f"     Ct.Th object rule: {cort_rule} ({rule_src})")
    if cort_rule == "raw_cort" and comp["cort"]["mismatches"] and not meas.corr:
        rec["warnings"].append(f"Ct.Th object is ipldt's stage 28 whose rendering differs from IPL's CORT_MASK_CT by {comp['cort']['mismatches']:,d} voxels: A's CORT_TH is not like for like")
    rec["B"]["renderings"] = comp
    rec["B"]["cort_rendering_candidates"] = T["comp_cort"]
    rec["B"]["masks"] = {k: dict(voxels=count(ours[k]), grid=grid_of(ours[k])) for k in ("cort", "trab")}
    rec["B"]["cort_object"] = dict(rule=cort_rule, voxels=count(cort_raster_ours), grid=grid_of(cort_raster_ours))
    # the patella layout has IPL's raw masks: compare stage 28 / 29 with them as well
    if L.get("raw_cort"):
        rec["B"]["raw_masks"] = {k: vfc.compare_masks(ours[k], set_vol(inp.vol(f"raw_{k}"))) for k in ("cort", "trab")}
        for k in ("cort", "trab"):
            c = rec["B"]["raw_masks"][k]
            log(f"     raw {k.upper()}_MASK vs IPL's: mismatches {c['mismatches']:,d} (+{c['ours_only']:,d} / -{c['ipl_only']:,d}) Dice {c['dice']:.6f}")
    del T
    rec["timings"]["step1"] = time.time() - t
    renderings_exact = all(comp[k]["mismatches"] == 0 for k in ("cort", "trab"))

    def a_cort_object():
        """The Ct.Th object of configurations A / C: IPL's STEP 2/3 raster (periosteal - IPL's trab rendering), IPL's raw
        CORT_MASK file (patella), or -- 'raw_cort' without a raw file (the OS_LH single-run evaluations) -- our stage 28, which
        stands in for the unexported CORT_MASK.AIM (like for like when its rendering equals IPL's CORT_MASK_CT)."""
        if cort_rule == "periosteal_minus_trab":
            return cort_object_pm(per, ipl_G["trab"])
        if L.get("raw_cort") and inp.vol("raw_cort") is not None:
            return set_vol(inp.vol("raw_cort"))
        return ours["cort"]

    # ================================================================================ configuration A
    A_frame = None
    tdef = L["trab_th_definition"]
    tdef_src = "layout"
    if getattr(args, "trab_th", "auto") != "auto":
        tdef, tdef_src = args.trab_th, ("Scripts 32/33/34: /dt_thickness on TRAB_SEG" if args.trab_th == "trab_seg"
                                        else "--trab-th seg: the whole SEG cropped to the trabecular gobj")
    if "A" in args.configs and not args.skip_dt:
        t = time.time()
        log("  [A] dt on IPL's SEG with IPL's renderings (both Tb.Th definitions)")
        A_frame = dict(trab_gobj=ipl_G["trab"], cort_gobj=ipl_G["cort"])
        if seg_ipl is not None:
            A_frame["seg"] = set_vol(seg_ipl)
            ts = inp.vol("trab_seg")
            if ts is not None:
                A_frame["trab_seg"] = set_vol(ts)
            elif trab_masked:
                # no TRAB_SEG file: SEG & trabecular rendering (differs from IPL's file only at rendering-overlap
                # voxels of 35..69-voxel components, see the module docstring)
                d, p = grid_of(A_frame["seg"])
                A_frame["trab_seg"] = bvol(A_frame["seg"]["data"] & on(ipl_G["trab"], d, p), d, p)
                rec["notes"].append("A: TRAB_SEG reconstructed as SEG & trabecular rendering (no TRAB_SEG file)")
            else:
                # IPL's own log says the '/gobj_maskaimpeel_ow -input_output trab_gauss' never ran for this
                # measurement, so its TRAB_SEG is the RAW component extraction and cropping the delivered SEG to the
                # trabecular contour would model the wrong object.  The closest reconstruction the delivered SEG can
                # give is its own support: SEG = 127 (cortical extraction & cortical contour) | 126 (the raw
                # trabecular extraction minus that 127), so SEG's support is the uncropped TRAB_SEG plus the
                # 35..69-voxel components the cortical label kept -- 13 to 742 voxels here.  TRAB_TH_tseg and
                # TRAB_TH_seg therefore coincide in configuration A by construction for these measurements, and A
                # cannot choose between the two Tb.Th definitions for them; configurations B and C build the two
                # objects separately and can.
                d, p = grid_of(A_frame["seg"])
                A_frame["trab_seg"] = bvol(A_frame["seg"]["data"], d, p)
                rec["notes"].append("A: IPL's own log records that this measurement's TRAB_SEG was never cropped to the trabecular contour, so it is "
                                    "reconstructed as the delivered SEG's support (the raw component extraction plus the 35..69-voxel components the "
                                    "cortical label kept); TRAB_TH_tseg and TRAB_TH_seg coincide in A by construction and A cannot choose between the "
                                    "two Tb.Th definitions for this measurement")
        A_frame["cort"] = a_cort_object()
        if cort_rule == "raw_cort" and not (L.get("raw_cort") and inp.vol("raw_cort") is not None):
            rec["notes"].append("A: Ct.Th object = ipldt's stage 28 (IPL's CORT_MASK.AIM raster is not exported); its rendering "
                                + ("equals" if comp["cort"]["mismatches"] == 0 else f"differs by {comp['cort']['mismatches']:,d} voxels from") + " IPL's CORT_MASK_CT")
        which_A = [m for m in which_maps if (MAP_OBJECT[m] == "cort" or seg_ipl is not None)]
        grids_A = frame_grids(A_frame, rule, ipl_maps=ipl_maps, which=which_A, ipl_map_for=IMF)
        stage, _ = map_stage(A_frame, which_A, grids_A, ipl_maps, args.backend, el[0], "   A ", IMF)
        if tdef == "auto":
            ms = stage["maps"].get("TRAB_TH_seg", {}).get("mismatches")
            mt = stage["maps"].get("TRAB_TH_tseg", {}).get("mismatches")
            if ms is not None and mt is not None:
                # A TIE goes to TRAB_SEG, the definition Scripts 32, 33 and 34 state: the two candidate objects then
                # reproduce IPL's exported map equally well and there is nothing to prefer the other reading on.  It
                # happens exactly where trab_seg_mask is 'none', because the reconstruction above makes the two
                # objects the same volume; no measurement with a cropped TRAB_SEG ties (the closest is 2,458
                # mismatches apart), so the tie-break is inert on the rest of the cohort.
                tdef = "trab_seg" if mt <= ms else "seg"
                tdef_src = (f"A: TRAB_TH_seg {ms:,d} vs TRAB_TH_tseg {mt:,d} mismatches against IPL's TRAB_TH"
                            + ("; equal, so the definition Scripts 32/33/34 state is reported (the two objects coincide "
                               "here because IPL's log says TRAB_SEG was never cropped to the trabecular contour)" if ms == mt else ""))
        bv_ipl = bvtv(A_frame["seg"], ipl_G["trab"]) if seg_ipl is not None else None
        if tdef == "auto":
            tdef, tdef_src = ("trab_seg" if (rule == "tight" or not trab_masked) else "seg"), ("grid rule" if trab_masked else
                             "IPL's log: TRAB_SEG was never cropped to the trabecular contour, so the two definitions name the same object")
        finish_metrics(stage, bv_ipl, bv_ipl, el[0], tdef)
        stage["cort_object"] = dict(rule=L["cort_object"], voxels=count(A_frame["cort"]), grid=grid_of(A_frame["cort"]))
        stage["grids"] = grids_A
        rec["A"] = stage
        rec["timings"]["A"] = time.time() - t
        log(f"  [A] done [{rec['timings']['A']:.1f}s]; IPL's Tb.Th definition: {tdef} ({tdef_src}); metrics ours | IPL: "
            + "  ".join(f"{lab} {stage['metrics_ours'].get(k, float('nan')):.6f} | {stage['metrics_ipl'].get(k, float('nan')):.6f}" for k, lab, _, _ in METRICS if k in stage["metrics_ours"]))
        gpu_release()
    elif "A" in args.configs:
        rec["A"] = dict(skipped="--skip-dt")
    if tdef == "auto":
        tdef, tdef_src = ("trab_seg" if (rule == "tight" or not trab_masked) else "seg"), ("grid rule (A not run)" if trab_masked else
                         "IPL's log: TRAB_SEG was never cropped to the trabecular contour, so the two definitions name the same object (A not run)")
    rec["trab_th_definition"] = dict(value=tdef, source=tdef_src)


    # ================================================================================ Laplace-Hamming threshold
    t = time.time()
    borders = [var["lh_border"]] + ([b for b in ("duplicate", "none", "zero") if b != var["lh_border"]] if args.lh_both else [])
    bm_by_border = {}
    for b in borders:
        tb = time.time()
        bm_by_border[b] = laplace_hamming_threshold(grey["data"], el, border=b, keep_border=True, pad_offset=var["lh_pad_offset"],
                                                    dtype=var["lh_dtype"], lp_cut_off_freq=var["lp_cut_off_freq"])
        log(f"  LH threshold (border {b}, pad offset {var['lh_pad_offset']}, dtype {var['lh_dtype']}, "
            f"cut-off {var['lp_cut_off_freq']:.5f}): "
            f"{int(bm_by_border[b][1:-1, 1:-1, 1:-1].sum()):,d} voxels on the native grid [{time.time() - tb:.1f}s]")
    rec["timings"]["lh"] = time.time() - t
    grey_grid = grid_of(grey)

    def seg_compare(s, label):
        out = dict(lh_voxels=s["lh_voxels"], overlap=s["overlap"], seg_box=s["seg_box"], variant=s["variant"], trab_seg_mask=s["trab_seg_mask"],
                   grid=grid_of(s["SEG"]), voxels=int((s["SEG"]["data"] != 0).sum()), trab_seg_voxels=count(s["TRAB_SEG"]), cort_seg_voxels=count(s["CORT_SEG"]))
        if seg_ipl is not None:
            out["SEG"] = vfc.compare_masks(s["SEG"], seg_ipl)
            out["SEG"]["label_mismatches"] = vfc.compare_labels(s["SEG"], seg_ipl)
            for nm in ("TRAB_SEG", "CORT_SEG"):
                f = inp.vol(nm.lower())
                if f is not None:
                    out[nm] = vfc.compare_masks(s[nm], f)
            if "TRAB_SEG" not in out:
                # no TRAB_SEG file (OS_LH): compare our TRAB_SEG with SEG & IPL's trabecular rendering
                d, p = grid_of(seg_ipl)
                out["TRAB_SEG_vs_seg_and_rendering"] = vfc.compare_masks(s["TRAB_SEG"], bvol((np.asarray(seg_ipl["data"]) != 0) & on(ipl_G["trab"], d, p), d, p))
            c = out["SEG"]
            log(f"     {label} SEG ours {c['voxels_ours']:,d} on {grid_str(c['grid_ours'])} | IPL {c['voxels_ipl']:,d} on {grid_str(c['grid_ipl'])} | mismatches {c['mismatches']:,d} "
                f"(+{c['ours_only']:,d} / -{c['ipl_only']:,d}) labels differ {c['label_mismatches']:,d} Dice {c['dice']:.6f}")
        else:
            log(f"     {label} SEG ours {out['voxels']:,d} on {grid_str(out['grid'])} (no IPL SEG to compare)")
        return out

    # ================================================================================ SEG assembly: B (our renderings) and C (IPL's), both variants
    t = time.time()
    variants = [var["seg_variant"]] + ([v for v in ("periosteal_first", "gobj_first") if v != var["seg_variant"]] if (args.seg_both or args.seg_variant == "best") else [])
    bm0 = bm_by_border[var["lh_border"]]
    log(f"  [B] SEG with OUR renderings (LH border {var['lh_border']}; TRAB_SEG mask {var['trab_seg_mask']}; variants {variants})")
    segB = {v: assemble_seg(bm0, grey_grid, per, G_ours["cort"], G_ours["trab"], v, trab_masked=trab_masked) for v in variants}
    cmpB = {v: seg_compare(segB[v], f"B [{v}]") for v in variants}
    need_C = "C" in args.configs and not (renderings_exact and not meas.corr)
    segC, cmpC = {}, {}
    if need_C:
        log(f"  [C] SEG with IPL's renderings{' (trab CORR)' if meas.corr else ''} (variants {variants})")
        segC = {v: assemble_seg(bm0, grey_grid, per, ipl_G["cort"], ipl_G["trab"], v, trab_masked=trab_masked) for v in variants}
        cmpC = {v: seg_compare(segC[v], f"C [{v}]") for v in variants}
    # the variant used downstream: the fixed or the proclog's one, or (--seg-variant best) the one whose SEG fits IPL's better
    chosen = var["seg_variant"]
    if args.seg_variant == "best" and seg_ipl is not None and len(variants) > 1:
        ref = cmpC if need_C else cmpB
        chosen = min(variants, key=lambda v: (ref[v]["SEG"]["mismatches"], v != var["seg_variant"]))
    rec["seg_variant_used"] = dict(value=chosen, proclog=var["seg_variant"], rule=args.seg_variant,
                                   mismatches_B={v: cmpB[v].get("SEG", {}).get("mismatches") for v in variants},
                                   mismatches_C={v: cmpC[v].get("SEG", {}).get("mismatches") for v in variants} if need_C else None)
    if chosen != var["seg_variant"]:
        rec["notes"].append(f"SEG variant {chosen} fits IPL's SEG better than the proclog's {var['seg_variant']} and is used for the maps")
    log(f"     SEG variant used: {chosen} (proclog {var['seg_variant']})")
    rec["B"]["seg"] = cmpB[chosen]
    rec["B"]["seg_alternatives"] = {f"variant={v},border={var['lh_border']}": cmpB[v] for v in variants if v != chosen}
    if args.lh_both and seg_ipl is not None:
        for b in borders[1:]:
            alt = assemble_seg(bm_by_border[b], grey_grid, per, G_ours["cort"], G_ours["trab"], chosen, trab_masked=trab_masked)
            rec["B"]["seg_alternatives"][f"variant={chosen},border={b}"] = seg_compare(alt, f"B alt (border {b})")
            del alt
    rec["timings"]["seg"] = time.time() - t

    # ================================================================================ cortical pore cascade (Ct.Po)
    # Script 32's Burghardt block (ipldt.porosity) on the cortical segmentation and the rendered cortical
    # contour, in both configurations, against IPL's own <base>_PORE.AIM.  A: IPL's CORT_SEG and IPL's
    # rendering.  B: this run's CORT_SEG (the chosen SEG variant) and our rendering of our cortical mask.
    if not args.skip_porosity:
        t = time.time()
        ipl_pore = inp.vol("pore")
        ipl_cort_seg = inp.vol("cort_seg")
        por = dict(available=ipl_pore is not None,
                   ipl_pore_voxels=(count(ipl_pore) if ipl_pore is not None else None),
                   ipl_pore_grid=(grid_of(ipl_pore) if ipl_pore is not None else None),
                   ipl_cort_seg=bool(ipl_cort_seg is not None))

        # the grid IPL renders its cortical contour on: its own rendering where one is delivered, else the
        # CORT_MASK | TRAB_MASK union box (IPL's gobj box to within a voxel).  Both configurations use it.
        if meas.files.get("cort_render") and ipl_G.get("cort") is not None:
            pore_grid = grid_of(ipl_G["cort"])
            por["contour_grid_source"] = "IPL's own cortical rendering"
        else:
            rc, rt = inp.vol("raw_cort"), inp.vol("raw_trab")
            pore_grid = (ops.union_grid(rc, rt) if rc is not None and rt is not None
                         else (grid_of(rc) if rc is not None else None))
            por["contour_grid_source"] = ("CORT_MASK | TRAB_MASK union box" if rc is not None and rt is not None
                                          else "raw CORT_MASK box")
        por["contour_grid"] = (dict(dim=list(pore_grid[0]), pos=list(pore_grid[1])) if pore_grid else None)

        def _char(v, grid=None):
            """The cascade works on IPL's char convention; our volumes are bool.  `grid` pastes it onto the
            contour grid IPL used, which the slice-wise rule is sensitive to."""
            d, p = grid_of(v)
            c = dict(data=(np.asarray(v["data"]) != 0).astype(np.uint8) * 127, dim=d, pos=p)
            if grid is None:
                return c
            gd, gp = tuple(grid[0]), tuple(grid[1])
            return ops.vol(ops.on_grid(c, gd, gp), gd, gp)

        def _pore_config(label, cort_render, cort_seg_v):
            if cort_render is None or cort_seg_v is None or pore_grid is None:
                return None
            cr = _char(cort_render, pore_grid)
            o = porosity.pore_cascade(cr, _char(cort_seg_v))["pore"]
            e = dict(pore_voxels=count(o), grid=grid_of(o),
                     contour_voxels=int(np.count_nonzero(cr["data"])),
                     ct_po=porosity.ct_po(o, cr)["ct_po"])
            if ipl_pore is not None:
                e.update(porosity.compare_pore(o, ipl_pore))
                e["ct_po_ipl"] = porosity.ct_po(ipl_pore, cr)["ct_po"]
                e["ct_po_diff"] = e["ct_po"] - e["ct_po_ipl"]
                log(f"     [{label}] PORE ours {e['pore_voxels']:,d} | IPL {por['ipl_pore_voxels']:,d} | "
                    f"mismatches {e['mismatch']:,d}  Ct.Po {e['ct_po']:.5f} | {e['ct_po_ipl']:.5f}")
            else:
                log(f"     [{label}] PORE ours {e['pore_voxels']:,d}, Ct.Po {e['ct_po']:.5f} (no IPL PORE to compare)")
            del o, cr
            return e

        try:
            if "A" in args.configs:
                por["A"] = _pore_config("A", ipl_G.get("cort"), ipl_cort_seg)
            if "B" in args.configs:
                por["B"] = _pore_config("B", G_ours.get("cort"), segB[chosen]["CORT_SEG"])
        except Exception as exc:
            por["error"] = f"{type(exc).__name__}: {exc}"
            rec["warnings"].append(f"cortical pore cascade failed: {por['error']}")
            log(f"     porosity FAILED: {por['error']}")
        rec["porosity"] = por
        rec["timings"]["porosity"] = time.time() - t

    # ================================================================================ configuration B: dt
    if not args.skip_dt and "B" in args.configs:
        t = time.time()
        sB = segB[chosen]
        B_frame = dict(seg=set_vol(sB["SEG"]), trab_seg=sB["TRAB_SEG"], cort=cort_raster_ours, trab_gobj=G_ours["trab"], cort_gobj=G_ours["cort"])
        grids_B = frame_grids(B_frame, rule, which=which_maps)
        stage, _ = map_stage(B_frame, which_maps, grids_B, ipl_maps, args.backend, el[0], "   B ", IMF)
        bv_ours = bvtv(B_frame["seg"], G_ours["trab"])
        bv_ipl = bvtv(set_vol(seg_ipl), ipl_G["trab"]) if seg_ipl is not None else None
        finish_metrics(stage, bv_ours, bv_ipl, el[0], tdef)
        stage["grids"] = grids_B
        rec["B"].update(stage)
        rec["timings"]["dt_B"] = time.time() - t
        log(f"  [B] done [{rec['timings']['dt_B']:.1f}s]; metrics ours | IPL: " + "  ".join(f"{lab} {stage['metrics_ours'].get(k, float('nan')):.6f} | {stage['metrics_ipl'].get(k, float('nan')):.6f}" for k, lab, _, _ in METRICS if k in stage["metrics_ours"] and k in stage["metrics_ipl"]))
        del B_frame
        gpu_release()

    # ================================================================================ configuration C
    if "C" in args.configs:
        t = time.time()
        if not need_C:
            rec["C"] = dict(identical_to_B_by_construction=True, note="our cortical and trabecular renderings equal IPL's, so C's SEG and maps equal B's")
            for k in ("seg", "seg_alternatives", "maps", "reports", "timing", "bvtv", "metrics_ours", "metrics_ipl", "metrics_ours_el", "metrics_ipl_el", "metric_diff", "grids", "trab_th_definition"):
                if k in rec["B"]:
                    rec["C"][k] = rec["B"][k]
            log("  [C] renderings identical to IPL's: C = B by construction")
        else:
            rec["C"] = dict(identical_to_B_by_construction=False, seg=cmpC[chosen], seg_alternatives={f"variant={v},border={var['lh_border']}": cmpC[v] for v in variants if v != chosen})
            if not args.skip_dt:
                sC = segC[chosen]
                C_frame = dict(seg=set_vol(sC["SEG"]), trab_seg=sC["TRAB_SEG"], trab_gobj=ipl_G["trab"], cort_gobj=ipl_G["cort"],
                               cort=(A_frame["cort"] if A_frame is not None else a_cort_object()))
                grids_C = frame_grids(C_frame, rule, which=which_maps)
                which_C = list(which_maps)
                if A_frame is not None and "CORT_TH" in rec.get("A", {}).get("maps", {}):
                    which_C = [m for m in which_C if m != "CORT_TH"]          # same cortical object and gobj as A: reuse
                stage, _ = map_stage(C_frame, which_C, grids_C, ipl_maps, args.backend, el[0], "   C ", IMF)
                if "CORT_TH" not in which_C and "CORT_TH" in rec["A"]["maps"]:
                    stage["maps"]["CORT_TH"] = rec["A"]["maps"]["CORT_TH"]
                    stage["reports"]["CORT_TH"] = rec["A"]["reports"]["CORT_TH"]
                    stage["timing"]["CORT_TH"] = 0.0
                    stage["cort_th_reused_from_A"] = True
                bv_ours = bvtv(C_frame["seg"], ipl_G["trab"])
                bv_ipl = bvtv(set_vol(seg_ipl), ipl_G["trab"]) if seg_ipl is not None else None
                finish_metrics(stage, bv_ours, bv_ipl, el[0], tdef)
                stage["grids"] = grids_C
                rec["C"].update(stage)
                log(f"  [C] done; metrics ours | IPL: " + "  ".join(f"{lab} {stage['metrics_ours'].get(k, float('nan')):.6f} | {stage['metrics_ipl'].get(k, float('nan')):.6f}" for k, lab, _, _ in METRICS if k in stage["metrics_ours"] and k in stage["metrics_ipl"]))
                del C_frame
                gpu_release()
        rec["timings"]["C"] = time.time() - t
    del segB, segC

    # ================================================================================ configuration A2 (CORR only)
    if "A2" in args.configs and meas.corr and not args.skip_dt and seg_ipl is not None and not args.no_a2:
        t = time.time()
        log("  [A2] IPL's SEG with OUR (uncorrected) trabecular rendering as the gobj")
        d, p = grid_of(seg_ipl)
        # A2 swaps the trabecular CONTOUR, not the assembly: where IPL's log says TRAB_SEG was never cropped to that
        # contour, the counterfactual contour changes only the gobj the map is cut to, so the object stays the
        # delivered SEG's support -- the same reconstruction configuration A uses for those measurements.
        A2_frame = dict(seg=set_vol(seg_ipl), trab_gobj=G_ours["trab"], cort_gobj=G_ours["cort"], cort=cort_raster_ours,
                        trab_seg=bvol((np.asarray(seg_ipl["data"]) != 0) & (on(G_ours["trab"], d, p) if trab_masked else True), d, p))
        th_name = "TRAB_TH_seg" if tdef == "seg" else "TRAB_TH_tseg"
        which_A2 = [m for m in which_maps if m in ("TRAB_SP", "TRAB_1N", th_name)]
        grids_A2 = frame_grids(A2_frame, rule, ipl_maps=ipl_maps, which=which_A2, ipl_map_for=IMF)
        if rule == "tight" and "TRAB_TH_tseg" in grids_A2:
            grids_A2["TRAB_TH_tseg"] = bbox_of(A2_frame["trab_seg"]) or grids_A2["TRAB_TH_tseg"]
        stage, _ = map_stage(A2_frame, which_A2, grids_A2, ipl_maps, args.backend, el[0], "   A2 ", IMF)
        bv_ours = bvtv(A2_frame["seg"], G_ours["trab"])
        bv_ipl = bvtv(A2_frame["seg"], ipl_G["trab"])
        finish_metrics(stage, bv_ours, bv_ipl, el[0], tdef)
        stage["grids"] = grids_A2
        stage["ipl_support_outside_our_trab_rendering"] = {IMF[name]: support_outside(ipl_maps[IMF[name]], G_ours["trab"]) for name in which_A2 if IMF.get(name) in ipl_maps}
        rec["A2"] = stage
        rec["timings"]["A2"] = time.time() - t
        # verdict: which trabecular contour did IPL's maps use?
        mA = sum(rec["A"]["maps"][m]["mismatches"] for m in which_A2 if m in rec.get("A", {}).get("maps", {}))
        mA2 = sum(stage["maps"][m]["mismatches"] for m in which_A2 if m in stage["maps"])
        outA = sum(v["outside_ipl_rendering"] for k, v in rec["ipl_support"].items() if k != "CORT_TH")
        outA2 = sum(stage["ipl_support_outside_our_trab_rendering"].values())
        verdict = "CORR" if (mA == 0 or (mA < mA2 and outA == 0)) else ("AUTO (uncorrected)" if (mA2 == 0 or (mA2 < mA and outA2 == 0)) else "undecided")
        rec["contour_used_by_ipl_maps"] = dict(verdict=verdict, mismatches_with_corr=mA, mismatches_with_auto=mA2,
                                               support_outside_corr=outA, support_outside_auto=outA2,
                                               cort_gobj_from_corr_mismatches=rec["B"].get("cort_gobj_from_ipl_trab", {}).get("mismatches"))
        log(f"  [A2] done [{rec['timings']['A2']:.1f}s]; trab-map mismatches with CORR {mA:,d} vs AUTO {mA2:,d}; IPL map support outside CORR {outA:,d} / outside AUTO {outA2:,d} -> {verdict}")
        del A2_frame
        gpu_release()
    elif meas.corr:
        rec["contour_used_by_ipl_maps"] = dict(verdict="not tested", reason="A2 skipped")

    del bm_by_border, ours, G_ours, cort_raster_ours, A_frame, per, ipl_G, ipl_maps, grey
    inp.release()
    gpu_release()
    rec["memory"] = mem_info()
    rec["dt_backend"] = ipldt.gpu.resolve_backend(args.backend) if not args.skip_dt else None
    rec["timings"]["total"] = time.time() - t0
    log(f"  {meas.id} total {rec['timings']['total']:.1f}s; peak RSS {rec['memory'].get('peak_rss_gb', float('nan')):.2f} GB")
    return rec


# ------------------------------------------------------------------------------------------------ tables
def _get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _variants_trab_seg_mask(rec):
    """`variants.trab_seg_mask` of a record, with the value a record written BEFORE that field existed carries
    implicitly.  This is not an assumption about the measurement: assemble_seg had a single trabecular branch until
    the field was added, and it cropped TRAB_SEG to the trabecular contour, so any record without the key was
    produced with 'gobj'.  Without this the counters read 'gobj': 0 for a run that reuses such records, which is
    false; re-running a measurement writes the field and the default stops applying to it."""
    return _get(rec, "variants", "trab_seg_mask") or "gobj"


def _variants_source(rec):
    """`variants.source` likewise, read out of the record's own evidence when the key predates the field: the old
    detect_variants took the SEG-processing-log branch whenever a SEG proclog existed and the cohort defaults
    otherwise, and it recorded 'no_seg_proclog' in the evidence in exactly the second case."""
    s = _get(rec, "variants", "source")
    if s:
        return s
    return "default" if _get(rec, "variants", "evidence", "no_seg_proclog") else "seg_proclog"


def _variants_pad_offset(rec):
    """The /fft_laplace_hamming padding offset a record was computed with.  Records written before 2026-09-17 carry
    no field: they were all computed under the 'floor' rule ipldt shipped then (see ormir.lh_pad_plan)."""
    return _get(rec, "variants", "lh_pad_offset") or "floor"


def _variants_lh_dtype(rec):
    """The FFT / filter arithmetic a record's Laplace-Hamming was computed with.  Records written before the dtype
    opt-in (2026-09-17) carry no field: they were all computed in float32, the shipped engine (ormir.LH_DTYPE)."""
    return _get(rec, "variants", "lh_dtype") or "float32"


def flat_row(rec):
    row = dict(id=rec["id"], group=rec.get("group") or "", base=rec["base"], bone=_get(rec, "meta", "bone_key", default=""), corr=rec.get("corr", False),
               excluded=exclusion_reason(rec) or "",
               preset=_get(rec, "preset", "name"), preset_source=_get(rec, "preset", "source"), lh_border=_get(rec, "variants", "lh_border"),
               lh_pad_offset=_variants_pad_offset(rec), lh_dtype=_variants_lh_dtype(rec),
               seg_variant_proclog=_get(rec, "variants", "seg_variant"), seg_variant=_get(rec, "seg_variant_used", "value"),
               trab_seg_mask=_variants_trab_seg_mask(rec), variant_source=_variants_source(rec),
               eval_log=_get(rec, "variants", "evidence", "eval_log_file"),
               trab_th_grid=rec.get("trab_th_grid_rule"), trab_th_def=_get(rec, "trab_th_definition", "value"), seg_missing=rec.get("seg_missing"),
               trab_gobj_after_step1_h=_get(rec, "gobj_timeline", "trab_gobj_minus_step1_h"), hidden_corr=_get(rec, "gobj_timeline", "trab_gobj_saved_after_step1"),
               grey_grid=grid_str(rec["grey"]["grid"]), el_x=rec["grey"]["el_size_mm"][0], el_z=rec["grey"]["el_size_mm"][2],
               periosteal_voxels=_get(rec, "periosteal", "voxels"), thr_lower=_get(rec, "B", "step1", "thresholds", "lower_native"),
               thr_upper=_get(rec, "B", "step1", "thresholds", "upper_native"))
    row["cort_rule"] = _get(rec, "cort_object_rule", "value")
    for rule, key in (("raw_cort", "G_cort_raw_mism"), ("periosteal_minus_trab", "G_cort_pm_mism")):
        row[key] = _get(rec, "cort_object_rule", "candidates", rule)
    for k in ("cort", "trab"):
        c = _get(rec, "B", "renderings", k) or {}
        row.update({f"G_{k}_ours": c.get("voxels_ours"), f"G_{k}_ipl": c.get("voxels_ipl"), f"G_{k}_mism": c.get("mismatches"), f"G_{k}_plus": c.get("ours_only"),
                    f"G_{k}_minus": c.get("ipl_only"), f"G_{k}_dice": c.get("dice"), f"G_{k}_grids_equal": c.get("grids_equal")})
        c = _get(rec, "B", "raw_masks", k)
        if c:
            row.update({f"raw_{k}_mism": c["mismatches"], f"raw_{k}_dice": c["dice"]})
    c = _get(rec, "B", "cort_gobj_from_ipl_trab")
    if c:
        row["cortgobj_from_ipltrab_mism"] = c["mismatches"]
    for cfg in ("B", "C"):
        s = _get(rec, cfg, "seg") or {}
        row.update({f"{cfg}_lh_voxels": s.get("lh_voxels"), f"{cfg}_seg_grid": grid_str(s["grid"]) if s.get("grid") else None})
        c = s.get("SEG") or {}
        row.update({f"{cfg}_SEG_ours": c.get("voxels_ours"), f"{cfg}_SEG_ipl": c.get("voxels_ipl"), f"{cfg}_SEG_mism": c.get("mismatches"), f"{cfg}_SEG_plus": c.get("ours_only"),
                    f"{cfg}_SEG_minus": c.get("ipl_only"), f"{cfg}_SEG_labels": c.get("label_mismatches"), f"{cfg}_SEG_dice": c.get("dice"), f"{cfg}_SEG_grids_equal": c.get("grids_equal")})
        for nm in ("TRAB_SEG", "CORT_SEG"):
            c = s.get(nm)
            if c:
                row.update({f"{cfg}_{nm}_mism": c["mismatches"], f"{cfg}_{nm}_dice": c["dice"]})
    for cfg in CONFIGS:
        st = rec.get(cfg)
        if not st or "maps" not in st:
            continue
        for name in MAP_NAMES:
            c = st["maps"].get(name)
            if c:
                row.update({f"{cfg}_{name}_mism": c["mismatches"], f"{cfg}_{name}_plus": c["ours_only"], f"{cfg}_{name}_minus": c["ipl_only"], f"{cfg}_{name}_both": c["both_nonzero_differ"],
                            f"{cfg}_{name}_voxels": c["n_voxels"], f"{cfg}_{name}_support_ours": c["support_ours"], f"{cfg}_{name}_support_ipl": c["support_ipl"],
                            f"{cfg}_{name}_grids_equal": c["grids_equal"]})
        for k, lab, unit, _ in METRICS:
            if k in st.get("metrics_ours", {}) and k in st.get("metrics_ipl", {}):
                o, i = st["metrics_ours"][k], st["metrics_ipl"][k]
                row.update({f"{cfg}_{k}_ours": o, f"{cfg}_{k}_ipl": i, f"{cfg}_{k}_diff": o - i, f"{cfg}_{k}_rel_pct": (100.0 * (o - i) / i if i else 0.0),
                            f"{cfg}_{k}_ours_el": st["metrics_ours_el"][k], f"{cfg}_{k}_ipl_el": st["metrics_ipl_el"][k]})
    v = rec.get("contour_used_by_ipl_maps")
    if v:
        row.update(maps_used_contour=v.get("verdict"), trabmap_mism_corr=v.get("mismatches_with_corr"), trabmap_mism_auto=v.get("mismatches_with_auto"))
    for cfg in ("B", "C"):
        for key, alt in (_get(rec, cfg, "seg_alternatives") or {}).items():
            row[f"{cfg}_alt[{key}]_SEG_mism"] = (alt.get("SEG") or {}).get("mismatches")
    for name, s in (rec.get("ipl_support") or {}).items():
        row[f"iplsupport_{name}_outside_rendering"] = s.get("outside_ipl_rendering")
    row.update({f"t_{k}": round(v, 1) for k, v in rec["timings"].items()})
    row["peak_rss_gb"] = round(_get(rec, "memory", "peak_rss_gb", default=0.0) or 0.0, 2)
    row["notes"] = "; ".join(rec.get("notes", []) + rec.get("warnings", []))
    return row


def write_table_md(rows, path, have_dt):
    md = vfc._md
    n_excl = sum(1 for r in rows if r.get("excluded"))
    P = ["# ipldt vs Scanco IPL V5.42, three configurations: per-measurement tables", "",
         f"{len(rows)} measurements" + (f", of which {n_excl} excluded from every aggregate in summary.md (column 'excluded', which carries the reason) "
                                        f"and {len(rows) - n_excl} counted; every measurement keeps its row here" if n_excl else "") +
         ".  Every comparison is by global voxel position on the union grid of the two files; '+' = voxels set by ipldt only, "
         "'-' = by IPL only.  Grids as dim @ pos (x, y, z).  CORR = IPL's trabecular contour is the manually corrected one.", "",
         "## Measurements, presets, detected script variants", "",
         md(rows, [("id", "id"), ("base", "base"), ("bone", "bone"), ("corr", "CORR"), ("excluded", "excluded from aggregates (reason)"),
                   ("preset", "preset"), ("preset_source", "preset source"), ("lh_border", "LH border"),
                   ("seg_variant_proclog", "SEG variant (detected)"), ("seg_variant", "SEG variant used"), ("trab_seg_mask", "TRAB_SEG cropped to trab contour"),
                   ("variant_source", "variants read from"), ("eval_log", "IPL evaluation log used"), ("trab_th_grid", "IPL TRAB_TH grid"), ("trab_th_def", "reported Tb.Th definition"),
                   ("cort_rule", "Ct.Th object rule"), ("seg_missing", "SEG missing"), ("trab_gobj_after_step1_h", "trab gobj saved after STEP 1 (h)", "{:+.1f}"), ("hidden_corr", "late correction?"),
                   ("grey_grid", "grey grid"), ("el_x", "el_x", "{:.7f}"), ("thr_lower", "seg_gauss lower"), ("thr_upper", "upper"), ("notes", "notes")]), "",
         "## B: our renderings (from ipldt step 1) vs IPL's renderings", "",
         md(rows, [("id", "id"), ("G_cort_ours", "cort ours"), ("G_cort_ipl", "cort IPL"), ("G_cort_mism", "mism."), ("G_cort_plus", "+"), ("G_cort_minus", "-"), ("G_cort_dice", "Dice"),
                   ("G_trab_ours", "trab ours"), ("G_trab_ipl", "trab IPL"), ("G_trab_mism", "mism."), ("G_trab_plus", "+"), ("G_trab_minus", "-"), ("G_trab_dice", "Dice"),
                   ("corr", "CORR"), ("cort_rule", "Ct.Th object rule"), ("cortgobj_from_ipltrab_mism", "render(all - IPL trab) vs CORT_MASK_CT mism."),
                   ("G_cort_raw_mism", "render(our 28) vs CORT_MASK_CT mism."), ("G_cort_pm_mism", "render(all - our trab) mism.")]), "",
         "## SEG vs IPL's SEG: B (our renderings) and C (IPL's renderings)", "",
         md(rows, [("id", "id"), ("B_lh_voxels", "LH voxels"), ("B_SEG_ours", "B SEG ours"), ("B_SEG_ipl", "IPL"), ("B_SEG_mism", "mism."), ("B_SEG_plus", "+"), ("B_SEG_minus", "-"),
                   ("B_SEG_labels", "labels"), ("B_SEG_dice", "Dice"), ("C_SEG_ours", "C SEG ours"), ("C_SEG_mism", "mism."), ("C_SEG_plus", "+"), ("C_SEG_minus", "-"), ("C_SEG_labels", "labels"),
                   ("C_SEG_dice", "Dice")]
            + [(k, k.replace("_alt[", " alt[")) for k in sorted({k for r in rows for k in r if "_alt[" in k})]), ""]
    if have_dt:
        P += ["Tb.Th definitions: TRAB_TH_seg = dt_thickness(whole SEG) -- what the shipped ORMIR-BQRL pipeline computes -- and TRAB_TH_tseg = dt_thickness(TRAB_SEG) -- what "
              "Scripts 32, 33 and 34 compute (/dt_thickness on IPL_FNAME5 = TRAB_SEG, cropped to the trabecular gobj).  Both are computed for every measurement and both are "
              "reported below; each is compared with the IPL file the layout names (OS_LH: IPL's single TRAB_TH; patella: TRAB_TH_old is the TRAB_SEG map and TRAB_TH the whole-SEG one).  "
              "The reported Tb.Th follows --trab-th (column 'reported Tb.Th definition'), which defaults to 'auto': each measurement is compared against the definition its own "
              "exported IPL file used.  Since the failed trabecular crop is modelled (variants.trab_seg_mask 'none', read from IPL's own logs) 'auto' selects TRAB_SEG for every "
              "OS_LH measurement -- on the eleven whose crop failed the two candidate objects coincide, so the choice is no longer a live one and the field records provenance.  "
              "--trab-th seg forces the whole-SEG reading everywhere.", ""]
        for cfg, title in (("A", "A: dt on IPL's SEG + IPL's renderings"), ("B", "B: ipldt from IPL's periosteal contour, our renderings"),
                           ("C", "C: our LH SEG with IPL's renderings"), ("A2", "A2: IPL's SEG with our uncorrected trabecular rendering (CORR only)")):
            cols = [("id", "id")] + [c for name in MAP_NAMES if any(f"{cfg}_{name}_mism" in r for r in rows) for c in
                                     ((f"{cfg}_{name}_mism", f"{name} mism."), (f"{cfg}_{name}_plus", "+"), (f"{cfg}_{name}_minus", "-"), (f"{cfg}_{name}_both", "both>0"),
                                      (f"{cfg}_{name}_support_ours", "support ours"), (f"{cfg}_{name}_support_ipl", "IPL"), (f"{cfg}_{name}_grids_equal", "grids ="))]
            if len(cols) > 1:
                P += [f"## {title}: maps vs IPL's maps", "", md(rows, cols), ""]
                P += [f"## {title}: metrics (voxel size {VOXEL_MM} mm; diff = ours - IPL, rel = diff / IPL)", "",
                      md(rows, [("id", "id")] + [c for k, lab, unit, _ in METRICS if any(f"{cfg}_{k}_ours" in r for r in rows) for c in
                                                 ((f"{cfg}_{k}_ours", f"{lab} ours"), (f"{cfg}_{k}_ipl", "IPL"), (f"{cfg}_{k}_diff", "diff", "{:+.3e}"), (f"{cfg}_{k}_rel_pct", "rel %", "{:+.5f}"))]), "",
                      f"## {title}: metrics at the header's x element size", "",
                      md(rows, [("id", "id"), ("el_x", "el_x", "{:.7f}")] + [c for k, lab, unit, _ in METRICS if any(f"{cfg}_{k}_ours_el" in r for r in rows) for c in
                                                                              ((f"{cfg}_{k}_ours_el", f"{lab} ours"), (f"{cfg}_{k}_ipl_el", "IPL"))]), ""]
        if any("maps_used_contour" in r for r in rows):
            P += ["## Which trabecular contour did IPL's maps use (CORR measurements)", "",
                  md(rows, [("id", "id"), ("corr", "CORR"), ("maps_used_contour", "verdict"), ("trabmap_mism_corr", "trab-map mism. with CORR (A)"), ("trabmap_mism_auto", "with AUTO (A2)"),
                            ("cortgobj_from_ipltrab_mism", "render(all - CORR) vs CORT_MASK_CT")] + [(k, k.replace("iplsupport_", "IPL ")) for k in sorted({k for r in rows for k in r if k.startswith("iplsupport_")})]), ""]
    P += ["## Timing (s) and memory", "", md(rows, [("id", "id"), ("t_read", "read"), ("t_A", "A"), ("t_step1", "step 1 (+renderings)"), ("t_lh", "LH"), ("t_seg", "SEG B+C"), ("t_dt_B", "dt B"),
                                                    ("t_C", "C"), ("t_A2", "A2"), ("t_total", "total"), ("peak_rss_gb", "peak RSS (GB)")]), ""]
    open(path, "w", encoding="utf-8").write("\n".join(P))


# ------------------------------------------------------------------------------------------------ summary
def _stats(vals):
    a = np.array([v for v in vals if v is not None], float)
    if a.size == 0:
        return dict(n=0)
    return dict(n=int(a.size), min=float(a.min()), max=float(a.max()), mean=float(a.mean()), sum=float(a.sum()))


def group_summary(records, rows, key_fn, label, have_dt):
    """Totals over a set of records: exact counts per stage, mismatch totals, Dice ranges, metric differences and
    sample-wise regressions per metric and configuration.

    Everything here is an AGGREGATE, so the caller hands in the KEPT records only: write_summary applies
    split_excluded once, at the top, and carves every block -- ALL, per group, per bone, CORR / uncorrected -- out of
    that same set, which is what keeps the denominators of the counters, the stages and the metrics consistent."""
    out = dict(label=label, n=len(records), corr=sum(1 for r in records if r.get("corr")),
               trab_th_definitions={d: sum(1 for r in records if _get(r, "trab_th_definition", "value") == d) for d in ("seg", "trab_seg")},
               seg_variants_used={v: sum(1 for r in records if _get(r, "seg_variant_used", "value") == v) for v in ("periosteal_first", "gobj_first")},
               lh_borders={b: sum(1 for r in records if _get(r, "variants", "lh_border") == b) for b in ("duplicate", "none", "zero")},
               lh_pad_offsets={o: sum(1 for r in records if _variants_pad_offset(r) == o) for o in ormir.LH_PAD_OFFSETS},
               lh_dtypes={d: sum(1 for r in records if _variants_lh_dtype(r) == d) for d in ormir.LH_DTYPES},
               trab_seg_masks={m: sum(1 for r in records if _variants_trab_seg_mask(r) == m) for m in ("gobj", "none")},
               variant_sources={s: sum(1 for r in records if _variants_source(r) == s) for s in ("seg_proclog", "evaluation_log", "default")},
               presets={pn: sum(1 for r in records if _get(r, "preset", "name") == pn) for pn in ("tibia", "radius")},
               cort_rules={cr: sum(1 for r in records if _get(r, "cort_object_rule", "value") == cr) for cr in ("periosteal_minus_trab", "raw_cort")},
               hidden_corrections=sum(1 for r in records if _get(r, "gobj_timeline", "trab_gobj_saved_after_step1") and not r.get("corr")))
    stages = []
    for k in ("cort", "trab"):
        cs = [r["B"]["renderings"][k] for r in records if _get(r, "B", "renderings", k)]
        if cs:
            stages.append(dict(stage=f"{k} rendering (B vs IPL)", exact=sum(c["mismatches"] == 0 for c in cs), n=len(cs), mismatches=sum(c["mismatches"] for c in cs),
                               plus=sum(c["ours_only"] for c in cs), minus=sum(c["ipl_only"] for c in cs), min_dice=min(c["dice"] for c in cs), max_dice=max(c["dice"] for c in cs),
                               grids_equal=sum(c["grids_equal"] for c in cs)))
    cs = [r["B"]["cort_gobj_from_ipl_trab"] for r in records if _get(r, "B", "cort_gobj_from_ipl_trab")]
    if cs:
        stages.append(dict(stage="render(all - IPL trab) vs CORT_MASK_CT", exact=sum(c["mismatches"] == 0 for c in cs), n=len(cs), mismatches=sum(c["mismatches"] for c in cs),
                           plus=sum(c["ours_only"] for c in cs), minus=sum(c["ipl_only"] for c in cs), min_dice=min(c["dice"] for c in cs), max_dice=max(c["dice"] for c in cs),
                           grids_equal=sum(c["grids_equal"] for c in cs)))
    for cfg in ("B", "C"):
        cs = [r[cfg]["seg"]["SEG"] for r in records if _get(r, cfg, "seg", "SEG")]
        if cs:
            stages.append(dict(stage=f"SEG ({cfg})", exact=sum(c["mismatches"] == 0 for c in cs), n=len(cs), mismatches=sum(c["mismatches"] for c in cs),
                               plus=sum(c["ours_only"] for c in cs), minus=sum(c["ipl_only"] for c in cs), min_dice=min(c["dice"] for c in cs), max_dice=max(c["dice"] for c in cs),
                               grids_equal=sum(c["grids_equal"] for c in cs), labels=sum(c.get("label_mismatches", 0) for c in cs)))
    if have_dt:
        for cfg in CONFIGS:
            for name in MAP_NAMES:
                cs = [r[cfg]["maps"][name] for r in records if _get(r, cfg, "maps", name)]
                if cs:
                    stages.append(dict(stage=f"{name} map ({cfg})", exact=sum(c["mismatches"] == 0 for c in cs), n=len(cs), mismatches=sum(c["mismatches"] for c in cs),
                                       plus=sum(c["ours_only"] for c in cs), minus=sum(c["ipl_only"] for c in cs), min_dice=None, max_dice=None,
                                       grids_equal=sum(c["grids_equal"] for c in cs)))
    out["stages"] = stages
    metrics = []
    if have_dt:
        for cfg in CONFIGS:
            for k, lab, unit, _ in METRICS:
                pairs = [(r["id"], r[cfg]["metrics_ipl"][k], r[cfg]["metrics_ours"][k]) for r in records
                         if _get(r, cfg, "metrics_ours") and k in r[cfg]["metrics_ours"] and k in r[cfg].get("metrics_ipl", {})]
                if not pairs:
                    continue
                i = np.array([p[1] for p in pairs])
                o = np.array([p[2] for p in pairs])
                d = o - i
                rel = 100.0 * d / np.where(i != 0, i, 1.0)
                reg = vai.ols(i, o) if len(pairs) > 1 else dict(slope=float("nan"), intercept=float("nan"), r2=float("nan"), n=len(pairs))
                icc = vai.icc_2_1(i, o) if len(pairs) > 1 else float("nan")
                metrics.append(dict(config=cfg, metric=lab, key=k, unit=unit, n=len(pairs), mean_ipl=float(i.mean()), mean_ours=float(o.mean()), mean_diff=float(d.mean()),
                                    mean_abs=float(np.abs(d).mean()), max_abs=float(np.abs(d).max()), mean_rel=float(np.abs(rel).mean()), max_rel=float(np.abs(rel).max()),
                                    exact=int((d == 0).sum()), slope=reg["slope"], intercept=reg["intercept"], r2=reg["r2"], icc=icc,
                                    pairs=[[p[0], float(p[1]), float(p[2])] for p in pairs]))
    out["metrics"] = metrics
    return out


def pooled_hists(records, have_dt):
    """{config: {map: pooled sparse joint histogram}} for the voxel-wise regression plots.

    Pooling is an aggregate: the caller passes the KEPT records (regression_data.json's per_measurement section still
    carries every measurement, excluded ones included)."""
    out = {}
    if not have_dt:
        return out
    for cfg in CONFIGS:
        for name in MAP_NAMES:
            P = {}
            for r in records:
                c = _get(r, cfg, "maps", name)
                if not c:
                    continue
                for o, i, n in c["hist"]:
                    P[(o, i)] = P.get((o, i), 0) + n
            if P:
                out.setdefault(cfg, {})[name] = [[o, i, n] for (o, i), n in sorted(P.items())]
    return out


def write_summary(records, rows, path, args, have_dt, layout_name):
    md = vfc._md
    # THE aggregate set.  Every number below -- every block, every exact count, every Dice range, every regression --
    # is over 'kept'; 'records' / 'rows' (all of them) are used only for the per-measurement listings, so that an
    # excluded measurement is still visible with its reason instead of disappearing.
    include_excluded = bool(getattr(args, "include_excluded", False))
    kept, kept_rows, excl = split_excluded(records, rows, include_excluded)
    n = len(kept)
    groups = sorted({r.get("group") or "(all)" for r in kept})
    bones = sorted({_get(r, "meta", "bone_key", default="?") for r in kept})
    blocks = [group_summary(kept, kept_rows, None, "ALL", have_dt)]
    for g in groups:
        sel = [r for r in kept if (r.get("group") or "(all)") == g]
        if len(groups) > 1:
            blocks.append(group_summary(sel, kept_rows, None, f"group {g}", have_dt))
    for b in bones:
        sel = [r for r in kept if _get(r, "meta", "bone_key", default="?") == b]
        if len(bones) > 1:
            blocks.append(group_summary(sel, kept_rows, None, f"bone {b}", have_dt))
    for flag, lab in ((True, "CORR (manually corrected trabecular contour)"), (False, "uncorrected")):
        sel = [r for r in kept if bool(r.get("corr")) == flag]
        if sel and len(sel) < n:
            blocks.append(group_summary(sel, kept_rows, None, lab, have_dt))
    presets = sorted({_get(r, "preset", "name") for r in kept})
    variants = sorted({f"{_get(r, 'variants', 'lh_border')} / {_get(r, 'variants', 'seg_variant')} / TRAB_SEG mask {_variants_trab_seg_mask(r)} / "
                       f"TRAB_TH grid {r.get('trab_th_grid_rule')}" for r in kept})
    L = [f"# ipldt vs Scanco IPL V5.42 on the {layout_name} dataset: summary of three configurations", "",
         f"{n} measurements ({sum(1 for r in kept if r.get('corr'))} CORR); groups {groups}; bones {bones}; presets {presets}; detected script variants (LH border / SEG assembly / TRAB_SEG mask / TRAB_TH grid) {variants}; "
         f"dt parameters {vai.IPL_PARAMS}; voxel size {VOXEL_MM} mm (metrics also at the header's x element size in table.csv); backend {(kept or records)[0].get('dt_backend')}; ipldt {ipldt.__version__}; "
         f"{time.strftime('%Y-%m-%d %H:%M')}.", ""]
    if excl:
        L += [f"**{len(excl)} of the {len(records)} measurements processed {'is' if len(excl) == 1 else 'are'} EXCLUDED from every number in this file** "
              f"(every exact count, mismatch and voxel total, Dice range, metric difference and regression below is over the remaining {n}); "
              f"{'it keeps its' if len(excl) == 1 else 'they keep their'} row in table.csv / table.md (column 'excluded') and in the per-measurement "
              f"listings at the end of this file, and --include-excluded reproduces the numbers with {'it' if len(excl) == 1 else 'them'} counted. "
              f"{'The exclusion is' if len(excl) == 1 else 'The exclusions are'} on a property of the measurement's own FILES, never on its agreement with ipldt:", ""]
        for r in excl:
            L.append(f"- **{r['id']}** ({r['base']}): {exclusion_reason(r)}.")
        L.append("")
    elif include_excluded:
        L += [f"--include-excluded: every processed measurement is counted below, including the {sum(1 for r in records if exclusion_reason(r))} that "
              f"EXCLUDED_IDS normally removes from the aggregates ({', '.join(r['id'] for r in records if exclusion_reason(r)) or 'none in this run'}).", ""]
    L += ["Configurations: A = dt on IPL's SEG with IPL's contour renderings (maps expected exact when these renderings are the ones the maps used); "
          "B = ipldt step 1 from IPL's periosteal rendering -> our renderings -> our Laplace-Hamming SEG (our renderings) -> our maps; "
          "C = our LH SEG assembled with IPL's renderings (like-for-like for the CORR cases); A2 = IPL's SEG with our uncorrected trabecular rendering (CORR only).", ""]
    for blk in blocks:
        L += [f"## {blk['label']}: {blk['n']} measurements ({blk['corr']} CORR)", "",
              f"Reported Tb.Th definition: {blk['trab_th_definitions']}; SEG variant used: {blk['seg_variants_used']}; LH border: {blk['lh_borders']}; "
              f"LH pad offset: {blk['lh_pad_offsets']}; LH dtype: {blk['lh_dtypes']}; "
              f"TRAB_SEG cropped to the trabecular contour: {blk['trab_seg_masks']}; variants read from: {blk['variant_sources']}; presets: {blk['presets']}; Ct.Th object rule: {blk['cort_rules']}; "
              f"uncorrected measurements whose trabecular contour file was saved > 15 min after ISQ_TO_AIM (a manual correction after the automatic run suspected): {blk['hidden_corrections']}.", "",
              "### Exactness per stage", "",
              md(blk["stages"], [("stage", "stage"), ("exact", "exact"), ("n", "of"), ("mismatches", "mismatching voxels"), ("plus", "ipldt only"), ("minus", "IPL only"),
                                 ("min_dice", "min Dice"), ("max_dice", "max Dice"), ("grids_equal", "identical grids"), ("labels", "SEG labels differ")]), ""]
        if blk["metrics"]:
            L += ["### Metrics (ours - IPL) and sample-wise regressions (ours on IPL)", "",
                  md(blk["metrics"], [("config", "cfg"), ("metric", "metric"), ("unit", "unit"), ("n", "n"), ("mean_ipl", "IPL mean"), ("mean_ours", "ipldt mean"), ("mean_diff", "mean diff", "{:+.3e}"),
                                      ("mean_abs", "mean |diff|", "{:.3e}"), ("max_abs", "max |diff|", "{:.3e}"), ("mean_rel", "mean |rel| %", "{:.5f}"), ("max_rel", "max |rel| %", "{:.5f}"),
                                      ("exact", "exact"), ("slope", "slope", "{:.6f}"), ("intercept", "intercept", "{:+.3e}"), ("r2", "R^2", "{:.6f}"), ("icc", "ICC(2,1)", "{:.6f}")]), ""]
    # The listings below are deliberately over EVERY processed measurement, excluded ones included and flagged: they
    # are inventories, not statistics, and an excluded measurement must never vanish from the report.
    verdicts = [(r["id"], exclusion_reason(r), r["contour_used_by_ipl_maps"]) for r in records if r.get("contour_used_by_ipl_maps")]
    if verdicts:
        L += ["## Which trabecular contour did IPL's maps use (CORR measurements)", "", "| id | verdict | trab-map mismatches with CORR (A) | with AUTO (A2) | IPL map support outside CORR | outside AUTO | render(all - CORR) vs CORT_MASK_CT |",
              "|---|---|---|---|---|---|---|"]
        for i, why, v in verdicts:
            tag = " (EXCLUDED from the aggregates)" if why and not include_excluded else ""
            L.append(f"| {i}{tag} | {v.get('verdict')} | {v.get('mismatches_with_corr', '-')} | {v.get('mismatches_with_auto', '-')} | {v.get('support_outside_corr', '-')} | {v.get('support_outside_auto', '-')} | {v.get('cort_gobj_from_corr_mismatches', '-')} |")
        L.append("")
    # 'excluded' holds the full reason in table.csv / table.md; here only a marker, so the wide table stays readable.
    ov_rows = [dict(r, excluded=("EXCLUDED" if r.get("excluded") else "")) for r in rows]
    L += ["## Per-measurement overview", "",
          f"All {len(rows)} processed measurements" + (f", including the {len(excl)} excluded from every aggregate above "
                                                       f"(column 'excluded'; the reason is at the top of this file and in table.csv)" if excl else "") + ".", "",
          md(ov_rows, [("id", "id"), ("bone", "bone"), ("corr", "CORR"), ("excluded", "excluded"), ("preset", "preset"), ("cort_rule", "Ct.Th rule"), ("lh_border", "LH border"), ("seg_variant", "SEG variant"), ("trab_th_def", "Tb.Th def."),
                    ("G_cort_mism", "cort rend. mism."), ("G_trab_mism", "trab rend. mism."),
                    ("B_SEG_mism", "B SEG mism."), ("C_SEG_mism", "C SEG mism.")] + ([(f"{cfg}_{name}_mism", f"{cfg} {name}") for cfg in ("A", "B", "C") for name in MAP_NAMES if any(f"{cfg}_{name}_mism" in r for r in rows)] if have_dt else [])
             + [("t_total", "time (s)", "{:.1f}"), ("peak_rss_gb", "peak RSS GB")]), ""]
    issues = []
    for r, row in zip(records, rows):
        bits = [f"{k} rendering {r['B']['renderings'][k]['mismatches']:,d}" for k in ("cort", "trab") if _get(r, "B", "renderings", k, "mismatches")]
        for cfg in ("B", "C"):
            c = _get(r, cfg, "seg", "SEG")
            if c and c["mismatches"]:
                bits.append(f"SEG {cfg} {c['mismatches']:,d} (+{c['ours_only']:,d}/-{c['ipl_only']:,d})")
        if have_dt:
            for cfg in CONFIGS:
                for m in MAP_NAMES:
                    c = _get(r, cfg, "maps", m)
                    if c and c["mismatches"]:
                        bits.append(f"{cfg} {m} {c['mismatches']:,d}")
        for w in r.get("warnings", []):
            bits.append(w)
        if bits:
            excl_tag = ", EXCLUDED from the aggregates" if exclusion_reason(r) and not include_excluded else ""
            issues.append(f"- {r['id']} ({r['base']}{', CORR' if r.get('corr') else ''}{excl_tag}): " + ", ".join(bits))
    L += ["## Measurements not exact at some stage", "",
          "Every processed measurement, excluded ones included and marked (a listing, not a statistic).", ""] + (issues or ["- none"]) + [""]
    notes = [f"- {r['id']}: " + "; ".join(r["notes"]) for r in records if r.get("notes")]
    if notes:
        L += ["## Notes per measurement", ""] + notes + [""]
    open(path, "w", encoding="utf-8").write("\n".join(L))
    return blocks


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="oslh", choices=sorted(LAYOUTS))
    ap.add_argument("--root", default=None, help="dataset root (default: the layout's)")
    ap.add_argument("--results", default=None, help="results folder (default validation/results/<dataset>/validate/; a partial run goes to a subset_ subfolder of it)")
    ap.add_argument("--subjects", nargs="*", default=None, help="measurement ids (Distal/CKD/345857, Distal_CKD_345857, 345857, X1496677; patella: PFJ-0be66a_R)")
    ap.add_argument("--groups", nargs="*", default=None, help="layout groups to include (OS_LH: Distal/CKD Distal/REPRO Diaphyseal/BMAT Diaphyseal/CKD Diaphyseal/REPRO rerun)")
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--skip-dt", action="store_true", help="masks, renderings and SEGs only (no maps / metrics)")
    ap.add_argument("--resume", action="store_true", help="skip measurements whose records/<id>.json exists")
    ap.add_argument("--include-excluded", action="store_true",
                    help="count the EXCLUDED_IDS measurements in the aggregates too, restoring the pre-exclusion numbers. "
                         "By default a measurement listed in EXCLUDED_IDS is dropped from every aggregate in summary.md and "
                         "from the tables' totals -- exact counts, mismatch and voxel totals, Dice ranges, metric differences, "
                         "sample-wise regressions and every per-group / per-cohort / per-flag breakdown -- because its own "
                         "files were not produced by the standard pipeline (Diaphyseal/CKD/991161: its delivered trabecular contour is "
                         "not the one its evaluation used; see EXCLUDED_IDS), never because it disagrees "
                         "with ipldt. It keeps its row in table.csv / table.md (column 'excluded', with the reason), its record "
                         "and its per-measurement entry in regression_data.json either way, and summary.md names it at the top.")
    ap.add_argument("--site-override", default="auto", choices=["auto", "tibia", "radius"], help="force the Step1Params preset")
    ap.add_argument("--no-preset-fallback", action="store_true", help="do not try the other preset when the cortical rendering of an uncorrected measurement differs")
    ap.add_argument("--configs", default="A,B,C,A2", help="comma-separated subset of A,B,C,A2 (B always runs up to the SEG)")
    ap.add_argument("--allow-overwrite-shipped", action="store_true",
                    help="write into (or under) one of the published result directories under validation/results/ (the "
                         "records the paper's numbers are built from); refused otherwise, exit status 2")
    ap.add_argument("--trab-th", default="auto", choices=["trab_seg", "seg", "auto"],
                    help="which object the REPORTED Tb.Th is computed on for the comparison against IPL. 'auto' (the default) "
                         "decides per measurement, from configuration A, which of the two definitions reproduces the Tb.Th map "
                         "IPL actually exported for that measurement, and compares like with like against it. With the failed "
                         "trabecular crop modelled (trab_seg_mask 'none') the two definitions coincide on the measurements that "
                         "had selected the whole SEG, so this is no longer a live per-measurement choice; it is kept for provenance. "
                         "'trab_seg' forces the definition Scripts 32, 33 and 34 use: /dt_thickness reads IPL_FNAME5 = "
                         "TRAB_SEG, cropped to the trabecular gobj. 'seg' forces the whole SEG cropped to the trabecular gobj, "
                         "which is what the shipped ORMIR-BQRL pipeline computes. Both definitions are always computed and "
                         "stored in the record either way; this flag selects which one the reported Tb.Th and the tables use.")
    ap.add_argument("--lh-border", default="auto", choices=["auto", "duplicate", "none", "zero"], help="Laplace-Hamming greyscale border (auto = from IPL's SEG processing log: duplicate / none)")
    ap.add_argument("--lh-both", action="store_true", help="also compute the other LH border variants' SEGs (reported, not used downstream)")
    ap.add_argument("--lh-pad-offset", default=ormir.LH_PAD_OFFSET, choices=list(ormir.LH_PAD_OFFSETS),
                    help="power-of-two padding offset of /fft_laplace_hamming: ceil (IPL's rule, probe 21, default) or floor "
                         "(the rule every result under validation/results was computed with before 2026-09-17)")
    ap.add_argument("--lh-dtype", default=ormir.LH_DTYPE, choices=list(ormir.LH_DTYPES),
                    help="FFT / filter arithmetic of /fft_laplace_hamming: float32 (the shipped engine, default; every result "
                         "under validation/results) or float64 (opt-in under study since 2026-09-17: the same operations in "
                         "double precision, rounded to float32 before IPL's float -> short conversion)")
    ap.add_argument("--seg-variant", default="periosteal_first", choices=["periosteal_first", "gobj_first", "auto", "best"],
                    help="SEG assembly order used for the maps: periosteal_first (default) = the shipped engine's order, the "
                         "order every published radius / tibia record uses; auto = the one detected from IPL's SEG processing "
                         "log; best = the variant whose SEG fits IPL's better (a diagnostic; both are computed and recorded)")
    ap.add_argument("--no-seg-both", dest="seg_both", action="store_false", help="do not compute the other SEG-assembly variant for the record (implied off by --seg-variant best)")
    ap.add_argument("--trab-seg-mask", default="auto", choices=["auto", "gobj", "none"],
                    help="whether Script 32 STEP 2's '/gobj_maskaimpeel_ow -gobj <base>_trab_mask.gobj' was applied to TRAB_SEG. "
                         "'gobj' (the normal behaviour) crops the trabecular component extraction to the trabecular contour; "
                         "'none' leaves it uncropped, which is what IPL's own logs record when that command does not run. "
                         "'auto' (the default) reads it from IPL's SEG processing log -- a D3P_GobjOrAimMaskAimPeel_OW after "
                         "D3P_Cl_ExtractNumber_CPP means 'gobj', its absence 'none' -- or from IPL's evaluation log for a "
                         "measurement that ships no SEG.")
    ap.add_argument("--no-a2", action="store_true", help="skip configuration A2 on CORR measurements")
    ap.add_argument("--skip-porosity", action="store_true",
                    help="do not run IPL's cortical pore cascade (Ct.Po) in configurations A and B")
    ap.add_argument("--no-hash", action="store_true", help="skip the md5 of every input file")
    ap.add_argument("--verbose-step1", action="store_true")
    ap.add_argument("--cache", default=vai.CACHE_DIR, help="renderings of the patella cohort's raw masks (bool .npy)")
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--fail-fast", action="store_true", help="stop at the first measurement that raises (default: record the error and continue)")
    a = ap.parse_args(argv)
    a.configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    layout = LAYOUTS[a.dataset]
    root = a.root or layout["root"]
    results_root = os.path.join(HERE, "results", a.dataset)
    if a.results is None:
        # results/<dataset>/validate/: results/oslh/records/ already holds the inventory's 123 records under the same
        # <group>_<id>.json names, so the runner never writes into results/<dataset>/ itself
        a.results = os.path.join(results_root, "validate")
        if a.subjects or a.groups:
            tag = "_".join((a.subjects or []) + (a.groups or [])).replace("/", "_")
            tag = tag if len(tag) <= 60 else f"{len(a.subjects or [])}subjects_{len(a.groups or [])}groups"
            a.results = os.path.join(results_root, "validate", f"subset_{tag}")
    # the published result directories are refused unless --allow-overwrite-shipped (docstring, "SHIPPED-DIRECTORY
    # GUARD"); runs before anything is written
    vfc.guard_shipped_results(a.results, a.lh_pad_offset, a.allow_overwrite_shipped, log_fn=log, lh_dtype=a.lh_dtype)
    os.makedirs(a.results, exist_ok=True)
    rec_dir = os.path.join(a.results, "records")
    os.makedirs(rec_dir, exist_ok=True)
    try:
        os.makedirs(a.log_dir, exist_ok=True)
        _LOG_FH[0] = open(os.path.join(a.log_dir, f"validate_dataset_{a.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.log"), "w", encoding="utf-8")
    except Exception as e:
        print(f"log file not opened: {e}")
    meas_list = discover(a.dataset, root, a.groups, a.subjects)
    inventory = read_inventory(results_root) if a.dataset == "oslh" else None
    log(f"ipldt {ipldt.__version__}; dataset {a.dataset}; root {root}; backend {ipldt.gpu.resolve_backend(a.backend) if not a.skip_dt else 'n/a'}; "
        f"{len(meas_list)} measurements; configs {a.configs}; dt {'skipped' if a.skip_dt else 'on'}; results {a.results}")
    if not meas_list:
        log("nothing to do")
        return 1
    records = []
    for meas in meas_list:
        f = os.path.join(rec_dir, f"{meas.tag}.json")
        if a.resume and os.path.exists(f):
            try:
                rec = json.load(open(f))
                if "error" not in rec and (a.skip_dt or "maps" in rec.get("B", {}) or "B" not in a.configs):
                    log(f"{meas.id}: reusing {f}")
                    records.append(rec)
                    continue
            except Exception:
                pass
        try:
            rec = process_measurement(meas, a, a.cache, inventory)
        except Exception as e:
            tb = traceback.format_exc()
            log(f"{meas.id}: FAILED: {e}\n{tb}")
            if a.fail_fast:
                raise
            rec = dict(id=meas.id, tag=meas.tag, group=meas.group, base=meas.base, dataset=a.dataset, corr=meas.corr, error=str(e), traceback=tb)
        json.dump(rec, open(f, "w"), indent=1, default=ormir._json_default)
        records.append(rec)
        gpu_release()
    good = [r for r in records if "error" not in r]
    failed = [r for r in records if "error" in r]
    if failed:
        log(f"{len(failed)} measurement(s) failed: {[r['id'] for r in failed]}")
    if not good:
        return 1
    have_dt = all("maps" in r.get("B", {}) for r in good) and not a.skip_dt
    rows = [flat_row(r) for r in good]
    # EXCLUDED_IDS removes a measurement from the AGGREGATES only: every record, every table row and every
    # per-measurement entry below is written for all of 'good' regardless (see the module docstring).
    kept, _, excl = split_excluded(good, None, a.include_excluded)
    if excl:
        for r in excl:
            log(f"{r['id']}: EXCLUDED from every aggregate (kept in records.json, table.csv / table.md with the reason in "
                f"column 'excluded', and in summary.md's per-measurement listings): {exclusion_reason(r)}")
        log(f"aggregates over {len(kept)} of {len(good)} measurements ({len(excl)} excluded; --include-excluded counts them again)")
    elif a.include_excluded:
        log(f"--include-excluded: aggregates over all {len(good)} measurements, EXCLUDED_IDS ignored")
    json.dump(records, open(os.path.join(a.results, "records.json"), "w"), indent=1, default=ormir._json_default)
    vfc.write_table_csv(rows, os.path.join(a.results, "table.csv"))
    write_table_md(rows, os.path.join(a.results, "table.md"), have_dt)
    blocks = write_summary(good, rows, os.path.join(a.results, "summary.md"), a, have_dt, a.dataset)
    reg = dict(dataset=a.dataset, voxel_mm=VOXEL_MM, blocks=[dict(label=b["label"], n=b["n"], metrics=b["metrics"]) for b in blocks], pooled_joint_histograms=pooled_hists(kept, have_dt),
               excluded={r["id"]: exclusion_reason(r) for r in excl}, include_excluded=bool(a.include_excluded),
               per_measurement={r["id"]: {cfg: {name: dict(hist=r[cfg]["maps"][name]["hist"], mismatches=r[cfg]["maps"][name]["mismatches"]) for name in r[cfg]["maps"]}
                                          for cfg in CONFIGS if _get(r, cfg, "maps")} for r in good} if have_dt else {})
    json.dump(reg, open(os.path.join(a.results, "regression_data.json"), "w"), default=ormir._json_default)
    for st in blocks[0]["stages"]:
        log(f"  {st['stage']:<42s} exact {st['exact']}/{st['n']}  mismatches {st['mismatches']:,d}" + (f"  Dice {st['min_dice']:.6f}..{st['max_dice']:.6f}" if st.get("min_dice") is not None else ""))
    for m in blocks[0]["metrics"]:
        log(f"  {m['config']:<2s} {m['metric']:<12s} n {m['n']}  mean |diff| {m['mean_abs']:.3e} {m['unit']}  max |diff| {m['max_abs']:.3e}  max |rel| {m['max_rel']:.5f} %  exact {m['exact']}/{m['n']}  slope {m['slope']:.6f} int {m['intercept']:+.2e} R2 {m['r2']:.6f}")
    log(f"wrote {a.results}/table.csv, table.md, summary.md, records.json, regression_data.json, records/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
