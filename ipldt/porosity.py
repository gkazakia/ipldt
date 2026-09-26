"""ipldt.porosity -- clean-room reimplementation of IPL V5.42's cortical-pore cascade: the Burghardt (Bone 2010,
Fig. 2) block that the manufacturer's standard evaluation (Script 32) runs in STEP 2 to produce
<base>_PORE.AIM, plus the one command in it that ipldt.ipl_ops did not have, /hysteresis_threshold.

Every rule below was derived from IPL's own on-screen help for /hysteresis_threshold, from the manufacturer's
standard evaluation script (Script 32) and from IPL's exported products only (no binary was
disassembled), and each is a DISCRETE hypothesis accepted only on exact voxel identity with IPL's own
exported PORE.AIM on scans of different sites, never fitted.  The operators the cascade is built from
(/set_value, /subtract_aims, /add_aims, /cl_rank_extract, /cl_nr_extract, /cl_slicewise_extractow,
/gobj_maskaimpeel_ow) are the reimplementations in ipldt.ipl_ops, each verified against IPL's exports; this module adds only the
hysteresis and the plumbing, and inherits ipl_ops' volume convention (dict(data=(z, y, x), dim=(x, y, z),
pos=(x, y, z)), set voxels are non-zero, char arithmetic on the union bounding box).

VALIDATION.  In configuration A of the paper (IPL's cortical segmentation and IPL's rendered cortical contour
in) pore_cascade reproduces IPL's exported PORE.AIM with 0 differing voxels on all 137 validation scans --
21 patellae, 29 ultradistal radii, 29 ultradistal tibiae and 58 diaphyses; 4,775,892,520 voxels compared,
71,277,708 IPL pore voxels, Dice 1 on every scan -- and on the excluded Diaphyseal/CKD/991161 as well
(validation/porosity_AB_stats.py; manuscript/facts/FACTS_BMD_CTPO.md).  The candidate readings of the
hysteresis were enumerated on five scans, the ones the refuted-reading table below is measured on
(Diaphyseal/CKD/251016, 558517 and 341269, Distal/REPRO/444590, Distal/CKD/345857); every other scan was run after the
reading had been fixed.

A WARNING ABOUT THE PATELLA COHORT, stated because it is easy to over-read those 21 exact results: on a
patella the hysteresis is a NO-OP.  Value 2 -- a void inside the compartment that is not marrow-connected
and not small in its slice, the only thing the hysteresis can grow into -- is EMPTY on every patella
measured (PFJ-ab6af9_R: 0 voxels of value 2 against 215,060 of value 3), so pores_F comes out empty and the
patella confirms the rest of the cascade only.  Measured: the accepted reading and both of its bound
variants (seed strictly < low, weak strictly < high) all give 0 mismatching voxels on PFJ-ab6af9_R, so no
patella separates them.  What separates the readings is the diaphyseal scans (251016 has 165,673 value-2
voxels, 79,261 of them in IPL's PORE) and Distal/REPRO/444590 (the only scan that pins the low bound).

The GRID the result is written on is IPL's own on 79 of the 137 scans; on the others the box of IPL's PORE.AIM
differs slightly from the one built here, and the pore voxels are identical in GLOBAL coordinates all the same
(every comparison is by global position).  For the patellae no contour rendering is on disk, and the box built
here (see THE CORTICAL CONTOUR) is within one voxel per axis of IPL's.

THE INPUT OF /hysteresis_threshold.  The block builds a label image cortseg_CDE whose voxels can only take
the values 0, 1, 2, 3, 125 and 127, and the whole derivation is the enumeration of what each value means
(the letters are Burghardt's Fig. 2):

    cortring_C  = /gobj_to_aim of CORT_MASK.GOBJ, /set_value 2 0        -> 2 inside the cortical compartment
    cortseg_D2  = CORT_SEG.AIM, /set_value 125 0                        -> 125 on cortical bone
    pores_E     = CORT_SEG inverted, then /cl_slicewise_extractow 0..5 % at value 1
                                                                        -> 1 on a void that is SMALL IN ITS SLICE
    pores_M     = rank-1 component of "everything that is not cortical bone", masked back to the compartment
                                                                        -> the marrow / outside blob
    cortring_C3 = cortring_C - pores_M                                  -> 2 inside the compartment, NOT marrow
    cortseg_CDE = cortring_C3 + (pores_E + cortseg_D2)

      0 = marrow, or anything outside the cortical compartment        1 = a slice-wise small void in that region
      2 = a void inside the compartment, not marrow-connected,        3 = the same, and SMALL IN ITS SLICE
          but LARGE in its slice (the weak pore evidence)                 (the strong pore evidence)
    125 = bone outside the compartment                              127 = cortical bone

THE READING ACCEPTED for the script's call (low_thresh 1, high_thresh 3, unit 5, mode 0, grow_axes 0 0 1,
value_in_range 127) is the SYMMETRIC LOW-INTENSITY reading, with BOTH bounds INCLUSIVE:

    mode 0 ("segment low intensity object"):  seed = v <= low_thresh      weak = v <= high_thresh
    mode 1 ("segment high intensity object"): seed = v >= high_thresh     weak = v >= low_thresh
    out = the weak voxels whose connected component -- with connectivity restricted to the axes flagged in
          grow_axes, so [0 0 1] means +-z only, i.e. a maximal run of weak voxels down one (x, y) column --
          contains at least one seed voxel; the seed voxels are kept (they are a subset of the weak set).

On the script's numbers that is: seed = {0, 1} (marrow and outside), weak = {0, 1, 2, 3} (everything that is
not bone), grown along z only.  The structural check that singles it out before any run: it is the ONLY
reading under which pores_M is entirely inside pores_F0, which is exactly what makes the script's next two
lines ("/set pores_M 127 0" then "/subtract_aims pores_F0 - pores_M") necessary; under every reading that
leaves the marrow out of the weak set, /subtract_aims writes -127 -- a SET voxel in IPL's char convention,
see ipl_ops.subtract_aims -- over the unselected marrow, and those voxels survive into pores_G and into the
written PORE.AIM.  That prediction was written down before the runs and is what the mismatch counts show.

REFUTED READINGS (mismatching voxels against IPL's PORE.AIM, in the order
251016 / 444590 / 345857 / 558517 / 341269; a 0 means that scan does not separate that reading, never that it is right):

    seed v >= high, weak v >= low (mode ignored)      12,316,298 / 8,034,272 / 2,224,802 / 6,046,520 / 3,835,827
    seed v > high,  weak v > low  (strict)            12,316,611 / 8,056,063 / 2,224,803 / 6,046,536 / 3,835,869
        -- both put the bone value 127 in the seed AND the weak set, so pores_F0 is the cortex itself
    weak = the BAND low <= v <= high, seed v >= high      157,561 /   892,874 /   199,816 /   115,118 /    35,733
    weak = the BAND low <= v <= high, seed v <= low       146,459 /   871,344 /   199,816 /   115,118 /    30,897
    weak v <= high, seed v >= high                       157,156 /   892,874 /   199,816 /   115,118 /    35,387
    weak v >= low,  seed v <= low                        331,809 / 3,078,497 /   483,245 / 1,503,496 /    31,083
        -- all four leave the marrow (value 0) out of the weak set: the -127 mechanism above
    weak v <= low  (one threshold only)                   80,413 /         0 /         0 /         0 /    21,617
    weak = every voxel (no upper bound)                3,109,051 / 7,016,784 / 1,987,594 / 2,428,488 /   562,621
    BOTH bounds strict (seed v < low, weak v < high)       2,019 /     7,363 /         0 /         0 /       756
    seed v <  low (strict), weak v <= high                     0 /     7,363 /         0 /         0 /         0
    seed v <= low, weak v <  high (strict)                 2,019 /         0 /         0 /         0 /       756
        -- so low_thresh is inclusive (444590 separates it) and high_thresh is inclusive (251016 and 341269 do)
    output = the grown set MINUS the seeds                     - /   892,874 /         - /         - /     9,280
    no seeding at all (output = the whole weak set)            - /         0 /         - /         - /    24,693
    grow_axes ignored, full 6-connectivity                83,281 /         0 /         - /         - /    24,693
    grow_axes read as "the axes NOT to grow along"        83,281 /         0 /         - /         - /    24,693

NOT OBSERVED, so not implemented: mode 1, any -unit other than 5 (native), and any grow_axes other than
[0 0 1].  mode 1 and the other grow_axes are implemented as the symmetric reading of the rule that the
script's call pins, and hysteresis_threshold says so in its docstring; -unit raises.

THE CORTICAL CONTOUR must be IPL's OWN /gobj_to_aim rendering ON ITS OWN GRID.  ipldt.contour.render_volume
of the raw CORT_MASK.AIM reproduces the CONTENT of IPL's CORT_MASK_CT.AIM exactly (0 differing voxels on
Distal/CKD/345857 and Diaphyseal/CKD/341269), but IPL renders it onto the periosteal gobj's bounding box, which
is 2-3 voxels wider per side than the raw mask's own box, and the cascade is sensitive to that: the
/cl_slicewise_extractow of pores_DF takes its denominator from the set voxels of each slice, and on the
tight cortical box the background outside the periosteal surface is cut by the box edges into corner
fragments that each fall under the 5 % bound and are kept as "pores".

    contour on the raw CORT_MASK box instead of IPL's gobj box:  2,356 ours-only voxels on PFJ-0be66a_R,
        22 on Distal/CKD/345857, 0 on PFJ-ab6af9_R and on Diaphyseal/CKD/341269 (the divergence appears at pores_G,
        never at pores_M / cortring_C3 / cortseg_CDE / pores_F, which is what identifies the slicewise
        denominator as the mechanism)
    the raw CORT_MASK raster used directly, without rendering it: 199 voxels on Diaphyseal/CKD/341269
        (7 ours-only, 192 IPL-only), 22 on Distal/CKD/345857, 0 on both patellae

    radius / tibia: <base>_CORT_MASK_CT.AIM is IPL's rendering, on IPL's grid -- use it as it is.
    patella: no rendering is on disk; render_volume(<base>_CORT_MASK_decompressed.AIM != 0) pasted onto the
        UNION BOX of the CORT_MASK and TRAB_MASK file grids is exact on all 21 patellae (the patella
        TRAB_MASK.AIM is stored on the periosteal box -- its file grid is much larger than the tight box of
        its own set voxels -- which is where that grid comes from; it is not taken from PORE.AIM).

THE GRID, FROM THE CONTOUR ALONE (2026-09-25).  The grid /gobj_to_aim renders the cortical contour on follows
from the contour itself: the slice headers of the cortical GOBJ store, per slice, the box of that slice's
chains as a size and a centre rounded up, and /gobj_to_aim renders onto that box grown by 2 voxels (the high
side by 3 where a slice of even extent reaches the top), clipped at 0 -- render_grid has the rule and its
evidence.  Applied to IPL's rendered cortical contour it gives IPL's own rendering grid on all 137 validation
scans that have one on disk, and united with the CORT_SEG grid it gives the grid of IPL's PORE.AIM on 138 of
138.  The boxes the files above provide differ from that grid by one voxel at one or two faces on 38 of the
117 radius / tibia scans (the staged <base>_CORT_MASK_CT.AIM) and on all 21 patellae (the union box), and reproduce
PORE.AIM there all the same.  pore_cascade_ipl_grid runs the cascade on that grid from inputs held on any
grid, which is how the workflows call it (ipldt.ormir.step5c_porosity): with IPL's own contour and CORT_SEG
pasted onto each scan's greyscale-AIM grid it reproduces IPL's PORE.AIM on 137 of 137 scans, where the same
inputs run on the greyscale-AIM grid itself differ on 4 (1,595,061 voxels, all extra pore voxels: the edge
fragments above on the distal scans Distal/REPRO/444590, 612066 and 346361, whose bone reaches the AIM's faces,
and the slice-wise denominator on the patella PFJ-6f5538_R).

Ct.Po.  Script 32 does not compute it; the result sheet does, and the sheet generator runs on the
scanner, so the definition was read off the printed values.  The reading that
reproduces them is

    Ct.Po = |PORE and CORT_MASK| / |CORT_MASK|            (rounded to three decimals for the sheet)

i.e. the pore voxels over the CORTICAL COMPARTMENT -- the RENDERED contour -- not Ct.Po.V / (Ct.Po.V +
Ct.BV).  Of the 137 validation scans, 30 have a readable single-measurement result sheet that prints Ct.Po
(all 21 patellae and 9 radius / tibia scans); the ratio above, on ipldt's configuration-A pore map, reproduces
the printed three decimals on all 30.  For the other 107 the sheet saved with the evaluation is a blank stub
(101) or a longitudinal follow-up sheet whose values describe the common region of two measurements rather
than the scan (6), so those are not compared (manuscript/facts/FACTS_BMD_CTPO.md, sheet.census).  The
alternatives, and the sheets that refute them:

    Ct.Po.V / (Ct.Po.V + Ct.BV), Ct.BV = |CORT_SEG|        reproduces none of the 30
        PFJ-ab6af9_R printed 0.096: 0.0961 against 0.1100.  251016 printed 0.080: 0.0804 against 0.0822.
    the same ratio over the RAW CORT_MASK.AIM instead of its rendering
        on the patellae the rendering and the raw raster differ enough to move the third decimal:
        PFJ-351dc7_R printed 0.155, 0.1547 rendered against 0.1564 raw, and PFJ-6f5538_R printed 0.228, 0.2278
        against 0.2286.
    truncation instead of rounding to three decimals      reproduces only 10 of the 21 patellae
        (PFJ-d81140_L printed 0.104 from 0.10351, PFJ-6f5538_R printed 0.228 from 0.22783, and 9 more)

IPL's PORE.AIM almost never has a voxel outside the compartment, so |PORE and CORT_MASK| = |PORE| on nearly
every scan.  It is not a universal law -- PFJ-351dc7_R has 36,501 of 3,402,115 pore voxels outside and PFJ-6f5538_R
8,722 of 2,617,440, identically in ours and IPL's, both patellae, where the contour is rendered onto a
reconstructed box rather than IPL's own gobj box (see THE CORTICAL CONTOUR).  Neither printed value moves
(PFJ-351dc7_R prints 0.155 against 0.1547, PFJ-6f5538_R 0.228 against 0.22783).  The intersection is written out
because it is the definition, not because it usually removes anything.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from . import ipl_ops as ops
from .contour import render_volume

SET = ops.SET                                   # 127, IPL's value_in_range for char masks

#: the /hysteresis_threshold arguments Script 32 uses, and the only ones this module is verified on
SCRIPT32_HYSTERESIS = dict(low_thresh=1, high_thresh=3, unit=5, mode=0, grow_axes=(0, 0, 1),
                           value_in_range=SET)
#: /cl_slicewise_extractow bounds of the two pore passes, and /cl_nr_extract's minimum pore size
SLICE_FRACTION = (0.0, 5.0)
MIN_PORE_VOXELS = 20


def _grow_structure(grow_axes):
    """The 3-D connectivity structure of -grow_axes (gx, gy, gz): a face neighbour along an axis is
    connected only where that axis' flag is set.  [1 1 1] is 6-connectivity, [0 0 1] links a voxel to the
    voxel above and below it and to nothing else, so a component is a maximal run down one (x, y) column."""
    gx, gy, gz = (int(a) for a in grow_axes)
    s = np.zeros((3, 3, 3), bool)
    s[1, 1, 1] = True
    if gz:
        s[0, 1, 1] = s[2, 1, 1] = True
    if gy:
        s[1, 0, 1] = s[1, 2, 1] = True
    if gx:
        s[1, 1, 0] = s[1, 1, 2] = True
    return s


def hysteresis_threshold(v, low_thresh, high_thresh, unit=5, mode=0, grow_axes=(0, 0, 1),
                         value_in_range=SET):
    """/hysteresis_threshold -in v -out ... -low_thresh -high_thresh -unit -mode -grow_axes -value_in_range.

    RULE (the module docstring has the evidence and every refuted alternative with its mismatch count):
    two sets are formed from the voxel values, BOTH BOUNDS INCLUSIVE --

        mode 0, "segment low intensity object":   seed = v <= low_thresh,   weak = v <= high_thresh
        mode 1, "segment high intensity object":  seed = v >= high_thresh,  weak = v >= low_thresh

    -- the weak set is labelled with connectivity restricted to the axes flagged in -grow_axes (so
    [0 0 1] = +-z only: a component is a maximal run of weak voxels down one (x, y) column, [1 1 0] is
    4-connectivity in the axial plane and [1 1 1] is full 6-connectivity), and the components that contain
    at least one seed voxel are written at value_in_range.  The seed set is a subset of the weak set under
    both modes, so the seeds are kept; grid unchanged, output char 0 / value_in_range.

    Verified on Script 32's own call (low 1, high 3, unit 5, mode 0, grow_axes 0 0 1, value_in_range 127)
    as part of pore_cascade: 0 differing voxels against IPL's exported PORE.AIM on every validation scan
    (see the module docstring).  Both bounds are pinned: the strict seed bound leaves 7,363 ours-only
    voxels on Distal/REPRO/444590, the strict weak bound 2,019 IPL-only voxels on Diaphyseal/CKD/251016 and 756
    on Diaphyseal/CKD/341269.  The competing polarities of `mode`, the band reading of the two thresholds,
    "output the grown set minus the seeds", "no seeding at all", and reading grow_axes as full
    6-connectivity or as the axes NOT to grow along are all refuted, by 756 to 12,316,611 voxels.

    ONLY THE SCRIPT'S CALL IS OBSERVED.  -unit 5 (native) is the only unit in the script and the only one
    implemented: any other raises, because the mgHA / permille / HU conversions would have to be guessed.
    mode 1 and grow_axes other than [0 0 1] are the symmetric reading of the rule the script's call pins
    (and IPL's help text spells the grow_axes semantics out), not something an IPL export has confirmed --
    treat them as unverified."""
    if int(unit) != 5:
        raise AssertionError(
            f"hysteresis_threshold: -unit {unit} was never observed; only -unit 5 (native, the thresholds "
            "taken as raw voxel values) is derived and implemented")
    mode = int(mode)
    if mode not in (0, 1):
        raise ValueError(f"hysteresis_threshold: -mode must be 0 or 1, got {mode}")
    lo, hi = int(low_thresh), int(high_thresh)
    if hi < lo:
        raise ValueError(f"hysteresis_threshold: -high_thresh {hi} is below -low_thresh {lo}")
    d = np.asarray(v["data"])
    if mode == 0:
        seed, weak = d <= lo, d <= hi
    else:
        seed, weak = d >= hi, d >= lo
    lab, n = ndi.label(weak, structure=_grow_structure(grow_axes))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[seed])] = True                       # seed is a subset of weak under both modes
    keep[0] = False
    return ops.mask_vol(keep[lab], v["dim"], v["pos"], value_in_range)


def contour_render(raw_mask, dim=None, pos=None):
    """IPL's /gobj_to_aim -peel_iter 0 of a contour, for the cascade's purposes: ipldt's rendering of the
    raw mask raster, pasted onto the grid IPL renders on.

    render_volume of the raw CORT_MASK.AIM reproduces the CONTENT of IPL's own CORT_MASK_CT.AIM exactly
    (0 differing voxels on Distal/CKD/345857 and Diaphyseal/CKD/341269), but IPL renders onto the periosteal
    gobj's box, not the mask's tight box, and pore_cascade is sensitive to that grid -- see the module
    docstring (2,356 voxels on PFJ-0be66a_R, 22 on Distal/CKD/345857).  Pass dim / pos to say which grid; with
    neither, the raw mask's own grid is used, which is NOT IPL's and is exact only by luck."""
    r = ops.mask_vol(render_volume(np.asarray(raw_mask["data"]) != 0), raw_mask["dim"], raw_mask["pos"])
    if dim is None and pos is None:
        return r
    if dim is None or pos is None:
        raise ValueError("contour_render: pass both dim and pos, or neither")
    return ops.vol(ops.on_grid(r, tuple(dim), tuple(pos)), tuple(dim), tuple(pos))


def pore_cascade(cort_render, cort_seg, slice_fraction=SLICE_FRACTION, min_pore_voxels=MIN_PORE_VOXELS,
                 hysteresis=None, keep_stages=False):
    """The Burghardt cortical-pore cascade (Burghardt et al., Bone 2010, Fig. 2) as the standard evaluation
    (Script 32, STEP 2) runs it, command for command.

    cort_render is /gobj_to_aim of CORT_MASK.GOBJ (IPL_FNAME0) -- IPL's own <base>_CORT_MASK_CT.AIM where
    there is one, otherwise contour_render(raw mask, *IPL's gobj grid); cort_seg is <base>_CORT_SEG.AIM
    (IPL_FNAME4).  Returns dict(pore=pores_H, ...) and, with keep_stages, every intermediate (named with the
    letters of Burghardt's Fig. 2).  pores_H is what IPL writes as <base>_PORE.AIM: 0 differing voxels on all
    137 validation scans (module docstring).

    The command sequence, with the value each stage carries:

        pores_E     = /set_value(cort_seg, 0, 127)                      invert: 127 on everything not bone
                      /cl_slicewise_extractow 0..5 %, value 1           the slice-wise small voids
        cortseg_D2  = /set_value(cort_seg, 125, 0)
        cortring_C  = /set_value(cort_render, 2, 0)
        cortseg_CD  = /set_value(cort_render, 125, 0) - cortseg_D2
        bckgrnd_C   = /set_value(cort_render, 0, 125)                   125 OUTSIDE the compartment
        pores_M0    = cortseg_CD + bckgrnd_C                            125 on everything not cortical bone
        pores_M     = /cl_rank_extract(pores_M0, 1, 1, connect_boundary false, value 2)   the marrow blob
                      /gobj_maskaimpeel_ow with CORT_MASK.GOBJ, peel 0
        cortring_C3 = cortring_C - pores_M
        cortseg_CDE = cortring_C3 + (pores_E + cortseg_D2)              values 0, 1, 2, 3, 125, 127
        pores_F0    = /hysteresis_threshold(cortseg_CDE, 1, 3, unit 5, mode 0, grow_axes 0 0 1, 127)
                      /gobj_maskaimpeel_ow with CORT_MASK.GOBJ, peel 0
        pores_F     = pores_F0 - /set_value(pores_M, 127, 0)
        pores_DF    = /set_value(pores_F + cort_seg, 0, 127)            invert
                      /cl_slicewise_extractow 0..5 %, value 127
        pores_G     = pores_F + pores_DF
        pores_H     = /cl_nr_extract(pores_G, min 20, max 0, value 127)

    /cl_rank_extract is ipl_ops.cl_ow_rank_extract (the script's call names a separate output, the reimplementation
    is in place; the ranking is the same command).  ipl_ops' char arithmetic is used throughout, including
    its -127 for a negative /subtract_aims difference, which is a SET voxel and is what refutes four of the
    hysteresis readings.  Memory: the intermediates are on the union of the two input grids; keep_stages
    holds all thirteen of them at once."""
    hys = dict(SCRIPT32_HYSTERESIS)
    if hysteresis:
        hys.update(hysteresis)
    lo_frac, up_frac = float(slice_fraction[0]), float(slice_fraction[1])

    # ---- 1st pore estimate: the slice-wise small voids of the cortical segmentation
    pores_E = ops.set_value(cort_seg, 0, 127)                               # invert
    pores_E = ops.cl_slicewise_extractow(pores_E, lo_frac, up_frac, value_in_range=1)

    cortseg_D2 = ops.set_value(cort_seg, 125, 0)
    cortring_C = ops.set_value(cort_render, 2, 0)

    # ---- marrow-connected voids: the rank-1 non-bone component, inside the compartment
    cortring_C2 = ops.set_value(cort_render, 125, 0)
    cortseg_CD = ops.subtract_aims(cortring_C2, cortseg_D2)
    del cortring_C2
    bckgrnd_C = ops.set_value(cort_render, 0, 125)
    pores_M0 = ops.add_aims(cortseg_CD, bckgrnd_C)
    del cortseg_CD, bckgrnd_C
    pores_M = ops.cl_ow_rank_extract(pores_M0, 1, 1, connect_boundary=False, value_in_range=2)
    pores_M = ops.gobj_maskaimpeel_ow(pores_M, cort_render, 0)
    cortring_C3 = ops.subtract_aims(cortring_C, pores_M)

    # ---- 2nd pore estimate: the label image, then the hysteresis along z
    cortseg_DE = ops.add_aims(pores_E, cortseg_D2)
    cortseg_CDE = ops.add_aims(cortring_C3, cortseg_DE)
    pores_F0 = hysteresis_threshold(cortseg_CDE, **hys)
    pores_F0 = ops.gobj_maskaimpeel_ow(pores_F0, cort_render, 0)
    pores_F = ops.subtract_aims(pores_F0, ops.set_value(pores_M, 127, 0))

    # ---- final estimate: refill the voids the new pores opened up, then drop the tiny components
    pores_DF = ops.add_aims(pores_F, cort_seg)
    pores_DF = ops.set_value(pores_DF, 0, 127)                              # invert
    pores_DF = ops.cl_slicewise_extractow(pores_DF, lo_frac, up_frac, value_in_range=SET)
    pores_G = ops.add_aims(pores_F, pores_DF)
    pores_H = ops.cl_nr_extract(pores_G, int(min_pore_voxels), 0, value_in_range=SET)

    out = dict(pore=pores_H)
    if keep_stages:
        out.update(pores_E=pores_E, cortseg_D2=cortseg_D2, cortring_C=cortring_C, pores_M0=pores_M0,
                   pores_M=pores_M, cortring_C3=cortring_C3, cortseg_DE=cortseg_DE,
                   cortseg_CDE=cortseg_CDE, pores_F0=pores_F0, pores_F=pores_F, pores_DF=pores_DF,
                   pores_G=pores_G, pores_H=pores_H)
    return out


# ============================================================================================ IPL's grids
#: /gobj_to_aim renders a contour on the box of its GOBJ slice headers grown by this many voxels per in-plane side
RENDER_GRID_MARGIN = 2


def _grid_of(dim, pos):
    return tuple(int(v) for v in dim), tuple(int(v) for v in pos)


def _union(*grids):
    """The union bounding box of (dim, pos) grids (None entries ignored)."""
    grids = [g for g in grids if g is not None]
    lo = [min(g[1][i] for g in grids) for i in range(3)]
    hi = [max(g[1][i] + g[0][i] for g in grids) for i in range(3)]
    return tuple(hi[i] - lo[i] for i in range(3)), tuple(lo)


def tight_grid(v):
    """(dim, pos) of the tight box of a volume's set voxels -- /bounding_box_cut -border 0 --, None when empty."""
    d = np.asarray(v["data"]) != 0
    zs = np.flatnonzero(d.any(axis=(1, 2)))
    if zs.size == 0:
        return None
    ys = np.flatnonzero(d.any(axis=(0, 2)))
    xs = np.flatnonzero(d.any(axis=(0, 1)))
    lo = (int(xs[0]), int(ys[0]), int(zs[0]))
    hi = (int(xs[-1]) + 1, int(ys[-1]) + 1, int(zs[-1]) + 1)
    return tuple(hi[i] - lo[i] for i in range(3)), tuple(int(v["pos"][i]) + lo[i] for i in range(3))


def render_grid(cort_render, margin=RENDER_GRID_MARGIN, clip_low=0):
    """(dim, pos) of the grid IPL's /gobj_to_aim renders a contour on, computed from the RENDERED contour alone
    (a volume dict on any grid); None when the contour is empty.  In Script 32's pore block this is the grid of
    cortring_C, the /gobj_to_aim of CORT_MASK.GOBJ.

    THE RULE.  /togobj_from_aim writes, for every slice with a contour, a slice header that stores the
    in-plane box of that slice's stored chains per axis as a size S = hi - lo + 1 and an integer centre
    C = ceil((lo + hi) / 2).  /gobj_to_aim renders onto [ceil(min C - S/2) - 2, floor(max C + S/2) + 2] per
    in-plane axis, the minimum and maximum taken over the slices, with the low end clipped at global position
    0, and onto the slices that carry a contour in z.  With the header's rounding this is

        low  = (min over the slices of lo) - 2, clipped at 0
        high = (max over the slices of hi + [S even]) + 2

    i.e. the contour's tight box grown by 2 on the low side and by 2 or 3 on the high side: 3 exactly when a
    slice of even extent reaches (or a slice of even extent ends one voxel short of) the highest coordinate.
    The chains of a slice span exactly the slice's rendering (the chain pixels plus the polygon interior,
    ipldt.contour.render), so lo and hi are read off the rendered raster.  The clip never cuts the contour:
    the low end is max(lo - margin, min(lo, clip_low)).

    EVIDENCE.  The slice headers: in the 186 CORT_MASK.GOBJ files of the 117 radius / tibia measurements of
    the validation (every version the scanner kept; not distributed), S equals the extent of the slice's own
    stored chains and C = ceil((lo + hi) / 2) on 62,486 of 62,496 slice-axes, every slice of 184 files;
    ipldt.contour reproduces the chains of 175 of them vertex for vertex from a CORT_MASK.AIM version on disk.
    (The contour headers inside a slice round the centre down instead, and are not what the grid follows; the
    periosteal, trabecular and operator-corrected GOBJs on disk round either way, so the rule is stated for
    the cortical GOBJ of the pore block.)  The other 10 slice-axes, in 2 files, store a box 1 to 6 voxels
    wider than their chains -- in one of the stored versions of the cortical GOBJ of Distal/CKD/487451, slices
    whose raw mask carries 1- to 7-voxel pieces that store no chain -- and none reaches its file's extreme,
    so no grid moves; a rule read off the rendering cannot see such pieces.  The grid: the rule applied to
    IPL's rendered cortical contour reproduces IPL's own /gobj_to_aim grid on every scan of the validation set
    that has one on disk (137 of 137: a stored version of IPL's rendering <base>_CORT_MASK_CT.AIM for 116 of
    the 117 radius / tibia scans, and IPL's /gobj_to_aim export of the cortical GOBJ from a dedicated scanner
    run for the 21 patellae; neither is distributed), and its union with IPL's CORT_SEG grid is the grid of
    IPL's PORE.AIM on 138 of 138.
    The clip is seen on PFJ-351dc7_R and PFJ-6f5538_R, whose contour touches y = 0: the unclipped rule gives y = -2,
    IPL's rendering starts at y = 0."""
    d = np.asarray(cort_render["data"]) != 0
    zs = np.flatnonzero(d.any(axis=(1, 2)))
    if zs.size == 0:
        return None
    pos = [int(p) for p in cort_render["pos"]]
    lo, hi = [0, 0], [0, 0]
    for a, prof in ((0, d.any(axis=1)), (1, d.any(axis=2))):         # (z, x) and (z, y) occupancy
        p = prof[zs]
        first = p.argmax(axis=1)
        last = p.shape[1] - 1 - p[:, ::-1].argmax(axis=1)
        even = (last - first + 1) % 2 == 0
        l0 = int(first.min()) + pos[a]
        lo[a] = max(l0 - int(margin), min(l0, int(clip_low))) if clip_low is not None else l0 - int(margin)
        hi[a] = int((last + even).max()) + pos[a] + int(margin)
    z0, z1 = int(zs[0]) + pos[2], int(zs[-1]) + pos[2]
    return (hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, z1 - z0 + 1), (lo[0], lo[1], z0)


def cascade_grids(cort_render, cort_seg, seg_grid=None):
    """The grids Script 32's pore block works on: (R, S, U) with R = render_grid(cort_render) (the cortical
    contour's /gobj_to_aim grid), S = the grid of CORT_SEG.AIM and U = R united with S, the grid every stage
    from cortseg_CD on -- and the written PORE.AIM -- lives on (pore_cascade works on the union bounding box of
    its inputs, as IPL's /add_aims and /subtract_aims do).

    IPL's CORT_SEG.AIM is written on /bounding_box_cut -border 0 of the periosteally masked segmentation
    (Script 32 STEP 2, seg_box), which lies inside the cortical contour's extent, so S lies inside R and U is
    R: S reaches outside R only on PFJ-351dc7_R and PFJ-6f5538_R, whose IPL CORT_SEG grid starts one row outside the
    image (y = -1; that row holds no set voxel).  Pass that grid as seg_grid where it is known (the
    validation, fed IPL's own CORT_SEG); without it S is the tight box of cort_seg's set voxels, which is what
    a workflow has, and lies inside R whenever cort_seg lies inside the contour.  S affects only pores_E,
    which is inert (it turns 0 into 1 and 2 into 3, both on the same side of the hysteresis), so the pore map
    depends on U alone."""
    R = render_grid(cort_render)
    if R is None:
        raise ValueError("cascade_grids: the cortical contour is empty")
    S = _grid_of(*seg_grid) if seg_grid is not None else tight_grid(cort_seg)
    S = S if S is not None else R
    return R, S, _union(R, S)


def pore_cascade_ipl_grid(cort_render, cort_seg, seg_grid=None, **kwargs):
    """pore_cascade on the grids IPL runs it on, whatever grid the inputs are held on (the workflows hold both
    on the whole input AIM): the rendered cortical contour is cut / zero-padded onto render_grid(cort_render)
    and CORT_SEG onto its own grid (cascade_grids), pore_cascade runs there, and the result is returned on the
    cascade grid U; paste it back with ipldt.io.align_to.  **kwargs go to pore_cascade.

    Why: the cascade is sensitive to its working grid, not only to its inputs' content.  The two slice-wise
    0..5 % passes measure every component against the non-bone voxels of its slice IN THE WORKING GRID, so a
    larger grid lowers the bar (PFJ-6f5538_R on the whole greyscale AIM: 25,428 extra pore voxels in 14 slices),
    and where the bone reaches the faces of a tight grid the background outside the contour is cut into
    pieces that each pass as a pore (Distal/REPRO/444590 on the greyscale AIM: 1,024,487 extra voxels).  IPL's
    render grid always extends at least 2 voxels past the contour, and the zero padding beyond the image is
    where it does.  With IPL's own contour and CORT_SEG this reproduces IPL's PORE.AIM on all 137 validation
    scans from their greyscale-AIM grid (module docstring, THE GRID; ormir_bqrl/README.md, "The pore map's grid").

    Returns pore_cascade's dict plus render_grid (R), seg_grid (S), grid (U) and cort_render (the contour on
    R, the compartment Ct.Po counts)."""
    R, S, U = cascade_grids(cort_render, cort_seg, seg_grid)
    cr = ops.vol(ops.on_grid(cort_render, *R), *R)
    cs = ops.vol(ops.on_grid(cort_seg, *S), *S)
    lost = int(np.count_nonzero(cort_render["data"])) - int(np.count_nonzero(cr["data"]))
    if lost:                                   # cannot happen: R contains the contour (the clip never cuts it)
        raise AssertionError(f"pore_cascade_ipl_grid: {lost} contour voxels fall outside the render grid {R}")
    out = pore_cascade(cr, cs, **kwargs)
    out.update(render_grid=R, seg_grid=S, grid=U, cort_render=cr)
    return out


def ct_po(pore, cort_render):
    """Ct.Po as IPL's result sheet prints it: |PORE and CORT_MASK| / |CORT_MASK|, the pore voxels over the
    CORTICAL COMPARTMENT.  Returns dict(ct_po, pore_voxels, pore_in_mask, mask_voxels).

    This is the reading that reproduces the printed value, not Ct.Po.V / (Ct.Po.V + Ct.BV): it reproduces the
    printed three decimals on all 30 validation scans whose single-measurement result sheet prints Ct.Po (21
    patellae, 9 radius / tibia scans), and the Ct.BV reading on none (module docstring).  The intersection
    with the compartment is written out because it is the definition, not because it usually removes
    anything."""
    dim, pos = ops.union_grid(pore, cort_render)
    p = ops.on_grid(pore, dim, pos) != 0
    m = ops.on_grid(cort_render, dim, pos) != 0
    pm, mv = int((p & m).sum()), int(m.sum())
    return dict(ct_po=(pm / mv if mv else float("nan")), pore_voxels=int(p.sum()), pore_in_mask=pm,
                mask_voxels=mv)


def compare_pore(ours, ipl):
    """Voxel-for-voxel comparison of two pore maps on the union of their grids (set = non-zero), the way
    every other ipldt validator reports one: dict(mismatch, ours, ipl, ours_only, ipl_only, dice,
    same_grid, compared_voxels)."""
    dim, pos = ops.union_grid(ours, ipl)
    a = ops.on_grid(ours, dim, pos) != 0
    b = ops.on_grid(ipl, dim, pos) != 0
    inter = int((a & b).sum())
    na, nb = int(a.sum()), int(b.sum())
    return dict(mismatch=int((a ^ b).sum()), ours=na, ipl=nb, ours_only=na - inter, ipl_only=nb - inter,
                dice=(2.0 * inter / (na + nb) if na + nb else 1.0),
                same_grid=(tuple(ours["dim"]) == tuple(ipl["dim"]) and tuple(ours["pos"]) == tuple(ipl["pos"])),
                compared_voxels=int(a.size), compared_grid=dict(dim=list(dim), pos=list(pos)))
