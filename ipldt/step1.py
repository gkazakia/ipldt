"""ipldt.step1 -- Scanco Script 32 STEP 1 (Script 33 differs in two parameters): IPL's cortical / trabecular
separation of an HR-pQCT scan from the greyscale and the periosteal contour, reimplemented command for command
with ipldt.ipl_ops and verified end to end against IPL's exported intermediates.

THE CHAIN (36 IPL commands from /gobj_to_aim to the two /togobj_from_aim, /write, /delete, /rename not counted;
34 change state -- the two '/set 127 0' no-ops on trab_open / trab_bbc do not -- and 32 of those carry the 32
tags 00..31 of the T16 test-run export: the /gobj_maskaimpeel_ow peel 0 of the greyscale and its /bounding_box_cut
-> aim_bbc carry no tag and are folded into 01; the 30 volume tags 00..29 are the keys of the returned `stages`,
30_cortgobj / 31_trabgobj are the renderings of the two contour files):
  00_all        /gobj_to_aim of the periosteal gobj, peel 0                    [input: the rendered contour]
  01_seggauss   /seg_gauss sigma 2 support 3, 500..3000 mgHA of aim_bbc = bounding_box_cut(grey masked to 00)
  02_trab0      00 - 01                       03_peel6   02 & peel(00, peel0)      04_inv      set 0 127 (invert)
  05_rank1      largest 6-cc of 04            06_bgrm    05 & peel(00, 0)          07_trab1    00 - 06
  08_trabrank   largest 6-cc                  09_ero3    erosion `erode`           10_erorank  largest 6-cc
  11_dil3       dilation `dilate`             12_close15 close `close1`            13_close15peel 12 & peel(00, peel0)
  14_bbc        bounding_box_cut              15_open15  open `open_`              16_corners  14 - 15
  17_cornero    erosion `corner_erode`        18_corncl  cl_nr >= corner_min       19_cornmajor dilation `corner_dilate`
  20_corncl2    cl_nr >= corner_min           21_corners2 cl_nr 1..corner_max      22_trabadd  21 + 15
  23_close50    close `close2`                24_close50peel 23 & peel(00, peel0)  25_slicewise slicewise lo..up
  26_cort       00 - 25                       27_cortslice slicewise lo..up        28_cortfinal bounding_box_cut -> CORT_MASK
  29_trabfinal  00 - 28 -> TRAB_MASK
(The contour files, /togobj_from_aim of 28 and 29, are the contour module's job.)

PARAMETERS: Step1Params.  Everything is literal in the script except IPL_PEEL0 (peel0, 6 in every
evaluation), IPL_MISC1_0 (corner_min: 200000 tibia / 800 radius) and IPL_MISC1_1 (close2: 50 tibia / 30
radius).  TIBIA is the Script 32 preset, RADIUS the Script 33 preset.  Every literal is a field too (since
2026-09-26: the rank range and -connect_boundary of stages 05 / 08 / 10, -continuous_at_boundary of 11 / 12 / 19 /
23, the peel of the peel-0 masking steps, the remaining /cl_nr_extract bounds of 18 / 20 / 21 and the
/bounding_box_cut border), with the script's value as its default, so a user can change any value of the chain
(ipldt.params, `--set step1.NAME=VALUE`) while TIBIA / RADIUS and every output stay what they were.  Both have a stage-by-stage oracle:
TIBIA the T16 export of PFJ-0be66a_R (X2420448, a tibia), RADIUS the T17 exports of PFJ-0be66a_R, PFJ-42293d_L and PFJ-6f5538_R
(2026-09-14: every stage 01..29 exact, isolated and cumulative); the stage tags keep the test-run-16 tibia names
('23_close50') for either preset.

VERIFICATION (PFJ-0be66a_R / X2420448, tibia, whole volumes compared by global position, grids compared too):
  isolated (every command fed IPL's previous stage): stages 01..29 all 0 mismatches, identical grids;
  cumulative from the greyscale + IPL's stage 00: stages 01..29 all 0 mismatches, identical grids; 28 equals
  X2420448_T16_28_CORTFINAL and the September evaluation's raw X2420448_CORT_MASK.AIM (7,154,580 voxels,
  738x343x168 @ 798,97,168), 29 equals T16_29 and X2420448_TRAB_MASK.AIM (20,329,639 voxels, 743x348x168
  @ 795,94,168), 0 mismatches each.  Log cross-checks: native thresholds 4524 / 17173 from the proclog;
  |peel 6| 25,892,693; 6-cc counts and largest sizes of 04 / 07 / 09 / 17 = 2145 / 550 / 1695 / 64 and
  23,425,237 / 19,959,727 / 15,730,845 / 3,500.  Stages 25 and 27 (/cl_slicewise_extractow) are no-ops on
  PFJ-0be66a and on the test phantom (one 4-connected component per slice); their wiring 24 -> 25 -> 26 -> 27 ->
  28 -> 29 is covered by a spy test.  (See tests/test_step1.py and the docstrings of ipl_ops.)

PRECONDITION: the periosteal contour must select at least one non-zero greyscale voxel.  An empty contour, a
contour on a grid disjoint from the greyscale, or an all-zero greyscale inside the contour raises ValueError
before /seg_gauss (IPL's behaviour for an empty aim_bbc is unobserved; without the check the whole greyscale
would be smoothed and the union grids of the later stages could grow without bound).

TEST RUN 18 (designed phantoms, scanner run 2026-09-13; details in the ipl_ops docstrings) settled what the
PFJ-0be66a oracle could not: /cl_slicewise_extractow's denominator (slice total), its inclusive 50 % tie and
4-connectivity; /cl_nr_extract's inclusive bounds; /seg_gauss's bounds inclusive at 4524 and 17173; every
chamfer threshold and boundary convention; the dilation's grid growth at N = 1 and 15.  Three readings were
refuted and corrected: /subtract_aims stores a negative difference as -127 (stage 02 = 00 - 01 carries one
-127 voxel for every seg_gauss voxel outside the periosteal rendering: 18,436 on PFJ-6f5538_R, 591 on PFJ-411dfd_R,
0 on PFJ-0be66a, reported in info['negatives'] and all removed by stage 03, so the masks are unchanged);
-connect_boundary true joins every face-touching component (not used by Script 32 / 33); and the rank
tie-break is an unstable sort, not 'lower label first' (rank 1..1 only is used here).
OPEN: other sigma / support values of /seg_gauss.  PFJ-411dfd_R's 3 voxels are explained and implemented (2026-09-14
evening: /open's dilation half reuses the erosion's edge-inclusive mirror margin of N + 2, ipl_ops.open_; the cohort
masks are exact on 21 / 21 since); test run 20 (predictions written 2026-09-14 18:12:42, before the implementation and
before any run) was its prospective confirmation and CONFIRMED it on 2026-09-15 -- IPL matched the mechanism's own
prediction on 46 of 46 exports carrying readings, 12 of them separating it from every alternative, 11 contrast
hypotheses refuted (test-run-20 verification).  Test run 20 also pinned -continuous_at_boundary as a
PER-AXIS fill in x, y, z order (ipl_ops.dilation / .close now take it; STEP 1 uses the default, the empty border,
which is unchanged), and settled thin volumes up to one repeat of the mirror (export 'z156', 12 slices with margin
17, 0 mismatching voxels): only a mirrored margin deeper than TWICE an axis is still untested on IPL, and that is
the only case ThinVolumeWarning now fires for.  See the open_ docstring.

Volumes: dict(data (z, y, x), dim (x, y, z), pos (x, y, z)) as returned by ipldt.io.read_aim.  `grey` is the
native short greyscale AIM (its processing log supplies the density calibration); `periosteal` is the RENDERED
periosteal contour as a char volume on the gobj's grid (IPL's stage 00 = /gobj_to_aim -peel_iter 0).
cort_trab_separation_from_raw renders a raw periosteal mask raster with ipldt.contour.render_volume on the
mask's bounding-box grid first (IPL's own gobj grid may carry a margin around that box, e.g. 3 / 2 voxels in
x and y on PFJ-0be66a: the cortical mask (28) is identical by construction, the trabecular mask (29) then lives on
the tight box instead of the gobj grid; compare by global position).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields

import numpy as np

from . import ipl_ops as ops
from .contour import render_volume

STAGES = ("00_all", "01_seggauss", "02_trab0", "03_peel6", "04_inv", "05_rank1", "06_bgrm", "07_trab1",
          "08_trabrank", "09_ero3", "10_erorank", "11_dil3", "12_close15", "13_close15peel", "14_bbc",
          "15_open15", "16_corners", "17_cornero", "18_corncl", "19_cornmajor", "20_corncl2", "21_corners2",
          "22_trabadd", "23_close50", "24_close50peel", "25_slicewise", "26_cort", "27_cortslice",
          "28_cortfinal", "29_trabfinal")


@dataclass(frozen=True)
class Step1Params:
    """The parameters of Script 32 / 33 STEP 1 (names follow the evaluation script; values are IPL's literals).

    The first sixteen are the script's options that the site presets and Supplementary Table S1 name.  The last
    twelve (2026-09-26) are the script's remaining literals, made parameters so that every value of the chain can be
    changed; their defaults are the script's, so TIBIA / RADIUS and every output are unchanged.  All are plain int /
    float (IPL's own options are integers: 0 / 1 for a flag, three fields for an x y z triple), so asdict() stays
    JSON-native and round-trips exactly.  ipldt.params.Parameters wraps this class as its 'step1' stage."""
    sigma: float = 2.0            # /seg_gauss -sigma
    support: int = 3              # /seg_gauss -support
    lower_mgha: float = 500.0     # /seg_gauss -lower_in_perm_aut_al (mg HA/ccm, -unit 2)
    upper_mgha: float = 3000.0    # /seg_gauss -upper_in_perm_aut_al
    peel0: int = 6                # IPL_PEEL0: /gobj_maskaimpeel_ow -peel_iter of stages 03, 13, 24
    erode: int = 3                # /erosion -erode_distance (09)
    dilate: int = 3               # /dilation -dilate_distance (11)
    close1: int = 15              # /close -close_distance (12)
    open_: int = 15               # /open -open_distance (15)
    corner_erode: int = 3         # /erosion (17)
    corner_min: int = 200000      # IPL_MISC1_0: /cl_nr_extract -min_number (18, 20); 200000 tibia, 800 radius
    corner_dilate: int = 3        # /dilation (19)
    corner_max: int = 500000      # /cl_nr_extract -max_number (21)
    close2: int = 50              # IPL_MISC1_1: /close -close_distance (23); 50 tibia, 30 radius
    slicewise_lo: float = 50.0    # /cl_slicewise_extractow -lo_vol_fract_in_perc (25, 27)
    slicewise_up: float = 100.0   # /cl_slicewise_extractow -up_vol_fract_in_perc
    # ---- the script's remaining literals (defaults = Script 32 / 33)
    rank_first: int = 1           # /cl_ow_rank_extract -first_rank of stages 05, 08, 10
    rank_last: int = 1            # /cl_ow_rank_extract -last_rank
    rank_connect_boundary: int = 0  # /cl_ow_rank_extract -connect_boundary (0 false, 1 true; test run 18 verified)
    continuous_x: int = 0         # /dilation and /close -continuous_at_boundary cx cy cz of stages 11, 12, 19, 23
    continuous_y: int = 0         #   (0 = an empty border, 1 = the object mirrored into it; test run 20 verified)
    continuous_z: int = 0
    mask_peel: int = 0            # /gobj_maskaimpeel_ow -peel_iter of the peel-0 steps (the greyscale before 01, stage 06)
    corner_min_max_number: int = 0  # /cl_nr_extract -max_number of stages 18, 20 (0 = no upper bound)
    corner_max_min_number: int = 1  # /cl_nr_extract -min_number of stage 21
    bbc_border_x: int = 0         # /bounding_box_cut -border bx by bz of the greyscale before 01 and of stages 14, 28
    bbc_border_y: int = 0         #   (border != 0 is implemented but never observed in IPL)
    bbc_border_z: int = 0

    @property
    def continuous_at_boundary(self):
        return (self.continuous_x, self.continuous_y, self.continuous_z)

    @property
    def bbc_border(self):
        return (self.bbc_border_x, self.bbc_border_y, self.bbc_border_z)

    def record(self):
        """The JSON record of the parameters that the reports and info['params'] carry: the sixteen preset fields
        always, and a script literal (the last twelve fields) only when it differs from the script's value.  So the
        record of TIBIA / RADIUS -- of every run at the script's literals -- has exactly the keys it had before
        the literals became parameters (a report written now can be read by the release before), and
        Step1Params(**record) rebuilds the parameters exactly."""
        d = asdict(self)
        return {k: v for k, v in d.items() if k in PRESET_FIELDS or v != _LITERAL_DEFAULTS[k]}

    def __post_init__(self):
        # numpy scalars (np.float32 / np.int64 ...) -> plain Python float / int, so asdict(), info['params'],
        # info['thresholds'] and the peel-count dict keys (info['counts']['peel'][peel0]) stay JSON-native.
        # Keyed on the default's Python type (f.type is a string under `from __future__ import annotations`).
        for f in fields(self):
            object.__setattr__(self, f.name, _plain_field(f.name, getattr(self, f.name), isinstance(f.default, float)))


def _plain_field(name, value, is_float):
    """A plain Python float for the float fields; a plain int for the int fields, accepting integral floats and
    numpy integers but refusing fractional values (no Step 1 integer parameter is fractional in IPL)."""
    x = float(value)
    if not np.isfinite(x):
        raise ValueError(f"Step1Params.{name}={value!r} is not finite")
    if is_float:
        return x
    if x != int(x):
        raise ValueError(f"Step1Params.{name}={value!r} must be an integer")
    return int(x)


PRESET_FIELDS = ("sigma", "support", "lower_mgha", "upper_mgha", "peel0", "erode", "dilate", "close1", "open_",
                 "corner_erode", "corner_min", "corner_dilate", "corner_max", "close2", "slicewise_lo", "slicewise_up")
_LITERAL_DEFAULTS = {f.name: f.default for f in fields(Step1Params) if f.name not in PRESET_FIELDS}

TIBIA = Step1Params()                                    # Script 32 (verified stage by stage on PFJ-0be66a_R, test run 16)
RADIUS = Step1Params(corner_min=800, close2=30)          # Script 33 (verified stage by stage on PFJ-0be66a_R, PFJ-42293d_L,
                                                         # PFJ-6f5538_R, test run 17, 2026-09-14)


def _count(v):
    return int(np.count_nonzero(v["data"]))


def cort_trab_separation(grey, periosteal, params=TIBIA, keep_stages=False, log=None, calibration=None):
    """Script 32 STEP 1: (greyscale AIM volume, rendered periosteal contour volume) -> cortical and
    trabecular masks.

    grey        dict from ipldt.io.read_aim of the native short AIM (data int16); its 'proclog' supplies the
                density calibration unless `calibration` = dict(slope, intercept, mu_scaling) is given.
    periosteal  the rendered periosteal contour (IPL's /gobj_to_aim -peel_iter 0) as a volume on the gobj's
                grid; any non-zero value counts as inside.
    params      Step1Params (TIBIA = Script 32, RADIUS = Script 33).
    keep_stages every intermediate stage is returned under its T16 tag (about 30 char volumes).
    log         optional callable(str) receiving one line per stage.
    Raises ValueError when the periosteal contour selects no non-zero greyscale voxel (empty contour, grids
    that do not overlap) or when the greyscale carries no density calibration.

    Returns dict(cort=volume '28_cortfinal' (char 0/127 on its bounding box = CORT_MASK),
                 trab=volume '29_trabfinal' (int8 0/127 on the union grid of 00 and 28 = TRAB_MASK; the
                      output of /subtract_aims, whose -127 never occurs there because 28 is a subset of 00),
                 stages={tag: volume} (only when keep_stages; 02_trab0 is int8 and carries -127 where the
                        seg_gauss stage reaches outside the periosteal rendering),
                 info=dict(params, calibration, thresholds{lower/upper native and mgHA}, box{dim, pos of the
                           seg_gauss input aim_bbc}, counts{tag: set voxels, i.e. non-zero, -127 included},
                           negatives{tag: -127 voxels, for the int8 stages 02, 07, 16, 22, 26, 29},
                           grids{tag: (dim, pos)}, timings{tag: seconds, 'total': seconds})).
    """
    p = params
    t_start = time.time()
    info = dict(params=p.record(), counts={}, negatives={}, grids={}, timings={})
    stages = {}
    t_last = [t_start]

    def stage(tag, v):
        now = time.time()
        info["counts"][tag] = _count(v)
        if np.asarray(v["data"]).dtype.kind == "i":                     # int8 output of /subtract_aims, /add_aims
            info["negatives"][tag] = int(np.count_nonzero(np.asarray(v["data"]) < 0))
        info["grids"][tag] = (tuple(v["dim"]), tuple(v["pos"]))
        info["timings"][tag] = now - t_last[0]
        t_last[0] = now
        if keep_stages:
            stages[tag] = v
        if log is not None:
            log(f"{tag:15s} set {info['counts'][tag]:>11,d}  dim {v['dim']} pos {v['pos']}  [{info['timings'][tag]:.1f}s]")
        return v

    # 00: the rendered periosteal contour (input) and its peel masks for /gobj_maskaimpeel_ow
    all_ = stage("00_all", ops.set_value(periosteal, ops.SET, 0))
    p0 = p.mask_peel                                                 # the script's peel-0 masking steps (0)
    peel = {p0: all_ if p0 == 0 else ops.peel_gobj_render(all_, p0), p.peel0: ops.peel_gobj_render(all_, p.peel0)}
    info["counts"]["peel"] = {p0: _count(peel[p0]), p.peel0: _count(peel[p.peel0])}
    cab = p.continuous_at_boundary                                   # /dilation and /close -continuous_at_boundary
    border = p.bbc_border                                            # /bounding_box_cut -border
    rank = dict(first_rank=p.rank_first, last_rank=p.rank_last, connect_boundary=bool(p.rank_connect_boundary))

    # 01: /gobj_maskaimpeel_ow peel 0 on the greyscale, /bounding_box_cut border 0, /seg_gauss
    cal = dict(calibration) if calibration is not None else ops.calibration_from_proclog(grey)
    lower = ops.mgha_to_native(p.lower_mgha, cal["slope"], cal["intercept"], cal["mu_scaling"])
    upper = ops.mgha_to_native(p.upper_mgha, cal["slope"], cal["intercept"], cal["mu_scaling"])
    info["calibration"] = cal
    info["thresholds"] = dict(lower_native=lower, upper_native=upper, lower_mgha=p.lower_mgha, upper_mgha=p.upper_mgha)
    masked = ops.mask_by_gobj(grey, peel[p0])
    if not masked["data"].any():
        raise ValueError("cort_trab_separation: the periosteal contour selects no non-zero greyscale voxel "
                         f"(grey {tuple(grey['dim'])} @ {tuple(grey['pos'])}, periosteal {tuple(all_['dim'])} @ "
                         f"{tuple(all_['pos'])}, |periosteal| = {info['counts']['peel'][p0]:,d}): the contour is "
                         "empty or the two grids do not overlap")
    aim_bbc = ops.bounding_box_cut(masked, border)
    del masked
    info["box"] = dict(dim=aim_bbc["dim"], pos=aim_bbc["pos"])
    s01 = stage("01_seggauss", ops.seg_gauss(aim_bbc, p.sigma, p.support, lower, upper))
    del aim_bbc

    # 02..08: the trabecular region = the contour minus the (gaussian) cortex, background removed
    s02 = stage("02_trab0", ops.subtract_aims(all_, s01))
    s03 = stage("03_peel6", ops.mask_by_gobj(s02, peel[p.peel0]))
    s04 = stage("04_inv", ops.set_value(s03, 0, ops.SET))
    s05 = stage("05_rank1", ops.cl_ow_rank_extract(s04, **rank))
    s06 = stage("06_bgrm", ops.mask_by_gobj(s05, peel[p0]))
    s07 = stage("07_trab1", ops.subtract_aims(all_, s06))
    s08 = stage("08_trabrank", ops.cl_ow_rank_extract(s07, **rank))
    del s01, s02, s03, s04, s05, s06, s07

    # 09..14: erosion / component / dilation, the large close, back to the contour and its box
    s09 = stage("09_ero3", ops.erosion(s08, p.erode))
    s10 = stage("10_erorank", ops.cl_ow_rank_extract(s09, **rank))
    s11 = stage("11_dil3", ops.dilation(s10, p.dilate, continuous_at_boundary=cab))
    s12 = stage("12_close15", ops.close(s11, p.close1, continuous_at_boundary=cab))
    s13 = stage("13_close15peel", ops.mask_by_gobj(s12, peel[p.peel0]))
    s14 = stage("14_bbc", ops.bounding_box_cut(s13, border))
    del s08, s09, s10, s11, s12, s13

    # 15..22: the open, and the corners it removed (kept only if large) added back
    s15 = stage("15_open15", ops.open_(s14, p.open_))
    s16 = stage("16_corners", ops.subtract_aims(s14, s15))
    s17 = stage("17_cornero", ops.erosion(s16, p.corner_erode))
    s18 = stage("18_corncl", ops.cl_nr_extract(s17, p.corner_min, p.corner_min_max_number))
    s19 = stage("19_cornmajor", ops.dilation(s18, p.corner_dilate, continuous_at_boundary=cab))
    s20 = stage("20_corncl2", ops.cl_nr_extract(s19, p.corner_min, p.corner_min_max_number))
    s21 = stage("21_corners2", ops.cl_nr_extract(s20, p.corner_max_min_number, p.corner_max))
    s22 = stage("22_trabadd", ops.add_aims(s21, s15))
    del s14, s16, s17, s18, s19, s20, s21

    # 23..29: the second close, minimum cortical thickness, slicewise clean-up, the two masks
    s23 = stage("23_close50", ops.close(s22, p.close2, continuous_at_boundary=cab))
    s24 = stage("24_close50peel", ops.mask_by_gobj(s23, peel[p.peel0]))
    s25 = stage("25_slicewise", ops.cl_slicewise_extractow(s24, p.slicewise_lo, p.slicewise_up))
    s26 = stage("26_cort", ops.subtract_aims(all_, s25))
    s27 = stage("27_cortslice", ops.cl_slicewise_extractow(s26, p.slicewise_lo, p.slicewise_up))
    s28 = stage("28_cortfinal", ops.bounding_box_cut(s27, border))
    s29 = stage("29_trabfinal", ops.subtract_aims(all_, s28))
    del s15, s22, s23, s24, s25, s26, s27

    info["timings"]["total"] = time.time() - t_start
    return dict(cort=s28, trab=s29, stages=stages if keep_stages else None, info=info)


def cort_trab_separation_from_raw(grey, periosteal_raw, params=TIBIA, render=True, grid=None, **kw):
    """Convenience wrapper: the periosteal contour is given as a raw mask raster (any grid, non-zero = inside);
    it is cut to its bounding box and rendered with IPL's contour rules (ipldt.contour.render_volume) to
    stand in for IPL's /gobj_to_aim stage 00, then cort_trab_separation runs.  render=False uses the raster
    as it is (for a raster that already IS a rendering, e.g. IPL's CORT_MASK | TRAB_MASK).  grid=(dim, pos)
    pastes the rendering onto that grid instead of the tight box (IPL's stage 00 lives on the GOBJ file's own
    grid, 743x348x168 @ 795,94,168 for PFJ-0be66a, which carries a margin around the box); without it the
    trabecular mask comes out on the tight box (see the module docstring).
    PFJ-0be66a (raw CORT_MASK | TRAB_MASK, which equals IPL's rendering): 28 identical to the export with either
    render setting; 29 identical by position, and on IPL's grid when grid=((743, 348, 168), (795, 94, 168))."""
    box = ops.bounding_box_cut(ops.set_value(periosteal_raw, ops.SET, 0))
    inside = box["data"] != 0
    if render:
        inside = render_volume(inside)
    per = ops.mask_vol(inside, box["dim"], box["pos"])
    if grid is not None:
        dim, pos = grid
        per = ops.mask_vol(ops.on_grid(per, dim, pos) != 0, dim, pos)
    return cort_trab_separation(grey, per, params, **kw)
