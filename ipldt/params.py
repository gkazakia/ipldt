"""ipldt.params -- the one parameter model of the whole workflow (ipldt.ormir.run_pipeline, ORMIR-BQRL's run / redo /
batch, the single ipldt commands).

EVERY TUNABLE VALUE OF EVERY STAGE IS A PARAMETER, AND THE DEFAULTS ARE THE VALIDATED IPL CONFIGURATION.  With no
override the workflows compute exactly what they computed before this module existed, voxel for voxel: the Script 32
(tibia) / Script 33 (radius) presets of STEP 1, the Laplace-Hamming variant of Script 32 STEP 2 with IPL's padding
('ceil', test run 21), the SEG assembly of Script 32 lines 407-472, the dt parameters of Script 32 and the render-grid
rule of the pore cascade.  TWO DEFAULTS DEPART FROM IPL'S EVALUATION ON PURPOSE, and for each the value compared with
IPL is IPL's own (IPL_CHOICE): the workflows report Tb.Th on the whole SEG (morphometry.tbth_object 'seg'), whereas
IPL's evaluation script, and the comparison with IPL, take it on TRAB_SEG ('trab_seg'); and ipldt.ormir.run_pipeline
masks the SEG with the periosteal raster (seg.periosteal_mask 'raw'), whereas IPL's evaluation, ORMIR-BQRL and the
comparison with IPL use the rendered contour ('rendered').  Every report's statement names the departures in effect, and
choosing IPL's value is reported as IPL's own choice, not as a departure from the validated configuration.  The defaults
are READ from the module constants that the engine has always carried (ipldt.ormir.LAPLACE_EPS, CC_MIN_VOXELS_*,
ipldt.porosity.SLICE_FRACTION, ...), so the existing names, Supplementary Table S1 and the validation harness keep
working; the workflows read the parameter object, never the globals.  What was validated against IPL are the IPL-derived
stages (STEP 1, the contour rendering, the Laplace-Hamming segmentation, the SEG assembly, the dt stage, the pore
cascade); the periosteal autocontour and BMD are ORMIR-XCT's, and their defaults are ORMIR-XCT's, which were not
compared with IPL.

    from ipldt.params import Parameters
    P = Parameters.defaults("radius")                         # Script 33's preset, everything else default
    P = P.override({"lh.laplace_eps": 0.5, "step1": {"close2": 40}})
    P.non_default()                                           # ['lh.laplace_eps', 'step1.close2']
    print(P.to_json())                                        # the complete set, loadable again with --params

THE STAGES (the prefix of every name: `--set STAGE.NAME=VALUE` on the command line)

    autocontour  ORMIR-XCT's periosteal autocontour (AutocontourKnee's periosteal parameters and the component)
    calibration  the density calibration: None = the AIM's own (processing log, else the ITK header)
    step1        Script 32 / 33 STEP 1 (ipldt.step1.Step1Params: the preset values and the script literals)
    render       IPL's contour rendering (/togobj_from_aim + /gobj_to_aim)
    lh           /fft_laplace_hamming, /norm_max, /threshold (Script 32 STEP 2)
    seg          the SEG assembly (/cl_nr_extract, the gobj masks, /set_value, /add_aims)
    dt           /dt_thickness, /dt_spacing, /dt_number (ipldt.ormir.DTParams)
    morphometry  the objects, grids and maps of the dt stage, BV/TV, the voxel size of the statistics
    porosity     the cortical pore cascade (Script 32 lines 480-670) and Ct.Po
    bmd          ORMIR-XCT's bmd_masked (its masks; the calibration is the calibration stage's)
    output       the value written into the set voxels of every mask file

PRECEDENCE: the site preset (Parameters.defaults(site)) < a parameter file (--params FILE, JSON) < the command
line (the dedicated flags such as --ridge-epsilon, then --set).  A parameter file may hold only the values to change,
nested ({"lh": {"laplace_eps": 0.5}}) or dotted ({"lh.laplace_eps": 0.5}), or a COMPLETE set (as `ormir-bqrl params`
or `ipldt-pipeline --print-params` print it, a run's <base>_parameters.json or a run report).  A complete set is read
as what it is: its preset plus the values that differ from that preset's defaults (and from the defaults of the
workflow that wrote it).  So it is neutral: with no --site it reproduces its run exactly (its preset is used); an
explicit --site replaces its preset and its changes apply on top; and a set written by one workflow does not carry
that workflow's own defaults (seg.periosteal_mask) into the other.

Every value is checked where a set is built (Parameters.__post_init__): unknown names, values of the wrong type,
out of range or not among a parameter's choices are refused (ParameterError, a ValueError) -- also for blocks built
directly (Parameters().replace(morphometry=MorphometryParams(...))).  Values ipldt implements but IPL's exports
never confirmed (e.g. /seg_gauss sigma other than 2) are accepted and reported as `unverified` (and warned about with
ParameterWarning).  A workflow records, besides the complete set, the values that did NOT take effect in that
invocation because their step did not run (Resolved.in_context: e.g. autocontour.* when a periosteal contour is
given, step1.* in a compartment redo) -- listed as `not_applied`, warned about, and never counted as non-default.
"""
from __future__ import annotations

import difflib
import json
import os
import warnings
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Mapping

from . import ormir as _engine
from . import porosity as _porosity
from .contour import render as _render
from .ormir import DTParams
from .step1 import RADIUS, TIBIA, Step1Params

SCHEMA = "ipldt-parameters/1"
STAGES = ("autocontour", "calibration", "step1", "render", "lh", "seg", "dt", "morphometry", "porosity", "bmd", "output")
SITES = {"tibia": TIBIA, "radius": RADIUS}
WORKFLOWS = ("ipldt", "ormir_bqrl")
MAP_NAMES = ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH")
IPL_STAGES_TEXT = ("STEP 1, the contour rendering, the Laplace-Hamming segmentation, the SEG assembly, the dt stage "
                   "and the pore cascade")
VALIDATED_SCOPE = (f"the IPL-derived stages ({IPL_STAGES_TEXT}) were validated against IPL at these defaults, with "
                   "IPL's own value where a workflow's default departs from IPL's evaluation on purpose: Tb.Th on TRAB_SEG "
                   "(morphometry.tbth_object trab_seg; the workflows' default is the whole SEG) and the rendered periosteal "
                   "contour in the SEG assembly (seg.periosteal_mask rendered; ipldt-pipeline's default is the raw raster); "
                   "the ORMIR-XCT periosteal autocontour and BMD use ORMIR-XCT's defaults, which were not compared with IPL")
# The defaults that depart from IPL's evaluation on purpose, and IPL's own value of each: the value the comparison with
# IPL used.  A set that holds IPL's value is IPL's own choice there (listed in non_default when it is not the
# workflow's default, but not a departure from the validated configuration); a set that holds the other value says so.
IPL_CHOICE = {"morphometry.tbth_object": "trab_seg", "seg.periosteal_mask": "rendered"}
DEPARTURE = {
    "morphometry.tbth_object": ("Tb.Th is taken on the whole SEG (morphometry.tbth_object seg, the workflows' definition; "
                                "the Tb.Th compared with IPL is IPL's own, on TRAB_SEG: trab_seg)"),
    "seg.periosteal_mask": ("the SEG is masked with the periosteal raster, not its rendered contour (seg.periosteal_mask "
                            "raw; IPL's evaluation, and the comparison with IPL, use the rendered contour: rendered)"),
}


class ParameterError(ValueError):
    """An unknown parameter name, a value of the wrong type or out of range, or a malformed --set / file."""


class ParameterWarning(UserWarning):
    """A value ipldt implements but that no IPL export has confirmed, or a value that does not take
    effect because its step does not run."""


# ============================================================================================ stage dataclasses
@dataclass(frozen=True)
class AutocontourParams:
    """ORMIR-XCT's periosteal autocontour (AutocontourKnee, ORMIR-XCT 1.1.0 defaults; not an IPL command).  At these
    values the workflows call ormir_xct's autocontour() exactly as before; any other value calls
    AutocontourKnee(**peri_*).get_periosteal_mask(image in mgHA, component) directly.  The calibration of the
    HU -> mgHA conversion is the calibration stage's (default the AIM's)."""
    component: int = 1                  # the bone to outline: 1 = the largest after RelabelComponent (ORMIR's prx_mask)
    peri_s1_sigma: float = 1.5
    peri_s1_support: int = 1
    peri_s1_lower: float = 350.0        # mgHA/ccm
    peri_s1_upper: float = 10000.0
    peri_s1_radius: int = 35
    peri_s2_sigma: float = 1.5
    peri_s2_support: int = 1
    peri_s2_lower: float = 250.0
    peri_s2_upper: float = 10000.0
    peri_s2_radius: int = 10
    peri_s3_sigma: float = 1.5
    peri_s3_support: int = 1
    peri_s3_lower: float = 350.0
    peri_s3_upper: float = 10000.0
    peri_s3_radius: int = 5
    peri_s4_open_radius: int = 8
    peri_s4_close_radius: int = 16

    def knee_kwargs(self):
        """The AutocontourKnee keyword arguments (every field but component)."""
        return {k: v for k, v in asdict(self).items() if k != "component"}


@dataclass(frozen=True)
class CalibrationParams:
    """The density calibration.  None = the AIM's own: STEP 1's /seg_gauss thresholds take slope / intercept /
    mu_scaling from the processing log ('Density: slope', 'Density: intercept', 'Mu_Scaling'), else from the ITK
    ScancoImageIO header; the autocontour and BMD read the HU image of the ITK reader, converted to mgHA with the
    header's mu_water / slope / intercept.  A value given here replaces the AIM's everywhere it is used: slope,
    intercept and mu_scaling in STEP 1, the autocontour and BMD (a mu_scaling or mu_water that differs from the
    header's recomputes the HU image from the native data with the reader's rule, ipldt.ormir.calibrated_hu);
    mu_water defines the HU scale only -- the HU image the autocontour and BMD read and ORMIR-BQRL's <base>_HU.nii.gz
    -- and cancels in HU -> mgHA except for the reader's integer truncation of HU."""
    slope: float | None = None          # mgHA/ccm per (native / mu_scaling)
    intercept: float | None = None      # mgHA/ccm
    mu_scaling: float | None = None
    mu_water: float | None = None       # 1/cm


@dataclass(frozen=True)
class RenderParams:
    """IPL's contour rendering (ipldt.contour): /togobj_from_aim -curvature_smooth 1, then /gobj_to_aim."""
    min_vertices: int = _render.MIN_VERTICES     # a chain with fewer vertices after smoothing is not stored


@dataclass(frozen=True)
class LHParams:
    """Script 32 STEP 2: /bounding_box_cut -border 1 + /fill_offset_duplicate, /fft_laplace_hamming, /norm_max,
    /threshold.  The threshold is given as IPL's permille options; the short value is int(permille / 1000 x 32767)
    (475 -> 15564, the validated value)."""
    laplace_eps: float = _engine.LAPLACE_EPS
    lp_cut_off_freq: float = _engine.LP_CUT_OFF_FREQ
    hamming_amp: float = _engine.HAMMING_AMP
    norm_max: float = _engine.NORM_MAX_VALUE
    lower_permille: float = 475.0
    upper_permille: float = 1000.0
    el_size_mm: Any = None              # None = the AIM header's (x, y, z) element sizes; a scalar or an (x, y, z) triple
    pad_offset: str = _engine.LH_PAD_OFFSET
    dtype: str = _engine.LH_DTYPE
    border: str = "duplicate"           # the 1-voxel greyscale border: duplicate | none | zero

    @property
    def threshold(self):
        """The /threshold lower bound in short units: int(lower_permille / 1000 x 32767)."""
        return permille_to_short(self.lower_permille)

    @property
    def upper_threshold(self):
        return permille_to_short(self.upper_permille)


def permille_to_short(p):
    """IPL /threshold -lower_in_perm p on a /norm_max short image: int(p / 1000 x 32767) (475 -> 15564)."""
    return int(float(p) / 1000 * _engine.INT16_MAX)


@dataclass(frozen=True)
class SegParams:
    """Script 32 lines 407-472: seg = LH & periosteal; cort_seg = cl_nr_extract(seg, 35) & cortical gobj; trab_seg =
    cl_nr_extract(seg, 70) & trabecular gobj; SEG = 127 cortical / 126 trabecular."""
    cc_min_cort: int = _engine.CC_MIN_VOXELS_CORT
    cc_min_trab: int = _engine.CC_MIN_VOXELS_TRAB
    cc_max_cort: int = 0                # /cl_nr_extract -max_number (0 = no upper bound)
    cc_max_trab: int = 0
    periosteal_mask: str | None = None  # None = the workflow's: 'raw' (ipldt.run_pipeline) / 'rendered' (ORMIR-BQRL)
    peel_periosteal: int = 0            # -peel_iter of the three /gobj_maskaimpeel_ow
    peel_cort: int = 0
    peel_trab: int = 0
    order: str = "periosteal_first"     # periosteal_first (Script 32 / the 2024 script) | gobj_first (the 2022 script)
    trab_seg_mask: str = "gobj"         # gobj = TRAB_SEG cut to the trabecular contour | none (the crop not applied)
    value_cort: int = _engine.SEG_VALUE_CORT
    value_trab: int = _engine.SEG_VALUE_TRAB
    overlap: str = "trab"               # the label of a voxel in both CORT_SEG and TRAB_SEG: trab (written last) | cort


@dataclass(frozen=True)
class MorphometryParams:
    """The dt stage around the four /dt_* calls."""
    tbth_object: str = "seg"            # seg = the whole SEG (TRAB_TH, the workflows' object) | trab_seg (Scripts 32/33/34)
    tbth_grid: str = "seg"              # with trab_seg: seg = the SEG box | tight = TRAB_SEG's own box (the 2022 script)
    ctth_object: str = "cort_mask"      # cort_mask = STEP 1's CORT_MASK | periosteal_minus_trab (Script 32 STEP 2/3, Script 34)
    bvtv_object: str = "seg"            # BV of BV/TV: seg (|SEG & G_trab|) | trab_seg (|TRAB_SEG & G_trab|)
    maps: tuple = MAP_NAMES             # the maps computed (any non-empty subset)
    grid_border: int = 0                # /bounding_box_cut -border of SEG and of the cortical object before the dt
    voxel_size_mm: float | None = None  # the voxel size of the mm statistics (and mm maps); None = the header's x size


@dataclass(frozen=True)
class PoreParams:
    """The cortical pore cascade (ipldt.porosity; Script 32 lines 480-670) and Ct.Po."""
    slice_lo: float = float(_porosity.SLICE_FRACTION[0])      # /cl_slicewise_extractow -lo_vol_fract_in_perc (both passes)
    slice_up: float = float(_porosity.SLICE_FRACTION[1])      # -up_vol_fract_in_perc
    min_pore_voxels: int = _porosity.MIN_PORE_VOXELS          # /cl_nr_extract -min_number
    max_pore_voxels: int = 0                                  # -max_number (0 = no upper bound)
    low_thresh: int = _porosity.SCRIPT32_HYSTERESIS["low_thresh"]
    high_thresh: int = _porosity.SCRIPT32_HYSTERESIS["high_thresh"]
    mode: int = _porosity.SCRIPT32_HYSTERESIS["mode"]
    grow_axes: tuple = tuple(_porosity.SCRIPT32_HYSTERESIS["grow_axes"])
    marrow_rank_first: int = 1                                # /cl_rank_extract of the marrow blob
    marrow_rank_last: int = 1
    marrow_connect_boundary: bool = False
    gobj_peel: int = 0                                        # -peel_iter of the cascade's two /gobj_maskaimpeel_ow
    render_grid_margin: int = _porosity.RENDER_GRID_MARGIN
    render_grid_clip_low: int | None = 0
    grid: str = "ipl"                                         # ipl = IPL's render grid | aim = the input AIM's grid
    ct_po: str = "contour"                                    # contour: |PORE & contour| / |contour| | pore_plus_bone


@dataclass(frozen=True)
class BMDParams:
    """ORMIR-XCT's bmd_masked (not IPL); its calibration is the calibration stage's."""
    masks: str = "rendered"             # rendered = the rendered contours (G_trab, G_cort) | raw = the mask rasters


@dataclass(frozen=True)
class OutputParams:
    mask_value: int = 127               # the value of the set voxels of every written mask (PRX / CORT / TRAB / *_GOBJ / *_SEG / PORE)


STAGE_CLASSES = {"autocontour": AutocontourParams, "calibration": CalibrationParams, "step1": Step1Params,
                 "render": RenderParams, "lh": LHParams, "seg": SegParams, "dt": DTParams,
                 "morphometry": MorphometryParams, "porosity": PoreParams, "bmd": BMDParams, "output": OutputParams}


# ============================================================================================ the schema
@dataclass(frozen=True)
class Spec:
    """How one parameter is checked and documented: kind (float, int, bool, choice, int3, maps, el, float?, int?),
    bounds (lo_open: the lower bound itself is refused, for divisors), choices, the IPL option it corresponds to and
    one line of documentation."""
    kind: str
    doc: str
    ipl: str = ""
    choices: tuple = ()
    lo: float | None = None
    hi: float | None = None
    verified: tuple | None = None       # the values IPL's exports confirm; others are reported as unverified
    lo_open: bool = False


def _S(kind, doc, ipl="", choices=(), lo=None, hi=None, verified=None, lo_open=False):
    return Spec(kind, doc, ipl, tuple(choices), lo, hi, verified, lo_open)


SCHEMA_SPECS = {
    "autocontour": {
        "component": _S("int", "the bone the periosteal mask outlines: 1 = the largest connected component", "", lo=1),
        **{f"peri_s{i}_{k}": _S(t, f"AutocontourKnee peri_s{i}_{k}", "", lo=lo, lo_open=(k == "sigma")) for i in (1, 2, 3)
           for k, t, lo in (("sigma", "float", 0), ("support", "int", 1), ("lower", "float", None), ("upper", "float", None),
                            ("radius", "int", 0))},
        "peri_s4_open_radius": _S("int", "AutocontourKnee peri_s4_open_radius", lo=0),
        "peri_s4_close_radius": _S("int", "AutocontourKnee peri_s4_close_radius", lo=0),
    },
    "calibration": {
        "slope": _S("float?", "density slope (mgHA/ccm per 1/cm); None = the AIM's ('Density: slope' / ITK rescale slope); "
                    "STEP 1, the autocontour and BMD", "Density: slope", lo=0, lo_open=True),
        "intercept": _S("float?", "density intercept (mgHA/ccm); None = the AIM's; STEP 1, the autocontour and BMD",
                        "Density: intercept"),
        "mu_scaling": _S("float?", "native units per 1/cm; None = the AIM's; STEP 1, the autocontour and BMD (the HU image "
                         "is recomputed from the native data)", "Mu_Scaling", lo=0, lo_open=True),
        "mu_water": _S("float?", "mu of water (1/cm), the HU scale; None = the AIM's; the HU image the autocontour and BMD "
                       "read (recomputed) and ORMIR-BQRL's <base>_HU; cancels in HU -> mgHA but for the integer HU",
                       "Mu_Water", lo=0, lo_open=True),
    },
    "step1": {
        "sigma": _S("float", "/seg_gauss standard deviation (voxels)", "/seg_gauss -sigma", lo=0, lo_open=True, verified=(2.0,)),
        "support": _S("int", "/seg_gauss kernel half-width", "/seg_gauss -support", lo=1, verified=(3,)),
        "lower_mgha": _S("float", "/seg_gauss lower density bound (mgHA/ccm, inclusive)", "/seg_gauss -lower_in_perm_aut_al"),
        "upper_mgha": _S("float", "/seg_gauss upper density bound (inclusive)", "/seg_gauss -upper_in_perm_aut_al"),
        "peel0": _S("int", "peel of the three peeled masking steps (03, 13, 24)", "IPL_PEEL0", lo=0),
        "erode": _S("int", "erosion distance of the trabecular candidate (09)", "/erosion -erode_distance", lo=0),
        "dilate": _S("int", "dilation distance (11)", "/dilation -dilate_distance", lo=0),
        "close1": _S("int", "first closing distance (12)", "/close -close_distance", lo=0),
        "open_": _S("int", "opening distance (15)", "/open -open_distance", lo=0),
        "corner_erode": _S("int", "erosion of the opening residue (17)", "/erosion -erode_distance", lo=0),
        "corner_min": _S("int", "smallest corner component kept (18, 20)", "IPL_MISC1_0 /cl_nr_extract -min_number", lo=0),
        "corner_dilate": _S("int", "dilation of the kept corners (19)", "/dilation -dilate_distance", lo=0),
        "corner_max": _S("int", "largest corner component added back (21; 0 = no bound)", "/cl_nr_extract -max_number", lo=0),
        "close2": _S("int", "second closing distance: the minimum cortical thickness (23)", "IPL_MISC1_1 /close -close_distance", lo=0),
        "slicewise_lo": _S("float", "slice-wise lower share of the slice (%, inclusive; 25, 27)", "/cl_slicewise_extractow -lo_vol_fract_in_perc", lo=0, hi=100),
        "slicewise_up": _S("float", "slice-wise upper share (%, inclusive)", "/cl_slicewise_extractow -up_vol_fract_in_perc", lo=0, hi=100),
        "rank_first": _S("int", "first rank of the three /cl_ow_rank_extract (05, 08, 10)", "/cl_ow_rank_extract -first_rank", lo=1),
        "rank_last": _S("int", "last rank of those three", "/cl_ow_rank_extract -last_rank", lo=1),
        "rank_connect_boundary": _S("int", "1 joins every face-touching component before ranking (05, 08, 10)", "/cl_ow_rank_extract -connect_boundary", lo=0, hi=1),
        "continuous_x": _S("int", "1 mirrors the object into the x border of /dilation and /close (11, 12, 19, 23)", "-continuous_at_boundary cx", lo=0, hi=1),
        "continuous_y": _S("int", "the same along y", "-continuous_at_boundary cy", lo=0, hi=1),
        "continuous_z": _S("int", "the same along z", "-continuous_at_boundary cz", lo=0, hi=1),
        "mask_peel": _S("int", "peel of the script's peel-0 masking steps (greyscale before 01, stage 06)", "/gobj_maskaimpeel_ow -peel_iter", lo=0),
        "corner_min_max_number": _S("int", "-max_number of the corner_min extractions (18, 20; 0 = none)", "/cl_nr_extract -max_number", lo=0),
        "corner_max_min_number": _S("int", "-min_number of the corner_max extraction (21)", "/cl_nr_extract -min_number", lo=0),
        "bbc_border_x": _S("int", "/bounding_box_cut border along x (before 01, stages 14, 28)", "/bounding_box_cut -border bx", lo=0, verified=(0,)),
        "bbc_border_y": _S("int", "the same along y", "/bounding_box_cut -border by", lo=0, verified=(0,)),
        "bbc_border_z": _S("int", "the same along z", "/bounding_box_cut -border bz", lo=0, verified=(0,)),
    },
    "render": {
        "min_vertices": _S("int", "a contour chain with fewer vertices after smoothing is not stored (4..6 match IPL; 3 and "
                           "7..8 are refuted)", "/togobj_from_aim", lo=3, verified=(4, 5, 6)),
    },
    "lh": {
        "laplace_eps": _S("float", "weight of the Laplacian term of the transfer function", "/fft_laplace_hamming -laplace_eps"),
        "lp_cut_off_freq": _S("float", "radial low-pass cut-off in units of 1 / el_z", "/fft_laplace_hamming -lp_cut_off_freq",
                              lo=0, lo_open=True),
        "hamming_amp": _S("float", "window amplitude (1 = Hann)", "/fft_laplace_hamming -hamming_amp"),
        "norm_max": _S("float", "the float value mapped to 32767", "/norm_max -max", lo=0, lo_open=True),
        "lower_permille": _S("float", "lower threshold, permille of 32767 (475 -> 15564)", "/threshold -lower_in_perm", lo=0, hi=1000),
        "upper_permille": _S("float", "upper threshold, permille of 32767", "/threshold -upper_in_perm", lo=0, hi=1000),
        "el_size_mm": _S("el", "element sizes of the filter (mm): None = the AIM header's (x, y, z); a scalar or a triple", "AIM header el_size"),
        "pad_offset": _S("choice", "power-of-two mirror padding offset (ceil = IPL's rule)", "D3P_FFT_AdjustDimensionsMirror",
                         choices=_engine.LH_PAD_OFFSETS),
        "dtype": _S("choice", "FFT / filter arithmetic", "", choices=_engine.LH_DTYPES),
        "border": _S("choice", "the greyscale border before the filter: duplicate (Script 32) | none (the 2024 diaphyseal run) | zero",
                     "/bounding_box_cut -border 1 + /fill_offset_duplicate", choices=_engine.LH_BORDERS),
    },
    "seg": {
        "cc_min_cort": _S("int", "smallest 6-connected component of CORT_SEG", "/cl_nr_extract -min_number", lo=0),
        "cc_min_trab": _S("int", "smallest 6-connected component of TRAB_SEG", "/cl_nr_extract -min_number", lo=0),
        "cc_max_cort": _S("int", "largest component of CORT_SEG (0 = no bound)", "/cl_nr_extract -max_number", lo=0),
        "cc_max_trab": _S("int", "largest component of TRAB_SEG (0 = no bound)", "/cl_nr_extract -max_number", lo=0),
        "periosteal_mask": _S("choice?", "the periosteal mask of seg = LH & periosteal: raw raster | rendered contour; None = the workflow's",
                              "/gobj_maskaimpeel_ow -gobj_filename (periosteal)", choices=("raw", "rendered")),
        "peel_periosteal": _S("int", "peel of the periosteal mask", "/gobj_maskaimpeel_ow -peel_iter", lo=0),
        "peel_cort": _S("int", "peel of the cortical contour mask", "/gobj_maskaimpeel_ow -peel_iter", lo=0),
        "peel_trab": _S("int", "peel of the trabecular contour mask", "/gobj_maskaimpeel_ow -peel_iter", lo=0),
        "order": _S("choice", "periosteal_first (Script 32) | gobj_first (label the whole thresholded volume first, the 2022 script)", "",
                    choices=("periosteal_first", "gobj_first")),
        "trab_seg_mask": _S("choice", "gobj = TRAB_SEG cut to the trabecular contour | none", "/gobj_maskaimpeel_ow (trab)", choices=("gobj", "none")),
        "value_cort": _S("int", "SEG label of cortical bone", "/set_value", lo=1, hi=255),
        "value_trab": _S("int", "SEG label of trabecular bone", "/set_value", lo=1, hi=255),
        "overlap": _S("choice", "label of a voxel in both CORT_SEG and TRAB_SEG: trab (written last) | cort (IPL's /add_aims saturation)",
                      "/add_aims", choices=("trab", "cort")),
    },
    "dt": {
        "ridge_epsilon": _S("float", "sphere-centre containment tolerance (also dt_number's mid-axis)", "-ridge_epsilon"),
        "assign_epsilon": _S("float", "sphere drawing tolerance |c - x| <= D/2 + eps", "-assign_epsilon"),
        "peel_iter": _S("int", "slice-wise erosions of the contour before the centre selection (-1 = automatic = 0)", "-peel_iter", lo=-1),
        "version": _S("choice", "diameter rule version", "-version", choices=(1, 2, 3)),
        "suppress_boundary": _S("int", "accepted for completeness (no effect on IPL's outputs)", "-suppress_boundary"),
    },
    "morphometry": {
        "tbth_object": _S("choice", "Tb.Th object: seg = the whole SEG (TRAB_TH) | trab_seg (Scripts 32/33/34, TRAB_TH_old)",
                          "/dt_thickness -input", choices=("seg", "trab_seg")),
        "tbth_grid": _S("choice", "grid of Tb.Th with trab_seg: seg = the SEG box | tight = TRAB_SEG's own box", "/bounding_box_cut",
                        choices=("seg", "tight")),
        "ctth_object": _S("choice", "Ct.Th object: cort_mask (STEP 1's CORT_MASK) | periosteal_minus_trab (rendered periosteal - "
                          "rendered trabecular contour, the Script 32 STEP 2/3 and Script 34 derivation)", "/dt_thickness -input",
                          choices=("cort_mask", "periosteal_minus_trab")),
        "bvtv_object": _S("choice", "BV of BV/TV: seg | trab_seg", "", choices=("seg", "trab_seg")),
        "maps": _S("maps", "the maps computed, any of TRAB_TH, TRAB_SP, TRAB_1N, CORT_TH", ""),
        "grid_border": _S("int", "/bounding_box_cut -border of SEG and of the cortical object (the dt grids)", "/bounding_box_cut -border", lo=0),
        "voxel_size_mm": _S("float?", "voxel size of the mm statistics and mm maps; None = the header's x element size", "",
                            lo=0, lo_open=True),
    },
    "porosity": {
        "slice_lo": _S("float", "slice-wise lower share (%) of both pore passes", "/cl_slicewise_extractow -lo_vol_fract_in_perc", lo=0, hi=100),
        "slice_up": _S("float", "slice-wise upper share (%)", "/cl_slicewise_extractow -up_vol_fract_in_perc", lo=0, hi=100),
        "min_pore_voxels": _S("int", "smallest pore component", "/cl_nr_extract -min_number", lo=0),
        "max_pore_voxels": _S("int", "largest pore component (0 = no bound)", "/cl_nr_extract -max_number", lo=0),
        "low_thresh": _S("int", "hysteresis seed bound (inclusive)", "/hysteresis_threshold -low_thresh"),
        "high_thresh": _S("int", "hysteresis weak bound (inclusive)", "/hysteresis_threshold -high_thresh"),
        "mode": _S("choice", "0 = low-intensity object (verified) | 1 = high-intensity (the symmetric reading, unverified)",
                   "/hysteresis_threshold -mode", choices=(0, 1), verified=(0,)),
        "grow_axes": _S("int3", "axes the hysteresis grows along (x y z; 0 0 1 = z only)", "/hysteresis_threshold -grow_axes",
                        verified=((0, 0, 1),)),
        "marrow_rank_first": _S("int", "first rank of the marrow blob", "/cl_rank_extract -first_rank", lo=1),
        "marrow_rank_last": _S("int", "last rank of the marrow blob", "/cl_rank_extract -last_rank", lo=1),
        "marrow_connect_boundary": _S("bool", "join face-touching components before ranking the marrow blob", "/cl_rank_extract -connect_boundary"),
        "gobj_peel": _S("int", "peel of the cascade's two cortical-contour masks", "/gobj_maskaimpeel_ow -peel_iter", lo=0),
        "render_grid_margin": _S("int", "IPL's /gobj_to_aim render grid: the contour box grown by this per in-plane side", "", lo=0),
        "render_grid_clip_low": _S("int?", "low clip of the render grid (global position); None = unclipped", ""),
        "grid": _S("choice", "working grid of the cascade: ipl (IPL's render grid) | aim (the input AIM grid)", "", choices=("ipl", "aim")),
        "ct_po": _S("choice", "Ct.Po: contour = |PORE & contour| / |contour| (the result sheet) | pore_plus_bone = "
                    "|PORE| / (|PORE| + |CORT_SEG|)", "", choices=("contour", "pore_plus_bone")),
    },
    "bmd": {
        "masks": _S("choice", "the masks BMD is averaged in: rendered contours | raw mask rasters", "", choices=("rendered", "raw")),
    },
    "output": {
        "mask_value": _S("int", "value of the set voxels of the written masks", "-value_in_range", lo=1, hi=255),
    },
}

# aliases: one name that sets several fields (IPL's three-value options)
ALIASES = {"step1.continuous_at_boundary": ("step1.continuous_x", "step1.continuous_y", "step1.continuous_z"),
           "step1.bbc_border": ("step1.bbc_border_x", "step1.bbc_border_y", "step1.bbc_border_z"),
           "porosity.slice_fraction": ("porosity.slice_lo", "porosity.slice_up")}

# The workflow steps a parameter feeds, where it does not feed a step that always runs (the SEG assembly, the dt stage,
# the renderings): a value whose steps all did not run in an invocation is `not_applied` there (Resolved.in_context).
STEPS = ("autocontour", "step1", "bmd", "porosity", "masks", "hu_volume")
STEP_USERS = {"autocontour": ("autocontour",), "step1": ("step1",), "porosity": ("porosity",), "bmd": ("bmd",),
              "calibration.slope": ("step1", "autocontour", "bmd"),
              "calibration.intercept": ("step1", "autocontour", "bmd"),
              "calibration.mu_scaling": ("step1", "autocontour", "bmd", "hu_volume"),
              "calibration.mu_water": ("autocontour", "bmd", "hu_volume"),
              "output.mask_value": ("masks",)}
STEP_ABSENT = {"autocontour": "the ORMIR-XCT autocontour did not run (the periosteal contour was given or taken from the run)",
               "step1": "STEP 1 did not run (a compartment redo takes the compartments from the run and the edit)",
               "bmd": "BMD was not computed", "porosity": "the pore cascade did not run",
               "masks": "no mask file was written", "hu_volume": "this workflow writes no HU volume"}


def users_of(name):
    """The steps (STEPS) the parameter STAGE.NAME feeds, or None when it feeds a step that always runs."""
    return STEP_USERS.get(name) or STEP_USERS.get(name.split(".", 1)[0])


def _check_schema():
    for stage, cls in STAGE_CLASSES.items():
        names = [f.name for f in fields(cls)]
        spec = SCHEMA_SPECS[stage]
        missing, extra = set(names) - set(spec), set(spec) - set(names)
        if missing or extra:                                   # pragma: no cover - a programming error
            raise RuntimeError(f"ipldt.params: schema of {stage!r} out of step with {cls.__name__}: missing {sorted(missing)}, "
                               f"extra {sorted(extra)}")


_check_schema()


def names(stage=None):
    """The dotted names of every parameter (of one stage)."""
    stages = STAGES if stage is None else (stage,)
    return [f"{s}.{f.name}" for s in stages for f in fields(STAGE_CLASSES[s])]


# ============================================================================================ value coercion
_TRUE, _FALSE = {"true", "yes", "on", "1"}, {"false", "no", "off", "0"}


def _as_seq(v):
    if isinstance(v, str):
        s = v.strip().strip("[]()")
        return [x for x in s.replace(",", " ").split() if x]
    if isinstance(v, (list, tuple)):
        return list(v)
    try:                                                        # numpy arrays
        return list(v)
    except TypeError:
        return [v]


def _num(v, what, integer=False):
    if isinstance(v, bool):
        if integer:
            return int(v)
        raise ParameterError(f"{what}: expected a number, got {v!r}")
    try:
        x = float(v) if not isinstance(v, str) else float(v.strip())
    except (TypeError, ValueError):
        raise ParameterError(f"{what}: expected {'an integer' if integer else 'a number'}, got {v!r}") from None
    if x != x or x in (float("inf"), float("-inf")):
        raise ParameterError(f"{what}: {v!r} is not finite")
    if integer:
        if x != int(x):
            raise ParameterError(f"{what}: expected an integer, got {v!r}")
        return int(x)
    return x


def _is_none(v):
    return v is None or (isinstance(v, str) and v.strip().lower() in ("none", "null", ""))


def coerce(stage, name, value):
    """The value of STAGE.NAME checked and converted to its type (ParameterError otherwise)."""
    spec = SCHEMA_SPECS[stage][name]
    what = f"{stage}.{name}"
    k = spec.kind
    if k in ("float?", "int?", "choice?") and _is_none(value):
        return None
    if k in ("float", "float?"):
        x = _num(value, what)
    elif k in ("int", "int?"):
        if isinstance(value, bool) and not (spec.lo == 0 and spec.hi == 1):      # true / false only for 0 / 1 flags
            raise ParameterError(f"{what}: expected an integer, got {value!r}")
        x = _num(value, what, integer=True)
    elif k == "bool":
        if isinstance(value, bool):
            x = value
        elif isinstance(value, (int, float)) and value in (0, 1):
            x = bool(value)
        elif isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
            x = value.strip().lower() in _TRUE
        else:
            raise ParameterError(f"{what}: expected true / false, got {value!r}")
        return x
    elif k in ("choice", "choice?"):
        opts = spec.choices
        if opts and isinstance(opts[0], int):
            x = _num(value, what, integer=True)
        else:
            x = str(value).strip() if not isinstance(value, str) else value.strip()
        if x not in opts:
            raise ParameterError(f"{what}: {value!r} is not one of {', '.join(str(o) for o in opts)}")
        return x
    elif k == "int3":
        seq = _as_seq(value)
        if len(seq) != 3:
            raise ParameterError(f"{what}: expected three integers (x y z), got {value!r}")
        x = tuple(_num(s, what, integer=True) for s in seq)
        return x
    elif k == "maps":
        seq = [str(s).strip().upper() for s in _as_seq(value)]
        bad = [s for s in seq if s not in MAP_NAMES]
        if bad or not seq:
            raise ParameterError(f"{what}: expected a non-empty list of {', '.join(MAP_NAMES)}, got {value!r}")
        return tuple(m for m in MAP_NAMES if m in seq)          # canonical order
    elif k == "el":
        if _is_none(value):
            return None
        seq = _as_seq(value)
        if len(seq) == 1:
            x = _num(seq[0], what)
            if x <= 0:
                raise ParameterError(f"{what}: must be > 0, got {value!r}")
            return x
        if len(seq) != 3:
            raise ParameterError(f"{what}: expected None, one element size or three (x y z), got {value!r}")
        xs = tuple(_num(s, what) for s in seq)
        if min(xs) <= 0:
            raise ParameterError(f"{what}: element sizes must be > 0, got {value!r}")
        return xs
    else:                                                        # pragma: no cover
        raise RuntimeError(f"unknown kind {k!r}")
    if spec.lo is not None and (x < spec.lo or (spec.lo_open and x <= spec.lo)):
        raise ParameterError(f"{what}: must be {'>' if spec.lo_open else '>='} {spec.lo:g}, got {value!r}")
    if spec.hi is not None and x > spec.hi:
        raise ParameterError(f"{what}: must be <= {spec.hi:g}, got {value!r}")
    return x


def _suggest(word, candidates):
    m = difflib.get_close_matches(word, list(candidates), n=1, cutoff=0.5)
    return f" (did you mean {m[0]!r}?)" if m else ""


def split_name(dotted):
    """'lh.laplace_eps' -> ('lh', 'laplace_eps'), with a helpful ParameterError for unknown names."""
    if not isinstance(dotted, str) or "." not in dotted:
        raise ParameterError(f"a parameter is named STAGE.NAME (e.g. lh.laplace_eps), got {dotted!r}; stages: {', '.join(STAGES)}")
    stage, name = dotted.split(".", 1)
    stage, name = stage.strip(), name.strip()
    if stage not in STAGE_CLASSES:
        raise ParameterError(f"unknown parameter stage {stage!r} in {dotted!r}{_suggest(stage, STAGES)}; valid stages: "
                             f"{', '.join(STAGES)}")
    valid = [f.name for f in fields(STAGE_CLASSES[stage])]
    aliases = [a.split(".", 1)[1] for a in ALIASES if a.startswith(stage + ".")]
    if name not in valid and f"{stage}.{name}" not in ALIASES:
        raise ParameterError(f"unknown parameter {dotted!r}{_suggest(name, valid + aliases)}; valid names for stage {stage!r}: "
                             f"{', '.join(valid + aliases)}")
    return stage, name


def flatten_overrides(mapping, _prefix=""):
    """A nested ({'lh': {'laplace_eps': 0.5}}), dotted ({'lh.laplace_eps': 0.5}) or mixed mapping -> an ordered
    {dotted name: raw value} dict (aliases expanded, names checked, values NOT yet coerced)."""
    out = {}
    if not isinstance(mapping, Mapping):
        raise ParameterError(f"parameter overrides must be a mapping, got {type(mapping).__name__}")
    for key, value in mapping.items():
        key = str(key)
        dotted = f"{_prefix}{key}" if _prefix else key
        if not _prefix and "." not in key:
            if key not in STAGE_CLASSES:
                raise ParameterError(f"unknown parameter stage {key!r}{_suggest(key, STAGES)}; valid stages: {', '.join(STAGES)}")
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ParameterError(f"stage {key!r} must map names to values, got {value!r}")
            out.update(flatten_overrides(value, key + "."))
            continue
        stage, name = split_name(dotted)
        full = f"{stage}.{name}"
        if full in ALIASES:
            parts = ALIASES[full]
            seq = _as_seq(value)
            if len(seq) != len(parts):
                raise ParameterError(f"{full}: expected {len(parts)} values, got {value!r}")
            for p, v in zip(parts, seq):
                out[p] = v
        else:
            out[full] = value
    return out


def _checked(mapping):
    """flatten_overrides + coerce: {dotted name: checked value}."""
    return {k: coerce(*k.split(".", 1), v) for k, v in flatten_overrides(mapping).items()}


def parse_set(item):
    """One --set STAGE.NAME=VALUE -> {dotted name: checked value} (an alias gives several); VALUE may be JSON (0.5, true, null, [0, 0, 1]) or a bare
    word; it is type-checked against the parameter's kind."""
    if not isinstance(item, str) or "=" not in item:
        raise ParameterError(f"--set expects STAGE.NAME=VALUE (e.g. --set lh.laplace_eps=0.5), got {item!r}")
    key, raw = item.split("=", 1)
    key, raw = key.strip(), raw.strip()
    if "." not in key:
        raise ParameterError(f"--set expects STAGE.NAME=VALUE (e.g. --set lh.laplace_eps=0.5), got {item!r}; stages: "
                             f"{', '.join(STAGES)}")
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        value = raw
    return _checked({key: value})                    # type-checked now: a bad --set fails early


# ============================================================================================ parameter files
@dataclass(frozen=True)
class ParamsFile:
    """What a parameter file (or a mapping) holds: the checked overrides and, for a complete set, the preset and the
    workflow it records (the overrides are then the values that differ from that preset's / workflow's defaults)."""
    overrides: dict
    complete: bool = False
    preset: str | None = None
    workflow: str | None = None
    path: str | None = None

    def describe(self):
        where = f"parameter file {self.path}" if self.path else "a parameter mapping"
        if not self.complete:
            return f"{where} ({len(self.overrides)} value(s))"
        return (f"{where} (a complete set of the {self.preset} preset"
                + (f", written by {self.workflow}" if self.workflow else "")
                + f": its {len(self.overrides)} value(s) that differ from that preset's defaults)")


def complete_set_block(data):
    """The mapping that holds a complete set's 'values' block -- a set printed by `ormir-bqrl params` /
    `ipldt-pipeline --print-params`, a <base>_parameters.json, or a run report's 'parameter_set' -- else None."""
    if isinstance(data, Mapping) and isinstance(data.get("parameter_set"), Mapping):
        data = data["parameter_set"]
    if isinstance(data, Mapping) and isinstance(data.get("values"), Mapping):
        return data
    return None


def read_params(data, path=None):
    """A parameter mapping (a file's JSON content) -> ParamsFile.  A complete set is read as its preset plus the
    values that differ from the defaults of that preset and of the workflow that wrote it (so it is neutral: it
    carries neither the preset over an explicit site nor a workflow's own defaults into the other workflow);
    anything else is a mapping of the values to change."""
    meta = complete_set_block(data)
    if meta is None:
        return ParamsFile(overrides=_checked(data), path=path)
    flat = _checked(meta["values"])
    preset = meta.get("preset") if meta.get("preset") in SITES else (meta.get("site") if meta.get("site") in SITES else None)
    if preset is None:
        s = site_of(Parameters.defaults("tibia").override(flat).step1)
        preset = s if s in SITES else "tibia"
    wf = meta.get("workflow") if meta.get("workflow") in WORKFLOWS else None
    full = Parameters.defaults(preset).override(flat)
    ref = Parameters.defaults(preset, wf) if wf else Parameters.defaults(preset)
    return ParamsFile(overrides={k: full.get(k) for k in full.diff(ref)}, complete=True, preset=preset, workflow=wf, path=path)


def read_params_file(path):
    """A parameter file (JSON) -> ParamsFile (read_params).  Accepted: a nested or dotted mapping of the values to
    change, the complete set as `ormir-bqrl params` / `ipldt-pipeline --print-params` print it, a run's
    <base>_parameters.json, or a run report (its 'parameter_set' block)."""
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise ParameterError(f"parameter file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        raise ParameterError(f"{path}: parameter files are JSON (YAML is not a dependency of ipldt)")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except ValueError as exc:
        raise ParameterError(f"{path}: not valid JSON ({exc})") from None
    try:
        return read_params(data, path=os.path.abspath(path))
    except ParameterError as exc:
        raise ParameterError(f"{path}: {exc}") from None


def load_params_file(path):
    """A parameter file -> {dotted name: checked value}: the values to change (for a complete set, those that differ
    from the defaults of the preset it records; read_params_file returns that preset too)."""
    return read_params_file(path).overrides


# ============================================================================================ the container
def _plain(v):
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    return v


def _normalised(stage, blk):
    """A stage block with every field coerce()d (type, range, choices); the same object when nothing changes."""
    cls = STAGE_CLASSES[stage]
    if not isinstance(blk, cls):
        raise ParameterError(f"stage {stage!r} must be a {cls.__name__}, not {type(blk).__name__}")
    vals = {f.name: getattr(blk, f.name) for f in fields(blk)}
    new = {n: coerce(stage, n, v) for n, v in vals.items()}
    if all(type(new[n]) is type(vals[n]) and new[n] == vals[n] for n in vals):
        return blk
    return cls(**new)


@dataclass(frozen=True)
class Parameters:
    """The complete parameter set of the workflow, one frozen block per stage (STAGES).  Build it with
    Parameters.defaults(site) (the validated IPL configuration) and change it with .override(mapping) or
    .replace(stage=block); every workflow accepts it as `parameters=` (ORMIR-BQRL's run / run_from_masks,
    ipldt.ormir.run_pipeline), as do dicts of overrides and JSON file paths.  Every field of every block is checked
    when the set is built (a block built directly with a wrong type, a value out of range or not among the
    choices is refused), and so are the cross-field rules (validate)."""
    autocontour: AutocontourParams = field(default_factory=AutocontourParams)
    calibration: CalibrationParams = field(default_factory=CalibrationParams)
    step1: Step1Params = field(default_factory=lambda: TIBIA)
    render: RenderParams = field(default_factory=RenderParams)
    lh: LHParams = field(default_factory=LHParams)
    seg: SegParams = field(default_factory=SegParams)
    dt: DTParams = field(default_factory=DTParams)
    morphometry: MorphometryParams = field(default_factory=MorphometryParams)
    porosity: PoreParams = field(default_factory=PoreParams)
    bmd: BMDParams = field(default_factory=BMDParams)
    output: OutputParams = field(default_factory=OutputParams)

    def __post_init__(self):
        for s in STAGES:
            blk = getattr(self, s)
            try:
                new = _normalised(s, blk)
            except ParameterError:
                raise
            except (TypeError, ValueError) as exc:                  # e.g. Step1Params' own checks
                raise ParameterError(f"stage {s!r}: {exc}") from None
            if new is not blk:
                object.__setattr__(self, s, new)
        self.validate()

    # ---------------------------------------------------------------------------------------- construction
    @classmethod
    def defaults(cls, site="tibia", workflow=None):
        """The validated IPL configuration: the site's STEP 1 preset ('tibia' = Script 32, 'radius' = Script 33) and
        every other stage at its default; with `workflow` ('ipldt' or 'ormir_bqrl') the workflow-specific default
        (seg.periosteal_mask) filled in."""
        key = str(site).lower()
        if key not in SITES:
            raise ParameterError(f"site must be one of {', '.join(SITES)}, not {site!r}")
        p = cls(step1=SITES[key])
        return p.for_workflow(workflow) if workflow else p

    def for_workflow(self, workflow):
        """The set with the workflow's own defaults filled in where the value is None (seg.periosteal_mask)."""
        if workflow not in WORKFLOWS:
            raise ParameterError(f"workflow must be one of {WORKFLOWS}, not {workflow!r}")
        if self.seg.periosteal_mask is None:
            return replace(self, seg=replace(self.seg, periosteal_mask="raw" if workflow == "ipldt" else "rendered"))
        return self

    @classmethod
    def from_dict(cls, mapping, site=None):
        """The site's defaults (default tibia, or the preset a complete set records) with a (nested / dotted /
        complete) mapping applied."""
        pf = read_params(mapping)
        return cls.defaults(site or pf.preset or "tibia").override(pf.overrides)

    @classmethod
    def from_file(cls, path, site=None):
        """The site's defaults (default tibia, or the preset a complete file records) with the file applied."""
        pf = read_params_file(path)
        return cls.defaults(site or pf.preset or "tibia").override(pf.overrides)

    def override(self, mapping):
        """A new set with the values of `mapping` (nested, dotted or mixed; aliases allowed) applied and checked."""
        flat = flatten_overrides(mapping or {})
        blocks = {s: getattr(self, s) for s in STAGES}
        changes = {}
        for dotted, value in flat.items():
            s, n = dotted.split(".", 1)
            changes.setdefault(s, {})[n] = coerce(s, n, value)
        for s, ch in changes.items():
            blk = blocks[s]
            if isinstance(blk, DTParams):
                blocks[s] = DTParams(**{**blk.kwargs(), **ch})
            else:
                blocks[s] = replace(blk, **ch)
        return Parameters(**blocks)

    def replace(self, **blocks):
        """dataclasses.replace for whole blocks (e.g. P.replace(step1=RADIUS)); every field is checked."""
        for k, v in blocks.items():
            if k not in STAGE_CLASSES:
                raise ParameterError(f"unknown stage {k!r}; valid stages: {', '.join(STAGES)}")
            if not isinstance(v, STAGE_CLASSES[k]):
                raise ParameterError(f"stage {k!r} must be a {STAGE_CLASSES[k].__name__}, not {type(v).__name__}")
        return replace(self, **blocks)

    # ---------------------------------------------------------------------------------------- checks
    def validate(self):
        """Cross-field checks (each field's own type, range and choices are checked by coerce when the set is built)."""
        s1, po, m, lh = self.step1, self.porosity, self.morphometry, self.lh
        errs = []
        if s1.slicewise_lo > s1.slicewise_up:
            errs.append(f"step1.slicewise_lo {s1.slicewise_lo:g} > step1.slicewise_up {s1.slicewise_up:g}")
        if s1.rank_first > s1.rank_last:
            errs.append(f"step1.rank_first {s1.rank_first} > step1.rank_last {s1.rank_last}")
        if po.slice_lo > po.slice_up:
            errs.append(f"porosity.slice_lo {po.slice_lo:g} > porosity.slice_up {po.slice_up:g}")
        if po.low_thresh > po.high_thresh:
            errs.append(f"porosity.low_thresh {po.low_thresh} > porosity.high_thresh {po.high_thresh}")
        if po.marrow_rank_first > po.marrow_rank_last:
            errs.append(f"porosity.marrow_rank_first {po.marrow_rank_first} > porosity.marrow_rank_last {po.marrow_rank_last}")
        if any(int(a) not in (0, 1) for a in po.grow_axes):
            errs.append(f"porosity.grow_axes must be 0 / 1 flags, got {po.grow_axes}")
        if lh.lower_permille > lh.upper_permille:
            errs.append(f"lh.lower_permille {lh.lower_permille:g} > lh.upper_permille {lh.upper_permille:g}")
        if m.tbth_grid == "tight" and m.tbth_object != "trab_seg":
            errs.append("morphometry.tbth_grid 'tight' applies to morphometry.tbth_object 'trab_seg' only")
        if not m.maps:
            errs.append("morphometry.maps is empty")
        if self.seg.value_cort == self.seg.value_trab:
            errs.append("seg.value_cort and seg.value_trab must differ")
        if errs:
            raise ParameterError("; ".join(errs))
        return self

    def unverified(self):
        """Human-readable notes on the values no IPL export confirms (empty for the defaults)."""
        notes = []
        for s in STAGES:
            blk = getattr(self, s)
            for n, spec in SCHEMA_SPECS[s].items():
                if spec.verified is None:
                    continue
                v = getattr(blk, n)
                if v not in spec.verified:
                    notes.append(f"{s}.{n} = {_fmt(v)}: "
                                 f"{'not IPL-verified' if s != 'render' else 'differs from IPL rendering (4..6 match; 3 and 7..8 refuted)'}"
                                 f" (verified: {', '.join(_fmt(x) for x in spec.verified)})")
        if self.lh.border == "zero":
            notes.append("lh.border = zero: refuted against IPL (kept for comparison)")
        return notes

    def warn_unverified(self):
        for n in self.unverified():
            warnings.warn(n, ParameterWarning, stacklevel=3)

    # ---------------------------------------------------------------------------------------- views
    def to_dict(self):
        """{stage: {name: value}} with JSON-native values (tuples as lists)."""
        out = {}
        for s in STAGES:
            blk = getattr(self, s)
            out[s] = {f.name: _plain(getattr(blk, f.name)) for f in fields(blk)}
        return out

    def flat(self):
        return {f"{s}.{n}": v for s, d in self.to_dict().items() for n, v in d.items()}

    def get(self, dotted):
        s, n = split_name(dotted)
        return getattr(getattr(self, s), n)

    def diff(self, other):
        """The dotted names whose values differ from `other`'s, in schema order."""
        a, b = self.flat(), other.flat()
        return [k for k in a if a[k] != b[k]]

    def non_default(self, site=None, workflow=None):
        """The names that differ from Parameters.defaults(site, workflow); site defaults to the preset the step1 block
        matches (tibia when it matches none).  Pass the workflow for a set a workflow resolved (its seg.periosteal_mask
        is then filled in)."""
        site = site or (site_of(self.step1) if site_of(self.step1) != "custom" else "tibia")
        return self.diff(Parameters.defaults(site, workflow))

    def to_json(self, **meta):
        return json.dumps({"schema": SCHEMA, **meta, "values": self.to_dict()}, indent=1)

    def describe(self, reference=None):
        """A text table of every parameter: name, value, default (when it differs), IPL option, documentation."""
        ref = reference.flat() if reference is not None else Parameters.defaults(site_of(self.step1) if site_of(self.step1) != "custom" else "tibia").flat()
        rows = []
        for s in STAGES:
            for n, spec in SCHEMA_SPECS[s].items():
                v = _plain(getattr(getattr(self, s), n))
                d = ref.get(f"{s}.{n}")
                mark = "*" if v != d else " "
                rows.append((f"{mark} {s}.{n}", _fmt(v), "" if v == d else f"(default {_fmt(d)})", spec.ipl, spec.doc))
        w0 = max(len(r[0]) for r in rows)
        w1 = max(len(r[1]) for r in rows)
        lines = [f"{r[0]:<{w0}}  {r[1]:<{w1}}  {r[2]}{'  ' if r[2] else ''}{r[3] + ': ' if r[3] else ''}{r[4]}" for r in rows]
        return "\n".join(lines)


def _fmt(v):
    if v is None:
        return "None"
    if isinstance(v, (list, tuple)):
        return " ".join(_fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def site_of(step1):
    """'tibia' / 'radius' when a Step1Params equals that preset, else 'custom'."""
    for name, p in SITES.items():
        if step1 == p:
            return name
    return "custom"


# ============================================================================================ resolution
@dataclass(frozen=True)
class Resolved:
    """What a workflow runs with: the complete set, the site name it reports, the preset it is compared with and
    where the values came from; after in_context also the steps that ran, the values that did not take effect
    because their step did not run (not_applied: (name, requested, used, reason) tuples; the set holds the value
    used) and the AIM header's element sizes (an explicit voxel / element size equal to the header's is the
    default's value and is not counted as non-default)."""
    params: Parameters
    site: str                   # 'tibia' | 'radius' | 'custom' (the report's site)
    preset: str                 # the preset the non-default list is taken against
    workflow: str
    sources: tuple
    steps: frozenset | None = None
    not_applied: tuple = ()
    header_el: tuple | None = None

    @property
    def reference(self):
        return Parameters.defaults(self.preset, self.workflow)

    @property
    def equal_to_default(self):
        """The names given explicitly with the value their default (None = the AIM header's) resolves to on this AIM:
        morphometry.voxel_size_mm equal to the header's x element size, lh.el_size_mm equal to the header's (x, y, z)."""
        if self.header_el is None:
            return ()
        el = tuple(float(e) for e in self.header_el)
        ref, P, out = self.reference, self.params, []
        v = P.morphometry.voxel_size_mm
        if v is not None and ref.morphometry.voxel_size_mm is None and float(v) == el[0]:
            out.append("morphometry.voxel_size_mm")
        e = P.lh.el_size_mm
        if e is not None and ref.lh.el_size_mm is None:
            t = tuple(float(x) for x in e) if isinstance(e, (tuple, list)) else (float(e),) * 3
            if t == el:
                out.append("lh.el_size_mm")
        return tuple(out)

    @property
    def non_default(self):
        eq = set(self.equal_to_default)
        return [k for k in self.params.diff(self.reference) if k not in eq]

    @property
    def is_default(self):
        return not self.non_default

    @property
    def statement(self):
        return statement(self.non_default, self.preset, self.steps, self.params.flat())

    def in_context(self, steps, baseline=None, header_el=None, baseline_label="the default"):
        """The set as this invocation runs it: `steps` are the steps that run (STEPS); a value of a step that does
        not run is not applied -- it is reset to `baseline`'s (default: the reference defaults; a redo passes the
        run's set) and recorded in not_applied -- when it differs from the baseline.  header_el: the AIM header's
        element sizes (equal_to_default)."""
        steps = frozenset(steps)
        bad = steps - set(STEPS)
        if bad:                                                       # pragma: no cover - a programming error
            raise ValueError(f"unknown steps {sorted(bad)}")
        base = baseline if baseline is not None else self.reference
        P = self.params
        na, reset = [], {}
        for name in P.diff(base):
            users = users_of(name)
            if users and not (set(users) & steps):
                why = "; ".join(STEP_ABSENT[u] for u in users)
                na.append((name, _plain(P.get(name)), _plain(base.get(name)), f"{why}; {baseline_label} is used"))
                reset[name] = base.get(name)
        if reset:
            P = P.override(reset)
        return replace(self, params=P, steps=steps, not_applied=tuple(na),
                       header_el=None if header_el is None else tuple(float(e) for e in header_el))

    def not_applied_notes(self):
        return [f"{n} = {_fmt(v)} not applied: {why} ({_fmt(u)})" for n, v, u, why in self.not_applied]

    def warn_not_applied(self):
        for n in self.not_applied_notes():
            warnings.warn(n, ParameterWarning, stacklevel=3)

    def block(self):
        """The report's parameter_set block."""
        ref = self.reference.flat()
        nd = self.non_default
        flat = self.params.flat()
        return {"schema": SCHEMA, "workflow": self.workflow, "site": self.site, "preset": self.preset,
                "validated_default": not nd,
                "statement": self.statement,
                "validated_scope": VALIDATED_SCOPE,
                "non_default": list(nd),
                "non_default_values": {k: {"value": flat[k], "default": ref[k]} for k in nd},
                "not_applied": [n for n, *_ in self.not_applied],
                "not_applied_values": {n: {"requested": v, "used": u, "reason": why} for n, v, u, why in self.not_applied},
                "equal_to_default": list(self.equal_to_default),
                "unverified": self.params.unverified(),
                "sources": list(self.sources),
                "values": self.params.to_dict()}


def statement(non_default, preset="tibia", steps=None, values=None):
    """The one line every report carries about its configuration: whether every value of the IPL-derived stages is
    the one validated against IPL (and which differ when not), which of the workflows' deliberate departures from
    IPL's evaluation are in effect (DEPARTURE; a value set to IPL's own choice, IPL_CHOICE, is named as that and is
    not a departure), and what the non-IPL stages were (the periosteal contour: ORMIR-XCT's autocontour or given; BMD:
    ORMIR-XCT's) -- neither was validated against IPL.  values: the effective set, {dotted: value} (Parameters.flat());
    None = the preset's defaults for every name not in non_default."""
    script = "Script 33 radius" if preset == "radius" else "Script 32 tibia"
    known = values is not None
    vals = dict(values) if known else Parameters.defaults(preset if preset in SITES else "tibia").flat()
    ipl = [n for n in non_default if known and n in IPL_CHOICE and vals.get(n) == IPL_CHOICE[n]]
    other = [n for n in non_default if n not in ipl]
    dep = [DEPARTURE[n] for n, v in IPL_CHOICE.items()
           if (known or n not in non_default) and vals.get(n) not in (None, v)
           and (n != "morphometry.tbth_object" or "TRAB_TH" in (vals.get("morphometry.maps") or MAP_NAMES))]
    listed = ", ".join(f"{n} ({_fmt(vals.get(n))}: IPL's own choice, the one compared with IPL)" if n in ipl else n
                       for n in non_default)
    if not non_default:
        head = (f"the validated IPL defaults ({script} preset): every value of the IPL-derived stages ({IPL_STAGES_TEXT}) "
                "is the one validated against IPL")
    elif not other:
        head = (f"differs from the validated IPL defaults ({script} preset) in {len(non_default)} value(s): {listed}; "
                f"every other value of the IPL-derived stages ({IPL_STAGES_TEXT}) is the one validated against IPL")
    else:
        head = (f"differs from the validated IPL defaults ({script} preset) in {len(non_default)} value(s): "
                f"{listed}; results are not the validated IPL-equivalent configuration")
    if dep:
        head += ("; " if other else ", except that ") + " and ".join(dep)
    if steps is None:
        steps = ("autocontour", "step1", "bmd")
    ac, bmd = "autocontour" in steps, "bmd" in steps
    if ac:
        tail = (f"the periosteal autocontour {'and BMD are' if bmd else 'is'} ORMIR-XCT's, "
                f"which {'were' if bmd else 'was'} not validated against IPL")
    else:
        tail = ("the periosteal contour was given (no autocontour)" if "step1" in steps else
                "the periosteal contour is the run's or the edit's (no autocontour)")
        if bmd:
            tail += "; BMD is ORMIR-XCT's, which was not validated against IPL"
    return f"{head}; {tail}"


def resolve(site=None, parameters=None, *, step1_params=None, dt_params=None, workflow="ipldt", overrides=None,
            source_label=None):
    """The complete parameter set of a workflow call.

    site          the STEP 1 preset the set starts from ('tibia' | 'radius'); None = the preset a complete parameter
                  file / mapping records, else 'tibia'
    parameters    None; a Parameters (complete: it replaces the preset); a mapping of overrides (nested or dotted) or
                  a complete set (as printed / saved: its preset plus its changes, read_params); or the path of a JSON
                  parameter file (either form)
    step1_params  the legacy explicit Step1Params (wins over both, reported as site 'custom' as before)
    dt_params     the legacy explicit DTParams (wins over both)
    overrides     further {dotted: value} applied last (the workflows' dedicated keyword arguments)
    source_label  where `parameters` came from, recorded in `sources` (default: a command line's
                  CommandLineOverrides.source, else a description of the file or mapping)
    Returns Resolved."""
    if workflow not in WORKFLOWS:
        raise ParameterError(f"workflow must be one of {WORKFLOWS}, not {workflow!r}")
    source_label = source_label or getattr(parameters, "source", None)     # a command line's CommandLineOverrides
    explicit = site is not None
    base = str(site).lower() if explicit else "tibia"
    if base not in SITES:
        raise ParameterError(f"site must be one of {', '.join(SITES)}, not {site!r}")
    pf = None
    if isinstance(parameters, (str, os.PathLike)):
        pf = read_params_file(parameters)
    elif isinstance(parameters, Mapping):
        pf = read_params(parameters)
    elif parameters is not None and not isinstance(parameters, Parameters):
        raise ParameterError(f"parameters must be None, a Parameters, a mapping or a JSON file path, not {type(parameters).__name__}")
    if pf is not None and pf.complete and not explicit:
        base = pf.preset
    P = Parameters.defaults(base)
    sources = [f"site preset {base!r}" + (" (recorded by the complete parameter set)" if pf is not None and pf.complete
                                           and not explicit else "")]
    if isinstance(parameters, Parameters):
        P = replace(parameters)                   # rebuilt, so every field is checked again (DTParams is mutable)
        sources = [source_label or "a complete Parameters object"]
    elif pf is not None:
        if pf.overrides or pf.complete or pf.path:
            P = P.override(pf.overrides)
            sources.append(source_label or pf.describe())
    if overrides:
        P = P.override(overrides)
        sources.append("keyword arguments: " + ", ".join(sorted(overrides)))
    legacy_custom = False
    if step1_params is not None:
        if not isinstance(step1_params, Step1Params):
            raise TypeError(f"params must be an ipldt.step1.Step1Params, not {type(step1_params).__name__}")
        P = P.replace(step1=step1_params)
        sources.append("step1_params")
        legacy_custom = True
    if dt_params is not None:
        if not isinstance(dt_params, DTParams):
            raise TypeError(f"dt params must be an ipldt.ormir.DTParams, not {type(dt_params).__name__}")
        P = P.replace(dt=dt_params)
        sources.append("dt_params")
    P = P.for_workflow(workflow)
    name = "custom" if legacy_custom else site_of(P.step1)
    preset = name if name in SITES else base
    return Resolved(params=P, site=name, preset=preset, workflow=workflow, sources=tuple(sources))


def write_params_file(path, resolved, **meta):
    """<base>_parameters.json: the complete effective set (loadable with --params, where it is read as its preset plus
    its changes), its site, preset, non-default list and the values not applied."""
    b = resolved.block()
    doc = {"schema": SCHEMA, **meta, "workflow": b["workflow"], "site": b["site"], "preset": b["preset"],
           "validated_default": b["validated_default"], "non_default": b["non_default"]}
    if b["not_applied"]:
        doc["not_applied"] = b["not_applied_values"]
    doc["values"] = b["values"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    return os.path.abspath(path)


def printed_set(resolved):
    """The complete set as `ormir-bqrl params` / `ipldt-pipeline --print-params` print it (loadable with --params)."""
    b = resolved.block()
    return {"schema": SCHEMA, "workflow": b["workflow"], "site": b["site"], "preset": b["preset"],
            "validated_default": b["validated_default"], "non_default": b["non_default"], "values": b["values"]}


# ============================================================================================ argparse glue
def add_parameter_arguments(parser, sets_help_extra="", stages=None, dedicated_flags=False):
    """--params FILE and repeatable --set STAGE.NAME=VALUE.  stages: the stages the command takes (None = all; the
    single dt commands pass ('dt',)); dedicated_flags: the command has shortcut flags for some parameters (their
    place in the precedence)."""
    if stages is not None and tuple(stages) == ("dt",):
        g = parser.add_argument_group("dt parameters (defaults: Script 32's; precedence: --params file < the dt flags "
                                      "< --set)")
        g.add_argument("--params", metavar="FILE", default=None,
                       help="JSON parameter file: the dt values to change ({\"dt\": {\"ridge_epsilon\": 0.7}} or "
                            "{\"dt.ridge_epsilon\": 0.7}), or a complete set (printed by `ormir-bqrl params` / "
                            "`ipldt-pipeline --print-params`, a run's <base>_parameters.json or report), of which this "
                            "command takes the dt block")
        g.add_argument("--set", dest="set_params", metavar="dt.NAME=VALUE", action="append", default=[],
                       help="override one dt parameter (repeatable), e.g. --set dt.ridge_epsilon=0.7 --set dt.version=2; "
                            "names: " + ", ".join(names("dt")) + sets_help_extra)
        return g
    order = "site preset < --params file < " + ("dedicated flags < " if dedicated_flags else "") + "--set"
    g = parser.add_argument_group(f"parameters (defaults: the validated IPL configuration apart from the departures every "
                                  f"report names; precedence: {order})")
    g.add_argument("--params", metavar="FILE", default=None,
                   help="JSON parameter file: the values to change (nested {\"lh\": {\"laplace_eps\": 0.5}} or dotted "
                        "{\"lh.laplace_eps\": 0.5}), or a complete set as printed by the params command / written as "
                        "<base>_parameters.json, or a run report (a complete set is its preset plus its changes: without "
                        "--site its preset is used, with --site its changes apply to that site's preset)")
    g.add_argument("--set", dest="set_params", metavar="STAGE.NAME=VALUE", action="append", default=[],
                   help="override one parameter (repeatable), e.g. --set lh.laplace_eps=0.5 --set step1.close2=40 "
                        "--set porosity.grow_axes=0,0,1; stages: " + ", ".join(STAGES) + sets_help_extra)
    return g


@dataclass(frozen=True)
class ArgParameters:
    """The parameters of a command line: the overrides in precedence order and the preset a complete --params file
    records (None when there is none)."""
    overrides: dict
    preset: str | None = None
    file: ParamsFile | None = None

    def site(self, site_arg, fallback="tibia"):
        """The site to run: an explicit --site, else the complete file's preset, else `fallback`."""
        return site_arg or self.preset or fallback


class CommandLineOverrides(dict):
    """The {dotted: value} overrides of a command line, a plain dict that also says where its values came from
    (`source`: the --params file, the dedicated flags and the --set names); resolve() records it in the report's
    parameter_set.sources."""
    source = None


def parameters_from_args(args, extra=None):
    """--params FILE, then `extra` (the dedicated flags), then --set, in precedence order (later wins) -> ArgParameters.
    Its overrides are a CommandLineOverrides whose `source` names the file, the flags and the --set names.
    Raises ParameterError."""
    out, pf, parts = CommandLineOverrides(), None, []
    if getattr(args, "params", None):
        pf = read_params_file(args.params)
        out.update(pf.overrides)
        parts.append(f"--params {pf.describe()}")
    if extra:
        ex = _checked(extra)
        out.update(ex)
        if ex:
            parts.append("dedicated flags: " + ", ".join(ex))
    names_set = []
    for item in getattr(args, "set_params", None) or []:
        one = parse_set(item)
        out.update(one)
        names_set += [n for n in one if n not in names_set]
    if names_set:
        parts.append("--set " + ", ".join(names_set))
    out.source = "; ".join(parts) or None
    return ArgParameters(overrides=out, preset=pf.preset if pf is not None and pf.complete else None, file=pf)


def overrides_from_args(args, extra=None):
    """The {dotted: value} overrides of --params FILE, then `extra` (the dedicated flags), then --set (later wins)."""
    return parameters_from_args(args, extra).overrides


__all__ = ["SCHEMA", "STAGES", "SITES", "WORKFLOWS", "MAP_NAMES", "STEPS", "VALIDATED_SCOPE", "IPL_CHOICE", "DEPARTURE",
           "ParameterError", "ParameterWarning", "AutocontourParams", "CalibrationParams", "RenderParams", "LHParams",
           "SegParams", "MorphometryParams", "PoreParams", "BMDParams", "OutputParams", "Step1Params", "DTParams",
           "STAGE_CLASSES", "Spec", "SCHEMA_SPECS", "ALIASES", "STEP_USERS", "users_of", "Parameters", "Resolved",
           "ParamsFile", "ArgParameters", "CommandLineOverrides", "resolve", "statement", "site_of", "names", "coerce",
           "split_name", "flatten_overrides", "parse_set", "complete_set_block", "read_params", "read_params_file",
           "load_params_file", "write_params_file", "printed_set", "permille_to_short", "add_parameter_arguments",
           "parameters_from_args", "overrides_from_args"]
