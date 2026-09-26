"""build_tableS1.py -- Supplementary Table S1 of the ipldt / ORMIR-BQRL manuscript: every parameter the package
uses, module by module, with the IPL option or command it corresponds to, its name in the package, its value
(TIBIA / RADIUS where the two site presets differ), what it controls and the supplementary-figure panel that
shows it.

The table is the one printed in the Supplement, cell for cell (TABLE_S1 and its caption and note below; the
Supplement's table was extended and edited by hand after this script first generated it, and since 2026-09-25 this
script carries the printed version).  Its values are CHECKED against the package at run time, so the printed table
cannot drift from the code unnoticed: code_rows() derives the value of every parameter from the package
(ipldt.step1.Step1Params, ipldt.ormir.DTParams and the Laplace-Hamming constants, ipldt.contour, ipldt.core,
ormir_bqrl), and every number of each derived value must appear in the value cell of the printed row with the same
module and package name; the cortical-porosity rows are checked against ipldt.porosity (its constants and the calls
of pore_cascade).  A value that no longer matches the code stops the script, and so does a derived row without a
printed counterpart.  The panel column follows the supplementary atlas (S1 dense-bone estimate, S2 chamfer-metric
morphology, S3 compartment chain and site presets, S4 contour rendering, S5 Laplace-Hamming, S6 distance-transform
engine, S7 dt_spacing / dt_number and the objects of the maps, S8 the derivation method, S9 the cortical pore
cascade).

Run from the repository root in the `ormir` environment:

    set PYTHONUTF8=1
    python manuscript/tables/build_tableS1.py

Writes manuscript/tables/tableS1_parameters.csv and tableS1_parameters.md (the printed table).
"""
from __future__ import annotations

import csv
import inspect
import json
import os
import re
import sys
from collections import OrderedDict
from dataclasses import fields

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)

import ipldt  # noqa: E402
from ipldt import core, ipl_ops, porosity  # noqa: E402
from ipldt.contour import render as crender, smooth as csmooth  # noqa: E402
from ipldt.step1 import RADIUS, TIBIA  # noqa: E402
from ipldt import ormir as engine  # noqa: E402
import ormir_bqrl  # noqa: E402
from ormir_bqrl import stages  # noqa: E402

FACTS = os.path.join(REPO, "manuscript", "facts", "facts.json")
OUT = os.path.join(HERE, "tableS1_parameters")
COLS = ["Module", "IPL option / command", "ipldt name", "Value (TIBIA / RADIUS where they differ)", "What it controls",
        "S-figure panel"]


def fact(key, default=None):
    try:
        F = json.load(open(FACTS, encoding="utf-8"))["facts"]
        v = F[key]
        return v["value"] if isinstance(v, dict) and "value" in v else v
    except (OSError, KeyError):
        return default


def num(v):
    """An integer or float as the table prints it (thousands separated by a thin space)."""
    if isinstance(v, float) and v == int(v) and abs(v) >= 1:
        v = int(v)
    if isinstance(v, int):
        return f"{v:,d}".replace(",", "\u2009")
    return f"{v:g}"


def preset(name):
    """The TIBIA value, or 'tibia / radius' when the two presets differ."""
    a, b = getattr(TIBIA, name), getattr(RADIUS, name)
    return num(a) if a == b else f"{num(a)} / {num(b)}"


def code_rows():
    """The value of every parameter as the package derives it (the check of the printed table)."""
    p1 = TIBIA
    dt = engine.IPL_SCRIPT32
    diff_fields = [f.name for f in fields(TIBIA) if getattr(TIBIA, f.name) != getattr(RADIUS, f.name)]
    assert sorted(diff_fields) == ["close2", "corner_min"], diff_fields
    assert core.dt_thickness.__defaults__ is not None
    ormir_ver = fact("software.ormir_xct.version", "1.1.0")
    n_step1_cmds = 36
    thr_lo = ipl_ops.mgha_to_native(p1.lower_mgha, 1619.07703, -394.095001, 8192.0)
    thr_up = ipl_ops.mgha_to_native(p1.upper_mgha, 1619.07703, -394.095001, 8192.0)
    lh_thr = engine.LH_THRESHOLD
    assert lh_thr == int(475 / 1000 * engine.INT16_MAX), lh_thr

    rows = []

    def row(module, ipl, name, value, what, panel):
        rows.append(OrderedDict(zip(COLS, (module, ipl, name, value, what, panel))))

    # ------------------------------------------------------------------------------------------ AIM input / output
    M = "AIM input / output"
    row(M, "/read (AIM v020 and v030; char and short data; both run-length compressions)", "read_aim", "\u2014",
        "reads the voxel data, the grid (dim, global voxel position pos), the per-axis element sizes and the "
        "processing log of every AIM; volumes on different grids are always combined by global position", "\u2014")
    row(M, "AIM header element sizes (x, y, z)", "el_size_mm", "read from the header of each scan (60.7 \u00b5m nominal)",
        "the anisotropic element sizes of the Laplace\u2013Hamming filter and, through its x value, the voxel size "
        "that scales the map statistics to millimetres", "S5B, S7I, S7J")
    row(M, "processing log: Density slope, Density intercept, Mu_Scaling (ITK ScancoImageIO header fields when the "
           "log is absent)", "calibration_from_proclog / step1_calibration",
        "per scan (this cohort: slope 1619.07703, intercept \u2212394.095001, Mu_Scaling 8192)",
        "the density calibration that converts the mg HA/cm\u00b3 thresholds of /seg_gauss to native numbers: "
        "rint((mg HA \u2212 intercept) / slope \u00d7 Mu_Scaling)", "S1B")
    row(M, "/write (uncompressed char AIM v020 on the input header) or NIfTI", "write_aim / write_nifti",
        "NIfTI by default; AIM on request", "how masks and maps are written back on the input grid "
        "(NIfTI origin = pos \u00d7 element size, so every output overlays the scan)", "\u2014")

    # ------------------------------------------------------------------------------------------ periosteal contour
    M = "Periosteal contour (ORMIR-XCT)"
    row(M, "\u2014 (the ORMIR-XCT autocontour, not an IPL command)", "step2_autocontour",
        f"ORMIR-XCT {ormir_ver} defaults, called with the scan's mu_water and rescale slope / intercept",
        "the outer (periosteal) contour of the bone by the dual-threshold method; the only stage not derived from "
        "IPL; a contour file (binary mask, labelmap, .seg.nrrd or AIM) can be supplied instead", "Fig. 1, stage 2")

    # ------------------------------------------------------------------------------------------ STEP 1
    M = "Compartment separation (STEP 1)"
    row(M, "evaluation script: Script 32 (tibia) or Script 33 (radius)", "site / Step1Params", "TIBIA / RADIUS",
        f"the preset of the {n_step1_cmds}-command chain; the two differ in corner_min and close2 only", "S3E\u2013G")
    row(M, "/seg_gauss -sigma", "sigma", preset("sigma"), "standard deviation (voxels) of the Gaussian low-pass that "
        "defines dense bone", "S1A, S1C\u2013F")
    row(M, "/seg_gauss -support", "support", preset("support"), "half-width of the kernel (taps \u2212support .. "
        "+support); the smoothed volume is valid only where the kernel fits, so its grid shrinks by support voxels "
        "on every side", "S1A")
    row(M, "/seg_gauss -lower_in_perm_aut_al (-unit 2, mg HA/cm\u00b3)", "lower_mgha", preset("lower_mgha"),
        f"lower density bound of dense bone, converted to a native number by the scan's calibration ({thr_lo} on "
        "this cohort) and applied inclusively", "S1B, S1C, S1G, S1H")
    row(M, "/seg_gauss -upper_in_perm_aut_al", "upper_mgha", preset("upper_mgha"),
        f"upper density bound, inclusive ({thr_up} native on this cohort)", "S1B, S1I")
    row(M, "/seg_gauss arithmetic (no option)", "gauss_lp_ipl / ipl_gauss_weights",
        "float32 taps normalised by the float32-rounded exact sum; one pass per axis in the order x, y, z, each pass "
        "truncated to a short before the next", "the numerical form of the smoothing, which decides the voxels "
        "whose smoothed value lands on a threshold", "S1A, S1C")
    row(M, "-value_in_range", "SET", num(ipl_ops.SET), "the value written into the set voxels of every char mask",
        "\u2014")
    row(M, "/gobj_maskaimpeel_ow -peel_iter (IPL_PEEL0)", "peel0", preset("peel0"),
        "number of slice-wise 4-connected erosions applied to the rendered periosteal contour before it masks a "
        "stage (three stages use it; the other masking steps use peel 0)", "S2H, S3A")
    row(M, "/erosion -erode_distance -metric 11", "erode", preset("erode"),
        "erosion distance N (voxels) of the trabecular candidate", "S2D, S3B")
    row(M, "/dilation -dilate_distance -metric 11", "dilate", preset("dilate"), "dilation distance that follows it",
        "S2E, S3B")
    row(M, "/close -close_distance -metric 11 (first closing)", "close1", preset("close1"),
        "closing distance that bridges the trabecular region", "S2F, S3C")
    row(M, "/open -open_distance -metric 11", "open_", preset("open_"),
        "opening distance; the voxels the opening removes are the input of the corner branch", "S2G, S3D")
    row(M, "/erosion -erode_distance (corner branch)", "corner_erode", preset("corner_erode"),
        "erosion of the opening residue before its components are measured", "S3E, S3F")
    row(M, "/cl_nr_extract -min_number (IPL_MISC1_0; applied before and after the corner dilation)", "corner_min",
        preset("corner_min"), "smallest component (voxels, inclusive) of the eroded residue that counts as a cortical "
        "corner to be restored", "S3E\u2013G")
    row(M, "/dilation -dilate_distance (corner branch)", "corner_dilate", preset("corner_dilate"),
        "dilation of the kept corner components", "S3E, S3F")
    row(M, "/cl_nr_extract -max_number", "corner_max", preset("corner_max"),
        "largest corner component (voxels, inclusive) added back to the trabecular region", "S3E, S3F")
    row(M, "/close -close_distance (IPL_MISC1_1; second closing)", "close2", preset("close2"),
        "second closing distance, which sets the minimum thickness of the cortical compartment", "S3E\u2013G")
    row(M, "/cl_slicewise_extractow -lo_vol_fract_in_perc", "slicewise_lo", preset("slicewise_lo"),
        "per slice, the 4-connected components holding at least this share (%) of the slice's set voxels are kept "
        "(inclusive); a slice in which no component qualifies is cleared", "S3H")
    row(M, "/cl_slicewise_extractow -up_vol_fract_in_perc", "slicewise_up", preset("slicewise_up"),
        "upper bound of that share (inclusive)", "S3H")
    row(M, "/cl_ow_rank_extract -first_rank 1 -last_rank 1 -connect_boundary false", "cl_ow_rank_extract(v, 1, 1)",
        "rank 1 .. 1", "keeps the largest 6-connected component (hole filling and background removal); components "
        "touching the faces of the grid are not joined", "S3B")
    row(M, "-topology 6", "label6", "6 (face neighbours)", "the 3-D connectivity of every component command; the "
        "slice-wise command uses its restriction to one slice (4-connectivity)", "S3B, S3H")
    row(M, "-metric 11", "chamfer_dt_345 / metric11_threshold",
        "chamfer weights 3 (face), 4 (edge), 5 (corner); keep where raw \u2265 3N + 2",
        "the discrete distance of every erosion, dilation, closing and opening: a voxel lies within distance N when "
        "its chamfer value satisfies raw \u2265 3N + 2 (erosion) or raw < 3N + 2 (dilation)", "S2A\u2013C")
    row(M, "/erosion border (no option)", "erosion",
        "margin of N + 2 voxels on all six faces filled with the mirror image of the input",
        "how the erosion treats the faces of the volume: nothing outside the volume counts as background", "S2I\u2013L")
    row(M, "/dilation and /close -continuous_at_boundary cx cy cz", "dilation(continuous_at_boundary=(0, 0, 0))",
        "0 0 0: an empty (background) margin of N + 2 per axis; the written grid grows by N + 1 on every side",
        "how the object grows at the faces of the volume (1 would mirror the object into the border); the "
        "evaluation scripts use 0 0 0 throughout", "S2E, S2J")
    row(M, "/close (composition)", "close", "the dilation, then the erosion on the same buffer, cropped to the input grid",
        "what a closing is made of", "S2F")
    row(M, "/open (composition)", "open_",
        "the erosion on the mirror-padded buffer, then the dilation seeded by every surviving voxel of that buffer "
        "(margin included), cropped to the input grid", "what an opening is made of; it differs from an erosion "
        "followed by a separate dilation only through the margin", "S2G, S2J\u2013L")
    row(M, "/subtract_aims, /add_aims", "subtract_aims, add_aims",
        f"char arithmetic on the union of the two grids, saturated to [{ipl_ops.CHAR_MIN}, {ipl_ops.CHAR_MAX}]",
        "the difference and the union of two masks (a negative difference is stored as \u2212127, a saturated sum "
        "as 127)", "S3E, S3F")
    row(M, "/set aim obj bg", "set_value", "127 0 (mask) or 0 127 (inverted mask)",
        "writes fixed values into the set and the unset voxels; 0 127 inverts a mask", "\u2014")
    row(M, "/bounding_box_cut -border 0 0 0", "bounding_box_cut", "border 0",
        "crops a volume to the tight box of its set voxels (the /seg_gauss input, the opened region and the "
        "cortical mask)", "\u2014")
    row(M, "/gobj_maskaimpeel_ow -gobj_filename", "mask_by_gobj / peel_gobj_render", "\u2014",
        "AND of a stage with the (peeled) rendered periosteal contour, pasted by global position; slices outside "
        "the contour are cleared", "S2H, S3A")

    # ------------------------------------------------------------------------------------------ contour rendering
    M = "Contour rendering"
    row(M, "/togobj_from_aim -curvature_smooth", "smooth_chain", "1",
        "smoothing of every traced chain: one pass of the corner, spur, 135\u00b0-turn and vertical-excursion "
        "rules (stage A), then all five rules including the horizontal excursion swept to convergence (stage B) "
        "whenever stage A or the pre-trace rule changed the chain; a chain that neither changes is stored as traced",
        "S4E, S4F, S4I")
    row(M, "/togobj_from_aim pre-trace rule (no option)", "phase1",
        "a mask pixel whose west and east neighbours are both background is dropped, except the first pixel of its "
        "component and a pixel with mask directly above and below", "which pixels of a slice are traced", "S4D")
    row(M, "/togobj_from_aim trace (no option)", "moore_trace / outer_components / holes",
        "counter-clockwise Moore trace from the raster-first pixel, entered from the west; one outer chain per "
        "8-connected component, one inner chain per 4-connected hole", "the polygon of each stored contour",
        "S4D")
    row(M, "/togobj_from_aim -min_elements (0 = no user minimum)", "MIN_VERTICES", num(crender.MIN_VERTICES),
        "a chain left with fewer vertices after smoothing is not stored, so its component or hole disappears from "
        "the rendering", "S4J")
    row(M, "\u2014", "MAX_SWEEPS", f"{num(csmooth.MAX_SWEEPS)} (or the length of the chain if larger)",
        "cap on the stage-B sweeps; a chain that has not converged by then raises an error instead of returning a "
        "guess", "\u2014")
    row(M, "/gobj_to_aim -peel_iter 0 (rasterisation)", "render_slice / render_volume",
        "even-odd fill of each polygon on the integer scanlines; an inner contour keeps its chain pixels and "
        "removes only its strict interior",
        "the voxel mask every gobj-masked command actually sees, rendered on the mask's own bounding box", "S4A, S4B, S4G, S4H")

    # ------------------------------------------------------------------------------------------ Laplace-Hamming
    M = "Laplace\u2013Hamming segmentation (STEP 2)"
    row(M, "/bounding_box_cut -border 1 1 1, /offset_add, /fill_offset_duplicate", "laplace_hamming_threshold",
        "1 voxel, values copied from the face", "the border added around the volume before the transform",
        "S5D")
    row(M, "/fft_laplace_hamming -redim_pow2 (0 0 0 in the evaluation script: each axis to the next power of two)", "lh_pad_plan, LH_PAD_OFFSET",
        f"'{engine.LH_PAD_OFFSET}': each axis padded to the next power of two with the mirror image of the data "
        "(face voxel not repeated); when the padding is odd, the extra voxel goes before the data",
        "where the volume sits inside the transform box and what fills the rest of it", "S5D")
    row(M, "/fft_laplace_hamming -laplace_eps", "LAPLACE_EPS", num(engine.LAPLACE_EPS),
        "weight of the Laplacian term of the transfer function H(k) = (2\u03c0)\u00b2 [(1 \u2212 \u03b5) + "
        "\u03b5 |k|\u00b2] \u00d7 window", "S5A")
    row(M, "/fft_laplace_hamming -lp_cut_off_freq", "LP_CUT_OFF_FREQ", f"{num(engine.LP_CUT_OFF_FREQ)} (in units of "
        "1 / el_z, i.e. a radial cut-off of 0.3 / el_z cycles per mm)",
        "the radial low-pass cut-off of the window", "S5B")
    row(M, "/fft_laplace_hamming -hamming_amp", "HAMMING_AMP", f"{num(engine.HAMMING_AMP)} (a Hann window)",
        "the window: (1 \u2212 a/2) + (a/2) cos(\u03c0 |k| / k_c) inside the cut-off, 0 outside", "S5C")
    row(M, "/fft_laplace_hamming element sizes (from the AIM header)", "lh_voxel_size_mm = None",
        "the header's (x, y, z) element sizes", "the frequency grid per axis, k_i = fftfreq(n_i, el_i): the filter "
        "is anisotropic when the element sizes differ", "S5B")
    row(M, "\u2014", "LH_DTYPE", f"{engine.LH_DTYPE} (float64 selectable)",
        "the precision of the padded volume, the transfer function and the forward and inverse transforms",
        "S5D, S5E")
    row(M, "/norm_max -max 200000 -type_out short", "NORM_MAX_VALUE, INT16_MAX",
        f"{num(engine.NORM_MAX_VALUE)} \u2192 {num(engine.INT16_MAX)}; short = trunc(float32(lh) \u00d7 "
        f"{num(engine.INT16_MAX)} / {num(engine.NORM_MAX_VALUE)}) after clipping",
        "scales the float filter output to the int16 range with truncation", "S5E")
    row(M, "/threshold -lower_in_perm 475 -upper_in_perm 1000", "LH_THRESHOLD",
        f"{num(lh_thr)} .. {num(engine.INT16_MAX)} (475 \u2030 of {num(engine.INT16_MAX)})",
        "the binarisation of the scaled image", "S5E")
    row(M, "/gobj_maskaimpeel_ow with the periosteal contour, -peel_iter 0", "ipl_seg_assembly (periosteal argument)",
        "the rendered periosteal contour", "restricts the thresholded volume to the bone before the component "
        "filters", "S5F")
    row(M, "/cl_nr_extract -min_number (cortical segmentation)", "CC_MIN_VOXELS_CORT", num(engine.CC_MIN_VOXELS_CORT),
        "6-connected components smaller than this are dropped before the cortical contour is applied", "S5F")
    row(M, "/cl_nr_extract -min_number (trabecular segmentation)", "CC_MIN_VOXELS_TRAB", num(engine.CC_MIN_VOXELS_TRAB),
        "the same filter before the trabecular contour is applied", "S5F")
    row(M, "/gobj_maskaimpeel_ow with the cortical / trabecular contour, -peel_iter 0", "ipl_seg_assembly (G_cort, G_trab)",
        "the rendered compartment contours", "splits the filtered segmentation into its cortical and trabecular "
        "parts", "S5F")
    row(M, "/set_value and /add_aims (SEG assembly)", "SEG_VALUE_CORT, SEG_VALUE_TRAB",
        f"{engine.SEG_VALUE_CORT} / {engine.SEG_VALUE_TRAB}", "the label values of cortical and trabecular bone in "
        "the assembled segmentation", "S5F")

    # ------------------------------------------------------------------------------------------ dt engine
    M = "Distance transforms (STEP 3)"
    row(M, "/dt_thickness, /dt_spacing, /dt_number -ridge_epsilon", "ridge_epsilon", num(dt.ridge_epsilon),
        "the containment test that selects sphere centres: a voxel x is a centre unless one of its 26 neighbours y "
        "inside the image satisfies |x \u2212 y| + s_x \u2212 s_y \u2264 \u03b5 (s = surface distance)", "S6D")
    row(M, "-assign_epsilon", "assign_epsilon", num(dt.assign_epsilon),
        "the reach of the drawn spheres: voxel c takes the largest diameter D of any centre x with |c \u2212 x| "
        "\u2264 D/2 + \u03b5", "S6I")
    row(M, "-peel_iter", "peel_iter", num(dt.peel_iter),
        "slice-wise 4-connected erosions of the rendered contour before the centre selection (\u22121 and 0: none); "
        "the map is clipped to the unpeeled contour", "S6E")
    row(M, "-version", "version", num(dt.version),
        "the diameter rule: 3 = a sphere pinned between the contact point of x and that of its neighbour one step "
        "against the surface direction, D = |P1 \u2212 P2| rounded half up when the midpoint lies within one voxel, "
        "else 2 s_x rounded half up (rule 1); rule 2 uses the neighbour's contact angle instead", "S6G, S6H")
    row(M, "-suppress_boundary", "suppress_boundary", num(dt.suppress_boundary),
        "accepted for completeness; the maps and statistics do not depend on it (values 0 to 3 give the same "
        "output)", "S6F")
    row(M, "-gobj_filename", "gobj", "the rendered trabecular (Tb.Th, Tb.Sp, 1/Tb.N) or cortical (Ct.Th) contour",
        "restricts the sphere centres and clips the map", "S6E")
    row(M, "vector distance transform (no option)", "sir_quad",
        "Danielsson-type vector transform; boustrophedon sweep over the four in-plane quadrant directions, one "
        "ascending and one descending pass in z; outside the image counts as object",
        "the offset from every object voxel to its nearest background voxel", "S6A, S6C")
    row(M, "surface distance (no option)", "surface_distance / surface_vector",
        "s = |p(v)| with p(u) = sign(u) \u00b7 max(|u| \u2212 \u00bd, 0) per component, in double precision",
        "the distance to the object surface, taken half a voxel inside the nearest background voxel", "S6B, S6C")
    row(M, "\u2014", "_RIDGE_TOL, _FEPS", f"{core._RIDGE_TOL:g}, {core._FEPS:g}",
        "floating-point guards of the ridge test and of the exact half-integer roundings; they never decide a "
        "non-tie", "\u2014")
    row(M, "/dt_thickness -input (Tb.Th)", "obj (trab_seg)",
        "the trabecular part of the segmentation inside the trabecular contour (the evaluation-script "
        "definition, reported in this paper; TRAB_TH_old in the code); the whole segmentation is the alternative "
        "object (TRAB_TH), computed by the same call and carried by the workflow report",
        "the binary object in which the Tb.Th spheres are fitted", "S7K, S7L; Fig. 4A")
    row(M, "/dt_thickness -input (Ct.Th)", "cort_mask, cort_gobj",
        "the cortical compartment mask with the cortical contour",
        "the object of Ct.Th is the compartment, not the cortical bone", "S7M")
    row(M, "/dt_spacing -input seg", "dt_spacing",
        "object = NOT segmentation (marrow), spheres restricted to the trabecular contour; the contour boundary "
        "is not a surface", "the object of the Tb.Sp map", "S7A\u2013D")
    row(M, "/dt_number -input seg", "dt_number",
        f"mid-axis = the containment ridge of the segmentation at ridge_epsilon {num(dt.ridge_epsilon)}; object = "
        f"NOT mid-axis; its centres use ridge_epsilon {num(dt.ridge_epsilon)} again",
        "the object of the 1/Tb.N map (ridge_epsilon enters twice)", "S7E\u2013H")
    row(M, "/bounding_box_cut -border 0 of the segmentation; the cortical mask's own box", "bbox_cut",
        "\u2014", "the grid on which each transform runs; its faces are treated as object, so the box matters",
        "\u2014")
    row(M, "\u2014", "backend", "auto (CuPy GPU when available, else CPU); gpu / cpu selectable",
        "where the ridge test and the sphere drawing run; the two back ends return identical maps, centres and "
        "statistics", "\u2014")
    row(M, "\u2014", "voxel_size_mm", f"the header's x element size ({num(engine.VOXEL_SIZE_MM_DEFAULT)} mm by default)",
        "scales the reported statistics; the maps themselves are integer sphere diameters in voxels", "S7I, S7J")

    # ------------------------------------------------------------------------------------------ metrics
    M = "Morphometry and BMD"
    row(M, "BV/TV (the evaluation script writes none; the sheet prints the ratio)", "BV_TV",
        "|segmentation AND rendered trabecular contour| / |rendered trabecular contour|",
        "bone volume fraction, in voxels of the segmentation grid", "Fig. 5A")
    row(M, "'Get Statistics' of each map", "Tb_Th_mm, Tb_Sp_mm, Ct_Th_mm (+ _sd_mm, _max_mm, _skew, _kurtosis, "
           "_valid_fraction)", "mean of the non-zero map voxels \u00d7 voxel size; population SD; maximum; skewness; "
        "kurtosis; fraction of object voxels that received a value",
        "the scalar indices reported per scan", "S7I, S7J")
    row(M, "MAT N (1/Th)", "Tb_N_per_mm", "1 / mean of the 1/Tb.N map (mm)", "trabecular number", "S7J")
    row(M, "\u2014 (ORMIR-XCT bmd_masked)", "Tb_BMD_mgHA_cm3, Ct_BMD_mgHA_cm3",
        "HU \u2192 mg HA/cm\u00b3 with the scan's mu_scaling, mu_water and rescale slope / intercept, averaged "
        "inside the rendered trabecular and cortical contours", "volumetric BMD of the two compartments (an "
        "ORMIR-XCT component; it never fails the morphometry)", "\u2014")

    # ------------------------------------------------------------------------------------------ workflow
    M = "Workflow and outputs (ORMIR-BQRL)"
    row(M, "\u2014", "--site", "tibia | radius (default tibia)", "the STEP 1 preset", "S3E\u2013G")
    row(M, "\u2014", "--periosteal FILE", "none (autocontour) | a mask, labelmap, .seg.nrrd or AIM",
        "an external or manually modified periosteal contour in place of the autocontour", "Fig. 1, stage 2")
    row(M, "\u2014", "--backend", "auto | gpu | cpu", "the distance-transform back end", "\u2014")
    row(M, "\u2014", "--map-units", "voxels (int16 sphere diameters, the AIM values) | mm (float32)",
        "the unit of the written maps", "\u2014")
    row(M, "\u2014", "--no-bmd", "BMD computed by default", "skips the ORMIR-XCT BMD step", "\u2014")
    row(M, "\u2014", "--no-porosity", "pore cascade and Ct.Po computed by default", "skips the cortical pore cascade and Ct.Po", "\u2014")
    row(M, "\u2014", "--map-format", "nifti | aim (default nifti; AIM maps in voxels, on the input header)", "the file format of the written masks, pore map and maps", "\u2014")
    row(M, "\u2014", "write_volumes / MASK_VALUE", f"NIfTI (or AIM, --map-format aim) on the AIM grid, masks written as 0 / {num(stages.MASK_VALUE)}; "
        "the pore map; 3D Slicer segmentation (.seg.nrrd) and labelmap; report as JSON, CSV and Markdown; preview PNG; run log",
        "the output set of a run", "Fig. 1, stage 8")
    row(M, "Script 34 re-entry", "run_from_masks (redo)",
        "a manually modified periosteal, cortical or trabecular contour; a modified compartment's rendering is passed to the "
        "segmentation as its own contour", "re-entry of the workflow after a manual correction in 3D Slicer: the "
        "stages downstream of the modified contour are recomputed", "Fig. 1, stage 8")
    return rows, ormir_ver


# ------------------------------------------------------------------------------------------ the printed table
# Supplementary Table S1 as printed (caption, rows, note).  Edit it together with the Supplement.
CAPTION = ('**Supplementary Table S1.** Every parameter of ipldt and ORMIR-BQRL: the IPL option or command it '
           'corresponds to, its name in the package, its value (TIBIA / RADIUS where the two site presets '
           'differ), what it controls and the supplementary panel that shows it.')
NOTE = ('Values are those of the package (ipldt 1.0.0, ORMIR-BQRL 0.1.0). TIBIA is the preset of the standard '
        'tibia evaluation script (Script 32), RADIUS that of the radius script (Script 33); a single value '
        'applies to both. Rows whose IPL entry carries no option name describe a fixed rule of the command '
        'with no user setting; "—" marks a package setting with no IPL counterpart or a rule with no '
        'dedicated panel. Distances are in voxels; N denotes the distance of a morphological command; s the '
        'surface distance; Δz the element size along the slice axis. Supplementary figures: S1 dense-bone '
        'estimate (/seg_gauss); S2 chamfer-metric morphology; S3 the compartment chain and its site presets; '
        'S4 contour rendering; S5 Laplace–Hamming segmentation; S6 the distance-transform engine; S7 '
        'dt_spacing, dt_number, the map statistics and the objects of the two thickness maps; S8 the '
        'derivation method (no parameters); S9 the cortical pore cascade and Ct.Po.')
TABLE_S1 = [
    ('AIM input / output',
     '/read (AIM v020 and v030; char and short data; both run-length compressions)',
     'read_aim',
     '—',
     'reads the voxel data, the grid (dim, global voxel position pos), the per-axis element sizes and the processing log of every AIM; volumes on different grids are always combined by global position',
     '—'),
    ('AIM input / output',
     'AIM header element sizes (x, y, z)',
     'el_size_mm',
     'read from the header of each scan (60.7 µm nominal)',
     'the anisotropic element sizes of the Laplace–Hamming filter and, through its x value, the voxel size that scales the map statistics to millimeters',
     'S5B, S7I, S7J'),
    ('AIM input / output',
     'processing log: Density slope, Density intercept, Mu_Scaling (ITK ScancoImageIO header fields when the log is absent)',
     'calibration_from_proclog / step1_calibration',
     'per scan (on 99 of the 137 scans: slope 1,619.07703, intercept −394.095001, Mu_Scaling 8192)',
     'the density calibration that converts the mg HA/cm³ thresholds of /seg_gauss to native numbers: rint((mg HA − intercept) / slope × Mu_Scaling)',
     'S1B'),
    ('AIM input / output',
     '/write (uncompressed char AIM v020 on the input header) or NIfTI',
     'write_aim / write_nifti',
     'NIfTI by default; AIM on request',
     'how masks and maps are written back on the input grid (NIfTI origin = pos × element size, so every output overlays the scan)',
     '—'),
    ('Periosteal contour (ORMIR-XCT)',
     '— (the ORMIR-XCT autocontour, not an IPL command)',
     'step2_autocontour',
     "ORMIR-XCT 1.1.0 defaults, called with the scan's mu_water and rescale slope / intercept",
     'the outer (periosteal) contour of the bone by the dual-threshold method; the only stage of the evaluation chain not derived from IPL (the BMD calculation, also from ORMIR-XCT, runs beside it); a contour file (binary mask, labelmap, .seg.nrrd or AIM) can be supplied instead',
     'Figure 1, stage 2'),
    ('Compartment separation (STEP 1)',
     'evaluation script: Script 32 (tibia) or Script 33 (radius)',
     'site / Step1Params',
     'TIBIA / RADIUS',
     'the preset of the 36-command chain; the two differ in corner_min and close2 only',
     'S3E–G'),
    ('Compartment separation (STEP 1)',
     '/seg_gauss -sigma',
     'sigma',
     '2',
     'standard deviation (voxels) of the Gaussian low-pass that defines dense bone',
     'S1A, S1C–F'),
    ('Compartment separation (STEP 1)',
     '/seg_gauss -support',
     'support',
     '3',
     'half-width of the kernel (taps −support .. +support); the smoothed volume is valid only where the kernel fits, so its grid shrinks by support voxels on every side',
     'S1A'),
    ('Compartment separation (STEP 1)',
     '/seg_gauss -lower_in_perm_aut_al (-unit 2, mg HA/cm³)',
     'lower_mgha',
     '500',
     "lower density bound of dense bone, converted to a native number by the scan's calibration (4,524 on 99 of the 137 scans) and applied inclusively",
     'S1B, S1C, S1G, S1H'),
    ('Compartment separation (STEP 1)',
     '/seg_gauss -upper_in_perm_aut_al',
     'upper_mgha',
     '3,000',
     'upper density bound, inclusive (17,173 native on 99 of the 137 scans)',
     'S1B, S1I'),
    ('Compartment separation (STEP 1)',
     '/seg_gauss arithmetic (no option)',
     'gauss_lp_ipl / ipl_gauss_weights',
     'float32 taps normalized by the float32-rounded exact sum; one pass per axis in the order x, y, z, each pass truncated to a short before the next',
     'the numerical form of the smoothing, which decides the voxels whose smoothed value lands on a threshold',
     'S1A, S1C'),
    ('Compartment separation (STEP 1)',
     '-value_in_range',
     'SET',
     '127',
     'the value written into the set voxels of every char mask',
     '—'),
    ('Compartment separation (STEP 1)',
     '/gobj_maskaimpeel_ow -peel_iter (IPL_PEEL0)',
     'peel0',
     '6',
     'number of slice-wise 4-connected erosions applied to the rendered periosteal contour before it masks a stage (three stages use it; the other masking steps use peel 0)',
     'S2H, S3A'),
    ('Compartment separation (STEP 1)',
     '/erosion -erode_distance -metric 11',
     'erode',
     '3',
     'erosion distance N (voxels) of the trabecular candidate',
     'S2D, S3B'),
    ('Compartment separation (STEP 1)',
     '/dilation -dilate_distance -metric 11',
     'dilate',
     '3',
     'dilation distance that follows it',
     'S2E, S3B'),
    ('Compartment separation (STEP 1)',
     '/close -close_distance -metric 11 (first closing)',
     'close1',
     '15',
     'closing distance that bridges the trabecular region',
     'S2F, S3C'),
    ('Compartment separation (STEP 1)',
     '/open -open_distance -metric 11',
     'open_',
     '15',
     'opening distance; the voxels the opening removes are the input of the corner branch',
     'S2G, S3D'),
    ('Compartment separation (STEP 1)',
     '/erosion -erode_distance (corner branch)',
     'corner_erode',
     '3',
     'erosion of the opening residue before its components are measured',
     'S3E, S3F'),
    ('Compartment separation (STEP 1)',
     '/cl_nr_extract -min_number (IPL_MISC1_0; applied before and after the corner dilation)',
     'corner_min',
     '200,000 / 800',
     'smallest component (voxels, inclusive) of the eroded residue that counts as a cortical corner to be restored',
     'S3E–G'),
    ('Compartment separation (STEP 1)',
     '/dilation -dilate_distance (corner branch)',
     'corner_dilate',
     '3',
     'dilation of the kept corner components',
     'S3E, S3F'),
    ('Compartment separation (STEP 1)',
     '/cl_nr_extract -max_number',
     'corner_max',
     '500,000',
     'largest corner component (voxels, inclusive) added back to the trabecular region',
     'S3E, S3F'),
    ('Compartment separation (STEP 1)',
     '/close -close_distance (IPL_MISC1_1; second closing)',
     'close2',
     '50 / 30',
     'second closing distance, which sets the minimum thickness of the cortical compartment',
     'S3E–G'),
    ('Compartment separation (STEP 1)',
     '/cl_slicewise_extractow -lo_vol_fract_in_perc',
     'slicewise_lo',
     '50',
     "per slice, the 4-connected components holding at least this share (%) of the slice's set voxels are kept (inclusive); a slice in which no component qualifies is cleared",
     'S3H'),
    ('Compartment separation (STEP 1)',
     '/cl_slicewise_extractow -up_vol_fract_in_perc',
     'slicewise_up',
     '100',
     'upper bound of that share (inclusive)',
     'S3H'),
    ('Compartment separation (STEP 1)',
     '/cl_ow_rank_extract -first_rank 1 -last_rank 1 -connect_boundary false',
     'cl_ow_rank_extract(v, 1, 1)',
     'rank 1',
     'keeps the largest 6-connected component (hole filling and background removal); components touching the faces of the grid are not joined',
     'S3B'),
    ('Compartment separation (STEP 1)',
     '-topology 6',
     'label6',
     '6 (face neighbors)',
     'the 3D connectivity of every component command; the slice-wise command uses its restriction to one slice (4-connectivity)',
     'S3B, S3H'),
    ('Compartment separation (STEP 1)',
     '-metric 11',
     'chamfer_dt_345 / metric11_threshold',
     'chamfer weights 3 (face), 4 (edge), 5 (corner); keep where raw ≥ 3N + 2',
     'the discrete distance of every erosion, dilation, closing and opening: an erosion keeps a voxel whose chamfer distance to the background satisfies raw ≥ 3N + 2, and a dilation adds a voxel whose chamfer distance to the object satisfies raw < 3N + 2',
     'S2A–C'),
    ('Compartment separation (STEP 1)',
     '/erosion border (no option)',
     'erosion',
     'margin of N + 2 voxels on all six faces filled with the edge-inclusive mirror image of the input (face voxel repeated)',
     'how the erosion treats the faces of the volume: nothing outside the volume counts as background',
     'S2I–L'),
    ('Compartment separation (STEP 1)',
     '/dilation and /close -continuous_at_boundary cx cy cz',
     'dilation(continuous_at_boundary=(0, 0, 0))',
     '0 0 0: an empty (background) margin of N + 2 per axis; the written grid grows by N + 1 on every side',
     'how the object grows at the faces of the volume (1 would mirror the object into the border); the evaluation scripts use 0 0 0 throughout',
     'S2E, S2J'),
    ('Compartment separation (STEP 1)',
     '/close (composition)',
     'close',
     'the dilation, then the erosion on the same buffer, cropped to the input grid',
     'what a closing is made of',
     'S2F'),
    ('Compartment separation (STEP 1)',
     '/open (composition)',
     'open_',
     'the erosion on the mirror-padded buffer, then the dilation seeded by every surviving voxel of that buffer (margin included), cropped to the input grid',
     'what an opening is made of; it differs from an erosion followed by a separate dilation only through the margin',
     'S2G, S2J–L'),
    ('Compartment separation (STEP 1)',
     '/subtract_aims, /add_aims',
     'subtract_aims, add_aims',
     'char arithmetic on the union of the two grids, saturated to [−127, 127]',
     'the difference and the union of two masks (a negative difference is stored as −127, a saturated sum as 127)',
     'S3E, S3F'),
    ('Compartment separation (STEP 1)',
     '/set_value',
     'set_value',
     '127 0 (mask) or 0 127 (inverted mask)',
     'writes fixed values into the set and the unset voxels; 0 127 inverts a mask',
     '—'),
    ('Compartment separation (STEP 1)',
     '/bounding_box_cut -border 0 0 0',
     'bounding_box_cut',
     'border 0',
     'crops a volume to the tight box of its set voxels (the /seg_gauss input, the opened region and the cortical mask)',
     '—'),
    ('Compartment separation (STEP 1)',
     '/gobj_maskaimpeel_ow -gobj_filename',
     'mask_by_gobj / peel_gobj_render',
     '—',
     'AND of a stage with the (peeled) rendered periosteal contour, pasted by global position; slices outside the contour are cleared',
     'S2H, S3A'),
    ('Contour rendering',
     '/togobj_from_aim -curvature_smooth',
     'smooth_chain',
     '1',
     'smoothing of every traced chain: one pass of the corner, spur, 135°-turn and vertical-excursion rules (stage A), then all five rules including the horizontal excursion swept to convergence (stage B) whenever stage A or the pre-trace rule changed the chain; a chain changed by neither is stored as traced',
     'S4E, S4F, S4I'),
    ('Contour rendering',
     '/togobj_from_aim pre-trace rule (no option)',
     'phase1',
     'a mask pixel whose west and east neighbors are both background is dropped, except the first pixel of its component and a pixel with mask directly above and below',
     'which pixels of a slice are traced',
     'S4D'),
    ('Contour rendering',
     '/togobj_from_aim trace (no option)',
     'moore_trace / outer_components / holes',
     'counterclockwise Moore trace from the raster-first pixel, entered from the west; one outer chain per 8-connected component, one inner chain per 4-connected hole',
     'the polygon of each stored contour',
     'S4D'),
    ('Contour rendering',
     '/togobj_from_aim -min_elements (0 = no user minimum)',
     'MIN_VERTICES',
     '4',
     'a chain left with fewer vertices after smoothing is not stored, so its component or hole disappears from the rendering',
     'S4J'),
    ('Contour rendering',
     '—',
     'MAX_SWEEPS',
     '1,000 (or the length of the chain if larger)',
     'cap on the stage-B sweeps; a chain that has not converged by then raises an error instead of returning a guess',
     '—'),
    ('Contour rendering',
     '/gobj_to_aim -peel_iter 0 (rasterization)',
     'render_slice / render_volume',
     'even-odd fill of each polygon on the integer scanlines; an inner contour keeps its chain pixels and removes only its strict interior',
     "the voxel mask every gobj-masked command actually sees, rendered on the mask's own bounding box",
     'S4A, S4B, S4G, S4H'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/bounding_box_cut -border 1 1 1, /offset_add, /fill_offset_duplicate',
     'laplace_hamming_threshold',
     '1 voxel, values copied from the face',
     'the border added around the volume before the transform',
     'S5D'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/fft_laplace_hamming -redim_pow2 (0 0 0 in the evaluation script: each axis to the next power of two)',
     'lh_pad_plan, LH_PAD_OFFSET',
     '"ceil": each axis padded to the next power of two with the mirror image of the data (face voxel not repeated); when the padding is odd, the extra voxel goes before the data',
     'where the volume sits inside the transform box and what fills the rest of it',
     'S5D'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/fft_laplace_hamming -laplace_eps',
     'LAPLACE_EPS',
     '0.45',
     'weight of the Laplacian term of the transfer function H(k) = (2π)² [(1 − ε) + ε ∣k∣²] × window',
     'S5A'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/fft_laplace_hamming -lp_cut_off_freq',
     'LP_CUT_OFF_FREQ',
     '0.3 (in units of 1 / Δz, i.e., a radial cutoff of 0.3 / Δz cycles per mm)',
     'the radial low-pass cutoff of the window',
     'S5B'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/fft_laplace_hamming -hamming_amp',
     'HAMMING_AMP',
     '1 (a Hann window)',
     'the window: (1 − A/2) + (A/2) cos(π ∣k∣ / k_c) inside the cutoff, 0 outside',
     'S5C'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/fft_laplace_hamming element sizes (from the AIM header)',
     'lh_voxel_size_mm = None',
     "the header's (x, y, z) element sizes",
     'the frequency grid per axis, k_i = fftfreq(n_i, el_i): the filter is anisotropic when the element sizes differ',
     'S5B'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '—',
     'LH_DTYPE',
     'float32 (float64 selectable)',
     'the precision of the padded volume, the transfer function and the forward and inverse transforms',
     '—'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/norm_max -max 200000 -type_out short',
     'NORM_MAX_VALUE, INT16_MAX',
     '200,000 → 32,767; short = trunc(float32(lh) × 32,767 / 200,000) after clipping',
     'scales the float filter output to the int16 range with truncation',
     'S5E'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/threshold -lower_in_perm 475 -upper_in_perm 1000',
     'LH_THRESHOLD',
     '15,564–32,767 (475‰ of 32,767)',
     'the binarization of the scaled image',
     'S5E'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/gobj_maskaimpeel_ow with the periosteal contour, -peel_iter 0',
     'ipl_seg_assembly (periosteal argument)',
     'the rendered periosteal contour',
     'restricts the thresholded volume to the bone before the component filters',
     'S5F'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/cl_nr_extract -min_number (cortical segmentation)',
     'CC_MIN_VOXELS_CORT',
     '35',
     '6-connected components smaller than this are dropped before the cortical contour is applied',
     'S5F'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/cl_nr_extract -min_number (trabecular segmentation)',
     'CC_MIN_VOXELS_TRAB',
     '70',
     'the same filter before the trabecular contour is applied',
     'S5F'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/gobj_maskaimpeel_ow with the cortical / trabecular contour, -peel_iter 0',
     'ipl_seg_assembly (G_cort, G_trab)',
     'the rendered compartment contours',
     'splits the filtered segmentation into its cortical and trabecular parts',
     'S5F'),
    ('Laplace–Hamming segmentation (STEP 2)',
     '/set_value and /add_aims (SEG assembly)',
     'SEG_VALUE_CORT, SEG_VALUE_TRAB',
     '127 / 126',
     'the label values of cortical and trabecular bone in the assembled segmentation',
     'S5F'),
    ('Distance transforms (STEP 3)',
     '/dt_thickness, /dt_spacing, /dt_number -ridge_epsilon',
     'ridge_epsilon',
     '0.9',
     'the containment test that selects sphere centers: a voxel x is a center unless one of its 26 neighbors y inside the image satisfies ∣x − y∣ + s_x − s_y ≤ ε (s = surface distance)',
     'S6D'),
    ('Distance transforms (STEP 3)',
     '-assign_epsilon',
     'assign_epsilon',
     '0.5',
     'the reach of the drawn spheres: voxel c takes the largest diameter D of any center x with ∣c − x∣ ≤ D/2 + ε',
     'S6I'),
    ('Distance transforms (STEP 3)',
     '-peel_iter',
     'peel_iter',
     '−1',
     'slice-wise 4-connected erosions of the rendered contour before the center selection (−1 and 0: none); the map is clipped to the unpeeled contour',
     'S6E'),
    ('Distance transforms (STEP 3)',
     '-version',
     'version',
     '3',
     "the diameter rule: 3 = a sphere pinned between the contact point of x and that of its neighbor one step against the surface direction, D = ∣P1 − P2∣ rounded half up when the midpoint lies within one voxel, else 2 s_x rounded half up (rule 1); rule 2 uses the neighbor's contact angle instead",
     'S6G, S6H'),
    ('Distance transforms (STEP 3)',
     '-suppress_boundary',
     'suppress_boundary',
     '2',
     'accepted for completeness; the maps and statistics do not depend on it (values 0 to 3 give identical maps and equal statistics)',
     'S6F'),
    ('Distance transforms (STEP 3)',
     '-gobj_filename',
     'gobj',
     'the rendered trabecular (Tb.Th, Tb.Sp, 1/Tb.N) or cortical (Ct.Th) contour',
     'restricts the sphere centers and clips the map',
     'S6E'),
    ('Distance transforms (STEP 3)',
     'vector distance transform (no option)',
     'sir_quad',
     'Danielsson-type vector transform; boustrophedon sweep over the four in-plane quadrant directions, one ascending and one descending pass in z; outside the image counts as object',
     'the offset from every object voxel to its nearest background voxel',
     'S6A, S6C'),
    ('Distance transforms (STEP 3)',
     'surface distance (no option)',
     'surface_distance / surface_vector',
     's = ∣p(v)∣ with p(u) = sign(u) · max(∣u∣ − ½, 0) per component, in double precision',
     'the distance to the object surface, taken half a voxel inside the nearest background voxel',
     'S6B, S6C'),
    ('Distance transforms (STEP 3)',
     '—',
     '_RIDGE_TOL, _FEPS',
     '1e-09, 1e-09',
     'floating-point guards of the ridge test and of the exact half-integer roundings; they never decide a non-tie',
     '—'),
    ('Distance transforms (STEP 3)',
     '/dt_thickness -input (Tb.Th)',
     'seg (TRAB_TH); trab_seg (TRAB_TH_old)',
     "reported by the workflow: the whole segmentation inside the trabecular contour (TRAB_TH), the segmentation Tb.Sp and Tb.N are also computed from; for the comparison with IPL: the trabecular part of the segmentation inside the trabecular contour (TRAB_SEG; TRAB_TH_old), the object of IPL's evaluation script; ipldt computes both, and dt_thickness takes either object",
     'the binary object in which the Tb.Th spheres are fitted; the two maps differ only where a trabecula meets the endocortical boundary, which TRAB_SEG truncates and the whole segmentation keeps at its full width',
     'Figure 4A; S7K, S7L'),
    ('Distance transforms (STEP 3)',
     '/dt_thickness -input (Ct.Th)',
     'cort_mask, cort_gobj',
     'the cortical compartment mask with the cortical contour',
     'the object of Ct.Th is the compartment, not the cortical bone',
     'S7M'),
    ('Distance transforms (STEP 3)',
     '/dt_spacing -input seg',
     'dt_spacing',
     'object = NOT segmentation (marrow), spheres restricted to the trabecular contour; the contour boundary is not a surface',
     'the object of the Tb.Sp map',
     'S7A–D'),
    ('Distance transforms (STEP 3)',
     '/dt_number -input seg',
     'dt_number',
     'mid-axis = the containment ridge of the segmentation at ridge_epsilon 0.9; object = NOT mid-axis; its centers use ridge_epsilon 0.9 again',
     'the object of the 1/Tb.N map (ridge_epsilon enters twice)',
     'S7E–H'),
    ('Distance transforms (STEP 3)',
     "/bounding_box_cut -border 0 of the segmentation; the cortical mask's own box",
     'bbox_cut',
     '—',
     'the grid on which each transform runs; its faces are treated as object, so the box matters',
     '—'),
    ('Distance transforms (STEP 3)',
     '—',
     'backend',
     'auto (CuPy GPU when available, else CPU); gpu / cpu selectable',
     'where the ridge test and the sphere drawing run; the two back ends return identical maps and centers and equal statistics',
     '—'),
    ('Distance transforms (STEP 3)',
     '—',
     'voxel_size_mm',
     "the header's x element size (0.0607 mm by default)",
     'scales the reported statistics; the maps themselves are integer sphere diameters in voxels',
     'S7I, S7J'),
    ('Cortical porosity (Ct.Po)',
     'evaluation script: the cortical pore block of Burghardt et al. (main-text reference 6) in Scripts 32 and 33, which writes <base>_PORE.AIM',
     'pore_cascade',
     '—',
     "the command-for-command cascade that turns the cortical compartment and the cortical segmentation into IPL's pore map; the same values are used at both sites",
     'S9A–H'),
    ('Cortical porosity (Ct.Po)',
     '/gobj_to_aim of CORT_MASK.GOBJ (the cortical contour), -peel_iter 0',
     'cort_render / contour_render',
     "IPL's own <base>_CORT_MASK_CT.AIM where one is delivered, otherwise render_volume of the raw cortical mask pasted onto the periosteal contour's bounding box",
     "the rendered cortical compartment every stage of the cascade is confined to; the cascade depends on the grid it is rendered on and not only on its content, because the slice-wise rule takes its denominator from the set voxels of each slice (the mask's own tight box leaves corner fragments that pass the 5% bound and add pore voxels)",
     'S9A'),
    ('Cortical porosity (Ct.Po)',
     '/gobj_to_aim grid of CORT_MASK.GOBJ (the grid the pore block runs on)',
     'RENDER_GRID_MARGIN',
     "2 (voxels; low end = the contour's lowest in-plane coordinate − 2, clipped at 0; high end = the highest coordinate, taken + 1 on a slice of even extent, + 2; z = the contour's slices)",
     "the grid the workflows run the cascade on, predicted from the rendered cortical contour alone (render_grid; equal to IPL's own render grid on 137/137 scans): the slice-wise rule's per-slice denominator depends on it",
     'S9A'),
    ('Cortical porosity (Ct.Po)',
     '/cl_slicewise_extractow -lo_vol_fract_in_perc (both pore passes)',
     'SLICE_FRACTION[0]',
     '0',
     "per slice, the 4-connected components of the inverted segmentation holding at least this share (%) of the slice's set voxels are kept (inclusive)",
     'S9B, S9G'),
    ('Cortical porosity (Ct.Po)',
     '/cl_slicewise_extractow -up_vol_fract_in_perc (both pore passes)',
     'SLICE_FRACTION[1]',
     '5',
     'upper bound of that share (inclusive): a void is a pore candidate only while it is small in its own slice, which is what separates a pore from the marrow space',
     'S9B, S9G'),
    ('Cortical porosity (Ct.Po)',
     '/cl_slicewise_extractow -value_in_range (first pass / second pass)',
     'value_in_range, SET',
     '1 (pores_E) / 127 (pores_DF)',
     'the value the kept voids carry: 1 in the first pass, so that adding it to the compartment label 2 distinguishes a small void (3) from a large one (2); 127 in the second pass, where the result is an ordinary mask',
     'S9B, S9E, S9G'),
    ('Cortical porosity (Ct.Po)',
     '/cl_rank_extract -first_rank 1 -last_rank 1',
     'cl_ow_rank_extract(v, 1, 1)',
     'rank 1',
     'keeps the largest 6-connected component of "everything that is not cortical bone" — the blob the pores must not be connected to',
     'S9C'),
    ('Cortical porosity (Ct.Po)',
     '/cl_rank_extract -connect_boundary',
     'connect_boundary',
     'false',
     'components touching the faces of the grid are not joined before the ranking',
     'S9C'),
    ('Cortical porosity (Ct.Po)',
     '/cl_rank_extract -value_in_range',
     'value_in_range',
     '2',
     'the value that component carries, so that /subtract_aims removes exactly the compartment label 2 from the voxels it covers',
     'S9C, S9E'),
    ('Cortical porosity (Ct.Po)',
     '/gobj_maskaimpeel_ow -peel_iter (the rank-1 component, and the hysteresis output)',
     'gobj_maskaimpeel_ow(v, cort_render, 0)',
     '0',
     'both masking steps use the rendered cortical compartment unpeeled; the second is what clears the part of the hysteresis output that lies outside the compartment',
     'S9C, S9E, S9F'),
    ('Cortical porosity (Ct.Po)',
     '/hysteresis_threshold -low_thresh',
     'low_thresh',
     '1',
     'the seed bound, inclusive: the seed set is v ≤ 1, the marrow and everything outside the compartment together with the slice-wise small voids there',
     'S9E, S9I'),
    ('Cortical porosity (Ct.Po)',
     '/hysteresis_threshold -high_thresh',
     'high_thresh',
     '3',
     'the weak bound, inclusive: the weak set is v ≤ 3, everything that is not cortical bone',
     'S9E, S9I'),
    ('Cortical porosity (Ct.Po)',
     '/hysteresis_threshold -unit',
     'unit',
     '5 (native)',
     'the thresholds are raw voxel values; the only unit the evaluation script uses and the only one implemented — any other raises an error rather than guessing a conversion',
     'S9E'),
    ('Cortical porosity (Ct.Po)',
     '/hysteresis_threshold -mode',
     'mode',
     '0',
     '"segment low intensity object": seed = v ≤ low_thresh and weak = v ≤ high_thresh (mode 1 is implemented as the symmetric high-intensity reading, which no IPL export has confirmed)',
     'S9E, S9I'),
    ('Cortical porosity (Ct.Po)',
     '/hysteresis_threshold -grow_axes',
     'grow_axes',
     '0 0 1',
     'the axes the weak set is labeled along: ±z only, so a component is a maximal run of weak voxels down one (x, y) column and a pore is connected to the marrow only through the slice axis',
     'S9E, S9I'),
    ('Cortical porosity (Ct.Po)',
     '/hysteresis_threshold -value_in_range',
     'value_in_range',
     '127',
     'the value written into every weak run that contains at least one seed voxel; the seed voxels are kept',
     'S9E'),
    ('Cortical porosity (Ct.Po)',
     '/set_value (the label values the cascade carries)',
     'set_value',
     '2 (compartment), 125 (segmentation and background), 1 (slice-wise small void), plus the sums 3 and 127',
     'the six values of the label image cortseg_CDE that /hysteresis_threshold reads — 0 marrow or outside, 1 a small void there, 2 a void in the compartment that is not marrow-connected and large in its slice, 3 the same and small in its slice, 125 bone outside the compartment, 127 cortical bone; the two thresholds 1 and 3 cut this encoding into the seed and weak sets',
     'S9D, S9E'),
    ('Cortical porosity (Ct.Po)',
     '/cl_nr_extract -min_number',
     'MIN_PORE_VOXELS',
     '20',
     'smallest 6-connected component (voxels, inclusive) kept as a pore; on the scan of Supplementary Figure S9 it drops 18,791 of the 28,070 components of pores_G, 116,153 of 1,192,833 voxels',
     'S9G'),
    ('Cortical porosity (Ct.Po)',
     '/cl_nr_extract -max_number',
     'cl_nr_extract(v, 20, 0)',
     '0 (no upper bound)',
     'no largest-pore limit',
     'S9G'),
    ('Morphometry and BMD',
     'BV/TV (the evaluation script writes none; the sheet prints the ratio)',
     'BV_TV',
     '∣segmentation ∩ rendered trabecular contour∣ / ∣rendered trabecular contour∣',
     'bone volume fraction, in voxels of the segmentation grid',
     'Figure 5A'),
    ('Morphometry and BMD',
     '"Get Statistics" of each map',
     'Tb_Th_mm, Tb_Sp_mm, Ct_Th_mm (+ _sd_mm, _max_mm, _skew, _kurtosis, _valid_fraction)',
     'mean of the non-zero map voxels × voxel size; population SD; maximum; skewness; kurtosis; fraction of object voxels that received a value',
     'the scalar indices reported per scan',
     'S7I, S7J'),
    ('Morphometry and BMD',
     'MAT N (1/Th)',
     'Tb_N_per_mm',
     '1 / mean of the 1/Tb.N map (mm)',
     'trabecular number',
     'S7J'),
    ('Morphometry and BMD',
     'Ct.Po (the evaluation script writes none; the evaluation sheet prints the ratio)',
     'ct_po',
     "∣PORE ∩ rendered CORT_MASK∣ / ∣rendered CORT_MASK∣ at full precision (IPL's evaluation sheet prints it to three decimals)",
     'cortical porosity: the pore voxels over the cortical compartment, not Ct.Po.V / (Ct.Po.V + Ct.BV); the compartment reading reproduces the printed value on all 30 scans whose evaluation sheet prints Ct.Po, the Ct.BV reading on none',
     'S9J'),
    ('Morphometry and BMD',
     '— (ORMIR-XCT bmd_masked)',
     'Tb_BMD_mgHA_cm3, Ct_BMD_mgHA_cm3',
     "HU (Hounsfield units) → mg HA/cm³ with the scan's mu_scaling, mu_water and rescale slope / intercept, averaged inside the rendered trabecular and cortical contours",
     'volumetric BMD of the two compartments (an ORMIR-XCT component; its failure does not stop the morphometry)',
     '—'),
    ('Workflow and outputs (ORMIR-BQRL)',
     '—',
     '--site',
     'tibia ∣ radius (default tibia)',
     'the STEP 1 preset',
     'S3E–G'),
    ('Workflow and outputs (ORMIR-BQRL)',
     '—',
     '--periosteal FILE',
     'none (autocontour) ∣ a mask, labelmap, .seg.nrrd or AIM',
     'an external or manually modified periosteal contour in place of the autocontour',
     'Figure 1, stage 2'),
    ('Workflow and outputs (ORMIR-BQRL)',
     '—',
     '--backend',
     'auto ∣ gpu ∣ cpu',
     'the distance-transform back end',
     '—'),
    ('Workflow and outputs (ORMIR-BQRL)',
     '—',
     '--map-units',
     'voxels (int16 sphere diameters, the AIM values) ∣ mm (float32)',
     'the unit of the written maps',
     '—'),
    ('Workflow and outputs (ORMIR-BQRL)',
     '—',
     '--no-bmd',
     'BMD computed by default',
     'skips the ORMIR-XCT BMD step',
     '—'),
    ('Workflow and outputs (ORMIR-BQRL)',
     '—',
     '--no-porosity',
     'pore cascade and Ct.Po computed by default',
     'skips the cortical pore cascade and Ct.Po',
     '—'),
    ('Workflow and outputs (ORMIR-BQRL)',
     '—',
     '--map-format',
     'nifti ∣ aim (default nifti; AIM maps in voxels, on the input header)',
     'the file format of the written masks, pore map and maps',
     '—'),
    ('Workflow and outputs (ORMIR-BQRL)',
     '—',
     'write_volumes / MASK_VALUE',
     'NIfTI (or AIM, --map-format aim) on the AIM grid, masks written as 0 / 127; the pore map; 3D Slicer segmentation (.seg.nrrd) and labelmap; report as JSON, CSV and Markdown; preview PNG; run log',
     'the output set of a run',
     'Figure 1, stage 8'),
    ('Workflow and outputs (ORMIR-BQRL)',
     'Script 34 re-entry',
     'run_from_masks (redo)',
     "a manually modified periosteal, cortical or trabecular contour; a modified compartment's rendering is passed to the segmentation as its own contour",
     're-entry of the workflow after a manual correction in 3D Slicer: the stages downstream of the modified contour are recomputed',
     'Figure 1, stage 8'),
]
# a derived row whose package name is printed differently
NAME_ALIAS = {("Distance transforms (STEP 3)", "obj (trab_seg)"): "seg (TRAB_TH); trab_seg (TRAB_TH_old)"}


def numbers(s):
    """The numbers of a cell, with thousands separators (comma or thin space) dropped and the typographic minus read."""
    s = s.replace("\u2009", "").replace("\u2212", "-")
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)
    return re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", s)


def check_against_code(printed):
    """Every derived value's numbers appear in the printed value cell of the same module and package name; the
    cortical-porosity rows agree with ipldt.porosity.  Returns the number of values checked."""
    queue = OrderedDict()                     # (module, package name) -> the printed rows, in table order
    for r in printed:
        queue.setdefault((r[0], r[2]), []).append(r)
    by_key = {k: v[0] for k, v in queue.items()}
    derived, _ = code_rows()
    problems = []
    for r in derived:                         # a name printed twice (e.g. open_) pairs its rows in order
        k = (r["Module"], NAME_ALIAS.get((r["Module"], r["ipldt name"]), r["ipldt name"]))
        p = queue[k].pop(0) if queue.get(k) else None
        if p is None:
            problems.append(f"derived row {k} has no printed row")
            continue
        missing = [x for x in numbers(r[COLS[3]]) if x not in numbers(p[3])]
        if missing:
            problems.append(f"{k}: the code gives {missing}, the printed value is {p[3]!r}")
    # the cortical pore cascade (Script 32 STEP 2): the constants and the calls pore_cascade makes
    M = "Cortical porosity (Ct.Po)"
    hys, cascade = porosity.SCRIPT32_HYSTERESIS, inspect.getsource(porosity.pore_cascade)
    expect = {
        "SLICE_FRACTION[0]": num(porosity.SLICE_FRACTION[0]), "SLICE_FRACTION[1]": num(porosity.SLICE_FRACTION[1]),
        "MIN_PORE_VOXELS": num(porosity.MIN_PORE_VOXELS), "low_thresh": num(hys["low_thresh"]),
        "high_thresh": num(hys["high_thresh"]), "unit": num(hys["unit"]), "mode": num(hys["mode"]),
        "grow_axes": " ".join(num(g) for g in hys["grow_axes"]),
        "RENDER_GRID_MARGIN": num(porosity.RENDER_GRID_MARGIN),
    }
    for name, val in expect.items():
        p = by_key.get((M, name))
        if p is None or not (p[3] == val or p[3].startswith(val + " ")):
            problems.append(f"{M} / {name}: the code gives {val!r}, the printed value is {p[3] if p else None!r}")
    calls = ["cl_slicewise_extractow(pores_E, lo_frac, up_frac, value_in_range=1)",
             "cl_slicewise_extractow(pores_DF, lo_frac, up_frac, value_in_range=SET)",
             "cl_ow_rank_extract(pores_M0, 1, 1, connect_boundary=False, value_in_range=2)",
             "cl_nr_extract(pores_G, int(min_pore_voxels), 0, value_in_range=SET)",
             "set_value(cort_render, 2, 0)", "set_value(cort_seg, 125, 0)"]
    problems += [f"{M}: pore_cascade no longer calls {c}" for c in calls if c not in cascade]
    if hys["value_in_range"] != porosity.SET or num(porosity.SET) != "127":
        problems.append(f"{M}: the hysteresis value_in_range is {hys['value_in_range']}, the printed table says 127")
    if problems:
        raise SystemExit("Table S1 disagrees with the code:\n  " + "\n  ".join(problems))
    return len(derived) + len(expect) + len(calls) + 1


def main():
    printed = [tuple(r) for r in TABLE_S1]
    assert all(len(r) == len(COLS) for r in printed)
    n_checked = check_against_code(printed)
    ormir_ver = fact("software.ormir_xct.version", "1.1.0")
    os.makedirs(HERE, exist_ok=True)
    with open(OUT + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(printed)
    lines = [CAPTION, "", "| " + " | ".join(COLS) + " |", "|" + "|".join("---" for _ in COLS) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in printed]
    lines += ["", NOTE]
    with open(OUT + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    modules = OrderedDict()
    for r in printed:
        modules[r[0]] = modules.get(r[0], 0) + 1
    print(f"wrote {os.path.relpath(OUT, REPO)}.csv / .md: {len(printed)} rows, the printed table "
          f"({n_checked} values checked against the code)")
    for m, n in modules.items():
        print(f"  {n:3d}  {m}")
    print(f"ipldt {ipldt.__version__}, ORMIR-BQRL {ormir_bqrl.__version__}, ORMIR-XCT {ormir_ver}")


if __name__ == "__main__":
    main()
