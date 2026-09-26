"""ipldt.ormir -- ORMIR-XCT adapter for ipldt's reimplementation of Scanco IPL morphometry.

Three layers, all adapters (ipldt.core / field / gpu / contour are untouched):

  SimpleITK wrappers      dt_thickness_sitk, dt_spacing_sitk, dt_number_sitk, render_volume_sitk,
                          ipl_trabecular_microarchitecture_sitk, ipl_cortical_thickness_sitk
                          -- image in, image out, so they drop into ormir_xct.core.
  IPL-style pipeline      run_pipeline(aim_path, out_dir, ...) follows the upstream ORMIR-XCT structure:
                            1  read the AIM: ORMIR file_reader (HU) + native int16 (ipldt.io.read_aim)
                            2  ORMIR autocontour -> periosteal mask
                            3  cortical / trabecular compartment masks = Scanco Script 32 / 33 STEP 1 reimplemented
                               command for command (ipldt.step1 on the reimplemented ipldt.ipl_ops commands: /seg_gauss with
                               IPL's float32 kernel, metric-11 morphology, /cl_*_extract, /gobj_maskaimpeel_ow);
                               the periosteal raster is rendered on its bounding box first (= /gobj_to_aim).
                               Identical to IPL's CORT_MASK / TRAB_MASK given IPL's contour (the paper).
                            4  contours rendered with IPL's rules (ipldt.render_volume); Laplace-Hamming
                               binarization assembled into IPL's SEG (Script 32 lines 407-472) with the
                               the raw periosteal raster, then the RENDERED cortical / trabecular gobjs
                            5  dt_thickness (Tb.Th: SEG with the trabecular gobj; Ct.Th: cortical mask
                               raster with the cortical gobj), dt_spacing, dt_number -- on the grids IPL
                               uses (SEG bounding box; cortical-mask bounding box)

  Tb.Th, A DELIBERATE DIVERGENCE FROM SCRIPTS 32 / 33 / 34 (decided 2026-09-15).  This pipeline computes
  Tb.Th on the WHOLE SEG cropped to the trabecular gobj -- the 'TRAB_TH' result above.  Scanco's own
  evaluation scripts do NOT: all three read TRAB_SEG for the thickness (Scripts 32 and 33 read back the
  trabecular segmentation they wrote in STEP 2 and run /dt_thickness on it with the trabecular contour as the
  gobj; Script 34's re-evaluation does the same), and reserve the whole SEG for Tb.Sp and Tb.N.
  `ipl_morphometry(..., trab_seg=...)` computes the script definition too, as 'TRAB_TH_old'.
  Why the divergence is deliberate: the two differ only where a trabecula is cut by the endocortical
  boundary, and taking the thickness on the whole SEG measures such a trabecula at its true width instead
  of the width of the part that survived the compartment split, so it is the less arbitrary quantity for
  new analyses.  It is also consistent with how these same scripts treat Tb.Sp and Tb.N, which read the
  whole SEG.  WHICH DEFINITION IPL'S OWN FILES FOLLOW: one, the scripts' (TRAB_SEG cropped to the trabecular
  contour).  On all 116 radius / tibia scans of the paper IPL's own TRAB_SEG was cropped (every validation record
  has variants.trab_seg_mask 'gobj'), and in configuration A the scripts' definition reproduces IPL's Tb.Th map on
  116 / 116 (137 / 137 with the 21 patellae).  validation/validate_dataset.py also detects, from a SEG's processing
  log or from IPL's evaluation log, an evaluation run in which that crop did not run (an earlier export of the same
  measurements held such runs; none is in the paper's set) and models it (`--trab-th`, default 'auto';
  `--trab-th seg` forces the whole-SEG reading everywhere; both definitions are computed, stored and tabulated
  either way).
  Anyone reporting Tb.Th from this pipeline alongside published IPL numbers must say which definition they
  used: the two are NOT interchangeable.  Over the 116 radius / tibia scans of the paper the whole-SEG Tb.Th
  exceeds the scripts' by 0.70-33.91 % (mean 5.17 %), and by 0.97-2.68 % (mean 1.37 %) over the 21 patellae
  (relative to the scripts' value: configuration A's `metrics_ours.TbTh_seg` against `TbTh_tseg` in the radius /
  tibia validation records, `TbTh_ours` against `TbTh_old_ours` in validation/results/sample_means.csv).
  A naming trap worth knowing: for the patella cohort IPL exported BOTH maps, and the file called
  TRAB_TH is the whole-SEG one while TRAB_TH_old is the Script-32 one -- the reverse of what the names
  suggest.  Its TRAB_TH was produced by an evaluation outside these three scripts: its processing log
  shows D3P_SetAllNonZeroOW to 126, D3P_Concatenate in overlay and D3P_BoundingBoxCut before the distance
  transform, against the gobj p5mask.gobj rather than the trabecular contour.
                            5c the cortical pore cascade (ipldt.porosity.pore_cascade) on the rendered cortical
                               contour and CORT_SEG, on the input AIM's grid -> the PORE map and Ct.Po
                               (compute_porosity, on by default)
                            6  JSON + CSV report (IPL's statistics + Tb.N + BV/TV + Ct.Po [+ BMD]) and the
                               masks, SEG, PORE map and dt maps as NIfTI (default) or AIM (input AIM header via
                               ipldt.io.write_aim)
  Reference runner        run_on_ipl_masks(...) evaluates step 5 on IPL's own SEG / TRAB_MASK / CORT_MASK
                          and counts voxel mismatches against IPL's TRAB_TH / TRAB_SP / TRAB_1N / CORT_TH.

Grids: IPL runs the trabecular dt_* on SEG.AIM, which is /bounding_box_cut -border 0 of the assembled
segmentation, and Ct.Th on CORT_MASK.AIM (the bounding box of the cortical raster).  The vector distance
transform treats 'outside the image' as object, so the box matters at its faces; this module reproduces
those boxes and pastes the results back onto the input grid by global position (ipldt.io.align_to).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.fft import fftn as _fftn, ifftn as _ifftn

from . import __version__
from . import ipl_ops as ops
from .contour import render_volume
from .core import DTResult, dt_number, dt_spacing, dt_thickness
from .io import align_to, read_aim, write_aim, write_nifti
from .step1 import RADIUS, TIBIA, Step1Params, cort_trab_separation

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - the wrappers need SimpleITK; the numpy layer does not
    sitk = None


# ============================================================================================ parameters
@dataclass
class DTParams:
    """IPL's dt_thickness / dt_spacing / dt_number parameters (Script 32 values by default)."""
    ridge_epsilon: float = 0.9
    assign_epsilon: float = 0.5
    peel_iter: int = -1
    version: int = 3
    suppress_boundary: int = 2

    def kwargs(self):
        return asdict(self)


IPL_SCRIPT32 = DTParams()

# Script 32 constants (validated against IPL's masks and SEG)
VOXEL_SIZE_MM_DEFAULT = 0.0607      # XtremeCT II standard protocol
# Laplace-Hamming element sizes (2026-09-13). IPL's fft_laplace_hamming uses the AIM header's THREE element
# sizes per axis (|k|^2 = kx^2 + ky^2 + kz^2 with k_i in cycles/mm from el_i; its log prints the per-axis
# physical lengths 62.156 / 31.078 / 15.539 mm of the 1024 x 512 x 256 padded volume) and the radial low-pass
# cut-off at 0.3 / el_z (log: lp_phys_freq 4.942469 = 0.3 / 0.0606984 and nyquist 8.237448 = 1 / (2 el_z);
# z and min(el) cannot be told apart on any header seen). The header values are export-dependent (a full
# 504-slice export carries 139852/2304, 139852/2304, 30592/504 um; a cropped export its own Orig-ISQ-Dim
# ratios), so they must be read from the AIM, never assumed: with 0.0607 isotropic the reimplementation differed from
# IPL's SEG by 184 voxels on PFJ-0be66a, with the per-axis header values by 15 (December evaluation, IPL's own
# contour renderings).
LH_EL_SIZE_FALLBACK_MM = (139852.0 / 2304.0 / 1000.0, 139852.0 / 2304.0 / 1000.0, 30592.0 / 504.0 / 1000.0)  # no-header fallback
# Step 3 (the compartment masks) takes its parameters from ipldt.step1.Step1Params: sigma 2 / support 3 and
# 500..3000 mgHA for /seg_gauss, peel0 6, erode / dilate 3, close1 15, open_ 15, corner_erode / corner_dilate 3,
# corner_min 200000 (tibia, Script 32) or 800 (radius, Script 33), corner_max 500000, close2 50 (tibia) or 30
# (radius), slicewise 50..100 %.  The former module constants SEG_GAUSS_SIGMA, CORT_LOWER_BMD, CORT_UPPER_BMD,
# PEEL_ITER, TRAB_*_R, CORNER_* and the chamfer-3-4-5 SimpleITK helpers (threshold radius + 1 instead of IPL's
# 3N + 2, background-padded erosion, SmoothingRecursiveGaussian instead of IPL's float32 kernel) were refuted
# by the P16 stage exports and removed on 2026-09-13.
SITE_PARAMS = {"tibia": TIBIA, "radius": RADIUS}
LP_CUT_OFF_FREQ = 0.3               # IPL fft_laplace_hamming
LAPLACE_EPS = 0.45
HAMMING_AMP = 1.0
NORM_MAX_VALUE = 200000.0           # IPL norm_max -max 200000 -type_out short
INT16_MAX = 32767
LH_THRESHOLD = 15564                # 475/1000 * 32767 (threshold -lower_in_perm 475)
CC_MIN_VOXELS_TRAB = 70             # cl_nr_extract min_number, trab_seg (Script 32 line 441)
CC_MIN_VOXELS_CORT = 35             # cl_nr_extract min_number, cort_seg (Script 32 line 423)
SEG_VALUE_CORT, SEG_VALUE_TRAB = 127, 126


# ============================================================================================ logging
class Logger:
    """print + collect; run_pipeline writes the lines to <base>_pipeline.log."""

    def __init__(self, echo=True):
        self.lines = []
        self.echo = echo
        self.t0 = time.time()

    def __call__(self, msg):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        self.lines.append(line)
        if self.echo:
            print(line, flush=True)


def _log_or_none(log):
    return log if log is not None else Logger(echo=False)


# ============================================================================================ grids
def bbox_cut(arr, pos, border=0):
    """IPL /bounding_box_cut -border b b b: crop a (z, y, x) volume to the bounding box of its
    non-zero voxels; returns dict(data, dim (x, y, z), pos (x, y, z)) as ipldt.io uses."""
    nz = np.nonzero(arr)
    if nz[0].size == 0:
        raise ValueError("bbox_cut: the volume is empty")
    lo = [max(int(n.min()) - border, 0) for n in nz]
    hi = [min(int(n.max()) + 1 + border, s) for n, s in zip(nz, arr.shape)]
    sub = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return dict(data=sub, dim=(hi[2] - lo[2], hi[1] - lo[1], hi[0] - lo[0]),
                pos=(int(pos[0]) + lo[2], int(pos[1]) + lo[1], int(pos[2]) + lo[0]))


def volume(data, dim, pos):
    """dict(data, dim, pos) for ipldt.io.align_to."""
    return dict(data=data, dim=tuple(int(d) for d in dim), pos=tuple(int(p) for p in pos))


def render_on_own_box(mask, pos):
    """Render a raw mask raster the way IPL does (togobj_from_aim on the bounding-box-cut raster,
    then the gobj rasterised back): returns the rendered contour on the mask's grid (bool)."""
    box = bbox_cut(np.asarray(mask, bool), pos)
    G = render_volume(box["data"])
    return align_to(volume(G.astype(np.uint8), box["dim"], box["pos"]), (mask.shape[2], mask.shape[1], mask.shape[0]), pos) > 0


# ============================================================================================ SimpleITK helpers
def _need_sitk():
    if sitk is None:
        raise ImportError("SimpleITK is required for the ipldt.ormir image wrappers")


def sitk_to_bool(img):
    _need_sitk()
    return sitk.GetArrayFromImage(img) > 0


def array_to_sitk(arr, ref_img, dtype=None):
    """(z, y, x) array -> SimpleITK image with ref_img's geometry."""
    _need_sitk()
    a = np.ascontiguousarray(arr if dtype is None else np.asarray(arr).astype(dtype))
    out = sitk.GetImageFromArray(a)
    out.CopyInformation(ref_img)
    return out


def voxel_size_of(img, voxel_size_mm=None):
    """The scalar voxel size used for IPL's statistics: an explicit value, else the image's x spacing."""
    if voxel_size_mm is not None:
        return float(voxel_size_mm)
    return float(img.GetSpacing()[0])


def _gobj_arg(gobj, rendered):
    """None, a raw raster (rendered inside ipldt with IPL's rules) or {'rendered': raster}."""
    if gobj is None:
        return None
    G = np.asarray(gobj, bool)
    return {"rendered": G} if rendered else G


def _map_image(res, ref_img, units, voxel_size_mm):
    if units == "mm":
        return array_to_sitk(res.map.astype(np.float32) * np.float32(voxel_size_mm), ref_img)
    if units == "voxels":
        return array_to_sitk(res.map, ref_img, np.int16)
    raise ValueError("units must be 'voxels' or 'mm'")


# ============================================================================================ SimpleITK wrappers
def render_volume_sitk(mask_img):
    """IPL contour rendering (/togobj_from_aim -curvature_smooth 1 followed by gobj rasterisation)
    of a binary SimpleITK mask; returns a uint8 0/1 image on the same grid."""
    G = render_volume(sitk_to_bool(mask_img))
    return array_to_sitk(G.astype(np.uint8), mask_img)


def _dt_sitk(fn, obj_img, gobj_img, gobj_rendered, voxel_size_mm, params, backend, units, overrides):
    p = (params or IPL_SCRIPT32).kwargs()
    p.update(overrides)
    vs = voxel_size_of(obj_img, voxel_size_mm)
    obj = sitk_to_bool(obj_img)
    gobj = None
    if gobj_img is not None:
        if gobj_img.GetSize() != obj_img.GetSize():
            raise ValueError("gobj image must be on the object's grid (resample or ipldt.io.align_to first)")
        gobj = _gobj_arg(sitk_to_bool(gobj_img), gobj_rendered)
    res = fn(obj, gobj=gobj, voxel_size_mm=vs, backend=backend, **p)
    return _map_image(res, obj_img, units, vs), res


def dt_thickness_sitk(obj_img, gobj_img=None, gobj_rendered=False, voxel_size_mm=None, params=None,
                      backend="auto", units="voxels", **overrides):
    """IPL /dt_thickness on a binary SimpleITK image.

    obj_img        the object (e.g. IPL's SEG for Tb.Th, or the cortical compartment mask for Ct.Th)
    gobj_img       the contour mask (raw raster rendered with IPL's rules unless gobj_rendered=True)
    voxel_size_mm  scalar for the statistics; default = x spacing of obj_img
    params         DTParams (Script 32 by default); keyword overrides (ridge_epsilon=..., version=...) win
    units          'voxels' (int16 diameters, bit-identical to IPL's AIM) or 'mm' (float32)
    Returns (map image, ipldt.DTResult)  -- DTResult.report holds Th_mm, Th_sd_mm, Th_max_mm, Th_skew,
    Th_kurtosis, valid_fraction as IPL's 'Get Statistics' block prints them."""
    return _dt_sitk(dt_thickness, obj_img, gobj_img, gobj_rendered, voxel_size_mm, params, backend, units, overrides)


def dt_spacing_sitk(seg_img, gobj_img=None, gobj_rendered=False, voxel_size_mm=None, params=None,
                    backend="auto", units="voxels", **overrides):
    """IPL /dt_spacing (Tb.Sp map) of a whole-bone segmentation, centres and map inside the gobj."""
    return _dt_sitk(dt_spacing, seg_img, gobj_img, gobj_rendered, voxel_size_mm, params, backend, units, overrides)


def dt_number_sitk(seg_img, gobj_img=None, gobj_rendered=False, voxel_size_mm=None, params=None,
                   backend="auto", units="voxels", **overrides):
    """IPL /dt_number (1/Tb.N map); DTResult.report['Tb_N_per_mm'] = 1 / mean(map) = IPL's 'MAT N (1/Th)'."""
    return _dt_sitk(dt_number, seg_img, gobj_img, gobj_rendered, voxel_size_mm, params, backend, units, overrides)


def ipl_trabecular_microarchitecture_sitk(seg_img, trab_mask_img, gobj_rendered=False, voxel_size_mm=None,
                                          params=None, backend="auto", units="voxels"):
    """ORMIR-style bundle: (metrics dict, Tb.Th image, Tb.Sp image, 1/Tb.N image) from IPL's SEG
    (cortical + trabecular bone) and the trabecular contour mask, as Script 32 evaluates them."""
    vs = voxel_size_of(seg_img, voxel_size_mm)
    seg = sitk_to_bool(seg_img)
    G = sitk_to_bool(trab_mask_img)
    if not gobj_rendered:
        G = render_volume(G)
    out = ipl_morphometry(seg, G, voxel_size_mm=vs, params=params, backend=backend, which=("TRAB_TH", "TRAB_SP", "TRAB_1N"))
    r = out["results"]
    return (out["metrics"], _map_image(r["TRAB_TH"], seg_img, units, vs), _map_image(r["TRAB_SP"], seg_img, units, vs),
            _map_image(r["TRAB_1N"], seg_img, units, vs))


def ipl_cortical_thickness_sitk(cort_mask_img, gobj_rendered=False, voxel_size_mm=None, params=None,
                                backend="auto", units="voxels"):
    """Ct.Th as Script 32 computes it: dt_thickness of the cortical COMPARTMENT mask raster with the
    cortical gobj (IPL evaluates the compartment, not the cortical bone).  Returns (metrics, image)."""
    vs = voxel_size_of(cort_mask_img, voxel_size_mm)
    cm = sitk_to_bool(cort_mask_img)
    G = cm if gobj_rendered else render_volume(cm)
    out = ipl_morphometry(None, None, cort_mask=cm, cort_gobj=G, voxel_size_mm=vs, params=params, backend=backend, which=("CORT_TH",))
    return out["metrics"], _map_image(out["results"]["CORT_TH"], cort_mask_img, units, vs)


# ============================================================================================ dt stage (numpy)
def _ipl_stats(res, prefix, name, denominator=None):
    r = res.report
    n = int(r[f"{name}_n_voxels"])
    d = {f"{prefix}_mm": r[f"{name}_mm"], f"{prefix}_sd_mm": r[f"{name}_sd_mm"], f"{prefix}_max_mm": r[f"{name}_max_mm"],
         f"{prefix}_skew": r.get(f"{name}_skew", 0.0), f"{prefix}_kurtosis": r.get(f"{name}_kurtosis", 0.0),
         f"{prefix}_mean_voxels": r[f"{name}_mm"] / r["voxel_size_mm"] if r["voxel_size_mm"] else 0.0,
         f"{prefix}_max_voxels": int(res.map.max()), f"{prefix}_n_voxels": n,
         f"{prefix}_valid_fraction": (n / denominator if denominator else r["valid_fraction"])}
    return d


def ipl_morphometry(seg, trab_gobj, cort_mask=None, cort_gobj=None, voxel_size_mm=VOXEL_SIZE_MM_DEFAULT, params=None,
                    backend="auto", which=("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH"), trab_seg=None, log=None):
    """Script 32's dt stage on numpy volumes.

    seg        bool (z, y, x): the whole-bone SEG (cortical + trabecular bone) on IPL's SEG grid
    trab_gobj  bool, RENDERED trabecular contour on the same grid
    cort_mask  bool, the cortical compartment raster (its own grid); cort_gobj its RENDERED contour
    trab_seg   optional bool on the SEG grid: IPL's TRAB_SEG, for the OLD Tb.Th definition ('TRAB_TH_old')
    which      subset of TRAB_TH, TRAB_SP, TRAB_1N, CORT_TH, TRAB_TH_old
    Returns dict(results={name: DTResult}, metrics={IPL statistics + Tb.N + BV/TV}, timing_s)."""
    log = _log_or_none(log)
    p = (params or IPL_SCRIPT32).kwargs()
    vs = float(voxel_size_mm)
    results, metrics, timing = {}, {}, {}
    if any(w in which for w in ("TRAB_TH", "TRAB_SP", "TRAB_1N", "TRAB_TH_old")):
        seg = np.ascontiguousarray(seg, dtype=bool)
        G = np.ascontiguousarray(trab_gobj, dtype=bool)
        gobj = {"rendered": G}
        bv = int((seg & G).sum())
        tv = int(G.sum())
        metrics.update({"BV_TV": bv / tv if tv else 0.0, "BV_voxels": bv, "TV_voxels": tv,
                        "BV_mm3": bv * vs ** 3, "TV_mm3": tv * vs ** 3})
        if "TRAB_TH" in which:
            t = time.time()
            res = dt_thickness(seg, gobj=gobj, voxel_size_mm=vs, backend=backend, **p)
            results["TRAB_TH"] = res
            metrics.update(_ipl_stats(res, "Tb_Th", "Th", denominator=bv))
            timing["TRAB_TH"] = time.time() - t
            log(f"  Tb.Th  (dt_thickness SEG | trab gobj): {metrics['Tb_Th_mm']:.6f} mm  "
                f"({metrics['Tb_Th_mean_voxels']:.4f} vox, centres {int(res.centres.sum()):,d}) [{timing['TRAB_TH']:.1f}s]")
        if "TRAB_SP" in which:
            t = time.time()
            res = dt_spacing(seg, gobj=gobj, voxel_size_mm=vs, backend=backend, **p)
            results["TRAB_SP"] = res
            metrics.update(_ipl_stats(res, "Tb_Sp", "Sp"))
            timing["TRAB_SP"] = time.time() - t
            log(f"  Tb.Sp  (dt_spacing SEG | trab gobj):   {metrics['Tb_Sp_mm']:.6f} mm  "
                f"({metrics['Tb_Sp_mean_voxels']:.4f} vox, centres {int(res.centres.sum()):,d}) [{timing['TRAB_SP']:.1f}s]")
        if "TRAB_1N" in which:
            t = time.time()
            res = dt_number(seg, gobj=gobj, voxel_size_mm=vs, backend=backend, **p)
            results["TRAB_1N"] = res
            metrics.update(_ipl_stats(res, "Tb_1N", "inv_N"))
            metrics["Tb_N_per_mm"] = res.report["Tb_N_per_mm"]
            timing["TRAB_1N"] = time.time() - t
            log(f"  1/Tb.N (dt_number SEG | trab gobj):    {metrics['Tb_1N_mm']:.6f} mm -> Tb.N {metrics['Tb_N_per_mm']:.6f} /mm "
                f"(centres {int(res.centres.sum()):,d}) [{timing['TRAB_1N']:.1f}s]")
        if "TRAB_TH_old" in which and trab_seg is not None:
            t = time.time()
            ts = np.ascontiguousarray(trab_seg, dtype=bool)
            res = dt_thickness(ts, gobj=gobj, voxel_size_mm=vs, backend=backend, **p)
            results["TRAB_TH_old"] = res
            metrics.update(_ipl_stats(res, "Tb_Th_old", "Th"))
            timing["TRAB_TH_old"] = time.time() - t
            log(f"  Tb.Th old (dt_thickness TRAB_SEG | trab gobj): {metrics['Tb_Th_old_mm']:.6f} mm [{timing['TRAB_TH_old']:.1f}s]")
    if "CORT_TH" in which and cort_mask is not None:
        t = time.time()
        cm = np.ascontiguousarray(cort_mask, dtype=bool)
        Gc = np.ascontiguousarray(cort_gobj if cort_gobj is not None else cm, dtype=bool)
        res = dt_thickness(cm, gobj={"rendered": Gc}, voxel_size_mm=vs, backend=backend, **p)
        results["CORT_TH"] = res
        metrics.update(_ipl_stats(res, "Ct_Th", "Th", denominator=int((cm & Gc).sum())))
        metrics["Ct_V_voxels"] = int(Gc.sum())
        metrics["Ct_V_mm3"] = int(Gc.sum()) * vs ** 3
        timing["CORT_TH"] = time.time() - t
        log(f"  Ct.Th  (dt_thickness CORT_MASK | cort gobj): {metrics['Ct_Th_mm']:.6f} mm  "
            f"({metrics['Ct_Th_mean_voxels']:.4f} vox, centres {int(res.centres.sum()):,d}) [{timing['CORT_TH']:.1f}s]")
    metrics["voxel_size_mm"] = vs
    return dict(results=results, metrics=metrics, timing_s=timing, params=p, backend=backend)


# ============================================================================================ SEG assembly helper
def cl_nr_extract(bm, min_voxels):
    """IPL /cl_nr_extract on a boolean volume: drop 6-connected components smaller than min_voxels
    (the connectivity that reproduces IPL's SEG.AIM; the labelling is ipldt.ipl_ops.label6, the same
    routine the step-1 reimplementation uses on volume dicts)."""
    lab, _, cnt = ops.label6(np.asarray(bm, bool))
    keep = cnt >= int(min_voxels)
    keep[0] = False
    return keep[lab]


# ============================================================================================ pipeline steps
def step1_load_aim(aim_path, log=None):
    """(1) HU image via ORMIR's file_reader (ITK ScancoImageIO converts native -> HU), the native int16
    volume via ipldt.io.read_aim (what IPL's fft_laplace_hamming sees), and the calibration constants."""
    _need_sitk()
    import itk
    from ormir_xct.core.util.file_reader import file_reader
    log = _log_or_none(log)
    log("STEP 1 - loading AIM ...")
    image_io = itk.ScancoImageIO.New()
    image_io.SetFileName(aim_path)
    image_io.ReadImageInformation()
    calib = dict(mu_scaling=float(image_io.GetMuScaling()), mu_water=float(image_io.GetMuWater()),
                 rescale_slope=float(image_io.GetRescaleSlope()), rescale_intercept=float(image_io.GetRescaleIntercept()))
    log(f"  calibration: {calib}")
    img_hu = sitk.Cast(file_reader(aim_path), sitk.sitkFloat32)
    native = read_aim(aim_path)                      # data (z, y, x) int16, dim / pos (x, y, z), header, el_size_mm
    if tuple(img_hu.GetSize()) != tuple(native["dim"]):
        raise RuntimeError(f"file_reader size {img_hu.GetSize()} != AIM header dim {native['dim']}")
    a = sitk.GetArrayFromImage(img_hu)
    log(f"  dim {native['dim']} pos {native['pos']} el_size {tuple(round(e, 6) for e in native['el_size_mm'])} mm; "
        f"HU [{a.min():.0f}, {a.max():.0f}], native int16 [{native['data'].min()}, {native['data'].max()}]")
    return img_hu, native, calib


def step2_autocontour(img_hu, calib, log=None):
    """(2) ORMIR autocontour -> periosteal (proximal) mask, uint8 0/1 on the image grid."""
    from ormir_xct.core.segmentation.autocontour.autocontour import autocontour
    log = _log_or_none(log)
    log("STEP 2 - ORMIR autocontour (periosteal mask) ...")
    _, prx_mask, _ = autocontour(img_hu, mu_water=calib["mu_water"], rescale_slope=calib["rescale_slope"],
                                 rescale_intercept=calib["rescale_intercept"])
    prx_mask = sitk.Cast(prx_mask > 0, sitk.sitkUInt8)
    log(f"  periosteal voxels: {int(sitk.GetArrayFromImage(prx_mask).sum()):,d}")
    return prx_mask


def step1_params_for(site="tibia", params=None):
    """The Step1Params of a site name: 'tibia' = Script 32 (TIBIA), 'radius' = Script 33 (RADIUS); an explicit
    Step1Params wins over the name."""
    if params is not None:
        if not isinstance(params, Step1Params):
            raise TypeError(f"params must be an ipldt.step1.Step1Params, not {type(params).__name__}")
        return params
    key = str(site).lower()
    if key not in SITE_PARAMS:
        raise ValueError(f"site must be one of {sorted(SITE_PARAMS)}, not {site!r}")
    return SITE_PARAMS[key]


def step1_calibration(native, calib=None):
    """The density calibration /seg_gauss converts its mgHA thresholds with: the AIM processing log's
    'Density: slope', 'Density: intercept' and 'Mu_Scaling' (ipl_ops.calibration_from_proclog -- what IPL
    itself reads) and, when the AIM carries no processing log, ITK ScancoImageIO's rescale slope /
    intercept and mu_scaling as step1_load_aim returns them (the same header fields: on PFJ-0be66a both give
    1619.07703 / -394.095001 / 8192 and the native thresholds 4524 / 17173).
    Returns (dict(slope, intercept, mu_scaling), source) with source 'proclog' or 'itk_scanco_header'."""
    try:
        return ops.calibration_from_proclog(native), "proclog"
    except ValueError as exc:
        if calib is None:
            raise ValueError("the AIM carries no density calibration in its processing log and no ITK "
                             "calibration dict was given (step1_load_aim's third return value)") from exc
        cal = dict(slope=float(calib["rescale_slope"]), intercept=float(calib["rescale_intercept"]),
                   mu_scaling=float(calib["mu_scaling"]))
        return cal, "itk_scanco_header"


def step3_trab_cort_seg(native, prx_mask, params=TIBIA, calib=None, log=None):
    """(3) Scanco Script 32 / 33 STEP 1: IPL's cortical / trabecular compartment masks from the native
    greyscale and the periosteal contour, reimplemented command for command (ipldt.step1.cort_trab_separation on
    the reimplemented ipldt.ipl_ops commands: /seg_gauss with IPL's float32 kernel and per-pass short truncation, metric-11
    chamfer morphology with the 3N + 2 threshold and IPL's boundary rules, /cl_ow_rank_extract, /cl_nr_extract,
    /cl_slicewise_extractow, /gobj_maskaimpeel_ow, /subtract_aims, /add_aims, /bounding_box_cut).

    native    the native int16 volume dict of step1_load_aim (ipldt.io.read_aim: data (z, y, x), dim / pos
              (x, y, z), proclog) -- IPL's /seg_gauss runs on native numbers, not on HU
    prx_mask  the periosteal mask (SimpleITK, non-zero = inside) on the image grid = the AIM grid.  It is cut
              to its bounding box and rendered with IPL's contour rules (ipldt.contour.render_volume), which
              is IPL's /gobj_to_aim of the periosteal gobj = the chain's stage 00
    params    ipldt.step1.Step1Params: TIBIA (Script 32, default) or RADIUS (Script 33); step1_params_for()
    calib     step1_load_aim's ITK calibration dict; used only when the AIM has no processing log
    Returns (cort, trab, info): CORT_MASK / TRAB_MASK as uint8 0/127 SimpleITK images on the image grid (the
    chain's stages 28 / 29 pasted back by global position) and the step-1 info dict (params, calibration and
    its source, native thresholds, the seg_gauss box, per-stage voxel counts and grids, timings).
    Verified (tests/test_ormir_step3.py): with IPL's own periosteal contour (raw CORT_MASK | TRAB_MASK of
    PFJ-0be66a, which equals IPL's rendered periosteal gobj) the two images equal the raw X2420448_CORT_MASK.AIM
    (7,154,580 voxels) and X2420448_TRAB_MASK.AIM (20,329,639) by position with 0 mismatches."""
    _need_sitk()
    log = _log_or_none(log)
    p = step1_params_for(params=params)
    log(f"STEP 3 - cortical / trabecular compartments (Script 32 STEP 1 reimplementation, ipldt.step1; corner_min "
        f"{p.corner_min:,d}, close2 {p.close2}, peel0 {p.peel0}) ...")
    dim, pos = tuple(int(d) for d in native["dim"]), tuple(int(q) for q in native["pos"])
    prx = sitk_to_bool(prx_mask)
    if prx.shape != tuple(dim[::-1]):
        raise ValueError(f"the periosteal mask {prx.shape[::-1]} is not on the AIM grid {dim}")
    t = time.time()
    box = bbox_cut(prx, pos)                                                  # the gobj's grid = the raster's box
    per = ops.mask_vol(render_volume(box["data"]), box["dim"], box["pos"])   # IPL's /gobj_to_aim (stage 00)
    n_per = int(np.count_nonzero(per["data"]))
    log(f"  periosteal raster {int(prx.sum()):,d} -> rendered contour {n_per:,d} voxels on its box dim {box['dim']} "
        f"pos {box['pos']} [{time.time() - t:.1f}s]")
    cal, src = step1_calibration(native, calib)
    res = cort_trab_separation(native, per, p, log=lambda s: log("    " + s), calibration=cal)
    info = res["info"]
    info["calibration_source"] = src
    info["periosteal"] = dict(raw_voxels=int(prx.sum()), rendered_voxels=n_per, dim=box["dim"], pos=box["pos"])
    thr = info["thresholds"]
    log(f"  seg_gauss {p.lower_mgha:g} / {p.upper_mgha:g} mgHA -> native {thr['lower_native']} / {thr['upper_native']} "
        f"(calibration from {src}: slope {cal['slope']:.5f}, intercept {cal['intercept']:.6f}, mu_scaling {cal['mu_scaling']:g}); "
        f"seg_gauss box dim {info['box']['dim']} pos {info['box']['pos']}")
    cort = align_to(res["cort"], dim, pos) != 0          # set = non-zero (29 is int8: /subtract_aims output)
    trab = align_to(res["trab"], dim, pos) != 0
    cn, tn = int(cort.sum()), int(trab.sum())
    if cn != info["counts"]["28_cortfinal"] or tn != info["counts"]["29_trabfinal"]:
        raise RuntimeError("the compartment masks do not fit the AIM grid (pasting by position lost voxels)")
    cort_out = array_to_sitk(cort.astype(np.uint8) * 127, prx_mask)
    trab_out = array_to_sitk(trab.astype(np.uint8) * 127, prx_mask)
    log(f"  CORT_MASK {cn:,d} voxels ({100 * cn / max(n_per, 1):.1f}% of the contour), TRAB_MASK {tn:,d} "
        f"({100 * tn / max(n_per, 1):.1f}%); grids 28 {info['grids']['28_cortfinal']}, 29 {info['grids']['29_trabfinal']} "
        f"[{info['timings']['total']:.1f}s]")
    return cort_out, trab_out, info


LH_PAD_OFFSET = "ceil"              # IPL's D3P_FFT_AdjustDimensionsMirror data offset: derived by probe 21 (see lh_pad_plan)
LH_PAD_OFFSETS = ("ceil", "floor")
LH_DTYPE = "float32"                # the FFT / filter arithmetic of lh_filter_core; "float64" is an opt-in (see there)
LH_DTYPES = ("float32", "float64")


def _npow2(n):
    """IPL's per-axis redimension for /fft_laplace_hamming: the next power of two >= n.  Probe 15's log names the
    routine D3P_FFT_AdjustDimensionsMirror and prints the box on its `dim:` line (748 x 353 x 170 -> 1024 x 512 x
    256); probe 21's log prints 64 for every 62 / 63 / 33 / 34 input and omits the routine altogether when n is
    already a power of two, so such an input is not padded at all."""
    return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))


def lh_pad_plan(shape, pad_offset=LH_PAD_OFFSET):
    """Where D3P_FFT_AdjustDimensionsMirror puts an array of `shape` (array order, i.e. (z, y, x)) inside its
    power-of-two box, and the numpy pad widths / crop slices that realise it.  Per axis, with d = nt - n:
      'ceil'   lo = d - d // 2  = ceil(d / 2): when d is odd the EXTRA padding voxel goes BEFORE the data.
               THIS IS IPL'S RULE -- probe 21 (prediction timestamped 2026-09-16T05:01:34Z, scanner run 2026-09-17)
               fed 31 impulse / plane / slab / real-data phantoms of 62, 63, 64, 33 and 34 voxels straight into
               /fft_laplace_hamming and compared IPL's FLOAT, SHORT and thresholded exports with every candidate of
               4 offsets x 5 mirror modes (+ per-axis floor / ceil mixes): 'ceil' + numpy 'reflect' (the
               edge-EXCLUSIVE mirror, the face voxel not repeated) is the only candidate with ZERO refutations at the
               FLOAT and SEG levels; floor / left / right and symmetric / edge / constant / wrap are each refuted
               independently at BOTH levels by x63i00 / x63i01.  Model-free readouts: x63i61_sg (non-empty only under
               floor + reflect) came back EMPTY, x63i01_sg (non-empty only under ceil + reflect) came back with 6
               voxels set; the corner phantom a63c01 agrees.  A 64^3 input is not padded, so the two rules coincide
               there (and on any power-of-two axis).
      'floor'  lo = d // 2: the rule ipldt shipped until 2026-09-17, kept selectable so that every number published
               under it stays reproducible.  On the 2022 / 2024 OS_LH and patella
               volumes it is one voxel off IPL on every odd-padded axis.
    The fill is numpy 'reflect' in both cases.  Returns (pad_widths, inner) with pad_widths a list of (lo, hi) per
    axis and inner the tuple of slices that crop the filtered box back to the input grid."""
    if pad_offset not in LH_PAD_OFFSETS:
        raise ValueError(f"pad_offset must be one of {LH_PAD_OFFSETS} (got {pad_offset!r})")
    pad_widths, inner = [], []
    for n in shape:
        n = int(n)
        d = _npow2(n) - n
        lo = (d - d // 2) if pad_offset == "ceil" else d // 2
        pad_widths.append((lo, d - lo))
        inner.append(slice(lo, lo + n))
    return pad_widths, tuple(inner)


def _lh_float_type(dtype):
    if dtype not in LH_DTYPES:
        raise ValueError(f"dtype must be one of {LH_DTYPES} (got {dtype!r})")
    return np.float32 if dtype == "float32" else np.float64


def _lh_transfer(shape, el, ft, lp_cut_off_freq=LP_CUT_OFF_FREQ):
    """The transfer function H(k) of /fft_laplace_hamming on the padded box `shape` (array order (z, y, x)), in the
    float type `ft`: (2 pi)^2 ((1 - eps) + eps |k|^2) times the radial Hann window (amp 1) cut at 0.3 / el_z, with
    |k| from numpy's fftfreq on the per-axis element sizes.  The element sizes enter as the header's float32 values
    in BOTH types (IPL reads them from the AIM header), so the two types differ only in the arithmetic.  The float32
    branch is, operation for operation, the code lh_filter_core shipped with (the three per-axis k^2 terms summed in
    the same order, so it is identical to the pre-2026-09-17 meshgrid form; tests/test_laplace_hamming_padding.py)."""
    nz, ny, nx = shape
    ex, ey, ez = (ft(np.float32(e)) for e in el)         # IPL: per-axis header spacings
    kx = np.fft.fftfreq(nx, d=float(ex)).astype(ft)
    ky = np.fft.fftfreq(ny, d=float(ey)).astype(ft)
    kz = np.fft.fftfreq(nz, d=float(ez)).astype(ft)
    K2 = (kx * kx)[None, None, :] + (ky * ky)[None, :, None]           # == KX * KX + KY * KY of the meshgrid form
    K2 = K2 + (kz * kz)[:, None, None]                                  # ... + KZ * KZ, the same order of additions
    Kmag = np.sqrt(K2)
    k_lp = ft(lp_cut_off_freq) / ez                   # IPL log: lp_phys_freq 4.942469 = 0.3 / el_z
    #  per measurement: IPL's SEG proclog prints it as fft.lp_cut_off_freq.  116 of the 117 OS_LH measurements
    #  use 0.30000; Diaphyseal/CKD/991161's automatic run used 0.20000 (validate_dataset reads it and passes it).
    half_amp = ft(HAMMING_AMP) * ft(0.5)
    win = np.where(Kmag < k_lp, (ft(1.0) - half_amp) + half_amp * np.cos(np.pi * Kmag / k_lp), ft(0.0)).astype(ft)
    del Kmag
    H = ft((2.0 * np.pi) ** 2) * ((ft(1.0) - ft(LAPLACE_EPS)) + ft(LAPLACE_EPS) * K2) * win
    del K2, win
    return H


def lh_filter_core(volume_float32, el_size_mm, pad_offset=LH_PAD_OFFSET, dtype=LH_DTYPE,
                   lp_cut_off_freq=LP_CUT_OFF_FREQ):
    """IPL /fft_laplace_hamming (eps 0.45, cut-off 0.3, amp 1 = Hann) followed by /norm_max -max 200000 -type_out
    short on `volume_float32` EXACTLY AS GIVEN -- no border duplicate, no crop: what IPL computes when an AIM is read
    and handed straight to the filter (the probe-21 phantoms, and the 2024 diaphyseal script whose
    /fill_offset_duplicate errored).  laplace_hamming_threshold wraps this with Script 32's 1-voxel duplicated border.
    el_size_mm: the (x, y, z) element sizes of the AIM header (IPL's filter is anisotropic: its log prints the
    per-axis physical lengths) with the radial Hann cut-off at 0.3 / el_z (log: lp_phys_freq 4.942630 = 0.3 /
    0.06069643).  pad_offset: see lh_pad_plan.
    dtype: the arithmetic of the padded volume, the transfer function, the forward and the inverse FFT --
    'float32' (LH_DTYPE, the shipped engine: nothing published changes) or 'float64' (an opt-in: the same
    operations in double precision, the result rounded to float32 BEFORE IPL's float -> short conversion below,
    which stays exactly as it is).  The two differ only in rounding; the float64 study of 2026-09-17
    (internal runs, not distributed) measured it against IPL's own exports: on IPL's full-volume probe-15 export float64
    sits at RMS 0.0160 vs 0.0243 float units, 89,114 vs 134,765 +/- 1 short flips and 2 vs 3 SEG flips of 44.9 M;
    the configuration-B SEG residual under 'ceil' drops from 39 to 27 voxels on the 53 radius / tibia
    measurements of that study and from 50 to 37 on the 21 patellae (pooled 89 -> 64; 7 measurements worse, none by more than 2 voxels); the
    CPU cost is about 2.3x on the LH stage (3.5 -> 8.1 s on the 1024 x 512 x 256 box) with about 1.4x the peak
    memory.  It is the documented opt-in, not the default: float32 stays the default so that no published SEG number moves
    (decided 2026-09-17).
    Returns (lh, short): the float32 filter output and IPL's short on the INPUT grid.  The short is
    trunc(float32(lh) * float32(32767 / 200000)) after clipping, which reproduces IPL's _NM exports voxel for voxel
    on the six probe-21 pairs tested (C64IMP, C64F6, C64MID, R64, X63CMB, Y63CMB: 0 mismatches; every other
    ordering or rounding fails).  NOISE FLOOR: against IPL's own FLOAT exports this arithmetic agrees to max |d|
    0.016 - 0.094 float units on peaks of 8.7e4 - 3.2e5 (r64: RMS 0.017): the difference between two FFT
    implementations.  ATTRIBUTION (the float64 verifier, 2026-09-17): the engine's scipy.fft float32 fftn error is
    the largest of the float32 variants measured -- on IPL's probe-15 full volume the engine sits at RMS 0.0243
    float units against IPL's float, a numpy float32 fftn at 0.0199, a scipy per-axis float32 transform at 0.0185,
    and float64 at 0.0160, where float64 is implementation-independent to 6e-8.  Float64 is therefore the point at
    which everything left (RMS 0.0160 vs IPL's float; 64 SEG voxels over the 74 measurements of that study, both
    cohorts) is on IPL's side: its own float32 FFT, which no arithmetic of ours can remove.  It is the TRANSFORM's
    precision that matters, not the transfer function's (H(k) in float64 with a float32 FFT changes nothing: RMS
    0.0238).  An axis-order float32 tweak is NOT adopted as a
    substitute: tuning the transform order to IPL's rounding is unprincipled without IPL's source and fragile
    across library versions.  The residual shows up as +/- 1 flips of the truncated short on the few voxels whose
    scaled value sits within ~0.015 of an integer (8 / 10 / 7 / 581 of 262,144 on C64IMP / C64F6 / C64MID / R64)
    and, on real data, as a handful of threshold decisions per volume.  It is NOT a conversion bug and must not be
    'fixed'."""
    ft = _lh_float_type(dtype)
    el = tuple(float(e) for e in el_size_mm)
    extended = np.asarray(volume_float32, dtype=np.float32)      # IPL's float image of the short input, either way
    pad_widths, inner = lh_pad_plan(extended.shape, pad_offset)
    padded = np.pad(extended, pad_widths, mode="reflect")     # D3P_FFT_AdjustDimensionsMirror: edge-exclusive mirror
    del extended
    if ft is not np.float32:
        padded = padded.astype(ft)                            # exact: every float32 is a float64
    H = _lh_transfer(padded.shape, el, ft, lp_cut_off_freq)
    F = _fftn(padded, workers=-1)                             # complex64 for float32, complex128 for float64
    del padded
    F *= H
    del H
    lh = np.real(_ifftn(F, workers=-1)).astype(np.float32)   # float64 -> float32 HERE, before IPL's conversion
    del F
    lh = np.ascontiguousarray(lh[inner])             # back on the input grid (elementwise steps below commute with the crop)
    scaled = lh * (np.float32(INT16_MAX) / np.float32(NORM_MAX_VALUE))
    np.clip(scaled, -INT16_MAX, INT16_MAX, out=scaled)
    lh_int16 = np.trunc(scaled).astype(np.int16)       # IPL's float -> short conversion truncates
    del scaled
    return lh, lh_int16


def laplace_hamming_threshold(native_int16, voxel_size_mm=None, pad_offset=LH_PAD_OFFSET, dtype=LH_DTYPE,
                              lp_cut_off_freq=LP_CUT_OFF_FREQ):
    """IPL fft_laplace_hamming (eps 0.45, cut-off 0.3, amp 1 = Hann) + norm_max 200000 short (truncating)
    + threshold 475/1000 on the NATIVE int16 volume, with IPL's border duplicate and power-of-2 mirror
    padding.  Returns the raw thresholded bool volume on the input grid (unmasked, uncleaned).
    = Script 32's /bounding_box_cut -border 1 1 1 + /offset_add + /fill_offset_duplicate (np.pad 'edge'), then
    lh_filter_core (the padding, FFT, filter, scale and truncation -- see there for the probe-21 derivation of the
    padding and for the FFT noise floor), then /threshold -lower_in_perm 475 (short >= 15564) and the crop of the
    border.  pad_offset 'ceil' (IPL's rule, the default since 2026-09-17) or 'floor' (the rule shipped before, see
    lh_pad_plan; with it this function is byte-identical to the pre-refactor code).  dtype: the FFT / filter
    arithmetic, 'float32' (LH_DTYPE, shipped) or 'float64' (opt-in; see lh_filter_core).
    voxel_size_mm: the AIM header's (x, y, z) element sizes -- IPL's filter is anisotropic (its log prints
    the per-axis physical lengths) with the radial cut-off at 0.3 / el_z (log: lp_phys_freq 4.942469).
    Pass el_size_mm of ipldt.io.read_aim; a scalar is used for all three axes; None = the full-stack
    protocol triple.  On PFJ-0be66a_R (December evaluation, IPL's own renderings) 15
    voxels differ from IPL's SEG under 'floor'."""
    if voxel_size_mm is None:
        el = LH_EL_SIZE_FALLBACK_MM
    elif np.ndim(voxel_size_mm) == 0:
        el = (float(voxel_size_mm),) * 3
    else:
        el = tuple(float(e) for e in voxel_size_mm)
    native = np.asarray(native_int16)
    extended = np.pad(native, ((1, 1), (1, 1), (1, 1)), mode="edge").astype(np.float32)
    lh, lh_int16 = lh_filter_core(extended, el, pad_offset, dtype, lp_cut_off_freq)
    del lh, extended
    bm = (lh_int16 >= LH_THRESHOLD) & (lh_int16 <= INT16_MAX)
    return bm[1:-1, 1:-1, 1:-1]


def ipl_seg_assembly(bm, prx_raw, cort_gobj, trab_gobj):
    """IPL Script 32 lines 407-452:
        seg      = LH threshold & periosteal RAW raster       (mask FIRST -- IPL's rendered periosteal
                   gobj equals CORT_MASK | TRAB_MASK exactly, 0 voxels differ on PFJ-0be66a, because Script 32
                   partitions gobj_to_aim(periosteal) into the two compartments; the cortical and
                   trabecular masks below ARE re-rendered contours)
        cort_seg = cl_nr_extract(seg, 35) & cortical gobj
        trab_seg = cl_nr_extract(seg, 70) & trabecular gobj
    Returns (cort_seg, trab_seg) bool; SEG = cort_seg * 127 + trab_seg * 126 (lines 458-472)."""
    seg0 = bm & prx_raw
    return cl_nr_extract(seg0, CC_MIN_VOXELS_CORT) & cort_gobj, cl_nr_extract(seg0, CC_MIN_VOXELS_TRAB) & trab_gobj


def step4_render_and_segment(native, prx_mask, cort_out, trab_out, lh_voxel_size_mm=None, log=None):
    """(4) render the three contours with IPL's rules (each on its bounding-box grid, as IPL's
    togobj_from_aim sees them), run the Laplace-Hamming threshold on the native int16 volume and
    assemble IPL's SEG.  Returns dict(G_prx, G_cort, G_trab, lh, cort_seg, trab_seg, seg) on the AIM grid."""
    log = _log_or_none(log)
    pos = native["pos"]
    log("STEP 4 - rendering contours (IPL togobj_from_aim -curvature_smooth 1 + gobj rasterisation) ...")
    t = time.time()
    G = {}
    for name, img in (("prx", prx_mask), ("cort", cort_out), ("trab", trab_out)):
        raw = sitk_to_bool(img)
        G[name] = render_on_own_box(raw, pos)
        log(f"  {name}: raw raster {int(raw.sum()):,d} -> rendered gobj {int(G[name].sum()):,d} voxels")
    log(f"  rendering took {time.time() - t:.1f}s")
    log("STEP 4 - Laplace-Hamming binarization (native int16) + IPL SEG assembly ...")
    t = time.time()
    if lh_voxel_size_mm is None:
        lh_el, src = tuple(float(e) for e in native["el_size_mm"]), "AIM header, per axis (IPL's rule)"
    elif np.ndim(lh_voxel_size_mm) == 0:
        lh_el, src = (float(lh_voxel_size_mm),) * 3, "override, isotropic"
    else:
        lh_el, src = tuple(float(e) for e in lh_voxel_size_mm), "override, per axis"
    log(f"  Laplace-Hamming element sizes (x, y, z) = {lh_el[0]:.7f} / {lh_el[1]:.7f} / {lh_el[2]:.7f} mm ({src}); "
        f"power-of-two padding offset '{LH_PAD_OFFSET}' (probe 21)")
    lh = laplace_hamming_threshold(native["data"], lh_el, pad_offset=LH_PAD_OFFSET)
    cort_seg, trab_seg = ipl_seg_assembly(lh, sitk_to_bool(prx_mask), G["cort"], G["trab"])
    seg = np.zeros(lh.shape, np.uint8)
    seg[cort_seg] = SEG_VALUE_CORT
    seg[trab_seg] = SEG_VALUE_TRAB
    log(f"  LH threshold {int(lh.sum()):,d}; CORT_SEG {int(cort_seg.sum()):,d}, TRAB_SEG {int(trab_seg.sum()):,d}, "
        f"SEG {int((seg > 0).sum()):,d} voxels [{time.time() - t:.1f}s]")
    return dict(G_prx=G["prx"], G_cort=G["cort"], G_trab=G["trab"], lh=lh, cort_seg=cort_seg, trab_seg=trab_seg, seg=seg,
                lh_el_size_mm=list(lh_el), lh_pad_offset=LH_PAD_OFFSET)


def step5_bmd(img_hu, G_trab, G_cort, calib, log=None):
    """ORMIR bmd_masked (HU -> mgHA/ccm) inside the rendered trabecular and cortical contours."""
    from ormir_xct.core.microarchitecture.bmd_masked import bmd_masked
    log = _log_or_none(log)
    out = {}
    for name, G in (("Tb", G_trab), ("Ct", G_cort)):
        m, s = bmd_masked(img_hu, array_to_sitk(G.astype(np.uint8), img_hu), "hu", calib["mu_scaling"], calib["mu_water"],
                          calib["rescale_slope"], calib["rescale_intercept"])
        out[f"{name}_BMD_mgHA_cm3"] = float(m)
        out[f"{name}_BMD_sd_mgHA_cm3"] = float(s)
        log(f"  {name}.BMD = {m:.2f} +- {s:.2f} mgHA/cm3 (ORMIR bmd_masked, rendered {name} contour)")
    return out


# ============================================================================================ output helpers
def _flatten(d, prefix=""):
    rows = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            rows.extend(_flatten(v, key + "."))
        elif isinstance(v, (list, tuple)):
            rows.append((key, " ".join(str(x) for x in v)))
        else:
            rows.append((key, v))
    return rows


def write_report(report, json_path, csv_path=None):
    """JSON (nested) and CSV (Parameter, Value -- the ORMIR trab-microarch layout) reports."""
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=_json_default)
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["Parameter", "Value"])
            for k, v in _flatten(report):
                w.writerow([k, v])


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_volume(path, data, fmt="nifti", el_size_mm=(VOXEL_SIZE_MM_DEFAULT,) * 3, pos=(0, 0, 0), template_header=None):
    """One map / mask to disk: 'nifti' (SimpleITK, origin = pos * el_size so every grid overlays) or
    'aim' (uncompressed char AIM v020 on the template header's grid -- data must be on that grid)."""
    if fmt == "aim":
        if template_header is None:
            raise ValueError("fmt='aim' needs the header of an AIM on the same grid")
        write_aim(path, data, template_header)
    elif fmt in ("nifti", "nii"):
        write_nifti(path, data, el_size_mm, pos)
    else:
        raise ValueError("fmt must be 'nifti' or 'aim'")


def _ext(fmt):
    return ".AIM" if fmt == "aim" else ".nii.gz"


def _map_data(res, units, vs):
    return res.map.astype(np.float32) * np.float32(vs) if units == "mm" else res.map.astype(np.int16)


# ============================================================================================ run_pipeline
def run_pipeline(aim_path, out_dir, map_format="nifti", backend="auto", voxel_size_mm=None, params=None,
                 map_units="voxels", compute_bmd=True, compute_porosity=True, save_masks=True, lh_voxel_size_mm=None,
                 site="tibia", step1_params=None, log=None):
    """AIM -> IPL-style morphometry.  See the module docstring for the six steps.

    aim_path        Scanco AIM (greyscale, native int16)
    out_dir         written: <base>_{PRX,CORT,TRAB}_MASK (raw rasters) and *_GOBJ (rendered contours),
                    <base>_{SEG,CORT_SEG,TRAB_SEG}, the maps <base>_{TRAB_TH,TRAB_SP,TRAB_1N,CORT_TH},
                    <base>_report.json / .csv, <base>_pipeline.log -- all on the input AIM grid
    map_format      'nifti' (default) or 'aim' (input AIM header, ipldt.io.write_aim)
    backend         'auto' (CuPy when available) / 'gpu' / 'cpu' -- identical results
    voxel_size_mm   scalar for the mm statistics; default = the AIM header's x element size
    params          DTParams (IPL Script 32 defaults)
    map_units       'voxels' (int16, IPL's integer diameters) or 'mm' (float32, NIfTI only)
    compute_bmd     also ORMIR bmd_masked inside the rendered contours (Tb.BMD, Ct.BMD)
    compute_porosity  also IPL's cortical pore cascade inside the rendered cortical contour (Ct.Po, PORE map)
    lh_voxel_size_mm  the element size(s) the Laplace-Hamming filter uses; None (default) = the AIM header's
                    (x, y, z) element sizes, IPL's own rule; a scalar (isotropic) or a triple overrides
    site            'tibia' (Script 32 STEP 1 parameters, default) or 'radius' (Script 33: corner_min 800,
                    close2 30) for the compartment masks of step 3
    step1_params    an explicit ipldt.step1.Step1Params for step 3 (overrides `site`)
    Returns the report dict."""
    _need_sitk()
    if map_units == "mm" and map_format == "aim":
        raise ValueError("AIM maps are integer diameters in voxels; use map_units='voxels' or map_format='nifti'")
    log = log if log is not None else Logger()
    params = params or IPL_SCRIPT32
    p1 = step1_params_for(site, step1_params)
    site_name = str(site).lower() if step1_params is None else "custom"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(aim_path))[0]
    t_start = time.time()
    timing = {}

    t = time.time()
    img_hu, native, calib = step1_load_aim(aim_path, log)
    dim, pos, el = native["dim"], native["pos"], native["el_size_mm"]
    vs = float(voxel_size_mm) if voxel_size_mm is not None else float(el[0])
    fmt = map_format
    hdr = native["header"]
    timing["1_load"] = time.time() - t

    def save(name, arr):
        write_volume(os.path.join(out_dir, f"{base}_{name}{_ext(fmt)}"), arr, fmt, el, pos, hdr)

    t = time.time()
    prx_mask = step2_autocontour(img_hu, calib, log)
    timing["2_autocontour"] = time.time() - t
    t = time.time()
    cort_out, trab_out, s1 = step3_trab_cort_seg(native, prx_mask, p1, calib, log)
    timing["3_compartments"] = time.time() - t
    t = time.time()
    s4 = step4_render_and_segment(native, prx_mask, cort_out, trab_out, lh_voxel_size_mm, log)
    timing["4_render_segment"] = time.time() - t

    if save_masks:
        save("PRX_MASK", sitk_to_bool(prx_mask).astype(np.uint8) * 127)
        save("CORT_MASK", sitk_to_bool(cort_out).astype(np.uint8) * 127)
        save("TRAB_MASK", sitk_to_bool(trab_out).astype(np.uint8) * 127)
        save("PRX_GOBJ", s4["G_prx"].astype(np.uint8) * 127)
        save("CORT_GOBJ", s4["G_cort"].astype(np.uint8) * 127)
        save("TRAB_GOBJ", s4["G_trab"].astype(np.uint8) * 127)
        save("SEG", s4["seg"])
        save("CORT_SEG", s4["cort_seg"].astype(np.uint8) * 127)
        save("TRAB_SEG", s4["trab_seg"].astype(np.uint8) * 127)

    # (5) IPL's grids: SEG = bounding box of the assembled segmentation; Ct.Th on the cortical raster's box
    log("STEP 5 - IPL dt_thickness / dt_spacing / dt_number ...")
    t = time.time()
    seg_box = bbox_cut(s4["seg"], pos)
    full = lambda a: volume(a.astype(np.uint8), dim, pos)                                     # noqa: E731
    G_trab_box = align_to(full(s4["G_trab"]), seg_box["dim"], seg_box["pos"]) > 0
    cort_raw = sitk_to_bool(cort_out)
    cort_box = bbox_cut(cort_raw, pos)
    G_cort_box = align_to(full(s4["G_cort"]), cort_box["dim"], cort_box["pos"]) > 0
    log(f"  SEG grid dim {seg_box['dim']} pos {seg_box['pos']}; CORT_MASK grid dim {cort_box['dim']} pos {cort_box['pos']}")
    m = ipl_morphometry(seg_box["data"] > 0, G_trab_box, cort_mask=cort_box["data"], cort_gobj=G_cort_box, voxel_size_mm=vs,
                        params=params, backend=backend, log=log)
    timing["5_dt"] = time.time() - t
    grids = {"TRAB_TH": seg_box, "TRAB_SP": seg_box, "TRAB_1N": seg_box, "CORT_TH": cort_box}
    for name, res in m["results"].items():
        g = grids[name]
        data = align_to(volume(_map_data(res, map_units, vs), g["dim"], g["pos"]), dim, pos)      # back onto the AIM grid
        save(name, data)

    bmd = {}
    if compute_bmd:
        log("STEP 5b - BMD (ORMIR bmd_masked) ...")
        t = time.time()
        try:
            bmd = step5_bmd(img_hu, s4["G_trab"], s4["G_cort"], calib, log)
        except Exception as exc:   # BMD is a courtesy: never fail the morphometry for it
            log(f"  BMD skipped: {type(exc).__name__}: {exc}")
        timing["5b_bmd"] = time.time() - t

    poro = {}
    if compute_porosity:
        log("STEP 5c - cortical pore cascade (Burghardt) + Ct.Po ...")
        t = time.time()
        try:
            from . import porosity as _por
            _cr = volume(s4["G_cort"].astype(np.uint8) * 127, dim, pos)
            _cs = volume(s4["cort_seg"].astype(np.uint8) * 127, dim, pos)
            _pore = _por.pore_cascade(_cr, _cs)["pore"]
            _po = _por.ct_po(_pore, _cr)
            poro = {"Ct_Po": float(_po["ct_po"]), "Ct_Po_pore_voxels": int(_po["pore_voxels"]),
                    "Ct_Po_compartment_voxels": int(_po["mask_voxels"])}
            log(f"  Ct.Po = {_po['ct_po']:.5f}  ({_po['pore_voxels']:,d} pore voxels of "
                f"{_po['mask_voxels']:,d} in the cortical compartment)")
            if save_masks:
                save("PORE", (align_to(_pore, dim, pos) != 0).astype(np.uint8) * 127)
        except Exception as exc:   # porosity is a courtesy: never fail the morphometry for it
            log(f"  porosity skipped: {type(exc).__name__}: {exc}")
        timing["5c_porosity"] = time.time() - t

    timing["total"] = time.time() - t_start
    report = {
        "sample": base, "input_aim": os.path.abspath(aim_path), "ipldt_version": __version__,
        "voxel_size_mm": vs, "el_size_mm": [float(e) for e in el], "dim_xyz": list(dim), "pos_xyz": list(pos),
        "grids": {"SEG": {"dim_xyz": list(seg_box["dim"]), "pos_xyz": list(seg_box["pos"])},
                  "CORT_MASK": {"dim_xyz": list(cort_box["dim"]), "pos_xyz": list(cort_box["pos"])}},
        "calibration": calib,
        "parameters": {**params.kwargs(), "backend": m["backend"], "lh_voxel_size_mm": s4["lh_el_size_mm"],
                       "lh_threshold": LH_THRESHOLD, "cl_nr_extract_min_cort": CC_MIN_VOXELS_CORT,
                       "cl_nr_extract_min_trab": CC_MIN_VOXELS_TRAB, "map_format": fmt, "map_units": map_units,
                       "site": site_name},
        "step1": {"site": site_name, "params": s1["params"], "calibration": s1["calibration"],
                  "calibration_source": s1["calibration_source"], "thresholds": s1["thresholds"],
                  "periosteal": s1["periosteal"], "seg_gauss_box": s1["box"],
                  "counts": {k: v for k, v in s1["counts"].items() if k != "peel"},
                  "peel_counts": {str(k): v for k, v in s1["counts"]["peel"].items()},
                  "grids": {k: {"dim_xyz": list(g[0]), "pos_xyz": list(g[1])} for k, g in s1["grids"].items()},
                  "timings_s": s1["timings"]},
        "compartments": {"PRX_MASK_voxels": int(sitk_to_bool(prx_mask).sum()), "PRX_GOBJ_voxels": int(s4["G_prx"].sum()),
                         "CORT_MASK_voxels": int(cort_raw.sum()), "CORT_GOBJ_voxels": int(s4["G_cort"].sum()),
                         "TRAB_MASK_voxels": int(sitk_to_bool(trab_out).sum()), "TRAB_GOBJ_voxels": int(s4["G_trab"].sum()),
                         "LH_threshold_voxels": int(s4["lh"].sum()), "CORT_SEG_voxels": int(s4["cort_seg"].sum()),
                         "TRAB_SEG_voxels": int(s4["trab_seg"].sum()), "SEG_voxels": int((s4["seg"] > 0).sum())},
        "morphometry": m["metrics"],
        "bmd": bmd, "porosity": poro,
        "timing_s": {**timing, **{f"dt_{k}": v for k, v in m["timing_s"].items()}},
    }
    write_report(report, os.path.join(out_dir, f"{base}_report.json"), os.path.join(out_dir, f"{base}_report.csv"))
    mm = report["morphometry"]
    log(f"DONE {base}: BV/TV {mm['BV_TV']:.4f}  Tb.Th {mm['Tb_Th_mm']:.4f}  Tb.Sp {mm['Tb_Sp_mm']:.4f}  "
        f"Tb.N {mm['Tb_N_per_mm']:.4f}  Ct.Th {mm['Ct_Th_mm']:.4f}  [{timing['total']:.0f}s]")
    if isinstance(log, Logger):
        with open(os.path.join(out_dir, f"{base}_pipeline.log"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(log.lines) + "\n")
    return report


# ============================================================================================ run on IPL's masks
IPL_SUFFIXES = ("SEG", "TRAB_MASK", "CORT_MASK", "TRAB_SEG", "TRAB_TH", "TRAB_TH_old", "TRAB_SP", "TRAB_1N", "CORT_TH")


def discover_ipl_subject(subject_dir):
    """<base>_{SEG,TRAB_MASK,...}_decompressed.AIM in a subject folder; base from the SEG file."""
    segs = [p for p in glob.glob(os.path.join(subject_dir, "*_SEG_decompressed.AIM"))
            if "TRAB_" not in os.path.basename(p) and "CORT_" not in os.path.basename(p)]
    if not segs:
        raise FileNotFoundError(f"no *_SEG_decompressed.AIM in {subject_dir}")
    base = os.path.basename(segs[0]).replace("_SEG_decompressed.AIM", "")
    paths = {}
    for s in IPL_SUFFIXES:
        p = os.path.join(subject_dir, f"{base}_{s}_decompressed.AIM")
        if os.path.exists(p):
            paths[s] = p
    return base, paths


def _read_bool_aim(path):
    a = read_aim(path)
    return volume(a["data"] > 0, a["dim"], a["pos"]), a


def run_on_ipl_masks(subject_dir=None, seg_path=None, trab_mask_path=None, cort_mask_path=None, trab_seg_path=None,
                     reference=None, out_dir=None, map_format="nifti", backend="auto", voxel_size_mm=None, params=None,
                     include_old_tbth=True, write_maps=True, log=None):
    """The dt stage on IPL's own SEG / TRAB_MASK / CORT_MASK (IPL AIMs, aligned by global position), with
    voxel-for-voxel comparison against IPL's maps when they are present.

    subject_dir     a folder holding <base>_{SEG,TRAB_MASK,CORT_MASK,...}_decompressed.AIM (auto-discovered)
    seg_path ...    or explicit AIM paths; reference = {'TRAB_TH': path, 'TRAB_SP': ..., 'TRAB_1N': ..., 'CORT_TH': ...,
                    'TRAB_TH_old': ...} (defaults to the discovered files)
    Returns the report dict (metrics, per-map mismatch counts vs IPL, grids)."""
    log = log if log is not None else Logger()
    params = params or IPL_SCRIPT32
    paths = {}
    base = None
    if subject_dir:
        base, paths = discover_ipl_subject(subject_dir)
    for k, v in (("SEG", seg_path), ("TRAB_MASK", trab_mask_path), ("CORT_MASK", cort_mask_path), ("TRAB_SEG", trab_seg_path)):
        if v:
            paths[k] = v
    if reference:
        paths.update({k: v for k, v in reference.items() if v})
    if "SEG" not in paths or "TRAB_MASK" not in paths:
        raise ValueError("need at least the SEG and TRAB_MASK AIMs")
    if base is None:
        base = os.path.basename(paths["SEG"]).replace("_SEG_decompressed.AIM", "").replace(".AIM", "")
    log(f"IPL masks for {base}: " + ", ".join(sorted(paths)))
    t0 = time.time()

    segv, seg_aim = _read_bool_aim(paths["SEG"])
    dim, pos, el = seg_aim["dim"], seg_aim["pos"], seg_aim["el_size_mm"]
    vs = float(voxel_size_mm) if voxel_size_mm is not None else float(el[0])
    t = time.time()
    tm, tm_aim = _read_bool_aim(paths["TRAB_MASK"])
    G_trab = align_to(volume(render_volume(tm["data"]).astype(np.uint8), tm["dim"], tm["pos"]), dim, pos) > 0   # rendered on ITS grid
    log(f"  trab gobj rendered: {int(G_trab.sum()):,d} voxels on the SEG grid [{time.time() - t:.1f}s]")
    trab_seg = None
    if include_old_tbth and "TRAB_SEG" in paths:
        trab_seg = align_to(_read_bool_aim(paths["TRAB_SEG"])[0], dim, pos)
    cort_mask = cort_gobj = None
    cm = cm_aim = None
    if "CORT_MASK" in paths:
        t = time.time()
        cm, cm_aim = _read_bool_aim(paths["CORT_MASK"])
        cort_mask = cm["data"]
        cort_gobj = render_volume(cort_mask)                                                                     # own grid
        log(f"  cort gobj rendered: {int(cort_gobj.sum()):,d} voxels on the CORT_MASK grid [{time.time() - t:.1f}s]")
    which = ["TRAB_TH", "TRAB_SP", "TRAB_1N"] + (["CORT_TH"] if cort_mask is not None else []) + (["TRAB_TH_old"] if trab_seg is not None else [])
    m = ipl_morphometry(segv["data"], G_trab, cort_mask=cort_mask, cort_gobj=cort_gobj, voxel_size_mm=vs, params=params,
                        backend=backend, which=tuple(which), trab_seg=trab_seg, log=log)

    grids = {"TRAB_TH": segv, "TRAB_SP": segv, "TRAB_1N": segv, "TRAB_TH_old": segv, "CORT_TH": cm}
    headers = {"TRAB_TH": seg_aim, "TRAB_SP": seg_aim, "TRAB_1N": seg_aim, "TRAB_TH_old": seg_aim, "CORT_TH": cm_aim}
    comparison = {}
    for name, res in m["results"].items():
        g = grids[name]
        if name in paths:
            ref = align_to(read_aim(paths[name]), g["dim"], g["pos"]).astype(np.int16)
            ours = res.map.astype(np.int16)
            mism = ours != ref
            n = int(mism.sum())
            entry = {"reference": paths[name], "voxels": int(ref.size), "mismatches": n,
                     "ours_only": int((mism & (ref == 0)).sum()), "ipl_only": int((mism & (ours == 0)).sum()),
                     "ipl_support": int((ref > 0).sum()), "ours_support": int((ours > 0).sum()),
                     "ipl_mean_voxels": float(ref[ref > 0].mean()) if (ref > 0).any() else 0.0,
                     "ours_mean_voxels": float(ours[ours > 0].mean()) if (ours > 0).any() else 0.0,
                     "ipl_mean_mm": float(ref[ref > 0].mean() * vs) if (ref > 0).any() else 0.0}
            if n:
                d = (ours.astype(np.int32) - ref.astype(np.int32))[mism]
                u, c = np.unique(d, return_counts=True)
                entry["diff_histogram"] = {int(k): int(v) for k, v in zip(u, c)}
                zs = np.unique(np.nonzero(mism)[0])
                entry["slices_with_mismatch"] = [int(z) for z in zs[:50]]
            comparison[name] = entry
            log(f"  {name:<12s} vs IPL: {n:,d} mismatches of {ref.size:,d} (ours-only {entry['ours_only']}, IPL-only {entry['ipl_only']}); "
                f"IPL mean {entry['ipl_mean_voxels']:.4f} vox, ours {entry['ours_mean_voxels']:.4f}")
        if write_maps and out_dir:
            os.makedirs(out_dir, exist_ok=True)
            write_volume(os.path.join(out_dir, f"{base}_{name}_ipldt{_ext(map_format)}"), res.map.astype(np.int16), map_format,
                         el, g["pos"], headers[name]["header"])
    report = {"sample": base, "inputs": paths, "ipldt_version": __version__, "voxel_size_mm": vs, "el_size_mm": [float(e) for e in el],
              "grids": {"SEG": {"dim_xyz": list(dim), "pos_xyz": list(pos)},
                        **({"CORT_MASK": {"dim_xyz": list(cm["dim"]), "pos_xyz": list(cm["pos"])}} if cm else {})},
              "parameters": {**params.kwargs(), "backend": m["backend"]},
              "gobj_voxels": {"TRAB": int(G_trab.sum()), **({"CORT": int(cort_gobj.sum())} if cort_gobj is not None else {})},
              "morphometry": m["metrics"], "comparison_vs_ipl": comparison,
              "timing_s": {**m["timing_s"], "total": time.time() - t0}}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        write_report(report, os.path.join(out_dir, f"{base}_ipl_masks_report.json"), os.path.join(out_dir, f"{base}_ipl_masks_report.csv"))
        if isinstance(log, Logger):
            with open(os.path.join(out_dir, f"{base}_ipl_masks.log"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(log.lines) + "\n")
    return report


__all__ = ["DTParams", "IPL_SCRIPT32", "SITE_PARAMS", "Logger", "bbox_cut", "volume", "render_on_own_box", "sitk_to_bool",
           "array_to_sitk", "render_volume_sitk", "dt_thickness_sitk", "dt_spacing_sitk", "dt_number_sitk",
           "ipl_trabecular_microarchitecture_sitk", "ipl_cortical_thickness_sitk", "ipl_morphometry",
           "step1_load_aim", "step2_autocontour", "step1_params_for", "step1_calibration", "step3_trab_cort_seg",
           "LH_PAD_OFFSET", "LH_PAD_OFFSETS", "LH_DTYPE", "LH_DTYPES", "lh_pad_plan", "lh_filter_core",
           "laplace_hamming_threshold", "ipl_seg_assembly", "step4_render_and_segment", "step5_bmd", "cl_nr_extract",
           "write_report", "write_volume", "run_pipeline", "discover_ipl_subject", "run_on_ipl_masks"]
