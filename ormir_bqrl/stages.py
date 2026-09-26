"""ormir_bqrl.stages -- the shared body of `run` and `redo`: the data contract between the stages and the
stage functions themselves.  Every numeric step is an ipldt.ormir public function called unchanged:

    load          step1_load_aim            (ITK ScancoImageIO + ipldt.io.read_aim)      IPL /read
    autocontour   step2_autocontour         (ORMIR-XCT)                                   the UCT autocontour / GOBJ0
    compartments  step3_trab_cort_seg       (ipldt.step1: Script 32 / 33 STEP 1)          36 IPL commands
    segment       render_on_own_box, laplace_hamming_threshold, ipl_seg_assembly         /fft_laplace_hamming ... /add_aims
    morphometry   bbox_cut, volume, ipl_morphometry, ipldt.io.align_to                    /dt_thickness /dt_spacing /dt_number
    bmd           step5_bmd                 (ORMIR-XCT bmd_masked)
    porosity      ipldt.porosity.pore_cascade + ct_po (run_pipeline's STEP 5c)            Script 32 STEP 2 pore cascade
    write_volumes write_volume (NIfTI via ipldt.io.write_nifti, or AIM via ipldt.io.write_aim with the input
                  AIM's header) + the Slicer files (ormir_bqrl.slicer)

`segment` is step4_render_and_segment with one addition: a gobj that is passed in is not rendered again (the
redo passes the rendering of an edited compartment as its own gobj).  The SEG assembly is masked with the
RENDERED periosteal contour (ALL = CORT_MASK | TRAB_MASK = IPL's /gobj seg GOBJ0 peel 0), which is the one
deliberate difference from ipldt.ormir.run_pipeline (it masks with the autocontour's raw raster); on rasters
whose rendering is the identity (IPL's own contours, the test phantom) the two agree exactly.

`porosity` is run_pipeline's STEP 5c unchanged: the rendered cortical contour (G_cort) and CORT_SEG, each as a
char volume 0 / 127 on the AIM grid, go to ipldt.porosity.pore_cascade; Ct.Po = ipldt.porosity.ct_po of its
pore map over that contour, i.e. |PORE & cortical contour| / |cortical contour|.  It runs after every
segmentation, so a run and every kind of redo recompute it from their own compartments.

Conventions: arrays are numpy (z, y, x); sizes / positions are (x, y, z) as ipldt.io uses them; masks are bool
on the AIM grid; SEG is uint8 127 / 126; maps are int16 diameters in voxels (IPL's AIM values) and become float32
mm only at write time (map_units="mm", NIfTI only).  SimpleITK images exist only at the boundary to ipldt.ormir's
step functions and are built with array_to_sitk(arr, grid.ref_image).
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from functools import cached_property

import numpy as np

from ipldt import ormir as engine
from ipldt.io import align_to, to_sitk, write_nifti
from ipldt.step1 import Step1Params

try:
    import SimpleITK as sitk
except ImportError:  # pragma: no cover - the workflow needs SimpleITK; the dataclasses do not
    sitk = None


# ============================================================================================ data contract
@dataclass(frozen=True)
class Grid:
    """The input AIM's lattice; every ORMIR-BQRL volume lives on it."""
    dim: tuple            # (x, y, z) voxels
    pos: tuple            # (x, y, z) global voxel position (AIM header 'pos')
    el: tuple             # element size mm per axis (AIM header VAX floats == ITK spacing)

    @property
    def origin_mm(self):
        """ITK / LPS origin = pos x el: what every file carries."""
        return tuple(float(p) * float(e) for p, e in zip(self.pos, self.el))

    @property
    def shape_zyx(self):
        return tuple(int(d) for d in self.dim[::-1])

    @cached_property
    def ref_image(self):
        """A uint8 zero image with this geometry (CopyInformation source for ipldt.ormir.array_to_sitk)."""
        return to_sitk(np.zeros(self.shape_zyx, np.uint8), self.el, self.pos)

    def volume(self, arr):
        """dict(data, dim, pos) for ipldt.io.align_to / ipldt.ormir.bbox_cut."""
        return engine.volume(arr, self.dim, self.pos)

    def as_dict(self):
        return {"dim_xyz": [int(d) for d in self.dim], "pos_xyz": [int(p) for p in self.pos],
                "el_size_mm": [float(e) for e in self.el]}

    def check(self, arr, what="mask"):
        """The array must be a (z, y, x) volume of this grid."""
        a = np.asarray(arr)
        if a.shape != self.shape_zyx:
            raise ValueError(f"the {what} {a.shape[::-1]} (x, y, z) is not on the AIM grid {tuple(self.dim)}")
        return a

    @classmethod
    def from_native(cls, native):
        return cls(tuple(int(d) for d in native["dim"]), tuple(int(p) for p in native["pos"]),
                   tuple(float(e) for e in native["el_size_mm"]))


@dataclass
class Loaded:
    """Stage 1: the AIM as ipldt.ormir.step1_load_aim returns it, plus the grid and the HU volume for Slicer."""
    aim_path: str
    base: str
    grid: Grid
    native: dict          # ipldt.io.read_aim: data (z, y, x) int16, dim / pos, el_size_mm, header, proclog
    img_hu: object        # SimpleITK float32 (ORMIR file_reader cast to float, what the autocontour / BMD use)
    hu_int16: object      # SimpleITK int16: file_reader's image before the cast (written as <base>_HU.nii.gz)
    calib: dict           # mu_scaling, mu_water, rescale_slope, rescale_intercept (ITK ScancoImageIO)


@dataclass
class Prov:
    """Where a mask came from (one per periosteal / cortical / trabecular; the report's provenance block)."""
    source: str                         # "ormir_xct.autocontour" | "file" | "array" | "manual" | "ipldt.step1" |
                                        #   "derived: periosteal - trabecular" | "derived: periosteal - cortical" | "run"
    path: str | None = None
    sha256: str | None = None
    voxels: int = 0                     # raw voxels of the raster as given / computed
    rendered_voxels: int | None = None  # voxels of its rendering (the gobj IPL sees)
    notes: list = field(default_factory=list)

    def as_dict(self):
        d = asdict(self)
        d["manual"] = self.source == "manual"
        return d


@dataclass
class Masks:
    """The contour rasters on the AIM grid.  ALL = CORT_MASK | TRAB_MASK is the rendered periosteal contour
    (stage 00); cort & trab is empty; G_cort / G_trab are the rendered compartment contours (set by segment);
    G_prx == ALL (written as PRX_GOBJ, never re-rendered); prx_raw is the raster the periosteal came from
    (autocontour / file) or None in a compartment-only redo."""
    ALL: np.ndarray
    cort: np.ndarray
    trab: np.ndarray
    prx_raw: np.ndarray | None = None
    G_cort: np.ndarray | None = None
    G_trab: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)

    @property
    def G_prx(self):
        return self.ALL

    def counts(self):
        n = lambda a: None if a is None else int(np.count_nonzero(a))     # noqa: E731
        return {"PRX_MASK_voxels": n(self.prx_raw), "PRX_GOBJ_voxels": n(self.ALL),
                "CORT_MASK_voxels": n(self.cort), "CORT_GOBJ_voxels": n(self.G_cort),
                "TRAB_MASK_voxels": n(self.trab), "TRAB_GOBJ_voxels": n(self.G_trab)}


@dataclass
class Segmentation:
    """Stage 4: the Laplace-Hamming threshold and IPL's SEG (127 cortical / 126 trabecular) on the AIM grid."""
    lh: np.ndarray
    cort_seg: np.ndarray
    trab_seg: np.ndarray
    seg: np.ndarray
    lh_el_size_mm: tuple
    lh_pad_offset: str = engine.LH_PAD_OFFSET     # /fft_laplace_hamming power-of-two padding offset (probe 21: 'ceil')

    def counts(self):
        return {"LH_threshold_voxels": int(self.lh.sum()), "CORT_SEG_voxels": int(self.cort_seg.sum()),
                "TRAB_SEG_voxels": int(self.trab_seg.sum()), "SEG_voxels": int((self.seg > 0).sum())}


@dataclass
class Morphometry:
    """Stage 5: IPL's four maps (int16 voxels, pasted back onto the AIM grid) and statistics."""
    maps: dict            # TRAB_TH, TRAB_SP, TRAB_1N, CORT_TH -> int16 (z, y, x) on the AIM grid
    metrics: dict         # ipl_morphometry's keys unchanged (Tb_Th_mm, ..., Tb_N_per_mm, BV_TV, Ct_Th_mm, ...)
    grids: dict           # {"SEG": Grid, "CORT_MASK": Grid}: the boxes IPL evaluates on
    timing_s: dict
    params: dict
    backend: str          # the resolved backend ("gpu" | "cpu")
    backend_requested: str = "auto"


@dataclass
class Porosity:
    """Stage 5c: the cortical pore map (bool on the AIM grid, None when not computed) and Ct.Po, with the keys of
    ipldt.ormir.run_pipeline's report['porosity'] block ({} when not computed)."""
    pore: np.ndarray | None
    metrics: dict
    computed: bool = False
    error: str | None = None


MAP_NAMES = ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH")
MASK_VALUE = 127
MAP_FORMATS = ("nifti", "aim")
AIM_MAX_VALUE = 255                   # ipldt.io.write_aim writes char AIMs (IPL's own map AIMs are char too)


# ============================================================================================ helpers
def _log(log):
    return log if log is not None else engine.Logger(echo=False)


def _need_sitk():
    if sitk is None:
        raise ImportError("SimpleITK is required by the ORMIR-BQRL workflow (pip install \"ipldt[bqrl] @ git+https://github.com/zhuyihua1234/ipldt\")")


def as_bool(arr, grid, what="mask"):
    """A bool (z, y, x) array on the grid from a bool / integer array or a SimpleITK image."""
    if sitk is not None and isinstance(arr, sitk.Image):
        arr = engine.sitk_to_bool(arr)
    return grid.check(np.asarray(arr) != 0, what)


def mask_image(mask, grid):
    """bool (z, y, x) -> SimpleITK uint8 0/1 image with the grid's geometry."""
    _need_sitk()
    return engine.array_to_sitk(np.asarray(mask, bool).astype(np.uint8), grid.ref_image)


def render(mask, grid):
    """IPL's /togobj_from_aim -curvature_smooth 1 followed by /gobj_to_aim of a raster: what IPL sees of any
    contour (ipldt.ormir.render_on_own_box, on the raster's bounding box)."""
    return engine.render_on_own_box(np.asarray(mask, bool), grid.pos)


def check_output_format(map_format, map_units, native=None):
    """The output options run_pipeline accepts: map_format 'nifti' | 'aim', map_units 'voxels' | 'mm', and AIM
    maps only as integer diameters in voxels.  With `native` (ipldt.io.read_aim's dict of the input AIM) an AIM
    output also needs a v020 header to use as the template (ipldt.io.write_aim)."""
    if map_format not in MAP_FORMATS:
        raise ValueError(f"map_format must be 'nifti' or 'aim', not {map_format!r}")
    if map_units not in ("voxels", "mm"):
        raise ValueError("map_units must be 'voxels' or 'mm'")
    if map_format == "aim" and map_units == "mm":
        raise ValueError("AIM maps are integer diameters in voxels; use map_units='voxels' or map_format='nifti'")
    if map_format == "aim" and native is not None and native.get("version") == "030":
        raise ValueError("AIM output needs a v020 input AIM (its header is the template of every AIM written); "
                         "this input is AIMDATA_V030: use map_format='nifti'")


# ============================================================================================ stages
def load(aim_path, log=None):
    """(1) the AIM: HU image (ORMIR file_reader), native int16 (ipldt.io.read_aim), calibration; the Grid."""
    _need_sitk()
    log = _log(log)
    aim_path = os.path.abspath(aim_path)
    img_hu, native, calib = engine.step1_load_aim(aim_path, log)
    grid = Grid.from_native(native)
    hu_int16 = sitk.Cast(img_hu, sitk.sitkInt16)          # file_reader's int16 values, exactly (the cast was float32)
    base = os.path.splitext(os.path.basename(aim_path))[0]
    return Loaded(aim_path=aim_path, base=base, grid=grid, native=native, img_hu=img_hu, hu_int16=hu_int16, calib=calib)


def autocontour(loaded, log=None):
    """(2) ORMIR-XCT's autocontour -> the periosteal raster (bool on the grid)."""
    prx = engine.step2_autocontour(loaded.img_hu, loaded.calib, _log(log))
    return as_bool(prx, loaded.grid, "periosteal mask")


def compartments(loaded, prx_raw, params, log=None):
    """(3) Script 32 / 33 STEP 1 from the periosteal raster (rendered on its box inside step 3 = stage 00).
    Returns (cort, trab, ALL, info): bool rasters on the grid, ALL = cort | trab = the rendered periosteal."""
    log = _log(log)
    grid = loaded.grid
    prx_raw = as_bool(prx_raw, grid, "periosteal mask")
    if not prx_raw.any():
        raise ValueError("the periosteal mask is empty: nothing to separate")
    if not isinstance(params, Step1Params):
        raise TypeError(f"params must be an ipldt.step1.Step1Params, not {type(params).__name__}")
    cort_img, trab_img, info = engine.step3_trab_cort_seg(loaded.native, mask_image(prx_raw, grid), params, loaded.calib, log)
    cort = engine.sitk_to_bool(cort_img)
    trab = engine.sitk_to_bool(trab_img)
    ALL = cort | trab
    if (cort & trab).any():
        raise RuntimeError("step 3 returned overlapping cortical and trabecular masks")
    if int(ALL.sum()) != int(info["periosteal"]["rendered_voxels"]):
        raise RuntimeError(f"CORT_MASK | TRAB_MASK ({int(ALL.sum()):,d} voxels) is not the rendered periosteal "
                           f"contour ({info['periosteal']['rendered_voxels']:,d})")
    return cort, trab, ALL, info


def segment(loaded, ALL, cort, trab, G_cort=None, G_trab=None, log=None):
    """(4) render the compartment contours that are not given (IPL's togobj_from_aim on each mask's own box),
    Laplace-Hamming threshold of the native int16 volume with the header's per-axis element sizes (IPL's rule),
    IPL's SEG assembly masked with the rendered periosteal ALL.  Returns (Segmentation, G_cort, G_trab)."""
    log = _log(log)
    grid = loaded.grid
    ALL = as_bool(ALL, grid, "periosteal contour")
    cort = as_bool(cort, grid, "cortical mask")
    trab = as_bool(trab, grid, "trabecular mask")
    log("STEP 4 - rendering contours (IPL togobj_from_aim -curvature_smooth 1 + gobj rasterisation) ...")
    t = time.time()
    if G_cort is None:
        G_cort = render(cort, grid)
        log(f"  cort: raw raster {int(cort.sum()):,d} -> rendered gobj {int(G_cort.sum()):,d} voxels")
    else:
        G_cort = as_bool(G_cort, grid, "cortical gobj")
        log(f"  cort: gobj given ({int(G_cort.sum()):,d} voxels), not re-rendered")
    if G_trab is None:
        G_trab = render(trab, grid)
        log(f"  trab: raw raster {int(trab.sum()):,d} -> rendered gobj {int(G_trab.sum()):,d} voxels")
    else:
        G_trab = as_bool(G_trab, grid, "trabecular gobj")
        log(f"  trab: gobj given ({int(G_trab.sum()):,d} voxels), not re-rendered")
    log(f"  periosteal: rendered contour ALL {int(ALL.sum()):,d} voxels (= CORT_MASK | TRAB_MASK); rendering took {time.time() - t:.1f}s")
    log("STEP 4 - Laplace-Hamming binarization (native int16) + IPL SEG assembly ...")
    t = time.time()
    lh_el = tuple(float(e) for e in loaded.native["el_size_mm"])
    log(f"  Laplace-Hamming element sizes (x, y, z) = {lh_el[0]:.7f} / {lh_el[1]:.7f} / {lh_el[2]:.7f} mm (AIM header, per axis, IPL's rule); "
        f"power-of-two padding offset '{engine.LH_PAD_OFFSET}' (probe 21)")
    lh = engine.laplace_hamming_threshold(loaded.native["data"], lh_el, pad_offset=engine.LH_PAD_OFFSET)
    cort_seg, trab_seg = engine.ipl_seg_assembly(lh, ALL, G_cort, G_trab)
    seg = np.zeros(lh.shape, np.uint8)
    seg[cort_seg] = engine.SEG_VALUE_CORT
    seg[trab_seg] = engine.SEG_VALUE_TRAB
    log(f"  LH threshold {int(lh.sum()):,d}; CORT_SEG {int(cort_seg.sum()):,d}, TRAB_SEG {int(trab_seg.sum()):,d}, "
        f"SEG {int((seg > 0).sum()):,d} voxels [{time.time() - t:.1f}s]")
    return Segmentation(lh=lh, cort_seg=cort_seg, trab_seg=trab_seg, seg=seg, lh_el_size_mm=lh_el,
                        lh_pad_offset=engine.LH_PAD_OFFSET), G_cort, G_trab


def morphometry(loaded, seg, G_trab, cort, G_cort, dt_params=None, backend="auto", log=None):
    """(5) Script 32's dt stage on IPL's grids: Tb.Th / Tb.Sp / Tb.N on /bounding_box_cut -border 0 of SEG with
    the trabecular gobj, Ct.Th of the cortical compartment raster on its own box with the cortical gobj; the maps
    are pasted back onto the AIM grid by global position (ipldt.io.align_to)."""
    from ipldt.gpu import resolve_backend
    log = _log(log)
    grid = loaded.grid
    dim, pos, el = grid.dim, grid.pos, grid.el
    params = dt_params or engine.IPL_SCRIPT32
    seg = grid.check(seg, "SEG")
    cort = as_bool(cort, grid, "cortical mask")
    G_trab = as_bool(G_trab, grid, "trabecular gobj")
    G_cort = as_bool(G_cort, grid, "cortical gobj")
    log("STEP 5 - IPL dt_thickness / dt_spacing / dt_number ...")
    t = time.time()
    seg_box = engine.bbox_cut(seg, pos)                                                    # SEG.AIM's grid
    G_trab_box = align_to(grid.volume(G_trab.astype(np.uint8)), seg_box["dim"], seg_box["pos"]) > 0
    cort_box = engine.bbox_cut(cort, pos)                                                  # CORT_MASK.AIM's grid
    G_cort_box = align_to(grid.volume(G_cort.astype(np.uint8)), cort_box["dim"], cort_box["pos"]) > 0
    log(f"  SEG grid dim {seg_box['dim']} pos {seg_box['pos']}; CORT_MASK grid dim {cort_box['dim']} pos {cort_box['pos']}")
    m = engine.ipl_morphometry(seg_box["data"] > 0, G_trab_box, cort_mask=cort_box["data"], cort_gobj=G_cort_box,
                               voxel_size_mm=float(el[0]), params=params, backend=backend, log=log)
    boxes = {"TRAB_TH": seg_box, "TRAB_SP": seg_box, "TRAB_1N": seg_box, "CORT_TH": cort_box}
    maps = {}
    for name, res in m["results"].items():
        g = boxes[name]
        maps[name] = align_to(engine.volume(res.map.astype(np.int16), g["dim"], g["pos"]), dim, pos)
    timing = dict(m["timing_s"])
    timing["total"] = time.time() - t
    grids = {"SEG": Grid(tuple(seg_box["dim"]), tuple(seg_box["pos"]), el),
             "CORT_MASK": Grid(tuple(cort_box["dim"]), tuple(cort_box["pos"]), el)}
    return Morphometry(maps=maps, metrics=dict(m["metrics"]), grids=grids, timing_s=timing, params=dict(m["params"]),
                       backend=resolve_backend(backend), backend_requested=str(backend))


def bmd(loaded, G_trab, G_cort, log=None):
    """(5b) ORMIR-XCT bmd_masked inside the rendered contours; a failure is logged and returns {} (BMD is a
    courtesy: it never fails the morphometry, as in ipldt.ormir.run_pipeline)."""
    log = _log(log)
    log("STEP 5b - BMD (ORMIR bmd_masked) ...")
    try:
        return engine.step5_bmd(loaded.img_hu, np.asarray(G_trab, bool), np.asarray(G_cort, bool), loaded.calib, log)
    except Exception as exc:
        log(f"  BMD skipped: {type(exc).__name__}: {exc}")
        return {}


def porosity(loaded, G_cort, cort_seg, log=None):
    """(5c) the cortical pore cascade and Ct.Po, exactly ipldt.ormir.run_pipeline's STEP 5c: the rendered cortical
    contour and CORT_SEG as char volumes (0 / 127) on the AIM grid -> ipldt.porosity.pore_cascade -> the pore map;
    ipldt.porosity.ct_po(pore, contour) = |PORE & contour| / |contour|.  As in run_pipeline a failure is logged
    and never fails the morphometry (Porosity.computed False, the message in Porosity.error)."""
    from ipldt import porosity as _por
    log = _log(log)
    grid = loaded.grid
    log("STEP 5c - cortical pore cascade (Burghardt) + Ct.Po ...")
    try:
        _cr = engine.volume(np.asarray(G_cort, bool).astype(np.uint8) * 127, grid.dim, grid.pos)
        _cs = engine.volume(np.asarray(cort_seg, bool).astype(np.uint8) * 127, grid.dim, grid.pos)
        _pore = _por.pore_cascade(_cr, _cs)["pore"]
        _po = _por.ct_po(_pore, _cr)
        metrics = {"Ct_Po": float(_po["ct_po"]), "Ct_Po_pore_voxels": int(_po["pore_voxels"]),
                   "Ct_Po_compartment_voxels": int(_po["mask_voxels"])}
        log(f"  Ct.Po = {_po['ct_po']:.5f}  ({_po['pore_voxels']:,d} pore voxels of "
            f"{_po['mask_voxels']:,d} in the cortical compartment)")
        return Porosity(pore=align_to(_pore, grid.dim, grid.pos) != 0, metrics=metrics, computed=True)
    except Exception as exc:     # porosity is a courtesy: never fail the morphometry for it (run_pipeline's rule)
        log(f"  porosity skipped: {type(exc).__name__}: {exc}")
        return Porosity(pore=None, metrics={}, computed=False, error=f"{type(exc).__name__}: {exc}")


# ============================================================================================ output
def _ext(map_format):
    return ".AIM" if map_format == "aim" else ".nii.gz"


def write_volumes(out_dir, loaded, masks, segmentation, morph, map_units="voxels", log=None, pore=None, map_format="nifti"):
    """The output set of section 7.1: every mask, SEG, the pore map and the dt maps on the AIM grid, as NIfTI
    (map_format 'nifti': origin = pos x el) or as char AIMs on the input AIM's header (map_format 'aim':
    ipldt.ormir.write_volume -> ipldt.io.write_aim, as run_pipeline writes them; maps as integer diameters in
    voxels, which must fit the char range 0..255 -- checked before anything is written); <base>_HU is always
    NIfTI (the greyscale for Slicer; the input AIM is the greyscale AIM); the two Slicer files
    (ormir_bqrl.slicer).  `pore` is the bool pore map of stage 5c or None.  Returns {name: absolute path}."""
    from . import slicer
    check_output_format(map_format, map_units, loaded.native)
    log = _log(log)
    grid, base = loaded.grid, loaded.base
    header = loaded.native.get("header")
    if map_format == "aim":
        for name in MAP_NAMES:
            data = morph.maps.get(name)
            if data is None or data.size == 0:
                continue
            lo, hi = int(np.min(data)), int(np.max(data))
            if lo < 0 or hi > AIM_MAX_VALUE:
                raise ValueError(f"the {name} map spans {lo}..{hi} voxels, outside the char AIM range 0..{AIM_MAX_VALUE}; "
                                 "write it with map_format='nifti'")
    os.makedirs(out_dir, exist_ok=True)
    out = {}

    def save(name, arr):
        path = os.path.abspath(os.path.join(out_dir, f"{base}_{name}{_ext(map_format)}"))
        engine.write_volume(path, arr, map_format, grid.el, grid.pos, header)
        out[name] = path

    t = time.time()
    p = os.path.abspath(os.path.join(out_dir, f"{base}_HU.nii.gz"))
    write_nifti(p, sitk.GetArrayFromImage(loaded.hu_int16).astype(np.int16), grid.el, grid.pos)
    out["HU"] = p
    if masks.prx_raw is not None:
        save("PRX_MASK", masks.prx_raw.astype(np.uint8) * MASK_VALUE)
    save("PRX_GOBJ", masks.ALL.astype(np.uint8) * MASK_VALUE)
    save("CORT_MASK", masks.cort.astype(np.uint8) * MASK_VALUE)
    save("TRAB_MASK", masks.trab.astype(np.uint8) * MASK_VALUE)
    save("CORT_GOBJ", masks.G_cort.astype(np.uint8) * MASK_VALUE)
    save("TRAB_GOBJ", masks.G_trab.astype(np.uint8) * MASK_VALUE)
    save("SEG", segmentation.seg)
    save("CORT_SEG", segmentation.cort_seg.astype(np.uint8) * MASK_VALUE)
    save("TRAB_SEG", segmentation.trab_seg.astype(np.uint8) * MASK_VALUE)
    if pore is not None:
        save("PORE", np.asarray(pore, bool).astype(np.uint8) * MASK_VALUE)
    for name in MAP_NAMES:
        if name not in morph.maps:
            continue
        data = morph.maps[name]
        if map_units == "mm":
            data = data.astype(np.float32) * np.float32(grid.el[0])
        save(name, data)
    labels = slicer.labels_from_masks(masks.cort, masks.trab)
    p = os.path.abspath(os.path.join(out_dir, f"{base}_compartments.seg.nrrd"))
    slicer.write_seg_nrrd(p, labels, grid)
    out["seg_nrrd"] = p
    p = os.path.abspath(os.path.join(out_dir, f"{base}_compartments_labelmap.nii.gz"))
    slicer.write_labelmap_nifti(p, labels, grid)
    out["labelmap"] = p
    log(f"  wrote {len(out)} volumes to {os.path.abspath(out_dir)} [{time.time() - t:.1f}s]")
    return out


__all__ = ["Grid", "Loaded", "Prov", "Masks", "Segmentation", "Morphometry", "Porosity", "MAP_NAMES", "MASK_VALUE",
           "MAP_FORMATS", "AIM_MAX_VALUE", "as_bool", "mask_image", "render", "check_output_format", "load", "autocontour",
           "compartments", "segment", "morphometry", "bmd", "porosity", "write_volumes"]
