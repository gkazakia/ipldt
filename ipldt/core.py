"""ipldt.core -- Scanco IPL /dt_thickness, /dt_spacing and /dt_number, reimplemented to agree with IPL voxel for voxel.

Every stage was derived from IPL's own outputs (help text, the manufacturer's evaluation scripts, logs and exported
AIMs only; no binary was disassembled) and validated voxel for voxel on 21 HR-pQCT
patellae and designed micro-phantoms (see the paper and its Supplement):

  field      Danielsson-type vector distance transform, slice-interleaved 'snake' sweep
             (ipldt.field.sir_quad).  Outside the image is object (no contacts).
  surface    s = |p(v)| with p(u) = sign(u) * max(|u| - 1/2, 0) per component, evaluated in
             float64 from the exact integer 4 s^2 (float32 misjudges near-ties of the ridge
             test in thick cortices: see surface_distance, 2026-09-14).
  ridge      x is a sphere centre unless a 26-neighbour y (inside the image) has
             |x - y| + s_x - s_y <= ridge_epsilon (float64, tolerance 1e-9; see ridge).
  diameter   version 1: V1 = floor(2 s_x + 1/2)
             version 2/3: pin a sphere between x's contact point P1 = p(v_x) and the
             contact point of the neighbour one step against the surface direction,
             y = x - rint(P1/|P1|), P2 = step + p(v_y);  D = |P1 - P2|,  M = (P1 + P2)/2,
             R = floor(D + 1/2) (round half up);
             version 2: R if y is object and cos(p(v_y), -P1) >= 1/2, else V1
             version 3: R if y is inside the image and |M| < 1, else V1
  drawing    every voxel with |c - x| <= D/2 + assign_epsilon takes the largest D.
  gobj       the trabecular/cortical contour is rendered from the mask raster exactly as
             IPL does (ipldt.contour); peel_iter erodes that raster slice-wise
             (4-connected) before the centre selection; the map is clipped to it.

Parameters follow IPL's names.  suppress_boundary is accepted for API symmetry but has
no effect: on the validation data IPL's outputs are identical for 0, 1, 2 and 3.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy import ndimage as ndi

from .field import sir_quad
from .contour import render_volume
from .gpu import resolve_backend

_NBR26 = [o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)]
_FEPS = 1e-9          # float guard for exact half-integers / unit boundaries
_RIDGE_TOL = 1e-9     # IPL's containment test is non-strict: exact ties prune.  Ties exist only at integer
                      # ridge_epsilon (1 + 2.5 - 3.5 = 0; sqrt8 + sqrt2 - sqrt18 = 0), and 1e-9 absorbs their
                      # float64 rounding (< 1e-12 for s < 4000) while staying 500x below the closest genuine
                      # non-tie of the shared-contact pair space |v_i| <= 80 (0.8999994949, s = 62).  The
                      # former 1e-6 (with float32 distances) is the 2026-09-14 OS_LH finding: see ridge().


# ---------------------------------------------------------------------------------- helpers
def surface_vector(v):
    """p(u) = sign(u) * max(|u| - 1/2, 0), the nearest point of the contact voxel's cube."""
    return np.sign(v) * np.maximum(np.abs(v) - 0.5, 0)


def surface_distance(V, xp=np):
    """s = |p(v)| for a (3, z, y, x) integer vector field, as float64; 0 on background.

    Precision rule (2026-09-14).  4 s^2 = sum over the non-zero components of (2|v_i| - 1)^2 is
    an integer (a sum of at most three odd squares), formed exactly in int64; s = sqrt(4 s^2) / 2
    is one correctly rounded float64 square root and an exact halving, so s carries 2^-53
    relative error (< 1e-13 absolute for s < 1000) and the same bits on the CPU and on the GPU
    (`xp` = numpy or cupy; IEEE sqrt on both).  float32 (spacing 7.6e-6 at s = 70) is NOT enough
    for the containment ridge (stage 3), whose value |x - y| + s_x - s_y has genuine near-ties at
    the 1e-6 level in thick cortices.  Evidence: on three OS_LH diaphyseal tibiae (Ct.Th of the
    cortical compartment, cortex about 140 voxels thick; Diaphyseal/REPRO 314619, 797620, 134330) one
    centre with v = (-5, 70, -7) (314619; the same squared distances recur on the other two),
    s = sqrt(19571)/2 = 69.9481951, and its +x neighbour with s = sqrt(19627)/2 = 70.0481977 has
    the exact ridge value 0.8999974, which IPL prunes at ridge_epsilon 0.9; float32 read it as
    0.9000015 and kept it, and the spurious diameter-140 sphere put 2,601 / 8 / 9 voxels of the
    three Ct.Th maps off IPL (0 / 0 / 0 in float64; the 21-patella cohort is unchanged, all five
    maps 0 mismatches on 21 / 21 in both precisions).  The former code, for the record:
    sqrt(sum(max(|v_i| - 1/2, 0)^2)) evaluated in float32."""
    a = xp.zeros(tuple(int(n) for n in V.shape[1:]), xp.int64)
    for i in range(3):
        u = xp.abs(xp.asarray(V[i]).astype(xp.int64))
        a += xp.where(u > 0, (2 * u - 1) ** 2, 0)
    return xp.sqrt(a.astype(xp.float64)) * 0.5


def ridge(obj, V, ridge_epsilon=0.9):
    """Sphere centres of the object: x survives unless a 26-neighbour y inside the image
    satisfies |x - y| + s_x - s_y <= ridge_epsilon (non-strict).  Not masked.

    Precision: the test is evaluated in float64 as  sep + s_x - s_y <= ridge_epsilon + _RIDGE_TOL
    with s from surface_distance (exact integer 4 s^2, one IEEE sqrt) and _RIDGE_TOL = 1e-9.  An
    exact tie sqrt(k) + (sqrt(A) - sqrt(B))/2 = ridge_epsilon (k = 1, 2, 3; A = 4 s_x^2, B = 4 s_y^2
    sums of one to three odd squares, so never 0 mod 4) is rational only when its square roots
    cancel, which forces an integer ridge_epsilon (0: e.g. 1 + 2.5 - 3.5, sqrt8 + sqrt2 - sqrt18,
    sqrt12 + sqrt3 - sqrt27; the B = 8 and B = 12 cancellations are impossible).  So at 0.9, 0.5
    and 0.25 the tolerance never decides and the only error is the float64 rounding (< 1e-13),
    while at 0 it absorbs the rounding of the true ties (< 1e-12 for s < 4000).  An exact integer
    decision (nested squaring of the three roots) would overflow int64 near s = 100 and is not
    needed.  The float32 evaluation used until 2026-09-14 (spacing 7.6e-6 at s = 70, tolerance
    1e-6) kept a centre with the exact value 0.8999974 on three OS_LH diaphyseal tibiae (2,601 / 8 /
    9 Ct.Th voxels off IPL); float64 reproduces IPL on all three."""
    s = surface_distance(V)
    s[~obj] = 0
    sp = np.pad(s, 1)
    op = np.pad(obj, 1, constant_values=False)          # padding = 'outside the image' for this test only
    keep = op & (sp > 0)
    for dz, dy, dx in _NBR26:
        sep = float(np.sqrt(dz * dz + dy * dy + dx * dx))
        keep &= ~(np.roll(op, (dz, dy, dx), axis=(0, 1, 2))
                  & ((sep + sp - np.roll(sp, (dz, dy, dx), axis=(0, 1, 2))) <= ridge_epsilon + _RIDGE_TOL))
    return keep[1:-1, 1:-1, 1:-1]


def diameters(obj, V, z, y, x, version=3):
    """IPL diameter at the voxels (z, y, x): version 1, 2 or 3 (see module docstring)."""
    N = np.array(obj.shape)
    v = np.stack([V[i][z, y, x].astype(np.float64) for i in range(3)], axis=1)
    P1 = surface_vector(v)
    s_x = np.linalg.norm(P1, axis=1)
    V1 = np.floor(2 * s_x + 0.5 + _FEPS).astype(np.int64)
    if version == 1:
        return V1
    n1 = s_x.copy()
    n1[n1 == 0] = 1
    step = -np.rint(P1 / n1[:, None])
    q = np.stack([z, y, x], axis=1) + step.astype(np.int64)
    inside = ((q >= 0) & (q < N)).all(axis=1)
    qc = tuple(np.clip(q[:, i], 0, N[i] - 1) for i in range(3))
    bg_y = ~obj[qc] | ~inside
    vy = np.stack([V[i][qc].astype(np.float64) for i in range(3)], axis=1)
    vy[bg_y] = 0
    Py = surface_vector(vy)
    P2 = step + Py
    D = np.linalg.norm(P1 - P2, axis=1)
    M = (P1 + P2) / 2
    R = np.floor(D + 0.5 + _FEPS).astype(np.int64)
    if version == 2:
        nPy = np.linalg.norm(Py, axis=1)
        cosg = np.where(nPy > 0, -(Py * P1).sum(1) / (np.maximum(nPy, 1e-12) * np.maximum(s_x, 1e-12)), -2.0)
        return np.where(inside & ~bg_y & (cosg >= 0.5 - _FEPS), R, V1)
    if version == 3:
        return np.where(inside & (np.linalg.norm(M, axis=1) < 1 - _FEPS), R, V1)
    raise ValueError(f"version must be 1, 2 or 3 (got {version})")


def draw_spheres(centres_z, centres_y, centres_x, diam, shape, assign_epsilon=0.5):
    """Max over covering spheres: voxel c is covered by centre x with diameter D when
    |c - x| <= D/2 + assign_epsilon (non-strict; evaluated exactly in squared integers)."""
    dm = np.zeros(shape, np.int16)
    dm[centres_z, centres_y, centres_x] = diam
    out = np.zeros(shape, np.int16)
    thr2 = lambda Dv: (Dv + 2.0 * assign_epsilon) ** 2 + 1e-6      # noqa: E731
    for Dv in range(1, int(diam.max()) + 1 if diam.size else 1):
        m = dm == Dv
        if not m.any():
            continue
        d2 = np.rint(ndi.distance_transform_edt(~m) ** 2)
        np.maximum(out, np.where(4 * d2 <= thr2(Dv), Dv, 0).astype(np.int16), out=out)
    return out


def peel_gobj(gobj, peel_iter):
    """IPL's gobj peel: erode the rendered contour raster slice-wise with 4-connectivity."""
    if peel_iter is None or peel_iter <= 0:
        return gobj
    s4 = np.zeros((3, 3, 3), bool)
    s4[1, 1, :] = True
    s4[1, :, 1] = True
    return ndi.binary_erosion(gobj, structure=s4, iterations=int(peel_iter), border_value=0)


def resolve_gobj(gobj, shape):
    """gobj may be None (whole image), a boolean raster already rendered, or a raw mask
    raster (rendered here with IPL's contour rules).  Pass a dict {'mask': raster} to force
    rendering, or {'rendered': raster} to use as is."""
    shape = tuple(int(s) for s in shape)
    if gobj is None:
        return np.ones(shape, bool)
    if isinstance(gobj, dict):
        G = np.asarray(gobj["rendered"]).astype(bool) if "rendered" in gobj else render_volume(np.asarray(gobj["mask"]).astype(bool))
    else:
        G = render_volume(np.asarray(gobj).astype(bool))
    if G.shape != shape:
        raise ValueError(f"gobj raster shape {G.shape} != image shape {shape}: volumes on different grids must be "
                         "pasted by global position first (ipldt.io.align_to)")
    return G


# ---------------------------------------------------------------------------------- report
@dataclass
class DTResult:
    """Output of dt_thickness / dt_spacing / dt_number."""
    map: np.ndarray                                   # int16 diameters in voxels, 0 = no sphere
    centres: np.ndarray                               # bool, the sphere centres used
    report: dict = dc_field(default_factory=dict)     # IPL-style statistics

    def __getitem__(self, k):
        return self.report[k]


def statistics(map_vox, obj, voxel_size_mm, name="Th"):
    """IPL's 'Get Statistics' block: mean / SD / max / skewness / kurtosis of the map over its
    non-zero voxels (in mm) and the fraction of object voxels that received a value."""
    vals = map_vox[map_vox > 0].astype(np.float64) * float(voxel_size_mm)
    n = vals.size
    rep = {f"{name}_mm": float(vals.mean()) if n else 0.0,
           f"{name}_sd_mm": float(vals.std(ddof=0)) if n else 0.0,
           f"{name}_max_mm": float(vals.max()) if n else 0.0,
           f"{name}_n_voxels": int(n),
           "valid_fraction": float(n / max(int(obj.sum()), 1)),
           "voxel_size_mm": float(voxel_size_mm)}
    if n > 2:
        d = vals - vals.mean()
        m2 = float((d ** 2).mean())
        rep[f"{name}_skew"] = float((d ** 3).mean() / m2 ** 1.5) if m2 > 0 else 0.0
        rep[f"{name}_kurtosis"] = float((d ** 4).mean() / m2 ** 2 - 3.0) if m2 > 0 else 0.0
    return rep


# ---------------------------------------------------------------------------------- engine
def _dt_core(obj, gobj_raster, ridge_epsilon, assign_epsilon, peel_iter, version, mask_centres=True, backend="auto"):
    be = resolve_backend(backend)
    obj = np.ascontiguousarray(obj, dtype=bool)
    V = sir_quad(obj)                                     # sequential sweep: CPU (numba)
    if be == "gpu":
        from .gpu import ridge_gpu
        cen = ridge_gpu(obj, V, ridge_epsilon, _RIDGE_TOL)
    else:
        cen = ridge(obj, V, ridge_epsilon)
    G = peel_gobj(gobj_raster, peel_iter)
    if mask_centres:
        cen = cen & G
    z, y, x = np.nonzero(cen)
    D = diameters(obj, V, z, y, x, version=version)
    if be == "gpu":
        from .gpu import draw_spheres_gpu
        out = draw_spheres_gpu(z, y, x, D, obj.shape, assign_epsilon)
    else:
        out = draw_spheres(z, y, x, D, obj.shape, assign_epsilon)
    out[~obj] = 0
    out[~gobj_raster] = 0
    return out, cen, V


def dt_thickness(obj, gobj=None, voxel_size_mm=1.0, ridge_epsilon=0.9, assign_epsilon=0.5,
                 peel_iter=-1, version=3, suppress_boundary=2, name="Th", backend="auto"):
    """IPL /dt_thickness.

    obj            boolean (z, y, x) object (e.g. the trabecular or cortical segmentation)
    gobj           the contour: a raw mask raster (rendered here), {'rendered': raster},
                   or None for the whole image
    voxel_size_mm  scales the reported statistics (the map itself is in voxels)
    ridge_epsilon, assign_epsilon, peel_iter, version : IPL's parameters
    suppress_boundary : accepted, no effect (IPL's outputs do not depend on it)
    backend        'auto' (CuPy GPU when available, else CPU), 'gpu' or 'cpu'; results are identical
    Returns DTResult(map, centres, report)."""
    G = resolve_gobj(gobj, obj.shape)
    out, cen, _ = _dt_core(obj, G, ridge_epsilon, assign_epsilon, peel_iter, version, backend=backend)
    return DTResult(out, cen, statistics(out, obj, voxel_size_mm, name))


def dt_spacing(seg, gobj=None, voxel_size_mm=1.0, ridge_epsilon=0.9, assign_epsilon=0.5,
               peel_iter=-1, version=3, suppress_boundary=2, backend="auto"):
    """IPL /dt_spacing: dt_thickness of the complement of `seg` (the whole bone segmentation),
    centres and map restricted to the gobj.  The gobj boundary is not a surface."""
    obj = ~np.ascontiguousarray(seg, dtype=bool)
    G = resolve_gobj(gobj, obj.shape)
    out, cen, _ = _dt_core(obj, G, ridge_epsilon, assign_epsilon, peel_iter, version, backend=backend)
    return DTResult(out, cen, statistics(out, obj & G, voxel_size_mm, "Sp"))


def dt_number(seg, gobj=None, voxel_size_mm=1.0, ridge_epsilon=0.9, assign_epsilon=0.5,
              peel_iter=-1, version=3, suppress_boundary=2, backend="auto"):
    """IPL /dt_number (1/Tb.N map): the containment ridge of `seg` (at ridge_epsilon) is the
    mid-axis; the result is dt_thickness of the complement of that mid-axis, restricted to
    the gobj.  Tb.N = 1 / mean(map) as IPL reports it ('MAT N (1/Th)')."""
    seg = np.ascontiguousarray(seg, dtype=bool)
    Vb = sir_quad(seg)
    mid_axis = ridge(seg, Vb, ridge_epsilon)
    obj = ~mid_axis
    G = resolve_gobj(gobj, obj.shape)
    out, cen, _ = _dt_core(obj, G, ridge_epsilon, assign_epsilon, peel_iter, version, backend=backend)
    rep = statistics(out, obj & G, voxel_size_mm, "inv_N")
    rep["Tb_N_per_mm"] = 1.0 / rep["inv_N_mm"] if rep["inv_N_mm"] > 0 else 0.0
    return DTResult(out, cen, rep)
