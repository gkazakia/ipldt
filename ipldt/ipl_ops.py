"""ipldt.ipl_ops -- clean-room reimplementations of the IPL V5.42 commands that Scanco Script 32 / 33 STEP 1 (the
cortical / trabecular separation) applies to char masks: /seg_gauss, the metric-11 morphology (/erosion,
/dilation, /close, /open), the connected-component commands (/cl_ow_rank_extract, /cl_nr_extract,
/cl_slicewise_extractow), /gobj_maskaimpeel_ow and the bookkeeping commands (/subtract_aims, /add_aims,
/set_value, /bounding_box_cut).

Every rule below was derived from IPL's help text, the manufacturer's evaluation scripts, IPL's logs and IPL's exported
intermediate AIMs only (no binary was disassembled) and is verified voxel for voxel against two oracles:
(a) the 29 exported stages of the P16 replay of Script 32 on PFJ-0be66a_R (scan X2420448, log P16_PFJ-0be66a_R.TXT):
0 mismatching voxels and identical grids for every command applied to IPL's own input stage; (b) probe 18
(designed phantoms, scanner run 2026-09-13): two phantoms on which the competing readings of the conventions
the P16 oracle does not exercise (43 questions) give DIFFERENT outputs, run through the same commands on the
scanner (28 exports X2420448_P18_<TAG>.AIM, logs P18_CL.TXT / P18_SG.TXT, every prediction written before
the run; probe-18 verifier, not distributed: 'every export matched a
predicted reading').  No parameter is fitted: each rule is a discrete hypothesis that reproduces IPL's
export exactly or is refuted (mismatch counts in the docstrings).  The /seg_gauss float32 formulation is
identified on ONE export only (stage 01 of PFJ-0be66a_R, sigma 2 / support 3) by enumerating 978 candidate
kernel / normaliser / accumulation formulations, 20 of which that export cannot separate (fused vs separate
multiply-add, the normaliser's summation, the last ulp of the outermost tap); its last-ulp neighbours move
2-8 stage-01 voxels and 0 CORT_MASK / TRAB_MASK voxels on PFJ-0be66a_R and PFJ-69bcb0_R.

PROBE 18 OUTCOME (verified on designed phantoms, probe 18, 2026-09-13).  CONFIRMED: the slicewise
denominator (slice total; exports SW), the inclusive 50 % tie and 4-connectivity (SW), the inclusive
cl_nr_extract bounds (NRMIN, NRMAX), add_aims saturation 127 + 127 = 127 (ADD), every chamfer threshold
3N + 2 and the open erosion boundary on the x0 / x1 / y0 / y1 / z faces (ERO3, ERO1, OPEN3), the dilation
ball raw < 3N + 2 and the written-grid growth N + 1 for N = 1 and 15 (DIL1, DIL15), close / open (CLOSE3,
OPEN3), the seg_gauss bounds inclusive at 4524 and 17173 (SG).  REFUTED, and corrected here: (1)
/subtract_aims stores a negative difference as -127 (byte 0x81), not 0 (SUB, SUB2); (2) -connect_boundary
true joins EVERY face-touching component into one component before ranking (CBT1, CBT2); (3) the rank
tie-break is not 'lower label first' (RANK1 / RANK2 put the later of two equal components first, RANK4 /
RANK5 the earlier one): an unstable sort of the size table, see cl_ow_rank_extract.

PROBE 17 OUTCOME (Script 33 STEP 1 replayed with every intermediate exported, scanner run fetched 2026-09-14;
probe-17 exports and verifier, not distributed).  The
RADIUS preset (corner_min 800, close2 30) on PFJ-0be66a_R (X2420448), PFJ-42293d_L (X3623103) and PFJ-6f5538_R (X5492058):
every stage 01..29 reproduced with 0 mismatching voxels and identical grids, isolated and cumulative, and the
renderings 30 / 31 with 0 voxels -- the corner branch is non-empty on all three (18_corncl 3,500 / 42,625 /
31,750 voxels, 19_cornmajor 12,879 / 139,031 / 91,420), /close 30 gets its oracle (23_close30 20,302,109 /
21,606,958 / 1,954,987 voxels, 0 mismatches), and the slicewise 4-connectivity is discriminated on real data
(PFJ-42293d_L 25_slicewise: 8-connectivity would differ by 507 voxels, PFJ-6f5538_R by 1).  The TIBIA block on
PFJ-411dfd_R (X5143651, corner_min 200000 / close2 50): stages 01..14 exact, 15_open15 <- 14_bbc leaves 3 IPL-ONLY
voxels (ours 17,696,230, IPL 17,696,233, same grid), every later stage 16..29 exact in isolation, so the 3
voxels propagate cumulatively to 28 / 29 and ARE the known 3-voxel difference to PFJ-411dfd_R's September masks;
30 / 31 and the 336 + 168 stored chains exact.  That residual was explained on 2026-09-14 evening (probe 19 and
the S3 search: /open is ONE pipeline on the erosion's mirror-padded buffer, see open_) and the mechanism is
implemented in open_ since that evening: 18 of 18 erosion / dilation / close / open oracles exact, PFJ-411dfd_R
15_open15 included (17,696,233 = 17,696,233), and the
probe-19 exports OPEN15 / SUBA_OPEN15 / SUBB_OPEN15 / SUBC_OPEN15 at 0.
PROBE 20 OUTCOME (the prospective confirmation of the /open mechanism; designed 2026-09-14 18:12, every prediction
written at 18:12:42 BEFORE any scanner run, run fetched 2026-09-15; probe-20 verifier log, not distributed): 'MECHANISM CONFIRMED: IPL matched the mechanism's own prediction on all 46 export(s)
with readings (the edge-inclusive mirror where the command has no -continuous_at_boundary flag or a flag of 1, the
empty border where the flag is 0); 12 export(s) separate it from every alternative ...; 11 contrast hypothes(es)
refuted, 0 never separated'.  CONFIRMED: the six-face edge-inclusive mirror margin of /open at every N of the scan
11 / 12 / 13 / 14 / 16 / 17 / 18 (0 mismatching voxels; the pre-mechanism rule 86 / 62 / 34 / 51 / 0 / 0 / 0
IPL-only voxels, exactly as predicted), its depth N + 2 (the cave T-scan restores 0 / 0 / 117 / 117 disc voxels),
its isotropy (the transposed phantoms), numpy 'symmetric' as the fill of a margin deeper than the volume (z156, 12
slices with margin 17), and -continuous_at_boundary obeyed PER AXIS in x y z order by /close and /dilation (0 =
empty border, 1 = mirror; 'the flag is ignored and IPL always mirrors' refuted by 38,345 voxels).  Implemented here:
open_ unchanged, and the flag added to dilation() / close() as continuous_at_boundary, default (0, 0, 0) = the form
Scripts 32 / 33 use.

Volume convention (ipldt.io): a volume is dict(data=(z, y, x) array, dim=(x, y, z), pos=(x, y, z)).  Set
voxels are non-zero (a -127 of /subtract_aims IS set); mask outputs are written as char 0 / 127 like IPL's,
the arithmetic outputs as int8 in [-127, 127].  Volumes on different grids are ALWAYS combined by global
position (union_grid / on_grid, built on ipldt.io.align_to), never by shape.

THE RULES (details and the refuted alternatives in each docstring)
  /seg_gauss      separable Gaussian in float32, intermediate stored as short (truncated) after EVERY 1-D
                  pass (x, then y, then z), float32 taps normalised by the float32-rounded exact sum, 'valid'
                  output grid (shrunk by `support` per side), inclusive integer threshold lower <= v <= upper
                  with the native numbers rint((mgHA - intercept) / slope * Mu_Scaling).
  metric 11       two-pass chamfer 3-4-5 distance map; keep <=> raw >= 3N + 2.  Erosion: IPL's D3P_BorderChange
                  pads a margin of N + 2 on all six faces with the EDGE-INCLUSIVE MIRROR image of the input,
                  which inside the volume equals an OPEN boundary (nothing outside the array is background: the
                  mirror is an isometry of the chamfer metric), grid unchanged.  Dilation pads a margin of N + 2 on
                  all six faces -- BACKGROUND per axis where -continuous_at_boundary is 0 (the default and the only
                  form Scripts 32 / 33 use), the edge-inclusive mirror where it is 1 -- out = in | (dt(~in) <
                  3N + 2), written grid = input grid grown by N + 1 per side.  close = that dilation then that
                  erosion on the same buffer, cropped to the input grid.  open = ONE pipeline on the mirror-padded
                  buffer: the erosion on the whole buffer, then the dilation seeded by EVERY survivor, margin
                  included (the margin is not re-filled), cropped to the input grid -- the mechanism of the PFJ-411dfd_R
                  /open 15 residual (found and implemented 2026-09-14 evening, CONFIRMED by probe 20 on 2026-09-15,
                  see open_).  18 of the 18 erosion / dilation / close / open oracles are exact.
  components      6-connected 3-D labelling (topology 6) in raster label order; connect_boundary true joins
                  every component touching any of the six faces into one component (label 1); ranks = an
                  unstable (shell) sort of the size table, largest first; slicewise = per slice, 4-connected,
                  keep 100 * size / (set voxels of that slice) in [lo, up] inclusive.
  gobj_maskaimpeel_ow   AND with the rendered gobj eroded slice-wise 4-connected peel_iter times, pasted by
                  global position (slices without contour are cleared).
  subtract / add  char arithmetic on the union bounding box of the two grids (pos = min, end = max):
                  out = in1 -/+ in2 saturated to [-127, 127], stored as int8 (0 - 127 = -127, 127 + 127 = 127).
  bounding_box_cut border 0: the tight box of the non-zero voxels.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
from numba import njit
from scipy import ndimage as ndi

from .core import peel_gobj
from .io import align_to

SET = 127                                   # IPL's value_in_range for char masks
_S6 = ndi.generate_binary_structure(3, 1)   # topology 6 in 3-D (face neighbours)
_S4 = ndi.generate_binary_structure(2, 1)   # topology 6 restricted to one slice = 4-connectivity


# ============================================================================================== volumes / grids
def vol(data, dim, pos):
    """A volume dict: data (z, y, x), dim (x, y, z), pos (x, y, z)."""
    data = np.asarray(data)
    dim = tuple(int(v) for v in dim)
    pos = tuple(int(v) for v in pos)
    if data.shape != dim[::-1]:
        raise ValueError(f"data shape {data.shape} does not match dim {dim} (expected {dim[::-1]})")
    return dict(data=data, dim=dim, pos=pos)


def mask_vol(data_bool, dim, pos, value=SET):
    """A char 0 / value volume from a boolean (z, y, x) array."""
    return vol(np.where(np.asarray(data_bool, dtype=bool), np.uint8(value), np.uint8(0)), dim, pos)


def union_grid(a, b):
    """The union bounding box of two grids: pos = component-wise min of the two pos, end = max of pos + dim.
    This is exactly the 'beg_pos / end_pos' IPL prints for /subtract_aims and /add_aims (D3P_Concatenate)."""
    lo = tuple(min(a["pos"][i], b["pos"][i]) for i in range(3))
    hi = tuple(max(a["pos"][i] + a["dim"][i], b["pos"][i] + b["dim"][i]) for i in range(3))
    return tuple(hi[i] - lo[i] for i in range(3)), lo


def on_grid(v, dim, pos, fill=0):
    """v['data'] pasted onto the (dim, pos) grid by GLOBAL position (never by shape); zero-filled outside."""
    dim = tuple(int(x) for x in dim)
    pos = tuple(int(x) for x in pos)
    if tuple(v["dim"]) == dim and tuple(v["pos"]) == pos:
        return v["data"]
    return align_to(dict(data=v["data"], dim=tuple(v["dim"]), pos=tuple(v["pos"])), dim, pos, fill=fill)


def _set(v):
    """The set voxels of a volume as a bool (z, y, x) array."""
    return np.asarray(v["data"]) != 0


def _u8(mask_bool):
    return np.ascontiguousarray(mask_bool, dtype=np.uint8)


# ============================================================================================== /seg_gauss
def ipl_gauss_weights(sigma=2.0, support=3):
    """The float32 taps of IPL's /seg_gauss kernel, w[-support .. support].

    RULE: taps g_k = float32(exp(-k^2 / (2 sigma^2))) (exp in double, rounded to float32); normaliser
    S = float32(sum_k g_k) with the sum taken exactly and rounded ONCE to float32; w_k = float32(g_k / S).
    For sigma 2, support 3: w0..w3 = 0x3e5d4ae1 0x3e434a39 0x3e06387f 0x3d8fafb2 = 0.216105953 0.190712824
    0.131074890 0.070159331 (sum 1.0000000447, one ulp above float32(exactly normalised) in w0, w2, w3).
    REFUTED on the PFJ-0be66a oracle (mismatches of stage 01): float32(exactly normalised weights) 2; running
    float32-sum normaliser (4.6273603 instead of 4.6273599) 4; float64 kernel and arithmetic 4,540.
    VERIFIED for sigma 2 / support 3 only (the single /seg_gauss oracle X2420448_P16_01_SEGGAUSS of PFJ-0be66a_R);
    other sigma / support values apply the same construction without an oracle."""
    k = np.arange(-support, support + 1, dtype=np.float64)
    g = np.exp(-k * k / (2.0 * float(sigma) * float(sigma))).astype(np.float32)
    s = np.float32(g.astype(np.float64).sum())
    return (g / s).astype(np.float32)


def mgha_to_native(mgha, slope, intercept, mu_scaling=8192.0):
    """Density threshold (mg HA/ccm) -> native short number, as the /seg_gauss log prints it
    ('Thresholds correspond to native numbers'): round-to-nearest of (mgHA - intercept) / slope * Mu_Scaling.
    PFJ-0be66a: 500 -> 4523.83 -> 4524 (truncation to 4523 is refuted: 4,540 mismatches), 3000 -> 17173.01 -> 17173."""
    mu = (float(mgha) - float(intercept)) / float(slope)
    return int(np.rint(mu * float(mu_scaling)))


def _native_threshold(x, name):
    """IPL's native thresholds are short numbers (the log prints them as integers): require an integral value
    (int, numpy int, or an integral float such as 4524.0).  A fractional value would otherwise be silently
    truncated (4523.83 -> 4523, the refuted rule); convert mgHA with mgha_to_native instead."""
    f = float(x)
    if not np.isfinite(f) or f != int(f):
        raise ValueError(f"seg_gauss: {name}={x!r} is not an integral native number; use mgha_to_native / "
                         "calibration_from_proclog to convert mgHA thresholds")
    return int(f)


def calibration_from_proclog(src):
    """Density slope / intercept and Mu_Scaling from an AIM processing log ('Density: slope', 'Density: intercept',
    'Mu_Scaling'); `src` is the proclog text or a volume dict from ipldt.io.read_aim (its 'proclog' entry).
    Raises ValueError when a field is missing (no default is substituted)."""
    log = src.get("proclog", "") if isinstance(src, dict) else src
    if isinstance(log, bytes):
        log = log.decode("latin-1", "replace")
    out = {}
    for key, pattern in (("slope", r"Density:\s*slope\s+([-+0-9.eE]+)"),
                         ("intercept", r"Density:\s*intercept\s+([-+0-9.eE]+)"),
                         ("mu_scaling", r"Mu_Scaling\s+([-+0-9.eE]+)")):
        m = re.search(pattern, log or "")
        if not m:
            raise ValueError(f"calibration field '{key}' not found in the AIM processing log")
        out[key] = float(m.group(1))
    return out


def _gauss_pass_f32_trunc(v, w, axis):
    """One 1-D pass of IPL's low-pass along `axis`: float32 centre-out PAIR accumulation
        acc = w0 * v[i]; acc += w1 * (v[i-1] + v[i+1]); acc += w2 * (v[i-2] + v[i+2]); ...
    with zero padding, then truncation toward zero (the C cast to short).  `v` is a float32 array holding
    integers; returns a float32 array of the same shape holding integers.  Every operation is a float32
    ufunc written in place (out=), so four float32 arrays live at the peak: v, its padded copy, acc, pair."""
    r = (len(w) - 1) // 2
    n = v.shape[axis]
    pad = [(0, 0)] * v.ndim
    pad[axis] = (r, r)
    vp = np.pad(v, pad, mode="constant", constant_values=np.float32(0))

    def shifted(k):
        sl = [slice(None)] * v.ndim
        sl[axis] = slice(r + k, r + k + n)
        return vp[tuple(sl)]

    acc = np.multiply(shifted(0), np.float32(w[r]), dtype=np.float32)
    pair = np.empty_like(acc)
    for k in range(1, r + 1):
        np.add(shifted(-k), shifted(k), out=pair)                    # exact: integers below 2^24
        np.multiply(pair, np.float32(w[r + k]), out=pair)
        np.add(acc, pair, out=acc)
    np.trunc(acc, out=acc)
    return acc


def gauss_lp_ipl(box_int, sigma=2.0, support=3):
    """IPL's separable Gaussian low-pass of a short volume (z, y, x): three passes x, y, z, each in float32
    with the result stored as short (truncated) before the next pass.  Returns a float32 array on the full
    input grid holding the integer smoothed values; the valid region is [r:-r, r:-r, r:-r], r = support.
    Transient memory: four float32 copies of the box at the peak (about 650 MiB for PFJ-0be66a's 42.5 M-voxel
    aim_bbc); the labelling stages of Step 1 remain the chain's ceiling (about 900 MiB on PFJ-0be66a).
    REFUTED (stage-01 mismatches on PFJ-0be66a): rint instead of truncation per pass ~6,900; no per-pass
    truncation 4,540; other axis orders 440-640; two axes grouped into one 2-D pass ~2,220; other float32
    accumulation orders (sequential -r..r / r..-r, outside-in pairs, centre-out single taps) 2 / 4 / 3 / 4."""
    w = ipl_gauss_weights(sigma, support)
    v = np.asarray(box_int).astype(np.float32)
    for axis in (2, 1, 0):                  # x, then y, then z
        v = _gauss_pass_f32_trunc(v, w, axis)
    return v


def seg_gauss(v, sigma, support, lower_native, upper_native, value_in_range=SET):
    """/seg_gauss -sigma -support -lower/upper (native numbers) -value_in_range on a short volume whose voxels
    outside the object are 0 (Script 32: the periosteal-masked greyscale cut to its bounding box, 'aim_bbc').
    Output: char volume on the input grid shrunk by `support` on every side ('valid' convolution), voxel set
    <=> lower_native <= smoothed <= upper_native, BOTH INCLUSIVE.  Lower bound on PFJ-0be66a: all 4,585 voxels at
    exactly 4524 are IPL-included, all 4,578 at 4523 excluded.  Both bounds verified on designed phantoms
    (probe 18, 2026-09-13, export X2420448_P18_SG: two 15x15 ramps painted into PFJ-0be66a's greyscale whose
    smoothed values 17171..17175 and 4522..4526 each occur on exactly 81 voxels): IPL keeps 17173 and drops
    17174 (refuting v < 17173 by 81 voxels and v <= 17174 by 82), keeps 4524 and drops 4523 (refuting >= 4525
    by 7 / 87 and >= 4523, the truncation of 4523.83, by 9 / 91 voxels in the two windows).  Cohort: only
    PFJ-6f5538_R (X5492058) smooths above the upper bound (770 voxels, max 19,289, window slices 126-136).
    Only sigma 2 / support 3 (Script 32 / 33) have an oracle; other values are untested, and support < 1 raises
    ValueError (no oracle, so no guess).  The native thresholds are integers (IPL's log prints them as such): a
    fractional value raises ValueError instead of being truncated (4523.83 -> 4523 would be the refuted rule).
    Verified: 0 mismatches over the 39,962,808 voxels of X2420448_P16_01_SEGGAUSS (732x337x162 @ 801,100,171).
    Use mgha_to_native / calibration_from_proclog to convert the script's mgHA thresholds."""
    r = int(support)
    if r < 1:
        raise ValueError(f"seg_gauss: support must be >= 1, got {support}")
    data = np.asarray(v["data"])
    if any(s <= 2 * r for s in data.shape):
        raise ValueError(f"seg_gauss: volume {v['dim']} is too small for support {r}")
    lo, up = _native_threshold(lower_native, "lower_native"), _native_threshold(upper_native, "upper_native")
    sm = gauss_lp_ipl(data, float(sigma), r)[r:-r, r:-r, r:-r]
    keep = (sm >= np.float32(lo)) & (sm <= np.float32(up))
    dim = tuple(d - 2 * r for d in v["dim"])
    pos = tuple(p + r for p in v["pos"])
    return mask_vol(keep, dim, pos, value_in_range)


# ============================================================================================== metric 11
_INF = np.int32(1_000_000_000)


@njit(cache=True)
def chamfer_dt_345(obj):
    """Two-pass (forward / backward raster, full 26-neighbourhood) chamfer distance with weights 3 (face),
    4 (edge), 5 (corner) of every non-zero voxel of `obj` (uint8, (z, y, x)) to the nearest zero voxel;
    zero voxels get 0.  Nothing outside the array is considered (open boundary): pad first if the outside
    must count as background.  IPL's log: 'Distance map with chamfer metric (3, 4, 5)'."""
    a, b, c = 3, 4, 5
    nz, ny, nx = obj.shape
    d = np.empty((nz, ny, nx), np.int32)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                d[z, y, x] = _INF if obj[z, y, x] != 0 else 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                v = d[z, y, x]
                if v == 0:
                    continue
                if x > 0:
                    t = d[z, y, x - 1] + a
                    if t < v:
                        v = t
                if y > 0:
                    t = d[z, y - 1, x] + a
                    if t < v:
                        v = t
                    if x > 0:
                        t = d[z, y - 1, x - 1] + b
                        if t < v:
                            v = t
                    if x < nx - 1:
                        t = d[z, y - 1, x + 1] + b
                        if t < v:
                            v = t
                if z > 0:
                    t = d[z - 1, y, x] + a
                    if t < v:
                        v = t
                    if x > 0:
                        t = d[z - 1, y, x - 1] + b
                        if t < v:
                            v = t
                    if x < nx - 1:
                        t = d[z - 1, y, x + 1] + b
                        if t < v:
                            v = t
                    if y > 0:
                        t = d[z - 1, y - 1, x] + b
                        if t < v:
                            v = t
                        if x > 0:
                            t = d[z - 1, y - 1, x - 1] + c
                            if t < v:
                                v = t
                        if x < nx - 1:
                            t = d[z - 1, y - 1, x + 1] + c
                            if t < v:
                                v = t
                    if y < ny - 1:
                        t = d[z - 1, y + 1, x] + b
                        if t < v:
                            v = t
                        if x > 0:
                            t = d[z - 1, y + 1, x - 1] + c
                            if t < v:
                                v = t
                        if x < nx - 1:
                            t = d[z - 1, y + 1, x + 1] + c
                            if t < v:
                                v = t
                d[z, y, x] = v
    for z in range(nz - 1, -1, -1):
        for y in range(ny - 1, -1, -1):
            for x in range(nx - 1, -1, -1):
                v = d[z, y, x]
                if v == 0:
                    continue
                if x < nx - 1:
                    t = d[z, y, x + 1] + a
                    if t < v:
                        v = t
                if y < ny - 1:
                    t = d[z, y + 1, x] + a
                    if t < v:
                        v = t
                    if x > 0:
                        t = d[z, y + 1, x - 1] + b
                        if t < v:
                            v = t
                    if x < nx - 1:
                        t = d[z, y + 1, x + 1] + b
                        if t < v:
                            v = t
                if z < nz - 1:
                    t = d[z + 1, y, x] + a
                    if t < v:
                        v = t
                    if x > 0:
                        t = d[z + 1, y, x - 1] + b
                        if t < v:
                            v = t
                    if x < nx - 1:
                        t = d[z + 1, y, x + 1] + b
                        if t < v:
                            v = t
                    if y > 0:
                        t = d[z + 1, y - 1, x] + b
                        if t < v:
                            v = t
                        if x > 0:
                            t = d[z + 1, y - 1, x - 1] + c
                            if t < v:
                                v = t
                        if x < nx - 1:
                            t = d[z + 1, y - 1, x + 1] + c
                            if t < v:
                                v = t
                    if y < ny - 1:
                        t = d[z + 1, y + 1, x] + b
                        if t < v:
                            v = t
                        if x > 0:
                            t = d[z + 1, y + 1, x - 1] + c
                            if t < v:
                                v = t
                        if x < nx - 1:
                            t = d[z + 1, y + 1, x + 1] + c
                            if t < v:
                                v = t
                d[z, y, x] = v
    return d


def metric11_threshold(n):
    """keep <=> raw chamfer >= 3N + 2, i.e. rint(raw / 3) >= N + 1 (IPL prints 'Thresholds are gray-scale:
    N+1 ...' for the map scaled to voxel units; raw is an integer, so there are no ties).
    REFUTED on stage 09 (erosion 3 of 08): raw >= 3N+1 leaves 299,145 mismatches, raw >= 3N+3 311,424,
    raw >= N+1 (raw units) 3,142,551; Euclidean, city-block and chessboard variants 0.7-2.1 M.
    Verified on designed phantoms (probe 18, 2026-09-13) for N = 1 and 3 (erosion, open, close) and N = 1 and
    15 (dilation): on a 13^3 block with a 1-voxel hole the alternatives 3N+1 / 3N+3 / Euclidean differ from
    3N+2 by 32 / 48 / 32 voxels (erosion 3) and 12 / 8 / 12 (erosion 1), all refuted; exports ERO3, ERO1,
    DIL15, DIL1, CLOSE3, OPEN3 all matched the 3N+2 reading exactly.
    Probe 19 (2026-09-14): at N = 15 on a real volume, PFJ-411dfd_R's stage 14 (X5143651_P17_14_BBC) and three
    /sub_get copies of it, /erosion 15 (raw >= 47) and /dilation 15 (raw < 47) match with 0 mismatches
    (13,640,114 / 20,637,201 set voxels on the full volume).  IPL's log prints 'Thresholds are gray-scale:
    16 32767' for the erosion and for both halves of /open, and '16 10000000' for the standalone dilation:
    16 = N + 1 in the scaled (voxel) units of the map, i.e. keep rint(raw / 3) >= 16 <=> raw >= 47.
    Probe 20 (2026-09-15) exercised the same threshold at seven more distances on that volume -- /open 11, 12, 13,
    14, 16, 17 and 18 (exports X5143651_P20_OPEN11 .. OPEN18) -- with 0 mismatching voxels at every N (17,833,531 /
    17,808,269 / 17,777,932 / 17,735,682 / 17,648,505 / 17,626,174 / 17,601,750 set voxels), and at N = 15 through
    /close and /dilation under both settings of -continuous_at_boundary (close15c0 / close15c1 / dil15c1 /
    dil15c001, all 0).  IPL's log prints one 'Thresholds are gray-scale: N+1 32767' pair per /open (two lines, one
    per phase) and one per /close and /dilation, and the BorderChange margin it announces grows with N as N + 2
    ('rel beg_pos -13 -13 -13' at N = 11 through '-20 -20 -20' at N = 18), which is what makes the cave T-scan read
    the margin depth off the data (see open_)."""
    return 3 * int(n) + 2


def _erode_bool(m_bool, n):
    """out = in & (dt(in) >= 3N+2) on the array as given (open boundary)."""
    return m_bool & (chamfer_dt_345(_u8(m_bool)) >= metric11_threshold(n))


def _dilate_padded(m_bool, n, margin):
    """Pad a background margin, out = in | (dt(~in) < 3N+2); returned on the padded grid."""
    mp = np.pad(np.asarray(m_bool, dtype=bool), int(margin))
    return mp | (chamfer_dt_345(_u8(~mp)) < metric11_threshold(n))


# ------------------------------------------------------------------------ -continuous_at_boundary (IPL order x y z)
BOUNDARY_FILLS = {0: "background", 1: "symmetric"}


def _boundary_modes(name, continuous_at_boundary):
    """IPL's '-continuous_at_boundary cx cy cz' -> the per-axis margin fill, returned in ipldt's INTERNAL (z, y, x)
    array order.  The argument is an int (the same flag on all three axes) or a 3-tuple in IPL's own x, y, z order;
    each flag is 0 or 1; IPL's help for /close and /dilation documents that 0 adds an empty (background) border and
    that 1 mirrors the object into the border, i.e. 0 -> a BACKGROUND margin and 1 -> the
    EDGE-INCLUSIVE MIRROR of the operand (numpy pad 'symmetric'), the same fill the erosion / open buffer carries.
    The flag is applied PER AXIS: probe 20's dil15c001 ('/dilation 15 -continuous_at_boundary 0 0 1', mirror on z
    only) matches this reading at 0 mismatching voxels, while the axis-swapped reading (mirror on x only) differs by
    598,104 voxels -- so the order is x y z (probe 20, 2026-09-15)."""
    c = continuous_at_boundary
    if isinstance(c, (bool, np.bool_, int, np.integer)):
        flags = (c,) * 3
    elif isinstance(c, (tuple, list, np.ndarray)):
        flags = tuple(c)
    else:
        raise ValueError(f"{name}: continuous_at_boundary must be an int or a 3-tuple (x, y, z), got {c!r}")
    if len(flags) != 3:
        raise ValueError(f"{name}: continuous_at_boundary must have 3 entries (x, y, z), got {c!r}")
    modes = []
    for f in flags:
        if not isinstance(f, (bool, np.bool_, int, np.integer)) or int(f) not in BOUNDARY_FILLS:
            raise ValueError(f"{name}: continuous_at_boundary entries must be the integer 0 (empty border) or 1 "
                             f"(object mirrored into the border), got {c!r}")
        modes.append(BOUNDARY_FILLS[int(f)])
    return tuple(modes[::-1])                                    # (x, y, z) -> (z, y, x)


def _pad_boundary(m_bool, margin, modes_zyx):
    """The operand padded by `margin` on all six faces, each axis with its own fill: 'background' (an empty border)
    or 'symmetric' (the edge-inclusive mirror).  Order-independent -- a background margin stays background under a
    mirror on another axis, and two mirrors commute -- so the corners are the same whichever axis is padded first."""
    out = np.asarray(m_bool, dtype=bool)
    margin = int(margin)
    if margin == 0:
        return out
    for ax, mode in enumerate(modes_zyx):
        pw = [(0, 0)] * 3
        pw[ax] = (margin, margin)
        if mode == "symmetric":
            out = np.pad(out, pw, mode="symmetric")
        else:
            out = np.pad(out, pw, mode="constant", constant_values=False)
    return out


OPEN_MARGIN_MODES = ("symmetric", "reflect", "edge", "wrap", "background", "object", "open_boundary")


def _open_padded(m_bool, n, margin, mode="symmetric"):
    """/open -open_distance N -metric 11 under ONE margin convention of the erosion's D3P_BorderChange buffer;
    returns a bool (z, y, x) array on the input grid (the buffer cropped by `margin` on every side).
    mode  'symmetric'     THE MECHANISM (what open_ uses with margin N + 2): the input padded by `margin` on all
                          six faces with its EDGE-INCLUSIVE mirror image (numpy pad 'symmetric': margin depth k
                          holds the slice at distance k - 1 from the face, depth 1 duplicates the face slice);
                          the erosion runs on the whole buffer, and EVERY survivor -- margin included -- seeds the
                          dilation on the same buffer (the margin is not re-filled).
          'reflect'       the edge-EXCLUSIVE mirror (numpy 'reflect': depth k holds the slice at distance k);
          'edge'          the face slice replicated; 'wrap' a periodic margin; 'background' all background (which
                          also erodes the faces); 'object' all object -- the discriminated alternatives (their
                          mismatch counts against IPL are in open_), named here so tests and probes can run them.
          'open_boundary' the PRE-MECHANISM rule (open_ before 2026-09-14 evening): the erosion on the UNPADDED
                          grid (nothing outside the array is background), then the dilation on a background
                          margin of `margin` -- i.e. the two chained primitives erosion() / dilation(), cropped.
    Thin volumes (an axis shorter than `margin`): numpy 'symmetric' / 'reflect' REPEAT the reflection.  IPL does the
    same -- probe 20's z156 export ('/sub_get' of 12 slices, 721x290x12, opened with N = 15 so margin 17 = one full
    reflection plus 5 repeated layers) matches 'symmetric' at 0 mismatching voxels and refutes a single reflection
    completed with background / object / the far face slice by 11,803 / 3 / 3 voxels (probe 20,
    2026-09-15).  Only margin > 2 * axis length (more than one repeat) remains untested; open_ warns there."""
    m_bool = np.asarray(m_bool, dtype=bool)
    n = _distance("open", n)
    margin = int(margin)
    if margin < 0:
        raise ValueError(f"open: margin must be >= 0, got {margin}")
    t = metric11_threshold(n)
    if mode == "open_boundary":
        d = _dilate_padded(_erode_bool(m_bool, n), n, margin)
    else:
        if mode in ("symmetric", "reflect", "edge", "wrap"):
            mp = np.pad(m_bool, margin, mode=mode)
        elif mode == "background":
            mp = np.pad(m_bool, margin, mode="constant", constant_values=False)
        elif mode == "object":
            mp = np.pad(m_bool, margin, mode="constant", constant_values=True)
        else:
            raise ValueError(f"open: unknown margin mode {mode!r}; one of {OPEN_MARGIN_MODES}")
        surv = mp & (chamfer_dt_345(_u8(mp)) >= t)
        d = surv | (chamfer_dt_345(_u8(~surv)) < t)
    if margin == 0:
        return d
    return d[margin:-margin, margin:-margin, margin:-margin]


class ThinVolumeWarning(UserWarning):
    """A mirrored margin DEEPER THAN TWICE an axis length: numpy's 'symmetric' pad then repeats the reflection more
    than once, and that is the one margin regime probe 20 does not settle.  A margin up to 2 * the axis length (one
    reflection plus at most one repeat) IS settled: probe 20's z156 export, 721x290x12 opened with N = 15 (margin 17
    against 12 slices), matches numpy 'symmetric' at 0 mismatching voxels (2026-09-15)."""


_thin_volume_warned = False


def _mirror_repeat_axes(shape, margin, modes_zyx):
    """The axes whose mirrored margin is deeper than twice the axis length -- i.e. where numpy's 'symmetric' pad
    repeats the reflection more than once, the regime probe 20 leaves untested.  shape and modes_zyx are in the
    internal (z, y, x) order; axes with a background margin are never thin."""
    margin = int(margin)
    return tuple(ax for ax in range(3) if modes_zyx[ax] == "symmetric" and margin > 2 * int(shape[ax]))


def _warn_thin_volume(name, dim, margin, axes):
    """Once per process: a mirrored margin of `margin` is more than twice as deep as one of the volume's axes, so
    numpy's 'symmetric' pad repeats the reflection more than once.  IPL's fill is verified up to one repeat (probe
    20 'z156': 12 slices, margin 17, 0 mismatching voxels, 2026-09-15) and untested beyond it."""
    global _thin_volume_warned
    if _thin_volume_warned:
        return
    _thin_volume_warned = True
    names = ", ".join("zyx"[ax] for ax in axes)
    warnings.warn(f"{name}: volume dim {tuple(dim)} is thinner than half the mirrored margin {margin} on axis "
                  f"{names}; numpy's 'symmetric' pad then repeats the reflection more than once, where IPL's fill "
                  "is untested (probe 20 settles it up to one repeat: export 'z156', 12 slices with margin 17, "
                  "0 mismatching voxels, 2026-09-15) (warned once per process)",
                  ThinVolumeWarning, stacklevel=3)


def _distance(name, n):
    """The -*_distance N of the metric-11 commands: an integer >= 0 (N = 0 is legal; a negative N has no
    meaning and raises ValueError instead of shrinking the grid or failing inside numpy)."""
    n = int(n)
    if n < 0:
        raise ValueError(f"{name}: distance must be >= 0, got {n}")
    return n


def erosion(v, n):
    """/erosion -erode_distance N -metric 11 (-use_previous_margin false).
    RULE: out = in & (chamfer_dt(in) >= 3N + 2) with an OPEN volume boundary: voxels outside the array are
    neither object nor background, so an object voxel on a face of the volume is kept unless an in-volume
    background voxel is closer than 3N + 2.  Grid unchanged (IPL's D3P_BorderChange margin of N + 2 is only
    'off' and is dropped when the AIM is written).
    WHAT IPL'S MARGIN HOLDS (settled 2026-09-14 evening through /open, see open_): the N + 2 margin is filled on
    all six faces with the EDGE-INCLUSIVE MIRROR image of the input (margin depth k = the slice at distance
    k - 1 from the face) and the erosion's chamfer map runs on the whole buffer.  Inside the volume that is
    exactly the open-boundary rule above: the reflection is an isometry of the chamfer metric and every mirrored
    voxel lies strictly farther from any in-volume voxel than its original, so no mirrored background voxel can
    erode an in-volume voxel its original does not (on PFJ-411dfd_R 15 <- 14 the in-volume erosion is voxel-identical
    with and without the mirrored margin: 0 differences).  The margin's own survivors are dropped with the margin when /erosion
    writes its result -- they matter only when the buffer is reused, which /open does.  This function therefore
    computes the open-boundary erosion on the input grid, unchanged since probe 18.
    Verified: 09_ero3 <- 08 (15,752,082 set voxels) and 17_cornero <- 16 (6,550) both 0 mismatches.  The open
    boundary is PROVEN on PFJ-0be66a for the z faces (09: 311,091 / 300,826 discriminating voxels at z0 / z1 =
    611,917) and the y0 face (17: 15 voxels), and on designed phantoms (probe 18, 2026-09-13) for ALL SIX
    faces: 20x9x9 blocks on the x0 / x1 / y0 / y1 faces and 9^3 blocks on the z faces, erosion 3 and 1
    (exports ERO3, ERO1; a background-padded face would have removed 27 / 49 voxels per block), and again
    inside /open 3 (OPEN3).  IPL's D3P_BorderChange margin is the same N + 2 on all three axes (log: rel
    beg_pos -5 -5 -5, out dim 753 358 178 for N = 3).
    Probe 17 (2026-09-14): 09_ero3 <- 08 and 17_cornero <- 16 0 mismatches on PFJ-0be66a_R, PFJ-42293d_L, PFJ-6f5538_R
    (RADIUS) and PFJ-411dfd_R (TIBIA): 15,752,082 / 13,548,957 / 699,709 / 12,925,282 and 6,550 / 45,279 /
    33,696 / 60,152 set voxels.
    Probe 19 (2026-09-14): /erosion 15 on PFJ-411dfd_R's stage 14 (X5143651_P17_14_BBC, 721x290x168 @ 856,24,0,
    the input of the /open residual documented in open_) exported as X5143651_P19_ERO15 = this rule exactly:
    13,640,114 set voxels, 0 mismatches, grid unchanged; a second run (ERO15B) is voxel-identical (IPL's erosion
    is deterministic); /erosion 15 on IPL's own /sub_get copies of 14 (suba z 100..167, subb z 0..166, subc
    x 400..720 y 40..289 z 100..167) 0 mismatches (5,521,647 / 13,572,701 / 2,677,855 set voxels), so the open
    boundary holds at freshly cut faces as well.  IPL's log of that run
    reports a margin of N + 2 = 17 voxels added on all six faces (721 x 290 x 168 -> 755 x 324 x 202), the
    two-pass 3-4-5 chamfer distance map and a grey-scale threshold of 16 (keep rint(raw / 3) >= 16 <=> raw >= 47);
    CPU 7.04 s.  The erosion half of /open is therefore exact on the very input of the
    residual; the residual is not in this primitive.
    Probe 20 (2026-09-15) CONFIRMED the mirror margin prospectively, through /open (see open_) and directly here:
    /erosion 15 on the region phantom (export X5143651_P20_K0_ERO15) = this function at 0 mismatching voxels
    (192,181 set voxels), and IPL's own /dilation 15 applied to that very export (K0_DIL15, 713,838 set voxels)
    equals dilation() at 0 -- while IPL's own /open 15 of the same phantom exceeds that chain by EXACTLY the 3
    residual voxels, which is the whole effect reproduced inside a 121x101x38 volume.  /erosion carries NO
    -continuous_at_boundary flag (neither does /open): its border is always the mirror, and that is why this
    function needs no argument -- the mirrored margin's own survivors are dropped when /erosion writes its result.
    REFUTED: background-padded erosion removes the 3 z-layers at each face of 09 (611,917 IPL-only voxels)
    and 15 voxels at the y0 face of 17."""
    return mask_vol(_erode_bool(_set(v), _distance("erosion", n)), v["dim"], v["pos"])


def dilation(v, n, continuous_at_boundary=(0, 0, 0)):
    """/dilation -dilate_distance N -continuous_at_boundary cx cy cz -use_previous_margin false -metric 11.
    RULE: a margin of N + 2 is added on all six faces (z included), out = in | (dt(~in) < 3N + 2)
    (background voxels with rint(raw / 3) <= N become object); the object grows into the margin.  The written
    grid is the input grid grown by N + 1 on every side (IPL keeps its N + 2 margin with off = 1).
    -continuous_at_boundary (an int or a 3-tuple in IPL's x, y, z order; DEFAULT (0, 0, 0) = the only form Scripts
    32 / 33 use, so every existing call site and oracle is unaffected): per axis, 0 fills that axis's margin with
    BACKGROUND and 1 with the EDGE-INCLUSIVE MIRROR of the operand (the two border modes IPL's help documents for
    /close and /dilation; see _boundary_modes).  Probe 20 (scanner run 2026-09-15) gives the
    oracles on PFJ-411dfd_R's stage 14: '1 1 1' (export X5143651_P20_DIL15C1) and '0 0 1' (DIL15C001) each match this
    rule at 0 mismatching voxels ON THE WHOLE WRITTEN GRID, the 16 written margin layers included -- 27,387,249 and
    27,062,074 set voxels on 753x322x200 @ 840,8,-16 -- so the written margin exhibits the fill directly.  The two
    exports differ from each other by 325,175 voxels; against DIL15C1 the edge-EXCLUSIVE mirror ('reflect') differs
    by 70,802 (57,263 ours-only / 13,539 IPL-only), edge replication by 416,854, a background margin (the default
    rule) by 787,238 and an object margin by 14,517,704; against DIL15C001 by 19,772 / 121,266 / 462,063 /
    6,637,962, and reading the flags in z, y, x order instead (mirror on x only) differs by 598,104 -- the per-axis
    application and the x y z order are both pinned by data (probe 20, 2026-09-15).
    Verified: 11_dil3 <- 10 (19,463,848 set voxels) 0 mismatches, grid 743x348x168 @ 795,94,168 ->
    751x356x176 @ 791,90,164 reproduced.  On designed phantoms (probe 18, 2026-09-13): an isolated 5^3 block
    dilated by 15 and by 1 (exports DIL15, DIL1) matched the raw < 3N + 2 ball exactly (3N+1 / 3N+3 /
    Euclidean r <= N / r <= N + 0.5 differ by 1,224 / 1,272 / 1,724 / 2,900 voxels at N = 15, 60 / 8 / 60 at
    N = 1) and the written headers were 775x380x200 @ 779,78,152 and 747x352x172 @ 793,92,166 = the input
    grid grown by N + 1 per side (N and N + 2 refuted).
    Probe 17 (2026-09-14): 11_dil3 <- 10 and 19_cornmajor <- 18 0 mismatches and the grown grids reproduced on
    all four subjects (11: 19,463,848 / 18,777,844 / 1,458,872 / 16,641,696 set voxels; 19: 12,879 / 139,031 /
    91,420 / 0 -- the corner branch is non-empty under the RADIUS preset).
    Probe 19 (2026-09-14): /dilation 15 (-continuous_at_boundary 0 0 0 -use_previous_margin false) on IPL's own
    /erosion 15 of PFJ-411dfd_R's stage 14 (X5143651_P19_ERO15) exported as X5143651_P19_DIL15 = this rule exactly:
    20,637,201 set voxels on 753x322x200 @ 840,8,-16 = the 721x290x168 @ 856,24,0 input grown by N + 1 = 16 per
    side, 0 mismatches -- the grid growth is now verified at N = 15 on a real volume (probe 18 verified it on a
    phantom); on IPL's /sub_get copies suba / subb / subc (see erosion) 0 mismatches as well (10,168,167 /
    20,545,954 / 5,094,680 set voxels on 753x322x100 @ 840,8,84, 753x322x199 @ 840,8,-16, 353x282x100 @
    1240,48,84).  IPL's log of the run reports an
    inversion, a border change that leaves the 755 x 324 x 202 buffer as it is (the in-memory erosion result still
    carries its N + 2 margin, off 17, so no growth), the four chamfer DT steps, a grey-scale threshold of 16 with
    the upper bound 10000000, and a re-inversion; the result's
    off is 1 (the written grid keeps N + 1 of the 17); CPU 7.97 s.  The dilation half of /open is therefore
    exact on the very input of the residual (see open_); the residual is not in this primitive.
    -use_previous_margin true (X5143651_P19_DIL15PM: /dilation 15 on ERO15B; NOT implemented here -- Scripts 32
    and 33 pass false): the written grid is the input grid SHRUNK by N + 1 per side, 689x258x136 @ 872,40,16,
    and its content is this rule's dilation restricted to that grid (0 mismatches, 14,356,093 set voxels; it also
    equals IPL's /open 15 there).  IPL's log reports that the margin of
    the previous erosion is reused, that -continuous_at_boundary is then ignored, that the dilation distance must
    not exceed the previous erosion distance and that the offset is set so that the output keeps the original box;
    /examine shows dim 755 324 202 off 33 33 33 (17 + 16).  Documented only.
    REFUTED: an open z boundary leaves 702,932 IPL-only voxels (IPL grows into the z margin)."""
    n = _distance("dilation", n)
    modes = _boundary_modes("dilation", continuous_at_boundary)
    m = n + 2
    mask = _set(v)
    thin = _mirror_repeat_axes(mask.shape, m, modes)
    if thin:
        _warn_thin_volume("dilation", v["dim"], m, thin)
    mp = _pad_boundary(mask, m, modes)
    g = (mp | (chamfer_dt_345(_u8(~mp)) < metric11_threshold(n)))[1:-1, 1:-1, 1:-1]
    k = n + 1
    return mask_vol(g, tuple(d + 2 * k for d in v["dim"]), tuple(p - k for p in v["pos"]))


def close(v, n, continuous_at_boundary=(0, 0, 0)):
    """/close -close_distance N -continuous_at_boundary cx cy cz -metric 11 = the dilation (margin N + 2) followed
    by the erosion (threshold 3N + 2) on that same buffer, cropped back to the input grid (IPL keeps off = N + 2).
    -continuous_at_boundary (an int or a 3-tuple in IPL's x, y, z order; DEFAULT (0, 0, 0) = the only form Scripts
    32 / 33 use, so every existing call site and oracle is unaffected): per axis, 0 = a BACKGROUND margin, 1 = the
    EDGE-INCLUSIVE MIRROR of the input, exactly as in dilation().
    Verified: 12_close15 <- 11 (21,034,566 set voxels) and 23_close50 <- 22 (20,333,641) 0 mismatches; on
    designed phantoms (probe 18, 2026-09-13, export CLOSE3) two 10^3 blocks 2 apart are bridged and two 6 apart
    (the bridging limit of close 3) exactly as the 3N+2 rule predicts (3N+1 / 3N+3 / Euclidean differ there by
    88 / 112 / 88 voxels).
    Probe 17 (2026-09-14): 12_close15 <- 11 on PFJ-0be66a_R / PFJ-42293d_L / PFJ-6f5538_R / PFJ-411dfd_R (21,034,566 /
    22,076,107 / 2,066,861 / 18,578,278 set voxels), 23_close30 <- 22 on the three RADIUS subjects (20,302,109 /
    21,606,958 / 1,954,987; the close-30 oracle of Script 33) and 23_close50 <- 22 on PFJ-411dfd_R (17,989,866):
    0 mismatches, identical grids -- close 3 / 15 / 30 / 50 are all verified.
    Probe 19 (2026-09-14): no new close oracle; the two primitives close is composed of were re-verified at
    N = 15 on a real volume, PFJ-411dfd_R's stage 14 and its three /sub_get copies (/erosion 15 and /dilation 15
    each 0 mismatches, the dilation's grid growth N + 1 included; see erosion, dilation), and the four close
    oracles above stand.  The 3-voxel residual of /open on that volume (open_) has no counterpart in any close
    oracle (12_close15 on the same subject is exact), and probe 19 located it inside /open's internal chaining,
    not in the primitives.  UNCHANGED by the /open mechanism (implemented 2026-09-14 evening, open_): /close
    -continuous_at_boundary 0 0 0 (the only form Scripts 32 / 33 use) is the dilation on a BACKGROUND margin
    (border mode 0 of IPL's help for /close and /dilation; see _boundary_modes), followed by the erosion on that
    buffer, which is exactly this rule, and
    the six close oracles (3 / 15 / 30 / 50) stay at 0 with the mechanism in place.
    Probe 20 (scanner run 2026-09-15) settled BOTH readings of the flag on PFJ-411dfd_R's stage 14, /close 15:
    '-continuous_at_boundary 0 0 0' (export X5143651_P20_CLOSE15C0) = this function's default at 0 mismatching
    voxels (17,964,657 set voxels), and '1 1 1' (CLOSE15C1) = the mirrored margin at 0 (18,003,002).  The two IPL
    exports differ from each other by 38,345 voxels, which is also the distance from each reading to the other
    export -- so the hypothesis that IPL IGNORES the flag and always mirrors is refuted by 38,345 voxels, and the
    mechanism's own prediction (the fill the flag names, per axis) is confirmed on both.  Against CLOSE15C1 an edge
    margin differs by 29,463 and an object margin by 954,300.  NOTE: /close alone does NOT discriminate the
    edge-inclusive from the edge-exclusive mirror -- 'symmetric' and 'reflect' both match CLOSE15C1 at 0, because
    /close's erosion phase runs on the DILATED buffer -- that discrimination comes from /open (reflect 213 / 162 at
    N = 11 / 12) and from /dilation with '1 1 1' (reflect 70,802); see open_ and dilation.
    REFUTED: an open z boundary in the dilation 278,112 (close 15) / 994,711 (close 50); thresholds 3N+1 /
    3N+3 in the erosion 301,515 / 305,557 (close 15), 997,345 / 1,001,019 (close 50)."""
    n = _distance("close", n)
    m = n + 2
    modes = _boundary_modes("close", continuous_at_boundary)
    mask = _set(v)
    thin = _mirror_repeat_axes(mask.shape, m, modes)
    if thin:
        _warn_thin_volume("close", v["dim"], m, thin)
    mp = _pad_boundary(mask, m, modes)
    d = mp | (chamfer_dt_345(_u8(~mp)) < metric11_threshold(n))
    e = _erode_bool(d, n)
    return mask_vol(e[m:-m, m:-m, m:-m], v["dim"], v["pos"])


def open_(v, n):
    """/open -open_distance N -metric 11 = ONE pipeline on the erosion's mirror-padded buffer (the mechanism of
    the PFJ-411dfd_R residual, found and implemented 2026-09-14 evening, CONFIRMED prospectively by probe 20 on
    2026-09-15; /open takes no -continuous_at_boundary flag, its border is always the mirror).
    RULE (in numpy): Mp = np.pad(M, N + 2, mode='symmetric'); surv = Mp & (chamfer_dt_345(Mp) >= 3N + 2);
    out = (surv | (chamfer_dt_345(~surv) < 3N + 2))[N+2:-(N+2), N+2:-(N+2), N+2:-(N+2)]  (= _open_padded(M, N,
    N + 2, 'symmetric')).  In words: the erosion's D3P_BorderChange adds a margin of N + 2 on all six faces and
    fills it with the EDGE-INCLUSIVE mirror image of the input (margin depth k holds the slice at distance k - 1
    from the face, depth 1 duplicates the face slice); the erosion's chamfer map runs on the whole padded buffer and
    keeps raw >= 3N + 2 everywhere, margin included; /open's dilation half reuses that buffer WITHOUT re-filling
    the margin (the log prints only one BorderChange), so every erosion survivor in the margin seeds the dilation
    ball (dt(~surv) < 3N + 2); the result is cropped to the input grid.  Anti-extensive on the input grid: a
    survivor of the buffer has every buffer voxel within 3N + 1 of it set, so every in-volume voxel its ball
    restores is object in the input (the pre-mechanism argument, applied to the buffer).
    WHY THE PRIMITIVES NEVER SHOWED IT.  The reflection is an isometry of the chamfer metric and every mirrored
    voxel lies strictly farther from any in-volume voxel than its original, so (a) inside the volume the erosion
    equals the open-boundary erosion of erosion() -- every erosion oracle is unchanged -- and (b) a margin survivor
    that is the mirror of an in-volume survivor restores nothing its partner does not.  Only the TRUNCATION of the
    mirror at depth N + 2 creates 'extra' survivors: a margin voxel whose partner is eroded solely by background
    lying deeper than N + 2 slices from the face sees no background within 3N + 1 and survives although its
    partner does not; at depth <= N its ball reaches the face slice.  On PFJ-411dfd_R 15 <- 14 there are 2,735 such
    extra survivors (none at depth < 7); exactly four of them, local (622, 154, 180), (619, 154, 181),
    (618, 155, 181), (612, 151, 182) (raw DT exactly 47 each; their partners at z 153..155 are eroded by background
    at z 145..148, outside the 17-deep mirror), add exactly the three residual voxels local (613, 151, 167),
    (615, 154, 167), (614, 155, 167) and nothing else.  A standalone
    /dilation does not show it because its own BorderChange re-fills the margin with background
    (-continuous_at_boundary 0 0 0, the empty border; IPL's help for /close and /dilation documents the
    mirror fill as border mode 1, so it is a documented IPL border mode; /erosion and /open
    have no such flag), and -use_previous_margin true crops exactly the reach of the margin content
    (see dilation).  /close = dilation with a background margin then erosion on that buffer: unchanged, exact.
    VERIFIED (0 mismatching voxels, identical grids): IPL's /open 15 on the full PFJ-411dfd_R stage 14
    (X5143651_P17_14_BBC -> X5143651_P17_15_OPEN15 and the probe-19 re-run X5143651_P19_OPEN15: 17,696,233 =
    17,696,233 set voxels) and on IPL's /sub_get copies suba / subb / subc (SUB?_IN -> SUB?_OPEN15: 7,196,729 /
    17,595,873 / 3,247,584), also with this implementation (probe-19 verification); the 18 oracle-table
    comparisons at 0 (PFJ-0be66a_R / PFJ-42293d_L / PFJ-6f5538_R / PFJ-411dfd_R P17 open
    15 and close 30 / 50, PFJ-0be66a_R P16 open 15 / close 50 / close 15, PFJ-411dfd_R P17 close 15, probe 18 OPEN3 /
    CLOSE3 / ERO3 / ERO1 / DIL15 / DIL1) first with p20_rules.open_ipl and then with
    this function; the 12 erosion / dilation stage oracles
    and the 8 probe-19 primitives at 0 (see erosion, dilation).  DISCRIMINATED on the full PFJ-411dfd_R volume
    (X5143651_P17_14_BBC vs X5143651_P19_OPEN15) and on suba, recomputed with _open_padded on 2026-09-14 evening
    (tests/test_ipl_ops.py pins them) -- mismatch counts
    (ours-only / IPL-only): the pre-mechanism rule ('open_boundary', the two chained primitives) 3 (0 / 3), exactly
    the residual; the mirror margin at depth N + 1 -> 19 (19 / 0), N + 3 -> 2 (0 / 2), N -> 48 (48 / 0), 2N -> 3
    (0 / 3): N + 2 is pinned by the data; the edge-EXCLUSIVE mirror ('reflect') 28 (27 / 1), 9 on suba; edge
    replication 14,423 (14,422 / 1), 8,445 on suba; wrap 76,168 (3,121 / 73,047), 61,036 on suba; margin = object
    28,338 (28,338 / 0), 16,635 on suba; margin = background 238,083 (0 / 238,083), 222,472 on suba.  The z faces
    alone mirrored with OBJECT x / y margins gave 30 (suba) / 540 (full) ours-only in the S3 search; the z faces
    alone with the pre-mechanism x / y conventions (p20_rules 'sym_z') is NOT separated by the /open 15 oracles
    (0 / 0 on full / suba, same log): the six-face mirror is what IPL's isotropic BorderChange implies (the same
    N + 2 margin on all three axes, 'rel beg_pos -17 -17 -17'; the help's description of the mirror mode names
    no axis) and probe 20 SETTLED it directly on 2026-09-15: the transposed copies tzx / tzx0 / tzy / tzy0 put the
    residual geometry on the x1 / x0 / y1 / y0 faces and IPL keeps it there (six-face mirror 0 mismatching voxels,
    'sym_z' 3 voxels each), and cavex -- the plate-over-cave phantom transposed onto the x1 face -- separates the
    two by 138 voxels ('symmetric' 0, 'sym_z' 138, like 'ipldt' / 'reflect' / 'edge').
    THIN VOLUMES (an axis shorter than N + 2): numpy 'symmetric' REPEATS the reflection, and so does IPL -- probe
    20's z156 export ('/sub_get' of 12 slices, 721x290x12 @ 856,24,156, opened with N = 15, i.e. a margin of 17
    against 12 slices = one full reflection plus 5 repeated layers) matches this function at 0 mismatching voxels
    (1,240,454 set voxels) and refutes the three special thin-volume readings -- one reflection completed with
    background 11,803, with object 3, with the far face slice repeated 3 -- as well as 'reflect' 82, 'edge' 15,673,
    'object' 24,740, 'wrap' 13,941 and the pre-mechanism rule 36.  Untested, and the only case left warning
    (ThinVolumeWarning): a margin deeper than TWICE an axis, where the reflection repeats more than once.  Scripts
    32 / 33 never open a volume thinner than 17 slices.
    STATUS: mechanism CONFIRMED by probe 20 on 2026-09-15 (implemented 2026-09-14 evening from the probe-19
    evidence; predictions written 2026-09-14 18:12:42, BEFORE any scanner run, in the probe-20 prediction file (not distributed);
    69 exports; probe-20 run log, not distributed: 'MECHANISM CONFIRMED: IPL matched the
    mechanism's own prediction on all 46 export(s) with readings ...; 12 export(s) separate it from every
    alternative; 11 contrast hypothes(es) refuted, 0 never separated').  The evidence, export by export:
      N-SCAN (open11 / open12 / open13 / open14 / open16 / open17 / open18 on this volume) -- this function matches
        IPL at 0 mismatching voxels at EVERY N, while the pre-mechanism rule leaves 86 / 62 / 34 / 51 / 0 / 0 / 0
        IPL-only voxels, digit for digit the prospective prediction; at N = 11 / 12 the alternatives differ by
        'reflect' 213 / 162, 'edge' 8,018 / 8,491, 'object' 14,981 / 16,617, 'wrap' 54,881 / 61,216, 'background'
        143,717 / 166,886.
      THE CAVE PHANTOMS (cave_open15, cavex_open15; purpose-built, predicted before the run) -- plates of thickness
        T over a background cave with a ring trench in the face slice, the 249-voxel disc inside the trench being
        unrestorable from inside the volume: IPL restores 0 / 0 / 117 / 117 disc voxels at T = 15 / 16 / 17 / 18
        (cave, z1 face) and 0 / 117 at T = 16 / 17 (cavex, x1 face) -- this rule's prediction exactly, where the
        pre-mechanism rule gives 0 / 0 / 0 / 0 and 0 / 0, the edge-exclusive mirror 0 / 0 / 0 / 249 and 0 / 0, an
        'edge' margin 0 / 0 / 0 / 0 and an 'object' margin 249 always.  T = 17 = N + 2 is the first thickness at
        which the truncation of the mirror bites: the T-scan reads the MARGIN DEPTH off the data.
      SIX FACES (cavex_open15 and the transposed crops tzx / tzx0 / tzy / tzy0, which put the residual geometry on
        the x1 / x0 / y1 / y0 face) -- 0 mismatches under the six-face mirror, while a z-faces-only mirror is
        refuted by 138 voxels (cavex) and 3 voxels each (the four crops).  The earlier hedge that the six-face
        claim rested on IPL's isotropic BorderChange rather than on data is RESOLVED BY DATA.
      LOCALITY AND CONTROLS (sreg / z150 / xy3 / x1cut / y1cut, IPL's own /sub_get crops; the k-scan k = 0, 1, 2,
        3, 5, 8, 11 .. 17 of empty slices appended above the region phantom; 14 uploaded phantoms with their round
        trips) -- 0 mismatches on all of them, the pre-mechanism rule off by 3 / 3 / 8 / 30 / 8 on the crops; k = 0
        keeps the 3 residual voxels and every k >= 1 loses them, exactly as an empty slice above the face predicts.
        IPL-vs-IPL: k0_open15 (uploaded phantom) == sreg_open15 (IPL's own crop of the same box) at 0; k0_ero15 and
        k0_dil15 equal erosion() and dilation() at 0 (192,181 and 713,838 set voxels), and IPL's own /open exceeds
        IPL's own chained /erosion + /dilation by EXACTLY the 3 residual voxels at global (1469,175,167),
        (1471,178,167), (1470,179,167) -- the whole effect reproduced inside a 121x101x38 phantom.
      BOUNDARY FLAGS (close15c0 / close15c1 / dil15c1 / dil15c001) -- see close() and dilation(): the flag is obeyed
        per axis in x y z order, 0 = empty border and 1 = the mirror, and 'IPL ignores the flag and always mirrors'
        is refuted by 38,345 voxels.  /open and /erosion carry NO such flag: their border is always the mirror.
    Patella STEP-1 masks from IPL's periosteal: PFJ-411dfd_R was the only non-exact subject (3 voxels), so 21 / 21 are
    expected exact.
    Verified before the mechanism (unchanged by it): 15_open15 <- 14_bbc (20,163,367 set voxels, grid 724x328x168 @
    805,103,168) 0 mismatches on PFJ-0be66a_R; on designed phantoms (probe 18, 2026-09-13, export OPEN3) a 10^3 block
    with 1x1 / 3x3 / 5x5 / 7x7 spikes of length 8 loses the first three and keeps the 7x7 with the corners of the
    3N+2 rule (3N+1 / 3N+3 / Euclidean differ by 137 / 155 / 137 voxels), and the face blocks confirm the open x0 /
    x1 / y1 boundary inside the open.  Probe 17 (2026-09-14): 15_open15 <- 14_bbc 0 mismatches on PFJ-0be66a_R
    (20,163,367 set voxels, the P16 volume again), PFJ-42293d_L (21,006,474) and PFJ-6f5538_R (1,763,947); probe 18 OPEN3
    89,677.
    HISTORY OF THE RESIDUAL (PFJ-411dfd_R, X5143651, 15_open15 <- 14_bbc, grid 721x290x168 @ 856,24,0; the pre-mechanism
    rule = erosion() then dilation(), cropped): IPL keeps 3 voxels that rule removes -- ours 17,696,230, IPL
    17,696,233, ours-only 0, IPL-only 3 at global (x, y, z) = (1469, 175, 167), (1471, 178, 167), (1470, 179, 167) =
    local (613, 151, 167), (615, 154, 167), (614, 155, 167), the LAST slice of the volume.  They are object in 14 and
    eroded (raw chamfer to the background 8 / 15 / 19); their raw dilation distance to the in-volume eroded set is
    47 / 47 / 49 (3N + 2 = 47), and their in-plane neighbours that IPL does NOT restore sit at 47 / 47 / 47 / 47 / 48
    / 48 / 50, so '<=' instead of '<' is refuted on the same slice; their z = 166 counterparts (45 / 44 / 46) are
    restored by both.  The 3 voxels reproduce in the P17 replay and in the September production run (a
    deterministic function of the input) and propagate unchanged through 16..29 (every later command exact in
    isolation) to the 3-voxel CORT_MASK / TRAB_MASK difference of PFJ-411dfd_R.  Proven (probe-17 analysis): under ANY dilation that is a chamfer-3-4-5 ball around a
    seed set, no in-volume seed set reproduces IPL -- every in-volume candidate within 46 of (1470, 179, 167) also
    restores a voxel that is object in 14 and background in IPL's 15, even if the output were ANDed with the input
    (0 admissible seeds), and every candidate within 46 of the background also restores the background voxel that
    realises that distance (a 10-voxel background dimple of the last slice at local x 614..617, y 146..149, 5-8
    voxels away); the only chamfer-ball seed positions restoring exactly the three lie in the z1 MARGIN at depth
    11..15 (13 positions, e.g. local (612, 151, 182)) -- which is where the mechanism puts them.  Cavities touching
    a z face are not the trigger (PFJ-42293d_L's 14 has 748- and 594-voxel cavities at z0 / z1, PFJ-6f5538_R a 5-voxel one
    at z0; both exact).
    REFUTED on PFJ-411dfd_R 15 <- 14 (mismatch counts; 3 = no change from the pre-mechanism rule), kept as history:
    the margin treated as object in the erosion with margin survivors seeding the dilation 28,338 (interior-only
    dilation scans 4,673; margin INF-survivors 7.2 M); leftover raw / scaled erosion DT in the margin as the
    dilation's initial values 6.77 M / 7.19 M; a zero (seed) margin 7.2 M; periodic margins 167,271 (z1 side only
    87,787); mirror / symmetric / edge margins of the ERODED set 3 -- NOTE: those mirrored the eroded set AFTER the
    erosion (filled between the two halves), whereas the mechanism mirrors the INPUT BEFORE the erosion and keeps
    that buffer through the dilation; the difference is exactly the extra survivors; copies of any in-volume eroded
    slice into the margin (none admissible); copies or mirrors of the dilation result or of the input into the
    margin 1,584 .. 279,346 (filled between the halves, not eroded); margin seeds by any raw-DT window in 47..59
    (best 4,821); erosion scans extended 1..17 layers into the margin 5,434 .. 28,338; the last slice keeping 3N+1
    2,265 (with AND 1,230; both faces 8,672 / 6,252); the dimple invisible to the erosion 2,849 (2,839); the last /
    first slice's background not seeded 4,773 / 1,080 / 5,850 / 1,378 / 5,566; every combination of a chamfer pass
    skipping the first or last slice in either half (44 combinations, 162 .. 424,816); every modified weight triple
    for steps into / within / out of the last slice (no improvement); the 47 inexact two-pass variants of another
    raster order (22,895 .. 260,612, all also breaking probe 18 ERO3 / DIL15 / OPEN3); a single wrong-neighbour
    read (x+a, y+b, z') + w at the last slice over a, b in -60..60, z' in 140..184, w in {0, 3, 4, 5} (no solution).
    Probe 19 (2026-09-14; the halves of /open run
    as separate commands on the very input): /erosion 15 on X5143651_P17_14_BBC (export X5143651_P19_ERO15) =
    erosion() exactly (13,640,114 set voxels, 0 mismatches) and /dilation 15 (-use_previous_margin false) on IPL's
    ero15 (DIL15, 753x322x200 @ 840,8,-16) = dilation() exactly (20,637,201, 0 mismatches): THE PRIMITIVES ARE EXACT
    on this input.  /open 15 run again (OPEN15) = the P17 stage-15 export voxel for voxel (deterministic) and differs
    from DIL15 inside the 14 grid by EXACTLY the 3 residual voxels (open15 has them, dil15 does not): /open is NOT
    /erosion followed by /dilation as separate commands -- the residual lives inside /open's internal chaining.  The
    log agrees (probe 19): /open reports ONE border change (the erosion's, 17 voxels on every face, out dim
    755 324 202), the four chamfer DT steps and a grey-scale threshold of 16, then the dilation's inversion with
    NO border change, the DT steps again and the same threshold (upper bound 32767 where the standalone
    /dilation reports 10000000), and a final back-inversion;
    CPU 13.01 s against 7.04 + 7.97 s for the halves.  A dilation that reuses the
    erosion's margin (-use_previous_margin true, DIL15PM on ERO15B) says nothing about the residual: its written
    grid is the input grid shrunk by 16 per side (689x258x136 @ 872,40,16, z = 167 outside) and its content equals
    DIL15 and OPEN15 there (see dilation).  The residual is LOCAL: on IPL's own /sub_get copies of 14 (suba = z
    100..167, 721x290x68 @ 856,24,100; subc = x 400..720, y 40..289, z 100..167, 321x250x68 @ 1256,64,100) IPL's
    /open 15 keeps the SAME 3 voxels at the same global positions (7,196,729 vs the pre-mechanism rule's 7,196,726;
    3,247,584 vs 3,247,581; the sub erosions and dilations all exact), and on subb (z 0..166, the last slice
    removed) /open 15 = the pre-mechanism rule exactly (17,595,873 set voxels): the residual follows the last-slice
    content, not the volume's dimensions or position.  /open's behaviour at cut faces: IPL's full open15 restricted
    to a sub grid differs from IPL's sub open by 2,600 / 4,412 / 719 voxels (suba / subb / subc), all within 11 /
    20 / 11 voxels of a cut face, and open_(full) restricted vs open_(sub) differs by the identical 2,600 / 4,412 /
    719.  Admissible chamfer-ball seeds: without masking, 13 z1-margin positions at
    depth 11..15 (e.g. local (626, 153..156, 178), (612, 151, 182)); with the output ANDed with 14, additionally one
    ray per target voxel at every depth 1..10; the margin survivors of the erosion-phase DT selected by raw value
    47..60 or by depth 11..17 all over-restore by 924 .. 28,279 voxels -- the true seed set is the four extra
    survivors above.
    REFUTED (PFJ-0be66a): a background-padded erosion inside the open 233,009 IPL-only voxels; thresholds 3N+1 / 3N+3
    10,616 / 13,797."""
    n = _distance("open", n)
    m = n + 2
    mask = _set(v)
    thin = _mirror_repeat_axes(mask.shape, m, ("symmetric",) * 3)
    if thin:
        _warn_thin_volume("open_", v["dim"], m, thin)
    return mask_vol(_open_padded(mask, n, m, "symmetric"), v["dim"], v["pos"])


# ============================================================================================== gobj peel mask
def peel_gobj_render(gobj_render, peel_iter):
    """The mask M of /gobj_maskaimpeel_ow: IPL's rendering of the gobj (/gobj_to_aim -peel_iter 0, e.g. the
    P16 stage 00) eroded slice-wise with 4-connectivity peel_iter times (ipldt.core.peel_gobj; 0 = the
    rendering itself), as a char volume on the rendering's grid.  |M| is what IPL prints as 'Set N'
    (PFJ-0be66a: 27,484,219 for peel 0, 25,892,693 for peel 6)."""
    g = _set(gobj_render)
    p = peel_gobj(g, int(peel_iter)) if int(peel_iter) > 0 else g
    return mask_vol(p, gobj_render["dim"], gobj_render["pos"])


def mask_by_gobj(v, gobj_mask):
    """out = in where the (peeled) gobj mask is set, 0 elsewhere; the mask is pasted by GLOBAL position into
    the input's grid, so slices of the input outside the gobj's z-range are cleared ('Sli k CLEARED').
    Grid and data type unchanged (works on the short greyscale as well as on char masks)."""
    m = on_grid(gobj_mask, v["dim"], v["pos"]) != 0
    data = np.asarray(v["data"])
    return vol(np.where(m, data, np.zeros((), data.dtype)), v["dim"], v["pos"])


def gobj_maskaimpeel_ow(v, gobj_render, peel_iter):
    """/gobj_maskaimpeel_ow -input_output v -gobj_filename G -peel_iter n: v AND peel(G, n) by global position.
    Verified: 03 <- 02 (peel 6), 06 <- 05 (peel 0), 13 <- 12 and 24 <- 23 (peel 6 on the larger 751x356x176
    and 732x336x176 grids) all 0 mismatches."""
    return mask_by_gobj(v, peel_gobj_render(gobj_render, peel_iter))


# ============================================================================================== components
def label6(mask_bool):
    """6-connected 3-D labelling: (labels, n, sizes) with sizes[0] = 0."""
    lab, n = ndi.label(np.asarray(mask_bool, dtype=bool), structure=_S6)
    cnt = np.bincount(lab.ravel(), minlength=n + 1)
    cnt[0] = 0
    return lab, n, cnt


def component_sizes(v):
    """Sizes of the 6-connected components of a volume, largest first (for log cross-checks: IPL prints the
    label count and the five largest; PFJ-0be66a stage 04: 2145 components, 23,425,237 4,046 3,558 3,461 3,412)."""
    _, _, cnt = label6(_set(v))
    return np.sort(cnt[1:])[::-1]


@njit(cache=True)
def _shell_sort_desc(key):
    """Shell sort of `key` (int64 array) into DESCENDING order with the K&R gap sequence n/2, n/4, ..., 1
    (gap-insertion: an element moves ahead of the elements strictly smaller than it); returns the
    permutation (original indices in sorted order).  NOT stable: equal keys end up in an order that depends
    on the whole array -- that is the point, see rank_order."""
    n = key.size
    k = key.copy()
    idx = np.arange(n)
    gap = n // 2
    while gap > 0:
        for j in range(gap, n):
            tk = k[j]
            ti = idx[j]
            i = j - gap
            while i >= 0 and k[i] < tk:
                k[i + gap] = k[i]
                idx[i + gap] = idx[i]
                i -= gap
            k[i + gap] = tk
            idx[i + gap] = ti
        gap //= 2
    return idx


def rank_order(sizes):
    """IPL's rank order of a component size table: rank 1 first.  `sizes[i]` is the size of label i + 1 in
    LABEL ORDER (the order in which a raster scan x fastest, then y, then z, all increasing, first meets the
    components = scipy.ndimage.label's order); returns the 0-based indices in rank order.
    RULE: the size table is sorted largest-first by an UNSTABLE shell sort (K&R gaps n/2, n/4, ..., 1), so
    two equal sizes are ordered by the sort's data-dependent moves, not by their labels.
    Basis (probe 18, 2026-09-13; exports X2420448_P18_RANK1..5 on a 41-component phantom): the two 20,000-voxel
    components T1 (label 19) and T2 (label 22) came out rank 1 = T2, rank 2 = T1 (the LATER label first),
    while the two 10,000-voxel components b (label 21) and c (label 23) came out rank 4 = b, rank 5 = c (the
    EARLIER label first); rank 3 = a (15,000).  No monotone rule on the label order (lower first, higher first,
    dense ranks) explains both, in either z direction, in any of the 48 signed axis permutations of a
    first-voxel order; the raster label order itself is pinned by the log's fingerprint ('Number of labels in
    first scan: 43', '74 equivalence entries' = the two spike / block interfaces of the op3 block, 3 labels /
    30 entries on the bordered sub-volume).  Of 42 sorting algorithms simulated on the 41-entry table exactly
    four reproduce all five ranks: shell sort descending with K&R gaps (implemented), with the Numerical
    Recipes 1st-edition log2 gaps (the same gaps 20, 10, 5, 2, 1 for n = 41) and with the NR 2nd-edition
    3h + 1 gaps, and a Bentley-McIlroy qsort descending; a stable sort, selection sort, heapsort and the
    other quicksorts put T1 first.  RESIDUAL AMBIGUITY, stated: the four fitting sorts can differ on other
    size tables, and IPL's first-scan label numbering on concave shapes (bone) is not pinned (the one-voxel
    lookahead scan that reproduces the phantom's 43 / 74 and the 2-D slice's 104 labels is 9-20 % high on
    the P16 bone stages), so a tie between equal-sized components of a bone volume may be ordered
    differently by IPL.  Script 32 / 33 extract rank 1..1 only, which this affects only when the two largest
    components have exactly the same size (PFJ-0be66a: 23,425,237 vs 4,046 at stage 04)."""
    s = np.ascontiguousarray(np.asarray(sizes), dtype=np.int64)
    return _shell_sort_desc(s)


def _join_boundary_components(lab, n, cnt):
    """-connect_boundary true: every component touching any of the six faces of the volume becomes ONE
    component that takes label 1 (IPL's log: 'Boundary marked with label 1'); the other components keep
    their relative (raster) order as labels 2, 3, ...  Returns (lut old label -> new label, new n, new
    sizes) without relabelling the volume.  When no component touches a face, label 1 is an empty entry
    (size 0; its effect on the tie order of the others is unobserved)."""
    faces = (lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1])
    touch = np.zeros(n + 1, bool)
    for f in faces:
        touch[np.unique(f)] = True
    touch[0] = False
    lut = np.zeros(n + 1, np.int64)
    lut[touch] = 1
    rest = np.nonzero(~touch[1:])[0] + 1
    lut[rest] = np.arange(2, 2 + rest.size)
    new_n = 1 + rest.size
    new_cnt = np.bincount(lut, weights=cnt, minlength=new_n + 1).astype(np.int64)
    new_cnt[0] = 0
    return lut, new_n, new_cnt


def cl_ow_rank_extract(v, first_rank=1, last_rank=1, connect_boundary=False, value_in_range=SET, topology=6):
    """/cl_ow_rank_extract: label the set voxels with 6-connectivity, rank the components by size (rank 1 =
    largest; ties by IPL's unstable sort, see rank_order), keep ranks first_rank..last_rank at
    value_in_range; grid unchanged.
    connect_boundary false (Script 32 / 33): components are NOT joined through the grid border (in stage 09,
    168 non-largest components touching the border were discarded).  connect_boundary true: EVERY component
    touching any of the six faces of the volume is joined into one component, ranked with the others (the
    joined component is label 1, IPL's 'Boundary marked with label 1'); verified on designed phantoms (probe
    18, 2026-09-13): on a 743x348x20 sub-volume with B1 (2,000 voxels) and B2 (200) touching its first slice,
    B4 (300) its last slice and B3 (1,500) interior, rank 1 = B1 + B2 + B4 (2,500 voxels, export
    X2420448_P18_CBT1) and rank 2 = B3 (CBT2); the readings 'no effect' (B1), 'same face only' (B1 + B2) and
    'face-touching components excluded' (B3 first) are refuted; the connect_boundary false control gave B1
    (CBF1).  The joined component's size is its object voxels only: IPL's log prints it as 'Label 1: 62.5000 %
    (2500)' of the sub-volume's 4,000 set voxels (the non-object face voxels are not counted), as here.
    Only topology 6 is implemented.  Ranks are 1-based (IPL convention): first_rank < 1 or last_rank <
    first_rank raises ValueError (a 0-based caller would otherwise silently get an empty mask); ranks beyond
    the component count give an empty mask.
    Verified: 05 <- 04, 08 <- 07, 10 <- 09 all 0 mismatches; component counts 2145 / 550 / 1695 and the five
    largest sizes equal the log; probe 18 RANK1..RANK5 (T2, T1, a, b, c) 0 mismatches each."""
    if topology != 6:
        raise AssertionError("cl_ow_rank_extract: only -topology 6 was observed and is implemented")
    fr, lr = int(first_rank), int(last_rank)
    if fr < 1 or lr < fr:
        raise ValueError("cl_ow_rank_extract: ranks are 1-based and need 1 <= first_rank <= last_rank "
                         f"(got {fr}, {lr})")
    lab, n, cnt = label6(_set(v))
    lut = None
    if connect_boundary:
        lut, n, cnt = _join_boundary_components(lab, n, cnt)
    order = rank_order(cnt[1:]) + 1                      # labels in rank order
    keep = np.zeros(n + 1, bool)
    keep[order[fr - 1:lr]] = True
    keep[0] = False
    if lut is not None:
        keep = keep[lut]                                 # back to the raster labels of `lab`
    return mask_vol(keep[lab], v["dim"], v["pos"], value_in_range)


def cl_nr_extract(v, min_number=1, max_number=0, value_in_range=SET, topology=6):
    """/cl_nr_extract: keep the 6-connected components with min_number <= size <= max_number (max_number 0 =
    no upper limit) at value_in_range, BOTH BOUNDS INCLUSIVE; grid unchanged.
    Verified: 18 <- 17 (min 200000: 64 components, largest 3,500 -> empty) and 21 <- 20 (1..500000 on an
    empty volume) 0 mismatches; the inclusive bounds on designed phantoms (probe 18, 2026-09-13): components
    of exactly 799, 800 and 801 voxels, -min_number 800 keeps 800 and 801 (export NRMIN), -max_number 800
    keeps 799 and 800 (NRMAX); the exclusive readings are refuted."""
    if topology != 6:
        raise AssertionError("cl_nr_extract: only -topology 6 was observed and is implemented")
    lab, n, cnt = label6(_set(v))
    keep = cnt >= int(min_number)
    if int(max_number) > 0:
        keep &= cnt <= int(max_number)
    keep[0] = False
    return mask_vol(keep[lab], v["dim"], v["pos"], value_in_range)


def cl_slicewise_extractow(v, lo_vol_fract_in_perc=50.0, up_vol_fract_in_perc=100.0, value_in_range=SET,
                           topology=6):
    """/cl_slicewise_extractow: for every slice independently, label the set voxels with 4-CONNECTIVITY
    (topology 6 restricted to the slice) and keep the components whose 100 * size / (SET VOXELS OF THAT SLICE)
    lies in [lo, up], INCLUSIVE at both ends; a slice in which no component qualifies is cleared ENTIRELY
    (e.g. fragments of 40 / 35 / 25 % at lo 50); empty slices stay empty; grid unchanged.
    Verified on designed phantoms (probe 18, 2026-09-13, export X2420448_P18_SW, lo 50 / up 100): a 60 / 40
    slice keeps the 60 only (the 'fraction of the largest component' reading, which keeps both, is refuted),
    40 / 35 / 25 and 45 / 30 / 25 slices are cleared, a 50 / 50 slice keeps both (the tie at exactly lo is
    inclusive), and 60 / 40 and 70 / 20 blocks touching at one diagonal only are two components (4-connected;
    8-connectivity refuted).  Basis for the denominator before the probe: the help text calls the bounds
    'Lower/Upper volume fraction [%]'; the per-slice histogram this command prints ('Label k: p % (n)') equals
    100 * n / (set voxels of the slice) in all 115 multi-component printouts of the cohort EVAL logs (e.g.
    'Label 1: 49.9871 % (27081)' of 54176) and never 100 * n / largest.
    Verified: 25 <- 24 and 27 <- 26 0 mismatches (both no-ops on PFJ-0be66a: every slice has one component); all
    4016 CORT_MASK slices of the 21 subjects have exactly one 4-connected component; X9463122's EVAL log shows
    the trab_close call removing a 75-voxel island (0.1086 % of the slice, removed under either reading)."""
    if topology != 6:
        raise AssertionError("cl_slicewise_extractow: only -topology 6 was observed and is implemented")
    m = _set(v)
    out = np.zeros(m.shape, np.uint8)
    for z in range(m.shape[0]):
        sl = m[z]
        tot = int(sl.sum())
        if tot == 0:
            continue
        lab, n = ndi.label(sl, structure=_S4)
        cnt = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
        frac = 100.0 * cnt / tot
        keep = (frac >= float(lo_vol_fract_in_perc)) & (frac <= float(up_vol_fract_in_perc))
        keep[0] = False
        out[z] = np.where(keep[lab], np.uint8(value_in_range), np.uint8(0))
    return vol(out, v["dim"], v["pos"])


# ============================================================================================== bookkeeping
CHAR_MIN, CHAR_MAX = -SET, SET                  # the value range of IPL's char arithmetic, [-127, 127]


def _char_operands(name, v1, v2):
    """subtract_aims / add_aims reproduce IPL's char arithmetic only (int8 output in [-127, 127])."""
    for k, v in (("input1", v1), ("input2", v2)):
        d = np.asarray(v["data"])
        if d.dtype not in (np.uint8, np.int8, np.bool_):
            raise TypeError(f"{name}: {k} is {d.dtype}; only char volumes (uint8 / int8 / bool) are "
                            f"supported -- IPL's arithmetic on a short greyscale was never observed")


def _char_arith(v1, v2, ufunc):
    """in1 (op) in2 on the union grid in int16, saturated to [-127, 127], returned as int8 (IPL's char is
    signed: the byte 0x81 of its export is -127).  Transient memory: one int16 copy of the union grid."""
    dim, pos = union_grid(v1, v2)
    a = on_grid(v1, dim, pos).astype(np.int16)                  # one int16 copy of the union grid
    ufunc(a, on_grid(v2, dim, pos), out=a)                      # char (op) char fits int16 exactly
    np.clip(a, CHAR_MIN, CHAR_MAX, out=a)
    return vol(a.astype(np.int8), dim, pos)


def subtract_aims(v1, v2):
    """/subtract_aims: out = in1 - in2 on the UNION bounding box of the two grids (pos = min, end = max),
    both pasted by global position, as CHAR ARITHMETIC: for 0 / 127 masks the result is 127 where only in1
    is set, 0 where both or neither are, and -127 where ONLY in2 is set.  Output dtype int8 (the signed
    reading of the char byte IPL writes: 0x81 = -127, which reads as 129 unsigned), values saturated to
    [-127, 127] (the lower bound is the implemented, unobserved choice; only -127, 0 and 127 were observed).
    A -127 voxel IS set (non-zero): every consumer in ipldt tests set-ness as != 0, and IPL's own
    -compress_type bin export of such a volume stores it as set (P16 / P17 stage 02 are bin exports).
    Char volumes only (a short operand raises TypeError).
    Verified: 02, 07, 16, 26 (grid 743x348x176 @ 795,94,164 = the union) and 29 of PFJ-0be66a all 0 mismatches
    (in2 is a subset of in1 there); the negative case on designed phantoms (probe 18, 2026-09-13): a 20x20x10
    block A1 minus an all-127 box b overlapping its corner (export X2420448_P18_SUB, written -compress_type
    none: 3,000 voxels 127, 1,000 voxels 0, 3,000 voxels byte 0x81 on the union grid 743x348x10 @ 795,94,268)
    and b - A1 (SUB2, the same histogram) -- the former 'clip to 0' reading is REFUTED by 3,000 voxels in each.
    Script 32: in2 is a subset of in1 by construction for 07 (06 = 05 AND peel0 of 00), 16 (15 = open of 14,
    anti-extensive), 26 and 29 (subsets of 00), but NOT for 02 = 00 - 01: seg_gauss reaches 3 voxels outside
    the contour and dense cortex at the contour edge exceeds the lower threshold there on 19 of the 21 cohort
    subjects (56,833 voxels in total, 1 .. 18,436 per subject; PFJ-0be66a_R and PFJ-411dfd_L are the only two with 0),
    so stage 02 carries that many -127 voxels (18,436 on PFJ-6f5538_R, 591 on PFJ-411dfd_R); 03 = 02 AND peel6(00)
    removes every voxel outside 00, so the products are unaffected (CORT / TRAB masks identical to the
    former clip on all 21 subjects), and ipldt.step1 reports them in info['negatives'].
    Transient memory: one int16 copy of the union grid (plus the int8 result)."""
    _char_operands("subtract_aims", v1, v2)
    return _char_arith(v1, v2, np.subtract)


def add_aims(v1, v2):
    """/add_aims: out = in1 + in2 on the union grid as char arithmetic, SATURATED at 127 (127 + 127 = 127;
    the wrap-around reading, byte 0xFE, is refuted), int8 output like subtract_aims (a negative operand,
    unobserved, saturates at -127).  Char volumes only (a short operand raises TypeError).
    Verified: 22 <- 21 + 15 (grid 732x336x176 @ 801,99,164) 0 mismatches (disjoint inputs); the saturation on
    designed phantoms (probe 18, 2026-09-13, export X2420448_P18_ADD written -compress_type none: A1 + b with a
    1,000-voxel overlap -> 7,000 voxels of 127, none of 254).
    Transient memory: one int16 copy of the union grid (plus the int8 result)."""
    _char_operands("add_aims", v1, v2)
    return _char_arith(v1, v2, np.add)


def set_value(v, value_object, value_background):
    """/set_value (the script's '/set aim obj bg'): non-zero -> value_object, zero -> value_background; grid
    unchanged.  '/set trab 0 127' inverts a mask (stage 04 <- 03, 0 mismatches)."""
    out = np.where(_set(v), np.uint8(value_object), np.uint8(value_background))
    return vol(out, v["dim"], v["pos"])


def bounding_box_cut(v, border=(0, 0, 0), z_only=False):
    """/bounding_box_cut -border bx by bz -z_only: the tight box of the non-zero voxels (dim = extent per axis,
    pos = pos + first non-zero index), grown by `border` and clipped to the input grid.  border != 0 and
    z_only true are unobserved.  An empty input is returned unchanged (unobserved in IPL; relied upon by
    step1 stage 14 when the trabecular branch is empty).
    Verified: 14 <- 13 (751x356x176 -> 724x328x168 @ 805,103,168) and 28 <- 27 (-> 738x343x168 @ 798,97,168)
    0 mismatches; the aim_bbc of the masked greyscale equals IPL's 738x343x168 @ 798,97,168."""
    data = np.asarray(v["data"])
    nz = np.nonzero(data)
    if nz[0].size == 0:
        return vol(data, v["dim"], v["pos"])
    lo = [int(nz[2].min()), int(nz[1].min()), int(nz[0].min())]
    hi = [int(nz[2].max()) + 1, int(nz[1].max()) + 1, int(nz[0].max()) + 1]
    if z_only:
        lo[0], lo[1], hi[0], hi[1] = 0, 0, v["dim"][0], v["dim"][1]
    lo = [max(0, lo[i] - int(border[i])) for i in range(3)]
    hi = [min(int(v["dim"][i]), hi[i] + int(border[i])) for i in range(3)]
    out = data[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    return vol(out, tuple(hi[i] - lo[i] for i in range(3)), tuple(v["pos"][i] + lo[i] for i in range(3)))
