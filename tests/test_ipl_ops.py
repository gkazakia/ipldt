"""ipldt.ipl_ops on small synthetic volumes: the chamfer 3-4-5 threshold 3N+2, the open-boundary erosion,
the dilation's grid growth (N+1 per side) and reach (N voxels), close / open on the input grid, the
/seg_gauss float32 weights, per-pass truncation (seeds 2 / 47 pin the float32 centre-out pair accumulation),
inclusive bounds and 'valid' output grid, the slicewise fraction-of-total rule (all-cleared slice, 50 % tie),
subtract / add on the union grid (char arithmetic: -127 where only input2 is set, saturation at 127),
bounding_box_cut, the component / peel / set_value commands (the rank tie-break of probe 18 on the phantom's
own 41-entry size table, -connect_boundary true), the by-position (never by-shape) paste of on_grid and the
parameter contracts (1-based ranks, integral native thresholds, support >= 1, distances >= 0).  Every
expectation is stated independently of the implementation.
Fast, added after probe 20 (2026-09-15): -continuous_at_boundary on dilation / close -- the default (0, 0, 0) is
the empty border of every existing call site, an int is the same flag on all three axes, and the flag is per axis
in IPL's x, y, z order (a block on the x1 face moves by 77 voxels under (1, 0, 0) only, the same block on the z1
face under (0, 0, 1) only; a 5x5 pit 2 deep in a face closes 24 of its 50 voxels with an empty border and all 50
with the mirror) -- checked against the chamfer ball itself, not against the distance transform.
Slow (needs the probe-18 folder): every probe-18 export of IPL (2026-09-13) reproduced whole-volume, grid
included, from the uploaded phantom.  The /open mechanism (ipl_ops.open_, implemented 2026-09-14 evening: the erosion's mirror-padded buffer reused by the
dilation half) on the plate-over-cave phantom of probe 20 (T = 15 / 16 / 17 / 18 restore 0 / 0 / 117 / 117 disc
voxels; the pre-mechanism rule 0 / 0 / 0 / 0; the edge-exclusive mirror 0 / 0 / 0 / 249), its equality with the
pre-mechanism rule where no extra margin survivor exists, the thin-volume warning and the helper's contracts.
Slow (needs the probe-17 folder): the open / close oracles of probe 17 (2026-09-14; open 15 on four subjects, close
15 / 30 / 50), PFJ-411dfd_R 15_open15 now exact (17,696,233 = 17,696,233), and the discrimination of the mechanism on
that volume: the pre-mechanism rule differs from IPL by exactly the 3 last-slice voxels, the mirror margin at depth
N + 1 / N + 3 / N by 19 / 2 / 48, the edge-exclusive mirror by 28, edge replication / wrap / object / background by
14,423 / 76,168 / 28,338 / 238,083 (so a change of the rule or of the margin depth is deliberate).
Slow (needs the probe-19 folder): the halves of /open 15 on that very input (2026-09-14): /erosion 15 and /dilation 15
as separate commands equal erosion / dilation exactly (the dilation's grid growth N + 1 = 16 per side included),
/open 15 is deterministic, equals open_ (the mechanism) exactly and differs from the chained primitives (the
pre-mechanism rule) by exactly the 3 residual voxels, the mechanism reproduces IPL's /open on the three /sub_get
copies (the residual persists on the two that keep the last slice and is absent on the one without it), /open's
cut-face behaviour equals the rule's (identical locality counts), and -use_previous_margin true writes the shrunk
grid.
Slow (needs the probe-20 folder): the prospective confirmation of the mechanism (predictions written 2026-09-14
18:12:42, scanner run fetched 2026-09-15).  The N-scan /open 11 / 12 / 13 / 14 / 16 / 17 / 18 (0 mismatches at every
N; the pre-mechanism rule 86 / 62 / 34 / 51 / 0 / 0 / 0 IPL-only voxels; reflect / edge / object / wrap / background
213 / 8,018 / 14,981 / 54,881 / 143,717 at N = 11), IPL's six /sub_get crops including the 12-slice z156 that settles
the repeated reflection, the boundary-flag oracles (close15c0 / close15c1 / dil15c1 / dil15c001 at 0; 'IPL ignores the
flag' refuted by 38,345, a z y x reading of the flags by 598,104, the edge-exclusive mirror by 70,802), the cave
T-scan (0 / 0 / 117 / 117 and 0 / 117 disc voxels, the discs located by labelling the phantom's own face slice), the
six faces (a z-only mirror refuted by 138 and by 3), the 14 phantoms with their round trips, and the k-scan with IPL's
own erosion / dilation inside the region phantom."""
import os
import warnings

import numpy as np
import pytest
from conftest import lab_path, vms_versions
from scipy import ndimage as ndi

from ipldt import ipl_ops as ops
from ipldt.io import read_aim


def V(data, pos=(0, 0, 0)):
    data = np.asarray(data)
    return ops.vol(data, data.shape[::-1], pos)


def chamfer_metric(dz, dy, dx):
    """The exact chamfer 3-4-5 distance of an offset (what the two-pass transform computes)."""
    a, b, c = sorted((abs(dz), abs(dy), abs(dx)), reverse=True)
    return 5 * c + 4 * (b - c) + 3 * (a - b)


def offsets_from(shape, centre):
    zz, yy, xx = np.indices(shape)
    return zz - centre[0], yy - centre[1], xx - centre[2]


# ------------------------------------------------------------------------------------------- chamfer / erosion
def test_chamfer_345_of_a_single_hole_is_the_exact_chamfer_metric():
    obj = np.ones((9, 9, 9), np.uint8)
    obj[4, 4, 4] = 0
    d = ops.chamfer_dt_345(obj)
    dz, dy, dx = offsets_from(obj.shape, (4, 4, 4))
    ref = np.vectorize(chamfer_metric)(dz, dy, dx)
    assert d[4, 4, 4] == 0
    assert d[4, 4, 3] == 3 and d[4, 3, 3] == 4 and d[3, 3, 3] == 5
    assert d[0, 0, 0] == 20                      # open boundary: the array corner sees only the hole
    assert np.array_equal(d, ref)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_erosion_keeps_exactly_raw_chamfer_at_least_3n_plus_2(n):
    """All-object volume with one background voxel: erosion N removes the voxels with chamfer < 3N+2
    (N = 1: the 6 face and 12 edge neighbours, raw 3 and 4; the 8 corner neighbours at raw 5 stay).
    The volume faces are open, so nothing is eroded from the outside."""
    obj = np.ones((11, 11, 11), np.uint8)
    obj[5, 5, 5] = 0
    out = ops.erosion(V(obj), n)
    dz, dy, dx = offsets_from(obj.shape, (5, 5, 5))
    ref = np.vectorize(chamfer_metric)(dz, dy, dx) >= 3 * n + 2
    assert out["dim"] == (11, 11, 11) and out["pos"] == (0, 0, 0)
    assert out["data"].dtype == np.uint8 and set(np.unique(out["data"])) <= {0, 127}
    assert np.array_equal(out["data"] != 0, ref)
    if n == 1:
        assert int((obj != 0).sum() - (out["data"] != 0).sum()) == 18
    assert (out["data"][0] != 0).all() and (out["data"][:, :, -1] != 0).all()


def test_erosion_open_boundary_on_a_block_touching_a_face():
    """A block touching the x = 0 face: erosion 2 shrinks it by 2 on every side that faces in-volume
    background but not at the face (a background-padded erosion would remove x = 0 and x = 1)."""
    m = np.zeros((12, 12, 12), bool)
    m[3:10, 3:10, 0:8] = True
    out = ops.erosion(V(m), 2)["data"] != 0
    exp = np.zeros_like(m)
    exp[5:8, 5:8, 0:6] = True
    assert np.array_equal(out, exp)
    assert out[5:8, 5:8, 0].all()


# ------------------------------------------------------------------------------------------- dilation / close / open
@pytest.mark.parametrize("n", [1, 2, 3])
def test_dilation_grows_the_grid_by_n_plus_1_and_reaches_n_voxels(n):
    m = np.zeros((5, 5, 5), bool)
    m[2, 2, 2] = True
    out = ops.dilation(ops.mask_vol(m, (5, 5, 5), (10, 20, 30)), n)
    assert out["dim"] == (5 + 2 * (n + 1),) * 3
    assert out["pos"] == (10 - (n + 1), 20 - (n + 1), 30 - (n + 1))
    c = 2 + n + 1                                   # the seed in the grown grid
    dz, dy, dx = offsets_from(out["data"].shape, (c, c, c))
    ref = np.vectorize(chamfer_metric)(dz, dy, dx) < 3 * n + 2
    assert np.array_equal(out["data"] != 0, ref)
    assert out["data"][c, c, c + n] != 0 and out["data"][c, c, c - n] != 0
    assert out["data"][c, c, c + n + 1] == 0 and out["data"][c + n + 1, c, c] == 0


# --------------------------------------------------------- -continuous_at_boundary (probe 20's flag, synthetic)
def chamfer_ball(n):
    """The structuring element of the metric-11 dilation: every offset whose EXACT chamfer 3-4-5 distance is
    < 3N + 2.  'out = in | (dt(~in) < 3N + 2)' is exactly the binary dilation by this ball and 'keep <=> dt(in) >=
    3N + 2' is exactly 'not in the dilation of the complement by this ball' -- reference formulations that use no
    distance transform, so the margin conventions below are checked against the metric, not against ipl_ops."""
    r = n + 1
    dz, dy, dx = np.indices((2 * r + 1,) * 3) - r
    return np.vectorize(chamfer_metric)(dz, dy, dx) < 3 * n + 2


def edge_inclusive_mirror(A, ax, margin):
    """A mirrored into a margin of `margin` on both faces of axis `ax`, EDGE-INCLUSIVE and written with explicit
    indices (not np.pad, so the convention is the test's own statement): margin depth k holds the slice at distance
    k - 1 from the face, i.e. depth 1 duplicates the face slice.  Needs margin <= A.shape[ax] (one reflection)."""
    d = A.shape[ax]
    assert margin <= d
    lo = np.take(A, np.arange(margin - 1, -1, -1), axis=ax)           # depth margin .. 1 -> slices margin-1 .. 0
    hi = np.take(A, np.arange(d - 1, d - margin - 1, -1), axis=ax)    # depth 1 .. margin -> slices d-1 .. d-margin
    return np.concatenate([lo, A, hi], axis=ax)


def ref_buffer(M, margin, flags_xyz):
    """The operand padded by `margin` on all six faces: an EMPTY (background) border on an axis whose IPL flag
    -continuous_at_boundary is 0, the edge-inclusive mirror where it is 1; flags in IPL's own x, y, z order."""
    out = np.asarray(M, bool)
    for ax, f in enumerate(tuple(flags_xyz)[::-1]):                   # (x, y, z) -> the array's (z, y, x)
        if f:
            out = edge_inclusive_mirror(out, ax, margin)
        else:
            pw = [(0, 0)] * 3
            pw[ax] = (margin, margin)
            out = np.pad(out, pw, constant_values=False)
    return out


def ref_dilation(M, n, flags_xyz):
    """/dilation N under those flags: the buffer dilated by the chamfer ball, on the grid grown N + 1 per side."""
    return ndi.binary_dilation(ref_buffer(M, n + 2, flags_xyz), chamfer_ball(n))[1:-1, 1:-1, 1:-1]


def ref_close(M, n, flags_xyz):
    """/close N under those flags: that dilation, then the erosion on the SAME buffer, cropped to the input grid."""
    B, m = chamfer_ball(n), n + 2
    d = ndi.binary_dilation(ref_buffer(M, n + 2, flags_xyz), B)
    return (d & ~ndi.binary_dilation(~d, B))[m:-m, m:-m, m:-m]


def test_continuous_at_boundary_is_per_axis_in_x_y_z_order_and_defaults_to_the_empty_border():
    """The flag added after probe 20 (2026-09-15), on volumes small enough to check against the chamfer metric
    itself.  (1) DEFAULT (0, 0, 0) = today's behaviour: an empty border on all six faces, identical to the
    no-argument call -- Scripts 32 / 33 pass 0 0 0, so no call site moves.  (2) The flag is PER AXIS and is read in
    IPL's x, y, z order: a 3x3x5 block touching only the x1 face of a 13^3 volume (every other face is 5 voxels
    away, deeper than the margin N + 2 = 4, so its mirror is background) changes by exactly 77 voxels under
    (1, 0, 0) and not at all under (0, 1, 0) or (0, 0, 1), and the same block on the z1 face changes by exactly 77
    under (0, 0, 1) and not at all under (1, 0, 0) or (0, 1, 0) -- reading the tuple in z, y, x order would swap
    the two.  (3) An int means the same flag on all three axes.  (4) /close: a 5x5 pit 2 deep in the z1 face of a
    slab has 50 background voxels; the empty border closes 24 of them and the mirrored z border -- which continues
    the slab above the face -- closes all 50, a difference of exactly 26 voxels, again on the z axis only.
    (5) /erosion and /open take NO such flag (IPL's help gives it only for /close and /dilation; their border is
    always the mirror), and a flag that is not 0 or 1 raises ValueError."""
    n = 2
    for flags, other in (((1, 0, 0), ((0, 1, 0), (0, 0, 1))), ((0, 0, 1), ((1, 0, 0), (0, 1, 0)))):
        M = np.zeros((13, 13, 13), bool)
        if flags == (1, 0, 0):
            M[5:8, 5:8, 8:13] = True                                  # touches the x1 face only
        else:
            M[8:13, 5:8, 5:8] = True                                  # touches the z1 face only
        assert int(M.sum()) == 45
        v = ops.mask_vol(M, (13, 13, 13), (5, 6, 7))
        base = ops.dilation(v, n)
        assert base["dim"] == (19, 19, 19) and base["pos"] == (2, 3, 4)          # grown by N + 1 per side
        b = base["data"] != 0
        assert np.array_equal(b, ref_dilation(M, n, (0, 0, 0))) and int(b.sum()) == 341
        assert np.array_equal(b, ops.dilation(v, n, (0, 0, 0))["data"] != 0)
        assert np.array_equal(b, ops.dilation(v, n, 0)["data"] != 0)
        mirrored = ops.dilation(v, n, flags)["data"] != 0
        assert np.array_equal(mirrored, ref_dilation(M, n, flags))
        assert int((mirrored != b).sum()) == 77 and int(mirrored.sum()) == 418
        assert not (b & ~mirrored).any()                                          # the mirror only adds
        for f in other:
            assert np.array_equal(ops.dilation(v, n, f)["data"] != 0, b), (flags, f)
        allm = ops.dilation(v, n, (1, 1, 1))["data"] != 0
        assert np.array_equal(allm, mirrored)                                     # only this axis's mirror bites
        assert np.array_equal(allm, ops.dilation(v, n, 1)["data"] != 0)           # an int = the same flag on x y z
        assert np.array_equal(allm, ref_dilation(M, n, (1, 1, 1)))
    # (4) /close: the pit in the z1 face
    M = np.zeros((13, 17, 17), bool)
    M[9:13, 4:13, 4:13] = True                                        # a slab on the z1 face, 4 from the x / y faces
    M[11:13, 6:11, 6:11] = False                                      # a 5x5 pit, 2 deep, open at the face
    pit = np.zeros_like(M)
    pit[11:13, 6:11, 6:11] = True
    assert int(pit.sum()) == 50 and not (M & pit).any() and int(M.sum()) == 274
    v = ops.mask_vol(M, (17, 17, 13), (0, 0, 0))
    c0 = ops.close(v, n)
    assert c0["dim"] == v["dim"] and c0["pos"] == v["pos"]
    a0 = c0["data"] != 0
    assert np.array_equal(a0, ref_close(M, n, (0, 0, 0))) and np.array_equal(a0, ops.close(v, n, (0, 0, 0))["data"] != 0)
    assert int((a0 & pit).sum()) == 24 and int(a0.sum()) == 298
    a1 = ops.close(v, n, (0, 0, 1))["data"] != 0
    assert np.array_equal(a1, ref_close(M, n, (0, 0, 1)))
    assert int((a1 & pit).sum()) == 50 and int((a1 != a0).sum()) == 26 and int(a1.sum()) == 324
    for f in ((1, 0, 0), (0, 1, 0)):
        assert np.array_equal(ops.close(v, n, f)["data"] != 0, a0), f
    assert np.array_equal(ops.close(v, n, 1)["data"] != 0, a1)
    assert np.array_equal(ops.close(v, n, (1, 1, 1))["data"] != 0, ref_close(M, n, (1, 1, 1)))
    assert not (a0 & ~a1).any() and (a1 & M).sum() == M.sum()                     # extensive, the mirror only adds
    # (5) the contracts
    with pytest.raises(TypeError):
        ops.open_(v, n, continuous_at_boundary=1)
    with pytest.raises(TypeError):
        ops.erosion(v, n, continuous_at_boundary=1)
    for bad in (2, -1, (1, 1), (0, 0, 0, 0), "111", 1.0, None, (1, 0, "1")):
        for fn in (ops.dilation, ops.close):
            with pytest.raises(ValueError):
                fn(v, n, bad)


def test_close_fills_a_gap_and_returns_the_input_grid():
    m = np.zeros((10, 10, 20), bool)
    m[2:8, 2:8, 2:8] = True
    m[2:8, 2:8, 10:16] = True
    v = ops.mask_vol(m, (20, 10, 10), (3, 4, 5))
    out = ops.close(v, 2)
    assert out["dim"] == v["dim"] and out["pos"] == v["pos"]
    o = out["data"] != 0
    assert o[m].all()                                # extensive
    assert o[4:6, 4:6, 8:10].all()                   # the core of the 2-voxel gap is bridged
    assert not o[0].any() and not o[:, 0].any() and not o[:, :, 0].any() and not o[:, :, -1].any()


def test_open_removes_a_spike_and_returns_the_input_grid():
    m = np.zeros((10, 10, 16), bool)
    m[2:8, 2:8, 2:8] = True
    m[4, 4, 8:13] = True                             # a 1-voxel spike
    v = ops.mask_vol(m, (16, 10, 10), (0, 0, 0))
    out = ops.open_(v, 2)
    assert out["dim"] == v["dim"] and out["pos"] == v["pos"]
    o = out["data"] != 0
    assert not o[4, 4, 8:13].any()
    assert not o[~m].any()                           # anti-extensive
    assert o[4:6, 4:6, 4:6].all()


# ------------------------------------------------------------------------------------------- seg_gauss
def test_gauss_weights_are_ipl_s_float32_taps():
    w = ops.ipl_gauss_weights(2.0, 3)
    assert w.dtype == np.float32 and w.shape == (7,)
    assert [hex(int(x)) for x in w[3:].view(np.int32)] == ["0x3e5d4ae1", "0x3e434a39", "0x3e06387f", "0x3d8fafb2"]
    assert np.array_equal(w, w[::-1])


def test_mgha_to_native_rounds_to_nearest():
    assert ops.mgha_to_native(500, 1619.07703, -394.095001, 8192) == 4524        # 4523.83
    assert ops.mgha_to_native(3000, 1619.07703, -394.095001, 8192) == 17173      # 17173.01


def test_calibration_from_proclog():
    log = "Mu_Scaling                                       8192\nDensity: slope                         1.61907703e+03\nDensity: intercept                    -3.94095001e+02\n"
    cal = ops.calibration_from_proclog(log)
    assert cal == dict(slope=1619.07703, intercept=-394.095001, mu_scaling=8192.0)
    assert ops.calibration_from_proclog(dict(proclog=log)) == cal
    with pytest.raises(ValueError):
        ops.calibration_from_proclog("Mu_Scaling 8192")


def _scalar_reference_lp(box, w):
    """The rule written as scalar float32 operations: three passes x, y, z, centre-out pairs, truncation."""
    f = np.float32
    r = (len(w) - 1) // 2
    v = box.astype(np.float32)
    for axis in (2, 1, 0):
        out = np.zeros_like(v)
        n = v.shape[axis]
        for idx in np.ndindex(*v.shape):
            i = idx[axis]

            def at(k):
                j = i + k
                if j < 0 or j >= n:
                    return f(0)
                jj = list(idx)
                jj[axis] = j
                return v[tuple(jj)]

            acc = f(w[r] * at(0))
            for k in range(1, r + 1):
                acc = f(acc + f(w[r + k] * f(at(-k) + at(k))))
            out[idx] = np.trunc(acc)
        v = out
    return v


def _random_box(seed):
    rng = np.random.default_rng(seed)
    box = rng.integers(0, 3000, size=(7, 8, 9)).astype(np.int16)
    box[rng.random(box.shape) < 0.3] = 0
    return box


def _f64_pair_lp(box, w):
    """The same three passes, centre-out pair order and per-pass truncation, but accumulated in float64 with
    the float32 taps: differs from the float32 rule on the seed-2 box at (z, y, x) = (3, 7, 3), 656 vs 655."""
    r = (len(w) - 1) // 2
    v = box.astype(np.float64)
    for axis in (2, 1, 0):
        n = v.shape[axis]
        pad = [(0, 0)] * 3
        pad[axis] = (r, r)
        vp = np.pad(v, pad)

        def sh(k):
            sl = [slice(None)] * 3
            sl[axis] = slice(r + k, r + k + n)
            return vp[tuple(sl)]

        acc = sh(0) * float(w[r])
        for k in range(1, r + 1):
            acc = acc + (sh(-k) + sh(k)) * float(w[r + k])
        v = np.trunc(acc)
    return v.astype(np.float32)


@pytest.mark.parametrize("seed", [2, 3, 47])   # 2 pins float32 accumulation, 47 the centre-out pair order
def test_gauss_lp_equals_the_scalar_float32_reference(seed):
    """Seed 3 alone cannot tell float64 accumulation or a sequential -r..r order from the float32 centre-out
    pair rule; seed 2 differs under float64 accumulation at (3, 7, 3) (656 vs 655) and seed 47 under the
    sequential order at (3, 7, 1) (535 vs 536)."""
    box = _random_box(seed)
    w = ops.ipl_gauss_weights(2.0, 3)
    assert np.array_equal(ops.gauss_lp_ipl(box, 2.0, 3), _scalar_reference_lp(box, w))


def test_seed_2_box_is_sensitive_to_float32_accumulation():
    box = _random_box(2)
    w = ops.ipl_gauss_weights(2.0, 3)
    assert not np.array_equal(_f64_pair_lp(box, w), ops.gauss_lp_ipl(box, 2.0, 3))


def test_seg_gauss_truncates_after_every_pass_and_returns_the_valid_grid():
    """A single voxel of 1000: x pass 1000*w0 -> 216, y pass 216*w0 -> 46, z pass 46*w0 -> 9; without the
    per-pass truncation the centre would be 1000*w0^3 = 10.09.  The output grid is shrunk by `support`."""
    box = np.zeros((9, 9, 9), np.int16)
    box[4, 4, 4] = 1000
    v = ops.vol(box, (9, 9, 9), (5, 6, 7))
    out = ops.seg_gauss(v, 2.0, 3, 10, 100)
    assert out["dim"] == (3, 3, 3) and out["pos"] == (8, 9, 10)
    assert not (out["data"] != 0).any()
    out9 = ops.seg_gauss(v, 2.0, 3, 9, 100)
    assert out9["data"][1, 1, 1] == 127 and int((out9["data"] != 0).sum()) == 1
    lp = ops.gauss_lp_ipl(box, 2.0, 3)
    assert lp[4, 4, 4] == 9 and lp[4, 4, 5] == 8 and lp[4, 4, 7] == 3       # 216*0.19 -> 41 -> 41*0.216 -> 8
    with pytest.raises(ValueError):
        ops.seg_gauss(ops.vol(np.zeros((6, 9, 9), np.int16), (9, 9, 6), (0, 0, 0)), 2.0, 3, 10, 100)


def test_seg_gauss_bounds_are_inclusive():
    """A plateau of 1000 smooths to exactly 1000 at the centre: the voxel is set when the value equals either
    bound (the upper bound is unobserved on the oracle; this pins the implemented inclusive choice)."""
    box = np.full((9, 9, 9), 1000, np.int16)
    v = ops.vol(box, (9, 9, 9), (0, 0, 0))
    c = int(ops.gauss_lp_ipl(box, 2.0, 3)[4, 4, 4])
    assert c == 1000
    assert ops.seg_gauss(v, 2.0, 3, c, c)["data"][1, 1, 1] == 127
    assert ops.seg_gauss(v, 2.0, 3, c + 1, 10**6)["data"][1, 1, 1] == 0
    assert ops.seg_gauss(v, 2.0, 3, 0, c - 1)["data"][1, 1, 1] == 0


def test_seg_gauss_requires_integral_native_thresholds():
    """4523.83 must not be silently truncated to 4523 (the refuted rule); integral floats and numpy ints pass."""
    box = np.zeros((9, 9, 9), np.int16)
    box[4, 4, 4] = 1000
    v = ops.vol(box, (9, 9, 9), (5, 6, 7))
    ref = ops.seg_gauss(v, 2.0, 3, 9, 100)
    for lo, up in ((9.0, 100), (np.int64(9), np.float64(100.0)), (np.int32(9), np.int32(100))):
        assert np.array_equal(ops.seg_gauss(v, 2.0, 3, lo, up)["data"], ref["data"])
    for lo, up in ((4523.83, 17173), (9, 100.5), (np.float32(9.25), 100), (float("nan"), 100), (9, float("inf"))):
        with pytest.raises(ValueError):
            ops.seg_gauss(v, 2.0, 3, lo, up)


def test_seg_gauss_and_metric11_reject_nonsense_parameters():
    """support 0 and negative distances have no oracle: they raise ValueError instead of failing inside numpy
    (or, for dilation -2, silently returning a shrunk grid); N = 0 stays legal."""
    box = np.zeros((7, 7, 7), np.int16)
    box[3, 3, 3] = 1000
    v = ops.vol(box, (7, 7, 7), (0, 0, 0))
    with pytest.raises(ValueError):
        ops.seg_gauss(v, 2.0, 0, 9, 100)
    m = np.zeros((7, 7, 7), bool)
    m[2:5, 2:5, 2:5] = True
    mv = ops.mask_vol(m, (7, 7, 7), (1, 2, 3))
    for fn, n in ((ops.close, -2), (ops.open_, -2), (ops.dilation, -2), (ops.erosion, -1), (ops.dilation, -1)):
        with pytest.raises(ValueError):
            fn(mv, n)
    assert np.array_equal(ops.erosion(mv, 0)["data"], mv["data"])
    c0 = ops.close(mv, 0)
    assert c0["dim"] == mv["dim"] and c0["pos"] == mv["pos"] and np.array_equal(c0["data"], mv["data"])
    d0 = ops.dilation(mv, 0)
    assert d0["dim"] == (9, 9, 9) and int((d0["data"] != 0).sum()) == 27


# ------------------------------------------------------------------------------------------- components
def test_slicewise_fraction_of_the_slice_total():
    v = np.zeros((4, 20, 40), np.uint8)
    v[0, 2:8, 2:12] = 127                 # 60 voxels (60 %)
    v[0, 12:16, 20:30] = 127              # 40 voxels (40 %)  -> removed
    v[1, 2:7, 2:12] = 127                 # 50 voxels (50 %)
    v[1, 12:17, 20:30] = 127              # 50 voxels (50 %)  -> both kept: the tie is inclusive
    v[2, 2:6, 2:12] = 127                 # 40 / 35 / 25 %    -> nothing reaches 50 %: the slice is cleared
    v[2, 8:13, 2:9] = 127
    v[2, 14:19, 2:7] = 127
    v[3, 2:8, 2:12] = 127                 # 60 voxels touching a 40-voxel block only diagonally
    v[3, 8:12, 12:22] = 127               #   4-connectivity: two components -> the 40 are removed
    out = ops.cl_slicewise_extractow(V(v), 50.0, 100.0)
    assert out["dim"] == (40, 20, 4) and out["pos"] == (0, 0, 0)
    counts = [int((out["data"][z] != 0).sum()) for z in range(4)]
    assert counts == [60, 100, 0, 60]
    assert set(np.unique(out["data"])) <= {0, 127}
    empty = ops.cl_slicewise_extractow(V(np.zeros((2, 5, 5), np.uint8)))
    assert not empty["data"].any()


def _two_blocks():
    m = np.zeros((8, 8, 8), bool)
    m[1:4, 1:4, 1:4] = True               # 27 voxels
    m[5:7, 5:7, 5:7] = True               # 8 voxels
    m[4, 4, 4] = True                     # 1 voxel touching the 27-block only at a corner (6-conn: separate)
    return m


def test_rank_extract_keeps_the_requested_ranks_6_connected():
    v = ops.mask_vol(_two_blocks(), (8, 8, 8), (1, 2, 3))
    r1 = ops.cl_ow_rank_extract(v, 1, 1)
    assert r1["dim"] == (8, 8, 8) and r1["pos"] == (1, 2, 3)
    assert int((r1["data"] != 0).sum()) == 27 and (r1["data"][1:4, 1:4, 1:4] == 127).all()
    r2 = ops.cl_ow_rank_extract(v, 2, 2)
    assert int((r2["data"] != 0).sum()) == 8
    r12 = ops.cl_ow_rank_extract(v, 1, 2)
    assert int((r12["data"] != 0).sum()) == 35 and r12["data"][4, 4, 4] == 0
    assert [int(s) for s in ops.component_sizes(v)] == [27, 8, 1]
    with pytest.raises(AssertionError):
        ops.cl_ow_rank_extract(v, 1, 1, topology=26)
    for bad in [(0, 1), (0, 0), (-1, 1), (2, 1), (1, 0)]:       # ranks are 1-based; rank 0 would wrap silently
        with pytest.raises(ValueError):
            ops.cl_ow_rank_extract(v, *bad)
    assert not ops.cl_ow_rank_extract(v, 4, 4)["data"].any()      # beyond the component count: empty, no error
    assert int((ops.cl_ow_rank_extract(v, 1, 99)["data"] != 0).sum()) == 36


# The 41 component sizes of the probe-18 phantom in raster label order (scipy.ndimage.label), compressed
# order-preservingly to small integers (ties kept): label 19 = T1 and 22 = T2 (20,000 -> 27), 20 = a
# (15,000 -> 26), 21 = b and 23 = c (10,000 -> 25).  IPL's exports RANK1..5 = T2, T1, a, b, c.
P18_SIZES = [13, 9, 5, 8, 8, 5, 4, 2, 7, 3, 2, 9, 5, 11, 1, 15, 14, 16, 27, 26, 25, 27, 25, 21, 10, 18, 12,
             24, 23, 19, 19, 19, 19, 22, 6, 17, 17, 17, 17, 20, 13]
P18_RANKS = [22, 19, 20, 21, 23]                 # labels of ranks 1..5 in IPL's exports


def _runs_along_x(sizes):
    """One run of `s` voxels per size, separated by one background voxel, on a single row: the raster label
    order is the list order."""
    row = np.zeros(sum(sizes) + len(sizes), np.uint8)
    x = 0
    for s in sizes:
        row[x:x + s] = 127
        x += s + 1
    return ops.vol(row[None, None, :], (row.size, 1, 1), (0, 0, 0)), row


def test_rank_tie_break_is_ipl_s_unstable_sort_on_the_probe18_table():
    """rank_order on the phantom's size table puts the later of the two largest equal components first and
    the earlier of the two 10,000 pair first -- what IPL's RANK1..5 exports showed -- while a stable
    largest-first sort would put label 19 (T1) at rank 1."""
    sizes = np.array(P18_SIZES)
    assert (ops.rank_order(sizes)[:5] + 1).tolist() == P18_RANKS
    assert (np.argsort(-sizes, kind="stable")[:5] + 1).tolist() == [19, 22, 20, 21, 23]
    v, row = _runs_along_x(P18_SIZES)
    lab, n, cnt = ops.label6(v["data"] != 0)
    assert n == 41 and cnt[1:].tolist() == P18_SIZES                       # label order = list order
    for rank, label in enumerate(P18_RANKS, start=1):
        out = ops.cl_ow_rank_extract(v, rank, rank)["data"] != 0
        assert np.array_equal(out, lab == label), f"rank {rank}: expected label {label}"
    both = ops.cl_ow_rank_extract(v, 1, 2)["data"] != 0
    assert np.array_equal(both, (lab == 19) | (lab == 22))
    assert ops.rank_order(np.array([5, 5, 5])).tolist() == [0, 1, 2]           # a full tie keeps the order
    assert ops.rank_order(np.array([], np.int64)).size == 0


def _cb_volume():
    """The probe-18 connect_boundary sub-volume at small scale: B1 (large) and B2 (small) touch z = 0, B4 touches
    z = last, B3 is interior; sizes chosen so every reading gives a different rank 1 / rank 2."""
    m = np.zeros((10, 20, 40), bool)
    m[0:4, 2:6, 2:12] = True                # B1 160, touches z0
    m[0:2, 2:4, 14:18] = True               # B2 16, touches z0
    m[3:7, 10:15, 20:30] = True             # B3 200, interior (largest single component)
    m[8:10, 2:5, 32:38] = True              # B4 36, touches z1
    return ops.mask_vol(m, (40, 20, 10), (5, 6, 7)), m


def test_rank_extract_connect_boundary_true_joins_every_face_touching_component():
    v, m = _cb_volume()
    B1, B2, B3, B4 = (np.zeros_like(m) for _ in range(4))
    B1[0:4, 2:6, 2:12] = B2[0:2, 2:4, 14:18] = B3[3:7, 10:15, 20:30] = B4[8:10, 2:5, 32:38] = True
    f1 = ops.cl_ow_rank_extract(v, 1, 1)["data"] != 0
    assert np.array_equal(f1, B3)                                          # false: B3 (200) is the largest
    t1 = ops.cl_ow_rank_extract(v, 1, 1, connect_boundary=True)
    assert t1["dim"] == v["dim"] and t1["pos"] == v["pos"]
    assert np.array_equal(t1["data"] != 0, B1 | B2 | B4)                   # 212 > 200: joined across z0 and z1
    assert np.array_equal(ops.cl_ow_rank_extract(v, 2, 2, connect_boundary=True)["data"] != 0, B3)
    assert not ops.cl_ow_rank_extract(v, 3, 3, connect_boundary=True)["data"].any()
    # x and y faces join too; a volume with nothing on a face ranks as with connect_boundary false
    w = np.zeros((6, 8, 8), bool)
    w[2:4, 0:2, 1:3] = True                 # 8 voxels on the y0 face (y 0..1)
    w[2:4, 5:7, 6:8] = True                 # 8 voxels on the x1 face (x 6..7)
    w[1:5, 3:5, 2:5] = True                 # 24 interior (y 3..4, x 2..4): one empty row / column to each
    wv = ops.mask_vol(w, (8, 8, 6), (0, 0, 0))
    assert ops.label6(w)[1] == 3
    assert int((ops.cl_ow_rank_extract(wv, 1, 1, connect_boundary=True)["data"] != 0).sum()) == 24   # joined 16 < 24
    assert int((ops.cl_ow_rank_extract(wv, 2, 2, connect_boundary=True)["data"] != 0).sum()) == 16   # y0 + x1 blocks
    w[1:5, 0:2, 1:4] = True                 # y0 block now 24: joined 32 > 24
    wv = ops.mask_vol(w, (8, 8, 6), (0, 0, 0))
    assert ops.label6(w)[1] == 3
    assert int((ops.cl_ow_rank_extract(wv, 1, 1, connect_boundary=True)["data"] != 0).sum()) == 32
    inner = np.zeros((6, 8, 8), bool)
    inner[1:5, 3:5, 3:5] = True
    iv = ops.mask_vol(inner, (8, 8, 6), (0, 0, 0))
    assert np.array_equal(ops.cl_ow_rank_extract(iv, 1, 1, connect_boundary=True)["data"], ops.cl_ow_rank_extract(iv, 1, 1)["data"])


def test_nr_extract_size_window():
    v = ops.mask_vol(_two_blocks(), (8, 8, 8), (0, 0, 0))
    assert int((ops.cl_nr_extract(v, 9, 0)["data"] != 0).sum()) == 27
    assert int((ops.cl_nr_extract(v, 1, 8)["data"] != 0).sum()) == 9
    assert int((ops.cl_nr_extract(v, 2, 26)["data"] != 0).sum()) == 8
    assert not ops.cl_nr_extract(v, 28, 0)["data"].any()
    assert not ops.cl_nr_extract(ops.cl_nr_extract(v, 28, 0), 1, 500000)["data"].any()


# ------------------------------------------------------------------------------------------- bookkeeping
def test_subtract_and_add_on_the_union_grid():
    a = ops.mask_vol(np.ones((4, 4, 4), bool), (4, 4, 4), (0, 0, 0))
    b = ops.mask_vol(np.ones((4, 4, 4), bool), (4, 4, 4), (2, 2, 2))
    s = ops.subtract_aims(a, b)
    assert s["dim"] == (6, 6, 6) and s["pos"] == (0, 0, 0) and s["data"].dtype == np.int8
    assert int((s["data"] == 127).sum()) == 64 - 8 and int((s["data"] == -127).sum()) == 64 - 8
    assert int((s["data"] != 0).sum()) == 2 * (64 - 8)                       # the -127 voxels ARE set
    assert (s["data"][:2, :2, :2] == 127).all() and not s["data"][2:4, 2:4, 2:4].any()
    assert (s["data"][4:, 4:, 4:] == -127).all()                             # only b is set there
    d = ops.add_aims(a, b)
    assert d["dim"] == (6, 6, 6) and d["pos"] == (0, 0, 0) and d["data"].dtype == np.int8
    assert int((d["data"] != 0).sum()) == 64 + 64 - 8 and d["data"].max() == 127 and (d["data"][2:4, 2:4, 2:4] == 127).all()
    assert ops.union_grid(b, a) == ((6, 6, 6), (0, 0, 0))
    s2 = ops.subtract_aims(b, a)
    assert s2["pos"] == (0, 0, 0) and int((s2["data"] == 127).sum()) == 56 and (s2["data"][4:, 4:, 4:] == 127).all()
    assert int((s2["data"] == -127).sum()) == 56 and (s2["data"][:2, :2, :2] == -127).all()
    assert np.array_equal(s2["data"], -s["data"])
    assert s2["data"].view(np.uint8)[0, 0, 0] == 129                         # IPL's byte 0x81


def test_subtract_and_add_keep_char_arithmetic_and_reject_short_operands():
    """The arithmetic is char arithmetic (a negative difference is stored, -127 for 0 - 127, as IPL's probe-18
    export SUB showed; sums saturate at 127, export ADD), never a boolean AND NOT / OR; a short greyscale
    operand is refused (IPL's arithmetic on a short volume was never observed, and it would silently become a
    constant-127 mask).  Saturation at -127 for a difference below it is the implemented, unobserved choice."""
    a = ops.vol(np.array([[[127, 126, 0, 100, 5]]], np.uint8), (5, 1, 1), (0, 0, 0))
    b = ops.vol(np.array([[[126, 127, 127, 100, 5]]], np.uint8), (5, 1, 1), (0, 0, 0))
    assert ops.subtract_aims(a, b)["data"].tolist() == [[[1, -1, -127, 0, 0]]]
    assert ops.add_aims(a, b)["data"].tolist() == [[[127, 127, 127, 127, 10]]]
    neg = ops.vol(np.array([[[-127, -127, 127]]], np.int8), (3, 1, 1), (0, 0, 0))
    pos = ops.vol(np.array([[[127, 0, 0]]], np.uint8), (3, 1, 1), (0, 0, 0))
    assert ops.subtract_aims(neg, pos)["data"].tolist() == [[[-127, -127, 127]]]          # saturated below
    assert ops.add_aims(neg, pos)["data"].tolist() == [[[0, -127, 127]]]
    grey = ops.vol(np.full((2, 2, 2), 3000, np.int16), (2, 2, 2), (0, 0, 0))
    mask = ops.mask_vol(np.zeros((2, 2, 2), bool), (2, 2, 2), (0, 0, 0))
    for fn in (ops.subtract_aims, ops.add_aims):
        with pytest.raises(TypeError):
            fn(grey, mask)
        with pytest.raises(TypeError):
            fn(mask, grey)
        for char in (np.ones((2, 2, 2), np.int8), np.ones((2, 2, 2), bool)):        # int8 and bool are char
            out = fn(mask, ops.vol(char, (2, 2, 2), (0, 0, 0)))
            assert out["data"].dtype == np.int8 and int(out["data"].sum()) == (8 if fn is ops.add_aims else -8)
    # a -127 voxel is 'set' for every consumer: set_value, the labelling commands, bounding_box_cut, morphology
    s = ops.subtract_aims(ops.mask_vol(np.zeros((3, 3, 3), bool), (3, 3, 3), (0, 0, 0)),
                          ops.mask_vol(np.ones((3, 3, 3), bool), (3, 3, 3), (0, 0, 0)))
    assert (s["data"] == -127).all()
    assert (ops.set_value(s, 127, 0)["data"] == 127).all() and not ops.set_value(s, 0, 127)["data"].any()
    assert int((ops.cl_ow_rank_extract(s, 1, 1)["data"] != 0).sum()) == 27
    assert int((ops.cl_nr_extract(s, 27, 27)["data"] != 0).sum()) == 27
    assert int((ops.cl_slicewise_extractow(s, 50, 100)["data"] != 0).sum()) == 27
    assert ops.bounding_box_cut(ops.subtract_aims(ops.mask_vol(np.zeros((5, 5, 5), bool), (5, 5, 5), (0, 0, 0)),
                                                  ops.mask_vol(np.pad(np.ones((3, 3, 3), bool), 1), (5, 5, 5), (0, 0, 0))))["dim"] == (3, 3, 3)
    assert int((ops.erosion(s, 0)["data"] != 0).sum()) == 27 and int((ops.dilation(s, 0)["data"] != 0).sum()) == 27


def test_on_grid_pastes_by_global_position():
    v = ops.mask_vol(np.ones((2, 2, 2), bool), (2, 2, 2), (5, 5, 5))
    g = ops.on_grid(v, (4, 4, 4), (4, 4, 4))
    assert g.shape == (4, 4, 4) and int((g != 0).sum()) == 8 and (g[1:3, 1:3, 1:3] != 0).all()
    assert ops.on_grid(v, (2, 2, 2), (5, 5, 5)) is v["data"]
    g2 = ops.on_grid(v, (2, 2, 2), (4, 4, 4))                        # equal dim, different pos: no by-shape shortcut
    assert g2 is not v["data"] and int((g2 != 0).sum()) == 1 and g2[1, 1, 1] != 0


def test_bounding_box_cut_is_the_tight_box():
    m = np.zeros((8, 6, 10), np.uint8)
    m[2:5, 1:4, 5:9] = 127
    v = ops.vol(m, (10, 6, 8), (10, 20, 30))
    out = ops.bounding_box_cut(v)
    assert out["dim"] == (4, 3, 3) and out["pos"] == (15, 21, 32)
    assert (out["data"] == 127).all()
    empty = ops.bounding_box_cut(ops.vol(np.zeros((2, 2, 2), np.uint8), (2, 2, 2), (0, 0, 0)))
    assert empty["dim"] == (2, 2, 2)
    grey = ops.vol(np.where(m != 0, np.int16(-5), np.int16(0)), (10, 6, 8), (0, 0, 0))
    assert ops.bounding_box_cut(grey)["dim"] == (4, 3, 3)                # non-zero, not positive


def test_set_value_and_invert():
    v = V(np.array([[[0, 5, 127]]], np.uint8))
    inv = ops.set_value(v, 0, 127)
    assert inv["data"].tolist() == [[[127, 0, 0]]]
    assert ops.set_value(v, 127, 0)["data"].tolist() == [[[0, 127, 127]]]


def test_gobj_maskaimpeel_pastes_by_position_and_peels_slicewise():
    g = ops.mask_vol(np.ones((3, 6, 6), bool), (6, 6, 3), (0, 0, 1))          # contour on z = 1..3 only
    v = ops.mask_vol(np.ones((5, 6, 6), bool), (6, 6, 5), (0, 0, 0))
    p0 = ops.gobj_maskaimpeel_ow(v, g, 0)
    assert p0["dim"] == v["dim"] and p0["pos"] == v["pos"]
    assert [int((p0["data"][z] != 0).sum()) for z in range(5)] == [0, 36, 36, 36, 0]
    p1 = ops.gobj_maskaimpeel_ow(v, g, 1)
    assert [int((p1["data"][z] != 0).sum()) for z in range(5)] == [0, 16, 16, 16, 0]
    assert (p1["data"][1, 1:5, 1:5] == 127).all()
    g_same = ops.mask_vol(np.ones((5, 6, 6), bool), (6, 6, 5), (0, 0, 1))  # SAME dim as v, shifted one slice
    p_same = ops.gobj_maskaimpeel_ow(v, g_same, 0)
    assert [int((p_same["data"][z] != 0).sum()) for z in range(5)] == [0, 36, 36, 36, 36]
    grey = ops.vol(np.full((5, 6, 6), -7, np.int16), (6, 6, 5), (0, 0, 0))
    mg = ops.mask_by_gobj(grey, ops.peel_gobj_render(g, 0))
    assert mg["data"].dtype == np.int16 and (mg["data"][1:4] == -7).all() and not mg["data"][0].any()


# ------------------------------------------------------------------------------------------- probe 18 (slow)
P18_ROOT = os.environ.get("IPLDT_PROBE18_ROOT", lab_path("Python/scripts/IPL/probes/p18_step1_edges"))


def _p18(tag):
    fs = vms_versions(os.path.join(P18_ROOT, "outputs"), f"X2420448_P18_{tag}.AIM")
    return read_aim(fs[-1]) if fs else None


def _same(ours, ref, valued=False):
    """(mismatches, grids equal) on the union grid by global position; set-ness unless `valued` (exact int8
    values: IPL's uncompressed char export read as uint8 is viewed as int8)."""
    dim, pos = ops.union_grid(ours, ref)
    a = ops.on_grid(ours, dim, pos)
    b = ops.on_grid(ref, dim, pos)
    if valued:
        a, b = a.astype(np.int8), np.asarray(b).astype(np.uint8).view(np.int8)
    else:
        a, b = a != 0, b != 0
    return int((a != b).sum()), (tuple(ours["dim"]), tuple(ours["pos"])) == (tuple(ref["dim"]), tuple(ref["pos"]))


def test_open_at_the_z_faces_is_open_boundary_erosion_then_padded_dilation():
    """The /open rule at a volume face (the configuration of PFJ-411dfd_R's residual, N = 2 here so 3N+2 = 8): a
    20x20 block through the whole z range (touching both z faces).  (1) open = erosion then dilation, cropped to
    the input grid, composed from the public primitives.  (2) The erosion is open-bounded on the z faces: no z
    layer is lost (a background-padded erosion would remove two layers per face), the in-plane shrink is by
    exactly 2 voxels (keep <=> raw >= 8, nearest background face-adjacent in-plane).  (3) The dilation restores
    raw < 8: on the last slice the corner-adjacent voxel at raw 7 (offset (2,1) = 4 + 3) comes back and the corner
    voxel at raw 8 (offset (2,2) = 4 + 4) does not -- the 3N+1 / 3N+2 pair that the PFJ-411dfd_R voxels sit on at
    47 / 48 for N = 15 -- so the open rounds the four corners of every slice, the last slice included, and a
    1-voxel spike on the last slice (raw 3 to the background, raw 9 to the eroded set) is removed.  Under the
    mirror-margin mechanism (open_ since 2026-09-14 evening) this volume has no extra margin survivor (the block's
    mirror image is the block continued, the spike's mirror is eroded like the spike), so the composition of the two
    primitives -- the pre-mechanism rule -- and open_ coincide here; test_open_mechanism_on_the_plate_over_cave_phantom
    is where they differ."""
    n, t = 2, ops.metric11_threshold(2)
    assert t == 8
    m = np.zeros((6, 30, 30), bool)
    m[:, 5:25, 5:25] = True
    m[5, 15, 25:28] = True                                  # spike on the last slice
    v = ops.mask_vol(m, (30, 30, 6), (100, 200, 300))
    out = ops.open_(v, n)
    assert out["dim"] == v["dim"] and out["pos"] == v["pos"]
    o = out["data"] != 0
    e = ops.erosion(v, n)
    E = e["data"] != 0
    expected_e = np.zeros_like(m)
    expected_e[:, 7:23, 7:23] = True                        # every z layer kept, 2 voxels off each in-plane side
    assert np.array_equal(E, expected_e)
    d = ops.dilation(e, n)
    assert d["dim"] == (36, 36, 12) and d["pos"] == (97, 197, 297)
    assert np.array_equal(ops.on_grid(d, v["dim"], v["pos"]) != 0, o)     # composition, cropped by position
    expected = m.copy()
    expected[5, 15, 25:28] = False                          # spike gone
    for y, x in [(5, 5), (5, 24), (24, 5), (24, 24)]:
        expected[:, y, x] = False                            # the corners: raw 8 = 3N+2 to the eroded set
    assert np.array_equal(o, expected)
    assert o[5, 6, 5] and o[5, 5, 6] and not o[5, 5, 5]     # last slice: raw 7 restored, raw 8 not
    assert int(o[0].sum()) == int(o[5].sum()) == 396 and int(o.sum()) == 6 * 396
    assert not o[~m].any()                                  # anti-extensive


# ------------------------------------------------------------------------------------------- the /open mechanism
def old_rule_open(v, n):
    """The pre-mechanism open_ (before 2026-09-14 evening): the open-boundary erosion on the input grid, then the
    dilation on a background margin of N + 2, cropped -- the two public primitives chained -- written independently
    of the helper so that _open_padded(..., 'open_boundary') is checked against it."""
    e = ops.erosion(v, n)
    d = ops.dilation(e, n)
    return ops.mask_vol(ops.on_grid(d, v["dim"], v["pos"]) != 0, v["dim"], v["pos"])


def cave_block(T, size=100, mg=18, H=40, r_in=9.0, w=2.0):
    """One block of the plate-over-cave phantom of probe 20 (make_phantoms20.cave_blocks with a single T): a plate of
    thickness T, size x size in-plane, touching the z1 face (z H-T .. H-1) over a background cave (z 0 .. H-T-1), with a
    ring trench 1 slice deep at radius r_in .. r_in + w around the block centre cut into the face slice; 18 background
    voxels around the plate.  Returns (volume bool (z, y, x), the disc r < r_in on the face slice)."""
    X = Y = 2 * mg + size
    a = np.zeros((H, Y, X), bool)
    a[H - T:H, mg:mg + size, mg:mg + size] = True
    c = mg + size // 2
    yy, xx = np.mgrid[0:Y, 0:X]
    r = np.hypot(xx - c, yy - c)
    a[H - 1][(r >= r_in) & (r < r_in + w)] = False
    disc = np.zeros_like(a)
    disc[H - 1] = r < r_in
    return a, disc


def test_open_mechanism_on_the_plate_over_cave_phantom():
    """The synthetic isolation of the mechanism (probe 20 block C, predictions written 2026-09-14 18:12 before any
    scanner run): in the volume the trench erodes the disc inside the ring and nothing restores it under the
    pre-mechanism rule (the nearest in-volume survivors are 27 voxels away), whatever T.  Under the mechanism the
    N + 2 = 17 margin above the face holds the mirror image of the plate: for T <= 16 the mirrored cave bottom lies at
    depth T + 1 <= 17, inside the margin, and erodes every margin voxel; for T >= 17 it lies beyond the margin, so the
    margin voxels above the disc at depth >= 14 (chamfer >= 47 from the mirrored trench: 4 * 9 + 3 * (d - 10)) survive
    and seed the dilation back to the face slice: 117 of the 249 disc voxels (and 21 more on the slice below, 138 in
    all) are restored, exactly the same count for T = 17 and 18.  The edge-exclusive mirror (numpy 'reflect', the
    nearest refuted alternative) puts the mirrored cave one slice nearer, so it restores nothing at T = 17 and, at
    T = 18, the whole disc (249).  These are the counts the probe predicted for its cave export (0 / 0 / 117 / 117),
    and IPL returned exactly them on 2026-09-15 -- against the export itself in
    test_probe20_cave_discs_and_the_six_face_mirror, on the same geometry rebuilt here."""
    n, m = 15, 17
    got = {}
    for T in (15, 16, 17, 18):
        a, disc = cave_block(T)
        v = ops.mask_vol(a, a.shape[::-1], (0, 0, 0))
        mech = ops.open_(v, n)["data"] != 0
        old = old_rule_open(v, n)["data"] != 0
        assert np.array_equal(old, ops._open_padded(a, n, m, "open_boundary"))
        refl = ops._open_padded(a, n, m, "reflect")
        assert not (mech & ~a).any() and not (old & ~a).any()          # anti-extensive
        assert not (old & ~mech).any()                                  # the mechanism only adds
        got[T] = (int((mech & disc).sum()), int((old & disc).sum()), int((refl & disc).sum()), int((mech & ~old).sum()))
    assert int(disc.sum()) == 249
    assert {T: g[0] for T, g in got.items()} == {15: 0, 16: 0, 17: 117, 18: 117}     # mechanism
    assert {T: g[1] for T, g in got.items()} == {15: 0, 16: 0, 17: 0, 18: 0}         # pre-mechanism rule
    assert {T: g[2] for T, g in got.items()} == {15: 0, 16: 0, 17: 0, 18: 249}       # edge-exclusive mirror
    assert {T: g[3] for T, g in got.items()} == {15: 0, 16: 0, 17: 138, 18: 138}     # whole-volume difference


def test_open_equals_the_pre_mechanism_rule_without_extra_margin_survivors():
    """Where no margin survivor lacks a surviving partner the mechanism restores exactly what the chained primitives
    restore: a centred ball (its mirror images are eroded by the mirrored background around it), a block through the
    z faces with a spike (the block continues into the margin) and a random structure that keeps 3 voxels off every
    face (the mirrored background at depth 1..3 erodes the whole margin at N = 2) -- open_ == erosion then dilation,
    cropped, and the same through every margin depth >= N + 2."""
    zz, yy, xx = np.indices((24, 24, 24))
    ball = (zz - 11.5) ** 2 + (yy - 11.5) ** 2 + (xx - 11.5) ** 2 <= 8.0 ** 2
    block = np.zeros((6, 30, 30), bool)
    block[:, 5:25, 5:25] = True
    block[5, 15, 25:28] = True
    rng = np.random.default_rng(7)
    rnd = np.zeros((14, 20, 20), bool)
    rnd[3:-3, 3:-3, 3:-3] = ndi.gaussian_filter(rng.standard_normal((8, 14, 14)), 1.2) > 0.1
    for a, n in ((ball, 3), (ball, 5), (block, 2), (rnd, 2)):
        v = ops.mask_vol(a, a.shape[::-1], (1, 2, 3))
        out = ops.open_(v, n)
        assert out["dim"] == v["dim"] and out["pos"] == v["pos"]
        assert np.array_equal(out["data"] != 0, old_rule_open(v, n)["data"] != 0)
        for margin in (n + 2, n + 4):
            assert np.array_equal(ops._open_padded(a, n, margin, "symmetric"), out["data"] != 0)
        assert not ((out["data"] != 0) & ~a).any()                                    # anti-extensive


def test_open_padded_contracts_and_the_thin_volume_warning(monkeypatch):
    """The helper: mode names, margin 0 = the uncropped buffer, negative margins / unknown modes raise; and the
    NARROWED thin-volume warning (2026-09-15).  Probe 20's z156 export -- 12 slices opened with N = 15, i.e. a
    mirrored margin of 17 against 12 slices, one full reflection plus 5 repeated layers -- matched numpy's
    'symmetric' at 0 mismatching voxels, so that regime is settled and no longer warns; ThinVolumeWarning now fires
    ONCE per process only when the mirrored margin is deeper than TWICE an axis (more than one repeat of the
    reflection), which probe 20 does not reach.  A background margin (the default of dilation / close) never
    warns, however thin the volume."""
    a = np.zeros((7, 7, 7), bool)
    a[2:5, 2:5, 2:5] = True
    assert set(ops.OPEN_MARGIN_MODES) == {"symmetric", "reflect", "edge", "wrap", "background", "object", "open_boundary"}
    for mode in ops.OPEN_MARGIN_MODES:
        out = ops._open_padded(a, 1, 3, mode)
        assert out.shape == a.shape and out.dtype == bool
    assert ops._open_padded(a, 1, 0, "symmetric").shape == a.shape
    assert ops._open_padded(a, 1, 0, "open_boundary").shape == a.shape
    with pytest.raises(ValueError):
        ops._open_padded(a, 1, 3, "mirror")
    with pytest.raises(ValueError):
        ops._open_padded(a, 1, -1, "symmetric")
    with pytest.raises(ValueError):
        ops._open_padded(a, -1, 3, "symmetric")
    monkeypatch.setattr(ops, "_thin_volume_warned", False)
    thin = ops.mask_vol(a, (7, 7, 7), (0, 0, 0))
    assert ops._mirror_repeat_axes(a.shape, 14, ("symmetric",) * 3) == ()          # margin = 2 * the axis: settled
    assert ops._mirror_repeat_axes(a.shape, 15, ("symmetric",) * 3) == (0, 1, 2)
    assert ops._mirror_repeat_axes(a.shape, 99, ("background",) * 3) == ()         # an empty border is never thin
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ops.open_(thin, 5)                                   # N + 2 = 7 = the axis length: one reflection
        ops.open_(thin, 6)                                   # margin 8 > 7 but <= 14: the z156 regime, settled
        ops.open_(thin, 12)                                  # margin 14 = 2 * 7: still one repeat
        ops.dilation(thin, 30)                               # a background margin, however deep
        ops.close(thin, 30)
        assert [x for x in w if issubclass(x.category, ops.ThinVolumeWarning)] == []
        ops.open_(thin, 13)                                  # margin 15 > 2 * 7: more than one repeat, warned
        ops.open_(thin, 13)                                  # once per process
        ops.close(thin, 13, (1, 1, 1))
        assert len([x for x in w if issubclass(x.category, ops.ThinVolumeWarning)]) == 1
    assert ops._thin_volume_warned is True
    out = ops.open_(thin, 13)["data"] != 0                   # numpy's repeated reflection: still valid and anti-extensive
    assert out.shape == a.shape and not (out & ~a).any()


# ------------------------------------------------------------------------------------------- probe 17 (slow)
P17_ROOT = os.environ.get("IPLDT_PROBE17_ROOT", lab_path("Python/scripts/IPL/probes/p17_step1_radius"))
P17_BASES = {"PFJ-0be66a_R": "X2420448", "PFJ-42293d_L": "X3623103", "PFJ-6f5538_R": "X5492058", "PFJ-411dfd_R": "X5143651"}


def _p17(subject, tag):
    fs = vms_versions(os.path.join(P17_ROOT, "aims_and_logs"), f"{P17_BASES[subject]}_P17_{tag}.AIM")
    if not fs:
        return None
    a = read_aim(fs[-1])
    return ops.vol(np.ascontiguousarray(a["data"]), a["dim"], a["pos"])


P17_RESIDUAL = [(1469, 175, 167), (1470, 179, 167), (1471, 178, 167)]        # sorted global (x, y, z), the last slice


def _mismatch_counts(out_bool, ref):
    """(mismatches, ours-only, IPL-only) of a bool array against an IPL export on the same grid."""
    b = ref["data"] != 0
    return int((out_bool != b).sum()), int((out_bool & ~b).sum()), int((~out_bool & b).sum())


@pytest.mark.slow
def test_probe17_open_and_close_oracles_and_the_pfj_411dfd_open_mechanism():
    """Probe 17 (scanner run fetched 2026-09-14): /open 15 (15 <- 14) and /close 30 (23 <- 22, the RADIUS preset)
    reproduce IPL's exports with 0 mismatching voxels and identical grids on PFJ-0be66a_R, PFJ-42293d_L and PFJ-6f5538_R;
    /close 15 (12 <- 11) and /close 50 (23 <- 22) likewise on PFJ-411dfd_R; and PFJ-411dfd_R 15 <- 14 -- the residual of
    the pre-mechanism rule (3 IPL-only voxels on the last slice) -- is exact under the mirror-margin mechanism
    implemented in open_ on 2026-09-14 evening (17,696,233 = 17,696,233).  The discrimination on that volume, from
    the same evening's run, is pinned so that a change of
    the rule or of the margin depth is deliberate: the pre-mechanism rule (the two chained primitives) differs from
    IPL by exactly the 3 voxels at global (x, y, z) (1469, 175, 167), (1471, 178, 167), (1470, 179, 167) and by
    nothing else; the mirror margin at depth N + 1 / N + 3 / N differs by 19 / 2 / 48; the edge-exclusive mirror by
    28 (27 ours-only, 1 IPL-only); edge replication, wrap, an object margin and a background margin by 14,423 /
    76,168 / 28,338 / 238,083.  Probe 20 (predictions written 2026-09-14 18:12) is the prospective confirmation."""
    if _p17("PFJ-411dfd_R", "14_bbc") is None or _p17("PFJ-0be66a_R", "14_bbc") is None:
        pytest.skip("probe-17 exports not found")
    table = [("PFJ-0be66a_R", "14_bbc", "15_open15", ops.open_, 15, 20_163_367),
             ("PFJ-0be66a_R", "22_trabadd", "23_close50", ops.close, 30, 20_302_109),
             ("PFJ-42293d_L", "14_bbc", "15_open15", ops.open_, 15, 21_006_474),
             ("PFJ-42293d_L", "22_trabadd", "23_close50", ops.close, 30, 21_606_958),
             ("PFJ-6f5538_R", "14_bbc", "15_open15", ops.open_, 15, 1_763_947),
             ("PFJ-6f5538_R", "22_trabadd", "23_close50", ops.close, 30, 1_954_987),
             ("PFJ-411dfd_R", "14_bbc", "15_open15", ops.open_, 15, 17_696_233),
             ("PFJ-411dfd_R", "11_dil3", "12_close15", ops.close, 15, 18_578_278),
             ("PFJ-411dfd_R", "22_trabadd", "23_close50", ops.close, 50, 17_989_866)]
    bad = {}
    for subj, tin, tout, fn, n, count in table:
        vin, ref = _p17(subj, tin), _p17(subj, tout)
        assert vin is not None and ref is not None, (subj, tin, tout)
        ours = fn(vin, n)
        mism, same = _same(ours, ref)
        if mism or not same or int((ref["data"] != 0).sum()) != count or int((ours["data"] != 0).sum()) != count:
            bad[(subj, tout)] = (mism, same, int((ours["data"] != 0).sum()), int((ref["data"] != 0).sum()))
    assert bad == {}, bad
    vin, ref = _p17("PFJ-411dfd_R", "14_bbc"), _p17("PFJ-411dfd_R", "15_open15")
    assert vin["dim"] == (721, 290, 168) and vin["pos"] == (856, 24, 0) and ref["dim"] == vin["dim"] and ref["pos"] == vin["pos"]
    assert (vin["data"] != 0)[167, [151, 154, 155], [613, 615, 614]].all()          # object in 14, last slice
    M, b = vin["data"] != 0, ref["data"] != 0
    assert int(b.sum()) == 17_696_233
    mech = ops.open_(vin, 15)["data"] != 0
    assert np.array_equal(mech, b) and np.array_equal(mech, ops._open_padded(M, 15, 17, "symmetric"))
    old = old_rule_open(vin, 15)["data"] != 0
    assert np.array_equal(old, ops._open_padded(M, 15, 17, "open_boundary"))
    assert int(old.sum()) == 17_696_230 and int((old & ~b).sum()) == 0
    assert sorted((int(x) + 856, int(y) + 24, int(z)) for z, y, x in np.argwhere(b & ~old)) == P17_RESIDUAL
    assert mech[167, [151, 154, 155], [613, 615, 614]].all()                        # the 3 voxels, restored
    got = {(mode, margin): _mismatch_counts(ops._open_padded(M, 15, margin, mode), ref)
           for mode, margin in (("symmetric", 16), ("symmetric", 18), ("symmetric", 15), ("symmetric", 30), ("reflect", 17),
                                ("edge", 17), ("wrap", 17), ("object", 17), ("background", 17))}
    assert got == {("symmetric", 16): (19, 19, 0), ("symmetric", 18): (2, 0, 2), ("symmetric", 15): (48, 48, 0),
                   ("symmetric", 30): (3, 0, 3), ("reflect", 17): (28, 27, 1), ("edge", 17): (14_423, 14_422, 1),
                   ("wrap", 17): (76_168, 3_121, 73_047), ("object", 17): (28_338, 28_338, 0),
                   ("background", 17): (238_083, 0, 238_083)}, got


@pytest.mark.slow
def test_probe18_exports_are_reproduced_whole_volume():
    """Every probe-18 export of IPL (scanner run 2026-09-13) from the uploaded char phantom and, for seg_gauss,
    from IPL's own masked / cut short phantom (SGBBC): 0 mismatching voxels and identical grids.  This pins the
    three corrections (SUB / SUB2: -127; CBT1 / CBT2: all face-touching components joined; RANK1..5: the tie
    order) and the confirmed readings (SW, NRMIN, NRMAX, ERO3, ERO1, DIL15, DIL1, CLOSE3, OPEN3, SG)."""
    upload = os.path.join(P18_ROOT, "upload", "x2420448_p18cl.aim")
    if not os.path.exists(upload) or _p18("RANK1") is None:
        pytest.skip("probe-18 phantom / exports not found")
    a = read_aim(upload)
    cl = ops.vol(np.ascontiguousarray(a["data"]), a["dim"], a["pos"])
    assert cl["dim"] == (743, 348, 168) and cl["pos"] == (795, 94, 168) and int((cl["data"] != 0).sum()) == 105_201
    assert _same(cl, _p18("RT")) == (0, True)

    def crop(v, z0, nz, x0=0, nx=None, y0=0, ny=None):          # /sub_get -global_pos_flag false
        nx = v["dim"][0] - x0 if nx is None else nx
        ny = v["dim"][1] - y0 if ny is None else ny
        d = np.ascontiguousarray(np.asarray(v["data"])[z0:z0 + nz, y0:y0 + ny, x0:x0 + nx])
        return ops.vol(d, (nx, ny, nz), (v["pos"][0] + x0, v["pos"][1] + y0, v["pos"][2] + z0))

    cbin = crop(cl, 78, 20)
    sain = crop(cl, 100, 10)
    sb = ops.set_value(crop(sain, 0, 10, 110, 20, 110, 20), 127, 127)
    checks = {
        "CBIN": cbin, "SAIN": sain, "SB": sb,
        "SW": ops.cl_slicewise_extractow(cl, 50.0, 100.0),
        "NRMIN": ops.cl_nr_extract(cl, 800, 0), "NRMAX": ops.cl_nr_extract(cl, 1, 800),
        "CBT1": ops.cl_ow_rank_extract(cbin, 1, 1, connect_boundary=True),
        "CBT2": ops.cl_ow_rank_extract(cbin, 2, 2, connect_boundary=True),
        "CBF1": ops.cl_ow_rank_extract(cbin, 1, 1, connect_boundary=False),
        "SUB": ops.subtract_aims(sain, sb), "SUB2": ops.subtract_aims(sb, sain), "ADD": ops.add_aims(sain, sb),
        "ERO3": ops.erosion(cl, 3), "ERO1": ops.erosion(cl, 1), "DIL15": ops.dilation(cl, 15), "DIL1": ops.dilation(cl, 1),
        "CLOSE3": ops.close(cl, 3), "OPEN3": ops.open_(cl, 3),
    }
    for k in range(1, 6):
        checks[f"RANK{k}"] = ops.cl_ow_rank_extract(cl, k, k)
    sgbbc = _p18("SGBBC")
    if sgbbc is not None:
        assert sgbbc["dim"] == (738, 343, 168) and sgbbc["pos"] == (798, 97, 168)
        checks["SG"] = ops.seg_gauss(ops.vol(np.ascontiguousarray(sgbbc["data"]), sgbbc["dim"], sgbbc["pos"]), 2.0, 3, 4524, 17173)
    bad = {}
    for tag, ours in checks.items():
        ref = _p18(tag)
        assert ref is not None, f"export {tag} missing"
        n, same = _same(ours, ref, valued=tag in ("SUB", "SUB2", "ADD"))
        if n or not same:
            bad[tag] = (n, same, tuple(ours["dim"]), tuple(ours["pos"]), tuple(ref["dim"]), tuple(ref["pos"]))
    assert bad == {}, f"exports not reproduced: {bad}"
    assert checks["DIL15"]["dim"] == (775, 380, 200) and checks["DIL15"]["pos"] == (779, 78, 152)
    assert checks["DIL1"]["dim"] == (747, 352, 172) and checks["DIL1"]["pos"] == (793, 92, 166)
    assert int((checks["SUB"]["data"] == -127).sum()) == 3000 and int((checks["SUB"]["data"] == 127).sum()) == 3000
    assert int((checks["ADD"]["data"] == 127).sum()) == 7000 and checks["ADD"]["data"].max() == 127
    assert int((checks["CBT1"]["data"] != 0).sum()) == 2500 and int((checks["CBT2"]["data"] != 0).sum()) == 1500
    assert [int((checks[f"RANK{k}"]["data"] != 0).sum()) for k in range(1, 6)] == [20000, 20000, 15000, 10000, 10000]


# ------------------------------------------------------------------------------------------- probe 19 (slow)
P19_ROOT = os.environ.get("IPLDT_PROBE19_ROOT", lab_path("Python/scripts/IPL/probes/p19_open_halves"))
P19_RESIDUAL = [(1469, 175, 167), (1470, 179, 167), (1471, 178, 167)]        # sorted global (x, y, z), the last slice


def _p19(tag):
    fs = vms_versions(os.path.join(P19_ROOT, "aims_and_logs"), f"X5143651_P19_{tag}.AIM")
    if not fs:
        return None
    a = read_aim(fs[-1])
    return ops.vol(np.ascontiguousarray(a["data"]), a["dim"], a["pos"])


def _restrict(v, like):
    """v pasted by global position onto like's grid (IPL's export restricted to a sub grid, or a grown dilation
    restricted to its input grid)."""
    return ops.mask_vol(ops.on_grid(v, like["dim"], like["pos"]) != 0, like["dim"], like["pos"])


def _ipl_only(ours, ref):
    """Sorted global (x, y, z) of the voxels set in ref (IPL) and not in ours; asserts equal grids and no ours-only
    voxel, so the returned list is the whole difference."""
    assert (tuple(ours["dim"]), tuple(ours["pos"])) == (tuple(ref["dim"]), tuple(ref["pos"]))
    a, b = ours["data"] != 0, ref["data"] != 0
    assert int((a & ~b).sum()) == 0
    p = ref["pos"]
    return sorted((int(x) + p[0], int(y) + p[1], int(z) + p[2]) for z, y, x in np.argwhere(b & ~a))


@pytest.mark.slow
def test_probe19_open_halves_primitives_are_exact_and_the_mechanism_reproduces_open():
    """Probe 19 (scanner run fetched 2026-09-14) on PFJ-411dfd_R's stage 14 (X5143651_P17_14_BBC, 721x290x168 @ 856,24,0),
    the input of the /open 15 residual.  (1) /erosion 15 (ERO15) == erosion(14): 13,640,114 set voxels, 0 mismatches;
    a second run (ERO15B) is identical.  (2) /dilation 15 -use_previous_margin false on IPL's ero15 (DIL15) ==
    dilation(ero15) on the grown grid 753x322x200 @ 840,8,-16 (N + 1 = 16 per side): 20,637,201, 0 mismatches.
    (3) /open 15 run again (OPEN15) == the P17 stage-15 export (deterministic) == open_(14) under the mirror-margin
    mechanism (implemented 2026-09-14 evening): 0 mismatches, 17,696,233 set voxels; the pre-mechanism rule (the two
    chained primitives) differs from it by exactly the 3 pinned IPL-only voxels, and DIL15 restricted to the 14 grid
    equals the pre-mechanism rule exactly -- so IPL's chained primitives differ from IPL's /open by exactly those 3
    voxels: the residual is inside /open's internal chaining (the erosion's margin buffer reused), not in either
    primitive.  (4) -use_previous_margin true (DIL15PM on ERO15B) writes the input grid shrunk by 16 per side
    (689x258x136 @ 872,40,16) with the standard dilation's content restricted to it (documented, not implemented).
    (5) On IPL's /sub_get copies of 14 (the crop convention pos_out = pos_in + pos_local): the sub erosions and
    dilations are exact; open_ reproduces IPL's /open 15 on suba (z 100..167), subb (z 0..166, the last slice removed)
    and subc (x 400..720, y 40..289, z 100..167) exactly, where the pre-mechanism rule misses the same 3 voxels on suba
    and subc and none on subb; the edge-exclusive mirror / object / wrap margins differ from IPL on suba by 9 / 16,635
    / 61,036; IPL's full open restricted to each sub grid differs from IPL's sub open by 2,600 / 4,412 / 719 voxels,
    exactly the counts the rule gives for open_(full) restricted vs open_(sub) -- /open's cut-face behaviour is the
    rule's."""
    v14 = _p17("PFJ-411dfd_R", "14_bbc")
    if v14 is None or _p19("OPEN15") is None or _p19("SUBA_IN") is None:
        pytest.skip("probe-17 stage 14 or probe-19 exports not found")
    ero, dil, opn, erob, pm = (_p19(t) for t in ("ERO15", "DIL15", "OPEN15", "ERO15B", "DIL15PM"))
    assert v14["dim"] == (721, 290, 168) and v14["pos"] == (856, 24, 0)
    # (1) the erosion half
    assert _same(ops.erosion(v14, 15), ero) == (0, True)
    assert int((ero["data"] != 0).sum()) == 13_640_114
    assert _same(ero, erob) == (0, True)
    # (2) the dilation half on IPL's own erosion, with the grown grid
    D = ops.dilation(ero, 15)
    assert D["dim"] == (753, 322, 200) and D["pos"] == (840, 8, -16)
    assert dil["dim"] == D["dim"] and dil["pos"] == D["pos"]
    assert _same(D, dil) == (0, True)
    assert int((dil["data"] != 0).sum()) == 20_637_201
    # (3) /open: deterministic, = open_ (the mechanism), = the chained primitives + exactly the 3 residual voxels
    assert _same(opn, _p17("PFJ-411dfd_R", "15_open15")) == (0, True)
    O = ops.open_(v14, 15)
    assert int((O["data"] != 0).sum()) == 17_696_233 and int((opn["data"] != 0).sum()) == 17_696_233
    assert _same(O, opn) == (0, True) and _ipl_only(O, opn) == []
    O_old = old_rule_open(v14, 15)
    assert int((O_old["data"] != 0).sum()) == 17_696_230
    assert _ipl_only(O_old, opn) == P19_RESIDUAL
    dil_inside = _restrict(dil, v14)
    assert _same(dil_inside, O_old) == (0, True)
    assert _ipl_only(dil_inside, opn) == P19_RESIDUAL
    assert (v14["data"] != 0)[167, [151, 154, 155], [613, 615, 614]].all()          # object in 14, eroded by both
    assert not (ero["data"] != 0)[167, [151, 154, 155], [613, 615, 614]].any()
    assert (O["data"] != 0)[167, [151, 154, 155], [613, 615, 614]].all()             # restored by the margin seeds
    # (4) -use_previous_margin true: the shrunk grid, content = the standard dilation restricted to it
    if pm is not None:
        assert pm["dim"] == (689, 258, 136) and pm["pos"] == (872, 40, 16)
        assert int((pm["data"] != 0).sum()) == 14_356_093
        assert _same(_restrict(dil, pm), pm) == (0, True)
        assert _same(_restrict(opn, pm), pm) == (0, True)
    # (5) the /sub_get copies: exact primitives, open_ exact, the pre-mechanism residual follows the last slice, cut
    #     faces as the rule; the alternatives' counts on suba
    subs = {"SUBA": ((0, 0, 100), (721, 290, 68), P19_RESIDUAL, 7_196_729, 2_600),
            "SUBB": ((0, 0, 0), (721, 290, 167), [], 17_595_873, 4_412),
            "SUBC": ((400, 40, 100), (321, 250, 68), P19_RESIDUAL, 3_247_584, 719)}
    for s, ((x0, y0, z0), dim, expect_old, count, locality) in subs.items():
        vin, se, sd, so = (_p19(f"{s}_{t}") for t in ("IN", "ERO15", "DIL15", "OPEN15"))
        assert None not in (vin, se, sd, so), s
        crop = ops.vol(np.ascontiguousarray(v14["data"][z0:z0 + dim[2], y0:y0 + dim[1], x0:x0 + dim[0]]), dim,
                       (v14["pos"][0] + x0, v14["pos"][1] + y0, v14["pos"][2] + z0))
        assert _same(crop, vin) == (0, True), s
        assert _same(ops.erosion(vin, 15), se) == (0, True), s
        assert _same(ops.dilation(se, 15), sd) == (0, True), s
        assert sd["dim"] == tuple(d + 32 for d in dim) and sd["pos"] == tuple(p - 16 for p in vin["pos"]), s
        Os = ops.open_(vin, 15)
        assert int((so["data"] != 0).sum()) == count, s
        assert _same(Os, so) == (0, True) and _ipl_only(Os, so) == [], s
        assert _ipl_only(old_rule_open(vin, 15), so) == expect_old, s
        n_ipl, _ = _same(_restrict(opn, so), so)
        n_ours, _ = _same(_restrict(O, so), Os)
        assert n_ipl == n_ours == locality, (s, n_ipl, n_ours)
        if s == "SUBA":
            Ms = vin["data"] != 0
            got = {mode: _mismatch_counts(ops._open_padded(Ms, 15, 17, mode), so) for mode in ("reflect", "object", "wrap", "open_boundary")}
            assert got == {"reflect": (9, 8, 1), "object": (16_635, 16_635, 0), "wrap": (61_036, 2_176, 58_860),
                           "open_boundary": (3, 0, 3)}, got


# ------------------------------------------------------------------------------------------- probe 20 (slow)
P20_ROOT = os.environ.get("IPLDT_PROBE20_ROOT", lab_path("Python/scripts/IPL/probes/p20_open_mechanism"))
P20_RESIDUAL = [(1469, 175, 167), (1470, 179, 167), (1471, 178, 167)]        # sorted global (x, y, z), the last slice


def _p20(tag):
    """IPL's probe-20 export X5143651_P20_<TAG>.AIM (the scanner run fetched 2026-09-15), or None."""
    fs = vms_versions(os.path.join(P20_ROOT, "aims_and_logs"), f"X5143651_P20_{tag}.AIM")
    if not fs:
        return None
    a = read_aim(fs[-1])
    return ops.vol(np.ascontiguousarray(a["data"]), a["dim"], a["pos"])


def _p20_upload(name):
    """One of the 14 phantoms uploaded to the scanner for probe 20 (the input of <NAME>_OPEN15), or None."""
    fs = vms_versions(os.path.join(P20_ROOT, "upload"), f"x5143651_p20_{name}.aim")
    if not fs:
        return None
    a = read_aim(fs[-1])
    return ops.vol(np.ascontiguousarray(a["data"]), a["dim"], a["pos"])


def _crop(v, pos_local, dim):
    """IPL's /sub_get: the local box (pos_local, dim) of v, written at pos_out = pos_in + pos_local."""
    x0, y0, z0 = pos_local
    return ops.vol(np.ascontiguousarray(v["data"][z0:z0 + dim[2], y0:y0 + dim[1], x0:x0 + dim[0]]), dim,
                   tuple(v["pos"][i] + pos_local[i] for i in range(3)))


def sym_z_open(M, n):
    """The 'z faces only' reading of /open (p20_rules 'sym_z'): the z margin mirrored, the x / y margins as the
    pre-mechanism rule (an open boundary in the erosion, a background margin in the dilation).  No oracle before
    probe 20 separated it from the six-face mirror; the transposed phantoms and cavex do."""
    m, t = n + 2, ops.metric11_threshold(n)
    Mp = np.pad(np.asarray(M, bool), ((m, m), (0, 0), (0, 0)), mode="symmetric")
    surv = Mp & (ops.chamfer_dt_345(ops._u8(Mp)) >= t)
    sp = np.pad(surv, ((0, 0), (m, m), (m, m)), constant_values=False)
    return (sp | (ops.chamfer_dt_345(ops._u8(~sp)) < t))[m:-m, m:-m, m:-m]


def _face_discs(a, face):
    """The discs inside the ring trenches of a plate-over-cave phantom, located IN THE PHANTOM ITSELF (not read
    from the probe's manifest): the 4-connected components of the face slice that have 249 voxels, ordered along
    the axis the blocks lie on (increasing T).  Returns a list of bool volumes on a's grid."""
    slc = a[-1] if face == "z1" else a[:, :, -1]
    lab, k = ndi.label(slc, ndi.generate_binary_structure(2, 1))
    counts = np.bincount(lab.ravel())
    discs = [i for i in range(1, k + 1) if counts[i] == 249]
    axis = 1 if face == "z1" else 0                                  # cave: blocks along x; cavex: along z
    out = []
    for L in sorted(discs, key=lambda L: float(np.argwhere(lab == L)[:, axis].mean())):
        d = np.zeros_like(a)
        if face == "z1":
            d[-1] = lab == L
        else:
            d[:, :, -1] = lab == L
        out.append(d)
    return out


@pytest.mark.slow
def test_probe20_n_scan_and_the_sub_get_crops():
    """PROBE 20, blocks 1 and 2 (the prospective confirmation of the /open mechanism; predictions written
    2026-09-14 18:12:42 BEFORE the scanner run, run fetched 2026-09-15).
    (1) THE N-SCAN on PFJ-411dfd_R's stage 14 (X5143651_P17_14_BBC, 721x290x168 @ 856,24,0): /open N for N = 11, 12,
    13, 14, 16, 17, 18 (exports OPEN11 .. OPEN18) equals open_ at 0 mismatching voxels with identical grids at
    every N, while the PRE-MECHANISM rule (the two chained primitives) leaves exactly 86 / 62 / 34 / 51 / 0 / 0 / 0
    IPL-only voxels and never an ours-only voxel -- digit for digit the prospective prediction.  At N = 11 / 12 the
    alternatives are refuted by reflect 213 / 162, edge 8,018 / 8,491, object 14,981 / 16,617, wrap 54,881 /
    61,216, background 143,717 / 166,886.
    (2) IPL'S OWN /sub_get CROPS (sreg / z150 / xy3 / x1cut / y1cut / z156): every <tag>_IN equals the crop of
    stage 14 at the same global position, and open_ reproduces <tag>_OPEN15 at 0 mismatches where the
    pre-mechanism rule is off by 3 / 3 / 8 / 30 / 8 / 36.  z156 is 12 slices deep -- THINNER than the margin
    N + 2 = 17, so numpy's 'symmetric' pad repeats the reflection -- and IPL matches it exactly (1,240,454 set
    voxels), which is why open_ no longer warns there."""
    v14 = _p17("PFJ-411dfd_R", "14_bbc")
    if v14 is None or _p20("OPEN11") is None:
        pytest.skip("probe-17 stage 14 or probe-20 exports not found")
    assert v14["dim"] == (721, 290, 168) and v14["pos"] == (856, 24, 0)
    M = v14["data"] != 0
    assert int(M.sum()) == 17_962_446
    counts = {11: 17_833_531, 12: 17_808_269, 13: 17_777_932, 14: 17_735_682,
              16: 17_648_505, 17: 17_626_174, 18: 17_601_750}
    old_only = {11: 86, 12: 62, 13: 34, 14: 51, 16: 0, 17: 0, 18: 0}
    got_old, got_mism = {}, {}
    for n in (11, 12, 13, 14, 16, 17, 18):
        ref = _p20(f"OPEN{n}")
        assert ref is not None, n
        ours = ops.open_(v14, n)
        got_mism[n] = _same(ours, ref)
        assert int((ref["data"] != 0).sum()) == counts[n] and int((ours["data"] != 0).sum()) == counts[n], n
        got_old[n] = _mismatch_counts(ops._open_padded(M, n, n + 2, "open_boundary"), ref)
    assert got_mism == {n: (0, True) for n in counts}, got_mism
    assert got_old == {n: (old_only[n], 0, old_only[n]) for n in counts}, got_old
    alt = {(mode, n): _mismatch_counts(ops._open_padded(M, n, n + 2, mode), _p20(f"OPEN{n}"))[0]
           for n in (11, 12) for mode in ("reflect", "edge", "object", "wrap", "background")}
    assert alt == {("reflect", 11): 213, ("reflect", 12): 162, ("edge", 11): 8_018, ("edge", 12): 8_491,
                   ("object", 11): 14_981, ("object", 12): 16_617, ("wrap", 11): 54_881, ("wrap", 12): 61_216,
                   ("background", 11): 143_717, ("background", 12): 166_886}, alt
    subs = {"SREG": ((560, 100, 130), (121, 101, 38), 291_699, 3),
            "Z150": ((0, 0, 150), (721, 290, 18), 1_873_509, 3),
            "XY3": ((560, 100, 130), (59, 59, 38), 132_200, 8),
            "X1CUT": ((560, 100, 130), (59, 101, 38), 170_348, 30),
            "Y1CUT": ((560, 100, 130), (121, 59, 38), 248_229, 8),
            "Z156": ((0, 0, 156), (721, 290, 12), 1_240_454, 36)}
    for tag, (pos_local, dim, count, old) in subs.items():
        vin, ref = _p20(f"{tag}_IN"), _p20(f"{tag}_OPEN15")
        assert vin is not None and ref is not None, tag
        assert vin["dim"] == dim, tag
        assert _same(_crop(v14, pos_local, dim), vin) == (0, True), tag
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ours = ops.open_(vin, 15)
            assert [x for x in w if issubclass(x.category, ops.ThinVolumeWarning)] == [], tag
        assert _same(ours, ref) == (0, True), tag
        assert int((ref["data"] != 0).sum()) == count, tag
        assert _mismatch_counts(ops._open_padded(vin["data"] != 0, 15, 17, "open_boundary"), ref) == (old, 0, old), tag
    thin = _p20("Z156_IN")
    assert thin["dim"][2] == 12 < 17 and ops._mirror_repeat_axes(thin["data"].shape, 17, ("symmetric",) * 3) == ()


@pytest.mark.slow
def test_probe20_boundary_flag_oracles():
    """PROBE 20, block 1: IPL's -continuous_at_boundary on PFJ-411dfd_R's stage 14, the oracles for the flag added to
    dilation() and close() on 2026-09-15.  /close 15 with '0 0 0' (CLOSE15C0) is the default at 0 mismatching
    voxels (17,964,657 set) and with '1 1 1' (CLOSE15C1) the mirrored margin at 0 (18,003,002); /dilation 15 with
    '1 1 1' (DIL15C1) and with '0 0 1' (DIL15C001) match at 0 ON THE WHOLE WRITTEN GRID, the 16 written margin
    layers included -- 27,387,249 and 27,062,074 set voxels on 753x322x200 @ 840,8,-16 -- so the written margin
    exhibits the fill itself.  The contrasts: IPL's two close exports differ from each other by 38,345 voxels, so
    'IPL ignores the flag and always mirrors' is refuted by 38,345; IPL's two dilation exports differ by 325,175;
    the default (empty border) is 787,238 from DIL15C1; reading the three flags in z, y, x order instead (mirror on
    x only) is 598,104 from DIL15C001; and the edge-EXCLUSIVE mirror is 70,802 from DIL15C1 -- the discrimination
    /close cannot make, since 'symmetric' and 'reflect' both match CLOSE15C1 at 0."""
    v14 = _p17("PFJ-411dfd_R", "14_bbc")
    c0, c1, d1, d001 = (_p20(t) for t in ("CLOSE15C0", "CLOSE15C1", "DIL15C1", "DIL15C001"))
    if v14 is None or c0 is None or d1 is None:
        pytest.skip("probe-17 stage 14 or probe-20 flag exports not found")
    assert _same(ops.close(v14, 15), c0) == (0, True)
    assert _same(ops.close(v14, 15, (0, 0, 0)), c0) == (0, True)
    assert _same(ops.close(v14, 15, (1, 1, 1)), c1) == (0, True)
    assert int((c0["data"] != 0).sum()) == 17_964_657 and int((c1["data"] != 0).sum()) == 18_003_002
    assert _same(ops.dilation(v14, 15, (1, 1, 1)), d1) == (0, True)
    assert _same(ops.dilation(v14, 15, (0, 0, 1)), d001) == (0, True)
    for d in (d1, d001):
        assert d["dim"] == (753, 322, 200) and d["pos"] == (840, 8, -16)
    assert int((d1["data"] != 0).sum()) == 27_387_249 and int((d001["data"] != 0).sum()) == 27_062_074
    assert _same(c0, c1)[0] == 38_345                                    # IPL vs IPL: the flag is not ignored
    assert _same(ops.close(v14, 15), c1)[0] == 38_345                    # 'always mirrored' refuted
    assert _same(ops.close(v14, 15, 1), c0)[0] == 38_345
    assert _same(d1, d001)[0] == 325_175                                 # IPL vs IPL: the flag is per axis
    assert _same(ops.dilation(v14, 15), d1)[0] == 787_238                # the empty border, refuted at '1 1 1'
    assert _same(ops.dilation(v14, 15, (1, 0, 0)), d001)[0] == 598_104   # the flags are x y z, not z y x
    t = ops.metric11_threshold(15)
    refl = np.pad(v14["data"] != 0, 17, mode="reflect")
    refl = (refl | (ops.chamfer_dt_345(ops._u8(~refl)) < t))[1:-1, 1:-1, 1:-1]
    assert _same(ops.mask_vol(refl, d1["dim"], d1["pos"]), d1)[0] == 70_802      # edge-EXCLUSIVE mirror, refuted


@pytest.mark.slow
def test_probe20_cave_discs_and_the_six_face_mirror():
    """PROBE 20, block 3: the purpose-built phantoms (predicted before the run).
    (1) THE CAVE T-SCAN.  Plates of thickness T touching a face over a background cave, with a 1-deep ring trench
    in the face slice around each block centre, so the 249-voxel disc inside the trench is eroded and NOTHING
    in-volume can restore it.  IPL restores 0 / 0 / 117 / 117 disc voxels at T = 15 / 16 / 17 / 18 (cave, the z1
    face) and 0 / 117 at T = 16 / 17 (cavex, the x1 face), and open_ restores exactly the same -- T = 17 = N + 2 is
    the first thickness at which the truncation of the mirror bites, so the T-scan reads the MARGIN DEPTH off the
    data.  The pre-mechanism rule restores 0 everywhere and the edge-exclusive mirror 0 / 0 / 0 / 249 and 0 / 0.
    The discs are located here by labelling the face slice of the phantom itself, not read from the probe's
    manifest.
    (2) THE SIX FACES.  cavex (the same construction transposed so the plate face is the x1 FACE) and the four
    transposed crops tzx / tzx0 / tzy / tzy0 (the residual geometry on the x1 / x0 / y1 / y0 face) are reproduced at
    0 mismatches, while a z-faces-only mirror is refuted by 138 voxels (cavex) and 3 each (the crops) and the
    edge-exclusive mirror by 138 and 9.  All 14 uploaded phantoms round-trip (<NAME>_RT) at 0 and are opened at 0."""
    if _p20_upload("cave") is None or _p20("CAVE_OPEN15") is None:
        pytest.skip("probe-20 phantoms not found")
    disc_counts = {}
    for name, face, Ts in (("cave", "z1", (15, 16, 17, 18)), ("cavex", "x1", (16, 17))):
        up = _p20_upload(name)
        ipl = _p20(f"{name.upper()}_OPEN15")
        a = up["data"] != 0
        ours = ops.open_(up, 15)
        assert _same(ours, ipl) == (0, True), name
        o, b = ours["data"] != 0, ipl["data"] != 0
        old = ops._open_padded(a, 15, 17, "open_boundary")
        refl = ops._open_padded(a, 15, 17, "reflect")
        discs = _face_discs(a, face)
        assert len(discs) == len(Ts) and all(int(d.sum()) == 249 for d in discs), name
        disc_counts[name] = dict(ipl=[int((b & d).sum()) for d in discs], ours=[int((o & d).sum()) for d in discs],
                                 old=[int((old & d).sum()) for d in discs],
                                 reflect=[int((refl & d).sum()) for d in discs])
        assert not (o & ~a).any(), name                                   # anti-extensive
    assert disc_counts["cave"] == dict(ipl=[0, 0, 117, 117], ours=[0, 0, 117, 117], old=[0, 0, 0, 0],
                                       reflect=[0, 0, 0, 249]), disc_counts["cave"]
    assert disc_counts["cavex"] == dict(ipl=[0, 117], ours=[0, 117], old=[0, 0], reflect=[0, 0]), disc_counts["cavex"]
    six = {}
    for name in ("cavex", "tzx", "tzx0", "tzy", "tzy0"):
        up, ipl = _p20_upload(name), _p20(f"{name.upper()}_OPEN15")
        a = up["data"] != 0
        six[name] = (_mismatch_counts(sym_z_open(a, 15), ipl)[0],
                     _mismatch_counts(ops._open_padded(a, 15, 17, "reflect"), ipl)[0])
    assert six == {"cavex": (138, 138), "tzx": (3, 9), "tzx0": (3, 9), "tzy": (3, 9), "tzy0": (3, 9)}, six
    sets = {"reg": 278_712, "zflip": 291_699, "tzx": 291_699, "tzx0": 291_699, "tzy": 291_699, "tzy0": 291_699,
            "dfill": 294_548, "ddeep": 291_352, "wedge": 291_696, "dup1": 299_728, "dup2": 307_744,
            "obj17": 502_844, "cave": 400_409, "cavex": 256_949}
    bad = {}
    for name, count in sets.items():
        up, rt, opn = _p20_upload(name), _p20(f"{name.upper()}_RT"), _p20(f"{name.upper()}_OPEN15")
        got = (_same(up, rt), _same(ops.open_(up, 15), opn), int((opn["data"] != 0).sum()))
        if got != ((0, True), (0, True), count):
            bad[name] = got
    assert bad == {}, bad


@pytest.mark.slow
def test_probe20_k_scan_and_the_primitives_inside_the_region_phantom():
    """PROBE 20, block 3: the whole effect reproduced inside a 121x101x38 phantom, and the k-scan that switches it
    off.  The uploaded 'reg' phantom is the residual's own region with 20 empty slices above; IPL's /sub_get of its
    first 38 + k slices gives the k-scan (k = 0, 1, 2, 3, 5, 8, 11 .. 17).  open_ matches IPL's K<k>_OPEN15 at 0
    mismatching voxels for EVERY k; k = 0 keeps the 3 residual voxels at global (1469,175,167), (1471,178,167),
    (1470,179,167) (291,699 set voxels) and every k >= 1 loses them (278,712) -- an empty slice above the face
    makes the mirrored margin background, so there is no extra survivor to seed the dilation.  IPL's own primitives
    on the same phantom: K0_ERO15 = erosion() at 0 (192,181 set voxels), K0_DIL15 = dilation() applied to IPL's own
    K0_ERO15 at 0 (713,838 on 153x133x70 @ 1400,108,114), and IPL's K0_OPEN15 exceeds that chain by EXACTLY the
    three residual voxels and by nothing the other way.  Controls: K0_OPEN15 (from the uploaded phantom) equals
    SREG_OPEN15 (IPL's own /sub_get of the same box) at 0."""
    reg = _p20_upload("reg")
    if reg is None or _p20("K0_OPEN15") is None:
        pytest.skip("probe-20 phantoms not found")
    assert reg["dim"] == (121, 101, 58) and reg["pos"] == (1416, 124, 130)
    bad = {}
    for k in (0, 1, 2, 3, 5, 8, 11, 12, 13, 14, 15, 16, 17):
        ref = _p20(f"K{k}_OPEN15")
        assert ref is not None, k
        ours = ops.open_(_crop(reg, (0, 0, 0), (121, 101, 38 + k)), 15)
        p, b = ours["pos"], ours["data"] != 0
        kept = [bool(b[z - p[2], y - p[1], x - p[0]]) for x, y, z in P20_RESIDUAL]
        got = (_same(ours, ref), int((ref["data"] != 0).sum()), kept)
        want = ((0, True), 291_699 if k == 0 else 278_712, [k == 0] * 3)
        if got != want:
            bad[k] = (got, want)
    assert bad == {}, bad
    k0, ero, dil = _p20("K0_OPEN15"), _p20("K0_ERO15"), _p20("K0_DIL15")
    crop0 = _crop(reg, (0, 0, 0), (121, 101, 38))
    assert _same(ops.erosion(crop0, 15), ero) == (0, True) and int((ero["data"] != 0).sum()) == 192_181
    assert _same(ops.dilation(ero, 15), dil) == (0, True) and int((dil["data"] != 0).sum()) == 713_838
    assert dil["dim"] == (153, 133, 70) and dil["pos"] == (1400, 108, 114)
    chain = ops.mask_vol(ops.on_grid(dil, k0["dim"], k0["pos"]) != 0, k0["dim"], k0["pos"])
    assert _ipl_only(chain, k0) == P20_RESIDUAL                          # /open minus IPL's own chained primitives
    assert _same(k0, _p20("SREG_OPEN15")) == (0, True)
