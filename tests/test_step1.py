"""ipldt.step1 (Script 32 STEP 1, the cortical / trabecular separation).

Fast: a synthetic cylinder with a dense shell -- cort and trab partition the periosteal rendering, cort is a
ring on every slice (one 4-connected component around a hole that trab fills), the stage tags are the T16
names, the thresholds come from the processing log, the raw-mask wrapper and the RADIUS preset run, an
empty / disjoint periosteal contour raises before /seg_gauss, numpy-scalar parameters give JSON-native info,
and the wiring of the slicewise stages 24 -> 25 -> 26 -> 27 -> 28 -> 29 is pinned by a spy (they are no-ops
on the cylinder and on PFJ-0be66a).
Slow (needs the T16 test-run exports and the patella folder): the whole chain from the greyscale
and IPL's stage 00 reproduces every exported stage 01..29 and the September evaluation's CORT_MASK /
TRAB_MASK voxel for voxel on identical grids."""
import json
import os
import re
from dataclasses import asdict

import numpy as np
import pytest
from conftest import lab_path, vms_versions
from scipy import ndimage as ndi

from ipldt import ipl_ops as ops
from ipldt.io import read_aim
from ipldt.step1 import RADIUS, STAGES, TIBIA, Step1Params, cort_trab_separation, cort_trab_separation_from_raw

PROCLOG = ("Mu_Scaling                                       8192\n"
           "Density: slope                         1.61907703e+03\n"
           "Density: intercept                    -3.94095001e+02\n")
SHAPE = (40, 90, 90)          # (z, y, x)
CENTRE = 44.5
R_OUT, R_IN = 40.0, 32.0


def _radius():
    yy, xx = np.mgrid[:SHAPE[1], :SHAPE[2]]
    return np.sqrt((yy - CENTRE) ** 2 + (xx - CENTRE) ** 2)


@pytest.fixture(scope="module")
def phantom():
    """grey: a cylinder r <= 40 on all slices, dense shell (12000 native, ~1900 mgHA) for 32 <= r <= 40,
    marrow 1000 native (~150 mgHA) inside, 300 outside; periosteal: the disc, given as the rendering."""
    r = _radius()
    grey2d = np.where(r <= R_OUT, np.where(r >= R_IN, 12000, 1000), 300).astype(np.int16)
    grey = ops.vol(np.broadcast_to(grey2d, SHAPE).copy(), SHAPE[::-1], (100, 200, 300))
    grey["proclog"] = PROCLOG
    per = ops.mask_vol(np.broadcast_to(r <= R_OUT, SHAPE).copy(), SHAPE[::-1], (100, 200, 300))
    return grey, per


def _ring_checks(res, per):
    dim, pos = per["dim"], per["pos"]
    G = ops.on_grid(per, dim, pos) != 0
    C = ops.on_grid(res["cort"], dim, pos) != 0
    T = ops.on_grid(res["trab"], dim, pos) != 0
    assert not (C & T).any(), "cort and trab overlap"
    assert np.array_equal(C | T, G), "cort and trab do not partition the periosteal rendering"
    cz, cy, cx = 20, 44, 44
    s4 = ndi.generate_binary_structure(2, 1)
    for z in range(dim[2]):
        assert not C[z, cy, cx] and T[z, cy, cx]
        n_c = ndi.label(C[z], structure=s4)[1]
        assert n_c == 1, f"slice {z}: cort has {n_c} components"
        holes = ndi.binary_fill_holes(C[z]) & ~C[z]
        assert holes[cy, cx] and np.array_equal(holes, T[z]), f"slice {z}: trab is not the hole of the cort ring"
        n_t = int(T[z].sum())
        assert np.pi * 28 ** 2 < n_t < np.pi * 36 ** 2, f"slice {z}: trab has {n_t} voxels"
    assert C[cz, cy, 44 + 38] and not T[cz, cy, 44 + 38]           # the shell is cortical


def test_synthetic_cylinder_gives_a_cortical_ring_and_a_trabecular_disc(phantom):
    grey, per = phantom
    res = cort_trab_separation(grey, per, TIBIA, keep_stages=True)
    _ring_checks(res, per)
    info = res["info"]
    assert info["thresholds"]["lower_native"] == 4524 and info["thresholds"]["upper_native"] == 17173
    assert info["calibration"]["slope"] == 1619.07703
    assert info["box"] == dict(dim=(80, 80, 40), pos=(105, 205, 300))     # the disc's tight box
    assert tuple(res["stages"]) == STAGES
    assert info["grids"]["01_seggauss"] == ((74, 74, 34), (108, 208, 303))  # valid grid: shrunk by 3
    assert info["grids"]["11_dil3"][0] == tuple(d + 8 for d in info["grids"]["10_erorank"][0])
    assert info["grids"]["12_close15"] == info["grids"]["11_dil3"]
    assert info["grids"]["28_cortfinal"] == (res["cort"]["dim"], res["cort"]["pos"])
    assert res["cort"]["dim"] == (80, 80, 40) and res["cort"]["pos"] == (105, 205, 300)
    assert res["trab"]["dim"] == per["dim"] and res["trab"]["pos"] == per["pos"]
    assert info["counts"]["28_cortfinal"] == int(np.count_nonzero(res["cort"]["data"]))
    assert info["counts"]["peel"][0] == int(np.count_nonzero(per["data"]))
    assert info["counts"]["peel"][6] < info["counts"]["peel"][0]
    assert info["timings"]["total"] > 0 and set(STAGES) <= set(info["timings"])
    assert res["cort"]["data"].dtype == np.uint8 and set(np.unique(res["cort"]["data"])) == {0, 127}


def test_keep_stages_false_returns_the_same_masks(phantom):
    grey, per = phantom
    a = cort_trab_separation(grey, per, TIBIA, keep_stages=True)
    b = cort_trab_separation(grey, per, TIBIA, keep_stages=False)
    assert b["stages"] is None
    assert np.array_equal(a["cort"]["data"], b["cort"]["data"]) and np.array_equal(a["trab"]["data"], b["trab"]["data"])


def test_stage_02_negatives_are_the_seg_gauss_voxels_outside_the_contour_and_stage_03_removes_them(phantom):
    """/subtract_aims stores -127 where only input 2 is set (test run 18).  On the cylinder the dense shell smooths
    above 4524 up to 3 voxels outside the disc, so 02 = 00 - 01 carries -127 there (a ring outside r = 40);
    info['negatives'] counts them for every int8 stage, 03 = 02 AND peel6(00) removes all of them (values 0 /
    127 from 03 on), the later differences (in2 a subset of in1) carry none, and TRAB (29) is int8 0 / 127."""
    grey, per = phantom
    res = cort_trab_separation(grey, per, TIBIA, keep_stages=True)
    st, info = res["stages"], res["info"]
    int8_tags = ("02_trab0", "07_trab1", "16_corners", "22_trabadd", "26_cort", "29_trabfinal")
    assert set(info["negatives"]) >= set(int8_tags) and all(st[t]["data"].dtype == np.int8 for t in int8_tags)
    assert set(info["negatives"]) <= {t for t in STAGES if st[t]["data"].dtype.kind == "i"}
    s00, s01, s02 = st["00_all"], st["01_seggauss"], st["02_trab0"]
    dim, pos = ops.union_grid(s00, s01)
    outside = (ops.on_grid(s01, dim, pos) != 0) & (ops.on_grid(s00, dim, pos) == 0)
    assert int(outside.sum()) > 0                                   # the phantom exercises the negative case
    assert np.array_equal(ops.on_grid(s02, dim, pos) < 0, outside)
    assert info["negatives"]["02_trab0"] == int(outside.sum()) == int((s02["data"] == -127).sum())
    assert set(np.unique(s02["data"])) == {-127, 0, 127}
    assert info["counts"]["02_trab0"] == int((s02["data"] != 0).sum())          # -127 counts as set
    assert info["counts"]["02_trab0"] - info["negatives"]["02_trab0"] == int((s02["data"] == 127).sum())
    assert set(np.unique(st["03_peel6"]["data"])) == {0, 127}
    assert all(info["negatives"][t] == 0 for t in int8_tags if t != "02_trab0")
    assert set(np.unique(res["trab"]["data"])) == {0, 127} and res["trab"]["data"].dtype == np.int8
    assert not (ops.on_grid(s02, dim, pos) < 0)[ops.on_grid(s00, dim, pos) != 0].any()   # none inside the contour


def test_explicit_calibration_and_log_callback(phantom):
    grey, per = phantom
    lines = []
    res = cort_trab_separation(grey, per, TIBIA, log=lines.append,
                               calibration=dict(slope=1619.07703, intercept=-394.095001, mu_scaling=8192.0))
    assert res["info"]["thresholds"]["lower_native"] == 4524
    assert len(lines) == len(STAGES) and lines[0].startswith("00_all") and lines[-1].startswith("29_trabfinal")
    assert all(re.search(r"set\s+[\d,]+\s+dim", ln) for ln in lines)
    g2 = dict(grey)
    g2["proclog"] = "no calibration here"
    with pytest.raises(ValueError):
        cort_trab_separation(g2, per, TIBIA)


def test_raw_mask_wrapper(phantom):
    grey, per = phantom
    direct = cort_trab_separation(grey, per, TIBIA)
    raw = ops.vol(np.where(per["data"] != 0, np.uint8(1), np.uint8(0)), per["dim"], per["pos"])
    w = cort_trab_separation_from_raw(grey, raw, TIBIA, render=False)
    assert w["info"]["grids"]["00_all"] == ((80, 80, 40), (105, 205, 300))     # the raster's bounding box
    assert np.array_equal(w["cort"]["data"], direct["cort"]["data"]) and w["cort"]["pos"] == direct["cort"]["pos"]
    T1 = ops.on_grid(w["trab"], per["dim"], per["pos"])
    assert np.array_equal(T1 != 0, direct["trab"]["data"] != 0)
    r = cort_trab_separation_from_raw(grey, raw, TIBIA, render=True)
    _ring_checks(r, ops.mask_vol(ops.on_grid(ops.set_value(r["cort"], 127, 0), per["dim"], per["pos"]) != 0
                                 | (ops.on_grid(r["trab"], per["dim"], per["pos"]) != 0), per["dim"], per["pos"]))
    # grid=(dim, pos): the rendering is pasted onto the given (larger) grid, like IPL's gobj grid with a margin
    g = cort_trab_separation_from_raw(grey, raw, TIBIA, render=False, grid=(per["dim"], per["pos"]))
    assert g["info"]["grids"]["00_all"] == (per["dim"], per["pos"])
    assert g["trab"]["dim"] == per["dim"] and g["trab"]["pos"] == per["pos"]
    assert np.array_equal(g["trab"]["data"], direct["trab"]["data"]) and np.array_equal(g["cort"]["data"], direct["cort"]["data"])


def test_presets():
    assert TIBIA == Step1Params() and TIBIA.corner_min == 200000 and TIBIA.close2 == 50 and TIBIA.peel0 == 6
    assert RADIUS.corner_min == 800 and RADIUS.close2 == 30
    assert {k: v for k, v in RADIUS.__dict__.items() if k not in ("corner_min", "close2")} == \
           {k: v for k, v in TIBIA.__dict__.items() if k not in ("corner_min", "close2")}
    with pytest.raises(Exception):
        TIBIA.close2 = 30                                                   # frozen


def test_radius_preset_runs(phantom):
    grey, per = phantom
    res = cort_trab_separation(grey, per, RADIUS)
    _ring_checks(res, per)
    assert res["info"]["params"]["close2"] == 30 and res["info"]["params"]["corner_min"] == 800


def test_empty_or_disjoint_periosteal_raises(phantom):
    """An empty contour, a contour on a grid disjoint from the greyscale, an empty raw raster and an all-zero
    greyscale inside the contour must fail before /seg_gauss: otherwise the whole greyscale is smoothed and
    the union grids of the later stages grow to the hull of both grids (414 GiB for a far-away contour)."""
    grey, per = phantom
    msg = "selects no non-zero greyscale voxel"
    empty = ops.mask_vol(np.zeros(SHAPE, bool), per["dim"], per["pos"])
    with pytest.raises(ValueError, match=msg):
        cort_trab_separation(grey, empty, TIBIA)
    far = ops.mask_vol(np.ones((10, 10, 10), bool), (10, 10, 10), (5000, 5000, 5000))
    with pytest.raises(ValueError, match=msg):
        cort_trab_separation(grey, far, TIBIA)
    raw_empty = ops.vol(np.zeros(SHAPE, np.uint8), per["dim"], per["pos"])
    with pytest.raises(ValueError, match=msg):
        cort_trab_separation_from_raw(grey, raw_empty, TIBIA)
    zero_grey = dict(grey, data=np.zeros(SHAPE, np.int16))
    with pytest.raises(ValueError, match=msg):
        cort_trab_separation(zero_grey, per, TIBIA)
    assert cort_trab_separation(grey, per, TIBIA)["info"]["counts"]["28_cortfinal"] > 0   # the valid input still runs


def test_numpy_scalar_params_give_json_native_info(phantom):
    """Step1Params built from numpy scalars (np.float32 sigma, np.int64 peel0 ...) runs identically and keeps
    res['info'] JSON-serialisable (asdict values plain, peel-count keys plain ints); an int field given a
    fractional or non-finite value is refused rather than truncated."""
    grey, per = phantom
    p = Step1Params(sigma=np.float32(2.0), lower_mgha=np.float64(500.0), support=np.int64(3),
                    corner_min=np.int64(200000), close2=np.int32(50), peel0=np.int64(6))
    assert p == TIBIA and asdict(p) == asdict(TIBIA)
    assert all(type(x) in (int, float) for x in asdict(p).values())
    res = cort_trab_separation(grey, per, p)
    ref = cort_trab_separation(grey, per, TIBIA)
    assert np.array_equal(res["cort"]["data"], ref["cort"]["data"])
    assert np.array_equal(res["trab"]["data"], ref["trab"]["data"])
    text = json.dumps(res["info"])
    assert json.loads(text)["params"] == TIBIA.record() and Step1Params(**json.loads(text)["params"]) == TIBIA
    assert list(res["info"]["counts"]["peel"]) == [0, 6] and type(list(res["info"]["counts"]["peel"])[1]) is int
    assert Step1Params(close2=30.0) == Step1Params(close2=30) and type(Step1Params(close2=30.0).close2) is int
    for bad in (dict(close2=2.5), dict(peel0=float("nan")), dict(sigma=float("inf"))):
        with pytest.raises(ValueError):
            Step1Params(**bad)


def test_slicewise_stages_are_wired_in_order(phantom, monkeypatch):
    """Stages 25 and 27 are no-ops on the cylinder (one component per slice) and on PFJ-0be66a, so a mis-wiring
    (26 <- 24, 28 <- 26) is invisible to the other tests.  Spy on the command: each call drops one voxel of
    the real result, so 25 != 24 and 27 != 26, and every edge 24->25->26->27->28->29 is checked by data."""
    grey, per = phantom
    real, calls = ops.cl_slicewise_extractow, []

    def spy(v, lo, up, *a, **k):
        out = real(v, lo, up, *a, **k)
        out["data"][tuple(np.argwhere(out["data"] != 0)[0])] = 0
        calls.append((ops.vol(v["data"].copy(), v["dim"], v["pos"]), out, lo, up))
        return out

    monkeypatch.setattr(ops, "cl_slicewise_extractow", spy)
    res = cort_trab_separation(grey, per, TIBIA, keep_stages=True)
    st, c = res["stages"], res["info"]["counts"]
    same = lambda a, b: _mismatch(a, b)[0] == 0                                                  # noqa: E731
    assert len(calls) == 2 and all(cl[2:] == (50.0, 100.0) for cl in calls)
    assert same(calls[0][0], st["24_close50peel"]) and same(calls[0][1], st["25_slicewise"])      # 25 <- 24
    assert same(calls[1][0], st["26_cort"]) and same(calls[1][1], st["27_cortslice"])           # 27 <- 26
    assert c["25_slicewise"] == c["24_close50peel"] - 1 and c["27_cortslice"] == c["26_cort"] - 1
    assert same(st["26_cort"], ops.subtract_aims(st["00_all"], st["25_slicewise"]))             # 26 <- 00 - 25
    assert c["26_cort"] == c["00_all"] - c["25_slicewise"]
    assert same(st["28_cortfinal"], st["27_cortslice"]) and c["28_cortfinal"] == c["27_cortslice"]  # 28 <- 27
    assert same(st["29_trabfinal"], ops.subtract_aims(st["00_all"], st["28_cortfinal"]))         # 29 <- 00 - 28
    assert same(res["cort"], st["28_cortfinal"]) and same(res["trab"], st["29_trabfinal"])


# ------------------------------------------------------------------------------------------- real data (slow)
RUN_ROOT = os.environ.get("IPLDT_RUN_ROOT", lab_path("ipl_test_runs/run15/aims_and_logs"))
T16_TAGS = STAGES[1:]


def _find(folder, pattern):
    """The newest OpenVMS version (numeric ';n' order, conftest.vms_versions) of `pattern` in `folder`, or None."""
    fs = vms_versions(folder, pattern)
    return fs[-1] if fs else None


def _mismatch(ours, ref):
    dim, pos = ops.union_grid(ours, ref)
    a = ops.on_grid(ours, dim, pos) != 0
    b = ops.on_grid(ref, dim, pos) != 0
    return int((a != b).sum()), tuple(ours["dim"]) == tuple(ref["dim"]) and tuple(ours["pos"]) == tuple(ref["pos"])


@pytest.mark.slow
def test_pfj_0be66a_chain_reproduces_ipl_exports_and_masks(data_root):
    grab = os.path.join(data_root, "PFJ-0be66a_R")
    if not os.path.isdir(RUN_ROOT) or not os.path.isdir(grab):
        pytest.skip("T16 test-run exports or PFJ-0be66a_R patella folder not found")
    grey_path = os.path.join(grab, "X2420448.AIM")
    p00 = _find(RUN_ROOT, "X2420448_T16_00_ALL.AIM")
    if p00 is None or not os.path.exists(grey_path):
        pytest.skip("X2420448.AIM or X2420448_T16_00_ALL.AIM missing")
    grey = read_aim(grey_path)
    all_ = read_aim(p00)
    res = cort_trab_separation(grey, all_, TIBIA, keep_stages=True)
    assert res["info"]["thresholds"]["lower_native"] == 4524 and res["info"]["thresholds"]["upper_native"] == 17173
    assert res["info"]["box"] == dict(dim=(738, 343, 168), pos=(798, 97, 168))
    assert res["info"]["counts"]["peel"][6] == 25_892_693
    bad = {}
    for tag in T16_TAGS:
        path = _find(RUN_ROOT, f"X2420448_T16_{tag}.AIM")
        assert path is not None, f"export for {tag} missing"
        n, same = _mismatch(res["stages"][tag], read_aim(path))
        if n or not same:
            bad[tag] = (n, same)
    assert bad == {}, f"stages differing from IPL's exports: {bad}"
    sizes = ops.component_sizes(res["stages"]["04_inv"])
    assert sizes.size == 2145 and int(sizes[0]) == 23_425_237
    for tag, name, count in (("28_cortfinal", "X2420448_CORT_MASK.AIM", 7_154_580), ("29_trabfinal", "X2420448_TRAB_MASK.AIM", 20_329_639)):
        ref = read_aim(os.path.join(grab, name))
        n, same = _mismatch(res["stages"][tag], ref)
        assert n == 0 and same, f"{tag} vs {name}: {n} mismatches, same grid {same}"
        assert res["info"]["counts"][tag] == count
    assert res["cort"]["dim"] == (738, 343, 168) and res["cort"]["pos"] == (798, 97, 168)
    assert res["trab"]["dim"] == (743, 348, 168) and res["trab"]["pos"] == (795, 94, 168)
