"""The launch schedule of the GPU sphere stamping (2026-10-07).

ipldt.gpu.draw_spheres_gpu used to stamp every sphere from one thread looping over the sphere's whole
(2R+1)^3 box, all in one launch; it now sorts the centres by box width, stamps the small ones one thread each
(the same stamp_spheres kernel) and the large ones one thread per (dz, dx) column (stamp_columns), in launches
of at most STAMP_LAUNCH_BUDGET box positions.  Scheduling only, so the maps must not change:

  * the plan (stamp_plan, pure) covers every centre and every column exactly once, within the work bounds;
  * the maps equal, bit for bit, the GPU output of the release before the change (750c94a), frozen in
    tests/data/gpu_stamp_golden.json by tools/gpu_stamp_golden.py, under the default schedule and under
    schedules forced to the extremes; so do dt_thickness / dt_spacing / dt_number (maps, centres, reports) on
    a dense phantom, with and without a gobj, and on a sparse one whose marrow spheres reach D = 235;
  * the maps equal the CPU drawing (ipldt.core.draw_spheres).  At a repeated centre the CPU keeps the last
    diameter written and the GPU the largest, before and after the change; the dt functions never repeat a
    centre (np.nonzero), and the test compares repeated-centre cases with the CPU on the largest D.
"""
import base64
import json
import os
import zlib

import numpy as np
import pytest

import ipldt.gpu as gpu
from ipldt import dt_number, dt_spacing, dt_thickness
from ipldt.core import draw_spheres

from conftest import requires_gpu

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gpu_stamp_golden.json")
FUNCS = {"thickness": dt_thickness, "spacing": dt_spacing, "number": dt_number}
# (small_box, budget): the default, every centre a column of a small launch, and two in between
SCHEDULES = [(None, None), (1, 1 << 16), (9, 1 << 20), (41, 41 ** 3)]


# ------------------------------------------------------------------------------------------- the plan (pure)
def _check_plan(R, budget, small_box):
    """Every invariant of stamp_plan, checked independently of its implementation; returns the plan."""
    plan = gpu.stamp_plan(R, budget, small_box)
    budget = gpu.STAMP_LAUNCH_BUDGET if budget is None else budget
    small_box = gpu.STAMP_SMALL_BOX if small_box is None else small_box
    W = 2 * np.asarray(R, np.int64).ravel() + 1
    assert np.array_equal(np.sort(plan.order), np.arange(W.size)), "order is not a permutation of the centres"
    Ws = W[plan.order]
    assert np.all(np.diff(Ws) >= 0), "centres not sorted by box width"
    ns = plan.n_small
    assert np.all(Ws[:ns] <= small_box) and np.all(Ws[ns:] > small_box)
    WL = Ws[ns:]
    assert plan.col0[0] == 0 and np.array_equal(np.diff(plan.col0), WL * WL), "a large centre owns != W^2 columns"
    nxt = {"point": 0, "columns": 0}
    for kind, a, b, steps in plan.launches:
        assert kind in nxt and a == nxt[kind] and b > a, "launches not consecutive and non-empty"
        nxt[kind] = b
        if kind == "point":
            assert steps == int((np.maximum(Ws[a:b], 1) ** 3).sum())
        else:                       # box positions = sum over the centres of (their columns in [a, b)) * W
            k0 = np.searchsorted(plan.col0, a, side="right") - 1
            k1 = np.searchsorted(plan.col0, b - 1, side="right") - 1
            ks = np.arange(k0, k1 + 1)
            cols = np.minimum(b, plan.col0[ks + 1]) - np.maximum(a, plan.col0[ks])
            assert np.all(cols > 0) and cols.sum() == b - a
            assert steps == int((cols * WL[ks]).sum())
        assert steps <= budget, f"a {kind} launch has {steps} box positions > budget {budget}"
    assert nxt["point"] == ns, "not every small centre is stamped exactly once"
    assert nxt["columns"] == int(plan.col0[-1]), "not every column is stamped exactly once"
    # per thread: a point thread loops over W^3 <= small_box^3 positions, a column thread over W
    assert Ws[:ns].size == 0 or int(Ws[:ns].max()) <= small_box
    return plan


@pytest.mark.parametrize("small_box", [1, 3, 27, 41])
@pytest.mark.parametrize("budget_kind", ["tight", "default"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_plan_invariants_random(seed, small_box, budget_kind):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 1500))
    R = np.where(rng.random(n) < 0.9, rng.integers(-2, 15, n), rng.integers(15, 141, n))
    _check_plan(R, max(small_box ** 3, 1 << 17) if budget_kind == "tight" else None, small_box)


@pytest.mark.parametrize("R", [np.zeros(0, np.int64), np.array([0]), np.array([-1, -3, 0]),
                               np.full(50, 13), np.full(7, 13 + 1), np.arange(200)],
                         ids=["empty", "one_point", "empty_boxes", "all_small", "all_just_large", "ramp"])
def test_plan_edge_cases(R):
    plan = _check_plan(R, 1 << 20, 27)
    if R.size == 0:
        assert plan.launches == [] and plan.n_small == 0


def test_plan_bounds_on_a_fault_sized_block():
    """A sphere set the size of the sparse PCCT blocks that drew Xid 13 faults (D up to 179; over 2e10 box
    positions): the old single launch held all of them and one thread 183^3 = 6.1e6; now no launch holds
    more than STAMP_LAUNCH_BUDGET and no thread more than max(STAMP_SMALL_BOX^3, 183) = 19,683."""
    rng = np.random.default_rng(5)
    D = np.concatenate([rng.integers(1, 25, 150_000), rng.integers(25, 180, 20_000), [179]])
    R = np.floor((D.astype(np.float32) + np.float32(1.0)) * np.float32(0.5)).astype(np.int64) + 1
    plan = _check_plan(R, None, None)
    W = 2 * R + 1
    total = float((W.astype(np.float64) ** 3).sum())
    assert total > 2e10 and int(W.max()) == 183
    assert max(s for *_, s in plan.launches) <= gpu.STAMP_LAUNCH_BUDGET
    assert len(plan.launches) >= total / gpu.STAMP_LAUNCH_BUDGET
    assert max(gpu.STAMP_SMALL_BOX ** 3, int(W.max())) == 19_683


def test_plan_rejects_bad_arguments():
    with pytest.raises(ValueError):
        gpu.stamp_plan([1, 2, 3], budget=26 ** 3, small_box=27)      # a point launch could not hold one centre
    with pytest.raises(ValueError):
        gpu.stamp_plan([1], small_box=0)
    with pytest.raises(ValueError):
        gpu.stamp_plan([2 ** 22], budget=1 << 27)                     # more than 2^62 box positions


# ------------------------------------------------------------------------- maps: equal to 750c94a and the CPU
def _decode(d):
    """An array stored by tools/gpu_stamp_golden.encode: dtype, shape, zlib bytes in base64 (bools bit-packed)."""
    raw = zlib.decompress(base64.b64decode(d["zlib_b64"]))
    if d["dtype"] == "bool":
        bits = np.unpackbits(np.frombuffer(raw, np.uint8), count=int(np.prod(d["shape"])))
        return bits.reshape(d["shape"]).astype(bool)
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN, encoding="utf-8") as fh:
        g = json.load(fh)
    assert g["meta"]["root_commit"].startswith("750c94a"), g["meta"]
    return g


def _case(c):
    cz, cy, cx, d = (_decode(c[k]) for k in ("cz", "cy", "cx", "diam"))
    return cz, cy, cx, d, tuple(c["shape"]), float(c["assign_epsilon"])


def _phantom(g, name):
    ph = g["phantoms"][name]
    return _decode(ph["obj"]), (None if ph["gobj"] is None else _decode(ph["gobj"]))


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("small_box,budget", SCHEDULES, ids=["default", "all_columns", "box9", "box41"])
def test_stamping_equals_750c94a(golden, small_box, budget):
    """Randomised sphere sets (D 0..300, odd shapes, flat volumes, spheres on faces, edges and corners,
    repeated centres, several assign_epsilon): identical to the frozen 750c94a output, dtype included."""
    for i, c in enumerate(golden["stamp"]):
        cz, cy, cx, d, shape, ae = _case(c)
        if small_box is None:
            m = gpu.draw_spheres_gpu(cz, cy, cx, d, shape, ae)
        else:
            m = gpu._draw_spheres_gpu(cz, cy, cx, d, shape, ae, budget=budget, small_box=small_box)
        exp = _decode(c["map"])
        assert m.dtype == exp.dtype == np.int16 and m.shape == exp.shape == shape
        assert np.array_equal(m, exp), f"case {i}: {int((m != exp).sum())} voxels differ from 750c94a"
    assert len({c["kind"] for c in golden["stamp"]}) == 7 and len(golden["stamp"]) >= 40


@requires_gpu
@pytest.mark.gpu
def test_stamping_equals_cpu(golden):
    """The CPU drawing on the same inputs.  Repeated centres keep the largest D on the GPU and the last one
    written on the CPU (unchanged since 750c94a); for those the CPU draws each position once with its largest D."""
    repeated = 0
    for i, c in enumerate(golden["stamp"]):
        cz, cy, cx, d, shape, ae = _case(c)
        m = gpu.draw_spheres_gpu(cz, cy, cx, d, shape, ae)
        lin = np.ravel_multi_index((cz.astype(np.int64), cy.astype(np.int64), cx.astype(np.int64)), shape)
        if np.unique(lin).size < lin.size:            # one entry per position, with its largest D
            repeated += 1
            o = np.lexsort((-d.astype(np.int64), lin))
            first = np.r_[True, lin[o][1:] != lin[o][:-1]]
            z, y, x = np.unravel_index(lin[o][first], shape)
            cpu = draw_spheres(z, y, x, d[o][first].astype(np.int64), shape, ae)
        else:
            cpu = draw_spheres(cz, cy, cx, d.astype(np.int64), shape, ae)
            assert c["gpu_vs_cpu_voxels"] == 0, "750c94a's GPU differed from the CPU on unique centres"
        assert np.array_equal(m, cpu), f"case {i}: {int((m != cpu).sum())} voxels differ from the CPU drawing"
    assert 0 < repeated < len(golden["stamp"])


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("fname", list(FUNCS))
@pytest.mark.parametrize("name", ["dense", "dense_gobj", "sparse"])
def test_dt_functions_equal_750c94a(golden, name, fname):
    obj, gob = _phantom(golden, name)
    r = FUNCS[fname](obj, gobj=None if gob is None else {"rendered": gob}, backend="gpu")
    exp = golden["phantoms"][name]["dt"][fname]
    m = _decode(exp["map"])
    assert r.map.dtype == m.dtype == np.int16
    assert np.array_equal(r.map, m), f"{int((r.map != m).sum())} map voxels differ from 750c94a"
    assert np.array_equal(r.centres, _decode(exp["centres"]))
    assert r.report == exp["report"]
    if name == "sparse" and fname != "thickness":
        assert r.map.max() > 150                     # the case the schedule exists for


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("fname", ["spacing", "number"])
def test_sparse_dt_gpu_equals_cpu(golden, fname):
    obj, _ = _phantom(golden, "sparse")
    a = FUNCS[fname](obj, backend="cpu")
    b = FUNCS[fname](obj, backend="gpu")
    assert np.array_equal(a.map, b.map) and np.array_equal(a.centres, b.centres) and a.report == b.report


# ----------------------------------------------------------------------------------------- the safety net
@requires_gpu
@pytest.mark.gpu
def test_schedule_never_changes_the_map():
    """Random large-sphere sets under eight schedules: one map."""
    rng = np.random.default_rng(11)
    shape = (37, 53, 61)
    n = 120
    cz, cy, cx = (rng.integers(0, s, n) for s in shape)
    d = np.where(rng.random(n) < 0.6, rng.integers(0, 25, n), rng.integers(25, 200, n))
    ref = gpu.draw_spheres_gpu(cz, cy, cx, d, shape, 0.5)
    for small_box, budget in SCHEDULES + [(3, 1 << 14), (27, 27 ** 3), (101, 1 << 22), (401, 1 << 30)]:
        m = gpu._draw_spheres_gpu(cz, cy, cx, d, shape, 0.5, budget=budget, small_box=small_box)
        assert np.array_equal(m, ref), (small_box, budget)


@requires_gpu
@pytest.mark.gpu
def test_launches_respect_the_budget():
    rng = np.random.default_rng(3)
    shape = (40, 64, 64)
    n = 300
    cz, cy, cx = (rng.integers(0, s, n) for s in shape)
    d = rng.integers(0, 120, n)
    timings = []
    gpu._draw_spheres_gpu(cz, cy, cx, d, shape, 0.5, budget=1 << 18, small_box=9, timings=timings)
    kinds = {t[0] for t in timings}
    assert kinds == {"point", "columns"}
    assert all(t[3] <= 1 << 18 and t[4] >= 0 for t in timings)


@requires_gpu
@pytest.mark.gpu
def test_column_kernel_refuses_a_plan_that_disagrees_with_its_radius(monkeypatch):
    """A plan built on wrong radii (here R + 1) is caught on the device: no map is returned."""
    real = gpu.stamp_plan
    monkeypatch.setattr(gpu, "stamp_plan",
                        lambda R, budget=None, small_box=None: real(np.asarray(R) + 1, budget, small_box))
    with pytest.raises(RuntimeError, match="disagrees"):
        gpu.draw_spheres_gpu(np.array([5]), np.array([6]), np.array([7]), np.array([60]), (12, 13, 14), 0.5)
