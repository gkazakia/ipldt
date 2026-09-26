"""The containment ridge (stage 3) must decide near-ties as the exact value does.

2026-09-14, OS_LH cohort: on three diaphyseal tibiae (Diaphyseal/REPRO 314619, 797620, 134330; Ct.Th of the
cortical compartment, cortex about 140 voxels thick) the float32 surface distances used until then
kept ONE sphere centre that IPL prunes.  With v_x = (-5, 70, -7) and the +x neighbour's
v_y = (-5, 70, -8) (the same contact voxel) the exact ridge value
    1 + s_x - s_y = 1 + sqrt(19571)/2 - sqrt(19627)/2 = 0.8999974
is below ridge_epsilon 0.9 (prune), but float32 (spacing 7.6e-6 at s = 70) gave 0.9000015 (keep); the
spurious diameter-140 sphere put 2,601 / 8 / 9 voxels of the three maps off IPL's.  The surface
distances are now float64 from the exact integer 4 s^2 and the tolerance is 1e-9 (ipldt.core).

The pairs below are the closest near-ties of the shared-contact pair space |v_i| <= 80 (every
26-neighbour offset o, v_y = v_x - o), on both sides of 0.9; the reference decision is computed with
60-digit Decimal arithmetic, independently of ipldt.
"""
import inspect
from decimal import Decimal, getcontext

import numpy as np
import pytest

import ipldt.gpu as gpu
from ipldt.core import _RIDGE_TOL, ridge, surface_distance

from conftest import requires_gpu

getcontext().prec = 60

# (v_x, offset o to the neighbour y, float32 decided wrongly)   -- v_y = v_x - o shares the contact voxel
NEAR_TIES = [
    pytest.param((-5, 70, -7), (0, 0, 1), True, id="oslh-314619-A19571-B19627-below"),      # exact 0.8999974, the finding
    pytest.param((-15, 46, -40), (-1, -1, 0), True, id="A15363-B15619-below-5e-7"),         # exact 0.8999994949: closest non-tie found
    pytest.param((5, -39, -32), (-1, 0, 0), False, id="A9979-B10019-below"),               # exact 0.8999949
    pytest.param((-35, 35, 70), (-1, -1, -1), False, id="A28843-B29411-above"),            # exact 0.9000081, oblique (sqrt3)
    pytest.param((6, -54, -27), (-1, 0, 0), False, id="A14379-B14427-above"),              # exact 0.9000104
    pytest.param((-16, 41, -26), (-1, -1, 0), False, id="A10123-B10331-above"),            # exact 0.9000103, sqrt2
]
BACKENDS = ["cpu", pytest.param("gpu", marks=[requires_gpu, pytest.mark.gpu])]


def sq4(v):
    """4 s^2 as an exact integer: the sum over the non-zero components of (2|v_i| - 1)^2."""
    return sum((2 * abs(int(c)) - 1) ** 2 for c in v if c != 0)


def exact_ridge_value(v_x, o):
    v_y = tuple(a - b for a, b in zip(v_x, o))
    k = sum(c * c for c in o)
    return Decimal(k).sqrt() + (Decimal(sq4(v_x)).sqrt() - Decimal(sq4(v_y)).sqrt()) / 2


def old_float32_value(v_x, o):
    """The evaluation ipldt used until 2026-09-14 (float32 surface distances, float32 sum)."""
    s = lambda v: np.sqrt(sum(np.maximum(np.abs(np.float32(c)) - np.float32(0.5), np.float32(0)) ** 2 for c in v)).astype(np.float32)   # noqa: E731
    v_y = tuple(a - b for a, b in zip(v_x, o))
    return np.float32(np.sqrt(sum(c * c for c in o))) + s(v_x) - s(v_y)


def two_voxel_field(v_x, o):
    """x at the centre of a 3 x 3 x 3 image, y = x + o; both object, with the given vectors."""
    obj = np.zeros((3, 3, 3), bool)
    V = np.zeros((3, 3, 3, 3), np.int16)
    x = (1, 1, 1)
    y = tuple(a + b for a, b in zip(x, o))
    obj[x] = obj[y] = True
    for i in range(3):
        V[i][x] = v_x[i]
        V[i][y] = v_x[i] - o[i]
    return obj, V, x, y


def run_ridge(backend, obj, V, eps):
    if backend == "gpu":
        return gpu.ridge_gpu(obj, V, eps, _RIDGE_TOL)
    return ridge(obj, V, eps)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("v_x,o,float32_wrong", NEAR_TIES)
def test_near_tie_decision_matches_the_exact_value(backend, v_x, o, float32_wrong):
    exact = exact_ridge_value(v_x, o)
    assert abs(exact - Decimal("0.9")) < Decimal("3e-5"), "not a near-tie"
    assert exact != Decimal("0.9")
    expect_prune = exact <= Decimal("0.9")
    obj, V, x, y = two_voxel_field(v_x, o)
    cen = run_ridge(backend, obj, V, 0.9)
    assert bool(cen[x]) == (not expect_prune), f"x kept={bool(cen[x])} but the exact ridge value is {exact:.12f}"
    assert bool(cen[y]), "y's own value 2|o| - f > 0.9: y stays a centre"
    # the float32 evaluation of record decided the flagged pairs the other way (the 2026-09-14 finding)
    old = old_float32_value(v_x, o) <= np.float32(0.9 + 1e-6)
    assert bool(old) == (expect_prune != float32_wrong)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exact_ties_prune_at_ridge_epsilon_zero(backend):
    """True ties (rational ridge value) exist only at integer epsilon; the 1e-9 tolerance must absorb their
    float64 rounding: 1 + 2.5 - 3.5, sqrt2 + sqrt(1/2) - sqrt(9/2), sqrt3 + sqrt(3/4) - sqrt(27/4) are all 0."""
    for v_x, o in [((0, 0, 3), (0, 0, -1)), ((0, 1, 1), (0, -1, -1)), ((1, 1, 1), (-1, -1, -1))]:
        assert abs(exact_ridge_value(v_x, o)) < Decimal("1e-50")          # a true tie (Decimal sqrt residue only)
        obj, V, x, y = two_voxel_field(v_x, o)
        s = surface_distance(V)
        f = float(np.sqrt(sum(c * c for c in o))) + float(s[x]) - float(s[y])
        assert abs(f) < 1e-12                                     # float64 rounding of a true tie
        cen = run_ridge(backend, obj, V, 0.0)
        assert not cen[x], f"tie {v_x} / {o} must prune at ridge_epsilon 0 (value {f:.3e})"
        assert cen[y]
        assert run_ridge(backend, obj, V, 0.0)[x] == run_ridge(backend, obj, V, 0.9)[x]


def test_surface_distance_is_float64_from_the_exact_integer():
    rng = np.random.default_rng(1)
    V = rng.integers(-80, 81, size=(3, 6, 7, 8)).astype(np.int16)
    V[:, 0, 0, :] = 0                                            # background voxels: s = 0
    s = surface_distance(V)
    assert s.dtype == np.float64 and s.shape == V.shape[1:]
    A = np.zeros(V.shape[1:], np.int64)
    for i in range(3):
        u = np.abs(V[i].astype(np.int64))
        A += np.where(u > 0, (2 * u - 1) ** 2, 0)
    assert np.array_equal(np.rint(4 * s * s), A)                  # 4 s^2 recovers the integer exactly
    ref = np.sqrt(sum(np.maximum(np.abs(V[i].astype(np.float64)) - 0.5, 0) ** 2 for i in range(3)))
    assert np.array_equal(s, ref)                                 # same bits as the float64 definition
    assert (s[0, 0] == 0).all() and (s[1:] > 0).all()
    # the finding in numbers
    x = np.array([[-5], [70], [-7]]).reshape(3, 1, 1, 1)
    y = np.array([[-5], [70], [-8]]).reshape(3, 1, 1, 1)
    f = 1.0 + float(surface_distance(x)[0, 0, 0]) - float(surface_distance(y)[0, 0, 0])
    assert abs(f - 0.899997423) < 1e-8 and f <= 0.9
    old = 1.0 + float(old_float32_value((-5, 70, -7), (0, 0, 1)) - np.float32(1.0))
    assert old > 0.9


@requires_gpu
@pytest.mark.gpu
def test_surface_distance_same_bits_on_the_gpu():
    import cupy as cp
    rng = np.random.default_rng(2)
    V = rng.integers(-90, 91, size=(3, 5, 6, 7)).astype(np.int16)
    assert np.array_equal(surface_distance(V), cp.asnumpy(surface_distance(V, xp=cp)))


def test_ridge_tolerance_is_small_enough():
    """1e-9: far above the float64 rounding of a true tie, far below the closest non-tie (5.05e-7 from 0.9)."""
    assert _RIDGE_TOL == 1e-9
    assert inspect.signature(gpu.ridge_gpu).parameters["tol"].default == _RIDGE_TOL
    closest = min(abs(exact_ridge_value(p.values[0], p.values[1]) - Decimal("0.9")) for p in NEAR_TIES)
    assert Decimal("4e-7") < closest < Decimal("6e-7")
    assert closest > 100 * Decimal(repr(_RIDGE_TOL))
