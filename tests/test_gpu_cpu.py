"""GPU (CuPy) and CPU backends must be bit-identical: maps, centres and reports, for all three
functions, all three diameter versions, with and without a gobj, and across assign_epsilon."""
import numpy as np
import pytest

import ipldt.gpu as gpu
from ipldt import dt_number, dt_spacing, dt_thickness

from conftest import requires_gpu

FUNCS = {"thickness": dt_thickness, "spacing": dt_spacing, "number": dt_number}


def _same(a, b):
    assert a.map.dtype == b.map.dtype == np.int16
    assert np.array_equal(a.centres, b.centres), "centre sets differ between backends"
    assert np.array_equal(a.map, b.map), f"{int((a.map != b.map).sum())} map voxels differ between backends"
    assert a.report == b.report


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("with_gobj", [False, True], ids=["nogobj", "gobj"])
@pytest.mark.parametrize("version", [1, 2, 3])
@pytest.mark.parametrize("fname", list(FUNCS))
def test_gpu_equals_cpu(phantom, phantom_gobj, fname, version, with_gobj):
    g = {"rendered": phantom_gobj} if with_gobj else None
    a = FUNCS[fname](phantom, gobj=g, version=version, backend="cpu")
    b = FUNCS[fname](phantom, gobj=g, version=version, backend="gpu")
    assert a.map.any(), "phantom produced an empty map; the comparison would be vacuous"
    _same(a, b)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("assign_epsilon", [0.0, 0.25, 0.5, 1.0])
def test_gpu_equals_cpu_assign_epsilon(phantom, phantom_gobj, assign_epsilon):
    """The atomicMax stamp kernel and the EDT-based CPU drawing use the same non-strict
    tolerance |c - x| <= D/2 + assign_epsilon."""
    a = dt_thickness(phantom, gobj={"rendered": phantom_gobj}, assign_epsilon=assign_epsilon, backend="cpu")
    b = dt_thickness(phantom, gobj={"rendered": phantom_gobj}, assign_epsilon=assign_epsilon, backend="gpu")
    _same(a, b)


@requires_gpu
@pytest.mark.gpu
def test_auto_backend_picks_gpu_and_matches(phantom):
    assert gpu.resolve_backend("auto") == "gpu"
    _same(dt_thickness(phantom, backend="auto"), dt_thickness(phantom, backend="cpu"))


@requires_gpu
@pytest.mark.gpu
def test_gpu_primitives_match_cpu(phantom):
    """The two GPU kernels against their CPU counterparts directly (no masking in between)."""
    from ipldt.core import _RIDGE_TOL, diameters, draw_spheres, ridge
    from ipldt.field import sir_quad
    V = sir_quad(phantom)
    c_cpu = ridge(phantom, V, 0.9)
    c_gpu = gpu.ridge_gpu(phantom, V, 0.9, _RIDGE_TOL)
    assert np.array_equal(c_cpu, c_gpu)
    z, y, x = np.nonzero(c_cpu)
    D = diameters(phantom, V, z, y, x, version=3)
    assert np.array_equal(draw_spheres(z, y, x, D, phantom.shape, 0.5),
                          gpu.draw_spheres_gpu(z, y, x, D, phantom.shape, 0.5))
    # no centres at all: both drawings are all-zero maps of the right shape/dtype
    empty = np.zeros(0, np.int64)
    a = draw_spheres(empty, empty, empty, empty, (3, 4, 5))
    b = gpu.draw_spheres_gpu(empty, empty, empty, empty, (3, 4, 5))
    assert a.shape == b.shape == (3, 4, 5) and not a.any() and not b.any() and b.dtype == np.int16


def test_backend_resolution_without_cupy(monkeypatch):
    """'auto' falls back to the CPU, 'gpu' raises, anything else is rejected."""
    monkeypatch.setattr(gpu, "_CUPY", False)
    assert not gpu.cupy_available()
    assert gpu.resolve_backend("auto") == "cpu"
    assert gpu.resolve_backend("cpu") == "cpu"
    with pytest.raises(RuntimeError):
        gpu.resolve_backend("gpu")
    with pytest.raises(ValueError):
        gpu.resolve_backend("tpu")


def test_cpu_backend_is_deterministic(phantom, phantom_gobj):
    a = dt_thickness(phantom, gobj={"rendered": phantom_gobj}, backend="cpu")
    b = dt_thickness(phantom, gobj={"rendered": phantom_gobj}, backend="cpu")
    _same(a, b)
