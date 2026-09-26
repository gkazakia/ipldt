"""Shared fixtures for the ipldt test suite.

The fast tests run on synthetic phantoms and need only numpy / scipy / numba.  The GPU tests
skip themselves when CuPy or a CUDA device is missing; the real-data test (marked `slow`)
skips itself when the (non-public) validation data are absent (set IPLDT_LAB_ROOT or IPLDT_DATA_ROOT; see
validation/datapaths.py).
vms_versions is the one way the real-data tests pick a file among OpenVMS ';n' versions.
"""
import os
import re

import numpy as np
import pytest
from scipy import ndimage as ndi

from ipldt.contour import render_volume
from ipldt.gpu import cupy_available

LAB_ROOT = os.environ.get("IPLDT_LAB_ROOT", "")


def lab_path(rel):
    """<IPLDT_LAB_ROOT>/<rel> (see validation/datapaths.py); a never-existing path when the variable is unset, so that
    the tests needing the non-public data skip."""
    return os.path.join(LAB_ROOT or "IPLDT_LAB_ROOT_is_not_set", rel).replace("\\", "/")


DEFAULT_DATA_ROOT = lab_path("PFJOA/XCT_masks_full_grab")
DATA_ROOT = os.environ.get("IPLDT_DATA_ROOT", DEFAULT_DATA_ROOT)

requires_gpu = pytest.mark.skipif(not cupy_available(), reason="CuPy / CUDA device not available")


def vms_versions(folder, pattern):
    """Files in `folder` named `pattern` with an optional OpenVMS ';n' suffix, oldest first by NUMERIC version
    (an unsuffixed file counts as 0; a plain string sort would put ';10' before ';2'); [] when the folder is
    absent.  The newest version is fs[-1]."""
    if not os.path.isdir(folder):
        return []
    rx = re.compile("^" + re.escape(pattern) + r"(;\d+)?$", re.I)
    key = lambda f: int(f.rsplit(";", 1)[1]) if ";" in f else 0    # noqa: E731
    return [os.path.join(folder, f) for f in sorted((f for f in os.listdir(folder) if rx.match(f)), key=key)]


def random_phantom(shape=(20, 40, 40), seed=0, sigma=2.0, fill=0.45):
    """A smooth random binary structure (thresholded filtered noise): plates and rods of
    diameters 1..~8 voxels, ~45 % object.  Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return f > np.quantile(f, 1.0 - fill)


def clean_cylinder(shape=(20, 40, 40), radius=14.0, z0=2, z1=18):
    """A disc on slices z0..z1-1, empty elsewhere.  With a half-integer centre the digital disc
    has no convex 90-degree corner and no 1-pixel run, so IPL's rendering leaves it unchanged."""
    z, y, x = shape
    yy, xx = np.mgrid[:y, :x]
    disc = (yy - (y - 1) / 2) ** 2 + (xx - (x - 1) / 2) ** 2 <= radius ** 2
    m = np.zeros(shape, bool)
    m[z0:z1] = disc
    return m


def cylinder_mask(shape=(20, 40, 40), radius=14.0, z0=2, z1=18):
    """A raw (unrendered) trabecular-style mask: the clean cylinder plus, on every slice, a
    1-pixel excursion on its east side and a 1-pixel bump above its top row -- features IPL's
    contour rendering removes, so the raw raster differs from its rendering."""
    m = clean_cylinder(shape, radius, z0, z1)
    ys, xs = np.nonzero(m[z0])
    row = (shape[1] - 1) // 2
    east = int(xs[ys == row].max())
    top = int(ys.min())
    xm = int(np.median(xs[ys == top]))
    m[z0:z1, row, east + 1] = True
    m[z0:z1, top - 1, xm] = True
    return m


@pytest.fixture(scope="session")
def phantom():
    return random_phantom()


@pytest.fixture(scope="session")
def raw_mask():
    return cylinder_mask()


@pytest.fixture(scope="session")
def clean_mask():
    return clean_cylinder()


@pytest.fixture(scope="session")
def phantom_gobj(raw_mask):
    """The cylinder mask rendered with IPL's contour rules (what dt_* use as the gobj)."""
    return render_volume(raw_mask)


@pytest.fixture(scope="session")
def data_root():
    return DATA_ROOT
