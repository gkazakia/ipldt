"""The float64 opt-in of /fft_laplace_hamming (ipldt.ormir.lh_filter_core dtype=, 2026-09-17).

WHAT IS PINNED HERE
  * the engine's default is float32 (ormir.LH_DTYPE) and an explicit dtype='float32' is the default path exactly,
    so nothing published moves; an unknown dtype is refused;
  * dtype='float64' runs the same operations in double precision and returns the SAME kinds of things: a float32
    filter output and IPL's short = trunc(float32(lh) * float32(32767 / 200000)) of THAT float32 -- the rounding to
    float32 happens before IPL's conversion, which is not touched;
  * the two arithmetics differ only by rounding: the short differs by at most +/- 1 (truncation flips at integer
    boundaries) and the float by a tiny fraction of the peak;
  * the padding rules (ceil / floor, power-of-two boxes) are the same under both dtypes;
  * validation/validate_dataset.py's harness copy passes dtype through to the engine;
  * against IPL's own exports (probe 21, skipped when the folder is not mounted): float64 sits at the FFT noise floor
    too (float within 1.0 of IPL's FLOAT export, short within +/- 1, SEG exact on the two sharpest phantoms).
"""
import os
import sys

import numpy as np
import pytest
from conftest import lab_path  # noqa: E402  (non-public data roots)

from ipldt import ormir
from ipldt.io import read_aim

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE21 = os.environ.get("IPLDT_PROBE21_DIR", lab_path("Python/scripts/IPL/probes/p21_lh_padding"))
THR = ormir.LH_THRESHOLD
EL = (0.0607, 0.0607, 0.0607)
SCALE = np.float32(ormir.INT16_MAX) / np.float32(ormir.NORM_MAX_VALUE)


def _greyscale(shape, seed=0):
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.standard_normal(shape), 1.5)
    return np.clip(f / np.abs(f).max() * 6000.0, -4000, 8000).astype(np.int16)


def _short_of(lh):
    scaled = np.asarray(lh, np.float32) * SCALE
    np.clip(scaled, -ormir.INT16_MAX, ormir.INT16_MAX, out=scaled)
    return np.trunc(scaled).astype(np.int16)


def test_default_is_float32_and_unknown_dtype_is_refused():
    assert ormir.LH_DTYPE == "float32" and set(ormir.LH_DTYPES) == {"float32", "float64"}
    v = _greyscale((13, 30, 21)).astype(np.float32)
    a = ormir.lh_filter_core(v, EL, "ceil")
    b = ormir.lh_filter_core(v, EL, "ceil", "float32")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    with pytest.raises(ValueError):
        ormir.lh_filter_core(v, EL, "ceil", "float16")
    with pytest.raises(ValueError):
        ormir.laplace_hamming_threshold(_greyscale((13, 30, 21)), EL, dtype="double")


def test_float64_returns_float32_lh_and_ipl_short_of_it():
    v = _greyscale((13, 30, 21), seed=2).astype(np.float32)
    lh32, sh32 = ormir.lh_filter_core(v, EL, "ceil", "float32")
    lh64, sh64 = ormir.lh_filter_core(v, EL, "ceil", "float64")
    assert lh64.dtype == np.float32 and sh64.dtype == np.int16 and lh64.shape == v.shape and sh64.shape == v.shape
    assert np.array_equal(sh64, _short_of(lh64))                     # IPL's conversion, applied to the float32 result
    d = np.abs(lh64.astype(np.float64) - lh32.astype(np.float64))
    assert 0 < float(d.max()) < 1e-5 * float(np.abs(lh32).max())     # rounding only: they differ, but by a whisker
    ds = np.abs(sh64.astype(np.int64) - sh32.astype(np.int64))
    assert int(ds.max()) <= 1                                        # +/- 1 truncation flips at integer boundaries
    assert int((ds != 0).sum()) < 0.05 * ds.size


def test_padding_rules_hold_under_float64():
    v = _greyscale((13, 30, 21), seed=4)                             # bordered 15 x 32 x 23: odd padding on z and x
    c = ormir.laplace_hamming_threshold(v, EL, pad_offset="ceil", dtype="float64")
    f = ormir.laplace_hamming_threshold(v, EL, pad_offset="floor", dtype="float64")
    assert c.shape == v.shape and c.dtype == bool and not np.array_equal(c, f)
    w = _greyscale((14, 30, 30), seed=5)                             # bordered 16 x 32 x 32: nothing padded
    assert np.array_equal(ormir.laplace_hamming_threshold(w, EL, "ceil", dtype="float64"),
                          ormir.laplace_hamming_threshold(w, EL, "floor", dtype="float64"))
    # and float64 is deterministic
    assert np.array_equal(c, ormir.laplace_hamming_threshold(v, EL, pad_offset="ceil", dtype="float64"))


def test_harness_copy_passes_dtype_through():
    validation = os.path.join(REPO, "validation")
    if validation not in sys.path:
        sys.path.insert(0, validation)
    vds = pytest.importorskip("validate_dataset", reason="validation/validate_dataset.py needs the validation extras")
    v = _greyscale((13, 30, 21), seed=3)
    for dtype in ("float32", "float64"):
        assert np.array_equal(vds.laplace_hamming_threshold(v, EL, border="duplicate", pad_offset="ceil", dtype=dtype),
                              ormir.laplace_hamming_threshold(v, EL, pad_offset="ceil", dtype=dtype))
        _, sh = ormir.lh_filter_core(v.astype(np.float32), EL, "ceil", dtype)
        assert np.array_equal(vds.laplace_hamming_threshold(v, EL, border="none", pad_offset="ceil", dtype=dtype), sh >= THR)


@pytest.fixture(scope="module")
def probe21():
    up, ex = os.path.join(PROBE21, "upload"), os.path.join(PROBE21, "aims_and_logs")
    if not (os.path.isdir(up) and os.path.isdir(ex)):
        pytest.skip(f"probe-21 folder not mounted: {PROBE21} (set IPLDT_PROBE21_DIR)")

    def load(tag):
        paths = dict(inp=os.path.join(up, f"x2420448_p21_{tag}.aim"),
                     **{k: os.path.join(ex, f"X2420448_P21_{tag.upper()}_{k}.AIM") for k in ("LH", "NM", "SG")})
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            pytest.skip(f"probe-21 files missing: {missing}")
        return {k: read_aim(p) for k, p in paths.items()}
    return load


@pytest.mark.parametrize("tag, ipl_sg_voxels", [("x63i61", 0), ("x63i01", 6)])
def test_probe21_phantom_matches_ipl_under_float64_too(probe21, tag, ipl_sg_voxels):
    X = probe21(tag)
    vol, el = np.asarray(X["inp"]["data"]), tuple(float(e) for e in X["inp"]["el_size_mm"])
    lh, sh = ormir.lh_filter_core(vol.astype(np.float32), el, "ceil", "float64")
    lh_ipl = np.asarray(X["LH"]["data"], np.float32)
    nm_ipl = np.asarray(X["NM"]["data"]).astype(np.int16)
    sg_ipl = np.asarray(X["SG"]["data"]) != 0
    assert float(np.abs(lh.astype(np.float64) - lh_ipl.astype(np.float64)).max()) <= 1.0
    assert int(np.abs(sh.astype(np.int64) - nm_ipl.astype(np.int64)).max()) <= 1
    assert int(sg_ipl.sum()) == ipl_sg_voxels
    assert int(((sh >= THR) != sg_ipl).sum()) == 0
