"""The power-of-two padding of /fft_laplace_hamming: IPL puts the extra voxel BEFORE the data (probe 21, 2026-09-17).

WHAT IS PINNED HERE
  * ormir.lh_pad_plan: 'ceil' (lo = d - d // 2, IPL's rule) and 'floor' (lo = d // 2, the rule shipped before
    2026-09-17) on the axis lengths probe 21 used, and that an unknown rule is refused;
  * pad_offset='floor' is BYTE-IDENTICAL to the pre-refactor laplace_hamming_threshold, whose code is copied
    verbatim below (the refactor into lh_filter_core moved the crop before the elementwise scaling, which must not
    move a bit);
  * on a power-of-two box the two rules coincide (no padding at all: IPL's log prints no
    D3P_FFT_AdjustDimensionsMirror there), on an odd-padded box they differ;
  * validation/validate_dataset.py's laplace_hamming_threshold, which delegates to the core, equals the engine
    under both rules, and its border='none' is the core's own threshold;
  * against IPL's OWN exports (the probe-21 folder, skipped when it is not mounted; IPLDT_PROBE21_DIR relocates
    it): on the two sharpest phantoms, x63i61 (an impulse one voxel below the high x face: its thresholded export
    is non-empty only under floor + reflect, and came back EMPTY) and x63i01 (one voxel above the low x face:
    non-empty only under ceil + reflect, and came back with 6 voxels set), the core under 'ceil' reproduces the
    FLOAT export to the FFT noise floor, the SHORT export up to +/- 1 truncation flips and the SEG export
    exactly, while 'floor' is off by thousands of float units and gets the segmentation wrong.
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


# ------------------------------------------------------------------- the pre-refactor function, VERBATIM (2026-09-17)
def shipped_floor_reference(native_int16, voxel_size_mm=None):
    """ipldt.ormir.laplace_hamming_threshold as it read before the probe-21 refactor (the 2026-09-16 file, lines
    528-576), copied verbatim with the module constants qualified."""
    if voxel_size_mm is None:
        el = ormir.LH_EL_SIZE_FALLBACK_MM
    elif np.ndim(voxel_size_mm) == 0:
        el = (float(voxel_size_mm),) * 3
    else:
        el = tuple(float(e) for e in voxel_size_mm)
    native = np.asarray(native_int16)
    extended = np.pad(native, ((1, 1), (1, 1), (1, 1)), mode="edge").astype(np.float32)

    def _npow2(n):
        return 1 if n <= 1 else 2 ** int(np.ceil(np.log2(n)))

    pad_widths, inner = [], []
    for n in extended.shape:
        nt = _npow2(n)
        lo = (nt - n) // 2
        pad_widths.append((lo, (nt - n) - lo))
        inner.append(slice(lo, lo + n))
    padded = np.pad(extended, pad_widths, mode="reflect")
    nz, ny, nx = padded.shape
    ex, ey, ez = (np.float32(e) for e in el)
    kx = np.fft.fftfreq(nx, d=float(ex)).astype(np.float32)
    ky = np.fft.fftfreq(ny, d=float(ey)).astype(np.float32)
    kz = np.fft.fftfreq(nz, d=float(ez)).astype(np.float32)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    K2 = KX * KX + KY * KY + KZ * KZ
    Kmag = np.sqrt(K2)
    k_lp = np.float32(ormir.LP_CUT_OFF_FREQ) / ez
    half_amp = np.float32(ormir.HAMMING_AMP) * np.float32(0.5)
    win = np.where(Kmag < k_lp, (np.float32(1.0) - half_amp) + half_amp * np.cos(np.pi * Kmag / k_lp), np.float32(0.0)).astype(np.float32)
    H = np.float32((2.0 * np.pi) ** 2) * ((np.float32(1.0) - np.float32(ormir.LAPLACE_EPS)) + np.float32(ormir.LAPLACE_EPS) * K2) * win
    F = ormir._fftn(padded, workers=-1)
    F *= H
    lh = np.real(ormir._ifftn(F, workers=-1)).astype(np.float32)
    del F, H, win, K2, Kmag, KX, KY, KZ
    scaled = lh * (np.float32(ormir.INT16_MAX) / np.float32(ormir.NORM_MAX_VALUE))
    np.clip(scaled, -ormir.INT16_MAX, ormir.INT16_MAX, out=scaled)
    lh_int16 = np.trunc(scaled).astype(np.int16)
    bm = (lh_int16 >= ormir.LH_THRESHOLD) & (lh_int16 <= ormir.INT16_MAX)
    return bm[tuple(inner)][1:-1, 1:-1, 1:-1]


def _greyscale(shape, seed=0):
    """A bone-like int16 volume: smooth random structure scaled into the native range, so that the threshold
    fires on a good fraction of the voxels, faces included."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.standard_normal(shape), 1.5)
    return np.clip(f / np.abs(f).max() * 6000.0, -4000, 8000).astype(np.int16)


# ------------------------------------------------------------------------------------------ the plan
def test_pad_plan_ceil_puts_the_odd_voxel_before_the_data():
    for n, nt, lo_ceil, lo_floor in ((64, 64, 0, 0), (63, 64, 1, 0), (62, 64, 1, 1), (33, 64, 16, 15), (34, 64, 15, 15),
                                     (170, 256, 43, 43), (353, 512, 80, 79), (748, 1024, 138, 138), (1, 1, 0, 0)):
        for rule, want in (("ceil", lo_ceil), ("floor", lo_floor)):
            pw, inner = ormir.lh_pad_plan((n,), rule)
            assert pw == [(want, nt - n - want)], (n, rule, pw)
            assert inner == (slice(want, want + n),)
    assert ormir.LH_PAD_OFFSET == "ceil" and set(ormir.LH_PAD_OFFSETS) == {"ceil", "floor"}
    with pytest.raises(ValueError):
        ormir.lh_pad_plan((63,), "left")


# --------------------------------------------------------------------------------- byte-identity of 'floor'
def test_floor_is_byte_identical_to_the_pre_refactor_code():
    v = _greyscale((13, 30, 21))                 # bordered 15 x 32 x 23 -> box 16 x 32 x 32: odd padding on z and x
    ref = shipped_floor_reference(v, EL)
    new = ormir.laplace_hamming_threshold(v, EL, pad_offset="floor")
    assert new.shape == v.shape and new.dtype == bool
    assert np.array_equal(ref, new)
    assert 0 < int(ref.sum()) < ref.size
    ceil = ormir.laplace_hamming_threshold(v, EL, pad_offset="ceil")
    assert not np.array_equal(ref, ceil)         # the rules differ on an odd-padded box ...
    assert ormir.laplace_hamming_threshold(v, EL).sum() == ceil.sum()       # ... and ceil is the default
    assert np.array_equal(ormir.laplace_hamming_threshold(v, EL), ceil)


def test_ceil_equals_floor_when_nothing_is_padded():
    v = _greyscale((14, 30, 30), seed=1)         # bordered 16 x 32 x 32: a power-of-two box, no padding at all
    a = ormir.lh_filter_core(np.pad(v, 1, mode="edge").astype(np.float32), EL, "floor")
    b = ormir.lh_filter_core(np.pad(v, 1, mode="edge").astype(np.float32), EL, "ceil")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[0].shape == (16, 32, 32) and a[0].dtype == np.float32 and a[1].dtype == np.int16
    assert np.array_equal(ormir.laplace_hamming_threshold(v, EL, "floor"), ormir.laplace_hamming_threshold(v, EL, "ceil"))


def test_core_short_is_truncated_scaled_float():
    v = _greyscale((13, 30, 21), seed=2).astype(np.float32)
    lh, sh = ormir.lh_filter_core(v, EL, "ceil")
    scaled = lh * (np.float32(ormir.INT16_MAX) / np.float32(ormir.NORM_MAX_VALUE))
    np.clip(scaled, -ormir.INT16_MAX, ormir.INT16_MAX, out=scaled)
    assert np.array_equal(sh, np.trunc(scaled).astype(np.int16))
    assert lh.shape == v.shape and sh.shape == v.shape


def test_harness_copy_is_the_engine():
    validation = os.path.join(REPO, "validation")
    if validation not in sys.path:
        sys.path.insert(0, validation)
    vds = pytest.importorskip("validate_dataset", reason="validation/validate_dataset.py needs the validation extras")
    v = _greyscale((13, 30, 21), seed=3)
    for rule in ("floor", "ceil"):
        assert np.array_equal(vds.laplace_hamming_threshold(v, EL, border="duplicate", pad_offset=rule),
                              ormir.laplace_hamming_threshold(v, EL, pad_offset=rule))
        _, sh = ormir.lh_filter_core(v.astype(np.float32), EL, rule)
        assert np.array_equal(vds.laplace_hamming_threshold(v, EL, border="none", pad_offset=rule), sh >= THR)
        kept = vds.laplace_hamming_threshold(v, EL, border="duplicate", keep_border=True, pad_offset=rule)
        assert kept.shape == tuple(s + 2 for s in v.shape)
        assert np.array_equal(kept[1:-1, 1:-1, 1:-1], ormir.laplace_hamming_threshold(v, EL, pad_offset=rule))
    defaults = dict(zip(vds.laplace_hamming_threshold.__code__.co_varnames[:vds.laplace_hamming_threshold.__code__.co_argcount][-len(vds.laplace_hamming_threshold.__defaults__):],
                        vds.laplace_hamming_threshold.__defaults__))
    assert defaults["pad_offset"] == ormir.LH_PAD_OFFSET
    assert defaults["dtype"] == ormir.LH_DTYPE == "float32"     # the harness follows the engine's defaults (2026-09-17: float32)


# ------------------------------------------------------------------------- against IPL's own exports (probe 21)
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
        X = {k: read_aim(p) for k, p in paths.items()}
        for k in ("LH", "NM", "SG"):
            assert tuple(X[k]["dim"]) == tuple(X["inp"]["dim"]) and tuple(X[k]["pos"]) == tuple(X["inp"]["pos"])
        return X
    return load


@pytest.mark.parametrize("tag, ipl_sg_voxels", [("x63i61", 0), ("x63i01", 6)])
def test_probe21_phantom_matches_ipl_under_ceil(probe21, tag, ipl_sg_voxels):
    X = probe21(tag)
    vol, el = np.asarray(X["inp"]["data"]), tuple(float(e) for e in X["inp"]["el_size_mm"])
    assert vol.dtype == np.int16 and vol.shape == (64, 64, 63)                   # (z, y, x): 63 on x only
    lh, sh = ormir.lh_filter_core(vol.astype(np.float32), el, "ceil")
    lh_ipl = np.asarray(X["LH"]["data"], np.float32)
    nm_ipl = np.asarray(X["NM"]["data"]).astype(np.int16)
    sg_ipl = np.asarray(X["SG"]["data"]) != 0
    d = np.abs(lh.astype(np.float64) - lh_ipl.astype(np.float64))
    assert float(np.abs(lh_ipl).max()) > 5e4                                     # the impulse response is large ...
    assert float(d.max()) <= 1.0                                                 # ... and we sit at the FFT noise floor
    ds = np.abs(sh.astype(np.int64) - nm_ipl.astype(np.int64))
    assert int(ds.max()) <= 1                                                    # only truncation flips at integer boundaries
    assert int(sg_ipl.sum()) == ipl_sg_voxels                                    # the model-free readout IPL returned
    assert int(((sh >= THR) != sg_ipl).sum()) == 0                               # and our decisions equal IPL's exactly


@pytest.mark.parametrize("tag", ["x63i61", "x63i01"])
def test_probe21_phantom_refutes_floor(probe21, tag):
    X = probe21(tag)
    vol, el = np.asarray(X["inp"]["data"]), tuple(float(e) for e in X["inp"]["el_size_mm"])
    lh_f, sh_f = ormir.lh_filter_core(vol.astype(np.float32), el, "floor")
    lh_ipl = np.asarray(X["LH"]["data"], np.float32)
    sg_ipl = np.asarray(X["SG"]["data"]) != 0
    assert float(np.abs(lh_f.astype(np.float64) - lh_ipl.astype(np.float64)).max()) > 1e3
    assert int(((sh_f >= THR) != sg_ipl).sum()) > 0
    if tag == "x63i61":
        assert int((sh_f >= THR).sum()) > 0 and int(sg_ipl.sum()) == 0            # floor predicted a segmentation; IPL returned none
    else:
        assert int((sh_f >= THR).sum()) == 0 and int(sg_ipl.sum()) == 6           # floor predicted none; IPL returned six voxels
