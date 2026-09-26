"""AIM write/read round trip on a synthetic v020 header, the run-length decoders, alignment by
global position (never by shape), and the NIfTI export."""
import os
import struct

import numpy as np
import pytest

from ipldt import align_to, read_aim, write_aim, write_map
from ipldt.io import TYPE_CHAR, TYPE_SHORT, _decode_rle_bits, _decode_rle_char


def make_header(dim, pos, dtype_code=TYPE_CHAR, proclog=b"!\n! Processing Log\n!\nCreated by ipldt tests\n"):
    """A minimal AIM v020 header in the layout read_aim expects: 20-byte pre-header
    (pre-header size, header size, log size, data size, 0), a 140-byte int block whose
    ints[5] is the data type, ints[6:9] the position and ints[9:12] the dimension, then the log."""
    ints = [16, 0, 0, 0, 0, int(dtype_code), *pos, *dim] + [0] * 23
    hdr = struct.pack("<35i", *ints)
    nbytes = int(np.prod(dim)) * (2 if dtype_code == TYPE_SHORT else 1)
    return struct.pack("<5i", 20, len(hdr), len(proclog), nbytes, 0) + hdr + proclog


# ------------------------------------------------------------------------------ AIM round trip
def test_aim_write_read_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    dim, pos = (9, 7, 5), (3, -2, 11)                         # (x, y, z)
    data = rng.integers(0, 256, size=dim[::-1], dtype=np.uint8)   # (z, y, x)
    path = str(tmp_path / "rt.aim")
    write_aim(path, data, make_header(dim, pos))
    r = read_aim(path)
    assert np.array_equal(r["data"], data)
    assert r["data"].dtype == np.uint8 and r["data"].shape == dim[::-1]
    assert r["dim"] == dim and r["pos"] == pos
    assert "Created by ipldt tests" in r["proclog"]
    assert os.path.getsize(path) % 512 == 0
    # the header read back is a valid template again (second generation)
    path2 = str(tmp_path / "rt2.aim")
    write_aim(path2, data[::-1], r["header"])
    r2 = read_aim(path2)
    assert np.array_equal(r2["data"], data[::-1]) and r2["dim"] == dim and r2["pos"] == pos


def test_write_aim_rewrites_type_and_size_from_a_short_template(tmp_path):
    """A template taken from a short (int16) AIM still yields a char AIM with the right data size."""
    dim, pos = (4, 3, 2), (0, 0, 0)
    data = (np.arange(24, dtype=np.uint8) % 7).reshape(2, 3, 4)
    tmpl = make_header(dim, pos, dtype_code=TYPE_SHORT)
    path = str(tmp_path / "short_tmpl.aim")
    write_aim(path, data, tmpl, pad_to_512=False)
    raw = open(path, "rb").read()
    pre = struct.unpack("<5i", raw[:20])
    ints = struct.unpack("<35i", raw[20:160])
    assert ints[5] == TYPE_CHAR and pre[3] == 24 and len(raw) == 20 + 140 + (len(tmpl) - 160) + 24
    assert np.array_equal(read_aim(path)["data"], data)


def test_read_aim_rejects_unknown_type(tmp_path):
    path = str(tmp_path / "bad.aim")
    hdr = make_header((2, 2, 2), (0, 0, 0), dtype_code=0x00990099)
    open(path, "wb").write(hdr + b"\0" * 8)
    with pytest.raises(ValueError):
        read_aim(path)


def test_rle_char_decoder():
    buf = np.array([5, 3, 0, 2, 7, 1], np.uint8)          # (value, count) pairs
    assert np.array_equal(_decode_rle_char(buf, 6), [5, 5, 5, 0, 0, 7])
    with pytest.raises(ValueError):
        _decode_rle_char(buf, 7)


def test_rle_bits_decoder():
    # v0, v1, then run lengths; a byte < 255 ends its run with a flip, 255 is a 254-run without a flip
    buf = np.array([0, 127, 3, 2, 1], np.uint8)
    assert np.array_equal(_decode_rle_bits(buf, 6), [0, 0, 0, 127, 127, 0])
    buf = np.array([9, 4, 255, 1, 2], np.uint8)
    out = _decode_rle_bits(buf, 257)
    assert (out[:255] == 9).all() and np.array_equal(out[255:], [4, 4])


# ------------------------------------------------------------------------------ align_to
def _src():
    return dict(data=np.arange(1, 25, dtype=np.int16).reshape(2, 3, 4), dim=(4, 3, 2), pos=(10, 20, 30))


def test_align_to_pastes_by_global_position():
    src = _src()
    out = align_to(src, dim=(8, 6, 4), pos=(8, 18, 29))          # src sits at offset (2, 2, 1) in (x, y, z)
    assert out.shape == (4, 6, 8) and out.dtype == np.int16
    assert np.array_equal(out[1:3, 2:5, 2:6], src["data"])
    assert int(out.sum()) == int(src["data"].sum())
    out[1:3, 2:5, 2:6] = 0
    assert not out.any()


def test_align_to_same_shape_different_position_shifts():
    """Never by shape: identical grids at different positions are shifted, not copied."""
    src = _src()
    same = align_to(src, dim=src["dim"], pos=src["pos"])
    assert np.array_equal(same, src["data"])
    shifted = align_to(src, dim=(4, 3, 2), pos=(11, 20, 30))
    assert np.array_equal(shifted[:, :, :3], src["data"][:, :, 1:]) and not shifted[:, :, 3].any()
    shifted = align_to(src, dim=(4, 3, 2), pos=(9, 20, 30))
    assert np.array_equal(shifted[:, :, 1:], src["data"][:, :, :3]) and not shifted[:, :, 0].any()
    shifted = align_to(src, dim=(4, 3, 2), pos=(10, 20, 31))
    assert np.array_equal(shifted[0], src["data"][1]) and not shifted[1].any()


def test_align_to_clips_and_fills():
    src = _src()
    none = align_to(src, dim=(4, 3, 2), pos=(100, 100, 100), fill=7)
    assert (none == 7).all()
    part = align_to(src, dim=(2, 2, 2), pos=(12, 21, 30))        # only x 12..13, y 21..22 overlap
    assert np.array_equal(part, src["data"][:, 1:3, 2:4])


def test_align_to_round_trip():
    src = _src()
    big = dict(data=align_to(src, dim=(9, 8, 7), pos=(5, 15, 25)), dim=(9, 8, 7), pos=(5, 15, 25))
    assert np.array_equal(align_to(big, dim=src["dim"], pos=src["pos"]), src["data"])


# ------------------------------------------------------------------------------ NIfTI
def test_write_map_nifti_round_trip(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    data = (np.arange(60, dtype=np.int16) % 5).reshape(3, 4, 5)
    path = str(tmp_path / "map.nii.gz")
    write_map(path, data, el_size_mm=(0.0607, 0.0607, 0.0607), pos=(10, 20, 30))
    img = sitk.ReadImage(path)
    assert np.array_equal(sitk.GetArrayFromImage(img), data)
    assert np.allclose(img.GetSpacing(), (0.0607,) * 3)
    assert np.allclose(img.GetOrigin(), (10 * 0.0607, 20 * 0.0607, 30 * 0.0607))
    with pytest.raises(ValueError):
        write_map(str(tmp_path / "x.aim"), data)                 # AIM output needs a template header
