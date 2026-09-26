"""ormir_bqrl.slicer -- the 3D Slicer interchange.

The NRRD codec (raw / gzip, 3-D and 4-D, key variants, the documented refusals, anatomical-frame conversion);
the .seg.nrrd writer read back by its own reader AND by SimpleITK's independent NRRD reader (a valid NRRD
with the NIfTI's geometry); the labelmap NIfTI round trip (sform flip in the bytes, position recovered);
edited copies saved three ways landing on the right AIM voxels; cropped / padded extents pasted by global
position; resampled / shifted / reoriented inputs refused; the selection rules; export_run; and, when the
repository's validation/ormir_run_PFJ-0be66a folder is present, the real PFJ-0be66a masks (marked slow)."""
import gzip
import os
import struct

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from ipldt.io import TYPE_CHAR, read_aim, write_aim, write_nifti          # noqa: E402
from ormir_bqrl import slicer                                              # noqa: E402
from ormir_bqrl.slicer import (COLORS, LABELS, GridSpec, NrrdError, as_grid, export_run, labels_from_masks,   # noqa: E402
                               masks_from_labels, nrrd_geometry, read_mask, read_nrrd, read_seg_nrrd,
                               write_labelmap_nifti, write_nrrd, write_seg_nrrd)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PFJ_0be66a_DIR = os.path.join(REPO, "validation", "ormir_run_PFJ-0be66a")

EL = (0.06069973483681679, 0.060700856149196625, 0.06069643050432205)      # PFJ-0be66a's (anisotropic) header values
DIM = (30, 24, 8)                                                            # (x, y, z)
POS = (793, 92, 168)
GRID = GridSpec(DIM, POS, EL)


# ------------------------------------------------------------------------------------------ helpers
def make_labels(grid=GRID, r_out=9.0, r_in=6.0, empty_ends=True):
    """A cortical ring (label 1) around a trabecular disc (label 2) on every slice but the first and last."""
    z, y, x = as_grid(grid).shape_zyx
    yy, xx = np.mgrid[:y, :x]
    r = np.sqrt((yy - (y - 1) / 2) ** 2 + (xx - (x - 1) / 2) ** 2)
    lab2d = np.where(r <= r_out, np.where(r >= r_in, 1, 2), 0).astype(np.uint8)
    labels = np.broadcast_to(lab2d, (z, y, x)).copy()
    if empty_ends and z > 2:
        labels[0] = 0
        labels[-1] = 0
    return labels


def _vax_pack(x):
    """Inverse of ipldt.io._vax_f: value = (-1)^s (0.5 + f / 2^24) 2^(e - 128) as a signed int32."""
    import math
    if x == 0:
        return 0
    m, e = math.frexp(x)
    f = int(round((m - 0.5) * 2 ** 24))
    w1 = ((e + 128) << 7) | (f >> 16)
    w2 = f & 0xFFFF
    u = (w2 << 16) | w1
    return u - 2 ** 32 if u >= 2 ** 31 else u


def aim_header(dim, pos, el, proclog=b"!\n! Processing Log\n!\nCreated by ormir_bqrl tests\n"):
    """A v020 char AIM header in ipldt.io's layout with VAX-float element sizes in ints 27..29."""
    ints = [16, 0, 0, 0, 0, TYPE_CHAR, *pos, *dim] + [0] * 23
    ints[27:30] = [_vax_pack(e) for e in el]
    hdr = struct.pack("<35i", *ints)
    return struct.pack("<5i", 20, len(hdr), len(proclog), int(np.prod(dim)), 0) + hdr + proclog


def write_slicer_style_nifti(path, arr, grid):
    """What Slicer does when it saves a labelmap as NIfTI: an ITK write with our LPS geometry."""
    g = as_grid(grid)
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    img.SetSpacing([float(e) for e in g.el])
    img.SetOrigin(list(g.origin_mm))
    sitk.WriteImage(img, path)


def write_sitk_nrrd(path, arr, grid):
    """SimpleITK's own NRRD writer (gzip): an independent producer for the codec's reader."""
    g = as_grid(grid)
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    img.SetSpacing([float(e) for e in g.el])
    img.SetOrigin(list(g.origin_mm))
    sitk.WriteImage(img, path, True)


def edited(labels, seed=0):
    """An edited copy: some cortical voxels moved to trabecular, some erased, a blob added outside the bone."""
    rng = np.random.default_rng(seed)
    out = labels.copy()
    cort = np.argwhere(out == 1)
    pick = cort[rng.choice(len(cort), size=min(40, len(cort)), replace=False)]
    out[tuple(pick.T)] = 2
    trab = np.argwhere(out == 2)
    pick = trab[rng.choice(len(trab), size=min(25, len(trab)), replace=False)]
    out[tuple(pick.T)] = 0
    out[3:5, 1:3, 1:4] = 1                       # a cortical blob far from the bone, still inside the grid
    assert not np.array_equal(out, labels)
    return out


# ------------------------------------------------------------------------------------------ 1. the NRRD codec
@pytest.mark.parametrize("encoding", ["raw", "gzip"])
@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32])
def test_nrrd_codec_round_trip_3d(tmp_path, encoding, dtype):
    rng = np.random.default_rng(3)
    arr = (rng.integers(-100, 100, size=(4, 5, 6)) if dtype != np.float32 else rng.standard_normal((4, 5, 6))).astype(dtype)
    if dtype == np.uint8:
        arr = np.abs(arr).astype(np.uint8)
    path = str(tmp_path / f"rt_{encoding}.nrrd")
    write_nrrd(path, arr, EL, GRID.origin_mm, custom={"Foo": "bar baz", "Segment0_Name": "X"}, encoding=encoding)
    back, header = read_nrrd(path)
    assert np.array_equal(back, arr) and back.dtype == arr.dtype and back.shape == (4, 5, 6)
    assert header["sizes"] == [6, 5, 4] and header["dimension"] == 3 and header["encoding"] == encoding
    assert header["custom"] == {"Foo": "bar baz", "Segment0_Name": "X"}
    assert header["space"] == "left-posterior-superior" and header["kinds"] == ["domain"] * 3
    geo = nrrd_geometry(header)
    assert geo["spacing"] == EL                                    # repr() doubles: exact
    assert geo["origin"] == GRID.origin_mm
    assert np.array_equal(geo["direction"], np.eye(3))
    head = open(path, "rb").read(600).split(b"\n\n")[0].decode("ascii")
    assert head.startswith("NRRD0004\n# Complete NRRD") and f"encoding: {encoding}" in head and "Foo:=bar baz" in head


def test_nrrd_codec_4d_list_axis(tmp_path):
    arr = np.zeros((3, 4, 5, 2), np.uint8)
    arr[1, 2, 3, 0] = 1
    arr[2, 1, 4, 1] = 2
    path = str(tmp_path / "layers.nrrd")
    write_nrrd(path, arr, EL, GRID.origin_mm)
    head = open(path, "rb").read(600).split(b"\n\n")[0].decode("ascii")
    assert "sizes: 2 5 4 3" in head and "kinds: list domain domain domain" in head and "space directions: none (" in head
    back, header = read_nrrd(path)
    assert np.array_equal(back, arr) and header["list_axis"] == 0 and header["space_directions"][0] is None
    assert nrrd_geometry(header)["spacing"] == EL
    # a list axis on another NRRD axis is moved last as well
    body = np.ascontiguousarray(np.moveaxis(arr, -1, 0)).tobytes()          # C (L, z, y, x) -> sizes X Y Z L
    hdr = ("NRRD0004\ntype: unsigned char\ndimension: 4\nspace: left-posterior-superior\nsizes: 5 4 3 2\n"
           "space directions: (%r,0,0) (0,%r,0) (0,0,%r) none\nkinds: domain domain domain list\nendian: little\n"
           "encoding: raw\nspace origin: (%r,%r,%r)\n\n" % (EL + GRID.origin_mm))
    path2 = str(tmp_path / "layers_last.nrrd")
    open(path2, "wb").write(hdr.encode("ascii") + body)
    back2, header2 = read_nrrd(path2)
    assert header2["list_axis"] == 3 and np.array_equal(back2, arr)


def test_nrrd_codec_reads_header_variants(tmp_path):
    arr = (np.arange(24, dtype=np.uint8) % 3).reshape(2, 3, 4)
    lines = ["NRRD0005", "# a comment", "type: uchar", "dimension: 3", "space: LPS", "sizes: 4 3 2",
             "space directions: ( 0.0607, 0, 0 ) (0,0.0607,0) (0, 0, 0.0607)", "kinds: domain domain domain",
             "endian: little", "encoding: gz", "space origin: (1.5, 2.5, 3.5)", "content: whatever", "Segment0_ID:=A: B := C",
             "Weird Key:= value with spaces "]
    path = str(tmp_path / "variants.nrrd")
    open(path, "wb").write(("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + gzip.compress(arr.tobytes()))
    back, header = read_nrrd(path)
    assert np.array_equal(back, arr)
    assert header["custom"]["Segment0_ID"] == "A: B := C" and header["custom"]["Weird Key"] == " value with spaces "
    geo = nrrd_geometry(header)
    assert geo["spacing"] == (0.0607, 0.0607, 0.0607) and geo["origin"] == (1.5, 2.5, 3.5)


def _write_raw_nrrd(path, arr, extra_lines, space="left-posterior-superior", type_name="unsigned char", dirs=None, body=None):
    dirs = dirs or "(%r,0,0) (0,%r,0) (0,0,%r)" % EL
    lines = ["NRRD0004", f"type: {type_name}", "dimension: 3", f"space: {space}", "sizes: %d %d %d" % arr.shape[::-1],
             f"space directions: {dirs}", "kinds: domain domain domain", "encoding: raw",
             "space origin: (%r,%r,%r)" % GRID.origin_mm] + list(extra_lines)
    open(path, "wb").write(("\n".join(lines) + "\n\n").encode("ascii") + (arr.tobytes() if body is None else body))


@pytest.mark.parametrize("case, extra, type_name, message", [
    ("big", ["endian: big"], "short", "big-endian"),
    ("bzip2", ["encoding: bzip2"], "unsigned char", "encoding 'bzip2'"),
    ("detached", ["data file: other.raw"], "unsigned char", "detached"),
    ("type", [], "long long", "type 'long long'"),
    ("skip", ["byte skip: 12"], "unsigned char", "byte skip"),
])
def test_nrrd_codec_refusals(tmp_path, case, extra, type_name, message):
    arr = np.zeros((2, 3, 4), np.int16 if type_name == "short" else np.uint8)
    path = str(tmp_path / f"{case}.nrrd")
    _write_raw_nrrd(path, arr, extra, type_name=type_name)
    with pytest.raises(NrrdError, match=message):
        read_nrrd(path)


def test_nrrd_non_anatomical_space_is_refused_and_truncated_data_too(tmp_path):
    arr = np.zeros((2, 3, 4), np.uint8)
    path = str(tmp_path / "scanner.nrrd")
    _write_raw_nrrd(path, arr, [], space="scanner-xyz")
    with pytest.raises(NrrdError, match="not an anatomical"):
        nrrd_geometry(read_nrrd(path)[1])
    path = str(tmp_path / "short_body.nrrd")
    _write_raw_nrrd(path, arr, [], body=b"\0" * 5)
    with pytest.raises(NrrdError, match="bytes"):
        read_nrrd(path)
    with pytest.raises(NrrdError, match="not a NRRD"):
        read_nrrd(_touch(tmp_path / "notnrrd.nrrd", b"hello\n\n"))


def _touch(path, content):
    open(str(path), "wb").write(content)
    return str(path)


def test_ras_frame_is_converted_to_lps(tmp_path):
    """A RAS NRRD with the x / y axes and origin negated is the same LPS geometry: identical voxels and position."""
    labels = make_labels()
    ox, oy, oz = GRID.origin_mm
    path = str(tmp_path / "ras.nrrd")
    _write_raw_nrrd(path, labels, [], space="right-anterior-superior", dirs="(%r,0,0) (0,%r,0) (0,0,%r)" % (-EL[0], -EL[1], EL[2]))
    raw = open(path, "rb").read().replace(("space origin: (%r,%r,%r)" % GRID.origin_mm).encode(),
                                          ("space origin: (%r,%r,%r)" % (-ox, -oy, oz)).encode())
    open(path, "wb").write(raw)
    geo = nrrd_geometry(read_nrrd(path)[1])
    assert geo["spacing"] == EL and geo["origin"] == (ox, oy, oz) and np.array_equal(geo["direction"], np.eye(3))
    assert any("converted to LPS" in n for n in geo["notes"])
    m, info = read_mask(path, GRID, "trab")
    assert np.array_equal(m, labels == 2) and info["source_grid"]["pos_xyz"] == list(POS)
    # an LPS file whose axes are flipped is a reorientation, refused
    path2 = str(tmp_path / "flipped.nrrd")
    _write_raw_nrrd(path2, labels, [], dirs="(%r,0,0) (0,%r,0) (0,0,%r)" % (-EL[0], EL[1], EL[2]))
    with pytest.raises(ValueError, match="reorient"):
        read_mask(path2, GRID)


# ------------------------------------------------------------------------------------------ 2. .seg.nrrd
def test_seg_nrrd_round_trip_own_reader(tmp_path):
    labels = make_labels()
    path = str(tmp_path / "c.seg.nrrd")
    write_seg_nrrd(path, labels, GRID)
    back, header, segments = read_seg_nrrd(path)
    assert np.array_equal(back, labels) and back.dtype == np.uint8 and back.ndim == 3
    assert [s["name"] for s in segments] == ["Cortical", "Trabecular"]
    assert [s["label"] for s in segments] == [1, 2] and [s["layer"] for s in segments] == [0, 0]
    assert [s["id"] for s in segments] == ["Cortical", "Trabecular"]
    for s, label in zip(segments, (1, 2)):
        nz = np.nonzero(labels == label)
        assert s["extent"] == [int(nz[2].min()), int(nz[2].max()), int(nz[1].min()), int(nz[1].max()), int(nz[0].min()), int(nz[0].max())]
        assert s["color"] == pytest.approx(COLORS[label])
        assert "Bone" in s["tags"]
    c = header["custom"]
    assert c["Segmentation_MasterRepresentation"] == "Binary labelmap" and c["Segmentation_SourceRepresentation"] == "Binary labelmap"
    assert c["Segmentation_ContainedRepresentationNames"] == "Binary labelmap|" and c["Segmentation_ReferenceImageExtentOffset"] == "0 0 0"
    head = open(path, "rb").read(3000).split(b"\n\n")[0].decode("ascii").split("\n")
    assert head[:3] == ["NRRD0004", "# Complete NRRD file format specification at:", "# http://teem.sourceforge.net/nrrd/format.html"]
    assert "type: unsigned char" in head and "dimension: 3" in head and "encoding: gzip" in head
    assert ("sizes: %d %d %d" % DIM) in head
    assert ("space origin: (%r,%r,%r)" % GRID.origin_mm) in head
    # an empty segment carries Slicer's empty extent
    path2 = str(tmp_path / "empty_trab.seg.nrrd")
    write_seg_nrrd(path2, np.where(labels == 2, 0, labels).astype(np.uint8), GRID)
    assert read_seg_nrrd(path2)[2][1]["extent"] == [0, -1, 0, -1, 0, -1]


def test_seg_nrrd_is_valid_nrrd_for_simpleitk_with_the_niftis_geometry(tmp_path):
    labels = make_labels()
    seg_path = str(tmp_path / "c.seg.nrrd")
    nii_path = str(tmp_path / "c_labelmap.nii.gz")
    write_seg_nrrd(seg_path, labels, GRID)
    write_labelmap_nifti(nii_path, labels, GRID)
    seg = sitk.ReadImage(seg_path)
    nii = sitk.ReadImage(nii_path)
    assert np.array_equal(sitk.GetArrayFromImage(seg), labels) and seg.GetPixelID() == sitk.sitkUInt8
    assert tuple(seg.GetOrigin()) == GRID.origin_mm                           # double-exact
    assert tuple(seg.GetSpacing()) == EL
    assert np.allclose(seg.GetDirection(), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), atol=1e-12)   # ITK renormalises: 1 - 1e-16
    assert seg.GetSize() == nii.GetSize() == DIM
    assert np.array_equal(sitk.GetArrayFromImage(nii), labels)
    assert np.allclose(nii.GetOrigin(), seg.GetOrigin(), atol=1e-5) and np.allclose(nii.GetSpacing(), seg.GetSpacing(), rtol=1e-6)
    assert np.allclose(nii.GetDirection(), seg.GetDirection(), atol=1e-12)
    assert seg.GetMetaData("Segment0_LabelValue") == "1" and seg.GetMetaData("Segment1_Name") == "Trabecular"
    assert seg.GetMetaData("Segmentation_MasterRepresentation") == "Binary labelmap"


# ------------------------------------------------------------------------------------------ 3. labelmap NIfTI
def test_labelmap_nifti_round_trip_and_sform(tmp_path):
    labels = make_labels()
    path = str(tmp_path / "lm.nii.gz")
    write_labelmap_nifti(path, labels, GRID)
    for select, expect in (("cort", labels == 1), ("trab", labels == 2), ("periosteal", labels > 0), (None, labels > 0),
                           ("Cortical", labels == 1), ("TRABECULAR", labels == 2)):
        m, info = read_mask(path, GRID, select)
        assert np.array_equal(m, expect), select
        assert info["voxels"] == int(expect.sum()) and info["dropped_outside_grid"] == 0 and info["format"] == "image"
        assert info["source_grid"] == dict(dim_xyz=list(DIM), pos_xyz=list(POS), el_size_mm=pytest.approx(list(EL), rel=1e-7))
        assert len(info["sha256"]) == 64 and info["selection"].startswith("labelmap")
    raw = gzip.open(path, "rb").read(352)
    qform_code, sform_code = struct.unpack("<hh", raw[252:256])
    srow = struct.unpack("<12f", raw[280:328])
    ox, oy, oz = GRID.origin_mm
    assert qform_code > 0 and sform_code > 0
    assert srow[0] == pytest.approx(-EL[0], rel=1e-6) and srow[5] == pytest.approx(-EL[1], rel=1e-6) and srow[10] == pytest.approx(EL[2], rel=1e-6)
    assert srow[3] == pytest.approx(-ox, rel=1e-6) and srow[7] == pytest.approx(-oy, rel=1e-6) and srow[11] == pytest.approx(oz, rel=1e-6)
    assert all(abs(srow[i]) < 1e-12 for i in (1, 2, 4, 6, 8, 9))
    with pytest.raises(ValueError, match="select"):
        read_mask(path, GRID, "bone")


@pytest.mark.parametrize("pos", [(0, 0, 0), (-7, 3, 12), (100000, 50000, 20000)])
def test_labelmap_nifti_recovers_the_position(tmp_path, pos):
    grid = GridSpec((7, 6, 5), pos, EL)
    labels = make_labels(grid, r_out=2.5, r_in=1.5, empty_ends=False)
    path = str(tmp_path / "pos.nii.gz")
    write_labelmap_nifti(path, labels, grid)
    m, info = read_mask(path, grid, None)
    assert info["source_grid"]["pos_xyz"] == list(pos) and np.array_equal(m, labels > 0)
    assert slicer.grid_from_file(path) == GridSpec((7, 6, 5), pos, pytest.approx(EL, rel=1e-7))


# ------------------------------------------------------------------------------------------ 4. edits by position
def test_edited_copies_land_on_the_right_aim_voxels(tmp_path):
    labels = make_labels()
    ed = edited(labels)
    files = {"seg.nrrd": str(tmp_path / "e.seg.nrrd"), "nii.gz": str(tmp_path / "e.nii.gz"), "sitk.nrrd": str(tmp_path / "e.nrrd")}
    write_seg_nrrd(files["seg.nrrd"], ed, GRID)
    write_slicer_style_nifti(files["nii.gz"], ed, GRID)
    write_sitk_nrrd(files["sitk.nrrd"], ed, GRID)
    for kind, path in files.items():
        for select, expect in (("cort", ed == 1), ("trab", ed == 2), (None, ed > 0)):
            m, info = read_mask(path, GRID, select)
            assert np.array_equal(m, expect), (kind, select)
            assert info["source_grid"]["pos_xyz"] == list(POS)
        c, t = masks_from_labels(ed)
        assert np.array_equal(read_mask(path, GRID, "trab")[0] ^ (labels == 2), t ^ (labels == 2))
    assert read_mask(files["sitk.nrrd"], GRID)[1]["format"] == "nrrd"
    assert read_mask(files["seg.nrrd"], GRID)[1]["format"] == "seg.nrrd"


def test_cropped_and_padded_extents_paste_by_global_position(tmp_path):
    labels = make_labels()
    nz = np.nonzero(labels)
    lo = [max(int(n.min()) - 2, 0) for n in nz]
    hi = [min(int(n.max()) + 3, s) for n, s in zip(nz, labels.shape)]
    sub = labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    sub_grid = GridSpec((sub.shape[2], sub.shape[1], sub.shape[0]), (POS[0] + lo[2], POS[1] + lo[1], POS[2] + lo[0]), EL)
    assert sub.shape != labels.shape
    p_seg, p_nii = str(tmp_path / "crop.seg.nrrd"), str(tmp_path / "crop.nii.gz")
    write_seg_nrrd(p_seg, sub, sub_grid)
    write_slicer_style_nifti(p_nii, sub, sub_grid)
    for p in (p_seg, p_nii):
        for select, expect in (("cort", labels == 1), ("trab", labels == 2), (None, labels > 0)):
            m, info = read_mask(p, GRID, select)
            assert np.array_equal(m, expect) and info["dropped_outside_grid"] == 0
            assert info["source_grid"]["dim_xyz"] == list(sub_grid.dim) and info["source_grid"]["pos_xyz"] == list(sub_grid.pos)
    # padded 3 voxels beyond the AIM grid on every side, with voxels set in the padding: dropped and counted
    pad = 3
    big = np.zeros((labels.shape[0] + 2 * pad, labels.shape[1] + 2 * pad, labels.shape[2] + 2 * pad), np.uint8)
    big[pad:-pad, pad:-pad, pad:-pad] = labels
    big[0, :, :] = 2
    big[:, -1, :] = 1
    n_outside = int((big > 0).sum()) - int((labels > 0).sum())
    big_grid = GridSpec((big.shape[2], big.shape[1], big.shape[0]), (POS[0] - pad, POS[1] - pad, POS[2] - pad), EL)
    p_big = str(tmp_path / "pad.seg.nrrd")
    write_seg_nrrd(p_big, big, big_grid)
    m, info = read_mask(p_big, GRID, None)
    assert np.array_equal(m, labels > 0) and info["dropped_outside_grid"] == n_outside and info["file_voxels"] == int((big > 0).sum())
    m, info = read_mask(p_big, GRID, "trab")
    assert np.array_equal(m, labels == 2) and info["dropped_outside_grid"] == int((big[0] == 2).sum())


def test_resampled_shifted_reoriented_and_disjoint_inputs_are_refused(tmp_path):
    labels = make_labels()
    # resampled (spacing doubled)
    p = str(tmp_path / "resampled.nii.gz")
    img = sitk.GetImageFromArray(labels)
    img.SetSpacing([2 * e for e in EL])
    img.SetOrigin(list(GRID.origin_mm))
    sitk.WriteImage(img, p)
    with pytest.raises(ValueError, match="resampled"):
        read_mask(p, GRID)
    # shifted by half a voxel: off the lattice
    p = str(tmp_path / "shifted.nii.gz")
    img = sitk.GetImageFromArray(labels)
    img.SetSpacing(list(EL))
    img.SetOrigin([GRID.origin_mm[0] + 0.5 * EL[0], GRID.origin_mm[1], GRID.origin_mm[2]])
    sitk.WriteImage(img, p)
    with pytest.raises(ValueError, match="shifted"):
        read_mask(p, GRID)
    # a whole-voxel shift is fine (pasted by position, not refused)
    p = str(tmp_path / "shifted_whole.nii.gz")
    img.SetOrigin([GRID.origin_mm[0] + EL[0], GRID.origin_mm[1], GRID.origin_mm[2]])
    sitk.WriteImage(img, p)
    m, info = read_mask(p, GRID, "trab")
    assert info["source_grid"]["pos_xyz"] == [POS[0] + 1, POS[1], POS[2]]
    assert np.array_equal(m[:, :, 1:], (labels == 2)[:, :, :-1]) and not m[:, :, 0].any()
    # reoriented (x axis flipped in LPS)
    p = str(tmp_path / "reoriented.nii.gz")
    img = sitk.GetImageFromArray(labels)
    img.SetSpacing(list(EL))
    img.SetOrigin(list(GRID.origin_mm))
    img.SetDirection((-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(img, p)
    with pytest.raises(ValueError, match="reorient"):
        read_mask(p, GRID)
    # no overlap with the grid at all
    p = str(tmp_path / "elsewhere.seg.nrrd")
    write_seg_nrrd(p, labels, GridSpec(DIM, (POS[0] + 1000, POS[1], POS[2]), EL))
    with pytest.raises(ValueError, match="does not overlap"):
        read_mask(p, GRID)
    # a spacing within tolerance (float32 header) passes: 1e-7 relative
    p = str(tmp_path / "fine.nii.gz")
    img = sitk.GetImageFromArray(labels)
    img.SetSpacing([e * (1 + 1e-7) for e in EL])
    img.SetOrigin(list(GRID.origin_mm))
    sitk.WriteImage(img, p)
    assert np.array_equal(read_mask(p, GRID, "cort")[0], labels == 1)
    with pytest.raises(FileNotFoundError):
        read_mask(str(tmp_path / "missing.nii.gz"), GRID)


# ------------------------------------------------------------------------------------------ 5. selection rules
def test_selection_rules(tmp_path):
    labels = make_labels()
    # a Slicer-style binary mask (0 / 1) is non-zero = inside under every select
    p = str(tmp_path / "binary.nii.gz")
    write_slicer_style_nifti(p, (labels == 2).astype(np.uint8), GRID)
    for select in ("cort", "trab", "periosteal", None):
        m, info = read_mask(p, GRID, select)
        assert np.array_equal(m, labels == 2) and info["selection"].startswith("non-zero") and info["labels_used"] == [1]
    # ours carry 127: same rule
    p = str(tmp_path / "ours.nii.gz")
    write_nifti(p, (labels == 1).astype(np.uint8) * 127, EL, POS)
    assert np.array_equal(read_mask(p, GRID, "trab")[0], labels == 1)
    # a {0, 1, 2} file is a labelmap; a {0, 1, 2, 3} file is not
    p = str(tmp_path / "three.nii.gz")
    three = labels.copy()
    three[1, 0, 0] = 3
    write_slicer_style_nifti(p, three, GRID)
    m, info = read_mask(p, GRID, "cort")
    assert np.array_equal(m, three > 0) and info["labels_used"] == [1, 2, 3]
    # a .seg.nrrd with one oddly named segment selects it whatever the select
    p = str(tmp_path / "odd.seg.nrrd")
    write_seg_nrrd(p, (labels == 2).astype(np.uint8) * 5, GRID, table={5: "my edit"}, colors={5: (1, 1, 0)})
    for select in ("cort", "trab", None):
        m, info = read_mask(p, GRID, select)
        assert np.array_equal(m, labels == 2) and info["segments"] == [dict(name="my edit", label=5, layer=0)]
    # two segments: by name (case-insensitive), else by label value, else refused with the listing
    p = str(tmp_path / "renamed.seg.nrrd")
    write_seg_nrrd(p, labels, GRID, table={1: "CORTICAL bone", 2: "inner"})
    assert np.array_equal(read_mask(p, GRID, "cort")[0], labels == 1)
    assert np.array_equal(read_mask(p, GRID, "trab")[0], labels == 2)          # by label value 2
    p = str(tmp_path / "unknown.seg.nrrd")
    write_seg_nrrd(p, np.where(labels == 2, 7, labels).astype(np.uint8), GRID, table={1: "outer", 7: "inner"})
    with pytest.raises(ValueError, match="no segment for 'Trabecular'.*'outer' \\(label 1, layer 0\\), 'inner' \\(label 7"):
        read_mask(p, GRID, "trab")
    assert np.array_equal(read_mask(p, GRID, None)[0], labels > 0)
    # an .AIM mask is aligned by its own header position (a sub-box at an offset)
    sub = (labels == 1)[2:6, 3:20, 4:25].astype(np.uint8) * 127
    p = str(tmp_path / "cort.aim")
    write_aim(p, sub, aim_header((sub.shape[2], sub.shape[1], sub.shape[0]), (POS[0] + 4, POS[1] + 3, POS[2] + 2), EL))
    assert read_aim(p)["el_size_mm"] == pytest.approx(EL, rel=1e-7)
    m, info = read_mask(p, GRID, "cort")
    expect = np.zeros(labels.shape, bool)
    expect[2:6, 3:20, 4:25] = (labels == 1)[2:6, 3:20, 4:25]
    assert np.array_equal(m, expect) and info["format"] == "aim" and info["source_grid"]["pos_xyz"] == [POS[0] + 4, POS[1] + 3, POS[2] + 2]
    # a two-layer .seg.nrrd (overlapping segments, as Slicer writes them) reads per layer
    layers = np.zeros(labels.shape + (2,), np.uint8)
    layers[..., 0] = (labels == 1) * 1
    layers[..., 1] = (labels > 0) * 1                                          # overlaps the cortex: its own layer
    custom = {"Segment0_ID": "Cortical", "Segment0_Name": "Cortical", "Segment0_LabelValue": "1", "Segment0_Layer": "0",
              "Segment1_ID": "Trabecular", "Segment1_Name": "Trabecular", "Segment1_LabelValue": "1", "Segment1_Layer": "1",
              "Segmentation_MasterRepresentation": "Binary labelmap"}
    p = str(tmp_path / "layers.seg.nrrd")
    write_nrrd(p, layers, EL, GRID.origin_mm, custom=custom)
    assert np.array_equal(read_mask(p, GRID, "cort")[0], labels == 1)
    assert np.array_equal(read_mask(p, GRID, "trab")[0], labels > 0)
    assert np.array_equal(read_mask(p, GRID, None)[0], labels > 0)
    arr, header, segments = read_seg_nrrd(p)
    assert arr.shape == labels.shape + (2,) and [s["layer"] for s in segments] == [0, 1]


# ------------------------------------------------------------------------------------------ 6. conversions, export_run
def test_labels_and_masks_are_inverses():
    labels = make_labels()
    c, t = masks_from_labels(labels)
    assert np.array_equal(labels_from_masks(c, t), labels) and labels_from_masks(c, t).dtype == np.uint8
    assert np.array_equal(labels_from_masks(c.astype(np.uint8) * 127, t.astype(np.uint8) * 127), labels)
    with pytest.raises(ValueError, match="overlap in 3 voxels"):
        labels_from_masks(c, t | np.isin(np.arange(labels.size).reshape(labels.shape), np.flatnonzero(c)[:3]))
    with pytest.raises(ValueError):
        write_seg_nrrd("x.seg.nrrd", labels[1:], GRID)
    with pytest.raises(ValueError):
        write_labelmap_nifti("x.nii.gz", labels[1:], GRID)


def test_as_grid_accepts_duck_typed_grids():
    class G:
        dim, pos, el = DIM, POS, EL
    assert as_grid(G()) == GRID
    assert as_grid(dict(dim=DIM, pos=POS, el_size_mm=EL)) == GRID
    assert as_grid((DIM, POS, EL)) == GRID and as_grid(GRID) is GRID
    with pytest.raises(TypeError):
        as_grid(42)
    with pytest.raises(ValueError):
        as_grid(dict(dim=DIM, pos=POS))


def test_export_run_rebuilds_the_slicer_files(tmp_path):
    labels = make_labels()
    c, t = masks_from_labels(labels)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_nifti(str(run_dir / "S1_CORT_MASK.nii.gz"), c.astype(np.uint8) * 127, EL, POS)
    write_nifti(str(run_dir / "S1_TRAB_MASK.nii.gz"), t.astype(np.uint8) * 127, EL, POS)
    res = export_run(str(run_dir))                                               # grid from the NIfTI header
    assert res["base"] == "S1" and os.path.isfile(res["seg_nrrd"]) and os.path.isfile(res["labelmap"])
    assert res["grid"]["pos_xyz"] == list(POS) and res["grid"]["dim_xyz"] == list(DIM)
    assert res["cortical_voxels"] == int(c.sum()) and res["trabecular_voxels"] == int(t.sum())
    assert np.array_equal(read_seg_nrrd(res["seg_nrrd"])[0], labels)
    assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(res["labelmap"])), labels)
    # with an ipldt-style report the grid's exact el / pos are used and the header repr is exact
    import json
    json.dump(dict(sample="S1", dim_xyz=list(DIM), pos_xyz=list(POS), el_size_mm=list(EL)), open(run_dir / "S1_report.json", "w"))
    res2 = export_run(str(run_dir), out_dir=str(tmp_path / "out"))
    assert res2["grid"]["el_size_mm"] == list(EL)
    assert nrrd_geometry(read_nrrd(res2["seg_nrrd"])[1])["spacing"] == EL
    assert np.array_equal(read_mask(res2["seg_nrrd"], GRID, "trab")[0], t)
    with pytest.raises(FileNotFoundError):
        export_run(str(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError, match="TRAB_MASK"):
        os.remove(run_dir / "S1_TRAB_MASK.nii.gz")
        export_run(str(run_dir))


# ------------------------------------------------------------------------------------------ 7. the real PFJ-0be66a masks
@pytest.mark.slow
@pytest.mark.skipif(not os.path.isfile(os.path.join(PFJ_0be66a_DIR, "X2420448_CORT_MASK.nii.gz")), reason="validation/ormir_run_PFJ-0be66a absent")
def test_pfj_0be66a_masks_round_trip(tmp_path):
    """PRX_GOBJ / CORT_MASK / TRAB_MASK of the recorded PFJ-0be66a run -> seg.nrrd + labelmap -> read back equal
    through the own codec, through SimpleITK, and through read_mask by position."""
    import json
    rep = json.load(open(os.path.join(PFJ_0be66a_DIR, "X2420448_report.json")))
    grid = GridSpec(tuple(rep["dim_xyz"]), tuple(rep["pos_xyz"]), tuple(rep["el_size_mm"]))
    prx, _ = read_mask(os.path.join(PFJ_0be66a_DIR, "X2420448_PRX_GOBJ.nii.gz"), grid)
    cort, _ = read_mask(os.path.join(PFJ_0be66a_DIR, "X2420448_CORT_MASK.nii.gz"), grid)
    trab, _ = read_mask(os.path.join(PFJ_0be66a_DIR, "X2420448_TRAB_MASK.nii.gz"), grid)
    assert np.array_equal(cort | trab, prx) and not (cort & trab).any()
    labels = labels_from_masks(cort, trab)
    res = export_run(PFJ_0be66a_DIR, out_dir=str(tmp_path))
    assert np.array_equal(read_seg_nrrd(res["seg_nrrd"])[0], labels)
    seg = sitk.ReadImage(res["seg_nrrd"])
    assert tuple(seg.GetOrigin()) == grid.origin_mm and tuple(seg.GetSpacing()) == grid.el
    assert np.array_equal(sitk.GetArrayFromImage(seg), labels)
    for select, expect in (("cort", cort), ("trab", trab), ("periosteal", prx)):
        assert np.array_equal(read_mask(res["seg_nrrd"], grid, select)[0], expect)
        assert np.array_equal(read_mask(res["labelmap"], grid, select)[0], expect)
