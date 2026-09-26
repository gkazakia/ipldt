"""ormir_bqrl.slicer -- the 3D Slicer interchange of ORMIR-BQRL.

Writes the compartment labelmap as a NIfTI and as a Slicer segmentation (.seg.nrrd, own NRRD codec: no
pynrrd, no nibabel), reads back whatever Slicer saves (.seg.nrrd single- or multi-layer, .nrrd labelmaps,
.nii / .nii.gz labelmaps or binary masks, raw or gzip NRRD bodies, Scanco .AIM masks) and puts an edited
mask back onto the AIM grid BY PHYSICAL POSITION (ipldt.io.align_to), refusing anything that was resampled,
shifted off the AIM lattice or reoriented.

Label table (one Slicer layer: the two labels partition the rendered periosteal contour exactly)

    label 0  background
    label 1  Cortical    the cortical compartment  (IPL CORT_MASK)   colour 0.85 0.25 0.20
    label 2  Trabecular  the trabecular compartment (IPL TRAB_MASK)  colour 0.20 0.45 0.90
    periosteal contour == label > 0

Geometry convention (stated once here, tested in tests/test_ormir_bqrl_slicer.py)

* Every ORMIR-BQRL volume lives on the input AIM's lattice: (x, y, z) dimension `dim`, global voxel position
  `pos` (the AIM header's pos) and per-axis element size `el` (the header's VAX floats == ITK's spacing).
  ITK/SimpleITK work in LPS; our images have identity direction, spacing `el` and origin `pos * el` (LPS).
* NIfTI (ipldt.io.write_nifti / SimpleITK): stored as sform = qform = diag(-sx, -sy, sz) with offset
  (-ox, -oy, oz) -- ITK's LPS-to-RAS sign flip, float32 header.  Read back through ITK the origin is
  (ox, oy, oz) again in LPS, and round(origin / spacing) recovers `pos` exactly (float32 headers put the
  error at ~1e-4 voxel for pos ~ 1e3).  Slicer reads and writes NIfTI through ITK, so a labelmap it saves
  comes back the same way.
* .seg.nrrd (this module): `space: left-posterior-superior`, `space directions: (sx,0,0) (0,sy,0) (0,0,sz)`,
  `space origin: (ox,oy,oz)` -- the same LPS numbers as the NIfTI, written as repr() doubles (exact round
  trip).  Slicer maps LPS to its RAS the same way for both formats, so the segmentation lands on the same
  voxels as the NIfTI (verified through SimpleITK's independent NRRD reader).
* NRRD data order: `sizes: X Y Z` lists the fastest axis first, so the body is exactly the bytes of the
  C-ordered (z, y, x) array; a 4-D file (`sizes: L X Y Z`, `kinds: list domain domain domain`,
  `space directions: none (...) (...) (...)`) is the C-ordered (z, y, x, L) array (one layer per L).
* A NRRD written in another anatomical frame (right-anterior-superior etc.) is converted to LPS by the
  per-axis sign flip, exactly as ITK's NRRD reader does; anything that is not an L/R-P/A-S/I frame is refused.
* read_mask(): spacing must equal `el` (1e-5 relative), the direction must be the identity, and
  origin / spacing must be within 1e-3 voxel of an integer per axis (or within the float32 header's own
  resolution, 2.4e-7 x |position|, whichever is larger); the volume is then pasted onto the AIM
  grid by global position, so Slicer's habit of saving a segmentation on the segments' bounding box (a
  cropped or padded extent) is handled, and voxels outside the AIM grid are dropped and counted.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
from typing import NamedTuple

import numpy as np

from ipldt.io import align_to, read_aim, write_nifti

__all__ = ["LABELS", "COLORS", "SEGMENT_ORDER", "GridSpec", "as_grid", "grid_from_image", "grid_from_file",
           "NrrdError", "read_nrrd", "write_nrrd", "nrrd_geometry", "parse_segments",
           "read_seg_nrrd", "write_seg_nrrd", "write_labelmap_nifti",
           "labels_from_masks", "masks_from_labels", "load_volume", "read_mask", "export_run", "sha256_of"]

# ------------------------------------------------------------------------------------------ the label table
LABELS = {1: "Cortical", 2: "Trabecular"}
COLORS = {1: (0.85, 0.25, 0.20), 2: (0.20, 0.45, 0.90)}
SEGMENT_ORDER = (1, 2)
SEGMENT_TAGS = ("TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list"
                "~SCT^123037004^Anatomical Structure~SCT^272673000^Bone~^^~Anatomic codes - DICOM master list~^^~^^|")

# what `select` may be called
_SELECT = {None: None, "": None, "periosteal": None, "prx": None, "all": None, "union": None, "any": None,
           "cort": 1, "cortical": 1, "cort_mask": 1, "trab": 2, "trabecular": 2, "trab_mask": 2}

# tolerances of read_mask
SPACING_RTOL = 1e-5             # relative, per axis
POSITION_TOL_VOXELS = 1e-3      # |origin / spacing - round(...)| per axis ...
POSITION_TOL_FLOAT32 = 4 * 2.0 ** -24   # ... or the float32 NIfTI header's resolution, 2.4e-7 x |position|, if larger
DIRECTION_ATOL = 1e-6           # against the identity

LABELMAP_SUFFIX = "_compartments_labelmap.nii.gz"
SEG_NRRD_SUFFIX = "_compartments.seg.nrrd"


# ------------------------------------------------------------------------------------------ the grid
class GridSpec(NamedTuple):
    """The AIM lattice: (x, y, z) dimension, global voxel position and element size in mm.  Any object with
    `dim`, `pos` and `el` (or `el_size_mm`) attributes -- ormir_bqrl.stages.Grid included -- or a read_aim-style
    dict (dim, pos, el_size_mm) is accepted wherever a grid is expected (see as_grid)."""
    dim: tuple
    pos: tuple
    el: tuple

    @property
    def origin_mm(self):
        return tuple(float(p) * float(e) for p, e in zip(self.pos, self.el))

    @property
    def shape_zyx(self):
        return tuple(int(d) for d in self.dim[::-1])

    @property
    def el_size_mm(self):
        return self.el

    def as_dict(self):
        return dict(dim_xyz=[int(d) for d in self.dim], pos_xyz=[int(p) for p in self.pos], el_size_mm=[float(e) for e in self.el])


def as_grid(grid) -> GridSpec:
    """GridSpec from a GridSpec, an object with dim / pos / el (or el_size_mm), a dict with those keys, or a
    (dim, pos, el) tuple."""
    if isinstance(grid, GridSpec):
        return grid
    if isinstance(grid, dict):
        dim, pos = grid["dim"], grid["pos"]
        el = grid["el"] if "el" in grid else grid.get("el_size_mm")
    elif hasattr(grid, "dim") and hasattr(grid, "pos"):
        dim, pos = grid.dim, grid.pos
        el = getattr(grid, "el", None)
        if el is None:
            el = getattr(grid, "el_size_mm", None)
    elif isinstance(grid, (tuple, list)) and len(grid) == 3:
        dim, pos, el = grid
    else:
        raise TypeError(f"not a grid: {type(grid).__name__}")
    if el is None:
        raise ValueError("the grid carries no element size (el / el_size_mm)")
    dim = tuple(int(d) for d in dim)
    pos = tuple(int(p) for p in pos)
    el = tuple(float(e) for e in el)
    if len(dim) != 3 or len(pos) != 3 or len(el) != 3 or any(d <= 0 for d in dim) or any(e <= 0 for e in el):
        raise ValueError(f"malformed grid: dim {dim}, pos {pos}, el {el}")
    return GridSpec(dim, pos, el)


def _pos_from_origin(origin, spacing, what="the file"):
    """Global voxel position from an LPS origin / spacing pair; refused when off the lattice."""
    pos = []
    for axis, o, s in zip("xyz", origin, spacing):
        pf = float(o) / float(s)
        p = int(round(pf))
        tol = max(POSITION_TOL_VOXELS, abs(pf) * POSITION_TOL_FLOAT32)      # float32 NIfTI headers: ~2.4e-7 x |pos|
        if abs(pf - p) > tol:
            raise ValueError(f"shifted: {what} is not on the AIM lattice (axis {axis}: origin {o!r} mm / spacing {s!r} mm = "
                             f"{pf:.4f} voxels, {abs(pf - p):.3g} voxels off an integer, tolerance {tol:.2g}); it was "
                             "resampled or moved -- edit the run's own masks in Slicer instead of a resampled copy")
        pos.append(p)
    return tuple(pos)


def grid_from_image(img) -> GridSpec:
    """GridSpec of a SimpleITK image whose geometry is ours (identity direction, origin on the lattice)."""
    direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
    if not np.allclose(direction, np.eye(3), atol=DIRECTION_ATOL):
        raise ValueError("reorient in Slicer: the image's IJK-to-LPS axes are not the AIM's "
                         f"(direction {np.round(direction, 6).tolist()})")
    spacing = tuple(float(s) for s in img.GetSpacing())
    return GridSpec(tuple(int(n) for n in img.GetSize()), _pos_from_origin(img.GetOrigin(), spacing, "the image"), spacing)


def grid_from_file(path) -> GridSpec:
    """GridSpec of an image file on our conventions (AIM: its header; NRRD: own codec; else SimpleITK)."""
    v = load_volume(path)
    return GridSpec(v["dim"], v["pos"], v["spacing"])


# ------------------------------------------------------------------------------------------ NRRD codec
class NrrdError(ValueError):
    """A NRRD file this codec does not read (the message says what to do in Slicer)."""


_NRRD_DTYPES = {
    "unsigned char": np.uint8, "uchar": np.uint8, "uint8": np.uint8, "uint8_t": np.uint8,
    "signed char": np.int8, "int8": np.int8, "int8_t": np.int8,
    "short": np.int16, "short int": np.int16, "signed short": np.int16, "signed short int": np.int16, "int16": np.int16, "int16_t": np.int16,
    "unsigned short": np.uint16, "unsigned short int": np.uint16, "ushort": np.uint16, "uint16": np.uint16, "uint16_t": np.uint16,
    "int": np.int32, "signed int": np.int32, "int32": np.int32, "int32_t": np.int32,
    "unsigned int": np.uint32, "uint": np.uint32, "uint32": np.uint32, "uint32_t": np.uint32,
    "float": np.float32, "double": np.float64,
}
_NRRD_TYPE_NAME = {np.dtype(np.uint8): "unsigned char", np.dtype(np.int8): "signed char", np.dtype(np.int16): "short",
                   np.dtype(np.uint16): "unsigned short", np.dtype(np.int32): "int", np.dtype(np.uint32): "unsigned int",
                   np.dtype(np.float32): "float", np.dtype(np.float64): "double"}
_VEC_RX = re.compile(r"\(([^)]*)\)|none", re.I)
_SPACE_SIGNS = {"left": ("x", 1.0), "right": ("x", -1.0), "posterior": ("y", 1.0), "anterior": ("y", -1.0),
                "superior": ("z", 1.0), "inferior": ("z", -1.0)}
_SPACE_ABBREV = {"l": "left", "r": "right", "p": "posterior", "a": "anterior", "s": "superior", "i": "inferior"}


def _norm_key(key):
    return re.sub(r"\s+", "", key.strip().lower())


def _parse_vectors(text):
    out = []
    for m in _VEC_RX.finditer(text):
        if m.group(1) is None:
            out.append(None)
        else:
            out.append(tuple(float(t) for t in m.group(1).replace(",", " ").split()))
    return out


def _space_signs(space):
    """Per-axis sign that maps a NRRD anatomical frame onto LPS, or None when the frame is not anatomical."""
    if space is None:
        return None
    s = space.strip().lower()
    parts = s.split("-") if "-" in s else ([_SPACE_ABBREV.get(c, c) for c in s] if len(s) == 3 else [s])
    if len(parts) != 3 or any(p not in _SPACE_SIGNS for p in parts):
        return None
    signs = {}
    for p in parts:
        axis, sign = _SPACE_SIGNS[p]
        if axis in signs:
            return None
        signs[axis] = sign
    if len(signs) != 3:
        return None
    return (signs["x"], signs["y"], signs["z"])


def _read_header(fh):
    """Parse the header of an open binary NRRD stream; returns (magic, fields, custom); the stream is left at
    the first data byte."""
    magic = fh.readline().rstrip(b"\r\n")
    if not magic.startswith(b"NRRD000") or len(magic) != 8 or not magic[7:8].isdigit():
        raise NrrdError(f"not a NRRD file (magic {magic[:12]!r})")
    fields, custom = {}, {}
    while True:
        raw = fh.readline()
        if raw == b"":
            raise NrrdError("truncated NRRD header (no blank line before the data)")
        line = raw.rstrip(b"\r\n")
        if line == b"":
            break
        if line.startswith(b"#"):
            continue
        text = line.decode("utf-8", "replace")
        i_custom, i_field = text.find(":="), text.find(": ")
        if i_custom >= 0 and (i_field < 0 or i_custom < i_field):
            custom[text[:i_custom].strip()] = text[i_custom + 2:]
        elif i_field >= 0:
            fields[_norm_key(text[:i_field])] = text[i_field + 2:].strip()
        elif text.endswith(":"):
            fields[_norm_key(text[:-1])] = ""
        else:
            raise NrrdError(f"unparsable NRRD header line: {text!r}")
    return magic.decode("ascii"), fields, custom


def read_nrrd(path):
    """Read a NRRD (attached data only): returns (array, header).

    array   C-ordered (z, y, x) for a 3-D file; (z, y, x, L) for a 4-D file whose non-domain axis (`kinds`
            list / vector, or a `none` space direction) is moved last, whatever NRRD axis it was on
    header  dict(magic, fields, custom, type (numpy dtype), dimension, sizes (NRRD order), encoding, endian,
            kinds, space, space_directions (NRRD order, None for the list axis), space_origin, list_axis)
    Accepts `key: value` fields and `key:=value` custom fields, comments, raw / gzip / gz encodings, little
    endian, dimension 3 or 4.  Refuses big endian, bzip2 / hex / ascii encodings, detached data files, line /
    byte skips and unknown types with a message naming the offending field."""
    with open(path, "rb") as fh:
        magic, fields, custom = _read_header(fh)
        body = fh.read()
    if "datafile" in fields:
        raise NrrdError(f"{os.path.basename(path)}: detached NRRD data files are not supported (data file: {fields['datafile']}); "
                        "save as a single .nrrd / .seg.nrrd from Slicer")
    for k, name in (("lineskip", "line skip"), ("byteskip", "byte skip")):
        if k in fields and fields[k].strip() not in ("0", ""):
            raise NrrdError(f"{os.path.basename(path)}: NRRD '{name}: {fields[k].strip()}' is not supported")
    try:
        ndim = int(fields["dimension"])
        sizes = [int(t) for t in fields["sizes"].split()]
        type_name = fields["type"].strip().lower()
    except KeyError as exc:
        raise NrrdError(f"{os.path.basename(path)}: NRRD header lacks the '{exc.args[0]}' field") from None
    if type_name not in _NRRD_DTYPES:
        raise NrrdError(f"{os.path.basename(path)}: NRRD type '{fields['type']}' is not supported")
    dtype = np.dtype(_NRRD_DTYPES[type_name])
    if ndim not in (3, 4) or len(sizes) != ndim:
        raise NrrdError(f"{os.path.basename(path)}: NRRD dimension {ndim} with sizes {sizes} is not a 3-D volume or a 4-D layer stack")
    endian = fields.get("endian", "little").strip().lower()
    if endian == "big":
        raise NrrdError(f"{os.path.basename(path)}: big-endian NRRD data is not supported; resave from Slicer (little endian)")
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    encoding = fields.get("encoding", "raw").strip().lower()
    if encoding in ("gzip", "gz"):
        try:
            body = gzip.decompress(body)
        except (OSError, EOFError) as exc:
            raise NrrdError(f"{os.path.basename(path)}: cannot gunzip the NRRD data ({exc})") from None
    elif encoding != "raw":
        raise NrrdError(f"{os.path.basename(path)}: NRRD encoding '{encoding}' is not supported (raw or gzip only); "
                        "resave from Slicer with compression on or off")
    nvox = int(np.prod(sizes))
    nbytes = nvox * dtype.itemsize
    if len(body) < nbytes:
        raise NrrdError(f"{os.path.basename(path)}: NRRD data is {len(body)} bytes, {nbytes} expected for sizes {sizes} {dtype}")
    arr = np.frombuffer(body[:nbytes], dtype=dtype).reshape(sizes[::-1])       # C order: NRRD axis 0 is the fastest
    if dtype.itemsize > 1:
        arr = arr.astype(dtype.newbyteorder("="))
    kinds = fields["kinds"].split() if "kinds" in fields else None
    dirs = _parse_vectors(fields["spacedirections"]) if "spacedirections" in fields else None
    if dirs is not None and len(dirs) != ndim:
        raise NrrdError(f"{os.path.basename(path)}: 'space directions' lists {len(dirs)} entries for dimension {ndim}")
    origin = None
    if "spaceorigin" in fields:
        v = _parse_vectors(fields["spaceorigin"])
        origin = v[0] if v and v[0] is not None else None
    # the non-domain axis of a 4-D file: kinds says list / vector / ..., or the direction is 'none'
    list_axis = None
    if ndim == 4:
        cand = set()
        if kinds is not None:
            cand |= {i for i, k in enumerate(kinds) if k.lower() not in ("domain", "space")}
        if dirs is not None:
            cand |= {i for i, d in enumerate(dirs) if d is None}
        if len(cand) != 1:
            raise NrrdError(f"{os.path.basename(path)}: a 4-D NRRD needs exactly one non-domain (list) axis; kinds {kinds}, "
                            f"space directions {fields.get('spacedirections')}")
        list_axis = cand.pop()
        c_axis = ndim - 1 - list_axis
        if c_axis != ndim - 1:
            arr = np.ascontiguousarray(np.moveaxis(arr, c_axis, -1))
    header = dict(magic=magic, fields=fields, custom=custom, type=np.dtype(dtype.newbyteorder("=")), dimension=ndim, sizes=sizes,
                  encoding=encoding, endian=endian, kinds=kinds, space=fields.get("space"), space_directions=dirs,
                  space_origin=origin, list_axis=list_axis)
    return arr, header


def nrrd_geometry(header):
    """LPS geometry of a read_nrrd header: dict(spacing (x, y, z), direction 3x3 (columns = axis directions),
    origin (x, y, z), notes).  Anatomical frames other than LPS are converted by the per-axis sign flip (what
    ITK's NRRD reader does); a non-anatomical `space` is refused."""
    notes = []
    fields = header["fields"]
    space = header.get("space")
    if space is None:
        signs = (1.0, 1.0, 1.0)
        notes.append("no 'space' field: the file's frame is taken as LPS")
    else:
        signs = _space_signs(space)
        if signs is None:
            raise NrrdError(f"NRRD space '{space}' is not an anatomical L/R-P/A-S/I frame; resave from Slicer")
        if signs != (1.0, 1.0, 1.0):
            notes.append(f"space '{space}' converted to LPS (axis signs {signs})")
    dirs = header.get("space_directions")
    spatial = [d for d in dirs if d is not None] if dirs is not None else None
    if spatial is None:
        if "spacings" in fields:
            sp = [float(t) for t in fields["spacings"].split() if t.lower() != "nan"]
            spatial = [tuple(sp[i] if j == i else 0.0 for j in range(3)) for i in range(3)] if len(sp) == 3 else None
            notes.append("no 'space directions': spacing taken from 'spacings', axes assumed identity")
        if spatial is None:
            raise NrrdError("NRRD header has no 'space directions' (nor 'spacings'): the geometry is unknown")
    if len(spatial) != 3 or any(len(v) != 3 for v in spatial):
        raise NrrdError(f"NRRD 'space directions' are not three 3-vectors: {spatial}")
    D = np.array(spatial, float).T * np.asarray(signs, float)[:, None]            # column j = direction of axis j (LPS)
    spacing = tuple(float(np.linalg.norm(D[:, j])) for j in range(3))
    if any(s <= 0 for s in spacing):
        raise NrrdError(f"NRRD 'space directions' contain a zero-length axis: {spatial}")
    direction = D / np.asarray(spacing)[None, :]
    origin = header.get("space_origin")
    if origin is None:
        origin = (0.0, 0.0, 0.0)
        notes.append("no 'space origin': taken as (0, 0, 0)")
    origin = tuple(float(o) * s for o, s in zip(origin, signs))
    return dict(spacing=spacing, direction=direction, origin=origin, notes=notes)


def _fmt(x):
    return repr(float(x))


def write_nrrd(path, arr, spacing, origin, custom=None, encoding="gzip", space="left-posterior-superior", comments=True):
    """Write a NRRD0004 with attached data: `arr` C-ordered (z, y, x) (3-D) or (z, y, x, L) (4-D: written with
    `sizes: L X Y Z`, `kinds: list domain domain domain`, a `none` direction for the list axis -- Slicer's
    multi-layer segmentation layout).  spacing / origin are the LPS numbers written as repr() doubles;
    `custom` is an ordered mapping written as `key:=value` lines after the fields."""
    a = np.ascontiguousarray(arr)
    if a.ndim not in (3, 4):
        raise ValueError(f"write_nrrd: a 3-D (z, y, x) or 4-D (z, y, x, L) array is needed, got shape {a.shape}")
    if a.dtype not in _NRRD_TYPE_NAME:
        raise ValueError(f"write_nrrd: dtype {a.dtype} has no NRRD type")
    if a.dtype.itemsize > 1 and a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    encoding = encoding.lower()
    if encoding not in ("raw", "gzip"):
        raise ValueError("write_nrrd: encoding must be 'raw' or 'gzip'")
    sx, sy, sz = (float(s) for s in spacing)
    dirs = [f"({_fmt(sx)},0,0)", f"(0,{_fmt(sy)},0)", f"(0,0,{_fmt(sz)})"]
    kinds = ["domain", "domain", "domain"]
    sizes = [a.shape[2], a.shape[1], a.shape[0]]
    if a.ndim == 4:
        sizes, dirs, kinds = [a.shape[3]] + sizes, ["none"] + dirs, ["list"] + kinds
    lines = ["NRRD0004"]
    if comments:
        lines += ["# Complete NRRD file format specification at:", "# http://teem.sourceforge.net/nrrd/format.html"]
    lines += [f"type: {_NRRD_TYPE_NAME[a.dtype]}", f"dimension: {a.ndim}", f"space: {space}",
              "sizes: " + " ".join(str(s) for s in sizes), "space directions: " + " ".join(dirs),
              "kinds: " + " ".join(kinds), "endian: little", f"encoding: {encoding}",
              "space origin: (" + ",".join(_fmt(o) for o in origin) + ")"]
    for k, v in (custom or {}).items():
        key = str(k).strip()
        val = str(v).replace("\r", "").replace("\n", " ")
        if not key or ":" in key or key.startswith("#"):
            raise ValueError(f"write_nrrd: bad custom key {k!r}")
        lines.append(f"{key}:={val}")
    body = a.tobytes()
    if encoding == "gzip":
        body = gzip.compress(body, compresslevel=6)
    with open(path, "wb") as fh:
        fh.write(("\n".join(lines) + "\n\n").encode("ascii"))
        fh.write(body)


# ------------------------------------------------------------------------------------------ .seg.nrrd
_SEGMENT_RX = re.compile(r"^Segment(\d+)_(\w+)$")


def parse_segments(custom):
    """The Segment<i>_* custom fields of a Slicer segmentation as a list (by index) of
    dict(index, id, name, label, layer, extent, color, tags, fields)."""
    by_index = {}
    for key, value in custom.items():
        m = _SEGMENT_RX.match(key)
        if m:
            by_index.setdefault(int(m.group(1)), {})[m.group(2)] = value
    segments = []
    for index in sorted(by_index):
        f = by_index[index]

        def _ints(text, n):
            try:
                v = [int(t) for t in text.split()]
                return v if len(v) == n else None
            except (ValueError, AttributeError):
                return None

        def _floats(text):
            try:
                return tuple(float(t) for t in text.split())
            except (ValueError, AttributeError):
                return None

        try:
            label = int(f["LabelValue"]) if "LabelValue" in f else index + 1
        except ValueError:
            label = index + 1
        try:
            layer = int(f["Layer"]) if "Layer" in f else 0
        except ValueError:
            layer = 0
        segments.append(dict(index=index, id=f.get("ID", f"Segment_{index + 1}"), name=f.get("Name", f.get("ID", f"Segment_{index + 1}")),
                             label=label, layer=layer, extent=_ints(f.get("Extent", ""), 6), color=_floats(f.get("Color", "")),
                             tags=f.get("Tags", ""), fields=f))
    return segments


def read_seg_nrrd(path):
    """(labels, header, segments): labels (z, y, x) for a single-layer file or (z, y, x, L) for a multi-layer
    one; header as read_nrrd; segments as parse_segments (empty for a plain NRRD)."""
    arr, header = read_nrrd(path)
    return arr, header, parse_segments(header["custom"])


def _extent_of(mask_zyx):
    nz = np.nonzero(mask_zyx)
    if nz[0].size == 0:
        return "0 -1 0 -1 0 -1"
    return " ".join(f"{int(nz[i].min())} {int(nz[i].max())}" for i in (2, 1, 0))


def _color_text(rgb):
    return " ".join(("%.6g" % float(c)) for c in rgb)


def write_seg_nrrd(path, labels_zyx, grid, table=None, colors=None, encoding="gzip"):
    """Write a Slicer segmentation (.seg.nrrd, 3-D single layer, gzip): the labelmap on `grid` with one
    Segment<i>_* block per label of `table` (default LABELS, in ascending label order), extents computed from
    the data, both Segmentation_MasterRepresentation (older Slicers) and _SourceRepresentation (>= 5.6)."""
    g = as_grid(grid)
    labels = np.asarray(labels_zyx)
    if labels.shape != g.shape_zyx:
        raise ValueError(f"write_seg_nrrd: labels shape {labels.shape} is not the grid's {g.shape_zyx}")
    if labels.dtype != np.uint8:
        if labels.min() < 0 or labels.max() > 255:
            raise ValueError("write_seg_nrrd: label values must fit an unsigned char")
        labels = labels.astype(np.uint8)
    table = dict(LABELS if table is None else table)
    colors = dict(COLORS if colors is None else colors)
    custom = {}
    for i, label in enumerate(sorted(table)):
        name = str(table[label])
        rgb = colors.get(label, (0.5, 0.5, 0.5))
        custom[f"Segment{i}_Color"] = _color_text(rgb)
        custom[f"Segment{i}_ColorAutoGenerated"] = "0"
        custom[f"Segment{i}_Extent"] = _extent_of(labels == label)
        custom[f"Segment{i}_ID"] = name
        custom[f"Segment{i}_LabelValue"] = str(int(label))
        custom[f"Segment{i}_Layer"] = "0"
        custom[f"Segment{i}_Name"] = name
        custom[f"Segment{i}_NameAutoGenerated"] = "0"
        custom[f"Segment{i}_Tags"] = SEGMENT_TAGS
    custom["Segmentation_ContainedRepresentationNames"] = "Binary labelmap|"
    custom["Segmentation_ConversionParameters"] = ""
    custom["Segmentation_MasterRepresentation"] = "Binary labelmap"
    custom["Segmentation_SourceRepresentation"] = "Binary labelmap"
    custom["Segmentation_ReferenceImageExtentOffset"] = "0 0 0"
    write_nrrd(path, labels, g.el, g.origin_mm, custom=custom, encoding=encoding)
    return path


def write_labelmap_nifti(path, labels_zyx, grid):
    """The uint8 labelmap as NIfTI on `grid` (ipldt.io.write_nifti: spacing el, origin pos * el)."""
    g = as_grid(grid)
    labels = np.asarray(labels_zyx)
    if labels.shape != g.shape_zyx:
        raise ValueError(f"write_labelmap_nifti: labels shape {labels.shape} is not the grid's {g.shape_zyx}")
    write_nifti(path, labels.astype(np.uint8), g.el, g.pos)
    return path


def labels_from_masks(cort, trab):
    """uint8 labelmap 1 * cort + 2 * trab; the two masks must not overlap."""
    c = np.asarray(cort) > 0
    t = np.asarray(trab) > 0
    if c.shape != t.shape:
        raise ValueError(f"labels_from_masks: shapes differ {c.shape} vs {t.shape}")
    n = int((c & t).sum())
    if n:
        raise ValueError(f"labels_from_masks: the cortical and trabecular masks overlap in {n} voxels")
    return (c.astype(np.uint8) * 1 + t.astype(np.uint8) * 2).astype(np.uint8)


def masks_from_labels(labels):
    """(cort, trab) bool from a labelmap (label 1 / label 2)."""
    a = np.asarray(labels)
    return a == 1, a == 2


# ------------------------------------------------------------------------------------------ reading masks
def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _lower_ext(path):
    name = os.path.basename(path).lower()
    if name.endswith(".seg.nrrd"):
        return ".seg.nrrd"
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(name)[1]


def load_volume(path):
    """Read any mask / labelmap file into dict(arr, spacing, direction, origin, dim, pos, format, header, segments,
    notes) in our conventions.  AIM: ipldt.io.read_aim (position from its header, direction identity).  NRRD /
    seg.nrrd: read_nrrd + nrrd_geometry (frames converted to LPS).  Everything else: SimpleITK (LPS).  The
    direction must be the identity and the origin must sit on the lattice (integer voxels), else ValueError."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"mask file not found: {path}")
    ext = _lower_ext(path)
    notes, segments, header = [], [], None
    if ext == ".aim":
        a = read_aim(path)
        arr = a["data"]
        spacing = tuple(float(e) for e in a["el_size_mm"])
        pos = tuple(int(p) for p in a["pos"])
        origin = tuple(p * s for p, s in zip(pos, spacing))
        direction = np.eye(3)
        fmt = "aim"
    elif ext in (".nrrd", ".seg.nrrd", ".nhdr"):
        arr, header = read_nrrd(path)
        geo = nrrd_geometry(header)
        spacing, direction, origin = geo["spacing"], geo["direction"], geo["origin"]
        notes += geo["notes"]
        segments = parse_segments(header["custom"])
        fmt = "seg.nrrd" if segments else "nrrd"
        if arr.ndim == 4 and not segments:
            raise NrrdError(f"{os.path.basename(path)}: a 4-D NRRD without Segment fields is not a mask")
        pos = None
    else:
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError(f"reading {ext} masks needs SimpleITK (pip install SimpleITK)") from exc
        img = sitk.ReadImage(path)
        if img.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"{os.path.basename(path)}: a vector (multi-component) image is not a mask")
        if img.GetDimension() == 4:
            if img.GetSize()[3] != 1:
                raise ValueError(f"{os.path.basename(path)}: a 4-D image with {img.GetSize()[3]} volumes is not a mask")
            img = sitk.Extract(img, list(img.GetSize()[:3]) + [0], [0, 0, 0, 0])
        elif img.GetDimension() != 3:
            raise ValueError(f"{os.path.basename(path)}: {img.GetDimension()}-D images are not masks")
        arr = sitk.GetArrayFromImage(img)
        spacing = tuple(float(s) for s in img.GetSpacing())
        origin = tuple(float(o) for o in img.GetOrigin())
        direction = np.asarray(img.GetDirection(), float).reshape(3, 3)
        fmt = "image"
        pos = None
    if not np.allclose(direction, np.eye(3), atol=DIRECTION_ATOL):
        raise ValueError(f"reorient in Slicer: the file's IJK-to-LPS axes are not the AIM's (direction "
                         f"{np.round(direction, 6).tolist()} in {os.path.basename(path)})")
    if pos is None:
        pos = _pos_from_origin(origin, spacing, os.path.basename(path))
    dim = (int(arr.shape[2]), int(arr.shape[1]), int(arr.shape[0]))
    return dict(arr=arr, spacing=spacing, direction=direction, origin=origin, dim=dim, pos=pos, format=fmt,
                header=header, segments=segments, notes=notes)


def _select_key(select):
    key = select.strip().lower() if isinstance(select, str) else select
    if key not in _SELECT:
        raise ValueError(f"select must be one of periosteal / cort / trab / None, not {select!r}")
    return _SELECT[key]


def _select_segments(v, wanted, path):
    """bool mask on the file grid from a segmentation's Segment fields; returns (mask, description, labels_used)."""
    arr, segments = v["arr"], v["segments"]
    layers = arr if arr.ndim == 4 else arr[..., None]
    for s in segments:
        if not 0 <= s["layer"] < layers.shape[-1]:
            raise ValueError(f"{os.path.basename(path)}: segment '{s['name']}' names layer {s['layer']} but the file has {layers.shape[-1]}")

    def union(segs):
        m = np.zeros(layers.shape[:3], bool)
        for s in segs:
            m |= layers[..., s["layer"]] == s["label"]
        return m

    listing = ", ".join(f"'{s['name']}' (label {s['label']}, layer {s['layer']})" for s in segments)
    if wanted is None:
        return union(segments), "union of all segments", [s["label"] for s in segments]
    if len(segments) == 1:
        s = segments[0]
        return union(segments), f"the only segment '{s['name']}' (label {s['label']})", [s["label"]]
    name = LABELS[wanted].lower()
    by_name = [s for s in segments if s["name"].strip().lower() == name or s["name"].strip().lower().startswith(name[:4])]
    if by_name:
        return union(by_name), "segment(s) named " + ", ".join(f"'{s['name']}'" for s in by_name), [s["label"] for s in by_name]
    by_label = [s for s in segments if s["label"] == wanted]
    if len(by_label) == 1:
        s = by_label[0]
        return union(by_label), f"segment '{s['name']}' by label value {wanted}", [wanted]
    raise ValueError(f"{os.path.basename(path)}: no segment for '{LABELS[wanted]}' (by name or label value {wanted}); "
                     f"the file has {listing}")


def _select_values(v, wanted):
    """bool mask from a plain image: an ORMIR-BQRL labelmap ({1, 2}) honours `wanted`, anything else is
    non-zero = inside."""
    arr = v["arr"]
    values = np.unique(arr)
    nonzero = [x for x in values.tolist() if x != 0]
    if set(nonzero) == {1, 2}:
        if wanted is None:
            return arr > 0, "labelmap: labels 1 + 2", [1, 2]
        return arr == wanted, f"labelmap: label {wanted} ({LABELS[wanted]})", [wanted]
    shown = nonzero if len(nonzero) <= 8 else nonzero[:8] + ["..."]
    return arr != 0, f"non-zero = inside (values {shown})", nonzero[:64]


def read_mask(path, grid, select=None):
    """Read a mask / labelmap / segmentation and put it on the AIM `grid` by global position.

    select   'periosteal' (or None): every non-zero voxel -- the union of all segments of a .seg.nrrd, labels
             1 + 2 of an ORMIR-BQRL labelmap, non-zero of a binary mask
             'cort' / 'trab': the Cortical / Trabecular segment of a .seg.nrrd (by name, case-insensitive, else
             by label value 1 / 2; a single-segment file selects that segment whatever its name), label 1 / 2
             of a labelmap whose non-zero values are exactly {1, 2}; any other file is non-zero = inside
    Returns (bool (z, y, x) on the grid, info) with info = dict(path, sha256, format, source_grid, select,
    selection, labels_used, segments, voxels, file_voxels, dropped_outside_grid, notes).
    Refuses (ValueError) a resampled file (spacing != el), a reoriented one (direction != identity), one that
    is off the lattice (non-integer voxel origin) and one that does not overlap the grid at all."""
    g = as_grid(grid)
    wanted = _select_key(select)
    v = load_volume(path)
    for axis, s, e in zip("xyz", v["spacing"], g.el):
        if abs(s - e) > SPACING_RTOL * e:
            raise ValueError(f"resampled: {os.path.basename(path)} has spacing {tuple(v['spacing'])} mm, the AIM element size is "
                             f"{tuple(g.el)} mm (axis {axis} differs by {abs(s - e) / e:.2e} relative); edit the run's own "
                             "masks in Slicer instead of a resampled copy")
    if v["segments"]:
        mask, how, labels_used = _select_segments(v, wanted, path)
    else:
        mask, how, labels_used = _select_values(v, wanted)
    n_file = int(mask.sum())
    pasted = align_to(dict(data=mask.astype(np.uint8), dim=v["dim"], pos=v["pos"]), g.dim, g.pos) > 0
    n_grid = int(pasted.sum())
    if n_file and not n_grid:
        raise ValueError(f"{os.path.basename(path)} does not overlap the AIM grid: file dim {v['dim']} at pos {v['pos']}, "
                         f"AIM dim {g.dim} at pos {g.pos}")
    info = dict(path=os.path.abspath(path), sha256=sha256_of(path), format=v["format"],
                source_grid=dict(dim_xyz=list(v["dim"]), pos_xyz=list(v["pos"]), el_size_mm=[float(s) for s in v["spacing"]]),
                select=select, selection=how, labels_used=labels_used,
                segments=[dict(name=s["name"], label=s["label"], layer=s["layer"]) for s in v["segments"]],
                voxels=n_grid, file_voxels=n_file, dropped_outside_grid=n_file - n_grid, notes=list(v["notes"]))
    return pasted, info


# ------------------------------------------------------------------------------------------ export of a run folder
def _grid_from_report(report_path):
    with open(report_path, "r", encoding="utf-8") as fh:
        rep = json.load(fh)
    src = rep.get("input_aim") if isinstance(rep.get("input_aim"), dict) else rep
    try:
        return GridSpec(tuple(int(d) for d in src["dim_xyz"]), tuple(int(p) for p in src["pos_xyz"]),
                        tuple(float(e) for e in src["el_size_mm"]))
    except (KeyError, TypeError, ValueError):
        return None


def _run_mask_path(run_dir, base, suffix):
    """<base>_<suffix>.nii.gz, else the .AIM an AIM-format run writes (the NIfTI path when neither exists)."""
    nii = os.path.join(run_dir, f"{base}_{suffix}.nii.gz")
    if os.path.isfile(nii):
        return nii
    for ext in (".AIM", ".aim"):
        p = os.path.join(run_dir, f"{base}_{suffix}{ext}")
        if os.path.isfile(p):
            return p
    return nii


def export_run(run_dir, base=None, out_dir=None, grid=None):
    """Rebuild <base>_compartments.seg.nrrd and <base>_compartments_labelmap.nii.gz from a run folder's
    <base>_CORT_MASK / <base>_TRAB_MASK (.nii.gz, or the .AIM of an AIM-format run; ORMIR-BQRL and ipldt-pipeline
    output alike).  The grid comes from <base>_report.json when present (the AIM's exact el / pos), else from the
    CORT_MASK header.  Returns dict(base, run_dir, seg_nrrd, labelmap, grid, cortical_voxels, trabecular_voxels,
    sources)."""
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run folder not found: {run_dir}")
    if base is None:
        bases = sorted({f[:-len("_CORT_MASK.nii.gz")] for f in os.listdir(run_dir) if f.endswith("_CORT_MASK.nii.gz")})
        if not bases:        # an AIM-format run: only folders that carry a report, so IPL's own AIM folders are not mistaken for one
            bases = sorted({f[:-len("_CORT_MASK.AIM")] for f in os.listdir(run_dir) if f.upper().endswith("_CORT_MASK.AIM")
                            and os.path.isfile(os.path.join(run_dir, f[:-len("_CORT_MASK.AIM")] + "_report.json"))})
        if not bases:
            raise FileNotFoundError(f"no <base>_CORT_MASK.nii.gz in {run_dir}")
        if len(bases) > 1:
            raise ValueError(f"{run_dir} holds several runs ({', '.join(bases)}); pass base=")
        base = bases[0]
    cort_path = _run_mask_path(run_dir, base, "CORT_MASK")
    trab_path = _run_mask_path(run_dir, base, "TRAB_MASK")
    for p in (cort_path, trab_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing in the run folder: {p}")
    if grid is None:
        report_path = os.path.join(run_dir, f"{base}_report.json")
        if os.path.isfile(report_path):
            grid = _grid_from_report(report_path)
        if grid is None:
            grid = grid_from_file(cort_path)
    g = as_grid(grid)
    cort, ic = read_mask(cort_path, g, select=None)
    trab, it = read_mask(trab_path, g, select=None)
    labels = labels_from_masks(cort, trab)
    out_dir = os.path.abspath(out_dir) if out_dir else run_dir
    os.makedirs(out_dir, exist_ok=True)
    seg_path = os.path.join(out_dir, base + SEG_NRRD_SUFFIX)
    lm_path = os.path.join(out_dir, base + LABELMAP_SUFFIX)
    write_seg_nrrd(seg_path, labels, g)
    write_labelmap_nifti(lm_path, labels, g)
    return dict(base=base, run_dir=run_dir, seg_nrrd=seg_path, labelmap=lm_path, grid=g.as_dict(),
                cortical_voxels=int(cort.sum()), trabecular_voxels=int(trab.sum()),
                sources=dict(cortical=dict(path=ic["path"], sha256=ic["sha256"]), trabecular=dict(path=it["path"], sha256=it["sha256"])))
