"""ipldt.io -- Scanco AIM (v020) reading/writing, global-position alignment and NIfTI export.

AIM reading covers uncompressed char / short data and the two run-length variants IPL
writes (0x00080002 char RLE as (value, count) pairs, 0x00150001 bit RLE).  Header and
processing log are kept verbatim so a written AIM opens in IPL / ITK / 3D Slicer.
"""
from __future__ import annotations

import os
import struct

import numpy as np

TYPE_CHAR = 0x00010001
TYPE_SHORT = 0x00020002
TYPE_RLE_CHAR = 0x00080002
TYPE_RLE_BITS = 0x00150001


def _decode_rle_char(buf, nvox):
    values = buf[0::2]
    counts = buf[1::2].astype(np.int64)
    n = min(len(values), len(counts))
    if int(counts[:n].sum()) != nvox:
        raise ValueError("run-length stream does not match the voxel count")
    return np.repeat(values[:n], counts[:n])


def _decode_rle_bits(buf, nvox):
    """0x00150001: two alternating values, then run lengths; 255 = emit 254 and keep the value."""
    vals2 = buf[:2].astype(np.uint8)
    lengths = buf[2:].astype(np.int64)
    escape = lengths == 255
    emit = np.where(escape, 254, lengths)
    tog = np.where(escape, 2, 1)
    prefix = (np.cumsum(tog) - tog) & 1
    if int(emit.sum()) != nvox:
        raise ValueError("bit run-length stream does not match the voxel count")
    return np.repeat(vals2[prefix], emit)


def _read_aim_v030(path, raw):
    """AIMDATA_V030 (what IPL V5.42 writes for float images, e.g. /fft_laplace_hamming output): a 16-byte magic,
    a pre-header of five int64 (its own size 40, header size, processing-log size, data size, associated-data
    size), then a header of int64 fields [version, (type int32 at byte 12), pos x3, dim x3, off x3, supdim x3,
    suppos x3, subdim x3, testoff x3, el_size x3 in nanometres], the processing log and the data."""
    pre = struct.unpack("<5q", raw[16:56])
    psize, hsize, lsize, dsize = pre[0], pre[1], pre[2], pre[3]
    h0 = 16 + psize
    hdr = raw[h0:h0 + hsize]
    q = struct.unpack("<" + "q" * (hsize // 8), hdr[:hsize // 8 * 8])
    dtype_code = struct.unpack("<i", hdr[12:16])[0]
    pos = tuple(int(v) for v in q[2:5])
    dim = tuple(int(v) for v in q[5:8])
    el = tuple(float(v) / 1e6 for v in q[23:26])
    if not all(0.001 < e < 10.0 for e in el):
        el = (139852.0 / 2304.0 / 1000.0, 139852.0 / 2304.0 / 1000.0, 30592.0 / 504.0 / 1000.0)
    d0 = h0 + hsize + lsize
    body = raw[d0:d0 + dsize]
    nvox = dim[0] * dim[1] * dim[2]
    if dsize == 4 * nvox:
        arr = np.frombuffer(body[:4 * nvox], np.float32)
    elif dsize == 2 * nvox:
        arr = np.frombuffer(body[:2 * nvox], np.int16)
    elif dsize == nvox:
        arr = np.frombuffer(body[:nvox], np.uint8)
    else:
        raise ValueError(f"AIMDATA_V030: unsupported data type 0x{dtype_code:08x} (data size {dsize} for {nvox} voxels)")
    return dict(data=arr.reshape(dim[::-1]), dim=dim, pos=pos, el_size_mm=el, header=raw[:d0],
                proclog=raw[h0 + hsize:d0].decode("latin-1", "replace"), type_code=dtype_code, version="030")


def read_aim(path):
    """Return dict(data (z, y, x), dim (x, y, z), pos (x, y, z), el_size_mm (x, y, z), header bytes).
    Reads the v020 layout (all Script 32 outputs) and the AIMDATA_V030 layout IPL uses for float images."""
    raw = open(path, "rb").read()
    if raw[:12] == b"AIMDATA_V030":
        return _read_aim_v030(path, raw)
    pre = struct.unpack("<5i", raw[:20])
    hsize, lsize, dsize = pre[1], pre[2], pre[3]
    hdr = raw[20:20 + hsize]
    ints = struct.unpack("<" + "i" * (hsize // 4), hdr)
    dtype_code = ints[5]
    pos = tuple(ints[6:9])
    dim = tuple(ints[9:12])
    # element sizes: the header's three VAX floats (ints 27..29), decoded exactly as ITK's ScancoImageIO does
    el = _element_size(path, ints)
    body = raw[20 + hsize + lsize: 20 + hsize + lsize + dsize]
    nvox = dim[0] * dim[1] * dim[2]
    if dtype_code == TYPE_CHAR:
        arr = np.frombuffer(body[:nvox], np.uint8)
    elif dtype_code == TYPE_SHORT:
        arr = np.frombuffer(body[:2 * nvox], np.int16)
    elif dtype_code in (TYPE_RLE_CHAR, TYPE_RLE_BITS):
        comp = struct.unpack("<i", body[:4])[0]                 # compressed body = int32 size + stream
        stream = np.frombuffer(body[4:comp], np.uint8)
        arr = _decode_rle_char(stream, nvox) if dtype_code == TYPE_RLE_CHAR else _decode_rle_bits(stream, nvox)
    elif dsize == 4 * nvox:
        # float AIM (e.g. the output of /fft_laplace_hamming): the type code is not one of the four above,
        # the body is one IEEE single per voxel
        arr = np.frombuffer(body[:4 * nvox], np.float32)
    else:
        raise ValueError(f"unsupported AIM data type 0x{dtype_code:08x} (data size {dsize} for {nvox} voxels)")
    return dict(data=arr.reshape(dim[::-1]), dim=dim, pos=pos, el_size_mm=el, header=raw[:20 + hsize + lsize],
                proclog=raw[20 + hsize: 20 + hsize + lsize].decode("latin-1", "replace"), type_code=dtype_code)


def _vax_f(u):
    """VAX F-floating (the AIM v020 header's float format): word-swapped, biased exponent 128, hidden 0.5."""
    u &= 0xFFFFFFFF
    w1, w2 = u & 0xFFFF, u >> 16
    s, e, f = w1 >> 15, (w1 >> 7) & 0xFF, ((w1 & 0x7F) << 16) | w2
    return 0.0 if e == 0 else (-1.0) ** s * (0.5 + f / 2.0 ** 24) * 2.0 ** (e - 128)


def _element_size(path, ints):
    """(x, y, z) element sizes in mm decoded from the header's three VAX floats (ints 27..29) -- the same
    values ITK's ScancoImageIO reports as spacing, without needing ITK.  The Laplace-Hamming filter depends
    on these to the 6th digit, so no nominal value is substituted unless the fields are unusable."""
    try:
        el = tuple(_vax_f(v) for v in ints[27:30])
        if all(0.001 < e < 10.0 for e in el):
            return el
    except Exception:
        pass
    import warnings
    warnings.warn(f"{os.path.basename(path)}: element sizes unreadable from the AIM header; using the XtremeCT II "
                  "full-stack values (139852/2304, 139852/2304, 30592/504 um)")
    return (139852.0 / 2304.0 / 1000.0, 139852.0 / 2304.0 / 1000.0, 30592.0 / 504.0 / 1000.0)


def align_to(src, dim, pos, fill=0):
    """Paste src['data'] into a (dim, pos) frame by GLOBAL voxel position (never by shape)."""
    out = np.full(dim[::-1], fill, dtype=src["data"].dtype)
    off = tuple(src["pos"][i] - pos[i] for i in range(3))
    sx, sy, sz = src["dim"]
    ox, oy, oz = off
    tx0, tx1 = max(0, ox), min(dim[0], ox + sx)
    ty0, ty1 = max(0, oy), min(dim[1], oy + sy)
    tz0, tz1 = max(0, oz), min(dim[2], oz + sz)
    if tx1 <= tx0 or ty1 <= ty0 or tz1 <= tz0:
        return out
    out[tz0:tz1, ty0:ty1, tx0:tx1] = src["data"][tz0 - oz:tz1 - oz, ty0 - oy:ty1 - oy, tx0 - ox:tx1 - ox]
    return out


def write_aim(path, data, template_header, pad_to_512=True):
    """Write an uncompressed char AIM v020 using `template_header` (the header bytes of an AIM
    on the same grid, as returned by read_aim); the data-type field is set to char and the
    pre-header data size updated.  IPL reads the result after
    SET FILE/ATTRIBUTES=(RFM:FIX,LRL:512,MRS:512,RAT:NONE) on the VMS side."""
    hdr = bytearray(template_header)
    pre = list(struct.unpack("<5i", hdr[:20]))
    hsize = pre[1]
    ints = list(struct.unpack("<" + "i" * (hsize // 4), hdr[20:20 + hsize]))
    ints[5] = TYPE_CHAR
    hdr[20:20 + hsize] = struct.pack("<" + "i" * (hsize // 4), *ints)
    body = np.ascontiguousarray(data, dtype=np.uint8).tobytes()
    pre[3] = len(body)
    hdr[:20] = struct.pack("<5i", *pre)
    pad = b"\0" * ((-(len(hdr) + len(body))) % 512) if pad_to_512 else b""
    with open(path, "wb") as fh:
        fh.write(bytes(hdr))
        fh.write(body)
        fh.write(pad)


def to_sitk(data, el_size_mm=(0.0607, 0.0607, 0.0607), pos=(0, 0, 0)):
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.ascontiguousarray(data))
    img.SetSpacing([float(s) for s in el_size_mm])
    img.SetOrigin([float(p * s) for p, s in zip(pos, el_size_mm)])
    return img


def write_nifti(path, data, el_size_mm=(0.0607, 0.0607, 0.0607), pos=(0, 0, 0)):
    import SimpleITK as sitk
    sitk.WriteImage(to_sitk(data, el_size_mm, pos), path)


def write_map(path, data, fmt=None, template_header=None, el_size_mm=(0.0607,) * 3, pos=(0, 0, 0)):
    """fmt 'aim' (needs template_header) or 'nifti' / 'nii' (default from the extension)."""
    ext = os.path.splitext(path)[1].lower()
    fmt = fmt or ("aim" if ext == ".aim" else "nifti")
    if fmt == "aim":
        if template_header is None:
            raise ValueError("write_map(..., fmt='aim') needs template_header from read_aim")
        write_aim(path, data, template_header)
    else:
        write_nifti(path, data, el_size_mm, pos)
