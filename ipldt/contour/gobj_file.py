"""pm_gobj -- reader for Scanco GOBJ files written by IPL /togobj_from_aim (clean-room: format read off the
files themselves).  Layout found:
  512-byte file header ('CTDATA-HEADER_V1' + int32s + text);
  per slice: 14-int32 slice header [1, X, Y, z, x0, y0, 0, 3, 20, 0, 0, 16512, 16512, n_contours];
  per contour: 17-int32 contour header [3, X, Y, z, W, H, 0, 3, 32, vaxf, vaxf, 16512, 16512, ox, oy, n_el, n_by]
               with (X, Y) the bounding-box centre, (W, H) its size and (ox, oy) the offset of the start pixel:
               start = (X + ox, Y + oy) in absolute scanner pixels; Freeman variant 8 (see path()) walks the chain
               followed by n_by bytes of LSB-first packed 3-bit Freeman chain codes: the first n_el - 1 steps
               of the closed contour (n_by = floor((3 n_el + 5) / 8); the bits after them are padding.  Every
               padding bit inside the last partly used byte is zero (10,436/10,436 contours of the 21-subject
               cohort); when n_el == 1 (mod 8) the formula leaves one whole spare byte that IPL does not
               initialise (non-zero in 1,311 of 1,470 such contours, 11 % of all 11,906), so that byte is
               unspecified and not reproducible).
  zero slice headers pad the end.
Stored code s means the image-frame step (DX[-s mod 8], DY[-s mod 8]) = STEP8[s] below (variant 8 of path());
chain_codes() encodes a vertex list the same way, so path(chain_codes(v)[:-1], v[0], 8) == v."""
import struct
import numpy as np

DX = np.array([1, 1, 0, -1, -1, -1, 0, 1])   # Freeman 0:+x 1:+x+y 2:+y 3:-x+y 4:-x 5:-x-y 6:-y 7:+x-y
DY = np.array([0, 1, 1, 1, 0, -1, -1, -1])
STEP8 = [(int(DX[(-s) % 8]), int(DY[(-s) % 8])) for s in range(8)]     # stored code -> (dx, dy), y down
CODE8 = {step: s for s, step in enumerate(STEP8)}


def decode(chain, n_el):
    bits = np.unpackbits(chain, bitorder="little")
    n = n_el - 1
    bits = bits[: 3 * n].reshape(n, 3)
    return (bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2]).astype(np.int64)


def read_gobj(path):
    raw = open(path, "rb").read()
    hdr = struct.unpack("<21i", raw[16:100])
    pos, slices = 512, []
    while pos + 56 <= len(raw):
        sh = struct.unpack("<14i", raw[pos:pos + 56]); pos += 56
        if sh[0] == 0 and sh[13] == 0:
            continue
        conts = []
        for k in range(sh[13]):
            ch = struct.unpack("<17i", raw[pos:pos + 68]); pos += 68
            n_el, n_by = ch[15], ch[16]
            chain = np.frombuffer(raw[pos:pos + n_by], np.uint8); pos += n_by
            conts.append({"h": ch, "start": (ch[1] + ch[13], ch[2] + ch[14]), "bbox_size": (ch[4], ch[5]),
                          "n_el": n_el, "codes": decode(chain, n_el)})
        slices.append({"h": sh, "z": sh[3], "contours": conts})
    return hdr, slices


def path(codes, start, variant=0):
    """vertex sequence (closed: first point repeated at the end is NOT included; the codes give n_el-1 steps
    ending 8-adjacent to the start).  variant = rot k (codes + k) + 8 * mirror (codes -> -codes)."""
    k, mirror = variant % 8, variant // 8
    c = np.mod((-codes if mirror else codes) + k, 8)
    x = start[0] + np.concatenate([[0], np.cumsum(DX[c])])
    y = start[1] + np.concatenate([[0], np.cumsum(DY[c])])
    return x, y




def chain_codes(vertices):
    """Freeman codes of a closed chain of 8-adjacent pixel centres as /togobj_from_aim stores them: one code
    per step, the last one closing the chain back to the start.  The file keeps the first n_el - 1 of the
    n_el = len(vertices) codes (the closing step is implicit; the trailing bits are padding), so a writer
    stores chain_codes(v)[:-1] with zero padding (byte-identical to IPL except the uninitialised spare byte
    of contours with n_el == 1 mod 8; a GOBJ writer is therefore not byte-exact) and
    path(chain_codes(v)[:-1], v[0], 8) gives v back."""
    v = np.asarray(vertices)
    if v.ndim != 2 or v.shape[1] != 2 or v.shape[0] == 0:
        raise ValueError(f"expected a non-empty chain of (x, y) vertices, got shape {v.shape}")
    n = len(vertices)
    codes = []
    for i in range(n):
        (x0, y0), (x1, y1) = vertices[i], vertices[(i + 1) % n]
        step = (x1 - x0, y1 - y0)
        if step not in CODE8:
            raise ValueError(f"vertices {i} and {(i + 1) % n} are not 8-adjacent: {vertices[i]} -> {vertices[(i + 1) % n]}")
        codes.append(CODE8[step])
    return np.array(codes, np.int64)
