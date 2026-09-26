"""sq_field -- independent implementation of the 'sir_quad' vector distance transform,
written directly from the definition below:

  object voxels start at BIG, background at the zero vector; outside the AIM is object.
  pass 1: slices z ascending; per slice, four sub-scans (y asc,x asc),(y asc,x desc),
          (y desc,x asc),(y desc,x desc).
  pass 2: slices z descending; mirrored sub-scan order.
  in a sub-scan (zd,yd,xd) each object voxel takes, in THIS order, v[p+o]+o for
      o = (-zd,0,0),(0,-yd,0),(0,0,-xd),(-zd,-yd,0),(-zd,0,-xd),(0,-yd,-xd),(-zd,-yd,-xd)
  replacing only when |candidate|^2 is STRICTLY smaller.  One round (2 z-passes).

Parameters expose the obvious neighbours (strictness, pass direction, sub-scan order,
offset order, number of rounds).

2026-09-12 (an independent re-check): the sub-scan order
above (SPEC_SUBS) misses ONE PFJ-0be66a voxel (98,105,664) and 3 on PFJ-8bcf88_R.  The order that is
exact — DM31/DM32/V1 0 mismatches on both subjects, ridge 0/0 vs RGV3/RGE00/RGE25, V3 0 wrong
with a refit table, map 99.9999% — is Danielsson's boustrophedon ('snake') over the four
quadrant directions:
    pass 1 (z asc):  (y asc,x asc), (y asc,x desc), (y desc,x desc), (y desc,x asc)
    pass 2 (z desc): (y desc,x desc), (y desc,x asc), (y asc,x asc), (y asc,x desc)   [= mirror]
It is the unique pass-1 order (of 24) with DM31 0; pass 2 is identified up to swapping its last
two sub-scans (both 0 wrong, 2 voxels differ).  SNAKE_SUBS is now the DEFAULT; SPEC_SUBS is
kept for the record.  `subs2` gives pass 2 explicitly (overrides `mirror`)."""
from __future__ import annotations

import numpy as np
from numba import njit

BIG = 1 << 20          # int32 sentinel for "unset"; any |component| >= BIG/2 means unset


@njit(cache=True)
def _pass(vz, vy, vx, obj, zd, sub_y, sub_x, offs, strict):
    nz, ny, nx = obj.shape
    lim = BIG // 2
    for zi in range(nz):
        z = zi if zd > 0 else nz - 1 - zi
        for q in range(sub_y.shape[0]):
            yd = sub_y[q]
            xd = sub_x[q]
            for yi in range(ny):
                y = yi if yd > 0 else ny - 1 - yi
                for xi in range(nx):
                    x = xi if xd > 0 else nx - 1 - xi
                    if not obj[z, y, x]:
                        continue
                    bz = vz[z, y, x]
                    by = vy[z, y, x]
                    bx = vx[z, y, x]
                    if bz >= lim:
                        best = np.int64(4) * lim * lim      # unset: anything finite wins
                    else:
                        best = np.int64(bz) * bz + np.int64(by) * by + np.int64(bx) * bx
                    for k in range(offs.shape[1]):
                        dz = offs[q, k, 0]
                        dy = offs[q, k, 1]
                        dx = offs[q, k, 2]
                        qz = z + dz
                        qy = y + dy
                        qx = x + dx
                        if qz < 0 or qz >= nz or qy < 0 or qy >= ny or qx < 0 or qx >= nx:
                            continue                      # outside the AIM is object: no contact
                        nzv = vz[qz, qy, qx]
                        if nzv >= lim:
                            continue                      # neighbour still unset
                        cz = np.int64(nzv) + dz
                        cy = np.int64(vy[qz, qy, qx]) + dy
                        cx = np.int64(vx[qz, qy, qx]) + dx
                        c = cz * cz + cy * cy + cx * cx
                        if c < best or ((not strict) and c == best):
                            best = c
                            bz = np.int32(cz)
                            by = np.int32(cy)
                            bx = np.int32(cx)
                    vz[z, y, x] = bz
                    vy[z, y, x] = by
                    vx[z, y, x] = bx


SPEC_SUBS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]          # (yd, xd) for the z-ascending pass (superseded)
SNAKE_SUBS = [(1, 1), (1, -1), (-1, -1), (-1, 1)]         # IPL's order (exact, 2026-09-12)
SNAKE_SUBS2 = [(-1, -1), (-1, 1), (1, 1), (1, -1)]        # its mirror = the z-descending pass
SNAKE_SUBS2_ALT = [(-1, -1), (-1, 1), (1, -1), (1, 1)]    # last two swapped: also 0 wrong on PFJ-0be66a (2 voxels differ)


def offsets_for(zd, yd, xd, order="spec"):
    o = [(-zd, 0, 0), (0, -yd, 0), (0, 0, -xd), (-zd, -yd, 0), (-zd, 0, -xd), (0, -yd, -xd), (-zd, -yd, -xd)]
    if order == "spec":
        return o
    if order == "reversed":
        return o[::-1]
    if order == "inslice_first":
        return [o[1], o[2], o[5], o[0], o[3], o[4], o[6]]
    if order == "raster":                                    # most negative first in the pass frame
        return sorted(o, key=lambda t: (zd * t[0], yd * t[1], xd * t[2]))
    if order == "diag_first":                                # S=3, S=2, S=1
        return [o[6], o[3], o[4], o[5], o[0], o[1], o[2]]
    raise ValueError(order)


def sir_quad(obj, strict=True, z_first=1, subs=SNAKE_SUBS, mirror=True, order="spec", rounds=1, subs2=None):
    """Run the structure; returns V (3, z, y, x) int16 with zeros on background."""
    obj = np.ascontiguousarray(obj, dtype=np.bool_)
    vz = np.where(obj, np.int32(BIG), np.int32(0)).astype(np.int32)
    vy = np.zeros_like(vz)
    vx = np.zeros_like(vz)
    for _ in range(rounds):
        for zd in (z_first, -z_first):
            sub = subs if zd == 1 else (subs2 if subs2 is not None else ([(-a, -b) for a, b in subs] if mirror else subs))
            sy = np.array([a for a, _ in sub], np.int64)
            sx = np.array([b for _, b in sub], np.int64)
            offs = np.array([offsets_for(zd, a, b, order) for a, b in sub], np.int64)
            _pass(vz, vy, vx, obj, np.int64(zd), sy, sx, offs, bool(strict))
    V = np.stack([vz, vy, vx])
    V[:, ~obj] = 0
    if (np.abs(V) >= BIG // 2).any():
        raise RuntimeError("unset object voxels remain")
    return V.astype(np.int16)
