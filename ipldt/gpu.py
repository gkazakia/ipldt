"""ipldt.gpu -- optional CuPy acceleration with a transparent CPU fallback.

What runs on the GPU (bit-identical to the CPU path, verified on the validation data):
  * the 26-neighbour containment ridge test          (ipldt.core.ridge; float64 surface distances from
                                                      the exact integer 4 s^2, IEEE sqrt on both devices)
  * the diameter evaluation at the centres            (vectorised, either device)
  * the sphere drawing, as a direct atomicMax stamp of every sphere (ipldt.core.draw_spheres), in launches
    of bounded work (stamp_plan)
What stays on the CPU: the slice-interleaved vector distance transform (a sequential sweep whose
tie choices ARE the result; numba, a few seconds) and the contour rendering (per-slice tracing).

backend = "auto" (GPU if CuPy sees a device, else CPU), "gpu" (error if unavailable), "cpu".
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

_CUPY = None


def cupy_available():
    global _CUPY
    if _CUPY is None:
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            _CUPY = cp
        except Exception:
            _CUPY = False
    return _CUPY is not False


def resolve_backend(backend="auto"):
    if backend == "cpu":
        return "cpu"
    if backend == "gpu":
        if not cupy_available():
            raise RuntimeError("backend='gpu' requested but CuPy / a CUDA device is not available")
        return "gpu"
    if backend == "auto":
        return "gpu" if cupy_available() else "cpu"
    raise ValueError(f"backend must be 'auto', 'gpu' or 'cpu' (got {backend!r})")


def _stamp_kernel():
    """Sphere stamping: every centre writes its diameter into its ball with an atomic max.
    Diameters are stored as int32 on the device (atomicMax has no 16-bit variant)."""
    import cupy as cp
    src = r"""
    extern "C" __global__
    void stamp_spheres(const int* __restrict__ cz, const int* __restrict__ cy, const int* __restrict__ cx,
                       const int* __restrict__ diam, const int n, const int nz, const int ny, const int nx,
                       const float two_ae, int* __restrict__ out)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int D = diam[i];
        const int z0 = cz[i], y0 = cy[i], x0 = cx[i];
        const float lim = (float)D + two_ae;
        const float lim2 = lim * lim;
        const int R = (int)floorf(lim * 0.5f) + 1;
        for (int dz = -R; dz <= R; ++dz) {
            const int z = z0 + dz; if (z < 0 || z >= nz) continue;
            for (int dy = -R; dy <= R; ++dy) {
                const int y = y0 + dy; if (y < 0 || y >= ny) continue;
                const long long row = ((long long)z * ny + y) * nx;
                for (int dx = -R; dx <= R; ++dx) {
                    const int x = x0 + dx; if (x < 0 || x >= nx) continue;
                    const int d2 = dz * dz + dy * dy + dx * dx;
                    if (4.0f * (float)d2 <= lim2 + 1e-6f) atomicMax(out + row + x, D);
                }
            }
        }
    }
    """
    return cp.RawKernel(src, "stamp_spheres")


def _stamp_columns_kernel():
    """Large spheres: one thread per (centre, dz, dx) column of the centre's (2R+1)^3 box, looping over dy.
    The (dz, dy, dx) positions, bounds checks, containment test and atomicMax are those of stamp_spheres,
    statement for statement; only the split of a box over threads differs.  col0[k] is the first column of
    centre k (ncen + 1 entries, the last = the total); a centre whose column count is not (2R+1)^2 for the
    kernel's own R raises the error flag instead of stamping, so a plan can never silently drop or add work."""
    import cupy as cp
    src = r"""
    extern "C" __global__
    void stamp_columns(const int* __restrict__ cz, const int* __restrict__ cy, const int* __restrict__ cx,
                       const int* __restrict__ diam, const long long* __restrict__ col0, const int ncen,
                       const long long g0, const long long ncols, const int nz, const int ny, const int nx,
                       const float two_ae, int* __restrict__ out, int* __restrict__ err)
    {
        const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (t >= ncols) return;
        const long long g = g0 + t;
        int lo = 0, hi = ncen;                                   /* col0[lo] <= g < col0[hi] */
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (col0[mid] <= g) lo = mid; else hi = mid;
        }
        const int i = lo;
        const int D = diam[i];
        const int z0 = cz[i], y0 = cy[i], x0 = cx[i];
        const float lim = (float)D + two_ae;
        const float lim2 = lim * lim;
        const int R = (int)floorf(lim * 0.5f) + 1;
        const long long W = 2LL * R + 1;
        if (R < 0 || col0[i + 1] - col0[i] != W * W) { atomicExch(err, 1); return; }
        const int dz = (int)((g - col0[i]) / W) - R;
        const int dx = (int)((g - col0[i]) % W) - R;
        const int z = z0 + dz; if (z < 0 || z >= nz) return;
        const int x = x0 + dx; if (x < 0 || x >= nx) return;
        for (int dy = -R; dy <= R; ++dy) {
            const int y = y0 + dy; if (y < 0 || y >= ny) continue;
            const long long row = ((long long)z * ny + y) * nx;
            const int d2 = dz * dz + dy * dy + dx * dx;
            if (4.0f * (float)d2 <= lim2 + 1e-6f) atomicMax(out + row + x, D);
        }
    }
    """
    return cp.RawKernel(src, "stamp_columns")


def _stamp_radius_kernel():
    """The box radius R of every centre, computed by the statements of stamp_spheres, for the launch plan."""
    import cupy as cp
    src = r"""
    extern "C" __global__
    void stamp_radius(const int* __restrict__ diam, const int n, const float two_ae, int* __restrict__ rad)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const int D = diam[i];
        const float lim = (float)D + two_ae;
        const int R = (int)floorf(lim * 0.5f) + 1;
        rad[i] = R;
    }
    """
    return cp.RawKernel(src, "stamp_radius")


# The launch schedule of the sphere stamping.  Until 2026-10-07 one launch ran one thread per centre over the
# centre's whole (2R+1)^3 box: on sparse segmentations a D = 179 sphere was 5.9e6 serial loop steps in one
# thread and a block 2.6e10 steps in one launch, and several processes doing this at once drew Xid 13 (FECS)
# faults from the driver.  Now every launch has at most STAMP_LAUNCH_BUDGET box positions, every thread at
# most max(STAMP_SMALL_BOX^3, 2R+1) of them, and the host waits for each launch to finish before the next.
# Scheduling only: every (centre, box position) test of stamp_spheres runs exactly once, with the same
# statements, and atomicMax on int32 is exact and order-independent, so the map is the same bit for bit.
STAMP_SMALL_BOX = 27            # box width 2R+1 up to which a centre is one thread (<= 27^3 = 19,683 steps)
STAMP_LAUNCH_BUDGET = 1 << 27   # box positions per launch

StampPlan = namedtuple("StampPlan", "order n_small col0 launches")


def stamp_plan(R, budget=None, small_box=None):
    """The launch schedule of the sphere stamping from the centres' box radii R (as stamp_radius computes
    them); a pure function (tests/test_gpu_stamp_schedule.py).

    The centres are ordered by box width W = 2R+1 (stable sort: `order`).  The first `n_small` of them
    (W <= small_box) are stamped by stamp_spheres, one thread per centre; each of the others owns W^2
    consecutive columns, col0[k] .. col0[k+1] - 1 (k counted from n_small), of W box positions each, stamped
    by stamp_columns.  `launches` lists ('point', a, b, steps) for the ordered centres a .. b-1 and
    ('columns', g0, g1, steps) for the columns g0 .. g1-1; together they cover every centre and every column
    exactly once, in order, and each launch has steps (box positions, W^3 per point centre) <= budget."""
    budget = STAMP_LAUNCH_BUDGET if budget is None else int(budget)
    small_box = STAMP_SMALL_BOX if small_box is None else int(small_box)
    if small_box < 1 or budget < small_box ** 3:
        raise ValueError(f"need small_box >= 1 and budget >= small_box**3 (got {small_box}, {budget})")
    R = np.asarray(R, dtype=np.int64).ravel()
    W = 2 * R + 1
    if float((np.maximum(W, 1).astype(np.float64) ** 3).sum()) > 2.0 ** 62:
        raise ValueError("sphere boxes too large to stamp: more than 2^62 box positions")
    small_range = W.size and -2 ** 15 <= W.min() and W.max() < 2 ** 15
    order = np.argsort(W.astype(np.int16) if small_range else W, kind="stable")   # int16: radix sort, same order
    Ws = W[order]
    n_small = int(np.searchsorted(Ws, small_box, side="right"))
    launches = []
    cum = np.concatenate(([0], np.cumsum(np.maximum(Ws[:n_small], 1) ** 3)))     # a box of W <= 0 is empty
    a = 0
    while a < n_small:
        b = max(int(np.searchsorted(cum, cum[a] + budget, side="right")) - 1, a + 1)
        launches.append(("point", a, b, int(cum[b] - cum[a])))
        a = b
    WL = Ws[n_small:]
    col0 = np.concatenate(([0], np.cumsum(WL * WL)))
    S = np.concatenate(([0], np.cumsum(WL ** 3)))                               # box positions before centre k

    def steps_before(g):
        k = int(np.searchsorted(col0, g, side="right")) - 1
        return int(S[k]) if k == WL.size else int(S[k] + (g - col0[k]) * WL[k])

    total = int(col0[-1])
    g = 0
    while g < total:
        s0 = steps_before(g)
        k = int(np.searchsorted(S, s0 + budget, side="right")) - 1
        g1 = total if k >= WL.size else int(col0[k] + (s0 + budget - S[k]) // WL[k])
        g1 = min(max(g1, g + 1), total)
        launches.append(("columns", g, g1, steps_before(g1) - s0))
        g = g1
    return StampPlan(order, n_small, col0, launches)


def draw_spheres_gpu(cz, cy, cx, diam, shape, assign_epsilon=0.5):
    """GPU sphere drawing; returns an int16 numpy map identical to the CPU EDT-based drawing."""
    return _draw_spheres_gpu(cz, cy, cx, diam, shape, assign_epsilon)


def _draw_spheres_gpu(cz, cy, cx, diam, shape, assign_epsilon=0.5, budget=None, small_box=None, timings=None):
    """draw_spheres_gpu with the schedule's knobs (tests, benchmarks); `timings`, a list, receives
    (kind, start, stop, steps, milliseconds) for every launch."""
    import cupy as cp
    n = int(np.asarray(diam).size)
    out = cp.zeros(int(np.prod(shape)), dtype=cp.int32)
    if n:
        nz, ny, nx = np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2])
        two_ae = np.float32(2.0 * assign_epsilon)
        czd, cyd, cxd, dd = (cp.asarray(np.asarray(a, dtype=np.int32)) for a in (cz, cy, cx, diam))
        rad = cp.empty(n, dtype=cp.int32)
        _stamp_radius_kernel()(((n + 127) // 128,), (128,), (dd, np.int32(n), two_ae, rad))
        plan = stamp_plan(cp.asnumpy(rad), budget, small_box)
        o = cp.asarray(plan.order)
        czs, cys, cxs, ds = (a[o] for a in (czd, cyd, cxd, dd))
        del czd, cyd, cxd, dd, rad, o
        ns = plan.n_small
        col0 = cp.asarray(plan.col0)
        err = cp.zeros(1, dtype=cp.int32)
        point, columns = _stamp_kernel(), _stamp_columns_kernel()
        stream = cp.cuda.get_current_stream()
        for kind, a, b, steps in plan.launches:
            if timings is not None:
                t0, t1 = cp.cuda.Event(), cp.cuda.Event()
                t0.record(stream)
            if kind == "point":
                point(((b - a + 127) // 128,), (128,), (czs[a:b], cys[a:b], cxs[a:b], ds[a:b], np.int32(b - a),
                                                        nz, ny, nx, two_ae, out))
            else:
                columns(((b - a + 255) // 256,), (256,), (czs[ns:], cys[ns:], cxs[ns:], ds[ns:], col0, np.int32(n - ns),
                                                          np.int64(a), np.int64(b - a), nz, ny, nx, two_ae, out, err))
            if timings is not None:
                t1.record(stream)
            stream.synchronize()
            if timings is not None:
                timings.append((kind, a, b, steps, cp.cuda.get_elapsed_time(t0, t1)))
        if int(err.get()[0]):
            raise RuntimeError("ipldt.gpu: the stamping plan disagrees with the kernel's sphere radius; "
                               "no map returned")
    return cp.asnumpy(out).astype(np.int16).reshape(shape)


def ridge_gpu(obj, V, ridge_epsilon=0.9, tol=1e-9):
    """26-neighbour containment test on the GPU: the same float64 arithmetic as ipldt.core.ridge
    (surface distances from the exact integer 4 s^2 through one IEEE sqrt, the same operation
    order, the same 1e-9 tolerance), so the centre sets are bit-identical.  Precision rule and
    the radius / tibia evidence (2026-09-14; float32 kept a centre of exact ridge value 0.8999974 on three
    diaphyseal tibiae, 2,601 / 8 / 9 Ct.Th voxels off IPL): ipldt.core.surface_distance / ridge."""
    import cupy as cp
    import itertools
    from .core import surface_distance
    objd = cp.asarray(obj)
    s = surface_distance(V, xp=cp)
    s[~objd] = 0
    sp = cp.pad(s, 1)
    del s
    op = cp.pad(objd, 1, constant_values=False)
    keep = op & (sp > 0)
    thr = np.float64(ridge_epsilon + tol)
    for dz, dy, dx in (o for o in itertools.product((-1, 0, 1), repeat=3) if any(o)):
        sep = np.float64(np.sqrt(dz * dz + dy * dy + dx * dx))
        keep &= ~(cp.roll(op, (dz, dy, dx), axis=(0, 1, 2))
                  & ((sep + sp - cp.roll(sp, (dz, dy, dx), axis=(0, 1, 2))) <= thr))
    return cp.asnumpy(keep[1:-1, 1:-1, 1:-1])
