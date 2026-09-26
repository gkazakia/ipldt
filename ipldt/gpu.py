"""ipldt.gpu -- optional CuPy acceleration with a transparent CPU fallback.

What runs on the GPU (bit-identical to the CPU path, verified on the validation data):
  * the 26-neighbour containment ridge test          (ipldt.core.ridge; float64 surface distances from
                                                      the exact integer 4 s^2, IEEE sqrt on both devices)
  * the diameter evaluation at the centres            (vectorised, either device)
  * the sphere drawing, as a direct atomicMax stamp of every sphere (ipldt.core.draw_spheres)
What stays on the CPU: the slice-interleaved vector distance transform (a sequential sweep whose
tie choices ARE the result; numba, a few seconds) and the contour rendering (per-slice tracing).

backend = "auto" (GPU if CuPy sees a device, else CPU), "gpu" (error if unavailable), "cpu".
"""
from __future__ import annotations

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


def draw_spheres_gpu(cz, cy, cx, diam, shape, assign_epsilon=0.5):
    """GPU sphere drawing; returns an int16 numpy map identical to the CPU EDT-based drawing."""
    import cupy as cp
    n = int(np.asarray(diam).size)
    out = cp.zeros(int(np.prod(shape)), dtype=cp.int32)
    if n:
        k = _stamp_kernel()
        czd, cyd, cxd = (cp.asarray(np.asarray(a, dtype=np.int32)) for a in (cz, cy, cx))
        dd = cp.asarray(np.asarray(diam, dtype=np.int32))
        threads = 128
        blocks = (n + threads - 1) // threads
        k((blocks,), (threads,), (czd, cyd, cxd, dd, np.int32(n), np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2]),
                                  np.float32(2.0 * assign_epsilon), out))
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
