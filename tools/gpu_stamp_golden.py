"""Golden arrays for the GPU sphere stamping (tests/test_gpu_stamp_schedule.py), frozen from a chosen checkout.

The 2026-10-07 launch schedule of ipldt.gpu.draw_spheres_gpu must leave every map bit-identical to the release
it replaced (750c94a, one launch of one thread per centre).  This script runs a checkout's GPU code on
randomised sphere sets and on the dt functions, and stores the inputs with the outputs:

    python tools/gpu_stamp_golden.py --root <checkout of 750c94a> --out tests/data/gpu_stamp_golden.json

`--root` is put first on sys.path, so its ipldt is the one imported (checked).  The cases come from
stamp_cases() and the phantoms from random_phantom() / sparse_rods() below; the file holds the inputs
themselves, so the test does not depend on regenerating them.  All data are synthetic.  Arrays are stored as
JSON ({dtype, shape, zlib-compressed bytes in base64}; boolean arrays bit-packed) since the repository keeps
no .npy / .npz files.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import zlib

import numpy as np

AE_OTHER = (0.0, 0.25, 0.45, 0.55, 1.0, 1.5, 0.3, 0.7, 0.1, 0.05)
KINDS = ("mixed", "faces", "dups", "tiny", "flat", "single", "dense_small")


def _shape(rng, lo, hi):
    return tuple(int(v) for v in rng.integers(lo, hi + 1, 3))


def _centres_on_faces(rng, shape, n):
    """n centres with 1 to 3 coordinates on a face (0 or N-1): faces, edges and corners."""
    c = np.stack([rng.integers(0, s, n) for s in shape], axis=1)
    for row in c:
        for ax in rng.choice(3, size=int(rng.integers(1, 4)), replace=False):
            row[ax] = 0 if rng.random() < 0.5 else shape[ax] - 1
    return c


def stamp_cases(n_cases, seed, max_side=72, big_d=260):
    """Randomised inputs of draw_spheres_gpu: a list of dicts (cz, cy, cx, diam int64, shape, assign_epsilon,
    kind).  Kinds, in rotation: 'mixed' (up to 400 centres, D mostly 0-30, some up to big_d), 'faces' (big
    spheres centred on faces, edges and corners, crossing every face), 'dups' (repeated centres, with equal
    and with different D), 'tiny' (D in 0..2), 'flat' (a volume one or two voxels thick), 'single' (one or
    two spheres, D up to big_d + 40), 'dense_small' (many small spheres in a small box).  Shapes are random,
    odd sizes included; half of the cases use assign_epsilon 0.5, the rest one of AE_OTHER."""
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n_cases):
        kind = KINDS[i % len(KINDS)]
        if kind == "mixed":
            shape = _shape(rng, 9, max_side)
            n = int(rng.integers(1, 401))
            c = np.stack([rng.integers(0, s, n) for s in shape], axis=1)
            d = np.where(rng.random(n) < 0.85, rng.integers(0, 31, n), rng.integers(31, big_d + 1, n))
        elif kind == "faces":
            shape = _shape(rng, 20, max_side + 18)
            n = int(rng.integers(1, 13))
            c = _centres_on_faces(rng, shape, n)
            d = rng.integers(100, big_d + 1, n)
        elif kind == "dups":
            shape = _shape(rng, 7, max_side // 2)
            m = int(rng.integers(1, 40))
            base = np.stack([rng.integers(0, s, m) for s in shape], axis=1)
            k = rng.integers(0, m, int(rng.integers(1, 3 * m + 1)))
            c = np.concatenate([base, base[k]])
            d = rng.integers(0, 60, c.shape[0])
            same = rng.random(k.size) < 0.5                    # half of the repeats with the same D
            d[m:][same] = d[k[same]]
            p = rng.permutation(c.shape[0])
            c, d = c[p], d[p]
        elif kind == "tiny":
            shape = _shape(rng, 3, max_side // 2)
            n = int(rng.integers(1, 300))
            c = np.stack([rng.integers(0, s, n) for s in shape], axis=1)
            d = rng.integers(0, 3, n)
        elif kind == "flat":
            shape = list(_shape(rng, 10, max_side))
            shape[int(rng.integers(0, 3))] = int(rng.integers(1, 3))
            shape = tuple(shape)
            n = int(rng.integers(1, 60))
            c = np.stack([rng.integers(0, s, n) for s in shape], axis=1)
            d = np.where(rng.random(n) < 0.7, rng.integers(0, 40, n), rng.integers(40, big_d + 1, n))
        elif kind == "single":
            shape = _shape(rng, 11, max_side + 8)
            n = int(rng.integers(1, 3))
            c = np.stack([rng.integers(0, s, n) for s in shape], axis=1)
            d = rng.integers(150, big_d + 41, n)
        else:                                                 # dense_small
            shape = _shape(rng, 5, 24)
            n = int(rng.integers(50, 2000))
            c = np.stack([rng.integers(0, s, n) for s in shape], axis=1)
            d = rng.integers(0, 12, n)
        ae = 0.5 if rng.random() < 0.5 else float(rng.choice(AE_OTHER))
        cases.append(dict(cz=c[:, 0].astype(np.int64), cy=c[:, 1].astype(np.int64), cx=c[:, 2].astype(np.int64),
                          diam=np.asarray(d, np.int64), shape=tuple(int(s) for s in shape), assign_epsilon=ae,
                          kind=kind))
    return cases


def random_phantom(shape=(20, 40, 40), seed=0, sigma=2.0, fill=0.45):
    """tests/conftest.py's dense phantom: thresholded smooth noise (plates and rods, ~45 % object)."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.standard_normal(shape), sigma)
    return f > np.quantile(f, 1.0 - fill)


def sparse_rods(shape=(40, 190, 190), n_rods=4, radius=1.3, seed=7):
    """A sparse 'trabecular' volume: n_rods straight rods (random lines through the box, radius in voxels),
    under 1 % object, so the marrow spheres of dt_spacing / dt_number reach D > 150."""
    rng = np.random.default_rng(seed)
    grid = np.stack(np.indices(shape), axis=-1).astype(np.float64)
    seg = np.zeros(shape, bool)
    for _ in range(n_rods):
        p = rng.uniform(0, 1, 3) * (np.array(shape) - 1)
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        v = grid - p
        t = v @ u
        seg |= (v * v).sum(-1) - t * t <= radius ** 2
    return seg


def phantoms():
    """name -> (object, gobj raster or None): the dense phantom with and without a gobj, the sparse rods."""
    from ipldt.contour import render_volume
    dense = random_phantom()
    shape = dense.shape
    yy, xx = np.mgrid[:shape[1], :shape[2]]
    disc = (yy - (shape[1] - 1) / 2) ** 2 + (xx - (shape[2] - 1) / 2) ** 2 <= 14.0 ** 2
    m = np.zeros(shape, bool)
    m[2:18] = disc
    return {"dense": (dense, None), "dense_gobj": (dense, render_volume(m)), "sparse": (sparse_rods(), None)}


def encode(a):
    """An array as JSON: dtype, shape and zlib-compressed bytes in base64 (bool arrays bit-packed)."""
    a = np.ascontiguousarray(a)
    bits = a.dtype == bool
    raw = np.packbits(a.ravel()) if bits else a
    return {"dtype": "bool" if bits else raw.dtype.str, "shape": list(a.shape),
            "zlib_b64": base64.b64encode(zlib.compress(raw.tobytes(), 9)).decode("ascii")}


def decode(d):
    raw = zlib.decompress(base64.b64decode(d["zlib_b64"]))
    if d["dtype"] == "bool":
        n = int(np.prod(d["shape"]))
        return np.unpackbits(np.frombuffer(raw, np.uint8), count=n).reshape(d["shape"]).astype(bool)
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def _git_head(root):
    try:
        return subprocess.run(["git", "--no-optional-locks", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="checkout whose ipldt is frozen (e.g. 750c94a)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-cases", type=int, default=42)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--max-side", type=int, default=48)
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    sys.path.insert(0, root)
    import ipldt
    from ipldt import dt_number, dt_spacing, dt_thickness
    from ipldt.core import draw_spheres
    from ipldt.gpu import draw_spheres_gpu
    if not os.path.abspath(ipldt.__file__).startswith(root):
        raise SystemExit(f"imported {ipldt.__file__}, not the checkout {root}")
    out = {"meta": {"root_commit": _git_head(root), "ipldt_version": ipldt.__version__, "seed": a.seed,
                    "n_cases": a.n_cases, "max_side": a.max_side,
                    "made_by": "tools/gpu_stamp_golden.py: draw_spheres_gpu and dt_* (backend='gpu') of root_commit"},
           "stamp": [], "phantoms": {}}
    for c in stamp_cases(a.n_cases, a.seed, a.max_side, big_d=260):
        m = draw_spheres_gpu(c["cz"], c["cy"], c["cx"], c["diam"], c["shape"], c["assign_epsilon"])
        cpu = draw_spheres(c["cz"], c["cy"], c["cx"], c["diam"], c["shape"], c["assign_epsilon"])
        out["stamp"].append(dict(kind=c["kind"], shape=list(c["shape"]), assign_epsilon=c["assign_epsilon"],
                                 **{k: encode(c[k].astype(np.int32)) for k in ("cz", "cy", "cx", "diam")},
                                 map=encode(m), gpu_vs_cpu_voxels=int((m != cpu).sum())))
    funcs = {"thickness": dt_thickness, "spacing": dt_spacing, "number": dt_number}
    for name, (obj, g) in phantoms().items():
        ph = out["phantoms"][name] = {"obj": encode(obj), "gobj": None if g is None else encode(g), "dt": {}}
        for fn, f in funcs.items():
            r = f(obj, gobj=None if g is None else {"rendered": g}, backend="gpu")
            ph["dt"][fn] = {"map": encode(r.map), "centres": encode(r.centres), "report": r.report}
            print(f"{name:11s} {fn:9s} max D {int(r.map.max()):4d}  centres {int(r.centres.sum()):7d}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=0, sort_keys=True)
        fh.write("\n")
    bad = [i for i, c in enumerate(out["stamp"]) if c["gpu_vs_cpu_voxels"]]
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB) from {out['meta']['root_commit']}; "
          f"stamp cases where GPU != CPU (repeated centres): {bad}")


if __name__ == "__main__":
    main()
