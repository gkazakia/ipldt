"""Shared argparse pieces and image I/O for the ipldt workflows."""
from __future__ import annotations

import argparse
import os

import numpy as np

from ..io import align_to, read_aim
from ..ormir import DTParams, write_volume


def add_dt_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("IPL dt_* parameters (Scanco Script 32 defaults)")
    g.add_argument("--ridge-epsilon", type=float, default=0.9, help="sphere-centre (ridge) containment tolerance (default 0.9)")
    g.add_argument("--assign-epsilon", type=float, default=0.5, help="sphere drawing tolerance |c - x| <= D/2 + eps (default 0.5)")
    g.add_argument("--peel-iter", type=int, default=-1, help="slice-wise 4-connected erosion of the contour before centre selection; -1 = automatic (0) (default -1)")
    g.add_argument("--dt-version", dest="dt_version", type=int, default=3, choices=(1, 2, 3), help="IPL's -version: the diameter rule (default 3)")
    g.add_argument("--suppress-boundary", type=int, default=2, help="accepted for symmetry; no effect on IPL's outputs (default 2)")
    g.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto", help="CuPy GPU when available (auto), or force gpu / cpu; results are identical")
    g.add_argument("--voxel-size", type=float, default=None, help="voxel size in mm for the statistics (default: AIM header element size / image spacing)")
    from .. import __version__
    parser.add_argument("--version", action="version", version=f"ipldt {__version__}")


def dt_params_from_args(args) -> DTParams:
    return DTParams(ridge_epsilon=args.ridge_epsilon, assign_epsilon=args.assign_epsilon, peel_iter=args.peel_iter,
                    version=args.dt_version, suppress_boundary=args.suppress_boundary)


def is_aim(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == ".aim"


def read_image(path: str) -> dict:
    """AIM (ipldt.io.read_aim: native grid, global position kept) or any SimpleITK-readable image.
    Returns dict(data (z, y, x), dim (x, y, z), pos (x, y, z), el_size_mm, header (AIM bytes or None))."""
    if is_aim(path):
        a = read_aim(path)
        return dict(data=a["data"], dim=tuple(a["dim"]), pos=tuple(a["pos"]), el_size_mm=tuple(a["el_size_mm"]), header=a["header"])
    import SimpleITK as sitk
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    sp = tuple(float(s) for s in img.GetSpacing())
    org = img.GetOrigin()
    pos = tuple(int(round(o / s)) for o, s in zip(org, sp))          # ipldt writes NIfTI with origin = pos * el_size
    return dict(data=arr, dim=tuple(int(n) for n in img.GetSize()), pos=pos, el_size_mm=sp, header=None)


def read_binary_on_grid(path: str, ref: dict | None = None) -> np.ndarray:
    """Read a mask and put it on ref's grid: AIMs are aligned by global position, other formats must
    already share ref's shape.  Returns bool (z, y, x)."""
    v = read_image(path)
    data = v["data"] > 0
    if ref is None:
        return data
    if v["header"] is not None and ref.get("header") is not None:
        return align_to(dict(data=data.astype(np.uint8), dim=v["dim"], pos=v["pos"]), ref["dim"], ref["pos"]) > 0
    if data.shape != ref["data"].shape:
        raise ValueError(f"{path}: shape {data.shape} does not match the object's {ref['data'].shape} (resample first, or use AIMs which are aligned by position)")
    return data


def write_image(path: str, data: np.ndarray, ref: dict, fmt: str | None = None) -> None:
    """Write on ref's grid; the format follows the extension (.AIM -> AIM using ref's header, else NIfTI/whatever SimpleITK infers)."""
    ext = os.path.splitext(path)[1].lower()
    fmt = fmt or ("aim" if ext == ".aim" else "nifti")
    if fmt == "aim":
        if ref.get("header") is None:
            raise ValueError("writing an AIM needs an AIM input on the same grid (for its header)")
        write_volume(path, data, "aim", ref["el_size_mm"], ref["pos"], ref["header"])
    elif ext in (".nii", ".gz"):
        write_volume(path, data, "nifti", ref["el_size_mm"], ref["pos"])
    else:
        import SimpleITK as sitk
        img = sitk.GetImageFromArray(np.ascontiguousarray(data))
        img.SetSpacing([float(s) for s in ref["el_size_mm"]])
        img.SetOrigin([float(p * s) for p, s in zip(ref["pos"], ref["el_size_mm"])])
        sitk.WriteImage(img, path)


def voxel_size_from(ref: dict, override: float | None) -> float:
    return float(override) if override is not None else float(ref["el_size_mm"][0])


def print_report(report: dict, keys=None) -> None:
    for k, v in report.items():
        if keys and k not in keys:
            continue
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
