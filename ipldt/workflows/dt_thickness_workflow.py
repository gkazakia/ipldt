"""ipldt-dt-thickness -- Scanco IPL /dt_thickness on a binary image, reimplemented to agree with IPL voxel for voxel."""
import argparse
import json

import numpy as np

from ipldt import dt_thickness
from ipldt.ormir import DTParams
from ipldt.workflows._cli import add_dt_arguments, dt_params_from_args, print_report, read_binary_on_grid, read_image, voxel_size_from, write_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipldt-dt-thickness",
        description="Compute IPL's /dt_thickness map (integer sphere diameters in voxels) and statistics.",
    )
    parser.add_argument("input_image", type=str, help="Binary object image (AIM, NIfTI, MHA ...): e.g. IPL's SEG for Tb.Th, the cortical compartment mask for Ct.Th")
    parser.add_argument("output_image", type=str, help="Output map (.AIM needs an AIM input; otherwise NIfTI/MHA by extension)")
    parser.add_argument("--gobj", type=str, default=None, help="Contour mask: a raw mask raster (rendered with IPL's contour rules) unless --gobj-rendered")
    parser.add_argument("--gobj-rendered", action="store_true", help="the --gobj image is already a rendered contour")
    parser.add_argument("--report", type=str, default=None, help="write the statistics to this JSON file")
    parser.add_argument("--map-units", choices=("voxels", "mm"), default="voxels", help="write the map in voxels (int16, default) or mm (float32)")
    add_dt_arguments(parser)
    return parser


def _run_dt(fn, name, args) -> int:
    ref = read_image(args.input_image)
    obj = ref["data"] > 0
    gobj = None
    if args.gobj:
        G = read_binary_on_grid(args.gobj, ref)
        gobj = {"rendered": G} if args.gobj_rendered else G
    vs = voxel_size_from(ref, args.voxel_size)
    notes = []
    dtp = dt_params_from_args(args, notes=notes)
    for note in notes:
        print(f"note: {note}")
    res = fn(obj, gobj=gobj, voxel_size_mm=vs, backend=args.backend, **dtp.kwargs())
    print_report(res.report)
    data = res.map.astype(np.float32) * np.float32(vs) if args.map_units == "mm" else res.map.astype(np.int16)
    print(f"Writing {name} map to {args.output_image}")
    write_image(args.output_image, data, ref)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(dict(res.report, input=args.input_image, gobj=args.gobj, centres=int(res.centres.sum()), voxel_size_mm=vs,
                           **dtp.kwargs(), non_default=[f"dt.{k}" for k, v in dtp.kwargs().items() if DTParams().kwargs()[k] != v]),
                      fh, indent=1)
    return 0


def check_parameters(parser, args):
    """--params / --set / the dt flags checked before anything runs (a usage error, exit 2)."""
    from ipldt.params import ParameterError
    try:
        dt_params_from_args(args)
    except ParameterError as exc:
        parser.error(str(exc))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    check_parameters(parser, args)
    try:
        return _run_dt(dt_thickness, "dt_thickness", args)
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
