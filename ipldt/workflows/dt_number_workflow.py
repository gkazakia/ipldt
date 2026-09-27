"""ipldt-dt-number -- Scanco IPL /dt_number (1/Tb.N map and Tb.N) on a whole-bone segmentation, reimplemented to agree with IPL voxel for voxel."""
import argparse

from ipldt import dt_number
from ipldt.workflows.dt_thickness_workflow import _run_dt, check_parameters
from ipldt.workflows._cli import add_dt_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipldt-dt-number",
        description="Compute IPL's /dt_number map (1/Tb.N: sphere diameters between mid-axes, in voxels); Tb.N = 1 / mean.",
    )
    parser.add_argument("input_image", type=str, help="Whole-bone segmentation (IPL's SEG: cortical + trabecular bone)")
    parser.add_argument("output_image", type=str, help="Output 1/Tb.N map (.AIM needs an AIM input; otherwise NIfTI/MHA by extension)")
    parser.add_argument("--gobj", type=str, default=None, help="Trabecular contour mask (raw raster, rendered with IPL's rules unless --gobj-rendered)")
    parser.add_argument("--gobj-rendered", action="store_true", help="the --gobj image is already a rendered contour")
    parser.add_argument("--report", type=str, default=None, help="write the statistics (incl. Tb_N_per_mm) to this JSON file")
    parser.add_argument("--map-units", choices=("voxels", "mm"), default="voxels")
    add_dt_arguments(parser)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    check_parameters(parser, args)
    try:
        return _run_dt(dt_number, "dt_number", args)
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
