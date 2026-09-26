"""ipldt-dt-spacing -- Scanco IPL /dt_spacing (Tb.Sp map) on a whole-bone segmentation, reimplemented to agree with IPL voxel for voxel."""
import argparse

from ipldt import dt_spacing
from ipldt.workflows.dt_thickness_workflow import _run_dt
from ipldt.workflows._cli import add_dt_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipldt-dt-spacing",
        description="Compute IPL's /dt_spacing map (trabecular separation: sphere diameters of the marrow in voxels) and statistics.",
    )
    parser.add_argument("input_image", type=str, help="Whole-bone segmentation (IPL's SEG: cortical + trabecular bone)")
    parser.add_argument("output_image", type=str, help="Output Tb.Sp map (.AIM needs an AIM input; otherwise NIfTI/MHA by extension)")
    parser.add_argument("--gobj", type=str, default=None, help="Trabecular contour mask (raw raster, rendered with IPL's rules unless --gobj-rendered)")
    parser.add_argument("--gobj-rendered", action="store_true", help="the --gobj image is already a rendered contour")
    parser.add_argument("--report", type=str, default=None, help="write the statistics to this JSON file")
    parser.add_argument("--map-units", choices=("voxels", "mm"), default="voxels")
    add_dt_arguments(parser)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return _run_dt(dt_spacing, "dt_spacing", args)
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
