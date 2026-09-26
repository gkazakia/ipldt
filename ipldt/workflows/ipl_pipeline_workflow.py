"""ipldt-pipeline -- Scanco AIM -> IPL-style HR-pQCT morphometry (ORMIR autocontour, IPL compartments,
Laplace-Hamming SEG, dt_thickness / dt_spacing / dt_number as IPL computes them, the cortical pore cascade and
Ct.Po; NIfTI or AIM output)."""
import argparse
import os

from ipldt.ormir import Logger, run_pipeline
from ipldt.workflows._cli import add_dt_arguments, dt_params_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipldt-pipeline",
        description="Run the IPL-style pipeline on a Scanco AIM: autocontour -> cortical/trabecular compartments -> "
                    "Laplace-Hamming SEG -> Tb.Th, Tb.Sp, Tb.N, Ct.Th (dt_* maps as IPL computes them) -> cortical pore map "
                    "and Ct.Po + JSON/CSV report.",
    )
    parser.add_argument("input_aim", type=str, help="Path to the greyscale AIM file")
    parser.add_argument("output_dir", type=str, help="Output directory (created; use --subfolder to append <basename>/)")
    parser.add_argument("--subfolder", action="store_true", help="write into <output_dir>/<basename>/ as the upstream batch runner does")
    parser.add_argument("--map-format", choices=("nifti", "aim"), default="nifti", help="map / mask file format (default nifti; aim uses the input AIM header)")
    parser.add_argument("--map-units", choices=("voxels", "mm"), default="voxels", help="maps as IPL's integer diameters in voxels (default) or float mm (NIfTI only)")
    parser.add_argument("--no-bmd", action="store_true", help="skip the ORMIR bmd_masked step")
    parser.add_argument("--no-masks", action="store_true", help="do not write the masks / contours / SEG, only the maps and the report")
    parser.add_argument("--lh-voxel-size", type=float, default=None, help="override the Laplace-Hamming element size (isotropic); default: the AIM header's (x, y, z) element sizes, which is IPL's rule")
    parser.add_argument("--site", choices=("tibia", "radius"), default="tibia",
                        help="Script 32 STEP 1 parameters for the compartment masks: tibia (corner_min 200000, close2 50; default) or radius (Script 33: 800, 30)")
    add_dt_arguments(parser)
    return parser


def run(input_aim, output_dir, subfolder=False, map_format="nifti", map_units="voxels", no_bmd=False, no_masks=False,
        lh_voxel_size=None, backend="auto", voxel_size=None, params=None, site="tibia") -> int:
    if subfolder:
        output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(input_aim))[0])
    report = run_pipeline(input_aim, output_dir, map_format=map_format, backend=backend, voxel_size_mm=voxel_size, params=params,
                          map_units=map_units, compute_bmd=not no_bmd, save_masks=not no_masks, lh_voxel_size_mm=lh_voxel_size,
                          site=site, log=Logger())
    m = report["morphometry"]
    print("\nIPL-style morphometry:")
    for k in ("BV_TV", "Tb_Th_mm", "Tb_Th_sd_mm", "Tb_Sp_mm", "Tb_Sp_sd_mm", "Tb_1N_mm", "Tb_N_per_mm", "Ct_Th_mm", "Ct_Th_sd_mm"):
        if k in m:
            print(f"{k}: {m[k]:.6f}")
    for k, v in report.get("bmd", {}).items():
        print(f"{k}: {v:.6f}")
    print(f"Report written to {os.path.join(output_dir, report['sample'] + '_report.json')} (and .csv)")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(input_aim=args.input_aim, output_dir=args.output_dir, subfolder=args.subfolder, map_format=args.map_format,
                   map_units=args.map_units, no_bmd=args.no_bmd, no_masks=args.no_masks, lh_voxel_size=args.lh_voxel_size,
                   backend=args.backend, voxel_size=args.voxel_size, params=dt_params_from_args(args), site=args.site)
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
