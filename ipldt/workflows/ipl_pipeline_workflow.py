"""ipldt-pipeline -- Scanco AIM -> IPL-style HR-pQCT morphometry (ORMIR autocontour, IPL compartments,
Laplace-Hamming SEG, dt_thickness / dt_spacing / dt_number as IPL computes them, the cortical pore cascade and
Ct.Po; NIfTI or AIM output).

Every tunable value of every stage is a parameter STAGE.NAME of ipldt.params, and the defaults are the validated IPL
configuration apart from two deliberate departures that every report names (Tb.Th on the whole SEG; the SEG masked
with the periosteal raster).  --params FILE (JSON) and repeatable --set STAGE.NAME=VALUE change them; the dedicated flags
(--ridge-epsilon ..., --lh-voxel-size, --voxel-size) are shortcuts for their parameters.  Precedence: the --site
preset < --params < the dedicated flags < --set.  A complete set given with --params is its preset plus its changes:
without --site its preset is used, with --site its changes apply to that site's preset.  --print-params prints the
complete effective set (JSON, or a table with --describe) and exits."""
import argparse
import json
import os
import sys

from ipldt.ormir import Logger, run_pipeline
from ipldt.params import ParameterError, parameters_from_args, printed_set, resolve
from ipldt.workflows._cli import add_dt_arguments, dt_flag_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipldt-pipeline",
        description="Run the IPL-style pipeline on a Scanco AIM: autocontour -> cortical/trabecular compartments -> "
                    "Laplace-Hamming SEG -> Tb.Th, Tb.Sp, Tb.N, Ct.Th (dt_* maps as IPL computes them) -> cortical pore map "
                    "and Ct.Po + JSON/CSV report.  "
                    "Every parameter can be changed (--params / --set; --print-params --describe lists them); the "
                    "defaults are the validated IPL configuration apart from the departures every report names.",
    )
    parser.add_argument("input_aim", type=str, nargs="?", help="Path to the greyscale AIM file")
    parser.add_argument("output_dir", type=str, nargs="?", help="Output directory (created; use --subfolder to append <basename>/)")
    parser.add_argument("--subfolder", action="store_true", help="write into <output_dir>/<basename>/ as the upstream batch runner does")
    parser.add_argument("--map-format", choices=("nifti", "aim"), default="nifti", help="map / mask file format (default nifti; aim uses the input AIM header)")
    parser.add_argument("--map-units", choices=("voxels", "mm"), default="voxels", help="maps as IPL's integer diameters in voxels (default) or float mm (NIfTI only)")
    parser.add_argument("--no-bmd", action="store_true", help="skip the ORMIR bmd_masked step")
    parser.add_argument("--no-porosity", action="store_true", help="skip the cortical pore cascade (no PORE map, no Ct.Po)")
    parser.add_argument("--no-masks", action="store_true", help="do not write the masks / contours / SEG, only the maps and the report")
    parser.add_argument("--periosteal", metavar="FILE", default=None,
                        help="use this periosteal mask instead of the ORMIR-XCT autocontour (.AIM aligned by position, or a "
                             "NIfTI / MHA / NRRD with origin = pos x element size)")
    parser.add_argument("--lh-voxel-size", type=float, nargs="+", default=None, metavar="MM",
                        help="override the Laplace-Hamming element size: one value (isotropic) or three (x y z); default: the "
                             "AIM header's (x, y, z) element sizes, which is IPL's rule (= --set lh.el_size_mm=...)")
    parser.add_argument("--site", choices=("tibia", "radius"), default=None,
                        help="Script 32 STEP 1 parameters for the compartment masks: tibia (corner_min 200000, close2 50; the "
                             "default, unless a complete --params set records another preset) or radius (Script 33: 800, 30)")
    parser.add_argument("--print-params", action="store_true",
                        help="print the complete effective parameter set (site preset < --params < flags < --set) as JSON and exit")
    parser.add_argument("--describe", action="store_true", help="with --print-params: a table with every parameter's IPL option and meaning")
    add_dt_arguments(parser, all_stages=True)
    return parser


def run(input_aim, output_dir, subfolder=False, map_format="nifti", map_units="voxels", no_bmd=False, no_masks=False,
        lh_voxel_size=None, backend="auto", voxel_size=None, params=None, site=None, parameters=None, periosteal=None,
        no_porosity=False) -> int:
    if subfolder:
        output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(input_aim))[0])
    if lh_voxel_size is not None and not isinstance(lh_voxel_size, (int, float)):
        lh_voxel_size = lh_voxel_size[0] if len(lh_voxel_size) == 1 else tuple(lh_voxel_size)
    report = run_pipeline(input_aim, output_dir, map_format=map_format, backend=backend, voxel_size_mm=voxel_size, params=params,
                          map_units=map_units, compute_bmd=not no_bmd, compute_porosity=not no_porosity, save_masks=not no_masks,
                          lh_voxel_size_mm=lh_voxel_size, site=site, log=Logger(), parameters=parameters, periosteal=periosteal)
    m = report["morphometry"]
    print("\nIPL-style morphometry:")
    for k in ("BV_TV", "Tb_Th_mm", "Tb_Th_sd_mm", "Tb_Sp_mm", "Tb_Sp_sd_mm", "Tb_1N_mm", "Tb_N_per_mm", "Ct_Th_mm", "Ct_Th_sd_mm"):
        if k in m:
            print(f"{k}: {m[k]:.6f}")
    for k, v in (report.get("porosity") or {}).items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    for k, v in report.get("bmd", {}).items():
        print(f"{k}: {v:.6f}")
    ps = report.get("parameter_set") or {}
    print(f"Parameters: {ps.get('statement', '-')}")
    print(f"Report written to {os.path.join(output_dir, report['sample'] + '_report.json')} (and .csv; the parameter set in "
          f"{report['sample']}_parameters.json)")
    return 0


def _overrides(args):
    """site preset < --params < the dedicated flags (dt flags, --lh-voxel-size, --voxel-size) < --set -> ArgParameters."""
    flags = dt_flag_overrides(args)
    if args.lh_voxel_size is not None:
        if len(args.lh_voxel_size) not in (1, 3):
            raise ParameterError(f"--lh-voxel-size takes one value or three (x y z), got {len(args.lh_voxel_size)}")
        flags["lh.el_size_mm"] = args.lh_voxel_size[0] if len(args.lh_voxel_size) == 1 else list(args.lh_voxel_size)
    if args.voxel_size is not None:
        flags["morphometry.voxel_size_mm"] = args.voxel_size
    return parameters_from_args(args, flags)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ap = _overrides(args)
    except ParameterError as exc:
        parser.error(str(exc))
    overrides, site = ap.overrides, ap.site(args.site)
    if args.print_params:
        R = resolve(site, overrides or None, workflow="ipldt")          # sources: the overrides' own (--params / flags / --set)
        if args.describe:
            print(R.params.describe(R.reference))
            print("\n" + ("every value is the validated IPL default\n" + R.statement if R.is_default else
                          "* = differs from the default; this configuration " + R.statement))
        else:
            print(json.dumps(printed_set(R), indent=1))
        return 0
    if not args.input_aim or not args.output_dir:
        parser.error("the following arguments are required: input_aim, output_dir (or --print-params)")
    try:
        return run(input_aim=args.input_aim, output_dir=args.output_dir, subfolder=args.subfolder, map_format=args.map_format,
                   map_units=args.map_units, no_bmd=args.no_bmd, no_masks=args.no_masks, backend=args.backend,
                   site=site, parameters=overrides or None, periosteal=args.periosteal, no_porosity=args.no_porosity)
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
