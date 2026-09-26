"""ipldt-ipl-morphometry -- Script 32's dt stage (Tb.Th, Tb.Sp, 1/Tb.N, Ct.Th) on existing masks.

Two modes:
  AIM mode (IPL's own files, aligned by global position): --subject-dir <folder with <base>_{SEG,TRAB_MASK,
      CORT_MASK,...}_decompressed.AIM> or explicit --seg / --trab-mask [--cort-mask] [--trab-seg] AIM paths.
      When IPL's TRAB_TH / TRAB_SP / TRAB_1N / CORT_TH maps are present they are compared voxel for voxel.
  image mode (NIfTI / MHA on one grid): --seg / --trab-mask [--cort-mask] via the SimpleITK wrappers.
"""
import argparse
import os

from ipldt.workflows._cli import add_dt_arguments, dt_params_from_args, is_aim


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipldt-ipl-morphometry",
        description="IPL dt_thickness / dt_spacing / dt_number (+ Ct.Th) on existing SEG and contour masks; compares against IPL's maps when given AIMs.",
    )
    parser.add_argument("output_dir", type=str, help="Output directory for the maps and the JSON/CSV report")
    parser.add_argument("--subject-dir", type=str, default=None, help="folder with <base>_{SEG,TRAB_MASK,CORT_MASK,TRAB_SEG,TRAB_TH,TRAB_SP,TRAB_1N,CORT_TH}_decompressed.AIM")
    parser.add_argument("--seg", type=str, default=None, help="whole-bone segmentation (cortical + trabecular bone)")
    parser.add_argument("--trab-mask", type=str, default=None, help="trabecular contour mask (raw raster)")
    parser.add_argument("--cort-mask", type=str, default=None, help="cortical compartment mask (raw raster) for Ct.Th")
    parser.add_argument("--trab-seg", type=str, default=None, help="IPL's TRAB_SEG for the old Tb.Th definition (AIM mode)")
    parser.add_argument("--rendered", action="store_true", help="image mode: the masks are already rendered contours")
    parser.add_argument("--map-format", choices=("nifti", "aim"), default="nifti")
    parser.add_argument("--no-maps", action="store_true", help="report only")
    parser.add_argument("--no-old-tbth", action="store_true", help="AIM mode: skip dt_thickness on TRAB_SEG (old Tb.Th)")
    add_dt_arguments(parser)
    return parser


def run(args) -> int:
    params = dt_params_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    aim_mode = args.subject_dir is not None or all(is_aim(p) for p in (args.seg, args.trab_mask, args.cort_mask, args.trab_seg) if p)
    if aim_mode:
        from ipldt.ormir import Logger, run_on_ipl_masks
        report = run_on_ipl_masks(subject_dir=args.subject_dir, seg_path=args.seg, trab_mask_path=args.trab_mask, cort_mask_path=args.cort_mask,
                                  trab_seg_path=args.trab_seg, out_dir=args.output_dir, map_format=args.map_format, backend=args.backend,
                                  voxel_size_mm=args.voxel_size, params=params, include_old_tbth=not args.no_old_tbth,
                                  write_maps=not args.no_maps, log=Logger())
        print("\nMorphometry:")
        for k, v in report["morphometry"].items():
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
        if report["comparison_vs_ipl"]:
            print("\nVoxel-for-voxel comparison with IPL's maps:")
            for k, v in report["comparison_vs_ipl"].items():
                print(f"{k}: {v['mismatches']} mismatches of {v['voxels']} voxels")
        return 0
    # image mode
    import SimpleITK as sitk
    from ipldt.ormir import ipl_cortical_thickness_sitk, ipl_trabecular_microarchitecture_sitk, write_report
    if not (args.seg and args.trab_mask):
        raise ValueError("image mode needs --seg and --trab-mask")
    seg = sitk.ReadImage(args.seg, sitk.sitkUInt8)
    trab = sitk.ReadImage(args.trab_mask, sitk.sitkUInt8)
    metrics, th, sp, n1 = ipl_trabecular_microarchitecture_sitk(seg, trab, gobj_rendered=args.rendered, voxel_size_mm=args.voxel_size,
                                                                params=params, backend=args.backend)
    base = os.path.splitext(os.path.basename(args.seg))[0].split(".")[0]
    ext = ".AIM" if args.map_format == "aim" else ".nii.gz"
    if args.map_format == "aim":
        raise ValueError("image mode writes NIfTI (an AIM header is only available in AIM mode)")
    if not args.no_maps:
        for name, img in (("TRAB_TH", th), ("TRAB_SP", sp), ("TRAB_1N", n1)):
            sitk.WriteImage(img, os.path.join(args.output_dir, f"{base}_{name}_ipldt{ext}"))
    if args.cort_mask:
        cort = sitk.ReadImage(args.cort_mask, sitk.sitkUInt8)
        cmetrics, ct = ipl_cortical_thickness_sitk(cort, gobj_rendered=args.rendered, voxel_size_mm=args.voxel_size, params=params, backend=args.backend)
        metrics.update(cmetrics)
        if not args.no_maps:
            sitk.WriteImage(ct, os.path.join(args.output_dir, f"{base}_CORT_TH_ipldt{ext}"))
    report = {"sample": base, "inputs": {"seg": args.seg, "trab_mask": args.trab_mask, "cort_mask": args.cort_mask},
              "parameters": {**params.kwargs(), "backend": args.backend}, "morphometry": metrics}
    write_report(report, os.path.join(args.output_dir, f"{base}_morphometry_report.json"), os.path.join(args.output_dir, f"{base}_morphometry_report.csv"))
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
