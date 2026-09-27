"""ormir-bqrl -- the command line of ORMIR-BQRL.

    ormir-bqrl run   <aim> <out_dir> [--site tibia|radius] [--periosteal FILE] [--no-bmd] [--no-porosity]
                     [--backend auto|gpu|cpu] [--map-units voxels|mm] [--map-format nifti|aim] [--subfolder] [--no-preview]
                     [--params FILE] [--set STAGE.NAME=VALUE ...]
    ormir-bqrl redo  <aim> <run_dir> [--periosteal FILE] [--trab FILE] [--cort FILE] [--site tibia|radius]
                     [--out DIR] [--no-bmd] [--no-porosity] [--backend ...] [--map-units ...] [--map-format ...]
                     [--params FILE] [--set STAGE.NAME=VALUE ...]
    ormir-bqrl slicer-export <run_dir> [--base BASE] [--out DIR]
    ormir-bqrl batch <out_root> <aim>... [--site] [--no-bmd] [--no-porosity] [--backend] [--map-units] [--map-format]
                     [--periosteal-dir DIR] [--params FILE] [--set STAGE.NAME=VALUE ...]
    ormir-bqrl params [--site tibia|radius] [--params FILE] [--set STAGE.NAME=VALUE ...] [--describe] [--out FILE]
    ormir-bqrl version

PARAMETERS.  Every tunable value of every stage is a parameter STAGE.NAME (ipldt.params; `ormir-bqrl params
--describe` lists all of them with the IPL option each one is), and the defaults are the validated IPL configuration
apart from the deliberate Tb.Th object (the whole SEG; IPL's evaluation script, and the comparison with IPL, use TRAB_SEG).
`--params FILE` reads a JSON file (the values to change, or a complete set as `ormir-bqrl params` prints it, a run's
<base>_parameters.json or a run report); `--set lh.laplace_eps=0.5` (repeatable) changes one value.  Precedence: the
--site preset < --params < --set.  A complete set is its preset plus its changes: without --site its preset is used
(so `--params <run>/<base>_parameters.json` repeats that run), with --site its changes apply to that site's preset.
A redo starts from the run's recorded set.  An unknown name or a value of the wrong type is a usage error (exit 2)
that lists the valid names; a non-default run says so in its report, and a value of a step that does not run (e.g.
autocontour.* with --periosteal) is warned about and listed as not applied.

Every run writes the masks, SEG, the cortical pore map (PORE) and the dt maps as NIfTI (default) or, with
--map-format aim, as char AIMs on the input AIM's header (maps as integer diameters in voxels, so not with
--map-units mm); Ct.Po (the fraction of the rendered cortical contour occupied by the pore map) is in the report
unless --no-porosity.  <base>_HU.nii.gz, the Slicer files and the report are the same in both formats.

Exit codes: 0; 1 with `Error: <message>` on stderr (as the ipldt workflows do); 2 for argparse usage errors.
main(argv=None) -> int for in-process use; `python -m ormir_bqrl.cli ...` works too.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
import traceback

from ipldt import params as _pm

from . import PRODUCT, __version__

SUMMARY_KEYS = ("BV_TV", "Tb_Th_mm", "Tb_Th_sd_mm", "Tb_Sp_mm", "Tb_Sp_sd_mm", "Tb_N_per_mm", "Ct_Th_mm", "Ct_Th_sd_mm",
                "Ct_Po", "Tb_BMD_mgHA_cm3", "Ct_BMD_mgHA_cm3")
PERIOSTEAL_CANDIDATES = ("{base}_PRX_MASK.nii.gz", "{base}_compartments.seg.nrrd", "{base}.seg.nrrd", "{base}_PRX_MASK.nrrd",
                         "{base}.nii.gz", "{base}.nrrd", "{base}.AIM", "{base}.aim")


# ------------------------------------------------------------------------------------------ parser
def _add_common(p, redo=False):
    p.add_argument("--site", choices=("tibia", "radius"), default=None,
                   help="Script 32 STEP 1 preset: tibia (Script 32) or radius (Script 33)"
                        + ("; a redo needs it only when the periosteal is edited (STEP 1 reruns), else the run report's "
                           "site is used" if redo else "; default tibia, or the preset of a complete --params set"))
    p.add_argument("--no-bmd", action="store_true", help="skip the ORMIR-XCT bmd_masked step")
    p.add_argument("--no-porosity", action="store_true",
                   help="skip the cortical pore cascade (no <base>_PORE, no Ct.Po); by default it runs inside the rendered cortical contour")
    p.add_argument("--backend", choices=("auto", "gpu", "cpu"), default="auto", help="dt_* on CuPy when available (auto), or force gpu / cpu; results are identical")
    p.add_argument("--map-units", choices=("voxels", "mm"), default=None if redo else "voxels",
                   help="dt maps as IPL's integer diameters in voxels (default) or float32 mm (NIfTI only)"
                        + ("; default: the run's" if redo else ""))
    p.add_argument("--map-format", choices=("nifti", "aim"), default=None if redo else "nifti",
                   help="masks, SEG, PORE and dt maps as NIfTI (default) or as char AIMs on the input AIM's header "
                        "(maps in voxels); <base>_HU.nii.gz and the Slicer files are written either way"
                        + ("; default: the run's" if redo else ""))
    _pm.add_parameter_arguments(p, "; `ormir-bqrl params --describe` lists every name"
                                + ("; a redo starts from the run's recorded parameters" if redo else ""))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ormir-bqrl",
                                     description=f"{PRODUCT} {__version__}: IPL-faithful HR-pQCT bone microstructure from a Scanco AIM "
                                                 "(ORMIR-XCT autocontour, ipldt's reimplementation of Script 32/33 STEP 1, Laplace-Hamming SEG, "
                                                 "dt_thickness / dt_spacing / dt_number and the cortical pore cascade), with 3D Slicer mask "
                                                 "correction and a Script-34-style redo.")
    parser.add_argument("--version", action="version", version=f"{PRODUCT} {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    p = sub.add_parser("run", help="AIM -> masks, SEG, dt maps, pore map, BMD, Ct.Po, report, Slicer files",
                       description="Run the whole workflow on one AIM: the masks, SEG, the cortical pore map and the dt maps "
                                   "(NIfTI or AIM), BV/TV, Tb.Th, Tb.Sp, Tb.N, Ct.Th, Ct.Po and BMD in a JSON / CSV / Markdown "
                                   "report, and a 3D Slicer segmentation of the two compartments.")
    p.add_argument("aim", help="the greyscale Scanco AIM")
    p.add_argument("out_dir", help="output folder (created; --subfolder appends <basename>/)")
    p.add_argument("--periosteal", metavar="FILE", default=None,
                   help="use this periosteal mask instead of the ORMIR-XCT autocontour (.nii/.nii.gz/.nrrd/.seg.nrrd/.AIM on the AIM grid)")
    p.add_argument("--subfolder", action="store_true", help="write into <out_dir>/<basename>/")
    p.add_argument("--no-preview", action="store_true", help="skip the preview PNG")
    _add_common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("redo", help="re-enter from masks corrected in 3D Slicer (Script 34 semantics)",
                       description="Re-run from a run folder with the masks you corrected in 3D Slicer.  Pass only the flag that names "
                                   "what you edited: --periosteal (Script 32 STEP 1 reruns from it), --trab (Script 34: cortical = "
                                   "periosteal - trabecular) or --cort (the symmetric extension); --trab and --cort together are refused.  "
                                   "Everything downstream of the compartments (SEG, dt maps, the pore map and Ct.Po, BMD) is recomputed; "
                                   "the output format and map units default to the run's.")
    p.add_argument("aim", help="the greyscale Scanco AIM of the run")
    p.add_argument("run_dir", help="the folder of the automatic run (or of a previous redo)")
    p.add_argument("--periosteal", metavar="FILE", default=None, help="edited periosteal contour")
    p.add_argument("--trab", metavar="FILE", default=None, help="edited trabecular compartment (.seg.nrrd, labelmap or binary mask)")
    p.add_argument("--cort", metavar="FILE", default=None, help="edited cortical compartment")
    p.add_argument("--out", metavar="DIR", default=None, help="output folder (default <run_dir>/redo, then redo_2, ...)")
    _add_common(p, redo=True)
    p.set_defaults(func=cmd_redo)

    p = sub.add_parser("slicer-export", help="rebuild the .seg.nrrd and labelmap NIfTI from a run folder's masks",
                       description="Rebuild <base>_compartments.seg.nrrd and <base>_compartments_labelmap.nii.gz from a run folder's "
                                   "<base>_CORT_MASK.nii.gz / <base>_TRAB_MASK.nii.gz (ORMIR-BQRL or ipldt-pipeline output).")
    p.add_argument("run_dir")
    p.add_argument("--base", default=None, help="sample base name when the folder holds several runs")
    p.add_argument("--out", metavar="DIR", default=None, help="write the two files elsewhere (default: the run folder)")
    p.set_defaults(func=cmd_slicer_export)

    p = sub.add_parser("batch", help="run several AIMs sequentially into <out_root>/<base>/ with a batch_summary.csv",
                       description="One run per AIM into <out_root>/<base>/; failures are logged to batch.log and skipped; "
                                   "batch_summary.csv collects one row per sample.")
    p.add_argument("out_root")
    p.add_argument("aims", nargs="+", metavar="aim")
    p.add_argument("--periosteal-dir", metavar="DIR", default=None,
                   help="inject <DIR>/<base>_PRX_MASK.nii.gz (or <base>.seg.nrrd / <base>.AIM) as the periosteal of each sample that has one there")
    p.add_argument("--no-preview", action="store_true", help="skip the preview PNGs")
    _add_common(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("params", help="print the complete parameter set (the defaults, or with --params / --set applied)",
                       description="Print the complete effective parameter set as JSON (loadable again with --params): the "
                                   "--site preset < --params FILE < --set.  --describe prints a table instead: every "
                                   "parameter with its value, the default where it differs, the IPL option and what it does.")
    p.add_argument("--site", choices=("tibia", "radius"), default=None,
                   help="the STEP 1 preset (default tibia, or the preset of a complete --params set)")
    p.add_argument("--describe", action="store_true", help="a table with the IPL option and documentation of every parameter")
    p.add_argument("--out", metavar="FILE", default=None, help="write the JSON to FILE instead of printing it")
    _pm.add_parameter_arguments(p)
    p.set_defaults(func=cmd_params)

    p = sub.add_parser("version", help="print the versions of ORMIR-BQRL, ipldt, ormir_xct, SimpleITK, itk and CuPy")
    p.set_defaults(func=cmd_version)
    return parser


# ------------------------------------------------------------------------------------------ helpers
def _import_pipeline():
    try:
        from .pipeline import run
    except ImportError as exc:
        raise RuntimeError(f"the ORMIR-BQRL pipeline is not available ({exc}); install the [bqrl] extra "
                           "(SimpleITK, itk, itk-scanco, ormir-xct)") from exc
    return run


def _import_redo():
    try:
        from .redo import run_from_masks
    except ImportError as exc:
        raise RuntimeError(f"the ORMIR-BQRL redo entry is not available ({exc}); install the [bqrl] extra") from exc
    return run_from_masks


def _report_json_path(report, out_dir):
    outputs = report.get("outputs") or {}
    for key in ("report_json", "report"):
        if outputs.get(key):
            return outputs[key]
    return os.path.join(out_dir, f"{report.get('sample', 'sample')}_report.json")


def print_summary(report, out_dir=None):
    src = report.get("summary") or report.get("morphometry") or {}
    bmd = report.get("bmd") or {}
    print(f"\n{PRODUCT} summary ({report.get('sample', '?')}, {report.get('site', '?')}, {(report.get('run') or {}).get('kind', 'run')}):")
    for k in SUMMARY_KEYS:
        v = src.get(k, bmd.get(k))
        if v is None:
            continue
        try:
            print(f"  {k}: {float(v):.6f}")
        except (TypeError, ValueError):
            print(f"  {k}: {v}")
    masks = report.get("masks") or {}
    prov = [f"{name} {info.get('source')}" for name, info in masks.items() if isinstance(info, dict) and info.get("source")]
    if prov:
        print("  masks: " + "; ".join(prov))
    if out_dir is not None:
        print(f"Report written to {_report_json_path(report, out_dir)} (and .csv / .md)")


def _summary_row(report):
    try:
        from .report import summary_row
        return dict(summary_row(report))
    except Exception:
        pass
    src = report.get("summary") or report.get("morphometry") or {}
    bmd = report.get("bmd") or {}
    row = dict(sample=report.get("sample"), site=report.get("site"), kind=(report.get("run") or {}).get("kind", "run"))
    for k in SUMMARY_KEYS:
        row[k] = src.get(k, bmd.get(k))
    masks = report.get("masks") or {}
    for name in ("periosteal", "cortical", "trabecular"):
        row[f"{name}_source"] = (masks.get(name) or {}).get("source") if isinstance(masks.get(name), dict) else None
    row["out_dir"] = (report.get("run") or {}).get("out_dir")
    return row


def find_periosteal(periosteal_dir, base):
    """The periosteal file of `base` in `periosteal_dir` (first of PERIOSTEAL_CANDIDATES that exists), or None."""
    if not periosteal_dir:
        return None
    for pattern in PERIOSTEAL_CANDIDATES:
        p = os.path.join(periosteal_dir, pattern.format(base=base))
        if os.path.isfile(p):
            return p
    return None


def _dist_version(dist_names, module_name=None, attr="__version__"):
    from importlib import metadata
    for name in ((dist_names,) if isinstance(dist_names, str) else dist_names):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    if module_name:
        try:
            mod = __import__(module_name)
            v = getattr(mod, attr, None)
            if callable(v):
                v = v()
            if v:
                return str(v)
        except Exception:
            return "not available"
    return "not available"


def version_lines():
    lines = [f"{PRODUCT} {__version__}", "ipldt " + _dist_version("ipldt", "ipldt"),
             "ormir_xct " + _dist_version(("ormir-xct", "ormir_xct"), "ormir_xct"),
             "SimpleITK " + _dist_version("SimpleITK", "SimpleITK", "Version_VersionString"),
             "itk " + _dist_version("itk", None)]
    cupy = _dist_version(("cupy", "cupy-cuda12x", "cupy-cuda13x", "cupy-cuda11x"), "cupy")
    if cupy != "not available":
        try:
            from ipldt.gpu import cupy_available
            cupy += " (CUDA device available)" if cupy_available() else " (no CUDA device: CPU path)"
        except Exception:
            cupy += " (device check failed: CPU path)"
    lines.append("CuPy " + cupy)
    lines.append(f"Python {platform.python_version()} ({sys.platform})")
    return lines


# ------------------------------------------------------------------------------------------ commands
def cmd_run(args) -> int:
    run = _import_pipeline()
    out_dir = args.out_dir
    if args.subfolder:
        out_dir = os.path.join(out_dir, os.path.splitext(os.path.basename(args.aim))[0])
    if not os.path.isfile(args.aim):
        raise FileNotFoundError(f"AIM not found: {args.aim}")
    if args.periosteal and not os.path.isfile(args.periosteal):
        raise FileNotFoundError(f"periosteal mask not found: {args.periosteal}")
    report = run(args.aim, out_dir, site=_site(args), periosteal=args.periosteal, compute_bmd=not args.no_bmd,
                 backend=args.backend, map_units=args.map_units, preview=not args.no_preview,
                 command=getattr(args, "command_line", None), compute_porosity=not args.no_porosity,
                 map_format=args.map_format, **_parameters_kw(args))
    print_summary(report, out_dir)
    return 0


def _site(args):
    """The site to run: --site, else the preset of a complete --params set, else tibia."""
    return args.site or getattr(args, "param_preset", None) or "tibia"


def _parameters_kw(args):
    """{'parameters': the --params / --set overrides} when any were given, else {} (the call is then exactly the one
    made before the parameter model)."""
    ov = getattr(args, "param_overrides", None)
    return {"parameters": ov} if ov else {}


def cmd_redo(args) -> int:
    run_from_masks = _import_redo()
    if not os.path.isfile(args.aim):
        raise FileNotFoundError(f"AIM not found: {args.aim}")
    if not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f"run folder not found: {args.run_dir}")
    for name in ("periosteal", "trab", "cort"):
        p = getattr(args, name)
        if p and not os.path.isfile(p):
            raise FileNotFoundError(f"--{name} file not found: {p}")
    kwargs = dict(out_dir=args.out, periosteal=args.periosteal, trab=args.trab, cort=args.cort, site=args.site,
                  compute_bmd=not args.no_bmd, backend=args.backend, command=getattr(args, "command_line", None),
                  compute_porosity=not args.no_porosity, **_parameters_kw(args))
    if args.map_units is not None:
        kwargs["map_units"] = args.map_units
    if args.map_format is not None:
        kwargs["map_format"] = args.map_format
    report = run_from_masks(args.aim, args.run_dir, **kwargs)
    out_dir = (report.get("run") or {}).get("out_dir") or args.out or args.run_dir
    print_summary(report, out_dir)
    return 0


def cmd_slicer_export(args) -> int:
    from .slicer import export_run
    res = export_run(args.run_dir, base=args.base, out_dir=args.out)
    print(f"{res['base']}: cortical {res['cortical_voxels']:,d} voxels, trabecular {res['trabecular_voxels']:,d} voxels on grid "
          f"dim {tuple(res['grid']['dim_xyz'])} pos {tuple(res['grid']['pos_xyz'])}")
    print(f"  {res['seg_nrrd']}")
    print(f"  {res['labelmap']}")
    return 0


def cmd_batch(args) -> int:
    run = _import_pipeline()
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    log_path = os.path.join(out_root, "batch.log")
    rows, n_ok = [], 0

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    ov = getattr(args, "param_overrides", None) or {}
    site = _site(args)
    log(f"{PRODUCT} {__version__} batch: {len(args.aims)} AIM(s) -> {out_root} (site {site}, bmd {'off' if args.no_bmd else 'on'}, "
        f"porosity {'off' if args.no_porosity else 'on'}, format {args.map_format}, backend {args.backend}, "
        f"periosteal-dir {args.periosteal_dir or '-'}"
        + (", parameters " + ", ".join(f"{k}={v}" for k, v in ov.items()) if ov else "") + ")")   # a default batch: as before
    for aim in args.aims:
        base = os.path.splitext(os.path.basename(aim))[0]
        out_dir = os.path.join(out_root, base)
        periosteal = find_periosteal(args.periosteal_dir, base)
        t0 = time.time()
        log(f"{base}: start" + (f" (periosteal from {periosteal})" if periosteal else " (autocontour)"))
        try:
            if not os.path.isfile(aim):
                raise FileNotFoundError(f"AIM not found: {aim}")
            report = run(aim, out_dir, site=site, periosteal=periosteal, compute_bmd=not args.no_bmd,
                         backend=args.backend, map_units=args.map_units, preview=not args.no_preview,
                         command=getattr(args, "command_line", None), compute_porosity=not args.no_porosity,
                         map_format=args.map_format, **_parameters_kw(args))
            row = _summary_row(report)
            row.update(status="ok", error="", out_dir=row.get("out_dir") or out_dir)
            n_ok += 1
            log(f"{base}: done in {time.time() - t0:.0f} s")
        except Exception as exc:
            row = dict(sample=base, site=site, kind="run", status="failed", error=f"{type(exc).__name__}: {exc}", out_dir=out_dir)
            log(f"{base}: FAILED after {time.time() - t0:.0f} s -- {type(exc).__name__}: {exc}")
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(traceback.format_exc())
        rows.append(row)
    columns = []
    for row in rows:
        for k in row:
            if k not in columns:
                columns.append(k)
    for k in ("sample", "status"):
        if k in columns:
            columns.remove(k)
            columns.insert(0 if k == "sample" else 1, k)
    if "non_default_parameters" in columns:          # present only for a non-default batch; last, after every other column
        columns.remove("non_default_parameters")
        columns.append("non_default_parameters")
    summary_path = os.path.join(out_root, "batch_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    log(f"batch done: {n_ok}/{len(rows)} succeeded; summary {summary_path}")
    return 0 if n_ok or not rows else 1


def cmd_params(args) -> int:
    R = _pm.resolve(_site(args), args.param_overrides or None, workflow="ormir_bqrl")   # sources: the overrides' own
    if args.describe:
        text = R.params.describe(R.reference) + "\n\n" + ("* = differs from the default; this configuration " + R.statement
                                                           if not R.is_default else "every value is the validated IPL default\n"
                                                           + R.statement)
    else:
        text = json.dumps(_pm.printed_set(R), indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote {os.path.abspath(args.out)}")
    else:
        print(text)
    return 0


def cmd_version(args) -> int:
    for line in version_lines():
        print(line)
    return 0


# ------------------------------------------------------------------------------------------ entry
def main(argv=None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv_list)
    args.command_line = ["ormir-bqrl", *argv_list]          # recorded as report['run']['command']
    if hasattr(args, "set_params"):                          # --params / --set: checked before anything runs
        try:
            ap = _pm.parameters_from_args(args)
        except _pm.ParameterError as exc:
            parser.error(str(exc))
        args.param_overrides, args.param_preset = ap.overrides, ap.preset
    try:
        return int(args.func(args))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
