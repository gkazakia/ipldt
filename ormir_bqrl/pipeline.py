"""ormir_bqrl.pipeline -- `run(aim_path, out_dir, ...)`: AIM -> everything.

  1  stages.load                  ITK ScancoImageIO + ipldt.io.read_aim
  2  the periosteal raster        ORMIR-XCT's autocontour (default), a file (ormir_bqrl.slicer.read_mask, any of a
                                  binary mask / ORMIR-BQRL labelmap / .seg.nrrd / IPL .AIM) or a bool array
  3  stages.compartments          Script 32 / 33 STEP 1 (ipldt.step1) from the rendered periosteal
  4  stages.segment               contours rendered, Laplace-Hamming threshold, IPL's SEG assembly
  5  stages.morphometry           dt_thickness / dt_spacing / dt_number on IPL's grids; 5b stages.bmd;
                                  5c stages.porosity (the cortical pore cascade and Ct.Po, run_pipeline's STEP 5c)
  out stages.write_volumes        masks, SEG, PORE and maps as NIfTI or AIM + the Slicer files; report (JSON / CSV /
                                  MD), preview PNG, log

ipldt.ormir.run_pipeline itself is not called (no periosteal-injection hook, no Slicer / preview / provenance);
the one numeric difference is stated in ormir_bqrl.stages: the SEG assembly is masked with the RENDERED
periosteal contour.  `redo.run_from_masks` re-enters run() for an edited periosteal through RunContext.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime

from ipldt import ormir as engine

from . import PRODUCT, __version__
from . import report as report_mod
from . import slicer, stages
from .stages import Masks, Prov

MASK_ATTR = {"periosteal": "ALL", "cortical": "cort", "trabecular": "trab"}


@dataclass
class RunContext:
    """How redo re-enters run() (private to the package): the kind recorded in the report, the provenance of a
    periosteal raster read by redo, the source report, the edit block and the run's masks for the before /
    after comparison."""
    kind: str = "run"
    periosteal_prov: Prov | None = None
    derived_from: str | None = None
    edits: dict | None = None
    run_masks: Masks | None = None
    log_name: str = "pipeline"
    extra_outputs: dict = field(default_factory=dict)


def mask_changes(before, after):
    """{'periosteal' | 'cortical' | 'trabecular': {'added': n, 'removed': n}} between two Masks."""
    out = {}
    for name, attr in MASK_ATTR.items():
        a, b = getattr(before, attr), getattr(after, attr)
        out[name] = {"added": int((b & ~a).sum()), "removed": int((a & ~b).sum())}
    return out


def done_line(base, kind, report):
    """The closing log line of a run / redo: the headline metrics (Ct.Po when it was computed) and the time."""
    mm = report["morphometry"]
    po = (report.get("porosity") or {}).get("Ct_Po")
    return (f"DONE {base} ({kind}): BV/TV {mm['BV_TV']:.4f}  Tb.Th {mm['Tb_Th_mm']:.4f}  Tb.Sp {mm['Tb_Sp_mm']:.4f}  "
            f"Tb.N {mm['Tb_N_per_mm']:.4f}  Ct.Th {mm['Ct_Th_mm']:.4f}" + (f"  Ct.Po {po:.4f}" if po is not None else "")
            + f"  [{report['timing_s']['total']:.0f}s]")


def _notes_from_info(info):
    notes = []
    n = int(info.get("dropped_outside_grid") or 0)
    if n:
        notes.append(f"{n} voxels outside the AIM grid dropped")
    return notes


def periosteal_raster(periosteal, loaded, log=None, file_source="file"):
    """The periosteal raster (bool on the grid) and its provenance: None -> ORMIR-XCT's autocontour; a path ->
    ormir_bqrl.slicer.read_mask(select='periosteal'); an array / SimpleITK image -> as is."""
    log = log if log is not None else engine.Logger(echo=False)
    grid = loaded.grid
    if periosteal is None:
        prx = stages.autocontour(loaded, log)
        return prx, Prov("ormir_xct.autocontour", voxels=int(prx.sum()))
    if isinstance(periosteal, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(periosteal))
        log(f"STEP 2 - periosteal mask from file {path} ...")
        prx, info = slicer.read_mask(path, grid, select="periosteal")
        prx = stages.as_bool(prx, grid, "periosteal mask")
        log(f"  periosteal voxels: {int(prx.sum()):,d} (labels {info.get('labels_used')}; dropped outside the grid: "
            f"{int(info.get('dropped_outside_grid') or 0)})")
        return prx, Prov(file_source, path=path, sha256=info.get("sha256"), voxels=int(prx.sum()), notes=_notes_from_info(info))
    prx = stages.as_bool(periosteal, grid, "periosteal mask")
    log(f"STEP 2 - periosteal mask given as an array: {int(prx.sum()):,d} voxels")
    return prx, Prov("array", voxels=int(prx.sum()))


def run(aim_path, out_dir, site="tibia", periosteal=None, compute_bmd=True, backend="auto", map_units="voxels",
        dt_params=None, step1_params=None, preview=True, log=None, command=None, compute_porosity=True, map_format="nifti",
        _ctx=None):
    """AIM -> everything.  Returns the report dict (ormir_bqrl/README.md, "Outputs"; also <base>_report.json).

    aim_path      Scanco AIM (greyscale, native int16)
    out_dir       the output folder (ormir_bqrl/README.md, "Outputs": <base>_HU.nii.gz, the mask / gobj / SEG / PORE / map volumes on the
                  AIM grid, <base>_compartments.seg.nrrd + _labelmap.nii.gz for Slicer, report.json / .csv / .md, preview)
    site          'tibia' (Script 32 STEP 1 parameters) or 'radius' (Script 33); step1_params overrides ('custom')
    periosteal    None = ORMIR-XCT autocontour; a file path (binary mask, ORMIR-BQRL labelmap, .seg.nrrd, .AIM);
                  a bool (z, y, x) array or SimpleITK image on the AIM grid
    compute_bmd   ORMIR-XCT bmd_masked inside the rendered contours (never fails the morphometry)
    backend       'auto' (CuPy when available) / 'gpu' / 'cpu' -- identical results
    map_units     'voxels' (int16 diameters, IPL's AIM values) or 'mm' (float32; NIfTI only)
    dt_params     ipldt.ormir.DTParams (Script 32's by default)
    preview       write <base>_preview.png (skipped with a log line when matplotlib is missing)
    log           an ipldt.ormir.Logger (default: a new echoing one; its lines go to <base>_pipeline.log)
    command       the argv to record in report['run']['command'] (the CLI passes it)
    compute_porosity  the cortical pore cascade inside the rendered cortical contour: <base>_PORE and Ct.Po
                  (report['porosity'], report['summary']['Ct_Po']); never fails the morphometry
    map_format    'nifti' (default) or 'aim': the masks, SEG, PORE and dt maps as char AIMs on the input AIM's
                  header (ipldt.io.write_aim, as ipldt-pipeline --map-format aim); needs map_units='voxels'"""
    ctx = _ctx or RunContext()
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    t_start = time.time()
    log = log if log is not None else engine.Logger()
    stages.check_output_format(map_format, map_units)
    p1 = engine.step1_params_for(site, step1_params)
    site_name = str(site).lower() if step1_params is None else "custom"
    dt_params = dt_params or engine.IPL_SCRIPT32
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    timing = {}
    log(f"{PRODUCT} {__version__} {ctx.kind}: {os.path.abspath(aim_path)} -> {out_dir} (site {site_name})")

    t = time.time()
    loaded = stages.load(aim_path, log)
    grid = loaded.grid
    stages.check_output_format(map_format, map_units, loaded.native)
    timing["1_load"] = time.time() - t

    t = time.time()
    if ctx.periosteal_prov is not None:                       # redo: the raster was read and attributed by redo
        prx_raw = stages.as_bool(periosteal, grid, "periosteal mask")
        prov = ctx.periosteal_prov
        log(f"STEP 2 - periosteal mask from the redo ({prov.source}): {int(prx_raw.sum()):,d} voxels")
    else:
        prx_raw, prov = periosteal_raster(periosteal, loaded, log)
    timing["2_periosteal"] = time.time() - t
    if not prx_raw.any():
        raise ValueError("the periosteal mask is empty: nothing to separate")

    t = time.time()
    cort, trab, ALL, s1 = stages.compartments(loaded, prx_raw, p1, log)
    timing["3_compartments"] = time.time() - t
    prov.voxels = int(prx_raw.sum())
    prov.rendered_voxels = int(ALL.sum())
    masks = Masks(ALL=ALL, cort=cort, trab=trab, prx_raw=prx_raw,
                  provenance={"periosteal": prov,
                              "cortical": Prov("ipldt.step1", voxels=int(cort.sum())),
                              "trabecular": Prov("ipldt.step1", voxels=int(trab.sum()))})

    t = time.time()
    segmentation, masks.G_cort, masks.G_trab = stages.segment(loaded, ALL, cort, trab, log=log)
    timing["4_render_segment"] = time.time() - t
    masks.provenance["cortical"].rendered_voxels = int(masks.G_cort.sum())
    masks.provenance["trabecular"].rendered_voxels = int(masks.G_trab.sum())

    t = time.time()
    morph = stages.morphometry(loaded, segmentation.seg, masks.G_trab, cort, masks.G_cort, dt_params, backend, log)
    timing["5_dt"] = time.time() - t

    bmd = {}
    if compute_bmd:
        t = time.time()
        bmd = stages.bmd(loaded, masks.G_trab, masks.G_cort, log)
        timing["5b_bmd"] = time.time() - t

    poro = stages.Porosity(pore=None, metrics={})
    if compute_porosity:
        t = time.time()
        poro = stages.porosity(loaded, masks.G_cort, segmentation.cort_seg, log)
        timing["5c_porosity"] = time.time() - t

    t = time.time()
    outputs = stages.write_volumes(out_dir, loaded, masks, segmentation, morph, map_units, log, pore=poro.pore,
                                   map_format=map_format)
    outputs.update(ctx.extra_outputs)
    outputs["log"] = os.path.join(out_dir, f"{loaded.base}_{ctx.log_name}.log")
    outputs["preview"] = None
    outputs["edit_preview"] = None
    timing["6_write"] = time.time() - t

    edits = None
    if ctx.edits is not None:
        edits = dict(ctx.edits)
        if ctx.run_masks is not None:
            edits["vs_run"] = mask_changes(ctx.run_masks, masks)
    timing["total"] = time.time() - t_start
    report = report_mod.build_report(kind=ctx.kind, loaded=loaded, masks=masks, segmentation=segmentation, morph=morph,
                                     bmd=bmd, site=site_name, step1_params=p1, step1_info=s1, dt_params=dt_params,
                                     map_units=map_units, compute_bmd=compute_bmd, outputs=outputs, timing=timing,
                                     started=started, duration_s=timing["total"], command=command, out_dir=out_dir,
                                     derived_from=ctx.derived_from, edits=edits, porosity=poro, map_format=map_format)
    if preview:
        try:
            from .preview import write_edit_preview, write_preview
            p = os.path.join(out_dir, f"{loaded.base}_preview.png")
            report["outputs"]["preview"] = write_preview(p, loaded, masks, segmentation, morph, report)
            if ctx.run_masks is not None and edits and edits.get("edited"):
                p = os.path.join(out_dir, f"{loaded.base}_redo_edits.png")
                report["outputs"]["edit_preview"] = write_edit_preview(p, ctx.run_masks, masks, edits["edited"][0])
        except ImportError as exc:
            log(f"  preview skipped: {exc}")
    report["timing_s"]["total"] = time.time() - t_start
    report["run"]["duration_s"] = report["timing_s"]["total"]
    report_mod.write_reports(report, out_dir, loaded.base)
    log(done_line(loaded.base, ctx.kind, report))
    if isinstance(log, engine.Logger):
        with open(outputs["log"], "w", encoding="utf-8") as fh:
            fh.write("\n".join(log.lines) + "\n")
    return report


__all__ = ["run", "RunContext", "periosteal_raster", "mask_changes", "done_line", "MASK_ATTR"]
