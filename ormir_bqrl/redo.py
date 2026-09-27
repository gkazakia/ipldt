"""ormir_bqrl.redo -- `run_from_masks(aim_path, run_dir, ...)`: the manual-correction re-entry, Scanco Script 34's
semantics stated precisely (ormir_bqrl/README.md, "Correcting masks in 3D Slicer").  Let P, T, C be the given rasters (bool on the AIM grid after
ormir_bqrl.slicer.read_mask) and R(m) = ipldt.ormir.render_on_own_box(m), IPL's /gobj_to_aim of the contour of m
(what IPL sees of any contour, because IPL has no rasters, only GOBJs):

  P only    run(aim, out, site, periosteal=P): Script 32 / 33 STEP 1 reruns from R(P); cort / trab automatic
            (provenance periosteal "manual", cortical / trabecular "ipldt.step1")           rule step1_from_periosteal
  T only    ALL from the run; refused unless T <= ALL (the offending voxels go to <base>_redo_outside.nii.gz);
            T_R = R(T) & ALL (rendering fill outside ALL clipped and counted); CORT = ALL - T_R (/subtract_aims);
            G_trab = T_R (the edited contour's own rendering IS its gobj), G_cort = R(CORT) (/togobj_from_aim of
            cort_mask, rendered); then SEG = ipl_seg_assembly(lh, ALL, G_cort, G_trab) and Script 32's dt stage:
            Tb.* on bbox(SEG) with G_trab, Ct.Th of CORT on bbox(CORT) with G_cort          rule script34_trab
            = Script 34 STEP 1 exactly, then Script 32's STEP 2 / 3 -- i.e. Script 34 without its two defects:
            TRAB_SEG IS cut by the trabecular contour (Script 34's /gobj targets the non-existent trab_gauss and
            never cuts it) and Ct.Th is evaluated on the bounding box of the derived cortical mask as in Script 32
            (Script 34 skips /bounding_box_cut, so its CORT_MASK lives on the periosteal GOBJ's grid).
  C only    symmetric (the lab's extension; Script 34 accepts only a trabecular contour): C_R = R(C) & ALL,
            TRAB = ALL - C_R, G_cort = C_R, G_trab = R(TRAB); Ct.Th of C_R on bbox(C_R) with gobj C_R
                                                                                            rule script34_cort_ext
  P + T     ALL = R(P) (exactly what step 3 would have done), then the T rule           rule script34_periosteal_trab
  P + C     ALL = R(P), then the C rule                                              rule script34_periosteal_cort_ext
  T + C     refused: the periosteal minus one compartment defines the other (Script 34)
  none      refused: nothing to redo

Consequences: a redo with the run's own masks and no edits reproduces the run bit for bit (the internal path
_from_run_masks, tested); passing the run's own TRAB_MASK as an edit does NOT (T_R = R(TRAB_MASK) differs from
the raw stage-29 raster by the rim voxels the renderer removes, which move into the cortex -- Script 34's
behaviour too; edits.vs_run records the shift); only the periosteal edit re-runs the automatic separation,
compartment edits are taken literally (rendered, never re-segmented).

Every redo recomputes everything downstream of the compartments from its own masks: the SEG assembly, the dt
maps, BMD and the cortical pore cascade (stages.porosity: Ct.Po and <base>_PORE from the redo's G_cort and
CORT_SEG), so an edited compartment always gets its own pore map.  The run folder may hold NIfTI or AIM volumes
(map_format of the run); the redo writes in the run's format unless map_format is given.

PARAMETERS.  A redo keeps the run's parameters: the complete set recorded in the run report (report['parameter_set'],
also <base>_parameters.json) is the redo's starting point, and `site`, `parameters` (a mapping of overrides, a JSON
file or a complete ipldt.params.Parameters) and `dt_params` override it in that order.  So a redo with no override
evaluates exactly as its run did, including under non-default parameters (the `_from_run_masks` invariant holds for
any parameter set).  A report that predates the parameter model (no parameter_set) gives the defaults with its
recorded site, Step1Params and DTParams, which is what those runs used.  seg.periosteal_mask is workflow-specific:
a redo of an ipldt-pipeline folder whose run used that workflow's own default ('raw') takes ORMIR-BQRL's
('rendered'), as it always has; with 'raw', a compartment-only redo masks the SEG with the run folder's
<base>_PRX_MASK.  An override of a step that does not run in the redo is not applied: a compartment redo runs neither
STEP 1 nor the autocontour, so step1.* (and `site`) and autocontour.* given to it keep the run's values; a periosteal
redo does not run the autocontour.  Such values are warned about and listed in report['parameter_set']['not_applied'].
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from datetime import datetime

import numpy as np

from ipldt import ormir as engine
from ipldt import params as _pm
from ipldt.io import write_nifti
from ipldt.step1 import Step1Params

from . import PRODUCT, __version__
from . import report as report_mod
from . import slicer, stages
from .pipeline import RunContext, done_line, mask_changes, run
from .stages import Masks, Prov

RULES = {("periosteal",): "step1_from_periosteal",
         ("trabecular",): "script34_trab",
         ("cortical",): "script34_cort_ext",
         ("periosteal", "trabecular"): "script34_periosteal_trab",
         ("periosteal", "cortical"): "script34_periosteal_cort_ext"}
RUN_FILES = {"ALL": "PRX_GOBJ", "cort": "CORT_MASK", "trab": "TRAB_MASK"}


class EditOutsidePeriosteal(ValueError):
    """An edited compartment has voxels outside the periosteal contour (IPL's /subtract_aims is undefined there)."""

    def __init__(self, name, outside):
        self.name = name
        self.outside = outside
        self.count = int(outside.sum())
        super().__init__(f"{self.count:,d} voxels of the edited {name} mask lie outside the periosteal contour; "
                         "correct the periosteal too (pass --periosteal) or keep the edit inside the bone")


@dataclass
class EditSpec:
    """What was given to the redo: rasters on the AIM grid, the files they came from, read_mask's info."""
    periosteal: np.ndarray | None = None
    trab: np.ndarray | None = None
    cort: np.ndarray | None = None
    paths: dict = field(default_factory=dict)
    infos: dict = field(default_factory=dict)

    @property
    def edited(self):
        names = []
        if self.periosteal is not None:
            names.append("periosteal")
        if self.cort is not None:
            names.append("cortical")
        if self.trab is not None:
            names.append("trabecular")
        return tuple(names)

    @property
    def rule(self):
        return RULES.get(self.edited)

    def prov(self, name, voxels, rendered=None, notes=()):
        info = self.infos.get(name) or {}
        return Prov("manual", path=self.paths.get(name), sha256=info.get("sha256"), voxels=int(voxels),
                    rendered_voxels=None if rendered is None else int(rendered), notes=list(notes))


# ============================================================================================ the rule
def derive_compartments(ALL, grid, trab=None, cort=None, min_vertices=None):
    """Script 34 STEP 1 on rasters: exactly one of trab / cort is the edited compartment X (bool on the grid).
    Refuses X outside ALL (EditOutsidePeriosteal, carrying the offending voxels); X_R = R(X) & ALL; the other
    compartment = ALL - X_R.  Returns dict(cort, trab, G_cort, G_trab (the edited one's rendering, the other None
    so stages.segment renders it), edited, derived, raw_voxels, rendered_voxels, rendering_clipped)."""
    if (trab is None) == (cort is None):
        raise ValueError("derive_compartments: pass exactly one of trab / cort")
    name = "trabecular" if trab is not None else "cortical"
    X = stages.as_bool(trab if trab is not None else cort, grid, f"{name} mask")
    ALL = stages.as_bool(ALL, grid, "periosteal contour")
    outside = X & ~ALL
    if outside.any():
        raise EditOutsidePeriosteal(name, outside)
    if not X.any():
        raise ValueError(f"the edited {name} mask is empty")
    R = stages.render(X, grid, min_vertices)
    clipped = int((R & ~ALL).sum())
    X_R = R & ALL
    if not X_R.any():
        raise ValueError(f"the edited {name} mask renders to nothing (IPL's contour rendering removes 1-voxel features)")
    other = ALL & ~X_R
    if not other.any():
        raise ValueError(f"the edited {name} mask fills the whole periosteal contour: the other compartment would be empty")
    out = dict(edited=name, raw_voxels=int(X.sum()), rendered_voxels=int(X_R.sum()), rendering_clipped=clipped)
    if trab is not None:
        out.update(cort=other, trab=X_R, G_cort=None, G_trab=X_R, derived="cortical")
    else:
        out.update(cort=X_R, trab=other, G_cort=X_R, G_trab=None, derived="trabecular")
    return out


# ============================================================================================ the run folder
def next_redo_dir(run_dir):
    """<run_dir>/redo, then redo_2, redo_3, ... (never overwrites)."""
    d = os.path.join(run_dir, "redo")
    n = 1
    while os.path.exists(d):
        n += 1
        d = os.path.join(run_dir, f"redo_{n}")
    return d


def read_run_report(run_dir, base):
    path = os.path.join(run_dir, f"{base}_report.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"run folder {run_dir} has no {base}_report.json (the report of the run to redo)")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh), os.path.abspath(path)


def run_defaults(run_report):
    """(site, Step1Params, DTParams, map_units) recorded in a run report -- ORMIR-BQRL's schema or
    ipldt.ormir.run_pipeline's (an ipldt-pipeline folder can be redone too)."""
    P = run_report.get("parameters", {})
    dt_names = engine.DTParams().kwargs().keys()
    if run_report.get("schema", "").startswith("ormir-bqrl/"):
        site = run_report.get("site", "tibia")
        s1 = Step1Params(**P["step1"])
        dt = engine.DTParams(**{k: P["dt"][k] for k in dt_names if k in P["dt"]})
        units = P.get("map_units", "voxels")
    else:
        site = P.get("site", "tibia")
        s1p = (run_report.get("step1") or {}).get("params")
        s1 = Step1Params(**s1p) if s1p else engine.step1_params_for(site)
        dt = engine.DTParams(**{k: P[k] for k in dt_names if k in P})
        units = P.get("map_units", "voxels")
    return site, s1, dt, units


def run_parameters(run_report):
    """(Parameters, preset) a run report records: its parameter_set (reports since 2026-09-26, both workflows), read as
    its preset plus the values that differ from that preset's and workflow's defaults (ipldt.params.read_params), or,
    for an older report, the defaults with the recorded site, Step1Params and DTParams (what those runs used).  So a
    set recorded by ipldt-pipeline whose seg.periosteal_mask is that workflow's own default comes back with None there,
    and the redo uses ORMIR-BQRL's default (the rendered contour), as a redo of such a folder always did."""
    site, s1, dt, _ = run_defaults(run_report)
    ps = run_report.get("parameter_set")
    if isinstance(ps, dict) and isinstance(ps.get("values"), dict):
        pf = _pm.read_params(ps)
        return _pm.Parameters.defaults(pf.preset).override(pf.overrides), pf.preset
    preset = site if site in _pm.SITES else "tibia"
    return _pm.Parameters.defaults(preset).replace(step1=s1, dt=dt), preset


def _apply_parameters(P, parameters, sources):
    """P with a redo's `parameters` (None, a mapping, a JSON file path or a complete Parameters) applied.  A complete
    set (file or mapping) contributes the values that differ from its own preset's defaults (ipldt.params.read_params);
    its preset does not replace the run's site (a redo takes `site` for that)."""
    if parameters is None:
        return P
    if isinstance(parameters, _pm.Parameters):
        sources[:] = ["a complete Parameters object"]
        return parameters
    if isinstance(parameters, (str, os.PathLike)):
        pf = _pm.read_params_file(parameters)
        sources.append(pf.describe())
        return P.override(pf.overrides)
    pf = _pm.read_params(parameters)
    if pf.overrides or pf.complete:                  # a command line's overrides say where they came from
        sources.append(getattr(parameters, "source", None)
                       or (pf.describe() if pf.complete else f"{len(pf.overrides)} override(s)"))
    return P.override(pf.overrides)


def run_map_format(run_report):
    """The output format recorded in a run report ('nifti' when the report predates the option); ORMIR-BQRL and
    ipldt.ormir.run_pipeline both keep it as parameters.map_format."""
    fmt = (run_report.get("parameters") or {}).get("map_format") or "nifti"
    return "nifti" if fmt == "nii" else fmt


def run_volume_path(run_dir, base, suffix):
    """<run_dir>/<base>_<suffix>.nii.gz, else the .AIM of an AIM-format run; None when neither exists."""
    for ext in (".nii.gz", ".AIM", ".aim"):
        path = os.path.join(run_dir, f"{base}_{suffix}{ext}")
        if os.path.exists(path):
            return path
    return None


def read_run_masks(run_dir, base, grid, log=None):
    """The run's PRX_GOBJ (= ALL), CORT_MASK, TRAB_MASK by global position (NIfTI, or AIM for an AIM-format run);
    refused when PRX_GOBJ != CORT_MASK | TRAB_MASK (the folder is not an ORMIR-BQRL / ipldt-pipeline run or was
    tampered with)."""
    log = log if log is not None else engine.Logger(echo=False)
    out, paths = {}, {}
    for key, suffix in RUN_FILES.items():
        path = run_volume_path(run_dir, base, suffix)
        if path is None:
            raise FileNotFoundError(f"run folder {run_dir} has no {base}_{suffix}.nii.gz (nor {base}_{suffix}.AIM)")
        m, info = slicer.read_mask(path, grid, select=None)
        out[key] = stages.as_bool(m, grid, suffix)
        paths[key] = os.path.abspath(path)
        log(f"  run {suffix}: {int(out[key].sum()):,d} voxels ({path})")
    if (out["cort"] & out["trab"]).any() or not np.array_equal(out["ALL"], out["cort"] | out["trab"]):
        raise ValueError(f"run folder {run_dir} is inconsistent: {base}_PRX_GOBJ.nii.gz != CORT_MASK | TRAB_MASK "
                         "(or the two compartments overlap); redo needs an untouched run folder")
    return Masks(ALL=out["ALL"], cort=out["cort"], trab=out["trab"], prx_raw=None,
                 provenance={n: Prov("run", path=paths[k]) for n, k in
                             (("periosteal", "ALL"), ("cortical", "cort"), ("trabecular", "trab"))}), paths


def _read_edit(value, grid, select, name, log):
    """A given edit: a path (read by ormir_bqrl.slicer.read_mask with the compartment selected) or an array."""
    if isinstance(value, (str, os.PathLike)):
        path = os.path.abspath(os.fspath(value))
        m, info = slicer.read_mask(path, grid, select=select)
        m = stages.as_bool(m, grid, f"{name} mask")
        log(f"  edited {name}: {int(m.sum()):,d} voxels from {path} (labels {info.get('labels_used')}; dropped outside the "
            f"grid: {int(info.get('dropped_outside_grid') or 0)})")
        return m, path, dict(info)
    m = stages.as_bool(value, grid, f"{name} mask")
    log(f"  edited {name}: {int(m.sum()):,d} voxels given as an array")
    return m, None, {}


# ============================================================================================ entry
def run_from_masks(aim_path, run_dir, out_dir=None, periosteal=None, trab=None, cort=None, site=None, compute_bmd=True,
                   backend="auto", map_units=None, dt_params=None, preview=True, log=None, command=None,
                   compute_porosity=True, map_format=None, parameters=None):
    """Re-enter the workflow from corrected masks (the table in the module docstring).

    aim_path      the run's AIM
    run_dir       the folder of the automatic run (or of a previous redo): unedited inputs are read from it
                  (<base>_PRX_GOBJ / _CORT_MASK / _TRAB_MASK, .nii.gz or .AIM); site, dt parameters, map units and
                  the output format default to the values in its <base>_report.json (explicit arguments override
                  and are recorded)
    out_dir       default <run_dir>/redo, then redo_2, ... (never overwrites)
    periosteal, trab, cort   the edited masks: a file path (binary mask, ORMIR-BQRL labelmap, .seg.nrrd -- the
                  named compartment is extracted -- or an .AIM) or a bool array on the AIM grid
    compute_porosity  rerun the cortical pore cascade on the redo's compartments (Ct.Po, <base>_PORE); default on
    map_format    'nifti' or 'aim' (default: the run's)
    parameters    overrides of the run's parameter set (a mapping, a JSON file or a complete ipldt.params.Parameters);
                  the redo starts from the run's recorded set (module docstring, PARAMETERS)
    Returns the report dict (kind 'redo', edits block, provenance with the manual masks marked)."""
    log = log if log is not None else engine.Logger()
    aim_path = os.path.abspath(aim_path)
    run_dir = os.path.abspath(run_dir)
    base = os.path.splitext(os.path.basename(aim_path))[0]
    if periosteal is None and trab is None and cort is None:
        raise ValueError("nothing to redo: pass the edited mask(s) as periosteal=, trab= or cort=")
    if trab is not None and cort is not None:
        raise ValueError("the periosteal minus one compartment defines the other (Script 34); pass only the compartment you edited")
    run_report, run_report_path = read_run_report(run_dir, base)
    site_run, s1_run, dt_run, units_run = run_defaults(run_report)
    P_run, preset_run = run_parameters(run_report)
    P_run = P_run.for_workflow("ormir_bqrl")
    site_name = str(site).lower() if site is not None else site_run
    preset = preset_run
    sources = [f"the run's parameter set ({run_report_path})"]
    P = P_run
    if site is not None:
        P = P.replace(step1=engine.step1_params_for(site_name))
        preset = site_name
        sources.append(f"site preset {site_name!r}")
    P = _apply_parameters(P, parameters, sources)
    if dt_params is not None:
        P = P.replace(dt=dt_params)
        sources.append("dt_params")
    P = P.for_workflow("ormir_bqrl")
    if P.step1 != (engine.step1_params_for(site_name) if site_name in _pm.SITES else P_run.step1):
        site_name = _pm.site_of(P.step1)
    step1_params, dt_params = P.step1, P.dt
    R = _pm.Resolved(params=P, site=site_name, preset=site_name if site_name in _pm.SITES else preset,
                     workflow="ormir_bqrl", sources=tuple(sources))
    map_units = map_units or units_run
    map_format = map_format or run_map_format(run_report)
    stages.check_output_format(map_format, map_units)
    out_dir = os.path.abspath(out_dir) if out_dir else next_redo_dir(run_dir)
    log(f"{PRODUCT} {__version__} redo of {run_dir} -> {out_dir}")

    started = datetime.now().astimezone().isoformat(timespec="seconds")
    t_start = time.time()
    timing = {}
    t = time.time()
    loaded = stages.load(aim_path, log)
    grid = loaded.grid
    stages.check_output_format(map_format, map_units, loaded.native)
    timing["1_load"] = time.time() - t

    log("REDO - reading the run's masks and the edits ...")
    t = time.time()
    run_masks, run_paths = read_run_masks(run_dir, base, grid, log)
    edit = EditSpec()
    for name, value, select, attr in (("periosteal", periosteal, "periosteal", "periosteal"), ("trabecular", trab, "trab", "trab"),
                                      ("cortical", cort, "cort", "cort")):
        if value is None:
            continue
        m, path, info = _read_edit(value, grid, select, name, log)
        setattr(edit, attr, m)
        edit.paths[name] = path
        edit.infos[name] = info
    timing["2_read_masks"] = time.time() - t
    edits = {"edited": list(edit.edited), "rule": edit.rule, "files": dict(edit.paths),
             "outside_periosteal_raw": 0, "rendering_clipped": 0, "vs_run": None}

    # ---- P only: Script 32 / 33 STEP 1 reruns from the corrected contour (through run())
    if edit.trab is None and edit.cort is None:
        if not edit.periosteal.any():
            raise ValueError("the edited periosteal mask is empty")
        prov = edit.prov("periosteal", edit.periosteal.sum())
        ctx = RunContext(kind="redo", periosteal_prov=prov, derived_from=run_report_path, edits=edits, run_masks=run_masks,
                         log_name="redo", param_sources=tuple(sources), param_baseline=P_run)
        return _run_with_params(aim_path, out_dir, site_name, step1_params, edit.periosteal, compute_bmd, backend, map_units,
                                dt_params, preview, log, command, ctx, compute_porosity=compute_porosity, map_format=map_format,
                                parameters=P, preset=R.preset)

    # ---- compartment edit (Script 34 STEP 1), with or without a corrected periosteal: STEP 1 and the autocontour do
    # not run, so their values stay the run's (an override of them is not applied; `site` is one)
    R = _redo_context(R, P_run, site_run, preset_run, grid, compute_bmd, compute_porosity)
    os.makedirs(out_dir, exist_ok=True)
    return _evaluate(loaded, run_masks, edit, edits, out_dir, R.site, R.params.step1, R.params.dt, map_units, compute_bmd,
                     backend, preview, log, command, run_report_path, started, t_start, timing,
                     compute_porosity=compute_porosity, map_format=map_format, resolved=R,
                     seg_periosteal=_seg_periosteal(R.params, edit, run_dir, base, grid, log))


def _redo_context(R, P_run, site_run, preset_run, grid, compute_bmd, compute_porosity):
    """R in the context of a redo that does not run STEP 1 (a compartment redo, _from_run_masks): the steps that run
    and, for a value of a step that does not run, the run's value (Resolved.in_context with the run's set as the
    baseline); when that resets the step1 block, the site and preset are the run's again."""
    steps = {"masks", "hu_volume"} | ({"bmd"} if compute_bmd else set()) | ({"porosity"} if compute_porosity else set())
    out = R.in_context(steps, baseline=P_run, header_el=grid.el, baseline_label="the run's value")
    if any(n.startswith("step1.") for n, *_ in out.not_applied):
        out = _dc_replace(out, site=site_run, preset=site_run if site_run in _pm.SITES else preset_run)
    return out


def _seg_periosteal(P, edit, run_dir, base, grid, log):
    """The raster seg.periosteal_mask 'raw' masks the SEG with in a compartment redo: the edited periosteal, else the run
    folder's <base>_PRX_MASK (None when the parameter is 'rendered', or when the folder has none)."""
    if P.seg.periosteal_mask != "raw":
        return None
    if edit.periosteal is not None:
        return edit.periosteal
    path = run_volume_path(run_dir, base, "PRX_MASK")
    if path is None:
        return None
    m, _ = slicer.read_mask(path, grid, select=None)
    log(f"  seg.periosteal_mask 'raw': the SEG assembly is masked with the run's {os.path.basename(path)}")
    return stages.as_bool(m, grid, "PRX_MASK")


def _run_with_params(aim_path, out_dir, site_name, step1_params, P, compute_bmd, backend, map_units, dt_params, preview, log,
                     command, ctx, compute_porosity=True, map_format="nifti", parameters=None, preset=None):
    """run() with the Step 1 parameters of the redo: the site's preset when they are one (site name kept), else the
    run's recorded custom Step1Params (reported as site 'custom'); `parameters` is the redo's complete parameter set
    and `preset` the preset its non-default list is taken against (the run's, or the redo's explicit site)."""
    site_preset = engine.SITE_PARAMS.get(site_name)
    common = dict(periosteal=P, compute_bmd=compute_bmd, backend=backend, map_units=map_units, dt_params=dt_params, preview=preview,
                  log=log, command=command, compute_porosity=compute_porosity, map_format=map_format, _ctx=ctx,
                  parameters=parameters)
    if site_preset is not None and step1_params == site_preset:
        return run(aim_path, out_dir, site=site_name, **common)
    return run(aim_path, out_dir, site=preset if preset in _pm.SITES else "tibia", step1_params=step1_params, **common)


def _evaluate(loaded, run_masks, edit, edits, out_dir, site_name, step1_params, dt_params, map_units, compute_bmd, backend,
              preview, log, command, derived_from, started, t_start, timing, kind="redo", log_name="redo",
              compute_porosity=True, map_format="nifti", resolved=None, seg_periosteal=None):
    """The compartment path shared by every redo that does not rerun step 1 (and by _from_run_masks).  resolved: the
    redo's ipldt.params.Resolved (None = the defaults with step1_params / dt_params); seg_periosteal: the raster of
    seg.periosteal_mask 'raw'."""
    grid, base = loaded.grid, loaded.base
    R = resolved or _pm.resolve(site_name if site_name in _pm.SITES else "tibia", None, dt_params=dt_params,
                                step1_params=None if site_name in _pm.SITES else step1_params, workflow="ormir_bqrl")
    if R.steps is None:                                           # not yet in context: STEP 1 and the autocontour do not run
        R = R.in_context({"masks", "hu_volume"} | ({"bmd"} if compute_bmd else set())
                         | ({"porosity"} if compute_porosity else set()), header_el=grid.el)
    P = R.params
    mv = P.render.min_vertices
    if not R.is_default:
        log(f"PARAMETERS: {R.statement}")
    for note in P.unverified():
        log(f"  WARNING: {note}")
    for note in R.not_applied_notes():
        log(f"  WARNING: {note}")
    R.warn_not_applied()
    t = time.time()
    if edit.periosteal is not None:
        if not edit.periosteal.any():
            raise ValueError("the edited periosteal mask is empty")
        ALL = stages.render(edit.periosteal, grid, mv)            # R(P): exactly what step 3 would have done
        prx_raw = edit.periosteal
        prov_p = edit.prov("periosteal", prx_raw.sum(), ALL.sum())
        log(f"  periosteal edited: raw {int(prx_raw.sum()):,d} -> rendered contour ALL {int(ALL.sum()):,d} voxels")
    else:
        ALL = run_masks.ALL
        prx_raw = None
        prov_p = Prov("run", path=run_masks.provenance["periosteal"].path, voxels=int(ALL.sum()), rendered_voxels=int(ALL.sum()))
    if edit.trab is None and edit.cort is None:                  # _from_run_masks: the run's own compartments
        cort, trab, G_cort, G_trab = run_masks.cort, run_masks.trab, None, None
        prov_c = Prov("run", path=run_masks.provenance["cortical"].path, voxels=int(cort.sum()))
        prov_t = Prov("run", path=run_masks.provenance["trabecular"].path, voxels=int(trab.sum()))
    else:
        try:
            d = derive_compartments(ALL, grid, trab=edit.trab, cort=edit.cort, min_vertices=mv)
        except EditOutsidePeriosteal as exc:
            p = os.path.join(out_dir, f"{base}_redo_outside.nii.gz")
            write_nifti(p, exc.outside.astype(np.uint8) * int(P.output.mask_value), grid.el, grid.pos)
            log(f"  REFUSED: {exc}; the offending voxels are in {p}")
            edits["outside_periosteal_raw"] = exc.count
            raise
        cort, trab, G_cort, G_trab = d["cort"], d["trab"], d["G_cort"], d["G_trab"]
        edits["rendering_clipped"] = d["rendering_clipped"]
        notes = [f"{d['rendering_clipped']} rendered voxels outside the periosteal clipped"] if d["rendering_clipped"] else []
        derived = f"derived: periosteal - {d['edited']}"
        if d["edited"] == "trabecular":
            prov_t = edit.prov("trabecular", d["raw_voxels"], d["rendered_voxels"], notes)
            prov_c = Prov(derived, voxels=int(cort.sum()))
        else:
            prov_c = edit.prov("cortical", d["raw_voxels"], d["rendered_voxels"], notes)
            prov_t = Prov(derived, voxels=int(trab.sum()))
        log(f"  {d['edited']} edited: raw {d['raw_voxels']:,d} -> rendered {d['rendered_voxels']:,d} voxels "
            f"({d['rendering_clipped']} clipped to the periosteal); {d['derived']} = periosteal - {d['edited']}: "
            f"{int((cort if d['derived'] == 'cortical' else trab).sum()):,d} voxels")
    masks = Masks(ALL=ALL, cort=cort, trab=trab, prx_raw=prx_raw,
                  provenance={"periosteal": prov_p, "cortical": prov_c, "trabecular": prov_t})
    timing["3_compartments"] = time.time() - t

    t = time.time()
    segmentation, masks.G_cort, masks.G_trab = stages.segment(loaded, ALL, cort, trab, G_cort=G_cort, G_trab=G_trab, log=log,
                                                              parameters=P,
                                                              prx_raw=seg_periosteal if seg_periosteal is not None else prx_raw)
    timing["4_render_segment"] = time.time() - t
    prov_c.rendered_voxels = int(masks.G_cort.sum())
    prov_t.rendered_voxels = int(masks.G_trab.sum())

    t = time.time()
    morph = stages.morphometry(loaded, segmentation.seg, masks.G_trab, cort, masks.G_cort, P.dt, backend, log,
                               parameters=P, trab_seg=segmentation.trab_seg, ALL=ALL)
    timing["5_dt"] = time.time() - t
    bmd = {}
    if compute_bmd:
        t = time.time()
        bmd = stages.bmd(loaded, masks.G_trab, masks.G_cort, log, parameters=P, trab=trab, cort=cort)
        timing["5b_bmd"] = time.time() - t
    poro = stages.Porosity(pore=None, metrics={})
    if compute_porosity:                                          # the pore cascade reruns on this redo's compartments
        t = time.time()
        poro = stages.porosity(loaded, masks.G_cort, segmentation.cort_seg, log, parameters=P)
        timing["5c_porosity"] = time.time() - t

    t = time.time()
    outputs = stages.write_volumes(out_dir, loaded, masks, segmentation, morph, map_units, log, pore=poro.pore,
                                   map_format=map_format, mask_value=P.output.mask_value,
                                   voxel_size_mm=P.morphometry.voxel_size_mm, calibration=P.calibration)
    outputs["parameters"] = _pm.write_params_file(os.path.join(out_dir, f"{base}_parameters.json"), R, sample=base)
    outputs["log"] = os.path.join(out_dir, f"{base}_{log_name}.log")
    outputs["preview"] = None
    outputs["edit_preview"] = None
    timing["6_write"] = time.time() - t
    if edits is not None:
        edits["vs_run"] = mask_changes(run_masks, masks)
    timing["total"] = time.time() - t_start
    report = report_mod.build_report(kind=kind, loaded=loaded, masks=masks, segmentation=segmentation, morph=morph, bmd=bmd,
                                     site=site_name, step1_params=P.step1, step1_info=None, dt_params=P.dt,
                                     map_units=map_units, compute_bmd=compute_bmd, outputs=outputs, timing=timing,
                                     started=started, duration_s=timing["total"], command=command, out_dir=out_dir,
                                     derived_from=derived_from, edits=edits,
                                     calibration_source=engine.step1_calibration(loaded.native, loaded.calib, P.calibration)[1],
                                     porosity=poro, map_format=map_format, resolved=R)
    if preview:
        try:
            from .preview import write_edit_preview, write_preview
            p = os.path.join(out_dir, f"{base}_preview.png")
            report["outputs"]["preview"] = write_preview(p, loaded, masks, segmentation, morph, report)
            if edits is not None and edits.get("edited"):
                p = os.path.join(out_dir, f"{base}_redo_edits.png")
                report["outputs"]["edit_preview"] = write_edit_preview(p, run_masks, masks, edits["edited"][-1])
        except ImportError as exc:
            log(f"  preview skipped: {exc}")
    report["timing_s"]["total"] = time.time() - t_start
    report["run"]["duration_s"] = report["timing_s"]["total"]
    report_mod.write_reports(report, out_dir, base)
    log(done_line(base, kind, report))
    if isinstance(log, engine.Logger):
        with open(outputs["log"], "w", encoding="utf-8") as fh:
            fh.write("\n".join(log.lines) + "\n")
    return report


def _from_run_masks(aim_path, run_dir, out_dir=None, compute_bmd=True, backend="auto", map_units=None, dt_params=None,
                    preview=True, log=None, compute_porosity=True, map_format=None, parameters=None):
    """Internal: the run's own masks, no edits -- reproduces the run bit for bit (SEG, the four maps, the pore
    map, every metric), under whatever parameter set the run recorded; the `unchanged redo == run` invariant the
    tests assert.  Not reachable from run_from_masks, which refuses a call without edits."""
    log = log if log is not None else engine.Logger()
    aim_path = os.path.abspath(aim_path)
    run_dir = os.path.abspath(run_dir)
    base = os.path.splitext(os.path.basename(aim_path))[0]
    run_report, run_report_path = read_run_report(run_dir, base)
    site_name, step1_params, dt_run, units_run = run_defaults(run_report)
    P_run, preset = run_parameters(run_report)
    P_run = P_run.for_workflow("ormir_bqrl")
    sources = [f"the run's parameter set ({run_report_path})"]
    P = _apply_parameters(P_run, parameters, sources)
    if dt_params is not None:
        P = P.replace(dt=dt_params)
        sources.append("dt_params")
    P = P.for_workflow("ormir_bqrl")
    R = _pm.Resolved(params=P, site=site_name, preset=site_name if site_name in _pm.SITES else preset, workflow="ormir_bqrl",
                     sources=tuple(sources))
    map_units = map_units or units_run
    map_format = map_format or run_map_format(run_report)
    stages.check_output_format(map_format, map_units)
    out_dir = os.path.abspath(out_dir) if out_dir else next_redo_dir(run_dir)
    os.makedirs(out_dir, exist_ok=True)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    t_start = time.time()
    timing = {}
    t = time.time()
    loaded = stages.load(aim_path, log)
    timing["1_load"] = time.time() - t
    t = time.time()
    run_masks, _ = read_run_masks(run_dir, base, loaded.grid, log)
    timing["2_read_masks"] = time.time() - t
    R = _redo_context(R, P_run, site_name, preset, loaded.grid, compute_bmd, compute_porosity)
    return _evaluate(loaded, run_masks, EditSpec(), None, out_dir, R.site, R.params.step1, R.params.dt,
                     map_units, compute_bmd, backend, preview, log, None, run_report_path, started, t_start, timing,
                     compute_porosity=compute_porosity, map_format=map_format, resolved=R,
                     seg_periosteal=_seg_periosteal(R.params, EditSpec(), run_dir, base, loaded.grid, log))


__all__ = ["run_from_masks", "derive_compartments", "EditSpec", "EditOutsidePeriosteal", "RULES", "next_redo_dir",
           "read_run_report", "run_defaults", "run_parameters", "run_map_format", "run_volume_path", "read_run_masks"]
