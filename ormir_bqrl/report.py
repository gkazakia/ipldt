"""ormir_bqrl.report -- the ORMIR-BQRL report (schema "ormir-bqrl/1"): one dict, written as
<base>_report.json (nested), <base>_report.csv (Parameter,Value; ipldt.ormir.write_report's flattening) and
<base>_report.md (headline table, provenance with the manual masks marked, the edit block of a redo, the
parameter set, the file table, timings).  summary_row() is the one-line dict `batch` collects into
batch_summary.csv.  Every value is JSON-native (numpy scalars converted) so the returned dict equals the file.

The metrics sit in `morphometry` (ipldt.ormir.ipl_morphometry), `bmd` (ORMIR-XCT bmd_masked) and `porosity`
(Ct_Po, Ct_Po_pore_voxels, Ct_Po_compartment_voxels: the keys of ipldt.ormir.run_pipeline's porosity block; {}
when the pore cascade was not run), and the headline values of all three in `summary`.

`parameters` lists the values the run actually used (read from its parameter set, never from the engine's
constants; its `step1` block is Step1Params.record(), the sixteen preset fields plus any script literal that differs
from the script's value, so a default report has the keys it always had), and `parameter_set`
(ipldt.params.Resolved.block) records the COMPLETE effective parameter set, the names that differ from the validated
IPL defaults (`non_default`, empty for a default run), a one-line `statement` that the Markdown report repeats at its
top, the values that did not take effect because their step did not run (`not_applied`), the values IPL's exports
never confirmed (`unverified`) and where the values came from (`sources`).  A calibration override adds
input_aim.calibration_effective (the calibration actually used) next to the header's.
"""
from __future__ import annotations

import hashlib
import os
import platform
import sys
from dataclasses import asdict, is_dataclass

import numpy as np

from ipldt import __version__ as ipldt_version
from ipldt import ormir as engine
from ipldt import params as _pm

from . import PRODUCT, __version__

SCHEMA = "ormir-bqrl/1"

SUMMARY_KEYS = ("BV_TV", "Tb_Th_mm", "Tb_Th_sd_mm", "Tb_Sp_mm", "Tb_Sp_sd_mm", "Tb_N_per_mm", "Ct_Th_mm", "Ct_Th_sd_mm")
BMD_KEYS = ("Tb_BMD_mgHA_cm3", "Ct_BMD_mgHA_cm3")
POROSITY_KEYS = ("Ct_Po",)
MASK_ORDER = ("periosteal", "cortical", "trabecular")


# ============================================================================================ helpers
def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def plain(obj):
    """A JSON-native copy: numpy scalars / arrays, tuples, dataclasses, paths -> int / float / list / dict / str."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return plain(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    return obj


def _version_of(module_name):
    try:
        mod = __import__(module_name)
    except Exception:
        return None
    v = getattr(mod, "__version__", None)
    if v is None:
        try:
            from importlib.metadata import version
            v = version(module_name.replace("_", "-"))
        except Exception:
            v = None
    return None if v is None else str(v)


def product_info():
    """The product block: ORMIR-BQRL, ipldt, ORMIR-XCT, SimpleITK, itk, CuPy (None when absent), Python, platform."""
    return {"name": PRODUCT, "version": __version__, "ipldt_version": ipldt_version,
            "ormir_xct_version": _version_of("ormir_xct"), "simpleitk": _version_of("SimpleITK"),
            "itk": _version_of("itk"), "cupy": _version_of("cupy"),
            "python": sys.version.split()[0], "platform": platform.platform()}


def step1_block(site_name, info):
    """run_pipeline's step-1 block from step3_trab_cort_seg's info dict (None when step 1 did not run)."""
    if info is None:
        return None
    return {"site": site_name, "params": info["params"], "calibration": info["calibration"],
            "calibration_source": info["calibration_source"], "thresholds": info["thresholds"],
            "periosteal": info["periosteal"], "seg_gauss_box": info["box"],
            "counts": {k: v for k, v in info["counts"].items() if k != "peel"},
            "peel_counts": {str(k): v for k, v in info["counts"]["peel"].items()},
            "grids": {k: {"dim_xyz": list(g[0]), "pos_xyz": list(g[1])} for k, g in info["grids"].items()},
            "timings_s": info["timings"]}


def summary_of(metrics, bmd, porosity=None):
    s = {k: metrics.get(k) for k in SUMMARY_KEYS}
    for k in BMD_KEYS:
        s[k] = (bmd or {}).get(k)
    for k in POROSITY_KEYS:
        s[k] = (porosity or {}).get(k)
    return s


def porosity_parameters(poro, pp=None):
    """The parameters block of stage 5c: the ipldt calls, the arguments the cascade ran with (pp, an
    ipldt.params.PoreParams; None = Script 32's) and the contour grid."""
    from ipldt import porosity as _por
    pp = pp or _pm.PoreParams()
    m = int(pp.render_grid_margin)
    clip = "clipped at 0" if pp.render_grid_clip_low == 0 else (
        "unclipped" if pp.render_grid_clip_low is None else f"clipped at {pp.render_grid_clip_low}")
    rule = (f"ipldt.porosity.pore_cascade_ipl_grid: the contour on IPL's /gobj_to_aim grid (its box grown by "
            f"{m} low and {m} or {m + 1} high per in-plane axis, {clip}), CORT_SEG on its tight box; the map pasted back "
            f"onto the AIM grid") if pp.grid == "ipl" else "ipldt.porosity.pore_cascade on the input AIM's grid (porosity.grid 'aim')"
    ct = ("ipldt.porosity.ct_po: |PORE & cortical contour| / |cortical contour|" if pp.ct_po == "contour" else
          "ipldt.porosity.ct_po(definition='pore_plus_bone'): |PORE| / (|PORE| + |CORT_SEG|)")
    hys = dict(_por.SCRIPT32_HYSTERESIS)
    hys.update(low_thresh=pp.low_thresh, high_thresh=pp.high_thresh, mode=pp.mode, grow_axes=tuple(pp.grow_axes))
    return {"method": "ipldt.porosity.pore_cascade", "ct_po": ct,
            "contour": "rendered cortical contour (CORT_GOBJ) and CORT_SEG (ipldt.ormir.step5c_porosity = run_pipeline STEP 5c)",
            "grid_rule": rule,
            "grids": None if poro is None else poro.grids,
            "hysteresis": hys, "slice_fraction_percent": [pp.slice_lo, pp.slice_up],
            "min_pore_voxels": pp.min_pore_voxels, "computed": bool(poro is not None and poro.computed),
            "error": None if poro is None else poro.error}


# ============================================================================================ build
def build_report(*, kind, loaded, masks, segmentation, morph, bmd, site, step1_params, step1_info, dt_params,
                 map_units, compute_bmd, outputs, timing, started, duration_s, command=None, out_dir=None,
                 derived_from=None, edits=None, calibration_source=None, porosity=None, map_format="nifti",
                 resolved=None):
    """The report dict (schema ormir-bqrl/1; ormir_bqrl/README.md, "Outputs").  `masks.provenance` carries a Prov per mask;
    `edits` is None for a run and the edit block of a redo; `porosity` is stage 5c's stages.Porosity (None when
    the pore cascade was not run); `resolved` is the run's ipldt.params.Resolved (the complete parameter set; None =
    the defaults of `site` with step1_params / dt_params)."""
    grid = loaded.grid
    R = resolved or _pm.resolve(site if site in _pm.SITES else "tibia", None,
                                step1_params=None if site in _pm.SITES and step1_params == _pm.SITES[site] else step1_params,
                                dt_params=dt_params, workflow="ormir_bqrl")
    P = R.params
    vs = float(P.morphometry.voxel_size_mm) if P.morphometry.voxel_size_mm is not None else float(grid.el[0])
    metrics = dict(morph.metrics)
    poro_metrics = dict(porosity.metrics) if porosity is not None else {}
    prov = {name: masks.provenance[name].as_dict() for name in MASK_ORDER}
    for name in MASK_ORDER:
        prov[name]["grid"] = {"dim_xyz": list(grid.dim), "pos_xyz": list(grid.pos)}
    prov["periosteal"]["raw_voxels"] = None if masks.prx_raw is None else int(masks.prx_raw.sum())
    prov["periosteal"]["rendered_voxels"] = int(masks.ALL.sum())
    prov["cortical"]["voxels"] = int(masks.cort.sum())
    prov["cortical"]["rendered_voxels"] = int(masks.G_cort.sum())
    prov["trabecular"]["voxels"] = int(masks.trab.sum())
    prov["trabecular"]["rendered_voxels"] = int(masks.G_trab.sum())
    cal_src = calibration_source or (step1_info["calibration_source"] if step1_info else None)
    cal_eff = engine.calibration_with(loaded.calib, P.calibration)
    report = {
        "schema": SCHEMA,
        "product": product_info(),
        "run": {"kind": kind, "started": started, "duration_s": float(duration_s),
                "command": list(command) if command else None,
                "out_dir": os.path.abspath(out_dir) if out_dir else None,
                "derived_from": os.path.abspath(derived_from) if derived_from else None},
        "sample": loaded.base,
        "input_aim": {"path": os.path.abspath(loaded.aim_path), "sha256": sha256_of(loaded.aim_path),
                      "dim_xyz": list(grid.dim), "pos_xyz": list(grid.pos), "el_size_mm": list(grid.el),
                      "calibration": dict(loaded.calib), "calibration_source": cal_src},
        "site": site,
        "parameters": {
            "step1": step1_params.record(),
            "laplace_hamming": {"laplace_eps": P.lh.laplace_eps, "lp_cut_off_freq": P.lh.lp_cut_off_freq,
                                "hamming_amp": P.lh.hamming_amp, "norm_max": P.lh.norm_max,
                                "threshold": P.lh.threshold, "el_size_mm": list(segmentation.lh_el_size_mm),
                                "pad_offset": getattr(segmentation, "lh_pad_offset", P.lh.pad_offset)},
            "seg_assembly": {"cl_nr_extract_min_cort": P.seg.cc_min_cort,
                             "cl_nr_extract_min_trab": P.seg.cc_min_trab,
                             "seg_values": {"cort": P.seg.value_cort, "trab": P.seg.value_trab},
                             "periosteal_mask": ("rendered contour (ALL)" if P.seg.periosteal_mask == "rendered"
                                                 else "raw periosteal raster")},
            "dt": {**dt_params.kwargs(), "backend": morph.backend, "backend_requested": morph.backend_requested,
                   "voxel_size_mm": vs},
            "bmd": {"method": "ormir_xct.bmd_masked", "units": "mgHA/cm3", "computed": bool(bmd)},
            "porosity": porosity_parameters(porosity, P.porosity),
            "map_units": map_units,
            "map_format": map_format},
        "parameter_set": R.block(),
        "masks": prov,
        "edits": edits,
        "grids": {"AIM": {"dim_xyz": list(grid.dim), "pos_xyz": list(grid.pos)},
                  **{k: {"dim_xyz": list(g.dim), "pos_xyz": list(g.pos)} for k, g in morph.grids.items()}},
        "step1": step1_block(site, step1_info),
        "compartments": {**masks.counts(), **segmentation.counts()},
        "morphometry": metrics,
        "bmd": dict(bmd or {}),
        "porosity": poro_metrics,
        "summary": summary_of(metrics, bmd, poro_metrics),
        "outputs": dict(outputs),
        "timing_s": {**dict(timing), **{f"dt_{k}": v for k, v in morph.timing_s.items() if k != "total"}},
    }
    if not compute_bmd:
        report["parameters"]["bmd"]["computed"] = False
    if cal_eff is not loaded.calib:                  # a calibration override: the calibration actually used
        report["input_aim"]["calibration_effective"] = dict(cal_eff)
    return plain(report)


# ============================================================================================ write
def write_reports(report, out_dir, base):
    """<base>_report.json / .csv (ipldt.ormir.write_report) and .md; the paths are added to report['outputs']
    before the JSON is written so the file lists itself.  Returns {json, csv, md}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {k: os.path.abspath(os.path.join(out_dir, f"{base}_report.{ext}")) for k, ext in
             (("json", "json"), ("csv", "csv"), ("md", "md"))}
    report.setdefault("outputs", {})
    report["outputs"]["report_json"] = paths["json"]
    report["outputs"]["report_csv"] = paths["csv"]
    report["outputs"]["report_md"] = paths["md"]
    engine.write_report(report, paths["json"], paths["csv"])
    with open(paths["md"], "w", encoding="utf-8") as fh:
        fh.write(render_markdown(report))
    return paths


def _fmt(v, nd=6):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return f"{v:,d}"
    return str(v)


def render_markdown(report):
    r = report
    run, s, prov = r["run"], r["summary"], r["masks"]
    L = [f"# {r['product']['name']} report: {r['sample']}", ""]
    L.append(f"- kind: **{run['kind']}**; site: {r['site']}; started {run['started']}; duration {run['duration_s']:.1f} s")
    L.append(f"- input AIM: `{r['input_aim']['path']}` (dim {r['input_aim']['dim_xyz']}, pos {r['input_aim']['pos_xyz']}, "
             f"el {['%.7f' % e for e in r['input_aim']['el_size_mm']]} mm)")
    if run.get("out_dir"):
        L.append(f"- output folder: `{run['out_dir']}`")
    if run.get("derived_from"):
        L.append(f"- derived from: `{run['derived_from']}`")
    L.append(f"- {r['product']['name']} {r['product']['version']}, ipldt {r['product']['ipldt_version']}, "
             f"ORMIR-XCT {r['product']['ormir_xct_version']}, dt backend {r['parameters']['dt']['backend']}")
    ps = r.get("parameter_set") or {}
    if ps:
        if ps.get("non_default"):
            L.append(f"- **parameters: NOT the validated IPL defaults.** This configuration {ps['statement']} "
                     "(the full set: 'Parameter set' below)")
        else:
            L.append(f"- parameters: {ps['statement']}")
        for note in ps.get("unverified") or []:
            L.append(f"- **unverified value:** {note}")
        for name, d in (ps.get("not_applied_values") or {}).items():
            L.append(f"- **not applied:** {name} = {_fmt_param(d.get('requested'))}: {d.get('reason')} "
                     f"({_fmt_param(d.get('used'))})")
    L += ["", "## Summary", "", "| Metric | Value | SD |", "|---|---|---|"]
    rows = (("BV/TV", "BV_TV", None, ""), ("Tb.Th", "Tb_Th_mm", "Tb_Th_sd_mm", "mm"), ("Tb.Sp", "Tb_Sp_mm", "Tb_Sp_sd_mm", "mm"),
            ("Tb.N", "Tb_N_per_mm", None, "/mm"), ("Ct.Th", "Ct_Th_mm", "Ct_Th_sd_mm", "mm"), ("Ct.Po", "Ct_Po", None, ""),
            ("Tb.BMD", "Tb_BMD_mgHA_cm3", None, "mgHA/cm3"), ("Ct.BMD", "Ct_BMD_mgHA_cm3", None, "mgHA/cm3"))
    for label, k, ksd, unit in rows:
        v = s.get(k)
        if v is None:
            continue
        sd = s.get(ksd) if ksd else None
        L.append(f"| {label} | {_fmt(v)} {unit} | {_fmt(sd) + ' ' + unit if sd is not None else ''} |")
    L += ["", "## Masks (provenance)", "", "| Mask | Source | Voxels | Rendered | File | sha256 | Notes |", "|---|---|---|---|---|---|---|"]
    for name in MASK_ORDER:
        p = prov[name]
        src = f"**manual**" if p.get("manual") else p["source"]
        vox = p.get("raw_voxels") if name == "periosteal" else p.get("voxels")
        sha = (p.get("sha256") or "")[:12]
        L.append(f"| {name} | {src} | {_fmt(vox)} | {_fmt(p.get('rendered_voxels'))} | {p.get('path') or ''} | {sha} | "
                 f"{'; '.join(p.get('notes') or [])} |")
    e = r.get("edits")
    if e:
        L += ["", "## Edits (redo)", "", f"- edited: {', '.join(e['edited'])}; rule: `{e['rule']}`",
              f"- raw voxels outside the periosteal: {_fmt(e.get('outside_periosteal_raw'))}; rendering voxels clipped to the periosteal: {_fmt(e.get('rendering_clipped'))}",
              "", "| Mask | added vs run | removed vs run |", "|---|---|---|"]
        for name in MASK_ORDER:
            d = (e.get("vs_run") or {}).get(name)
            if d:
                L.append(f"| {name} | {_fmt(d['added'])} | {_fmt(d['removed'])} |")
    P = r["parameters"]
    L += ["", "## Parameters", ""]
    L.append(f"- site `{r['site']}`; Step 1: " + ", ".join(f"{k}={v}" for k, v in P["step1"].items()))
    lh = P["laplace_hamming"]
    L.append(f"- Laplace-Hamming: eps {lh['laplace_eps']}, cut-off {lh['lp_cut_off_freq']}, amp {lh['hamming_amp']}, "
             f"norm_max {lh['norm_max']:g}, threshold {lh['threshold']}, el {['%.7f' % x for x in lh['el_size_mm']]} mm")
    sa = P["seg_assembly"]
    L.append(f"- SEG assembly: cl_nr_extract min {sa['cl_nr_extract_min_cort']} (cort) / {sa['cl_nr_extract_min_trab']} (trab), "
             f"values {sa['seg_values']}, masked with the {sa['periosteal_mask']}")
    dt = P["dt"]
    L.append(f"- dt: ridge_epsilon {dt['ridge_epsilon']}, assign_epsilon {dt['assign_epsilon']}, peel_iter {dt['peel_iter']}, "
             f"version {dt['version']}, suppress_boundary {dt['suppress_boundary']}, voxel size {dt['voxel_size_mm']:.7f} mm, "
             f"backend {dt['backend']}")
    L.append(f"- BMD: {P['bmd']['method']} ({P['bmd']['units']}), computed {P['bmd']['computed']}")
    pp = P.get("porosity")
    if pp:
        h = pp["hysteresis"]
        line = (f"- Ct.Po: cortical pore cascade ({pp['method']}; slice-wise {pp['slice_fraction_percent'][0]:g}-"
                f"{pp['slice_fraction_percent'][1]:g} %, hysteresis {h['low_thresh']} / {h['high_thresh']} along z, components "
                f">= {pp['min_pore_voxels']} voxels) in the rendered cortical contour, computed {pp['computed']}")
        if pp.get("error"):
            line += f" ({pp['error']})"
        L.append(line)
        if pp.get("grid_rule"):
            L.append(f"- Ct.Po grid: {pp['grid_rule']} (the grids: 'porosity' rows below)")
    L.append(f"- output format: {P.get('map_format', 'nifti')}; map units: {P['map_units']}")
    if ps.get("values"):
        nd = set(ps.get("non_default") or [])
        dv = ps.get("non_default_values") or {}
        L += ["", "## Parameter set", "",
              f"The complete effective parameter set (site {ps.get('site')}, compared with the {ps.get('preset')} preset; "
              f"sources: {'; '.join(ps.get('sources') or [])}).  Non-default values are in bold with the default "
              "after them.", "", "| Parameter | Value | Default |", "|---|---|---|"]
        for stage, block in ps["values"].items():
            for name, v in block.items():
                key = f"{stage}.{name}"
                if key in nd:
                    L.append(f"| **{key}** | **{_fmt_param(v)}** | {_fmt_param((dv.get(key) or {}).get('default'))} |")
                else:
                    L.append(f"| {key} | {_fmt_param(v)} | |")
    L += ["", "## Grids (x, y, z)", "", "| Grid | dim | pos |", "|---|---|---|"]
    for k, g in r["grids"].items():
        L.append(f"| {k} | {g['dim_xyz']} | {g['pos_xyz']} |")
    for k, g in ((pp or {}).get("grids") or {}).items():         # the pore cascade's grids (stage 5c)
        L.append(f"| porosity {k} | {g['dim_xyz']} | {g['pos_xyz']} |")
    c = r["compartments"]
    L += ["", "## Voxel counts", "", "| Volume | Voxels |", "|---|---|"]
    for k, v in c.items():
        L.append(f"| {k.replace('_voxels', '')} | {_fmt(v)} |")
    po = r.get("porosity") or {}
    if po:
        L.append(f"| PORE | {_fmt(po.get('Ct_Po_pore_voxels'))} |")
        L.append(f"| cortical contour (Ct.Po denominator) | {_fmt(po.get('Ct_Po_compartment_voxels'))} |")
    L += ["", "## Files", "", "| Output | Path |", "|---|---|"]
    for k, v in r["outputs"].items():
        L.append(f"| {k} | {v if v else '-'} |")
    L += ["", "## Timings (s)", "", "| Step | s |", "|---|---|"]
    for k, v in r["timing_s"].items():
        L.append(f"| {k} | {v:.1f} |" if isinstance(v, (int, float)) else f"| {k} | {v} |")
    L.append("")
    return "\n".join(L)


def _fmt_param(v):
    if isinstance(v, (list, tuple)):
        return " ".join(_fmt_param(x) for x in v)
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def summary_row(report):
    """The one-line dict of a sample for batch_summary.csv."""
    r = report
    row = {"sample": r["sample"], "site": r["site"], "kind": r["run"]["kind"]}
    row.update({k: r["summary"].get(k) for k in SUMMARY_KEYS + POROSITY_KEYS + BMD_KEYS})
    for name in MASK_ORDER:
        row[f"{name}_source"] = r["masks"][name]["source"]
    row["out_dir"] = r["run"].get("out_dir")
    nd = (r.get("parameter_set") or {}).get("non_default")
    if nd:                                           # only for a non-default run (`batch` writes it as the last column)
        row["non_default_parameters"] = " ".join(nd)
    return row


__all__ = ["SCHEMA", "SUMMARY_KEYS", "BMD_KEYS", "POROSITY_KEYS", "MASK_ORDER", "sha256_of", "plain", "product_info",
           "step1_block", "summary_of", "porosity_parameters", "build_report", "write_reports", "render_markdown", "summary_row"]
