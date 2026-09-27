"""ipldt.params -- the parameter model (every tunable value of every stage is a parameter; the defaults are the
validated IPL configuration).

  * the defaults ARE the engine's constants for both site presets (Script 32 tibia, Script 33 radius), and a default
    run through the parameter model is bit-identical to the call sequence without it;
  * every stage's override reaches the stage and changes its output on a synthetic phantom: STEP 1 (preset values and
    the new script literals), the renderer's minimum chain, Laplace-Hamming (eps, cut-off, amplitude, norm, the
    permille threshold, the border, the padding offset), the SEG assembly (component sizes, masks, labels, order), the
    dt parameters, the Tb.Th / Ct.Th / BV objects, the maps, the pore cascade (slice fraction, component bounds,
    hysteresis, grid, Ct.Po definition);
  * --set / --params parsing, type checking, precedence (site preset < file < --set) and the error messages;
  * report provenance (the complete set, the non-default list, the statement in the Markdown report) and a redo that
    keeps the run's parameters (the unchanged redo reproduces a NON-default run exactly).

The workflow tests use the pore phantom of test_ormir_bqrl_porosity (a v020 short AIM with a cortical shell, pores
and trabecular rods; a periosteal disc on which IPL's rendering is the identity; the autocontour is never called)."""
import json
import os
from dataclasses import asdict, fields, replace

import numpy as np
import pytest
from scipy import ndimage as ndi

from ipldt import ipl_ops as ops
from ipldt import ormir as engine
from ipldt import params as pm
from ipldt import porosity
from ipldt.contour import render as crender
from ipldt.contour import render_volume
from ipldt.params import (AutocontourParams, LHParams, MorphometryParams, ParameterError, Parameters, PoreParams,
                          SegParams)
from ipldt.step1 import RADIUS, TIBIA, Step1Params, cort_trab_separation

HERE = os.path.dirname(os.path.abspath(__file__))


# ============================================================================================ 1. the defaults
@pytest.mark.parametrize("site,preset", [("tibia", TIBIA), ("radius", RADIUS)])
def test_defaults_are_the_validated_constants(site, preset):
    P = Parameters.defaults(site)
    assert P.step1 == preset and P.step1 == engine.SITE_PARAMS[site]
    assert (P.lh.laplace_eps, P.lh.lp_cut_off_freq, P.lh.hamming_amp, P.lh.norm_max) == \
        (engine.LAPLACE_EPS, engine.LP_CUT_OFF_FREQ, engine.HAMMING_AMP, engine.NORM_MAX_VALUE)
    assert P.lh.threshold == engine.LH_THRESHOLD == 15564 and P.lh.upper_threshold == engine.INT16_MAX
    assert (P.lh.pad_offset, P.lh.dtype, P.lh.border, P.lh.el_size_mm) == (engine.LH_PAD_OFFSET, engine.LH_DTYPE, "duplicate", None)
    s = P.seg
    assert (s.cc_min_cort, s.cc_min_trab, s.cc_max_cort, s.cc_max_trab) == (engine.CC_MIN_VOXELS_CORT, engine.CC_MIN_VOXELS_TRAB, 0, 0)
    assert (s.value_cort, s.value_trab, s.overlap, s.order, s.trab_seg_mask) == \
        (engine.SEG_VALUE_CORT, engine.SEG_VALUE_TRAB, "trab", "periosteal_first", "gobj")
    assert (s.peel_periosteal, s.peel_cort, s.peel_trab, s.periosteal_mask) == (0, 0, 0, None)
    assert P.dt == engine.DTParams() == engine.IPL_SCRIPT32
    po, hys = P.porosity, porosity.SCRIPT32_HYSTERESIS
    assert (po.slice_lo, po.slice_up) == tuple(porosity.SLICE_FRACTION) and po.min_pore_voxels == porosity.MIN_PORE_VOXELS
    assert (po.low_thresh, po.high_thresh, po.mode, po.grow_axes) == (hys["low_thresh"], hys["high_thresh"], hys["mode"],
                                                                      tuple(hys["grow_axes"]))
    assert (po.render_grid_margin, po.render_grid_clip_low, po.grid, po.ct_po) == (porosity.RENDER_GRID_MARGIN, 0, "ipl", "contour")
    assert (po.max_pore_voxels, po.marrow_rank_first, po.marrow_rank_last, po.marrow_connect_boundary, po.gobj_peel) == (0, 1, 1, False, 0)
    assert P.render.min_vertices == crender.MIN_VERTICES == 4
    m = P.morphometry
    assert (m.tbth_object, m.tbth_grid, m.ctth_object, m.bvtv_object, m.maps, m.grid_border, m.voxel_size_mm) == \
        ("seg", "seg", "cort_mask", "seg", ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH"), 0, None)
    assert P.output.mask_value == ops.SET == 127 and P.bmd.masks == "rendered"
    assert asdict(P.calibration) == dict(slope=None, intercept=None, mu_scaling=None, mu_water=None)
    assert P.non_default() == [] and P.unverified() == []
    # the script literals made parameters on 2026-09-26 carry the script's values in both presets
    for p in (TIBIA, RADIUS):
        assert (p.rank_first, p.rank_last, p.rank_connect_boundary, p.continuous_at_boundary, p.mask_peel,
                p.corner_min_max_number, p.corner_max_min_number, p.bbc_border) == (1, 1, 0, (0, 0, 0), 0, 0, 1, (0, 0, 0))


def test_the_presets_differ_in_corner_min_and_close2_only_and_workflow_defaults():
    assert Parameters.defaults("radius").diff(Parameters.defaults("tibia")) == ["step1.corner_min", "step1.close2"]
    assert Parameters.defaults(workflow="ipldt").seg.periosteal_mask == "raw"
    assert Parameters.defaults(workflow="ormir_bqrl").seg.periosteal_mask == "rendered"
    assert Parameters.defaults("radius", "ipldt").non_default(workflow="ipldt") == []
    with pytest.raises(ParameterError, match="site must be one of"):
        Parameters.defaults("femur")


def test_autocontour_defaults_are_ormir_xct_s():
    pytest.importorskip("ormir_xct")
    import inspect
    from ormir_xct.core.segmentation.autocontour.AutocontourKnee import AutocontourKnee
    sig = inspect.signature(AutocontourKnee.__init__).parameters
    for k, v in AutocontourParams().knee_kwargs().items():
        assert sig[k].default == v, k


def test_schema_covers_every_field_and_the_set_round_trips():
    for stage, cls in pm.STAGE_CLASSES.items():
        assert [f.name for f in fields(cls)] == list(pm.SCHEMA_SPECS[stage]), stage
    assert len(pm.names()) == sum(len(fields(c)) for c in pm.STAGE_CLASSES.values()) > 100
    P = Parameters.defaults("radius").override({"porosity.grow_axes": [1, 1, 1], "lh.el_size_mm": [0.06, 0.06, 0.07],
                                                "morphometry.maps": "TRAB_SP,CORT_TH"})
    d = json.loads(json.dumps(P.to_dict()))
    assert Parameters.defaults("tibia").override(d) == P, "a complete set replaces the site preset, JSON round trip exact"
    assert P.porosity.grow_axes == (1, 1, 1) and P.lh.el_size_mm == (0.06, 0.06, 0.07)
    assert P.morphometry.maps == ("TRAB_SP", "CORT_TH")
    assert Parameters.from_dict({"step1.close2": 30, "step1.corner_min": 800}) == Parameters.defaults("radius")


# ============================================================================================ 2. parsing and errors
def test_set_parsing_types_and_aliases():
    assert pm.parse_set("lh.laplace_eps=0.5") == {"lh.laplace_eps": 0.5}
    assert pm.parse_set("step1.close2=40") == {"step1.close2": 40}
    assert pm.parse_set(" seg.overlap = cort ") == {"seg.overlap": "cort"}
    assert pm.parse_set("step1.continuous_at_boundary=1,1,0") == {"step1.continuous_x": 1, "step1.continuous_y": 1,
                                                                   "step1.continuous_z": 0}
    assert pm.parse_set("porosity.slice_fraction=[0, 10]") == {"porosity.slice_lo": 0, "porosity.slice_up": 10}
    P = Parameters().override(pm.parse_set("porosity.marrow_connect_boundary=true"))
    assert P.porosity.marrow_connect_boundary is True
    P = Parameters().override({"calibration.slope": "1600.5", "morphometry.voxel_size_mm": "none", "lh.el_size_mm": "0.061"})
    assert P.calibration.slope == 1600.5 and P.morphometry.voxel_size_mm is None and P.lh.el_size_mm == 0.061
    assert Parameters().override({"step1.close2": 40.0}).step1.close2 == 40 and type(Parameters().override({"step1.close2": 40.0}).step1.close2) is int
    assert pm.parse_set("step1.rank_connect_boundary=true") == {"step1.rank_connect_boundary": 1}, "a 0 / 1 flag takes true"


@pytest.mark.parametrize("item,match", [
    ("lh.eps=0.5", r"unknown parameter 'lh.eps'.*valid names for stage 'lh': laplace_eps, lp_cut_off_freq"),
    ("lhh.laplace_eps=0.5", r"unknown parameter stage 'lhh'.*did you mean 'lh'.*valid stages: autocontour"),
    ("laplace_eps=0.5", r"STAGE.NAME"),
    ("lh.laplace_eps", r"STAGE.NAME=VALUE"),
    ("step1.close2=abc", r"step1.close2: expected an integer"),
    ("step1.close2=2.5", r"step1.close2: expected an integer"),
    ("lh.laplace_eps=nan", r"not finite"),
    ("dt.version=4", r"dt.version: 4 is not one of 1, 2, 3"),
    ("lh.pad_offset=middle", r"is not one of ceil, floor"),
    ("porosity.grow_axes=0,1", r"three integers"),
    ("output.mask_value=300", r"must be <= 255"),
    ("step1.continuous_x=2", r"must be <= 1"),
    ("morphometry.maps=TB_TH", r"non-empty list of TRAB_TH"),
    ("porosity.marrow_connect_boundary=maybe", r"true / false"),
    ("step1.close2=true", r"step1.close2: expected an integer"),
])
def test_bad_names_and_values_are_refused_with_the_valid_names(item, match):
    with pytest.raises(ParameterError, match=match):
        pm.parse_set(item)


def test_cross_field_checks():
    with pytest.raises(ParameterError, match="low_thresh 5 > porosity.high_thresh 3"):
        Parameters().override({"porosity.low_thresh": 5})
    with pytest.raises(ParameterError, match="tbth_grid 'tight' applies"):
        Parameters().override({"morphometry.tbth_grid": "tight"})
    with pytest.raises(ParameterError, match="must differ"):
        Parameters().override({"seg.value_trab": 127})
    Parameters().override({"morphometry.tbth_grid": "tight", "morphometry.tbth_object": "trab_seg"})


def test_parameter_files_and_precedence(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"lh": {"laplace_eps": 0.5}, "step1.close2": 40, "porosity.min_pore_voxels": 5}))
    assert pm.load_params_file(str(f)) == {"lh.laplace_eps": 0.5, "step1.close2": 40, "porosity.min_pore_voxels": 5}
    # site preset < file < --set
    R = pm.resolve("radius", str(f))
    assert R.params.step1.corner_min == 800 and R.params.step1.close2 == 40 and R.site == "custom" and R.preset == "radius"
    assert R.non_default == ["step1.close2", "lh.laplace_eps", "porosity.min_pore_voxels"]
    import argparse
    ap = argparse.ArgumentParser()
    pm.add_parameter_arguments(ap)
    args = ap.parse_args(["--params", str(f), "--set", "lh.laplace_eps=0.6", "--set", "step1.close2=50"])
    ov = pm.overrides_from_args(args, {"lh.laplace_eps": 0.55, "dt.ridge_epsilon": 0.8})
    assert ov["lh.laplace_eps"] == 0.6 and ov["step1.close2"] == 50 and ov["dt.ridge_epsilon"] == 0.8
    # a complete set (as `params` prints it) and a report's parameter_set block are files too
    full = tmp_path / "full.json"
    full.write_text(Parameters.defaults("tibia").override({"lh.laplace_eps": 0.4}).to_json(site="tibia"))
    # a complete set is its preset plus its changes: an explicit site replaces the preset, the changes apply on top
    Rr = pm.resolve("radius", str(full))
    assert Rr.params.step1 == RADIUS and Rr.params.lh.laplace_eps == 0.4 and Rr.non_default == ["lh.laplace_eps"]
    Rn = pm.resolve(None, str(full))
    assert Rn.params.step1 == TIBIA and Rn.preset == "tibia" and Rn.non_default == ["lh.laplace_eps"]
    assert pm.read_params_file(str(full)).complete and pm.load_params_file(str(full)) == {"lh.laplace_eps": 0.4}
    rep = tmp_path / "rep.json"
    rep.write_text(json.dumps({"parameter_set": pm.resolve("tibia", {"seg.cc_min_trab": 60}).block()}))
    assert pm.resolve("tibia", str(rep)).non_default == ["seg.cc_min_trab"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"lh": {"eps": 1}}))
    with pytest.raises(ParameterError, match=r"bad.json: unknown parameter 'lh.eps'"):
        pm.load_params_file(str(bad))
    (tmp_path / "x.yaml").write_text("lh: {}")
    with pytest.raises(ParameterError, match="JSON"):
        pm.load_params_file(str(tmp_path / "x.yaml"))
    with pytest.raises(ParameterError, match="not found"):
        pm.load_params_file(str(tmp_path / "missing.json"))


def test_legacy_keywords_win_and_name_the_site():
    R = pm.resolve("radius", {"step1.close2": 40}, step1_params=Step1Params(close2=45), dt_params=engine.DTParams(ridge_epsilon=0.8))
    assert R.params.step1 == Step1Params(close2=45) and R.site == "custom" and R.params.dt.ridge_epsilon == 0.8
    assert pm.resolve("tibia", step1_params=TIBIA).site == "custom", "an explicit Step1Params is 'custom', as before"
    assert pm.resolve("Radius").site == "radius" and pm.resolve(None).site == "tibia"
    assert pm.resolve("tibia", {"step1.corner_min": 800, "step1.close2": 30}).site == "radius"


def test_unverified_values_warn_and_are_listed():
    P = Parameters().override({"step1.sigma": 1.5, "porosity.mode": 1, "render.min_vertices": 8})
    notes = P.unverified()
    assert any(n.startswith("step1.sigma = 1.5") for n in notes) and any(n.startswith("porosity.mode = 1") for n in notes)
    assert any("render.min_vertices = 8" in n and "refuted" in n for n in notes)
    with pytest.warns(pm.ParameterWarning):
        P.warn_unverified()
    assert pm.resolve("tibia", {"step1.sigma": 1.5}).block()["unverified"]


# ============================================================================================ 3. every stage's override bites
SHAPE = (40, 90, 90)
POS = (100, 200, 300)
PROCLOG = ("Mu_Scaling                                       8192\n"
           "Density: slope                         1.61907703e+03\n"
           "Density: intercept                    -3.94095001e+02\n")


def _r():
    yy, xx = np.mgrid[:SHAPE[1], :SHAPE[2]]
    return np.sqrt((yy - 44.5) ** 2 + (xx - 44.5) ** 2)


@pytest.fixture(scope="module")
def step1_phantom():
    r = _r()
    grey = np.broadcast_to(np.where(r <= 40, np.where(r >= 32, 12000, 1000), 300).astype(np.int16), SHAPE).copy()
    f = ndi.gaussian_filter(np.random.default_rng(0).standard_normal(SHAPE), 2.0)
    grey[(f > np.quantile(f, 0.65)) & np.broadcast_to(r < 28, SHAPE)] = 9000
    g = ops.vol(grey, SHAPE[::-1], POS)
    g["proclog"] = PROCLOG
    per = ops.mask_vol(np.broadcast_to(r <= 40, SHAPE).copy(), SHAPE[::-1], POS)
    return g, per, cort_trab_separation(g, per, TIBIA)


@pytest.mark.parametrize("change", [dict(lower_mgha=1500.0), dict(close2=40), dict(erode=1), dict(sigma=1.0),
                                    dict(continuous_x=1, continuous_y=1, continuous_z=1),
                                    dict(bbc_border_x=2, bbc_border_y=2, bbc_border_z=2), dict(rank_last=2)],
                         ids=lambda d: ",".join(d))
def test_step1_overrides_change_the_masks(step1_phantom, change):
    grey, per, ref = step1_phantom
    P = Parameters().override({f"step1.{k}": v for k, v in change.items()})
    res = cort_trab_separation(grey, per, P.step1)
    a = ops.on_grid(ref["cort"], per["dim"], per["pos"]) != 0
    b = ops.on_grid(res["cort"], per["dim"], per["pos"]) != 0
    assert int((a ^ b).sum()) > 0
    assert res["info"]["params"] == P.step1.record() and Step1Params(**res["info"]["params"]) == P.step1


def test_step1_default_literals_are_the_previous_calls(step1_phantom, monkeypatch):
    """The defaults of the new literals are passed on unchanged: the chain calls the reimplemented commands with the script's values."""
    grey, per, ref = step1_phantom
    seen = []
    real = ops.cl_ow_rank_extract
    monkeypatch.setattr(ops, "cl_ow_rank_extract", lambda v, *a, **k: seen.append((a, k)) or real(v, *a, **k))
    res = cort_trab_separation(grey, per, TIBIA)
    assert seen == [((), dict(first_rank=1, last_rank=1, connect_boundary=False))] * 3
    assert np.array_equal(res["cort"]["data"], ref["cort"]["data"]) and np.array_equal(res["trab"]["data"], ref["trab"]["data"])


def test_render_min_vertices_is_a_parameter():
    """A vertical 1 x 2 hole: its raw 6-vertex inner chain is stored at 4 (IPL, test run 19) and dropped at 8."""
    m = np.zeros((1, 16, 16), bool)
    m[0, 3:13, 3:13] = True
    m[0, 7:9, 8] = False
    assert np.array_equal(render_volume(m, min_vertices=4), render_volume(m))
    assert not render_volume(m)[0, 7, 8]
    assert render_volume(m, min_vertices=8)[0, 7, 8], "with 8 the hole's contour is not stored and the hole fills"


@pytest.fixture(scope="module")
def lh_phantom():
    """A native volume with odd padding on every axis (so 'ceil' and 'floor' differ), a periosteal disc and two
    compartment gobjs."""
    shape = (25, 47, 47)
    yy, xx = np.mgrid[:shape[1], :shape[2]]
    r = np.sqrt((yy - 23) ** 2 + (xx - 23) ** 2)
    grey = np.broadcast_to(np.where(r <= 21, np.where(r >= 16, 12000, 1000), 300).astype(np.int16), shape).copy()
    f = ndi.gaussian_filter(np.random.default_rng(3).standard_normal(shape), 1.5)
    grey[(f > np.quantile(f, 0.6)) & np.broadcast_to(r < 15, shape)] = 9000
    per = np.broadcast_to(r <= 21, shape).copy()
    Gc = np.broadcast_to((r <= 21) & (r >= 16), shape).copy()
    Gt = np.broadcast_to(r < 16, shape).copy()
    return grey, per, Gc, Gt


EL = (0.0607, 0.0607, 0.0607)


def test_lh_segment_defaults_are_the_previous_call_sequence(lh_phantom):
    grey, per, Gc, Gt = lh_phantom
    bm = engine.laplace_hamming_threshold(grey, EL, pad_offset=engine.LH_PAD_OFFSET)          # the call made before
    c, t = engine.ipl_seg_assembly(bm, per, Gc, Gt)
    r = engine.lh_segment(grey, EL, per, Gc, Gt, LHParams(), SegParams())
    assert np.array_equal(r["lh"], bm) and np.array_equal(r["cort_seg"], c) and np.array_equal(r["trab_seg"], t)
    seg = np.zeros(bm.shape, np.uint8)
    seg[c] = 127
    seg[t] = 126
    assert np.array_equal(r["seg"], seg) and r["threshold"] == 15564


@pytest.mark.parametrize("name,value", [("laplace_eps", 0.6), ("lp_cut_off_freq", 0.2), ("hamming_amp", 0.5),
                                        ("norm_max", 150000.0), ("lower_permille", 400.0), ("upper_permille", 500.0),
                                        ("border", "none"), ("border", "zero"), ("pad_offset", "floor"),
                                        ("el_size_mm", 0.08)])
def test_each_lh_parameter_changes_the_threshold(lh_phantom, name, value):
    grey, per, Gc, Gt = lh_phantom
    ref = engine.lh_segment(grey, EL, per, Gc, Gt)["lh"]
    P = Parameters().override({f"lh.{name}": value})
    el = engine.lh_el_size({"el_size_mm": EL}, P.lh.el_size_mm)[0]
    got = engine.lh_segment(grey, el, per, Gc, Gt, P.lh)["lh"]
    assert got.shape == ref.shape and int((got ^ ref).sum()) > 0


def test_lh_dtype_reaches_the_filter(lh_phantom, monkeypatch):
    grey, per, Gc, Gt = lh_phantom
    seen = {}
    real = engine.lh_filter_core
    monkeypatch.setattr(engine, "lh_filter_core", lambda *a, **k: seen.setdefault("args", a) and real(*a, **k))
    engine.lh_segment(grey, EL, per, Gc, Gt, Parameters().override({"lh.dtype": "float64", "lh.laplace_eps": 0.4}).lh)
    assert seen["args"][3] == "float64" and seen["args"][5] == 0.4


@pytest.mark.parametrize("change,what", [({"cc_min_trab": 10 ** 7}, "trab_seg"), ({"cc_min_cort": 10 ** 7}, "cort_seg"),
                                         ({"cc_max_trab": 50}, "trab_seg"), ({"trab_seg_mask": "none"}, "trab_seg"),
                                         ({"peel_trab": 2}, "trab_seg"), ({"peel_cort": 2}, "cort_seg"),
                                         ({"peel_periosteal": 3}, "cort_seg"), ({"value_cort": 200}, "seg"),
                                         ({"value_trab": 100}, "seg")])
def test_seg_assembly_overrides_change_the_segmentation(lh_phantom, change, what):
    grey, per, Gc, Gt = lh_phantom
    ref = engine.lh_segment(grey, EL, per, Gc, Gt)
    P = Parameters().override({f"seg.{k}": v for k, v in change.items()})
    got = engine.lh_segment(grey, EL, per, Gc, Gt, P.lh, P.seg)
    assert not np.array_equal(got[what], ref[what])


def test_seg_overlap_and_order(lh_phantom):
    grey, per, Gc, Gt = lh_phantom
    Gc2 = np.broadcast_to(ndi.binary_dilation(Gc[0], iterations=2), Gc.shape).copy()     # compartments that overlap
    a = engine.lh_segment(grey, EL, per, Gc2, Gt)
    b = engine.lh_segment(grey, EL, per, Gc2, Gt, seg=SegParams(overlap="cort"))
    both = a["cort_seg"] & a["trab_seg"]
    assert both.any() and (a["seg"][both] == 126).all() and (b["seg"][both] == 127).all()
    assert np.array_equal(a["seg"][~both], b["seg"][~both])
    g = engine.lh_segment(grey, EL, np.zeros_like(per), Gc, Gt, seg=SegParams(order="gobj_first"))
    assert g["cort_seg"].any(), "gobj_first does not apply the periosteal mask"


@pytest.fixture(scope="module")
def dt_phantom():
    from conftest import random_phantom
    seg = random_phantom((20, 40, 40), seed=1)
    yy, xx = np.mgrid[:40, :40]
    r = np.sqrt((yy - 19.5) ** 2 + (xx - 19.5) ** 2)
    G_trab = np.broadcast_to(r <= 13, seg.shape).copy()
    ring = np.broadcast_to((r > 13) & (r <= 18), seg.shape).copy()
    trab_seg = seg & ndi.binary_erosion(G_trab, iterations=2)
    pad = lambda a: np.pad(a, ((0, 0), (6, 6), (6, 6)))                  # noqa: E731  (boxes smaller than the grid)
    return pad((seg & (G_trab | ring)) | ring), pad(trab_seg), pad(G_trab), pad(ring), pad(ring | G_trab)


def _dt(ph, morph=None, dt=None):
    seg, trab_seg, G_trab, cort, G_prx = ph
    shape = seg.shape
    return engine.dt_stage(seg.astype(np.uint8) * 127, G_trab, cort, cort, shape[::-1], (0, 0, 0), 0.0607, dt, "cpu",
                           morph=morph, trab_seg=trab_seg, G_prx=G_prx)


def test_dt_stage_defaults_and_overrides(dt_phantom):
    m0, boxes0, _ = _dt(dt_phantom)
    assert set(m0["results"]) == {"TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH"}
    m1, _, _ = _dt(dt_phantom, dt=engine.DTParams(ridge_epsilon=0.0))
    assert not np.array_equal(m1["results"]["TRAB_TH"].map, m0["results"]["TRAB_TH"].map)
    m2, _, _ = _dt(dt_phantom, dt=engine.DTParams(assign_epsilon=0.0))
    assert m2["metrics"]["Tb_Sp_mm"] != m0["metrics"]["Tb_Sp_mm"]
    # the Tb.Th object: the Scripts' TRAB_SEG instead of the whole SEG
    m3, _, _ = _dt(dt_phantom, morph=MorphometryParams(tbth_object="trab_seg"))
    assert m3["metrics"]["Tb_Th_mm"] != m0["metrics"]["Tb_Th_mm"]
    assert m3["metrics"]["Tb_Sp_mm"] == m0["metrics"]["Tb_Sp_mm"] and m3["metrics"]["BV_TV"] == m0["metrics"]["BV_TV"]
    # ... on TRAB_SEG's own box (the 2022 script)
    m4, boxes4, _ = _dt(dt_phantom, morph=MorphometryParams(tbth_object="trab_seg", tbth_grid="tight"))
    assert boxes4["TRAB_TH"]["dim"] != boxes0["TRAB_TH"]["dim"] and "TRAB_TH" in m4["results"]
    # BV of BV/TV, the maps computed, the Ct.Th object, the grid border
    m5, _, _ = _dt(dt_phantom, morph=MorphometryParams(bvtv_object="trab_seg"))
    assert m5["metrics"]["BV_TV"] < m0["metrics"]["BV_TV"]
    m6, _, _ = _dt(dt_phantom, morph=MorphometryParams(maps=("TRAB_SP",)))
    assert set(m6["results"]) == {"TRAB_SP"} and m6["metrics"]["Tb_Sp_mm"] == m0["metrics"]["Tb_Sp_mm"]
    m7, _, cobj = _dt(dt_phantom, morph=MorphometryParams(ctth_object="periosteal_minus_trab"))
    assert np.array_equal(cobj, dt_phantom[4] & ~dt_phantom[2])
    m8, boxes8, _ = _dt(dt_phantom, morph=MorphometryParams(grid_border=3))
    grow = lambda d: tuple(v + (0 if i == 2 else 6) for i, v in enumerate(d))        # noqa: E731  (z is full: clipped)
    assert tuple(boxes8["TRAB_SP"]["dim"]) == grow(boxes0["TRAB_SP"]["dim"])
    assert tuple(boxes8["CORT_TH"]["dim"]) == grow(boxes0["CORT_TH"]["dim"])


@pytest.fixture(scope="module")
def pore_inputs():
    """The pore phantom's rendered cortical contour and CORT_SEG, from a default ORMIR-BQRL segmentation."""
    pytest.importorskip("SimpleITK")
    pytest.importorskip("itk")
    pytest.importorskip("ormir_xct")
    import test_ormir_bqrl_porosity as tp
    from ormir_bqrl import stages
    import tempfile
    d = tempfile.mkdtemp(prefix="ipldt_params_")
    loaded = stages.load(tp.make_aim(os.path.join(d, "PORE.AIM")))
    cort, trab, ALL, _ = stages.compartments(loaded, tp.periosteal_disc(), TIBIA)
    segm, G_cort, G_trab = stages.segment(loaded, ALL, cort, trab)
    return G_cort, segm.cort_seg, loaded.grid


def test_porosity_defaults_and_overrides(pore_inputs):
    G_cort, cort_seg, grid = pore_inputs
    ref_pore, ref, _ = engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos)
    same, m, _ = engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos, params=PoreParams())
    assert np.array_equal(same, ref_pore) and m == ref and ref["Ct_Po"] > 0
    for name, value in (("min_pore_voxels", 400), ("slice_up", 50.0), ("slice_lo", 1.0), ("max_pore_voxels", 30),
                        ("mode", 1), ("render_grid_margin", 5)):
        P = Parameters().override({f"porosity.{name}": value})
        pore, met, _ = engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos, params=P.porosity)
        assert not np.array_equal(pore, ref_pore), name
    _, met, _ = engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos, params=PoreParams(ct_po="pore_plus_bone"))
    assert met["Ct_Po"] == met["Ct_Po_pore_voxels"] / (met["Ct_Po_pore_voxels"] + int(cort_seg.sum())) != ref["Ct_Po"]
    pa, _, ga = engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos, params=PoreParams(grid="aim"))
    assert ga["cascade"] == {"dim_xyz": list(grid.dim), "pos_xyz": list(grid.pos)}
    _, _, gm = engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos, params=PoreParams(render_grid_margin=5))
    assert gm["render"] != engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos)[2]["render"]


def test_pore_parameters_reach_the_cascade_commands(pore_inputs, monkeypatch):
    """The hysteresis is a no-op on this phantom (value 2 is empty, as on a patella), so its arguments, the marrow
    blob's rank / -connect_boundary and the masks' peel are checked where they arrive: the reimplemented commands."""
    G_cort, cort_seg, grid = pore_inputs
    seen = {"hys": [], "rank": [], "peel": [], "nr": []}
    real_h, real_r, real_g, real_n = porosity.hysteresis_threshold, ops.cl_ow_rank_extract, ops.gobj_maskaimpeel_ow, ops.cl_nr_extract
    monkeypatch.setattr(porosity, "hysteresis_threshold", lambda v, **k: seen["hys"].append(k) or real_h(v, **k))
    monkeypatch.setattr(ops, "cl_ow_rank_extract", lambda v, *a, **k: seen["rank"].append((a, k)) or real_r(v, *a, **k))
    monkeypatch.setattr(ops, "gobj_maskaimpeel_ow", lambda v, g, n: seen["peel"].append(n) or real_g(v, g, n))
    monkeypatch.setattr(ops, "cl_nr_extract", lambda v, *a, **k: seen["nr"].append(a) or real_n(v, *a, **k))
    P = Parameters().override({"porosity.low_thresh": 0, "porosity.high_thresh": 2, "porosity.grow_axes": "1 1 0",
                               "porosity.marrow_rank_last": 2, "porosity.marrow_connect_boundary": True,
                               "porosity.gobj_peel": 2, "porosity.max_pore_voxels": 5000})
    engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos, params=P.porosity)
    assert seen["hys"] == [dict(low_thresh=0, high_thresh=2, unit=5, mode=0, grow_axes=(1, 1, 0), value_in_range=127)]
    assert seen["rank"] == [((1, 2), dict(connect_boundary=True, value_in_range=2))]
    assert seen["peel"] == [2, 2] and seen["nr"] == [(20, 5000)]
    for k in seen:
        seen[k].clear()
    engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos)
    assert seen["hys"] == [dict(porosity.SCRIPT32_HYSTERESIS)] and seen["rank"] == [((1, 1), dict(connect_boundary=False, value_in_range=2))]
    assert seen["peel"] == [0, 0] and seen["nr"] == [(20, 0)]


# ============================================================================================ 4. the workflows
@pytest.fixture(scope="module")
def phantom_aim(tmp_path_factory):
    pytest.importorskip("SimpleITK")
    pytest.importorskip("itk")
    pytest.importorskip("ormir_xct")
    import test_ormir_bqrl_porosity as tp
    return tp.make_aim(str(tmp_path_factory.mktemp("params_aim") / "PHANTOM.AIM")), tp.periosteal_disc()


@pytest.fixture(scope="module")
def default_run(phantom_aim, tmp_path_factory):
    from ormir_bqrl.pipeline import run
    aim, disc = phantom_aim
    out = str(tmp_path_factory.mktemp("params_run") / "default")
    return run(aim, out, periosteal=disc, compute_bmd=False, preview=False, log=engine.Logger(echo=False))


def _vol(path):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


VOLS = ("SEG", "CORT_SEG", "TRAB_SEG", "CORT_MASK", "TRAB_MASK", "CORT_GOBJ", "TRAB_GOBJ", "PORE", "TRAB_TH", "TRAB_SP",
        "TRAB_1N", "CORT_TH")


def _same_outputs(a, b, names=VOLS):
    return {n: bool(np.array_equal(_vol(a["outputs"][n]), _vol(b["outputs"][n]))) for n in names}


def test_default_run_records_the_validated_defaults(default_run, phantom_aim, tmp_path):
    from ormir_bqrl.pipeline import run
    rep = default_run
    ps = rep["parameter_set"]
    assert ps["non_default"] == [] and ps["validated_default"] is True and ps["site"] == "tibia" and ps["preset"] == "tibia"
    assert ps["statement"].startswith("the validated IPL defaults (Script 32 tibia preset)")
    assert ps["values"] == Parameters.defaults("tibia", "ormir_bqrl").to_dict() and ps["unverified"] == []
    md = open(rep["outputs"]["report_md"], encoding="utf-8").read()
    assert "- parameters: the validated IPL defaults" in md and "## Parameter set" in md and "| lh.laplace_eps | 0.45 | |" in md
    assert "NOT the validated" not in md
    csv_text = open(rep["outputs"]["report_csv"], encoding="utf-8").read()
    assert "parameter_set.values.lh.laplace_eps,0.45" in csv_text and "parameter_set.validated_default,True" in csv_text
    saved = json.load(open(rep["outputs"]["parameters"], encoding="utf-8"))
    assert saved["values"] == ps["values"] and saved["non_default"] == []
    assert pm.resolve("tibia", rep["outputs"]["parameters"], workflow="ormir_bqrl").is_default
    # the same run with the defaults given explicitly (a Parameters, an empty mapping, the saved file) is bit-identical
    aim, disc = phantom_aim
    for i, prm in enumerate((Parameters.defaults("tibia"), {}, rep["outputs"]["parameters"])):
        r = run(aim, str(tmp_path / f"explicit{i}"), periosteal=disc, compute_bmd=False, preview=False,
                log=engine.Logger(echo=False), parameters=prm)
        assert all(_same_outputs(rep, r).values()) and r["morphometry"] == rep["morphometry"] and r["porosity"] == rep["porosity"]
        assert r["parameter_set"]["non_default"] == []


@pytest.mark.parametrize("override,changed", [
    ({"lh.laplace_eps": 0.6}, "SEG"), ({"lh.lp_cut_off_freq": 0.2}, "SEG"), ({"lh.lower_permille": 400}, "SEG"),
    ({"seg.cc_min_trab": 400}, "TRAB_SEG"), ({"seg.cc_min_cort": 10 ** 7}, "CORT_SEG"),
    ({"porosity.slice_up": 40.0}, "PORE"), ({"porosity.min_pore_voxels": 300}, "PORE"),
    ({"dt.ridge_epsilon": 0.5}, "TRAB_TH"), ({"morphometry.tbth_object": "trab_seg"}, "TRAB_TH"),
    ({"step1.lower_mgha": 1500.0}, "CORT_MASK"), ({"step1.erode": 1}, "CORT_MASK"), ({"output.mask_value": 1}, "CORT_MASK"),
])
def test_nondefault_run_changes_the_output_and_says_so(default_run, phantom_aim, tmp_path, override, changed):
    from ormir_bqrl.pipeline import run
    aim, disc = phantom_aim
    rep = run(aim, str(tmp_path / "nd"), periosteal=disc, compute_bmd=False, preview=False, log=engine.Logger(echo=False),
              parameters=override)
    same = _same_outputs(default_run, rep)
    assert not same[changed], (override, same)
    ps = rep["parameter_set"]
    assert ps["non_default"] == list(override) and ps["validated_default"] is False
    assert ps["statement"].startswith("differs from the validated IPL defaults") and list(override)[0] in ps["statement"]
    assert ("IPL's own choice" in ps["statement"]) == (override == {"morphometry.tbth_object": "trab_seg"})
    md = open(rep["outputs"]["report_md"], encoding="utf-8").read()
    assert "**parameters: NOT the validated IPL defaults.**" in md and f"| **{list(override)[0]}** |" in md
    assert "NON-DEFAULT PARAMETERS" in open(rep["outputs"]["log"], encoding="utf-8").read()
    key, val = next(iter(override.items()))
    assert pm.flatten_overrides({key: val}) and rep["parameter_set"]["non_default_values"][key]["value"] == \
        Parameters().override(override).flat()[key]
    if key.startswith("lh."):
        assert rep["parameters"]["laplace_hamming"][{"lh.laplace_eps": "laplace_eps", "lh.lp_cut_off_freq": "lp_cut_off_freq",
                                                     "lh.lower_permille": "threshold"}[key]] == \
            (val if key != "lh.lower_permille" else pm.permille_to_short(val))
    if key == "seg.cc_min_trab":
        assert rep["parameters"]["seg_assembly"]["cl_nr_extract_min_trab"] == 400
    if key == "porosity.min_pore_voxels":
        assert rep["parameters"]["porosity"]["min_pore_voxels"] == 300


def test_redo_keeps_the_run_parameters(phantom_aim, tmp_path):
    """A NON-default run redone without edits reproduces itself exactly (the redo reads the run's parameter set);
    a compartment redo keeps the set; an explicit override on the redo is applied on top and recorded."""
    from ormir_bqrl import redo as redo_mod
    from ormir_bqrl.pipeline import run
    aim, disc = phantom_aim
    ov = {"lh.laplace_eps": 0.5, "seg.cc_min_trab": 100, "porosity.min_pore_voxels": 10, "dt.assign_epsilon": 0.4}
    out = str(tmp_path / "run")
    rep = run(aim, out, periosteal=disc, compute_bmd=False, preview=False, log=engine.Logger(echo=False), parameters=ov)
    same = redo_mod._from_run_masks(aim, out, str(tmp_path / "same"), compute_bmd=False, preview=False, log=engine.Logger(echo=False))
    assert all(_same_outputs(rep, same).values()) and same["morphometry"] == rep["morphometry"] and same["porosity"] == rep["porosity"]
    assert same["parameter_set"]["values"] == rep["parameter_set"]["values"]
    assert sorted(same["parameter_set"]["non_default"]) == sorted(ov) == sorted(rep["parameter_set"]["non_default"])
    assert same["parameter_set"]["sources"][0].startswith("the run's parameter set")
    labels = _vol(rep["outputs"]["labelmap"])
    T = labels == 2
    for z in range(10, 30):
        T[z] |= (labels[z] == 1) & ndi.binary_dilation(labels[z] == 2, iterations=2)
    r1 = redo_mod.run_from_masks(aim, out, str(tmp_path / "trab"), trab=T, compute_bmd=False, preview=False,
                                 log=engine.Logger(echo=False))
    assert r1["parameter_set"]["values"] == rep["parameter_set"]["values"] and r1["parameters"]["dt"]["assign_epsilon"] == 0.4
    r2 = redo_mod.run_from_masks(aim, out, str(tmp_path / "trab2"), trab=T, compute_bmd=False, preview=False,
                                 log=engine.Logger(echo=False), parameters={"dt.assign_epsilon": 0.5, "lh.laplace_eps": 0.45})
    assert r2["parameter_set"]["non_default"] == ["seg.cc_min_trab", "porosity.min_pore_voxels"]
    assert r2["morphometry"] != r1["morphometry"]
    # a periosteal redo reruns STEP 1 through run() with the run's set
    P = labels > 0
    P[:, _r() > 38] = False
    r3 = redo_mod.run_from_masks(aim, out, str(tmp_path / "peri"), periosteal=P, compute_bmd=False, preview=False,
                                 log=engine.Logger(echo=False))
    assert r3["parameter_set"]["values"] == rep["parameter_set"]["values"] and r3["step1"] is not None
    # an explicit site on a compartment redo is not applied (STEP 1 does not run): the run's step1 block is kept,
    # the site is the run's, and the redo says so
    with pytest.warns(pm.ParameterWarning, match="step1.corner_min = 800 not applied"):
        r4 = redo_mod.run_from_masks(aim, out, str(tmp_path / "radius"), trab=T, site="radius", compute_bmd=False,
                                     preview=False, log=engine.Logger(echo=False))
    assert r4["site"] == "tibia" and r4["parameter_set"]["values"]["step1"]["close2"] == 50
    assert r4["parameter_set"]["not_applied"] == ["step1.corner_min", "step1.close2"]
    assert sorted(r4["parameter_set"]["non_default"]) == sorted(ov)
    assert all(_same_outputs(r1, r4).values()) and r4["morphometry"] == r1["morphometry"]


def test_run_parameters_of_old_reports():
    from ormir_bqrl.redo import run_parameters
    old = {"schema": "ormir-bqrl/1", "site": "radius", "parameters": {"step1": {k: v for k, v in asdict(RADIUS).items()
                                                                                  if k not in ("rank_first", "mask_peel")},
                                                                         "dt": {"ridge_epsilon": 0.9, "assign_epsilon": 0.5,
                                                                                "peel_iter": -1, "version": 3,
                                                                                "suppress_boundary": 2}}}
    P, preset = run_parameters(old)
    assert P == Parameters.defaults("radius") and preset == "radius"
    ipldt_style = {"parameters": {"site": "tibia", "ridge_epsilon": 0.8}, "step1": {"params": asdict(TIBIA)}}
    P, preset = run_parameters(ipldt_style)
    assert P.dt.ridge_epsilon == 0.8 and P.step1 == TIBIA and preset == "tibia"
    new_ipldt = {"parameters": {"site": "tibia"}, "parameter_set": pm.resolve("tibia", workflow="ipldt").block()}
    P, _ = run_parameters(new_ipldt)
    assert P.seg.periosteal_mask is None, "ipldt's own default (raw) is not carried into an ORMIR-BQRL redo"


def test_run_pipeline_takes_parameters_and_a_periosteal(phantom_aim, tmp_path, monkeypatch):
    aim, disc = phantom_aim
    monkeypatch.setattr(engine, "step2_autocontour", lambda img, calib, log=None: engine.array_to_sitk(disc.astype(np.uint8), img))
    ref = engine.run_pipeline(aim, str(tmp_path / "ref"), compute_bmd=False, log=engine.Logger(echo=False))
    assert ref["parameter_set"]["non_default"] == [] and ref["parameter_set"]["workflow"] == "ipldt"
    assert ref["parameter_set"]["values"]["seg"]["periosteal_mask"] == "raw"
    assert os.path.exists(str(tmp_path / "ref" / "PHANTOM_parameters.json"))
    monkeypatch.undo()
    got = engine.run_pipeline(aim, str(tmp_path / "per"), compute_bmd=False, log=engine.Logger(echo=False), periosteal=disc)
    assert got["morphometry"] == ref["morphometry"] and got["porosity"] == ref["porosity"]
    assert got["periosteal_source"] == "array"
    for n in ("SEG", "TRAB_TH", "CORT_TH", "PORE"):
        assert np.array_equal(_vol(str(tmp_path / "ref" / f"PHANTOM_{n}.nii.gz")), _vol(str(tmp_path / "per" / f"PHANTOM_{n}.nii.gz"))), n
    nd = engine.run_pipeline(aim, str(tmp_path / "nd"), compute_bmd=False, log=engine.Logger(echo=False), periosteal=disc,
                             parameters={"porosity.min_pore_voxels": 300, "lh.el_size_mm": 0.05}, lh_voxel_size_mm=0.07)
    assert nd["parameter_set"]["non_default"] == ["lh.el_size_mm", "porosity.min_pore_voxels"]
    assert nd["parameter_set"]["values"]["lh"]["el_size_mm"] == 0.07, "the dedicated keyword wins over parameters"
    assert nd["parameters"]["lh_voxel_size_mm"] == [0.07] * 3 and nd["porosity"] != ref["porosity"]
    assert nd["compartments"]["SEG_voxels"] != ref["compartments"]["SEG_voxels"]
    # params= takes a complete Parameters too
    full = engine.run_pipeline(aim, str(tmp_path / "full"), compute_bmd=False, log=engine.Logger(echo=False), periosteal=disc,
                               params=Parameters.defaults("tibia"))
    assert full["morphometry"] == ref["morphometry"]


# ============================================================================================ 5. the command lines
def test_ormir_bqrl_params_command(tmp_path, capsys):
    from ormir_bqrl import cli
    assert cli.main(["params", "--site", "radius"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["values"]["step1"]["corner_min"] == 800 and d["non_default"] == [] and d["site"] == "radius"
    assert cli.main(["params", "--set", "lh.laplace_eps=0.5", "--describe"]) == 0
    out = capsys.readouterr().out
    assert "* lh.laplace_eps" in out and "(default 0.45)" in out and "/fft_laplace_hamming -laplace_eps" in out
    assert "* = differs from the default; this configuration differs from the validated IPL defaults" in out
    assert out.count("differs from the validated IPL defaults") == 1
    f = tmp_path / "set.json"
    assert cli.main(["params", "--site", "tibia", "--out", str(f), "--set", "step1.close2=30", "--set", "step1.corner_min=800"]) == 0
    capsys.readouterr()
    assert json.load(open(f))["site"] == "radius"
    with pytest.raises(SystemExit) as e:
        cli.main(["params", "--set", "dt.ridge_eps=1"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "unknown parameter 'dt.ridge_eps'" in err and "ridge_epsilon, assign_epsilon" in err


def test_ormir_bqrl_run_redo_batch_pass_the_overrides(tmp_path, monkeypatch, capsys):
    from ormir_bqrl import cli
    aim = tmp_path / "S.AIM"
    aim.write_bytes(b"\0" * 64)
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"lh": {"laplace_eps": 0.4}, "seg.cc_min_trab": 60}))
    seen = []
    fake = lambda aim_path, out_dir, **kw: seen.append(kw.get("parameters")) or dict(  # noqa: E731
        sample="S", site="tibia", run=dict(kind="run", out_dir=str(out_dir)), summary={}, masks={}, outputs={})
    fake_redo = lambda aim_path, run_dir, **kw: fake(aim_path, run_dir, **{k: v for k, v in kw.items() if k != "out_dir"})  # noqa: E731
    monkeypatch.setattr(cli, "_import_pipeline", lambda: fake)
    monkeypatch.setattr(cli, "_import_redo", lambda: fake_redo)
    assert cli.main(["run", str(aim), str(tmp_path / "o"), "--params", str(f), "--set", "lh.laplace_eps=0.6"]) == 0
    assert seen[-1] == {"lh.laplace_eps": 0.6, "seg.cc_min_trab": 60}
    # the overrides say where they came from (the report's parameter_set.sources)
    assert seen[-1].source == f"--params parameter file {f} (2 value(s)); --set lh.laplace_eps"
    assert pm.resolve("tibia", seen[-1], workflow="ormir_bqrl").sources[-1] == seen[-1].source
    assert cli.main(["redo", str(aim), str(tmp_path), "--trab", str(aim), "--set", "dt.version=2"]) == 0
    assert seen[-1] == {"dt.version": 2} and seen[-1].source == "--set dt.version"
    assert cli.main(["batch", str(tmp_path / "b"), str(aim), "--set", "porosity.min_pore_voxels=5"]) == 0
    assert seen[-1] == {"porosity.min_pore_voxels": 5}
    assert "porosity.min_pore_voxels=5" in open(tmp_path / "b" / "batch.log", encoding="utf-8").read()
    assert cli.main(["run", str(aim), str(tmp_path / "o2")]) == 0
    assert seen[-1] is None, "no override: run() is called exactly as before (no parameters keyword)"
    with pytest.raises(SystemExit) as e:
        cli.main(["run", str(aim), str(tmp_path / "o3"), "--set", "step1.close2=x"])
    assert e.value.code == 2 and "step1.close2: expected an integer" in capsys.readouterr().err


def test_ipldt_pipeline_print_params_and_errors(capsys, monkeypatch):
    from ipldt.workflows import ipl_pipeline_workflow as wf
    assert wf.main(["--print-params", "--site", "radius", "--ridge-epsilon", "0.8", "--set", "lh.border=none",
                    "--lh-voxel-size", "0.06", "0.06", "0.07"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["workflow"] == "ipldt" and d["values"]["seg"]["periosteal_mask"] == "raw" and d["values"]["step1"]["close2"] == 30
    assert d["non_default"] == ["lh.el_size_mm", "lh.border", "dt.ridge_epsilon"]
    assert d["values"]["lh"]["el_size_mm"] == [0.06, 0.06, 0.07]
    # precedence: the dedicated flag < --set
    wf.main(["--print-params", "--ridge-epsilon", "0.8", "--set", "dt.ridge_epsilon=0.7"])
    assert json.loads(capsys.readouterr().out)["values"]["dt"]["ridge_epsilon"] == 0.7
    assert wf.main(["--print-params", "--describe"]) == 0
    assert "every value is the validated IPL default" in capsys.readouterr().out
    for argv, msg in ((["--print-params", "--set", "seg.min=1"], "unknown parameter 'seg.min'"),
                      (["--print-params", "--lh-voxel-size", "0.06", "0.07"], "one value or three"),
                      ([], "input_aim, output_dir")):
        with pytest.raises(SystemExit) as e:
            wf.main(argv)
        assert e.value.code == 2 and msg in capsys.readouterr().err


def test_single_dt_commands_take_the_dt_stage(tmp_path, capsys, monkeypatch):
    import argparse
    from ipldt.workflows._cli import add_dt_arguments, dt_params_from_args
    ap = argparse.ArgumentParser()
    add_dt_arguments(ap)
    assert dt_params_from_args(ap.parse_args([])) == engine.DTParams()
    f = tmp_path / "dt.json"
    f.write_text(json.dumps({"dt": {"ridge_epsilon": 0.7, "version": 2}}))
    p = dt_params_from_args(ap.parse_args(["--params", str(f), "--assign-epsilon", "0.4", "--set", "dt.version=1"]))
    assert (p.ridge_epsilon, p.assign_epsilon, p.version) == (0.7, 0.4, 1)
    with pytest.raises(ParameterError, match="dt stage only"):
        dt_params_from_args(ap.parse_args(["--set", "lh.laplace_eps=0.5"]))


# ============================================================================================ 6. the autocontour
def test_autocontour_direct_path_equals_the_wrapper_and_parameters_bite(phantom_aim):
    """At ORMIR-XCT's defaults the workflows call ormir_xct's autocontour() as before; any other value goes through
    AutocontourKnee(**peri_*).get_periosteal_mask directly.  That direct path, fed the defaults, gives the wrapper's
    mask exactly (the component-1 call does not depend on the component-2 call), so a parameter change is the only
    difference; and a changed threshold changes the mask."""
    import SimpleITK as sitk
    from ormir_xct.core.segmentation.autocontour.AutocontourKnee import AutocontourKnee
    from ormir_xct.core.util.file_reader import verify_image
    from ormir_xct.core.util.hrpqct_rescale import convert_hu_to_bmd
    from ormir_bqrl import stages
    aim, _ = phantom_aim
    loaded = stages.load(aim)
    c = loaded.calib
    wrapper = sitk.GetArrayFromImage(engine.step2_autocontour(loaded.img_hu, c))
    img = convert_hu_to_bmd(verify_image(loaded.img_hu, sitk.sitkFloat32), c["mu_water"], c["rescale_slope"], c["rescale_intercept"])
    direct = sitk.GetArrayFromImage(AutocontourKnee(**AutocontourParams().knee_kwargs()).get_periosteal_mask(img, 1)) > 0
    assert wrapper.any() and np.array_equal(wrapper > 0, direct)
    changed = sitk.GetArrayFromImage(engine.step2_autocontour(loaded.img_hu, c, params=AutocontourParams(peri_s4_close_radius=2)))
    assert not np.array_equal(changed > 0, wrapper > 0)
    # the workflow passes the block (and the calibration overrides) on; the defaults keep the three-argument call
    calls = []
    real = engine.step2_autocontour
    try:
        engine.step2_autocontour = lambda img, calib, log=None, **k: calls.append((dict(calib), k)) or real(img, calib, log, **k)
        stages.autocontour(loaded)
        stages.autocontour(loaded, parameters=Parameters().override({"autocontour.peri_s1_lower": 400, "calibration.mu_water": 0.25}))
    finally:
        engine.step2_autocontour = real
    assert calls[0] == (c, {})
    assert calls[1][1] == {"params": AutocontourParams(peri_s1_lower=400.0)} and calls[1][0]["mu_water"] == 0.25


def test_calibration_overrides_reach_step1():
    grey = {"proclog": PROCLOG}
    cal, src = engine.step1_calibration(grey)
    assert (cal, src) == (dict(slope=1619.07703, intercept=-394.095001, mu_scaling=8192.0), "proclog")
    cal2, src2 = engine.step1_calibration(grey, override=pm.CalibrationParams(slope=1600.0))
    assert cal2 == dict(cal, slope=1600.0) and src2 == "proclog+override"
    cal3, src3 = engine.step1_calibration({}, override=pm.CalibrationParams(slope=1.0, intercept=2.0, mu_scaling=3.0))
    assert cal3 == dict(slope=1.0, intercept=2.0, mu_scaling=3.0) and src3 == "override"
    c = dict(mu_scaling=8192.0, mu_water=0.2409, rescale_slope=1619.07703, rescale_intercept=-394.095001)
    assert engine.calibration_with(c, pm.CalibrationParams()) is c
    assert engine.calibration_with(c, pm.CalibrationParams(intercept=-400.0, mu_water=0.25)) == dict(c, rescale_intercept=-400.0, mu_water=0.25)


def test_ipldt_pipeline_command_runs_with_a_periosteal_file_and_set(phantom_aim, tmp_path, capsys):
    from ipldt.io import write_nifti
    from ipldt.workflows import ipl_pipeline_workflow as wf
    import test_ormir_bqrl_porosity as tp
    aim, disc = phantom_aim
    nii = str(tmp_path / "disc.nii.gz")
    write_nifti(nii, disc.astype(np.uint8), tp.EL, tp.POS)
    out = str(tmp_path / "cli")
    assert wf.main([aim, out, "--periosteal", nii, "--no-bmd", "--set", "porosity.min_pore_voxels=300"]) == 0
    assert "differs from the validated IPL defaults" in capsys.readouterr().out
    rep = json.load(open(os.path.join(out, "PHANTOM_report.json"), encoding="utf-8"))
    assert rep["parameter_set"]["non_default"] == ["porosity.min_pore_voxels"] and rep["periosteal_source"].startswith("file ")
    assert rep["parameter_set"]["sources"] == ["site preset 'tibia'", "--set porosity.min_pore_voxels"]
    assert rep["compartments"]["PRX_MASK_voxels"] == int(disc.sum())
    saved = json.load(open(os.path.join(out, "PHANTOM_parameters.json"), encoding="utf-8"))
    assert saved["values"]["porosity"]["min_pore_voxels"] == 300 and saved["workflow"] == "ipldt"


# ============================================================================================ 7. checks added on 2026-09-26
def test_blocks_built_directly_are_checked_and_dispatches_refuse_unknown_values(lh_phantom, dt_phantom, pore_inputs):
    """Every field is checked where a set is built, also for blocks built directly (Parameters(...), .replace(...)):
    a typo in a choice is refused instead of silently running the other branch; a numeric string is converted.  The
    stage functions that dispatch on a choice refuse a value outside it (no 'else = the alternative')."""
    from ipldt.params import BMDParams, OutputParams
    for bad, match in ((dict(morphometry=MorphometryParams(ctth_object="cortmask")), "morphometry.ctth_object: 'cortmask'"),
                       (dict(bmd=BMDParams(masks="rendred")), "bmd.masks: 'rendred'"),
                       (dict(porosity=PoreParams(grid="IPL")), "porosity.grid: 'IPL'"),
                       (dict(seg=SegParams(overlap="cortex")), "seg.overlap: 'cortex'"),
                       (dict(output=OutputParams(mask_value=300)), "must be <= 255"),
                       (dict(dt=engine.DTParams(version=4)), "dt.version: 4"),
                       (dict(lh=LHParams(norm_max=0.0)), "lh.norm_max: must be > 0")):
        with pytest.raises(ParameterError, match=match):
            Parameters().replace(**bad)
        with pytest.raises(ParameterError, match=match):
            Parameters(**bad)
    P = Parameters().replace(lh=LHParams(laplace_eps="0.5"))
    assert P.lh.laplace_eps == 0.5 and type(P.lh.laplace_eps) is float
    assert pm.resolve("tibia", P).block()["non_default_values"]["lh.laplace_eps"]["value"] == 0.5
    with pytest.raises(ParameterError, match="stage 'seg' must be a SegParams"):
        Parameters(seg=LHParams())
    # the engine's dispatches
    grey, per, Gc, Gt = lh_phantom
    with pytest.raises(ValueError, match="seg.overlap"):
        engine.lh_segment(grey, EL, per, Gc, Gt, seg=SegParams(overlap="cortex"))
    with pytest.raises(ValueError, match="seg.trab_seg_mask"):
        engine.lh_segment(grey, EL, per, Gc, Gt, seg=SegParams(trab_seg_mask="yes"))
    with pytest.raises(ValueError, match="seg.periosteal_mask"):
        engine.periosteal_for_seg("rendred", raw=per, rendered=per)
    with pytest.raises(ValueError, match="ctth_object"):
        _dt(dt_phantom, morph=MorphometryParams(ctth_object="cortmask"))
    with pytest.raises(ValueError, match="tbth_grid"):
        _dt(dt_phantom, morph=MorphometryParams(tbth_grid="box"))
    G_cort, cort_seg, grid = pore_inputs
    with pytest.raises(ValueError, match="porosity.grid"):
        engine.step5c_porosity(G_cort, cort_seg, grid.dim, grid.pos, params=PoreParams(grid="IPL"))


@pytest.mark.parametrize("item", ["lh.norm_max=0", "calibration.slope=0", "calibration.mu_scaling=0", "calibration.mu_water=0",
                                  "morphometry.voxel_size_mm=0", "lh.lp_cut_off_freq=0", "step1.sigma=0",
                                  "autocontour.peri_s1_sigma=0", "calibration.slope=-1"])
def test_divisors_have_an_exclusive_lower_bound(item):
    with pytest.raises(ParameterError, match="must be > 0"):
        pm.parse_set(item)


def test_min_vertices_three_is_accepted_and_flagged_as_refuted():
    notes = Parameters().override({"render.min_vertices": 3}).unverified()
    assert notes and "3 and 7..8 refuted" in notes[0]
    assert "3 and 7..8 are refuted" in pm.SCHEMA_SPECS["render"]["min_vertices"].doc


def test_step1_record_keeps_the_report_keys():
    """The report / info record of STEP 1 is the sixteen preset fields, plus a script literal only when it differs
    from the script's value: a default report has exactly the keys it had before the literals became parameters."""
    from ipldt.step1 import PRESET_FIELDS
    for p in (TIBIA, RADIUS):
        assert list(p.record()) == list(PRESET_FIELDS) and Step1Params(**p.record()) == p
    q = replace(TIBIA, continuous_z=1, rank_last=2)
    assert list(q.record()) == list(PRESET_FIELDS) + ["rank_last", "continuous_z"] and Step1Params(**q.record()) == q


def test_default_report_keeps_its_format_and_the_batch_column_is_last(default_run, tmp_path, monkeypatch):
    """(1) parameters.step1 / step1.params of a default run carry the sixteen keys; (2) batch_summary.csv of a default
    batch has the columns it always had, and a non-default batch adds non_default_parameters as the LAST column;
    (3) the log words the element sizes as before."""
    import csv
    from ipldt.step1 import PRESET_FIELDS
    from ormir_bqrl import cli
    from ormir_bqrl.report import summary_row
    rep = default_run
    assert list(rep["parameters"]["step1"]) == list(PRESET_FIELDS) and list(rep["step1"]["params"]) == list(PRESET_FIELDS)
    assert "non_default_parameters" not in summary_row(rep)
    log = open(rep["outputs"]["log"], encoding="utf-8").read()
    assert "(AIM header, per axis, IPL's rule)" in log and "(AIM header, per axis (IPL's rule))" not in log
    aim = tmp_path / "S.AIM"
    aim.write_bytes(b"\0" * 64)
    nd = []

    def fake(aim_path, out_dir, **kw):
        r = json.loads(json.dumps(rep))
        r["run"]["out_dir"] = str(out_dir)
        r["parameter_set"]["non_default"] = list(nd)
        return r
    monkeypatch.setattr(cli, "_import_pipeline", lambda: fake)
    assert cli.main(["batch", str(tmp_path / "b0"), str(aim)]) == 0
    head0 = next(csv.reader(open(tmp_path / "b0" / "batch_summary.csv", encoding="utf-8")))
    assert "non_default_parameters" not in head0 and head0[-1] == "error"
    nd[:] = ["lh.laplace_eps"]
    assert cli.main(["batch", str(tmp_path / "b1"), str(aim), "--set", "lh.laplace_eps=0.5"]) == 0
    rows = list(csv.reader(open(tmp_path / "b1" / "batch_summary.csv", encoding="utf-8")))
    assert rows[0][:-1] == head0 and rows[0][-1] == "non_default_parameters" and rows[1][-1] == "lh.laplace_eps"


def test_explicit_sizes_equal_to_the_header_are_the_default(phantom_aim, tmp_path):
    """A voxel size / Laplace-Hamming element size given explicitly with the header's own value computes exactly what
    the default computes, and is reported as such (equal_to_default), not as a non-default value."""
    import test_ormir_bqrl_porosity as tp
    aim, disc = phantom_aim
    ref = engine.run_pipeline(aim, str(tmp_path / "ref"), compute_bmd=False, log=engine.Logger(echo=False), periosteal=disc)
    el = engine.read_aim(aim)["el_size_mm"]
    got = engine.run_pipeline(aim, str(tmp_path / "eq"), compute_bmd=False, log=engine.Logger(echo=False), periosteal=disc,
                              voxel_size_mm=el[0], lh_voxel_size_mm=list(el))
    ps = got["parameter_set"]
    assert ps["non_default"] == [] and ps["validated_default"] is True
    assert ps["equal_to_default"] == ["morphometry.voxel_size_mm", "lh.el_size_mm"]
    assert got["morphometry"] == ref["morphometry"] and got["porosity"] == ref["porosity"]
    other = engine.run_pipeline(aim, str(tmp_path / "ne"), compute_bmd=False, log=engine.Logger(echo=False), periosteal=disc,
                                voxel_size_mm=tp.EL[0] * 1.01)
    assert other["parameter_set"]["non_default"] == ["morphometry.voxel_size_mm"]


def test_values_of_steps_that_do_not_run_are_not_applied(default_run, phantom_aim, tmp_path):
    """autocontour.* with a periosteal given, bmd.* without BMD, porosity.* without the cascade: warned about, reset to
    the default, listed as not_applied and not counted as non-default; the outputs are the default run's."""
    from ormir_bqrl.pipeline import run
    aim, disc = phantom_aim
    with pytest.warns(pm.ParameterWarning, match="autocontour.peri_s1_lower = 500 not applied"):
        rep = run(aim, str(tmp_path / "na"), periosteal=disc, compute_bmd=False, preview=False, log=engine.Logger(echo=False),
                  parameters={"autocontour.peri_s1_lower": 500, "bmd.masks": "raw", "seg.cc_min_trab": 71})
    ps = rep["parameter_set"]
    assert ps["not_applied"] == ["autocontour.peri_s1_lower", "bmd.masks"] and ps["non_default"] == ["seg.cc_min_trab"]
    assert ps["values"]["autocontour"]["peri_s1_lower"] == 350.0 and ps["values"]["bmd"]["masks"] == "rendered"
    assert ps["not_applied_values"]["autocontour.peri_s1_lower"] == {
        "requested": 500.0, "used": 350.0, "reason": "the ORMIR-XCT autocontour did not run (the periosteal contour was "
                                                     "given or taken from the run); the default is used"}
    log = open(rep["outputs"]["log"], encoding="utf-8").read()
    assert "WARNING: autocontour.peri_s1_lower = 500 not applied" in log
    md = open(rep["outputs"]["report_md"], encoding="utf-8").read()
    assert "- **not applied:** autocontour.peri_s1_lower = 500" in md
    saved = json.load(open(rep["outputs"]["parameters"], encoding="utf-8"))
    assert saved["not_applied"]["bmd.masks"]["requested"] == "raw"
    only = run(aim, str(tmp_path / "na2"), periosteal=disc, compute_bmd=False, compute_porosity=False, preview=False,
               log=engine.Logger(echo=False), parameters={"autocontour.peri_s4_close_radius": 2, "porosity.min_pore_voxels": 5})
    assert only["parameter_set"]["non_default"] == [] and only["parameter_set"]["validated_default"] is True
    assert only["parameter_set"]["not_applied"] == ["autocontour.peri_s4_close_radius", "porosity.min_pore_voxels"]
    assert all(_same_outputs(default_run, only, [n for n in VOLS if n != "PORE"]).values())


def test_compartment_redo_does_not_apply_step1_or_autocontour_overrides(phantom_aim, tmp_path):
    from ormir_bqrl import redo as redo_mod
    from ormir_bqrl.pipeline import run
    aim, disc = phantom_aim
    out = str(tmp_path / "run")
    rep = run(aim, out, periosteal=disc, compute_bmd=False, preview=False, log=engine.Logger(echo=False),
              parameters={"step1.close2": 40})
    labels = _vol(rep["outputs"]["labelmap"])
    T = labels == 2
    base = redo_mod.run_from_masks(aim, out, str(tmp_path / "t0"), trab=T, compute_bmd=False, preview=False,
                                   log=engine.Logger(echo=False))
    with pytest.warns(pm.ParameterWarning, match="not applied: STEP 1 did not run"):
        r = redo_mod.run_from_masks(aim, out, str(tmp_path / "t1"), trab=T, compute_bmd=False, preview=False,
                                    log=engine.Logger(echo=False),
                                    parameters={"step1.close2": 10, "autocontour.peri_s1_lower": 100})
    ps = r["parameter_set"]
    assert ps["not_applied"] == ["autocontour.peri_s1_lower", "step1.close2"]
    assert ps["not_applied_values"]["step1.close2"]["used"] == 40 and ps["values"]["step1"]["close2"] == 40
    assert ps["non_default"] == ["step1.close2"] == base["parameter_set"]["non_default"], "the run's own value is kept"
    assert r["site"] == "custom" and all(_same_outputs(base, r).values())
    assert "no autocontour" in ps["statement"]


def test_periosteal_redo_and_rerun_keep_the_run_preset(phantom_aim, tmp_path):
    """A radius run with a custom step1: its periosteal redo and a rerun from its <base>_parameters.json are compared
    with the radius preset (non_default ['step1.close2'] only, the Script 33 statement)."""
    from ormir_bqrl import redo as redo_mod
    from ormir_bqrl.pipeline import run
    aim, disc = phantom_aim
    out = str(tmp_path / "rad")
    rep = run(aim, out, site="radius", periosteal=disc, compute_bmd=False, preview=False, log=engine.Logger(echo=False),
              parameters={"step1.close2": 20})
    assert rep["parameter_set"]["preset"] == "radius" and rep["parameter_set"]["non_default"] == ["step1.close2"]
    P = _vol(rep["outputs"]["labelmap"]) > 0
    P[:, _r() > 38] = False
    red = redo_mod.run_from_masks(aim, out, str(tmp_path / "peri"), periosteal=P, compute_bmd=False, preview=False,
                                  log=engine.Logger(echo=False))
    ps = red["parameter_set"]
    assert ps["preset"] == "radius" and ps["non_default"] == ["step1.close2"] and "Script 33 radius" in ps["statement"]
    assert red["site"] == "custom" and ps["values"]["step1"]["corner_min"] == 800
    R = pm.resolve(None, rep["outputs"]["parameters"], workflow="ormir_bqrl")
    assert R.preset == "radius" and R.non_default == ["step1.close2"] and R.params.step1 == replace(RADIUS, close2=20)
    again = run(aim, str(tmp_path / "again"), periosteal=disc, compute_bmd=False, preview=False, log=engine.Logger(echo=False),
                parameters=rep["outputs"]["parameters"])
    assert all(_same_outputs(rep, again).values()) and again["parameter_set"]["values"] == rep["parameter_set"]["values"]


def test_saved_and_printed_sets_are_neutral(tmp_path, capsys):
    """A set printed / saved as the defaults changes nothing when it is fed back: an explicit --site wins over its
    preset, and the other workflow does not inherit seg.periosteal_mask."""
    from ipldt.workflows import ipl_pipeline_workflow as wf
    from ormir_bqrl import cli
    my = tmp_path / "my.json"
    assert cli.main(["params", "--set", "lh.laplace_eps=0.5", "--out", str(my)]) == 0
    capsys.readouterr()
    assert cli.main(["params", "--site", "radius", "--params", str(my), "--set", "porosity.slice_fraction=0,10"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["site"] == "radius" and d["values"]["step1"]["close2"] == 30
    assert d["non_default"] == ["lh.laplace_eps", "porosity.slice_up"]
    assert cli.main(["params", "--params", str(my)]) == 0
    assert json.loads(capsys.readouterr().out)["site"] == "tibia"
    rad = tmp_path / "rad.json"
    assert cli.main(["params", "--site", "radius", "--out", str(rad)]) == 0
    capsys.readouterr()
    assert cli.main(["params", "--params", str(rad)]) == 0
    assert json.loads(capsys.readouterr().out)["site"] == "radius", "without --site, a complete set's own preset"
    bq = tmp_path / "bq.json"
    assert cli.main(["params", "--out", str(bq)]) == 0
    capsys.readouterr()
    assert wf.main(["--print-params", "--params", str(bq)]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["non_default"] == [] and d["values"]["seg"]["periosteal_mask"] == "raw"
    ip = tmp_path / "ip.json"
    assert wf.main(["--print-params"]) == 0
    ip.write_text(capsys.readouterr().out)
    assert cli.main(["params", "--params", str(ip)]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["non_default"] == [] and d["values"]["seg"]["periosteal_mask"] == "rendered"


def test_statement_names_what_was_validated():
    s = pm.statement([], "tibia", {"autocontour", "step1", "bmd", "porosity", "masks"})
    assert s.startswith("the validated IPL defaults (Script 32 tibia preset)")
    assert "IPL-derived stages" in s and "the periosteal autocontour and BMD are ORMIR-XCT's, which were not validated against IPL" in s
    s = pm.statement([], "radius", {"step1", "masks"})
    assert "Script 33 radius" in s and "the periosteal contour was given (no autocontour)" in s and "BMD" not in s
    s = pm.statement(["lh.laplace_eps"], "tibia", {"step1", "bmd"})
    assert s.startswith("differs from the validated IPL defaults") and "BMD is ORMIR-XCT's, which was not validated" in s
    assert "not compared with IPL" in pm.resolve().block()["validated_scope"]
    # the deliberate departures from IPL's evaluation are named, and IPL's own value is not called a departure
    bq, ip = pm.resolve(workflow="ormir_bqrl"), pm.resolve(workflow="ipldt")
    assert "except that Tb.Th is taken on the whole SEG" in bq.statement and "periosteal raster" not in bq.statement
    assert "Tb.Th is taken on the whole SEG" in ip.statement and "SEG is masked with the periosteal raster" in ip.statement
    assert "morphometry.tbth_object trab_seg" in pm.VALIDATED_SCOPE and "seg.periosteal_mask rendered" in pm.VALIDATED_SCOPE
    for wf, name, value in (("ormir_bqrl", "morphometry.tbth_object", "trab_seg"), ("ipldt", "seg.periosteal_mask", "rendered")):
        R = pm.resolve(parameters={name: value}, workflow=wf)
        assert R.non_default == [name] and f"{name} ({value}: IPL's own choice, the one compared with IPL)" in R.statement
        assert "results are not the validated IPL-equivalent configuration" not in R.statement
    R = pm.resolve(parameters={"morphometry.tbth_object": "trab_seg", "morphometry.maps": ["TRAB_SP"]}, workflow="ipldt")
    assert "results are not the validated" in R.statement and "Tb.Th is taken" not in R.statement


def test_calibration_overrides_recalibrate_the_hu_image(phantom_aim, tmp_path):
    """No override: the reader's HU image and header calibration, the same objects.  mu_scaling / mu_water that
    differ from the header's: HU recomputed from the native data with the reader's own rule (which reproduces the
    reader exactly with the header's values), so a mu_scaling override recalibrates the autocontour and BMD too."""
    import SimpleITK as sitk
    from ormir_bqrl import stages
    aim, disc = phantom_aim
    loaded = stages.load(aim)
    c = loaded.calib
    assert np.array_equal(engine.hu_from_native(loaded.native["data"], c["mu_scaling"], c["mu_water"]),
                          sitk.GetArrayFromImage(loaded.hu_int16))
    hu, cal = engine.calibrated_hu(loaded.img_hu, loaded.native, c, pm.CalibrationParams())
    assert hu is loaded.img_hu and cal is c
    hu, cal = engine.calibrated_hu(loaded.img_hu, loaded.native, c, pm.CalibrationParams(slope=1500.0))
    assert hu is loaded.img_hu and cal["rescale_slope"] == 1500.0
    hu, cal = engine.calibrated_hu(loaded.img_hu, loaded.native, c, pm.CalibrationParams(mu_scaling=c["mu_scaling"] * 2))
    got = sitk.GetArrayFromImage(hu)
    assert cal["mu_scaling"] == c["mu_scaling"] * 2
    assert np.array_equal(got, engine.hu_from_native(loaded.native["data"], c["mu_scaling"] * 2, c["mu_water"]).astype(np.float32))
    rep = engine.run_pipeline(aim, str(tmp_path / "cal"), log=engine.Logger(echo=False), periosteal=disc,
                              parameters={"calibration.mu_scaling": c["mu_scaling"] * 2})
    ref = engine.run_pipeline(aim, str(tmp_path / "ref"), log=engine.Logger(echo=False), periosteal=disc)
    assert rep["calibration_effective"]["mu_scaling"] == c["mu_scaling"] * 2 and "calibration_effective" not in ref
    assert rep["bmd"]["Tb_BMD_mgHA_cm3"] != ref["bmd"]["Tb_BMD_mgHA_cm3"]
    assert rep["step1"]["thresholds"] != ref["step1"]["thresholds"]
    # mu_water: in ipldt-pipeline with the periosteal given and no BMD nothing reads HU -> not applied
    with pytest.warns(pm.ParameterWarning, match="calibration.mu_water = 0.3 not applied"):
        na = engine.run_pipeline(aim, str(tmp_path / "mw"), compute_bmd=False, log=engine.Logger(echo=False), periosteal=disc,
                                 parameters={"calibration.mu_water": 0.3})
    assert na["parameter_set"]["non_default"] == [] and na["parameter_set"]["not_applied"] == ["calibration.mu_water"]


def test_dt_commands_help_and_a_complete_parameter_file(tmp_path, capsys):
    import argparse
    from ipldt.workflows._cli import add_dt_arguments, dt_params_from_args
    ap = argparse.ArgumentParser()
    add_dt_arguments(ap)
    h = " ".join(ap.format_help().split())
    assert "--set dt.ridge_epsilon=0.7" in h and "lh.laplace_eps" not in h and "step1.close2" not in h
    assert "dedicated flags" not in h and "site preset" not in h
    full = tmp_path / "run_parameters.json"
    full.write_text(json.dumps(pm.printed_set(pm.resolve("radius", {"lh.laplace_eps": 0.5, "dt.ridge_epsilon": 0.7}))))
    notes = []
    p = dt_params_from_args(ap.parse_args(["--params", str(full)]), notes=notes)
    assert p.ridge_epsilon == 0.7 and notes and "takes its dt block only" in notes[0] and "lh.laplace_eps" in notes[0]
    with pytest.raises(ParameterError, match="dt stage only"):
        dt_params_from_args(ap.parse_args(["--set", "lh.laplace_eps=0.5"]))
    part = tmp_path / "part.json"
    part.write_text(json.dumps({"lh.laplace_eps": 0.5}))
    with pytest.raises(ParameterError, match="dt stage only"):
        dt_params_from_args(ap.parse_args(["--params", str(part)]))
    from ormir_bqrl import cli
    sub = next(a for a in cli.build_parser()._actions if a.dest == "command")
    assert "dedicated flags" not in sub.choices["run"].format_help()


def test_redo_outside_mask_uses_the_mask_value(phantom_aim, tmp_path):
    from ormir_bqrl import redo as redo_mod
    from ormir_bqrl.pipeline import run
    aim, disc = phantom_aim
    out = str(tmp_path / "run")
    run(aim, out, periosteal=disc, compute_bmd=False, compute_porosity=False, preview=False, log=engine.Logger(echo=False),
        parameters={"output.mask_value": 255})
    T = np.zeros(disc.shape, bool)
    T[:, 0:3, 0:3] = True                                        # outside the periosteal disc
    with pytest.raises(redo_mod.EditOutsidePeriosteal):
        redo_mod.run_from_masks(aim, out, str(tmp_path / "out"), trab=T, compute_bmd=False, compute_porosity=False,
                                preview=False, log=engine.Logger(echo=False))
    m = _vol(str(tmp_path / "out" / "PHANTOM_redo_outside.nii.gz"))
    assert set(np.unique(m).tolist()) == {0, 255}
