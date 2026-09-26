"""ORMIR-BQRL `run()` and `run_from_masks()` on a synthetic AIM (design sections 5, 6, 9).

The phantom (written as a real v020 short AIM with VAX-float element sizes and a calibration processing log, so
the file goes through ITK ScancoImageIO, ORMIR's file_reader and ipldt.io.read_aim like a scan): a cylinder r <= 40
on 40 slices with a dense shell 32 <= r <= 40 (native 12000), marrow 1000, outside 300, plus a rod / plate structure
of filtered noise (native 9000) inside r < 28 so the Laplace-Hamming threshold and the dt stage see trabeculae.
The periosteal disc has a half-integer centre, so IPL's contour rendering is the identity on it: the engine-parity
test against ipldt.ormir.run_pipeline is exact and the labelmap union equals the injected raster.

Skips without SimpleITK / itk / ormir_xct (step1_load_aim needs them; no test calls the autocontour).  The GPU is
never required (backend 'auto').  The slow test needs the XCT_masks_full_grab folder (PFJ-0be66a_R).
"""
import json
import math
import os
import shutil
import struct

import numpy as np
import pytest
from scipy import ndimage as ndi

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("itk")
pytest.importorskip("ormir_xct")

from ipldt import ormir as engine                                                  # noqa: E402
from ipldt.io import align_to, read_aim, write_nifti                               # noqa: E402
from ipldt.step1 import Step1Params                                                # noqa: E402

import ormir_bqrl                                                                  # noqa: E402
from ormir_bqrl import redo as redo_mod                                            # noqa: E402
from ormir_bqrl import report as report_mod                                        # noqa: E402
from ormir_bqrl import slicer, stages                                              # noqa: E402
from ormir_bqrl.pipeline import run                                                # noqa: E402
from ormir_bqrl.redo import EditOutsidePeriosteal, derive_compartments, run_from_masks   # noqa: E402

SHAPE = (40, 90, 90)                      # (z, y, x)
DIM = SHAPE[::-1]
POS = (100, 200, 300)
EL = (0.06069973, 0.06069973, 0.06069973)
CENTRE = 44.5
R_OUT, R_IN, R_ROD = 40.0, 32.0, 28.0
PROCLOG = ("! Processing Log\n"
           "Mu_Scaling                                       8192\n"
           "Mu_Water                                       0.2409\n"
           "Density: unit                                  mg HA/ccm\n"
           "Density: slope                         1.61907703e+03\n"
           "Density: intercept                    -3.94095001e+02\n")
MASKS = ("PRX_MASK", "PRX_GOBJ", "CORT_MASK", "TRAB_MASK", "CORT_GOBJ", "TRAB_GOBJ", "CORT_SEG", "TRAB_SEG")
MAPS = ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH")
REPORT_KEYS = ("schema", "product", "run", "sample", "input_aim", "site", "parameters", "masks", "edits", "grids", "step1",
               "compartments", "morphometry", "bmd", "porosity", "summary", "outputs", "timing_s")


# ------------------------------------------------------------------------------------------- the phantom
def _radius():
    yy, xx = np.mgrid[:SHAPE[1], :SHAPE[2]]
    return np.sqrt((yy - CENTRE) ** 2 + (xx - CENTRE) ** 2)


def _vax_pack(x):
    """The inverse of ipldt.io._vax_f: value = (-1)^s (0.5 + f / 2^24) 2^(e - 128), word-swapped."""
    if x == 0:
        return 0
    m, e = math.frexp(x)
    f = int(round((m - 0.5) * 2 ** 24))
    w1 = ((e + 128) << 7) | (f >> 16)
    w2 = f & 0xFFFF
    u = (w2 << 16) | w1
    return u - 2 ** 32 if u >= 2 ** 31 else u


def phantom_grey(seed=0):
    """native int16 (z, y, x): shell 12000, marrow 1000, outside 300, rods 9000 inside r < R_ROD."""
    r = _radius()
    grey2d = np.where(r <= R_OUT, np.where(r >= R_IN, 12000, 1000), 300).astype(np.int16)
    grey = np.broadcast_to(grey2d, SHAPE).copy()
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.standard_normal(SHAPE), 2.0)
    rods = (f > np.quantile(f, 1.0 - 0.35)) & np.broadcast_to(r < R_ROD, SHAPE)
    grey[rods] = 9000
    return grey


def make_aim(path, seed=0):
    """A v020 short AIM (20-byte pre-header, 35-int header with the type, pos, dim and VAX-float element sizes,
    the processing log, the data), readable by ITK ScancoImageIO, ORMIR's file_reader and ipldt.io.read_aim."""
    data = phantom_grey(seed)
    ints = [0] * 35
    ints[0] = 16
    ints[5] = 0x00020002
    ints[6:9] = POS
    ints[9:12] = DIM
    ints[27:30] = [_vax_pack(e) for e in EL]
    hdr = struct.pack("<35i", *ints)
    log = PROCLOG.encode("latin-1")
    body = data.tobytes()
    raw = struct.pack("<5i", 20, len(hdr), len(log), len(body), 0) + hdr + log + body
    raw += b"\0" * ((-len(raw)) % 512)
    with open(path, "wb") as fh:
        fh.write(raw)
    return path


def periosteal_disc():
    """r <= R_OUT on every slice (half-integer centre: IPL's rendering is the identity on it)."""
    return np.broadcast_to(_radius() <= R_OUT, SHAPE).copy()


def edit_labelmap(labels, kind):
    """Slicer-style edits of the ORMIR-BQRL labelmap (1 cortical, 2 trabecular)."""
    lab = np.array(labels, np.uint8, copy=True)
    r = _radius()
    if kind == "thicken_trab":            # the innermost 3 voxels of the cortical ring become trabecular, slices 10..29
        for z in range(10, 30):
            inner = (lab[z] == 1) & ndi.binary_dilation(lab[z] == 2, iterations=3)
            lab[z][inner] = 2
    elif kind == "thicken_cort":          # the outermost 3 voxels of the trabecular disc become cortical
        for z in range(10, 30):
            outer = (lab[z] == 2) & ndi.binary_dilation(lab[z] == 1, iterations=3)
            lab[z][outer] = 1
    elif kind == "shrink_periosteal":     # erase r > 38 on all slices (the union shrinks)
        lab[:, r > 38] = 0
    elif kind == "overshoot":             # trabecular painted outside the bone on 5 slices
        for z in range(15, 20):
            lab[z][(r > R_OUT) & (r <= 42)] = 2
    else:
        raise ValueError(kind)
    return lab


# ------------------------------------------------------------------------------------------- helpers
def vol(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def mask(path):
    return vol(path) > 0


def grid_of(report):
    ia = report["input_aim"]
    return stages.Grid(tuple(ia["dim_xyz"]), tuple(ia["pos_xyz"]), tuple(ia["el_size_mm"]))


def outputs_equal(rep_a, rep_b, names=MAPS + ("SEG",)):
    for n in names:
        assert np.array_equal(vol(rep_a["outputs"][n]), vol(rep_b["outputs"][n])), n


# ------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def aim_path(tmp_path_factory):
    return make_aim(str(tmp_path_factory.mktemp("aim") / "PHANTOM.AIM"))


@pytest.fixture(scope="module")
def disc():
    return periosteal_disc()


@pytest.fixture(scope="module")
def run_out(aim_path, disc, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("run") / "PHANTOM")
    report = run(aim_path, out, site="tibia", periosteal=disc, compute_bmd=False, log=engine.Logger(echo=False))
    return out, report


def _copy_run(run_out, tmp_path, name):
    """A private copy of the module's run folder (without the redo folders other tests may have added to it)."""
    out, _ = run_out
    dst = str(tmp_path / name)
    shutil.copytree(out, dst, ignore=shutil.ignore_patterns("redo*"))
    return dst


# ------------------------------------------------------------------------------------------- run()
def test_phantom_aim_reads_like_a_scan(aim_path):
    a = read_aim(aim_path)
    assert a["dim"] == DIM and a["pos"] == POS and a["data"].dtype == np.int16
    assert np.allclose(a["el_size_mm"], EL, rtol=1e-6)
    assert np.array_equal(a["data"], phantom_grey())
    assert "Density: slope" in a["proclog"]


def test_output_set_and_mask_invariants(run_out, disc):
    out, rep = run_out
    base = rep["sample"]
    assert base == "PHANTOM"
    for n in ("HU",) + MASKS + ("SEG",) + MAPS:
        p = rep["outputs"][n]
        assert p == os.path.abspath(os.path.join(out, f"{base}_{n}.nii.gz")) and os.path.exists(p), n
    for n in ("seg_nrrd", "labelmap", "report_json", "report_csv", "report_md", "log"):
        assert os.path.exists(rep["outputs"][n]), n
    assert os.path.exists(rep["outputs"]["preview"]) and rep["outputs"]["preview"].endswith("_preview.png")
    # dtypes and values
    hu = sitk.ReadImage(rep["outputs"]["HU"])
    assert hu.GetPixelID() == sitk.sitkInt16 and hu.GetSize() == DIM
    assert np.allclose(hu.GetOrigin(), [p * e for p, e in zip(POS, EL)], atol=1e-4) and np.allclose(hu.GetSpacing(), EL, rtol=1e-6)
    for n in MASKS:
        v = vol(rep["outputs"][n])
        assert v.dtype == np.uint8 and v.shape == SHAPE and set(np.unique(v)) <= {0, 127}, n
    seg = vol(rep["outputs"]["SEG"])
    assert seg.dtype == np.uint8 and set(np.unique(seg)) == {0, 126, 127}
    for n in MAPS:
        v = vol(rep["outputs"][n])
        assert v.dtype == np.int16 and v.shape == SHAPE and v.min() >= 0 and v.max() > 0, n
    # the mask algebra
    prx, ALL = mask(rep["outputs"]["PRX_MASK"]), mask(rep["outputs"]["PRX_GOBJ"])
    cort, trab = mask(rep["outputs"]["CORT_MASK"]), mask(rep["outputs"]["TRAB_MASK"])
    assert np.array_equal(prx, disc)
    assert np.array_equal(ALL, prx), "the disc renders to itself"
    assert np.array_equal(cort | trab, ALL) and not (cort & trab).any()
    assert cort.any() and trab.any()
    assert np.array_equal(mask(rep["outputs"]["CORT_SEG"]) | mask(rep["outputs"]["TRAB_SEG"]), seg > 0)
    assert (mask(rep["outputs"]["TRAB_SEG"]) <= mask(rep["outputs"]["TRAB_GOBJ"])).all()
    assert (mask(rep["outputs"]["CORT_SEG"]) <= mask(rep["outputs"]["CORT_GOBJ"])).all()
    # the Slicer files
    labels = vol(rep["outputs"]["labelmap"])
    assert labels.dtype == np.uint8 and np.array_equal(labels, cort.astype(np.uint8) * 1 + trab.astype(np.uint8) * 2)
    nr = sitk.ReadImage(rep["outputs"]["seg_nrrd"])
    assert np.array_equal(sitk.GetArrayFromImage(nr), labels)
    assert np.allclose(nr.GetOrigin(), hu.GetOrigin(), atol=1e-4) and np.allclose(nr.GetSpacing(), EL, rtol=1e-6)
    c2, t2 = slicer.masks_from_labels(labels)
    assert np.array_equal(c2, cort) and np.array_equal(t2, trab)
    # a ring around a disc on every slice with rods inside
    for z in range(SHAPE[0]):
        assert ndi.label(cort[z], structure=ndi.generate_binary_structure(2, 1))[1] == 1
        assert trab[z, 44, 44] and not cort[z, 44, 44]
    assert (seg[:, _radius() < R_ROD] == 126).any(), "the rods are trabecular bone in SEG"


def test_report_schema_and_files(run_out, aim_path):
    out, rep = run_out
    assert tuple(rep) == REPORT_KEYS
    assert rep["schema"] == report_mod.SCHEMA == "ormir-bqrl/1"
    assert rep["product"]["name"] == "ORMIR-BQRL" and rep["product"]["version"] == ormir_bqrl.__version__
    assert rep["product"]["ipldt_version"] == engine.__version__
    assert rep["run"]["kind"] == "run" and rep["run"]["derived_from"] is None and rep["run"]["out_dir"] == os.path.abspath(out)
    assert rep["run"]["duration_s"] > 0 and rep["run"]["command"] is None
    assert rep["input_aim"]["path"] == os.path.abspath(aim_path) and len(rep["input_aim"]["sha256"]) == 64
    assert rep["input_aim"]["dim_xyz"] == list(DIM) and rep["input_aim"]["pos_xyz"] == list(POS)
    assert rep["input_aim"]["calibration_source"] == "proclog"
    assert rep["site"] == "tibia"
    P = rep["parameters"]
    assert P["step1"] == dict(engine.TIBIA.__dict__)
    assert P["laplace_hamming"]["threshold"] == 15564 and len(P["laplace_hamming"]["el_size_mm"]) == 3
    assert P["seg_assembly"]["cl_nr_extract_min_cort"] == 35 and P["seg_assembly"]["cl_nr_extract_min_trab"] == 70
    assert P["seg_assembly"]["periosteal_mask"] == "rendered contour (ALL)"
    assert P["dt"]["ridge_epsilon"] == 0.9 and P["dt"]["backend"] in ("gpu", "cpu") and P["dt"]["voxel_size_mm"] == rep["input_aim"]["el_size_mm"][0]
    assert P["bmd"]["computed"] is False and P["map_units"] == "voxels"
    m = rep["masks"]
    assert m["periosteal"]["source"] == "array" and m["periosteal"]["manual"] is False
    assert m["periosteal"]["raw_voxels"] == m["periosteal"]["rendered_voxels"] == int(periosteal_disc().sum())
    assert m["cortical"]["source"] == "ipldt.step1" and m["trabecular"]["source"] == "ipldt.step1"
    assert m["cortical"]["voxels"] == rep["compartments"]["CORT_MASK_voxels"] and m["trabecular"]["rendered_voxels"] == rep["compartments"]["TRAB_GOBJ_voxels"]
    assert rep["edits"] is None
    assert set(rep["grids"]) == {"AIM", "SEG", "CORT_MASK"}
    assert rep["step1"]["site"] == "tibia" and rep["step1"]["thresholds"] == dict(lower_native=4524, upper_native=17173, lower_mgha=500.0, upper_mgha=3000.0)
    assert "29_trabfinal" in rep["step1"]["counts"] and rep["step1"]["peel_counts"]["6"] > 0
    c = rep["compartments"]
    assert c["PRX_MASK_voxels"] == c["PRX_GOBJ_voxels"] == c["CORT_MASK_voxels"] + c["TRAB_MASK_voxels"]
    assert c["SEG_voxels"] > 0 and c["CORT_SEG_voxels"] > 0 and c["TRAB_SEG_voxels"] > 0
    mm = rep["morphometry"]
    s = rep["summary"]
    for k in report_mod.SUMMARY_KEYS:
        assert s[k] == mm[k] and np.isfinite(s[k]) and s[k] > 0
    assert s["Tb_BMD_mgHA_cm3"] is None and rep["bmd"] == {}
    assert abs(mm["Tb_N_per_mm"] - 1.0 / mm["Tb_1N_mm"]) < 1e-9
    assert mm["BV_TV"] == mm["BV_voxels"] / mm["TV_voxels"]
    assert set(rep["timing_s"]) >= {"1_load", "2_periosteal", "3_compartments", "4_render_segment", "5_dt", "total", "dt_TRAB_TH", "dt_CORT_TH"}
    # the JSON on disk equals the dict; the CSV and MD carry the headline numbers
    with open(rep["outputs"]["report_json"], "r", encoding="utf-8") as fh:
        assert json.load(fh) == rep
    csv_text = open(rep["outputs"]["report_csv"], encoding="utf-8").read()
    assert csv_text.startswith("Parameter,Value") and "morphometry.Tb_Th_mm" in csv_text
    md = open(rep["outputs"]["report_md"], encoding="utf-8").read()
    assert "# ORMIR-BQRL report: PHANTOM" in md and "| Tb.Th |" in md and "ipldt.step1" in md
    row = report_mod.summary_row(rep)
    assert row["sample"] == "PHANTOM" and row["kind"] == "run" and row["periosteal_source"] == "array" and row["Tb_Th_mm"] == mm["Tb_Th_mm"]
    log = open(rep["outputs"]["log"], encoding="utf-8").read()
    assert "STEP 3" in log and "DONE PHANTOM (run)" in log


def test_engine_parity_with_run_pipeline(run_out, aim_path, disc, tmp_path, monkeypatch):
    """Bit-identical to ipldt.ormir.run_pipeline fed the same periosteal (the disc renders to itself, so masking
    the SEG assembly with the rendered contour or the raw raster is the same); and stages.segment without gobjs
    equals step4_render_and_segment field by field."""
    out, rep = run_out
    monkeypatch.setattr(engine, "step2_autocontour", lambda img, calib, log=None: engine.array_to_sitk(disc.astype(np.uint8), img))
    out2 = str(tmp_path / "engine")
    rp = engine.run_pipeline(aim_path, out2, site="tibia", compute_bmd=False, log=engine.Logger(echo=False))
    for n in MAPS + ("SEG", "CORT_MASK", "TRAB_MASK", "CORT_GOBJ", "TRAB_GOBJ", "PRX_GOBJ"):
        assert np.array_equal(vol(rep["outputs"][n]), vol(os.path.join(out2, f"PHANTOM_{n}.nii.gz"))), n
    assert rp["morphometry"] == rep["morphometry"]
    assert rp["compartments"] == rep["compartments"]
    assert rp["grids"] == {k: rep["grids"][k] for k in ("SEG", "CORT_MASK")}
    loaded = stages.load(aim_path)
    cort, trab, ALL, _ = stages.compartments(loaded, disc, engine.TIBIA)
    segm, G_cort, G_trab = stages.segment(loaded, ALL, cort, trab)
    s4 = engine.step4_render_and_segment(loaded.native, stages.mask_image(disc, loaded.grid), stages.mask_image(cort, loaded.grid),
                                         stages.mask_image(trab, loaded.grid))
    assert np.array_equal(G_cort, s4["G_cort"]) and np.array_equal(G_trab, s4["G_trab"]) and np.array_equal(ALL, s4["G_prx"])
    assert np.array_equal(segm.lh, s4["lh"]) and np.array_equal(segm.cort_seg, s4["cort_seg"]) and np.array_equal(segm.trab_seg, s4["trab_seg"])
    assert np.array_equal(segm.seg, s4["seg"]) and list(segm.lh_el_size_mm) == s4["lh_el_size_mm"]


def test_sites_units_params_and_errors(aim_path, disc, tmp_path):
    q = engine.Logger(echo=False)
    rep = run(aim_path, str(tmp_path / "radius"), site="radius", periosteal=disc, compute_bmd=False, map_units="mm", preview=False, log=q)
    assert rep["site"] == "radius" and rep["step1"]["params"]["close2"] == 30 and rep["parameters"]["step1"]["corner_min"] == 800
    assert rep["parameters"]["map_units"] == "mm" and rep["outputs"]["preview"] is None
    th = sitk.ReadImage(rep["outputs"]["TRAB_TH"])
    assert th.GetPixelID() == sitk.sitkFloat32
    assert np.isclose(sitk.GetArrayFromImage(th).max(), rep["morphometry"]["Tb_Th_max_mm"], rtol=1e-6)
    rep2 = run(aim_path, str(tmp_path / "custom"), site="tibia", periosteal=disc, compute_bmd=False, preview=False, log=q,
               step1_params=Step1Params(close2=40))
    assert rep2["site"] == "custom" and rep2["parameters"]["step1"]["close2"] == 40 and rep2["step1"]["site"] == "custom"
    with pytest.raises(ValueError, match="empty"):
        run(aim_path, str(tmp_path / "empty"), periosteal=np.zeros(SHAPE, bool), compute_bmd=False, preview=False, log=q)
    with pytest.raises(ValueError, match="AIM grid"):
        run(aim_path, str(tmp_path / "wrong"), periosteal=np.ones((10, 10, 10), bool), compute_bmd=False, preview=False, log=q)
    with pytest.raises(ValueError):
        run(aim_path, str(tmp_path / "units"), periosteal=disc, map_units="inches", log=q)
    # a periosteal given as a file (binary 0/1 mask, the way Slicer exports one) reads back the disc
    p = str(tmp_path / "disc_mask.nii.gz")
    write_nifti(p, disc.astype(np.uint8), EL, POS)
    rep3 = run(aim_path, str(tmp_path / "fromfile"), periosteal=p, compute_bmd=False, preview=False, log=q)
    assert rep3["masks"]["periosteal"]["source"] == "file" and rep3["masks"]["periosteal"]["path"] == os.path.abspath(p)
    assert rep3["masks"]["periosteal"]["sha256"] == report_mod.sha256_of(p)
    assert np.array_equal(mask(rep3["outputs"]["PRX_MASK"]), disc)


# ------------------------------------------------------------------------------------------- redo
def test_unchanged_redo_equals_run(run_out, aim_path, tmp_path):
    out, rep = run_out
    with pytest.raises(ValueError, match="nothing to redo"):
        run_from_masks(aim_path, out)
    rep2 = redo_mod._from_run_masks(aim_path, out, str(tmp_path / "same"), compute_bmd=False, preview=False, log=engine.Logger(echo=False))
    outputs_equal(rep, rep2, MAPS + ("SEG", "CORT_SEG", "TRAB_SEG", "CORT_GOBJ", "TRAB_GOBJ", "PRX_GOBJ", "CORT_MASK", "TRAB_MASK"))
    assert rep2["morphometry"] == rep["morphometry"] and rep2["compartments"] == {**rep["compartments"], "PRX_MASK_voxels": None}
    assert rep2["run"]["kind"] == "redo" and rep2["edits"] is None and rep2["step1"] is None
    assert all(rep2["masks"][n]["source"] == "run" for n in ("periosteal", "cortical", "trabecular"))
    assert "PRX_MASK" not in rep2["outputs"]
    # the run's own TRAB_MASK passed as an edit: the documented relation only (the renderer may move rim voxels)
    rep3 = run_from_masks(aim_path, out, str(tmp_path / "own_trab"), trab=rep["outputs"]["TRAB_MASK"], compute_bmd=False, preview=False,
                          log=engine.Logger(echo=False))
    T_run, ALL = mask(rep["outputs"]["TRAB_MASK"]), mask(rep["outputs"]["PRX_GOBJ"])
    R = stages.render(T_run, grid_of(rep))
    C3, T3 = mask(rep3["outputs"]["CORT_MASK"]), mask(rep3["outputs"]["TRAB_MASK"])
    assert np.array_equal(T3, R & ALL) and np.array_equal(C3, ALL & ~T3)
    assert ((C3 != mask(rep["outputs"]["CORT_MASK"])) <= (T_run ^ R)).all()
    assert rep3["edits"]["rule"] == "script34_trab"


def _check_compartment_redo(rep_run, rep, edited, T_or_C):
    """The Script 34 relations of a compartment redo (edited = 'trabecular' | 'cortical', T_or_C the edited raster)."""
    other = "cortical" if edited == "trabecular" else "trabecular"
    key_e, key_o = ("TRAB", "CORT") if edited == "trabecular" else ("CORT", "TRAB")
    ALL = mask(rep_run["outputs"]["PRX_GOBJ"])
    g = grid_of(rep_run)
    E = mask(rep["outputs"][f"{key_e}_MASK"])
    O = mask(rep["outputs"][f"{key_o}_MASK"])
    R = stages.render(T_or_C, g)
    assert np.array_equal(mask(rep["outputs"]["PRX_GOBJ"]), ALL), "the periosteal is unchanged"
    assert np.array_equal(E, R & ALL), "the edited compartment is its own rendering, clipped to the periosteal"
    assert np.array_equal(O, ALL & ~E), "the other compartment is periosteal minus edited"
    assert np.array_equal(mask(rep["outputs"][f"{key_e}_GOBJ"]), E), "the edited contour's rendering is its gobj"
    assert np.array_equal(mask(rep["outputs"][f"{key_o}_GOBJ"]), stages.render(O, g)), "the derived compartment is rendered"
    assert (mask(rep["outputs"]["TRAB_SEG"]) <= mask(rep["outputs"]["TRAB_GOBJ"])).all(), "TRAB_SEG is cut by the trabecular contour"
    assert (mask(rep["outputs"]["CORT_SEG"]) <= mask(rep["outputs"]["CORT_GOBJ"])).all()
    m = rep["masks"]
    assert m["periosteal"]["source"] == "run" and m["periosteal"]["manual"] is False
    assert m[edited]["source"] == "manual" and m[edited]["manual"] is True and m[edited]["rendered_voxels"] == int(E.sum())
    assert m[other]["source"] == f"derived: periosteal - {edited}" and m[other]["voxels"] == int(O.sum())
    e = rep["edits"]
    assert e["edited"] == [edited] and e["outside_periosteal_raw"] == 0 and e["vs_run"]["periosteal"] == {"added": 0, "removed": 0}
    assert rep["run"]["kind"] == "redo" and rep["step1"] is None
    assert rep["run"]["derived_from"] == rep_run["outputs"]["report_json"]
    # the tight boxes of Script 32
    seg = vol(rep["outputs"]["SEG"])
    nz = np.nonzero(seg)
    assert rep["grids"]["SEG"]["dim_xyz"] == [int(n.max() - n.min() + 1) for n in nz[::-1]]
    assert rep["grids"]["SEG"]["pos_xyz"] == [int(p + n.min()) for p, n in zip(POS, nz[::-1])]
    C = mask(rep["outputs"]["CORT_MASK"])
    nz = np.nonzero(C)
    assert rep["grids"]["CORT_MASK"]["dim_xyz"] == [int(n.max() - n.min() + 1) for n in nz[::-1]]
    assert "PRX_MASK" not in rep["outputs"]
    assert os.path.exists(rep["outputs"]["edit_preview"]) and rep["outputs"]["edit_preview"].endswith("_redo_edits.png")
    md = open(rep["outputs"]["report_md"], encoding="utf-8").read()
    assert "**manual**" in md and "## Edits (redo)" in md


def test_trabecular_edit_script34(run_out, aim_path, tmp_path):
    out, rep = run_out
    g = grid_of(rep)
    labels = edit_labelmap(vol(rep["outputs"]["labelmap"]), "thicken_trab")
    T = labels == 2
    assert T.sum() > mask(rep["outputs"]["TRAB_MASK"]).sum()
    p_lm = str(tmp_path / "edited_labelmap.nii.gz")
    slicer.write_labelmap_nifti(p_lm, labels, g)
    p_seg = str(tmp_path / "edited.seg.nrrd")
    slicer.write_seg_nrrd(p_seg, labels, g)
    q = engine.Logger(echo=False)
    rep_a = run_from_masks(aim_path, out, str(tmp_path / "redo_lm"), trab=p_lm, compute_bmd=False, log=q)
    rep_b = run_from_masks(aim_path, out, str(tmp_path / "redo_seg"), trab=p_seg, compute_bmd=False, log=q)
    outputs_equal(rep_a, rep_b, MAPS + ("SEG", "CORT_MASK", "TRAB_MASK", "CORT_GOBJ", "TRAB_GOBJ"))
    assert rep_a["morphometry"] == rep_b["morphometry"]
    for r, p in ((rep_a, p_lm), (rep_b, p_seg)):
        _check_compartment_redo(rep, r, "trabecular", T)
        assert r["edits"]["rule"] == "script34_trab"
        assert r["masks"]["trabecular"]["path"] == os.path.abspath(p) and r["masks"]["trabecular"]["sha256"] == report_mod.sha256_of(p)
        assert r["edits"]["vs_run"]["trabecular"]["added"] > 0 and r["edits"]["vs_run"]["cortical"]["removed"] > 0
    # the rendering clipped nothing on this edit (it stays inside the disc)
    assert rep_a["edits"]["rendering_clipped"] == 0
    # the log went to <base>_redo.log
    assert rep_a["outputs"]["log"].endswith("PHANTOM_redo.log") and os.path.exists(rep_a["outputs"]["log"])


def test_cortical_edit_extension(run_out, aim_path, tmp_path):
    out, rep = run_out
    labels = edit_labelmap(vol(rep["outputs"]["labelmap"]), "thicken_cort")
    C = labels == 1
    p = str(tmp_path / "edited_cort.nii.gz")
    write_nifti(p, C.astype(np.uint8), EL, POS)                       # a Slicer-style binary mask of the cortical segment
    r = run_from_masks(aim_path, out, str(tmp_path / "redo_cort"), cort=p, compute_bmd=False, log=engine.Logger(echo=False))
    _check_compartment_redo(rep, r, "cortical", C)
    assert r["edits"]["rule"] == "script34_cort_ext"
    assert r["edits"]["vs_run"]["cortical"]["added"] > 0 and r["edits"]["vs_run"]["trabecular"]["removed"] > 0
    # Ct.Th's object and gobj are both the edited contour's rendering
    assert np.array_equal(mask(r["outputs"]["CORT_GOBJ"]), mask(r["outputs"]["CORT_MASK"]))


def test_periosteal_edit_reruns_step1(run_out, aim_path, tmp_path):
    out, rep = run_out
    labels = edit_labelmap(vol(rep["outputs"]["labelmap"]), "shrink_periosteal")
    P = labels > 0
    p = str(tmp_path / "edited_all.seg.nrrd")
    slicer.write_seg_nrrd(p, labels, grid_of(rep))
    q = engine.Logger(echo=False)
    r = run_from_masks(aim_path, out, periosteal=p, compute_bmd=False, log=q)
    assert r["run"]["out_dir"] == os.path.abspath(os.path.join(out, "redo"))
    assert r["run"]["kind"] == "redo" and r["step1"] is not None and r["edits"]["rule"] == "step1_from_periosteal"
    assert r["edits"]["edited"] == ["periosteal"] and r["run"]["derived_from"] == rep["outputs"]["report_json"]
    m = r["masks"]
    assert m["periosteal"]["source"] == "manual" and m["periosteal"]["manual"] and m["periosteal"]["path"] == os.path.abspath(p)
    assert m["periosteal"]["sha256"] == report_mod.sha256_of(p)
    assert m["cortical"]["source"] == "ipldt.step1" and m["trabecular"]["source"] == "ipldt.step1"
    assert np.array_equal(mask(r["outputs"]["PRX_MASK"]), P)
    ALL = mask(r["outputs"]["PRX_GOBJ"])
    assert np.array_equal(ALL, mask(r["outputs"]["CORT_MASK"]) | mask(r["outputs"]["TRAB_MASK"]))
    assert np.array_equal(ALL, stages.render(P, grid_of(rep)))
    assert r["edits"]["vs_run"]["periosteal"]["removed"] > 0 and r["edits"]["vs_run"]["periosteal"]["added"] == 0
    assert os.path.exists(r["outputs"]["edit_preview"]) and r["outputs"]["log"].endswith("PHANTOM_redo.log")
    # equals a run from the edited raster
    r2 = run(aim_path, str(tmp_path / "run_edited"), periosteal=P, compute_bmd=False, preview=False, log=q)
    outputs_equal(r, r2, MAPS + ("SEG", "CORT_MASK", "TRAB_MASK", "PRX_GOBJ"))
    assert r2["morphometry"] == r["morphometry"] and r2["step1"]["counts"] == r["step1"]["counts"]


def test_periosteal_plus_compartment(run_out, aim_path, tmp_path):
    out, rep = run_out
    g = grid_of(rep)
    labels = edit_labelmap(edit_labelmap(vol(rep["outputs"]["labelmap"]), "shrink_periosteal"), "thicken_trab")
    P, T = labels > 0, labels == 2
    p = str(tmp_path / "edited_both.seg.nrrd")
    slicer.write_seg_nrrd(p, labels, g)
    q = engine.Logger(echo=False)
    r = run_from_masks(aim_path, out, str(tmp_path / "redo_pt"), periosteal=p, trab=p, compute_bmd=False, log=q)
    ALL = mask(r["outputs"]["PRX_GOBJ"])
    assert np.array_equal(ALL, stages.render(P, g)) and np.array_equal(mask(r["outputs"]["PRX_MASK"]), P)
    T_R = stages.render(T, g) & ALL
    assert np.array_equal(mask(r["outputs"]["TRAB_MASK"]), T_R) and np.array_equal(mask(r["outputs"]["CORT_MASK"]), ALL & ~T_R)
    assert r["edits"]["rule"] == "script34_periosteal_trab" and r["edits"]["edited"] == ["periosteal", "trabecular"]
    m = r["masks"]
    assert m["periosteal"]["manual"] and m["trabecular"]["manual"] and m["cortical"]["source"] == "derived: periosteal - trabecular"
    assert r["step1"] is None
    # periosteal + cortical likewise
    labels_c = edit_labelmap(edit_labelmap(vol(rep["outputs"]["labelmap"]), "shrink_periosteal"), "thicken_cort")
    C = labels_c == 1
    pc = str(tmp_path / "edited_pc_labelmap.nii.gz")
    slicer.write_labelmap_nifti(pc, labels_c, g)
    r2 = run_from_masks(aim_path, out, str(tmp_path / "redo_pc"), periosteal=pc, cort=pc, compute_bmd=False, log=q)
    ALL2 = mask(r2["outputs"]["PRX_GOBJ"])
    C_R = stages.render(C, g) & ALL2
    assert np.array_equal(ALL2, stages.render(labels_c > 0, g))
    assert np.array_equal(mask(r2["outputs"]["CORT_MASK"]), C_R) and np.array_equal(mask(r2["outputs"]["TRAB_MASK"]), ALL2 & ~C_R)
    assert r2["edits"]["rule"] == "script34_periosteal_cort_ext" and r2["masks"]["trabecular"]["source"] == "derived: periosteal - cortical"


def test_refusals(run_out, aim_path, tmp_path):
    out, rep = run_out
    q = engine.Logger(echo=False)
    g = grid_of(rep)
    labels = vol(rep["outputs"]["labelmap"])
    with pytest.raises(ValueError, match="only the compartment you edited"):
        run_from_masks(aim_path, out, trab=labels == 2, cort=labels == 1, log=q)
    # an edit that leaves the bone: refused with the count, the offending voxels written
    over = edit_labelmap(labels, "overshoot")
    p = str(tmp_path / "overshoot.seg.nrrd")
    slicer.write_seg_nrrd(p, over, g)
    out_dir = str(tmp_path / "redo_over")
    n_out = int(((over == 2) & ~(labels > 0)).sum())
    with pytest.raises(EditOutsidePeriosteal) as ei:
        run_from_masks(aim_path, out, out_dir, trab=p, compute_bmd=False, log=q)
    assert ei.value.count == n_out and f"{n_out:,d}" in str(ei.value)
    outside = mask(os.path.join(out_dir, "PHANTOM_redo_outside.nii.gz"))
    assert int(outside.sum()) == n_out and np.array_equal(outside, (over == 2) & ~(labels > 0))
    # the pure rule
    with pytest.raises(EditOutsidePeriosteal):
        derive_compartments(labels > 0, g, trab=over == 2)
    with pytest.raises(ValueError, match="exactly one"):
        derive_compartments(labels > 0, g, trab=labels == 2, cort=labels == 1)
    d = derive_compartments(labels > 0, g, trab=labels == 2)
    assert d["edited"] == "trabecular" and d["derived"] == "cortical" and d["G_cort"] is None and np.array_equal(d["G_trab"], d["trab"])
    assert np.array_equal(d["cort"] | d["trab"], labels > 0) and not (d["cort"] & d["trab"]).any()
    # an inconsistent run folder (PRX_GOBJ tampered) and a missing file
    bad = _copy_run(run_out, tmp_path, "bad_run")
    prx = vol(os.path.join(bad, "PHANTOM_PRX_GOBJ.nii.gz"))
    prx[20, 44, 44] = 0
    write_nifti(os.path.join(bad, "PHANTOM_PRX_GOBJ.nii.gz"), prx, EL, POS)
    with pytest.raises(ValueError, match="inconsistent"):
        run_from_masks(aim_path, bad, trab=labels == 2, log=q)
    missing = _copy_run(run_out, tmp_path, "missing_run")
    os.remove(os.path.join(missing, "PHANTOM_TRAB_MASK.nii.gz"))
    with pytest.raises(FileNotFoundError, match="PHANTOM_TRAB_MASK.nii.gz"):
        run_from_masks(aim_path, missing, trab=labels == 2, log=q)
    with pytest.raises(FileNotFoundError, match="PHANTOM_report.json"):
        run_from_masks(aim_path, str(tmp_path / "nowhere"), trab=labels == 2, log=q)
    # an empty edit
    with pytest.raises(ValueError, match="empty"):
        run_from_masks(aim_path, out, str(tmp_path / "empty_edit"), trab=np.zeros(SHAPE, bool), log=q)


def test_out_dir_policy_and_defaults(run_out, aim_path, tmp_path):
    src = _copy_run(run_out, tmp_path, "policy_run")
    rep = json.load(open(os.path.join(src, "PHANTOM_report.json")))
    labels = edit_labelmap(vol(os.path.join(src, "PHANTOM_compartments_labelmap.nii.gz")), "thicken_trab")
    q = engine.Logger(echo=False)
    r1 = run_from_masks(aim_path, src, trab=labels == 2, compute_bmd=False, preview=False, log=q)
    assert r1["run"]["out_dir"] == os.path.abspath(os.path.join(src, "redo"))
    r2 = run_from_masks(aim_path, src, trab=labels == 2, compute_bmd=False, preview=False, log=q)
    assert r2["run"]["out_dir"] == os.path.abspath(os.path.join(src, "redo_2"))
    assert r1["morphometry"] == r2["morphometry"] and r1["site"] == rep["site"] == "tibia"
    assert r1["parameters"]["dt"] == rep["parameters"]["dt"] and r1["parameters"]["map_units"] == "voxels"
    # an explicit site / map_units are honoured and recorded; a redo of a redo works (derived_from chains)
    r3 = run_from_masks(aim_path, r1["run"]["out_dir"], str(tmp_path / "explicit"), trab=labels == 2, site="radius", map_units="mm",
                        compute_bmd=False, preview=False, log=q)
    assert r3["site"] == "radius" and r3["parameters"]["step1"]["close2"] == 30 and r3["parameters"]["map_units"] == "mm"
    assert r3["run"]["derived_from"] == r1["outputs"]["report_json"] and r3["run"]["out_dir"] == os.path.abspath(str(tmp_path / "explicit"))
    assert sitk.ReadImage(r3["outputs"]["TRAB_TH"]).GetPixelID() == sitk.sitkFloat32
    assert r3["morphometry"] == r1["morphometry"]          # the compartment path does not use Step 1's parameters


# ------------------------------------------------------------------------------------------- real data (slow)
@pytest.mark.slow
def test_pfj_0be66a_redo_from_ipl_periosteal_reproduces_ipl_masks(data_root, tmp_path):
    """IPL's own periosteal contour (raw CORT_MASK | TRAB_MASK of PFJ-0be66a_R) given to run() reproduces IPL's
    CORT_MASK / TRAB_MASK (0 mismatches), and IPL's TRAB_MASK given as a trabecular edit on top of it derives
    the cortex as periosteal minus the rendered trabecular contour (Script 34)."""
    grab = os.path.join(data_root, "PFJ-0be66a_R")
    paths = {k: os.path.join(grab, f"X2420448{k}.AIM") for k in ("", "_CORT_MASK", "_TRAB_MASK")}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("PFJ-0be66a_R/X2420448{,_CORT_MASK,_TRAB_MASK}.AIM not found")
    native = read_aim(paths[""])
    dim, pos = native["dim"], native["pos"]
    C_ipl = align_to(read_aim(paths["_CORT_MASK"]), dim, pos) > 0
    T_ipl = align_to(read_aim(paths["_TRAB_MASK"]), dim, pos) > 0
    out = str(tmp_path / "X2420448")
    rep = run(paths[""], out, site="tibia", periosteal=C_ipl | T_ipl, compute_bmd=False, preview=False)
    C, T = mask(rep["outputs"]["CORT_MASK"]), mask(rep["outputs"]["TRAB_MASK"])
    assert int((C != C_ipl).sum()) == 0 and int((T != T_ipl).sum()) == 0
    assert int(C.sum()) == 7_154_580 and int(T.sum()) == 20_329_639
    r = run_from_masks(paths[""], out, trab=paths["_TRAB_MASK"], compute_bmd=False, preview=False)
    T_R = stages.render(T_ipl, grid_of(rep)) & (C_ipl | T_ipl)
    assert np.array_equal(mask(r["outputs"]["TRAB_MASK"]), T_R) and np.array_equal(mask(r["outputs"]["CORT_MASK"]), (C_ipl | T_ipl) & ~T_R)
    assert r["masks"]["trabecular"]["manual"] and r["masks"]["cortical"]["source"] == "derived: periosteal - trabecular"
