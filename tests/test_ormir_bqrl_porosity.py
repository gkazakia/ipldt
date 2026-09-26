"""ORMIR-BQRL cortical porosity (Ct.Po and the PORE map) and the AIM output option (--map-format aim).

The phantom is the cylinder of test_ormir_bqrl_pipeline (a v020 short AIM with VAX-float element sizes and a
calibration processing log; a periosteal disc with a half-integer centre, so IPL's contour rendering is the
identity on it and ORMIR-BQRL and ipldt.ormir.run_pipeline see the same contours) with intracortical pores
added: vertical channels and small cavities of marrow density inside the dense shell, so the Laplace-Hamming
SEG has voids inside the cortical contour and the pore cascade has something to find.

Every porosity check is against the calls ipldt.ormir.run_pipeline makes in its STEP 5c: ipldt.porosity's
pore_cascade_ipl_grid (the cascade on IPL's render grid of the cortical contour, see test_porosity_grid) and ct_po
on the rendered cortical contour and CORT_SEG, both handed over as char volumes on the AIM grid.
The AIM checks read every written volume back through ipldt.io.read_aim and compare it with the NIfTI run and,
byte for byte, with run_pipeline(map_format='aim').

Skips without SimpleITK / itk / ormir_xct (the loader needs them; the autocontour is never called).
"""
import json
import math
import os
import struct

import numpy as np
import pytest
from scipy import ndimage as ndi

sitk = pytest.importorskip("SimpleITK")
pytest.importorskip("itk")
pytest.importorskip("ormir_xct")

from ipldt import ormir as engine                                                   # noqa: E402
from ipldt import porosity                                                          # noqa: E402
from ipldt.io import TYPE_CHAR, TYPE_SHORT, align_to, read_aim, write_nifti         # noqa: E402

from ormir_bqrl import cli, slicer, stages                                          # noqa: E402
from ormir_bqrl import redo as redo_mod                                             # noqa: E402
from ormir_bqrl.pipeline import run                                                 # noqa: E402
from ormir_bqrl.redo import run_from_masks                                          # noqa: E402

SHAPE = (40, 90, 90)                      # (z, y, x)
DIM = SHAPE[::-1]
POS = (100, 200, 300)
EL = (0.06069973, 0.06069973, 0.06069973)
CENTRE = 44.5
R_OUT, R_IN, R_ROD, R_PORES = 40.0, 32.0, 28.0, 36.0
PROCLOG = ("! Processing Log\n"
           "Mu_Scaling                                       8192\n"
           "Mu_Water                                       0.2409\n"
           "Density: unit                                  mg HA/ccm\n"
           "Density: slope                         1.61907703e+03\n"
           "Density: intercept                    -3.94095001e+02\n")
MASKS = ("PRX_MASK", "PRX_GOBJ", "CORT_MASK", "TRAB_MASK", "CORT_GOBJ", "TRAB_GOBJ", "SEG", "CORT_SEG", "TRAB_SEG")
MAPS = ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH")
PORO_KEYS = {"Ct_Po", "Ct_Po_pore_voxels", "Ct_Po_compartment_voxels"}
QUIET = dict(compute_bmd=False, preview=False)


# ------------------------------------------------------------------------------------------- the phantom
def _radius():
    yy, xx = np.mgrid[:SHAPE[1], :SHAPE[2]]
    return np.sqrt((yy - CENTRE) ** 2 + (xx - CENTRE) ** 2)


def _vax_pack(x):
    """The inverse of ipldt.io._vax_f (word-swapped VAX F-floating)."""
    if x == 0:
        return 0
    m, e = math.frexp(x)
    f = int(round((m - 0.5) * 2 ** 24))
    w1 = ((e + 128) << 7) | (f >> 16)
    w2 = f & 0xFFFF
    u = (w2 << 16) | w1
    return u - 2 ** 32 if u >= 2 ** 31 else u


def phantom_grey(seed=0):
    """native int16 (z, y, x): shell 12000 (32 <= r <= 40), marrow 1000, outside 300, rods 9000 inside r < 28;
    in the shell, at r = 36, eight vertical channels (radius 1.8, slices 6..33) and sixteen cavities (radius 2.5)
    of marrow density 1000."""
    r = _radius()
    grey = np.broadcast_to(np.where(r <= R_OUT, np.where(r >= R_IN, 12000, 1000), 300).astype(np.int16), SHAPE).copy()
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.standard_normal(SHAPE), 2.0)
    grey[(f > np.quantile(f, 1.0 - 0.35)) & np.broadcast_to(r < R_ROD, SHAPE)] = 9000
    yy, xx = np.mgrid[:SHAPE[1], :SHAPE[2]]
    zz = np.arange(SHAPE[0])[:, None, None]
    for k in range(8):
        a = 2.0 * np.pi * k / 8.0
        cy, cx = CENTRE + R_PORES * np.sin(a), CENTRE + R_PORES * np.cos(a)
        grey[6:34, np.hypot(yy - cy, xx - cx) <= 1.8] = 1000
        b = a + np.pi / 8.0
        cy, cx = CENTRE + R_PORES * np.sin(b), CENTRE + R_PORES * np.cos(b)
        for cz in (12, 26):
            grey[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= 2.5 ** 2] = 1000
    return grey


def make_aim(path, seed=0):
    """A v020 short AIM readable by ITK ScancoImageIO, ORMIR's file_reader and ipldt.io.read_aim."""
    data = phantom_grey(seed)
    ints = [0] * 35
    ints[0] = 16
    ints[5] = TYPE_SHORT
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
    return np.broadcast_to(_radius() <= R_OUT, SHAPE).copy()


def thicken(labels, grow):
    """A Slicer-style boundary edit of the labelmap on slices 10..29: grow = 1 moves the innermost 3 voxels of the
    cortical ring into the trabecular compartment's place (cortex thicker), grow = 2 the reverse."""
    lab = np.array(labels, np.uint8, copy=True)
    other = 2 if grow == 1 else 1
    for z in range(10, 30):
        rim = (lab[z] == other) & ndi.binary_dilation(lab[z] == grow, iterations=3)
        lab[z][rim] = grow
    return lab


# ------------------------------------------------------------------------------------------- helpers
Q = engine.Logger(echo=False)


def vol(path):
    """A written volume as a (z, y, x) array, NIfTI through SimpleITK, AIM through ipldt.io.read_aim."""
    if path.lower().endswith(".aim"):
        return read_aim(path)["data"]
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def mask(path):
    return vol(path) > 0


def pore_like_run_pipeline(G_cort, cort_seg):
    """ipldt.ormir.run_pipeline's STEP 5c on the given rasters: the rendered cortical contour and CORT_SEG as
    char volumes on the AIM grid, ipldt.porosity.pore_cascade_ipl_grid (the cascade on the contour's /gobj_to_aim
    grid), then ct_po.  Returns (PORE bool on the AIM grid, ct_po's dict, the cascade's (R, S, U) grids)."""
    cr = engine.volume(np.asarray(G_cort, bool).astype(np.uint8) * 127, DIM, POS)
    cs = engine.volume(np.asarray(cort_seg, bool).astype(np.uint8) * 127, DIM, POS)
    out = porosity.pore_cascade_ipl_grid(cr, cs)
    pore = out["pore"]
    return align_to(pore, DIM, POS) != 0, porosity.ct_po(pore, cr), (out["render_grid"], out["seg_grid"], out["grid"])


def check_porosity_against_engine(rep):
    """The report's porosity block and PORE file equal run_pipeline's STEP 5c on the run's own CORT_GOBJ / CORT_SEG."""
    po = rep["porosity"]
    assert set(po) == PORO_KEYS
    pore_ref, ref, grids = pore_like_run_pipeline(mask(rep["outputs"]["CORT_GOBJ"]), mask(rep["outputs"]["CORT_SEG"]))
    g = lambda t: {"dim_xyz": list(t[0]), "pos_xyz": list(t[1])}                     # noqa: E731
    assert rep["parameters"]["porosity"]["grids"] == {"render": g(grids[0]), "cort_seg": g(grids[1]), "cascade": g(grids[2])}
    assert po["Ct_Po"] == ref["ct_po"]
    assert po["Ct_Po_pore_voxels"] == ref["pore_voxels"] and po["Ct_Po_compartment_voxels"] == ref["mask_voxels"]
    assert po["Ct_Po_compartment_voxels"] == rep["compartments"]["CORT_GOBJ_voxels"]
    assert np.array_equal(mask(rep["outputs"]["PORE"]), pore_ref)
    assert rep["summary"]["Ct_Po"] == po["Ct_Po"]
    return po


def autocontour_is_the_disc(monkeypatch, disc):
    monkeypatch.setattr(engine, "step2_autocontour", lambda img, calib, log=None: engine.array_to_sitk(disc.astype(np.uint8), img))


# ------------------------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def aim_path(tmp_path_factory):
    return make_aim(str(tmp_path_factory.mktemp("aim_poro") / "PHANTOM.AIM"))


@pytest.fixture(scope="module")
def disc():
    return periosteal_disc()


@pytest.fixture(scope="module")
def disc_file(disc, tmp_path_factory):
    p = str(tmp_path_factory.mktemp("disc") / "PHANTOM_PRX_MASK.nii.gz")
    write_nifti(p, disc.astype(np.uint8) * 127, EL, POS)
    return p


@pytest.fixture(scope="module")
def run_nii(aim_path, disc, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("run_nii") / "PHANTOM")
    return out, run(aim_path, out, site="tibia", periosteal=disc, log=Q, **QUIET)


@pytest.fixture(scope="module")
def run_aim(aim_path, disc, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("run_aim") / "PHANTOM")
    return out, run(aim_path, out, site="tibia", periosteal=disc, map_format="aim", log=Q, **QUIET)


# ------------------------------------------------------------------------------------------- Ct.Po / PORE
def test_ctpo_and_pore_map_in_run(run_nii):
    out, rep = run_nii
    po = check_porosity_against_engine(rep)
    assert 0.0 < po["Ct_Po"] < 0.5 and po["Ct_Po_pore_voxels"] > 0, "the phantom's channels and cavities are pores"
    p = rep["outputs"]["PORE"]
    assert p == os.path.abspath(os.path.join(out, "PHANTOM_PORE.nii.gz"))
    v = vol(p)
    assert v.dtype == np.uint8 and v.shape == SHAPE and set(np.unique(v).tolist()) == {0, 127}
    assert not (v > 0)[mask(rep["outputs"]["CORT_SEG"])].any(), "a pore is never cortical bone"
    img = sitk.ReadImage(p)
    assert np.allclose(img.GetOrigin(), [q * e for q, e in zip(POS, EL)], atol=1e-4) and np.allclose(img.GetSpacing(), EL, rtol=1e-6)
    P = rep["parameters"]["porosity"]
    assert P["computed"] is True and P["error"] is None and P["method"] == "ipldt.porosity.pore_cascade"
    assert P["hysteresis"] == dict(porosity.SCRIPT32_HYSTERESIS, grow_axes=list(porosity.SCRIPT32_HYSTERESIS["grow_axes"]))
    assert P["min_pore_voxels"] == porosity.MIN_PORE_VOXELS and P["slice_fraction_percent"] == list(porosity.SLICE_FRACTION)
    assert rep["parameters"]["map_format"] == "nifti"
    assert "5c_porosity" in rep["timing_s"]
    with open(rep["outputs"]["report_json"], "r", encoding="utf-8") as fh:
        assert json.load(fh) == rep
    csv_text = open(rep["outputs"]["report_csv"], encoding="utf-8").read()
    assert f"porosity.Ct_Po,{po['Ct_Po']}" in csv_text and "summary.Ct_Po," in csv_text and "porosity.Ct_Po_pore_voxels," in csv_text
    md = open(rep["outputs"]["report_md"], encoding="utf-8").read()
    assert "| Ct.Po |" in md and "| PORE |" in md and "pore cascade" in md
    assert "STEP 5c" in open(rep["outputs"]["log"], encoding="utf-8").read()


def test_porosity_parity_with_run_pipeline(run_nii, aim_path, disc, tmp_path, monkeypatch):
    """The same Ct.Po, pore counts and PORE map as ipldt.ormir.run_pipeline fed the same periosteal."""
    out, rep = run_nii
    autocontour_is_the_disc(monkeypatch, disc)
    out2 = str(tmp_path / "engine")
    rp = engine.run_pipeline(aim_path, out2, site="tibia", compute_bmd=False, log=engine.Logger(echo=False))
    assert rp["porosity"] == rep["porosity"]
    assert rp["porosity_grids"] == rep["parameters"]["porosity"]["grids"] and rp["porosity_grids"]["cascade"]["dim_xyz"] != list(DIM)
    assert np.array_equal(vol(rep["outputs"]["PORE"]), vol(os.path.join(out2, "PHANTOM_PORE.nii.gz")))
    assert rp["morphometry"] == rep["morphometry"]


def test_no_porosity_skips_the_cascade(run_nii, aim_path, disc, disc_file, tmp_path):
    out, rep_on = run_nii
    rep = run(aim_path, str(tmp_path / "off"), periosteal=disc, compute_porosity=False, log=Q, **QUIET)
    assert rep["porosity"] == {} and rep["summary"]["Ct_Po"] is None and "PORE" not in rep["outputs"]
    assert not os.path.exists(tmp_path / "off" / "PHANTOM_PORE.nii.gz")
    assert rep["parameters"]["porosity"]["computed"] is False and "5c_porosity" not in rep["timing_s"]
    assert rep["morphometry"] == rep_on["morphometry"] and rep["compartments"] == rep_on["compartments"]
    assert "| Ct.Po |" not in open(rep["outputs"]["report_md"], encoding="utf-8").read()
    # the CLI flag (with the AIM format at the same time)
    rc = cli.main(["run", aim_path, str(tmp_path / "cli"), "--periosteal", disc_file, "--no-bmd", "--no-preview", "--no-porosity",
                   "--map-format", "aim", "--backend", "cpu"])
    assert rc == 0
    files = os.listdir(tmp_path / "cli")
    assert "PHANTOM_PORE.AIM" not in files and "PHANTOM_PORE.nii.gz" not in files
    assert "PHANTOM_SEG.AIM" in files and "PHANTOM_TRAB_TH.AIM" in files and "PHANTOM_HU.nii.gz" in files
    r = json.load(open(tmp_path / "cli" / "PHANTOM_report.json", encoding="utf-8"))
    assert r["porosity"] == {} and r["parameters"]["map_format"] == "aim"
    assert r["run"]["command"][-5:] == ["--no-porosity", "--map-format", "aim", "--backend", "cpu"]


def test_redo_recomputes_porosity(run_nii, aim_path, tmp_path):
    out, rep = run_nii
    # the run's own masks: the run's porosity, exactly
    same = redo_mod._from_run_masks(aim_path, out, str(tmp_path / "same"), log=Q, **QUIET)
    assert same["porosity"] == rep["porosity"]
    assert np.array_equal(vol(same["outputs"]["PORE"]), vol(rep["outputs"]["PORE"]))
    labels = vol(rep["outputs"]["labelmap"])
    # a cortical edit (the cortex thickened inward): the cascade reruns on the new cortical contour
    lab_c = thicken(labels, 1)
    rc = run_from_masks(aim_path, out, str(tmp_path / "cort"), cort=lab_c == 1, log=Q, **QUIET)
    check_porosity_against_engine(rc)
    assert rc["porosity"]["Ct_Po_compartment_voxels"] > rep["porosity"]["Ct_Po_compartment_voxels"]
    assert rc["parameters"]["porosity"]["computed"] is True
    # a trabecular edit (the cortex thinned)
    lab_t = thicken(labels, 2)
    rt = run_from_masks(aim_path, out, str(tmp_path / "trab"), trab=lab_t == 2, log=Q, **QUIET)
    check_porosity_against_engine(rt)
    assert rt["porosity"]["Ct_Po_compartment_voxels"] < rep["porosity"]["Ct_Po_compartment_voxels"]
    # a periosteal edit (STEP 1 reruns through run())
    P = labels > 0
    P[:, _radius() > 39] = False
    rp = run_from_masks(aim_path, out, str(tmp_path / "prx"), periosteal=P, log=Q, **QUIET)
    check_porosity_against_engine(rp)
    assert rp["porosity"]["Ct_Po_compartment_voxels"] != rep["porosity"]["Ct_Po_compartment_voxels"]
    # the opt-out in a redo
    r0 = run_from_masks(aim_path, out, str(tmp_path / "off"), cort=lab_c == 1, compute_porosity=False, log=Q, **QUIET)
    assert r0["porosity"] == {} and "PORE" not in r0["outputs"] and r0["morphometry"] == rc["morphometry"]


# ------------------------------------------------------------------------------------------- AIM output
def test_aim_output_round_trips(run_nii, run_aim, aim_path):
    out_n, rn = run_nii
    out_a, ra = run_aim
    src = read_aim(aim_path)
    assert ra["parameters"]["map_format"] == "aim" and ra["parameters"]["map_units"] == "voxels"
    for n in MASKS + MAPS + ("PORE",):
        p = ra["outputs"][n]
        assert p == os.path.abspath(os.path.join(out_a, f"PHANTOM_{n}.AIM")) and os.path.isfile(p), n
        a = read_aim(p)
        assert a["type_code"] == TYPE_CHAR and a["data"].dtype == np.uint8, n
        assert a["dim"] == DIM and a["pos"] == POS and a["el_size_mm"] == src["el_size_mm"], n
        assert np.array_equal(a["data"].astype(np.int16), vol(rn["outputs"][n]).astype(np.int16)), n
        assert not os.path.exists(os.path.join(out_a, f"PHANTOM_{n}.nii.gz")), n
    assert set(np.unique(read_aim(ra["outputs"]["SEG"])["data"]).tolist()) == {0, 126, 127}
    assert set(np.unique(read_aim(ra["outputs"]["PORE"])["data"]).tolist()) == {0, 127}
    assert read_aim(ra["outputs"]["CORT_TH"])["data"].max() > 0
    # the greyscale for Slicer, the Slicer files and every number are those of the NIfTI run
    assert ra["outputs"]["HU"].endswith("PHANTOM_HU.nii.gz") and np.array_equal(vol(ra["outputs"]["HU"]), vol(rn["outputs"]["HU"]))
    assert np.array_equal(vol(ra["outputs"]["seg_nrrd"]), vol(rn["outputs"]["seg_nrrd"]))
    assert np.array_equal(vol(ra["outputs"]["labelmap"]), vol(rn["outputs"]["labelmap"]))
    for block in ("morphometry", "porosity", "compartments", "summary", "masks"):
        a, b = dict(ra[block]), dict(rn[block])
        if block == "masks":
            a = {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in a.items()}
            b = {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in b.items()}
        assert a == b, block
    # an AIM mask reads back by position through the Slicer reader too
    grid = stages.Grid(DIM, POS, tuple(src["el_size_mm"]))
    m, info = slicer.read_mask(ra["outputs"]["CORT_MASK"], grid)
    assert info["format"] == "aim" and np.array_equal(m, mask(rn["outputs"]["CORT_MASK"]))


def test_aim_output_equals_run_pipeline_bytes(run_aim, aim_path, disc, tmp_path, monkeypatch):
    """Written through the same ipldt.ormir.write_volume / ipldt.io.write_aim with the input AIM's header: every AIM
    ORMIR-BQRL writes is byte-identical to run_pipeline(map_format='aim')'s file of the same name."""
    out_a, ra = run_aim
    autocontour_is_the_disc(monkeypatch, disc)
    out2 = str(tmp_path / "engine_aim")
    rp = engine.run_pipeline(aim_path, out2, map_format="aim", site="tibia", compute_bmd=False, log=engine.Logger(echo=False))
    assert rp["parameters"]["map_format"] == "aim"
    for n in MASKS + MAPS + ("PORE",):
        with open(ra["outputs"][n], "rb") as fa, open(os.path.join(out2, f"PHANTOM_{n}.AIM"), "rb") as fb:
            assert fa.read() == fb.read(), n


def test_aim_refusals(run_nii, aim_path, disc, tmp_path):
    with pytest.raises(ValueError, match="voxels"):
        run(aim_path, str(tmp_path / "mm"), periosteal=disc, map_format="aim", map_units="mm", log=Q, **QUIET)
    assert not os.path.exists(tmp_path / "mm" / "PHANTOM_SEG.AIM")
    with pytest.raises(ValueError, match="map_format"):
        run(aim_path, str(tmp_path / "png"), periosteal=disc, map_format="png", log=Q, **QUIET)
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", aim_path, str(tmp_path / "x"), "--map-format", "png"])
    assert exc.value.code == 2
    # a map that does not fit IPL's char AIM (0..255 voxels) is refused before anything is written
    out, rn = run_nii
    loaded = stages.load(aim_path)
    masks = stages.Masks(ALL=mask(rn["outputs"]["PRX_GOBJ"]), cort=mask(rn["outputs"]["CORT_MASK"]), trab=mask(rn["outputs"]["TRAB_MASK"]),
                         G_cort=mask(rn["outputs"]["CORT_GOBJ"]), G_trab=mask(rn["outputs"]["TRAB_GOBJ"]))
    seg = vol(rn["outputs"]["SEG"])
    segm = stages.Segmentation(lh=seg > 0, cort_seg=seg == 127, trab_seg=seg == 126, seg=seg, lh_el_size_mm=EL)
    big = vol(rn["outputs"]["TRAB_SP"]).astype(np.int16)
    big[20, 44, 44] = 300
    morph = stages.Morphometry(maps={"TRAB_SP": big}, metrics={}, grids={}, timing_s={}, params={}, backend="cpu")
    dst = tmp_path / "too_big"
    with pytest.raises(ValueError, match="TRAB_SP.*255"):
        stages.write_volumes(str(dst), loaded, masks, segm, morph, "voxels", Q, map_format="aim")
    assert not dst.exists() or not os.listdir(dst)


def test_redo_of_an_aim_run_folder(run_nii, run_aim, aim_path, tmp_path):
    """redo reads an AIM run folder's masks, keeps its format by default, and gives the numbers of the NIfTI folder."""
    out_n, rn = run_nii
    out_a, ra = run_aim
    labels = vol(ra["outputs"]["labelmap"])
    T = thicken(labels, 2) == 2
    r_a = run_from_masks(aim_path, out_a, str(tmp_path / "redo_aim"), trab=T, log=Q, **QUIET)
    r_n = run_from_masks(aim_path, out_n, str(tmp_path / "redo_nii"), trab=T, log=Q, **QUIET)
    assert r_a["parameters"]["map_format"] == "aim" and r_n["parameters"]["map_format"] == "nifti"
    assert r_a["outputs"]["PORE"].endswith("PHANTOM_PORE.AIM") and r_a["outputs"]["TRAB_MASK"].endswith("PHANTOM_TRAB_MASK.AIM")
    assert r_a["masks"]["periosteal"]["path"].endswith("PHANTOM_PRX_GOBJ.AIM")
    assert r_a["morphometry"] == r_n["morphometry"] and r_a["porosity"] == r_n["porosity"]
    for n in ("TRAB_MASK", "CORT_MASK", "PORE", "SEG", "CORT_TH"):
        assert np.array_equal(vol(r_a["outputs"][n]).astype(np.int16), vol(r_n["outputs"][n]).astype(np.int16)), n
    # an explicit format overrides the run's
    r_x = run_from_masks(aim_path, out_a, str(tmp_path / "redo_x"), trab=T, map_format="nifti", log=Q, **QUIET)
    assert r_x["outputs"]["PORE"].endswith("PHANTOM_PORE.nii.gz") and r_x["porosity"] == r_n["porosity"]
    # slicer-export works on the AIM folder as well
    res = slicer.export_run(out_a, out_dir=str(tmp_path / "export"))
    assert np.array_equal(slicer.read_seg_nrrd(res["seg_nrrd"])[0], vol(ra["outputs"]["seg_nrrd"]))


def test_cli_flags_parse():
    p = cli.build_parser()
    a = p.parse_args(["run", "a.AIM", "out"])
    assert a.map_format == "nifti" and a.no_porosity is False
    a = p.parse_args(["redo", "a.AIM", "run", "--trab", "t.nii.gz", "--no-porosity", "--map-format", "aim"])
    assert a.map_format == "aim" and a.no_porosity is True
    assert p.parse_args(["redo", "a.AIM", "run", "--trab", "t.nii.gz"]).map_format is None
    a = p.parse_args(["batch", "out", "a.AIM", "b.AIM", "--map-format", "aim", "--no-porosity"])
    assert a.map_format == "aim" and a.no_porosity is True
