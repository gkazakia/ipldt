"""ipldt.ormir.step3_trab_cort_seg -- the pipeline's step 3 on ipldt.step1 (Script 32 STEP 1 reimplementation).

Fast (needs SimpleITK): a synthetic cylinder with a dense shell goes through the SimpleITK step -- the native
volume dict as step1_load_aim returns it and a periosteal mask image on the image grid -- and the returned
CORT / TRAB images are uint8 0/127 on the image grid, partition the periosteal rendering, and form a cortical
ring around a trabecular disc on every slice; the info dict carries the Step1Params, the native thresholds,
the seg_gauss box and the per-stage counts; the site parameter and the calibration fallback work.
Slow (needs the XCT_masks_full_grab folder): the pipeline-level proof -- step 3 fed with IPL's own periosteal
contour (the raw CORT_MASK | TRAB_MASK rasters of PFJ-0be66a_R, which equal IPL's rendered periosteal gobj)
returns IPL's raw X2420448_CORT_MASK.AIM / TRAB_MASK.AIM voxel for voxel (0 mismatches)."""
import os

import numpy as np
import pytest
from scipy import ndimage as ndi

sitk = pytest.importorskip("SimpleITK")

from ipldt import ipl_ops as ops                                                    # noqa: E402
from ipldt.io import align_to, read_aim                                             # noqa: E402
from ipldt.ormir import (RADIUS, TIBIA, SITE_PARAMS, array_to_sitk, sitk_to_bool,   # noqa: E402
                         step1_calibration, step1_params_for, step3_trab_cort_seg)
from ipldt.step1 import STAGES, Step1Params                                          # noqa: E402

PROCLOG = ("Mu_Scaling                                       8192\n"
           "Density: slope                         1.61907703e+03\n"
           "Density: intercept                    -3.94095001e+02\n")
CALIB_ITK = dict(mu_scaling=8192.0, mu_water=0.2403, rescale_slope=1619.07703, rescale_intercept=-394.095001)
SHAPE = (40, 90, 90)          # (z, y, x)
CENTRE = 44.5
R_OUT, R_IN = 40.0, 32.0
EL = (0.0607, 0.0607, 0.0607)


def _radius():
    yy, xx = np.mgrid[:SHAPE[1], :SHAPE[2]]
    return np.sqrt((yy - CENTRE) ** 2 + (xx - CENTRE) ** 2)


@pytest.fixture(scope="module")
def phantom():
    """native: a cylinder r <= 40 on all slices, dense shell (12000 native, ~1900 mgHA) for 32 <= r <= 40,
    marrow 1000 native inside, 300 outside, as a read_aim-style dict with a processing log; prx: the disc as
    a uint8 0/1 SimpleITK image on the image grid (half-integer centre: IPL's rendering leaves it unchanged)."""
    r = _radius()
    grey2d = np.where(r <= R_OUT, np.where(r >= R_IN, 12000, 1000), 300).astype(np.int16)
    native = dict(data=np.broadcast_to(grey2d, SHAPE).copy(), dim=SHAPE[::-1], pos=(100, 200, 300), el_size_mm=EL,
                  proclog=PROCLOG)
    ref = sitk.GetImageFromArray(np.zeros(SHAPE, np.uint8))
    ref.SetSpacing(EL)
    ref.SetOrigin([p * e for p, e in zip(native["pos"], EL)])
    prx = array_to_sitk(np.broadcast_to(r <= R_OUT, SHAPE).astype(np.uint8), ref)
    return native, prx


def _ring_checks(cort_img, trab_img, prx_img):
    G = sitk_to_bool(prx_img)
    C = sitk_to_bool(cort_img)
    T = sitk_to_bool(trab_img)
    assert C.shape == T.shape == G.shape
    assert not (C & T).any(), "cort and trab overlap"
    assert np.array_equal(C | T, G), "cort and trab do not partition the periosteal rendering"
    cy, cx = 44, 44
    s4 = ndi.generate_binary_structure(2, 1)
    for z in range(G.shape[0]):
        assert not C[z, cy, cx] and T[z, cy, cx]
        assert ndi.label(C[z], structure=s4)[1] == 1, f"slice {z}: cort is not one ring"
        holes = ndi.binary_fill_holes(C[z]) & ~C[z]
        assert np.array_equal(holes, T[z]), f"slice {z}: trab is not the hole of the cort ring"
        assert np.pi * 28 ** 2 < int(T[z].sum()) < np.pi * 36 ** 2
    assert C[20, cy, 44 + 38] and not T[20, cy, 44 + 38]           # the shell is cortical


def test_step3_images_and_info(phantom):
    native, prx = phantom
    lines = []
    cort, trab, info = step3_trab_cort_seg(native, prx, TIBIA, CALIB_ITK, log=lines.append)
    for img in (cort, trab):
        assert img.GetPixelID() == sitk.sitkUInt8
        assert img.GetSize() == prx.GetSize() and img.GetSpacing() == prx.GetSpacing() and img.GetOrigin() == prx.GetOrigin()
        assert set(np.unique(sitk.GetArrayFromImage(img))) == {0, 127}
    _ring_checks(cort, trab, prx)
    assert info["params"] == dict(TIBIA.__dict__)
    assert info["calibration_source"] == "proclog"
    assert info["calibration"] == dict(slope=1619.07703, intercept=-394.095001, mu_scaling=8192.0)
    assert info["thresholds"]["lower_native"] == 4524 and info["thresholds"]["upper_native"] == 17173
    assert info["box"] == dict(dim=(80, 80, 40), pos=(105, 205, 300))                  # the disc's tight box
    assert info["periosteal"] == dict(raw_voxels=int(sitk_to_bool(prx).sum()), rendered_voxels=int(sitk_to_bool(prx).sum()),
                                      dim=(80, 80, 40), pos=(105, 205, 300))
    assert set(STAGES) <= set(info["counts"]) and set(STAGES) <= set(info["grids"]) and "total" in info["timings"]
    assert info["counts"]["28_cortfinal"] == int(sitk_to_bool(cort).sum())
    assert info["counts"]["29_trabfinal"] == int(sitk_to_bool(trab).sum())
    assert any("STEP 3" in ln for ln in lines) and any("29_trabfinal" in ln for ln in lines)


def test_step3_matches_the_numpy_chain(phantom):
    """The SimpleITK step is the numpy chain (cort_trab_separation_from_raw) pasted onto the image grid."""
    from ipldt.step1 import cort_trab_separation_from_raw
    native, prx = phantom
    cort, trab, _ = step3_trab_cort_seg(native, prx, TIBIA, CALIB_ITK)
    raw = ops.mask_vol(sitk_to_bool(prx), native["dim"], native["pos"])
    ref = cort_trab_separation_from_raw(native, raw, TIBIA)
    assert np.array_equal(sitk_to_bool(cort), align_to(ref["cort"], native["dim"], native["pos"]) != 0)
    assert np.array_equal(sitk_to_bool(trab), align_to(ref["trab"], native["dim"], native["pos"]) != 0)
    assert ref["trab"]["data"].dtype == np.int8 and set(np.unique(ref["trab"]["data"])) == {0, 127}


def test_calibration_fallback_and_site(phantom):
    native, prx = phantom
    no_log = dict(native, proclog="no calibration here")
    cal, src = step1_calibration(no_log, CALIB_ITK)
    assert src == "itk_scanco_header" and cal == dict(slope=1619.07703, intercept=-394.095001, mu_scaling=8192.0)
    with pytest.raises(ValueError):
        step1_calibration(no_log, None)
    with pytest.raises(ValueError):
        step3_trab_cort_seg(no_log, prx, TIBIA)
    cort_a, trab_a, info_a = step3_trab_cort_seg(native, prx, TIBIA)
    cort_b, trab_b, info_b = step3_trab_cort_seg(no_log, prx, TIBIA, CALIB_ITK)
    assert info_a["calibration_source"] == "proclog" and info_b["calibration_source"] == "itk_scanco_header"
    assert info_a["thresholds"] == info_b["thresholds"]
    assert np.array_equal(sitk_to_bool(cort_a), sitk_to_bool(cort_b)) and np.array_equal(sitk_to_bool(trab_a), sitk_to_bool(trab_b))
    # site names and presets
    assert step1_params_for("tibia") is TIBIA and step1_params_for("Radius") is RADIUS
    assert step1_params_for("tibia", RADIUS) is RADIUS and SITE_PARAMS == {"tibia": TIBIA, "radius": RADIUS}
    with pytest.raises(ValueError):
        step1_params_for("femur")
    with pytest.raises(TypeError):
        step1_params_for("tibia", {"close2": 30})
    cort_r, trab_r, info_r = step3_trab_cort_seg(native, prx, RADIUS)
    _ring_checks(cort_r, trab_r, prx)
    assert info_r["params"]["close2"] == 30 and info_r["params"]["corner_min"] == 800
    custom = Step1Params(close2=40)
    assert step3_trab_cort_seg(native, prx, custom)[2]["params"]["close2"] == 40


def test_periosteal_mask_must_be_on_the_aim_grid(phantom):
    native, prx = phantom
    small = sitk.GetImageFromArray(sitk.GetArrayFromImage(prx)[:, :80, :80])
    with pytest.raises(ValueError):
        step3_trab_cort_seg(native, small, TIBIA)


# ------------------------------------------------------------------------------------------- real data (slow)
@pytest.mark.slow
def test_pfj_0be66a_step3_with_ipl_contour_reproduces_ipl_masks(data_root):
    """Pipeline-level proof: step 3 given IPL's own periosteal contour returns IPL's CORT_MASK / TRAB_MASK."""
    grab = os.path.join(data_root, "PFJ-0be66a_R")
    paths = {k: os.path.join(grab, f"X2420448{k}.AIM") for k in ("", "_CORT_MASK", "_TRAB_MASK")}
    if not all(os.path.exists(p) for p in paths.values()):
        pytest.skip("PFJ-0be66a_R/X2420448{,_CORT_MASK,_TRAB_MASK}.AIM not found")
    native = read_aim(paths[""])
    dim, pos = native["dim"], native["pos"]
    C_ipl = align_to(read_aim(paths["_CORT_MASK"]), dim, pos) > 0
    T_ipl = align_to(read_aim(paths["_TRAB_MASK"]), dim, pos) > 0
    assert not (C_ipl & T_ipl).any()
    ref = sitk.GetImageFromArray(np.zeros(native["data"].shape, np.uint8))
    ref.SetSpacing([float(e) for e in native["el_size_mm"]])
    prx = array_to_sitk((C_ipl | T_ipl).astype(np.uint8), ref)
    cort, trab, info = step3_trab_cort_seg(native, prx, TIBIA)
    C, T = sitk_to_bool(cort), sitk_to_bool(trab)
    print(f"\nPFJ-0be66a step 3 with IPL's contour: CORT {int(C.sum()):,d} ({int((C != C_ipl).sum())} mismatches), "
          f"TRAB {int(T.sum()):,d} ({int((T != T_ipl).sum())} mismatches); chain {info['timings']['total']:.1f} s")
    assert info["thresholds"]["lower_native"] == 4524 and info["thresholds"]["upper_native"] == 17173
    assert info["calibration_source"] == "proclog"
    assert info["box"] == dict(dim=(738, 343, 168), pos=(798, 97, 168))
    assert info["periosteal"]["rendered_voxels"] == 27_484_219
    assert int(C.sum()) == 7_154_580 and int(T.sum()) == 20_329_639
    assert int((C != C_ipl).sum()) == 0 and int((T != T_ipl).sum()) == 0
