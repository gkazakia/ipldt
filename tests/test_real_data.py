"""Real-data validation against IPL's own exported map (slow): PFJ-0be66a_R (X2420448) Tb.Sp.

Script 32 computes TRAB_SP as `/dt_spacing` of the SEG (cortical 127 + trabecular 126) with the
trabecular mask contour as gobj.  Here the mask is rendered with ipldt's contour rules on its
own grid, aligned to the SEG grid by global position, and dt_spacing must reproduce TRAB_SP on
every voxel.  Skipped when the validation folder is absent (set IPLDT_DATA_ROOT to relocate it).
"""
import glob
import os
import time

import numpy as np
import pytest

from ipldt import align_to, dt_spacing, read_aim, render_volume

pytestmark = pytest.mark.slow

SUBJECT = "PFJ-0be66a_R"
IPL_TRAB_GOBJ_VOXELS = 20_328_498       # IPL's logged 'inmask_nr' for X2420448's trabecular contour


def subject_base(folder):
    segs = [q for q in glob.glob(os.path.join(folder, "*_SEG_decompressed.AIM"))
            if "TRAB_" not in os.path.basename(q) and "CORT_" not in os.path.basename(q)]
    if not segs:
        pytest.skip(f"no *_SEG_decompressed.AIM in {folder}")
    return os.path.basename(segs[0]).replace("_SEG_decompressed.AIM", "")


@pytest.fixture(scope="module")
def pfj_0be66a(data_root):
    folder = os.path.join(data_root, SUBJECT)
    if not os.path.isdir(folder):
        pytest.skip(f"validation data folder not found: {folder}")
    base = subject_base(folder)
    f = lambda tag: os.path.join(folder, f"{base}_{tag}_decompressed.AIM")   # noqa: E731
    for tag in ("SEG", "TRAB_MASK", "TRAB_SP"):
        if not os.path.exists(f(tag)):
            pytest.skip(f"missing {f(tag)}")
    seg_aim = read_aim(f("SEG"))
    dim, pos = seg_aim["dim"], seg_aim["pos"]
    mask_aim = read_aim(f("TRAB_MASK"))
    t0 = time.time()
    G_mask_grid = render_volume(mask_aim["data"] > 0)                      # render on the mask's OWN grid
    t_render = time.time() - t0
    G = align_to(dict(data=G_mask_grid.astype(np.uint8), dim=mask_aim["dim"], pos=mask_aim["pos"]), dim, pos) > 0
    sp = align_to(read_aim(f("TRAB_SP")), dim, pos).astype(np.int16)
    return dict(base=base, seg=seg_aim["data"] > 0, el=float(seg_aim["el_size_mm"][0]), G=G,
                G_mask_grid_count=int(G_mask_grid.sum()), sp=sp, t_render=t_render, dim=dim, pos=pos)


def test_rendered_trabecular_gobj_count_equals_ipl_log(pfj_0be66a):
    assert pfj_0be66a["base"] == "X2420448"
    assert pfj_0be66a["G_mask_grid_count"] == IPL_TRAB_GOBJ_VOXELS
    assert int(pfj_0be66a["G"].sum()) == IPL_TRAB_GOBJ_VOXELS            # the SEG grid contains the whole contour


def test_pfj_0be66a_tb_sp_has_zero_mismatches(pfj_0be66a):
    seg, G, sp = pfj_0be66a["seg"], pfj_0be66a["G"], pfj_0be66a["sp"]
    assert seg.shape == sp.shape == G.shape == tuple(pfj_0be66a["dim"][::-1])
    t0 = time.time()
    r = dt_spacing(seg, gobj={"rendered": G}, voxel_size_mm=pfj_0be66a["el"], backend="auto")
    t_dt = time.time() - t0
    diff = r.map != sp
    n_mis = int(diff.sum())
    print(f"\nPFJ-0be66a Tb.Sp: {n_mis} mismatching voxels of {sp.size:,d}; centres {int(r.centres.sum()):,d}; "
          f"max D {int(r.map.max())}; render {pfj_0be66a['t_render']:.1f} s, dt_spacing {t_dt:.1f} s; "
          f"Tb.Sp {r.report['Sp_mm']:.5f} mm ({r.report['Sp_mm'] / pfj_0be66a['el']:.5f} vox)")
    assert n_mis == 0, f"{n_mis} voxels differ from IPL's TRAB_SP"
    assert int(sp.max()) == int(r.map.max()) and int((sp > 0).sum()) == r.report["Sp_n_voxels"]
    assert 0.05 < pfj_0be66a["el"] < 0.07                                       # XtremeCT II 60.7 um protocol
