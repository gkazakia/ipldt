"""The report block against a hand computation, and dt_* on structures whose IPL answer is
known in closed form (a slab, a lone voxel, a cube)."""
import numpy as np
import pytest

from ipldt import dt_number, dt_spacing, dt_thickness, statistics


# ------------------------------------------------------------------------------ statistics
def test_statistics_matches_hand_computation():
    vals = [1, 2, 2, 3, 5, 8]
    m = np.zeros((2, 3, 4), np.int16)
    obj = np.zeros((2, 3, 4), bool)
    for k, v in enumerate(vals):
        m[0, k // 4, k % 4] = v
        obj[0, k // 4, k % 4] = True
    obj[1, 0, :4] = True                                   # four object voxels with no value
    vox = 0.5
    rep = statistics(m, obj, vox, name="Th")
    v = np.array(vals, float) * vox
    mean = v.sum() / len(v)
    d = v - mean
    m2 = (d ** 2).sum() / len(v)
    assert rep["Th_mm"] == pytest.approx(mean)
    assert rep["Th_sd_mm"] == pytest.approx(np.sqrt(m2))              # population SD (ddof = 0)
    assert rep["Th_max_mm"] == 8 * vox
    assert rep["Th_n_voxels"] == 6
    assert rep["valid_fraction"] == pytest.approx(6 / 10)
    assert rep["voxel_size_mm"] == vox
    assert rep["Th_skew"] == pytest.approx((d ** 3).sum() / len(v) / m2 ** 1.5)
    assert rep["Th_kurtosis"] == pytest.approx((d ** 4).sum() / len(v) / m2 ** 2 - 3.0)
    # the name prefixes every statistic
    assert set(k for k in statistics(m, obj, vox, name="Sp") if k.startswith("Sp_")) == \
        {"Sp_mm", "Sp_sd_mm", "Sp_max_mm", "Sp_n_voxels", "Sp_skew", "Sp_kurtosis"}


def test_statistics_edge_cases():
    empty = statistics(np.zeros((2, 2, 2), np.int16), np.ones((2, 2, 2), bool), 0.1)
    assert empty["Th_mm"] == 0.0 and empty["Th_n_voxels"] == 0 and empty["valid_fraction"] == 0.0
    assert "Th_skew" not in empty
    two = np.zeros((1, 1, 3), np.int16)
    two[0, 0, :2] = [2, 4]
    rep = statistics(two, two > 0, 1.0)
    assert rep["Th_mm"] == 3.0 and rep["Th_sd_mm"] == 1.0 and "Th_skew" not in rep      # n <= 2: no moments
    const = np.full((1, 2, 2), 3, np.int16)
    rep = statistics(const, const > 0, 1.0)
    assert rep["Th_skew"] == 0.0 and rep["Th_kurtosis"] == 0.0                          # zero variance guard
    assert statistics(const, np.zeros((1, 2, 2), bool), 1.0)["valid_fraction"] == 4.0    # obj.sum() == 0 guard


def test_voxel_size_scales_the_report_only(phantom):
    a = dt_thickness(phantom, voxel_size_mm=1.0, backend="cpu")
    b = dt_thickness(phantom, voxel_size_mm=0.0607, backend="cpu")
    assert np.array_equal(a.map, b.map)
    for k in ("Th_mm", "Th_sd_mm", "Th_max_mm"):
        assert b.report[k] == pytest.approx(a.report[k] * 0.0607)
    for k in ("Th_n_voxels", "valid_fraction", "Th_skew", "Th_kurtosis"):
        assert b.report[k] == pytest.approx(a.report[k])


# ------------------------------------------------------------------------------ closed-form structures
def slab(t, nz=15, ny=12, nx=10):
    """A slab of thickness t spanning the whole x-y extent: outside the image is object, so its
    only surfaces are the two z faces and the problem is one-dimensional."""
    obj = np.zeros((nz, ny, nx), bool)
    z0 = (nz - t) // 2
    obj[z0:z0 + t] = True
    return obj, z0


@pytest.mark.parametrize("t", [3, 5, 7])
@pytest.mark.parametrize("version", [1, 2, 3])
def test_odd_slab_all_versions(t, version):
    obj, z0 = slab(t)
    r = dt_thickness(obj, voxel_size_mm=0.1, version=version, backend="cpu")
    assert (r.map[obj] == t).all() and not r.map[~obj].any()
    mid = np.zeros_like(obj)
    mid[z0 + t // 2] = True
    assert np.array_equal(r.centres, mid), "only the middle plane survives the containment ridge"
    assert r.report["Th_mm"] == pytest.approx(t * 0.1)
    assert r.report["Th_sd_mm"] == pytest.approx(0.0, abs=1e-12)
    assert r.report["valid_fraction"] == 1.0 and r.report["Th_n_voxels"] == int(obj.sum())


@pytest.mark.parametrize("t", [4, 6])
def test_even_slab_versions_differ(t):
    """Even thickness: version 1 rounds 2 s_x = t - 1 down (t - 1); versions 2/3 pin the sphere
    between the two contact points and recover t (the hand computation in the state file)."""
    obj, z0 = slab(t)
    v1 = dt_thickness(obj, version=1, backend="cpu")
    v2 = dt_thickness(obj, version=2, backend="cpu")
    v3 = dt_thickness(obj, version=3, backend="cpu")
    assert (v1.map[obj] == t - 1).all()
    assert (v2.map[obj] == t).all() and (v3.map[obj] == t).all()
    two_mid = np.zeros_like(obj)
    two_mid[z0 + t // 2 - 1:z0 + t // 2 + 1] = True
    for r in (v1, v2, v3):
        assert np.array_equal(r.centres, two_mid) and not r.map[~obj].any()


def test_lone_voxel_per_version():
    """s_x = 1/2 -> version 1 gives floor(1 + 1/2) = 1.  Version 3 pins the sphere between the
    contact point (1/2) and the centre of the background voxel behind it (P2 = step, -1):
    D = 3/2, |M| = 1/4 < 1 -> 2.  Version 2 requires y to be object, so it falls back to 1."""
    obj = np.zeros((11, 11, 11), bool)
    obj[5, 5, 5] = True
    for version, expected in ((1, 1), (2, 1), (3, 2)):
        r = dt_thickness(obj, version=version, backend="cpu")
        assert r.map[5, 5, 5] == expected and int((r.map > 0).sum()) == 1 and r.centres[5, 5, 5]


@pytest.mark.parametrize("version", [1, 2, 3])
def test_cube(version):
    """5-cube: only the centre voxel survives the containment ridge (s = 2.5, every neighbour
    has a smaller s and |x - y| + s_x - s_y >= 1 > 0.9), its diameter is 5 under every version,
    and its sphere (|c - x| <= 3) reaches every cube voxel except the 8 corners (|c - x| = sqrt 12)."""
    obj = np.zeros((11, 11, 11), bool)
    obj[3:8, 3:8, 3:8] = True
    r = dt_thickness(obj, version=version, backend="cpu")
    assert int(r.centres.sum()) == 1 and r.centres[5, 5, 5]
    assert int(r.map.max()) == 5 and r.map[5, 5, 5] == 5
    assert set(np.unique(r.map[obj]).tolist()) == {0, 5}
    assert int((r.map[obj] > 0).sum()) == 125 - 8
    for z in (3, 7):
        for y in (3, 7):
            for x in (3, 7):
                assert r.map[z, y, x] == 0
    assert not r.map[~obj].any()
    assert r.report["valid_fraction"] == pytest.approx(117 / 125)


def test_spacing_is_thickness_of_the_complement(phantom, phantom_gobj):
    """Probe 13: IPL's dt_thickness on the marrow AIM equals its dt_spacing on the bone."""
    sp = dt_spacing(phantom, gobj={"rendered": phantom_gobj}, voxel_size_mm=0.0607, backend="cpu")
    th = dt_thickness(~phantom, gobj={"rendered": phantom_gobj}, voxel_size_mm=0.0607, backend="cpu")
    assert np.array_equal(sp.map, th.map) and np.array_equal(sp.centres, th.centres)
    assert sp.report["Sp_mm"] == th.report["Th_mm"]
    assert not sp.map[phantom].any() and not sp.map[~phantom_gobj].any()
    assert sp.report["valid_fraction"] == pytest.approx(int((sp.map > 0).sum()) / int((~phantom & phantom_gobj).sum()))


def test_number_report(phantom, phantom_gobj):
    n = dt_number(phantom, gobj={"rendered": phantom_gobj}, voxel_size_mm=0.0607, backend="cpu")
    assert n.report["Tb_N_per_mm"] == pytest.approx(1.0 / n.report["inv_N_mm"])
    assert n.report["inv_N_mm"] == pytest.approx(n.map[n.map > 0].mean() * 0.0607)
    assert not n.map[~phantom_gobj].any()
    # the mid-axis (containment ridge of the bone) never carries a value: it is the background here
    from ipldt.core import ridge
    from ipldt.field import sir_quad
    S = ridge(phantom, sir_quad(phantom), 0.9)
    assert S.any() and not n.map[S].any()
