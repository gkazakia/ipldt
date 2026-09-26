"""IPL parameter semantics as derived on the scanner (state file, test run 12):
assign_epsilon is the sphere-drawing tolerance (so the map grows monotonically with it),
peel_iter erodes the rendered gobj slice-wise with 4-connectivity before the centre selection,
suppress_boundary has no effect, and gobj accepts a raw raster, {'mask': ...}, {'rendered': ...} or None."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from ipldt import dt_number, dt_spacing, dt_thickness, peel_gobj, render_volume
from ipldt.core import resolve_gobj

S4_2D = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
EPSILONS = (0.0, 0.25, 0.5, 1.0)


# ------------------------------------------------------------------------------ assign_epsilon
@pytest.mark.parametrize("fn", [dt_thickness, dt_spacing, dt_number], ids=["thickness", "spacing", "number"])
def test_assign_epsilon_is_monotone(phantom, phantom_gobj, fn):
    results = [fn(phantom, gobj={"rendered": phantom_gobj}, assign_epsilon=e, backend="cpu") for e in EPSILONS]
    for lo, hi in zip(results, results[1:]):
        assert (hi.map >= lo.map).all(), "a larger drawing tolerance can only raise a voxel's value"
        assert int((hi.map > 0).sum()) >= int((lo.map > 0).sum())
        assert np.array_equal(hi.centres, lo.centres), "assign_epsilon must not touch the centre set"
    assert not np.array_equal(results[0].map, results[-1].map), "assign_epsilon had no effect on this phantom"
    # the support never leaves the object / gobj whatever the tolerance
    for r in results:
        assert not (r.map[~phantom_gobj]).any()


@pytest.mark.parametrize("bad_shape", [(1, 12, 12), (6, 10, 10)], ids=["one-slice", "smaller-grid"])
def test_gobj_on_another_grid_is_refused(phantom, bad_shape):
    """A gobj raster must live on the image grid (callers paste by global position first); a wrong grid
    is a ValueError at the entry, not an IndexError or a broadcast error deep inside the sweep."""
    with pytest.raises(ValueError, match="different grids"):
        dt_thickness(phantom, gobj={"rendered": np.ones(bad_shape, bool)}, backend="cpu")
    with pytest.raises(ValueError, match="different grids"):
        resolve_gobj(np.ones(bad_shape, bool), phantom.shape)


def test_assign_epsilon_zero_is_the_tight_ball():
    """With assign_epsilon = 0 a centre of diameter D only reaches voxels with |c - x| <= D/2:
    a lone voxel (D = 1 under version 1) covers nothing but itself.  Version 3 pins the sphere
    against the far-side background voxel centre (D = 1.5 -> 2), still only covering itself."""
    obj = np.zeros((5, 5, 5), bool)
    obj[2, 2, 2] = True
    r = dt_thickness(obj, assign_epsilon=0.0, version=1, backend="cpu")
    assert r.map[2, 2, 2] == 1 and int((r.map > 0).sum()) == 1
    r = dt_thickness(obj, assign_epsilon=0.0, version=3, backend="cpu")
    assert r.map[2, 2, 2] == 2 and int((r.map > 0).sum()) == 1


# ------------------------------------------------------------------------------ peel_iter
@pytest.mark.parametrize("n", [1, 2, 3])
def test_peel_gobj_is_slicewise_4connected_erosion(phantom_gobj, n):
    expected = np.stack([ndi.binary_erosion(sl, structure=S4_2D, iterations=n, border_value=0)
                         for sl in phantom_gobj])
    got = peel_gobj(phantom_gobj, n)
    assert np.array_equal(got, expected)
    assert int(got.sum()) < int(phantom_gobj.sum())
    # not an 8-connected in-plane erosion (which would also remove diagonal-only pixels)
    eight = np.stack([ndi.binary_erosion(sl, structure=np.ones((3, 3), bool), iterations=n, border_value=0)
                      for sl in phantom_gobj])
    assert not np.array_equal(got, eight)


def test_peel_gobj_does_not_couple_slices():
    """A one-slice-thick plate survives an in-plane peel (a 3-D 6-connected erosion would erase it)."""
    plate = np.zeros((5, 20, 20), bool)
    plate[2, 3:17, 3:17] = True
    p1 = peel_gobj(plate, 1)
    assert p1[2, 4:16, 4:16].all() and int(p1.sum()) == 12 * 12
    assert not ndi.binary_erosion(plate, structure=ndi.generate_binary_structure(3, 1), iterations=1).any()


def test_peel_iter_automatic_and_zero_are_no_ops(phantom_gobj):
    assert peel_gobj(phantom_gobj, -1) is phantom_gobj
    assert peel_gobj(phantom_gobj, 0) is phantom_gobj
    assert peel_gobj(phantom_gobj, None) is phantom_gobj


@pytest.mark.parametrize("n", [1, 2])
def test_peel_iter_removes_centres_only(phantom, phantom_gobj, n):
    """peel_iter N keeps only the centres inside the N-times-eroded contour; the map is still
    clipped to the unpeeled contour (IPL: 'mask with gobj (peel 0)' ... 'mask with gobj (peel -1)')."""
    r0 = dt_thickness(phantom, gobj={"rendered": phantom_gobj}, peel_iter=0, backend="cpu")
    rm = dt_thickness(phantom, gobj={"rendered": phantom_gobj}, peel_iter=-1, backend="cpu")
    rn = dt_thickness(phantom, gobj={"rendered": phantom_gobj}, peel_iter=n, backend="cpu")
    assert np.array_equal(r0.map, rm.map) and np.array_equal(r0.centres, rm.centres)
    assert np.array_equal(rn.centres, r0.centres & peel_gobj(phantom_gobj, n))
    assert int(rn.centres.sum()) < int(r0.centres.sum())
    assert not rn.map[~phantom_gobj].any()
    assert (rn.map <= r0.map).all()          # fewer spheres, never a larger value


# ------------------------------------------------------------------------------ gobj forms
def test_gobj_forms(phantom, raw_mask, clean_mask, phantom_gobj):
    G = render_volume(raw_mask)
    assert np.array_equal(G, phantom_gobj)
    assert not np.array_equal(raw_mask, clean_mask) and int(raw_mask.sum()) > int(clean_mask.sum())
    assert not np.array_equal(G, raw_mask), "the rendering must remove the 1-pixel features of the raw mask"
    assert np.array_equal(G, clean_mask), "the rendering of the bumped disc is the clean disc"
    assert np.array_equal(render_volume(clean_mask), clean_mask), "a clean digital disc is rendering-invariant"
    assert np.array_equal(resolve_gobj(None, phantom.shape), np.ones(phantom.shape, bool))
    assert np.array_equal(resolve_gobj(raw_mask, phantom.shape), G)
    assert np.array_equal(resolve_gobj({"mask": raw_mask}, phantom.shape), G)
    assert resolve_gobj({"rendered": raw_mask}, phantom.shape).dtype == bool
    assert np.array_equal(resolve_gobj({"rendered": raw_mask}, phantom.shape), raw_mask)
    a = dt_thickness(phantom, gobj=raw_mask, backend="cpu")
    b = dt_thickness(phantom, gobj={"mask": raw_mask}, backend="cpu")
    c = dt_thickness(phantom, gobj={"rendered": G}, backend="cpu")
    d = dt_thickness(phantom, gobj={"rendered": raw_mask}, backend="cpu")
    assert np.array_equal(a.map, b.map) and np.array_equal(b.map, c.map)
    assert not np.array_equal(c.map, d.map), "the rendered and the raw raster must give different maps"
    # a uint8 raster (as read from an AIM) is accepted for a rendered gobj
    e = dt_thickness(phantom, gobj={"rendered": G.astype(np.uint8)}, backend="cpu")
    assert np.array_equal(c.map, e.map)


def test_gobj_masks_map_and_centres_but_not_the_field(phantom, phantom_gobj):
    """The gobj boundary is not a surface: values inside the contour equal the unmasked values."""
    full = dt_thickness(phantom, backend="cpu")
    masked = dt_thickness(phantom, gobj={"rendered": phantom_gobj}, backend="cpu")
    assert not masked.map[~phantom_gobj].any()
    assert not masked.centres[~phantom_gobj].any()
    assert np.array_equal(masked.centres, full.centres & phantom_gobj)
    assert (masked.map <= full.map).all()
    # voxels no sphere centred outside the contour can reach: deeper than max D / 2 + assign_epsilon
    inner = ndi.distance_transform_edt(phantom_gobj) > full.map.max() / 2 + 0.5
    assert inner.any()
    assert np.array_equal(masked.map[inner], full.map[inner])


# ------------------------------------------------------------------------------ other parameters
@pytest.mark.parametrize("fn", [dt_thickness, dt_spacing, dt_number], ids=["thickness", "spacing", "number"])
def test_suppress_boundary_has_no_effect(phantom, phantom_gobj, fn):
    ref = fn(phantom, gobj={"rendered": phantom_gobj}, suppress_boundary=2, backend="cpu")
    for sb in (0, 1, 3):
        r = fn(phantom, gobj={"rendered": phantom_gobj}, suppress_boundary=sb, backend="cpu")
        assert np.array_equal(r.map, ref.map) and np.array_equal(r.centres, ref.centres)


def test_ridge_epsilon_changes_the_centre_set(phantom):
    r0 = dt_thickness(phantom, ridge_epsilon=0.0, backend="cpu")
    r9 = dt_thickness(phantom, ridge_epsilon=0.9, backend="cpu")
    assert int(r0.centres.sum()) > int(r9.centres.sum()), "a looser containment test prunes more centres"
    assert (r9.centres <= r0.centres).all(), "the eps-0.9 centres are a subset of the eps-0 centres"


def test_versions_are_distinct_and_valid(phantom):
    maps = [dt_thickness(phantom, version=v, backend="cpu").map for v in (1, 2, 3)]
    assert not np.array_equal(maps[0], maps[2]), "version 1 and 3 coincide on this phantom"
    with pytest.raises(ValueError):
        dt_thickness(phantom, version=4, backend="cpu")


def test_result_container(phantom, phantom_gobj):
    r = dt_thickness(phantom, gobj={"rendered": phantom_gobj}, voxel_size_mm=0.0607, backend="cpu")
    assert r.map.shape == phantom.shape and r.centres.shape == phantom.shape
    assert r.map.dtype == np.int16 and r.centres.dtype == bool
    assert r["Th_mm"] == r.report["Th_mm"] and r.report["voxel_size_mm"] == 0.0607
    assert set(r.report) >= {"Th_mm", "Th_sd_mm", "Th_max_mm", "Th_n_voxels", "valid_fraction", "voxel_size_mm"}
