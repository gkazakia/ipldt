"""The grid of the cortical pore cascade: ipldt.porosity.render_grid (IPL's /gobj_to_aim grid of a contour, from its
rendering), cascade_grids and pore_cascade_ipl_grid (the cascade on IPL's grids from inputs held on any grid).

The rule (render_grid's docstring has the evidence): per slice with contour pixels the GOBJ slice header stores the
in-plane box of the slice's chains as size S = hi - lo + 1 and centre C = ceil((lo + hi) / 2); /gobj_to_aim renders
onto [ceil(min C - S/2) - 2, floor(max C + S/2) + 2] per in-plane axis, clipped below at 0, over the contour's
slices.  Equivalently low = min lo - 2 and high = max(hi + [S even]) + 2.

Synthetic phantoms show why the grid matters: the slice-wise 0..5 % passes of the cascade measure every component
against the non-bone voxels of its slice in the working grid, so (1) a grid larger than IPL's turns a small marrow
cavity into a "pore", and (2) a grid whose faces the contour touches cuts the background into corner pieces that
each pass as a "pore".  On IPL's grid neither happens, and the result no longer depends on the grid the inputs are
held on.  The slow test checks the patella PFJ-6f5538_R, whose pore map on the whole greyscale-AIM grid differs from
IPL's PORE.AIM by 25,428 voxels, against IPL's own export.
"""
import math
import os

import numpy as np
import pytest
from scipy import ndimage as ndi

from ipldt import ipl_ops as ops
from ipldt import porosity
from ipldt.contour import render_volume, slice_chains
from ipldt.io import align_to, read_aim


# ------------------------------------------------------------------------------------------- helpers
def volume(a, pos):
    """bool (z, y, x) array at global pos -> char 0 / 127 volume dict."""
    a = np.asarray(a, bool)
    return ops.mask_vol(a, a.shape[::-1], pos)


def on(v, dim, pos):
    """a volume dict pasted onto (dim, pos) by global position."""
    return ops.vol(ops.on_grid(v, dim, pos), dim, pos)


def header_rule(G, pos, margin=2, clip_low=0):
    """The /gobj_to_aim grid from the GOBJ slice headers ipldt.contour's stored chains would carry: per slice the
    box of the chains' vertices, S = hi - lo + 1, C = ceil((lo + hi) / 2), then [ceil(min C - S/2) - margin,
    floor(max C + S/2) + margin], low clipped at clip_low (never below the contour)."""
    lo_c, hi_c, lo_v, zs = [math.inf, math.inf], [-math.inf, -math.inf], [math.inf, math.inf], []
    for z in range(G.shape[0]):
        ch = slice_chains(G[z])
        if not ch:
            continue
        zs.append(z + pos[2])
        vx = [p for c in ch for p in c["vertices"]]
        for a in (0, 1):
            lo = min(p[a] for p in vx) + pos[a]
            hi = max(p[a] for p in vx) + pos[a]
            S, C = hi - lo + 1, math.ceil((lo + hi) / 2)
            lo_c[a], hi_c[a], lo_v[a] = min(lo_c[a], C - S / 2), max(hi_c[a], C + S / 2), min(lo_v[a], lo)
    low = [max(math.ceil(lo_c[a]) - margin, min(lo_v[a], clip_low)) for a in (0, 1)]
    high = [math.floor(hi_c[a]) + margin for a in (0, 1)]
    return (high[0] - low[0] + 1, high[1] - low[1] + 1, zs[-1] - zs[0] + 1), (low[0], low[1], zs[0])


def ring_phantom(Z=12, N=32, r_out=14.2, r_in=3.2):
    """A cortical ring (disc of radius r_out minus a small marrow cavity of radius r_in, centred on a half-integer
    so IPL's rendering is the identity) with two intracortical channels on slices 2..9; returns (contour, bone,
    marrow) bool (z, y, x)."""
    yy, xx = np.mgrid[:N, :N]
    c = (N - 1) / 2
    r = np.hypot(yy - c, xx - c)
    contour = np.broadcast_to((r <= r_out) & (r > r_in), (Z, N, N)).copy()
    bone = contour.copy()
    for cy, cx in ((c, c + 9.0), (c - 9.0, c)):
        bone[2:10, np.hypot(yy - cy, xx - cx) <= 1.3] = False
    return contour, bone, np.broadcast_to(r <= r_in, (Z, N, N)).copy()


def square_phantom(Z=12, N=40, chamfer=6, wall=4):
    """A cortical compartment shaped as a square with chamfered corners (rendering is the identity) whose wall is
    `wall` voxels thick, with one intracortical channel on slices 2..9; returns (contour, bone, the filled square)."""
    sq = np.ones((N, N), bool)
    for i in range(chamfer):
        for j in range(chamfer - i):
            for y, x in ((i, j), (i, N - 1 - j), (N - 1 - i, j), (N - 1 - i, N - 1 - j)):
                sq[y, x] = False
    full = np.broadcast_to(sq, (Z, N, N)).copy()
    inner = ndi.binary_erosion(full, structure=np.ones((1, 3, 3), bool), iterations=wall)
    contour = full & ~inner
    bone = contour.copy()
    bone[2:10, 20:22, 1:3] = False
    return contour, bone, full


# ------------------------------------------------------------------------------------------- the rule
def test_render_grid_rule_on_boxes():
    pos = (100, 200, 7)
    G = np.zeros((6, 30, 40), bool)
    G[1, 5:14, 10:20] = True                 # x 110..119 (10 wide, even), y 205..213 (9 wide, odd)
    v = volume(G, pos)
    assert porosity.render_grid(v) == ((15, 13, 1), (108, 203, 8))      # x high = 119 + 1 + 2, y high = 213 + 2
    assert porosity.tight_grid(v) == ((10, 9, 1), (110, 205, 8))
    # a second slice: odd width 11 reaching x 120 gives the same high end as the even slice's 119 + 1
    G[3, 5:14, 10:21] = True
    assert porosity.render_grid(volume(G, pos)) == ((15, 13, 3), (108, 203, 8))
    # an even slice reaching the highest x: high margin 3
    G[4, 6:12, 10:22] = True                 # x 110..121, 12 wide
    assert porosity.render_grid(volume(G, pos)) == ((17, 13, 4), (108, 203, 8))
    # only the parity of the slice that reaches highest counts
    G2 = np.zeros((2, 30, 40), bool)
    G2[0, 5:10, 10:22] = True                # x 110..121 even -> 122
    G2[1, 5:10, 8:23] = True                 # x 108..122 odd  -> 122
    assert porosity.render_grid(volume(G2, pos))[1][0] == 106 and porosity.render_grid(volume(G2, pos))[0][0] == 19
    assert porosity.render_grid(volume(np.zeros((3, 4, 5), bool), pos)) is None


def test_render_grid_clips_at_zero_and_never_cuts_the_contour():
    G = np.zeros((2, 20, 20), bool)
    G[:, 0:9, 3:12] = True
    assert porosity.render_grid(volume(G, (0, 0, 0)))[1][:2] == (1, 0)          # y would start at -2: clipped to 0
    assert porosity.render_grid(volume(G, (0, 1, 0)))[1][:2] == (1, 0)          # y 1 - 2 = -1 -> 0
    assert porosity.render_grid(volume(G, (0, 2, 0)))[1][:2] == (1, 0)          # y 2 - 2 = 0
    assert porosity.render_grid(volume(G, (0, 5, 0)))[1][:2] == (1, 3)
    # a contour at negative positions keeps all its voxels (margin 0 there: unobserved in IPL)
    dim, p = porosity.render_grid(volume(G, (-4, -1, 0)))
    assert p[:2] == (-1, -1) and dim[:2] == (11, 11)                           # -1 .. 7 + 2
    assert porosity.render_grid(volume(G, (0, 0, 0)), clip_low=None)[1][:2] == (1, -2)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_render_grid_equals_the_slice_header_formula(seed):
    """render_grid reads the rendering's per-slice extent; the formula reads the stored chains' vertices through
    the header's rounding.  They agree on random smooth contours (the chains span exactly their rendering)."""
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.standard_normal((9, 48, 56)), (1.0, 4.0, 4.0))
    mask = f > np.quantile(f, 0.55)
    mask[:, :2] = mask[:, -2:] = False
    mask[:, :, :2] = mask[:, :, -2:] = False
    mask[rng.integers(0, 9)] = False                                   # a slice without contour
    G = render_volume(mask)
    for pos in ((0, 0, 0), (301, 58, 11), (7, 1, 0)):
        assert porosity.render_grid(volume(G, pos)) == header_rule(G, pos), pos


def test_cascade_grids():
    contour, bone, _ = ring_phantom()
    pos = (50, 60, 5)
    R, S, U = porosity.cascade_grids(volume(contour, pos), volume(bone, pos))
    assert R == ((33, 33, 12), (50, 60, 5))       # tight 28 x 28 @ 52, 62: low - 2, high + 2 + 1 (even width)
    assert S == porosity.tight_grid(volume(bone, pos)) and U == R
    # an explicit CORT_SEG grid reaching outside R widens the cascade grid (IPL's y = -1 row on two patellae)
    R2, S2, U2 = porosity.cascade_grids(volume(contour, pos), volume(bone, pos), seg_grid=((28, 29, 12), (52, 59, 5)))
    assert R2 == R and U2 == ((33, 34, 12), (50, 59, 5))
    with pytest.raises(ValueError, match="empty"):
        porosity.cascade_grids(volume(np.zeros_like(contour), pos), volume(bone, pos))


# ------------------------------------------------------------------------------------------- the grid matters
def test_larger_grid_turns_the_marrow_into_a_pore():
    """The denominator effect: on a grid larger than IPL's, the marrow cavity (outside the cortical contour) falls
    under 5 % of its slice's non-bone voxels and is kept as a pore; on IPL's grid it is not."""
    contour, bone, marrow = ring_phantom()
    pos = (50, 60, 5)
    own = contour.shape[::-1]
    R = porosity.render_grid(volume(contour, pos))
    on_R = porosity.pore_cascade(on(volume(contour, pos), *R), on(volume(bone, pos), *R))["pore"]
    ref = ops.on_grid(on_R, own, pos) != 0
    assert int(ref.sum()) == np.count_nonzero(on_R["data"]) > 0 and not (ref & marrow).any()
    for pad in (3, 20):
        dim, gpos = (own[0] + 2 * pad, own[1] + 2 * pad, own[2]), (pos[0] - pad, pos[1] - pad, pos[2])
        cr, cs = on(volume(contour, pos), dim, gpos), on(volume(bone, pos), dim, gpos)
        full = ops.on_grid(porosity.pore_cascade(cr, cs)["pore"], own, pos) != 0
        assert (full & marrow).sum() == marrow.sum(), "on the padded grid the whole marrow cavity is a 'pore'"
        out = porosity.pore_cascade_ipl_grid(cr, cs)
        assert out["grid"] == R and out["render_grid"] == R
        assert np.array_equal(ops.on_grid(out["pore"], own, pos) != 0, ref)
        assert porosity.compare_pore(out["pore"], on_R)["mismatch"] == 0


def test_tight_grid_cuts_the_background_into_pores():
    """The edge effect: when the contour touches the faces of the working grid, the background outside it is cut
    into corner pieces that each pass the 5 % test; IPL's grid reaches 2-3 voxels past the contour (zero padding
    past the AIM), so the background stays one component."""
    contour, bone, filled = square_phantom()
    pos = (50, 60, 5)
    own = contour.shape[::-1]
    outside = ~filled                                                  # outside the periosteal surface
    cr, cs = volume(contour, pos), volume(bone, pos)                  # the AIM grid = the contour's tight box
    full = porosity.pore_cascade(cr, cs)["pore"]
    assert (ops.on_grid(full, own, pos)[outside] != 0).sum() > 900, "corner fragments kept as pores"
    out = porosity.pore_cascade_ipl_grid(cr, cs)
    assert out["grid"] == ((45, 45, 12), (48, 58, 5))
    got = ops.on_grid(out["pore"], own, pos) != 0
    assert not got[outside].any() and got.sum() == np.count_nonzero(out["pore"]["data"]) > 0
    pad = 5
    dim, gpos = (own[0] + 2 * pad, own[1] + 2 * pad, own[2]), (pos[0] - pad, pos[1] - pad, pos[2])
    assert porosity.compare_pore(porosity.pore_cascade(on(cr, dim, gpos), on(cs, dim, gpos))["pore"], out["pore"])["mismatch"] == 0


@pytest.mark.parametrize("phantom", ["ring", "square"])
def test_pore_cascade_ipl_grid_does_not_depend_on_the_input_grid(phantom):
    contour, bone = (ring_phantom() if phantom == "ring" else square_phantom())[:2]
    pos = (50, 60, 5)
    own = contour.shape[::-1]
    cr0, cs0 = volume(contour, pos), volume(bone, pos)
    ref = porosity.pore_cascade_ipl_grid(cr0, cs0)
    for lo, hi in ((0, 0), (1, 7), (6, 2), (25, 25)):
        dim, gpos = (own[0] + lo + hi, own[1] + hi, own[2] + 2), (pos[0] - lo, pos[1], pos[2] - 1)
        out = porosity.pore_cascade_ipl_grid(on(cr0, dim, gpos), on(cs0, dim, gpos))
        assert out["grid"] == ref["grid"] and out["render_grid"] == ref["render_grid"]
        assert np.array_equal(out["pore"]["data"], ref["pore"]["data"])
        assert porosity.ct_po(out["pore"], on(cr0, dim, gpos)) == porosity.ct_po(ref["pore"], cr0)
    assert np.count_nonzero(ref["pore"]["data"]) > 0
    # the contour the cascade saw is the input contour on R, voxel for voxel
    assert np.array_equal(ops.on_grid(ref["cort_render"], own, pos) != 0, contour)


def test_pore_cascade_ipl_grid_keeps_stages_and_explicit_seg_grid():
    contour, bone, _ = ring_phantom()
    pos = (50, 60, 5)
    cr, cs = volume(contour, pos), volume(bone, pos)
    out = porosity.pore_cascade_ipl_grid(cr, cs, keep_stages=True)
    assert out["pores_E"]["dim"] == out["seg_grid"][0] and tuple(out["cortring_C"]["dim"]) == out["render_grid"][0]
    assert tuple(out["pore"]["dim"]) == out["grid"][0] and tuple(out["pore"]["pos"]) == out["grid"][1]
    # CORT_SEG on a wider explicit grid (IPL's seg box) leaves the pore map alone here: pores_E is inert
    S = ((30, 30, 12), (51, 61, 5))
    alt = porosity.pore_cascade_ipl_grid(cr, cs, seg_grid=S)
    assert alt["seg_grid"] == S and porosity.compare_pore(alt["pore"], out["pore"])["mismatch"] == 0


# ------------------------------------------------------------------------------------------- real data (slow)
@pytest.mark.slow
def test_pfj_6f5538_pore_map_from_the_greyscale_grid(data_root):
    """PFJ-6f5538_R (X5492058), a flat patella whose pore map on the whole greyscale-AIM grid differs from IPL's
    PORE.AIM by 25,428 voxels: with IPL's own inputs pasted onto the greyscale grid (the way the workflows hold
    them), pore_cascade_ipl_grid reproduces IPL's PORE.AIM exactly, and render_grid of the rendered contour united
    with IPL's CORT_SEG grid is IPL's PORE.AIM grid."""
    folder = os.path.join(data_root, "PFJ-6f5538_R")
    f = lambda tag: os.path.join(folder, f"X5492058_{tag}.AIM")        # noqa: E731
    need = [f("CORT_MASK_decompressed"), f("CORT_SEG_decompressed"), f("PORE_decompressed"), os.path.join(folder, "X5492058.AIM")]
    for p in need:
        if not os.path.exists(p):
            pytest.skip(f"missing {p}")
    grey = read_aim(need[3])
    dim, pos = tuple(grey["dim"]), tuple(grey["pos"])
    del grey
    raw = read_aim(need[0])
    G = render_volume(raw["data"] != 0)
    cr_own = ops.mask_vol(G, raw["dim"], raw["pos"])
    cs_file = read_aim(need[1])
    ipl = read_aim(need[2])
    R = porosity.render_grid(cr_own)
    assert R == ((746, 200, 168), (744, 0, 168))                          # IPL's own /gobj_to_aim grid (its export, not distributed)
    Sg = (tuple(cs_file["dim"]), tuple(cs_file["pos"]))
    assert porosity._union(R, Sg) == (tuple(ipl["dim"]), tuple(ipl["pos"]))
    cr = ops.vol(align_to(cr_own, dim, pos), dim, pos)
    cs = ops.mask_vol(align_to(cs_file, dim, pos) != 0, dim, pos)
    full = porosity.pore_cascade(cr, cs)["pore"]
    assert porosity.compare_pore(full, ipl)["ours_only"] == 25_428
    out = porosity.pore_cascade_ipl_grid(cr, cs)
    c = porosity.compare_pore(out["pore"], ipl)
    assert c["mismatch"] == 0 and c["ipl"] == 2_617_440
    assert porosity.ct_po(out["pore"], cr)["ct_po"] == porosity.ct_po(ipl, cr)["ct_po"]
