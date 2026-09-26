"""IPL's contour rendering (/togobj_from_aim -curvature_smooth 1 followed by /gobj_to_aim) on synthetic
masks: the phase-1 pixel rule with its two exceptions (trace start, vertical neck), the staged smoothing
and its raw-store gate (both halves: stage-A events and pre-activation), stage A being exactly one sweep,
the sweep order and in-place moves, the inner-chain rules, the even-odd polygon fill of the inner contour,
the stored-chain conventions and the Freeman round trip; plus one slow real-data check against IPL's
exported cortical rendering (rasters, and every stored chain in stored order).

The rules are the data-derived ones of ipldt.contour.render / ipldt.contour.smooth (2026-09-13): where an
expectation below differs from the earlier synthetic intuition, the docstring says which rule decides it.
The 'conventions' section pins choices the cohort cannot discriminate (connectivity, a diagonal neck, a
component nested in another component's hole): they are the package's convention, labelled as such, so that
a change is deliberate -- not IPL-verified behaviour.  The tiny-component rule (a chain the smoothing leaves
at 3 vertices is a collapsing chain and is not stored: MIN_VERTICES = 4, 2026-09-14) IS IPL-verified: PFJ-6f5538_R
T17 29_trabfinal z332 (8-px hexagon, IPL stores no contour) and the frozen-at-3 chains of
X3931708_CORT_MASK_version2 / the 8-element contours of X2420448_CORT_MASK.GOBJ (version 1).  The MIN_VERTICES class is
pinned by the test-run-19 tiny-hole phantom (2026-09-14; a test on the test-run-19 exports, which are not
distributed, so it skips without them): IPL stores the raw un-activated 6-vertex inner chain of a vertical 1x2
hole and keeps the hole (4..6 confirmed, 7..8 refuted; 4 vs 5 vs 6 is not observable), and the 17 other tiny
holes / components render and chain exactly as the rule says.
"""
import json
import os
import re

import numpy as np
import pytest
from conftest import lab_path  # noqa: E402  (non-public data roots)
from scipy import ndimage as ndi

from ipldt.contour import render_slice, render_volume, slice_chains, chain_codes, gobj_path, read_gobj
from ipldt.contour.gobj_file import decode
from ipldt.contour.moore import moore_trace, codes_of
from ipldt.contour.render import phase1, outer_components, raster_first, polygon_fill, holes, S4, MIN_VERTICES
from ipldt.contour.smooth import smooth_chain, sweep, STAGE_A_RULES, ALL_RULES, MAX_SWEEPS

Y0, Y1, X0, X1 = 10, 30, 8, 32          # the block occupies rows Y0..Y1-1, columns X0..X1-1
RUN15 = os.environ.get("IPLDT_RUN15", lab_path("ipl_test_runs/run15/aims_and_logs"))


def block(shape=(40, 40)):
    sl = np.zeros(shape, bool)
    sl[Y0:Y1, X0:X1] = True
    return sl


def disc(shape=(40, 40), radius=14.0):
    """A digital disc with a half-integer centre: its boundary trace has no corner, spur, 135-degree
    turn or 1-pixel excursion, so IPL stores its chain raw and the rendering equals the mask."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - (shape[0] - 1) / 2) ** 2 + (xx - (shape[1] - 1) / 2) ** 2 <= radius ** 2


@pytest.fixture(scope="module")
def clean():
    return render_slice(block())


# ------------------------------------------------------------------------------------ block: the smoothing
def test_clean_block_keeps_its_body(clean):
    """Only the four convex 90-degree corners are rounded off; the rest of the block is kept."""
    body = np.zeros_like(clean)
    body[Y0:Y1, X0:X1] = True
    corners = [(Y0, X0), (Y0, X1 - 1), (Y1 - 1, X0), (Y1 - 1, X1 - 1)]
    assert clean[Y0 + 1:Y1 - 1, X0:X1].all() and clean[Y0:Y1, X0 + 1:X1 - 1].all()
    assert not clean[~body].any()
    dropped = [c for c in corners if not clean[c]]
    assert int(clean.sum()) == int(body.sum()) - len(dropped)
    assert len(dropped) == 4, "IPL's curvature smoothing rounds every convex 90-degree corner"


@pytest.mark.parametrize("bump", [(20, X1), (Y0 - 1, 20), (20, X0 - 1), (Y1, 20)],
                         ids=["east", "north", "west", "south"])
def test_one_pixel_bump_is_dropped(clean, bump):
    """East/west bumps are 1-pixel excursions on vertical travel (moved in stage A); the south bump is a
    horizontal run of length 1 dropped by phase 1 (and, lying on the boundary trace, it activates stage B);
    the north bump is the component's raster-first pixel, kept by phase 1 as the trace start and moved down
    by stage B, which the corner deletions of stage A activate."""
    sl = block()
    sl[bump] = True
    R = render_slice(sl)
    assert not R[bump]
    assert np.array_equal(R, clean)


@pytest.mark.parametrize("notch", [(20, X1 - 1), (Y0, 20), (20, X0), (Y1 - 1, 20)],
                         ids=["east", "north", "west", "south"])
def test_one_pixel_notch_is_filled(clean, notch):
    """East/west notches (vertical travel) are bridged by stage A.  The north/south notches are excursions
    on HORIZONTAL travel, which stage A does not touch: they are bridged only because the block's corner
    deletions activate stage B -- on an otherwise event-free contour they would stay (see the disc tests)."""
    sl = block()
    sl[notch] = False
    R = render_slice(sl)
    assert R[notch]
    assert np.array_equal(R, clean)


def test_count_stable_under_bump_and_notch_together(clean):
    sl = block()
    sl[20, X1] = True           # bump on the east side
    sl[Y0, 20] = False          # notch on the north side
    sl[25, X0 - 1] = True       # bump on the west side
    R = render_slice(sl)
    assert np.array_equal(R, clean)


def test_two_pixel_features_are_not_single_pixel_features(clean):
    """A 2-pixel-wide bump is not a horizontal run of length 1 and survives phase 1;
    a 2-pixel notch is not bridged by the midpoint rule."""
    sl = block()
    sl[Y0 - 1, 20:22] = True
    R = render_slice(sl)
    assert R[Y0 - 1, 20:22].any()
    sl = block()
    sl[20, X1 - 2:X1] = False
    R = render_slice(sl)
    assert not R[20, X1 - 2:X1].all()


# ------------------------------------------------------------------------------------ phase 1
def test_phase1_drops_horizontal_runs_of_one_except_vertical_necks():
    """phase1 drops a pixel whose W and E neighbours are background unless mask lies directly above AND
    below it (the data-derived vertical-neck exception: PFJ-8bcf88_R z167, PFJ-351dc7_R z413)."""
    sl = np.zeros((6, 9), bool)
    sl[3, 1:4] = True           # run of 3: kept
    sl[3, 6] = True             # run of 1, nothing above or below: dropped
    sl[0, 3] = True             # top of a 1-pixel-wide column: mask below only -> dropped
    sl[1, 3] = True             # mask above (0,3) AND below (2,3): vertical neck -> kept
    sl[2, 3] = True             # mask above (1,3) AND below (3,3): vertical neck -> kept
    p = phase1(sl)
    assert p[3, 1:4].all()
    assert not p[3, 6]
    assert not p[0, 3]
    assert p[1, 3] and p[2, 3]


def test_phase1_start_exception_and_activation_flags():
    """outer_components keeps the raster-first pixel of every component (the trace start) even when it is
    a run of length 1, and reports pre_activated only for a dropped pixel lying on the component's own
    boundary trace: a dropped bump on the outer boundary activates, a dropped bump facing a hole does not."""
    sl = block()
    sl[Y0 - 1, X0] = True                              # raster-first pixel, a run of 1
    (comp, pre), = outer_components(sl)
    assert comp[Y0 - 1, X0] and not phase1(sl)[Y0 - 1, X0]
    assert pre is False
    sl = block()
    sl[Y1, 20] = True                                  # south bump on the outer boundary: dropped, activates
    (comp, pre), = outer_components(sl)
    assert not comp[Y1, 20] and pre is True
    sl = block()
    sl[16:24, 16:24] = False
    sl[16, 20] = True                                  # a bump hanging into the hole: dropped, no activation
    (comp, pre), = outer_components(sl)
    assert not comp[16, 20] and pre is False


def test_vertical_neck_keeps_a_lobe_attached():
    """A lobe joined to the body through a 1-pixel-wide vertical neck: the neck pixel is a run of length 1
    with mask above and below, so phase 1 keeps it and the lobe is traced as part of one contour and
    rendered.  The same pixel with the lobe removed is a plain bump and is dropped."""
    sl = np.zeros((30, 30), bool)
    sl[5:12, 8:22] = True
    sl[12, 15] = True                                  # the neck
    sl[13:18, 12:19] = True                            # the lobe
    assert phase1(sl)[12, 15]
    R = render_slice(sl)
    assert R[12, 15] and R[14:17, 13:18].all()
    assert len(slice_chains(sl)) == 1
    sl[13:18, 12:19] = False
    assert not phase1(sl)[12, 15]
    assert not render_slice(sl)[12, 15]


# ------------------------------------------------------------------------------------ the raw-store gate
def test_event_free_disc_is_stored_raw_and_rendering_invariant():
    sl = disc()
    raw = moore_trace(sl, raster_first(sl), backtrack=(-1, 0), ccw=True)
    assert sweep(raw, STAGE_A_RULES)[1] is False and sweep(raw)[1] is False
    chains = slice_chains(sl)
    assert len(chains) == 1 and chains[0]["kind"] == "outer" and chains[0]["vertices"] == raw
    assert np.array_equal(render_slice(sl), sl)


def test_notch_in_a_horizontal_run_of_an_event_free_contour_stays_open():
    """The raw-store gate: a chain whose only defect is a 1-pixel excursion on horizontal travel is stored
    RAW by IPL (PFJ-d81140_R z0, PFJ-0fb201_R z168, PFJ-6f5538_R z168, PFJ-6b714e_R z0, 135 cortical inner contours) --
    the notch in the disc's top row is not bridged and the rendering equals the notched mask."""
    sl = disc()
    ys, xs = np.nonzero(sl)
    top = int(ys.min())
    xm = int((xs[ys == top].min() + xs[ys == top].max()) // 2)
    sl[top, xm] = False
    raw = moore_trace(sl, raster_first(sl), backtrack=(-1, 0), ccw=True)
    assert sweep(raw)[1] is True, "the full rule would move the notch vertex"
    assert sweep(raw, STAGE_A_RULES)[1] is False, "stage A sees nothing"
    assert slice_chains(sl)[0]["vertices"] == raw
    R = render_slice(sl)
    assert not R[top, xm]
    assert np.array_equal(R, sl)


def test_notch_in_a_vertical_run_of_the_same_contour_is_bridged():
    """The mirror case: a 1-pixel excursion on vertical travel is a stage-A event, so the chain is smoothed
    and the notch bridged."""
    sl = disc()
    ys, xs = np.nonzero(sl)
    row = sl.shape[0] // 2
    east = int(xs[ys == row].max())
    sl[row, east] = False
    R = render_slice(sl)
    assert R[row, east]
    assert np.array_equal(R, disc())


def test_pre_activation_alone_runs_stage_b():
    """The other half of the raw-store gate: a chain whose stage-A sweep changes nothing is still smoothed
    when phase 1 dropped a pixel of the component's own boundary trace (pre_activated).  Disc with a south
    bump (a run of 1 on the raw boundary, dropped -> pre True) and a notch in the top row (horizontal-travel
    excursion, invisible to stage A): stage B bridges the notch.  Cohort: 22 pre-only trabecular chains
    (PFJ-6b714e_L 6, PFJ-6f5538_R 6, PFJ-351dc7_R/PFJ-8bcf88_L/PFJ-0fb201_R/PFJ-a56aae_R 2, PFJ-ab6af9_R/PFJ-69bcb0_L 1), 20 of them changed by stage B."""
    sl = disc()
    ys, xs = np.nonzero(sl)
    top, bot = int(ys.min()), int(ys.max())
    xm = int((xs[ys == top].min() + xs[ys == top].max()) // 2)
    sl[bot + 1, xm] = True                             # south bump: dropped by phase 1, on the raw trace
    sl[top, xm] = False                                # notch on horizontal travel
    (comp, pre), = outer_components(sl)
    assert pre is True and not comp[bot + 1, xm]
    raw = moore_trace(comp, raster_first(comp), backtrack=(-1, 0), ccw=True)
    assert sweep(raw, STAGE_A_RULES)[1] is False, "stage A sees nothing"
    assert smooth_chain(raw, False) == raw
    assert smooth_chain(raw, True) != raw
    assert slice_chains(sl)[0]["vertices"] != raw
    R = render_slice(sl)
    assert R[top, xm] and not R[bot + 1, xm]
    assert np.array_equal(R, disc())


LOBE_AND_NOTCH = """\
.####..
#####..
######.
###.##."""      # a 3-pixel lobe at the lower right next to a 1-pixel notch in the bottom (horizontal) run


def test_stage_a_is_exactly_one_sweep():
    """After the first stage-A sweep the chain still holds a 135-degree turn at the lobe (a stage-A event)
    AND a 1-pixel notch in the bottom horizontal run (a stage-B-only event).  IPL's rule runs stage B right
    away: the notch is bridged first and the lobe survives (16 rendered pixels).  A stage A iterated to
    convergence would cut the lobe before the notch is bridged (12 pixels, the 8-vertex chain) -- the
    variant refuted by PFJ-351dc7_R cort z478 and PFJ-0fb201_R cort z335 (5 voxels, ipldt.contour.smooth)."""
    sl = np.pad(np.array([[c == "#" for c in row] for row in LOBE_AND_NOTCH.split("\n")]), 2)
    (comp, pre), = outer_components(sl)
    assert pre is False
    raw = moore_trace(comp, raster_first(comp), backtrack=(-1, 0), ccw=True)
    p1, changed1 = sweep(raw, STAGE_A_RULES)
    p2, changed2 = sweep(p1, STAGE_A_RULES)
    assert changed1 and changed2, "a second stage-A sweep would still change this chain"
    assert len(p2) == 8
    stats = {}
    pts = smooth_chain(raw, pre, stats=stats)
    assert stats == dict(stage_a_changed=True, activated=True, sweeps=2)
    assert pts == [(3, 2), (2, 3), (2, 4), (3, 5), (4, 5), (5, 5), (6, 4), (6, 3), (5, 2), (4, 2)]
    assert [c["vertices"] for c in slice_chains(sl)] == [pts]
    R = render_slice(sl)
    assert R[2, 3:6].all() and R[3:5, 2:7].all() and R[5, 3:6].all() and int(R.sum()) == 16


# ------------------------------------------------------------------------------------ sweep order, in-place moves, inner rules
def test_start_vertex_is_swept_last_and_moves_are_seen_in_place():
    """A 1-pixel tooth on the block's top row, kept by phase 1 as the raster-first trace start, with a
    1-pixel notch between it and a run (PFJ-0be66a_R cortical z=3).  The start s=(20,Y0-1) and its predecessor
    q=(21,Y0) are both horizontal excursions (|d|=(2,0)).  Swept forward with the start LAST and in place, q
    is moved up into the notch first, so s then sees d=(-2,1) and stays: the stored chain starts at the raw
    tooth and ends at the bridged notch, and both pixels render.  Visiting the start first moves s down and
    leaves q (chain starts (20,Y0), tooth not rendered); Jacobi moves (neighbours read from the pre-sweep
    chain) move both."""
    sl = block()
    sl[Y0 - 1, 20] = True                    # the tooth: raster-first pixel, a run of 1 kept as the trace start
    sl[Y0 - 1, 22:X1] = True                 # the run, one notch (21, Y0-1) away from the tooth
    (chain,) = slice_chains(sl)
    v = chain["vertices"]
    assert v[0] == (20, Y0 - 1), "the start vertex is visited last and stays at the raw tooth"
    assert v[-1] == (21, Y0 - 1), "its predecessor was moved into the notch before the start was visited"
    assert (20, Y0) not in v
    R = render_slice(sl)
    assert R[Y0 - 1, 20] and R[Y0 - 1, 21] and R[Y0 - 1, 22:X1 - 1].all()   # (X1-1, Y0-1) is a rounded corner
    # the same rule on the bare chain: one sweep with the full rule, start last
    raw = moore_trace(sl, (20, Y0 - 1), backtrack=(-1, 0), ccw=True)
    assert raw[0] == (20, Y0 - 1) and raw[-1] == (21, Y0) and raw[1] == (19, Y0)
    once, changed = sweep(raw)
    assert changed and once[0] == (20, Y0 - 1) and once[-1] == (21, Y0 - 1)


def test_inner_start_tooth_between_two_notches_is_swept_last():
    """The inner-chain twin (PFJ-0be66a_R cortical z=26 / z=29): the hole's first-row run is a 2-pixel notch in
    the top wall, the inner start is the 1-pixel tooth after it, followed by a 1-pixel notch.  The start and
    its successor are both horizontal excursions; with the start swept last the successor is moved down onto
    the wall row and the start stays (a west-wall bump activates stage B).  Visiting the start first moves
    the start up instead."""
    sl = block()
    sl[16:24, 16:24] = False
    sl[15, 17:19] = False                    # 2-pixel notch: the hole's first-row run
    sl[15, 20] = False                       # 1-pixel notch after the tooth (19, 15)
    sl[20, 16] = True                        # west-wall bump into the hole: a stage-A event (activator)
    outer, inner = slice_chains(sl)
    assert inner["kind"] == "inner"
    v = inner["vertices"]
    assert v[0] == (19, 15) and v[1] == (20, 15) and v[2] == (21, 15)
    assert (20, 14) not in v and (19, 14) not in v
    R = render_slice(sl)
    assert R[15, 19] and R[15, 20] and not R[15, 17:19].any()


def test_inner_chain_has_no_phase_1_and_no_pre_activation():
    """A 1-pixel endosteal bump on the BOTTOM wall of a square hole (not on the first row, where it would be
    the protected inner start).  Phase 1 does not apply to inner chains, so the bump is traced; it is a
    horizontal excursion only, stage A sees nothing on the corner-cut inner trace, and inner chains carry
    no pre-activation, so the chain is stored RAW with the bump and the bump renders.  Phase 1 on the inner
    trace would drop it; pre-activating inner chains would run stage B and move it onto the wall."""
    sl = block()
    sl[16:24, 16:24] = False
    sl[23, 20] = True                        # bump hanging up from the bottom wall
    lab, ids = holes(sl)
    (h,) = ids
    raw = moore_trace(~(lab == h), (24, 16), backtrack=(-1, 0), ccw=True)
    assert (20, 23) in raw
    assert sweep(raw, STAGE_A_RULES)[1] is False and sweep(raw)[1] is True
    outer, inner = slice_chains(sl)
    assert inner["vertices"] == raw, "stored raw: no phase 1 and no pre-activation on inner chains"
    assert smooth_chain(raw, pre_activated=True) != raw
    R = render_slice(sl)
    assert R[23, 20] and not R[16:23, 16:24].any() and not R[23, 16:20].any() and not R[23, 21:24].any()


# ------------------------------------------------------------------------------------ holes and the inner fill
def test_hole_stays_open():
    """A square ring (cortical-shell-like): the rendered raster keeps the hole open and keeps
    the ring, up to the rounding of its four outer corners.  The inner trace of a square hole cuts the
    hole's corners diagonally, so the inner chain is event-free and stored raw."""
    sl = block()
    sl[16:24, 16:24] = False
    R = render_slice(sl)
    assert not R[16:24, 16:24].any()
    assert R[Y0 + 1:16, X0 + 1:X1 - 1].all() and R[24:Y1 - 1, X0 + 1:X1 - 1].all()
    assert R[16:24, X0 + 1:16].all() and R[16:24, 24:X1 - 1].all()
    assert int(R.sum()) == int(sl.sum()) - 4


LOBE = """\
...#...
..###..
.####..
#####..
.#####.
.#####.
.#####.
..#####
...####
...####
...###.
..###..
.###..."""      # a cortical lobe hanging into the medullary cavity, modelled on PFJ-a56aae_R z258


def ring_with_lobe():
    sl = np.zeros((40, 40), bool)
    sl[4:36, 4:36] = True
    sl[10:30, 10:30] = False
    for i, row in enumerate(LOBE.split("\n")):
        for j, ch in enumerate(row):
            if ch == "#":
                sl[17 + i, 15 + j] = True
    return sl


def test_inner_fill_is_the_even_odd_polygon_interior():
    """A lobe the inner chain wraps around through a narrow neck: its core pixels are enclosed by chain
    pixels (binary_fill_holes of the chain raster would remove them) but lie OUTSIDE the inner polygon,
    and IPL keeps them (PFJ-ab6af9_R z433, PFJ-351dc7_R z413, PFJ-8bcf88_R z167, PFJ-0fb201_R z335, PFJ-a56aae_R z258,
    PFJ-69bcb0_L z167: 142 voxels)."""
    sl = ring_with_lobe()
    lab, ids = holes(sl)
    assert len(ids) == 1
    hole = lab == ids[0]
    chains = slice_chains(sl)
    assert [c["kind"] for c in chains] == ["outer", "inner"]
    chain, strict = polygon_fill(chains[1]["vertices"], sl.shape)
    enclosed = ndi.binary_fill_holes(chain, structure=S4) & ~chain
    lobe_core = enclosed & ~strict
    assert int(lobe_core.sum()) >= 15
    assert sl[lobe_core].all()
    R = render_slice(sl)
    assert R[lobe_core].all()
    assert not (R & strict).any()
    assert int((R & hole).sum()) <= 1                  # one bridged notch vertex at the lobe's base
    assert R[5:9, 5:35].all() and R[31:35, 5:35].all()  # the ring's walls (outer corners excepted)
    R_fill_holes = R & ~enclosed
    assert int((R & ~R_fill_holes).sum()) == int(lobe_core.sum())


# ------------------------------------------------------------------------------------ conventions (not IPL-verified)
def test_outer_components_are_8_connected():
    """CONVENTION: object components are 8-connected (no cohort slice has a 4-vs-8 difference in the object
    partition, so IPL's choice is unobserved).  Two blocks joined only through the corner pixel (6,6) --
    itself a run of 1 with mask below only, and the raster-first pixel of the second 4-component -- give two
    outer chains here; 4-connected labelling would give one chain of 35 vertices and 61 pixels."""
    sl = np.zeros((12, 12), bool)
    sl[:6, :6] = True
    sl[7:, 6:] = True
    sl[6, 6] = True
    chains = slice_chains(sl)
    assert [(c["kind"], len(c["vertices"])) for c in chains] == [("outer", 16), ("outer", 14)]
    assert int(render_slice(sl).sum()) == 58


def test_holes_are_4_connected():
    """CONVENTION: background holes are 4-connected (no cohort slice has a 4-vs-8 difference in the
    background partition).  Two 2x2 holes touching at a corner are two holes with two inner chains; an
    8-connected background would merge them into one hole (one inner chain, 4 pixels differ)."""
    sl = np.ones((14, 14), bool)
    sl[3:5, 3:5] = False
    sl[5:7, 5:7] = False
    chains = slice_chains(sl)
    assert [(c["kind"], len(c["vertices"])) for c in chains] == [("outer", 48), ("inner", 8), ("inner", 8)]
    assert int(render_slice(sl).sum()) == 184


def test_diagonal_wall_gap_keeps_the_hole():
    """CONVENTION (the same pairing): a hole whose wall is breached only diagonally stays a hole under the
    4-connected background rule (an 8-connected background would join it to the exterior: no inner chain,
    17 pixels differ)."""
    sl = np.zeros((14, 14), bool)
    sl[2:12, 2:12] = True
    sl[5:9, 5:9] = False
    sl[2, 6] = sl[3, 6] = sl[4, 7] = False
    chains = slice_chains(sl)
    assert [(c["kind"], len(c["vertices"])) for c in chains] == [("outer", 32), ("inner", 16)]
    assert int(render_slice(sl).sum()) == 76


def test_diagonal_neck_splits_the_component():
    """CONVENTION (untested against IPL: the cohort's only vertical necks are the 2 pixels of the exception,
    and none of its 42,508 dropped run-of-1 pixels splits a component): a run-of-1 pixel attached only
    through NW/SE is dropped by phase 1, the raw component splits, and outer_components emits one outer
    chain per piece, both with the same pre_activated flag (the drop is keyed on the raw component).
    The same pixel with mask N and S is the vertical-neck exception and is kept."""
    sl = np.zeros((11, 11), bool)
    sl[1:5, 1:5] = True
    sl[6:10, 6:10] = True
    sl[5, 5] = True                                   # neighbours in the mask: NW (4,4) and SE (6,6) only
    assert ndi.label(sl, structure=np.ones((3, 3), bool))[1] == 1
    assert not phase1(sl)[5, 5]
    comps = outer_components(sl)
    assert [pre for _, pre in comps] == [True, True]
    chains = slice_chains(sl)
    assert [(c["kind"], len(c["vertices"]), c["vertices"][0]) for c in chains] == [("outer", 8, (2, 1)), ("outer", 8, (7, 6))]
    R = render_slice(sl)
    assert not R[5, 5] and int(R.sum()) == 24


def test_component_nested_in_a_hole_is_kept():
    """CONVENTION (outside the verified domain: every cohort slice with a hole has exactly one component):
    an island inside another component's hole is rendered like the global even-odd rule would -- an inner
    contour's strict interior is removed only from the outer fill whose polygon INTERIOR meets the hole,
    never from the island, whose only contact with the hole is a smoothed chain vertex moved onto a hole
    pixel (the 1-pixel notch).  A sequential fill-outers-then-erase-inners renderer would erase the
    island; matching on the whole fill (chain pixels included) erased it with the notch and kept it without."""
    for notch in (False, True):
        sl = np.zeros((60, 60), bool)
        sl[4:56, 4:56] = True
        sl[10:50, 10:50] = False
        sl[20:40, 20:40] = True
        if notch:
            sl[30, 39] = False
        assert [c["kind"] for c in slice_chains(sl)] == ["outer", "outer", "inner"]
        R = render_slice(sl)
        assert int(R[20:40, 20:40].sum()) == 396                    # the island, its four corners rounded
        assert R[30, 39] and int(R.sum()) == 1496
        assert not R[10:20, 10:50].any() and not R[40:50, 10:50].any()


def test_multi_component_slice_renders_each_component_on_its_own():
    """Several rings (3x3 holes, stored inner chains) and solid blocks on one slice, plus a bump and a notch:
    the rendering equals the OR of the components rendered one at a time (the hole matching pairs each
    inner contour with its own ring only), pinning the multi-component / multi-hole path of render_slice
    (20 chains, 668 pixels)."""
    sl = np.zeros((48, 48), bool)
    for y, x in [(2, 2), (2, 14), (2, 26), (14, 2), (14, 26), (26, 14), (26, 26)]:
        sl[y:y + 9, x:x + 9] = True
        sl[y + 3:y + 6, x + 3:x + 6] = False
    for y, x in [(14, 14), (38, 2), (38, 20), (38, 38), (2, 38), (20, 38)]:
        sl[y:y + 6, x:x + 6] = True
    sl[38, 12] = True
    sl[10, 40] = False
    lab, n = ndi.label(sl, structure=np.ones((3, 3), bool))
    assert n == 14 and len(holes(sl)[1]) == 7
    chains = slice_chains(sl)
    assert len(chains) == 20 and [c["kind"] for c in chains] == ["outer"] * 13 + ["inner"] * 7
    R = render_slice(sl)
    per = np.zeros_like(sl)
    for k in range(1, n + 1):
        per |= render_slice(lab == k)
    assert np.array_equal(R, per)
    assert int(R.sum()) == 668
    for y, x in [(2, 2), (2, 14), (2, 26), (14, 2), (14, 26), (26, 14), (26, 26)]:
        assert not R[y + 3:y + 6, x + 3:x + 6].any()


# ------------------------------------------------------------------------------------ stored-chain conventions
def test_slice_chain_conventions_on_a_ring():
    """Outer chain first, then the inner chain.  The outer trace starts at the raster-first pixel (X0, Y0);
    the smoothing deletes that corner, and the stored chain then starts at the surviving vertex BEFORE it,
    (X0 + 1, Y0) -- the convention read off the 4 cohort chains whose start is deleted (PFJ-ab6af9_R trab z469,
    PFJ-351dc7_R trab z376/z381/z382).  The inner start is the object pixel right after the hole's first-row run."""
    sl = block()
    sl[16:24, 16:24] = False
    chains = slice_chains(sl)
    assert [c["kind"] for c in chains] == ["outer", "inner"]
    outer, inner = chains[0]["vertices"], chains[1]["vertices"]
    assert (X0, Y0) not in outer and outer[0] == (X0 + 1, Y0) and outer[1] == (X0, Y0 + 1)
    assert inner[0] == (24, 16)
    for v in outer + inner:
        assert sl[v[1], v[0]]
    for pts in (outer, inner):
        for a, b in zip(pts, pts[1:] + pts[:1]):
            assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
    assert slice_chains(np.zeros((8, 8), bool)) == []


def test_freeman_codes_round_trip():
    """chain_codes() encodes a chain as /togobj_from_aim stores it: gobj_path(codes[:-1], start, 8) walks
    it back, the codes agree with moore.codes_of, and packing them LSB-first 3 bits each reproduces the
    file layout that gobj_file.decode reads."""
    sl = ring_with_lobe()
    for c in slice_chains(sl):
        v = c["vertices"]
        codes = chain_codes(v)
        assert len(codes) == len(v)
        x, y = gobj_path(codes[:-1], v[0], 8)
        assert list(zip(x.tolist(), y.tolist())) == v
        assert codes.tolist() == codes_of(v)
        n_el = len(v)
        bits = np.zeros(3 * (n_el - 1), np.uint8)
        for i, s in enumerate(codes[:-1]):
            bits[3 * i:3 * i + 3] = [(s >> k) & 1 for k in range(3)]
        packed = np.packbits(bits, bitorder="little")
        assert decode(packed, n_el).tolist() == codes[:-1].tolist()
    with pytest.raises(ValueError):
        chain_codes([(0, 0), (2, 0), (0, 0)])


def test_collapsed_inner_chain_is_not_stored():
    """Chains of fewer than 4 vertices are left alone by the sweep (guard).  The inner chain of a 5-pixel
    diamond hole (PFJ-6f5538_R z265) collapses under the smoothing to 2 vertices; a chain of fewer than
    MIN_VERTICES = 4 vertices is not stored, so the hole is filled.  A 3x3 square hole keeps its (raw,
    event-free) inner chain and stays open."""
    tri = [(0, 0), (1, 1), (0, 1)]
    assert smooth_chain(tri) == tri
    sl = np.zeros((14, 14), bool)
    sl[2:12, 2:12] = True
    for y, x in [(6, 7), (7, 6), (7, 7), (7, 8), (8, 7)]:
        sl[y, x] = False
    lab, ids = holes(sl)
    assert len(ids) == 1
    raw = moore_trace(~(lab == ids[0]), (8, 6), backtrack=(-1, 0), ccw=True)   # start: right after the first-row run
    assert len(raw) == 8 and len(smooth_chain(raw)) == 2 < MIN_VERTICES
    assert [c["kind"] for c in slice_chains(sl)] == ["outer"]
    assert render_slice(sl)[6:9, 6:9].all()
    sl = np.zeros((14, 14), bool)
    sl[2:12, 2:12] = True
    sl[6:9, 6:9] = False
    assert [c["kind"] for c in slice_chains(sl)] == ["outer", "inner"]
    assert not render_slice(sl)[6:9, 6:9].any()


# ------------------------------------------------------------------------------------ volume / edge cases
def test_moore_trace_of_a_block_is_its_boundary():
    sl = block()
    pts = moore_trace(sl, (X0, Y0), backtrack=(-1, 0), ccw=True)
    assert len(pts) == 2 * (Y1 - Y0) + 2 * (X1 - X0) - 4
    assert len(set(pts)) == len(pts)
    for x, y in pts:
        assert sl[y, x]
        assert not sl[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].all()      # every vertex touches the outside


def test_render_volume_is_per_slice_and_leaves_empty_slices_empty():
    vol = np.zeros((5, 40, 40), bool)
    vol[1] = block()
    vol[2] = block()
    vol[2, 20, X1] = True       # bump on slice 2 only
    vol[3] = block()
    vol[3, Y0, 20] = False      # notch on slice 3 only
    G = render_volume(vol)
    assert not G[0].any() and not G[4].any()
    assert np.array_equal(G[1], render_slice(vol[1]))
    assert np.array_equal(G[2], G[1]) and np.array_equal(G[3], G[1])
    assert G.dtype == bool and G.shape == vol.shape


def test_empty_and_tiny_slices():
    """A 1- or 2-pixel component gives a chain of fewer than MIN_VERTICES = 4 vertices, which is not stored,
    so nothing is rendered; a 3-pixel line's trace doubles back and collapses under the spur rule.  (The
    cohort has no component of 1-3 pixels; X3931708_CORT_MASK_version2 has 36 single pixels, 26 2-px and 9
    3-px components, none with an IPL contour.)  The 5-pixel diamond component IS data-verified: PFJ-6f5538_R
    trab z265 (phase 1 drops its bottom pixel, the 4-vertex trace collapses to nothing), IPL stores no
    contour there and renders 0 voxels."""
    assert not render_slice(np.zeros((8, 8), bool)).any()
    sl = np.zeros((8, 8), bool)
    sl[4, 4] = True
    assert int(render_slice(sl).sum()) == 0 and slice_chains(sl) == []
    sl[4, 5] = True
    assert int(render_slice(sl).sum()) == 0
    sl[4, 6] = True
    assert int(render_slice(sl).sum()) == 0
    sl = np.zeros((8, 8), bool)
    for y, x in [(3, 4), (4, 3), (4, 4), (4, 5), (5, 4)]:
        sl[y, x] = True                                # PFJ-6f5538_R trab z265: (1129,63),(1128..1130,64),(1129,65)
    assert slice_chains(sl) == [] and int(render_slice(sl).sum()) == 0


def test_frozen_three_vertex_chain_is_not_stored():
    """PFJ-6f5538_R T17 29_trabfinal z332 (2026-09-14): an 8-pixel hexagon (rows ..##.. / .####. / ..##..) on the mask's
    last slice.  Phase 1 drops nothing, the CCW Moore trace has 6 vertices, stage A leaves 3 -- two vertical
    excursions moved, two corners and one 135-degree turn deleted, the start among them -- and the sweep's
    guard (n < 4) freezes the chain there.  IPL stored NO contour for that slice (its /gobj_to_aim grid ends one
    slice short of the mask), so the frozen triangle is not a stored chain: MIN_VERTICES = 4, nothing rendered.
    MIN_VERTICES = 3 rendered the 3-pixel triangle (the stage-31 residual of the test-run-17 verification, 3 -> 0)."""
    assert MIN_VERTICES == 4
    sl = np.zeros((8, 10), bool)
    for y, x in [(2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (3, 5), (4, 3), (4, 4)]:      # local (x - 1123, y - 34)
        sl[y, x] = True
    assert np.array_equal(phase1(sl), sl)
    raw = moore_trace(sl, raster_first(sl), backtrack=(-1, 0), ccw=True)
    assert raw == [(3, 2), (2, 3), (3, 4), (4, 4), (5, 3), (4, 2)]
    a, changed = sweep(raw, STAGE_A_RULES)
    assert changed and a == [(4, 3), (3, 3), (4, 4)]
    assert sweep(a, ALL_RULES) == (a, False)                    # the guard: a 3-chain is left alone ...
    assert smooth_chain(raw) == a                               # ... so the smoothing freezes at 3 vertices
    assert slice_chains(sl) == [] and int(render_slice(sl).sum()) == 0
    vol = np.zeros((3, 8, 10), bool)
    vol[1] = sl
    assert int(render_volume(vol).sum()) == 0


def test_sweep_has_no_fixed_point_below_8_vertices():
    """The rule behind MIN_VERTICES = 4: over every closed chain of n distinct 8-adjacent vertices, the sweep (all
    five rules) has no fixed point for n = 3..7 (24 / 96 / 360 / 1,512 / 6,664 chains), so a chain still at 3 vertices
    is collapsing, and the smallest fixed points are 8-vertex rings (16 of 31,056).  Hence every stored contour has
    >= 8 elements (IPL's shortest stored contours have exactly 8) and any threshold 4..8 stores the same chains; a
    guard-free sweep would delete all 24 closed 3-chains in one sweep (every vertex of such a chain is a corner, a
    135-degree turn or a spur)."""
    import itertools
    N8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

    def chains(n):
        for steps in itertools.product(N8, repeat=n - 1):
            pts = [(0, 0)]
            for dx, dy in steps:
                pts.append((pts[-1][0] + dx, pts[-1][1] + dy))
            if len(set(pts)) == n and (pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) in N8:
                yield pts

    counts = {}
    for n in (3, 4, 5, 6, 7, 8):
        total = fixed = 0
        for pts in chains(n):
            total += 1
            out, changed = sweep(pts, ALL_RULES)
            if n == 3:
                assert (out, changed) == (pts, False)           # the guard returns a 3-chain unchanged ...
                for i in range(3):                              # ... though every vertex is a deletion trigger
                    p, q = pts[(i - 1) % 3], pts[(i + 1) % 3]
                    assert (abs(q[0] - p[0]), abs(q[1] - p[1])) in ((1, 1), (1, 0), (0, 1), (0, 0))
            elif not changed and len(out) == n:
                fixed += 1
        counts[n] = (total, fixed)
    assert counts == {3: (24, 0), 4: (96, 0), 5: (360, 0), 6: (1512, 0), 7: (6664, 0), 8: (31056, 16)}


T17_ROOT = os.environ.get("IPLDT_RUN17_ROOT", lab_path("ipl_test_runs/run17/aims_and_logs"))


def _run17(base, tag):
    from conftest import vms_versions
    fs = vms_versions(T17_ROOT, f"{base}_T17_{tag}.AIM")
    return fs[-1] if fs else None


@pytest.mark.slow
def test_pfj_6f5538_run17_trabecular_rendering_matches_ipl_export():
    """The tiny-component oracle on the real volume: render_volume(IPL's X5492058_T17_29_TRABFINAL) vs IPL's
    /gobj_to_aim of the new trabecular gobj (X5492058_T17_31_TRABGOBJ), by global position on the union grid:
    0 mismatching voxels, |IPL| = |ours| = 1,893,500.  The mask has 165 non-empty slices (global z 168..332), IPL's
    rendering 164 (its grid ends at z 331): the 8-px hexagon of z 332 gets no contour.  MIN_VERTICES = 3 gave 3
    ours-only voxels at global (1127, 37, 332), (1128, 37, 332), (1128, 38, 332)."""
    from ipldt.io import read_aim, align_to
    f29, f31 = _run17("X5492058", "29_trabfinal"), _run17("X5492058", "31_trabgobj")
    if not (f29 and f31):
        pytest.skip("test-run-17 exports of PFJ-6f5538_R not available")
    m = read_aim(f29)
    M, dim, pos = m["data"] > 0, m["dim"], m["pos"]
    z_last = pos[2] + int(np.nonzero(M.any(axis=(1, 2)))[0].max())
    assert z_last == 332 and int(M[z_last - pos[2]].sum()) == 8
    ipl = read_aim(f31)
    assert tuple(ipl["dim"]) == (559, 115, 164) and tuple(ipl["pos"]) == (907, 26, 168)
    G = render_volume(M)
    lo = [min(pos[i], ipl["pos"][i]) for i in range(3)]
    hi = [max(pos[i] + dim[i], ipl["pos"][i] + ipl["dim"][i]) for i in range(3)]
    udim, upos = tuple(h - l for h, l in zip(hi, lo)), tuple(lo)
    Gu = align_to(dict(data=G.astype(np.uint8), dim=dim, pos=pos), udim, upos) > 0
    Iu = align_to(ipl, udim, upos) > 0
    assert int(Iu.sum()) == 1_893_500 and int(Gu.sum()) == 1_893_500
    assert int((Gu != Iu).sum()) == 0


def test_long_filament_collapses_within_the_scaled_sweep_cap():
    """A 1-pixel-wide axis-aligned ribbon of L pixels traces to 2L-2 vertices and loses only 2 vertices per
    free tip and sweep (spur rule), so it needs ~L/2 sweeps per tip: the stage-B cap scales with the raw
    chain length (max(MAX_SWEEPS, len(raw))) so that a legal chain cannot trip it.  An isolated 2300-px line
    (1149 sweeps) collapses to < 3 vertices and is not stored; a 1100-px hair on a 20x20 blob (one free
    tip, 1100 sweeps > MAX_SWEEPS) collapses onto the blob's 72-vertex contour.  (The rule's extrapolation:
    no IPL export holds such a ribbon; the cohort needs at most 14 sweeps.)"""
    sl = np.zeros((3, 2302), bool)
    sl[1, 1:2301] = True
    raw = moore_trace(sl, raster_first(sl), backtrack=(-1, 0), ccw=True)
    stats = {}
    assert len(raw) == 4598 and len(smooth_chain(raw, stats=stats)) < 3 and stats["sweeps"] == 1149
    assert slice_chains(sl) == [] and not render_slice(sl).any()
    sl = np.zeros((40, 1140), bool)
    sl[10:30, 5:25] = True
    sl[20, 25:1125] = True
    (comp, pre), = outer_components(sl)
    raw = moore_trace(comp, raster_first(comp), backtrack=(-1, 0), ccw=True)
    stats = {}
    pts = smooth_chain(raw, pre, stats=stats)
    assert stats["sweeps"] == 1100 > MAX_SWEEPS and len(pts) == 72
    chains = slice_chains(sl)
    assert len(chains) == 1 and chains[0]["kind"] == "outer" and chains[0]["vertices"] == pts


def test_wrong_rank_and_wrong_chain_inputs_are_refused():
    """The entry points name the offending shape instead of failing inside scipy / numpy."""
    with pytest.raises(ValueError, match=r"\(5, 5\)"):
        render_volume(np.ones((5, 5), bool))
    with pytest.raises(ValueError, match=r"\(2, 5, 5\)"):
        render_slice(np.ones((2, 5, 5), bool))
    with pytest.raises(ValueError, match=r"\(2, 5, 5\)"):
        slice_chains(np.ones((2, 5, 5), bool))
    with pytest.raises(ValueError, match=r"\(5,\)"):
        render_volume(np.ones(5, bool))
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        chain_codes(np.zeros((3, 3), int))
    with pytest.raises(ValueError, match=r"\(0,\)"):
        chain_codes([])


# ------------------------------------------------------------------------------------ real data (slow)
def _find(pattern):
    if not os.path.isdir(RUN15):
        return []
    rx = re.compile("^" + re.escape(pattern) + "(;\\d+)?$", re.I)
    fs = sorted((f for f in os.listdir(RUN15) if rx.match(f)), key=lambda f: int(f.split(";")[1]) if ";" in f else 0)
    return [os.path.join(RUN15, f) for f in fs]


@pytest.mark.slow
def test_pfj_0be66a_cortical_rendering_matches_ipl_export(data_root):
    """render_volume(X2420448_CORT_MASK) == IPL's /gobj_to_aim export of the cortical gobj (test run 15),
    compared by global position on the union of the two grids, and every chain of the stored
    CORT_MASK.GOBJ reproduced with its start vertex, in the stored order and with no extra chain
    (per-slice list equality on all 168 slices, 336 chains)."""
    from ipldt.io import read_aim, align_to
    mask_f = os.path.join(data_root, "PFJ-0be66a_R", "X2420448_CORT_MASK_decompressed.AIM")
    g2a = _find("X2420448_T15_CORT_G2A.AIM")
    gobj = _find("X2420448_CORT_MASK.GOBJ")
    if not (os.path.exists(mask_f) and g2a and gobj):
        pytest.skip("PFJ-0be66a_R cortical mask or test-run-15 exports not available")
    m = read_aim(mask_f)
    M, dim, pos = m["data"] > 0, m["dim"], m["pos"]
    G = render_volume(M)
    ipl = read_aim(g2a[0])
    lo = [min(pos[i], ipl["pos"][i]) for i in range(3)]
    hi = [max(pos[i] + dim[i], ipl["pos"][i] + ipl["dim"][i]) for i in range(3)]
    udim, upos = tuple(h - l for h, l in zip(hi, lo)), tuple(lo)
    Gu = align_to(dict(data=G.astype(np.uint8), dim=dim, pos=pos), udim, upos) > 0
    Iu = align_to(ipl, udim, upos) > 0
    assert int(Iu.sum()) == 7_154_297
    assert int((Gu != Iu).sum()) == 0
    _, slices = read_gobj(gobj[-1])
    n = slices_ok = 0
    for s in slices:
        z = s["z"] - pos[2]
        if not s["contours"] or not (0 <= z < dim[2]):
            continue
        ours = [c["vertices"] for c in slice_chains(M[z])]
        ipl = []
        for c in s["contours"]:
            x, y = gobj_path(c["codes"], c["start"], 8)
            ipl.append([(int(a) - pos[0], int(b) - pos[1]) for a, b in zip(x, y)])
        n += len(ipl)
        slices_ok += ours == ipl        # same chains, count, stored order and start per slice
    assert n == 336 and slices_ok == 168


# ------------------------------------------------------------------------------- test run 19, the tiny-hole phantom (fast)
T19_MIRROR = os.environ.get("IPLDT_RUN19_MIRROR",
                            lab_path("ipl_test_runs/run19_mirror"))
T19_CHAIN_LENGTHS = {                       # stored contour lengths per test slice (outer first, then inner)
    "h_v12": [72, 6], "h_h12": [72], "h_11": [72], "h_22": [72, 8], "h_diag": [72], "h_v13": [72, 8], "h_v14": [72, 10],
    "c_hex8": [], "c_line3h": [], "c_line3v": [], "c_2x2": [], "c_3x3": [], "c_4x4": [8], "c_5x5": [12],
    "c_rib2x12": [], "c_rib3x12": [], "c_rib4x12": [24], "c_disc12": [8]}


def _phantom19():
    """The test-run-19 char phantom rebuilt from its manifest (the design of the test-run-19 phantom generator, not distributed: one test per 3-slice
    z range on PFJ-0be66a_R's TRAB_MASK grid, 20x20 blocks with one tiny hole each, free-standing tiny components):
    (bool (z, y, x) volume, manifest)."""
    man = json.load(open(os.path.join(T19_MIRROR, "phantom19_manifest.json")))
    a = np.zeros(tuple(man["dim"])[::-1], bool)
    for t in man["tests"].values():
        z0, z1 = t["z"]
        if t["kind"] == "hole":
            b = t["block"]
            a[z0:z1 + 1, b["y"][0]:b["y"][1] + 1, b["x"][0]:b["x"][1] + 1] = True
            for x, y in t["hole_pixels"]:
                a[z0:z1 + 1, y, x] = False
        else:
            for x, y in t["pixels"]:
                a[z0:z1 + 1, y, x] = True
    return a, man


@pytest.mark.skipif(not os.path.exists(os.path.join(T19_MIRROR, "phantom19_manifest.json")),
                    reason="the test-run-19 exports are not public (set IPLDT_RUN19_MIRROR or IPLDT_LAB_ROOT)")
def test_run19_tiny_hole_phantom_pins_min_vertices_class_4_to_6():
    """Test run 19 block B (scanner run 2026-09-14; the exports are not distributed): the
    phantom uploaded to IPL survives the read / write round trip (PHRT, 8,910 voxels); IPL's /togobj_from_aim
    -curvature_smooth 1 + /gobj_to_aim rendering (TINY, 8,544 voxels on the contour bounding box 217x59x88 @
    893,192,172) equals render_volume under MIN_VERTICES = 4 voxel for voxel, and would differ under 8 by exactly the
    6 hole voxels of h_v12 (a vertical 1x2 hole on 3 slices, global (905, 204..205, 172..174)); every stored chain of
    T19TINY.GOBJ (45 chains on 33 slices) equals slice_chains under 4 -- count, order, start vertex and vertices --
    while 8 would drop the 6-vertex inner chain of h_v12 on its 3 slices and nothing else.  So the raw un-activated
    6-vertex inner chain IS stored: MIN_VERTICES in 4..6 (the package's 4), 7..8 refuted.  The controls: the 2x2 /
    1x3 / 1x4 holes are stored raw with 8 / 8 / 10 elements, the 1x1 / horizontal 1x2 / diagonal holes are filled,
    the 8-px hexagon of PFJ-6f5538_R z332, the 3-px lines, the 2x2 / 3x3 blocks and the 2x12 / 3x12 ribbons get no
    contour, the 4x4 / 5x5 / 4x12 / 12-px disc are stored with 8 / 12 / 24 / 8 elements."""
    from ipldt.io import read_aim, align_to
    from ipldt.contour import render as R
    a, man = _phantom19()
    dim, pos = tuple(man["dim"]), tuple(man["pos"])
    assert dim == (743, 348, 168) and pos == (795, 94, 168) and int(a.sum()) == 8_910
    phrt = read_aim(os.path.join(T19_MIRROR, "oracle", "X2420448_T19_PHRT.AIM"))
    assert tuple(phrt["dim"]) == dim and tuple(phrt["pos"]) == pos
    assert np.array_equal(phrt["data"] != 0, a)
    tiny = read_aim(os.path.join(T19_MIRROR, "oracle", "X2420448_T19_TINY.AIM"))
    assert tuple(tiny["dim"]) == (217, 59, 88) and tuple(tiny["pos"]) == (893, 192, 172)
    T = align_to(tiny, dim, pos) > 0
    assert int(T.sum()) == 8_544
    assert MIN_VERTICES == 4 and R.MIN_VERTICES == 4
    G4 = render_volume(a)
    assert np.array_equal(G4, T)
    R.MIN_VERTICES = 8
    try:
        G8 = render_volume(a)
        ours8 = {z: [c["vertices"] for c in slice_chains(a[z])] for z in range(dim[2]) if a[z].any()}
    finally:
        R.MIN_VERTICES = 4
    d = G8 != T
    assert int(d.sum()) == 6 and int((G8 & ~T).sum()) == 6
    hv = man["tests"]["h_v12"]
    assert sorted((int(x) + pos[0], int(y) + pos[1], int(z) + pos[2]) for z, y, x in np.argwhere(d)) == \
        sorted((x + pos[0], y + pos[1], z + pos[2]) for x, y in hv["hole_pixels"] for z in range(hv["z"][0], hv["z"][1] + 1))
    assert sorted(set(map(tuple, np.argwhere(d)[:, ::-1] + pos))) == [(905, 204, 172), (905, 204, 173), (905, 204, 174),
                                                                       (905, 205, 172), (905, 205, 173), (905, 205, 174)]
    # the stored chains
    _, slices = read_gobj(os.path.join(T19_MIRROR, "oracle", "T19TINY.GOBJ"))
    ipl = {}
    for s in slices:
        if s["contours"]:
            ipl[int(s["z"]) - pos[2]] = [[(int(x) - pos[0], int(y) - pos[1]) for x, y in zip(*gobj_path(c["codes"], c["start"], 8))]
                                          for c in s["contours"]]
    assert len(ipl) == 33 and sum(len(v) for v in ipl.values()) == 45
    assert set(ipl) <= set(ours8)
    for name, t in man["tests"].items():
        for z in range(t["z"][0], t["z"][1] + 1):
            stored = ipl.get(z, [])
            assert [len(c) for c in stored] == T19_CHAIN_LENGTHS[name], (name, z)
            assert [c["vertices"] for c in slice_chains(a[z])] == stored, (name, z)          # MIN_VERTICES 4
            if name == "h_v12":
                assert ours8[z] == stored[:1] and len(stored[1]) == 6, z                     # 8 drops the 6-chain
            else:
                assert ours8[z] == stored, (name, z)
