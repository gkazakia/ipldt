"""ipldt.contour.render -- IPL's /togobj_from_aim -curvature_smooth 1 followed by /gobj_to_aim (the
rendered contour raster behind every gobj-masked command of Script 32), reimplemented slice by slice.

THE RULE, per slice of the mask raster:

  phase 1   drop every mask pixel whose W and E neighbours are both background (a horizontal run of
            length 1) EXCEPT (a) the raster-first pixel of its 8-connected component (the trace start) and
            (b) a pixel with mask directly above AND below it (a vertical neck: dropping it would split off
            a lobe that IPL keeps attached -- PFJ-8bcf88_R z167, PFJ-351dc7_R z413).  The vertical-neck exception
            rests on those two pixels, the cohort's only run-of-1 pixels with mask N and S (both are
            articulation points of their component; 0 of the 42,508 dropped run-of-1 pixels in the
            21-subject cohort have both N and S background or would split their component when dropped).
            A diagonal neck -- a run-of-1 pixel attached only through NW/SE or NE/SW, or a bump whose
            removal disconnects a lobe -- is therefore untested: it is dropped, the raw component splits,
            and outer_components emits one outer chain per piece, all with the same pre_activated flag;
            IPL's behaviour there is unknown.
  outer     one contour per 8-connected component of the phase-1 mask: the counter-clockwise-on-screen
            Moore trace from the component's raster-first pixel entered from the west (ipldt.contour.moore),
            smoothed by ipldt.contour.smooth.smooth_chain with pre_activated = phase 1 dropped a pixel of the
            raw component that lay on the raw component's own boundary trace (an endosteal drop, i.e. a
            dropped pixel facing a hole, does not activate: PFJ-351dc7_R z415).
  inner     one contour per 4-connected background hole not touching the slice frame: the CCW Moore trace
            of the hole's COMPLEMENT (no phase 1: endosteal bumps are never dropped) starting at the object
            pixel right after the hole's first-row run, backtrack west; smoothed with pre_activated False.
            A chain that the smoothing collapses is not stored (outer or inner): MIN_VERTICES = 4, i.e. a
            chain left with 3 vertices is dropped too.  The sweep has no fixed point below 8 vertices (every
            closed chain of 3..7 8-adjacent vertices changes under the five rules; the 16 smallest fixed points
            are 8-vertex rings), so a chain at 3 vertices -- where sweep() stops -- is still collapsing, and IPL
            stores nothing for it.  The theorem covers chains that stage B processes; an un-activated raw
            chain bypasses stage B and is stored raw, and stage A alone has 12 six-vertex fixed points, so a
            raw un-activated 6-vertex chain is the one shape that separates the thresholds 4..6 from 7..8.
            Probe 19 (2026-09-14, the tiny-hole phantom on PFJ-0be66a_R's TRAB_MASK grid, exports
            X2420448_P19_TINY / _PHRT and P19TINY.GOBJ, not distributed):
            a vertical 1x2 hole in a 20x20 block (test h_v12) yields exactly that chain and IPL STORES it -- a
            6-element inner contour on each of the 3 test slices, vertex for vertex ipldt's, the hole kept in
            /gobj_to_aim's rendering (MIN_VERTICES 4: 0 differing voxels on the whole export; 8 would fill the
            hole: 6 voxels at global (905, 204..205, 172..174)).  MIN_VERTICES in 4..6 is CONFIRMED, 7..8
            REFUTED.  The value within 4..6 stays unpinned, for a structural reason: no raw chain of 4, 5 or
            7 vertices ever reaches the stored list un-activated (the stage-A sweep has 0 fixed points among
            the 96 / 360 / 6,664 closed chains of 4 / 5 / 7 8-adjacent vertices, 12 of 1,512 with 6, 64 of
            31,056 with 8), every activated survivor has >= 8 (the stage-B theorem), and the 3-vertex frozen
            chains never reach it either -- so 4, 5 and 6 are observationally identical: no mask can make IPL
            store or drop a 4- or 5-vertex chain, and the package keeps 4 as its convention.  The other 17
            phantom tests match under MIN 4 (and, being identical under both classes, under 8): h_h12 /
            h_11 / h_diag one 72-vertex outer chain each, hole filled; h_22 / h_v13 / h_v14 raw un-activated
            8 / 8 / 10-vertex inner chains stored raw; c_hex8 (the 8-px hexagon of PFJ-6f5538_R z332), c_line3h,
            c_line3v, c_2x2, c_3x3, c_rib2x12, c_rib3x12 NOT stored (0 voxels rendered, no contour);
            c_4x4 8-vertex chain, c_5x5 12, c_rib4x12 24, c_disc12 raw 8 -- 45 stored chains on 54 test
            slices, all exact; the phantom round trip PHRT exact (8,910 voxels); IPL's /gobj_to_aim grid is
            the contour bounding box (217x59x88 @ 893,192,172) and its log prints 'Sli N CLEARED' for every
            contour-free slice.  Oracles before the phantom (2026-09-14): PFJ-6f5538_R
            P17 29_trabfinal z332, an 8-px hexagon on the last slice (raw 6 -> stage A 3 vertices): IPL's
            P17TRAB gobj has no contour on that slice (its /gobj_to_aim grid ends at z 331 while the mask ends
            at z 332), MIN_VERTICES = 3 rendered 3 voxels; X2420448_CORT_MASK_version1 vs
            X2420448_CORT_MASK.GOBJ (version 1): 14/17/18/18/20-px components stored with 8/8/8/8/12 elements, exact.
            Consistent evidence, not an oracle: X3931708_CORT_MASK_version2 vs X3931708_CORT_MASK.GOBJ (version 2) is not
            an exact mask -> gobj pair (48 of its slices differ for other reasons); on it 13 more chains freeze
            at 3 (6..40-px components and pieces of larger ones) without an IPL contour, while every small
            component whose chain survives has its IPL chain (27 px -> 8 vertices at z401, 12 vertices at z92,
            ...).  The cohort's two fully collapsed chains, PFJ-6f5538_R z265 trab (5-px
            diamond component: 0 vertices) and cort (5-px diamond hole: 2 vertices), are not stored either.
            REFUTED: a minimum pixel count (14-px components are stored, 28..61-px components whose chain
            collapses are not: version2 z47, z64, z48, z22, z61, z51, z58, z52); a degenerate-polygon test
            (the frozen triangle has area 1/2); -min_elements 0 is 'no user minimum' (IPL help).
  order     the stored slice lists the outer contours first (raster order of the components), then the
            inner contours (raster order of the holes).  Verified with several outer chains on PFJ-ab6af9_R's
            earlier cortical gobj X3931708_CORT_MASK.GOBJ (version 2) (46 slices with 2-4 outer chains of
            XCT_masks/PFJ-ab6af9_R/X3931708_CORT_MASK_version2.AIM reproduced in stored order, 0 order
            mismatches); the order among several inner chains and the outer/inner interleaving of a slice
            with several outers AND a hole are the natural reading (no IPL gobj on disk has a slice with two
            holes) and do not affect the raster (render_slice is order-independent; the package writes no
            GOBJ).  A chain starts at its trace start (the raster-first pixel / the pixel after the hole's
            first-row run), moved or not; when the smoothing deletes the start vertex the chain starts at the
            surviving vertex before it (ipldt.contour.smooth).
  fill      a chain is a closed polygon through pixel centres.  Rendered pixels of an outer contour = its
            chain pixels + the pixels whose centres lie strictly inside the polygon (even-odd rule on the
            integer scanlines: every non-horizontal edge joins 8-adjacent centres and crosses exactly one
            scanline, at its lower-y endpoint; a pixel is inside iff an odd number of crossings lie strictly
            to its left).  An inner contour KEEPS its chain pixels and removes only its strict interior from
            the outer fill whose polygon interior contains pixels of its hole.  A lobe of cortex that the
            inner chain wraps around through a 1-pixel neck is enclosed by chain pixels but outside the
            polygon, and IPL keeps it (PFJ-ab6af9_R z433, PFJ-351dc7_R z413, PFJ-8bcf88_R z167, PFJ-0fb201_R z335,
            PFJ-a56aae_R z258, PFJ-69bcb0_L z167).  Verified domain: every cohort slice with a hole has exactly one
            8-connected component (3,946 holes, 0 islands, 0 slices with >= 2 outer chains).  A component
            nested inside another component's hole is outside the verified domain: the code keeps it (global
            even-odd behaviour; a sequential fill/erase renderer would erase it).

VERIFICATION (2026-09-13, probe-15 exports, whole volumes compared by global position on the union of the
two grids; probe-15 exports, not distributed):
  cortical  render_volume(<base>_CORT_MASK.AIM) vs <base>_P15_CORT_G2A.AIM (gobj_to_aim of the cortical
            gobj) and vs <base>_P15_CORT_CORTGRID_P0.AIM (gobj_maskaimpeel peel 0 on the CORT_MASK grid):
            0 mismatches on all 21 subjects, |IPL| == |ours| (e.g. PFJ-ab6af9_R 19,042,175, PFJ-0be66a_R 7,154,297,
            PFJ-351dc7_R 21,755,789, PFJ-6f5538_R 11,450,390, PFJ-69bcb0_L 12,269,733);
  trabecular render_volume(<base>_TRAB_MASK.AIM) vs <base>_P15_TRAB_G2A.AIM and vs
            <base>_P15_TRAB_SEGGRID_P0.AIM (peel 0 on the SEG grid; peel 0 == peel -1 there): 0 mismatches
            on all 21 subjects (e.g. PFJ-ab6af9_R 35,988,981, PFJ-0be66a_R 20,328,498, PFJ-69bcb0_R 26,282,267);
  chains    every stored contour of the newest TRAB_MASK.GOBJ / CORT_MASK.GOBJ of the 21 subjects
            (P5MASK.GOBJ / P7MASK.GOBJ for PFJ-0be66a_R / PFJ-8bcf88_R trabecular) reproduced vertex for vertex,
            start and stored order included: 11,906 / 11,906 (3,945 trabecular, 4,016 + 3,945 cortical);
            the two derivation modules (firstslice_rule / cortpoly_rule, not distributed) give the same chains on
            all 7,961 slices up to the start vertex: on the 4 chains whose start the smoothing deletes
            (PFJ-ab6af9_R trab z469, PFJ-351dc7_R trab z376, z381, z382) both scratch modules keep the deleted
            start's successor as the start while IPL stores its predecessor (a production-only rule in
            ipldt.contour.smooth.sweep, verified on those 4 chains); firstslice_rule's drop-every-run-of-1
            phase 1 additionally splits off a lobe on 2 cortical slices (PFJ-351dc7_R z413, PFJ-8bcf88_R z167) that
            IPL does not store as a contour (exact-list comparison: fs differs on 6 slices, cp on 4);
            stored order: one outer + at most one inner per cohort slice; multi-outer order on
            X3931708_CORT_MASK.GOBJ (version 2) (46 slices);
  replay    PFJ-0be66a_R's P16 replay: render_volume(X2420448_P16_28_CORTFINAL) vs X2420448_P16_30_CORTGOBJ 0,
            render_volume(X2420448_P16_29_TRABFINAL) vs X2420448_P16_31_TRABGOBJ 0; P16CORT.GOBJ 336/336
            and P16TRAB.GOBJ 168/168 chains.
  tiny      2026-09-14 (MIN_VERTICES 3 -> 4): the 21 x 2 cohort renderings, the P17 30/31 renderings
            of PFJ-0be66a_R / PFJ-42293d_L / PFJ-6f5538_R / PFJ-411dfd_R and PFJ-0be66a_R's P16 30/31 re-run at 0 mismatches (52
            volumes; PFJ-6f5538_R P17 31_trabgobj 3 -> 0, |ours| = |IPL| = 1,893,500) and 12,914 / 12,914 stored chains
            exact (11,906 cohort + P17CORT/P17TRAB of PFJ-411dfd_R 336 + 168 + P16CORT/P16TRAB 336 + 168); a
            guard-free sweep (chains swept down to 3 vertices, MIN_VERTICES 3) gives the same 52 rasters.

REFUTED by the exports: no phase 1 (4,989 trabecular voxels); phase 1 dropping EVERY run of 1 (31 cortical
voxels in 2 slices); phase-1 drops anywhere in the component as the activator (PFJ-351dc7_R z415); binary_fill_holes
of the inner chain raster (142 IPL-only voxels in 6 slices); unconditional smoothing (1,444 cortical voxels).
Undetermined: the outer fill -- binary_fill_holes of the outer chain raster and the even-odd interior agree on
every slice of the cohort; the even-odd interior is used for both contours as the single polygon mechanism.
The connectivity conventions -- 8-connected object components (outer_components) and 4-connected background
holes (holes) -- are assumed (the Jordan-consistent pairing for the Moore 8-neighbour trace): no slice of the
21-subject cohort (7,962 raw CORT/TRAB slices, 4,016 periosteal-union slices) has a 4-vs-8 difference in
either the object or the background partition, so IPL's choice is unobserved.  Likewise unobserved: a
component nested in another component's hole (fill paragraph) and a diagonal neck (phase-1 paragraph).

COST: per slice roughly 0.5 ms per component or hole (full-slice label comparisons in outer_components and
holes) plus one full-slice scanline fill per chain; each inner contour is matched to the outer fills through
the hole label array (one gather per outer chain, not one full-slice AND per outer x hole pair, which was
quadratic: 1,600 synthetic rings 45 s -> 4.7 s, PFJ-0be66a SEG slice 84 2.2 s -> 1.3 s, compartment masks
unchanged).  Meant for compartment masks (periosteal / cortical / trabecular: 1-2 contours per slice, 2-5 s
per PFJ-0be66a volume of 168 slices), not for SEG-like rasters with hundreds of components per slice (PFJ-0be66a SEG
slice 84, 416 components / 234 holes: 1.3 s for the one slice).
"""
import numpy as np
from scipy import ndimage as ndi

from .moore import moore_trace
from .smooth import smooth_chain

S8 = np.ones((3, 3), bool)
S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
MIN_VERTICES = 4          # a chain with fewer vertices is not stored.  The smoothing sweep (ipldt.contour.smooth.sweep) has
                          # no fixed point below 8 vertices (brute force over every closed chain of 3..7 8-adjacent vertices),
                          # so a chain still shrinking at 3 vertices -- where sweep() stops (guard n < 4) -- is a collapsing
                          # chain, and IPL stores nothing for it: PFJ-6f5538_R P17 29_trabfinal z332 (8-px hexagon, 6 -> 3
                          # vertices, no contour in IPL's gobj, whose z range ends one slice short) and, as consistent evidence
                          # rather than an oracle (not an exact mask -> gobj pair), 13 frozen-at-3 chains of
                          # X3931708_CORT_MASK_version2 vs X3931708_CORT_MASK.GOBJ (version 2).  IPL's shortest stored contours have
                          # 8 elements (14-px component, PFJ-0be66a_R X2420448_CORT_MASK.GOBJ (version 1) z204), so the values 4..8 agree
                          # on every stored chain of the cohort oracles.  Probe 19 (2026-09-14, tiny-hole phantom): an
                          # un-activated raw 6-vertex inner chain (vertical 1x2 hole, which bypasses stage B) IS stored by
                          # IPL and its hole kept -> 4..6 confirmed, 7..8 refuted.  4 vs 5 vs 6 is unpinned for good: no raw
                          # chain of 4, 5 or 7 vertices is ever un-activated (stage A has no fixed point of those lengths)
                          # and activated survivors have >= 8, so the three values are observationally identical
                          # (ipldt.contour.smooth, TINY CHAINS); 3 stored the frozen triangle (2026-09-14, p17fix/contour).


def raster_first(mask):
    """(x, y) of the first True pixel in raster order (top row first, left to right)."""
    ys, xs = np.nonzero(mask)
    return int(xs[0]), int(ys[0])


def phase1(sl):
    """The pre-trace pixel rule on a bool slice: a mask pixel whose W and E neighbours are both background
    is dropped unless mask lies directly above AND below it (a vertical neck).  The raster-first pixel of
    each component is protected by outer_components(), not here.  The vertical-neck exception rests on two
    cohort pixels (PFJ-8bcf88_R z167, PFJ-351dc7_R z413); a diagonal neck (attached only through NW/SE or NE/SW) is
    dropped and splits the component -- untested against IPL (module docstring)."""
    W = np.zeros_like(sl)
    E = np.zeros_like(sl)
    N = np.zeros_like(sl)
    S = np.zeros_like(sl)
    W[:, 1:] = sl[:, :-1]
    E[:, :-1] = sl[:, 1:]
    N[1:] = sl[:-1]
    S[:-1] = sl[1:]
    run_of_one = sl & ~W & ~E
    return sl & ~(run_of_one & ~(N & S))


def holes(sl):
    """(labels, ids) of the 4-connected background components that do not touch the slice frame."""
    lab, n = ndi.label(~sl, structure=S4)
    frame = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    return lab, [h for h in range(1, n + 1) if h not in frame]


def outer_components(sl):
    """[(component, pre_activated)] for the 8-connected components of the phase-1 mask, in raster order.
    The raster-first pixel of every raw component is kept; pre_activated is True when phase 1 dropped a
    pixel of the raw component that lay on the raw component's CCW Moore trace."""
    lab0, n0 = ndi.label(sl, structure=S8)
    m = phase1(sl)
    for k in range(1, n0 + 1):
        x0, y0 = raster_first(lab0 == k)
        m[y0, x0] = True
    dropped = sl & ~m
    lab, n = ndi.label(m, structure=S8)
    boundary = {}
    out = []
    for k in range(1, n + 1):
        comp = lab == k
        k0 = int(lab0[comp][0])                       # the raw component this piece came from
        drops = dropped & (lab0 == k0)
        pre = False
        if drops.any():
            if k0 not in boundary:
                comp0 = lab0 == k0
                boundary[k0] = set(moore_trace(comp0, raster_first(comp0), backtrack=(-1, 0), ccw=True))
            pre = any((int(x), int(y)) in boundary[k0] for y, x in zip(*np.nonzero(drops)))
        out.append((comp, pre))
    return out


def outer_chain(comp, pre_activated=False):
    """IPL's stored outer chain of one phase-1 component: CCW Moore trace from the raster-first pixel
    (entered from the west) + the staged smoothing; None when it is not stored (< MIN_VERTICES = 4 vertices)."""
    raw = moore_trace(comp, raster_first(comp), backtrack=(-1, 0), ccw=True)
    pts = smooth_chain(raw, pre_activated)
    return pts if len(pts) >= MIN_VERTICES else None


def inner_chain(hole):
    """IPL's stored inner (endosteal) chain of one hole (bool mask of the 4-connected background region):
    CCW Moore trace of the hole's complement from the object pixel right after the hole's first-row run,
    backtrack west, then the staged smoothing; None when it is not stored (< MIN_VERTICES = 4 vertices)."""
    hx, hy = raster_first(hole)
    xe = hx
    while xe + 1 < hole.shape[1] and hole[hy, xe + 1]:
        xe += 1
    raw = moore_trace(~hole, (xe + 1, hy), backtrack=(-1, 0), ccw=True)
    pts = smooth_chain(raw)
    return pts if len(pts) >= MIN_VERTICES else None


def _contours(sl):
    """([(kind, vertices, region, hole id)] of one bool slice in stored order -- the outer chains with their
    component (hole id 0), then the inner chains with their hole and its label in `hole_labels` --,
    hole_labels): the 4-connected background labelling that holes() computed."""
    out = []
    for comp, pre in outer_components(sl):
        pts = outer_chain(comp, pre)
        if pts is not None:
            out.append(("outer", pts, comp, 0))
    lab, ids = holes(sl)
    for h in ids:
        hole = lab == h
        pts = inner_chain(hole)
        if pts is not None:
            out.append(("inner", pts, hole, h))
    return out, lab


def _check_slice(sl):
    sl = np.asarray(sl, bool)
    if sl.ndim != 2:
        raise ValueError(f"expected a 2-D (y, x) slice, got shape {sl.shape}")
    return sl


def slice_chains(sl):
    """IPL's stored contours of one slice: [dict(kind='outer' | 'inner', vertices=[(x, y), ...])] in the
    stored order (outer chains first, then inner chains), vertices in slice pixel coordinates (add the
    volume's pos for scanner coordinates), each chain starting at IPL's stored start vertex."""
    sl = _check_slice(sl)
    if not sl.any():
        return []
    contours, _ = _contours(sl)
    return [dict(kind=kind, vertices=pts) for kind, pts, _, _ in contours]


def polygon_fill(vertices, shape):
    """(chain raster, strict interior) of the closed polygon through `vertices` (pixel centres, 8-adjacent
    consecutive vertices), even-odd rule on integer scanlines: every non-horizontal edge crosses the
    scanline of its lower-y endpoint at that endpoint's x; a pixel is inside iff an odd number of crossings
    lie strictly to its left.  Chain pixels are excluded from the strict interior."""
    H, W = shape
    n = len(vertices)
    xs = np.fromiter((p[0] for p in vertices), np.int64, n)
    ys = np.fromiter((p[1] for p in vertices), np.int64, n)
    x1, y1 = np.roll(xs, -1), np.roll(ys, -1)
    dy = y1 - ys
    cx = np.where(dy > 0, xs, x1)[dy != 0]
    cy = np.where(dy > 0, ys, y1)[dy != 0]
    cross = np.zeros((H, W + 1), np.int64)
    np.add.at(cross, (cy, cx), 1)
    parity = (np.cumsum(cross, axis=1)[:, :-1] % 2).astype(bool)       # crossings at x' <= x
    chain = np.zeros(shape, bool)
    chain[ys, xs] = True
    inside = np.zeros(shape, bool)
    inside[:, 1:] = parity[:, :-1]                                       # crossings strictly left of x
    return chain, inside & ~chain


def render_slice(sl):
    """The rendered gobj raster (bool) of one bool slice: the union of the outer fills, each with the strict
    interior of every inner contour whose hole its polygon interior meets removed."""
    sl = _check_slice(sl)
    G = np.zeros(sl.shape, bool)
    if not sl.any():
        return G
    contours, hole_labels = _contours(sl)
    inner = {h: polygon_fill(pts, sl.shape)[1] for kind, pts, _, h in contours if kind == "inner"}
    for kind, pts, _, _ in contours:
        if kind != "outer":
            continue
        chain, interior = polygon_fill(pts, sl.shape)
        F = chain | interior
        for h in np.unique(hole_labels[interior]):    # the holes (label > 0) this polygon's interior meets
            strict = inner.get(int(h))
            if strict is not None:
                F &= ~strict
        G |= F
    return G


def render_volume(mask, verbose=False):
    """The rendered gobj raster (bool, same shape) of a bool (z, y, x) mask volume, slice by slice."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 3:
        raise ValueError(f"expected a 3-D (z, y, x) mask, got shape {mask.shape}")
    G = np.zeros(mask.shape, bool)
    for z in range(mask.shape[0]):
        if mask[z].any():
            try:
                G[z] = render_slice(mask[z])
            except RuntimeError as e:
                raise RuntimeError(f"slice {z}: {e}") from e
        if verbose and z % 50 == 0:
            print("   slice", z, flush=True)
    return G
