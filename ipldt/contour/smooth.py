"""ipldt.contour.smooth -- IPL's /togobj_from_aim -curvature_smooth 1 on one closed chain of 8-adjacent
pixel centres (the raw Moore trace of a component or of a hole, see ipldt.contour.render).

THE RULE.  Vertex rule on p[i] with d = p[i+1] - p[i-1], the surviving neighbours at their positions as
already updated in the current sweep:

    |d| == (1, 1)                        delete p[i]                    corner (90-degree turn)
    |d| == (0, 0)                        delete p[i] and p[i+1]         spur tip (the chain doubles back)
    |d| in {(1, 0), (0, 1)}              delete p[i]                    135-degree turn (neighbours 4-adjacent)
    |d| == (0, 2), p[i] off the line     p[i] := midpoint(p[i-1], p[i+1])   excursion on VERTICAL travel: a
                                                                        1-pixel bump or notch in a vertical
                                                                        run, p[i] is displaced horizontally
    |d| == (2, 0), p[i] off the line     p[i] := midpoint                excursion on HORIZONTAL travel: a
                                                                        bump or notch in a horizontal run,
                                                                        p[i] is displaced vertically

A sweep visits the vertices forward and in place (a deletion or a move is seen by the vertices after it in
the same sweep), the start vertex (index 0) last.  When the start vertex itself is deleted, the surviving
vertex BEFORE it in trace order becomes the chain's start (it is the stored start, and the vertex visited
last in later sweeps): the start is deleted in 4 of the cohort's 11,906 chains (PFJ-ab6af9_R trab z469, PFJ-351dc7_R
trab z376, z381, z382) and IPL's stored start is the predecessor in all four; whether the rotation happens
at once or after the last sweep makes no difference on those chains.  The rules are applied in two stages:

    stage A   exactly ONE sweep with the first four rules -- no horizontal-travel excursions;
    stage B   all five rules, swept until a sweep changes nothing.  Stage B runs ONLY if stage A deleted or
              moved a vertex, or the caller reports a pre-trace change (`pre_activated`: phase 1 removed a
              pixel that lay on the raw component's boundary trace, ipldt.contour.render).  Otherwise the
              chain is returned RAW (the stage-A sweep changed nothing, so nothing is discarded).

Consequence: a chain whose only defects are 1-pixel bumps or notches in horizontal runs, and whose component
lost no boundary pixel to phase 1, is stored unsmoothed by IPL -- 4 whole trabecular slices (PFJ-d81140_R z0,
PFJ-0fb201_R z168, PFJ-6f5538_R z168, PFJ-6b714e_R z0), 135 cortical inner contours and 1 cortical outer contour
(PFJ-351dc7_R z415) of the 21-subject cohort; 4,154 of its 11,906 stored chains are raw under the rule: those
140 with horizontal-travel excursions only, plus 4,014 event-free chains (4,009 cortical outer, 4 cortical
inner, 1 trabecular: PFJ-351dc7_R z484, so 5 trabecular chains are raw in all).  4,016 stored chains are
event-free in total: the 4,014 plus 2 trabecular chains (PFJ-351dc7_R z485, PFJ-6f5538_R z261) activated by
pre_activated alone, whose stage B then changes nothing.
The two pre-activation criteria of the derivation (a dropped pixel on the raw boundary trace / a dropped pixel
W-E adjacent to exterior background) never differ on the cohort's 7,961 outer chains.

TINY CHAINS (2026-09-14, test run 17).  sweep() leaves a chain of fewer than 4 vertices untouched (guard n < 4),
so a chain that a sweep brings down to 3 vertices freezes there.  A frozen 3-chain is NOT a stored contour:
the sweep has no fixed point below 8 vertices -- brute force over every closed chain of 3 / 4 / 5 / 6 / 7
distinct 8-adjacent vertices (24 / 96 / 360 / 1,512 / 6,664 chains): 0 fixed points; the 16 smallest fixed
points are 8-vertex rings (of 31,056 8-chains) -- so a chain at 3 vertices is still collapsing, and IPL stores
nothing for it.  ipldt.contour.render.MIN_VERTICES = 4 drops it.  IPL's shortest stored cohort contours have
8 elements (7 + 7 in X3931708_CORT_MASK.GOBJ versions 2 / 3, 4 in X2420448_CORT_MASK.GOBJ (version 1), 18 in all), so the
thresholds 4..8 agree on every stored chain of those oracles, and the no-fixed-point-below-8 theorem covers
only chains that stage B processes: an UN-ACTIVATED raw chain bypasses stage B, stage A alone has 12
six-vertex fixed points (brute force over the same 1,512 6-chains; 0 of 96 / 360 / 6,664 with 4 / 5 / 7
vertices, 64 of 31,056 with 8), and a vertical 1x2 hole yields an un-activated raw 6-vertex inner chain that
MIN_VERTICES = 4 stores (hole kept) and 8 would drop (hole filled).  Test run 19 (2026-09-14, the tiny-hole
phantom on PFJ-0be66a_R's TRAB_MASK grid; exports X2420448_T19_TINY / _PHRT, T19TINY.GOBJ, not
distributed) put that hole in front of IPL: the 6-vertex chain IS stored (a
6-element inner contour on each of the 3 test slices, vertex for vertex ours) and the hole is kept in
/gobj_to_aim's rendering (0 differing voxels under MIN 4 on the whole export; MIN 8 differs by the 6 hole
voxels).  MIN_VERTICES in 4..6 is CONFIRMED and 7..8 REFUTED; 4 vs 5 vs 6 remains unpinned, and it cannot be
pinned by any mask: a raw chain of 4, 5 or 7 vertices is never un-activated (no stage-A fixed point of those
lengths), activated survivors have >= 8, and frozen 3-chains are dropped under all three values -- the three
thresholds are observationally identical, so 4 is a convention of the package with no IPL-observable
alternative inside the class.  The test run's other 17 tests (2x2 / 1x3 / 1x4 holes stored raw with 8 / 8 / 10
elements; the 1x1, horizontal 1x2 and diagonal holes filled; the 8-px hexagon, 3-px lines, 2x2 / 3x3 blocks
and 2x12 / 3x12 ribbons not stored; 4x4 / 5x5 / 4x12 / 12-px disc stored with 8 / 12 / 24 / 8 elements) are
exact under the rule: 45 / 45 stored chains, rendering 8,544 voxels with 0 differences.  A guard-free sweep
deletes every one of the 24 closed 3-chains within one sweep and gives the same 52 verification rasters with
MIN_VERTICES = 3; the guard is kept (it also protects the 2-vertex remains of a spur deletion) and the
threshold does the work.
Oracles: PFJ-6f5538_R T17 29_trabfinal z332, an 8-px hexagon on the last slice (raw 6 vertices -> stage A 3,
frozen): IPL's T17TRAB gobj has no contour on that slice and its /gobj_to_aim grid ends one slice short;
MIN_VERTICES = 3 rendered 3 voxels there (the T17 stage-31 residual, 3 -> 0); X2420448_CORT_MASK_version1 vs
X2420448_CORT_MASK.GOBJ (version 1): 14 / 17 / 18 / 18 / 20-px components stored with 8 / 8 / 8 / 8 / 12 elements,
exact.  Consistent evidence, NOT an oracle: X3931708_CORT_MASK_version2 vs X3931708_CORT_MASK.GOBJ (version 2) is not
an exact mask -> gobj pair (48 of its slices differ for other reasons); on it 13 more chains freeze at 3
(6..40-px components and pieces of larger ones), none with an IPL contour, while every small component whose
chain survives (27 px -> 8 vertices at z401, 12 at z92, ...) has its IPL chain.  Refuted: a minimum pixel count
(14-px components are stored, 28..61-px ones whose chain collapses are not), a degenerate-polygon test (the
frozen triangle has area 1/2), -min_elements (0 = no user minimum, IPL help).

VERIFICATION (2026-09-13, test-run-15 exports; the raster numbers are in ipldt.contour.render): every stored
chain of the 21 subjects' newest TRAB_MASK.GOBJ / CORT_MASK.GOBJ (P5MASK.GOBJ / P7MASK.GOBJ for the
trabecular contours of PFJ-0be66a_R / PFJ-8bcf88_R) reproduced vertex for vertex, stored start included, in stored
order: 11,906 / 11,906 chains (3,945 trabecular outer, 4,016 cortical outer, 3,945 cortical inner), plus the
504 contours of PFJ-0be66a's T16 replay gobjs (T16TRAB.GOBJ 168, T16CORT.GOBJ 336).  Under the rule 4,015 outer
and 139 inner chains are stored raw; stage B needed at most 14 sweeps.

REFUTED by the stored chains (counts from the derivation, scratch modules firstslice_rule / cortpoly_rule):
  - unconditional smoothing (this module's earlier rule, stage B alone): 142 of 11,906 chains, 1,444
    cortical voxels, 8/6/9/6 trabecular voxels on PFJ-d81140_R/PFJ-0fb201_R/PFJ-6f5538_R/PFJ-6b714e_R;
  - stage A iterated to convergence instead of one sweep: PFJ-351dc7_R z478, PFJ-0fb201_R z335 (5 voxels);
  - the gate evaluated on the RAW chain's events followed by one combined sweep: PFJ-8bcf88_L z128, PFJ-8bcf88_R z92;
  - Jacobi (non in-place) moves 2,709 chains; y-moves committed after the sweep 2,348; convex-only corner
    test 87; discarding the moves of every deletion-free sweep 254; a deletions-only first sweep 237-243;
  - an equivalent single-loop formulation (all five rules in every sweep, deletions and horizontal
    displacements setting a flag, an unflagged first sweep discarded, corner test on sweep-start y) gives
    the same result on all 11,906 chains; the staged form is kept because it needs no sweep-start state.
"""

CORNER, SPUR, SHARP, EXCURSION_V, EXCURSION_H = "corner", "spur", "sharp", "excursion_v", "excursion_h"
STAGE_A_RULES = frozenset((CORNER, SPUR, SHARP, EXCURSION_V))
ALL_RULES = STAGE_A_RULES | {EXCURSION_H}
MAX_SWEEPS = 1000        # floor of the stage-B cap: the effective cap is max(MAX_SWEEPS, len(raw)).  Stage B
                         # converges in <= 14 sweeps on the cohort; a 1-3 px thick axis-aligned ribbon needs
                         # ~L/2 sweeps per free tip (only the spur rule fires there, 2 vertices per tip and
                         # sweep) and collapses to < 3 vertices; exceeding the cap raises RuntimeError


def sweep(pts, rules=ALL_RULES):
    """One forward in-place sweep of the vertex rule over a closed chain, start vertex last.
    `rules` selects the active triggers.  Returns (new chain, changed); if the start vertex was deleted the
    new chain starts at its surviving predecessor.  A chain of fewer than 4 vertices is returned unchanged
    (guard): a chain frozen at 3 vertices is a collapsing chain and is not stored (render.MIN_VERTICES = 4;
    TINY CHAINS in the module docstring)."""
    n = len(pts)
    if n < 4:
        return list(pts), False
    new = list(pts)
    keep = [True] * n
    changed = False
    for i in list(range(1, n)) + [0]:
        if not keep[i]:
            continue
        ip = (i - 1) % n
        while not keep[ip]:
            ip = (ip - 1) % n
        inx = (i + 1) % n
        while not keep[inx]:
            inx = (inx + 1) % n
        if ip == i or inx == i:
            break
        (px, py), (nx, ny), here = new[ip], new[inx], new[i]
        dx, dy = nx - px, ny - py
        ad = (abs(dx), abs(dy))
        if ad == (1, 1):
            if CORNER in rules:
                keep[i] = False
                changed = True
        elif ad == (0, 0):
            if SPUR in rules:
                keep[i] = False
                keep[inx] = False
                changed = True
        elif ad in ((1, 0), (0, 1)):
            if SHARP in rules:
                keep[i] = False
                changed = True
        elif ad == (0, 2):
            if EXCURSION_V in rules:
                mid = (px, py + dy // 2)
                if mid != here:
                    new[i] = mid
                    changed = True
        elif ad == (2, 0):
            if EXCURSION_H in rules:
                mid = (px + dx // 2, py)
                if mid != here:
                    new[i] = mid
                    changed = True
    survivors = [new[k] for k in range(n) if keep[k]]
    if not keep[0] and survivors:
        survivors = survivors[-1:] + survivors[:-1]
    return survivors, changed


def smooth_chain(raw, pre_activated=False, stats=None):
    """IPL's curvature_smooth-1 result for one raw closed chain [(x, y), ...].
    pre_activated: the chain's component lost a boundary pixel to phase 1 (stage B then runs even if
    stage A changes nothing).  `stats`, if given, receives stage_a_changed, activated and sweeps."""
    pts, stage_a_changed = sweep(raw, STAGE_A_RULES)
    activated = stage_a_changed or bool(pre_activated)
    sweeps = 0
    if activated:
        cap = max(MAX_SWEEPS, len(raw))   # a sweep that deletes a vertex can happen at most len(raw) times
        for sweeps in range(1, cap + 1):
            pts, changed = sweep(pts, ALL_RULES)
            if not changed:
                break
        else:
            raise RuntimeError(f"curvature smoothing did not converge in {cap} sweeps "
                               f"(chain of {len(raw)} raw vertices starting at {raw[0]})")
    else:
        pts = list(raw)
    if stats is not None:
        stats.update(stage_a_changed=stage_a_changed, activated=activated, sweeps=sweeps)
    return pts


def cyclic_equal(a, b):
    """True if the closed chains a and b are the same vertex sequence up to a rotation of the start."""
    if len(a) != len(b):
        return False
    if not b:
        return True
    try:
        k = a.index(b[0])
    except ValueError:
        return False
    return a[k:] + a[:k] == b
