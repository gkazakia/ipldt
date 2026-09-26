"""S2_morphology.py -- Supplementary Figure S2 of the ipldt / ORMIR-BQRL manuscript: the chamfer-metric
morphology of IPL (-metric 11) as ipldt reimplements it.

Panels
  A  the chamfer 3-4-5 metric on a 13 x 13 grid (synthetic): the raw distance of every voxel to one voxel and
     the ball raw < 3N + 2 that every metric-11 command uses (N = 3);
  B  IPL's dilation ball on scanner output: a synthetic 5 x 5 x 5 block dilated by 15 in IPL, in the plane
     through its centre, outlined with the reimplemented ball and the Euclidean and 3N + 3 alternatives;
  C  how many voxels of the 39^3 window each of the five threshold readings of B gets wrong against IPL;
  D-G  /erosion 3, /dilation 3, /close 15 and /open 15 (IPL's standard distances of the compartment separation)
     on one 120 x 120-voxel window of the mid slice of a patella scan: the input stage, the voxels IPL removes or
     adds, ipldt's result as an outline, and the whole-volume comparison;
  H  the 6-voxel gobj peel (/gobj_maskaimpeel_ow -peel_iter 6) applied to the trabecular candidate, same slice;
  I  the erosion at a face of the volume: nothing outside the volume counts as background (an x-z section
     through the first 24 slices);
  J  the mirrored margin of /open on a second patella scan: an x-z section through the last slices and the
     N + 2 = 17-deep margin, the erosion survivors in the margin that seed the dilation, and the voxels of the
     last slice they restore;
  K  the eleven margin conventions against IPL's /open 15 of that volume;
  L  /open N for N = 11 .. 18 on the same volume, each convention against IPL's export (N = 15 is the standard).

Run from the repository root in the `ormir` environment (numpy, scipy, numba, matplotlib; no GPU needed):

    python manuscript/figures/supp/S2_morphology.py [--recompute]

Writes manuscript/figures/supp/S2_morphology.png (300 dpi, 180 mm wide), S2_morphology.svg and
S2_morphology_numbers.json (every number drawn).  The computed crops and counts are cached in
manuscript/figures/cache/S2_morphology_cache.{npz,json} after the first run (about 1 min: 46 full-volume
opens for panels K and L); --recompute redoes everything.

Inputs (read only): IPL's stage exports of the patella scan's compartment separation (P16_DIR), IPL's dilation
of the synthetic block volume and the volume's manifest (P18_DIR), the stage-14 / stage-15 exports of the second
patella scan (P17_DIR) and IPL's /open N of its stage 14 for N = 11 .. 18 (P20_DIR).  Every ipldt number is
computed here by calling ipldt.ipl_ops on those files; nothing under ipldt/ is modified.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
import ipldt.ipl_ops as ops  # noqa: E402
from ipldt.io import read_aim, align_to  # noqa: E402

# ------------------------------------------------------------------------------------------------ inputs
P16_DIR = lab_path("Python/scripts/IPL/probes/p15_gobj_render/aims_and_logs")   # patella 1, STEP 1 exports
P16_BASE = "X2420448_P16_"
P18_DIR = lab_path("Python/scripts/IPL/probes/p18_step1_edges")                  # synthetic block volume
P17_DIR = lab_path("Python/scripts/IPL/probes/p17_step1_radius/aims_and_logs")   # patella 2, stages 14 / 15
P20_DIR = lab_path("Python/scripts/IPL/probes/p20_open_mechanism/aims_and_logs") # patella 2, /open 11..18
P17_BASE = "X5143651_P17_"
P20_BASE = "X5143651_P20_"
CACHE = os.path.join(REPO, "manuscript", "figures", "cache", "S2_morphology_cache")
OUT = os.path.join(HERE, "S2_morphology")

VOX_MM = 0.0607
Z_MID = 252                 # global z of the shown slice of patella 1 (its stage volumes span z 168..335)
WIN = 120                   # window of panels D-H (voxels)
FACE_W, FACE_Z = 60, 24     # window of panel I: x voxels, slices from the face
N_OPEN = 15                 # /open -open_distance of the compartment separation
PEEL = 6                    # /gobj_maskaimpeel_ow -peel_iter

# the four operators of panels D-G: (input export, IPL's export, operator, N, letter, IPL command)
STAGE_OPS = [("08_TRABRANK", "09_ERO3", "erosion", 3, "D", "/erosion 3"),
             ("10_ERORANK", "11_DIL3", "dilation", 3, "E", "/dilation 3"),
             ("11_DIL3", "12_CLOSE15", "close", 15, "F", "/close 15"),
             ("14_BBC", "15_OPEN15", "open", 15, "G", "/open 15")]

# the margin conventions of panel K (label, margin depth for N, pad mode of ipldt.ipl_ops._open_padded)
CONVENTIONS = [("edge-inclusive mirror, depth N + 2", lambda n: n + 2, "symmetric"),
               ("edge-inclusive mirror, depth N + 1", lambda n: n + 1, "symmetric"),
               ("edge-inclusive mirror, depth N + 3", lambda n: n + 3, "symmetric"),
               ("edge-inclusive mirror, depth N", lambda n: n, "symmetric"),
               ("edge-inclusive mirror, depth 2N", lambda n: 2 * n, "symmetric"),
               ("edge-exclusive mirror, depth N + 2", lambda n: n + 2, "reflect"),
               ("unpadded erosion, background margin", lambda n: n + 2, "open_boundary"),
               ("face slice copied, depth N + 2", lambda n: n + 2, "edge"),
               ("periodic margin, depth N + 2", lambda n: n + 2, "wrap"),
               ("all-object margin, depth N + 2", lambda n: n + 2, "object"),
               ("all-background margin, depth N + 2", lambda n: n + 2, "background")]
IPL_CONVENTION = CONVENTIONS[0][0]
NSCAN_CONVENTIONS = [CONVENTIONS[0][0], CONVENTIONS[6][0], CONVENTIONS[5][0], CONVENTIONS[7][0], CONVENTIONS[10][0]]
NSCAN_N = list(range(11, 19))

# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito (colour-blind safe, the palette of every figure of the paper)
C_BLUE, C_ORANGE, C_GREEN, C_VERM, C_PURPLE, C_SKY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"
C_INK, C_INK2, C_MUTED, C_GRID, C_AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C_FILL = "#b9b8b3"            # the input stage / the object
C_MIRROR = "#dfe8ef"          # the mirror image in the margin
C_SKY_LIGHT = "#bfe0f4"       # IPL's ball (B), the shaded ball (A)
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.5, "axes.edgecolor": C_AXIS, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.color": C_INK2, "ytick.color": C_INK2, "savefig.dpi": 300,
    "figure.dpi": 100, "pdf.fonttype": 42, "svg.fonttype": "none", "text.color": C_INK, "axes.labelcolor": C_INK2,
    "hatch.linewidth": 0.4, "hatch.color": "#7f8b94",
})

T0 = time.time()


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ------------------------------------------------------------------------------------------------ helpers
def aim(path):
    """Read an AIM; a path without the VMS ';n' suffix picks the highest version present."""
    cands = [path] if os.path.exists(path) else sorted(glob.glob(path + ";*"), key=lambda p: int(p.rsplit(";", 1)[1]))
    if not cands:
        raise FileNotFoundError(path)
    v = read_aim(cands[-1])
    return dict(data=np.asarray(v["data"]), dim=tuple(int(x) for x in v["dim"]), pos=tuple(int(x) for x in v["pos"]))


def p16(tag):
    return aim(os.path.join(P16_DIR, P16_BASE + tag + ".AIM"))


def setv(v):
    return np.asarray(v["data"]) != 0


def u8(m):
    return np.ascontiguousarray(m, dtype=np.uint8)


def chamfer(m):
    return ops.chamfer_dt_345(u8(m))


def count(v):
    return int(np.count_nonzero(v["data"]))


def mism(a, b):
    """Two volumes on any grids, pasted onto their union grid by global position:
    (differing voxels, a-only, b-only, grids identical)."""
    dim, pos = ops.union_grid(a, b)
    A = ops.on_grid(a, dim, pos) != 0
    B = ops.on_grid(b, dim, pos) != 0
    same = tuple(a["dim"]) == tuple(b["dim"]) and tuple(a["pos"]) == tuple(b["pos"])
    return int((A != B).sum()), int((A & ~B).sum()), int((B & ~A).sum()), bool(same)


def bvol(mask_bool, like):
    return dict(data=u8(mask_bool), dim=tuple(like["dim"]), pos=tuple(like["pos"]))


def crop_global(v, box):
    """v's data inside the global box (x0, x1, y0, y1, z0, z1), inclusive; zeros where v does not cover it."""
    x0, x1, y0, y1, z0, z1 = box
    dim = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
    return align_to(v, dim, (x0, y0, z0))


def choose_crop(change2d, size):
    """Top-left corner of the size x size window with the most changed voxels (deterministic)."""
    dens = ndi.uniform_filter(change2d.astype(np.float32), size=size, mode="constant")
    iy, ix = np.unravel_index(int(np.argmax(dens)), dens.shape)
    y0 = int(np.clip(iy - size // 2, 0, change2d.shape[0] - size))
    x0 = int(np.clip(ix - size // 2, 0, change2d.shape[1] - size))
    return y0, x0


def fmt(n):
    """An integer with comma thousands separators, as in the legends and the text."""
    return f"{int(n):,d}"


# ------------------------------------------------------------------------------------------------ compute
def compute_metric():
    """Panel A: the raw chamfer 3-4-5 distance to one voxel (synthetic), plane z = 0, and the N = 3 ball."""
    N = 3
    R = N + 3
    n = 2 * R + 1
    obj = np.ones((n, n, n), np.uint8)
    obj[R, R, R] = 0
    d = ops.chamfer_dt_345(obj)
    T = ops.metric11_threshold(N)
    plane = d[R].astype(int)
    say(f"A: N = {N}, threshold 3N + 2 = {T}; the ball holds {int((d < T).sum())} voxels, {int((plane < T).sum())} in the plane")
    return dict(N=N, R=R, threshold=T, plane=plane, ball_voxels=int((d < T).sum()), plane_voxels=int((plane < T).sum()),
                weights=(3, 4, 5))


def compute_ball():
    """Panels B, C: IPL's /dilation 15 of the isolated 5^3 block of the synthetic volume vs the five readings."""
    man = json.load(open(os.path.join(P18_DIR, "phantom18_manifest.json")))
    cl = man["cl"]
    px, py, pz = cl["pos"]
    D = cl["regions"]["d15"]["blocks"]["D"]
    M = 17
    box = (px + D["x"][0] - M, px + D["x"][1] + M, py + D["y"][0] - M, py + D["y"][1] + M,
           pz + D["z"][0] - M, pz + D["z"][1] + M)
    dil15 = aim(os.path.join(P18_DIR, "outputs", "X2420448_P18_DIL15.AIM"))
    ipl_w = crop_global(dil15, box) != 0                      # (z, y, x), 39^3
    w = ipl_w.shape
    bx, by, bz = (D[k][1] - D[k][0] + 1 for k in "xyz")
    blk = np.zeros(w, bool)
    blk[M:M + bz, M:M + by, M:M + bx] = True
    N = 15
    T = ops.metric11_threshold(N)
    pad = N + 2
    bp = np.pad(blk, pad)
    dc = chamfer(~bp)
    de = ndi.distance_transform_edt(~bp)
    sl = tuple(slice(pad, -pad) for _ in range(3))
    preds = [("raw < 3N + 1", (bp | (dc < T - 1))[sl]),
             ("raw < 3N + 2", (bp | (dc < T))[sl]),
             ("raw < 3N + 3", (bp | (dc < T + 1))[sl]),
             ("Euclidean r ≤ N", (bp | (de <= N))[sl]),
             ("Euclidean r ≤ N + 0.5", (bp | (de <= N + 0.5))[sl])]
    diffs = {lab: int((p != ipl_w).sum()) for lab, p in preds}
    zc = M + bz // 2
    say("B/C: IPL's dilation 15 of the block, voxels differing per reading in the 39^3 window:", diffs)
    assert diffs["raw < 3N + 2"] == 0, diffs
    return dict(N=N, threshold=T, box=box, plane_z=box[4] + zc, window_side=w[0], block=(bx, by, bz),
                ipl_plane=ipl_w[zc], blk_plane=blk[zc], pred_planes={lab: p[zc] for lab, p in preds},
                diffs=diffs, readings=[lab for lab, _ in preds], ipl_set_window=int(ipl_w.sum()),
                ipl_grid=(dil15["dim"], dil15["pos"]))


def compute_stages():
    """Panels D-G: the four operators on the patella's stage exports, whole-volume comparison + mid-slice crops."""
    out = []
    fns = {"erosion": ops.erosion, "dilation": ops.dilation, "close": ops.close, "open": ops.open_}
    for tin, tout, opname, N, letter, cmd in STAGE_OPS:
        inp, ipl = p16(tin), p16(tout)
        t = time.time()
        rep = fns[opname](inp, N)
        d, ao, bo, same = mism(rep, ipl)
        dim, pos = ops.union_grid(inp, ipl)
        A = ops.on_grid(inp, dim, pos) != 0
        B = ops.on_grid(ipl, dim, pos) != 0
        O = ops.on_grid(rep, dim, pos) != 0
        zc = Z_MID - pos[2]
        a2, b2, o2 = A[zc], B[zc], O[zc]
        y0, x0 = choose_crop(a2 != b2, WIN)
        win = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
        rec = dict(letter=letter, command=cmd, op=opname, N=N, threshold=ops.metric11_threshold(N),
                   mismatches=d, ours_only=ao, ipl_only=bo, same_grid=same, voxels_ipl=count(ipl), voxels_ours=count(rep),
                   voxels_in=count(inp), removed=int((A & ~B).sum()), added=int((B & ~A).sum()),
                   grid_in=[list(inp["dim"]), list(inp["pos"])], grid_out=[list(ipl["dim"]), list(ipl["pos"])],
                   grid_ours=[list(rep["dim"]), list(rep["pos"])], window_origin=[pos[0] + x0, pos[1] + y0, Z_MID],
                   window_differing=int((b2[win] != o2[win]).sum()), seconds=round(time.time() - t, 2))
        say(f"{letter} {cmd}: {d} differing (whole volume), IPL {rec['voxels_ipl']:,} ipldt {rec['voxels_ours']:,}, "
            f"removed {rec['removed']:,} added {rec['added']:,}, grid {inp['dim']}->{ipl['dim']}, {rec['seconds']} s")
        out.append(dict(numbers=rec, inp=a2[win].copy(), ipl=b2[win].copy(), ours=o2[win].copy()))
    return out


def compute_peel():
    """Panel H: /gobj_maskaimpeel_ow -peel_iter 6 of the trabecular candidate (stage 02) -> stage 03."""
    s00, s02, s03 = p16("00_ALL"), p16("02_TRAB0"), p16("03_PEEL6")
    peel = ops.peel_gobj_render(s00, PEEL)
    ours = ops.gobj_maskaimpeel_ow(s02, s00, PEEL)
    d, ao, bo, same = mism(ours, s03)
    dim, pos = ops.union_grid(s02, s03)
    dim, pos = ops.union_grid(dict(dim=dim, pos=pos), s00)
    G = ops.on_grid(s00, dim, pos) != 0
    P = ops.on_grid(peel, dim, pos) != 0
    A = ops.on_grid(s02, dim, pos) != 0
    B = ops.on_grid(s03, dim, pos) != 0
    O = ops.on_grid(ours, dim, pos) != 0
    zc = Z_MID - pos[2]
    y0, x0 = choose_crop(A[zc] != B[zc], WIN)
    win = (slice(y0, y0 + WIN), slice(x0, x0 + WIN))
    rec = dict(peel_iter=PEEL, mismatches=d, ours_only=ao, ipl_only=bo, same_grid=same, voxels_ipl=count(s03),
               voxels_ours=count(ours), voxels_in=count(s02), removed=int((A & ~B).sum()), periosteal_voxels=count(s00),
               peeled_voxels=count(peel), band_voxels=count(s00) - count(peel), window_origin=[pos[0] + x0, pos[1] + y0, Z_MID],
               window_differing=int((B[zc][win] != O[zc][win]).sum()))
    say(f"H peel {PEEL}: {d} differing; periosteal {rec['periosteal_voxels']:,} -> peeled {rec['peeled_voxels']:,}; "
        f"stage 02 {rec['voxels_in']:,} -> 03 {rec['voxels_ipl']:,} (removed {rec['removed']:,})")
    return dict(numbers=rec, peri=G[zc][win].copy(), peeled=P[zc][win].copy(), inp=A[zc][win].copy(),
                ipl=B[zc][win].copy(), ours=O[zc][win].copy())


def compute_face():
    """Panel I: /erosion 3 at the first slices of the volume -- the open boundary vs a background-padded reading."""
    s08, s09 = p16("08_TRABRANK"), p16("09_ERO3")
    assert s08["dim"] == s09["dim"] and s08["pos"] == s09["pos"]
    m8, ipl9 = setv(s08), setv(s09)
    T = ops.metric11_threshold(3)
    e_rule = m8 & (chamfer(m8) >= T)
    mz = np.pad(m8, ((1, 1), (0, 0), (0, 0)))
    e_zpad = m8 & (chamfer(mz)[1:-1] >= T)
    per_slice = (ipl9 & ~e_zpad).sum(axis=(1, 2))
    nz = np.nonzero(per_slice)[0]
    rec = dict(N=3, threshold=T, rule_mismatches=int((e_rule != ipl9).sum()), zpad_mismatches=int((e_zpad != ipl9).sum()),
               zpad_ipl_only=int((ipl9 & ~e_zpad).sum()), zpad_ours_only=int((e_zpad & ~ipl9).sum()),
               slices_ipl_only={int(z + s08["pos"][2]): int(per_slice[z]) for z in nz},
               face_slice_kept_by_ipl=int(ipl9[0].sum()), face_slice_input=int(m8[0].sum()),
               face_slice_kept_zpad=int(e_zpad[0].sum()), voxels_ipl=count(s09), voxels_in=count(s08))
    # the section: the row of the first slice with the most kept voxels, then the x window with the most change
    y_c = int(np.argmax(ipl9[0].sum(axis=1)))
    sec = dict(inp=m8[:FACE_Z, y_c, :], ipl=ipl9[:FACE_Z, y_c, :], zpad=e_zpad[:FACE_Z, y_c, :], rule=e_rule[:FACE_Z, y_c, :])
    score = ((sec["inp"] != sec["ipl"]).sum(axis=0) + (sec["ipl"] & ~sec["zpad"]).sum(axis=0)).astype(np.float32)
    dens = ndi.uniform_filter1d(score, FACE_W, mode="constant")
    x0 = int(np.clip(int(np.argmax(dens)) - FACE_W // 2, 0, sec["inp"].shape[1] - FACE_W))
    rec.update(section_y=y_c + s08["pos"][1], section_x0=x0 + s08["pos"][0], section_z0=s08["pos"][2],
               section_w=FACE_W, section_slices=FACE_Z)
    say(f"I erosion at the face: rule {rec['rule_mismatches']} differing; background-padded reading {rec['zpad_mismatches']} "
        f"({rec['zpad_ipl_only']:,} IPL-only in slices {sorted(rec['slices_ipl_only'])}); face slice keeps "
        f"{rec['face_slice_kept_by_ipl']:,} of {rec['face_slice_input']:,} (padded: {rec['face_slice_kept_zpad']})")
    return dict(numbers=rec, **{k: v[:, x0:x0 + FACE_W].copy() for k, v in sec.items()})


def point_ball(N):
    """The chamfer ball raw < 3N + 2 around one voxel as a bool cube, and its half-size."""
    h = N + 1
    s = 2 * h + 1
    m = np.ones((s, s, s), np.uint8)
    m[h, h, h] = 0
    return ops.chamfer_dt_345(m) < ops.metric11_threshold(N), h


def compute_mirror():
    """Panel J: the /open buffer of the second patella scan -- mirror margin, survivors, extra survivors, restored voxels."""
    inp = aim(os.path.join(P17_DIR, P17_BASE + "14_BBC.AIM"))
    opn = aim(os.path.join(P17_DIR, P17_BASE + "15_OPEN15.AIM"))
    N = N_OPEN
    T = ops.metric11_threshold(N)
    MARGIN = N + 2
    M = setv(inp)
    mech = ops.open_(inp, N)
    old = bvol(ops._open_padded(M, N, MARGIN, "open_boundary"), inp)
    r_mech, r_old = mism(mech, opn), mism(old, opn)
    say(f"J: /open 15 of patella 2: mirrored margin {r_mech[:3]} vs IPL; unpadded erosion + background margin {r_old[:3]}")
    assert r_mech[0] == 0
    restored_mask = setv(mech) & ~setv(old)
    zz, yy, xx = np.nonzero(restored_mask)
    restored = [dict(local=(int(x), int(y), int(z))) for z, y, x in zip(zz, yy, xx)]

    Mp = np.pad(M, MARGIN, mode="symmetric")
    dt = chamfer(Mp)
    surv = Mp & (dt >= T)
    nz, ny, nx = M.shape

    def partner_index(p, n):
        q = p - MARGIN
        if q < 0:
            return MARGIN + (-q - 1)
        if q >= n:
            return MARGIN + (n - 1) - (q - n)
        return p

    zi = np.array([partner_index(p, nz) for p in range(Mp.shape[0])])
    yi = np.array([partner_index(p, ny) for p in range(Mp.shape[1])])
    xi = np.array([partner_index(p, nx) for p in range(Mp.shape[2])])
    surv_partner = surv[np.ix_(zi, yi, xi)]
    inside = np.zeros(Mp.shape, dtype=bool)
    inside[MARGIN:-MARGIN, MARGIN:-MARGIN, MARGIN:-MARGIN] = True
    extra = surv & ~surv_partner & ~inside
    ball, h = point_ball(N)
    responsible = []
    for z, y, x in zip(*np.nonzero(extra)):
        lz, ly, lx = int(z) - MARGIN, int(y) - MARGIN, int(x) - MARGIN
        hits = []
        for r in restored:
            rx, ry, rz = r["local"]
            dz, dy, dx = rz - lz, ry - ly, rx - lx
            if max(abs(dz), abs(dy), abs(dx)) <= h and ball[dz + h, dy + h, dx + h]:
                hits.append(list(r["local"]))
        if hits:
            depth = lz - (nz - 1) if lz >= nz else None
            responsible.append(dict(local=(lx, ly, lz), raw_dt=int(dt[z, y, x]), depth=depth,
                                    partner_local_z=((nz - 1) - (depth - 1)) if depth else None, restores=hits))
    say(f"J: {int(extra.sum()):,} extra survivors in the margin (partner eroded); {len(responsible)} reach the "
        f"{len(restored)} restored voxels of the last slice: {[r['local'] for r in restored]}")
    # the section through the restored voxels
    ys = [r["local"][1] for r in restored] + [s["local"][1] for s in responsible]
    y_sec = int(np.median(ys))
    xs = [r["local"][0] for r in restored] + [s["local"][0] for s in responsible]
    x0, x1 = min(xs) - 16, max(xs) + 16
    z1 = nz - 1 + MARGIN
    z0 = z1 - 44
    cat = np.zeros((z1 - z0 + 1, x1 - x0 + 1), dtype=np.int8)
    for iz, lz in enumerate(range(z0, z1 + 1)):
        for ix, lx in enumerate(range(x0, x1 + 1)):
            pz, py, px = lz + MARGIN, y_sec + MARGIN, lx + MARGIN
            if not Mp[pz, py, px]:
                continue
            if inside[pz, py, px]:
                cat[iz, ix] = 3 if surv[pz, py, px] else 1
            else:
                cat[iz, ix] = 5 if extra[pz, py, px] else (4 if surv[pz, py, px] else 2)
    rec = dict(N=N, threshold=T, margin=MARGIN, input_grid=[list(inp["dim"]), list(inp["pos"])], input_voxels=count(inp),
               ipl_open_voxels=count(opn), mirror_vs_ipl=list(r_mech[:3]), unpadded_vs_ipl=list(r_old[:3]),
               margin_survivors=int((surv & ~inside).sum()), extra_survivors=int(extra.sum()),
               restored=[r["local"] for r in restored], responsible=responsible, section_y=y_sec,
               section_x=[x0, x1], section_z=[z0, z1], last_slice=nz - 1)
    return dict(numbers=rec, cat=cat, ball=ball, h=h)


def compute_conventions():
    """Panels K, L: every margin convention against IPL's /open N of the second patella scan, N = 11 .. 18."""
    inp = aim(os.path.join(P17_DIR, P17_BASE + "14_BBC.AIM"))
    M = setv(inp)
    ipl = {N: aim(os.path.join(P17_DIR, P17_BASE + "15_OPEN15.AIM")) if N == N_OPEN
           else aim(os.path.join(P20_DIR, P20_BASE + f"OPEN{N}.AIM")) for N in NSCAN_N}
    rows = {}
    for N in NSCAN_N:
        convs = CONVENTIONS if N == N_OPEN else [c for c in CONVENTIONS if c[0] in NSCAN_CONVENTIONS]
        for lab, margin_of, mode in convs:
            t = time.time()
            r = bvol(ops._open_padded(M, N, margin_of(N), mode), inp)
            d, ao, bo, _ = mism(r, ipl[N])
            rows[(N, lab)] = dict(N=N, label=lab, margin=margin_of(N), mode=mode, mismatches=d, ours_only=ao, ipl_only=bo,
                                  ipl_voxels=count(ipl[N]), seconds=round(time.time() - t, 2))
            say(f"K/L N = {N:2d} {lab:40s} {d:>9,} ({ao:,} / {bo:,}) {rows[(N, lab)]['seconds']} s")
    assert all(rows[(N, IPL_CONVENTION)]["mismatches"] == 0 for N in NSCAN_N)
    return dict(input_grid=[list(inp["dim"]), list(inp["pos"])], input_voxels=count(inp),
                rows=[dict(v) for v in rows.values()])


# ------------------------------------------------------------------------------------------------ cache
def save_cache(R):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    arrays = dict(A_plane=R["A"]["plane"], B_ipl=R["B"]["ipl_plane"], B_blk=R["B"]["blk_plane"],
                  H_peri=R["H"]["peri"], H_peeled=R["H"]["peeled"], H_inp=R["H"]["inp"], H_ipl=R["H"]["ipl"], H_ours=R["H"]["ours"],
                  I_inp=R["I"]["inp"], I_ipl=R["I"]["ipl"], I_zpad=R["I"]["zpad"], I_rule=R["I"]["rule"],
                  J_cat=R["J"]["cat"], J_ball=R["J"]["ball"])
    for i, lab in enumerate(R["B"]["readings"]):
        arrays[f"B_pred_{i}"] = R["B"]["pred_planes"][lab]
    for s in R["stages"]:
        L = s["numbers"]["letter"]
        arrays[f"{L}_inp"], arrays[f"{L}_ipl"], arrays[f"{L}_ours"] = s["inp"], s["ipl"], s["ours"]
    np.savez_compressed(CACHE + ".npz", **arrays)
    nums = dict(A={k: v for k, v in R["A"].items() if k != "plane"},
                B={k: v for k, v in R["B"].items() if k not in ("ipl_plane", "blk_plane", "pred_planes")},
                stages=[s["numbers"] for s in R["stages"]], H=R["H"]["numbers"], I=R["I"]["numbers"],
                J=dict(R["J"]["numbers"], h=R["J"]["h"]), K=R["K"])
    json.dump(nums, open(CACHE + ".json", "w"), indent=1)


def load_cache():
    z = np.load(CACHE + ".npz")
    nums = json.load(open(CACHE + ".json"))
    R = dict(A=dict(nums["A"], plane=z["A_plane"]),
             B=dict(nums["B"], ipl_plane=z["B_ipl"], blk_plane=z["B_blk"],
                    pred_planes={lab: z[f"B_pred_{i}"] for i, lab in enumerate(nums["B"]["readings"])}),
             stages=[dict(numbers=s, inp=z[f"{s['letter']}_inp"], ipl=z[f"{s['letter']}_ipl"], ours=z[f"{s['letter']}_ours"])
                     for s in nums["stages"]],
             H=dict(numbers=nums["H"], peri=z["H_peri"], peeled=z["H_peeled"], inp=z["H_inp"], ipl=z["H_ipl"], ours=z["H_ours"]),
             I=dict(numbers=nums["I"], inp=z["I_inp"], ipl=z["I_ipl"], zpad=z["I_zpad"], rule=z["I_rule"]),
             J=dict(numbers=nums["J"], cat=z["J_cat"], ball=z["J_ball"], h=nums["J"]["h"]),
             K=nums["K"])
    return R


# ------------------------------------------------------------------------------------------------ drawing
FIG_W = 180.0                        # mm; the height is derived from the layout in draw()
FIG_H = 240.0
TITLE_H = 9.0                        # letter + title + two-line description above each panel
NOTE_DY = 2.9                        # line pitch of the note lines under a panel


def ax_mm(fig, x, y_top, w, h):
    """Axes at (x, y_top) mm from the top-left corner of the figure, w x h mm."""
    return fig.add_axes([x / FIG_W, 1 - (y_top + h) / FIG_H, w / FIG_W, h / FIG_H])


def ftext(fig, x, y_top, s, **kw):
    """fig.text at (x, y_top) in mm from the top-left corner."""
    kw.setdefault("va", "top")
    return fig.text(x / FIG_W, 1 - y_top / FIG_H, s, **kw)


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def head(fig, x, yt, w, letter, title, desc, checks):
    """The panel head: bold letter upper-left, bold title beside it, a two-line description under them."""
    ftext(fig, x, yt, letter, fontsize=9, fontweight="bold", ha="left")
    t1 = ftext(fig, x + 3.4, yt + 0.25, title, fontsize=7.5, fontweight="bold", ha="left")
    t2 = ftext(fig, x, yt + 3.9, desc, fontsize=7, color=C_INK2, ha="left", linespacing=1.15)
    checks += [(t1, x, x + w, title), (t2, x, x + w, desc)]


def notes(fig, x, yt, w, lines, checks):
    """Note lines under a panel: (text, bold?, colour)."""
    for i, (s, bold, col) in enumerate(lines):
        t = ftext(fig, x, yt + i * NOTE_DY, s, fontsize=6.6, fontweight="bold" if bold else "normal", color=col, ha="left")
        checks.append((t, x, x + w, s))


def frame(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(C_AXIS)
        s.set_linewidth(0.4)


def tick_style(ax):
    ax.tick_params(labelsize=6.5, length=1.8, pad=1.2, width=0.4)
    for s in ax.spines.values():
        s.set_edgecolor(C_AXIS)
        s.set_linewidth(0.4)


def legend_axes(fig, x, yt, w, h, handles, **kw):
    axl = ax_mm(fig, x, yt, w, h)
    axl.axis("off")
    kw.setdefault("frameon", False)
    kw.setdefault("borderaxespad", 0)
    kw.setdefault("handletextpad", 0.5)
    axl.legend(handles=handles, **kw)
    return axl


def check_text_widths(fig, items):
    """Warn when a text runs past the horizontal span it was given: items = [(Text, x0_mm, x1_mm, label)]."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = 0
    for t, x0, x1, label in items:
        bb = t.get_window_extent(renderer=r)
        lo, hi = bb.x0 / fig.dpi * 25.4, bb.x1 / fig.dpi * 25.4
        if lo < x0 - 0.3 or hi > x1 + 0.3:
            bad += 1
            say(f"  WARNING text overflows its {x1 - x0:.1f} mm span by {max(x0 - lo, hi - x1):.1f} mm: {label!r}")
    say(f"text-width check: {len(items)} texts, {bad} overflow")
    return bad


def draw(R):
    global FIG_H
    # ------------------------------------------------------------------ the vertical layout, in mm from the top
    y1 = 0.8                                     # row 1: A, B, C
    img1 = 36.0
    row1_end = y1 + TITLE_H + img1 + 6.0 + 8.6
    y2 = row1_end + 2.0                          # row 2: D-G
    ncol, x_left, gap = 4, 2.0, 4.4
    pw = (FIG_W - 2 * x_left - (ncol - 1) * gap) / ncol
    yn2 = y2 + TITLE_H + pw + 1.0
    y_leg2 = yn2 + 3 * NOTE_DY + 0.4
    row2_end = y_leg2 + 3.2
    y3 = row2_end + 2.2                          # row 3: H, I, J
    xH, wH = 2.0, pw
    xI, wI = xH + wH + 4.8, 60.0
    xJ = xI + wI + 4.8
    wJ = FIG_W - 2.0 - xJ
    ynH = y3 + TITLE_H + wH + 1.0
    rowH_end = ynH + 3 * NOTE_DY + 0.2 + 5.6
    imgI_w = wI - 7.0
    imgI_h = imgI_w * FACE_Z / FACE_W
    ynI = y3 + TITLE_H + imgI_h + 8.6
    rowI_end = ynI + 4 * NOTE_DY + 0.2 + 5.6
    imgJ_h = 31.0
    catJ = R["J"]["cat"]
    imgJ_w = imgJ_h * catJ.shape[1] / catJ.shape[0]
    ynJ = y3 + TITLE_H + imgJ_h + 6.0
    rowJ_end = ynJ + 3 * NOTE_DY
    y4 = max(rowH_end, rowI_end, rowJ_end) + 2.2   # row 4: K, L
    xK, wK = 2.0, 80.0
    xL = xK + wK + 5.0
    wL = FIG_W - 2.0 - xL
    axes4_h = 28.0
    FIG_H = y4 + TITLE_H + 1.0 + axes4_h + 7.6
    say(f"layout: rows at {y1:.1f} / {y2:.1f} / {y3:.1f} / {y4:.1f} mm, figure {FIG_W:.0f} x {FIG_H:.1f} mm")

    fig = plt.figure(figsize=(FIG_W / 25.4, FIG_H / 25.4))
    checks = []
    P = {}                                   # every number printed, for the sidecar

    # ================================================================= row 1: A the metric, B the ball, C the readings
    xA, xB, xC, w1 = 2.0, 62.0, 122.0, 56.0
    A = R["A"]
    head(fig, xA, y1, w1, "A", "Chamfer metric 3-4-5",
         "raw distance from one voxel; a step costs\n3 (face), 4 (edge) or 5 (corner, out of plane)", checks)
    ax = ax_mm(fig, xA + 6.5, y1 + TITLE_H, img1, img1)
    plane, Rr, T = A["plane"], A["R"], A["threshold"]
    n = plane.shape[0]
    inside = plane < T
    rgb = np.ones(plane.shape + (3,))
    rgb[inside] = hex_rgb(C_SKY_LIGHT)
    ax.imshow(rgb, extent=(-Rr - 0.5, Rr + 0.5, Rr + 0.5, -Rr - 0.5), interpolation="nearest")
    for iy in range(n):
        for ix in range(n):
            centre = (iy, ix) == (Rr, Rr)
            ax.text(ix - Rr, iy - Rr, str(plane[iy, ix]), ha="center", va="center", fontsize=5.4,
                    color=C_INK if inside[iy, ix] else C_MUTED, fontweight="bold" if centre else "normal")
    ax.add_patch(patches.Rectangle((0.5, -0.5), 1, 1, fill=False, ec=C_BLUE, lw=1.0))        # face step: 3
    ax.add_patch(patches.Rectangle((0.5, -1.5), 1, 1, fill=False, ec=C_GREEN, lw=1.0))       # edge step: 4
    ax.set_xticks([-6, -3, 0, 3, 6])
    ax.set_yticks([-6, -3, 0, 3, 6])
    tick_style(ax)
    ax.set_xlabel("x offset from the voxel (voxels)", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel("y offset (voxels)", fontsize=6.5, labelpad=1.5)
    yA = y1 + TITLE_H + img1 + 6.0
    notes(fig, xA, yA, w1, [(f"shaded: raw < 3N + 2 = {T} (N = {A['N']}): the ball of", False, C_INK2),
                            (f"every metric-11 command ({A['ball_voxels']} voxels; {A['plane_voxels']} here)", False, C_INK2)], checks)
    legend_axes(fig, xA + 3, yA + 2 * NOTE_DY, 45, 3,
                [Line2D([], [], marker="s", ms=5, mfc="none", mec=C_BLUE, mew=1.0, ls="none", label="face step: 3"),
                 Line2D([], [], marker="s", ms=5, mfc="none", mec=C_GREEN, mew=1.0, ls="none", label="edge step: 4")],
                loc="center left", ncol=2, fontsize=6.5, handlelength=1.0, columnspacing=1.5)
    P["A"] = dict(N=A["N"], threshold=T, ball_voxels=A["ball_voxels"], plane_voxels=A["plane_voxels"], weights=list(A["weights"]))

    # ---- B: IPL's dilation of the isolated block
    B = R["B"]
    bx, by, bz = B["block"]
    head(fig, xB, y1, w1, "B", "The ball IPL draws",
         f"{bx}×{by}×{bz} block in a synthetic volume, dilated\nby {B['N']} with IPL; the plane through its center", checks)
    ax = ax_mm(fig, xB + 6.5, y1 + TITLE_H, img1, img1)
    box = B["box"]
    ipl_p, blk_p = B["ipl_plane"], B["blk_plane"]
    rgb = np.ones(ipl_p.shape + (3,))
    rgb[ipl_p] = hex_rgb(C_SKY_LIGHT)
    rgb[blk_p] = hex_rgb(C_BLUE)
    ext = (box[0] - 0.5, box[1] + 0.5, box[3] + 0.5, box[2] - 0.5)
    ax.imshow(rgb, extent=ext, interpolation="nearest")
    xs = np.arange(box[0], box[1] + 1)
    ys = np.arange(box[2], box[3] + 1)
    X, Y = np.meshgrid(xs, ys)
    outlines = [("raw < 3N + 2", C_BLUE, "-", 0.9), ("Euclidean r ≤ N", C_GREEN, (0, (2.2, 1.4)), 0.9),
                ("raw < 3N + 3", C_VERM, (0, (0.8, 1.2)), 1.0)]
    for lab, col, ls, lw in outlines:
        ax.contour(X, Y, B["pred_planes"][lab].astype(float), levels=[0.5], colors=[col], linestyles=[ls], linewidths=lw)
    tick_style(ax)
    ax.set_xlabel("x (voxels)", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel("y (voxels)", fontsize=6.5, labelpad=1.5)
    legend_axes(fig, xB, yA - 0.6, w1, 8.6,
                [patches.Patch(fc=C_SKY_LIGHT, ec="none", label="IPL's dilation"),
                 Line2D([], [], color=C_BLUE, lw=0.9, label="raw < 3N + 2 (ipldt)"),
                 patches.Patch(fc=C_BLUE, ec="none", label="the block"),
                 Line2D([], [], color=C_GREEN, lw=0.9, ls=(0, (2.2, 1.4)), label="Euclidean r ≤ N"),
                 Line2D([], [], color=C_VERM, lw=1.0, ls=(0, (0.8, 1.2)), label="raw < 3N + 3")],
                loc="upper left", ncol=2, fontsize=6.4, handlelength=1.4, columnspacing=1.2, labelspacing=0.4)
    P["B"] = dict(N=B["N"], threshold=B["threshold"], block=list(B["block"]), window_side=B["window_side"], plane_z=B["plane_z"],
                  diffs=B["diffs"], ipl_grid=B["ipl_grid"])

    # ---- C: the readings
    head(fig, xC, y1, w1, "C", "Threshold readings",
         f"voxels differing from IPL's ball of B in the\n{B['window_side']}³ window, one reading at a time", checks)
    ax = ax_mm(fig, xC + 24.0, y1 + TITLE_H + 1.0, 30.0, img1 - 2.0)
    order = B["readings"]
    vals = [B["diffs"][k] for k in order]
    ypos = np.arange(len(order))[::-1]
    cols = {"raw < 3N + 2": C_BLUE, "Euclidean r ≤ N": C_GREEN, "raw < 3N + 3": C_VERM}
    ax.barh(ypos, vals, color=[cols.get(k, C_MUTED) for k in order], height=0.62, lw=0)
    ax.set_yticks(ypos)
    ax.set_yticklabels([("IPL: " if k == "raw < 3N + 2" else "") + k for k in order], fontsize=6.5)
    ax.set_xlim(0, max(vals) * 1.42)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xticklabels(["0", "1,000", "2,000", "3,000"])
    tick_style(ax)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color=C_GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("voxels differing from IPL", fontsize=6.5, labelpad=1.5)
    for y, v, k in zip(ypos, vals, order):
        ax.text(v + max(vals) * 0.03, y, "0 (identical)" if v == 0 else fmt(v), va="center", fontsize=6.5,
                fontweight="bold" if v == 0 else "normal", color=C_INK)
    P["C"] = dict(readings=order, differing=vals)

    # ================================================================= row 2: D-G the four operators
    DESC = {"erosion": "object voxels at raw distance\n≥ 3N + 2 from background: kept",
            "dilation": "background voxels within raw\n< 3N + 2 of the object: added",
            "close": "dilation then erosion, on a\nbackground margin of N + 2 voxels",
            "open": "erosion then dilation, on a\nmirrored margin of N + 2 (J–L)"}
    TITLES = {"erosion": "Erosion", "dilation": "Dilation", "close": "Closing", "open": "Opening"}
    P["stages"] = []
    for k, s in enumerate(R["stages"]):
        num = s["numbers"]
        x = x_left + k * (pw + gap)
        head(fig, x, y2, pw, num["letter"], f"{TITLES[num['op']]}: {num['command']}", DESC[num["op"]], checks)
        ax = ax_mm(fig, x, y2 + TITLE_H, pw, pw)
        inp, ipl, ours = s["inp"], s["ipl"], s["ours"]
        rgb = np.ones(inp.shape + (3,))
        rgb[inp] = hex_rgb(C_FILL)
        rgb[inp & ~ipl] = hex_rgb(C_ORANGE)
        rgb[ipl & ~inp] = hex_rgb(C_GREEN)
        rgb[ipl != ours] = hex_rgb(C_VERM)
        H, W = inp.shape
        ax.imshow(rgb, interpolation="nearest", origin="upper", extent=(-0.5, W - 0.5, H - 0.5, -0.5))
        ax.contour(ours.astype(float), levels=[0.5], colors=[C_BLUE], linewidths=0.45)
        frame(ax)
        if k == 0:
            nb = 2.0 / VOX_MM
            xb, yb = W * 0.05, H * 0.93
            ax.plot([xb, xb + nb], [yb, yb], color=C_INK, lw=1.2, solid_capstyle="butt")
            ax.text(xb + nb / 2, yb - H * 0.03, "2 mm", ha="center", va="bottom", fontsize=6.5, color=C_INK)
        d = num["mismatches"]
        lines = [(f"{fmt(d)} voxels differ (whole volume)", True, C_INK if d == 0 else C_VERM)]
        if num["removed"] and not num["added"]:
            lines.append((f"IPL removes {fmt(num['removed'])} voxels", False, C_INK2))
        elif num["added"] and not num["removed"]:
            lines.append((f"IPL adds {fmt(num['added'])} voxels", False, C_INK2))
        else:
            lines.append((f"removes {fmt(num['removed'])}, adds {fmt(num['added'])}", False, C_INK2))
        lines.append((f"{fmt(num['voxels_ipl'])} voxels in each result", False, C_INK2))
        notes(fig, x, yn2, pw, lines, checks)
        P["stages"].append(num)
    # legend of D-H
    legend_axes(fig, x_left, y_leg2, FIG_W - 2 * x_left, 3.2,
                [patches.Patch(fc=C_FILL, ec="none", label="input stage (IPL's export)"),
                 patches.Patch(fc=C_ORANGE, ec="none", label="removed by IPL's command"),
                 patches.Patch(fc=C_GREEN, ec="none", label="added by IPL's command"),
                 Line2D([], [], color=C_BLUE, lw=0.9, label="ipldt's result (outline)"),
                 patches.Patch(fc=C_VERM, ec="none", label="voxels that differ (none)")],
                loc="center", ncol=5, fontsize=6.6, handlelength=1.5, columnspacing=1.5)

    # ================================================================= row 3: H the peel, I the face, J the mirror
    # ---- H
    Hn = R["H"]["numbers"]
    head(fig, xH, y3, wH, "H", f"Peel: peel_iter {Hn['peel_iter']}",
         f"the periosteal region eroded by\n{Hn['peel_iter']} voxels in-plane, as a mask", checks)
    ax = ax_mm(fig, xH, y3 + TITLE_H, wH, wH)
    inp, ipl, ours, peri, peeled = R["H"]["inp"], R["H"]["ipl"], R["H"]["ours"], R["H"]["peri"], R["H"]["peeled"]
    rgb = np.ones(inp.shape + (3,))
    rgb[inp] = hex_rgb(C_FILL)
    rgb[inp & ~ipl] = hex_rgb(C_ORANGE)
    rgb[ipl != ours] = hex_rgb(C_VERM)
    Hh, Ww = inp.shape
    ax.imshow(rgb, interpolation="nearest", origin="upper", extent=(-0.5, Ww - 0.5, Hh - 0.5, -0.5))
    ax.contour(peri.astype(float), levels=[0.5], colors=[C_INK2], linewidths=0.55)
    ax.contour(peeled.astype(float), levels=[0.5], colors=[C_PURPLE], linewidths=0.7, linestyles=[(0, (2.0, 1.2))])
    ax.contour(ours.astype(float), levels=[0.5], colors=[C_BLUE], linewidths=0.45)
    frame(ax)
    notes(fig, xH, ynH, wH, [(f"{fmt(Hn['mismatches'])} voxels differ (whole volume)", True, C_INK if Hn["mismatches"] == 0 else C_VERM),
                             (f"IPL removes {fmt(Hn['removed'])} voxels", False, C_INK2),
                             (f"{fmt(Hn['voxels_ipl'])} voxels in each result", False, C_INK2)], checks)
    legend_axes(fig, xH, ynH + 3 * NOTE_DY + 0.2, wH, 5.6,
                [Line2D([], [], color=C_INK2, lw=0.8, label="periosteal contour"),
                 Line2D([], [], color=C_PURPLE, lw=0.8, ls=(0, (2.0, 1.2)), label=f"peeled region ({fmt(Hn['peeled_voxels'])} voxels)")],
                loc="upper left", ncol=1, fontsize=6.5, handlelength=1.5, labelspacing=0.3)
    P["H"] = Hn

    # ---- I: the erosion at the face of the volume
    In = R["I"]["numbers"]
    head(fig, xI, y3, wI, "I", "Erosion at a face of the volume",
         "nothing outside the volume counts as background:\nobject voxels on the face slices are kept", checks)
    sec_in, sec_ipl, sec_zpad, sec_rule = R["I"]["inp"], R["I"]["ipl"], R["I"]["zpad"], R["I"]["rule"]
    nzs, nxs = sec_in.shape
    ax = ax_mm(fig, xI + 7.0, y3 + TITLE_H + 1.6, imgI_w, imgI_h - 1.6)      # clear of the second subtitle line
    rgb = np.ones(sec_in.shape + (3,))
    rgb[sec_in] = hex_rgb(C_FILL)
    rgb[sec_in & ~sec_ipl] = hex_rgb(C_ORANGE)
    rgb[sec_ipl & ~sec_zpad] = hex_rgb(C_VERM)
    gx0, gz0 = In["section_x0"], In["section_z0"]
    ext = (gx0 - 0.5, gx0 + nxs - 0.5, gz0 - 0.5, gz0 + nzs - 0.5)
    ax.imshow(rgb, interpolation="nearest", origin="lower", extent=ext, aspect="equal")
    Xs, Zs = np.meshgrid(np.arange(gx0, gx0 + nxs), np.arange(gz0, gz0 + nzs))
    ax.contour(Xs, Zs, sec_rule.astype(float), levels=[0.5], colors=[C_BLUE], linewidths=0.45)
    ax.axhline(gz0 - 0.5, color=C_INK, lw=1.6)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_yticks([gz0, gz0 + 8, gz0 + 16, gz0 + nzs - 1])
    ax.set_xticks([gx0, gx0 + 20, gx0 + 40, gx0 + nxs - 1])
    tick_style(ax)
    ax.set_xlabel(f"x (voxels), row y = {In['section_y']}\nthick line: the volume's edge (slice {gz0})", fontsize=6.5, labelpad=1.5,
                  linespacing=1.15)
    ax.set_ylabel("z (slice)", fontsize=6.5, labelpad=1.5)
    notes(fig, xI, ynI, wI, [(f"{fmt(In['rule_mismatches'])} voxels differ (whole volume)", True, C_INK if In["rule_mismatches"] == 0 else C_VERM),
                             (f"first slice: IPL keeps {fmt(In['face_slice_kept_by_ipl'])} of {fmt(In['face_slice_input'])} voxels;", False, C_INK2),
                             (f"a background-padded volume would keep {In['face_slice_kept_zpad']} and", False, C_INK2),
                             (f"erode {fmt(In['zpad_ipl_only'])} more in the 3 slices at each z face", False, C_INK2)], checks)
    legend_axes(fig, xI, ynI + 4 * NOTE_DY + 0.2, wI, 5.6,
                [patches.Patch(fc=C_FILL, ec="none", label="kept by IPL"),
                 patches.Patch(fc=C_ORANGE, ec="none", label="eroded by IPL"),
                 patches.Patch(fc=C_VERM, ec="none", label="kept by IPL, eroded under padding")],
                loc="upper left", ncol=2, fontsize=6.4, handlelength=1.3, columnspacing=1.0, labelspacing=0.3)
    P["I"] = In

    # ---- J: the mirrored margin of /open
    Jn = R["J"]["numbers"]
    MARGIN = Jn["margin"]
    head(fig, xJ, y3, wJ, "J", "Opening: the mirrored margin",
         f"the erosion's buffer mirrors the volume {MARGIN} voxels\nbeyond each face; every survivor seeds the dilation", checks)
    cat = R["J"]["cat"]
    (x0, x1), (z0, z1) = Jn["section_x"], Jn["section_z"]
    nzs, nxs = cat.shape
    ax = ax_mm(fig, xJ + 7.0, y3 + TITLE_H + 1.6, imgJ_w, imgJ_h - 1.6)      # clear of the second subtitle line
    lut = {0: "#ffffff", 1: C_FILL, 2: C_MIRROR, 3: C_BLUE, 4: C_SKY, 5: C_VERM}
    rgb = np.ones(cat.shape + (3,))
    for k, col in lut.items():
        rgb[cat == k] = hex_rgb(col)
    ext = (x0 - 0.5, x1 + 0.5, z0 - 0.5, z1 + 0.5)
    ax.imshow(rgb, interpolation="nearest", origin="lower", extent=ext, aspect="equal")
    last = Jn["last_slice"]
    ax.add_patch(patches.Rectangle((x0 - 0.5, last + 0.5), nxs, MARGIN, facecolor="none", edgecolor="#7f8b94", hatch="////", lw=0))
    ax.axhline(last + 0.5, color=C_INK, lw=1.4)
    ball, h = R["J"]["ball"], R["J"]["h"]
    for s in Jn["responsible"]:
        lx, ly, lz = s["local"]
        dy = Jn["section_y"] - ly
        in_sec = dy == 0
        ax.plot(lx, lz, marker="o", ms=4.0, mfc=C_VERM if in_sec else "white", mec=C_INK, mew=0.7, ls="none", zorder=5)
        if abs(dy) <= h:
            sec = ball[:, dy + h, :]
            ax.contour(np.arange(lx - h, lx + h + 1), np.arange(lz - h, lz + h + 1), sec.astype(float), levels=[0.5],
                       colors=[C_VERM], linewidths=0.7, linestyles=["-" if in_sec else (0, (2.0, 1.4))])
    for r in Jn["restored"]:
        rx, ry, rz = r
        ax.add_patch(patches.Rectangle((rx - 0.5, rz - 0.5), 1, 1, facecolor="none", edgecolor=C_INK, linewidth=1.3, zorder=6))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xticks([x0, x0 + 20, x0 + 40])
    ax.set_yticks([z0, last - 20, last, last + MARGIN])
    tick_style(ax)
    ax.set_xlabel(f"x (voxels), row y = {Jn['section_y']}", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel("z (slice)", fontsize=6.5, labelpad=1.5)
    ax.text(x1 + 1.2, (last + 0.5 + z1 + 0.5) / 2, "margin", fontsize=6.0, va="center", ha="left",
            color=C_INK2, rotation=90, clip_on=False)
    ax.text(x1 + 1.2, (z0 - 0.5 + last + 0.5) / 2, "volume", fontsize=6.0, va="center", ha="left", color=C_INK2,
            rotation=90, clip_on=False)
    n_rest = len(Jn["restored"])
    n_resp = len(Jn["responsible"])
    notes(fig, xJ, ynJ, wJ, [(f"{fmt(Jn['mirror_vs_ipl'][0])} voxels differ (whole volume)", True, C_INK if Jn["mirror_vs_ipl"][0] == 0 else C_VERM),
                             (f"{fmt(Jn['extra_survivors'])} margin survivors have no surviving partner;", False, C_INK2),
                             (f"{n_resp} of them (circles) restore {n_rest} voxels of the last slice (squares)", False, C_INK2)], checks)
    legend_axes(fig, xJ + 7.0 + imgJ_w + 4.5, y3 + TITLE_H - 0.5, wJ - imgJ_w - 11.5, imgJ_h + 1,
                [patches.Patch(fc=C_FILL, ec="none", label="object, in the volume"),
                 patches.Patch(fc=C_MIRROR, ec="#7f8b94", hatch="////", lw=0, label="mirror image,\nin the margin"),
                 patches.Patch(fc=C_BLUE, ec="none", label="erosion survivor,\nin the volume"),
                 patches.Patch(fc=C_SKY, ec="none", label="survivor in the margin,\npartner survives"),
                 patches.Patch(fc=C_VERM, ec="none", label="survivor in the margin,\npartner eroded"),
                 Line2D([], [], color=C_VERM, lw=0.7, label="its dilation ball\n(this section)")],
                loc="upper left", ncol=1, fontsize=6.2, handlelength=1.3, labelspacing=0.45)
    P["J"] = {k: v for k, v in Jn.items()}

    # ================================================================= row 4: K the conventions, L the N-scan
    K = R["K"]
    rows = {(r["N"], r["label"]): r for r in K["rows"]}
    ig = K["input_grid"][0]
    head(fig, xK, y4, wK, "K", "Margin conventions of the opening",
         f"voxels differing from IPL's /open {N_OPEN} of the whole\n{ig[0]}×{ig[1]}×{ig[2]}-voxel volume of J, one convention at a time", checks)
    labs = [c[0] for c in CONVENTIONS]
    vals = [rows[(N_OPEN, lab)]["mismatches"] for lab in labs]
    ax = ax_mm(fig, xK + 46.0, y4 + TITLE_H + 1.0, wK - 46.0, axes4_h)
    ypos = np.arange(len(labs))[::-1]
    floor = 0.55
    ax.barh(ypos, [max(v, floor) for v in vals], color=[C_BLUE if lab == IPL_CONVENTION else C_MUTED for lab in labs],
            height=0.62, lw=0)
    ax.set_xscale("log")
    ax.set_xlim(floor, max(vals) * 12)
    ax.set_yticks(ypos)
    ax.set_yticklabels([("IPL: " if lab == IPL_CONVENTION else "") + lab for lab in labs], fontsize=6.4)
    ax.set_xticks([1, 100, 10000, 1000000])
    ax.set_xticklabels(["1", "100", "10,000", "1,000,000"])
    ax.minorticks_off()
    tick_style(ax)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color=C_GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("voxels differing from IPL (log scale)", fontsize=6.5, labelpad=1.5)
    for y, v, lab in zip(ypos, vals, labs):
        ax.text(max(v, floor) * 1.35, y, "0 (identical)" if v == 0 else fmt(v), va="center", fontsize=6.4,
                fontweight="bold" if v == 0 else "normal", color=C_INK)
    P["K"] = dict(N=N_OPEN, input_grid=K["input_grid"], input_voxels=K["input_voxels"],
                  rows=[rows[(N_OPEN, lab)] for lab in labs])

    # ---- L: the N-scan
    head(fig, xL, y4, wL, "L", "Opening distance N",
         f"/open N of the same volume for N = {NSCAN_N[0]}–{NSCAN_N[-1]}, each\nconvention against IPL's export; N = {N_OPEN} is the standard", checks)
    axes_w = 44.0
    ax = ax_mm(fig, xL + 10.0, y4 + TITLE_H + 1.0, axes_w, axes4_h)
    series = [(CONVENTIONS[0][0], "edge-inclusive mirror,\ndepth N + 2 (IPL)", C_BLUE, "o"),
              (CONVENTIONS[6][0], "unpadded erosion,\nbackground margin", C_VERM, "s"),
              (CONVENTIONS[5][0], "edge-exclusive mirror,\ndepth N + 2", C_GREEN, "^"),
              (CONVENTIONS[7][0], "face slice copied,\ndepth N + 2", C_PURPLE, "D"),
              (CONVENTIONS[10][0], "all-background margin,\ndepth N + 2", C_INK2, "v")]
    P["L"] = dict(N=NSCAN_N, series={})
    handles = []
    for lab, name, col, mk in series:
        v = [rows[(N, lab)]["mismatches"] for N in NSCAN_N]
        ln, = ax.plot(NSCAN_N, v, color=col, lw=1.0, marker=mk, ms=3.2, mec=col, mfc=col if lab == IPL_CONVENTION else "white",
                      mew=0.8, label=name, zorder=4 if lab == IPL_CONVENTION else 3)
        handles.append(ln)
        P["L"]["series"][lab] = v
    ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
    ax.set_ylim(-0.6, 10 ** 6)
    ax.set_yticks([0, 1, 10, 100, 1000, 10000, 100000])
    ax.set_yticklabels(["0", "1", "10", "100", "1,000", "10,000", "100,000"])
    ax.minorticks_off()
    ax.set_xticks(NSCAN_N)
    ax.set_xlim(NSCAN_N[0] - 0.5, NSCAN_N[-1] + 0.5)
    tick_style(ax)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=C_GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.axvline(N_OPEN, color=C_AXIS, lw=0.7, ls=(0, (2.5, 2.0)), zorder=1)
    ax.text(N_OPEN + 0.12, 600, "standard\nN = 15", fontsize=6.3, color=C_INK2, ha="left", va="center", linespacing=1.1)
    ax.set_xlabel("opening distance N (voxels)", fontsize=6.5, labelpad=1.5)
    ax.set_ylabel("voxels differing from IPL", fontsize=6.5, labelpad=1.5)
    legend_axes(fig, xL + 10.0 + axes_w + 1.5, y4 + TITLE_H + 0.5, wL - axes_w - 11.5, axes4_h + 1, handles,
                loc="upper left", ncol=1, fontsize=6.2, handlelength=1.6, labelspacing=0.45)

    bad = check_text_widths(fig, checks)
    return fig, P, bad


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="ignore the cache and recompute everything")
    args = ap.parse_args()
    if not args.recompute and os.path.exists(CACHE + ".npz") and os.path.exists(CACHE + ".json"):
        say("loading the cache", CACHE)
        R = load_cache()
    else:
        R = dict(A=compute_metric(), B=compute_ball(), stages=compute_stages(), H=compute_peel(), I=compute_face(),
                 J=compute_mirror(), K=compute_conventions())
        save_cache(R)
        say("cache written", CACHE)
    fig, P, bad = draw(R)
    os.makedirs(HERE, exist_ok=True)
    fig.savefig(OUT + ".png", dpi=300, facecolor="white")
    fig.savefig(OUT + ".svg", facecolor="white")
    plt.close(fig)
    P["figure"] = dict(width_mm=FIG_W, height_mm=round(FIG_H, 1), dpi=300, voxel_mm=VOX_MM, slice_global_z=Z_MID, window_voxels=WIN,
                       text_overflows=bad, generated=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(P, open(OUT + "_numbers.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    say("wrote", OUT + ".png", OUT + ".svg", OUT + "_numbers.json")


if __name__ == "__main__":
    main()
