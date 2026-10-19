"""graphical_abstract.py -- the graphical abstract of the ipldt manuscript (JBMR Plus): one landscape figure,
8 x 4.5 in at 300 dpi (2400 x 1350 px), white background, no text below 14 pt, Okabe-Ito colours only.

Three zones.  Left: the mid-stack slice of one patellar scan and a zoomed region of it.  Middle: the same five
stages (compartment masks, rendered contours, Laplace-Hamming segmentation, cortical pores, Tb.Th map) in two lanes,
Scanco IPL and ipldt, whose thumbnails are IPL's products in the zoomed region -- identical in both lanes, which is
the point.  Right: the agreement numbers.  The cortical-pore thumbnail is a schematic (the cache holds no pore
map); everything else is drawn from the cached slice crops of Figure 1 (cache/fig1_workflow_slices.npz, read only)
with the conventions of fig1_workflow.py (grey window, mask tones, contour colours, Tb.Th colour map).

Run from the repository root in the ormir environment (set PYTHONUTF8=1):
    python manuscript/figures/graphical_abstract.py
Writes graphical_abstract.png (RGB, no alpha) and graphical_abstract.svg next to this file.  Deterministic.
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors, patches  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "fig1_workflow_slices.npz")
PORE_CACHE = os.path.join(HERE, "cache", "graphical_abstract_pore.npz")   # IPL's pore map, the same slice and window
OUT_PNG = os.path.join(HERE, "graphical_abstract.png")
OUT_SVG = os.path.join(HERE, "graphical_abstract.svg")

W_IN, H_IN, DPI = 8.0, 4.5, 300
W, H = W_IN * 72.0, H_IN * 72.0             # the canvas in points, y down
BLUE, ORANGE, GREEN, GREY, INK = "#0072B2", "#E69F00", "#009E73", "#999999", "#000000"   # Okabe-Ito
GREY_WINDOW = (-400, 8200)                  # native int16 display window, as in fig1_workflow.py
CROP = (slice(100, 300), slice(140, 300))   # (rows, cols) of the cached slice shown by the zoom and the thumbnails
FS, FS_HEAD, FS_NUM = 14.0, 16.0, 28.0      # points: smallest text, heading and phrases, headline numbers
LINE, CAP, DESC = 1.1, 0.72, 0.22           # baseline advance, cap height and descender depth (em)
MARGIN, GUT, TOP = 8.0, 8.0, 6.0
L_W, M_W, R_W = 100.0, 256.0, 188.0         # zone widths
L_X, M_X, R_X = MARGIN, MARGIN + L_W + GUT, MARGIN + L_W + GUT + M_W + GUT
N_ST, GAP_A, GAP_T = 5, 10.0, 6.0           # stages per lane, arrow gap, smallest gap between tiles
T_W = (M_W - (N_ST - 1) * GAP_A) / N_ST
T_H = T_W * (CROP[0].stop - CROP[0].start) / (CROP[1].stop - CROP[1].start)
STAGES = (("masks", "masks"), ("contours", "contours"), ("seg", "SEG"), ("pores", "pores"), ("tbth", "Tb.Th"))
LANES = ((ORANGE, "Scanco IPL", " \u2014 on the scanner,", "closed source"),
         (BLUE, "ipldt", " \u2014 open-source Python,", "CPU or GPU, any computer"))
TILES = (("0", ("of 18.3 billion", "voxels differ"), "Tb.Th, Tb.Sp, Tb.N, Ct.Th maps from IPL's segmentation", None),
         ("19.8", ("per million",), "segmentation voxels, Dice\u00a0\u2265\u00a00.998", None),
         ("\u2264 0.05 %", (), "BV/TV, Tb.Th, Tb.Sp, Tb.N; Ct.Th identical", None),
         ("\u2264 0.94 %", (), None, "Ct.Po"))
PORES = ((20, .45, .035), (55, .65, .025), (95, .35, .045), (140, .6, .03), (175, .4, .025), (215, .7, .04),
         (250, .3, .03), (290, .55, .035), (330, .65, .025), (355, .35, .03))   # angle, radial position, size

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "figure.dpi": DPI, "savefig.dpi": DPI, "svg.fonttype": "path"})


class Page:
    """The canvas in points with y pointing down; text is placed by its baseline."""

    def __init__(self):
        self.fig = plt.figure(figsize=(W_IN, H_IN))
        self.ax = self.fig.add_axes([0, 0, 1, 1], zorder=10)     # drawn above the image axes; its patch is not drawn
        self.ax.set(xlim=(0, W), ylim=(H, 0))
        self.ax.axis("off")
        self.ren = self.fig.canvas.get_renderer()

    def width(self, s, fs, weight="normal"):
        t = self.ax.text(0, 0, s, fontsize=fs, fontweight=weight)
        w = t.get_window_extent(self.ren).width / self.fig.dpi * 72.0
        t.remove()
        return w

    def text(self, x, y, s, fs=FS, color=INK, weight="normal", ha="left", va="baseline"):
        return self.ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight, ha=ha, va=va)

    def wrap(self, s, width, fs, weight="normal"):
        """Greedy word wrap by measured width; splits on plain spaces only (U+00A0 keeps words together)."""
        lines, cur = [], ""
        for w in s.split(" "):
            cand = (cur + " " + w).strip()
            if cur and self.width(cand, fs, weight) > width:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        return lines + [cur]

    def block(self, x, y, s, width, fs=FS, color=INK, weight="normal"):
        """Wrapped text whose first baseline is y; returns the bottom of the last line (descender included)."""
        for ln in self.wrap(s, width, fs, weight):
            self.text(x, y, ln, fs, color, weight)
            y += fs * LINE
        return y - fs * LINE + DESC * fs

    def axes(self, x, y, w, h, color=GREY, lw=0.6):
        """An image axes at (x, y, w, h) page points, framed in `color`."""
        ax = self.fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
        ax.set(xticks=[], yticks=[])
        for s in ax.spines.values():
            s.set(edgecolor=color, linewidth=lw)
        return ax

    def arrow(self, x0, x1, y):
        self.ax.add_patch(patches.FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10, lw=1.3,
                                                  color=GREY, shrinkA=0, shrinkB=0))


def draw_stage(ax, kind, D):
    """One stage thumbnail of the cropped region, drawn as fig1_workflow.py draws the same arrays."""
    c = CROP
    white = np.ones(D["grey"][c].shape + (3,))
    if kind == "masks":
        rgb = white.copy()
        rgb[D["trab"][c]], rgb[D["cort"][c]] = 0.80, 0.38
        ax.imshow(rgb, interpolation="antialiased")
    elif kind == "contours":
        ax.imshow(white, interpolation="antialiased")
        for m, col in ((D["g_cort"][c], INK), (D["g_trab"][c], GREEN)):
            ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=0.8)
    elif kind == "seg":
        rgb, s = white.copy(), D["seg"][c]
        rgb[s == 126], rgb[s == 127] = 0.55, 0.12
        ax.imshow(rgb, interpolation="antialiased")
    elif kind == "tbth":                                          # white for 0, viridis for 1..vmax voxel diameters
        vmax = int(D["tbth"].max())
        cm = colors.ListedColormap(np.vstack([[1, 1, 1, 1], plt.get_cmap("viridis", vmax)(np.arange(vmax))]))
        ax.imshow(D["tbth"][c], cmap=cm, norm=colors.BoundaryNorm(np.arange(-0.5, vmax + 1.5, 1), cm.N),
                  interpolation="nearest")
        ax.contour(D["g_trab"][c].astype(float), levels=[0.5], colors=[GREY], linewidths=0.5)
    elif kind == "pores" and D.get("pore") is not None:   # IPL's pore map: cortical compartment pale grey, pores dark
        rgb = white.copy()
        rgb[D["cort"][c]], rgb[D["pore"][c]] = 0.72, 0.12
        ax.imshow(rgb, interpolation="antialiased")
    elif kind == "pores":          # fallback when the pore cache is absent: a cortical ring with a few dark pores
        w, h = T_W, T_H
        ax.set(xlim=(0, w), ylim=(h, 0), aspect="equal")
        cx, cy, ro, ri = w / 2, h / 2, 0.42 * w, 0.25 * w
        ax.add_patch(patches.Circle((cx, cy), ro, fc="0.72", ec="none"))
        ax.add_patch(patches.Circle((cx, cy), ri, fc="white", ec="none"))
        for ang, f, s in PORES:
            a, r = np.deg2rad(ang), ri + f * (ro - ri)
            ax.add_patch(patches.Circle((cx + r * np.cos(a), cy + r * np.sin(a)), s * w, fc="0.12", ec="none"))


def draw_title(p):
    """One line across the top, at the largest size (<= 22 pt) that fits the width; returns the zones' top."""
    name, rest = "ipldt", ": an open-source reimplementation of IPL bone morphometry"
    fs = 20.0 * (W - 2 * MARGIN) / (p.width(name, 20, "bold") + p.width(rest, 20))
    fs = min(22.0, np.floor(fs * 10) / 10)
    assert fs >= FS, fs
    wn, wr = p.width(name, fs, "bold"), p.width(rest, fs)
    x, y = (W - wn - wr) / 2, TOP + CAP * fs
    p.text(x, y, name, fs, INK, "bold")
    p.text(x + wn, y, rest, fs, INK)
    print(f"title at {fs:.1f} pt")
    return y + DESC * fs + 7.0


def draw_left(p, D, y0):
    b = y0 + CAP * FS
    p.text(L_X, b, "HR-pQCT scan", FS, INK, "bold")
    y = p.block(L_X, b + FS * LINE, "(XtremeCT II, 60.7 \u00b5m)", L_W, FS, GREY) + 4.0
    lo, hi = GREY_WINDOW
    g = np.repeat(np.clip((D["grey"] - lo) / (hi - lo), 0, 1)[..., None], 3, -1)
    hs, ws = g.shape[:2]
    s = L_W / ws                                                  # points per voxel of the full slice
    ax = p.axes(L_X, y, L_W, hs * s)
    ax.imshow(g, interpolation="antialiased")
    r0, r1, c0, c1 = CROP[0].start, CROP[0].stop, CROP[1].start, CROP[1].stop
    ax.add_patch(patches.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fc="none", ec="white", lw=0.8))
    zy = y + hs * s + 9.0
    zax = p.axes(L_X, zy, L_W, L_W * (r1 - r0) / (c1 - c0))
    zax.imshow(g[CROP], interpolation="antialiased")
    for cx, zx in ((c0, L_X), (c1, L_X + L_W)):                  # where the zoom comes from
        p.ax.plot([L_X + cx * s, zx], [y + r1 * s, zy], color=GREY, lw=0.6)
    n = 2.0 / float(D["el"])                                      # 2 mm in voxels
    x0, yb = (c1 - c0) * 0.05, (r1 - r0) * 0.94
    zax.plot([x0, x0 + n], [yb, yb], color="white", lw=2.0, solid_capstyle="butt")
    zax.text(x0 + n + 4, yb, "2 mm", color="white", fontsize=FS, ha="left", va="center")


def lane_label(p, y, col, name, tail, line2):
    """Colour chip, bold name and the rest of the line, then a second line; returns the bottom."""
    b, side = y + CAP * FS, CAP * FS
    p.ax.add_patch(patches.FancyBboxPatch((M_X, b - side), side, side, boxstyle="round,pad=0,rounding_size=1.5",
                                          fc=col, ec="none"))
    x = M_X + side + 5.0
    p.text(x, b, name, FS, INK, "bold")
    p.text(x + p.width(name, FS, "bold"), b, tail, FS, INK)
    p.text(x, b + FS * LINE, line2, FS, INK)
    return b + FS * LINE + DESC * FS


def stage_names(p, y, cx):
    for x, (_, name) in zip(cx, STAGES):
        p.text(x, y + CAP * FS, name, FS, GREY, ha="center")
    return y + (CAP + DESC) * FS


def draw_middle(p, D, y0, y1):
    lab_h, nam_h, eq_h = (CAP + LINE + DESC) * FS, (CAP + DESC) * FS, CAP * FS_NUM
    gap = min(9.0, (y1 - y0 - 2 * (lab_h + nam_h + T_H) - eq_h) / 6)
    assert gap >= 3.0, gap
    cx = [M_X + i * (T_W + GAP_A) + T_W / 2 for i in range(N_ST)]
    y = y0
    for k, (col, name, tail, line2) in enumerate(LANES):
        if k == 0:                                                # label and names above the IPL lane ...
            y = lane_label(p, y, col, name, tail, line2) + gap
            y = stage_names(p, y, cx) + gap
        for i, (kind, _) in enumerate(STAGES):
            x = M_X + i * (T_W + GAP_A)
            draw_stage(p.axes(x, y, T_W, T_H, col, 1.3), kind, D)
            if i:
                p.arrow(x - GAP_A + 1.5, x - 1.5, y + T_H / 2)
        y += T_H + gap
        if k == 0:                                                # ... the equality between the lanes ...
            ym = y + eq_h / 2
            p.text(cx[-1], ym, "=", FS_NUM, GREEN, "bold", ha="center", va="center_baseline")
            p.text(cx[-1] - p.width("=", FS_NUM, "bold") / 2 - 7.0, ym, "same output, voxel for voxel",
                   FS, GREEN, "bold", ha="right", va="center_baseline")
            y += eq_h + gap
        else:                                                     # ... names and label below the ipldt lane
            y = lane_label(p, stage_names(p, y, cx) + gap, col, name, tail, line2)
    assert y <= y1 + 0.5, (y, y1)


def tile_height(p, num, phrase, sub):
    """Height of a tile (number, phrase beside it, wrapped sub-label), measured without drawing."""
    h = CAP * FS_NUM + 1.0
    if len(phrase) == 1:
        h = CAP * FS_NUM + DESC * FS_HEAD
    elif len(phrase) == 2:
        h = max(h, CAP * FS_NUM / 2 + (CAP * FS_HEAD + FS_HEAD * LINE) / 2 + DESC * FS_HEAD)
    if sub:
        h += 3.0 + CAP * FS + (len(p.wrap(sub, R_W, FS)) - 1) * FS * LINE + DESC * FS
    return h


def draw_right(p, y0, y1):
    b = y0 + CAP * FS_HEAD
    for ln in ("Agreement on", "137 scans, 5 sites"):
        p.text(R_X, b, ln, FS_HEAD, INK, "bold")
        b += FS_HEAD * LINE
    y = b - FS_HEAD * LINE + DESC * FS_HEAD
    heights = [tile_height(p, n, ph, s) for n, ph, s, _ in TILES]
    gap = min(14.0, max(GAP_T, (y1 - y - sum(heights)) / len(TILES)))      # the tiles spread down the zone
    for (num, phrase, sub, beside), h in zip(TILES, heights):
        y += gap
        base = y + CAP * FS_NUM
        p.text(R_X, base, num, FS_NUM, GREEN, "bold")
        xp = R_X + p.width(num, FS_NUM, "bold") + 6.0
        bottom = base + 1.0
        if len(phrase) == 1:                                      # on the number's baseline
            p.text(xp, base, phrase[0], FS_HEAD, GREEN)
            bottom = base + DESC * FS_HEAD
        elif len(phrase) == 2:                                    # stacked beside the number, centred on its height
            b1 = base - CAP * FS_NUM / 2 - (CAP * FS_HEAD + FS_HEAD * LINE) / 2 + CAP * FS_HEAD
            for j, ln in enumerate(phrase):
                p.text(xp, b1 + j * FS_HEAD * LINE, ln, FS_HEAD, GREEN)
            bottom = max(bottom, b1 + FS_HEAD * LINE + DESC * FS_HEAD)
        assert all(xp + p.width(ln, FS_HEAD) <= R_X + R_W for ln in phrase), phrase
        if beside:
            p.text(xp, base, beside, FS, GREY)
        if sub:
            bottom = p.block(R_X, bottom + 3.0 + CAP * FS, sub, R_W, FS, GREY)
        assert abs(bottom - (y + h)) < 0.01, (bottom, y + h)
        y = bottom
    assert y <= y1 + 0.5, (y, y1)


def main():
    z = np.load(CACHE)
    D = {k: z[k] for k in ("grey", "cort", "trab", "g_cort", "g_trab", "seg", "tbth", "el")}
    if os.path.exists(PORE_CACHE):
        pz = np.load(PORE_CACHE)
        assert int(pz["z_global"]) == int(z["z_global"]) and np.array_equal(pz["frame"], z["frame"])
        D["pore"] = pz["pore"]
    p = Page()
    y0 = draw_title(p)
    y1 = H - MARGIN - (CAP + DESC) * FS - 6.0                     # above the repository line
    draw_left(p, D, y0)
    draw_middle(p, D, y0, y1)
    draw_right(p, y0, y1)
    p.text(W - MARGIN, H - MARGIN - DESC * FS, "github.com/gkazakia/ipldt \u00b7 MIT license", FS, INK, ha="right")
    p.fig.savefig(OUT_PNG, dpi=DPI, facecolor="white")
    p.fig.savefig(OUT_SVG, facecolor="white")
    from PIL import Image
    im = Image.open(OUT_PNG)
    assert im.size == (2400, 1350) and im.getchannel("A").getextrema() == (255, 255), (im.size, im.mode)
    im.convert("RGB").save(OUT_PNG, dpi=(DPI, DPI))                # opaque white background, no alpha channel
    print("wrote", OUT_PNG, im.size)
    print("wrote", OUT_SVG)


if __name__ == "__main__":
    main()
