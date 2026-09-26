"""S8_derivation.py -- Supplementary Figure S8: how one rule of the reimplementation was derived, shown as method.

One worked example: the padding rule of IPL's Laplace-Hamming filter.  /fft_laplace_hamming pads each axis to
the next power of two before the FFT, and two things had to be settled: where an odd pad places its extra voxel,
and what that voxel holds.  The figure lays the derivation out step by step:

  A  the controlled input: a synthetic 63 x 64 x 64 volume (all zero, one voxel of 25,000 on the mid-line) in four
     copies whose impulse sits at x = 0, 1, 61 or 62; a 64-wide transform box needs a single padding layer on x,
     and its side and its content are the unknowns.
  B  the candidate rules: 4 offsets (where the layer goes) x 5 fill modes (what it contains) = 20 uniform
     candidates (+ 12 per-axis mixtures, 32 in all); a cell of B names the data index copied into the padding
     layer.  Copying the impulse's own index doubles its response, which lifts the thresholded export above 0.
  C  the predictions, fixed before the scanner run: the expected count of set voxels in the thresholded export
     for each candidate and input; the last row is the scanner's result.
  D, E  the float-level readout for the impulse at x = 1 (D) and x = 61 (E): the mid-line of IPL's exported filter
     output, with the values the selected rule and the alternative offset (padding after the data) predict there.
  F  the same differential test over every controlled input of the run (30 synthetic inputs and 6 blocks from a
     real scan) and all 32 candidates: green where a candidate's predicted float agrees with IPL's export within
     1 float unit at every voxel, grey otherwise; a single candidate agrees on 36/36.

Every number drawn is read from the prospective prediction file and IPL's exported AIMs of that run; nothing is
recomputed from a model.  Panel letters are capitals in the upper-left corner.

Run from the repository root in the `ormir` environment:

    set PYTHONUTF8=1
    python manuscript/figures/supp/S8_derivation.py

Writes manuscript/figures/supp/S8_derivation.png (300 dpi, 180 mm wide), S8_derivation.svg and
S8_derivation_numbers.json (every number drawn or quoted).  Runtime a few seconds.  Deterministic.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
sys.path.insert(0, REPO)
from ipldt.io import read_aim  # noqa: E402

# ------------------------------------------------------------------------------------------------ inputs
RUN21_DIR = lab_path("ipl_test_runs/run21")
AIMS = os.path.join(RUN21_DIR, "aims_and_logs")                # IPL's exports of the run
VOLS = os.path.join(RUN21_DIR, "prediction_volumes")           # the prospective predictions, per candidate group
PRED_JSON = os.path.join(RUN21_DIR, "run21_prediction.json")  # the prediction file written before the run
PHANTOMS = os.path.join(RUN21_DIR, "phantom21_manifest.json")
MANIFEST = os.path.join(RUN21_DIR, "run21_manifest.json")
BASE, TAGP = "X2420448", "T21"
OUT_PNG = os.path.join(HERE, "S8_derivation.png")
OUT_SVG = os.path.join(HERE, "S8_derivation.svg")
OUT_NUM = os.path.join(HERE, "S8_derivation_numbers.json")

FLOAT_TOL = 1.0                       # float units: a candidate is refuted by an input above this
IMPULSE_INPUTS = ["x63i00", "x63i01", "x63i61", "x63i62"]   # the four single-impulse inputs of the worked example
IMPULSE_X = {"x63i00": 0, "x63i01": 1, "x63i61": 61, "x63i62": 62}
OFFSETS = ["floor", "ceil", "left", "right"]
MODES = ["reflect", "symmetric", "edge", "constant", "wrap"]
SELECTED, SHIPPED = "ceil+reflect", "floor+reflect"

T0 = time.time()
NUM = {}


def say(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


def fmt_int(n):
    """Thousands separated by a thin space, journal style."""
    return f"{int(n):,d}"


# ------------------------------------------------------------------------------------------------ style
# Okabe-Ito, colour-blind safe
C_BLUE, C_ORANGE, C_VERM, C_GREEN, C_SKY = "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9"
INK, INK2, GRID = "#1A1A1A", "#4D4D4D", "#D9D9D9"
C_CELL, C_PAD, C_HIT = "#F2F2F2", "#FBE9C8", "#E69F00"          # data cell, padding layer, impulse / copied cell
C_AGREE, C_REFUTE = "#7FCBB0", "#DEDEDE"                         # F: within tolerance / refuted
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0, "axes.titlesize": 7.0, "axes.labelsize": 7.0, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "savefig.dpi": 300, "figure.dpi": 100, "svg.fonttype": "none", "pdf.fonttype": 42,
})
FIG_W_MM, FIG_H_MM = 180.0, 227.8          # <= 228 mm: printed 1:1 (the manuscript build), every text >= 6 pt


def mm_axes(fig, x, y, w, h, **kw):
    """An axes placed in millimetres from the top-left corner of the figure."""
    return fig.add_axes([x / FIG_W_MM, 1.0 - (y + h) / FIG_H_MM, w / FIG_W_MM, h / FIG_H_MM], **kw)


def panel_letter(fig, x, y, s):
    fig.text(x / FIG_W_MM, 1.0 - y / FIG_H_MM, s, fontsize=9, fontweight="bold", ha="left", va="top", color=INK)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------------------------------------ data
def aim(tag, kind):
    return read_aim(os.path.join(AIMS, f"{BASE}_{TAGP}_{tag.upper()}_{kind}.AIM"))["data"]


class Pred:
    """The prospective predictions of one input: per candidate group, the float image and the thresholded bits."""

    def __init__(self, vol):
        self.z = np.load(os.path.join(VOLS, f"{vol}.npz"))
        self.dim = tuple(int(v) for v in self.z["dim_xyz"])
        self.cands = [str(c) for c in self.z["candidates"]]
        self.group_of = {c: int(g) for c, g in zip(self.cands, self.z["group_of"])}

    def float_of(self, cand):
        return self.z[f"f{self.group_of[cand]:02d}"]

    def seg_of(self, cand):
        n = int(np.prod(self.dim))
        b = np.unpackbits(self.z[f"b{self.group_of[cand]:02d}"])[:n].astype(bool)
        return b.reshape(self.dim[::-1])


say("reading the prediction file and the manifests")
PRED = json.load(open(PRED_JSON, encoding="utf-8"))
PH = json.load(open(PHANTOMS, encoding="utf-8"))
MAN = json.load(open(MANIFEST, encoding="utf-8"))
CANDS = [c["name"] for c in PRED["candidates"]]
assert len(CANDS) == 32 and SELECTED in CANDS and SHIPPED in CANDS
K = PRED["constants"]
THR_FLOAT = K["threshold"] * K["norm_max"] / K["int16_max"]        # the threshold in float units
NUM["constants"] = dict(K, threshold_float_units=THR_FLOAT)
NUM["candidates"] = CANDS
NUM["separation_tolerance_float_units"] = FLOAT_TOL

# the synthetic inputs of the worked example
inputs = {}
for v in IMPULSE_INPUTS:
    f = PH["files"][v]
    assert f["dim"] == [63, 64, 64] and f["set_voxels"] == 1 and f["content"][0]["xyz"] == [IMPULSE_X[v], 32, 32]
    inputs[v] = {"dim": f["dim"], "pad": f["pad"], "impulse_xyz": f["content"][0]["xyz"], "value": f["value_max"]}
NUM["inputs"] = inputs

# panel C: the prediction table (voxels set in the thresholded export) and the scanner's counts
table = {}
for v in IMPULSE_INPUTS:
    t = PRED["seg_readout_by_axis_rule"][v]["table"]
    table[v] = {c: int(t[c]) for c in CANDS if c in t}
    assert len(table[v]) == 20
scanner_seg = {v: int((aim(v, "SG") > 0).sum()) for v in IMPULSE_INPUTS}
NUM["prediction_table_seg_voxels"] = table
NUM["scanner_seg_voxels"] = scanner_seg
uniform = [f"{o}+{m}" for o in OFFSETS for m in MODES]
consistent = [c for c in uniform if all(table[v][c] == scanner_seg[v] for v in IMPULSE_INPUTS)]
NUM["candidates_consistent_with_the_four_inputs"] = consistent
say("scanner thresholded counts", scanner_seg, "consistent:", consistent)

# panels D / E: mid-line profiles
profiles = {}
for v in ["x63i01", "x63i61"]:
    lh = aim(v, "LH")
    P = Pred(v)
    sel, shp = P.float_of(SELECTED), P.float_of(SHIPPED)
    profiles[v] = {
        "x": list(range(lh.shape[2])),
        "ipl": lh[32, 32, :].astype(float).tolist(),
        "selected": sel[32, 32, :].astype(float).tolist(),
        "shipped": shp[32, 32, :].astype(float).tolist(),
        "max_abs_diff_selected": float(np.abs(sel.astype(np.float64) - lh).max()),
        "max_abs_diff_shipped": float(np.abs(shp.astype(np.float64) - lh).max()),
        "ipl_peak": float(lh.max()),
        "ipl_peak_x": int(np.unravel_index(int(np.argmax(lh)), lh.shape)[2]),
        "seg_set_ipl": scanner_seg[v],
        "seg_set_selected": int(P.seg_of(SELECTED).sum()),
        "seg_set_shipped": int(P.seg_of(SHIPPED).sum()),
        "seg_differ_selected": int((P.seg_of(SELECTED) != (aim(v, "SG") > 0)).sum()),
    }
    say(v, "max|IPL - selected|", profiles[v]["max_abs_diff_selected"], "max|IPL - shipped|",
        profiles[v]["max_abs_diff_shipped"], "IPL peak", profiles[v]["ipl_peak"], "at x =", profiles[v]["ipl_peak_x"])
NUM["profiles"] = profiles

# panel F: every controlled input x every candidate
INPUT_LABEL = {
    "c64imp": "64\u00b3: center impulse", "c64plx": "64\u00b3: plane normal to x", "c64plz": "64\u00b3: plane normal to z",
    "c64f6": "64\u00b3: 6 face impulses", "c64mid": "64\u00b3: 7\u00b3 cube",
    "x63i00": "63\u00d764\u00d764: impulse x = 0", "x63i01": "63\u00d764\u00d764: impulse x = 1",
    "x63i61": "63\u00d764\u00d764: impulse x = 61", "x63i62": "63\u00d764\u00d764: impulse x = 62",
    "x63p00": "63\u00d764\u00d764: plane x = 0", "x63p01": "63\u00d764\u00d764: plane x = 1",
    "x63p61": "63\u00d764\u00d764: plane x = 61", "x63p62": "63\u00d764\u00d764: plane x = 62",
    "x63cmb": "63\u00d764\u00d764: 6 impulses (x)", "y63cmb": "64\u00d763\u00d764: 6 impulses (y)",
    "z63cmb": "64\u00d764\u00d763: 6 impulses (z)", "x62cmb": "62\u00d764\u00d764: 6 impulses (x)",
    "z62cmb": "64\u00d764\u00d762: 6 impulses (z)", "x33cmb": "33\u00d764\u00d764: 7 impulses (x)",
    "z33cmb": "64\u00d764\u00d733: 7 impulses (z)", "x34cmb": "34\u00d764\u00d764: 7 impulses (x)",
    "z34cmb": "64\u00d764\u00d734: 7 impulses (z)", "a636462": "63\u00d764\u00d762: 8 impulses",
    "a626364": "62\u00d763\u00d764: 8 impulses", "a63c62": "63\u00b3: impulse (62, 62, 62)",
    "a63c61": "63\u00b3: impulse (61, 61, 61)", "a63c00": "63\u00b3: impulse (0, 0, 0)",
    "x63step": "63\u00d764\u00d764: slab x \u2265 32", "x63sthf": "63\u00d764\u00d764: stepped slab",
    "x63mid": "63\u00d764\u00d764: 7\u00b3 cube",
    "r64": "scan block 64\u00b3", "r63": "scan block 63\u00b3", "r62": "scan block 62\u00b3",
    "r63x": "scan block 63\u00d764\u00d764", "r63z": "scan block 64\u00d764\u00d763", "r33": "scan block 33\u00d764\u00d764",
}
SYNTH = list(PH["order"])
REAL = list(MAN["blocks"][1]["subs"].keys())
ALL_INPUTS = SYNTH + REAL
assert all(v in INPUT_LABEL for v in ALL_INPUTS) and len(ALL_INPUTS) == 36

say("differential test over", len(ALL_INPUTS), "inputs x", len(CANDS), "candidates")
maxd = np.zeros((len(CANDS), len(ALL_INPUTS)))
segd = np.zeros((len(CANDS), len(ALL_INPUTS)), dtype=int)
for j, v in enumerate(ALL_INPUTS):
    lh = aim(v, "LH").astype(np.float64)
    sg = aim(v, "SG") > 0
    P = Pred(v)
    cache = {}
    for i, c in enumerate(CANDS):
        gi = P.group_of[c]
        if gi not in cache:
            cache[gi] = (float(np.abs(P.float_of(c).astype(np.float64) - lh).max()), int((P.seg_of(c) != sg).sum()))
        maxd[i, j], segd[i, j] = cache[gi]
refuted = maxd > FLOAT_TOL
n_ref = refuted.sum(axis=1)
i_sel = CANDS.index(SELECTED)
assert n_ref[i_sel] == 0 and segd[i_sel].max() == 0, "the selected rule must match every export"
order = sorted(range(len(CANDS)), key=lambda i: (int(n_ref[i]), CANDS[i]))
NUM["F"] = {
    "inputs": ALL_INPUTS, "input_labels": [INPUT_LABEL[v] for v in ALL_INPUTS],
    "n_synthetic": len(SYNTH), "n_real_scan_blocks": len(REAL),
    "max_abs_diff_float_units": {c: maxd[i].tolist() for i, c in enumerate(CANDS)},
    "seg_differing_voxels": {c: segd[i].tolist() for i, c in enumerate(CANDS)},
    "n_inputs_refuting": {c: int(n_ref[i]) for i, c in enumerate(CANDS)},
    "selected_max_abs_diff_over_all_inputs": float(maxd[i_sel].max()),
    "min_refuting_inputs_among_others": int(min(n_ref[i] for i in range(len(CANDS)) if i != i_sel)),
    "min_seg_differing_inputs_among_others": int(min((segd[i] > 0).sum() for i in range(len(CANDS)) if i != i_sel)),
    "row_order": [CANDS[i] for i in order],
}
say("selected rule: max|IPL - prediction| over all inputs", NUM["F"]["selected_max_abs_diff_over_all_inputs"],
    "; every other candidate refuted by >=", NUM["F"]["min_refuting_inputs_among_others"], "inputs")


# ------------------------------------------------------------------------------------------------ helpers
def cand_label(c):
    """Human-readable candidate name: 'ceil + reflect', or 'x ceil, y floor, z floor + reflect' for the mixtures."""
    if c.startswith("perax("):
        spec, mode = c[len("perax("):].split(")+")
        parts = [p.split("=") for p in spec.split(",")]
        return ", ".join(f"{a} {b}" for a, b in parts) + f" + {mode}"
    o, m = c.split("+")
    return f"{o} + {m}"


def cells(ax, x0, y0, n, w, h, labels=None, fill=None, ec=INK2, lw=0.4, fs=5.5, tc=INK):
    """A row of n cells starting at (x0, y0); returns their x centres."""
    xs = []
    for i in range(n):
        fc = fill(i) if callable(fill) else (fill or C_CELL)
        ax.add_patch(Rectangle((x0 + i * w, y0), w, h, facecolor=fc, edgecolor=ec, linewidth=lw))
        if labels is not None and labels[i] is not None:
            ax.text(x0 + (i + 0.5) * w, y0 + h / 2, labels[i], ha="center", va="center", fontsize=fs, color=tc)
        xs.append(x0 + (i + 0.5) * w)
    return xs


# ------------------------------------------------------------------------------------------------ figure
fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))

# ---- A: the controlled input ------------------------------------------------------------------------------
axA = mm_axes(fig, 6, 7, 82, 40)
bare(axA)
axA.set_xlim(-17, 66)
axA.set_ylim(0, 10)
panel_letter(fig, 3, 4, "A")
axA.text(-17, 9.6, f"Controlled input: 63 \u00d7 64 \u00d7 64 voxels, one voxel of {fmt_int(inputs['x63i01']['value'])} "
         "on the mid-line", ha="left", va="center", fontsize=7)
# the transform box: 64 cells, one of them padding, before or after the data
yb, hb = 7.55, 0.9
axA.add_patch(Rectangle((-1, yb), 65, hb, facecolor="none", edgecolor=INK, linewidth=0.7, linestyle=(0, (2, 1.2))))
for xq in (-1, 63):
    axA.add_patch(Rectangle((xq, yb), 1, hb, facecolor=C_PAD, edgecolor=INK, linewidth=0.5, linestyle=(0, (2, 1.2))))
    axA.text(xq + 0.5, yb + hb / 2, "?", ha="center", va="center", fontsize=6.5, fontweight="bold", color=C_VERM)
axA.text(31.5, yb + hb + 0.15, "transform box, 64 voxels: the 63 data voxels + ONE padding layer, before (?) or after (?)",
         ha="center", va="bottom", fontsize=6.0, color=INK2)
# the four inputs: strips of 63 data cells, the impulse cell filled
ys = [6.15, 4.85, 3.55, 2.25]
for v, y in zip(IMPULSE_INPUTS, ys):
    ix = IMPULSE_X[v]
    cells(axA, 0, y, 63, 1, 0.9, fill=lambda i, ix=ix: C_HIT if i == ix else C_CELL, lw=0.25)
    axA.text(-0.8, y + 0.45, f"impulse at x = {ix}", ha="right", va="center", fontsize=6.2)
    axA.annotate("", xy=(ix + 0.5, y + 0.9), xytext=(ix + 0.5, y + 1.22),
                 arrowprops=dict(arrowstyle="-", color=C_VERM, lw=0.7))
for xi in (0, 62):
    axA.text(xi + 0.5, 1.95, str(xi), ha="center", va="top", fontsize=6.0, color=INK2)
axA.text(31.5, 1.95, "data index x", ha="center", va="top", fontsize=6.0, color=INK2)
axA.text(-17, 0.95, "Filter at IPL's standard values: \u03b5 = 0.45, cutoff 0.3, amplitude 1;",
         ha="left", va="center", fontsize=6.0, color=INK2)
axA.text(-17, 0.25, "output scaled to \u00b1200,000 \u2192 int16 and thresholded at 475\u2030 of that range",
         ha="left", va="center", fontsize=6.0, color=INK2)

# ---- B: the candidate rules -------------------------------------------------------------------------------
axB = mm_axes(fig, 98, 7, 80, 40)
bare(axB)
axB.set_xlim(0, 100)
axB.set_ylim(0, 40)
panel_letter(fig, 92, 4, "B")
axB.text(0, 38.6, "Candidate rules: 4 offsets \u00d7 5 fill modes, + 12 per-axis mixtures = 32",
         ha="left", va="center", fontsize=7)
# two columns: padding layer after the data (floor, left) / before the data (ceil, right)
COLX = {"after": 37.5, "before": 72.8}        # the left column holds a one-line description of each mode
axB.text(COLX["after"] + 13.6, 35.0, "layer AFTER the data\noffsets floor, left", ha="center", va="center", fontsize=6.0,
         linespacing=1.05)
axB.text(COLX["before"] + 13.6, 35.0, "layer BEFORE the data\noffsets ceil, right", ha="center", va="center", fontsize=6.0,
         linespacing=1.05)
# what each mode copies into the padding layer P (for one padding voxel)
COPY = {  # mode: (source index when the layer is after the data, source index when it is before)
    "reflect": (61, 1), "symmetric": (62, 0), "edge": (62, 0), "constant": (None, None), "wrap": (0, 62),
}
MODE_TEXT = {"reflect": "mirror, face voxel not repeated", "symmetric": "mirror, face voxel repeated",
             "edge": "face voxel copied", "constant": "zero", "wrap": "periodic"}
NUM["B_copied_index"] = {m: {"layer_after_data": COPY[m][0], "layer_before_data": COPY[m][1]} for m in MODES}
strip_labels = ["0", "1", "2", "\u2026", "60", "61", "62"]
idx_of = {"0": 0, "1": 1, "2": 2, "60": 60, "61": 61, "62": 62}
cw, ch = 3.4, 2.8
for r, m in enumerate(MODES):
    y = 29.0 - r * 5.3
    axB.text(0, y + ch / 2 + 0.9, m, ha="left", va="center", fontsize=6.5, fontweight="bold")
    axB.text(0, y + ch / 2 - 1.35, MODE_TEXT[m], ha="left", va="center", fontsize=6.0, color=INK2)
    for side in ("after", "before"):
        x0 = COLX[side]
        src = COPY[m][0] if side == "after" else COPY[m][1]
        seq = (strip_labels + ["P"]) if side == "after" else (["P"] + strip_labels)
        ip = seq.index("P")
        labels = [("0" if src is None else str(src)) if sq == "P" else sq for sq in seq]

        def fc(i, seq=seq, src=src):
            sq = seq[i]
            if sq == "P":
                return C_PAD
            if sq in idx_of and idx_of[sq] == src:
                return C_HIT
            return C_CELL

        xs = cells(axB, x0, y, len(seq), cw, ch, labels=labels, fill=fc, lw=0.35, fs=6.0)
        axB.add_patch(Rectangle((x0 + ip * cw, y), cw, ch, facecolor="none", edgecolor=INK, linewidth=0.6,
                                linestyle=(0, (1.2, 0.8))))
        if src is not None:
            i_src = [i for i, sq in enumerate(seq) if sq in idx_of and idx_of[sq] == src][0]
            chord = abs(xs[ip] - xs[i_src])
            rad = 2 * 1.25 / chord               # an arc about 1.25 mm high whatever the distance
            axB.add_patch(FancyArrowPatch((xs[i_src], y + ch), (xs[ip], y + ch), connectionstyle=f"arc3,rad={-rad:.3f}",
                                          arrowstyle="-|>", mutation_scale=4, color=C_VERM, lw=0.55, shrinkA=0, shrinkB=0))
for yl, line in zip((5.0, 2.55, 0.1, -2.35), (
        "Dashed cell: the padding layer, printed with the data index it receives (orange cell);",
        "\"constant\" writes the value 0.  With ONE padding voxel floor \u2261 left and ceil \u2261 right;",
        "the 62- and 33-voxel inputs of F (2 and 31 padding voxels) separate them.",
        "The per-axis mixtures apply a different offset on each axis.")):
    axB.text(0, yl, line, ha="left", va="center", fontsize=6.0, color=INK2)

# ---- C: the prediction table ------------------------------------------------------------------------------
axC = mm_axes(fig, 6, 55, 82, 66)
bare(axC)
axC.set_xlim(0, 82)
axC.set_ylim(0, 66)
panel_letter(fig, 3, 52, "C")
axC.text(0, 65, "Prediction recorded before the scanner run: voxels set in the thresholded export",
         ha="left", va="center", fontsize=7)
x_off, x_mode, x_col0, colw = 0, 11, 34, 11.5
row_h = 2.6
y_top = 61.5
# header
axC.text(x_col0 + 2 * colw, y_top + 0.9, "input: impulse at x =", ha="center", va="center", fontsize=6.0, color=INK2)
for j, v in enumerate(IMPULSE_INPUTS):
    axC.text(x_col0 + (j + 0.5) * colw, y_top - 1.2, str(IMPULSE_X[v]), ha="center", va="center", fontsize=6.5,
             fontweight="bold")
axC.text(x_off, y_top - 1.2, "offset", ha="left", va="center", fontsize=6.0, color=INK2)
axC.text(x_mode, y_top - 1.2, "fill mode", ha="left", va="center", fontsize=6.0, color=INK2)
axC.plot([0, x_col0 + 4 * colw], [y_top - 2.4, y_top - 2.4], color=INK2, lw=0.5)
y = y_top - 2.4
for o in OFFSETS:
    y_group_top = y
    for m in MODES:
        c = f"{o}+{m}"
        y -= row_h
        ok = c in consistent
        if ok:
            axC.add_patch(Rectangle((x_off - 0.5, y), x_col0 + 4 * colw + 1, row_h, facecolor="none",
                                    edgecolor=C_GREEN, linewidth=0.9, zorder=3))
        axC.text(x_mode, y + row_h / 2, m, ha="left", va="center", fontsize=6.2,
                 color=C_GREEN if ok else INK, fontweight="bold" if ok else "normal")
        for j, v in enumerate(IMPULSE_INPUTS):
            n = table[v][c]
            axC.add_patch(Rectangle((x_col0 + j * colw, y), colw, row_h, facecolor=C_PAD if n else "white",
                                    edgecolor=GRID, linewidth=0.3))
            axC.text(x_col0 + (j + 0.5) * colw, y + row_h / 2, str(n), ha="center", va="center", fontsize=6.2,
                     color=C_GREEN if ok else INK, fontweight="bold" if ok else "normal")
    axC.text(x_off, (y_group_top + y) / 2, o, ha="left", va="center", fontsize=6.5, fontweight="bold")
    axC.plot([0, x_col0 + 4 * colw], [y, y], color=GRID, lw=0.4)
# the scanner's row
y -= row_h + 0.6
axC.add_patch(Rectangle((x_off - 0.5, y), x_col0 + 4 * colw + 1, row_h, facecolor="#EBEBEB", edgecolor="none"))
axC.text(x_off, y + row_h / 2, "scanner output", ha="left", va="center", fontsize=6.5, fontweight="bold")
for j, v in enumerate(IMPULSE_INPUTS):
    axC.text(x_col0 + (j + 0.5) * colw, y + row_h / 2, str(scanner_seg[v]), ha="center", va="center", fontsize=6.5,
             fontweight="bold")
for dy, line in zip((2.0, 4.4, 6.8), (
        f"Green: rows equal to the scanner's on all four inputs ({len(consistent)} of 20; with one padding voxel",
        "ceil and right coincide, F separates them).  Shaded: the padding layer received",
        "the impulse's own index (B), doubling the response across the threshold.")):
    axC.text(0, y - dy, line, ha="left", va="center", fontsize=6.0, color=INK2)

# ---- D, E: the float-level readout ------------------------------------------------------------------------
def profile_panel(ax, v, xr, letter, lx, ly, legend):
    pr = profiles[v]
    x = np.array(pr["x"])
    sel = np.array(pr["selected"]) / 1e3
    shp = np.array(pr["shipped"]) / 1e3
    ipl = np.array(pr["ipl"]) / 1e3
    m = (x >= xr[0]) & (x <= xr[1])
    left = IMPULSE_X[v] < 32                      # the impulse sits at the left end of the window (D) or the right (E)
    ax.axhline(THR_FLOAT / 1e3, color=INK2, lw=0.6, ls=(0, (1.5, 1.5)), zorder=1)
    ax.plot(x[m], sel[m], "-", color=C_GREEN, lw=1.3, zorder=2, label="ceil + reflect (selected)")
    ax.plot(x[m], shp[m], "--", color=C_BLUE, lw=1.1, zorder=2, label="floor + reflect (alternative)")
    ax.plot(x[m], ipl[m], "o", mfc="white", mec=INK, mew=0.8, ms=4.0, zorder=4, label="IPL's exported output")
    ax.set_xlim(xr[0] - 0.4, xr[1] + 0.4)
    ax.set_xticks(list(range(xr[0], xr[1] + 1)))
    ax.set_xlabel("x (voxel index along the padded axis)", labelpad=1.5)
    ax.set_ylabel("filter output (\u00d710\u00b3)", labelpad=1.5)
    ax.tick_params(pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(-25, 165)
    # the threshold label on the free side
    ax.text(xr[1] + 0.3 if left else xr[0] - 0.3, THR_FLOAT / 1e3 - 3, "threshold", ha="right" if left else "left",
            va="top", fontsize=6.0, color=INK2)
    ax.text(0.02, 0.99, f"impulse at x = {IMPULSE_X[v]}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5,
            fontweight="bold")
    # the separation between the two predictions where it is largest inside the window
    d = np.abs(sel - shp)
    xa = int(x[m][np.argmax(d[m])])
    lo, hi = sorted((sel[xa], shp[xa]))
    ax.annotate("", xy=(xa, hi), xytext=(xa, lo), arrowprops=dict(arrowstyle="<->", color=C_VERM, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(xa + (0.3 if left else -0.3), (lo + hi) / 2, fmt_int(round(d[xa] * 1e3)), ha="left" if left else "right",
            va="center", fontsize=6.0, color=C_VERM)
    txt = (f"max |IPL \u2212 prediction|: {pr['max_abs_diff_selected']:.3f} (selected), "
           f"{fmt_int(round(pr['max_abs_diff_shipped']))} (alternative)\n"
           f"thresholded export: {pr['seg_set_ipl']} voxels set (predicted {pr['seg_set_selected']} / {pr['seg_set_shipped']})")
    ax.text(0.99 if left else 0.01, 0.89 if left else 0.88, txt, transform=ax.transAxes, ha="right" if left else "left",
            va="top", fontsize=6.0, color=INK2, linespacing=1.15)
    panel_letter(fig, lx, ly, letter)
    if legend:
        ax.legend(loc="center right", frameon=False, handlelength=1.6, borderaxespad=0.0, bbox_to_anchor=(1.0, 0.33),
                  labelspacing=0.25, fontsize=6.0)


axD = mm_axes(fig, 106, 58, 70, 26)
profile_panel(axD, "x63i01", (0, 8), "D", 92, 51.4, legend=True)
axE = mm_axes(fig, 106, 92.5, 70, 26)
profile_panel(axE, "x63i61", (54, 62), "E", 92, 86.5, legend=False)

# ---- F: the differential test over every controlled input -------------------------------------------------
n_r, n_c = len(CANDS), len(ALL_INPUTS)
map_x, map_y, map_w = 44, 131, 120
cell_h = 1.9                                  # 6-pt row labels without touching
map_h = cell_h * n_r
axF = mm_axes(fig, map_x, map_y, map_w, map_h)
bare(axF)
axF.set_xlim(0, n_c)
axF.set_ylim(n_r, 0)
panel_letter(fig, 3, 127, "F")
fig.text(8 / FIG_W_MM, 1 - (map_y - 2.2) / FIG_H_MM,
         "Differential test over every controlled input of the run (30 synthetic, 6 blocks of a real scan) "
         "\u00d7 32 candidates", ha="left", va="bottom", fontsize=7)
for r, i in enumerate(order):
    for j in range(n_c):
        axF.add_patch(Rectangle((j, r), 1, 1, facecolor=C_REFUTE if refuted[i, j] else C_AGREE, edgecolor="white", lw=0.3))
    sel = (i == i_sel)
    axF.text(-0.3, r + 0.5, cand_label(CANDS[i]), ha="right", va="center", fontsize=6.0,
             color=C_GREEN if sel else INK, fontweight="bold" if sel else "normal")
    axF.text(n_c + 0.3, r + 0.5, str(int(n_ref[i])), ha="left", va="center", fontsize=6.0,
             color=C_GREEN if sel else INK, fontweight="bold" if sel else "normal")
axF.add_patch(Rectangle((0, order.index(i_sel)), n_c, 1, facecolor="none", edgecolor=C_GREEN, lw=0.9, zorder=3))
axF.text(n_c + 0.3, -0.4, "inputs\nrefuting", ha="left", va="bottom", fontsize=6.0, color=INK2, linespacing=1.0)
# column labels below the map, then the two families bracketed and the cell legend (figure coordinates, mm)
for j, v in enumerate(ALL_INPUTS):
    axF.text(j + 0.5, n_r + 0.3, INPUT_LABEL[v], ha="center", va="top", fontsize=6.0, rotation=90, color=INK)
colw_mm = map_w / n_c
y_br = map_y + map_h + 27.0
for j0, j1, lab in ((0, len(SYNTH), "synthetic inputs"), (len(SYNTH), n_c, "blocks of a real scan")):
    x0, x1 = map_x + (j0 + 0.15) * colw_mm, map_x + (j1 - 0.15) * colw_mm
    fig.add_artist(Line2D([x0 / FIG_W_MM, x1 / FIG_W_MM], [1 - y_br / FIG_H_MM] * 2, color=INK2, lw=0.6))
    fig.text((x0 + x1) / 2 / FIG_W_MM, 1 - (y_br + 0.8) / FIG_H_MM, lab, ha="center", va="top", fontsize=6.0, color=INK2)
lg = [Line2D([], [], marker="s", ls="none", ms=6, mfc=C_AGREE, mec="white",
             label=f"prediction matches IPL's export (max |difference| \u2264 {FLOAT_TOL:g} filter unit)"),
      Line2D([], [], marker="s", ls="none", ms=6, mfc=C_REFUTE, mec="white", label="prediction refuted")]
fig.legend(handles=lg, loc="upper left", bbox_to_anchor=(map_x / FIG_W_MM, 1 - (y_br + 4.5) / FIG_H_MM), frameon=False,
           ncol=2, handletextpad=0.4, columnspacing=1.5, fontsize=6.0)
NUM["F"]["selected_seg_differing_inputs"] = int((segd[i_sel] > 0).sum())

fig.savefig(OUT_PNG, dpi=300)
fig.savefig(OUT_SVG)
with open(OUT_NUM, "w", encoding="utf-8") as fh:
    json.dump(NUM, fh, indent=1)
say("wrote", OUT_PNG, OUT_SVG, OUT_NUM)
