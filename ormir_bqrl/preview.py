"""ormir_bqrl.preview -- the 2 x 4 mid-slice figure of a run / redo and the before / after / difference figure of
a redo's edit (matplotlib, Agg).  Both raise ImportError without matplotlib; the callers log and skip.

write_preview: the axial slice with the most SEG voxels (the old script's "best slice" rule applied to SEG).
  row 1  HU greyscale | HU + periosteal outline (ALL, green) + cortical (red) / trabecular (blue) fills
         | SEG (cortical white, trabecular grey) | text: sample, site, kind, the summary metrics with SDs (Ct.Po
         when the pore cascade ran), provenance per mask (manual ones in bold red), ORMIR-BQRL / ipldt versions
  row 2  Tb.Th (hot, inside TRAB_GOBJ) | Tb.Sp (cool) | 1/Tb.N (viridis) | Ct.Th (plasma, inside CORT_GOBJ),
         each titled with its mean in mm; colour ranges 0.05-0.40 / 0.10-1.00 / 0.20-1.50 / 0.10-3.00 mm
write_edit_preview: 1 x 3 at the slice with the most changed voxels of the edited mask: run (labelmap colours)
  | redo | difference (added green, removed magenta) with the counts in the title.
"""
from __future__ import annotations

import os

import numpy as np

# the label colours of ormir_bqrl.slicer's table (kept local: the preview has no Slicer dependency)
LABEL_COLORS = {1: (0.85, 0.25, 0.20), 2: (0.20, 0.45, 0.90)}
MAP_PANELS = (("TRAB_TH", "Tb.Th", "Tb_Th_mm", "hot", 0.05, 0.40, "G_trab"),
              ("TRAB_SP", "Tb.Sp", "Tb_Sp_mm", "cool", 0.10, 1.00, "G_trab"),
              ("TRAB_1N", "1/Tb.N", "Tb_1N_mm", "viridis", 0.20, 1.50, "G_trab"),
              ("CORT_TH", "Ct.Th", "Ct_Th_mm", "plasma", 0.10, 3.00, "G_cort"))
ADDED, REMOVED = (0.10, 0.85, 0.20), (0.90, 0.15, 0.85)


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(f"matplotlib is required for the preview figures ({exc})") from exc
    return plt


def best_slice(vol):
    """The axial slice with the most non-zero voxels (the middle slice when the volume is empty)."""
    counts = (np.asarray(vol) > 0).sum(axis=(1, 2))
    return int(np.argmax(counts)) if counts.max() > 0 else int(vol.shape[0] // 2)


def _hu_slice(loaded, z):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(loaded.hu_int16)[z].astype(np.float32)


def labels_rgb(cort2d, trab2d, background=0.0):
    """An RGB image of a labelmap slice in the Slicer table's colours."""
    rgb = np.full(cort2d.shape + (3,), background, np.float32)
    rgb[cort2d > 0] = LABEL_COLORS[1]
    rgb[trab2d > 0] = LABEL_COLORS[2]
    return rgb


def _text_lines(report):
    s = report["summary"]
    prov = report["masks"]
    fmt = lambda v, nd=4: "-" if v is None else f"{v:.{nd}f}"     # noqa: E731
    lines = [(f"{report['sample']}   site {report['site']}   {report['run']['kind']}", "black", "bold"),
             ("", "black", "normal"),
             (f"BV/TV   {fmt(s.get('BV_TV'))}", "black", "normal"),
             (f"Tb.Th   {fmt(s.get('Tb_Th_mm'))} +- {fmt(s.get('Tb_Th_sd_mm'))} mm", "black", "normal"),
             (f"Tb.Sp   {fmt(s.get('Tb_Sp_mm'))} +- {fmt(s.get('Tb_Sp_sd_mm'))} mm", "black", "normal"),
             (f"Tb.N    {fmt(s.get('Tb_N_per_mm'))} /mm", "black", "normal"),
             (f"Ct.Th   {fmt(s.get('Ct_Th_mm'))} +- {fmt(s.get('Ct_Th_sd_mm'))} mm", "black", "normal")]
    if s.get("Ct_Po") is not None:
        lines.append((f"Ct.Po   {fmt(s.get('Ct_Po'))}", "black", "normal"))
    if s.get("Tb_BMD_mgHA_cm3") is not None:
        lines.append((f"Tb.BMD  {fmt(s.get('Tb_BMD_mgHA_cm3'), 1)} mgHA/cm3", "black", "normal"))
        lines.append((f"Ct.BMD  {fmt(s.get('Ct_BMD_mgHA_cm3'), 1)} mgHA/cm3", "black", "normal"))
    lines.append(("", "black", "normal"))
    for name in ("periosteal", "cortical", "trabecular"):
        p = prov[name]
        manual = bool(p.get("manual"))
        lines.append((f"{name:<11s} {p['source']}", "darkred" if manual else "black", "bold" if manual else "normal"))
    e = report.get("edits")
    if e:
        lines.append(("", "black", "normal"))
        lines.append((f"edit: {', '.join(e['edited'])} ({e['rule']})", "darkred", "bold"))
    pr = report["product"]
    lines.append(("", "black", "normal"))
    lines.append((f"{pr['name']} {pr['version']}  ipldt {pr['ipldt_version']}  dt {report['parameters']['dt']['backend']}", "dimgray", "normal"))
    return lines


def write_preview(path, loaded, masks, segmentation, morph, report, dpi=150):
    plt = _plt()
    z = best_slice(segmentation.seg)
    hu = _hu_slice(loaded, z)
    vmin, vmax = float(np.percentile(hu, 1)), float(np.percentile(hu, 99))
    if vmax <= vmin:
        vmin, vmax = float(hu.min()), float(hu.max() + 1)
    el = float(loaded.grid.el[0])
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f"{report['product']['name']}  {report['sample']}  ({report['run']['kind']}, axial slice {z} of {segmentation.seg.shape[0]})",
                 fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.imshow(hu, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("HU", fontsize=9)

    ax = axes[0, 1]
    ax.imshow(hu, cmap="gray", vmin=vmin, vmax=vmax)
    c2, t2 = masks.cort[z], masks.trab[z]
    over = np.zeros(hu.shape + (4,), np.float32)
    over[c2] = LABEL_COLORS[1] + (0.45,)
    over[t2] = LABEL_COLORS[2] + (0.35,)
    ax.imshow(over, interpolation="nearest")
    if masks.ALL[z].any():
        ax.contour(masks.ALL[z].astype(np.float32), levels=[0.5], colors=["lime"], linewidths=0.7)
    ax.set_title("periosteal contour (green) | cortical (red) | trabecular (blue)", fontsize=9)

    ax = axes[0, 2]
    seg2 = segmentation.seg[z]
    img = np.zeros(hu.shape, np.float32)
    img[seg2 == 126] = 0.55
    img[seg2 == 127] = 1.0
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("SEG (cortical white, trabecular grey)", fontsize=9)

    ax = axes[0, 3]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    y = 0.96
    for text, color, weight in _text_lines(report):
        ax.text(0.02, y, text, fontsize=9, family="monospace", color=color, fontweight=weight, va="top", ha="left")
        y -= 0.055
    ax.set_title("summary", fontsize=9)

    for ax, (name, label, mkey, cmap, lo, hi, gkey) in zip(axes[1], MAP_PANELS):
        arr = morph.maps.get(name)
        G = getattr(masks, gkey)
        if arr is None or G is None:
            ax.imshow(np.zeros(hu.shape), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{label}: not computed", fontsize=9)
            continue
        mm = arr[z].astype(np.float32) * el
        ax.imshow(np.zeros(hu.shape), cmap="gray", vmin=0, vmax=1)
        disp = np.ma.masked_where((G[z] == 0) | (mm <= 0), mm)
        im = ax.imshow(disp, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        mean = morph.metrics.get(mkey)
        ax.set_title(f"{label}  mean = {mean:.4f} mm" if mean is not None else label, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(path)


def write_edit_preview(path, before, after, edited, dpi=150):
    """before / after: ormir_bqrl.stages.Masks; edited: 'periosteal' | 'cortical' | 'trabecular'."""
    plt = _plt()
    attr = {"periosteal": "ALL", "cortical": "cort", "trabecular": "trab"}[edited]
    a, b = np.asarray(getattr(before, attr), bool), np.asarray(getattr(after, attr), bool)
    changed = a ^ b
    z = best_slice(changed) if changed.any() else best_slice(after.ALL)
    added, removed = int((b & ~a).sum()), int((a & ~b).sum())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(labels_rgb(before.cort[z], before.trab[z]), interpolation="nearest")
    axes[0].set_title("run (cortical red, trabecular blue)", fontsize=9)
    axes[1].imshow(labels_rgb(after.cort[z], after.trab[z]), interpolation="nearest")
    axes[1].set_title("redo", fontsize=9)
    diff = np.full(a[z].shape + (3,), 0.0, np.float32)
    diff[after.ALL[z]] = (0.25, 0.25, 0.25)
    diff[(b & ~a)[z]] = ADDED
    diff[(a & ~b)[z]] = REMOVED
    axes[2].imshow(diff, interpolation="nearest")
    axes[2].set_title(f"{edited}: {added:,d} voxels added (green), {removed:,d} removed (magenta); slice {z}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(path)


__all__ = ["write_preview", "write_edit_preview", "best_slice", "labels_rgb", "LABEL_COLORS", "MAP_PANELS"]
