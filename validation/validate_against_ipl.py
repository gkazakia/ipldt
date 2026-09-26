"""validate_against_ipl -- voxel-for-voxel validation of ipldt against Scanco IPL V5.42 on the
21-patella HR-pQCT cohort (configuration A of the paper).

For every subject the five Script-32 maps are recomputed with ipldt from IPL's own inputs and
compared with the maps IPL wrote, aligned by global voxel position:

  key        ipldt call                                            IPL map       grid
  TbTh       dt_thickness(SEG,      gobj = rendered TRAB_MASK)     TRAB_TH       SEG   (the REPORTED Tb.Th here)
  TbSp       dt_spacing  (SEG,      gobj = rendered TRAB_MASK)     TRAB_SP       SEG
  TbN        dt_number   (SEG,      gobj = rendered TRAB_MASK)     TRAB_1N       SEG   (Tb.N = 1 / mean map)
  CtTh       dt_thickness(CORT_MASK raster, gobj = rendered CORT_MASK)  CORT_TH  CORT_MASK
  TbTh_old   dt_thickness(TRAB_SEG, gobj = rendered TRAB_MASK)     TRAB_TH_old   SEG   (the scripts' definition)

WHICH Tb.Th IS REPORTED, AND THE TRAP IN IPL'S FILENAMES.  Scripts 32, 33 (byte-identical command
files) and 34 all write TRAB_SEG as IPL_FNAME5 and then run /dt_thickness on it, cropped to the
trabecular gobj (32/33 lines 452 / 708 / 710; 34 lines 190 / 446 / 448), so the scripts' Tb.Th is
dt_thickness(TRAB_SEG) -- and for this cohort that is IPL's file <base>_TRAB_TH_old, NOT <base>_TRAB_TH.
The filenames are the REVERSE of the intuitive reading: TRAB_TH_old holds the Script-32 (TRAB_SEG) map,
TRAB_TH holds the whole-SEG one.  Misreading them that way round is what misled this project, so nothing
here is called 'old' or 'new' any more: the labels are 'Tb.Th (whole SEG, as ORMIR-BQRL ships)' -- the
Tb.Th reported first here, because it is compared with IPL's delivered <base>_TRAB_TH and is what the
shipped pipeline computes -- and 'Tb.Th (TRAB_SEG, Scripts 32/33/34)', the scripts' own definition,
reported beside it and compared with IPL's <base>_TRAB_TH_old.  Both maps are computed and cached, under
the historical keys TbTh and TbTh_old (the keys are also the .npy cache filenames and the records.json /
joint_hist_*.npz identifiers).  The validation deliberately compares each cohort against the definition
IPL's own delivered file used rather than forcing one designation on both.

SEG = cortical (127) + trabecular (126) bone; TRAB_SEG = SEG inside the rendered trabecular
contour; the masks are rendered on their own grid with IPL's contour rules (ipldt.contour)
and pasted onto the target grid by global position.  Maps are integer sphere diameters in
voxels; IPL's statistics (mean over the non-zero voxels) are reported in mm at 0.0607 mm.

Every computed map is cached as .npy under validation/cache/ (gitignored) so the notebook
and the CLI rerun instantly; the comparison itself works from joint histograms
(ours x IPL value counts), from which mismatch counts, supports, means and the voxel-wise
regression follow exactly.

CLI (cohort validation)
    python validate_against_ipl.py                       # everything: compute missing maps, tables, figures, summary
                                                         # (into results/against_ipl_run/; the published
                                                         # results/records.json needs --allow-overwrite-shipped)
    python validate_against_ipl.py --subjects PFJ-0be66a_R   # one subject
    python validate_against_ipl.py --compute-only        # fill the cache and stop
    python validate_against_ipl.py --no-compute          # tables + figures from the cache only
    python validate_against_ipl.py --backend cpu         # force the CPU path (results are identical)

The reviewer bundle (IPL's products of one patella and a check script) is deposited separately (see README.md).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datapaths import lab_path  # noqa: E402  (non-public data roots: validation/datapaths.py)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import ipldt  # noqa: E402

DATA_ROOT = os.environ.get("IPLDT_DATA_ROOT") or lab_path("PFJOA/XCT_masks_full_grab")
CACHE_DIR = os.path.join(HERE, "cache")
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
RUN_DIR = os.path.join(RESULTS_DIR, "against_ipl_run")        # default output of a CLI run (not a published folder)
RUN_FIG_DIR = os.path.join(RUN_DIR, "figures")


def published_dirs():
    """The published result directories: validation/results/ itself (records.json, sample_means.csv) and the record
    sets the paper's numbers are built from."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from result_sets import OSLH_AUTO, OSLH_NOEDIT
    return [os.path.join(RESULTS_DIR, d) for d in ("from_ipl_contour_ceil_dt", "from_ipl_contour_ceil", "porosity_AB_patella",
                                                   "patella_bvtv_A", OSLH_AUTO, OSLH_NOEDIT)]


def refuse_published(results, allow, flag="--allow-overwrite-shipped"):
    """True (after printing why) when `results` is validation/results/ itself or lies in a published directory."""
    p = os.path.normcase(os.path.abspath(results))
    top = os.path.normcase(os.path.abspath(RESULTS_DIR))
    hit = p == top or any(p == os.path.normcase(os.path.abspath(d)) or p.startswith(os.path.normcase(os.path.abspath(d)) + os.sep)
                          for d in published_dirs())
    if hit and not allow:
        print(f"REFUSED: --results {results} would overwrite published records under validation/results/; pass "
              f"another folder, or {flag} if replacing them is intended. Nothing was written.", file=sys.stderr)
    return hit and not allow
VOXEL_MM = 0.0607                      # XtremeCT II el_size (the AIM headers carry 0.06070 +- 1e-6)

# --------------------------------------------------------------------------------------- metrics
class Metric:
    def __init__(self, key, label, ipl_suffix, grid, what, map_unit, sample_label, sample_unit):
        self.key, self.label, self.ipl_suffix, self.grid = key, label, ipl_suffix, grid
        self.what, self.map_unit = what, map_unit
        self.sample_label, self.sample_unit = sample_label, sample_unit

    def __repr__(self):
        return f"Metric({self.key})"


TBTH_TSEG = "Tb.Th (TRAB_SEG, Scripts 32/33/34)"     # reported beside it        -- IPL's file TRAB_TH_old
TBTH_SEG = "Tb.Th (whole SEG, as ORMIR-BQRL ships)"  # the reported Tb.Th here -- IPL's file TRAB_TH
METRICS = {
    "TbTh": Metric("TbTh", TBTH_SEG, "TRAB_TH", "seg",
                   "dt_thickness(SEG, gobj = trabecular contour)", f"{TBTH_SEG} map (mm)", TBTH_SEG, "mm"),
    "TbSp": Metric("TbSp", "Tb.Sp", "TRAB_SP", "seg",
                   "dt_spacing(SEG, gobj = trabecular contour)", "Tb.Sp map (mm)", "Tb.Sp", "mm"),
    "TbN": Metric("TbN", "Tb.N", "TRAB_1N", "seg",
                  "dt_number(SEG, gobj = trabecular contour)", "1/Tb.N map (mm)", "Tb.N", "1/mm"),
    "CtTh": Metric("CtTh", "Ct.Th", "CORT_TH", "cort",
                   "dt_thickness(CORT_MASK, gobj = cortical contour)", "Ct.Th map (mm)", "Ct.Th", "mm"),
    "TbTh_old": Metric("TbTh_old", TBTH_TSEG, "TRAB_TH_old", "seg",
                       "dt_thickness(TRAB_SEG, gobj = trabecular contour)", f"{TBTH_TSEG} map (mm)", TBTH_TSEG, "mm"),
}
# the reported Tb.Th (the whole-SEG one, compared with IPL's delivered TRAB_TH) leads; the scripts' TRAB_SEG definition
# is reported last, beside it
METRIC_ORDER = ["TbTh", "TbSp", "TbN", "CtTh", "TbTh_old"]
TBTH_LEGEND = (
    f"Tb.Th is reported under BOTH definitions.  **{TBTH_SEG}** is dt_thickness of the whole SEG cropped to the "
    "trabecular gobj, what the shipped pipeline computes, compared with IPL's delivered file `<base>_TRAB_TH`; it is the "
    f"one reported first here.  **{TBTH_TSEG}** is what Scanco's evaluation scripts compute -- /dt_thickness on TRAB_SEG "
    "(IPL_FNAME5) cropped to the same gobj -- and it is compared with IPL's file `<base>_TRAB_TH_old`.  NOTE the trap: for "
    "this cohort IPL's filenames are the reverse of the intuitive reading -- TRAB_TH_old is the Script-32 (TRAB_SEG) map "
    "and TRAB_TH is the whole-SEG one.  The record / CSV keys stay `TbTh_old` (TRAB_SEG) and `TbTh` (whole SEG) for continuity."
)
IPL_PARAMS = dict(ridge_epsilon=0.9, assign_epsilon=0.5, peel_iter=-1, version=3, suppress_boundary=2)

_T0 = time.time()


def log(*a):
    print(f"[{time.time() - _T0:7.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------------------- data access
def list_subjects(root=DATA_ROOT):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def subject_base(subject, root=DATA_ROOT):
    """The L/C number of a subject folder, from its <base>_SEG_decompressed.AIM (not TRAB_/CORT_)."""
    hits = [q for q in glob.glob(os.path.join(root, subject, "*_SEG_decompressed.AIM"))
            if "TRAB_" not in os.path.basename(q) and "CORT_" not in os.path.basename(q)]
    if len(hits) != 1:
        raise FileNotFoundError(f"{subject}: expected one *_SEG_decompressed.AIM, found {hits}")
    return os.path.basename(hits[0]).replace("_SEG_decompressed.AIM", "")


class Subject:
    """Lazy access to one subject's IPL inputs and outputs (all on disk as decompressed AIMs)."""

    def __init__(self, subject, root=DATA_ROOT, cache_dir=CACHE_DIR):
        self.subject = subject
        self.root = root
        self.cache_dir = cache_dir
        self.folder = os.path.join(root, subject)
        self.base = subject_base(subject, root)
        self._aims = {}
        self._gobj = {}

    def path(self, suffix):
        return os.path.join(self.folder, f"{self.base}_{suffix}_decompressed.AIM")

    def aim(self, suffix):
        if suffix not in self._aims:
            self._aims[suffix] = ipldt.read_aim(self.path(suffix))
        return self._aims[suffix]

    def grid(self, which):
        """(dim, pos) of the SEG grid or the CORT_MASK grid."""
        a = self.aim("SEG" if which == "seg" else "CORT_MASK")
        return a["dim"], a["pos"]

    def el_size_mm(self):
        return tuple(float(v) for v in self.aim("SEG")["el_size_mm"])

    def on_grid(self, suffix, which):
        dim, pos = self.grid(which)
        return ipldt.align_to(self.aim(suffix), dim, pos)

    # ---- inputs
    def seg(self):
        return self.aim("SEG")["data"] > 0

    def trab_seg(self):
        return self.on_grid("TRAB_SEG", "seg") > 0

    def cort_mask(self):
        return self.aim("CORT_MASK")["data"] > 0

    def gobj(self, which):
        """The rendered contour: trabecular mask on the SEG grid, or cortical mask on its own grid.
        Rendering takes ~10 s, so the raster is cached (bool .npy)."""
        if which in self._gobj:
            return self._gobj[which]
        f = os.path.join(self.cache_dir, f"{self.subject}_{self.base}_gobj_{which}.npy")
        if os.path.exists(f):
            G = np.load(f)
        else:
            os.makedirs(self.cache_dir, exist_ok=True)
            if which == "trab":
                m = self.aim("TRAB_MASK")
                R = ipldt.render_volume(m["data"] > 0)                      # on the mask's own grid
                dim, pos = self.grid("seg")
                G = ipldt.align_to(dict(data=R, dim=m["dim"], pos=m["pos"]), dim, pos)
            elif which == "cort":
                G = ipldt.render_volume(self.cort_mask())
            else:
                raise ValueError(which)
            np.save(f, G)
        self._gobj[which] = G
        return G

    # ---- IPL's map for a metric, on that metric's grid
    def ipl_map(self, metric):
        m = METRICS[metric] if isinstance(metric, str) else metric
        return self.on_grid(m.ipl_suffix, m.grid).astype(np.int16)

    def cache_path(self, metric, ext=".npy"):
        key = metric if isinstance(metric, str) else metric.key
        return os.path.join(self.cache_dir, f"{self.subject}_{self.base}_{key}{ext}")

    def release(self):
        self._aims.clear()
        self._gobj.clear()


# --------------------------------------------------------------------------------------- computing
def compute_map(subj: Subject, metric, backend="auto"):
    """Run the ipldt call that corresponds to `metric` on IPL's inputs; returns (map int16, report)."""
    m = METRICS[metric] if isinstance(metric, str) else metric
    t0 = time.time()
    if m.grid == "seg":
        G = {"rendered": subj.gobj("trab")}
        if m.key == "TbTh":
            r = ipldt.dt_thickness(subj.seg(), gobj=G, voxel_size_mm=VOXEL_MM, backend=backend, **IPL_PARAMS)
        elif m.key == "TbTh_old":
            r = ipldt.dt_thickness(subj.trab_seg(), gobj=G, voxel_size_mm=VOXEL_MM, backend=backend, **IPL_PARAMS)
        elif m.key == "TbSp":
            r = ipldt.dt_spacing(subj.seg(), gobj=G, voxel_size_mm=VOXEL_MM, backend=backend, **IPL_PARAMS)
        elif m.key == "TbN":
            r = ipldt.dt_number(subj.seg(), gobj=G, voxel_size_mm=VOXEL_MM, backend=backend, **IPL_PARAMS)
        else:
            raise ValueError(m.key)
    elif m.key == "CtTh":
        cm = subj.cort_mask()
        r = ipldt.dt_thickness(cm, gobj={"rendered": subj.gobj("cort")}, voxel_size_mm=VOXEL_MM, backend=backend, **IPL_PARAMS)
    else:
        raise ValueError(m.key)
    rep = dict(r.report)
    rep.update(n_centres=int(r.centres.sum()), compute_time_s=round(time.time() - t0, 1),
               backend=ipldt.gpu.resolve_backend(backend), el_size_mm=subj.el_size_mm())
    return r.map.astype(np.int16), rep


def cached_map(subj: Subject, metric, compute=True, backend="auto", verbose=True):
    """The cached ipldt map for (subject, metric); computed and cached when missing (if compute)."""
    f = subj.cache_path(metric)
    fr = subj.cache_path(metric, "_report.json")
    if os.path.exists(f):
        rep = json.load(open(fr)) if os.path.exists(fr) else {}
        return np.load(f), rep
    if not compute:
        raise FileNotFoundError(f"{f} is not cached; run validate_against_ipl.py (or pass compute=True)")
    os.makedirs(subj.cache_dir, exist_ok=True)
    out, rep = compute_map(subj, metric, backend)
    np.save(f, out)
    json.dump(rep, open(fr, "w"), indent=1)
    if verbose:
        log(f"{subj.subject} {metric:<8s} computed in {rep['compute_time_s']:6.1f} s ({rep['backend']}), "
            f"{rep['n_centres']:,d} centres, cached -> {os.path.basename(f)}")
    return out, rep


# --------------------------------------------------------------------------------------- comparing
def joint_histogram(ours, ipl):
    """counts[i, j] = number of voxels with ours == i and IPL == j (both maps int16 >= 0)."""
    K = int(max(int(ours.max()), int(ipl.max()))) + 1
    idx = ours.astype(np.int32) * np.int32(K)
    idx += ipl.astype(np.int32)
    return np.bincount(idx.ravel(), minlength=K * K).reshape(K, K)


def summarize_joint(J, voxel_mm=VOXEL_MM):
    """Mismatch counts, supports and means, exactly, from the joint histogram."""
    K = J.shape[0]
    i = np.arange(K, dtype=np.float64)
    total = int(J.sum())
    agree = int(np.trace(J))
    support_ours = total - int(J[0, :].sum())
    support_ipl = total - int(J[:, 0].sum())
    row = J.sum(1).astype(np.float64)              # ours value counts
    col = J.sum(0).astype(np.float64)              # IPL value counts
    mean_ours = float((row * i).sum() / support_ours) if support_ours else 0.0
    mean_ipl = float((col * i).sum() / support_ipl) if support_ipl else 0.0
    # statistics over the voxels WITH a value: the zero bin (row[0] / col[0]) is excluded from the SD sums
    sd_ours = float(np.sqrt((row[1:] * (i[1:] - mean_ours) ** 2).sum() / support_ours)) if support_ours else 0.0
    sd_ipl = float(np.sqrt((col[1:] * (i[1:] - mean_ipl) ** 2).sum() / support_ipl)) if support_ipl else 0.0
    nonzero = total - int(J[0, 0])
    return dict(
        n_voxels=total, mismatches=total - agree,
        ours_only=int(J[1:, 0].sum()), ipl_only=int(J[0, 1:].sum()),
        both_nonzero_differ=total - agree - int(J[1:, 0].sum()) - int(J[0, 1:].sum()),
        support_ours=support_ours, support_ipl=support_ipl,
        nonzero_either=nonzero,
        exact_fraction_nonzero=float((agree - int(J[0, 0])) / nonzero) if nonzero else 1.0,
        max_ours=int(np.nonzero(row)[0].max()), max_ipl=int(np.nonzero(col)[0].max()),
        mean_ours_vox=mean_ours, mean_ipl_vox=mean_ipl,
        sd_ours_vox=sd_ours, sd_ipl_vox=sd_ipl,
        mean_ours_mm=mean_ours * voxel_mm, mean_ipl_mm=mean_ipl * voxel_mm,
    )


def mismatch_locations(ours, ipl, limit=1000):
    """(z, y, x, ours, ipl) rows of the mismatching voxels, first `limit` in raster order."""
    zyx = np.argwhere(ours != ipl)[:limit]
    return [(int(z), int(y), int(x), int(ours[z, y, x]), int(ipl[z, y, x])) for z, y, x in zyx]


def compare_subject(subj: Subject, metric, compute=True, backend="auto", locations=True):
    """Compare ipldt's (cached) map with IPL's for one subject and metric."""
    m = METRICS[metric] if isinstance(metric, str) else metric
    ours, rep = cached_map(subj, m.key, compute=compute, backend=backend)
    ipl = subj.ipl_map(m)
    if ours.shape != ipl.shape:
        raise ValueError(f"{subj.subject} {m.key}: grid mismatch {ours.shape} vs {ipl.shape}")
    J = joint_histogram(ours, ipl)
    rec = dict(subject=subj.subject, base=subj.base, metric=m.key, grid=m.grid,
               dim_x=ipl.shape[2], dim_y=ipl.shape[1], dim_z=ipl.shape[0])
    rec.update(summarize_joint(J))
    rec.update(compute_time_s=rep.get("compute_time_s"), backend=rep.get("backend"),
               n_centres=rep.get("n_centres"), el_size_mm=subj.el_size_mm()[0])
    if locations and rec["mismatches"]:
        rec["mismatch_zyx"] = mismatch_locations(ours, ipl)
    return rec, J


def run_all(subjects=None, metrics=METRIC_ORDER, compute=True, backend="auto", results_dir=RESULTS_DIR,
            cache_dir=CACHE_DIR, root=DATA_ROOT, verbose=True, save=True):
    """Compare every (subject, metric); returns (records, joints) and writes results/*.

    records: list of dicts (one per subject and metric);  joints: {metric: {subject: KxK counts}}."""
    subjects = subjects or list_subjects(root)
    records, joints = [], {k: {} for k in metrics}
    for s in subjects:
        subj = Subject(s, root, cache_dir)
        for k in metrics:
            rec, J = compare_subject(subj, k, compute=compute, backend=backend)
            records.append(rec)
            joints[k][s] = J
            if verbose:
                log(f"{s:<9s} {subj.base:<9s} {METRICS[k].label:<12s} voxels {rec['n_voxels']:>12,d}  "
                    f"mismatches {rec['mismatches']:>6,d}  support IPL {rec['support_ipl']:>11,d} ours {rec['support_ours']:>11,d}  "
                    f"mean IPL {rec['mean_ipl_vox']:.4f} ours {rec['mean_ours_vox']:.4f} vox")
        subj.release()
    if save:
        save_results(records, joints, results_dir)
    return records, joints


# --------------------------------------------------------------------------------------- tables
def records_to_frame(records):
    import pandas as pd
    cols = ["subject", "base", "metric", "grid", "dim_x", "dim_y", "dim_z", "n_voxels", "mismatches", "ours_only",
            "ipl_only", "both_nonzero_differ", "support_ipl", "support_ours", "nonzero_either",
            "exact_fraction_nonzero", "max_ipl", "max_ours", "mean_ipl_vox", "mean_ours_vox", "sd_ipl_vox",
            "sd_ours_vox", "mean_ipl_mm", "mean_ours_mm", "n_centres", "compute_time_s", "backend", "el_size_mm"]
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in records])
    df["metric"] = pd.Categorical(df["metric"], [k for k in METRIC_ORDER if k in set(df["metric"])])
    return df.sort_values(["subject", "metric"]).reset_index(drop=True)


def sample_table(records):
    """Per-subject sample statistics in physical units: IPL vs ours for each metric
    (Tb.Th / Tb.Sp / Ct.Th / Tb.Th old = mean map in mm; Tb.N = 1 / mean 1/Tb.N map)."""
    import pandas as pd
    rows = {}
    for r in records:
        m = METRICS[r["metric"]]
        d = rows.setdefault(r["subject"], {"subject": r["subject"], "base": r["base"]})
        if m.key == "TbN":
            d[f"{m.key}_ipl"] = 1.0 / r["mean_ipl_mm"] if r["mean_ipl_mm"] else np.nan
            d[f"{m.key}_ours"] = 1.0 / r["mean_ours_mm"] if r["mean_ours_mm"] else np.nan
        else:
            d[f"{m.key}_ipl"] = r["mean_ipl_mm"]
            d[f"{m.key}_ours"] = r["mean_ours_mm"]
        d[f"{m.key}_mismatches"] = r["mismatches"]
    return pd.DataFrame(sorted(rows.values(), key=lambda d: d["subject"])).reset_index(drop=True)


def _md_table(df, floatfmt="{:.6f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            elif isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):,d}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_mismatch_table(records, results_dir=RESULTS_DIR):
    import pandas as pd
    os.makedirs(results_dir, exist_ok=True)
    df = records_to_frame(records)
    df.to_csv(os.path.join(results_dir, "mismatch_table.csv"), index=False)
    metrics = [k for k in METRIC_ORDER if k in set(df["metric"].astype(str))]
    # wide views for the markdown
    wide_mm = df.pivot(index="subject", columns="metric", values="mismatches")[metrics]
    wide_mm.columns = [METRICS[k].label for k in metrics]
    wide_mm.insert(0, "base", df.drop_duplicates("subject").set_index("subject")["base"])
    wide_mm["voxels (SEG grid)"] = df[df.metric == "TbTh"].set_index("subject")["n_voxels"] if "TbTh" in metrics else df.groupby("subject")["n_voxels"].max()
    wide_mm = wide_mm.reset_index()
    tot = {"subject": "TOTAL", "base": ""}
    for k in metrics:
        tot[METRICS[k].label] = int(df[df.metric == k]["mismatches"].sum())
    tot["voxels (SEG grid)"] = int(wide_mm["voxels (SEG grid)"].sum())
    wide_mm = pd.concat([wide_mm, pd.DataFrame([tot])], ignore_index=True)

    parts = ["# ipldt vs IPL: voxel-for-voxel mismatch table", "",
             f"{df.subject.nunique()} subjects, {len(metrics)} maps each; every map compared over its whole grid "
             f"(SEG grid for the trabecular maps, CORT_MASK grid for Ct.Th), aligned by global voxel position.",
             "", TBTH_LEGEND,
             "", "## Mismatching voxels per subject and map", "", _md_table(wide_mm), ""]
    for k in metrics:
        m = METRICS[k]
        sub = df[df.metric == k][["subject", "base", "n_voxels", "support_ipl", "support_ours", "mismatches",
                                   "ours_only", "ipl_only", "both_nonzero_differ", "max_ipl", "max_ours",
                                   "mean_ipl_vox", "mean_ours_vox", "mean_ipl_mm", "mean_ours_mm", "compute_time_s"]]
        sub = sub.rename(columns={"n_voxels": "voxels", "support_ipl": "support IPL", "support_ours": "support ours",
                                  "ours_only": "ours>0, IPL 0", "ipl_only": "IPL>0, ours 0",
                                  "both_nonzero_differ": "both>0, differ", "max_ipl": "max IPL", "max_ours": "max ours",
                                  "mean_ipl_vox": "mean IPL (vox)", "mean_ours_vox": "mean ours (vox)",
                                  "mean_ipl_mm": "mean IPL (mm)", "mean_ours_mm": "mean ours (mm)",
                                  "compute_time_s": "ipldt time (s)"})
        parts += [f"## {m.label}: {m.what} vs IPL {m.ipl_suffix}", "", _md_table(sub, "{:.4f}"), ""]
    # mismatch locations (slice index) for the few non-zero cases
    loc = [r for r in records if r.get("mismatch_zyx")]
    if loc:
        parts += ["## Where the mismatching voxels are", "",
                  "| subject | map | mismatches | slices z (count) | (ours, IPL) value pairs (count) |", "|---|---|---|---|---|"]
        for r in loc:
            zs = {}
            pairs = {}
            for z, y, x, o, i in r["mismatch_zyx"]:
                zs[z] = zs.get(z, 0) + 1
                pairs[(o, i)] = pairs.get((o, i), 0) + 1
            zs_s = ", ".join(f"{z} ({n})" for z, n in sorted(zs.items()))
            pr_s = ", ".join(f"({o},{i}) x{n}" for (o, i), n in sorted(pairs.items()))
            parts.append(f"| {r['subject']} | {METRICS[r['metric']].label} | {r['mismatches']} | {zs_s} | {pr_s} |")
        parts.append("")
    open(os.path.join(results_dir, "mismatch_table.md"), "w", encoding="utf-8").write("\n".join(parts))
    return df


def save_results(records, joints, results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    json.dump(records, open(os.path.join(results_dir, "records.json"), "w"), indent=1)
    for k, d in joints.items():
        np.savez_compressed(os.path.join(results_dir, f"joint_hist_{k}.npz"), **d)
    write_mismatch_table(records, results_dir)
    sample_table(records).to_csv(os.path.join(results_dir, "sample_means.csv"), index=False)


def load_results(results_dir=RESULTS_DIR):
    records = json.load(open(os.path.join(results_dir, "records.json")))
    joints = {}
    for k in METRIC_ORDER:
        f = os.path.join(results_dir, f"joint_hist_{k}.npz")
        if os.path.exists(f):
            with np.load(f) as z:
                joints[k] = {s: z[s] for s in z.files}
    return records, joints


# --------------------------------------------------------------------------------------- statistics
def pooled_joint(joint_by_subject):
    K = max(J.shape[0] for J in joint_by_subject.values())
    P = np.zeros((K, K), np.int64)
    for J in joint_by_subject.values():
        P[:J.shape[0], :J.shape[1]] += J
    return P


def voxelwise_stats(J, voxel_mm=VOXEL_MM):
    """Weighted least squares of ours on IPL over the voxels where either map is non-zero
    (exact, from the joint histogram), plus the exact-agreement fraction."""
    K = J.shape[0]
    v = np.arange(K, dtype=np.float64) * voxel_mm
    W = J.astype(np.float64).copy()
    W[0, 0] = 0.0
    n = W.sum()
    y = v[:, None]                     # ours (rows)
    x = v[None, :]                     # IPL (cols)
    mx = (W * x).sum() / n
    my = (W * y).sum() / n
    sxx = (W * (x - mx) ** 2).sum()
    syy = (W * (y - my) ** 2).sum()
    sxy = (W * (x - mx) * (y - my)).sum()
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = sxy ** 2 / (sxx * syy)
    agree = float(np.trace(W))
    return dict(n_voxels=int(n), exact_fraction=agree / n, mismatches=int(n - agree),
                slope=float(slope), intercept_mm=float(intercept), r2=float(r2),
                mean_ipl_mm=float(mx), mean_ours_mm=float(my))


def ols(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).sum()
    syy = ((y - my) ** 2).sum()
    sxy = ((x - mx) * (y - my)).sum()
    slope = sxy / sxx if sxx > 0 else np.nan
    intercept = my - slope * mx
    r2 = sxy ** 2 / (sxx * syy) if sxx > 0 and syy > 0 else (1.0 if np.allclose(x, y) else np.nan)
    return dict(slope=float(slope), intercept=float(intercept), r2=float(r2), n=int(x.size))


def icc_2_1(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement (Shrout & Fleiss)."""
    data = np.stack([np.asarray(x, float), np.asarray(y, float)], 1)
    n, k = data.shape
    if n < 2:
        return np.nan
    grand = data.mean()
    ssr = k * ((data.mean(1) - grand) ** 2).sum()
    ssc = n * ((data.mean(0) - grand) ** 2).sum()
    sse = ((data - grand) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


def bland_altman(x, y):
    """x = IPL, y = ours: bias = mean(y - x), 1.96 SD limits of agreement."""
    d = np.asarray(y, float) - np.asarray(x, float)
    m = (np.asarray(x, float) + np.asarray(y, float)) / 2
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    bias = float(d.mean())
    return dict(bias=bias, sd=sd, loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd,
                max_abs_diff=float(np.abs(d).max()), mean=m, diff=d)


def summary_stats(records, joints, metrics=METRIC_ORDER, voxel_mm=VOXEL_MM):
    """Everything the summary table and the figures need, per metric."""
    st = sample_table(records)
    out = {}
    for k in metrics:
        if k not in joints:
            continue
        m = METRICS[k]
        x = st[f"{k}_ipl"].to_numpy()
        y = st[f"{k}_ours"].to_numpy()
        P = pooled_joint(joints[k])
        out[k] = dict(metric=m, sample_ipl=x, sample_ours=y, subjects=st["subject"].tolist(),
                      voxel=voxelwise_stats(P, voxel_mm), pooled_joint=P,
                      ols=ols(x, y), icc=icc_2_1(x, y), ba=bland_altman(x, y),
                      ipl_mean=float(x.mean()), ipl_sd=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                      ours_mean=float(y.mean()), ours_sd=float(y.std(ddof=1)) if y.size > 1 else 0.0,
                      total_mismatches=int(sum(r["mismatches"] for r in records if r["metric"] == k)),
                      total_voxels=int(sum(r["n_voxels"] for r in records if r["metric"] == k)),
                      subjects_exact=int(sum(1 for r in records if r["metric"] == k and r["mismatches"] == 0)),
                      n_subjects=int(sum(1 for r in records if r["metric"] == k)))
    return out


def write_summary(stats, path=os.path.join(RESULTS_DIR, "summary.md")):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    L = ["# ipldt vs Scanco IPL V5.42: summary", "",
         "Voxel-for-voxel and sample-wise agreement of the ipldt maps with the maps IPL wrote for the same inputs "
         f"(Script 32 parameters: ridge_epsilon 0.9, assign_epsilon 0.5, peel_iter -1, version 3, suppress_boundary 2; "
         f"voxel size {VOXEL_MM} mm).  Tb.N = 1 / mean(1/Tb.N map).  ICC(2,1): two-way random effects, absolute "
         "agreement, single measurement.  Limits of agreement: bias +- 1.96 SD of the per-subject differences (ours - IPL).",
         "", TBTH_LEGEND,
         "", "## Voxel-wise agreement (all subjects pooled, whole grids)", "",
         "| map | subjects exact | voxels compared | mismatching voxels | exact agreement (voxels with a value) | slope | intercept (mm) | R^2 |",
         "|---|---|---|---|---|---|---|---|"]
    for k, s in stats.items():
        v = s["voxel"]
        L.append(f"| {s['metric'].label} | {s['subjects_exact']}/{s['n_subjects']} | {s['total_voxels']:,d} | {s['total_mismatches']:,d} "
                 f"| {100 * v['exact_fraction']:.6f}% ({v['n_voxels']:,d} voxels) | {v['slope']:.6f} | {v['intercept_mm']:.2e} | {v['r2']:.6f} |")
    L += ["", "## Sample-wise agreement (per-subject values, n = number of subjects)", "",
          "| metric | unit | IPL mean +- SD | ipldt mean +- SD | mean difference (ours - IPL) +- SD | max abs difference | limits of agreement | R^2 | ICC(2,1) | slope | intercept |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in stats.items():
        m, ba, o = s["metric"], s["ba"], s["ols"]
        L.append(f"| {m.sample_label} | {m.sample_unit} | {s['ipl_mean']:.5f} +- {s['ipl_sd']:.5f} | {s['ours_mean']:.5f} +- {s['ours_sd']:.5f} "
                 f"| {ba['bias']:+.2e} +- {ba['sd']:.2e} | {ba['max_abs_diff']:.2e} | [{ba['loa_low']:+.2e}, {ba['loa_high']:+.2e}] "
                 f"| {o['r2']:.6f} | {s['icc']:.6f} | {o['slope']:.6f} | {o['intercept']:.2e} |")
    L += ["", "## Per-subject values", ""]
    first = next(iter(stats.values()))
    hdr = "| subject | " + " | ".join(f"{s['metric'].sample_label} IPL | {s['metric'].sample_label} ipldt" for s in stats.values()) + " |"
    L += [hdr, "|" + "---|" * (1 + 2 * len(stats))]
    for i, sub in enumerate(first["subjects"]):
        cells = []
        for s in stats.values():
            cells += [f"{s['sample_ipl'][i]:.5f}", f"{s['sample_ours'][i]:.5f}"]
        L.append(f"| {sub} | " + " | ".join(cells) + " |")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return path


# --------------------------------------------------------------------------------------- figures
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e2"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("ipldt_blue", SEQ)


def plot_voxelwise(ax, J, metric, voxel_mm=VOXEL_MM, title=None):
    """2-D histogram (log colour) of ipldt vs IPL map values over the voxels where either is non-zero."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    m = METRICS[metric] if isinstance(metric, str) else metric
    st = voxelwise_stats(J, voxel_mm)
    K = J.shape[0]
    H = J.astype(np.float64).copy()
    H[0, 0] = np.nan
    H[H == 0] = np.nan
    edges = (np.arange(K + 1) - 0.5) * voxel_mm
    pc = ax.pcolormesh(edges, edges, np.ma.masked_invalid(H), cmap=_seq_cmap(), norm=LogNorm(vmin=1, vmax=np.nanmax(H)),
                       rasterized=True)
    lim = (0, K * voxel_mm)
    ax.plot(lim, lim, "--", color=INK2, linewidth=0.8, label="identity", zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"IPL {m.map_unit}", color=INK, fontsize=9)
    ax.set_ylabel(f"ipldt {m.map_unit}", color=INK, fontsize=9)
    ax.set_title(title or m.label, color=INK, fontsize=10, loc="left")
    txt = (f"{st['n_voxels']:,d} voxels\nexact agreement {100 * st['exact_fraction']:.6f}%\n"
           f"({st['mismatches']:,d} differ)\nslope {st['slope']:.6f}\nintercept {st['intercept_mm']:+.1e} mm\n"
           f"R$^2$ = {st['r2']:.6f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID))
    cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("voxels", color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    _style(ax)
    return st


def plot_samplewise(ax, x, y, metric, title=None):
    """Per-subject values: ipldt vs IPL with the identity and the OLS line."""
    m = METRICS[metric] if isinstance(metric, str) else metric
    o = ols(x, y)
    icc = icc_2_1(x, y)
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.06 * (hi - lo) if hi > lo else 0.05 * hi
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "--", color=INK2, linewidth=0.8, label="identity", zorder=2)
    xx = np.array(lim)
    ax.plot(xx, o["slope"] * xx + o["intercept"], "-", color=ORANGE, linewidth=1.2, label="regression", zorder=3)
    ax.scatter(x, y, s=22, facecolor=BLUE, edgecolor="white", linewidth=0.6, zorder=4, label="subject")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(f"IPL {m.sample_label} ({m.sample_unit})", color=INK, fontsize=9)
    ax.set_ylabel(f"ipldt {m.sample_label} ({m.sample_unit})", color=INK, fontsize=9)
    ax.set_title(title or m.label, color=INK, fontsize=10, loc="left")
    txt = (f"n = {o['n']}\ny = {o['slope']:.6f} x {o['intercept']:+.1e}\nR$^2$ = {o['r2']:.6f}\nICC(2,1) = {icc:.6f}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID))
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    _style(ax)
    return dict(ols=o, icc=icc)


def plot_bland_altman(ax, x, y, metric, title=None):
    """Mean of the two methods vs their difference (ipldt - IPL), bias and 1.96 SD limits."""
    m = METRICS[metric] if isinstance(metric, str) else metric
    ba = bland_altman(x, y)
    mean, d = ba["mean"], ba["diff"]
    span = max(abs(ba["loa_low"]), abs(ba["loa_high"]), np.abs(d).max(), 1e-6)
    ax.axhline(0, color=GRID, linewidth=0.8, zorder=1)
    ax.axhline(ba["bias"], color=ORANGE, linewidth=1.2, label=f"bias {ba['bias']:+.2e}", zorder=2)
    ax.axhline(ba["loa_high"], color=INK2, linewidth=0.8, linestyle="--", label="bias +- 1.96 SD", zorder=2)
    ax.axhline(ba["loa_low"], color=INK2, linewidth=0.8, linestyle="--", zorder=2)
    ax.scatter(mean, d, s=22, facecolor=BLUE, edgecolor="white", linewidth=0.6, zorder=4, label="subject")
    ax.set_ylim(-1.7 * span, 2.9 * span)          # the top band stays free for the annotation and the legend
    xpad = 0.06 * (mean.max() - mean.min()) if mean.max() > mean.min() else 0.05 * mean.max()
    ax.set_xlim(mean.min() - xpad, mean.max() + xpad)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(7)
    ax.set_xlabel(f"mean of IPL and ipldt {m.sample_label} ({m.sample_unit})", color=INK, fontsize=9)
    ax.set_ylabel(f"ipldt - IPL ({m.sample_unit})", color=INK, fontsize=9)
    ax.set_title(title or m.label, color=INK, fontsize=10, loc="left")
    txt = (f"bias {ba['bias']:+.2e} {m.sample_unit}\nSD {ba['sd']:.2e}\nLoA [{ba['loa_low']:+.2e}, {ba['loa_high']:+.2e}]\n"
           f"max |diff| {ba['max_abs_diff']:.2e}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GRID))
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    _style(ax)
    return ba


# plot titles: never 'old' / 'new' -- each panel says which object the thickness was computed on
_PANEL_TITLES = {"TbTh": TBTH_SEG, "TbSp": "Tb.Sp", "TbN": "Tb.N", "CtTh": "Ct.Th",
                 "TbTh_old": TBTH_TSEG}
_VOXEL_TITLES = {"TbTh": f"{TBTH_SEG} map", "TbSp": "Tb.Sp map", "TbN": "1/Tb.N map", "CtTh": "Ct.Th map",
                 "TbTh_old": f"{TBTH_TSEG} map"}


def _grid_figure(stats, panel, titles, fname_stem, fig_dir, dpi=200, singles=True):
    import matplotlib.pyplot as plt
    keys = [k for k in METRIC_ORDER if k in stats]
    letters = "abcdefgh"
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.6), facecolor="white")
    axes = axes.ravel()
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    for i, k in enumerate(keys):
        panel(axes[i], stats[k], f"({letters[i]}) {titles[k]}")
    fig.tight_layout(w_pad=2.0, h_pad=2.5)
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(fig_dir, f"{fname_stem}.{ext}")
        fig.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight")
        paths.append(p)
    if singles:
        for k in keys:
            f1, ax = plt.subplots(figsize=(4.8, 4.4), facecolor="white")
            panel(ax, stats[k], titles[k])
            f1.tight_layout()
            p = os.path.join(fig_dir, f"{fname_stem}_{k}.png")
            f1.savefig(p, dpi=dpi, facecolor="white", bbox_inches="tight")
            paths.append(p)
            plt.close(f1)
    return fig, paths


def figure_voxelwise(stats, fig_dir=FIG_DIR, **kw):
    return _grid_figure(stats, lambda ax, s, t: plot_voxelwise(ax, s["pooled_joint"], s["metric"], title=t),
                        _VOXEL_TITLES, "voxelwise_regression", fig_dir, **kw)


def figure_samplewise(stats, fig_dir=FIG_DIR, **kw):
    return _grid_figure(stats, lambda ax, s, t: plot_samplewise(ax, s["sample_ipl"], s["sample_ours"], s["metric"], title=t),
                        _PANEL_TITLES, "samplewise_regression", fig_dir, **kw)


def figure_bland_altman(stats, fig_dir=FIG_DIR, **kw):
    return _grid_figure(stats, lambda ax, s, t: plot_bland_altman(ax, s["sample_ipl"], s["sample_ours"], s["metric"], title=t),
                        _PANEL_TITLES, "bland_altman", fig_dir, **kw)


def make_figures(records, joints, fig_dir=FIG_DIR, show=False):
    import matplotlib.pyplot as plt
    stats = summary_stats(records, joints)
    paths = []
    for fn in (figure_voxelwise, figure_samplewise, figure_bland_altman):
        fig, p = fn(stats, fig_dir)
        paths += p
        if show:
            plt.show()
        else:
            plt.close(fig)
    return stats, paths


# --------------------------------------------------------------------------------------- CLI
def main(argv=None):
    """CLI entry.  A partial run (--subjects and/or a --metrics subset) with the default output
    folders is redirected to results/against_ipl_run/subset_<tag>/ so the cohort tables and the combined figures
    of a full run are not overwritten.  The published results/records.json and sample_means.csv are never
    written without --allow-overwrite-shipped."""
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=None, help="subject folders (default: all 21)")
    ap.add_argument("--metrics", nargs="*", default=METRIC_ORDER, choices=METRIC_ORDER)
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--compute-only", action="store_true", help="fill the map cache and stop")
    ap.add_argument("--no-compute", action="store_true", help="use the cache only (error if a map is missing)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--root", default=DATA_ROOT)
    ap.add_argument("--cache", default=CACHE_DIR)
    ap.add_argument("--results", default=RUN_DIR, help="output folder (default results/against_ipl_run/)")
    ap.add_argument("--figures", default=RUN_FIG_DIR)
    ap.add_argument("--allow-overwrite-shipped", action="store_true",
                    help="allow --results to be validation/results/ itself or a published record folder")
    a = ap.parse_args(argv)
    if not a.compute_only and refuse_published(a.results, a.allow_overwrite_shipped):
        return 2
    subjects = a.subjects or list_subjects(a.root)
    subset = bool(a.subjects) or list(a.metrics) != list(METRIC_ORDER)
    if subset and a.results == RUN_DIR and a.figures == RUN_FIG_DIR:
        # a partial run (some subjects or some metrics) would otherwise overwrite the cohort
        # tables in results/ and the combined figures in results/figures/; keep it in its own folder
        tag = "_".join(a.subjects) if a.subjects else "all"
        tag = tag if len(tag) <= 60 else f"{len(a.subjects)}subjects"
        a.results = os.path.join(RUN_DIR, f"subset_{tag}")
        a.figures = os.path.join(a.results, "figures")
        log(f"partial run: results and figures go to {a.results} (pass --results/--figures to override)")
    log(f"ipldt {ipldt.__version__}; backend {ipldt.gpu.resolve_backend(a.backend)}; {len(subjects)} subjects; metrics {a.metrics}")
    if a.compute_only:
        for s in subjects:
            subj = Subject(s, a.root, a.cache)
            for k in a.metrics:
                cached_map(subj, k, compute=True, backend=a.backend)
            subj.release()
        log("cache complete")
        return 0
    records, joints = run_all(subjects, a.metrics, compute=not a.no_compute, backend=a.backend,
                              results_dir=a.results, cache_dir=a.cache, root=a.root)
    tot = {k: sum(r["mismatches"] for r in records if r["metric"] == k) for k in a.metrics}
    log("total mismatching voxels per map: " + ", ".join(f"{METRICS[k].label} {v:,d}" for k, v in tot.items()))
    stats = summary_stats(records, joints, a.metrics)
    write_summary(stats, os.path.join(a.results, "summary.md"))
    log(f"wrote {a.results}/mismatch_table.csv, mismatch_table.md, sample_means.csv, summary.md")
    if not a.no_figures:
        import matplotlib
        matplotlib.use("Agg")
        _, paths = make_figures(records, joints, a.figures)
        log(f"wrote {len(paths)} figure files under {a.figures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
