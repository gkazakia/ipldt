"""build_facts.py -- the facts sheet of the ipldt / ORMIR-BQRL paper (n = 137).

Every number in FACTS.md / facts.json is computed here from the de-identified validation RECORDS (never from prose):

  patella, configuration A   validation/results/records.json            (105 = 21 subjects x 5 maps)
                             validation/results/sample_means.csv         (per-subject metric values)
                             validation/results/patella_bvtv_A/records.json  BV/TV = the ratio on IPL's SEG and IPL's
                             rendered trabecular contour (validation/patella_bvtv_A.py), IPL's printed sheet value beside it
  patella, configuration B   validation/results/from_ipl_contour_ceil_dt/records/*.json  STEP 1, contours, SEG, dt maps,
                             metrics and BV/TV (LH pad offset 'ceil', float32); its SEG stage is asserted identical to
                             validation/results/from_ipl_contour_ceil/records/*.json (cross-check only); its STEP-1 preset
                             is read from validation/results/from_ipl_contour_ceil_dt/summary.md
  radius / tibia, A and B    validation/results/<OSLH_AUTO>/records/*.json    63 files; the 62 counted ultradistal and
                                                                            diaphyseal scans (Diaphyseal_CKD_991161 dropped)
                             validation/results/<OSLH_NOEDIT>/records/*.json  54 files, all diaphyseal
                             (the two set names are in validation/result_sets.py); record files <Group>_<Study>_<n>.json
  study names                validation/results/cohort_studies.csv and each record's meta.study_label (the study token of
                             the scan header); participants are counted from the records' pseudonymous
                             inventory.identity.subject
  parameters                 ipldt/step1.py, ipldt/ormir.py, ipldt/core.py (literals copied by hand, cited by line)

Cross-check: the pooled numbers must equal validation/results/pooled_137_comparison.json (manuscript/figures/pooled_statistics.py,
block n137); the build stops on any disagreement (meta.crosscheck records the comparison).

Run from the repository root:  python manuscript/facts/build_facts.py
Outputs: facts.json (full precision, stable keys) and FACTS.md (rounded).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, OrderedDict

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "validation"))  # noqa: E402  (datapaths, result_sets)
from datapaths import lab_path, public_path  # noqa: E402
from result_sets import OSLH_AUTO, OSLH_NOEDIT  # noqa: E402
DEFAULT_OUT = os.path.join(REPO, "manuscript", "facts")

_ap = argparse.ArgumentParser(description="THE facts sheet of the ipldt / ORMIR-BQRL manuscript")
_ap.add_argument("--set", dest="vset", choices=("137",), default="137",
                 help="137: the 21 patellae + the two radius/tibia sets of validation/result_sets.py (62 + 54)")
_ap.add_argument("--out", default=None, help="output directory (default manuscript/facts)")
ARGS = _ap.parse_args()
OUT_DIR = os.path.abspath(ARGS.out) if ARGS.out else DEFAULT_OUT
os.makedirs(OUT_DIR, exist_ok=True)

SRC_PAT_A_RECORDS = "validation/results/records.json"
SRC_PAT_A_MEANS = "validation/results/sample_means.csv"
SRC_PAT_A_TABLE = "validation/results/mismatch_table.md"
SRC_PAT_B_DT = "validation/results/from_ipl_contour_ceil_dt/records"
SRC_PAT_B_SEG = SRC_PAT_B_DT                                                # one run supplies every patella-B number
SRC_PAT_B_SEG_XCHECK = "validation/results/from_ipl_contour_ceil/records"   # SEG-only ceil run: cross-check only
SRC_PAT_B_SUMMARY = "validation/results/from_ipl_contour_ceil_dt/summary.md"
SRC_PAT_A_BVTV = "validation/results/patella_bvtv_A/records.json"
SRC_STUDY_TABLE = "validation/results/cohort_studies.csv"
if ARGS.vset == "137":
    SRC_OSLH_DIRS = OrderedDict([(OSLH_AUTO, f"validation/results/{OSLH_AUTO}/records"),
                                 (OSLH_NOEDIT, f"validation/results/{OSLH_NOEDIT}/records")])
    SRC_OSLH = f"validation/results/{{{OSLH_AUTO},{OSLH_NOEDIT}}}/records"
OSLH_EXPECT = {OSLH_AUTO: 63, OSLH_NOEDIT: 54}                      # record files matching the glob
OSLH_GLOB = re.compile(r"^[A-Za-z]+_[A-Za-z]+_\d+\.json$")
OSLH_DROP = {"Diaphyseal_CKD_991161"}
VOXEL_MM = 0.0607

F = OrderedDict()          # key -> dict(value, unit, source, note)
TABLES = OrderedDict()     # per-scan tables for figure makers


def put(key, value, source, unit=None, note=None):
    if isinstance(value, (np.integer,)):
        value = int(value)
    elif isinstance(value, (np.floating,)):
        value = float(value)
    d = OrderedDict(value=value, source=source)
    if unit is not None:
        d["unit"] = unit
    if note is not None:
        d["note"] = note
    F[key] = d
    return value


def rel(p):
    return os.path.join(REPO, p)


# ================================================================================================ load
def load_patella_A():
    recs = json.load(open(rel(SRC_PAT_A_RECORDS), encoding="utf-8"))
    means = list(csv.DictReader(open(rel(SRC_PAT_A_MEANS), encoding="utf-8")))
    return recs, means


def load_json_dir(d, keep=None):
    out = OrderedDict()
    for p in sorted(glob.glob(rel(os.path.join(d, "*.json")))):
        name = os.path.basename(p)
        if keep is not None and not keep(name):
            continue
        out[name[:-5]] = json.load(open(p, encoding="utf-8"))
    return out


pat_A_recs, pat_A_means = load_patella_A()
pat_B_dt = load_json_dir(SRC_PAT_B_DT)
pat_B_seg = pat_B_dt
pat_B_xc = load_json_dir(SRC_PAT_B_SEG_XCHECK)
pat_A_bvtv = json.load(open(rel(SRC_PAT_A_BVTV), encoding="utf-8"))
_oslh_by_dir = OrderedDict()
oslh_src = {}                                                               # record key -> results directory name
for _name, _d in SRC_OSLH_DIRS.items():
    _recs = load_json_dir(_d, keep=lambda n: bool(OSLH_GLOB.match(n)))
    assert len(_recs) == OSLH_EXPECT[_name], (_name, len(_recs))
    for _k in _recs:
        assert _k not in oslh_src, ("record in two result directories", _k)
        oslh_src[_k] = _name
    _oslh_by_dir[_name] = _recs
# one ordering for every aggregate: the record file name
oslh_all = OrderedDict(sorted(((k, v) for recs in _oslh_by_dir.values() for k, v in recs.items()), key=lambda kv: kv[0]))
oslh = OrderedDict((k, v) for k, v in oslh_all.items() if k not in OSLH_DROP)
N_OSLH_FILES = len(oslh_all)
N_OSLH = len(oslh)
N_TOTAL = 21 + N_OSLH
assert N_OSLH_FILES == sum(OSLH_EXPECT[n] for n in SRC_OSLH_DIRS) and N_OSLH == N_OSLH_FILES - len(OSLH_DROP & set(oslh_all)), (N_OSLH_FILES, N_OSLH)

assert len(pat_B_seg) == 21 and len(pat_B_dt) == 21, (len(pat_B_seg), len(pat_B_dt))
assert set(pat_B_seg) == set(pat_B_dt)
assert len(pat_A_means) == 21 and len(pat_A_recs) == 105
assert all(r["variants"]["lh_pad_offset"] == "ceil" and r["variants"]["lh_dtype"] == "float32" for r in oslh.values())
assert all(r["seg"]["lh_pad_offset"] == "ceil" and r["seg"]["lh_dtype"] == "float32" for r in pat_B_dt.values())
# the dt-inclusive run's STEP 1 / contour / SEG stage must equal the SEG-only ceil run it supersedes
assert set(pat_B_xc) == set(pat_B_dt)
for _k in pat_B_dt:
    for _part in ("cort", "trab"):
        assert pat_B_dt[_k]["step1"][_part]["mismatches"] == pat_B_xc[_k]["step1"][_part]["mismatches"], (_k, _part)
        assert pat_B_dt[_k]["contours"][_part]["mismatches"] == pat_B_xc[_k]["contours"][_part]["mismatches"], (_k, _part)
    for _s in ("SEG", "TRAB_SEG", "CORT_SEG"):
        for _f in ("mismatches", "ours_only", "ipl_only", "voxels_ours", "voxels_ipl", "union_voxels"):
            assert pat_B_dt[_k]["seg"][_s][_f] == pat_B_xc[_k]["seg"][_s][_f], (_k, _s, _f)
assert len(pat_A_bvtv["records"]) == 21 and set(pat_A_bvtv["records"]) == set(pat_B_dt)
assert all(r["equals_ceil_dt_ipl"] for r in pat_A_bvtv["records"].values())
assert all(r["dt_backend"] == "gpu" for r in oslh.values())
assert all(r["dt_backend"] == "gpu" for r in pat_B_dt.values())
assert all(r["backend"] == "gpu" for r in pat_A_recs)
assert all(r["trab_th_definition"]["value"] == "trab_seg" for r in oslh.values())
PAT_SUBJECTS = list(pat_B_seg.keys())

SITE_OF = {"UD Radius": "UD radius", "UD Tibia": "UD tibia", "D Tibia": "D tibia", "D Radius": "D radius"}
oslh_site = {k: SITE_OF[v["meta"]["bone_key"]] for k, v in oslh.items()}
oslh_study = {k: v["group"].split("/")[1] for k, v in oslh.items()}
SITES = ["UD radius", "UD tibia", "D radius", "D tibia"]
SITE_PRESET = {"UD radius": "radius", "UD tibia": "tibia", "D radius": "radius", "D tibia": "tibia"}
# per-site blocks, plus the two diaphyseal sites pooled (the figures' and the porosity summary's 'diaphyseal' cohort) and
# the two ultradistal sites pooled
SITE_GROUPS = OrderedDict([(s, [s]) for s in SITES] + [("diaphyseal", ["D radius", "D tibia"]), ("ultradistal", ["UD radius", "UD tibia"])])
COHORTS = ["patella"] + SITES
GROUPS = OrderedDict([("patella", ["patella"]), ("oslh", SITES), ("pooled", COHORTS)])
STUDIES = [s for s in ("CKD", "REPRO", "BMAT") if s in set(oslh_study.values())]
assert set(oslh_study.values()) == set(STUDIES), set(oslh_study.values())


def skey(s):
    return s.replace(" ", "_")


def site_members_label(members):
    return " + ".join(members)


# ================================================================================================ helpers
def stats_pair(ipl, ours):
    """Sample-wise agreement of ours against IPL (the reference): n, means, SDs, bias, SD of the
    differences, 95 % limits of agreement, mean and max |diff|, mean and max |rel| %, exact count,
    OLS slope / intercept of ours on IPL, R^2, Pearson r, ICC(2,1) (two-way random, absolute
    agreement, single measurement)."""
    x = np.asarray(ipl, dtype=np.float64)
    y = np.asarray(ours, dtype=np.float64)
    n = int(x.size)
    d = y - x
    # differences below 1e-12 relative are summation-order noise between two identical maps: treated as 0
    # (the bitwise count is kept in 'exact_bitwise'); when every pair is exact the fit is the identity.
    noise = np.abs(d) <= 1e-12 * np.maximum(1.0, np.abs(x))
    d_raw = d.copy()
    d = np.where(noise, 0.0, d)
    y = x + d
    out = OrderedDict(n=n)
    out["mean_ipl"] = float(x.mean())
    out["sd_ipl"] = float(x.std(ddof=1)) if n > 1 else 0.0
    out["mean_ours"] = float(y.mean())
    out["sd_ours"] = float(y.std(ddof=1)) if n > 1 else 0.0
    out["bias"] = float(d.mean())
    out["sd_diff"] = float(d.std(ddof=1)) if n > 1 else 0.0
    out["loa_low"] = out["bias"] - 1.96 * out["sd_diff"]
    out["loa_high"] = out["bias"] + 1.96 * out["sd_diff"]
    out["mean_abs_diff"] = float(np.abs(d).mean())
    out["max_abs_diff"] = float(np.abs(d).max())
    r_ = np.abs(d) / np.abs(x) * 100.0
    out["mean_rel_pct"] = float(r_.mean())
    out["max_rel_pct"] = float(r_.max())
    # 'exact' = identical to 12 significant digits (differences of a few 1e-16 arise from summation order when
    # two identical maps are averaged by different code paths); 'exact_bitwise' = identical float64 values
    out["exact"] = int(noise.sum())
    out["exact_bitwise"] = int((d_raw == 0).sum())
    if out["exact"] == n:
        out["slope"], out["intercept"], out["r2"], out["pearson_r"], out["icc21"] = 1.0, 0.0, 1.0, 1.0, 1.0
    elif n >= 2 and x.std() > 0:
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out["slope"] = float(slope)
        out["intercept"] = float(intercept)
        out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        out["pearson_r"] = float(np.corrcoef(x, y)[0, 1]) if y.std() > 0 else 1.0
        out["icc21"] = icc21(x, y)
    else:
        out["slope"] = out["intercept"] = out["r2"] = out["pearson_r"] = out["icc21"] = None
    return out


def icc21(x, y):
    data = np.stack([x, y], axis=1)
    n, k = data.shape
    grand = data.mean()
    rm = data.mean(axis=1)
    cm = data.mean(axis=0)
    msr = k * ((rm - grand) ** 2).sum() / (n - 1)
    msc = n * ((cm - grand) ** 2).sum() / (k - 1)
    sse = ((data - rm[:, None] - cm[None, :] + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den != 0 else 1.0


def q(v, p):
    return float(np.percentile(np.asarray(v, dtype=np.float64), p))


def summ(v):
    v = [float(t) for t in v]
    return OrderedDict(n=len(v), min=min(v), median=float(statistics.median(v)), max=max(v), mean=float(np.mean(v)),
                       q1=q(v, 25), q3=q(v, 75), total=float(sum(v)))


def breakdown(counter, order=None):
    keys = order if order is not None else sorted(counter)
    return ", ".join(f"{k} {counter[k]}" for k in keys if counter.get(k))


# ================================================================================================ cohort
def study_table():
    """{record id: study} from cohort_studies.csv (the study token of each scan's header, written by tools/deidentify_results.py)."""
    rows = list(csv.DictReader(open(rel(SRC_STUDY_TABLE), encoding="utf-8")))
    return {r["id"]: r for r in rows}


def cohort_section():
    S = SRC_OSLH + "/*.json (meta.bone_key, group)"
    put("cohort.n_total", N_TOTAL, "count of records: " + SRC_PAT_B_SEG + f" (21) + {SRC_OSLH} ({N_OSLH})")
    put("cohort.patella.n", 21, SRC_PAT_B_SEG + " (21 records; also 21 rows of " + SRC_PAT_A_MEANS + ")")
    put("cohort.oslh.n", N_OSLH, f"{SRC_OSLH} ({N_OSLH_FILES} records matching the glob, {', '.join(sorted(OSLH_DROP))} not used)")
    n_src = Counter(oslh_src[k] for k in oslh)
    for name, d in SRC_OSLH_DIRS.items():
        put(f"cohort.oslh.source.{name}.n", n_src[name], d + "/*.json",
            note={OSLH_AUTO: "63 radius / tibia measurements at IPL's automatic evaluation run (recovered from the scanner's "
                             "version history where the delivered evaluation had later been corrected), run with the shipped SEG assembly order (periosteal_first on every scan), the pore cascade, and configuration A reading IPL's own TRAB_SEG file",
                  OSLH_NOEDIT: "diaphyseal measurements whose delivered evaluation is the automatic one (staged from each run's own files: validation/stage_noedit_set.py)"}[name])
    cnt = Counter(oslh_site.values())
    for s in SITES:
        put(f"cohort.site.{skey(s)}.n", cnt[s], S)
    put("cohort.site.diaphyseal.n", cnt["D radius"] + cnt["D tibia"], S, note="diaphyseal radius + diaphyseal tibia (the figures' 'diaphyseal' cohort)")
    # ---- studies (the study token of the patient name; the record's group names it)
    tok = {}
    for k, v in oslh.items():
        m = re.search(r"^([A-Za-z]+study)$", v["meta"].get("study_label") or "", re.I)
        tok.setdefault(oslh_study[k], Counter())[m.group(1) if m else "?"] += 1
    desc = {"CKD": "chronic kidney disease study", "REPRO": "reproducibility study",
            "BMAT": "a third study; its aim is not stated in the records"}
    grp = Counter(oslh_study.values())
    for st in STUDIES:
        by_site = Counter(oslh_site[k] for k in oslh if oslh_study[k] == st)
        label = tok[st].most_common(1)[0][0]
        put(f"cohort.oslh.study.{st}.n", grp[st], S, note=f"{desc[st]} ('{label}' in the scan header): {breakdown(by_site, SITES)}")
        put(f"cohort.oslh.study.{st}.label", label, SRC_OSLH + "/*.json meta.study_label (the study token of the scan header)",
            note="spellings in the scan headers: " + breakdown(tok[st]))
        for s in SITES:
            if by_site[s]:
                put(f"cohort.study.{st}.site.{skey(s)}.n", by_site[s], S)
    # the study of every scan as read from its header's study token: patella study + cross-check
    st_tab = study_table()
    pat_st = Counter(st_tab[k]["study"] for k in PAT_SUBJECTS)
    assert len(pat_st) == 1, pat_st
    put("cohort.patella.study", next(iter(pat_st)), SRC_STUDY_TABLE + " (study token of each scan's header)",
        note=f"{pat_st.most_common(1)[0][1]} of 21 patellae; the patellofemoral osteoarthritis study")
    for k in oslh:
        sid = oslh[k]["id"]
        assert st_tab[sid]["study"].upper() == oslh_study[k].upper() + "STUDY", (k, st_tab[sid]["study"], oslh_study[k])
    # ---- participants: the pseudonymous participant code (inventory.identity.subject; oslh_inventory normalises typos)
    subj = {k: oslh[k]["inventory"]["identity"]["subject"] for k in oslh}
    fixes = [dict(id=oslh[k]["id"], note="; ".join(oslh[k]["inventory"]["identity"].get("name_notes") or []))
             for k in oslh if oslh[k]["inventory"]["identity"].get("name_notes")]
    Sp = SRC_OSLH + "/*.json inventory.identity.subject"
    tot = 0
    for st in STUDIES:
        ids = [k for k in oslh if oslh_study[k] == st]
        parts = Counter(subj[k] for k in ids)
        tot += len(parts)
        per = Counter(parts.values())
        put(f"cohort.oslh.study.{st}.participants", len(parts), Sp,
            note="distinct participant codes of the study among its counted scans (pseudonyms in the published records; codes are not copied here)")
        put(f"cohort.oslh.study.{st}.scans_per_participant", OrderedDict((str(a), per[a]) for a in sorted(per)), Sp,
            note="{scans per participant: number of participants}")
        rep = Counter((subj[k], oslh_site[k]) for k in ids)
        put(f"cohort.oslh.study.{st}.same_site_repeats", sum(1 for v in rep.values() if v > 1), Sp + " + meta.bone_key",
            note="participant-site pairs with more than one scan in the set (0 = at most one scan per participant and site)")
    put("cohort.oslh.participants", tot, Sp, note="sum over the studies; each study codes its own participants, so whether one person took part in two studies cannot be told from the records")
    put("cohort.oslh.participant_code_fixes", fixes, SRC_OSLH + "/*.json inventory.identity.name_notes",
        note="participant codes normalised by validation/oslh_inventory.py (one-character typos in the scan header); after it every participant has at most one scan per site (cohort.oslh.study.*.same_site_repeats)")
    for s in SITES:
        put(f"cohort.site.{skey(s)}.participants", len({(oslh_study[k], subj[k]) for k in oslh if oslh_site[k] == s}), Sp)
    pat_part = Counter(k.rsplit("_", 1)[0] for k in PAT_SUBJECTS)
    assert all(k.rsplit("_", 1)[1] in ("L", "R") for k in PAT_SUBJECTS)
    put("cohort.patella.participants", len(pat_part), SRC_PAT_B_SEG + " subject ids (<participant>_<L|R>)",
        note=f"{sum(pat_part.values())} patellae from {len(pat_part)} participants (distinct subject id before the knee side)")
    put("cohort.patella.participants_bilateral", sum(1 for v in pat_part.values() if v == 2), SRC_PAT_B_SEG + " subject ids",
        note="participants who contribute both knees")
    put("cohort.participants.total", len(pat_part) + tot, "cohort.patella.participants + cohort.oslh.participants",
        note="assumes no person took part in two of the studies, which the records cannot establish")
    # ---- the STEP-1 parameter set of each scan: read from the record (preset.name), per site
    pre = Counter(v["preset"]["name"] for v in oslh.values())
    put("cohort.oslh.preset.tibia.n", pre["tibia"], SRC_OSLH + "/*.json preset.name")
    put("cohort.oslh.preset.radius.n", pre["radius"], SRC_OSLH + "/*.json preset.name")
    for s in SITES:
        c = Counter(oslh[k]["preset"]["name"] for k in oslh if oslh_site[k] == s)
        for p in ("tibia", "radius"):
            put(f"cohort.site.{skey(s)}.preset.{p}.n", c[p], SRC_OSLH + f"/*.json preset.name (bone_key {s})")
    put("cohort.oslh.preset.source", dict(Counter(v["preset"]["source"] for v in oslh.values())), SRC_OSLH + "/*.json preset.source",
        note="'layout rule' = the site's own preset (TIBIA for a tibia, RADIUS for a radius); 'fallback (better rendering match)' = the other preset, taken because only it reproduces IPL's compartment renderings")
    other = [k for k in oslh if oslh[k]["preset"]["name"] != SITE_PRESET[oslh_site[k]]]
    put("cohort.oslh.preset.other_site.n", len(other), SRC_OSLH + "/*.json preset.name vs meta.bone_key")
    put("cohort.oslh.preset.other_site.scans",
        [dict(id=oslh[k]["id"], site=oslh_site[k], preset=oslh[k]["preset"]["name"], source=oslh[k]["preset"]["source"],
              trials={p: t["score"] for p, t in oslh[k]["B"]["preset_trials"].items()}) for k in other],
        SRC_OSLH + "/*.json preset, B.preset_trials", note="scans evaluated by IPL with the other site's STEP-1 parameter set; 'trials' = differing rendering voxels under each preset tried")
    chk = [k for k in ("Diaphyseal_BMAT_610892", "Diaphyseal_REPRO_787166", "Diaphyseal_REPRO_950609") if k in oslh]
    if chk:
        put("cohort.oslh.preset.checked_scans",
            [dict(id=oslh[k]["id"], site=oslh_site[k], site_index=oslh[k]["meta"].get("site_index"), preset=oslh[k]["preset"]["name"],
                  source=oslh[k]["preset"]["source"], trials={p: t["score"] for p, t in oslh[k]["B"]["preset_trials"].items()}) for k in chk],
            SRC_OSLH + "/*.json meta, preset, B.preset_trials",
            note="three scans an internal review listed as run with the other site's preset; "
                 "the records confirm it for 610892 only: 787166 and 950609 carry their own site's preset by the layout rule, reproduce IPL's renderings exactly with it, "
                 "and their scan header and scanner site index (38 = tibia, 20 = radius) agree with that site")
    # the launch-script check (the vendor's per-scan launch scripts) is internal and not part of the public build
    put("cohort.oslh.preset.launch_script.scans_with_file", None, "vendor launch scripts (not distributed)", note="launch-script check not run in the public build")
    pat_pre = re.search(r"site preset '(\w+)'", open(rel(SRC_PAT_B_SUMMARY), encoding="utf-8").read())
    put("cohort.patella.preset", pat_pre.group(1) if pat_pre else None, SRC_PAT_B_SUMMARY + " (site preset of the patella configuration-B run)")
    # voxel size: header element sizes
    el_x = [v["grey"]["el_size_mm"][0] for v in oslh.values()] + [v["grey"]["el_size_mm"][0] for v in pat_B_seg.values()]
    el_z = [v["grey"]["el_size_mm"][2] for v in oslh.values()] + [v["grey"]["el_size_mm"][2] for v in pat_B_seg.values()]
    put("cohort.voxel_size_mm.nominal", VOXEL_MM, "morphometry.voxel_size_mm of every record; IPL's standard XtremeCT II protocol", "mm")
    put("cohort.voxel_size_mm.header_x.min", min(el_x), f"grey.el_size_mm[0] of the {N_TOTAL} records", "mm")
    put("cohort.voxel_size_mm.header_x.max", max(el_x), f"grey.el_size_mm[0] of the {N_TOTAL} records", "mm")
    put("cohort.voxel_size_mm.header_z.min", min(el_z), f"grey.el_size_mm[2] of the {N_TOTAL} records", "mm")
    put("cohort.voxel_size_mm.header_z.max", max(el_z), f"grey.el_size_mm[2] of the {N_TOTAL} records", "mm")
    # slices / grids
    pz = {k: v["grey"]["grid"][0][2] for k, v in pat_B_seg.items()}
    oz = {k: v["grey"]["grid"][0][2] for k, v in oslh.items()}
    put("cohort.patella.slices.counts", dict(Counter(pz.values())), SRC_PAT_B_SEG + " grey.grid", note="z extent of the greyscale AIM per scan")
    put("cohort.oslh.slices.counts", dict(Counter(oz.values())), SRC_OSLH + " grey.grid")
    for s in SITES:
        put(f"cohort.site.{skey(s)}.slices.counts", dict(Counter(oz[k] for k in oslh if oslh_site[k] == s)), SRC_OSLH + f" grey.grid (bone_key {s})")
    # IPL version
    n_v = 0
    for name, d in SRC_OSLH_DIRS.items():
        for p in glob.glob(rel(d + "/*.json")):
            if OSLH_GLOB.match(os.path.basename(p)) and "IPL V5.42" in open(p, encoding="utf-8").read():
                n_v += 1
    put("cohort.ipl_version", "V5.42", "IPL V5.42, the version every evaluation ran (the records' processing-log evidence; the header of validation/results/from_ipl_contour_ceil_dt/summary.md)")
    put("cohort.scanner", "XtremeCT II (Scanco Medical AG)", "scan headers: Orig-ISQ-Dim 2304 x 2304; study protocol")
    longer = sorted(z for z in pz.values() if z != 168)
    put("cohort.patella.acquisition", "68 kVp, 1470 uA, 60.7 um isotropic, ~2 min per knee, ~5 uSv per knee; UCSF IRB, written informed consent",
        "study protocol (paper, Section 2.2)",
        note=f"the protocol is one 1-cm stack (168 slices) per knee; {len(longer)} scan(s) in the records have other stack lengths ({', '.join(map(str, longer))} slices; cohort.patella.slices.counts)")
    # ---- Table 1: one row per site, every column read from the records
    rows = []
    pat_row = OrderedDict(site="Patella", key="patella", n=21,
                          studies={F["cohort.patella.study"]["value"].replace("STUDY", ""): 21},
                          participants=F["cohort.patella.participants"]["value"],
                          stack_lengths=OrderedDict((str(z), c) for z, c in sorted(Counter(pz.values()).items())),
                          preset={F["cohort.patella.preset"]["value"]: 21})
    rows.append(pat_row)
    lab = {"UD radius": "Ultradistal radius", "UD tibia": "Ultradistal tibia", "D radius": "Diaphyseal radius", "D tibia": "Diaphyseal tibia"}
    for s in SITES:
        ids = [k for k in oslh if oslh_site[k] == s]
        st = Counter(oslh_study[k] for k in ids)
        rows.append(OrderedDict(site=lab[s], key=skey(s), n=len(ids), studies=OrderedDict((x, st[x]) for x in STUDIES if st[x]),
                                participants=F[f"cohort.site.{skey(s)}.participants"]["value"],
                                stack_lengths=OrderedDict((str(z), c) for z, c in sorted(Counter(oz[k] for k in ids).items())),
                                preset=OrderedDict((p, c) for p, c in sorted(Counter(oslh[k]["preset"]["name"] for k in ids).items()))))
    tot_st = Counter()
    tot_z = Counter()
    tot_p = Counter()
    for r_ in rows:
        tot_st.update(r_["studies"])
        tot_z.update(r_["stack_lengths"])
        tot_p.update(r_["preset"])
    rows.append(OrderedDict(site="Total", key="total", n=N_TOTAL, studies=dict(tot_st), participants=F["cohort.participants.total"]["value"],
                            stack_lengths=dict(tot_z), preset=dict(tot_p)))
    assert sum(r_["n"] for r_ in rows[:-1]) == N_TOTAL
    TABLES["cohort.table1"] = rows


# ================================================================================================ config A
MAP_KEYS_A_PAT = OrderedDict([("TbTh_trabseg", "TbTh_old"), ("TbTh_wholeseg", "TbTh"), ("TbSp", "TbSp"), ("TbN", "TbN"), ("CtTh", "CtTh")])
MAP_KEYS_OSLH = OrderedDict([("TbTh_trabseg", "TRAB_TH_tseg"), ("TbTh_wholeseg", "TRAB_TH_seg"), ("TbSp", "TRAB_SP"), ("TbN", "TRAB_1N"), ("CtTh", "CORT_TH")])
MAP_KEYS_B_PAT = OrderedDict([("TbTh_trabseg", "TRAB_TH_old"), ("TbTh_wholeseg", "TRAB_TH"), ("TbSp", "TRAB_SP"), ("TbN", "TRAB_1N"), ("CtTh", "CORT_TH")])
MAP_LABEL = OrderedDict([("TbTh_trabseg", "Tb.Th map (dt_thickness on TRAB_SEG, Script 32 definition)"),
                         ("TbTh_wholeseg", "Tb.Th map (dt_thickness on the whole SEG; internal, not the Script 32 definition)"),
                         ("TbSp", "Tb.Sp map (dt_spacing)"), ("TbN", "1/Tb.N map (dt_number)"), ("CtTh", "Ct.Th map (dt_thickness on CORT_MASK)")])


def map_agg(items, src, prefix, note=None):
    """items: list of dicts with n_voxels, mismatches, ours_only, ipl_only, support_ipl (and optional both_nonzero_differ)."""
    n = len(items)
    mism = [int(i["mismatches"]) for i in items]
    put(f"{prefix}.n_scans", n, src, note=note)
    put(f"{prefix}.exact_scans", sum(1 for m in mism if m == 0), src)
    put(f"{prefix}.mismatches", sum(mism), src, "voxels")
    put(f"{prefix}.ours_only", sum(int(i["ours_only"]) for i in items), src, "voxels")
    put(f"{prefix}.ipl_only", sum(int(i["ipl_only"]) for i in items), src, "voxels")
    if all("both_nonzero_differ" in i for i in items):
        put(f"{prefix}.both_nonzero_differ", sum(int(i["both_nonzero_differ"]) for i in items), src, "voxels")
    put(f"{prefix}.voxels_compared", sum(int(i["n_voxels"]) for i in items), src, "voxels")
    put(f"{prefix}.support_ipl", sum(int(i["support_ipl"]) for i in items), src, "voxels", note="non-zero voxels of IPL's map")
    put(f"{prefix}.per_scan_max", max(mism), src, "voxels")
    put(f"{prefix}.per_scan_median", float(statistics.median(mism)), src, "voxels")
    return mism


def config_A_section():
    # patella
    byk = {}
    for r in pat_A_recs:
        byk.setdefault(r["metric"], []).append(r)
    for mk, rk in MAP_KEYS_A_PAT.items():
        items = byk[rk]
        assert len(items) == 21
        map_agg(items, SRC_PAT_A_RECORDS + f" (metric='{rk}')", f"A.patella.map.{mk}")
        put(f"A.patella.map.{mk}.time_s.median", float(statistics.median(r["compute_time_s"] for r in items)), SRC_PAT_A_RECORDS + " compute_time_s", "s")
        put(f"A.patella.map.{mk}.time_s.min", min(r["compute_time_s"] for r in items), SRC_PAT_A_RECORDS + " compute_time_s", "s")
        put(f"A.patella.map.{mk}.time_s.max", max(r["compute_time_s"] for r in items), SRC_PAT_A_RECORDS + " compute_time_s", "s")
    put("A.patella.maps.all_exact", all(F[f"A.patella.map.{mk}.mismatches"]["value"] == 0 for mk in MAP_KEYS_A_PAT), SRC_PAT_A_RECORDS)
    put("A.patella.voxels_compared.seg_grid", F["A.patella.map.TbSp.voxels_compared"]["value"], SRC_PAT_A_RECORDS, "voxels", note="SEG-grid voxels of the 21 subjects (each of Tb.Th, Tb.Sp, 1/Tb.N)")
    put("A.patella.voxels_compared.cort_grid", F["A.patella.map.CtTh.voxels_compared"]["value"], SRC_PAT_A_RECORDS, "voxels", note="CORT_MASK-grid voxels (Ct.Th)")
    put("A.patella.voxels_compared.total_four_maps", 3 * F["A.patella.map.TbSp.voxels_compared"]["value"] + F["A.patella.map.CtTh.voxels_compared"]["value"],
        SRC_PAT_A_RECORDS, "voxels", note="Tb.Th (TRAB_SEG) + Tb.Sp + 1/Tb.N + Ct.Th")
    put("A.patella.voxels_compared.total_five_maps", 4 * F["A.patella.map.TbSp.voxels_compared"]["value"] + F["A.patella.map.CtTh.voxels_compared"]["value"],
        SRC_PAT_A_RECORDS, "voxels", note="both Tb.Th definitions + Tb.Sp + 1/Tb.N + Ct.Th")
    # patella BV/TV under configuration A: ipldt's ratio on IPL's SEG and IPL's rendered trabecular contour
    Sb = SRC_PAT_A_BVTV + " (validation/patella_bvtv_A.py)"
    recs = pat_A_bvtv["records"]
    rows = [recs[k] for k in PAT_SUBJECTS]
    put("A.patella.BVTV.formula", pat_A_bvtv["meta"]["formula"], Sb, note="the ratio validation/validate_dataset.py computes as bvtv(A_frame['seg'], ipl_G['trab']) for the radius/tibia records")
    put("A.patella.BVTV.inputs", pat_A_bvtv["meta"]["inputs"], Sb)
    put("A.patella.BVTV.n_scans", len(rows), Sb)
    put("A.patella.BVTV.bv_voxels", sum(r["bv_voxels"] for r in rows), Sb, "voxels", note="|SEG AND rendered trabecular contour|, summed over the 21 patellae")
    put("A.patella.BVTV.tv_voxels", sum(r["tv_voxels"] for r in rows), Sb, "voxels", note="|rendered trabecular contour|, summed over the 21 patellae")
    put("A.patella.BVTV.exact_scans", sum(1 for r in rows if r["equals_ceil_dt_ipl"]), Sb,
        note="scans on which ipldt's ratio equals the ratio taken on the same files by validate_from_ipl_contour.ipl_bvtv (bvtv.ipl of " + SRC_PAT_B_DT + "): BV, TV and the quotient bitwise")
    put("A.patella.BVTV.ipl_reference", "the same ratio on IPL's own files (IPL's SEG, IPL's rendered trabecular contour); Script 32 writes no BV/TV, and the evaluation sheet prints three decimals", Sb)
    put("A.patella.BVTV.printed.source", pat_A_bvtv["meta"]["printed_source"], Sb, note="produced by the scanner's evaluation program (3D Density and Structure Analysis sheet), not by a command of Script 32; its exact voxel basis is not documented there")
    put("A.patella.BVTV.printed.n_with_value", pat_A_bvtv["meta"]["printed_available"], Sb)
    put("A.patella.BVTV.printed.decimals", 3, Sb)
    put("A.patella.BVTV.printed.agree_after_rounding", pat_A_bvtv["meta"]["printed_agree_after_rounding"], Sb, note="scans whose configuration-A ratio, rounded to three decimals, equals the sheet value")
    put("A.patella.BVTV.printed.agree_after_rounding_B", pat_A_bvtv["meta"]["printed_agree_after_rounding_B"], Sb, note="the same test for configuration B's ratio (ipldt's SEG and contour)")
    put("A.patella.BVTV.printed.max_abs_diff", pat_A_bvtv["meta"]["max_abs_diff_to_printed"], Sb, "-", note="max |ratio - printed| over the 21 scans; the rounding bound of a three-decimal value is 0.0005")
    put("A.patella.BVTV.printed.within_0_0006", sum(1 for r in rows if abs(r["diff_to_printed"]) < 0.0006), Sb)
    diff_scans = [k for k in PAT_SUBJECTS if not recs[k]["printed_agrees_after_rounding"]]
    bdist = lambda v: abs((v * 1000.0) % 1.0 - 0.5) / 1000.0
    put("A.patella.BVTV.printed.differing_scans", [dict(id=k, ratio=recs[k]["bvtv"], printed=recs[k]["printed"]["BVTV"], diff=recs[k]["diff_to_printed"], distance_to_rounding_boundary=bdist(recs[k]["bvtv"])) for k in diff_scans], Sb,
        note="scans whose sheet value differs from the rounded ratio by one unit in the third decimal")
    put("A.patella.BVTV.printed.differing_max_distance_to_boundary", max((bdist(recs[k]["bvtv"]) for k in diff_scans), default=0.0), Sb, "-",
        note="how far the ratio of each differing scan lies from the nearest three-decimal rounding boundary (max over the differing scans)")
    TABLES["A.patella.BVTV.per_scan"] = [OrderedDict(scan=k, bv_voxels=recs[k]["bv_voxels"], tv_voxels=recs[k]["tv_voxels"], bvtv=recs[k]["bvtv"], bvtv_ipl_same_files=recs[k]["ceil_dt_record"]["ipl"],
                                                    bvtv_B_ours=recs[k]["ceil_dt_record"]["ours_B"], printed=recs[k]["printed"]["BVTV"], agrees_after_rounding=recs[k]["printed_agrees_after_rounding"]) for k in PAT_SUBJECTS]
    # radius / tibia: all, per site, and the two diaphyseal sites pooled
    for mk, rk in MAP_KEYS_OSLH.items():
        items = [oslh[k]["A"]["maps"][rk] for k in oslh]
        map_agg(items, SRC_OSLH + f" A.maps.{rk}", f"A.oslh.map.{mk}")
        for g, members in SITE_GROUPS.items():
            ids = [k for k in oslh if oslh_site[k] in members]
            items = [oslh[k]["A"]["maps"][rk] for k in ids]
            map_agg(items, SRC_OSLH + f" A.maps.{rk} (bone_key {site_members_label(members)})", f"A.site.{skey(g)}.map.{mk}")
    # the differing Tb.Th voxel(s)
    hits = [(k, oslh[k]["A"]["maps"]["TRAB_TH_tseg"]) for k in oslh if oslh[k]["A"]["maps"]["TRAB_TH_tseg"]["mismatches"] > 0]
    put("A.oslh.map.TbTh_trabseg.nonexact_scans", [dict(id=k, site=oslh_site[k], mismatches=m["mismatches"], ours_only=m["ours_only"], ipl_only=m["ipl_only"],
                                                        ipl_value_vox=None) for k, m in hits], SRC_OSLH + " A.maps.TRAB_TH_tseg")
    # pooled A (patella + radius / tibia) for the four reported maps
    for mk in ("TbTh_trabseg", "TbSp", "TbN", "CtTh"):
        for stat in ("n_scans", "exact_scans", "mismatches", "ours_only", "ipl_only", "voxels_compared", "support_ipl"):
            put(f"A.pooled.map.{mk}.{stat}", F[f"A.patella.map.{mk}.{stat}"]["value"] + F[f"A.oslh.map.{mk}.{stat}"]["value"],
                "sum of A.patella and A.oslh")
    put("A.pooled.voxels_compared.four_maps", sum(F[f"A.pooled.map.{mk}.voxels_compared"]["value"] for mk in ("TbTh_trabseg", "TbSp", "TbN", "CtTh")),
        f"sum over the four reported maps, {N_TOTAL} scans", "voxels")
    put("A.pooled.mismatches.four_maps", sum(F[f"A.pooled.map.{mk}.mismatches"]["value"] for mk in ("TbTh_trabseg", "TbSp", "TbN", "CtTh")),
        f"sum over the four reported maps, {N_TOTAL} scans", "voxels")
    put("A.pooled.exact_comparisons.four_maps", sum(F[f"A.pooled.map.{mk}.exact_scans"]["value"] for mk in ("TbTh_trabseg", "TbSp", "TbN", "CtTh")),
        f"sum over the four reported maps, {N_TOTAL} scans", note=f"scan-map comparisons with 0 differing voxels, of {4 * N_TOTAL}")
    # timings radius / tibia A
    put("A.oslh.time_s.dt_all_maps.median", float(statistics.median(r["timings"]["A"] for r in oslh.values())), SRC_OSLH + " timings.A", "s",
        note="dt of the five A maps (both Tb.Th objects, Tb.Sp, 1/Tb.N, Ct.Th) per measurement, GPU")
    put("A.oslh.time_s.dt_all_maps.min", min(r["timings"]["A"] for r in oslh.values()), SRC_OSLH + " timings.A", "s")
    put("A.oslh.time_s.dt_all_maps.max", max(r["timings"]["A"] for r in oslh.values()), SRC_OSLH + " timings.A", "s")
    for mk, rk in MAP_KEYS_OSLH.items():
        t = [r["A"]["timing"][rk] for r in oslh.values()]
        put(f"A.oslh.map.{mk}.time_s.median", float(statistics.median(t)), SRC_OSLH + f" A.timing.{rk}", "s")
        put(f"A.oslh.map.{mk}.time_s.min", min(t), SRC_OSLH + f" A.timing.{rk}", "s")
        put(f"A.oslh.map.{mk}.time_s.max", max(t), SRC_OSLH + f" A.timing.{rk}", "s")


# ================================================================================================ config B
def seg_agg(items, src, prefix, ids, note=None):
    """items: dicts with mismatches, ours_only, ipl_only, dice, union_voxels (SEG-like)."""
    mism = [int(i["mismatches"]) for i in items]
    dice = [float(i["dice"]) for i in items]
    put(f"{prefix}.n_scans", len(items), src, note=note)
    put(f"{prefix}.exact_scans", sum(1 for m in mism if m == 0), src)
    put(f"{prefix}.mismatches", sum(mism), src, "voxels")
    put(f"{prefix}.ours_only", sum(int(i["ours_only"]) for i in items), src, "voxels")
    put(f"{prefix}.ipl_only", sum(int(i["ipl_only"]) for i in items), src, "voxels")
    put(f"{prefix}.voxels_compared", sum(int(i["union_voxels"]) for i in items), src, "voxels", note="union grid of the two volumes, summed over scans")
    put(f"{prefix}.voxels_ipl", sum(int(i["voxels_ipl"]) for i in items), src, "voxels", note="set voxels of IPL's file, summed")
    put(f"{prefix}.dice.min", min(dice), src)
    put(f"{prefix}.dice.median", float(statistics.median(dice)), src)
    put(f"{prefix}.dice.mean", float(np.mean(dice)), src)
    put(f"{prefix}.dice.ge_0_9999", sum(1 for d in dice if d >= 0.9999), src)
    put(f"{prefix}.dice.ge_0_999", sum(1 for d in dice if d >= 0.999), src)
    put(f"{prefix}.per_scan.min", min(mism), src, "voxels")
    put(f"{prefix}.per_scan.q1", q(mism, 25), src, "voxels")
    put(f"{prefix}.per_scan.median", float(statistics.median(mism)), src, "voxels")
    put(f"{prefix}.per_scan.q3", q(mism, 75), src, "voxels")
    put(f"{prefix}.per_scan.max", max(mism), src, "voxels")
    put(f"{prefix}.per_scan.le_10", sum(1 for m in mism if m <= 10), src)
    put(f"{prefix}.per_scan.le_100", sum(1 for m in mism if m <= 100), src)
    put(f"{prefix}.per_scan.le_1000", sum(1 for m in mism if m <= 1000), src)
    put(f"{prefix}.mismatch_ppm", 1e6 * sum(mism) / sum(int(i["union_voxels"]) for i in items), src, "per million voxels compared")
    return mism, dice


def config_B_section():
    # ---- patella STEP 1 / contours / SEG (ceil run)
    S = SRC_PAT_B_SEG + "/*.json"
    for part, key in (("cort", "cort"), ("trab", "trab")):
        items = [pat_B_seg[k]["step1"][key] for k in PAT_SUBJECTS]
        put(f"B.patella.step1.{part}.n_scans", 21, S + f" step1.{key}")
        put(f"B.patella.step1.{part}.exact_scans", sum(1 for i in items if i["mismatches"] == 0), S + f" step1.{key}")
        put(f"B.patella.step1.{part}.mismatches", sum(i["mismatches"] for i in items), S + f" step1.{key}", "voxels")
        put(f"B.patella.step1.{part}.dice.min", min(i["dice"] for i in items), S + f" step1.{key}")
        put(f"B.patella.step1.{part}.voxels_ipl", sum(i["voxels_ipl"] for i in items), S + f" step1.{key}", "voxels")
        put(f"B.patella.step1.{part}.grids_identical", sum(1 for i in items if i["grids_equal"]), S + f" step1.{key}.grids_equal")
        c = [pat_B_seg[k]["contours"][key] for k in PAT_SUBJECTS]
        put(f"B.patella.contour.{part}.n_scans", 21, S + f" contours.{key}")
        put(f"B.patella.contour.{part}.exact_scans", sum(1 for i in c if i["mismatches"] == 0), S + f" contours.{key}")
        put(f"B.patella.contour.{part}.mismatches", sum(i["mismatches"] for i in c), S + f" contours.{key}", "voxels")
        put(f"B.patella.contour.{part}.voxels_ipl", sum(i["voxels_ipl"] for i in c), S + f" contours.{key}", "voxels")
    for seg_key in ("SEG", "TRAB_SEG", "CORT_SEG"):
        items = [pat_B_seg[k]["seg"][seg_key] for k in PAT_SUBJECTS]
        seg_agg(items, S + f" seg.{seg_key}", f"B.patella.{seg_key}", PAT_SUBJECTS)
    put("B.patella.SEG.label_mismatches", sum(pat_B_seg[k]["seg"]["SEG"].get("label_mismatches", 0) for k in PAT_SUBJECTS), S + " seg.SEG.label_mismatches", "voxels",
        note="voxels set in both SEGs but with different labels (127 vs 126)")
    TABLES["B.patella.SEG.per_scan"] = [OrderedDict(scan=k, mismatches=pat_B_seg[k]["seg"]["SEG"]["mismatches"], ours_only=pat_B_seg[k]["seg"]["SEG"]["ours_only"],
                                                    ipl_only=pat_B_seg[k]["seg"]["SEG"]["ipl_only"], dice=pat_B_seg[k]["seg"]["SEG"]["dice"],
                                                    union_voxels=pat_B_seg[k]["seg"]["SEG"]["union_voxels"], voxels_ipl=pat_B_seg[k]["seg"]["SEG"]["voxels_ipl"],
                                                    trab_seg_mismatches=pat_B_seg[k]["seg"]["TRAB_SEG"]["mismatches"], cort_seg_mismatches=pat_B_seg[k]["seg"]["CORT_SEG"]["mismatches"])
                                        for k in PAT_SUBJECTS]
    # timings: one dt-inclusive run (read, step1, render, seg, dt); 'total' keeps its v1 meaning (read + step1 + render + seg, no dt)
    for st in ("read", "step1", "render", "seg"):
        t = [pat_B_seg[k]["timings"][st] for k in PAT_SUBJECTS]
        put(f"B.patella.time_s.{st}.median", float(statistics.median(t)), S + f" timings.{st}", "s", note="dt-inclusive run of 2026-09-18 under the shipped engine (GPU)")
        put(f"B.patella.time_s.{st}.min", min(t), S + f" timings.{st}", "s")
        put(f"B.patella.time_s.{st}.max", max(t), S + f" timings.{st}", "s")
    t = [sum(pat_B_seg[k]["timings"][st] for st in ("read", "step1", "render", "seg")) for k in PAT_SUBJECTS]
    put("B.patella.time_s.total.median", float(statistics.median(t)), S + " timings.read + step1 + render + seg", "s", note="read + STEP 1 + rendering + SEG per scan, without dt (the same meaning as in v1)")
    put("B.patella.time_s.total.min", min(t), S + " timings.read + step1 + render + seg", "s")
    put("B.patella.time_s.total.max", max(t), S + " timings.read + step1 + render + seg", "s")
    Sd = SRC_PAT_B_DT + "/*.json"
    for st in ("dt", "total"):
        t = [pat_B_dt[k]["timings"][st] for k in PAT_SUBJECTS]
        put(f"B.patella.time_s.dt_run.{st}.median", float(statistics.median(t)), Sd + f" timings.{st}", "s", note="dt-inclusive run of 2026-09-18 under the shipped engine (five maps, GPU); 'total' includes read, step1, render, seg, dt")
        put(f"B.patella.time_s.dt_run.{st}.min", min(t), Sd + f" timings.{st}", "s")
        put(f"B.patella.time_s.dt_run.{st}.max", max(t), Sd + f" timings.{st}", "s")
    # ---- patella B maps: the dt-inclusive run under the shipped engine (LH pad offset 'ceil', float32)
    PROV = "dt-inclusive patella run of 2026-09-18 under the shipped engine (LH pad offset 'ceil', float32; SEG 50 voxels); the same run supplies the SEG stage above"
    for mk, rk in MAP_KEYS_B_PAT.items():
        items = [pat_B_dt[k]["maps"][rk] for k in PAT_SUBJECTS]
        map_agg(items, Sd + f" maps.{rk}", f"B.patella.map.{mk}", note=PROV)
    items = [pat_B_dt[k]["seg"]["SEG"] for k in PAT_SUBJECTS]
    put("B.patella.dt_run.SEG.mismatches", sum(i["mismatches"] for i in items), Sd + " seg.SEG", "voxels", note="the SEG of the run that produced the patella B maps/metrics; equals B.patella.SEG.mismatches (same run)")
    put("B.patella.dt_run.SEG.exact_scans", sum(1 for i in items if i["mismatches"] == 0), Sd + " seg.SEG")
    assert F["B.patella.dt_run.SEG.mismatches"]["value"] == F["B.patella.SEG.mismatches"]["value"]
    put("B.patella.floor_run.SEG.mismatches", None, "retired 2026-09-18", note="v1 held the SEG (138 voxels) of the superseded floor-padding run validation/results/from_ipl_contour; that directory is no longer a source of any number. See B.patella.dt_run.SEG.*")
    put("B.patella.floor_run.SEG.exact_scans", None, "retired 2026-09-18", note="see B.patella.floor_run.SEG.mismatches")
    # ---- radius / tibia renderings, SEG, maps
    So = SRC_OSLH + "/*.json"
    for part in ("cort", "trab"):
        items = [oslh[k]["B"]["renderings"][part] for k in oslh]
        put(f"B.oslh.rendering.{part}.n_scans", len(items), So + f" B.renderings.{part}")
        put(f"B.oslh.rendering.{part}.exact_scans", sum(1 for i in items if i["mismatches"] == 0), So + f" B.renderings.{part}")
        put(f"B.oslh.rendering.{part}.mismatches", sum(i["mismatches"] for i in items), So + f" B.renderings.{part}", "voxels")
        put(f"B.oslh.rendering.{part}.dice.min", min(i["dice"] for i in items), So + f" B.renderings.{part}")
        put(f"B.oslh.rendering.{part}.voxels_ipl", sum(i["voxels_ipl"] for i in items), So + f" B.renderings.{part}", "voxels")
        for g, members in SITE_GROUPS.items():
            items = [oslh[k]["B"]["renderings"][part] for k in oslh if oslh_site[k] in members]
            put(f"B.site.{skey(g)}.rendering.{part}.exact_scans", sum(1 for i in items if i["mismatches"] == 0), So + f" B.renderings.{part} (bone_key {site_members_label(members)})")
            put(f"B.site.{skey(g)}.rendering.{part}.n_scans", len(items), So)
            put(f"B.site.{skey(g)}.rendering.{part}.mismatches", sum(i["mismatches"] for i in items), So, "voxels")
            put(f"B.site.{skey(g)}.rendering.{part}.voxels_ipl", sum(i["voxels_ipl"] for i in items), So, "voxels")
    items = [oslh[k]["B"]["seg"]["SEG"] for k in oslh]
    seg_agg(items, So + " B.seg.SEG", "B.oslh.SEG", list(oslh))
    put("B.oslh.SEG.label_mismatches", sum(i.get("label_mismatches", 0) for i in items), So + " B.seg.SEG.label_mismatches", "voxels")
    per = sorted(((oslh[k]["B"]["seg"]["SEG"]["mismatches"], k) for k in oslh), reverse=True)
    top3 = per[:3]
    put("B.oslh.SEG.top3_scans", [dict(id=k, site=oslh_site[k], mismatches=m) for m, k in top3], So + " B.seg.SEG (three largest)")
    put("B.oslh.SEG.top3_mismatches", sum(m for m, _ in top3), So + " B.seg.SEG", "voxels")
    put("B.oslh.SEG.top3_share_pct", 100.0 * sum(m for m, _ in top3) / sum(i["mismatches"] for i in items), So + " B.seg.SEG", "%")
    put("B.oslh.SEG.without_top3.mismatches", sum(i["mismatches"] for i in items) - sum(m for m, _ in top3), So + " B.seg.SEG", "voxels", note=f"the remaining {len(per) - 3} scans")
    put("B.oslh.SEG.without_top3.n_scans", len(per) - 3, So + " B.seg.SEG")
    put("B.oslh.SEG.without_top3.per_scan_max", per[3][0], So + " B.seg.SEG", "voxels")
    put("B.oslh.SEG.without_top3.per_scan_median", float(statistics.median(m for m, _ in per[3:])), So + " B.seg.SEG", "voxels")
    for g, members in SITE_GROUPS.items():
        its = [oslh[k]["B"]["seg"]["SEG"] for k in oslh if oslh_site[k] in members]
        seg_agg(its, So + f" B.seg.SEG (bone_key {site_members_label(members)})", f"B.site.{skey(g)}.SEG", [k for k in oslh if oslh_site[k] in members])
    TABLES["B.oslh.SEG.per_scan"] = [OrderedDict(scan=k, site=oslh_site[k], study=oslh_study[k], source=oslh_src[k], mismatches=oslh[k]["B"]["seg"]["SEG"]["mismatches"],
                                                 ours_only=oslh[k]["B"]["seg"]["SEG"]["ours_only"], ipl_only=oslh[k]["B"]["seg"]["SEG"]["ipl_only"],
                                                 dice=oslh[k]["B"]["seg"]["SEG"]["dice"], union_voxels=oslh[k]["B"]["seg"]["SEG"]["union_voxels"],
                                                 voxels_ipl=oslh[k]["B"]["seg"]["SEG"]["voxels_ipl"]) for k in oslh]
    # pooled SEG
    pooled = [pat_B_seg[k]["seg"]["SEG"] for k in PAT_SUBJECTS] + [oslh[k]["B"]["seg"]["SEG"] for k in oslh]
    seg_agg(pooled, "B.patella.SEG (ceil run) + B.oslh.SEG", "B.pooled.SEG", PAT_SUBJECTS + list(oslh))
    # radius / tibia B maps
    for mk, rk in MAP_KEYS_OSLH.items():
        items = [oslh[k]["B"]["maps"][rk] for k in oslh]
        map_agg(items, So + f" B.maps.{rk}", f"B.oslh.map.{mk}")
        for g, members in SITE_GROUPS.items():
            its = [oslh[k]["B"]["maps"][rk] for k in oslh if oslh_site[k] in members]
            map_agg(its, So + f" B.maps.{rk} (bone_key {site_members_label(members)})", f"B.site.{skey(g)}.map.{mk}")
        t = [r["B"]["timing"][rk] for r in oslh.values()]
        put(f"B.oslh.map.{mk}.time_s.median", float(statistics.median(t)), So + f" B.timing.{rk}", "s")
        put(f"B.oslh.map.{mk}.time_s.min", min(t), So + f" B.timing.{rk}", "s")
        put(f"B.oslh.map.{mk}.time_s.max", max(t), So + f" B.timing.{rk}", "s")
    for mk in ("TbTh_trabseg", "TbSp", "TbN", "CtTh"):
        for stat in ("n_scans", "exact_scans", "mismatches", "ours_only", "ipl_only", "voxels_compared", "support_ipl"):
            put(f"B.pooled.map.{mk}.{stat}", F[f"B.patella.map.{mk}.{stat}"]["value"] + F[f"B.oslh.map.{mk}.{stat}"]["value"],
                "sum of B.patella and B.oslh")
    # radius / tibia timings per stage
    n_por = sum(1 for r in oslh.values() if "porosity" in r["timings"])
    for st in ("read", "step1", "render_B", "lh", "seg", "dt_B", "total"):
        t = [r["timings"][st] for r in oslh.values()]
        put(f"B.oslh.time_s.{st}.median", float(statistics.median(t)), So + f" timings.{st}", "s",
            note={"dt_B": "dt of the five B maps (both Tb.Th objects, Tb.Sp, 1/Tb.N, Ct.Th), GPU",
                  "total": "read + step1 + render + LH + SEG + dt(A) + dt(B) + bookkeeping" + (f"; on the {n_por} records whose run included the cortical pore cascade also the cascade in both configurations (timings.porosity; see B.oslh.time_s.total_excl_porosity)" if n_por else ""),
                  "render_B": "contour rendering of the two compartments"}.get(st))
        put(f"B.oslh.time_s.{st}.min", min(t), So + f" timings.{st}", "s")
        put(f"B.oslh.time_s.{st}.max", max(t), So + f" timings.{st}", "s")
    # per site group: the pooled radius/tibia medians move with the site mix (diaphyseal volumes are smaller)
    for g, members in SITE_GROUPS.items():
        ids = [k for k in oslh if oslh_site[k] in members]
        src_runs = dict(Counter(oslh_src[k] for k in ids))
        for st in ("read", "step1", "render_B", "lh", "seg", "dt_B"):
            t = [oslh[k]["timings"][st] for k in ids]
            put(f"B.site.{skey(g)}.time_s.{st}.median", float(statistics.median(t)), So + f" timings.{st} (bone_key {site_members_label(members)})", "s",
                note=(f"{len(ids)} scans; validation runs {src_runs}" if st == "read" else None))
            put(f"B.site.{skey(g)}.time_s.{st}.min", min(t), So + f" timings.{st} (bone_key {site_members_label(members)})", "s")
            put(f"B.site.{skey(g)}.time_s.{st}.max", max(t), So + f" timings.{st} (bone_key {site_members_label(members)})", "s")
        for mk, rk in MAP_KEYS_OSLH.items():
            t = [oslh[k]["B"]["timing"][rk] for k in ids]
            put(f"B.site.{skey(g)}.map.{mk}.time_s.median", float(statistics.median(t)), So + f" B.timing.{rk} (bone_key {site_members_label(members)})", "s")
        grid = [int(np.prod(oslh[k]["grey"]["grid"][0])) for k in ids]
        put(f"B.site.{skey(g)}.grey_voxels.median", float(statistics.median(grid)), So + f" grey.grid (bone_key {site_members_label(members)})", "voxels",
            note="size of the greyscale AIM grid, the main driver of the run time")
    t = [r["timings"]["total"] - r["timings"].get("porosity", 0.0) for r in oslh.values()]
    put("B.oslh.time_s.total_excl_porosity.median", float(statistics.median(t)), So + " timings.total - timings.porosity", "s",
        note="the validation record total without the pore cascade, comparable across the two runs")
    put("B.oslh.time_s.total_excl_porosity.min", min(t), So + " timings.total - timings.porosity", "s")
    put("B.oslh.time_s.total_excl_porosity.max", max(t), So + " timings.total - timings.porosity", "s")
    tp = [r["timings"]["porosity"] for r in oslh.values() if "porosity" in r["timings"]]
    put("B.oslh.time_s.porosity.n", len(tp), So + " timings.porosity", note="records whose run included the cortical pore cascade; FACTS_BMD_CTPO pore.time_s.* covers all scans")
    if tp:
        put("B.oslh.time_s.porosity.median", float(statistics.median(tp)), So + " timings.porosity", "s", note="both configurations' cascades, IPL's PORE / CORT_SEG reads and the comparisons")
        put("B.oslh.time_s.porosity.min", min(tp), So + " timings.porosity", "s")
        put("B.oslh.time_s.porosity.max", max(tp), So + " timings.porosity", "s")
    rss = OrderedDict((k, r["memory"]["peak_rss_gb"]) for k, r in oslh.items())
    put("B.oslh.peak_rss_gb.max", max(rss.values()), So + " memory.peak_rss_gb", "GB",
        note="peak resident set of the validation worker process (both configurations and the comparisons, not the pipeline alone); "
             "peak_wset is the process's lifetime peak, so each record carries the running maximum of its worker and only the maximum over records is meaningful")
    put("B.oslh.peak_rss_gb.max_by_run", OrderedDict((name, max(v for k, v in rss.items() if oslh_src[k] == name)) for name in SRC_OSLH_DIRS),
        So + " memory.peak_rss_gb", "GB", note="the same maximum per validation run (results directory)")


# ================================================================================================ metrics
METRICS = OrderedDict([
    ("BVTV", dict(label="BV/TV", unit="-")),
    ("TbTh_trabseg", dict(label="Tb.Th (TRAB_SEG, Script 32 definition)", unit="mm")),
    ("TbTh_wholeseg", dict(label="Tb.Th (whole SEG; internal)", unit="mm")),
    ("TbTh_as_delivered", dict(label="Tb.Th (definition of IPL's delivered file: patella whole SEG, radius/tibia TRAB_SEG)", unit="mm")),
    ("TbSp", dict(label="Tb.Sp", unit="mm")),
    ("TbN", dict(label="Tb.N", unit="1/mm")),
    ("CtTh", dict(label="Ct.Th", unit="mm")),
])


def patella_pairs(cfg):
    """(ipl, ours) per metric for the patella under A (sample_means.csv; BV/TV from patella_bvtv_A) or B (the ceil_dt records)."""
    out = {m: OrderedDict() for m in METRICS}
    means = {row["subject"]: row for row in pat_A_means}
    for k in PAT_SUBJECTS:
        row = means[k]
        rb = pat_B_dt[k]
        ipl = dict(TbTh_trabseg=float(row["TbTh_old_ipl"]), TbTh_wholeseg=float(row["TbTh_ipl"]), TbSp=float(row["TbSp_ipl"]),
                   TbN=float(row["TbN_ipl"]), CtTh=float(row["CtTh_ipl"]))
        # the B records carry IPL's map means too; they must agree with sample_means (same IPL maps)
        ipl_b = dict(TbTh_trabseg=rb["maps"]["TRAB_TH_old"]["mean_ipl_mm"], TbTh_wholeseg=rb["maps"]["TRAB_TH"]["mean_ipl_mm"],
                     TbSp=rb["maps"]["TRAB_SP"]["mean_ipl_mm"], TbN=1.0 / rb["maps"]["TRAB_1N"]["mean_ipl_mm"], CtTh=rb["maps"]["CORT_TH"]["mean_ipl_mm"])
        for m in ipl:
            assert abs(ipl[m] - ipl_b[m]) <= 1e-9 * max(1.0, abs(ipl[m])), (k, m, ipl[m], ipl_b[m])
        if cfg == "A":
            ours = dict(TbTh_trabseg=float(row["TbTh_old_ours"]), TbTh_wholeseg=float(row["TbTh_ours"]), TbSp=float(row["TbSp_ours"]),
                        TbN=float(row["TbN_ours"]), CtTh=float(row["CtTh_ours"]))
            ra = pat_A_bvtv["records"][k]
            ipl["BVTV"] = rb["bvtv"]["ipl"]          # the ratio on IPL's SEG and IPL's rendered contour (validate_from_ipl_contour.ipl_bvtv)
            ours["BVTV"] = ra["bvtv"]                # the same ratio by validate_dataset.bvtv on the same files (patella_bvtv_A.py)
            assert ra["equals_ceil_dt_ipl"] and ours["BVTV"] == ipl["BVTV"], k
        else:
            mo = rb["morphometry"]
            ours = dict(TbTh_trabseg=mo["Tb_Th_old_mm"], TbTh_wholeseg=mo["Tb_Th_mm"], TbSp=mo["Tb_Sp_mm"], TbN=mo["Tb_N_per_mm"], CtTh=mo["Ct_Th_mm"])
            ipl["BVTV"] = rb["bvtv"]["ipl"]
            ours["BVTV"] = rb["bvtv"]["ours"]
        ipl["TbTh_as_delivered"] = ipl["TbTh_wholeseg"]
        ours["TbTh_as_delivered"] = ours["TbTh_wholeseg"]
        for m in ipl:
            out[m][k] = (ipl[m], ours[m])
    return out


def oslh_pairs(cfg):
    out = {m: OrderedDict() for m in METRICS}
    key = {"BVTV": "BVTV", "TbTh_trabseg": "TbTh_tseg", "TbTh_wholeseg": "TbTh_seg", "TbTh_as_delivered": "TbTh", "TbSp": "TbSp", "TbN": "TbN", "CtTh": "CtTh"}
    for k, r in oslh.items():
        assert r["trab_th_definition"]["value"] == "trab_seg", k
        mi, mo = r[cfg]["metrics_ipl"], r[cfg]["metrics_ours"]
        for m, rk in key.items():
            out[m][k] = (mi[rk], mo[rk])
    return out


def metrics_section():
    for cfg in ("A", "B"):
        pp = patella_pairs(cfg)
        op = oslh_pairs(cfg)
        src_p = {"A": SRC_PAT_A_MEANS + " (BV/TV: " + SRC_PAT_A_BVTV + ")", "B": SRC_PAT_B_DT + "/*.json (morphometry, bvtv, maps.*.mean_ipl_mm)"}[cfg]
        src_o = SRC_OSLH + f"/*.json {cfg}.metrics_ipl / {cfg}.metrics_ours"
        for m in METRICS:
            for grp_name, members in GROUPS.items():
                pairs = OrderedDict()
                if "patella" in members:
                    pairs.update(pp[m])
                for k in oslh:
                    if oslh_site[k] in members:
                        pairs[k] = op[m][k]
                if not pairs:
                    continue
                write_stats(cfg, m, grp_name, pairs, f"{src_p} + {src_o}" if grp_name == "pooled" else (src_p if grp_name == "patella" else src_o))
            for g, members in SITE_GROUPS.items():
                pairs = OrderedDict((k, op[m][k]) for k in oslh if oslh_site[k] in members)
                write_stats(cfg, m, "site." + skey(g), pairs, src_o + f" (bone_key {site_members_label(members)})")
        # per-scan table
        TABLES[f"metrics.{cfg}.per_scan"] = []
        for k in PAT_SUBJECTS:
            row = OrderedDict(scan=k, cohort="patella", study="PFJ")
            for m in METRICS:
                if k in pp[m]:
                    row[f"{m}_ipl"], row[f"{m}_ours"] = pp[m][k]
            TABLES[f"metrics.{cfg}.per_scan"].append(row)
        for k in oslh:
            row = OrderedDict(scan=k, cohort=oslh_site[k], study=oslh_study[k])
            for m in METRICS:
                row[f"{m}_ipl"], row[f"{m}_ours"] = op[m][k]
            TABLES[f"metrics.{cfg}.per_scan"].append(row)


def write_stats(cfg, m, grp, pairs, src):
    ipl = [v[0] for v in pairs.values()]
    ours = [v[1] for v in pairs.values()]
    st = stats_pair(ipl, ours)
    note = None
    if cfg == "A" and m == "BVTV":
        note = ("configuration A BV/TV = |SEG AND rendered trabecular contour| / |rendered trabecular contour| with IPL's SEG and IPL's contour on both sides: "
                "radius/tibia from the records' A.metrics (bvtv(A_frame['seg'], ipl_G['trab'])), patella from " + SRC_PAT_A_BVTV + " (the same function on IPL's files); "
                "IPL's value is the same ratio on the same files, so every pair is identical by construction; the patella sheet values (three decimals) equal the rounded ratio on "
                f"{F['A.patella.BVTV.printed.agree_after_rounding']['value']}/21, all 21 within 0.0006 (A.patella.BVTV.printed.*)")
    if cfg == "B" and grp in ("patella", "pooled"):
        note = (note + "; " if note else "") + "patella B metrics are from the dt-inclusive run under the shipped engine (" + SRC_PAT_B_DT + ")"
    for k, v in st.items():
        put(f"metrics.{cfg}.{grp}.{m}.{k}", v, src, METRICS[m]["unit"] if k in ("mean_ipl", "sd_ipl", "mean_ours", "sd_ours", "bias", "sd_diff", "loa_low", "loa_high", "mean_abs_diff", "max_abs_diff", "intercept") else None,
            note=note if k == "n" else None)


# ================================================================================================ parameters, software, old draft
def parameters_section():
    P = "ipldt/step1.py Step1Params (TIBIA = Script 32 literals; RADIUS = Script 33: corner_min 800, close2 30)"
    step1 = OrderedDict(sigma=2.0, support=3, lower_mgha=500.0, upper_mgha=3000.0, peel0=6, erode=3, dilate=3, close1=15, open_=15,
                        corner_erode=3, corner_min_tibia=200000, corner_min_radius=800, corner_dilate=3, corner_max=500000,
                        close2_tibia=50, close2_radius=30, slicewise_lo_pct=50.0, slicewise_up_pct=100.0)
    for k, v in step1.items():
        put(f"params.step1.{k}", v, P)
    put("params.step1.metric", "IPL chamfer metric-11 (3-4-5 type) morphology with the 3N + 2 threshold and IPL's boundary rules; 6-connected components; slicewise 4-connected extraction",
        "ipldt/ipl_ops.py docstrings; ipldt/ormir.py step3_trab_cort_seg docstring")
    put("params.step1.n_ipl_commands", 36, "ipldt/step1.py module docstring (36 IPL commands, 30 volume stages 00..29)")
    put("params.contour.command", "/togobj_from_aim -curvature_smooth 1 followed by /gobj_to_aim (rasterisation); slice-wise chain smoothing, 4-connected fill", "ipldt/contour/render.py, smooth.py")
    LH = "ipldt/ormir.py constants (LP_CUT_OFF_FREQ, LAPLACE_EPS, HAMMING_AMP, NORM_MAX_VALUE, LH_THRESHOLD, CC_MIN_VOXELS_*, SEG_VALUE_*, LH_PAD_OFFSET, LH_DTYPE)"
    lh = OrderedDict(lp_cut_off_freq=0.3, laplace_eps=0.45, hamming_amp=1.0, window="Hann (amp 1: (1 - A/2) + (A/2) cos)", norm_max=200000.0, short_max=32767,
                     threshold_permille_lower=475, threshold_permille_upper=1000, threshold_native_lower=15564, threshold_native_upper=32767,
                     cc_min_voxels_cort=35, cc_min_voxels_trab=70, seg_value_cort=127, seg_value_trab=126, border="1-voxel duplicated border (bounding_box_cut -border 1, offset_add, fill_offset_duplicate)",
                     pad_rule="next power of two per axis, edge-exclusive mirror ('reflect'); when the pad is odd the extra voxel goes BEFORE the data (LH_PAD_OFFSET = 'ceil')",
                     fft_dtype="float32 (LH_DTYPE; float64 opt-in)", short_conversion="trunc(float32(f) * float32(32767 / 200000)) after clipping",
                     cut_off_physical="0.3 / el_z cycles per mm (per-axis header element sizes)")
    for k, v in lh.items():
        put(f"params.lh.{k}", v, LH)
    DT = "ipldt/ormir.py DTParams (IPL_SCRIPT32); ipldt/core.py module docstring"
    dt = OrderedDict(ridge_epsilon=0.9, assign_epsilon=0.5, peel_iter=-1, version=3, suppress_boundary=2, ridge_tolerance=1e-9,
                     surface_distance="s = |p(v)|, p(u) = sign(u) max(|u| - 1/2, 0), exact integer 4 s^2 in int64, one float64 sqrt",
                     ridge_rule="x kept unless a 26-neighbour y inside the image has |x - y| + s_x - s_y <= ridge_epsilon",
                     diameter_v3="R = floor(D + 1/2) with D = |P1 - P2| when the pinned midpoint |M| < 1, else floor(2 s_x + 1/2)",
                     drawing="every voxel with |c - x| <= D/2 + assign_epsilon takes the largest D (atomicMax on the GPU)",
                     field="Danielsson-type vector distance transform, slice-interleaved sweep (ipldt.field.sir_quad); outside the image is object",
                     tb_n="Tb.N = 1 / mean of the 1/Tb.N map (dt_number) inside the trabecular contour")
    for k, v in dt.items():
        put(f"params.dt.{k}", v, DT)
    put("params.dt.tbth_object.shipped_default", "whole SEG (label 127 + 126) with the trabecular contour as gobj -- ipldt.ormir.ipl_morphometry key TRAB_TH, called by ormir_bqrl.stages.morphometry",
        "ipldt/ormir.py ipl_morphometry; ormir_bqrl/stages.py morphometry", note="Script 32 computes Tb.Th on TRAB_SEG (IPL_FNAME5); ipl_morphometry exposes that object only as which='TRAB_TH_old' with trab_seg given")
    put("params.dt.tbth_object.script32", "TRAB_SEG (label 126, components >= 70 voxels, cropped to the trabecular contour) with the trabecular contour as gobj", "ipldt/ormir.py ipl_morphometry docstring; validation summaries 'Tb.Th (TRAB_SEG, Scripts 32/33/34)'")
    put("params.dt.ctth_object", "CORT_MASK raster (the cortical compartment) with the cortical contour as gobj", "ipldt/ormir.py ipl_morphometry")
    put("params.autocontour", "ORMIR-XCT 1.1.0 ormir_xct.core.segmentation.autocontour.autocontour (AutocontourKnee.get_periosteal_mask; Buie 2007 dual threshold); called with the AIM's own mu_water, rescale slope and intercept",
        "ipldt/ormir.py step2_autocontour; installed ormir_xct 1.1.0")
    put("params.bmd", "ORMIR-XCT bmd_masked (HU -> mgHA/cm3 with the AIM calibration) inside the rendered trabecular and cortical contours", "ipldt/ormir.py step5_bmd")
    put("params.calibration", "density slope, intercept and mu_scaling read from the AIM processing log (ITK ScancoImageIO header as fallback); /seg_gauss thresholds 500 / 3000 mgHA -> native units per scan (e.g. 4524 / 17173 or 4507 / 17095 on the patellae)",
        "ipldt/ormir.py step1_calibration; records step1.thresholds")


def software_section():
    put("software.ipldt.version", "1.0.0", "ipldt/__init__.py (__version__); pyproject.toml")
    put("software.ormir_bqrl.version", "0.1.0", "ormir_bqrl/__init__.py (__version__); README.md")
    put("software.ormir_xct.version", "1.1.0", "installed package in the ormir env (ormir_xct.__version__)")
    put("software.tests.collected_total", 434, "pytest --collect-only -q tests (2026-09-27)")
    put("software.tests.gpu_cpu_identity", 26, "pytest --collect-only tests/test_gpu_cpu.py (26 of the 43 collected together with test_ridge_precision.py's 17; 2026-09-18)",
        note="tests/test_gpu_cpu.py: 'GPU (CuPy) and CPU backends must be bit-identical: maps, centres and reports, for all three functions, all three diameter versions, with and without a gobj, and across assign_epsilon'; the GPU tests skip without CUDA")
    put("software.tests.ridge_precision", 17, "pytest --collect-only tests/test_ridge_precision.py (2026-09-18)", note="float64 ridge test against a 60-digit decimal reference, CPU and GPU bit identity")
    put("software.gpu_cpu_identity.evidence", "tests/test_gpu_cpu.py (26 tests: maps, centres, reports identical for dt_thickness / dt_spacing / dt_number, versions 1-3, with and without gobj, assign_epsilon sweep; ridge_gpu == ridge, draw_spheres_gpu == CPU) and tests/test_ridge_precision.py (GPU bit identity of the ridge test)",
        "tests/", note="all cohort runs in the records used backend 'gpu' (dt_backend / backend fields); the CPU-vs-GPU identity is established by the test suite on phantoms, not by a second cohort run")
    put("software.backends", "GPU: CuPy (ridge test and sphere drawing kernels); CPU: numpy + numba (the vector-distance sweep is numba on both)", "ipldt/gpu.py, ipldt/field.py")
    put("software.records_backend", "gpu", f"dt_backend of all {N_OSLH} radius/tibia and 21 patella-B records; backend of all 105 patella-A records")
    put("software.lh_engine_state", "pad offset 'ceil' + float32 (the shipped engine) in every configuration-B number of this sheet (B.patella.*, B.oslh.*, metrics.B.*)", "records variants.lh_pad_offset / seg.lh_pad_offset")
    put("software.outputs", "report JSON/CSV; masks and maps as NIfTI or AIM; Slicer-ready labelmaps (.seg.nrrd); manual-correction re-entry (ormir_bqrl.redo, Script-34 REDO semantics)",
        "README.md; ipldt/ormir.py run_pipeline and ormir_bqrl/stages.py write_volumes (NIfTI or AIM: --map-format of ipldt-pipeline and ormir-bqrl, the pore map included; ORMIR-BQRL adds the Slicer files and the redo)")


# ================================================================================================ environment (measured, not from the records)
def environment_section():
    """software.environment.* and software.packaging.*: measured 2026-09-18 in the validation environment on the validation
    workstation, and pyproject.toml."""
    E = ("measured 2026-09-18 in the validation environment on the validation workstation (sys.version, module __version__, "
         "cupy.cuda.runtime, Win32_Processor / Win32_ComputerSystem); the records store dt_backend 'gpu' but no library versions")
    put("software.environment.python", "3.11.15", E)
    put("software.environment.numpy", "2.3.5", E)
    put("software.environment.scipy", "1.15.3", E)
    put("software.environment.numba", "0.66.0", E)
    put("software.environment.cupy", "14.2.0", E, note="cupy-cuda12x")
    put("software.environment.cuda_runtime", "12.9", E, note="runtimeGetVersion 12090")
    put("software.environment.itk", "5.4.6", E)
    put("software.environment.simpleitk", "2.5.5", E)
    put("software.environment.gpu", "NVIDIA GeForce RTX 4090", E)
    put("software.environment.gpu_memory_gb", 24, E, "GB", note="totalGlobalMem 25.76e9 bytes = 24 GiB")
    put("software.environment.cpu", "AMD Ryzen Threadripper 9960X", E)
    put("software.environment.cpu_cores", 24, E, note="48 logical processors")
    put("software.environment.ram_gb", 96, E, "GB", note="TotalPhysicalMemory 95.3 GiB")
    put("software.environment.os", "Windows 11 Pro", E)
    put("software.packaging.requires_python", ">=3.10", "pyproject.toml requires-python")
    put("software.packaging.license", "MIT", "pyproject.toml classifiers; LICENSE")
    put("software.packaging.dependencies", "numpy>=1.24, scipy>=1.10, numba>=0.58; extras gpu (cupy-cuda12x), io (itk, itk-ioscanco, SimpleITK), ormir (ormir-xct), bqrl (matplotlib>=3.7)", "pyproject.toml")


# ================================================================================================ FACTS.md
def r(v, nd=6):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,d}"
    if isinstance(v, float):
        if v == 0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e6:
            return f"{v:.3e}"
        return f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v}"
    return str(v)


def fv(key, nd=None):
    if nd is None:
        last = key.rsplit(".", 1)[-1]
        nd = 8 if ".dice." in key else (7 if last in ("r2", "icc21", "pearson_r") else 6)
    return r(F[key]["value"], nd)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def fmt_counts(d):
    return ", ".join(f"{k} ({v})" for k, v in d.items()) if len(d) > 1 else ", ".join(str(k) for k in d)


def write_facts_md(gap_list):
    N = N_TOTAL
    L = []
    A = L.append
    A(f"# FACTS -- the numbers of the ipldt / ORMIR-BQRL manuscript (n = {N})")
    A("")
    A(f"Built by `manuscript/facts/build_facts.py` on {META_BUILT} ({META_VERSION_SHORT}) from the validation records only. `facts.json` holds every "
      "value at full precision under the key printed in the `key` column; this file rounds. Cite keys, not prose.")
    A("")
    A("**Sources.** Patella configuration A: `validation/results/records.json` (105 = 21 subjects x 5 maps), `validation/results/sample_means.csv` and, for BV/TV, "
      "`validation/results/patella_bvtv_A/records.json` (`validation/patella_bvtv_A.py`: the ratio on IPL's SEG and IPL's rendered trabecular contour, with IPL's printed sheet value beside it). "
      "Patella configuration B, every stage (STEP 1, contours, SEG, dt maps, metrics, BV/TV): `validation/results/from_ipl_contour_ceil_dt/records/*.json`, the dt-inclusive run of 2026-09-18 "
      "under the shipped engine (LH pad offset 'ceil', float32); its SEG stage is asserted identical to the SEG-only run `validation/results/from_ipl_contour_ceil/records/*.json`. "
      "Radius/tibia, A and B: " + " and ".join(f"`{d}/*.json` ({OSLH_EXPECT[n]} files, {Counter(oslh_src[k] for k in oslh)[n]} counted)" for n, d in SRC_OSLH_DIRS.items())
      + f" matching `^[A-Za-z]+_[A-Za-z]+_\\d+\\.json$`; `{', '.join(sorted(OSLH_DROP))}` is not used; n = {N_OSLH}. Study names: `{SRC_STUDY_TABLE}` and meta.study_label. "
      "Parameters: `ipldt/step1.py`, `ipldt/ormir.py`, `ipldt/core.py`.")
    A("")
    A("**Configurations.** A = IPL's own SEG and contour renderings in, ipldt's distance-transform engine and cortical pore cascade out (tests modules 6-7). "
      "B = IPL's periosteal contour in; ipldt's compartment separation, contour rendering, Laplace-Hamming segmentation, distance transforms and cortical pore cascade out (tests modules 3-7). "
      "The reference is IPL V5.42; the metric is voxels that differ by global position; 'exact' means 0 differing voxels. "
      "Cortical porosity is in `FACTS_BMD_CTPO.md` (`make_facts_bmd_ctpo.py`; the file name is historical).")
    A("")
    A(f"**Tb.Th definitions.** `TbTh_trabseg` = dt_thickness on TRAB_SEG cropped to the trabecular contour (what Script 32 computes; exists for all {N} scans: "
      "patella IPL file `<base>_TRAB_TH_old`, radius/tibia file `<base>_TRAB_TH`). `TbTh_wholeseg` = dt_thickness on the whole SEG (internal; the patella "
      "IPL file `<base>_TRAB_TH`). `TbTh_as_delivered` = the definition of IPL's delivered TRAB_TH file per cohort (patella whole SEG, radius/tibia TRAB_SEG); "
      f"The paper reports `TbTh_trabseg` (one definition, all {N}).")
    A("")
    A("**Site groups.** `site.UD_radius`, `site.UD_tibia`, `site.D_radius`, `site.D_tibia` (meta.bone_key of the record) and `site.diaphyseal` = D radius + D tibia pooled, "
      "the figures' 'diaphyseal' cohort. `oslh` = all radius/tibia scans; `pooled` = patellae + radius/tibia.")
    A("")
    # ---------------- cohort
    A("## 1. Cohort")
    A("")
    A(f"Table 1 rows (`tables.cohort.table1`; every column read from the records):")
    A("")
    rows = []
    for t in TABLES["cohort.table1"]:
        rows.append([t["site"], f"{t['n']:,d}", fmt_counts(t["studies"]), fmt_counts(t["stack_lengths"]), fmt_counts(t["preset"]), f"{t['participants']:,d}",
                     ("cohort.patella.*" if t["key"] == "patella" else ("cohort.n_total, cohort.participants.total" if t["key"] == "total" else f"cohort.site.{t['key']}.*"))])
    A(md_table(["site", "n", "study (scans)", "stack length, slices (scans)", "STEP-1 parameter set (scans)", "participants", "key"], rows))
    A("")
    for st in STUDIES:
        spp = F[f"cohort.oslh.study.{st}.scans_per_participant"]["value"]
        spp_s = ", ".join(f"{b} participant(s) with {a} scan(s)" for a, b in spp.items())
        A(f"- Study {st} (`cohort.oslh.study.{st}.*`): {fv(f'cohort.oslh.study.{st}.n')} scan(s), {fv(f'cohort.oslh.study.{st}.participants')} participant(s) "
          f"({spp_s}), same-site repeats {fv(f'cohort.oslh.study.{st}.same_site_repeats')}; {F[f'cohort.oslh.study.{st}.n']['note']}.")
    A(f"- Patellae: study {F['cohort.patella.study']['value']} (`cohort.patella.study`), {fv('cohort.patella.participants')} participants, {fv('cohort.patella.participants_bilateral')} with both knees; "
      f"STEP-1 preset {F['cohort.patella.preset']['value']} (`cohort.patella.preset`). Participants over all studies: {fv('cohort.participants.total')} (`cohort.participants.total`; {F['cohort.participants.total']['note']}).")
    fixes = F["cohort.oslh.participant_code_fixes"]["value"]
    A(f"- Participant codes normalised for one-character typos (`cohort.oslh.participant_code_fixes`): " + ("; ".join(f"{d['id']} ({d['note']})" for d in fixes) or "none") + ".")
    A(f"- Radius/tibia presets: tibia {fv('cohort.oslh.preset.tibia.n')} / radius {fv('cohort.oslh.preset.radius.n')} (`cohort.oslh.preset.*`); preset source {F['cohort.oslh.preset.source']['value']}. "
      f"Scans evaluated with the other site's parameter set: {fv('cohort.oslh.preset.other_site.n')} -- "
      + ("; ".join(f"{d['id']} ({d['site']}, {d['preset']} preset, rendering voxels differing per preset tried {d['trials']})" for d in F["cohort.oslh.preset.other_site.scans"]["value"]) or "none")
      + " (`cohort.oslh.preset.other_site.*`).")
    if "cohort.oslh.preset.checked_scans" in F:
        A("- Checked individually (`cohort.oslh.preset.checked_scans`): " + "; ".join(
            f"{d['id']} {d['site']} (scanner site index {d['site_index']}): record preset {d['preset']} by {d['source']}, trials {d['trials']}" for d in F["cohort.oslh.preset.checked_scans"]["value"]) + ".")
    if F["cohort.oslh.preset.launch_script.scans_with_file"]["value"] is not None:
        A(f"- Launch-script check (`cohort.oslh.preset.launch_script.*`): {fv('cohort.oslh.preset.launch_script.scans_with_file')} of {N_OSLH} counted radius/tibia runs have their "
          f"launch script available; on {fv('cohort.oslh.preset.launch_script.agrees')} of them IPL_MISC1 names the record's preset. Listed: "
          + "; ".join(f"{d['id']} {d['site']}: record {d['record_preset']}, " + ", ".join(f"{c['file']} IPL_MISC1 {c['IPL_MISC1']} EVAL_SITE {c['EVAL_SITE']}" for c in d["com"])
                      for d in F["cohort.oslh.preset.launch_script.listed"]["value"]) + ".")
    A(f"- Scanner XtremeCT II (Scanco Medical AG); IPL {fv('cohort.ipl_version')} (`cohort.ipl_version`); nominal voxel size {fv('cohort.voxel_size_mm.nominal')} mm "
      f"(header element sizes x {fv('cohort.voxel_size_mm.header_x.min', 7)}-{fv('cohort.voxel_size_mm.header_x.max', 7)} mm, z {fv('cohort.voxel_size_mm.header_z.min', 7)}-{fv('cohort.voxel_size_mm.header_z.max', 7)} mm).")
    A(f"- Slices per scan: patella {F['cohort.patella.slices.counts']['value']}; radius/tibia {F['cohort.oslh.slices.counts']['value']} (`cohort.*.slices.counts`).")
    A(f"- Patella acquisition: {F['cohort.patella.acquisition']['value']} (`cohort.patella.acquisition`).")
    A("")
    A("### Voxels compared")
    A("")
    rows = [
        ["A, patella, each trabecular map (SEG grid)", fv("A.patella.voxels_compared.seg_grid"), "A.patella.voxels_compared.seg_grid"],
        ["A, patella, Ct.Th (CORT_MASK grid)", fv("A.patella.voxels_compared.cort_grid"), "A.patella.voxels_compared.cort_grid"],
        ["A, patella, four reported maps", fv("A.patella.voxels_compared.total_four_maps"), "A.patella.voxels_compared.total_four_maps"],
        ["A, radius/tibia, Tb.Th (TRAB_SEG) map", fv("A.oslh.map.TbTh_trabseg.voxels_compared"), "A.oslh.map.TbTh_trabseg.voxels_compared"],
        [f"A, pooled {N}, four reported maps", fv("A.pooled.voxels_compared.four_maps"), "A.pooled.voxels_compared.four_maps"],
        ["B, patella, SEG (union grids)", fv("B.patella.SEG.voxels_compared"), "B.patella.SEG.voxels_compared"],
        ["B, radius/tibia, SEG (union grids)", fv("B.oslh.SEG.voxels_compared"), "B.oslh.SEG.voxels_compared"],
        [f"B, pooled {N}, SEG", fv("B.pooled.SEG.voxels_compared"), "B.pooled.SEG.voxels_compared"],
        [f"B, pooled {N}, Tb.Th (TRAB_SEG) map", fv("B.pooled.map.TbTh_trabseg.voxels_compared"), "B.pooled.map.TbTh_trabseg.voxels_compared"],
    ]
    A(md_table(["comparison", "voxels", "key"], rows))
    A("")
    # ---------------- per module
    A("## 2. Agreement per module")
    A("")
    A("### 2.1 Cortical / trabecular compartment separation (STEP 1) and contour rendering -- configuration B")
    A("")
    rows = []
    for part, lab in (("cort", "cortical"), ("trab", "trabecular")):
        rows.append([f"patella, {lab} mask (STEP 1)", f"{fv(f'B.patella.step1.{part}.exact_scans')}/{fv(f'B.patella.step1.{part}.n_scans')}", fv(f"B.patella.step1.{part}.mismatches"), fv(f"B.patella.step1.{part}.voxels_ipl"), fv(f"B.patella.step1.{part}.dice.min"), f"B.patella.step1.{part}.*"])
    for part, lab in (("cort", "cortical"), ("trab", "trabecular")):
        rows.append([f"patella, {lab} contour rendering", f"{fv(f'B.patella.contour.{part}.exact_scans')}/{fv(f'B.patella.contour.{part}.n_scans')}", fv(f"B.patella.contour.{part}.mismatches"), fv(f"B.patella.contour.{part}.voxels_ipl"), "1", f"B.patella.contour.{part}.*"])
    for part, lab in (("cort", "cortical"), ("trab", "trabecular")):
        rows.append([f"radius/tibia, {lab} contour rendering (STEP 1 -> contour)", f"{fv(f'B.oslh.rendering.{part}.exact_scans')}/{fv(f'B.oslh.rendering.{part}.n_scans')}", fv(f"B.oslh.rendering.{part}.mismatches"), fv(f"B.oslh.rendering.{part}.voxels_ipl"), fv(f"B.oslh.rendering.{part}.dice.min"), f"B.oslh.rendering.{part}.*"])
    for g in SITE_GROUPS:
        sk = skey(g)
        rows.append([f"  {g}, cortical / trabecular renderings", f"{fv(f'B.site.{sk}.rendering.cort.exact_scans')}/{fv(f'B.site.{sk}.rendering.cort.n_scans')} / {fv(f'B.site.{sk}.rendering.trab.exact_scans')}/{fv(f'B.site.{sk}.rendering.trab.n_scans')}", f"{fv(f'B.site.{sk}.rendering.cort.mismatches')} / {fv(f'B.site.{sk}.rendering.trab.mismatches')}", f"{fv(f'B.site.{sk}.rendering.cort.voxels_ipl')} / {fv(f'B.site.{sk}.rendering.trab.voxels_ipl')}", "", f"B.site.{sk}.rendering.*"])
    A(md_table(["stage", "exact scans", "mismatching voxels", "IPL set voxels", "min Dice", "key"], rows))
    A("")
    ex_c = F["B.patella.step1.cort.exact_scans"]["value"] + F["B.oslh.rendering.cort.exact_scans"]["value"]
    ex_t = F["B.patella.step1.trab.exact_scans"]["value"] + F["B.oslh.rendering.trab.exact_scans"]["value"]
    mm_ = sum(F[k]["value"] for k in ("B.patella.step1.cort.mismatches", "B.patella.step1.trab.mismatches", "B.oslh.rendering.cort.mismatches", "B.oslh.rendering.trab.mismatches"))
    A(f"Radius/tibia STEP-1 masks are validated through their renderings (IPL's raw mask rasters were not exported); pooled over {N} scans the cortical compartment is exact on "
      f"{ex_c}/{N} and the trabecular on {ex_t}/{N}, with {mm_:,d} differing voxels.")
    A("")
    A("### 2.2 Laplace-Hamming segmentation (SEG) -- configuration B")
    A("")
    rows = []
    for grp, lab in (("patella", "patella"), ("oslh", f"radius/tibia ({N_OSLH})"), ("pooled", f"pooled ({N})")):
        p = f"B.{grp}.SEG"
        rows.append([lab, f"{fv(p + '.exact_scans')}/{fv(p + '.n_scans')}", fv(p + ".mismatches"), fv(p + ".ours_only"), fv(p + ".ipl_only"),
                     f"{fv(p + '.per_scan.min')} / {fv(p + '.per_scan.median')} / {fv(p + '.per_scan.max')}", fv(p + ".dice.min"), fv(p + ".dice.median"),
                     f"{fv(p + '.dice.ge_0_9999')}/{fv(p + '.n_scans')}", fv(p + ".mismatch_ppm", 3), p + ".*"])
    for g in SITE_GROUPS:
        p = f"B.site.{skey(g)}.SEG"
        rows.append([f"  {g}", f"{fv(p + '.exact_scans')}/{fv(p + '.n_scans')}", fv(p + ".mismatches"), fv(p + ".ours_only"), fv(p + ".ipl_only"),
                     f"{fv(p + '.per_scan.min')} / {fv(p + '.per_scan.median')} / {fv(p + '.per_scan.max')}", fv(p + ".dice.min"), fv(p + ".dice.median"),
                     f"{fv(p + '.dice.ge_0_9999')}/{fv(p + '.n_scans')}", fv(p + ".mismatch_ppm", 3), p + ".*"])
    A(md_table(["cohort", "exact", "mismatching voxels", "ours only", "IPL only", "per scan min / median / max", "min Dice", "median Dice", "Dice >= 0.9999", "per million", "key"], rows))
    A("")
    t3 = F["B.oslh.SEG.top3_scans"]["value"]
    t3s = ", ".join("{} {} {:,d}".format(d["id"], d["site"], d["mismatches"]) for d in t3)
    A(f"- Radius/tibia: the three scans with the largest SEG residual carry {fv('B.oslh.SEG.top3_mismatches')} voxels = {fv('B.oslh.SEG.top3_share_pct', 1)} % of the total "
      f"({t3s}); the other {fv('B.oslh.SEG.without_top3.n_scans')} scans total {fv('B.oslh.SEG.without_top3.mismatches')} voxels (median {fv('B.oslh.SEG.without_top3.per_scan_median')}, max {fv('B.oslh.SEG.without_top3.per_scan_max')}) (`B.oslh.SEG.top3_*`, `B.oslh.SEG.without_top3.*`).")
    A(f"- Scans with <= 10 / <= 100 / <= 1,000 differing SEG voxels, pooled: {fv('B.pooled.SEG.per_scan.le_10')} / {fv('B.pooled.SEG.per_scan.le_100')} / {fv('B.pooled.SEG.per_scan.le_1000')} of {N} (`B.pooled.SEG.per_scan.le_*`).")
    A(f"- Label mismatches (set in both, different label): patella {fv('B.patella.SEG.label_mismatches')}, radius/tibia {fv('B.oslh.SEG.label_mismatches')}.")
    A(f"- Patella TRAB_SEG: {fv('B.patella.TRAB_SEG.mismatches')} voxels ({fv('B.patella.TRAB_SEG.ours_only')} ours only / {fv('B.patella.TRAB_SEG.ipl_only')} IPL only), exact {fv('B.patella.TRAB_SEG.exact_scans')}/21, min Dice {fv('B.patella.TRAB_SEG.dice.min')}; "
      f"CORT_SEG: {fv('B.patella.CORT_SEG.mismatches')} voxels ({fv('B.patella.CORT_SEG.ours_only')} / {fv('B.patella.CORT_SEG.ipl_only')}), exact {fv('B.patella.CORT_SEG.exact_scans')}/21, min Dice {fv('B.patella.CORT_SEG.dice.min')} (`B.patella.TRAB_SEG.*`, `B.patella.CORT_SEG.*`). "
      "Radius/tibia: no TRAB_SEG files delivered; CORT_SEG comparisons are in FACTS_BMD_CTPO (pore.B.cort_seg.*).")
    A("")
    A("### 2.3 Distance-transform maps -- configuration A (IPL's SEG and contours in)")
    A("")
    rows = []
    for mk in ("TbTh_trabseg", "TbSp", "TbN", "CtTh", "TbTh_wholeseg"):
        for grp, lab in (("patella", "patella"), ("oslh", "radius/tibia"), ("pooled", f"pooled {N}")):
            if grp == "pooled" and mk == "TbTh_wholeseg":
                continue
            p = f"A.{grp}.map.{mk}"
            rows.append([MAP_LABEL[mk], lab, f"{fv(p + '.exact_scans')}/{fv(p + '.n_scans')}", fv(p + ".mismatches"), fv(p + ".ours_only"), fv(p + ".ipl_only"), fv(p + ".voxels_compared"), fv(p + ".support_ipl"), p + ".*"])
    A(md_table(["map", "cohort", "exact scans", "mismatching voxels", "ours only", "IPL only", "voxels compared", "IPL non-zero voxels", "key"], rows))
    A("")
    hits = F["A.oslh.map.TbTh_trabseg.nonexact_scans"]["value"]
    hits_s = ", ".join("{} ({} scan), Tb.Th (TRAB_SEG) map, {} voxel(s) ({} ours only / {} IPL only)".format(h["id"], h["site"], h["mismatches"], h["ours_only"], h["ipl_only"]) for h in hits) or "none"
    A(f"- The differing voxels of the pooled configuration A (four reported maps, {fv('A.pooled.voxels_compared.four_maps')} voxels compared, {fv('A.pooled.mismatches.four_maps')} differing; "
      f"{fv('A.pooled.exact_comparisons.four_maps')} of {4 * N} scan-map comparisons exact): {hits_s} (`A.oslh.map.TbTh_trabseg.nonexact_scans`).")
    parts = []
    for g in SITE_GROUPS:
        sk = skey(g)
        parts.append("{} {}/{}, {}, {}, {}".format(g, fv(f"A.site.{sk}.map.TbTh_trabseg.exact_scans"), fv(f"A.site.{sk}.map.TbTh_trabseg.n_scans"),
                                                  fv(f"A.site.{sk}.map.TbSp.exact_scans"), fv(f"A.site.{sk}.map.TbN.exact_scans"), fv(f"A.site.{sk}.map.CtTh.exact_scans")))
    A("- Per site, configuration A, Tb.Th (TRAB_SEG) / Tb.Sp / 1/Tb.N / Ct.Th exact scans: " + "; ".join(parts) + " (`A.site.*`).")
    A(f"- The whole-SEG Tb.Th object (internal) differs from IPL's delivered radius/tibia Tb.Th files by {fv('A.oslh.map.TbTh_wholeseg.mismatches')} voxels "
      f"({fv('A.oslh.map.TbTh_wholeseg.exact_scans')}/{fv('A.oslh.map.TbTh_wholeseg.n_scans')} exact): IPL's standard evaluation uses TRAB_SEG.")
    A("")
    A("### 2.4 Distance-transform maps -- configuration B (whole chain from IPL's periosteal contour)")
    A("")
    rows = []
    for mk in ("TbTh_trabseg", "TbSp", "TbN", "CtTh"):
        for grp, lab in (("patella", "patella"), ("oslh", "radius/tibia"), ("pooled", f"pooled {N}")):
            p = f"B.{grp}.map.{mk}"
            rows.append([MAP_LABEL[mk], lab, f"{fv(p + '.exact_scans')}/{fv(p + '.n_scans')}", fv(p + ".mismatches"), fv(p + ".ours_only"), fv(p + ".ipl_only"), fv(p + ".voxels_compared"), p + ".*"])
    A(md_table(["map", "cohort", "exact scans", "mismatching voxels", "ours only", "IPL only", "voxels compared", "key"], rows))
    A("")
    parts = []
    for g in SITE_GROUPS:
        sk = skey(g)
        parts.append("{}: Tb.Th {} ({}/{} exact), Tb.Sp {} ({}), 1/Tb.N {} ({}), Ct.Th {} ({})".format(
            g, fv(f"B.site.{sk}.map.TbTh_trabseg.mismatches"), fv(f"B.site.{sk}.map.TbTh_trabseg.exact_scans"), fv(f"B.site.{sk}.map.TbTh_trabseg.n_scans"),
            fv(f"B.site.{sk}.map.TbSp.mismatches"), fv(f"B.site.{sk}.map.TbSp.exact_scans"), fv(f"B.site.{sk}.map.TbN.mismatches"), fv(f"B.site.{sk}.map.TbN.exact_scans"),
            fv(f"B.site.{sk}.map.CtTh.mismatches"), fv(f"B.site.{sk}.map.CtTh.exact_scans")))
    A("Per site (B): " + "; ".join(parts) + " (`B.site.*.map.*`).")
    A("")
    # ---------------- metrics
    A("## 3. Scalar metrics (ours vs IPL; IPL is x)")
    A("")
    A("Statistics: mean +- SD of both, bias = mean(ours - IPL), SD of the differences, 95 % limits of agreement = bias +- 1.96 SD, max |diff|, max |rel diff| in %, "
      "exact = scans with identical values (to 12 significant digits; differences of a few 1e-16 between two identical maps averaged by different code paths are treated as 0, the bitwise count is `exact_bitwise`; when every pair is exact the fit is the identity), "
      "OLS slope / intercept of ours on IPL, R^2, ICC(2,1) (two-way random, absolute agreement, single measurement). "
      "Metric values at the nominal 0.0607 mm voxel size. Keys: `metrics.<A|B>.<patella|oslh|pooled|site.X>.<metric>.<stat>`.")
    A("")
    grp_rows = [("pooled", "pooled"), ("patella", "patella"), ("oslh", "radius/tibia")] + [("site." + skey(g), g) for g in SITE_GROUPS]
    for cfg, title in (("B", f"3.1 Configuration B (the whole chain), pooled n = {N} and per cohort"), ("A", "3.2 Configuration A (IPL's SEG in), pooled and per cohort")):
        A(f"### {title}")
        A("")
        rows = []
        for m in ("BVTV", "TbTh_trabseg", "TbSp", "TbN", "CtTh", "TbTh_as_delivered", "TbTh_wholeseg"):
            for grp, lab in grp_rows:
                p = f"metrics.{cfg}.{grp}.{m}"
                if p + ".n" not in F:
                    continue
                rows.append([METRICS[m]["label"], lab, fv(p + ".n"), f"{fv(p + '.mean_ipl', 6)} +- {fv(p + '.sd_ipl', 6)}", f"{fv(p + '.mean_ours', 6)} +- {fv(p + '.sd_ours', 6)}",
                             fv(p + ".bias"), fv(p + ".sd_diff"), f"[{fv(p + '.loa_low')}, {fv(p + '.loa_high')}]", fv(p + ".max_abs_diff"), fv(p + ".max_rel_pct", 4),
                             f"{fv(p + '.exact')}/{fv(p + '.n')}", fv(p + ".slope", 6), fv(p + ".intercept"), fv(p + ".r2"), fv(p + ".icc21"), p + ".*"])
        A(md_table(["metric", "cohort", "n", "IPL mean +- SD", "ours mean +- SD", "bias", "SD diff", "95 % LoA", "max |diff|", "max |rel| %", "exact", "slope", "intercept", "R^2", "ICC(2,1)", "key"], rows))
        A("")
    A(f"Per-scan pairs for the figures: `facts.json` -> `tables.metrics.A.per_scan`, `tables.metrics.B.per_scan` ({N} rows each, columns `<metric>_ipl` / `<metric>_ours`, `cohort`, `study`).")
    A("")
    A(f"- Configuration A BV/TV, patella (n = 21): ipldt's ratio on IPL's SEG and IPL's rendered trabecular contour equals the ratio taken on the same files by the B run's IPL side on "
      f"{fv('A.patella.BVTV.exact_scans')}/21 (BV {fv('A.patella.BVTV.bv_voxels')} / TV {fv('A.patella.BVTV.tv_voxels')} voxels pooled), so the configuration-A pair is identical by construction, as for the radius/tibia. "
      f"IPL's printed sheet BV/TV (three decimals) exists for {fv('A.patella.BVTV.printed.n_with_value')}/21 and equals the rounded ratio on {fv('A.patella.BVTV.printed.agree_after_rounding')}/21 "
      f"(configuration B's ratio: {fv('A.patella.BVTV.printed.agree_after_rounding_B')}/21); max |ratio - printed| {fv('A.patella.BVTV.printed.max_abs_diff')} (rounding bound 0.0005), the "
      f"{len(F['A.patella.BVTV.printed.differing_scans']['value'])} differing ratios lie within {fv('A.patella.BVTV.printed.differing_max_distance_to_boundary')} of a rounding boundary. "
      "Table footnote: IPL's BV/TV is the same ratio taken on IPL's own files; Script 32 writes no BV/TV and the sheet prints three decimals (`A.patella.BVTV.*`; per scan `tables.A.patella.BVTV.per_scan`).")
    A("")
    # ---------------- timings
    A("## 4. Timings (GPU backend in every record) and memory")
    A("")
    rows = []

    def trow(lab, base, key_suffix=""):
        rows.append([lab, fv(base + ".median", 1), f"{fv(base + '.min', 1)} - {fv(base + '.max', 1)}", base + ".*"])
    trow("radius/tibia: AIM read", "B.oslh.time_s.read")
    trow("radius/tibia: STEP 1 compartment separation", "B.oslh.time_s.step1")
    trow("radius/tibia: contour rendering (both compartments)", "B.oslh.time_s.render_B")
    trow("radius/tibia: Laplace-Hamming filter", "B.oslh.time_s.lh")
    trow("radius/tibia: SEG assembly", "B.oslh.time_s.seg")
    trow("radius/tibia: dt, five B maps", "B.oslh.time_s.dt_B")
    for mk in ("TbTh_trabseg", "TbSp", "TbN", "CtTh"):
        trow(f"radius/tibia: dt {MAP_LABEL[mk].split(' (')[0]} (B)", f"B.oslh.map.{mk}.time_s")
    trow("radius/tibia: dt, five A maps", "A.oslh.time_s.dt_all_maps")
    trow("radius/tibia: validation record total (A + B + comparisons)", "B.oslh.time_s.total")
    trow("radius/tibia: validation record total without the pore cascade", "B.oslh.time_s.total_excl_porosity")
    if "B.oslh.time_s.porosity.median" in F:
        trow(f"radius/tibia: pore cascade, both configurations ({fv('B.oslh.time_s.porosity.n')} records)", "B.oslh.time_s.porosity")
    trow("patella: AIM read", "B.patella.time_s.read")
    trow("patella: STEP 1", "B.patella.time_s.step1")
    trow("patella: contour rendering", "B.patella.time_s.render")
    trow("patella: Laplace-Hamming + SEG", "B.patella.time_s.seg")
    trow("patella: read + STEP 1 + rendering + SEG", "B.patella.time_s.total")
    trow("patella: dt, five maps", "B.patella.time_s.dt_run.dt")
    for mk in ("TbTh_trabseg", "TbSp", "TbN", "CtTh"):
        trow(f"patella: dt {MAP_LABEL[mk].split(' (')[0]} (A)", f"A.patella.map.{mk}.time_s")
    A(md_table(["stage", "median s", "range s", "key"], rows))
    A("")
    A("Radius/tibia medians per site group (s; the pooled medians above move with the site mix, because a diaphyseal greyscale is smaller than an ultradistal one):")
    A("")
    rows = []
    for g in SITE_GROUPS:
        sk = skey(g)
        rows.append([g, f"{F[f'B.site.{sk}.grey_voxels.median']['value']:,.0f}"] + [fv(f"B.site.{sk}.time_s.{st}.median", 1) for st in ("read", "step1", "render_B", "lh", "seg", "dt_B")]
                    + [fv(f"B.site.{sk}.map.TbN.time_s.median", 1), f"B.site.{sk}.time_s.*"])
    A(md_table(["site group", "grey voxels (median)", "read", "STEP 1", "rendering", "LH filter", "SEG", "dt, five B maps", "1/Tb.N map", "key"], rows))
    A("")
    A(f"- Peak resident memory of the radius/tibia validation worker: {fv('B.oslh.peak_rss_gb.max', 2)} GB; per validation run "
      + ", ".join(f"{k} {v:.2f} GB" for k, v in F["B.oslh.peak_rss_gb.max_by_run"]["value"].items())
      + " (`B.oslh.peak_rss_gb.*`; validation process, not the pipeline alone; each record carries its worker's running peak).")
    A(f"- GPU == CPU: {F['software.gpu_cpu_identity.evidence']['value']} (`software.gpu_cpu_identity.evidence`). Test suite: {fv('software.tests.collected_total')} tests collected, measured {F['software.tests.collected_total']['source'].rsplit('(', 1)[-1].rstrip(')')} (`software.tests.collected_total`).")
    A("")
    # ---------------- parameters
    A("## 5. IPL parameter values the package uses")
    A("")
    rows = []
    for k, v in F.items():
        if k.startswith("params."):
            rows.append([k, str(v["value"]), v["source"]])
    A(md_table(["key", "value", "source"], rows))
    A("")
    A("## 6. Software facts")
    A("")
    rows = [[k, str(v["value"]), v["source"]] for k, v in F.items() if k.startswith("software.") and not k.startswith(("software.environment.", "software.packaging."))]
    A(md_table(["key", "value", "source"], rows))
    A("")
    A("### 6b. Execution environment and packaging (measured 2026-09-18 in the validation environment, not from the records)")
    A("")
    A("Measured in the `ormir` conda env on the validation workstation (the only environment the pipeline runs in); the records store `dt_backend = gpu` but no library versions.")
    A("")
    how = {"python": "sys.version", "numpy": "numpy.__version__", "scipy": "scipy.__version__", "numba": "numba.__version__", "cupy": "cupy.__version__",
           "cuda_runtime": "cupy.cuda.runtime.runtimeGetVersion() = 12090", "itk": "itk.__version__", "simpleitk": "SimpleITK.__version__", "gpu": "cupy device properties",
           "gpu_memory_gb": "totalGlobalMem 25.76e9 bytes", "cpu": "Win32_Processor", "cpu_cores": "Win32_Processor", "ram_gb": "Win32_ComputerSystem TotalPhysicalMemory 95.3 GiB",
           "os": "Win32_OperatingSystem"}
    rows = []
    for k, v in F.items():
        if k.startswith("software.environment."):
            val = str(v["value"]) + (" (" + v["note"] + ")" if k.endswith((".cupy", ".cpu_cores")) else "")
            rows.append([k, val, how[k.rsplit(".", 1)[-1]]])
        elif k.startswith("software.packaging."):
            rows.append([k, str(v["value"]), v["source"]])
    A(md_table(["key", "value", "source"], rows))
    A("")
    A("## 9. Headline numbers for the abstract (with keys)")
    A("")
    for line in headline_lines():
        A(f"- {line}")
    A("")
    open(os.path.join(OUT_DIR, "FACTS.md"), "w", encoding="utf-8").write("\n".join(L))


def headline_lines():
    N = N_TOTAL
    H = []
    H.append(f"n = {fv('cohort.n_total')} XtremeCT II scans: {fv('cohort.patella.n')} patellae, {fv('cohort.site.UD_radius.n')} ultradistal radii, {fv('cohort.site.UD_tibia.n')} ultradistal tibiae, {fv('cohort.site.D_radius.n')} diaphyseal radii, {fv('cohort.site.D_tibia.n')} diaphyseal tibiae; IPL {fv('cohort.ipl_version')}, 60.7 um  [cohort.*]")
    H.append(f"Participants: {fv('cohort.patella.participants')} (patellae) + " + " + ".join(f"{fv(f'cohort.oslh.study.{st}.participants')} ({st})" for st in STUDIES) + f" = {fv('cohort.participants.total')}  [cohort.patella.participants, cohort.oslh.study.*.participants]")
    H.append(f"Configuration A (IPL's SEG in): {fv('A.pooled.mismatches.four_maps')} differing voxel(s) in {fv('A.pooled.voxels_compared.four_maps')} compared over the four maps of {N} scans; Tb.Sp, 1/Tb.N and Ct.Th maps identical on {fv('A.pooled.map.TbSp.exact_scans')}/{fv('A.pooled.map.TbN.exact_scans')}/{fv('A.pooled.map.CtTh.exact_scans')} of {N}, Tb.Th (TRAB_SEG) on {fv('A.pooled.map.TbTh_trabseg.exact_scans')}/{N}  [A.pooled.mismatches.four_maps, A.pooled.voxels_compared.four_maps, A.pooled.map.*.exact_scans]")
    H.append(f"Configuration A, BV/TV, n = {fv('metrics.A.pooled.BVTV.n')}: identical values on {fv('metrics.A.pooled.BVTV.exact')}/{fv('metrics.A.pooled.BVTV.n')} (the ratio on IPL's SEG and rendered contour; patella sheet values equal the rounded ratio on {fv('A.patella.BVTV.printed.agree_after_rounding')}/21)  [metrics.A.pooled.BVTV.*, A.patella.BVTV.printed.*]")
    H.append(f"Configuration A, patella: 0 differing voxels on all 5 maps x 21 subjects, {fv('A.patella.voxels_compared.total_five_maps')} voxels compared  [A.patella.maps.all_exact, A.patella.voxels_compared.total_five_maps]")
    H.append(f"Configuration B: compartment masks / contour renderings identical on {F['B.patella.step1.cort.exact_scans']['value'] + F['B.oslh.rendering.cort.exact_scans']['value']}/{N} (cortical) and {F['B.patella.step1.trab.exact_scans']['value'] + F['B.oslh.rendering.trab.exact_scans']['value']}/{N} (trabecular) scans (patella STEP-1 masks {fv('B.patella.step1.cort.exact_scans')}/21 and {fv('B.patella.step1.trab.exact_scans')}/21; radius/tibia renderings {fv('B.oslh.rendering.cort.exact_scans')}/{N_OSLH} and {fv('B.oslh.rendering.trab.exact_scans')}/{N_OSLH})  [B.patella.step1.*, B.oslh.rendering.*]")
    H.append(f"Configuration B, Laplace-Hamming SEG pooled: {fv('B.pooled.SEG.mismatches')} differing voxels in {fv('B.pooled.SEG.voxels_compared')} ({fv('B.pooled.SEG.mismatch_ppm', 2)} per million), exact on {fv('B.pooled.SEG.exact_scans')}/{N}, min Dice {fv('B.pooled.SEG.dice.min')}, median Dice {fv('B.pooled.SEG.dice.median')}, Dice >= 0.9999 on {fv('B.pooled.SEG.dice.ge_0_9999')}/{N}; patella {fv('B.patella.SEG.mismatches')} voxels ({fv('B.patella.SEG.ours_only')} ours / {fv('B.patella.SEG.ipl_only')} IPL), radius/tibia {fv('B.oslh.SEG.mismatches')} of which {fv('B.oslh.SEG.top3_mismatches')} in three scans  [B.pooled.SEG.*, B.patella.SEG.*, B.oslh.SEG.*]")
    H.append(f"Configuration B, Ct.Th map identical on {fv('B.pooled.map.CtTh.exact_scans')}/{N} ({fv('B.pooled.map.CtTh.voxels_compared')} voxels); Tb.Th (TRAB_SEG) / Tb.Sp / 1/Tb.N maps differ by {fv('B.pooled.map.TbTh_trabseg.mismatches')} / {fv('B.pooled.map.TbSp.mismatches')} / {fv('B.pooled.map.TbN.mismatches')} voxels over {N} scans  [B.pooled.map.*]")
    for m, lab in (("BVTV", "BV/TV"), ("TbTh_trabseg", "Tb.Th (TRAB_SEG)"), ("TbSp", "Tb.Sp"), ("TbN", "Tb.N")):
        p = f"metrics.B.pooled.{m}"
        H.append(f"Configuration B, {lab}, n = {fv(p + '.n')}: slope {fv(p + '.slope', 6)}, R^2 {fv(p + '.r2')}, ICC(2,1) {fv(p + '.icc21')}, bias {fv(p + '.bias')} {METRICS[m]['unit']}, 95 % LoA [{fv(p + '.loa_low')}, {fv(p + '.loa_high')}], max |rel| {fv(p + '.max_rel_pct', 4)} %, exact {fv(p + '.exact')}/{fv(p + '.n')}  [{p}.*]")
    p = "metrics.B.pooled.CtTh"
    H.append(f"Configuration B, Ct.Th, n = {fv(p + '.n')}: identical values on {fv(p + '.exact')}/{fv(p + '.n')} (bias {fv(p + '.bias')}, max |rel| {fv(p + '.max_rel_pct', 4)} %)  [{p}.*]")
    return H


# ================================================================================================ cross-check
SRC_POOLED_XCHECK = "validation/results/pooled_137_comparison.json"


def crosscheck_pooled():
    """The pooled numbers against manuscript/figures/pooled_statistics.py's own (validation/results/pooled_137_comparison.json,
    block 'n137'): integers exactly, floats to 1e-9 relative, the OLS intercept to 1e-12 absolute.  A
    disagreement stops the build: it is a bug to find, not a choice to make."""
    p = rel(SRC_POOLED_XCHECK)
    if not os.path.exists(p):
        return OrderedDict(file=SRC_POOLED_XCHECK, available=False)
    blk = json.load(open(p, encoding="utf-8"))["n" + ARGS.vset]
    v = lambda k: F[k]["value"]  # noqa: E731
    pairs = [("n", blk["n"], v("cohort.n_total")), ("cohorts.patella", blk["cohorts"]["patella"], v("cohort.patella.n")),
             ("cohorts.UD radius", blk["cohorts"]["UD radius"], v("cohort.site.UD_radius.n")), ("cohorts.UD tibia", blk["cohorts"]["UD tibia"], v("cohort.site.UD_tibia.n")),
             ("cohorts.diaphyseal", blk["cohorts"]["diaphyseal"], v("cohort.site.diaphyseal.n")),
             ("A_mismatch", blk["A_mismatch"], v("A.pooled.mismatches.four_maps")), ("A_compared", blk["A_compared"], v("A.pooled.voxels_compared.four_maps")),
             ("A_exact_comparisons", blk["A_exact_comparisons"], v("A.pooled.exact_comparisons.four_maps")),
             ("A_comparisons", blk["A_comparisons"], sum(v(f"A.pooled.map.{m}.n_scans") for m in ("TbTh_trabseg", "TbSp", "TbN", "CtTh")))]
    for k, m in (("TbTh", "TbTh_trabseg"), ("TbSp", "TbSp"), ("TbN", "TbN"), ("CtTh", "CtTh")):
        pairs += [("A_exact_" + k, blk["A_exact_" + k], v(f"A.pooled.map.{m}.exact_scans")), ("A_n_" + k, blk["A_n_" + k], v(f"A.pooled.map.{m}.n_scans"))]
    for a, b in (("B_seg_n", "n_scans"), ("B_seg_mismatch", "mismatches"), ("B_seg_compared", "voxels_compared"), ("B_seg_ppm", "mismatch_ppm"),
                 ("B_seg_exact", "exact_scans"), ("B_seg_median", "per_scan.median"), ("B_dice_min", "dice.min"), ("B_dice_median", "dice.median"),
                 ("B_dice_ge_9999", "dice.ge_0_9999")):
        pairs.append((a, blk[a], v("B.pooled.SEG." + b)))
    for k, m in (("BVTV", "BVTV"), ("TbTh", "TbTh_trabseg"), ("TbSp", "TbSp"), ("TbN", "TbN"), ("CtTh", "CtTh")):
        st = blk["metrics"][k]
        for a, b in (("n", "n"), ("mean_ipl", "mean_ipl"), ("sd_ipl", "sd_ipl"), ("mean_ours", "mean_ours"), ("sd_ours", "sd_ours"), ("bias", "bias"),
                     ("sd_diff", "sd_diff"), ("loa_low", "loa_low"), ("loa_high", "loa_high"), ("max_abs_diff", "max_abs_diff"), ("max_rel_pct", "max_rel_pct"),
                     ("identical", "exact"), ("slope", "slope"), ("intercept", "intercept"), ("r2", "r2"), ("icc21", "icc21")):
            pairs.append((f"metrics.{k}.{a}", st[a], v(f"metrics.B.pooled.{m}.{b}")))
    bad, worst = [], 0.0
    for name, theirs, mine in pairs:
        if isinstance(theirs, int) and not isinstance(theirs, bool) and isinstance(mine, int):
            ok = theirs == mine
        elif name.endswith(".intercept"):
            ok = abs(float(theirs) - float(mine)) <= 1e-12
        else:
            a, b = float(theirs), float(mine)
            dev = 0.0 if a == b else abs(a - b) / max(abs(a), abs(b))
            ok = dev <= 1e-9
            worst = max(worst, dev) if ok else worst
        if not ok:
            bad.append((name, theirs, mine))
    if bad:
        raise SystemExit(f"facts disagree with {SRC_POOLED_XCHECK} (block n{ARGS.vset}): {bad}")
    return OrderedDict(file=SRC_POOLED_XCHECK, block="n" + ARGS.vset, compared=len(pairs), disagreeing=0, worst_relative_deviation=worst)


# ================================================================================================ main
META_BUILT = (time.strftime("%Y-%m-%d", time.gmtime(int(os.environ["SOURCE_DATE_EPOCH"])))  # reproducible builds
              if os.environ.get("SOURCE_DATE_EPOCH") else time.strftime("%Y-%m-%d"))
META_VERSION_SHORT = "n = 137, public build"


def main():
    cohort_section()
    config_A_section()
    config_B_section()
    metrics_section()
    parameters_section()
    software_section()
    environment_section()
    xcheck = crosscheck_pooled()
    gap_list = []
    meta = OrderedDict(built=META_BUILT,
                       version=f"n = 137 (public build): radius/tibia records from {OSLH_AUTO} (62 counted) + {OSLH_NOEDIT} (54, all diaphyseal); 21 patellae",
                       set=ARGS.vset, builder="manuscript/facts/build_facts.py", repo=".",
                       sources=OrderedDict(patella_A_records=SRC_PAT_A_RECORDS, patella_A_means=SRC_PAT_A_MEANS, patella_A_bvtv=SRC_PAT_A_BVTV, patella_B=SRC_PAT_B_DT,
                                           patella_B_seg_stage=SRC_PAT_B_SEG, patella_B_dt=SRC_PAT_B_DT, patella_B_seg_crosscheck=SRC_PAT_B_SEG_XCHECK,
                                           patella_B_summary=SRC_PAT_B_SUMMARY, study_table=SRC_STUDY_TABLE,
                                           oslh=list(SRC_OSLH_DIRS.values()), oslh_dirs=OrderedDict(SRC_OSLH_DIRS), oslh_glob=OSLH_GLOB.pattern,
                                           oslh_dropped=sorted(OSLH_DROP), launch_scripts=None),
                       configurations=OrderedDict(A="IPL's SEG and contour renderings in; ipldt dt engine (modules 6-7)",
                                                  B="IPL's periosteal contour in; ipldt STEP 1, contour rendering, Laplace-Hamming SEG, dt (modules 3-7)"),
                       tbth_definitions=OrderedDict(TbTh_trabseg="dt_thickness on TRAB_SEG cropped to the trabecular contour (Script 32); the paper's Tb.Th",
                                                    TbTh_wholeseg="dt_thickness on the whole SEG cropped to the trabecular contour (internal)",
                                                    TbTh_as_delivered="the definition of IPL's delivered TRAB_TH file per cohort (patella whole SEG, radius/tibia TRAB_SEG)"),
                       site_groups=OrderedDict((skey(g), m) for g, m in SITE_GROUPS.items()),
                       crosscheck=xcheck,
                       gaps=gap_list,
                       amended=[])
    doc = OrderedDict(meta=meta, facts=F, tables=TABLES)
    json.dump(doc, open(os.path.join(OUT_DIR, "facts.json"), "w", encoding="utf-8"), indent=1)
    write_facts_md(gap_list)
    print(f"facts ({'n = ' + str(N_TOTAL)}): {len(F)} keys -> {OUT_DIR}; tables: {list(TABLES)}")
    print(f"cross-check against {xcheck['file']}: " + (f"{xcheck['compared']} numbers of block {xcheck['block']}, 0 disagree, worst relative deviation "
                                                       f"{xcheck['worst_relative_deviation']:.1e}" if xcheck.get("compared") else "file not on disk"))
    for line in headline_lines():
        print("*", line)


if __name__ == "__main__":
    main()
