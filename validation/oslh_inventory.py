"""oslh_inventory -- inventory and consistency audit of the radius / tibia cross-validation export
(<IPLDT_LAB_ROOT>/radius_tibia/delivery: 123 HR-pQCT measurement folders, distal and diaphyseal
tibia / radius, CKD / REPRO / BMAT studies, six re-evaluated duplicates under rerun/).

PRIVACY: the inventory reads the scans' header identity fields (patient name, patient index, dates) to derive
the study, the participant code and the site, and its output (validation/results/oslh/) therefore carries
identifiers: it is for internal use only, is git-ignored, and is never published.  The published records carry
only the fields tools/deidentify_results.py keeps.

For every measurement folder the script records, WITHOUT recomputing anything with ipldt (this is the map of
the terrain the validation scripts will walk, not the validation):

  identity     group (Distal/CKD ...), measurement id (folder name), base (L/D number of the greyscale), study,
               subject id (raw and normalised), region UD / D, bone (from the patient name AND the site index,
               20 / 21 radius, 38 / 39 tibia), scan date (ISQ creation), ISQ_TO_AIM time, greyscale grid / pos / el.
  calibration  Density slope / intercept / Mu_Scaling from the greyscale processing log and the resulting
               /seg_gauss native thresholds for 500 / 3000 mg HA (ipldt.ipl_ops.mgha_to_native), plus the
               Laplace-Hamming /threshold native numbers the SEG processing log carries (475 permille -> 15564).
  files        every file with its kind, size, grid (dim @ pos), header element sizes (gobj-derived renderings
               carry garbage el fields = Orig-GOBJ-Dim-um / Orig-GOBJ-Dim-p), data type, set-voxel count, and
               the header fields of its processing log (Created by, every 'Time' line, Original file = the gobj a
               rendering came from, Original Creation-Date = when that gobj was written, Patient Name).
  SEG          the processing-log procedure list with its numeric parameters (Laplace-Hamming settings, threshold
               permille / native, cl_nr min numbers, gobj masks and peel), whether STEP 1 (seg_gauss / morphology)
               procedures appear in it, the 126 / 127 label counts, and the SEG.AIM (compressed) vs
               SEG_DECOMPRESSED identity where both exist.
  maps         per map (TRAB_TH, TRAB_SP, TRAB_1N, CORT_TH): 'with GOBJ_file' / 'and peel_iter' and the other dt
               parameters, support, value range, mean over the non-zero voxels (voxels and mm at the header's
               x element size -- IPL's printed Th / BG Th / MAT N).
  checks       voxel consistency on the union grid by GLOBAL position (ipldt.io.align_to through
               ipldt.ipl_ops.on_grid):
               (a) periosteal rendering _CT  vs  CORT_MASK_CT | TRAB_MASK(_CORR)_CT  (and the CORT & TRAB overlap)
               (b) SEG > 0 vs _CT; SEG == 127 vs CORT_MASK_CT; SEG == 126 vs the trabecular rendering provided
                   (Script 32's assembly puts 127 inside the cortical and 126 inside the trabecular contour)
               (c) each map's support vs the rendering of the gobj its log names (TRAB_* vs the trabecular
                   rendering, CORT_TH vs CORT_MASK_CT) and vs the SEG labels (TRAB_TH inside SEG == 126, TRAB_SP /
                   TRAB_1N in the marrow, CORT_TH vs SEG == 127)
               (d) grid relations (SEG grid vs the tight box of its support and vs the _CT grid, map grids vs the
                   SEG grid / the tight boxes of the labels / the CORT grid)
               (e) (--render-check pass) ipldt.contour.render_volume(_CT minus the trabecular rendering) vs
                   CORT_MASK_CT: 0 mismatches means CORT_MASK.GOBJ was /togobj_from_aim of periosteal - trab_mask.gobj
                   (the re-evaluation workflow of the double-underscore logs) and the periosteal voxels in neither
                   rendering are exactly the contour-conversion loss; plus render_volume(CORT_MASK_CT) vs itself.
               On the 52 measurements whose trabecular rendering is TRAB_MASK_CORR_CT (a manually corrected
               contour) (b) + (c) + (e) decide whether IPL's SEG and maps were computed with that corrected contour
               or with the uncorrected trab_mask.gobj that every map log names.
  logs         the six IPL evaluation logs (EVAL_*.LOG): session date / user / node, IPL module version and build, the
               uct_evaluation trailer (P1..P5), the resolved symbols (IPL_MISC1_0 / IPL_MISC1_1 = corner cl_nr
               min_number / second close distance, IPL_PEEL0), every /cl_nr_extract -min_number, /close
               -close_distance, /seg_gauss + /threshold native numbers, every /dt_* call with its gobj and
               peel_iter and the printed statistics (Th, BG Th, MAT N ...), and whether the mask stage ran STEP 1
               (/seg_gauss chain) or only re-derived the masks from existing gobjs (cort = periosteal - trab_mask,
               the double-underscore variant); the printed dt values are matched against the on-disk map means.

Outputs (validation/results/oslh/): inventory.json (one record per measurement + logs + cohort summary),
inventory.md (counts per group / bone / region, the CORR measurements and what their maps used, anomalies, the
per-measurement validation plan: exact expected / manual-correction expected / not comparable), and
records/<group>_<measurement>.json for --resume.  Console + per-measurement lines go to results/oslh/logs.

CLI
    python validation/oslh_inventory.py                       # all 123 measurements (about 10 min)
    python validation/oslh_inventory.py --measurements 345857 581203
    python validation/oslh_inventory.py --no-voxel-checks     # proclogs and headers only (seconds)
    python validation/oslh_inventory.py --resume              # keep records already written, rebuild the report
    python validation/oslh_inventory.py --render-check        # add check (e) to the records written (about 10 min), rebuild
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import re
import struct
import sys
import time
import traceback
from collections import Counter, OrderedDict

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datapaths import lab_path  # noqa: E402  (non-public data roots: validation/datapaths.py)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import ipldt  # noqa: E402
from ipldt import ipl_ops as ops  # noqa: E402

ROOT = os.environ.get("OSLH_ROOT") or lab_path("radius_tibia/delivery")
OUT_DIR = os.path.join(HERE, "results", "oslh")
LOG_DIR = os.path.join(OUT_DIR, "logs")
TOP_GROUPS = ("Distal", "Diaphyseal")
RERUN = "rerun"
MAPS = ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH")
MAP_GOBJ = {"TRAB_TH": "trab", "TRAB_SP": "trab", "TRAB_1N": "trab", "CORT_TH": "cort"}
SITE_BONE = {20: "radius", 21: "radius", 38: "tibia", 39: "tibia"}
STEP1_PROCS = ("SegGauss", "Gauss", "Erosion", "Dilation", "Close", "Open", "OwRank", "Slicewise", "Metric")
SEG_GAUSS_MGHA = (500.0, 3000.0)
_T0 = time.time()
_LOGFH = None


def log(*a):
    line = f"[{time.time() - _T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(line, flush=True)
    if _LOGFH is not None:
        _LOGFH.write(line + "\n")
        _LOGFH.flush()


# ============================================================================================ file helpers
def read_proclog(path):
    """The processing log of an AIM v020 without reading the body."""
    with open(path, "rb") as fh:
        raw = fh.read(20)
        pre = struct.unpack("<5i", raw)
        hsize, lsize = pre[1], pre[2]
        fh.seek(20 + hsize)
        return fh.read(lsize).decode("latin-1", "replace")


def header_fields(proclog):
    """First occurrence of the named header fields + every 'Time' line (in order)."""
    out = {}
    for key in ("Created by", "Original file", "Original Creation-Date", "Orig-ISQ-Dim-p", "Orig-ISQ-Dim-um",
                "Orig-GOBJ-Dim-p", "Orig-GOBJ-Dim-um", "Patient Name", "Index Patient", "Index Measurement", "Site",
                "Scanner ID", "Mu_Scaling", "Density: slope", "Density: intercept", "Peel Iterations", "Cutborder"):
        m = re.search(r"^" + re.escape(key) + r"\s{2,}(.*?)\s*$", proclog, re.M)
        if m:
            out[key] = m.group(1)
    out["Time"] = [m.group(1) for m in re.finditer(r"^Time\s{2,}(.*?)\s*$", proclog, re.M)]
    return out


_STAT_KEYS = ("Parameter", "Minimum", "Maximum", "Average", "Standard", "Scaled")


def parse_procedures(proclog):
    """The 'Procedure:' blocks of a processing log (between the !---- separators): name(s) and the key / value
    lines that are not the statistics block."""
    procs = []
    for block in proclog.split("!" + "-" * 79):
        names = re.findall(r"^Procedure:\s+(.*?)\s*$", block, re.M)
        if not names:
            continue
        params = OrderedDict()
        for line in block.splitlines():
            if line.startswith("Procedure:") or line.startswith("!") or not line.strip():
                continue
            m = re.match(r"^(\S.*?\S):?\s{2,}(\S.*?)\s*$", line)
            if m and not m.group(1).startswith(_STAT_KEYS):
                params[m.group(1)] = m.group(2)
        procs.append(dict(name=" / ".join(n.replace("D3P_", "").replace("()", "") for n in names), params=dict(params)))
    return procs


def _sq(s):
    return " ".join(str(s).split()) if s is not None else None


def proc_summary(procs):
    """Compact one-line summary of a procedure list: name(key params)."""
    parts = []
    for p in procs:
        n, q = p["name"], {k: _sq(v) for k, v in p["params"].items()}
        extra = ""
        if n.startswith("Cl_ExtractNumber"):
            extra = f"min {q.get('min_nr')}"
        elif n.startswith("GobjOrAimMaskAimPeel"):
            extra = f"{os.path.basename(q.get('Gobj File', '?')).split(']')[-1]} peel {q.get('Peel Iterations')}"
        elif n.startswith("SupThreshold"):
            extra = f"{q.get('low_th_input')}..{q.get('upp_th_input')} unit {q.get('Wished Unit of thresholds')} -> native {q.get('low corresponds to data value')}..{q.get('upp corresponds to data value')}"
        elif n.startswith("BoundingBoxCut"):
            extra = f"border {q.get('Null Border')}"
        elif n.startswith("SetAllNonZero"):
            extra = f"-> {q.get('value_non_zero')}"
        elif n.startswith("FFT_LaplaceHamming"):
            extra = f"cutoff {q.get('fft.lp_cut_off_freq')} amp {q.get('fft.laplace_hamming.amplitude')} eps {q.get('fft.laplace.eps')}"
        elif n.startswith("Std_FloatNormMax"):
            extra = f"max {q.get('max')}"
        elif "DT_Object" in n or "Dt_Obj" in n:
            extra = f"{os.path.basename(q.get('with GOBJ_file', '?')).split(']')[-1]} peel {q.get('and peel_iter')}"
        parts.append(f"{n}({extra})" if extra else n)
    return " > ".join(parts)


def grid_of(v):
    return [int(x) for x in v["dim"]], [int(x) for x in v["pos"]]


def grid_str(g):
    (dx, dy, dz), (px, py, pz) = g
    return f"{dx}x{dy}x{dz} @ {px},{py},{pz}"


def tight_box(mask_bool, pos):
    """(dim, pos) of the tight bounding box of the set voxels of a (z, y, x) bool array on a grid at `pos`."""
    if not mask_bool.any():
        return None
    zz, yy, xx = np.nonzero(mask_bool)
    lo = (int(xx.min()), int(yy.min()), int(zz.min()))
    hi = (int(xx.max()) + 1, int(yy.max()) + 1, int(zz.max()) + 1)
    return [hi[i] - lo[i] for i in range(3)], [lo[i] + int(pos[i]) for i in range(3)]


def grid_inside(inner, outer):
    (di, pi), (do, po) = inner, outer
    return all(pi[k] >= po[k] and pi[k] + di[k] <= po[k] + do[k] for k in range(3))


def bone_from_name(name):
    u = (name or "").upper()
    if "TIBIA" in u:
        return "tibia"
    if "RADIUS" in u:
        return "radius"
    return None


def parse_patient_name(name):
    """'<PI>; <STUDY>Study <CODE>_<nnn> <UD|D> <Tibia|Radius>' -> study, subject (raw + normalised), region, bone."""
    out = dict(raw=name, pi=None, study=None, subject_raw=None, subject=None, region=None, bone=None, notes=[])
    if not name:
        return out
    parts = name.split(";", 1)
    out["pi"] = parts[0].strip()
    tokens = (parts[1] if len(parts) > 1 else "").split()
    if len(tokens) >= 4:
        study_tok, subj, region, bone = tokens[0], tokens[1], tokens[2], tokens[3]
    else:
        out["notes"].append(f"unexpected patient-name layout: {name!r}")
        return out
    study = re.sub(r"study$", "", study_tok, flags=re.I).upper()
    out["study"] = study
    out["subject_raw"] = subj
    subj_norm = subj.upper()
    m = re.match(r"^(.+?)[_-](\d+)$", subj_norm) or re.match(r"^([A-Z]+)(\d+)$", subj_norm)
    if m:
        prefix, num = m.group(1), m.group(2)
        fixed = {"CDK": "CKD", "REPR0": "REPRO"}.get(prefix, prefix)
        if fixed != prefix:
            out["notes"].append(f"subject prefix typo {prefix} -> {fixed}")
        if fixed != study and not (study == "BMAT" and fixed == "MAT"):
            out["notes"].append(f"subject prefix {fixed} differs from study {study}")
        subj_norm = f"{fixed}_{int(num):03d}"
    out["subject"] = subj_norm
    out["region"] = region.upper()
    if out["region"] not in ("UD", "D"):
        out["notes"].append(f"unexpected region token {region!r}")
    out["bone"] = bone_from_name(bone)
    if out["bone"] is None:
        out["notes"].append(f"unexpected bone token {bone!r}")
    return out


def classify_file(name, base):
    """Kind of a file in a measurement folder."""
    n = name.upper()
    b = base.upper()
    if n == f"{b}.AIM":
        return "grey"
    if n == f"{b}_CT.AIM":
        return "CT"
    if n == f"{b}_CORT_MASK_CT.AIM":
        return "CORT_MASK_CT"
    if n == f"{b}_TRAB_MASK_CT.AIM":
        return "TRAB_MASK_CT"
    if n == f"{b}_TRAB_MASK_CORR_CT.AIM":
        return "TRAB_MASK_CORR_CT"
    if n == f"{b}_SEG_DECOMPRESSED.AIM":
        return "SEG_DECOMPRESSED"
    if n == f"{b}_SEG.AIM":
        return "SEG"
    for mp in MAPS:
        if n == f"{b}_{mp}_DECOMPRESSED.AIM":
            return f"{mp}_DECOMPRESSED"
        if n == f"{b}_{mp}_COMPRESSED.AIM":
            return f"{mp}_COMPRESSED"
        if n.startswith(f"{b}_{mp}_") and n.endswith(".AIM"):
            return f"{mp}_MISSPELLED"
    if n.endswith(".LOG"):
        return "LOG"
    if n.startswith("CROSS_VALIDATION") and n.endswith(".CSV"):
        return "old_pipeline_csv"
    if n.endswith(".ZIP") or n.endswith(".PNG"):
        return "preview"
    return "other"


# ============================================================================================ IPL log parsing
_RE_SESSION = re.compile(r"^\s*(\d{1,2}-[A-Z]{3}-\d{4} \d\d:\d\d:\d\d\.\d\d)\s+User:\s+(\S+)\s+Process ID:\s+(\S+)", re.M)
_RE_NODE = re.compile(r"^\s+Node:\s+(\S+)\s+Process name:\s+\"([^\"]*)\"", re.M)
_RE_SYMBOL = re.compile(r"translated symbol \"(\w+) to '([^']*)'")
_RE_METRIC = re.compile(r"^!> ([A-Za-z0-9./ ()]+?)\s+=\s+([-+0-9.]+)\s+\[([^\]]*)\]", re.M)


def parse_ipl_log(path):
    """One IPL evaluation log -> dict (session, module, sections with commands / parameters / printed
    statistics, resolved symbols, trailer)."""
    text = open(path, "rb").read().decode("latin-1", "replace")
    lines = text.splitlines()
    out = dict(path=path.replace("\\", "/"), name=os.path.basename(path), lines=len(lines), variant="double_underscore"
               if "XT2__" in os.path.basename(path) else "single_underscore")
    m = _RE_SESSION.search(text)
    if m:
        out["session_time"], out["user"], out["process_id"] = m.group(1), m.group(2), m.group(3)
    m = _RE_NODE.search(text)
    if m:
        out["node"], out["process_name"] = m.group(1), m.group(2)
    out["module_versions"] = sorted(set(re.findall(r"Module:\s+(Scanco Module [^\n]*?V[\d.]+)", text)))
    out["built"] = sorted(set(re.findall(r"Built:\s+([^\n]+?)\s*$", text, re.M)))
    out["trailer"] = {k: v for k, v in re.findall(r"^\s+(P\d) = \"([^\"]*)\"", text, re.M)}
    out["trailer_lines"] = [l.strip() for l in lines if l.startswith("UCT_EVAL") or l.startswith("Job ") or "job terminated" in l]
    out["patient_names"] = sorted(set(re.findall(r"^!> Patient Name:\s+(.*?)\s*$", text, re.M)))
    # sections
    idx = [i for i, l in enumerate(lines) if "IPL Processing List Scan" in l]
    idx.append(len(lines))
    sections = []
    for s, e in zip(idx[:-1], idx[1:]):
        body = "\n".join(lines[s:e])
        sec = dict(start_line=s + 1, end_line=e, symbols=OrderedDict(), commands=[])
        for k, v in _RE_SYMBOL.findall(body):
            sec["symbols"].setdefault(k.upper(), []).append(v)
        sec["symbols"] = {k: sorted(set(v)) for k, v in sec["symbols"].items()}
        i = s
        cur = None
        while i < e:
            l = lines[i]
            if l.startswith("/") and not l.startswith("//"):
                cur = dict(cmd=l.strip().split()[0], params=OrderedDict(), line=i + 1, out=[])
                sec["commands"].append(cur)
            elif cur is not None and re.match(r"^\s+-\w", l):
                mm = re.match(r"^\s+-(\w+)\s+(.*?)\s*$", l)
                if mm:
                    cur["params"][mm.group(1)] = mm.group(2)
            elif cur is not None and (l.startswith("!>") or l.startswith("!%")):
                if any(k in l for k in ("native numbers", "Valid DT", "Found Min", "inmask_nr", "Rel Vol", "-> Set", "completed:", "Total Processing", "aimpos", "dim:", "beg_pos", "end_pos")) or _RE_METRIC.match(l):
                    cur["out"].append(l.strip())
            i += 1
        for c in sec["commands"]:
            c["params"] = dict(c["params"])
            txt = "\n".join(c["out"])
            mm = re.search(r"native numbers:\s+(-?\d+)\s+(-?\d+)", txt)
            if mm:
                c["native_thresholds"] = [int(mm.group(1)), int(mm.group(2))]
            if c["cmd"].startswith("/dt_"):
                c["printed"] = OrderedDict((k.strip(), [float(v), u]) for k, v, u in _RE_METRIC.findall(txt))
                mm = re.search(r"Valid DT Obj values calculated for ([\d.]+)%", txt)
                if mm:
                    c["valid_pct"] = float(mm.group(1))
                mm = re.search(r"inmask_nr\s+(\d+)\s+of total\s+=\s+(\d+)", txt)
                if mm:
                    c["inmask_nr"], c["total"] = int(mm.group(1)), int(mm.group(2))
            mm = re.search(r"Found Min (\d+) and Max (\d+)", txt)
            if mm:
                c["written_min_max"] = [int(mm.group(1)), int(mm.group(2))]
        cmds = [c["cmd"] for c in sec["commands"]]
        if "/seg_gauss" in cmds:
            sec["kind"] = "mask_step1"
        elif "/togobj_from_aim" in cmds and "/subtract_aims" in cmds:
            sec["kind"] = "mask_from_gobjs"
        elif "/fft_laplace_hamming" in cmds or "/threshold" in cmds:
            sec["kind"] = "seg_laplace_hamming"
        elif any(c.startswith("/dt_") for c in cmds):
            sec["kind"] = "dt_" + "_".join(c[4:] for c in cmds if c.startswith("/dt_"))
        elif "/cl_slicewise_extractow" in cmds:
            sec["kind"] = "cortical_pores"
        else:
            sec["kind"] = "other"
        mm = re.search(r"Total Processing Time:\s+CPU:\s+(\S+)\s+ELAP:\s+(\S+)", body)
        if mm:
            sec["cpu"], sec["elapsed"] = mm.group(1), mm.group(2)
        sections.append(sec)
    out["sections"] = sections
    # resolved STEP 1 parameters and the numbers the validation needs
    sym = {}
    for sec in sections:
        for k, v in sec["symbols"].items():
            sym.setdefault(k, set()).update(v)
    out["symbols"] = {k: sorted(v) for k, v in sym.items()}
    out["resolved"] = dict(IPL_MISC1_0=out["symbols"].get("IPL_MISC1_0"), IPL_MISC1_1=out["symbols"].get("IPL_MISC1_1"),
                           IPL_PEEL0=out["symbols"].get("IPL_PEEL0"))
    out["cl_nr_min_numbers"] = [(sec["kind"], int(c["params"].get("min_number", -1))) for sec in sections for c in sec["commands"] if c["cmd"] == "/cl_nr_extract"]
    out["close_distances"] = [(sec["kind"], int(c["params"].get("close_distance", -1))) for sec in sections for c in sec["commands"] if c["cmd"] == "/close"]
    out["open_distances"] = [(sec["kind"], int(c["params"].get("open_distance", -1))) for sec in sections for c in sec["commands"] if c["cmd"] == "/open"]
    out["peel_iters"] = sorted(set(int(c["params"]["peel_iter"]) for sec in sections for c in sec["commands"] if "peel_iter" in c["params"]))
    out["seg_gauss"] = [dict(sigma=c["params"].get("sigma"), support=c["params"].get("support"), lower=c["params"].get("lower_in_perm_aut_al"),
                             upper=c["params"].get("upper_in_perm_aut_al"), native=c.get("native_thresholds"))
                        for sec in sections for c in sec["commands"] if c["cmd"] == "/seg_gauss"]
    out["lh_threshold"] = [dict(lower=c["params"].get("lower_in_perm_aut_al"), upper=c["params"].get("upper_in_perm_aut_al"), unit=c["params"].get("unit"),
                                native=c.get("native_thresholds"))
                           for sec in sections for c in sec["commands"] if c["cmd"] == "/threshold"]
    out["dt_calls"] = []
    for sec in sections:
        inputs = [c["params"].get("filename") for c in sec["commands"] if c["cmd"] == "/read"]
        for c in sec["commands"]:
            if c["cmd"].startswith("/dt_"):
                writes = [c2["params"].get("filename") for c2 in sec["commands"] if c2["cmd"].startswith("/write") and c2["line"] > c["line"]]
                out["dt_calls"].append(dict(cmd=c["cmd"], input_files=inputs, gobj=os.path.basename(c["params"].get("gobj_filename", "")).split("]")[-1],
                                            peel_iter=c["params"].get("peel_iter"), ridge_epsilon=c["params"].get("ridge_epsilon"),
                                            assign_epsilon=c["params"].get("assign_epsilon"), version=c["params"].get("version"),
                                            printed=c.get("printed", {}), valid_pct=c.get("valid_pct"), inmask_nr=c.get("inmask_nr"),
                                            written_to=[os.path.basename(w).split("]")[-1] for w in writes if w][:1]))
    out["mask_stage"] = next((sec["kind"] for sec in sections if sec["kind"].startswith("mask")), None)
    out["section_kinds"] = [sec["kind"] for sec in sections]
    out["written_files"] = sorted(set(os.path.basename(c["params"]["filename"]).split("]")[-1] for sec in sections for c in sec["commands"]
                                      if c["cmd"].startswith("/write") and "filename" in c["params"]))
    return out


# ============================================================================================ one measurement
def scan_folder(folder):
    """The files of one measurement folder, classified; base from the greyscale name."""
    names = sorted(os.listdir(folder))
    greys = [n for n in names if re.fullmatch(r"[A-Za-z]\d{7}\.AIM", n, re.I)]
    if len(greys) != 1:
        raise FileNotFoundError(f"{folder}: expected one greyscale <base>.AIM, found {greys}")
    base = os.path.splitext(greys[0])[0]
    files = OrderedDict()
    for n in names:
        p = os.path.join(folder, n)
        if os.path.isdir(p):
            files[n] = dict(kind="directory", size=None)
            continue
        files[n] = dict(kind=classify_file(n, base), size=os.path.getsize(p))
    return base, files


def aim_info(path, read_data=True):
    """Header + proclog fields (+ data-derived counts when read_data) of an AIM."""
    v = ipldt.read_aim(path) if read_data else None
    pl = v["proclog"] if v is not None else read_proclog(path)
    hf = header_fields(pl)
    info = dict(header=hf, procedures=parse_procedures(pl))
    if v is not None:
        info["grid"] = grid_of(v)
        info["el_header_mm"] = [float(e) for e in v["el_size_mm"]]
        info["type_code"] = f"0x{int(v.get('type_code', 0)):08x}"
        info["dtype"] = str(v["data"].dtype)
        vals, counts = np.unique(v["data"], return_counts=True)
        info["set_voxels"] = int((v["data"] != 0).sum())
        info["values"] = {int(a): int(b) for a, b in zip(vals.tolist(), counts.tolist())} if len(vals) <= 12 else None
        info["value_min"], info["value_max"] = int(vals.min()), int(vals.max())
    return v, info


def el_from_gobj_dims(hf):
    """Orig-GOBJ-Dim-um / Orig-GOBJ-Dim-p -- what the el fields of a gobj-derived AIM header turn out to be."""
    try:
        p = [int(x) for x in hf["Orig-GOBJ-Dim-p"].split()]
        um = [int(x) for x in hf["Orig-GOBJ-Dim-um"].split()]
        return [um[i] / p[i] / 1000.0 for i in range(3)]
    except Exception:
        return None


def measurement_record(group, meas, folder, voxel_checks=True):
    t0 = time.time()
    rec = OrderedDict(group=group, measurement=meas, folder=folder.replace("\\", "/"), errors=[], notes=[])
    base, files = scan_folder(folder)
    rec["base"] = base
    rec["files"] = files
    kinds = {f["kind"]: n for n, f in files.items()}
    rec["present"] = {k: (k in kinds) for k in ("grey", "CT", "CORT_MASK_CT", "TRAB_MASK_CT", "TRAB_MASK_CORR_CT", "SEG_DECOMPRESSED", "SEG",
                                                  "TRAB_TH_DECOMPRESSED", "TRAB_SP_DECOMPRESSED", "TRAB_1N_DECOMPRESSED", "CORT_TH_DECOMPRESSED",
                                                  "TRAB_TH_COMPRESSED", "TRAB_SP_COMPRESSED", "TRAB_1N_COMPRESSED", "CORT_TH_COMPRESSED", "LOG")}
    rec["logs"] = [n for n, f in files.items() if f["kind"] == "LOG"]
    rec["misspelled"] = [n for n, f in files.items() if f["kind"].endswith("_MISSPELLED")]
    rec["other_files"] = [n for n, f in files.items() if f["kind"] in ("other", "old_pipeline_csv", "preview", "directory")]
    rec["trab_is_corr"] = "TRAB_MASK_CORR_CT" in kinds
    vols = {}

    # ---- greyscale: identity, calibration, thresholds
    try:
        v, info = aim_info(os.path.join(folder, kinds["grey"]), read_data=voxel_checks)
        hf = info["header"]
        if not voxel_checks:
            # header only: grid from the pre-header ints (cheap) -- read_aim would decode 90 MB
            raw = open(os.path.join(folder, kinds["grey"]), "rb").read(20 + 140)
            pre = struct.unpack("<5i", raw[:20])
            ints = struct.unpack("<" + "i" * (pre[1] // 4), raw[20:20 + pre[1]])
            info["grid"] = [list(ints[9:12]), list(ints[6:9])]
            info["el_header_mm"] = list(ipldt.io._element_size(kinds["grey"], ints))
        pn = parse_patient_name(hf.get("Patient Name"))
        site = int(hf["Site"]) if "Site" in hf else None
        bone_site = SITE_BONE.get(site)
        bone = pn["bone"] or bone_site
        if pn["bone"] and bone_site and pn["bone"] != bone_site:
            rec["notes"].append(f"bone from patient name ({pn['bone']}) differs from site index {site} ({bone_site}); using the site index")
            bone = bone_site
        rec["identity"] = dict(patient_name=hf.get("Patient Name"), pi=pn["pi"], study=pn["study"], subject_raw=pn["subject_raw"], subject=pn["subject"],
                               region=pn["region"], bone=bone, bone_from_name=pn["bone"], site=site, bone_from_site=bone_site,
                               index_patient=hf.get("Index Patient"), index_measurement=hf.get("Index Measurement"),
                               scan_date=hf.get("Original Creation-Date"), aim_time=hf["Time"][0] if hf["Time"] else None,
                               isq=hf.get("Original file"), scanner_id=hf.get("Scanner ID"), name_notes=pn["notes"])
        if hf.get("Index Measurement") and str(int(hf["Index Measurement"])) != str(int(meas)):
            rec["notes"].append(f"folder id {meas} differs from the proclog's Index Measurement {hf['Index Measurement']}")
        rec["greyscale"] = dict(file=kinds["grey"], grid=info["grid"], el_mm=info["el_header_mm"], dtype=info.get("dtype"), type_code=info.get("type_code"),
                                created_by=hf.get("Created by"), times=hf["Time"], orig_isq_dim_p=hf.get("Orig-ISQ-Dim-p"), orig_isq_dim_um=hf.get("Orig-ISQ-Dim-um"))
        cal = ops.calibration_from_proclog(v["proclog"] if v is not None else read_proclog(os.path.join(folder, kinds["grey"])))
        rec["calibration"] = dict(slope=cal["slope"], intercept=cal["intercept"], mu_scaling=cal["mu_scaling"],
                                  seg_gauss_native=[ops.mgha_to_native(m, cal["slope"], cal["intercept"], cal["mu_scaling"]) for m in SEG_GAUSS_MGHA],
                                  seg_gauss_mgha=list(SEG_GAUSS_MGHA))
        el = rec["greyscale"]["el_mm"]
        if v is not None:
            vols["grey"] = v
    except Exception as e:
        rec["errors"].append(f"greyscale: {e!r}")
        el = None

    # ---- renderings, SEG, maps: headers / proclogs (and data when voxel_checks)
    rec["renderings"] = OrderedDict()
    for kind in ("CT", "CORT_MASK_CT", "TRAB_MASK_CT", "TRAB_MASK_CORR_CT"):
        if kind not in kinds:
            continue
        try:
            v, info = aim_info(os.path.join(folder, kinds[kind]), read_data=voxel_checks)
            hf = info["header"]
            gobj = os.path.basename(hf.get("Original file", "")).split("]")[-1]
            procs = info["procedures"]
            rendered_time = hf["Time"][0] if hf["Time"] else None
            is_export = bool(re.match(r"^DK0:\[", hf.get("Original file", ""))) and bool(re.search(r"-2026 ", rendered_time or "")) and not procs
            if is_export:
                provenance = f"2026 export: D3P_GobjCreateAimPeel rendering of {gobj} (gobj written {hf.get('Original Creation-Date')})"
            elif any(p["name"].startswith("Concatenate") for p in procs):
                provenance = (f"written by the evaluation on {rendered_time}: rendering of {gobj} followed by {[p['name'] for p in procs]} "
                              "-- i.e. the periosteal rendering MINUS the trabecular rendering (subtract_aims), the raw CORT_MASK.AIM of a re-evaluation, not a rendering of CORT_MASK.GOBJ")
            else:
                provenance = f"written by the evaluation on {rendered_time}: rendering of {gobj} (the raw <base>_{kind.replace('_CT', '')}.AIM of the run)" + (f" + {[p['name'] for p in procs]}" if procs else "")
            r = dict(file=kinds[kind], gobj=gobj, gobj_kind=("periosteal" if not re.search(r"_(CORT|TRAB)_MASK", gobj.upper()) else
                                                            ("cort_mask" if "CORT_MASK" in gobj.upper() else ("trab_mask_corr" if "CORR" in gobj.upper() else "trab_mask"))),
                     rendered_time=rendered_time, gobj_creation=hf.get("Original Creation-Date"), created_by=hf.get("Created by"),
                     is_2026_export=is_export, provenance=provenance, derived_by_subtraction=any(p["name"].startswith("Concatenate") for p in procs),
                     peel_iterations=hf.get("Peel Iterations"), patient_name=hf.get("Patient Name"), orig_gobj_dim_p=hf.get("Orig-GOBJ-Dim-p"),
                     orig_gobj_dim_um=hf.get("Orig-GOBJ-Dim-um"), el_from_gobj_dims=el_from_gobj_dims(hf), procedures=proc_summary(procs))
            if v is not None:
                r.update(grid=info["grid"], el_header_mm=info["el_header_mm"], set_voxels=info["set_voxels"], values=info["values"], dtype=info["dtype"], type_code=info["type_code"])
                r["el_header_ok"] = bool(el is not None and all(abs(info["el_header_mm"][i] - el[i]) < 2e-5 for i in range(3)))
                r["tight_box"] = tight_box(v["data"] != 0, v["pos"])
                vols[kind] = v
            if rec.get("identity") and hf.get("Patient Name") and hf["Patient Name"].strip() != rec["identity"]["patient_name"].strip():
                r["patient_name_differs_from_greyscale"] = True
            rec["renderings"][kind] = r
        except Exception as e:
            rec["errors"].append(f"{kind}: {e!r}")

    seg_kind = "SEG_DECOMPRESSED" if "SEG_DECOMPRESSED" in kinds else ("SEG" if "SEG" in kinds else None)
    rec["seg"] = None
    if seg_kind:
        try:
            v, info = aim_info(os.path.join(folder, kinds[seg_kind]), read_data=voxel_checks)
            procs = info["procedures"]
            names = [p["name"] for p in procs]
            s = dict(file=kinds[seg_kind], created_by=info["header"].get("Created by"), times=info["header"]["Time"],
                     procedure_sequence=" > ".join(names), procedures=proc_summary(procs), procedure_details=procs,
                     has_step1_procedures=any(any(k in n for k in STEP1_PROCS) for n in names),
                     step1_procedures=[n for n in names if any(k in n for k in STEP1_PROCS)],
                     fill_offset_duplicate="FillOffsetDuplicate" in names,
                     gobj_masks=[(os.path.basename(p["params"].get("Gobj File", "?")).split("]")[-1], p["params"].get("Peel Iterations")) for p in procs if p["name"].startswith("GobjOrAimMaskAimPeel")],
                     cl_nr_min=[int(p["params"]["min_nr"]) for p in procs if p["name"].startswith("Cl_ExtractNumber") and "min_nr" in p["params"]],
                     lh={k: p["params"].get(k) for p in procs if p["name"].startswith("FFT_LaplaceHamming") for k in ("fft.lp_cut_off_freq", "fft.laplace_hamming.amplitude", "fft.laplace.eps", "fft.redim_power_of_2")},
                     norm_max=[p["params"].get("max") for p in procs if p["name"].startswith("Std_FloatNormMax")],
                     threshold={k: p["params"].get(k) for p in procs if p["name"].startswith("SupThreshold") for k in ("Wished Unit of thresholds", "low_th_input", "upp_th_input", "low corresponds to data value", "upp corresponds to data value", "in_range_value")},
                     set_values=[(p["params"].get("value_zero"), p["params"].get("value_non_zero")) for p in procs if p["name"].startswith("SetAllNonZero")])
            s["lh_native_thresholds"] = [int(s["threshold"]["low corresponds to data value"]), int(s["threshold"]["upp corresponds to data value"])] if s["threshold"].get("low corresponds to data value") else None
            if v is not None:
                s.update(grid=info["grid"], el_header_mm=info["el_header_mm"], set_voxels=info["set_voxels"], values=info["values"], dtype=info["dtype"], type_code=info["type_code"])
                s["n127"], s["n126"] = info["values"].get(127, 0) if info["values"] else None, info["values"].get(126, 0) if info["values"] else None
                s["other_values"] = {k: c for k, c in (info["values"] or {}).items() if k not in (0, 126, 127)}
                s["tight_box"] = tight_box(v["data"] != 0, v["pos"])
                s["tight_box_126"] = tight_box(v["data"] == 126, v["pos"])
                s["tight_box_127"] = tight_box(v["data"] == 127, v["pos"])
                vols["SEG"] = v
            if "SEG_DECOMPRESSED" in kinds and "SEG" in kinds and voxel_checks:
                c = ipldt.read_aim(os.path.join(folder, kinds["SEG"]))
                s["compressed_seg"] = dict(file=kinds["SEG"], grid=grid_of(c), type_code=f"0x{int(c.get('type_code', 0)):08x}",
                                           identical_to_decompressed=bool(grid_of(c) == info["grid"] and np.array_equal(c["data"], v["data"])),
                                           proclog_identical=bool(c["proclog"] == v["proclog"]))
            rec["seg"] = s
        except Exception as e:
            rec["errors"].append(f"SEG: {e!r}")

    rec["maps"] = OrderedDict()
    for mp in MAPS:
        kind = next((k for k in (f"{mp}_DECOMPRESSED", f"{mp}_COMPRESSED") if k in kinds), None)
        if kind is None:
            continue
        try:
            v, info = aim_info(os.path.join(folder, kinds[kind]), read_data=voxel_checks)
            procs = info["procedures"]
            dt = next((p for p in reversed(procs) if "DT_Object" in p["name"] or "Dt_Obj" in p["name"]), None)
            q = dt["params"] if dt else {}
            gobj = os.path.basename(q.get("with GOBJ_file", "")).split("]")[-1]
            m = dict(file=kinds[kind], compressed_on_disk=kind.endswith("_COMPRESSED"), created_by=info["header"].get("Created by"), times=info["header"]["Time"],
                     head_original_file=info["header"].get("Original file"), gobj=gobj,
                     gobj_kind=("cort_mask" if "CORT_MASK" in gobj.upper() else ("trab_mask_corr" if "CORR" in gobj.upper() else ("trab_mask" if "TRAB_MASK" in gobj.upper() else "?"))),
                     peel_iter=q.get("and peel_iter"), ridge_epsilon=q.get("ridge_epsilon"), assign_epsilon=q.get("assign_epsilon"),
                     suppress_boundary=q.get("suppress_boundary"), version=q.get("version"), metric=q.get("metric"),
                     procedures=proc_summary(procs), gobj_masks=[(os.path.basename(p["params"].get("Gobj File", "?")).split("]")[-1], p["params"].get("Peel Iterations")) for p in procs if p["name"].startswith("GobjOrAimMaskAimPeel")],
                     cl_nr_min=[int(p["params"]["min_nr"]) for p in procs if p["name"].startswith("Cl_ExtractNumber") and "min_nr" in p["params"]])
            if v is not None:
                d = v["data"]
                nz = d[d != 0].astype(np.float64)
                m.update(grid=info["grid"], el_header_mm=info["el_header_mm"], support=int(nz.size), value_min=info["value_min"], value_max=info["value_max"], dtype=info["dtype"], type_code=info["type_code"])
                m["mean_vox"] = float(nz.mean()) if nz.size else 0.0
                elx = float(info["el_header_mm"][0])
                m["mean_mm_header_el"] = m["mean_vox"] * elx
                m["mean_mm_grey_el"] = m["mean_vox"] * float(el[0]) if el else None
                if mp == "TRAB_1N":
                    m["tb_n_per_mm_header_el"] = (1.0 / m["mean_mm_header_el"]) if m["mean_mm_header_el"] else None
                m["tight_box"] = tight_box(d != 0, v["pos"])
                vols[mp] = v
            rec["maps"][mp] = m
        except Exception as e:
            rec["errors"].append(f"{mp}: {e!r}")
    for n in rec["misspelled"]:
        try:
            sib = re.sub(r"_TRAB_1N_[A-Z]+\.AIM$", "_TRAB_1N_DECOMPRESSED.AIM", n, flags=re.I)
            same = os.path.exists(os.path.join(folder, sib)) and open(os.path.join(folder, n), "rb").read() == open(os.path.join(folder, sib), "rb").read()
            rec["notes"].append(f"misspelled duplicate {n}: byte-identical to {sib} = {same}")
        except Exception as e:
            rec["errors"].append(f"misspelled {n}: {e!r}")

    # ---- evaluation times
    times = OrderedDict()
    if rec.get("greyscale"):
        times["isq_creation (scan)"] = rec["identity"]["scan_date"]
        times["isq_to_aim (evaluation start)"] = rec["greyscale"]["times"][0] if rec["greyscale"]["times"] else None
    for kind, r in rec["renderings"].items():
        times[f"{kind} gobj creation"] = r.get("gobj_creation")
        times[f"{kind} rendered (export)"] = r.get("rendered_time")
    if rec.get("seg"):
        times["SEG proclog Time lines"] = rec["seg"]["times"]
    for mp, m in rec["maps"].items():
        times[f"{mp} proclog Time lines"] = m["times"]
    rec["times"] = times
    # STEP 1 ran when CORT_MASK.GOBJ was written (CORT_TH's proclog head is the /gobj_to_aim of the periosteal at STEP 1 start)
    rec["step1_time"] = rec["renderings"].get("CORT_MASK_CT", {}).get("gobj_creation")
    rec["periosteal_gobj_time"] = rec["renderings"].get("CT", {}).get("gobj_creation")
    rec["trab_gobj_time"] = (rec["renderings"].get("TRAB_MASK_CORR_CT") or rec["renderings"].get("TRAB_MASK_CT") or {}).get("gobj_creation")

    # ---- voxel consistency checks
    if voxel_checks:
        try:
            rec["checks"] = consistency_checks(vols, rec)
        except Exception as e:
            rec["errors"].append(f"checks: {e!r}\n{traceback.format_exc()}")
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def consistency_checks(vols, rec):
    """(a)-(d) on the union grid of everything present, by global position."""
    trab_kind = "TRAB_MASK_CORR_CT" if "TRAB_MASK_CORR_CT" in vols else ("TRAB_MASK_CT" if "TRAB_MASK_CT" in vols else None)
    parts = [vols[k] for k in ("CT", "CORT_MASK_CT", trab_kind, "SEG", *MAPS) if k and k in vols]
    if not parts:
        return dict(error="nothing to check")
    dim, pos = grid_of(parts[0])
    for v in parts[1:]:
        dim, pos = ops.union_grid(dict(dim=dim, pos=pos), v)
    dim, pos = tuple(dim), tuple(pos)
    out = OrderedDict(frame=[list(dim), list(pos)], trab_rendering=trab_kind)
    P = (ops.on_grid(vols["CT"], dim, pos) != 0) if "CT" in vols else None
    C = (ops.on_grid(vols["CORT_MASK_CT"], dim, pos) != 0) if "CORT_MASK_CT" in vols else None
    T = (ops.on_grid(vols[trab_kind], dim, pos) != 0) if trab_kind else None
    n = lambda m: int(np.count_nonzero(m))
    # (a)
    a = OrderedDict()
    if P is not None and C is not None and T is not None:
        U = C | T
        a.update(periosteal=n(P), cort=n(C), trab=n(T), union=n(U), cort_trab_overlap=n(C & T), periosteal_only=n(P & ~U), union_only=n(U & ~P))
        a["equal"] = a["periosteal_only"] == 0 and a["union_only"] == 0
        a["partition"] = a["equal"] and a["cort_trab_overlap"] == 0
        if not a["equal"]:
            a["periosteal_only_slices"] = [int(z + pos[2]) for z in np.unique(np.nonzero(P & ~U)[0])][:200]
            a["union_only_slices"] = [int(z + pos[2]) for z in np.unique(np.nonzero(U & ~P)[0])][:200]
            a["union_only_in_trab"] = n(T & ~P)
            a["union_only_in_cort"] = n(C & ~P)
            a["periosteal_only_bbox"] = tight_box(P & ~U, pos)
            a["union_only_bbox"] = tight_box(U & ~P, pos)
        a["grid_CT"], a["grid_CORT"], a["grid_TRAB"] = grid_of(vols["CT"]), grid_of(vols["CORT_MASK_CT"]), grid_of(vols[trab_kind])
        a["grid_CORT_equals_CT"] = grid_of(vols["CT"]) == grid_of(vols["CORT_MASK_CT"])
        a["grid_TRAB_inside_CT"] = grid_inside(grid_of(vols[trab_kind]), grid_of(vols["CT"]))
    else:
        a["skipped"] = "missing rendering"
    out["a_periosteal_vs_cort_or_trab"] = a
    # (b)
    b = OrderedDict()
    if "SEG" in vols:
        S = ops.on_grid(vols["SEG"], dim, pos)
        s127, s126, sset = S == 127, S == 126, S != 0
        b.update(seg_set=n(sset), seg127=n(s127), seg126=n(s126))
        if P is not None:
            b["seg_outside_periosteal"] = n(sset & ~P)
        if C is not None:
            b["seg127_outside_cort"] = n(s127 & ~C)
            b["seg126_inside_cort"] = n(s126 & C)
        if T is not None:
            b["seg126_outside_trab"] = n(s126 & ~T)
            b["seg127_inside_trab"] = n(s127 & T)
            if b["seg126_outside_trab"]:
                b["seg126_outside_trab_slices"] = [int(z + pos[2]) for z in np.unique(np.nonzero(s126 & ~T)[0])][:200]
                b["seg126_outside_trab_bbox"] = tight_box(s126 & ~T, pos)
            if b["seg127_inside_trab"]:
                b["seg127_inside_trab_slices"] = [int(z + pos[2]) for z in np.unique(np.nonzero(s127 & T)[0])][:200]
                b["seg127_inside_trab_bbox"] = tight_box(s127 & T, pos)
        if C is not None and T is not None and P is not None:
            b["consistent_with_renderings"] = (b["seg_outside_periosteal"] == 0 and b["seg127_outside_cort"] == 0 and b["seg126_outside_trab"] == 0)
            b["partition_consistent"] = b["consistent_with_renderings"] and b["seg126_inside_cort"] == 0 and b["seg127_inside_trab"] == 0
    else:
        b["skipped"] = "no SEG"
        S = None
    out["b_seg_vs_renderings"] = b
    # (c)
    c = OrderedDict()
    for mp in MAPS:
        if mp not in vols:
            continue
        M = ops.on_grid(vols[mp], dim, pos) != 0
        R = C if MAP_GOBJ[mp] == "cort" else T
        e = OrderedDict(support=n(M))
        if R is not None:
            e["outside_rendering"] = n(M & ~R)
            if e["outside_rendering"]:
                e["outside_rendering_slices"] = [int(z + pos[2]) for z in np.unique(np.nonzero(M & ~R)[0])][:200]
                e["outside_rendering_bbox"] = tight_box(M & ~R, pos)
        if P is not None:
            e["outside_periosteal"] = n(M & ~P)
        other = T if MAP_GOBJ[mp] == "cort" else C
        if other is not None:
            e["inside_other_rendering"] = n(M & other)          # CORT_TH inside the trab rendering / TRAB_* inside CORT_MASK_CT
        if S is not None:
            if mp == "TRAB_TH":
                e["outside_seg126"] = n(M & ~(S == 126))
                e["inside_seg127"] = n(M & (S == 127))
            elif mp in ("TRAB_SP", "TRAB_1N"):
                e["inside_seg"] = n(M & (S != 0))
            else:
                e["inside_seg126"] = n(M & (S == 126))
                e["outside_seg127"] = n(M & ~(S == 127))
                e["inside_seg127"] = n(M & (S == 127))
        c[mp] = e
    if T is not None and S is not None and all(k in vols for k in ("TRAB_TH", "TRAB_SP")):
        touched = (S == 126) | (ops.on_grid(vols["TRAB_TH"], dim, pos) != 0) | (ops.on_grid(vols["TRAB_SP"], dim, pos) != 0)
        if "TRAB_1N" in vols:
            touched |= ops.on_grid(vols["TRAB_1N"], dim, pos) != 0
        c["trab_rendering_untouched"] = dict(voxels=n(T & ~touched), fraction=float(n(T & ~touched)) / max(1, n(T)))
    out["c_map_supports"] = c
    # (d)
    d = OrderedDict()
    if "SEG" in vols:
        sg = grid_of(vols["SEG"])
        d["seg_grid"] = sg
        d["seg_grid_is_tight"] = rec["seg"].get("tight_box") == sg
        if "CT" in vols:
            d["seg_grid_inside_CT_grid"] = grid_inside(sg, grid_of(vols["CT"]))
            d["seg_grid_inside_CT_tight_box"] = grid_inside(sg, rec["renderings"]["CT"]["tight_box"]) if rec["renderings"]["CT"].get("tight_box") else None
        for mp in MAPS:
            if mp in vols:
                mg = grid_of(vols[mp])
                d[f"{mp}_grid"] = mg
                d[f"{mp}_grid_equals_seg"] = mg == sg
                d[f"{mp}_grid_is_tight"] = rec["maps"][mp].get("tight_box") == mg
        if "TRAB_TH" in vols:
            d["TRAB_TH_grid_equals_tight_seg126"] = grid_of(vols["TRAB_TH"]) == rec["seg"].get("tight_box_126")
        if "CORT_TH" in vols and "CORT_MASK_CT" in vols:
            d["CORT_TH_grid_equals_CORT_grid"] = grid_of(vols["CORT_TH"]) == grid_of(vols["CORT_MASK_CT"])
            d["CORT_TH_grid_equals_tight_seg127"] = grid_of(vols["CORT_TH"]) == rec["seg"].get("tight_box_127")
    for k in ("CT", "CORT_MASK_CT", trab_kind):
        if k and k in vols:
            d[f"{k}_grid_is_tight"] = rec["renderings"][k].get("tight_box") == grid_of(vols[k])
    out["d_grids"] = d
    # inference: which trabecular gobj did SEG / maps use?
    inf = OrderedDict()
    if rec["trab_is_corr"] and "SEG" in vols and T is not None:
        outside = b.get("seg126_outside_trab", 0) + sum(c.get(mp, {}).get("outside_rendering", 0) for mp in ("TRAB_TH", "TRAB_SP", "TRAB_1N"))
        inside = b.get("seg127_inside_trab", 0) + c.get("CORT_TH", {}).get("inside_other_rendering", 0)
        if outside == 0 and inside == 0 and a.get("equal"):
            inf["verdict"] = "CORR == the partition used: CT == CORT | TRAB_CORR and SEG / maps sit inside it -- the evaluation used this contour (the correction changed nothing visible, or trab_mask.gobj IS the corrected contour)"
        elif outside == 0 and inside == 0:
            inf["verdict"] = ("SEG 126 / TRAB maps inside TRAB_MASK_CORR_CT, SEG 127 / CORT_TH outside it, CT != CORT | TRAB_CORR only by periosteal voxels in neither: "
                              "consistent with the evaluation having used the CORR contour as trab_mask.gobj (P-only voxels = contour-conversion loss of CORT_MASK.GOBJ = periosteal - CORR)")
        elif outside == 0:
            inf["verdict"] = f"trab used is SMALLER than TRAB_MASK_CORR_CT: {inside} voxels of SEG 127 / CORT_TH inside the CORR rendering -> the evaluation used the uncorrected (or another) trab_mask.gobj"
        elif inside == 0:
            inf["verdict"] = f"trab used is LARGER than TRAB_MASK_CORR_CT: {outside} voxels of SEG 126 / TRAB maps outside the CORR rendering -> the evaluation used the uncorrected (or another) trab_mask.gobj"
        else:
            inf["verdict"] = f"trab used differs from TRAB_MASK_CORR_CT both ways ({outside} outside, {inside} inside) -> the evaluation used the uncorrected (or another) trab_mask.gobj"
        inf["seg126_outside_trab"] = b.get("seg126_outside_trab")
        inf["seg127_inside_trab"] = b.get("seg127_inside_trab")
        inf["cort_th_inside_trab"] = c.get("CORT_TH", {}).get("inside_other_rendering")
        inf["map_support_outside_trab"] = {mp: c.get(mp, {}).get("outside_rendering") for mp in ("TRAB_TH", "TRAB_SP", "TRAB_1N")}
        inf["used_corr"] = outside == 0 and inside == 0
    out["inference"] = inf
    return out


def render_checks(rec):
    """(e) ipldt.contour round trips on the renderings provided (the --render-check pass):
         cort_from_periosteal_minus_trab   render_volume(_CT & ~TRAB rendering)  vs  CORT_MASK_CT
             0 mismatches <=> CORT_MASK.GOBJ was /togobj_from_aim of (periosteal - trab_mask.gobj rendering), i.e. the
             re-evaluation workflow of the double-underscore logs (cort re-derived from the possibly corrected trab contour),
             and the periosteal voxels in neither rendering are exactly the contour-conversion loss ipldt.contour reproduces;
         cort_idempotent                   render_volume(CORT_MASK_CT) vs CORT_MASK_CT (0 <=> the export rendering is a fixed
             point of IPL's contour conversion as ipldt implements it)."""
    from ipldt.contour import render_volume
    folder = rec["folder"]
    kinds = {f["kind"]: n for n, f in rec["files"].items()}
    trab_kind = "TRAB_MASK_CORR_CT" if "TRAB_MASK_CORR_CT" in kinds else ("TRAB_MASK_CT" if "TRAB_MASK_CT" in kinds else None)
    if not all(k in kinds for k in ("CT", "CORT_MASK_CT")) or trab_kind is None:
        return dict(skipped="missing rendering")
    t0 = time.time()
    P = ipldt.read_aim(os.path.join(folder, kinds["CT"]))
    C = ipldt.read_aim(os.path.join(folder, kinds["CORT_MASK_CT"]))
    T = ipldt.read_aim(os.path.join(folder, kinds[trab_kind]))
    d1, p1 = ops.union_grid(P, C)
    dim, pos = ops.union_grid(dict(dim=d1, pos=p1), T)
    dim, pos = tuple(dim), tuple(pos)
    p = ops.on_grid(P, dim, pos) != 0
    c = ops.on_grid(C, dim, pos) != 0
    t = ops.on_grid(T, dim, pos) != 0
    out = OrderedDict(frame=[list(dim), list(pos)], trab_rendering=trab_kind)
    r = render_volume(p & ~t)
    out["cort_from_periosteal_minus_trab"] = dict(rendered=int(r.sum()), cort_mask_ct=int(c.sum()), mismatches=int((r != c).sum()),
                                                  ours_only=int((r & ~c).sum()), ipl_only=int((c & ~r).sum()))
    if out["cort_from_periosteal_minus_trab"]["mismatches"]:
        out["cort_from_periosteal_minus_trab"]["slices"] = [int(z + pos[2]) for z in np.unique(np.nonzero(r != c)[0])][:100]
    rc = render_volume(c)
    out["cort_idempotent"] = dict(mismatches=int((rc != c).sum()))
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ============================================================================================ cohort
def discover(root=ROOT, only=None):
    """[(group, measurement id, folder)] for Distal/*/<id>, Diaphyseal/*/<id> and rerun/<id>."""
    items = []
    for top in TOP_GROUPS:
        for study in sorted(os.listdir(os.path.join(root, top))):
            sp = os.path.join(root, top, study)
            if not os.path.isdir(sp):
                continue
            for meas in sorted(os.listdir(sp)):
                if os.path.isdir(os.path.join(sp, meas)):
                    items.append((f"{top}/{study}", meas, os.path.join(sp, meas)))
    rp = os.path.join(root, RERUN)
    if os.path.isdir(rp):
        for meas in sorted(os.listdir(rp)):
            if os.path.isdir(os.path.join(rp, meas)):
                items.append((RERUN, meas, os.path.join(rp, meas)))
    if only:
        items = [it for it in items if it[1] in only or f"{it[0]}/{it[1]}" in only]
    return items


def rerun_identity(rec, records):
    """rerun/<id>: byte-compare with the original folder of the same id."""
    orig = [r for r in records if r["measurement"] == rec["measurement"] and r["group"] != RERUN]
    if len(orig) != 1:
        rec["rerun_of"] = None
        rec["notes"].append(f"rerun: {len(orig)} original folders with id {rec['measurement']}")
        return
    o = orig[0]
    rec["rerun_of"] = f"{o['group']}/{o['measurement']}"
    same, diff, only_here, only_orig = [], [], [], []
    for name, f in rec["files"].items():
        if f["kind"] == "directory":
            continue
        po = os.path.join(o["folder"], name)
        if not os.path.exists(po):
            only_here.append(name)
            continue
        pa = os.path.join(rec["folder"], name)
        if os.path.getsize(pa) == os.path.getsize(po) and open(pa, "rb").read() == open(po, "rb").read():
            same.append(name)
        else:
            diff.append(name)
    for name in o["files"]:
        if name not in rec["files"]:
            only_orig.append(name)
    rec["rerun_compare"] = dict(identical=same, different=diff, only_in_rerun=only_here, only_in_original=only_orig,
                                all_identical=not diff and not only_here)


def _vms_time(s):
    try:
        return _dt.datetime.strptime(s.strip().split(".")[0], "%d-%b-%Y %H:%M:%S")
    except Exception:
        return None


def gobj_timeline(rec):
    """Seconds between the gobj creation times (periosteal -> CORT_MASK.GOBJ = STEP 1 end -> trab gobj) and the evaluation
    start (ISQ_TO_AIM): a trab gobj written long after CORT_MASK.GOBJ is a contour re-saved (edited) after the evaluation."""
    t_p = _vms_time(rec.get("periosteal_gobj_time") or "")
    t_c = _vms_time(rec.get("step1_time") or "")
    t_t = _vms_time(rec.get("trab_gobj_time") or "")
    t_a = _vms_time(rec.get("identity", {}).get("aim_time") or "")
    out = OrderedDict(periosteal_gobj=rec.get("periosteal_gobj_time"), isq_to_aim=rec.get("identity", {}).get("aim_time"),
                      cort_gobj=rec.get("step1_time"), trab_gobj=rec.get("trab_gobj_time"))
    out["cort_minus_aim_s"] = (t_c - t_a).total_seconds() if t_c and t_a else None
    out["trab_minus_cort_s"] = (t_t - t_c).total_seconds() if t_t and t_c else None
    d = out["trab_minus_cort_s"]
    if d is None:
        out["reading"] = "unknown"
    elif -120 <= d <= 120:
        out["reading"] = "trab gobj written with CORT_MASK.GOBJ (same STEP 1 / re-evaluation run)"
    elif d > 120:
        out["reading"] = f"trab gobj written {d / 60:.0f} min AFTER CORT_MASK.GOBJ: a contour saved (edited) after the evaluation, or a later re-evaluation that did not rewrite CORT_MASK.GOBJ"
    else:
        out["reading"] = f"trab gobj written {-d / 60:.0f} min BEFORE CORT_MASK.GOBJ: the cortical contour was re-derived from this (corrected) trab contour by a later re-evaluation (double-underscore workflow)"
    return out


def plan_for(rec, logs_by_meas):
    """The validation plan of one measurement from its record."""
    p = OrderedDict()
    rec["gobj_timeline"] = gobj_timeline(rec)
    p["gobj_timeline"] = rec["gobj_timeline"]["reading"]
    ident = rec.get("identity", {})
    bone = ident.get("bone")
    group = rec["group"]
    p["step1_preset_from_bone"] = {"tibia": "TIBIA (200000 / 50)", "radius": "RADIUS (800 / 30)"}.get(bone, "?")
    lg = logs_by_meas.get(rec["measurement"], [])
    misc = sorted({(tuple(l["resolved"].get("IPL_MISC1_0") or []), tuple(l["resolved"].get("IPL_MISC1_1") or [])) for l in lg if l["resolved"].get("IPL_MISC1_0")})
    if misc:
        p["step1_preset_from_log"] = ", ".join(f"corner_min {m0[0]} / close2 {m1[0]}" for m0, m1 in misc)
    if group.startswith("Diaphyseal"):
        p["step1_preset_note"] = "diaphyseal logs (610892 D-tibia, 581203 D-radius) both resolve IPL_MISC1 to 800 / 30: run STEP 1 with RADIUS for every diaphyseal measurement and TIBIA only as a cross-check"
    ch = rec.get("checks", {})
    a, b, c = ch.get("a_periosteal_vs_cort_or_trab", {}), ch.get("b_seg_vs_renderings", {}), ch.get("c_map_supports", {})
    if rec.get("rerun_of"):
        p["status"] = "duplicate"
        p["masks"] = p["seg"] = p["maps"] = f"byte-identical copy of {rec['rerun_of']}" if rec.get("rerun_compare", {}).get("all_identical") else f"copy of {rec['rerun_of']} with differences {rec['rerun_compare'].get('different')}"
        return p
    if rec["errors"]:
        p["status"] = "errors"
        p["masks"] = p["seg"] = p["maps"] = "record has errors: " + "; ".join(rec["errors"])[:300]
        return p
    has_seg = rec.get("seg") is not None
    corr = rec["trab_is_corr"]
    cort_r = rec.get("renderings", {}).get("CORT_MASK_CT", {})
    cort_note = (" CORT_MASK_CT is the raw mask a re-evaluation wrote as periosteal MINUS the trab rendering (no CORT_MASK.GOBJ round trip): compare our CORT as CT - render(our TRAB)."
                 if cort_r.get("derived_by_subtraction") else "")
    small = 200
    overlap = a.get("cort_trab_overlap") or 0
    u_only = a.get("union_only") or 0
    a_ok = bool(a) and u_only <= small and overlap <= small
    p_only = a.get("periosteal_only") or 0
    if a and 0 < overlap <= small:
        cort_note += f" The CORT and TRAB renderings overlap in {overlap} contour-boundary voxels (SEG carries 127 there: 127-inside-trab {b.get('seg127_inside_trab')})."
    if a and 0 < u_only <= small and not rec["trab_is_corr"]:
        if str(rec.get("measurement")) == "353308" or str(rec.get("id", "")) == "Diaphyseal/REPRO/353308":
            cort_note += (f" {u_only} rendering voxels lie outside the periosteal rendering: an input-version case (classified 2026-09-14 evening, "
                          "CLOSED 2026-09-15 with direct evidence) -- "
                          "the periosteal gobj on disk is younger than the compartment gobjs (by 31 min) and differs from the version "
                          "IPL consumed by two spike pixels, (1463, 737, 91) and (1453, 739, 154), which added to the periosteal input reproduce "
                          "IPL's CORT_MASK_CT / TRAB_MASK_CT at 0 / 0. Test run 20 block 4 fetched both gobj versions from the scanner on 2026-09-15: "
                          "the older periosteal GOBJ version starts its z 91 / z 154 chains exactly at (1463, 737) and (1453, 739) and "
                          "renders both spikes, while the newer version contains neither vertex, renders neither pixel "
                          "and renders the on-disk X2077003_CT.AIM at 0 mismatching voxels (4,927,853 = 4,927,853); both STEP-1 contour files render "
                          "their own _CT AIMs at 0 (3,506,710 and 1,421,038). Not a contour-rule case and not a STEP-1 difference.")
        else:
            cort_note += f" {u_only} rendering voxels lie outside the periosteal rendering (contour-boundary ties)."
    e = ch.get("e_render", {}).get("cort_from_periosteal_minus_trab", {})
    if e and cort_r.get("derived_by_subtraction"):
        cort_note += (f" CHECK (e): rendering CT - {ch['e_render']['trab_rendering']} with ipldt.contour would drop {e.get('ipl_only')} voxels of this raster "
                      f"(+{e.get('ours_only')}): the contour-conversion loss a CORT_MASK.GOBJ round trip would cause; the raster itself is CT - TRAB exactly (check (a))."
                      if a.get("equal") else f" CHECK (a): the raster is NOT CT - TRAB (periosteal-only {p_only}, union-only {a.get('union_only')}).")
    elif e:
        if e.get("mismatches") == 0:
            cort_note += (f" CHECK (e): CORT_MASK_CT == ipldt.contour.render_volume(CT - {ch['e_render']['trab_rendering']}) exactly, so the cortical contour was "
                          "re-derived from this trabecular contour (the double-underscore re-evaluation workflow) and the P-only voxels are the round-trip loss: exact expected by that formula.")
        else:
            cort_note += (f" CHECK (e): render_volume(CT - {ch['e_render']['trab_rendering']}) differs from CORT_MASK_CT by {e['mismatches']} voxels "
                          f"(+{e['ours_only']} / -{e['ipl_only']}): CORT_MASK_CT is not periosteal minus this trabecular contour (STEP 1's own cortical mask, or another trab contour).")
    if not a:
        p["masks"] = "no renderings to compare"
        mask_level = "none"
    elif not corr:
        if a.get("partition"):
            p["masks"] = "exact expected (STEP 1 from the _CT periosteal rendering, rendered, vs CORT_MASK_CT / TRAB_MASK_CT)." + cort_note
            mask_level = "exact"
        elif a_ok:
            p["masks"] = (f"exact expected after contour rendering: {p_only} periosteal voxels are in neither rendering (contour-conversion loss of the "
                          f"/togobj_from_aim round trip, to be reproduced by ipldt.contour); compare rendered masks, not rasters." + cort_note)
            mask_level = "exact" if overlap == 0 else "caveat"
        else:
            p["masks"] = f"inconsistent: CT != CORT | TRAB (periosteal-only {p_only}, union-only {a.get('union_only')}, overlap {overlap})" + cort_note
            mask_level = "inconsistent"
    else:
        if a.get("equal"):
            p["masks"] = "TRAB_MASK_CORR_CT | CORT_MASK_CT == CT: the corrected contour partitions the periosteal exactly with the cortical mask; exact expected for CORT, TRAB = CT - CORT; the correction itself is only testable against the automatic TRAB (not exported)." + cort_note
        elif a_ok:
            p["masks"] = (f"manual-correction expected for TRAB (hand-edited contour; {p_only} periosteal voxels in neither rendering); CORT_MASK_CT: exact expected "
                          f"if it is periosteal - CORR (test our CORT = CT - TRAB_CORR), otherwise it is STEP 1's automatic cortical mask -> test STEP 1 against it." + cort_note)
        else:
            p["masks"] = (f"manual-correction reaching OUTSIDE the periosteal rendering or overlapping the cortical mask: union-only {a.get('union_only')} "
                          f"(in TRAB_CORR {a.get('union_only_in_trab')}, in CORT {a.get('union_only_in_cort')}), CORT & TRAB_CORR overlap {overlap}, periosteal-only {p_only}" + cort_note)
        mask_level = "manual"
    if not has_seg:
        p["seg"] = "not comparable (no SEG file); maps only, from our own SEG"
        seg_level = "none"
    elif b.get("partition_consistent"):
        p["seg"] = "exact expected (Laplace-Hamming SEG assembled with the renderings provided; float-FFT rounding class of mismatches as on the patella cohort)"
        seg_level = "exact"
    elif (corr and b.get("seg127_outside_cort", 0) == 0 and b.get("seg126_outside_trab", 0) == 0 and b.get("seg127_inside_trab", 0) <= small
          and 0 < b.get("seg_outside_periosteal", 0) <= (a.get("union_only_in_trab") or 0)):
        p["seg"] = (f"exact expected WITH the corrected contour: it reaches {a.get('union_only_in_trab')} voxels outside the periosteal rendering and the SEG carries "
                    f"{b.get('seg_outside_periosteal')} label-126 voxels there (so the LH threshold must be masked by CORT | TRAB_CORR, not by the periosteal rendering alone)"
                    + (f"; {b.get('seg127_inside_trab')} label-127 voxels inside TRAB_CORR (overlap with the cortical rendering)" if b.get("seg127_inside_trab") else ""))
        seg_level = "exact" if not b.get("seg127_inside_trab") else "caveat"
    elif (b.get("seg_outside_periosteal", 0) == 0 and b.get("seg127_outside_cort", 0) == 0 and b.get("seg127_inside_trab", 0) <= small
          and (b.get("seg126_outside_trab", 0) <= small or b.get("seg126_outside_trab_adjacent6_fraction", 0) >= 0.999)):
        p["seg"] = (f"exact expected up to {b.get('seg126_outside_trab')} label-126 voxels outside the trab rendering / {b.get('seg127_inside_trab')} label-127 inside it "
                    f"(contour-boundary voxels, {b.get('seg126_outside_trab_adjacent6', '?')} of them 6-adjacent to the rendering: /gobj_maskaimpeel_ow vs /gobj_to_aim "
                    "rasterisation of the same contour)")
        seg_level = "caveat"
    else:
        p["seg"] = (f"NOT directly comparable with the renderings provided: SEG outside periosteal {b.get('seg_outside_periosteal')}, 127 outside cort {b.get('seg127_outside_cort')}, "
                    f"126 outside trab {b.get('seg126_outside_trab')}, 127 inside trab {b.get('seg127_inside_trab')} -> the evaluation used another trabecular contour; "
                    "regenerate the automatic TRAB (= CT - our CORT from STEP 1) and compare against that")
        seg_level = "inconsistent"
    bad_maps = {mp: e.get("outside_rendering") for mp, e in c.items() if isinstance(e, dict) and e.get("outside_rendering")}
    if not rec.get("maps"):
        p["maps"] = "no maps"
        map_level = "none"
    elif not bad_maps:
        p["maps"] = "exact expected (dt on IPL's SEG + the renderings provided)" + ("" if has_seg else " -- only from our own SEG (no SEG file)")
        map_level = "exact"
    elif all(v <= small for v in bad_maps.values()):
        p["maps"] = f"exact expected up to boundary voxels outside the rendering: {bad_maps}"
        map_level = "caveat"
    else:
        p["maps"] = f"maps computed with a different trabecular contour (support outside the rendering: {bad_maps}); compare against the regenerated automatic TRAB"
        map_level = "inconsistent"
    if any(m.get("compressed_on_disk") for m in rec.get("maps", {}).values()):
        p["maps"] += "; maps are the run-length COMPRESSED files (decoded by ipldt.io)"
    levels = {mask_level, seg_level, map_level}
    if "inconsistent" in levels:
        p["status"] = "inconsistent"
    elif mask_level == "manual":
        p["status"] = "manual-correction" + ("" if "caveat" not in levels else "+caveat")
    elif "caveat" in levels:
        p["status"] = "exact-with-caveats"
    elif seg_level == "none":
        p["status"] = "exact-maps-only"
    else:
        p["status"] = "exact"
    return p


# ============================================================================================ reports
def fmt_grid(g):
    return grid_str(g) if g else "-"


def write_markdown(path, records, logs, summary):
    L = []
    w = L.append
    w("# radius / tibia inventory -- 123 HR-pQCT measurements, IPL exports of 2022-2026 evaluations")
    w("")
    w(f"Generated {summary['generated']} by validation/oslh_inventory.py from `{ROOT}`.  Records: {len(records)} "
      f"({summary['n_with_errors']} with errors).  Voxel checks: {'yes' if summary['voxel_checks'] else 'no'}.")
    w("")
    w("## 1. Cohort")
    w("")
    w("| group | measurements | tibia | radius | UD | D | TRAB_CORR | SEG file | maps | logs |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for g, s in summary["per_group"].items():
        w(f"| {g} | {s['n']} | {s['tibia']} | {s['radius']} | {s['UD']} | {s['D']} | {s['corr']} | {s['seg']} | {s['maps']} | {s['logs']} |")
    w(f"| **all** | {summary['n']} | {summary['tibia']} | {summary['radius']} | {summary['UD']} | {summary['D']} | {summary['corr']} | {summary['seg']} | {summary['maps']} | {summary['n_logs']} |")
    w("")
    w("Studies / subjects: " + "; ".join(f"{k}: {v}" for k, v in summary["per_study"].items()) + ".")
    w(f"Site indices: {summary['sites']}.  Scan dates (ISQ creation) {summary['scan_date_range'][0]} .. {summary['scan_date_range'][1]}; "
      f"ISQ_TO_AIM (evaluation start) {summary['aim_time_range'][0]} .. {summary['aim_time_range'][1]}; STEP 1 (CORT_MASK.GOBJ creation) "
      f"{summary['step1_range'][0]} .. {summary['step1_range'][1]}; contour renderings exported {summary['render_range'][0]} .. {summary['render_range'][1]}.")
    w("")
    w("Calibration: " + "; ".join(f"{k}: {v} measurements" for k, v in summary["calibrations"].items()) + ".")
    w("seg_gauss native thresholds (500 / 3000 mg HA): " + "; ".join(f"{k}: {v}" for k, v in summary["seg_gauss_native"].items()) + ".")
    w("Laplace-Hamming /threshold native numbers in the SEG logs: " + "; ".join(f"{k}: {v}" for k, v in summary["lh_native"].items()) + ".")
    w("Greyscale element sizes (header): " + "; ".join(f"{k}: {v}" for k, v in summary["el_sizes"].items()) + ".")
    w("")
    w("## 2. Evaluation variants (SEG processing-log procedure sequences)")
    w("")
    for i, (seq, members) in enumerate(summary["seg_variants"].items(), 1):
        groups = Counter(m.rsplit("/", 1)[0] for m in members)
        w(f"**Variant {i}** ({len(members)} measurements; {dict(groups)}; e.g. {', '.join(members[:6])}{' ...' if len(members) > 6 else ''})")
        w("")
        w(f"    {seq}")
        w("")
    w("STEP 1 procedures (seg_gauss / morphology) in any SEG or map processing log: "
      f"{summary['step1_in_proclogs']} measurements.  The SEG logs start at the greyscale's ISQ_TO_AIM and record only the "
      "Laplace-Hamming segmentation chain of the trabecular branch (the cortical branch's cl_nr 35 / cort_mask.gobj mask is not in them).")
    w("")
    w("Map logs: " + "; ".join(f"{k}: {v}" for k, v in summary["map_gobjs"].items()) + ".")
    w("")
    w("File provenance of the four contour files (from their processing logs):")
    w("")
    for kind, cnt in summary["provenance"].items():
        w(f"- {kind}: " + "; ".join(f"{k}: {v}" for k, v in cnt.items()))
    w(f"- CORT_MASK_CT provenance per group: {summary['provenance_by_group']}")
    w("")
    w("Gobj timeline (trab gobj creation relative to CORT_MASK.GOBJ creation = end of STEP 1) per group:")
    w("")
    for g, cnt in summary["timeline_classes"].items():
        w(f"- {g}: " + "; ".join(f"{k}: {v}" for k, v in cnt.items()))
    w(f"- trab gobj written more than 2 min AFTER CORT_MASK.GOBJ (contour re-saved after the evaluation): {summary['trab_after_cort'] or 'none'}")
    w("")
    w("## 3. IPL evaluation logs")
    w("")
    for lg in logs:
        w(f"### {lg['measurement']}: {lg['name']}")
        w("")
        w(f"- session {lg.get('session_time')} user {lg.get('user')} node {lg.get('node')} ({lg.get('process_name')}); module {lg['module_versions']} built {lg['built']}; {lg['lines']} lines")
        w(f"- patient {lg['patient_names']}; sections: {lg['section_kinds']}")
        w(f"- resolved symbols: IPL_MISC1_0 {lg['resolved']['IPL_MISC1_0']} IPL_MISC1_1 {lg['resolved']['IPL_MISC1_1']} IPL_PEEL0 {lg['resolved']['IPL_PEEL0']}; "
          f"cl_nr min_numbers {lg['cl_nr_min_numbers']}; close distances {lg['close_distances']}; open distances {lg['open_distances']}; peel_iters {lg['peel_iters']}")
        w(f"- seg_gauss {lg['seg_gauss']}; LH threshold {lg['lh_threshold']}")
        w(f"- mask stage: {lg['mask_stage']}; written files {lg['written_files']}")
        for dc in lg["dt_calls"]:
            printed = ", ".join(f"{k} = {v[0]} [{v[1]}]" for k, v in dc["printed"].items())
            w(f"- {dc['cmd']} input {dc['input_files']} gobj {dc['gobj']} peel {dc['peel_iter']} -> {dc['written_to']}: {printed}; valid {dc.get('valid_pct')} %; inmask {dc.get('inmask_nr')}")
        if lg.get("map_match"):
            w(f"- on-disk maps vs printed: {lg['map_match']}")
        w(f"- trailer: {lg['trailer']}")
        w("")
    w(summary["log_conclusions"])
    w("")
    w(f"## 4. TRAB_MASK_CORR measurements ({summary['corr']}) -- what did SEG and the maps use?")
    w("")
    w(f"Verdict counts (used_corr): {summary['corr_used']}.")
    w("")
    w("| group | meas | base | bone | region | CT == CORT|TRAB_CORR (P-only / U-only / overlap) | SEG 126 outside TRAB_CORR | SEG 127 inside TRAB_CORR | CORT_TH inside TRAB_CORR | TH / SP / 1N support outside TRAB_CORR | (e) CORT vs render(CT - CORR) | CORR gobj created | CORT_MASK gobj created | used CORR | verdict |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        if not r["trab_is_corr"]:
            continue
        ch = r.get("checks", {})
        a, b, c, inf = ch.get("a_periosteal_vs_cort_or_trab", {}), ch.get("b_seg_vs_renderings", {}), ch.get("c_map_supports", {}), ch.get("inference", {})
        e = ch.get("e_render", {}).get("cort_from_periosteal_minus_trab", {})
        w(f"| {r['group']} | {r['measurement']} | {r['base']} | {r['identity'].get('bone')} | {r['identity'].get('region')} | "
          f"{a.get('equal')} ({a.get('periosteal_only')} / {a.get('union_only')} / {a.get('cort_trab_overlap')}) | {b.get('seg126_outside_trab')} | {b.get('seg127_inside_trab')} | "
          f"{c.get('CORT_TH', {}).get('inside_other_rendering')} | "
          f"{c.get('TRAB_TH', {}).get('outside_rendering')} / {c.get('TRAB_SP', {}).get('outside_rendering')} / {c.get('TRAB_1N', {}).get('outside_rendering')} | "
          f"{e.get('mismatches', '-')} | {r.get('trab_gobj_time')} | {r.get('step1_time')} | {inf.get('used_corr')} | {(inf.get('verdict') or '')[:110]} |")
    w("")
    w("## 5. Anomalies")
    w("")
    w(f"Check (e) -- CORT_MASK_CT vs ipldt.contour.render_volume(CT - trabecular rendering): {dict(summary['e_classes'])}; "
      f"export renderings not fixed points of the contour conversion: {summary['e_idempotent_fail'] or 'none'}.  Measurements where the formula fails:")
    w("")
    for line in summary["e_mismatch_list"] or ["- none"]:
        w(f"- {line}" if not line.startswith("-") else line)
    w("")
    for line in summary["anomalies"]:
        w(f"- {line}")
    w("")
    w("## 6. Per-measurement table")
    w("")
    w("| group | meas | base | subject | reg | bone | site | scan date | grey grid | el x (mm) | thr 500/3000 | LH thr | files | CORR | CORT prov. | (a) P-only/U-only/ovl | (b) out-P / 127-out-C / 126-out-T / 127-in-T | (c) TH/SP/1N/CT out | (d) SEG grid | (e) | plan |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        ident, cal = r.get("identity", {}), r.get("calibration", {})
        ch = r.get("checks", {})
        a, b, c, d = ch.get("a_periosteal_vs_cort_or_trab", {}), ch.get("b_seg_vs_renderings", {}), ch.get("c_map_supports", {}), ch.get("d_grids", {})
        e = ch.get("e_render", {}).get("cort_from_periosteal_minus_trab", {})
        cprov = r.get("renderings", {}).get("CORT_MASK_CT", {})
        cprov = "export" if cprov.get("is_2026_export") else ("eval P-T" if cprov.get("derived_by_subtraction") else ("eval" if cprov else "-"))
        files = "".join(k for k, present in (("G", r["present"].get("grey")), ("P", r["present"].get("CT")), ("C", r["present"].get("CORT_MASK_CT")),
                                             ("T", r["present"].get("TRAB_MASK_CT") or r["present"].get("TRAB_MASK_CORR_CT")), ("S", r["present"].get("SEG_DECOMPRESSED")),
                                             ("s", r["present"].get("SEG")), ("M", all(r["present"].get(f"{m}_DECOMPRESSED") for m in MAPS)),
                                             ("m", any(r["present"].get(f"{m}_COMPRESSED") for m in MAPS)), ("L", r["present"].get("LOG"))) if present)
        seg = r.get("seg") or {}
        w(f"| {r['group']} | {r['measurement']} | {r['base']} | {ident.get('subject')} | {ident.get('region')} | {ident.get('bone')} | {ident.get('site')} | "
          f"{(ident.get('scan_date') or '')[:11]} | {fmt_grid(r.get('greyscale', {}).get('grid'))} | {r.get('greyscale', {}).get('el_mm', [0])[0]:.6f} | "
          f"{cal.get('seg_gauss_native')} | {seg.get('lh_native_thresholds')} | {files} | {'CORR' if r['trab_is_corr'] else ''} | {cprov} | "
          f"{a.get('periosteal_only', '-')}/{a.get('union_only', '-')}/{a.get('cort_trab_overlap', '-')} | "
          f"{b.get('seg_outside_periosteal', '-')}/{b.get('seg127_outside_cort', '-')}/{b.get('seg126_outside_trab', '-')}/{b.get('seg127_inside_trab', '-')} | "
          f"{c.get('TRAB_TH', {}).get('outside_rendering', '-')}/{c.get('TRAB_SP', {}).get('outside_rendering', '-')}/{c.get('TRAB_1N', {}).get('outside_rendering', '-')}/{c.get('CORT_TH', {}).get('outside_rendering', '-')} | "
          f"{fmt_grid(d.get('seg_grid'))} | {e.get('mismatches', '-')} | {r.get('plan', {}).get('status')} |")
    w("")
    w("Files: G greyscale, P periosteal _CT, C CORT_MASK_CT, T TRAB_MASK(_CORR)_CT, S SEG_DECOMPRESSED, s SEG.AIM (compressed), M four decompressed maps, m compressed maps, L IPL log.")
    w("")
    w("## 7. Recommended validation plan")
    w("")
    for line in summary["plan_text"]:
        w(line)
    w("")
    w("## 8. Per-measurement plan details")
    w("")
    for r in records:
        p = r.get("plan", {})
        w(f"- **{r['group']}/{r['measurement']}** ({r['base']}, {r.get('identity', {}).get('bone')} {r.get('identity', {}).get('region')}) [{p.get('status')}]: "
          f"masks: {p.get('masks')}; SEG: {p.get('seg')}; maps: {p.get('maps')}; STEP 1 preset: {p.get('step1_preset_from_bone')}"
          + (f" (log: {p['step1_preset_from_log']})" if p.get("step1_preset_from_log") else "") + f"; gobj timeline: {p.get('gobj_timeline')}")
        if r["notes"]:
            w(f"  - notes: {'; '.join(r['notes'])}")
        if r["errors"]:
            w(f"  - errors: {'; '.join(e.splitlines()[0] for e in r['errors'])}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def _range(values):
    """(min, max) of VMS date strings 'd-MON-yyyy hh:mm:ss.cc'."""
    def key(s):
        try:
            return _dt.datetime.strptime(s.strip().split(".")[0], "%d-%b-%Y %H:%M:%S")
        except Exception:
            return None
    vs = [(key(v), v) for v in values if v]
    vs = [x for x in vs if x[0] is not None]
    if not vs:
        return (None, None)
    vs.sort()
    return (vs[0][1].strip(), vs[-1][1].strip())


def summarize(records, logs, voxel_checks):
    s = OrderedDict(generated=_dt.datetime.now().isoformat(timespec="seconds"), voxel_checks=voxel_checks)
    main = [r for r in records if r["group"] != RERUN]
    s["n"], s["n_with_errors"] = len(records), sum(1 for r in records if r["errors"])
    per_group = OrderedDict()
    for r in records:
        g = per_group.setdefault(r["group"], Counter())
        g["n"] += 1
        ident = r.get("identity", {})
        g[ident.get("bone") or "?"] += 1
        g[ident.get("region") or "?"] += 1
        g["corr"] += int(r["trab_is_corr"])
        g["seg"] += int(r.get("seg") is not None)
        g["maps"] += int(len(r.get("maps", {})) == 4)
        g["logs"] += len(r["logs"])
    s["per_group"] = {g: {k: c.get(k, 0) for k in ("n", "tibia", "radius", "UD", "D", "corr", "seg", "maps", "logs")} for g, c in per_group.items()}
    for k in ("tibia", "radius", "UD", "D", "corr", "seg", "maps"):
        s[k] = sum(v[k] for v in s["per_group"].values())
    s["n_logs"] = len(logs)
    study = Counter()
    subjects = {}
    for r in main:
        ident = r.get("identity", {})
        study[ident.get("study")] += 1
        subjects.setdefault(ident.get("study"), set()).add(ident.get("subject"))
    s["per_study"] = {k: f"{v} measurements, {len(subjects[k])} subjects" for k, v in study.items()}
    s["sites"] = dict(Counter(str(r.get("identity", {}).get("site")) for r in main))
    s["scan_date_range"] = _range([r.get("identity", {}).get("scan_date") for r in main])
    s["aim_time_range"] = _range([r.get("identity", {}).get("aim_time") for r in main])
    s["step1_range"] = _range([r.get("step1_time") for r in main])
    s["render_range"] = _range([x.get("rendered_time") for r in main for x in r.get("renderings", {}).values()])
    s["calibrations"] = dict(Counter(f"slope {r['calibration']['slope']} intercept {r['calibration']['intercept']} mu {r['calibration']['mu_scaling']}" for r in main if r.get("calibration")))
    s["seg_gauss_native"] = dict(Counter(str(r["calibration"]["seg_gauss_native"]) for r in main if r.get("calibration")))
    s["lh_native"] = dict(Counter(str((r.get("seg") or {}).get("lh_native_thresholds")) for r in main))
    s["el_sizes"] = dict(Counter(" / ".join(f"{e:.6f}" for e in r["greyscale"]["el_mm"]) for r in main if r.get("greyscale")))
    variants = OrderedDict()
    for r in main:
        if r.get("seg"):
            key = re.sub(re.escape(r["base"]), "<base>", r["seg"]["procedures"], flags=re.I)
            key = re.sub(r"max_nr \d+", "max_nr N", key)
            variants.setdefault(key, []).append(f"{r['group']}/{r['measurement']}")
    s["seg_variants"] = variants
    s["step1_in_proclogs"] = sum(1 for r in main if (r.get("seg") or {}).get("has_step1_procedures") or any("Gauss" in m.get("procedures", "") for m in r.get("maps", {}).values()))
    _base_re = re.compile(r"^[A-Za-z]\d{7}")
    s["map_gobjs"] = dict(Counter(f"{mp} with <base>{_base_re.sub('', m.get('gobj') or '?')} peel {m.get('peel_iter')}" for r in main for mp, m in r.get("maps", {}).items()))
    s["provenance"] = OrderedDict()
    for kind in ("CT", "CORT_MASK_CT", "TRAB_MASK_CT", "TRAB_MASK_CORR_CT"):
        cnt = Counter()
        for r in main:
            x = r.get("renderings", {}).get(kind)
            if x:
                cnt["2026 export rendering" if x.get("is_2026_export") else ("evaluation-written, periosteal MINUS trab (Concatenate)" if x.get("derived_by_subtraction") else f"evaluation-written rendering ({x.get('rendered_time', '')[:11]})")] += 1
        s["provenance"][kind] = dict(cnt)
    s["provenance_by_group"] = OrderedDict()
    for r in main:
        x = r.get("renderings", {}).get("CORT_MASK_CT", {})
        s["provenance_by_group"].setdefault(r["group"], Counter())["export" if x.get("is_2026_export") else "evaluation-written"] += 1
    s["provenance_by_group"] = {g: dict(c) for g, c in s["provenance_by_group"].items()}
    s["a_classes"] = dict(Counter(("CT == CORT | TRAB" if (r.get("checks", {}).get("a_periosteal_vs_cort_or_trab", {}).get("equal")) else
                                   ("periosteal voxels in neither only" if r.get("checks", {}).get("a_periosteal_vs_cort_or_trab", {}).get("union_only") == 0
                                    and r.get("checks", {}).get("a_periosteal_vs_cort_or_trab", {}).get("cort_trab_overlap") == 0 else "union-only / overlap voxels"))
                                  for r in main if r.get("checks", {}).get("a_periosteal_vs_cort_or_trab", {}).get("periosteal") is not None))
    s["b_classes"] = dict(Counter(("consistent partition" if b.get("partition_consistent") else
                                   (f"126 outside trab <= 200 (boundary voxels)" if b.get("seg_outside_periosteal", 0) == 0 and b.get("seg127_outside_cort", 0) == 0
                                    and 0 < b.get("seg126_outside_trab", 0) <= 200 and b.get("seg127_inside_trab", 0) == 0 else "other"))
                                  for r in main for b in [r.get("checks", {}).get("b_seg_vs_renderings", {})] if b and not b.get("skipped")))
    s["seg126_outside_trab_total"] = sum(r.get("checks", {}).get("b_seg_vs_renderings", {}).get("seg126_outside_trab", 0) or 0 for r in main)
    s["seg126_outside_trab_list"] = [(f"{r['group']}/{r['measurement']}", b.get("seg126_outside_trab"), b.get("seg126_outside_trab_adjacent6"), b.get("seg126_outside_trab_adjacent26"))
                                     for r in main for b in [r.get("checks", {}).get("b_seg_vs_renderings", {})] if b.get("seg126_outside_trab")]
    s["corr_used"] = dict(Counter(str(r.get("checks", {}).get("inference", {}).get("used_corr")) for r in main if r["trab_is_corr"]))
    s["timeline_classes"] = OrderedDict()
    for g in sorted({r["group"] for r in main}):
        s["timeline_classes"][g] = dict(Counter(r.get("gobj_timeline", {}).get("reading", "unknown").split(":")[0] for r in main if r["group"] == g))
    s["trab_after_cort"] = [f"{r['group']}/{r['measurement']} ({r['base']}, {'CORR' if r['trab_is_corr'] else 'plain'}, +{r['gobj_timeline']['trab_minus_cort_s'] / 60:.0f} min)"
                            for r in main if (r.get("gobj_timeline", {}).get("trab_minus_cort_s") or 0) > 120]
    s["e_classes"] = OrderedDict()
    for label, sel in (("CORR", lambda r: r["trab_is_corr"]), ("plain", lambda r: not r["trab_is_corr"])):
        s["e_classes"][label] = dict(Counter(("not run" if "e_render" not in r.get("checks", {}) else
                                              ("skipped" if r["checks"]["e_render"].get("skipped") else
                                               ("CORT == render(CT - TRAB)" if r["checks"]["e_render"]["cort_from_periosteal_minus_trab"]["mismatches"] == 0 else "differs")))
                                             for r in main if sel(r)))
    s["e_idempotent_fail"] = [f"{r['group']}/{r['measurement']}" for r in main if r.get("checks", {}).get("e_render", {}).get("cort_idempotent", {}).get("mismatches")]
    s["d_classes"] = dict(Counter(("CORT_TH grid == CORT_MASK_CT grid" if d.get("CORT_TH_grid_equals_CORT_grid") else
                                   ("CORT_TH grid == tight box of SEG 127" if d.get("CORT_TH_grid_equals_tight_seg127") else "CORT_TH on its own tight box"))
                                  for r in main for d in [r.get("checks", {}).get("d_grids", {})] if "CORT_TH_grid" in d))
    s["d_trab_th"] = dict(Counter(("TRAB_TH grid == SEG grid" if d.get("TRAB_TH_grid_equals_seg") else ("TRAB_TH grid == tight box of SEG 126" if d.get("TRAB_TH_grid_equals_tight_seg126") else "other"))
                                  for r in main for d in [r.get("checks", {}).get("d_grids", {})] if "TRAB_TH_grid" in d))
    s["e_mismatch_list"] = []
    for r in main:
        e = r.get("checks", {}).get("e_render", {}).get("cort_from_periosteal_minus_trab", {})
        if e.get("mismatches"):
            a = r.get("checks", {}).get("a_periosteal_vs_cort_or_trab", {})
            s["e_mismatch_list"].append(f"{r['group']}/{r['measurement']} ({r['base']}, {'CORR' if r['trab_is_corr'] else 'plain'}, CORT {'raster' if r['renderings'].get('CORT_MASK_CT', {}).get('derived_by_subtraction') else 'export'}): "
                                        f"render(CT - TRAB) vs CORT_MASK_CT {e['mismatches']} (+{e['ours_only']} / -{e['ipl_only']}), slices {e.get('slices', [])[:30]}; "
                                        f"(a) P-only {a.get('periosteal_only')} U-only {a.get('union_only')} overlap {a.get('cort_trab_overlap')}")
    s["map_match"] = [(lg["measurement"], lg["name"], {k: v["diff"] for k, v in lg.get("map_match", {}).items()}) for lg in logs]
    # anomalies
    an = []
    for r in records:
        tag = f"{r['group']}/{r['measurement']} ({r['base']})"
        for e in r["errors"]:
            an.append(f"{tag}: ERROR {e.splitlines()[0]}")
        for n in r["notes"]:
            an.append(f"{tag}: {n}")
        if not r.get("seg"):
            an.append(f"{tag}: no SEG file")
        if any(m.get("compressed_on_disk") for m in r.get("maps", {}).values()):
            an.append(f"{tag}: maps are *_COMPRESSED (run-length) files")
        if len(r.get("maps", {})) != 4:
            an.append(f"{tag}: maps present {list(r.get('maps', {}))}")
        name_diff = []
        for kind, x in r.get("renderings", {}).items():
            if "el_header_ok" in x and not x["el_header_ok"]:
                an.append(f"{tag}: {kind} header el {['%.5f' % e for e in x['el_header_mm']]} (= Orig-GOBJ-Dim-um / Orig-GOBJ-Dim-p {x.get('el_from_gobj_dims') and ['%.5f' % e for e in x['el_from_gobj_dims']]}); use the greyscale el")
            if x.get("patient_name_differs_from_greyscale"):
                name_diff.append(f"{kind} {x.get('patient_name')!r}")
            if not x.get("is_2026_export"):
                an.append(f"{tag}: {kind} is not a 2026 export rendering but a file the evaluation wrote: {x.get('provenance')}")
        if name_diff:
            an.append(f"{tag}: gobj header patient name differs from the greyscale {r['identity']['patient_name']!r}: " + "; ".join(name_diff))
        seg = r.get("seg") or {}
        if seg.get("other_values"):
            an.append(f"{tag}: SEG carries values other than 0/126/127: {seg['other_values']}")
        lh = seg.get("lh") or {}
        if lh and (lh.get("fft.lp_cut_off_freq") != "0.30000" or lh.get("fft.laplace_hamming.amplitude") != "1.00000" or lh.get("fft.laplace.eps") != "0.45000"):
            an.append(f"{tag}: NON-STANDARD Laplace-Hamming parameters in the SEG log: {lh} (standard: cutoff 0.30000, amplitude 1.00000, eps 0.45000)")
        if seg.get("lh_native_thresholds") and seg["lh_native_thresholds"] != [15564, 32767]:
            an.append(f"{tag}: non-standard LH threshold native numbers {seg['lh_native_thresholds']}")
        if seg.get("cl_nr_min") and seg["cl_nr_min"] != [70]:
            an.append(f"{tag}: SEG log cl_nr min numbers {seg['cl_nr_min']} (expected [70] for the trabecular branch)")
        if seg.get("compressed_seg") and not seg["compressed_seg"]["identical_to_decompressed"]:
            an.append(f"{tag}: SEG.AIM differs from SEG_DECOMPRESSED")
        ch = r.get("checks", {})
        a, b, c, d = ch.get("a_periosteal_vs_cort_or_trab", {}), ch.get("b_seg_vs_renderings", {}), ch.get("c_map_supports", {}), ch.get("d_grids", {})
        # (a): the 'periosteal voxels in neither rendering' class and the <= 200-voxel CORT/TRAB rendering overlap of the 2026 single-run
        # evaluations are documented in section 7; list union-only voxels and larger overlaps only
        if a and (a.get("union_only") or (a.get("cort_trab_overlap") or 0) > 200):
            an.append(f"{tag}: (a) union-only {a['union_only']} (in TRAB {a.get('union_only_in_trab')}, in CORT {a.get('union_only_in_cort')}), CORT & TRAB overlap {a['cort_trab_overlap']}, periosteal-only {a['periosteal_only']}")
        # (b): the single-boundary-voxel class (<= 200) is counted in section 7; list the rest
        if b and not b.get("skipped") and not b.get("partition_consistent", True):
            if (b.get("seg_outside_periosteal", 0) > (a.get("union_only_in_trab") or 0) or b.get("seg127_outside_cort", 0) or b.get("seg126_outside_trab", 0) > 200
                    or b.get("seg127_inside_trab", 0) > 200 or b.get("seg126_inside_cort", 0) > 200):
                an.append(f"{tag}: (b) SEG vs renderings: outside periosteal {b.get('seg_outside_periosteal')}, 127 outside cort {b.get('seg127_outside_cort')}, 126 outside trab {b.get('seg126_outside_trab')}, 126 inside cort {b.get('seg126_inside_cort')}, 127 inside trab {b.get('seg127_inside_trab')}")
        for mp, e in c.items():
            overlap_class = mp == "TRAB_TH" and e.get("outside_seg126") == e.get("inside_seg127") == e.get("inside_other_rendering") and (e.get("inside_other_rendering") or 0) <= 200
            if isinstance(e, dict) and not overlap_class and (e.get("outside_rendering") or e.get("outside_periosteal") or e.get("outside_seg126") or (e.get("inside_other_rendering") or 0) > 200
                                                              or (mp == "TRAB_SP" and e.get("inside_seg")) or e.get("inside_seg126")):
                an.append(f"{tag}: (c) {mp} support {e.get('support')}: outside rendering {e.get('outside_rendering')}, outside periosteal {e.get('outside_periosteal')}, "
                          f"inside the other compartment's rendering {e.get('inside_other_rendering')}, "
                          + ", ".join(f"{k} {v}" for k, v in e.items() if k in ("outside_seg126", "inside_seg127", "inside_seg", "inside_seg126")))
        if d:
            if d.get("seg_grid_is_tight") is False:
                an.append(f"{tag}: (d) SEG grid {fmt_grid(d.get('seg_grid'))} is not the tight box of its support")
            if d.get("seg_grid_inside_CT_grid") is False:
                an.append(f"{tag}: (d) SEG grid {fmt_grid(d.get('seg_grid'))} not inside the _CT grid")
            if d.get("TRAB_TH_grid_equals_seg") is False and d.get("TRAB_TH_grid_equals_tight_seg126") is False:
                an.append(f"{tag}: (d) TRAB_TH grid {fmt_grid(d.get('TRAB_TH_grid'))} is neither the SEG grid nor the tight box of SEG == 126")
    s["anomalies"] = an
    # log conclusions
    lc = []
    for lg in logs:
        lc.append(f"- {lg['measurement']} {lg['name']}: mask stage {lg['mask_stage']}, IPL_MISC1_0 {lg['resolved']['IPL_MISC1_0']}, IPL_MISC1_1 {lg['resolved']['IPL_MISC1_1']}, "
                  f"cl_nr min_numbers {[m for _, m in lg['cl_nr_min_numbers']]}, close {[c for _, c in lg['close_distances']]}, seg_gauss native {[g['native'] for g in lg['seg_gauss']]}, "
                  f"LH native {[t['native'] for t in lg['lh_threshold']]}, session {lg.get('session_time')}, trailer P4 {lg['trailer'].get('P4')}")
    s["log_conclusions"] = "\n".join(lc)
    # plan text
    pt = []
    pt.append("Per measurement the plan classes are: **exact** (every stage should reproduce IPL's file voxel for voxel from the inputs provided, "
              "like the 21-patella cohort), **manual-correction** (TRAB_MASK_CORR_CT is a hand-edited contour: STEP 1 cannot produce it, and whether "
              "IPL's SEG / maps used it is decided by checks (b) + (c) above), **inconsistent** (the files contradict each other), **duplicate** (rerun/ copies).")
    pt.append("")
    cnt = Counter(r.get("plan", {}).get("status") for r in records)
    pt.append("Counts: " + ", ".join(f"{k}: {v}" for k, v in cnt.items()) + ".")
    pt.append("")
    pt.append("Facts the plan rests on: (i) every map log names <base>_trab_mask.gobj / <base>_cort_mask.gobj with peel_iter -1, never the _CORR gobj, so which contour "
              f"the CORR measurements used is decided per measurement by checks (b) + (c) (used_corr counts: {s['corr_used']}); (ii) check (a) classes: {s['a_classes']} -- "
              "'periosteal voxels in neither' are contour-conversion losses of the /togobj_from_aim round trip (the cortical or trabecular gobj smooths away thin bits), "
              f"never union-only voxels; (iii) check (b) classes: {s['b_classes']}, label-126 voxels outside the trab rendering per measurement "
              f"(measurement, count, 6-adjacent to the rendering, 26-adjacent): {s['seg126_outside_trab_list']} -- except 991161 these are contour-boundary voxels "
              "(IPL's /gobj_maskaimpeel_ow keeps a contour-boundary voxel that /gobj_to_aim does not), a mismatch class ipldt's SEG assembly will show; "
              "(iv) CORT_MASK_CT provenance per group: "
              f"{s['provenance_by_group']} -- in the 2024 diaphyseal evaluations the exported CORT_MASK_CT / TRAB_MASK_CT are the raw masks the double-underscore re-evaluation "
              "wrote (cort = periosteal rendering MINUS trab rendering, no CORT_MASK.GOBJ round trip), and the on-disk maps reproduce THAT run's printed values exactly "
              f"(diffs {[(m, d) for m, _, d in s['map_match']]}); (v) check (e) -- CORT_MASK_CT vs ipldt.contour.render_volume(CT - trab rendering): {dict(s['e_classes'])}; "
              f"export renderings that are not fixed points of ipldt's contour conversion: {s['e_idempotent_fail'] or 'none'}.")
    pt.append("")
    pt.append(f"Grids (check d): CORT_TH {s['d_classes']}; TRAB_TH {s['d_trab_th']}; the trabecular maps TRAB_SP / TRAB_1N always share the SEG grid.  "
              "CORT_TH lives on the CORT_MASK.AIM raster grid of the evaluation, which is not the margin-carrying gobj grid of the 2026 CORT_MASK_CT export -- compare by position.")
    pt.append("")
    pt.append("1. STEP 1 (ipldt.step1.cort_trab_separation from the _CT periosteal rendering): compare with CORT_MASK_CT and, where no CORR, TRAB_MASK_CT; "
              "with CORR compare TRAB against CT - CORT_MASK_CT (the automatic trabecular mask IS periosteal minus the cortical mask; Script 32: 29 = 00 - 28) and "
              "report the CORR difference separately.  Preset: RADIUS (800 / 30) for the radius sites 20 / 21; for the diaphyseal tibiae the two diaphyseal logs "
              "resolve IPL_MISC1 to 800 / 30 as well (610892 is a D tibia), so run both presets and record which one reproduces CORT_MASK_CT; distal tibiae: TIBIA (200000 / 50), unverified by a log.")
    pt.append("2. SEG (ipldt.ormir.laplace_hamming_threshold + ipl_seg_assembly on the renderings provided): meaningful on every measurement whose (b) counts are 0; "
              "where SEG 126 lies outside TRAB_MASK_CORR_CT the SEG was assembled with the uncorrected trab_mask.gobj -> compare against our regenerated uncorrected TRAB instead.  "
              "Note the three SEG-log variants (section 2): the distal 2022 chain has no periosteal masking before cl_nr and applies bounding_box_cut after the trab mask, "
              "the diaphyseal chains mask with the periosteal gobj first (as Script 32 lines 407-472) -- the ipl_seg_assembly order must follow the variant.")
    pt.append("3. dt maps (ipldt.ormir.ipl_morphometry on IPL's SEG + the renderings): TRAB_TH from SEG == 126 (TRAB_SEG), TRAB_SP / TRAB_1N from SEG with the trabecular contour, "
              "CORT_TH from CORT_MASK_CT with the cortical contour; expected 0 mismatches wherever (c) shows the supports inside the renderings; 610892 has no SEG, so only the maps from our own SEG.")
    pt.append("4. Metrics: BV/TV, Tb.Th, Tb.Sp, Tb.N, Ct.Th from the maps at the header's x element size; the six logs give printed IPL values for 610892 and 581203 (both runs).")
    s["plan_text"] = pt
    return s


# ============================================================================================ main
def main(argv=None):
    global _LOGFH
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--log-dir", default=LOG_DIR)
    ap.add_argument("--measurements", nargs="*", default=None, help="measurement ids (or group/id) to process")
    ap.add_argument("--no-voxel-checks", action="store_true")
    ap.add_argument("--resume", action="store_true", help="keep records/<group>_<id>.json already written")
    ap.add_argument("--render-check", action="store_true",
                    help="add check (e) (ipldt.contour round trips, ~5 s per measurement) to the records already written, then rebuild the report")
    args = ap.parse_args(argv)
    if args.render_check:
        args.resume = True
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "records"), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    _LOGFH = open(os.path.join(args.log_dir, f"inventory_{_dt.datetime.now():%Y%m%d_%H%M%S}.log"), "w", encoding="utf-8")
    voxel = not args.no_voxel_checks
    items = discover(args.root, set(args.measurements) if args.measurements else None)
    log(f"{len(items)} measurement folders under {args.root}; voxel checks {'on' if voxel else 'off'}")
    records = []
    for group, meas, folder in items:
        rp = os.path.join(args.out, "records", f"{group.replace('/', '_')}_{meas}.json")
        if args.resume and os.path.exists(rp):
            with open(rp, encoding="utf-8") as fh:
                rec = json.load(fh)
            if args.render_check and "checks" in rec and "e_render" not in rec["checks"]:
                try:
                    rec["checks"]["e_render"] = render_checks(rec)
                except Exception as ex:
                    rec["errors"].append(f"render_check: {ex!r}")
                    rec["checks"]["e_render"] = dict(error=repr(ex))
                e = rec["checks"]["e_render"]
                log(f"{group}/{meas} {rec.get('base')}: (e) CORT_MASK_CT vs render(CT - {e.get('trab_rendering')}): {e.get('cort_from_periosteal_minus_trab', e)} | idempotent {e.get('cort_idempotent')} | {e.get('seconds')}s")
                with open(rp, "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=1)
            else:
                log(f"{group}/{meas}: resumed")
            records.append(rec)
            continue
        try:
            rec = measurement_record(group, meas, folder, voxel_checks=voxel)
        except Exception as e:
            rec = OrderedDict(group=group, measurement=meas, folder=folder.replace("\\", "/"), base=None, errors=[f"record: {e!r}\n{traceback.format_exc()}"],
                              notes=[], files={}, present={}, logs=[], misspelled=[], other_files=[], trab_is_corr=False, renderings={}, maps={}, seg=None)
        records.append(rec)
        ident = rec.get("identity", {})
        ch = rec.get("checks", {})
        a, b, c = ch.get("a_periosteal_vs_cort_or_trab", {}), ch.get("b_seg_vs_renderings", {}), ch.get("c_map_supports", {})
        log(f"{group}/{meas} {rec.get('base')} {ident.get('subject')} {ident.get('region')} {ident.get('bone')} site {ident.get('site')} "
            f"| grey {fmt_grid(rec.get('greyscale', {}).get('grid'))} | thr {rec.get('calibration', {}).get('seg_gauss_native')} | "
            f"{'CORR' if rec['trab_is_corr'] else 'plain'} | (a) eq {a.get('equal')} P-only {a.get('periosteal_only')} U-only {a.get('union_only')} ovl {a.get('cort_trab_overlap')} "
            f"| (b) outP {b.get('seg_outside_periosteal')} 127outC {b.get('seg127_outside_cort')} 126outT {b.get('seg126_outside_trab')} 127inT {b.get('seg127_inside_trab')} "
            f"| (c) out {[c.get(mp, {}).get('outside_rendering') for mp in MAPS]} | {rec.get('seconds')}s"
            + (f" | ERRORS {len(rec['errors'])}" if rec["errors"] else ""))
        for e in rec["errors"]:
            log("   ERROR " + e.replace("\n", "\n   "))
        with open(rp, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
    for rec in records:
        if rec["group"] == RERUN:
            rerun_identity(rec, records)
    # logs
    logs = []
    for rec in records:
        for n in rec["logs"]:
            try:
                lg = parse_ipl_log(os.path.join(rec["folder"], n))
            except Exception as e:
                lg = dict(name=n, path=os.path.join(rec["folder"], n), error=repr(e))
            lg["measurement"] = f"{rec['group']}/{rec['measurement']}"
            # match the printed dt statistics against the on-disk map means (header el)
            mm = {}
            for dc in lg.get("dt_calls", []):
                target = {"/dt_thickness": ("TRAB_TH", "Th") if "trab_seg" in " ".join(str(x) for x in dc["input_files"]) else ("CORT_TH", "Th"),
                          "/dt_spacing": ("TRAB_SP", "BG Th") if dc["gobj"].endswith("_trab_mask.gobj") else (None, None),
                          "/dt_number": ("TRAB_1N", "MAT BG Th")}.get(dc["cmd"], (None, None))
                mp, key = target
                if mp and mp in rec.get("maps", {}) and key in dc["printed"] and rec["maps"][mp].get("mean_mm_header_el") is not None:
                    ours = rec["maps"][mp]["mean_mm_header_el"]
                    mm[mp] = dict(printed=dc["printed"][key][0], file_mean_mm=round(ours, 6), diff=round(ours - dc["printed"][key][0], 6))
            lg["map_match"] = mm
            logs.append(lg)
    logs_by_meas = {}
    for lg in logs:
        logs_by_meas.setdefault(lg["measurement"].split("/")[-1], []).append(lg)
    for rec in records:
        rec["plan"] = plan_for(rec, logs_by_meas)
    summary = summarize(records, logs, voxel)
    with open(os.path.join(args.out, "inventory.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(summary=summary, records=records, logs=logs), fh, indent=1)
    write_markdown(os.path.join(args.out, "inventory.md"), records, logs, summary)
    log(f"wrote {os.path.join(args.out, 'inventory.json')} and inventory.md; {len(records)} records, {len(logs)} logs, {len(summary['anomalies'])} anomaly lines")
    for line in summary["plan_text"][2:3]:
        log(line)
    _LOGFH.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
