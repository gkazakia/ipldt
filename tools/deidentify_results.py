#!/usr/bin/env python
"""De-identify the per-scan validation records for publication, and scan a tree for identifiers.

    python tools/deidentify_results.py stage --src <internal repository> --key <key file>
           --oslh-auto <set> --oslh-noedit <set> [--dst .] [--mapping <csv>] [--dry-run]
    python tools/deidentify_results.py verify [--root .] [--src <internal repository>]

`stage` reads <src>/validation/results/ (never writes there) and writes de-identified copies into
<dst>/validation/results/ under the same directory names, so the scripts under validation/ and manuscript/ read them
unchanged.  It also writes validation/result_sets.py, the one place that names the two radius/tibia result sets, and
then carries the scan pseudonyms (below) through every other text file of <dst> and every file name.
`verify` scans a whole tree for identifiers, vendor material, local paths and references to internal material and exits
non-zero on any hit; with --src it adds the literal identifiers found in the internal records (names, codes, dates,
file names, patient indices, every hh:mm:ss.cc time of day and every scan identifier that is not printed by the paper)
to the scan (held in memory only).

The policy is an allowlist and fails closed: a record key that is neither kept nor dropped below stops the run, and
so does any identifier left in a kept value.

What is removed from every record written by validation/validate_dataset.py (radius/tibia sets, patella porosity):
  meta.patient_name and meta.scan_time (meta.study_label, the study token of the name, is added instead);
  folder, inputs.<any>.path / mtime / md5, duplicates[].path / md5, gobj_timeline, the contour-file and
  evaluation-log details of variants.evidence;
  the whole inventory block except identity.{study, region, bone, site, index_measurement, name_notes}, calibration
  and plan.status; identity.subject is replaced by a keyed pseudonym (HMAC-SHA256, see below).
The patella configuration-B records carry no identifiers and pass through a key allowlist; the patella BV/TV table
loses its local path and the sheet file names; the table of IPL's printed values keeps only
id, cohort, site, excluded, sheet_ok, sheet_type and Ct_Po; cohort_studies.csv holds the study of each scan.

Pseudonyms: identity.subject (the study's own participant code) becomes '<study>-<6 hex digits>' of
HMAC-SHA256(key, '<study>|<code>').  An unkeyed hash of a sequential code could be reversed by enumeration, so the
key is a secret file that must live OUTSIDE the public tree; the same key gives the same pseudonyms on every
re-staging.  --init-key creates it.  Two codes of one study that map to one pseudonym stop the run.

Scan identifiers get keyed pseudonyms from the same key and the same HMAC-SHA256, each in the form the scripts parse:
  the scanner's measurement number of a radius / tibia record (<region>/<study>/<nnnn>, tag <region>_<study>_<nnnn>,
    identity.index_measurement)  ->  six decimal digits, 100000 + HMAC(key, 'MEAS|<nnnn>') mod 900000;
  the patella study code PFJ<nnn>_<L|R>  ->  'PFJ-<6 hex digits>_<L|R>', the participant part being
    HMAC(key, 'PFJ|PFJ<nnn>') like every other participant code (both knees of a participant share it);
  the scanner's file base (<letter>0nnnnnn, the name of every AIM / GOBJ file of a measurement)  ->
    'X' and seven digits, 1000000 + HMAC(key, 'BASE|<base>') mod 9000000.
They are applied consistently to file names, record fields, CSV cells, sidecars, facts, tables, documentation and the
comments, docstrings and strings of the shipped code and tests, so every cross-reference still resolves.  No scan
identifier is kept as it is: the paper prints none (its figures name a scan by its site), so PRINTED_BY_THE_PAPER is
empty; `stage` re-reads the paper under <src>/manuscript (text and the drawn text of its SVG figures) and stops if it
prints a scan identifier that the list does not hold (or the list holds one it does not print).
--mapping writes every pseudonym with its code (PRIVATE; outside the tree); an existing map is checked first, and a
pseudonym that would change (a different key) stops the run.

Private terms: further words that must not appear in the published tree and are not scan identifiers (names of
material that is not distributed) are not listed in this file.  `stage` and `verify` read them from an optional pattern
file kept outside the tree (--denylist, or the environment variable IPLDT_DENYLIST): one regular expression per line,
preceded by its scope, `text` (text files and file names) or `any` (also every byte of every file and the base64
text of embedded images), matched case-insensitively; lines starting with '#' are comments.  A hit is reported as
'private_term' without the matched text.
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import shutil
import sys
import tokenize
import zlib
from collections import Counter, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SELF = "tools/deidentify_results.py"

# ------------------------------------------------------------------------------------------------ the result sets
PATELLA_VD_SET = "porosity_AB_patella"                         # validate_dataset schema, 21 patellae
PATELLA_B_SETS = ["from_ipl_contour_ceil_dt", "from_ipl_contour_ceil"]
COPY_FILES = ["records.json", "sample_means.csv", "from_ipl_contour_ceil_dt/summary.md",
              "porosity_AB_summary_n137.json", "pooled_137_comparison.json"]
BVTV_A = "patella_bvtv_A/records.json"
PRINTED_IN = "ipl_printed_values_137.csv"
PRINTED_COLS = ["id", "cohort", "site", "excluded", "sheet_ok", "sheet_type", "Ct_Po"]
STUDIES_OUT = "cohort_studies.csv"
OSLH_SET_RX = re.compile(r"^oslh_(auto|noedit)_v\d+$")
RECORD_NAME_RX = re.compile(r"^(?:[A-Za-z]+_[A-Za-z]+_\d+|PFJ\d+_[LR])\.json$")

# ------------------------------------------------------------------------------------------------ record policy
VD_TOP_KEEP = {"A", "B", "base", "corr", "cort_object_rule", "dataset", "dt_backend", "duplicates", "grey", "group", "id",
               "inputs", "inventory", "ipl_maps", "ipl_renderings", "ipl_support", "memory", "meta", "notes", "periosteal",
               "porosity", "preset", "seg_missing", "seg_variant_used", "tag", "timings", "trab_th_definition",
               "trab_th_grid_rule", "variants", "warnings"}
VD_TOP_DROP = {"folder", "gobj_timeline"}
META_KEEP = {"bone", "bone_key", "region", "site_index", "study_label"}
META_DROP = {"patient_name", "scan_time"}
INPUT_KEEP = {"type", "dim", "pos", "size"}
INPUT_DROP = {"path", "mtime", "md5"}
DUP_KEEP = {"size", "identical_to"}
DUP_DROP = {"path", "md5"}
# contour-file names with the scanner's disk paths, and the evaluation-log details (file names carry the scanner's
# patient index, the transcript report carries run dates)
EVIDENCE_DROP = {"gobj_mask_files", "eval_log", "eval_log_path", "eval_log_file", "rejected"}
IDENTITY_KEEP = ["study", "region", "bone", "site", "index_measurement", "name_notes"]
PATELLA_B_TOP = {"base", "bvtv", "contours", "dt_backend", "grey", "ipl_log", "maps", "morphometry", "periosteal", "seg",
                 "step1", "subject", "timings"}
STUDY_TOKEN = re.compile(r";\s*([A-Za-z]+study)\b", re.I)

# ------------------------------------------------------------------------------------------------ scan identifiers
# The scan identifiers the paper prints would stay as they are (everything else of the three kinds is pseudonymised).
# The paper prints none: its text and figures name a scan by its site only, so the list is empty and every scan
# identifier is pseudonymised.  `stage` checks the list against the paper (check_paper), identifier -> where printed.
PRINTED_BY_THE_PAPER = OrderedDict()
KEEP_MEAS = {k.rsplit("/", 1)[1] for k in PRINTED_BY_THE_PAPER if "/" in k}
KEEP_KNEES = {k for k in PRINTED_BY_THE_PAPER if k.startswith("PF")}
KEEP_PARTICIPANTS = {k.split("_")[0] for k in KEEP_KNEES}
KEEP_BASES = set()
# synthetic file bases of the unit tests (not scanner files)
SYNTHETIC_BASES = {"X0000000"}
PATELLA_STUDY = "PF" + "J"
# the parts of a radius / tibia record id (written so that this file's own text does not match the rules below)
_LB = r"(?:(?<![A-Za-z0-9])|(?<=\\[ntr]))"           # a word start, also right after a backslash-n escape
_REG = r"(?:Dista[l]|Diaphysea[l])"
_STU = r"(?:CK[D]|REPR[O]|BMA[T])"
_IDSEP = r"[\"'\s/_\\,]{1,6}"                         # detection (verify, the paper): any short separator
_IDSEP_RW = r"(?:\s*[/_\\]{1,2}\s*|[\"']\s*,\s*[\"']|\s+)"  # rewriting: a path, a tag, a quoted tuple or spaces
_LISTSEP = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*/\s*)"
_TAIL = r"(?:[/\\:,;)\]'\"](?= ))? *"                # what follows an identifier: one mark, then spaces
RX_RECORD = re.compile(rf"{_LB}({_REG})({_IDSEP_RW})({_STU})({_IDSEP_RW})(\d{{4}})(?!\d)((?:{_LISTSEP}\d{{4}}(?!\d))*)({_TAIL})")
RX_RECORD_DETECT = re.compile(rf"{_LB}({_REG})({_IDSEP})({_STU})({_IDSEP})(\d{{4}})(?!\d)((?:{_LISTSEP}\d{{4}}(?!\d))*)")
RX_STUDY_FIRST = re.compile(rf"{_LB}({_STU})([_/])({_REG})([_/])(\d{{4}})(?!\d)({_TAIL})")
RX_RERUN = re.compile(rf"{_LB}(reru[n])({_IDSEP_RW})(\d{{4}})(?!\d)({_TAIL})")
RX_KNEE = re.compile(_LB + r"PF[J](\d{3})((?:_[LR])?)(?!\d)((?:/\d{3}_[LR](?![A-Za-z0-9]))*)(" + _TAIL + ")")
# a patella code in lower case, as in a Python name (test_pf<j><nnn>_..., a fixture called pf<j><nnn>)
RX_KNEE_LOWER = re.compile(_LB + r"pf[j]_?(\d{3})(?![0-9A-Fa-f])")
RX_BASE = re.compile(_LB + r"([A-Za-z])(\d{7})(?!\d)")
# a measurement number delimited by an underscore, as in a file name (EVAL_..._<nnnn>.LOG) or a Python name
RX_UNDERSCORED = re.compile(r"(?<=_)(\d{4})(?!\d)(?![A-Za-z])")
RX_LISTITEM = re.compile(rf"({_LISTSEP})(\d{{4}})")
RX_SHORTKNEE = re.compile(r"/(\d{3})(_[LR])")
# bare measurement numbers in prose (comments, docstrings and strings of code; documentation lines; JSON text values)
RX_BARE = re.compile(r"(?:(?<![\w.,/])|(?<=\\[ntr]))(?<!\d-)(\d{4})(?![\w/]|-\d|[.,]\d)")
RX_UNIT_AFTER = re.compile(r"\s*(?:voxels?\b|vox\b|mm\b|\u00b5m|um\b|%|slices?\b|components?\b|bytes?\b|ms\b|s\b|"
                           r"\u00d7|x\b|px\b|pixels?\b|times\b)")
RX_KEYWORD_BEFORE = re.compile(r"(?:measurements?|meas\.?|--measurements|--subjects|scans?)\s*$", re.I)
RX_LIST_GAP = re.compile(r"^(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*/\s*|\s*;\s*|\s+)$")
RX_JSON_STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')
RX_JSON_MEAS_KEY = re.compile(r'"(?:index_measurement|measurement|meas_no|fu_meas_no|meas)"\s*:\s*"?(\d{4})"?(?!\d)')
YEARS = range(1990, 2036)            # a bare number in this range may be a year: rewritten only inside a list


class PolicyError(RuntimeError):
    pass


def _hmac(key, ns, code):
    return hmac.new(key, f"{ns}|{code}".encode("utf-8"), hashlib.sha256).hexdigest()


def pseudonym(key, study, code):
    return "%s-%s" % (study, _hmac(key, study, code)[:6])


def measurement_pseudonym(key, number):
    """Six decimal digits (record ids stay <region>/<study>/<digits>, the form every script parses)."""
    return str(100000 + int(_hmac(key, "MEAS", str(number)), 16) % 900000)


def base_pseudonym(key, base):
    """'X' and seven digits (a file base stays <letter><7 digits>, the form the harness parses)."""
    return "X%d" % (1000000 + int(_hmac(key, "BASE", base.upper()), 16) % 9000000)


def _only(d, keep, drop, where):
    unknown = set(d) - keep - drop
    if unknown:
        raise PolicyError(f"{where}: unknown key(s) {sorted(unknown)} -- add them to the keep or drop list first")
    return OrderedDict((k, v) for k, v in d.items() if k in keep)


def deid_vd_record(r, key, where, pseud_seen):
    """A validate_dataset record (radius/tibia or patella porosity)."""
    out = _only(r, VD_TOP_KEEP, VD_TOP_DROP, where)
    meta = dict(out.get("meta") or {})
    name = meta.get("patient_name")
    if name is not None and "study_label" not in meta:
        m = STUDY_TOKEN.search(name)
        meta["study_label"] = m.group(1) if m else None
    out["meta"] = _only(meta, META_KEEP, META_DROP, where + " meta")
    if not out["meta"].get("study_label"):
        raise PolicyError(f"{where}: no study label (meta.patient_name has no '<Study>Study' token)")
    inputs = OrderedDict()
    for k, v in (out.get("inputs") or {}).items():
        inputs[k] = _only(v, INPUT_KEEP, INPUT_DROP, f"{where} inputs.{k}") if isinstance(v, dict) else v
    out["inputs"] = inputs
    out["duplicates"] = [(_only(d, DUP_KEEP, DUP_DROP, where + " duplicates[]") if isinstance(d, dict) else d)
                         for d in (out.get("duplicates") or [])]
    var = out.get("variants")
    if isinstance(var, dict) and isinstance(var.get("evidence"), dict):
        var = dict(var)
        var["evidence"] = OrderedDict((k, v) for k, v in var["evidence"].items() if k not in EVIDENCE_DROP)
        out["variants"] = var
    inv = out.get("inventory")
    if isinstance(inv, dict):
        idn = inv.get("identity") or {}
        new_id = OrderedDict((k, idn[k]) for k in IDENTITY_KEEP if k in idn)
        if "subject" in idn:
            study = idn.get("study") or "S"
            p = pseudonym(key, study, str(idn["subject"]))
            prev = pseud_seen.setdefault((study, p), str(idn["subject"]))
            if prev != str(idn["subject"]):
                raise PolicyError(f"pseudonym collision in study {study}: two participant codes map to {p}")
            new_id["subject"] = p
        new = OrderedDict(identity=new_id)
        if isinstance(inv.get("calibration"), dict):
            new["calibration"] = _only(inv["calibration"], {"slope", "intercept", "mu_scaling", "seg_gauss_native",
                                                            "seg_gauss_mgha"}, set(), where + " inventory.calibration")
        if isinstance(inv.get("plan"), dict) and "status" in inv["plan"]:
            new["plan"] = OrderedDict(status=inv["plan"]["status"])
        out["inventory"] = new
    elif inv is not None:
        raise PolicyError(f"{where}: inventory is not an object")
    return out


def deid_patella_b_record(r, where):
    out = _only(r, PATELLA_B_TOP, set(), where)
    if out.get("ipl_log") is not None:
        il = out["ipl_log"]
        if not (isinstance(il, dict) and set(il) <= {"thresholds"}):
            raise PolicyError(f"{where}: ipl_log carries more than thresholds; extend the policy first")
    return out


def deid_bvtv_a(doc):
    doc = json.loads(json.dumps(doc))
    meta = doc.get("meta") or {}
    meta.pop("data_root", None)
    if "printed_source" in meta:
        meta["printed_source"] = "IPL's evaluation result sheet of each patella (three decimals; not distributed)"
    recs = doc.get("records")
    for rr in (recs.values() if isinstance(recs, dict) else recs or []):
        pr = rr.get("printed") if isinstance(rr, dict) else None
        if isinstance(pr, dict):
            for k in ("sheet", "filename_on_sheet", "eval_date", "meas_date", "patient_name", "pat_no", "born"):
                pr.pop(k, None)
    return doc


# ------------------------------------------------------------------------------------------------ scan identifiers
def _kind(rel):
    low = rel.lower()
    for ext, k in ((".py", "py"), (".md", "md"), (".txt", "md"), (".rst", "md"), (".json", "json"), (".csv", "csv"),
                   (".tsv", "csv"), (".svg", "svg")):
        if low.endswith(ext):
            return k
    return "other"


def prose_spans(s, rel):
    """(start, end) of the prose of a text: comments and strings of Python code (tokenize), the lines of a document
    that are not table rows, the string values of JSON that contain a letter.  None when Python code does not
    tokenize (the caller fails closed)."""
    k = _kind(rel)
    if k == "py":
        starts, pos = [0], 0
        for line in io.StringIO(s):
            pos += len(line)
            starts.append(pos)
        want = {tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)}
        spans = []
        try:
            for t in tokenize.generate_tokens(io.StringIO(s).readline):
                if t.type in want:
                    spans.append((starts[t.start[0] - 1] + t.start[1], starts[t.end[0] - 1] + t.end[1]))
        except (tokenize.TokenError, SyntaxError, IndentationError):
            return None
        return spans
    if k == "md":
        spans, pos = [], 0
        for line in io.StringIO(s):
            if not line.lstrip().startswith("|"):
                spans.append((pos, pos + len(line)))
            pos += len(line)
        return spans
    if k == "json":
        return [(m.start(), m.end()) for m in RX_JSON_STRING.finditer(s) if re.search(r"[A-Za-z]", m.group(0))]
    return []


def _bare_tokens(text):
    """Bare four-digit numbers of a prose span, each with whether it stands in a list of measurement-like numbers."""
    toks = [(m.start(1), m.end(1), m.group(1)) for m in RX_BARE.finditer(text)]
    listed = [False] * len(toks)
    for i in range(len(toks) - 1):
        if RX_LIST_GAP.match(text[toks[i][1]:toks[i + 1][0]]):
            listed[i] = listed[i + 1] = True
    return toks, listed


class ScanIds:
    """The real scan identifiers of the internal records (memory only)."""

    def __init__(self):
        self.meas = {}          # measurement number -> study (radius / tibia)
        self.meas_patella = {}  # measurement number -> 'PFJ' (patellae; never in a record id)
        self.knees = set()      # PFJnnn_S
        self.bases = {}         # file base (upper case) -> study

    @property
    def participants(self):
        return {k.split("_")[0] for k in self.knees}

    def all_meas(self):
        return set(self.meas) | set(self.meas_patella)

    def hidden_meas(self):
        return self.all_meas() - KEEP_MEAS

    @classmethod
    def from_src(cls, src):
        ids = cls()
        R = os.path.join(src, "validation", "results")
        rec_rx = re.compile(rf"^{_REG}[/_]({_STU})[/_](\d{{4}})$")
        for p in glob.glob(os.path.join(R, "*", "records", "*.json")):
            try:
                r = json.load(io.open(p, encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(r, dict):
                continue
            rid = str(r.get("id") or r.get("subject") or os.path.basename(p)[:-5])
            m = rec_rx.match(rid)
            study = None
            if m:
                study = m.group(1)
                ids.meas[m.group(2)] = study
            elif re.match(r"^PF[J]\d{3}_[LR]$", rid):
                ids.knees.add(rid)
                study = PATELLA_STUDY
            else:
                m = re.match(r"^reru[n][/_](\d{4})$", rid)
                if m:
                    ids.meas.setdefault(m.group(1), "")
            idn = ((r.get("inventory") or {}).get("identity") or {}) if isinstance(r.get("inventory"), dict) else {}
            if isinstance(idn, dict) and str(idn.get("index_measurement") or "").isdigit() and study != PATELLA_STUDY:
                ids.meas.setdefault(str(idn["index_measurement"]), study or "")
            if isinstance(r.get("base"), str) and RX_BASE.fullmatch(r["base"].strip()):
                ids.bases.setdefault(r["base"].strip().upper(), study or "")
        for name in ("ipl_printed_values_137.csv", "ipl_printed_values.csv"):
            p = os.path.join(R, name)
            if not os.path.exists(p):
                continue
            for row in csv.DictReader(io.open(p, encoding="utf-8")):
                rid = (row.get("id") or "").strip()
                pat = bool(re.match(r"^PF[J]\d{3}_[LR]$", rid))
                if pat:
                    ids.knees.add(rid)
                for k in ("base", "filename"):
                    v = (row.get(k) or "").strip()
                    if RX_BASE.fullmatch(v):
                        ids.bases.setdefault(v.upper(), PATELLA_STUDY if pat else "")
                for k in ("meas_no", "fu_meas_no"):
                    v = (row.get(k) or "").strip()
                    if re.fullmatch(r"\d{4}", v):
                        if pat:
                            ids.meas_patella.setdefault(v, PATELLA_STUDY)
                        else:
                            ids.meas.setdefault(v, "")
        p = os.path.join(R, BVTV_A)
        if os.path.exists(p):
            recs = json.load(io.open(p, encoding="utf-8")).get("records") or {}
            for rr in (recs.values() if isinstance(recs, dict) else recs):
                rr = rr or {}
                if re.match(r"^PF[J]\d{3}_[LR]$", str(rr.get("subject") or "")):
                    ids.knees.add(rr["subject"])
                if isinstance(rr.get("base"), str) and RX_BASE.fullmatch(rr["base"]):
                    ids.bases.setdefault(rr["base"].upper(), PATELLA_STUDY)
                m = re.match(r"^(\d{4})Meas$", str((rr.get("printed") or {}).get("filename_on_sheet") or ""))
                if m:
                    ids.meas_patella.setdefault(m.group(1), PATELLA_STUDY)
        for n in list(ids.meas_patella):                      # a number of both kinds is a radius / tibia number
            if n in ids.meas:
                del ids.meas_patella[n]
        return ids

    # -------------------------------------------------------------------------------- verify: what is left in a text
    def hits(self, s, rel):
        """Scan identifiers that are not printed by the paper and are still in a text (for verify)."""
        out = []
        hm = self.hidden_meas()
        hp = self.participants - KEEP_PARTICIPANTS
        if not hasattr(self, "_lit_rx"):
            lits = sorted({*(k for k in self.knees if k.split("_")[0] in hp), *hp,
                           *self.bases, *(b.lower() for b in self.bases),
                           *("0000" + n for n in hm), *(n + "Meas" for n in hm)}, key=len, reverse=True)
            self._lit_rx = (re.compile(r"(?<![A-Za-z0-9])(?:%s)(?![0-9])" % "|".join(map(re.escape, lits)), re.I)
                            if lits else None)
        if self._lit_rx is not None:
            for m in self._lit_rx.finditer(s):
                out.append(("dynamic_scan_identifier", "<internal scan identifier, %d chars>" % len(m.group(0))))
        for m in RX_UNDERSCORED.finditer(s):
            if m.group(1) in hm:
                out.append(("dynamic_underscored_measurement_number", "<measurement number after '_'>"))
        for m in RX_KNEE.finditer(s):
            for sm in RX_SHORTKNEE.finditer(m.group(3) or ""):
                if PATELLA_STUDY + sm.group(1) in hp:
                    out.append(("dynamic_scan_identifier", "<short patella code>"))
        for m in RX_JSON_MEAS_KEY.finditer(s):
            if m.group(1) in hm:
                out.append(("dynamic_measurement_field", "<measurement number>"))
        spans = prose_spans(s, rel)
        if spans is None:
            out.append(("untokenizable_code", rel))
            spans = []
        for a, b in spans:
            text = s[a:b]
            toks, _ = _bare_tokens(text)
            for t0, t1, n in toks:
                if n in hm and not RX_UNIT_AFTER.match(text, t1):
                    out.append(("dynamic_bare_measurement_number", "<measurement number in prose>"))
        return out


class Pseudonyms:
    """The keyed pseudonyms of the scan identifiers, and their rewriting of texts, records and file names."""

    def __init__(self, key, ids):
        self.ids = ids
        self.meas = {n: measurement_pseudonym(key, n) for n in sorted(ids.hidden_meas())}
        self.part = {p: pseudonym(key, PATELLA_STUDY, p) for p in sorted(ids.participants - KEEP_PARTICIPANTS)}
        self.base = {b: base_pseudonym(key, b) for b in sorted(set(ids.bases) - KEEP_BASES)}
        for what, mp in (("measurement", self.meas), ("patella participant", self.part), ("file base", self.base)):
            if len(set(mp.values())) != len(mp):
                raise PolicyError(f"pseudonym collision among the {what} pseudonyms: extend the pseudonym length")
        if set(self.meas.values()) & ids.all_meas() or set(self.base.values()) & set(ids.bases):
            raise PolicyError("a pseudonym equals a real identifier")
        self.all_meas = ids.all_meas()
        self.counts = Counter()             # occurrences rewritten per class
        self.seen = {"measurement": set(), "patella": set(), "base": set()}

    def _m(self, n):
        if n in self.meas:
            self.counts["measurement"] += 1
            self.seen["measurement"].add(n)
            return self.meas[n]
        return n

    def _list(self, tail):
        """', <nnnn> and <nnnn>' after a record id: the other numbers of a list of measurements."""
        out, pos = [], 0
        for m in RX_LISTITEM.finditer(tail):
            if m.start() != pos or m.group(2) not in self.all_meas:
                break
            out.append(m.group(1) + self._m(m.group(2)))
            pos = m.end()
        return "".join(out) + tail[pos:]

    @staticmethod
    def _aligned(old, new, tail):
        """A longer pseudonym takes its extra width from the run of spaces after it (and after one punctuation mark;
        two spaces stay), so that a column that was aligned after the identifier stays aligned."""
        mark = tail.rstrip(" ")
        spaces = tail[len(mark):]
        d = len(new) - len(old)
        return new + mark + (spaces[d:] if 0 < d and len(spaces) >= d + 2 else spaces)

    def _record(self, m):
        old = m.group(0)[:len(m.group(0)) - len(m.group(7))]
        new = m.group(1) + m.group(2) + m.group(3) + m.group(4) + self._m(m.group(5)) + self._list(m.group(6) or "")
        return self._aligned(old, new, m.group(7))

    def _knee(self, m):
        old = m.group(0)[:len(m.group(0)) - len(m.group(4))]
        return self._aligned(old, self._knee_core(m), m.group(4))

    def _knee_core(self, m):
        p = PATELLA_STUDY + m.group(1)
        head = m.group(0)[:len(m.group(0)) - len(m.group(3) or "") - len(m.group(4))]
        if p in self.part:
            self.counts["patella"] += 1
            self.seen["patella"].add(p)
            head = self.part[p] + (m.group(2) or "")

        def short(sm):
            q = PATELLA_STUDY + sm.group(1)
            if q in self.part:
                self.counts["patella"] += 1
                self.seen["patella"].add(q)
                return "/" + self.part[q] + sm.group(2)
            return sm.group(0)
        return head + RX_SHORTKNEE.sub(short, m.group(3) or "")

    def _base(self, m):
        b = (m.group(1) + m.group(2)).upper()
        if b not in self.base:
            return m.group(0)
        self.counts["base"] += 1
        self.seen["base"].add(b)
        p = self.base[b]
        return p.lower() if m.group(1).islower() else p

    def anchored(self, s):
        s = RX_RECORD.sub(self._record, s)
        s = RX_STUDY_FIRST.sub(lambda m: self._aligned(m.group(0)[:m.start(6) - m.start()], m.group(1) + m.group(2)
                                                      + m.group(3) + m.group(4) + self._m(m.group(5)), m.group(6)), s)
        s = RX_RERUN.sub(lambda m: self._aligned(m.group(0)[:m.start(4) - m.start()],
                                                m.group(1) + m.group(2) + self._m(m.group(3)), m.group(4)), s)
        s = RX_KNEE.sub(self._knee, s)
        s = RX_KNEE_LOWER.sub(self._knee_lower, s)
        s = RX_BASE.sub(self._base, s)
        s = RX_UNDERSCORED.sub(lambda m: self._m(m.group(1)), s)
        return s

    def _knee_lower(self, m):
        """pf<j><nnn> in lower case (a Python name) -> 'pfj_' and the participant's six hex digits."""
        p = PATELLA_STUDY + m.group(1)
        if p not in self.part:
            return m.group(0)
        self.counts["patella"] += 1
        self.seen["patella"].add(p)
        return self.part[p].replace("-", "_").lower()

    def prose(self, s, rel):
        spans = prose_spans(s, rel)
        if not spans:
            return s
        parts, last = [], 0
        for a, b in spans:
            text = s[a:b]
            toks, listed = _bare_tokens(text)
            new, pos = [], 0
            for (t0, t1, n), li in zip(toks, listed):
                if n not in self.meas or RX_UNIT_AFTER.match(text, t1):
                    continue
                if int(n) in YEARS and not li and not RX_KEYWORD_BEFORE.search(text[:t0]):
                    continue                                       # could be a year: left for verify to flag
                p = self._m(n)
                sp = len(text[t1:]) - len(text[t1:].lstrip(" "))
                d = len(p) - len(n)
                new.append(text[pos:t0] + p)
                pos = t1 + (d if 0 < d and sp >= d + 2 else 0)
            parts.append(s[last:a] + "".join(new) + text[pos:])
            last = b
        return "".join(parts) + s[last:]

    def text(self, s, rel):
        if _kind(rel) == "svg":                                    # never inside embedded images
            out, last = [], 0
            for m in B64.finditer(s):
                out.append(self.anchored(s[last:m.start()]) + m.group(0))
                last = m.end()
            return "".join(out) + self.anchored(s[last:])
        if _kind(rel) == "py":
            s = self._py_names(s)
        s = self.anchored(s)
        if _kind(rel) in ("py", "md", "json"):
            s = self.prose(s, rel)
        return s

    def _py_names(self, s):
        """A patella code inside a Python name (e.g. a PF<J><nnn>_DIR constant) takes its pseudonym with '_' for '-'."""
        if not re.search(r"PF[J]\d{3}", s):
            return s
        starts, pos = [0], 0
        for line in io.StringIO(s):
            pos += len(line)
            starts.append(pos)
        edits = []
        try:
            for t in tokenize.generate_tokens(io.StringIO(s).readline):
                if t.type == tokenize.NAME and re.search(r"PF[J]\d{3}", t.string):
                    a = starts[t.start[0] - 1] + t.start[1]
                    edits.append((a, a + len(t.string), re.sub(r"PF[J](\d{3})", self._part_name, t.string)))
        except (tokenize.TokenError, SyntaxError, IndentationError):
            return s
        for a, b, new in reversed(edits):
            s = s[:a] + new + s[b:]
        return s

    def _part_name(self, m):
        p = PATELLA_STUDY + m.group(1)
        if p not in self.part:
            return m.group(0)
        self.counts["patella"] += 1
        self.seen["patella"].add(p)
        return self.part[p].replace("-", "_")

    def name(self, fname):
        return self.anchored(fname)

    def obj(self, o, key=None):
        """A JSON object: every key and string value; identity.index_measurement-like fields by value."""
        if isinstance(o, dict):
            return OrderedDict((self.obj(k) if isinstance(k, str) else k, self.obj(v, k)) for k, v in o.items())
        if isinstance(o, list):
            return [self.obj(v) for v in o]
        if key in ("index_measurement", "measurement", "meas_no", "fu_meas_no") and str(o) in self.meas:
            v = self._m(str(o))
            return int(v) if isinstance(o, int) else v
        if isinstance(o, str):
            s = self.anchored(o)
            if not re.search(r"[A-Za-z]", s):
                return s
            return json.loads(self.prose(json.dumps(s, ensure_ascii=False), "value.json"))
        return o

    def tree(self, root, skip=(), dry=False):
        """Rewrite every text file under root (except the paths in skip and this tool) and rename every file whose
        name carries a scan identifier.  Returns (files changed, files renamed)."""
        changed, renamed = [], []
        skip = {os.path.normcase(os.path.abspath(p)) for p in skip}
        for dp, dn, fn in os.walk(root):
            dn[:] = sorted(d for d in dn if d not in SKIP_DIRS and not d.endswith(".egg-info"))
            for f in sorted(fn):
                p = os.path.join(dp, f)
                rel = os.path.relpath(p, root).replace("\\", "/")
                if rel == SELF or os.path.normcase(os.path.abspath(p)) in skip:
                    continue
                low = f.lower()
                if low.endswith(TEXT_EXT) or low in (".gitignore", "license", "manifest.in"):
                    raw = open(p, "rb").read()
                    try:
                        s = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        raise PolicyError(f"{rel}: not UTF-8 text")
                    s2 = self.text(s, rel)
                    if s2 != s:
                        changed.append(rel)
                        if not dry:
                            with open(p, "wb") as fh:
                                fh.write(s2.encode("utf-8"))
                g = self.name(f)
                if g != f:
                    q = os.path.join(dp, g)
                    if os.path.exists(q):
                        raise PolicyError(f"{rel}: cannot rename to {g}, which exists")
                    renamed.append(rel)
                    if not dry:
                        os.replace(p, q)
        return changed, renamed

    def mapping_rows(self, pseud_seen):
        rows = [(study, p, code, "participant") for (study, p), code in sorted(pseud_seen.items())]
        rows += [(PATELLA_STUDY, p, code, "participant") for code, p in sorted(self.part.items())]
        rows += [(self.ids.meas.get(n) or self.ids.meas_patella.get(n) or "", p, n, "measurement")
                 for n, p in sorted(self.meas.items())]
        rows += [(self.ids.bases.get(b) or "", p, b, "file_base") for b, p in sorted(self.base.items())]
        return rows


def check_mapping(path, rows):
    """An existing map must be reproduced exactly (same key): no pseudonym may change between stagings."""
    if not path or not os.path.exists(path):
        return 0
    new = {(r[3], r[2]): r[1] for r in rows}
    new_part = {(r[0], r[2]): r[1] for r in rows if r[3] == "participant"}
    bad, n = [], 0
    for row in csv.DictReader(io.open(path, encoding="utf-8")):
        kind = row.get("kind") or "participant"
        code = row.get("real_code") or row.get("participant_code") or ""
        got = new_part.get((row.get("study"), code)) if kind == "participant" else new.get((kind, code))
        n += 1
        if got is not None and got != row.get("pseudonym"):
            bad.append(row.get("pseudonym"))
    if bad:
        raise PolicyError(f"{len(bad)} pseudonym(s) of the existing map would change (a different key?): stopped")
    return n


def _svg_drawn_text(s):
    s = B64.sub("", s)
    parts = re.findall(r"<!--(.*?)-->", s, re.S)
    parts += [re.sub(r"<[^>]+>", "", t) for t in re.findall(r"<text\b[^>]*>(.*?)</text>", s, re.S)]
    return "\n".join(parts)


def check_paper(src, ids):
    """The scan identifiers printed by the paper (<src>/manuscript/MANUSCRIPT.md, SUPPLEMENT.md and the drawn text of
    the SVG figures) must be exactly PRINTED_BY_THE_PAPER.  Returns {identifier: [where]}, or None without a paper."""
    ms = os.path.join(src, "manuscript")
    texts = [(f, os.path.join(ms, f)) for f in ("MANUSCRIPT.md", "SUPPLEMENT.md")]
    if not all(os.path.exists(p) for _, p in texts):
        return None
    texts = [(f, io.open(p, encoding="utf-8").read()) for f, p in texts]
    for p in sorted(glob.glob(os.path.join(ms, "figures", "*.svg")) + glob.glob(os.path.join(ms, "figures", "supp", "*.svg"))):
        texts.append((os.path.relpath(p, ms).replace("\\", "/"), _svg_drawn_text(io.open(p, encoding="utf-8").read())))
    printed = OrderedDict()
    for where, t in texts:
        for m in RX_RECORD_DETECT.finditer(t):
            nums = [m.group(5)] + [x.group(2) for x in RX_LISTITEM.finditer(m.group(6) or "")]
            for n in nums:
                if n in ids.all_meas():
                    rid = f"{m.group(1)}/{m.group(3)}/{n}" if n == m.group(5) else n
                    printed.setdefault(("meas", n), []).append(f"{where}: {rid}")
        for m in RX_KNEE.finditer(t):
            p = PATELLA_STUDY + m.group(1)
            knees = [p + (m.group(2) or "")] + [p[:-3] + x.group(1) + x.group(2) for x in RX_SHORTKNEE.finditer(m.group(3) or "")]
            for k in knees:
                if k[:6] in ids.participants:                 # a participant is kept with every knee of it
                    printed.setdefault(("patella", k[:6]), []).append(f"{where}: {k}")
        for m in RX_BASE.finditer(t):
            if (m.group(1) + m.group(2)).upper() in ids.bases:
                printed.setdefault(("base", (m.group(1) + m.group(2)).upper()), []).append(where)
    keep = {("meas", n) for n in KEEP_MEAS} | {("patella", p) for p in KEEP_PARTICIPANTS} | {("base", b) for b in KEEP_BASES}
    extra, stale = set(printed) - keep, keep - set(printed)
    if extra or stale:
        raise PolicyError("PRINTED_BY_THE_PAPER disagrees with the paper: printed but not kept %s; kept but not printed %s"
                          % (sorted("%s %s" % k for k in extra), sorted("%s %s" % k for k in stale)))
    return printed


# ------------------------------------------------------------------------------------------------ the scanner
def _rx(p, flags=0):
    return re.compile(p, flags)


# Patterns are written so that their own source text does not match them (e.g. "dk[0]:\["), so this file can be
# scanned.  Terms that are not identifiers are not listed here (see --denylist and PRIVATE_TEXT below).
STATIC = OrderedDict([
    ("vms_date", _rx(r"\b\d{1,2}[-\u2212](?:JA[N]|FE[B]|MA[R]|AP[R]|MA[Y]|JU[N]|JU[L]|AU[G]|SE[P]|OC[T]|NO[V]|DE[C])[-\u2212]\d{4}\b", re.I)),
    ("iso_datetime_s", _rx(r"\b(?:19|20)\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d")),
    ("vms_disk_path", _rx(r"\bdk[0]:\[", re.I)),
    ("scanner_data_dir", _rx(r"xct[2]\.data", re.I)),
    ("scanner_account", _rx(r"User:\s*XC[T]2")),
    ("vms_logical_UE", _rx(r"\bU[E]:[A-Z_]")),
    ("versioned_vms_file", _rx(r"\.(?:CO[M]|GOB[J]|AI[M]|IS[Q]|PD[F]|LO[G]|DA[T]|TX[T])\s*;\s*\d+", re.I)),
    ("eval_log_patient_index", _rx(r"EVA[L]_\w*_\d{8}_\d{8}\.LOG", re.I)),
    ("vendor_com_file", _rx(r"\b[A-Z0-9_$]+\.CO[M]\b")),
    ("vendor_batch", _rx(r"\$ IPL_BATC[H]")),
    ("vendor_eval_job", _rx(r"UCT_EVALUATIO[N]")),
    ("vendor_script_name", _rx(r"IPLV6_[A-Z]")),
    ("abs_path_windows", _rx(r"(?<![A-Za-z0-9])[C-Zc-z]:[/\\]{1,2}(?:Research|Users|Program|anaconda|Temp)\b", re.I)),
    ("home_path", _rx(r"(?:Users|home)[/\\]{1,2}[A-Za-z][\w.-]*[/\\]", re.I)),
    ("patient_name_field", _rx(r"\b[A-Za-z]+;\s*[A-Za-z]+[Ss]tud[y]\b")),
    ("study_code", _rx(r"\b[A-Za-z]+[Ss]tud[y] [A-Z0-9]+_\d{3}\b")),
    ("participant_code", _rx(r"\b(?:CK[D]|CD[K]|REPR[O0]|MA[T]|BMA[T]|XCTPF[J])_?\d{3}\b")),
    ("participant_code_pfj", _rx(r"XCTPF[J]")),
    ("scanco_sample_name", _rx(r"\b[A-Z]{3,}_[_][A-Z0-9]+")),
    # scan identifiers in their raw form (the ones the paper prints would be exempt: PRINTED_BY_THE_PAPER, empty)
    ("raw_patella_code", _rx(_LB + r"(?:PF[J]\s?\d{2,3}(?:_[LR])?(?!\d)|PF[J]_\d{3}(?![0-9A-Fa-f]))", re.I)),
    ("raw_record_id", _rx(rf"{_LB}{_REG}{_IDSEP}{_STU}{_IDSEP}\d{{4}}(?!\d)")),
    ("raw_record_id_study_first", _rx(rf"{_LB}{_STU}[_/]{_REG}[_/]\d{{4}}(?!\d)")),
    ("raw_rerun_folder", _rx(rf"{_LB}reru[n]{_IDSEP}\d{{4}}(?!\d)")),
    ("raw_scanco_file_base", _rx(_LB + r"[A-Za-z]0\d{6}(?!\d)")),
    ("file_transfer", _rx(r"\bsft[p]\b", re.I)),
    ("secret_assignment", _rx(r"(?i)\b(?:api[_-]?key|secre[t]|passwor[d]|passw[d]|acces[s]_token)\b\s*[:=]\s*['\"][^'\"\s]{8,}")),
    ("github_token", _rx(r"\bgh[pousr]_[A-Za-z0-9]{30,}")),
    ("aws_key", _rx(r"\bAKI[A][0-9A-Z]{16}\b")),
    ("private_key", _rx(r"-----BEGIN [A-Z ]*PRIVATE KE[Y]-----")),
])


# ------------------------------------------------------------------------------------------------ private terms
PRIVATE_TEXT = []        # compiled patterns searched in text and file names (from --denylist; empty without it)
PRIVATE_BYTES = []       # the `any` ones as bytes patterns: every byte of every file, and embedded images
PRIVATE_B64 = []         # the `any` ones again, searched in the base64 text of embedded images


def load_denylist(path):
    """Read the optional private pattern file (see the module docstring); returns the number of patterns."""
    del PRIVATE_TEXT[:], PRIVATE_BYTES[:], PRIVATE_B64[:]
    if not path:
        return 0
    if os.path.normcase(os.path.abspath(path)).startswith(os.path.normcase(REPO) + os.sep):
        raise SystemExit("the --denylist file must live outside the public tree")
    for n, line in enumerate(io.open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scope, _, pat = line.partition(" ")
        pat = pat.strip()
        if scope not in ("text", "any") or not pat:
            raise SystemExit(f"{path}:{n}: expected '<text|any> <regular expression>'")
        PRIVATE_TEXT.append(re.compile(pat, re.I))
        if scope == "any":
            PRIVATE_BYTES.append(re.compile(pat.encode("utf-8"), re.I))
            PRIVATE_B64.append(re.compile(pat, re.I))
    return len(PRIVATE_TEXT)


def _kept(rule, text):
    """The scan identifiers the paper prints, and the synthetic file bases of the tests, are not hits."""
    t = re.sub(r"\s", "", text)
    if rule == "raw_patella_code":
        return t in KEEP_KNEES or t in KEEP_PARTICIPANTS
    if rule in ("raw_record_id", "raw_record_id_study_first", "raw_rerun_folder"):
        return t[-4:] in KEEP_MEAS
    if rule == "raw_scanco_file_base":
        return t.upper() in SYNTHETIC_BASES or t.upper() in KEEP_BASES
    return False


# In data and documents (not in code, which may name a header field in order to discard it).
DATA_ONLY = OrderedDict([
    ("patient_field_label", _rx(r"patient_nam[e]|Patient Nam[e]|\bpat_n[o]\b|\bPat-N[o]|\bBor[n]\b|Meas-Dat[e]|Eval-Dat[e]|"
                                r"Index Patien[t]|Original Creation-Dat[e]|Process I[D]\b|\bAge:")),
    ("internal_folder_ref", _rx(r"data/ipl_prob[e]s")),
])
# Per-file exceptions, each with its reason (the rule still runs everywhere else).  None since 2026-09-26: the two
# staging scripts build the scanner's versioned file names in code instead of spelling a version out.
ALLOW = {}
# Synthetic VMS dates are allowed only in the parser tests, and only in the year 2000.
SYNTHETIC_DATE_FILES = {"tests/test_validate_dataset_variants.py": "2000"}
WORDING = _rx(r"\breplic(?:a|as|ated|ation)\b|\bbit[- ]exact\b|\bbit for bi[t]\b", re.I)
WORDING_OK = _rx(r"edge\s+replication|face slice replicated|replicate[ds]? (?:the )?(?:face|edge|border)|"
                 r"reproduces the run bit for bit|self-consistency", re.I)
EMAIL = _rx(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+")
EMAIL_OK = set()                    # no personal address is published (GitHub noreply addresses pass, see scan_text)
TEXT_EXT = (".py", ".md", ".json", ".csv", ".svg", ".txt", ".toml", ".cfg", ".cff", ".yml", ".yaml", ".ini", ".html",
            ".gitignore", ".in", ".sh", ".bat", ".tsv")
CODE_EXT = (".py", ".sh", ".bat")
BAD_EXT = (".aim", ".isq", ".gobj", ".rsq", ".nii", ".nii.gz", ".nrrd", ".npz", ".npy", ".zip", ".pdf", ".docx", ".doc",
           ".ps", ".com", ".log", ".h5", ".hdf5", ".mat", ".dcm", ".tif", ".tiff", ".mha", ".mhd", ".raw", ".pkl",
           ".xlsx", ".ipynb")
BINARY_MARKERS = [b"Patient Nam" + b"e", b"AIMDAT" + b"A", b"dk0" + b":[", b"DK0" + b":[", b"Index Patien" + b"t"]
B64 = re.compile(r"data:image/[a-z+]+;base64,([A-Za-z0-9+/=\s]+)")
MAX_BYTES = 50 * 1024 * 1024
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".eggs"}


TIME_OF_DAY = re.compile(r"(?<![\d:.])(\d{1,2}:\d{2}:\d{2}\.\d{2,3})(?!\d)")


def _times_of_day(obj, out):
    """Every 'hh:mm:ss.cc' time of day inside a record (scan, AIM, contour-file and evaluation-log times): a time of
    day to the hundredth of a second is as identifying as the date it came with, so each one is a literal too."""
    if isinstance(obj, dict):
        for v in obj.values():
            _times_of_day(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _times_of_day(v, out)
    elif isinstance(obj, str):
        out.update(TIME_OF_DAY.findall(obj))


def dynamic_denylist(src):
    """Literal identifiers taken from the INTERNAL records and tables (memory only; never written anywhere)."""
    lit = set()
    times = set()
    R = os.path.join(src, "validation", "results")
    for p in glob.glob(os.path.join(R, "*", "records", "*.json")):
        try:
            r = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(r, dict):
            continue
        _times_of_day(r, times)
        meta = r.get("meta") or {}
        if isinstance(meta, dict) and meta.get("patient_name"):
            lit.add(meta["patient_name"].strip())
        inv = r.get("inventory") or {}
        idn = inv.get("identity") if isinstance(inv, dict) else None
        if isinstance(idn, dict):
            for k in ("patient_name", "subject", "subject_raw", "scan_date", "aim_time", "isq"):
                v = idn.get(k)
                if isinstance(v, str) and len(v.strip()) >= 5:
                    lit.add(v.strip())
            if idn.get("pi"):
                lit.add(str(idn["pi"]).strip() + ";")
            if idn.get("index_patient"):
                lit.add("IDX:%08d" % int(idn["index_patient"]))
    for p in (os.path.join(R, "ipl_printed_values_137.csv"), os.path.join(R, "ipl_printed_values.csv")):
        if not os.path.exists(p):
            continue
        for row in csv.DictReader(io.open(p, encoding="utf-8")):
            _times_of_day(dict(row), times)
            base = (row.get("base") or "").strip()
            for k in ("patient_name", "born", "meas_date", "eval_date", "fu_eval_date"):
                v = (row.get(k) or "").strip()
                if len(v) >= 6:
                    lit.add(v)
                    lit.add(v.replace("\u2212", "-"))
            v = (row.get("pat_no") or "").strip()
            if v.isdigit():
                lit.add("IDX:%08d" % int(v))
            v = (row.get("filename") or "").strip()
            if len(v) >= 6 and v != base:
                lit.add(v)
    p = os.path.join(R, BVTV_A)
    if os.path.exists(p):
        doc = json.load(io.open(p, encoding="utf-8"))
        recs = doc.get("records") or {}
        _times_of_day(doc, times)
        for rr in (recs.values() if isinstance(recs, dict) else recs):
            v = ((rr or {}).get("printed") or {}).get("filename_on_sheet")
            if v:
                lit.add(str(v))
    for n in list(lit):
        if n.endswith(";") and "; " not in n:
            continue
        m = STUDY_TOKEN.search(n) if ";" in n else None
        if m:
            lit.add(n.split(";")[0].strip() + ";")
    idx = {x[4:] for x in lit if x.startswith("IDX:")}
    lit = {x for x in lit if not x.startswith("IDX:")} | times
    return lit, idx


def scan_text(s, rel, dyn=None, idx=None, ids=None):
    hits = []
    code = rel.lower().endswith(CODE_EXT)
    pats = list(STATIC.items()) + ([] if code else list(DATA_ONLY.items()))
    for name, rx in pats:
        for m in rx.finditer(s):
            if not _kept(name, m.group(0)):
                hits.append((name, m.group(0)))
    for m in EMAIL.finditer(s):
        if m.group(0) not in EMAIL_OK and not m.group(0).endswith(("@users.noreply.github.com",)):
            hits.append(("email", m.group(0)))
    for m in WORDING.finditer(s):
        ctx = s[max(0, m.start() - 40): m.end() + 40]
        if not WORDING_OK.search(ctx):
            hits.append(("wording", m.group(0)))
    for rx in PRIVATE_TEXT:
        for m in rx.finditer(s):
            hits.append(("private_term", "<private term, %d chars>" % len(m.group(0))))
    if dyn:
        for lit in dyn:
            if lit in s:
                hits.append(("dynamic_literal", "<internal identifier, %d chars>" % len(lit)))
    if idx:
        for i in idx:
            if re.search(r"(?<![\d.])%s(?!\d)" % re.escape(i), s):
                hits.append(("dynamic_patient_index", "<8-digit index>"))
    if ids is not None:
        hits += ids.hits(s, rel)
    return hits


def _png_texts(raw):
    """The text chunks of a PNG (tEXt, zTXt, iTXt): the only place a PNG carries words."""
    out, i = [], 8
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        return ""
    while i + 8 <= len(raw):
        n = int.from_bytes(raw[i:i + 4], "big")
        t, d = raw[i + 4:i + 8], raw[i + 8:i + 8 + n]
        try:
            if t == b"tEXt":
                out.append(d.replace(b"\0", b": ").decode("latin-1"))
            elif t == b"zTXt":
                k, _, rest = d.partition(b"\0")
                out.append(k.decode("latin-1") + ": " + zlib.decompress(rest[1:]).decode("latin-1", "replace"))
            elif t == b"iTXt":
                k, _, rest = d.partition(b"\0")
                flag, rest = rest[0], rest[2:]
                _, _, rest = rest.partition(b"\0")
                _, _, rest = rest.partition(b"\0")
                out.append(k.decode("latin-1") + ": " + (zlib.decompress(rest) if flag else rest).decode("utf-8", "replace"))
        except Exception:
            out.append("<unreadable %s chunk>" % t.decode("latin-1"))
        if t == b"IEND":
            break
        i += 12 + n
    return "\n".join(out)


def scan_tree(root, dyn=None, idx=None, allow=None, ids=None):
    """Return {relative path: [(rule, match), ...]} for the whole tree (text and binary checks)."""
    allow = ALLOW if allow is None else allow
    report = OrderedDict()
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, root).replace("\\", "/")
            low = f.lower()
            hits = []
            size = os.path.getsize(p)
            if size > MAX_BYTES:
                hits.append(("file_over_50MB", str(size)))
            if low.endswith(BAD_EXT):
                hits.append(("forbidden_file_type", f))
            # the file name itself (a record is named after its id)
            hits += [(n, "file name: " + m) for n, m in scan_text(f, "name.txt") if n.startswith(("raw_", "private_term"))]
            if low.endswith(TEXT_EXT) or low in (".gitignore", "license", "manifest.in"):
                s = io.open(p, encoding="utf-8", errors="replace").read()
                if low.endswith(".svg"):
                    for m in B64.finditer(s):
                        img = base64.b64decode(re.sub(r"\s", "", m.group(1)))
                        for mk in BINARY_MARKERS:
                            if mk in img[:65536]:
                                hits.append(("svg_raster_header", mk.decode()))
                        img_low = img.lower()
                        for rx in PRIVATE_BYTES:
                            if rx.search(img_low):
                                hits.append(("private_term_bytes", "<private term in an embedded image>"))
                        pt = _png_texts(img)
                        if pt:
                            hits += [h for h in scan_text(pt, rel + ".png-text.txt", dyn, idx, ids) if h[0] != "wording"]
                        # the base64 text itself is random, but a term spelled in it is still a match for a text search
                        for rx in PRIVATE_B64:
                            for nm in rx.finditer(m.group(1)):
                                hits.append(("private_term_in_base64", "<%d letters in an embedded image's base64>" % len(nm.group(0))))
                    s = B64.sub("data:image/png;base64,", s)
                hits += scan_text(s, rel, dyn, idx, ids)
            else:
                raw = open(p, "rb").read(MAX_BYTES)
                head = raw[:65536]
                for mk in BINARY_MARKERS:
                    if mk in head:
                        hits.append(("binary_header", mk.decode()))
                low_raw = raw.lower()
                for rx in PRIVATE_BYTES:
                    if rx.search(low_raw):
                        hits.append(("private_term_bytes", "<private term in the file's bytes>"))
                pt = _png_texts(raw)
                if pt:
                    hits += [h for h in scan_text(pt, rel + ".png-text.txt", dyn, idx, ids) if h[0] != "wording"]
            ok = allow.get(rel, set())
            hits = [h for h in hits if h[0] not in ok]
            if rel in SYNTHETIC_DATE_FILES:
                yr = SYNTHETIC_DATE_FILES[rel]
                hits = [h for h in hits if not (h[0] == "vms_date" and h[1].endswith(yr))]
            if hits:
                report[rel] = hits
    return report


# ------------------------------------------------------------------------------------------------ helpers
def jload(p):
    return json.load(io.open(p, encoding="utf-8"), object_pairs_hook=OrderedDict)


def jdump(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.write("\n")


def check_values(obj, where, dyn, idx, ids=None):
    s = json.dumps(obj, ensure_ascii=False, indent=1)
    hits = scan_text(s, where + ".json", dyn, idx, ids)
    hits = [h for h in hits if h[0] != "wording"]
    if hits:
        raise PolicyError(f"{where}: identifier(s) left after de-identification: {sorted({h[0] for h in hits})}")


def write_result_sets(dst, auto, noedit, dry):
    p = os.path.join(dst, "validation", "result_sets.py")
    txt = ('"""The two radius/tibia result sets under validation/results/ that the paper\'s numbers are built from.\n\n'
           "Written by tools/deidentify_results.py; every script under validation/ and manuscript/ imports these names\n"
           'instead of spelling the directories out."""\n'
           f'OSLH_AUTO = "{auto}"       # 63 radius/tibia measurements (62 counted), each at IPL\'s automatic evaluation run\n'
           f'OSLH_NOEDIT = "{noedit}"   # 54 diaphyseal radius/tibia measurements, each at IPL\'s automatic evaluation run\n')
    if not dry:
        with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(txt)
    return p


# ------------------------------------------------------------------------------------------------ stage
def stage(a):
    src = os.path.abspath(a.src)
    dst = os.path.abspath(a.dst)
    SR = os.path.join(src, "validation", "results")
    DR = os.path.join(dst, "validation", "results")
    if os.path.normcase(dst).startswith(os.path.normcase(src) + os.sep) or os.path.normcase(dst) == os.path.normcase(src):
        raise SystemExit("--dst must not be inside --src")
    if not os.path.isdir(SR):
        raise SystemExit(f"no validation/results under --src ({src})")
    key_path = os.path.abspath(a.key)
    if os.path.normcase(key_path).startswith(os.path.normcase(dst) + os.sep):
        raise SystemExit("the key file must live outside the public tree")
    if a.init_key and not os.path.exists(key_path):
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        with open(key_path, "w") as fh:
            fh.write(secrets.token_hex(32) + "\n")
        print(f"created a new pseudonym key at {key_path}: keep it private and keep it (pseudonyms depend on it)")
    if not os.path.exists(key_path):
        raise SystemExit(f"key file {key_path} not found (pass --init-key once to create it)")
    if a.mapping:
        mp = os.path.abspath(a.mapping)
        if os.path.normcase(mp).startswith(os.path.normcase(dst) + os.sep):
            raise SystemExit("--mapping must be written outside the public tree")
    key = open(key_path).read().strip().encode()
    dyn, idx = dynamic_denylist(src)
    print(f"dynamic denylist: {len(dyn)} literal identifiers and {len(idx)} patient indices from the internal records (memory only)")
    ids = ScanIds.from_src(src)
    printed = check_paper(src, ids)
    if printed is None:
        print("  (no paper under --src/manuscript: PRINTED_BY_THE_PAPER not re-checked against it)")
    else:
        print(f"the paper prints {len(printed)} scan identifier(s), exactly the kept ones"
              + (": " + "; ".join(f"{k[1]} ({len(v)} place(s))" for k, v in printed.items()) if printed else ""))
    ps = Pseudonyms(key, ids)
    print(f"scan identifiers: {len(ids.meas)} radius/tibia measurement numbers ({len(ids.meas) - len(set(ids.meas) - KEEP_MEAS)} "
          f"printed by the paper and kept), {len(ids.meas_patella)} patella measurement numbers, {len(ids.knees)} patella "
          f"codes of {len(ids.participants)} participants ({len(ids.participants & KEEP_PARTICIPANTS)} kept), "
          f"{len(ids.bases)} file bases")
    for name in (a.oslh_auto, a.oslh_noedit):
        if not OSLH_SET_RX.match(name) or not os.path.isdir(os.path.join(SR, name, "records")):
            raise SystemExit(f"result set {name} not found under {SR}")
    plan, pseud_seen, studies = [], {}, OrderedDict()
    n_written = Counter()
    staged = []                                                            # (path, object or text)

    # validate_dataset records
    for name in (a.oslh_auto, a.oslh_noedit, PATELLA_VD_SET):
        files = sorted(glob.glob(os.path.join(SR, name, "records", "*.json")))
        for p in files:
            f = os.path.basename(p)
            if not RECORD_NAME_RX.match(f):
                raise PolicyError(f"{name}/records/{f}: unexpected record file name")
            r = jload(p)
            real_id = r.get("id")
            out = ps.obj(deid_vd_record(r, key, f"{name}/{f}", pseud_seen))
            check_values(out, f"{name}/{f}", dyn, idx, ids)
            staged.append((os.path.join(DR, name, "records", ps.name(f)), out))
            studies[out["id"]] = dict(id=out["id"], tag=out.get("tag", ""), set=name, real=real_id,
                                      site=(out["meta"].get("bone_key") or out["meta"].get("bone") or ""),
                                      study=out["meta"]["study_label"].upper())
            n_written[name] += 1
    # patella configuration B
    for name in PATELLA_B_SETS:
        for p in sorted(glob.glob(os.path.join(SR, name, "records", "*.json"))):
            f = os.path.basename(p)
            if not RECORD_NAME_RX.match(f):
                raise PolicyError(f"{name}/records/{f}: unexpected record file name")
            out = ps.obj(deid_patella_b_record(jload(p), f"{name}/{f}"))
            check_values(out, f"{name}/{f}", dyn, idx, ids)
            staged.append((os.path.join(DR, name, "records", ps.name(f)), out))
            n_written[name] += 1
    # plain copies (free of personal identifiers; the scan identifiers are pseudonymised; scanned anyway)
    for rel in COPY_FILES:
        p = os.path.join(SR, rel)
        if not os.path.exists(p):
            print(f"  (not present, skipped: {rel})")
            continue
        txt = ps.text(io.open(p, encoding="utf-8").read(), rel)
        hits = [h for h in scan_text(txt, rel, dyn, idx, ids) if h[0] != "wording"]
        if hits:
            raise PolicyError(f"{rel}: identifier(s): {sorted({h[0] for h in hits})}")
        staged.append((os.path.join(DR, rel), txt))
        n_written["copied"] += 1
    # patella BV/TV (configuration A)
    out = ps.obj(deid_bvtv_a(jload(os.path.join(SR, BVTV_A))))
    check_values(out, BVTV_A, dyn, idx, ids)
    staged.append((os.path.join(DR, BVTV_A), out))
    # IPL's printed values: reduced columns only
    rows = list(csv.DictReader(io.open(os.path.join(SR, PRINTED_IN), encoding="utf-8")))
    missing = [c for c in PRINTED_COLS if c not in rows[0]]
    if missing:
        raise PolicyError(f"{PRINTED_IN}: columns {missing} missing")
    buf = io.StringIO()
    w = csv.DictWriter(buf, PRINTED_COLS, lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow({c: row[c] for c in PRINTED_COLS})
    txt = ps.text(buf.getvalue(), PRINTED_IN)
    hits = [h for h in scan_text(txt, PRINTED_IN, dyn, idx, ids) if h[0] != "wording"]
    if hits:
        raise PolicyError(f"{PRINTED_IN} (reduced): identifier(s): {sorted({h[0] for h in hits})}")
    staged.append((os.path.join(DR, PRINTED_IN), txt))
    # study of every scan (from the records' study token; the patellae from their porosity records); an optional
    # table of the source repository (--study-table: id, study, cohort) cross-checks the labels and gives the cohort
    xc = os.path.abspath(a.study_table) if getattr(a, "study_table", None) else None
    if xc and not os.path.exists(xc):
        raise SystemExit(f"--study-table {xc} not found")
    if xc:
        tab = {r["id"]: r for r in csv.DictReader(io.open(xc, encoding="utf-8"))}
        bad = [s["real"] for s in studies.values() if s["real"] in tab and tab[s["real"]]["study"].upper() != s["study"]]
        if bad:
            raise PolicyError(f"study labels disagree with the internal study table for {len(bad)} scan(s)")
        cohort = {i: tab[s["real"]]["cohort"] for i, s in studies.items() if s["real"] in tab}
    else:
        cohort = {}
    buf = io.StringIO()
    w = csv.DictWriter(buf, ["id", "tag", "cohort", "site", "study"], lineterminator="\n")
    w.writeheader()
    for i in sorted(studies, key=lambda t: (studies[t]["set"] != PATELLA_VD_SET, t)):
        s = studies[i]
        w.writerow(dict(id=i, tag=s["tag"], cohort=cohort.get(i, ""), site=s["site"], study=s["study"]))
    staged.append((os.path.join(DR, STUDIES_OUT), buf.getvalue()))

    print("de-identified: " + ", ".join(f"{k} {v}" for k, v in n_written.items()) +
          f"; {len(pseud_seen)} participant pseudonyms; {len(studies)} scans in {STUDIES_OUT}")
    rows_map = ps.mapping_rows(pseud_seen)
    n_old = check_mapping(os.path.abspath(a.mapping) if a.mapping else None, rows_map)
    if n_old:
        print(f"the existing pseudonym map is reproduced: {n_old} row(s), none changed")
    if a.dry_run:
        ch, rn = ps.tree(dst, dry=True)
        print(f"--dry-run: nothing written (would write {len(staged)} files under {DR}, rewrite {len(ch)} other "
              f"file(s) and rename {len(rn)})")
        return 0
    # remove result sets of earlier stagings that are not the current ones, then write
    if os.path.isdir(DR):
        for d in os.listdir(DR):
            if OSLH_SET_RX.match(d) and d not in (a.oslh_auto, a.oslh_noedit):
                shutil.rmtree(os.path.join(DR, d))
                print(f"removed the superseded staged set {d}")
        for name in (a.oslh_auto, a.oslh_noedit, PATELLA_VD_SET, *PATELLA_B_SETS):
            rd = os.path.join(DR, name, "records")
            if os.path.isdir(rd):
                shutil.rmtree(rd)
    for p, obj in staged:
        if isinstance(obj, str):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(obj)
        else:
            jdump(obj, p)
    rs = write_result_sets(dst, a.oslh_auto, a.oslh_noedit, False)
    print(f"wrote {len(staged)} files under {DR} and {os.path.relpath(rs, dst)}")
    ch, rn = ps.tree(dst, skip=[p for p, _ in staged])
    print(f"scan pseudonyms carried through the rest of the tree: {len(ch)} file(s) rewritten, {len(rn)} renamed")
    print("scan identifiers pseudonymised: %d measurement numbers, %d patella participants, %d file bases "
          "(occurrences rewritten: %s)" % (len(ps.seen["measurement"]), len(ps.seen["patella"]), len(ps.seen["base"]),
                                           dict(ps.counts)))
    if a.mapping:
        with io.open(mp, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["study", "pseudonym", "real_code", "kind"])
            for row in rows_map:
                w.writerow(row)
        print(f"pseudonym map (PRIVATE) -> {mp}: {Counter(r[3] for r in rows_map)}")
    return 0


# ------------------------------------------------------------------------------------------------ verify
def verify(a):
    root = os.path.abspath(a.root)
    dyn = idx = ids = None
    if a.src:
        dyn, idx = dynamic_denylist(os.path.abspath(a.src))
        ids = ScanIds.from_src(os.path.abspath(a.src))
        print(f"dynamic denylist: {len(dyn)} literal identifiers, {len(idx)} patient indices, scan identifiers not printed "
              f"by the paper: {len(ids.hidden_meas())} measurement numbers, {len(ids.participants - KEEP_PARTICIPANTS)} "
              f"patella participants, {len(set(ids.bases) - KEEP_BASES)} file bases (memory only)")
    rep = scan_tree(root, dyn, idx, ids=ids)
    tot = Counter()
    for rel, hits in rep.items():
        c = Counter(h[0] for h in hits)
        tot.update(c)
        shown = sorted({h[1] for h in hits if not h[0].startswith("dynamic")})[:4]
        print(f"  {rel}: {dict(c)}  {shown if a.show else ''}")
    print(f"verify {root}: {sum(tot.values())} hit(s) in {len(rep)} file(s) {dict(tot)}")
    return 1 if rep else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="write de-identified results into the public tree")
    s.add_argument("--src", required=True, help="the internal repository (read only)")
    s.add_argument("--dst", default=REPO, help="the public tree (default: this repository)")
    s.add_argument("--key", required=True, help="pseudonym key file, kept OUTSIDE the public tree")
    s.add_argument("--init-key", action="store_true", help="create the key file if it does not exist")
    s.add_argument("--oslh-auto", required=True, help="the first radius/tibia result set (63 records), e.g. oslh_auto_v4")
    s.add_argument("--oslh-noedit", required=True, help="the second radius/tibia result set (54 records), e.g. oslh_noedit_v3")
    s.add_argument("--mapping", default=None, help="write the pseudonym map here (PRIVATE; outside the tree)")
    s.add_argument("--study-table", default=None,
                   help="optional CSV of the source repository (id, study, cohort) that cross-checks the study labels "
                        "and fills the cohort column of cohort_studies.csv")
    s.add_argument("--denylist", default=os.environ.get("IPLDT_DENYLIST"),
                   help="optional private pattern file of further terms (outside the tree; see the module docstring)")
    s.add_argument("--dry-run", action="store_true")
    v = sub.add_parser("verify", help="scan a tree for identifiers, vendor material and local paths")
    v.add_argument("--root", default=REPO)
    v.add_argument("--src", default=None, help="internal repository: add its literal identifiers to the scan")
    v.add_argument("--show", action="store_true", help="print the matched text of static rules")
    v.add_argument("--denylist", default=os.environ.get("IPLDT_DENYLIST"),
                   help="optional private pattern file of further terms (outside the tree; see the module docstring)")
    a = ap.parse_args(argv)
    n_private = load_denylist(a.denylist)
    if n_private:
        print(f"private terms: {n_private} pattern(s) from the deny-list (not printed)")
    try:
        return stage(a) if a.cmd == "stage" else verify(a)
    except PolicyError as e:
        print("POLICY STOP:", e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
