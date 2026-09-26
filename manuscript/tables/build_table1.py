#!/usr/bin/env python
"""Table 1 of the paper (the 137 scans of the validation set) as CSV and Markdown.

    python manuscript/tables/build_table1.py          (from the repository root, after manuscript/facts/build_facts.py)

Every cell is read from manuscript/facts/facts.json, tables["cohort.table1"], which build_facts.py computes from the
validation records (study token, participant pseudonym, stack length and STEP-1 parameter set of every scan).  The
script checks that each row's study, stack-length and parameter-set counts add up to its n and that the site rows add
up to the total row, and stops otherwise.  Writes manuscript/tables/table1_cohort.csv and table1_cohort.md.
"""
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FACTS = os.path.join(REPO, "manuscript", "facts", "facts.json")
OUT = os.path.join(HERE, "table1_cohort")

# the study token of the scan header -> the study, as Section 2.2 of the paper names it
STUDY_NAMES = {"PFJ": "patellofemoral osteoarthritis study", "CKD": "chronic kidney disease study",
               "REPRO": "reproducibility study", "BMAT": "BMAT study"}
COLUMNS = ["Site", "n", "Study (scans)", "Stack length (slices)", "IPL parameter set", "Participants"]


def counts(d, capital=False):
    """{'168': 19, '320': 1} -> '168 (19 scans), 320 (1)', most frequent first; a single entry -> its label alone."""
    items = sorted(d.items(), key=lambda kv: (-kv[1], str(kv[0])))
    labels = [str(k) for k, _ in items]
    if capital:
        labels[0] = labels[0][:1].upper() + labels[0][1:]
    if len(items) == 1:
        return labels[0]
    return ", ".join(f"{lab} ({n}{' scans' if i == 0 else ''})" for i, (lab, (_, n)) in enumerate(zip(labels, items)))


def main():
    facts = json.load(io.open(FACTS, encoding="utf-8"))
    rows = facts["tables"]["cohort.table1"]
    total = next(r for r in rows if r["key"] == "total")
    sites = [r for r in rows if r["key"] != "total"]
    for r in rows:
        for k in ("studies", "stack_lengths", "preset"):
            if sum(r[k].values()) != r["n"]:
                raise SystemExit(f"Table 1: {r['site']}: {k} add up to {sum(r[k].values())}, not n = {r['n']}")
    if sum(r["n"] for r in sites) != total["n"]:
        raise SystemExit("Table 1: the site rows' n do not add up to the total row")
    for k in ("studies", "stack_lengths", "preset"):          # participants are not additive over sites
        keys = set(total[k]) | {x for r in sites for x in r[k]}
        if any(sum(r[k].get(x, 0) for r in sites) != total[k].get(x, 0) for x in keys):
            raise SystemExit(f"Table 1: the site rows' {k} do not add up to the total row")
    out = []
    for r in rows:
        if r["key"] == "total":        # worded as the paper's Total row: "Patellofemoral osteoarthritis 21; ..."
            studies = [f"{STUDY_NAMES.get(k, k).removesuffix(' study')} {v}" for k, v in r["studies"].items()]
        else:
            studies = [f"{STUDY_NAMES.get(k, k)} ({v})" for k, v in r["studies"].items()]
        studies[0] = studies[0][:1].upper() + studies[0][1:]
        out.append({"Site": r["site"], "n": r["n"], "Study (scans)": "; ".join(studies),
                    "Stack length (slices)": counts(r["stack_lengths"]), "IPL parameter set": counts(r["preset"], capital=True),
                    "Participants": r["participants"]})
    with io.open(OUT + ".csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    L = [f"**Table 1. The {total['n']} XtremeCT II scans of the validation set.**", "",
         "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    L += ["| " + " | ".join(str(r[c]) for c in COLUMNS) + " |" for r in out]
    L += ["", "Read from `manuscript/facts/facts.json` (`tables[\"cohort.table1\"]`), which build_facts.py computes from the "
              "validation records. Participant counts are per study; the total adds them up, assuming that nobody was "
              "enrolled in two of the studies.", ""]
    with io.open(OUT + ".md", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    print(f"wrote {os.path.relpath(OUT, REPO).replace(os.sep, '/')}.csv and .md ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
