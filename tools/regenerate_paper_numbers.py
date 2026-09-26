#!/usr/bin/env python
"""Regenerate the paper's numbers, Tables 1, 2, 3 and S1 and Figure 5 from the published validation records.

    python tools/regenerate_paper_numbers.py            (from the repository root)

Runs, in order, with this repository's ipldt on the path and no non-public data:

  1 validation/porosity_AB_stats.py --set 137       validation/results/porosity_AB_summary_n137.json
  2 manuscript/facts/build_facts.py                  manuscript/facts/facts.json + FACTS.md
  3 manuscript/figures/pooled_statistics.py          validation/results/pooled_137_comparison.json
  4 manuscript/facts/build_facts.py (again)          the pooled numbers cross-checked against step 3
  5 manuscript/tables/build_table1.py                Table 1 (from facts.json)
  6 manuscript/facts/make_facts_bmd_ctpo.py          manuscript/facts/FACTS_BMD_CTPO.json + .md (cortical porosity)
  7 manuscript/figures/fig5_agreement.py             Figure 5 (PNG, SVG, sidecars) and Tables 2 and 3
  8 manuscript/tables/build_tableS1.py               Table S1 (the printed table, every value checked against the code)

Every script stops on an internal inconsistency.  In the validation environment (Python 3.11, NumPy 2.3, SciPy 1.15)
the numbers are identical to the published ones; other library versions can move the last digits of a few
floating-point statistics (see manuscript/README.md).  The build date printed in FACTS.md is the day of the run (set
SOURCE_DATE_EPOCH to fix it).  Exit status 0 when every step succeeded.
"""
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = [("porosity_AB_stats", "validation/porosity_AB_stats.py", ["--set", "137"]),
         ("build_facts", "manuscript/facts/build_facts.py", []),
         ("pooled_statistics", "manuscript/figures/pooled_statistics.py", []),
         ("build_facts (cross-check)", "manuscript/facts/build_facts.py", []),
         ("build_table1", "manuscript/tables/build_table1.py", []),
         ("make_facts_bmd_ctpo", "manuscript/facts/make_facts_bmd_ctpo.py", []),
         ("fig5_agreement", "manuscript/figures/fig5_agreement.py", []),
         ("build_tableS1", "manuscript/tables/build_tableS1.py", [])]


def main():
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env["PYTHONPATH"] = REPO + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log_dir = os.environ.get("IPLDT_REGEN_LOGS")
    for name, rel, args in STEPS:
        t = time.time()
        out = open(os.path.join(log_dir, name.split(" ")[0] + (".2" if "(" in name else "") + ".log"), "w", encoding="utf-8") if log_dir else None
        rc = subprocess.call([sys.executable, os.path.join(REPO, rel), *args], cwd=REPO, env=env,
                             stdout=out, stderr=subprocess.STDOUT if out else None)
        if out:
            out.close()
        print(f"{name}: {'ok' if rc == 0 else 'FAILED (exit %d)' % rc} ({time.time() - t:.0f} s)", flush=True)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
