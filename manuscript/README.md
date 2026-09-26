# The paper's figures, tables and numbers

This folder holds the scripts that produced every figure, table and number of the paper, and their outputs. The
text of the paper and of its Supplement is not included (the paper is under review; the reference is added on
publication).

- `facts/`: `build_facts.py` computes every number of the main text and Table 1 from the validation records into
  `facts.json` (full precision, one key per number, with its source) and `FACTS.md` (rounded, readable);
  `make_facts_bmd_ctpo.py` does the same for the cortical-porosity module (`FACTS_BMD_CTPO.json` / `.md`; the file
  name is historical: the BMD validation is not part of the paper or of this repository).
- `figures/`: one script per figure (`fig1_workflow.py` ... `fig6_porosity.py`, `supp/S1_seggauss.py` ...
  `supp/S9_porosity.py`), the 300-dpi PNG and the SVG of each, and sidecars (`*_numbers.json`, `*.csv`) with every
  number the figure draws or prints and its check against `facts.json`.
- `tables/`: Tables 1, 2, 3 and S1 as CSV and Markdown (`table1_cohort.*` is written by `tables/build_table1.py` from
  `tables["cohort.table1"]` of `facts/facts.json`, `table2_modules.*` and `table3_metrics.*` by
  `figures/fig5_agreement.py`, `tableS1_parameters.*` by `tables/build_tableS1.py`, which carries the table as printed in the Supplement and stops
  if a value no longer matches the code). The last column of Tables 2 and 3,
  `facts key`, is not part of the published tables: it names the key of `facts/facts.json` (or of
  `facts/FACTS_BMD_CTPO.json`) each row is read from.

## What can be regenerated from this repository

`python tools/regenerate_paper_numbers.py` (repository root) runs, from the de-identified records in
`validation/results/` only:

| Output | Script | Inputs |
|---|---|---|
| `validation/results/porosity_AB_summary_n137.json` | `validation/porosity_AB_stats.py --set 137` | the three record sets with a porosity block |
| `facts/facts.json`, `facts/FACTS.md` (all main-text numbers, Table 1) | `facts/build_facts.py` | `records.json`, `sample_means.csv`, `from_ipl_contour_ceil_dt/`, `from_ipl_contour_ceil/`, `patella_bvtv_A/`, the two radius / tibia sets, `cohort_studies.csv`, `pooled_137_comparison.json` |
| `validation/results/pooled_137_comparison.json` | `figures/pooled_statistics.py` | the records, through `fig5_agreement.py`'s loader (the independent cross-check of `build_facts.py`) |
| Table 1 | `tables/build_table1.py` | `facts.json` |
| `facts/FACTS_BMD_CTPO.json`, `.md` (pore map, Ct.Po; the pore-map rows of Table 2 and the Ct.Po rows of Table 3) | `facts/make_facts_bmd_ctpo.py` | the porosity records, `porosity_AB_summary_n137.json`, `ipl_printed_values_137.csv`, `facts.json` |
| Figure 5, Tables 2 and 3 | `figures/fig5_agreement.py` | the records, `facts.json`, `FACTS_BMD_CTPO.json` |
| Table S1 (as printed in the Supplement; every value checked against the code) | `tables/build_tableS1.py` | the ipldt / ORMIR-BQRL source, `facts.json` |

In the validation environment (Python 3.11.15, NumPy 2.3.5, SciPy 1.15.3, numba 0.66.0) the regenerated numbers,
tables and Figure 5 sidecars are identical to the published ones, apart from the build date printed in `FACTS.md`,
`facts.json` and `figures/fig5_agreement_numbers.json` (the day of the run; set `SOURCE_DATE_EPOCH` to fix it). Other library versions can move the last
digits of floating-point statistics: with Python 3.12.7, NumPy 1.26.4 and SciPy 1.13.1 on Windows every count and
Tables 1, 2, 3 and S1 come out identical, and regression coefficients differ in their last digits (slopes by at most 3e-15
relative, intercepts by at most 1.2e-15 absolute). Voxel totals are summed in 64-bit integers
on every platform.

## What needs data that are not public

The other figures show slices of the scans and IPL's products, or IPL's exports of probe runs on the scanner. Their
scripts are included as the record of how each panel was made; they stop with a Python FileNotFoundError naming
the missing path when the data are absent (set `IPLDT_LAB_ROOT`, see `validation/datapaths.py`). They, and the `slow`
tests, name the scans by their pseudonyms (the de-identification rewrote the code as well), so they find the
laboratory's data only when it is laid out under those names. Their image caches
(`figures/cache/`, crops of patient scans) are not included.

| Figure | Non-public inputs | Panels drawn from the records alone |
|---|---|---|
| 1 workflow | PFJ-0be66a_R greyscale, IPL's masks, SEG and Tb.Th map; the ORMIR-XCT periosteal mask of that scan (`IPLDT_FIG1_PRX_MASK`) | -- |
| 2 STEP 1 | IPL's stage exports of PFJ-0be66a_R, its greyscale | I |
| 3 Laplace-Hamming | PFJ-0be66a_R greyscale and IPL's STEP-2 exports; `--verify` re-reads every greyscale | C |
| 4 distance transforms | PFJ-0be66a_R SEG, TRAB_SEG, TRAB_MASK and IPL's Tb.Th / Tb.Sp maps | -- |
| 6 porosity | IPL's products of Diaphyseal/CKD/2422 and Distal/REPRO/2095 | C-F |
| S1 /seg_gauss | PFJ-0be66a_R greyscale and IPL's stage exports | the calibration panel (header values in the records) |
| S2 morphology | IPL's STEP-1 exports and probe exports (synthetic blocks, patella volumes) | -- |
| S3 STEP-1 presets | greyscale and IPL's renderings of one ultradistal tibia and one ultradistal radius | -- |
| S4 contours | IPL's masks, renderings and contour files of the 21 patellae; the probe-19 phantom exports (`IPLDT_PROBE19_MIRROR`) | -- (D-G are synthetic but read the phantom exports) |
| S5 Laplace-Hamming padding | PFJ-0be66a_R and IPL's STEP-2 exports; the probe-21 impulse-phantom exports | -- |
| S6 distance-transform parameters | PFJ-0be66a_R and IPL's parameter-sweep exports | -- |
| S7 Tb.Sp and Tb.N | PFJ-0be66a_R and IPL's maps | -- |
| S8 derivation example | the probe-21 predictions and IPL's exports | -- |
| S9 porosity | IPL's products of two radius / tibia scans (the patella label census of the legend is kept in `S9_porosity_numbers.json`, `patella_labels`) | J |

`figures/fig3_lh_verify_oslh.json` and `fig3_lh_verify_patella.json` list, for the voxels on which ipldt's and IPL's
Laplace-Hamming segmentations differ, the position of each and ipldt's normalised Laplace-Hamming value there (the
short compared with the threshold, and its offset from it): sparse verification evidence (a few thousand isolated
voxels over 137 scans), not images.

The file paths recorded in the sidecars are written relative to the repository or as `<IPLDT_LAB_ROOT>/...` for the
non-public data.
