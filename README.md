# ipldt and ORMIR-BQRL

**ipldt** is an open-source Python reimplementation of the bone morphometry evaluation that Scanco's Image Processing
Language (IPL V5.42) runs on XtremeCT II HR-pQCT scans: the Laplace-Hamming variant of the manufacturer's standard
evaluation script, from the periosteal contour to the trabecular and cortical metrics. **ORMIR-BQRL** is the workflow
built on it: one command from a Scanco AIM to masks, segmentation, the cortical pore map, distance-transform maps and
a report, with manual correction of the masks in 3D Slicer and re-entry from the corrected masks.

| Stage | What ipldt reimplements |
|---|---|
| AIM input / output | Scanco AIM reading (v020; v030 for IPL's float images) and v020 writing, calibration from the processing log (`ipldt.io`) |
| Periosteal contour | taken from ORMIR-XCT's autocontour, or supplied (e.g. IPL's own contour) |
| Compartment separation | Script 32 / 33 STEP 1: `/seg_gauss`, the chamfer morphology, the connected-component commands, `/gobj_maskaimpeel_ow` (`ipldt.step1`, `ipldt.ipl_ops`) |
| Contour rendering | `/togobj_from_aim -curvature_smooth 1` + `/gobj_to_aim` (`ipldt.contour`) |
| Segmentation | `/fft_laplace_hamming`, normalisation and threshold, assembled into IPL's SEG (`ipldt.ormir`) |
| Cortical pores | the pore-extraction cascade of Burghardt et al. (Bone 2010) with `/hysteresis_threshold` (`ipldt.porosity`) |
| Distance transforms | `/dt_thickness`, `/dt_spacing`, `/dt_number`: IPL's sphere fitting (Hildebrand and Rüegsegger 1997), CPU (numba) or GPU (CuPy) with identical results (`ipldt.core`) |
| Metrics | BV/TV, Tb.Th, Tb.Sp, Tb.N and Ct.Th (`ipldt.ormir`, `ormir_bqrl`); Ct.Po from the pore map (`ipldt.porosity`); Tb.BMD and Ct.BMD through ORMIR-XCT |

Two command-line entry points run the whole chain on a scan. They differ in what else they offer, and in one detail
of the segmentation: `ormir-bqrl` masks the Laplace-Hamming segmentation with the rendered periosteal contour, as IPL's
evaluation does, and `ipldt-pipeline` with the autocontour's raw raster (the parameter `seg.periosteal_mask`, see
"Parameters"), so their SEG and cortical segmentation (and what is computed from them) can differ by a few voxels;
they agree exactly when rendering leaves the periosteal raster unchanged.

| | `ormir-bqrl run` (the workflow) | `ipldt-pipeline` |
|---|---|---|
| Periosteal contour | ORMIR-XCT autocontour, or a mask given with `--periosteal` | ORMIR-XCT autocontour, or a mask given with `--periosteal` |
| Compartments, SEG, Tb.Th / Tb.Sp / Tb.N / Ct.Th maps, BV/TV, BMD | yes | yes |
| Cortical pore map and Ct.Po | yes (`--no-porosity` skips them) | yes (`--no-porosity` skips them) |
| Masks, SEG, pore map and maps written as | NIfTI, or AIM with `--map-format aim` | NIfTI, or AIM with `--map-format aim` |
| Also written | report (JSON, CSV, Markdown), the parameter set, the greyscale in HU, the 3D Slicer segmentation, a preview | report (JSON, CSV), the parameter set |
| Parameters (`--params FILE`, `--set STAGE.NAME=VALUE`) | every stage | every stage |
| Correction in 3D Slicer and re-entry (`ormir-bqrl redo`) | yes | no |

Both run the cortical pore cascade on the grid IPL runs it on: IPL's render grid of the cortical contour (the grid
IPL's `/gobj_to_aim` renders the contour onto), which `ipldt.porosity.render_grid` predicts from the rendered contour
alone. Per in-plane axis the grid runs from the contour's lowest coordinate minus 2, clipped at 0, to the largest
over the contour's slices of (the slice's highest coordinate, plus 1 when the slice's extent is even) plus 2; in z it
spans the contour's slices; where it reaches past the input AIM it is zero-padded. The cascade runs there on the
rendered cortical contour and the cortical segmentation, and the pore map is pasted back onto the input AIM's grid
(the reports record the grids). The grid matters because the cascade's slice-wise steps measure every component
against the non-bone voxels of its slice in the grid they run on. On the 137 validation scans the predicted grid is
IPL's own render grid on 137 of 137, and with IPL's configuration-A inputs (IPL's rendered cortical contour and
cortical segmentation) run through the workflows' code the pore map is identical to IPL's on all 116 radius and tibia
scans and all 21 patellae; run on the plain grid of the input AIM instead, as the code did before this rule, it
differed from IPL's on 4 of the 137 (`ormir_bqrl/README.md`, "The pore map's grid").

## How it was derived

ipldt was built without IPL's source code or binaries; nothing in IPL was decompiled or disassembled. Its rules come
from the published algorithms, IPL's command documentation and per-command help, the evaluation scripts the
manufacturer supplies with the scanner (they fix which commands run, in which order and with which parameters), the
processing log IPL writes into each output file, and the intermediate products IPL can export. Each candidate rule
was then checked against IPL treated as a black box, on synthetic phantoms designed so that competing rules give
different answers and on real scans with a single option changed per run. The expected output was written down
before every scanner run, and a rule was kept only when IPL's export matched it exactly. The paper and its Supplement
describe the procedure, with one worked example.

## Agreement with IPL

The products of every stage were compared with IPL's own on 137 XtremeCT II scans (21 patellae and 29 each of the
ultradistal radius, ultradistal tibia, diaphyseal radius and diaphyseal tibia), by counting the voxels on which they
differ, in two configurations: A gives ipldt IPL's own segmentation and contours and tests the distance-transform and
pore stages; B gives ipldt only IPL's periosteal contour and runs the whole chain. The results are reported in the
paper. Every number of the main text, Tables 1-3 and Figure 5 is regenerated from the per-scan validation records
in this repository (`manuscript/facts/FACTS.md` and `facts.json` list them with their sources), and Supplementary
Table S1 is checked against the code; the numbers printed in the legends of the image figures are recorded in the
figures' `*_numbers.json` sidecars (see "Regenerating the paper's numbers").

## Installation

Python 3.10 or later; NumPy, SciPy and numba are the only required dependencies.

```bash
pip install "ipldt @ git+https://github.com/zhuyihua1234/ipldt"                 # the engine
pip install "ipldt[bqrl] @ git+https://github.com/zhuyihua1234/ipldt"           # + ORMIR-BQRL (ITK, SimpleITK, ORMIR-XCT)
pip install "ipldt[bqrl,gpu] @ git+https://github.com/zhuyihua1234/ipldt"       # + CuPy for the GPU back end
```

or, from a clone, `pip install -e ".[bqrl]"`. Extras: `io` (ITK with its Scanco IO module, SimpleITK), `ormir`
(ORMIR-XCT, which needs Python 3.11 or later), `bqrl` (io + ormir + matplotlib), `gpu` (CuPy for CUDA 12), `test`
(pytest) and `paper` (the validation and figure scripts). The validation ran with Python 3.11.15, NumPy 2.3.5,
SciPy 1.15.3, numba 0.66.0, CuPy 14.2.0, ITK 5.4.6, SimpleITK 2.5.5 and ORMIR-XCT 1.1.0.

## Quick start

The whole workflow on one scan:

```bash
ormir-bqrl run SCAN.AIM out/ --site tibia          # masks, SEG, pore map, Tb.Th / Tb.Sp / Tb.N / Ct.Th maps, Ct.Po, BMD,
                                                   # report, Slicer files (NIfTI; --map-format aim for AIMs)
ormir-bqrl redo SCAN.AIM out/ --trab out/SCAN_compartments.seg.nrrd   # re-enter after correcting the trabecular mask in 3D Slicer
ormir-bqrl batch results/ a.AIM b.AIM c.AIM        # several scans, with batch_summary.csv
ormir-bqrl params --describe                       # every parameter, its default and the IPL option it corresponds to
ormir-bqrl --help
ipldt-pipeline SCAN.AIM out/ --site tibia --map-format aim   # the same chain, without the Slicer files and re-entry, as AIMs
```

The single IPL commands:

```bash
ipldt-dt-thickness TRAB_SEG.AIM TRAB_TH.nii.gz --gobj TRAB_MASK.AIM   # /dt_thickness with the trabecular contour
ipldt-dt-spacing   SEG.AIM TRAB_SP.nii.gz --gobj TRAB_MASK.AIM
ipldt-dt-number    SEG.AIM TRAB_1N.nii.gz --gobj TRAB_MASK.AIM
ipldt-ipl-morphometry --help                                          # all maps and IPL's statistics from IPL's files
```

These reproduce IPL when given IPL's own files, which IPL stores on the bounding box of the segmentation. The distance
transforms count the faces of the grid as object, so a map depends on the box it is computed on: `ormir-bqrl` and
`ipldt-pipeline` compute their maps on that box but write SEG and TRAB_SEG on the input AIM's grid, so to recompute
their maps from those files (Tb.N shows the difference most), cut the file to the bounding box of its set voxels first
(`ipldt.ipl_ops.bounding_box_cut`).

From Python, on IPL's exports of one scan:

```python
from ipldt import read_aim, align_to, render_volume, dt_thickness, dt_spacing, cort_trab_separation, TIBIA

seg = read_aim("SEG.AIM")                  # IPL's segmentation: cortical 127 + trabecular 126
trab_seg = read_aim("TRAB_SEG.AIM")        # its trabecular part, cropped to the trabecular contour (SEG grid)
mask = read_aim("TRAB_MASK.AIM")           # the raw trabecular mask, stored on its own grid
# the contour as IPL renders it: the mask rendered on its own grid, placed on the SEG grid by global position
G = align_to(dict(data=render_volume(mask["data"] > 0).astype("uint8"), dim=mask["dim"], pos=mask["pos"]),
             seg["dim"], seg["pos"]) > 0
th = dt_thickness(trab_seg["data"] > 0, gobj={"rendered": G}, voxel_size_mm=0.0607)
print(th.report["Th_mm"])                  # Tb.Th as IPL's evaluation script computes it; th.map: diameters in voxels
sp = dt_spacing(seg["data"] > 0, gobj={"rendered": G}, voxel_size_mm=0.0607)   # Tb.Sp and Tb.N use the whole SEG

grey = read_aim("SCAN.AIM"); per = read_aim("PERIOSTEAL.AIM")    # greyscale and rendered periosteal contour
r = cort_trab_separation(grey, per, TIBIA)                       # r["cort"], r["trab"]: IPL's CORT_MASK / TRAB_MASK
```

**Tb.Th has two definitions.** IPL's evaluation script takes Tb.Th on TRAB_SEG, as above, and the paper's comparison
with IPL uses that definition. ORMIR-BQRL and `ipldt-pipeline` report Tb.Th on the whole SEG inside the trabecular
contour, which measures a trabecula cut by the endocortical boundary at its full width; the scripts' definition is
computed by `ipldt-dt-thickness` on TRAB_SEG, `ipldt-ipl-morphometry --trab-seg` or
`ipldt.ormir.ipl_morphometry(..., trab_seg=...)` (`dt_thickness` itself takes either object), and the two workflows
report it with `--set morphometry.tbth_object=trab_seg`. The two are not interchangeable, so state which one you
report (`ormir_bqrl/README.md`, "Tb.Th").

**Privacy note.** An AIM written by `ipldt.write_aim`, and so every AIM that `ormir-bqrl` and `ipldt-pipeline` write
with `--map-format aim`, carries the input AIM's header and processing log, which name the patient and the scan
dates. Write NIfTI (the default of the command-line tools) or remove the header before sharing outputs.

## Parameters

Every tunable value of every stage is a parameter that you can change, and **the defaults are the configuration that
was validated against IPL**. The parameters keep IPL's names and values: the tibia (Script 32) and radius (Script 33)
presets of STEP 1, the contour rendering, the Laplace-Hamming filter and threshold, the SEG assembly, the
distance-transform parameters (ridge_epsilon 0.9, assign_epsilon 0.5, peel_iter -1, version 3), the objects and grids
of the dt stage, and the pore cascade. Two defaults depart from IPL's evaluation on purpose, and for each the value
compared with IPL is IPL's own: both workflows report Tb.Th on the whole SEG (`morphometry.tbth_object=seg`; the
paper compares IPL's definition, `trab_seg`, see "Tb.Th has two definitions" above), and `ipldt-pipeline` masks the
segmentation with the periosteal raster (`seg.periosteal_mask=raw`; IPL's evaluation, `ormir-bqrl` and the paper
use the rendered contour, `rendered`). The parameters are held by `ipldt.params.Parameters`, one checked block per
stage, 104 in all, each named `STAGE.NAME`:

| Stage | What it holds |
|---|---|
| `autocontour` | ORMIR-XCT's periosteal autocontour (`AutocontourKnee`'s periosteal settings and the component) |
| `calibration` | density slope, intercept, mu scaling and mu of water (default: the AIM's own) |
| `step1` | Script 32 / 33 STEP 1: the preset values and the script's other literals |
| `render` | the contour rendering (`min_vertices`) |
| `lh` | `/fft_laplace_hamming`, `/norm_max` and `/threshold` (in permille), the element sizes, the padding, the border |
| `seg` | the SEG assembly: component sizes, masks and their peel, order, labels |
| `dt` | `/dt_thickness`, `/dt_spacing`, `/dt_number` |
| `morphometry` | the objects of Tb.Th, Ct.Th and BV/TV, the maps computed, the grids, the voxel size of the statistics |
| `porosity` | the pore cascade (slice fractions, component bounds, hysteresis, grid) and the Ct.Po definition |
| `bmd` | the masks BMD is averaged in |
| `output` | the value written into the masks |

```bash
ormir-bqrl params                          # the complete default set as JSON (--site radius for the radius preset)
ormir-bqrl params --describe               # a table: name, value, default, IPL option, what it does
ormir-bqrl run SCAN.AIM out/eps05 --set lh.laplace_eps=0.5 --set seg.cc_min_trab=100
ormir-bqrl params --set lh.laplace_eps=0.5 --out my_params.json
ormir-bqrl batch results/ a.AIM b.AIM --params my_params.json
ormir-bqrl run SCAN.AIM out/again --params out/eps05/SCAN_parameters.json    # repeats that run exactly
ipldt-pipeline SCAN.AIM out/ --site radius --set porosity.slice_fraction=0,10
ipldt-pipeline --print-params --describe
ipldt-dt-thickness TRAB_SEG.AIM TRAB_TH.nii.gz --gobj TRAB_MASK.AIM --set dt.ridge_epsilon=0.8   # the dt stage only
```

```python
from ormir_bqrl import run
from ipldt.params import Parameters
from ipldt.ormir import run_pipeline

run("SCAN.AIM", "out/eps05", parameters={"lh.laplace_eps": 0.5, "step1": {"close2": 40}})   # overrides
run("SCAN.AIM", "out/p10", parameters=Parameters.defaults("radius").override({"porosity.min_pore_voxels": 10}))
report = run_pipeline("SCAN.AIM", "out/x", parameters="my_params.json")                    # a JSON file
report["parameter_set"]["non_default"]    # [] for the validated defaults; the complete set is in ["values"]
```

- **Precedence**: the site preset (`--site tibia|radius`) < a parameter file (`--params FILE`, JSON) < the dedicated
  flags (such as `--ridge-epsilon` or `--lh-voxel-size`) < `--set STAGE.NAME=VALUE` (repeatable). A file may hold only
  the values to change, nested (`{"lh": {"laplace_eps": 0.5}}`) or dotted (`{"lh.laplace_eps": 0.5}`), or a complete
  set as `ormir-bqrl params` prints it (a run's `<base>_parameters.json` or its report works too); a complete set is
  its preset plus its changes, so it repeats its run. The single dt commands take the `dt` stage only. An unknown
  name or a value of the wrong type, out of range or not among a parameter's choices is a usage error (exit 2) that
  lists the valid names; in Python every block is checked where a set is built.
- **Provenance**: every report records the complete effective parameter set, the names that differ from the
  validated defaults, where the values came from (`sources`: the site preset, the `--params` file, the dedicated
  flags and the `--set` names) and a one-line statement (`report["parameter_set"]`; the Markdown report of
  `ormir-bqrl` opens with it), and every run writes `<base>_parameters.json`. A value whose step did not run (for example an
  `autocontour` value when the periosteal contour is given) is not applied: it is warned about and listed as
  `not_applied`. A redo starts from the run's recorded parameters.
- **What was validated**: the defaults of the IPL-derived stages (STEP 1, the contour rendering, the Laplace-Hamming
  segmentation, the SEG assembly, the dt stage, the pore cascade) are the configuration compared with IPL in the
  paper, except for the two deliberate departures above, where the paper compared IPL's own value (Tb.Th on
  TRAB_SEG, `morphometry.tbth_object=trab_seg`; the rendered periosteal contour, `seg.periosteal_mask=rendered`).
  A report's statement names the departures in effect, and a run that sets either parameter to IPL's value is
  reported as using IPL's own choice. Any other **non-default value is not validated against IPL**: a run with one
  says so in its report, and values that no IPL export confirms (such as a `/seg_gauss` sigma other than 2) are also warned about as unverified.
  The periosteal autocontour and BMD come from ORMIR-XCT at ORMIR-XCT's defaults, which were not compared with IPL.
- With no override every output is identical to what it was before the parameters could be changed (checked on
  eleven real scans through both workflows). Fixed rules of IPL's commands that have no setting in IPL either, and
  IPL options of which only the observed setting is implemented, are not parameters; Supplementary Table S1 gives
  the defaults and says which of its values can be changed.

## Tests

```bash
pip install -e ".[bqrl,test]"      # Python 3.11+; on Python 3.10, ".[io,test]" (the ORMIR-XCT tests then skip)
pytest -m "not slow"               # synthetic phantoms; add the gpu extra for the CuPy tests (they skip without CUDA)
```

The `slow` tests and a few others compare against IPL's exported products of real scans and of dedicated test runs on
the scanner (the paper's probes), which are not public; they skip when the data are absent. Laboratory users point
`IPLDT_LAB_ROOT` at the data folder (`validation/datapaths.py` lists the layout and the per-location variables). The
de-identification rewrote the code and tests as well, so they name the scans by their pseudonyms and find the data
only when it is laid out under those names and that layout; with the scanner's own names the real-data tests skip and
the image-figure scripts stop.

## Regenerating the paper's numbers

```bash
pip install -e ".[paper]"
python tools/regenerate_paper_numbers.py
```

rebuilds, from the de-identified per-scan records under `validation/results/` and with no other data, the facts sheet
(`manuscript/facts/facts.json`, `FACTS.md`), the cortical-porosity facts (`FACTS_BMD_CTPO.json`), Tables 1, 2 and 3,
Supplementary Table S1 (as printed, every value checked against the code) and Figure 5. The figures that show images (Figures 1-4 and 6, Supplementary Figures S1-S9) need the scans and
IPL's products and cannot be redrawn from this repository; their PNG / SVG files and `*_numbers.json` sidecars are
the record. `manuscript/README.md` maps every figure, table and fact to its script and inputs, and
`validation/README.md` describes the validation harness and the record format.

## What is not in this repository, and why

- **Scans and images.** The grayscale reconstructions are patient images and are not shared, and neither are masks,
  maps or crops derived from them. The validation records carry voxel counts, statistics and voxel coordinates, not
  images. The figures of the paper are included as submitted, except Figure 5, which
  `tools/regenerate_paper_numbers.py` redraws from the de-identified records (the same data; overlapping markers can
  be drawn in a different order than in the submitted file).
- **Identifiers.** The records were de-identified with `tools/deidentify_results.py`: no names, birth dates, scan or
  evaluation dates and times, scanner paths or header dumps. The participant codes that the radius and tibia scans
  carry in their headers, the patellae's study codes, the scanner's measurement numbers and the scanner's file
  identifiers are all replaced by keyed pseudonyms (HMAC-SHA256; the key stays with the authors), consistently in
  file names, records, tables, figure sidecars, documentation and the comments and tests of the code, so that every
  cross-reference still resolves: a radius / tibia scan is `<region>/<study>/<six-digit pseudonym>`, a patella
  `PFJ-<six hex digits>_<L|R>`, a file identifier `X<seven digits>`. No scan identifier is kept as it is: the paper
  prints none (its figures name a scan by its site), and the tool's `stage` checks that against the paper. The list
  that links the pseudonyms to the scans stays with the authors.
- **Vendor material.** The manufacturer's evaluation scripts, IPL's help pages and logs and the result sheets are
  vendor material and are not redistributed; the code names IPL's commands and parameters only as far as needed to
  say what it reimplements.
- **IPL's products of the validation scans.** A reviewer bundle with IPL's products for one patella and a check script
  is deposited at Zenodo ([DOI to be added]); a second deposit, of IPL's products for all 21 patellae, is proposed
  (not yet made).
- **BMD validation.** ORMIR-BQRL computes BMD with ORMIR-XCT, but BMD is not part of the paper's validation, and its
  validation material is not included.

## Repository layout

```
ipldt/          the package (core distance transforms, IPL commands, STEP 1, contours, Laplace-Hamming, pores, I/O,
                the parameter model ipldt.params)
ormir_bqrl/     the workflow (pipeline, report, Slicer interchange, redo, preview) and the ormir-bqrl command
tests/          pytest suite (synthetic phantoms; real-data tests skip without the non-public data)
validation/     the validation harness and the de-identified per-scan records (validation/results/)
manuscript/     the scripts, figures, tables and facts of the paper (not its text)
tools/          de-identification / verification of the records, and the one-command regeneration
```

## Citation

If you use ipldt or ORMIR-BQRL, please cite the paper (reference to be added on publication) and the software
(`CITATION.cff`). ipldt builds on [ORMIR-XCT](https://github.com/ORMIRcommunity/ormir-xct) and follows the ORMIR
community guidelines for open and reproducible research in musculoskeletal imaging.

## License

MIT (see `LICENSE`). Scanco, IPL and XtremeCT are trademarks of Scanco Medical AG; this project is not affiliated with
or endorsed by Scanco.

Contact: Yihua Zhu, Bone Quality Research Lab, University of California, San Francisco; questions and bug reports
through the repository's issue tracker (https://github.com/zhuyihua1234/ipldt/issues).
