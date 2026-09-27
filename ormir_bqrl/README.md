# ORMIR-BQRL

The Bone Quality Research Lab's HR-pQCT bone microstructure workflow, built on the `ipldt` engine: a Scanco
XtremeCT II `.AIM` in, IPL's metrics out (BV/TV, Tb.Th, Tb.Sp, Tb.N, Ct.Th, Ct.Po, and Tb.BMD / Ct.BMD through ORMIR-XCT),
fully automated, with a manual-correction loop through 3D Slicer that re-enters the workflow the way the
manufacturer's re-evaluation script (Script 34) re-enters the standard evaluation (Script 32). Package `ormir_bqrl`,
command `ormir-bqrl`, version 0.1.0, MIT licence.

Every numerical stage is ipldt's reimplementation of IPL V5.42 (the compartment separation of Script 32 / 33 STEP 1,
IPL's contour rendering, the Laplace-Hamming segmentation and its SEG assembly, the cortical pore cascade, the
distance-transform morphometry).
The periosteal autocontour, the AIM-to-HU reader and `bmd_masked` come from
[ORMIR-XCT](https://github.com/ORMIRcommunity/ormir-xct). ORMIR-BQRL composes the step functions of `ipldt.ormir` and
adds the 3D Slicer interchange, the re-entry from corrected masks, reports with provenance, a preview figure and the
command line. Its one numerical difference from `ipldt-pipeline`: the Laplace-Hamming segmentation is masked with the
rendered periosteal contour, as in IPL's evaluation, rather than with the autocontour's raw raster (the default of the
parameter `seg.periosteal_mask` in each workflow), so the two SEGs can differ by a few voxels (they agree exactly when
rendering leaves the periosteal raster unchanged). Every value of every stage is a parameter, and the defaults are the
configuration validated against IPL, apart from the deliberate Tb.Th definition (see "Parameters" and "Tb.Th").

The cortical pore map and Ct.Po come from `ipldt.porosity`, called exactly as `ipldt-pipeline` calls it
(`ipldt.ormir.step5c_porosity`): the cascade runs on the rendered cortical contour and the cortical segmentation on
IPL's render grid of the cortical contour, predicted from the rendered contour alone, and the pore map is pasted back
onto the input AIM's grid (see "The pore map's grid").

## Install

```bash
pip install "ipldt[bqrl] @ git+https://github.com/zhuyihua1234/ipldt"          # add ,gpu for the CuPy back end
```

ORMIR-XCT needs Python 3.11 or later.

## Run

```bash
ormir-bqrl run   <aim> <out_dir> [--site tibia|radius] [--periosteal FILE] [--no-bmd] [--no-porosity]
                 [--backend auto|gpu|cpu] [--map-units voxels|mm] [--map-format nifti|aim] [--subfolder] [--no-preview]
                 [--params FILE] [--set STAGE.NAME=VALUE ...]
ormir-bqrl redo  <aim> <run_dir> [--periosteal FILE] [--trab FILE] [--cort FILE] [--site tibia|radius] [--out DIR]
                 [--no-bmd] [--no-porosity] [--backend ...] [--map-units ...] [--map-format ...]
                 [--params FILE] [--set STAGE.NAME=VALUE ...]
ormir-bqrl slicer-export <run_dir>          # rebuild <base>_compartments.seg.nrrd and _labelmap.nii.gz
ormir-bqrl batch <out_root> <aim>... [--site] [--no-bmd] [--no-porosity] [--backend] [--map-units] [--map-format]
                 [--periosteal-dir DIR] [--params FILE] [--set STAGE.NAME=VALUE ...]
ormir-bqrl params [--site tibia|radius] [--params FILE] [--set STAGE.NAME=VALUE ...] [--describe] [--out FILE]
ormir-bqrl version
```

- `run` is the automatic evaluation. `--periosteal FILE` replaces the autocontour by a mask from disk (a binary
  NIfTI, an ORMIR-BQRL labelmap or `.seg.nrrd`, or an AIM mask, aligned by global position).
- `--no-porosity` skips the pore cascade (no `<base>_PORE`, no Ct.Po); by default it runs.
- `--map-format aim` writes the masks, SEG, the pore map and the maps as uncompressed char AIMs on the input AIM's
  header instead of NIfTI (the writer `ipldt-pipeline --map-format aim` uses). The maps are then integer sphere
  diameters in voxels, so it cannot be combined with `--map-units mm`, and a map whose values do not fit the char
  range 0-255 is refused before anything is written. These AIMs carry the input AIM's header and processing log,
  which name the patient and the scan dates: remove the header before sharing them.
- `redo` re-enters from corrected masks; the flag names what you edited, the unedited masks are read from `<run_dir>`
  (NIfTI or AIM), and the output format and map units default to the run's.
- `batch` runs one `run` per AIM into `<out_root>/<base>/`, logs and skips failures, and writes
  `<out_root>/batch_summary.csv`.
- `--params FILE` / `--set STAGE.NAME=VALUE` change any parameter of any stage; the defaults are the validated IPL
  configuration, and `params` prints the complete set (JSON, loadable with `--params`; `--describe` for a table). See
  "Parameters". `--site` defaults to tibia, or to the preset of a complete `--params` set; a redo starts from the
  run's recorded parameters.
- Exit codes: 0; 1 with `Error: <message>` on stderr; 2 for a usage error (an unknown parameter name or a value of the
  wrong type is one, with the valid names in the message).

From Python:

```python
from ormir_bqrl import run, run_from_masks

report = run("scan/SCAN.AIM", "out/SCAN", site="tibia")
report["summary"]          # BV_TV, Tb_Th_mm, Tb_Sp_mm, Tb_N_per_mm, Ct_Th_mm (+ SDs), Ct_Po, Tb_BMD_mgHA_cm3, Ct_BMD_mgHA_cm3
report["porosity"]         # Ct_Po, Ct_Po_pore_voxels, Ct_Po_compartment_voxels ({} with compute_porosity=False)
report["parameters"]["porosity"]["grids"]   # the grids the cascade ran on: render, cort_seg, cascade (dim_xyz, pos_xyz)
report["parameter_set"]["non_default"]      # [] for the validated defaults; the complete set is in ["values"]
redo = run_from_masks("scan/SCAN.AIM", "out/SCAN", trab="out/SCAN/SCAN_compartments.seg.nrrd")
eps = run("scan/SCAN.AIM", "out/eps05", parameters={"lh.laplace_eps": 0.5})      # any parameter changed
```

## Outputs

Everything is written into `<out_dir>/` on the input AIM's grid, as NIfTI with the AIM's element sizes and position,
so every file overlays every other one in 3D Slicer and SimpleITK. With `--map-format aim` every mask, SEG, pore-map
and map file of the table is written instead as `<base>_<NAME>.AIM` (an uncompressed char AIM carrying the input
AIM's header, maps as integer diameters in voxels), which `ipldt.io.read_aim`, `redo` and `slicer-export` read
back; the greyscale in HU, the two Slicer files, the report and the preview are written in both formats.

The distance-transform maps are computed on the grids IPL uses (the bounding box of the segmentation, and of the
cortical mask for Ct.Th) and pasted back onto the input AIM's grid. The transforms count the faces of the grid as
object, so to recompute the maps from the written SEG or TRAB_SEG with the single commands (`ipldt-dt-number` and the
others), cut the file to the bounding box of its set voxels first (`ipldt.ipl_ops.bounding_box_cut`); on the full grid
Tb.N in particular comes out slightly different.

| File | Content |
|---|---|
| `<base>_HU.nii.gz` | the greyscale in HU (for 3D Slicer) |
| `<base>_PRX_MASK.nii.gz`, `<base>_PRX_GOBJ.nii.gz` | the periosteal mask and its rendered contour |
| `<base>_CORT_MASK.nii.gz`, `<base>_TRAB_MASK.nii.gz`, `_CORT_GOBJ`, `_TRAB_GOBJ` | the compartment masks and their rendered contours |
| `<base>_SEG.nii.gz`, `_CORT_SEG`, `_TRAB_SEG` | the segmentation (127 cortical / 126 trabecular) and its two compartments |
| `<base>_PORE.nii.gz` | the cortical pore map (IPL's PORE.AIM), from the rendered cortical contour and CORT_SEG on IPL's render grid, pasted back onto the AIM grid; absent with `--no-porosity` |
| `<base>_TRAB_TH.nii.gz`, `_TRAB_SP`, `_TRAB_1N`, `_CORT_TH` | the distance-transform maps (sphere diameters in voxels, or mm with `--map-units mm`) |
| `<base>_compartments.seg.nrrd`, `<base>_compartments_labelmap.nii.gz` | the 3D Slicer segmentation (1 cortical, 2 trabecular) |
| `<base>_report.json`, `.csv`, `.md` | metrics (Ct.Po in `summary` and in the `porosity` block), the parameters used and the complete parameter set (`parameter_set`), provenance of every mask, timings |
| `<base>_parameters.json` | the complete effective parameter set; `--params` reads it back, so the run can be repeated exactly |
| `<base>_preview.png`, `<base>_pipeline.log` | a preview figure and the log |

## The pore map's grid

The cortical pore cascade depends on the grid it runs on, not only on its inputs: its two slice-wise steps (the 0-5 %
passes) measure every component against the non-bone voxels of its slice in the working grid, so a larger grid lowers
the bar and turns a small marrow cavity into a "pore", and where the bone reaches the faces of a tight grid the
background outside the contour is cut into pieces that each pass as a pore. The workflow therefore runs the cascade
on the grid IPL runs it on, IPL's render grid of the cortical contour (the grid IPL's `/gobj_to_aim` renders the
contour onto), which `ipldt.porosity.render_grid` predicts from the rendered contour alone:

- in-plane, per axis: low = the contour's lowest coordinate - 2 (`RENDER_GRID_MARGIN`), clipped at 0; high = the
  largest, over the contour's slices, of (the slice's highest coordinate + 1 where the slice's extent is even) + 2;
- in z: the slices that carry the contour;
- where the grid reaches past the input AIM it is zero-padded; the cortical segmentation sits on the tight box of
  its voxels.

`ipldt.porosity.pore_cascade_ipl_grid` moves the inputs onto these grids, runs the cascade there, and the pore map is
pasted back onto the AIM grid by global position; Ct.Po (`ipldt.porosity.ct_po`, the pore voxels over the rendered
cortical contour) is unchanged. The report records the rule (`parameters.porosity.grid_rule`) and the grids the
cascade ran on (`parameters.porosity.grids`: `render`, `cort_seg`, `cascade`, each `dim_xyz` / `pos_xyz`; null when
it did not run); `ipldt-pipeline` writes the same grids under `porosity_grids`.

Validation: on the 137 scans of the paper the predicted grid is IPL's own render grid on 137 of 137. With IPL's
configuration-A inputs (IPL's rendered cortical contour and cortical segmentation) held on each scan's greyscale-AIM
grid, as the workflow holds its arrays, and run through the workflow's code, the pore map is identical to IPL's on all
116 radius and tibia scans and all 21 patellae; run on the plain grid of the input AIM instead, as the code did before
this rule, it differed from IPL's on 4 of the 137.

## Correcting masks in 3D Slicer

Load `<base>_HU.nii.gz` and `<base>_compartments.seg.nrrd`, edit the trabecular (or the cortical) segment, save the
segmentation, and run `ormir-bqrl redo <aim> <run_dir> --trab <saved .seg.nrrd>` (or `--cort`, or `--periosteal` for
a corrected bone contour). What an edit recomputes:

| Edited | Rule |
|---|---|
| periosteal | the compartment separation reruns from the corrected contour |
| trabecular | cortical = periosteal minus the rendered trabecular contour (Script 34's rule); refused if the edit leaves the bone |
| cortical | the symmetric extension: trabecular = periosteal minus the rendered cortical contour |
| trabecular and cortical | refused: pass only the compartment you edited |

The segmentation, the maps, the pore map with Ct.Po, BMD and the report are then recomputed from the corrected masks.
A redo keeps the run's parameters (it starts from the parameter set the run recorded; `--params` / `--set` change it
where their step runs). A compartment redo does not rerun the compartment separation, so an explicit `--site` there
is reported as not applied (a warning, and `parameter_set.not_applied` in the report).

## Parameters

Every tunable value of every stage is one parameter of `ipldt.params.Parameters`, named `STAGE.NAME`, and **the
defaults of the IPL-derived stages are the configuration validated against IPL**: the Script 32 (tibia) / Script 33
(radius) presets of the compartment separation, the contour rendering, the Laplace-Hamming variant with IPL's
padding, the SEG assembly, the distance-transform parameters and IPL's render-grid rule of the pore cascade. One
default departs from IPL's evaluation on purpose: the workflow reports Tb.Th on the whole segmentation
(`morphometry.tbth_object=seg`, see "Tb.Th"), whereas the Tb.Th compared with IPL is IPL's own definition on TRAB_SEG
(`trab_seg`); `ipldt-pipeline` also departs in `seg.periosteal_mask` (`raw`, where this workflow's `rendered` is
IPL's rule). A run that sets IPL's value is reported as using IPL's own choice, not as a departure from the
validated configuration. The periosteal autocontour and BMD are ORMIR-XCT's; their defaults
are ORMIR-XCT's, which were not compared with IPL. A run with no override is identical to a run of the code before
the parameters could be changed (checked on eleven real scans through both workflows, every volume voxel for voxel
and every report value).

| Stage | Parameters (104 in all; `ormir-bqrl params --describe` lists each with its IPL option) |
|---|---|
| `autocontour` | ORMIR-XCT's `AutocontourKnee` periosteal parameters (`peri_s1_sigma` ... `peri_s4_close_radius`) and `component` (1 = the largest bone) |
| `calibration` | `slope`, `intercept`, `mu_scaling`, `mu_water`: None = the AIM's own (processing log, else the header); `slope`, `intercept` and `mu_scaling` recalibrate the compartment separation, the autocontour and BMD (a `mu_scaling` that differs from the header's recomputes the HU image from the native data); `mu_water` is the HU scale; the report records the calibration used (`input_aim.calibration_effective`) |
| `step1` | the sixteen values of the site presets and the script's remaining literals: `rank_first` / `rank_last` / `rank_connect_boundary`, `continuous_x/y/z` (alias `step1.continuous_at_boundary=1,1,1`), `mask_peel`, `corner_min_max_number` / `corner_max_min_number`, `bbc_border_x/y/z` (alias `step1.bbc_border`) |
| `render` | `min_vertices` (IPL's rule: 4) |
| `lh` | `laplace_eps`, `lp_cut_off_freq`, `hamming_amp`, `norm_max`, `lower_permille` / `upper_permille` (475 / 1000), `el_size_mm` (None = the header's per-axis sizes), `pad_offset` (`ceil` / `floor`), `dtype` (`float32` / `float64`), `border` (`duplicate` / `none` / `zero`) |
| `seg` | `cc_min_cort` / `cc_min_trab` (35 / 70), `cc_max_*`, `periosteal_mask` (`rendered` here, `raw` in `ipldt-pipeline`), `peel_periosteal` / `peel_cort` / `peel_trab`, `order`, `trab_seg_mask`, `value_cort` / `value_trab` (127 / 126), `overlap` |
| `dt` | `ridge_epsilon`, `assign_epsilon`, `peel_iter`, `version`, `suppress_boundary` |
| `morphometry` | `tbth_object` (`seg` = the whole segmentation / `trab_seg` = the scripts' definition), `tbth_grid`, `ctth_object`, `bvtv_object`, `maps`, `grid_border`, `voxel_size_mm` |
| `porosity` | `slice_lo` / `slice_up` (0 / 5 %, alias `porosity.slice_fraction`), `min_pore_voxels` / `max_pore_voxels`, `low_thresh` / `high_thresh` / `mode` / `grow_axes`, `marrow_rank_first` / `_last` / `marrow_connect_boundary`, `gobj_peel`, `render_grid_margin` / `render_grid_clip_low`, `grid` (`ipl` / `aim`), `ct_po` (`contour` / `pore_plus_bone`) |
| `bmd` | `masks` (`rendered` / `raw`) |
| `output` | `mask_value` (127) |

**Precedence**: the `--site` preset < a parameter file (`--params FILE`, JSON) < `--set STAGE.NAME=VALUE`
(repeatable). A file may hold only the values to change, nested (`{"lh": {"laplace_eps": 0.5}}`) or dotted
(`{"lh.laplace_eps": 0.5}`), or the complete set as `ormir-bqrl params` prints it, a run's `<base>_parameters.json`
or a run report. A complete set is its preset plus the values that differ from that preset's defaults: without
`--site` its own preset is used, so `--params <run>/<base>_parameters.json` repeats the run exactly; with `--site` its
changes apply to that site's preset; and a set written by `ipldt-pipeline` does not carry that workflow's own
`seg.periosteal_mask` into ORMIR-BQRL (nor the reverse). An unknown name, or a value of the wrong type, out of range
or not among a parameter's choices, is a usage error (exit 2) that names the valid parameters; in Python the same
checks run on every block, also one built directly.

```bash
ormir-bqrl params                                   # the complete default set as JSON (--site radius for the radius preset)
ormir-bqrl params --describe                        # a table: name, value, IPL option, what it does
ormir-bqrl params --set lh.laplace_eps=0.5 --out my_params.json
ormir-bqrl run  SCAN.AIM out/eps05 --set lh.laplace_eps=0.5 --set seg.cc_min_trab=100
ormir-bqrl run  SCAN.AIM out/scripts_tbth --set morphometry.tbth_object=trab_seg
ormir-bqrl run  SCAN.AIM out/custom --site radius --params my_params.json --set porosity.slice_fraction=0,10
ormir-bqrl batch out/cohort *.AIM --params my_params.json
ormir-bqrl redo SCAN.AIM out/eps05 --trab edited.seg.nrrd                     # keeps the run's parameters
ormir-bqrl run  SCAN.AIM out/again --params out/eps05/SCAN_parameters.json    # repeats that run exactly
```

```python
from ormir_bqrl import run
from ipldt.params import Parameters
run(aim, out, parameters={"lh.laplace_eps": 0.5, "step1": {"close2": 40}})          # overrides on the site preset
P = Parameters.defaults("radius").override({"porosity.min_pore_voxels": 10})
run(aim, out, parameters=P)                                                          # a complete set
run(aim, out, parameters="out/eps05/SCAN_parameters.json")                          # a file
```

`step1_params=` and `dt_params=` keep working and win over `parameters`. **What is recorded**: every report carries
`parameter_set` -- the complete effective set (`values`), the names that differ from the validated defaults of the
preset (`non_default`, with each value and its default), `validated_default` (true / false), a one-line `statement`
(whether every value of the IPL-derived stages is the one validated against IPL, which deliberate departures from
IPL's evaluation are in effect -- by default the Tb.Th object -- and that the periosteal autocontour and BMD are
ORMIR-XCT's, not validated against IPL, or that the periosteal contour was given),
`validated_scope`, the values no IPL export confirms (`unverified`, such as a `/seg_gauss` sigma other than 2, also
warned about at run time) and the `sources` of the values (the site preset, for a redo the run's recorded set, the
`--params` file, the `--set` names). The Markdown report opens with the statement ("**parameters:
NOT the validated IPL defaults.**" with the names, for a non-default run) and lists the whole set, non-default values
in bold; the run log, the preview's text panel and `batch_summary.csv` (a last column `non_default_parameters`, only
for a non-default batch) flag a non-default run too. **Non-default values other than IPL's own choices are not
validated against IPL.**

**A value of a step that does not run is not applied**: `autocontour.*` when the periosteal contour is given, `bmd.*`
with `--no-bmd`, `porosity.*` with `--no-porosity`, and in a compartment redo `step1.*` (and `--site`) and
`autocontour.*`. Such a value is warned about, the set keeps the value actually used (the default, or in a redo the
run's), and the report lists it under `not_applied`, never as non-default. A voxel size or Laplace-Hamming element
size given with the AIM header's own value is the default's value (`equal_to_default`).

**Not parameters** (fixed rules of IPL's commands with no setting in IPL either, or IPL options of which only the
observed setting is implemented): the `/seg_gauss` float32 arithmetic, the chamfer metric of the morphology,
`-topology 6`, `-metric 11`, the contour trace and fill rules and `-curvature_smooth 1`, the power-of-two box and its
mirror fill, the float-to-short truncation, the distance-transform engine's rules, the pore cascade's label encoding
and `/hysteresis_threshold -unit 5`, and file-format facts. The preview figure's colour ranges are display settings.

## Tb.Th

The workflow reports Tb.Th on the whole segmentation cropped to the trabecular contour, which measures a trabecula
cut by the endocortical boundary at its full width. The manufacturer's scripts compute it on the trabecular
segmentation (TRAB_SEG, which the workflow writes as `<base>_TRAB_SEG.nii.gz`). ipldt computes the scripts'
definition when it is given TRAB_SEG (`ipldt.ormir.ipl_morphometry(..., trab_seg=..., which=(..., "TRAB_TH_old"))`,
`ipldt-ipl-morphometry --trab-seg`, or `ipldt-dt-thickness` on the TRAB_SEG, as in the main README), and the workflow
reports it with `--set morphometry.tbth_object=trab_seg`. The two are not interchangeable: state which one you report.
The paper's comparison with IPL uses the scripts' definition.

## Tests

`pytest tests/test_ormir_bqrl_cli.py tests/test_ormir_bqrl_pipeline.py tests/test_ormir_bqrl_porosity.py
tests/test_ormir_bqrl_slicer.py tests/test_porosity_grid.py tests/test_params.py` (synthetic AIMs and phantoms; the
slow tests need non-public data and skip without it). `tests/test_porosity_grid.py` covers the pore cascade's grid:
the rule on synthetic boxes and random rendered contours, and two phantoms on which the working grid changes the pore
map while `pore_cascade_ipl_grid` gives the same map whatever grid the inputs are held on. `tests/test_params.py`
(104 tests) covers the parameter model: the defaults equal the engine's values for both presets and a default run
makes exactly the previous calls; every stage's override reaches its stage and changes its output on a phantom;
`--set` / `--params` parsing, precedence and error messages; the report's parameter block and statement; values of
steps that do not run; and a redo that keeps a non-default run's parameters and reproduces that run exactly.
