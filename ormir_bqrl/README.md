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
rendered periosteal contour, as in IPL's evaluation, rather than with the autocontour's raw raster, so the two SEGs
can differ by a few voxels (they agree exactly when rendering leaves the periosteal raster unchanged).

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
ormir-bqrl redo  <aim> <run_dir> [--periosteal FILE] [--trab FILE] [--cort FILE] [--site tibia|radius] [--out DIR]
                 [--no-bmd] [--no-porosity] [--backend ...] [--map-units ...] [--map-format ...]
ormir-bqrl slicer-export <run_dir>          # rebuild <base>_compartments.seg.nrrd and _labelmap.nii.gz
ormir-bqrl batch <out_root> <aim>... [--site] [--no-bmd] [--no-porosity] [--backend] [--map-units] [--map-format]
                 [--periosteal-dir DIR]
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

From Python:

```python
from ormir_bqrl import run, run_from_masks

report = run("scan/SCAN.AIM", "out/SCAN", site="tibia")
report["summary"]          # BV_TV, Tb_Th_mm, Tb_Sp_mm, Tb_N_per_mm, Ct_Th_mm (+ SDs), Ct_Po, Tb_BMD_mgHA_cm3, Ct_BMD_mgHA_cm3
report["porosity"]         # Ct_Po, Ct_Po_pore_voxels, Ct_Po_compartment_voxels ({} with compute_porosity=False)
report["parameters"]["porosity"]["grids"]   # the grids the cascade ran on: render, cort_seg, cascade (dim_xyz, pos_xyz)
redo = run_from_masks("scan/SCAN.AIM", "out/SCAN", trab="out/SCAN/SCAN_compartments.seg.nrrd")
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
| `<base>_report.json`, `.csv`, `.md` | metrics (Ct.Po in `summary` and in the `porosity` block), parameters, provenance of every mask, timings |
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

## Tb.Th

The workflow reports Tb.Th on the whole segmentation cropped to the trabecular contour, which measures a trabecula
cut by the endocortical boundary at its full width. The manufacturer's scripts compute it on the trabecular
segmentation (TRAB_SEG, which the workflow writes as `<base>_TRAB_SEG.nii.gz`). ipldt computes the scripts'
definition when it is given TRAB_SEG (`ipldt.ormir.ipl_morphometry(..., trab_seg=..., which=(..., "TRAB_TH_old"))`,
`ipldt-ipl-morphometry --trab-seg`, or `ipldt-dt-thickness` on the TRAB_SEG, as in the main README). The two are not
interchangeable: state which one you report. The paper's comparison with IPL uses the scripts' definition.

## Tests

`pytest tests/test_ormir_bqrl_cli.py tests/test_ormir_bqrl_pipeline.py tests/test_ormir_bqrl_porosity.py
tests/test_ormir_bqrl_slicer.py tests/test_porosity_grid.py` (synthetic AIMs and phantoms; the slow tests need
non-public data and skip without it). `tests/test_porosity_grid.py` covers the pore cascade's grid: the rule on
synthetic boxes and random rendered contours, and two phantoms on which the working grid changes the pore map while
`pore_cascade_ipl_grid` gives the same map whatever grid the inputs are held on.
