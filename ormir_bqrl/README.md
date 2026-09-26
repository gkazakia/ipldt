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

The cortical pore map and Ct.Po come from `ipldt.porosity` (`pore_cascade`, `ct_po`), called exactly as
`ipldt-pipeline` calls it: the cascade receives the rendered cortical contour and the cortical segmentation on the
input AIM's grid. The paper's validation placed the cortical contour on IPL's own box (the box of the periosteal
contour, where IPL renders it). The cascade's slice-wise steps depend on the grid they run on, so the pore map could
differ between the two grids; on the one scan of the validation set run both ways (a patella, from IPL's periosteal
contour) they gave the same pore map, identical to IPL's voxel for voxel, and the other validation scans, the radius
and tibia scans among them, have not been run on the workflow's grid.

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
| `<base>_PORE.nii.gz` | the cortical pore map (IPL's PORE.AIM), from the rendered cortical contour and CORT_SEG; absent with `--no-porosity` |
| `<base>_TRAB_TH.nii.gz`, `_TRAB_SP`, `_TRAB_1N`, `_CORT_TH` | the distance-transform maps (sphere diameters in voxels, or mm with `--map-units mm`) |
| `<base>_compartments.seg.nrrd`, `<base>_compartments_labelmap.nii.gz` | the 3D Slicer segmentation (1 cortical, 2 trabecular) |
| `<base>_report.json`, `.csv`, `.md` | metrics (Ct.Po in `summary` and in the `porosity` block), parameters, provenance of every mask, timings |
| `<base>_preview.png`, `<base>_pipeline.log` | a preview figure and the log |

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
tests/test_ormir_bqrl_slicer.py` (synthetic AIMs; the slow test needs non-public data and skips without it).
