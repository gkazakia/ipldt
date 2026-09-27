# Changelog

## ipldt 1.0.0 / ORMIR-BQRL 0.1.0 (2026-09-27)

First public release, accompanying the paper (under review).

- **ipldt 1.0.0**: reimplementation of the IPL V5.42 bone morphometry evaluation (the Laplace-Hamming variant of the
  manufacturer's standard evaluation script): AIM input / output, Script 32 / 33 STEP 1 compartment separation,
  contour rendering, Laplace-Hamming segmentation, the cortical pore cascade, `/dt_thickness`, `/dt_spacing` and
  `/dt_number` on CPU (numba) and GPU (CuPy), metrics; SimpleITK and ORMIR-XCT adapters; the `ipldt-*` command-line
  tools. `ipldt-pipeline` runs the whole chain on a scan, including the cortical pore map and Ct.Po, and writes
  NIfTI or AIM.
- **ORMIR-BQRL 0.1.0**: the workflow on top of ipldt and ORMIR-XCT (`ormir-bqrl run | redo | slicer-export | batch |
  params | version`): BV/TV, Tb.Th, Tb.Sp, Tb.N, Ct.Th, Ct.Po and BMD in a JSON / CSV / Markdown report; masks, SEG,
  the cortical pore map and the distance-transform maps as NIfTI or, with `--map-format aim`, as AIMs on the input
  AIM's header (`--no-porosity` skips the pore cascade); 3D Slicer segmentations, and re-entry from masks corrected
  in 3D Slicer in the manner of the manufacturer's re-evaluation script (Script 34), which recomputes everything
  downstream of the corrected masks, the pore map and Ct.Po included.
- **The pore cascade's grid**: both workflows run the cortical pore cascade on IPL's render grid of the cortical
  contour, predicted from the rendered contour alone (`ipldt.porosity.render_grid`, `pore_cascade_ipl_grid`,
  `RENDER_GRID_MARGIN` = 2: the contour's tight in-plane box, low end - 2 clipped at 0, high end the largest of
  (slice high end + 1 where the slice's extent is even) + 2, z the contour's slices), and paste the pore map back onto
  the input AIM's grid; the grids are recorded in the reports (`parameters.porosity.grids`, `porosity_grids`). The
  prediction is IPL's own render grid on 137 of the 137 validation scans, and with IPL's configuration-A inputs run
  through the workflow code the pore map is identical to IPL's on 116 of 116 radius / tibia scans and 21 of 21
  patellae; on the plain grid of the input AIM, which the code used before, 4 of the 137 differed.
- The validation harness, the de-identified per-scan validation records of the 137 scans of the paper, and the scripts,
  figures, tables and facts of the paper.

### 2026-09-27: every parameter can be changed; the defaults are the validated IPL configuration

- **One parameter model**, `ipldt.params`: `Parameters` holds one checked block per stage (`autocontour`,
  `calibration`, `step1` = `Step1Params`, `render`, `lh`, `seg`, `dt` = `DTParams`, `morphometry`, `porosity`, `bmd`,
  `output`; 104 parameters named `STAGE.NAME`, each type-, range- and choice-checked). `Parameters.defaults(site)` is
  the configuration validated against IPL: the Script 32 (tibia) / Script 33 (radius) presets of STEP 1, the
  Laplace-Hamming variant with IPL's `ceil` padding, the SEG assembly and distance-transform parameters of Script 32
  and the render-grid rule of the pore cascade. Two defaults depart from IPL's evaluation on purpose, and for each
  the value compared with IPL is IPL's own: Tb.Th on the whole SEG (`morphometry.tbth_object=seg`; the paper compares
  IPL's definition, `trab_seg`) and, in `ipldt-pipeline`, the periosteal raster in the SEG assembly
  (`seg.periosteal_mask=raw`; IPL's rule: `rendered`). `Step1Params` gains the script's remaining STEP 1 literals as
  fields with the script's values; `TIBIA` / `RADIUS` are unchanged in value, and the module constants keep their
  names.
- **Python and command line**: `parameters=` on `ipldt.ormir.run_pipeline`, `ormir_bqrl.run` and `run_from_masks` (a
  mapping of overrides, a JSON file or a complete `Parameters`; a redo starts from the run's recorded set);
  `--params FILE` (JSON) and repeatable `--set STAGE.NAME=VALUE` on `ipldt-pipeline` and `ormir-bqrl run | redo |
  batch`, and for the `dt` stage on the single dt commands; `ormir-bqrl params [--site] [--describe]` and
  `ipldt-pipeline --print-params [--describe]` print the complete set. Precedence: site preset < `--params` <
  dedicated flags < `--set`; an unknown name or a wrong value is a usage error that lists the valid names.
  `ipldt-pipeline` also gains `--periosteal FILE`, `--no-porosity` and a three-value `--lh-voxel-size`.
- **Provenance**: every report records the complete effective set, the names that differ from the validated defaults,
  where the values came from (`sources`: the preset, the `--params` file, the dedicated flags, the `--set` names) and
  a one-line statement that names the deliberate departures in effect and reports a value set to IPL's own choice as
  that (`report["parameter_set"]`; the Markdown report opens with it), and every run writes
  `<base>_parameters.json`, which `--params` reads back to repeat the run. Values of a step that does not run are not
  applied (warned about and listed as `not_applied`); values no IPL export confirms are warned about as unverified.
  **Other non-default values are not validated against IPL**, and the periosteal autocontour and BMD use ORMIR-XCT's
  defaults, which were not compared with IPL.
- **Defaults unchanged**: with no override every output volume and every report value is identical to the code
  before this change (eleven real scans through both workflows, the command lines, the redo and a batch); the reports
  gain only the parameter blocks. Supplementary Table S1's note says which of its values can be changed. Tests:
  `tests/test_params.py` (104 tests; the suite collects 434).
