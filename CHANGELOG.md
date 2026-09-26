# Changelog

## ipldt 1.0.0 / ORMIR-BQRL 0.1.0 (2026-09-26)

First public release, accompanying the paper (under review).

- **ipldt 1.0.0**: reimplementation of the IPL V5.42 bone morphometry evaluation (the Laplace-Hamming variant of the
  manufacturer's standard evaluation script): AIM input / output, Script 32 / 33 STEP 1 compartment separation,
  contour rendering, Laplace-Hamming segmentation, the cortical pore cascade, `/dt_thickness`, `/dt_spacing` and
  `/dt_number` on CPU (numba) and GPU (CuPy), metrics; SimpleITK and ORMIR-XCT adapters; the `ipldt-*` command-line
  tools. `ipldt-pipeline` runs the whole chain on a scan, including the cortical pore map and Ct.Po, and writes
  NIfTI or AIM.
- **ORMIR-BQRL 0.1.0**: the workflow on top of ipldt and ORMIR-XCT (`ormir-bqrl run | redo | slicer-export | batch |
  version`): BV/TV, Tb.Th, Tb.Sp, Tb.N, Ct.Th, Ct.Po and BMD in a JSON / CSV / Markdown report; masks, SEG, the
  cortical pore map and the distance-transform maps as NIfTI or, with `--map-format aim`, as AIMs on the input AIM's
  header (`--no-porosity` skips the pore cascade); 3D Slicer segmentations, and re-entry from masks corrected in
  3D Slicer in the manner of the manufacturer's re-evaluation script (Script 34), which recomputes everything
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
