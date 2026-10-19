# ipldt documentation

These pages carry the implementation detail that the paper describing ipldt (under review at JBMR Plus) summarizes: how each stage of the Scanco IPL V5.42 standard bone morphometry evaluation for XtremeCT II images is reimplemented, how the rules were derived without access to IPL's source code, the software environment and runtime of the validation, and the parameter atlas.

- [How ipldt reimplements the IPL evaluation, stage by stage](implementation.md) — the nine stages from AIM input to the manual-correction re-entry, with every IPL command name, parameter value, and convention, followed by the table of modules and their voxel-level agreement with IPL.
- [How the reimplementation was derived](derivation.md) — the sources, the method, one worked test run, and the scans used in the derivation.
- [Software environment and runtime](environment.md) — requirements, versions, hardware, memory, runtime per scan, and the GPU and CPU back ends.
- [Documentation figures D1–D9](figures.md) — the parameter atlas, one panel per parameter or rule, with complete legends.

Every parameter of ipldt and ORMIR-BQRL, with its IPL option, its value in the tibia and radius presets, and the panel that shows it, is listed in the [parameter table](../manuscript/tables/tableS1_parameters.md). Installation and usage are described in the [package README](../README.md). Figure numbers in these pages follow the figure files of `manuscript/figures/` (`fig1_workflow` … `fig6_porosity`); in the published paper, `fig3_lh` is Supplementary Figure S1 and `fig4_dt`, `fig5_agreement`, and `fig6_porosity` are Figures 3, 4, and 5.
