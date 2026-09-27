"""ormir_bqrl -- ORMIR-BQRL: the BQRL bone-microstructure HR-pQCT workflow.

A Scanco AIM in, IPL-equivalent metrics out (BV/TV, Tb.Th, Tb.Sp, Tb.N, Ct.Th, Ct.Po, Tb.BMD, Ct.BMD), fully
automated, with a manual-correction loop through 3D Slicer that re-enters the workflow the way Scanco Script 34
re-enters Script 32.  `ipldt` is the engine (the reimplemented IPL commands, the cortical pore cascade included);
ORMIR-XCT contributes the periosteal autocontour, the AIM-to-HU reader and bmd_masked; everything else is the
lab's.  Masks, SEG, the pore map and the dt maps are written as NIfTI or, with map_format="aim", as AIMs on the
input AIM's header; a JSON / CSV / Markdown report and a 3D Slicer segmentation (.seg.nrrd) come with every run.

    from ormir_bqrl import run, run_from_masks
    report = run("X2420448.AIM", "out/X2420448", site="tibia")                       # AIM -> everything
    report = run_from_masks("X2420448.AIM", "out/X2420448", trab="edited.seg.nrrd")  # Script 34 re-entry
    report = run("X2420448.AIM", "out/eps05", parameters={"lh.laplace_eps": 0.5})    # any parameter changed

Every tunable value of every stage is a parameter (ipldt.params; `ormir-bqrl params --describe` lists them), and the
defaults are the validated IPL configuration apart from the deliberate Tb.Th object (the whole SEG; IPL's evaluation script, and the comparison with IPL, use TRAB_SEG); a report says when its
configuration differs from them.

Modules: stages (the shared body and data contract), pipeline (run), redo (run_from_masks), slicer (the Slicer
interchange), report, preview, cli (`ormir-bqrl run | redo | slicer-export | batch | version`).
"""
__version__ = "0.1.0"
PRODUCT = "ORMIR-BQRL"

from .pipeline import run                 # noqa: E402
from .redo import run_from_masks          # noqa: E402

__all__ = ["__version__", "PRODUCT", "run", "run_from_masks"]
