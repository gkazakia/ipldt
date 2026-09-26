"""Regression tests for validation/validate_dataset.py's script-variant detection.

WHAT IS PINNED HERE
  * a measurement that ships a SEG keeps reading its variants from the SEG's processing log and never looks at an
    evaluation log, even when one is sitting in the same folder (two of the three OS_LH folders that ship an
    evaluation log DO ship a SEG, so this is the case that must not change);
  * that SEG processing log now also decides trab_seg_mask, from the same fact the evaluation log decides it by --
    a D3P_GobjOrAimMaskAimPeel_OW after D3P_Cl_ExtractNumber_CPP means TRAB_SEG was cropped to the trabecular
    contour, its absence means it was not -- which selects eleven measurements of the cohort and no others;
  * a measurement that ships NO SEG reads them from IPL's own evaluation log instead, including the two commands
    that ERRORED in Diaphyseal/BMAT/610892's second run -- '/fill_offset_duplicate -input cort' (so the
    Laplace-Hamming input has no duplicated border) and '/gobj_maskaimpeel_ow -input_output trab_gauss' (so
    TRAB_SEG was never cropped to the trabecular contour);
  * when two evaluation runs exist, the run is chosen from the DELIVERED files' own processing logs, not assumed;
  * assemble_seg's new trab_masked flag changes the trabecular branch and only the trabecular branch.

The synthetic logs below are minimal stand-ins written for these tests: only the lines the parser reads (the job
stamps, the session marker, the STEP 2 command names with the options it uses, one short error marker per failed
command and the '-> Set' line), with synthetic dates, names and counts; the two `slow` tests at the end assert the
same things against the real cohort and skip themselves when it is not mounted.
"""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION = os.path.join(REPO, "validation")
for _p in (REPO, VALIDATION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

vd = pytest.importorskip("validate_dataset", reason="validation/validate_dataset.py needs the validation extras")

COHORT = os.environ.get("OSLH_ROOT", os.path.join(os.environ.get("IPLDT_LAB_ROOT") or "IPLDT_LAB_ROOT_is_not_set",
                                                 "Cross_validation_IPL", "OS_LH"))


# ----------------------------------------------------------------------------- synthetic evaluation logs
def _log(start, end, fill_input, fill_ok, trab_ok):
    """A minimal synthetic evaluation log: the STEP 2 commands whose outcome the variants depend on, each with the
    options the parser reads (options follow their command line directly), and an error marker where one failed."""
    fill_err = "" if fill_ok else f"!% Error reading file: {fill_input}\n"
    trab_obj = "trab_seg" if trab_ok else "trab_gauss"
    trab_body = "!% -> Set 1000000 of total 4000000\n" if trab_ok else "!% Error reading file: trab_gauss\n"
    return f"""\
!% job start {start}
!% IPL Processing Starts
/read
  -name                      org
  -filename                  disk:[dir]x1469636.aim
/fill_offset_duplicate
  -input                     {fill_input}
{fill_err}/fft_laplace_hamming
  -input                     org
  -output                    lh
/gobj_maskaimpeel_ow
  -input_output              seg
  -gobj_filename             disk:[dir]x1469636.gobj
/cl_nr_extract
  -input                     seg_box
  -output                    cort_seg
/gobj_maskaimpeel_ow
  -input_output              cort_seg
  -gobj_filename             disk:[dir]x1469636_cort_mask.gobj
/cl_nr_extract
  -input                     seg_box
  -output                    trab_seg
/gobj_maskaimpeel_ow
  -input_output              {trab_obj}
  -gobj_filename             disk:[dir]x1469636_trab_mask.gobj
{trab_body}!% job terminated at  {end}
"""


# a delivered map's processing log: the chain it inherited from the segmentation that produced it
def _product(fill, mask_after_extract, stamp):
    chain = ["ISQ_TO_AIM", "D3P_BoundingBoxCut()", "IPL_ProcStd_OffsetAdd()"]
    if fill:
        chain.append("D3P_FillOffsetDuplicate()")
    chain += ["D3P_FFT_LaplaceHamming_M()", "D3P_Std_FloatNormMax_M()", "D3P_SupThreshold()",
              "D3P_GobjOrAimMaskAimPeel_OW()", "D3P_BoundingBoxCut()", "D3P_Cl_Label_CPP()", "D3P_Cl_ExtractNumber_CPP()"]
    if mask_after_extract:
        chain.append("D3P_GobjOrAimMaskAimPeel_OW()")
    chain.append("D3P_Dt_Obj")
    return f"Time                          {stamp}\n" + "\n".join(f"Procedure:                    {c}" for c in chain)


RUN1 = dict(start="1-JAN-2000 10:00:00.00", end="1-JAN-2000 10:10:00.00")
RUN2 = dict(start="2-JAN-2000 11:00:00.00", end="2-JAN-2000 11:10:00.00")


@pytest.fixture()
def two_run_folder(tmp_path):
    """A measurement folder with both evaluation logs and no SEG, as Diaphyseal/BMAT/610892 ships them."""
    (tmp_path / "EVAL_LH_1_610892.LOG").write_text(
        _log(RUN1["start"], RUN1["end"], "org", True, True), encoding="latin-1")
    (tmp_path / "EVAL_LH__1_610892.LOG").write_text(
        _log(RUN2["start"], RUN2["end"], "cort", False, False), encoding="latin-1")
    return tmp_path


# ----------------------------------------------------------------------------- the log parser
def test_parse_eval_log_reads_the_two_command_failures(two_run_folder):
    pr = vd.parse_eval_log(str(two_run_folder / "EVAL_LH__1_610892.LOG"))
    assert pr["lh_border"] == "none"
    assert pr["trab_seg_mask"] == "none"
    assert pr["seg_variant"] == "periosteal_first"
    assert pr["job_start"] == RUN2["start"] and pr["job_end"] == RUN2["end"]
    # the evidence points at the lines, and names the object each command could not find
    assert pr["evidence"]["fill_offset_duplicate"]["input"] == "cort"
    assert pr["evidence"]["fill_offset_duplicate"]["ok"] is False
    assert "cort" in pr["evidence"]["fill_offset_duplicate"]["error"]
    assert pr["evidence"]["trab_gobj_mask"]["input_output"] == "trab_gauss"
    assert pr["evidence"]["trab_gobj_mask"]["ok"] is False
    assert pr["evidence"]["fill_offset_duplicate"]["line"] < pr["evidence"]["fft_laplace_hamming"]["line"]


def test_parse_eval_log_reads_a_run_where_both_commands_succeeded(two_run_folder):
    pr = vd.parse_eval_log(str(two_run_folder / "EVAL_LH_1_610892.LOG"))
    assert (pr["lh_border"], pr["trab_seg_mask"], pr["seg_variant"]) == ("duplicate", "gobj", "periosteal_first")
    assert pr["evidence"]["trab_gobj_mask"]["set_line"].endswith("of total 4000000")


def test_fill_offset_duplicate_on_another_object_is_not_a_border(tmp_path):
    """The command only gives the Laplace-Hamming input a border when it ran on that very object."""
    p = tmp_path / "EVAL_x_1_2.LOG"
    p.write_text(_log(RUN1["start"], RUN1["end"], "cort", True, True), encoding="latin-1")
    pr = vd.parse_eval_log(str(p))
    assert pr["lh_border"] == "none"
    assert pr["evidence"]["fill_offset_duplicate"]["applied_to_lh_input"] is False


# ----------------------------------------------------------------------------- choosing the run
def test_choose_eval_log_follows_the_delivered_files(two_run_folder):
    products = {"map:TRAB_SP": _product(False, False, "1-JAN-2000 10:00:00.50"),
                "map:CORT_TH": "Created by  D3P_GobjCreateAimPeel (IPL)\nTime  2-JAN-2000 11:00:00.50"}
    pr, report = vd.choose_eval_log(vd.find_eval_logs(str(two_run_folder)), products=products)
    assert report["chosen"] == "EVAL_LH__1_610892.LOG"
    assert pr["lh_border"] == "none" and pr["trab_seg_mask"] == "none"
    # the run that had already terminated when CORT_TH recorded its last step is refuted by that alone
    rows = {r["file"]: r for r in report["candidates"]}
    assert rows["EVAL_LH_1_610892.LOG"]["refuted_by_date"] is True
    assert rows["EVAL_LH__1_610892.LOG"]["refuted_by_date"] is False
    assert rows["EVAL_LH__1_610892.LOG"]["chain_disagreements"] == 0
    assert rows["EVAL_LH_1_610892.LOG"]["chain_agreements"] == 0


def test_choose_eval_log_would_follow_the_other_run_if_the_products_said_so(two_run_folder):
    """Nothing here is hard-wired to the second run: products carrying the first run's chain select the first."""
    products = {"map:TRAB_SP": _product(True, True, "1-JAN-2000 10:00:00.50")}
    _, report = vd.choose_eval_log(vd.find_eval_logs(str(two_run_folder)), products=products)
    assert report["chosen"] == "EVAL_LH_1_610892.LOG"


# ----------------------------------------------------------------------------- detect_variants
def test_detect_variants_without_seg_takes_them_from_the_evaluation_log(two_run_folder):
    products = {"map:TRAB_SP": _product(False, False, "1-JAN-2000 10:00:00.50"),
                "map:CORT_TH": "Created by  D3P_GobjCreateAimPeel (IPL)\nTime  2-JAN-2000 11:00:00.50"}
    var = vd.detect_variants(None, vd.LAYOUTS["oslh"], folder=str(two_run_folder), base="X1469636", products=products)
    assert var["source"] == "evaluation_log"
    assert var["lh_border"] == "none"
    assert var["trab_seg_mask"] == "none"
    assert var["seg_variant"] == "periosteal_first"
    assert var["evidence"]["eval_log_file"] == "EVAL_LH__1_610892.LOG"
    assert any("evaluation log" in n for n in var["notes"])


@pytest.mark.parametrize("fill,mask_after,border,variant,tsmask", [
    (True, True, "duplicate", "gobj_first", "gobj"),
    (False, True, "none", "periosteal_first", "gobj"),
    (False, False, "none", "periosteal_first", "none"),
])
def test_detect_variants_with_a_seg_ignores_the_evaluation_log(two_run_folder, fill, mask_after, border, variant, tsmask):
    """The SEG's processing log stays the only source when a SEG exists, unaffected by an evaluation log in the same
    folder (the cohort's two 581203 folders ship both) -- and it now decides trab_seg_mask as well, from the same fact
    the evaluation log decides it by: whether a D3P_GobjOrAimMaskAimPeel_OW follows D3P_Cl_ExtractNumber_CPP."""
    pl = _product(fill, mask_after, "3-JAN-2000 12:00:00.00")
    if variant == "gobj_first":                                      # the 2022 order: labelling before the gobj mask
        pl = pl.replace("D3P_GobjOrAimMaskAimPeel_OW()\nProcedure:                    D3P_BoundingBoxCut()\n"
                        "Procedure:                    D3P_Cl_Label_CPP()",
                        "D3P_Cl_Label_CPP()\nProcedure:                    D3P_BoundingBoxCut()\n"
                        "Procedure:                    D3P_GobjOrAimMaskAimPeel_OW()")
    var = vd.detect_variants({"proclog": pl}, vd.LAYOUTS["oslh"], folder=str(two_run_folder), base="X1469636",
                             products={"map:TRAB_SP": pl})
    assert var["source"] == "seg_proclog"
    assert var["lh_border"] == border
    assert var["seg_variant"] == variant
    assert var["trab_seg_mask"] == tsmask
    assert "eval_log" not in var["evidence"]
    assert var["evidence"]["trab_gobj_mask_absent_from_seg_proclog"] is (not mask_after)
    assert var["evidence"]["gobj_mask_after_extract"] is mask_after
    if not mask_after:
        assert any("was never cropped" in n or "raw component extraction" in n for n in var["notes"])


def test_detect_variants_seg_proclog_needs_the_mask_AFTER_the_extraction(two_run_folder):
    """A gobj mask that only precedes the labelling is the PERIOSTEAL one; it does not make TRAB_SEG cropped.
    This is the case the whole rule turns on, so it is pinned separately from the parametrised sweep above."""
    pl = _product(False, False, "3-JAN-2000 12:00:00.00")
    assert pl.count("D3P_GobjOrAimMaskAimPeel_OW") == 1                      # the periosteal mask, before the labelling
    assert pl.find("D3P_GobjOrAimMaskAimPeel_OW") < pl.find("D3P_Cl_ExtractNumber_CPP")
    var = vd.detect_variants({"proclog": pl}, vd.LAYOUTS["oslh"], folder=str(two_run_folder), base="X1469636")
    assert var["trab_seg_mask"] == "none"
    assert var["evidence"]["gobj_mask_entries"] == 1
    assert var["evidence"]["gobj_mask_after_extract_index"] == -1 or \
           var["evidence"]["gobj_mask_after_extract_index"] < 0


def test_detect_variants_trab_seg_mask_override_still_wins(two_run_folder):
    """A layout that pins trab_seg_mask keeps it (the patella layout does), and --trab-seg-mask overrides the
    detected value at the call site; the detected evidence is recorded either way."""
    pl = _product(False, False, "3-JAN-2000 12:00:00.00")
    layout = dict(vd.LAYOUTS["oslh"], trab_seg_mask="gobj")
    var = vd.detect_variants({"proclog": pl}, layout, folder=str(two_run_folder), base="X1469636")
    assert var["trab_seg_mask"] == "gobj"
    assert var["evidence"]["gobj_mask_after_extract"] is False


def test_detect_variants_products_may_be_a_callable_and_is_not_called_when_a_seg_exists(two_run_folder):
    calls = []

    def products():
        calls.append(1)
        return {}

    vd.detect_variants({"proclog": _product(True, True, "3-JAN-2000 12:00:00.00")}, vd.LAYOUTS["oslh"],
                       folder=str(two_run_folder), base="X1469636", products=products)
    assert calls == [], "reading the delivered maps must not be forced on a measurement that has a SEG"
    vd.detect_variants(None, vd.LAYOUTS["oslh"], folder=str(two_run_folder), base="X1469636", products=products)
    assert calls == [1]


def test_detect_variants_rejects_a_log_that_segments_another_measurement(two_run_folder):
    """A log in the folder is only believed when its segmentation reads THIS measurement's greyscale."""
    var = vd.detect_variants(None, vd.LAYOUTS["oslh"], folder=str(two_run_folder), base="X0000000", products={})
    assert var["source"] == "default"
    assert (var["lh_border"], var["seg_variant"], var["trab_seg_mask"]) == ("duplicate", "periosteal_first", "gobj")
    assert "x1469636.aim" in var["evidence"]["rejected"]


def test_detect_variants_falls_back_to_the_cohort_default_without_seg_or_log(tmp_path):
    var = vd.detect_variants(None, vd.LAYOUTS["oslh"], folder=str(tmp_path), base="X1469636")
    assert var["source"] == "default"
    assert (var["lh_border"], var["seg_variant"], var["trab_seg_mask"]) == ("duplicate", "periosteal_first", "gobj")


def test_detect_variants_with_a_seg_but_no_proclog_never_reads_an_evaluation_log(two_run_folder):
    """A measurement that ships a SEG keeps the pre-fix behaviour unconditionally.  No measurement of either cohort
    has an empty SEG processing log, but the branch must not be reachable from a SEG at all: the evaluation log is
    for a measurement that ships NO SEG, and this closes the one path by which a SEG-shipping measurement could
    have been changed silently."""
    var = vd.detect_variants({"proclog": ""}, vd.LAYOUTS["oslh"], folder=str(two_run_folder), base="X1469636",
                             products={"map:TRAB_SP": _product(False, False, "2-JAN-2000 11:00:00.50")})
    assert var["source"] == "default"
    assert (var["lh_border"], var["seg_variant"], var["trab_seg_mask"]) == ("duplicate", "periosteal_first", "gobj")
    assert "eval_log" not in var["evidence"] and var["evidence"]["seg_without_proclog"] is True


def test_reporting_defaults_fill_the_fields_a_record_written_before_them_lacks():
    """flat_row / the summary counters must not read 'gobj': 0 for records that predate the field: assemble_seg had
    a single, masked trabecular branch when they were written, and their own evidence says which source they used."""
    old_seg = dict(variants=dict(lh_border="none", seg_variant="periosteal_first",
                                 evidence=dict(fill_offset_duplicate=False, gobj_mask_index=10, cl_label_index=20)))
    old_no_seg = dict(variants=dict(lh_border="duplicate", seg_variant="periosteal_first",
                                    evidence=dict(no_seg_proclog=True)))
    new = dict(variants=dict(lh_border="none", seg_variant="periosteal_first", trab_seg_mask="none",
                             source="evaluation_log", evidence={}))
    assert (vd._variants_trab_seg_mask(old_seg), vd._variants_source(old_seg)) == ("gobj", "seg_proclog")
    assert (vd._variants_trab_seg_mask(old_no_seg), vd._variants_source(old_no_seg)) == ("gobj", "default")
    assert (vd._variants_trab_seg_mask(new), vd._variants_source(new)) == ("none", "evaluation_log")


def test_layout_pins_win_over_the_evaluation_log(two_run_folder):
    """A layout that fixes a variant (the patella one does) keeps it; the log is still parsed and recorded."""
    layout = dict(vd.LAYOUTS["oslh"], lh_border="duplicate", trab_seg_mask="gobj")
    var = vd.detect_variants(None, layout, folder=str(two_run_folder), base="X1469636", products={})
    assert var["lh_border"] == "duplicate" and var["trab_seg_mask"] == "gobj"
    assert var["evidence"]["fill_offset_duplicate"]["ok"] is False


# ----------------------------------------------------------------------------- assemble_seg
def _slab_inputs():
    """A 20^3 greyscale grid whose thresholded volume is one 10x10x10 slab, with the cortical contour over its
    left half and the trabecular contour over its right half."""
    dim_g, pos_g = (20, 20, 20), (0, 0, 0)
    dim_e, pos_e = (22, 22, 22), (-1, -1, -1)
    bm = np.zeros(dim_e[::-1], bool)
    bm[5:15, 5:15, 5:15] = True
    per = vd.bvol(np.ones(dim_e[::-1], bool), dim_e, pos_e)
    cort = np.zeros(dim_e[::-1], bool)
    cort[:, :, :11] = True
    trab = np.zeros(dim_e[::-1], bool)
    trab[:, :, 11:] = True
    return bm, (dim_g, pos_g), per, vd.bvol(cort, dim_e, pos_e), vd.bvol(trab, dim_e, pos_e)


@pytest.mark.parametrize("variant", ["periosteal_first", "gobj_first"])
def test_assemble_seg_trab_masked_flag_only_moves_the_trabecular_branch(variant):
    bm, grid, per, cort_g, trab_g = _slab_inputs()
    masked = vd.assemble_seg(bm, grid, per, cort_g, trab_g, variant, trab_masked=True)
    raw = vd.assemble_seg(bm, grid, per, cort_g, trab_g, variant, trab_masked=False)
    assert masked["trab_seg_mask"] == "gobj" and raw["trab_seg_mask"] == "none"
    # the cortical compartment is untouched (the slab's 6 columns inside the cortical contour)
    assert vd.count(masked["CORT_SEG"]) == vd.count(raw["CORT_SEG"]) == 600
    # masked: the 4 columns inside the trabecular contour; unmasked: the whole component extraction
    assert vd.count(masked["TRAB_SEG"]) == 400
    assert vd.count(raw["TRAB_SEG"]) == 1000
    # and SEG gains exactly the voxels the 127 cortical label did not already claim -- here none, because the two
    # contours tile the volume, so the delivered SEG is the same either way
    assert int((masked["SEG"]["data"] != 0).sum()) == int((raw["SEG"]["data"] != 0).sum()) == 1000


def test_assemble_seg_default_is_masked():
    bm, grid, per, cort_g, trab_g = _slab_inputs()
    a = vd.assemble_seg(bm, grid, per, cort_g, trab_g, "periosteal_first")
    b = vd.assemble_seg(bm, grid, per, cort_g, trab_g, "periosteal_first", trab_masked=True)
    assert a["trab_seg_mask"] == "gobj"
    assert np.array_equal(a["TRAB_SEG"]["data"], b["TRAB_SEG"]["data"])


# ----------------------------------------------------------------------------- the real cohort
@pytest.mark.slow
def test_real_610892_reads_its_variants_from_the_second_evaluation_run():
    folder = os.path.join(COHORT, "Diaphyseal", "BMAT", "610892")
    if not os.path.isdir(folder):
        pytest.skip(f"OS_LH cohort not mounted: {folder}")
    import ipldt
    products = {f"map:{m}": ipldt.read_aim(os.path.join(folder, f"X1469636_{m}_COMPRESSED.AIM"))["proclog"]
                for m in ("TRAB_TH", "TRAB_SP", "TRAB_1N", "CORT_TH")}
    var = vd.detect_variants(None, vd.LAYOUTS["oslh"], folder=folder, base="X1469636", products=products)
    assert var["source"] == "evaluation_log"
    second = [os.path.basename(p) for p in vd.find_eval_logs(folder) if "__" in os.path.basename(p)]
    assert len(second) == 1 and var["evidence"]["eval_log_file"] == second[0]     # the double-underscore (second) run
    assert (var["lh_border"], var["seg_variant"], var["trab_seg_mask"]) == ("none", "periosteal_first", "none")
    assert var["evidence"]["fill_offset_duplicate"]["line"] == 221
    assert var["evidence"]["trab_gobj_mask"]["line"] == 434


@pytest.mark.slow
def test_real_581203_ships_both_evaluation_logs_and_still_uses_its_seg():
    """581203 is the cross-check between the two sources: it ships BOTH evaluation logs AND a SEG.  Its SEG processing
    log must be the source, and it must reach the same answer the failed run's transcript does -- lh_border 'none'
    and trab_seg_mask 'none' -- because the delivered SEG came from that run (its CORT_TH's printed 2.126924 is the
    later run's, not the first run's 2.125546)."""
    folder = os.path.join(COHORT, "Diaphyseal", "CKD", "581203")
    if not os.path.isdir(folder):
        pytest.skip(f"OS_LH cohort not mounted: {folder}")
    import ipldt
    assert len(vd.find_eval_logs(folder)) == 2
    seg = ipldt.read_aim(os.path.join(folder, "X9114091_SEG_DECOMPRESSED.AIM"))
    var = vd.detect_variants(seg, vd.LAYOUTS["oslh"], folder=folder, base="X9114091")
    assert var["source"] == "seg_proclog"
    assert (var["lh_border"], var["seg_variant"], var["trab_seg_mask"]) == ("none", "periosteal_first", "none")
    assert var["evidence"]["fill_offset_duplicate"] is False
    assert var["evidence"]["gobj_mask_after_extract"] is False
    assert "eval_log" not in var["evidence"]


@pytest.mark.slow
def test_real_cohort_trab_seg_mask_detection_is_the_eleven_and_nothing_else():
    """The detection over the whole cohort: 'gobj' for every measurement whose SEG log carries the masking entry and
    'none' for exactly the eleven measurements whose log does not (the ten plus 610892, which ships no SEG and is read
    from its evaluation log; rerun/581203 is the duplicate folder of Diaphyseal/CKD/581203 and is not in the validated
    117).  The two halves of the same failure never disagree: D3P_FillOffsetDuplicate is present exactly where the
    masking entry is."""
    if not os.path.isdir(COHORT):
        pytest.skip(f"OS_LH cohort not mounted: {COHORT}")
    import struct

    def proclog(path):
        with open(path, "rb") as fh:
            head = fh.read(20)
            hsize, lsize = struct.unpack("<5i", head)[1:3]
            body = fh.read(hsize + lsize)
        return body[hsize:hsize + lsize].decode("latin-1", "replace")

    expected_none = {"Diaphyseal/BMAT/610892", "Diaphyseal/CKD/2422", "Diaphyseal/CKD/558517", "Diaphyseal/CKD/449824",
                     "Diaphyseal/CKD/581203", "Diaphyseal/CKD/433045", "Diaphyseal/CKD/341269", "Distal/CKD/386723",
                     "Distal/CKD/802246", "Distal/REPRO/472790", "Distal/REPRO/512830", "rerun/581203"}
    # the ids above are this tree's measurement pseudonyms: a cohort laid out under the scanner's own measurement
    # numbers cannot be compared with them (and a failure would print those numbers), so the test skips
    if not all(os.path.isdir(os.path.join(COHORT, *i.split("/"))) for i in expected_none):
        pytest.skip("the OS_LH cohort is not laid out under the measurement pseudonyms of this tree")
    got, mixed = set(), []
    for m in vd.discover("oslh", COHORT, None, None):
        names = {f.upper(): f for f in os.listdir(m.folder)}
        seg = next((os.path.join(m.folder, names[p.format(base=m.base).upper()])
                    for p in m.L["seg"] if p.format(base=m.base).upper() in names), None)
        var = vd.detect_variants({"proclog": proclog(seg)} if seg else None, m.L, folder=m.folder, base=m.base,
                                 products=lambda: {})
        if var["trab_seg_mask"] == "none":
            got.add(m.id)
        ev = var["evidence"]
        if seg and bool(ev["fill_offset_duplicate"]) != bool(ev["gobj_mask_after_extract"]):
            mixed.append(m.id)
    assert got == expected_none
    assert mixed == [], "the two halves of the failed run must never disagree"


@pytest.mark.slow
def test_real_cohort_gobj_mask_names_agree_with_the_positional_rule():
    """The SEG processing log NAMES the gobj of every D3P_GobjOrAimMaskAimPeel_OW, so the trabecular crop can be
    identified by name as well as by position.  Over all 122 OS_LH SEG logs the two readings must agree: every
    measurement with a post-extract entry names a '_trab_mask.gobj', and no '_trab_mask.gobj' appears anywhere in
    the logs of those that have none (their only entry names the periosteal '<base>.gobj').  This is what closes
    the positional rule's one stated ambiguity -- a log whose sole late masking entry were a periosteal mask."""
    if not os.path.isdir(COHORT):
        pytest.skip(f"OS_LH cohort not mounted: {COHORT}")
    import struct

    def proclog(path):
        with open(path, "rb") as fh:
            head = fh.read(20)
            hsize, lsize = struct.unpack("<5i", head)[1:3]
            body = fh.read(hsize + lsize)
        return body[hsize:hsize + lsize].decode("latin-1", "replace")

    n_gobj = n_none = 0
    disagree = []
    for m in vd.discover("oslh", COHORT, None, None):
        names = {f.upper(): f for f in os.listdir(m.folder)}
        seg = next((os.path.join(m.folder, names[p.format(base=m.base).upper()])
                    for p in m.L["seg"] if p.format(base=m.base).upper() in names), None)
        if seg is None:
            continue
        var = vd.detect_variants({"proclog": proclog(seg)}, m.L, folder=m.folder, base=m.base, products=lambda: {})
        ev = var["evidence"]
        assert ev["gobj_mask_files"], f"{m.id}: no gobj file name parsed out of the masking block(s)"
        if ev["gobj_mask_names_trab_mask"] != ev["gobj_mask_after_extract"]:
            disagree.append((m.id, ev["gobj_mask_files"]))
        if var["trab_seg_mask"] == "gobj":
            n_gobj += 1
            assert any("_trab_mask.gobj" in g for g in ev["gobj_mask_files"]), m.id
        else:
            n_none += 1
            assert not any("_trab_mask.gobj" in g for g in ev["gobj_mask_files"]), m.id
            assert all(g.endswith(f"{m.base.lower()}.gobj") for g in ev["gobj_mask_files"]), (m.id, ev["gobj_mask_files"])
    assert disagree == [], f"name and position must agree on every SEG log: {disagree}"
    assert (n_gobj, n_none) == (111, 11)
