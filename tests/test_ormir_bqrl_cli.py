"""ormir_bqrl.cli -- the `ormir-bqrl` command line.

In-process: `version` and every `--help` (exit 0), argparse usage errors (exit 2), a missing input (exit 1 with
`Error:` on stderr), `slicer-export` on a fake run folder, `find_periosteal`; one subprocess for the module entry
point (`python -m ormir_bqrl.cli version`); and, when ormir_bqrl.pipeline is importable together with SimpleITK,
itk and ormir_xct, one `run` on a synthetic AIM with an injected periosteal disc (the autocontour is never called)."""
import os
import struct
import subprocess
import sys

import numpy as np
import pytest

from ipldt.io import TYPE_SHORT, write_nifti                         # noqa: E402
from ormir_bqrl import cli                                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


# ------------------------------------------------------------------------------------------ helpers
def _vax_pack(x):
    import math
    if x == 0:
        return 0
    m, e = math.frexp(x)
    f = int(round((m - 0.5) * 2 ** 24))
    w1 = ((e + 128) << 7) | (f >> 16)
    w2 = f & 0xFFFF
    u = (w2 << 16) | w1
    return u - 2 ** 32 if u >= 2 ** 31 else u


PROCLOG = ("!\n! Processing Log\n!\n"
           "Mu_Scaling                                       8192\n"
           "Mu_Water                                       0.2409\n"
           "Density: unit                                  mg HA/ccm\n"
           "Density: slope                         1.61907703e+03\n"
           "Density: intercept                    -3.94095001e+02\n").encode("latin-1")
SHAPE = (40, 90, 90)              # (z, y, x)
POS = (100, 200, 300)
EL = (0.06069973, 0.06069973, 0.06069973)
CENTRE, R_OUT, R_IN = 44.5, 40.0, 32.0


def phantom_native(seed=0):
    """The step-3 cylinder (dense shell 32 <= r <= 40 at native 12000, marrow 1000, outside 300) with a rod /
    plate structure of native 8000 inside r < 28 so that the Laplace-Hamming threshold has something to find."""
    from scipy import ndimage as ndi
    yy, xx = np.mgrid[:SHAPE[1], :SHAPE[2]]
    r = np.sqrt((yy - CENTRE) ** 2 + (xx - CENTRE) ** 2)
    grey2d = np.where(r <= R_OUT, np.where(r >= R_IN, 12000, 1000), 300).astype(np.int16)
    vol = np.broadcast_to(grey2d, SHAPE).copy()
    rng = np.random.default_rng(seed)
    f = ndi.gaussian_filter(rng.standard_normal(SHAPE), 2.0)
    rods = (f > np.quantile(f, 0.70)) & np.broadcast_to(r < 28.0, SHAPE)
    vol[rods] = 8000
    return vol, (r <= R_OUT)


def write_synthetic_aim(path, data_zyx, pos=POS, el=EL, proclog=PROCLOG):
    """A v020 short AIM in the layout ipldt.io.read_aim, itk.ScancoImageIO and ormir_xct's file_reader read."""
    data = np.ascontiguousarray(data_zyx, dtype=np.int16)
    dim = data.shape[::-1]
    ints = [0] * 35
    ints[0] = 16
    ints[5] = TYPE_SHORT
    ints[6:9] = pos
    ints[9:12] = dim
    ints[27:30] = [_vax_pack(e) for e in el]
    hdr = struct.pack("<35i", *ints)
    body = data.tobytes()
    raw = struct.pack("<5i", 20, len(hdr), len(proclog), len(body), 0) + hdr + proclog + body
    raw += b"\0" * ((-len(raw)) % 512)
    open(path, "wb").write(raw)
    return path


# ------------------------------------------------------------------------------------------ in-process
def test_version_prints_the_stack(capsys):
    assert cli.main(["version"]) == 0
    out = capsys.readouterr().out
    for name in ("ORMIR-BQRL 0.1.0", "ipldt ", "ormir_xct ", "SimpleITK ", "itk ", "CuPy ", "Python "):
        assert name in out, name
    assert "ipldt not available" not in out


@pytest.mark.parametrize("argv", [["--help"], ["run", "--help"], ["redo", "--help"], ["slicer-export", "--help"], ["batch", "--help"],
                                  ["version", "--help"], ["--version"]])
def test_help_exits_zero(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "ormir-bqrl" in out or "ORMIR-BQRL" in out


@pytest.mark.parametrize("argv", [[], ["run"], ["run", "a.AIM", "out", "--site", "femur"], ["nonsense"], ["redo", "a.AIM"],
                                  ["run", "a.AIM", "out", "--backend", "tpu"]])
def test_usage_errors_exit_two(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_missing_inputs_exit_one_with_error_on_stderr(tmp_path, capsys):
    assert cli.main(["slicer-export", str(tmp_path / "nowhere")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error:") and "nowhere" in err
    assert cli.main(["run", str(tmp_path / "missing.AIM"), str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error:")
    assert cli.main(["redo", str(tmp_path / "missing.AIM"), str(tmp_path)]) == 1
    assert capsys.readouterr().err.startswith("Error:")


def test_slicer_export_command(tmp_path, capsys):
    sitk = pytest.importorskip("SimpleITK")
    from ormir_bqrl.slicer import read_mask, read_seg_nrrd, GridSpec
    z, y, x = 6, 20, 24
    yy, xx = np.mgrid[:y, :x]
    r = np.sqrt((yy - (y - 1) / 2) ** 2 + (xx - (x - 1) / 2) ** 2)
    cort = np.broadcast_to((r <= 8) & (r >= 5), (z, y, x)).copy()
    trab = np.broadcast_to(r < 5, (z, y, x)).copy()
    el, pos = (0.0607, 0.0607, 0.0607), (10, 20, 30)
    run_dir = tmp_path / "S7"
    run_dir.mkdir()
    write_nifti(str(run_dir / "S7_CORT_MASK.nii.gz"), cort.astype(np.uint8) * 127, el, pos)
    write_nifti(str(run_dir / "S7_TRAB_MASK.nii.gz"), trab.astype(np.uint8) * 127, el, pos)
    assert cli.main(["slicer-export", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "S7:" in out and "S7_compartments.seg.nrrd" in out and "S7_compartments_labelmap.nii.gz" in out
    seg = run_dir / "S7_compartments.seg.nrrd"
    lm = run_dir / "S7_compartments_labelmap.nii.gz"
    assert seg.is_file() and lm.is_file()
    labels = read_seg_nrrd(str(seg))[0]
    assert np.array_equal(labels, cort * 1 + trab * 2)
    grid = GridSpec((x, y, z), pos, el)
    assert np.array_equal(read_mask(str(lm), grid, "cort")[0], cort) and np.array_equal(read_mask(str(seg), grid, "trab")[0], trab)
    assert np.array_equal(sitk.GetArrayFromImage(sitk.ReadImage(str(seg))), labels)
    # --out elsewhere, identical files
    assert cli.main(["slicer-export", str(run_dir), "--out", str(tmp_path / "elsewhere"), "--base", "S7"]) == 0
    assert np.array_equal(read_seg_nrrd(str(tmp_path / "elsewhere" / "S7_compartments.seg.nrrd"))[0], labels)


def test_find_periosteal(tmp_path):
    assert cli.find_periosteal(None, "S1") is None
    assert cli.find_periosteal(str(tmp_path), "S1") is None
    open(tmp_path / "S1.AIM", "wb").write(b"")
    assert cli.find_periosteal(str(tmp_path), "S1") == str(tmp_path / "S1.AIM")
    open(tmp_path / "S1_PRX_MASK.nii.gz", "wb").write(b"")
    assert cli.find_periosteal(str(tmp_path), "S1") == str(tmp_path / "S1_PRX_MASK.nii.gz")


def test_summary_row_fallback_and_print(capsys):
    report = dict(sample="S1", site="tibia", run=dict(kind="run", out_dir="/o"), summary=dict(BV_TV=0.25, Tb_Th_mm=0.2),
                  bmd=dict(Tb_BMD_mgHA_cm3=180.0), masks=dict(periosteal=dict(source="file"), cortical=dict(source="ipldt.step1"),
                                                            trabecular=dict(source="ipldt.step1")))
    row = cli._summary_row(report)
    assert row["sample"] == "S1" and row["BV_TV"] == 0.25
    cli.print_summary(report, "/o")
    out = capsys.readouterr().out
    assert "BV_TV: 0.250000" in out and "Tb_BMD_mgHA_cm3: 180.000000" in out and "periosteal file" in out and "S1_report.json" in out


def test_run_and_redo_record_the_command_line(tmp_path, monkeypatch):
    """The CLI passes its argv to run() / run_from_masks() as command= (report['run']['command'])."""
    aim = tmp_path / "S.AIM"
    aim.write_bytes(b"\0" * 64)
    seen = {}

    def fake_run(aim_path, out_dir, **kw):
        seen["run"] = kw.get("command")
        return dict(sample="S", site="tibia", run=dict(kind="run", out_dir=str(out_dir)), summary={}, masks={}, outputs={})

    def fake_redo(aim_path, run_dir, **kw):
        seen["redo"] = kw.get("command")
        return dict(sample="S", site="tibia", run=dict(kind="redo", out_dir=str(run_dir)), summary={}, masks={}, outputs={})

    monkeypatch.setattr(cli, "_import_pipeline", lambda: fake_run)
    monkeypatch.setattr(cli, "_import_redo", lambda: fake_redo)
    argv = ["run", str(aim), str(tmp_path / "out"), "--no-bmd"]
    assert cli.main(argv) == 0
    assert seen["run"] == ["ormir-bqrl", *argv]
    argv = ["redo", str(aim), str(tmp_path), "--trab", str(aim)]
    assert cli.main(argv) == 0
    assert seen["redo"] == ["ormir-bqrl", *argv]


# ------------------------------------------------------------------------------------------ the entry point
def test_module_entry_point_subprocess():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-m", "ormir_bqrl.cli", "version"], capture_output=True, text=True, env=env, cwd=REPO, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("ORMIR-BQRL 0.1.0")


# ------------------------------------------------------------------------------------------ a run on the phantom
def _pipeline_available():
    for mod in ("SimpleITK", "itk", "ormir_xct"):
        try:
            __import__(mod)
        except ImportError:
            return False
    try:
        import ormir_bqrl.pipeline  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _pipeline_available(), reason="ormir_bqrl.pipeline / SimpleITK / itk / ormir_xct not importable")
def test_run_on_the_synthetic_phantom(tmp_path, capsys):
    import json
    vol, disc = phantom_native()
    aim = write_synthetic_aim(str(tmp_path / "PHANTOM.AIM"), vol)
    prx = str(tmp_path / "PHANTOM_PRX_MASK.nii.gz")
    write_nifti(prx, np.broadcast_to(disc, SHAPE).astype(np.uint8) * 127, EL, POS)
    out = tmp_path / "out"
    rc = cli.main(["run", aim, str(out), "--site", "tibia", "--periosteal", prx, "--no-bmd", "--no-preview", "--backend", "cpu"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    report_path = out / "PHANTOM_report.json"
    assert report_path.is_file(), os.listdir(out)
    rep = json.load(open(report_path))
    assert rep.get("sample") == "PHANTOM"
    for name in ("PHANTOM_SEG.nii.gz", "PHANTOM_CORT_MASK.nii.gz", "PHANTOM_TRAB_MASK.nii.gz", "PHANTOM_compartments.seg.nrrd",
                 "PHANTOM_compartments_labelmap.nii.gz"):
        assert (out / name).is_file(), name
    assert "summary" in captured.out and "Tb_Th_mm" in captured.out
    # the Slicer files agree with the masks
    from ormir_bqrl.slicer import GridSpec, read_mask, read_seg_nrrd
    grid = GridSpec(SHAPE[::-1], POS, EL)
    cort = read_mask(str(out / "PHANTOM_CORT_MASK.nii.gz"), grid)[0]
    trab = read_mask(str(out / "PHANTOM_TRAB_MASK.nii.gz"), grid)[0]
    labels = read_seg_nrrd(str(out / "PHANTOM_compartments.seg.nrrd"))[0]
    assert np.array_equal(labels == 1, cort) and np.array_equal(labels == 2, trab)
    assert np.array_equal(read_mask(str(out / "PHANTOM_compartments_labelmap.nii.gz"), grid, "trab")[0], trab)
