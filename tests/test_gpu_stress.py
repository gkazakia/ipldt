"""Opt-in GPU stress test (Windows; about 10 minutes): four processes run dt_spacing + dt_number on the GPU on a
sparse volume (spheres up to D = 190) while the System event log is watched for NVIDIA driver events (provider
nvlddmkm, e.g. Xid 13 'Graphics FECS Exception'); at the first event every worker is killed and the test fails.

    set IPLDT_GPU_STRESS=1                 (optional: IPLDT_GPU_STRESS_MINUTES=10, IPLDT_GPU_STRESS_WORKERS=4)
    python -m pytest tests/test_gpu_stress.py -q

tools/gpu_stress.py runs the same check against any checkout (--root), e.g. an older release for comparison.
"""
import json
import os
import subprocess
import sys

import pytest

from conftest import requires_gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(sys.platform != "win32", reason="reads the Windows System event log")
@pytest.mark.skipif(os.environ.get("IPLDT_GPU_STRESS") != "1",
                    reason="opt-in: set IPLDT_GPU_STRESS=1 (about 10 minutes)")
def test_gpu_stress_no_driver_events(tmp_path):
    report = tmp_path / "gpu_stress.json"
    p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "gpu_stress.py"), "--root", ROOT,
                        "--workers", os.environ.get("IPLDT_GPU_STRESS_WORKERS", "4"),
                        "--minutes", os.environ.get("IPLDT_GPU_STRESS_MINUTES", "10"), "--report", str(report)],
                       capture_output=True, text=True)
    rep = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
    assert p.returncode == 0, f"{rep.get('status')}: {rep.get('event_details')}\n{p.stdout}\n{p.stderr}"
    assert rep["events"] == 0 and all(n > 0 for n in rep["worker_rounds"].values()), rep
