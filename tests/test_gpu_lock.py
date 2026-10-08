"""The machine-wide GPU lock (ipldt.gpu._gpu_section): processes sharing a GPU take turns in ridge_gpu and
draw_spheres_gpu.  The lock decides only when a call runs; the maps are covered by test_gpu_cpu.py and
test_gpu_stamp_schedule.py, which run with it on (the default).  Here: on by default and off with
IPLDT_GPU_LOCK=0, mutual exclusion between processes and between threads, re-entrancy, the lock of a holder that
died is taken over, and both GPU entry points wait for the device's lock while another process holds it.
"""
import contextlib
import os
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import pytest

import ipldt.gpu as gpu

from conftest import requires_gpu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def _name():
    """A lock name no device uses, unique to this test."""
    return f"ipldt-test-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _python(code, *args):
    return [sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); " + code, ROOT, *args]


def test_lock_is_on_by_default_and_off_with_the_variable(monkeypatch):
    monkeypatch.delenv("IPLDT_GPU_LOCK", raising=False)
    assert gpu.gpu_lock_enabled()
    for off in ("0", "false", "No", " off "):
        monkeypatch.setenv("IPLDT_GPU_LOCK", off)
        assert not gpu.gpu_lock_enabled()
    monkeypatch.setenv("IPLDT_GPU_LOCK", "1")
    assert gpu.gpu_lock_enabled()


def test_switched_off_lock_never_reaches_the_os(monkeypatch):
    monkeypatch.setenv("IPLDT_GPU_LOCK", "0")
    monkeypatch.setattr(gpu, "_os_lock", lambda name: pytest.fail("locked although IPLDT_GPU_LOCK=0"))
    with gpu._gpu_section(_name()):
        pass


def test_processes_take_turns(monkeypatch):
    """Three processes each hold the lock for 0.3 s: their sections never overlap."""
    monkeypatch.delenv("IPLDT_GPU_LOCK", raising=False)
    name = _name()
    code = ("import time, ipldt.gpu as g\n"
            "with g._gpu_section(sys.argv[2]):\n"
            "    t0 = time.time(); time.sleep(0.3); t1 = time.time()\n"
            "print(t0, t1)")
    procs = [subprocess.Popen(_python(code, name), stdout=subprocess.PIPE, text=True, env=ENV) for _ in range(3)]
    spans = sorted(tuple(map(float, p.communicate(timeout=120)[0].split())) for p in procs)
    assert all(p.returncode == 0 for p in procs) and len(spans) == 3
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end - 0.005, f"two processes held the lock at once: {spans}"


def test_threads_take_turns_and_the_lock_is_reentrant(monkeypatch):
    monkeypatch.delenv("IPLDT_GPU_LOCK", raising=False)
    name, spans = _name(), []

    def work():
        with gpu._gpu_section(name):
            with gpu._gpu_section(name):                  # nested: no deadlock
                t0 = time.time()
                time.sleep(0.15)
                spans.append((t0, time.time()))
    threads = [threading.Thread(target=work) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    spans.sort()
    assert len(spans) == 3
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end - 0.005, f"two threads held the lock at once: {spans}"
    assert gpu._LOCK_DEPTH[0] == 0


def test_lock_of_a_dead_holder_is_taken_over(monkeypatch):
    """A process that dies holding the lock (no release, no clean-up) does not block the next one."""
    monkeypatch.delenv("IPLDT_GPU_LOCK", raising=False)
    name = _name()
    holder = subprocess.Popen(_python("import os, ipldt.gpu as g\ng._os_lock(sys.argv[2]); print('held', flush=True)\n"
                                      "import time; time.sleep(1); os._exit(0)", name),
                              stdout=subprocess.PIPE, text=True, env=ENV)
    assert holder.stdout.readline().strip() == "held"
    holder.wait(timeout=60)
    taker = subprocess.run(_python("import ipldt.gpu as g\nwith g._gpu_section(sys.argv[2]):\n    print('ok')", name),
                           capture_output=True, text=True, env=ENV, timeout=120)
    assert taker.returncode == 0 and taker.stdout.strip() == "ok", taker.stderr


@requires_gpu
@pytest.mark.gpu
def test_gpu_entry_points_wait_for_the_device_lock(monkeypatch, phantom):
    """While another process holds this GPU's lock, ridge_gpu and draw_spheres_gpu wait for it."""
    monkeypatch.delenv("IPLDT_GPU_LOCK", raising=False)
    from ipldt.core import _RIDGE_TOL
    from ipldt.field import sir_quad
    assert gpu._lock_name().startswith("ipldt-gpu-")
    V = sir_quad(phantom)
    gpu.ridge_gpu(phantom, V, 0.9, _RIDGE_TOL)                         # warm up CuPy outside the timing
    calls = (lambda: gpu.ridge_gpu(phantom, V, 0.9, _RIDGE_TOL),
             lambda: gpu.draw_spheres_gpu(np.array([3]), np.array([4]), np.array([5]), np.array([6]), (8, 9, 10)))
    for call in calls:
        holder = subprocess.Popen(_python("import time, ipldt.gpu as g\nwith g._gpu_section(g._lock_name()):\n"
                                          "    print('held', flush=True); time.sleep(1.5)"),
                                  stdout=subprocess.PIPE, text=True, env=ENV)
        assert holder.stdout.readline().strip() == "held"
        t0 = time.time()
        call()
        waited = time.time() - t0
        holder.wait(timeout=60)
        assert waited > 0.8, f"the call ran while another process held the GPU lock ({waited:.2f} s)"


@requires_gpu
@pytest.mark.gpu
def test_gpu_entry_points_take_the_lock(monkeypatch, phantom):
    held = []
    real = gpu._gpu_section

    @contextlib.contextmanager
    def spy(name=None):
        with real(name):
            held.append(name)
            yield
    monkeypatch.setattr(gpu, "_gpu_section", spy)
    from ipldt.core import _RIDGE_TOL
    from ipldt.field import sir_quad
    gpu.ridge_gpu(phantom, sir_quad(phantom), 0.9, _RIDGE_TOL)
    gpu.draw_spheres_gpu(np.array([3]), np.array([4]), np.array([5]), np.array([6]), (8, 9, 10))
    gpu.draw_spheres_gpu(np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), (3, 4, 5))
    assert held == [None, None, None]
