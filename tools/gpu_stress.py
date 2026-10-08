"""Opt-in GPU stress test for the sphere stamping (Windows): several processes run dt_spacing + dt_number on the GPU,
on a sparse volume whose marrow spheres reach D ~ 180, while this process watches the System event log for
NVIDIA driver events (provider nvlddmkm, e.g. Xid 13 'Graphics FECS Exception') and kills every worker at the
first one.

    python tools/gpu_stress.py [--root CHECKOUT] [--workers 4] [--minutes 10] [--report stress.json]

--root is the checkout whose ipldt the workers import (default: the one this file belongs to), so the same
harness can run an older release for comparison.  Exit status: 0 no driver event, 1 a driver event (workers
killed at once), 2 a worker failed, 3 the event log could not be read.  tests/test_gpu_stress.py runs it on
this checkout when IPLDT_GPU_STRESS=1.  The workers inherit the environment, so IPLDT_GPU_LOCK=0 runs them without
the cross-process GPU lock (ipldt.gpu._gpu_section).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
# The stress volume: 18 rods in 64 x 300 x 300 voxels (0.2 % object).  Its dt_spacing / dt_number spheres reach
# D = 190 with 1.5e10 box positions per map, 7.2e6 of them in the old kernel's longest thread: the scale of the
# sparse PCCT blocks that drew Xid 13 faults (D 179, 2.6e10, 5.9e6).  Every worker stamps the same volume.
VOLUME = dict(shape=(64, 300, 300), n_rods=18, radius=1.5, seed=0)

_PS_WATCH = r"""
$filter = @{{LogName='System'; ProviderName='nvlddmkm'; StartTime=[datetime]'{since}'}}
while ($true) {{
  try {{ $n = @(Get-WinEvent -FilterHashtable $filter -ErrorAction Stop).Count }}
  catch {{ if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {{ $n = 0 }} else {{ $n = 'ERR' }} }}
  [Console]::Out.WriteLine("$n"); [Console]::Out.Flush()
  Start-Sleep -Milliseconds {interval_ms}
}}
"""

_PS_DETAILS = r"""
$filter = @{{LogName='System'; ProviderName='nvlddmkm'; StartTime=[datetime]'{since}'}}
$time = @{{n='time'; e={{$_.TimeCreated.ToString('o')}}}}
$msg = @{{n='message'; e={{($_.Message -split "`n")[0]}}}}
Get-WinEvent -FilterHashtable $filter -ErrorAction SilentlyContinue |
  Select-Object -First 20 $time, Id, LevelDisplayName, $msg | ConvertTo-Json -Compress
"""


def stress_volume():
    sys.path.insert(0, HERE)
    from gpu_stamp_golden import sparse_rods
    return sparse_rods(VOLUME["shape"], VOLUME["n_rods"], VOLUME["radius"], VOLUME["seed"])


def worker(root, seconds):
    """One stress process: dt_spacing + dt_number on the GPU until the time is up; one JSON line per round."""
    sys.path.insert(0, root)
    import numpy as np
    import ipldt
    from ipldt import dt_number, dt_spacing
    if not os.path.normcase(os.path.abspath(ipldt.__file__)).startswith(os.path.normcase(os.path.abspath(root))):
        raise SystemExit(f"imported {ipldt.__file__}, not the checkout {root}")
    seg = stress_volume()
    print(json.dumps({"ready": True, "ipldt": ipldt.__file__, "bvtv_pct": round(100 * float(seg.mean()), 3)}),
          flush=True)
    t_end = time.time() + seconds
    rounds = 0
    while time.time() < t_end:
        t0 = time.time()
        a = dt_spacing(seg, backend="gpu")
        b = dt_number(seg, backend="gpu")
        rounds += 1
        print(json.dumps({"round": rounds, "seconds": round(time.time() - t0, 2), "max_D_spacing": int(a.map.max()),
                          "max_D_number": int(b.map.max()), "centres": int(a.centres.sum() + b.centres.sum())}),
              flush=True)
    print(json.dumps({"done": rounds}), flush=True)
    return 0


def _event_details(since):
    p = subprocess.run(["powershell.exe", "-NoProfile", "-Command", _PS_DETAILS.format(since=since)],
                       capture_output=True, text=True, timeout=60)
    try:
        d = json.loads(p.stdout) if p.stdout.strip() else []
        return d if isinstance(d, list) else [d]
    except ValueError:
        return [p.stdout.strip()]


def run(root, workers=4, minutes=10.0, report=None, interval_ms=1000, log=print):
    """Start the workers and the event-log watcher; returns (exit status, report dict)."""
    if sys.platform != "win32":
        raise SystemExit("gpu_stress reads the Windows System event log: Windows only")
    since = (datetime.datetime.now() - datetime.timedelta(seconds=2)).strftime("%Y-%m-%dT%H:%M:%S")
    rep = {"root": root, "workers": workers, "minutes": minutes, "since": since, "volume": VOLUME,
           "IPLDT_GPU_LOCK": os.environ.get("IPLDT_GPU_LOCK", "unset (lock on)"),
           "events": 0, "event_details": [], "worker_rounds": {}, "worker_tail": {}, "status": None}
    watch = subprocess.Popen(["powershell.exe", "-NoProfile", "-Command",
                              _PS_WATCH.format(since=since, interval_ms=interval_ms)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    counts = []
    first = threading.Event()

    def read_watch():
        for line in watch.stdout:
            counts.append(line.strip())
            if line.strip() not in ("0", ""):
                first.set()
    threading.Thread(target=read_watch, daemon=True).start()
    while not counts:                                  # the log must be readable before the load starts
        if watch.poll() is not None:
            break
        time.sleep(0.2)
    if not counts or counts[0] != "0":
        watch.kill()
        rep["status"] = f"event log not readable / already has events since {since}: {counts[:1]}"
        return 3, rep
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", "--root", root,
                               "--seconds", str(int(minutes * 60))],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
             for k in range(workers)]
    tails = {k: [] for k in range(workers)}

    def read_worker(k, p):
        for line in p.stdout:
            tails[k] = (tails[k] + [line.rstrip()])[-6:]
    for k, p in enumerate(procs):
        threading.Thread(target=read_worker, args=(k, p), daemon=True).start()
    log(f"{workers} workers on {root} for {minutes} min; watching nvlddmkm events since {since}")
    t_stop = time.time() + minutes * 60 + 600         # workers stop by themselves; 10 min grace for a last round
    status, t_log = 0, time.time()
    while True:
        if first.wait(0.2):
            for p in procs:                           # at the first driver event: kill everything at once
                p.kill()
            status = 1
            break
        if counts and counts[-1] == "ERR":
            status = 3
            break
        codes = [p.poll() for p in procs]
        if any(c not in (None, 0) for c in codes):
            status = 2
            break
        if all(c == 0 for c in codes) or time.time() > t_stop:
            break
        if time.time() - t_log > 60:
            t_log = time.time()
            log(f"  {datetime.datetime.now():%H:%M:%S} events {counts[-1] if counts else '?'}; "
                + "; ".join(f"w{k}: {tails[k][-1][:60] if tails[k] else '-'}" for k in tails))
    for p in procs:
        if p.poll() is None:
            p.kill()
    for p in procs:
        p.wait(timeout=60)
    if status in (0, 2):                             # a late event still counts
        time.sleep(5)
        n_late = counts[-1] if counts else "ERR"
        if n_late not in ("0", "ERR"):
            status = 1
    watch.kill()
    n = next((int(c) for c in reversed(counts) if c.isdigit()), None)
    rep["events"] = n
    if status == 1 or (n or 0) > 0:
        rep["event_details"] = _event_details(since)
    for k in tails:
        rounds = [json.loads(x)["round"] for x in tails[k] if x.startswith('{"round"')]
        rep["worker_rounds"][k] = max(rounds) if rounds else 0
        rep["worker_tail"][k] = tails[k]
    rep["status"] = {0: "no driver events", 1: "driver event: workers killed", 2: "a worker failed",
                     3: "event log not readable"}[status]
    rep["ended"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if report:
        with open(report, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, indent=1)
    return status, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=REPO, help="checkout whose ipldt the workers import (default: this one)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--report", default=None, help="write the result as JSON here")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seconds", type=float, default=600, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        return worker(os.path.abspath(a.root), a.seconds)
    status, rep = run(os.path.abspath(a.root), a.workers, a.minutes, a.report)
    print(json.dumps({k: rep[k] for k in ("status", "events", "worker_rounds", "since", "ended", "IPLDT_GPU_LOCK")},
                     indent=1))
    for e in rep["event_details"]:
        print("  event:", e)
    return status


if __name__ == "__main__":
    sys.exit(main())
