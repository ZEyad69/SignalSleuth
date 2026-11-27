"""Collect the alert events of every series: the real test part and the backtest on the train part.

Usage:
    py scripts\\run_abstention.py                               (all files, detectors mad, shape, iforest, diff, random)
    py scripts\\run_abstention.py --detectors mad,diff,iforest  (fast first pass, no shape)
    py scripts\\run_abstention.py --limit 30                    (30 files spread over the archive)
    py scripts\\run_abstention.py --drift level:1.0 --out reports\\abstention_drift_level.csv
    py scripts\\run_abstention.py --workers 4
    py scripts\\run_abstention.py --pseudo-fraction 0.8 --out reports\\abstention_events_pf80.csv

Parallel, and it can be stopped and started again (a finished (file, detector) is never redone).
--drift kind:strength changes ONLY the test part (kinds: level, trend, scale; see signalsleuth.abstention).
Then run:  py scripts\\analyze_abstention.py
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from signalsleuth.abstention import add_drift, series_events
from signalsleuth.compare import DEFAULT_SETUPS
from signalsleuth.data import load_series

DEFAULT_DIR = Path(
    r"C:\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021"
    r"\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"
)
REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIELDS = ["name", "detector", "phase", "event_id", "start", "end", "peak", "excess", "is_true",
          "threshold", "scale", "anomaly_start", "anomaly_end", "anomaly_len", "train_end", "n",
          "seconds", "error"]


def work(path_str, detector, params, drift, pseudo_fraction):
    """Runs in a worker process. Returns the rows of this (file, detector); never raises."""
    t0 = time.time()
    name = Path(path_str).stem
    try:
        s = load_series(path_str)
        if drift:
            kind, strength = drift
            s.values = add_drift(s.values, s.train_end, kind, strength)
        res = series_events(s, detector, pseudo_fraction=pseudo_fraction, **params)
        base = {"name": name, "detector": detector, "anomaly_start": s.anomaly_start,
                "anomaly_end": s.anomaly_end, "anomaly_len": s.anomaly_end - s.anomaly_start + 1,
                "train_end": s.train_end, "n": len(s.values), "error": ""}
        rows = []
        for phase in ("real", "pseudo"):
            ph = res[phase]
            # event_id -1 is a marker: it says "this phase was done" even when it has no event
            rows.append({**base, "phase": phase, "event_id": -1, "threshold": ph["threshold"],
                         "scale": ph["scale"]})
            for k, ev in enumerate(ph["events"]):
                rows.append({**base, "phase": phase, "event_id": k, "threshold": ph["threshold"],
                             "scale": ph["scale"], **ev})
    except Exception as e:
        rows = [{"name": name, "detector": detector, "phase": "error", "event_id": -1, "error": str(e)}]
    sec = round(time.time() - t0, 1)
    for r in rows:
        r["seconds"] = sec
    return rows


def already_done(out):
    done = set()
    if out.exists():
        with open(out, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["name"], r["detector"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--detectors", default=",".join(d for d, _ in DEFAULT_SETUPS))
    ap.add_argument("--drift", default=None, help="kind:strength, e.g. level:1.0")
    ap.add_argument("--pseudo-fraction", type=float, default=0.7,
                    help="share of the train part that plays 'train' in the backtest (default 0.7)")
    ap.add_argument("--out", default=str(REPORTS / "abstention_events.csv"))
    args = ap.parse_args()

    drift = None
    if args.drift:
        kind, strength = args.drift.split(":")
        drift = (kind, float(strength))
    wanted = set(args.detectors.split(","))
    setups = [(d, p) for d, p in DEFAULT_SETUPS if d in wanted]

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    files = sorted(Path(args.folder).glob("*.txt"))
    if args.limit:
        files = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, args.limit).astype(int))]
    print("files:", len(files), "| detectors:", [d for d, _ in setups], "| drift:", drift,
          "| workers:", args.workers, "| output:", out)

    done = already_done(out)
    tasks = []
    for f in files:
        train_end = int(f.stem.split("_")[-3])
        for d, p in setups:
            if (f.stem, d) not in done:
                cost = f.stat().st_size * train_end if d == "shape" else f.stat().st_size
                tasks.append((cost, f, d, p))
    tasks.sort(key=lambda t: -t[0])
    print("already done:", len(done), "| still to run:", len(tasks))
    if not tasks:
        return

    new_file = not out.exists() or out.stat().st_size == 0
    t_start = time.time()
    with open(out, "a", newline="") as fh, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        futures = [pool.submit(work, str(f), d, p, drift, args.pseudo_fraction) for _, f, d, p in tasks]
        for k, fut in enumerate(as_completed(futures), start=1):
            rows = fut.result()
            writer.writerows(rows)
            fh.flush()
            r0 = rows[0]
            n_ev = sum(1 for r in rows if r["event_id"] != -1)
            status = "FAILED" if r0["error"] else "%d events" % n_ev
            print("[%4d/%d] %-8s %-10s %7.1fs  %s" % (k, len(tasks), r0["detector"], status,
                                                       r0["seconds"], r0["name"][:45]), flush=True)
    print("finished in %.1f min -> %s" % ((time.time() - t_start) / 60, out))


if __name__ == "__main__":
    main()
