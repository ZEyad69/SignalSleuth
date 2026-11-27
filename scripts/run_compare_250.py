"""All detectors on ALL UCR files: parallel, and it can be stopped and started again.

Usage:
    py scripts\\run_compare_250.py                      (all files, workers = CPU cores - 1)
    py scripts\\run_compare_250.py --workers 4
    py scripts\\run_compare_250.py --limit 12           (a quick test on 12 files spread over the archive)
    py scripts\\run_compare_250.py <folder> --out reports\\my_run.csv

Same rules as run_compare.py (setups in signalsleuth.compare.DEFAULT_SETUPS).
Every finished (file, detector) pair is written to the CSV at once. When you start the script
again it reads the CSV and skips the pairs that are already there, so Ctrl+C never loses work.
A failed pair is also written (with the error text) and is NOT retried; to retry it, delete its row.
"""
import os

# one thread per worker: the workers already run in parallel, more threads would only fight
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

from signalsleuth.compare import DEFAULT_SETUPS, run_detector

DEFAULT_DIR = Path(
    r"C:\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021"
    r"\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"
)
REPORTS = Path(__file__).resolve().parents[1] / "reports"
FIELDS = ["name", "detector", "error", "seconds", "tp", "fp", "fn", "precision", "recall", "f1",
          "pa_precision", "pa_recall", "pa_f1", "pak_auc", "false_alarm_events",
          "anomaly_detected", "ucr_hit", "top_index", "threshold",
          "anomaly_len", "train_end", "n"]


def work(path_str, detector, params):
    """Runs in a worker process. Never raises: a failure becomes a row with an error text."""
    t0 = time.time()
    try:
        row = run_detector(path_str, detector, **params)
        row["error"] = ""
    except Exception as e:
        row = {"name": Path(path_str).stem, "detector": detector, "error": str(e)}
    row["seconds"] = round(time.time() - t0, 1)
    return row


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
    ap.add_argument("--out", default=str(REPORTS / "compare_250.csv"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    files = sorted(Path(args.folder).glob("*.txt"))
    if args.limit:
        files = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, args.limit).astype(int))]
    print("files:", len(files), "| workers:", args.workers, "| output:", out)

    done = already_done(out)
    tasks = []
    for f in files:
        train_end = int(f.stem.split("_")[-3])
        for detector, params in DEFAULT_SETUPS:
            if (f.stem, detector) not in done:
                # rough cost: the shape detector needs (series length x train length); big ones first
                cost = f.stat().st_size * train_end if detector == "shape" else f.stat().st_size
                tasks.append((cost, f, detector, params))
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
        futures = [pool.submit(work, str(f), d, p) for _, f, d, p in tasks]
        for k, fut in enumerate(as_completed(futures), start=1):
            row = fut.result()
            writer.writerow(row)
            fh.flush()                                   # saved at once, Ctrl+C loses nothing
            status = "FAILED" if row["error"] else ("hit" if row["ucr_hit"] else "miss")
            print("[%4d/%d] %-8s %-6s %7.1fs  %s" % (k, len(tasks), row["detector"], status,
                                                      row["seconds"], row["name"][:45]), flush=True)
    print("finished in %.1f min -> %s" % ((time.time() - t_start) / 60, out))


if __name__ == "__main__":
    main()
