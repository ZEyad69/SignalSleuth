"""Run the whole project with one command.

Usage:
    py scripts\\reproduce.py <folder_with_the_UCR_files>            (full run: many hours for the shape detector)
    py scripts\\reproduce.py <folder> --quick                       (12 files, a few minutes, writes quick_* files)
    py scripts\\reproduce.py <folder> --workers 4 --skip-tests

<folder> can be any folder above the data (for example where you unzipped the archive): the script
looks inside it for the folder that holds the *_UCR_Anomaly_*.txt files.

Steps: tests -> five detectors on all files -> tables -> summary figure -> abstention events -> abstention tables and figure.
Everything is written to reports\\ and figures\\. Each step can be stopped with Ctrl+C and the run restarted:
the two long steps continue where they stopped.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def find_data_dir(root):
    """The folder under `root` with the most UCR anomaly files."""
    root = Path(root)
    best, best_n = None, 0
    for d in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        n = len(list(d.glob("*_UCR_Anomaly_*.txt")))
        if n > best_n:
            best, best_n = d, n
    return best, best_n


def run(title, cmd):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78, flush=True)
    t0 = time.time()
    code = subprocess.call([sys.executable] + [str(c) for c in cmd], cwd=ROOT)
    if code != 0:
        sys.exit("\nSTOPPED: step failed (%s). Fix it and run the same command again." % title)
    print("-- done in %.1f min" % ((time.time() - t0) / 60), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--quick", action="store_true", help="12 files, output files start with quick_")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    data, n = find_data_dir(args.folder)
    if data is None:
        sys.exit("No *_UCR_Anomaly_*.txt files found under %s" % args.folder)
    print("data folder:", data, "|", n, "files")

    pre = "quick_" if args.quick else ""
    limit = ["--limit", "12"] if args.quick else []
    workers = ["--workers", str(args.workers)] if args.workers else []
    cmp_csv = ROOT / "reports" / (pre + "compare_250.csv")
    abs_csv = ROOT / "reports" / (pre + "abstention_events.csv")

    if not args.skip_tests:
        run("1/6 tests", ["-m", "pytest", "-q"])
    run("2/6 five detectors on the files", [SCRIPTS / "run_compare_250.py", data, "--out", cmp_csv] + limit + workers)
    run("3/6 comparison tables", [SCRIPTS / "analyze_compare_250.py", cmp_csv])
    run("4/6 summary figure", [SCRIPTS / "make_summary_figure.py", cmp_csv, ROOT / "figures" / (pre + "summary_hit_rates.png")])
    run("5/6 abstention events (alert events, backtest on the train part)",
        [SCRIPTS / "run_abstention.py", data, "--out", abs_csv] + limit + workers)
    run("6/6 abstention tables and figure",
        [SCRIPTS / "analyze_abstention.py", abs_csv, "--fig", ROOT / "figures" / (pre + "abstention_risk_coverage.png")])
    print("\nALL DONE. Look in reports\\ and figures\\ (files starting with %s)." % (pre or "the usual names"))


if __name__ == "__main__":
    main()
