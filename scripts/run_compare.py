"""Run all detectors on the SAME 30 files, with the SAME rules, and report with confidence intervals.

Usage:
    py scripts\\run_compare.py              (uses the UCR folder below)
    py scripts\\run_compare.py <folder>     (any folder with UCR-style .txt files)

Rules, fixed BEFORE looking at any result:
- the same 30 files as run_many.py (evenly spread over the sorted archive)
- mad: window 250 | shape: m = 100 | iforest: m = 100, seed 0 | diff, random: no parameters
- the same threshold rule for all: highest score in the calibration part
  (the last 30% of the train part, never used as reference / training data)
- failures are recorded, never skipped; the summary uses only series where ALL detectors ran
- metrics: UCR top-1 hit (no threshold), anomaly found, F1, F1 after PA, false-alarm events
"""
import sys
import time
from pathlib import Path

# let Python find our package inside src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from signalsleuth.compare import DEFAULT_SETUPS, run_detector
from signalsleuth.evaluation_extra import bootstrap_ci, mcnemar_exact, wilcoxon_paired

DEFAULT_DIR = Path(
    r"C:\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021"
    r"\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"
)
DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
N_FILES = 30
SETUPS = DEFAULT_SETUPS
REPORTS = Path(__file__).resolve().parents[1] / "reports"
REPORTS.mkdir(exist_ok=True)
OUT = REPORTS / "compare_30_v2.csv"
OUT_SUMMARY = REPORTS / "compare_30_v2_summary.csv"

# the same rule as run_many.py: files spread evenly over the sorted archive
files = sorted(DATA_DIR.glob("*.txt"))
positions = np.unique(np.linspace(0, len(files) - 1, N_FILES).astype(int))
chosen = [files[i] for i in positions]
print("files chosen:", len(chosen), "of", len(files))

rows = []
for k, f in enumerate(chosen, start=1):
    for detector, params in SETUPS:
        t0 = time.time()
        try:
            row = run_detector(f, detector, **params)
            row["error"] = None
            status = "hit" if row["ucr_hit"] else "miss"
        except Exception as e:
            # do not hide failures: record them
            row = {"name": f.stem, "detector": detector, "error": str(e)}
            status = "FAILED"
        rows.append(row)
        print("[%2d/%d] %-8s %-6s %5.1fs  %s" % (k, len(chosen), detector, status,
                                                  time.time() - t0, f.stem[:40]), flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)      # save as we go, so a crash loses nothing

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("\nsaved:", OUT)

order = [d for d, _ in SETUPS]
failed = {d: int(df[(df["detector"] == d) & df["error"].notna()].shape[0]) for d in order}
print("failures per detector:", failed)

ok = df[df["error"].isna()].copy()
for col in ["ucr_hit", "anomaly_detected"]:
    ok[col] = ok[col].astype(bool).astype(float)
for col in ["f1", "pa_f1", "false_alarm_events"]:
    ok[col] = ok[col].astype(float)

# keep only series where EVERY detector ran, so all detectors are compared on the same series
counts = ok.groupby("name")["detector"].nunique()
common = counts[counts == len(order)].index
ok = ok[ok["name"].isin(common)]
print("series where all detectors ran:", len(common), "of", len(chosen))
if len(common) == 0:
    sys.exit("no common series, nothing to summarise")

# ---- summary per detector, with 95% bootstrap CI over series ----
table = []
for d in order:
    sub = ok[ok["detector"] == d].sort_values("name")
    hit, h_lo, h_hi = bootstrap_ci(sub["ucr_hit"])
    found, _, _ = bootstrap_ci(sub["anomaly_detected"])
    f1, _, _ = bootstrap_ci(sub["f1"], stat=np.median)
    pa, pa_lo, pa_hi = bootstrap_ci(sub["pa_f1"], stat=np.median)
    fa, fa_lo, fa_hi = bootstrap_ci(sub["false_alarm_events"], stat=np.median)
    table.append({
        "detector": d, "series": len(sub),
        "ucr_hit_rate": hit, "ucr_hit_lo": h_lo, "ucr_hit_hi": h_hi,
        "anomaly_found_rate": found,
        "median_f1": f1,
        "median_pa_f1": pa, "pa_f1_lo": pa_lo, "pa_f1_hi": pa_hi,
        "median_false_alarm_events": fa, "fa_lo": fa_lo, "fa_hi": fa_hi,
    })
summary = pd.DataFrame(table).set_index("detector")
summary.round(3).to_csv(OUT_SUMMARY)
print("\n=== summary per detector (95% bootstrap CI over series) ===")
print(summary.round(3).T.to_string())

# ---- every detector against the MAD baseline, on the same series ----
print("\n=== paired tests vs mad (same series) ===")
mad = ok[ok["detector"] == "mad"].sort_values("name").reset_index(drop=True)
for d in order:
    if d == "mad":
        continue
    other = ok[ok["detector"] == d].sort_values("name").reset_index(drop=True)
    d_only, mad_only, p_hit = mcnemar_exact(other["ucr_hit"], mad["ucr_hit"])
    p_pa = wilcoxon_paired(other["pa_f1"], mad["pa_f1"])
    print("%-8s UCR hit: %s only=%d, mad only=%d, p=%.3f | PA-F1 Wilcoxon p=%.3f"
          % (d, d, d_only, mad_only, p_hit, p_pa))
print("\nwith 30 series, small p-values are rare; treat these as a first look, the 250 run decides.")
