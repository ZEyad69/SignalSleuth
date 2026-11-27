"""Does saying "not sure" help? Tables and a risk-coverage figure from the events file.

Usage:
    py scripts\\analyze_abstention.py                          (reports\\abstention_events.csv)
    py scripts\\analyze_abstention.py reports\\abstention_drift_level.csv --cuts-from reports\\abstention_events.csv

The strength cut of every detector is learned ONLY from the backtest events (the pseudo-test inside
the train part), pooled over the series. The real test events are used to evaluate it, never to set it.
With --cuts-from the cut is learned from another file (for the drift check: cut from the clean run,
evaluation on the drifted run).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from signalsleuth.abstention import confidence_cut, risk_coverage, summarize_at_cut
from signalsleuth.evaluation_extra import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["mad", "shape", "iforest", "diff", "random"]
Q_MAIN = 0.9
Q_GRID = [0.5, 0.75, 0.9, 0.95, 0.99]
pd.set_option("display.width", 220)


def load(path):
    raw = pd.read_csv(path)
    bad = raw[raw["phase"] == "error"]
    ok = raw[raw["phase"] != "error"].copy()
    ok["is_true"] = ok["is_true"].astype(str).str.lower().eq("true")
    return ok, bad


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ci(v):
    est, lo, hi = bootstrap_ci(np.asarray(v, dtype=float))
    return "%.2f (%.2f to %.2f)" % (est, lo, hi)


ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?", default=str(ROOT / "reports" / "abstention_events.csv"))
ap.add_argument("--cuts-from", default=None)
ap.add_argument("--q", type=float, default=Q_MAIN)
ap.add_argument("--fig", default=None)
args = ap.parse_args()
csv_path = Path(args.csv)
stem = csv_path.parent / csv_path.stem

ev, bad = load(csv_path)
cut_ev, _ = load(args.cuts_from) if args.cuts_from else (ev, None)
dets = [d for d in ORDER if d in set(ev["detector"])]
print("file:", csv_path.name, "| detectors:", dets, "| failed (file, detector):", len(bad),
      "| cuts learned from:", Path(args.cuts_from).name if args.cuts_from else "the same file (backtest part)")

rows, rc_rows, curves = [], [], {}
for d in dets:
    e = ev[ev["detector"] == d]
    names = sorted(e.loc[(e["phase"] == "real") & (e["event_id"] == -1), "name"].unique())
    real = e[(e["phase"] == "real") & (e["event_id"] >= 0)]
    pseudo = cut_ev[(cut_ev["detector"] == d) & (cut_ev["phase"] == "pseudo") & (cut_ev["event_id"] >= 0)]
    n_pseudo = len(pseudo)
    if n_pseudo < 30:
        print("WARNING: %s has only %d backtest events; its cut is not reliable" % (d, n_pseudo))

    base = summarize_at_cut(real, names, -np.inf)
    rows.append({"detector": d, "q": "none", "cut": np.nan, "backtest_events": n_pseudo,
                 "false_events_per_series": ci(base["false_events_all"]),
                 "anomaly_caught": "%.3f" % base["caught_all"].mean(),
                 "event_precision": "%.3f" % base["event_precision"],
                 "real_false_events_above_cut": "", "target": "", "true_events_demoted": 0})
    for q in Q_GRID:
        cut = confidence_cut(pseudo["excess"], q)
        r = summarize_at_cut(real, names, cut)
        k = int(r["caught_confident"].sum())
        lo, hi = wilson(k, len(names))
        rows.append({"detector": d, "q": q, "cut": round(cut, 3), "backtest_events": n_pseudo,
                     "false_events_per_series": ci(r["false_events_confident"]),
                     "anomaly_caught": "%.3f (%.3f to %.3f)" % (k / len(names), lo, hi),
                     "event_precision": "%.3f" % r["event_precision"],
                     "real_false_events_above_cut": "%.3f" % r["false_confident_rate"],
                     "target": "%.2f" % (1 - q), "true_events_demoted": r["true_demoted"]})

    cov, risk, base_risk = risk_coverage(real)
    grid = np.linspace(0.05, 1.0, 20)
    risk_at = np.interp(grid, cov, risk)
    rc_rows.append({"detector": d, "events": len(real), "base_risk": round(base_risk, 3),
                    "risk_at_50%_coverage": round(float(np.interp(0.5, cov, risk)), 3),
                    "risk_at_25%_coverage": round(float(np.interp(0.25, cov, risk)), 3),
                    "mean_risk_over_coverage": round(float(risk_at.mean()), 3)})

    # operating curve: false confident events per series vs share of series where the anomaly is caught
    sweep = np.unique(np.quantile(real["excess"], np.linspace(0, 1, 60))) if len(real) else np.array([0.0])
    pts = []
    for c in np.concatenate([[-np.inf], sweep, [np.inf]]):
        r = summarize_at_cut(real, names, c)
        pts.append((r["false_events_confident"].mean(), r["caught_confident"].mean()))
    curves[d] = (cov, risk, base_risk, np.array(pts))

table = pd.DataFrame(rows)
rc = pd.DataFrame(rc_rows).set_index("detector")
table.to_csv(str(stem) + "_table.csv", index=False)
rc.to_csv(str(stem) + "_riskcov.csv")

q_main = args.q
print("\n=== 1. effect of the second cut (q = share of backtest false events that stay BELOW it) ===")
print("columns: false_events_per_series and anomaly_caught are for CONFIDENT alerts only;")
print("real_false_events_above_cut should be close to target = 1 - q if the backtest transfers\n")
for d in dets:
    print("--", d)
    print(table[table["detector"] == d].drop(columns="detector").to_string(index=False))

print("\n=== 2. risk-coverage (event level; risk = share of kept alert events that are false alarms) ===")
print("a useful confidence score makes the risk fall below base_risk as coverage falls")
print(rc.to_string())

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
colors = {"mad": "tab:blue", "shape": "tab:red", "iforest": "tab:green", "diff": "tab:orange", "random": "tab:gray"}
for d, (cov, risk, base_risk, pts) in curves.items():
    axes[0].plot(cov, risk, color=colors[d], label=d)
    axes[0].axhline(base_risk, color=colors[d], ls=":", lw=0.8)
    axes[1].plot(pts[:, 0], pts[:, 1], color=colors[d], label=d)
axes[0].set_xlabel("coverage (share of alert events kept as confident)")
axes[0].set_ylabel("risk (share of kept events that are false alarms)")
axes[0].set_title("Risk-coverage (dotted line = no abstention)")
axes[0].set_ylim(0, 1.02)
axes[1].set_xlabel("false confident alert events per series")
axes[1].set_ylabel("share of series where the anomaly is caught")
axes[1].set_title("Operating curve: raising the cut moves left and down")
axes[1].set_ylim(0, 1.02)
for ax in axes:
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
fig_path = Path(args.fig) if args.fig else ROOT / "figures" / (csv_path.stem + "_risk_coverage.png")
fig_path.parent.mkdir(exist_ok=True)
plt.savefig(fig_path, dpi=150)
print("\nsaved figure:", fig_path)
print("saved tables:", stem.name + "_table.csv,", stem.name + "_riskcov.csv")
