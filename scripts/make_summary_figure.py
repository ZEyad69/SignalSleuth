"""One figure that sums up Result #1 (reports/compare_250.csv -> figures/summary_hit_rates.png).

Left:  how often the single highest score after the train part lands on the anomaly (UCR top-1 hit),
       with 95% Wilson intervals. The dotted line is the random score.
Right: the same alerts judged with three metrics; Point Adjustment (PA-F1) is far above the others.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "compare_250.csv"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures" / "summary_hit_rates.png"
NAMES = {"shape": "Shape\n(nearest normal\nwindow)", "diff": "Diff\n(no training)", "iforest": "Isolation\nForest",
         "mad": "Rolling\nMAD", "random": "Random\nscore"}
COLORS = {"shape": "#c0392b", "diff": "#e67e22", "iforest": "#27ae60", "mad": "#2980b9", "random": "#7f8c8d"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


df = pd.read_csv(CSV)
df = df[df["error"].isna()].copy()
df["ucr_hit"] = df["ucr_hit"].astype(str).str.lower().eq("true")
counts = df.groupby("name")["detector"].nunique()
df = df[df["name"].isin(counts[counts == counts.max()].index)]
n = df["name"].nunique()

order = df[df.detector != "random"].groupby("detector")["ucr_hit"].mean().sort_values(ascending=False).index.tolist() + ["random"]
hits = {d: int(df[df.detector == d]["ucr_hit"].sum()) for d in order}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})

# ---- left: hit rate with Wilson intervals
x = np.arange(len(order))
rates = np.array([hits[d] / n for d in order]) * 100
ci = np.array([wilson(hits[d], n) for d in order]) * 100
ax1.bar(x, rates, color=[COLORS[d] for d in order], width=0.62)
ax1.errorbar(x, rates, yerr=[rates - ci[:, 0], ci[:, 1] - rates], fmt="none", ecolor="black", capsize=4, lw=1.2)
for i, d in enumerate(order):
    ax1.text(i, ci[i, 1] + 1.5, "%d/%d" % (hits[d], n), ha="center", fontsize=10)
ax1.axhline(rates[-1], color=COLORS["random"], ls=":", lw=1.2)
ax1.set_xticks(x)
ax1.set_xticklabels([NAMES[d] for d in order], fontsize=9)
ax1.set_ylabel("series where the top-scoring point hits the anomaly (%)")
ax1.set_title("Threshold-free hit rate on %d UCR series" % n, fontsize=12)
ax1.set_ylim(0, max(ci[:, 1]) + 9)
ax1.grid(axis="y", alpha=0.3)

# ---- right: three metrics for the same alerts
metrics = [("f1", "point-wise F1", "#bdc3c7"), ("pa_f1", "F1 after Point Adjustment", "#e74c3c"),
           ("pak_auc", "PA%K area (stricter)", "#34495e")]
w = 0.26
for j, (col, lab, colr) in enumerate(metrics):
    vals = [df[df.detector == d][col].astype(float).mean() for d in order]
    ax2.bar(x + (j - 1) * w, vals, width=w, color=colr, label=lab)
ax2.set_xticks(x)
ax2.set_xticklabels([NAMES[d] for d in order], fontsize=9)
ax2.set_ylabel("mean score over the %d series" % n)
ax2.set_title("The same alerts, three metrics", fontsize=12)
ax2.legend(frameon=False, loc="upper right")
ax2.grid(axis="y", alpha=0.3)

fig.suptitle("SignalSleuth: five detectors, one threshold rule, UCR Anomaly Archive", fontsize=13, y=1.0)
plt.tight_layout()
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=160, bbox_inches="tight")
print("saved:", OUT)
print({d: "%d/%d" % (hits[d], n) for d in order})
