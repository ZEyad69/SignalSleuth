import sys
from pathlib import Path

# let Python find our package inside src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from signalsleuth.data import load_series
from signalsleuth.detectors import rolling_mad_score

DATA_DIR = Path(
    r"C:\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021"
    r"\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"
)
FILE = next(DATA_DIR.glob("011_*"))

# window in one variable, so I can change it in one place
WINDOW = 100

# load the series and compute one score for every point
s = load_series(FILE)
scores = rolling_mad_score(s.values, window=WINDOW)

# sanity check: one score per point, so the two numbers must be equal
print(len(s.values), len(scores))
train_max = scores[:s.train_end].max()
anomaly_max = scores[s.anomaly_start:s.anomaly_end + 1].max()
# 1. threshold chosen from the train part only
threshold = train_max

# 2. all alerts after the train part
alerts_after_train = (scores[s.train_end:] > threshold).sum()

# 3. alerts inside the real anomaly (the correct ones)
alerts_in_anomaly = (scores[s.anomaly_start:s.anomaly_end + 1] > threshold).sum()

# 4. false alarms = alerts that are not inside the anomaly
false_alarms = alerts_after_train - alerts_in_anomaly

print("threshold:", threshold)
print("alerts after train:", alerts_after_train)
print("alerts inside anomaly:", alerts_in_anomaly)
print("false alarms:", false_alarms)
# two plots, one above the other, sharing the same x axis
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(s.values, linewidth=0.6)
axes[0].set_title(f"{s.name} - values")

axes[1].plot(scores, linewidth=0.6, color="orange")
axes[1].set_title(f"rolling MAD score (window={WINDOW})")

# shade train (green) and anomaly (red) on both plots
for ax in axes:
    ax.axvspan(0, s.train_end, color="green", alpha=0.15)
    ax.axvspan(s.anomaly_start, s.anomaly_end, color="red", alpha=0.4)

plt.tight_layout()
out = Path(__file__).resolve().parents[1] / "figures" / f"scores_011_w{WINDOW}.png"
plt.savefig(out, dpi=150)
print("saved:", out)
plt.show()

