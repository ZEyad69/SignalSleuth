import sys
from pathlib import Path

# let Python find our package inside src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from signalsleuth.data import load_series

DATA_DIR = Path(
    r"C:\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021"
    r"\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"
)
FILE = next(DATA_DIR.glob("011_*"))

s = load_series(FILE)
print("name:", s.name)
print("length:", len(s.values))
print("train ends at:", s.train_end)
print("anomaly:", s.anomaly_start, "to", s.anomaly_end)

fig, axes = plt.subplots(2, 1, figsize=(12, 7))

# top: the whole series
ax = axes[0]
ax.plot(s.values, linewidth=0.6)
ax.axvspan(0, s.train_end, color="green", alpha=0.15, label="train")
ax.axvspan(s.anomaly_start, s.anomaly_end, color="red", alpha=0.4, label="anomaly")
ax.set_title(s.name)
ax.legend()

# bottom: zoom around the anomaly
margin = 500
lo = max(0, s.anomaly_start - margin)
hi = min(len(s.values), s.anomaly_end + margin)
ax = axes[1]
ax.plot(range(lo, hi), s.values[lo:hi], linewidth=0.8)
ax.axvspan(s.anomaly_start, s.anomaly_end, color="red", alpha=0.4)
ax.set_title("Zoom around the anomaly")

plt.tight_layout()
out = Path(__file__).resolve().parents[1] / "figures" / "series_011.png"
plt.savefig(out, dpi=150)
print("saved:", out)
plt.show()