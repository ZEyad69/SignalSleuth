import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt

from signalsleuth.data import load_series

DATA_DIR = Path(
    r"C:\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021"
    r"\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"
)

# 8 files spread evenly across all 250, so we see different kinds of data
files = sorted(DATA_DIR.glob("*.txt"))
step = len(files) // 8
chosen = files[::step][:8]

fig, axes = plt.subplots(4, 2, figsize=(14, 10))

for ax, f in zip(axes.flat, chosen):
    s = load_series(f)
    ax.plot(s.values, linewidth=0.5)
    ax.axvspan(0, s.train_end, color="green", alpha=0.15)
    ax.axvspan(s.anomaly_start, s.anomaly_end, color="red", alpha=0.5)
    ax.set_title(f"{s.name[:45]}  (n={len(s.values)})", fontsize=8)
    print(f"{s.name[:40]:40s} length={len(s.values):7d} "
          f"anomaly_len={s.anomaly_end - s.anomaly_start + 1}")

plt.tight_layout()
out = Path(__file__).resolve().parents[1] / "figures" / "overview.png"
plt.savefig(out, dpi=150)
print("saved:", out)
plt.show()