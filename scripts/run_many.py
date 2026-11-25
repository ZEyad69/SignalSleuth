import sys
from pathlib import Path

# let Python find our package inside src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from signalsleuth.experiments import run_series

DATA_DIR = Path(
    r"C:\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021"
    r"\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"
)
WINDOW = 250
N_FILES = 30

# rule decided BEFORE looking at any result: 30 files spread evenly over all 250
files = sorted(DATA_DIR.glob("*.txt"))
positions = np.linspace(0, len(files) - 1, N_FILES).astype(int)
chosen = [files[i] for i in positions]

rows = []
for f in chosen:
    try:
        row = run_series(f, window=WINDOW)
        row["error"] = None
        print("ok    ", f.stem[:45], "detected:", row["anomaly_detected"])
    except Exception as e:
        # do not hide failures: record them
        row = {"name": f.stem, "error": str(e)}
        print("FAILED", f.stem[:45], e)
    rows.append(row)

df = pd.DataFrame(rows)
out = Path(__file__).resolve().parents[1] / "reports" / f"run_many_w{WINDOW}.csv"
df.to_csv(out, index=False)
print("saved:", out)

ok = df[df["error"].isna()]
print()
print("series run:", len(df), "| failed:", len(df) - len(ok))
print("anomaly found:", int(ok["anomaly_detected"].astype(bool).sum()), "of", len(ok))
print("mean F1 (point-wise):", round(ok["f1"].mean(), 4))
print("mean F1 (after PA):  ", round(ok["pa_f1"].mean(), 4))
print("mean false-alarm events:", round(ok["false_alarm_events"].mean(), 2))