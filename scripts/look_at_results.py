from pathlib import Path

import pandas as pd

path = Path(__file__).resolve().parents[1] / "reports" / "run_many_w250.csv"
df = pd.read_csv(path)

cols = ["name", "tp", "fp", "fn", "f1", "pa_f1", "false_alarm_events"]

# 1. the series where the anomaly was found
print(df[df["anomaly_detected"] == True][cols].to_string(index=False))

# 2. the median is less affected by a few extreme series than the mean
print()
print("median false-alarm events:", df["false_alarm_events"].median())
print("mean false-alarm events:  ", round(df["false_alarm_events"].mean(), 2))

# 3. the 5 series with the most false-alarm events
print()
print(df.sort_values("false_alarm_events", ascending=False)[["name", "false_alarm_events"]]
      .head(5).to_string(index=False))