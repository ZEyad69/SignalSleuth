# Phase 1 Notes: What I learned from the UCR data

1. Each series has its own scale (from about -1..1 up to 30,000), so a fixed threshold cannot work. The detector must be relative to each series.
2. Anomalies are tiny, often under 1% of the series, so Accuracy is meaningless. Precision, Recall and false alarms are needed.
3. Anomalies are often a change in SHAPE, not a big value. Some large spikes in normal regions would fool a simple detector.
4. Many series are repeating patterns (heartbeats, breathing, walking), which is why Matrix Profile is a good baseline to include.
5. Every file name encodes train_end, anomaly_start and anomaly_end, and load_series() parses them automatically.