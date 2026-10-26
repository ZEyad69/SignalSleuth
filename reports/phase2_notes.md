# Phase 2 Notes: First detector (rolling median + MAD)

## Setup
- Detector: rolling median + MAD, uses only past points (shift(1)), no training.
- Series: 011_UCR_Anomaly_DISTORTEDECG1_10000_11800_12100 (30,000 points).
- Train part: points 0-10,000. Anomaly: points 11,800-12,100.
- Measured: the highest score inside the train part and inside the anomaly.

## Results

| Window | Highest score in train | Highest score in anomaly | Anomaly higher? |
|--------|------------------------|--------------------------|-----------------|
| 100    | 42.65                  | 27.06                    | No              |
| 500    | 18.53                  | 30.25                    | Yes             |

## What this shows
- Same detector, same series: changing only the window flipped the result.
- If the threshold is taken from the train part only (its highest score):
  - window 100: threshold 42.65, no point in the anomaly passes it, so the anomaly is missed.
  - window 500: threshold 18.53, at least one point in the anomaly passes it.
- A larger window made scores more stable: the highest score in train dropped from 42.65 to 18.53.
- The first real score appears after about 2 x window points (median, then MAD),
  so scores before that are 0.

## My observation
(write one sentence here in your own words)

## Limits (do not over-claim)
- This is ONE series and ONE run. I cannot say that window 500 is better in general.
- "Highest score" is a rough measure. It does not say how many other high scores
  exist outside the anomaly (possible false alarms). Next step: count alerts.
## Alert counts (threshold = highest score in the train part)

Series 011, anomaly = 301 points (11,800 to 12,100 inclusive).

| Window | Threshold | Alerts after train | Inside anomaly (TP) | False alarms (FP) |
|--------|-----------|--------------------|---------------------|-------------------|
| 100    | 42.65     | 5                  | 0                   | 5                 |
| 500    | 18.53     | 13                 | 6                   | 7                 |

Counts are in POINTS, not events. The true anomaly is one event, so "6 alerts
inside" means 6 points above the threshold. The 7 false alarms may be only a few
events close together. I have not counted events yet.

## Metrics, with and without point adjustment (PA)

| Case                  | TP  | FP | FN  | Precision | Recall | F1   |
|-----------------------|-----|----|-----|-----------|--------|------|
| Window 500, no PA     | 6   | 7  | 295 | 0.46      | 0.02   | 0.04 |
| Window 500, with PA   | 301 | 7  | 0   | 0.98      | 1.00   | 0.99 |
| Window 100 (both)     | 0   | 5  | 301 | 0.00      | 0.00   | 0.00 |

## What this shows
- Same detector, same alerts: F1 went from 0.04 to 0.99 only because of PA.
  The detector really flagged 6 of 301 points (about 2% of the event).
- PA changes nothing when no point is detected (window 100).
- This matches the warning in the AAAI 2022 paper: a high F1 after PA can hide
  a weak detector.
- I expected PA to matter less on UCR because anomalies are short. That holds
  for series with 2-point anomalies, but here the anomaly is 301 points, so PA
  has a huge effect. The effect depends on anomaly length, which I will measure
  on all 250 series instead of guessing.
- The threshold came from the train part only, so there are no false alarms in
  train by construction. This does not mean the detector is good there.

## Limits
- ONE series and ONE run. Nothing here is general.
- Point-level counts only. Event-level counting is still to do.
## Next
- Count alerts above the threshold (total and inside the anomaly).
- Then write precision / recall / F1 functions in evaluation.py.