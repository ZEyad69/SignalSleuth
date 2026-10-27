# Phase 2 Notes: First detector, evaluation functions, run_series

## Setup
- Detector: rolling median + MAD, uses only past points (shift(1)), no training.
- Series: 011_UCR_Anomaly_DISTORTEDECG1_10000_11800_12100 (30,000 points).
- Train part: points 0-10,000. Anomaly: points 11,800-12,100 (301 points).
- Threshold: the highest score in the train part only (never looks at the test part).

## First look: highest score in train vs in anomaly

| Window | Highest in train | Highest in anomaly | Anomaly higher? |
|--------|------------------|--------------------|-----------------|
| 100    | 42.65            | 27.06              | No              |
| 500    | 18.53            | 30.25              | Yes             |

Same detector, same series: changing only the window flipped the result.
A larger window made the scores more stable (highest train score: 42.65 -> 18.53).
The first real score appears after about 2 x window points (median, then MAD).

## Alert counts (threshold = highest train score)

| Window | Threshold | Alerts after train | Inside anomaly (TP) | False alarm points (FP) |
|--------|-----------|--------------------|---------------------|-------------------------|
| 100    | 42.65     | 5                  | 0                   | 5                       |
| 500    | 18.53     | 13                 | 6                   | 7                       |

Counts are in POINTS. The anomaly is ONE event of 301 points, so "6 inside" means
6 points above the threshold.

## Metrics, with and without point adjustment (PA)

| Case                | TP  | FP | FN  | Precision | Recall | F1    |
|---------------------|-----|----|-----|-----------|--------|-------|
| Window 500, no PA   | 6   | 7  | 295 | 0.46      | 0.02   | 0.04  |
| Window 500, with PA | 301 | 7  | 0   | 0.98      | 1.00   | 0.99  |
| Window 100 (both)   | 0   | 5  | 301 | 0.00      | 0.00   | 0.00  |

- Same detector, same alerts: F1 went from 0.04 to 0.99 only because of PA.
  The detector really flagged 6 of 301 points (about 2% of the event).
- PA changes nothing when no point is detected (window 100).
- This matches the warning in Kim et al. (AAAI 2022): a high F1 after PA can hide a weak detector.
- I expected PA to matter less on UCR because anomalies are short. That holds for series
  with 2-point anomalies, but here the anomaly is 301 points, so PA has a huge effect.
  The effect depends on anomaly length, so I will measure it on many series instead of guessing.

## Events, not only points (measured with run_series, window 500)
- The 7 false-alarm points form 4 separate groups (false_alarm_events = 4).
- So an operator would be disturbed 4 times, not 1 and not 7.
- Caution: the number depends on how an "event" is defined here (a gap of one
  non-alert point already splits two groups). I will revisit this when comparing detectors.

## What I built in Phase 2 (all tested with pytest, 22 tests passing)
- make_labels, confusion_counts, precision_recall_f1, point_adjust, count_segments,
  evaluate_alerts (in evaluation.py).
- run_series (in experiments.py): loads one UCR file, scores it, picks the threshold
  from the train part only, evaluates only after the train part.

## Bugs I made and caught (lessons)
- First point_adjust started from an all-False array, so false alarms outside the
  anomaly were lost. Fix: start from a copy of the original alerts.
- A draft of run_series passed the arguments to evaluate_alerts in the wrong order
  (alerts, labels instead of labels, alerts). It does not crash, it silently swaps FP and FN.
  Lesson: check results against numbers computed by hand.

## My observation
(write one sentence here in your own words)

## Limits (do not over-claim)
- ONE series and ONE run so far. Nothing here is general.
- The threshold rule (highest train score) is simple. One strange point in train
  would raise it a lot. Other options to test: a high quantile of the train scores.
- Rolling MAD looks at values only. Anomalies that change the SHAPE of the signal
  (like in series 011) are what it misses, which motivates Phase 3.

## Still to check
- run_series with window 100 should give tp 0 and fp 5 (same as the first run).
- run_series on 20-30 series. Series with a short train part may have a bad threshold,
  because the first 2 x window scores are zeros.