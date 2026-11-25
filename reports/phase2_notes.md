# Phase 2 Notes: First detector, evaluation functions, run_series, first run on 30 series

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
- I expected PA to matter less on UCR because anomalies are short. I have not tested
  very short anomalies yet. Here the anomaly is 301 points, so PA has a huge effect.
  The effect depends on anomaly length, so I will measure it on many series instead of guessing.

## Events, not only points (measured with run_series)

| Window | False-alarm points | Separate groups (events) |
|--------|--------------------|--------------------------|
| 100    | 5                  | 2                        |
| 500    | 7                  | 4                        |

- With window 500, an operator would be disturbed 4 times, not 1 and not 7.
- Caution: the number depends on how an "event" is defined here (a gap of one
  non-alert point already splits two groups). I will revisit this when comparing detectors.

## What I built in Phase 2 (all tested with pytest, 25 tests passing)
- make_labels, confusion_counts, precision_recall_f1, point_adjust, count_segments,
  evaluate_alerts (in evaluation.py).
- run_series (in experiments.py): loads one UCR file, scores it, picks the threshold
  from the train part only, evaluates only after the train part.
- run_series gives the same numbers as my first manual run (score_series.py) for
  window 500 (tp 6, fp 7, fn 295) and window 100 (tp 0, fp 5, fn 301).
- run_many.py: runs run_series on 30 files and saves a CSV; failures are recorded, not skipped.

## Decision: window = 250 for all series
- The warm-up of the detector is about 2 x window points (all scores are 0 there).
- I counted the train part of all 250 files from their names only: the smallest
  train_end is 1000, the 10 smallest are 1000, 1000, 1200, 1200, 1200, 1300, 1300,
  1500, 1500, 1500, and none is below 1000.
- With window 500 the warm-up is about 1000 points, so a series with train_end = 1000
  would have no real train scores and a threshold of 0.
- Window 250 leaves about 500 real train scores even in the shortest case.
- Chosen from this constraint of the data, not from results. No series is excluded.
- Limits: one fixed window for all series, and short-train series get a threshold from
  few scores.
- The earlier numbers for series 011 use windows 100 and 500, so they are not comparable
  with runs at window 250.

## Run on 30 series (window 250)

Rules fixed BEFORE looking at results: 30 files spread evenly over the sorted archive,
window 250 for all series, failures are recorded and not skipped, threshold = highest
train score, evaluation only after the train part.

| Measure | Value |
|---------|-------|
| Series run / failed | 30 / 0 |
| Anomaly found (at least one point) | 2 of 30 |
| Mean F1, point-wise | 0.005 |
| Mean F1, after PA | 0.042 |
| False-alarm events: median / mean | 1.5 / 10.57 |

The two series where the anomaly was found:

| Series | TP | FP | FN | F1 | F1 after PA | False-alarm events |
|--------|----|----|----|----|-------------|--------------------|
| 112_BIDMC1 | 18 | 51 | 183 | 0.133 | 0.887 | 7 |
| 198_tiltAPB2 | 29 | 2711 | 798 | 0.016 | 0.379 | 219 |

### What this shows
- The mean F1 values come entirely from these two series (0.887 + 0.379 over 30 gives
  0.042), so the other 28 series have F1 = 0. A mean F1 alone is misleading here, so I
  report the detection rate (2 of 30), medians and a per-series table.
- Point adjustment multiplies F1 by about 7 (series 112) and about 23 (series 198), while
  the detector flagged only 18 of 201 and 29 of 827 anomalous points. This repeats the
  series 011 lesson on more than one example.
- False alarms are concentrated: series 198 alone has 219 of about 317 events (about 70%).
  Without it the mean would be about 3.4 events per series. The median (1.5) is a better
  summary than the mean (10.57).
- The 30 files are evenly spaced, not a random sample, so this is a first look and not
  an estimate for the whole archive.

### Not tested (hypotheses only)
- The anomalies are changes in shape, not extreme values.
- The threshold (highest train score) is too strict for most series.
- Series 198 may contain level shifts that cause long runs of alerts.
- One fixed window does not suit every series.

## Bugs found during development (lessons)
- First point_adjust started from an all-False array, so false alarms outside the
  anomaly were lost. Fix: start from a copy of the original alerts.
- A draft of run_series passed the arguments to evaluate_alerts in the wrong order
  (alerts, labels instead of labels, alerts). It does not crash, it silently swaps FP and FN.
  Lesson: check results against numbers computed by hand.

## My observation
(write one sentence here in your own words)

## Limits (do not over-claim)
- 30 series, one run, one threshold rule. Nothing here is general.
- The threshold rule (highest train score) is simple. One strange point in train
  would raise it a lot. Other options to test: a high quantile of the train scores.
- Rolling MAD looks at values only. Anomalies that change the SHAPE of the signal
  (like in series 011) are what it is likely to miss, which motivates Phase 3.

## Next
- Phase 3: a shape-aware detector, plus Isolation Forest, run on the same 30 files with
  the same rules, so the comparison with rolling MAD is fair.
- Phase 4: all 250 series, several metrics, an untrained baseline, confidence intervals.