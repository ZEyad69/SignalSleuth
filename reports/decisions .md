# Decisions

One entry per decision: what, why, and what it cost. Newer entries can change older ones; say so in the entry.

## D-01 Threshold from the train part only
- Decision: the alert threshold is taken from scores on the train part. It is never taken from the test part or from test labels.
- Why: a threshold chosen on the test labels leaks information and inflates results (the first AAAI paper criticises this).
- Options I considered: the highest train score (chosen, to compare with my first run on series 011) or the 99.9% quantile (less affected by one strange point).
- Changed by: D-05.

## D-02 Evaluate only after the train part
- Decision: labels and alerts are cut at `train_end` in the same way, and metrics use only what comes after.
- Why: the detector or the reference has seen the train part, so scores there are too optimistic.

## D-03 The 30 files: spread evenly, fixed before looking at results
- Decision: 30 files evenly spaced over the sorted list of 250.
- Why: avoid choosing files that make a detector look good. The rule was written before any result.

## D-04 Failures are recorded, never skipped
- Decision: a run that fails is written in the results file with its error text. Summaries use only series where ALL detectors ran, so every detector is compared on the same series.
- Why: dropping failures silently changes the sample and can favour one detector.

## D-05 One calibration rule for every detector
- Decision: the train part is split 70% / 30%. The first 70% is reference / training data, the last 30% is calibration, and the threshold is the highest score in the calibration part, for all five detectors.
- Why: before this, Isolation Forest used a shorter region than MAD and shape, and shape used the calibration windows as reference. A threshold from a shorter region is lower, so the comparison was not fair.
- Cost: MAD numbers differ from `run_many_w250.csv` (`calib_fraction=0` reproduces the old rule). Series with a very short train part would fail for shape and Isolation Forest; none failed on the 250.

## D-06 Detector settings fixed before the big run
- Decision: MAD window 250, shape m = 100, Isolation Forest m = 100 with seed 0, `diff` and `random` without parameters (seed 0).
- Why: no tuning on the results. One setting per detector, so the comparison is not about who tuned more.
- Limit: one window per detector; other windows were not tested.

## D-07 Third metric: UCR top-1 hit, as an addition
- Decision: hit = the highest-scoring point after the train part lies inside the anomaly extended by max(100, anomaly length) on each side.
- Why: it needs no threshold, so it does not depend on the calibration rule or on Point Adjustment. Wu & Keogh also argue for returning one most likely location per series.
- Limit: the tolerance is our choice.

## D-08 PA%K: strict "greater than K"
- Decision: a true anomaly segment is adjusted only if the share of its points that are alerts is strictly larger than K%. The metric is the area under the F1-vs-K curve, K = 0..100.
- Why: with "greater than", K = 0 is ordinary PA and K = 100 is plain F1. With "greater than or equal", K = 0 would adjust segments with no alert at all.
- Note: sources word the condition slightly differently. Tests check K = 0 against `pa_f1` and K = 100 against `f1` on 200 random cases.

## D-09 Baselines without training: `diff` and `random`
- Decision: `diff` = |x_t - x_(t-1)|; `random` = seeded noise.
- Why: any detector must beat simple baselines (both papers). `random` shows the chance level (14 of 250 hits).
- Checked against Wu & Keogh: `diff` is the core of their one-liner `abs(diff(TS)) > b`, but they search `b` per series with the labels and I use the calibration threshold. So it is a one-liner-style baseline, not their exact procedure.

## D-10 How results are reported
- On 30 series most medians were 0, so for the 250 run I report means with 95% intervals (Wilson for hit rates, bootstrap over series for the others) and keep medians only for false-alarm events.
- Paired tests between detectors on the same series: exact McNemar for hit / miss, Wilcoxon for PA-F1 and PA%K-AUC.
- Holm correction: the p-values of the 10 detector pairs are corrected with Holm, separately for each metric, because ten tests at 5% would give some false "significant" results by chance. Example: shape vs `diff` on the hit metric, p = 0.014 before and 0.028 after Holm.
- Limit: series from the same source may not be independent, so intervals may be too optimistic.

## D-11 The 250 run is parallel and can be restarted
- Decision: each finished (file, detector) pair is written to the CSV at once, and a restart skips pairs already there. Heavy shape runs start first.
- Why: the shape detector needs about 328 CPU minutes for 250 series and one file alone took 865 s on the 30-series run.
- Note: a failed pair is not retried; delete its row to retry.

## D-12 Abstention works on alert EVENTS, not points
- Decision: alert points are grouped into events (runs closer than 100 points are merged). An event is TRUE if it touches the anomaly extended by max(100, anomaly length); all other events are false alarms.
- Why: on the 250 series the window-based detectors gave 4 to 11 false-alarm points per false event (mad 4.1, iforest 9.7, shape 10.9), so points are not independent. The number of false points was 5 to 97 times what an independent-scores guarantee predicts, while `random` matched it (1.16). Events are close to independent.

## D-13 Strength of an alert event: excess over the threshold
- Decision: strength = (peak score - threshold) / (highest calibration score - median calibration score), capped at 1000.
- Why: it has no unit, so one cut can be pooled over series, and it still works when scores are 0 or flat (a ratio peak / threshold would not).

## D-14 Second cut learned from a backtest on the train part
- Decision: the same pipeline is run on the train part alone (its first 70% plays "train", its last 30% plays "test"). All alert events there are false by construction. The cut of a detector is the 90% quantile of their strengths, pooled over all series. q = 0.9 was fixed before looking at results. Between the base threshold and the cut an alert is "uncertain"; above it, "confident".
- Why: the calibration part cannot say anything above its own maximum.
- Uses: only train parts (of all series), no test data and no label.
- Check: the share of REAL false events above the cut should be close to 1 - q. If not, the backtest does not transfer to this detector.
- Known mismatch: the calibration part of the backtest (21% of the train part) is shorter than in the real run (30%). `--pseudo-fraction` can change the split.
- Limit: it assumes the end of the train part looks like the test part. Drift breaks that.

## D-15 Drift stress test
- Decision: `add_drift` changes only the test part (level step, linear trend, or scale change). Cuts are learned on the clean run and evaluated on the drifted run.
- Why: the roadmap asks how abstention behaves under drift; the calibration check above tells how far the guarantee breaks.
