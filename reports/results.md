# Result #1: five detectors on all 250 UCR Anomaly series

All numbers come from `reports/compare_250.csv` (1250 runs, no failures) and `scripts/analyze_compare_250.py`.

## Setup
- All 250 files of the UCR Anomaly Archive, the same files for every detector.
- Detectors: rolling median/MAD (window 250), shape detector (nearest z-normalised window in the reference part, m = 100), Isolation Forest (m = 100, seed 0), `diff` (absolute difference to the previous point, no training), `random` (seeded noise).
- One rule for all: the first 70% of the train part is reference / training data, the last 30% is calibration, the threshold is the highest calibration score. Evaluation uses only the part after the train part.
- Metrics, shown side by side: UCR top-1 hit (no threshold; the highest score after the train part lies within max(100, anomaly length) of the anomaly), point-wise F1, F1 after Point Adjustment (PA-F1), and the area under the PA%K curve (K = 0..100).
- Uncertainty: Wilson intervals for hit rates; percentile bootstrap over series for means; exact McNemar (hit / miss) and Wilcoxon (PA-F1, PA%K-AUC) for paired tests, Holm-corrected over the 10 detector pairs.

## Main table

| Detector | UCR hit | 95% CI | Mean F1 | Mean PA-F1 | Mean PA%K-AUC | Mean recall | False-alarm events (mean / median) |
|---|---|---|---|---|---|---|---|
| shape | 107/250 = 42.8% | 36.8 to 49.0 | 0.130 | 0.325 | 0.165 | 0.256 | 27.9 / 9 |
| diff (no training) | 80/250 = 32.0% | 26.5 to 38.0 | 0.053 | 0.283 | 0.069 | 0.058 | 12.2 / 5 |
| iforest | 43/250 = 17.2% | 13.0 to 22.4 | 0.064 | 0.163 | 0.081 | 0.099 | 23.5 / 8 |
| mad | 20/250 = 8.0% | 5.2 to 12.0 | 0.010 | 0.112 | 0.016 | 0.013 | 18.1 / 6 |
| random | 14/250 = 5.6% | 3.4 to 9.2 | 0.001 | 0.083 | 0.002 | 0.001 | 12.1 / 6 |

## What the numbers support
1. **The shape detector is the best of the five on the hit metric.** It beats MAD (95 series only shape found, 8 only MAD, p < 0.0001), Isolation Forest (85 vs 21, p < 0.0001) and the training-free `diff` baseline (70 vs 43, McNemar p = 0.014, Holm-corrected 0.028).
2. **Rolling MAD, as configured here, cannot be told apart from random on the hit metric** (20 series only MAD found, 14 only random, p = 0.39). Its PA%K-AUC is higher than random (0.016 vs 0.002, Holm p < 0.0001) and its mean F1 is higher too (0.010 vs 0.001, not tested), but all these values are very small.
3. **The simple `diff` baseline is strong.** It found 80 of 250 series, more than Isolation Forest (63 vs 26 series only one found, Holm p = 0.0005) and about three quarters of what the shape detector found.
4. **The conclusion about a detector depends on the metric.**
   - Shape vs `diff`: the hit rate and PA%K-AUC separate them (Holm p < 0.0001 for PA%K-AUC), PA-F1 does not (0.325 vs 0.283, Holm p = 0.44).
   - Isolation Forest vs `diff`: `diff` is better on hit rate and on PA-F1 (0.283 vs 0.163, Holm p = 0.0003), but PA%K-AUC finds no difference (0.081 vs 0.069, Holm p = 0.80).
   - MAD vs random: indistinguishable on hit rate (p = 0.39) and on PA-F1 (Holm p = 0.57).
5. **How often the order of the detectors changes with the metric** (share of the 250 series where some pair of detectors is ordered in the opposite way by two metrics): F1 vs PA-F1 30.8%, PA-F1 vs PA%K-AUC 29.2%, F1 vs PA%K-AUC 4.4%, hit vs F1 11.2%, hit vs PA-F1 14.0%. In 76% of the series at least one point metric separates the detectors; among those, 40% flip between F1 and PA-F1.

## Point Adjustment
- PA raises scores a lot without a better detection: `diff` goes from F1 0.053 to PA-F1 0.283, and a random score from 0.001 to 0.083. Random reaches PA-F1 >= 0.5 in 20 of 250 series while the hit metric says it missed.
- Inflation (PA-F1 minus F1) grows with the anomaly length for shape (Spearman 0.33), Isolation Forest (0.26), MAD (0.19) and random (0.15), but not for `diff` (-0.01). Mean PA-F1 of the shape detector is 0.130 for the shortest third of anomalies (length up to 69), 0.413 for the middle third (70 to 201) and 0.436 for the longest (202 to 1701). This is a correlation: longer anomalies are also easier to catch, so it is not a pure PA effect.

## Exploratory, not conclusions
- Recall vs false alarms: with one fixed threshold rule, the detectors with more recall also have more false-alarm events (shape 27.9, Isolation Forest 23.5 on average), but `diff` has more recall than MAD with fewer false alarms, and the random score already gives 12.1 events. So this is not a clean trade-off; it needs the threshold sweep of the abstention phase.
- Data family (92 DISTORTED, 16 NOISE, 142 other): shape 0.359 / 0.250 / 0.493, `diff` 0.370 / 0.188 / 0.303. The advantage of shape over `diff` appears in the "other" group only; the NOISE group is too small to say anything.
- Train length (terciles of `train_end`, edges 1000 / 5000 / 18913 / 250000): hit rate of shape 0.569 / 0.439 / 0.244, `diff` 0.510 / 0.167 / 0.207, random 0.098 / 0.030 / 0.024. All five detectors have a lower hit rate in the longest-train group than in the shortest one (the decline is not strictly steady for `diff` and Isolation Forest), including `diff` and random, which have no reference part. Train length also goes with series length, so the groups are confounded. This split does not support the earlier hypothesis that a long reference hurts the shape detector in particular; it does not test it either.
- Compute time (CPU minutes for all 250): shape 328, Isolation Forest 14, MAD 3, `diff` 2, random 2.

## Limits
- UCR only: univariate series with one labelled anomaly each. Nothing here says how the detectors behave on other benchmarks.
- One window size per detector (m = 100, window 250), one seed, no tuning, one threshold rule (highest calibration score).
- The archive contains several series from the same source (the names repeat). Independence between series was not checked, so intervals and p-values may be too optimistic.
- The hit metric uses a tolerance of max(100, anomaly length) that was chosen by us; a different tolerance may change the numbers.
- Point-wise F1 and PA metrics are computed on one fixed threshold, which is a weak choice for every detector (precision is low everywhere).
- `diff` is the core of the one-liner `abs(diff(TS)) > b` in Wu & Keogh (2021). Their threshold `b` was searched for each series with the labels, ours comes from the calibration part and uses no labels, so `diff` is a one-liner-style baseline, not their exact procedure. They write that a small fraction of the UCR archive may be solvable by a one-liner but give no number; here `diff` has a hit in 80 of 250 series (32%) under our metric and tolerance, so simple rules are far from useless on this archive. That is not a contradiction of the paper, because the two measures are different.
