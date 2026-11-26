# Phase 3 conclusion: shape detector vs baselines on 30 UCR series

## Setup
- 30 series spread evenly over the sorted UCR Anomaly Archive (250 files). Same files for every detector.
- Detectors: rolling median/MAD (window 250), shape detector (nearest z-normalised window in the reference part, m = 100), Isolation Forest (window features, m = 100, seed 0), `diff` (absolute difference to the previous point, no training), `random` (seeded noise).
- One rule for all: the first 70% of the train part is the reference / training data, the last 30% is calibration, and the threshold is the highest score in the calibration part. Nothing from the test part is used.
- Main metric: UCR top-1 hit (the highest-scoring point after the train part lies inside the anomaly extended by max(100, anomaly length)). It needs no threshold. 95% percentile bootstrap CIs over series. Raw results: `reports/compare_30_v2.csv`.

## Results

| Detector | Series found (UCR hit) | Rate (95% CI) | Median false-alarm events |
|---|---|---|---|
| mad | 1 / 30 | 3% (0 to 10%) | 2.5 |
| shape | 13 / 30 | 43% (27 to 60%) | 4.5 |
| iforest | 6 / 30 | 20% (7 to 37%) | 4.5 |
| diff | 9 / 30 | 30% (13 to 47%) | 4.0 |
| random | 0 / 30 | 0% | 4.0 |

Median point-wise F1 is 0 to 0.002 and median PA-F1 is 0 to 0.004 for every detector.

## What the numbers support
1. Rolling MAD found the anomaly in 1 of 30 series. It did worse than the training-free `diff` baseline, so on UCR it is a weak baseline.
2. The shape detector beat MAD: it found 12 series that MAD missed, and MAD found none that shape missed (exact McNemar p < 0.001).
3. Shape vs `diff`: shape-only 8 series, diff-only 4 series, McNemar p = 0.39. With 30 series we cannot say the shape detector is better than the simple baseline. The 250-series run has to decide this.
4. Isolation Forest found 6 series. Against `diff` the difference is not significant (5 vs 2 series, p = 0.45).
5. The `random` score never hit, so the hit rates above are not luck.

## What the numbers do not support
- F1 is about 0 for every detector: with this threshold rule there are few alerts inside the anomaly and several false-alarm events outside it. This motivates the abstention phase. It says nothing yet about how much Point Adjustment inflates scores on UCR.
- Runtime and failures (exploratory, post hoc): the shape detector took 18.8 minutes for the 30 series, 77% of it in one file. It found 3 of the 11 series where it ran 5 s or longer, against 10 of 19 for the others (Fisher p = 0.26, not significant). A possible cause is that a long, varying reference makes normal windows look unusual. This was NOT tested.

## Check of the shape detector
Its distances were compared with stumpy's matrix profile (AB-join, same windows, same z-normalisation): maximum absolute difference 0.00003 on a made-up series and 0.0001 on `112_UCR_Anomaly_BIDMC1`, with the same top-scoring window. Neither series has flat windows, which the two implementations treat differently.

## Limits
One sample of 30 series, one window size per detector, one seed, no tuning, one real file in the stumpy check. No claim about the archive as a whole is made until the 250-series run.
