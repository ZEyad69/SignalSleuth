# Result #2: saying "not sure" (abstention) on the 250 UCR series (draft)

Numbers come from `reports/abstention_events_table.csv`, `reports/abstention_events_riskcov.csv` and `figures/abstention_events_risk_coverage.png`.

## What was done
- Every alert is an EVENT (alert points closer than 100 points are merged). An event is TRUE if it touches the anomaly extended by max(100, anomaly length); all other events are false alarms.
- Three states: normal (below the base threshold), uncertain (alert event weaker than a second cut), confident (alert event at or above the cut).
- The second cut is learned only from the train part: the same pipeline is run on the train part alone (first 70% = "train", last 30% = "test"), every alert event there is false by construction, and the cut is the 90% quantile of their strengths pooled over the 250 series (q = 0.9, fixed before looking at results). Between 394 and 481 backtest events per detector (183 for `random`).
- Strength of an event = (peak - threshold) / (highest - median calibration score).

## Does the cut do what it is meant to? (calibration check)
Share of the REAL false events that reach the cut, against the target 1 - q:

| q | target | mad | shape | iforest | diff | random |
|---|---|---|---|---|---|---|
| 0.75 | 0.25 | 0.252 | 0.194 | 0.230 | 0.176 | 0.090 |
| 0.90 | 0.10 | 0.080 | 0.102 | 0.085 | 0.053 | 0.049 |
| 0.95 | 0.05 | 0.035 | 0.038 | 0.037 | 0.019 | 0.037 |

For MAD, shape and Isolation Forest the backtest transfers well at q = 0.9 (0.080 to 0.102 against 0.10). For `diff` and `random` it is conservative (about 0.05). In 13 of the 15 cells the real rate is below the target; the two cells slightly above it are MAD at q = 0.75 (0.252 against 0.25) and shape at q = 0.9 (0.102 against 0.10).

## Effect at q = 0.9 (confident alerts only)

| Detector | False events per series | Anomaly caught (share of series) | Event precision | True events demoted to "uncertain" |
|---|---|---|---|---|
| shape | 15.20 -> 1.54 | 0.688 -> 0.288 | 0.053 -> 0.166 | 136 |
| diff | 9.73 -> 0.51 | 0.416 -> 0.240 | 0.050 -> 0.347 | 59 |
| iforest | 11.06 -> 0.94 | 0.380 -> 0.104 | 0.039 -> 0.103 | 86 |
| mad | 12.84 -> 1.03 | 0.240 -> 0.072 | 0.032 -> 0.079 | 83 |
| random | 10.39 -> 0.51 | 0.224 -> 0.028 | 0.026 -> 0.066 | 60 |

## What the numbers support
1. **The cut controls the false-alarm rate.** About 90% fewer false confident events for every detector (shape 90%, mad 92%, iforest 92%, diff 95%), at a rate that was set from the train part only and that the real test data confirm.
2. **It costs anomalies.** For the shape detector the share of series where the anomaly is caught falls from 0.688 to 0.288. Abstention is a second, higher threshold chosen in a principled way; it does not make a detector better. What it adds is knowing in advance roughly how many false confident alerts to expect.
3. **Confident alerts are still mostly false.** Event precision after abstention: diff 0.347, shape 0.166, iforest 0.103, mad 0.079.
4. **`diff` and shape order their events by strength clearly better than chance, Isolation Forest a little, MAD hardly at all.** In the risk-coverage table the risk among the strongest 25% of events is 0.856 for `diff` and 0.869 for shape against a base risk of 0.950 and 0.947; for Isolation Forest 0.929 against 0.961; for MAD 0.948 against 0.968, close to the random score (0.964 against 0.974). The relative precision gain of MAD and Isolation Forest (2.5x and 2.6x) equals that of the random score (2.5x), so that gain alone says nothing about their confidence.
5. **Recall costs false alarms, and the price differs.** On the operating curve the shape detector catches 0.29 of the series at 1.5 false confident events per series, 0.44 at 3.0, 0.53 at 5.9 and 0.69 at 15.2. Below about 1.7 false events per series `diff` catches as many series as shape or more (0.24 at 0.51, 0.31 at 1.71); above about 2 shape is ahead (0.44 at 2.96 against 0.35 for `diff` at 3.12). MAD is close to `random` along the whole curve.

## Drift test: three synthetic drifts in the test part, cuts learned on the clean run
Each drift changes only the test part: a level step of +1 train standard deviation, a linear trend growing to +2 standard deviations, or an amplitude that grows slowly to twice its size. Run for MAD, Isolation Forest, `diff` and `random`. The shape detector was not run.

| Detector | Data | False events per series, no abstention | Share of false events above the cut at q = 0.90 (target 0.10) | False confident events per series at q = 0.90 |
|---|---|---|---|---|
| mad | clean | 12.84 | 0.080 | 1.03 |
| | level / trend / scale | 12.92 / 13.11 / 13.00 | 0.084 / 0.081 / 0.079 | 1.08 / 1.06 / 1.03 |
| iforest | clean | 11.06 | 0.085 | 0.94 |
| | level / trend / scale | 48.96 / 39.20 / 51.54 | 0.114 / 0.077 / 0.058 | 5.58 / 3.02 / 2.98 |
| diff | clean | 9.73 | 0.053 | 0.51 |
| | level / trend / scale | 10.21 / 12.31 / **102.71** | 0.095 / 0.042 / **0.458** | 0.97 / 0.51 / **47.00** |
| random | all four | 10.39 | 0.049 | 0.51 |

What it shows:
1. **MAD does not notice any of the three drifts.** The rolling median and the MAD follow the level and the amplitude, so the false alarms and the calibration stay as on the clean data (0.079 to 0.084 against the target 0.10). The cost is that MAD is hardly better than random as a detector (Result #1).
2. **Isolation Forest sees all three.** The number of false alert events grows 3.5 to 4.7 times. The share of false events that reach the cut stays near the target at high cuts (0.058 to 0.114 at q = 0.90) but is too high at low cuts (0.62 to 0.67 against 0.50 at q = 0.5). Since there are many more candidate events, the false confident events grow 3 to 6 times (0.94 to 2.98 to 5.58 per series) and the event precision falls from 0.103 to between 0.056 and 0.095.
3. **`diff` breaks when the amplitude changes.** With the amplitude growing to twice its size it gives 102.71 false events per series (10.6 times more), 0.458 of them pass the cut that should let 0.10 pass, and it gives 47.00 false confident events per series (92 times more); its event precision falls from 0.347 to 0.023. A likely reason, not tested: the score is an absolute difference, so it grows with the amplitude and the whole test part moves above the calibration maximum. A slow trend leaves `diff` unchanged at q = 0.90 (0.51 false confident events per series), and a level step only adds the single jump at the start of the test part (0.51 to 0.97).
4. **The promise of abstention is about the share, not the number.** It keeps "about 10% of the false events become confident" in most cases, but when drift makes the number of false events explode (Isolation Forest, `diff` with a scale change) the number of false confident alerts explodes with it, and for `diff` under a scale change the share itself fails.
5. The `random` rows are identical in all four runs, as they must be (its scores do not depend on the values), which also confirms that the cuts were the same.

An idea, not tested: the number of alert events is itself a drift signal. `diff` under a scale change gives about ten times the usual number of events; a system could stop trusting the cut when the event rate is far above what the calibration part predicts.

## Limits
- The cut of one detector pools the train parts of all series. No test data and no label is used, but the cut is not made series by series.
- q = 0.9 was fixed in advance; the other values of q are descriptive.
- "True event" depends on our tolerance (max(100, anomaly length)). Each series has one anomaly, so "caught" is yes or no.
- The `random` control uses the same seed in every series, so its scores repeat from series to series and the control is less strict than it looks. A different seed per series is better.
- Series from the same source may not be independent, so intervals may be too optimistic.
- Drift: three synthetic drifts of strengths chosen by us (1 and 2 standard deviations, a factor 2), on MAD, Isolation Forest, `diff` and `random`. The shape detector was not run under drift. Real drift is more varied, and no remedy was tested.
