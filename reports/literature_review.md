# Paper 1: Towards a Rigorous Evaluation of Time-Series Anomaly Detection
Kim et al., AAAI 2022. https://arxiv.org/abs/2109.05257

## Problem
Many papers report very high F1 scores for anomaly detectors. But they use a trick
called point adjustment (PA). With PA, if the detector flags one point inside an
anomaly, the whole anomaly counts as found. This makes scores look better than they are.

## Main idea
- A random score can get a very high F1 with PA. So a high PA score proves nothing.
- A simple detector with no training works about as well as many complex models,
  even without PA. So new methods must beat this simple baseline.
- The authors suggest PA%K: use PA only if the detector finds at least K% of the anomaly.

## Data
Five public datasets: SWaT, WADI, SMD, MSL and SMAP. Seven published models were tested.

## How they evaluate
They compute F1 with PA and without PA. They compare every model with random scores
and with the simple baseline. Most models did not beat the baseline without PA.

## Limitations
- They did not test on the UCR archive, which we use.
- They picked the threshold that gave the best F1 on the test data. This is unfair.
- PA%K needs us to choose K.

## What I will use in SignalSleuth
- Report F1 with PA and without PA, never PA alone.
- Add a simple baseline and compare every detector with it.
- Choose thresholds from training data only, never from test labels.
- Check if PA matters less on UCR, because its anomalies are short.