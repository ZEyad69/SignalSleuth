import numpy as np

from signalsleuth.data import load_series
from signalsleuth.detectors import rolling_mad_score
from signalsleuth.evaluation import make_labels, evaluate_alerts


def run_series(path, window=500):
    # 1. load the series (values, where train ends, where the anomaly is)
    series = load_series(path)

    # 2. one anomaly score for every point
    scores = rolling_mad_score(series.values, window=window)

    # 3. threshold from the TRAIN part only (never look at the test part here)
    # decision: use the highest train score, so I can compare with my first run on series 011
    # other option: np.quantile(train_scores, 0.999), less affected by one strange point
    train_scores = scores[:series.train_end]
    threshold = float(train_scores.max())

    # 4. alerts (True/False per point) and the real labels (0/1 per point)
    alerts = scores > threshold
    labels = make_labels(len(series.values), series.anomaly_start, series.anomaly_end)

    # evaluate only AFTER the train part; cut both arrays in the same way
    result = evaluate_alerts(labels[series.train_end:], alerts[series.train_end:])

    # 5. add the name and the threshold to the result
    result["name"] = series.name
    result["threshold"] = threshold
    return result
    