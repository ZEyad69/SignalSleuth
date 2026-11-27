"""Run different detectors on one series with exactly the same rules.

Rules (the same for every detector):
1. The detector gives a score for every point.
2. The train part is split: the first (1 - calib_fraction) is "fit" (reference / training data),
   the last calib_fraction is "calibration". No detector sees the calibration part as reference.
3. The threshold is the highest score in the calibration part.
4. Alerts are the points whose score is above the threshold.
5. Evaluation uses only the part AFTER the train part:
   point-wise, PA and PA%K-AUC metrics (with the threshold) and the UCR top-1 hit (without it).
"""
from signalsleuth.advanced_detectors import iforest_score, nn_shape_score
from signalsleuth.baselines import abs_diff_score, random_score
from signalsleuth.data import load_series
from signalsleuth.detectors import rolling_mad_score
from signalsleuth.evaluation import evaluate_alerts, make_labels
from signalsleuth.evaluation_extra import pa_k_auc, ucr_top1_hit

CALIB_FRACTION = 0.3

# the setups used in every comparison run (fixed BEFORE looking at results)
DEFAULT_SETUPS = [
    ("mad", {"window": 250}),
    ("shape", {"m": 100}),
    ("iforest", {"m": 100, "seed": 0}),
    ("diff", {}),
    ("random", {"seed": 0}),
]


def _fit_end(series, calib_fraction):
    return int(series.train_end * (1.0 - calib_fraction))


def _mad(series, calib_fraction, window=250):
    # warm-up: the first `window` scores are zeros, they cannot change the maximum
    if series.train_end <= 2 * window:
        raise ValueError("train part is too short for this window")
    # calib_fraction = 0 gives the old rule (threshold from the WHOLE train part), kept so the
    # first runs (experiments.run_series) can still be reproduced exactly
    calib_start = 0 if calib_fraction == 0 else _fit_end(series, calib_fraction)
    return rolling_mad_score(series.values, window=window), calib_start


def _shape(series, calib_fraction, m=100):
    fit_end = _fit_end(series, calib_fraction)
    scores = nn_shape_score(series.values, series.train_end, m=m, ref_end=fit_end)
    return scores, fit_end


def _iforest(series, calib_fraction, m=100, seed=0):
    scores, calib_start = iforest_score(series.values, series.train_end, m=m, seed=seed,
                                        calib_fraction=calib_fraction)
    assert calib_start == _fit_end(series, calib_fraction)
    return scores, calib_start


def _diff(series, calib_fraction):
    return abs_diff_score(series.values), _fit_end(series, calib_fraction)


def _random(series, calib_fraction, seed=0):
    return random_score(series.values, seed=seed), _fit_end(series, calib_fraction)


DETECTORS = {"mad": _mad, "shape": _shape, "iforest": _iforest, "diff": _diff, "random": _random}


def compute_scores(series, detector, calib_fraction=CALIB_FRACTION, **params):
    """Scores for every point, where the calibration part starts, and the threshold.

    The threshold is the highest score in the calibration part (the end of the train part).
    Returns (scores, calib_start, threshold).
    """
    scores, calib_start = DETECTORS[detector](series, calib_fraction, **params)
    threshold = float(scores[calib_start:series.train_end].max())
    return scores, calib_start, threshold


def run_detector(path, detector, calib_fraction=CALIB_FRACTION, **params):
    series = load_series(path)
    scores, calib_start, threshold = compute_scores(series, detector, calib_fraction, **params)

    alerts = scores > threshold
    labels = make_labels(len(series.values), series.anomaly_start, series.anomaly_end)

    # evaluate only after the train part, cutting both arrays in the same way
    result = evaluate_alerts(labels[series.train_end:], alerts[series.train_end:])

    # threshold-free metric: is the single highest score in the test part on the anomaly?
    hit, top = ucr_top1_hit(scores, series.train_end, series.anomaly_start, series.anomaly_end)

    # PA%K area under the F1-vs-K curve, same alerts and same cut as the other metrics
    result["pak_auc"] = pa_k_auc(labels[series.train_end:], alerts[series.train_end:])

    # facts needed later (effect of anomaly length on PA, effect of train length on runtime)
    result["anomaly_len"] = series.anomaly_end - series.anomaly_start + 1
    result["train_end"] = series.train_end
    result["n"] = len(series.values)

    result["name"] = series.name
    result["detector"] = detector
    result["threshold"] = threshold
    result["ucr_hit"] = hit
    result["top_index"] = top
    return result
