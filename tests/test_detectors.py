import numpy as np

from signalsleuth.detectors import rolling_mad_score


def make_normal_series():
    # a repeating pattern of 30 points, no anomaly
    return np.array([10, 11, 10, 12, 11] * 6, dtype=float)


def test_finds_planted_anomaly():
    x = make_normal_series()
    x[25] = 50  # plant one anomaly at index 25
    scores = rolling_mad_score(x, window=5)
    # the highest score must be at the anomaly
    assert scores.argmax() == 25


def test_normal_series_has_low_scores():
    x = make_normal_series()
    scores = rolling_mad_score(x, window=5)
    # no anomaly, so no point should look very far from normal
    assert scores.max() < 1


def test_flat_series_has_no_nan():
    x = np.full(50, 5.0)  # the same value 50 times
    scores = rolling_mad_score(x, window=5)
    # no NaN or infinity, and no fake alarms
    assert np.isfinite(scores).all()
    assert scores.max() == 0


def test_no_peeking_at_the_future():
    x = make_normal_series()
    y = x.copy()
    y[25] = 500  # change only a FUTURE point (index 25)
    sx = rolling_mad_score(x, window=5)
    sy = rolling_mad_score(y, window=5)
    # scores before index 25 must not change
    np.testing.assert_allclose(sx[:25], sy[:25])