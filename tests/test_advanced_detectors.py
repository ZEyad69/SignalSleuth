import numpy as np

from signalsleuth.advanced_detectors import iforest_score, nn_shape_score


def make_periodic(n=3000, period=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(n)


def slow_reference(x, train_end, m):
    """Slow and simple version: loops over every pair of windows."""
    n = len(x)
    ref_starts = range(0, train_end - m + 1)

    def z(w):
        s = w.std()
        return np.zeros_like(w) if s < 1e-8 else (w - w.mean()) / s

    out = np.zeros(n)
    for q in range(0, n - m + 1):
        zq = z(x[q:q + m])
        best = np.inf
        for r in ref_starts:
            if abs(q - r) < m:          # overlapping windows are not compared
                continue
            d = np.sqrt(np.sum((zq - z(x[r:r + m])) ** 2))
            best = min(best, d)
        out[q + m - 1] = 0.0 if not np.isfinite(best) else best
    return out


def test_shape_score_matches_slow_reference():
    x = make_periodic(n=500, period=40)
    fast = nn_shape_score(x, train_end=250, m=20)
    slow = slow_reference(x, train_end=250, m=20)
    np.testing.assert_allclose(fast, slow, atol=2e-2)


def test_shape_score_finds_a_shape_change_that_keeps_the_height():
    x = make_periodic(n=3000, period=60)
    # same amplitude, but a square wave instead of a sine wave
    x[2000:2120] = np.sign(np.sin(2 * np.pi * np.arange(2000, 2120) / 60))
    scores = nn_shape_score(x, train_end=1000, m=40)
    assert scores[2000:2160].max() > 2 * scores[:1000].max()


def test_shape_score_normal_series_has_no_big_scores():
    x = make_periodic(n=3000, period=60)
    scores = nn_shape_score(x, train_end=1000, m=40)
    assert scores[1000:].max() < 2 * scores[:1000].max()


def test_shape_score_flat_series_has_no_nan():
    x = np.full(1000, 3.0)
    scores = nn_shape_score(x, train_end=500, m=20)
    assert np.isfinite(scores).all()
    assert scores.max() == 0


def test_shape_score_never_looks_at_the_future():
    x = make_periodic(n=2000, period=60)
    y = x.copy()
    y[1500] += 50.0                       # change one FUTURE point
    sx = nn_shape_score(x, train_end=800, m=40)
    sy = nn_shape_score(y, train_end=800, m=40)
    np.testing.assert_allclose(sx[:1500], sy[:1500])


def test_shape_score_test_part_does_not_change_train_scores():
    x = make_periodic(n=2000, period=60)
    y = x.copy()
    y[900:1000] = 7.0                     # change only the test part
    sx = nn_shape_score(x, train_end=800, m=40)
    sy = nn_shape_score(y, train_end=800, m=40)
    np.testing.assert_allclose(sx[:800], sy[:800])


def test_shape_score_rejects_short_train_part():
    x = make_periodic(n=1000)
    try:
        nn_shape_score(x, train_end=50, m=40)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_iforest_is_reproducible_with_the_same_seed():
    x = make_periodic(n=4000)
    a, cs_a = iforest_score(x, train_end=2000, m=40, seed=3)
    b, cs_b = iforest_score(x, train_end=2000, m=40, seed=3)
    np.testing.assert_allclose(a, b)
    assert cs_a == cs_b == 1400


def test_iforest_finds_a_burst_of_high_variance():
    x = make_periodic(n=4000)
    rng = np.random.default_rng(5)
    x[3000:3100] += 3.0 * rng.standard_normal(100)
    scores, calib_start = iforest_score(x, train_end=2000, m=40)
    assert scores[3000:3140].max() > scores[calib_start:2000].max()


def test_iforest_scores_before_fit_end_do_not_define_the_threshold_region():
    x = make_periodic(n=4000)
    scores, calib_start = iforest_score(x, train_end=2000, m=40)
    assert 0 < calib_start < 2000
    assert len(scores) == len(x)
