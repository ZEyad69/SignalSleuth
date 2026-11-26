import numpy as np

from signalsleuth.advanced_detectors import nn_shape_score


def make_periodic(n=3000, period=60, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(n)


def test_ref_end_default_is_the_old_behaviour():
    x = make_periodic()
    np.testing.assert_allclose(nn_shape_score(x, 1000, m=40),
                               nn_shape_score(x, 1000, m=40, ref_end=1000))


def test_ref_end_part_is_not_used_as_reference():
    x = make_periodic()
    y = x.copy()
    y[700:1000] = 9.0                      # change only the calibration part (ref_end = 700)
    sx = nn_shape_score(x, train_end=1000, m=40, ref_end=700)
    sy = nn_shape_score(y, train_end=1000, m=40, ref_end=700)
    # scores of windows that end before the change are identical: nothing leaks backwards
    np.testing.assert_allclose(sx[:700], sy[:700])
    # and scores after the change differ only because the QUERY changed, not the reference:
    # the reference is unchanged, so a normal window far after the change scores the same
    np.testing.assert_allclose(sx[1500:], sy[1500:])


def test_ref_end_rejects_bad_values():
    x = make_periodic()
    for bad in (50, 1200):
        try:
            nn_shape_score(x, train_end=1000, m=40, ref_end=bad)
            assert False, "should have raised ValueError"
        except ValueError:
            pass
