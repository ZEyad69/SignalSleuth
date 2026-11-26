"""Two training-free baselines. Both give one score per point (higher = more anomalous).

- abs_diff_score: how much the value jumped since the previous point.
- random_score:   pure noise. This is the floor: any real detector must beat it.
The score at point t only uses data up to t.
"""
import numpy as np


def abs_diff_score(values):
    x = np.asarray(values, dtype=float)
    scores = np.zeros(len(x))
    scores[1:] = np.abs(np.diff(x))
    return scores


def random_score(values, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(len(values))
