import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from signalsleuth.baselines import abs_diff_score, random_score
from signalsleuth.evaluation_extra import bootstrap_ci, mcnemar_exact, ucr_top1_hit


def test_ucr_hit_inside_and_outside():
    scores = np.zeros(1000)
    scores[600] = 1.0
    hit, top = ucr_top1_hit(scores, train_end=300, anomaly_start=550, anomaly_end=580)
    assert hit and top == 600                      # 600 is within 100 of 580
    hit, _ = ucr_top1_hit(scores, train_end=300, anomaly_start=800, anomaly_end=820)
    assert not hit


def test_ucr_ignores_train_part():
    scores = np.zeros(1000)
    scores[100] = 99.0                             # huge score inside train: must be ignored
    scores[700] = 1.0
    hit, top = ucr_top1_hit(scores, train_end=300, anomaly_start=690, anomaly_end=710)
    assert hit and top == 700


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    v = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=float)
    est, lo, hi = bootstrap_ci(v, n_boot=2000, seed=0)
    assert lo <= est <= hi
    assert bootstrap_ci(v, n_boot=2000, seed=0) == (est, lo, hi)


def test_mcnemar_counts_and_no_difference():
    a_only, b_only, p = mcnemar_exact([1, 1, 0, 1], [1, 0, 0, 1])
    assert (a_only, b_only) == (1, 0)
    assert mcnemar_exact([1, 0], [1, 0])[2] == 1.0


def test_baselines_shape_and_causality():
    x = np.array([1.0, 3.0, 2.0, 2.0])
    assert abs_diff_score(x).tolist() == [0.0, 2.0, 1.0, 0.0]
    assert len(random_score(x, seed=1)) == 4
    assert np.array_equal(random_score(x, seed=1), random_score(x, seed=1))
