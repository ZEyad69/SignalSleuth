import numpy as np

from signalsleuth.evaluation import evaluate_alerts, make_labels
from signalsleuth.evaluation_extra import pa_k_auc, pa_k_f1_curve


def test_k0_is_point_adjustment_and_k100_is_plain_f1_on_random_cases():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = 120
        labels = np.zeros(n, dtype=int)
        for _seg in range(rng.integers(1, 4)):            # up to 3 true segments
            s = rng.integers(0, n - 10)
            labels[s:s + rng.integers(1, 10)] = 1
        alerts = rng.random(n) < rng.choice([0.02, 0.1, 0.4])
        r = evaluate_alerts(labels, alerts)
        curve = pa_k_f1_curve(labels, alerts, ks=[0, 100])
        assert abs(curve[0] - r["pa_f1"]) < 1e-12
        assert abs(curve[1] - r["f1"]) < 1e-12


def test_f1_never_increases_when_k_increases():
    rng = np.random.default_rng(1)
    for _ in range(100):
        labels = make_labels(200, 80, 130)
        alerts = rng.random(200) < 0.15
        curve = pa_k_f1_curve(labels, alerts)
        assert np.all(np.diff(curve) <= 1e-12)


def test_one_alert_in_a_long_anomaly_only_counts_for_small_k():
    labels = make_labels(1000, 500, 599)                   # 100 points
    alerts = np.zeros(1000, dtype=bool)
    alerts[550] = True                                     # 1% of the segment
    curve = pa_k_f1_curve(labels, alerts, ks=[0, 0.5, 1, 20, 100])
    assert curve[0] == 1.0 and curve[1] == 1.0             # share 1% > K for K < 1
    assert curve[2] < 0.05 and curve[3] < 0.05 and curve[4] < 0.05   # K = 1: 1% > 1% is false


def test_auc_between_zero_and_one_and_perfect_detector_gets_one():
    labels = make_labels(500, 200, 260)
    perfect = labels.astype(bool)
    assert abs(pa_k_auc(labels, perfect) - 1.0) < 1e-12
    nothing = np.zeros(500, dtype=bool)
    assert pa_k_auc(labels, nothing) == 0.0


def test_auc_of_a_lucky_single_alert_is_small():
    labels = make_labels(10000, 5000, 5299)                # 300 points
    alerts = np.zeros(10000, dtype=bool)
    alerts[5100] = True
    assert evaluate_alerts(labels, alerts)["pa_f1"] == 1.0   # PA says perfect
    assert pa_k_auc(labels, alerts) < 0.02                   # PA%K-AUC does not
