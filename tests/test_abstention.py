import numpy as np
import pandas as pd

from signalsleuth.abstention import (EXCESS_CAP, add_drift, calibration_scale, confidence_cut,
                                     excess, find_events, overlaps_anomaly, pseudo_series,
                                     risk_coverage, series_events, summarize_at_cut)
from signalsleuth.data import Series


def make_series(n=6000, train_end=2500, seed=0, spike=None):
    rng = np.random.default_rng(seed)
    x = np.sin(2 * np.pi * np.arange(n) / 60) + 0.1 * rng.standard_normal(n)
    if spike is not None:
        x[spike] += 6.0
    return Series("t", x, train_end, 4000, 4050)


# ---- events -----------------------------------------------------------------
def test_find_events_uses_strict_threshold_and_range():
    s = np.zeros(100)
    s[10] = 5.0
    s[50] = 3.0
    assert find_events(s, 0, 100, 3.0) == [(10, 10, 5.0)]          # 3.0 is not > 3.0
    assert find_events(s, 20, 100, 1.0) == [(50, 50, 3.0)]         # event at 10 is outside the range


def test_find_events_merges_close_runs_and_keeps_far_ones_apart():
    s = np.zeros(1000)
    s[100:103] = 2.0
    s[150:152] = 4.0                  # 47 non-alert points after the first run
    s[600] = 1.0                      # far away
    ev = find_events(s, 0, 1000, 0.5, merge_gap=50)
    assert ev == [(100, 151, 4.0), (600, 600, 1.0)]
    assert len(find_events(s, 0, 1000, 0.5, merge_gap=10)) == 3


def test_find_events_empty():
    assert find_events(np.zeros(50), 0, 50, 1.0) == []


# ---- strength ----------------------------------------------------------------
def test_scale_never_zero_and_excess_capped():
    flat = np.zeros(100)
    sc = calibration_scale(flat, 0, 100, 0.0)
    assert sc > 0
    assert excess(5.0, 0.0, sc) == EXCESS_CAP


def test_excess_grows_with_the_peak():
    assert excess(12.0, 10.0, 2.0) == 1.0
    assert excess(14.0, 10.0, 2.0) > excess(12.0, 10.0, 2.0)


def test_overlaps_anomaly_uses_tolerance():
    assert overlaps_anomaly(1000, 1010, 1100, 1120)        # tolerance 100 reaches 1000..1220
    assert not overlaps_anomaly(500, 600, 1100, 1120)
    assert overlaps_anomaly(1300, 1400, 1000, 1400)        # anomaly 401 long: tolerance 401


# ---- backtest uses only the train part ---------------------------------------
def test_pseudo_series_is_the_train_part_only():
    s = make_series()
    p = pseudo_series(s)
    assert len(p.values) == s.train_end and p.train_end == int(0.7 * s.train_end)
    assert np.array_equal(p.values, s.values[:s.train_end])


def test_pseudo_phase_does_not_depend_on_the_test_part():
    a = make_series()
    b = make_series()
    b.values = b.values.copy()
    b.values[a.train_end:] += 9.0                          # change only the test part
    ea = series_events(a, "diff")
    eb = series_events(b, "diff")
    assert ea["pseudo"] == eb["pseudo"]
    assert ea["real"]["threshold"] == eb["real"]["threshold"]   # threshold is from train data


def test_real_events_find_a_planted_spike_and_mark_it_true():
    s = make_series(spike=4010)
    r = series_events(s, "diff")["real"]
    true_events = [e for e in r["events"] if e["is_true"]]
    assert true_events and max(e["excess"] for e in true_events) > 5


# ---- drift -------------------------------------------------------------------
def test_drift_changes_only_the_test_part():
    x = np.sin(np.arange(1000) / 7.0)
    for kind in ("level", "trend", "scale"):
        y = add_drift(x, 400, kind, 2.0)
        assert np.array_equal(y[:400], x[:400])
        assert not np.allclose(y[400:], x[400:])
    assert np.allclose(add_drift(x, 400, "level", 0.0), x)


# ---- cut and evaluation ------------------------------------------------------
def test_confidence_cut_is_the_quantile():
    assert abs(confidence_cut(np.arange(101), 0.9) - 90.0) < 1e-9
    assert confidence_cut([], 0.9) == float("inf")


def _events():
    return pd.DataFrame({"name": ["a", "a", "b", "b", "c"],
                         "excess": [5.0, 1.0, 3.0, 0.5, 9.0],
                         "is_true": [True, False, False, False, True]})


def test_summarize_at_cut_counts():
    ev = _events()
    r = summarize_at_cut(ev, ["a", "b", "c", "d"], cut=2.0)     # confident: a:5(true) b:3(false) c:9(true)
    assert r["n_confident"] == 3 and r["n_events"] == 5
    assert r["false_events_all"].tolist() == [1, 2, 0, 0]
    assert r["false_events_confident"].tolist() == [0, 1, 0, 0]
    assert r["caught_confident"].tolist() == [True, False, True, False]
    assert abs(r["event_precision"] - 2 / 3) < 1e-12
    assert abs(r["false_confident_rate"] - 1 / 3) < 1e-12       # 1 of the 3 false events reaches the cut
    assert r["true_demoted"] == 0


def test_summarize_cut_zero_keeps_everything_and_cut_inf_keeps_nothing():
    ev = _events()
    assert summarize_at_cut(ev, ["a", "b", "c"], 0.0)["n_confident"] == 5
    r = summarize_at_cut(ev, ["a", "b", "c"], float("inf"))
    assert r["n_confident"] == 0 and r["true_demoted"] == 2


def test_risk_coverage_for_a_useful_score_goes_down():
    ev = _events()
    cov, risk, base = risk_coverage(ev)
    assert abs(base - 3 / 5) < 1e-12
    assert risk[0] == 0.0                     # strongest event (9.0) is true
    assert cov[-1] == 1.0 and abs(risk[-1] - base) < 1e-12
