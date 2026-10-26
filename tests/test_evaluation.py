import numpy as np

from signalsleuth.evaluation import make_labels, confusion_counts


def test_make_labels_includes_both_ends():
    labels = make_labels(length=10, start=3, end=5)
    assert labels.sum() == 3
    assert labels[2] == 0 and labels[3] == 1 and labels[5] == 1 and labels[6] == 0


def test_confusion_counts_accepts_bool_and_int():
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    alerts_int = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    alerts_bool = alerts_int.astype(bool)
    assert confusion_counts(labels, alerts_int) == (1, 2, 2)
    assert confusion_counts(labels, alerts_bool) == (1, 2, 2)


def test_series_011_window_500_numbers():
    # same counts as my real run: anomaly = 301 points,
    # 6 alerts inside it, 7 alerts outside it
    labels = make_labels(length=30000, start=11800, end=12100)
    alerts = np.zeros(30000, dtype=bool)
    alerts[11800:11806] = True   # 6 inside
    alerts[13000:13007] = True   # 7 outside
    assert confusion_counts(labels, alerts) == (6, 7, 295)

from signalsleuth.evaluation import precision_recall_f1


def test_prf_series_011_window_500():
    p, r, f1 = precision_recall_f1(6, 7, 295)
    assert abs(p - 6 / 13) < 1e-9
    assert abs(r - 6 / 301) < 1e-9
    assert abs(f1 - 12 / 314) < 1e-9


def test_prf_series_011_window_100_nothing_found():
    assert precision_recall_f1(0, 5, 301) == (0.0, 0.0, 0.0)


def test_prf_no_alerts_gives_no_nan():
    p, r, f1 = precision_recall_f1(0, 0, 301)
    assert not np.isnan(p) and not np.isnan(r) and not np.isnan(f1)

from signalsleuth.evaluation import point_adjust


def test_point_adjust_small_example():
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    alerts = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    adjusted = point_adjust(labels, alerts)
    assert confusion_counts(labels, adjusted) == (3, 2, 0)


def test_point_adjust_does_nothing_if_segment_missed():
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    alerts = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    adjusted = point_adjust(labels, alerts)
    assert (adjusted == alerts.astype(bool)).all()


def test_point_adjust_does_not_change_input():
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])
    alerts = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0])
    before = alerts.copy()
    point_adjust(labels, alerts)
    assert (alerts == before).all()


def test_point_adjust_series_011_numbers():
    labels = make_labels(30000, 11800, 12100)
    alerts = np.zeros(30000, dtype=bool)
    alerts[11800:11806] = True   # 6 inside
    alerts[13000:13007] = True   # 7 outside
    assert confusion_counts(labels, point_adjust(labels, alerts)) == (301, 7, 0)


def test_point_adjust_two_segments_only_hit_one_is_adjusted():
    labels = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0])
    alerts = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])
    adjusted = point_adjust(labels, alerts)
    assert confusion_counts(labels, adjusted) == (2, 0, 3)

from signalsleuth.evaluation import count_segments


def test_count_segments_basic():
    assert count_segments(np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])) == 3


def test_count_segments_edges_and_empty():
    assert count_segments(np.array([0, 0, 0, 0])) == 0
    assert count_segments(np.array([1, 1, 1, 1])) == 1
    assert count_segments(np.array([1, 0, 1])) == 2


def test_count_segments_accepts_bool():
    assert count_segments(np.array([True, False, True, True])) == 2


def test_count_segments_false_alarms_series_011_style():
    # 7 false-alarm points: as ONE group vs as THREE groups (positions are made up)
    one = np.zeros(30000, dtype=bool)
    one[13000:13007] = True
    three = np.zeros(30000, dtype=bool)
    three[13000:13003] = True
    three[15000:15002] = True
    three[20000:20002] = True
    assert one.sum() == three.sum() == 7
    assert count_segments(one) == 1
    assert count_segments(three) == 3

from signalsleuth.evaluation import evaluate_alerts


def test_evaluate_series_011_window_500():
    labels = make_labels(30000, 11800, 12100)
    alerts = np.zeros(30000, dtype=bool)
    alerts[11800:11806] = True    # 6 inside
    alerts[13000:13007] = True    # 7 outside, one group
    r = evaluate_alerts(labels, alerts)
    assert (r["tp"], r["fp"], r["fn"]) == (6, 7, 295)
    assert abs(r["f1"] - 12 / 314) < 1e-9
    assert abs(r["pa_f1"] - 602 / 609) < 1e-9
    assert r["false_alarm_events"] == 1
    assert r["anomaly_detected"] is True


def test_evaluate_series_011_window_100_nothing_found():
    labels = make_labels(30000, 11800, 12100)
    alerts = np.zeros(30000, dtype=bool)
    alerts[13000:13005] = True    # 5 false alarms, none inside
    r = evaluate_alerts(labels, alerts)
    assert r["f1"] == 0.0 and r["pa_f1"] == 0.0
    assert r["anomaly_detected"] is False


def test_false_alarm_points_vs_events():
    labels = make_labels(100, 40, 50)
    alerts = np.zeros(100, dtype=bool)
    alerts[[5, 6, 7, 80, 90]] = True    # 5 false-alarm points, 3 separate groups
    r = evaluate_alerts(labels, alerts)
    assert r["fp"] == 5
    assert r["false_alarm_events"] == 3