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