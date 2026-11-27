import numpy as np
import pytest

from signalsleuth.compare import run_detector
from signalsleuth.experiments import run_series


def make_fake_ucr_file(folder, test_part_value=None, n=2000, train_end=800,
                       start=1500, end=1559):
    """A periodic series with a SHAPE anomaly (square wave) between start and end."""
    rng = np.random.default_rng(0)
    t = np.arange(n)
    x = np.sin(2 * np.pi * t / 60) + 0.1 * rng.standard_normal(n)
    x[start:end + 1] = np.sign(np.sin(2 * np.pi * t[start:end + 1] / 60))
    if test_part_value is not None:
        x[1000:1100] = test_part_value        # a change that lies only in the test part
    folder.mkdir(exist_ok=True)
    path = folder / ("001_UCR_Anomaly_fake_%d_%d_%d.txt" % (train_end, start, end))
    np.savetxt(path, x)
    return path


PARAMS = {"mad": {"window": 20}, "shape": {"m": 40}, "iforest": {"m": 40}, "diff": {}, "random": {}}
ALL = ["mad", "shape", "iforest", "diff", "random"]


def test_mad_through_run_detector_equals_run_series(tmp_path):
    path = make_fake_ucr_file(tmp_path)
    a = run_series(path, window=20)
    b = run_detector(path, "mad", calib_fraction=0, window=20)   # 0 = the old whole-train rule
    for key in a:
        assert a[key] == b[key], key


@pytest.mark.parametrize("detector", ALL)
def test_result_has_the_expected_keys(tmp_path, detector):
    path = make_fake_ucr_file(tmp_path)
    r = run_detector(path, detector, **PARAMS[detector])
    for key in ["tp", "fp", "fn", "precision", "recall", "f1", "pa_f1",
                "false_alarm_events", "anomaly_detected", "name", "detector", "threshold",
                "ucr_hit", "top_index", "pak_auc", "anomaly_len", "train_end", "n"]:
        assert key in r
    assert r["detector"] == detector


@pytest.mark.parametrize("detector", ALL)
def test_threshold_does_not_depend_on_the_test_part(tmp_path, detector):
    r1 = run_detector(make_fake_ucr_file(tmp_path), detector, **PARAMS[detector])
    other = tmp_path / "other"
    r2 = run_detector(make_fake_ucr_file(other, test_part_value=9.0), detector,
                      **PARAMS[detector])
    assert r1["threshold"] == r2["threshold"]


def test_shape_detector_finds_a_shape_change(tmp_path):
    path = make_fake_ucr_file(tmp_path)
    shape = run_detector(path, "shape", m=40)
    assert shape["anomaly_detected"] is True


def test_threshold_comes_from_the_calibration_part_only(tmp_path):
    # new rule: the threshold is the max score in the LAST 30% of the train part,
    # so a spike in the FIRST 70% of the train part must not change it
    r1 = run_detector(make_fake_ucr_file(tmp_path), "diff")
    other = tmp_path / "other"
    other.mkdir()
    path = make_fake_ucr_file(other)
    x = np.loadtxt(path)
    x[300] += 20.0                      # inside the fit part (0..559), not the calibration part
    np.savetxt(path, x)
    r2 = run_detector(path, "diff")
    assert r1["threshold"] == r2["threshold"]


def test_ucr_hit_for_the_shape_detector(tmp_path):
    r = run_detector(make_fake_ucr_file(tmp_path), "shape", m=40)
    assert r["ucr_hit"] is True
    assert 1500 - 100 <= r["top_index"] <= 1559 + 100


def test_random_baseline_is_reproducible(tmp_path):
    path = make_fake_ucr_file(tmp_path)
    a = run_detector(path, "random", seed=4)
    b = run_detector(path, "random", seed=4)
    assert a["threshold"] == b["threshold"] and a["top_index"] == b["top_index"]
