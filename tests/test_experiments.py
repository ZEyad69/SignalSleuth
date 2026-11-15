import numpy as np

from signalsleuth.experiments import run_series


def make_fake_ucr_file(tmp_path, anomaly_value):
    # 500 normal points; train ends at 300; anomaly = points 400..409
    x = np.array([10, 11, 10, 12, 11] * 100, dtype=float)
    x[400:410] = anomaly_value
    path = tmp_path / "001_UCR_Anomaly_fake_300_400_409.txt"
    np.savetxt(path, x)
    return path


def test_run_series_finds_planted_anomaly(tmp_path):
    path = make_fake_ucr_file(tmp_path, anomaly_value=50.0)
    r = run_series(path, window=5)
    assert r["anomaly_detected"] is True


def test_threshold_uses_only_train_part(tmp_path):
    # the two files differ ONLY after the train part (index 400..409)
    r1 = run_series(make_fake_ucr_file(tmp_path, 50.0), window=5)
    other = tmp_path / "other"
    other.mkdir()
    r2 = run_series(make_fake_ucr_file(other, 500.0), window=5)
    assert r1["threshold"] == r2["threshold"]


def test_run_series_returns_name(tmp_path):
    path = make_fake_ucr_file(tmp_path, anomaly_value=50.0)
    r = run_series(path, window=5)
    assert r["name"] == "001_UCR_Anomaly_fake_300_400_409"