"""Extra evaluation tools: a threshold-free metric, bootstrap CIs and paired tests."""
import numpy as np
from scipy import stats


def ucr_top1_hit(scores, train_end, anomaly_start, anomaly_end, min_tol=100):
    """UCR-style metric, needs NO threshold.

    Take the single highest-scoring point AFTER the train part. It is a hit if it falls
    inside the anomaly extended by tol on each side, tol = max(min_tol, anomaly length).
    Returns (hit, index_of_the_top_point).
    """
    test = np.asarray(scores)[train_end:]
    if len(test) == 0:
        raise ValueError("no test part after train_end")
    top = int(np.argmax(test)) + train_end
    tol = max(min_tol, anomaly_end - anomaly_start + 1)
    hit = (anomaly_start - tol) <= top <= (anomaly_end + tol)
    return bool(hit), top


def bootstrap_ci(values, stat=np.mean, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI, resampling SERIES (one value per series).

    Returns (point_estimate, low, high).
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([stat(v[row]) for row in idx])
    low, high = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(low), float(high)


def mcnemar_exact(a, b):
    """Exact McNemar test for two detectors on the SAME series, binary outcomes (hit / miss).

    Only series where they disagree matter. Returns (a_only, b_only, p_value).
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(b & ~a))
    n = a_only + b_only
    p = 1.0 if n == 0 else float(stats.binomtest(a_only, n, 0.5).pvalue)
    return a_only, b_only, p


def wilcoxon_paired(a, b):
    """Wilcoxon signed-rank test for a numeric metric (e.g. F1) on the same series. Returns p."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a, b):
        return 1.0
    return float(stats.wilcoxon(a, b).pvalue)
