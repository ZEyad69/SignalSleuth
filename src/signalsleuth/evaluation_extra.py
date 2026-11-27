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


# ---------------------------------------------------------------------------
# PA%K (Kim et al., AAAI 2022): point adjustment only for well-detected segments
# ---------------------------------------------------------------------------
def _segments(mask):
    """Start (included) and end (excluded) of every run of True values."""
    m = np.concatenate([[0], np.asarray(mask).astype(int), [0]])
    d = np.diff(m)
    return np.where(d == 1)[0], np.where(d == -1)[0]


def pa_k_f1_curve(labels, alerts, ks=range(0, 101)):
    """F1 after PA%K for every K (in percent).

    A true anomaly segment is adjusted (counted as fully detected) only if the share of its
    points that are alerts is STRICTLY larger than K%. So K = 0 is ordinary PA (one alert is
    enough) and K = 100 is plain point-wise F1 (nothing is ever adjusted).
    Alerts outside the true segments are never changed, so they stay false alarms.
    """
    labels = np.asarray(labels).astype(bool)
    alerts = np.asarray(alerts).astype(bool)
    if labels.shape != alerts.shape:
        raise ValueError("labels and alerts must have the same length")

    starts, ends = _segments(labels)
    lengths = ends - starts
    hit_in_seg = np.array([alerts[s:e].sum() for s, e in zip(starts, ends)], dtype=float)
    share = hit_in_seg / np.maximum(lengths, 1)                 # share of detected points, 0..1
    fp = int(np.sum(alerts & ~labels))
    total_pos = int(labels.sum())

    f1 = []
    for k in ks:
        adjusted = share > (k / 100.0)
        tp = int(np.where(adjusted, lengths, hit_in_seg).sum())
        fn = total_pos - tp
        denom = 2 * tp + fp + fn
        f1.append(2 * tp / denom if denom > 0 else 0.0)         # same as 2PR/(P+R)
    return np.array(f1)


def pa_k_auc(labels, alerts, ks=range(0, 101)):
    """Area under the F1-vs-K curve, scaled to 0..1 (trapezoid rule, K from 0 to 100)."""
    ks = np.asarray(list(ks), dtype=float)
    f1 = pa_k_f1_curve(labels, alerts, ks)
    area = np.sum((f1[1:] + f1[:-1]) / 2.0 * np.diff(ks))
    return float(area / (ks[-1] - ks[0]))
