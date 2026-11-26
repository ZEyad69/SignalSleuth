"""Two detectors that go beyond rolling median + MAD.

Both return a score for every point. A higher score means "more anomalous".
The score at point t only uses data up to t (the window ENDS at t), and only the
train part of the series is used as reference or as training data.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# 1. Shape-aware detector: distance to the nearest normal shape
# ---------------------------------------------------------------------------
def _znorm(windows):
    """Z-normalise every window (row). Flat windows (std = 0) become all zeros."""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    flat = std[:, 0] < 1e-8
    z = (windows - mean) / np.where(std < 1e-8, 1.0, std)
    z[flat] = 0.0
    return z.astype(np.float32), flat


def nn_shape_score(values, train_end, m=100, ref_end=None):
    """For every window of m points, the distance to its most similar window in the REFERENCE part.

    The reference part is the first `ref_end` points (default: the whole train part).
    Pass ref_end < train_end to keep the rest of the train part as calibration data.

    Windows are z-normalised, so the score measures SHAPE, not height or offset.
    Overlapping windows are never compared with each other (no trivial matches).
    The score is placed on the LAST point of the window.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if ref_end is None:
        ref_end = train_end
    if ref_end > train_end:
        raise ValueError("ref_end cannot be larger than train_end")
    if ref_end < 3 * m:
        raise ValueError("reference part is too short for this window length")
    if n < train_end:
        raise ValueError("train_end is larger than the series")

    windows = sliding_window_view(x, m)              # view, no copy: (n - m + 1, m)
    n_win = len(windows)
    ref_n = ref_end - m + 1                          # windows fully inside the reference part
    z_ref, ref_flat = _znorm(windows[:ref_n])
    ref_flat_idx = np.where(ref_flat)[0]

    scores = np.zeros(n)
    chunk = int(max(16, min(2048, 2e7 // ref_n)))    # keeps the distance matrix small
    for i0 in range(0, n_win, chunk):
        i1 = min(i0 + chunk, n_win)
        z_q, q_flat = _znorm(windows[i0:i1])
        corr = (z_q @ z_ref.T) / m                   # correlation of every query with every reference

        # two flat windows are identical (correlation 1), a flat and a non-flat one get 0
        q_flat_idx = np.where(q_flat)[0]
        if len(q_flat_idx) and len(ref_flat_idx):
            corr[np.ix_(q_flat_idx, ref_flat_idx)] = 1.0

        # never compare a window with a train window that overlaps it
        if i0 <= ref_n - 1 + m - 1:
            for row in range(i1 - i0):
                q = i0 + row
                lo = max(0, q - m + 1)
                hi = min(ref_n, q + m)
                if lo < hi:
                    corr[row, lo:hi] = -np.inf

        best = corr.max(axis=1)
        dist_sq = np.maximum(2.0 * m * (1.0 - best), 0.0)
        dist = np.sqrt(dist_sq)
        dist[~np.isfinite(dist)] = 0.0
        scores[i0 + m - 1:i1 + m - 1] = dist
    return scores


# ---------------------------------------------------------------------------
# 2. Isolation Forest on window features
# ---------------------------------------------------------------------------
def _window_features(x, m, start_idx=0, stop_idx=None):
    """Features of every window of m points: mean, std, min, max, mean absolute step."""
    windows = sliding_window_view(x, m)
    stop = len(windows) if stop_idx is None else stop_idx
    out = []
    for i0 in range(start_idx, stop, 50000):
        w = windows[i0:min(i0 + 50000, stop)]
        out.append(np.column_stack([
            w.mean(axis=1), w.std(axis=1), w.min(axis=1), w.max(axis=1),
            np.abs(np.diff(w, axis=1)).mean(axis=1),
        ]))
    return np.vstack(out)


def iforest_score(values, train_end, m=100, seed=0, calib_fraction=0.3, max_fit=20000):
    """Isolation Forest trained on the first part of the train data.

    The forest is fitted on windows from the first (1 - calib_fraction) of the train part.
    The rest of the train part is kept as calibration data: the threshold must be taken
    from it, because scores of the data the forest was fitted on are too optimistic.
    Returns (scores, calib_start).
    """
    x = np.asarray(values, dtype=float)
    fit_end = int(train_end * (1.0 - calib_fraction))     # windows must end before this point
    n_fit_windows = fit_end - m + 1
    if n_fit_windows < 200:
        raise ValueError("train part is too short to fit the Isolation Forest")

    feats_fit = _window_features(x, m, 0, n_fit_windows)
    if len(feats_fit) > max_fit:                          # evenly spaced subsample, no randomness
        idx = np.linspace(0, len(feats_fit) - 1, max_fit).astype(int)
        feats_fit = feats_fit[idx]

    forest = IsolationForest(n_estimators=200, random_state=seed)
    forest.fit(feats_fit)

    feats_all = _window_features(x, m)
    window_scores = -forest.score_samples(feats_all)      # higher = more anomalous
    scores = np.zeros(len(x))
    scores[m - 1:] = window_scores                        # score sits on the last point of the window
    return scores, fit_end
