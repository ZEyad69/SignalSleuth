"""Abstention ("not sure") for anomaly alerts.

Three states for every candidate alert EVENT:
    normal      score never goes above the base threshold (the highest calibration score)
    uncertain   an alert event whose strength is below a second, higher cut
    confident   an alert event whose strength is at or above that cut

Why events and not points: window-based detectors give a long run of high scores for one oddity
(10 or more points per false alarm), so points are not independent but events are close to it.

Why a backtest: the calibration part cannot say anything about scores above its own maximum.
So the second cut is learned on a PSEUDO-TEST: the same pipeline is run on the normal train part
alone (the first 70% of it plays the train part, the last 30% plays the test part). Every alert
event there is a false alarm by construction. Its strength (see `excess`) is recorded, and the cut
is a high quantile of those strengths pooled over many series. Nothing from the real test part,
and no label, is used.
"""
import numpy as np

from signalsleuth.compare import compute_scores
from signalsleuth.data import Series

MERGE_GAP = 100          # alert runs closer than this many points belong to one event
PSEUDO_FRACTION = 0.7    # share of the train part that plays "train" in the backtest
EXCESS_CAP = 1e3         # keeps flat series (scale ~ 0) from producing infinite strengths


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
def find_events(scores, start, stop, threshold, merge_gap=MERGE_GAP):
    """Alert events inside [start, stop): runs of points with score > threshold.

    Runs separated by at most `merge_gap` non-alert points are merged into one event.
    Returns a list of (first_index, last_index, peak_score), indices in the full series.
    """
    seg = np.asarray(scores[start:stop], dtype=float)
    idx = np.flatnonzero(seg > threshold)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > merge_gap + 1)      # gap = difference - 1
    firsts = np.concatenate([[idx[0]], idx[breaks + 1]])
    lasts = np.concatenate([idx[breaks], [idx[-1]]])
    return [(int(f) + start, int(l) + start, float(seg[f:l + 1].max()))
            for f, l in zip(firsts, lasts)]


def calibration_scale(scores, calib_start, calib_end, threshold):
    """Spread of the calibration scores: (highest - median). Never zero."""
    med = float(np.median(np.asarray(scores[calib_start:calib_end], dtype=float)))
    return max(threshold - med, 1e-8 * max(1.0, abs(threshold)))


def excess(peak, threshold, scale):
    """Strength of an alert event: how far its peak is above the threshold, in calibration spreads."""
    return float(min((peak - threshold) / scale, EXCESS_CAP))


def overlaps_anomaly(ev_first, ev_last, a_start, a_end, min_tol=100):
    """An event is TRUE if it touches the anomaly extended by max(min_tol, anomaly length)."""
    tol = max(min_tol, a_end - a_start + 1)
    return ev_last >= a_start - tol and ev_first <= a_end + tol


# ---------------------------------------------------------------------------
# backtest on the train part
# ---------------------------------------------------------------------------
def pseudo_series(series, pseudo_fraction=PSEUDO_FRACTION):
    """The train part alone, with its own train / pseudo-test split. No anomaly: all normal."""
    te = int(series.train_end * pseudo_fraction)
    return Series(name=series.name + "__pseudo", values=series.values[:series.train_end],
                  train_end=te, anomaly_start=-1, anomaly_end=-1)


def _phase(series, detector, calib_fraction, stop, merge_gap, truth, **params):
    scores, calib_start, thr = compute_scores(series, detector, calib_fraction, **params)
    scale = calibration_scale(scores, calib_start, series.train_end, thr)
    events = []
    for first, last, peak in find_events(scores, series.train_end, stop, thr, merge_gap):
        is_true = overlaps_anomaly(first, last, *truth) if truth else False
        events.append({"start": first, "end": last, "peak": peak,
                       "excess": excess(peak, thr, scale), "is_true": bool(is_true)})
    return {"threshold": thr, "scale": scale, "events": events}


def series_events(series, detector, calib_fraction=0.3, pseudo_fraction=PSEUDO_FRACTION,
                  merge_gap=MERGE_GAP, **params):
    """Alert events of one series: the real test part and the pseudo-test inside the train part."""
    real = _phase(series, detector, calib_fraction, len(series.values), merge_gap,
                  (series.anomaly_start, series.anomaly_end), **params)
    pseudo = _phase(pseudo_series(series, pseudo_fraction), detector, calib_fraction,
                    series.train_end, merge_gap, None, **params)
    return {"real": real, "pseudo": pseudo}


# ---------------------------------------------------------------------------
# drift (stress test, applied to the TEST part only)
# ---------------------------------------------------------------------------
def add_drift(values, train_end, kind, strength):
    """Change the test part of a series; the train part is never touched.

    level: add a constant step of `strength` train standard deviations.
    trend: add a ramp that grows from 0 to `strength` train standard deviations.
    scale: stretch the signal around the train mean, from x1 to x(1 + strength).
    """
    x = np.asarray(values, dtype=float).copy()
    sd = float(x[:train_end].std()) or 1.0
    ramp = np.linspace(0.0, 1.0, len(x) - train_end)
    if kind == "level":
        x[train_end:] += strength * sd
    elif kind == "trend":
        x[train_end:] += strength * sd * ramp
    elif kind == "scale":
        m = float(x[:train_end].mean())
        x[train_end:] = m + (x[train_end:] - m) * (1.0 + strength * ramp)
    else:
        raise ValueError("kind must be level, trend or scale")
    return x


# ---------------------------------------------------------------------------
# cut and evaluation (work on plain numpy arrays / pandas frames)
# ---------------------------------------------------------------------------
def confidence_cut(pseudo_excess, q):
    """Strength cut = the q-quantile of the strengths of the pseudo-test (false) alert events."""
    v = np.asarray(pseudo_excess, dtype=float)
    if len(v) == 0:
        return float("inf")
    return float(np.quantile(v, q))


def summarize_at_cut(events, series_names, cut):
    """Series-level and event-level numbers for one cut.

    `events`: frame of REAL alert events with columns name, excess, is_true.
    `series_names`: all series of this detector, also those with no event.
    Returns a dict of per-series arrays and totals.
    """
    names = list(series_names)
    conf = events[events["excess"] >= cut]
    base_false = events[~events["is_true"]].groupby("name").size().reindex(names, fill_value=0)
    conf_false = conf[~conf["is_true"]].groupby("name").size().reindex(names, fill_value=0)
    conf_true = conf[conf["is_true"]].groupby("name").size().reindex(names, fill_value=0)
    any_true = events[events["is_true"]].groupby("name").size().reindex(names, fill_value=0)
    n_conf = len(conf)
    return {
        "false_events_all": base_false.to_numpy(),
        "false_events_confident": conf_false.to_numpy(),
        "caught_all": (any_true.to_numpy() > 0),
        "caught_confident": (conf_true.to_numpy() > 0),
        "event_precision": float(conf["is_true"].mean()) if n_conf else float("nan"),
        "n_confident": n_conf,
        "n_events": len(events),
        # calibration check: share of the real false events that reach the cut
        "false_confident_rate": float((events[~events["is_true"]]["excess"] >= cut).mean())
        if (~events["is_true"]).any() else float("nan"),
        "true_demoted": int(((events["is_true"]) & (events["excess"] < cut)).sum()),
    }


def risk_coverage(events):
    """Sort alert events from strongest to weakest and keep the top k, for every k.

    coverage = share of events kept; risk = share of the kept events that are false alarms.
    Returns (coverage, risk, base_risk). `base_risk` is what keeping events at random would give.
    """
    ev = events.sort_values("excess", ascending=False, kind="mergesort")
    false = (~ev["is_true"].to_numpy()).astype(float)
    k = np.arange(1, len(ev) + 1)
    return k / max(len(ev), 1), np.cumsum(false) / np.maximum(k, 1), float(false.mean()) if len(ev) else float("nan")
