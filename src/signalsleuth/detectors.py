import numpy as np
import pandas as pd


def rolling_mad_score(values, window=100):
    # give every point an anomaly score, using only the points before it

    # put the numbers in a pandas Series so I can use rolling()
    s = pd.Series(values)

    # median of the last `window` points = what is "normal" right now
    # shift(1) so the current point is not part of its own median (no leakage)
    med = s.rolling(window, min_periods=window).median().shift(1)

    # how far each point is from the normal value (abs removes the minus sign)
    abs_dev = (s - med).abs()

    # MAD = the median of those distances over the last `window` points
    # it tells me how far points usually are from normal
    mad = abs_dev.rolling(window, min_periods=window).median().shift(1)

    # score = distance / usual distance
    # 1.4826 makes MAD comparable to standard deviation
    # 1e-8 avoids dividing by zero when the series is flat
    score = abs_dev / (1.4826 * mad + 1e-8)

    # the first points have no history (NaN), so I set their score to 0
    return score.fillna(0.0).to_numpy()