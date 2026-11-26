"""Check nn_shape_score against stumpy's matrix profile (an independent implementation).

Usage:
    py scripts\\check_with_stumpy.py                 (a made-up periodic series)
    py scripts\\check_with_stumpy.py <ucr_file.txt>  (a real UCR file)

Idea: stumpy.stump(T_A, m, T_B, ignore_trivial=False) gives, for every window of T_A,
the distance to its nearest window in T_B (z-normalised Euclidean distance).
Our score is the same thing with T_B = the reference part and T_A = everything after it.
For windows that start after the reference part there is no overlap, so the two must agree.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import stumpy
from numpy.lib.stride_tricks import sliding_window_view

from signalsleuth.advanced_detectors import nn_shape_score
from signalsleuth.data import load_series

M = 100
CALIB_FRACTION = 0.3

if len(sys.argv) > 1:
    s = load_series(sys.argv[1])
    x, train_end = s.values.astype(float), s.train_end
    # stumpy gets slow on long series, so cut the test part
    x = x[: train_end + 20000]
    print("file:", s.name)
else:
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * np.arange(6000) / 70) + 0.1 * rng.standard_normal(6000)
    train_end = 2500
    print("made-up series")

ref_end = int(train_end * (1 - CALIB_FRACTION))

# our scores (sit on the LAST point of each window)
ours = nn_shape_score(x, train_end, m=M, ref_end=ref_end)

# stumpy: windows of x[ref_end:] against the reference x[:ref_end]
T_A, T_B = x[ref_end:], x[:ref_end]
P = stumpy.stump(T_A, M, T_B, ignore_trivial=False)
theirs = P[:, 0].astype(float)                       # one distance per window of T_A
ours_aligned = ours[ref_end + M - 1: ref_end + M - 1 + len(theirs)]

# flat windows are treated differently by the two methods, so tell the user if there are any
n_flat = int((sliding_window_view(x, M).std(axis=1) < 1e-8).sum())
print("flat windows in the series:", n_flat, "(if > 0, small differences are expected there)")

diff = np.abs(ours_aligned - theirs)
print("windows compared:", len(theirs))
print("max abs difference:   %.5f" % diff.max())
print("mean abs difference:  %.5f" % diff.mean())
print("largest score (ours / stumpy): %.3f / %.3f" % (ours_aligned.max(), theirs.max()))
print("same top window?", int(ours_aligned.argmax()) == int(theirs.argmax()))
print("OK" if diff.max() < 0.05 else "DIFFERENT: look at the windows with the largest difference")
