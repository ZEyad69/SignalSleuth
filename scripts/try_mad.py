import sys
from pathlib import Path

# let Python find our package inside src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from signalsleuth.detectors import rolling_mad_score

# a normal repeating pattern, 30 points
x = np.array([10, 11, 10, 12, 11] * 6, dtype=float)

# plant one anomaly at index 25 (I know where it is)
x[25] = 50

# run my detector, small window because the series is tiny
scores = rolling_mad_score(x, window=5)

print("scores:", np.round(scores, 2))
print("highest score at index:", scores.argmax())