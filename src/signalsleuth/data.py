from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Series:
    name: str
    values: np.ndarray
    train_end: int
    anomaly_start: int
    anomaly_end: int


def load_series(path):
    """Read a UCR anomaly file; parse split and anomaly location from its name."""
    path = Path(path)
    parts = path.stem.split("_")
    train_end, anomaly_start, anomaly_end = (int(x) for x in parts[-3:])
    values = np.loadtxt(path)
    return Series(
        name=path.stem,
        values=values,
        train_end=train_end,
        anomaly_start=anomaly_start,
        anomaly_end=anomaly_end,
    )