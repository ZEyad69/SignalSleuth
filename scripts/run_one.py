import sys
from pathlib import Path

# let Python find our package inside src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from signalsleuth.experiments import run_series

DATA_DIR = Path(
    r"C:\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021"
    r"\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"
)
FILE = next(DATA_DIR.glob("011_*"))

result = run_series(FILE, window=500)

# print every key and its value, one per line
for key, value in result.items():
    print(key, value)