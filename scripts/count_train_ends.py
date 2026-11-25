from pathlib import Path

DATA_DIR = Path(
    r"C:\UCR_TimeSeriesAnomalyDatasets2021\AnomalyDatasets_2021"
    r"\UCR_TimeSeriesAnomalyDatasets2021\FilesAreInHere\UCR_Anomaly_FullData"
)

# all 250 files, sorted by name (we only read the names, we never open the files)
files = sorted(DATA_DIR.glob("*.txt"))

train_ends = []
for f in files:
    parts = f.stem.split("_")
    # the last 3 parts of the name are always: train_end, anomaly_start, anomaly_end
    train_ends.append(int(parts[-3]))

# sanity check: one train_end for every file
assert len(train_ends) == len(files)

print("files:", len(files))
print("smallest train_end:", min(train_ends))
print("10 smallest:", sorted(train_ends)[:10])
print("below 200:", sum(1 for t in train_ends if t < 200))
print("below 1000:", sum(1 for t in train_ends if t < 1000))