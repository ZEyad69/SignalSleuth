import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from signalsleuth.evaluation_extra import mcnemar_exact

df = pd.read_csv(Path(__file__).resolve().parents[1] / "reports" / "compare_30_v2.csv")
df = df[df["error"].isna()]                                   # failed runs have no result
hit = df.pivot(index="name", columns="detector", values="ucr_hit").astype(bool)
print("shape only, diff only, p =", mcnemar_exact(hit["shape"], hit["diff"]))
