"""Turn reports/compare_250.csv into numbers we can defend.

Usage:  py scripts\\analyze_compare_250.py [path_to_csv]
Writes small CSV tables next to the input (reports/compare_250_*.csv) and prints everything.
Only series where ALL detectors ran are used, so every detector is compared on the same series.
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from signalsleuth.evaluation_extra import bootstrap_ci, mcnemar_exact, wilcoxon_paired

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "reports" / "compare_250.csv"
OUT = CSV.parent / CSV.stem
ORDER = ["mad", "shape", "iforest", "diff", "random"]
pd.set_option("display.width", 200)


def holm(pvals):
    """Holm step-down correction for a list of p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion (better than the bootstrap near 0 or 1)."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ci_text(values, stat=np.mean):
    est, lo, hi = bootstrap_ci(values, stat=stat)
    return "%.3f (%.3f to %.3f)" % (est, lo, hi)


# ---------------------------------------------------------------- load
raw = pd.read_csv(CSV)
ok = raw[raw["error"].isna()].copy()
for c in ["ucr_hit", "anomaly_detected"]:
    ok[c] = ok[c].astype(str).str.lower().eq("true").astype(float)
counts = ok.groupby("name")["detector"].nunique()
common = counts[counts == len(ORDER)].index
failed = raw[raw["error"].notna()].groupby("detector").size().to_dict()
ok = ok[ok["name"].isin(common)].sort_values(["detector", "name"])
print("series in file:", raw["name"].nunique(), "| used (all detectors ran):", len(common), "| failed rows:", failed or "none")


def wide(col):
    return ok.pivot(index="name", columns="detector", values=col)[ORDER]


hit, f1, paf1, pak = wide("ucr_hit"), wide("f1"), wide("pa_f1"), wide("pak_auc")
found, fa, rec, prec = wide("anomaly_detected"), wide("false_alarm_events"), wide("recall"), wide("precision")
meta = ok[ok["detector"] == "mad"].set_index("name")[["anomaly_len", "train_end", "n"]].loc[hit.index]

# ---------------------------------------------------------------- 1. summary
rows = []
for d in ORDER:
    rows.append({"detector": d,
                 "ucr_hit_rate": ci_text(hit[d]),
                 "ucr_hit_wilson": "%d/%d  (%.3f to %.3f)" % ((int(hit[d].sum()), len(hit)) + wilson(hit[d].sum(), len(hit))),
                 "anomaly_found_rate": "%.3f" % found[d].mean(),
                 "mean_f1": ci_text(f1[d]),
                 "mean_pa_f1": ci_text(paf1[d]),
                 "mean_pak_auc": ci_text(pak[d]),
                 "mean_recall": "%.3f" % rec[d].mean(),
                 "median_false_alarm_events": ci_text(fa[d], np.median),
                 "mean_false_alarm_events": "%.1f" % fa[d].mean()})
summary = pd.DataFrame(rows).set_index("detector")
summary.to_csv(str(OUT) + "_summary.csv")
print("\n=== 1. summary per detector (mean or median, 95% bootstrap CI over series) ===")
print(summary.T.to_string())

# ---------------------------------------------------------------- 2. paired tests
pairs = list(combinations(ORDER, 2))
t = []
for a, b in pairs:
    a_only, b_only, p_hit = mcnemar_exact(hit[a], hit[b])
    t.append({"A": a, "B": b, "A_only_hits": a_only, "B_only_hits": b_only, "p_hit": p_hit,
              "p_pa_f1": wilcoxon_paired(paf1[a], paf1[b]), "p_pak_auc": wilcoxon_paired(pak[a], pak[b])})
tests = pd.DataFrame(t)
for c in ["p_hit", "p_pa_f1", "p_pak_auc"]:
    tests[c + "_holm"] = holm(tests[c])
tests.round(4).to_csv(str(OUT) + "_tests.csv", index=False)
print("\n=== 2. paired tests, all %d pairs (Holm-corrected p over the %d tests of each column) ===" % (len(pairs), len(pairs)))
print(tests[["A", "B", "A_only_hits", "B_only_hits", "p_hit", "p_hit_holm", "p_pa_f1_holm", "p_pak_auc_holm"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- 3. does the ranking change with the metric?
def signs(df):
    cols = []
    for a, b in pairs:
        cols.append(np.sign(np.round(df[a].to_numpy() - df[b].to_numpy(), 12)))
    return np.column_stack(cols)           # series x pairs, values -1 / 0 / +1


S = {"f1": signs(f1), "pa_f1": signs(paf1), "pak_auc": signs(pak), "ucr_hit": signs(hit)}
r = []
for a, b in [("f1", "pa_f1"), ("f1", "pak_auc"), ("pa_f1", "pak_auc"), ("f1", "ucr_hit"), ("pa_f1", "ucr_hit"), ("pak_auc", "ucr_hit")]:
    changed = (S[a] != S[b]).any(axis=1)
    flipped = (S[a] * S[b] == -1).any(axis=1)             # one metric says X > Y, the other says Y > X
    informative = (S[a] != 0).any(axis=1) | (S[b] != 0).any(axis=1)
    r.append({"metric_A": a, "metric_B": b,
              "order_changed_%": round(100 * changed.mean(), 1),
              "strict_flip_%": round(100 * flipped.mean(), 1),
              "series_where_any_metric_separates_detectors_%": round(100 * informative.mean(), 1),
              "flip_%_among_those": round(100 * flipped[informative].mean(), 1) if informative.any() else np.nan})
rank = pd.DataFrame(r)
rank.to_csv(str(OUT) + "_rank_changes.csv", index=False)
print("\n=== 3. share of series where the detector ORDER changes between two metrics ===")
print("(changed = any pair of detectors is ordered differently, ties count; flip = some pair is reversed)")
print(rank.to_string(index=False))

# ---------------------------------------------------------------- 4. Point Adjustment vs anomaly length
print("\n=== 4. Point Adjustment and anomaly length ===")
codes, edges = pd.qcut(meta["anomaly_len"], 3, retbins=True, duplicates="drop", labels=False)
names = ["1_short", "2_mid", "3_long"][:int(codes.max()) + 1]
meta["len_bin"] = codes.map(dict(enumerate(names)))
print("anomaly length bin edges:", [int(e) for e in edges], "(short / mid / long)")
lines = []
for d in ORDER:
    infl = paf1[d] - f1[d]
    rho, p = stats.spearmanr(meta["anomaly_len"], infl) if infl.nunique() > 1 else (np.nan, np.nan)
    flattered = int(((paf1[d] >= 0.5) & (hit[d] == 0)).sum())
    lines.append({"detector": d, "mean_pa_inflation(pa_f1 - f1)": round(infl.mean(), 3),
                  "spearman(anomaly_len, inflation)": round(rho, 3), "p": round(p, 4),
                  "series_with_pa_f1>=0.5_but_ucr_miss": flattered})
print(pd.DataFrame(lines).to_string(index=False))
b = []
for lbl, idx in meta.groupby("len_bin").groups.items():
    for d in ORDER:
        b.append({"anomaly_len_bin": lbl, "n": len(idx), "detector": d, "mean_f1": f1.loc[idx, d].mean(),
                  "mean_pa_f1": paf1.loc[idx, d].mean(), "mean_pak_auc": pak.loc[idx, d].mean(),
                  "ucr_hit_rate": hit.loc[idx, d].mean()})
bins = pd.DataFrame(b).round(3)
bins.to_csv(str(OUT) + "_pa_vs_length.csv", index=False)
print("\nby anomaly length (terciles):")
print(bins.pivot(index="detector", columns="anomaly_len_bin", values="mean_pa_f1").reindex(ORDER).to_string())
print("series per bin:", meta.groupby("len_bin").size().to_dict())
print("(values = mean PA-F1; the table with F1, PA%K-AUC and hit rate is in *_pa_vs_length.csv)")

# ---------------------------------------------------------------- 5. where does each detector work? (exploratory)
print("\n=== 5. UCR hit rate by data family and by train length (exploratory, small groups) ===")
tag = meta.index.to_series().str.extract(r"UCR_Anomaly_(DISTORTED|NOISE|)")[0].fillna("")
meta["family"] = np.where(tag == "DISTORTED", "DISTORTED", np.where(tag == "NOISE", "NOISE", "natural"))
fam = pd.DataFrame({d: hit[d].groupby(meta["family"]).mean() for d in ORDER}).round(3)
fam.insert(0, "n", meta.groupby("family").size())
print(fam.to_string())
tcodes, tedges = pd.qcut(meta["train_end"], 3, retbins=True, duplicates="drop", labels=False)
meta["train_bin"] = tcodes.map(dict(enumerate(["1_short_train", "2_mid_train", "3_long_train"][:int(tcodes.max()) + 1])))
print("train_end bin edges:", [int(e) for e in tedges])
trn = pd.DataFrame({d: hit[d].groupby(meta["train_bin"]).mean() for d in ORDER}).round(3)
trn.insert(0, "n", meta.groupby("train_bin").size())
print("\nby train length (terciles of train_end):")
print(trn.to_string())
fam.to_csv(str(OUT) + "_by_family.csv"); trn.to_csv(str(OUT) + "_by_train_len.csv")

# ---------------------------------------------------------------- 6. runtime
sec = ok.pivot(index="name", columns="detector", values="seconds")[ORDER]
print("\n=== 6. compute time (sum over series, minutes) ===")
print((sec.sum() / 60).round(1).to_string())
print("\nsaved tables:", OUT.name + "_*.csv")
