"""Step 2: quality control and a first look at a merged exercise dataset.
Usage: python 02_prepare.py          (human: needs results/01_counts.tsv.gz and 01_samples.tsv)
       python 02_prepare.py 01b      (mouse: needs results/01b_counts.tsv.gz and 01b_samples.tsv)
Prints library sizes, time-point parsing, the paired design per study, how much of the variation study / exercise / library
type explain, and saves results/02_pca.png, results/02_logcpm.tsv.gz, results/02_samples.tsv and 02_prepare.txt."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent
R = ROOT / "results"
TAG = sys.argv[1] if len(sys.argv) > 1 else "01"
OUT = "02" if TAG == "01" else "02b"  # mouse outputs get a b so human files are not overwritten
report = []


def say(text=""):
    print(text)
    report.append(str(text))


counts = pd.read_csv(R / f"{TAG}_counts.tsv.gz", sep="\t", index_col=0)
counts = counts.groupby(level=0).sum()  # ARCHS4 lists some mouse symbols on several rows: add them up
info = pd.read_csv(R / f"{TAG}_samples.tsv", sep="\t", index_col=0, dtype=str, keep_default_na=False).loc[counts.columns]
say(f"counts: {counts.shape[0]} genes x {counts.shape[1]} samples, {info.datasets.nunique()} studies")

# library size
lib = counts.sum()
info["lib_size_M"] = (lib / 1e6).round(2)
say("\nlibrary size (million counts) per study, median [min, max]:")
say(info.assign(l=info.lib_size_M).groupby("datasets").l.agg(["median", "min", "max"]).round(1).to_string())
low = info.index[info.lib_size_M < 1]
say(f"samples with fewer than 1 million counts: {len(low)} {list(low)[:10]}")

# time after exercise, parsed from the biopsy time point label
def hours(x):
    x = x.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*h", x)
    if m:
        return (float(m[1]) + float(m[2])) / 2
    m = re.search(r"(\d+(?:\.\d+)?)\s*h", x)
    if m:
        return float(m[1])
    m = re.search(r"(?:day\s*(\d+)|(\d+)\s*d\b)", x)
    if m:
        return 24.0 * float(m[1] or m[2])
    return np.nan

info["hours_after"] = [hours(t) if c == "acute exercise" else np.nan for t, c in zip(info["biopsy timepoint"], info.classification)]
tp = info[info.classification == "acute exercise"].groupby("biopsy timepoint", dropna=False).agg(n=("GSM" if "GSM" in info else "datasets", "size"), hours=("hours_after", "first"))
say("\nacute-exercise time points and how they were parsed (check these):\n" + tp.to_string())
say(f"acute samples with no parsed time: {int(info[info.classification == 'acute exercise'].hours_after.isna().sum())}")

# paired design: people with a baseline sample and an exercise sample
info["subject"] = info.datasets + "|" + info["subject id(or sample id)"]
g = info.groupby("datasets")
rows = []
for gse, d in g:
    s = d.groupby("subject").classification.agg(set)
    rows.append((gse, len(d), len(s), sum({"inactivity", "acute exercise"} <= v for v in s), sum({"inactivity", "longterm training"} <= v for v in s)))
design = pd.DataFrame(rows, columns=["study", "samples", "people", "people_base+acute", "people_base+training"])
say("\npaired design per study:\n" + design.to_string(index=False))
say(f"people with baseline and acute: {design['people_base+acute'].sum()}, baseline and training: {design['people_base+training'].sum()}")

# filter genes and log-CPM
cpm = counts / counts.sum() * 1e6
keep = (cpm > 1).sum(axis=1) >= 0.1 * counts.shape[1]
logcpm = np.log2(cpm[keep] + 1)
say(f"\ngenes kept (CPM > 1 in at least 10% of samples): {keep.sum()} of {len(keep)}")


def pca(mat, n=20):
    top = mat.loc[mat.var(axis=1).nlargest(5000).index]
    p = PCA(n_components=n, random_state=0).fit(top.T - top.T.mean())
    return p.transform(top.T - top.T.mean()), p.explained_variance_ratio_


def adj_r2(Z, w, labels):
    y = pd.Series(labels).astype(str).values
    d = pd.get_dummies(y, drop_first=True).values.astype(float)
    n, k = d.shape
    X = np.c_[np.ones(n), d]
    res = Z - X @ np.linalg.lstsq(X, Z, rcond=None)[0]
    r2 = 1 - (res**2).sum(0) / ((Z - Z.mean(0)) ** 2).sum(0)
    return float((((1 - (1 - r2) * (n - 1) / (n - k - 1)) * w).sum()) / w.sum())


Z, w = pca(logcpm)
say("variance in the first 5 PCs: " + ", ".join(f"{x * 100:.1f}%" for x in w[:5]))
say("\nadjusted R2 (weighted over 20 PCs), all samples:")
for name, col in [("study", "datasets"), ("exercise class", "classification"), ("RNA type", "extracted molecule"), ("read layout", "layout")]:
    say(f"  {name:15s} {adj_r2(Z, w, info[col].values):.3f}  ({info[col].nunique()} levels)")

# remove each study's mean, then ask how much exercise class explains
centred = logcpm.T.groupby(info.datasets.values).transform(lambda x: x - x.mean()).T
Zc, wc = pca(centred)
say(f"after removing each study's mean: exercise class adjusted R2 = {adj_r2(Zc, wc, info.classification.values):.3f}")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, col, title, data in [(ax[0], "datasets", "all samples, by study", Z), (ax[1], "classification", "all samples, by exercise class", Z),
                            (ax[2], "classification", "study mean removed, by exercise class", Zc)]:
    for c in sorted(info[col].unique()):
        m = (info[col] == c).values
        a.scatter(data[m, 0], data[m, 1], s=6, label=c)
    a.set_title(title)
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")
    a.legend(fontsize=6, markerscale=2, ncol=2 if col == "datasets" else 1)
plt.tight_layout()
plt.savefig(R / f"{OUT}_pca.png", dpi=130)

info.drop(columns=["subject"]).to_csv(R / f"{OUT}_samples.tsv", sep="\t")
logcpm.round(3).astype("float32").to_csv(R / f"{OUT}_logcpm.tsv.gz", sep="\t")
(ROOT / f"{OUT}_prepare.txt").write_text("\n".join(report), encoding="utf-8")
print(f"\nsaved {OUT}_prepare.txt and the {OUT}_ files in results")