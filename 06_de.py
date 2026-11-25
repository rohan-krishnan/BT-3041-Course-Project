"""Step 6: differential expression, exercise vs no exercise, with two methods.
Usage: python 06_de.py <human|mouse> <exercise|acute|training> [paired]
A) DESeq2 (pydeseq2) on raw counts, design ~ study + group, or ~ person + group with 'paired' (people with both groups only).
B) Chi-square contingency tables. Each gene is called high or low relative to a median, giving a 2x2 table
   (high/low x exercise/no exercise) per gene:
   naive chi-square = one table from all samples with the overall median (ignores study);
   Cochran-Mantel-Haenszel (CMH) chi-square = one table per study using that study's own median, combined across studies.
Needs database2_detailed.csv and results/01_counts.tsv.gz (human) or results/01b_counts.tsv.gz (mouse).
Saves results/06_<species>_<which>[_paired].csv and the matching .txt in the project folder."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from scipy.stats import chi2, chi2_contingency, fisher_exact

species, which = sys.argv[1], sys.argv[2]
paired = "paired" in sys.argv[3:]
ROOT = Path(__file__).parent
R = ROOT / "results"
name = f"06_{species}_{which}" + ("_paired" if paired else "")
report = []


def say(text=""):
    print(text)
    report.append(str(text))


def bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    r = np.minimum.accumulate((pv[order] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(r, 1)
    q[ok] = out
    return q


# samples
info = pd.read_csv(ROOT / "database2_detailed.csv", usecols=["sample", "species", "study", "person", "condition"], index_col="sample")
info = info[info.species == species]
keep_cond = {"exercise": ["acute exercise", "long-term training"], "acute": ["acute exercise"], "training": ["long-term training"]}[which]
info = info[info.condition.isin(["no exercise"] + keep_cond)].copy()
info["group"] = np.where(info.condition == "no exercise", "none", "exercise")
if paired:
    n = info.groupby("person").group.nunique()
    info = info[info.person.isin(n[n == 2].index)]
n = info.groupby("study").group.nunique()
info = info[info.study.isin(n[n == 2].index)]  # a study needs both groups
if len(info) == 0:
    sys.exit("no samples left for this design (mouse has no paired samples)")
say(f"{species}, {which}{' paired' if paired else ''}: {len(info)} samples ({(info.group == 'exercise').sum()} exercise, {(info.group == 'none').sum()} no exercise) "
    f"from {info.study.nunique()} studies" + (f", {info.person.nunique()} people" if paired else ""))

# counts, gene symbols in upper case, genes with CPM > 1 in at least 10% of samples
counts = pd.read_csv(R / ("01_counts.tsv.gz" if species == "human" else "01b_counts.tsv.gz"), sep="\t", index_col=0)[info.index]
if species == "human":
    a = pd.read_csv(ROOT / "geo" / "Human.GRCh38.p13.annot.tsv.gz", sep="\t", dtype=str)
    cols = {c.lower(): c for c in a.columns}
    counts.index = counts.index.astype(str).map(a.set_index(cols["geneid"])[cols["symbol"]])
counts = counts[counts.index.notna()]
counts.index = counts.index.astype(str).str.upper()
counts = counts.groupby(level=0).sum()
cpm = counts / counts.sum() * 1e6
keep = (cpm > 1).sum(axis=1) >= 0.1 * counts.shape[1]
counts, cpm = counts[keep], cpm[keep]
say(f"genes tested: {len(counts)}")

# A) DESeq2
meta = info[["study", "person", "group"]].astype(str)
dds = DeseqDataSet(counts=counts.T.astype(int), metadata=meta, design="~ person + group" if paired else "~ study + group", quiet=True)
dds.deseq2()
ds = DeseqStats(dds, contrast=["group", "exercise", "none"], quiet=True)
ds.summary()
res = ds.results_df.copy()
res["p_bonferroni"] = np.minimum(res.pvalue * res.pvalue.notna().sum(), 1)

# B) chi-square contingency tables
lc = np.log2(cpm.values + 1)
ex = (info.group == "exercise").values
N, n1 = len(ex), ex.sum()
high = lc > np.median(lc, axis=1, keepdims=True)
a = high[:, ex].sum(1)
m1 = high.sum(1)
with np.errstate(divide="ignore", invalid="ignore"):
    chi_naive = N * (a * (N - n1 - m1 + a) - (n1 - a) * (m1 - a)) ** 2 / (n1 * (N - n1) * m1 * (N - m1))
S_a = S_E = S_V = S_ad = S_bc = 0
for s in info.study.unique():
    m = (info.study == s).values
    e, x = ex[m], lc[:, m]
    n_s, n1_s = m.sum(), e.sum()
    h = x > np.median(x, axis=1, keepdims=True)
    a_s, m1_s = h[:, e].sum(1), h.sum(1)
    S_a = S_a + a_s
    S_E = S_E + n1_s * m1_s / n_s
    S_V = S_V + n1_s * (n_s - n1_s) * m1_s * (n_s - m1_s) / (n_s**2 * (n_s - 1))
    S_ad = S_ad + a_s * (n_s - n1_s - m1_s + a_s) / n_s
    S_bc = S_bc + (n1_s - a_s) * (m1_s - a_s) / n_s
with np.errstate(divide="ignore", invalid="ignore"):
    cmh = np.maximum(np.abs(S_a - S_E) - 0.5, 0) ** 2 / S_V
    mh_or = S_ad / S_bc
cm = pd.DataFrame({"chi2_naive": chi_naive, "p_naive": chi2.sf(chi_naive, 1), "cmh": cmh, "p_cmh": chi2.sf(cmh, 1),
                   "mh_odds_ratio": mh_or, "cmh_direction": np.where(S_a > S_E, "up", "down")}, index=counts.index)
cm["q_naive"], cm["q_cmh"] = bh(cm.p_naive), bh(cm.p_cmh)
out = res.join(cm)
out.sort_values("pvalue").to_csv(R / f"{name}.csv")

# summary
de, cs, nv = out.padj < 0.05, out.q_cmh < 0.05, out.q_naive < 0.05
say(f"\nDESeq2: padj < 0.05 {de.sum()} (up {(de & (out.log2FoldChange > 0)).sum()}, down {(de & (out.log2FoldChange < 0)).sum()}); "
    f"and |log2FC| > 1: {(de & (out.log2FoldChange.abs() > 1)).sum()}; Bonferroni < 0.05: {(out.p_bonferroni < 0.05).sum()}")
say(f"Chi-square, CMH by study: q < 0.05 {cs.sum()}   naive chi-square (study ignored): q < 0.05 {nv.sum()}")
say(f"genes significant by the naive chi-square but not by CMH: {(nv & ~cs).sum()} (study confounding)")
t = pd.crosstab(de.rename("DESeq2 significant"), cs.rename("CMH significant"))
say("\nagreement between DESeq2 and CMH:\n" + t.to_string())
if t.shape == (2, 2):
    c2, p, _, _ = chi2_contingency(t, correction=False)
    say(f"chi-square on this table = {c2:.1f}, p = {p:.2e}; Fisher odds ratio = {fisher_exact(t)[0]:.1f}")
say("\ntop genes by DESeq2 p-value:\n" + out.sort_values("pvalue").head(12)[["baseMean", "log2FoldChange", "padj", "p_cmh", "q_cmh", "mh_odds_ratio"]].round(4).to_string())
(ROOT / f"{name}.txt").write_text("\n".join(report), encoding="utf-8")
say(f"\nsaved results/{name}.csv and {name}.txt")
