"""Step 7: do simple tests agree with DESeq2?
Usage: python 07_simple_tests.py <human|mouse> <exercise|acute|training> [paired]      (run step 6 with the same arguments first)
Unpaired: log2(CPM + 1) with each study's mean removed, Welch t-test and Mann-Whitney U test, exercise vs no exercise.
Paired: per person difference (exercise minus no exercise, averaged over that person's samples), paired t-test and Wilcoxon signed-rank test.
Benjamini-Hochberg and Bonferroni; agreement with the DESeq2 result of step 6 as 2x2 contingency tables.
Saves results/07_<species>_<which>[_paired].csv and the matching .txt in the project folder."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, spearmanr, ttest_1samp, ttest_ind, wilcoxon

species, which = sys.argv[1], sys.argv[2]
paired = "paired" in sys.argv[3:]
ROOT = Path(__file__).parent
R = ROOT / "results"
name = f"{species}_{which}" + ("_paired" if paired else "")
report = []


def say(text=""):
    print(text)
    report.append(str(text))


def bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    o = np.argsort(pv)
    r = np.minimum.accumulate((pv[o] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    out = np.empty_like(pv)
    out[o] = np.minimum(r, 1)
    q[ok] = out
    return q


info = pd.read_csv(ROOT / "database2_detailed.csv", usecols=["sample", "species", "study", "person", "condition"], index_col="sample")
info = info[info.species == species]
keep_cond = {"exercise": ["acute exercise", "long-term training"], "acute": ["acute exercise"], "training": ["long-term training"]}[which]
info = info[info.condition.isin(["no exercise"] + keep_cond)].copy()
info["group"] = np.where(info.condition == "no exercise", "none", "exercise")
if paired:
    n = info.groupby("person").group.nunique()
    info = info[info.person.isin(n[n == 2].index)]
n = info.groupby("study").group.nunique()
info = info[info.study.isin(n[n == 2].index)]
if len(info) == 0:
    sys.exit("no samples left for this design (mouse has no paired samples)")

counts = pd.read_csv(R / ("01_counts.tsv.gz" if species == "human" else "01b_counts.tsv.gz"), sep="\t", index_col=0)[info.index]
if species == "human":
    a = pd.read_csv(ROOT / "geo" / "Human.GRCh38.p13.annot.tsv.gz", sep="\t", dtype=str)
    cols = {c.lower(): c for c in a.columns}
    counts.index = counts.index.astype(str).map(a.set_index(cols["geneid"])[cols["symbol"]])
counts = counts[counts.index.notna()]
counts.index = counts.index.astype(str).str.upper()
counts = counts.groupby(level=0).sum()
cpm = counts / counts.sum() * 1e6
cpm = cpm[(cpm > 1).sum(axis=1) >= 0.1 * counts.shape[1]]
lc = np.log2(cpm + 1)
ex = (info.group == "exercise").values

if paired:
    people = info.person.unique()
    D = np.column_stack([lc.loc[:, ((info.person == p) & (info.group == "exercise")).values].mean(axis=1).values -
                         lc.loc[:, ((info.person == p) & (info.group == "none")).values].mean(axis=1).values for p in people])
    effect = D.mean(axis=1)
    t_stat, p_t = ttest_1samp(D, 0, axis=1)
    p_rank = wilcoxon(D, axis=1).pvalue
    names = ("paired t-test", "Wilcoxon signed-rank")
    say(f"{species}, {which} paired: {len(people)} people, {len(lc)} genes. Smallest possible Wilcoxon p with {len(people)} people: {2 / 2 ** len(people):.2e}")
else:
    cen = lc.T.groupby(info.study.values).transform(lambda x: x - x.mean()).T.values
    effect = cen[:, ex].mean(axis=1) - cen[:, ~ex].mean(axis=1)
    t_stat, p_t = ttest_ind(cen[:, ex], cen[:, ~ex], equal_var=False, axis=1)
    p_rank = mannwhitneyu(cen[:, ex], cen[:, ~ex], axis=1).pvalue
    names = ("Welch t-test", "Mann-Whitney U")
    say(f"{species}, {which}: {ex.sum()} exercise vs {(~ex).sum()} no-exercise samples, {len(lc)} genes (study means removed)")

out = pd.DataFrame({"mean_log2_difference": effect, "t_stat": t_stat, "p_t": p_t, "p_rank": p_rank}, index=lc.index)
out["q_t"], out["q_rank"] = bh(out.p_t), bh(out.p_rank)
de = pd.read_csv(R / f"06_{name}.csv", index_col=0)[["log2FoldChange", "padj"]]
out = out.join(de)
out.sort_values("p_t").to_csv(R / f"07_{name}.csv")

for label, q, p in [(names[0], "q_t", "p_t"), (names[1], "q_rank", "p_rank")]:
    say(f"{label}: q < 0.05 {(out[q] < 0.05).sum()}   Bonferroni < 0.05 {(out[p] * out[p].notna().sum() < 0.05).sum()}")
say(f"DESeq2: padj < 0.05 {(out.padj < 0.05).sum()}")
say(f"Spearman correlation, DESeq2 log2FC vs mean log2 difference: {spearmanr(out.log2FoldChange, out.mean_log2_difference, nan_policy='omit')[0]:.3f}")
for label, q in [(names[0], "q_t"), (names[1], "q_rank")]:
    t = pd.crosstab(out.padj < 0.05, (out[q] < 0.05)).reindex(index=[False, True], columns=[False, True], fill_value=0)
    t.index.name, t.columns.name = "DESeq2 significant", f"{label} significant"
    c2, p, _, _ = chi2_contingency(t, correction=False)
    say(f"\nagreement with DESeq2, {label}:\n{t.to_string()}\nchi-square = {c2:.1f}, p = {p:.2e}; Fisher odds ratio = {fisher_exact(t)[0]:.1f}")
(ROOT / f"07_{name}.txt").write_text("\n".join(report), encoding="utf-8")
