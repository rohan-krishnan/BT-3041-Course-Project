"""Step 9: is the exercise response shared between human and mouse?
Usage: python 09_species.py <human result> <mouse result> [min |log2FC|]      e.g. python 09_species.py 06_human_training 06_mouse_training 0
Compare the same contrast in both species (training with training, acute with acute). Default |log2FC| cutoff is 0, so status depends on padj < 0.05 and the sign.
Genes are matched by symbol. Reports the correlation of log2 fold changes, a 3x3 contingency table of
significance status (up / down / not significant) in human against mouse with a chi-square test,
Fisher and chi-square tests on the overlap of the up and down lists, and a sign test on the genes significant in both.
Saves results/09_conserved_genes.csv and 09_species.txt."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, fisher_exact, spearmanr

PADJ = 0.05
LFC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
ROOT = Path(__file__).parent
R = ROOT / "results"
report = []


def say(text=""):
    print(text)
    report.append(str(text))


h = pd.read_csv(R / f"{sys.argv[1]}.csv", index_col=0).dropna(subset=["padj"])
m = pd.read_csv(R / f"{sys.argv[2]}.csv", index_col=0).dropna(subset=["padj"])
d = h[["log2FoldChange", "padj"]].merge(m[["log2FoldChange", "padj"]], left_index=True, right_index=True, suffixes=("_human", "_mouse"))
say(f"genes tested in both species: {len(d)} (human {len(h)}, mouse {len(m)})")
rho, p = spearmanr(d.log2FoldChange_human, d.log2FoldChange_mouse)
say(f"Spearman correlation of log2 fold changes: rho = {rho:.3f}, p = {p:.2e}")


def status(lfc, padj):
    return np.where(padj >= PADJ, "not significant", np.where(lfc > LFC, "up", np.where(lfc < -LFC, "down", "not significant")))


d["human"] = status(d.log2FoldChange_human, d.padj_human)
d["mouse"] = status(d.log2FoldChange_mouse, d.padj_mouse)
order = ["up", "down", "not significant"]
t = pd.crosstab(d.human, d.mouse).reindex(index=order, columns=order, fill_value=0)
say(f"\nsignificance status (padj < {PADJ}, |log2FC| > {LFC}), rows human, columns mouse:\n" + t.to_string())
chi, p, dof, exp = chi2_contingency(t)
say(f"chi-square test of independence: chi2 = {chi:.1f}, df = {dof}, p = {p:.2e}")

for dirn in ["up", "down"]:
    a = ((d.human == dirn) & (d.mouse == dirn)).sum()
    b, c = (d.human == dirn).sum() - a, (d.mouse == dirn).sum() - a
    table = [[a, b], [c, len(d) - a - b - c]]
    c2, pc = chi2_contingency(table, correction=False)[:2]
    say(f"{dirn} in both: {a} genes (human {a + b}, mouse {a + c}); Fisher odds ratio {fisher_exact(table, alternative='greater')[0]:.1f}, "
        f"p = {fisher_exact(table, alternative='greater')[1]:.2e}; chi-square p = {pc:.2e}")

both = d[(d.human != "not significant") & (d.mouse != "not significant")]
same = (both.human == both.mouse).sum()
if len(both):
    say(f"\ngenes significant in both: {len(both)}, same direction: {same} ({same / len(both):.0%}); sign test p = {binomtest(same, len(both), 0.5).pvalue:.2e}")
cons = both[both.human == both.mouse].assign(direction=lambda x: x.human)
cons = cons.drop(columns=["human", "mouse"]).assign(rank_sum=lambda x: x.padj_human.rank() + x.padj_mouse.rank()).sort_values("rank_sum")
cons.to_csv(R / "09_conserved_genes.csv")
say("\ntop conserved genes:\n" + cons.head(15).round(4).to_string())
(ROOT / "09_species.txt").write_text("\n".join(report), encoding="utf-8")
