"""Step 5: structure of the samples, from database1_overview.csv.
Chi-square tests of independence (contingency tables) between exercise condition and study, sex and age group, per species:
a significant result means the groups are unbalanced across that factor, which is why models must include study.
Also a joint PCA of human and mouse samples. Saves results/05_pca.png and 05_structure.txt."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent
R = ROOT / "results"
R.mkdir(exist_ok=True)
report = []


def say(text=""):
    print(text)
    report.append(str(text))


d = pd.read_csv(ROOT / "database1_overview.csv", index_col=0, low_memory=False)
genes = d.columns[6:]

for species, g in d.groupby("species"):
    say(f"\n{species}: {len(g)} samples")
    for factor in ["study", "sex", "age_group"]:
        x = g[g.sex != "unknown"] if factor == "sex" else g
        t = pd.crosstab(x[factor], x.condition)
        if min(t.shape) < 2:
            continue
        chi2, p, dof, exp = chi2_contingency(t)
        v = np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1)))
        say(f"  condition x {factor}: chi-square = {chi2:.1f}, df = {dof}, p = {p:.2e}, Cramer's V = {v:.2f}, cells with expected count < 5: {(exp < 5).mean():.0%}")
        if factor != "study":
            say(t.to_string())

x = d[genes]
top = x.loc[:, x.var().nlargest(5000).index]
pca = PCA(n_components=10, random_state=0).fit(top - top.mean())
Z = pca.transform(top - top.mean())
say("\njoint PCA, variance in PC1 to PC3: " + ", ".join(f"{v * 100:.1f}%" for v in pca.explained_variance_ratio_[:3]))

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for sp, c in [("human", "tab:blue"), ("mouse", "tab:red")]:
    m = (d.species == sp).values
    ax[0].scatter(Z[m, 0], Z[m, 1], s=6, c=c, label=sp)
ax[0].set_title("by species")
for cond, c in [("no exercise", "tab:gray"), ("acute exercise", "tab:orange"), ("long-term training", "tab:green")]:
    for sp, mk in [("human", "o"), ("mouse", "^")]:
        m = ((d.condition == cond) & (d.species == sp)).values
        ax[1].scatter(Z[m, 0], Z[m, 1], s=8, c=c, marker=mk, label=f"{cond} ({sp})")
ax[1].set_title("by condition")
for a in ax:
    a.set_xlabel("PC1")
    a.set_ylabel("PC2")
    a.legend(fontsize=7, markerscale=2)
plt.tight_layout()
plt.savefig(R / "05_pca.png", dpi=130)
(ROOT / "05_structure.txt").write_text("\n".join(report), encoding="utf-8")
