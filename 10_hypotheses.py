"""Step 10: test gene hypotheses taken from papers against our data.
Usage: python 10_hypotheses.py <human|mouse> [paired]
Reads hypotheses.csv (gene, direction up/down, contrast acute/training/exercise, source; add your own rows) and the step 6 results
results/06_<species>_<contrast>[_paired].csv. A hypothesis is confirmed when DESeq2 padj < 0.05 and the direction matches.
Because many genes are significant in large datasets, a random-gene null is also run: how often do the same number of random
genes with the same directions reach as many confirmations? Saves results/10_<species>_hypotheses.csv and 10_<species>.txt."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

species = sys.argv[1]
suffix = "_paired" if "paired" in sys.argv[2:] else ""
ROOT = Path(__file__).parent
R = ROOT / "results"
report = []


def say(text=""):
    print(text)
    report.append(str(text))


hyp = pd.read_csv(ROOT / "hypotheses.csv")
hyp["gene"] = hyp.gene.str.upper()
rows, tested = [], {}
for _, r in hyp.iterrows():
    f = R / f"06_{species}_{r.contrast}{suffix}.csv"
    if not f.exists():
        rows.append({**r, "verdict": f"run first: python 06_de.py {species} {r.contrast}{' paired' if suffix else ''}"})
        continue
    res = tested.setdefault(r.contrast, pd.read_csv(f, index_col=0).dropna(subset=["padj"]))
    if r.gene not in res.index:
        rows.append({**r, "verdict": "gene not tested"})
        continue
    x = res.loc[r.gene]
    obs = "up" if x.log2FoldChange > 0 else "down"
    verdict = "not significant" if x.padj >= 0.05 else ("confirmed" if obs == r.direction else "opposite direction")
    rows.append({**r, "log2FoldChange": round(x.log2FoldChange, 3), "padj": x.padj, "verdict": verdict})
out = pd.DataFrame(rows)
out.to_csv(R / f"10_{species}_hypotheses.csv", index=False)
say(out[["gene", "contrast", "direction", "log2FoldChange", "padj", "verdict"]].to_string(index=False))

ok = out[out.verdict.isin(["confirmed", "not significant", "opposite direction"])]
if len(ok) == 0:
    sys.exit("no testable hypotheses yet, run the step 6 commands listed above")
k = (ok.verdict == "confirmed").sum()
say(f"\n{k} of {len(ok)} testable hypotheses confirmed ({(out.verdict == 'opposite direction').sum()} opposite direction)")

rng = np.random.default_rng(0)
sims = []
for _ in range(10000):
    hits = 0
    for _, r in ok.iterrows():
        res = tested[r.contrast]
        g = res.iloc[rng.integers(len(res))]
        hits += (g.padj < 0.05) and (("up" if g.log2FoldChange > 0 else "down") == r.direction)
    sims.append(hits)
sims = np.array(sims)
say(f"random genes with the same directions: on average {sims.mean():.1f} confirmed, p(at least {k}) = {(sims >= k).mean():.4f}")
(ROOT / f"10_{species}.txt").write_text("\n".join(report), encoding="utf-8")
