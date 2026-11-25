"""Step 8: pathway over-representation with 2x2 contingency tables (chi-square test and Fisher's exact test).
Usage: python 08_enrich.py <result name> [min |log2FC|]      e.g. python 08_enrich.py 06_human_exercise 1
Significant genes = DESeq2 padj < 0.05 and |log2FC| above the cutoff (default 1), split into up and down.
For every pathway: table [in pathway / not] x [significant / not], background = all tested genes. Benjamini-Hochberg across pathways.
Needs internet the first time (Hallmark and Reactome gene sets come from Enrichr through gseapy).
Saves results/08_<result name>_pathways.csv and 08_<result name>.txt."""
import sys
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

stem = sys.argv[1]
LFC = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
LIBS = ["MSigDB_Hallmark_2020", "Reactome_2022"]
ROOT = Path(__file__).parent
R = ROOT / "results"
report = []


def say(text=""):
    print(text)
    report.append(str(text))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    r = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty_like(p)
    q[o] = np.minimum(r, 1)
    return q


res = pd.read_csv(R / f"{stem}.csv", index_col=0).dropna(subset=["padj"])
bg = set(res.index)
lists = {"up": set(res[(res.padj < 0.05) & (res.log2FoldChange > LFC)].index), "down": set(res[(res.padj < 0.05) & (res.log2FoldChange < -LFC)].index)}
say(f"{stem}: {len(bg)} genes tested; significant with |log2FC| > {LFC:g}: up {len(lists['up'])}, down {len(lists['down'])}")

rows = []
for lib_name in LIBS:
    try:
        lib = gp.get_library(lib_name, organism="human")
    except Exception as e:
        say(f"could not load {lib_name}: {e}")
        names = [n for n in gp.get_library_name() if lib_name.split("_")[0].lower() in n.lower()]
        say(f"available libraries with a similar name: {names[:15]}")
        continue
    sets = {t: {g.upper() for g in genes} & bg for t, genes in lib.items()}
    for direction, sig in lists.items():
        if len(sig) < 5:
            continue
        out = []
        for term, s in sets.items():
            if len(s) < 5:
                continue
            a = len(s & sig)
            table = [[a, len(sig) - a], [len(s) - a, len(bg) - len(sig) - len(s) + a]]
            chi, p_chi, _, exp = chi2_contingency(table, correction=False)
            out.append((lib_name, term, direction, len(s), a, (a / len(sig)) / (len(s) / len(bg)), chi, p_chi, fisher_exact(table, alternative="greater")[1],
                        exp.min(), ", ".join(sorted(s & sig)[:8])))
        df = pd.DataFrame(out, columns=["library", "pathway", "direction", "genes_in_pathway", "hits", "fold_enrichment", "chi2", "p_chi2",
                                        "p_fisher", "min_expected_count", "example_hits"])
        df["q_chi2"], df["q_fisher"] = bh(df.p_chi2), bh(df.p_fisher)
        rows.append(df)
        top = df[df.fold_enrichment > 1].sort_values("q_fisher").head(8)
        say(f"\n{lib_name}, {direction}-regulated genes, top pathways (Fisher q; chi-square q):")
        say(top.assign(q_fisher=top.q_fisher.map("{:.1e}".format), q_chi2=top.q_chi2.map("{:.1e}".format))[["pathway", "hits", "genes_in_pathway", "fold_enrichment", "q_fisher", "q_chi2"]].round(2).to_string(index=False))
if rows:
    pd.concat(rows).to_csv(R / f"08_{stem}_pathways.csv", index=False)
(ROOT / f"08_{stem}.txt").write_text("\n".join(report), encoding="utf-8")
