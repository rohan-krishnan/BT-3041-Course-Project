"""Step 13: figures.  Usage: python 13_figures.py [figure numbers]      e.g. python 13_figures.py 3 5      (no number = all)
Reads database2_detailed.csv and the result files of steps 6 to 12. A figure whose input files are missing is skipped with a message.
Writes results/13_fig<number>_<name>.png at 300 dpi.
  1 data overview    2 structure and study effect    3 differential expression and methods    4 pathways
  5 literature hypotheses    6 human against mouse    7 extra questions (step 11)    8 reverse engineering (step 12)"""
import re
import sys
import textwrap
import traceback
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter, ScalarFormatter

ROOT = Path(__file__).parent
R = ROOT / "results"
INK, INK2, GRID, SURF = "#2b2f36", "#7a808a", "#eceef1", "#ffffff"
BASE, ACUTE, TRAIN = "#aeb4be", "#f0906a", "#7480dc"  # no exercise, acute exercise, long-term training
BLUE, AQUA, MAGENTA, RED, NS = "#5b8def", "#4cc2a3", "#e68fb5", "#e26a68", "#dfe1e6"
LBLUE, DBLUE = "#a9c4f7", "#3a5fb0"
COND = {"no exercise": BASE, "acute exercise": ACUTE, "long-term training": TRAIN}
DIV = LinearSegmentedColormap.from_list("div", ["#4a74c0", "#f8f8f6", "#e07a52"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#d9e6fb", "#6f98e8", "#34509a"])
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                     "axes.edgecolor": "#d5d8de", "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 3, "ytick.major.size": 0,
                     "xtick.major.width": 0.8, "xtick.major.pad": 4, "ytick.major.pad": 4, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.titlelocation": "left", "axes.titleweight": "normal", "axes.titlesize": 11.5, "axes.titlepad": 12,
                     "axes.labelsize": 9.5, "axes.labelweight": "normal", "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
                     "legend.fontsize": 8.5, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True, "font.size": 9.5, "figure.dpi": 100})
_DB = {}


def find(name):
    return next((p for p in (R / name, ROOT / name) if p.exists()), None)


def need(*names):
    miss = [n for n in names if find(n) is None]
    if miss:
        print("  skipped, missing:", ", ".join(miss))
    return not miss


def read(name, **kw):
    return pd.read_csv(find(name), **kw)


def load():
    if not _DB:
        db = pd.read_csv(ROOT / "database2_detailed.csv", index_col=0, low_memory=False)
        last = list(db.columns).index("library_size_millions") + 1
        meta, X = db.iloc[:, :last].copy(), db.iloc[:, last:]
        meta["hours"] = pd.to_numeric(meta.hours_after_exercise, errors="coerce")
        _DB.update(meta=meta, X=X)
    return _DB["meta"], _DB["X"]


def title(ax, letter, text, sub=None):
    ax.set_title(f"{letter}   {text}", fontsize=11.5, color=INK, pad=12)


def save(fig, name):
    fig.savefig(R / f"13_{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  saved", f"13_{name}.png")


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


def label_genes(ax, df, x, y, n=6):
    for g, r in df.head(n).iterrows():
        ax.annotate(g, (r[x], r[y]), xytext=(4, 3), textcoords="offset points", fontsize=8, color=INK, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])


# 1 data overview ----------------------------------------------------------------------------------------------------
def fig1():
    meta, _ = load()
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw={"width_ratios": [1.2, 1]})
    for a, sp, letter in [(ax[0, 0], "human", "a"), (ax[1, 0], "mouse", "b")]:
        ct = meta[meta.species == sp].groupby(["study", "condition"]).size().unstack(fill_value=0).reindex(columns=list(COND), fill_value=0)
        ct = ct.loc[ct.sum(axis=1).sort_values().index]
        left = np.zeros(len(ct))
        for c, col in COND.items():
            a.barh(ct.index, ct[c], left=left, color=col, label=c, height=0.72, edgecolor=SURF, linewidth=1.5)
            left += ct[c].values
        for i, tot in enumerate(left):
            a.text(tot + 1, i, int(tot), va="center", fontsize=8, color=INK2)
        title(a, letter, f"{sp.capitalize()} samples per study", f"{int(left.sum())} samples from {len(ct)} studies")
        a.set_xlabel("samples")
        a.grid(axis="x")
    ax[0, 0].legend(loc="lower right")
    h = meta[(meta.species == "human") & (meta.condition == "acute exercise")]
    cnt = h.hours.value_counts().sort_index()
    ax[0, 1].bar(range(len(cnt)), cnt.values, color=ACUTE, width=0.7)
    for i, v in enumerate(cnt.values):
        ax[0, 1].text(i, v + 1, v, ha="center", fontsize=9, color=INK2)
    ax[0, 1].set_xticks(range(len(cnt)), [f"{int(i)} h" for i in cnt.index])
    ax[0, 1].set(ylabel="samples")
    ax[0, 1].grid(axis="y")
    title(ax[0, 1], "c", "Biopsy time after one exercise bout", "human acute samples")
    hm = meta[meta.species == "human"].groupby("person").condition.agg(frozenset)
    lab = {frozenset({"no exercise"}): "baseline only", frozenset({"no exercise", "acute exercise"}): "baseline and acute",
           frozenset({"no exercise", "long-term training"}): "baseline and training", frozenset({"long-term training"}): "training, no baseline",
           frozenset({"acute exercise"}): "acute, no baseline"}
    n = hm.map(lab).fillna("other").value_counts().sort_values()
    ax[1, 1].barh(n.index, n.values, color=BLUE, height=0.7)
    for i, v in enumerate(n.values):
        ax[1, 1].text(v + 1, i, v, va="center", fontsize=9, color=INK2)
    ax[1, 1].set(xlabel="people")
    ax[1, 1].grid(axis="x")
    title(ax[1, 1], "d", "Paired samples in the human data", "each person with the samples they contribute")
    fig.tight_layout()
    save(fig, "fig1_data")


# 2 structure --------------------------------------------------------------------------------------------------------
def pca(M, k=2000):
    M = M.loc[:, M.var().nlargest(k).index]
    U, S, _ = np.linalg.svd((M - M.mean()).values, full_matrices=False)
    return U * S, S**2 / (S**2).sum()


def adj_r2(v, g):
    g = np.asarray(g)
    mean = pd.Series(v).groupby(g).transform("mean").values
    r2 = 1 - ((v - mean) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    n, k = len(v), len(np.unique(g))
    return max(0.0, 1 - (1 - r2) * (n - 1) / (n - k))


def fig2():
    meta, X = load()
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for r, sp in enumerate(["human", "mouse"]):
        m = meta[meta.species == sp]
        Xs = X.loc[m.index]
        none = (m.condition == "no exercise").values
        rel = Xs - Xs[none].groupby(m.study[none]).mean().reindex(m.study).values
        for c, (M, name) in enumerate([(Xs, "raw log2 CPM"), (rel, "study baseline removed")]):
            P, ve = pca(M)
            a = ax[r, c]
            for cond, col in COND.items():
                k = (m.condition == cond).values
                a.scatter(P[k, 0], P[k, 1], s=14, color=col, alpha=0.75, lw=0, label=cond)
            a.set(xlabel=f"PC1 ({ve[0]:.0%})", ylabel=f"PC2 ({ve[1]:.0%})")
            title(a, "abcdef"[r * 3 + c], f"{sp.capitalize()}, {name}", f"{len(m)} samples")
            if c == 0:
                P0, ve0 = P, ve
        npc = 5
        rs = pd.DataFrame({"study": [adj_r2(P0[:, i], m.study.values) for i in range(npc)], "condition": [adj_r2(P0[:, i], m.condition.values) for i in range(npc)]})
        a = ax[r, 2]
        x = np.arange(npc)
        a.bar(x - 0.2, rs.study, 0.38, color=INK2, label="study")
        a.bar(x + 0.2, rs.condition, 0.38, color=AQUA, label="condition")
        a.set_xticks(x, [f"PC{i + 1}" for i in x])
        a.set(ylabel="adjusted R2", ylim=(0, 1))
        a.grid(axis="y")
        title(a, "abcdef"[r * 3 + 2], f"{sp.capitalize()}, variance explained", "raw PCs by study and by condition")
    ax[0, 0].legend(loc="best", markerscale=1.6)
    ax[0, 2].legend()
    fig.tight_layout()
    save(fig, "fig2_structure")


# 3 differential expression and methods ------------------------------------------------------------------------------
CON = [("human acute", "human_acute"), ("human training", "human_training"), ("mouse training", "mouse_training")]


def volcano(a, de, name, thr=0.5):
    de = de.dropna(subset=["padj"])
    x, y, sig = de.log2FoldChange, -np.log10(np.maximum(de.padj, 1e-300)), de.padj < 0.05
    up, dn = sig & (x > thr), sig & (x < -thr)
    mid = sig & ~up & ~dn
    a.scatter(x[~sig], y[~sig], s=4, c=NS, lw=0, rasterized=True)
    a.scatter(x[mid], y[mid], s=4, c="#c3c6cd", lw=0, rasterized=True)
    a.scatter(x[up], y[up], s=8, c=RED, lw=0, rasterized=True)
    a.scatter(x[dn], y[dn], s=8, c=BLUE, lw=0, rasterized=True)
    for v in (-thr, thr):
        a.axvline(v, color=INK2, lw=0.8, ls="--")
    a.axhline(-np.log10(0.05), color=INK2, lw=0.8, ls="--")
    a.set(xlabel="log2 fold change (exercise vs none)", ylabel="-log10 adjusted p")
    top = de[~de.index.astype(str).str.match(r"^(LOC|MIR|LINC)")].assign(a=lambda d: d.stat.abs()).sort_values("a", ascending=False)
    label_genes(a, top.assign(y=y.reindex(top.index)), "log2FoldChange", "y", 5)
    return up.sum(), dn.sum(), sig.sum(), len(de)


def fig3():
    if not need(*[f"06_{s}.csv" for _, s in CON]):
        return
    de = {n: read(f"06_{s}.csv", index_col=0) for n, s in CON}
    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    for i, (n, s) in enumerate(CON):
        u, d, t, tot = volcano(ax[0, i], de[n], n)
        title(ax[0, i], "abc"[i], n.capitalize(), f"{t} significant; {u} up, {d} down at |log2FC| > 0.5")
    meth = {"DESeq2": ("padj", BLUE), "chi-square, stratified by study": ("q_cmh", AQUA), "chi-square, study ignored": ("q_naive", BASE)}
    rows = []
    for n, s in CON:
        d = de[n]
        row = {"contrast": n}
        for k, (col, _) in meth.items():
            if col in d:
                row[k] = int((d[col] < 0.05).sum())
        if find(f"07_{s}.csv") is not None:
            t7 = read(f"07_{s}.csv", index_col=0)
            row["Welch t-test"], row["Mann-Whitney"] = int((t7.q_t < 0.05).sum()), int((t7.q_rank < 0.05).sum())
        rows.append(row)
    M = pd.DataFrame(rows).set_index("contrast")
    cols = {**{k: c for k, (_, c) in meth.items()}, "Welch t-test": MAGENTA, "Mann-Whitney": TRAIN}
    a, w = ax[1, 0], 0.8 / len(M.columns)
    for j, k in enumerate(M.columns):
        xs = np.arange(len(M)) + (j - (len(M.columns) - 1) / 2) * w
        a.bar(xs, M[k], w * 0.9, color=cols[k], label=k)
        for x_, v in zip(xs, M[k]):
            a.text(x_, v + 40, v, ha="center", fontsize=7, color=INK2, rotation=90)
    a.set_xticks(range(len(M)), M.index)
    a.set(ylabel="significant genes (q < 0.05)")
    a.legend(fontsize=8, loc="upper right")
    a.grid(axis="y")
    title(a, "d", "Genes called by each method")
    a = ax[1, 1]
    comp = pd.DataFrame({n: {"both": ((de[n].q_naive < 0.05) & (de[n].q_cmh < 0.05)).sum(), "study-stratified only": ((de[n].q_naive >= 0.05) & (de[n].q_cmh < 0.05)).sum(),
                             "study ignored only": ((de[n].q_naive < 0.05) & (de[n].q_cmh >= 0.05)).sum()} for n, _ in CON}).T if "q_cmh" in de[CON[0][0]] else None
    if comp is not None:
        left = np.zeros(len(comp))
        for k, c in [("both", AQUA), ("study-stratified only", BLUE), ("study ignored only", BASE)]:
            a.barh(comp.index, comp[k], left=left, color=c, label=k, height=0.6, edgecolor=SURF, linewidth=1.5)
            left += comp[k].values
        a.invert_yaxis()
        a.legend(loc="lower right")
        a.set(xlabel="genes")
        a.grid(axis="x")
    title(a, "e", "Ignoring study adds genes", "chi-square on median-split expression")
    a = ax[1, 2]
    es = pd.DataFrame({n: {"significant": (de[n].padj < 0.05).mean() * 100, "and |log2FC| > 1": ((de[n].padj < 0.05) & (de[n].log2FoldChange.abs() > 1)).mean() * 100} for n, _ in CON}).T
    xs = np.arange(len(es))
    a.bar(xs - 0.2, es["significant"], 0.38, color=LBLUE, label="significant")
    a.bar(xs + 0.2, es["and |log2FC| > 1"], 0.38, color=DBLUE, label="significant and |log2FC| > 1")
    for x_, v in zip(xs - 0.2, es["significant"]):
        a.text(x_, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8, color=INK2)
    for x_, v in zip(xs + 0.2, es["and |log2FC| > 1"]):
        a.text(x_, v + 0.8, f"{v:.1f}%", ha="center", fontsize=8, color=INK2)
    a.set_xticks(xs, es.index)
    a.set(ylabel="% of tested genes")
    a.legend()
    a.grid(axis="y")
    title(a, "f", "Most significant genes change little")
    fig.tight_layout()
    save(fig, "fig3_de")


# 4 pathways ---------------------------------------------------------------------------------------------------------
def fig4():
    PW = [("06_human_acute", "up", "Human acute, up"), ("06_human_training", "up", "Human training, up"), ("06_mouse_training", "up", "Mouse training, up"),
          ("06_mouse_training", "down", "Mouse training, down")]
    PW = [p for p in PW if find(f"08_{p[0]}_pathways.csv") is not None]
    if not PW:
        print("  skipped, missing: 08_*_pathways.csv")
        return
    fig, ax = plt.subplots(2, len(PW), figsize=(4.8 * len(PW), 9), squeeze=False)
    for j, (stem, dirn, name) in enumerate(PW):
        df = read(f"08_{stem}_pathways.csv")
        for i, (lib, lname) in enumerate([("MSigDB_Hallmark_2020", "Hallmark"), ("Reactome_2022", "Reactome")]):
            a = ax[i, j]
            d = df[(df.library == lib) & (df.direction == dirn) & (df.fold_enrichment > 1) & (df.hits >= 3)].sort_values("q_fisher").head(8).iloc[::-1]
            title(a, "abcdefgh"[i * len(PW) + j], f"{name}, {lname}")
            if d.empty:
                a.text(0.5, 0.5, "no pathway with 3 or more hits", transform=a.transAxes, ha="center", color=INK2)
                a.set_axis_off()
                continue
            for y, (_, r) in enumerate(d.iterrows()):
                c = SEQ(min(-np.log10(max(r.q_fisher, 1e-300)), 20) / 20)
                a.scatter(r.fold_enrichment, y, s=30 + 14 * min(r.hits, 40), facecolors=c if r.q_fisher < 0.05 else "none", edgecolors=c, linewidths=1.6)
            a.set_yticks(range(len(d)), [textwrap.shorten(re.sub(r" R-HSA-\d+$", "", p), 40, placeholder="...") for p in d.pathway])
            a.set_xscale("log")
            a.set_xlim(d.fold_enrichment.min() / 1.6, d.fold_enrichment.max() * 1.6)
            a.xaxis.set_major_formatter(ScalarFormatter())
            a.xaxis.set_minor_formatter(NullFormatter())
            a.set_xlabel("fold enrichment")
            a.grid(axis="x")
            a.set_ylim(-0.6, len(d) - 0.4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    cax = fig.add_axes([0.35, 0.0, 0.3, 0.012])
    fig.colorbar(ScalarMappable(Normalize(0, 20), SEQ), cax=cax, orientation="horizontal", label="-log10 q (Fisher); hollow = q at least 0.05; dot size = genes in the list")
    save(fig, "fig4_pathways")


# 5 literature hypotheses --------------------------------------------------------------------------------------------
def fig5():
    if not need("10_human_hypotheses.csv"):
        return
    H = read("10_human_hypotheses.csv")
    Mo = read("10_mouse_hypotheses.csv") if find("10_mouse_hypotheses.csv") is not None else None
    col = {"confirmed": AQUA, "opposite direction": RED, "not significant": BASE}
    fig, a = plt.subplots(figsize=(10.5, 6.5))
    n = len(H)
    for i, r in H.iterrows():
        y = n - 1 - i
        for df, off, mk in [(H, 0.14, "o"), (Mo, -0.14, "s")]:
            if df is None:
                continue
            q = df[(df.gene == r.gene) & (df.contrast == r.contrast)]
            if q.empty:
                continue
            q = q.iloc[0]
            if q.verdict in col:
                a.scatter(q.log2FoldChange, y + off, s=70, marker=mk, color=col[q.verdict], zorder=3)
            else:
                a.text(0.02, y + off, f"not tested ({'human' if mk == 'o' else 'mouse'})", fontsize=8, color=INK2, va="center")
    a.set_yticks(range(n), [f"{r.gene}   {r.contrast}, {r.direction}" for _, r in H.iloc[::-1].iterrows()])
    a.axvline(0, color=INK2, lw=0.8)
    a.set_xlabel("log2 fold change (exercise vs none)")
    a.grid(axis="x")
    a.set_title("Genes reported in the exercise literature, tested in this data", fontsize=11.5, color=INK, pad=12)
    a.legend(handles=[Patch(color=c, label=k_) for k_, c in col.items()] + [Line2D([], [], marker="o", color=INK2, ls="", label="human"),
                                                                           Line2D([], [], marker="s", color=INK2, ls="", label="mouse")], loc="lower right")
    fig.tight_layout()
    save(fig, "fig5_hypotheses")


# 6 human against mouse ----------------------------------------------------------------------------------------------
def species_scatter(a, h, m, name):
    d = h[["log2FoldChange", "padj", "stat"]].join(m[["log2FoldChange", "padj", "stat"]], lsuffix="_h", rsuffix="_m", how="inner").dropna()
    sh, sm = d.padj_h < 0.05, d.padj_m < 0.05
    for mask, c, s, lab in [(~sh & ~sm, NS, 4, "neither"), (sh ^ sm, BLUE, 6, "one species"), (sh & sm, TRAIN, 14, "both")]:
        a.scatter(d.log2FoldChange_h[mask], d.log2FoldChange_m[mask], s=s, c=c, lw=0, label=f"{lab} ({mask.sum()})", rasterized=True)
    a.axhline(0, color=INK2, lw=0.6)
    a.axvline(0, color=INK2, lw=0.6)
    r, p = spearmanr(d.log2FoldChange_h, d.log2FoldChange_m)
    lim = np.percentile(np.abs(d.log2FoldChange_m), 99.9) * 1.15
    a.set_ylim(-lim, lim)
    a.set_xlim(-d.log2FoldChange_h.abs().max() * 1.08, d.log2FoldChange_h.abs().max() * 1.08)
    a.set(xlabel="human log2 fold change", ylabel="mouse log2 fold change")
    a.legend(loc="lower right", fontsize=8)
    both = d[sh & sm & (np.sign(d.log2FoldChange_h) == np.sign(d.log2FoldChange_m))].assign(s=lambda x: np.minimum(x.stat_h.abs(), x.stat_m.abs())).sort_values("s", ascending=False)
    label_genes(a, both, "log2FoldChange_h", "log2FoldChange_m", 6)
    return r, p, len(d)


def fig6():
    pairs = [("human_acute", "mouse_acute", "Acute exercise"), ("human_training", "mouse_training", "Long-term training")]
    ok = [p for p in pairs if find(f"06_{p[0]}.csv") is not None and find(f"06_{p[1]}.csv") is not None]
    reg = find("12_regulators.csv") is not None
    if not ok:
        print("  skipped, missing: 06 files for both species")
        return
    fig, ax = plt.subplots(1, len(ok) + reg, figsize=(6.2 * (len(ok) + reg), 6))
    ax = np.atleast_1d(ax)
    for i, (hn, mn, nm) in enumerate(ok):
        r, p, n = species_scatter(ax[i], read(f"06_{hn}.csv", index_col=0), read(f"06_{mn}.csv", index_col=0), nm)
        title(ax[i], "abc"[i], nm)
        ax[i].text(0.03, 0.97, f"Spearman {r:.2f}", transform=ax[i].transAxes, va="top", fontsize=9, color=INK2)
    if reg:
        g = read("12_regulators.csv")
        g = g[g.kind == "TF"].pivot(index="regulator", columns="contrast", values="z")
        if {"human training", "mouse training"} <= set(g.columns):
            g = g.dropna(subset=["human training", "mouse training"])
            a = ax[len(ok)]
            a.scatter(g["human training"], g["mouse training"], s=14, c=BLUE, lw=0, alpha=0.7)
            a.axhline(0, color=INK2, lw=0.6)
            a.axvline(0, color=INK2, lw=0.6)
            r = spearmanr(g["human training"], g["mouse training"])[0]
            sc = (g["human training"] * g["mouse training"]).where(np.sign(g["human training"]) == np.sign(g["mouse training"])).abs().sort_values(ascending=False)
            for t in sc.index[:7]:
                a.annotate(t, (g.loc[t, "human training"], g.loc[t, "mouse training"]), xytext=(4, 3), textcoords="offset points", fontsize=8)
            a.set(xlabel="human training, TF activity z", ylabel="mouse training, TF activity z")
            title(a, "abc"[len(ok)], "Transcription factor activity, training")
            a.text(0.03, 0.97, f"Spearman {r:.2f}", transform=a.transAxes, va="top", fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "fig6_species")


# 7 extra questions --------------------------------------------------------------------------------------------------
def fig7():
    meta, X = load()
    human, none, acute, train = meta.species == "human", meta.condition == "no exercise", meta.condition == "acute exercise", meta.condition == "long-term training"
    pb = X[none].groupby(meta.person[none]).mean()

    def paired(mask, cols):
        idx = meta.index[mask & meta.person.isin(pb.index)]
        return X.loc[idx, cols] - pb.loc[meta.person[idx], cols].values

    fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
    if find("11_stopwatch_genes.csv") is not None:  # a
        sw = read("11_stopwatch_genes.csv", index_col=0)
        show = list(sw.rho.nsmallest(2).index) + list(sw.rho.nlargest(2).index)
        D = paired(human & acute, show)
        hrs = [0, 1, 3, 4, 24]
        for g, c in zip(show, [BLUE, ACUTE, AQUA, MAGENTA]):
            mm = [D[g][meta.hours[D.index] == h].mean() for h in hrs]
            se = [D[g][meta.hours[D.index] == h].sem() for h in hrs]
            ax[0, 0].errorbar(range(5), mm, yerr=se, marker="o", ms=6, lw=2, color=c, label=f"{g} (rho {sw.rho[g]:.2f})")
        ax[0, 0].axhline(0, color=INK2, lw=0.6)
        ax[0, 0].set_xticks(range(5), [f"{h} h" for h in hrs])
        ax[0, 0].set(ylabel="change against own baseline (log2 CPM)")
        ax[0, 0].legend(fontsize=8)
        ax[0, 0].grid(axis="y")
        title(ax[0, 0], "a", "Stopwatch genes", "strongest correlations with hours after exercise")
    if find("11_response_scores.csv") is not None:  # b, c
        sc = read("11_response_scores.csv", index_col=0)
        order = sc.groupby("group").score.median().sort_values().index
        for i, g in enumerate(order):
            s = sc[sc.group == g]
            ax[0, 1].scatter(i + np.random.default_rng(0).uniform(-0.22, 0.22, len(s)), s.score, s=12, c=np.where(s.age_group == "aged", MAGENTA, BLUE), lw=0)
        ax[0, 1].set_xticks(range(len(order)), order, rotation=90, fontsize=8)
        ax[0, 1].set(ylabel="immediate-early score (log2 change)")
        ax[0, 1].legend(handles=[Line2D([], [], marker="o", color=BLUE, ls="", label="young"), Line2D([], [], marker="o", color=MAGENTA, ls="", label="aged")], loc="upper left")
        ax[0, 1].grid(axis="y")
        title(ax[0, 1], "b", "Response of each person", "mean change of 11 immediate-early genes")
        pos, labels = [], []
        for k, h in enumerate([0, 3]):
            for j, g in enumerate(["young", "aged"]):
                s = sc[(sc.hours == h) & (sc.age_group == g)].score
                x = k * 2.8 + j
                if len(s):
                    ax[0, 2].boxplot(s, positions=[x], widths=0.5, showfliers=False, medianprops={"color": INK, "lw": 1.5}, boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
                    ax[0, 2].scatter(x + np.random.default_rng(1).uniform(-0.15, 0.15, len(s)), s, s=12, c=BLUE if g == "young" else MAGENTA, lw=0, zorder=3)
                pos.append(x)
                labels.append(f"{g}\nn={len(s)}")
        ax[0, 2].set_xticks(pos, labels)
        ax[0, 2].set(ylabel="immediate-early score (log2 change)")
        ax[0, 2].set_ylim(top=ax[0, 2].get_ylim()[1] * 1.16)
        for k, h in enumerate([0, 3]):
            y, a_ = sc[(sc.hours == h) & (sc.age_group == "young")].score, sc[(sc.hours == h) & (sc.age_group == "aged")].score
            if len(y) > 2 and len(a_) > 2:
                ax[0, 2].text(k * 2.8 + 0.5, ax[0, 2].get_ylim()[1] * 0.99, f"{h} h after exercise, Mann-Whitney p = {mannwhitneyu(y, a_).pvalue:.2g}", ha="center", va="top", fontsize=8.5, color=INK2)
        ax[0, 2].grid(axis="y")
        title(ax[0, 2], "c", "Aged against young", " ")
    g3 = ["MYH7", "MYH2", "MYH1"]  # d
    cpm = 2 ** X[g3] - 1
    share = cpm.div(cpm.sum(axis=1), axis=0)
    ps = (2 ** pb[g3] - 1)
    ps = ps.div(ps.sum(axis=1), axis=0)
    idx = meta.index[human & train & meta.person.isin(pb.index)]
    before, after = ps.loc[meta.person[idx]], share.loc[idx]
    xs = np.arange(3)
    ax[1, 0].bar(xs - 0.2, before.mean(), 0.38, color=BASE, label="before training")
    ax[1, 0].bar(xs + 0.2, after.mean(), 0.38, color=TRAIN, label="after training")
    ax[1, 0].errorbar(xs - 0.2, before.mean(), before.sem(), fmt="none", ecolor=INK2, lw=1)
    ax[1, 0].errorbar(xs + 0.2, after.mean(), after.sem(), fmt="none", ecolor=INK2, lw=1)
    ax[1, 0].set_xticks(xs, ["MYH7\n(type I)", "MYH2\n(type IIa)", "MYH1\n(type IIx)"])
    ax[1, 0].set(ylabel="share of the three myosin genes")
    ax[1, 0].legend()
    ax[1, 0].grid(axis="y")
    title(ax[1, 0], "d", "Fibre type genes after training", f"{meta.person[idx].nunique()} people measured before and after")
    circ = [g for g in "NR1D1 NR1D2 PER1 PER2 PER3 CRY1 CRY2 CLOCK DBP TEF HLF NPAS2 NFIL3 RORA CIART".split() if g in X.columns]  # e
    Dc = paired(human & acute, circ)
    hrs = [h for h in [0, 1, 3, 4, 24] if (meta.hours[Dc.index] == h).sum() >= 10]
    med = pd.DataFrame({h: Dc[meta.hours[Dc.index] == h].median() for h in hrs})
    pv = pd.DataFrame({h: [wilcoxon(Dc[c][meta.hours[Dc.index] == h]).pvalue for c in circ] for h in hrs}, index=circ)
    qv = pd.DataFrame(bh(pv.values.ravel()).reshape(pv.shape), index=circ, columns=hrs)
    im = ax[1, 1].imshow(med.values, cmap=DIV, vmin=-1.5, vmax=1.5, aspect="auto")
    for i in range(len(circ)):
        for j in range(len(hrs)):
            ax[1, 1].text(j, i, f"{med.values[i, j]:.1f}" + ("*" if qv.values[i, j] < 0.05 else ""), ha="center", va="center", fontsize=8, color=INK)
    ax[1, 1].set_xticks(range(len(hrs)), [f"{h} h" for h in hrs])
    ax[1, 1].set_yticks(range(len(circ)), circ)
    for s in ax[1, 1].spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax[1, 1], label="median change (log2 CPM); * = BH q < 0.05", shrink=0.8)
    title(ax[1, 1], "e", "Circadian genes after one bout", "* = BH q < 0.05 over the table")
    if find("11_fitness_genes.csv") is not None:  # f
        f = read("11_fitness_genes.csv", header=[0, 1], index_col=0)
        x, y = f[("GSE120862 centred", "log2FC")], f[("GSE151066 centred", "log2FC")]
        ok = x.notna() & y.notna()
        ax[1, 2].scatter(x[ok], y[ok], s=4, c=BLUE, lw=0, alpha=0.5, rasterized=True)
        ax[1, 2].axhline(0, color=INK2, lw=0.6)
        ax[1, 2].axvline(0, color=INK2, lw=0.6)
        ax[1, 2].set(xlabel="trained vs untrained (GSE120862), log2FC", ylabel="active vs sedentary (GSE151066), log2FC")
        title(ax[1, 2], "f", "Resting fitness in two studies", f"{ok.sum()} genes, Spearman {spearmanr(x[ok], y[ok])[0]:.2f}")
    fig.tight_layout()
    save(fig, "fig7_questions")


# 8 reverse engineering ----------------------------------------------------------------------------------------------
def fig8():
    if not need("12_regulators.csv", "12_cascade.csv", "12_composition.csv"):
        return
    reg, C, ct = read("12_regulators.csv"), read("12_cascade.csv", index_col=0), read("12_composition.csv")
    fig, ax = plt.subplots(2, 3, figsize=(17, 11))
    evc = lambda e: DBLUE if e >= 3 else (LBLUE if e == 2 else NS)
    for i, c in enumerate(["human acute", "human training", "mouse training"]):
        d = reg[(reg.contrast == c) & (reg.kind == "TF")]
        a = ax[0, i]
        if d.empty:
            a.set_axis_off()
            continue
        s = pd.concat([d.nlargest(7, "z"), d.nsmallest(7, "z")]).drop_duplicates("regulator").sort_values("z")
        a.barh(s.regulator, s.z, color=[evc(e) for e in s.evidence], height=0.7)
        a.axvline(0, color=INK2, lw=0.8)
        a.set_xlabel("activity z (positive = targets go up)")
        a.grid(axis="x")
        title(a, "abc"[i], f"Transcription factors, {c}", f"{(d.q_activity < 0.05).sum()} of {len(d)} reach q < 0.05 in the activity test")
    pick = list(dict.fromkeys(sum([list(C[b].abs().nlargest(5).index) for b in C.columns], [])))
    Cp = C.loc[pick].sort_values(C.columns[0])
    im = ax[1, 0].imshow(Cp.values, cmap=DIV, vmin=-4, vmax=4, aspect="auto")
    for i in range(Cp.shape[0]):
        for j in range(Cp.shape[1]):
            ax[1, 0].text(j, i, f"{Cp.values[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax[1, 0].set_xticks(range(Cp.shape[1]), Cp.columns)
    ax[1, 0].set_yticks(range(len(Cp)), Cp.index)
    for s_ in ax[1, 0].spines.values():
        s_.set_visible(False)
    fig.colorbar(im, ax=ax[1, 0], label="activity z", shrink=0.8)
    title(ax[1, 0], "d", "Transcription factor activity over time", "after one exercise bout")
    pv = ct.pivot(index="score", columns="contrast", values="change")
    qv = ct.pivot(index="score", columns="contrast", values="q_t")
    im = ax[1, 1].imshow(pv.values, cmap=DIV, vmin=-1.5, vmax=1.5, aspect="auto")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax[1, 1].text(j, i, f"{pv.values[i, j]:.2f}" + ("*" if qv.values[i, j] < 0.05 else ""), ha="center", va="center", fontsize=7.5)
    ax[1, 1].set_xticks(range(pv.shape[1]), [c.replace("fitness ", "").replace(", unpaired", " (unpaired)").replace(", paired", " (paired)") for c in pv.columns], rotation=40, ha="right", fontsize=8)
    ax[1, 1].set_yticks(range(pv.shape[0]), pv.index)
    for s_ in ax[1, 1].spines.values():
        s_.set_visible(False)
    fig.colorbar(im, ax=ax[1, 1], label="change in marker score (SD units); * = BH q < 0.05", shrink=0.8)
    title(ax[1, 1], "e", "Cell population marker scores", "* = BH q < 0.05")
    t = find("12_reverse.txt")
    rows = re.findall(r"(human acute|human training|mouse training): significant genes (\d+) before, (\d+) after adjusting; (\d+) stay", t.read_text()) if t is not None else []
    if rows:
        s = pd.DataFrame(rows, columns=["c", "before", "after", "stay"]).set_index("c").astype(int)
        xs = np.arange(len(s))
        ax[1, 2].bar(xs - 0.2, s.before, 0.38, color=BASE, label="significant before adjusting")
        ax[1, 2].bar(xs + 0.2, s.stay, 0.38, color=BLUE, label="still significant after adjusting")
        for x_, b, k in zip(xs + 0.2, s.before, s.stay):
            ax[1, 2].text(x_, k + 40, f"{k / b:.0%}", ha="center", fontsize=9, color=INK2)
        ax[1, 2].set_xticks(xs, s.index)
        ax[1, 2].set(ylabel="genes (marker genes excluded)")
        ax[1, 2].legend()
        ax[1, 2].grid(axis="y")
    title(ax[1, 2], "f", "Effect of adjusting for cell composition")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.legend(handles=[Patch(color=DBLUE, label="TF evidence 3 or 4"), Patch(color=LBLUE, label="TF evidence 2"), Patch(color=NS, label="TF evidence 0 or 1")], loc="lower center", ncol=3)
    save(fig, "fig8_reverse")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8}
for n in [int(a) for a in sys.argv[1:]] or list(FIGS):
    print(f"figure {n}")
    try:
        FIGS[n]()
    except Exception:
        traceback.print_exc()