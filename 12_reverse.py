"""Step 12: reverse engineering. Which regulators, cell populations and gene modules explain the expression changes?  Usage: python 12_reverse.py
Needs database2_detailed.csv and results/06_human_acute.csv, 06_human_training.csv, 06_mouse_training.csv (run step 6 first).
Internet is needed once: gene-set libraries come from Enrichr through gseapy and are cached as results/12_lib_*.json.
  A  upstream regulators   transcription factors and kinases. Two tests per contrast: Fisher's exact test on the DESeq2 up and down lists,
                           and a competitive target-set test on all genes (mean t of the targets against the other genes, variance
                           inflated for gene-gene correlation estimated from 300 label permutations inside each study). Each regulator
                           gets one point per line of evidence (see the evidence column)
  B  time cascade          the same activity test at 0-1 h, 3-4 h and 24 h after one bout (paired changes, sign-flip null)
  C  biopsy composition    marker scores for fibre types, endothelium, fibroblasts, satellite cells, pericytes, immune cells;
                           how they change, and how much of each contrast survives adjusting for them
  D  co-expression         modules among the strongest acute genes, hub genes, leave-one-study-out stability of the hubs
Writes results/12_reverse.txt, 12_reverse.png, 12_regulators.csv, 12_cascade.csv, 12_composition.csv, 12_modules.csv."""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import hypergeom, mannwhitneyu, norm, spearmanr, t as tdist, ttest_1samp, ttest_ind, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
R = ROOT / "results"
B = 300  # permutations
rng = np.random.default_rng(1)
report = []
fmt = lambda v: f"{v:.3g}"


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


db = pd.read_csv(ROOT / "database2_detailed.csv", index_col=0, low_memory=False)
last = list(db.columns).index("library_size_millions") + 1
meta, X = db.iloc[:, :last].copy(), db.iloc[:, last:]
meta["hours"] = pd.to_numeric(meta.hours_after_exercise, errors="coerce")
genes = list(X.columns)
human, mouse = (meta.species == "human"), (meta.species == "mouse")
none, acute, train = (meta.condition == "no exercise"), (meta.condition == "acute exercise"), (meta.condition == "long-term training")
base = X[none].groupby(meta.study[none]).mean()
rel = X - base.reindex(meta.study).values  # expression relative to the baseline of the same study
pbase = X[none].groupby(meta.person[none]).mean()  # baseline of each person


def paired(mask, cols=None):
    idx = meta.index[mask & meta.person.isin(pbase.index)]
    cols = genes if cols is None else cols
    return X.loc[idx, cols] - pbase.loc[meta.person[idx], cols].values


# gene-set libraries --------------------------------------------------------------------------------------------------
LIBS = [("TF", "ChEA_2022"), ("TF", "ENCODE_TF_ChIP-seq_2015"), ("TF", "TRRUST_Transcription_Factors_2019"), ("kinase", "KEA_2015"),
        ("cell", "PanglaoDB_Augmented_2021"), ("cell", "CellMarker_Augmented_2021"), ("pathway", "MSigDB_Hallmark_2020"), ("pathway", "Reactome_2022")]


def load_lib(name):
    f = R / f"12_lib_{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    try:
        import gseapy as gp
        avail = gp.get_library_name()
        use = name if name in avail else next(iter(sorted([a for a in avail if a.startswith(name.split("_20")[0])], reverse=True)), None)
        lib = gp.get_library(use, organism="human")
    except Exception as e:
        say(f"library {name} unavailable ({type(e).__name__}), skipped")
        return None
    f.write_text(json.dumps(lib))
    return lib


SETS = []  # (kind, library, name, genes); TF and kinase sets are merged per symbol
for kind, name in LIBS:
    lib = load_lib(name)
    if lib is None:
        continue
    if kind in ("TF", "kinase"):
        per = defaultdict(list)
        for term, gl in lib.items():
            per[re.split(r"[ _]", term)[0].upper()].append({x.split(",")[0].upper() for x in gl})
        for sym, ss in per.items():
            if name.startswith(("ChEA", "ENCODE")) and len(ss) > 1:  # keep targets found in at least two experiments
                c = Counter(x for s in ss for x in s)
                SETS.append((kind, name, sym, {x for x, n in c.items() if n >= 2}))
            else:
                SETS.append((kind, name, sym, set().union(*ss)))
    else:
        SETS += [(kind, name, term, {x.split(",")[0].upper() for x in gl}) for term, gl in lib.items()]
say(f"gene-set libraries loaded: {sorted({s[1] for s in SETS})}  ({len(SETS)} sets)")
TFNAMES = {s[2] for s in SETS if s[0] == "TF"}


def membership(universe):
    pos = {g: i for i, g in enumerate(universe)}
    r, c, info = [], [], []
    for kind, lib, name, gs in SETS:
        ix = [pos[x] for x in gs if x in pos]
        if 10 <= len(ix) <= 1500:
            r += [len(info)] * len(ix)
            c += ix
            info.append((kind, lib, name, len(ix)))
    M = sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(info), len(universe)))
    return M, pd.DataFrame(info, columns=["kind", "lib", "name", "size"])


def fisher(sig, universe, direction):
    """one-sided Fisher exact test (hypergeometric) of every set against a list of significant genes"""
    rows, N, K = [], len(universe), len(sig)
    for kind, lib, name, gs in SETS:
        g = gs & universe
        m = len(g)
        k = len(g & sig)
        if 10 <= m <= 1500 and k >= 3:
            rows.append((kind, lib, name, direction, k, m, (k / K) / (m / N), hypergeom.sf(k - 1, N, m, K)))
    d = pd.DataFrame(rows, columns=["kind", "lib", "name", "direction", "hits", "size", "fold", "p"])
    if len(d):
        d["q"] = d.groupby("lib").p.transform(lambda s: pd.Series(bh(s.values), index=s.index))
    return d


def activity(stat, null, M, info):
    """competitive test of every set (Wu and Smyth, CAMERA): mean statistic inside the set against outside, with the variance inflated by the
    correlation between the genes of the set, which is read from the label permutations (null: B x genes)"""
    N = len(stat)
    m = np.asarray(M.sum(1)).ravel()
    inside = (M @ stat) / m
    outside = (N * stat.mean() - m * inside) / (N - m)
    vif = np.maximum(((M @ null.T) / m[:, None]).var(1) * m / null.var(0).mean(), 1)
    z = (inside - outside) / np.sqrt(stat.var() * (vif / m + 1 / (N - m)))
    out = info.copy()
    out["z"] = z
    out["vif"] = vif
    out["p"] = 2 * norm.sf(np.abs(z))
    out["q"] = out.groupby("lib").p.transform(lambda v: pd.Series(bh(v.values), index=v.index))
    return out


# linear model per gene: log2 CPM ~ study (+ covariates) + group; t statistic of the group ----------------------------
def prep(idx, extra=None):
    st = pd.get_dummies(meta.study[idx], drop_first=True, dtype=float).values
    C = np.column_stack([np.ones(len(idx)), st] + ([extra] if extra is not None else []))
    U, s, _ = np.linalg.svd(C, full_matrices=False)
    Q = U[:, s > 1e-8 * s[0]]
    Y = X.loc[idx].values.astype(float)
    Ys = Y - Q @ (Q.T @ Y)
    return Q, Ys, (Ys**2).sum(0)


def tstat(g, fit):
    Q, Ys, ss = fit
    gs = g - Q @ (Q.T @ g)
    den = gs @ gs
    beta = gs @ Ys / den
    df = len(g) - Q.shape[1] - 1
    return beta / np.sqrt((ss - beta**2 * den) / df / den), df


def permute(g, groups):
    out = g.copy()
    for ix in groups:
        out[ix] = rng.permutation(g[ix])
    return out


CONTRASTS = [("human", "acute", 1.0), ("human", "training", 0.5), ("mouse", "training", 0.0)]  # name, contrast, |log2FC| cutoff for the Fisher lists
design = {}
for sp, which, thr in CONTRASTS:
    sel = (meta.species == sp) & (none | (acute if which == "acute" else train))
    ok = [s for s, d in meta[sel].groupby("study") if d.condition.nunique() == 2]
    idx = meta.index[sel & meta.study.isin(ok)]
    design[sp, which] = (idx, (meta.condition[idx] != "no exercise").values.astype(float), meta.study[idx].values)

# A upstream regulators ----------------------------------------------------------------------------------------------
say("\nA. Upstream regulators")
M, info = membership(genes) if SETS else (None, None)
tobs, regs = {}, []
for sp, which, thr in CONTRASTS:
    name = f"{sp} {which}"
    idx, g, st = design[sp, which]
    groups = [np.where(st == s)[0] for s in np.unique(st)]
    fit = prep(idx)
    t_obs, _ = tstat(g, fit)
    tobs[name] = pd.Series(t_obs, index=genes)
    f = R / f"06_{sp}_{which}.csv"
    de = pd.read_csv(f, index_col=0).dropna(subset=["padj"]) if f.exists() else None
    say(f"\n{name}: {len(idx)} samples from {len(np.unique(st))} studies")
    if de is not None:
        common = de.index.intersection(genes)
        say(f"  agreement of the linear-model t with the DESeq2 statistic over {len(common)} genes: Spearman {spearmanr(tobs[name][common], de.loc[common, 'stat'])[0]:.2f}")
    if not SETS:
        continue
    null = np.array([tstat(permute(g, groups), fit)[0] for _ in range(B)])
    act = activity(t_obs, null, M, info)
    fish = pd.DataFrame()
    if de is not None:
        uni = set(de.index)
        up, dn = set(de[(de.padj < 0.05) & (de.log2FoldChange > thr)].index), set(de[(de.padj < 0.05) & (de.log2FoldChange < -thr)].index)
        say(f"  DESeq2 lists: {len(up)} up, {len(dn)} down (padj<0.05, |log2FC|>{thr:g})")
        fish = pd.concat([fisher(up, uni, "up") if len(up) >= 5 else None, fisher(dn, uni, "down") if len(dn) >= 5 else None])
    rows = []
    for (kind, sym), d in act[act.kind.isin(["TF", "kinase"])].groupby(["kind", "name"]):
        best = d.loc[d.z.abs().idxmax()]
        sign = np.sign(best.z)
        libs_ok = int(((d.q < 0.05) & (np.sign(d.z) == sign)).sum())
        fq = np.nan
        if len(fish):
            ff = fish[(fish.kind == kind) & (fish.name == sym) & (fish.direction == ("up" if sign > 0 else "down"))]
            fq = ff.q.min() if len(ff) else np.nan
        lfc, padj = (de.log2FoldChange.get(sym, np.nan), de.padj.get(sym, np.nan)) if de is not None else (np.nan, np.nan)
        pts = [best.q < 0.05, libs_ok >= 2, fq < 0.05, bool(padj < 0.05 and np.sign(lfc) == sign)]
        rows.append({"contrast": name, "kind": kind, "regulator": sym, "z": best.z, "q_activity": best.q,
                     "libraries_supporting": libs_ok, "q_fisher": fq, "mRNA_log2FC": lfc, "mRNA_padj": padj, "evidence": int(sum(pts))})
    reg = pd.DataFrame(rows).assign(absz=lambda d: d.z.abs()).sort_values(["evidence", "absz"], ascending=False).drop(columns="absz")
    regs.append(reg)
    cols = ["regulator", "z", "q_activity", "libraries_supporting", "q_fisher", "mRNA_log2FC", "mRNA_padj", "evidence"]
    for kind, n in [("TF", 12), ("kinase", 5)]:
        if len(reg) and (reg.kind == kind).any():
            say(f"  top {kind}s (activity z > 0 means targets go up; evidence = significant activity, 2+ libraries, Fisher list, own mRNA moves the same way)\n" +
                reg[reg.kind == kind].head(n)[cols].to_string(index=False, float_format=fmt))
    for kind, n in [("cell", 6), ("pathway", 6)]:
        d = act[(act.kind == kind) & (act.q < 0.05)].sort_values("z", key=abs, ascending=False)
        if len(d):
            say(f"  top {kind} sets by activity\n" + d.head(n)[["lib", "name", "size", "z", "q"]].to_string(index=False, float_format=fmt))
if regs:
    allreg = pd.concat(regs)
    allreg.round(5).to_csv(R / "12_regulators.csv", index=False)
    tf = allreg[allreg.kind == "TF"].pivot(index="regulator", columns="contrast", values="z")
    if tf.shape[1] == 3:
        both = tf.dropna()
        for a, b in [("human acute", "human training"), ("human training", "mouse training")]:
            r, p = spearmanr(both[a], both[b])
            say(f"\n  agreement of TF activity between {a} and {b}: Spearman {r:.2f} over {len(both)} TFs, p = {p:.2g}")
        top = both[(both["human training"].abs() > 2) & (both["mouse training"].abs() > 2) & (np.sign(both["human training"]) == np.sign(both["mouse training"]))]
        say(f"  TFs with |z|>2 in the same direction in human and mouse training: {', '.join(top.index[:15]) if len(top) else 'none'}")

# B time cascade -----------------------------------------------------------------------------------------------------
say("\nB. Time cascade after one bout (paired changes against each person's baseline)")
BINS = {"0-1h": [0, 1], "3-4h": [3, 4], "24h": [24]}
casc, mrna = {}, {}
if SETS:
    for bname, hrs in BINS.items():
        D = paired(human & acute & meta.hours.isin(hrs))
        n = len(D)
        if n < 10:
            continue
        Dv = D.values
        mean = Dv.mean(0)
        sumsq = (Dv**2).sum(0)
        t_obs = mean / np.sqrt((sumsq - n * mean**2) / (n - 1) / n)
        null = np.empty((B, len(genes)))
        for b in range(B):
            s = rng.choice([-1.0, 1.0], size=n)
            mn = s @ Dv / n
            null[b] = mn / np.sqrt((sumsq - n * mn**2) / (n - 1) / n)
        act = activity(t_obs, null, M, info)
        d = act[act.kind.isin(["TF", "kinase"])]
        casc[bname] = d.assign(a=d.z.abs()).sort_values("a").drop_duplicates(["kind", "name"], keep="last").set_index(["kind", "name"]).z
        mrna[bname] = pd.Series(mean, index=genes)
        say(f"  {bname}: {n} paired samples")
    if casc:
        C = pd.DataFrame(casc)
        C = C[C.index.get_level_values(0) == "TF"].droplevel(0)
        pick = sorted(set().union(*[C[b].abs().nlargest(6).index for b in C.columns]))
        tab = C.loc[pick].round(1)
        for b in C.columns:
            tab[f"{b} mRNA"] = [round(mrna[b].get(g_, np.nan), 2) for g_ in pick]
        say("  TF activity z per window (and the change in the TF's own mRNA, log2):\n" + tab.to_string())
        for b in C.columns:
            both = pd.DataFrame({"z": C[b], "m": [mrna[b].get(g_, np.nan) for g_ in C.index]}).dropna()
            say(f"    {b}: TFs whose own mRNA rises tend to have more active targets: Spearman {spearmanr(both.z, both.m)[0]:.2f} (n={len(both)})")
        C.round(3).to_csv(R / "12_cascade.csv")

# C biopsy composition -----------------------------------------------------------------------------------------------
say("\nC. Biopsy composition")
MARK = {"slow fibre": "MYH7 TNNT1 TNNI1 TNNC1 MYL3 MYL2 ATP2A2 MYBPC1",
        "fast fibre": "MYH2 MYH1 TNNT3 TNNI2 TNNC2 MYL1 MYLPF ATP2A1 MYBPC2 PVALB ACTN3",
        "endothelium": "PECAM1 VWF CDH5 CLDN5 KDR ERG ROBO4 CD34 EMCN FLT1 PLVAP",
        "pericyte": "RGS5 PDGFRB ACTA2 MYH11 NOTCH3 CSPG4",
        "fibroblast": "PDGFRA DCN LUM COL1A1 COL1A2 COL3A1 TCF21 DPT FBN1",
        "satellite cell": "PAX7 MYF5 CALCR CDH15 PAX3 VCAM1",
        "immune": "PTPRC CD68 CD163 CSF1R MRC1 LYZ CD14 C1QA C1QB C1QC AIF1 CD53"}
MARK = {k: [x for x in v.split() if x in X.columns] for k, v in MARK.items()}
markers = {x for v in MARK.values() for x in v}
say("  marker genes found: " + ", ".join(f"{k} {len(v)}" for k, v in MARK.items()))
comp = pd.DataFrame(index=meta.index, columns=list(MARK), dtype=float)
for sp in ("human", "mouse"):
    ix = meta.index[meta.species == sp]
    for k, gl in MARK.items():
        comp.loc[ix, k] = (rel.loc[ix, gl] / rel.loc[ix, gl].std()).mean(axis=1)
cbase = comp[none].groupby(meta.person[none]).mean()
tests = []


def one(label, k, d):
    tests.append((label, k, len(d), d.mean(), ttest_1samp(d, 0).pvalue, wilcoxon(d).pvalue))


def two(label, k, a, b):
    tests.append((label, k, len(a) + len(b), a.mean() - b.mean(), ttest_ind(a, b, equal_var=False).pvalue, mannwhitneyu(a, b).pvalue))


for k in MARK:
    for label, mask in [("human acute, paired", human & acute), ("human training, paired", human & train)]:
        idx = meta.index[mask & meta.person.isin(cbase.index)]
        one(label, k, comp.loc[idx, k] - cbase.loc[meta.person[idx], k].values)
    two("human training, unpaired", k, comp.loc[human & train, k], comp.loc[human & none, k])
    two("mouse training, unpaired", k, comp.loc[mouse & train, k], comp.loc[mouse & none, k])
    for study, (hi, lo) in {"GSE120862": ("trained", "untrained"), "GSE151066": ("active", "sedentary")}.items():
        s = none & (meta.study == study)
        two(f"fitness {study}, {hi} vs {lo}", k, comp.loc[s & (meta.fitness == hi), k], comp.loc[s & (meta.fitness == lo), k])
ct = pd.DataFrame(tests, columns=["contrast", "score", "n", "change", "p_t", "p_rank"])
ct["q_t"] = bh(ct.p_t)
ct.round(5).to_csv(R / "12_composition.csv", index=False)
mat = ct.pivot(index="score", columns="contrast", values="change").reindex(list(MARK))
sig = ct.pivot(index="score", columns="contrast", values="q_t").reindex(list(MARK)) < 0.05
say("  change in each marker score (units of gene standard deviation; * = BH q<0.05 over all tests)\n" +
    (mat.round(2).astype(str) + np.where(sig, "*", " ")).to_string())
say("  adjusting each contrast for all marker scores (marker genes left out of the counts):")
nonm = np.array([x not in markers for x in genes])
for sp, which, thr in CONTRASTS:
    idx, g, st = design[sp, which]
    t0, df0 = tstat(g, prep(idx))
    t1, df1 = tstat(g, prep(idx, comp.loc[idx].values))
    q0, q1 = bh(2 * tdist.sf(np.abs(t0), df0)), bh(2 * tdist.sf(np.abs(t1), df1))
    s0, s1 = (q0 < 0.05) & nonm, (q1 < 0.05) & nonm
    keep = s0 & s1 & (np.sign(t0) == np.sign(t1))
    say(f"    {sp} {which}: significant genes {s0.sum()} before, {s1.sum()} after adjusting; {keep.sum()} stay significant with the same sign "
        f"({keep.sum() / max(s0.sum(), 1):.0%}); correlation of t before and after {spearmanr(t0, t1)[0]:.2f}")
    top = pd.Series(t1, index=genes)[keep].sort_values(key=abs, ascending=False).head(8)
    say("      strongest genes that survive: " + ", ".join(f"{k} ({v:.1f})" for k, v in top.items()))
say("  Marker scores and group are related by design, so adjusting removes any change that moves with a cell population. Genes that survive are changes inside cells,"
    " genes that drop out may be either proportion changes or changes within that cell type.")

# D co-expression modules and hubs -----------------------------------------------------------------------------------
say("\nD. Co-expression among the strongest acute genes")
ast = meta.study[human & acute].unique()
nidx = meta.index[human & (none | acute) & meta.study.isin(ast)]
topg = list(tobs["human acute"].abs().nlargest(500).index)
E = rel.loc[nidx, topg].rank()
E = (E - E.mean()) / E.std()
mod = fcluster(linkage(E.values.T, "ward"), 6, "maxclust")
pw = [s for s in SETS if s[0] == "pathway"]


def eigen(A):
    u, s, _ = np.linalg.svd(A - A.mean(0), full_matrices=False)
    pc = u[:, 0] * s[0]
    return pc if np.corrcoef(pc, A.mean(1))[0, 1] >= 0 else -pc


label = pd.Series(np.where(meta.condition[nidx] == "no exercise", "baseline", "acute"), index=nidx)
hours = meta.hours[nidx]
stu = meta.study[nidx].values
mrows, kmes, eg_all = [], [], {}
for k in sorted(set(mod)):
    cols = [c for c, m_ in zip(topg, mod) if m_ == k]
    A = E[cols].values
    eg = eigen(A)
    kme = pd.Series([np.corrcoef(A[:, i], eg)[0, 1] for i in range(A.shape[1])], index=cols).sort_values(ascending=False)
    hubs = list(kme.index[:8])
    jac = []
    for s in np.unique(stu):
        keep = stu != s
        eg_s = eigen(A[keep])
        kme_s = pd.Series([np.corrcoef(A[keep][:, i], eg_s)[0, 1] for i in range(A.shape[1])], index=cols).sort_values(ascending=False)
        h2 = set(kme_s.index[:8])
        jac.append(len(h2 & set(hubs)) / len(h2 | set(hubs)))
    enr = fisher(set(cols), set(genes), "module") if pw else pd.DataFrame()
    enr = enr[enr.kind == "pathway"].sort_values("p").head(2) if len(enr) else enr
    direction = np.sign(tobs["human acute"][cols].mean())
    ac = label == "acute"
    r_t = spearmanr(eg[ac.values], hours[ac].fillna(-1).values)[0] if ac.sum() > 5 else np.nan
    eg_all[k] = eg
    mrows.append({"module": k, "genes": len(cols), "direction": "up" if direction > 0 else "down", "eigengene vs acute": np.corrcoef(eg, (label == "acute").astype(float))[0, 1],
                  "eigengene vs hours (acute)": r_t, "hubs": ", ".join(hubs), "TF hubs": ", ".join(h for h in hubs if h in TFNAMES),
                  "hub stability (LOSO Jaccard)": np.mean(jac), "top pathways": "; ".join(enr.name.str.replace(r" R-HSA-\d+", "", regex=True)) if len(enr) else ""})
    kmes.append(pd.DataFrame({"gene": kme.index, "module": k, "kME": kme.values}))
mt = pd.DataFrame(mrows)
say(f"  {len(nidx)} human samples, 500 genes, 6 modules (Ward linkage on rank-transformed expression)")
for _, r_ in mt.iterrows():
    say(f"  module {r_.module} ({r_.genes} genes, {r_.direction}): eigengene vs acute {r_['eigengene vs acute']:.2f}, vs hours {r_['eigengene vs hours (acute)']:.2f}, "
        f"hub stability {r_['hub stability (LOSO Jaccard)']:.2f}\n     hubs: {r_.hubs}\n     TF hubs: {r_['TF hubs'] or 'none'}\n     pathways: {r_['top pathways'] or 'none'}")
pd.concat(kmes).round(4).to_csv(R / "12_modules.csv", index=False)

# figure -------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
for a, name in zip(ax[0], tobs):
    d = allreg[(allreg.contrast == name) & (allreg.kind == "TF")] if regs else pd.DataFrame()
    if len(d):
        d = pd.concat([d.nlargest(8, "z"), d.nsmallest(8, "z")]).drop_duplicates("regulator").sort_values("z")
        a.barh(d.regulator, d.z, color=np.where(d.evidence >= 3, "tab:red", "tab:gray"))
    a.set(title=f"{name}: TF activity z (red = evidence 3 or 4)", xlabel="activity z")
if casc:
    im = ax[1, 0].imshow(C.loc[pick].values, cmap="RdBu_r", vmin=-6, vmax=6, aspect="auto")
    ax[1, 0].set(xticks=range(C.shape[1]), xticklabels=C.columns, yticks=range(len(pick)), yticklabels=pick, title="B TF activity by time after exercise")
    fig.colorbar(im, ax=ax[1, 0])
im = ax[1, 1].imshow(mat.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax[1, 1].set(xticks=range(mat.shape[1]), yticks=range(mat.shape[0]), yticklabels=mat.index, title="C change in marker scores")
ax[1, 1].set_xticklabels(mat.columns, rotation=60, ha="right", fontsize=6)
fig.colorbar(im, ax=ax[1, 1])
grp = pd.Series(np.where(label == "baseline", "baseline", np.select([hours.isin([0, 1]), hours.isin([3, 4]), hours == 24], ["0-1h", "3-4h", "24h"], "other")), index=nidx)
egm = pd.DataFrame({k: v for k, v in eg_all.items()}, index=nidx).groupby(grp).mean().reindex(["baseline", "0-1h", "3-4h", "24h"])
egm = (egm - egm.loc["baseline"]) / egm.std()
im = ax[1, 2].imshow(egm.T.values, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax[1, 2].set(xticks=range(4), xticklabels=egm.index, yticks=range(egm.shape[1]), yticklabels=[f"module {k}" for k in egm.columns], title="D module eigengene, relative to baseline")
fig.colorbar(im, ax=ax[1, 2])
fig.tight_layout()
fig.savefig(R / "12_reverse.png", dpi=150)
(R / "12_reverse.txt").write_text("\n".join(report))
