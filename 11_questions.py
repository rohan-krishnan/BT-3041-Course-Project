"""Step 11: six extra questions in one script.  Usage: python 11_questions.py
Reads database2_detailed.csv (log2 CPM) and writes results/11_questions.txt, 11_questions.png and three csv tables.
Expression is shown relative to the baseline (no exercise) samples of the same study, or of the same person where paired.
  Q1 resting fitness      baseline samples, trained vs untrained (GSE120862) and active vs sedentary (GSE151066)
  Q2 stopwatch genes      Spearman of each gene with hours after exercise; ridge regression with leave-one-study-out
  Q3 who responds         immediate-early score per paired person, spread and consistency between time points
  Q4 fibre type           MYH7, MYH2, MYH1 (and MYH4 in mouse) before and after training
  Q5 body clock           circadian genes, paired change at each time point
  Q6 ageing               aged against young immediate-early score at matched time points"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr, t as tdist, ttest_1samp, ttest_ind, wilcoxon
from sklearn.linear_model import Ridge

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
R = ROOT / "results"
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


db = pd.read_csv(ROOT / "database2_detailed.csv", index_col=0, low_memory=False)
last = list(db.columns).index("library_size_millions") + 1
meta, X = db.iloc[:, :last].copy(), db.iloc[:, last:]
meta["hours"] = pd.to_numeric(meta.hours_after_exercise, errors="coerce")
meta["age"] = pd.to_numeric(meta.exact_age, errors="coerce")
human, mouse = (meta.species == "human"), (meta.species == "mouse")
none, acute, train = (meta.condition == "no exercise"), (meta.condition == "acute exercise"), (meta.condition == "long-term training")
base = X[none].groupby(meta.study[none]).mean()  # baseline mean of each study
rel = X - base.reindex(meta.study).values  # expression relative to the study baseline
pbase = X[none].groupby(meta.person[none]).mean()  # baseline mean of each person
IEG = [g for g in ["NR4A3", "FOS", "JUN", "JUNB", "EGR1", "ATF3", "IER2", "MAFF", "SIK1", "MYC", "NR4A1"] if g in X.columns]


def paired(mask, genes):
    """change of each sample in mask relative to the same person's baseline (samples without a baseline are dropped)"""
    idx = meta.index[mask & meta.person.isin(pbase.index)]
    return X.loc[idx, genes] - pbase.loc[meta.person[idx], genes].values


# Q1 resting fitness -------------------------------------------------------------------------------------------------
say("Q1. Resting fitness: baseline samples only, compared inside each study")
expr = X.columns[X.mean() > 3]
Xc = X.sub(X[expr].median(axis=1), axis=0)  # each sample minus its median over expressed genes (guards against a shift in muscle content)
fit = {}
for study, (hi, lo) in {"GSE120862": ("trained", "untrained"), "GSE151066": ("active", "sedentary")}.items():
    ia, ib = X.index[none & (meta.study == study) & (meta.fitness == hi)], X.index[none & (meta.study == study) & (meta.fitness == lo)]
    for label, M in [("raw", X), ("centred", Xc)]:
        a, b = M.loc[ia], M.loc[ib]
        d = pd.DataFrame({"log2FC": a.mean() - b.mean(), "p_t": ttest_ind(a, b, equal_var=False).pvalue, "p_rank": mannwhitneyu(a, b, axis=0).pvalue})
        d["q_t"], d["q_rank"] = bh(d.p_t), bh(d.p_rank)
        fit[study, label] = d
        say(f"  {study} {label}: {hi} (n={len(a)}) vs {lo} (n={len(b)}); Welch q<0.05: {(d.q_t < 0.05).sum()} (up {((d.q_t < 0.05) & (d.log2FC > 0)).sum()}), "
            f"Mann-Whitney q<0.05: {(d.q_rank < 0.05).sum()}; median change over all genes {d.log2FC.median():.2f}, genes up {(d.log2FC > 0).mean():.0%}")
A, B = fit["GSE120862", "centred"], fit["GSE151066", "centred"]
sa, sb = A.p_t < 0.05, B.p_t < 0.05
tab = pd.crosstab(sa, sb).reindex(index=[False, True], columns=[False, True], fill_value=0)
same = (np.sign(A.log2FC) == np.sign(B.log2FC))[sa & sb].mean()
say(f"  after centring, genes with p<0.05 in both studies: {(sa & sb).sum()} (expected by chance {sa.sum() * sb.sum() / len(A):.0f}), same direction in {same:.0%}; "
    f"Fisher odds ratio {fisher_exact(tab)[0]:.2f}, p = {fisher_exact(tab)[1]:.2g}")
top = A[A.q_t < 0.05].sort_values("p_t").head(10)
say("  top genes after centring, trained vs untrained (GSE120862):\n" + top[["log2FC", "p_t", "q_t"]].to_string(float_format=lambda v: f"{v:.3g}"))
pd.concat({f"{k[0]} {k[1]}": v for k, v in fit.items()}, axis=1).round(5).to_csv(R / "11_fitness_genes.csv")

# Q2 stopwatch genes -------------------------------------------------------------------------------------------------
say("\nQ2. Stopwatch genes: human acute samples with a known time after exercise")
m = human & acute & meta.hours.notna()
Z, y, st = rel[m], np.log1p(meta.hours[m]), meta.study[m]
n = len(y)
rho = Z.rank().corrwith(y.rank())
tt = rho * np.sqrt((n - 2) / (1 - rho**2))
sw = pd.DataFrame({"rho": rho, "p": 2 * tdist.sf(np.abs(tt), n - 2)})
sw["q"] = bh(sw.p)
sw.round(5).to_csv(R / "11_stopwatch_genes.csv")
say(f"  {n} samples from {st.nunique()} studies; genes with q<0.05 against log(hours): {(sw.q < 0.05).sum()}")
say("  strongest genes:\n" + sw.reindex(sw.rho.abs().sort_values(ascending=False).index).head(8).to_string(float_format=lambda v: f"{v:.3g}"))
F = Z[Z.var().nlargest(2000).index]
F = (F - F.mean()) / F.std()


def loso(target):
    pred = pd.Series(0.0, index=target.index)
    for s in st.unique():
        te = (st == s).values
        pred[te] = Ridge(alpha=1000).fit(F[~te], target[~te]).predict(F[te])
    return pred


pred = loso(y)
obs = spearmanr(pred, y)[0]
rng = np.random.default_rng(1)
null = []
for _ in range(100):  # hours shuffled inside each study
    null.append(spearmanr(loso(y.groupby(st).transform(lambda v: rng.permutation(v.values))), y)[0])
say(f"  ridge regression, each study predicted from the others: Spearman(predicted, true hours) = {obs:.2f}, "
    f"shuffled-label p = {(1 + sum(v >= obs for v in null)) / 101:.3f}, median error {np.median(np.abs(np.expm1(pred) - meta.hours[m])):.1f} h")
for s in st.unique():
    k = (st == s).values
    if y[k].nunique() > 1:
        say(f"    {s}: n={k.sum()}, {[int(v) for v in sorted(meta.hours[m][k].unique())]} h, Spearman {spearmanr(pred[k], y[k])[0]:.2f}")

# Q3 who responds ----------------------------------------------------------------------------------------------------
say("\nQ3. Who responds: immediate-early score = mean change of " + ", ".join(IEG) + " against the person's own baseline")
D = paired(human & acute, IEG)
sc = pd.DataFrame({"score": D.mean(axis=1), "study": meta.study[D.index], "person": meta.person[D.index], "hours": meta.hours[D.index],
                   "age_group": meta.age_group[D.index], "age": meta.age[D.index], "type": meta.exercise_type[D.index]})
sc["group"] = sc.study + " " + sc.hours.astype(int).astype(str) + "h"
sc.round(4).to_csv(R / "11_response_scores.csv")
g = sc.groupby("group").score
tb = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sd": g.std(), "min": g.min(), "max": g.max(), "type": sc.groupby("group").type.agg(lambda v: v.mode().iat[0])})
say(f"  {sc.person.nunique()} paired people, {len(sc)} person-time points\n" + tb.round(2).to_string())
ssb = (g.transform("mean") - sc.score.mean()).pow(2).sum()
sst = (sc.score - sc.score.mean()).pow(2).sum()
k = sc.group.nunique()
adj = 1 - (1 - ssb / sst) * (len(sc) - 1) / (len(sc) - k)
say(f"  study and time point explain {adj:.0%} of the variance in the score (adjusted R2); the rest is variation between people")
sc["resid"] = sc.score - g.transform("mean")
ka = sc.age.notna()
if ka.sum() > 5:
    r, p = spearmanr(sc.resid[ka], sc.age[ka])
    say(f"  residual score against exact age ({ka.sum()} points, ages {sc.age[ka].min():.0f} to {sc.age[ka].max():.0f}): Spearman {r:.2f}, p = {p:.2g}")
wide = sc.pivot_table(index="person", columns=["study", "hours"], values="score")
for study in wide.columns.levels[0]:
    w = wide[study].dropna(thresh=2) if study in wide.columns.get_level_values(0) else None
    if w is not None and w.shape[1] > 1 and len(w) >= 8:
        cols = w.columns[:2]
        w = w[cols].dropna()
        r, p = spearmanr(w.iloc[:, 0], w.iloc[:, 1])
        say(f"  {study}: score at {cols[0]:.0f} h against {cols[1]:.0f} h in the same people (n={len(w)}): Spearman {r:.2f}, p = {p:.2g}")

# Q4 fibre type ------------------------------------------------------------------------------------------------------
say("\nQ4. Fibre type: myosin heavy chain genes after training")
fig_share = {}
for sp, sel, genes in [("human", human, ["MYH7", "MYH2", "MYH1"]), ("mouse", mouse, ["MYH7", "MYH2", "MYH1", "MYH4"])]:
    genes = [x for x in genes if x in X.columns]
    cpm = 2 ** X[genes] - 1
    share = cpm.div(cpm.sum(axis=1), axis=0)
    sbase = share[none].groupby(meta.study[none]).mean()
    srel = share - sbase.reindex(meta.study).values
    a, b = sel & train, sel & none
    rows = []
    for gname in genes:
        rows.append({"gene": gname, "log2 change": rel.loc[a, gname].mean() - rel.loc[b, gname].mean(),
                     "p_welch": ttest_ind(rel.loc[a, gname], rel.loc[b, gname], equal_var=False).pvalue,
                     "p_rank": mannwhitneyu(rel.loc[a, gname], rel.loc[b, gname]).pvalue,
                     "share before": share.loc[b, gname].mean(), "share change": srel.loc[a, gname].mean() - srel.loc[b, gname].mean()})
    say(f"  {sp}: {a.sum()} training vs {b.sum()} no-exercise samples (study means removed)\n" + pd.DataFrame(rows).set_index("gene").round(4).to_string())
    if sp == "human":
        Dt = paired(train & human, genes)
        pshare = (2 ** pbase[genes] - 1)
        pshare = pshare.div(pshare.sum(axis=1), axis=0)
        Sd = share.loc[Dt.index] - pshare.loc[meta.person[Dt.index]].values
        say(f"  human, {Dt.index.map(meta.person).nunique()} people measured before and after training (paired)")
        pr = pd.DataFrame({"log2 change": Dt.mean(), "p_paired_t": ttest_1samp(Dt, 0).pvalue, "p_wilcoxon": [wilcoxon(Dt[c]).pvalue for c in genes],
                           "share change": Sd.mean()})
        say(pr.round(4).to_string())
        fig_share = {"before": pshare.loc[meta.person[Dt.index]].mean(), "after": share.loc[Dt.index].mean()}

# Q5 body clock ------------------------------------------------------------------------------------------------------
CIRC = [x for x in ["NR1D1", "NR1D2", "PER1", "PER2", "PER3", "CRY1", "CRY2", "CLOCK", "DBP", "TEF", "HLF", "NPAS2", "NFIL3", "RORA", "CIART"] if x in X.columns]
say("\nQ5. Body clock: median paired change in log2 CPM after one exercise bout (* = BH q<0.05 over the table)")
say("  BMAL1/ARNTL is not in the shared gene list (human and mouse annotations use different symbols for it)")
Dc = paired(human & acute, CIRC)
hrs = [h for h in sorted(meta.hours[Dc.index].dropna().unique()) if (meta.hours[Dc.index] == h).sum() >= 10]
med, pv = pd.DataFrame(index=CIRC, columns=hrs, dtype=float), pd.DataFrame(index=CIRC, columns=hrs, dtype=float)
for h in hrs:
    sub = Dc[meta.hours[Dc.index] == h]
    med[h] = sub.median()
    pv[h] = [wilcoxon(sub[c]).pvalue for c in CIRC]
qv = pd.DataFrame(bh(pv.values.ravel()).reshape(pv.shape), index=CIRC, columns=hrs)
shown = med.round(2).astype(str) + np.where(qv < 0.05, "*", " ")
shown.columns = [f"{int(h)}h (n={(meta.hours[Dc.index] == h).sum()})" for h in hrs]
say(shown.to_string())
say("  the clock time of each biopsy is not in the metadata, so a drift in time of day between baseline and later biopsies cannot be separated from an exercise effect")

# Q6 ageing ----------------------------------------------------------------------------------------------------------
say("\nQ6. Ageing: immediate-early score, aged against young people at the same time after exercise")
for h in [0, 3]:
    ag, yg = sc[(sc.hours == h) & (sc.age_group == "aged")], sc[(sc.hours == h) & (sc.age_group == "young")]
    if len(ag) > 2 and len(yg) > 2:
        say(f"  {h} h: aged n={len(ag)} ({', '.join(ag.study.unique())}) median {ag.score.median():.2f}; young n={len(yg)} ({', '.join(yg.study.unique())}) "
            f"median {yg.score.median():.2f}; Welch p = {ttest_ind(ag.score, yg.score, equal_var=False).pvalue:.2g}, Mann-Whitney p = {mannwhitneyu(ag.score, yg.score).pvalue:.2g}")
say("  all aged people come from one study, so age and study cannot be separated; treat this as a descriptive comparison")

# figure -------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
for s in st.unique():
    k = (st == s).values
    ax[0, 0].scatter(meta.hours[m][k], np.expm1(pred[k]), s=14, alpha=0.6, label=s)
ax[0, 0].plot([0, 24], [0, 24], "k--", lw=1)
ax[0, 0].set(xlabel="true hours after exercise", ylabel="predicted hours (study held out)", title="Q2 stopwatch genes")
ax[0, 0].legend(fontsize=6)
order = sc.groupby("group").score.median().sort_values().index
for i, gname in enumerate(order):
    s_ = sc[sc.group == gname]
    ax[0, 1].scatter(np.full(len(s_), i) + np.random.default_rng(0).uniform(-0.2, 0.2, len(s_)), s_.score, s=12, c=np.where(s_.age_group == "aged", "tab:red", "tab:blue"))
ax[0, 1].set(xticks=range(len(order)), xticklabels=order, ylabel="immediate-early score (log2 change)", title="Q3 and Q6 response per person (red = aged)")
ax[0, 1].tick_params(axis="x", rotation=90, labelsize=6)
if fig_share:
    xs = np.arange(len(fig_share["before"]))
    ax[1, 0].bar(xs - 0.2, fig_share["before"], 0.4, label="before")
    ax[1, 0].bar(xs + 0.2, fig_share["after"], 0.4, label="after training")
    ax[1, 0].set(xticks=xs, xticklabels=fig_share["before"].index, ylabel="share of MYH7+MYH2+MYH1 reads", title="Q4 fibre type (paired people)")
    ax[1, 0].legend()
im = ax[1, 1].imshow(med.values.astype(float), cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
ax[1, 1].set(xticks=range(len(hrs)), xticklabels=[f"{int(h)}h" for h in hrs], yticks=range(len(CIRC)), yticklabels=CIRC, title="Q5 circadian genes, median log2 change")
fig.colorbar(im, ax=ax[1, 1])
fig.tight_layout()
fig.savefig(R / "11_questions.png", dpi=150)
(R / "11_questions.txt").write_text("\n".join(report))
