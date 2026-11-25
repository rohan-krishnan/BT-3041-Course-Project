"""Step 3: one combined dataset with human and mouse samples.
Needs the outputs of step 2 for both species (results/02_* and results/02b_*) and geo/Human.GRCh38.p13.annot.tsv.gz.
Genes are matched between species by gene symbol (human symbols in upper case equal mouse symbols in upper case).
Saves results/03_all.tsv.gz: one row per sample, sample description columns first, then log2(CPM+1) of the shared genes.
Also saves results/03_meta_columns.txt (names of the description columns) and 03_combine.txt."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
R = ROOT / "results"
report = []


def say(text=""):
    print(text)
    report.append(str(text))


KEEP = ["species", "datasets", "tissue", "sampling site", "extracted molecule", "layout", "library prepared method", "platform ID",
        "mus genotype", "group", "classification", "exercise type", "protocol", "duration", "frequency", "intensity",
        "biopsy timepoint", "hours_after", "gender", "age", "exact age", "fitness", "exercise response", "subject id(or sample id)", "lib_size_M"]

annot = pd.read_csv(ROOT / "geo" / "Human.GRCh38.p13.annot.tsv.gz", sep="\t", dtype=str)
cols = {c.lower(): c for c in annot.columns}
if "geneid" not in cols or "symbol" not in cols:
    raise SystemExit(f"annotation columns are {list(annot.columns)}; tell me the names of the gene ID and symbol columns")
sym = annot.set_index(cols["geneid"])[cols["symbol"]].str.upper()

parts, meta = {}, []
for species, tag in [("human", "02"), ("mouse", "02b")]:
    x = pd.read_csv(R / f"{tag}_logcpm.tsv.gz", sep="\t", index_col=0)  # genes x samples
    s = pd.read_csv(R / f"{tag}_samples.tsv", sep="\t", index_col=0, dtype=str, keep_default_na=False).loc[x.columns]
    s["species"] = species
    if species == "human":
        x.index = x.index.astype(str).map(sym)
        x = x[x.index.notna()]
    else:
        x.index = x.index.str.upper()
    x = x.loc[x.mean(axis=1).sort_values(ascending=False).index]  # if two rows share a symbol keep the most expressed
    x = x[~x.index.duplicated()]
    say(f"{species}: {x.shape[1]} samples, {x.shape[0]} genes with a symbol")
    parts[species], meta = x, meta + [s]

shared = parts["human"].index.intersection(parts["mouse"].index)
say(f"genes present in both species: {len(shared)}")
expr = pd.concat([parts["human"].loc[shared], parts["mouse"].loc[shared]], axis=1).T
meta = pd.concat(meta)
meta = meta.reindex(columns=[c for c in KEEP if c in meta.columns])
out = meta.join(expr)
out.index.name = "GSM"
out.to_csv(R / "03_all.tsv.gz", sep="\t")
(R / "03_meta_columns.txt").write_text("\n".join(meta.columns), encoding="utf-8")
say(f"\nsaved results/03_all.tsv.gz: {out.shape[0]} samples x {meta.shape[1]} description columns + {len(shared)} genes")
say("samples per species and class:\n" + out.groupby(["species", "classification"]).size().to_string())
(ROOT / "03_combine.txt").write_text("\n".join(report), encoding="utf-8")
