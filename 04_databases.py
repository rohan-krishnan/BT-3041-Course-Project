"""Step 6: the two shareable databases (CSV), built from results/03_all.tsv.gz.
database1_overview.csv  one row per sample: species, study, condition, sex, age group, then expression of the shared genes.
database2_detailed.csv  the same rows with every description column (protocol, exercise type, duration, frequency, intensity,
                        biopsy time point, hours after exercise, group label, muscle, fitness, technical columns), then the same genes.
Expression is log2(CPM + 1). Raw counts stay in results/01_counts.tsv.gz (human) and results/01b_counts.tsv.gz (mouse)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
R = ROOT / "results"

src = next(p for p in [R / "03_all.tsv.gz", R / "03_all.tsv"] if p.exists())  # works zipped or unzipped
d = pd.read_csv(src, sep="\t", index_col=0, low_memory=False)
last = list(d.columns).index("lib_size_M") + 1
meta, genes = d.iloc[:, :last].copy(), d.iloc[:, last:].round(2)
for c in meta.columns:  # tidy text: stray spaces, non-breaking spaces
    if not pd.api.types.is_numeric_dtype(meta[c]):
        meta[c] = meta[c].fillna("").astype(str).str.replace("\xa0", " ").str.strip().replace({"nan": "", "NaN": "", "<NA>": ""})

label = {"inactivity": "no exercise", "acute exercise": "acute exercise", "longterm training": "long-term training"}
meta["condition"] = meta.classification.map(label)
meta["exercise"] = meta.condition.map({"no exercise": "no"}).fillna("yes")
meta["person"] = meta.datasets + " | " + meta["subject id(or sample id)"]
meta["sex"] = meta.gender.replace("", "unknown")
meta["hours_after"] = pd.to_numeric(meta.hours_after, errors="coerce")
meta["exact age"] = pd.to_numeric(meta["exact age"], errors="coerce")
meta["lib_size_M"] = pd.to_numeric(meta.lib_size_M, errors="coerce")

meta["_c"] = meta.condition.map({"no exercise": 0, "acute exercise": 1, "long-term training": 2})
order = meta.sort_values(["species", "datasets", "person", "_c", "hours_after"]).index
meta, genes = meta.loc[order], genes.loc[order]

db1 = pd.DataFrame({"species": meta.species, "study": meta.datasets, "condition": meta.condition, "exercise": meta.exercise,
                    "sex": meta.sex, "age_group": meta.age}, index=meta.index)
db2 = pd.DataFrame({"species": meta.species, "study": meta.datasets, "person": meta.person, "condition": meta.condition,
                    "exercise": meta.exercise, "group_label": meta.group, "muscle": meta["sampling site"], "protocol": meta.protocol,
                    "exercise_type": meta["exercise type"], "duration": meta.duration, "frequency": meta.frequency,
                    "intensity": meta.intensity, "biopsy_timepoint": meta["biopsy timepoint"], "hours_after_exercise": meta.hours_after,
                    "sex": meta.sex, "age_group": meta.age, "exact_age": meta["exact age"], "fitness": meta.fitness,
                    "exercise_response": meta["exercise response"], "genotype": meta["mus genotype"], "rna_type": meta["extracted molecule"],
                    "read_layout": meta.layout, "library_prep": meta["library prepared method"], "platform": meta["platform ID"],
                    "library_size_millions": meta.lib_size_M}, index=meta.index)
for name, db in [("database1_overview", db1), ("database2_detailed", db2)]:
    out = db.join(genes)
    out.index.name = "sample"
    out.to_csv(ROOT / f"{name}.csv")
    print(f"{name}.csv: {out.shape[0]} samples, {db.shape[1]} description columns + {genes.shape[1]} genes")
print(db1.groupby(["species", "condition"]).size().to_string())
