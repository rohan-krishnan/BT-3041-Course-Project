"""Step 1b: build the mouse exercise dataset from ARCHS4 (uniformly processed counts, one gene ID system).
Usage: pip install h5py s3fs ; python 01b_mouse.py <GEPREP mouse metadata file> [local ARCHS4 .h5 file]
Selects wild-type/control skeletal-muscle studies with sedentary and exercise samples, fetches the gene counts of those
GSM ids from the ARCHS4 mouse file (read remotely, or from a local copy of the file) and reports which samples ARCHS4 does not hold.
Saves results/01b_counts.tsv.gz (gene symbols x samples), results/01b_samples.tsv and 01b_mouse.txt."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import s3fs

TISSUES = ["skeletal muscle"]  # add e.g. "heart", "brain" to widen the dataset
CONTROL = {"wild-type", "B6 control", "control", "Cre-Control", "NSG"}
EXCLUDE = {"GSE179394"}  # Ldlr-/- background, all samples are knockouts

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
report = []


def say(text=""):
    print(text)
    report.append(str(text))


meta = pd.read_csv(sys.argv[1], sep="\t", dtype=str, keep_default_na=False)
meta["mus genotype"] = meta["mus genotype"].replace("", "wild-type")  # blank genotype = unmodified animals
m = meta[meta.tissue.isin(TISSUES) & meta["mus genotype"].isin(CONTROL) & ~meta.datasets.isin(EXCLUDE)]
types = m.groupby(["datasets", "tissue"]).classification.agg(set)
keys = [k for k, t in types.items() if "inactivity" in t and t & {"acute exercise", "longterm training"}]
sel = pd.concat([m[(m.datasets == g) & (m.tissue == t)] for g, t in keys]).set_index("GSM")
say(f"selected {sel.datasets.nunique()} studies, {len(sel)} samples")

def text(x):
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in x])


state = {}  # kept between reconnects so a dropped connection resumes where it stopped


def read(f):
    if "idx" not in state:
        state["gsm"] = text(f["meta/samples/geo_accession"][:])
        gkey = next(k for k in ["meta/genes/gene_symbol", "meta/genes/symbol", "meta/gene_symbol"] if k in f)
        state["genes"] = text(f[gkey][:])
        state["idx"] = sorted(i for i, g in enumerate(state["gsm"]) if g in sel.index)
        state["cols"] = {}
        say(f"ARCHS4 file: {len(state['gsm'])} samples, {len(state['genes'])} genes; reading {len(state['idx'])} columns, this can take a long time")
    exp = f["data/expression"]
    for i in state["idx"]:  # one sample at a time, more reliable than fancy indexing on a remote file
        if i not in state["cols"]:
            state["cols"][i] = exp[:, i] if exp.shape[1] == len(state["gsm"]) else exp[i, :]
            if len(state["cols"]) % 20 == 0:
                print(f"  read {len(state['cols'])} of {len(state['idx'])} samples", flush=True)


if len(sys.argv) > 2:
    with h5py.File(sys.argv[2], "r") as f:
        read(f)
else:
    cfg = {"read_timeout": 300, "connect_timeout": 60, "retries": {"max_attempts": 3}}
    sources = [(s3fs.S3FileSystem(anon=True, config_kwargs=cfg, client_kwargs={"endpoint_url": "https://s3.dev.maayanlab.cloud"}), "archs4/files/mouse_gene_v2.latest.h5"),
               (s3fs.S3FileSystem(anon=True, config_kwargs=cfg), "mssm-data/mouse_gene_v2.latest.h5")]
    for attempt in range(40):
        fs, path = sources[(attempt // 3) % 2]  # three tries per server before switching
        try:
            with h5py.File(fs.open(path, "rb", cache_type="blockcache", block_size=4 * 2**20), "r") as f:
                read(f)
            break
        except Exception as e:
            say(f"attempt {attempt + 1} on {path} stopped after {len(state.get('cols', {}))} samples: {type(e).__name__}")
    else:
        sys.exit("ARCHS4 could not be read remotely. Tell me and we will decide whether to keep the mouse arm.")
gsm, genes, idx = state["gsm"], state["genes"], state["idx"]
block = np.column_stack([state["cols"][i] for i in idx])
counts = pd.DataFrame(block, index=genes, columns=gsm[idx]).astype("int64")
counts = counts.loc[:, ~counts.columns.duplicated()]
say(f"ARCHS4 holds {counts.shape[1]} of {len(sel)} samples")

info = sel.loc[counts.columns]
log = pd.DataFrame({"in_metadata": sel.groupby("datasets").size(), "in_ARCHS4": info.groupby("datasets").size()}).fillna(0).astype(int)
say("\n" + log.to_string())
counts.to_csv(RESULTS / "01b_counts.tsv.gz", sep="\t")
info.to_csv(RESULTS / "01b_samples.tsv", sep="\t")
say(f"\nmerged matrix: {counts.shape[0]} genes x {counts.shape[1]} samples from {info.datasets.nunique()} studies")
say("samples per class:\n" + info.classification.value_counts().to_string())
(ROOT / "01b_mouse.txt").write_text("\n".join(report), encoding="utf-8")