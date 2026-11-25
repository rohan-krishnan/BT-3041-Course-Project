"""Step 1: build the human exercise dataset.
Usage: python 01_download.py <GEPREP human metadata file>
Picks healthy skeletal-muscle studies that have baseline and exercise samples, downloads the NCBI-generated raw count
matrix of each study from GEO (same genome annotation for every study), keeps the selected samples and merges them.
Saves results/01_counts.tsv.gz (genes x samples), results/01_samples.tsv and 01_download.txt."""
import gzip
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).parent
GEO, RESULTS = ROOT / "geo", ROOT / "results"
GEO.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
report = []


def say(text=""):
    print(text)
    report.append(str(text))


reasons = []


def fetch(url, path):
    """Download url to path. Returns True only if the response is a gzip file; otherwise stores the reason."""
    for _ in range(2):
        try:
            data = urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=120).read()
            if data[:2] == b"\x1f\x8b":
                path.write_bytes(data)
                return True
            reasons.append(f"got a web page instead of a file (first bytes: {data[:80]!r})")
            return False
        except (URLError, TimeoutError) as e:
            reasons.append(f"{type(e).__name__}: {e}")
            time.sleep(2)
    return False


meta = pd.read_csv(sys.argv[1], sep="\t", dtype=str, keep_default_na=False)
muscle = meta[(meta["health status"] == "healthy") & (meta.tissue == "skeletal muscle")]
types = muscle.groupby("datasets").classification.agg(set)
studies = sorted(g for g, t in types.items() if "inactivity" in t and t & {"acute exercise", "longterm training"})
sel = muscle[muscle.datasets.isin(studies)].set_index("GSM")
say(f"selected {len(studies)} studies, {len(sel)} samples: {', '.join(studies)}")

annot = GEO / "Human.GRCh38.p13.annot.tsv.gz"
if not annot.exists():
    fetch("https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts&file=Human.GRCh38.p13.annot.tsv.gz", annot)

tables, rows = [], []
for gse in studies:
    f = next(iter(sorted(GEO.glob(f"{gse}_raw_counts*"))), GEO / f"{gse}_raw_counts_GRCh38.p13_NCBI.tsv.gz")
    if not f.exists():
        name = quote(f.name, safe="")
        for url in [f"https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc={gse}&format=file&file={name}",
                    f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={gse}&format=file&file={name}"]:
            if fetch(url, f):
                break
    expected = (sel.datasets == gse).sum()
    if not f.exists():
        stub = gse[:-3] + "nnn"
        rows.append((gse, "no NCBI counts", expected, 0))
        say(f"{gse}: no NCBI count file. Look for a counts file by hand at https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{gse}/suppl/")
        continue
    t = pd.read_csv(f, sep="\t", index_col=0, compression="infer")
    keep = [c for c in t.columns if c in sel.index]
    rows.append((gse, "ok", expected, len(keep)))
    tables.append(t[keep])

if reasons:
    say(f"\ndownload problem (first reason seen): {reasons[0]}")
    say("If NCBI blocks scripts, download each study's 'RNA-seq raw counts matrix' in the browser from https://www.ncbi.nlm.nih.gov/geo/download/?acc=<GSE> and put the files in the geo folder, then run this script again.")
log = pd.DataFrame(rows, columns=["study", "status", "samples_in_metadata", "samples_with_counts"])
say("\n" + log.to_string(index=False))
if tables:
    counts = pd.concat(tables, axis=1, join="outer").fillna(0).astype("int64")
    info = sel.loc[counts.columns]
    counts.to_csv(RESULTS / "01_counts.tsv.gz", sep="\t")
    info.to_csv(RESULTS / "01_samples.tsv", sep="\t")
    say(f"\nmerged matrix: {counts.shape[0]} genes x {counts.shape[1]} samples from {info.datasets.nunique()} studies")
    say("samples per class:\n" + info.classification.value_counts().to_string())
(ROOT / "01_download.txt").write_text("\n".join(report), encoding="utf-8")
print("\nsaved 01_download.txt")