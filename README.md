# Nguasach

Cross-linguistic phonosemantics: is a word's sound predictive of its meaning,
and is any of that shared across languages? A rebuild of a 2021 amateur project
for rigour and reproducibility.

Three analyses over the same phonetic / semantic representations:

1. **Retrieval** — fit a ridge map between two languages' phonetic-similarity
   spaces on 90 % of translation pairs; can it retrieve a held-out word's
   translation from sound alone, above a label-permutation null? (10-fold CV,
   CSLS, bootstrap CIs, BH-FDR.)
2. **Mantel** — correlate the pairwise form-distance matrix with the
   meaning-distance matrix, within and across languages, partialling out
   orthographic (edit-distance) similarity. The field-standard test, immune to
   the retrieval metric's hubness.
3. **Phoneme–meaning association** — which phonemes cluster with which of 18
   semantic poles (the original "hexagram" analysis).

Headline claims use the four manually verified columns (**English, Chinese,
French, Irish**); all 22 languages are a labelled exploratory extension with
translation-QC flags.

## Status & results

Pipeline complete; both frozen runs done.

| run | config | frozen output | one-line result |
|-----|--------|---------------|-----------------|
| confirmatory | `configs/paper_confirmatory.yaml` | `results/paper_confirmatory/`, `figures/` | within-language `form~meaning` *r* ≈ 0.02 survives the orthographic control in all 4; related-language retrieval is beaten by a plain edit-distance ranker (cognates). |
| exploratory | `configs/paper_exploratory.yaml` | `results/paper_exploratory/`, `figures/exploratory/` | `form~meaning` significant in **all 22 languages** (partial *r* holds wherever the orthographic control is meaningful); cross-family `form~form` small (median *r* ≈ 0.01) but present for ~80 % of pairs. |

Write-up: `manuscript/manuscript.md` (+ `references.bib`, pandoc `Makefile`).
Original scripts and outputs are in `legacy_src/` and `legacy_results/`.

## Setup

Plain venv (Python 3.10–3.12):

```bash
py -3.12 -m venv .venv           # or: python3.12 -m venv .venv
.venv/Scripts/python -m pip install -e .   # Linux/mac: .venv/bin/python
```

Or conda, if you have it:

```bash
conda env create -f environment.yml && conda activate nguasach && pip install -e .
```

The `ipa` stage needs the **espeak-ng** binary on PATH (conda installs it; on
Windows otherwise use `scoop install espeak-ng` or the official installer). Every
other stage is pure Python.

Heavy deps (`torch`, `gensim`) are only needed from the `align` stage on; the
`data` / `semantics` / `ipa` / `phonetics` stages need just
`numpy pandas openpyxl pyyaml scikit-learn`.

Large inputs (`model.txt`, `cc.en.300.vec`, …) are gitignored — see
[`data/README.md`](data/README.md) for checksums and how to obtain them.

## Pipeline

```
nguasach run <stage> --config configs/default.yaml     # one stage
nguasach all          --config configs/smoke.yaml       # whole DAG
```

Stage DAG:

```
data ──▶ translate-qc ──▶ ipa ──▶ phonetics ──┐
                                              ├──▶ align ──▶ associate ──▶ report
                        semantics ────────────┘
```

| stage | module | output |
|-------|--------|--------|
| `data` | `data.py` | `data/interim/nguasach.utf8.csv` (canonical, integrity-checked) + folds |
| `translate-qc` | `translate_qc.py` | `data/interim/translation_qc.csv` (per-cell flags) |
| `ipa` | `ipa.py` | `data/interim/<Lang>V.txt` (PSV input format) |
| `phonetics` | `phonetics.py` → vendored `generate.py` | `data/processed/<Lang>Emb.txt` |
| `semantics` | `semantics.py` | `data/processed/SemanticsEmb.txt` |
| `align` | `align.py`, `crossval.py`, `nulls.py`, `baselines.py` | `results/accuracy_by_pair.csv`, `results/null_distributions.parquet` |
| `associate` | `association.py` | `results/association_z.csv` |
| `report` | `report.py` | `figures/*`, tables for `paper/` |

Configs: `default` (dev, all 22), `smoke` (CI, 5 langs / 150 concepts),
`paper_confirmatory` (frozen, verified-core), `paper_exploratory` (frozen, all 22).

## Layout

```
src/nguasach/     pipeline package
configs/          YAML run configs
third_party/phonetic-similarity-vectors/   vendored PSV code (see COMMIT.txt)
data/raw/         immutable inputs
data/{interim,processed}/   generated (gitignored)
results/ figures/            generated (gitignored)
notebooks/        01 walkthrough, 02 figures (regenerates every manuscript figure)
paper/            manuscript + refs
tests/            integrity + smoke + no-leakage + null-is-chance
attic/            quarantined exploratory offshoots
```

## Provenance of the method

`third_party/phonetic-similarity-vectors/` is a vendored copy of a fork of
Allison Parrish's [phonetic-similarity-vectors](https://github.com/aparrish/phonetic-similarity-vectors)
(feature-bigram counts → whitened PCA). See its `COMMIT.txt` and `LICENSE`.
