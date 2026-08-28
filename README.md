# Nguasach

Cross-linguistic phonosemantics. Two coupled analyses over the same aligned
phonetic↔semantic space:

1. **Confirmatory** — does a word's *phonetic form alone* predict its correct
   translation above a label-permutation null, across language pairs? (transvec /
   ridge alignment + top-*k* retrieval, randomized *k*-fold CV, bootstrap CIs.)
2. **Interpretability** — which phonemes are over-represented in words whose
   meaning sits near each of 18 semantic poles? (per-pole phoneme z-scores with a
   permutation null and FDR correction — the "hexagram" analysis.)

Headline claims are tested only on the four manually verified language columns
(**English, Chinese, French, Irish**); all 22 languages are reported as a
clearly-labelled exploratory extension with translation-QC flags.

## Status

Rebuild in progress. See `../.claude/plans/lively-juggling-mist.md` for the full
plan. Phase 0 (salvage / restructure / pin) is landing now; Phases 1–3 (pipeline,
statistics, paper) follow.

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
