---
title: "Measuring form–meaning systematicity across languages with phonetic-similarity embeddings"
author: "Tony Zhou"
status: draft — Results/Discussion pending the confirmatory + exploratory runs
---

## 1. Introduction

Linguistic form is not wholly arbitrary. Beyond a handful of iconic pockets
(ideophones, size-sound symbolism, the *bouba/kiki* effect), corpus-scale work
has found a small but reliable **systematic** relationship between how words
sound and what they mean — within languages [@monaghan2014; @dautriche2017;
@pimentel2019] and, more contentiously, in cross-linguistic sound–meaning
biases shared across unrelated languages [@blasi2016; @joo2020]. Deep-learning
studies have since asked whether the mapping is rich enough for a model to
*translate* from phonology alone [@devarda2022].

This project (an amateur effort begun in 2021, here rebuilt for rigour and
reproducibility) takes the translation-retrieval framing seriously and pairs it
with the distance-matrix (Mantel) analysis standard in the iconicity
literature. For each of 22 languages we hold ~1,840 concepts, each translated
and transcribed to IPA, and embed every word with Parrish's
phonetic-similarity vectors (articulatory feature-bigram counts → whitened PCA)
[@parrish2017]. We then ask two questions:

1. **Cross-lingual (retrieval).** Fitting a regularized linear map between two
   languages' phonetic spaces on 90 % of translation pairs, can we retrieve the
   correct translation of a held-out word *from its sound alone*, above a
   label-permutation null?
2. **Form ↔ meaning (Mantel).** Do words that sound similar have meanings that
   are similar — within a language, and after residualizing out orthographic
   (edit-distance) similarity, which absorbs cognate and borrowing overlap?

Headline claims are restricted to four **manually verified** language columns
(English, Chinese, French, Irish); the remaining 18 columns, whose
machine translations we audit but do not trust cell-by-cell, are reported as a
labelled exploratory extension.

## 2. Method

### 2.1 Concepts and translations

The concept list is 1,842 entries (function words, a large verb set, concrete
and abstract nouns), stored as `nguasach.xlsx`. English, Chinese, French and
Irish were checked by hand; the other 18 languages were produced by Google
Translate (2021–2023) and are treated as frozen raw data. An offline
quality-control pass (`translate-qc`) flags cells that are (a) identical to the
English prompt in a language where that is implausible as a cognate,
(b) pure-ASCII in a non-Latin-script column, or (c) copied across ≥4 columns.
Flag rates: Swahili 14.6 %, Hungarian 8.5 %, Indonesian 5.7 %, Welsh 5.2 %,
others < 5 %. Flagged cells are dropped from the exploratory analysis
(`qc_mode: exclude_flagged`); the verified core is never dropped.

Chinese homographs collapse to a single semantic key (~2 % of concepts); this,
and the residual translation noise, are recorded in the data statement
(Appendix A).

### 2.2 Phonetic transcription and embeddings

IPA comes from eSpeak NG 1.52 (via `phonemizer` + `espeakng-loader`, a pinned
wheel — no system install) for 21 languages, and from a hand-built
pinyin→IPA map for Mandarin (retroflex series, aspiration, apical vowels;
after the original project). Suprasegmentals (stress, length, tone) and
diacritics the feature model does not represent are stripped, matching the
input format Parrish's `generate.py` expects. Each word becomes a bag of
articulatory feature bigrams, row-normalized, then reduced to `d = 300`
dimensions by whitened PCA. Semantic vectors are word2vec (`model.txt`,
26,660-word vocabulary) reduced to 50 whitened PCA dimensions.

### 2.3 Cross-lingual retrieval

For an ordered pair (source → target) we fit a ridge map
`W = (Xᵀ_src X_src + αI)⁻¹ Xᵀ_src X_tgt` (α = 1) on the training folds'
translation pairs and score held-out source words by cosine retrieval of the
predicted target vector, with **CSLS** de-hubbing [@conneau2018]. A word counts
as retrieved if its gold translation is within the top *k* = 100 of 1,842.
Cross-validation is 10-fold with **randomized** (seeded) partitions — the
original used a single contiguous 80/20 split over a semantically ordered
list. We report the mean accuracy, its 95 % bootstrap CI (2,000 resamples of
test concepts) and the Nadeau–Bengio variance-corrected SE.

**Null.** For 1,000 permutations we shuffle the source↔target concept pairing
and re-run the entire fit-and-retrieve, giving an empirical
p = (1 + #{null ≥ observed}) / 1,001. Benjamini–Hochberg FDR is applied
separately to the 12 confirmatory ordered pairs and to the exploratory matrix.
A homograph guard flags test items whose gold target string also occurs in
training; we report accuracy with and without them.

### 2.4 Distance-matrix (Mantel) analysis

On a 700-concept subsample we build the pairwise phonetic-distance matrix
`D_form` (1 − cosine on the phonetic embeddings), the meaning-distance matrix
`D_mean` (1 − cosine on the semantic vectors) and the orthographic matrix
`D_orth` (1 − normalized Levenshtein on surface forms). The Mantel statistic is
the Pearson correlation of the upper triangles; the **partial** Mantel
[@smouse1986] residualizes `D_form` and `D_mean` on `D_orth` before
correlating. Significance is a 1,000-permutation null over concept labels. We
run this within each language (`D_form(L)` vs `D_mean`) and between each pair of
verified-core languages (`D_form(L1)` vs `D_form(L2)`).

### 2.5 Baselines

Every baseline runs through the identical CV + null + FDR machinery.

* **`editdist`** — non-learned retrieval by normalized Levenshtein similarity of
  surface strings. The cognate/borrowing control.
* **`orth`** — character n-gram (2, 3) count vectors → PCA → ridge map. The
  orthographic analogue of the phonetic pipeline.
* **`feat`** — mean `panphon` articulatory-feature vector per word → PCA → ridge
  map. A coarser phonological representation.

### 2.6 Phoneme–meaning association

Eighteen semantic "poles" (hand-built English seed-word clusters, from the
original I-Ching–themed design; the framing is cosmetic, the poles are just
regions of semantic space). Each concept is assigned to its nearest pole in the
phonetic→semantic-aligned space; per pole we z-score every phoneme's frequency
against its distribution across poles, with a 1,000-permutation null over the
concept↔pole assignment and BH-FDR across all phoneme × pole cells.

### 2.7 Reproducibility

One command — `nguasach all --config configs/paper_confirmatory.yaml` — runs
`data → translate-qc → ipa → phonetics → semantics → align → mantel →
associate → baselines → report`. Every stage writes a JSON run manifest (config
hash, git SHA, input checksums, backend versions). `notebooks/figures.py`
regenerates every figure from `results/`. Frozen configs, a pinned environment
(`pyproject.toml` / `environment.yml`, Python 3.10), and the vendored
`phonetic-similarity-vectors` code (commit `58f5639`) are in the repository.

## 3. Results

*(pending the confirmatory and exploratory runs — see `results/summary.md`,
`results/mantel.csv`, `results/baselines.csv`)*

### 3.1 Cross-lingual retrieval

### 3.2 Phonetic vs. control representations

### 3.3 Form–meaning correlations

### 3.4 Phoneme–meaning associations

### 3.5 Exploratory: all 22 languages

## 4. Discussion

## 5. Limitations

- Google-Translate provenance and residual noise in 18 of 22 columns; QC is
  heuristic, not cell-verified.
- One written word per concept: no senses, no frequency weighting, no
  morphology.
- eSpeak coverage and consistency vary by language; the Mandarin transcription
  uses a different (hand-built) system than the rest.
- The ridge map has ~n/d ≈ 5.5 training pairs per parameter; regularization and
  the permutation null mitigate but do not eliminate overfitting.
- The semantic space is a 26k-word word2vec model, itself English-anchored.

## Appendix A. Data statement

Per-language translation source and date, IPA backend and version, QC flag
counts, and known failure cases are in `data/README.md`,
`data/interim/translation_qc_report.json`, and `data/INVENTORY.md`.

## References

Bibliography in `manuscript/references.bib`.
