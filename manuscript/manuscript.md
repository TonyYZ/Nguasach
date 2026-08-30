---
title: "Measuring form–meaning systematicity across languages with phonetic-similarity embeddings"
author: "Tony Zhou"
status: draft — confirmatory results in; §3.5 exploratory pending its run
---

## 1. Introduction

Linguistic form is not wholly arbitrary. The relationship between sound and
meaning is now usually decomposed into *iconicity* (a resemblance between form
and referent) and *systematicity* (a statistical, not necessarily motivated,
regularity in which similar-sounding words have similar meanings)
[@dingemanse2015]. Beyond a handful of strongly iconic pockets — ideophones,
size-sound symbolism, the *bouba/kiki* effect — corpus-scale work has found a
small but reliable systematic signal: within languages [@monaghan2014;
@dautriche2017; @pimentel2019] and, more contentiously, in cross-linguistic
sound–meaning biases shared across unrelated languages [@blasi2016; @joo2020].
Deep-learning studies have since asked whether the mapping is rich enough for a
model to *translate* from phonology alone [@devarda2022].

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

Confirmatory run: `configs/paper_confirmatory.yaml`
(`d4fdffa3635611fa`), 1,842 concepts, 10-fold CV, 300 permutations, 2,000
bootstrap resamples, k = 100 (chance ≈ 100/1842 ≈ 0.054). Figures 1–3 are
regenerated from `results/` by `notebooks/figures.py`.

### 3.1 Cross-lingual retrieval (Fig. 1)

All 12 ordered verified-core pairs retrieve translations above the permutation
null (every p at the 1/301 floor; BH-FDR q = 0.003). Effect sizes scale with
genealogical proximity:

| pair | accuracy [95 % CI] | null |
|---|---|---|
| French ↔ English | 0.42 [0.40, 0.44] | 0.05 |
| English ↔ Irish | 0.26 [0.25, 0.28] | 0.05 |
| French ↔ Irish | 0.20 [0.19, 0.21] | 0.05 |
| Chinese ↔ {English, French, Irish} | 0.08–0.10 | 0.05 |

The homograph-clean accuracies (test items whose gold string also occurs in
training removed) are within 0.01 of these throughout, so string collisions do
not drive the result.

### 3.2 Phonetic vs. control representations (Fig. 2)

For every Indo-European pair the learned phonetic map is **matched or beaten by
the non-learned string baselines**: French → English phonetic 0.42 vs.
`editdist` 0.47 vs. `orth` 0.46; Irish → English 0.26 vs. 0.31 vs. 0.29. The
cross-lingual retrieval signal among related languages is therefore cognate /
borrowing overlap that a plain edit-distance ranker captures at least as well
as the feature-bigram embedding.

For the Chinese ↔ European pairs the string baselines are structurally at zero
(`editdist` and `orth` = 0.000 — disjoint scripts), while the phonetic map
reaches 0.08–0.10. The coarse articulatory baseline `feat` recovers 0.06–0.07
of that, and both sit just above the null (0.05). What remains is a small
effect that a script-independent bag of broad phonological features nearly
fully explains.

### 3.3 Form–meaning correlations (Fig. 3; Mantel, n = 700)

**Within language.** All four verified languages show a positive `form ~
meaning` Mantel correlation that remains significant after partialling out the
orthographic edit-distance matrix:

| language | r | r \| orthography | p (partial) |
|---|---|---|---|
| English | 0.026 | 0.024 | 0.003 |
| French | 0.021 | 0.014 | 0.003 |
| Irish | 0.017 | 0.015 | 0.003 |
| Chinese | 0.017 | 0.004 | 0.023 |

The magnitudes (r ≈ 0.02) match prior corpus estimates of lexical
systematicity [@dautriche2017; @pimentel2019]. Chinese's partial correlation is
the weakest and rests on a weak control (its "orthography" here is a
pinyin-derived string).

**Between languages.** `form ~ form` correlations survive the orthographic
control for the Indo-European pairs (English~French r = 0.054, partial 0.048;
English~Irish 0.034 / 0.029; French~Irish 0.019 / 0.015, all p = 0.003) but
**not** for any Chinese pair: English~Chinese r = 0.004, p = 0.09; Chinese~French
and Chinese~Irish fall to p = 0.30 and p = 0.10 once orthography is partialled
out, with r ≈ 0.003.

### 3.4 Phoneme–meaning associations

Zero phoneme × pole cells survive BH-FDR at q < 0.10 (a lower-power 1,000-
permutation run earlier surfaced 9 marginal Chinese-only cells at q ≈ 0.08).
The bucketed-z-score interpretability layer yields no robust finding at this
scale.

### 3.5 Translation quality (exploratory tier)

`translate-qc` flags 14.6 % of Swahili cells, 8.5 % Hungarian, 5.7 %
Indonesian, 5.2 % Welsh, < 5 % elsewhere — mostly cells returned untranslated
by Google Translate. The full 22-language exploratory analysis (with these
cells excluded) is the subject of a companion run.

## 4. Discussion

Three things hold up and one does not.

**Weak within-language systematicity replicates.** A small, orthography-independent
`form ~ meaning` correlation (r ≈ 0.02) is present in all four hand-verified
languages, English, French, Irish and — more tentatively — Chinese. This is the
robust positive result and it is consistent with the broader literature.

**Related-language retrieval is cognates.** The headline cross-lingual numbers
from the original project (French↔English ≈ 0.4) are real but are not evidence
of sound symbolism: a non-learned edit-distance ranker does as well or better,
and the `form ~ form` correlation, while significant, tracks shared inherited
vocabulary. The phonetic-similarity embedding adds nothing over string overlap
here.

**No Chinese–European sound–meaning bias.** The one setting where the phonetic
map beats the string baselines — Chinese paired with a European language — shows
a retrieval accuracy of ~2× chance that (a) is nearly fully accounted for by
coarse articulatory features and (b) does not appear as a `form ~ form` Mantel
correlation once orthography is controlled. We find no support for a systematic
sound–meaning correspondence shared between Chinese and Indo-European in this
concept set.

**The phoneme-pole analysis is uninformative** as designed; a cross-linguistic
pooled design in the spirit of @blasi2016 would be the way to revisit it.

The net picture is a partial, deflationary replication: the rebuilt pipeline
recovers the weak lexical systematicity that the field expects, and shows that
the original project's larger cross-lingual claims were carried by cognates and
by a retrieval metric sensitive to embedding-space geometry.

### 4.1 Relation to prior work

The direct predecessor is @devarda2022, who trained LSTMs to map a word's
phonetic-feature sequence onto its word2vec vector on five languages and tested
zero-shot on a held-out sixth from a different family. Two convergences are
worth noting. First, effect sizes agree: their semantic experiment reports
Cohen's *d* between 0.05 and 0.22 — the same "small but present" regime as our
Mantel *r* ≈ 0.02. Second, **both studies find a non-Indo-European language
where the effect fails** — their Vietnamese (contrast *n.s.*, *d* = −0.02) and
our Chinese (`form ~ form` *n.s.* after the orthographic control).

Our contribution relative to that work is on the control side. de Varda &
Strapparava's only baseline is a shuffled-pairs model — the equivalent of our
permutation null — and their translated word forms carry the same cognate
contamination ours do; their six-family design mitigates it but never measures
it. Adding the non-learned edit-distance baseline and the partial Mantel lets
us *quantify* that contamination, and show that within Indo-European the
retrieval signal is essentially string overlap. Conversely, their LSTM reads
phoneme *order*; our bag-of-feature-bigrams followed by PCA does not, so
position-dependent sound symbolism (word-initial vs. word-final effects) is
invisible to the present model.

Against the large cross-linguistic bias literature [@blasi2016; @joo2020;
@johansson2020] the design here is inverted: many concepts (~1,800), few
languages (4 verified / 22 total), where those studies use ~100 concepts and
hundreds-to-thousands of languages. With three of four verified languages
Indo-European we cannot separate a universal bias from shared inheritance —
which is exactly what the results show. The magnitude of the within-language
`form ~ meaning` correlation matches the large-lexicon single-language work
[@dautriche2017; @pimentel2019; @monaghan2014].

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
