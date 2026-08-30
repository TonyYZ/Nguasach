---
title: "Measuring form–meaning systematicity across languages with phonetic-similarity embeddings"
author: "Tony Zhou"
status: draft — confirmatory + exploratory runs complete
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

**Null.** We shuffle the source↔target concept pairing and re-run the entire
fit-and-retrieve — 300 permutations in the confirmatory run, 200 in the
exploratory — giving an empirical p = (1 + #{null ≥ observed}) / (iters + 1).
Benjamini–Hochberg FDR is applied separately to the confirmatory pairs and to
the exploratory set. A homograph guard flags test items whose gold target
string also occurs in training; we report accuracy with and without them.

### 2.4 Distance-matrix (Mantel) analysis

On a concept subsample (700 confirmatory, 600 exploratory) we build the
pairwise phonetic-distance matrix `D_form` (1 − cosine on the phonetic
embeddings), the meaning-distance matrix `D_mean` (1 − cosine on the semantic
vectors) and the orthographic matrix `D_orth` (1 − normalized Levenshtein on
surface forms). The Mantel statistic is the Pearson correlation of the upper
triangles; the **partial** Mantel [@smouse1986] residualizes `D_form` and
`D_mean` on `D_orth` before correlating, with a degeneracy guard that reports
the partial as the raw value (flagged) when `D_orth` has near-zero variance.
For **logographic** scripts (Chinese, Japanese) a character edit-distance
matrix is not an orthographic-similarity measure, so their partial Mantel is
always flagged not-interpretable. Significance is a permutation null over
concept labels (matching the retrieval null iteration count). We run
`D_form(L)` vs `D_mean` for every language and `D_form(L1)` vs `D_form(L2)` for
every language pair.

### 2.5 Baselines

All three baselines are **non-learned** direct-similarity retrieval (no ridge
map) run through the identical CV + CSLS + null + FDR machinery, so the learned
map's contribution is isolable. They are computed for the verified core only.

* **`editdist`** — normalized Levenshtein similarity of surface strings. The
  cognate/borrowing control.
* **`orth`** — cosine of character n-gram (2, 3) count vectors over a
  vocabulary shared across languages; cross-script pairs score at chance.
* **`feat`** — cosine of the mean `panphon` articulatory-feature vector per
  word (script-independent, from the IPA).

### 2.6 Phoneme–meaning association

Eighteen semantic "poles" (hand-built English seed-word clusters, from the
original I-Ching–themed design; the framing is cosmetic, the poles are just
regions of semantic space). Each concept is assigned to its nearest pole in the
phonetic→semantic-aligned space; per pole we z-score every phoneme's frequency
against its distribution across poles, with a permutation null over the
concept↔pole assignment (iteration count matching the run) and BH-FDR across all phoneme × pole cells.

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

### 3.5 Exploratory: all 22 languages

Config `configs/paper_exploratory.yaml` (`7dfa4915905ccd4c`): 22 languages,
QC-flagged cells excluded (`translate-qc` flags 14.6 % of Swahili cells, 8.5 %
Hungarian, 5.7 % Indonesian, 5.2 % Welsh, < 5 % elsewhere — mostly cells
returned untranslated by Google Translate); retrieval restricted to
English↔X and X→Semantics (64 pairs); 200 permutations; Mantel on n = 600.

**Within-language `form ~ meaning` is near-universal.** The raw Mantel *r* is
positive and significant (*p* = .005) in **all 22 languages**, from 0.013
(Hebrew) to 0.039 (Japanese). After partialling out orthography it stays
significant in **every language where the control is meaningful** — 20 of 22
outright; Japanese survives (partial *r* = 0.036) though its mixed kana/kanji
script makes the character edit-distance control only partial; **Chinese is the
sole language whose partial correlation is not interpretable** (logographic
script — a character edit-distance matrix is not an orthographic-similarity
measure; its raw *r* = 0.015, *p* = .005, is real).

**Cross-language `form ~ form` is small but pervasive.** Across all 231 language
pairs, 205 have a significant raw correlation and 164 of 190 non-logographic
pairs remain significant after the orthographic control (median partial
*r* ≈ 0.011). Splitting by genealogy:

| pair type | n | raw sig. | partial sig. | median partial *r* |
|---|---|---|---|---|
| both Indo-European | 45 | 45/45 | 45/45 | 0.019 |
| ≥ 1 non-Indo-European | 186 | 160 | 119/145 | 0.009 |

The strongest correlations are the obvious cognate cases (Spanish~Italian
*r* = 0.17, Italian~French 0.12, Spanish~French 0.11); the interesting residue
is that ~80 % of *cross-family* pairs still show a small correlation that
survives partialling out orthography — e.g. Korean~Chinese partial 0.017,
Vietnamese~Chinese 0.010 (both plausibly Sino-xenic loan strata),
Turkish~Arabic 0.022, German~Japanese 0.018, English~Hindi 0.048 (Indo-Aryan).
A handful of cross-family pairs are genuinely null even before partialling —
notably **English~Chinese (raw *r* = 0.001, *p* = .71)**.

**Retrieval scales with genealogical proximity to English.** English↔German
0.46, ↔Spanish/Italian 0.38, ↔Hindi 0.32, ↔Russian 0.23, ↔Greek 0.16, down to
↔Vietnamese and ↔Hebrew 0.12–0.14 — all above their permutation null (≈ 0.053),
all *q* = .005, but tracking shared vocabulary. Every language's X→Semantics
retrieval lands in 0.17–0.26 (null ≈ 0.054), consistent with the universal
`form ~ meaning` Mantel result. (One artifact: English→Japanese has an inflated
null of 0.159 — residual hubness in the Japanese phonetic space that CSLS does
not fully remove — so its 0.27 accuracy overstates the effect.)

The phoneme–pole association analysis again yields zero cells at *q* < 0.10.

## 4. Discussion

**Within-language systematicity is small and near-universal.** An
orthography-independent `form ~ meaning` correlation (*r* ≈ 0.013–0.039) is
present in every one of the 22 languages, and survives the orthographic control
wherever that control is meaningful (all but Chinese, whose logographic script
defeats an edit-distance measure). This is the robust positive result, and its
magnitude and pervasiveness match the 100-language estimate of @dautriche2017.
The four-language confirmatory run understated this — with only English, French,
Irish and Chinese, French's partial correlation looked marginal; the full run
shows French squarely in the significant band and the effect holding across
Uralic, Turkic, Austroasiatic, Austronesian, Japonic, Koreanic, Kra-Dai and
Afro-Asiatic languages.

**Cross-language form correspondence is real but tiny, and mostly not
cognates.** Of 231 language pairs, ~80 % show a small `form ~ form` correlation
(median partial *r* ≈ 0.01) that survives partialling out orthographic
similarity. Indo-European pairs are stronger (median 0.019) and the very top of
the range is transparent cognate overlap (Spanish~Italian *r* = 0.17). But the
residual signal among *unrelated* families — Korean~Chinese, Turkish~Arabic,
German~Japanese — is above chance after the orthographic control. It is far too
weak to carry a strong universality claim, and some of it is contact rather
than shared bias (Sino-xenic vocabulary in Korean and Vietnamese), but it is
not nothing.

**The retrieval framing is dominated by cognates and by embedding geometry.**
The original project's headline cross-lingual numbers (French↔English ≈ 0.4)
are real but a non-learned edit-distance ranker matches or beats them on every
Indo-European pair; the learned phonetic map adds nothing over string overlap
there. And for degenerate representations the retrieval null sits far above
chance (the char-n-gram and mean-panphon baselines before CSLS; Japanese even
after). Retrieval accuracy is a poor primary estimand for this question; the
Mantel correlation is the one to report.

**Chinese is the recurring exception, for at least two different reasons.**
Its logographic script rules out the orthographic control, so we cannot say
whether its within-language `form ~ meaning` correlation (raw *r* = 0.015)
survives it. And English~Chinese is the one language pair with *no* `form ~
form` correlation at all (raw *r* = 0.001) — where Chinese does correlate
(Korean, Vietnamese) it is attributable to loan vocabulary. The bespoke
pinyin→IPA transcription (§2.2) may also contribute.

**The phoneme-pole analysis is uninformative** as designed; a cross-linguistic
pooled design in the spirit of @blasi2016 would be the way to revisit it.

The net picture: the weak lexical systematicity the field expects is here, and
it generalises across families; the original project's larger cross-lingual
retrieval claims were carried by cognates and by a metric sensitive to
embedding-space geometry, but a small genuine cross-family form–meaning signal
does survive the controls.

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
