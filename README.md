# Nguasach — A Cross-Linguistic Phonetic-Semantic Corpus

Nguasach is a multilingual dataset and analysis pipeline for studying **language iconicity**: the non-arbitrary correspondence between the sound structure of words and their meanings. The project builds phonetic vector representations of basic vocabulary across 22 languages and uses cross-lingual transfer learning to map phonetic and semantic spaces, asking whether words that mean similar things tend to sound similar across languages. The name comes from the Irish word *cnuasach* (collection), the Vietnamese word *ngữ* (language), and the Vietnamese word *sách* (book).

---

## Motivation

The arbitrariness of the linguistic sign — the Saussurean claim that form and meaning are unrelated — has been challenged by cross-linguistic studies of sound symbolism and iconicity. This project provides a computational framework for measuring phonetic-semantic correspondences at scale, across a typologically diverse set of languages.

---

## Languages and Data

- **Core vocabulary**: ~3,000 basic words per language (sourced from translation resources and the THINGS dataset)
- **Languages covered**: 22 languages including Mandarin Chinese, Japanese, Thai, Sanskrit, Tibetan, and others
- **Data files**:
  - `nguasach.xlsx` — full multilingual vocabulary table
  - `collection.csv` — raw translation results
  - `nguasachV.csv` — vocabulary formatted for Phonetic Similarity Vector (PSV) input
  - `Other.csv` — IPA representations for all non-Chinese languages after processing

---

## Pipeline Overview

The pipeline proceeds in four stages:

### 1. Data Collection
- `main.py` — imports and processes Google Translate output files
- `thingsDict.py` — generates translations of the THINGS dataset across all languages, using MUSE multilingual dictionaries with Google Translate as fallback, cross-validated against word2word alternatives
- `thingsDictCompare.csv` — records comparison results between translation sources

### 2. Phonetic Transcription
Converts vocabulary into IPA (International Phonetic Alphabet) representations for each language:
- `pronounce.py` — the main transcription script; uses eSpeak NG for most languages, language-specific libraries for Sanskrit and Tibetan, a dictionary file for Japanese, and `ThaiV.txt` for Thai
- `processThai.py` — extracts Thai IPA from raw dictionary data (`Thai.txt`)
- `resultV.csv` / `resultVT.xlsx` — transcription results in original and transposed format
- `transposeResult.py` — utility for transposing the results table

### 3. Vector Generation (PSV — Phonetic Similarity Vectors)
Converts IPA transcriptions into dense phonetic embedding vectors:
- `processChn.py` — converts Chinese vocabulary into PSV-compatible format → `ChineseV.txt`
- `processAll.py` — converts all other languages into PSV-compatible format, saves as `[Language]V.txt` in the `PSV/` folder, and calls `generate.py` for each
- `generate.py` — reads a PSV-formatted IPA file and generates a phonetic embedding vector per word → `[Language]Emb.txt`
- `generateTable.py` — generates a syllable table (consonant × vowel) for a given language, called by `processAll.py`
- `similarity.py` — interactive script: takes a word as input and returns the closest words by phonetic vector distance
- `sortHexagram.py` — converts continuous embedding values to binary (0/1) sequences inspired by Yijing hexagrams → `[Language]hex.txt`

### 4. Cross-Lingual Transfer Learning
Maps phonetic vector spaces across language pairs to test cross-linguistic phonetic-semantic correspondences:
- `transPhone.py` — loads two languages, builds KeyedVectors models, trains a bilingual model using `transvec` on 80% of vocabulary, and evaluates on the remaining 20%
- `neuralNetwork.py` — neural network variant of the cross-lingual transfer in `transPhone.py`
- `compressSemantics.py` — reduces dimensionality of semantic embedding files (`model.txt` → `SemanticsEmb.txt`)
- `loadLarge.py` — loads ~600,000 English vectors from `cc.en.300.vec`, filters to ~25,000 content words, augments with vocabulary from Nguasach

---

## Key Dependencies

- [eSpeak NG](https://github.com/espeak-ng/espeak-ng) — phonetic transcription for most languages
- [transvec](https://github.com/Babylonpartners/transvec) — cross-lingual vector space alignment
- [MUSE](https://github.com/facebookresearch/MUSE) — multilingual word embeddings
- [word2word](https://github.com/kakaobrain/word2word) — bilingual lexicons
- Python: `gensim`, `pandas`, `numpy`

---

## Related Projects

- [Aitia](https://github.com/TonyYZ/Aitia) — Bayesian concept learning using a spatial language of thought formalized from Yijing trigrams
- [Ítí (Ete)](https://github.com/TonyYZ/Ete) — an artistic constructed language and web-based learnability experiment

---

## Author

Yutong (Tony) Zhou — M1 Cognitive Science, ENS-PSL  
Background in cognitive science, computational linguistics, and cross-linguistic semantics.
