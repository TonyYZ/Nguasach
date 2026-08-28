# Legacy file inventory

Status of the pre-existing files at the repo root, and what happens to each in
the rerun. Nothing here is deleted yet; the disruptive move to `legacy_results/`
is staged for a follow-up once the new pipeline reproduces the outputs.

## Inputs — keep, promoted to `data/raw/`

| file | role | action |
|------|------|--------|
| `nguasach.xlsx` | canonical concept table (1,842 × 22) | copied to `data/raw/nguasach.xlsx` |
| `nguasachV.csv` | old 26k table; `Semantics` column only | copied to `data/raw/nguasachV.csv` |
| `nguasach.csv` | **corrupted** export (mojibake + `?`) | ignore; do not read |
| `Thai.txt`, `ThaiV.txt`, `ThaiPure.txt` | thai-language.com dictionary scrape + extract | inputs to `ipa.py` (Thai) |
| `Chinese.txt` | Chinese concept labels (== old nguasachV) | superseded by `data.py` |
| `KoreanRaw.txt` | raw Korean dictionary dump | input to `purifyKorean.py` port |
| `necessary.txt`, `to be added.txt` | concept-list working notes | reference only |

## Pipeline code — ported into `src/nguasach/` (Phase 1)

`main.py`, `pronounce.py`, `processAll.py`, `processChn.py`, `processThai.py`,
`purifyKorean.py`, `compressSemantics.py`, `loadLarge.py`, `lemmatize.py`,
`transPhone.py`, `sortHexagram.py`, `hexFilter.py`, `neuralNetwork.py`.
Originals stay at root until each port is verified, then move to `legacy_src/`.

## Outputs — freeze to `legacy_results/` for before/after comparison

| group | files | produced by | notes |
|-------|-------|-------------|-------|
| phoneme-association tables | `<Lang>Hex.txt`, `<Lang>ZScore.txt` for 20 languages + `Semantics*`, `Uni*` | `transPhone.generateHex` / `sortHexagram.py` | current "hexagram" results; `2nd` / `O` / `Translated` / `Double` variants are parameter sweeps with no manifest |
| duplicated/older | `EnglishHex2.txt`, `SemanticsHex2.txt` | earlier runs | supersede |
| etymology tables | `etymTable0..12*.csv` (+ `.xlsx`, `.pdf`) | `generateTable` modes | exploratory; not on critical path |
| THINGS mapping | `things{,2,3}.txt`, `thingsDict*.{csv,xlsx,py}` | `thingsDict.py` | exploratory (dataset translation cross-check) |
| FrameNet | `frameCorpus.txt`, `frameDict.txt`, `frameReference.txt` | `translateByFrameNet.py` (attic) | exploratory |
| figures | `Heatmap_EngChn{,2}.png`, `English.png`, `Altaic.png`, `Iti.png` | ad hoc matplotlib | regenerate from `results/` |
| misc | `nodes.html` (pyvis lang network), `average.csv`, `heatmap test.txt`, `trainLabels.txt`, `lemmatized.txt` | various | intermediate scratch |

## Already quarantined → `attic/`

See `attic/README.md`.
