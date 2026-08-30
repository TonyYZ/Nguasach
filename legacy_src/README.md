# legacy_src/ — the original scripts, superseded

These are the pre-restructure scripts. Their **logic has been ported** into the
`src/nguasach/` package (config-driven, tested); they are kept only for
reference and provenance. Not on any code path, not tracked in git.

| original | ported to |
|----------|-----------|
| `main.py` (Google-Translate driver) | `data.py` treats `nguasach.xlsx` as frozen input; no live translation |
| `pronounce.py` (eSpeak/bophono/dict IPA) | `ipa.py` (eSpeak NG 1.52 via phonemizer) |
| `processChn.py` (pinyin→IPA map) | `_zh.py` (verbatim map) + `ipa.py` |
| `processAll.py` / `processThai.py` / `purifyKorean.py` | folded into `ipa.py` (`purify`) |
| `compressSemantics.py` / `loadLarge.py` / `lemmatize.py` | `semantics.py` |
| `transPhone.py` (1182 lines) | `align.py` + `crossval.py` + `nulls.py` + `association.py` + `report.py` |
| `sortHexagram.py` / `hexFilter.py` | `association.py` |
| `neuralNetwork.py` | `align.py` `map: nn` (stub — not reimplemented) |
| `transposeResult.py` / `transcriber_data.py` / `init.py` / `test.py` | obsolete |

Exploratory offshoots that were never on the critical path are in `../attic/`.
