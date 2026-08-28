"""Stage ``ipa``: each translation -> space-separated IPA phones.

Output per language: ``data/interim/<Lang>V.txt`` in the PSV input format the
vendored ``generate.py`` expects -- one line ``<label>  <p h o n e s>`` (two
spaces), ``<label>`` from ``labels.json`` so it stays concept_id-aligned.

Backends (mirrors the original ``pronounce.py`` / ``processChn.py``):

* eSpeak NG 1.52 via ``phonemizer`` + ``espeakng-loader`` (pinned wheel, no
  system install) for the 21 non-Chinese languages -- Thai and Japanese
  included, which the old bundled 1.50 build had no voice for.
* Chinese: the bespoke pinyin->IPA map in :mod:`nguasach._zh`.
"""

from __future__ import annotations

import json
import logging
import unicodedata

from .config import Config
from . import data as _data

logging.getLogger("phonemizer").setLevel(logging.ERROR)

# Language column -> eSpeak/phonemizer voice code.
LANG_VOICE: dict[str, str] = {
    "Hungarian": "hu", "Finnish": "fi", "Greek": "el", "Russian": "ru",
    "German": "de", "Spanish": "es", "Italian": "it", "French": "fr-fr",
    "Irish": "ga", "Welsh": "cy", "English": "en-gb", "Vietnamese": "vi",
    "Japanese": "ja", "Korean": "ko", "Thai": "th", "Indonesian": "id",
    "Turkish": "tr", "Arabic": "ar", "Hebrew": "he", "Swahili": "sw",
    "Hindi": "hi",
}

# eSpeak-ng voice quirks: raw output substring -> IPA replacement, applied per
# language before purify(). eSpeak's Irish voice emits a literal uppercase "A"
# for a low back vowel (still true in 1.52).
LANG_FIXUPS: dict[str, dict[str, str]] = {"Irish": {"A": "ɑ"}}

# Combining marks kept and re-attached to their segment: nasalization (U+0303,
# which featurephone maps to 'nzd') and dental (U+032A). Every other combining
# mark eSpeak emits -- lowered U+031E, centralized U+0308, syllabic U+0329,
# non-syllabic U+032F, raised, breve, ... -- is dropped (the feature-bigram
# model does not represent them and ipa2feature.csv lacks the combinations).
_KEEP_COMBINING = {"̃", "̪", "͡"}  # nasal, dental, tie bar (all re-attached below)
_REGLUE = ["̪", "ʰ", "ʲ", "̃", "ᵝ"]  # dental, aspirated, palatalized, nasal, compressed

# Modifier letters / marks with no feature-model counterpart: stress, length,
# prosody, tone letters, pharyngealization/velarization/glottalization, quotes.
# Tone *digits* are handled by the isdigit() filter below.
_STRIP_CHARS = set("ˈˌːˑ‿.|-?\"'" + "˥˦˧˨˩" + "ˤˠˀ")


def purify(s: str) -> str:
    """processAll.purify + NFD normalization + suprasegmental/diacritic
    stripping, so output matches the original <Lang>V.txt format (which has no
    stress, length, tone, or exotic diacritics). Space every char, then glue
    tie bars and the kept combining marks back onto their segment."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(
        c for c in s
        if not (unicodedata.combining(c) and c not in _KEEP_COMBINING)
        and not c.isdigit()
        and c not in _STRIP_CHARS
    )
    s = " ".join(ch for ch in s.replace(" ", ""))
    s = s.replace(" ͡ ", "͡")
    for d in _REGLUE:
        s = s.replace(" " + d, d)
    return s.strip()


# ------------------------------------------------------------------ phonemizer
_BACKENDS: dict[str, object] = {}


def _setup_espeak() -> None:
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


def _backend(code: str):
    if code not in _BACKENDS:
        from phonemizer.backend import EspeakBackend

        _BACKENDS[code] = EspeakBackend(
            code, with_stress=True, tie=True, language_switch="remove-flags"
        )
    return _BACKENDS[code]


def espeak_version() -> str:
    from phonemizer.backend import EspeakBackend

    return ".".join(str(x) for x in EspeakBackend("en-gb").version())


def espeak_ipa(words: list[str], code: str) -> list[str]:
    out = _backend(code).phonemize(words, strip=True, njobs=1)
    if len(out) != len(words):
        raise RuntimeError(f"phonemizer({code}) returned {len(out)}/{len(words)} lines")
    return out


# ------------------------------------------------------------------ stage main
def _write_v(path, labels: list[str], ipas: list[str]) -> int:
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for lab, ipa in zip(labels, ipas):
            ipa = ipa.strip()
            if ipa:
                fh.write(f"{lab}  {ipa}\n")
                n += 1
    return n


def run(cfg: Config) -> dict:
    interim = cfg.paths.resolve("interim")
    interim.mkdir(parents=True, exist_ok=True)
    labels = json.loads((interim / "labels.json").read_text(encoding="utf-8"))
    df = _data.load_raw(cfg)

    _setup_espeak()
    version = espeak_version()

    written, skipped = {}, []
    for lang in cfg.languages:
        words = df[lang].tolist()
        labs = labels[lang]

        if lang in LANG_VOICE:
            raw = espeak_ipa(words, LANG_VOICE[lang])
            fix = LANG_FIXUPS.get(lang)
            if fix:
                raw = ["".join(fix.get(ch, ch) for ch in line) for line in raw]
            ipas = [purify(x) for x in raw]
        elif lang == "Chinese":
            from ._zh import chinese_ipa

            ipas = [chinese_ipa(w) for w in words]
        else:
            skipped.append(lang)
            continue

        written[lang] = _write_v(interim / f"{lang}V.txt", labs, ipas)

    report = {
        "stage": "ipa",
        "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "espeak_version": version,
        "backend": "phonemizer+espeakng-loader",
        "written": written,
        "skipped": skipped,
    }
    (interim / "ipa_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not skipped:
        (interim / "ipa.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report
