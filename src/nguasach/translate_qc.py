"""Stage ``translate-qc``: per-cell quality flags for the unverified languages.

The four verified-core columns pass through as ``ok``. For the other 18, offline
heuristics catch the failure modes actually seen in this data (Google-Translate
fallbacks left untranslated -- "cashew", "pistachio" in Swahili; romanized or
copied cells; non-native scripts). An optional ``--online`` back-translation
pass (``deep-translator``) can be layered on later.

Output: ``data/interim/translation_qc.csv`` (concept_id, english, language, cell,
flag) + summary counts. ``cfg.qc_mode`` decides whether the ``align`` /
``associate`` stages drop flagged cells.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter

from .config import Config
from . import data as _data

# columns whose script is non-Latin -> a pure-ASCII cell is almost always a
# failed translation (romanization or untranslated fallback)
_NONLATIN = {
    "Greek", "Russian", "Chinese", "Japanese", "Korean", "Thai",
    "Arabic", "Hebrew", "Hindi",
}
_ASCII = re.compile(r"^[\x00-\x7f]+$")

# Languages with enough shared Germanic/Latin vocabulary that a cell identical
# to the English word is often a real cognate, not a failed translation.
_CLOSE_TO_EN = {"German", "Spanish", "Italian"}


def _flag_cell(cell: str, english: str, lang: str, en_vocab: set[str],
               cross_lang_count: int) -> str:
    c = cell.strip().lower()
    if not c:
        return "empty"
    if c == english.strip().lower():
        return "possible_cognate" if lang in _CLOSE_TO_EN else "untranslated"
    if lang in _NONLATIN and _ASCII.match(cell):
        # an English word in ASCII sitting in a non-Latin column -> fallback
        return "ascii_in_nonlatin" if c not in en_vocab else "english_fallback"
    if lang not in ("English",) and c in en_vocab and lang not in _NONLATIN:
        # a Latin-script column holding a bare English word (weak signal)
        return "english_fallback_weak"
    if cross_lang_count >= 4:
        return "shared_across_langs"               # same string in >=4 other columns
    return "ok"


def run(cfg: Config) -> dict:
    interim = cfg.paths.resolve("interim")
    interim.mkdir(parents=True, exist_ok=True)
    df = _data.load_raw(cfg)
    core = set(cfg.verified_core)

    en_vocab = {unicodedata.normalize("NFC", v).strip().lower() for v in df["English"]}

    rows = []
    per_lang = Counter()
    flagged = Counter()
    for cid in range(len(df)):
        english = df.at[cid, "English"]
        cells = {lang: df.at[cid, lang] for lang in df.columns}
        # how many columns share each surface string, for the copy-detector
        surf_counts = Counter(v.strip().lower() for v in cells.values())
        for lang, cell in cells.items():
            per_lang[lang] += 1
            if lang in core:
                flag = "ok"
            else:
                other = surf_counts[cell.strip().lower()] - 1
                flag = _flag_cell(cell, english, lang, en_vocab, other)
            if flag != "ok":
                flagged[(lang, flag)] += 1
            rows.append({"concept_id": cid, "english": english, "language": lang,
                         "cell": cell, "flag": flag})

    _csv(interim / "translation_qc.csv", rows)

    summary = {}
    for lang in df.columns:
        tot = per_lang[lang]
        f = {k[1]: v for k, v in flagged.items() if k[0] == lang}
        summary[lang] = {"n": tot, "n_flagged": sum(f.values()),
                         "flag_rate": round(sum(f.values()) / tot, 4), "by_flag": f}

    report = {
        "stage": "translate-qc", "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "verified_core": list(cfg.verified_core),
        "per_language": summary,
        "worst": sorted(
            ((l, s["flag_rate"]) for l, s in summary.items() if l not in core),
            key=lambda x: -x[1],
        )[:5],
    }
    (interim / "translation_qc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (interim / "translate-qc.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report


def flagged_concepts(cfg: Config, language: str) -> set[int]:
    """concept_ids to drop for ``language`` given ``cfg.qc_mode`` (used by align/associate)."""
    if cfg.qc_mode == "off" or language in cfg.verified_core:
        return set()
    path = cfg.paths.resolve("interim") / "translation_qc.csv"
    if not path.exists():
        return set()
    import csv

    drop = set()
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if (r["language"] == language
                    and r["flag"] not in ("ok", "english_fallback_weak", "possible_cognate")):
                drop.add(int(r["concept_id"]))
    return drop


def _csv(path, rows: list[dict]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["concept_id", "english", "language", "cell", "flag"])
        w.writeheader()
        w.writerows(rows)
