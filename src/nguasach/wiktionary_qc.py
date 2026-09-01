"""Stage ``wiktionary-qc``: cross-check every translation against Wiktionary.

A second, independent reference to :mod:`nguasach.lexibank_qc` (NorthEuraLex),
with two advantages: it covers **all 22 languages** -- including Vietnamese,
Thai, Indonesian, Arabic and Swahili, which NorthEuraLex lacks -- and it has the
abstract / modern vocabulary Concepticon never mapped.

The English Wiktionary's ``Translations`` sections are pulled via the MediaWiki
API (50 titles/request, cached in ``data/interim/wiktionary_cache.json``) and
the ``{{t|xx|form}}`` / ``{{t+|xx|form}}`` templates parsed out.

A cell that disagrees with **both** NorthEuraLex and Wiktionary is the strong
signal; agreeing with either is almost always fine.

Output: ``data/interim/wiktionary_qc.csv`` +
``data/interim/translation_qc_combined.csv`` (per cell: nel / wiktionary /
consensus) + ``wiktionary_qc.json``.
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from .config import Config
from . import data as _data
from .lexibank_qc import _agree, _norm

_API = "https://en.wiktionary.org/w/api.php"
_UA = "nguasach-translation-qc/0.1 (cross-linguistic phonosemantics research)"

# Wiktionary language code -> our column
_WK_TO_OURS = {
    "hu": "Hungarian", "fi": "Finnish", "el": "Greek", "ru": "Russian",
    "de": "German", "es": "Spanish", "it": "Italian", "fr": "French",
    "ga": "Irish", "cy": "Welsh", "en": "English", "zh": "Chinese",
    "cmn": "Chinese", "vi": "Vietnamese", "ja": "Japanese", "ko": "Korean",
    "th": "Thai", "id": "Indonesian", "tr": "Turkish", "ar": "Arabic",
    "he": "Hebrew", "sw": "Swahili", "hi": "Hindi",
}
_T_TEMPLATE = re.compile(r"\{\{t{1,2}\+?\|([a-z][a-z-]*)\|([^|}\n]+)")
_WIKI_MARKUP = re.compile(r"\[\[|\]\]|'''?|<[^>]+>")


def _clean_form(s: str) -> str:
    s = _WIKI_MARKUP.sub("", s).strip()
    return s.split("|")[-1].strip()      # [[lemma|display]] already split by regex; guard


def _parse_translations(wikitext: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for code, form in _T_TEMPLATE.findall(wikitext):
        ours = _WK_TO_OURS.get(code)
        f = _clean_form(form)
        if ours and f:
            out.setdefault(ours, set()).add(f)
    return out


def fetch(words: list[str], cache_path: Path, batch: int = 50) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    todo = [w for w in words if w not in cache]
    # English Wiktionary main namespace is case-sensitive: "false" (adjective,
    # with the Translations table) is a different page from "False". Query the
    # lowercase variant too and merge both pages' translations.
    want = {w: {w, w.lower()} for w in todo}
    titles = sorted({t for s in want.values() for t in s})
    pages: dict[str, dict[str, list[str]]] = {}
    for i in range(0, len(titles), batch):
        chunk = titles[i : i + batch]
        q = urllib.parse.urlencode({
            "action": "query", "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "titles": "|".join(chunk),
            "format": "json", "formatversion": "2",
        })
        req = urllib.request.Request(f"{_API}?{q}", headers={"User-Agent": _UA})
        for attempt in range(6):
            try:
                data = json.loads(urllib.request.urlopen(req, timeout=45).read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise
        else:
            raise RuntimeError("wiktionary API kept rate-limiting")
        for pg in data.get("query", {}).get("pages", []):
            wt = (pg.get("revisions", [{}])[0].get("slots", {})
                  .get("main", {}).get("content", "")) if pg.get("revisions") else ""
            pages[pg["title"]] = {k: sorted(v) for k, v in _parse_translations(wt).items()}
        for w in todo:
            if w in cache or not want[w] <= set(pages):
                continue
            merged: dict[str, set[str]] = {}
            for t in want[w]:
                for lang, forms in pages.get(t, {}).items():
                    merged.setdefault(lang, set()).update(forms)
            cache[w] = {lang: sorted(v) for lang, v in merged.items()}
        print(f"[wiktionary-qc] fetched {min(i + batch, len(titles))}/{len(titles)} titles")
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        time.sleep(1.2)
    return cache


def run(cfg: Config, n_jobs: int = 1) -> dict:
    interim = cfg.paths.resolve("interim")
    interim.mkdir(parents=True, exist_ok=True)
    df = _data.load_raw(cfg)
    verified = set(cfg.verified_core)
    langs = [l for l in _WK_TO_OURS.values() if l in df.columns]
    langs = sorted(set(langs) - {"English"})

    words = sorted({str(df.at[i, "English"]).strip() for i in range(len(df))})
    wik = fetch(words, interim / "wiktionary_cache.json")

    rows, per_lang = [], {l: {"checked": 0, "match": 0, "differ": 0} for l in langs}
    per_cell: dict[tuple[int, str], str] = {}
    for cid in range(len(df)):
        eng = str(df.at[cid, "English"]).strip()
        refs = wik.get(eng, {})
        for lang in langs:
            attested = set(refs.get(lang, []))
            if not attested:
                continue
            cell = df.at[cid, lang]
            ok = _agree(cell, attested, lang)
            per_lang[lang]["checked"] += 1
            per_lang[lang]["match" if ok else "differ"] += 1
            per_cell[(cid, lang)] = "match" if ok else "differ"
            if not ok:
                rows.append({"concept_id": cid, "english": eng, "language": lang,
                             "ours": cell,
                             "wiktionary": " / ".join(sorted(attested)[:6])})

    rows.sort(key=lambda r: (r["language"], r["english"]))
    with (interim / "wiktionary_qc.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["concept_id", "english", "language", "ours", "wiktionary"])
        for r in rows:
            w.writerow([r[k] for k in ("concept_id", "english", "language",
                                       "ours", "wiktionary")])

    combined = _combine(interim, df, per_cell, verified)
    for l, d in per_lang.items():
        d["disagree_rate"] = round(d["differ"] / d["checked"], 3) if d["checked"] else None

    summary = {
        "stage": "wiktionary-qc", "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "source": "English Wiktionary Translations sections (MediaWiki API)",
        "n_words": len(words), "total_checked": sum(d["checked"] for d in per_lang.values()),
        "total_disagree": len(rows), "per_language": per_lang,
        "combined": combined,
        "note": ("Independent second reference to lexibank-qc. Cells in "
                 "translation_qc_combined.csv marked 'differ_both' disagree with "
                 "NorthEuraLex AND Wiktionary -- the strong error signal."),
    }
    (interim / "wiktionary_qc.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (interim / "wiktionary-qc.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return summary


def _combine(interim: Path, df, wik_cell: dict, verified: set) -> dict:
    """Join the wiktionary verdicts with the lexibank-qc (NorthEuraLex) ones."""
    nel_disagree = set()
    p = interim / "lexibank_qc.csv"
    if p.exists():
        for r in csv.DictReader(p.read_text(encoding="utf-8").splitlines()):
            nel_disagree.add((int(r["concept_id"]), r["language"]))

    keys = set(wik_cell) | nel_disagree
    n = {"differ_both": 0, "differ_nel_only": 0, "differ_wik_only": 0, "ok": 0}
    out_rows = []
    for (cid, lang) in sorted(keys):
        if lang in verified:
            continue
        w = wik_cell.get((cid, lang))            # "match" | "differ" | None
        nel = (cid, lang) in nel_disagree
        if w == "differ" and nel:
            verdict = "differ_both"
        elif nel:
            verdict = "differ_nel_only"
        elif w == "differ":
            verdict = "differ_wik_only"
        else:
            verdict = "ok"
        n[verdict] += 1
        if verdict.startswith("differ"):
            out_rows.append({
                "concept_id": cid, "english": str(df.at[cid, "English"]).strip(),
                "language": lang, "ours": df.at[cid, lang], "verdict": verdict,
            })
    out_rows.sort(key=lambda r: (r["verdict"] != "differ_both", r["language"], r["english"]))
    with (interim / "translation_qc_combined.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["concept_id", "english", "language", "ours", "verdict"])
        for r in out_rows:
            w.writerow([r[k] for k in ("concept_id", "english", "language",
                                       "ours", "verdict")])
    return n
