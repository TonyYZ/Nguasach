"""Stage ``vocalize``: add harakat (Arabic) / niqqud (Hebrew) to those columns.

Both columns are written mostly *unvocalized* (Arabic 98 %, Hebrew 59 %), which
is normal for the scripts but leaves eSpeak-NG guessing vowels for a large,
uncontrolled fraction of each column. This stage fills them in from, in order:

1. the matching **Wiktionary** translation form (hand-vocalised lemmas -- Arabic
   Wiktionary forms are 100 % vocalised), then
2. **CAMeL Tools** MLE diacritiser for Arabic / the **Dicta "Nakdan"** API for
   Hebrew (both operate on the isolated word, so ~80-90 % accurate -- verbs and
   ambiguous forms are the weak spot).

Output: ``data/interim/vocalized_ar_he.xlsx`` -- a review sheet (concept_id,
english, <lang>_old, <lang>_new, source) for a human to check before the values
are merged into ``nguasach.xlsx``. Nothing is written to the corpus here.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

from .config import Config
from . import data as _data

_AR_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_HE_MARKS = re.compile(r"[֑-ׇֽֿׁׂׅׄ]")
_AR_CASE_TAIL = re.compile(r"[ً-َِ-ِْ]$")   # trailing i'rab / sukun
_NAKDAN = "https://nakdan-5-3.loadbalancer.dicta.org.il/api"
_UA = "nguasach-vocalize/0.1 (phonosemantics research)"


def _strip(s: str, lang: str) -> str:
    return (_AR_MARKS if lang == "Arabic" else _HE_MARKS).sub("", str(s or "")).strip()


# ------------------------------------------------------------------ Wiktionary
def _wikt_lookup(cfg: Config):
    p = cfg.paths.resolve("interim") / "wiktionary_cache.json"
    cache = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    out = {"Arabic": {}, "Hebrew": {}}
    for _w, entry in cache.items():
        for lang in ("Arabic", "Hebrew"):
            for form in entry.get(lang, []):
                bare = _strip(form, lang)
                if bare and (_AR_MARKS if lang == "Arabic" else _HE_MARKS).search(form):
                    out[lang].setdefault(bare, form)      # first vocalised form wins
    return out


# ------------------------------------------------------------------ Arabic (local)
def _camel_diacritizer():
    from camel_tools.disambig.mle import MLEDisambiguator

    mle = MLEDisambiguator.pretrained()

    def diac(word: str) -> str:
        toks = word.split()
        res = []
        for t in toks:
            an = mle.disambiguate([t])[0].analyses
            d = an[0].analysis["diac"] if an else t
            res.append(_AR_CASE_TAIL.sub("", d))          # drop citation-form case vowel
        return " ".join(res)

    return diac


# ------------------------------------------------------------------ Hebrew (Dicta)
def _nakdan(words: list[str], cache_path: Path) -> dict[str, str]:
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [w for w in words if w and w not in cache]
    for i, w in enumerate(todo):
        body = json.dumps({"task": "nakdan", "data": w, "genre": "modern",
                           "addmorph": False, "keepqq": False}).encode()
        req = urllib.request.Request(_NAKDAN, data=body,
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": _UA})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=30).read())
            # addmorph=False -> options is a flat list of vocalised strings;
            # "sep" entries carry the whitespace/punctuation between words.
            parts = []
            for opt in r:
                if opt.get("sep"):
                    parts.append(opt.get("word", " "))
                elif opt.get("options"):
                    o = opt["options"][0]
                    parts.append(o if isinstance(o, str) else o[0])
            cache[w] = "".join(parts) or w
        except Exception:
            cache[w] = w
        if i % 40 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"[vocalize] nakdan {i}/{len(todo)}")
        time.sleep(0.25)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


# ------------------------------------------------------------------ stage entry
def run(cfg: Config, n_jobs: int = 1) -> dict:
    interim = cfg.paths.resolve("interim")
    df = _data.load_raw(cfg)
    wikt = _wikt_lookup(cfg)

    rows = []
    stats = {"Arabic": {}, "Hebrew": {}}
    # Hebrew: batch the Nakdan calls up front
    he_cells = sorted({_strip(df.at[i, "Hebrew"], "Hebrew") for i in range(len(df))})
    he_cells = [c for c in he_cells if c and not wikt["Hebrew"].get(c)]
    nak = _nakdan(he_cells, interim / "nakdan_cache.json") if he_cells else {}
    ar_diac = _camel_diacritizer()

    for cid in range(len(df)):
        eng = df.at[cid, "English"]
        rec = {"concept_id": cid, "english": eng}
        for lang, tool_cache, tool_fn in (("Arabic", None, ar_diac),
                                          ("Hebrew", nak, None)):
            old = str(df.at[cid, lang]).strip()
            bare = _strip(old, lang)
            if not bare:
                new, src = old, "empty"
            elif bare in wikt[lang]:
                new, src = wikt[lang][bare], "wiktionary"
            elif lang == "Arabic":
                new, src = tool_fn(bare), "camel"
            else:
                new, src = nak.get(bare, bare), "nakdan"
            rec[f"{lang}_old"] = old
            rec[f"{lang}_new"] = new
            rec[f"{lang}_source"] = src
            stats[lang][src] = stats[lang].get(src, 0) + 1
        rows.append(rec)

    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "vocalize"
    cols = ["concept_id", "english",
            "Arabic_old", "Arabic_new", "Arabic_source",
            "Hebrew_old", "Hebrew_new", "Hebrew_source"]
    ws.append(cols)
    for r in rows:
        ws.append([r.get(c, "") for c in cols])
    out = interim / "vocalized_ar_he.xlsx"
    wb.save(out)

    report = {"stage": "vocalize", "config": cfg.name,
              "config_fingerprint": cfg.fingerprint(),
              "n_concepts": len(rows), "by_source": stats,
              "output": str(out.relative_to(cfg.paths.resolve("xlsx").parents[1]))}
    (interim / "vocalize.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    (interim / "vocalize.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report
