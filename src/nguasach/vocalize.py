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
_HE_INF = re.compile("^ל[ְֶ]")   # Hebrew infinitive prefix ל(shva|segol)
_NAKDAN = "https://nakdan-5-3.loadbalancer.dicta.org.il/api"
_UA = "nguasach-vocalize/0.1 (phonosemantics research)"

_ONTO_POS = {"Action/Process": "verb", "Property": "adj", "Number": "num",
             "Person/Thing": "noun"}


def _strip(s: str, lang: str) -> str:
    return (_AR_MARKS if lang == "Arabic" else _HE_MARKS).sub("", str(s or "")).strip()


def _load_pos(cfg: Config) -> dict:
    """concept_id -> coarse POS from the pos_*.txt strata lists and the
    concepticon ONTOLOGICAL_CATEGORY, whichever fires."""
    import csv as _csv

    raw = cfg.paths.resolve("xlsx").parent
    df = _data.load_raw(cfg)
    eng = {i: str(df.at[i, "English"]).strip().lower() for i in range(len(df))}
    pos = {}
    for tag, fn in (("verb", "pos_verb.txt"), ("noun", "pos_noun.txt"),
                    ("adj", "pos_adj.txt")):
        p = raw / fn
        if not p.exists():
            continue
        words = {w.strip().lower().removeprefix("to ")
                 for w in p.read_text(encoding="utf-8").splitlines() if w.strip()}
        for i, e in eng.items():
            if e in words or e.removeprefix("to ") in words:
                pos.setdefault(i, tag)
    ccat = {}
    ccp = raw / "concepticon-data" / "concepticondata" / "concepticon.tsv"
    if ccp.exists():
        for r in _csv.DictReader(ccp.read_text(encoding="utf-8").splitlines(), delimiter="\t"):
            ccat[r["ID"]] = _ONTO_POS.get(r.get("ONTOLOGICAL_CATEGORY", ""), "")
    cm = cfg.paths.resolve("interim") / "concepticon_map.csv"
    if cm.exists():
        for r in _csv.DictReader(cm.read_text(encoding="utf-8").splitlines()):
            cid = int(r["concept_id"])
            if cid not in pos and ccat.get(r.get("concepticon_id", "")):
                pos[cid] = ccat[r["concepticon_id"]]
    return pos


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
    _AR_POS = {"verb": {"verb"}, "noun": {"noun", "noun_prop", "noun_num"},
               "adj": {"adj", "adj_comp", "adj_num"}, "num": {"noun_num", "adj_num"}}

    def diac(word: str, want: str | None = None):
        toks, out, ok_all = word.split(), [], True
        for t in toks:
            ans = mle.disambiguate([t])[0].analyses
            if not ans:
                out.append(t); ok_all = False; continue
            chosen, pos_ok = ans[0].analysis, want is None
            if want and want in _AR_POS:
                for a in ans[:6]:
                    if a.analysis.get("pos") in _AR_POS[want]:
                        chosen, pos_ok = a.analysis, True
                        break
            out.append(_AR_CASE_TAIL.sub("", chosen["diac"]))
            ok_all &= pos_ok
        return " ".join(out), ok_all

    return diac


# ------------------------------------------------------------------ Hebrew (Dicta)
def _nakdan(words: list[str], cache_path: Path) -> dict:
    """bare word -> {"opts": [vocalised, ...], "confident": bool}."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [w for w in words if w and w not in cache]
    for i, w in enumerate(todo):
        body = json.dumps({"task": "nakdan", "data": w, "genre": "modern",
                           "addmorph": True, "keepqq": False}).encode()
        req = urllib.request.Request(_NAKDAN, data=body,
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": _UA})
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=30).read())
            per_word = [o for o in r if not o.get("sep")]
            if len(per_word) == 1:
                opts = [o[0] if isinstance(o, list) else o
                        for o in per_word[0].get("options", [])]
                cache[w] = {"opts": opts[:8],
                            "confident": bool(per_word[0].get("fconfident"))}
            else:                                   # multi-word: just top per token
                parts = []
                for o in r:
                    if o.get("sep"):
                        parts.append(o.get("word", " "))
                    elif o.get("options"):
                        x = o["options"][0]
                        parts.append(x[0] if isinstance(x, list) else x)
                cache[w] = {"opts": ["".join(parts) or w], "confident": True}
        except Exception:
            cache[w] = {"opts": [w], "confident": False}
        if i % 40 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            print(f"[vocalize] nakdan {i}/{len(todo)}")
        time.sleep(0.25)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def _pick_hebrew(entry: dict, want: str | None):
    """(vocalised, pos_ok, confident) — for a noun/adj concept, skip an
    infinitive-shaped top option in favour of the first non-infinitive one."""
    opts = entry.get("opts") or [""]
    conf = entry.get("confident", False)
    if want in ("noun", "adj", "num"):
        for o in opts:
            if not _HE_INF.match(o) and "|" not in o:
                return o, o != opts[0] or not _HE_INF.match(opts[0]), conf
    return opts[0], want in (None, "verb"), conf


# ------------------------------------------------------------------ stage entry
def run(cfg: Config, n_jobs: int = 1) -> dict:
    interim = cfg.paths.resolve("interim")
    df = _data.load_raw(cfg)
    wikt = _wikt_lookup(cfg)
    pos = _load_pos(cfg)

    rows = []
    stats = {"Arabic": {}, "Hebrew": {}}
    conf_stats = {"Arabic": {}, "Hebrew": {}}
    he_cells = sorted({_strip(df.at[i, "Hebrew"], "Hebrew") for i in range(len(df))})
    he_cells = [c for c in he_cells if c and not wikt["Hebrew"].get(c)]
    nak = _nakdan(he_cells, interim / "nakdan_cache.json") if he_cells else {}
    ar_diac = _camel_diacritizer()

    for cid in range(len(df)):
        rec = {"concept_id": cid, "english": df.at[cid, "English"],
               "pos": pos.get(cid, "")}
        want = pos.get(cid)
        for lang in ("Arabic", "Hebrew"):
            old = str(df.at[cid, lang]).strip()
            bare = _strip(old, lang)
            if not bare:
                new, src, conf = old, "empty", "high"
            elif bare in wikt[lang]:
                new, src, conf = wikt[lang][bare], "wiktionary", "high"
            elif lang == "Arabic":
                new, pos_ok = ar_diac(bare, want)
                src = "camel"
                conf = "high" if (pos_ok and want) else ("med" if pos_ok else "low")
            else:
                new, pos_ok, ok_conf = _pick_hebrew(nak.get(bare, {"opts": [bare]}), want)
                src = "nakdan"
                conf = ("high" if (pos_ok and want and ok_conf)
                        else "med" if (pos_ok and (want or ok_conf))
                        else "low")
            rec[f"{lang}_old"] = old
            rec[f"{lang}_new"] = new
            rec[f"{lang}_source"] = src
            rec[f"{lang}_conf"] = conf
            stats[lang][src] = stats[lang].get(src, 0) + 1
            conf_stats[lang][conf] = conf_stats[lang].get(conf, 0) + 1
        rows.append(rec)

    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "vocalize"
    cols = ["concept_id", "english", "pos",
            "Arabic_old", "Arabic_new", "Arabic_source", "Arabic_conf",
            "Hebrew_old", "Hebrew_new", "Hebrew_source", "Hebrew_conf"]
    ws.append(cols)
    for r in sorted(rows, key=lambda x: (x["Arabic_conf"] != "low"
                                         and x["Hebrew_conf"] != "low", x["concept_id"])):
        ws.append([r.get(c, "") for c in cols])
    out = interim / "vocalized_ar_he.xlsx"
    try:
        wb.save(out)
    except PermissionError:
        out = interim / "vocalized_ar_he.NEW.xlsx"
        wb.save(out)
        print(f"[vocalize] target locked; wrote {out.name} instead")

    report = {"stage": "vocalize", "config": cfg.name,
              "config_fingerprint": cfg.fingerprint(),
              "n_concepts": len(rows), "by_source": stats, "by_confidence": conf_stats,
              "output": str(out.relative_to(cfg.paths.resolve("xlsx").parents[1]))}
    (interim / "vocalize.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    (interim / "vocalize.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report
