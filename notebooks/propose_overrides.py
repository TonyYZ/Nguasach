"""Build a review sheet of mechanical translation fixes from the QC output.

    python notebooks/propose_overrides.py

Reads the ``lexibank-qc`` / ``wiktionary-qc`` outputs + the corpus and proposes
a correction for each cell in a few automatically-recognisable failure classes
(Google left the English word, "για να ..." purpose clauses, Korean
sentence-glosses, script homoglyphs). Writes
``data/interim/override_batch2_proposed.xlsx`` for a human to check the
``final`` column, then feed to ``apply_overrides.py``.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import openpyxl

from nguasach.config import Config
from nguasach import data as _data
from nguasach.lexibank_qc import _attestations, _norm

CFG = Config.load("configs/paper_exploratory.yaml")
INTERIM = Path("data/interim")
NONLATIN = {"Hindi", "Arabic", "Hebrew", "Korean", "Japanese", "Thai",
            "Chinese", "Greek", "Russian"}
GR_VERB = re.compile(r"(ω|ώ|μαι|άω|άει)$")
ASCII_ALPHA = re.compile(r"^[A-Za-z][A-Za-z .-]*$")
KO_SENT = re.compile(r"(습니다|입니다|ㅂ니다|는다|ㄴ다)$|그것은")


def _ko_dictform(cell: str) -> str:
    """Turn a Google 'sentence' gloss into a citation form by ending, not by
    dictionary lookup (which mixes senses): copula -> the noun, -합니다 -> -하다,
    -습니다 -> -다."""
    s = re.sub(r"^그것은\s*", "", cell).strip()
    s = re.sub(r"([가-힣])\s+([럽롭])", r"\1\2", s)   # 시끄 럽습니다 -> 시끄럽습니다
    if s.endswith(("입니다", "이다")):
        return s[:-3] if s.endswith("입니다") else s[:-2]
    if s.endswith("합니다"):
        return s[:-3] + "하다"
    if s.endswith(("습니다", "ㅂ니다")):
        return s[:-3] + "다"
    if s.endswith("해"):                       # casual 이상해 -> 이상하다
        return s[:-1] + "하다"
    if s.endswith(("하다", "되다")) and " " not in s:   # already a citation form
        return s
    return ""
_CYR_MAP = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с",
                          "y": "у", "x": "х", "A": "А", "E": "Е", "O": "О",
                          "P": "Р", "C": "С", "H": "Н", "T": "Т", "B": "В",
                          "K": "К", "M": "М"})


def main() -> None:
    df = _data.load_raw(CFG)
    verified = set(CFG.verified_core)
    differ_both = {
        (int(r["concept_id"]), r["language"])
        for r in csv.DictReader((INTERIM / "translation_qc_combined.csv").read_text("utf-8").splitlines())
        if r["verdict"] == "differ_both"
    }
    cc_by_cid = {
        int(r["concept_id"]): r["concepticon_id"]
        for r in csv.DictReader((INTERIM / "concepticon_map.csv").read_text("utf-8").splitlines())
        if r["concepticon_id"]
    }
    nel = _attestations(Path("data/raw/northeuralex-cldf"))
    wik = json.loads((INTERIM / "wiktionary_cache.json").read_text("utf-8"))

    def nel_forms(cid, lang):
        cc = cc_by_cid.get(cid)
        return sorted(nel.get((cc, lang), [])) if cc else []

    def wik_forms(cid, lang):
        return wik.get(str(df.at[cid, "English"]).strip(), {}).get(lang, [])

    def pick(cid, lang, pos=None):
        nf, wf = nel_forms(cid, lang), wik_forms(cid, lang)
        if pos == "gr-verb":
            for pool, tag in ((nf, "nel"), (wf, "wikt")):
                for f in pool:
                    if GR_VERB.search(f):
                        return f, tag
        if nf:
            return nf[0], "nel"
        if wf:
            return wf[0], "wikt"
        return "", ""

    props = []
    for cid in range(len(df)):
        eng = str(df.at[cid, "English"]).strip()
        for lang in _data.ALL_LANGUAGES:
            if lang in verified or lang == "English":
                continue
            cell = str(df.at[cid, lang]).strip()
            if not cell:
                continue
            cat = pv = src = None
            if lang in NONLATIN and ASCII_ALPHA.match(cell):
                cat, (pv, src) = "untranslated-nonlatin", pick(cid, lang)
            elif _norm(cell) == _norm(eng) and (cid, lang) in differ_both:
                cat, (pv, src) = "untranslated-differ-both", pick(cid, lang)
            elif lang == "Greek" and (cell.startswith("να ") or cell.startswith("για να ")):
                cat, (pv, src) = "greek-na-verb", pick(cid, lang, "gr-verb")
                if not pv:
                    pv, src = re.sub(r"^(για )?να ", "", cell), "rule"
            elif lang == "Korean" and KO_SENT.search(cell):
                cat = "korean-sentence"
                pv = _ko_dictform(cell)
                src = "rule" if pv else ""
                if not pv:
                    pv, src = pick(cid, lang)
            elif re.search(r"[а-яА-Я]", cell) and re.search(r"[a-zA-Z]", cell):
                cat, src = "homoglyph", "rule"
                pv = cell.translate(_CYR_MAP)
            if not cat:
                continue
            review = ("" if (src in ("nel", "rule") and pv)
                      else "fill manually (no ref)" if not pv
                      else "from wiktionary - glance")
            props.append({"concept_id": cid, "english": eng, "language": lang,
                          "old": cell, "proposed": pv or "", "source": src or "",
                          "category": cat, "review": review, "final": ""})

    print(f"{len(props)} proposals  |  by category: {dict(Counter(p['category'] for p in props))}")
    print(f"review flags: {dict(Counter(p['review'] or 'clean' for p in props))}")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "batch2"
    cols = ["concept_id", "english", "language", "old", "proposed", "source",
            "category", "review", "final"]
    ws.append(cols)
    for p in sorted(props, key=lambda x: (x["review"] == "", x["category"],
                                          x["language"], x["english"])):
        ws.append([p[c] for c in cols])
    out = INTERIM / "override_batch2_proposed.xlsx"
    wb.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
