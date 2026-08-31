"""Stage ``lexibank-qc``: cross-check every translation against NorthEuraLex.

For each concept that :mod:`nguasach.concepticon` mapped to a Concepticon set,
and each of the 17 corpus languages NorthEuraLex covers, compare our cell to the
attested standard orthographic form(s) in NorthEuraLex (Dellert et al. 2020,
~107 languages, Concepticon-linked).

A **disagreement is not necessarily an error** -- NorthEuraLex records one
lexeme per concept, our cell may be a valid synonym, a different register, or an
inflected form. So the output is a *disagreement rate* per language plus the
list of disagreeing cells for human review, not a verdict.

Needs ``data/raw/northeuralex-cldf/{forms,parameters,languages}.csv`` (fetched
by :func:`fetch`) and ``data/interim/concepticon_map.csv`` (stage
``concepticon``).

Output: ``data/interim/lexibank_qc.csv`` + ``lexibank_qc.json`` (summary).
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

from .config import Config
from . import data as _data

_CLDF_URL = "https://raw.githubusercontent.com/lexibank/northeuralex/master/cldf/"
_FILES = ("forms.csv", "parameters.csv", "languages.csv")

# NorthEuraLex language name -> our column
_NEL_TO_OURS = {
    "Hungarian": "Hungarian", "Finnish": "Finnish", "Modern Greek": "Greek",
    "Russian": "Russian", "German": "German", "Spanish": "Spanish",
    "Italian": "Italian", "French": "French", "Irish": "Irish", "Welsh": "Welsh",
    "English": "English", "Mandarin Chinese": "Chinese", "Japanese": "Japanese",
    "Korean": "Korean", "Turkish": "Turkish", "Modern Hebrew": "Hebrew",
    "Hindi": "Hindi",
}
# corpus languages with no NorthEuraLex coverage
NOT_COVERED = ["Vietnamese", "Thai", "Indonesian", "Arabic", "Swahili"]


def fetch(raw_dir: Path) -> Path:
    dst = raw_dir / "northeuralex-cldf"
    dst.mkdir(parents=True, exist_ok=True)
    have = all((dst / f).exists() for f in _FILES)
    if not have:
        import urllib.request

        for f in _FILES:
            print(f"[lexibank-qc] fetching {f}")
            (dst / f).write_bytes(urllib.request.urlopen(_CLDF_URL + f, timeout=120).read())
    return dst


# --------------------------------------------------------------- normalisation
_LEAD = re.compile(r"^(to |the |a |an |der |die |das |el |la |le |les |il |lo )")
_PUNCT = re.compile(r"[.,;:!?()\[\]{}'\"/\\]")


def _t2s():
    try:
        from opencc import OpenCC

        return OpenCC("t2s").convert
    except Exception:
        return lambda s: s


_T2S = _t2s()


def _norm(s: str, lang: str = "") -> str:
    s = unicodedata.normalize("NFC", str(s or "")).strip().lower()
    # After NFC, real accented letters are precomposed; any *remaining* combining
    # mark (category Mn) is a dictionary annotation -- Russian/Greek stress,
    # Hebrew niqqud, Arabic harakat -- which our cells carry inconsistently.
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFC", s)
    s = _PUNCT.sub("", s)
    s = _LEAD.sub("", s).strip()
    if lang == "Chinese":
        s = _T2S(s)                       # fold traditional -> simplified both ways
    return re.sub(r"\s+", " ", s)


def _likely_error(ours: str, attested: set[str]) -> bool:
    """Heuristic: the cell looks like a Google-Translate failure, not a synonym.
    A phrase/sentence where the reference is a single word, a digit left in, or a
    cell far longer than every attested form (an explanation, not a lexeme)."""
    o = _norm(ours)
    if not o:
        return True
    if any(ch.isdigit() for ch in o):
        return True
    ref_max = max((len(_norm(a)) for a in attested), default=0)
    if len(o.split()) >= 3 and all(len(_norm(a).split()) <= 2 for a in attested):
        return True
    return ref_max and len(o) > ref_max * 2 + 4


def _agree(ours: str, attested: set[str], lang: str = "") -> bool:
    o = _norm(ours, lang)
    if not o:
        return False
    for a in attested:
        n = _norm(a, lang)
        if not n:
            continue
        if o == n or o in n or n in o:
            return True
        if set(o.split()) & set(n.split()):
            return True
    return False


# ------------------------------------------------------------------- attestations
def _attestations(cldf: Path) -> dict[tuple[str, str], set[str]]:
    """(concepticon_id, our_language) -> {attested orthographic forms}."""
    lang = {r["ID"]: r["Name"]
            for r in csv.DictReader((cldf / "languages.csv").read_text(encoding="utf-8").splitlines())}
    param = {r["ID"]: r["Concepticon_ID"]
             for r in csv.DictReader((cldf / "parameters.csv").read_text(encoding="utf-8").splitlines())}
    out: dict[tuple[str, str], set[str]] = {}
    for r in csv.DictReader((cldf / "forms.csv").read_text(encoding="utf-8").splitlines()):
        cc = param.get(r["Parameter_ID"])
        ours = _NEL_TO_OURS.get(lang.get(r["Language_ID"], ""))
        if cc and ours and r["Value"]:
            out.setdefault((cc, ours), set()).add(r["Value"])
    return out


# ------------------------------------------------------------------- stage entry
def run(cfg: Config, n_jobs: int = 1) -> dict:
    interim = cfg.paths.resolve("interim")
    raw_dir = cfg.paths.resolve("xlsx").parent
    cmap_path = interim / "concepticon_map.csv"
    if not cmap_path.exists():
        raise FileNotFoundError("run `nguasach run concepticon` first")

    cldf = fetch(raw_dir)
    att = _attestations(cldf)
    cc_by_cid = {int(r["concept_id"]): r["concepticon_id"]
                 for r in csv.DictReader(cmap_path.read_text(encoding="utf-8").splitlines())
                 if r["concepticon_id"]}

    df = _data.load_raw(cfg)
    ours_langs = [l for l in _NEL_TO_OURS.values() if l in df.columns]

    rows = []
    per_lang = {l: {"checked": 0, "match": 0, "differ": 0} for l in ours_langs}
    for cid in range(len(df)):
        cc = cc_by_cid.get(cid)
        if not cc:
            continue
        eng = df.at[cid, "English"]
        for lang in ours_langs:
            if lang == "English":
                continue
            attested = att.get((cc, lang))
            if not attested:
                continue
            cell = df.at[cid, lang]
            ok = _agree(cell, attested, lang)
            per_lang[lang]["checked"] += 1
            per_lang[lang]["match" if ok else "differ"] += 1
            if not ok:
                rows.append({
                    "concept_id": cid, "english": eng, "concepticon_id": cc,
                    "language": lang, "ours": cell,
                    "northeuralex": " / ".join(sorted(attested)),
                    "likely_error": _likely_error(cell, attested),
                })

    rows.sort(key=lambda r: (not r["likely_error"], r["language"], r["english"]))
    cols = ["concept_id", "english", "concepticon_id", "language", "ours",
            "northeuralex", "likely_error"]
    with (interim / "lexibank_qc.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[k] for k in cols])

    for l, d in per_lang.items():
        d["disagree_rate"] = round(d["differ"] / d["checked"], 3) if d["checked"] else None
        d["likely_error"] = sum(1 for r in rows if r["language"] == l and r["likely_error"])
    verified = [l for l in ("French", "Irish") if l in per_lang and per_lang[l]["checked"]]
    baseline = (round(sum(per_lang[l]["differ"] for l in verified)
                      / sum(per_lang[l]["checked"] for l in verified), 3)
                if verified else None)
    summary = {
        "stage": "lexibank-qc", "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "source": "NorthEuraLex (lexibank/northeuralex CLDF)",
        "n_concepts_with_concepticon": len(cc_by_cid),
        "languages_covered": [l for l in ours_langs if l != "English"],
        "languages_not_covered": NOT_COVERED,
        "total_checked": sum(d["checked"] for d in per_lang.values()),
        "total_disagree": len(rows),
        "total_likely_error": sum(1 for r in rows if r["likely_error"]),
        "verified_baseline_disagree_rate": baseline,
        "per_language": per_lang,
        "note": ("A disagreement is a lexical-choice difference from the single "
                 "NorthEuraLex reference lexeme -- synonyms, register and "
                 "inflection all count. The hand-verified French/Irish columns "
                 f"disagree at {baseline} here, so treat that as the floor; "
                 "'likely_error' rows (phrase-for-word, digits, explanations) "
                 "are the automatable Google-Translate failures worth fixing "
                 "first. Review data/interim/lexibank_qc.csv."),
    }
    (interim / "lexibank_qc.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    (interim / "lexibank-qc.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return summary
