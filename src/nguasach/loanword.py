"""Internationalism / Wanderwort flags for a sensitivity analysis.

Concepts whose translation is a widely-borrowed, phonetically-adapted form
(chocolate -> chocolat -> Schokolade -> 巧克力) inflate cross-lingual form
similarity for a reason unrelated to sound symbolism. For same-script pairs the
orthographic / edit-distance controls absorb this; for cross-script pairs they
do not. This module flags such concepts two ways so results can be reported
with and without them:

1. **curated** -- ``data/raw/internationalisms.txt`` matched on the English column.
2. **data-driven** -- concepts in the top ``q`` quantile of *mean pairwise
   phonetic-embedding cosine across all languages* (script-independent, catches
   loans not on the list).

Writes ``data/interim/loanword_flags.csv``; ``flagged_ids`` is used by
``data.load_raw`` when ``cfg.exclude_loanwords`` is set.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache

import numpy as np

from .align import load_emb
from .config import ALL_LANGUAGES, Config


def _curated_set(cfg: Config) -> set[str]:
    path = cfg.paths.resolve("xlsx").parent / "internationalisms.txt"
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip().lower()
        line = re.sub(r"\s*\(.*?\)\s*$", "", line)   # drop "tablet (medicine)"
        if line:
            out.add(line)
    return out


def crosslingual_similarity(cfg: Config) -> np.ndarray:
    """Per-concept mean pairwise normalized Levenshtein similarity of the **IPA
    strings** across all languages (from ``<Lang>V.txt``). High = the same word
    in phonetic disguise everywhere -- the internationalism signature. The
    feature-bigram embeddings capture phonological texture, not word identity,
    so string similarity on the IPA is the right measure here."""
    import Levenshtein

    from . import data as _data

    interim = cfg.paths.resolve("interim")
    lj = json.loads((interim / "labels.json").read_text(encoding="utf-8"))
    n = len(_data.load_raw(cfg))

    ipa: dict[str, list[str]] = {}
    for lang in ALL_LANGUAGES:
        v = interim / f"{lang}V.txt"
        if not v.exists():
            continue
        by = {}
        for line in v.read_text(encoding="utf-8").splitlines():
            if "  " in line:
                lab, ph = line.split("  ", 1)
                by[lab] = ph.replace(" ", "")
        ipa[lang] = [by.get(lj[lang][c], "") if c < len(lj.get(lang, [])) else ""
                     for c in range(n)]

    langs = list(ipa)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    for cid in range(n):
        forms = [ipa[L][cid] for L in langs if ipa[L][cid]]
        for i in range(len(forms)):
            for j in range(i + 1, len(forms)):
                acc[cid] += Levenshtein.ratio(forms[i], forms[j])
                cnt[cid] += 1
    return acc / np.maximum(cnt, 1)


def run(cfg: Config, q: float = 0.95) -> dict:
    from . import data as _data

    df = _data.load_raw(cfg)
    interim = cfg.paths.resolve("interim")
    curated = _curated_set(cfg)
    eng = df["English"].str.strip().str.lower()

    sim = crosslingual_similarity(cfg)
    thr = float(np.quantile(sim, q))

    rows = []
    n_cur = n_dat = n_any = 0
    for cid in range(len(df)):
        in_cur = eng.iat[cid] in curated
        in_dat = bool(sim[cid] >= thr)
        flagged = in_cur or in_dat
        n_cur += in_cur
        n_dat += in_dat
        n_any += flagged
        rows.append({"concept_id": cid, "english": df.at[cid, "English"],
                     "xling_sim": round(float(sim[cid]), 4),
                     "in_curated": in_cur, "data_driven": in_dat, "flagged": flagged})

    with (interim / "loanword_flags.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    top = sorted(rows, key=lambda r: -r["xling_sim"])[:30]
    report = {
        "stage": "loanword", "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "n_concepts": len(df), "quantile": q, "xling_sim_threshold": round(thr, 4),
        "n_curated_matched": n_cur, "n_data_driven": n_dat, "n_flagged_total": n_any,
        "top_examples": [r["english"] for r in top],
    }
    (interim / "loanword_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (interim / "loanword.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report


@lru_cache(maxsize=4)
def flagged_ids(cfg: Config) -> frozenset[int]:
    path = cfg.paths.resolve("interim") / "loanword_flags.csv"
    if not path.exists():
        return frozenset()
    with path.open(encoding="utf-8", newline="") as fh:
        return frozenset(int(r["concept_id"]) for r in csv.DictReader(fh)
                         if r["flagged"] == "True")
