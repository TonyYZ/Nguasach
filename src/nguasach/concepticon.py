"""Stage ``concepticon``: map every concept to a Concepticon concept set.

Concepticon (https://concepticon.clld.org) is the standard catalogue of
normalised lexical meanings; the community wordlists (Swadesh, Leipzig-Jakarta,
...) are all expressed against it. Attaching a Concepticon ID to each of our
1,842 concepts gives:

* stable identifiers to join against Lexibank / NorthEuraLex for translation QC
  (:mod:`nguasach.lexibank_qc`), and
* an exact, catalogue-defined membership test for the Swadesh and
  Leipzig-Jakarta strata (replacing the hand-built ``data/raw/*.txt`` lists),
  including *which* canonical entries our concept set is missing.

Needs the ``concepticon-data`` repo at ``data/raw/concepticon-data`` (a ~30 MB
shallow clone); :func:`fetch` does it. Uses ``pyconcepticon``'s fuzzy matcher.

Output: ``data/interim/concepticon_map.csv`` +
``data/interim/concepticon_gaps.json`` (canonical Swadesh / LJ concepts with no
match in our list).
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from .config import Config
from . import data as _data

_REPO_URL = "https://github.com/concepticon/concepticon-data.git"
_SWADESH = "Swadesh-1955-215"
_LEIPZIG_JAKARTA = "Tadmor-2009-100"


def fetch(raw_dir: Path) -> Path:
    dst = raw_dir / "concepticon-data"
    if not dst.exists():
        print(f"[concepticon] shallow-cloning {_REPO_URL}")
        subprocess.run(["git", "clone", "--depth", "1", _REPO_URL, str(dst)],
                       check=True)
    return dst


def _api(repo: Path):
    from pyconcepticon import Concepticon

    return Concepticon(str(repo))


def _parse_row(lookup_row, term: str, max_sim: int = 4):
    """``Concepticon.lookup`` yields a set of (term, id, gloss, similarity).
    Return (best_id, best_gloss, best_sim, {plausible_ids}). ``best`` is the
    exact-gloss hit if there is one, else the lowest-similarity hit; the id set
    keeps everything at similarity <= ``max_sim`` for the membership test."""
    t = term.strip().lower()
    best = None
    ids: set[str] = set()
    for _, cid, gloss, sim in lookup_row:
        s = int(getattr(sim, "value", sim))
        if s <= max_sim:
            ids.add(cid)
        exact = gloss.strip().lower() == t
        rank = (0 if exact else 1, s)
        if best is None or rank < best[0]:
            best = (rank, cid, gloss, s)
    if best is None:
        return "", "", 99, set()
    return best[1], best[2], best[3], ids


def run(cfg: Config, n_jobs: int = 1) -> dict:
    raw_dir = cfg.paths.resolve("xlsx").parent
    interim = cfg.paths.resolve("interim")
    interim.mkdir(parents=True, exist_ok=True)
    repo = fetch(raw_dir)
    api = _api(repo)

    swa_ids = {c.concepticon_id for c in api.conceptlists[_SWADESH].concepts.values()
               if c.concepticon_id}
    ljk = api.conceptlists[_LEIPZIG_JAKARTA].concepts
    ljk_ids = {c.concepticon_id for c in ljk.values() if c.concepticon_id}
    ljk_gloss = {c.concepticon_id: (c.concepticon_gloss or c.english)
                 for c in ljk.values() if c.concepticon_id}
    swa_gloss = {c.concepticon_id: (c.concepticon_gloss or c.english)
                 for c in api.conceptlists[_SWADESH].concepts.values()
                 if c.concepticon_id}

    df = _data.load_raw(cfg)
    glosses = [str(df.at[i, "English"]).strip() for i in range(len(df))]
    # lookup is a generator of one result-set per input term, in order
    looked = list(api.lookup(glosses, language="en", full_search=True))

    rows, matched = [], set()
    for cid, (eng, res) in enumerate(zip(glosses, looked)):
        cc_id, cc_gloss, sim, cand = _parse_row(res, eng)
        matched |= cand
        rows.append({
            "concept_id": cid, "english": eng,
            "concepticon_id": cc_id, "concepticon_gloss": cc_gloss,
            "similarity": sim if cc_id else "",
            "on_swadesh215": bool(cand & swa_ids),
            "on_leipzig_jakarta": bool(cand & ljk_ids),
        })

    with (interim / "concepticon_map.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(rows[0]))
        for r in rows:
            w.writerow(list(r.values()))

    swa_missing = [{"concepticon_id": i, "gloss": swa_gloss.get(i, "")}
                   for i in sorted(swa_ids - matched, key=lambda x: int(x))]
    ljk_missing = [{"concepticon_id": i, "gloss": ljk_gloss.get(i, "")}
                   for i in sorted(ljk_ids - matched, key=lambda x: int(x))]
    (interim / "concepticon_gaps.json").write_text(json.dumps(
        {"swadesh215_missing": swa_missing, "leipzig_jakarta_missing": ljk_missing},
        ensure_ascii=False, indent=2), encoding="utf-8")

    n_map = sum(1 for r in rows if r["concepticon_id"])
    strong = sum(1 for r in rows if r["concepticon_id"] and r["similarity"] <= 2)
    report = {
        "stage": "concepticon", "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "n_concepts": len(rows), "n_mapped": n_map,
        "n_strong": strong, "map_rate": round(n_map / len(rows), 3),
        "swadesh215": {"total": len(swa_ids), "covered": len(swa_ids & matched),
                       "missing": len(swa_missing)},
        "leipzig_jakarta": {"total": len(ljk_ids), "covered": len(ljk_ids & matched),
                            "missing": len(ljk_missing)},
    }
    (interim / "concepticon.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report
