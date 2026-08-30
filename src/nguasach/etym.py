"""Stage ``etym``: the Mandarin syllable x etym-structure table.

Rebuilds the original project's ``etymTable`` idea against the current pipeline.
The grid is Mandarin **onset (聲母) x rhyme (韻母)**; each cell collects the
concepts whose Chinese translation contains that syllable, and annotates the
cell's *mean* meaning vector with:

* its nearest single trigram pole  (九卦 x 靜/動 -> 18 poles, from hexLabels.yaml)
* its nearest **parallel composition** -- the mean of two pole anchors, over all
  C(18,2)+18 combinations (additive "glue")
* the nearest word2vec neighbours of the cell mean
* per-cell phoneme skew (which phones are over-represented vs. the whole grid)

Outputs ``results/etym_table.json`` (nested onset->rhyme->cell) and a flat
``results/etym_table.csv``. Browse it with the etym visualiser artifact.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations_with_replacement

import numpy as np

from ._zh import pinyin_map, purify_chn
from .align import load_emb
from .config import Config
from . import data as _data

_VOWELS = set("iaeouyɑɯɤɔə")


def _syllable_onset_rhyme(ipa: str) -> tuple[str, str]:
    """Split a purified pinyin-IPA syllable ('ʈ͡ʂʰ ə ŋ') -> (onset, rhyme)."""
    parts = ipa.split()
    if not parts:
        return "", ""
    if parts[0][0] in _VOWELS:
        return "∅", "".join(parts)
    return parts[0], "".join(parts[1:]) or "∅"


def _pinyin_syllables(hanzi: str):
    import sys
    from pathlib import Path

    tp = str(Path(__file__).resolve().parents[2] / "third_party")
    if tp not in sys.path:
        sys.path.insert(0, tp)
    from hanzi2pinyin.Converter import Converter

    conv = Converter()
    for syl in conv.convert(hanzi.strip()).split(","):
        base = syl[:-1] if syl and syl[-1].isdigit() else syl
        if base in pinyin_map:
            yield base, purify_chn(pinyin_map[base])


def _pole_anchors(cfg: Config):
    import yaml

    poles = yaml.safe_load(cfg.paths.resolve("hex_labels").read_text(encoding="utf-8"))["poles"]
    labels, mat = load_emb(cfg.paths.resolve("processed") / "SemanticsEmb.txt")
    idx = {lab: i for i, lab in enumerate(labels)}
    names, glosses, vecs = [], [], []
    for p in poles:
        rows = [mat[idx[f'{w}_']] for w in p["words"] if f'{w}_' in idx]
        if rows:
            names.append(p["name"])
            glosses.append(p.get("gloss", ""))
            vecs.append(np.mean(rows, axis=0))
    v = np.asarray(vecs, np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
    # parallel (additive) compositions: mean of every unordered pole pair (+ singles)
    pair_names, pair_vecs = [], []
    for i, j in combinations_with_replacement(range(len(names)), 2):
        pair_names.append(names[i] if i == j else f"{names[i]} + {names[j]}")
        pair_vecs.append((v[i] + v[j]) / 2)
    pv = np.asarray(pair_vecs, np.float32)
    pv /= np.linalg.norm(pv, axis=1, keepdims=True).clip(min=1e-12)
    return names, glosses, v, pair_names, pv


def run(cfg: Config, n_jobs: int = 1) -> dict:
    rdir = cfg.paths.resolve("results")
    rdir.mkdir(parents=True, exist_ok=True)
    df = _data.load_raw(cfg)
    sem_keys = json.loads((cfg.paths.resolve("interim") / "semantics_keys.json").read_text(encoding="utf-8"))
    labels, mat = load_emb(cfg.paths.resolve("processed") / "SemanticsEmb.txt")
    sidx = {lab: i for i, lab in enumerate(labels)}
    names, glosses, anch, pair_names, pair_anch = _pole_anchors(cfg)

    # grid[rhyme][onset] -> list of concept dicts
    grid: dict[str, dict[str, list[dict]]] = {}
    onsets, rhymes = set(), set()
    all_phones = Counter()
    for cid in range(len(df)):
        han = df.at[cid, "Chinese"]
        eng = df.at[cid, "English"]
        key = sem_keys[cid]
        vec = mat[sidx[key]] if key in sidx else None
        for py, ipa in _pinyin_syllables(han):
            onset, rhyme = _syllable_onset_rhyme(ipa)
            if not rhyme:
                continue
            onsets.add(onset)
            rhymes.add(rhyme)
            all_phones.update(ipa.split())
            grid.setdefault(rhyme, {}).setdefault(onset, []).append(
                {"hanzi": han, "english": eng, "pinyin": py,
                 "ipa": ipa, "cid": cid,
                 "vec": None if vec is None else vec}
            )

    total_ph = sum(all_phones.values()) or 1
    global_rate = {p: c / total_ph for p, c in all_phones.items()}

    cells = []
    table: dict[str, dict[str, dict]] = {}
    for rhyme in sorted(rhymes):
        for onset in sorted(grid.get(rhyme, {})):
            items = grid[rhyme][onset]
            vecs = [it["vec"] for it in items if it["vec"] is not None]
            entry = {
                "onset": onset, "rhyme": rhyme, "syllable": f"{'' if onset=='∅' else onset}{rhyme}",
                "n": len(items),
                "words": [{"hanzi": it["hanzi"], "english": it["english"], "pinyin": it["pinyin"]}
                          for it in items[:40]],
            }
            if vecs:
                m = np.mean(vecs, axis=0)
                m = m / (np.linalg.norm(m) or 1)
                sp = anch @ m
                spp = pair_anch @ m
                order = np.argsort(-sp)[:3]
                entry["pole"] = [{"name": names[k], "gloss": glosses[k],
                                  "score": round(float(sp[k]), 3)} for k in order]
                pk = int(np.argmax(spp))
                entry["parallel_pole"] = {"name": pair_names[pk], "score": round(float(spp[pk]), 3)}
                nbr = mat @ m
                entry["neighbors"] = [labels[k].rstrip("_")
                                      for k in np.argsort(-nbr)[:12] if labels[k].rstrip("_")]
            # phoneme skew for this cell
            cph = Counter(p for it in items for p in it["ipa"].split())
            ct = sum(cph.values()) or 1
            skew = sorted(((p, cph[p] / ct - global_rate.get(p, 0)) for p in cph),
                          key=lambda x: -x[1])
            entry["phoneme_skew"] = [{"p": p, "delta": round(d, 3)} for p, d in skew[:6]]
            table.setdefault(onset, {})[rhyme] = entry
            cells.append(entry)

    out = {
        "stage": "etym", "config": cfg.name, "config_fingerprint": cfg.fingerprint(),
        "onsets": sorted(onsets), "rhymes": sorted(rhymes),
        "n_cells": len(cells),
        "pole_names": names, "pole_glosses": glosses,
        "poles": _pole_phoneme_profiles(cfg, names),
        "table": table,
    }
    (rdir / "etym_table.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    _csv(rdir / "etym_table.csv", cells)
    (cfg.paths.resolve("interim") / "etym.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return {k: out[k] for k in ("stage", "config", "n_cells", "onsets", "rhymes")}


def _pole_phoneme_profiles(cfg: Config, pole_names: list[str]) -> dict:
    """Per trigram pole: which Chinese phonemes it over- / under-uses.

    The original project's intent -- assign every Chinese concept to its nearest
    pole (ridge phonetic->semantic projection, as in association.py), tally
    phonemes per pole, z-score each phoneme's rate across the 18 poles. None
    clear FDR at this scale, so the raw z-scores are the browsable artefact.
    """
    from . import association as A

    spec = A.load_poles(cfg)
    names, anchors = A.pole_anchors(cfg, spec)
    pole_of, concepts = A.assign_poles(cfg, "Chinese", anchors)
    phones = A.phoneme_rows(cfg, "Chinese")
    z, vocab, counts = A.zscores(pole_of, concepts, phones, len(names))
    sizes = np.bincount(pole_of, minlength=len(names))
    gloss = {p["name"]: p.get("gloss", "") for p in spec}
    words = {p["name"]: p["words"] for p in spec}

    prof = {}
    for i, nm in enumerate(names):
        row = z[i]
        over = [{"p": vocab[j], "z": round(float(row[j]), 2), "n": int(counts[i, j])}
                for j in np.argsort(-row) if counts[i, j] >= 3 and row[j] > 0][:8]
        under = [{"p": vocab[j], "z": round(float(row[j]), 2), "n": int(counts[i, j])}
                 for j in np.argsort(row) if row[j] < 0][:8]
        prof[nm] = {"gloss": gloss.get(nm, ""), "n_concepts": int(sizes[i]),
                    "seed_words": words.get(nm, [])[:14], "over": over, "under": under}
    return prof


def _csv(path, cells: list[dict]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["syllable", "onset", "rhyme", "n", "pole_1", "pole_2", "pole_3",
                    "parallel_pole", "phoneme_skew", "neighbors", "example_words"])
        for c in cells:
            pole = c.get("pole", [])
            w.writerow([
                c["syllable"], c["onset"], c["rhyme"], c["n"],
                pole[0]["name"] if len(pole) > 0 else "",
                pole[1]["name"] if len(pole) > 1 else "",
                pole[2]["name"] if len(pole) > 2 else "",
                c.get("parallel_pole", {}).get("name", ""),
                " ".join(f"{s['p']}{s['delta']:+.2f}" for s in c.get("phoneme_skew", [])),
                " ".join(c.get("neighbors", [])[:8]),
                " ".join(f"{x['hanzi']}({x['english']})" for x in c["words"][:12]),
            ])
