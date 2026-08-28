"""Stage ``associate``: which phonemes cluster with which meanings.

The interpretability layer. Port of ``transPhone.generateHex`` /
``generateHexVectors`` / ``getZScore``:

1. 18 semantic poles (``data/raw/hexLabels.yaml``); each pole's anchor = mean of
   its seed-word vectors in the compressed semantic space (``SemanticsEmb.txt``).
2. For a language L, fit L(phonetic) -> Semantics on **all** concepts (a
   descriptive projection, not a held-out prediction), then assign each concept
   to its nearest pole.
3. Per pole, tally phoneme frequencies from ``<L>V.txt`` and z-score each
   phoneme's rate across the 18 poles.

A label-permutation null (shuffle the concept<->pole assignment) with BH-FDR
across phoneme x pole cells is applied in :func:`run` when ``cfg.null_iters``.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import yaml

from .align import load_emb, make_map
from .config import Config
from . import crossval
from . import data as _data


def load_poles(cfg: Config) -> list[dict]:
    return yaml.safe_load(cfg.paths.resolve("hex_labels").read_text(encoding="utf-8"))["poles"]


def pole_anchors(cfg: Config, poles: list[dict]) -> tuple[list[str], np.ndarray]:
    labels, mat = load_emb(cfg.paths.resolve("processed") / "SemanticsEmb.txt")
    idx = {lab: i for i, lab in enumerate(labels)}
    names, vecs = [], []
    for pole in poles:
        rows = [mat[idx[f"{w}_"]] for w in pole["words"] if f"{w}_" in idx]
        if rows:
            names.append(pole["name"])
            vecs.append(np.mean(rows, axis=0))
    v = np.asarray(vecs, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-12)
    return names, v


def assign_poles(cfg: Config, lang: str, anchor_vecs: np.ndarray) -> np.ndarray:
    """Concept -> pole index, via a full-fit L(phonetic)->Semantics projection."""
    pd = crossval.load_pair_data(cfg, lang, "Semantics")
    model = make_map(cfg.map, cfg.ridge_alpha).fit(pd.xs, pd.xt)
    proj = model.predict(pd.xs)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True).clip(min=1e-12)
    return np.argmax(proj @ anchor_vecs.T, axis=1), pd.concepts


def phoneme_rows(cfg: Config, lang: str) -> dict[int, list[str]]:
    """concept_id -> list of phones, from data/interim/<lang>V.txt."""
    lj = json.loads((cfg.paths.resolve("interim") / "labels.json").read_text(encoding="utf-8"))
    label_to_cid = {lab: i for i, lab in enumerate(lj[lang])}
    out: dict[int, list[str]] = {}
    for line in (cfg.paths.resolve("interim") / f"{lang}V.txt").read_text(encoding="utf-8").splitlines():
        if "  " not in line:
            continue
        lab, phones = line.split("  ", 1)
        if lab in label_to_cid:
            out[label_to_cid[lab]] = phones.split()
    return out


def zscores(pole_of: np.ndarray, concepts: np.ndarray, phones: dict[int, list[str]],
            n_poles: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(z[pole, phoneme], phoneme_list, count[pole, phoneme])."""
    freq = [Counter() for _ in range(n_poles)]
    for local, cid in enumerate(concepts):
        for ph in phones.get(int(cid), []):
            freq[pole_of[local]][ph] += 1
    vocab = sorted({p for c in freq for p in c})
    counts = np.array([[c.get(p, 0) for p in vocab] for c in freq], dtype=float)
    totals = counts.sum(axis=1, keepdims=True).clip(min=1)
    rate = counts / totals
    mu = rate.mean(axis=0, keepdims=True)
    sd = rate.std(axis=0, ddof=0, keepdims=True)
    z = np.divide(rate - mu, sd, out=np.zeros_like(rate), where=sd > 0)
    return z, vocab, counts


def run(cfg: Config, n_jobs: int = 1) -> dict:
    from .stats import bh_fdr, empirical_p

    results_dir = cfg.paths.resolve("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    poles = load_poles(cfg)
    names, anchors = pole_anchors(cfg, poles)
    n_poles = len(names)

    rows = []
    per_lang = {}
    rng = np.random.default_rng(cfg.seed)
    for lang in cfg.languages:
        pole_of, concepts = assign_poles(cfg, lang, anchors)
        phones = phoneme_rows(cfg, lang)
        z, vocab, counts = zscores(pole_of, concepts, phones, n_poles)

        null_iters = cfg.null_iters
        null_abs = np.zeros((null_iters, *z.shape))
        for it in range(null_iters):
            perm = rng.permutation(len(pole_of))
            zn, _, _ = zscores(pole_of[perm], concepts, phones, n_poles)
            null_abs[it] = np.abs(zn)

        p = np.empty_like(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                p[i, j] = empirical_p(abs(z[i, j]), null_abs[:, i, j])
        q = bh_fdr(p.ravel()).reshape(p.shape)

        pole_sizes = np.bincount(pole_of, minlength=n_poles).tolist()
        n_sig = 0
        for i, pole in enumerate(names):
            for j, ph in enumerate(vocab):
                if counts[i, j] < 3:
                    continue
                sig = bool(q[i, j] < 0.10)
                n_sig += sig
                rows.append({
                    "language": lang, "pole": pole, "phoneme": ph,
                    "z": round(float(z[i, j]), 3), "count": int(counts[i, j]),
                    "p_perm": round(float(p[i, j]), 4), "q_fdr": round(float(q[i, j]), 4),
                    "significant": sig,
                })
        per_lang[lang] = {
            "pole_sizes": dict(zip(names, pole_sizes)),
            "n_phonemes": len(vocab), "n_significant_cells": n_sig,
        }

    rows.sort(key=lambda r: (r["language"], not r["significant"], r["q_fdr"], -abs(r["z"])))
    out = {
        "stage": "associate", "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "n_poles": n_poles, "null_iters": cfg.null_iters,
        "per_language": per_lang,
        "n_significant_total": sum(v["n_significant_cells"] for v in per_lang.values()),
        "cells": rows,
    }
    (results_dir / "association_z.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_csv(results_dir / "association_z.csv", rows)
    (cfg.paths.resolve("interim") / "associate.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return out


def _write_csv(path, rows: list[dict]) -> None:
    import csv

    cols = ["language", "pole", "phoneme", "z", "count", "p_perm", "q_fdr", "significant"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
