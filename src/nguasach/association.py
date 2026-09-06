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
from .config import LANGUAGE_FAMILY, Config
from . import crossval
from . import data as _data


def _macro_family(lang: str) -> str:
    """Collapse the Indo-European branches to one unit -- the relevant grain for
    treating languages as independent evidence in the pooled test."""
    fam = LANGUAGE_FAMILY.get(lang, "?")
    return "Indo-European" if fam.startswith("IE-") else fam


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
    """Concept -> pole index, via a full-fit L(phonetic)->Semantics projection.

    The pole assignment is on the *projected sound* (that is the thing under
    test); the optional ``pole_margin_quantile`` filter is on the concept's
    *actual meaning* -- keep only concepts whose semantic vector is genuinely
    near some pole, so an iconic-pole run isn't diluted by everyday concepts
    that belong to no pole."""
    pd = crossval.load_pair_data(cfg, lang, "Semantics")
    model = make_map(cfg.map, cfg.ridge_alpha).fit(pd.xs, pd.xt)
    proj = model.predict(pd.xs)
    proj /= np.linalg.norm(proj, axis=1, keepdims=True).clip(min=1e-12)
    pole_of = np.argmax(proj @ anchor_vecs.T, axis=1)
    concepts = pd.concepts
    q = getattr(cfg, "pole_margin_quantile", 0.0)
    if q > 0.0:
        sem = pd.xt / np.linalg.norm(pd.xt, axis=1, keepdims=True).clip(min=1e-12)
        near = (sem @ anchor_vecs.T).max(axis=1)
        keep = near >= np.quantile(near, q)
        pole_of, concepts = pole_of[keep], concepts[keep]
    return pole_of, concepts


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
    null_iters = cfg.null_iters

    rows = []
    per_lang = {}
    rng = np.random.default_rng(cfg.seed)
    # per-language observed + null z, kept on each language's own vocab, so the
    # cross-linguistic pooled test below can align them onto a shared phoneme set.
    stash: list[dict] = []
    for lang in cfg.languages:
        pole_of, concepts = assign_poles(cfg, lang, anchors)
        phones = phoneme_rows(cfg, lang)
        z, vocab, counts = zscores(pole_of, concepts, phones, n_poles)

        perms = [rng.permutation(len(pole_of)) for _ in range(null_iters)]
        null_z = np.zeros((null_iters, *z.shape))
        for it, perm in enumerate(perms):
            null_z[it] = zscores(pole_of[perm], concepts, phones, n_poles)[0]
        null_abs = np.abs(null_z)

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
        stash.append({"lang": lang, "z": z, "null_z": null_z,
                      "counts": counts, "vocab": vocab})

    rows.sort(key=lambda r: (r["language"], not r["significant"], r["q_fdr"], -abs(r["z"])))
    gloss = {p["name"]: p.get("gloss", "") for p in poles}
    pooled = _pooled_test(stash, names, gloss, null_iters, cfg.seed,
                          min_langs=getattr(cfg, "associate_min_langs", 5),
                          min_count=3)
    out = {
        "stage": "associate", "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "n_poles": n_poles, "null_iters": cfg.null_iters,
        "per_language": per_lang,
        "n_significant_total": sum(v["n_significant_cells"] for v in per_lang.values()),
        "cells": rows,
        "pooled": pooled,
    }
    (results_dir / "association_z.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_csv(results_dir / "association_z.csv", rows)
    _write_pooled_csv(results_dir / "association_pooled.csv", pooled["cells"])
    (cfg.paths.resolve("interim") / "associate.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return out


def _pooled_test(stash: list[dict], names: list[str], gloss: dict, null_iters: int,
                 seed: int, min_langs: int, min_count: int) -> dict:
    """Cross-linguistic pooled phoneme x pole association.

    For every (pole, phoneme) the per-language z-scores are averaged over the
    languages where that cell is attested (count >= ``min_count``); cells seen in
    fewer than ``min_langs`` languages are dropped. The null re-uses each
    language's own concept->pole permutations and averages them the same way, so
    a weak bias shared across unrelated languages accumulates while
    language-idiosyncratic noise cancels. BH-FDR across the surviving cells --
    one test family, ~n_poles x |shared vocab|, far smaller than the per-language
    grids summed.
    """
    from .stats import bh_fdr, empirical_p

    n_poles = len(names)
    vocab = sorted({p for s in stash for p in s["vocab"]})
    vpos = {p: j for j, p in enumerate(vocab)}
    V = len(vocab)
    fams = sorted({_macro_family(s["lang"]) for s in stash})
    fpos = {f: i for i, f in enumerate(fams)}

    # accumulate the pooled mean incrementally so memory stays O(iters x pole x V)
    obs_sum = np.zeros((n_poles, V))
    ncnt = np.zeros((n_poles, V))                    # languages contributing per cell
    tot_count = np.zeros((n_poles, V))
    null_sum = np.zeros((null_iters, n_poles, V))
    fam_sum = np.zeros((len(fams), n_poles, V))      # within-family sum of z
    fam_cnt = np.zeros((len(fams), n_poles, V))
    for s in stash:
        cols = np.array([vpos[p] for p in s["vocab"]])
        att = s["counts"] >= min_count               # (pole, vL) attested this language
        obs_sum[:, cols] += np.where(att, s["z"], 0.0)
        ncnt[:, cols] += att
        tot_count[:, cols] += s["counts"]
        null_sum[:, :, cols] += np.where(att[None], s["null_z"], 0.0)
        fi = fpos[_macro_family(s["lang"])]
        fam_sum[fi][:, cols] += np.where(att, s["z"], 0.0)
        fam_cnt[fi][:, cols] += att

    k = ncnt
    keep = k >= min_langs
    denom = np.where(ncnt > 0, ncnt, 1)
    zbar = obs_sum / denom                            # (pole, V)
    zbar_null = null_sum / denom[None]                # (iters, pole, V)
    fam_mean = np.divide(fam_sum, fam_cnt, out=np.full_like(fam_sum, np.nan),
                         where=fam_cnt > 0)           # (family, pole, V)

    cells = []
    p_list, idx = [], []
    for i in range(n_poles):
        for j in range(V):
            if not keep[i, j]:
                continue
            pv = empirical_p(abs(zbar[i, j]), np.abs(zbar_null[:, i, j]))
            idx.append((i, j))
            p_list.append(pv)
    q_list = bh_fdr(np.array(p_list)) if p_list else np.array([])
    for (i, j), pv, qv in zip(idx, p_list, q_list):
        fm = fam_mean[:, i, j]
        seen = ~np.isnan(fm)
        concord = (float(np.mean(np.sign(fm[seen]) == np.sign(zbar[i, j])))
                   if seen.any() else 0.0)
        cells.append({
            "pole": names[i], "gloss": gloss.get(names[i], ""), "phoneme": vocab[j],
            "mean_z": round(float(zbar[i, j]), 3),
            "n_langs": int(k[i, j]), "n_families": int(seen.sum()),
            "family_sign_concord": round(concord, 2),
            "total_count": int(tot_count[i, j]),
            "p_perm": round(float(pv), 4), "q_fdr": round(float(qv), 4),
            "significant": bool(qv < 0.10 and seen.sum() >= 4 and concord >= 0.75),
        })
    cells.sort(key=lambda c: (not c["significant"], c["q_fdr"], -abs(c["mean_z"])))
    return {"min_langs": min_langs, "min_count": min_count,
            "families": fams, "n_cells_tested": len(cells),
            "n_significant": sum(c["significant"] for c in cells),
            "n_fdr_only": sum(c["q_fdr"] < 0.10 for c in cells),
            "note": ("'significant' also requires the bias to appear in >=4 macro-"
                     "families with >=75% sign concordance -- a guard against "
                     "phylogenetic non-independence that the within-language "
                     "permutation null does not model."),
            "cells": cells}


def _write_pooled_csv(path, cells: list[dict]) -> None:
    import csv

    cols = ["pole", "gloss", "phoneme", "mean_z", "n_langs", "n_families",
            "family_sign_concord", "total_count", "p_perm", "q_fdr", "significant"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for c in cells:
            w.writerow([c[k] for k in cols])


def _write_csv(path, rows: list[dict]) -> None:
    import csv

    cols = ["language", "pole", "phoneme", "z", "count", "p_perm", "q_fdr", "significant"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
