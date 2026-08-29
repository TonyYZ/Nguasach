"""Stage ``mantel``: distance-matrix correlation between form and meaning.

The primary iconicity analysis, and the one the field uses (Blasi et al. 2016;
Dautriche et al. 2017; Pimentel et al. 2019): correlate the pairwise
*form*-distance matrix with the pairwise *meaning*-distance matrix over concepts.
Immune to the retrieval metric's hubness. A **partial** Mantel additionally
residualizes out the orthographic edit-distance matrix, isolating sound–meaning
structure not attributable to spelling / cognate overlap.

* within-language  : D_form(L)  vs  D_meaning        (+ partial | D_orth(L))
* cross-language   : D_form(L1) vs D_form(L2)        (+ partial | D_orth pair)

Permutation p-values shuffle concept labels of one matrix.
"""

from __future__ import annotations

import json

import numpy as np

from .align import load_emb
from .config import Config
from . import data as _data


def _cosine_dist(mat: np.ndarray) -> np.ndarray:
    m = mat / np.linalg.norm(mat, axis=1, keepdims=True).clip(min=1e-12)
    d = 1.0 - m @ m.T
    np.fill_diagonal(d, 0.0)
    return d


def _edit_dist_matrix(strings: np.ndarray) -> np.ndarray:
    import Levenshtein

    n = len(strings)
    d = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        si = strings[i]
        d[i, i + 1 :] = [1.0 - Levenshtein.ratio(si, strings[j]) for j in range(i + 1, n)]
    return d + d.T


def _residualize(mat: np.ndarray, z: np.ndarray, iu) -> np.ndarray:
    """Elementwise mat - (a + b*z), a,b fitted on the upper triangle."""
    b, a = np.polyfit(z[iu], mat[iu], 1)
    return mat - (a + b * z)


def _mantel(dx: np.ndarray, dy: np.ndarray, iters: int, seed: int,
            dz: np.ndarray | None = None) -> tuple[float, float]:
    """Mantel r between dx, dy (partial on dz if given); permutation p (two-sided).

    Partial Mantel (Smouse et al. 1986): residualize both matrices on dz, then
    Mantel-correlate the residual matrices with row/col permutation.
    """
    n = dx.shape[0]
    iu = np.triu_indices(n, k=1)
    X, Y = dx.copy(), dy.copy()
    if dz is not None:
        X = _residualize(X, dz, iu)
        Y = _residualize(Y, dz, iu)

    xv = X[iu]
    xv = (xv - xv.mean()) / xv.std()

    def corr(Ym: np.ndarray) -> float:
        yv = Ym[iu]
        yv = (yv - yv.mean()) / yv.std()
        return float((xv * yv).mean())

    r_obs = corr(Y)
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(iters):
        p = rng.permutation(n)
        if abs(corr(Y[p][:, p])) >= abs(r_obs):
            ge += 1
    return r_obs, ge / (iters + 1)


def _subsample(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def run(cfg: Config, n_jobs: int = 1) -> dict:
    rdir = cfg.paths.resolve("results")
    rdir.mkdir(parents=True, exist_ok=True)
    processed = cfg.paths.resolve("processed")
    interim = cfg.paths.resolve("interim")
    df = _data.load_raw(cfg)
    lj = json.loads((interim / "labels.json").read_text(encoding="utf-8"))

    cap = getattr(cfg, "mantel_cap", 700)
    idx = _subsample(len(df), cap, cfg.seed)

    # meaning distance (shared across all within-language analyses)
    sem_keys = json.loads((interim / "semantics_keys.json").read_text(encoding="utf-8"))
    s_labels, s_mat = load_emb(processed / "SemanticsEmb.txt")
    s_pos = {lab: i for i, lab in enumerate(s_labels)}
    sem_ok = np.array([i for i in idx if sem_keys[i] in s_pos])
    d_mean = _cosine_dist(s_mat[[s_pos[sem_keys[i]] for i in sem_ok]])

    def form_dist(lang: str, ids: np.ndarray) -> np.ndarray:
        labels, mat = load_emb(processed / f"{lang}Emb.txt")
        pos = {lab: i for i, lab in enumerate(labels)}
        return _cosine_dist(mat[[pos[lj[lang][i]] for i in ids]])

    rows = []
    for lang in cfg.languages:
        d_form = form_dist(lang, sem_ok)
        d_orth = _edit_dist_matrix(df[lang].to_numpy()[sem_ok])
        r, p = _mantel(d_form, d_mean, cfg.null_iters, cfg.seed)
        rp, pp = _mantel(d_form, d_mean, cfg.null_iters, cfg.seed, dz=d_orth)
        rows.append({"analysis": "form~meaning", "unit": lang, "n": len(sem_ok),
                     "r": round(r, 4), "p_perm": round(p, 4),
                     "r_partial_orth": round(rp, 4), "p_partial": round(pp, 4)})

    core = list(cfg.verified_core)
    for a in range(len(core)):
        for b in range(a + 1, len(core)):
            l1, l2 = core[a], core[b]
            df1 = form_dist(l1, idx)
            df2 = form_dist(l2, idx)
            do1 = _edit_dist_matrix(df[l1].to_numpy()[idx])
            r, p = _mantel(df1, df2, cfg.null_iters, cfg.seed)
            rp, pp = _mantel(df1, df2, cfg.null_iters, cfg.seed, dz=do1)
            rows.append({"analysis": "form~form", "unit": f"{l1}~{l2}", "n": len(idx),
                         "r": round(r, 4), "p_perm": round(p, 4),
                         "r_partial_orth": round(rp, 4), "p_partial": round(pp, 4)})

    out = {"stage": "mantel", "config": cfg.name,
           "config_fingerprint": cfg.fingerprint(),
           "n_subsample": int(cap), "rows": rows}
    (rdir / "mantel.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _csv(rdir / "mantel.csv", rows)
    (interim / "mantel.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return out


def _csv(path, rows: list[dict]) -> None:
    import csv

    cols = ["analysis", "unit", "n", "r", "p_perm", "r_partial_orth", "p_partial"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])
