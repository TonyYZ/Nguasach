"""Label-permutation null for the retrieval test.

Each iteration re-runs the align+CV for a pair with the source<->target concept
pairing randomly permuted, giving an empirical distribution of top-k accuracy
under "no systematic sound-meaning correspondence". Replaces transPhone.py's
analytic ``randomBaseline`` as the significance test.

For the ridge map (the default) there is a fast path: the projection operator
``P_f = Xc_te (Xc_tr^T Xc_tr + aI)^-1 Xc_tr^T`` depends only on the *source*
vectors, so it is factored once per fold and every permutation is then a single
``P_f @ Xt[perm]`` matmul. ~100x fewer FLOPs than refitting per permutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla

from .align import rank_of_gold, topk_hits
from .config import Config
from .crossval import PairData, score_pair


# ------------------------------------------------------------- ridge fast path
def _fold_operator(xs_tr: np.ndarray, xs_te: np.ndarray, alpha: float):
    mx = xs_tr.mean(0)
    xc_tr = (xs_tr - mx).astype(np.float64)
    xc_te = (xs_te - mx).astype(np.float64)
    g = xc_tr.T @ xc_tr
    g[np.diag_indices_from(g)] += alpha
    P = xc_te @ sla.cho_solve(sla.cho_factor(g, lower=True), xc_tr.T)   # (n_te, n_tr)
    return P, P.sum(1)


def _ridge_null(
    pd: PairData,
    folds: list[tuple[np.ndarray, np.ndarray]],
    k: int,
    alpha: float,
    seeds: np.ndarray,
    csls_k: int = 0,
) -> np.ndarray:
    local = []
    for tr_c, te_c in folds:
        tr = np.array([pd.pos[int(c)] for c in tr_c if int(c) in pd.pos])
        te = np.array([pd.pos[int(c)] for c in te_c if int(c) in pd.pos])
        if len(tr) and len(te):
            P, Psum = _fold_operator(pd.xs[tr], pd.xs[te], alpha)
            local.append((tr, te, P, Psum))

    xt = pd.xt.astype(np.float64)
    out = np.empty(len(seeds))
    for i, seed in enumerate(seeds):
        perm = np.random.default_rng(int(seed)).permutation(len(pd.concepts))
        accs = []
        for tr, te, P, Psum in local:
            yt = xt[perm[tr]]
            my = yt.mean(0)
            pred = P @ yt - np.outer(Psum, my) + my
            pred /= np.linalg.norm(pred, axis=1, keepdims=True).clip(min=1e-12)
            ranks = rank_of_gold(pred.astype(np.float32), pd.xt, perm[te], csls_k=csls_k)
            accs.append(float(topk_hits(ranks, k).mean()))
        out[i] = np.mean(accs)
    return out


# ------------------------------------------------------------------- dispatch
def null_distribution(
    cfg: Config, pd: PairData, folds, iters: int, seed: int, n_jobs: int = 1
) -> np.ndarray:
    seeds = np.random.default_rng(seed).integers(1, 2**31 - 1, size=iters)
    if cfg.map == "ridge":
        return _ridge_null(pd, folds, cfg.k, cfg.ridge_alpha, seeds, csls_k=cfg.csls_k)
    # generic (slow) path for other maps
    return np.array([
        score_pair(pd, folds, k=cfg.k, map_kind=cfg.map, alpha=cfg.ridge_alpha,
                   csls_k=cfg.csls_k, permute_seed=int(s)).summary()["acc_mean"]
        for s in seeds
    ])


def cache_path(cfg: Config, source: str, target: str) -> Path:
    d = cfg.paths.resolve("results") / "null_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{source}__{target}__{cfg.fingerprint()}.json"


def null_for_pair(cfg: Config, pd: PairData, folds, n_jobs: int = 1) -> np.ndarray:
    cp = cache_path(cfg, pd.source, pd.target)
    if cp.exists():
        return np.array(json.loads(cp.read_text()))
    dist = null_distribution(cfg, pd, folds, cfg.null_iters, cfg.seed, n_jobs)
    cp.write_text(json.dumps(dist.tolist()), encoding="utf-8")
    return dist
