"""Label-permutation null for the retrieval test.

Each iteration re-runs the full align+CV for a pair with the source<->target
concept pairing randomly permuted, giving an empirical distribution of top-k
accuracy under "no systematic sound-meaning correspondence". Replaces
transPhone.py's analytic ``randomBaseline`` as the significance test.

The pair's data (:class:`nguasach.crossval.PairData`) is loaded once and reused
across all iterations; only a fresh permutation seed changes per iteration.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
from pathlib import Path

import numpy as np

from .config import Config
from .crossval import PairData, score_pair

_CTX: dict = {}


def _worker_init(pd: PairData, folds, k, map_kind, alpha):
    _CTX["pd"], _CTX["folds"] = pd, folds
    _CTX["k"], _CTX["map"], _CTX["alpha"] = k, map_kind, alpha


def _worker(seed: int) -> float:
    return score_pair(
        _CTX["pd"], _CTX["folds"], k=_CTX["k"], map_kind=_CTX["map"],
        alpha=_CTX["alpha"], permute_seed=int(seed),
    ).summary()["acc_mean"]


def null_distribution(
    cfg: Config, pd: PairData, folds, iters: int, seed: int, n_jobs: int = 1
) -> np.ndarray:
    seeds = np.random.default_rng(seed).integers(1, 2**31 - 1, size=iters)
    if n_jobs == 1:
        _worker_init(pd, folds, cfg.k, cfg.map, cfg.ridge_alpha)
        return np.array([_worker(s) for s in seeds])
    with cf.ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=_worker_init,
        initargs=(pd, folds, cfg.k, cfg.map, cfg.ridge_alpha),
    ) as ex:
        return np.array(list(ex.map(_worker, seeds.tolist(), chunksize=8)))


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
