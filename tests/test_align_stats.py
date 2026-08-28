"""Alignment retrieval, CV scoring, permutation null, and resampling stats."""

from __future__ import annotations

import numpy as np
import pytest

from nguasach import align, crossval, data, ipa, phonetics, stats
from nguasach.config import Config


# ------------------------------------------------------------------- stats
def test_bh_fdr_matches_known_values():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = stats.bh_fdr(p)
    assert np.allclose(q, [0.05, 0.05, 0.05, 0.05, 0.05])
    assert stats.bh_fdr(np.array([0.001, 0.5]))[0] == pytest.approx(0.002)


def test_empirical_p_has_plus_one_correction():
    null = np.zeros(99)
    assert stats.empirical_p(1.0, null) == pytest.approx(1 / 100)
    assert stats.empirical_p(-1.0, null) == pytest.approx(100 / 100)


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    x = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    m, lo, hi = stats.bootstrap_ci(x, iters=2000, seed=0)
    assert lo <= m <= hi
    assert stats.bootstrap_ci(x, 2000, 0) == stats.bootstrap_ci(x, 2000, 0)


def test_nadeau_bengio_se_inflates_naive():
    s = np.array([0.2, 0.25, 0.15, 0.3, 0.1])
    naive = s.std(ddof=1) / np.sqrt(len(s))
    assert stats.nadeau_bengio_se(s, n_train=180, n_test=20) > naive


# --------------------------------------------------------------- retrieval
def test_rank_of_gold_and_topk():
    tgt = np.eye(4, dtype=np.float32)
    pred = np.array([[0.9, 0.1, 0, 0], [0, 0, 0.2, 0.9]], dtype=np.float32)
    ranks = align.rank_of_gold(pred, tgt, np.array([0, 3]))
    assert list(ranks) == [1, 1]
    assert align.topk_hits(np.array([1, 5, 3]), k=3).tolist() == [True, False, True]


def test_procrustes_recovers_a_rotation():
    rng = np.random.default_rng(0)
    xs = rng.normal(size=(50, 8)).astype(np.float32)
    theta = 0.7
    R = np.eye(8, dtype=np.float32)
    R[:2, :2] = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    xt = xs @ R
    W = align.ProcrustesAlign().fit(xs, xt).W
    assert np.allclose(xs @ W, xt, atol=1e-4)


# ------------------------------------------------------- pair scoring / null
@pytest.fixture(scope="module")
def smoke_ready() -> Config:
    cfg = Config.load("configs/smoke.yaml")
    try:
        ipa._setup_espeak()
    except Exception:
        pytest.skip("espeak backend unavailable")
    data.run(cfg)
    ipa.run(cfg)
    phonetics.run(cfg)
    return cfg


def test_real_pairing_beats_permuted_for_german_english(smoke_ready):
    cfg = smoke_ready
    folds = crossval._folds(cfg)
    pd = crossval.load_pair_data(cfg, "German", "English")

    obs = crossval.score_pair(pd, folds, k=cfg.k, map_kind="ridge", alpha=1.0).summary()
    null = np.array([
        crossval.score_pair(pd, folds, k=cfg.k, map_kind="ridge", alpha=1.0,
                            permute_seed=s).summary()["acc_mean"]
        for s in range(40)
    ])
    assert obs["acc_mean"] > np.quantile(null, 0.95)
    assert stats.empirical_p(obs["acc_mean"], null) < 0.05


def test_score_pair_is_deterministic(smoke_ready):
    cfg = smoke_ready
    folds = crossval._folds(cfg)
    pd = crossval.load_pair_data(cfg, "French", "English")
    a = crossval.score_pair(pd, folds, k=cfg.k, map_kind="ridge", alpha=1.0).summary()
    b = crossval.score_pair(pd, folds, k=cfg.k, map_kind="ridge", alpha=1.0).summary()
    assert a["acc_folds"] == b["acc_folds"]


def test_permuted_null_is_near_chance(smoke_ready):
    cfg = smoke_ready
    folds = crossval._folds(cfg)
    pd = crossval.load_pair_data(cfg, "Chinese", "English")
    null = np.array([
        crossval.score_pair(pd, folds, k=cfg.k, map_kind="ridge", alpha=1.0,
                            permute_seed=s).summary()["acc_mean"]
        for s in range(40)
    ])
    # chance for top-k over ~n concepts is k/n
    chance = cfg.k / pd.xt.shape[0]
    assert abs(null.mean() - chance) < 0.05
