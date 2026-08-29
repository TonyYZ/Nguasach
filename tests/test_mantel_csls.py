"""Mantel / partial-Mantel test and CSLS de-hubbing."""

from __future__ import annotations

import numpy as np

from nguasach import align, mantel


def test_mantel_detects_planted_correlation():
    rng = np.random.default_rng(0)
    n = 60
    coords = rng.normal(size=(n, 3))
    dx = np.linalg.norm(coords[:, None] - coords[None], axis=2)
    dy = dx + rng.normal(scale=0.05, size=(n, n))
    dy = (dy + dy.T) / 2
    r, p, _ = mantel._mantel(dx, dy, iters=200, seed=1)
    assert r > 0.9 and p < 0.01

    dnull = rng.normal(size=(n, n))
    dnull = np.abs(dnull + dnull.T)
    r0, p0, _ = mantel._mantel(dx, dnull, iters=200, seed=1)
    assert abs(r0) < 0.2 and p0 > 0.05


def test_partial_mantel_removes_shared_driver():
    rng = np.random.default_rng(1)
    n = 60
    dz = np.abs(rng.normal(size=(n, n)))
    dz = (dz + dz.T) / 2
    dx = dz + 0.01 * np.abs(rng.normal(size=(n, n)))
    dy = dz + 0.01 * np.abs(rng.normal(size=(n, n)))
    dx = (dx + dx.T) / 2
    dy = (dy + dy.T) / 2
    r_raw, _, _ = mantel._mantel(dx, dy, iters=100, seed=2)
    r_par, p_par, _ = mantel._mantel(dx, dy, iters=100, seed=2, dz=dz)
    assert r_raw > 0.8                      # both driven by dz
    assert abs(r_par) < 0.4                 # partialling dz kills most of it


def test_csls_lowers_rank_for_a_hub():
    # target 0 is a hub: close to everything; gold for query 0 is target 3
    tgt = np.eye(6, dtype=np.float32)
    tgt[0] = 0.4
    tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)
    pred = tgt[[3]].copy()
    r_plain = align.rank_of_gold(pred, tgt, np.array([3]), csls_k=0)[0]
    r_csls = align.rank_of_gold(pred, tgt, np.array([3]), csls_k=3)[0]
    assert r_csls <= r_plain
