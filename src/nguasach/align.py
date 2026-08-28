"""Cross-lingual phonetic->phonetic (and phonetic->semantic) alignment.

Given translation-pair training rows (source vector, target vector), fit a map
source->target, then for a held-out source word retrieve the nearest target
concepts. A "hit" = the gold translation is within the top *k*.

Maps:
  ridge     -- sklearn Ridge (regularized linear); the plan's default, robust
               when n_train is not >> dim
  procrustes-- orthogonal least squares (classic BLI baseline)
  transvec  -- transvec.TranslationWordVectorizer (unregularized OLS; original)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- io
def load_emb(path: Path) -> tuple[list[str], np.ndarray]:
    """word2vec text ('<count> <dim>' header) -> (labels, L2-normalized matrix)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    n, dim = (int(x) for x in lines[0].split())
    labels, rows = [], np.empty((n, dim), dtype=np.float32)
    for i, line in enumerate(lines[1 : n + 1]):
        head, *nums = line.split(" ")
        labels.append(head)
        rows[i] = np.asarray(nums, dtype=np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True).clip(min=1e-12)
    return labels, rows


# ------------------------------------------------------------------ maps
class RidgeAlign:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, xs: np.ndarray, xt: np.ndarray) -> "RidgeAlign":
        # Closed-form ridge with an explicit Cholesky solve on the (dim x dim)
        # Gram matrix -- ~15x faster than sklearn Ridge(solver="auto"), which
        # lands on an SVD path for float32 multi-target here (7 ms vs 108 ms).
        xs = xs.astype(np.float64)
        xt = xt.astype(np.float64)
        self._mx = xs.mean(0)
        self._my = xt.mean(0)
        xc, yc = xs - self._mx, xt - self._my
        gram = xc.T @ xc
        gram[np.diag_indices_from(gram)] += self.alpha
        self._w = np.linalg.solve(gram, xc.T @ yc)
        return self

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return ((xs.astype(np.float64) - self._mx) @ self._w + self._my).astype(np.float32)


class ProcrustesAlign:
    """Orthogonal W minimizing ||xs W - xt||; W = U V^T from SVD of xs^T xt."""

    def fit(self, xs: np.ndarray, xt: np.ndarray) -> "ProcrustesAlign":
        u, _, vt = np.linalg.svd(xs.T @ xt, full_matrices=False)
        self.W = (u @ vt).astype(np.float32)
        return self

    def predict(self, xs: np.ndarray) -> np.ndarray:
        return xs @ self.W


def make_map(kind: str, alpha: float = 1.0):
    if kind == "ridge":
        return RidgeAlign(alpha)
    if kind == "procrustes":
        return ProcrustesAlign()
    if kind == "transvec":
        raise NotImplementedError("transvec map is wired in crossval via gensim; use 'ridge'")
    raise ValueError(f"unknown map kind: {kind}")


# ------------------------------------------------------------- retrieval
def rank_of_gold(
    pred: np.ndarray, target_mat: np.ndarray, gold_rows: np.ndarray
) -> np.ndarray:
    """1-based rank of each gold target row under cosine similarity to pred.

    pred, target_mat assumed L2-normalized (so pred @ target_mat.T = cosine).
    """
    sims = pred @ target_mat.T                       # (n_test, n_targets)
    gold_sim = sims[np.arange(len(pred)), gold_rows]
    return (sims > gold_sim[:, None]).sum(axis=1) + 1


def topk_hits(ranks: np.ndarray, k: int) -> np.ndarray:
    return ranks <= k
