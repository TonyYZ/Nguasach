"""Resampling statistics: bootstrap CIs, empirical p-values, BH-FDR,
Nadeau-Bengio corrected variance for k-fold scores."""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: np.ndarray, iters: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(point estimate = mean, lo, hi) via percentile bootstrap over `values`."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = values[rng.integers(0, n, size=(iters, n))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def empirical_p(observed: float, null: np.ndarray) -> float:
    """One-sided (observed >= null) empirical p with the +1 correction."""
    null = np.asarray(null, dtype=float)
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def nadeau_bengio_se(
    fold_scores: np.ndarray, n_train: int, n_test: int
) -> float:
    """Corrected SE of the mean CV score (Nadeau & Bengio 2003).

    Inflates the naive SE by the train/test overlap factor (1/K + n_test/n_train).
    """
    s = np.asarray(fold_scores, dtype=float)
    k = len(s)
    if k < 2:
        return float("nan")
    var = s.var(ddof=1)
    return float(np.sqrt(var * (1.0 / k + n_test / n_train)))
