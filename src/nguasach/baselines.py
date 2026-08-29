"""Stage ``baselines``: non-learned control retrieval for language pairs.

The phonetic result uses a *learned* ridge map. These baselines ask what a
**direct similarity** (no learning) already buys you, so the map's contribution
is isolable. All three build an n_src x n_tgt similarity matrix, apply CSLS
de-hubbing, and score top-k retrieval through the same 10-fold + 1000-permutation
+ BH-FDR machinery, so the numbers sit beside ``accuracy_by_pair.csv`` directly.

* ``editdist`` -- normalized Levenshtein similarity of surface strings.
  The cognate / borrowing control.
* ``orth``     -- cosine of character n-gram (2,3) count vectors of the surface
  strings. Cross-script pairs (Chinese chars vs Latin) correctly score ~chance.
* ``feat``     -- cosine of the mean ``panphon`` articulatory-feature vector per
  word (from the IPA). A coarse, script-independent phonological similarity.

There is no ``-> Semantics`` baseline: that comparison is the partial Mantel
(form ~ meaning | orthography) in the ``mantel`` stage.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .align import rank_of_gold, topk_hits
from .config import Config
from . import data as _data
from .stats import bh_fdr, bootstrap_ci, empirical_p, nadeau_bengio_se


# ------------------------------------------------------------- representations
def _grams(w: str, ngram: tuple[int, ...]) -> list[str]:
    w = f"^{w}$"
    return [w[i : i + n] for n in ngram for i in range(len(w) - n + 1)]


def _char_ngram_matrices(
    by_lang: dict[str, np.ndarray], ngram: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """One L2-normalized count matrix per language over a **shared** n-gram vocab,
    so cross-language cosine (``A @ B.T``) is well-defined."""
    per = {l: [Counter(_grams(str(w), ngram)) for w in ss] for l, ss in by_lang.items()}
    vocab = {g: i for i, g in enumerate(sorted(
        {g for cs in per.values() for c in cs for g in c}))}
    out = {}
    for lang, cs in per.items():
        m = np.zeros((len(cs), len(vocab)), dtype=np.float32)
        for r, c in enumerate(cs):
            for g, v in c.items():
                m[r, vocab[g]] = v
        out[lang] = _l2(m)
    return out


def _panphon_vectors(phone_rows: list[list[str]]) -> np.ndarray:
    import panphon

    ft = panphon.FeatureTable()
    cache: dict[str, np.ndarray | None] = {}

    def seg(p: str):
        if p not in cache:
            vl = ft.word_to_vector_list(p, numeric=True)
            cache[p] = np.mean(vl, axis=0) if vl else None
        return cache[p]

    d = len(ft.names)
    m = np.zeros((len(phone_rows), d), dtype=np.float32)
    for r, phones in enumerate(phone_rows):
        vs = [v for v in (seg(p) for p in phones) if v is not None]
        if vs:
            m[r] = np.mean(vs, axis=0)
    return _l2(m)


def _l2(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-12)


def _read_v(path, labels: list[str]) -> list[list[str]]:
    by = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "  " in line:
            lab, ph = line.split("  ", 1)
            by[lab] = ph.split()
    return [by.get(lab, []) for lab in labels]


# ------------------------------------------------------------------- scoring
def _score_from_sim(
    sim: np.ndarray, folds, k: int, csls_k: int, iters: int, seed: int,
    boot_iters: int,
) -> dict:
    """sim is (n, n), concept-aligned and FIXED (no learning), so CSLS is applied
    once to the whole matrix and the permutation null is then pure relabelling."""
    n = sim.shape[0]
    s = sim.astype(np.float64)
    kk = min(csls_k, n - 1)
    if kk > 0:
        r_t = -np.partition(-s, kk, axis=0)[:kk].mean(axis=0)
        r_s = -np.partition(-s, kk, axis=1)[:, :kk].mean(axis=1)
        s = 2 * s - r_t[None, :] - r_s[:, None]

    def fold_accs(perm: np.ndarray) -> list[float]:
        out = []
        for _, te in folds:
            te = np.asarray(te)
            rows = s[te]
            gs = rows[np.arange(len(te)), perm[te]]
            ranks = (rows > gs[:, None]).sum(1) + 0.5 * ((rows == gs[:, None]).sum(1) - 1) + 1
            out.append(float((ranks <= k).mean()))
        return out

    fa = fold_accs(np.arange(n))
    obs = float(np.mean(fa))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(fold_accs(rng.permutation(n)))) for _ in range(iters)])
    _, lo, hi = bootstrap_ci(fa, boot_iters, seed)
    return {
        "acc_mean": obs, "boot_ci95": [lo, hi],
        "nb_se": nadeau_bengio_se(fa, n * 0.9, n * 0.1),
        "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
        "p_perm": empirical_p(obs, null),
    }


# ------------------------------------------------------------------- stage main
def run(cfg: Config, n_jobs: int = 1) -> dict:
    interim = cfg.paths.resolve("interim")
    rdir = cfg.paths.resolve("results")
    rdir.mkdir(parents=True, exist_ok=True)
    df = _data.load_raw(cfg)
    n = len(df)
    lj = json.loads((interim / "labels.json").read_text(encoding="utf-8"))
    folds_raw = json.loads((interim / "folds.json").read_text(encoding="utf-8"))
    folds = [(np.array(f["train"]), np.array(f["test"])) for f in folds_raw]
    covered = sorted(int(i) for f in folds_raw for i in f["test"])
    if covered != list(range(n)):
        raise RuntimeError(
            f"folds.json covers {len(covered)} concepts but the table has {n}; "
            "run the `data` stage for this config first."
        )

    # precompute per-language representations once
    import Levenshtein

    surf = {l: df[l].to_numpy() for l in cfg.languages}
    ng = _char_ngram_matrices(surf, cfg.char_ngram) if "orth" in cfg.baselines else {}
    pf = {l: _panphon_vectors(_read_v(interim / f"{l}V.txt", lj[l])) for l in cfg.languages} \
        if "feat" in cfg.baselines else {}

    rows = []
    for src in cfg.languages:
        for tgt in cfg.languages:
            if src == tgt:
                continue
            for kind in cfg.baselines:
                if kind == "editdist":
                    s, t = surf[src], surf[tgt]
                    sim = np.array([[Levenshtein.ratio(a, b) for b in t] for a in s],
                                   dtype=np.float32)
                elif kind == "orth":
                    sim = ng[src] @ ng[tgt].T
                elif kind == "feat":
                    sim = pf[src] @ pf[tgt].T
                else:
                    continue
                m = _score_from_sim(sim, folds, cfg.k, cfg.csls_k,
                                    cfg.null_iters, cfg.seed, cfg.bootstrap_iters)
                rows.append({"baseline": kind, "source": src, "target": tgt,
                             "k": cfg.k, "n_concepts": n, **m})

    for kind in {r["baseline"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["baseline"] == kind]
        q = bh_fdr(np.array([rows[i]["p_perm"] for i in idx]))
        for j, i in enumerate(idx):
            rows[i]["q_fdr"] = float(q[j])

    out = {"stage": "baselines", "config": cfg.name,
           "config_fingerprint": cfg.fingerprint(), "rows": rows}
    (rdir / "baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _csv(rdir / "baselines.csv", rows)
    (interim / "baselines.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return out


def _csv(path, rows: list[dict]) -> None:
    import csv

    cols = ["baseline", "source", "target", "acc_mean", "boot_ci95",
            "null_mean", "null_p95", "p_perm", "q_fdr"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
