"""Stage ``baselines``: control analyses the phonetic result must beat.

The permutation null already rules out "any two languages' phoneme inventories
align enough for a learned map". These baselines address the two remaining
confounds:

* ``editdist`` -- non-learned orthographic string similarity. If Levenshtein
  retrieval already recovers translations (French<->English cognates!), the
  phonetic result is form overlap from borrowing, not sound-symbolism.
* ``orth`` -- character n-gram embeddings run through the *same* ridge-align +
  CV + null machinery. Orthographic analogue of the phonetic pipeline; also
  gives an orthography->Semantics comparison.
* ``feat`` -- mean panphon articulatory-feature vector per word, same machinery.
  Coarser phonology than the PSV feature-bigram embedding: if ``feat`` ~=
  ``phonetic`` the bigram structure adds little.

All rows use the same columns as ``results/accuracy_by_pair.csv`` so
``report.py`` can lay them beside the phonetic result.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from .align import rank_of_gold, topk_hits
from .config import Config
from . import crossval
from . import data as _data
from .stats import bh_fdr, bootstrap_ci, empirical_p, nadeau_bengio_se


# --------------------------------------------------------------- embeddings
def _pca(counts: np.ndarray, dim: int, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    counts = counts / np.linalg.norm(counts, axis=1, keepdims=True).clip(min=1e-12)
    n = min(dim, *counts.shape)
    return PCA(n_components=n, whiten=True, random_state=seed).fit_transform(counts).astype(np.float32)


def char_ngram_emb(words: list[str], ngram: tuple[int, ...], dim: int, seed: int):
    def grams(w: str):
        w = f"^{w}$"
        return [w[i : i + n] for n in ngram for i in range(len(w) - n + 1)]

    per = [Counter(grams(w)) for w in words]
    vocab = sorted({g for c in per for g in c})
    idx = {g: i for i, g in enumerate(vocab)}
    counts = np.zeros((len(words), len(vocab)), dtype=np.float64)
    for r, c in enumerate(per):
        for g, v in c.items():
            counts[r, idx[g]] = v
    return _pca(counts, dim, seed)


def panphon_feat_emb(phone_rows: list[list[str]], dim: int, seed: int):
    import panphon

    ft = panphon.FeatureTable()
    cache: dict[str, np.ndarray] = {}

    def seg_vec(ph: str) -> np.ndarray | None:
        if ph not in cache:
            vl = ft.word_to_vector_list(ph, numeric=True)
            cache[ph] = np.mean(vl, axis=0) if vl else None
        return cache[ph]

    d = len(ft.names)
    mat = np.zeros((len(phone_rows), d), dtype=np.float64)
    for r, phones in enumerate(phone_rows):
        vs = [v for v in (seg_vec(p) for p in phones) if v is not None]
        if vs:
            mat[r] = np.mean(vs, axis=0)
    return _pca(mat, dim, seed)


def _write_emb(path, labels: list[str], vecs: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{len(labels)} {vecs.shape[1]}\n")
        for lab, v in zip(labels, vecs):
            fh.write(lab + " " + " ".join(f"{x:.6f}" for x in v) + "\n")


def build_baseline_embeddings(cfg: Config) -> None:
    interim = cfg.paths.resolve("interim")
    processed = cfg.paths.resolve("processed")
    lj = json.loads((interim / "labels.json").read_text(encoding="utf-8"))
    df = _data.load_raw(cfg)

    for lang in cfg.languages:
        labels = lj[lang]
        if "orth" in cfg.baselines:
            vecs = char_ngram_emb(df[lang].tolist(), cfg.char_ngram, cfg.dim, cfg.seed)
            _write_emb(processed / f"{lang}OrthEmb.txt", labels, vecs)
        if "feat" in cfg.baselines:
            rows = _read_v(interim / f"{lang}V.txt", labels)
            vecs = panphon_feat_emb(rows, cfg.dim, cfg.seed)
            _write_emb(processed / f"{lang}FeatEmb.txt", labels, vecs)


def _read_v(path, labels: list[str]) -> list[list[str]]:
    by_label = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "  " in line:
            lab, phones = line.split("  ", 1)
            by_label[lab] = phones.split()
    return [by_label.get(lab, []) for lab in labels]


# ------------------------------------------------------ edit-distance retrieval
def editdist_pair(cfg: Config, source: str, target: str, folds, iters: int, seed: int) -> dict:
    import Levenshtein

    df = _data.load_raw(cfg)
    s = df[source].to_numpy()
    t = df[target].to_numpy()
    n = len(df)
    sim = np.empty((n, n), dtype=np.float32)
    for i in range(n):
        si = s[i]
        sim[i] = [Levenshtein.ratio(si, tj) for tj in t]

    def fold_accs(perm: np.ndarray) -> list[float]:
        accs = []
        for _, te in folds:
            gold = perm[te]
            rows = sim[te]
            gs = rows[np.arange(len(te)), gold]
            # midrank so a wall of ties (e.g. cross-script pairs -> all ratios 0)
            # gives rank ~ n/2, not rank 1.
            ranks = (rows > gs[:, None]).sum(1) + 0.5 * ((rows == gs[:, None]).sum(1) - 1) + 1
            accs.append(float((ranks <= cfg.k).mean()))
        return accs

    fa = fold_accs(np.arange(n))
    obs = float(np.mean(fa))
    rng = np.random.default_rng(seed)
    null = np.array([float(np.mean(fold_accs(rng.permutation(n)))) for _ in range(iters)])
    _, lo, hi = bootstrap_ci(fa, cfg.bootstrap_iters, seed)
    return {
        "baseline": "editdist", "source": source, "target": target, "k": cfg.k,
        "n_concepts": n, "acc_mean": obs, "acc_clean_mean": obs,
        "boot_ci95": [lo, hi], "nb_se": nadeau_bengio_se(fa, n * 0.9, n * 0.1),
        "null_mean": float(null.mean()), "null_p95": float(np.quantile(null, 0.95)),
        "p_perm": empirical_p(obs, null), "mean_rank": float("nan"),
        "n_collision_total": 0,
    }


# --------------------------------------------------------------- ridge baselines
def _ridge_baseline_rows(cfg: Config, tag: str, folds, targets: list[str]) -> list[dict]:
    from .nulls import null_distribution

    rows = []
    for src in cfg.languages:
        for tgt in targets:
            if src == tgt:
                continue
            pd = crossval.load_pair_data(cfg, src, tgt, emb_tag=tag)
            r = crossval.score_pair(pd, folds, k=cfg.k, map_kind="ridge",
                                    alpha=cfg.ridge_alpha).summary()
            null = null_distribution(cfg, pd, folds, cfg.null_iters, cfg.seed)
            _, lo, hi = bootstrap_ci(r["acc_folds"], cfg.bootstrap_iters, cfg.seed)
            rows.append({
                "baseline": {"Orth": "orth", "Feat": "feat"}[tag],
                "source": src, "target": tgt, "k": cfg.k,
                "n_concepts": r["n_concepts"], "acc_mean": r["acc_mean"],
                "acc_clean_mean": r["acc_clean_mean"], "boot_ci95": [lo, hi],
                "nb_se": nadeau_bengio_se(r["acc_folds"], r["n_train_per_fold"], r["n_test_per_fold"]),
                "null_mean": float(np.mean(null)),
                "null_p95": float(np.quantile(null, 0.95)),
                "p_perm": empirical_p(r["acc_mean"], null),
                "mean_rank": r["mean_rank"], "n_collision_total": r["n_collision_total"],
            })
    return rows


# ------------------------------------------------------------------- stage main
def run(cfg: Config, n_jobs: int = 1) -> dict:
    interim = cfg.paths.resolve("interim")
    rdir = cfg.paths.resolve("results")
    rdir.mkdir(parents=True, exist_ok=True)
    folds_raw = json.loads((interim / "folds.json").read_text(encoding="utf-8"))
    folds = [(np.array(f["train"]), np.array(f["test"])) for f in folds_raw]

    build_baseline_embeddings(cfg)

    rows: list[dict] = []
    if "editdist" in cfg.baselines:
        for src in cfg.languages:
            for tgt in cfg.languages:
                if src != tgt:
                    rows.append(editdist_pair(cfg, src, tgt, folds, cfg.null_iters, cfg.seed))
    have_sem = (cfg.paths.resolve("processed") / "SemanticsEmb.txt").exists()
    if "orth" in cfg.baselines:
        rows += _ridge_baseline_rows(cfg, "Orth",
                                     folds, list(cfg.languages) + (["Semantics"] if have_sem else []))
    if "feat" in cfg.baselines:
        rows += _ridge_baseline_rows(cfg, "Feat",
                                     folds, list(cfg.languages) + (["Semantics"] if have_sem else []))

    for base in {r["baseline"] for r in rows}:
        idx = [i for i, r in enumerate(rows) if r["baseline"] == base]
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

    cols = ["baseline", "source", "target", "acc_mean", "acc_clean_mean",
            "boot_ci95", "null_mean", "null_p95", "p_perm", "q_fdr",
            "mean_rank", "n_collision_total"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
