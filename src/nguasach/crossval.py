"""k-fold cross-validated retrieval accuracy for one ordered language pair.

Replaces transPhone.py's single contiguous 80/20 split + one-sample t-test with
seeded randomized folds (:func:`nguasach.data.make_folds`) and reports the
homograph-collision-clean accuracy alongside the raw one (see the leakage
finding in the ``data`` stage).

Per-pair inputs (xlsx read, embedding files) are loaded **once** into a
:class:`PairData`; the permutation null then reuses it across thousands of
iterations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .align import load_emb, make_map, rank_of_gold, topk_hits
from .config import Config
from . import data as _data


# --------------------------------------------------------------- pair data
@dataclass
class PairData:
    source: str
    target: str
    xs: np.ndarray            # (n, dim_s) L2-normalized, concept-aligned
    xt: np.ndarray            # (n, dim_t) L2-normalized, concept-aligned
    surf: np.ndarray          # (n,) target surface strings (for collision guard)
    concepts: np.ndarray      # (n,) concept_ids kept
    pos: dict                 # concept_id -> local row


@lru_cache(maxsize=8)
def _emb(path_str: str):
    from pathlib import Path

    return load_emb(Path(path_str))


def _labels_json(cfg: Config) -> dict:
    return json.loads((cfg.paths.resolve("interim") / "labels.json").read_text(encoding="utf-8"))


def load_pair_data(cfg: Config, source: str, target: str, emb_tag: str = "") -> PairData:
    """``emb_tag`` selects the embedding family: "" -> ``<Lang>Emb.txt`` (phonetic),
    "Orth" -> ``<Lang>OrthEmb.txt``, "Feat" -> ``<Lang>FeatEmb.txt`` (baselines).
    The Semantics target always uses ``SemanticsEmb.txt`` regardless of tag."""
    interim = cfg.paths.resolve("interim")
    processed = cfg.paths.resolve("processed")
    lj = _labels_json(cfg)
    df = _data.load_raw(cfg)

    s_lab = lj[source]
    s_labels, s_mat = _emb(str(processed / f"{source}{emb_tag}Emb.txt"))
    s_idx = {lab: i for i, lab in enumerate(s_labels)}

    if target == "Semantics":
        t_lab = json.loads((interim / "semantics_keys.json").read_text(encoding="utf-8"))
        t_labels, t_mat = _emb(str(processed / "SemanticsEmb.txt"))
    else:
        t_lab = lj[target]
        t_labels, t_mat = _emb(str(processed / f"{target}{emb_tag}Emb.txt"))
    t_idx = {lab: i for i, lab in enumerate(t_labels)}

    drop: set[int] = set()
    if cfg.qc_mode != "off":
        from .translate_qc import flagged_concepts

        drop = flagged_concepts(cfg, source) | (
            flagged_concepts(cfg, target) if target != "Semantics" else set()
        )
    concepts = np.array(
        [i for i in range(len(df))
         if i not in drop and i < len(s_lab) and i < len(t_lab)
         and s_lab[i] in s_idx and t_lab[i] in t_idx],
        dtype=int,
    )
    xs = s_mat[[s_idx[s_lab[i]] for i in concepts]]
    xt = t_mat[[t_idx[t_lab[i]] for i in concepts]]
    surf = (df[target].to_numpy()[concepts] if target != "Semantics"
            else np.array([t_lab[i] for i in concepts]))
    pos = {int(c): j for j, c in enumerate(concepts)}
    return PairData(source, target, xs, xt, surf, concepts, pos)


# ------------------------------------------------------------------ scoring
@dataclass
class PairResult:
    source: str
    target: str
    k: int
    n_concepts: int
    fold_acc: list[float] = field(default_factory=list)
    fold_acc_clean: list[float] = field(default_factory=list)
    fold_mean_rank: list[float] = field(default_factory=list)
    n_test: int = 0
    n_train: int = 0
    n_collision: int = 0

    def summary(self) -> dict:
        a = np.array(self.fold_acc)
        return {
            "source": self.source, "target": self.target, "k": self.k,
            "n_concepts": self.n_concepts, "n_train_per_fold": self.n_train,
            "n_test_per_fold": self.n_test, "n_collision_total": self.n_collision,
            "acc_mean": float(a.mean()), "acc_folds": self.fold_acc,
            "acc_clean_mean": float(np.nanmean(self.fold_acc_clean)),
            "acc_clean_folds": self.fold_acc_clean,
            "mean_rank": float(np.mean(self.fold_mean_rank)),
        }


def score_pair(
    pd: PairData,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    k: int,
    map_kind: str,
    alpha: float,
    csls_k: int = 0,
    permute_seed: int | None = None,
) -> PairResult:
    n = len(pd.concepts)
    perm = (np.random.default_rng(permute_seed).permutation(n)
            if permute_seed is not None else np.arange(n))

    res = PairResult(pd.source, pd.target, k, n)
    for tr_c, te_c in folds:
        tr = np.array([pd.pos[int(c)] for c in tr_c if int(c) in pd.pos])
        te = np.array([pd.pos[int(c)] for c in te_c if int(c) in pd.pos])
        if len(tr) == 0 or len(te) == 0:
            continue

        gold = perm[te]
        model = make_map(map_kind, alpha).fit(pd.xs[tr], pd.xt[perm[tr]])
        pred = model.predict(pd.xs[te])
        pred = pred / np.linalg.norm(pred, axis=1, keepdims=True).clip(min=1e-12)

        ranks = rank_of_gold(pred, pd.xt, gold, csls_k=csls_k)
        hits = topk_hits(ranks, k)

        train_surf = set(pd.surf[perm[tr]])
        collision = np.array([pd.surf[g] in train_surf for g in gold])

        res.fold_acc.append(float(hits.mean()))
        clean = hits[~collision]
        res.fold_acc_clean.append(float(clean.mean()) if len(clean) else float("nan"))
        res.fold_mean_rank.append(float(ranks.mean()))
        res.n_test, res.n_train = len(te), len(tr)
        res.n_collision += int(collision.sum())
    return res


# --------------------------------------------------------------- stage main
def _folds(cfg: Config):
    interim = cfg.paths.resolve("interim")
    return [
        (np.array(f["train"]), np.array(f["test"]))
        for f in json.loads((interim / "folds.json").read_text(encoding="utf-8"))
    ]


def run(cfg: Config, n_jobs: int = 1) -> dict:
    from .nulls import null_for_pair
    from .stats import bh_fdr, bootstrap_ci, empirical_p, nadeau_bengio_se

    interim = cfg.paths.resolve("interim")
    results_dir = cfg.paths.resolve("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    folds = _folds(cfg)

    langs = list(cfg.languages)
    core = set(cfg.verified_core)
    have_sem = (cfg.paths.resolve("processed") / "SemanticsEmb.txt").exists()
    targets = langs + (["Semantics"] if have_sem else [])

    rows = []
    for src in langs:
        for tgt in targets:
            if src == tgt:
                continue
            pd = load_pair_data(cfg, src, tgt)
            r = score_pair(pd, folds, k=cfg.k, map_kind=cfg.map,
                           alpha=cfg.ridge_alpha, csls_k=cfg.csls_k).summary()
            null = null_for_pair(cfg, pd, folds, n_jobs=n_jobs)
            _, lo, hi = bootstrap_ci(r["acc_folds"], cfg.bootstrap_iters, cfg.seed)
            r["boot_ci95"] = [lo, hi]
            r["nb_se"] = nadeau_bengio_se(r["acc_folds"], r["n_train_per_fold"], r["n_test_per_fold"])
            r["null_mean"] = float(np.mean(null))
            r["null_p95"] = float(np.quantile(null, 0.95))
            r["p_perm"] = empirical_p(r["acc_mean"], null)
            r["family"] = "confirmatory" if (src in core and tgt in core) else "exploratory"
            rows.append(r)

    for fam in ("confirmatory", "exploratory"):
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        if idx:
            q = bh_fdr(np.array([rows[i]["p_perm"] for i in idx]))
            for j, i in enumerate(idx):
                rows[i]["q_fdr"] = float(q[j])

    out = {
        "stage": "align", "config": cfg.name, "config_fingerprint": cfg.fingerprint(),
        "map": cfg.map, "k": cfg.k, "folds": cfg.folds,
        "null_iters": cfg.null_iters, "bootstrap_iters": cfg.bootstrap_iters,
        "pairs": rows,
    }
    (results_dir / "accuracy_by_pair.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    _write_csv(results_dir / "accuracy_by_pair.csv", rows)
    (interim / "align.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return out


def _write_csv(path, rows: list[dict]) -> None:
    import csv

    cols = ["family", "source", "target", "k", "n_concepts", "acc_mean",
            "acc_clean_mean", "boot_ci95", "nb_se", "null_mean", "null_p95",
            "p_perm", "q_fdr", "mean_rank", "n_collision_total"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
