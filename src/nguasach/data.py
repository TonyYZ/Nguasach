"""Stage ``data``: build the canonical concept table, labels, and CV folds.

Everything downstream aligns on **concept_id** = the 0-based row position in
``nguasach.xlsx``. This module is the single place that reads the spreadsheet,
normalizes it, and freezes a UTF-8 checkpoint so no later stage has to touch the
xlsx (or, worse, the corrupt ``nguasach.csv``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALL_LANGUAGES, Config

# Columns written in a non-Latin script. A cell here that is only "?" / spaces is
# the signature of the corrupt nguasach.csv (codepage export that replaced
# unrepresentable characters with '?'); we refuse to run on it.
NON_LATIN = (
    "Greek", "Russian", "Chinese", "Japanese", "Korean", "Thai",
    "Arabic", "Hebrew", "Hindi",
)

_WS = re.compile(r"\s+")
_Q_ONLY = re.compile(r"[?�\s]+")


# --------------------------------------------------------------------------- io
def _norm_cell(s: object) -> str:
    """NFC-normalize, collapse internal whitespace, strip ends. Script-preserving."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    return _WS.sub(" ", s).strip()


@lru_cache(maxsize=4)
def load_raw(cfg: Config) -> pd.DataFrame:
    """Read ``nguasach.xlsx`` -> normalized DataFrame, columns == ALL_LANGUAGES.

    Row order is sheet order (the positional key the pipeline aligns on).
    ``concept_set`` / ``max_concepts`` from the config are applied here.
    Cached: called once per pair in the null loop.
    """
    path = cfg.paths.resolve("xlsx")
    raw = pd.read_excel(path, dtype=str, keep_default_na=False, engine="openpyxl")
    raw.columns = [str(c).strip() for c in raw.columns]

    missing = [lang for lang in ALL_LANGUAGES if lang not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} is missing language columns: {missing}")

    df = raw[list(ALL_LANGUAGES)].map(_norm_cell)
    df = _apply_concept_set(df, cfg)
    df = df.reset_index(drop=True)
    df.index.name = "concept_id"
    return df


def _apply_concept_set(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    sel = cfg.concept_set
    if sel and sel != "all":
        list_path = Path("data/raw/swadesh.txt") if sel == "swadesh" else Path(sel)
        list_path = list_path if list_path.is_absolute() else cfg.paths.resolve("xlsx").parents[1] / list_path
        if not list_path.exists():
            raise FileNotFoundError(f"concept_set list not found: {list_path}")
        wanted = {
            _norm_cell(x).lower()
            for x in list_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        }
        df = df[df["English"].str.lower().isin(wanted)]
    if cfg.max_concepts is not None:
        df = df.iloc[: cfg.max_concepts]
    return df


# ------------------------------------------------------------------- integrity
def check_integrity(df: pd.DataFrame) -> dict:
    """Raise on a malformed table; return a JSON-able report."""
    report: dict = {"n_rows": int(len(df)), "n_cols": int(df.shape[1])}

    if list(df.columns) != list(ALL_LANGUAGES):
        raise AssertionError("column set/order does not match ALL_LANGUAGES")

    empty = {c: int((df[c] == "").sum()) for c in df.columns}
    empty = {c: n for c, n in empty.items() if n}
    report["empty_cells"] = empty
    if empty:
        raise AssertionError(f"empty cells present: {empty}")

    corrupt = {c: int(df[c].map(lambda s: bool(_Q_ONLY.fullmatch(s))).sum()) for c in NON_LATIN}
    corrupt = {c: n for c, n in corrupt.items() if n}
    report["placeholder_only_cells"] = corrupt
    if corrupt:
        raise AssertionError(
            f"non-Latin columns look corrupt (cells are only '?'/�): {corrupt}. "
            "Read data/raw/nguasach.xlsx, never the repo-root nguasach.csv."
        )

    eng_lower = df["English"].str.lower()
    dup_mask = eng_lower.duplicated(keep=False)
    report["duplicate_english"] = {
        "n_rows_involved": int(dup_mask.sum()),
        "n_distinct_repeated": int(eng_lower[dup_mask].nunique()),
        "examples": sorted(eng_lower[dup_mask].unique().tolist())[:20],
    }
    return report


# ---------------------------------------------------------------------- labels
def _sanitize_label(cell: str) -> str:
    return cell.replace(",", "").replace(" ", "-")


def make_labels(df: pd.DataFrame) -> dict[str, list[str]]:
    """Per language, a concept_id-aligned list of unique, gensim-safe labels.

    Mirrors transPhone.py's scheme (drop commas, spaces->'-', disambiguate
    repeats) but uses a readable '#2' suffix instead of trailing '/'.
    """
    labels: dict[str, list[str]] = {}
    for lang in df.columns:
        seen: dict[str, int] = {}
        col: list[str] = []
        for cid, cell in enumerate(df[lang]):
            base = _sanitize_label(cell) or f"_blank{cid}"
            seen[base] = seen.get(base, 0) + 1
            col.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
        labels[lang] = col
    return labels


# ----------------------------------------------------------------------- folds
def make_folds(
    n: int, folds: int, seed: int, test_folds: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Randomized, seeded k-fold partitions of ``range(n)``.

    Replaces transPhone.py's contiguous ``startRate`` windows (which sliced a
    semantically ordered list). Returns ``[(train_idx, test_idx), ...]`` with
    each index array sorted ascending.
    """
    if test_folds >= folds:
        raise ValueError("test_folds must be < folds")
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(n), folds)
    out = []
    for i in range(0, folds, test_folds):
        hi = min(i + test_folds, folds)
        test = np.concatenate(chunks[i:hi])
        train = np.concatenate([chunks[j] for j in range(folds) if not (i <= j < hi)])
        out.append((np.sort(train), np.sort(test)))
    return out


def leakage_report(
    df: pd.DataFrame, folds: list[tuple[np.ndarray, np.ndarray]]
) -> dict:
    """Assert disjoint train/test; quantify identical-surface-string overlap.

    Real translation-pair leakage is impossible here (splits are on concept_id,
    disjoint). The residual risk is *homographs*: a test concept whose target
    string is byte-identical to some train concept's, which a retriever could
    "hit" via the wrong row. We report that rate per fold, per language.
    """
    per_fold = []
    for k, (tr, te) in enumerate(folds):
        if set(tr.tolist()) & set(te.tolist()):
            raise AssertionError(f"fold {k}: train/test indices overlap")
        collisions = {}
        for lang in df.columns:
            train_surface = set(df[lang].iloc[tr])
            n_hit = int(df[lang].iloc[te].isin(train_surface).sum())
            if n_hit:
                collisions[lang] = n_hit
        per_fold.append(
            {"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)),
             "surface_collisions": collisions}
        )
    all_rates = [
        v / f["n_test"]
        for f in per_fold
        for v in f["surface_collisions"].values()
    ]
    return {
        "per_fold": per_fold,
        "max_collision_rate": max(all_rates, default=0.0),
        "mean_collision_rate": float(np.mean(all_rates)) if all_rates else 0.0,
    }


# ------------------------------------------------------------------ stage main
def run(cfg: Config) -> dict:
    """Execute the ``data`` stage: write the canonical table, labels, folds."""
    interim = cfg.paths.resolve("interim")
    interim.mkdir(parents=True, exist_ok=True)

    df = load_raw(cfg)
    integrity = check_integrity(df)
    labels = make_labels(df)
    folds = make_folds(len(df), cfg.folds, cfg.seed, cfg.test_folds)
    leakage = leakage_report(df, folds)

    df.to_csv(interim / "nguasach.utf8.csv", encoding="utf-8", index=True)
    (interim / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    (interim / "folds.json").write_text(
        json.dumps(
            [{"fold": i, "train": tr.tolist(), "test": te.tolist()}
             for i, (tr, te) in enumerate(folds)],
            indent=0,
        ),
        encoding="utf-8",
    )
    report = {
        "stage": "data",
        "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "integrity": integrity,
        "folds": {"n": cfg.folds, "test_folds": cfg.test_folds, "seed": cfg.seed},
        "leakage": leakage,
    }
    (interim / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (interim / "data.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report
