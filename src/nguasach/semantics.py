"""Stage ``semantics``: per-concept semantic vectors + the PCA-compressed space.

The alignment stage treats "Semantics" as one more "language" whose per-concept
vectors happen to be word2vec vectors instead of phonetic ones; aligning a
language's *phonetic* vectors onto this space and retrieving the right concept is
the iconicity test. This module produces:

* ``data/interim/semantics_keys.json``  -- concept_id-aligned list of model.txt keys
* ``data/processed/SemanticsEmb.txt``   -- word2vec-format file (header + rows) for
  every resolved concept key **and** every ``hexLabels`` pole seed word, each
  compressed to ``cfg.semantic_dim`` via whitened PCA fit on the full model.txt
  (faithful to the original ``compressSemantics.py``).

Semantic keys come from the ``Semantics`` column of the legacy ``nguasachV.csv``
(``<english>_``), joined on lowercased English. The ~57 concepts absent there
(mostly multi-word: "apple pie", "black hole") fall back to a mechanical key and,
if that is not a single model.txt token, to the mean of their constituent-word
vectors. Homographs ("bark" = dog / tree) share one key -- a known limitation
(~2% of concepts), recorded in the report and the data statement.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import yaml

from .config import Config
from . import data as _data

_KEY_CLEAN = re.compile(r"[^a-z0-9]+")


def _mech_key(english: str) -> str:
    """Mechanical model.txt-style key: lowercase, non-alphanumerics -> '_', trailing '_'."""
    slug = _KEY_CLEAN.sub("_", english.strip().lower()).strip("_")
    return f"{slug}_"


def load_semantics_column(cfg: Config) -> dict[str, str]:
    """lowercased English -> Semantics key, from the legacy nguasachV.csv (first match)."""
    path = cfg.paths.resolve("semantics_source_csv")
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        header = [h.lstrip("﻿").strip() for h in next(reader)]
        ei, si = header.index("English"), header.index("Semantics")
        for row in reader:
            if len(row) <= max(ei, si):
                continue
            eng = row[ei].strip().lower()
            key = row[si].strip()
            if eng and key and eng not in out:
                out[eng] = key
    return out


def build_keys(cfg: Config, english: list[str]) -> tuple[list[str], list[int]]:
    """concept_id-aligned semantic keys; also return the indices that fell back."""
    lookup = load_semantics_column(cfg)
    keys, fallback = [], []
    for cid, eng in enumerate(english):
        e = eng.strip().lower()
        if e in lookup:
            keys.append(lookup[e])
        else:
            keys.append(_mech_key(eng))
            fallback.append(cid)
    return keys, fallback


def load_model_txt(path: Path) -> tuple[list[str], np.ndarray]:
    """Read the headerless 300-d model.txt -> (words, matrix)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is gitignored (~60 MB word2vec text, '<word>_' keys); "
            "see data/README.md for provenance."
        )
    words: list[str] = []
    rows: list[np.ndarray] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            words.append(parts[0])
            rows.append(np.asarray(parts[1:], dtype=np.float32))
    mat = np.vstack(rows)
    return words, mat


def _resolve(key: str, vocab: dict[str, np.ndarray], dim: int) -> np.ndarray | None:
    """Direct hit, else mean of constituent '<part>_' vectors, else None."""
    if key in vocab:
        return vocab[key]
    parts = [p for p in key.strip("_").split("_") if p]
    vecs = [vocab[f"{p}_"] for p in parts if f"{p}_" in vocab]
    if vecs and len(vecs) == len(parts):
        return np.mean(vecs, axis=0)
    return None


def run(cfg: Config) -> dict:
    interim = cfg.paths.resolve("interim")
    processed = cfg.paths.resolve("processed")
    processed.mkdir(parents=True, exist_ok=True)

    df = _data.load_raw(cfg)
    english = df["English"].tolist()
    keys, fallback = build_keys(cfg, english)

    words, mat = load_model_txt(cfg.paths.resolve("word2vec_model"))
    vocab = {w: v for w, v in zip(words, mat)}

    from sklearn.decomposition import PCA

    pca = PCA(n_components=cfg.semantic_dim, whiten=cfg.semantic_whiten,
              random_state=cfg.seed)
    compressed = pca.fit_transform(mat).astype(np.float32)
    cvocab = {w: v for w, v in zip(words, compressed)}

    # pole seed words
    pole_words: set[str] = set()
    hex_path = cfg.paths.resolve("hex_labels")
    if hex_path.exists():
        poles = yaml.safe_load(hex_path.read_text(encoding="utf-8"))["poles"]
        for pole in poles:
            pole_words.update(f"{w}_" for w in pole["words"])

    # resolve every needed key against the *compressed* space
    unresolved: list[str] = []
    out_rows: dict[str, np.ndarray] = {}
    for key in dict.fromkeys(keys):                       # unique, order-preserving
        vec = _resolve(key, cvocab, cfg.semantic_dim)
        if vec is None:
            unresolved.append(key)
        else:
            out_rows[key] = vec
    n_pole_missing = 0
    for pw in sorted(pole_words):
        vec = _resolve(pw, cvocab, cfg.semantic_dim)
        if vec is None:
            n_pole_missing += 1
        else:
            out_rows.setdefault(pw, vec)

    emb_path = processed / "SemanticsEmb.txt"
    with emb_path.open("w", encoding="utf-8") as fh:
        fh.write(f"{len(out_rows)} {cfg.semantic_dim}\n")
        for key, vec in out_rows.items():
            fh.write(key + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")

    (interim / "semantics_keys.json").write_text(
        json.dumps(keys, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    report = {
        "stage": "semantics",
        "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "n_concepts": len(keys),
        "n_from_nguasachV": len(keys) - len(fallback),
        "n_mechanical_fallback": len(fallback),
        "n_unique_keys": len(dict.fromkeys(keys)),
        "n_keys_unresolved": len(unresolved),
        "unresolved_examples": unresolved[:20],
        "n_homograph_collisions": int(len(keys) - len(set(keys))),
        "model_txt_vocab": len(words),
        "semantic_dim": cfg.semantic_dim,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "n_pole_seed_words": len(pole_words),
        "n_pole_seed_missing": n_pole_missing,
        "semantics_emb_rows": len(out_rows),
    }
    (interim / "semantics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (interim / "semantics.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report
