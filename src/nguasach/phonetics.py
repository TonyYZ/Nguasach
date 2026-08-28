"""Stage ``phonetics``: ``<Lang>V.txt`` -> phonetic word vectors ``<Lang>Emb.txt``.

Reimplements the vendored ``generate.py`` (feature-bigram counts -> row-normalize
-> whitened PCA) as importable, seeded code instead of a stdin/stdout subprocess.
The feature mapping itself (``featurephone.feature_bigrams`` +
``ipa2feature.csv``) is used unchanged from
``third_party/phonetic-similarity-vectors/``.

Output is word2vec text format: a ``<count> <dim>`` header then ``<label> v1 .. vD``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .config import REPO_ROOT, Config

_PSV_DIR = REPO_ROOT / "third_party" / "phonetic-similarity-vectors"


def _load_featurephone():
    """Import the vendored featurephone with its CWD-relative ipa2feature.csv."""
    import os

    if str(_PSV_DIR) not in sys.path:
        sys.path.insert(0, str(_PSV_DIR))
    cwd = os.getcwd()
    try:
        os.chdir(_PSV_DIR)                       # featurephone.py opens 'ipa2feature.csv'
        import featurephone                      # noqa: PLC0415

        import importlib

        importlib.reload(featurephone)
        return featurephone
    finally:
        os.chdir(cwd)


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_file(v_path: Path, dim: int, seed: int, min_feature_count: int = 2):
    """PSV-format file -> (labels, vectors, meta)."""
    from sklearn.decomposition import PCA

    fp = _load_featurephone()

    def known(ph: str) -> bool:
        try:
            fp.phone_to_features(ph)
            return True
        except KeyError:
            return False

    def resolve(ph: str) -> list[str]:
        """Known -> [ph]; unknown tied diphthong 'x͡y' -> split to known halves."""
        if known(ph):
            return [ph]
        if "͡" in ph:
            return [p for p in ph.split("͡") if known(p)]
        return []

    labels: list[str] = []
    per_word: list[Counter] = []
    feat_df = Counter()
    unknown = Counter()
    for line in v_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        label, phones = line.split("  ", 1)
        toks: list[str] = []
        for t in phones.split():
            r = resolve(t)
            if r:
                toks.extend(r)
            else:
                unknown[t] += 1
        feats = Counter(fp.feature_bigrams(toks))
        labels.append(label)
        per_word.append(feats)
        feat_df.update(feats.keys())

    vocab = sorted(f for f, c in feat_df.items() if c >= min_feature_count)
    if not vocab or not labels:
        raise ValueError(f"{v_path.name}: no usable feature bigrams")

    counts = np.array(
        [[w.get(f, 0) for f in vocab] for w in per_word], dtype=np.float64
    )
    counts = _normalize_rows(counts)

    n_comp = min(dim, counts.shape[0], counts.shape[1])
    pca = PCA(n_components=n_comp, whiten=True, random_state=seed)
    vecs = pca.fit_transform(counts).astype(np.float32)

    meta = {
        "n_words": len(labels),
        "n_feature_bigrams": len(vocab),
        "dim": int(n_comp),
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "n_unknown_phone_types": len(unknown),
        "unknown_phones": dict(unknown.most_common(15)),
    }
    return labels, vecs, meta


def write_emb(path: Path, labels: list[str], vecs: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(f"{len(labels)} {vecs.shape[1]}\n")
        for lab, v in zip(labels, vecs):
            fh.write(lab + " " + " ".join(f"{x:.6f}" for x in v) + "\n")


def run(cfg: Config) -> dict:
    interim = cfg.paths.resolve("interim")
    processed = cfg.paths.resolve("processed")
    processed.mkdir(parents=True, exist_ok=True)

    per_lang, missing = {}, []
    for lang in cfg.languages:
        v_path = interim / f"{lang}V.txt"
        if not v_path.exists():
            missing.append(lang)
            continue
        labels, vecs, meta = embed_file(v_path, cfg.dim, cfg.seed)
        write_emb(processed / f"{lang}Emb.txt", labels, vecs)
        per_lang[lang] = meta

    report = {
        "stage": "phonetics",
        "config": cfg.name,
        "config_fingerprint": cfg.fingerprint(),
        "languages": per_lang,
        "missing_v_files": missing,
    }
    (interim / "phonetics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not missing:
        (interim / "phonetics.done").write_text(cfg.fingerprint(), encoding="utf-8")
    return report
