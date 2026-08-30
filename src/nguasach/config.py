"""Typed configuration for the Nguasach pipeline.

Every knob that used to be a module-level boolean in ``transPhone.py``
(``initialExecution``, ``useUni``, ``useVecMap``, ``useNeuralNetwork``,
``needShuffle``) or a hand-edited call argument in ``main()`` lives here instead
and is loaded from a YAML file under ``configs/``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

# Repo root = two levels up from this file (src/nguasach/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]

# The four language columns the user manually verified. Headline (confirmatory)
# claims are restricted to pairs drawn from this set.
VERIFIED_CORE = ("English", "Chinese", "French", "Irish")

ALL_LANGUAGES = (
    "Hungarian", "Finnish", "Greek", "Russian", "German", "Spanish", "Italian",
    "French", "Irish", "Welsh", "English", "Chinese", "Vietnamese", "Japanese",
    "Korean", "Thai", "Indonesian", "Turkish", "Arabic", "Hebrew", "Swahili",
    "Hindi",
)

# Logographic / non-alphabetic scripts: a character edit-distance matrix is not
# a meaningful "orthographic similarity" control for these, so the partial
# Mantel against it is reported but flagged not-interpretable.
LOGOGRAPHIC = {"Chinese", "Japanese"}

# Genealogical family, for stratifying the exploratory cross-language analyses.
LANGUAGE_FAMILY = {
    "Hungarian": "Uralic", "Finnish": "Uralic",
    "Greek": "IE-Hellenic", "Russian": "IE-Slavic",
    "German": "IE-Germanic", "English": "IE-Germanic",
    "Spanish": "IE-Romance", "Italian": "IE-Romance", "French": "IE-Romance",
    "Irish": "IE-Celtic", "Welsh": "IE-Celtic",
    "Hindi": "IE-Indic",
    "Chinese": "Sino-Tibetan", "Vietnamese": "Austroasiatic",
    "Japanese": "Japonic", "Korean": "Koreanic", "Thai": "Kra-Dai",
    "Indonesian": "Austronesian", "Turkish": "Turkic",
    "Arabic": "Afro-Asiatic", "Hebrew": "Afro-Asiatic", "Swahili": "Atlantic-Congo",
}


@dataclass(frozen=True)
class Paths:
    """Filesystem locations, resolved against the repo root unless absolute."""

    xlsx: str = "data/raw/nguasach.xlsx"                 # canonical concept table
    semantics_source_csv: str = "data/raw/nguasachV.csv"  # old file, for the Semantics-key join
    hex_labels: str = "data/raw/hexLabels.yaml"           # semantic-pole word clusters
    word2vec_model: str = "model.txt"                     # 60 MB word2vec text (gitignored)
    fasttext_vec: str = "cc.en.300.vec"                   # 4.5 GB fastText text (gitignored)
    psv_dir: str = "third_party/phonetic-similarity-vectors"  # vendored generate.py etc.
    interim: str = "data/interim"
    processed: str = "data/processed"
    results: str = "results"
    figures: str = "figures"

    def resolve(self, attr: str) -> Path:
        raw = Path(getattr(self, attr))
        return raw if raw.is_absolute() else (REPO_ROOT / raw)


@dataclass(frozen=True)
class Config:
    name: str = "default"

    # --- corpus / scope ---
    languages: tuple[str, ...] = ALL_LANGUAGES
    verified_core: tuple[str, ...] = VERIFIED_CORE
    concept_set: str = "all"          # "all" | "swadesh" | path to a newline list
    max_concepts: int | None = None   # truncate after dedupe (smoke configs use this)

    # --- alignment ---
    map: str = "ridge"                # "transvec" | "ridge" | "vecmap" | "nn"
    dim: int = 300                    # PSV embedding dimensionality (generate.py default)
    ridge_alpha: float = 1.0
    k: int = 100                      # retrieval top-k for a "hit"
    csls_k: int = 10                  # CSLS de-hubbing neighbourhood (0 = plain cosine)

    # --- semantics ---
    semantic_dim: int = 50            # PCA target for model.txt (compressSemantics.py default)
    semantic_whiten: bool = True

    # --- baselines stage ---
    baselines: tuple[str, ...] = ("editdist", "orth", "feat")
    char_ngram: tuple[int, ...] = (2, 3)

    # --- align scope ---
    # "all"     -- every ordered language pair + each -> Semantics (N^2; confirmatory)
    # "english" -- English<->X and X->Semantics only (linear in N; exploratory)
    align_scope: str = "all"

    # --- cross-validation ---
    folds: int = 10
    test_folds: int = 1
    seed: int = 20240828

    # --- resampling ---
    null_iters: int = 1000            # label permutations
    bootstrap_iters: int = 2000       # test-concept bootstrap for CIs
    mantel_cap: int = 700             # concept subsample for the O(n^2) Mantel matrices

    # --- translation QC ---
    qc_mode: str = "exclude_flagged"  # "exclude_flagged" | "downweight" | "off"

    # --- internationalism sensitivity ---
    exclude_loanwords: bool = False   # drop loanword.flagged_ids from align/mantel/associate

    paths: Paths = field(default_factory=Paths)

    # ------------------------------------------------------------------ helpers
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        paths = Paths(**data.pop("paths", {}))
        for key in ("languages", "verified_core", "baselines", "char_ngram"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        cfg = cls(paths=paths, **data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        unknown = set(self.languages) - set(ALL_LANGUAGES)
        if unknown:
            raise ValueError(f"unknown languages in config: {sorted(unknown)}")
        if not set(self.verified_core) <= set(self.languages):
            raise ValueError("verified_core must be a subset of languages")
        if self.map not in {"transvec", "ridge", "vecmap", "nn"}:
            raise ValueError(f"unknown map type: {self.map}")
        if self.qc_mode not in {"exclude_flagged", "downweight", "off"}:
            raise ValueError(f"unknown qc_mode: {self.qc_mode}")
        if self.test_folds >= self.folds:
            raise ValueError("test_folds must be < folds")

    def fingerprint(self) -> str:
        """Stable hash of the config, for run manifests / cache keys."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
