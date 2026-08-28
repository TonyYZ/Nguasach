"""Stage ``semantics``: key join + fallback + compressed-space construction.

The full run needs ``model.txt`` (gitignored); those checks skip when it is
absent so CI without the blob still passes the pure-logic tests.
"""

from __future__ import annotations

import pytest

from nguasach.config import Config
from nguasach import data, semantics


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load("configs/default.yaml")


def test_mechanical_key():
    assert semantics._mech_key("apple pie") == "apple_pie_"
    assert semantics._mech_key("  Black-Hole ") == "black_hole_"
    assert semantics._mech_key("how many") == "how_many_"


def test_keys_are_concept_aligned_and_mostly_joined(cfg):
    df = data.load_raw(cfg)
    keys, fallback = semantics.build_keys(cfg, df["English"].tolist())
    assert len(keys) == len(df)
    assert all(k.endswith("_") for k in keys)
    # the legacy Semantics column should cover the large majority
    assert len(fallback) / len(keys) < 0.05


def test_resolve_prefers_direct_then_constituent_mean():
    import numpy as np

    vocab = {"apple_": np.ones(3), "pie_": np.zeros(3), "black_hole_": np.full(3, 9.0)}
    assert np.array_equal(semantics._resolve("black_hole_", vocab, 3), np.full(3, 9.0))
    assert np.allclose(semantics._resolve("apple_pie_", vocab, 3), np.full(3, 0.5))
    assert semantics._resolve("nonesuch_", vocab, 3) is None


@pytest.mark.skipif(
    not Config.load("configs/default.yaml").paths.resolve("word2vec_model").exists(),
    reason="model.txt not present",
)
def test_full_run_writes_word2vec_format(cfg, tmp_path):
    rep = semantics.run(cfg)
    assert rep["n_keys_unresolved"] <= 20
    assert rep["n_pole_seed_missing"] == 0
    emb = cfg.paths.resolve("processed") / "SemanticsEmb.txt"
    header = emb.read_text(encoding="utf-8").splitlines()[0].split()
    assert int(header[0]) == rep["semantics_emb_rows"]
    assert int(header[1]) == cfg.semantic_dim
