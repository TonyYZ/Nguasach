"""Stages ``ipa`` and ``phonetics``: normalization + embedding construction."""

from __future__ import annotations

import numpy as np
import pytest

from nguasach.config import Config
from nguasach import data, ipa, phonetics


def test_purify_strips_suprasegmentals_and_glues_diacritics():
    # eSpeak-style spaced tokens with stress + length
    assert ipa.purify("b ˈ i ː") == "b i"
    # tie bar joins neighbours; combining aspiration re-attaches
    assert ipa.purify("t ͡ s ʰ a") == "t͡sʰ a"
    # Mandarin tone digits and syllable dots go away
    assert ipa.purify("s . ˈ i . 5") == "s i"
    # hyphen / non-syllabic mark dropped
    assert ipa.purify("a ̯ ɪ") == "a ɪ"


def test_irish_A_fixup_is_registered():
    assert ipa.LANG_FIXUPS["Irish"]["A"] == "ɑ"


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load("configs/smoke.yaml")


@pytest.fixture(scope="module")
def _has_espeak() -> bool:
    try:
        ipa._setup_espeak()
        return True
    except Exception:
        return False


def test_pipeline_data_ipa_phonetics(cfg, _has_espeak):
    if not _has_espeak:
        pytest.skip("phonemizer/espeakng-loader not available")

    data.run(cfg)
    ir = ipa.run(cfg)
    assert set(ir["written"]) >= {"English", "French", "Irish", "Chinese"}
    assert ir["skipped"] == []
    assert ir["espeak_version"].startswith("1.52")

    ph = phonetics.run(cfg)
    for lang in ("English", "French", "Irish", "Chinese"):
        m = ph["languages"][lang]
        assert m["n_words"] == cfg.max_concepts
        assert m["dim"] == cfg.dim
        assert m["unknown_phones"] == {}, f"{lang}: {m['unknown_phones']}"

    emb = cfg.paths.resolve("processed") / "EnglishEmb.txt"
    head = emb.read_text(encoding="utf-8").splitlines()[0].split()
    assert int(head[0]) == cfg.max_concepts and int(head[1]) == cfg.dim


def test_embed_file_is_deterministic(cfg, _has_espeak):
    if not _has_espeak:
        pytest.skip("bundled espeak-ng not found")
    data.run(cfg)
    ipa.run(cfg)
    v = cfg.paths.resolve("interim") / "FrenchV.txt"
    a = phonetics.embed_file(v, dim=32, seed=7)[1]
    b = phonetics.embed_file(v, dim=32, seed=7)[1]
    assert np.allclose(a, b)
