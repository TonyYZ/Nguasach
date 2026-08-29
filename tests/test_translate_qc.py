"""Stage ``translate-qc``: offline per-cell quality flags."""

from __future__ import annotations

from nguasach.config import Config
from nguasach import translate_qc as qc


def test_flag_rules():
    v = {"cashew", "the", "run"}
    # identical to English, distant language -> untranslated
    assert qc._flag_cell("cashew", "cashew", "Swahili", v, 0) == "untranslated"
    # identical to English, close language -> possible cognate (kept)
    assert qc._flag_cell("Arm", "arm", "German", v, 0) == "possible_cognate"
    # ASCII in a non-Latin column -> flagged
    assert qc._flag_cell("run", "run", "Thai", v, 0) in ("untranslated", "ascii_in_nonlatin",
                                                          "english_fallback")
    assert qc._flag_cell("вода", "water", "Russian", v, 0) == "ok"
    assert qc._flag_cell("   ", "water", "Swahili", v, 0) == "empty"


def test_run_and_core_passthrough(tmp_path):
    cfg = Config.load("configs/default.yaml")
    rep = qc.run(cfg)
    for lang in cfg.verified_core:
        assert rep["per_language"][lang]["n_flagged"] == 0
    # Swahili is the known-worst column in this dataset
    assert rep["per_language"]["Swahili"]["flag_rate"] > 0.05
    assert dict(rep["worst"])["Swahili"] == rep["per_language"]["Swahili"]["flag_rate"]


def test_flagged_concepts_respects_qc_mode():
    cfg_off = Config.load("configs/paper_confirmatory.yaml")   # qc_mode: off
    assert qc.flagged_concepts(cfg_off, "Swahili") == set()

    cfg_on = Config.load("configs/default.yaml")               # qc_mode: exclude_flagged
    qc.run(cfg_on)
    assert len(qc.flagged_concepts(cfg_on, "Swahili")) > 50
    assert qc.flagged_concepts(cfg_on, "English") == set()     # verified core never dropped
