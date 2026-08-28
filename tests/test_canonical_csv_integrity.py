"""Integrity of the canonical concept table + fold construction (stage ``data``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nguasach.config import ALL_LANGUAGES, Config
from nguasach import data


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load("configs/default.yaml")


@pytest.fixture(scope="module")
def df(cfg) -> pd.DataFrame:
    return data.load_raw(cfg)


def test_shape_and_columns(df):
    assert list(df.columns) == list(ALL_LANGUAGES)
    assert len(df) == 1842
    assert df.index.name == "concept_id"


def test_no_empty_or_corrupt_cells(df):
    report = data.check_integrity(df)
    assert report["empty_cells"] == {}
    assert report["placeholder_only_cells"] == {}
    # the 41 repeated English strings are homographs, expected — just make sure
    # the detector sees them so semantics.py can special-case the join.
    assert report["duplicate_english"]["n_distinct_repeated"] >= 30


def test_corrupt_placeholder_cells_are_rejected():
    bad = pd.DataFrame([["x"] * len(ALL_LANGUAGES)], columns=list(ALL_LANGUAGES))
    bad.loc[0, "Chinese"] = "????"          # the nguasach.csv corruption signature
    with pytest.raises(AssertionError, match="corrupt"):
        data.check_integrity(bad)


def test_whitespace_is_normalized(df):
    assert not df.map(lambda s: s != s.strip() or "  " in s).to_numpy().any()


def test_labels_unique_and_aligned(df):
    labels = data.make_labels(df)
    assert set(labels) == set(ALL_LANGUAGES)
    for lang, col in labels.items():
        assert len(col) == len(df)
        assert len(set(col)) == len(col), f"{lang} has duplicate labels"


def test_folds_partition_deterministically():
    a = data.make_folds(1842, folds=10, seed=20240828)
    b = data.make_folds(1842, folds=10, seed=20240828)
    for (tr1, te1), (tr2, te2) in zip(a, b):
        assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)

    covered = np.concatenate([te for _, te in a])
    covered.sort()
    assert np.array_equal(covered, np.arange(1842))          # every concept tested once

    for tr, te in a:
        assert set(tr.tolist()).isdisjoint(te.tolist())      # no train/test overlap
        assert len(tr) + len(te) == 1842


def test_folds_depend_on_seed():
    a = data.make_folds(500, folds=5, seed=1)
    b = data.make_folds(500, folds=5, seed=2)
    assert not all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_leakage_report_flags_disjoint_and_measures_collisions(df):
    folds = data.make_folds(len(df), folds=10, seed=20240828)
    rep = data.leakage_report(df, folds)
    assert len(rep["per_fold"]) == 10
    # verified-core columns should have a low homograph-collision rate
    for f in rep["per_fold"]:
        for lang in ("English", "Chinese"):
            assert f["surface_collisions"].get(lang, 0) / f["n_test"] < 0.10
