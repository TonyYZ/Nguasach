"""concept_overrides.xlsx is applied, and never touches a verified column."""

import pandas as pd
import pytest

from nguasach.config import Config
from nguasach import data


@pytest.fixture(scope="module")
def cfg():
    return Config.load("configs/paper_exploratory.yaml")


def test_overrides_apply(cfg):
    df = data.load_raw(cfg)
    assert df.attrs.get("overrides_applied", 0) >= 1
    # a representative fix: Welsh "who" was the mistranslated "World Health Org."
    row = df[df["English"] == "who"]
    if not row.empty:
        assert row.iloc[0]["Welsh"] == "pwy"


def test_overrides_skip_verified_columns(cfg):
    ov = pd.read_excel(cfg.paths.resolve("xlsx").parent / "concept_overrides.xlsx",
                       sheet_name="overrides", dtype=str, keep_default_na=False)
    assert set(ov["language"]).isdisjoint(set(cfg.verified_core))


def test_row_count_unchanged_by_overrides(cfg):
    # overrides edit cells, never add/remove rows
    df = data.load_raw(cfg)
    assert len(df) == 1842 + 6      # 6 concept_additions rows
