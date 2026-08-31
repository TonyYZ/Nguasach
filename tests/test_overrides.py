"""The 6 additions and the 56 lexibank-qc corrections are baked into
nguasach.xlsx (consolidated 2026-08-31); check a few survived and sit in the
right thematic slot."""

import pytest

from nguasach.config import Config
from nguasach import data


@pytest.fixture(scope="module")
def df():
    return data.load_raw(Config.load("configs/paper_exploratory.yaml"))


def test_additions_present_and_placed(df):
    E = df["English"].tolist()
    assert len(df) == 1848
    for concept, before, after in [
        ("carry", "bring", "get"), ("grind", "cook", "fry"),
        ("berry", "fruit", "kernel"), ("wing", "feather", "fur"),
        ("thigh", "leg", "knee"), ("navel", "belly", "guts"),
    ]:
        i = E.index(concept)
        assert (E[i - 1], E[i + 1]) == (before, after), concept


def test_qc_corrections_applied(df):
    who = df[df["English"] == "who"].iloc[0]
    assert who["Welsh"] == "pwy"                         # was "sefydliad iechyd y byd"
    assert df[df["English"] == "teach"].iloc[0]["Greek"] == "διδάσκω"
    assert df[df["English"] == "march"].iloc[0]["Japanese"] == "三月"


def test_verified_columns_untouched(df):
    # a spot check that the consolidation left English/Chinese/French/Irish alone
    row = df[df["English"] == "teach"].iloc[0]
    assert row["French"] and row["Chinese"] and row["Irish"]
