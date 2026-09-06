"""The round-1 additions (6) + 56 lexibank-qc corrections were baked into
nguasach.xlsx (consolidated 2026-08-31); the round-2 everyday-vocabulary
additions (120) were spliced in at their thematic anchors (2026-09-03). Check a
few of each survived and sit in the right thematic slot."""

import pytest

from nguasach.config import Config
from nguasach import data


@pytest.fixture(scope="module")
def df():
    return data.load_raw(Config.load("configs/paper_exploratory.yaml"))


def test_additions_present_and_placed(df):
    E = df["English"].tolist()
    assert len(df) == 1968
    for concept, before, after in [
        # carry/grind neighbours shifted when round-2 spliced `lift` after
        # `carry` and `raw`/`cooked` after `cook`
        ("carry", "bring", "lift"), ("grind", "cooked", "fry"),
        ("berry", "fruit", "kernel"), ("wing", "feather", "fur"),
        ("thigh", "leg", "knee"), ("navel", "belly", "guts"),
    ]:
        i = E.index(concept)
        assert (E[i - 1], E[i + 1]) == (before, after), concept


def test_additions2_spliced_at_anchors(df):
    E = df["English"].tolist()
    for concept, after in [("breakfast", "soup"), ("throat", "neck"),
                           ("belt", "hat"), ("eleven", "ten"), ("cooked", "raw")]:
        i = E.index(concept)
        assert E[i - 1] == after, f"{concept} not after {after}"


def test_qc_corrections_applied(df):
    who = df[df["English"] == "who"].iloc[0]
    assert who["Welsh"] == "pwy"                         # was "sefydliad iechyd y byd"
    assert df[df["English"] == "teach"].iloc[0]["Greek"] == "διδάσκω"
    assert df[df["English"] == "march"].iloc[0]["Japanese"] == "三月"


def test_verified_columns_untouched(df):
    # a spot check that the consolidation left English/Chinese/French/Irish alone
    row = df[df["English"] == "teach"].iloc[0]
    assert row["French"] and row["Chinese"] and row["Irish"]
