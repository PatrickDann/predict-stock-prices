"""Macro series storage against SQLite (portable with Postgres)."""

import pandas as pd
import pytest

from market_intel.storage.db import init_db, make_engine, make_session_factory
from market_intel.storage.macro_repo import get_macro, upsert_macro


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/test.db")
    init_db(engine)
    return make_session_factory(engine)


def _frame(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.DataFrame({"value": values}, index=idx)


def test_upsert_and_read(session_factory):
    with session_factory() as s:
        n = upsert_macro(s, _frame([100.0, 101.0, 102.0]), "GDP")
    assert n == 3
    with session_factory() as s:
        out = get_macro(s, "GDP")
    assert out["value"].tolist() == [100.0, 101.0, 102.0]
    assert out.index.is_monotonic_increasing


def test_upsert_idempotent(session_factory):
    with session_factory() as s:
        upsert_macro(s, _frame([100.0, 101.0]), "GDP")
    with session_factory() as s:
        upsert_macro(s, _frame([100.0, 199.0]), "GDP")  # update second point
    with session_factory() as s:
        out = get_macro(s, "GDP")
    assert len(out) == 2
    assert out["value"].tolist() == [100.0, 199.0]


def test_nan_values_skipped(session_factory):
    df = _frame([100.0, float("nan"), 102.0])
    with session_factory() as s:
        n = upsert_macro(s, df, "GDP")
    assert n == 2  # NaN row skipped
    with session_factory() as s:
        assert len(get_macro(s, "GDP")) == 2
