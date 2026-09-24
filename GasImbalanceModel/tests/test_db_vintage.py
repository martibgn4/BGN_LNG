"""Vintaging tests. SPEC 8 - "a hard requirement, not a nice-to-have".

The failure mode: ENTSOG and AGSI restate a gas day for several days after
publication. If a backtest reads the restated value, it is scoring a forecast
against information that did not exist when the forecast was made. These tests
assert that a restatement never overwrites, and that an as-of read returns what
was actually published at the time.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from gasbalance.db import OBSERVATION_COLUMNS, RunContext, Store

GAS_DAY = date(2025, 1, 15)
FIRST_PUBLICATION = datetime(2025, 1, 16, 6, 0)
RESTATEMENT = datetime(2025, 1, 20, 6, 0)


@pytest.fixture
def store(tmp_path) -> Store:
    with Store(tmp_path / "test.duckdb") as s:
        yield s


def observation(value: float, vintage: datetime, run_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "obs_date": GAS_DAY,
                "country": "NL",
                "point_key": "ITP-BBL",
                "operator_key": "NL-TSO-0001",
                "direction_key": "entry",
                "series_id": "physical_flow_entry",
                "value": value,
                "unit": "mcm/d",
                "source": "entsog",
                "origin": None,
                "vintage": vintage,
                "source_updated_at": None,
                "is_estimated": False,
                "run_id": run_id,
            }
        ]
    )[OBSERVATION_COLUMNS]


def seed(store: Store) -> None:
    for value, vintage, tag in (
        (100.0, FIRST_PUBLICATION, "first"),
        (112.0, RESTATEMENT, "restated"),
    ):
        ctx = RunContext.new(f"test-{tag}")
        store.register_run(ctx)
        store.write_observations(observation(value, vintage, ctx.run_id), ctx)


def test_restatement_appends_rather_than_overwrites(store: Store) -> None:
    seed(store)
    total = store.con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    assert total == 2, "the restatement overwrote the original observation"


def test_as_of_read_returns_the_value_that_existed_then(store: Store) -> None:
    seed(store)
    early = store.read_as_of(datetime(2025, 1, 17))
    assert len(early) == 1
    assert early["value"].iloc[0] == pytest.approx(100.0)


def test_as_of_read_after_restatement_returns_the_new_value(store: Store) -> None:
    seed(store)
    late = store.read_as_of(datetime(2025, 2, 1))
    assert len(late) == 1
    assert late["value"].iloc[0] == pytest.approx(112.0)


def test_as_of_before_first_publication_sees_nothing(store: Store) -> None:
    seed(store)
    assert store.read_as_of(datetime(2025, 1, 15)).empty


def test_restatement_history_exposes_the_revision(store: Store) -> None:
    seed(store)
    history = store.restatement_history("physical_flow_entry", "NL")
    assert len(history) == 2
    revision = history["value"].iloc[-1] - history["value"].iloc[0]
    assert revision == pytest.approx(12.0)


def test_write_rejects_a_frame_missing_provenance_columns(store: Store) -> None:
    ctx = RunContext.new("test-bad")
    bad = observation(1.0, FIRST_PUBLICATION, ctx.run_id).drop(columns=["vintage"])
    with pytest.raises(ValueError, match="missing columns"):
        store.write_observations(bad, ctx)


def test_run_context_records_the_config_hash() -> None:
    ctx = RunContext.new("test")
    assert ctx.config_hash and len(ctx.config_hash) == 16
