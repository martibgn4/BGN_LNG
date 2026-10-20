"""BSRCH power-outage snapshots (D-043): series registry, the two-window request, the split of a snapshot into
forecast and realised rows, idempotent storage, and the live refusal when live calls are not authorised."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from ttf_forecaster.data import outages as out
from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    BackendUnavailable,
    validate_frame,
)
from ttf_forecaster.data.bloomberg import BloombergConfig
from ttf_forecaster.data.storage import ParquetStore
from ttf_forecaster.pipelines.outages import pull_snapshot

AS_OF = date(2026, 10, 13)
REQUEST = pd.Timestamp("2026-10-19T16:00:00Z")


def _raw(location: str, days: list[date], start_value: float = 100.0) -> pd.DataFrame:
    _, _, fuels = out.LOCATIONS[location]
    rows = []
    for k, d in enumerate(days):
        row = {"Applicable Date": pd.Timestamp(d)}
        row.update({f: start_value + k + j for j, f in enumerate(fuels)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_registry_has_one_series_per_location_and_fuel() -> None:
    assert len(out.OUTAGE_SERIES) == 9 + 12 + 11
    assert "power.fr.outage_nuclear_unavail_mw" in out.OUTAGE_SERIES
    assert "power.de.outage_other_gas_unavail_mw" in out.OUTAGE_SERIES
    assert "power.nordpool.outage_solar_unavail_mw" in out.OUTAGE_SERIES


def test_yaml_registry_matches_the_code_and_is_known() -> None:
    cfg = BloombergConfig.load()
    ids = out.registered_outage_ids(cfg)  # raises if any id is missing or not known
    assert set(ids) == set(out.OUTAGE_SERIES)
    for sid in ids:
        spec = cfg.series[sid]
        assert spec.status == "known" and spec.ticker is None
        assert not spec.is_known  # no ticker: the ticker adapter refuses it
        assert spec.source_file == "bloomberg_outages.yaml" and spec.unit == "MW"


def test_two_windows_per_snapshot_and_the_realised_window_is_one_year() -> None:
    realised = out.bsrch_options("FR", AS_OF, realised=True)
    forward = out.bsrch_options("FR", AS_OF, realised=False)
    assert realised["START_DATE"] == "2025-10-13" and realised["END_DATE"] == "2026-10-12"
    assert forward["START_DATE"] == "2026-10-13" and forward["END_DATE"] == "2028-12-31"
    assert forward["AS_OF_DATE"] == "2026-10-13" and forward["STACK"] == "FALSE"


def test_snapshot_before_the_earliest_honoured_as_of_is_refused() -> None:
    with pytest.raises(ValueError, match="before"):
        out.split_snapshot(_raw("FR", [date(2026, 10, 12)]), "FR", date(2026, 10, 12), REQUEST)


def test_split_puts_the_plan_in_forecast_rows_and_the_past_in_observation_rows() -> None:
    days = [date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14)]
    fc, obs = out.split_snapshot(_raw("FR", days), "FR", AS_OF, REQUEST)
    validate_frame(fc, FORECAST_SCHEMA)
    validate_frame(obs, OBSERVATION_SCHEMA)
    # forecast: applicable day >= as-of, 2 days x 9 fuels
    assert len(fc) == 2 * 9
    assert (fc["issue_time"] == pd.Timestamp("2026-10-13", tz="UTC")).all()
    assert (fc["publication_time"] == REQUEST).all()
    assert fc["pit_grade"].eq("A").all() and fc["lead_hours"].min() == 0.0
    # observation: applicable day < as-of, 1 day x 9 fuels, published D+1 12:00 or the request, whichever is later
    assert len(obs) == 9 and obs["pit_grade"].eq("B").all()
    assert (obs["publication_time"] == REQUEST).all()  # D+1 12:00 (2026-10-13) is before the request
    assert obs["revision"].eq(0).all()  # 2026-10-13 is the earliest as-of: revision 0


def test_a_later_snapshot_revises_a_past_day_as_a_new_revision() -> None:
    days = [date(2026, 10, 12)]
    _, obs_early = out.split_snapshot(_raw("FR", days, 100.0), "FR", date(2026, 10, 13), REQUEST)
    _, obs_late = out.split_snapshot(_raw("FR", days, 200.0), "FR", date(2026, 10, 19), REQUEST)
    assert set(obs_early["revision"]) == {0} and set(obs_late["revision"]) == {6}


def test_missing_fuel_column_is_refused() -> None:
    raw = _raw("FR", [date(2026, 10, 13)]).drop(columns=["Nuclear"])
    with pytest.raises(ValueError, match="lacks fuel columns"):
        out.split_snapshot(raw, "FR", AS_OF, REQUEST)


def test_live_adapter_refuses_when_live_calls_are_not_authorised() -> None:
    cfg = BloombergConfig.load()
    if cfg.live_calls_authorised:  # the committed config must never authorise live calls
        pytest.fail("committed configs authorise live calls")
    with pytest.raises(BackendUnavailable):
        out.LiveOutageAdapter(cfg)


class _FakeAdapter:
    """Stands in for LiveOutageAdapter: returns a fixed BSRCH-shaped result, counts the calls."""

    def __init__(self) -> None:
        self.config = BloombergConfig.load()
        self.calls: list[tuple[str, date]] = []

    def snapshot(self, location: str, as_of: date) -> pd.DataFrame:
        self.calls.append((location, as_of))
        return _raw(location, [as_of - timedelta(days=1), as_of])


def test_pull_snapshot_stores_once_and_a_rerun_on_the_same_day_adds_nothing(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    adapter = _FakeAdapter()
    first = pull_snapshot(AS_OF, store, adapter, request_time=REQUEST)
    assert all(v.startswith("written") for v in first.values()), first
    calls_after_first = len(adapter.calls)
    second = pull_snapshot(AS_OF, store, adapter, request_time=REQUEST)
    assert all(v.startswith("skipped") for v in second.values()), second
    assert len(adapter.calls) == calls_after_first  # the skipped locations made no request
