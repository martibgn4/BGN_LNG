"""Ingest pipeline over a tiny mock adapter. SYNTHETIC data only."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from datetime import date

import pandas as pd
import pytest

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import (
    Backend,
    BackendUnavailable,
    DataError,
    FundamentalsAdapter,
    SchemaError,
)
from ttf_forecaster.data.storage import ParquetStore, VintageConflictError
from ttf_forecaster.data.validation import DataQualityError
from ttf_forecaster.features.point_in_time import resolve_as_of
from ttf_forecaster.pipelines.ingest import ingest
from ttf_forecaster.settings import Paths

from .helpers import obs_frame

T_INGEST = pd.Timestamp("2026-10-16 09:00", tz="UTC")


class MockFundamentals:
    """Minimal FundamentalsAdapter serving synthetic storage; can be told to misbehave."""

    backend = Backend.MOCK

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.calls: list[tuple[tuple[str, ...], date, date]] = []
        cfg = synthetic.SyntheticConfig(start="2023-01-02", end="2023-03-31")
        self._data = synthetic.storage_observations(cfg)

    def get_series(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        self.calls.append((tuple(series_ids), start, end))
        if self.mode == "unavailable":
            raise BackendUnavailable("mock backend switched off")
        d = self._data
        local = d["event_time"].dt.tz_convert("Europe/Amsterdam").dt.date
        out = d.loc[d["series_id"].isin(series_ids) & (local >= start) & (local <= end)].copy()
        if self.mode == "naive":
            out["event_time"] = out["event_time"].dt.tz_localize(None)
        if self.mode == "mixed_units":
            out.loc[out.index[:5], "unit"] = "GWh"
        if self.mode == "not_a_frame":
            return out.to_dict()  # type: ignore[return-value]
        return out.reset_index(drop=True)


SID = "storage.eu.inventory_twh"


@pytest.fixture
def store(tmp_path) -> ParquetStore:
    return ParquetStore(Paths(tmp_path / "data"))


def _call(adapter: MockFundamentals, start=date(2023, 1, 2), end=date(2023, 2, 28)):
    return functools.partial(adapter.get_series, [SID], start, end)


def test_mock_satisfies_protocol() -> None:
    assert isinstance(MockFundamentals(), FundamentalsAdapter)


def test_happy_path(store: ParquetStore) -> None:
    adapter = MockFundamentals()
    s = ingest(_call(adapter), store, kind="observation", source="synthetic",
               ingestion_time=T_INGEST, expected_units={SID: "TWh"})
    assert s.n_received > 0 and s.n_written == s.n_received and s.n_duplicates_skipped == 0
    assert s.is_synthetic and s.quality is not None and s.quality.ok
    assert s.raw_path is not None and s.raw_path.exists()
    assert "ingest_date=2026-10-16" in str(s.raw_path)
    stored = store.read_normalised("observation")
    assert (stored["ingestion_time"] == T_INGEST).all()
    # Raw keeps the adapter's frame as received (its own ingestion_time, not ours).
    raw = pd.read_parquet(s.raw_path)
    assert not (raw["ingestion_time"] == T_INGEST).any()
    assert s.as_dict()["n_written"] == s.n_written


def test_idempotent_reingest(store: ParquetStore) -> None:
    adapter = MockFundamentals()
    first = ingest(_call(adapter), store, kind="observation", source="synthetic",
                   ingestion_time=T_INGEST)
    again = ingest(_call(adapter), store, kind="observation", source="synthetic",
                   ingestion_time=T_INGEST + pd.Timedelta(days=1))
    assert again.n_written == 0 and again.n_duplicates_skipped == first.n_received
    assert len(store.read_normalised("observation")) == first.n_written


def test_overlapping_windows_append_only_new(store: ParquetStore) -> None:
    adapter = MockFundamentals()
    a = ingest(_call(adapter, end=date(2023, 1, 31)), store, kind="observation",
               source="synthetic", ingestion_time=T_INGEST)
    b = ingest(_call(adapter, start=date(2023, 1, 15), end=date(2023, 2, 28)), store,
               kind="observation", source="synthetic", ingestion_time=T_INGEST)
    full = ingest(_call(adapter), store, kind="observation", source="synthetic",
                  ingestion_time=T_INGEST)
    assert full.n_written == 0
    assert len(store.read_normalised("observation")) == a.n_written + b.n_written


def test_stored_view_matches_adapter_as_of(store: ParquetStore) -> None:
    adapter = MockFundamentals()
    ingest(_call(adapter), store, kind="observation", source="synthetic", ingestion_time=T_INGEST)
    as_of = pd.Timestamp("2023-02-10 12:00", tz="UTC")
    direct = resolve_as_of(adapter.get_series([SID], date(2023, 1, 2), date(2023, 2, 28)), as_of)
    stored = resolve_as_of(store.read_normalised("observation"), as_of)
    cols = ["series_id", "event_time", "value", "revision", "publication_time"]
    pd.testing.assert_frame_equal(direct[cols], stored[cols])


def test_backend_unavailable_propagates(store: ParquetStore) -> None:
    with pytest.raises(BackendUnavailable):
        ingest(_call(MockFundamentals("unavailable")), store, kind="observation",
               source="synthetic", ingestion_time=T_INGEST)
    assert not list((store.root / "raw").rglob("*.*"))


def test_schema_failure_keeps_raw_but_not_normalised(store: ParquetStore) -> None:
    with pytest.raises(SchemaError):
        ingest(_call(MockFundamentals("naive")), store, kind="observation", source="synthetic",
               ingestion_time=T_INGEST)
    assert list((store.root / "raw" / "synthetic").rglob("*.parquet"))
    assert store.read_normalised("observation").empty


def test_quality_failure_blocks_unless_disabled(store: ParquetStore) -> None:
    adapter = MockFundamentals("mixed_units")
    with pytest.raises(DataQualityError, match="mixed units"):
        ingest(_call(adapter), store, kind="observation", source="synthetic",
               ingestion_time=T_INGEST)
    assert store.read_normalised("observation").empty
    s = ingest(_call(adapter), store, kind="observation", source="synthetic",
               ingestion_time=T_INGEST, fail_on_quality=False)
    assert s.n_written > 0 and any(w.startswith("HARD") for w in s.warnings)


def test_not_a_frame(store: ParquetStore) -> None:
    with pytest.raises(DataError, match="expected a pandas DataFrame"):
        ingest(_call(MockFundamentals("not_a_frame")), store, kind="observation",
               source="synthetic", ingestion_time=T_INGEST)


def test_empty_result(store: ParquetStore) -> None:
    adapter = MockFundamentals()
    s = ingest(_call(adapter, start=date(2030, 1, 1), end=date(2030, 1, 2)), store,
               kind="observation", source="synthetic", ingestion_time=T_INGEST)
    assert s.n_received == 0 and s.n_written == 0 and s.warnings


def test_naive_ingestion_time_refused(store: ParquetStore) -> None:
    with pytest.raises(DataError, match="tz-aware"):
        ingest(_call(MockFundamentals()), store, kind="observation", source="synthetic",
               ingestion_time=pd.Timestamp("2026-10-16"))


def test_conflicting_redelivery(store: ParquetStore) -> None:
    rows = [("x.eu.y_twh", "2024-01-01", 1.0, "2024-01-02", 0)]
    ingest(lambda: obs_frame(rows), store, kind="observation", source="synthetic",
           ingestion_time=T_INGEST)
    changed = [("x.eu.y_twh", "2024-01-01", 2.0, "2024-01-02", 0)]
    with pytest.raises(VintageConflictError):
        ingest(lambda: obs_frame(changed), store, kind="observation", source="synthetic",
               ingestion_time=T_INGEST)


def test_received_before_publication_warns(store: ParquetStore) -> None:
    rows = [("x.eu.y_twh", "2024-01-01", 1.0, "2030-01-02", 0)]
    s = ingest(lambda: obs_frame(rows), store, kind="observation", source="synthetic",
               ingestion_time=T_INGEST, call_kwargs={})
    assert any("before their publication_time" in w for w in s.warnings)


def test_keep_adapter_ingestion_time(store: ParquetStore) -> None:
    rows = [("x.eu.y_twh", "2024-01-01", 1.0, "2024-01-02", 0)]
    ingest(lambda: obs_frame(rows), store, kind="observation", source="synthetic",
           ingestion_time=T_INGEST, keep_adapter_ingestion_time=True)
    stored = store.read_normalised("observation")
    assert stored["ingestion_time"].iloc[0] == pd.Timestamp("2024-01-02", tz="UTC")


def test_forecast_kind(store: ParquetStore, syn_weather: pd.DataFrame) -> None:
    sub = syn_weather.iloc[:200].reset_index(drop=True)
    s = ingest(lambda: sub, store, kind="forecast", source="synthetic", ingestion_time=T_INGEST,
               raw_name="weather_runs")
    assert s.n_written == 200 and s.raw_path is not None and s.raw_path.name.startswith("weather_runs")
