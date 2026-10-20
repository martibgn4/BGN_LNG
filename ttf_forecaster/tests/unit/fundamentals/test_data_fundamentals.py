"""Fundamentals registry and adapter backends (no network). SYNTHETIC data only."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import (
    OBSERVATION_SCHEMA,
    BackendUnavailable,
    DataError,
    IdentifierStatus,
    IdentifierUnknown,
    validate_frame,
)
from ttf_forecaster.data.fundamentals import (
    LiveFundamentalsAdapter,
    LocalFileFundamentalsAdapter,
    MockFundamentalsAdapter,
    SyntheticFundamentalsConfig,
    load_fundamentals_registry,
    make_fundamentals_adapter,
    to_daily,
)
from ttf_forecaster.settings import ConfigError

SHORT = SyntheticFundamentalsConfig(start="2022-01-01", end="2022-12-31")


class TestRegistry:
    def test_known_series_have_ticker_field_and_provenance(self, registry: dict) -> None:
        for spec in registry.values():
            if spec.status is IdentifierStatus.KNOWN:
                assert spec.ticker and spec.field == "px_last", spec.series_id
                # cited provenance: the sibling project it was copied from, or the 2026-10-19
                # Bloomberg discovery that found and reference-checked it (docs/bloomberg_discovery.md)
                assert ("GasImbalanceModel" in spec.provenance
                        or "Bloomberg discovery" in spec.provenance
                        or "BloombergNEF" in spec.provenance), spec.series_id
            else:
                assert spec.ticker is None, f"{spec.series_id}: non-known must not carry a ticker"

    def test_storage_is_grade_b_with_d2_lag(self, registry: dict) -> None:
        spec = registry["storage.eu.inventory_twh"]
        assert spec.pit_grade.value == "B"
        assert spec.publication_lag.days == 2 and spec.publication_lag.time == "07:00"

    def test_no_public_source_is_known(self, registry: dict) -> None:
        assert all(s.status is not IdentifierStatus.KNOWN
                   for s in registry.values() if s.vendor != "bloomberg")

    def test_duplicate_series_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "sources").mkdir()
        entry = ("series:\n  x.eu.a_gwhd: {vendor: public, ticker: null, unit: GWh/d, "
                 "status: requires_discovery, pit_grade: B}\n")
        (tmp_path / "sources" / "fundamentals.yaml").write_text(entry)
        (tmp_path / "sources" / "bloomberg_fundamentals.yaml").write_text(entry)
        with pytest.raises(ConfigError, match="declared in both"):
            load_fundamentals_registry(str(tmp_path))


class TestMock:
    def test_rows_are_synthetic_and_valid(self, registry: dict) -> None:
        df = MockFundamentalsAdapter(registry, SHORT).get_series(
            ["storage.eu.inventory_twh", "lng.eu.sendout_gwhd"], date(2022, 3, 1), date(2022, 3, 31))
        validate_frame(df, OBSERVATION_SCHEMA)
        assert set(df["source"]) == {"SYNTHETIC"} and set(df["entitlement"]) == {"synthetic"}
        assert df["revision"].max() == 1  # storage vintages present

    def test_publication_lag_applied(self, registry: dict) -> None:
        df = MockFundamentalsAdapter(registry, SHORT).get_series(
            ["storage.eu.inventory_twh"], date(2022, 7, 1), date(2022, 7, 1))
        first = df.loc[df["revision"] == 0].iloc[0]
        assert first["publication_time"] == pd.Timestamp("2022-07-03 07:00", tz="Europe/Amsterdam")
        assert bool(first["publication_time_estimated"])

    def test_deterministic(self, registry: dict) -> None:
        args = (["lng.eu.sendout_gwhd"], date(2022, 1, 1), date(2022, 6, 30))
        a = MockFundamentalsAdapter(registry, SHORT).get_series(*args)
        b = MockFundamentalsAdapter(registry, SHORT).get_series(*args)
        pd.testing.assert_frame_equal(a, b)

    def test_no_generator_raises(self, registry: dict) -> None:
        with pytest.raises(DataError, match="no synthetic generator"):
            MockFundamentalsAdapter(registry, SHORT).get_series(
                ["demand.fr.total_gwhd"], date(2022, 1, 1), date(2022, 1, 5))

    def test_undeclared_series_raises(self, registry: dict) -> None:
        with pytest.raises(ConfigError, match="not declared"):
            MockFundamentalsAdapter(registry, SHORT).get_series(
                ["made.up.series"], date(2022, 1, 1), date(2022, 1, 2))

    def test_storage_identity_holds_in_world(self, world: pd.DataFrame) -> None:
        first = world.loc[world["revision"] == 0]
        inv = to_daily(first, "storage.eu.inventory_twh")
        net = to_daily(first, "storage.eu.injection_gwhd") - to_daily(first, "storage.eu.withdrawal_gwhd")
        diff = inv.diff().dropna() * 1000.0
        np.testing.assert_allclose(diff.to_numpy(), net.loc[diff.index].to_numpy(), atol=1e-6)


class TestLive:
    def test_unknown_identifier_raises_first(self, registry: dict) -> None:
        live = LiveFundamentalsAdapter(registry, authorised=True)
        with pytest.raises(IdentifierUnknown):
            live.get_series(["flows.no.pipeline_exports_gwhd"], date(2024, 1, 1), date(2024, 1, 2))

    def test_not_authorised(self, registry: dict) -> None:
        live = LiveFundamentalsAdapter(registry, authorised=False)
        with pytest.raises(BackendUnavailable, match="not authorised"):
            live.get_series(["storage.eu.inventory_twh"], date(2024, 1, 1), date(2024, 1, 2))

    def test_default_is_not_authorised(self, registry: dict) -> None:
        assert make_fundamentals_adapter("live", registry=registry).authorised is False

    def test_non_bloomberg_has_no_connector(self, registry: dict) -> None:
        spec = replace(registry["flows.no.pipeline_exports_gwhd"], status=IdentifierStatus.KNOWN,
                       ticker="placeholder-for-test")
        live = LiveFundamentalsAdapter({spec.series_id: spec}, authorised=True)
        with pytest.raises(BackendUnavailable, match="no live connector"):
            live.get_series([spec.series_id], date(2024, 1, 1), date(2024, 1, 2))


class TestLocalFile:
    def test_vendor_extract_sign_and_lag(self, registry: dict, tmp_path: Path) -> None:
        root = tmp_path / "fundamentals"
        root.mkdir()
        pd.DataFrame({"event_date": ["2024-01-15", "2024-01-16"], "value": [-50.0, -60.0]}).to_csv(
            root / "demand.be.ldz_mcmd.csv", index=False)
        df = LocalFileFundamentalsAdapter(tmp_path, registry).get_series(
            ["demand.be.ldz_mcmd"], date(2024, 1, 1), date(2024, 1, 31))
        assert list(df["value"]) == [50.0, 60.0]  # Fluxys negative offtake flipped once
        assert df["publication_time"].iloc[0] == pd.Timestamp("2024-01-17 07:00",
                                                              tz="Europe/Amsterdam")
        assert set(df["pit_grade"]) == {"B"}

    def test_missing_file_raises(self, registry: dict, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="no local extract"):
            LocalFileFundamentalsAdapter(tmp_path, registry).get_series(
                ["storage.eu.inventory_twh"], date(2024, 1, 1), date(2024, 1, 2))
