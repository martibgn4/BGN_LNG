"""Spark adapter, vessel-snapshot schema and PIT arrival forecasts. SYNTHETIC data only;
no network: the live adapter is driven by a fake transport."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    Backend,
    BackendUnavailable,
    DataError,
    IdentifierUnknown,
    LngDataAdapter,
    SchemaError,
    validate_frame,
)
from ttf_forecaster.data import spark
from ttf_forecaster.settings import load_config

REGIONS = {"T_NWE": ["NWE", "EU"], "T_MED": ["MED", "EU"], "T_ASIA": ["ASIA"]}


def _snaps(rows: list[dict], grade: str = "A") -> pd.DataFrame:
    """rows: snapshot_time, vessel_id, dest, eta, pub (optional), laden (optional)."""
    recs = []
    for r in rows:
        st = pd.Timestamp(r["snapshot_time"], tz="UTC")
        recs.append({
            "snapshot_time": st, "vessel_id": r["vessel_id"], "voyage_id": r.get("voyage"),
            "laden": r.get("laden", True), "origin": "US", "load_time": st,
            "declared_destination": r.get("dest"), "declared_destination_region": None,
            "declared_eta": pd.Timestamp(r["eta"], tz="UTC") if r.get("eta") else pd.NaT,
            "cargo_volume": 160_000.0, "cargo_volume_unit": "m3_LNG",
            "publication_time": pd.Timestamp(r.get("pub", r["snapshot_time"]), tz="UTC"),
            "ingestion_time": pd.Timestamp(r.get("pub", r["snapshot_time"]), tz="UTC"),
            "revision": 0, "unit": "m3_LNG", "source": "SYNTHETIC", "quality_flag": "synthetic",
            "entitlement": "synthetic", "pit_grade": grade, "publication_time_estimated": False,
            "source_timezone": "UTC"})
    df = pd.DataFrame(recs)
    df["declared_eta"] = pd.to_datetime(df["declared_eta"], utc=True)
    for c in ("voyage_id", "declared_destination", "declared_destination_region"):
        df[c] = df[c].astype(object)
    df["revision"] = df["revision"].astype("int64")
    return df


def _value(fc: pd.DataFrame, sid: str) -> float:
    return float(fc.loc[fc["series_id"] == sid, "value"].iloc[0])


# ---------------------------------------------------------------------------------------------
class TestConfig:
    def test_bloomberg_lng_entries_complete(self) -> None:
        cfg = load_config("sources/bloomberg_lng")
        keys = {"vendor", "ticker", "field", "unit", "frequency", "geography", "status",
                "provenance", "pit_grade", "publication_lag", "entitlement"}
        for sid, e in cfg["series"].items():
            assert keys <= set(e), sid
            assert e["status"] == "known"
            assert e["field"] == "px_last"
            if sid.startswith("lng."):
                assert e["pit_grade"] == "B", sid      # revision behaviour unknown

    def test_bloomberg_lng_covers_brief(self) -> None:
        tickers = {e["ticker"].split()[0] for e in
                   load_config("sources/bloomberg_lng")["series"].values()}
        brief = {"LNGG20DT", "LNGG20DC", "LNGG30DT", "LNGG30DC", "LNGG20DU", "LNGG20DQ",
                 "LNGG20DA", "LNGG20DN", "LNGG20DR", "LNGG20DO", "LNGGTRPA", "LNGGTRSU",
                 "LNGGTRCA", "LNGGTRNS", "AHOYAMEU", "AHOYAMNA", "AHOYAMSA", "AHOYAMMD",
                 "AHOYAMAM", "AHOYEXUS", "AHOYEXQA", "AHOYEXTL", "AHOYIMTL", "IKA1", "IKD1",
                 "IKI1"}
        assert brief <= tickers

    def test_spark_unknown_blocks_and_credentials(self, spark_cfg: dict) -> None:
        for block in ("vessels", "arrivals", "ais"):
            assert spark_cfg[block]["status"] == "requires_user_input"
            assert spark_cfg[block]["dataset"] is None
        creds = spark_cfg["api"]["credentials"]
        assert creds == {"client_id_env": "SPARK_CLIENT_ID",
                         "client_secret_env": "SPARK_CLIENT_SECRET"}
        assert "client_credentials" not in json.dumps(spark_cfg)

    def test_every_netback_port_has_via_point(self, spark_cfg: dict) -> None:
        for port in spark_cfg["netbacks"]["fob_ports"].values():
            assert port["via_point"]

    def test_resolve_series(self, spark_cfg: dict) -> None:
        s = spark.resolve_series(spark_cfg, "route.corpus_christi.jkm_price_mi2_usd_mmbtu")
        assert s.params == {"port": "corpus_christi", "field": "jkm_price", "n": "2"}
        assert spark.resolve_series(spark_cfg, "gas.nwe.des_basis_fin_mplus1_usd_mmbtu"
                                    ).params == {"k": "1"}
        for bad in ("route.unknown_port.arb_nea_minus_nwe_mi1_usd_mmbtu",
                    "lng.slots.x.available_m99_count", "spark.vessels.positions"):
            with pytest.raises(IdentifierUnknown):
                spark.resolve_series(spark_cfg, bad)


# ---------------------------------------------------------------------------------------------
class TestSnapshotsAndArrivals:
    T = pd.Timestamp("2024-01-10 18:00", tz="UTC")

    def test_snapshot_publication_before_snapshot_rejected(self) -> None:
        df = _snaps([{"snapshot_time": "2024-01-10", "vessel_id": "V1", "dest": "T_NWE",
                      "eta": "2024-01-15", "pub": "2024-01-09"}])
        with pytest.raises(SchemaError, match="publication_time < snapshot_time"):
            spark.validate_snapshots(df)

    def test_counts_by_horizon_and_region(self) -> None:
        df = _snaps([
            {"snapshot_time": "2024-01-10", "vessel_id": "V1", "dest": "T_NWE", "eta": "2024-01-13"},
            {"snapshot_time": "2024-01-10", "vessel_id": "V2", "dest": "T_MED", "eta": "2024-01-20"},
            {"snapshot_time": "2024-01-10", "vessel_id": "V3", "dest": "T_ASIA", "eta": "2024-01-12"},
            {"snapshot_time": "2024-01-10", "vessel_id": "V4", "dest": None, "eta": "2024-01-12"},
            {"snapshot_time": "2024-01-10", "vessel_id": "V5", "dest": "T_NWE",
             "eta": "2024-01-12", "laden": False},
        ])
        fc = spark.build_arrival_forecasts(df, [self.T], terminal_regions=REGIONS,
                                           regions=("NWE", "EU"), horizons_days=(7, 14))
        validate_frame(fc, FORECAST_SCHEMA)
        assert _value(fc, "lng.arrivals.nwe.cargoes_7d") == 1
        assert _value(fc, "lng.arrivals.eu.cargoes_7d") == 1
        assert _value(fc, "lng.arrivals.eu.cargoes_14d") == 2
        assert _value(fc, "lng.arrivals.eu.volume_14d_m3_lng") == pytest.approx(320_000)
        row = fc.loc[fc["series_id"] == "lng.arrivals.eu.cargoes_7d"].iloc[0]
        assert row["valid_time"] == pd.Timestamp("2024-01-18", tz="UTC")
        assert row["lead_hours"] == 168.0 and row["publication_time"] == self.T

    def test_uses_destination_as_declared_then(self) -> None:
        """A diversion published after t must not change the forecast issued at t."""
        rows = [{"snapshot_time": "2024-01-09", "vessel_id": "V1", "dest": "T_NWE",
                 "eta": "2024-01-14"},
                {"snapshot_time": "2024-01-10 12:00", "vessel_id": "V1", "dest": "T_ASIA",
                 "eta": "2024-02-05", "pub": "2024-01-11"}]
        fc = spark.build_arrival_forecasts(_snaps(rows), [self.T], terminal_regions=REGIONS,
                                           regions=("NWE",), horizons_days=(7,))
        assert _value(fc, "lng.arrivals.nwe.cargoes_7d") == 1
        later = spark.build_arrival_forecasts(_snaps(rows), [self.T + pd.Timedelta(days=1)],
                                              terminal_regions=REGIONS, regions=("NWE",),
                                              horizons_days=(7,))
        assert _value(later, "lng.arrivals.nwe.cargoes_7d") == 0

    def test_leakage_canary(self, mock_adapter: spark.MockSparkAdapter) -> None:
        snaps = mock_adapter.get_lng_vessels(date(2022, 2, 1), date(2022, 4, 30))
        t = pd.Timestamp("2022-03-15 18:00", tz="UTC")
        kw = dict(terminal_regions=spark.MOCK_TERMINAL_REGIONS, regions=("NWE", "EU"))
        before = spark.build_arrival_forecasts(snaps, [t], **kw)
        pert = snaps.copy()
        fut = pert["publication_time"] > t
        pert.loc[fut, "declared_destination"] = "SYN_NWE_A"
        pert.loc[fut, "declared_eta"] = t + pd.Timedelta(days=2)
        after = spark.build_arrival_forecasts(pert, [t], **kw)
        pd.testing.assert_frame_equal(before, after)

    def test_no_snapshots_means_no_rows_not_zero(self) -> None:
        df = _snaps([{"snapshot_time": "2024-01-01", "vessel_id": "V1", "dest": "T_NWE",
                      "eta": "2024-01-20"}])
        fc = spark.build_arrival_forecasts(df, [self.T], terminal_regions=REGIONS,
                                           regions=("NWE",))
        assert fc.empty          # snapshot older than max_snapshot_age

    def test_grade_propagates(self) -> None:
        df = _snaps([{"snapshot_time": "2024-01-10", "vessel_id": "V1", "dest": "T_NWE",
                      "eta": "2024-01-13"}], grade="C")
        fc = spark.build_arrival_forecasts(df, [self.T], terminal_regions=REGIONS,
                                           regions=("NWE",))
        assert set(fc["pit_grade"]) == {"C"}

    def test_mixed_units_raise(self) -> None:
        df = _snaps([{"snapshot_time": "2024-01-10", "vessel_id": "V1", "dest": "T_NWE",
                      "eta": "2024-01-13"}])
        df["cargo_volume_unit"] = "t_LNG"
        with pytest.raises(SchemaError, match="density"):
            spark.build_arrival_forecasts(df, [self.T], terminal_regions=REGIONS,
                                          regions=("NWE",), volume_unit="m3_LNG")

    def test_diversion_counts(self) -> None:
        rows = [
            {"snapshot_time": "2024-01-01", "vessel_id": "V1", "dest": "T_NWE", "eta": "2024-01-25"},
            {"snapshot_time": "2024-01-10", "vessel_id": "V1", "dest": "T_ASIA", "eta": "2024-02-10"},
            {"snapshot_time": "2024-01-01", "vessel_id": "V2", "dest": "T_ASIA", "eta": "2024-01-30"},
            {"snapshot_time": "2024-01-10", "vessel_id": "V2", "dest": "T_MED", "eta": "2024-01-20"},
            {"snapshot_time": "2024-01-10", "vessel_id": "V3", "dest": None, "eta": "2024-01-20"},
        ]
        obs = spark.destination_observations(
            _snaps(rows), [self.T], terminal_regions=REGIONS, focus_region="EU",
            lookback_days=7, max_snapshot_age=pd.Timedelta(days=10))
        validate_frame(obs, OBSERVATION_SCHEMA)
        val = dict(zip(obs["series_id"], obs["value"], strict=True))
        assert val["lng.diversions.eu.away_count_7d"] == 1
        assert val["lng.diversions.eu.into_count_7d"] == 1
        assert val[spark.UNDECLARED_SHARE_SERIES] == pytest.approx(1 / 3)

    def test_realised_zero_filled_and_published_late(
            self, mock_adapter: spark.MockSparkAdapter) -> None:
        obs = mock_adapter.get_realised_arrivals(date(2022, 3, 1), date(2022, 3, 31))
        sub = obs.loc[obs["series_id"] == "lng.arrivals_realised.nwe.cargoes_count"]
        assert len(sub) == 31
        assert (sub["publication_time"] > sub["event_time"] + pd.Timedelta(days=1)).all()


# ---------------------------------------------------------------------------------------------
class TestMockAndLocal:
    def test_mock_is_adapter_and_labelled(self, mock_adapter: spark.MockSparkAdapter) -> None:
        assert isinstance(mock_adapter, LngDataAdapter)
        df = mock_adapter.get_freight_assessments(["freight.atlantic.spark30s_usd_per_d"],
                                                  date(2022, 3, 1), date(2022, 3, 10))
        validate_frame(df, OBSERVATION_SCHEMA)
        assert (df["source"] == "SYNTHETIC").all()
        assert (df["publication_time"] > df["event_time"]).all()
        fc = mock_adapter.get_lng_arrivals(date(2022, 3, 1), date(2022, 3, 3))
        assert set(fc["source"]) == {"SYNTHETIC"}

    def test_mock_rejects_wrong_kind(self, mock_adapter: spark.MockSparkAdapter) -> None:
        with pytest.raises(IdentifierUnknown):
            mock_adapter.get_freight_assessments(
                ["route.sabine_pass.arb_nea_minus_nwe_mi1_usd_mmbtu"], date(2022, 3, 1),
                date(2022, 3, 2))

    def test_local_file_round_trip(self, tmp_path: Path, spark_cfg: dict,
                                   mock_adapter: spark.MockSparkAdapter) -> None:
        d = tmp_path / "spark"
        d.mkdir()
        snaps = mock_adapter.get_lng_vessels(date(2022, 3, 1), date(2022, 3, 5))
        snaps.to_parquet(d / "vessels.parquet")
        obs = mock_adapter.get_freight_assessments(["freight.atlantic.spark30s_usd_per_d"],
                                                   date(2022, 3, 1), date(2022, 3, 10))
        obs.to_parquet(d / "observations.parquet")
        a = spark.make_spark_adapter("local_file", config=spark_cfg, extracts_dir=tmp_path)
        assert a.backend is Backend.LOCAL_FILE
        assert len(a.get_lng_vessels(date(2022, 3, 1), date(2022, 3, 5))) == len(snaps)
        with pytest.raises(DataError, match="no rows"):
            a.get_freight_assessments(["freight.pacific.spark25s_usd_per_d"],
                                      date(2022, 3, 1), date(2022, 3, 10))

    def test_local_csv_naive_timestamps_rejected(self, tmp_path: Path, spark_cfg: dict,
                                                 mock_adapter: spark.MockSparkAdapter) -> None:
        d = tmp_path / "spark"
        d.mkdir()
        obs = mock_adapter.get_freight_assessments(["freight.atlantic.spark30s_usd_per_d"],
                                                   date(2022, 3, 1), date(2022, 3, 4))
        obs["event_time"] = obs["event_time"].dt.tz_localize(None)
        obs.to_csv(d / "observations.csv", index=False)
        a = spark.LocalFileSparkAdapter(spark_cfg, extracts_dir=tmp_path)
        with pytest.raises(SchemaError, match="without a UTC offset"):
            a.get_freight_assessments(["freight.atlantic.spark30s_usd_per_d"],
                                      date(2022, 3, 1), date(2022, 3, 4))


# ---------------------------------------------------------------------------------------------
class FakeTransport:
    """Records requests; answers from a dict keyed by URL substring. No network."""

    def __init__(self, routes: dict[str, tuple[int, object]]) -> None:
        self.routes = routes
        self.calls: list[str] = []

    def request(self, method, url, headers, body, timeout):  # type: ignore[no-untyped-def]
        self.calls.append(url)
        if url.endswith("/oauth/token/"):
            return 201, json.dumps({"accessToken": "tok", "expiresIn": 3600}).encode()
        for key, (status, payload) in self.routes.items():
            if key in url:
                raw = payload if isinstance(payload, str) else json.dumps(payload)
                return status, raw.encode()
        return 404, b"{}"


def _secrets(name: str, required: bool = True) -> str | None:
    return {"SPARK_CLIENT_ID": "id", "SPARK_CLIENT_SECRET": "secret"}.get(name)


def _live(cfg: dict, transport: FakeTransport) -> spark.LiveSparkAdapter:
    return spark.LiveSparkAdapter(cfg, live_calls_authorised=True, transport=transport,
                                  secret_getter=_secrets)


class TestLiveSkeleton:
    def test_refuses_without_authorisation(self, spark_cfg: dict) -> None:
        with pytest.raises(BackendUnavailable, match="not authorised"):
            spark.LiveSparkAdapter(spark_cfg, live_calls_authorised=False)
        with pytest.raises(BackendUnavailable):
            spark.make_spark_adapter("live", config=spark_cfg)   # data_sources: false

    def test_vessels_and_arrivals_unknown(self, spark_cfg: dict) -> None:
        a = _live(spark_cfg, FakeTransport({}))
        with pytest.raises(IdentifierUnknown):
            a.get_lng_vessels(date(2024, 1, 1), date(2024, 1, 2))
        with pytest.raises(IdentifierUnknown):
            a.get_lng_arrivals(date(2024, 1, 1), date(2024, 1, 2))

    def test_missing_credentials(self, spark_cfg: dict) -> None:
        a = spark.LiveSparkAdapter(spark_cfg, live_calls_authorised=True,
                                   transport=FakeTransport({}),
                                   secret_getter=lambda n, required=True: None)
        with pytest.raises(BackendUnavailable, match="credentials"):
            a.get_freight_assessments(["freight.atlantic.spark30s_usd_per_d"],
                                      date(2024, 1, 1), date(2024, 1, 2))

    def test_netback_uri_requires_via_point(self, spark_cfg: dict) -> None:
        a = _live(spark_cfg, FakeTransport({}))
        assert "via-point=cogh" in a.netback_uri("sabine_pass", "2024-01-02")
        cfg = json.loads(json.dumps(spark_cfg))
        cfg["netbacks"]["fob_ports"]["sabine_pass"]["via_point"] = None
        with pytest.raises(spark.ViaPointMissing):
            _live(cfg, FakeTransport({})).netback_uri("sabine_pass", "2024-01-02")

    def test_empty_netback_payload_is_error(self, spark_cfg: dict) -> None:
        tr = FakeTransport({
            "reference-data": (200, {"data": {"staticData": {"sparkReleases": ["2024-01-02"]}}}),
            "/v1.0/netbacks/?": (200, {"data": {"netbacks": []}}),
        })
        with pytest.raises(spark.EmptyPayloadError, match="via-point"):
            _live(spark_cfg, tr).get_route_economics(
                ["route.sabine_pass.arb_nea_minus_nwe_mi1_usd_mmbtu"], date(2024, 1, 1),
                date(2024, 1, 3))

    def test_netback_parse_and_publication(self, spark_cfg: dict) -> None:
        point = {"monthIdx": 1, "neaMinusNwe": {"ttfBasis": {"usdPerMMBtu": "0.42"}}}
        tr = FakeTransport({
            "reference-data": (200, {"data": {"staticData": {"sparkReleases": ["2024-01-02"]}}}),
            "/v1.0/netbacks/?": (200, {"data": {"netbacks": [point]}}),
        })
        df = _live(spark_cfg, tr).get_route_economics(
            ["route.sabine_pass.arb_nea_minus_nwe_mi1_usd_mmbtu"], date(2024, 1, 1),
            date(2024, 1, 3))
        assert df["value"].tolist() == [0.42]
        assert df["publication_time"].iloc[0] == pd.Timestamp("2024-01-02 19:00",
                                                              tz="Europe/London")
        assert df["entitlement"].iloc[0] == "licensed_spark"

    def test_spot_release_with_many_points_raises(self, spark_cfg: dict) -> None:
        rel = [{"releaseDate": "2024-01-02", "data": [{"dataPoints": [
            {"deliveryPeriod": {"startAt": "2024-01-01"},
             "derivedPrices": {"usdPerDay": {"spark": 1}}},
            {"deliveryPeriod": {"startAt": "2024-02-01"},
             "derivedPrices": {"usdPerDay": {"spark": 2}}}]}]}]
        spec = spark.resolve_series(spark_cfg, "freight.atlantic.spark30s_usd_per_d")
        with pytest.raises(DataError, match="expected exactly 1"):
            spark.parse_price_releases(rel, spec)
        curve = spark.resolve_series(spark_cfg, "freight.atlantic.spark30ffa_mplus1_usd_per_d")
        parsed = spark.parse_price_releases(rel, curve)
        assert parsed.set_index("k")["value"].to_dict() == {0: 1.0, 1: 2.0}

    def test_terminal_slot_csv(self) -> None:
        text = "ReleaseDate,Total,M+0,M+1\n2024-01-02,10,3,4\n"
        df = spark.parse_terminal_slots_csv(text, terminal="X",
                                            buckets={"total": "Total", "m0": "M+0", "m1": "M+1"})
        assert df.set_index("bucket")["value"].to_dict() == {"total": 10.0, "m0": 3.0, "m1": 4.0}
        with pytest.raises(DataError, match="lacks columns"):
            spark.parse_terminal_slots_csv(text, terminal="X", buckets={"m2": "M+2"})

    def test_live_not_executed_for_unknown_status(self, spark_cfg: dict) -> None:
        cfg = json.loads(json.dumps(spark_cfg))
        cfg["series"]["freight.atlantic.spark30s_usd_per_d"]["status"] = "requires_discovery"
        tr = FakeTransport({})
        with pytest.raises(IdentifierUnknown):
            _live(cfg, tr).get_freight_assessments(["freight.atlantic.spark30s_usd_per_d"],
                                                   date(2024, 1, 1), date(2024, 1, 2))
        assert tr.calls == []
