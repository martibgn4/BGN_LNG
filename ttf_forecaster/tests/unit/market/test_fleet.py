"""LNG fleet snapshot (BSRCH COMDTY:SHIP) on a SYNTHETIC result: mapping to VESSEL_SNAPSHOT_SCHEMA, the ETA and staleness
rules, arrival forecasts and destination observations from the snapshot, point-in-time vintages, and idempotence per day."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ttf_forecaster.data import fleet as fl
from ttf_forecaster.data.spark import build_arrival_forecasts
from ttf_forecaster.data.storage import ParquetStore
from ttf_forecaster.pipelines import fleet as pf
from ttf_forecaster.settings import load_config

CFG = load_config("sources/lng_fleet")["fleet_snapshot"]
SNAP = pd.Timestamp("2026-10-20 09:00", tz="UTC")


def raw_rows(rows: list[dict]) -> pd.DataFrame:
    base = {"Vessel Name": "V", "Last Seen": SNAP - pd.Timedelta(hours=1), "Navigation Status": "Moored",
            "Draft %": 80, "Cargo status": "laden", "Cargo product": "Liquefied Natural Gas",
            "Cargo quantity": 70000, "Destination (normalized)": "Rotterdam",
            "Destination Country/Region (normalized)": "Netherlands, Western Europe",
            "ETA": SNAP + pd.Timedelta(days=3), "Depart time": None, "Arrival port": None, "Arrival time": None,
            "Vessel Type 4": "LNG Tanker", "Vessel Class": "LNG", "Beneficial Owner": "X"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_mapping_to_the_vessel_snapshot_schema() -> None:
    raw = raw_rows([{"IMO": 9000001, "Cargo status": "laden"},
                    {"IMO": 9000002, "Cargo status": "ballast", "Cargo quantity": "",
                     "Destination (normalized)": "FOR ORDERS",
                     "Destination Country/Region (normalized)": "Status, Other"}])
    res = fl.normalise(raw, SNAP, CFG)
    v = res.vessels.set_index("vessel_id")
    assert set(v.index) == {"9000001", "9000002"}  # normalise() validates the rows against VESSEL_SNAPSHOT_SCHEMA
    assert bool(v.loc["9000001", "laden"]) and not bool(v.loc["9000002", "laden"])
    assert v.loc["9000001", "declared_destination"] == "Rotterdam"
    assert v.loc["9000001", "declared_destination_region"] == "NWE"
    assert v.loc["9000002", "declared_destination"] is None  # FOR ORDERS is not a destination
    assert v.loc["9000002", "declared_destination_region"] is None  # unmapped country
    assert v["origin"].isna().all()  # no departure-port field: never guessed
    assert v.loc["9000001", "cargo_volume"] == 70000 and v.loc["9000001", "cargo_volume_unit"] == "t_LNG"
    assert v["pit_grade"].eq("A").all()
    assert (v["snapshot_time"] == SNAP).all() and (v["publication_time"] == SNAP).all()


def test_an_eta_before_the_snapshot_is_unknown_not_arrived() -> None:
    raw = raw_rows([{"IMO": 9000003, "ETA": SNAP - pd.Timedelta(days=2)}])
    res = fl.normalise(raw, SNAP, CFG)
    assert res.side.loc[0, "eta_unknown"] and pd.isna(res.vessels.loc[0, "declared_eta"])


def test_a_stale_vessel_is_kept_in_the_side_table_but_not_in_forecasts() -> None:
    raw = raw_rows([{"IMO": 9000004, "Last Seen": SNAP - pd.Timedelta(days=10)},
                    {"IMO": 9000005}])
    res = fl.normalise(raw, SNAP, CFG)
    assert len(res.side) == 2 and res.side["stale"].sum() == 1
    assert list(res.vessels["vessel_id"]) == ["9000005"]


def test_arrival_counts_use_the_snapshot_and_no_row_exists_before_it() -> None:
    raw = raw_rows([{"IMO": 9000006, "ETA": SNAP + pd.Timedelta(days=2)},
                    {"IMO": 9000007, "ETA": SNAP + pd.Timedelta(days=20),
                     "Destination Country/Region (normalized)": "Spain, Western Europe"}])
    vessels = fl.normalise(raw, SNAP, CFG).vessels
    before = build_arrival_forecasts(vessels, [SNAP - pd.Timedelta(days=1)], terminal_regions={},
                                     regions=["NWE", "EU"], horizons_days=(7, 30), volume_unit="t_LNG")
    assert before.empty  # not back-filled: no snapshot existed yet
    out = build_arrival_forecasts(vessels, [SNAP], terminal_regions={}, regions=["NWE", "EU"], horizons_days=(7, 30),
                                  volume_unit="t_LNG")
    c = out[out.series_id.str.contains("cargoes")].set_index("series_id")
    assert c.loc["lng.arrivals.nwe.cargoes_7d", "value"] == 1.0  # the Rotterdam vessel, ETA in 3 days
    assert c.loc["lng.arrivals.eu.cargoes_30d", "value"] == 1.0  # the Spanish vessel, ETA in 20 days
    assert out["pit_grade"].eq("A").all()


def test_a_later_snapshot_does_not_change_an_earlier_forecast() -> None:
    first = fl.normalise(raw_rows([{"IMO": 9000008, "Destination (normalized)": "Rotterdam"}]), SNAP, CFG).vessels
    later_snap = SNAP + pd.Timedelta(days=1)
    later = fl.normalise(raw_rows([{"IMO": 9000008, "Destination (normalized)": "Spain",
                                    "Destination Country/Region (normalized)": "Spain, Western Europe"}]),
                         later_snap, CFG).vessels
    both = pd.concat([first, later], ignore_index=True)
    at_first = build_arrival_forecasts(both, [SNAP], terminal_regions={}, regions=["NWE", "EU"], horizons_days=(7,), volume_unit="t_LNG")
    assert at_first.loc[at_first.series_id == "lng.arrivals.nwe.cargoes_7d", "value"].iloc[0] == 1.0
    at_later = build_arrival_forecasts(both, [later_snap], terminal_regions={}, regions=["NWE", "EU"], horizons_days=(7,), volume_unit="t_LNG")
    assert at_later.loc[at_later.series_id == "lng.arrivals.nwe.cargoes_7d", "value"].iloc[0] == 0.0


def test_a_missing_column_is_refused() -> None:
    with pytest.raises(ValueError, match="lacks columns"):
        fl.normalise(raw_rows([{"IMO": 1}]).drop(columns=["ETA"]), SNAP, CFG)


def test_run_snapshot_is_idempotent_per_day(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "live")
    calls: list[str] = []

    def runner(domain: str, options: dict) -> pd.DataFrame:
        calls.append(domain)
        return raw_rows([{"IMO": 9000009}])

    first = pf.run_snapshot(store, runner, request_time=SNAP, cfg=CFG)
    assert "written" in first and first["written"]["arrivals"] > 0
    second = pf.run_snapshot(store, runner, request_time=SNAP + pd.Timedelta(hours=2), cfg=CFG)
    assert "skipped" in second and len(calls) == 1  # no request on the second run the same day
    fc = store.read_normalised("forecast", series_ids=["lng.arrivals.nwe.cargoes_7d"])
    assert len(fc) == 1
