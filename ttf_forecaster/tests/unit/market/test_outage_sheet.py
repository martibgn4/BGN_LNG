"""Gas/LNG outage spreadsheet loader on a SYNTHETIC sheet (user request, 2026-10-20): parsing, sign convention, gas-day
expansion, aggregates, unit twins, idempotent reload, revision tracking through the side table."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from ttf_forecaster.contracts.units import GasQuality
from ttf_forecaster.data import outage_sheet as osheet
from ttf_forecaster.data.base import FORECAST_SCHEMA, validate_frame
from ttf_forecaster.data.storage import ParquetStore
from ttf_forecaster.pipelines import outage_sheet as ps

HEADER = ["Region", "Operator/Source", "Facility", "Start Date", "Duration/End", "Prod./Reduction", "Last Update", "notes"]
CFG = {"sheet_name": "Worksheet", "lng_terminal_country": {"Hammerfest LNG (Snohvit)": "NO"}}
GQ = GasQuality(gcv_kwh_per_m3=10.55, reference="test")
SNAP1 = datetime(2026, 10, 20, 8, 30, 0, tzinfo=timezone.utc)
SNAP2 = datetime(2026, 10, 21, 9, 0, 0, tzinfo=timezone.utc)

ROWS_V1 = [
    ["N. Sea", "Gassco", "Nyhamna", "28-Sep-2026", "30-Sep-2026", "-19.8 mcm/day", "19-Oct-2026",
     "Yearly maintenance\nEVENT - 2026-93\nhttps://umm.example/1"],
    ["N. Sea", "Gassco", "Troll", "19-Oct-2026", "20-Oct-2026", "-5.0 mcm/day", "19-Oct-2026",
     "Unplanned;  compressor failure\nEVENT - 2026-352"],
    ["Atlantic", "Gassco", "Hammerfest LNG (Snohvit)", "21-Oct-2026", "21-Oct-2026", "-18.4 mcm/day", "16-Oct-2026",
     "Yearly maintenance\nEVENT - 2026-1"],
    ["Pacific", "Chevron", "Wheatstone", "07-Oct-2026", "19-Nov-2026", "8.9 mtpa", "15-Oct-2026", "no mcm figure"],
    ["N. Sea", "Gassco", "Kollsnes", "01-Dec-2026", "n.a.", "--", "n.a.", "no end yet"],
    ["N. Sea", "Gassco", "Oseberg", "23-Oct-2026", "24-Oct-2026", "-5.7 mcm/day", "15-Oct-2026",
     "Planned - yearly maintenance"],
]


def make_sheet(path: Path, rows: list[list], mtime: datetime) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Worksheet"
    ws.append(HEADER)
    for r in rows:
        ws.append(r)
    wb.save(path)
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))
    return path


def test_parse_units_categories_and_ids(tmp_path: Path) -> None:
    f = make_sheet(tmp_path / "s.xlsx", ROWS_V1, SNAP1)
    ev = {e.facility: e for e in osheet.parse_events(osheet.read_sheet(f))}
    assert ev["Nyhamna"].reduction_mcmd == pytest.approx(19.8) and ev["Nyhamna"].unit == "mcm/day"  # sign reversed
    assert ev["Nyhamna"].event_id == "2026-93" and ev["Nyhamna"].category == "planned"
    assert ev["Troll"].category == "unplanned"
    assert ev["Wheatstone"].reduction_mcmd is None and ev["Wheatstone"].unit == "mtpa"
    assert ev["Kollsnes"].reduction_mcmd is None and ev["Kollsnes"].end is None


def test_a_missing_event_id_gets_a_stable_hash(tmp_path: Path) -> None:
    rows = [["N. Sea", "Gassco", "Edvard Grieg", "01-Nov-2026", "02-Nov-2026", "-1.0 mcm/day", "19-Oct-2026", "x"]]
    f = make_sheet(tmp_path / "s.xlsx", rows, SNAP1)
    a = osheet.parse_events(osheet.read_sheet(f))[0].event_id
    b = osheet.parse_events(osheet.read_sheet(f))[0].event_id
    assert a == b and a.startswith("h")


def test_gas_days_run_start_to_end_inclusive(tmp_path: Path) -> None:
    f = make_sheet(tmp_path / "s.xlsx", ROWS_V1, SNAP1)
    nyhamna = next(e for e in osheet.parse_events(osheet.read_sheet(f)) if e.facility == "Nyhamna")
    days = osheet.gas_days(nyhamna)
    assert len(days) == 3 and days[0].isoformat() == "2026-09-28" and days[-1].isoformat() == "2026-09-30"


def test_aggregates_are_sea_only_mcmd_with_zero_fill_and_gwh_twin(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "live")
    f = make_sheet(tmp_path / "s.xlsx", ROWS_V1, SNAP1)
    summary = ps.load_snapshot(f, store, cfg=CFG, gas_quality=GQ)
    assert not summary.skipped and summary.events == len(ROWS_V1)
    planned = store.read_normalised("forecast", series_ids=["outage.no.gassco_planned_mcmd"])
    planned = planned.set_index(pd.to_datetime(planned.valid_time, utc=True))["value"]
    # from the snapshot day (2026-10-20): Oseberg 23-24 Oct (5.7); Nyhamna is in the past, so not in the forward plan
    assert planned[pd.Timestamp("2026-10-20", tz="UTC")] == pytest.approx(0.0)
    assert planned[pd.Timestamp("2026-10-21", tz="UTC")] == pytest.approx(18.4)  # Hammerfest LNG, Atlantic, planned
    assert planned[pd.Timestamp("2026-10-23", tz="UTC")] == pytest.approx(5.7)
    assert planned[pd.Timestamp("2026-10-24", tz="UTC")] == pytest.approx(5.7)
    assert planned[pd.Timestamp("2026-10-22", tz="UTC")] == pytest.approx(0.0)  # zero-filled between events, not NaN
    unplanned = store.read_normalised("forecast", series_ids=["outage.no.gassco_unplanned_mcmd"])
    unpl = unplanned.set_index(pd.to_datetime(unplanned.valid_time, utc=True))["value"]
    assert unpl[pd.Timestamp("2026-10-20", tz="UTC")] == pytest.approx(5.0)  # Troll, unplanned, ends 20 Oct
    assert unpl[pd.Timestamp("2026-10-21", tz="UTC")] == pytest.approx(0.0)  # zero-filled over the same horizon
    gwh = store.read_normalised("forecast", series_ids=["outage.no.gassco_planned_gwhd"])
    row = gwh.loc[gwh.valid_time == pd.Timestamp("2026-10-23", tz="UTC")]
    assert row["value"].iloc[0] == pytest.approx(5.7 * 10.55)


def test_pacific_and_unmapped_lng_are_handled(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "live")
    rows = [["Atlantic", "Gassco", "Some LNG Plant", "25-Oct-2026", "26-Oct-2026", "-2.0 mcm/day", "19-Oct-2026",
             "Planned EVENT - 9"]]
    f = make_sheet(tmp_path / "s.xlsx", rows, SNAP1)
    summary = ps.load_snapshot(f, store, cfg=CFG, gas_quality=GQ)
    assert any("not in the country map" in x for x in summary.failures)


def test_reload_of_the_same_file_writes_nothing(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "live")
    f = make_sheet(tmp_path / "s.xlsx", ROWS_V1, SNAP1)
    ps.load_snapshot(f, store, cfg=CFG, gas_quality=GQ)
    before = store.read_normalised("forecast", series_ids=["outage.no.gassco_planned_mcmd"])
    again = ps.load_snapshot(f, store, cfg=CFG, gas_quality=GQ)
    after = store.read_normalised("forecast", series_ids=["outage.no.gassco_planned_mcmd"])
    assert again.skipped and len(before) == len(after)


def test_a_revised_event_is_a_new_snapshot_and_the_side_table_keeps_both(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "live")
    f1 = make_sheet(tmp_path / "a.xlsx", ROWS_V1, SNAP1)
    rows2 = [list(r) for r in ROWS_V1]
    rows2[5][4] = "27-Oct-2026"  # Oseberg extended by three days
    f2 = make_sheet(tmp_path / "b.xlsx", rows2, SNAP2)
    ps.load_snapshot(f1, store, cfg=CFG, gas_quality=GQ)
    ps.load_snapshot(f2, store, cfg=CFG, gas_quality=GQ)
    side = pd.read_parquet(ps.side_path(store))
    oseberg = side[side.facility == "Oseberg"]
    assert len(oseberg) == 2 and oseberg.file_sha256.nunique() == 2
    assert set(oseberg["end"].dt.date.astype(str)) == {"2026-10-24", "2026-10-27"}
    # the aggregate issued at SNAP2 covers the extension
    fc = store.read_normalised("forecast", series_ids=["outage.no.gassco_planned_mcmd"])
    fc = fc[fc.issue_time == pd.Timestamp(SNAP2)]
    assert fc.loc[fc.valid_time == pd.Timestamp("2026-10-27", tz="UTC"), "value"].iloc[0] == pytest.approx(5.7)


def test_rows_are_forecast_schema_valid_and_graded_a(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "live")
    f = make_sheet(tmp_path / "s.xlsx", ROWS_V1, SNAP1)
    ps.load_snapshot(f, store, cfg=CFG, gas_quality=GQ)
    fc = store.read_normalised("forecast", series_ids=["outage.no.gassco_planned_mcmd"])
    validate_frame(fc.head(), FORECAST_SCHEMA)
    assert fc.pit_grade.unique().tolist() == ["A"]
    assert fc.publication_time.unique().tolist() == [pd.Timestamp(SNAP1)]
