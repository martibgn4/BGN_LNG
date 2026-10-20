"""LNG cargo imports (BSRCH TPD:DEX) and the implied realised Norwegian outage (GAANTOTI), on SYNTHETIC data.

Covers: cargo rows and the side table; zero-filled daily observations; publication D+1 12:00 UTC; idempotent re-pull;
the known-only country list; the implied outage's strictly-past proxy, its clip, its event flag, publication D+1 07:00
Amsterdam, the GWh twin, and a leakage canary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.contracts.units import GasQuality
from ttf_forecaster.data import implied_outage as io
from ttf_forecaster.data import lng_imports as li
from ttf_forecaster.data.base import OBSERVATION_SCHEMA, PitGrade, validate_frame
from ttf_forecaster.data.storage import ParquetStore
from ttf_forecaster.pipelines import lng_imports as pli
from ttf_forecaster.settings import ConfigError, load_config

GQ = GasQuality(gcv_kwh_per_m3=10.55, reference="test")


# ----------------------------------------------------------------- LNG imports

def raw_result(rows: list[dict]) -> pd.DataFrame:
    base = {"Visit_Type": "Import", "Vessel_Name": "V", "Port_Name": "P", "Berth_Name": "B",
            "Departure": "2026-01-02", "Qtymtonnes": -70000.0, "Qtymcm": -90.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_cargo_rows_are_absolute_quantities_and_one_per_imo_arrival() -> None:
    raw = raw_result([{"Imo": 9000001, "Arrival": "2026-01-01"}, {"Imo": 9000002, "Arrival": "2026-01-01"}])
    c = li.cargo_rows(raw, "NL", "KEY")
    assert len(c) == 2 and (c["tonnes"] > 0).all() and (c["mcm"] > 0).all()
    dup = raw_result([{"Imo": 9000001, "Arrival": "2026-01-01"}, {"Imo": 9000001, "Arrival": "2026-01-01"}])
    with pytest.raises(ValueError, match="duplicate"):
        li.cargo_rows(dup, "NL", "KEY")


def test_daily_counts_are_zero_filled_and_published_the_day_after_at_noon() -> None:
    raw = raw_result([{"Imo": 1, "Arrival": "2026-01-01"}, {"Imo": 2, "Arrival": "2026-01-01"},
                      {"Imo": 3, "Arrival": "2026-01-04"}])
    cargoes = li.cargo_rows(raw, "NL", "K")
    daily = li.country_daily(cargoes, li.calendar_of(cargoes))
    obs = li.country_observations(daily, "NL", "NL", pd.Timestamp("2026-01-10", tz="UTC"))
    validate_frame(obs, OBSERVATION_SCHEMA)
    cnt = obs[obs.series_id == "lng.nl.imports_cargoes_count"].set_index("event_time")["value"]
    assert cnt[pd.Timestamp("2026-01-01", tz="UTC")] == 2
    assert cnt[pd.Timestamp("2026-01-02", tz="UTC")] == 0  # a real zero, not NaN
    assert cnt[pd.Timestamp("2026-01-04", tz="UTC")] == 1
    assert (obs.publication_time == obs.event_time + pd.Timedelta(days=1, hours=12)).all()
    assert obs.pit_grade.unique().tolist() == [PitGrade.B.value]


def test_truncated_history_is_unknown_not_zero() -> None:
    raw = raw_result([{"Imo": 1, "Arrival": "2026-01-01"}, {"Imo": 2, "Arrival": "2026-01-05"}])
    cargoes = li.cargo_rows(raw, "ES", "K")
    daily = li.country_daily(cargoes, li.calendar_of(cargoes), truncated_before="2026-01-03")
    obs = li.country_observations(daily, "ES", "ES", pd.Timestamp("2026-01-10", tz="UTC"))
    days = set(obs.loc[obs.series_id == "lng.es.imports_cargoes_count", "event_time"])
    assert pd.Timestamp("2026-01-01", tz="UTC") not in days  # before the cap: not stored at all (not zero)
    assert pd.Timestamp("2026-01-03", tz="UTC") in days  # 0 cargo, inside the complete history


def test_merge_of_several_keys_is_deduplicated_by_imo_and_arrival() -> None:
    k1 = li.cargo_rows(raw_result([{"Imo": 1, "Arrival": "2026-01-01"}, {"Imo": 2, "Arrival": "2026-01-02"}]), "IT", "A")
    k2 = li.cargo_rows(raw_result([{"Imo": 2, "Arrival": "2026-01-02"}, {"Imo": 3, "Arrival": "2026-01-03"}]), "IT", "B")
    merged = li.merge_cargoes(k1, k2)
    assert len(merged) == 3 and sorted(merged["imo"].astype(int)) == [1, 2, 3]


def test_regional_total_is_unknown_when_a_member_is_unknown() -> None:
    cal = pd.date_range("2026-01-01", periods=3, freq="D")
    a = pd.DataFrame({"cargoes": [1.0, 2.0, 3.0], "tonnes": [10.0, 20.0, 30.0], "mcm": [1.0, 2.0, 3.0]}, index=cal)
    b = a.copy()
    b.iloc[0] = np.nan  # a truncated day for one member
    total = li.regional_daily({"NL": a, "BE": b}, ["NL", "BE"])
    assert total["cargoes"].isna().iloc[0] and total["cargoes"].iloc[1] == 4.0
    with pytest.raises(ValueError, match="missing"):
        li.regional_daily({"NL": a}, ["NL", "BE"])


def test_known_countries_need_keys_and_the_committed_config_has_fifteen() -> None:
    cfg = {"countries": {"NL": {"bikeys": ["K"], "status": "known"},
                         "BE": {"bikeys": [], "status": "requires_user_input"}}}
    assert list(li.country_entries(cfg)) == ["NL"]
    with pytest.raises(ValueError, match="no bikeys"):
        li.country_entries({"countries": {"FR": {"bikeys": [" "], "status": "known"}}})
    committed = li.country_entries(load_config("sources/lng_bsrch"))
    assert len(committed) == 15
    assert committed["ES"]["history_truncated_before"] == "2018-07-16"
    assert committed["FR"]["history_truncated_before"] == "2020-02-19"


def test_pull_is_idempotent_and_keeps_one_row_per_cargo(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path / "live")
    raw = raw_result([{"Imo": 1, "Arrival": "2026-01-01"}, {"Imo": 2, "Arrival": "2026-01-03"}])
    cfg = {"countries": {"NL": {"bikeys": ["K1", "K2"], "status": "known", "geography": "NL"}},
           "regions": {"NWE": ["NL"]}}
    first = pli.pull_all(store, lambda key: raw, cfg)
    n_obs = len(store.read_normalised("observation", series_ids=["lng.nl.imports_cargoes_count"]))
    second = pli.pull_all(store, lambda key: raw, cfg)
    assert first["cargoes_total"] == 2 and second["cargoes_total"] == 2  # two keys, same cargoes: not doubled
    assert len(store.read_normalised("observation", series_ids=["lng.nl.imports_cargoes_count"])) == n_obs
    assert first["regions"]["NWE"] != "" and len(pd.read_parquet(pli.side_path(store))) == 2


# ----------------------------------------------------------------- implied outage

def flow_series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


CFG = {"window_days": 5, "percentile": 0.95, "min_periods": 3, "threshold_mcmd": 10.0}


def test_proxy_uses_only_past_days() -> None:
    f = flow_series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 300.0])  # a spike on the last day
    out = io.implied_outage(f, CFG)
    assert out.loc["2026-01-07", "capacity_proxy"] == pytest.approx(100.0)  # the spike does not raise its own proxy
    assert out.loc["2026-01-07", "outage_mcmd"] == 0.0  # high flow is never a negative outage


def test_no_proxy_before_min_periods_gives_nan_not_an_outage() -> None:
    f = flow_series([100.0] * 10)
    out = io.implied_outage(f, CFG)
    # min_periods 3 past flows, shifted one day: days 0..2 have no proxy and no outage claim; day 3 has one
    assert out["capacity_proxy"].iloc[:3].isna().all() and out["outage_mcmd"].iloc[:3].isna().all()
    assert out["capacity_proxy"].iloc[3] == pytest.approx(100.0)


def test_shortfall_and_event_flag_against_the_threshold() -> None:
    f = flow_series([100.0, 100.0, 100.0, 100.0, 100.0, 85.0, 95.0])
    out = io.implied_outage(f, CFG)
    assert out.loc["2026-01-06", "outage_mcmd"] == pytest.approx(15.0)
    assert out.loc["2026-01-06", "event"] == 1.0  # 15 > 10
    assert out.loc["2026-01-07", "outage_mcmd"] == pytest.approx(5.0)
    assert out.loc["2026-01-07", "event"] == 0.0  # 5 is not above 10


def test_leakage_canary_a_change_of_day_d_does_not_move_outages_before_d() -> None:
    base = flow_series(list(np.linspace(300, 320, 20)))
    changed = base.copy()
    changed.iloc[12] = 0.0  # a huge shortfall on day 12 only
    a = io.implied_outage(base, CFG)
    b = io.implied_outage(changed, CFG)
    before = a.index < changed.index[12]
    pd.testing.assert_series_equal(a.loc[before, "outage_mcmd"], b.loc[before, "outage_mcmd"], check_names=False)


def test_observation_rows_are_published_d_plus_1_0700_amsterdam_and_graded_b() -> None:
    cfg = {"flow_series": "flow.no.exit_total_intraday_mcmd", "window_days": 5, "percentile": 0.95, "min_periods": 3,
           "threshold_mcmd": 10.0, "outage_series": "flow.no.implied_outage_mcmd",
           "event_series": "flow.no.implied_outage_event", "flow_gwhd_series": "flow.no.exit_total_intraday_gwhd",
           "outage_gwhd_series": "flow.no.implied_outage_gwhd"}
    days = pd.date_range("2026-01-01", periods=8, freq="D")
    flows = [100.0, 100.0, 100.0, 100.0, 100.0, 85.0, 95.0, 100.0]
    flow = pd.DataFrame({
        "event_time": [pd.Timestamp(d.date(), tz="Europe/Amsterdam").tz_convert("UTC") for d in days],
        "value": flows, "publication_time": pd.Timestamp("2026-02-01", tz="UTC"),
    })
    rows = io.observation_rows(flow, cfg, GQ, pd.Timestamp("2026-02-02", tz="UTC"))
    out = rows[rows.series_id == "flow.no.implied_outage_mcmd"].set_index("event_time")
    day6 = pd.Timestamp("2026-01-06", tz="Europe/Amsterdam").tz_convert("UTC")
    assert out.loc[day6, "value"] == pytest.approx(15.0)
    # flow of 6 Jan is usable from 7 Jan 07:00 Amsterdam (= 06:00 UTC in winter)
    assert out.loc[day6, "publication_time"] == pd.Timestamp("2026-01-07 06:00", tz="UTC")
    assert set(rows.pit_grade) == {PitGrade.B.value}


def test_gwh_twin_is_mcm_times_the_project_gcv() -> None:
    cfg = {"flow_series": "f", "window_days": 5, "percentile": 0.95, "min_periods": 3, "threshold_mcmd": 10.0,
           "outage_series": "o", "event_series": "e", "flow_gwhd_series": "fg", "outage_gwhd_series": "og"}
    days = pd.date_range("2026-01-01", periods=6, freq="D")
    flow = pd.DataFrame({"event_time": [pd.Timestamp(d.date(), tz="Europe/Amsterdam").tz_convert("UTC") for d in days],
                         "value": [100.0, 100.0, 100.0, 100.0, 100.0, 80.0],
                         "publication_time": pd.Timestamp("2026-02-01", tz="UTC")})
    rows = io.observation_rows(flow, cfg, GQ, pd.Timestamp("2026-02-02", tz="UTC"))
    fg = rows[rows.series_id == "fg"].set_index("event_time")["value"]
    assert fg.iloc[-1] == pytest.approx(80.0 * 10.55)
    og = rows[rows.series_id == "og"].set_index("event_time")["value"]
    assert og.iloc[-1] == pytest.approx(20.0 * 10.55)
