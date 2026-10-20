"""Derived power series (D-034): sums and rescalings of raw MW components, point-in-time.

Everything here is SYNTHETIC. Component ids are the real internal ids of bloomberg_power.yaml, but
the values are invented to test the derivation rules, not to measure anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data import derived as dv
from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    CURVE_SCHEMA,
    FORECAST_SCHEMA,
    OBSERVATION_SCHEMA,
    DataBundle,
    DataError,
    empty_frame,
    validate_frame,
)
from ttf_forecaster.data.bloomberg import BloombergConfig
from ttf_forecaster.features import power as power_feat
from ttf_forecaster.features.point_in_time import resolve_as_of
from ttf_forecaster.pipelines.build_features import (
    build_feature_matrix,
    leakage_canary,
    load_feature_config,
)
from ttf_forecaster.settings import ConfigError, load_config
from tests.unit.weather.conftest import origins

WIND = "power.de.wind_fc_gw"
ON, OFF = "power.de.wind_onshore_fc_mw", "power.de.wind_offshore_fc_mw"
SMALL_CFG = {
    "derived": {WIND: {"sum": [ON, OFF], "scale": 0.001}},
    "series": {WIND: {"unit": "GW", "schema": "forecast"}},
}
SPEC = dv.load_derived_specs(SMALL_CFG)[WIND]
DAY = "2024-01-15"


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def row(sid: str, event: str, value: float, pub: str, *, unit: str = "MW", grade: str = "B",
        revision: int = 0, source: str = "SYNTHETIC", geography: str = "DE") -> dict[str, object]:
    ent = "synthetic" if source == "SYNTHETIC" else "licensed_bloomberg"
    return {"series_id": sid, "event_time": utc(event), "value": float(value),
            "geography": geography, "publication_time": utc(pub),
            "ingestion_time": utc(pub) + pd.Timedelta(minutes=5), "revision": revision,
            "unit": unit, "source": source, "quality_flag": "synthetic"
            if source == "SYNTHETIC" else "ok", "entitlement": ent, "pit_grade": grade,
            "publication_time_estimated": False, "source_timezone": "Europe/Amsterdam"}


def obs(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ("event_time", "publication_time", "ingestion_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["value"] = df["value"].astype(float)
    df["revision"] = df["revision"].astype("int64")
    df["publication_time_estimated"] = df["publication_time_estimated"].astype(bool)
    return validate_frame(df, OBSERVATION_SCHEMA)


def derive(rows: list[dict[str, object]]) -> pd.DataFrame:
    return dv.derive_observations(obs(rows), {WIND: SPEC})


def value_at(frame: pd.DataFrame, at: str, grade_ok: frozenset[str] | None = None) -> float:
    """Value of the single DAY row visible at ``at`` (NaN if none is visible)."""
    r = resolve_as_of(frame, utc(at), allowed_grades=grade_ok, validate=False)
    r = r.loc[r["series_id"] == WIND]
    return float(r["value"].iloc[0]) if len(r) else np.nan


# Day-ahead components: published the evening before, 16:00 UTC (SYNTHETIC timing).
def fc_pair(on: float = 1000.0, off: float = 500.0, on_pub: str = "2024-01-14 16:00",
            off_pub: str = "2024-01-14 16:00", grade: str = "B") -> list[dict[str, object]]:
    return [row(ON, DAY, on, on_pub, grade=grade), row(OFF, DAY, off, off_pub, grade=grade)]


class TestSpecs:
    def test_unit_from_id_suffix(self) -> None:
        assert dv.unit_for_id(WIND) == ("GW", "MW")
        assert dv.unit_for_id("power.alp_nord.hydro_reservoir_gwh") == ("GWh", "MWh")
        with pytest.raises(ConfigError):
            dv.unit_for_id("power.de.wind_fc_mw")

    def test_real_power_yaml_declares_six_derived_series(self) -> None:
        specs = dv.load_derived_specs(load_config("sources/power"))
        assert set(specs) == {
            "power.de.wind_fc_gw", "power.de.solar_fc_gw", "power.de.wind_gen_gw",
            "power.de.solar_gen_gw", "power.fr.nuclear_avail_gw", "power.fr.nuclear_avail_fc_gw",
            "power.alp_nord.hydro_reservoir_gwh"}
        assert {k for k, s in specs.items() if s.forecast_view} == {
            "power.de.wind_fc_gw", "power.de.solar_fc_gw"}
        assert all(s.scale == pytest.approx(0.001) for s in specs.values())

    def test_real_components_exist_with_the_units_the_rule_needs(self) -> None:
        bb = BloombergConfig.load()
        for spec in dv.load_derived_specs(load_config("sources/power")).values():
            for cid in spec.components:
                assert bb.spec(cid).unit == spec.component_unit, (spec.series_id, cid)

    @pytest.mark.parametrize("node, match", [
        ({"sum": [ON, OFF], "scale": 1.0}, "scale"),
        ({"sum": [], "scale": 0.001}, "sum"),
        ({"sum": [ON, ON], "scale": 0.001}, "duplicate"),
        ({"sum": [ON], "scale": 0.001, "offset": 1}, "unknown keys"),
    ])
    def test_bad_declarations_are_rejected(self, node: dict, match: str) -> None:
        cfg = {"derived": {WIND: node}, "series": {WIND: {"unit": "GW", "schema": "forecast"}}}
        with pytest.raises(ConfigError, match=match):
            dv.load_derived_specs(cfg)

    def test_series_unit_must_match_the_derived_unit(self) -> None:
        cfg = {"derived": {WIND: {"sum": [ON], "scale": 0.001}},
               "series": {WIND: {"unit": "MW", "schema": "forecast"}}}
        with pytest.raises(ConfigError, match="series unit"):
            dv.load_derived_specs(cfg)


class TestDerivation:
    def test_sum_and_scale_with_provenance(self) -> None:
        d = derive(fc_pair())
        assert len(d) == 1
        r = d.iloc[0]
        assert r["value"] == pytest.approx(1.5)
        assert (r["series_id"], r["unit"], r["pit_grade"]) == (WIND, "GW", "B")
        assert r["publication_time"] == utc("2024-01-14 16:00")

    def test_missing_component_gives_nan_never_a_partial_sum(self) -> None:
        d = derive([row(ON, DAY, 1000.0, "2024-01-14 16:00")])
        assert len(d) == 1 and np.isnan(d["value"].iloc[0])
        assert d["quality_flag"].iloc[0] == "missing_input"
        # Once the second component exists, the complete sum takes over from its publication.
        d = derive([row(ON, DAY, 1000.0, "2024-01-14 16:00"),
                    row(OFF, DAY, 500.0, "2024-01-14 17:00")])
        assert np.isnan(value_at(d, "2024-01-14 16:30"))
        assert value_at(d, "2024-01-14 17:00") == pytest.approx(1.5)

    def test_nan_component_value_gives_nan(self) -> None:
        d = derive([row(ON, DAY, np.nan, "2024-01-14 16:00"), row(OFF, DAY, 500.0, "2024-01-14 16:00")])
        assert np.isnan(d["value"].iloc[0])
        assert d["quality_flag"].iloc[0] == "missing_input"

    def test_never_visible_before_the_last_component_is_published(self) -> None:
        # ON at 15:00, OFF at 18:30 (the evening before the day): the sum is not knowable before 18:30.
        d = derive(fc_pair(on_pub="2024-01-14 15:00", off_pub="2024-01-14 18:30"))
        assert np.isnan(value_at(d, "2024-01-14 17:59"))
        assert np.isnan(value_at(d, "2024-01-14 18:29"))
        assert value_at(d, "2024-01-14 18:30") == pytest.approx(1.5)
        assert d.loc[d["value"].notna(), "publication_time"].iloc[0] == utc("2024-01-14 18:30")

    def test_revision_is_visible_only_from_its_publication(self) -> None:
        rows = fc_pair() + [row(ON, DAY, 2000.0, "2024-01-15 08:00", revision=1)]
        d = derive(rows)
        assert value_at(d, "2024-01-15 07:59") == pytest.approx(1.5)
        assert value_at(d, "2024-01-15 08:00") == pytest.approx(2.5)

    def test_grade_is_the_worst_component_grade_and_c_stays_out_of_clean_reads(self) -> None:
        rows = [row(ON, DAY, 1000.0, "2024-01-14 16:00", grade="B"),
                row(OFF, DAY, 500.0, "2024-01-14 16:00", grade="C")]
        d = derive(rows)
        assert d["pit_grade"].iloc[0] == "C"
        assert np.isnan(value_at(d, "2024-01-15 00:00", frozenset({"A", "B"})))
        assert value_at(d, "2024-01-15 00:00") == pytest.approx(1.5)  # allowed_grades=None path

    def test_wrong_component_unit_is_refused(self) -> None:
        with pytest.raises(DataError, match="must be in MW"):
            derive([row(ON, DAY, 1.0, "2024-01-14 16:00", unit="GW"), row(OFF, DAY, 1.0, "2024-01-14 16:00")])

    def test_matches_the_sanctioned_as_of_read_at_every_time(self) -> None:
        """Random vintages and missing rows: at every hour the derived visible value must equal the
        sum of the components' ``resolve_as_of`` values (NaN when a component is not visible)."""
        rng = np.random.default_rng(7)
        rows: list[dict[str, object]] = []
        for k in range(5):
            ev = pd.Timestamp("2024-01-11", tz="UTC") + pd.Timedelta(days=k)
            for sid in (ON, OFF):
                if sid == OFF and k == 2:
                    continue  # offshore absent for one event time
                base = ev - pd.Timedelta(hours=int(rng.integers(6, 12)))
                rows.append(row(sid, str(ev), float(rng.uniform(100, 900)),
                                str(base + pd.Timedelta(hours=int(rng.integers(-3, 3))))))
                if k != 1:
                    rows.append(row(sid, str(ev), float(rng.uniform(100, 900)),
                                    str(ev + pd.Timedelta(hours=int(rng.integers(1, 30)))),
                                    revision=1))
        o = obs(rows)
        d = dv.derive_observations(o, {WIND: SPEC})
        for t in pd.date_range("2024-01-10 00:00", "2024-01-17 00:00", freq="2h", tz="UTC"):
            comps = resolve_as_of(o, t, allowed_grades=None, validate=False)
            if comps.empty:  # nothing published yet: the derived series must be empty too
                assert len(resolve_as_of(d, t, allowed_grades=None, validate=False)) == 0
                continue
            piv = comps.pivot(index="event_time", columns="series_id", values="value")
            piv = piv.reindex(columns=[ON, OFF])
            expected = (SPEC.scale * piv[[ON, OFF]].sum(axis=1)).where(piv[[ON, OFF]].notna().all(axis=1))
            got = resolve_as_of(d, t, allowed_grades=None, validate=False)
            got = got.set_index("event_time")["value"] if len(got) else pd.Series(dtype=float)
            for ev in sorted(set(expected.index) | set(got.index)):
                e = expected.get(ev, np.nan)
                g = got.get(ev, np.nan)
                assert (np.isnan(e) and np.isnan(g)) or e == pytest.approx(g), (t, ev, e, g)

    def test_derived_rows_pass_the_observation_schema(self) -> None:
        d = derive(fc_pair())
        validate_frame(d, OBSERVATION_SCHEMA)


class TestForecastView:
    def test_valid_time_is_the_market_day_at_utc_midnight(self) -> None:
        # Amsterdam-midnight stamping (what the Bloomberg adapter produces) and UTC-midnight stamping
        # must give the same valid day.
        for event, pub in (("2024-01-14 23:00", "2024-01-14 16:00"), (DAY, "2024-01-14 16:00")):
            d = derive([row(ON, event, 1000.0, pub), row(OFF, event, 500.0, pub)])
            fv = dv.forecast_rows(d, {WIND: SPEC})
            validate_frame(fv, FORECAST_SCHEMA)
            assert fv["valid_time"].iloc[0] == utc(DAY)
            assert fv["issue_time"].iloc[0] == utc(pub)
            assert fv["model"].iloc[0] == dv.DERIVED_MODEL

    def test_non_forecast_ids_are_not_exposed_as_forecasts(self) -> None:
        cfg = {"derived": {"power.de.wind_gen_gw": {"sum": [ON], "scale": 0.001}},
               "series": {"power.de.wind_gen_gw": {"unit": "GW", "schema": "observation"}}}
        spec = dv.load_derived_specs(cfg)
        d = dv.derive_observations(obs([row(ON, DAY, 1000.0, "2024-01-14 16:00")]), spec)
        assert len(dv.forecast_rows(d, spec)) == 0


class TestWithDerivedSeries:
    def test_rejects_a_bundle_that_already_carries_the_derived_id(self) -> None:
        o = obs(fc_pair() + [row(WIND, DAY, 1.5, "2024-01-14 16:00", unit="GW")])
        b = DataBundle(o, empty_frame(FORECAST_SCHEMA), empty_frame(CURVE_SCHEMA),
                       empty_frame(CONTRACT_METADATA_SCHEMA))
        with pytest.raises(DataError, match="already present"):
            dv.with_derived_series(b, {WIND: SPEC})

    def test_no_specs_returns_the_bundle_unchanged(self) -> None:
        b = DataBundle.empty()
        assert dv.with_derived_series(b, {}) is b


# ---------------------------------------------------------------------------------------------
# features/power.py sees the derived ids without changes to its templates
# ---------------------------------------------------------------------------------------------

DAYS = pd.date_range("2024-01-01", "2024-01-25", freq="D")


def _power_bundle(*, off_pub_hours: float = -9.0, gap_day: str | None = None,
                  revised: bool = True, seed: int = 0) -> DataBundle:
    """Raw MW components. Day-ahead forecasts stamped at UTC midnight of the delivery day and
    published ``-9h`` (offshore ``off_pub_hours``), actuals published the next day."""
    rng = np.random.default_rng(seed)
    hist = pd.date_range("2018-01-01", "2024-01-25", freq="D")
    rows: list[dict[str, object]] = []
    for ev in DAYS:
        if gap_day and str(ev.date()) == gap_day:
            off_rows = []
        else:
            off_rows = [row(OFF, str(ev.date()), 500.0 + rng.normal(0, 20),
                            str(ev + pd.Timedelta(hours=off_pub_hours)))]
        rows.append(row(ON, str(ev.date()), 1000.0 + rng.normal(0, 30),
                        str(ev - pd.Timedelta(hours=9))))
        rows += off_rows
        for sid, base in (("power.de.solar_fc_tennet_mw", 400.0), ("power.de.solar_fc_50hertz_mw", 300.0),
                          ("power.de.solar_fc_amprion_mw", 200.0)):
            rows.append(row(sid, str(ev.date()), base, str(ev - pd.Timedelta(hours=9))))
        if revised and ev.day == 10:
            rows.append(row(ON, str(ev.date()), 1100.0, str(ev + pd.Timedelta(days=1, hours=8)),
                            revision=1))
    for ev in hist:
        pub = ev + pd.Timedelta(days=1, hours=6)
        rows.append(row("power.de.wind_total_gen_mw", str(ev.date()), 1200.0, str(pub)))
        rows.append(row("power.de.solar_gen_mw", str(ev.date()), 300.0, str(pub)))
        rows.append(row("power.fr.nuclear_gen_mw", str(ev.date()), 40000.0, str(pub)))
    for ev in hist[::7]:
        for sid in ("power.no.hydro_reservoir_mwh", "power.se.hydro_reservoir_mwh",
                    "power.ch.hydro_reservoir_mwh", "power.at.hydro_reservoir_mwh",
                    "power.fr.hydro_reservoir_mwh"):
            rows.append(row(sid, str(ev.date()), 1000.0, str(ev + pd.Timedelta(days=7)), unit="MWh"))
    return DataBundle(obs(rows), empty_frame(FORECAST_SCHEMA), empty_frame(CURVE_SCHEMA),
                      empty_frame(CONTRACT_METADATA_SCHEMA))


class TestPowerWiring:
    def test_wind_forecast_level_and_anomaly_come_from_the_components(self) -> None:
        o = origins("2024-01-15", "2024-01-15")
        f = power_feat.build_power_features(_power_bundle(), o, {})
        # ON ~1000 + OFF ~500 MW -> ~1.5 GW; the actual normal is 1.2 GW (1200 MW, derived).
        assert f["power__de_wind_fc_gw"].iloc[0] == pytest.approx(1.5, abs=0.2)
        assert f["power__de_wind_fc_anom_gw"].iloc[0] == pytest.approx(
            f["power__de_wind_fc_gw"].iloc[0] - 1.2, abs=1e-9)
        assert f.attrs["pit_grade"]["power__de_wind_fc_gw"] == "B"

    def test_a_missing_component_leaves_the_feature_nan_and_flagged(self) -> None:
        # Origin 15 Jan feeds delivery day 16 Jan, the day with no offshore row.
        o = origins("2024-01-15", "2024-01-16")
        f = power_feat.build_power_features(_power_bundle(gap_day="2024-01-16"), o, {})
        assert np.isnan(f["power__de_wind_fc_gw"].iloc[0])
        assert np.isfinite(f["power__de_wind_fc_gw"].iloc[1])  # delivery 17 Jan is complete
        assert "power__de_wind_fc_gw" in f.attrs["flags"]

    def test_the_bundle_passed_in_is_not_mutated(self) -> None:
        b = _power_bundle()
        n = len(b.observations)
        power_feat.build_power_features(b, origins("2024-01-15", "2024-01-15"), {})
        assert len(b.observations) == n


# ---------------------------------------------------------------------------------------------
# Leakage canary on the power family, synthetic bundle with raw MW components (PIT policy 6)
# ---------------------------------------------------------------------------------------------


def _canary_bundle(off_pub_hours: float) -> DataBundle:
    return _power_bundle(off_pub_hours=off_pub_hours, revised=True, seed=3)


def _power_cfg() -> dict:
    cfg = load_feature_config(None)
    cfg["families"]["power"] = {**(cfg["families"].get("power") or {}), "enabled": True}
    return cfg


CHECKS = origins("2024-01-16", "2024-01-16")
ORIG = origins("2024-01-15", "2024-01-18")


def test_leakage_canary_power_family_with_raw_mw_components() -> None:
    b = _canary_bundle(off_pub_hours=-7.0)  # offshore published 17:00 UTC, before the 18:00 origin
    cfg = _power_cfg()
    full = build_feature_matrix(b, ORIG, cfg, families=["power"])
    assert full["power__de_wind_fc_gw"].notna().all()  # non-vacuous: the sum is visible at origin
    rep = leakage_canary(b, ORIG, cfg, families=["power"], check_origins=CHECKS)
    assert rep.checked_origins
    rep.raise_if_failed()


def test_canary_catches_a_derivation_that_publishes_too_early(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: stamping the derived vintage with its EARLIEST component publication makes
    the sum visible before offshore is published. The canary must flag it."""
    b = _canary_bundle(off_pub_hours=-5.5)  # offshore published 18:30 UTC, after the 18:00 origin
    cfg = _power_cfg()
    monkeypatch.setattr(dv, "_vintage_pub", lambda g: g["publication_time"].min())
    rep = leakage_canary(b, ORIG, cfg, families=["power"], check_origins=CHECKS)
    assert not rep.ok
