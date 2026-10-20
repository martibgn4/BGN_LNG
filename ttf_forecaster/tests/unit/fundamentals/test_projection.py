"""Calendar, PIT seasonal normals, net-flow profile, scenario projection. SYNTHETIC only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.fundamentals.scenarios import (
    InsufficientHistory,
    Scenario,
    day_of_gas_year,
    estimate_net_flow_profile,
    gas_year,
    load_scenario_config,
    scenarios_from_config,
    seasonal_normal,
)
from ttf_forecaster.fundamentals.storage_projection import (
    Ratchet,
    project_storage,
    season_end,
    simulate_paths,
)

DAYS = pd.date_range("2016-10-01", "2023-09-30", freq="D")


def _seasonal_pct() -> pd.Series:
    """% WGV/day: injection Apr-Sep, withdrawal Oct-Mar (deterministic, SYNTHETIC)."""
    rng = np.random.default_rng(3)
    base = 0.3 * np.sin(2 * np.pi * (DAYS.dayofyear - 100) / 365.25)
    return pd.Series(base + rng.normal(0, 0.05, len(DAYS)), index=DAYS)


class TestCalendar:
    def test_gas_year_and_day(self) -> None:
        assert gas_year(pd.Timestamp("2023-09-30")) == 2022
        assert gas_year(pd.Timestamp("2023-10-01")) == 2023
        assert day_of_gas_year(pd.Timestamp("2023-10-01")) == 0
        assert list(day_of_gas_year(pd.DatetimeIndex(["2023-10-02", "2024-09-30"]))) == [1, 365]

    def test_season_end(self) -> None:
        assert season_end(pd.Timestamp("2023-11-15")) == pd.Timestamp("2024-03-31")
        assert season_end(pd.Timestamp("2024-02-01")) == pd.Timestamp("2024-03-31")
        assert season_end(pd.Timestamp("2024-05-01")) == pd.Timestamp("2024-09-30")


class TestSeasonalNormal:
    def test_uses_only_prior_gas_years(self) -> None:
        s = pd.Series(1.0, index=DAYS)
        target = pd.DatetimeIndex(["2022-12-01"])
        base = seasonal_normal(s, target, 2022, min_years=3)
        s2 = s.copy()
        s2.loc["2022-10-01":] = 999.0  # current gas year and later must be ignored
        np.testing.assert_allclose(seasonal_normal(s2, target, 2022, min_years=3)["mean"], base["mean"])
        assert base["n_years"].iloc[0] == 6

    def test_insufficient_years_nan(self) -> None:
        s = pd.Series(1.0, index=DAYS)
        out = seasonal_normal(s, pd.DatetimeIndex(["2018-12-01"]), 2018, min_years=3)
        assert np.isnan(out["mean"].iloc[0]) and out["n_years"].iloc[0] == 2


class TestProfile:
    def test_excludes_current_year(self) -> None:
        pct = _seasonal_pct()
        p1 = estimate_net_flow_profile(pct, 2021)
        pct2 = pct.copy()
        pct2.loc["2021-10-01":] += 5.0
        p2 = estimate_net_flow_profile(pct2, 2021)
        np.testing.assert_allclose(p1.pct_per_day, p2.pct_per_day)
        assert max(p1.years) == 2020

    def test_insufficient_history_raises(self) -> None:
        with pytest.raises(InsufficientHistory):
            estimate_net_flow_profile(_seasonal_pct(), 2018, min_years=3)


class TestProjection:
    @pytest.fixture()
    def setup(self) -> tuple:
        cfg = load_scenario_config()
        profile = estimate_net_flow_profile(_seasonal_pct(), 2023)
        return cfg, profile, scenarios_from_config(cfg)

    def _run(self, setup, name="normal", level=700.0, day="2023-11-15", **kw):
        cfg, profile, scen = setup
        d = pd.Timestamp(day)
        return project_storage(level0_twh=level, start_day=d - pd.Timedelta(days=2), wgv_twh=1100.0,
                               profile=profile, scenario=kw.pop("scenario", scen[name]),
                               origin_day=d, scen_cfg=cfg, seed_key=(d.toordinal(),), **kw)

    def test_scenario_ordering_winter(self, setup) -> None:
        r = {n: self._run(setup, n, bootstrap=False).end_fill_pct
             for n in ("normal", "cold", "mild", "low_lng", "high_outage")}
        assert r["cold"] < r["normal"] < r["mild"]
        assert r["low_lng"] < r["normal"] and r["high_outage"] < r["normal"]

    def test_bounds_and_binding(self, setup) -> None:
        crash = Scenario("crash", additive_gwhd=-1e6)
        res = self._run(setup, scenario=crash, bootstrap=True)
        assert res.end_level_twh == pytest.approx(0.0, abs=1e-9) and res.binding_days > 0
        assert all(0.0 <= v <= 100.0 for v in res.quantiles_fill_pct.values())
        flood = Scenario("flood", additive_gwhd=1e6)
        assert self._run(setup, scenario=flood,
                         bootstrap=False).end_level_twh == pytest.approx(1100.0)

    def test_ratchet_limits_withdrawal(self) -> None:
        r = Ratchet.from_config({"enabled": True, "withdrawal_max_gwhd":
                                 {"fill_pct": [0.0, 100.0], "gwhd": [1000.0, 1000.0]}})
        end, low, binding = simulate_paths(500.0, np.full((1, 10), -5000.0), 1000.0, r)
        assert end[0] == pytest.approx(490.0) and binding[0] == 10 and low[0] == end[0]

    def test_bootstrap_reproducible_and_probabilities(self, setup) -> None:
        a, b = self._run(setup), self._run(setup)
        assert a.quantiles_fill_pct == b.quantiles_fill_pct and a.n_paths == 1000
        assert set(a.prob_below_pct) == {20.0, 30.0}  # winter thresholds
        assert all(0.0 <= p <= 1.0 for p in a.prob_below_pct.values())
        q = list(a.quantiles_fill_pct.values())
        assert q == sorted(q)

    def test_rejects_profile_from_origin_gas_year(self, setup) -> None:
        cfg, _, scen = setup
        bad = estimate_net_flow_profile(_seasonal_pct(), 2024)  # includes gas year 2022 = origin's
        with pytest.raises(ValueError, match="strictly before"):
            project_storage(level0_twh=500.0, start_day=pd.Timestamp("2022-11-01"), wgv_twh=1100.0,
                            profile=bad, scenario=scen["normal"],
                            origin_day=pd.Timestamp("2022-11-03"), scen_cfg=cfg)
