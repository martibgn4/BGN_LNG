"""Tests for the market and estimation layers.

Aimed at the roll gap, the lookahead guards, and the overlapping-returns
inflation - the three things that would each, on their own, produce a
convincing and entirely false backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wxreturns import backtest
from wxreturns.config import get
from wxreturns.data.base import HistoricalIssueUnsupported
from wxreturns.data.google_weather import GoogleWeatherFetcher
from wxreturns.data.openmeteo import HistoricalForecastFetcher
from wxreturns.market import contracts, returns as market_returns
from wxreturns.model import ridge


# --- provider guards ---------------------------------------------------------

def test_google_refuses_a_past_issue_date(monkeypatch) -> None:
    """Serving today's run under a past issue date is undetectable lookahead."""
    monkeypatch.setenv("GOOGLE_WEATHER_API_KEY", "test-key")
    fetcher = GoogleWeatherFetcher()
    past = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)).date()
    with pytest.raises(HistoricalIssueUnsupported, match="no forecast archive"):
        fetcher.fetch(region="us_gas", issue_date=past)


def test_open_meteo_refuses_leads_beyond_the_archive() -> None:
    """Past lead 7 the API returns nulls, not an error. Catch it at the door."""
    fetcher = HistoricalForecastFetcher()
    limit = get("sources", "open_meteo", "previous_runs", "max_lead_days")
    with pytest.raises(Exception, match="outside the supported range"):
        fetcher._previous_run_columns([limit + 1])
    assert fetcher._previous_run_columns([1, 3, limit]) == [1, 3, limit]


def test_ensemble_refuses_a_horizon_its_model_cannot_reach() -> None:
    """The null-padding trap, found against the live API on 2026-09-07.

    Every ensemble model returns whatever time axis is requested and pads past
    its real horizon with nulls instead of erroring. gfs025 answers a 35-day
    request with 10 days of data and 25 of nulls. Dropping those quietly gives
    a panel that looks complete and is not, so a 30-day request to a 10-day
    model must fail loudly.
    """
    from wxreturns.data.base import SourceUnavailable
    from wxreturns.data.openmeteo import EnsembleFetcher

    with pytest.raises(SourceUnavailable, match="usable days"):
        EnsembleFetcher().fetch(region="us_gas", model="gfs025",
                                forecast_days=30)


def test_only_one_ensemble_model_reaches_thirty_days() -> None:
    """Guards the config against a well-meaning edit back to gfs025."""
    models = get("sources", "open_meteo", "ensemble_models")
    long_range = [name for name, cfg in models.items()
                  if cfg["max_lead_days"] >= 30]
    assert long_range == ["gfs05"], (
        f"expected gfs05 alone to reach 30 days, got {long_range}. The "
        "advertised axis length is not the usable depth.")
    assert models["gfs05"]["default"] is True


def test_unverified_bloomberg_root_is_refused() -> None:
    from wxreturns.data.bloomberg import require_verified
    from wxreturns.data.base import SourceUnavailable

    require_verified("NG")            # the one root confirmed in this repo
    with pytest.raises(SourceUnavailable, match="verified: false"):
        require_verified("CA")


# --- contracts and roll ------------------------------------------------------

def test_ng_expiry_is_three_business_days_before_the_month() -> None:
    """Feb 2025 delivery: 1 Feb is a Saturday, so three business days back is
    Wednesday 29 January. Asserted as a round trip rather than by counting the
    dates in between, which is off by one depending on which end is included.
    """
    expiry = contracts.expiry_date("NG", 2, 2025)
    assert expiry == pd.Timestamp("2025-01-29")
    assert expiry.weekday() < 5
    offset = get("targets", "contracts")["NG"]["expiry_offset_business_days"]
    assert (expiry + pd.tseries.offsets.BDay(offset)) >= pd.Timestamp("2025-02-01")


def test_delivery_window_is_the_contract_month_for_gas() -> None:
    start, end = contracts.delivery_window("NG", 1, 2025)
    assert start == pd.Timestamp("2025-01-01")
    assert end == pd.Timestamp("2025-01-31")


def test_front_contract_window_rolls_within_a_month() -> None:
    """The drift trap: NG1 points at January early in December and at February
    after the December expiry. A fixed weather window would miss that."""
    dates = pd.bdate_range("2024-12-02", "2024-12-31")
    calendar = contracts.roll_calendar("NG", dates)
    front = calendar[calendar["slot"] == "front"].set_index("obs_date")

    early = front.loc[pd.Timestamp("2024-12-02"), "window_start"]
    late = front.loc[pd.Timestamp("2024-12-31"), "window_start"]
    assert early == pd.Timestamp("2025-01-01")
    assert late == pd.Timestamp("2025-02-01"), (
        "after the January contract rolls off, the front must reference "
        "February gas")


def test_ng_delivery_window_is_unreachable_at_archive_leads() -> None:
    """The structural constraint found on real data on 2026-09-07.

    NG expires three business days before its delivery month begins and the
    calendar rolls five business days before that, so the front contract always
    references a month at least eight business days out. The previous-run
    archive stops at seven days of lead. A contract-delivery-window revision
    for front NG is therefore empty ALWAYS, not merely on a short sample.

    This test exists so that if someone later switches window_mode back to
    contract_delivery, the reason it cannot work is stated rather than
    rediscovered.
    """
    dates = pd.bdate_range("2024-11-01", "2025-03-31")
    max_lead = get("sources", "open_meteo", "previous_runs", "max_lead_days")
    report = contracts.window_reachability("NG", dates, max_lead)

    assert report["front"]["reachable_rows"] == 0
    assert report["front"]["min_days_to_window"] > max_lead
    assert report["front1"]["min_days_to_window"] > report["front"]["min_days_to_window"]

    with pytest.raises(ValueError, match="structural"):
        contracts.weather_windows("NG", dates, mode="contract_delivery",
                                  max_lead=max_lead)


def test_balance_of_month_window_is_reachable_at_short_leads() -> None:
    """The mode that does work at leads 1-7, and it must never be empty."""
    dates = pd.bdate_range("2024-11-01", "2025-03-31")
    windows = contracts.weather_windows("NG", dates, mode="balance_of_month")
    assert (windows["window_end"] >= windows["window_start"]).all(), (
        "the balance of the month must never be an empty or reversed span, "
        "including on the last day of a month")
    reach = (windows["window_start"] - windows["obs_date"]).dt.days
    assert (reach == 1).all(), "balance of month starts tomorrow"


def test_configured_window_mode_matches_the_available_lead() -> None:
    """Guards against config drifting into a mode the archive cannot feed."""
    mode = get("features", "revisions", "window_mode")
    assert mode in {"balance_of_month", "forecast_horizon", "contract_delivery"}
    if mode == "contract_delivery":
        raise AssertionError(
            "contract_delivery needs leads of roughly 15+ days and the archive "
            "gives 7. Switch it on only once live collection has accumulated "
            "the far leads.")


def test_front1_is_always_one_contract_beyond_front() -> None:
    dates = pd.bdate_range("2024-11-01", "2024-12-15")
    calendar = contracts.roll_calendar("NG", dates)
    pivot = calendar.pivot_table(index="obs_date", columns="slot",
                                 values="window_start", aggfunc="first")
    assert (pivot["front1"] > pivot["front"]).all()


# --- returns -----------------------------------------------------------------

def _settles() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-11-01", "2024-12-13")
    prices = np.linspace(3.0, 3.4, len(dates))
    settles = pd.DataFrame({
        "obs_date": dates, "commodity": "NG", "generic": 1,
        "px_last": prices, "px_volume": 1000,
    })
    roll = contracts.roll_calendar("NG", dates)
    return settles, roll


def test_roll_day_return_is_dropped_not_smoothed() -> None:
    """The roll gap is a change of underlying, not a return anyone earned."""
    settles, roll = _settles()
    # A large jump exactly on the roll, of the kind a real generic shows.
    front = roll[roll["slot"] == "front"].sort_values("obs_date")
    roll_dates = front.loc[front["code"] != front["code"].shift(1), "obs_date"]
    jump_date = roll_dates.iloc[-1]
    settles.loc[settles["obs_date"] >= jump_date, "px_last"] += 1.0

    daily = market_returns.daily_returns(settles, roll)
    front_returns = daily[daily["slot"] == "front"]
    on_roll = front_returns[front_returns["obs_date"] == jump_date]

    assert on_roll["is_roll_day"].all()
    assert on_roll["return"].isna().all(), (
        "the roll-day return spans two different contracts and must be NaN, "
        "not a 30% gain")
    others = front_returns[~front_returns["is_roll_day"]]["return"].dropna()
    assert (others.abs() < 0.1).all()


def test_forward_return_does_not_bridge_a_roll() -> None:
    settles, roll = _settles()
    daily = market_returns.daily_returns(settles, roll)
    forward = market_returns.forward_returns(daily, horizons=[5])
    front = forward[forward["slot"] == "front"]
    # Some horizon windows span the roll and must be missing rather than
    # silently reinstating the gap the daily series removed.
    assert front["fwd_return_5d"].isna().any()


def test_costs_are_charged_on_position_change_not_per_day() -> None:
    signal = pd.Series([1.0, 1.0, 1.0, 0.0])
    realised = pd.Series([0.0, 0.0, 0.0, 0.0])
    pnl = market_returns.net_pnl(signal, realised, "NG")
    cost = market_returns.transaction_cost("NG")
    # Cost on entry and on exit, nothing for the two days simply held.
    assert np.isclose(pnl.iloc[0], -cost)
    assert np.isclose(pnl.iloc[1], 0.0)
    assert np.isclose(pnl.iloc[2], 0.0)
    assert np.isclose(pnl.iloc[3], -cost)


# --- estimation --------------------------------------------------------------

def test_standardiser_survives_a_constant_column() -> None:
    features = np.column_stack([np.ones(50), np.arange(50, dtype=float)])
    fitted = ridge.Standardiser.fit(features)
    assert np.isfinite(fitted.apply(features)).all()


def test_hac_lag_is_at_least_the_horizon() -> None:
    for horizon in get("targets", "horizons_business_days"):
        assert ridge.hac_lag_for(horizon) > horizon


def test_hac_inflates_errors_when_regressor_and_errors_are_both_persistent() -> None:
    """The whole point of HAC, and the precise condition under which it bites.

    Overlapping returns alone are NOT enough. The HAC correction acts on the
    score x_t*e_t, so with an iid regressor the autocorrelation in e_t is
    multiplied by white noise and largely cancels - HAC and white errors come
    out about the same. It is when the REGRESSOR is also persistent that the
    score inherits the dependence and naive errors understate.

    That is exactly this project's case: a cold snap revises the forecast in
    the same direction for several days running, so the revision features are
    strongly autocorrelated and the overlapping returns are too. Tested at that
    condition rather than at the easier one that would have passed vacuously.

    Compared against the same sandwich at zero lags, so the only difference is
    the Bartlett term.
    """
    rng = np.random.default_rng(0)
    n, horizon, persistence = 800, 20, 0.9

    innovations = rng.normal(size=n)
    x = np.empty(n)
    x[0] = innovations[0]
    for i in range(1, n):
        x[i] = persistence * x[i - 1] + innovations[i]
    x = x.reshape(-1, 1)

    noise = rng.normal(size=n + horizon)
    y = np.array([noise[i:i + horizon].sum() for i in range(n)])

    fitted = ridge.fit(x, y, ["x"], horizon)
    design = fitted.standardiser.apply(x)
    residuals = y - fitted.intercept - design @ fitted.coefficients

    gram_inverse = np.linalg.inv(design.T @ design + fitted.lambda_ * np.eye(1))
    white = ridge._newey_west(design, residuals, 0)
    bartlett = ridge._newey_west(design, residuals, fitted.hac_lags)
    white_se = float(np.sqrt((gram_inverse @ (white * n) @ gram_inverse)[0, 0]))
    hac_se = float(np.sqrt((gram_inverse @ (bartlett * n) @ gram_inverse)[0, 0]))

    assert hac_se > white_se, (
        f"HAC se {hac_se:.4f} must exceed the white se {white_se:.4f} when both "
        "regressor and errors are persistent; if not, every t-stat in the "
        "project is overstated")
    assert np.isclose(fitted.standard_errors[0], hac_se), (
        "fit() must report the Bartlett-corrected error, not the white one")


def test_effective_n_divides_by_the_horizon() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 2))
    y = rng.normal(size=600)
    fitted = ridge.fit(x, y, ["a", "b"], horizon=30)
    assert fitted.effective_n == 600 // 30


def test_oos_r2_is_measured_against_a_zero_forecast() -> None:
    """The denominator is the sum of squared RETURNS, not their variance.

    A forecast of zero therefore scores exactly zero, and a forecast with the
    right sign scores positive. Note what this test does NOT assert: that the
    sample mean scores badly. It scores well here, which is precisely why the
    test-window mean is not allowed as a benchmark - knowing it is lookahead,
    not skill. `make_folds` is what enforces that, not this metric.
    """
    actual = np.array([0.01, -0.02, 0.03, 0.005])
    assert np.isclose(ridge.oos_r2(actual, np.zeros_like(actual)), 0.0)
    # A forecast that leans the right way beats zero; one that leans the wrong
    # way is worse than useless and must score negative.
    assert ridge.oos_r2(actual, actual / 2) > 0.0
    assert ridge.oos_r2(actual, -actual) < 0.0


def test_hit_rate_excludes_untraded_days() -> None:
    actual = np.array([0.05, -0.05, 0.01])
    predicted = np.array([0.05, -0.05, 0.0])
    outcome = ridge.directional_hit_rate(actual, predicted, threshold=0.01)
    assert outcome["n_traded"] == 2
    assert np.isclose(outcome["hit_rate"], 1.0)


# --- walk-forward ------------------------------------------------------------

def test_folds_leave_a_purge_and_embargo_gap() -> None:
    """Without the gap, training rows overlap the first test return and leak."""
    dates = pd.Series(pd.bdate_range("2018-01-01", periods=1400))
    horizon = 21
    folds = backtest.make_folds(dates, horizon)
    embargo = get("model", "backtest", "embargo_business_days")

    for train_index, test_index in folds:
        gap = test_index[0] - train_index[-1]
        assert gap >= horizon + embargo, (
            f"only {gap} rows between train and test at a {horizon}-day "
            "horizon; the last training returns overlap the test window")


def test_folds_are_strictly_forward_looking() -> None:
    dates = pd.Series(pd.bdate_range("2018-01-01", periods=1400))
    for train_index, test_index in backtest.make_folds(dates, 5):
        assert train_index.max() < test_index.min()


def test_short_sample_refuses_rather_than_shrinking_the_test() -> None:
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=200))
    with pytest.raises(ValueError, match="too short"):
        backtest.make_folds(dates, 30)


def test_hit_rate_interval_widens_at_effective_n() -> None:
    """A 52% hit rate is decisive at N=1000 and meaningless at N_eff=33."""
    wide = backtest.hit_rate_interval(0.52, 33)
    narrow = backtest.hit_rate_interval(0.52, 1000)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])
    assert wide[0] < 0.5 < wide[1], (
        "at the effective sample size a 52% hit rate must not exclude a coin toss")
