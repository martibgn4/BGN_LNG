"""Adversarial point-in-time audit (model-risk review). SYNTHETIC data only.

Every probe here tries to break PIT. A probe that finds a real defect is an
``xfail(strict=True)`` with the explanation in ``reason``: when the owning agent fixes it the test
XPASSes, fails the suite, and must be flipped to a plain test. Probes that document an accepted
risk (not a bug) are plain tests whose docstring says so.

The leakage canary (``pipelines.build_features.leakage_canary``) is run on EVERY feature family
through the real assembly path, in both modes (perturb, drop), with check origins placed on DST
transition days.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import (
    MARKET_TZ,
    OBSERVATION_SCHEMA,
    DataBundle,
    validate_frame,
)
from ttf_forecaster.evaluation import diagnostics as d
from ttf_forecaster.features.registry import FAMILIES, FeatureRegistryError, load_dotted
from ttf_forecaster.pipelines.build_features import (
    build_feature_matrix,
    leakage_canary,
    load_feature_config,
)


def _local(days: list[str], hour: int = 19) -> pd.DatetimeIndex:
    return (pd.DatetimeIndex(days) + pd.Timedelta(hours=hour)).tz_localize(MARKET_TZ).tz_convert(
        "UTC")


def _cfg(families: dict[str, dict]) -> dict:
    cfg = load_feature_config(None)
    for name, spec in families.items():
        cur = cfg["families"].get(name, {}) or {}
        cfg["families"][name] = {**cur, **spec, "enabled": True}
    return cfg


def _canary(bundle: DataBundle, origins: pd.DatetimeIndex, cfg: dict, fams: list[str],
            checks: list[str] | None = None, n_checks: int = 2) -> None:
    rep = leakage_canary(bundle, origins, cfg, families=fams,
                         check_origins=_local(checks) if checks else None, n_checks=n_checks)
    assert rep.checked_origins
    rep.raise_if_failed()


# ---------------------------------------------------------------------------------------------
# 1. Leakage canary on every family (through build_feature_matrix)
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def market_bundle() -> DataBundle:
    from tests.unit.features.helpers import bundle

    return bundle()


@pytest.mark.parametrize("fam", ["price", "curve", "cross_commodity", "calendar"])
def test_canary_market_families_across_dst(market_bundle: DataBundle, fam: str) -> None:
    """Origins straddle both 2022 DST switches; settlement 19:00 local is 17:00 UTC in summer
    and 18:00 UTC in winter, so an origin-vs-publication off-by-one-hour bug would show here."""
    o = _local(pd.bdate_range("2022-03-14", "2022-04-08").strftime("%Y-%m-%d").tolist()
               + pd.bdate_range("2022-10-17", "2022-11-11").strftime("%Y-%m-%d").tolist())
    _canary(market_bundle, o, _cfg({}), [fam], checks=["2022-03-25", "2022-03-28",
                                                         "2022-10-28", "2022-10-31"])


def test_canary_interactions_post_stage(market_bundle: DataBundle) -> None:
    o = _local(pd.bdate_range("2022-06-01", "2022-12-30").strftime("%Y-%m-%d").tolist())
    _canary(market_bundle, o, _cfg({}), ["curve", "calendar", "interactions"], n_checks=2)


def test_interactions_do_not_depend_on_sample_start(market_bundle: DataBundle) -> None:
    """Probe of the matrix-stage expanding z-score: an interaction value at a given origin must not
    depend on where the origin sample starts (else two runs with different ``first_origin`` would
    not be comparable). Checked: where both runs define the value, it is identical."""
    days = pd.bdate_range("2022-02-01", "2023-06-29").strftime("%Y-%m-%d").tolist()
    fams = ["curve", "calendar", "interactions"]
    a = build_feature_matrix(market_bundle, _local(days), _cfg({}), families=fams)
    b = build_feature_matrix(market_bundle, _local(days[120:]), _cfg({}), families=fams)
    c = "interactions__expiry_proximity_x_prompt_stress"
    common = b.index[b[c].notna()]
    assert len(common) > 50 and np.allclose(a.loc[common, c], b.loc[common, c])


@pytest.fixture(scope="module")
def fundamentals_bundle() -> DataBundle:
    from ttf_forecaster.data.fundamentals import (
        SyntheticFundamentalsConfig,
        load_fundamentals_registry,
        synthetic_fundamentals,
    )

    obs = synthetic_fundamentals(load_fundamentals_registry(),
                                 SyntheticFundamentalsConfig(start="2017-10-01", end="2021-12-31",
                                                             seed=5))
    b = DataBundle.empty()
    return DataBundle(obs, b.forecasts, b.curves, b.contracts)


def test_canary_storage_with_projection_and_revisions(fundamentals_bundle: DataBundle) -> None:
    """Storage is grade B with revisions; the projection bootstrap is seeded by the origin day.
    Check origins on the autumn DST weekend and around a revision."""
    o = _local(pd.bdate_range("2021-10-25", "2021-11-03").strftime("%Y-%m-%d").tolist())
    _canary(fundamentals_bundle, o, _cfg({"storage": {}}), ["storage"],
            checks=["2021-10-29", "2021-11-01"])


def test_canary_flows(fundamentals_bundle: DataBundle) -> None:
    o = _local(pd.bdate_range("2021-03-22", "2021-04-09").strftime("%Y-%m-%d").tolist())
    _canary(fundamentals_bundle, o, _cfg({"flows": {}}), ["flows"],
            checks=["2021-03-26", "2021-03-29"])


def test_storage_revision_used_only_from_its_publication(fundamentals_bundle: DataBundle) -> None:
    """Revised-data probe: a revision with a huge value, published one second after the origin,
    must not move any storage feature at that origin; one second before, it must."""
    obs = fundamentals_bundle.observations
    o = _local(["2021-11-02"])[0]
    inv = obs.loc[(obs["series_id"] == "storage.eu.inventory_twh")
                  & (obs["publication_time"] <= o)]
    last = inv.sort_values("event_time").iloc[[-1]].copy()
    fams = ["storage"]
    cfg = _cfg({"storage": {"params": {"projection_enabled": False}}})
    base = build_feature_matrix(fundamentals_bundle, pd.DatetimeIndex([o]), cfg, families=fams)
    for delta, should_move in ((pd.Timedelta(seconds=1), False),
                               (pd.Timedelta(seconds=-1), True)):
        rev = last.copy()
        rev["revision"] = int(inv["revision"].max()) + 5
        rev["value"] = rev["value"] * 3.0
        rev["publication_time"] = o + delta
        rev["ingestion_time"] = o + delta
        b2 = DataBundle(pd.concat([obs, rev], ignore_index=True), fundamentals_bundle.forecasts,
                        fundamentals_bundle.curves, fundamentals_bundle.contracts)
        alt = build_feature_matrix(b2, pd.DatetimeIndex([o]), cfg, families=fams)
        moved = not np.allclose(base["storage__level_twh"], alt["storage__level_twh"])
        assert moved is should_move


@pytest.fixture(scope="module")
def lng_bundle() -> DataBundle:
    from tests.unit.lng.conftest import END, START, bloomberg_lng_observations

    from ttf_forecaster.data.spark import MockSparkAdapter, load_spark_config

    ad = MockSparkAdapter(load_spark_config(), world_start="2022-01-01", world_end="2022-07-31")
    route_ids = ["route.sabine_pass.arb_nea_minus_nwe_mi1_usd_mmbtu",
                 "lng.slots.syn_grain.available_m1_count", "lng.slots.syn_gate.available_m1_count"]
    obs = pd.concat([bloomberg_lng_observations(), ad.get_route_economics(route_ids, START, END),
                     ad.get_destination_statistics(START, END),
                     ad.get_realised_arrivals(date(2022, 1, 1), END)], ignore_index=True)
    b = DataBundle.empty()
    return DataBundle(obs, ad.get_lng_arrivals(START, END), b.curves, b.contracts)


def test_canary_lng(lng_bundle: DataBundle) -> None:
    o = _local(pd.bdate_range("2022-03-21", "2022-04-14").strftime("%Y-%m-%d").tolist())
    _canary(lng_bundle, o, _cfg({"lng": {}}), ["lng"], checks=["2022-03-25", "2022-03-28"])


def test_lng_grade_c_arrivals_never_reach_clean_matrix(lng_bundle: DataBundle) -> None:
    """Destination-based arrival forecasts are grade C (D-023): a clean build must drop them."""
    o = _local(pd.bdate_range("2022-04-04", "2022-04-14").strftime("%Y-%m-%d").tolist())
    m = build_feature_matrix(lng_bundle, o, _cfg({"lng": {}}), families=["lng"])
    assert all(g in ("A", "B") for g in m.attrs["pit_grade"].values())
    assert not m.attrs["contains_grade_c"]


@pytest.fixture(scope="module")
def weather_world() -> tuple[DataBundle, dict]:
    from tests.unit.weather.conftest import POINTS

    from ttf_forecaster.data.weather import (
        SYNTH_DET_MODEL,
        SYNTH_ENS_MODEL,
        SyntheticWeatherConfig,
        WeatherPoint,
        WeatherRegion,
        synthetic_point_weather,
    )

    fc, obs = synthetic_point_weather(POINTS, SyntheticWeatherConfig(
        obs_start="2014-10-01", start="2022-12-01", end="2023-02-28", n_members=4))
    region = WeatherRegion("synthetic", "points",
                           tuple(WeatherPoint(p, p, None, None, None) for p in POINTS),
                           "requires_user_input")
    params = {"region_obj": region, "allow_synthetic_equal_weights": True,
              "deterministic": {"model": SYNTH_DET_MODEL, "member": "det"},
              "ensemble": {"model": SYNTH_ENS_MODEL, "min_members": 3},
              "normals": {"min_years": 5}}
    b = DataBundle.empty()
    return DataBundle(obs, fc, b.curves, b.contracts), params


def test_canary_weather(weather_world: tuple[DataBundle, dict]) -> None:
    b, params = weather_world
    o = _local(pd.bdate_range("2023-01-09", "2023-01-20").strftime("%Y-%m-%d").tolist())
    _canary(b, o, _cfg({"weather": {"params": params}}), ["weather"],
            checks=["2023-01-13", "2023-01-17"])


def test_weather_12z_visibility_flips_with_dst() -> None:
    """RISK (timezone): origin = 19:00 Amsterdam = 18:00 UTC in winter, 17:00 UTC in summer. A
    12z run published at issue + 6 h (18:00 UTC) is visible at a winter origin (pub <= as_of) and
    invisible at a summer one. Whether a same-day run is in the feature therefore depends on the
    season, and in winter it hinges on a boundary-inclusive comparison at the exact instant.
    Evidence the true availability time per provider before treating 12z features as clean."""
    winter = _local(["2023-01-16"])[0]
    summer = _local(["2023-07-17"])[0]
    pub_w = pd.Timestamp("2023-01-16 12:00", tz="UTC") + pd.Timedelta(hours=6)
    pub_s = pd.Timestamp("2023-07-17 12:00", tz="UTC") + pd.Timedelta(hours=6)
    assert pub_w <= winter and not pub_s <= summer


@pytest.fixture(scope="module")
def power_bundle() -> DataBundle:
    """Raw MW components (GW ids are derived, D-034); day-ahead forecasts published the evening before."""
    from tests.unit.weather.conftest import bundle, make_fc, make_obs

    hist = pd.date_range("2018-01-01", "2024-01-25", freq="D")
    fc_days = pd.date_range("2024-01-01", "2024-01-27", freq="D")
    issue = pd.date_range("2024-01-01", "2024-01-25", freq="D")
    lag = pd.Timedelta(days=1, hours=18)
    dayahead = pd.Timedelta(hours=-9)
    rng = np.random.default_rng(0)
    wind_fc = 30.0 + fc_days.day.to_numpy() % 5  # GW; split 2/3 onshore, 1/3 offshore
    obs = pd.concat([
        make_obs("power.de.wind_onshore_fc_mw", fc_days, 1000.0 * (2 / 3) * wind_fc, dayahead, "MW"),
        make_obs("power.de.wind_offshore_fc_mw", fc_days, 1000.0 * (1 / 3) * wind_fc, dayahead, "MW"),
        make_obs("power.de.wind_total_gen_mw", hist, 20000 + rng.normal(0, 3000, len(hist)),
                 unit="MW"),
        make_obs("power.de.solar_gen_mw", hist, 5000 + rng.normal(0, 1000, len(hist)), unit="MW"),
        make_obs("power.fr.outage_nuclear_unavail_mw", hist, 13000 + rng.normal(0, 2000, len(hist)),
                 unit="MW"),  # avail = 63 GW - unavail
        make_obs("gas.ttf.m1_eur_mwh", hist[-60:], 30 + rng.normal(0, 1, 60), lag, "EUR/MWh", "A"),
        make_obs("power.de.base_m1_eur_mwh", hist[-60:], 100.0, lag, "EUR/MWh", "A"),
        make_obs("coal.api2.m1_usd_t", hist[-60:], 120.0, lag, "USD/t", "A"),
        make_obs("carbon.eua.dec_eur_t", hist[-60:], 70.0, lag, "EUR/t", "A"),
        make_obs("fx.eurusd.spot_usd_per_eur", hist[-60:], 1.2, lag, "USD/EUR", "A"),
    ], ignore_index=True)
    fc = pd.concat([
        make_fc("power.de.load_fc_gw", issue, range(1, 3), lambda i, ld: 60.0 + i.day),
        make_fc("power.fr.outage_nuclear_unavail_mw", issue, range(1, 30),
                lambda i, ld: 18000.0 - 1000.0 * (i.day % 3), unit="MW"),  # avail 45 + (day % 3) GW
    ], ignore_index=True)
    return bundle(obs, fc)


def test_canary_power(power_bundle: DataBundle) -> None:
    o = _local(pd.bdate_range("2024-01-15", "2024-01-23").strftime("%Y-%m-%d").tolist())
    params = {"economics": {"gas_plant_efficiency": 0.5, "coal_plant_efficiency": 0.4}}
    _canary(power_bundle, o, _cfg({"power": {"params": params}}), ["power"],
            checks=["2024-01-17", "2024-01-19"])


@pytest.fixture(scope="module")
def regime_bundle() -> DataBundle:
    cfg = synthetic.SyntheticConfig(start="2019-01-02", end="2021-06-30", seed=3)
    b = DataBundle.empty()
    return DataBundle(synthetic.storage_observations(cfg), b.forecasts, synthetic.ttf_curve(cfg),
                      b.contracts)


REGIME = {"regime": {"builder": "ttf_forecaster.models.regimes:build_regime_features",
                     "params": {"include_markov": False, "include_changepoint": True}}}


def test_canary_regime_rule_based_and_changepoint(regime_bundle: DataBundle) -> None:
    o = _local(pd.bdate_range("2021-03-15", "2021-04-09").strftime("%Y-%m-%d").tolist())
    _canary(regime_bundle, o, _cfg(REGIME), ["regime"], checks=["2021-03-26", "2021-03-29"])


def test_canary_regime_markov_filtered(regime_bundle: DataBundle) -> None:
    """Filtered (not smoothed) Markov probabilities on an anchored refit schedule must be
    unaffected by later data."""
    spec = {"regime": {**REGIME["regime"], "params": {"include_markov": True,
                                                      "include_changepoint": False}}}
    o = _local(pd.bdate_range("2021-05-24", "2021-06-04").strftime("%Y-%m-%d").tolist())
    _canary(regime_bundle, o, _cfg(spec), ["regime"], checks=["2021-05-28"])


def test_regime_is_not_a_registered_family() -> None:
    """RISK: the regime family exists (``models/regimes.py:build_regime_features``) but is not in
    ``features.registry.FAMILIES`` nor ``configs/features.yaml``, so the ablation harness and
    ``registry.select`` cannot treat it as a family unless a custom builder path is configured."""
    assert "regime" not in FAMILIES


def _text_obs() -> pd.DataFrame:
    days = pd.date_range("2026-01-05", "2026-01-30", freq="D")
    gas_day = (days + pd.Timedelta(hours=6)).tz_localize(MARKET_TZ).tz_convert("UTC")
    n = len(days)
    rng = np.random.default_rng(1)
    pub = gas_day - pd.to_timedelta(rng.integers(1, 72, n), unit="h")  # announced in advance
    df = pd.DataFrame({
        "series_id": "text.nwe.outage_count", "event_time": gas_day,
        "value": rng.integers(0, 3, n).astype(float), "geography": "NWE",
        "publication_time": pub, "ingestion_time": pub, "revision": np.zeros(n, dtype="int64"),
        "unit": "count", "source": "SYNTHETIC", "quality_flag": "synthetic",
        "entitlement": "synthetic", "pit_grade": "B",
        "publication_time_estimated": np.zeros(n, dtype=bool), "source_timezone": MARKET_TZ})
    return validate_frame(df, OBSERVATION_SCHEMA)


def test_text_registry_path_resolves() -> None:
    """Was a CONFIRMED BUG during this review: FAMILIES['text'] pointed to the non-existent
    module ``ttf_forecaster.text.features``. Fixed concurrently by the registry owner; this test
    keeps it fixed."""
    assert callable(load_dotted(FAMILIES["text"].builder))


def test_canary_text_forward_dated_events() -> None:
    """Text events are announced before their gas day (event_time > publication_time). The
    builder must use only announcements published by the origin."""
    b0 = DataBundle.empty()
    b = DataBundle(_text_obs(), b0.forecasts, b0.curves, b0.contracts)
    spec = {"text": {"builder": "ttf_forecaster.text.to_features:build_text_features",
                     "params": {"series": ["text.nwe.outage_count"], "offset_days": 1,
                                "coverage_start": "2026-01-01T00:00Z"}}}
    o = _local(pd.bdate_range("2026-01-08", "2026-01-23").strftime("%Y-%m-%d").tolist())
    cfg = _cfg(spec)
    rep = leakage_canary(b, o, cfg, families=["text"], n_checks=3)
    rep.raise_if_failed()


def test_text_absence_is_coded_zero_after_coverage_start() -> None:
    """RISK (missing-not-at-random): after ``coverage_start`` a missing reviewed event is coded
    0 ("no event"), not NaN. Valid only if the reviewed feed is complete from that date; a review
    backlog would silently read as "no outage". Grade C by default keeps it out of clean runs."""
    b0 = DataBundle.empty()
    obs = _text_obs()
    obs = obs.iloc[:0]
    b = DataBundle(validate_frame(obs, OBSERVATION_SCHEMA), b0.forecasts, b0.curves, b0.contracts)
    f = load_dotted("ttf_forecaster.text.to_features:build_text_features")(
        b, _local(["2026-01-12"]), {"series": ["text.nwe.outage_count"],
                                    "coverage_start": "2026-01-01T00:00Z"})
    assert f.iloc[0, 0] == 0.0


# ---------------------------------------------------------------------------------------------
# 2. Target / overlapping-label leakage
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def null_curve() -> pd.DataFrame:
    cfg = synthetic.SyntheticConfig(start="2021-01-04", end="2022-12-30", seed=9,
                                    contract_noise=0.002)
    return synthetic.ttf_curve(cfg)


def test_t3_targets_never_cross_a_roll(null_curve: pd.DataFrame) -> None:
    """Roll artefact probe: every T3 label is a log price change of ONE contract between two
    settlements on which that contract trades, and its exit is before the contract's last trade."""
    t = d.build_targets(null_curve, 20).dropna(subset=["value"])
    c = null_curve.assign(day=null_curve["trade_date"])
    px = c.pivot_table(index="day", columns="contract_id", values="value")
    pub = pd.DatetimeIndex(t["origin_time"]).tz_convert(MARKET_TZ).normalize().tz_convert("UTC")
    exit_ = pd.DatetimeIndex(t["label_known_time"]).tz_convert(MARKET_TZ).normalize(
    ).tz_convert("UTC")
    p0 = np.array([px.at[a, cid] for a, cid in zip(pub, t["contract_id"], strict=True)])
    p1 = np.array([px.at[b, cid] for b, cid in zip(exit_, t["contract_id"], strict=True)])
    np.testing.assert_allclose(np.log(p1) - np.log(p0), t["value"].to_numpy(), atol=1e-12)


def test_price_features_have_no_roll_jump(null_curve: pd.DataFrame) -> None:
    """The synthetic curve has an 8 % winter premium, so Sep->Oct contract levels jump ~8 % at a
    roll. Every contract shares one factor (plus 0.2 % idiosyncratic noise), so a correctly spliced
    1-day return must equal the factor's within-contract return on every date, roll days
    included."""
    b0 = DataBundle.empty()
    b = DataBundle(b0.observations, b0.forecasts, null_curve, b0.contracts)
    c = null_curve.sort_values(["contract_id", "trade_date"])
    c = c.assign(r=c.groupby("contract_id")["value"].transform(lambda s: np.log(s).diff()))
    factor = c.dropna(subset=["r"]).groupby("trade_date")["r"].median()
    t = d.build_targets(null_curve, 1)
    o = pd.DatetimeIndex(t["origin_time"].unique())
    X = build_feature_matrix(b, o, _cfg({}), families=["price"])
    r = X["price__ret_1d"].dropna()
    day = r.index.tz_convert(MARKET_TZ).normalize().tz_convert("UTC")
    f = factor.reindex(day).to_numpy()
    ok = np.isfinite(f)
    assert ok.mean() > 0.95
    assert np.abs(r.to_numpy()[ok] - f[ok]).max() < 0.012


def test_t3_label_overlap_matches_horizon(null_curve: pd.DataFrame) -> None:
    t = d.build_targets(null_curve, 5)
    assert d.label_overlap_rows(t) == 4
    assert d.inner_gap_violations(t, gap=5) == 0
    assert d.target_lag_violations(t, horizon=5, p=3) == 0


@pytest.mark.parametrize("k", [1, 3])
def test_t5_horizon_field_is_a_month_count_not_an_overlap(null_curve: pd.DataFrame, k: int
                                                          ) -> None:
    """CONFIRMED RISK (horizon semantics): T5 rows carry ``horizon = k`` (months) while their labels
    are known only at the contract's last trade, tens of sessions later. Everything that turns
    ``horizon`` into a row count - model inner-CV gap, ``tail_split`` gap, ``make_target_lags``,
    the HAC lag ``h - 1`` in DM/CW - is then wrong by an order of magnitude if fed ``k``."""
    t = d.build_targets(null_curve, k, target="delivery_excess_fut_log")
    assert int(t["horizon"].iloc[0]) == k
    overlap = d.label_overlap_rows(t)
    assert overlap >= 15 * k
    assert d.inner_gap_violations(t, gap=k) > 0          # a k-row gap leaks labels
    assert d.target_lag_violations(t, horizon=k) > 0     # AR lags not yet known at the origin
    assert d.inner_gap_violations(t, gap=overlap + 1) == 0


class _SpyModel:
    """Records whether the runner hands label_known_time to fit()."""

    seen: list[bool] = []

    def fit(self, X, y, label_known_time=None):  # type: ignore[no-untyped-def]
        _SpyModel.seen.append(label_known_time is not None)
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        return np.zeros(len(X))


def test_walk_forward_passes_label_known_time_to_fit(null_curve: pd.DataFrame) -> None:
    """Was a CONFIRMED GAP during this review (D-011/D-021): models accepted label_known_time
    but walk_forward did not pass it, so T4/T5 inner tuning relied on a row gap = month count.
    Fixed concurrently in evaluation/backtest.py (signature-inspected pass-through); this test
    keeps it fixed for any model whose fit() accepts the keyword."""
    from tests.unit.model_risk.conftest import toy_split

    t = d.build_targets(null_curve, 5)
    o = pd.DatetimeIndex(t["origin_time"])
    X = pd.DataFrame({"price__x": np.arange(len(o), dtype=float)}, index=o)
    from ttf_forecaster.evaluation.backtest import walk_forward
    from ttf_forecaster.evaluation.splits import make_splits

    _SpyModel.seen = []
    folds = make_splits(t, horizon=5, cfg=toy_split(), valid_origins=X.index)
    walk_forward(_SpyModel, X, t, folds, None, model_id="spy", dropna_features=False)
    assert _SpyModel.seen and all(_SpyModel.seen)


def test_walk_forward_training_labels_known_at_refit(null_curve: pd.DataFrame) -> None:
    """Outer purge: for every refit, every training label is known by the refit time, and no
    test origin of the same fold trains its own model (re-checked independently here)."""
    from tests.unit.model_risk.conftest import toy_split

    from ttf_forecaster.evaluation.splits import make_splits

    t = d.build_targets(null_curve, 20)
    lkt = t.set_index("origin_time")["label_known_time"]
    for f in make_splits(t, horizon=20, cfg=toy_split()):
        for s in f.segments:
            assert (lkt.reindex(s.train_origins) <= s.refit_time).all()
            assert not s.train_origins.isin(s.predict_origins).any()


# ---------------------------------------------------------------------------------------------
# 3. Missing-not-at-random and preprocessing fitted inside fit()
# ---------------------------------------------------------------------------------------------


def test_preprocessor_imputes_seasonal_nan_without_indicator() -> None:
    """RISK (MNAR): season-specific NaN (e.g. storage__prob_eos_below_* in winter, D-024) is
    median-imputed inside fit with no missingness indicator; an all-NaN training column is kept
    and imputed to 0 (keep_empty_features). NaN is never 0 in the *features*, but the model sees
    "not applicable" as "typical value". Acceptable for ridge; a reason to prefer native-NaN
    handling (LightGBM with prep off) or explicit indicators for seasonal features."""
    from ttf_forecaster.models.interface import TabularPreprocessor

    idx = pd.date_range("2021-01-01", periods=10, freq="B", tz="UTC")
    X = pd.DataFrame({"storage__seasonal": [np.nan] * 5 + [1.0, 2, 3, 4, 5],
                      "storage__empty": np.nan}, index=idx)
    prep = TabularPreprocessor(standardize=False).fit(X)
    Z = prep.transform(X)
    assert Z.shape[1] == 2 and np.allclose(Z[:5, 0], 3.0) and np.allclose(Z[:, 1], 0.0)


def test_preprocessor_moments_come_from_train_rows_only() -> None:
    from ttf_forecaster.models.interface import TabularPreprocessor

    idx = pd.date_range("2021-01-01", periods=40, freq="B", tz="UTC")
    tr = pd.DataFrame({"price__a": np.arange(20.0)}, index=idx[:20])
    te = pd.DataFrame({"price__a": np.arange(20.0) + 1000}, index=idx[20:])
    p = TabularPreprocessor(family_pca=None).fit(tr)
    assert np.isclose(p.scaler_.mean_[0], 9.5)
    assert p.transform(te).min() > 10  # test rows are NOT re-centred on their own moments


def test_no_full_sample_fit_in_feature_matrix(market_bundle: DataBundle) -> None:
    """Scaler/PCA outside fit probe: appending later origins must not change earlier rows of
    the assembled matrix (any full-sample normalisation would)."""
    days = pd.bdate_range("2022-06-01", "2022-12-30").strftime("%Y-%m-%d").tolist()
    fams = ["price", "curve", "calendar", "interactions"]
    a = build_feature_matrix(market_bundle, _local(days[:80]), _cfg({}), families=fams)
    b = build_feature_matrix(market_bundle, _local(days), _cfg({}), families=fams)
    pd.testing.assert_frame_equal(a, b.loc[a.index], check_freq=False)


# ---------------------------------------------------------------------------------------------
# 4. Feature-shift cross-check on the assembled pipeline (null world)
# ---------------------------------------------------------------------------------------------


def test_price_features_carry_no_skill_in_null_world(null_curve: pd.DataFrame) -> None:
    """If price/curve features leaked the target (e.g. via the held contract's future settles)
    a ridge on them would show large OOS R^2 in a world where returns are unpredictable."""
    from tests.unit.model_risk.conftest import toy_split

    b0 = DataBundle.empty()
    b = DataBundle(b0.observations, b0.forecasts, null_curve, b0.contracts)
    t = d.build_targets(null_curve, 1)
    X = d.build_features(b, pd.DatetimeIndex(t["origin_time"].unique()),
                         ["price", "curve", "calendar"])
    exp = d.Experiment(X, t, 1, toy_split(), "ridge", {"alphas": [0.1, 10.0, 1000.0]})
    s = d.skill_vs_zero(d.run_experiment(exp), "model", 1)
    assert s["oos_r2"] < 0.02
    out = d.feature_shift_test(exp, ["price__ret_1d", "price__ret_5d"], shifts=(0, 1))
    assert out.attrs["suspect_leak"] is False
