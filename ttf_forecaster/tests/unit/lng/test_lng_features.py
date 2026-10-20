"""build_lng_features: contract, point-in-time behaviour, grades. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import DataBundle
from ttf_forecaster.features.lng import build_lng_features
from ttf_forecaster.features.registry import check_family_frame


@pytest.fixture(scope="module")
def feats(lng_bundle: DataBundle, origins: pd.DatetimeIndex) -> pd.DataFrame:
    return build_lng_features(lng_bundle, origins, {})


def test_contract(feats: pd.DataFrame, origins: pd.DatetimeIndex) -> None:
    check_family_frame(feats, "lng", origins)
    assert all(c.startswith("lng__") for c in feats.columns)


def test_core_features_populated(feats: pd.DataFrame) -> None:
    for col in ("lng__arrivals_nwe_cargoes_14d", "lng__arrivals_eu_volume_30d",
                "lng__onwater20_volume_t", "lng__onwater20_vessels_chg7d",
                "lng__onwater20_share_usa", "lng__transits_cape_sum30d",
                "lng__transits_cape_share30d", "lng__amer_nwe_share30d",
                "lng__arb_nea_minus_nwe_usd_mmbtu", "lng__diversions_eu_net_14d",
                "lng__destination_undeclared_share", "lng__arrivals_nwe_cargoes_14d_err",
                "lng__slots_m1_available"):
        assert feats[col].notna().mean() > 0.8, col
    shares = feats.filter(like="onwater20_share_").sum(axis=1)
    assert np.allclose(shares.dropna(), 1.0)


def test_grades(feats: pd.DataFrame) -> None:
    g = feats.attrs["pit_grade"]
    assert g["lng__onwater20_volume_t"] == "B"
    assert g["lng__arb_nea_minus_nwe_usd_mmbtu"] == "A"
    assert g["lng__arrivals_nwe_cargoes_14d"] == "A"            # mock snapshots are vintaged
    assert g["lng__arrivals_nwe_cargoes_14d_err"] == "B"        # realised arrivals are B


def test_leakage_canary(lng_bundle: DataBundle, origins: pd.DatetimeIndex) -> None:
    """Perturbing anything published after an origin leaves that origin's row unchanged."""
    sel = origins[[10, 30]]
    cut = sel[0]
    obs, fc = lng_bundle.observations.copy(), lng_bundle.forecasts.copy()
    fo = obs["publication_time"] > cut
    obs.loc[fo, "value"] = obs.loc[fo, "value"] * 7 + 3
    ff = fc["publication_time"] > cut
    fc.loc[ff, "value"] = fc.loc[ff, "value"] + 99
    pert = DataBundle(obs, fc, lng_bundle.curves, lng_bundle.contracts)
    a = build_lng_features(lng_bundle, sel, {}).iloc[0]
    b = build_lng_features(pert, sel, {}).iloc[0]
    pd.testing.assert_series_equal(a, b)


def test_grade_c_excluded_unless_contaminated_run(lng_bundle: DataBundle,
                                                 origins: pd.DatetimeIndex) -> None:
    fc = lng_bundle.forecasts.copy()
    fc["pit_grade"] = "C"
    obs = lng_bundle.observations.copy()
    obs.loc[obs["series_id"].str.startswith("lng.diversions"), "pit_grade"] = "C"
    b = DataBundle(obs, fc, lng_bundle.curves, lng_bundle.contracts)
    sel = origins[:5]
    clean = build_lng_features(b, sel, {})
    assert clean["lng__arrivals_eu_cargoes_7d"].isna().all()
    assert clean["lng__diversions_eu_net_14d"].isna().all()
    assert clean.attrs["pit_grade"]["lng__arrivals_eu_cargoes_7d"] == "C"
    dirty = build_lng_features(b, sel, {"allow_grade_c": True})
    assert dirty["lng__arrivals_eu_cargoes_7d"].notna().all()
    assert dirty.attrs["pit_grade"]["lng__arrivals_eu_cargoes_7d"] == "C"


def test_error_only_after_realisation_published(lng_bundle: DataBundle) -> None:
    """The error at origin t uses a forecast whose whole window was realised AND published."""
    t = pd.Timestamp("2022-05-02 17:00", tz="UTC")
    obs = lng_bundle.observations
    real = obs.loc[obs["series_id"] == "lng.arrivals_realised.nwe.cargoes_count"]
    latest_known_day = real.loc[real["publication_time"] <= t, "event_time"].max()
    fc = lng_bundle.forecasts
    runs = fc.loc[(fc["series_id"] == "lng.arrivals.nwe.cargoes_14d")
                  & (fc["publication_time"] <= t)]
    usable = runs.loc[runs["valid_time"] - pd.Timedelta(days=1) <= latest_known_day]
    run = usable.sort_values("issue_time").iloc[-1]
    days = pd.date_range(run["valid_time"] - pd.Timedelta(days=14), periods=14, freq="D")
    realised = real.set_index("event_time").loc[days, "value"].sum()
    out = build_lng_features(lng_bundle, pd.DatetimeIndex([t]), {})
    assert out["lng__arrivals_nwe_cargoes_14d_err"].iloc[0] == pytest.approx(
        run["value"] - realised)


def test_empty_bundle_gives_nan_not_zero(origins: pd.DatetimeIndex) -> None:
    out = build_lng_features(DataBundle.empty(), origins[:3], {})
    check_family_frame(out, "lng", origins[:3])
    assert out.isna().all().all()
    assert out.attrs["pit_grade"]["lng__arrivals_eu_cargoes_7d"] == "C"
    assert out.attrs["pit_grade"]["lng__onwater20_volume_t"] == "B"


def test_zero_flow_legs_give_nan_share_not_inf(lng_bundle: DataBundle,
                                               origins: pd.DatetimeIndex) -> None:
    obs = lng_bundle.observations.copy()
    obs.loc[obs["series_id"].str.startswith("lng.amer_to_"), "value"] = 0.0
    out = build_lng_features(DataBundle(obs, lng_bundle.forecasts, lng_bundle.curves,
                                        lng_bundle.contracts), origins[:3], {})
    assert out["lng__amer_nwe_share30d"].isna().all()


def test_naive_origins_rejected(lng_bundle: DataBundle) -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        build_lng_features(lng_bundle, pd.DatetimeIndex(["2022-05-02"]), {})
