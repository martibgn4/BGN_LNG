"""Storage and flows feature builders: contract, PIT canary, NaN rules. SYNTHETIC data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.data.base import SYNTHETIC_SOURCE, DataBundle, PointInTimeViolation, QualityFlag
from ttf_forecaster.data.fundamentals import observation_rows
from ttf_forecaster.features.flows import build_flows_features
from ttf_forecaster.features.storage import build_storage_features

from .conftest import local_origins

ORIGINS = local_origins("2022-01-12", "2022-06-15", "2023-02-01", "2023-08-20")


def _perturb_future(bundle: DataBundle, origin: pd.Timestamp) -> DataBundle:
    obs = bundle.observations.copy()
    late = obs["publication_time"] > origin.tz_convert("UTC")
    obs.loc[late, "value"] = obs.loc[late, "value"] * 3.0 + 1e4
    out = DataBundle.empty()
    out.observations = obs
    return out


@pytest.mark.parametrize("builder,family", [(build_storage_features, "storage"),
                                            (build_flows_features, "flows")])
class TestContract:
    def test_shape_names_grades(self, bundle, builder, family) -> None:
        f = builder(bundle, ORIGINS, {})
        assert list(f.index) == list(ORIGINS.tz_convert("UTC")) and f.index.name == "origin_time"
        assert all(c.startswith(f"{family}__") for c in f.columns)
        assert set(f.attrs["pit_grade"]) == set(f.columns)
        assert set(f.attrs["pit_grade"].values()) <= {"A", "B"}
        assert f.notna().any().any()

    def test_leakage_canary(self, bundle, builder, family) -> None:
        origin = ORIGINS[2]
        a = builder(bundle, ORIGINS[2:3], {})
        b = builder(_perturb_future(bundle, origin), ORIGINS[2:3], {})
        pd.testing.assert_frame_equal(a, b)

    def test_naive_origin_rejected(self, bundle, builder, family) -> None:
        with pytest.raises(PointInTimeViolation):
            builder(bundle, pd.DatetimeIndex(["2023-01-01 19:00"]), {})

    def test_stale_data_gives_nan(self, bundle, builder, family) -> None:
        f = builder(bundle, local_origins("2024-09-01"), {})
        level_col = "storage__level_twh" if family == "storage" else "flows__no_flow_gwhd"
        assert np.isnan(f[level_col].iloc[0])


class TestStorage:
    def test_values_consistent(self, bundle) -> None:
        f = build_storage_features(bundle, ORIGINS, {})
        assert np.allclose(f["storage__full_pct"], f["storage__level_twh"] / 1100.0 * 100.0)
        dev = f["storage__full_pct"] - f["storage__normal_full_pct"]
        assert np.allclose(f["storage__deviation_pct_pts"], dev, equal_nan=True)
        # winter origin: winter thresholds only; summer origin: summer thresholds only
        assert np.isnan(f["storage__prob_eos_below_80pct"].iloc[2])
        assert np.isnan(f["storage__prob_eos_below_20pct"].iloc[3])
        assert np.isfinite(f["storage__prob_eos_below_90pct"].iloc[3])
        # policy target only inside the injection season of an active regime
        assert np.isnan(f["storage__policy_gap_pct_pts"].iloc[2])
        assert np.isfinite(f["storage__policy_required_injection_gwhd"].iloc[3])

    def test_short_history_nan_not_zero(self, bundle) -> None:
        # origin in the 2nd gas year of the world: < 3 prior gas years -> normals / projection NaN
        f = build_storage_features(bundle, local_origins("2019-01-15"), {})
        assert np.isfinite(f["storage__level_twh"].iloc[0])
        assert np.isnan(f["storage__normal_full_pct"].iloc[0])
        assert np.isnan(f["storage__proj_eos_normal_pct"].iloc[0])

    def test_missing_demand_gives_nan_cover(self, bundle) -> None:
        obs = bundle.observations
        b = DataBundle.empty()
        b.observations = obs.loc[obs["series_id"] != "demand.eu.total_gwhd"]
        f = build_storage_features(b, ORIGINS[:1], {"projection_enabled": False})
        assert np.isnan(f["storage__days_of_cover"].iloc[0])
        assert np.isfinite(f["storage__level_twh"].iloc[0])

    def test_deterministic(self, bundle) -> None:
        pd.testing.assert_frame_equal(build_storage_features(bundle, ORIGINS[2:3], {}),
                                      build_storage_features(bundle, ORIGINS[2:3], {}))


def _outage_bundle(bundle: DataBundle, registry: dict) -> DataBundle:
    """Add a planned outage (1,000 GWh/d on 2023-03-10..12) announced 2023-03-01 06:00 local."""
    spec = registry["flows.no.outage_unavailable_gwhd"]
    days = pd.date_range("2023-03-10", "2023-03-12", freq="D")
    pub = pd.DatetimeIndex([pd.Timestamp("2023-03-01 06:00", tz="Europe/Amsterdam")] * 3).tz_convert("UTC")
    extra = observation_rows(spec, days, np.full(3, 1000.0), source=SYNTHETIC_SOURCE,
                             quality_flag=QualityFlag.SYNTHETIC.value, entitlement="synthetic",
                             revision=5, publication_time=pub)
    out = DataBundle.empty()
    out.observations = pd.concat([bundle.observations, extra], ignore_index=True)
    return out


class TestFlows:
    def test_planned_outage_visible_only_after_announcement(self, bundle, registry) -> None:
        b = _outage_bundle(bundle, registry)
        params = {"outage_forward_days": 14}
        origins = local_origins("2023-02-28", "2023-03-02")  # both windows cover 10-12 March
        col = "flows__no_unavailable_next14d_gwhd"
        with_msg = build_flows_features(b, origins, params)[col]
        without = build_flows_features(bundle, origins, params)[col]
        assert with_msg.iloc[0] == without.iloc[0]  # announced after the first origin
        unav = bundle.observations.query("series_id == 'flows.no.outage_unavailable_gwhd'")
        res = unav.sort_values("revision").drop_duplicates("event_time", keep="last")
        day = res["event_time"].dt.tz_convert("Europe/Amsterdam").dt.strftime("%Y-%m-%d")
        base = res.loc[day.isin(["2023-03-10", "2023-03-11", "2023-03-12"]), "value"].sum()
        assert with_msg.iloc[1] - without.iloc[1] == pytest.approx((3000.0 - base) / 14)

    def test_missing_supply_component_nan(self, bundle) -> None:
        obs = bundle.observations
        b = DataBundle.empty()
        b.observations = obs.loc[obs["series_id"] != "lng.eu.sendout_gwhd"]
        f = build_flows_features(b, ORIGINS, {})
        assert f["flows__supply_surprise_gwhd"].isna().all()
        assert f["flows__lng_sendout_gwhd"].isna().all()
        assert f["flows__no_flow_gwhd"].notna().all()

    def test_utilisation_bounds(self, bundle) -> None:
        f = build_flows_features(bundle, ORIGINS, {})
        u = f["flows__no_utilisation_pct"]
        assert ((u > 0) & (u <= 100.0 + 1e-9)).all()
        assert (f["flows__no_outage_adj_capacity_gwhd"] <= 3800.0 + 1e-9).all()
