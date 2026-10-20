"""Feature-matrix assembly and the leakage canary. SYNTHETIC data only."""

from __future__ import annotations

import pandas as pd
import pytest

from ttf_forecaster.data.base import DataBundle, PointInTimeViolation
from ttf_forecaster.features.registry import FeatureRegistryError
from ttf_forecaster.pipelines.build_features import (
    build_feature_matrix,
    enabled_families,
    leakage_canary,
)

from .helpers import bundle, config, settle_origins

LEAKY = "tests.unit.features.leaky_builders"


@pytest.fixture(scope="module")
def b() -> DataBundle:
    return bundle()


@pytest.fixture(scope="module")
def origins() -> pd.DatetimeIndex:
    return settle_origins("2022-04-01", "2023-03-31")


def test_matrix_contract(b: DataBundle, origins: pd.DatetimeIndex) -> None:
    shuffled = origins[::-1]
    m = build_feature_matrix(b, shuffled)
    assert m.index.equals(shuffled) and m.index.name == "origin_time"
    assert not m.columns.duplicated().any()
    assert set(m.attrs["pit_grade"]) == set(m.columns)
    assert m.attrs["is_synthetic"] is True
    assert len(m.attrs["feature_hash"]) == 64
    assert set(m.attrs["families"]) == {"price", "curve", "cross_commodity", "calendar",
                                        "interactions"}
    assert m["interactions__expiry_proximity_x_prompt_stress"].notna().any()


def test_ablation_subset(b: DataBundle, origins: pd.DatetimeIndex) -> None:
    m = build_feature_matrix(b, origins, families=["price", "calendar"])
    assert {c.split("__")[0] for c in m.columns} == {"price", "calendar"}


def test_missing_module_fails_clearly(b: DataBundle, origins: pd.DatetimeIndex) -> None:
    cfg = config()
    cfg["families"]["storage"] = {"enabled": True,
                                  "builder": "ttf_forecaster.features.no_such_module:build_x"}
    with pytest.raises(FeatureRegistryError, match="no_such_module"):
        build_feature_matrix(b, origins, cfg)


def test_contract_violation_detected(b: DataBundle, origins: pd.DatetimeIndex) -> None:
    cfg = config()
    cfg["families"]["dupe"] = {"enabled": True, "builder": f"{LEAKY}:build_dupe_features"}
    with pytest.raises(FeatureRegistryError, match="prefix"):
        build_feature_matrix(b, origins, cfg)


def test_grade_c_dropped_unless_allowed(b: DataBundle, origins: pd.DatetimeIndex) -> None:
    cfg = config()
    cfg["families"]["gradec"] = {"enabled": True, "builder": f"{LEAKY}:build_grade_c_features"}
    m = build_feature_matrix(b, origins, cfg, families=["calendar", "gradec"])
    assert "gradec__x" not in m.columns and m.attrs["dropped_grade_c"] == ["gradec__x"]
    m2 = build_feature_matrix(b, origins, cfg, families=["calendar", "gradec"],
                              allow_grade_c=True)
    assert "gradec__x" in m2.columns and m2.attrs["contains_grade_c"] is True


def test_clean_backtest_ineligible_family_dropped(b: DataBundle,
                                                  origins: pd.DatetimeIndex) -> None:
    cfg = config()
    cfg["families"]["notclean"] = {"enabled": True,
                                   "builder": f"{LEAKY}:build_notclean_features"}
    m = build_feature_matrix(b, origins, cfg, families=["calendar", "notclean"])
    assert "notclean__x" not in m.columns and m.attrs["not_clean_eligible"] == ["notclean__x"]
    m2 = build_feature_matrix(b, origins, cfg, families=["calendar", "notclean"],
                              allow_grade_c=True)
    assert "notclean__x" in m2.columns and m2.attrs["contains_grade_c"] is True


def test_default_config_enables_only_delivered_families() -> None:
    assert set(enabled_families(config())) == {"price", "curve", "cross_commodity", "calendar",
                                               "interactions"}


def test_canary_passes_on_real_builders(b: DataBundle, origins: pd.DatetimeIndex) -> None:
    report = leakage_canary(b, origins, n_checks=3)
    report.raise_if_failed()
    assert report.ok and len(report.checked_origins) == 3


@pytest.mark.parametrize("builder", ["build_future_peek_features", "build_fullsample_z_features"])
def test_canary_catches_leaks(b: DataBundle, origins: pd.DatetimeIndex, builder: str) -> None:
    cfg = config()
    fam = builder.removeprefix("build_").removesuffix("_features")
    fam = {"future_peek": "future_peek", "fullsample_z": "fullz"}[fam]
    cfg["families"][fam] = {"enabled": True, "builder": f"{LEAKY}:{builder}"}
    report = leakage_canary(b, origins, cfg, families=[fam], n_checks=2)
    assert not report.ok
    with pytest.raises(PointInTimeViolation, match="leakage canary failed"):
        report.raise_if_failed()
