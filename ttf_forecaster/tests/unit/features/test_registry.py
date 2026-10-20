"""Registry: prefix mapping, selection for ablations, hashing, lazy loading."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.features.registry import (
    FAMILIES,
    PRICE_DERIVED_COLUMNS,
    PRICE_ONLY_FAMILIES,
    FeatureRegistryError,
    family_of,
    feature_hash,
    load_dotted,
    select,
)


def _frame() -> pd.DataFrame:
    cols = ["price__ret_1d", "curve__m1_m2", "storage__dev", "lng__arrivals", "calendar__month"]
    f = pd.DataFrame(np.zeros((3, len(cols))), columns=cols)
    f.attrs["pit_grade"] = {"price__ret_1d": "A", "curve__m1_m2": "A", "storage__dev": "B",
                            "lng__arrivals": "C", "calendar__month": "A"}
    return f


def test_family_of() -> None:
    assert family_of("cross_commodity__jkm_minus_ttf_usd_mmbtu") == "cross_commodity"
    with pytest.raises(FeatureRegistryError):
        family_of("no_prefix")
    assert set(FAMILIES) >= {"price", "curve", "cross_commodity", "storage", "flows", "lng",
                             "weather", "power", "calendar", "interactions", "text"}


def test_select_grades_and_families() -> None:
    f = _frame()
    clean = select(f)
    assert "lng__arrivals" not in clean.columns and "storage__dev" in clean.columns
    assert "lng__arrivals" not in clean.attrs["pit_grade"]
    assert list(select(f, max_grade="A").columns) == ["price__ret_1d", "curve__m1_m2",
                                                      "calendar__month"]
    assert list(select(f, families=PRICE_ONLY_FAMILIES).columns) == [
        "price__ret_1d", "curve__m1_m2", "calendar__month"]
    assert "storage__dev" not in select(f, exclude=["storage"]).columns
    assert "lng__arrivals" in select(f, max_grade="C").columns


def test_feature_hash_stable_and_sensitive() -> None:
    h1 = feature_hash(["b", "a"], {"x": 1, "y": [1, 2]})
    assert h1 == feature_hash(["a", "b"], {"y": [1, 2], "x": 1})
    assert h1 != feature_hash(["a", "b"], {"x": 2, "y": [1, 2]})
    assert h1 != feature_hash(["a", "b", "c"], {"x": 1, "y": [1, 2]})


def test_load_dotted_errors() -> None:
    with pytest.raises(FeatureRegistryError, match="not available"):
        load_dotted("ttf_forecaster.features.does_not_exist:build_x_features")
    with pytest.raises(FeatureRegistryError, match="no attribute"):
        load_dotted("ttf_forecaster.features.prices:nope")
    with pytest.raises(FeatureRegistryError, match="package.module"):
        load_dotted("no_colon")


def test_price_only_includes_price_derived_power_columns() -> None:
    cols = ["price__ret_1d", "calendar__month", "power__clean_spark_eur_mwh",
            "power__gas_minus_switch_eur_mwh", "power__de_residual_load_rev_gw", "storage__dev"]
    f = pd.DataFrame(np.zeros((2, len(cols))), columns=cols)
    f.attrs["pit_grade"] = dict.fromkeys(cols, "A")
    h0 = select(f, price_only=True)
    assert list(h0.columns) == ["price__ret_1d", "calendar__month", "power__clean_spark_eur_mwh",
                                "power__gas_minus_switch_eur_mwh"]
    assert set(PRICE_DERIVED_COLUMNS) >= {"power__clean_spark_eur_mwh",
                                          "power__switch_gas_price_eur_mwh"}
    with pytest.raises(FeatureRegistryError):
        select(f, price_only=True, families=["price"])
