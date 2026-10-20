"""Interactions: train-only fit, PIT-safe expanding mode, grade propagation. SYNTHETIC."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ttf_forecaster.features.interactions import (
    INTERACTIONS,
    InteractionTransformer,
    build_interactions_features,
    expiry_proximity,
)
from ttf_forecaster.features.registry import FeatureRegistryError

SPECS = {
    "cold_x_low_storage": {"inputs": {"cold": "weather__hdd_anom",
                                      "storage": "storage__dev"}},
    "expiry_proximity_x_prompt_stress": {
        "inputs": {"bdays_to_expiry": "calendar__bdays", "prompt_spread": "curve__m1_m2"},
        "window_bdays": 10},
}


def _matrix(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-03 18:00", periods=n, freq="B", tz="UTC", name="origin_time")
    m = pd.DataFrame({
        "weather__hdd_anom": rng.normal(2, 3, n),
        "storage__dev": rng.normal(-5, 10, n),
        "calendar__bdays": rng.integers(1, 25, n).astype(float),
        "curve__m1_m2": rng.normal(0.5, 1.0, n),
    }, index=idx)
    m.attrs["pit_grade"] = {"weather__hdd_anom": "A", "storage__dev": "B", "calendar__bdays": "A",
                            "curve__m1_m2": "A"}
    return m


def test_transformer_fits_on_train_only() -> None:
    m = _matrix()
    train, test = m.iloc[:200], m.iloc[200:]
    t = InteractionTransformer(SPECS).fit(train)
    assert t.mean_["storage__dev"] == pytest.approx(train["storage__dev"].mean())
    out1 = t.transform(test)
    shocked = test.copy()
    shocked.iloc[-1] = shocked.iloc[-1] * 100  # altering one test row changes only that row
    out2 = t.transform(shocked)
    pd.testing.assert_frame_equal(out1.iloc[:-1], out2.iloc[:-1])
    z_c = (test["weather__hdd_anom"] - train["weather__hdd_anom"].mean()) / train[
        "weather__hdd_anom"].std()
    z_s = (test["storage__dev"] - train["storage__dev"].mean()) / train["storage__dev"].std()
    np.testing.assert_allclose(out1["interactions__cold_x_low_storage"],
                               z_c.clip(lower=0) * (-z_s).clip(lower=0))
    assert out1.attrs["pit_grade"] == {"interactions__cold_x_low_storage": "B",
                                       "interactions__expiry_proximity_x_prompt_stress": "A"}


def test_expanding_mode_is_point_in_time() -> None:
    m = _matrix()
    p = {"specs": SPECS, "min_periods": 30, "standardise": "expanding"}
    full = build_interactions_features(m, m.index, p)
    later = m.copy()
    later.iloc[150:] = later.iloc[150:] * -7.0
    later.attrs = m.attrs
    alt = build_interactions_features(later, m.index, p)
    pd.testing.assert_frame_equal(full.iloc[:150], alt.iloc[:150])
    assert full.iloc[:29].isna().all().all() or full.iloc[:29, 0].isna().all()


def test_expiry_proximity_shape() -> None:
    s = pd.Series([0.0, 5.0, 10.0, 20.0])
    assert expiry_proximity(s, 10).tolist() == [1.0, 0.5, 0.0, 0.0]


def test_missing_input_and_unknown_name() -> None:
    m = _matrix().drop(columns=["storage__dev"])
    with pytest.raises(FeatureRegistryError, match="storage__dev"):
        build_interactions_features(m, m.index, {"specs": SPECS})
    with pytest.raises(FeatureRegistryError, match="unknown interaction"):
        build_interactions_features(m, m.index, {"specs": {"made_up": {"inputs": {}}}})
    with pytest.raises(FeatureRegistryError, match="roles"):
        build_interactions_features(_matrix(), m.index, {"specs": {
            "cold_x_low_storage": {"inputs": {"cold": "weather__hdd_anom"}}}})


def test_constant_training_input_rejected() -> None:
    m = _matrix()
    m["storage__dev"] = 1.0
    with pytest.raises(ValueError, match="standardise"):
        InteractionTransformer(SPECS).fit(m)


# one-sided legs: (role -> value in the "tight" direction, value in the "loose" direction)
_TIGHT_LOOSE = {
    "cold": (1.0, -1.0), "storage": (-1.0, 1.0), "outage": (1.0, -1.0),
    "residual_load": (1.0, -1.0), "gas_minus_switch": (-1.0, 1.0),
    "jkm_premium": (1.0, -1.0), "arrivals_change": (-1.0, 1.0),
    "slots_available": (-1.0, 1.0), "arrivals": (1.0, -1.0),
    "bdays_to_expiry": (0.0, 30.0), "prompt_spread": (1.0, -1.0),
}


@pytest.mark.parametrize("name", sorted(INTERACTIONS))
def test_both_loose_legs_give_zero_not_bullish(name: str) -> None:
    """Sign fix: e.g. warm AND full storage (-1 x -1 as z-scores) must give 0, not +1."""
    d = INTERACTIONS[name]
    tight = {r: pd.Series([_TIGHT_LOOSE[r][0]]) for r in d.roles}
    loose = {r: pd.Series([_TIGHT_LOOSE[r][1]]) for r in d.roles}
    assert d.func(**tight).iloc[0] > 0
    assert d.func(**loose).iloc[0] == 0.0
    for r in d.roles:  # one leg loose -> 0
        mixed = {**tight, r: loose[r]}
        assert d.func(**mixed).iloc[0] == 0.0


def test_warm_and_full_storage_is_zero_in_matrix() -> None:
    m = _matrix()
    m["weather__hdd_anom"] = -2.0 + 0.01 * np.arange(len(m)) % 1  # all warm-ish but varying
    m.iloc[100:, m.columns.get_loc("weather__hdd_anom")] = -50.0  # very warm
    m.iloc[100:, m.columns.get_loc("storage__dev")] = 80.0        # very full
    out = build_interactions_features(m, m.index, {"specs": SPECS, "min_periods": 30})
    assert (out["interactions__cold_x_low_storage"].iloc[100:] == 0.0).all()


def test_active_months_mask_and_skip_policy() -> None:
    m = _matrix()
    spec = {"cold_x_low_storage": {**SPECS["cold_x_low_storage"],
                                   "active_months": [10, 11, 12, 1, 2, 3]}}
    out = build_interactions_features(m, m.index, {"specs": spec, "min_periods": 30})
    summer = out.index.tz_convert("Europe/Amsterdam").month.isin([5, 6, 7, 8])
    col = out["interactions__cold_x_low_storage"]
    assert (col[summer].dropna() == 0.0).all() and (col[~summer].dropna() > 0).any()
    missing = {"cold_x_low_storage": {"inputs": {"cold": "weather__nope", "storage": "storage__dev"}},
               **{k: v for k, v in SPECS.items() if k != "cold_x_low_storage"}}
    out2 = build_interactions_features(m, m.index, {"specs": missing, "on_missing_inputs": "skip"})
    assert list(out2.columns) == ["interactions__expiry_proximity_x_prompt_stress"]
