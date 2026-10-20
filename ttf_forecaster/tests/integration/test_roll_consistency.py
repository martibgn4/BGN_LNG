"""The feature-side front month (``features/prices.py`` reference splice) and the target-side
roll (``contracts/rolls.py``) must agree on every session.

modelling_log D-025 makes ``contracts/rolls.py`` canonical, but ``configs/features.yaml`` still has
``return_series_provider: null``, so the reference splice is what runs. Until the provider is
wired, this test guards against the two definitions drifting. SYNTHETIC curve, no holidays: it
does NOT cover holiday handling (features use ``common.holidays``; targets use the empirical
session calendar from ``configs/instruments.yaml``).
"""

from __future__ import annotations

import pandas as pd

from ttf_forecaster.contracts.rolls import front_contract_series
from ttf_forecaster.data import synthetic
from ttf_forecaster.features import prices
from ttf_forecaster.pipelines.build_features import load_feature_config
from ttf_forecaster.targets.common import CurveContext


def test_front_month_agrees_between_features_and_targets() -> None:
    curve = synthetic.ttf_curve(synthetic.SyntheticConfig(start="2019-01-02", end="2021-12-31"))
    ctx = CurveContext.from_curve(curve)
    fc = load_feature_config(None)
    params = {**fc["common"], **fc["families"]["price"]["params"]}
    series = prices.front_return_series(curve, None, params)
    days = pd.DatetimeIndex(series["date"])
    from_rolls = front_contract_series(days, ctx.chain, ctx.rule.roll_days)
    from_prices = pd.Series(
        ["TTF.M." + pd.Timestamp(m).strftime("%Y-%m") for m in series["front_month"]], index=days
    )
    mismatch = from_rolls.to_numpy() != from_prices.to_numpy()
    assert len(days) > 500
    assert not mismatch.any(), pd.DataFrame(
        {"rolls": from_rolls[mismatch], "prices": from_prices[mismatch]}
    ).head(10)


def test_roll_offset_configured_identically() -> None:
    ctx = CurveContext.from_curve(
        synthetic.ttf_curve(synthetic.SyntheticConfig(start="2022-01-03", end="2022-06-30")))
    fc = load_feature_config(None)
    assert int(fc["common"]["roll_offset_bdays"]) == int(ctx.rule.roll_days)
