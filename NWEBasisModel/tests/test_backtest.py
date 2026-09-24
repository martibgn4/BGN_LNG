"""Backtest plumbing tests. These guard the two ways this backtest could lie.

Both failures below produce a scorecard that looks better than reality, which
is exactly the class of bug that is never noticed in time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nwebasis import backtest as bt

SEED = 20260904


def synthetic_panel(n_months: int = 30, days_per_month: int = 21) -> pd.DataFrame:
    """A panel shaped like the real one: each delivery month walks M+6 to M+1."""
    rng = np.random.default_rng(SEED)
    rows = []
    start = pd.Timestamp("2024-01-01")
    for month in range(n_months):
        delivery = (start + pd.DateOffset(months=month + 6)).normalize().replace(day=1)
        level = rng.standard_normal() / 10
        for tenor_index in range(6, 0, -1):
            for day in range(days_per_month):
                offset = (month * days_per_month
                          + (6 - tenor_index) * days_per_month + day)
                rows.append({
                    "release_date": start + pd.tseries.offsets.BDay(offset),
                    "tenor": f"M+{tenor_index}",
                    "delivery_start": delivery,
                    "delivery_month": delivery.month,
                    "value": level + rng.standard_normal() / 100,
                })
    return pd.DataFrame(rows)


def test_targets_are_built_across_the_whole_curve() -> None:
    """Filtering to a tenor BEFORE building targets destroys the sample.

    A delivery month sits at M+1 for about a month, so grouping a tenor-filtered
    frame by delivery month leaves ~21-row groups and a 21-day shift empties
    them. This test pins the ordering that avoids it.
    """
    panel = synthetic_panel()
    correct = bt.build_targets(panel, horizon=21)
    m2_rows = (correct["tenor"] == "M+2").sum()

    wrong = bt.build_targets(panel[panel["tenor"] == "M+2"], horizon=21)
    assert m2_rows > 10 * max(len(wrong), 1)


def test_horizon_beyond_contract_life_is_explained_not_silently_empty() -> None:
    """M+1 at a 21-day horizon has no target - the month has expired."""
    panel = synthetic_panel()
    with pytest.raises(ValueError, match="expired"):
        bt.run(panel, ["delivery_month"], [], horizon=21, tenor="M+1")


def test_random_walk_benchmark_is_todays_own_quote() -> None:
    panel = synthetic_panel()
    framed = bt.build_targets(panel, horizon=5)
    assert (framed["benchmark_random_walk"] == framed["value"]).all()


def test_realised_is_the_same_delivery_month_later() -> None:
    """The target must not silently roll to a different month's gas."""
    panel = synthetic_panel()
    horizon = 5
    framed = bt.build_targets(panel, horizon=horizon)
    sample = framed.iloc[0]
    same_month = panel[panel["delivery_start"] == sample["delivery_start"]]
    same_month = same_month.sort_values("release_date").reset_index(drop=True)
    position = same_month.index[
        same_month["release_date"] == sample["release_date"]][0]
    assert sample["realised"] == pytest.approx(
        same_month.loc[position + horizon, "value"])


def test_skill_score_is_signed_the_right_way() -> None:
    """Positive skill must mean the model beat the benchmark, not the reverse."""
    predictions = pd.DataFrame({
        "prediction": [1.0, 1.0, 1.0],
        "realised": [1.0, 1.0, 1.0],
        "benchmark_random_walk": [0.0, 0.0, 0.0],
        "benchmark_next_month_curve": [0.0, 0.0, 0.0],
        "delivery_month": [1, 1, 1],
    })
    table = bt.scorecard(predictions, "M+2", 21)
    walk = table[table["estimator"] == "random_walk"].iloc[0]
    assert walk["skill_vs_benchmark"] == pytest.approx(1.0)
