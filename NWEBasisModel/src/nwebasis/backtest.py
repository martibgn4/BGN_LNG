"""Walk-forward backtest against the benchmarks that actually matter.

THE BENCHMARK CHOICE IS THE WHOLE TEST. Beating a naive forecast of the basis
is easy and worth nothing, because you cannot trade against a naive forecast -
you trade against the market. Three benchmarks are carried:

  ``random_walk``    Today's assessment for delivery month m, as the forecast
                     of the same month's assessment h days from now. For a
                     fixed delivery month this IS the market's own forecast:
                     the monthly contract is a martingale under an efficient
                     market. This is the benchmark to beat.

  ``forward_curve``  Fixing the TENOR instead of the month: today's M+2 as the
                     forecast of M+1 one month out. The difference between this
                     and the random walk is the roll, which is where the
                     seasonal lives.

  ``seasonal_mean``  The training-sample mean for that delivery month. A sanity
                     floor - a model that cannot beat this has learnt nothing.

NO-LOOKAHEAD DISCIPLINE. Every fold refits on data strictly before the test
window opens. The standardiser is fitted on train only. Features have already
been pushed forward by the publication lag in ``features.py``. If any of those
three slip, the result is a backtest that looks tradeable and is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nwebasis.config import get
from nwebasis.model import ecm


@dataclass
class FoldResult:
    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_test: int
    alpha: float
    predictions: pd.DataFrame


def _horizon_message(tenor: str, horizon: int, usable: int, available: int,
                     needed: int | None) -> str:
    """One explanation for both ways a tenor/horizon pair can come up short.

    The common cause is not a data gap. A delivery month occupies the M+1 slot
    for about a month, so from an M+1 observation there is no assessment 21
    business days later: the contract has EXPIRED. Saying so is the difference
    between a two-minute fix and an afternoon spent hunting for missing rows.
    """
    need = "" if needed is None else f", but the walk-forward needs at least {needed}"
    return (
        f"{tenor} at a {horizon}-business-day horizon: {usable} usable "
        f"observations ({available} had a realised target at all){need}."
        "\n\nIf that count is very small the horizon outruns the contract - the "
        "delivery month has expired before the forecast date, so no assessment "
        "exists to score against. Forecast from a tenor with at least the "
        "horizon left to run: M+2 for a one-month view, M+3 for two. Otherwise "
        "shorten the training window."
    )


def _rmse(errors: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(errors.dropna()))))


def _mae(errors: pd.Series) -> float:
    return float(np.mean(np.abs(errors.dropna())))


def build_targets(panel: pd.DataFrame, horizon: int, target: str = "value"
                  ) -> pd.DataFrame:
    """Attach the realised future basis and the market benchmarks.

    ``realised`` is the same DELIVERY MONTH's assessment ``horizon`` business
    days later. Keeping the delivery month fixed is what makes the comparison
    fair: a tenor-fixed target silently swaps the underlying gas every roll, so
    part of the "forecast error" would just be the seasonal step between two
    different months.

    MUST BE CALLED ON THE FULL PANEL, before filtering to a tenor. A given
    delivery month occupies the M+1 slot for only about one month, so grouping
    a tenor-filtered frame by delivery month leaves each group about 20 rows
    long and a 21-day shift annihilates the sample. Tracked across the whole
    curve, one delivery month is a continuous daily series that walks M+6 to
    M+1 as it approaches, which is the series a forecast actually lives on.

    Two benchmarks are attached:

      ``random_walk``       today's assessment for that delivery month. For a
                            fixed delivery month this IS the tradeable forward,
                            because the monthly contract is a martingale under
                            an efficient market. The two are the same number
                            here, and only one of them is carried so the
                            scorecard does not report it twice under two names.
      ``next_month_curve``  today's assessment for the ADJACENT delivery month.
                            Not a competing forecast so much as a test of
                            whether curve shape carries information the level
                            does not.
    """
    out = panel.sort_values(["delivery_start", "release_date"]).copy()
    grouped = out.groupby("delivery_start", sort=False)
    out["realised"] = grouped[target].shift(-horizon)
    out["benchmark_random_walk"] = out[target]

    adjacent = out.sort_values(["release_date", "delivery_start"])
    adjacent["benchmark_next_month_curve"] = (
        adjacent.groupby("release_date", sort=False)[target].shift(-1))
    out = out.merge(
        adjacent[["release_date", "delivery_start", "benchmark_next_month_curve"]],
        on=["release_date", "delivery_start"], how="left")
    return out.dropna(subset=["realised"])


def run(panel: pd.DataFrame, anchor_names: list[str], short_run_names: list[str],
        horizon: int, tenor: str, target: str = "value") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk forward over one tenor. Returns per-fold predictions and a scorecard."""
    # Targets first, on the whole curve; the tenor filter selects only WHICH
    # observations we forecast from. Reversing these two lines is the bug that
    # leaves a 21-day horizon with about 19 usable rows.
    with_targets = build_targets(panel, horizon, target)
    frame = (with_targets[with_targets["tenor"] == tenor]
             .sort_values("release_date").reset_index(drop=True))
    if frame.empty:
        available = int(with_targets["tenor"].value_counts().get(tenor, 0))
        raise ValueError(_horizon_message(tenor, horizon, 0, available, None))

    initial = int(get("model", "backtest", "initial_train_business_days"))
    step = int(get("model", "backtest", "refit_every_business_days"))
    minimum_test = int(get("model", "backtest", "min_test_business_days"))

    if len(frame) < initial + minimum_test:
        available = int(with_targets["tenor"].value_counts().get(tenor, 0))
        raise ValueError(_horizon_message(tenor, horizon, len(frame), available,
                                          initial + minimum_test))

    folds: list[FoldResult] = []
    predictions: list[pd.DataFrame] = []
    for fold, start in enumerate(range(initial, len(frame) - 1, step)):
        train = frame.iloc[:start]
        test = frame.iloc[start:start + step]
        if test.empty:
            break
        try:
            model = ecm.fit(train, anchor_names, short_run_names, target)
        except (ecm.SpecificationError, np.linalg.LinAlgError) as exc:
            # A fold that will not fit is recorded, not skipped silently.
            predictions.append(test.assign(prediction=np.nan, fit_error=str(exc)))
            continue
        forecast = ecm.predict(model, test, horizon, target)
        block = test.loc[forecast.index].assign(prediction=forecast, fit_error=None,
                                                fold=fold, alpha=model.alpha)
        predictions.append(block)
        folds.append(FoldResult(fold=fold, train_end=train["release_date"].iloc[-1],
                                test_start=test["release_date"].iloc[0],
                                test_end=test["release_date"].iloc[-1],
                                n_test=len(test), alpha=model.alpha,
                                predictions=block))

    if not predictions:
        raise ValueError(f"{tenor}: no fold produced a prediction")

    combined = pd.concat(predictions, ignore_index=True)
    return combined, scorecard(combined, tenor, horizon)


def scorecard(predictions: pd.DataFrame, tenor: str, horizon: int) -> pd.DataFrame:
    """RMSE, MAE and skill against each benchmark, plus the directional test."""
    scored = predictions.dropna(subset=["prediction", "realised"])
    if scored.empty:
        raise ValueError("no rows with both a prediction and a realised value")

    rows = []
    model_errors = scored["prediction"] - scored["realised"]
    model_rmse = _rmse(model_errors)
    rows.append({"tenor": tenor, "horizon_bd": horizon, "estimator": "model",
                 "n": len(scored), "rmse": model_rmse, "mae": _mae(model_errors),
                 "skill_vs_benchmark": np.nan})

    for name in get("model", "backtest", "benchmarks"):
        column = f"benchmark_{name}"
        if column not in scored.columns:
            if name == "seasonal_mean":
                seasonal = scored.groupby("delivery_month")["realised"].transform("mean")
                errors = seasonal - scored["realised"]
            else:
                continue
        else:
            errors = scored[column] - scored["realised"]
        benchmark_rmse = _rmse(errors)
        # Skill score: positive means the model beat the benchmark, and the
        # units are the fraction of the benchmark's error removed.
        skill = 1 - model_rmse / benchmark_rmse if benchmark_rmse else np.nan
        rows.append({"tenor": tenor, "horizon_bd": horizon, "estimator": name,
                     "n": int(errors.notna().sum()), "rmse": benchmark_rmse,
                     "mae": _mae(errors), "skill_vs_benchmark": skill})

    table = pd.DataFrame(rows)
    if get("model", "backtest", "report_directional_hit_rate"):
        table.attrs["directional"] = directional(scored)
    return table


def directional(scored: pd.DataFrame) -> dict:
    """Does the model call the right side of the market, and does it pay?

    RMSE rewards being close. A desk gets paid for being on the right side. The
    two disagree often enough that reporting only the first is misleading, so
    the hit rate is on the sign of (realised - market), which is the sign of
    the P&L on a position taken against the market at the model's signal.
    """
    threshold = float(get("model", "backtest", "signal_threshold_usd_per_mmbtu"))
    market = scored["benchmark_random_walk"]
    signal = scored["prediction"] - market
    realised_move = scored["realised"] - market
    traded = signal.abs() >= threshold
    if not traded.any():
        return {"n_signals": 0, "note": f"no signal exceeded {threshold} USD/MMBtu"}
    hits = np.sign(signal[traded]) == np.sign(realised_move[traded])
    pnl = np.sign(signal[traded]) * realised_move[traded]
    return {
        "n_signals": int(traded.sum()),
        "hit_rate": float(hits.mean()),
        "mean_pnl_usd_per_mmbtu": float(pnl.mean()),
        "total_pnl_usd_per_mmbtu": float(pnl.sum()),
        "pnl_sharpe_per_signal": float(pnl.mean() / pnl.std(ddof=1)) if pnl.std(ddof=1) else np.nan,
    }
