"""Walk-forward evaluation, built to be capable of returning a negative answer.

THE BENCHMARK IS ZERO. Under an efficient market a futures price is a
martingale, so the honest competitor for a return forecast is "no forecast at
all". Beating a seasonal mean or a persistence rule proves nothing, because
neither is tradeable. Those two are carried as diagnostics only.

PURGE AND EMBARGO. With an h-day forward return, the last h training
observations overlap the first test observation - they share price days that
sit inside the test window. Training on them leaks the answer. Purging removes
them; the embargo adds a further gap for feature autocorrelation. Omitting this
is, after the coverage bug in `revisions.py`, the easiest way to manufacture a
signal in this project.

EVERY SCORECARD CARRIES effective N AND NET-OF-COST P&L. A result without
those two is not reportable: at a 30-day horizon a thousand daily observations
are about thirty-three independent ones, and a signal that dies at one tick of
cost was never a signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from wxreturns.config import get
from wxreturns.market.returns import transaction_cost
from wxreturns.model import ridge

log = logging.getLogger(__name__)


@dataclass
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    lambda_: float


def _purge_days(horizon: int) -> int:
    rule = get("model", "backtest", "purge_business_days")
    return int(horizon) if rule == "horizon" else int(rule)


def make_folds(dates: pd.Series, horizon: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds with a purge and embargo between train and test."""
    cfg = get("model", "backtest")
    initial = cfg["initial_train_business_days"]
    step = cfg["refit_every_business_days"]
    minimum_test = cfg["min_test_business_days"]
    gap = _purge_days(horizon) + cfg["embargo_business_days"]

    order = np.argsort(dates.to_numpy())
    n = len(order)
    if n < initial + gap + minimum_test:
        raise ValueError(
            f"{n} observations cannot support a walk-forward needing "
            f"{initial} train + {gap} purge/embargo + {minimum_test} test. "
            f"At a {horizon}-day horizon this sample is too short; the "
            "Open-Meteo previous-run archive starts mid-2021, which is the "
            "binding constraint (see model.yaml sample)."
        )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    train_end = initial
    while train_end + gap + step <= n:
        test_start = train_end + gap
        test_end = min(test_start + step, n)
        folds.append((order[:train_end], order[test_start:test_end]))
        train_end += step

    if not folds:
        raise ValueError("no folds could be formed")
    return folds


def select_lambda(features: np.ndarray, target: np.ndarray, dates: pd.Series,
                  horizon: int) -> float:
    """Walk-forward CV over the configured grid. Never an in-sample fit."""
    grid = get("model", "ridge", "lambda_grid")
    folds = make_folds(dates, horizon)

    best_lambda, best_score = None, -np.inf
    for candidate in grid:
        scores: list[float] = []
        for train_index, test_index in folds:
            try:
                fitted = ridge.fit(features[train_index], target[train_index],
                                   [f"x{i}" for i in range(features.shape[1])],
                                   horizon, float(candidate))
            except ValueError:
                continue
            predicted = fitted.predict(features[test_index])
            score = ridge.oos_r2(target[test_index], predicted)
            if np.isfinite(score):
                scores.append(score)
        if scores:
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_lambda, best_score = float(candidate), mean_score

    if best_lambda is None:
        raise ValueError(
            "no ridge penalty produced a finite out-of-sample score on any "
            "fold; the design matrix is probably degenerate")
    log.info("selected lambda=%s (mean OOS R2 %.5f)", best_lambda, best_score)
    return best_lambda


def _benchmark_predictions(name: str, target: np.ndarray, train_index: np.ndarray,
                           test_index: np.ndarray, momentum: np.ndarray | None
                           ) -> np.ndarray:
    if name == "zero":
        return np.zeros(len(test_index))
    if name == "seasonal_mean":
        return np.full(len(test_index), float(np.nanmean(target[train_index])))
    if name == "momentum_5d":
        if momentum is None:
            return np.full(len(test_index), np.nan)
        return momentum[test_index]
    raise ValueError(f"unknown benchmark {name!r}")


def walk_forward(panel: pd.DataFrame, feature_columns: list[str],
                 target_column: str, horizon: int, commodity: str,
                 slot: str, momentum_column: str | None = None) -> dict:
    """Fit and score one (commodity, slot, horizon) cell.

    Returns a scorecard carrying, always, the effective sample size and the
    net-of-cost P&L. Those are the two numbers that decide whether anything
    here is real.
    """
    frame = panel.sort_values("obs_date").reset_index(drop=True)
    features = frame[feature_columns].to_numpy(dtype=float)
    target = frame[target_column].to_numpy(dtype=float)
    momentum = (frame[momentum_column].to_numpy(dtype=float)
                if momentum_column and momentum_column in frame else None)

    lambda_ = select_lambda(features, target, frame["obs_date"], horizon)
    folds = make_folds(frame["obs_date"], horizon)

    predictions = np.full(len(frame), np.nan)
    fold_records: list[Fold] = []
    for index, (train_index, test_index) in enumerate(folds):
        try:
            fitted = ridge.fit(features[train_index], target[train_index],
                               feature_columns, horizon, lambda_)
        except ValueError as exc:
            log.warning("fold %d skipped: %s", index, exc)
            continue
        predictions[test_index] = fitted.predict(features[test_index])
        fold_records.append(Fold(
            index=index,
            train_start=frame["obs_date"].iloc[train_index[0]],
            train_end=frame["obs_date"].iloc[train_index[-1]],
            test_start=frame["obs_date"].iloc[test_index[0]],
            test_end=frame["obs_date"].iloc[test_index[-1]],
            n_train=len(train_index), n_test=len(test_index), lambda_=lambda_))

    if not fold_records:
        raise ValueError("every fold failed to fit")

    tested = np.isfinite(predictions) & np.isfinite(target)
    n_test = int(tested.sum())
    if n_test == 0:
        raise ValueError("no scored observations")

    actual = target[tested]
    predicted = predictions[tested]

    threshold = get("model", "signal", "min_abs_forecast_return")
    cap = get("model", "signal", "max_position_units")
    signal = np.clip(predicted / threshold, -cap, cap)
    signal[np.abs(predicted) < threshold] = 0.0

    cost = transaction_cost(commodity)
    turnover = np.abs(np.diff(signal, prepend=0.0))
    gross = signal * actual
    net = gross - turnover * cost

    annual = get("model", "volatility", "annualisation_days")
    net_std = float(np.std(net, ddof=1)) if len(net) > 1 else 0.0
    sharpe = (float(np.mean(net)) / net_std * float(np.sqrt(annual))
              if net_std > 0.0 else float("nan"))

    benchmarks = {}
    for name in get("model", "backtest", "benchmarks"):
        values = np.concatenate([
            _benchmark_predictions(name, target, train_index, test_index, momentum)
            for train_index, test_index in folds])[:len(predictions)]
        aligned = np.full(len(predictions), np.nan)
        aligned[:len(values)] = values
        benchmarks[name] = ridge.oos_r2(target[tested], aligned[tested])

    hit = ridge.directional_hit_rate(actual, predicted, threshold)
    correlation = (float(np.corrcoef(actual, predicted)[0, 1])
                   if len(actual) > 1 else float("nan"))

    return {
        "commodity": commodity,
        "slot": slot,
        "horizon": horizon,
        "lambda": lambda_,
        "n_folds": len(fold_records),
        "n_scored": n_test,
        # The honest denominator. A thousand daily observations at a 30-day
        # horizon are about thirty-three independent ones.
        "effective_n": int(n_test // max(horizon, 1)),
        "oos_r2_vs_zero": ridge.oos_r2(actual, predicted),
        "benchmark_r2": benchmarks,
        "information_coefficient": correlation,
        "directional_hit_rate": hit["hit_rate"],
        "n_traded": hit["n_traded"],
        "gross_pnl_per_unit": float(np.mean(gross)),
        "net_pnl_per_unit": float(np.mean(net)),
        "cost_per_round_turn": cost,
        "sharpe_annualised": sharpe,
        "config_hash": None,
    }


def hit_rate_interval(hit_rate: float, effective_n: int) -> tuple[float, float]:
    """Normal-approximation interval at the EFFECTIVE sample size.

    Computed at effective N rather than nominal N on purpose. At nominal N a
    52% hit rate over a thousand overlapping observations looks decisive; at
    the thirty-three independent ones it rests on, the interval comfortably
    contains a coin toss.
    """
    if effective_n <= 0 or not np.isfinite(hit_rate):
        return (float("nan"), float("nan"))
    z = float(get("model", "inference", "confidence_z"))
    error = z * float(np.sqrt(hit_rate * (1.0 - hit_rate) / effective_n))
    return (hit_rate - error, hit_rate + error)
