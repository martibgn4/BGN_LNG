"""Walk-forward evaluation.

The protocol, and why each piece is there:

  expanding window   at this signal-to-noise a one-year rolling window
                     estimates weights that are mostly noise.

  purge              the label for origin t is not known until t+h, so a
                     training row within h days of the test block has seen
                     part of the test period. Those rows are dropped. Without
                     this, every horizon past h=1 reports inflated skill.

  embargo            extra days dropped on top, so a slow-moving feature
                     cannot carry test-period information back into training.

  refit every 21bd   monthly. Refitting daily is cost without benefit and
                     makes the run an hour long for no change in the result.

Everything is reported per horizon, including n, because the horizons do NOT
share a sample: the 2023 weekly-only stretch removes most 1-day targets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from lngfreight.config import get
from lngfreight.features import TARGET_PREFIX, family_of
from lngfreight.model import (FfaForward, RandomWalk, build_estimators,
                             own_lags_model)

log = logging.getLogger(__name__)

# Which estimator's standardised coefficients feed the family-importance
# table. Ridge because it is the full-feature linear fit: its weights are
# comparable across columns once standardised, which tree gains are not.
COEF_SOURCE = "ridge"


@dataclass
class HorizonResult:
    horizon: int
    predictions: pd.DataFrame          # index=origin, columns=model name, plus 'actual'
    metrics: pd.DataFrame              # one row per model
    coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _hit_rate(actual: np.ndarray, pred: np.ndarray) -> float:
    """Share of origins where the predicted sign matched the realised sign.

    Origins where either is exactly zero are excluded rather than counted as
    a hit; a freight assessment that did not move is not a directional call.
    """
    mask = (np.sign(actual) != 0) & (np.sign(pred) != 0)
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(actual[mask]) == np.sign(pred[mask])).mean())


def _information_coefficient(actual: np.ndarray, pred: np.ndarray) -> float:
    if len(actual) < 3 or np.std(pred) == 0:
        return float("nan")
    rho, _ = stats.spearmanr(actual, pred)
    return float(rho)


def _newey_west_se(x: np.ndarray, lags: int) -> float:
    """HAC standard error of a mean, Bartlett kernel."""
    n = len(x)
    if n < 2:
        return float("nan")
    centred = x - x.mean()
    gamma0 = float(centred @ centred / n)
    total = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cov = float(centred[lag:] @ centred[:-lag] / n)
        total += 2.0 * weight * cov
    if total <= 0:
        return float("nan")
    return float(np.sqrt(total / n))


def diebold_mariano(actual: np.ndarray, pred_a: np.ndarray,
                    pred_b: np.ndarray, horizon: int) -> tuple[float, float]:
    """Test that model A's squared error differs from benchmark B's.

    Negative statistic = A has the smaller error. HAC-corrected with h-1
    lags: with overlapping h-day targets the naive t-stat on a difference of
    squared errors is badly oversized, and at h=15 that is the difference
    between "significant" and nothing at all.
    """
    loss_diff = (actual - pred_a) ** 2 - (actual - pred_b) ** 2
    lags = max(int(horizon) - 1, 0)
    se = _newey_west_se(loss_diff, lags)
    if not np.isfinite(se) or se == 0:
        return float("nan"), float("nan")
    stat = float(loss_diff.mean() / se)
    pvalue = float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
    return stat, pvalue


def _score(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = actual - pred
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "hit_rate": _hit_rate(actual, pred),
        "ic": _information_coefficient(actual, pred),
    }


# --------------------------------------------------------------------------
# the walk
# --------------------------------------------------------------------------

def _blocks(origins: pd.DatetimeIndex, min_train: int, step: int):
    """Yield (train_end_position, test_slice) over the origin index."""
    n = len(origins)
    start = min_train
    while start < n:
        stop = min(start + step, n)
        yield start, slice(start, stop)
        start = stop


def run_horizon(X: pd.DataFrame, y: pd.Series, forward_signal: pd.Series,
                horizon: int) -> HorizonResult:
    wf = get("model", "walk_forward")
    min_train = int(wf["min_train_bd"])
    step = int(wf["refit_every_bd"])
    embargo = int(wf["embargo_bd"])
    purge_cfg = wf["purge_bd"]
    purge = int(horizon) if purge_cfg == "horizon" else int(purge_cfg)
    gap = purge + embargo

    # Every model is scored on an IDENTICAL set of origins, including the
    # benchmarks. A benchmark evaluated on a slightly different sample than
    # the model is not a comparison. The FFA signal has a handful of holes
    # where the freight print is older than the forward-fill limit, and those
    # origins are dropped for everyone rather than for the forward alone.
    usable = (y.dropna().index
              .intersection(X.dropna(how="all").index)
              .intersection(forward_signal.replace([np.inf, -np.inf], np.nan)
                            .dropna().index))
    origins = pd.DatetimeIndex(sorted(usable))
    if len(origins) <= min_train:
        raise ValueError(
            f"h={horizon}: only {len(origins)} usable origins against a "
            f"{min_train}-day minimum training window. Nothing is fitted; "
            "shorten min_train_bd or lengthen the sample rather than "
            "reporting a result from a handful of refits."
        )

    Xo, yo = X.loc[origins], y.loc[origins]
    signal = forward_signal.reindex(origins)

    estimators = build_estimators()
    own_lags = own_lags_model()
    if own_lags is not None:
        estimators = estimators + [own_lags]
    names = [m.name for m in estimators] + [RandomWalk.name, FfaForward.name]
    preds = {name: pd.Series(index=origins, dtype=float) for name in names}
    coef_rows: list[pd.Series] = []
    n_refits = 0

    for train_end, test in _blocks(origins, min_train, step):
        test_origins = origins[test]
        # Purge: a training origin whose label window reaches into the test
        # block has already seen part of it.
        cutoff = train_end - gap
        if cutoff <= 0:
            continue
        train_idx = origins[:cutoff]
        X_train, y_train = Xo.loc[train_idx], yo.loc[train_idx]
        if len(train_idx) < min_train // 2 or y_train.std() == 0:
            continue
        X_test = Xo.loc[test_origins]
        n_refits += 1

        for model in estimators:
            fitted = model.fit(X_train, y_train)
            preds[model.name].loc[test_origins] = fitted.predict(X_test)
            # Coefficients are collected from the full-feature ridge only.
            # Stacking huber and own_lags into the same frame would mix three
            # different column sets under one index and make the family
            # importance table meaningless.
            if model.name == COEF_SOURCE:
                coefs = fitted.coefficients(list(Xo.columns))
                if coefs is not None:
                    coef_rows.append(coefs.rename(test_origins[0]))

        preds[RandomWalk.name].loc[test_origins] = RandomWalk().fit(
            X_train, y_train).predict(X_test)

        fwd = FfaForward().fit(signal.loc[train_idx], y_train)
        preds[FfaForward.name].loc[test_origins] = fwd.predict(signal.loc[test_origins])

    frame = pd.DataFrame(preds)
    frame["actual"] = yo
    frame = frame.dropna(subset=["actual"]).dropna(how="all",
                                                   subset=list(preds.keys()))
    if frame.empty:
        raise ValueError(f"h={horizon}: the walk produced no predictions")

    actual = frame["actual"].to_numpy(dtype=float)
    rows = []
    rw_pred = frame[RandomWalk.name].to_numpy(dtype=float)
    ffa_pred = frame[FfaForward.name].to_numpy(dtype=float)
    rw_rmse = float(np.sqrt(np.mean((actual - rw_pred) ** 2)))
    ffa_rmse = float(np.sqrt(np.mean((actual - ffa_pred) ** 2)))
    has_own = "own_lags" in frame.columns
    own_pred = frame["own_lags"].to_numpy(dtype=float) if has_own else None
    own_rmse = (float(np.sqrt(np.mean((actual - own_pred) ** 2)))
                if has_own else float("nan"))

    for name in names:
        pred = frame[name].to_numpy(dtype=float)
        row = {"model": name, "n": int(len(frame)), "n_refits": n_refits}
        row.update(_score(actual, pred))
        row["skill_vs_rw"] = 1.0 - row["rmse"] / rw_rmse if rw_rmse else float("nan")
        row["skill_vs_ffa"] = 1.0 - row["rmse"] / ffa_rmse if ffa_rmse else float("nan")
        row["skill_vs_own_lags"] = (1.0 - row["rmse"] / own_rmse
                                    if own_rmse and np.isfinite(own_rmse)
                                    else float("nan"))
        dm_stat, dm_p = diebold_mariano(actual, pred, rw_pred, horizon)
        row["dm_vs_rw"], row["dm_p_vs_rw"] = dm_stat, dm_p
        if has_own:
            # The test that decides whether the fundamental features earn
            # their place: this model against the same ridge fitted on
            # freight's own past alone.
            own_stat, own_p = diebold_mariano(actual, pred, own_pred, horizon)
            row["dm_vs_own"], row["dm_p_vs_own"] = own_stat, own_p
        rows.append(row)

    metrics = pd.DataFrame(rows).set_index("model")
    coefficients = (pd.DataFrame(coef_rows) if coef_rows else pd.DataFrame())
    return HorizonResult(horizon=int(horizon), predictions=frame,
                         metrics=metrics, coefficients=coefficients)


def run(bundle: dict[str, pd.DataFrame]) -> dict[int, HorizonResult]:
    X, Y, F = bundle["X"], bundle["Y"], bundle["forward"]
    results: dict[int, HorizonResult] = {}
    for h in get("target", "target", "horizons_bd"):
        h = int(h)
        column = f"{TARGET_PREFIX}{h}"
        log.info("walk-forward h=%d", h)
        results[h] = run_horizon(X, Y[column], F[column], h)
    return results


def summary(results: dict[int, HorizonResult]) -> pd.DataFrame:
    frames = []
    for h, res in sorted(results.items()):
        block = res.metrics.copy()
        block.insert(0, "horizon_bd", h)
        frames.append(block.reset_index())
    return pd.concat(frames, ignore_index=True)


def family_importance(res: HorizonResult) -> pd.Series:
    """Mean absolute standardised coefficient, aggregated to family.

    Aggregated rather than per column on purpose: twenty on-water columns
    each carrying a small weight is one idea, and reading them individually
    invites the same mistake the sibling StockPicker made.
    """
    if res.coefficients.empty:
        return pd.Series(dtype=float)
    mean_abs = res.coefficients.abs().mean()
    return mean_abs.groupby([family_of(c) for c in mean_abs.index]).sum(
        ).sort_values(ascending=False)
