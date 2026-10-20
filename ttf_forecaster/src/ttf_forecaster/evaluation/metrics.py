"""Forecast and trading metrics, groupable by fold, year, regime and horizon.

Spec: ``research_specification.md`` §7 (Metrics). Conventions:

* Inputs are aligned 1-D arrays; pairs with a NaN in any input are dropped and the count of used
  observations is returned as ``n``. A metric that needs more data than is present returns NaN.
* **OOS R² vs benchmark** (``1 - MSE_model / MSE_bench``) is the headline skill number. For return
  targets the benchmark is the zero forecast (the forward curve).
* ``r2_vs_test_mean`` is the classic R² against the *test-period* mean. That mean is hindsight, so
  this number is **not** a skill measure; it is reported only because people ask for it.
* MASE needs an explicit scale computed on TRAINING data (``mase_scale``); without one it is NaN
  rather than silently scaled by the test sample.
* Overlapping multi-day targets make consecutive errors dependent: metric *levels* are fine, but
  any standard error must come from ``statistical_tests`` (HAC), never from these numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import MARKET_TZ

ArrayLike = Sequence[float] | np.ndarray | pd.Series

# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _clean(*arrays: ArrayLike) -> list[np.ndarray]:
    arrs = [np.asarray(a, dtype=float).ravel() for a in arrays]
    n = {len(a) for a in arrs}
    if len(n) != 1:
        raise ValueError(f"metric inputs have different lengths {sorted(n)}")
    ok = np.ones(arrs[0].shape, dtype=bool)
    for a in arrs:
        ok &= np.isfinite(a)
    return [a[ok] for a in arrs]


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0.0,) and np.isfinite(b) else float("nan")


# ---------------------------------------------------------------------------------------------
# Point forecasts
# ---------------------------------------------------------------------------------------------


def oos_r2(y: ArrayLike, yhat: ArrayLike, ybench: ArrayLike) -> float:
    """Out-of-sample R² vs a benchmark: ``1 - MSE(model) / MSE(bench)`` on common non-NaN rows.

    Positive = the model has lower squared error than the benchmark. For return targets with the
    zero benchmark, ``MSE(bench) = mean(y²)``.
    """
    y_, f_, b_ = _clean(y, yhat, ybench)
    if len(y_) == 0:
        return float("nan")
    return 1.0 - _safe_div(float(np.mean((y_ - f_) ** 2)), float(np.mean((y_ - b_) ** 2)))


def mase(y: ArrayLike, yhat: ArrayLike, scale: float) -> float:
    """Mean absolute scaled error with a TRAINING-sample ``scale`` (see ``mase_scale``)."""
    y_, f_ = _clean(y, yhat)
    if len(y_) == 0 or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(np.mean(np.abs(y_ - f_)) / scale)


def mase_scale(y_train: ArrayLike, *, naive: str = "zero", lag: int = 1) -> float:
    """MASE denominator from TRAINING data only.

    Args:
        naive: ``"zero"`` - mean |y| (the zero/forward benchmark; use for return targets);
            ``"lag"`` - mean |y_t - y_{t-lag}| (random walk / seasonal naive; level targets).
    """
    (y_,) = _clean(y_train)
    if naive == "zero":
        return float(np.mean(np.abs(y_))) if len(y_) else float("nan")
    if naive == "lag":
        return float(np.mean(np.abs(y_[lag:] - y_[:-lag]))) if len(y_) > lag else float("nan")
    raise ValueError(f"naive must be 'zero' or 'lag', got {naive!r}")


def point_metrics(
    y: ArrayLike,
    yhat: ArrayLike,
    ybench: ArrayLike | None = None,
    *,
    mase_scale_: float | None = None,
) -> dict[str, float]:
    """MAE, RMSE, median AE, bias, correlation, R² (caveat), MASE and OOS R² vs ``ybench``.

    Rows are restricted to those where ``y``, ``yhat`` (and ``ybench`` if given) are all finite, so
    model and benchmark are always compared on the same origins.
    """
    if ybench is None:
        y_, f_ = _clean(y, yhat)
        b_ = None
    else:
        y_, f_, b_ = _clean(y, yhat, ybench)
    n = len(y_)
    out: dict[str, float] = {"n": float(n)}
    if n == 0:
        for k in ("mae", "rmse", "median_ae", "bias", "corr", "r2_vs_test_mean", "mase"):
            out[k] = float("nan")
        if b_ is not None:
            out.update(oos_r2=float("nan"), mse_bench=float("nan"), rel_mae_vs_bench=float("nan"))
        return out
    e = y_ - f_
    out["mae"] = float(np.mean(np.abs(e)))
    out["mse"] = float(np.mean(e**2))
    out["rmse"] = float(np.sqrt(out["mse"]))
    out["median_ae"] = float(np.median(np.abs(e)))
    out["bias"] = float(np.mean(f_ - y_))  # positive = over-forecast
    out["corr"] = (
        float(np.corrcoef(y_, f_)[0, 1]) if n > 2 and np.std(f_) > 0 and np.std(y_) > 0
        else float("nan")
    )
    ss_tot = float(np.sum((y_ - y_.mean()) ** 2))
    out["r2_vs_test_mean"] = 1.0 - _safe_div(float(np.sum(e**2)), ss_tot)
    out["mase"] = mase(y_, f_, mase_scale_) if mase_scale_ is not None else float("nan")
    if b_ is not None:
        mse_b = float(np.mean((y_ - b_) ** 2))
        out["mse_bench"] = mse_b
        out["oos_r2"] = 1.0 - _safe_div(out["mse"], mse_b)
        out["rel_mae_vs_bench"] = _safe_div(out["mae"], float(np.mean(np.abs(y_ - b_))))
    return out


# ---------------------------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------------------------


def classification_metrics(
    y_true: ArrayLike, prob: ArrayLike, threshold: float = 0.5
) -> dict[str, float]:
    """Balanced accuracy, precision, recall, MCC, Brier and base rate for a binary label.

    ``prob`` is P(y=1). A threshold other than 0.5 must be chosen on training/inner folds only.
    """
    y_, p_ = _clean(y_true, prob)
    if len(y_) and not np.isin(y_, (0.0, 1.0)).all():
        raise ValueError("y_true must be binary 0/1")
    if len(p_) and ((p_ < 0) | (p_ > 1)).any():
        raise ValueError("prob must lie in [0, 1]")
    n = len(y_)
    if n == 0:
        return {k: float("nan") for k in
                ("n", "balanced_accuracy", "precision", "recall", "mcc", "brier", "base_rate")}
    pred = (p_ >= threshold).astype(float)
    tp = float(np.sum((pred == 1) & (y_ == 1)))
    tn = float(np.sum((pred == 0) & (y_ == 0)))
    fp = float(np.sum((pred == 1) & (y_ == 0)))
    fn = float(np.sum((pred == 0) & (y_ == 1)))
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "n": float(n),
        "balanced_accuracy": float(np.nanmean([tpr, tnr])),
        "precision": _safe_div(tp, tp + fp),
        "recall": tpr,
        "mcc": _safe_div(tp * tn - fp * fn, float(denom)) if denom > 0 else 0.0,
        "brier": float(np.mean((p_ - y_) ** 2)),
        "base_rate": float(np.mean(y_)),
    }


def calibration_curve(y_true: ArrayLike, prob: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: equal-width probability bins with mean forecast, observed frequency, n."""
    y_, p_ = _clean(y_true, prob)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p_, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = b == i
        rows.append(
            {
                "bin_lo": edges[i],
                "bin_hi": edges[i + 1],
                "n": int(m.sum()),
                "mean_prob": float(p_[m].mean()) if m.any() else float("nan"),
                "observed_freq": float(y_[m].mean()) if m.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Probabilistic
# ---------------------------------------------------------------------------------------------


def pinball_loss(y: ArrayLike, q: ArrayLike, tau: float) -> float:
    """Mean pinball (quantile) loss of quantile forecast ``q`` at level ``tau``."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must be in (0, 1)")
    y_, q_ = _clean(y, q)
    if len(y_) == 0:
        return float("nan")
    d = y_ - q_
    return float(np.mean(np.maximum(tau * d, (tau - 1.0) * d)))


def interval_metrics(y: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> dict[str, float]:
    """Empirical coverage and mean width of the interval [lo, hi]."""
    y_, lo_, hi_ = _clean(y, lo, hi)
    if len(y_) == 0:
        return {"n": 0.0, "coverage": float("nan"), "width": float("nan")}
    if (hi_ < lo_).any():
        raise ValueError("interval upper bound below lower bound (quantile crossing)")
    return {
        "n": float(len(y_)),
        "coverage": float(np.mean((y_ >= lo_) & (y_ <= hi_))),
        "width": float(np.mean(hi_ - lo_)),
    }


def crps_from_quantiles(y: ArrayLike, quantiles: np.ndarray | pd.DataFrame,
                        taus: Sequence[float]) -> float:
    """Approximate CRPS as ``2 x`` the mean pinball loss over the supplied quantile levels.

    Exact in the limit of a dense, evenly spaced grid of ``taus`` (CRPS = 2∫ pinball_τ dτ). With a
    coarse grid (e.g. 0.1/0.5/0.9) it is a *comparative* score only: compare models on the same
    grid, never against a CRPS computed another way.
    """
    qs = np.asarray(quantiles, dtype=float)
    if qs.ndim != 2 or qs.shape[1] != len(taus):
        raise ValueError("quantiles must be (n, len(taus))")
    losses = [pinball_loss(y, qs[:, j], t) for j, t in enumerate(taus)]
    return float(2.0 * np.mean(losses))


# ---------------------------------------------------------------------------------------------
# Trading
# ---------------------------------------------------------------------------------------------


def max_drawdown(pnl: ArrayLike) -> float:
    """Largest peak-to-trough fall of cumulative P&L (positive number, same unit as ``pnl``)."""
    (p,) = _clean(pnl)
    if len(p) == 0:
        return float("nan")
    cum = np.cumsum(p)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return float(np.max(peak - cum))


def pnl_concentration(pnl: ArrayLike, top_share: float = 0.05) -> float:
    """Share of total net P&L earned on the best ``top_share`` of days.

    ``sum(top 5% days) / sum(all days)``. > 1 means the rest of the sample lost money; NaN when the
    total is <= 0 (concentration of a loss is not meaningful).
    """
    (p,) = _clean(pnl)
    if len(p) == 0:
        return float("nan")
    total = float(p.sum())
    if total <= 0:
        return float("nan")
    k = max(1, int(np.ceil(top_share * len(p))))
    return float(np.sort(p)[::-1][:k].sum() / total)


def trading_metrics(daily: pd.DataFrame, annualisation_days: int = 252) -> dict[str, float]:
    """Summary of a daily P&L frame from ``backtest.simulate_trading``.

    Expects columns ``gross_pnl``, ``net_pnl``, ``cost``, ``traded``. P&L is EUR/MWh per unit
    position. Sharpe and vol use net P&L. Hit rate counts days with a non-zero position.
    """
    need = {"gross_pnl", "net_pnl", "cost", "traded"}
    if not need <= set(daily.columns):
        raise ValueError(f"daily P&L frame lacks columns {sorted(need - set(daily.columns))}")
    d = daily.dropna(subset=["net_pnl"])
    net = d["net_pnl"].to_numpy(dtype=float)
    n = len(net)
    if n == 0:
        return {"n_days": 0.0}
    sd = float(np.std(net, ddof=1)) if n > 1 else float("nan")
    ann_vol = sd * np.sqrt(annualisation_days)
    active = d["position_held"].to_numpy() != 0 if "position_held" in d else np.ones(n, bool)
    act = net[active]
    wins, losses = act[act > 0], act[act < 0]
    return {
        "n_days": float(n),
        "gross_pnl": float(d["gross_pnl"].sum()),
        "net_pnl": float(net.sum()),
        "total_cost": float(d["cost"].sum()),
        "mean_daily_net": float(net.mean()),
        "ann_vol": ann_vol,
        "sharpe": _safe_div(float(net.mean()) * annualisation_days, ann_vol),
        "max_drawdown": max_drawdown(net),
        "max_dd_over_ann_vol": _safe_div(max_drawdown(net), ann_vol),
        "turnover_per_day": float(d["traded"].mean()),
        "hit_rate": float(np.mean(act > 0)) if len(act) else float("nan"),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "pnl_top5pct_share": pnl_concentration(net, 0.05),
    }


# ---------------------------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------------------------


def add_group_columns(
    frame: pd.DataFrame,
    regimes: pd.Series | None = None,
    time_col: str = "origin_time",
) -> pd.DataFrame:
    """Add ``year`` (Europe/Amsterdam year of ``time_col``) and ``regime`` (from a Series indexed
    by the same timestamps; labels must be PIT - computed with data known at each origin)."""
    out = frame.copy()
    t = pd.DatetimeIndex(out[time_col])
    out["year"] = (t.tz_convert(MARKET_TZ) if t.tz is not None else t).year
    if regimes is not None:
        out["regime"] = regimes.reindex(t).to_numpy()
    return out


def paired_frame(
    predictions: pd.DataFrame, model_id: str, benchmark_id: str
) -> pd.DataFrame:
    """Align a model's and a benchmark's predictions on (origin_time, target_id, horizon).

    ``predictions`` is the long frame from ``backtest.walk_forward`` (must contain ``actual``).
    Returns columns ``origin_time, target_id, horizon, fold_id, actual, model, bench``. Raises if
    the two models were not evaluated on the same origins (a silent inner join would let a model
    choose its own test sample).
    """
    keys = ["origin_time", "target_id", "horizon"]
    m = predictions.loc[predictions["model_id"] == model_id]
    b = predictions.loc[predictions["model_id"] == benchmark_id]
    if m.empty or b.empty:
        raise ValueError(f"no predictions for {model_id!r} or {benchmark_id!r}")
    mk = m.set_index(keys).index
    bk = b.set_index(keys).index
    if not mk.sort_values().equals(bk.sort_values()):
        raise ValueError(
            f"{model_id!r} and {benchmark_id!r} cover different origins; evaluate both on the "
            "same folds"
        )
    out = m[[*keys, "fold_id", "actual", "prediction"]].rename(columns={"prediction": "model"})
    out = out.merge(b[[*keys, "prediction"]].rename(columns={"prediction": "bench"}), on=keys)
    return out.sort_values("origin_time").reset_index(drop=True)


def metrics_by(
    frame: pd.DataFrame,
    by: str | Sequence[str] | None,
    fn: Callable[[pd.DataFrame], Mapping[str, Any]],
) -> pd.DataFrame:
    """Apply ``fn`` to each group of ``frame`` (``by`` None -> a single 'all' row)."""
    if by is None:
        return pd.DataFrame([{"group": "all", **fn(frame)}])
    cols = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, g in frame.groupby(cols, sort=True, dropna=False):
        key_t = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(cols, key_t, strict=True)), **fn(g)})
    return pd.DataFrame(rows)


def point_metrics_by(paired: pd.DataFrame, by: str | Sequence[str] | None = None,
                     mase_scale_: float | None = None) -> pd.DataFrame:
    """``point_metrics`` (model vs bench) for each group of a ``paired_frame`` output."""
    return metrics_by(
        paired, by,
        lambda g: point_metrics(g["actual"], g["model"], g["bench"], mase_scale_=mase_scale_),
    )


def trading_metrics_by(daily: pd.DataFrame, by: str | Sequence[str] | None = None,
                       annualisation_days: int = 252) -> pd.DataFrame:
    """``trading_metrics`` for each group of a daily P&L frame (e.g. by ``year`` or ``regime``)."""
    return metrics_by(daily, by, lambda g: trading_metrics(g, annualisation_days))
