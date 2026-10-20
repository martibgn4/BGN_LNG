"""Forecast comparison tests and the spec §8 success-criteria check (criteria 1-2).

Which test to read
------------------
Every return model here **nests** the zero-return (forward-curve) benchmark: set its coefficients to
zero and it *is* the benchmark. Under the null that the extra predictors are useless, the larger
model still pays estimation noise, so its population MSE equals the benchmark's but its *sample*
MSE is higher. The Diebold-Mariano test is therefore biased **against** the larger model (undersized,
low power) in nested comparisons. Clark & West (2007) adjust the loss differential for that noise.

* ``diebold_mariano`` - the spec's headline test (§8 criterion 1). Conservative for nested models:
  a significant DM is strong evidence; an insignificant DM is not proof of no skill.
* ``clark_west`` - the correct test of "the predictors add information" for nested models. Report
  it next to DM. It tests population predictability, not that the estimated model would have
  beaten the benchmark in this sample - a significant CW with a negative OOS R² means "the signal
  exists but is too weak to exploit with this estimator", which is not a usable forecast.

Common assumptions / limitations (both tests)
---------------------------------------------
* Loss differentials are covariance-stationary with autocorrelation dying out after ``h - 1`` lags
  (h-step forecasts on overlapping windows are MA(h-1)). Newey-West (Bartlett) HAC with lag ``h-1``
  is used; with h = 1 it reduces to the sample variance.
* Regime shifts (2021-2023) violate stationarity; the pooled p-value then averages over regimes.
  This is why §8 criterion 2 re-evaluates excluding the crisis window.
* Rolling/expanding re-estimation is assumed; the tests do not account for model selection or
  hyperparameter search across many specifications - a p-value from the best of N models is not a
  p-value (data snooping; see White's reality check / Hansen SPA, not implemented).
* HLN correction: derived for the rectangular-window DM variance; applying it with a Bartlett
  kernel is standard practice and conservative in small samples. Student-t(T-1) reference.
* Effective sample is far smaller than T for overlapping targets; with few independent episodes
  (e.g. one crisis) asymptotics are poor whatever the correction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ttf_forecaster.data.base import MARKET_TZ
from ttf_forecaster.evaluation.metrics import oos_r2
from ttf_forecaster.settings import load_config
from ttf_forecaster.targets.common import session_horizon

ALTERNATIVES: tuple[str, ...] = ("less", "two-sided", "greater")


@dataclass(frozen=True)
class TestResult:
    """Outcome of a forecast-comparison test.

    ``mean_loss_diff`` is mean(model loss - benchmark loss) for DM (negative favours the model) and
    the mean CW-adjusted differential for CW (positive favours the larger model).
    """

    __test__ = False  # not a pytest test class

    method: str
    statistic: float
    p_value: float
    mean_loss_diff: float
    n: int
    hac_lag: int
    alternative: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def newey_west_variance(x: np.ndarray, lag: int) -> float:
    """Long-run variance of ``x`` (demeaned) with Bartlett weights ``1 - k/(lag+1)``."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    v = float(np.dot(x, x) / n)
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        v += 2.0 * w * float(np.dot(x[k:], x[:-k]) / n)
    return v


def _loss(e: np.ndarray, loss: str) -> np.ndarray:
    if loss == "mse":
        return e**2
    if loss == "mae":
        return np.abs(e)
    raise ValueError(f"loss must be 'mse' or 'mae', got {loss!r}")


def _clean_pairs(*arrays: Sequence[float] | np.ndarray | pd.Series) -> list[np.ndarray]:
    arrs = [np.asarray(a, dtype=float).ravel() for a in arrays]
    if len({len(a) for a in arrs}) != 1:
        raise ValueError("inputs must have equal length")
    ok = np.logical_and.reduce([np.isfinite(a) for a in arrs])
    return [a[ok] for a in arrs]


def session_hac_lag(targets: pd.DataFrame, declared: int | None = None) -> int:
    """DM / CW HAC lag for a target frame: ``session_horizon - 1`` (= label overlap rows).

    D-032 F1: the lag is taken from the label overlap, never from a ``horizon`` column that may hold
    a month count (T4/T5). ``declared`` is refused when it disagrees with the overlap.
    """
    return session_horizon(targets, declared) - 1


def diebold_mariano(
    e_model: Sequence[float] | np.ndarray | pd.Series,
    e_bench: Sequence[float] | np.ndarray | pd.Series,
    h: int,
    *,
    loss: str = "mse",
    alternative: str = "less",
    hln: bool = True,
) -> TestResult:
    """Diebold-Mariano test of equal predictive accuracy with NW-HAC (lag h-1) and HLN correction.

    Args:
        e_model, e_bench: forecast errors (actual - forecast) in TIME ORDER on the same origins.
        h: forecast horizon in trading SESSIONS (overlap -> MA(h-1) loss differential). Derive it
            with :func:`session_horizon`; a month count must never be passed (D-032 F1).
        loss: ``"mse"`` or ``"mae"``.
        alternative: ``"less"`` (H1: model loss < benchmark loss; spec §8), ``"two-sided"`` or
            ``"greater"``.
        hln: apply the Harvey-Leybourne-Newbold small-sample factor and a t(T-1) reference.

    Limitations: biased against the larger model when it nests the benchmark (see module doc);
    non-stationary differentials across regimes; no allowance for model search.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")
    if h < 1:
        raise ValueError("h must be >= 1")
    em, eb = _clean_pairs(e_model, e_bench)
    d = _loss(em, loss) - _loss(eb, loss)
    n = len(d)
    if n < max(10, 2 * h):
        return TestResult("DM-HLN" if hln else "DM", float("nan"), float("nan"),
                          float(d.mean()) if n else float("nan"), n, h - 1, alternative)
    lrv = newey_west_variance(d, h - 1)
    if lrv <= 0:
        lrv = float(np.var(d)) or np.finfo(float).tiny
    stat = float(d.mean() / np.sqrt(lrv / n))
    if hln:
        stat *= float(np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n))
        dist: Any = stats.t(df=n - 1)
    else:
        dist = stats.norm()
    if alternative == "less":
        p = float(dist.cdf(stat))
    elif alternative == "greater":
        p = float(dist.sf(stat))
    else:
        p = float(2 * dist.sf(abs(stat)))
    return TestResult("DM-HLN" if hln else "DM", stat, p, float(d.mean()), n, h - 1, alternative)


def clark_west(
    y: Sequence[float] | np.ndarray | pd.Series,
    f_small: Sequence[float] | np.ndarray | pd.Series,
    f_large: Sequence[float] | np.ndarray | pd.Series,
    h: int,
) -> TestResult:
    """Clark-West (2007) MSPE-adjusted test for NESTED models; one-sided, H1: large model better.

    ``f_t = (y - f_small)² - [(y - f_large)² - (f_small - f_large)²]``; t-statistic of mean(f) with
    NW-HAC (lag h-1), compared with the standard normal upper tail (CW's recommendation; slightly
    conservative for rolling schemes, approximately right for expanding ones).

    Use with ``f_small = 0`` for any return model vs the forward-curve benchmark. Only valid when the
    small model really is nested in the large one (same target, large model's predictor set
    contains the small one's). Tests population predictability, not realised OOS superiority.
    """
    if h < 1:
        raise ValueError("h must be >= 1")
    y_, s_, l_ = _clean_pairs(y, f_small, f_large)
    f = (y_ - s_) ** 2 - ((y_ - l_) ** 2 - (s_ - l_) ** 2)
    n = len(f)
    if n < max(10, 2 * h):
        return TestResult("CW", float("nan"), float("nan"),
                          float(f.mean()) if n else float("nan"), n, h - 1, "greater")
    lrv = newey_west_variance(f, h - 1)
    if lrv <= 0:
        lrv = float(np.var(f)) or np.finfo(float).tiny
    stat = float(f.mean() / np.sqrt(lrv / n))
    return TestResult("CW", stat, float(stats.norm.sf(stat)), float(f.mean()), n, h - 1, "greater")


# ---------------------------------------------------------------------------------------------
# Success criteria (spec §8, criteria 1-2)
# ---------------------------------------------------------------------------------------------


def _crisis_mask(origins: pd.Series, start: str, end: str) -> np.ndarray:
    local_date = pd.DatetimeIndex(origins).tz_convert(MARKET_TZ).tz_localize(None).normalize()
    return np.asarray((local_date >= pd.Timestamp(start)) & (local_date <= pd.Timestamp(end)))


def evaluate_success_criteria(
    paired: pd.DataFrame,
    horizon: int,
    *,
    clean_pit: bool,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate spec §8 criteria 1 and 2 for one target x horizon; pass/fail with the numbers.

    Args:
        paired: ``metrics.paired_frame`` output (``origin_time, fold_id, actual, model, bench``) for
            OUTER folds only (never inner folds, never the holdout unless that is the question).
        horizon: trading-day horizon (DM/CW HAC lag = h - 1).
        clean_pit: True only if every feature used has pit_grade A or B (criterion 1 is stated
            "on the clean-PIT feature set"; False fails it outright).
        cfg: ``backtests`` config mapping (YAML when None).

    Returns:
        ``{"criterion_1": {...}, "criterion_2": {...}}`` each with ``passed`` and the evidence.
        CW is reported under criterion 1 as supplementary (it does not change pass/fail; the spec's
        test is HLN-DM). Criteria 3-7 need other inputs (price-only model, trading, calibration,
        review, holdout) and are not evaluated here.
    """
    cfg = load_config("backtests") if cfg is None else cfg
    sc = cfg["success_criteria"]
    share_req = float(sc["fold_win_share"])
    alpha = float(sc["dm_alpha"])
    p = paired.sort_values("origin_time").dropna(subset=["actual", "model", "bench"])
    if p.empty:
        raise ValueError("no complete (actual, model, bench) rows to evaluate")

    fold_rows = []
    for fid, g in p.groupby("fold_id", sort=True):
        mse_m = float(np.mean((g["actual"] - g["model"]) ** 2))
        mse_b = float(np.mean((g["actual"] - g["bench"]) ** 2))
        fold_rows.append({"fold_id": fid, "n": len(g), "mse_model": mse_m, "mse_bench": mse_b,
                          "mse_reduction": mse_b - mse_m})
    folds = pd.DataFrame(fold_rows)
    win_share = float((folds["mse_reduction"] > 0).mean())
    e_m = (p["actual"] - p["model"]).to_numpy()
    e_b = (p["actual"] - p["bench"]).to_numpy()
    dm = diebold_mariano(e_m, e_b, horizon, alternative="less", hln=True)
    cw = clark_west(p["actual"], p["bench"], p["model"], horizon)
    min_folds = int(sc.get("min_folds", 4))
    evaluable = len(folds) >= min_folds
    c1_pass = bool(evaluable and clean_pit and win_share >= share_req
                   and np.isfinite(dm.p_value) and dm.p_value < alpha)
    notes = []
    if not evaluable:
        # D-031: a 2-fold synthetic NULL world "passed" (HLN-DM p = 0.004); too few folds.
        notes.append(f"not evaluable: {len(folds)} outer folds < min_folds={min_folds}")
    if not clean_pit:
        notes.append("feature set is not clean-PIT: criterion fails by definition")
    if len(folds) < 4:
        notes.append(f"only {len(folds)} outer folds: fold-share is coarse")

    crisis = sc["crisis_window"]
    mask = _crisis_mask(p["origin_time"], str(crisis["start"]), str(crisis["end"]))
    r2_all = oos_r2(p["actual"], p["model"], p["bench"])
    ex = p.loc[~mask]
    r2_ex = oos_r2(ex["actual"], ex["model"], ex["bench"]) if len(ex) else float("nan")
    c2_notes = []
    if len(ex) == 0:
        c2_notes.append("no origins outside the crisis window: criterion cannot pass")
    c2_pass = bool(np.isfinite(r2_all) and np.isfinite(r2_ex) and r2_all > 0 and r2_ex > 0)

    return {
        "criterion_1": {
            "passed": c1_pass,
            "fold_win_share": win_share,
            "required_share": share_req,
            "n_folds": len(folds),
            "dm_stat": dm.statistic,
            "dm_p_one_sided": dm.p_value,
            "alpha": alpha,
            "clean_pit": clean_pit,
            "cw_stat_supplementary": cw.statistic,
            "cw_p_supplementary": cw.p_value,
            "per_fold": folds.to_dict("records"),
            "notes": notes,
        },
        "criterion_2": {
            "passed": c2_pass,
            "oos_r2_all": r2_all,
            "oos_r2_ex_crisis": r2_ex,
            "n_all": len(p),
            "n_ex_crisis": len(ex),
            "crisis_window": [str(crisis["start"]), str(crisis["end"])],
            "notes": c2_notes,
        },
    }
