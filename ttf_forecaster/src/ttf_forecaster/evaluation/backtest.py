"""Walk-forward forecast evaluation and a trading simulation with explicit execution conventions.

Two separate exercises, never to be confused:

1. ``walk_forward`` - **forecast evaluation**. Fits a fresh model at every refit point of every
   fold on purged training rows only, predicts the fold's test origins, and returns
   ``PREDICTION_SCHEMA`` rows (plus ``actual`` and ``contract_id``) with full provenance. Benchmarks
   go through exactly the same folds and rows.
2. ``simulate_trading`` - converts one model's forecasts into daily P&L in **EUR/MWh per unit
   position** under configured conventions (``configs/backtests.yaml: trading``):

   * signal observed at the origin (close of trading date t, after settlement publication);
   * filled at the settlement ``execution_delay`` trading days later (default 1 = next settlement;
     0 is refused unless ``allow_same_settlement`` - the same settlement that produced the signal
     is not tradeable);
   * instrument = the target's held contract for that origin (``contract_id`` from the targets
     frame, i.e. the roll rule of ``target_definitions.md`` §1); marked settle-to-settle on that
     contract only, so no P&L ever crosses a roll;
   * position = ``clip(forecast / forecast_unit, ±position_limit)`` (``scaled``) or
     ``±position_limit`` (``sign``); a new position every ``rebalance_every`` origins;
   * cost per unit traded = half bid-offer + fee per side (ASSUMPTIONS) × ``cost_multiplier``;
     a change of contract (roll) trades both legs (|old| + |new|); the final position is closed on
     the last day;
   * missing settlements give NaN P&L for that day (counted), never an interpolated price.

The model interface is duck-typed (``models/interface.py`` is written separately): ``fit(X, y)``
and ``predict(X)``, optionally ``predict_quantiles(X, quantiles)`` -> (n, len(quantiles)) and
``predict_proba(X)`` -> (n,) or (n, 2), each used only if the model does not declare
``supports_quantiles`` / ``supports_proba`` False (``models.interface.ForecastModel`` convention).
Quantile levels are passed sorted ascending and returned columns are taken in that order.
"""

from __future__ import annotations

import inspect

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    MARKET_TZ,
    PREDICTION_SCHEMA,
    TARGET_SCHEMA,
    PointInTimeViolation,
    QualityFlag,
    validate_frame,
)
from ttf_forecaster.evaluation.metrics import paired_frame, trading_metrics
from ttf_forecaster.evaluation.splits import Fold, HoldoutLockedError, SplitError
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.settings import ConfigError, load_config

log = get_logger(__name__)

EXERCISES: dict[str, str] = {
    "forecast_evaluation": (
        "FORECAST EVALUATION: P&L measures the economic value of the forecast under stated "
        "conventions. It is a diagnostic, not a deployable strategy or a capacity estimate."
    ),
    "directional_speculation": (
        "DIRECTIONAL SPECULATION: P&L of a stand-alone futures strategy. Requires confirmed costs, "
        "margin and liquidity before it can be quoted."
    ),
    "hedging_improvement": (
        "HEDGING IMPROVEMENT: forecast used to time an existing hedge; needs the underlying "
        "exposure and is not implemented."
    ),
}
SIZING_METHODS: tuple[str, ...] = ("scaled", "sign")


# ---------------------------------------------------------------------------------------------
# Benchmark model and helpers
# ---------------------------------------------------------------------------------------------


class ZeroForecast:
    """The forward-curve benchmark for excess-return targets: always predicts 0."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ZeroForecast:  # noqa: N803
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return np.zeros(len(X))


def feature_hash(X: pd.DataFrame) -> str:
    """Stable short hash of the feature set (column names and dtypes, in order).

    ``X.attrs["feature_hash"]`` (set by the feature pipeline) takes precedence.
    """
    if "feature_hash" in X.attrs:
        return str(X.attrs["feature_hash"])
    spec = "|".join(f"{c}:{X[c].dtype}" for c in X.columns)
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]


def _check_utc_index(idx: pd.Index, name: str) -> pd.DatetimeIndex:
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or str(idx.tz) != "UTC":
        raise SplitError(f"{name} must be indexed by a tz-aware UTC DatetimeIndex (origin_time)")
    if idx.has_duplicates:
        raise SplitError(f"{name} has duplicated origins")
    return idx


def _prepare_targets(targets: pd.DataFrame) -> pd.DataFrame:
    validate_frame(targets, TARGET_SCHEMA)
    if targets[["target_id", "horizon"]].drop_duplicates().shape[0] != 1:
        raise SplitError("targets must hold exactly one (target_id, horizon)")
    t = targets.set_index(pd.DatetimeIndex(targets["origin_time"]), drop=False)
    _check_utc_index(t.index, "targets")
    return t.sort_index()


# ---------------------------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Output of ``walk_forward``.

    Attributes:
        predictions: long frame, PREDICTION_SCHEMA columns + ``actual``, ``contract_id`` and
            optional ``q<level>`` / ``prob`` columns; one row per (model_id, origin).
        fit_log: one row per (model_id, fold, refit) with training size and drops.
        folds: ``Fold.describe()`` rows.
    """

    predictions: pd.DataFrame
    fit_log: pd.DataFrame
    folds: pd.DataFrame

    def paired(self, model_id: str, benchmark_id: str = "zero") -> pd.DataFrame:
        """``metrics.paired_frame`` for one model vs one benchmark."""
        return paired_frame(self.predictions, model_id, benchmark_id)


def _accepts_label_known_time(model: Any) -> bool:
    """True if ``model.fit`` takes a ``label_known_time`` keyword (models/interface.py)."""
    try:
        return "label_known_time" in inspect.signature(model.fit).parameters
    except (TypeError, ValueError):
        return False


def _supports(model: Any, method: str, flag: str) -> bool:
    """True if ``model`` offers ``method`` and does not declare ``flag = False``.

    ``models.interface.ForecastModel`` always defines ``predict_quantiles``/``predict_proba`` but
    raises NotImplementedError unless ``supports_quantiles``/``supports_proba`` is set.
    """
    return callable(getattr(model, method, None)) and bool(getattr(model, flag, True))


def _verify_segment(fold: Fold, seg_i: int, train: pd.DatetimeIndex, pred: pd.DatetimeIndex,
                    refit: pd.Timestamp, lkt: pd.Series) -> None:
    """Defence in depth: re-check the PIT invariants a Fold promises."""
    where = f"{fold.fold_id} segment {seg_i}"
    if len(pred) == 0 or pred[0] != refit:
        raise SplitError(f"{where}: refit time must equal the first predicted origin")
    if len(train) and train.max() >= refit:
        raise PointInTimeViolation(f"{where}: training origin {train.max()} >= refit time {refit}")
    late = lkt.reindex(train) > refit
    if late.any():
        raise PointInTimeViolation(
            f"{where}: {int(late.sum())} training labels not known at refit time {refit} "
            "(purge failed)"
        )


def _run_one(
    factory: Callable[[], Any],
    mid: str,
    X: pd.DataFrame,
    t: pd.DataFrame,
    splits: Sequence[Fold],
    fhash: str,
    synthetic: bool,
    quantiles: Sequence[float] | None,
    dropna_features: bool,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    rows: list[pd.DataFrame] = []
    logs: list[dict[str, Any]] = []
    lkt = t["label_known_time"]
    for fold in splits:
        for i, seg in enumerate(fold.segments):
            _verify_segment(fold, i, seg.train_origins, seg.predict_origins, seg.refit_time, lkt)
            tr = seg.train_origins[seg.train_origins.isin(X.index)]
            Xtr = X.loc[tr]
            ytr = t.loc[tr, "value"].astype(float)
            keep = ytr.notna().to_numpy()
            if dropna_features:
                keep &= Xtr.notna().all(axis=1).to_numpy()
            n_drop = int((~keep).sum())
            Xtr, ytr = Xtr.loc[keep], ytr.loc[keep]
            if len(ytr) == 0:
                raise SplitError(f"{mid} {fold.fold_id}/{i}: no usable training rows")
            model = factory()
            try:
                if _accepts_label_known_time(model):
                    # D-030: models purge their own inner/early-stopping/calibration splits on it
                    model.fit(Xtr, ytr, label_known_time=lkt.loc[Xtr.index])
                else:
                    model.fit(Xtr, ytr)
            except Exception as exc:  # re-raised with context
                raise RuntimeError(f"{mid} fit failed in {fold.fold_id}/{i}: {exc}") from exc
            po = seg.predict_origins[seg.predict_origins.isin(X.index)]
            Xp = X.loc[po]
            ok = Xp.notna().all(axis=1).to_numpy() if dropna_features else np.ones(len(po), bool)
            pred = np.full(len(po), np.nan)
            if ok.any():
                out = np.asarray(model.predict(Xp.loc[ok]), dtype=float).ravel()
                if len(out) != int(ok.sum()):
                    raise RuntimeError(f"{mid}: predict returned {len(out)} values for {ok.sum()}")
                pred[ok] = out
            frame = pd.DataFrame(
                {
                    "origin_time": po,
                    "target_id": t.loc[po, "target_id"].to_numpy(),
                    "horizon": t.loc[po, "horizon"].to_numpy().astype("int64"),
                    "model_id": mid,
                    "prediction": pred,
                    "train_end": seg.refit_time,
                    "label_known_time": t.loc[po, "label_known_time"].to_numpy(),
                    "fold_id": fold.fold_id,
                    "feature_hash": fhash,
                    "is_synthetic": synthetic,
                    "actual": t.loc[po, "value"].to_numpy(dtype=float),
                    "contract_id": t.loc[po, "contract_id"].to_numpy(),
                }
            )
            if quantiles and _supports(model, "predict_quantiles", "supports_quantiles")                     and ok.any():
                lv = sorted(float(q) for q in quantiles)
                qv = np.asarray(model.predict_quantiles(Xp.loc[ok], lv), dtype=float)
                if qv.shape != (int(ok.sum()), len(lv)):
                    raise RuntimeError(f"{mid}: predict_quantiles shape {qv.shape} is wrong")
                for j, q in enumerate(lv):
                    col = np.full(len(po), np.nan)
                    col[ok] = qv[:, j]
                    frame[f"q{q:g}"] = col
            if _supports(model, "predict_proba", "supports_proba") and ok.any():
                pv = np.asarray(model.predict_proba(Xp.loc[ok]), dtype=float)
                pv = pv[:, -1] if pv.ndim == 2 else pv
                col = np.full(len(po), np.nan)
                col[ok] = pv
                frame["prob"] = col
            rows.append(frame)
            logs.append(
                {
                    "model_id": mid,
                    "fold_id": fold.fold_id,
                    "segment": i,
                    "refit_time": seg.refit_time,
                    "n_train": len(ytr),
                    "n_train_dropped_nan": n_drop,
                    "first_train_origin": tr.min() if len(tr) else pd.NaT,
                    "last_train_origin": tr.max() if len(tr) else pd.NaT,
                    "n_predict": len(po),
                    "n_predict_nan_features": int((~ok).sum()),
                }
            )
    return rows, logs


def walk_forward(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    targets: pd.DataFrame,
    splits: Sequence[Fold],
    benchmark_factories: Mapping[str, Callable[[], Any]] | None = None,
    *,
    model_id: str = "model",
    feature_hash_: str | None = None,
    quantiles: Sequence[float] | None = None,
    is_synthetic: bool | None = None,
    unlock_holdout: bool = False,
    dropna_features: bool = True,
) -> BacktestResult:
    """Fit-and-predict over ``splits`` for a model and its benchmarks, on identical rows.

    Args:
        model_factory: zero-argument callable returning an UNFITTED model (fresh per refit).
        X: feature matrix indexed by ``origin_time`` (tz-aware UTC), built point-in-time.
        targets: TARGET_SCHEMA frame for exactly one (target_id, horizon).
        splits: folds from ``evaluation.splits`` (outer, inner or - if unlocked - holdout).
        benchmark_factories: ``{benchmark_id: factory}``; default ``{"zero": ZeroForecast}``.
            Pass AR/rolling-mean/price-only models here: the spec compares to the BEST benchmark.
        model_id: id written to ``model_id`` for ``model_factory``'s rows.
        feature_hash_: provenance hash of the feature set (default: ``feature_hash(X)``).
        quantiles: levels to request via ``predict_quantiles`` (columns ``q<level>``).
        is_synthetic: provenance flag; default True if any target row is flagged synthetic.
        unlock_holdout: required to run a fold of kind ``holdout``.
        dropna_features: drop training rows with NaN features and return NaN predictions for test
            rows with NaN features (set False for models that handle NaN natively).

    Returns:
        ``BacktestResult``; ``predictions`` passes ``validate_frame(PREDICTION_SCHEMA)``.
        ``train_end`` is the refit time: the information cutoff of the fit (every training label
        was known by then).

    Raises:
        HoldoutLockedError, PointInTimeViolation, SplitError.
    """
    _check_utc_index(X.index, "X")
    t = _prepare_targets(targets)
    if not splits:
        raise SplitError("no folds supplied")
    for f in splits:
        if f.kind == "holdout" and not unlock_holdout:
            raise HoldoutLockedError("holdout fold supplied without unlock_holdout=True")
    benches = (dict(benchmark_factories) if benchmark_factories is not None
               else {"zero": ZeroForecast})  # {} means "no benchmarks" (D-028)
    if model_id in benches:
        raise ValueError(f"model_id {model_id!r} collides with a benchmark id")
    fhash = feature_hash_ or feature_hash(X)
    synthetic = (
        bool((t["quality_flag"] == QualityFlag.SYNTHETIC.value).any())
        if is_synthetic is None else bool(is_synthetic)
    )
    all_rows: list[pd.DataFrame] = []
    all_logs: list[dict[str, Any]] = []
    for mid, fac in [(model_id, model_factory), *benches.items()]:
        rows, logs = _run_one(fac, mid, X, t, splits, fhash, synthetic, quantiles,
                              dropna_features)
        all_rows += rows
        all_logs += logs
    preds = pd.concat(all_rows, ignore_index=True)
    preds["horizon"] = preds["horizon"].astype("int64")
    preds["is_synthetic"] = preds["is_synthetic"].astype(bool)
    preds["train_end"] = pd.DatetimeIndex(preds["train_end"]).tz_convert("UTC")
    validate_frame(preds, PREDICTION_SCHEMA)
    log.info("walk_forward_done", extra={"fields": {
        "model_id": model_id, "n_folds": len(splits), "n_rows": len(preds),
        "is_synthetic": synthetic}})
    return BacktestResult(
        predictions=preds,
        fit_log=pd.DataFrame(all_logs),
        folds=pd.DataFrame([f.describe() for f in splits]),
    )


# ---------------------------------------------------------------------------------------------
# Trading simulation
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TradingConventions:
    """Execution conventions. Cost values are ASSUMPTIONS (``configs/backtests.yaml``)."""

    exercise: str = "forecast_evaluation"
    execution_delay: int = 1
    price_field: str = "settle"
    rebalance_every: int = 1
    sizing_method: str = "scaled"
    forecast_unit: float = 0.01
    position_limit: float = 1.0
    half_bid_offer: float = 0.025
    fee_per_side: float = 0.005
    cost_multiplier: float = 1.0
    annualisation_days: int = 252
    allow_same_settlement: bool = False

    def __post_init__(self) -> None:
        if self.exercise not in EXERCISES:
            raise ConfigError(f"trading.exercise {self.exercise!r} not in {sorted(EXERCISES)}")
        if self.execution_delay < 0:
            raise ConfigError("execution_delay must be >= 0")
        if self.execution_delay == 0 and not self.allow_same_settlement:
            raise ConfigError(
                "execution_delay = 0 fills at the settlement that produced the signal; refused "
                "unless allow_same_settlement=True (spec §8 criterion 4 requires >= 1)"
            )
        if self.sizing_method not in SIZING_METHODS:
            raise ConfigError(f"sizing.method {self.sizing_method!r} not in {SIZING_METHODS}")
        if self.rebalance_every < 1 or self.forecast_unit <= 0 or self.position_limit <= 0:
            raise ConfigError("rebalance_every >= 1, forecast_unit > 0, position_limit > 0")
        if min(self.half_bid_offer, self.fee_per_side, self.cost_multiplier) < 0:
            raise ConfigError("costs must be non-negative")

    @property
    def cost_per_unit(self) -> float:
        """EUR/MWh charged per unit traded (one side)."""
        return (self.half_bid_offer + self.fee_per_side) * self.cost_multiplier

    @property
    def statement(self) -> str:
        return EXERCISES[self.exercise]

    def with_(self, **changes: Any) -> TradingConventions:
        return replace(self, **changes)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None = None, **overrides: Any) -> TradingConventions:
        """Build from ``backtests.trading`` (YAML when ``cfg`` is None); kwargs override."""
        cfg = load_config("backtests") if cfg is None else cfg
        tr = cfg["trading"]
        sizing, costs = tr.get("sizing", {}), tr.get("costs", {})
        kw: dict[str, Any] = {
            "exercise": tr.get("exercise", "forecast_evaluation"),
            "execution_delay": int(tr.get("execution_delay_settlements", 1)),
            "price_field": tr.get("price_field", "settle"),
            "rebalance_every": int(tr.get("rebalance_every", 1)),
            "sizing_method": sizing.get("method", "scaled"),
            "forecast_unit": float(sizing.get("forecast_unit", 0.01)),
            "position_limit": float(sizing.get("position_limit", 1.0)),
            "half_bid_offer": float(costs.get("half_bid_offer_eur_mwh", 0.0)),
            "fee_per_side": float(costs.get("fee_per_side_eur_mwh", 0.0)),
            "annualisation_days": int(tr.get("annualisation_days", 252)),
        }
        kw.update(overrides)
        return cls(**kw)


@dataclass
class TradingResult:
    """Daily P&L (EUR/MWh per unit position) and its summary.

    ``daily`` columns: ``trade_date`` (local date), ``position_held`` and ``contract_held`` (over
    the day ending at this settlement), ``price_change``, ``gross_pnl``, ``traded``, ``cost``,
    ``net_pnl``, ``position_eod``, ``contract_eod``, ``is_roll``.
    """

    daily: pd.DataFrame
    conventions: TradingConventions
    summary: dict[str, float]
    statement: str
    n_missing_price_days: int
    n_nan_forecasts: int
    notes: list[str] = field(default_factory=list)


def _settle_matrix(prices: pd.DataFrame, field_: str) -> pd.DataFrame:
    need = {"trade_date", "contract_id", "value"}
    if not need <= set(prices.columns):
        raise ValueError(f"prices lacks columns {sorted(need - set(prices.columns))}")
    p = prices if "field" not in prices.columns else prices.loc[prices["field"] == field_]
    td = pd.DatetimeIndex(p["trade_date"])
    local = (td.tz_convert(MARKET_TZ) if td.tz is not None else td).tz_localize(None).normalize()
    m = pd.DataFrame({"d": local, "c": p["contract_id"].to_numpy(), "v": p["value"].to_numpy()})
    if m.duplicated(["d", "c"]).any():
        raise ValueError("prices has duplicated (trade_date, contract_id) rows")
    return m.pivot(index="d", columns="c", values="v").sort_index()


def _size(forecast: float, conv: TradingConventions) -> float:
    if not np.isfinite(forecast):
        return 0.0
    if conv.sizing_method == "sign":
        return float(np.sign(forecast) * conv.position_limit)
    return float(np.clip(forecast / conv.forecast_unit, -conv.position_limit, conv.position_limit))


def simulate_trading(
    forecasts: pd.Series,
    targets: pd.DataFrame,
    prices: pd.DataFrame,
    conventions: TradingConventions | None = None,
) -> TradingResult:
    """Daily P&L of trading the held contract on a forecast, under explicit conventions.

    Args:
        forecasts: one model's forecasts indexed by ``origin_time`` (tz-aware UTC). Typically
            ``predictions.loc[predictions.model_id == m].set_index("origin_time")["prediction"]``.
        targets: TARGET_SCHEMA rows of the same target x horizon (``contract_id`` = held contract).
        prices: settlements in CURVE_SCHEMA layout (``trade_date``, ``contract_id``, ``field``,
            ``value``); ``trade_date`` local midnight (UTC) as in the curve schema. The trading
            calendar is the set of dates present.
        conventions: ``TradingConventions`` (from config when None).

    Returns:
        ``TradingResult``. A NaN forecast means flat (counted in ``n_nan_forecasts``).

    Timing: origin t (trading date d_t) -> trade at settlement of calendar date d_t + delay;
    held until the next decision's fill; P&L of day d = position held over (d-1, d] × (F_c(d) -
    F_c(d-1)) on that position's contract.
    """
    conv = conventions or TradingConventions.from_config()
    if conv.exercise == "hedging_improvement":
        raise NotImplementedError(EXERCISES["hedging_improvement"])
    _check_utc_index(forecasts.index, "forecasts")
    t = _prepare_targets(targets)
    F = _settle_matrix(prices, conv.price_field)  # noqa: N806
    cal = F.index
    fc = forecasts.sort_index()
    missing_t = ~fc.index.isin(t.index)
    if missing_t.any():
        raise ValueError(f"{int(missing_t.sum())} forecast origins have no target row/contract")
    origin_dates = fc.index.tz_convert(MARKET_TZ).tz_localize(None).normalize()
    pos_idx = cal.get_indexer(origin_dates)
    if (pos_idx < 0).any():
        raise ValueError("some forecast origins are not on a trading date present in prices")
    contracts = t.loc[fc.index, "contract_id"].to_numpy()
    if pd.isna(contracts).any():
        raise ValueError("held contract_id missing for some origins")

    sel = np.arange(len(fc)) % conv.rebalance_every == 0
    exec_idx = pos_idx[sel] + conv.execution_delay
    dec_pos = np.array([_size(v, conv) for v in fc.to_numpy()[sel]])
    dec_con = contracts[sel]
    n_nan = int((~np.isfinite(fc.to_numpy()[sel])).sum())
    fits = exec_idx < len(cal)
    exec_idx, dec_pos, dec_con = exec_idx[fits], dec_pos[fits], dec_con[fits]
    notes: list[str] = []
    if (~fits).any():
        notes.append(f"{int((~fits).sum())} decisions dropped: fill date beyond price history")
    if len(exec_idx) == 0:
        raise ValueError("no decision can be filled within the price history")
    if (np.diff(exec_idx) <= 0).any():
        raise ValueError("fill dates are not strictly increasing (duplicate origin dates?)")

    decisions = {int(e): (float(p), str(c)) for e, p, c in zip(exec_idx, dec_pos, dec_con,
                                                               strict=True)}
    first = int(exec_idx[0])
    last = min(len(cal) - 1, int(exec_idx[-1]) + conv.rebalance_every)
    cpu = conv.cost_per_unit
    pos, con = 0.0, None
    recs: list[dict[str, Any]] = []
    n_missing = 0
    for d in range(first, last + 1):
        held_pos, held_con = pos, con
        dp = np.nan
        if d > first and held_con is not None and held_pos != 0.0:
            p1 = F.iat[d, F.columns.get_loc(held_con)] if held_con in F.columns else np.nan
            p0 = F.iat[d - 1, F.columns.get_loc(held_con)] if held_con in F.columns else np.nan
            dp = p1 - p0
            if not np.isfinite(dp):
                n_missing += 1
        gross = held_pos * dp if held_pos != 0.0 else 0.0
        traded, is_roll = 0.0, False
        if d in decisions:
            new_pos, new_con = decisions[d]
            if con is None or new_con == con:
                traded = abs(new_pos - pos)
            else:
                traded = abs(pos) + abs(new_pos)
                is_roll = pos != 0.0
            pos, con = new_pos, new_con
        if d == last:  # close out
            traded += abs(pos)
            pos = 0.0
        cost = traded * cpu
        recs.append(
            {
                "trade_date": cal[d],
                "position_held": held_pos,
                "contract_held": held_con,
                "price_change": dp,
                "gross_pnl": gross,
                "traded": traded,
                "cost": cost,
                "net_pnl": gross - cost,
                "position_eod": pos,
                "contract_eod": con,
                "is_roll": is_roll,
            }
        )
    daily = pd.DataFrame(recs)
    if n_missing:
        notes.append(f"{n_missing} days with a missing settlement on the held contract (NaN P&L)")
        log.warning("trading_missing_prices", extra={"fields": {"n": n_missing}})
    summary = trading_metrics(daily, conv.annualisation_days)
    return TradingResult(daily, conv, summary, conv.statement, n_missing, n_nan, notes)
