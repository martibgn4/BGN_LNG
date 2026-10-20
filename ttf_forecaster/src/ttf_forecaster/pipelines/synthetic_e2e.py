"""Synthetic end-to-end run: ingestion -> normalisation -> PIT features -> targets -> walk-forward
-> metrics -> report.

**Verifies architecture only.** All data come from ``data/synthetic.py`` and
``data/fundamentals.synthetic_fundamentals``; every artefact is written with ``is_synthetic=True``
and the report carries a SYNTHETIC banner. Two worlds are run so the harness is tested both ways:

* ``null``    prices independent of fundamentals -> a correct pipeline finds no skill;
* ``planted`` lagged storage surprise drives returns -> a correct pipeline finds it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    DataBundle,
    empty_frame,
)
from ttf_forecaster.data.fundamentals import (
    SyntheticFundamentalsConfig,
    load_fundamentals_registry,
    synthetic_fundamentals,
)
from ttf_forecaster.data.storage import ParquetStore
from ttf_forecaster.evaluation.backtest import TradingConventions, simulate_trading, walk_forward
from ttf_forecaster.evaluation.diagnostics import delay_targets
from ttf_forecaster.evaluation.metrics import add_group_columns, paired_frame, point_metrics_by
from ttf_forecaster.evaluation.splits import SplitConfig, make_splits
from ttf_forecaster.evaluation.statistical_tests import (
    clark_west,
    diebold_mariano,
    evaluate_success_criteria,
)
from ttf_forecaster.features.registry import feature_hash
from ttf_forecaster.logging_config import get_logger
from ttf_forecaster.models.baselines import make_target_lags
from ttf_forecaster.models.interface import make_model
from ttf_forecaster.models.regimes import load_regime_config, regime_label_series
from ttf_forecaster.pipelines.build_features import build_feature_matrix, load_feature_config
from ttf_forecaster.pipelines.ingest import ingest
from ttf_forecaster.settings import Paths, load_config
from ttf_forecaster.targets import futures_targets
from ttf_forecaster.targets.build import CurveContext
from ttf_forecaster.targets.common import session_horizon

log = get_logger(__name__)


@dataclass
class WorldResult:
    """Everything the report needs for one synthetic world."""

    world: str
    run_id: str
    n_origins: int
    n_features: int
    feature_columns: list[str]
    criteria: dict[str, dict[str, Any]] = field(default_factory=dict)
    criteria_vs_best: dict[str, dict[str, Any]] = field(default_factory=dict)
    # D-032 F7 / amendment A1: criteria 1-2 re-scored on the executable label r(t+1, t+1+h)
    criteria_exec: dict[str, dict[str, Any]] = field(default_factory=dict)
    price_only_test_exec: dict[str, Any] = field(default_factory=dict)
    # A pass requires criteria 1 and 2 on BOTH the fixed-contract and the executable label
    headline_pass: dict[str, bool] = field(default_factory=dict)
    best_benchmark: str = ""
    clean_pit: bool = False
    year_metrics: dict[str, pd.DataFrame] = field(default_factory=dict)
    regime_metrics: dict[str, pd.DataFrame] = field(default_factory=dict)
    fold_metrics: dict[str, pd.DataFrame] = field(default_factory=dict)
    price_only_test: dict[str, Any] = field(default_factory=dict)
    trading: dict[str, dict[str, Any]] = field(default_factory=dict)
    ingest_summaries: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)


def _ingest_synthetic(store: ParquetStore, frames: dict[str, pd.DataFrame],
                      ingestion_time: pd.Timestamp) -> list[str]:
    """Push each synthetic frame through the real ingest path (raw -> checks -> normalised)."""
    out = []
    for kind, frame in frames.items():
        summary = ingest(lambda f=frame: f, store, kind=kind, source="synthetic",
                         raw_name=f"synthetic_{kind}", ingestion_time=ingestion_time,
                         keep_adapter_ingestion_time=True, fail_on_quality=False)
        out.append(f"{kind}: {summary}")
    return out


def _bundle_from_store(store: ParquetStore) -> DataBundle:
    """Read the normalised layer back: the rest of the pipeline sees only stored data."""
    return DataBundle(
        observations=store.read_normalised("observation"),
        forecasts=store.read_normalised("forecast"),
        curves=store.read_normalised("curve"),
        contracts=empty_frame(CONTRACT_METADATA_SCHEMA),
    )


def _select_columns(x: pd.DataFrame, ev: dict[str, Any], before: pd.Timestamp) -> pd.DataFrame:
    """Drop excluded prefixes and near-empty columns. Coverage is measured only on rows before
    ``before`` (the first test origin), so no test-period information chooses the feature set."""
    coverage = x.loc[x.index < before].notna().mean()
    keep = [c for c in coverage.index[coverage > float(ev["min_feature_coverage"])]
            if not c.startswith(tuple(ev["exclude_prefixes"]))]
    out = x[keep].copy()
    out.attrs = {"pit_grade": {c: g for c, g in x.attrs.get("pit_grade", {}).items()
                               if c in keep}}
    return out


def _regime_codes(bundle: DataBundle, config_dir: str | None) -> pd.Series:
    """PIT rule-regime label per origin timestamp (evaluation grouping only, never a feature)."""
    lab = regime_label_series(bundle, load_regime_config(config_dir=config_dir),
                              include_markov=False, include_changepoint=False)
    col = "label" if "label" in lab.columns else "code"
    return lab[col]


def _executable_predictions(predictions: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """The same forecasts scored against the executable label r(t+1, t+1+h) (D-032 F7).

    A forecast made at origin t is traded at the settlement of t+1 (D-021), so the label it is
    judged on is delay_targets(targets, 1). Models are not re-fitted: the forecast of origin t
    uses no information after t, so only the actual changes.
    """
    exe = delay_targets(targets, 1).set_index("origin_time")["value"]
    out = predictions.copy()
    out["actual"] = out["origin_time"].map(exe)
    return out


def evaluate_bundle(bundle: DataBundle, *, label: str, run_id: str, store: ParquetStore,
                    is_synthetic: bool, config_dir: str | None = None,
                    first_origin: str | None = None, horizon: int | None = None,
                    summaries: list[str] | None = None,
                    timings: dict[str, float] | None = None) -> WorldResult:
    """Steps 3-7 of the pipeline on any stored bundle (synthetic or real; D-036).

    Targets on fixed held contracts (vendor expiries from ``bundle.contracts`` when present),
    point-in-time features, purged walk-forward vs the benchmark set and the price-only twin,
    criteria on both the fixed and the executable label, trading at the configured cost
    multipliers, immutable artefacts. ``is_synthetic`` is carried into every artefact.
    """
    cfg = load_config("pipeline", config_dir)
    tgt, ev = cfg["target"], cfg["evaluation"]
    h = int(horizon or tgt["horizon"])
    split_cfg = SplitConfig.from_config(load_config("backtests", config_dir))
    timings = dict(timings or {})
    summaries = list(summaries or [])
    t0 = time.perf_counter() - sum(timings.values())

    # 3. Targets (fixed held contract; label_known_time drives purging) -----------------------
    meta = bundle.contracts if len(bundle.contracts) else None  # vendor expiries (D-035)
    ctx = CurveContext.from_curve(bundle.curves, meta, config_dir=config_dir)
    builder = getattr(futures_targets, f"build_{tgt['builder']}")
    targets = builder(ctx, [h])
    # D-032 F1: the model horizon is the label overlap + 1 (sessions); a month count is refused.
    h = session_horizon(targets, declared=h)
    first = pd.Timestamp(first_origin or ev["first_origin"], tz="UTC")
    # AR benchmark lags: built on the full contiguous origin grid BEFORE trimming.
    y_full = targets.set_index("origin_time")["value"]
    lags = make_target_lags(y_full, horizon=h, p=int(ev["ar_lags"]))
    targets = targets.loc[targets["origin_time"] >= first].reset_index(drop=True)
    origins = pd.DatetimeIndex(targets["origin_time"].unique())
    timings["targets"] = time.perf_counter() - t0 - sum(timings.values())

    # 4. Point-in-time features ----------------------------------------------------------------
    fc = load_feature_config(None if config_dir is None else Path(config_dir) / "features.yaml")
    if config_dir is not None:  # the roll-splice provider reads its calendar and roll from here
        fc["families"].setdefault("price", {}).setdefault("params", {})["config_dir"] = config_dir
    for fam in ev["feature_families"]:
        fc["families"].setdefault(fam, {})["enabled"] = True
    x_full = build_feature_matrix(bundle, origins, fc, families=ev["feature_families"])
    grades = dict(x_full.attrs.get("pit_grade", {}))
    x_full = x_full.join(lags.reindex(x_full.index))
    grades.update({c: "A" for c in lags.columns})  # built from settlements (grade A)
    x_full.attrs = {"pit_grade": grades}
    first_test = make_splits(targets, horizon=h, cfg=split_cfg,
                             valid_origins=x_full.index)[0].test_start
    x = _select_columns(x_full, ev, first_test)
    grade_values = list(x.attrs["pit_grade"].values())
    clean_pit = bool(grade_values) and all(g in ("A", "B") for g in grade_values)
    price_cols = [c for c in x.columns if c.split("__")[0] in ev["price_only_families"]]
    timings["features"] = time.perf_counter() - t0 - sum(timings.values())

    # 5. Walk-forward ------------------------------------------------------------------------
    folds = make_splits(targets, horizon=h, cfg=split_cfg, valid_origins=x.index)
    lag_cols = list(lags.columns)
    bench = {}
    for name, spec in ev["benchmarks"].items():
        extra = {"lag_columns": lag_cols} if spec["model"] == "ar" else {}
        bench[name] = (lambda n=spec["model"], e=extra: make_model(
            n, horizon=h, config_dir=config_dir, **e))
    fhash = feature_hash(list(x.columns), {"world": label, "families": ev["feature_families"]})
    preds = []
    for i, name in enumerate(ev["models"]):
        res = walk_forward(lambda n=name: make_model(n, horizon=h, config_dir=config_dir), x,
                           targets, folds,
                           bench if i == 0 else {}, model_id=name, feature_hash_=fhash,
                           is_synthetic=is_synthetic, dropna_features=False)
        preds.append(res.predictions)
    # Price-only twin of the first model (spec §8 criterion 3: do fundamentals add anything?)
    po_name = f"{ev['models'][0]}_price_only"
    po_hash = feature_hash(price_cols, {"world": label, "families": ev["price_only_families"]})
    res_po = walk_forward(lambda: make_model(ev["models"][0], horizon=h, config_dir=config_dir),
                          x[price_cols], targets,
                          folds, {}, model_id=po_name, feature_hash_=po_hash,
                          is_synthetic=is_synthetic, dropna_features=False)
    preds.append(res_po.predictions)
    predictions = pd.concat(preds, ignore_index=True)
    timings["walk_forward"] = time.perf_counter() - t0 - sum(timings.values())

    # 6. Metrics, tests, trading ----------------------------------------------------------------
    principal = ev["principal_benchmark"]
    result = WorldResult(world=label, run_id=run_id, n_origins=len(x), n_features=x.shape[1],
                         feature_columns=list(x.columns), ingest_summaries=summaries,
                         clean_pit=clean_pit)
    # Best benchmark = lowest pooled OOS MSE in the benchmark set. Chosen ex post, which favours
    # the benchmark and is therefore conservative against the model.
    bench_preds = predictions.loc[predictions["model_id"].isin(list(ev["benchmarks"]))]
    mse = {b: float(((g["actual"] - g["prediction"]) ** 2).mean())
           for b, g in bench_preds.groupby("model_id")}
    result.best_benchmark = min(mse, key=lambda b: mse[b])
    regimes = _regime_codes(bundle, config_dir) if ev.get("regime_breakdown") else None
    exe_predictions = _executable_predictions(predictions, targets)
    for m in [*ev["models"], po_name, *[b for b in ev["benchmarks"] if b != principal]]:
        paired = paired_frame(predictions, m, principal)
        result.criteria[m] = evaluate_success_criteria(paired, h, clean_pit=clean_pit)
        exe_paired = paired_frame(exe_predictions, m, principal).dropna(subset=["actual"])
        result.criteria_exec[m] = evaluate_success_criteria(exe_paired, h, clean_pit=clean_pit)
        if m in ev["models"]:
            c_zero, c_exe = result.criteria[m], result.criteria_exec[m]
            result.headline_pass[m] = bool(
                c_zero["criterion_1"]["passed"] and c_exe["criterion_1"]["passed"]
                and c_zero["criterion_2"]["passed"] and c_exe["criterion_2"]["passed"])
        result.fold_metrics[m] = point_metrics_by(paired, "fold_id")
        grouped = add_group_columns(paired, regimes)
        result.year_metrics[m] = point_metrics_by(grouped, "year")
        if regimes is not None:
            result.regime_metrics[m] = point_metrics_by(grouped.dropna(subset=["regime"]),
                                                        "regime")
        if m in (*ev["models"], po_name) and result.best_benchmark not in (principal, m):
            result.criteria_vs_best[m] = evaluate_success_criteria(
                paired_frame(predictions, m, result.best_benchmark), h, clean_pit=clean_pit)
    def _price_only(frame: pd.DataFrame) -> dict[str, Any]:
        # model = full feature set, bench = price-only twin (nested: price-only is the small model)
        return {
            "dm": diebold_mariano(frame["actual"] - frame["model"],
                                  frame["actual"] - frame["bench"], h),
            "cw": clark_west(frame["actual"], frame["bench"], frame["model"], h),
        }

    full_vs_po = paired_frame(predictions, ev["models"][0], po_name)
    result.price_only_test = _price_only(full_vs_po)
    exe_vs_po = paired_frame(exe_predictions, ev["models"][0], po_name).dropna(subset=["actual"])
    result.price_only_test_exec = _price_only(exe_vs_po)
    conv = TradingConventions.from_config(load_config("backtests", config_dir)).with_(
        rebalance_every=h)
    for m in ev["models"]:
        fc_series = predictions.loc[predictions["model_id"] == m].set_index("origin_time")[
            "prediction"]
        for mult in ev["cost_multipliers"]:
            tr = simulate_trading(fc_series, targets, bundle.curves,
                                  conv.with_(cost_multiplier=float(mult)))
            result.trading[f"{m}@{mult}x"] = tr.summary
    timings["metrics"] = time.perf_counter() - t0 - sum(timings.values())

    # 7. Persist artefacts (immutable run; synthetic flag carried into the manifest) -------
    # D-031: the manifest hash covers every resolved config, not only the evaluation block.
    full_cfg = {"world": label, "pipeline": cfg, "features": fc,
                **{n: load_config(n, config_dir) for n in
                   ("models", "backtests", "targets", "instruments", "regimes")}}
    store.write_run_artifact("features", run_id, "feature_matrix", x.reset_index(),
                             is_synthetic=is_synthetic, config=full_cfg)
    store.write_run_artifact("predictions", run_id, "predictions", predictions,
                             is_synthetic=is_synthetic, config=full_cfg)
    result.timings = timings
    log.info("evaluation_done", extra={"fields": {"world": label, "run_id": run_id,
                                                       **{k: round(v, 1) for k, v in
                                                          timings.items()}}})
    return result


def run_world(world: str, *, data_root: str | None = None, config_dir: str | None = None,
              run_tag: str | None = None) -> WorldResult:
    """Run the full synthetic pipeline for one world and persist artefacts (SYNTHETIC).

    config_dir (SQ-04) is threaded through every loader used here: pipeline, features, models,
    backtests (splits, trading), targets (roll, calendar), instruments, regimes and fundamentals.
    """
    cfg = load_config("pipeline", config_dir)
    syn, tgt, ev = cfg["synthetic"], cfg["target"], cfg["evaluation"]
    h = int(tgt["horizon"])
    split_cfg = SplitConfig.from_config(load_config("backtests", config_dir))
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    # 1. Generate (fake vendor payloads) ------------------------------------------------------
    reg = load_fundamentals_registry(config_dir)
    fcfg = SyntheticFundamentalsConfig(start=syn["start"], end=syn["end"], seed=int(syn["seed"]))
    obs = synthetic_fundamentals(reg, fcfg)
    scfg = synthetic.SyntheticConfig(start=syn["start"], end=syn["end"], seed=int(syn["seed"]),
                                     world=world, contract_noise=float(syn["contract_noise"]),
                                     planted_beta=float(syn["planted_beta"]),
                                     planted_driver=str(syn.get("planted_driver", "level")))
    curve = synthetic.ttf_curve(scfg, storage=obs)
    weather = synthetic.weather_forecast_runs(scfg)
    timings["generate"] = time.perf_counter() - t0

    # 2. Ingest + normalise through the store, then read back ----------------------------------
    run_id = f"synthetic_{world}_{run_tag or pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%S')}"
    paths = Paths.from_env(data_root)
    store = ParquetStore(paths.root / "synthetic_runs" / run_id) if data_root is None else \
        ParquetStore(paths.root)
    ingest_t = pd.Timestamp(syn["end"], tz="UTC") + pd.Timedelta(days=30)
    summaries = _ingest_synthetic(store, {"observation": obs, "forecast": weather, "curve": curve},
                                  ingest_t)
    bundle = _bundle_from_store(store)
    timings["ingest"] = time.perf_counter() - t0 - sum(timings.values())

    return evaluate_bundle(bundle, label=world, run_id=run_id, store=store, is_synthetic=True,
                           config_dir=config_dir, summaries=summaries, timings=timings)
