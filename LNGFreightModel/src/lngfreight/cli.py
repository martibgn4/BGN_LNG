"""Command line entry points.

    python -m lngfreight.cli backtest          walk-forward, the headline table
    python -m lngfreight.cli sanity            leakage checks
    python -m lngfreight.cli lag-sensitivity   re-run one day slower
    python -m lngfreight.cli importance        family attribution
    python -m lngfreight.cli baltic-test      Baltic route, on its own window
    python -m lngfreight.cli predict           today's forecast, all horizons
    python -m lngfreight.cli fetch --refresh   repull every source
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import pickle
import sys

import numpy as np
import pandas as pd

from lngfreight import backtest, dataset, features
from lngfreight.config import get, output_dir
from lngfreight.features import TARGET_PREFIX
from lngfreight.model import FfaForward, build_estimators, own_lags_model

log = logging.getLogger(__name__)

RESULTS_PICKLE = "results.pkl"
SUMMARY_CSV = "backtest_summary.csv"
PREDICTIONS_CSV = "live_predictions.csv"
_RULE_WIDTH = 70

_TABLE_COLUMNS = ["horizon_bd", "model", "n", "rmse", "mae", "hit_rate", "ic",
                  "skill_vs_rw", "skill_vs_own_lags", "skill_vs_ffa",
                  "dm_p_vs_rw", "dm_p_vs_own"]


def _quiet_panel(refresh: bool = False) -> pd.DataFrame:
    """Load the panel, muting the Spark helper's per-release chatter."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return dataset.load_or_build(refresh=refresh)


def _bundle(refresh: bool = False) -> dict[str, pd.DataFrame]:
    panel = _quiet_panel(refresh=refresh)
    wide = dataset.to_wide(panel)
    return features.build(wide, dataset.lags(panel))


def cmd_fetch(args: argparse.Namespace) -> int:
    panel = _quiet_panel(refresh=args.refresh)
    per_series = panel.groupby("series_id").agg(
        n=("value", "size"), first=("obs_date", "min"),
        last=("obs_date", "max"), lag=("publication_lag_days", "first"))
    print(per_series.to_string())
    print(f"\n{len(panel):,} rows across {panel['series_id'].nunique()} series")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    bundle = _bundle(refresh=args.refresh)
    results = backtest.run(bundle)
    table = backtest.summary(results)

    out = output_dir()
    table.to_csv(out / SUMMARY_CSV, index=False)
    with (out / RESULTS_PICKLE).open("wb") as fh:
        pickle.dump(results, fh)

    print(table[_TABLE_COLUMNS].round(4).to_string(index=False))
    print(f"\nwritten: {out / SUMMARY_CSV}")
    _headline(table)
    return 0


def _headline(table: pd.DataFrame) -> None:
    """State the result the way it should be read, not the way it looks best."""
    print("\n" + "=" * _RULE_WIDTH)
    print("HEADLINE - read skill_vs_own_lags, not skill_vs_rw")
    print("=" * _RULE_WIDTH)
    for h in sorted(table["horizon_bd"].unique()):
        block = table[table["horizon_bd"] == h].set_index("model")
        best = block.drop(index=["random_walk", "ffa_forward", "own_lags"],
                          errors="ignore")["skill_vs_own_lags"].idxmax()
        skill = block.loc[best, "skill_vs_own_lags"]
        pval = block.loc[best, "dm_p_vs_own"]
        verdict = "significant" if pval < get(
            "model", "evaluation", "significance", "alpha") else "NOT significant"
        print(f"  h={h:2d}bd  best={best:9s}  skill vs own-lags={skill:+.4f}  "
              f"DM p={pval:.3f}  -> {verdict}")


def cmd_sanity(args: argparse.Namespace) -> int:
    """Two checks that a broken pipeline fails and a correct one passes."""
    bundle = _bundle()
    X, Y, F = bundle["X"], bundle["Y"], bundle["forward"]
    horizon = args.horizon
    column = f"{TARGET_PREFIX}{horizon}"

    print(f"=== shuffled target, h={horizon} - every skill must be ~0 ===")
    rng = np.random.default_rng(args.seed)
    y = Y[column].copy()
    observed = y.dropna()
    values = observed.to_numpy().copy()
    rng.shuffle(values)
    y.loc[observed.index] = values
    shuffled = backtest.run_horizon(X, y, F[column], horizon)
    print(shuffled.metrics[["rmse", "ic", "skill_vs_rw",
                            "skill_vs_own_lags"]].round(4).to_string())

    print(f"\n=== one extra day of lag, h={horizon} - skill must DEGRADE ===")
    delayed = backtest.run_horizon(X.shift(1), Y[column], F[column], horizon)
    print(delayed.metrics[["rmse", "ic", "skill_vs_rw",
                           "skill_vs_own_lags"]].round(4).to_string())
    return 0


def cmd_lag_sensitivity(args: argparse.Namespace) -> int:
    """How much of the result depends on same-day information?

    Re-runs the whole walk-forward with every feature delayed one more
    business day. A model whose skill survives only at zero extra lag is
    racing the tape rather than forecasting, and this prints the difference
    rather than asserting it.
    """
    bundle = _bundle()
    X, Y, F = bundle["X"], bundle["Y"], bundle["forward"]
    rows = []
    for h in get("target", "target", "horizons_bd"):
        h = int(h)
        column = f"{TARGET_PREFIX}{h}"
        for label, matrix in (("as_configured", X), ("plus_1bd", X.shift(1))):
            metrics = backtest.run_horizon(matrix, Y[column], F[column], h).metrics
            for model in metrics.index:
                rows.append({"horizon_bd": h, "lag": label, "model": model,
                             "rmse": metrics.loc[model, "rmse"],
                             "ic": metrics.loc[model, "ic"],
                             "skill_vs_rw": metrics.loc[model, "skill_vs_rw"]})
    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index=["horizon_bd", "model"], columns="lag",
                              values="skill_vs_rw")
    pivot["cost_of_one_day"] = pivot["as_configured"] - pivot["plus_1bd"]
    print(pivot.round(4).to_string())
    return 0


def cmd_baltic_test(args: argparse.Namespace) -> int:
    """Does the Baltic traded route add anything, where its data exists?

    The Baltic generics start 2024-02, two years into the model sample, so a
    whole-sample comparison judges them mostly on rows where they are absent
    and median-imputed. This restricts to the window where they exist and
    varies only the feature set, holding the estimator fixed.
    """
    from lngfreight.model.estimators import ColumnSubsetModel, _ridge

    bundle = _bundle()
    start = get("features", "families", "baltic", "baltic_sample_start")
    X = bundle["X"].loc[start:]
    Y = bundle["Y"].loc[start:]
    F = bundle["forward"].loc[start:]
    # A shorter training window than the main backtest: the whole restricted
    # sample is only ~665 days. Stated rather than hidden - it is why these
    # numbers are not directly comparable to the headline table.
    min_train = args.min_train or int(
        get("features", "families", "baltic", "baltic_test_min_train_bd"))
    step = int(get("model", "walk_forward", "refit_every_bd"))
    embargo = int(get("model", "walk_forward", "embargo_bd"))

    def walk(horizon: int, families: list[str]):
        column = f"{TARGET_PREFIX}{horizon}"
        y, signal = Y[column], F[column]
        usable = (y.dropna().index
                  .intersection(X.dropna(how="all").index)
                  .intersection(signal.replace([np.inf, -np.inf], np.nan)
                                .dropna().index))
        origins = pd.DatetimeIndex(sorted(usable))
        Xo, yo = X.loc[origins], y.loc[origins]
        pred = pd.Series(index=origins, dtype=float)
        for train_end, test in backtest._blocks(origins, min_train, step):
            cutoff = train_end - horizon - embargo
            if cutoff <= 0:
                continue
            train, test_idx = origins[:cutoff], origins[test]
            model = ColumnSubsetModel("subset", _ridge().pipeline, families)
            pred.loc[test_idx] = model.fit(
                Xo.loc[train], yo.loc[train]).predict(Xo.loc[test_idx])
        joined = pd.concat([pred.rename("p"), yo.rename("a")], axis=1).dropna()
        actual, predicted = joined["a"].to_numpy(), joined["p"].to_numpy()
        return actual, predicted

    sets = {"own lags": ["frt"], "own lags + baltic": ["frt", "blt"],
            "baltic alone": ["blt"], "own lags + curve": ["frt", "crv"]}
    rows = []
    for horizon in get("target", "target", "horizons_bd"):
        horizon = int(horizon)
        base_actual, base_pred = walk(horizon, sets["own lags"])
        base_rmse = float(np.sqrt(np.mean((base_actual - base_pred) ** 2)))
        for label, families in sets.items():
            actual, predicted = walk(horizon, families)
            rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
            _, pvalue = backtest.diebold_mariano(actual, predicted, base_pred, horizon)
            rows.append({"horizon_bd": horizon, "feature set": label,
                         "n": len(actual), "rmse": rmse,
                         "skill_vs_own_lags": 1.0 - rmse / base_rmse,
                         "ic": backtest._information_coefficient(actual, predicted),
                         "hit_rate": backtest._hit_rate(actual, predicted),
                         "dm_p": pvalue})
    frame = pd.DataFrame(rows)
    print(f"sample from {start}, min_train={min_train}bd, estimator held fixed")
    print()
    print(frame.round(4).to_string(index=False))
    return 0


def cmd_importance(args: argparse.Namespace) -> int:
    path = output_dir() / RESULTS_PICKLE
    if not path.exists():
        print(f"no {path.name}; run `backtest` first", file=sys.stderr)
        return 1
    with path.open("rb") as fh:
        results = pickle.load(fh)
    table = {f"h={h}": backtest.family_importance(res)
             for h, res in sorted(results.items())}
    print("sum of |standardised ridge coefficient| by feature family\n")
    print(pd.DataFrame(table).round(4).to_string())
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Fit on everything available and forecast forward from the last print.

    This is the live path. It refits on the full history rather than reusing
    a walk-forward fit, and it prints the benchmarks beside the model because
    the backtest says the benchmarks are most of the story.
    """
    bundle = _bundle(refresh=args.refresh)
    X, Y, F = bundle["X"], bundle["Y"], bundle["forward"]
    panel = _quiet_panel()
    spot_series = dataset.to_wide(panel)["atlantic_spot"].dropna()
    spot, asof = float(spot_series.iloc[-1]), spot_series.index[-1]

    rows = []
    for h in get("target", "target", "horizons_bd"):
        h = int(h)
        column = f"{TARGET_PREFIX}{h}"
        y = Y[column]
        usable = (y.dropna().index
                  .intersection(X.dropna(how="all").index)
                  .intersection(F[column].dropna().index))
        X_train, y_train = X.loc[usable], y.loc[usable]
        latest = X.loc[[X.index[-1]]]

        models = build_estimators()
        own = own_lags_model()
        if own is not None:
            models = models + [own]
        for model in models:
            pred = float(model.fit(X_train, y_train).predict(latest)[0])
            rows.append({"horizon_bd": h, "model": model.name,
                         "log_change": pred, "level_usd_day": spot * np.exp(pred)})
        forward = FfaForward().fit(F[column].loc[usable], y_train)
        pred = float(forward.predict(F[column].loc[[X.index[-1]]])[0])
        rows.append({"horizon_bd": h, "model": "ffa_forward",
                     "log_change": pred, "level_usd_day": spot * np.exp(pred)})
        rows.append({"horizon_bd": h, "model": "random_walk",
                     "log_change": 0.0, "level_usd_day": spot})

    frame = pd.DataFrame(rows)
    frame["as_of"] = asof.date()
    frame["spot_now"] = spot
    frame.to_csv(output_dir() / PREDICTIONS_CSV, index=False)
    print(f"Spark30S as of {asof.date()}: {spot:,.0f} USD/day\n")
    print(frame.pivot_table(index="model", columns="horizon_bd",
                            values="level_usd_day").round(0).to_string())
    print("\nThese are point forecasts from a model whose out-of-sample edge "
          "over its own lags is not statistically significant. Read the README "
          "before trading on them.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lngfreight")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="pull and cache every source")
    fetch.add_argument("--refresh", action="store_true")
    fetch.set_defaults(func=cmd_fetch)

    bt = sub.add_parser("backtest", help="walk-forward evaluation")
    bt.add_argument("--refresh", action="store_true")
    bt.set_defaults(func=cmd_backtest)

    sanity = sub.add_parser("sanity", help="leakage checks")
    sanity.add_argument("--horizon", type=int, default=5)
    sanity.add_argument("--seed", type=int, default=0)
    sanity.set_defaults(func=cmd_sanity)

    lag = sub.add_parser("lag-sensitivity", help="re-run one day slower")
    lag.set_defaults(func=cmd_lag_sensitivity)

    baltic = sub.add_parser("baltic-test",
                            help="does the Baltic traded route add anything")
    baltic.add_argument("--min-train", dest="min_train", type=int, default=None,
                        help="override the training window from features.yaml")
    baltic.set_defaults(func=cmd_baltic_test)

    imp = sub.add_parser("importance", help="family attribution")
    imp.set_defaults(func=cmd_importance)

    pred = sub.add_parser("predict", help="today's forecast")
    pred.add_argument("--refresh", action="store_true")
    pred.set_defaults(func=cmd_predict)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
