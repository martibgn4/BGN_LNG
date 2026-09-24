"""Full research pipeline: panel -> features -> IC study -> backtest.

    python -m StockPicker.scripts.run_research
    python -m StockPicker.scripts.run_research --blend lightgbm --n 5

Caches the assembled panel so subsequent runs are fast; pass --rebuild to
force reassembly after changing feature code.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from ..backtest.engine import benchmark_returns, deciles, run_backtest
from ..backtest.metrics import format_summary, regime_breakdown, summary, yearly_table
from ..config import CACHE, MODEL, SECTOR_COL, STRATEGY
from ..data.panel import apply_filters, build_panel
from ..features.crosssec import standardize_panel
from ..features.registry import all_features, expected_signs, family_of
from ..model.combine import (
    build_family_scores,
    feature_report,
    ic_weighted_score,
    rolling_ic_weights,
    walk_forward_lgbm,
)

PANEL_PATH = CACHE / "panel.parquet"
SCORED_PATH = CACHE / "panel_scored.parquet"
log = logging.getLogger("research")


def get_panel(rebuild: bool = False, start=None, end=None) -> pd.DataFrame:
    if PANEL_PATH.exists() and not rebuild:
        log.info("Loading cached panel from %s", PANEL_PATH)
        p = pd.read_parquet(PANEL_PATH)
        p["date"] = pd.to_datetime(p["date"])
        return p
    panel = build_panel(start=start, end=end)
    panel.to_parquet(PANEL_PATH, index=False)
    log.info("Panel cached to %s", PANEL_PATH)
    return panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="reassemble the panel from cache")
    ap.add_argument("--blend", default="ic_weighted", choices=["ic_weighted", "lightgbm", "both"])
    ap.add_argument("--n", type=int, default=None, help="positions (default from config)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--no-vol-target", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    # ---------------- panel ----------------------------------------------
    panel = get_panel(rebuild=args.rebuild, start=args.start, end=args.end)
    panel = apply_filters(panel)
    log.info("Investable panel: %s rows | %s dates | %s names",
             len(panel), panel["date"].nunique(), panel["security"].nunique())

    feats = [f for f in all_features() if f in panel.columns]
    missing = [f for f in all_features() if f not in panel.columns]
    if missing:
        log.warning("Features absent from panel (%s): %s", len(missing), ", ".join(missing))

    # ---------------- standardise ----------------------------------------
    log.info("Standardising %s features cross-sectionally (sector-neutral=%s) ...",
             len(feats), MODEL.neutralize_sector)
    panel = standardize_panel(
        panel, feats,
        sector_col=SECTOR_COL if MODEL.neutralize_sector else None,
        method="rank", winsor_pct=MODEL.winsorize_pct,
    )
    zcols = [f"{f}_z" for f in feats]

    # ---------------- feature diagnostics --------------------------------
    log.info("Computing per-feature information coefficients ...")
    fr = feature_report(panel, zcols)
    fr.index = [i[:-2] for i in fr.index]
    fr["family"] = [family_of(i) for i in fr.index]
    signs = expected_signs()
    fr["expected"] = [signs.get(i, 0) for i in fr.index]
    fr["sign_ok"] = [
        (np.sign(r.mean_ic) == r.expected) or r.expected == 0
        for r in fr.itertuples()
    ]

    print("\n" + "=" * 108)
    print("FEATURE INFORMATION COEFFICIENTS  (full sample, in-sample diagnostic only)")
    print("=" * 108)
    show = fr[["family", "mean_ic", "ic_ir", "t_stat", "hit_rate",
               "ic_first_half", "ic_second_half", "sign_stable", "expected", "sign_ok", "n"]]
    print(show.round(4).to_string())

    strong = fr[(fr["t_stat"].abs() > 2) & fr["sign_stable"]]
    print(f"\n{len(strong)} of {len(fr)} features have |t| > 2 AND a stable sign across halves.")

    # ---------------- family aggregation ----------------------------------
    # Blend at the level of *ideas*, not individual columns. See
    # model.combine.build_family_scores for why this matters.
    panel, fam_cols = build_family_scores(panel)

    print("\n--- Family score ICs ---")
    fam_ic = feature_report(panel, fam_cols)
    print(fam_ic[["mean_ic", "ic_ir", "t_stat", "hit_rate",
                  "ic_first_half", "ic_second_half", "sign_stable", "n"]].round(4).to_string())

    # ---------------- scores ---------------------------------------------
    results = {}
    blends = ["ic_weighted", "lightgbm"] if args.blend == "both" else [args.blend]
    model_cols = fam_cols

    for blend in blends:
        log.info("--- blending: %s ---", blend)
        if blend == "ic_weighted":
            w = rolling_ic_weights(panel, model_cols, halflife=MODEL.ic_halflife_weeks)
            panel["score"] = ic_weighted_score(panel, model_cols, w)
            results["ic_weights"] = w
            print("\n--- Mean IC-derived blend weight per family (final date) ---")
            print(w.iloc[-1].sort_values(key=abs, ascending=False).round(4).to_string())
        else:
            preds, imp = walk_forward_lgbm(panel, model_cols)
            panel["score"] = preds
            results["lgbm_importance"] = imp

        scored = panel.dropna(subset=["score"])
        log.info("Scored rows: %s (%s dates)", len(scored), scored["date"].nunique())

        # decile monotonicity
        dec = deciles(scored, "score")
        print(f"\n--- Forward-return by score decile [{blend}] ---")
        print(dec.round(5).to_string())

        # backtest
        bt = run_backtest(
            scored, score_col="score",
            n_positions=args.n or STRATEGY.n_positions,
            apply_vol_target=not args.no_vol_target,
        )
        bench = benchmark_returns(scored)
        s = summary(bt["returns"]["net_ret"], benchmark=bench)
        print("\n" + format_summary(s, f"STRATEGY [{blend}] -- {args.n or STRATEGY.n_positions} names, net of costs"))

        gross_s = summary(bt["returns"]["gross_ret"], benchmark=bench)
        print(f"\n  (gross of costs: CAGR {gross_s.get('cagr', float('nan'))*100:.2f}%, "
              f"Sharpe {gross_s.get('sharpe', float('nan')):.2f}) "
              f"-- cost drag {(gross_s.get('cagr',0)-s.get('cagr',0))*100:.2f}%/yr")

        print("\n--- Calendar years ---")
        print((yearly_table(bt["returns"]["net_ret"], bench) * 100).round(2).to_string())

        print("\n--- Regime breakdown ---")
        rb = regime_breakdown(bt["returns"]["net_ret"], bench)
        if not rb.empty:
            print(rb.round(4).to_string())

        results[blend] = {"backtest": bt, "summary": s, "deciles": dec}

        # concentration sensitivity: does 5 names cost us risk-adjusted return?
        print("\n--- Concentration sensitivity (net) ---")
        rows = []
        for n in (3, 5, 10, 20, 30):
            b = run_backtest(scored, score_col="score", n_positions=n,
                             apply_vol_target=not args.no_vol_target, verbose=False)
            ss = summary(b["returns"]["net_ret"], benchmark=bench)
            rows.append({
                "n_positions": n, "cagr": ss.get("cagr"), "vol": ss.get("ann_vol"),
                "sharpe": ss.get("sharpe"), "max_dd": ss.get("max_drawdown"),
                "turnover": b["returns"]["turnover"].mean(),
            })
        print(pd.DataFrame(rows).set_index("n_positions").round(4).to_string())

    # ---------------- structural validation -------------------------------
    from ..backtest.validate import lag_sensitivity, run_all, shuffle_test

    run_all(panel, zcols, score_col="score")

    scored = panel.dropna(subset=["score"])
    print("\n--- Shuffle test (scores permuted within date; edge must vanish) ---")
    print(shuffle_test(scored, n_trials=3, n_positions=args.n or STRATEGY.n_positions).round(4).to_string(index=False))

    print("\n--- Lag sensitivity (extra weeks of delay before acting) ---")
    print(lag_sensitivity(scored, n_positions=args.n or STRATEGY.n_positions).round(4).to_string(index=False))

    panel.to_parquet(SCORED_PATH, index=False)
    log.info("Scored panel saved to %s", SCORED_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
