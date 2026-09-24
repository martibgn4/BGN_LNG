"""Generate this week's picks and the HTML report.

    python -m StockPicker.scripts.run_weekly              # refresh data + pick
    python -m StockPicker.scripts.run_weekly --no-refresh # use cached data

Writes ``StockPicker/reports/weekly_picks_<date>.html`` and appends the book to
``StockPicker/cache/live_picks.parquet`` so a genuine out-of-sample track
record accumulates from the first run onward.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

import pandas as pd

from ..backtest.engine import benchmark_returns, run_backtest
from ..backtest.metrics import summary
from ..bbg.session import BloombergSession
from ..bbg.universe import all_securities, build_membership, fetch_metadata
from ..config import (
    BENCHMARK_TICKER,
    CACHE,
    ESTIMATES_DIR,
    ESTIMATE_FIELDS,
    ESTIMATE_OVERRIDES,
    FUNDA_DIR,
    HISTORY_START,
    INDEX_TICKER,
    MODEL,
    OPTIONS_DIR,
    OPTIONS_FIELDS,
    PRICES_DIR,
    PRICE_FIELDS,
    SECTOR_COL,
    SLOW_FIELDS,
    STRATEGY,
)
from ..data.panel import apply_filters, build_panel
from ..data.store import fetch_series
from ..features.crosssec import standardize_panel
from ..features.registry import all_features
from ..model.combine import build_family_scores, ic_weighted_score, rolling_ic_weights
from ..model.portfolio import build_book
from ..report.weekly import build_report, save_report

LIVE_PICKS = CACHE / "live_picks.parquet"
log = logging.getLogger("weekly")


def refresh_data() -> None:
    """Top the cache up to today before scoring."""
    end = pd.Timestamp.today().normalize()
    with BloombergSession(timeout_ms=60_000) as bbg:
        membership = build_membership(bbg, start=HISTORY_START)
        secs = all_securities(membership)
        fetch_metadata(bbg, secs)
        fetch_series(bbg, [BENCHMARK_TICKER, INDEX_TICKER], PRICE_FIELDS,
                     HISTORY_START, end, PRICES_DIR, kind="price")
        fetch_series(bbg, secs, PRICE_FIELDS, HISTORY_START, end,
                     PRICES_DIR, kind="price", batch_size=25)
        fetch_series(bbg, secs, SLOW_FIELDS, HISTORY_START, end, FUNDA_DIR,
                     kind="funda", periodicity="WEEKLY", batch_size=15,
                     scale_market_cap=False)
        # Estimates need the forward-period override to return real history.
        fetch_series(bbg, secs, ESTIMATE_FIELDS, HISTORY_START, end, ESTIMATES_DIR,
                     kind="estimates", periodicity="WEEKLY", batch_size=20,
                     scale_market_cap=False, overrides=ESTIMATE_OVERRIDES)
        fetch_series(bbg, secs, OPTIONS_FIELDS, "2014-01-01", end, OPTIONS_DIR,
                     kind="options", periodicity="WEEKLY", batch_size=15,
                     scale_market_cap=False)
        log.info("Bloomberg usage: %s", bbg.stats.summary())


def load_previous_book() -> pd.DataFrame:
    if not LIVE_PICKS.exists():
        return pd.DataFrame()
    picks = pd.read_parquet(LIVE_PICKS)
    if picks.empty:
        return picks
    picks["date"] = pd.to_datetime(picks["date"])
    return picks[picks["date"] == picks["date"].max()]


def append_live_picks(book: pd.DataFrame, asof) -> None:
    rec = book.copy()
    rec["date"] = pd.Timestamp(asof)
    cols = [c for c in ["date", "security", "NAME", SECTOR_COL, "weight", "raw_weight",
                        "score", "vol_60", "beta", "vol_scalar", "cash_weight"] if c in rec.columns]
    rec = rec[cols]
    if LIVE_PICKS.exists():
        old = pd.read_parquet(LIVE_PICKS)
        old["date"] = pd.to_datetime(old["date"])
        rec = pd.concat([old[old["date"] != pd.Timestamp(asof)], rec], ignore_index=True)
    rec.to_parquet(LIVE_PICKS, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="skip the Bloomberg refresh")
    ap.add_argument("--rebuild", action="store_true", help="reassemble the panel")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--asof", default=None, help="override the rebalance date (YYYY-MM-DD)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    if not args.no_refresh:
        log.info("Refreshing Bloomberg cache ...")
        refresh_data()

    panel = build_panel() if args.rebuild else None
    if panel is None:
        cached = CACHE / "panel.parquet"
        if cached.exists() and not args.rebuild:
            panel = pd.read_parquet(cached)
            panel["date"] = pd.to_datetime(panel["date"])
        else:
            panel = build_panel()
            panel.to_parquet(cached, index=False)

    panel = apply_filters(panel)
    feats = [f for f in all_features() if f in panel.columns]
    panel = standardize_panel(
        panel, feats,
        sector_col=SECTOR_COL if MODEL.neutralize_sector else None,
        method="rank", winsor_pct=MODEL.winsorize_pct,
    )
    # Blend at family level -- identical to run_research, so the live picks and
    # the backtested strategy are the same model rather than two that have
    # quietly drifted apart.
    panel, fam_cols = build_family_scores(panel)
    weights = rolling_ic_weights(panel, fam_cols, halflife=MODEL.ic_halflife_weeks)
    panel["score"] = ic_weighted_score(panel, fam_cols, weights)

    # ---- the live cross-section ------------------------------------------
    asof = pd.Timestamp(args.asof) if args.asof else panel["date"].max()
    cs = panel[panel["date"] == asof].copy()
    log.info("Scoring cross-section for %s: %s names", asof.date(), len(cs))
    if cs["score"].notna().sum() < 20:
        log.error("Too few scored names on %s -- is the cache current?", asof.date())
        return 1

    prev = load_previous_book()
    book = build_book(
        cs, score_col="score",
        incumbents=list(prev["security"]) if not prev.empty else None,
        n=args.n or STRATEGY.n_positions,
    )

    # ---- backtest context for the report ---------------------------------
    hist = panel[panel["date"] < asof].dropna(subset=["score", "fwd_ret_1w"])
    perf = None
    if hist["date"].nunique() > 60:
        bt = run_backtest(hist, score_col="score", n_positions=args.n or STRATEGY.n_positions)
        perf = summary(bt["returns"]["net_ret"], benchmark=benchmark_returns(hist))
        perf["cost_bps"] = STRATEGY.cost_bps

    notes = [
        "Universe is point-in-time S&P 500 membership; the backtest includes the "
        "456 names that have left the index since 2006, so it is free of survivorship bias.",
        "Scores use only data available before the rebalance; execution is modelled at VWAP.",
        f"Costs charged at {STRATEGY.cost_bps:.0f}bps on turnover.",
        "A five-name book carries large idiosyncratic risk. Position sizes are inverse-volatility "
        "weighted and capped, but single-stock event risk (earnings, M&A, fraud) is not diversified away.",
    ]

    html_str = build_report(
        book, asof=asof,
        previous_book=prev if not prev.empty else None,
        performance=perf,
        universe_size=len(cs),
        notes=notes,
    )
    path = save_report(html_str, asof)
    append_live_picks(book, asof)

    print("\n" + "=" * 78)
    print(f"WEEKLY PICKS -- {asof:%d-%b-%Y}")
    print("=" * 78)
    for _, r in book.iterrows():
        print(f"  {r['security'].replace(' US Equity',''):8s} "
              f"{str(r.get('NAME',''))[:34]:34s} "
              f"w={r['weight']*100:5.1f}%  score={r['score']:+.3f}  "
              f"{str(r.get(SECTOR_COL,''))[:22]}")
    print(f"\n  Cash: {book['cash_weight'].iloc[0]*100:.1f}%   "
          f"vol-scalar: {book['vol_scalar'].iloc[0]:.2f}")
    print(f"\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
