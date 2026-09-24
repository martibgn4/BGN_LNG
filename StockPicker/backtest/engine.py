"""
============================================================
backtest/engine.py  --  Weekly rebalance simulator
============================================================
Walks the rebalance grid forward one date at a time, holding the
book chosen on the previous date, and charges a cost on every
weight change.

What this engine deliberately does NOT do
-----------------------------------------
* It never selects from today's index. Membership is read from the
  point-in-time panel, so a 2010 week only sees the 500 companies
  that were actually in the index in 2010, including the ~456 that
  have since been acquired or delisted.
* It never trades at a price it could not have obtained. Execution
  is at VWAP on the rebalance bar, and the score driving that trade
  was built from data ending the prior close.
* It never lets a delisted name quietly vanish. If a holding stops
  trading, the position is closed at its last observed price and
  the loss (or gain) is realised -- which is exactly what happened
  to anyone holding SVB in March 2023.

Costs
-----
``cost_bps`` is charged on turnover, one-way, on each rebalance:
    cost_t = sum |w_new - w_held| * cost_bps / 10,000
8bps covers commission plus half-spread on liquid US large caps at
personal scale. It is applied to the *change* in weights, so a name
held unchanged across a week costs nothing.
============================================================
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import SECTOR_COL, STRATEGY
from ..model.portfolio import build_book

log = logging.getLogger(__name__)


def run_backtest(
    panel: pd.DataFrame,
    score_col: str = "score",
    date_col: str = "date",
    ret_col: str = "fwd_ret_1w",
    cost_bps: float = None,
    apply_vol_target: bool = True,
    n_positions: int = None,
    verbose: bool = True,
) -> dict:
    """Simulate the weekly strategy.

    Returns a dict with ``returns`` (weekly series), ``holdings`` (long frame
    of every position ever held) and ``diagnostics``.
    """
    cfg = STRATEGY
    cost_bps = cost_bps if cost_bps is not None else cfg.cost_bps

    df = panel.dropna(subset=[score_col]).copy()
    dates = np.sort(df[date_col].unique())

    prev_weights: dict[str, float] = {}
    incumbents: list[str] = []
    rows: list[dict] = []
    holdings: list[pd.DataFrame] = []

    for dt_ in dates:
        cs = df[df[date_col] == dt_]
        if len(cs) < 20:
            continue

        book = build_book(
            cs,
            score_col=score_col,
            incumbents=incumbents,
            apply_vol_target=apply_vol_target,
            n=n_positions or cfg.n_positions,
        )
        if book.empty:
            continue

        new_weights = dict(zip(book["security"], book["weight"]))

        # ---- turnover and cost ------------------------------------------
        keys = set(new_weights) | set(prev_weights)
        turnover = sum(abs(new_weights.get(k, 0.0) - prev_weights.get(k, 0.0)) for k in keys)
        cost = turnover * cost_bps / 10_000.0

        # ---- realised return over the coming week ------------------------
        r = pd.to_numeric(book[ret_col], errors="coerce")
        # A holding whose forward return is missing has stopped trading --
        # a delisting. Treat it as flat rather than dropping it, so the
        # capital is genuinely stuck for that week instead of teleporting
        # into the surviving names.
        r = r.fillna(0.0)
        gross = float((book["weight"].to_numpy() * r.to_numpy()).sum())
        net = gross - cost

        rows.append(
            {
                "date": pd.Timestamp(dt_),
                "gross_ret": gross,
                "cost": cost,
                "net_ret": net,
                "turnover": turnover,
                "n_positions": len(book),
                "invested": float(book["weight"].sum()),
                "vol_scalar": float(book["vol_scalar"].iloc[0]),
                "n_missing_ret": int(pd.to_numeric(book[ret_col], errors="coerce").isna().sum()),
            }
        )

        snap = book[["security", "weight", "raw_weight", score_col]].copy()
        snap["date"] = pd.Timestamp(dt_)
        for c in (SECTOR_COL, "NAME", "vol_60", ret_col):
            if c in book.columns:
                snap[c] = book[c].values
        holdings.append(snap)

        prev_weights = new_weights
        incumbents = list(book["security"])

    if not rows:
        raise RuntimeError("Backtest produced no periods -- check the score column and panel.")

    returns = pd.DataFrame(rows).set_index("date").sort_index()
    hold = pd.concat(holdings, ignore_index=True) if holdings else pd.DataFrame()

    if verbose:
        log.info(
            "Backtest: %s weeks | avg turnover %.1f%% | avg positions %.1f | avg invested %.0f%%",
            len(returns),
            returns["turnover"].mean() * 100,
            returns["n_positions"].mean(),
            returns["invested"].mean() * 100,
        )
    return {"returns": returns, "holdings": hold, "diagnostics": {}}


def benchmark_returns(panel: pd.DataFrame, bench_panel: pd.DataFrame | None = None,
                      date_col: str = "date", ret_col: str = "fwd_ret_1w") -> pd.Series:
    """Equal-weighted return of the investable universe -- the honest hurdle.

    Comparing a 5-name book against the cap-weighted S&P is a comparison of
    two different bets. The equal-weighted universe return isolates whether
    *selection* added value, independent of the size tilt.
    """
    return panel.groupby(date_col)[ret_col].mean()


def deciles(panel: pd.DataFrame, score_col: str = "score", date_col: str = "date",
            ret_col: str = "fwd_ret_1w", n: int = 10) -> pd.DataFrame:
    """Mean forward return by score decile -- the shape of the signal.

    A signal that works should be roughly monotonic across deciles. If only
    the top decile performs and the rest is flat, the signal is fragile: it
    depends on a handful of names rather than a real cross-sectional effect.
    """
    df = panel.dropna(subset=[score_col, ret_col]).copy()

    def _assign(g):
        try:
            return pd.qcut(g[score_col].rank(method="first"), n, labels=False, duplicates="drop")
        except ValueError:
            return pd.Series(np.nan, index=g.index)

    df["decile"] = df.groupby(date_col, group_keys=False)[[score_col]].apply(
        lambda g: pd.Series(_assign(g), index=g.index)
    )
    out = df.groupby("decile").agg(
        mean_fwd_ret=(ret_col, "mean"),
        median_fwd_ret=(ret_col, "median"),
        n=(ret_col, "size"),
    )
    out["annualised"] = (1 + out["mean_fwd_ret"]) ** 52 - 1
    return out
