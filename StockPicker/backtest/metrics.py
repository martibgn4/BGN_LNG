"""
============================================================
backtest/metrics.py  --  Performance & risk statistics
============================================================
Deliberately includes the unflattering measures. A weekly 5-stock
strategy will always produce a good-looking CAGR on some sample;
what matters is whether it survives drawdown, cost, regime change
and the question "is this distinguishable from luck".

Statistics that earn their place here
-------------------------------------
* **Deflated / t-stat on alpha** -- a Sharpe of 1.0 over 200 weeks
  is roughly 2 standard errors from zero. Reporting Sharpe without
  its uncertainty is how backtests mislead.
* **Max drawdown and time under water** -- the numbers that decide
  whether a strategy is actually holdable. A 5-name book will have
  drawdowns that a 50-name book does not.
* **Rolling 1-year excess return** -- shows whether outperformance
  is broad or comes from one lucky year.
* **Regime breakdown** -- performance in up vs down markets, and in
  high vs low volatility, which is where concentrated momentum
  books usually hide their weakness.
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 52


def _ann_return(r: pd.Series) -> float:
    if len(r) == 0:
        return np.nan
    total = (1 + r).prod()
    years = len(r) / PERIODS_PER_YEAR
    return total ** (1 / years) - 1 if years > 0 and total > 0 else np.nan


def _ann_vol(r: pd.Series) -> float:
    return r.std() * np.sqrt(PERIODS_PER_YEAR)


def drawdown_series(r: pd.Series) -> pd.Series:
    equity = (1 + r).cumprod()
    return equity / equity.cummax() - 1.0


def summary(r: pd.Series, benchmark: pd.Series | None = None, rf: float = 0.0) -> dict:
    """Headline performance statistics for a weekly return series."""
    r = r.dropna()
    if len(r) < 8:
        return {"n_weeks": len(r)}

    ann_ret, ann_vol = _ann_return(r), _ann_vol(r)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol and np.isfinite(ann_vol) else np.nan

    downside = r[r < 0]
    dd_vol = downside.std() * np.sqrt(PERIODS_PER_YEAR) if len(downside) > 1 else np.nan
    sortino = (ann_ret - rf) / dd_vol if dd_vol and dd_vol > 0 else np.nan

    dd = drawdown_series(r)
    max_dd = dd.min()

    # Time under water: longest run of consecutive weeks below a prior peak.
    under = dd < -1e-9
    longest, run = 0, 0
    for u in under:
        run = run + 1 if u else 0
        longest = max(longest, run)

    out = {
        "n_weeks": len(r),
        "years": len(r) / PERIODS_PER_YEAR,
        "total_return": (1 + r).prod() - 1,
        "cagr": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": ann_ret / abs(max_dd) if max_dd and max_dd < 0 else np.nan,
        "weeks_under_water": longest,
        "hit_rate": (r > 0).mean(),
        "best_week": r.max(),
        "worst_week": r.min(),
        "skew": r.skew(),
        "kurtosis": r.kurtosis(),
        # Is the Sharpe distinguishable from zero? t ~= SR * sqrt(years)
        "sharpe_t_stat": sharpe * np.sqrt(len(r) / PERIODS_PER_YEAR) if np.isfinite(sharpe) else np.nan,
    }

    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        common = r.index.intersection(b.index)
        r2, b2 = r.loc[common], b.loc[common]
        if len(common) > 8:
            excess = r2 - b2
            te = excess.std() * np.sqrt(PERIODS_PER_YEAR)
            out.update(
                {
                    "bench_cagr": _ann_return(b2),
                    "bench_vol": _ann_vol(b2),
                    "excess_cagr": _ann_return(r2) - _ann_return(b2),
                    "tracking_error": te,
                    "information_ratio": (_ann_return(r2) - _ann_return(b2)) / te if te > 0 else np.nan,
                    "beta_to_bench": r2.cov(b2) / b2.var() if b2.var() > 0 else np.nan,
                    "corr_to_bench": r2.corr(b2),
                    "up_capture": r2[b2 > 0].mean() / b2[b2 > 0].mean() if (b2 > 0).any() else np.nan,
                    "down_capture": r2[b2 < 0].mean() / b2[b2 < 0].mean() if (b2 < 0).any() else np.nan,
                    "win_vs_bench": (r2 > b2).mean(),
                }
            )
            # Alpha t-stat from a simple market-model regression.
            if b2.var() > 0:
                beta = r2.cov(b2) / b2.var()
                alpha = (r2 - beta * b2).mean()
                resid_sd = (r2 - beta * b2).std()
                se = resid_sd / np.sqrt(len(common)) if len(common) else np.nan
                out["alpha_weekly"] = alpha
                out["alpha_ann"] = alpha * PERIODS_PER_YEAR
                out["alpha_t_stat"] = alpha / se if se and se > 0 else np.nan
    return out


def regime_breakdown(r: pd.Series, benchmark: pd.Series) -> pd.DataFrame:
    """Performance split by market direction and by volatility regime.

    Concentrated momentum strategies typically look excellent in calm,
    trending markets and lose badly at turning points. This is where that
    shows up, and it is the table to read before sizing the strategy.
    """
    common = r.index.intersection(benchmark.index)
    r, b = r.loc[common], benchmark.loc[common]
    if len(common) < 20:
        return pd.DataFrame()

    bench_vol = b.rolling(13, min_periods=8).std()
    hi_vol = bench_vol > bench_vol.median()

    rows = []
    for label, mask in [
        ("Market up", b > 0),
        ("Market down", b <= 0),
        ("High vol regime", hi_vol.fillna(False)),
        ("Low vol regime", ~hi_vol.fillna(True)),
    ]:
        rr, bb = r[mask], b[mask]
        if len(rr) < 5:
            continue
        rows.append(
            {
                "regime": label,
                "n_weeks": len(rr),
                "strategy_mean": rr.mean(),
                "bench_mean": bb.mean(),
                "excess": rr.mean() - bb.mean(),
                "strategy_ann": (1 + rr.mean()) ** PERIODS_PER_YEAR - 1,
                "hit_rate": (rr > 0).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


def yearly_table(r: pd.Series, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Calendar-year returns -- exposes whether alpha is broad or one-off."""
    df = pd.DataFrame({"strategy": r})
    if benchmark is not None:
        df["benchmark"] = benchmark.reindex(r.index)
    out = df.groupby(df.index.year).apply(lambda g: (1 + g).prod() - 1, include_groups=False)
    if benchmark is not None and "benchmark" in out.columns:
        out["excess"] = out["strategy"] - out["benchmark"]
    return out


def format_summary(s: dict, title: str = "Performance") -> str:
    """Human-readable block for logs and the console."""
    def pct(x):
        return f"{x*100:7.2f}%" if isinstance(x, (int, float)) and np.isfinite(x) else "      -"

    def num(x):
        return f"{x:7.2f}" if isinstance(x, (int, float)) and np.isfinite(x) else "      -"

    lines = [
        f"--- {title} ---",
        f"  Period            {s.get('years', float('nan')):.1f}y ({s.get('n_weeks', 0)} weeks)",
        f"  CAGR              {pct(s.get('cagr'))}",
        f"  Volatility        {pct(s.get('ann_vol'))}",
        f"  Sharpe            {num(s.get('sharpe'))}   (t = {num(s.get('sharpe_t_stat'))})",
        f"  Sortino           {num(s.get('sortino'))}",
        f"  Max drawdown      {pct(s.get('max_drawdown'))}",
        f"  Calmar            {num(s.get('calmar'))}",
        f"  Weeks underwater  {s.get('weeks_under_water', '-')}",
        f"  Weekly hit rate   {pct(s.get('hit_rate'))}",
    ]
    if "bench_cagr" in s:
        lines += [
            f"  -- vs benchmark --",
            f"  Benchmark CAGR    {pct(s.get('bench_cagr'))}   vol {pct(s.get('bench_vol'))}",
            f"  Excess CAGR       {pct(s.get('excess_cagr'))}",
            f"  Info ratio        {num(s.get('information_ratio'))}",
            f"  Alpha (ann)       {pct(s.get('alpha_ann'))}   (t = {num(s.get('alpha_t_stat'))})",
            f"  Beta / Corr       {num(s.get('beta_to_bench'))} / {num(s.get('corr_to_bench'))}",
            f"  Up / Down capture {num(s.get('up_capture'))} / {num(s.get('down_capture'))}",
        ]
    return "\n".join(lines)
