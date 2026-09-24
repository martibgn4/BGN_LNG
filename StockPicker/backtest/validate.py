"""
============================================================
backtest/validate.py  --  Look-ahead and sanity checks
============================================================
A backtest is only worth its weakest assumption. These checks are
designed to *fail* if the pipeline has quietly become optimistic,
and are meant to be run whenever feature code changes.

Checks
------
1. ``check_feature_lag``      Are features genuinely known before the
                             bar they are used on?
2. ``check_no_future_returns`` Does any feature correlate suspiciously
                             with the *contemporaneous* return, which
                             would indicate the return leaked in?
3. ``check_survivorship``    Does the panel actually contain names that
                             stopped trading, and do they die properly?
4. ``check_membership_pit``  Does any rebalance select a name that was
                             not an index member at the time?
5. ``shuffle_test``          Destroy the signal by permuting scores
                             within each date. Performance must collapse
                             to roughly the universe return. If a shuffled
                             signal still "works", the edge is coming from
                             the mechanics, not the signal.
6. ``lag_sensitivity``       Delay execution by one extra week. A real
                             weekly signal decays but should not invert;
                             a signal that only works with instant
                             execution is not tradable.
============================================================
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def check_feature_lag(panel: pd.DataFrame, price_col: str = "exec_px",
                      feature: str = "mom_1w", date_col: str = "date") -> dict:
    """A one-week momentum feature must not equal this bar's own return.

    If ``mom_1w`` at date t were computed from the price at t rather than
    t-1, it would correlate ~1.0 with the trailing return measured on the
    same bar. A high correlation here is the signature of a missing shift.
    """
    if feature not in panel.columns or "trail_ret_1w" not in panel.columns:
        return {"skipped": True}
    sub = panel[[feature, "trail_ret_1w"]].dropna()
    corr = sub[feature].corr(sub["trail_ret_1w"]) if len(sub) > 100 else np.nan
    ok = not (np.isfinite(corr) and corr > 0.95)
    return {
        "check": "feature_lag",
        "corr_feature_vs_same_bar_return": corr,
        "pass": ok,
        "note": "corr near 1.0 would mean the feature was not lagged",
    }


def check_no_future_returns(panel: pd.DataFrame, feature_cols: list[str],
                            target: str = "fwd_ret_1w") -> pd.DataFrame:
    """Flag any feature whose raw correlation with the forward return is
    implausibly high.

    Genuine weekly cross-sectional signals correlate with next week's return
    at roughly 0.01-0.06. Anything above ~0.15 is almost certainly a leak.
    """
    rows = []
    for c in feature_cols:
        if c not in panel.columns:
            continue
        sub = panel[[c, target]].dropna()
        if len(sub) < 500:
            continue
        r = sub[c].corr(sub[target])
        rows.append({"feature": c, "corr_with_fwd_ret": r, "suspicious": abs(r) > 0.15})
    out = pd.DataFrame(rows).sort_values("corr_with_fwd_ret", key=abs, ascending=False)
    return out


def check_survivorship(panel: pd.DataFrame, date_col: str = "date") -> dict:
    """Confirm the panel contains names that genuinely stop trading."""
    last_seen = panel.groupby("security")[date_col].max()
    panel_end = panel[date_col].max()
    # Names whose last appearance is well before the end of the sample.
    dead = last_seen[last_seen < panel_end - pd.Timedelta(days=60)]
    return {
        "check": "survivorship",
        "total_securities": panel["security"].nunique(),
        "securities_that_stop_early": len(dead),
        "pct_dead": len(dead) / max(panel["security"].nunique(), 1),
        "pass": len(dead) > 50,
        "note": "a bias-free S&P 500 panel since 2006 should contain hundreds of exits",
    }


def check_membership_pit(panel: pd.DataFrame) -> dict:
    """Every scored row must have been an index member at that date."""
    if "in_index" not in panel.columns:
        return {"skipped": True}
    non_member = (~panel["in_index"]).sum()
    return {
        "check": "membership_pit",
        "rows": len(panel),
        "non_member_rows": int(non_member),
        "pass": non_member == 0,
        "note": "apply_filters should have removed every non-member row",
    }


def shuffle_test(panel: pd.DataFrame, score_col: str = "score", n_trials: int = 5,
                 n_positions: int = 5, seed: int = 0) -> pd.DataFrame:
    """Permute scores within each date and re-run. Edge should vanish.

    This is the strongest single test of whether the result is real. If a
    randomly-ranked 5-name portfolio earns a similar Sharpe, the apparent
    alpha is coming from the universe, the weighting scheme or a mechanical
    artefact -- not from stock selection.
    """
    from .engine import run_backtest
    from .metrics import summary

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_trials):
        p = panel.copy()
        p[score_col] = p.groupby("date")[score_col].transform(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index)
        )
        bt = run_backtest(p.dropna(subset=[score_col]), score_col=score_col,
                          n_positions=n_positions, verbose=False)
        s = summary(bt["returns"]["net_ret"])
        rows.append({"trial": i, "cagr": s.get("cagr"), "sharpe": s.get("sharpe"),
                     "max_dd": s.get("max_drawdown")})
    return pd.DataFrame(rows)


def lag_sensitivity(panel: pd.DataFrame, score_col: str = "score",
                    n_positions: int = 5, max_lag: int = 3) -> pd.DataFrame:
    """Re-run with the score delayed by 1..max_lag extra weeks.

    Real signals decay gracefully. A strategy that only works at zero lag is
    relying on information you would not have had time to act on.
    """
    from .engine import benchmark_returns, run_backtest
    from .metrics import summary

    rows = []
    bench = benchmark_returns(panel.dropna(subset=[score_col]))
    for lag in range(0, max_lag + 1):
        p = panel.copy()
        if lag:
            p[score_col] = p.groupby("security")[score_col].shift(lag)
        p = p.dropna(subset=[score_col])
        if p["date"].nunique() < 50:
            continue
        bt = run_backtest(p, score_col=score_col, n_positions=n_positions, verbose=False)
        s = summary(bt["returns"]["net_ret"], benchmark=bench)
        rows.append({"extra_weeks_lag": lag, "cagr": s.get("cagr"),
                     "sharpe": s.get("sharpe"), "excess_cagr": s.get("excess_cagr"),
                     "info_ratio": s.get("information_ratio")})
    return pd.DataFrame(rows)


def run_all(panel: pd.DataFrame, feature_cols: list[str], score_col: str = "score",
            n_positions: int = 5) -> dict:
    """Run every structural check and print a verdict."""
    results = {}
    for fn in (check_feature_lag, check_membership_pit, check_survivorship):
        try:
            r = fn(panel)
            results[r.get("check", fn.__name__)] = r
        except Exception as exc:  # noqa: BLE001
            results[fn.__name__] = {"error": str(exc)}

    leaks = check_no_future_returns(panel, feature_cols)
    results["leak_scan"] = leaks

    print("\n" + "=" * 78)
    print("STRUCTURAL VALIDATION")
    print("=" * 78)
    for k, v in results.items():
        if k == "leak_scan":
            continue
        status = "PASS" if v.get("pass") else ("SKIP" if v.get("skipped") else "FAIL")
        print(f"  [{status}] {k}")
        for kk, vv in v.items():
            if kk in ("check", "pass", "skipped"):
                continue
            print(f"          {kk}: {vv}")

    if not leaks.empty:
        susp = leaks[leaks["suspicious"]]
        print(f"\n  Leak scan: {len(susp)} of {len(leaks)} features exceed |corr| > 0.15 with forward return")
        print("  Top 8 by absolute correlation:")
        print(leaks.head(8).to_string(index=False))
        if len(susp):
            print("\n  WARNING -- inspect these features for look-ahead:")
            print(susp.to_string(index=False))
    return results
