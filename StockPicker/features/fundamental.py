"""
============================================================
features/fundamental.py  --  Fundamentals, estimates, sentiment
============================================================
Slower-moving signals sampled weekly. These are the ones that
carry information price series cannot: what analysts are changing
their minds about, how expensive the name is, how crowded the
short side is, and what the news flow tone looks like.

A note on point-in-time integrity
---------------------------------
Bloomberg's ``BEST_*`` consensus fields are genuinely as-of-date
when pulled through BDH -- the value on 2015-03-31 is the
consensus that *existed* on 2015-03-31, not a later restatement.
That is what makes revisions usable in a backtest.

Reported fundamentals (PE_RATIO, RETURN_COM_EQY, ...) are a
different matter: they are stamped against the price date but the
underlying accounting figure was only *published* some weeks after
period end. To avoid using earnings before they were public, every
reported-fundamental feature is lagged by ``REPORTING_LAG_DAYS``.
This costs a little signal and removes a large look-ahead bias.

Signal families
---------------
Revisions   The fastest-decaying and among the most reliable equity
            signals. Analysts revise in the same direction for months
            (anchoring), so the *change* in consensus EPS predicts
            returns far better than its level.
Value       Cheapness. Weak standalone in recent regimes but a genuine
            diversifier against momentum, and it is what stops the
            model buying the top of a bubble.
Quality     Profitability and cash generation. Novy-Marx showed gross
            profitability rivals value as a predictor; quality also
            damps drawdowns, which serves the low-volatility objective.
Crowding    Short interest. High days-to-cover predicts negative
            returns on average, but interacts with momentum.
Sentiment   Bloomberg's NLP news sentiment, used as a level and as a
            change. The change matters more -- the market prices the
            level quickly.
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FFILL_LIMIT_WEEKS, REPORTED_SPARSE_FIELDS

# Conservative lag applied to reported (as opposed to estimated) fundamentals
# so we never trade on an accounting figure before it was published.
REPORTING_LAG_DAYS = 45


def _pct_change(s: pd.Series, periods: int) -> pd.Series:
    """Percent change that is robust to sign flips and zeros.

    A naive pct_change on a series that crosses zero (EPS going from -0.1 to
    +0.1) produces a nonsense -200%. Scaling by the absolute base keeps the
    sign meaningful and the magnitude bounded.
    """
    base = s.shift(periods).abs()
    return (s - s.shift(periods)) / base.replace(0, np.nan)


def compute_fundamental_features(
    df: pd.DataFrame,
    price: pd.Series | None = None,
    weekly_periods: tuple[int, ...] = (4, 13, 26),
) -> pd.DataFrame:
    """Compute fundamental/estimate/sentiment features for one security.

    Parameters
    ----------
    df
        Weekly rows for a single security containing the SLOW_FIELDS,
        sorted by date.
    price
        Daily close series for the same security, used for target-price
        upside. Reindexed onto the weekly grid.
    weekly_periods
        Revision horizons in weeks (~1m, ~3m, ~6m).
    """
    d = df.sort_values("date").set_index("date")
    out = pd.DataFrame(index=d.index)

    # ---- forward-fill the fields Bloomberg only prints on report dates ----
    # Requested weekly, RETURN_COM_EQY / GROSS_MARGIN / TRAIL_12M_EPS and
    # friends return roughly four points a year -- they are stamped on
    # reporting dates and are NaN in between. Left alone that makes the whole
    # quality and value block ~93% missing, which is what was silently
    # emptying those factors.
    #
    # Carrying the last printed value forward is the point-in-time correct
    # treatment: a company's most recently *reported* ROE remains the latest
    # publicly known figure until it reports again. The carry is capped so a
    # discontinued or delisted field cannot propagate indefinitely.
    for f in REPORTED_SPARSE_FIELDS:
        if f in d.columns:
            d[f] = pd.to_numeric(d[f], errors="coerce").ffill(limit=FFILL_LIMIT_WEEKS)

    def col(name: str) -> pd.Series:
        return pd.to_numeric(d[name], errors="coerce") if name in d.columns else pd.Series(np.nan, index=d.index)

    # Number of weekly rows corresponding to the reporting publication lag.
    lag_rows = max(1, REPORTING_LAG_DAYS // 7)

    # ---------------- estimate revisions (no lag: already as-of) ----------
    best_eps = col("BEST_EPS")
    for p in weekly_periods:
        out[f"eps_rev_{p}w"] = _pct_change(best_eps, p)
    # Revision *breadth over time*: how consistently estimates have risen.
    out["eps_rev_consistency"] = (
        best_eps.diff().rolling(26, min_periods=13).apply(lambda x: np.sign(x).mean(), raw=True)
    )

    target = col("BEST_TARGET_PRICE")
    for p in (4, 13):
        out[f"tp_rev_{p}w"] = _pct_change(target, p)

    rating = col("BEST_ANALYST_RATING")
    out["rating_level"] = rating
    for p in (4, 13):
        out[f"rating_chg_{p}w"] = rating - rating.shift(p)
    out["analyst_coverage"] = col("TOT_ANALYST_REC")

    if price is not None and len(price):
        px_w = price.reindex(d.index, method="ffill")
        # Implied upside to consensus target. Strong contrarian character:
        # very high implied upside often flags a falling knife, so this is
        # left to the model to weight rather than assumed positive.
        out["target_upside"] = target / px_w.replace(0, np.nan) - 1.0

    # ---------------- value (lagged: reported figures) --------------------
    pe = col("PE_RATIO").shift(lag_rows)
    pb = col("PX_TO_BOOK_RATIO").shift(lag_rows)
    ev_ebitda = col("EV_TO_T12M_EBITDA").shift(lag_rows)
    fcf_yield = col("FREE_CASH_FLOW_YIELD").shift(lag_rows)

    # Inverted so that "higher is cheaper" for every value feature, which
    # keeps the sign convention uniform across the whole feature block.
    out["earnings_yield"] = 1.0 / pe.where(pe > 0)
    out["book_to_price"] = 1.0 / pb.where(pb > 0)
    out["ebitda_yield"] = 1.0 / ev_ebitda.where(ev_ebitda > 0)
    out["fcf_yield"] = fcf_yield

    # ---------------- quality (lagged) ------------------------------------
    out["roe"] = col("RETURN_COM_EQY").shift(lag_rows)
    out["gross_margin"] = col("GROSS_MARGIN").shift(lag_rows)
    out["margin_trend"] = out["gross_margin"] - out["gross_margin"].shift(52)

    # Net share issuance. Buybacks predict positive returns, dilution
    # negative -- one of the more robust anomalies and free from the
    # accounting-restatement problem.
    sh_out = col("EQY_SH_OUT").shift(lag_rows)
    out["share_issuance_1y"] = -_pct_change(sh_out, 52)

    # ---------------- crowding --------------------------------------------
    si = col("SHORT_INT_RATIO")
    out["short_interest"] = si
    out["short_interest_chg"] = si - si.shift(13)

    # ---------------- sentiment -------------------------------------------
    sent = col("NEWS_SENTIMENT_DAILY_AVG")
    out["news_sentiment"] = sent.rolling(4, min_periods=2).mean()
    out["news_sentiment_chg"] = out["news_sentiment"] - out["news_sentiment"].shift(13)
    # Dispersion in tone: unstable narrative is a risk marker.
    out["news_sentiment_vol"] = sent.rolling(13, min_periods=6).std()

    # ---------------- risk descriptors ------------------------------------
    out["bbg_vol_90d"] = col("VOLATILITY_90D")
    out["bbg_beta"] = col("BETA_ADJ_OVERRIDABLE")

    out = out.reset_index()
    out.insert(1, "security", df["security"].iloc[0])
    return out


def compute_sector_features(
    panel: pd.DataFrame,
    sector_col: str,
    date_col: str = "date",
    ret_col: str = "fwd_ret_1w",
) -> pd.DataFrame:
    """Sector-level relative strength, joined back onto each name.

    Captures the rotation dimension the user asked for: which *sectors* have
    been leading, and whether the name is a leader within a leading sector.
    Computed from realised trailing returns only -- ``ret_col`` here is the
    trailing return column, never a forward one.
    """
    df = panel.copy()
    if sector_col not in df.columns or ret_col not in df.columns:
        return df

    sec_ret = (
        df.groupby([date_col, sector_col])[ret_col]
        .mean()
        .rename("sector_ret")
        .reset_index()
    )
    sec_ret = sec_ret.sort_values([sector_col, date_col])
    g = sec_ret.groupby(sector_col)["sector_ret"]
    # Trailing sector momentum over ~3m and ~6m of weekly observations.
    sec_ret["sector_mom_13w"] = g.transform(lambda s: s.rolling(13, min_periods=6).sum())
    sec_ret["sector_mom_26w"] = g.transform(lambda s: s.rolling(26, min_periods=13).sum())

    # Sector strength relative to the average sector on that date.
    for c in ("sector_mom_13w", "sector_mom_26w"):
        sec_ret[f"{c}_rel"] = sec_ret[c] - sec_ret.groupby(date_col)[c].transform("mean")

    keep = [date_col, sector_col, "sector_mom_13w_rel", "sector_mom_26w_rel"]
    return df.merge(sec_ret[keep], on=[date_col, sector_col], how="left")
