"""
============================================================
features/registry.py  --  The signal catalogue
============================================================
One place that names every feature, groups it into a family, and
records the sign we expect it to carry based on the published
literature.

The expected sign is not used to *force* anything -- the model
learns weights from data. It exists so that when a signal comes
out of the research with the opposite sign to forty years of
evidence, that is visible rather than buried. A value factor that
prices as negative is usually a bug, a data alignment error, or a
regime worth understanding, and all three deserve a look before
the model is trusted.
============================================================
"""

from __future__ import annotations

# family -> {feature: expected sign}
FEATURE_FAMILIES: dict[str, dict[str, int]] = {
    "momentum": {
        "mom_12_1": +1,      # Jegadeesh-Titman continuation
        "mom_6_1": +1,
        "mom_3_1": +1,
        "resid_mom": +1,     # Blitz-Huij-Martens, beta stripped out
        "mom_x_fip": +1,     # smooth-path momentum persists longer
        "mom_x_adx": +1,     # momentum confirmed by trend strength
        "mom_sharpe": +1,    # risk-adjusted momentum
        "pct_52w_high": +1,  # George-Hwang anchoring
    },
    "reversal": {
        "mom_1m": -1,        # short-horizon reversal
        "mom_1w": -1,
        "bollinger_z": -1,   # stretched vs its own band
        "rsi_14": -1,        # overbought fades at weekly horizon
        # Moved here from "trend" after the IC study, on reasoning rather than
        # on the sample alone. Both measure how stretched a price is against
        # its own recent path, which is the same quantity bollinger_z measures
        # -- and bollinger_z was assigned -1 a priori and confirmed at t=-3.9.
        # At a one-week holding period short-horizon mean reversion dominates
        # trend continuation, so a "trend" reading of these two was simply the
        # wrong prior for this horizon: measured, both came in negative
        # (macd_hist t=-3.3, dist_ma_50 t=-1.6).
        #
        # Flagging the reclassification explicitly because moving a feature
        # after seeing its IC is exactly how in-sample bias enters a model.
        # The defence is that it is coherent with a prior already set
        # independently, not that the number came out nicely.
        "macd_hist": -1,
        "dist_ma_50": -1,
    },
    "risk": {
        "vol_60": -1,        # low-volatility anomaly
        "vol_252": -1,
        "ivol": -1,          # idiosyncratic vol -- Ang et al.
        "downside_dev": -1,
        "beta": -1,          # betting-against-beta
        "max_dd_252": +1,    # less-damaged names do better
        "vol_ratio": -1,     # vol accelerating is a warning
    },
    "liquidity": {
        "dollar_vol": 0,     # mostly a filter; sign ambiguous in large caps
        "vol_trend": +1,     # expanding volume confirms a move
        "amihud": -1,        # fragile, expensive names underperform
        "vol_px_corr": +1,   # accumulation
    },
    "trend": {
        # Only the long-horizon trend measures remain here. The short ones
        # (macd_hist, dist_ma_50) behave as reversal at a weekly holding
        # period and now sit in that family.
        "dist_ma_200": +1,
        "adx_14": 0,         # conditioner rather than a directional signal
    },
    "revisions": {
        "eps_rev_4w": +1,    # fastest-decaying, strongest of the group
        "eps_rev_13w": +1,
        "eps_rev_26w": +1,
        "eps_rev_consistency": +1,
        "tp_rev_4w": +1,
        "tp_rev_13w": +1,
        "rating_chg_4w": +1,
        "rating_chg_13w": +1,
        "target_upside": 0,  # high upside often flags a falling knife
    },
    "value": {
        "earnings_yield": +1,
        "book_to_price": +1,
        "ebitda_yield": +1,
        "fcf_yield": +1,
    },
    "quality": {
        "roe": +1,
        "gross_margin": +1,   # Novy-Marx gross profitability
        "margin_trend": +1,
        "share_issuance_1y": +1,  # buybacks good, dilution bad
    },
    "crowding": {
        "short_interest": -1,
        "short_interest_chg": -1,
    },
    "sentiment": {
        "news_sentiment": +1,
        "news_sentiment_chg": +1,
        "news_sentiment_vol": -1,
    },
    "sector": {
        "sector_mom_13w_rel": +1,
        "sector_mom_26w_rel": +1,
    },
    # Options-implied. Only available from ~2015 on this entitlement, so these
    # are NaN over the early sample and the blend simply carries no weight on
    # them there -- which is the correct behaviour, not a gap to be filled.
    "options": {
        "iv_skew_norm": -1,       # steep put skew = paying up for protection
        "iv_skew_chg_4w": -1,     # skew steepening is the active signal
        "iv_skew_z52": -1,        # vs the name's own history
        "vrp_z52": 0,             # ambiguous: fear priced in, or fear justified
        "iv_chg_4w": -1,          # rising implied vol
        "put_call_vol_z52": -1,   # unusual put buying
        "put_call_oi_chg_13w": -1,
        "option_oi_trend": 0,     # attention, direction ambiguous
    },
}


def all_features() -> list[str]:
    return [f for fam in FEATURE_FAMILIES.values() for f in fam]


def expected_signs() -> dict[str, int]:
    return {f: s for fam in FEATURE_FAMILIES.values() for f, s in fam.items()}


def family_of(feature: str) -> str:
    for fam, feats in FEATURE_FAMILIES.items():
        if feature in feats:
            return fam
    return "other"


def z_names(features: list[str] | None = None) -> list[str]:
    """Standardised column names, as produced by ``standardize_panel``."""
    return [f"{f}_z" for f in (features or all_features())]
