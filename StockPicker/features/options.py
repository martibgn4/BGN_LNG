"""
============================================================
features/options.py  --  Options-implied signals
============================================================
Price and volume tell you what happened. The options surface tells
you what people are *paying to be protected from*, which is a
different and partly independent piece of information.

Three ideas do most of the work here.

**Skew.** The implied vol of a 90%-moneyness put minus that of a
110%-moneyness call. When skew steepens, someone is paying up for
downside protection. Steep and steepening skew has historically
preceded weakness; unusually flat skew tends to mark complacency.
Crucially this is a *relative* measure across the cross-section --
some names are structurally skewed (biotech), so what matters is
skew versus that name's own recent history.

**Volatility risk premium.** Implied vol minus realised vol. This
is the compensation option sellers demand. A very high VRP means
the market has already priced the risk, which is often the point at
which the bad news is in; a compressed VRP means the option market
sees no trouble, which is when trouble is cheapest to hedge.

**Positioning.** Put/call ratios in volume and open interest.
Volume is the flow (what changed this week), open interest the
stock (how the book is positioned). The *change* in these is far
more informative than the level, since the level is dominated by
persistent structural hedging differences between names.

All of these enter the model as ordinary cross-sectional signals --
none is assumed to work, and the IC study reports on each.
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

IV_ATM = "30DAY_IMPVOL_100.0%MNY_DF"
IV_PUT = "30DAY_IMPVOL_90.0%MNY_DF"
IV_CALL = "30DAY_IMPVOL_110.0%MNY_DF"


def compute_options_features(df: pd.DataFrame) -> pd.DataFrame:
    """Options-implied features for one security from its weekly panel."""
    d = df.sort_values("date").set_index("date")
    out = pd.DataFrame(index=d.index)

    def col(name: str) -> pd.Series:
        return pd.to_numeric(d[name], errors="coerce") if name in d.columns else pd.Series(np.nan, index=d.index)

    iv_atm, iv_put, iv_call = col(IV_ATM), col(IV_PUT), col(IV_CALL)
    rv = col("VOLATILITY_30D")

    # ---- skew ------------------------------------------------------------
    skew = iv_put - iv_call
    out["iv_skew"] = skew
    # Normalised by ATM level so a 5-vol skew on a 20-vol name is not treated
    # the same as a 5-vol skew on an 80-vol name.
    out["iv_skew_norm"] = skew / iv_atm.replace(0, np.nan)
    out["iv_skew_chg_4w"] = skew - skew.shift(4)
    # Skew versus the name's own 52-week history -- this is the version that
    # is comparable across the cross-section.
    out["iv_skew_z52"] = (
        (skew - skew.rolling(52, min_periods=26).mean())
        / skew.rolling(52, min_periods=26).std().replace(0, np.nan)
    )

    # ---- volatility risk premium ----------------------------------------
    vrp = iv_atm - rv
    out["vrp"] = vrp
    out["vrp_z52"] = (
        (vrp - vrp.rolling(52, min_periods=26).mean())
        / vrp.rolling(52, min_periods=26).std().replace(0, np.nan)
    )
    out["iv_atm"] = iv_atm
    out["iv_chg_4w"] = iv_atm - iv_atm.shift(4)

    # ---- positioning -----------------------------------------------------
    pc_vol = col("PUT_CALL_VOLUME_RATIO_CUR_DAY")
    pc_oi = col("PUT_CALL_OPEN_INTEREST_RATIO")
    # Smooth the volume ratio: a single day's put/call is mostly noise.
    out["put_call_vol"] = pc_vol.rolling(4, min_periods=2).mean()
    out["put_call_oi"] = pc_oi
    out["put_call_oi_chg_13w"] = pc_oi - pc_oi.shift(13)
    out["put_call_vol_z52"] = (
        (out["put_call_vol"] - out["put_call_vol"].rolling(52, min_periods=26).mean())
        / out["put_call_vol"].rolling(52, min_periods=26).std().replace(0, np.nan)
    )

    # Total option open interest relative to its own base -- a proxy for how
    # much attention the name is getting in the derivatives market.
    oi_total = col("OPEN_INT_TOTAL_CALL") + col("OPEN_INT_TOTAL_PUT")
    out["option_oi_trend"] = (
        oi_total.rolling(4, min_periods=2).mean()
        / oi_total.rolling(52, min_periods=26).mean().replace(0, np.nan)
    )

    out = out.reset_index()
    out.insert(1, "security", df["security"].iloc[0])
    return out
