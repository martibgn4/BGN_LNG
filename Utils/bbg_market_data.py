"""
============================================================
bbg_market_data.py  —  Bloomberg Data Loader
============================================================
Pulls all live/historical market data required by the Dated
Brent Chooser Option pricer from Bloomberg, with a graceful
fallback to caller-supplied hardcoded values when Bloomberg
is unavailable (e.g. offline dev environment).

Bloomberg tickers used
──────────────────────
Forward prices  (BDP, PX_LAST as-of date):
  CO1 Comdty   — front-month ICE Brent crude futures  (generic cont.)
  CO2 Comdty   — second-month generic continuation
  CO3 Comdty   — third-month generic continuation

  Tenor mapping (accounting for roll schedule):
    If valuation_date is BEFORE the CO1 notice/roll date this month:
        M-1 ← CO1,  M ← CO2,  M+1 ← CO3
    If valuation_date is ON or AFTER roll:
        M-1 ← last close of the just-expired CO1 (via BDH lookback)
        M   ← CO1,  M+1 ← CO2

Implied vols  (BDP fields IVOL_MID; fallback to HIST_VOL_260D):
  COA Comdty   — ATM 1M implied vol (Bloomberg BVOL surface)
  COB Comdty   — ATM 2M implied vol
  COC Comdty   — ATM 3M implied vol

Risk-free rate (BDP, PX_LAST):
  USOSFR1Z BGN Curncy  — 1M SOFR OIS mid
  Converted: ACT/360 simple → ACT/365 continuous

Historical prices for correlation (BDH, PX_LAST):
  CO1 Comdty, CO2 Comdty, CO3 Comdty
  Rolling window = corr_lookback business days

Bloomberg connectivity
──────────────────────
Tries xbbg first (lightweight Desktop API wrapper), then falls
back to pdblp.  Both require:
  - Bloomberg Desktop (DAPI) or B-Pipe running on localhost:8194
  - blpapi Python SDK  : pip install blpapi
  - xbbg               : pip install xbbg        (preferred)
  OR
  - pdblp              : pip install pdblp        (alternative)

Usage
─────
    from bbg_market_data import load_market_data_with_fallback
    mkt = load_market_data_with_fallback(
        valuation_date = "2026-03-10",
        bl_date        = "2026-04-10",
        fallback_data  = {...},
        verbose        = True,
    )
============================================================
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional

# ─────────────────────────────────────────────────────────────
# Bloomberg ticker / field constants
# ─────────────────────────────────────────────────────────────

_BBG_FUTURES = {
    "CO1": "CO1 Comdty",
    "CO2": "CO2 Comdty",
    "CO3": "CO3 Comdty",
}

_BBG_IVOL = {
    "CO1": "COA Comdty",   # 1M ATM implied vol from BVOL surface
    "CO2": "COB Comdty",   # 2M ATM implied vol
    "CO3": "COC Comdty",   # 3M ATM implied vol
}

_BBG_RATE        = "USOSFR1Z BGN Curncy"   # 1M SOFR OIS mid
_FIELD_PX        = "PX_LAST"
_FIELD_IVOL      = "IVOL_MID"
_FIELD_IVOL_FB   = "HIST_VOL_260D"         # fallback if IVOL_MID N/A


# ─────────────────────────────────────────────────────────────
# 1.  Bloomberg session wrapper  (xbbg ↔ pdblp abstraction)
# ─────────────────────────────────────────────────────────────

class _BloombergSession:
    """
    Thin adapter that exposes bdp() and bdh() regardless of whether
    xbbg or pdblp is installed.
    """

    def __init__(self, host: str = "localhost", port: int = 8194):
        self._conn = None
        self._lib  = None
        self._port = port
        self._connect()

    # ── connection ────────────────────────────────────────────

    def _connect(self):
        # 1. Try xbbg (no explicit session management needed)
        try:
            from xbbg import blp as xblp
            probe = xblp.bdp("CO1 Comdty", "PX_LAST")
            if probe is not None and not probe.empty:
                self._conn = xblp
                self._lib  = "xbbg"
                return
        except Exception:
            pass

        # 2. Try pdblp
        try:
            import pdblp
            con = pdblp.BCon(debug=False, port=self._port, timeout=10_000)
            con.start()
            self._conn = con
            self._lib  = "pdblp"
            return
        except Exception:
            pass

        raise ConnectionError(
            "Cannot reach Bloomberg. "
            "Ensure Bloomberg Desktop (DAPI) or B-Pipe is running on "
            f"localhost:{self._port}, and that blpapi + xbbg or pdblp "
            "are installed (pip install xbbg  OR  pip install pdblp)."
        )

    # ── BDP ───────────────────────────────────────────────────

    def bdp(
        self,
        tickers : list[str],
        fields  : list[str],
        as_of   : Optional[str] = None,   # "YYYYMMDD"
    ) -> pd.DataFrame:
        """
        Reference data (point-in-time).
        Returns DataFrame indexed by TICKER (upper-case), columns = fields.
        as_of = None  → live quote; as_of = "YYYYMMDD" → EOD snapshot.
        """
        overrides = {}
        if as_of:
            overrides["PRICING_DATE"] = as_of

        if self._lib == "xbbg":
            from xbbg import blp
            kwargs = {}
            if overrides:
                kwargs["ovrds"] = list(overrides.items())
            df = blp.bdp(tickers, fields, **kwargs)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            df.index   = [str(i).upper() for i in df.index]
            df.columns = [str(c).upper() for c in df.columns]
            return df
        else:
            ovrds = list(overrides.items()) if overrides else []
            df    = self._conn.bdp(tickers, fields, ovrds=ovrds)
            df.index   = [str(i).upper() for i in df.index]
            df.columns = [str(c).upper() for c in df.columns]
            return df

    # ── BDH ───────────────────────────────────────────────────

    def bdh(
        self,
        tickers    : list[str],
        fields     : list[str],
        start_date : str,   # "YYYYMMDD"
        end_date   : str,   # "YYYYMMDD"
    ) -> pd.DataFrame:
        """
        Historical time series.
        Returns wide DataFrame: DatetimeIndex, columns = tickers (upper).
        """
        if self._lib == "xbbg":
            from xbbg import blp
            df = blp.bdh(tickers, fields, start_date, end_date)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(-1)   # drop field level
            df.columns = [str(c).upper() for c in df.columns]
            df.index   = pd.to_datetime(df.index)
            return df
        else:
            df = self._conn.bdh(tickers, fields, start_date, end_date)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(-1)
            df.columns = [str(c).upper() for c in df.columns]
            df.index   = pd.to_datetime(df.index)
            return df

    def close(self):
        if self._lib == "pdblp" and self._conn:
            try:
                self._conn.stop()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# 2.  Helpers
# ─────────────────────────────────────────────────────────────

def _to_bbg_date(date_str: str) -> str:
    """YYYY-MM-DD → YYYYMMDD."""
    return date_str.replace("-", "")


def _get_scalar(df: pd.DataFrame, ticker: str, field: str) -> float:
    """Robustly extract a scalar from a BDP result DataFrame."""
    ticker_up = ticker.upper()
    field_up  = field.upper()
    # Try index lookup first
    if ticker_up in df.index:
        row = df.loc[ticker_up]
    else:
        # Fall back to positional
        row = df.iloc[0]
    val = row[field_up] if field_up in row.index else row.iloc[0]
    return float(val)


# ─────────────────────────────────────────────────────────────
# 3.  Roll-aware tenor mapping
# ─────────────────────────────────────────────────────────────

def _resolve_tenor_mapping(
    session        : _BloombergSession,
    valuation_date : str,
    verbose        : bool = False,
) -> dict:
    """
    Determine the CO1/CO2/CO3 mapping to M-1/M/M+1 on valuation_date,
    accounting for the ICE Brent futures roll.

    ICE Brent rolls roughly 3 business days before the last trading
    day of the month prior to delivery.

    Returns
    -------
    dict with keys 'M-1', 'M', 'M+1' → Bloomberg ticker string,
    plus optional '_m1_via_bdh': True if M-1 must be sourced via BDH.
    """
    bbg_date = _to_bbg_date(valuation_date)
    val_dt   = pd.Timestamp(valuation_date)

    try:
        # Pull notice/expiry date of the current CO1 contract
        df_meta = session.bdp(
            ["CO1 Comdty"],
            ["FUT_NOTICE_FIRST", "FUT_LAST_TRADE_DT"],
            as_of=bbg_date,
        )
        notice_raw = (
            df_meta.loc["CO1 COMDTY", "FUT_NOTICE_FIRST"]
            if "CO1 COMDTY" in df_meta.index
            else df_meta.iloc[0]["FUT_NOTICE_FIRST"]
        )
        notice_dt = pd.to_datetime(notice_raw)

        if val_dt >= notice_dt:
            # CO1 has already rolled to the new front month
            # M-1 is the just-expired contract → source from BDH
            if verbose:
                print(f"  [BBG] CO1 rolled (notice={notice_dt.date()}) → "
                      f"M-1 via BDH, M=CO1, M+1=CO2")
            return {
                "M-1"         : "__BDH__",
                "M"           : "CO1 Comdty",
                "M+1"         : "CO2 Comdty",
                "_m1_via_bdh" : True,
            }
        else:
            if verbose:
                print(f"  [BBG] Pre-roll (notice={notice_dt.date()}) → "
                      f"M-1=CO1, M=CO2, M+1=CO3")
            return {
                "M-1"         : "CO1 Comdty",
                "M"           : "CO2 Comdty",
                "M+1"         : "CO3 Comdty",
                "_m1_via_bdh" : False,
            }
    except Exception as e:
        # If metadata unavailable, default to pre-roll mapping
        if verbose:
            print(f"  [BBG] Roll date lookup failed ({e}); defaulting pre-roll mapping")
        return {
            "M-1"         : "CO1 Comdty",
            "M"           : "CO2 Comdty",
            "M+1"         : "CO3 Comdty",
            "_m1_via_bdh" : False,
        }


# ─────────────────────────────────────────────────────────────
# ICE Brent futures month codes
# ─────────────────────────────────────────────────────────────

_BRENT_MONTH_CODES = {
    1: "F",  2: "G",  3: "H",  4: "J",  5: "K",  6: "M",
    7: "N",  8: "Q",  9: "U", 10: "V", 11: "X", 12: "Z",
}


def _bl_date_to_tickers(bl_date: str) -> dict[str, str]:
    """
    Derive specific ICE Brent contract tickers for M-1, M, M+1
    from the Bill-of-Lading date.

    M   = delivery month of bl_date  (e.g. bl_date in Apr 2026 → COJ26)
    M-1 = one calendar month prior   (e.g. Mar 2026 → COH26)
    M+1 = one calendar month after   (e.g. May 2026 → COK26)

    Ticker format: CO{month_code}{2-digit year}  (e.g. COJ26)
    2-digit year avoids decade-boundary ambiguity (2030 → "30" not "0").
    """
    bl_dt  = pd.Timestamp(bl_date)
    m_dt   = bl_dt.replace(day=1)
    m1_dt  = m_dt - pd.DateOffset(months=1)
    mp1_dt = m_dt + pd.DateOffset(months=1)

    def _to_ticker(dt: pd.Timestamp) -> str:
        code = _BRENT_MONTH_CODES[dt.month]
        year = f"{dt.year % 100:02d}"   # 2026 → "26", 2030 → "30"
        return f"CO{code}{year} Comdty"

    return {
        "M-1": _to_ticker(m1_dt),
        "M"  : _to_ticker(m_dt),
        "M+1": _to_ticker(mp1_dt),
    }


# ─────────────────────────────────────────────────────────────
# 4.  Forward price fetch
# ─────────────────────────────────────────────────────────────

def fetch_forward_prices(
    session        : _BloombergSession,
    valuation_date : str,
    bl_date        : str,
    verbose        : bool = False,
) -> dict[str, float]:
    """
    Fetch M-1, M, M+1 ICE Brent forward prices from Bloomberg
    as of valuation_date, using specific contract tickers derived
    from bl_date.

    e.g. bl_date = "2026-04-10"  →  M-1=COH26, M=COJ26, M+1=COK26

    Uses BDP with PRICING_DATE override for point-in-time prices.
    No BDH fallback needed: all three contracts are live specific tickers.
    """
    bbg_date = _to_bbg_date(valuation_date)
    mapping  = _bl_date_to_tickers(bl_date)
    all_tickers = [mapping["M-1"], mapping["M"], mapping["M+1"]]

    if verbose:
        print(f"  [BBG] bl_date={bl_date} → contract tickers:")
        for lbl, t in mapping.items():
            print(f"        {lbl}: {t}")

    df_px = session.bdh(all_tickers, [_FIELD_PX], start_date=bbg_date, end_date=bbg_date)

    F = {}
    for label, ticker in mapping.items():
        F[label] = _get_scalar(df_px, ticker, _FIELD_PX)

    if verbose:
        print(f"  [BBG] Forward prices ({valuation_date}):")
        for lbl, p in F.items():
            print(f"        {lbl} ({mapping[lbl]}): {p:.3f} $/bbl")

    return F

# ─────────────────────────────────────────────────────────────
# 5.  Implied vol fetch
# ─────────────────────────────────────────────────────────────

def fetch_implied_vols(
    session        : _BloombergSession,
    valuation_date : str,
    verbose        : bool = False,
) -> dict[str, float]:
    """
    Fetch ATM implied vols for M-1, M, M+1 tenors.

    Primary  : IVOL_MID on COA/COB/COC Comdty  (Bloomberg BVOL surface)
    Fallback : HIST_VOL_260D on CO1/CO2/CO3     (historical realised vol)

    Bloomberg returns vols as percent (e.g. 28.5 = 28.5%).
    We convert to decimal (0.285).
    """
    bbg_date   = _to_bbg_date(valuation_date)
    iv_tickers = ["COA Comdty", "COB Comdty", "COC Comdty"]
    hv_tickers = ["CO1 Comdty", "CO2 Comdty", "CO3 Comdty"]
    labels     = ["M-1", "M", "M+1"]
    sigma      = {}

    # ── Primary: IVOL_MID ────────────────────────────────────
    try:
        df_iv = session.bdp(iv_tickers, [_FIELD_IVOL], as_of=bbg_date)
        for i, label in enumerate(labels):
            v = _get_scalar(df_iv, iv_tickers[i], _FIELD_IVOL)
            sigma[label] = v / 100.0 if v > 1.0 else v    # % → decimal

        if verbose:
            print(f"  [BBG] ATM implied vols IVOL_MID ({valuation_date}):")
            for lbl, v in sigma.items():
                print(f"        {lbl}: {v:.2%}")
        return sigma

    except Exception as e_iv:
        if verbose:
            print(f"  [BBG] IVOL_MID unavailable ({e_iv}) → using HIST_VOL_260D")

    # ── Fallback: HIST_VOL_260D ──────────────────────────────
    df_hv = session.bdp(hv_tickers, [_FIELD_IVOL_FB], as_of=bbg_date)
    for i, label in enumerate(labels):
        v = _get_scalar(df_hv, hv_tickers[i], _FIELD_IVOL_FB)
        sigma[label] = v / 100.0 if v > 1.0 else v

    if verbose:
        print(f"  [BBG] Historical vols HIST_VOL_260D ({valuation_date}):")
        for lbl, v in sigma.items():
            print(f"        {lbl}: {v:.2%}")

    return sigma


# ─────────────────────────────────────────────────────────────
# 6.  Correlation fetch (rolling BDH)
# ─────────────────────────────────────────────────────────────

def fetch_correlations(
    session        : _BloombergSession,
    valuation_date : str,
    lookback_bd    : int  = 90,
    verbose        : bool = False,
) -> dict[tuple[str, str], float]:
    """
    Estimate pairwise log-return correlations from BDH history.

    Pulls `lookback_bd` + 15 business days of daily PX_LAST for
    CO1/CO2/CO3, then computes rolling Pearson correlation of
    log daily returns over the last `lookback_bd` observations.
    """
    val_dt   = pd.Timestamp(valuation_date)
    start_dt = (val_dt - pd.tseries.offsets.BusinessDay(lookback_bd + 15))
    bbg_s    = start_dt.strftime("%Y%m%d")
    bbg_e    = _to_bbg_date(valuation_date)

    tickers  = ["CO1 Comdty", "CO2 Comdty", "CO3 Comdty"]
    df_hist  = session.bdh(tickers, [_FIELD_PX], bbg_s, bbg_e)
    df_hist  = df_hist.ffill().dropna()

    if df_hist.shape[0] < 10:
        raise ValueError(
            f"Insufficient history: got {df_hist.shape[0]} rows, need ≥ 10 "
            f"for correlation estimation."
        )

    # Use exactly the last `lookback_bd` observations
    df_hist  = df_hist.tail(lookback_bd)
    log_ret  = np.log(df_hist / df_hist.shift(1)).dropna()
    corr_mat = log_ret.corr().values   # 3 × 3 numpy array

    labels = ["M-1", "M", "M+1"]
    corr   = {}
    for i in range(3):
        for j in range(i + 1, 3):
            rho = float(np.clip(corr_mat[i, j], -0.9999, 0.9999))
            corr[(labels[i], labels[j])] = rho

    if verbose:
        print(f"  [BBG] {lookback_bd}bd log-return correlations ({valuation_date}):")
        for (a, b), r in corr.items():
            print(f"        ρ({a},{b}) = {r:.4f}")

    return corr


# ─────────────────────────────────────────────────────────────
# 7.  Risk-free rate fetch
# ─────────────────────────────────────────────────────────────

def fetch_risk_free_rate(
    session        : _BloombergSession,
    valuation_date : str,
    verbose        : bool = False,
) -> float:
    """
    Fetch 1M SOFR OIS rate from Bloomberg (USOSFR1Z BGN Curncy).
    Bloomberg PX_LAST is in percent, e.g. 5.30 = 5.30% p.a. ACT/360.

    Conversion chain:
        % p.a. ACT/360 simple  →  decimal  →  continuous ACT/365
    """
    bbg_date = _to_bbg_date(valuation_date)
    default_r = 0.05

    try:
        df    = session.bdp([_BBG_RATE], [_FIELD_PX], as_of=bbg_date)
        r_pct = _get_scalar(df, _BBG_RATE, _FIELD_PX)     # e.g. 5.30
        r_dec = r_pct / 100.0                               # 0.0530

        # ACT/360 simple → ACT/365 continuous
        # r_cont = ln(1 + r_simple_360 * 30/360) * 365/30
        r_cont = np.log(1.0 + r_dec * 30.0 / 360.0) * 365.0 / 30.0

        if verbose:
            print(f"  [BBG] 1M SOFR OIS ({valuation_date}): "
                  f"{r_pct:.4f}% → {r_cont:.6f} cont. ACT/365")
        return r_cont

    except Exception as e:
        if verbose:
            print(f"  [BBG] Rate fetch failed ({e}); using default r={default_r}")
        return default_r


# ─────────────────────────────────────────────────────────────
# 8.  Master public API
# ─────────────────────────────────────────────────────────────

def load_bloomberg_market_data(
    valuation_date : str,
    bl_date        : str,
    corr_lookback  : int  = 90,
    verbose        : bool = False,
) -> "BrentMarketData":
    """
    Connect to Bloomberg and fetch all market inputs needed by the
    Dated Brent Chooser Option pricer.

    Parameters
    ----------
    valuation_date : str   Pricing date    YYYY-MM-DD
    bl_date        : str   B/L date        YYYY-MM-DD
    corr_lookback  : int   Business days for rolling correlation window
    verbose        : bool  Print Bloomberg query diagnostics

    Returns
    -------
    BrentMarketData populated with live Bloomberg data.
    """
    # Deferred import to avoid circular dependency with main module
    from chooser_option_brent import BrentMarketData

    if verbose:
        print(f"\n  [BBG] Initiating Bloomberg connection...")

    session = _BloombergSession()
    if verbose:
        print(f"  [BBG] Connected via {session._lib}")

    try:
        F     = fetch_forward_prices(session, valuation_date, bl_date, verbose)
        sigma = fetch_implied_vols  (session, valuation_date, verbose)
        corr  = fetch_correlations  (session, valuation_date, corr_lookback, verbose)
        r     = fetch_risk_free_rate(session, valuation_date, verbose)
    finally:
        session.close()

    return BrentMarketData(
        valuation_date = valuation_date,
        bl_date        = bl_date,
        F              = F,
        sigma          = sigma,
        corr           = corr,
        r              = r,
    )


def load_market_data_with_fallback(
    valuation_date : str,
    bl_date        : str,
    fallback_data  : dict,
    corr_lookback  : int  = 90,
    verbose        : bool = True,
) -> "BrentMarketData":
    """
    Attempt to load live Bloomberg market data; if Bloomberg is
    unreachable or any error occurs, fall back to hardcoded values
    supplied by the caller and emit a UserWarning.

    Parameters
    ----------
    valuation_date : str   YYYY-MM-DD  pricing snapshot date
    bl_date        : str   YYYY-MM-DD  Bill-of-Lading date
    fallback_data  : dict  keys: 'F', 'sigma', 'corr', 'r'
                           (matches BrentMarketData constructor kwargs)
    corr_lookback  : int   business days for Bloomberg correlation window
    verbose        : bool  print progress / diagnostics

    Returns
    -------
    BrentMarketData — live from Bloomberg when available, else fallback.
    """
    from chooser_option_brent import BrentMarketData

    try:
        mkt = load_bloomberg_market_data(
            valuation_date = valuation_date,
            bl_date        = bl_date,
            corr_lookback  = corr_lookback,
            verbose        = verbose,
        )
        if verbose:
            print("  [BBG] ✓ All market data loaded from Bloomberg")
        return mkt

    except Exception as exc:
        warnings.warn(
            f"\n{'!'*60}\n"
            f"Bloomberg connection failed: {exc}\n"
            f"Using FALLBACK hardcoded market data.\n"
            f"Results are illustrative only — NOT for trading.\n"
            f"{'!'*60}",
            UserWarning,
            stacklevel=2,
        )
        if verbose:
            print(f"\n  [FALLBACK] Bloomberg unavailable: {exc}")
            print("  [FALLBACK] Using hardcoded illustrative market data")

        return BrentMarketData(
            valuation_date = valuation_date,
            bl_date        = bl_date,
            **fallback_data,
        )


def load_historical_snapshot(
    as_of_date    : str,
    bl_date       : str,
    corr_lookback : int  = 90,
    verbose       : bool = False,
) -> "BrentMarketData":
    """
    Convenience wrapper for backtesting: load a historical Bloomberg
    snapshot using BDP PRICING_DATE override.

    Parameters
    ----------
    as_of_date    : YYYY-MM-DD  historical pricing date
    bl_date       : YYYY-MM-DD  hypothetical B/L date
    """
    return load_bloomberg_market_data(
        valuation_date = as_of_date,
        bl_date        = bl_date,
        corr_lookback  = corr_lookback,
        verbose        = verbose,
    )


# ─────────────────────────────────────────────────────────────
# Standalone connectivity test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    date = sys.argv[1] if len(sys.argv) > 1 else "2026-03-09"
    bl   = sys.argv[2] if len(sys.argv) > 2 else "2026-04-10"

    _FALLBACK = {
        "F"    : {"M-1": 82.50, "M": 82.20, "M+1": 81.90},
        "sigma": {"M-1": 0.280, "M": 0.295, "M+1": 0.310},
        "corr" : {("M-1","M"): 0.970, ("M-1","M+1"): 0.930, ("M","M+1"): 0.980},
        "r"    : 0.05,
    }

    print(f"\nBloomberg connectivity test  ({date})")
    print("─" * 50)

    mkt = load_market_data_with_fallback(
        valuation_date = date,
        bl_date        = bl,
        fallback_data  = _FALLBACK,
        verbose        = True,
    )

    print("\nMarket data loaded:")
    print(f"  Forwards : {mkt.F}")
    print(f"  Vols     : {mkt.sigma}")
    print(f"  Correls  : {mkt.corr}")
    print(f"  Rate     : {mkt.r:.6f}")
    print(f"  T_ex     : {mkt.T_ex:.4f} yrs")
