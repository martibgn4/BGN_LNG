"""
============================================================
features/technical.py  --  Price & volume derived signals
============================================================
Every function here takes a *single security's* daily history,
sorted ascending by date, and returns a Series aligned to it.
All of them are strictly backward-looking: the value at row t uses
data up to and including t, never after. That discipline is what
lets the backtest be trusted.

Signal families implemented
---------------------------
Momentum
  mom_12_1        Jegadeesh-Titman workhorse: 12-month return skipping
                  the most recent month, because the last month shows
                  short-horizon *reversal* that contaminates momentum.
  mom_6_1         Faster variant; decays quicker but adapts to regime turns.
  mom_1m          Short-term reversal -- expected to predict with a NEGATIVE
                  sign, and included precisely for that.
  resid_mom       Residual momentum (Blitz-Huij-Martens). Momentum of the
                  part of returns NOT explained by the market. Raw momentum
                  is heavily a bet on high-beta names, so it crashes when
                  beta rallies; residual momentum strips that out and has
                  historically shown a materially higher Sharpe.
  fip             Frog-in-the-Pan (Da-Gurun-Warachka). Same total return
                  delivered smoothly (many small up days) is under-reacted
                  to and continues; delivered in one jump it is already
                  priced. This measures the *path*, not the magnitude.
  pct_52w_high    George-Hwang. Proximity to the 52-week high; anchoring
                  makes investors under-react near highs.

Risk / low-volatility
  vol_60, vol_252 Realised volatility -- enters with a negative sign
                  (low-vol anomaly) and drives inverse-vol weighting.
  downside_dev    Semi-deviation. Penalises only downside, unlike vol.
  max_drawdown    Worst peak-to-trough over the window.
  beta            Rolling market beta.
  ivol            Idiosyncratic vol -- residual sd from the market model.

Volume / liquidity
  dollar_vol      Liquidity screen and a tradability constraint.
  vol_trend       Volume expanding vs its own base -- confirms a move.
  amihud          Amihud illiquidity: |return| per dollar traded. A price
                  that moves a lot on little volume is fragile.
  vol_price_corr  Are up-days the high-volume days? Accumulation signal.

Trend / oscillator
  rsi_14, macd_hist, dist_ma_50/200, bollinger_z, adx_14
============================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _safe_div(a, b):
    return np.where((b == 0) | ~np.isfinite(b), np.nan, a / b)


def log_returns(px: pd.Series) -> pd.Series:
    return np.log(px / px.shift(1))


# ----------------------------------------------------------------------
# momentum
# ----------------------------------------------------------------------
def momentum(px: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Total return over ``lookback`` days, skipping the most recent ``skip``.

    The skip is not a detail -- it is the difference between a momentum
    signal and a reversal signal. Without it the last month's mean reversion
    cancels most of the 12-month continuation.
    """
    return px.shift(skip) / px.shift(lookback) - 1.0


def residual_momentum(
    px: pd.Series,
    mkt_px: pd.Series,
    lookback: int = 252,
    skip: int = 21,
    min_obs: int = 120,
) -> pd.Series:
    """Momentum of market-adjusted returns.

    Regress the stock's daily returns on the market over a rolling window,
    then cumulate the residuals. What remains is stock-specific drift with
    the beta bet removed -- historically a cleaner and less crash-prone
    signal than raw momentum, since it does not implicitly load on beta.

    Implemented with rolling covariance rather than a loop of regressions:
    beta_t = cov(r, m) / var(m) over the window, residual = r - beta_t * m.
    """
    r = log_returns(px)
    m = log_returns(mkt_px).reindex(r.index)
    cov = r.rolling(lookback, min_periods=min_obs).cov(m)
    var = m.rolling(lookback, min_periods=min_obs).var()
    beta = cov / var.replace(0, np.nan)
    resid = r - beta * m
    # Cumulative residual over the window, skipping the last month.
    cum = resid.rolling(lookback - skip, min_periods=min_obs).sum()
    return cum.shift(skip)


def frog_in_the_pan(px: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Information discreteness -- how *smoothly* the momentum was delivered.

    ID = sign(cumulative return) x (%negative days - %positive days)

    Negative ID means a steady, continuous path (many small moves in the
    winning direction); positive ID means a jumpy path dominated by a few
    large days. Continuous information is absorbed gradually by investors,
    so those trends persist; discrete information is repriced at once.

    Used as an *interaction*: it says which momentum names to trust, and is
    most valuable multiplied against the momentum score itself.
    """
    r = px.pct_change(fill_method=None)
    window = lookback - skip
    cum = px.shift(skip) / px.shift(lookback) - 1.0
    pos = (r > 0).rolling(window, min_periods=window // 2).mean().shift(skip)
    neg = (r < 0).rolling(window, min_periods=window // 2).mean().shift(skip)
    return np.sign(cum) * (neg - pos)


def pct_of_52w_high(px: pd.Series, window: int = 252) -> pd.Series:
    """Current price as a fraction of the trailing 52-week high."""
    high = px.rolling(window, min_periods=window // 2).max()
    return px / high


# ----------------------------------------------------------------------
# risk
# ----------------------------------------------------------------------
def realised_vol(px: pd.Series, window: int = 60, annualise: bool = True) -> pd.Series:
    r = log_returns(px)
    v = r.rolling(window, min_periods=max(10, window // 2)).std()
    return v * np.sqrt(TRADING_DAYS) if annualise else v


def downside_deviation(px: pd.Series, window: int = 60) -> pd.Series:
    """Volatility of negative returns only."""
    r = log_returns(px)
    neg = r.where(r < 0, 0.0)
    return neg.rolling(window, min_periods=max(10, window // 2)).std() * np.sqrt(TRADING_DAYS)


def rolling_max_drawdown(px: pd.Series, window: int = 252) -> pd.Series:
    peak = px.rolling(window, min_periods=window // 2).max()
    return px / peak - 1.0


def rolling_beta(px: pd.Series, mkt_px: pd.Series, window: int = 252, min_obs: int = 120) -> pd.Series:
    r = log_returns(px)
    m = log_returns(mkt_px).reindex(r.index)
    cov = r.rolling(window, min_periods=min_obs).cov(m)
    var = m.rolling(window, min_periods=min_obs).var()
    return cov / var.replace(0, np.nan)


def idiosyncratic_vol(px: pd.Series, mkt_px: pd.Series, window: int = 252, min_obs: int = 120) -> pd.Series:
    """Residual volatility from a rolling market model.

    Low idiosyncratic vol is one of the most persistent anomalies in equities
    and is the single best lever for the "minimise volatility" objective,
    because idio risk is what actually survives into a 5-name portfolio.
    """
    r = log_returns(px)
    m = log_returns(mkt_px).reindex(r.index)
    cov = r.rolling(window, min_periods=min_obs).cov(m)
    var = m.rolling(window, min_periods=min_obs).var()
    beta = cov / var.replace(0, np.nan)
    resid = r - beta * m
    return resid.rolling(window, min_periods=min_obs).std() * np.sqrt(TRADING_DAYS)


# ----------------------------------------------------------------------
# volume / liquidity
# ----------------------------------------------------------------------
def dollar_volume(px: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    return (px * volume).rolling(window, min_periods=max(5, window // 2)).mean()


def volume_trend(volume: pd.Series, fast: int = 20, slow: int = 120) -> pd.Series:
    """Short-run volume relative to its own long-run base.

    A breakout on expanding volume is materially more reliable than the same
    price move on thin trade, so this is used to confirm momentum rather than
    to generate a signal alone.
    """
    f = volume.rolling(fast, min_periods=fast // 2).mean()
    s = volume.rolling(slow, min_periods=slow // 2).mean()
    return f / s.replace(0, np.nan)


def amihud_illiquidity(px: pd.Series, volume: pd.Series, window: int = 60) -> pd.Series:
    """Average |return| per dollar of volume, scaled.

    High Amihud = price moves a lot on little volume = fragile and expensive
    to trade. Doubles as a risk signal and a tradability filter.
    """
    r = px.pct_change(fill_method=None).abs()
    dv = (px * volume).replace(0, np.nan)
    return (r / dv).rolling(window, min_periods=window // 2).mean() * 1e9


def volume_price_corr(px: pd.Series, volume: pd.Series, window: int = 60) -> pd.Series:
    """Correlation of daily returns with volume -- accumulation vs distribution."""
    r = px.pct_change(fill_method=None)
    v = volume.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return r.rolling(window, min_periods=window // 2).corr(v)


# ----------------------------------------------------------------------
# trend / oscillators
# ----------------------------------------------------------------------
def rsi(px: pd.Series, window: int = 14) -> pd.Series:
    delta = px.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd_histogram(px: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram, normalised by price so it is comparable across names."""
    ema_f = px.ewm(span=fast, adjust=False).mean()
    ema_s = px.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    sig = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig) / px


def distance_from_ma(px: pd.Series, window: int = 200) -> pd.Series:
    ma = px.rolling(window, min_periods=window // 2).mean()
    return px / ma - 1.0


def bollinger_z(px: pd.Series, window: int = 20, ) -> pd.Series:
    """Position within the Bollinger band, in standard deviations."""
    ma = px.rolling(window, min_periods=window // 2).mean()
    sd = px.rolling(window, min_periods=window // 2).std()
    return (px - ma) / sd.replace(0, np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index -- trend *strength* regardless of direction.

    Used as a conditioner: momentum signals are far more reliable in names
    that are genuinely trending than in ones chopping sideways.
    """
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / window, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------
def compute_technical_features(df: pd.DataFrame, mkt: pd.Series) -> pd.DataFrame:
    """Compute every technical feature for one security.

    Parameters
    ----------
    df
        Daily rows for a single security with columns ``date, PX_LAST,
        PX_HIGH, PX_LOW, PX_VOLUME`` (extras ignored), sorted by date.
    mkt
        Benchmark close series indexed by date, used for beta/residual work.
    """
    d = df.sort_values("date").set_index("date")
    px = pd.to_numeric(d["PX_LAST"], errors="coerce")
    vol = pd.to_numeric(d.get("PX_VOLUME"), errors="coerce")
    high = pd.to_numeric(d.get("PX_HIGH", px), errors="coerce").fillna(px)
    low = pd.to_numeric(d.get("PX_LOW", px), errors="coerce").fillna(px)
    mkt_al = mkt.reindex(px.index).ffill()

    out = pd.DataFrame(index=px.index)
    # momentum family
    out["mom_12_1"] = momentum(px, 252, 21)
    out["mom_6_1"] = momentum(px, 126, 21)
    out["mom_3_1"] = momentum(px, 63, 21)
    out["mom_1m"] = px / px.shift(21) - 1.0          # short-term reversal
    out["mom_1w"] = px / px.shift(5) - 1.0
    out["resid_mom"] = residual_momentum(px, mkt_al, 252, 21)
    out["fip"] = frog_in_the_pan(px, 252, 21)
    out["pct_52w_high"] = pct_of_52w_high(px, 252)

    # risk family
    out["vol_60"] = realised_vol(px, 60)
    out["vol_252"] = realised_vol(px, 252)
    out["downside_dev"] = downside_deviation(px, 60)
    out["max_dd_252"] = rolling_max_drawdown(px, 252)
    out["beta"] = rolling_beta(px, mkt_al, 252)
    out["ivol"] = idiosyncratic_vol(px, mkt_al, 252)
    # Vol-of-vol: unstable risk is itself a risk signal.
    out["vol_ratio"] = out["vol_60"] / out["vol_252"].replace(0, np.nan)

    # volume family
    if vol is not None and vol.notna().any():
        out["dollar_vol"] = dollar_volume(px, vol, 20)
        out["vol_trend"] = volume_trend(vol, 20, 120)
        out["amihud"] = amihud_illiquidity(px, vol, 60)
        out["vol_px_corr"] = volume_price_corr(px, vol, 60)

    # trend / oscillator family
    out["rsi_14"] = rsi(px, 14)
    out["macd_hist"] = macd_histogram(px)
    out["dist_ma_50"] = distance_from_ma(px, 50)
    out["dist_ma_200"] = distance_from_ma(px, 200)
    out["bollinger_z"] = bollinger_z(px, 20)
    out["adx_14"] = adx(high, low, px, 14)

    # interaction: momentum you can trust (smooth path, real trend)
    out["mom_x_fip"] = out["mom_12_1"] * (-out["fip"])
    out["mom_x_adx"] = out["mom_12_1"] * (out["adx_14"] / 100.0)
    # risk-adjusted momentum -- return per unit of risk taken to get it
    out["mom_sharpe"] = out["mom_12_1"] / out["vol_252"].replace(0, np.nan)

    out = out.reset_index()
    out.insert(1, "security", df["security"].iloc[0])
    return out
