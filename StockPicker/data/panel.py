"""
============================================================
data/panel.py  --  Model-ready weekly panel assembly
============================================================
Joins prices, technicals, fundamentals, sector data and
point-in-time membership into one weekly cross-sectional panel,
then attaches the forward return the model is trained to rank.

The execution timeline (this is the part that must not be fudged)
-----------------------------------------------------------------
    Friday close t-1   ... all features are computed from data up to
                           and including this bar, and no later.
    Monday  t          ... rebalance. We buy at Monday's VWAP, which
                           is an achievable price for a retail order
                           worked through the day.
    Monday  t+1        ... exit at that day's VWAP.

    forward return = VWAP(t+1) / VWAP(t) - 1

Two deliberate choices:

* **Features are lagged one trading day** relative to the rebalance
  bar. Using Monday's own close to decide a Monday trade is the
  single most common look-ahead error in weekly equity backtests --
  it quietly assumes you can trade on information you only have
  after the close.

* **VWAP rather than close-to-close.** Close-to-close backtests
  overstate returns because they assume execution at a price you
  can observe but not reliably obtain. VWAP is both observable and
  attainable, and Bloomberg gives it to us directly via
  ``EQY_WEIGHTED_AVG_PX``.
============================================================
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..bbg.universe import METADATA_PATH, MEMBERSHIP_PATH
from ..config import (
    BENCHMARK_TICKER,
    ESTIMATES_DIR,
    ESTIMATE_FIELDS,
    OPTIONS_DIR,
    SECTOR_COL,
    STRATEGY,
)
from ..data.store import _read_dir, read_fundamentals, read_prices
from ..features.fundamental import compute_fundamental_features, compute_sector_features
from ..features.options import compute_options_features
from ..features.technical import compute_technical_features

log = logging.getLogger(__name__)


def load_membership() -> pd.DataFrame:
    m = pd.read_parquet(MEMBERSHIP_PATH)
    m["date"] = pd.to_datetime(m["date"])
    return m


def load_metadata() -> pd.DataFrame:
    if not METADATA_PATH.exists():
        return pd.DataFrame(columns=["security", SECTOR_COL])
    meta = pd.read_parquet(METADATA_PATH)
    if SECTOR_COL in meta.columns:
        meta[SECTOR_COL] = meta[SECTOR_COL].fillna("Unknown")
    return meta


def membership_mask(membership: pd.DataFrame, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Expand monthly snapshots onto the weekly rebalance grid.

    For each rebalance date we take the most recent snapshot at or before it,
    which is exactly the information set available on that morning.
    """
    snaps = np.sort(membership["date"].unique())
    rows = []
    for rd in rebal_dates:
        prior = snaps[snaps <= np.datetime64(rd)]
        if len(prior) == 0:
            continue
        snap = prior[-1]
        members = membership.loc[membership["date"] == snap, "security"]
        rows.append(pd.DataFrame({"date": rd, "security": members.values}))
    if not rows:
        return pd.DataFrame(columns=["date", "security"])
    return pd.concat(rows, ignore_index=True)


def build_panel(
    rebalance_freq: str | None = None,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    securities: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the full weekly panel with features and forward returns."""
    freq = rebalance_freq or STRATEGY.rebalance_day

    log.info("Loading cached prices ...")
    prices = read_prices(securities)
    if prices.empty:
        raise RuntimeError("No cached prices. Run: python -m StockPicker.scripts.ingest_history")

    membership = load_membership()
    meta = load_metadata()

    # ---- benchmark series for beta / residual momentum -------------------
    bench = prices[prices["security"] == BENCHMARK_TICKER]
    if bench.empty:
        raise RuntimeError(f"Benchmark {BENCHMARK_TICKER} missing from cache.")
    mkt = bench.set_index("date")["PX_LAST"].astype(float).sort_index()

    stock_prices = prices[prices["security"] != BENCHMARK_TICKER]
    all_secs = stock_prices["security"].unique()
    log.info("Computing technical features for %s securities ...", len(all_secs))

    # ---- technical features, per security --------------------------------
    tech_frames = []
    for i, (sec, grp) in enumerate(stock_prices.groupby("security", sort=False), 1):
        if len(grp) < 260:  # need ~1y for the momentum family to exist at all
            continue
        try:
            tech_frames.append(compute_technical_features(grp, mkt))
        except Exception as exc:  # noqa: BLE001
            log.warning("technical features failed for %s: %s", sec, exc)
        if i % 250 == 0:
            log.info("  %s/%s securities", i, len(all_secs))
    if not tech_frames:
        raise RuntimeError("No technical features computed -- is the price cache populated?")
    tech = pd.concat(tech_frames, ignore_index=True)

    # ---- rebalance grid --------------------------------------------------
    lo = pd.Timestamp(start) if start else tech["date"].min()
    hi = pd.Timestamp(end) if end else tech["date"].max()
    rebal = pd.date_range(lo, hi, freq=freq)
    log.info("Rebalance grid: %s dates (%s -> %s)", len(rebal), rebal.min().date(), rebal.max().date())

    # ---- execution prices: VWAP on the rebalance bar ---------------------
    px = stock_prices[["date", "security", "PX_LAST", "EQY_WEIGHTED_AVG_PX", "PX_VOLUME", "CUR_MKT_CAP"]].copy()
    px["exec_px"] = pd.to_numeric(px["EQY_WEIGHTED_AVG_PX"], errors="coerce")
    # Fall back to close where VWAP is unavailable (rare, older history).
    px["exec_px"] = px["exec_px"].fillna(pd.to_numeric(px["PX_LAST"], errors="coerce"))

    # Snap each rebalance date to the next available trading bar for that
    # security, so a holiday Monday rolls to Tuesday rather than silently
    # dropping the week.
    #
    # The alignment is done per rebalance date rather than by position, because
    # a security whose history starts mid-sample has every earlier rebalance
    # date collapse onto its first bar. Pairing by position there would stamp
    # a 2006 rebalance date onto a 2014 price. The 7-day tolerance is what
    # discards those: a bar more than a week after the rebalance date is not
    # that week's execution, it is a name that was not trading yet.
    rebal_s = pd.Series(rebal, name="rebal_date")
    exec_rows = []
    for sec, g in px.groupby("security", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        pos = g["date"].searchsorted(rebal_s.to_numpy(), side="left")
        ok = pos < len(g)
        if not ok.any():
            continue
        sub = g.iloc[pos[ok]].copy()
        sub["rebal_date"] = rebal_s[ok].to_numpy()
        # Drop rows where the located bar is too far after the rebalance date.
        gap = (sub["date"] - sub["rebal_date"]).dt.days
        sub = sub[(gap >= 0) & (gap <= 7)]
        if not sub.empty:
            exec_rows.append(sub)
    if not exec_rows:
        raise RuntimeError("No securities aligned to the rebalance grid.")
    execp = pd.concat(exec_rows, ignore_index=True)

    # ---- forward return: this bar's VWAP -> next rebalance bar's VWAP -----
    execp = execp.sort_values(["security", "rebal_date"])
    grp = execp.groupby("security")
    execp["next_exec_px"] = grp["exec_px"].shift(-1)
    execp["next_rebal"] = grp["rebal_date"].shift(-1)

    # Guard against a missing week silently becoming a two-week return. If the
    # next rebalance row is more than ~10 days away, this week's forward return
    # is not observable and must be NaN rather than a mislabelled longer one.
    gap_fwd = (execp["next_rebal"] - execp["rebal_date"]).dt.days
    execp["fwd_ret_1w"] = np.where(
        gap_fwd.between(1, 10),
        execp["next_exec_px"] / execp["exec_px"] - 1.0,
        np.nan,
    )

    # ---- terminal return for delistings ----------------------------------
    # A name that stops trading has no "next rebalance bar", but the money did
    # not simply stop moving: an acquisition pays a premium, a bankruptcy goes
    # to roughly zero. Leaving it NaN and defaulting to flat would quietly
    # erase exactly the outcomes that survivorship bias is about, so the final
    # week is marked to the last price the security ever printed.
    last_px = (
        stock_prices.sort_values("date")
        .groupby("security")
        .agg(last_px=("PX_LAST", "last"), last_date=("date", "last"))
    )
    execp = execp.merge(last_px, on="security", how="left")
    is_final = execp["next_exec_px"].isna()
    # Only treat it as a delisting if the security genuinely stops shortly
    # after -- not if we are simply at the right-hand edge of the sample.
    stops_soon = (execp["last_date"] - execp["rebal_date"]).dt.days.between(0, 30)
    terminal = is_final & stops_soon & (execp["last_date"] < stock_prices["date"].max() - pd.Timedelta(days=10))
    execp.loc[terminal, "fwd_ret_1w"] = (
        execp.loc[terminal, "last_px"] / execp.loc[terminal, "exec_px"] - 1.0
    )
    n_term = int(terminal.sum())
    if n_term:
        log.info("Marked %s terminal (delisting) weeks to last traded price.", n_term)

    # trailing 1w return (for sector momentum -- backward looking only).
    # Regrouped after the merge above, which reset the index.
    execp = execp.sort_values(["security", "rebal_date"])
    execp["trail_ret_1w"] = (
        execp["exec_px"] / execp.groupby("security")["exec_px"].shift(1) - 1.0
    )

    # ---- lag features by one trading day, then sample on rebalance dates --
    log.info("Aligning features (lagged 1 trading day) to rebalance dates ...")
    tech = tech.sort_values(["security", "date"])
    feat_cols = [c for c in tech.columns if c not in ("date", "security")]
    # shift(1) => the value carried into date t is the one known at t-1 close
    tech[feat_cols] = tech.groupby("security", sort=False)[feat_cols].shift(1)

    merged = pd.merge_asof(
        execp.sort_values("date")[
            ["date", "security", "rebal_date", "exec_px", "fwd_ret_1w", "trail_ret_1w",
             "PX_VOLUME", "PX_LAST", "CUR_MKT_CAP"]
        ],
        tech.sort_values("date"),
        on="date",
        by="security",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )

    # ---- fundamentals ----------------------------------------------------
    funda = read_fundamentals(securities)

    # Overlay the override-pulled estimates. The plain BEST_EPS in the
    # fundamentals cache only starts in 2021; the ESTIMATES_DIR copy was
    # pulled with BEST_FPERIOD_OVERRIDE="1BF" and runs the full sample, so it
    # replaces the sparse column wherever it has a value.
    est = _read_dir(ESTIMATES_DIR, securities)
    if not funda.empty and not est.empty:
        est = est.rename(columns={c: f"{c}__est" for c in est.columns if c not in ("date", "security")})
        funda = funda.merge(est, on=["date", "security"], how="left")
        for c in ESTIMATE_FIELDS:
            if f"{c}__est" in funda.columns:
                base = pd.to_numeric(funda.get(c), errors="coerce") if c in funda.columns else np.nan
                funda[c] = pd.to_numeric(funda[f"{c}__est"], errors="coerce").fillna(base)
        funda = funda.drop(columns=[c for c in funda.columns if c.endswith("__est")])
        log.info("Estimates overlaid: BEST_EPS non-null %.1f%%", funda["BEST_EPS"].notna().mean() * 100)

    if not funda.empty:
        log.info("Computing fundamental features for %s securities ...", funda["security"].nunique())
        f_frames = []
        px_by_sec = {s: g.set_index("date")["PX_LAST"].astype(float) for s, g in stock_prices.groupby("security")}
        for sec, grp in funda.groupby("security", sort=False):
            try:
                f_frames.append(compute_fundamental_features(grp, px_by_sec.get(sec)))
            except Exception as exc:  # noqa: BLE001
                log.warning("fundamental features failed for %s: %s", sec, exc)
        if f_frames:
            fund = pd.concat(f_frames, ignore_index=True).sort_values(["security", "date"])
            fcols = [c for c in fund.columns if c not in ("date", "security")]
            fund[fcols] = fund.groupby("security", sort=False)[fcols].shift(1)
            merged = pd.merge_asof(
                merged.sort_values("date"),
                fund.sort_values("date"),
                on="date", by="security", direction="backward",
                tolerance=pd.Timedelta(days=21),
            )

    # ---- options-implied -------------------------------------------------
    opts = _read_dir(OPTIONS_DIR, securities)
    if not opts.empty:
        log.info("Computing options features for %s securities ...", opts["security"].nunique())
        o_frames = []
        for sec, grp in opts.groupby("security", sort=False):
            try:
                o_frames.append(compute_options_features(grp))
            except Exception as exc:  # noqa: BLE001
                log.warning("options features failed for %s: %s", sec, exc)
        if o_frames:
            opt = pd.concat(o_frames, ignore_index=True).sort_values(["security", "date"])
            ocols = [c for c in opt.columns if c not in ("date", "security")]
            opt[ocols] = opt.groupby("security", sort=False)[ocols].shift(1)
            merged = pd.merge_asof(
                merged.sort_values("date"),
                opt.sort_values("date"),
                on="date", by="security", direction="backward",
                tolerance=pd.Timedelta(days=21),
            )

    # ---- membership mask + metadata --------------------------------------
    mask = membership_mask(membership, rebal).rename(columns={"date": "rebal_date"})
    mask["in_index"] = True
    merged = merged.merge(mask, on=["rebal_date", "security"], how="left")
    merged["in_index"] = merged["in_index"].notna()

    if not meta.empty:
        keep = ["security", SECTOR_COL, "NAME", "GICS_SECTOR_NAME"]
        keep = [c for c in keep if c in meta.columns]
        merged = merged.merge(meta[keep], on="security", how="left")
    if SECTOR_COL in merged.columns:
        merged[SECTOR_COL] = merged[SECTOR_COL].fillna("Unknown")

    # ---- promote the rebalance date to the panel's primary date ----------
    # From here on ``date`` means "the rebalance this row belongs to". The
    # underlying price-bar date is kept as ``px_date`` for auditing, so a
    # holiday roll can always be traced back to the bar actually used.
    merged = merged.rename(columns={"date": "px_date", "rebal_date": "date"})

    # ---- sector relative strength ----------------------------------------
    merged = compute_sector_features(
        merged, sector_col=SECTOR_COL, ret_col="trail_ret_1w"
    )

    log.info("Panel assembled: %s rows | %s dates | %s securities",
             len(merged), merged["date"].nunique(), merged["security"].nunique())
    return merged


def apply_filters(panel: pd.DataFrame, cfg=STRATEGY) -> pd.DataFrame:
    """Investability screen: index membership, liquidity, price, size.

    Applied *before* scoring so the model never learns to rank names it could
    not have bought, and so cross-sectional standardisation is computed over
    the tradable set rather than a wider one.
    """
    df = panel.copy()
    n0 = len(df)

    df = df[df["in_index"].fillna(False)]
    n1 = len(df)

    if "dollar_vol" in df.columns:
        df = df[df["dollar_vol"].fillna(0) >= cfg.min_dollar_volume]
    n2 = len(df)

    df = df[pd.to_numeric(df["PX_LAST"], errors="coerce").fillna(0) >= cfg.min_price]
    n3 = len(df)

    if "CUR_MKT_CAP" in df.columns:
        mc = pd.to_numeric(df["CUR_MKT_CAP"], errors="coerce")
        df = df[(mc.isna()) | (mc >= cfg.min_market_cap)]
    n4 = len(df)

    df = df[df["fwd_ret_1w"].notna() | (df["date"] == df["date"].max())]

    log.info(
        "Filters: %s -> in_index %s -> liquidity %s -> price %s -> mcap %s -> final %s",
        n0, n1, n2, n3, n4, len(df),
    )
    return df.reset_index(drop=True)
