from __future__ import annotations
import os

import spgci as ci


"""
Platts MOC — APAC LNG (JKM) Derivatives flow analytics
======================================================

Input : a pandas DataFrame named `all_MOC_data` with columns
        [product, update_time, price, order_quantity,
         buyer_mnemonic, seller_mnemonic, market_maker_mnemonic]

Core analytical idea
--------------------
`market_maker_mnemonic` identifies the PASSIVE side of each trade — the
participant whose order was already resting in the window. The counterparty
is therefore the AGGRESSOR:

    market_maker == seller  ->  a resting OFFER was LIFTED  ->  buy-initiated
    market_maker == buyer   ->  a resting BID was HIT       ->  sell-initiated

Net position tells you where risk ended up. Aggressor-signed flow tells you
who *wanted* it and paid up to get it. The two frequently disagree, and the
disagreement is the tradable information: a market maker can end the day
short simply because they were quoting, not because they have a view.

Everything below is grouped by trade date so it works across several MOC days.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Conventions — adjust to your contract specs
# --------------------------------------------------------------------------
LOT_MMBTU = 10_000        # JKM swap lot size (MMBtu)
CARGO_MMBTU = 3_400_000   # ~1 standard 160k cbm cargo, for intuition only

PRODUCT_MAP = {
    "Platts APAC LNG Derivatives":            "JKM_BASE",     # front contract
    "Platts APAC LNG Derivatives - NextDay":  "JKM_NEXT",     # next contract
    "Platts APAC LNG Derivatives Spr":        "JKM_SPREAD",   # NEXT - BASE
}

OUTRIGHTS = ["JKM_BASE", "JKM_NEXT"]

# Columns of the raw eWindow BOTE tape the analytics below need.
MOC_COLUMNS = ["product", "update_time", "price", "order_quantity",
               "buyer_mnemonic", "seller_mnemonic", "market_maker_mnemonic"]

# Platts (S&P Global CI) login. Set the environment variables to keep the
# credentials out of the source tree.
SPGCI_USERNAME = os.environ.get("SPGCI_USERNAME", "marti.fernandezreal@bgn-int.com")
SPGCI_PASSWORD = os.environ.get("SPGCI_PASSWORD", "Cacapipi1!")


def fetch_moc_trades(market: str = "Asia LNG Derivative") -> pd.DataFrame:
    """Pull the consummated MOC tape for `market` and return it prepared."""
    ci.set_credentials(SPGCI_USERNAME, SPGCI_PASSWORD)
    market_data = ci.EWindowMarketData()
    botes = market_data.get_botes(market=market, order_state="consummated")
    return prepare_moc_data(botes[MOC_COLUMNS])


# ==========================================================================
# 1. Preparation & feature engineering
# ==========================================================================
def prepare_moc_data(
        all_MOC_data: pd.DataFrame,
        lot_mmbtu: int = LOT_MMBTU,
        spread_tolerance_ms: int = 150,
) -> pd.DataFrame:
    """Clean, type, and enrich the raw MOC tape."""
    df = all_MOC_data.copy()

    # Drop the unnamed index column if it came through from CSV
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    df["update_time"] = pd.to_datetime(df["update_time"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["order_quantity"] = pd.to_numeric(df["order_quantity"], errors="coerce")
    df = df.dropna(subset=["price", "order_quantity", "update_time"])

    for c in ["buyer_mnemonic", "seller_mnemonic", "market_maker_mnemonic"]:
        df[c] = df[c].astype(str).str.strip().str.upper()

    df["contract"] = df["product"].map(PRODUCT_MAP).fillna(df["product"])
    df["is_spread"] = df["contract"] == "JKM_SPREAD"

    df["trade_date"] = df["update_time"].dt.date
    df["lots"] = df["order_quantity"]
    df["volume_mmbtu"] = df["lots"] * lot_mmbtu
    df["cargo_equiv"] = df["volume_mmbtu"] / CARGO_MMBTU
    # Notional is meaningless for the spread leg, so leave it NaN there
    df["notional_usd"] = np.where(
        df["is_spread"], np.nan, df["price"] * df["volume_mmbtu"]
    )

    # ---- Aggressor reconstruction -------------------------------------
    mm, byr, slr = (
        df["market_maker_mnemonic"],
        df["buyer_mnemonic"],
        df["seller_mnemonic"],
    )
    df["aggressor_side"] = np.select(
        [mm == slr, mm == byr],
        ["BUY", "SELL"],
        default="UNKNOWN",
    )
    df["aggressor"] = np.select(
        [df["aggressor_side"] == "BUY", df["aggressor_side"] == "SELL"],
        [byr, slr],
        default="UNKNOWN",
    )
    df["passive"] = np.select(
        [df["aggressor_side"] == "BUY", df["aggressor_side"] == "SELL"],
        [slr, byr],
        default=mm,
    )
    # +1 = trade printed on the offer, -1 = printed on the bid
    df["signed_lots"] = np.where(
        df["aggressor_side"] == "BUY", df["lots"],
        np.where(df["aggressor_side"] == "SELL", -df["lots"], 0),
    )
    # The passive quote level implied by each print
    df["implied_offer"] = np.where(df["aggressor_side"] == "BUY", df["price"], np.nan)
    df["implied_bid"] = np.where(df["aggressor_side"] == "SELL", df["price"], np.nan)

    df = df.sort_values("update_time").reset_index(drop=True)
    df = _tag_spread_legs(df, tolerance_ms=spread_tolerance_ms)
    return df


def _tag_spread_legs(df: pd.DataFrame, tolerance_ms: int = 150) -> pd.DataFrame:
    """
    A spread execution prints three near-simultaneous rows (spread + two
    outright legs). Flag the outright legs so they can be excluded from
    'outright conviction' measures — they are hedge mechanics, not a view
    on flat price.
    """
    df = df.copy()
    df["is_spread_leg"] = False
    if not df["is_spread"].any():
        return df

    tol = pd.Timedelta(milliseconds=tolerance_ms)
    times = df["update_time"].values

    for _, sp in df[df["is_spread"]].iterrows():
        lo, hi = sp["update_time"] - tol, sp["update_time"] + tol
        window = (times >= np.datetime64(lo)) & (times <= np.datetime64(hi))
        parties = {sp["buyer_mnemonic"], sp["seller_mnemonic"],
                   sp["market_maker_mnemonic"]}
        shares_party = (
                df["buyer_mnemonic"].isin(parties)
                | df["seller_mnemonic"].isin(parties)
                | df["market_maker_mnemonic"].isin(parties)
        )
        df.loc[window & shares_party & ~df["is_spread"], "is_spread_leg"] = True
    return df


def validate_spread_pricing(df: pd.DataFrame, tolerance: float = 0.02) -> pd.DataFrame:
    """
    Sanity-check that JKM_SPREAD ~= JKM_NEXT - JKM_BASE using the last print
    of each outright at or before the spread timestamp. Confirms the product
    mapping is right before you trust anything downstream.
    """
    sp = df[df["is_spread"]].copy()
    if sp.empty:
        return pd.DataFrame()

    out = {}
    for c in OUTRIGHTS:
        s = df.loc[df["contract"] == c, ["update_time", "price"]].dropna()
        out[c] = s.set_index("update_time")["price"]

    rows = []
    for _, r in sp.iterrows():
        base = out["JKM_BASE"].asof(r["update_time"]) if len(out["JKM_BASE"]) else np.nan
        nxt = out["JKM_NEXT"].asof(r["update_time"]) if len(out["JKM_NEXT"]) else np.nan
        rows.append({
            "update_time": r["update_time"],
            "spread_printed": r["price"],
            "base": base,
            "next": nxt,
            "next_minus_base": nxt - base,
            "residual": r["price"] - (nxt - base),
        })
    res = pd.DataFrame(rows)
    res["consistent"] = res["residual"].abs() <= tolerance
    return res


# ==========================================================================
# 2. Market-level summary
# ==========================================================================
def _vwap(g: pd.DataFrame) -> float:
    w = g["lots"].sum()
    return np.average(g["price"], weights=g["lots"]) if w else np.nan


def daily_market_summary(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC, VWAP, volume and order-flow imbalance per contract per day."""
    g = df.groupby(["trade_date", "contract"], dropna=False)
    summary = g.apply(
        lambda x: pd.Series({
            "trades": len(x),
            "lots": x["lots"].sum(),
            "mmbtu": x["volume_mmbtu"].sum(),
            "cargo_equiv": x["cargo_equiv"].sum(),
            "open": x.sort_values("update_time")["price"].iloc[0],
            "high": x["price"].max(),
            "low": x["price"].min(),
            "close": x.sort_values("update_time")["price"].iloc[-1],
            "vwap": _vwap(x),
            "twap": x["price"].mean(),
            "px_range": x["price"].max() - x["price"].min(),
            "implied_bid_avg": x["implied_bid"].mean(),
            "implied_offer_avg": x["implied_offer"].mean(),
            "n_participants": pd.unique(
                x[["buyer_mnemonic", "seller_mnemonic"]].values.ravel()
            ).size,
            "buy_agg_lots": x.loc[x["aggressor_side"] == "BUY", "lots"].sum(),
            "sell_agg_lots": x.loc[x["aggressor_side"] == "SELL", "lots"].sum(),
        }),
        include_groups=False,
    )
    summary["implied_spread"] = (
            summary["implied_offer_avg"] - summary["implied_bid_avg"]
    )
    summary["ofi_lots"] = summary["buy_agg_lots"] - summary["sell_agg_lots"]
    summary["ofi_pct"] = (
            100 * summary["ofi_lots"] / (summary["buy_agg_lots"] + summary["sell_agg_lots"])
    )
    summary["close_vs_vwap"] = summary["close"] - summary["vwap"]
    return summary.round(4)


# ==========================================================================
# 3. Counterparty flow — who bought, who sold, at what price
# ==========================================================================
def to_participant_long(df: pd.DataFrame) -> pd.DataFrame:
    """Explode each trade into two participant-side rows."""
    buy = df.assign(
        counterparty=df["buyer_mnemonic"],
        other=df["seller_mnemonic"],
        side="BUY",
    )
    sell = df.assign(
        counterparty=df["seller_mnemonic"],
        other=df["buyer_mnemonic"],
        side="SELL",
    )
    long = pd.concat([buy, sell], ignore_index=True)
    long["signed"] = np.where(long["side"] == "BUY", long["lots"], -long["lots"])
    long["was_aggressor"] = long["counterparty"] == long["aggressor"]
    return long


def counterparty_flow(
        df: pd.DataFrame,
        contract: str | None = None,
        by_day: bool = False,
        exclude_spread_legs: bool = False,
) -> pd.DataFrame:
    """
    Per-counterparty league table: gross/net lots, buy & sell VWAPs, and the
    aggressive-only net (the conviction measure).
    """
    d = df[~df["is_spread"]]
    if contract:
        d = d[d["contract"] == contract]
    if exclude_spread_legs:
        d = d[~d["is_spread_leg"]]

    long = to_participant_long(d)
    keys = (["trade_date"] if by_day else []) + ["counterparty"]

    def agg(x: pd.DataFrame) -> pd.Series:
        b, s = x[x["side"] == "BUY"], x[x["side"] == "SELL"]
        ab = x[(x["side"] == "BUY") & x["was_aggressor"]]
        asl = x[(x["side"] == "SELL") & x["was_aggressor"]]
        return pd.Series({
            "trades": len(x),
            "buy_lots": b["lots"].sum(),
            "sell_lots": s["lots"].sum(),
            "net_lots": x["signed"].sum(),
            "gross_lots": x["lots"].sum(),
            "buy_vwap": _vwap(b) if len(b) else np.nan,
            "sell_vwap": _vwap(s) if len(s) else np.nan,
            "agg_buy_lots": ab["lots"].sum(),
            "agg_sell_lots": asl["lots"].sum(),
            "aggressor_trades": int(x["was_aggressor"].sum()),
            "n_counterparties": x["other"].nunique(),
        })

    out = long.groupby(keys, dropna=False).apply(agg, include_groups=False)
    out["net_mmbtu"] = out["net_lots"] * LOT_MMBTU
    out["net_cargo_equiv"] = out["net_mmbtu"] / CARGO_MMBTU
    out["net_aggressive_lots"] = out["agg_buy_lots"] - out["agg_sell_lots"]
    out["aggressor_ratio"] = out["aggressor_trades"] / out["trades"]
    # Two-way traders: crude realised edge per MMBtu
    out["round_turn_edge"] = out["sell_vwap"] - out["buy_vwap"]
    out["round_turn_lots"] = out[["buy_lots", "sell_lots"]].min(axis=1)
    out["est_round_turn_pnl_usd"] = (
            out["round_turn_edge"] * out["round_turn_lots"] * LOT_MMBTU
    )
    out["share_of_volume_pct"] = 100 * out["gross_lots"] / out["gross_lots"].sum()
    return out.sort_values("gross_lots", ascending=False).round(4)


def directional_view(df: pd.DataFrame, contract: str | None = None) -> pd.DataFrame:
    """
    Condensed read: net position vs aggressive net, and whether the two agree.
    'CONVICTION' = they built the position by paying up / hitting bids.
    'PASSIVE'    = the position was accumulated by being quoted against.
    """
    cf = counterparty_flow(df, contract=contract)
    v = cf[["buy_lots", "sell_lots", "net_lots", "net_aggressive_lots",
            "buy_vwap", "sell_vwap", "aggressor_ratio",
            "share_of_volume_pct"]].copy()

    def label(r):
        if r["net_lots"] == 0 and r["net_aggressive_lots"] == 0:
            return "FLAT"
        if r["net_lots"] * r["net_aggressive_lots"] > 0:
            return "CONVICTION " + ("LONG" if r["net_lots"] > 0 else "SHORT")
        if r["net_aggressive_lots"] == 0:
            return "PASSIVE " + ("LONG" if r["net_lots"] > 0 else "SHORT")
        return "MIXED"

    v["read"] = v.apply(label, axis=1)
    return v.sort_values("net_lots", ascending=False)


# ==========================================================================
# 4. Market making & execution quality
# ==========================================================================
def market_maker_league(df: pd.DataFrame) -> pd.DataFrame:
    """Who is providing liquidity, and how two-sided are they."""
    d = df[~df["is_spread"]]
    g = d.groupby("market_maker_mnemonic", dropna=False)
    mm = g.apply(
        lambda x: pd.Series({
            "trades_quoted": len(x),
            "lots_quoted": x["lots"].sum(),
            "quoted_offer_lots": x.loc[x["aggressor_side"] == "BUY", "lots"].sum(),
            "quoted_bid_lots": x.loc[x["aggressor_side"] == "SELL", "lots"].sum(),
            "avg_offer_hit": x["implied_offer"].mean(),
            "avg_bid_hit": x["implied_bid"].mean(),
            "distinct_counterparties": pd.unique(
                x[["buyer_mnemonic", "seller_mnemonic"]].values.ravel()
            ).size,
        }),
        include_groups=False,
    )
    mm["two_sided_ratio"] = (
            mm[["quoted_offer_lots", "quoted_bid_lots"]].min(axis=1)
            / mm[["quoted_offer_lots", "quoted_bid_lots"]].max(axis=1).replace(0, np.nan)
    )
    mm["capture_per_mmbtu"] = mm["avg_offer_hit"] - mm["avg_bid_hit"]
    mm["share_of_quoted_pct"] = 100 * mm["lots_quoted"] / mm["lots_quoted"].sum()
    return mm.sort_values("lots_quoted", ascending=False).round(4)


def execution_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each counterparty's VWAP vs the daily contract VWAP.
    Positive edge = bought below / sold above the market's average.
    """
    d = df[~df["is_spread"]]
    bench = (
        d.groupby(["trade_date", "contract"])
        .apply(_vwap, include_groups=False)
        .rename("mkt_vwap")
        .reset_index()
    )
    long = to_participant_long(d).merge(bench, on=["trade_date", "contract"])
    long["edge"] = np.where(
        long["side"] == "BUY",
        long["mkt_vwap"] - long["price"],
        long["price"] - long["mkt_vwap"],
    )

    out = long.groupby(["contract", "counterparty"]).apply(
        lambda x: pd.Series({
            "lots": x["lots"].sum(),
            "edge_per_mmbtu": np.average(x["edge"], weights=x["lots"]),
            "edge_usd": (x["edge"] * x["volume_mmbtu"]).sum(),
            "pct_trades_at_better_than_vwap": 100 * (x["edge"] > 0).mean(),
        }),
        include_groups=False,
    )
    return out.sort_values("edge_usd", ascending=False).round(4)


# ==========================================================================
# 5. Microstructure & network
# ==========================================================================
def intraday_profile(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Volume, VWAP and order-flow imbalance through the window."""
    d = df[~df["is_spread"]].copy()
    d["bucket"] = d["update_time"].dt.floor(freq)
    out = d.groupby(["contract", "bucket"]).apply(
        lambda x: pd.Series({
            "trades": len(x),
            "lots": x["lots"].sum(),
            "vwap": _vwap(x),
            "last": x.sort_values("update_time")["price"].iloc[-1],
            "ofi_lots": x["signed_lots"].sum(),
        }),
        include_groups=False,
    )
    out["cum_ofi"] = out.groupby(level=0)["ofi_lots"].cumsum()
    out["px_chg"] = out.groupby(level=0)["last"].diff()
    return out.round(4)


def ofi_price_relationship(df: pd.DataFrame, freq: str = "5min") -> pd.Series:
    """
    Correlation between bucketed order-flow imbalance and price change.
    High positive = flow is moving the market (informed / momentum).
    Near zero    = the window is absorbing flow (deep liquidity, mean reversion).
    """
    prof = intraday_profile(df, freq).dropna(subset=["px_chg"])
    return prof.groupby(level=0).apply(
        lambda x: x["ofi_lots"].corr(x["px_chg"]) if len(x) > 2 else np.nan
    ).rename(f"ofi_px_corr_{freq}")


def counterparty_matrix(df: pd.DataFrame, value: str = "lots") -> pd.DataFrame:
    """Buyer (rows) x Seller (cols) volume matrix — reveals recurring pairings."""
    d = df[~df["is_spread"]]
    return d.pivot_table(
        index="buyer_mnemonic", columns="seller_mnemonic",
        values=value, aggfunc="sum", fill_value=0,
    )


def concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Herfindahl index and top-N share of gross volume, per contract."""
    rows = []
    for c, d in df[~df["is_spread"]].groupby("contract"):
        long = to_participant_long(d)
        share = long.groupby("counterparty")["lots"].sum()
        share = share / share.sum()
        rows.append({
            "contract": c,
            "n_participants": len(share),
            "hhi": float((share ** 2).sum()),
            "top1_pct": 100 * share.nlargest(1).sum(),
            "top3_pct": 100 * share.nlargest(3).sum(),
            "top5_pct": 100 * share.nlargest(5).sum(),
        })
    return pd.DataFrame(rows).set_index("contract").round(4)


def new_and_departed(df: pd.DataFrame) -> pd.DataFrame:
    """Participation by day — spot who has entered or left the window."""
    long = to_participant_long(df[~df["is_spread"]])
    piv = long.pivot_table(
        index="counterparty", columns="trade_date",
        values="lots", aggfunc="sum", fill_value=0,
    )
    if piv.shape[1] >= 2:
        piv["chg_vs_prev"] = piv.iloc[:, -1] - piv.iloc[:, -2]
    return piv.sort_values(piv.columns[-1], ascending=False)


# ==========================================================================
# 6. Orchestration
# ==========================================================================
def run_full_analysis(all_MOC_data: pd.DataFrame, verbose: bool = True) -> dict:
    df = prepare_moc_data(all_MOC_data)

    res = {
        "trades": df,
        "spread_check": validate_spread_pricing(df),
        "daily_summary": daily_market_summary(df),
        "flow_all": counterparty_flow(df),
        "flow_by_contract": {
            c: counterparty_flow(df, contract=c) for c in
            df.loc[~df["is_spread"], "contract"].unique()
        },
        "flow_by_day": counterparty_flow(df, by_day=True),
        "directional_view": directional_view(df),
        "market_makers": market_maker_league(df),
        "execution_quality": execution_quality(df),
        "intraday": intraday_profile(df),
        "ofi_corr": ofi_price_relationship(df),
        "matrix": counterparty_matrix(df),
        "concentration": concentration(df),
        "participation_by_day": new_and_departed(df),
    }

    if verbose:
        _print_report(res)
    return res


def _print_report(res: dict) -> None:
    df = res["trades"]
    line = "=" * 78
    print(line)
    print("PLATTS MOC — APAC LNG (JKM) DERIVATIVES  |  FLOW REPORT")
    print(f"{df['update_time'].min():%Y-%m-%d %H:%M}  ->  "
          f"{df['update_time'].max():%Y-%m-%d %H:%M}   "
          f"({df['trade_date'].nunique()} session(s), {len(df)} prints)")
    print(line)

    print("\n--- MARKET SUMMARY ---")
    print(res["daily_summary"][
              ["trades", "lots", "cargo_equiv", "open", "high", "low", "close",
               "vwap", "implied_spread", "ofi_lots", "ofi_pct"]
          ].to_string())

    if not res["spread_check"].empty:
        ok = res["spread_check"]["consistent"].mean()
        print(f"\nSpread/outright consistency: {ok:.0%} of spread prints "
              f"reconcile to NEXT - BASE")

    print("\n--- NET POSITION & CONVICTION (all outrights) ---")
    print(res["directional_view"].round(3).to_string())

    print("\n--- TOP BUYERS (net lots) ---")
    print(res["flow_all"].nlargest(5, "net_lots")[
              ["buy_lots", "sell_lots", "net_lots", "buy_vwap",
               "net_aggressive_lots", "net_cargo_equiv"]].to_string())

    print("\n--- TOP SELLERS (net lots) ---")
    print(res["flow_all"].nsmallest(5, "net_lots")[
              ["buy_lots", "sell_lots", "net_lots", "sell_vwap",
               "net_aggressive_lots", "net_cargo_equiv"]].to_string())

    print("\n--- LIQUIDITY PROVISION ---")
    print(res["market_makers"][
              ["lots_quoted", "share_of_quoted_pct", "quoted_bid_lots",
               "quoted_offer_lots", "two_sided_ratio", "capture_per_mmbtu"]
          ].head(8).to_string())

    print("\n--- EXECUTION QUALITY (edge vs daily VWAP, $/MMBtu) ---")
    eq = res["execution_quality"]
    print(pd.concat([eq.head(5), eq.tail(5)])[
              ["lots", "edge_per_mmbtu", "edge_usd"]].to_string())

    print("\n--- CONCENTRATION ---")
    print(res["concentration"].to_string())

    print("\n--- FLOW/PRICE RELATIONSHIP ---")
    print(res["ofi_corr"].round(3).to_string())
    print("\n" + line)


# ==========================================================================
# 7. Visualisation  —  two charts only
#    (a) daily net lots per counterparty, full history, line chart
#    (b) net position heatmap, last N sessions
# ==========================================================================
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#4a4f57",
    "axes.grid": True,
    "grid.color": "#dfe2e6",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

# Palette for the charts when they are embedded in the (dark) daily report.
# Applied through rc_context so the standalone light styling above is untouched.
DARK_RC = {
    "figure.facecolor": "#0D1117",
    "savefig.facecolor": "#0D1117",
    "axes.facecolor": "#0D1117",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": "#C9D1D9",
    "axes.titlecolor": "#C9D1D9",
    "grid.color": "#30363D",
    "text.color": "#C9D1D9",
    "xtick.color": "#C9D1D9",
    "ytick.color": "#C9D1D9",
    "legend.labelcolor": "#C9D1D9",
}


def _on_dark_background() -> bool:
    """True when the active rcParams draw onto a dark axes background."""
    rgb = matplotlib.colors.to_rgb(plt.rcParams["axes.facecolor"])
    return matplotlib.colors.rgb_to_hsv(rgb)[2] < 0.5


def daily_net_matrix(
        df: pd.DataFrame,
        contract: str | None = None,
        top_n: int | None = None,
        exclude_spread_legs: bool = False,
) -> pd.DataFrame:
    """
    Counterparty (rows) x trade_date (cols) matrix of NET lots.
    Sessions with no activity for a name are 0, not NaN — a flat day is
    genuinely a zero net, not missing data.
    """
    d = df[~df["is_spread"]]
    if contract:
        d = d[d["contract"] == contract]
    if exclude_spread_legs:
        d = d[~d["is_spread_leg"]]

    long = to_participant_long(d)
    piv = long.pivot_table(index="counterparty", columns="trade_date",
                           values="signed", aggfunc="sum", fill_value=0)

    # ensure every session in the sample appears, even if a name never traded
    all_days = sorted(df["trade_date"].unique())
    piv = piv.reindex(columns=all_days, fill_value=0)

    # rank by gross activity, not net, so two-way names are not hidden
    gross = long.groupby("counterparty")["lots"].sum()
    piv = piv.reindex(gross.sort_values(ascending=False).index)
    if top_n:
        piv = piv.head(top_n)
    return piv


# -- (a) HISTORY LINES -----------------------------------------------------
def plot_net_history(
        df: pd.DataFrame,
        contract: str | None = "JKM_NEXT",
        top_n: int = 10,
        cumulative: bool = False,
        smooth: int | None = None,
        ax=None,
):
    """
    Line chart of daily net lots per counterparty across the whole sample.

    top_n      : keep the N most active names (by gross lots); the rest are
                 aggregated into a single grey 'OTHER' line so the chart still
                 balances to zero.
    cumulative : plot the running net instead of the daily net — this is the
                 position build, and is usually the more readable of the two
                 once you are past ~15 sessions.
    smooth     : optional centred rolling mean, in sessions, applied to the
                 daily net to strip single-session noise.
    """
    piv_all = daily_net_matrix(df, contract=contract)
    piv = piv_all.head(top_n).copy()
    rest = piv_all.iloc[top_n:]
    if len(rest):
        piv.loc["OTHER"] = rest.sum()

    if cumulative:
        piv = piv.cumsum(axis=1)
        ylab, kind = "Cumulative net lots", "Cumulative net position"
    elif smooth:
        piv = piv.T.rolling(smooth, center=True, min_periods=1).mean().T
        ylab = f"Net lots ({smooth}-session rolling mean)"
        kind = "Daily net flow, smoothed"
    else:
        ylab, kind = "Net lots (+bought / −sold)", "Daily net flow"

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 6))

    x = [pd.Timestamp(c) for c in piv.columns]
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    styles = ["-", "--", "-."]

    for i, name in enumerate(piv.index):
        if name == "OTHER":
            ax.plot(x, piv.loc[name], color="#9aa0a6", lw=1.4, ls=":",
                    label="OTHER", zorder=1)
            continue
        ax.plot(x, piv.loc[name], color=cmap[i % 20], lw=1.7,
                ls=styles[(i // 20) % 3], marker="o", ms=3.2,
                label=name, alpha=0.9)

    ax.axhline(0, color=plt.rcParams["axes.edgecolor"], lw=1.0, zorder=0)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Session")
    ax.set_title(f"Platts JKM MOC: {kind} by counterparty"
                 f"{f' — {contract}' if contract else ' — all outrights'}"
                 f"   ({len(piv_all.columns)} sessions, "
                 f"{x[0]:%d %b} – {x[-1]:%d %b %Y})")
    ax.legend(frameon=False, fontsize=8, ncols=1, loc="center left",
              bbox_to_anchor=(1.06, 0.5), title="Counterparty",
              title_fontsize=8.5)
    ax.tick_params(axis="x", rotation=45)

    sec = ax.secondary_yaxis(
        "right", functions=(lambda v: v * LOT_MMBTU / CARGO_MMBTU,
                            lambda v: v * CARGO_MMBTU / LOT_MMBTU))
    sec.set_ylabel("Cargo equivalents", fontsize=8)
    return ax


# -- (b) RECENT HEATMAP ----------------------------------------------------
def plot_net_heatmap(
        df: pd.DataFrame,
        contract: str | None = "JKM_NEXT",
        days: int = 10,
        top_n: int = 15,
        ax=None,
):
    """
    Counterparty x session grid of net lots for the most recent `days`
    sessions. Blue = net long, red = net short. Ordered by net over the
    window, so persistent accumulators sit at the top.
    """
    piv = daily_net_matrix(df, contract=contract)
    piv = piv.iloc[:, -days:]
    piv = piv.loc[piv.abs().sum(axis=1).nlargest(top_n).index]
    piv = piv.reindex(piv.sum(axis=1).sort_values(ascending=False).index)

    if ax is None:
        _, ax = plt.subplots(figsize=(0.95 * piv.shape[1] + 4.5,
                                      0.42 * len(piv) + 2.4))

    lim = max(abs(piv.values).max(), 1)
    im = ax.imshow(piv.values, cmap="RdBu",
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
                   aspect="auto")

    ax.set_xticks(range(piv.shape[1]),
                  [pd.Timestamp(c).strftime("%d-%b") for c in piv.columns],
                  rotation=45, ha="right")
    ax.set_yticks(range(len(piv)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if v:
                ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 * lim else "#22252a")

    # window total on the right edge — it sits on the figure background, so the
    # long/short colours have to lift on a dark theme
    pos_c, neg_c = (("#58b6d9", "#f07a63") if _on_dark_background()
                    else ("#1f6f8b", "#c1442e"))
    tot = piv.sum(axis=1)
    for i, v in enumerate(tot):
        ax.text(piv.shape[1] - 0.35, i, f"  {v:+,.0f}", va="center",
                ha="left", fontsize=8, fontweight="bold",
                color=pos_c if v > 0 else neg_c if v < 0 else "#8a8f98")

    ax.set_title(f"Platts JKM MOC: Net lots by counterparty — last {piv.shape[1]} sessions"
                 f"{f' ({contract})' if contract else ''}"
                 f"\nrightmost column = window total")
    ax.grid(visible=False)
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.13, label="Net lots per session")
    return ax


# -- convenience -----------------------------------------------------------
def plot_flow_report(
        df: pd.DataFrame,
        contract: str | None = "JKM_NEXT",
        history_top_n: int = 10,
        heatmap_days: int = 10,
        heatmap_top_n: int = 15,
        cumulative: bool = False,
        save_path: str | None = None,
        show: bool = False,
        dark: bool = False,
):
    """Both charts on one page: full-history lines above, recent heatmap below."""
    with matplotlib.rc_context(DARK_RC if dark else {}):
        fig = plt.figure(figsize=(15, 13))
        gs = fig.add_gridspec(2, 1, hspace=0.32, height_ratios=[1, 1.15])
        plot_net_history(df, contract, top_n=history_top_n,
                         cumulative=cumulative, ax=fig.add_subplot(gs[0]))
        plot_net_heatmap(df, contract, days=heatmap_days,
                         top_n=heatmap_top_n, ax=fig.add_subplot(gs[1]))
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def get_platts_flow_figs(
        df: pd.DataFrame | None = None,
        contract: str | None = None,
        history_top_n: int = 10,
        heatmap_days: int = 10,
        heatmap_top_n: int = 15,
        cumulative: bool = True,
        dark: bool = True,
):
    """Return the two MOC flow figures for the daily report.

    df : an already-prepared tape; None pulls today's from the Platts API.
    Returns (history_fig, heatmap_fig) as separate figures so each can be laid
    out full width — the combined `plot_flow_report` page is too tall to embed.
    """
    if df is None:
        df = fetch_moc_trades()

    with matplotlib.rc_context(DARK_RC if dark else {}):
        ax_history = plot_net_history(df, contract, top_n=history_top_n,
                                      cumulative=cumulative)
        ax_heatmap = plot_net_heatmap(df, contract, days=heatmap_days,
                                      top_n=heatmap_top_n)
    return ax_history.figure, ax_heatmap.figure


if __name__ == "__main__":
    # markets = ["Asia LNG Derivative", "Asia LNG Physical"]
    df = fetch_moc_trades()

    # results = run_full_analysis(df)
    plot_flow_report(df,
                     contract=None,
                     save_path="jkm_flow.png",
                     cumulative=True)
    # results["flow_by_contract"]["JKM_NEXT"].to_excel("jkm_next_flow.xlsx")