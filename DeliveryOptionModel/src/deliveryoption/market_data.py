"""Live market inputs from Bloomberg: forwards, fixings, ATM vols, FX.

Used by the calibration job only; the pricing job reads files.

Verified on this terminal (2026-09-23):
  * TTF futures on the TZT root (EUR/MWh) - what the daily report uses.
  * TTF options hang off the FJS root, not TZT: OPT_CHAIN on TZTX26 is empty,
    on FJSX26 it is populated. FJS and TZT settle to the same number.
  * HH futures and options on NG (USD/MMBtu). OPT_CHAIN needs the two-digit
    year ('NGX26 Comdty'); the option tickers themselves carry one digit.
  * 30DAY_IMPVOL_100.0%MNY_DF is a constant-maturity number - identical on
    NGX26 and NGZ26 - so it cannot give a per-contract vol. Per-contract ATM
    vol is read off the chain (IVOL_MID) and interpolated to the forward.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

MONTH_CODES = "FGHJKMNQUVXZ"
# Live mid first; far months without a live market still carry the vol
# implied from the exchange settlement (verified on FJSJ27, FJSV27). Far NG
# (May-27 on) returns neither on this terminal.
VOL_FIELDS = ("IVOL_MID", "IVOL_LAST")
OPTION_RE = re.compile(r"^[A-Z]+[FGHJKMNQUVXZ]\d(?P<cp>[CP])\s+(?P<strike>[\d.]+)\s+Comdty$")


def contract_ticker(root: str, month: pd.Period) -> str:
    return f"{root}{MONTH_CODES[month.month - 1]}{month.year % 100:02d} Comdty"


def contract_month(spec: dict, delivery: pd.Period) -> pd.Period:
    """The futures month that prices a cargo delivered in `delivery`.

    `contract_month_offset` in the hub registry: 0 for TTF and HH (the
    delivery-month contract). ICE Brent contract X expires at the end of month
    X-2, so the prompt Brent during delivery month m is contract m+2: offset 2.
    """
    return delivery + int(spec.get("contract_month_offset", 0))


def hub_contract(spec: dict, delivery: pd.Period, root_key: str = "future_root") -> str:
    """Ticker of the hub's future (or option chain root) for a delivery month."""
    return contract_ticker(spec[root_key], contract_month(spec, delivery))


def fetch_contracts(hubs: dict, months_by_hub: dict[str, list[pd.Period]]) -> pd.DataFrame:
    """One row per (hub, month): forward, fixing, lot size and ATM implied vol.

    A contract whose vol cannot be read (a far month with no listed strikes
    around the forward) keeps its row with NaN vol and the reason in `note`:
    the pricing job only fails if a trade actually needs it.
    """
    from xbbg import blp

    rows = []
    for hub, months in months_by_hub.items():
        spec = hubs[hub]
        futures = [hub_contract(spec, m) for m in months]
        ref = blp.bdp(futures, ["PX_LAST", "LAST_TRADEABLE_DT", "FUT_CONT_SIZE"])
        for month, fut in zip(months, futures):
            if fut not in ref.index or pd.isna(ref.loc[fut, "px_last"]):
                rows.append({"hub": hub, "month": str(month), "future": fut, "note": "no price"})
                continue
            fwd = float(ref.loc[fut, "px_last"])
            row = {"hub": hub, "month": str(month), "contract_month": str(contract_month(spec, month)),
                   "future": fut, "fwd": fwd,
                   "fixing": pd.Timestamp(ref.loc[fut, "last_tradeable_dt"]).date(),
                   "lot_size": float(ref.loc[fut, "fut_cont_size"]), "note": ""}
            try:
                vol, source, expiry = atm_vol(hub_contract(spec, month, "option_root"), fwd)
                row.update(atm_vol=vol, vol_source=source, option_expiry=expiry.date())
            except ValueError as e:
                row.update(atm_vol=np.nan, vol_source="", option_expiry=None, note=str(e))
            rows.append(row)
    return pd.DataFrame(rows)


def atm_vol(option_future: str, fwd: float) -> tuple[float, str, pd.Timestamp]:
    """ATM IVOL_MID, linear in strike between the two strikes bracketing fwd.

    Calls are used; a strike whose call has no vol falls back to its put, which
    under Black-76 put-call parity carries the same implied vol.
    """
    from xbbg import blp

    chain = blp.bds(option_future, "OPT_CHAIN")
    if chain.empty:
        raise ValueError(f"Empty OPT_CHAIN for {option_future}")
    by_strike: dict[float, dict[str, str]] = {}
    for tkr in chain.iloc[:, 0].dropna():
        m = OPTION_RE.match(tkr.strip())
        if m:
            by_strike.setdefault(float(m["strike"]), {})[m["cp"]] = tkr.strip()

    strikes = np.array(sorted(by_strike))
    below = strikes[strikes <= fwd]
    above = strikes[strikes >= fwd]
    if not len(below) or not len(above):
        raise ValueError(f"{option_future}: forward {fwd} outside the listed strikes")
    pick = sorted({below.max(), above.min()})

    tickers = [t for k in pick for t in by_strike[k].values()]
    # xbbg drops a field that came back empty for every ticker, rather than
    # returning a NaN column - e.g. an expiring contract's options.
    data = blp.bdp(tickers, [*VOL_FIELDS, "OPT_EXPIRE_DT"]).reindex(
        columns=[*(f.lower() for f in VOL_FIELDS), "opt_expire_dt"])

    vols, used = [], []
    for k in pick:
        v = np.nan
        for field in VOL_FIELDS:
            for cp in ("C", "P"):
                t = by_strike[k].get(cp)
                if t in data.index and pd.notna(data.loc[t, field.lower()]):
                    v = float(data.loc[t, field.lower()]) / 100.0
                    used.append(f"{t} [{field}]")
                    break
            if not np.isnan(v):
                break
        vols.append(v)
    if np.isnan(vols).any():
        raise ValueError(f"{option_future}: no {' or '.join(VOL_FIELDS)} at strikes {pick}")

    vol = vols[0] if len(pick) == 1 else float(np.interp(fwd, pick, vols))
    expiry = pd.Timestamp(data["opt_expire_dt"].dropna().iloc[0])
    return vol, " / ".join(used), expiry


def fill_missing_vols(contracts: pd.DataFrame, how: str, profile: pd.DataFrame | None = None,
                      as_of: pd.Timestamp | None = None, scale_months: int = 12) -> pd.DataFrame:
    """Fill `vol` where Bloomberg gave none; `atm_vol` stays as read.

    samuelson: the hub's realised vol-by-maturity curve, averaged from 0 to
               the contract's expiry (vols.profile_vol), times one scale per
               hub = median(market vol / curve vol) over the hub's
               `scale_months` longest-dated contracts that do have a market
               vol - the ones nearest the gap. Keeps the term-structure decay;
               carries no seasonality.
    previous:  carry the last earlier month of the same hub that has a vol.
    none:      leave NaN; pricing a trade that needs it then fails loudly.
    """
    out = contracts.copy()
    out["vol"] = out["atm_vol"]
    if how == "none":
        return out
    if how == "previous":
        for hub, g in out.sort_values("month").groupby("hub"):
            last_month, last_vol = None, np.nan
            for i, row in g.iterrows():
                if pd.notna(row["atm_vol"]):
                    last_month, last_vol = row["month"], row["atm_vol"]
                elif pd.notna(row.get("fwd")) and last_month is not None:
                    out.loc[i, "vol"] = last_vol
                    out.loc[i, "note"] = f"vol FILLED from {last_month}; " + str(row["note"])
        return out
    if how != "samuelson":
        raise ValueError(f"unknown missing_vol_fill {how!r}")

    from DeliveryOptionModel.src.deliveryoption.vols import profile_vol

    expiry = pd.to_datetime(out["option_expiry"]).fillna(pd.to_datetime(out["fixing"]))
    t_days = (expiry - as_of).dt.days
    for hub, g in out.sort_values("month").groupby("hub"):
        prof = profile[profile["hub"] == hub]
        base = pd.Series({i: profile_vol(prof, t_days[i]) for i in g.index})
        have = g[g["atm_vol"].notna()].tail(scale_months)
        if have.empty:
            # Nothing to scale the curve to. Calibrate with the near strip
            # (strip_months > 0), whose liquid months supply the scale.
            gap = g["atm_vol"].isna() & g["fwd"].notna()
            out.loc[gap[gap].index, "note"] = (f"vol NOT FILLED: no {hub} market vol among the "
                                               "calibrated months to scale from - include the strip; "
                                               + out.loc[gap[gap].index, "note"].astype(str))
            continue
        scale = float((have["atm_vol"] / base[have.index]).median())
        gap = g[g["atm_vol"].isna() & g["fwd"].notna()]
        for i in gap.index:
            out.loc[i, "vol"] = scale * base[i]
            out.loc[i, "note"] = (f"vol FILLED samuelson: realised curve x{scale:.3f} "
                                  f"(scale from {have['month'].iloc[0]}..{have['month'].iloc[-1]}); "
                                  + str(out.loc[i, "note"]))
    return out


def fx_forward_curve(spot: float, pips: dict[int, float], months: list[pd.Period],
                     as_of: pd.Timestamp, pips_per_unit: float) -> dict[str, float]:
    """Outright forward per delivery month, linear in time between tenors.

    Same convention as the daily report's EURUSD curve: each month is taken
    at its 15th; flat past the last quoted tenor.
    """
    grid = sorted([(0.0, spot)] + [(float(n), spot + p / pips_per_unit) for n, p in pips.items()])
    x, y = zip(*grid)
    t = [(pd.Timestamp(m.year, m.month, 15) - as_of).days / 365.25 * 12 for m in months]
    return dict(zip((str(m) for m in months), np.interp(t, x, y)))


def fetch_fx_forwards(spec: dict, months: list[pd.Period], as_of: pd.Timestamp) -> dict[str, float]:
    """{month: outright} from '<root><tenor> BGN Curncy' pips over spot."""
    from xbbg import blp

    tenors = {f"{spec['root']}{tenor} BGN Curncy": n for tenor, n in spec["tenors"].items()}
    q = blp.bdp([spec["spot"], *tenors], "PX_LAST")["px_last"]
    if spec["spot"] not in q.index:
        raise ValueError(f"No FX spot for {spec['spot']}")
    pips = {n: float(q[t]) for t, n in tenors.items() if t in q.index and pd.notna(q[t])}
    return fx_forward_curve(float(q[spec["spot"]]), pips, months, as_of, spec["pips_per_unit"])


def fetch_fx(tickers: list[str]) -> dict[str, float]:
    from xbbg import blp
    if not tickers:
        return {}
    data = blp.bdp(tickers, "PX_LAST")
    return {t: float(data.loc[t, "px_last"]) for t in tickers}
