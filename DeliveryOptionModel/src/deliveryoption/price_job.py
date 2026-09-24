"""Job 2: price every CSO in trades.yaml off a calibration folder. No Bloomberg.

    python price.py                                  # latest calibration
    python price.py --calib 2026-09-23_edited        # a named / edited one
    python price.py --only NovDec26,DecJan27 --as-of 2026-09-24

Writes output/<timestamp>/:
    trades.csv          one row per trade: value, intrinsic, extrinsic, exercise split
    trades_summary.csv  headline subset of trades.csv (summary_columns), 3 decimals
    legs.csv        one row per trade and leg: inputs used, delta, lots, vega
    portfolio.csv   risk summed by futures contract across all trades
    grids.csv       extrinsic vs correlation x vol scale, per trade
    run.json        calibration used, as-of, settings, notes and errors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from DeliveryOptionModel.src.deliveryoption import deal as deal_mod, pricer, snapshot

ROOT = Path(__file__).resolve().parents[2]
RULE = "-" * 78

# Units of every trades.csv column, written as the file's second row. A column
# missing here makes write() fail, so a new column cannot ship without a unit.
# Money is per MMBtu of cargo unless the name says usd; TTF-leg sensitivities
# live in legs.csv.
PER_MMBTU = "USD/MMBtu"
TRADE_UNITS = {
    "trade": "id", "m0": "month", "m1": "month", "volume_mmbtu": "MMBtu",
    "sell": "weight*hub + constant (USD/MMBtu)", "buy": "weight*hub + constant (USD/MMBtu)",
    "k0": PER_MMBTU, "k1": PER_MMBTU,
    "can_cancel": "bool", "decision_date": "date", "vol_horizon": "method",
    "option_value_mc": PER_MMBTU, "option_stderr_mc": PER_MMBTU,
    "option_intrinsic": PER_MMBTU, "option_time_value_mc": PER_MMBTU,
    "option_value_usd_mc": "USD", "option_value_kirk": PER_MMBTU,
    "pv_margin_m0": PER_MMBTU, "pv_margin_m1": PER_MMBTU, "intrinsic": PER_MMBTU,
    "value_mc": PER_MMBTU, "stderr_mc": PER_MMBTU,
    "value_kirk": PER_MMBTU, "value_bachelier": PER_MMBTU,
    "extrinsic_mc": PER_MMBTU, "extrinsic_usd_mc": "USD", "value_usd_mc": "USD",
    "prob_m1_mc": "probability 0-1", "prob_cancel_mc": "probability 0-1",
    "overrides": "text",
}
CORR_SENS_PREFIX = "corr_sens_usd_mc "
VOL_PREFIX = "vol "            # + leg, e.g. "vol TTF_M1"
# Per-leg risk columns (+ leg). Deltas are in futures lots of that leg's
# contract, so the hedge is minus the number: the whole delivery (margin locked
# in + option), and the option alone (what option_value_mc prices). Vega is the
# same for both: the committed M margin carries no vol.
DELTA_PREFIX = "delta_lots_mc "
OPTION_DELTA_PREFIX = "option_delta_lots_mc "
VEGA_PREFIX = "vega_usd_mc "
CORR_PREFIX = "corr "          # + leg pair, e.g. "corr TTF_M/TTF_M1"


HAIRCUT_KINDS = ("vol", "corr")

# trades_summary.csv: a fixed subset of trades.csv. The vol / corr / risk
# columns follow the hubs the run's trades use, in the order they appear:
#   vols        vol <H>_M, vol <H>_M1                     for each hub
#   corrs       corr <H>_M/<H>_M1, then corr <H>_M/<K>_M  for each later hub K
#   risk        corr_sens <H>_M/<H>_M1 for each hub, then every "all cross pairs"
# For TTF + HH that is exactly: vol TTF_M, vol TTF_M1, vol HH_M, vol HH_M1,
# corr TTF_M/TTF_M1, corr TTF_M/HH_M, corr HH_M/HH_M1, ... TTF/HH all cross pairs.
SUMMARY_TERMS = ["trade", "m0", "m1", "volume_mmbtu", "sell", "buy", "k0", "k1",
                 "can_cancel", "decision_date"]
SUMMARY_VALUES = ["option_value_mc", "option_intrinsic", "option_time_value_mc",
                  "option_value_usd_mc", "option_value_kirk", "extrinsic_mc",
                  "extrinsic_usd_mc", "value_usd_mc", "prob_m1_mc"]
# Every summary column except SUMMARY_TERMS is rounded to this many decimals in
# trades_summary.csv only; trades.csv keeps full precision.
SUMMARY_DECIMALS = 3


def summary_columns(columns) -> list[str]:
    columns = list(columns)
    hubs = list(dict.fromkeys(c[len(VOL_PREFIX):-len("_M")] for c in columns
                              if c.startswith(VOL_PREFIX) and c.endswith("_M")))
    vols = [f"{VOL_PREFIX}{h}_{tag}" for h in hubs for tag in ("M", "M1")]
    corrs = []
    for i, h in enumerate(hubs):
        corrs.append(f"{CORR_PREFIX}{h}_M/{h}_M1")
        corrs += [c for k in hubs[i + 1:] if (c := f"{CORR_PREFIX}{h}_M/{k}_M") in columns]
    sens = [f"{CORR_SENS_PREFIX}{h}_M/{h}_M1" for h in hubs]
    sens += [c for c in columns if c.startswith(CORR_SENS_PREFIX) and c.endswith("all cross pairs")]
    legs = [f"{h}_{tag}" for h in hubs for tag in ("M", "M1")]
    # The summary reports the option, so its delta is the option's own hedge
    # (trades.csv also has the whole-delivery delta).
    deltas = [f"{OPTION_DELTA_PREFIX}{leg}" for leg in legs]
    vegas = [f"{VEGA_PREFIX}{leg}" for leg in legs]
    return SUMMARY_TERMS + vols + corrs + SUMMARY_VALUES + sens + deltas + vegas


def leg_label(hub: str, weight: tuple, const: tuple) -> str:
    """'0.166*Brent + 1.5', or one per month when M and M+1 differ."""
    def one(i):
        text = f"{weight[i]:g}*{hub}"
        if const[i]:
            text += f" {'+' if const[i] > 0 else '-'} {abs(const[i]):g}"
        return text
    if (weight[0], const[0]) == (weight[1], const[1]):
        return one(0)
    return f"{one(0)} (M) | {one(1)} (M+1)"


def hub_pairs(hubs) -> list[tuple[str, str]]:
    """Hub pairs in registry order: (TTF, HH) -> key TTF_HH_corr_haircut."""
    hubs = list(hubs)
    return [(a, b) for i, a in enumerate(hubs) for b in hubs[i + 1:]]


def haircuts(pricing_cfg: dict, hubs) -> dict[str, dict]:
    """{'vol': {hub: x}, 'corr': {hub: x}, 'cross': {(a, b): x}} from
    pricing.<HUB>_vol_haircut, <HUB>_corr_haircut and <A>_<B>_corr_haircut.

    Missing keys mean 1.0. The retired single `vol_haircut` is rejected rather
    than silently ignored, so an old config cannot price at full vol by
    accident.
    """
    if "vol_haircut" in pricing_cfg:
        raise KeyError("pricing.vol_haircut is replaced by per-hub keys: "
                       + ", ".join(f"{h}_vol_haircut" for h in hubs))
    out = {kind: {h: float(pricing_cfg.get(f"{h}_{kind}_haircut", 1.0)) for h in hubs}
           for kind in HAIRCUT_KINDS}
    out["cross"] = {(a, b): float(pricing_cfg.get(f"{a}_{b}_corr_haircut", 1.0))
                    for a, b in hub_pairs(hubs)}
    return out


def trade_units(columns, bump_corr: float, bump_vol: float = 0.01) -> list[str]:
    units = []
    for c in columns:
        if c.startswith(OPTION_DELTA_PREFIX):
            units.append("futures lots, option only (hedge = minus)")
        elif c.startswith(DELTA_PREFIX):
            units.append("futures lots, whole delivery (hedge = minus)")
        elif c.startswith(VEGA_PREFIX):
            units.append(f"USD per +{bump_vol * 100:g} vol pt")
        elif c.endswith("_vol_haircut"):
            units.append("multiplier on calibrated vols")
        elif c.endswith("_corr_haircut") and c[:-len("_corr_haircut")].count("_") == 1:
            units.append("multiplier on calibrated cross-hub corr (all 4 pairs)")
        elif c.endswith("_corr_haircut"):
            units.append("multiplier on calibrated M/M+1 corr")
        elif c.startswith(CORR_SENS_PREFIX):
            units.append(f"USD per +{bump_corr:g} corr")
        elif c.startswith(VOL_PREFIX):
            units.append("annualised vol, 0.30 = 30%")
        elif c.startswith(CORR_PREFIX):
            units.append("correlation -1..1")
        elif c in TRADE_UNITS:
            units.append(TRADE_UNITS[c])
        else:
            raise KeyError(f"trades.csv column {c!r} has no unit - add it to TRADE_UNITS")
    return units


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default=str(ROOT / "config" / "market.yaml"))
    ap.add_argument("--trades", default=str(ROOT / "config" / "trades_2028_2031.yaml"))
    ap.add_argument("--calib", default="latest", help="'latest', a folder name, or a path")
    ap.add_argument("--as-of", help="pricing date; default the calibration's as-of")
    ap.add_argument("--only", help="comma-separated trade ids")
    # Haircut flags depend on the hub registry, so read --market first.
    known, _ = ap.parse_known_args(argv)
    market = yaml.safe_load(Path(known.market).read_text())
    flags = {}                                   # argparse dest -> pricing key
    for hub in market["hubs"]:
        for kind, what in (("vol", "vols, e.g. 0.8 = 80%% of the vol"),
                           ("corr", "M/M+1 correlation; below 1 RAISES the option value")):
            key = f"{hub}_{kind}_haircut"
            ap.add_argument(f"--{key.lower().replace('_', '-')}", type=float,
                            help=f"multiply {hub}'s calibrated {what}; overrides pricing.{key}")
            flags[key.lower()] = key
    for a, b in hub_pairs(market["hubs"]):
        key = f"{a}_{b}_corr_haircut"
        ap.add_argument(f"--{key.lower().replace('_', '-')}", type=float,
                        help=f"multiply all four calibrated {a}/{b} cross correlations; "
                             f"overrides pricing.{key}")
        flags[key.lower()] = key
    args = ap.parse_args(argv)

    for dest, key in flags.items():
        if getattr(args, dest) is not None:
            market["pricing"][key] = getattr(args, dest)
    trades_cfg = yaml.safe_load(Path(args.trades).read_text())
    calib_root = _abs(market["calibration"]["output_dir"])
    snap = snapshot.read(snapshot.resolve(calib_root, args.calib))
    as_of = pd.Timestamp(args.as_of) if args.as_of else snap.as_of

    trades = deal_mod.parse_trades(trades_cfg, market["pricing"])
    if args.only:
        wanted = set(args.only.split(","))
        trades = [t for t in trades if t.id in wanted]
        if missing := wanted - {t.id for t in trades}:
            raise KeyError(f"unknown trade ids {sorted(missing)}")

    result = run(market, trades, snap, as_of)
    folder = _abs(market["pricing"]["output_dir"]) / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}"
    write(folder, result, snap, as_of, market)
    print(f"\nWritten to {folder}")


def run(market: dict, trades: list, snap: snapshot.Snapshot, as_of: pd.Timestamp) -> dict:
    cfg = market["pricing"]
    cuts = haircuts(cfg, market["hubs"])
    trade_rows, leg_rows, grids, notes, errors = [], [], [], {}, {}
    for t in trades:
        try:
            d = deal_mod.build(t, snap, market["hubs"], market["rate"], as_of,
                               vol_haircuts=cuts["vol"], corr_haircuts=cuts["corr"],
                               cross_corr_haircuts=cuts["cross"])
            trade_row, legs = price_deal(d, cfg, cuts)
        except (KeyError, ValueError, FileNotFoundError) as e:
            errors[t.id] = str(e)
            print(f"[{t.id}] NOT PRICED: {e}")
            continue
        trade_rows.append(trade_row)
        leg_rows.append(legs)
        notes[t.id] = d.notes
        if cfg.get("grid"):
            grids.append(grid(d, cfg))
    return {
        "trades": pd.DataFrame(trade_rows),
        "legs": pd.concat(leg_rows, ignore_index=True) if leg_rows else pd.DataFrame(),
        "grids": pd.concat(grids, ignore_index=True) if grids else pd.DataFrame(),
        "notes": notes, "errors": errors,
    }


def price_deal(d: deal_mod.Deal, cfg: dict,
               cuts: dict[str, dict[str, float]] | None = None) -> tuple[dict, pd.DataFrame]:
    """`cuts` is only reported here; it was applied when `d` was built."""
    cuts = cuts or {kind: {} for kind in (*HAIRCUT_KINDS, "cross")}
    m, p, t = d.model, d.payoff, d.trade
    names = list(m.names)
    # Pairs whose sensitivity is reported: M/M1 within each hub, and M/M across hubs.
    # Correlation risk: each hub's M/M+1 pair on its own, and all cross-hub
    # pairs moved together. Bumping one cross pair alone would change how the
    # two calendar spreads co-move - not a TTF/HH correlation view anyone holds.
    groups = {f"{h}_M/{h}_M1": [(names.index(f"{h}_M"), names.index(f"{h}_M1"))] for h in t.hubs}
    for a, b in zip(t.hubs, t.hubs[1:]):
        groups[f"{a}/{b} all cross pairs"] = [(names.index(f"{a}_{x}"), names.index(f"{b}_{y}"))
                                              for x in ("M", "M1") for y in ("M", "M1")]
    rk_mc = pricer.risk_mc(m, p, cfg["n_paths"], cfg["seed"], cfg["bump_vol"], cfg["bump_corr"], groups)

    intrinsic = p.intrinsic(m.fwd)
    x0, x1 = (float(v[0]) for v in p.pv_legs(m.fwd[None, :]))
    kirk = bach = np.nan
    if not p.can_cancel:
        kirk = pricer.value_kirk_basket(m, p)["value"]
        bach = pricer.value_bachelier(m, p)["value"]

    vol = t.volume_mmbtu
    # Inputs the engine actually priced with: horizon vols after overrides,
    # and the full leg correlation matrix after overrides and PSD repair.
    # (The file vol before the horizon adjustment is in legs.csv.)
    used = {f"{VOL_PREFIX}{n}": float(v) for n, v in zip(names, m.vol)}
    used.update({f"{CORR_PREFIX}{names[i]}/{names[j]}": float(m.corr[i, j])
                 for i in range(len(names)) for j in range(i + 1, len(names))})
    row = {
        "trade": t.id, "m0": str(t.m0), "m1": str(t.m1), "volume_mmbtu": vol,
        "sell": leg_label(t.sell_hub, t.sell_weight, t.sell_const),
        "buy": leg_label(t.buy_hub, t.buy_weight, t.buy_const),
        "k0": t.k0, "k1": t.k1, "can_cancel": t.can_cancel,
        "decision_date": d.decision_date.date(), "vol_horizon": t.vol_horizon,
        **{f"{h}_{kind}_haircut": cuts[kind].get(h, 1.0)
           for kind in HAIRCUT_KINDS for h in t.hubs},
        **{f"{a}_{b}_corr_haircut": h
           for (a, b), h in cuts.get("cross", {}).items() if a in t.hubs and b in t.hubs},
        **used,
        # Columns ending in _mc are Monte Carlo; _kirk / _bachelier are the
        # closed forms; the rest (intrinsic, pv_margin_*) use today's forwards.
        #
        # The option alone: the right to switch from M to M+1 (and to cancel,
        # if allowed), valued against committing to M today. Always >= 0.
        # option_value_mc = option_intrinsic + option_time_value_mc, and the
        # time value is the same number as `extrinsic_mc` below.
        "option_value_mc": rk_mc["option_value_mc"], "option_stderr_mc": rk_mc["option_stderr_mc"],
        "option_intrinsic": intrinsic - x0,
        "option_time_value_mc": rk_mc["option_value_mc"] - (intrinsic - x0),
        "option_value_usd_mc": rk_mc["option_value_mc"] * vol,
        "option_value_kirk": kirk - x0,
        # The whole delivery: margin locked in plus the option. Negative when
        # both months are under water today.
        "pv_margin_m0": x0, "pv_margin_m1": x1, "intrinsic": intrinsic,
        "value_mc": rk_mc["value_mc"], "stderr_mc": rk_mc["stderr_mc"],
        "value_kirk": kirk, "value_bachelier": bach,
        "extrinsic_mc": rk_mc["value_mc"] - intrinsic,
        "extrinsic_usd_mc": (rk_mc["value_mc"] - intrinsic) * vol,
        "value_usd_mc": rk_mc["value_mc"] * vol,
        "prob_m1_mc": rk_mc["prob_m1_mc"], "prob_cancel_mc": rk_mc["prob_cancel_mc"],
        "overrides": "; ".join(d.notes),
    }
    for label, s_mc in rk_mc["corr_sens_mc"].items():
        row[f"{CORR_SENS_PREFIX}{label}"] = s_mc * vol

    legs = d.legs.reset_index()
    legs.insert(0, "trade", t.id)
    legs["delta_per_mmbtu_mc"] = rk_mc["delta_mc"]
    # P&L per unit move in the leg's own price, and the futures that offset it.
    legs["qty_native_mc"] = rk_mc["delta_mc"] * vol / legs["fx_to_usd"]
    legs["lots_mc"] = legs["qty_native_mc"] / legs["lot_size"]
    # Hedge of the option alone: the deal's delta less that of the committed M
    # margin, whose delta is exactly df0 * w0.
    opt_delta_mc = rk_mc["delta_mc"] - p.df0 * p.w0
    legs["option_lots_mc"] = opt_delta_mc * vol / legs["fx_to_usd"] / legs["lot_size"]
    legs["vega_usd_per_pt_mc"] = rk_mc["vega_mc"] * vol
    corr = pd.DataFrame(m.corr, names, names)
    legs["corr_used_M_M1"] = [corr.loc[f"{r.hub}_M", f"{r.hub}_M1"] for r in legs.itertuples()]

    # Per-leg risk on the trade's own row (legs.csv keeps the full detail).
    for r in legs.itertuples():
        row[f"{DELTA_PREFIX}{r.leg}"] = r.lots_mc
        row[f"{OPTION_DELTA_PREFIX}{r.leg}"] = r.option_lots_mc
    for r in legs.itertuples():
        row[f"{VEGA_PREFIX}{r.leg}"] = r.vega_usd_per_pt_mc
    return row, legs


def grid(d: deal_mod.Deal, cfg: dict) -> pd.DataFrame:
    """Extrinsic vs one correlation and a vol scale; skipped if the legs are absent.

    Kirk where it applies (fast); Monte Carlo when the trade can cancel, which
    Kirk cannot price. The `method` column says which produced each row.
    """
    g = cfg["grid"]
    names = list(d.model.names)
    a, b = g["corr_legs"]
    if a not in names or b not in names:
        return pd.DataFrame()
    i, j = names.index(a), names.index(b)
    m, p = d.model, d.payoff
    intrinsic = p.intrinsic(m.fwd)
    rows = []
    for scale in g["vol_scale"]:
        for rho in g["corr"]:
            c = m.corr.copy()
            c[i, j] = c[j, i] = rho
            bumped = m.bumped(vol=m.vol * scale, corr=pricer.nearest_correlation(c))
            if p.can_cancel:
                method = "mc"
                v = pricer.value_mc(bumped, p, cfg["n_paths"] // 4, cfg["seed"])["value_mc"]
            else:
                method = "kirk"
                v = pricer.value_kirk_basket(bumped, p)["value"]
            rows.append({"trade": d.trade.id, "corr_legs": f"{a}/{b}", "corr": rho,
                         "vol_scale": scale, "method": method, "extrinsic": v - intrinsic})
    return pd.DataFrame(rows)


def portfolio(legs: pd.DataFrame) -> pd.DataFrame:
    if legs.empty:
        return legs
    return (legs.groupby(["future", "hub", "month"], as_index=False)
            [["qty_native_mc", "lots_mc", "option_lots_mc", "vega_usd_per_pt_mc"]].sum()
            .sort_values(["hub", "month"]))


def trades_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """summary_columns() of trades.csv, numbers rounded to SUMMARY_DECIMALS."""
    if trades.empty:
        return trades
    cols = summary_columns(trades.columns)
    missing = [c for c in cols if c not in trades.columns]
    if missing:
        raise KeyError(f"trades_summary.csv: columns not produced by this run: {missing}")
    return trades[cols].round({c: SUMMARY_DECIMALS for c in cols if c not in SUMMARY_TERMS})


def write_trades_csv(path: Path, trades: pd.DataFrame, bump_corr: float, bump_vol: float = 0.01):
    """Header, then a units row, then the data.

    Read it back with pd.read_csv(path, skiprows=[1]); without skipping the
    units row every numeric column comes back as text.
    """
    if trades.empty:
        trades.to_csv(path, index=False)
        return
    units = pd.DataFrame([trade_units(trades.columns, bump_corr, bump_vol)], columns=trades.columns)
    units.to_csv(path, index=False)
    trades.to_csv(path, index=False, header=False, mode="a")


def write(folder: Path, result: dict, snap: snapshot.Snapshot, as_of: pd.Timestamp, market: dict):
    folder.mkdir(parents=True, exist_ok=True)
    bumps = (market["pricing"]["bump_corr"], market["pricing"]["bump_vol"])
    write_trades_csv(folder / "trades.csv", result["trades"], *bumps)
    write_trades_csv(folder / "trades_summary.csv", trades_summary(result["trades"]), *bumps)
    result["legs"].to_csv(folder / "legs.csv", index=False)
    port = portfolio(result["legs"])
    port.to_csv(folder / "portfolio.csv", index=False)
    if not result["grids"].empty:
        result["grids"].to_csv(folder / "grids.csv", index=False)
    (folder / "run.json").write_text(json.dumps({
        "as_of": str(as_of.date()), "calibration": str(snap.path),
        "calibration_as_of": str(snap.as_of.date()), "pricing": market["pricing"],
        "rate": market["rate"], "notes": result["notes"], "errors": result["errors"],
    }, indent=2, default=str))
    _report(result, port, snap, as_of, haircuts(market["pricing"], market["hubs"]))


def _report(result: dict, port: pd.DataFrame, snap, as_of, cuts: dict | None = None):
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(RULE)
    print(f"Pricing as of {as_of:%Y-%m-%d} off calibration {snap.path.name}")
    if as_of != snap.as_of:
        print(f"  NOTE: calibration is as of {snap.as_of:%Y-%m-%d}; forwards and vols are that day's")
    for kind, per_key in (cuts or {}).items():
        for key, h in per_key.items():
            if h == 1.0:
                continue
            if kind == "cross":
                print(f"  HAIRCUT: {key[0]}/{key[1]} cross correlation (all 4 pairs) x{h:g}")
            else:
                what = "vols" if kind == "vol" else "M/M+1 correlation"
                print(f"  HAIRCUT: {key} {what} x{h:g}")
    print(RULE)
    tr = result["trades"]
    if tr.empty:
        print("No trades priced.")
        return
    print("option = right to switch from M to M+1, always >= 0 | "
          "value = whole delivery (margin + option), can be < 0 | _mc = Monte Carlo")
    cols = ["trade", "m0", "m1", "decision_date", "option_value_mc", "option_stderr_mc",
            "option_intrinsic", "option_time_value_mc", "option_value_usd_mc",
            "value_mc", "value_usd_mc", "prob_m1_mc", "prob_cancel_mc"]
    print(tr[cols].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nTotal option value (MC) USD {tr['option_value_usd_mc'].sum():,.0f}   "
          f"(time value USD {tr['extrinsic_usd_mc'].sum():,.0f})   "
          f"total deal value (MC) USD {tr['value_usd_mc'].sum():,.0f}")

    print(f"\n{RULE}\nInputs and Monte Carlo risk per leg")
    lc = ["trade", "leg", "future", "fwd", "vol_file", "vol_horizon", "vol", "corr_used_M_M1",
          "lots_mc", "option_lots_mc", "vega_usd_per_pt_mc"]
    print(result["legs"][lc].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print(f"\n{RULE}\nPortfolio by contract, Monte Carlo (hedge = minus lots)")
    print(port.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    for tid, n in result["notes"].items():
        if n:
            print(f"  [{tid}] {'; '.join(n)}")
    for tid, e in result["errors"].items():
        print(f"  [{tid}] NOT PRICED: {e}")


def _abs(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


if __name__ == "__main__":
    main()
