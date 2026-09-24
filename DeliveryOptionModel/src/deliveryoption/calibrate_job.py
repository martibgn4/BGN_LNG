"""Job 1: calibrate vols and correlations from Bloomberg and write them to files.

    python calibrate.py                     # pairs from trades.yaml + the strip
    python calibrate.py --strip-months 18

Writes data/calibration/<as_of>/ (see snapshot.py for the file layout), the
price history it used in history/, and diagnostics/ plots (diagnostics.py). Never
overwrites: a second run on the same day gets a time suffix, so a folder that
has been edited by hand is safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from DeliveryOptionModel.src.deliveryoption import calibration, diagnostics, market_data, snapshot

ROOT = Path(__file__).resolve().parents[2]
RULE = "-" * 78


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default=str(ROOT / "config" / "market.yaml"))
    ap.add_argument("--trades", default=str(ROOT / "config" / "trades_2028_2031.yaml"))
    ap.add_argument("--strip-months", type=int, help="override calibration.strip_months")
    ap.add_argument("--no-strip", action="store_true", help="only the pairs in --trades")
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="skip the plots in <calibration>/diagnostics/ (python diagnose.py later)")
    args = ap.parse_args(argv)

    market = yaml.safe_load(Path(args.market).read_text())
    trades = yaml.safe_load(Path(args.trades).read_text())
    if args.strip_months is not None:
        market["calibration"]["strip_months"] = args.strip_months
    if args.no_strip:
        market["calibration"]["strip_months"] = 0
    folder = run(market, trades, as_of=pd.Timestamp.today().normalize())
    print(f"\nCalibration written to {folder}")
    if not args.no_diagnostics:
        # Diagnostics are a view on the calibration, not part of it: a plotting
        # failure must not lose a finished calibration.
        try:
            print(f"Diagnostics written to {diagnostics.run(folder, fetch=False)}")
        except Exception as e:                                  # noqa: BLE001
            print(f"Diagnostics failed ({e}); rerun with: python diagnose.py --calib {folder.name}")


def pairs_to_calibrate(market: dict, trades: dict, as_of: pd.Timestamp) -> list[tuple[pd.Period, pd.Period]]:
    """Every pair a trade uses, plus consecutive pairs along the strip.

    The strip means a new consecutive CSO within it can be priced without
    re-calibrating.
    """
    pairs = {tuple(pd.Period(m, "M") for m in t["months"]) for t in trades.get("trades", [])}
    first = as_of.to_period("M") + 1
    for i in range(market["calibration"]["strip_months"]):
        pairs.add((first + i, first + i + 1))
    return sorted(pairs)


def run(market: dict, trades: dict, as_of: pd.Timestamp) -> Path:
    hub_specs, cal = market["hubs"], market["calibration"]
    hubs = list(hub_specs)
    pairs = pairs_to_calibrate(market, trades, as_of)

    # ---- live: forwards, fixings, ATM vols, FX ------------------------------
    months = sorted({m for p in pairs for m in p})
    print(f"Fetching {len(months)} months x {len(hubs)} hubs: {months[0]} .. {months[-1]}")
    contracts = market_data.fetch_contracts(hub_specs, {h: months for h in hubs})
    fx = market_data.fetch_fx(sorted({s["fx"] for s in hub_specs.values() if s.get("fx")}))
    contracts["fx"] = np.nan
    for hub, spec in hub_specs.items():
        if spec.get("fx"):
            fwd_fx = market_data.fetch_fx_forwards(market["fx_forwards"][spec["fx"]], months, as_of)
            is_hub = contracts["hub"] == hub
            contracts.loc[is_hub, "fx"] = contracts.loc[is_hub, "month"].map(fwd_fx)

    fixing = (contracts.dropna(subset=["fixing"])
              .assign(fixing=lambda d: pd.to_datetime(d["fixing"]))
              .groupby("month")["fixing"].min())
    live_pairs = [p for p in pairs if str(p[0]) in fixing.index and fixing[str(p[0])] > as_of]
    dropped = sorted(set(pairs) - set(live_pairs))
    if dropped:
        print(f"Skipping pairs whose first month has fixed or has no price: {dropped}")

    # ---- history: one pull covering every pair and the vol profile ----------
    windows, anchors = {}, {}
    for m0, m1 in live_pairs:
        days_to_fix = (fixing[str(m0)] - as_of).days
        windows[(m0, m1)] = (0, max(days_to_fix, cal["min_window_days"]))
        anchors[(m0, m1)] = calibration.anchor_months(m0, as_of, cal["lookback_months"],
                                                      cal["anchor_filter"] == "same_pair")
    hist_months = {m for (m0, m1), ks in anchors.items() for k in ks for m in (k, k + (m1 - m0).n)}
    reach = max([w[1] for w in windows.values()] + [cal["vol_profile_buckets_days"][-1]])
    start = min(hist_months).start_time - pd.Timedelta(days=reach + 31)
    print(f"Fetching history for {len(hist_months) * len(hubs)} contracts from {start:%Y-%m-%d}")
    px, expiry = calibration.fetch_history(hub_specs, hubs, hist_months, start, as_of)

    fitted, summary = [], []
    for (m0, m1), w in windows.items():
        try:
            p = calibration.calibrate_pair(px, expiry, hub_specs, hubs, m0, m1, anchors[(m0, m1)],
                                           w, cal["return_days"], as_of)
        except ValueError as e:
            summary.append({"pair": calibration.pair_key(m0, m1), "error": str(e)})
            continue
        fitted.append(p)
        row = {"pair": calibration.pair_key(m0, m1), "window_lo": w[0], "window_hi": w[1],
               "n_obs": p.n_obs, "n_anchors": p.n_anchors, "return_days": p.return_days}
        for h in hubs:
            row[f"{h}_M/{h}_M1"] = p.corr.loc[f"{h}_M", f"{h}_M1"]
            row[f"{h}_M/{h}_M1 daily"] = p.corr_daily.loc[f"{h}_M", f"{h}_M1"]
        for a, b in ((a, b) for i, a in enumerate(hubs) for b in hubs[i + 1:]):
            row[f"{a}_M/{b}_M"] = p.corr.loc[f"{a}_M", f"{b}_M"]
            row[f"{a}_M/{b}_M daily"] = p.corr_daily.loc[f"{a}_M", f"{b}_M"]
        row["max_stale_share"] = float(p.stale_share.max())
        summary.append(row)
    summary = pd.DataFrame(summary)

    profile = calibration.vol_profile(px, expiry, hub_specs, hubs, cal["vol_profile_buckets_days"],
                                      cal["return_days"], as_of)
    contracts = market_data.fill_missing_vols(contracts, cal["missing_vol_fill"], profile, as_of,
                                              cal["fill_scale_months"])

    folder = Path(cal["output_dir"])
    folder = (folder if folder.is_absolute() else ROOT / folder) / f"{as_of:%Y-%m-%d}"
    if folder.exists():
        folder = folder.with_name(f"{folder.name}_{pd.Timestamp.now():%H%M%S}")
    snapshot.write(folder, as_of, contracts, fx, profile, fitted, summary,
                   meta={"created": pd.Timestamp.now().isoformat(timespec="seconds"),
                         "calibration": cal, "hubs": hub_specs,
                         "pairs": [calibration.pair_key(*p) for p in live_pairs]})
    diagnostics.save_history(folder, px, expiry)
    _report(contracts, fx, profile, summary)
    return folder


def _report(contracts, fx, profile, summary):
    pd.set_option("display.width", 160)
    print(RULE)
    print("Contracts (vol = ATM implied to option expiry; edit `vol` in contracts.csv)")
    cols = ["hub", "month", "future", "fwd", "fixing", "fx", "atm_vol", "vol", "option_expiry", "note"]
    print(contracts[cols].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"\nFX: {fx}")
    print(f"\n{RULE}\nRealised vol by days to expiry")
    print(profile.pivot(index=["tau_lo_days", "tau_hi_days"], columns="hub", values="realised_vol")
          .to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n{RULE}\nPair correlations")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if "error" in summary and summary["error"].notna().any():
        print(f"\n{summary['error'].notna().sum()} pair(s) failed - see the error column")


if __name__ == "__main__":
    main()
