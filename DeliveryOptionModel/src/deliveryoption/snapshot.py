"""The file contract between the calibration job and the pricing job.

A calibration is one folder, written once by the calibration job and then
free to be edited by hand before pricing:

    contracts.csv                 one row per (hub, month). The pricer reads
                                  `fwd` and `vol`; `vol` starts equal to
                                  `atm_vol` (the Bloomberg read) and is the
                                  column to edit. It is the vol to the option
                                  expiry, before any horizon adjustment.
    fx.csv                        ticker, rate
    vol_profile.csv               realised vol by hub and days to expiry; only
                                  its shape is used (see vols.py)
    correlations/<M>_<M1>.csv     leg correlation matrix per month pair,
                                  weekly returns - used by the pricer
    correlations/<M>_<M1>_daily.csv   same windows, daily - diagnostic only
    correlation_summary.csv       n_obs, window and headline numbers per pair
    meta.json                     as-of date and the settings that produced it

To keep an edited set, copy the folder under a name that sorts after the
original (2026-09-23 -> 2026-09-23_edited): `latest` picks it up.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CORR_DIR = "correlations"


@dataclass(frozen=True)
class Snapshot:
    path: Path
    as_of: pd.Timestamp
    contracts: pd.DataFrame        # indexed by (hub, month)
    fx: dict[str, float]
    vol_profile: pd.DataFrame
    meta: dict

    def contract(self, hub: str, month: pd.Period) -> pd.Series:
        key = (hub, str(month))
        if key not in self.contracts.index:
            raise KeyError(f"{hub} {month} not in {self.path / 'contracts.csv'} - "
                           "re-run calibration with this month in trades.yaml or the strip")
        row = self.contracts.loc[key]
        if pd.isna(row.get("fwd")) or pd.isna(row.get("vol")):
            raise ValueError(f"{hub} {month}: no forward or vol in the calibration "
                             f"({row.get('note') or 'blank'}) - fill it in contracts.csv")
        return row

    def pair_corr(self, m0: pd.Period, m1: pd.Period) -> pd.DataFrame:
        f = self.path / CORR_DIR / f"{m0}_{m1}.csv"
        if not f.exists():
            raise FileNotFoundError(f"No correlation for {m0}/{m1} ({f}) - re-run calibration")
        return pd.read_csv(f, index_col=0)


def write(folder: Path, as_of: pd.Timestamp, contracts: pd.DataFrame, fx: dict,
          vol_profile: pd.DataFrame, pairs: list, pair_summary: pd.DataFrame, meta: dict):
    (folder / CORR_DIR).mkdir(parents=True, exist_ok=True)
    contracts.to_csv(folder / "contracts.csv", index=False)
    pd.Series(fx, name="rate").rename_axis("ticker").to_csv(folder / "fx.csv")
    vol_profile.to_csv(folder / "vol_profile.csv", index=False)
    for p in pairs:
        p.corr.to_csv(folder / CORR_DIR / f"{p.m0}_{p.m1}.csv", float_format="%.6f")
        p.corr_daily.to_csv(folder / CORR_DIR / f"{p.m0}_{p.m1}_daily.csv", float_format="%.6f")
    pair_summary.to_csv(folder / "correlation_summary.csv", index=False)
    (folder / "meta.json").write_text(json.dumps({"as_of": str(as_of.date()), **meta},
                                                 indent=2, default=str))


def read(folder: Path) -> Snapshot:
    meta = json.loads((folder / "meta.json").read_text())
    contracts = pd.read_csv(folder / "contracts.csv", parse_dates=["fixing", "option_expiry"])
    contracts["month"] = contracts["month"].astype(str)
    fx = pd.read_csv(folder / "fx.csv", index_col=0)["rate"].to_dict()
    return Snapshot(path=folder, as_of=pd.Timestamp(meta["as_of"]),
                    contracts=contracts.set_index(["hub", "month"]), fx=fx,
                    vol_profile=pd.read_csv(folder / "vol_profile.csv"), meta=meta)


def resolve(root: Path, which: str) -> Path:
    """'latest' -> the last folder by name under root; else a name or a path."""
    if which == "latest":
        folders = sorted(p for p in root.iterdir() if (p / "meta.json").exists())
        if not folders:
            raise FileNotFoundError(f"No calibration under {root} - run calibrate.py first")
        return folders[-1]
    p = Path(which)
    return p if p.is_absolute() else root / which
