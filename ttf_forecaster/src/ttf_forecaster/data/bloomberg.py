"""Bloomberg market-data adapter (``MarketDataAdapter``) with mock, local-file and live backends.

* **Mock** - synthetic data from :mod:`ttf_forecaster.data.synthetic`, labelled ``SYNTHETIC``.
  Serves every configured series and curve regardless of identifier status (nothing is requested
  from a vendor). Never evidence about TTF.
* **LocalFile** - user-placed extracts under ``data_sources.yaml`` ``local_file.extracts_dir``:

  ``observations/<series_id>.parquet|csv``
      required ``event_time`` (ISO date = local date in the series' ``tz``, or tz-aware ISO
      timestamp), ``value``; optional ``publication_time`` (tz-aware; if absent the series'
      ``publication_lag`` rule is applied and ``publication_time_estimated=True``), ``revision``.
  ``curves/<hub>.parquet|csv``
      required ``trade_date`` (local date), ``value``, and ``contract_id`` (``TTF.M.2026-11``)
      **or** ``ticker`` (``TZTX26 Comdty``, parsed with the configured rule); optional ``field``
      (default ``settle``), ``publication_time``.
  ``contracts/<hub>.parquet|csv``
      ``contract_id`` or ``ticker``, ``last_trade_date`` (vendor metadata, local date).

  ``ingestion_time`` is the file's modification time (UTC).
* **Live** - xbbg against a logged-in Terminal. Construction refuses unless
  ``live_calls_authorised: true`` (set only in git-ignored ``configs/local_data_sources.yaml`` on
  the user's instruction) and xbbg/blpapi are importable at a supported xbbg version; every request
  refuses identifiers whose status is not ``known``. No live call runs in tests.

Every series is served from ``configs/sources/bloomberg*.yaml`` (``series:``); futures curves from
``configs/sources/bloomberg.yaml`` (``curves:``). publication_time comes from each entry's
``publication_lag`` via :func:`apply_publication_lag`; ``pit_grade`` from config.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.contracts.calendars import TradingCalendar
from ttf_forecaster.contracts.futures import (
    BloombergCurveIds,
    ExpiryRule,
    FuturesContract,
    bloomberg_ticker,
    bloomberg_ticker_candidates,
    contract_metadata,
    contracts_between,
    parse_bloomberg_ticker,
    parse_contract_id,
)
from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import (
    CONTRACT_METADATA_SCHEMA,
    CURVE_SCHEMA,
    MARKET_TZ,
    OBSERVATION_SCHEMA,
    SYNTHETIC_SOURCE,
    Backend,
    BackendUnavailable,
    DataError,
    Entitlement,
    IdentifierStatus,
    IdentifierUnknown,
    PitGrade,
    PublicationLag,
    QualityFlag,
    apply_publication_lag,
    empty_frame,
    validate_frame,
)
from ttf_forecaster.settings import CONFIG_DIR, PROJECT_ROOT, ConfigError, get, load_config

log = logging.getLogger(__name__)

BLOOMBERG_SOURCE = "bloomberg"
_REQUIRED_SERIES_KEYS = ("unit", "status", "pit_grade", "publication_lag")
_CURVE_FIELD_KEYS = {"settle": "settle", "bid": "bid", "ask": "ask", "volume": "volume",
                     "open_interest": "open_interest"}


# ---------------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesSpec:
    """One ``series:`` entry (shared per-series schema in ``configs/data_sources.yaml``)."""

    series_id: str
    ticker: str | None
    field: str | None
    unit: str
    status: str
    pit_grade: str
    lag: PublicationLag
    geography: str
    entitlement: str
    source_file: str
    provenance: str = ""

    @property
    def is_known(self) -> bool:
        return (self.status == IdentifierStatus.KNOWN.value and bool(self.ticker)
                and bool(self.field))


@dataclass(frozen=True)
class CurveSpec:
    """One ``curves:`` entry of ``configs/sources/bloomberg.yaml``."""

    hub: str
    ids: BloombergCurveIds
    unit: str
    currency: str
    status: str
    pit_grade: str
    lag: PublicationLag
    entitlement: str


@dataclass
class BloombergConfig:
    """Merged view of ``data_sources.yaml`` and every ``configs/sources/bloomberg*.yaml``."""

    series: dict[str, SeriesSpec]
    curves: dict[str, CurveSpec]
    fields: dict[str, dict[str, Any]]
    supported_versions: tuple[str, ...]
    live_calls_authorised: bool
    extracts_dir: Path
    config_dir: str | None = None
    files: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, config_dir: str | None = None) -> BloombergConfig:
        base = Path(config_dir) if config_dir else CONFIG_DIR
        ds = load_config("data_sources", config_dir)
        files = sorted(p for p in (base / "sources").glob("bloomberg*.yaml")
                       if not p.name.startswith("local_"))
        if not files:
            raise ConfigError(f"No bloomberg*.yaml under {base / 'sources'}")
        series: dict[str, SeriesSpec] = {}
        main: dict[str, Any] = {}
        for path in files:
            cfg = load_config(f"sources/{path.stem}", config_dir)
            if path.stem == "bloomberg":
                main = cfg
            for sid, entry in (cfg.get("series") or {}).items():
                if sid in series:
                    raise ConfigError(f"series {sid!r} defined in both {series[sid].source_file} "
                                      f"and {path.name}")
                series[sid] = _series_spec(sid, entry or {}, path.name)
        if not main:
            raise ConfigError("configs/sources/bloomberg.yaml is required (curves, fields)")
        rule = get(main, "ticker_rule")
        curves = {
            hub: CurveSpec(
                hub=hub,
                ids=BloombergCurveIds.from_mappings(hub, c, rule),
                unit=str(c["unit"]),
                currency=str(c["currency"]),
                status=str(c["status"]),
                pit_grade=str(c["pit_grade"]),
                lag=PublicationLag.from_config(c["publication_lag"]),
                entitlement=str(c.get("entitlement", Entitlement.LICENSED_BLOOMBERG.value)),
            )
            for hub, c in (main.get("curves") or {}).items()
        }
        extracts = Path(str(get(ds, "local_file", "extracts_dir")))
        if not extracts.is_absolute():
            extracts = (base.parent if config_dir else PROJECT_ROOT) / extracts
        return cls(
            series=series,
            curves=curves,
            fields=dict(main.get("fields") or {}),
            supported_versions=tuple(str(v) for v in get(main, "client", "supported_versions")),
            live_calls_authorised=bool(ds.get("live_calls_authorised", False)),
            extracts_dir=extracts,
            config_dir=config_dir,
            files=[p.name for p in files],
        )

    def spec(self, series_id: str) -> SeriesSpec:
        try:
            return self.series[series_id]
        except KeyError as exc:
            raise IdentifierUnknown(
                f"series {series_id!r} is not configured in any configs/sources/bloomberg*.yaml"
            ) from exc

    def curve(self, hub: str) -> CurveSpec:
        try:
            return self.curves[hub]
        except KeyError as exc:
            raise IdentifierUnknown(f"hub {hub!r} has no curves entry in bloomberg.yaml") from exc

    def field_name(self, key: str) -> str:
        """Vendor field name for logical ``key``; raises unless its status is known."""
        f = self.fields.get(key)
        if not f or f.get("status") != IdentifierStatus.KNOWN.value or not f.get("name"):
            raise IdentifierUnknown(f"Bloomberg field {key!r} is not known "
                                    f"(status {None if not f else f.get('status')!r})")
        return str(f["name"])


def _series_spec(sid: str, entry: Mapping[str, Any], fname: str) -> SeriesSpec:
    missing = [k for k in _REQUIRED_SERIES_KEYS if entry.get(k) is None]
    if missing:
        raise ConfigError(f"{fname}: series {sid!r} lacks {missing}")
    status = str(entry["status"])
    if status not in {s.value for s in IdentifierStatus}:
        raise ConfigError(f"{fname}: series {sid!r} has invalid status {status!r}")
    grade = str(entry["pit_grade"])
    if grade not in {g.value for g in PitGrade}:
        raise ConfigError(f"{fname}: series {sid!r} has invalid pit_grade {grade!r}")
    return SeriesSpec(
        series_id=sid,
        ticker=entry.get("ticker"),
        field=entry.get("field"),
        unit=str(entry["unit"]),
        status=status,
        pit_grade=grade,
        lag=PublicationLag.from_config(entry["publication_lag"]),
        geography=str(entry.get("geography") or "UNKNOWN"),
        entitlement=str(entry.get("entitlement") or Entitlement.LICENSED_BLOOMBERG.value),
        source_file=fname,
        provenance=str(entry.get("provenance") or ""),
    )


# ---------------------------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------------------------


def _local_dates_to_utc(days: pd.Series | pd.DatetimeIndex, tz: str) -> pd.Series:
    idx = pd.DatetimeIndex(days)
    if idx.tz is None:
        idx = idx.normalize().tz_localize(tz)
    return pd.Series(idx.tz_convert("UTC"))


def _observation_frame(spec: SeriesSpec, event_utc: pd.Series, values: np.ndarray, *,
                       source: str, publication: pd.Series | None, ingestion: pd.Series | None,
                       revision: np.ndarray | None = None, synthetic_data: bool = False
                       ) -> pd.DataFrame:
    n = len(event_utc)
    event_utc = pd.Series(pd.DatetimeIndex(event_utc).tz_convert("UTC"))
    estimated = publication is None
    pub = apply_publication_lag(event_utc, spec.lag) if publication is None else publication
    pub = pd.Series(pd.DatetimeIndex(pub).tz_convert("UTC"))
    ing = pub + pd.Timedelta(minutes=5) if ingestion is None else ingestion
    df = pd.DataFrame(
        {
            "series_id": [spec.series_id] * n,
            "event_time": event_utc.to_numpy(),
            "value": np.asarray(values, dtype=float),
            "geography": [spec.geography] * n,
            "publication_time": pub.to_numpy(),
            "ingestion_time": pd.Series(pd.DatetimeIndex(ing).tz_convert("UTC")).to_numpy(),
            "revision": (np.zeros(n, dtype="int64") if revision is None
                         else np.asarray(revision, dtype="int64")),
            "unit": [spec.unit] * n,
            "source": [source] * n,
            "quality_flag": [(QualityFlag.SYNTHETIC if synthetic_data else QualityFlag.OK).value] * n,
            "entitlement": [(Entitlement.SYNTHETIC.value if synthetic_data else spec.entitlement)] * n,
            "pit_grade": [spec.pit_grade] * n,
            "publication_time_estimated": np.full(n, estimated),
            "source_timezone": [spec.lag.tz] * n,
        }
    )
    for col in ("event_time", "publication_time", "ingestion_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df.loc[df["value"].isna(), "quality_flag"] = QualityFlag.MISSING_INPUT.value
    return validate_frame(df, OBSERVATION_SCHEMA)


def _curve_frame(spec: CurveSpec, rows: pd.DataFrame, *, source: str, synthetic_data: bool,
                 ingestion: pd.Timestamp | None = None) -> pd.DataFrame:
    """CURVE_SCHEMA from ``rows`` with ``trade_day`` (naive local), ``contract_id``, ``field``,
    ``value`` and optional ``publication_time``."""
    if rows.empty:
        return empty_frame(CURVE_SCHEMA)
    contracts = {cid: parse_contract_id(cid) for cid in rows["contract_id"].unique()}
    tz = MARKET_TZ
    trade_utc = _local_dates_to_utc(rows["trade_day"], tz)
    pub = (apply_publication_lag(trade_utc, spec.lag)
           if "publication_time" not in rows or rows["publication_time"].isna().all()
           else pd.Series(pd.to_datetime(rows["publication_time"], utc=True).to_numpy()))
    n = len(rows)
    ds = [contracts[c].delivery_start for c in rows["contract_id"]]
    de = [contracts[c].delivery_end for c in rows["contract_id"]]
    df = pd.DataFrame(
        {
            "hub": [spec.hub] * n,
            "contract_id": rows["contract_id"].to_numpy(),
            "tenor_type": [contracts[c].tenor_type for c in rows["contract_id"]],
            "delivery_start": _local_dates_to_utc(pd.DatetimeIndex(ds), tz).to_numpy(),
            "delivery_end": _local_dates_to_utc(pd.DatetimeIndex(de), tz).to_numpy(),
            "trade_date": trade_utc.to_numpy(),
            "field": rows["field"].to_numpy(),
            "value": rows["value"].to_numpy(dtype=float),
            "currency": [spec.currency] * n,
            "publication_time": pd.DatetimeIndex(pub).tz_convert("UTC"),
            "ingestion_time": (pd.DatetimeIndex(pub).tz_convert("UTC") + pd.Timedelta(minutes=5)
                               if ingestion is None else [ingestion] * n),
            "revision": np.zeros(n, dtype="int64"),
            "unit": [spec.unit] * n,
            "source": [source] * n,
            "quality_flag": [(QualityFlag.SYNTHETIC if synthetic_data else QualityFlag.OK).value] * n,
            "entitlement": [(Entitlement.SYNTHETIC.value if synthetic_data else spec.entitlement)] * n,
            "pit_grade": [spec.pit_grade] * n,
            "publication_time_estimated": np.full(n, "publication_time" not in rows
                                                  or rows["publication_time"].isna().all()),
            "source_timezone": [tz] * n,
        }
    )
    for col in ("delivery_start", "delivery_end", "trade_date", "publication_time",
                "ingestion_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return validate_frame(df, CURVE_SCHEMA)


def _in_range(days: pd.Series, start: date, end: date) -> pd.Series:
    d = pd.DatetimeIndex(days).normalize()
    return pd.Series((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)), index=days.index)


# ---------------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------------


class _BloombergBase:
    backend: Backend

    def __init__(self, config: BloombergConfig | None = None, *, config_dir: str | None = None
                 ) -> None:
        self.config = config or BloombergConfig.load(config_dir)

    def _expiry_metadata(self, contract_ids: Sequence[str], calendar: TradingCalendar,
                         vendor: Mapping[str, Any] | None = None) -> pd.DataFrame:
        frames = []
        by_hub: dict[str, list[FuturesContract]] = {}
        for cid in contract_ids:
            c = parse_contract_id(cid)
            by_hub.setdefault(c.hub, []).append(c)
        for hub, cs in by_hub.items():
            spec = self.config.curve(hub)
            rule = ExpiryRule.from_config(hub, self.config.config_dir)
            exp_cfg = (((load_config("instruments", self.config.config_dir).get("hubs") or {})
                        .get(hub) or {}).get("expiry") or {})
            frames.append(contract_metadata(cs, calendar, rule, vendor_last_trade=vendor,
                                            currency=spec.currency, unit=spec.unit,
                                            vendor_authoritative=bool(
                                                exp_cfg.get("vendor_authoritative", False))))
        if not frames:
            return empty_frame(CONTRACT_METADATA_SCHEMA)
        return validate_frame(pd.concat(frames, ignore_index=True), CONTRACT_METADATA_SCHEMA)


class MockBloombergAdapter(_BloombergBase):
    """Synthetic backend. Weekday calendar, deterministic per series/hub seed."""

    backend = Backend.MOCK

    def __init__(self, config: BloombergConfig | None = None, *, config_dir: str | None = None,
                 synthetic_config: synthetic.SyntheticConfig | None = None) -> None:
        super().__init__(config, config_dir=config_dir)
        self.syn = synthetic_config or synthetic.SyntheticConfig()

    def _syn_cfg(self, start: date, end: date, key: str) -> synthetic.SyntheticConfig:
        seed = self.syn.seed if key == "TTF" else self.syn.seed + zlib.crc32(key.encode()) % 10_000
        return synthetic.SyntheticConfig(
            start=str(pd.Timestamp(start).date()), end=str(pd.Timestamp(end).date()), seed=seed,
            world=self.syn.world, daily_vol=self.syn.daily_vol,
            months_listed=self.syn.months_listed,
            expiry_bdays_before_delivery=self.syn.expiry_bdays_before_delivery,
        )

    def get_historical_series(self, series_ids: Sequence[str], start: date, end: date
                              ) -> pd.DataFrame:
        frames = []
        for sid in series_ids:
            spec = self.config.spec(sid)
            if sid == "storage.eu.inventory_twh":
                df = synthetic.storage_observations(self._syn_cfg(start, end, sid))
                frames.append(df)
                continue
            rng = np.random.default_rng(zlib.crc32(sid.encode()))
            days = pd.bdate_range(start, end)
            values = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(days))))
            frames.append(_observation_frame(
                spec, _local_dates_to_utc(days, spec.lag.tz), values, source=SYNTHETIC_SOURCE,
                publication=None, ingestion=None, synthetic_data=True))
        if not frames:
            return empty_frame(OBSERVATION_SCHEMA)
        return validate_frame(pd.concat(frames, ignore_index=True), OBSERVATION_SCHEMA)

    def get_reference_data(self, instrument_ids: Sequence[str], fields: Sequence[str]
                           ) -> pd.DataFrame:
        """Only ``last_trade_date`` is mockable (rule-computed); other fields are NaN."""
        meta = self.get_contract_metadata(instrument_ids)
        ltd = dict(zip(meta["contract_id"], meta["last_trade_date"], strict=True))
        rows = [{"instrument_id": i, "field": f,
                 "value": ltd.get(i) if f == "last_trade_date" else np.nan}
                for i in instrument_ids for f in fields]
        return pd.DataFrame(rows, columns=["instrument_id", "field", "value"])

    def get_curve_snapshot(self, hub: str, start: date, end: date,
                           fields: Sequence[str] = ("settle",)) -> pd.DataFrame:
        spec = self.config.curve(hub)
        if set(fields) - {"settle"}:
            raise IdentifierUnknown(f"mock serves only 'settle', requested {list(fields)}")
        curve = synthetic.ttf_curve(self._syn_cfg(start, end, hub))
        rows = pd.DataFrame({
            "trade_day": curve["trade_date"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None),
            "contract_id": curve["contract_id"].str.replace("TTF.", f"{hub}.", n=1, regex=False),
            "field": "settle",
            "value": curve["value"].to_numpy(),
        })
        return _curve_frame(spec, rows.reset_index(drop=True), source=SYNTHETIC_SOURCE,
                            synthetic_data=True)

    def get_contract_metadata(self, contract_ids: Sequence[str]) -> pd.DataFrame:
        cal = TradingCalendar(future_holidays=(), rule_start="2000-01-03")  # synthetic weekdays
        return self._expiry_metadata(contract_ids, cal)


def _read_table(path_stem: Path) -> tuple[pd.DataFrame, pd.Timestamp]:
    for ext in (".parquet", ".csv"):
        p = path_stem.parent / (path_stem.name + ext)  # series ids contain dots
        if p.exists():
            df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
            mtime = pd.Timestamp(p.stat().st_mtime, unit="s", tz="UTC")
            return df, mtime
    raise DataError(f"No extract {path_stem}.parquet|.csv (see data/bloomberg.py docstring)")


def _require(df: pd.DataFrame, cols: Sequence[str], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"{what}: missing columns {missing}")


class LocalFileBloombergAdapter(_BloombergBase):
    """Reads user-placed extracts (layout in the module docstring)."""

    backend = Backend.LOCAL_FILE

    def __init__(self, config: BloombergConfig | None = None, *, config_dir: str | None = None,
                 extracts_dir: str | Path | None = None,
                 calendar: TradingCalendar | None = None) -> None:
        super().__init__(config, config_dir=config_dir)
        self.root = Path(extracts_dir) if extracts_dir else self.config.extracts_dir
        self.calendar = calendar

    def get_historical_series(self, series_ids: Sequence[str], start: date, end: date
                              ) -> pd.DataFrame:
        frames = []
        for sid in series_ids:
            spec = self.config.spec(sid)
            df, mtime = _read_table(self.root / "observations" / sid)
            _require(df, ("event_time", "value"), f"observations/{sid}")
            ev = pd.to_datetime(df["event_time"], utc=False)
            if getattr(ev.dt, "tz", None) is None:
                ev_utc = _local_dates_to_utc(ev.dt.normalize(), spec.lag.tz)
            else:
                ev_utc = pd.Series(pd.DatetimeIndex(ev).tz_convert("UTC"))
            local = pd.Series(pd.DatetimeIndex(ev_utc).tz_convert(spec.lag.tz).tz_localize(None))
            keep = _in_range(local, start, end).to_numpy()
            pub = None
            if "publication_time" in df and df["publication_time"].notna().all():
                pub_all = pd.to_datetime(df["publication_time"], utc=True)
                pub = pd.Series(pub_all.to_numpy()[keep])
            rev = df["revision"].to_numpy()[keep] if "revision" in df else None
            frames.append(_observation_frame(
                spec, ev_utc[keep].reset_index(drop=True), df["value"].to_numpy(dtype=float)[keep],
                source=BLOOMBERG_SOURCE, publication=pub,
                ingestion=pd.Series([mtime] * int(keep.sum())), revision=rev))
        if not frames:
            return empty_frame(OBSERVATION_SCHEMA)
        return validate_frame(pd.concat(frames, ignore_index=True), OBSERVATION_SCHEMA)

    def _contract_ids(self, df: pd.DataFrame, spec: CurveSpec) -> pd.Series:
        if "contract_id" in df:
            return df["contract_id"].astype(str)
        if "ticker" in df:
            return df["ticker"].map(lambda t: parse_bloomberg_ticker(str(t), spec.ids).contract_id)
        raise DataError("extract needs a contract_id or ticker column")

    def get_curve_snapshot(self, hub: str, start: date, end: date,
                           fields: Sequence[str] = ("settle",)) -> pd.DataFrame:
        spec = self.config.curve(hub)
        df, mtime = _read_table(self.root / "curves" / hub)
        _require(df, ("trade_date", "value"), f"curves/{hub}")
        rows = pd.DataFrame({
            "trade_day": pd.to_datetime(df["trade_date"]).dt.normalize(),
            "contract_id": self._contract_ids(df, spec).to_numpy(),
            "field": df["field"].astype(str).to_numpy() if "field" in df else "settle",
            "value": df["value"].to_numpy(dtype=float),
        })
        if "publication_time" in df:
            rows["publication_time"] = pd.to_datetime(df["publication_time"], utc=True)
        rows = rows.loc[_in_range(rows["trade_day"], start, end) & rows["field"].isin(fields)]
        return _curve_frame(spec, rows.reset_index(drop=True), source=BLOOMBERG_SOURCE,
                            synthetic_data=False, ingestion=mtime)

    def _vendor_last_trade(self, hubs: set[str]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for hub in hubs:
            try:
                df, _ = _read_table(self.root / "contracts" / hub)
            except DataError:
                continue
            _require(df, ("last_trade_date",), f"contracts/{hub}")
            ids = self._contract_ids(df, self.config.curve(hub))
            out.update(dict(zip(ids, pd.to_datetime(df["last_trade_date"]), strict=True)))
        return out

    def get_contract_metadata(self, contract_ids: Sequence[str]) -> pd.DataFrame:
        """Rule-computed expiries cross-checked against ``contracts/<hub>`` extracts (if any)."""
        hubs = {parse_contract_id(c).hub for c in contract_ids}
        cal = self.calendar or TradingCalendar.from_config(config_dir=self.config.config_dir)
        return self._expiry_metadata(contract_ids, cal, self._vendor_last_trade(hubs))

    def get_reference_data(self, instrument_ids: Sequence[str], fields: Sequence[str]
                           ) -> pd.DataFrame:
        vendor = self._vendor_last_trade({parse_contract_id(c).hub for c in instrument_ids})
        rows = [{"instrument_id": i, "field": f,
                 "value": vendor.get(i, np.nan) if f == "last_trade_date" else np.nan}
                for i in instrument_ids for f in fields]
        return pd.DataFrame(rows, columns=["instrument_id", "field", "value"])


def check_live_environment(config: BloombergConfig) -> str:
    """Raise ``BackendUnavailable`` unless live calls are authorised and xbbg/blpapi are usable.

    Returns the installed xbbg version. Does not import xbbg (importing opens nothing, but the
    check stays side-effect free).
    """
    if not config.live_calls_authorised:
        raise BackendUnavailable(
            "Live Bloomberg calls are not authorised. Set live_calls_authorised: true in the "
            "git-ignored configs/local_data_sources.yaml only on the user's explicit instruction."
        )
    for pkg in ("xbbg", "blpapi"):
        if importlib.util.find_spec(pkg) is None:
            raise BackendUnavailable(f"Package {pkg!r} is not importable in this environment")
    version = importlib.metadata.version("xbbg")
    if version not in config.supported_versions:
        raise BackendUnavailable(
            f"xbbg {version} is not supported (supported: {config.supported_versions}); "
            "xbbg 1.x returns polars frames"
        )
    return version


class LiveBloombergAdapter(_BloombergBase):
    """xbbg backend. Validates authorisation and environment at construction; validates every
    identifier before any request. ``client`` (default ``xbbg.blp``) is injectable for testing
    the refusal paths without a Terminal."""

    backend = Backend.LIVE

    def __init__(self, config: BloombergConfig | None = None, *, config_dir: str | None = None,
                 client: Any | None = None, calendar: TradingCalendar | None = None) -> None:
        super().__init__(config, config_dir=config_dir)
        self.xbbg_version = check_live_environment(self.config)
        self._client = client
        self.calendar = calendar
        # ticker -> number of contracts whose resolving ticker it was (last curve/reference call)
        self.last_ticker_counts: dict[str, int] = {}

    @property
    def client(self) -> Any:
        if self._client is None:
            from xbbg import blp  # noqa: PLC0415 - imported only on an authorised live request

            self._client = blp
        return self._client

    def _known_specs(self, series_ids: Sequence[str]) -> list[SeriesSpec]:
        specs = [self.config.spec(s) for s in series_ids]
        unknown = [f"{s.series_id} ({s.status})" for s in specs if not s.is_known]
        if unknown:
            raise IdentifierUnknown(f"Refusing live request; identifiers not known: {unknown}")
        return specs

    def get_historical_series(self, series_ids: Sequence[str], start: date, end: date
                              ) -> pd.DataFrame:
        specs = self._known_specs(series_ids)
        ingest = pd.Timestamp.now(tz="UTC")
        frames = []
        for spec in specs:
            raw = self.client.bdh(spec.ticker, spec.field, start, end)
            series = _single_column(raw, str(spec.ticker), str(spec.field))
            days = pd.DatetimeIndex(series.index).normalize()
            frames.append(_observation_frame(
                spec, _local_dates_to_utc(days, spec.lag.tz), series.to_numpy(dtype=float),
                source=BLOOMBERG_SOURCE, publication=None,
                ingestion=pd.Series([ingest] * len(days))))
        if not frames:
            return empty_frame(OBSERVATION_SCHEMA)
        return validate_frame(pd.concat(frames, ignore_index=True), OBSERVATION_SCHEMA)

    def get_curve_snapshot(self, hub: str, start: date, end: date,
                           fields: Sequence[str] = ("settle",)) -> pd.DataFrame:
        spec = self.config.curve(hub)
        if spec.status != IdentifierStatus.KNOWN.value:
            raise IdentifierUnknown(f"curve {hub} status {spec.status!r}")
        vendor_fields = {f: self.config.field_name(_CURVE_FIELD_KEYS.get(f, f)) for f in fields}
        months_ahead = int(get(load_config("instruments", self.config.config_dir), "hubs", hub,
                               "listing", "months_ahead"))
        last = pd.Timestamp(end) + pd.DateOffset(months=months_ahead)
        contracts = [c for t in spec.ids.roots for c in contracts_between(hub, t, start, last)]
        # every candidate ticker of every contract; one contract may have several (pnx form)
        cands = {c.contract_id: bloomberg_ticker_candidates(c, spec.ids) for c in contracts}
        ticker_cid = {t: cid for cid, ts in cands.items() for t in ts}
        raw = self.client.bdh(list(ticker_cid), list(vendor_fields.values()), start, end)
        hits: dict[str, list[tuple[str, pd.DataFrame]]] = {}
        for ticker, cid in ticker_cid.items():
            for logical, vf in vendor_fields.items():
                try:
                    s = _single_column(raw, ticker, vf)
                except DataError:
                    continue
                s = s.dropna()
                if s.empty:
                    continue
                frame = pd.DataFrame({"trade_day": pd.DatetimeIndex(s.index).normalize(),
                                      "contract_id": cid, "field": logical,
                                      "value": s.to_numpy(dtype=float)})
                hits.setdefault(cid, []).append((ticker, frame))
        rows, counts = [], {}
        for cid, found in hits.items():
            used = {t for t, _ in found}
            if len(used) > 1:
                # Two forms of one security (e.g. PNGZ5M and PNGZ5M25 while the contract is listed)
                # are accepted only if they agree on every common day. Disagreement is refused.
                by_ticker = {t: f.set_index("trade_day")["value"] for t, f in found}
                names = sorted(by_ticker)
                ref = by_ticker[names[0]]
                for other in names[1:]:
                    common = ref.index.intersection(by_ticker[other].index)
                    if not len(common) or not np.allclose(ref.loc[common], by_ticker[other].loc[common]):
                        raise DataError(f"{cid}: data under {names} disagree; refusing to guess")
                # keep the expired form (it names the year), the candidate listed last
                chosen = [t for t in cands[cid] if t in used][-1]
                found = [(t, f) for t, f in found if t == chosen]
                used = {chosen}
            counts[next(iter(used))] = counts.get(next(iter(used)), 0) + 1
            rows.extend(f for _, f in found)
        self.last_ticker_counts = counts
        if not rows:
            return empty_frame(CURVE_SCHEMA)
        return _curve_frame(spec, pd.concat(rows, ignore_index=True), source=BLOOMBERG_SOURCE,
                            synthetic_data=False, ingestion=pd.Timestamp.now(tz="UTC"))

    def get_reference_data(self, instrument_ids: Sequence[str], fields: Sequence[str]
                           ) -> pd.DataFrame:
        """``instrument_ids`` are internal contract ids; ``fields`` logical keys of ``fields:``."""
        vendor_fields = {f: self.config.field_name(f) for f in fields}
        cands: dict[str, list[str]] = {}
        for cid in instrument_ids:
            c = parse_contract_id(cid)
            cands[cid] = list(bloomberg_ticker_candidates(c, self.config.curve(c.hub).ids))
        raw = self.client.bdp(sorted({t for ts in cands.values() for t in ts}),
                              list(vendor_fields.values()))
        rows, counts = [], {}
        for cid, ts in cands.items():
            for logical, vf in vendor_fields.items():
                val, used = np.nan, None
                # Static reference data has no date to pick the decade from, and the listed 1-digit
                # form can name a different (later) security once the contract has expired. So the
                # expired form, which carries the full year, wins whenever it has a value.
                for ticker in reversed(ts):
                    if ticker in raw.index and vf.lower() in raw.columns and pd.notna(
                            raw.at[ticker, vf.lower()]):
                        val, used = raw.at[ticker, vf.lower()], ticker
                        break
                if used is not None and logical == next(iter(vendor_fields)):
                    counts[used] = counts.get(used, 0) + 1
                rows.append({"instrument_id": cid, "field": logical, "value": val})
        self.last_ticker_counts = counts
        return pd.DataFrame(rows, columns=["instrument_id", "field", "value"])

    def get_contract_metadata(self, contract_ids: Sequence[str]) -> pd.DataFrame:
        ref = self.get_reference_data(contract_ids, ["last_trade_date"])
        vendor = {r.instrument_id: pd.to_datetime(r.value) for r in ref.itertuples()
                  if pd.notna(r.value)}
        cal = self.calendar or expiry_calendar(self.config.config_dir)
        return self._expiry_metadata(contract_ids, cal, vendor)


def expiry_calendar(config_dir: str | None = None, hub: str = "TTF") -> TradingCalendar:
    """Business-day calendar for EXPIRY arithmetic (D-035): ``calendars.<hub>_EXPIRY`` when
    configured (weekdays minus England bank holidays for TTF), else the hub's trading calendar.
    TTF settles on some England bank holidays, so the two calendars differ."""
    cals = load_config("instruments", config_dir).get("calendars") or {}
    key = f"{hub}_EXPIRY" if f"{hub}_EXPIRY" in cals else hub
    return TradingCalendar.from_config(hub=key, config_dir=config_dir)


def _single_column(raw: pd.DataFrame, ticker: str, fld: str) -> pd.Series:
    """Extract one (ticker, field) series from an xbbg ``bdh`` frame (fields lowercased)."""
    if raw is None or not isinstance(raw, pd.DataFrame):
        raise BackendUnavailable("xbbg returned a non-pandas object; check the xbbg version")
    if raw.empty:
        raise DataError(f"empty response for {ticker} {fld}")
    key = (ticker, fld.lower())
    if isinstance(raw.columns, pd.MultiIndex):
        if key not in raw.columns:
            raise DataError(f"{ticker} {fld} missing from response")
        return raw[key]
    if fld.lower() in raw.columns:
        return raw[fld.lower()]
    raise DataError(f"{ticker} {fld} missing from response")


_FACTORY: dict[Backend, Callable[..., _BloombergBase]] = {
    Backend.MOCK: MockBloombergAdapter,
    Backend.LOCAL_FILE: LocalFileBloombergAdapter,
    Backend.LIVE: LiveBloombergAdapter,
}


def make_market_adapter(backend: str | Backend | None = None, **kwargs: Any) -> _BloombergBase:
    """Market-data adapter for ``backend`` (default: ``data_sources.yaml`` ``default_backend``).

    Raises:
        ConfigError: unknown backend name.
        BackendUnavailable: live backend not authorised / not installed.
    """
    if backend is None:
        backend = str(load_config("data_sources", kwargs.get("config_dir")).get("default_backend",
                                                                              "mock"))
    try:
        b = Backend(backend)
    except ValueError as exc:
        raise ConfigError(f"Unknown backend {backend!r}; expected {[x.value for x in Backend]}"
                          ) from exc
    return _FACTORY[b](**kwargs)


__all__ = [
    "BloombergConfig",
    "CurveSpec",
    "LiveBloombergAdapter",
    "LocalFileBloombergAdapter",
    "MockBloombergAdapter",
    "SeriesSpec",
    "check_live_environment",
    "make_market_adapter",
]
