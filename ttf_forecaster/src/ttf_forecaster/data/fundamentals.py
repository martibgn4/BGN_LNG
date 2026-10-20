"""European gas fundamentals adapter: storage, pipeline flows, LNG send-out, demand.

Implements the ``FundamentalsAdapter`` protocol of ``data/base.py`` with three backends:

* ``mock`` — a physically consistent **SYNTHETIC** EU balance (storage, flows, demand, outages),
  labelled ``SYNTHETIC`` throughout. For tests and the synthetic end-to-end run only.
* ``local_file`` — vendor extracts the user placed under ``<extracts_dir>/fundamentals/``.
* ``live`` — a skeleton: Bloomberg-sourced series are routed through
  ``ttf_forecaster.data.bloomberg.make_market_adapter("live").get_historical_series`` (imported
  lazily, so this module never needs ``blpapi``); every other source raises, because no connector
  exists (``configs/sources/fundamentals.yaml``).

Series are described in ``configs/sources/fundamentals.yaml`` and
``configs/sources/bloomberg_fundamentals.yaml``; this module never holds an identifier.

Gas-day convention: ``event_time`` is local midnight (Europe/Amsterdam) of the gas-day date,
converted to UTC. The physical gas day runs 06:00–06:00 local; only its date is used here.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
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
    SchemaError,
    apply_publication_lag,
    validate_frame,
)
from ttf_forecaster.settings import PROJECT_ROOT, ConfigError, load_config

FUNDAMENTALS_CONFIGS: tuple[str, ...] = ("sources/fundamentals", "sources/bloomberg_fundamentals")

# ---------------------------------------------------------------------------------------------
# Series registry
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesSpec:
    """One configured fundamentals series (see the shared schema in ``configs/data_sources.yaml``).

    Attributes:
        series_id: internal id, e.g. ``storage.eu.inventory_twh``.
        vendor: ``bloomberg`` | ``public`` | ``spark`` | ``other``.
        ticker: vendor identifier or ``None`` when unknown.
        field: vendor field or ``None``.
        unit: physical unit of the *normalised* value (after ``sign``).
        geography: ISO country / region label.
        status: identifier status; only ``known`` may be requested live.
        pit_grade: PIT grade of the series as sourced.
        publication_lag: lag rule, or ``None`` when the source carries its own timestamps.
        entitlement: data entitlement label.
        sign: multiply the vendor value by this to obtain the positive physical quantity.
        measure_type: ``stock`` | ``flow`` | ``capacity`` | ``ratio``.
        provenance: where the identifier comes from.
        source_config: the config file that declared it.
    """

    series_id: str
    vendor: str
    ticker: str | None
    field: str | None
    unit: str
    geography: str
    status: IdentifierStatus
    pit_grade: PitGrade
    publication_lag: PublicationLag | None
    entitlement: str
    sign: int = 1
    measure_type: str = "flow"
    provenance: str = ""
    source_config: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, series_id: str, cfg: Mapping[str, Any], source_config: str) -> SeriesSpec:
        """Build a spec from one YAML entry; raises ``ConfigError`` naming the series on problems."""
        try:
            lag_cfg = cfg.get("publication_lag")
            sign = int(cfg.get("sign", 1))
            if sign not in (1, -1):
                raise ValueError(f"sign must be +1 or -1, got {sign}")
            entitlement = str(cfg.get("entitlement", Entitlement.PUBLIC.value))
            Entitlement(entitlement)
            return cls(
                series_id=series_id,
                vendor=str(cfg["vendor"]),
                ticker=cfg.get("ticker"),
                field=cfg.get("field"),
                unit=str(cfg["unit"]),
                geography=str(cfg.get("geography", "")),
                status=IdentifierStatus(cfg["status"]),
                pit_grade=PitGrade(cfg["pit_grade"]),
                publication_lag=PublicationLag.from_config(lag_cfg) if lag_cfg else None,
                entitlement=entitlement,
                sign=sign,
                measure_type=str(cfg.get("measure_type", "flow")),
                provenance=str(cfg.get("provenance", "")),
                source_config=source_config,
                extra=dict(cfg),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ConfigError(f"{source_config}: series {series_id!r} is malformed: {exc}") from exc

    def __post_init__(self) -> None:
        if self.status is IdentifierStatus.KNOWN and not self.ticker:
            raise ConfigError(f"{self.series_id}: status 'known' requires a ticker")


def load_fundamentals_registry(config_dir: str | None = None) -> dict[str, SeriesSpec]:
    """Merge the series of every fundamentals config into one registry.

    Raises:
        ConfigError: a series id is declared twice, or an entry is malformed.
    """
    registry: dict[str, SeriesSpec] = {}
    for name in FUNDAMENTALS_CONFIGS:
        cfg = load_config(name, config_dir)
        for sid, entry in (cfg.get("series") or {}).items():
            if sid in registry:
                raise ConfigError(
                    f"series {sid!r} declared in both {registry[sid].source_config} and {name}"
                )
            registry[sid] = SeriesSpec.from_config(sid, entry, name)
    return registry


def _require_specs(registry: Mapping[str, SeriesSpec], series_ids: Iterable[str]) -> list[SeriesSpec]:
    missing = [s for s in series_ids if s not in registry]
    if missing:
        raise ConfigError(
            f"series {missing} are not declared in {list(FUNDAMENTALS_CONFIGS)}; add them there first"
        )
    return [registry[s] for s in series_ids]


# ---------------------------------------------------------------------------------------------
# Row construction and long -> daily helpers
# ---------------------------------------------------------------------------------------------


def gas_day_event_time(local_dates: pd.DatetimeIndex | pd.Series) -> pd.DatetimeIndex:
    """Naive local gas-day dates -> tz-aware UTC ``event_time`` (local midnight)."""
    idx = pd.DatetimeIndex(local_dates).normalize()
    return idx.tz_localize(MARKET_TZ, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


def observation_rows(
    spec: SeriesSpec,
    local_dates: pd.DatetimeIndex,
    values: np.ndarray | Sequence[float],
    *,
    source: str,
    quality_flag: str,
    entitlement: str | None = None,
    revision: int = 0,
    publication_time: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """OBSERVATION_SCHEMA rows for one series.

    ``publication_time`` defaults to the series' lag rule (then ``publication_time_estimated`` is
    True). A series without a lag rule must be given explicit publication times.
    """
    event = gas_day_event_time(local_dates)
    n = len(event)
    if publication_time is None:
        if spec.publication_lag is None:
            raise DataError(
                f"{spec.series_id}: no publication_lag rule and no source timestamps - cannot "
                "place the rows in time"
            )
        pub = pd.DatetimeIndex(apply_publication_lag(pd.Series(event), spec.publication_lag))
        estimated = True
    else:
        pub = pd.DatetimeIndex(publication_time)
        estimated = False
    frame = pd.DataFrame(
        {
            "series_id": [spec.series_id] * n,
            "event_time": event,
            "value": np.asarray(values, dtype=float),
            "geography": [spec.geography] * n,
            "publication_time": pub,
            "ingestion_time": pub + pd.Timedelta(minutes=5),
            "revision": np.full(n, revision, dtype="int64"),
            "unit": [spec.unit] * n,
            "source": [source] * n,
            "quality_flag": [quality_flag] * n,
            "entitlement": [entitlement or spec.entitlement] * n,
            "pit_grade": [spec.pit_grade.value] * n,
            "publication_time_estimated": np.full(n, estimated, dtype=bool),
            "source_timezone": [MARKET_TZ] * n,
        }
    )
    return frame.loc[frame["publication_time"].notna()].reset_index(drop=True)


def to_daily(resolved: pd.DataFrame, series_id: str) -> pd.Series:
    """One series of an **as-of-resolved** frame as a float Series indexed by naive local gas day.

    Missing days are absent (callers reindex; a gap stays NaN). Duplicate gas days, which a resolved
    frame cannot contain, raise.
    """
    rows = resolved.loc[resolved["series_id"] == series_id]
    if rows.empty:
        return pd.Series(dtype=float, name=series_id, index=pd.DatetimeIndex([]))
    day = rows["event_time"].dt.tz_convert(MARKET_TZ).dt.tz_localize(None).dt.normalize()
    out = pd.Series(rows["value"].astype(float).to_numpy(), index=pd.DatetimeIndex(day),
                    name=series_id).sort_index()
    if out.index.has_duplicates:
        raise DataError(f"{series_id}: duplicate gas days after resolution - frame not resolved?")
    return out


def to_daily_wide(resolved: pd.DataFrame, series_ids: Sequence[str],
                  index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Several series of a resolved frame as a wide daily frame (NaN where absent)."""
    cols = {sid: to_daily(resolved, sid) for sid in series_ids}
    frame = pd.DataFrame(cols)
    if index is not None:
        frame = frame.reindex(index)
    return frame


def grades_of(resolved: pd.DataFrame, series_ids: Iterable[str]) -> list[str]:
    """PIT grades present in ``resolved`` for ``series_ids`` (empty list if none present)."""
    sub = resolved.loc[resolved["series_id"].isin(list(series_ids)), "pit_grade"]
    return sorted(set(sub.dropna().astype(str)))


# ---------------------------------------------------------------------------------------------
# Synthetic (mock) world
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticFundamentalsConfig:
    """Parameters of the SYNTHETIC EU balance. All magnitudes are illustrative, in GWh/d / TWh."""

    start: str = "2015-10-01"
    end: str = "2027-12-31"
    seed: int = 11
    wgv_twh: float = 1100.0
    mean_level_twh: float = 600.0
    net_flow_amplitude_gwhd: float = 3000.0
    net_flow_noise_gwhd: float = 300.0
    mean_reversion: float = 0.02
    balancing_noise_gwhd: float = 80.0
    revision_prob: float = 0.2
    norway_capacity_gwhd: float = 3800.0
    norway_nominal_gwhd: float = 3400.0
    lng_capacity_gwhd: float = 6500.0
    planned_outages_per_year: int = 4
    unplanned_outages_per_year: int = 6
    planned_notice_days: int = 30
    outage_horizon_days: int = 45


def _day_of_year_phase(days: pd.DatetimeIndex, shift: float) -> np.ndarray:
    return 2 * np.pi * (days.dayofyear.to_numpy() - shift) / 365.25


def synthetic_fundamentals(
    registry: Mapping[str, SeriesSpec], cfg: SyntheticFundamentalsConfig | None = None
) -> pd.DataFrame:
    """A physically consistent SYNTHETIC EU gas balance in OBSERVATION_SCHEMA.

    Construction (GWh/d unless stated): supply components are drawn first; a seasonal net storage
    flow with AR noise and mean reversion is drawn; demand is set so that
    ``net = supply - demand - exports + balancing`` with a small ``balancing`` noise. Hence the
    balance residual is ``-balancing`` by construction, which tests can check. Norwegian flow is
    capped by outage-adjusted capacity; planned outages are published ``planned_notice_days`` ahead,
    unplanned ones on the day. Storage vintages: some days are revised 3 days after first release.

    Only series present in ``registry`` with a generator are emitted. Every row is SYNTHETIC.
    """
    cfg = cfg or SyntheticFundamentalsConfig()
    rng = np.random.default_rng(cfg.seed)
    days = pd.date_range(cfg.start, cfg.end, freq="D")
    n = len(days)

    # --- Norwegian capacity and outages (daily unavailable GWh/d, with announcement times)
    unavailable = np.zeros(n)
    announce = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    years = max(1.0, n / 365.25)
    for kind, count, notice in (
        ("planned", int(cfg.planned_outages_per_year * years), cfg.planned_notice_days),
        ("unplanned", int(cfg.unplanned_outages_per_year * years), 0),
    ):
        for _ in range(count):
            start_i = int(rng.integers(notice + 1, n - 20))
            if kind == "planned":  # maintenance concentrates in summer
                month = days[start_i].month
                if month not in (4, 5, 6, 7, 8, 9) and rng.random() < 0.7:
                    continue
            dur = int(rng.integers(3, 15))
            size = float(rng.uniform(300, 1500))
            sl = slice(start_i, min(n, start_i + dur))
            unavailable[sl] += size
            ann = days[start_i] - pd.Timedelta(days=notice) + pd.Timedelta(hours=6)
            current = announce[sl]
            announce[sl] = np.where(np.isnat(current) | (current > np.datetime64(ann)),
                                    np.datetime64(ann), current)
    unavailable = np.minimum(unavailable, cfg.norway_capacity_gwhd)
    norway_cap = np.full(n, cfg.norway_capacity_gwhd)
    norway = np.minimum(
        cfg.norway_nominal_gwhd + rng.normal(0, 120, n), norway_cap - unavailable
    ).clip(0)
    norway_cont = 0.8 * norway

    # --- other supply
    lng = (4200 + 800 * np.cos(_day_of_year_phase(days, 30)) + rng.normal(0, 350, n)).clip(500)
    lng = np.minimum(lng, cfg.lng_capacity_gwhd)
    production = 1200 + rng.normal(0, 40, n)
    n_africa = 900 + rng.normal(0, 60, n)
    tap = 300 + rng.normal(0, 15, n)
    russia = 400 + rng.normal(0, 30, n)
    gb_imp = (200 + 150 * np.cos(_day_of_year_phase(days, 15)) + rng.normal(0, 30, n)).clip(0)
    exp_gb = (300 - 200 * np.cos(_day_of_year_phase(days, 15)) + rng.normal(0, 30, n)).clip(0)
    exp_ua = 100 + rng.normal(0, 10, n)
    exp_other = 200 + rng.normal(0, 15, n)

    # --- storage: seasonal net flow (+ injection in summer) with AR noise and mean reversion
    seasonal_net = cfg.net_flow_amplitude_gwhd * np.sin(_day_of_year_phase(days, 100))
    seasonal_level = cfg.mean_level_twh - (
        cfg.net_flow_amplitude_gwhd * 365.25 / (2 * np.pi) / 1000.0
    ) * np.cos(_day_of_year_phase(days, 100))
    level = np.empty(n)
    net = np.empty(n)
    ar = 0.0
    prev = seasonal_level[0]
    for i in range(n):
        ar = 0.8 * ar + rng.normal(0, cfg.net_flow_noise_gwhd)
        f = seasonal_net[i] + ar - cfg.mean_reversion * (prev - seasonal_level[i]) * 1000.0
        f = float(np.clip(f, -prev * 1000.0, (cfg.wgv_twh - prev) * 1000.0))
        net[i] = f
        prev = prev + f / 1000.0
        level[i] = prev
    injection = np.clip(net, 0, None)
    withdrawal = np.clip(-net, 0, None)
    balancing = rng.normal(0, cfg.balancing_noise_gwhd, n)
    supply_in_perimeter = production + norway_cont + n_africa + tap + russia + gb_imp + lng
    exports = exp_gb + exp_ua + exp_other
    demand = supply_in_perimeter - exports - net + balancing

    generators: dict[str, np.ndarray] = {
        "storage.eu.inventory_twh": level,
        "storage.eu.injection_gwhd": injection,
        "storage.eu.withdrawal_gwhd": withdrawal,
        "storage.eu.working_gas_volume_twh": np.full(n, cfg.wgv_twh),
        "lng.eu.sendout_gwhd": lng,
        "lng.eu.sendout_capacity_gwhd": np.full(n, cfg.lng_capacity_gwhd),
        "flows.no.pipeline_exports_gwhd": norway,
        "flow.no.exit_total_intraday_gwhd": norway,  # the mapped feature id (flows_features.norway_flow_series)
        "flows.no.pipeline_exports_continent_gwhd": norway_cont,
        "flows.no.technical_capacity_gwhd": norway_cap,
        "production.eu.domestic_gwhd": production,
        "flows.eu.imports_north_africa_gwhd": n_africa,
        "flows.eu.imports_tap_gwhd": tap,
        "flows.eu.imports_russia_gwhd": russia,
        "flows.eu.imports_gb_gwhd": gb_imp,
        "demand.eu.total_gwhd": demand,
        "flows.eu.exports_gb_gwhd": exp_gb,
        "flows.eu.exports_ukraine_moldova_gwhd": exp_ua,
        "flows.eu.exports_other_gwhd": exp_other,
    }
    revised_series = {
        "storage.eu.inventory_twh", "storage.eu.injection_gwhd", "storage.eu.withdrawal_gwhd",
    }
    frames: list[pd.DataFrame] = []
    syn = {"source": SYNTHETIC_SOURCE, "quality_flag": QualityFlag.SYNTHETIC.value,
           "entitlement": Entitlement.SYNTHETIC.value}
    for sid, values in generators.items():
        if sid not in registry:
            continue
        spec = registry[sid]
        base = observation_rows(spec, days, values, **syn)
        frames.append(base)
        if sid in revised_series:
            rrng = np.random.default_rng([cfg.seed, zlib.crc32(sid.encode())])
            mask = rrng.random(len(base)) < cfg.revision_prob
            rev = base.loc[mask].copy()
            scale = 2.0 if spec.unit == "TWh" else 40.0
            rev["value"] = rev["value"] + rrng.normal(0, scale, int(mask.sum()))
            if spec.measure_type == "flow":
                rev["value"] = rev["value"].clip(lower=0)
            rev["revision"] = 1
            rev["publication_time"] = rev["publication_time"] + pd.Timedelta(days=3)
            rev["ingestion_time"] = rev["publication_time"] + pd.Timedelta(minutes=5)
            frames.append(rev)

    sid = "flows.no.outage_unavailable_gwhd"
    if sid in registry:
        spec = registry[sid]
        # revision 0: "no outage known" baseline, published outage_horizon_days ahead
        base_pub = gas_day_event_time(days) - pd.Timedelta(days=cfg.outage_horizon_days)
        frames.append(observation_rows(spec, days, np.zeros(n), revision=0,
                                       publication_time=base_pub, **syn))
        hit = unavailable > 0
        ann = pd.DatetimeIndex(announce[hit]).tz_localize(MARKET_TZ).tz_convert("UTC")
        frames.append(observation_rows(spec, days[hit], unavailable[hit], revision=1,
                                       publication_time=ann, **syn))
    if not frames:
        return pd.DataFrame(columns=OBSERVATION_SCHEMA.names)
    out = pd.concat(frames, ignore_index=True)
    return validate_frame(out, OBSERVATION_SCHEMA)


# ---------------------------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------------------------


def _event_window(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    day = frame["event_time"].dt.tz_convert(MARKET_TZ).dt.date
    return frame.loc[(day >= start) & (day <= end)].reset_index(drop=True)


class MockFundamentalsAdapter:
    """SYNTHETIC fundamentals. Raises for series without a synthetic generator."""

    backend = Backend.MOCK

    def __init__(self, registry: Mapping[str, SeriesSpec] | None = None,
                 cfg: SyntheticFundamentalsConfig | None = None) -> None:
        self.registry = dict(registry) if registry is not None else load_fundamentals_registry()
        self.cfg = cfg or SyntheticFundamentalsConfig()
        self._world: pd.DataFrame | None = None

    def get_series(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """OBSERVATION_SCHEMA rows (all vintages) for ``series_ids`` with event dates in [start, end]."""
        _require_specs(self.registry, series_ids)
        if self._world is None:
            self._world = synthetic_fundamentals(self.registry, self.cfg)
        available = set(self._world["series_id"].unique())
        missing = [s for s in series_ids if s not in available]
        if missing:
            raise DataError(f"mock backend has no synthetic generator for {missing}")
        sub = self._world.loc[self._world["series_id"].isin(list(series_ids))]
        return validate_frame(_event_window(sub, start, end), OBSERVATION_SCHEMA)


class LocalFileFundamentalsAdapter:
    """Reads ``<extracts_dir>/fundamentals/<series_id>.parquet|.csv``.

    Two accepted layouts:

    * **canonical** — every OBSERVATION_SCHEMA column (already normalised, signed, vintaged);
      returned after validation.
    * **vendor extract** — columns ``event_date`` (local gas day) and ``value`` (vendor raw, latest
      vintage). The configured ``sign`` and publication-lag rule are applied, ``revision = 0`` and
      ``publication_time_estimated = True``.
    """

    backend = Backend.LOCAL_FILE

    def __init__(self, extracts_dir: str | Path | None = None,
                 registry: Mapping[str, SeriesSpec] | None = None) -> None:
        self.registry = dict(registry) if registry is not None else load_fundamentals_registry()
        if extracts_dir is None:
            rel = load_config("data_sources").get("local_file", {}).get("extracts_dir")
            if not rel:
                raise ConfigError("data_sources.yaml local_file.extracts_dir is not set")
            extracts_dir = PROJECT_ROOT / rel
        self.root = Path(extracts_dir) / "fundamentals"

    def _read(self, series_id: str) -> pd.DataFrame:
        for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
            path = self.root / f"{series_id}{suffix}"
            if path.exists():
                return reader(path)
        raise DataError(
            f"no local extract for {series_id!r}: expected {self.root / series_id}.parquet or .csv"
        )

    def get_series(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        specs = _require_specs(self.registry, series_ids)
        frames = []
        for spec in specs:
            raw = self._read(spec.series_id)
            if set(OBSERVATION_SCHEMA.names) <= set(raw.columns):
                for col in ("event_time", "publication_time", "ingestion_time"):
                    raw[col] = pd.to_datetime(raw[col], utc=True)
                frame = validate_frame(raw, OBSERVATION_SCHEMA)
                if (frame["series_id"] != spec.series_id).any():
                    raise SchemaError(f"{spec.series_id}: extract contains other series ids")
            elif {"event_date", "value"} <= set(raw.columns):
                dates = pd.DatetimeIndex(pd.to_datetime(raw["event_date"]))
                values = raw["value"].astype(float).to_numpy() * spec.sign
                frame = observation_rows(
                    spec, dates, values, source=f"{spec.vendor}:local_file",
                    quality_flag=QualityFlag.OK.value,
                )
            else:
                raise SchemaError(
                    f"{spec.series_id}: extract needs the OBSERVATION_SCHEMA columns or "
                    f"[event_date, value]; got {list(raw.columns)}"
                )
            frames.append(_event_window(frame, start, end))
        return validate_frame(pd.concat(frames, ignore_index=True), OBSERVATION_SCHEMA)


class LiveFundamentalsAdapter:
    """Live skeleton. Never called in tests or without explicit user authorisation.

    * status != known -> ``IdentifierUnknown`` (raised before anything else);
    * ``live_calls_authorised`` false in ``data_sources.yaml`` -> ``BackendUnavailable``;
    * vendor ``bloomberg`` -> the market-data agent's Bloomberg adapter (lazy import), then the
      configured ``sign`` is applied here, once;
    * any other vendor -> ``BackendUnavailable`` (no connector exists).
    """

    backend = Backend.LIVE

    def __init__(self, registry: Mapping[str, SeriesSpec] | None = None,
                 authorised: bool | None = None) -> None:
        self.registry = dict(registry) if registry is not None else load_fundamentals_registry()
        if authorised is None:
            authorised = bool(load_config("data_sources").get("live_calls_authorised", False))
        self.authorised = authorised

    def get_series(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        specs = _require_specs(self.registry, series_ids)
        unknown = [s.series_id for s in specs if s.status is not IdentifierStatus.KNOWN]
        if unknown:
            raise IdentifierUnknown(
                f"series {unknown} have no confirmed vendor identifier (status != known); see "
                f"{list(FUNDAMENTALS_CONFIGS)}"
            )
        if not self.authorised:
            raise BackendUnavailable(
                "live fundamentals calls are not authorised (data_sources.yaml "
                "live_calls_authorised: false); flip only on explicit user instruction"
            )
        no_connector = [s for s in specs if s.vendor != "bloomberg"]
        if no_connector:
            detail = {s.series_id: s.extra.get("candidate_sources") for s in no_connector}
            raise BackendUnavailable(f"no live connector exists for {detail}")
        try:
            from ttf_forecaster.data.bloomberg import make_market_adapter  # lazy by design
        except ImportError as exc:  # pragma: no cover - depends on the market-data agent
            raise BackendUnavailable(f"Bloomberg adapter unavailable: {exc}") from exc
        frame = make_market_adapter("live").get_historical_series(
            [s.series_id for s in specs], start, end
        )
        frame = frame.copy()
        for spec in specs:
            if spec.sign == -1:
                mask = frame["series_id"] == spec.series_id
                frame.loc[mask, "value"] = -frame.loc[mask, "value"]
        return validate_frame(frame, OBSERVATION_SCHEMA)


def make_fundamentals_adapter(
    backend: str | Backend | None = None, **kwargs: Any
) -> MockFundamentalsAdapter | LocalFileFundamentalsAdapter | LiveFundamentalsAdapter:
    """Factory. ``backend`` defaults to ``data_sources.yaml`` ``default_backend``."""
    if backend is None:
        backend = load_config("data_sources").get("default_backend", Backend.MOCK.value)
    b = Backend(backend)
    if b is Backend.MOCK:
        return MockFundamentalsAdapter(**kwargs)
    if b is Backend.LOCAL_FILE:
        return LocalFileFundamentalsAdapter(**kwargs)
    return LiveFundamentalsAdapter(**kwargs)
