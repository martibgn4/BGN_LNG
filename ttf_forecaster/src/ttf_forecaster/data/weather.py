"""Weather adapter: vintaged forecast runs and lagged observations (``WeatherAdapter`` protocol).

Backends (``data/base.py::Backend``):

* ``mock`` — SYNTHETIC point temperatures (deterministic run, ensemble members, ERA5-like
  observations) generated here, plus the frozen ``data/synthetic.py`` ``weather.eu.hdd`` series.
* ``local_file`` — canonical long files placed under ``<extracts_dir>/weather/`` (git-ignored).
* ``live`` — a skeleton that validates configuration and refuses: no network client exists in this
  build, live calls need ``live_calls_authorised`` and the Open-Meteo licence needs the paid tier
  for desk use.

Two Open-Meteo traps are encoded as validation rather than left to documentation
(``configs/sources/weather.yaml``, probed by WeatherReturnsModel on 2026-09-07):

1. ``_previous_dayN`` with ``N > 7`` returns a column of nulls, not an error. A careless pipeline
   turns that into a *zero revision*, i.e. it asserts the forecast did not change. Leads above the
   configured maximum are refused and an all-null lead column is an error.
2. Ensemble models pad a long time axis with nulls past their real horizon. Requested depth is
   checked against the measured non-null depth.

Region weights are ``requires_user_input`` (BNEF LDZ workbook, licensed, never copied here). Equal
weights are available **only** for synthetic tests and are labelled as such.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data import synthetic
from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
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
from ttf_forecaster.settings import PROJECT_ROOT, ConfigError, get, load_config

WEATHER_CONFIG = "sources/weather"
SYNTHETIC_EQUAL_WEIGHTS = "SYNTHETIC_EQUAL_WEIGHTS"
SYNTH_DET_MODEL = "SYNTH_DET"
SYNTH_ENS_MODEL = "SYNTH_ENS"
DETERMINISTIC_MEMBER = "det"
_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d\d:?\d\d)$")


class WeightsUnavailable(DataError):
    """Region weights are not confirmed (status != known) and synthetic equal weights were not
    explicitly requested."""


class LeadUnavailable(DataError):
    """A forecast lead was requested beyond the source's real horizon, or came back all-null."""


# ---------------------------------------------------------------------------------------------
# Configuration: regions, points, weights
# ---------------------------------------------------------------------------------------------


def load_weather_config(config_dir: str | None = None) -> dict[str, Any]:
    """``configs/sources/weather.yaml`` merged with a git-ignored ``local_weather.yaml``."""
    return load_config(WEATHER_CONFIG, config_dir)


@dataclass(frozen=True)
class WeatherPoint:
    """One representative coordinate. ``weight`` is ``None`` until the user supplies it."""

    point_id: str
    name: str
    lat: float | None
    lon: float | None
    weight: float | None


@dataclass(frozen=True)
class WeatherRegion:
    """A weighted set of points, or a pre-aggregated HDD series.

    Attributes:
        region_id: config key.
        input: ``"points"`` or ``"preaggregated_hdd"``.
        points: the points (empty for pre-aggregated regions).
        weights_status: ``known`` / ``requires_user_input`` / ``not_applicable``.
        hdd_forecast_series_id / hdd_observed_series_id: pre-aggregated series ids.
    """

    region_id: str
    input: str
    points: tuple[WeatherPoint, ...]
    weights_status: str
    hdd_forecast_series_id: str | None = None
    hdd_observed_series_id: str | None = None

    def weights(self, *, allow_synthetic_equal: bool = False) -> tuple[dict[str, float], str]:
        """Normalised point weights and a provenance label.

        Raises:
            WeightsUnavailable: weights are not ``known`` (or any is null/non-positive) and
                ``allow_synthetic_equal`` is False. Equal weights are returned only when explicitly
                allowed, labelled ``SYNTHETIC_EQUAL_WEIGHTS``; they are never a production default.
        """
        if self.input != "points":
            raise ConfigError(f"region {self.region_id} is {self.input}; it has no point weights")
        raw = [p.weight for p in self.points]
        complete = all(w is not None and float(w) > 0 for w in raw)
        if self.weights_status == IdentifierStatus.KNOWN.value and complete:
            total = float(sum(float(w) for w in raw))  # type: ignore[arg-type]
            return {p.point_id: float(p.weight) / total for p in self.points}, "config"  # type: ignore[arg-type]
        if allow_synthetic_equal:
            n = len(self.points)
            return {p.point_id: 1.0 / n for p in self.points}, SYNTHETIC_EQUAL_WEIGHTS
        raise WeightsUnavailable(
            f"Region {self.region_id}: weights_status={self.weights_status!r}, "
            f"{sum(w is None for w in raw)} of {len(raw)} weights null. Supply LDZ demand shares "
            "from the BNEF LDZ workbook in a git-ignored configs/local_weather.yaml (licensed: do "
            "not commit) and set weights_status: known. Equal weights are for synthetic tests only."
        )


def point_series_id(cfg: Mapping[str, Any], point_id: str, kind: str) -> str:
    """Internal series id for a point: ``kind`` in {forecast, observed}."""
    template = get(cfg, "series_templates", kind)
    return str(template).format(point=point_id)


def region_from_config(cfg: Mapping[str, Any], region_id: str) -> WeatherRegion:
    """Build a ``WeatherRegion`` from the ``regions`` block; raises ``ConfigError`` if absent."""
    node = get(cfg, "regions", region_id)
    kind = str(node.get("input", "points"))
    if kind == "points":
        pts = tuple(
            WeatherPoint(
                point_id=str(p["id"]),
                name=str(p.get("name", p["id"])),
                lat=None if p.get("lat") is None else float(p["lat"]),
                lon=None if p.get("lon") is None else float(p["lon"]),
                weight=None if p.get("weight") is None else float(p["weight"]),
            )
            for p in node.get("points") or []
        )
        if not pts:
            raise ConfigError(f"region {region_id} has input=points but no points")
        return WeatherRegion(region_id, kind, pts, str(node.get("weights_status")))
    if kind == "preaggregated_hdd":
        return WeatherRegion(
            region_id,
            kind,
            (),
            str(node.get("weights_status", "not_applicable")),
            hdd_forecast_series_id=node.get("hdd_forecast_series_id"),
            hdd_observed_series_id=node.get("hdd_observed_series_id"),
        )
    raise ConfigError(f"region {region_id}: unknown input {kind!r}")


# ---------------------------------------------------------------------------------------------
# Validation of the two silent Open-Meteo failure modes
# ---------------------------------------------------------------------------------------------


def validate_lead_coverage(forecasts: pd.DataFrame, max_lead_days: float) -> pd.DataFrame:
    """Refuse leads beyond ``max_lead_days`` and any lead whose values are entirely null.

    An all-null lead is how Open-Meteo answers an unavailable ``_previous_dayN``; keeping the rows
    lets a downstream ``fillna(0)`` or a difference against NaN masquerade as "no revision".

    Raises:
        LeadUnavailable: listing every offending (series_id, model, member, lead_days).
    """
    if forecasts.empty:
        return forecasts
    problems: list[str] = []
    too_far = forecasts["lead_hours"] > max_lead_days * 24.0
    if too_far.any():
        problems.append(
            f"{int(too_far.sum())} rows with lead > {max_lead_days} days (max "
            f"{forecasts['lead_hours'].max() / 24:.1f} d): beyond the archive's real horizon"
        )
    lead_day = np.ceil(forecasts["lead_hours"] / 24.0).astype(int)
    keys = [forecasts["series_id"], forecasts["model"], forecasts["member"], lead_day]
    nonnull = forecasts["value"].notna().groupby(keys).sum()
    dead = nonnull[nonnull == 0]
    if len(dead):
        listed = ", ".join(f"{s}/{m}/{mem}/lead{ld}d" for s, m, mem, ld in dead.index[:10])
        problems.append(f"all-null lead columns ({len(dead)}): {listed}")
    if problems:
        raise LeadUnavailable(
            "Forecast lead coverage invalid: " + "; ".join(problems) + ". A null lead must stay "
            "NaN and be flagged; it must never become a zero revision."
        )
    return forecasts


def check_ensemble_depth(
    forecasts: pd.DataFrame,
    requested_days: int,
    real_max_lead_days: int,
    min_usable_fraction: float,
) -> pd.DataFrame:
    """Refuse an ensemble pull whose non-null depth is short of what was requested.

    Ensemble endpoints pad the time axis with nulls past the model's real horizon; reading the axis
    length overstates the horizon (gfs025: 35-day axis, 10 non-null days).

    Raises:
        LeadUnavailable: requested beyond the measured horizon, or any run's non-null valid days
            fall below ``min_usable_fraction`` of the request.
    """
    if requested_days > real_max_lead_days:
        raise LeadUnavailable(
            f"Requested {requested_days} ensemble days but the model's measured non-null horizon is "
            f"{real_max_lead_days} days; the remainder would be null padding."
        )
    if forecasts.empty:
        return forecasts
    nn = forecasts.loc[forecasts["value"].notna()]
    depth = nn.groupby(["series_id", "model", "issue_time"])["valid_time"].nunique()
    short = depth[depth < min_usable_fraction * requested_days]
    runs = forecasts.groupby(["series_id", "model", "issue_time"]).ngroups
    if len(short) or len(depth) < runs:
        raise LeadUnavailable(
            f"{len(short) + runs - len(depth)} ensemble runs have fewer than "
            f"{min_usable_fraction:.0%} of {requested_days} requested days non-null (null padding)."
        )
    return forecasts


# ---------------------------------------------------------------------------------------------
# Conversion of Open-Meteo hourly payloads (as returned, already parsed) to canonical rows
# ---------------------------------------------------------------------------------------------


def _require_utc(ts: pd.Series, name: str) -> pd.Series:
    if not isinstance(ts.dtype, pd.DatetimeTZDtype):
        raise SchemaError(f"{name}: timestamps must be tz-aware (request Open-Meteo with UTC)")
    return ts.dt.tz_convert("UTC")


def hourly_to_daily(hourly: pd.DataFrame, value_col: str, min_hours: int) -> pd.DataFrame:
    """UTC-day mean of an hourly column. Days with fewer than ``min_hours`` non-null hours are NaN.

    Args:
        hourly: frame with tz-aware ``time`` and ``value_col``.
        value_col: column to aggregate.
        min_hours: completeness threshold (24 = every hour required).

    Returns:
        Frame ``valid_time`` (UTC midnight), ``value``, ``n_hours``.
    """
    t = _require_utc(hourly["time"], "time")
    vals = pd.to_numeric(hourly[value_col], errors="coerce")
    day = t.dt.floor("D")
    g = pd.DataFrame({"valid_time": day, "v": vals}).groupby("valid_time")["v"]
    out = pd.DataFrame({"value": g.mean(), "n_hours": g.count()}).reset_index()
    out.loc[out["n_hours"] < min_hours, "value"] = np.nan
    return out


def _provenance_cols(
    n: int,
    unit: str,
    publication: pd.Series,
    ingestion: pd.Timestamp | pd.Series,
    source: str,
    grade: PitGrade,
    estimated: bool,
    entitlement: str,
    quality: pd.Series | None = None,
) -> dict[str, object]:
    pub = pd.Series(publication).reset_index(drop=True)
    ing = (
        pd.Series([ingestion] * n, dtype="datetime64[ns, UTC]")
        if isinstance(ingestion, pd.Timestamp)
        else pd.Series(ingestion).reset_index(drop=True)
    )
    return {
        "publication_time": pub,
        "ingestion_time": ing,
        "revision": np.zeros(n, dtype="int64"),
        "unit": [unit] * n,
        "source": [source] * n,
        "quality_flag": (
            list(quality) if quality is not None else [QualityFlag.OK.value] * n
        ),
        "entitlement": [entitlement] * n,
        "pit_grade": [grade.value] * n,
        "publication_time_estimated": np.full(n, estimated),
        "source_timezone": ["UTC"] * n,
    }


def previous_runs_to_forecasts(
    hourly: pd.DataFrame,
    *,
    series_id: str,
    geography: str,
    model: str,
    ingestion_time: pd.Timestamp,
    cfg: Mapping[str, Any] | None = None,
    leads: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Turn a parsed historical-forecast payload (hourly ``temperature_2m`` and its
    ``_previous_dayN`` columns, tz-aware ``time``) into daily FORECAST_SCHEMA rows.

    Mapping (``open_meteo.previous_runs`` in config): the column for ``N`` at valid day ``V`` gets
    ``issue_time = V - N days + nominal_issue_hour``, ``lead_hours = 24 N - nominal_issue_hour``.
    ``publication_time`` follows ``publication_rule``: ``end_of_issue_day`` (default, conservative:
    end of day V-N plus the availability lag) or ``issue_plus_lag``. It is always flagged
    ``publication_time_estimated=True``.

    Raises:
        LeadUnavailable: a requested lead exceeds ``max_lead_days`` or its column is all-null.
        SchemaError: missing columns or naive timestamps.
    """
    cfg = cfg if cfg is not None else load_weather_config()
    pr = get(cfg, "open_meteo", "previous_runs")
    var = str(pr["hourly_variable"])
    suffix = str(pr["suffix_template"])
    max_lead = int(pr["max_lead_days"])
    leads = list(leads) if leads is not None else list(range(0, max_lead + 1))
    bad = [n for n in leads if n < 0 or n > max_lead]
    if bad:
        raise LeadUnavailable(
            f"previous-run leads {bad} outside 0..{max_lead}: Open-Meteo returns all-null columns "
            "beyond its archive horizon, which a careless pipeline turns into zero revisions."
        )
    min_hours = int(get(cfg, "open_meteo", "daily_aggregation", "min_hours_per_day"))
    issue_hour = int(pr["nominal_issue_hour_utc"])
    lag_h = float(pr["availability_lag_hours"])
    rule = str(pr["publication_rule"])
    if rule not in ("end_of_issue_day", "issue_plus_lag"):
        raise ConfigError(f"open_meteo.previous_runs.publication_rule {rule!r} unknown")
    if "time" not in hourly.columns:
        raise SchemaError("previous-runs payload needs a 'time' column")
    frames: list[pd.DataFrame] = []
    for n in leads:
        col = var if n == 0 else var + suffix.format(n=n)
        if col not in hourly.columns:
            raise SchemaError(f"previous-runs payload lacks column {col!r}")
        if pd.to_numeric(hourly[col], errors="coerce").notna().sum() == 0:
            raise LeadUnavailable(
                f"{series_id}: column {col!r} is entirely null - the archive does not cover this "
                "lead/window. Shorten the range or drop the lead; never fill with zero."
            )
        daily = hourly_to_daily(hourly, col, min_hours)
        issue_day = daily["valid_time"] - pd.Timedelta(days=n)
        issue = issue_day + pd.Timedelta(hours=issue_hour)
        if rule == "end_of_issue_day":
            pub = issue_day + pd.Timedelta(days=1, hours=lag_h)
        else:
            pub = issue + pd.Timedelta(hours=lag_h)
        m = len(daily)
        quality = pd.Series(
            np.where(daily["value"].isna(), QualityFlag.MISSING_INPUT.value, QualityFlag.OK.value)
        )
        frames.append(
            pd.DataFrame(
                {
                    "series_id": [series_id] * m,
                    "issue_time": issue.reset_index(drop=True),
                    "valid_time": daily["valid_time"].reset_index(drop=True),
                    "lead_hours": np.full(m, 24.0 * n - issue_hour),
                    "model": [model] * m,
                    "member": [DETERMINISTIC_MEMBER] * m,
                    "value": daily["value"].astype(float).reset_index(drop=True),
                    "geography": [geography] * m,
                    **_provenance_cols(
                        m, "degC", pub.reset_index(drop=True), ingestion_time,
                        "open_meteo_historical_forecast", PitGrade.A, True,
                        Entitlement.PUBLIC.value, quality,
                    ),
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    validate_frame(out, FORECAST_SCHEMA)
    return validate_lead_coverage(out, max_lead)


def observed_hourly_to_observations(
    hourly: pd.DataFrame,
    *,
    series_id: str,
    geography: str,
    ingestion_time: pd.Timestamp,
    cfg: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """ERA5 hourly ``temperature_2m`` -> daily OBSERVATION_SCHEMA rows, grade B, publication from
    the ``era5_observed`` lag rule (estimated). Never a forecast substitute."""
    cfg = cfg if cfg is not None else load_weather_config()
    var = str(get(cfg, "open_meteo", "previous_runs", "hourly_variable"))
    min_hours = int(get(cfg, "open_meteo", "daily_aggregation", "min_hours_per_day"))
    lag = PublicationLag.from_config(get(cfg, "lag_rules", "era5_observed"))
    daily = hourly_to_daily(hourly, var, min_hours)
    m = len(daily)
    pub = apply_publication_lag(daily["valid_time"], lag)
    quality = pd.Series(
        np.where(daily["value"].isna(), QualityFlag.MISSING_INPUT.value, QualityFlag.OK.value)
    )
    out = pd.DataFrame(
        {
            "series_id": [series_id] * m,
            "event_time": daily["valid_time"],
            "value": daily["value"].astype(float),
            "geography": [geography] * m,
            **_provenance_cols(m, "degC", pub, ingestion_time, "open_meteo_era5_archive",
                               PitGrade.B, True, Entitlement.PUBLIC.value, quality),
        }
    )
    return validate_frame(out, OBSERVATION_SCHEMA)


# ---------------------------------------------------------------------------------------------
# Synthetic point weather (SYNTHETIC; never evidence of anything)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticWeatherConfig:
    """Synthetic point temperatures. Observations start earlier than forecasts so normals exist."""

    obs_start: str = "2012-10-01"
    start: str = "2022-01-01"
    end: str = "2023-12-31"
    seed: int = 11
    max_lead_days: int = 7
    n_members: int = 0
    ens_max_lead_days: int = 10
    fc_error_sd_per_lead: float = 0.6  # degC per day of lead
    member_sd: float = 1.0  # degC
    publication_lag_hours: float = 6.0
    obs_lag_days: int = 5


def synthetic_point_weather(
    point_ids: Sequence[str], cfg: SyntheticWeatherConfig = SyntheticWeatherConfig()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SYNTHETIC daily mean temperatures per point: (forecasts, observations).

    Truth = seasonal cycle + AR(1) anomaly. Daily 00z runs, leads 1..max_lead_days, error grows
    with lead (each run carries its own persistent error, so consecutive runs revise). Ensemble
    members (model ``SYNTH_ENS``) scatter around each run. Observations are the truth published
    ``obs_lag_days`` later (grade B, estimated). Series ids: ``weather.<point>.t2m_mean_degc`` and
    ``weather.<point>.t2m_mean_obs_degc``.
    """
    rng = np.random.default_rng(cfg.seed)
    obs_days = pd.date_range(cfg.obs_start, cfg.end, freq="D", tz="UTC")
    fc_issue = pd.date_range(cfg.start, cfg.end, freq="D", tz="UTC")
    fc_frames: list[pd.DataFrame] = []
    obs_frames: list[pd.DataFrame] = []
    all_days = pd.date_range(cfg.obs_start, pd.Timestamp(cfg.end) + pd.Timedelta(days=40),
                             freq="D", tz="UTC")
    doy = all_days.dayofyear.to_numpy()
    for k, pid in enumerate(point_ids):
        offset = 2.0 * (k % 3) - 2.0
        seasonal = 10.0 + offset + 9.0 * np.cos(2 * np.pi * (doy - 200) / 365.25)
        anom = np.zeros(len(all_days))
        shocks = rng.normal(0, 1.6, len(all_days))
        for i in range(1, len(all_days)):
            anom[i] = 0.8 * anom[i - 1] + shocks[i]
        truth = pd.Series(seasonal + anom, index=all_days)
        # Observations
        o_truth = truth.loc[obs_days]
        n = len(o_truth)
        pub = o_truth.index.to_series().reset_index(drop=True) + pd.Timedelta(
            days=cfg.obs_lag_days, hours=7
        )
        obs_frames.append(
            pd.DataFrame(
                {
                    "series_id": f"weather.{pid}.t2m_mean_obs_degc",
                    "event_time": o_truth.index,
                    "value": o_truth.to_numpy(dtype=float),
                    "geography": pid.upper(),
                    **_synthetic_prov(n, pub, PitGrade.B, estimated=True),
                }
            )
        )
        # Forecast runs (vectorised over issue x lead): each run's error is a random walk along
        # lead, so the error grows with lead and consecutive runs genuinely revise.
        pos0 = all_days.get_indexer(fc_issue)
        tv = truth.to_numpy()

        def runs(n_leads: int) -> tuple[np.ndarray, np.ndarray]:
            leads = np.arange(1, n_leads + 1)
            err = np.cumsum(rng.normal(0, cfg.fc_error_sd_per_lead, (len(fc_issue), n_leads)),
                            axis=1)
            return leads, tv[pos0[:, None] + leads[None, :]] + err

        leads, det = runs(cfg.max_lead_days)
        fc_frames.append(_synthetic_fc_rows(pid, fc_issue, leads, SYNTH_DET_MODEL,
                                            DETERMINISTIC_MEMBER, det, cfg.publication_lag_hours))
        if cfg.n_members:
            e_leads, base = runs(cfg.ens_max_lead_days)
            for mbr in range(cfg.n_members):
                vals = base + rng.normal(0, 1.0, base.shape) * (
                    cfg.member_sd * np.sqrt(e_leads / 3.0)
                )[None, :]
                fc_frames.append(_synthetic_fc_rows(pid, fc_issue, e_leads, SYNTH_ENS_MODEL,
                                                    f"m{mbr:02d}", vals,
                                                    cfg.publication_lag_hours))
    fc = pd.concat(fc_frames, ignore_index=True)
    obs = pd.concat(obs_frames, ignore_index=True)
    return validate_frame(fc, FORECAST_SCHEMA), validate_frame(obs, OBSERVATION_SCHEMA)


def _synthetic_prov(n: int, pub: pd.Series | pd.DatetimeIndex, grade: PitGrade,
                    estimated: bool) -> dict[str, object]:
    pub_s = pd.Series(pd.DatetimeIndex(pub))
    return {
        "publication_time": pub_s,
        "ingestion_time": pub_s + pd.Timedelta(minutes=5),
        "revision": np.zeros(n, dtype="int64"),
        "unit": ["degC"] * n,
        "source": [SYNTHETIC_SOURCE] * n,
        "quality_flag": [QualityFlag.SYNTHETIC.value] * n,
        "entitlement": [Entitlement.SYNTHETIC.value] * n,
        "pit_grade": [grade.value] * n,
        "publication_time_estimated": np.full(n, estimated),
        "source_timezone": ["UTC"] * n,
    }


def _synthetic_fc_rows(pid: str, issues: pd.DatetimeIndex, leads: np.ndarray, model: str,
                       member: str, values: np.ndarray, pub_lag_hours: float) -> pd.DataFrame:
    """Rows for every (issue, lead) pair; ``values`` has shape (len(issues), len(leads))."""
    issue_idx = issues.repeat(len(leads))
    lead_h = np.tile(leads.astype(float) * 24.0, len(issues))
    n = len(issue_idx)
    return pd.DataFrame(
        {
            "series_id": f"weather.{pid}.t2m_mean_degc",
            "issue_time": issue_idx,
            "valid_time": issue_idx + pd.to_timedelta(lead_h, unit="h"),
            "lead_hours": lead_h,
            "model": model,
            "member": member,
            "value": values.reshape(-1).astype(float),
            "geography": pid.upper(),
            **_synthetic_prov(n, issue_idx + pd.Timedelta(hours=pub_lag_hours), PitGrade.A,
                              estimated=False),
        }
    )


# ---------------------------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------------------------


def _date_mask(ts: pd.Series, start: date, end: date) -> pd.Series:
    d = ts.dt.date
    return (d >= start) & (d <= end)


class MockWeatherAdapter:
    """SYNTHETIC backend. Serves ``weather.<point>.*`` from ``synthetic_point_weather`` and
    ``weather.eu.hdd`` from ``data/synthetic.weather_forecast_runs``."""

    backend = Backend.MOCK

    def __init__(
        self,
        point_ids: Sequence[str] = ("aa", "bb", "cc"),
        cfg: SyntheticWeatherConfig = SyntheticWeatherConfig(),
        hdd_cfg: synthetic.SyntheticConfig | None = None,
    ) -> None:
        self.point_ids = tuple(point_ids)
        self.cfg = cfg
        self.hdd_cfg = hdd_cfg or synthetic.SyntheticConfig(start=cfg.start, end=cfg.end)
        self._fc: pd.DataFrame | None = None
        self._obs: pd.DataFrame | None = None

    def _build(self) -> None:
        if self._fc is None:
            fc, obs = synthetic_point_weather(self.point_ids, self.cfg)
            hdd = synthetic.weather_forecast_runs(self.hdd_cfg)
            self._fc = pd.concat([fc, hdd], ignore_index=True)
            self._obs = obs

    def get_forecast_runs(
        self, series_ids: Sequence[str], issue_start: date, issue_end: date
    ) -> pd.DataFrame:
        """FORECAST_SCHEMA rows (SYNTHETIC) for runs issued in [issue_start, issue_end]."""
        self._build()
        assert self._fc is not None
        unknown = sorted(set(series_ids) - set(self._fc["series_id"]))
        if unknown:
            raise IdentifierUnknown(f"mock weather backend has no series {unknown}")
        mask = self._fc["series_id"].isin(series_ids) & _date_mask(
            self._fc["issue_time"], issue_start, issue_end
        )
        return validate_frame(self._fc.loc[mask].reset_index(drop=True), FORECAST_SCHEMA)

    def get_observed(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """OBSERVATION_SCHEMA rows (SYNTHETIC) with event dates in [start, end]."""
        self._build()
        assert self._obs is not None
        unknown = sorted(set(series_ids) - set(self._obs["series_id"]))
        if unknown:
            raise IdentifierUnknown(f"mock weather backend has no observed series {unknown}")
        mask = self._obs["series_id"].isin(series_ids) & _date_mask(
            self._obs["event_time"], start, end
        )
        return validate_frame(self._obs.loc[mask].reset_index(drop=True), OBSERVATION_SCHEMA)


def _parse_utc(col: pd.Series, name: str) -> pd.Series:
    """Parse a timestamp column that must carry an explicit offset; naive input is an error."""
    if isinstance(col.dtype, pd.DatetimeTZDtype):
        return col.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_dtype(col):
        raise SchemaError(f"{name}: naive timestamps in extract; store UTC with an offset")
    s = col.astype("string")
    bad = s.notna() & ~s.str.contains(_TZ_SUFFIX, regex=True)
    if bad.any():
        raise SchemaError(f"{name}: {int(bad.sum())} timestamps lack a UTC offset (e.g. 'Z')")
    return pd.to_datetime(s, utc=True, format="ISO8601")


def _read_extract(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise SchemaError(f"unsupported extract format {path.suffix} ({path.name})")


class LocalFileWeatherAdapter:
    """Reads canonical long files from ``<extracts_dir>/<subdir>/`` (git-ignored).

    Missing ``publication_time`` is derived (forecasts: ``issue_time + available_after_hours`` of
    the model; observations: the ``era5_observed`` lag rule) and flagged estimated. Lead coverage is
    validated per model against its configured maximum.
    """

    backend = Backend.LOCAL_FILE

    def __init__(self, extracts_dir: str | Path | None = None,
                 config_dir: str | None = None) -> None:
        self.cfg = load_weather_config(config_dir)
        if extracts_dir is None:
            ds = load_config("data_sources", config_dir)
            extracts_dir = PROJECT_ROOT / str(get(ds, "local_file", "extracts_dir"))
        self.root = Path(extracts_dir) / str(get(self.cfg, "local_file", "subdir"))

    def _files(self, key: str) -> list[Path]:
        pattern = str(get(self.cfg, "local_file", key))
        files = sorted(p for p in self.root.glob(pattern) if p.suffix in (".parquet", ".csv"))
        if not files:
            raise DataError(f"no weather extracts matching {pattern!r} under {self.root}")
        return files

    def _model_max_lead(self, model: str) -> float | None:
        om = get(self.cfg, "open_meteo")
        for block in ("models", "ensemble_models"):
            if model in (om.get(block) or {}):
                return float(om[block][model]["max_lead_days"])
        return None

    def get_forecast_runs(
        self, series_ids: Sequence[str], issue_start: date, issue_end: date
    ) -> pd.DataFrame:
        """FORECAST_SCHEMA rows from local extracts for runs issued in [issue_start, issue_end]."""
        df = pd.concat([_read_extract(p) for p in self._files("forecast_glob")], ignore_index=True)
        for c in ("issue_time", "valid_time", "publication_time", "ingestion_time"):
            if c in df.columns:
                df[c] = _parse_utc(df[c], c)
        if "publication_time" not in df.columns:
            lags = df["model"].map(
                lambda m: get(self.cfg, "open_meteo", "models", m, "available_after_hours")
            )
            df["publication_time"] = df["issue_time"] + pd.to_timedelta(lags.astype(float),
                                                                        unit="h")
            df["publication_time_estimated"] = True
        df = df.loc[df["series_id"].isin(series_ids)
                    & _date_mask(df["issue_time"], issue_start, issue_end)].reset_index(drop=True)
        if df.empty:
            raise DataError(f"no forecast rows for {list(series_ids)} issued "
                            f"{issue_start}..{issue_end} under {self.root}")
        validate_frame(df, FORECAST_SCHEMA)
        for model, part in df.groupby("model"):
            max_lead = self._model_max_lead(str(model))
            validate_lead_coverage(part, max_lead if max_lead is not None else np.inf)
        return df

    def get_observed(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        """OBSERVATION_SCHEMA rows from local extracts with event dates in [start, end]."""
        df = pd.concat([_read_extract(p) for p in self._files("observed_glob")], ignore_index=True)
        for c in ("event_time", "publication_time", "ingestion_time"):
            if c in df.columns:
                df[c] = _parse_utc(df[c], c)
        if "publication_time" not in df.columns:
            lag = PublicationLag.from_config(get(self.cfg, "lag_rules", "era5_observed"))
            df["publication_time"] = apply_publication_lag(df["event_time"], lag)
            df["publication_time_estimated"] = True
            df["pit_grade"] = PitGrade.B.value
        df = df.loc[df["series_id"].isin(series_ids)
                    & _date_mask(df["event_time"], start, end)].reset_index(drop=True)
        return validate_frame(df, OBSERVATION_SCHEMA)


class LiveWeatherAdapter:
    """Live skeleton. Validates every precondition and then refuses: no network client exists in
    this build. Preconditions, in order: live calls authorised by the user; Open-Meteo licence
    confirmed for commercial (desk) use; every requested series ``status: known``."""

    backend = Backend.LIVE

    def __init__(self, *, live_calls_authorised: bool | None = None,
                 config_dir: str | None = None) -> None:
        self.cfg = load_weather_config(config_dir)
        if live_calls_authorised is None:
            ds = load_config("data_sources", config_dir)
            live_calls_authorised = bool(ds.get("live_calls_authorised", False))
        if not live_calls_authorised:
            raise BackendUnavailable(
                "Live weather calls are not authorised (data_sources.live_calls_authorised=false). "
                "Only the user can authorise live calls, in a git-ignored local config."
            )
        if not bool(get(self.cfg, "licence", "commercial_use_confirmed")):
            raise BackendUnavailable(
                "Open-Meteo free tier is CC-BY-4.0 NON-COMMERCIAL; desk use needs the paid tier. "
                "Set licence.commercial_use_confirmed only after the user confirms the licence."
            )

    def _check_known(self, series_ids: Sequence[str]) -> None:
        series = get(self.cfg, "series")
        for sid in series_ids:
            key = re.sub(r"^weather\.[a-z0-9_]+\.", "weather.point.", sid)
            entry = series.get(sid) or series.get(key)
            if entry is None or entry.get("status") != IdentifierStatus.KNOWN.value:
                raise IdentifierUnknown(f"weather series {sid} is not status=known in config")

    def get_forecast_runs(
        self, series_ids: Sequence[str], issue_start: date, issue_end: date
    ) -> pd.DataFrame:
        self._check_known(series_ids)
        raise BackendUnavailable("No live Open-Meteo client in this build (no network allowed).")

    def get_observed(self, series_ids: Sequence[str], start: date, end: date) -> pd.DataFrame:
        self._check_known(series_ids)
        raise BackendUnavailable("No live Open-Meteo client in this build (no network allowed).")


def make_weather_adapter(
    backend: str | Backend | None = None, *, config_dir: str | None = None, **kwargs: Any
) -> MockWeatherAdapter | LocalFileWeatherAdapter | LiveWeatherAdapter:
    """Factory. ``backend`` defaults to ``data_sources.default_backend``."""
    if backend is None:
        backend = load_config("data_sources", config_dir).get("default_backend", "mock")
    b = Backend(str(backend))
    if b is Backend.MOCK:
        return MockWeatherAdapter(**kwargs)
    if b is Backend.LOCAL_FILE:
        return LocalFileWeatherAdapter(config_dir=config_dir, **kwargs)
    return LiveWeatherAdapter(config_dir=config_dir, **kwargs)


__all__ = [
    "DETERMINISTIC_MEMBER",
    "LeadUnavailable",
    "LiveWeatherAdapter",
    "LocalFileWeatherAdapter",
    "MockWeatherAdapter",
    "SYNTHETIC_EQUAL_WEIGHTS",
    "SYNTH_DET_MODEL",
    "SYNTH_ENS_MODEL",
    "SyntheticWeatherConfig",
    "WeatherPoint",
    "WeatherRegion",
    "WeightsUnavailable",
    "check_ensemble_depth",
    "hourly_to_daily",
    "load_weather_config",
    "make_weather_adapter",
    "observed_hourly_to_observations",
    "point_series_id",
    "previous_runs_to_forecasts",
    "region_from_config",
    "synthetic_point_weather",
    "validate_lead_coverage",
]
