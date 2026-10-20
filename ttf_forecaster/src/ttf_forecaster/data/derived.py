"""Derived power series: sums and rescalings of raw Bloomberg components, point-in-time (D-034).

``configs/sources/power.yaml`` ``derived:`` declares, for each internal GW / GWh series id, the raw
components it is built from (MW / MWh series in ``configs/sources/bloomberg_power.yaml``), the
``sum`` of those components and a ``scale``. This module turns the declarations into rows the
feature builders can read:

* ``event_time``: the components' shared event time. If a component has no row for an event time,
  the derived value is NaN for that event time, never a partial sum.
* ``publication_time``: the latest publication among the component rows that define the derived
  vintage, so a derived value is never visible before its last component is published, revisions
  included. A new derived row is written only when the component state changes.
* ``pit_grade``: the worst grade among the component rows of that vintage. Grade C is kept here so
  the grade propagates; the readers (``resolve_as_of`` defaults) exclude it.
* ``unit``: from the derived id (``_gw`` -> GW, ``_gwh`` -> GWh).

Series whose ``schema`` is ``forecast`` in power.yaml (the day-ahead wind and solar forecasts) are
also exposed as FORECAST_SCHEMA rows, because ``features/power.py`` reads those ids from
``bundle.forecasts``. Their ``valid_time`` is the market-local date stamped at UTC midnight (the
daily convention of ``features/weather.py``) and their ``issue_time`` is the publication time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    FORECAST_SCHEMA,
    MARKET_TZ,
    PIT_GRADE_ORDER,
    OBSERVATION_SCHEMA,
    SYNTHETIC_SOURCE,
    DataBundle,
    DataError,
    PitGrade,
    QualityFlag,
    empty_frame,
    validate_frame,
)
from ttf_forecaster.features.point_in_time import raise_on_conflicting_ties
from ttf_forecaster.settings import ConfigError

DERIVED_SOURCE = "derived"
DERIVED_MODEL = "derived"
DERIVED_MEMBER = "sum"

# derived unit suffix -> (derived unit, required component unit)
_UNITS = {"gw": ("GW", "MW"), "gwh": ("GWh", "MWh")}
_MW_TO_GW = 0.001  # the only scale that converts MW -> GW (and MWh -> GWh)
_ALLOWED_KEYS = frozenset({"sum", "terms", "constant", "constant_status", "constant_provenance", "scale", "note",
                            "components_kind"})
_KINDS = ("observation", "forecast")


@dataclass(frozen=True)
class DerivedSpec:
    """One validated ``derived:`` entry of power.yaml."""

    series_id: str
    components: tuple[str, ...]
    scale: float
    unit: str
    component_unit: str
    forecast_view: bool  # also expose as FORECAST_SCHEMA rows (power.yaml schema: forecast)
    coefs: tuple[float, ...] = ()  # one per component; () means every coefficient is 1 (a plain sum)
    constant: float = 0.0  # added before scaling, in the component unit (e.g. installed capacity in MW)
    components_kind: str = "observation"  # "forecast": components are FORECAST rows keyed on (issue, valid)

    def coefficient(self, series_id: str) -> float:
        return dict(zip(self.components, self.coefs or (1.0,) * len(self.components)))[series_id]


def unit_for_id(series_id: str) -> tuple[str, str]:
    """(derived unit, component unit) from the id suffix. Unknown suffixes are a config error."""
    suffix = series_id.rsplit("_", 1)[-1].lower()
    if suffix not in _UNITS:
        raise ConfigError(f"derived id {series_id!r}: unit suffix must be _gw or _gwh")
    return _UNITS[suffix]


def _components(sid: str, node: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """(component ids, coefficients) from ``sum`` (all 1) or ``terms: [{series, coef}]``."""
    if "sum" in node and "terms" in node:
        raise ConfigError(f"derived {sid!r}: give 'sum' or 'terms', not both")
    if "terms" in node:
        comps: list[str] = []
        coefs: list[float] = []
        for term in node.get("terms") or []:
            t = dict(term or {})
            if set(t) != {"series", "coef"}:
                raise ConfigError(f"derived {sid!r}: each term needs exactly 'series' and 'coef', got {sorted(t)}")
            coef = float(t["coef"])
            if not np.isfinite(coef) or coef == 0.0:
                raise ConfigError(f"derived {sid!r}: coef for {t['series']!r} must be a non-zero number")
            comps.append(str(t["series"]))
            coefs.append(coef)
    else:
        comps = [str(c) for c in (node.get("sum") or [])]
        coefs = [1.0] * len(comps)
    if not comps:
        raise ConfigError(f"derived {sid!r}: 'sum' or 'terms' must list at least one component")
    if len(set(comps)) != len(comps):
        raise ConfigError(f"derived {sid!r}: duplicate components {comps}")
    return tuple(comps), tuple(coefs)


def _constant(sid: str, node: Mapping[str, Any]) -> float:
    """The constant term. A non-zero constant must say where it comes from and how sure we are."""
    constant = float(node.get("constant", 0.0))
    if not np.isfinite(constant):
        raise ConfigError(f"derived {sid!r}: constant must be finite")
    if constant != 0.0:
        if str(node.get("constant_status")) not in ("known", "assumed"):
            raise ConfigError(f"derived {sid!r}: a non-zero constant needs constant_status known or assumed")
        if not str(node.get("constant_provenance") or "").strip():
            raise ConfigError(f"derived {sid!r}: a non-zero constant needs constant_provenance")
    return constant


def load_derived_specs(cfg: Mapping[str, Any]) -> dict[str, DerivedSpec]:
    """Validate the ``derived:`` block of a loaded power.yaml. Empty dict when there is none."""
    derived = cfg.get("derived") or {}
    series = cfg.get("series") or {}
    specs: dict[str, DerivedSpec] = {}
    for sid, raw in derived.items():
        node = dict(raw or {})
        extra = set(node) - _ALLOWED_KEYS
        if extra:
            raise ConfigError(f"derived {sid!r}: unknown keys {sorted(extra)}")
        comps, coefs = _components(sid, node)
        unit, comp_unit = unit_for_id(sid)
        scale = float(node.get("scale", 1.0))
        if not np.isclose(scale, _MW_TO_GW):
            raise ConfigError(f"derived {sid!r}: scale {scale} must be {_MW_TO_GW} "
                              f"({comp_unit} -> {unit})")
        constant = _constant(sid, node)
        kind = str(node.get("components_kind", "observation"))
        if kind not in _KINDS:
            raise ConfigError(f"derived {sid!r}: components_kind must be one of {_KINDS}")
        entry = series.get(sid)
        if not entry:
            raise ConfigError(f"derived {sid!r} has no series entry in power.yaml")
        if str(entry.get("unit")) != unit:
            raise ConfigError(f"derived {sid!r}: series unit {entry.get('unit')!r} != {unit!r}")
        schema = str(entry.get("schema", "observation"))
        if kind == "forecast" and schema != "forecast":
            raise ConfigError(f"derived {sid!r}: components_kind forecast needs a forecast series")
        specs[sid] = DerivedSpec(
            series_id=sid, components=comps, scale=scale, unit=unit, component_unit=comp_unit,
            forecast_view=(schema == "forecast" and kind == "observation"),
            coefs=coefs, constant=constant, components_kind=kind)
    return specs


def _vintage_pub(groups: Any) -> pd.Series:
    """Publication of each derived vintage: the latest component publication it depends on."""
    return groups["publication_time"].max()


def _single(values: pd.Series, what: str) -> str:
    uniq = sorted(set(values.astype(str)))
    if len(uniq) != 1:
        raise DataError(f"components disagree on {what}: {uniq}")
    return uniq[0]


_GRADE_BY_RANK = {rank: grade.value for grade, rank in PIT_GRADE_ORDER.items()}


def _derive_one(comps: pd.DataFrame, spec: DerivedSpec) -> pd.DataFrame:
    """Derived rows for one spec, vectorised.

    Visibility is the as-of rule of ``features/point_in_time.py`` (``resolve_as_of``): at a step
    ``p`` a component's state is its visible row (``publication_time <= p``) with the highest
    revision, then the latest publication. Steps are the distinct component publication times of
    each event time. Grade C is kept so it propagates to the derived row.
    """
    raise_on_conflicting_ties(comps, ["series_id", "event_time", "revision", "publication_time"])
    entitlement = _single(comps["entitlement"], "entitlement")
    tz = _single(comps["source_timezone"], "source_timezone")
    steps = (comps[["event_time", "publication_time"]].drop_duplicates()
             .rename(columns={"publication_time": "step"}))
    cand = steps.merge(comps, on="event_time")
    cand = cand.loc[cand["publication_time"] <= cand["step"]]
    cand = cand.sort_values(["event_time", "step", "series_id", "revision", "publication_time",
                             "ingestion_time"], kind="mergesort")
    best = cand.drop_duplicates(["event_time", "step", "series_id"], keep="last").copy()
    best["rank"] = best["pit_grade"].map(lambda g: PIT_GRADE_ORDER[PitGrade(g)])
    best["w"] = best["value"] * best["series_id"].map(spec.coefficient)
    best["nonok"] = best["quality_flag"].where(best["quality_flag"] != QualityFlag.OK.value)
    best["synthetic"] = best["source"] == SYNTHETIC_SOURCE
    best["is_nan"] = best["value"].isna()
    g = best.groupby(["event_time", "step"], sort=True)
    st = pd.DataFrame({
        "n": g["series_id"].nunique(),
        "total": g["w"].sum(),
        "any_nan": g["is_nan"].any(),
        "pub": _vintage_pub(g),
        "ing": g["ingestion_time"].max(),
        "rank": g["rank"].max(),
        "synthetic": g["synthetic"].any(),
        "est": g["publication_time_estimated"].any(),
        "nonok": g["nonok"].min(),
        "geography": g["geography"].agg(lambda s: "+".join(sorted(set(s.astype(str))))),
    }).reset_index()
    complete = (st["n"] == len(spec.components)) & ~st["any_nan"]
    st["value"] = np.where(complete, spec.scale * (spec.constant + st["total"]), np.nan)
    st["grade"] = st["rank"].map(_GRADE_BY_RANK)
    st["flag"] = np.select(
        [st["value"].isna(), st["synthetic"], st["nonok"].notna()],
        [QualityFlag.MISSING_INPUT.value, QualityFlag.SYNTHETIC.value, st["nonok"].astype(str)],
        default=QualityFlag.OK.value)
    # A step whose derived state equals the previous step's writes no row.
    prev_val = st["value"].shift()
    same = ((st["event_time"] == st["event_time"].shift())
            & ((st["value"] == prev_val) | (st["value"].isna() & prev_val.isna()))
            & (st["pub"] == st["pub"].shift()) & (st["grade"] == st["grade"].shift()))
    st = st.loc[~same].copy()
    st["revision"] = st.groupby("event_time").cumcount().astype("int64")
    return pd.DataFrame({
        "series_id": spec.series_id,
        "event_time": st["event_time"].to_numpy(),
        "value": st["value"].astype(float).to_numpy(),
        "geography": st["geography"].to_numpy(),
        "publication_time": st["pub"].to_numpy(),
        "ingestion_time": pd.concat([st["ing"], st["pub"]], axis=1).max(axis=1).to_numpy(),
        "revision": st["revision"].to_numpy(),
        "unit": spec.unit,
        "source": np.where(st["synthetic"], SYNTHETIC_SOURCE, DERIVED_SOURCE),
        "quality_flag": st["flag"].to_numpy(),
        "entitlement": entitlement,
        "pit_grade": st["grade"].to_numpy(),
        "publication_time_estimated": st["est"].astype(bool).to_numpy(),
        "source_timezone": tz,
    })


def derive_observations(observations: pd.DataFrame, specs: Mapping[str, DerivedSpec]
                        ) -> pd.DataFrame:
    """OBSERVATION_SCHEMA rows for every spec, built from the component rows in ``observations``.

    Component rows of every vintage and grade are used; the as-of rule is applied by the readers.
    Raises ``DataError`` if a component carries the wrong unit, or the components disagree on
    entitlement or timezone.
    """
    frames: list[pd.DataFrame] = []
    for spec in specs.values():
        if spec.components_kind != "observation":
            continue
        comps = observations.loc[observations["series_id"].isin(spec.components)]
        if comps.empty:
            continue
        wrong = comps.loc[comps["unit"] != spec.component_unit]
        if len(wrong):
            raise DataError(f"{spec.series_id}: components must be in {spec.component_unit}, "
                            f"got {sorted(set(wrong['unit']))}")
        frames.append(_derive_one(comps, spec))
    if not frames:
        return empty_frame(OBSERVATION_SCHEMA)
    out = pd.concat(frames, ignore_index=True)
    for col in ("event_time", "publication_time", "ingestion_time"):
        out[col] = pd.to_datetime(out[col], utc=True)
    return validate_frame(out, OBSERVATION_SCHEMA)


def forecast_rows(derived: pd.DataFrame, specs: Mapping[str, DerivedSpec]) -> pd.DataFrame:
    """FORECAST_SCHEMA view of the derived rows whose spec has ``forecast_view``."""
    ids = [s.series_id for s in specs.values() if s.forecast_view]
    fv = derived.loc[derived["series_id"].isin(ids)].copy()
    if fv.empty:
        return empty_frame(FORECAST_SCHEMA)
    local_day = (pd.DatetimeIndex(fv["event_time"]).tz_convert(MARKET_TZ).tz_localize(None)
                 .normalize())
    fv["valid_time"] = local_day.tz_localize("UTC")
    fv["issue_time"] = fv["publication_time"]
    fv["lead_hours"] = (fv["valid_time"] - fv["issue_time"]).dt.total_seconds() / 3600.0
    fv["model"] = DERIVED_MODEL
    fv["member"] = DERIVED_MEMBER
    if (fv.groupby(["series_id", "valid_time"])["event_time"].nunique() > 1).any():
        raise DataError("two event times map to one valid day in a derived forecast view")
    return validate_frame(fv[FORECAST_SCHEMA.names].reset_index(drop=True), FORECAST_SCHEMA)


def derive_forecasts(forecasts: pd.DataFrame, specs: Mapping[str, DerivedSpec]) -> pd.DataFrame:
    """FORECAST_SCHEMA rows for the specs whose components are forecasts (``components_kind: forecast``).

    One derived row per (issue_time, valid_time), and only where EVERY component has a row for that issue: a missing
    component gives no row (not a NaN row). Each component keeps its latest publication. The derived row is published
    at the latest component publication and carries the worst component grade, so it is never visible before its last
    input. Raises ``DataError`` on a wrong component unit or on components that disagree on entitlement or timezone.
    """
    frames: list[pd.DataFrame] = []
    for spec in specs.values():
        if spec.components_kind != "forecast":
            continue
        comps = forecasts.loc[forecasts["series_id"].isin(spec.components)]
        if comps.empty:
            continue
        wrong = comps.loc[comps["unit"] != spec.component_unit]
        if len(wrong):
            raise DataError(f"{spec.series_id}: components must be in {spec.component_unit}, "
                            f"got {sorted(set(wrong['unit']))}")
        _single(comps["entitlement"], "entitlement")
        tz = _single(comps["source_timezone"], "source_timezone")
        comps = (comps.sort_values(["series_id", "issue_time", "valid_time", "publication_time", "ingestion_time"],
                                   kind="mergesort")
                 .drop_duplicates(["series_id", "issue_time", "valid_time"], keep="last").copy())
        comps["w"] = comps["value"] * comps["series_id"].map(spec.coefficient)
        comps["rank"] = comps["pit_grade"].map(lambda g: PIT_GRADE_ORDER[PitGrade(g)])
        g = comps.groupby(["issue_time", "valid_time"], sort=True)
        st = pd.DataFrame({
            "n": g["series_id"].nunique(),
            "total": g["w"].sum(),
            "any_nan": g["value"].agg(lambda s: bool(s.isna().any())),
            "pub": g["publication_time"].max(),
            "ing": g["ingestion_time"].max(),
            "rank": g["rank"].max(),
            "est": g["publication_time_estimated"].any(),
            "geography": g["geography"].agg(lambda s: "+".join(sorted(set(s.astype(str))))),
            "entitlement": g["entitlement"].first(),
        }).reset_index()
        st = st.loc[(st["n"] == len(spec.components)) & ~st["any_nan"]]
        if st.empty:
            continue
        issue = pd.to_datetime(st["issue_time"], utc=True)
        valid = pd.to_datetime(st["valid_time"], utc=True)
        pub = pd.to_datetime(st["pub"], utc=True)
        ing = pd.concat([pd.to_datetime(st["ing"], utc=True), pub], axis=1).max(axis=1)
        frames.append(pd.DataFrame({
            "series_id": spec.series_id,
            "issue_time": issue.to_numpy(),
            "valid_time": valid.to_numpy(),
            "lead_hours": (valid - issue).dt.total_seconds().to_numpy() / 3600.0,
            "model": DERIVED_MODEL,
            "member": DERIVED_MEMBER,
            "value": (spec.scale * (spec.constant + st["total"])).astype(float).to_numpy(),
            "geography": st["geography"].to_numpy(),
            "publication_time": pub.to_numpy(),
            "ingestion_time": ing.to_numpy(),
            "revision": np.zeros(len(st), dtype="int64"),
            "unit": spec.unit,
            "source": DERIVED_SOURCE,
            "quality_flag": QualityFlag.OK.value,
            "entitlement": st["entitlement"].to_numpy(),
            "pit_grade": st["rank"].map(_GRADE_BY_RANK).to_numpy(),
            "publication_time_estimated": st["est"].astype(bool).to_numpy(),
            "source_timezone": tz,
        }))
    if not frames:
        return empty_frame(FORECAST_SCHEMA)
    out = pd.concat(frames, ignore_index=True)
    for col in ("issue_time", "valid_time", "publication_time", "ingestion_time"):
        out[col] = pd.to_datetime(out[col], utc=True)
    return validate_frame(out[FORECAST_SCHEMA.names], FORECAST_SCHEMA)


def with_derived_series(bundle: DataBundle, specs: Mapping[str, DerivedSpec]) -> DataBundle:
    """``bundle`` plus the derived rows for ``specs``, built from its raw components.

    The bundle must carry the raw components only: a derived id already present is an error, so
    a derived value can never be defined twice.
    """
    if not specs:
        return bundle
    clash_obs = set(bundle.observations["series_id"]) & set(specs)
    clash_fc = set(bundle.forecasts["series_id"]) & set(specs)
    if clash_obs or clash_fc:
        raise DataError(f"derived ids already present in the bundle: {sorted(clash_obs | clash_fc)}")
    derived = derive_observations(bundle.observations, specs)
    fc_new = pd.concat([forecast_rows(derived, specs), derive_forecasts(bundle.forecasts, specs)],
                       ignore_index=True)
    obs = (pd.concat([bundle.observations, derived], ignore_index=True)
           if len(derived) else bundle.observations)
    fc = (pd.concat([bundle.forecasts, fc_new], ignore_index=True)
          if len(fc_new) else bundle.forecasts)
    return DataBundle(obs, fc, bundle.curves, bundle.contracts)
