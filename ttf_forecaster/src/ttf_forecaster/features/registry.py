"""Feature-family registry: builders, column -> family mapping, PIT grades, selection, hashing.

Every feature column is named ``<family>__<feature>``; the family is the prefix before the first
``__``. Family builders are loaded lazily by dotted path (``"package.module:function"``) so a
family whose module does not exist yet only fails when it is enabled, with a clear error.

Builder contract (``data/base.py::FeatureBuilder``)::

    build_<family>_features(bundle: DataBundle, origins: pd.DatetimeIndex,
                            params: dict) -> pd.DataFrame
        index  = origins (tz-aware UTC, name "origin_time"), same order, no extra rows
        columns = "<family>__<name>", numeric
        attrs["pit_grade"] = {column: "A" | "B" | "C"}   # worst grade of the column's inputs

Post-stage families (``interactions``) take the assembled matrix instead of the bundle::

    build_<family>_features(matrix: pd.DataFrame, origins: pd.DatetimeIndex,
                            params: dict) -> pd.DataFrame
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ttf_forecaster.data.base import PIT_GRADE_ORDER, PitGrade, worst_grade

SEP = "__"


class FeatureRegistryError(RuntimeError):
    """Unknown family, unloadable builder, or a frame that breaks the column/grade contract."""


@dataclass(frozen=True)
class FamilySpec:
    """A feature family.

    Attributes:
        name: family name = column prefix.
        builder: default dotted path ``"module:function"``.
        stage: ``"bundle"`` (built from the DataBundle) or ``"post"`` (built from the matrix).
        description: one line for reports.
    """

    name: str
    builder: str
    stage: str = "bundle"
    description: str = ""


_F = "ttf_forecaster.features"
FAMILIES: dict[str, FamilySpec] = {
    s.name: s
    for s in (
        FamilySpec("price", f"{_F}.prices:build_price_features", description=(
            "lagged spliced front returns, realised vol, momentum, mean-reversion z")),
        FamilySpec("curve", f"{_F}.prices:build_curve_features", description=(
            "M1-M2, slope, curvature, prompt vs deferred, seasonal strip spreads")),
        FamilySpec("cross_commodity", f"{_F}.prices:build_cross_commodity_features",
                   description="JKM-TTF, HH netback spread, Brent ratio, NBP/THE/PEG basis"),
        FamilySpec("storage", f"{_F}.storage:build_storage_features"),
        FamilySpec("flows", f"{_F}.flows:build_flows_features"),
        FamilySpec("lng", f"{_F}.lng:build_lng_features"),
        FamilySpec("weather", f"{_F}.weather:build_weather_features"),
        FamilySpec("power", f"{_F}.power:build_power_features"),
        FamilySpec("calendar", f"{_F}.calendar:build_calendar_features"),
        FamilySpec("interactions", f"{_F}.interactions:build_interactions_features",
                   stage="post", description="interpretable cross-family interactions"),
        FamilySpec("text", "ttf_forecaster.text.to_features:build_text_features",
                   description="LLM-extracted events; excluded until provenance exists"),
    )
}

PRICE_ONLY_FAMILIES: tuple[str, ...] = ("price", "curve", "calendar")
"""The H0 control set (``research_specification.md`` §8.3)."""

PRICE_DERIVED_COLUMNS: tuple[str, ...] = (
    "power__clean_spark_eur_mwh",
    "power__clean_dark_eur_mwh",
    "power__spark_minus_dark_eur_mwh",
    "power__switch_gas_price_eur_mwh",
    "power__gas_minus_switch_eur_mwh",
)
"""Columns of non-price families that embed the TTF price itself. ``select(price_only=True)``
puts them in the H0 set so criterion 3 (fundamentals beat price-only) is not biased toward
fundamentals by price information hiding in a fundamental family."""


def load_dotted(path: str) -> Callable[..., Any]:
    """Import ``"package.module:attr"`` and return the attribute.

    Raises:
        FeatureRegistryError: malformed path, module missing, or attribute missing.
    """
    if ":" not in path:
        raise FeatureRegistryError(f"builder path {path!r} must look like 'package.module:function'")
    mod_name, _, attr = path.partition(":")
    try:
        module = importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        raise FeatureRegistryError(
            f"feature builder module {mod_name!r} is not available ({exc}). Disable the family in "
            "configs/features.yaml or wait for its owning agent to deliver it."
        ) from exc
    try:
        return getattr(module, attr)  # type: ignore[no-any-return]
    except AttributeError as exc:
        raise FeatureRegistryError(f"module {mod_name!r} has no attribute {attr!r}") from exc


def get_family(name: str) -> FamilySpec:
    try:
        return FAMILIES[name]
    except KeyError as exc:
        raise FeatureRegistryError(f"unknown feature family {name!r}; known: {sorted(FAMILIES)}") \
            from exc


def family_of(column: str) -> str:
    """Family of a column from its ``<family>__`` prefix."""
    fam, sep, rest = column.partition(SEP)
    if not sep or not fam or not rest:
        raise FeatureRegistryError(f"column {column!r} is not named '<family>__<feature>'")
    return fam


def column_families(columns: Iterable[str]) -> dict[str, str]:
    return {c: family_of(c) for c in columns}


def check_family_frame(frame: pd.DataFrame, family: str, origins: pd.DatetimeIndex) -> None:
    """Enforce the builder contract for one family's output. Raises ``FeatureRegistryError``."""
    problems: list[str] = []
    if not frame.index.equals(origins):
        problems.append("index is not exactly the origins passed in (same values, same order)")
    bad_prefix = [c for c in frame.columns if not str(c).startswith(f"{family}{SEP}")]
    if bad_prefix:
        problems.append(f"columns without '{family}{SEP}' prefix: {bad_prefix[:5]}")
    if frame.columns.duplicated().any():
        problems.append(f"duplicate columns {frame.columns[frame.columns.duplicated()].tolist()}")
    grades = frame.attrs.get("pit_grade")
    if not isinstance(grades, Mapping):
        problems.append('attrs["pit_grade"] missing')
    else:
        missing = [c for c in frame.columns if c not in grades]
        if missing:
            problems.append(f"pit_grade missing for {missing[:5]}")
        invalid = {g for g in grades.values() if g not in {x.value for x in PitGrade}}
        if invalid:
            problems.append(f"invalid pit grades {sorted(invalid)}")
    non_num = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_num:
        problems.append(f"non-numeric columns {non_num[:5]}")
    else:
        inf = [c for c in frame.columns if frame[c].abs().eq(float("inf")).any()]
        if inf:
            problems.append(f"infinite values in {inf[:5]} (unavailable must be NaN)")
    if problems:
        raise FeatureRegistryError(f"family {family!r} output violates the contract: "
                                   + "; ".join(problems))


def grade_of(frame: pd.DataFrame, column: str) -> str:
    try:
        return str(frame.attrs["pit_grade"][column])
    except KeyError as exc:
        raise FeatureRegistryError(f"no pit_grade recorded for {column!r}") from exc


def worst_grade_of(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    """Worst PIT grade over ``columns`` of ``frame`` (propagation rule for derived features)."""
    return worst_grade([grade_of(frame, c) for c in columns]).value


def select(
    frame: pd.DataFrame,
    *,
    families: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
    max_grade: str | PitGrade = PitGrade.B,
    price_only: bool = False,
) -> pd.DataFrame:
    """Columns for an ablation or a clean run.

    Args:
        frame: feature matrix with ``attrs["pit_grade"]``.
        families: keep only these families (``None`` = all).
        exclude: drop these families.
        max_grade: drop columns graded worse than this (``"B"`` = clean run, ``"C"`` = all).
        price_only: the H0 set: ``PRICE_ONLY_FAMILIES`` plus ``PRICE_DERIVED_COLUMNS``. Cannot be
            combined with ``families``.

    Returns:
        A new frame; ``attrs`` copied with ``pit_grade`` restricted to the kept columns.
    """
    limit = PIT_GRADE_ORDER[PitGrade(max_grade)]
    fam = column_families(frame.columns)
    if price_only and families is not None:
        raise FeatureRegistryError("select(): pass either families or price_only, not both")
    extra = set(PRICE_DERIVED_COLUMNS) if price_only else set()
    keep_f = set(PRICE_ONLY_FAMILIES) if price_only else (
        set(families) if families is not None else None)
    drop_f = set(exclude or ())
    cols = [
        c for c in frame.columns
        if (keep_f is None or fam[c] in keep_f or c in extra)
        and fam[c] not in drop_f
        and PIT_GRADE_ORDER[PitGrade(grade_of(frame, c))] <= limit
    ]
    out = frame[cols].copy()
    out.attrs = {**frame.attrs, "pit_grade": {c: frame.attrs["pit_grade"][c] for c in cols}}
    return out


def feature_hash(columns: Iterable[str], params: Mapping[str, Any] | None = None) -> str:
    """sha256 of the sorted column names plus the (JSON-canonicalised) parameters."""
    payload = json.dumps(
        {"columns": sorted(columns), "params": params or {}}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
