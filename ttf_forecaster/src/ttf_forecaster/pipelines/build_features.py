"""Assemble the as-of feature matrix from the enabled families, and the leakage canary.

``build_feature_matrix(bundle, origins, config)`` calls every enabled family builder from
``configs/features.yaml`` (lazily imported), checks each against the builder contract, joins them
on ``origin_time``, runs post-stage families (interactions) on the joined matrix, and drops grade-C
columns, and columns of families with ``attrs["clean_backtest_eligible"] = False``, unless the
config explicitly allows a contaminated run.

``leakage_canary`` implements PIT policy §6: for chosen origins *o*, every bundle row with
``publication_time > o`` is perturbed (or dropped) and the matrix is rebuilt on the origins up to
*o*; every value must be bit-identical (NaN == NaN) to the unperturbed full build.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import (
    SYNTHETIC_SOURCE,
    DataBundle,
    PitGrade,
    PointInTimeViolation,
)
from ttf_forecaster.features.panel import as_origin_index
from ttf_forecaster.features.registry import (
    FAMILIES,
    FamilySpec,
    FeatureRegistryError,
    check_family_frame,
    feature_hash,
    load_dotted,
)
from ttf_forecaster.settings import ConfigError, load_config

log = logging.getLogger(__name__)


def load_feature_config(config: Mapping[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """``None`` -> ``configs/features.yaml``; a path -> that YAML; a mapping -> deep copy."""
    if config is None:
        return copy.deepcopy(load_config("features"))
    if isinstance(config, (str, Path)):
        p = Path(config)
        return copy.deepcopy(load_config(p.stem, str(p.parent)))
    return copy.deepcopy(dict(config))


def family_params(cfg: Mapping[str, Any], family: str) -> dict[str, Any]:
    """``common`` params overlaid with the family's own ``params``."""
    fam = (cfg.get("families") or {}).get(family) or {}
    return {**copy.deepcopy(cfg.get("common") or {}), **copy.deepcopy(fam.get("params") or {})}


def _spec(cfg: Mapping[str, Any], family: str) -> FamilySpec:
    fam = (cfg.get("families") or {}).get(family) or {}
    builder = fam.get("builder")
    if family in FAMILIES:
        base = FAMILIES[family]
        return FamilySpec(family, builder or base.builder, fam.get("stage", base.stage),
                          base.description)
    if not builder:
        raise FeatureRegistryError(f"family {family!r} is not registered and has no 'builder'")
    log.warning("custom feature family %r from %s", family, builder)
    return FamilySpec(family, builder, fam.get("stage", "bundle"))


def enabled_families(cfg: Mapping[str, Any], only: Iterable[str] | None = None) -> list[str]:
    fams = cfg.get("families")
    if not isinstance(fams, Mapping):
        raise ConfigError("features config needs a 'families' mapping")
    names = [n for n, f in fams.items() if (f or {}).get("enabled", False)]
    if only is not None:
        wanted = set(only)
        unknown = wanted - set(fams)
        if unknown:
            raise ConfigError(f"families {sorted(unknown)} not in features config")
        names = [n for n in names if n in wanted]
    return names


def _is_synthetic(bundle: DataBundle) -> bool:
    return any(
        len(f) and "source" in f.columns and (f["source"] == SYNTHETIC_SOURCE).any()
        for f in (bundle.observations, bundle.forecasts, bundle.curves)
    )


def build_feature_matrix(
    bundle: DataBundle,
    origins: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    families: Iterable[str] | None = None,
    allow_grade_c: bool | None = None,
) -> pd.DataFrame:
    """Feature matrix indexed by ``origin_time`` (the origins, in the given order).

    Args:
        bundle: all vintages.
        origins: tz-aware forecast origins.
        config: features config (see ``load_feature_config``).
        families: restrict to these enabled families (ablations).
        allow_grade_c: override ``config["allow_grade_c"]``.

    Returns:
        DataFrame with ``attrs``: ``pit_grade`` (column -> grade), ``families``,
        ``feature_hash``, ``dropped_grade_c`` (list), ``contains_grade_c`` (bool),
        ``is_synthetic`` (bool).

    Raises:
        FeatureRegistryError: missing builder module, contract violation, duplicate columns.
    """
    cfg = load_feature_config(config)
    idx = as_origin_index(origins)
    names = enabled_families(cfg, families)
    specs = [_spec(cfg, n) for n in names]
    frames: list[pd.DataFrame] = []
    grades: dict[str, str] = {}
    not_clean: list[str] = []
    params_used: dict[str, Any] = {}

    def _add(spec: FamilySpec, frame: pd.DataFrame) -> None:
        check_family_frame(frame, spec.name, idx)
        dup = set(frame.columns) & set(grades)
        if dup:
            raise FeatureRegistryError(f"duplicate feature columns across families: {sorted(dup)}")
        if frame.attrs.get("clean_backtest_eligible", True) is False:
            not_clean.extend(frame.columns)
        frames.append(frame)
        grades.update({c: str(frame.attrs["pit_grade"][c]) for c in frame.columns})

    for spec in [s for s in specs if s.stage == "bundle"]:
        params = family_params(cfg, spec.name)
        params_used[spec.name] = params
        _add(spec, load_dotted(spec.builder)(bundle, idx, params))

    matrix = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=idx)
    matrix.attrs["pit_grade"] = dict(grades)
    for spec in [s for s in specs if s.stage == "post"]:
        params = family_params(cfg, spec.name)
        params_used[spec.name] = params
        if spec.name == "interactions" and params.get("standardise", "expanding") == "expanding":
            params_used[spec.name] = {**params, "_sample_first_origin": str(idx.min())}
        post = load_dotted(spec.builder)(matrix, idx, params)
        _add(spec, post)
        matrix = pd.concat([matrix, post], axis=1)
    matrix.index.name = "origin_time"

    allow_c = bool(cfg.get("allow_grade_c", False) if allow_grade_c is None else allow_grade_c)
    # columns from a family that declares attrs["clean_backtest_eligible"] = False (text) are
    # treated exactly like grade C: dropped unless a contaminated run is explicitly allowed.
    grade_c = [c for c, g in grades.items() if g == PitGrade.C.value or c in set(not_clean)]
    if grade_c and not allow_c:
        log.warning("dropping %d grade-C feature columns: %s", len(grade_c), grade_c[:10])
        matrix = matrix.drop(columns=grade_c)
        for c in grade_c:
            grades.pop(c)
    elif grade_c:
        log.warning("CONTAMINATED run: %d grade-C columns kept", len(grade_c))
    matrix.attrs = {
        "pit_grade": {c: grades[c] for c in matrix.columns},
        "families": names,
        "feature_hash": feature_hash(matrix.columns, params_used),
        "dropped_grade_c": [] if allow_c else grade_c,
        "not_clean_eligible": list(not_clean),
        "contains_grade_c": bool(grade_c) and allow_c,
        "is_synthetic": _is_synthetic(bundle),
    }
    return matrix


# ---------------------------------------------------------------------------------------------
# Leakage canary
# ---------------------------------------------------------------------------------------------


def perturb_after(bundle: DataBundle, as_of: pd.Timestamp, mode: str,
                  rng: np.random.Generator) -> DataBundle:
    """Copy of ``bundle`` with every row published after ``as_of`` perturbed or dropped.

    ``mode="perturb"`` multiplies values by U(0.5, 1.5) (sign-preserving, so price logs stay
    defined); ``mode="drop"`` removes the rows. Contract metadata is static and left unchanged.
    """
    if mode not in ("perturb", "drop"):
        raise ValueError(f"unknown canary mode {mode!r}")

    def one(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()
        late = (df["publication_time"] > as_of).to_numpy()
        if mode == "drop":
            return df.loc[~late].reset_index(drop=True)
        out = df.copy()
        out.loc[late, "value"] = out.loc[late, "value"] * rng.uniform(0.5, 1.5, int(late.sum()))
        return out

    return DataBundle(one(bundle.observations), one(bundle.forecasts), one(bundle.curves),
                      bundle.contracts.copy())


@dataclass
class CanaryReport:
    """Outcome of a leakage canary. ``failures``: (origin, mode, differing columns)."""

    checked_origins: list[pd.Timestamp] = field(default_factory=list)
    failures: list[tuple[pd.Timestamp, str, list[str]]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def raise_if_failed(self) -> None:
        if self.failures:
            lines = [f"origin {o.isoformat()} [{m}]: {cols[:10]}" for o, m, cols in self.failures]
            raise PointInTimeViolation("leakage canary failed — features changed when data "
                                       "published after the origin was altered:\n"
                                       + "\n".join(lines))


def _differing(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    if list(a.columns) != list(b.columns):
        return sorted(set(a.columns) ^ set(b.columns)) or ["<column order>"]
    return [c for c in a.columns
            if not np.array_equal(a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float),
                                  equal_nan=True)]


def leakage_canary(
    bundle: DataBundle,
    origins: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    config: Mapping[str, Any] | str | Path | None = None,
    *,
    check_origins: Sequence[pd.Timestamp] | None = None,
    n_checks: int = 4,
    modes: Sequence[str] = ("perturb", "drop"),
    seed: int = 0,
    families: Iterable[str] | None = None,
) -> CanaryReport:
    """Rebuild after perturbing/dropping everything published after each check origin.

    The reference is the full unperturbed build over all ``origins``; for check origin *o* the
    perturbed rebuild uses the origins ``<= o`` only, so dependence on later origins (e.g. a
    full-sample normalisation) is caught as well as dependence on later data.

    Returns:
        ``CanaryReport``; call ``raise_if_failed()`` to turn failures into
        ``PointInTimeViolation``.
    """
    cfg = load_feature_config(config)
    fams = list(families) if families is not None else None
    idx = as_origin_index(origins)
    full = build_feature_matrix(bundle, idx, cfg, families=fams)
    if check_origins is None:
        srt = idx.sort_values()
        pos = np.unique(np.linspace(len(srt) // 4, len(srt) - 2, n_checks).astype(int))
        checks = [srt[i] for i in pos if 0 <= i < len(srt)]
    else:
        checks = list(as_origin_index(check_origins))
    rng = np.random.default_rng(seed)
    report = CanaryReport(checked_origins=checks)
    for o in checks:
        sub = idx[idx <= o]
        ref = full.loc[sub]
        for mode in modes:
            alt = build_feature_matrix(perturb_after(bundle, o, mode, rng), sub, cfg,
                                       families=fams)
            diff = _differing(ref, alt)
            if diff:
                report.failures.append((o, mode, diff))
    return report
