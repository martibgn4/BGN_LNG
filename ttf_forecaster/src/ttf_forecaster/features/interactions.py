"""Interpretable interaction features (``research_specification.md`` §5, ``interactions`` family).

Each interaction is a named, documented function of **named input columns** (roles mapped to
matrix columns in ``configs/features.yaml``). Inputs are standardised, each leg is made one-sided
(``relu``) in the tightness direction, then multiplied: the value is > 0 only when **both**
stresses are present (e.g. colder *and* emptier storage) and 0 when either is absent. A plain
product of signed z-scores would score "warm and full" as bullish; that is ruled out by test.
``active_months`` (any spec) sets the interaction to 0 outside those local origin months.

Standardisation never sees data after the point it is used:

* ``InteractionTransformer`` (fit/transform): mean/std fitted on the **training rows only**; use
  this inside a model pipeline, refitted per fold.
* ``build_interactions_features`` (matrix stage): **expanding** z-score over origins
  ``<= origin`` (``min_periods``), so the matrix itself is PIT-safe without fold knowledge. Its
  values depend on the first origin of the sample, which is part of ``feature_hash`` params.

PIT grade of an interaction = worst grade of its inputs.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import MARKET_TZ, worst_grade
from ttf_forecaster.features.panel import as_origin_index
from ttf_forecaster.features.registry import FeatureRegistryError

FAMILY = "interactions"
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------------
# The interactions. Arguments are already-standardised Series unless listed in ``raw_roles``.
#
# Every leg is ONE-SIDED (relu) before multiplying. A product of two signed z-scores would be
# positive for "warm AND full storage" (-1 x -1) and wrongly read as bullish; with relu legs the
# interaction is > 0 only when BOTH stresses are present and 0 otherwise.
# ---------------------------------------------------------------------------------------------


def relu(x: pd.Series) -> pd.Series:
    """max(x, 0), NaN preserved."""
    return x.clip(lower=0.0)


def cold_x_low_storage(cold: pd.Series, storage: pd.Series, **_: Any) -> pd.Series:
    """relu(z cold revision, + = colder) x relu(-z storage deviation, + = emptier than normal)."""
    return relu(cold) * relu(-storage)


def norway_outage_x_low_storage(outage: pd.Series, storage: pd.Series, **_: Any) -> pd.Series:
    """relu(z Norwegian unavailable capacity) x relu(-z storage deviation) (H4)."""
    return relu(outage) * relu(-storage)


def residual_load_x_gas_below_switch(
    residual_load: pd.Series, gas_minus_switch: pd.Series, **_: Any
) -> pd.Series:
    """relu(z upward residual-load revision) x relu(-z (gas - coal-switch price)): more thermal
    demand while gas is the cheaper fuel, so gas-for-power burn rises."""
    return relu(residual_load) * relu(-gas_minus_switch)


def jkm_premium_x_falling_arrivals(
    jkm_premium: pd.Series, arrivals_change: pd.Series, **_: Any
) -> pd.Series:
    """relu(raw NEA-minus-NWE arb, USD/MMBtu: only when Asia actually pays more) x
    relu(-z change in expected EU arrivals) (H3)."""
    return relu(jkm_premium) * relu(-arrivals_change)


def no_spare_slots_x_high_arrivals(
    slots_available: pd.Series, arrivals: pd.Series, **_: Any
) -> pd.Series:
    """relu(-z available regas slots, + = no spare capacity) x relu(z expected arrivals):
    cargoes arriving into a full terminal system. Expected sign on prompt spread: +."""
    return relu(-slots_available) * relu(arrivals)


def expiry_proximity(bdays_to_expiry: pd.Series, window_bdays: float) -> pd.Series:
    """1 at expiry, falling linearly to 0 at ``window_bdays`` business days before it."""
    if window_bdays <= 0:
        raise ValueError("window_bdays must be positive")
    return (1.0 - bdays_to_expiry / float(window_bdays)).clip(lower=0.0, upper=1.0)


def expiry_proximity_x_prompt_stress(
    bdays_to_expiry: pd.Series, prompt_spread: pd.Series, *, window_bdays: float = 10.0,
    **_: Any,
) -> pd.Series:
    """Expiry proximity in [0, 1] (raw business days) x relu(z deseasonalised M1-M2)."""
    return expiry_proximity(bdays_to_expiry, window_bdays) * relu(prompt_spread)


@dataclass(frozen=True)
class InteractionDef:
    """A named interaction: its roles, the function, and roles used raw (not standardised)."""

    name: str
    roles: tuple[str, ...]
    func: Callable[..., pd.Series]
    raw_roles: tuple[str, ...] = ()
    options: tuple[str, ...] = ()


_GENERIC_OPTIONS = ("active_months",)

INTERACTIONS: dict[str, InteractionDef] = {
    d.name: d
    for d in (
        InteractionDef("cold_x_low_storage", ("cold", "storage"), cold_x_low_storage),
        InteractionDef("norway_outage_x_low_storage", ("outage", "storage"),
                       norway_outage_x_low_storage),
        InteractionDef("residual_load_x_gas_below_switch", ("residual_load", "gas_minus_switch"),
                       residual_load_x_gas_below_switch),
        InteractionDef("jkm_premium_x_falling_arrivals", ("jkm_premium", "arrivals_change"),
                       jkm_premium_x_falling_arrivals, raw_roles=("jkm_premium",)),
        InteractionDef("no_spare_slots_x_high_arrivals", ("slots_available", "arrivals"),
                       no_spare_slots_x_high_arrivals),
        InteractionDef("expiry_proximity_x_prompt_stress", ("bdays_to_expiry", "prompt_spread"),
                       expiry_proximity_x_prompt_stress, raw_roles=("bdays_to_expiry",),
                       options=("window_bdays",)),
    )
}


@dataclass(frozen=True)
class _Resolved:
    d: InteractionDef
    inputs: dict[str, str]
    options: dict[str, Any]


def resolve_specs(specs: Mapping[str, Mapping[str, Any]]) -> list[_Resolved]:
    """Validate the ``specs`` block of the config; skip entries with ``enabled: false``."""
    out: list[_Resolved] = []
    for name, spec in specs.items():
        if not spec.get("enabled", True):
            continue
        if name not in INTERACTIONS:
            raise FeatureRegistryError(f"unknown interaction {name!r}; known {sorted(INTERACTIONS)}")
        d = INTERACTIONS[name]
        inputs = dict(spec.get("inputs") or {})
        if set(inputs) != set(d.roles):
            raise FeatureRegistryError(
                f"interaction {name}: inputs must map exactly roles {d.roles}, got {sorted(inputs)}"
            )
        opts = {k: spec[k] for k in (*d.options, *_GENERIC_OPTIONS) if k in spec}
        out.append(_Resolved(d, {r: str(inputs[r]) for r in d.roles}, opts))
    return out


def _check_columns(frame: pd.DataFrame, specs: list[_Resolved]) -> None:
    missing = sorted({c for s in specs for c in s.inputs.values() if c not in frame.columns})
    if missing:
        raise FeatureRegistryError(
            f"interaction inputs not in the feature matrix: {missing} (enable the parent family or "
            "fix the column names in configs/features.yaml interactions.params.specs)"
        )


def _grades(frame: pd.DataFrame, s: _Resolved) -> str | None:
    g = frame.attrs.get("pit_grade")
    if not isinstance(g, Mapping):
        return None
    try:
        return worst_grade([g[c] for c in s.inputs.values()]).value
    except KeyError as exc:
        raise FeatureRegistryError(f"no pit_grade for interaction input {exc}") from exc


def _combine(s: _Resolved, z: Mapping[str, pd.Series], index: pd.Index) -> pd.Series:
    opts = {k: v for k, v in s.options.items() if k not in _GENERIC_OPTIONS}
    out = s.d.func(**{r: z[r] for r in s.d.roles}, **opts).astype(float)
    months = s.options.get("active_months")
    if months is not None:
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            raise FeatureRegistryError(f"{s.d.name}: active_months needs a tz-aware origin index")
        local = index.tz_convert(MARKET_TZ).month
        # structurally inactive outside the season: 0 (not NaN) where inputs exist
        out = out.where(np.isin(local, list(months)), out * 0.0)
    return out


@dataclass
class InteractionTransformer:
    """Train-only standardisation + interaction products (sklearn-style fit/transform).

    Args:
        specs: the ``specs`` mapping from ``configs/features.yaml``.
        min_std: a training std below this is an error (constant input cannot be standardised).

    ``transform`` returns only the ``interactions__*`` columns, with ``attrs["pit_grade"]`` from
    the grades seen at ``fit`` (or the transform frame's attrs).
    """

    specs: Mapping[str, Mapping[str, Any]]
    min_std: float = 1e-12
    mean_: dict[str, float] = field(default_factory=dict, init=False)
    std_: dict[str, float] = field(default_factory=dict, init=False)
    grades_: dict[str, str | None] = field(default_factory=dict, init=False)

    def fit(self, X: pd.DataFrame, y: object = None) -> InteractionTransformer:  # noqa: N803
        resolved = resolve_specs(self.specs)
        _check_columns(X, resolved)
        for s in resolved:
            for role, col in s.inputs.items():
                if role in s.d.raw_roles:
                    continue
                x = X[col].astype(float)
                mu, sd = float(x.mean()), float(x.std(ddof=1))
                if not np.isfinite(sd) or sd < self.min_std:
                    raise ValueError(f"cannot standardise {col!r}: training std {sd}")
                self.mean_[col], self.std_[col] = mu, sd
            self.grades_[s.d.name] = _grades(X, s)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        if not self.grades_:
            raise RuntimeError("InteractionTransformer.transform called before fit")
        resolved = resolve_specs(self.specs)
        _check_columns(X, resolved)
        out = pd.DataFrame(index=X.index)
        grades: dict[str, str] = {}
        for s in resolved:
            z = {
                r: (X[c].astype(float) if r in s.d.raw_roles
                    else (X[c].astype(float) - self.mean_[c]) / self.std_[c])
                for r, c in s.inputs.items()
            }
            col = f"{FAMILY}__{s.d.name}"
            out[col] = _combine(s, z, out.index)
            g = self.grades_.get(s.d.name) or _grades(X, s)
            if g is not None:
                grades[col] = g
        out.attrs["pit_grade"] = grades
        return out

    def fit_transform(self, X: pd.DataFrame, y: object = None) -> pd.DataFrame:  # noqa: N803
        return self.fit(X).transform(X)


def expanding_zscore(x: pd.Series, min_periods: int) -> pd.Series:
    """z-score of each value vs the mean/std of all values at or before it (time order)."""
    order = x.sort_index()
    roll = order.expanding(min_periods=min_periods)
    sd = roll.std(ddof=1)
    z = (order - roll.mean()) / sd.where(sd > 0)
    return z.reindex(x.index)


def build_interactions_features(
    matrix: pd.DataFrame, origins: pd.DatetimeIndex, params: Mapping[str, Any]
) -> pd.DataFrame:
    """Matrix-stage builder: expanding (PIT-safe) standardisation, or ``standardise: none``.

    Params: ``standardise`` (``expanding`` | ``none``), ``min_periods``, ``specs``,
    ``on_missing_inputs`` (``error`` default | ``skip`` with a warning).
    """
    idx = as_origin_index(origins)
    resolved = resolve_specs(params.get("specs") or {})
    policy = str(params.get("on_missing_inputs", "error"))
    if policy == "skip":
        keep = [s for s in resolved if all(c in matrix.columns for c in s.inputs.values())]
        for s in resolved:
            if s not in keep:
                log.warning("interaction %s skipped: inputs %s not in the matrix", s.d.name,
                            [c for c in s.inputs.values() if c not in matrix.columns])
        resolved = keep
    elif policy != "error":
        raise ValueError(f"on_missing_inputs must be 'error' or 'skip', got {policy!r}")
    _check_columns(matrix, resolved)
    mode = str(params.get("standardise", "expanding"))
    if mode not in ("expanding", "none"):
        raise ValueError(f"interactions.standardise must be 'expanding' or 'none', got {mode!r}"
                         " (train-only fitting: use InteractionTransformer in the model pipeline)")
    mp = int(params.get("min_periods", 60))
    out = pd.DataFrame(index=idx)
    grades: dict[str, str] = {}
    for s in resolved:
        z = {}
        for r, c in s.inputs.items():
            x = matrix[c].astype(float).reindex(idx)
            z[r] = x if (r in s.d.raw_roles or mode == "none") else expanding_zscore(x, mp)
        col = f"{FAMILY}__{s.d.name}"
        out[col] = _combine(s, z, out.index)
        g = _grades(matrix, s)
        if g is None:
            raise FeatureRegistryError("matrix has no attrs['pit_grade']; cannot grade interactions")
        grades[col] = g
    out.index.name = "origin_time"
    out.attrs["pit_grade"] = grades
    return out
