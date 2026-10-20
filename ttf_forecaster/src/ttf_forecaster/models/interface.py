"""Common model interface, registry, factory, time-ordered inner CV and train-only preprocessing.

Every model in the ladder (Level 0 baselines … Level 6 ensembles) is a :class:`ForecastModel`:

* ``fit(X, y)`` — ``X`` is a feature frame indexed by ``origin_time`` (monotone increasing, unique),
  ``y`` the target aligned on the same index. **Everything** a model learns — imputation medians,
  scaler moments, PCA loadings, lag order, penalty strength, early-stopping round, calibration
  map — is estimated inside ``fit`` from these rows only. Purging of overlapping labels against
  the *outer* test block is the runner's job (``evaluation/splits.py``); purging inside the model's
  own inner CV is done here with a gap of ``horizon`` rows.
* ``predict(X)`` → ``pd.Series`` indexed like ``X`` (point forecast; for classifiers the event
  probability).
* ``predict_quantiles(X, levels)`` → frame with columns ``q<level>`` (e.g. ``q0.1``), rows
  non-decreasing across levels. Only for models that support it.
* ``predict_proba(X)`` → ``pd.Series`` of P(event). Only for classifiers.

Rows are assumed to be **one per trading day** (the forecast-origin grid). Gaps in rows are counted
in rows, so a gap of ``horizon`` rows equals ``horizon`` trading days on that grid.

Models are registered by ``name`` and built with :func:`make_model`, which merges class defaults
with ``configs/models.yaml`` and caller overrides.
"""

from __future__ import annotations

import copy
import hashlib
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, TypeVar

import numpy as np
import pandas as pd

from ttf_forecaster.data.base import MARKET_TZ
from ttf_forecaster.settings import ConfigError, load_config

# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------


class ModelError(RuntimeError):
    """Base class for model-layer failures."""


class NotFittedError(ModelError):
    """``predict*`` called before ``fit``."""


class InsufficientDataError(ModelError):
    """Too few training rows for the requested procedure (inner CV, early stopping, calibration)."""


class CointegrationGateError(ModelError):
    """An error-correction model was requested on series that fail the Engle–Granger gate."""


class LeakageGuardError(ModelError):
    """Inputs would let information from after ``train_end`` enter a fit."""


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

TASKS = ("regression", "classification", "volatility")
FAMILY_SEP = "__"


def quantile_column(level: float) -> str:
    """Column name for a quantile level, matching PREDICTION_SCHEMA's ``q<level>`` (``q0.1``)."""
    if not 0.0 < float(level) < 1.0:
        raise ValueError(f"quantile level must be in (0, 1), got {level}")
    return f"q{float(level):g}"


def feature_family(column: str) -> str:
    """Family of a ``<family>__<feature>`` column (FeatureBuilder contract); ``"other"`` if none."""
    return column.split(FAMILY_SEP, 1)[0] if FAMILY_SEP in column else "other"


def rearrange_quantiles(values: np.ndarray) -> np.ndarray:
    """Monotone rearrangement (Chernozhukov, Fernández-Val & Galichon 2010): sort each row.

    ``values`` is (n_rows, n_levels) in increasing level order. Sorting is the rearrangement for a
    finite grid and never worsens pinball loss.
    """
    return np.sort(np.asarray(values, dtype=float), axis=1)


def _to_ns(values: Any) -> np.ndarray:
    """tz-aware timestamps -> int64 UTC nanoseconds (tz-independent comparisons)."""
    idx = pd.DatetimeIndex(values)
    if idx.tz is None:
        raise TypeError("label_known_time and origins must be tz-aware timestamps")
    return np.asarray(idx.tz_convert("UTC").asi8)


def purge_by_label_time(train: np.ndarray, first_val: int, label_known_time: pd.Series | None,
                        origins: pd.Index | None) -> np.ndarray:
    """Drop training positions whose label is not yet known at the first validation origin.

    ``label_known_time`` and ``origins`` are positional (length n). With ``label_known_time=None``
    ``train`` is returned unchanged (the caller's row gap is then the only protection).
    """
    if label_known_time is None:
        return train
    if origins is None or len(origins) != len(label_known_time):
        raise ValueError("purge needs origins aligned with label_known_time")
    lkt = _to_ns(label_known_time)
    cut = _to_ns(origins)[first_val]
    return train[lkt[train] <= cut]


def inner_time_splits(
    n: int,
    *,
    gap: int,
    n_splits: int = 3,
    val_fraction: float = 0.25,
    min_train: int = 30,
    label_known_time: pd.Series | None = None,
    origins: pd.Index | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding walk-forward inner splits over ``n`` time-ordered rows (research spec §7).

    The last ``val_fraction`` of rows is cut into ``n_splits`` contiguous validation blocks. For a
    block starting at row ``b0`` the training rows are ``[0, b0 - gap)``: with ``gap >= horizon``
    no training label (known at row ``i + horizon``) is unknown at the block's first origin.
    If ``label_known_time`` (positional, with ``origins``) is given, training rows whose label is
    known after the block's first origin are **also** purged; this is what makes T4/T5 (labels
    known after the delivery month) safe. Never shuffled.

    Raises:
        InsufficientDataError: if no split leaves at least ``min_train`` training rows.
    """
    if gap < 0:
        raise ValueError("gap must be >= 0")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    start = int(np.floor(n * (1.0 - val_fraction)))
    blocks = [b for b in np.array_split(np.arange(start, n), n_splits) if len(b)]
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        train_stop = int(block[0]) - gap
        if train_stop <= 0:
            continue
        train = purge_by_label_time(np.arange(0, train_stop), int(block[0]), label_known_time,
                                    origins)
        if len(train) >= min_train:
            splits.append((train, block))
    if not splits:
        raise InsufficientDataError(
            f"inner_time_splits: n={n}, gap={gap}, val_fraction={val_fraction} leaves no split "
            f"with >= {min_train} training rows"
        )
    return splits


def tail_split(n: int, *, fraction: float, gap: int, min_head: int = 30,
               min_tail: int = 20, label_known_time: pd.Series | None = None,
               origins: pd.Index | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Split ``n`` rows into a head and a held-out tail (last ``fraction``), with ``gap`` rows
    dropped between them and, if ``label_known_time`` is given, every head row whose label is
    known after the tail's first origin also dropped. Used for early stopping, conformal and
    probability calibration."""
    n_tail = int(np.ceil(n * fraction))
    head_stop = n - n_tail - gap
    head = np.arange(0, max(head_stop, 0))
    if 1 <= n_tail < n and len(head):
        head = purge_by_label_time(head, n - n_tail, label_known_time, origins)
    if n_tail < min_tail or len(head) < min_head:
        raise InsufficientDataError(
            f"tail_split: n={n}, fraction={fraction}, gap={gap} gives head={len(head)} (after "
            f"purge), tail={n_tail}; need >= {min_head} and >= {min_tail}"
        )
    return head, np.arange(n - n_tail, n)


# ---------------------------------------------------------------------------------------------
# Train-only preprocessing (impute -> standardise -> optional per-family PCA)
# ---------------------------------------------------------------------------------------------


class TabularPreprocessor:
    """Median imputation, standardisation and optional per-family PCA, all fitted on train only.

    * Imputation uses ``SimpleImputer(strategy="median", keep_empty_features=True)``: a column that
      is all-NaN in the training window is **kept** (imputed to 0) so that the output column count
      and the coefficient↔column mapping never change between walk-forward windows.
    * Standardisation: ``StandardScaler`` (zero-variance columns get scale 1).
    * ``family_pca={"k": 3, "families": None | [...]}``: for each family (``<family>__*`` prefix)
      with more than ``k`` columns, replace its columns by its first ``k`` principal components,
      fitted on the standardised training rows. Families with ``<= k`` columns pass through.
      Output names ``pca__<family>__pc<i>``.

    Args:
        impute: apply median imputation (must be True if ``standardize`` or PCA is used and X can
            contain NaN).
        standardize: z-score columns with training moments.
        family_pca: ``None`` or a mapping with ``k`` (int) and optional ``families`` (list).
    """

    def __init__(self, *, impute: bool = True, standardize: bool = True,
                 family_pca: dict[str, Any] | None = None) -> None:
        self.impute = impute
        self.standardize = standardize
        self.family_pca = dict(family_pca) if family_pca else None
        if self.family_pca is not None:
            k = int(self.family_pca.get("k", 0))
            if k < 1:
                raise ValueError("family_pca.k must be >= 1")
            self.family_pca["k"] = k

    def fit(self, X: pd.DataFrame) -> TabularPreprocessor:
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        self.columns_in_ = list(X.columns)
        arr = X.to_numpy(dtype=float)
        if self.impute:
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            arr = self.imputer_.fit_transform(arr)
        if self.standardize:
            self.scaler_ = StandardScaler()
            arr = self.scaler_.fit_transform(arr)
        self.pca_: dict[str, tuple[list[int], Any]] = {}
        if self.family_pca is not None:
            if np.isnan(arr).any():
                raise ModelError("family PCA needs imputed inputs; set impute=True")
            k = self.family_pca["k"]
            wanted = self.family_pca.get("families")
            groups: dict[str, list[int]] = {}
            for i, c in enumerate(self.columns_in_):
                groups.setdefault(feature_family(c), []).append(i)
            for fam, idx in groups.items():
                if (wanted is None or fam in wanted) and len(idx) > k:
                    n_comp = min(k, arr.shape[0])
                    pca = PCA(n_components=n_comp, svd_solver="full")
                    pca.fit(arr[:, idx])
                    self.pca_[fam] = (idx, pca)
        self.output_names_ = self._output_names()
        return self

    def _output_names(self) -> list[str]:
        reduced = {i for idx, _ in self.pca_.values() for i in idx}
        names = [c for i, c in enumerate(self.columns_in_) if i not in reduced]
        for fam, (_, pca) in self.pca_.items():
            names += [f"pca{FAMILY_SEP}{fam}{FAMILY_SEP}pc{j + 1}" for j in range(pca.n_components_)]
        return names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "columns_in_"):
            raise NotFittedError("TabularPreprocessor.transform before fit")
        arr = X[self.columns_in_].to_numpy(dtype=float)
        if self.impute:
            arr = self.imputer_.transform(arr)
        if self.standardize:
            arr = self.scaler_.transform(arr)
        if not self.pca_:
            return arr
        reduced = {i for idx, _ in self.pca_.values() for i in idx}
        keep = [i for i in range(arr.shape[1]) if i not in reduced]
        parts = [arr[:, keep]]
        for idx, pca in self.pca_.values():
            parts.append(pca.transform(arr[:, idx]))
        return np.hstack(parts)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------------------------

_COMMON_PARAMS: dict[str, Any] = {"horizon": None, "seed": 20261016}
"""``horizon`` has no default: a forgotten horizon would silently under-purge h=5/20 targets."""


def _params_hash(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


class ForecastModel(ABC):
    """Base class of every forecasting model.

    Class attributes:
        name: registry key (also the ``configs/models.yaml`` key).
        level: model-ladder level 0–6 (research spec).
        task: ``regression`` | ``classification`` | ``volatility``.
        family_requirements: feature families the model *requires* (e.g. ``("storage",)``);
            empty means "any feature set".
        default_params: every accepted hyperparameter with its default. ``seed`` and the
            **mandatory** ``horizon`` (trading days; drives inner-CV gaps and HAC lags; no
            default) are accepted by all models.
    """

    name: ClassVar[str] = "base"
    level: ClassVar[int] = -1
    task: ClassVar[str] = "regression"
    family_requirements: ClassVar[tuple[str, ...]] = ()
    default_params: ClassVar[dict[str, Any]] = {}
    supports_quantiles: ClassVar[bool] = False
    supports_proba: ClassVar[bool] = False

    def __init__(self, **params: Any) -> None:
        allowed = set(self.default_params) | set(_COMMON_PARAMS)
        unknown = sorted(set(params) - allowed)
        if unknown:
            raise TypeError(
                f"{type(self).__name__}: unknown parameter(s) {unknown}; allowed {sorted(allowed)}"
            )
        merged = copy.deepcopy({**_COMMON_PARAMS, **self.default_params})
        merged.update(copy.deepcopy(params))
        self.params: dict[str, Any] = merged
        if self.params["horizon"] is None:
            raise TypeError(
                f"{type(self).__name__}: 'horizon' (trading days) is required; pass "
                "make_model(..., horizon=h) or set it in the experiment's model params. There is "
                "no default because a wrong horizon silently under-purges inner CV and HAC"
            )
        if int(self.params["horizon"]) < 1:
            raise ValueError("horizon must be >= 1 trading day")
        self.params["horizon"] = int(self.params["horizon"])
        self._validate_params()
        self.is_fitted: bool = False

    # -- identity --------------------------------------------------------------------------
    @property
    def model_id(self) -> str:
        """``<name>-<8-hex hash of params>``: stable across runs for identical parameters."""
        return f"{self.name}-{_params_hash(self.params)}"

    @property
    def horizon(self) -> int:
        return int(self.params["horizon"])

    @property
    def seed(self) -> int:
        return int(self.params["seed"])

    def required_columns(self) -> list[str]:
        """Columns that must be present in ``X`` (beyond whatever the model uses generically)."""
        return []

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params})"

    # -- hooks -----------------------------------------------------------------------------
    def _validate_params(self) -> None:
        """Override to validate ``self.params``; raise ValueError with a clear message."""

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def _predict_quantiles(self, X: pd.DataFrame, levels: Sequence[float]) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not produce quantile forecasts")

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not produce probabilities")

    # -- public API ------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series,
            label_known_time: pd.Series | None = None) -> ForecastModel:
        """Fit on time-ordered training rows. Returns ``self``.

        Args:
            X: features indexed by tz-aware origin times, one row per trading day.
            y: target aligned on ``X.index``.
            label_known_time: optional tz-aware Series aligned on ``X.index`` (TARGET_SCHEMA's
                ``label_known_time``). When given, every internal split (inner CV, early-stopping,
                conformal and calibration tails) also purges training rows whose label is known
                after the first origin of the held-out slice. Without it only the row gap of
                ``horizon`` rows protects those splits, so the grid must be one row per trading
                day (checked: several rows per day or weekend origins are refused).
        """
        X, y = self._check_fit_inputs(X, y)
        self.fit_label_known_time_: pd.Series | None = _check_label_known_time(
            label_known_time, X.index)
        _check_trading_day_grid(X.index, strict=self.fit_label_known_time_ is None)
        self.feature_names_in_: list[str] = list(X.columns)
        self.n_train_: int = len(X)
        self.train_start_: Any = X.index[0]
        self.train_end_: Any = X.index[-1]
        self._fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Point forecast (classifiers: probability of the event), indexed like ``X``."""
        X = self._check_predict_inputs(X)
        out = np.asarray(self._predict(X), dtype=float).reshape(-1)
        return pd.Series(out, index=X.index, name=self.model_id, dtype=float)

    def predict_quantiles(self, X: pd.DataFrame, levels: Sequence[float]) -> pd.DataFrame:
        """Quantile forecasts; columns ``q<level>`` in increasing level order, non-crossing."""
        if not self.supports_quantiles:
            raise NotImplementedError(f"{type(self).__name__} does not produce quantile forecasts")
        lv = sorted(float(v) for v in levels)
        cols = [quantile_column(v) for v in lv]
        if len(set(cols)) != len(cols):
            raise ValueError(f"duplicate quantile levels {levels}")
        X = self._check_predict_inputs(X)
        vals = rearrange_quantiles(self._predict_quantiles(X, lv))
        return pd.DataFrame(vals, index=X.index, columns=cols)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Probability of the positive event, in [0, 1]."""
        if not self.supports_proba:
            raise NotImplementedError(f"{type(self).__name__} does not produce probabilities")
        X = self._check_predict_inputs(X)
        p = np.clip(np.asarray(self._predict_proba(X), dtype=float).reshape(-1), 0.0, 1.0)
        return pd.Series(p, index=X.index, name=self.model_id, dtype=float)

    # -- validation ------------------------------------------------------------------------
    def _check_fit_inputs(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame indexed by origin_time")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series indexed like X")
        if len(X) == 0:
            raise InsufficientDataError("fit() received no rows")
        if not X.index.equals(y.index):
            raise ValueError("X and y must share the same index (origin_time), in the same order")
        _check_time_index(X.index, "training")
        if y.isna().any():
            raise ValueError(
                f"y has {int(y.isna().sum())} NaN values; drop NaN targets before fit "
                "(target_definitions §0) and report the count"
            )
        if not np.isfinite(y.to_numpy(dtype=float)).all():
            raise ValueError("y contains inf")
        if np.isinf(X.to_numpy(dtype=float)).any():
            raise ValueError("X contains inf; NaN is allowed, inf is not")
        missing = [c for c in self.required_columns() if c not in X.columns]
        if missing:
            raise KeyError(f"{type(self).__name__}: required column(s) {missing} not in X")
        if self.task == "classification":
            vals = set(np.unique(y.to_numpy()))
            if not vals <= {0, 1, 0.0, 1.0, True, False}:
                raise ValueError(f"classification target must be binary 0/1, got {sorted(vals)}")
        return X, y.astype(float)

    def _check_predict_inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__}.predict called before fit")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        missing = [c for c in self.feature_names_in_ if c not in X.columns]
        if missing:
            raise KeyError(f"{type(self).__name__}: X lacks training column(s) {missing}")
        if np.isinf(X[self.feature_names_in_].to_numpy(dtype=float)).any():
            raise ValueError("X contains inf")
        return X[self.feature_names_in_]


    # -- helpers for subclasses: purge-aware internal splits ------------------------------
    def _lkt_for(self, X: pd.DataFrame) -> pd.Series | None:
        """The fit's ``label_known_time`` restricted to ``X``'s rows (None if not given)."""
        lkt = getattr(self, "fit_label_known_time_", None)
        return None if lkt is None else lkt.reindex(X.index)

    def _tail_split(self, X: pd.DataFrame, *, fraction: float, **kw: Any
                    ) -> tuple[np.ndarray, np.ndarray]:
        """:func:`tail_split` over ``X``'s rows with gap = horizon and label-time purge."""
        return tail_split(len(X), fraction=fraction, gap=self.horizon,
                          label_known_time=self._lkt_for(X), origins=X.index, **kw)


def _check_label_known_time(lkt: pd.Series | None, index: pd.Index) -> pd.Series | None:
    if lkt is None:
        return None
    if not isinstance(lkt, pd.Series):
        raise TypeError("label_known_time must be a pandas Series aligned on X.index")
    if not lkt.index.equals(index):
        raise ValueError("label_known_time must share X's index (origin_time), in the same order")
    if lkt.isna().any():
        raise ValueError(f"label_known_time has {int(lkt.isna().sum())} NaN values")
    if not isinstance(lkt.dtype, pd.DatetimeTZDtype):
        raise TypeError("label_known_time must be tz-aware datetimes")
    if (_to_ns(lkt) < _to_ns(index)).any():
        raise ValueError("label_known_time earlier than its origin: a label cannot be known "
                         "before the forecast is made")
    return lkt


def _check_trading_day_grid(index: pd.Index, *, strict: bool) -> None:
    """Refuse (``strict``) or warn about grids on which a row gap is not a trading-day gap.

    Detects more than one row per local market date (median business-day spacing < 1, or > 5 %
    same-day steps) and calendar-day grids (origins on Saturday/Sunday). Holiday gaps are fine:
    they only make a row gap more conservative.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("X must be indexed by a DatetimeIndex of origin times")
    if len(index) < 3:
        return
    local = index.tz_convert(MARKET_TZ).tz_localize(None) if index.tz is not None else index
    days = np.asarray(local.normalize(), dtype="datetime64[D]")
    problems = []
    spacing = np.busday_count(days[:-1], days[1:])
    if np.median(spacing) < 1 or (spacing == 0).mean() > 0.05:
        problems.append("more than one row per trading day (business-day spacing < 1)")
    weekend = ~np.is_busday(days)
    if weekend.any():
        problems.append(f"{int(weekend.sum())} origins fall on weekends (calendar-day grid?)")
    if not problems:
        return
    msg = ("origin grid is not one row per trading day: " + "; ".join(problems)
           + ". A gap of `horizon` rows would not be `horizon` trading days")
    if strict:
        raise ValueError(msg + "; pass label_known_time to fit() or resample to trading days")
    warnings.warn(msg + "; label_known_time purging is applied", stacklevel=3)


def _check_time_index(index: pd.Index, what: str) -> None:
    if not index.is_unique:
        raise ValueError(f"{what} index has duplicate origin times")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{what} index must be time-ordered (monotone increasing); never shuffle")


def origin_months(index: pd.Index) -> np.ndarray:
    """Calendar month (1–12) of each origin, in market time for tz-aware indexes."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("a DatetimeIndex of origin times is required")
    if index.tz is not None:
        index = index.tz_convert(MARKET_TZ)
    return np.asarray(index.month)


# ---------------------------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[ForecastModel]] = {}
M = TypeVar("M", bound=type[ForecastModel])


def register_model(cls: M) -> M:
    """Class decorator: add ``cls`` to :data:`MODEL_REGISTRY` under ``cls.name``."""
    if cls.name in MODEL_REGISTRY and MODEL_REGISTRY[cls.name] is not cls:
        raise ValueError(f"model name {cls.name!r} registered twice")
    if cls.task not in TASKS:
        raise ValueError(f"{cls.__name__}.task {cls.task!r} not in {TASKS}")
    if not 0 <= cls.level <= 6:
        raise ValueError(f"{cls.__name__}.level must be 0..6")
    MODEL_REGISTRY[cls.name] = cls
    return cls


_MODEL_MODULES = ("baselines", "linear", "boosting", "probabilistic", "ensemble")


def _ensure_registered() -> None:
    import importlib

    for mod in _MODEL_MODULES:
        importlib.import_module(f"ttf_forecaster.models.{mod}")


def available_models() -> dict[str, type[ForecastModel]]:
    """All registered model classes by name (imports every model module)."""
    _ensure_registered()
    return dict(MODEL_REGISTRY)


def model_config(config_dir: str | None = None) -> dict[str, Any]:
    """The ``models`` mapping from ``configs/models.yaml`` (empty if absent from the file)."""
    cfg = load_config("models", config_dir)
    models = cfg.get("models", {})
    if not isinstance(models, dict):
        raise ConfigError("configs/models.yaml: 'models' must be a mapping")
    return models


def make_model(name: str, *, config_dir: str | None = None, **params: Any) -> ForecastModel:
    """Build a model by config entry or registry name.

    Resolution: if ``name`` is a key under ``models:`` in ``configs/models.yaml`` its ``model``
    field (default: ``name``) selects the class and its ``params`` are the defaults; caller
    ``params`` override them. Otherwise ``name`` must be a registered class name and class
    defaults apply. Unknown parameters raise ``TypeError``.
    """
    _ensure_registered()
    entries = model_config(config_dir)
    entry = entries.get(name, {}) or {}
    cls_name = entry.get("model", name)
    if cls_name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; config entries {sorted(entries)}, "
            f"registered classes {sorted(MODEL_REGISTRY)}"
        )
    defaults = load_config("models", config_dir).get("defaults", {}) or {}
    bad = sorted(set(defaults) - {"seed"})
    if bad:
        raise ConfigError(f"configs/models.yaml defaults may only set 'seed', got {bad}; "
                          "horizon must be explicit per experiment")
    merged = {**copy.deepcopy(defaults), **copy.deepcopy(entry.get("params", {}) or {}), **params}
    return MODEL_REGISTRY[cls_name](**merged)


# ---------------------------------------------------------------------------------------------
# PREDICTION_SCHEMA helper
# ---------------------------------------------------------------------------------------------


def to_prediction_frame(
    point: pd.Series,
    *,
    target_id: str,
    horizon: int,
    model_id: str,
    train_end: pd.Timestamp,
    label_known_time: pd.Series,
    fold_id: str,
    feature_hash: str,
    is_synthetic: bool,
    quantiles: pd.DataFrame | None = None,
    prob: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble PREDICTION_SCHEMA rows (plus optional ``q<level>`` and ``prob`` columns).

    ``point`` must be indexed by tz-aware UTC origin times; ``label_known_time`` aligned on it.
    """
    idx = point.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
        raise TypeError("prediction index must be tz-aware origin times")
    lkt = label_known_time.reindex(idx)
    if lkt.isna().any():
        raise ValueError("label_known_time missing for some prediction rows")
    te = pd.Timestamp(train_end)
    te = te.tz_convert("UTC") if te.tzinfo else te.tz_localize("UTC")
    if (idx <= te).any():
        raise LeakageGuardError("prediction origins must be strictly after train_end")
    out = pd.DataFrame(
        {
            "origin_time": idx.tz_convert("UTC"),
            "target_id": target_id,
            "horizon": np.full(len(idx), int(horizon), dtype="int64"),
            "model_id": model_id,
            "prediction": point.to_numpy(dtype=float),
            "train_end": pd.DatetimeIndex([te] * len(idx)),
            "label_known_time": pd.DatetimeIndex(lkt).tz_convert("UTC"),
            "fold_id": fold_id,
            "feature_hash": feature_hash,
            "is_synthetic": np.full(len(idx), bool(is_synthetic)),
        }
    )
    if quantiles is not None:
        for c in quantiles.columns:
            out[c] = quantiles[c].reindex(idx).to_numpy(dtype=float)
    if prob is not None:
        out["prob"] = prob.reindex(idx).to_numpy(dtype=float)
    return out.reset_index(drop=True)


def grid_search_time_cv(
    X: pd.DataFrame,
    y: pd.Series,
    grid: Iterable[Any],
    fit_score: Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Any], float],
    *,
    gap: int,
    n_splits: int,
    val_fraction: float,
    min_train: int = 30,
    label_known_time: pd.Series | None = None,
    rule: str = "min",
) -> tuple[Any, pd.DataFrame]:
    """Pick the grid value by validation loss over :func:`inner_time_splits`.

    ``fit_score(X_tr, y_tr, X_va, y_va, value)`` must fit everything (preprocessing included) on
    ``X_tr`` only and return a loss. Ties go to the earlier grid value (order the grid from most
    to least regularised so ties favour the simpler model).

    ``rule`` (D-032 F5):
      * ``"min"``: the grid value with the lowest mean loss (can pick a near-OLS fit on short windows).
      * ``"one_se"``: the most regularised grid value whose mean loss is within one standard error
        of the minimum, with se = std over inner folds / sqrt(n folds). Prefers the simpler model
        when the inner CV cannot tell the candidates apart.

    Returns:
        (best value, frame of mean/std loss per grid value).
    """
    if rule not in ("min", "one_se"):
        raise ValueError(f"rule must be 'min' or 'one_se', got {rule!r}")
    lkt = None if label_known_time is None else label_known_time.reindex(X.index)
    splits = inner_time_splits(len(X), gap=gap, n_splits=n_splits, val_fraction=val_fraction,
                               min_train=min_train, label_known_time=lkt, origins=X.index)
    values = list(grid)
    rows = []
    for v in values:
        losses = [
            fit_score(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va], v) for tr, va in splits
        ]
        rows.append({"value": v, "mean_loss": float(np.mean(losses)),
                     "std_loss": float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
                     "n_folds": len(losses)})
    table = pd.DataFrame(rows)
    means = table["mean_loss"].to_numpy()
    i_min = int(np.argmin(means))
    if rule == "min":
        return values[i_min], table
    se = float(table["std_loss"].iloc[i_min]) / np.sqrt(max(len(splits), 1))
    threshold = float(means[i_min]) + se
    # grid is ordered most -> least regularised, so the first qualifying value is the simplest
    i_1se = int(np.flatnonzero(means <= threshold)[0])
    table["within_1se"] = means <= threshold
    return values[i_1se], table
