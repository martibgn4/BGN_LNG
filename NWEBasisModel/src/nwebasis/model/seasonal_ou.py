"""Seasonal Ornstein-Uhlenbeck fit for the NWE basis.

    d b_t = kappa * (theta(m) - b_t) dt + sigma dW_t

where ``theta(m)`` is the long-run level for delivery month m. Estimated by
exact discretisation: the OU has an exact AR(1) representation at any fixed
step, so a linear regression of b_t on b_{t-1} recovers kappa and sigma without
a numerical optimiser. On daily data that is not an approximation.

    b_t = c + phi * b_{t-1} + e_t
    kappa = -ln(phi) / dt
    sigma = sd(e) * sqrt(2 * kappa / (1 - phi^2))
    theta = c / (1 - phi)

WHY THIS EXISTS SEPARATELY FROM THE ECM. The ECM forecasts the conditional
mean. This produces the distribution, and specifically the three parameters
``adhoc_scripts/NWE_option_pricer.py`` currently sets by hand. Fitted on
2023-07-01..2026-09-03 M+1 the estimates are kappa 4.10, sigma 0.535,
theta -0.371 against the pricer's 3.0, 0.35 and -0.505 - a faster, wider,
narrower-centred process than assumed. That matters for a cancellation option,
whose value is mostly vega and mostly in the tails.

WHAT THIS CANNOT DO. A single OU cannot represent the 2022 regime, where the
basis reached -10.52 and mean-reverted at a different speed on the way out.
Fitting one process to both regimes gives a sigma that is too wide for today
and a theta that is too wide for any market that has existed since. Hence
``fit`` takes an explicit sample start and refuses to silently span the break.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from nwebasis.config import get


class InsufficientHistory(RuntimeError):
    """Too few observations to estimate a mean-reverting process honestly."""


@dataclass(frozen=True)
class OUParameters:
    """Annualised OU parameters, plus what they were fitted on."""

    tenor: str
    n_observations: int
    sample_start: str
    sample_end: str
    phi: float
    kappa_annual: float
    sigma_annual: float
    theta: float
    half_life_days: float
    residual_sd_daily: float

    def as_pricer_block(self) -> str:
        """The parameter block in the form NWE_option_pricer.py expects."""
        return (
            f"# fitted on {self.sample_start}..{self.sample_end}, "
            f"{self.tenor}, n={self.n_observations}\n"
            f"NWE_MEAN = {self.theta:.3f}\n"
            f"NWE_VOL = {self.sigma_annual:.3f}\n"
            f"NWE_KAPPA = {self.kappa_annual:.2f}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def deseasonalise(frame: pd.DataFrame, value_column: str = "value"
                  ) -> tuple[pd.Series, pd.Series]:
    """Strip the delivery-month mean off the basis.

    Returns the de-seasonalised series and the twelve monthly offsets. The
    offsets are centred so they sum to zero, which keeps ``theta`` on the same
    scale as the raw basis instead of absorbing the seasonal mean.
    """
    monthly = frame.groupby("delivery_month")[value_column].mean()
    monthly = monthly - monthly.mean()
    offsets = frame["delivery_month"].map(monthly)
    return frame[value_column] - offsets, monthly


def fit_ou(series: pd.Series, tenor: str, sample_start: str, sample_end: str,
           min_observations: int | None = None) -> OUParameters:
    """Exact-discretisation AR(1) fit. No optimiser, no starting values."""
    values = series.dropna().to_numpy(dtype=float)
    floor = min_observations or int(get("model", "ou", "min_observations"))
    if len(values) < floor:
        raise InsufficientHistory(
            f"{tenor}: {len(values)} observations, need {floor}. Refusing to "
            "report a mean-reversion speed that the sample cannot support."
        )

    previous, current = values[:-1], values[1:]
    design = np.column_stack([np.ones_like(previous), previous])
    coefficients, *_ = np.linalg.lstsq(design, current, rcond=None)
    intercept, phi = float(coefficients[0]), float(coefficients[1])

    if not 0 < phi < 1:
        raise InsufficientHistory(
            f"{tenor}: AR(1) coefficient {phi:.4f} is outside (0, 1), so the "
            "series is not mean-reverting on this sample and an OU fit would "
            "report a meaningless half-life."
        )

    half_life = np.log(2) / -np.log(phi)
    ceiling = float(get("model", "ou", "max_half_life_days"))
    if half_life > ceiling:
        raise InsufficientHistory(
            f"{tenor}: fitted half-life {half_life:.0f} days exceeds the "
            f"{ceiling:.0f}-day ceiling, so this series is not distinguishable "
            "from a random walk on a sample of this length. Reporting a kappa "
            "here would put a confident mean-reversion speed on something that "
            "does not visibly mean-revert."
        )

    residuals = current - design @ coefficients
    periods = int(get("model", "ou", "business_days_per_year"))
    step = 1 / periods
    kappa = -np.log(phi) / step
    residual_sd = float(residuals.std(ddof=2))
    sigma = residual_sd * np.sqrt(2 * kappa / (1 - phi**2))
    theta = intercept / (1 - phi)

    return OUParameters(
        tenor=tenor,
        n_observations=len(values),
        sample_start=sample_start,
        sample_end=sample_end,
        phi=phi,
        kappa_annual=float(kappa),
        sigma_annual=float(sigma),
        theta=float(theta),
        half_life_days=float(half_life),
        residual_sd_daily=residual_sd,
    )


def fit_by_tenor(panel: pd.DataFrame, sample_start: str | None = None
                 ) -> tuple[pd.DataFrame, pd.Series]:
    """Fit one OU per tenor on the de-seasonalised basis.

    Prompt tenors revert faster than deferred ones, so a single kappa across
    the curve misprices anything with optionality past the front month. The
    per-tenor table is the point.
    """
    start = sample_start or get("model", "sample", "stable_start")
    window = panel[panel["release_date"] >= pd.Timestamp(start)].copy()
    if window.empty:
        raise InsufficientHistory(f"no observations on or after {start}")

    deseasonalised, offsets = deseasonalise(window)
    window = window.assign(basis_ds=deseasonalised)

    end = str(window["release_date"].max().date())
    rows = []
    for tenor, group in window.groupby("tenor", sort=False):
        series = group.sort_values("release_date").set_index("release_date")["basis_ds"]
        try:
            rows.append(fit_ou(series, tenor, str(start), end).to_dict())
        except InsufficientHistory as exc:
            rows.append({"tenor": tenor, "n_observations": len(series),
                         "sample_start": str(start), "sample_end": end,
                         "error": str(exc)})
    table = pd.DataFrame(rows)
    return _order_by_tenor(table), offsets


def _order_by_tenor(table: pd.DataFrame) -> pd.DataFrame:
    """Sort M+1, M+2, ... numerically rather than as strings."""
    order = table["tenor"].str.extract(r"(\d+)").astype(float)[0]
    return table.assign(_order=order).sort_values("_order").drop(columns="_order")
