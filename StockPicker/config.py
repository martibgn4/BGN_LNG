"""
============================================================
config.py  --  Central configuration for the stock picker
============================================================
Every tunable lives here so the research loop and the live
weekly run cannot drift apart.
============================================================
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
UNIVERSE_DIR = CACHE / "universe"
PRICES_DIR = CACHE / "prices"
FUNDA_DIR = CACHE / "fundamentals"
REPORTS_DIR = ROOT / "reports"

for _d in (CACHE, UNIVERSE_DIR, PRICES_DIR, FUNDA_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# universe
# ----------------------------------------------------------------------
INDEX_TICKER = "SPX Index"          # S&P 500
BENCHMARK_TICKER = "SPY US Equity"  # tradable benchmark for backtest comparison

# History start. 2006 gives us the GFC, the 2018 vol shock, COVID, the 2022
# bear and the 2023-25 concentration regime -- enough distinct regimes that a
# model surviving all of them is unlikely to be a pure curve fit.
HISTORY_START = dt.date(2006, 1, 1)

# Membership is resampled monthly. S&P 500 changes are infrequent enough that
# month-end snapshots capture essentially every add/drop, at 1/20th the data cost
# of daily snapshots.
MEMBERSHIP_FREQ = "ME"

# ----------------------------------------------------------------------
# Bloomberg fields
# ----------------------------------------------------------------------
# Daily price/volume series. Split- and dividend-adjusted via the BDH
# adjustment flags so momentum is not corrupted by corporate actions.
PRICE_FIELDS = [
    "PX_LAST",
    "PX_OPEN",
    "PX_HIGH",
    "PX_LOW",
    "PX_VOLUME",
    "EQY_WEIGHTED_AVG_PX",   # VWAP -- realistic weekly execution price
    "CUR_MKT_CAP",
]

# Slower-moving descriptors, sampled weekly. Each was verified to return data
# on the Desktop API entitlement on this machine.
SLOW_FIELDS = [
    "PE_RATIO",
    "PX_TO_BOOK_RATIO",
    "EV_TO_T12M_EBITDA",
    "FREE_CASH_FLOW_YIELD",
    "RETURN_COM_EQY",
    "GROSS_MARGIN",
    "TRAIL_12M_EPS",
    "BEST_EPS",                 # consensus forward EPS -> revisions signal
    "BEST_TARGET_PRICE",        # consensus target -> implied upside
    "BEST_ANALYST_RATING",      # consensus rating -> rating drift
    "TOT_ANALYST_REC",
    "SHORT_INT_RATIO",          # days-to-cover -> crowding / squeeze
    "NEWS_SENTIMENT_DAILY_AVG", # Bloomberg NLP news sentiment
    "VOLATILITY_90D",
    "BETA_ADJ_OVERRIDABLE",
    "EQY_SH_OUT",
]

# Options-implied fields, sampled weekly.
#
# These are the closest thing available to a view of what informed,
# leveraged money is paying to hedge. Verified on this terminal to carry
# full weekly history back to 2015 (shorter than the price history, so the
# options block is only usable from ~2015 onward).
#
#   skew        IV(90% moneyness) - IV(110%) : the price of downside
#               protection relative to upside. Rising skew means someone is
#               paying up for puts.
#   vrp         implied vol - realised vol : how much fear is already in the
#               price. A high VRP means the market has already noticed.
#   put/call    positioning, in both volume and open interest.
OPTIONS_FIELDS = [
    "30DAY_IMPVOL_100.0%MNY_DF",
    "30DAY_IMPVOL_90.0%MNY_DF",
    "30DAY_IMPVOL_110.0%MNY_DF",
    "PUT_CALL_VOLUME_RATIO_CUR_DAY",
    "PUT_CALL_OPEN_INTEREST_RATIO",
    "VOLATILITY_30D",
    "OPEN_INT_TOTAL_CALL",
    "OPEN_INT_TOTAL_PUT",
]

OPTIONS_DIR = CACHE / "options"
OPTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Analyst estimates, pulled SEPARATELY because they need an override.
#
# Requested plainly, ``BEST_EPS`` returns only ~141 weekly points starting in
# 2021 on this terminal -- Bloomberg defaults to the current fiscal period,
# which has almost no usable history. With BEST_FPERIOD_OVERRIDE = "1BF"
# ("1 period best forward") the same field returns 1,078 weekly points back to
# 2006-01-06, i.e. the full sample.
#
# This matters more than it looks: estimate revisions are among the strongest
# and best-documented equity signals, and without the override the entire
# revisions family is silently empty before 2021.
ESTIMATE_FIELDS = ["BEST_EPS", "BEST_SALES"]
ESTIMATE_OVERRIDES = {"BEST_FPERIOD_OVERRIDE": "1BF"}

ESTIMATES_DIR = CACHE / "estimates"
ESTIMATES_DIR.mkdir(parents=True, exist_ok=True)

# Fields that Bloomberg only prints on reporting dates (~4 points a year via
# BDH). They are forward-filled before use: a quarterly ROE remains the latest
# publicly known value until the next report, so carrying it forward is
# correct, whereas leaving it NaN discards the entire quality/value block.
REPORTED_SPARSE_FIELDS = [
    "RETURN_COM_EQY",
    "GROSS_MARGIN",
    "TRAIL_12M_EPS",
    "EV_TO_T12M_EBITDA",
    "SHORT_INT_RATIO",
    "EQY_SH_OUT",
]
# Cap the carry so a discontinued field cannot propagate indefinitely.
FFILL_LIMIT_WEEKS = 26

# Static descriptors pulled once per name.
#
# GICS is the industry standard but Bloomberg does not populate it for dead
# tickers, which are exactly the names that make the backtest honest. BICS
# level 1 *is* populated for delisted names (verified on this terminal), so it
# is the primary sector key and GICS is kept for cross-checking live names.
STATIC_FIELDS = [
    "NAME",
    "BICS_LEVEL_1_SECTOR_NAME",  # primary -- full coverage incl. dead tickers
    "INDUSTRY_SECTOR",           # Bloomberg legacy sector, also dead-safe
    "INDUSTRY_GROUP",
    "INDUSTRY_SUBGROUP",
    "GICS_SECTOR_NAME",          # live names only
    "GICS_INDUSTRY_NAME",
    "ID_ISIN",
]

SECTOR_COL = "BICS_LEVEL_1_SECTOR_NAME"

# BDH returns CUR_MKT_CAP in millions while BDP returns absolute units.
# Everything downstream works in absolute USD, so historical pulls are scaled.
MKT_CAP_BDH_SCALE = 1e6

# ----------------------------------------------------------------------
# strategy parameters
# ----------------------------------------------------------------------


@dataclass
class StrategyConfig:
    """Portfolio construction and rebalancing rules."""

    n_positions: int = 5
    rebalance_day: str = "W-MON"     # weekly, Monday open

    # Liquidity floor. At <$1m AUM a 5-name book puts ~$200k per name; a $20m
    # ADV floor keeps us under ~1% of daily volume, so market impact is noise.
    min_dollar_volume: float = 20e6
    min_price: float = 5.0           # avoid penny-stock microstructure
    min_market_cap: float = 2e9

    # Risk weighting. Inverse-volatility rather than equal-weight, so a
    # 60%-vol name does not silently dominate portfolio risk.
    weighting: str = "inverse_vol"   # {equal, inverse_vol, inverse_var}
    max_weight: float = 0.35         # cap any single name
    min_weight: float = 0.05
    max_per_sector: int = 2          # diversification guard on 5 names

    # Costs. Personal scale, liquid large caps: commission + half-spread.
    # 8bps round-trip is conservative for retail on S&P 500 names.
    cost_bps: float = 8.0

    # Volatility targeting overlay -- scales gross exposure so realised
    # portfolio vol tracks a target, cash is the residual.
    # Targeting 18% against a benchmark that realises ~19% meant the overlay
    # was permanently de-risking -- average exposure 84%, so ~16% of the
    # equity risk premium was given up in all weather for protection that is
    # only wanted in genuine stress. Set just above benchmark vol so it binds
    # in crises and is otherwise inert.
    vol_target: float | None = 0.22  # annualised; None disables
    vol_lookback: int = 60
    max_leverage: float = 1.0        # long-only, no leverage

    # Turnover control -- a rank buffer.
    #
    # Buying requires a top-N rank; selling only happens once a held name
    # falls below rank N * hold_rank_multiple. The gap between those two
    # thresholds is what stops the book churning every week between names
    # whose score difference is statistically indistinguishable.
    #
    # At n=5 and multiple=4 a name is bought in the top 5 and held to rank 20.
    # Measured at multiple=4 the book still turned over 65% a week -- 34x a
    # year, costing 2.9%/yr, which was larger than the entire selection edge.
    # Widening the hold band is the highest-value single lever available:
    # a name bought in the top 5 is now held until it leaves the top 40.
    hold_rank_multiple: int = 8
    replacement_buffer: float = 0.15  # retained for the legacy score-gap path


@dataclass
class ModelConfig:
    """Signal combination and validation."""

    # Walk-forward: train on an expanding window, predict the next block,
    # never look forward. Retrain cadence in weeks.
    min_train_weeks: int = 156       # 3y before first prediction
    retrain_every_weeks: int = 13    # quarterly

    # Forward return horizon the model is trained to rank.
    forward_weeks: int = 1

    # Cross-sectional preprocessing
    winsorize_pct: float = 0.02
    neutralize_sector: bool = True   # rank within sector, not vs whole market
    neutralize_beta: bool = False

    blend: str = "ic_weighted"       # {equal, ic_weighted, lightgbm}
    # IC estimates are expanding-window with a long EWM blended in. A short
    # window was tried first and failed: at an IC information ratio of ~0.1,
    # a single year of observations cannot distinguish a real factor from
    # noise, and the weights came out uncorrelated with the actual ICs.
    ic_halflife_weeks: int = 260

    lgbm_params: dict = field(
        default_factory=lambda: {
            "objective": "regression",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 100,
            "subsample": 0.7,
            "subsample_freq": 1,
            "colsample_bytree": 0.6,
            "reg_lambda": 5.0,
            "verbose": -1,
        }
    )


STRATEGY = StrategyConfig()
MODEL = ModelConfig()
