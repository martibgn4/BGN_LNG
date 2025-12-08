import logging
import os
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd

from general_utils import memoize
from quantity import DAYS_PER_YEAR
from thorn.core.calibration.base.contract_keys import Contract
from thorn.core.calibration.base.vol_corr_time_periods import intra_month_tenors
from thorn.core.calibration.historical_analysis.price_data import PriceData
from time_period import DateRange, RelativeDateRange, FixedDateRange, NEVER_DR, str_vault

DEBUG = False
ALMOST_ZERO = 1e-7


@memoize
def valid_tenor(tenor_str):
    """ not-so-fine-grained filtering """
    if tenor_str in intra_month_tenors:
        return True
    try:
        tenor = RelativeDateRange.parse(tenor_str)
        if isinstance(tenor, FixedDateRange):
            return tenor.date_range.duration() > 1
    except (AssertionError, ValueError):
        return False
    return False


def get_abs_contract_label(rel_contract, obs_date, number_of_days_before_rollover=0):
    """Get the absolute quoted contract label (string) given a string representing a relative contract."""
    if rel_contract in ['DA', 'WKD', 'M00', 'BOM']:
        return rel_contract
    else:
        obs_date += timedelta(number_of_days_before_rollover)
        return str_vault(RelativeDateRange.parse(rel_contract).get_absolute_date_range(obs_date))


class ReturnsCalculator:

    def __init__(self, calib_period, calib_info_dict, vc_calib_params):
        self.returns_window = vc_calib_params.returns_window[0]
        self.calib_period = calib_period
        self.calib_info_dict = calib_info_dict

        self.ext_calib_period = DateRange(self.calib_period.start - timedelta(self.returns_window),
                                          self.calib_period.end)
        self.annual_factor = np.sqrt(DAYS_PER_YEAR / self.returns_window)
        self.freq = timedelta(days=self.returns_window)
        self.normal_calibration = vc_calib_params.normal_calibration

    @staticmethod
    def _convert_to_relative_tenors(underlying, calib_info, in_df):
        series_list = []
        rel_contracts = set(
            calib_info.fitting_contracts() +
            calib_info.diagnostic_contracts +
            calib_info.additional_historical_contracts
        )
        rollover = calib_info.number_of_days_before_rollover
        dates = [(dt, dt.date()) for dt in in_df.index]

        def rel_series(lsu, tenor):
            rel_contract_key = Contract(lsu, tenor).key
            if rel_contract_key in in_df:
                return in_df[rel_contract_key]

            series = pd.Series(index=in_df.index, name=rel_contract_key)
            for dt, day in dates:
                abs_contract_key = Contract(lsu, get_abs_contract_label(tenor, day, rollover)).key
                try:
                    series[dt] = in_df[abs_contract_key][dt]
                except (KeyError, ValueError):  # pragma: no cover
                    continue
            return series

        for load_shape in calib_info.price_process_weekly_shapes:
            lsu = underlying.get_shaped(load_shape)
            for rel_contract in rel_contracts:
                series_list.append(rel_series(lsu, rel_contract.tenor))
        return pd.concat(series_list, axis=1, keys=[s.name for s in series_list])

    def _long_term_price(self, price_data, underlying):
        def _dr(tenor):
            try:
                return DateRange.parse(tenor)
            except ValueError:
                return NEVER_DR

        prices = price_data.get_price_dataframe_for_underlying(
            self.calib_info_dict[underlying].proxy_lt_underlying,
            self.calib_info_dict[self.calib_info_dict[underlying].proxy_lt_underlying].auxiliary_price_underlying,
        )
        lt_prices = []
        for i in range(5):
            # gather quotes as of calibration end date - i days
            quotes = {_dr(Contract.parse(key).tenor): price for key, price in prices.iloc[-1 - i].dropna().items()}

            lt_price_list = [(t, price) for t, price in quotes.items()]
            # Sort lexicographically, first by duration of period:
            # ('Never' first, then months, then quarters, seasons, years), ignoring length discrepancies e.g. 30/31-day months
            # secondly by start date of period
            lt_price_list_sorted = sorted(lt_price_list, key=lambda tp: ((tp[0].duration() + 3) // 10, tp[0].start))
            # Get the price of the last of the longest-duration periods
            lt_price = lt_price_list_sorted[-1][1]

            lt_prices.append(lt_price)

        # return mean over last week
        return np.mean(lt_prices)

    def _validate_output(self, rel_returns_df):
        log_missing_contracts_and_dates(rel_returns_df)
        check_window = max(self.returns_window + 5, 12)
        assert_dates_start_at_calib_window(rel_returns_df.index.date, self.calib_period, check_window)
        assert_dates_end_at_calib_window(rel_returns_df.index.date, self.calib_period, check_window)
        assert_dates_cover_most_of_calib_period(rel_returns_df.index.date, self.calib_period)

    def calculate(self, price_history):
        price_data = PriceData(price_history, self.calib_info_dict, window=self.ext_calib_period, tenor_filter=valid_tenor)

        df_list = []
        for underlying, calib_info in self.calib_info_dict.items():
            prices = price_data.get_price_dataframe_for_underlying(underlying, calib_info.auxiliary_price_underlying)

            if underlying.base_underlying in self.normal_calibration or (
                underlying.vol is not None and underlying.vol.is_normal
            ):
                log_prices = prices
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    if np.any(prices < ALMOST_ZERO):
                        log = logging.getLogger(__name__)
                        indxs = np.where(prices < ALMOST_ZERO)
                        log.warning(
                            f"Bad quotes: {underlying}, "
                            f"series id: {calib_info.series_id}, "
                            f"tenor: [" + ", ".join([
                                f"{prices.columns[contract]}:{prices.index[day].date()}"
                                for day, contract in zip(*indxs)
                            ]) + f"], value: {prices.values[indxs[0], indxs[1]].tolist()}"
                        )
                    log_prices = np.log(prices)

            # Calculate the relative returns over the time_period self.freq
            if log_prices.empty:
                rel_returns = log_prices  # If empty make sure that the df_list is not empty (or concat fails)
            else:
                abs_returns = self.annual_factor * (log_prices - log_prices.shift(freq=self.freq))
                rel_returns = self._convert_to_relative_tenors(underlying, calib_info, abs_returns)
                if underlying.base_underlying in self.normal_calibration:
                    rel_returns /= self._long_term_price(price_data, underlying)
            df_list.append(rel_returns)

        rel_returns_df = pd.concat(
            df_list, axis=1,
        ).dropna(
            axis="columns", how="all",
        ).dropna(
            axis="rows", how="all",
        ).sort_index(
            axis=1,  # Sort columns
        )
        self._validate_output(rel_returns_df)

        abs_price_dfs = {}  # Used for diagnostics
        for underlying, ci in self.calib_info_dict.items():
            for load_shape in ci.series_id.keys():
                lsu = underlying.get_shaped(load_shape)
                price_df = price_data.get_price_dataframe_for_lsu(lsu)
                abs_price_dfs[lsu] = price_df

        if DEBUG:  # pragma: no cover
            abs_prices_df = pd.concat([df for df in abs_price_dfs.values()], axis=1)
            abs_prices_df = abs_prices_df[sorted(abs_prices_df)]
            for file_name, dataframe in {'prices': abs_prices_df, 'returns_df': rel_returns_df}.items():
                dataframe.to_csv(os.path.join(r'C:\Temp', f'{file_name}.csv'), columns=sorted(dataframe.columns))

        return rel_returns_df, abs_price_dfs


def log_missing_contracts_and_dates(log_returns_df):
    log = logging.getLogger(__name__)
    missing_contract_series = log_returns_df.isnull().all()
    missing_contracts = list(missing_contract_series[missing_contract_series].index)
    if missing_contracts:  # pragma: no cover
        log.info(f"The following contracts did not have any data\n{missing_contracts}")
    filtered = log_returns_df.dropna(axis="columns", how="all")
    missing_row_data_mask = filtered.isnull().any(axis=1)
    dates_with_missing_data = missing_row_data_mask[missing_row_data_mask]
    if len(dates_with_missing_data) > 0:  # pragma: no cover
        log.info(f"The following dates are missing data:\n")
        for d in dates_with_missing_data.index:
            missing_contract_series = filtered[filtered.index == d].isnull().all()
            missing_contracts = list(missing_contract_series[missing_contract_series].index)
            log.info(f"{d.date().strftime('%Y-%m-%d %a')}: {sorted(missing_contracts)}")


def assert_dates_start_at_calib_window(dates, calib_period, check_window):
    """Raise and error if the returns start date is not within a tolerance of the calibration period start."""
    first_date = dates[0]
    error_msg = (
        f"Data missing at beginning of calibration period. "
        f"First date: {first_date.isoformat()}, expected first date: {calib_period.start.isoformat()}"
    )
    if check_date_is_within_window(first_date, calib_period.start, check_window) is False:
        raise RuntimeError(error_msg)


def assert_dates_end_at_calib_window(dates, calib_period, check_window):
    """Raise and error if the returns end date is not within a tolerance of the calibration period end."""
    last_date = dates[-1]
    error_msg = (
        f"Data missing at end of calibration period. "
        f"Last date: {last_date.isoformat()}, expected last date: {calib_period.end.isoformat()}"
    )
    if check_date_is_within_window(calib_period.end, last_date, check_window) is False:
        raise RuntimeError(error_msg)


def assert_dates_cover_most_of_calib_period(dates, calib_period):
    """Raise an error if the returns dates don't cover at least 70% of the weekdays calibration period."""
    threshold = len(calib_period) * (5 / 7) * 0.7
    if len(dates) < threshold:
        raise RuntimeError(f"Not enough dates where all contracts are quoted ({len(dates)} < {threshold})")


def check_date_is_within_window(date1, date2, check_window):
    """Check that two dates are within a window"""
    return (date1 - date2).days <= check_window


