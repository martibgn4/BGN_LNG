"""
The functions in this file can be used to build DataFrames, that have
column headings of string representations of Tenors
"""

from collections import defaultdict

import pandas as pd

from general_utils import parse_date, memoize
from thorn.core.calibration.base.contract_keys import Contract
from time_period import (
    DateRange, BASE, PEAK, OFFPEAK, LoadShapedDateRange, RelativeDateRange,
    quarter_containing, season_containing, year_containing, str_vault, ALWAYS,
)
from underlying import FX


def build_price_data_frame(contract_data, window=ALWAYS, tenor_remap=None, tenor_filter=lambda _: True):
    df = pd.DataFrame.from_dict(contract_data, orient="index")
    df = df.rename(index=parse_date, columns=lambda t: (tenor_remap or {}).get(t, t))  # cast row + col names
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    return df.loc[window.start:window.end, filter(tenor_filter, df.columns)]  # apply row and col filters


class PeakOffpeakPriceConverter:

    @memoize
    def _peak_offpeak_durations(self, contract):
        if contract in ["DA", "DA01", "DA02", "DA03"]:
            return 0.5, 0.5

        if contract in ["W01", "W02", "W03", "W04"]:
            return 60 / 24, 108 / 24

        dr = DateRange(contract)
        pk_duration = LoadShapedDateRange(dr, PEAK).duration()
        return pk_duration, dr.duration() - pk_duration

    @memoize
    def _is_valid_peak_offpeak_tenor(self, t):
        if t in {"DA", "DA01", "DA02", "DA03", "W01", "W02", "W03", "W04"}:
            return True
        return RelativeDateRange.is_valid(t)

    def build_offpeak_prices(self, peak_prices, base_prices):
        """ Combines peak and base prices into OffPeak prices using durations of the tenors """

        common_tenors = [
            t for t in set(peak_prices.columns).intersection(base_prices.columns)
            if self._is_valid_peak_offpeak_tenor(t)
        ]

        op_px = {}
        for tenor in common_tenors:
            try:
                peak_dn, op_dn = self._peak_offpeak_durations(tenor)
            except ValueError:
                continue
            else:
                total_duration = peak_dn + op_dn
                op_px[tenor] = (total_duration * base_prices[tenor] - peak_dn * peak_prices[tenor]) / op_dn

        return pd.DataFrame(op_px)


def enrich_with_synthetic_prices(price_df):
    """aggregates monthly quotes into quarters, seasons, years"""
    aggregates = defaultdict(list)
    for col in price_df.columns:
        try:
            tenor = DateRange(col)
            if tenor.is_month:
                d = tenor.start
                aggregates[(3, str_vault(quarter_containing(d)))].append(col)
                aggregates[(6, str_vault(season_containing(d)))].append(col)
                aggregates[(12, str_vault(year_containing(d)))].append(col)
        except ValueError:
            continue

    for (num_months, super_col), months in aggregates.items():
        if len(months) == num_months:
            super_series = price_df[months[0]]
            for month in months[1:]:
                super_series = super_series + price_df[month]
            price_df[super_col] = super_series / num_months
    return price_df


def combine_primary_and_secondary_dataframes(price_df, secondary_price_df, secondary_contracts):
    """ Adds tenors from the secondary price dataframe to the primary """
    for column in secondary_price_df.columns:
        if column in secondary_contracts:
            price_df[column] = secondary_price_df[column]
    return price_df


class PriceData:
    """
    Takes price history data, a price data configuration (dict(underlying, price_data_info))
    and loads the data by underlying on request
    """

    def __init__(self, price_history, price_data_infos, window=ALWAYS, tenor_filter=lambda _: True):
        self.price_history = price_history
        self.price_data_infos = price_data_infos
        self.window = window
        self.tenor_filter = tenor_filter
        self.converter = PeakOffpeakPriceConverter()
        self._df_cache = {}  # Cache from LSU to dataframe

    def _build_price_data_frame(self, series_id, price_data_info):
        contract_data = self.price_history.get(series_id, {})
        tenor_remap = price_data_info.tenor_remap.get(series_id, {})
        return build_price_data_frame(
            contract_data,
            window=self.window,
            tenor_remap=tenor_remap,
            tenor_filter=self.tenor_filter
        )

    def _load_data_for_cache(self, lsu):
        """
        Loads a price DataFrame for a load_shape of a PriceDataInfo

        Converts dict to df,
        Adds secondary series
        Compute offpeak and synthetic prices

        """
        load_shape = lsu.load_shape
        price_data_info = self.price_data_infos[lsu.base_underlying]

        def _build_price_df(series_id, hub_series_id):
            # get primary price series
            price_df = self._build_price_data_frame(series_id, price_data_info)
            if hub_series_id:
                # add hub price (because primary series is a spread to the hub)
                hub_price_df = self._build_price_data_frame(hub_series_id, price_data_info)
                price_df = price_df + hub_price_df
            return price_df.dropna(axis="columns", how="all")

        if load_shape in price_data_info.series_id:
            series_id = price_data_info.series_id[load_shape]
            hub_series_id = price_data_info.hub_series_id.get(load_shape)
            price_df = _build_price_df(series_id, hub_series_id)

            # fill in from secondary price series
            if load_shape in price_data_info.secondary_series_id:
                secondary_series_id = price_data_info.secondary_series_id[load_shape]
                secondary_hub_series_id = price_data_info.secondary_hub_series_id.get(load_shape)
                secondary_price_df = _build_price_df(secondary_series_id, secondary_hub_series_id)

                price_df = combine_primary_and_secondary_dataframes(
                    price_df, secondary_price_df, price_data_info.secondary_series_contracts
                )

        else:
            # Construct the prices for the load shape (e.g. OFFPEAK) from the available (e.g. PEAK + BASE)
            assert load_shape == OFFPEAK
            peak_prices = self._load_data(lsu.get_shaped(PEAK))
            base_prices = self._load_data(lsu.get_shaped(BASE))
            price_df = self.converter.build_offpeak_prices(peak_prices, base_prices)

        if price_data_info.create_synthetic_contracts:
            price_df = enrich_with_synthetic_prices(price_df)

        return price_df.dropna(axis="rows", how="all")

    def _load_data(self, lsu):
        df = self._df_cache.get(lsu, None)
        if df is None:
            df = self._load_data_for_cache(lsu)
            self._df_cache[lsu] = df
        return df

    def get_price_dataframe_for_lsu(self, lsu, columns_with_lsu=True, aux_lsu=None):
        """
        Convert dict to df, add secondary series and compute offpeak
        """
        # Load the raw data from the calib_indo
        price_df = self._load_data(lsu)

        # Load the auxiliary data, and prepend to price data if required
        if aux_lsu:
            aux_price_df = self._load_auxiliary_price_data(lsu, aux_lsu)
            # Prepend the aux price data to the time-series
            if price_df.empty:
                price_df = aux_price_df
            else:
                first_good_date = price_df.index[0]
                aux_price_df_pre = aux_price_df[aux_price_df.index < first_good_date]
                price_df = pd.concat([aux_price_df_pre, price_df], axis=0, sort=True)

        # Add the lsu to the columns names
        if columns_with_lsu:
            new_columns = [Contract(lsu, col_key).key for col_key in price_df.columns]
            price_df = price_df.rename(columns=dict(zip(price_df.columns, new_columns)))

        return price_df

    def _load_auxiliary_price_data(self, lsu, aux_lsu):
        aux_prices = self._load_data(aux_lsu).copy()

        # If auxiliary currency is not this currency, convert using Spot FX
        if lsu.currency != aux_lsu.currency:
            num_ccy, denom_ccy = lsu.currency, aux_lsu.currency
            fx = FX.get(num_ccy, denom_ccy)
            if fx in self.price_data_infos:
                operation = aux_prices.mul  # Multiply aux_prices by FX on or before date
            else:  # Expect inverse FX
                fx = fx.inverse
                operation = aux_prices.div  # Div aux_price by inv. FX on or before date

            _fx_data = self._load_data(fx)
            _fx_spot = _fx_data['SPOT'].reindex(aux_prices.index, method='ffill')
            aux_prices = operation(_fx_spot, axis=0)

        return aux_prices

    def get_price_dataframe_for_underlying(self, underlying, auxiliary_price_underlying=None):
        calib_info = self.price_data_infos[underlying]

        df_list = [pd.DataFrame([])]
        for load_shape in calib_info.price_process_weekly_shapes:
            lsu = underlying.get_shaped(load_shape)
            aux_lsu = auxiliary_price_underlying.get_shaped(load_shape) if auxiliary_price_underlying else None
            price_df = self.get_price_dataframe_for_lsu(lsu, aux_lsu=aux_lsu)
            df_list.append(price_df)
        prices = pd.concat(df_list, axis=1, sort=True)

        return prices


