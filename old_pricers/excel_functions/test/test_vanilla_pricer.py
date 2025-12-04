import datetime
from unittest.mock import call, patch

import pytest
from pytest_quants import assert_almost_equal

from thorn.api import (
    vanilla_smile_price, vanilla_smile_delta, vanilla_smile_gamma, vanilla_smile_vega,
    vanilla_smile_theta, black76_price, black76_delta, black76_gamma, black76_vega,
    black76_theta, build_vanilla_smile, vanilla_smile_pillar_vega
)
from thorn.api.test.context_fixtures import underlying_loader_context
from thorn_excel import (
    CACHE, QAVanillaSmile, QAVanillaSmilePrice, QAVanillaSmileDelta, QAVanillaSmileGamma,
    QAVanillaSmileVega, QAVanillaSmileTheta, QABlack76Price, QABlack76Delta, QABlack76Gamma,
    QABlack76Vega, QABlack76Theta, QAVanillaSmileSkewVega, QABuildPSC, QACommodityCurve, QAMonthlyEuropean
)
from thorn_excel.functions.external import STMonthlyEuropean
from thorn_excel.functions.external.test.fixtures import CallPut2025Q3
from thorn_excel.functions.test.excel_utils import xlwings_call_udf
from thorn_excel.utils.test.context_fixtures import cache_context
from underlying.fixtures.underlyings import TTF


class TestVanillaPricerDecorators:
    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.build_vanilla_smile",
           autospec=build_vanilla_smile, return_value=6)
    def test_vanilla_smile(self, m_func):
        args = ('2015-01-01', '2015-01-01', '2015-01-01', ['0.1*', 'ATM', '0.9*'], [0.55, 0.34, 0.60])
        label = xlwings_call_udf(QAVanillaSmile, args)[0][0]
        actual = CACHE.lookup_by_label(label)
        assert 6 == actual
        expected_args = (datetime.date(2015, 1, 1), datetime.date(2015, 1, 1),
                         ['0.1*', 'ATM', '0.9*'], [0.55, 0.34, 0.60], False, False, False, False)
        assert call(*expected_args) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.build_vanilla_smile",
           autospec=build_vanilla_smile, return_value=None)
    def test_vanilla_smile_date_mismatch(self, m_func):
        args = ('2015-01-01', '2015-01-02', '2015-01-01', ['0.1*', 'ATM', '0.9*'], [0.55, 0.34, 0.60])
        label = xlwings_call_udf(QAVanillaSmile, args)[0][0]
        assert label.startswith("ERROR: Pricing Date != Calibration Date")
        with pytest.raises(KeyError):
            CACHE.lookup_by_label(label)
        assert 0 == m_func.call_count

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.vanilla_smile_price", autospec=vanilla_smile_price, return_value=5)
    def test_vanilla_smile_price(self, m_func):
        CACHE.add_entry('func', (1, 2), 'THING_ADDED_TO_CACHE4', None, 'Smile')
        actual = xlwings_call_udf(QAVanillaSmilePrice, ['Smile', 'C', 1.2, 1.0])[0][0]
        assert 5 == actual
        assert call('THING_ADDED_TO_CACHE4', 'C', 1.2, 1.0, 1.0) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.vanilla_smile_delta", autospec=vanilla_smile_delta, return_value=5)
    def test_vanilla_smile_delta(self, m_func):
        CACHE.add_entry('func', (1, 2), 'THING_ADDED_TO_CACHE4', None, 'Smile')
        actual = xlwings_call_udf(QAVanillaSmileDelta, ['Smile', 'C', 1.2, 1.0])[0][0]
        assert 5 == actual
        assert call('THING_ADDED_TO_CACHE4', 'C', 1.2, 1.0, 1.0) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.vanilla_smile_gamma", autospec=vanilla_smile_gamma, return_value=5)
    def test_vanilla_smile_gamma(self, m_func):
        CACHE.add_entry('func', (1, 2), 'THING_ADDED_TO_CACHE4', None, 'Smile')
        actual = xlwings_call_udf(QAVanillaSmileGamma, ['Smile', 1.2, 1.0])[0][0]
        assert 5 == actual
        assert call('THING_ADDED_TO_CACHE4', 1.2, 1.0, 1.0) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.vanilla_smile_vega", autospec=vanilla_smile_vega, return_value=5)
    def test_vanilla_smile_vega(self, m_func):
        CACHE.add_entry('func', (1, 2), 'THING_ADDED_TO_CACHE4', None, 'Smile')
        actual = xlwings_call_udf(QAVanillaSmileVega, ['Smile', 1.2, 1.0])[0][0]
        assert 5 == actual
        assert call('THING_ADDED_TO_CACHE4', 1.2, 1.0, 1.0) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.vanilla_smile_pillar_vega", autospec=vanilla_smile_pillar_vega, return_value=5)
    def test_vanilla_smile_skew_vega(self, m_func):
        CACHE.add_entry('func', (1, 2), 'THING_ADDED_TO_CACHE100', None, 'Smile')
        actual = xlwings_call_udf(QAVanillaSmileSkewVega, ['Smile', 1.2, 1.0, 'ATM'])[0][0]
        assert 5 == actual
        assert call('THING_ADDED_TO_CACHE100', 1.2, 1.0, 'ATM', 1.0) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.vanilla_smile_theta", autospec=vanilla_smile_theta, return_value=5)
    def test_vanilla_smile_theta(self, m_func):
        CACHE.add_entry('func', (1, 2), 'THING_ADDED_TO_CACHE4', None, 'Smile')
        actual = xlwings_call_udf(QAVanillaSmileTheta, ['Smile', 'C', 1.2, 1.0])[0][0]
        assert 5 == actual
        assert call('THING_ADDED_TO_CACHE4', 'C', 1.2, 1.0, 1.0, 1.0) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.black76_price", autospec=black76_price, return_value=5)
    def test_vanilla_black_76_price(self, m_func):
        actual = xlwings_call_udf(QABlack76Price, ['2016-08-01', '2017-01-01', 1.2, 1.0, 'C', 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 'C', 0.2, 0.99) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.black76_delta", autospec=black76_delta, return_value=5)
    def test_vanilla_black_76_delta(self, m_func):
        actual = xlwings_call_udf(QABlack76Delta, ['2016-08-01', '2017-01-01', 1.2, 1.0, 'C', 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 'C', 0.2, 0.99) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.black76_gamma", autospec=black76_gamma, return_value=5)
    def test_vanilla_black_76_gamma(self, m_func):
        actual = xlwings_call_udf(QABlack76Gamma, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.2, 0.99) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.black76_vega", autospec=black76_vega, return_value=5)
    def test_vanilla_black_76_vega(self, m_func):
        actual = xlwings_call_udf(QABlack76Vega, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.2, 0.99) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.vanilla_pricer.black76_theta", autospec=black76_theta, return_value=5)
    def test_vanilla_black_76_theta(self, m_func):
        actual = xlwings_call_udf(QABlack76Theta, ['2016-08-01', '2017-01-01', 1.2, 1.0, 'C', 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 'C', 0.2, 0.99) == m_func.call_args


class TestVanillaMacroFunctions:
    @cache_context()
    @underlying_loader_context()
    def test_QAMonthlyEuropean_integration(self):
        psc_label = xlwings_call_udf(QABuildPSC, [CallPut2025Q3.calibration_date, CallPut2025Q3.psc_data])[0][0]
        curve_label = xlwings_call_udf(QACommodityCurve, [CallPut2025Q3.curve_data])[0][0]

        result = xlwings_call_udf(
            QAMonthlyEuropean,
            [
                curve_label, psc_label, CallPut2025Q3.tenor, CallPut2025Q3.cross,
                [CallPut2025Q3.strike_1, CallPut2025Q3.strike_2],
                [CallPut2025Q3.opt_1, CallPut2025Q3.opt_2]
            ]
        )
        assert_almost_equal(CallPut2025Q3.expected, result[0])

        # check that this returns the same as the reference function
        result_ref_fn = xlwings_call_udf(
            STMonthlyEuropean,
            [
                curve_label, psc_label, CallPut2025Q3.tenor, CallPut2025Q3.cross,
                CallPut2025Q3.strike_1, CallPut2025Q3.opt_1,
                CallPut2025Q3.weight_2, CallPut2025Q3.strike_2, CallPut2025Q3.opt_2
            ]
        )
        assert_almost_equal(result_ref_fn[0], result[0])

    @cache_context()
    @underlying_loader_context()
    def test_QAMonthlyEuropean_integration_buy_call_sell_put(self):
        psc_label = xlwings_call_udf(QABuildPSC, [CallPut2025Q3.calibration_date, CallPut2025Q3.psc_data])[0][0]
        curve_label = xlwings_call_udf(QACommodityCurve, [CallPut2025Q3.curve_data])[0][0]

        result = xlwings_call_udf(
            QAMonthlyEuropean,
            [
                curve_label, psc_label, CallPut2025Q3.tenor, CallPut2025Q3.cross,
                [CallPut2025Q3.strike_1, CallPut2025Q3.strike_1],
                [CallPut2025Q3.opt_1, CallPut2025Q3.opt_2],
                [CallPut2025Q3.weight_1, -CallPut2025Q3.weight_1],  # negative weight for 'sell'
            ]
        )
        expected = [
            CallPut2025Q3.cross - CallPut2025Q3.strike_1,  # PV
            1.0,  # Delta
            0.0,  # Gamma
            0.0,  # Vega / 100
            0.0,  # Theta
            1.0,  # B76 Delta
            CallPut2025Q3.exp_vol,  # Vol Leg 1
            CallPut2025Q3.exp_vol,  # Vol Leg 2
            CallPut2025Q3.exp_delta_star,  # Delta Star Leg 1
            CallPut2025Q3.exp_delta_star  # Delta Star Leg 2
        ]
        assert_almost_equal(expected, result[0])

        # 2 * half weighted sold puts. This should also be faster to compute as it should look up smiles in cache
        expected = [
            *expected[:6],
            CallPut2025Q3.exp_vol, CallPut2025Q3.exp_vol, CallPut2025Q3.exp_vol,
            CallPut2025Q3.exp_delta_star, CallPut2025Q3.exp_delta_star, CallPut2025Q3.exp_delta_star
        ]
        result2 = xlwings_call_udf(
            QAMonthlyEuropean,
            [
                curve_label, psc_label, CallPut2025Q3.tenor, CallPut2025Q3.cross,
                [CallPut2025Q3.strike_1, CallPut2025Q3.strike_1, CallPut2025Q3.strike_1],
                [CallPut2025Q3.opt_1, CallPut2025Q3.opt_2, CallPut2025Q3.opt_2],
                [CallPut2025Q3.weight_1, -0.5 * CallPut2025Q3.weight_1, -0.5 * CallPut2025Q3.weight_1],
            ]
        )
        assert_almost_equal(expected, result2[0])

    @cache_context()
    @underlying_loader_context()
    def test_QAMonthlyEuropean_integration_reference_fwd_equal_to_cross(self):
        psc_label = xlwings_call_udf(QABuildPSC, [CallPut2025Q3.calibration_date, CallPut2025Q3.psc_data])[0][0]
        curve_label = xlwings_call_udf(QACommodityCurve, [[
            [f'{TTF}', 'BASE'],
            ['2025_07', 100],
            ['2025_08', 100],
            ['2025_09', 100],
        ]])[0][0]  # these prices should be ignored. values are silly to make it obvious

        result = xlwings_call_udf(
            QAMonthlyEuropean,
            [
                curve_label, psc_label, CallPut2025Q3.tenor, CallPut2025Q3.cross,
                [CallPut2025Q3.strike_1, CallPut2025Q3.strike_1],
                [CallPut2025Q3.opt_1, CallPut2025Q3.opt_2],
                [CallPut2025Q3.weight_1, -CallPut2025Q3.weight_1],  # negative weight for 'sell'
                True
            ]
        )
        expected = [
            CallPut2025Q3.cross - CallPut2025Q3.strike_1,  # PV
            1.0,  # Delta
            0.0,  # Gamma
            0.0,  # Vega / 100
            0.0,  # Theta
            1.0,  # B76 Delta
            0.1254305655999806,  # Vol Leg 1
            0.1254305655999806,  # Vol Leg 2
            0.12383001325326212,  # Delta Star Leg 1
            0.12383001325326212  # Delta Star Leg 2
        ]
        assert_almost_equal(expected, result[0])
        # should be the same as using a curve flat price at cross
        cross_curve_label = xlwings_call_udf(QACommodityCurve, [[
            [f'{TTF}', 'BASE'],
            ['2025_07', CallPut2025Q3.cross],
            ['2025_08', CallPut2025Q3.cross],
            ['2025_09', CallPut2025Q3.cross],
        ]])[0][0]
        result_cross_curve = xlwings_call_udf(
            QAMonthlyEuropean,
            [
                cross_curve_label, psc_label, CallPut2025Q3.tenor, CallPut2025Q3.cross,
                [CallPut2025Q3.strike_1, CallPut2025Q3.strike_1],
                [CallPut2025Q3.opt_1, CallPut2025Q3.opt_2],
                [CallPut2025Q3.weight_1, -CallPut2025Q3.weight_1],  # negative weight for 'sell'
                False
            ]
        )
        assert_almost_equal(result[0], result_cross_curve[0])

        # 2 * half weighted sold puts. This should also be faster to compute as it should look up smiles in cache
        expected = [
            *expected[:6],
            0.1254305655999806, 0.1254305655999806, 0.1254305655999806,
            0.12383001325326212, 0.12383001325326212, 0.12383001325326212
        ]
        result2 = xlwings_call_udf(
            QAMonthlyEuropean,
            [
                curve_label, psc_label, CallPut2025Q3.tenor, CallPut2025Q3.cross,
                [CallPut2025Q3.strike_1, CallPut2025Q3.strike_1, CallPut2025Q3.strike_1],
                [CallPut2025Q3.opt_1, CallPut2025Q3.opt_2, CallPut2025Q3.opt_2],
                [CallPut2025Q3.weight_1, -0.5 * CallPut2025Q3.weight_1, -0.5 * CallPut2025Q3.weight_1],
                True
            ]
        )
        assert_almost_equal(expected, result2[0])
        # should be the same as using a curve flat price at cross
        result_cross_curve2 = xlwings_call_udf(
            QAMonthlyEuropean,
            [
                cross_curve_label, psc_label, CallPut2025Q3.tenor, CallPut2025Q3.cross,
                [CallPut2025Q3.strike_1, CallPut2025Q3.strike_1, CallPut2025Q3.strike_1],
                [CallPut2025Q3.opt_1, CallPut2025Q3.opt_2, CallPut2025Q3.opt_2],
                [CallPut2025Q3.weight_1, -0.5 * CallPut2025Q3.weight_1, -0.5 * CallPut2025Q3.weight_1],
                False
            ]
        )
        assert_almost_equal(result2[0], result_cross_curve2[0])

