import datetime
from unittest.mock import call, patch

from thorn.api import (
    bachelier_price, bachelier_delta, bachelier_gamma, bachelier_vega, bachelier_theta, bachelier_vol_from_premium
)
from thorn_excel import (
    QABachelierPrice, QABachelierDelta, QABachelierGamma, QABachelierVega, QABachelierTheta, QABachelierVolFromPremium
)
from thorn_excel.functions.test.excel_utils import xlwings_call_udf
from thorn_excel.utils.test.context_fixtures import cache_context


class TestNormalPricerFunctions:

    @cache_context()
    @patch("thorn_excel.functions.normal_pricer.bachelier_price", autospec=bachelier_price, return_value=5)
    def test_bachelier_price(self, m_func):
        actual = xlwings_call_udf(QABachelierPrice, ['2016-08-01', '2017-01-01', 1.2, 1.0, 'C', 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 'C', 0.2, 0.99) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.normal_pricer.bachelier_delta", autospec=bachelier_delta, return_value=5)
    def test_bachelier_delta(self, m_func):
        actual = xlwings_call_udf(QABachelierDelta, ['2016-08-01', '2017-01-01', 1.2, 1.0, 'C', 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 'C', 0.2, 0.99) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.normal_pricer.bachelier_gamma", autospec=bachelier_gamma, return_value=5)
    def test_bachelier_gamma(self, m_func):
        actual = xlwings_call_udf(QABachelierGamma, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.2, 0.99) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.normal_pricer.bachelier_vega", autospec=bachelier_vega, return_value=5)
    def test_bachelier_vega(self, m_func):
        actual = xlwings_call_udf(QABachelierVega, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.2, 0.99) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.normal_pricer.bachelier_theta", autospec=bachelier_theta, return_value=5)
    def test_bachelier_theta(self, m_func):
        actual = xlwings_call_udf(QABachelierTheta, ['2016-08-01', '2017-01-01', 1.2, 1.0, 'C', 0.2, 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 'C', 0.2, 0.99) == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.normal_pricer.bachelier_vol_from_premium", autospec=bachelier_vol_from_premium,
           return_value=5)
    def test_bachelier_vol_from_premium(self, m_func):
        actual = xlwings_call_udf(QABachelierVolFromPremium, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.1, 'C', 0.99])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.1, 'C', 0.99) == m_func.call_args


