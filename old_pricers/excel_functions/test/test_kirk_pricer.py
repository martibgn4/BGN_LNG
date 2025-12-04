import datetime
from unittest.mock import call, patch

from thorn.api import kirk_price, kirk_delta, kirk_gamma, kirk_vega, kirk_theta
from thorn_excel import QAKirkPrice, QAKirkDelta, QAKirkGamma, QAKirkVega, QAKirkTheta, QAKirkVegaByLeg
from thorn_excel.functions.test.excel_utils import xlwings_call_udf
from thorn_excel.utils.test.context_fixtures import cache_context


class TestNormalPricerFunctions:

    @cache_context()
    @patch("thorn_excel.functions.kirk_pricer.kirk_price", autospec=kirk_price, return_value=5)
    def test_kirk_price(self, m_func):
        actual = xlwings_call_udf(QAKirkPrice, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.1, 'C', 0.4, 0.5, 0.9])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.1, 0.4, 0.5, 0.9, 'C') == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.kirk_pricer.kirk_delta", autospec=kirk_delta, return_value=(5, 5))
    def test_kirk_delta(self, m_func):
        actual = xlwings_call_udf(QAKirkDelta, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.1, 'C', 0.4, 0.5, 0.9])[0]
        assert [5, 5] == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.1, 0.4, 0.5, 0.9, 'C') == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.kirk_pricer.kirk_gamma", autospec=kirk_gamma, return_value=(5, 5))
    def test_kirk_gamma(self, m_func):
        actual = xlwings_call_udf(QAKirkGamma, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.1, 'C', 0.4, 0.5, 0.9])[0]
        assert [5, 5] == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.1, 0.4, 0.5, 0.9, 'C') == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.kirk_pricer.kirk_vega", autospec=kirk_vega, return_value=5)
    def test_kirk_vega(self, m_func):
        actual = xlwings_call_udf(QAKirkVega, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.1, 'C', 0.4, 0.5, 0.9])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.1, 0.4, 0.5, 0.9, 'C') == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.kirk_pricer.kirk_vega_by_leg", autospec=kirk_vega, return_value=(5, 5))
    def test_kirk_vega_by_leg(self, m_func):
        actual = xlwings_call_udf(QAKirkVegaByLeg, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.1, 'C', 0.4, 0.5, 0.9])[0]
        assert [5, 5] == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.1, 0.4, 0.5, 0.9, 'C') == m_func.call_args

    @cache_context()
    @patch("thorn_excel.functions.kirk_pricer.kirk_theta", autospec=kirk_theta, return_value=5)
    def test_kirk_theta(self, m_func):
        actual = xlwings_call_udf(QAKirkTheta, ['2016-08-01', '2017-01-01', 1.2, 1.0, 0.1, 'C', 0.4, 0.5, 0.9])[0][0]
        assert 5 == actual
        assert call(datetime.date(2016, 8, 1), datetime.date(2017, 1, 1), 1.2, 1.0, 0.1, 0.4, 0.5, 0.9, 'C') == m_func.call_args


