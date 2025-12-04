from datetime import date, datetime, time, timedelta, timezone
from unittest import TestCase

from general_utils import (
    FRI, MON, SAT, SUN, THU, TUE, WED, convert_ole_to_date, create_date, force_date, force_datetime,
    get_first_of_next_month, is_weekday, parse_date, parse_datetime, parse_month, previous_weekday, time_between,
    workdays, BST, convert_datetime_to_ole, days_split, is_excel_date, last_occurrence_of_day_in_month, dst_start,
    dst_end, is_dst_start, is_dst_end, is_weekend_day
)
from general_utils.datetime_utils import this_or_next_weekday, next_weekday, this_or_previous_weekday, \
    get_nth_of_next_month


class TestDatetimeUtils(TestCase):
    """Test cases for datetime_utils"""

    def test_parse_month(self):
        self.assertEqual(parse_month("Mar-15"), (2015, 3))
        with self.assertRaises(ValueError):
            _ = parse_month("Foo-2019")
        with self.assertRaises(ValueError):
            _ = parse_month("Dec-2015")

    def test_time_between(self):
        a = date(2015, 1, 1)
        b = date(2016, 1, 1)
        c = date(2015, 7, 1)
        self.assertEqual(time_between(a, a), 0)
        self.assertEqual(time_between(a, b), 1)
        self.assertEqual(time_between(b, a), -1)
        self.assertEqual(round(time_between(a, c), 2), 0.5)

    def test_force_datetime(self):
        expected = datetime(2015, 5, 3, 0)
        self.assertEqual(expected, force_datetime(expected))
        self.assertEqual(expected, force_datetime(expected.date()))

    def test_force_date(self):
        expected = date(2015, 5, 3)
        self.assertEqual(expected, force_date(expected))
        self.assertEqual(expected, force_date(datetime.combine(expected, time(12))))

    def test_parse_date(self):
        dates = [
            ("2014-10-01", date(2014, 10, 1)),
            ("2010-01-10", date(2010, 1, 10)),
            ("3/2/2015", date(2015, 2, 3)),
            ("14-12-01", date(2001, 12, 14)),
            ("14-12-2001", date(2001, 12, 14)),
            ("Fri 14/12/2001", date(2001, 12, 14)),
            ("14-Dec-2001", date(2001, 12, 14)),
            (" 14-Dec-2001 ", date(2001, 12, 14)),
            (date(2014, 10, 1), date(2014, 10, 1)),
            # Excel format dates
            (42103, date(2015, 4, 9)),
            (42103.0, date(2015, 4, 9)),
        ]
        error_strings = [
            ('', ValueError),
            ('14-13-01', ValueError),
            ('Not a date', ValueError),
            (None, TypeError),
        ]
        for string, date_obj in dates:
            with self.subTest(string):
                self.assertEqual(date_obj, parse_date(string))
        for string, error_type in error_strings:
            with self.subTest(string):
                self.assertRaises(error_type, parse_date, string)

    def test_parse_datetime(self):
        self.assertEqual(parse_datetime("2008-12-31 23:00:00"),
                         datetime(2008, 12, 31, 23))
        self.assertEqual(parse_datetime("2008-12-31 23:00"),
                         datetime(2008, 12, 31, 23))
        self.assertEqual(parse_datetime("11/03/2013 17:00"),
                         datetime(2013, 3, 11, 17))
        self.assertEqual(parse_datetime("11/03/2013 17:00:05"),
                         datetime(2013, 3, 11, 17, 0, 5))
        self.assertEqual(parse_datetime(datetime(2013, 3, 11, 17, 0, 5)),
                         datetime(2013, 3, 11, 17, 0, 5))
        self.assertEqual(parse_datetime("2025-05-21 02:49:20.050000"),
                         datetime(2025, 5, 21, 2, 49, 20, 50000))

    def workdays_check_1(self, start, offset):
        self.assertEqual(workdays(start, start + timedelta(offset), (MON, TUE, WED, THU, FRI, SAT, SUN)),
                         offset + 1)

    def test_workdays(self):
        start = date(2000, 1, 1)
        self.workdays_check_1(start, 1)
        self.workdays_check_1(start, 2)
        self.workdays_check_1(start, 3)
        self.workdays_check_1(start, 4)
        self.workdays_check_1(start, 5)
        self.workdays_check_1(start, 6)
        self.workdays_check_1(start, 7)
        self.workdays_check_1(start, 8)
        self.workdays_check_1(start, 10)
        self.assertEqual(
            workdays(start, start + timedelta(7), (MON, TUE, WED, THU, FRI)), 5)
        self.assertEqual(workdays(start, start + timedelta(7), (MON,)), 1)

    def test_days_split(self):
        # over several weeks
        actual = days_split(date(2019, 2, 27), date(2019, 3, 16))
        expected = [2, 2, 3, 3, 3, 3, 2]
        self.assertEqual(expected, actual)

        # wed to sat
        actual = days_split(date(2019, 2, 27), date(2019, 3, 2))
        expected = [0, 0, 1, 1, 1, 1, 0]
        self.assertEqual(expected, actual)

        # single monday
        self.assertEqual([1] + [0] * 6, days_split(date(2019, 3, 4), date(2019, 3, 4)))

        # end before start
        self.assertRaises(AssertionError, days_split, date(2019, 3, 16), date(2019, 2, 27))

    def test_create_date(self):
        d = create_date(2000, 60, 12)
        self.assertEqual(d, date(2004, 12, 12))
        d = create_date(2000, 1, 12)
        self.assertEqual(d, date(2000, 1, 12))
        d = create_date(2000, 12, 12)
        self.assertEqual(d, date(2000, 12, 12))

    def test_get_nth_of_next_month(self):
        d = date(2001, 12, 20)
        for i in range(1, 31 + 1):  # valid days in January
            expected = date(2002, 1, i)
            self.assertEqual(expected, get_nth_of_next_month(d, i))

        for i in (0, 32):  # out of range days for January
            with self.assertRaises(ValueError):
                _ = get_nth_of_next_month(d, i)

    def test_get_first_of_next_month(self):
        d1 = date(2000, 1, 1)
        d2 = date(2000, 1, 20)
        d3 = date(2000, 1, 31)
        for d in [d1, d2, d3]:
            self.assertEqual(get_first_of_next_month(d), date(2000, 2, 1))

        d4 = date(2000, 12, 25)
        self.assertEqual(get_first_of_next_month(d4), date(2001, 1, 1))

    def test_is_weekday(self):
        test_dates = [
            (date(2014, 5, 9), True),  # Friday
            (date(2014, 5, 5), True),  # Monday
            (date(2014, 5, 10), False),  # Saturday
            (date(2014, 5, 11), False),  # Sunday
        ]
        for day, expt in test_dates:
            self.assertEqual(expt, is_weekday(day), f"{expt} != {is_weekday(day)} for {day}")
            self.assertEqual(not expt, is_weekend_day(day), f"{not expt} != {is_weekday(day)} for {day}")
            self.assertTrue(is_weekday(previous_weekday(day)), f"{previous_weekday(day)} is not a weekday ({day})")

    def test_this_or_next_weekday(self):
        for i in range(40):
            d1 = date(2021, 1, 1) + timedelta(i)
            d2 = d1
            while not is_weekday(d2):
                d2 = d2 + timedelta(1)

            self.assertEqual(d2, this_or_next_weekday(d1))

    def test_previous_weekday(self):
        cases = [
            (date(2022, 4, 18), date(2022, 4, 15)),  # Monday -> Friday
            (date(2022, 4, 19), date(2022, 4, 18)),  # Tuesday -> Monday
            (date(2022, 4, 20), date(2022, 4, 19)),
            (date(2022, 4, 21), date(2022, 4, 20)),
            (date(2022, 4, 22), date(2022, 4, 21)),
            (date(2022, 4, 23), date(2022, 4, 22)),  # Saturday -> Friday
            (date(2022, 4, 24), date(2022, 4, 22)),  # Sunday -> Friday
        ]
        for t, expected in cases:
            self.assertEqual(expected, previous_weekday(t))

    def test_this_or_previous_weekday(self):
        for i in range(40):
            d1 = date(2021, 1, 1) + timedelta(i)
            d2 = d1
            while not is_weekday(d2):
                d2 = d2 - timedelta(1)

            self.assertEqual(d2, this_or_previous_weekday(d1))

    def test_next_weekday(self):
        for i in range(40):
            d0 = date(2021, 1, 1) + timedelta(i)
            d1 = next_weekday(d0)
            self.assertTrue(is_weekday(d1))
            d2 = next_weekday(d1)
            d3 = previous_weekday(d2)
            self.assertEqual(d1, d3)

    def test_convert_ole_to_date(self):
        self.assertEqual(date(2015, 2, 3), convert_ole_to_date(42038))

    def test_convert_datetime_to_ole(self):
        self.assertEqual(convert_datetime_to_ole(datetime(2018, 5, 11, 12, 0)), 43231.5)

    def test_is_excel_date(self):
        self.assertTrue(is_excel_date(35000))
        self.assertTrue(is_excel_date(35000.1))
        self.assertFalse(is_excel_date(25000))
        self.assertFalse(is_excel_date(25000.1))
        self.assertFalse(is_excel_date(None))
        self.assertFalse(is_excel_date("35000"))
        self.assertTrue(is_excel_date("01/01/2015"))
        self.assertFalse(is_excel_date("Not a Date"))


class BSTTestCase(TestCase):
    """ Tests on the BST tzinfo object """

    def test_name(self):
        self.assertEqual("BST", BST().tzname(None))

    def test_to_dst(self):
        test_cases = [
            (datetime(2017, 1, 1), 0),
            (datetime(2017, 4, 30), 1),
            (datetime(2017, 6, 1), 1),
            (datetime(2017, 10, 1), 1),
            (datetime(2017, 11, 1), 0),
        ]
        for day, offset in test_cases:
            with self.subTest(day):
                self.assertEqual(timedelta(hours=offset), BST().utcoffset(day))
                self.assertEqual(BST().utcoffset(day), BST().dst(day))

    def test_equality(self):
        self.assertEqual(timedelta(0),
                         (datetime(2017, 4, 29, 23, 0, 0, tzinfo=timezone.utc) -
                          datetime(2017, 4, 30, 0, 0, 0, tzinfo=BST())))


class DSTFunctionsTestCase(TestCase):
    def test_last_occurrence_of_day_in_month(self):
        month = 7
        year = 2021
        cases = [
            (0, 26),  # Monday
            (1, 27), (2, 28), (3, 29), (4, 30), (5, 31),
            (6, 25)  # Sunday
        ]
        for day, expected in cases:
            self.assertEqual(expected, last_occurrence_of_day_in_month(year, month, day))

    def test_dst_start(self):
        cases = [
            (2020, date(2020, 3, 29)),
            (2021, date(2021, 3, 28)),
            (2022, date(2022, 3, 27))
        ]
        for year, expected in cases:
            self.assertEqual(expected, dst_start(year))

    def test_dst_end(self):
        cases = [
            (2020, date(2020, 10, 25)),
            (2021, date(2021, 10, 31)),
            (2022, date(2022, 10, 30))
        ]
        for year, expected in cases:
            self.assertEqual(expected, dst_end(year))

    def test_is_dst_start(self):
        self.assertFalse(is_dst_start(date(2021, 2, 15)))  # wrong month case
        cases = [(date(2021, 3, d), True if d == 28 else False) for d in range(1, 31 + 1)]  # Only true for 28 Mar 2021
        for d, expected in cases:
            self.assertEqual(expected, is_dst_start(d))

    def test_is_dst_end(self):
        self.assertFalse(is_dst_end(date(2021, 2, 15)))  # wrong month case
        cases = [(date(2021, 10, d), True if d == 31 else False) for d in range(1, 31 + 1)]  # Only true for 31 Oct 2021
        for d, expected in cases:
            self.assertEqual(expected, is_dst_end(d))


