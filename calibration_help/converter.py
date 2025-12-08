"""
These converters should only convert to Python Data Types.  These are responsible for forcing
a data type on the excel input as excel can give different types depending on the cell formatting.
DON'T USE THORN DATA TYPES HERE
"""
import math
from collections import defaultdict
from datetime import date, datetime
from numbers import Number

import pandas as pd
from general_utils import parse_date, parse_datetime, to_bool, NamedDataFrame, ParseableNamedEnum, date_formats as _date_formats
from thorn_excel import CACHE
from thorn_excel.constants import EXCEL_NA
from thorn_excel.mvc.errors import InvalidUserInputError
from time_period import LoadShape
from xlwings.conversion import Converter

from .cell_value_utils import valid_excel_date
from .replay_logger_utils import replay_logger_converter


class _Converter(Converter):
    """ Extends Converter class provided by Xlwings so that ranges/lists are handled by default """

    @classmethod
    @replay_logger_converter
    def read_value(cls, value, options=None):
        """ Called by Xlwings when reading a value (or list of values) from Excel """
        if isinstance(value, list):
            return [cls.read_value(v, options).value for v in value]
        return cls._read(value, options)

    @classmethod
    def _read(cls, value, options=None):
        """ Read a single value """
        return value

    @classmethod
    def write_value(cls, value, options=None):
        """ Called by Xlwings when writing a value (or list of values) to Excel """
        if isinstance(value, list):
            return [cls.write_value(v, options) for v in value]
        return cls._write(value, options)

    @classmethod
    def _write(cls, value, options=None):
        """ Write a single value """
        return value


class UnionConverter(Converter):
    """Allows a number of converters to be chained and return the first successful result. Hence order matters!"""

    @staticmethod
    def read_value(value, options):
        for converter in options['converters']:
            try:
                return converter.read_value(value)
            except Exception as e:
                continue
        if options.get('continue_on_fail'):
            return value
        raise ValueError(f"Failed to convert {value}")


class DateConverter(_Converter):
    """XlWings converter to use when parsing date arguments (or list of date arguments)"""

    @classmethod
    def _read(cls, value, options=None):
        if not value:
            return None
        if value == EXCEL_NA:
            return None
        return parse_date(value)


class DateTimeConverter(_Converter):
    """XlWings converter to use when parsing date arguments (or list of date arguments)"""

    @classmethod
    def _read(cls, value, options=None):
        if not value:
            return None
        if value == EXCEL_NA:
            return None
        return parse_datetime(value)


class FormattedDateConverter(DateConverter):
    """
    Let excel & xlwings handle date formatting.

    Context: When writing a string of a data (not datetime) to am excel range, excel will recognise it is date and
    cast to a date and will format the date. The is an issue when with dd/mm/yyyy vs mm/dd/yyyy date formats, since
    excel does nto know which style it is. The way excel handles this depends on a few factors.

    Solution: Parse all date strings to date objects when writing them to the XlRange. Xlwings and Excel now know this
    is a date and will the true date will be written to the sheet.
    """
    date_formats = tuple(x for x in _date_formats if x != "%Y%m%d")

    @classmethod
    def _write(cls, value, options=None):
        if value is None or isinstance(value, (Number, date, datetime)):
            return value

        try:
            return parse_date(value, formats=cls.date_formats)
        except ValueError:
            return str(value)


class TenorConverter(_Converter):
    """XlWings converter to use when parsing tenor arguments"""

    @classmethod
    def _read(cls, value, options=None):
        if isinstance(value, (int, float)) and not valid_excel_date(value):
            return str(int(value))  # return the invalid date value as a string
        return value


class ForwardQuotesConverter(TenorConverter):
    """
    Reads a table of quotes, returning a sanitised list of lists suitable for ForwardQuotes.from_table()
    """

    @classmethod
    @replay_logger_converter
    def read_value(cls, value, options=None):
        clean_value = RemoveBlanks.read_value(value).value
        # Check if the first row holds the load shape labels
        try:
            LoadShape.parse(clean_value[0][1])
        except Exception as e:
            if len(clean_value[0]) == 2:
                clean_value = [['', 'BASE']] + clean_value
            else:
                raise ValueError("Quotes provided are missing load shape labels") from e

        converted_values = [
                               clean_value[0]
                           ] + [
                               [cls._read(t), *v] for t, *v in clean_value[1:]
                           ]
        return converted_values


class CacheConverterDict(Converter):
    """Use when you have a key to a label that is stored in the cache."""

    @staticmethod
    @replay_logger_converter
    def read_value(value, *_):
        clean_value = RemoveBlanks.read_value(value).value
        if len(clean_value) == 1 and len(clean_value[0]) == 1:
            return {"BASE": CACHE.lookup_by_label(clean_value[0][0])}
        else:
            return {item[0]: CACHE.lookup_by_label(item[1]) for item in clean_value}


class CacheConverterList(Converter):
    """Use when you want to return a list of objects from the cache. This handles
    cases of list of labels or single labels."""

    @staticmethod
    def parse_handles(value):
        results = []
        if value is not None:
            if isinstance(value, list):
                clean_value = RemoveBlanks.read_value(value).value
                for item in clean_value:
                    if isinstance(item, list):
                        results.append(CACHE.lookup_by_label(item[0]))
                    else:
                        results.append(CACHE.lookup_by_label(item))
            else:
                results.append(CACHE.lookup_by_label(value))
        return results

    @staticmethod
    @replay_logger_converter
    def read_value(value, *_):
        return CacheConverterList.parse_handles(value)


class CacheConverterListOfList(Converter):
    """Takes a list of list of cache labels are returns a list of objects from the cache."""

    @staticmethod
    @replay_logger_converter
    def read_value(value, *_):
        results = []
        clean_rows = RemoveBlanks.read_value(value).value
        for row in clean_rows:
            for item in row:
                results.append(CACHE.lookup_by_label(item))
        return results


class IndicatorListCacheConverter(Converter):
    @staticmethod
    @replay_logger_converter
    def read_value(value, *_):
        items = CacheConverterList.parse_handles(value)
        results = []
        for item in items:
            if isinstance(item, pd.DataFrame):
                results.extend(CacheConverterList.parse_handles(list(item.index)))
            else:
                results.append(item)
        return results


class LoadShapedTableConverter(Converter):
    """Converts a sparse matrix of load shaped tables from excel."""

    @staticmethod
    @replay_logger_converter
    def read_value(value, *_):
        value = RemoveBlanks.read_value(value).value
        load_shaped_values = defaultdict(dict)
        load_shapes = [LoadShape.parse(ls_name.upper()) for ls_name in value[0][1:]]
        for row in value[1:]:
            row_label, *row_values = row
            for ls, row_value in zip(load_shapes, row_values):
                if row_value:
                    load_shaped_values[ls][row_label] = row_value
        return dict(load_shaped_values)


class RemoveBlanks(Converter):
    """
    Given a list of lists (matrix): remove rows and columns if the entire row or column contains all Nones.
    Given a row or column remove nones.
    """
    _BLANK_VALUES = {None, EXCEL_NA, ""}  # Possible values for a read operation

    @classmethod
    def is_invalid(cls, value, options=None):
        return value in cls._BLANK_VALUES

    @classmethod
    @replay_logger_converter
    def read_value(cls, value, options=None):
        def remove_invalid_rows(table):
            return [row for row in table if not all(cls.is_invalid(r) for r in row)]

        def transpose(x):
            return list(map(list, zip(*x)))

        if all(isinstance(v, list) for v in value):
            value = remove_invalid_rows(value)
            value = remove_invalid_rows(transpose(value))
            return transpose(value)

        return [item for item in value if not cls.is_invalid(item)]


class RemoveNA(RemoveBlanks):
    _BLANK_VALUES = {None, EXCEL_NA}


class NamedDataFrameOrCacheConverter(Converter):

    @staticmethod
    @replay_logger_converter
    def read_value(value, *_):
        if isinstance(value, str):
            return CACHE.lookup_by_label(value)
        else:
            name = value[0][0]
            columns = value[0][1:]
            values = RemoveNA.read_value(value[1:]).value
            index = [x[0] for x in values]
            data = [x[1:] for x in values]
            return NamedDataFrame(name, pd.DataFrame(data, index=index, columns=columns))


class JsonConverter(Converter):

    @staticmethod
    def json_value(v):
        if isinstance(v, date):
            return v.strftime("%d/%m/%Y")
        try:
            v_float = float(v)
            # We want to use integer representations of values where possible
            # Otherwise we will write as a float string (e.g. '10.0') which then can't be parsed as an integer
            return int(v_float) if int(v_float) == v_float else v_float
        except Exception:
            return v

    @classmethod
    def _post_process(cls, data_dict):
        # post process to gather values and units
        unit_keys = [key for key in data_dict.keys() if "UNIT" in key]
        for unit_key in unit_keys:
            value_key = unit_key.replace("UNIT", "").strip()
            data_dict[value_key] = {"VALUE": data_dict[value_key], "UNIT": data_dict[unit_key]}
            del data_dict[unit_key]
        return data_dict

    @classmethod
    @replay_logger_converter
    def read_value(cls, value, *_):
        json_objs = []
        dict_keys = [h.upper() for h in value[0]]
        for row in value[1:]:
            if any(r is not None for r in row):
                d = {k.upper(): cls.json_value(v) for k, v in zip(dict_keys, row) if v is not None}
                d = cls._post_process(d)
                json_objs.append(d)
        return json_objs


class BooleanConverter(_Converter):

    @classmethod
    def _read(cls, value, options=None):
        ignore_nones = options.get("ignore_nones", False) if options else False
        if ignore_nones and value is None:
            return None
        return to_bool(value)


class DelimitedListConverter(_Converter):

    @classmethod
    def _read(cls, value, options=None):
        """
        Convert a delimited list to a python list given the delimiter and reference to a parser function.
        n.b for Enums inheriting from ParseableNamedEnum, parser_fn can just reference the class e.g TradeStoreStatus

        :param value: Cell value to convert
        :param options:
        :return: List
        """
        _options = options or {}
        delimiter = _options.get("list_delimiter", ",")
        parser_fn = _options.get("parser_fn", str)

        if not value or value == EXCEL_NA:
            return []

        try:
            result = [parser_fn(i) for i in value.split(delimiter)]
        except ValueError as e:
            raise InvalidUserInputError(str(e))

        return result


class InfinityConverter(_Converter):
    """
    Read and write +ve/-ve infinity to Excel.
    By default Excel writes both +ve/-ve infinity as 65535.00
    """
    _INF_VALUES = {"inf", "-inf", "'inf", "'-inf"}

    @classmethod
    def _read(cls, value, options=None):
        if isinstance(value, str) and value in cls._INF_VALUES:
            return float(value.lstrip("'"))
        else:
            return value

    @classmethod
    def _write(cls, value, options=None):
        if isinstance(value, float):
            if value == math.inf:
                return "inf"
            elif value == -math.inf:
                return "'-inf"
        return value


class ParseableNamedEnumConverter(_Converter):
    target_type_key = "target_type"

    @classmethod
    def _read(cls, value, options=None):
        ignore_nones = options.get("ignore_nones", False) if options else False
        if ignore_nones and value is None:
            return None
        return options[cls.target_type_key].parse(value)

    @classmethod
    def _write(cls, value, options=None):
        return None if value is None else str(value.value)

    @classmethod
    def read_value(cls, value, options=None):
        if not options or ParseableNamedEnumConverter.target_type_key not in options:
            raise ValueError(f"Must provide a '{ParseableNamedEnumConverter.target_type_key}' in options dict")

        if not issubclass(options[cls.target_type_key], ParseableNamedEnum):
            raise TypeError(f"'{cls.target_type_key}', must inherit from ParseableNamedEnum."
                            f"Got '{options[cls.target_type_key]}' instead.")

        return super().read_value(value, options)


