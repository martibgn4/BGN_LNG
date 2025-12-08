import subprocess
from collections import namedtuple
from functools import wraps

import datetime
import logging
import openpyxl
import os
import pickle
import xlwings as xw

from xlwings.udfs import xlfunc

from file_utils import write_pickle_file
from official_config.version import __version__ as config_version
from thorn_excel.version import __version__ as version


COUNTER = 0
LOGGER = None
QALOGLEVEL = "QALOGLEVEL"

LOG_PATH = os.environ['TEMP']
PID = os.getpid()

date_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CACHE_FILE = os.path.join(LOG_PATH, f"xlwings_cache_{date_stamp}_{PID}.pkl")
LOG_FILE = os.path.join(LOG_PATH, f"xlwings_replay_{date_stamp}_{PID}.log")

ConvertedValue = namedtuple('ConvertedValue', ['value', 'raw'])


def write_cache(cache):
    write_pickle_file(cache, CACHE_FILE)


def _get_logger():
    """ Use of a local function make patching (i.e. redirect) more robust when wrapping functions outside this file """
    global LOGGER

    if not LOGGER:
        LOGGER = logging.getLogger()
        LOGGER.setLevel(int(os.getenv(QALOGLEVEL, logging.INFO)))
        handler = logging.FileHandler(LOG_FILE)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        LOGGER.addHandler(handler)
        LOGGER.info(f"Thorn Desktop Version Started")
        LOGGER.info(f"Thorn Desktop Version ({version}) "
                    f"Tool Version ({tool_version()}) "
                    f"Config Version ({config_version})")
    return LOGGER


def get_logger():
    return _get_logger()  # Allows more robust patching (i.e. redirect) of get_logger()


def replay_logger(raise_exceptions=False):
    def inner(f):
        orig_xlfunc = xlfunc()(f)

        @wraps(f)
        def wrapper(*args):
            logger = get_logger()

            global COUNTER
            raw_args = [arg.raw if isinstance(arg, ConvertedValue) else arg for arg in args]
            logger.info(f"COMMAND({COUNTER}): {orig_xlfunc.__xlfunc__['name']}{tuple(raw_args)}".replace("\n", " "))
            new_args = [arg.value if isinstance(arg, ConvertedValue) else arg for arg in args]

            try:
                return_val = f(*new_args)
            except Exception as e:
                logger.exception(f"EXCEPTION({COUNTER}): {e.__class__.__name__}: {e}")
                if raise_exceptions:
                    raise
                else:
                    msg = f"ERROR: {e}\n"
                    # workaround: https://github.com/ZoomerAnalytics/xlwings/issues/456 - not worth monkey patching yet
                    if len(msg) > 255:
                        msg = msg[:250] + ' ...'
                    return msg

            logger.info(f"RETURN({COUNTER}): {return_val}")
            COUNTER += 1
            return return_val

        wrapper_xlfunc = xlfunc()(wrapper)
        wrapper_xlfunc.__xlfunc__ = orig_xlfunc.__xlfunc__

        return wrapper_xlfunc
    return inner


class ConverterRaw:
    def __init__(self, func, data):
        self.func = func
        self.data = data

    def __repr__(self):
        return f"{self.func}(" + repr(self.data) + ")"


def replay_logger_converter(f):
    @wraps(f)
    def wrapper(raw_value, *args, **kwargs):
        converted_value = f(raw_value, *args, **kwargs)
        return ConvertedValue(value=converted_value, raw=ConverterRaw(f.__qualname__, raw_value))

    return wrapper


def tool_version():
    try:
        xw_book = xw.Book.caller()
        archive = openpyxl.reader.excel._validate_archive(xw_book.fullname)
        src = openpyxl.xml.functions.fromstring(archive.read(openpyxl.xml.constants.ARC_CORE))
        wb_properties = openpyxl.packaging.core.DocumentProperties.from_tree(src)
        version = wb_properties.version
        if version is not None:
            return version
    except Exception:
        pass
    return "Unversioned"


def tail_logs():  # pragma: no cover
    subprocess.Popen(
        f"powershell.exe Get-Content {LOG_FILE} -Wait -Tail 10",
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )


