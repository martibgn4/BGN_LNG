import inspect
import time
from base64 import b64encode
from functools import wraps
from uuid import uuid4

from xlwings.conversion import Converter
from xlwings.udfs import xlfunc

from file_utils import pickle_dumps
from thorn_excel.utils.replay_logger_utils import write_cache, replay_logger_converter
from value_object import is_value_object


def is_immutable(arg):
    """Immutability here is in the sense of XlWings.  Will Xlwings give us a new object from
    the inputs made into excel? or will it pass us back a cached object from a handle."""
    if isinstance(arg, list):
        return all(is_immutable(i) for i in arg)
    elif isinstance(arg, dict):
        return all(all([is_immutable(k), is_immutable(v)]) for k, v in arg.items())
    return is_value_object(type(arg))


class Cache:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        self.guid_to_object = {}
        self.inputs_to_guid = {}
        self.label_to_guid = {}
        self.guid_to_runtime = {}

    def __contains__(self, item):
        return item in self.label_to_guid

    def lookup_by_label(self, label):
        guid = self.label_to_guid[label]
        return self.guid_to_object[guid]

    def lookup_by_label_if_exists(self, label):
        try:
            return self.lookup_by_label(label)
        except KeyError:
            return None

    def lookup_label_for_inputs(self, function_name, args, label=None):
        hashed_args = self._hashed_args(args)
        guid = self.inputs_to_guid[(function_name, hashed_args)]
        if label:
            self.label_to_guid[label] = guid
        else:
            label = guid
        return label

    @staticmethod
    def _create_guid(prefix):
        if prefix:
            prefix = prefix + "_"
        return f"{prefix}{str(uuid4())[:8]}"

    @staticmethod
    def _hashed_args(args):
        args_to_hash = []
        for arg in args:
            if is_immutable(arg):
                args_to_hash.append(arg)
            else:
                args_to_hash.append(id(arg))

        return b64encode(pickle_dumps(args_to_hash))

    def add_entry(self, func_name, args, value, label_prefix=None, label=None, runtime=None):
        guid = CACHE._create_guid(label_prefix)
        if label is None:
            label = guid

        hashed_args = self._hashed_args(args)
        input_hash = (func_name, hashed_args)
        self.guid_to_object[guid] = value
        self.inputs_to_guid[input_hash] = guid
        self.label_to_guid[label] = guid
        self.guid_to_runtime[guid] = runtime

        return label

    def clear(self):
        """Remove all keys from the cache"""
        self.guid_to_object = {}
        self.inputs_to_guid = {}
        self.label_to_guid = {}
        self.guid_to_runtime = {}

    def pop(self, label, default=None):
        """Remove entries for label from the Cache """
        guid_for_label = self.label_to_guid.pop(label, None)
        if guid_for_label is not None:
            _ = self.guid_to_runtime.pop(guid_for_label)
            obj = self.guid_to_object.pop(guid_for_label)
            input_hashes = {input_hash for input_hash, guid in self.inputs_to_guid.items() if guid == guid_for_label}
            for input_hash in input_hashes:
                self.inputs_to_guid.pop(input_hash)
            return obj
        else:
            return default


CACHE = Cache()


class CacheConverter(Converter):
    """XlWings converter to use when parsing cache handlers"""

    @staticmethod
    @replay_logger_converter
    def read_value(value, *_):
        if value:
            return CACHE.lookup_by_label(value)
        return None


def get_cache_label(f, args):
    signature = inspect.signature(f)
    kwargs = dict(zip(signature.parameters, args))
    return kwargs.get('cache_label')


def add_cache_function_entry(f, args, label_prefix, cache_label):
    """Add the results of a function call into the cache"""
    start_time = time.time()
    return_val = f(*args)
    runtime = time.time() - start_time
    label = CACHE.add_entry(f.__name__, args, return_val, label_prefix, cache_label, runtime)
    write_cache(CACHE)
    return label


def add_cache_entry(value, cache_label, do_write_cache=True):
    """Add a value into the Cache under a label"""
    CACHE.add_entry("", (), value, label=cache_label)
    if do_write_cache:
        write_cache(CACHE)


def clear_cache():
    """Delete all items from the Cache"""
    CACHE.clear()
    write_cache(CACHE)


def pop_entry(label, default=None):
    """Pop an item from the Cache"""
    value = CACHE.pop(label, default)
    write_cache(CACHE)
    return value


def volatile_results(label_prefix=""):
    """
    Stores results in cache but expects those values to change - hence always recalcs.
    Similar to a volatile function in Excel.
    """

    def inner(f):
        orig_xlfunc = xlfunc()(f)

        @wraps(f)
        def volatile_wrapper(*args):
            cache_label = get_cache_label(f, args)
            return add_cache_function_entry(f, args, label_prefix, cache_label)

        cache_xlfunc = xlfunc()(volatile_wrapper)
        cache_xlfunc.__xlfunc__ = orig_xlfunc.__xlfunc__

        return cache_xlfunc

    return inner


def cache_results(label_prefix=""):
    def inner(f):
        orig_xlfunc = xlfunc()(f)

        @wraps(f)
        def cache_wrapper(*args):
            cache_label = get_cache_label(f, args)
            try:
                return CACHE.lookup_label_for_inputs(f.__name__, args, cache_label)
            except KeyError:
                return add_cache_function_entry(f, args, label_prefix, cache_label)

        cache_xlfunc = xlfunc()(cache_wrapper)
        cache_xlfunc.__xlfunc__ = orig_xlfunc.__xlfunc__

        return cache_xlfunc

    return inner


