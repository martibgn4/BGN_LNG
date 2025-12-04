from enum import auto

from general_utils import ParseableNamedEnum

__all__ = [
    "OPTION_TYPES",
    "OptionType",
]


class OptionType(ParseableNamedEnum):
    """ PUT / CALL """
    CALL = auto()
    PUT = auto()

    @property
    def inverse(self):
        return {self.PUT: self.CALL, self.CALL: self.PUT}[self]

    @classmethod
    def create(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):  # Handle legacy "is_call" behaviour
            return cls.CALL if value else cls.PUT
        return cls.parse(value)

    @classmethod
    def parse(cls, string):
        if isinstance(string, str) and len(string) == 1:
            if string.upper() == "C":
                return OptionType.CALL
            elif string.upper() == "P":
                return OptionType.PUT
        return super().parse(string)


OPTION_TYPES = (OptionType.CALL, OptionType.PUT)


