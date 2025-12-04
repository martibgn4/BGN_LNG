from enum import Enum

__all__ = [
    "AutoName",
    "ParseableNamedEnum",
]


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class ParseableNamedEnum(AutoName):
    """
    Auto named Enum with a parser, derive a new class and create members like:
      . CAL = auto()
      . SEASON = auto()

    @note: the parser is based on upper case comparison
    """

    @classmethod
    def parse(cls, string):
        _string = string if isinstance(string, str) else str(string)  # Handle the parsing of an object of the class
        _string = _string.strip().upper()
        for name, member in cls.__members__.items():
            if _string == name.upper():  # name cannot start or end with whitespace
                return member
            elif isinstance(member.value, str) and _string == member.value.upper():
                return member
        raise ValueError(f"'{string}' not in {[n for n in cls.__members__]}")

    @classmethod
    def _missing_(cls, value):
        return cls.parse(value)

    @classmethod
    def has_member(cls, member):
        try:
            _ = cls.parse(member)
        except ValueError:
            return False
        else:
            return True

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.name < other.name
        else:
            return self.__class__.__name__ < other.__class__.__name__

    def __le__(self, other):
        return self == other or self < other

    @classmethod
    def members(cls):
        return list(cls.__members__.values())

    @classmethod
    def values(cls):
        return [member.value for member in cls]


