from enum import auto
from unittest import TestCase

from general_utils import ParseableNamedEnum


class MockEnum(ParseableNamedEnum):
    A = auto()
    b = auto()


class NamedEnum(ParseableNamedEnum):
    First = 'one'
    Second = 'tWo'


class TestParseableNamedEnum(TestCase):

    def test_parse(self):
        self.assertEqual(MockEnum.A, MockEnum.parse('A'))
        self.assertEqual(MockEnum.A, MockEnum.parse('a'))
        self.assertNotEqual(MockEnum.b, MockEnum.parse('a'))
        self.assertEqual(MockEnum.b, MockEnum.parse(' b '))   # Note whitespace
        self.assertEqual(MockEnum.b, MockEnum.parse(MockEnum.b))

        self.assertEqual(NamedEnum.First, NamedEnum.parse("one"))
        self.assertEqual(NamedEnum.Second, NamedEnum.parse("two"))
        self.assertEqual(NamedEnum.Second, NamedEnum.parse(" TwO "))

        with self.assertRaises(ValueError):
            _ = MockEnum.parse("ab")

    def test_cast(self):
        self.assertEqual(MockEnum.A, MockEnum('A'))
        self.assertEqual(MockEnum.A, MockEnum('a'))
        self.assertNotEqual(MockEnum.b, MockEnum('a'))
        self.assertEqual(MockEnum.b, MockEnum('b '))  # Note whitespace
        self.assertEqual(MockEnum.b, MockEnum(MockEnum.b))
        self.assertIs(MockEnum.b, MockEnum(MockEnum.b))

        self.assertEqual(NamedEnum.First, NamedEnum("one"))
        self.assertEqual(NamedEnum.Second, NamedEnum("two"))
        self.assertIs(NamedEnum.Second, NamedEnum("two"))

        with self.assertRaises(ValueError):
            _ = MockEnum("ab")

    def test_str(self):
        self.assertEqual(str(MockEnum.A), 'A')

    def test_repr(self):
        self.assertEqual(repr(MockEnum.A), 'A')

    def test_lt(self):
        self.assertTrue(MockEnum.A < MockEnum.b)
        self.assertTrue(MockEnum.b > MockEnum.A)

    def test_lt_heterogenous(self):
        self.assertTrue(MockEnum.A < self)

    def test_le(self):
        self.assertTrue(MockEnum.A <= MockEnum.A)
        self.assertTrue(MockEnum.A <= MockEnum.b)

    def test_members(self):
        self.assertEqual([MockEnum.A, MockEnum.b], MockEnum.members())
        self.assertEqual([NamedEnum.First, NamedEnum.Second], NamedEnum.members())
        self.assertNotEqual(["one", "two"], NamedEnum.members())

    def test_values(self):
        self.assertEqual([MockEnum.A.value, MockEnum.b.value], MockEnum.values())
        self.assertEqual([NamedEnum.First.value, NamedEnum.Second.value], NamedEnum.values())

    def test_members_vs_values(self):
        for klass in [MockEnum, NamedEnum]:
            self.assertNotEqual(klass.members(), klass.values())

    def test_has_member(self):
        self.assertTrue(MockEnum.has_member("A"))
        self.assertTrue(MockEnum.has_member("a"))
        self.assertTrue(MockEnum.has_member(MockEnum.A))
        self.assertTrue(MockEnum.has_member("b"))
        self.assertTrue(MockEnum.has_member("B"))
        self.assertTrue(MockEnum.has_member(MockEnum.b))
        self.assertFalse(MockEnum.has_member("c"))
        self.assertFalse(MockEnum.has_member("C"))


