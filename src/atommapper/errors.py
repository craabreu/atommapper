"""Exceptions shared by every layer."""


class InvalidInput(ValueError):
    """An input that cannot be parsed or sanitized."""


class UnsupportedInput(Exception):
    """A valid input that this version cannot represent faithfully."""
