import datetime
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import parse_date_from_string


def test_parse_iso_format():
    assert parse_date_from_string('2024-01-15') == datetime.date(2024, 1, 15)
    assert parse_date_from_string('15-Jan-2024') == datetime.date(2024, 1, 15)


def test_parse_datetime_date_repr():
    assert parse_date_from_string('datetime.date(2024, 1, 15)') == datetime.date(2024, 1, 15)


def test_parse_invalid_inputs():
    for bad in ['not a date', '', None, 123]:
        assert parse_date_from_string(bad) is None
