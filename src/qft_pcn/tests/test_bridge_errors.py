"""Tests for BridgeError hierarchy and code stability."""

from __future__ import annotations

import pytest

from src.qft_pcn.bridge.errors import (
    BridgeError,
    BadJsonError, BadSchemaError, BadReferenceError, BadTermError,
    TypeError as BridgeTypeError, TermUnsupportedError,
    UnsupportedMethodError, SitesOutOfRangeError, TooLargeError,
    ConstraintNotAdjacentError, NumericFailureError, InternalError,
    ALL_CODES,
)


def test_bridge_error_has_code_and_details():
    e = BadJsonError(message="bad json at pos 5", details={"position": 5})
    assert e.code == "BRIDGE_E_BAD_JSON"
    assert e.details == {"position": 5}
    assert "bad json at pos 5" in str(e)


def test_every_subclass_has_distinct_code():
    codes = set()
    for cls in (BadJsonError, BadSchemaError, BadReferenceError, BadTermError,
                BridgeTypeError, TermUnsupportedError, UnsupportedMethodError,
                SitesOutOfRangeError, TooLargeError, ConstraintNotAdjacentError,
                NumericFailureError, InternalError):
        codes.add(cls.code)
    assert len(codes) == 12


def test_all_codes_set_is_stable():
    expected = {
        "BRIDGE_E_BAD_JSON",
        "BRIDGE_E_BAD_SCHEMA",
        "BRIDGE_E_BAD_REFERENCE",
        "BRIDGE_E_BAD_TERM",
        "BRIDGE_E_TYPE_ERROR",
        "BRIDGE_E_TERM_UNSUPPORTED",
        "BRIDGE_E_UNSUPPORTED_METHOD",
        "BRIDGE_E_SITES_OUT_OF_RANGE",
        "BRIDGE_E_TOO_LARGE",
        "BRIDGE_E_CONSTRAINT_NOT_ADJACENT",
        "BRIDGE_E_NUMERIC_FAILURE",
        "BRIDGE_E_INTERNAL",
    }
    assert set(ALL_CODES) == expected


def test_bridge_error_to_dict():
    e = BadSchemaError(message="missing 'sites'", details={"pointer": "/sites"})
    d = e.to_dict()
    assert d == {
        "code": "BRIDGE_E_BAD_SCHEMA",
        "message": "missing 'sites'",
        "details": {"pointer": "/sites"},
    }


def test_default_details_is_empty_dict():
    e = InternalError(message="oops")
    assert e.details == {}
