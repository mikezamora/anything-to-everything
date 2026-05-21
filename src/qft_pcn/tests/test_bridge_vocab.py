"""Tests for bridge/dsl/vocab.py (spec §3.5, §4.1)."""

from __future__ import annotations

import pytest

from src.qft_pcn.bridge.dsl.vocab import resolve_basis, CANONICAL_FIELDS


def test_canonical_field_set():
    assert CANONICAL_FIELDS == ("kind", "type", "bid", "value")


def test_resolve_kind_lam():
    assert resolve_basis("kind", "KIND_LAM") == 2


def test_resolve_type_int():
    assert resolve_basis("type", "T_INT") == 1


def test_resolve_value_true():
    assert resolve_basis("value", "V_TRUE") == 1


def test_resolve_bid_b0():
    assert resolve_basis("bid", "B_0") == 1


def test_resolve_int_literal_value():
    # V_INT(3) -> 3 + INT_LIT_OFFSET = 10
    assert resolve_basis("value", "V_INT(3)") == 10


def test_resolve_int_literal_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        resolve_basis("value", "V_INT(99)")


def test_resolve_passes_through_ints():
    assert resolve_basis("kind", 3) == 3
    assert resolve_basis("kind", 7) == 7


def test_resolve_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown basis name"):
        resolve_basis("kind", "NOT_A_THING")


def test_resolve_non_canonical_field_only_accepts_ints():
    assert resolve_basis("custom_field", 4) == 4
    with pytest.raises(ValueError, match="non-canonical field"):
        resolve_basis("custom_field", "KIND_LAM")
