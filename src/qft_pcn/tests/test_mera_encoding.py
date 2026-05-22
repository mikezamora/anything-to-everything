"""Tests for MERA encoding constants (spec §4, §8.1)."""
from __future__ import annotations
from src.qft_pcn.logic.mera_encoding import (
    MERA_LEAF_DIM, MERA_KIND_CUTOFF, MERA_TYPE_CUTOFF,
    SPECIES_ORDER, SPECIES_LEAF_OFFSET, LEAVES_PER_NODE,
    KIND_ZERO, KIND_SUCC, KIND_NATLIT, KIND_NIL, KIND_CONS,
    KIND_EQ, KIND_FORALL, KIND_FIX,
    TYPE_NAT, TYPE_LIST, TYPE_EQ, TYPE_PROP,
)


def test_leaf_dim_is_16():
    assert MERA_LEAF_DIM == 16


def test_kind_and_type_cutoffs_fit_in_leaf():
    assert MERA_KIND_CUTOFF == 16
    assert MERA_TYPE_CUTOFF == 16


def test_species_order_is_canonical():
    assert SPECIES_ORDER == ("kind", "type", "bid", "value", "tobl")
    assert LEAVES_PER_NODE == 5


def test_species_leaf_offset():
    assert SPECIES_LEAF_OFFSET == {
        "kind": 0, "type": 1, "bid": 2, "value": 3, "tobl": 4,
    }


def test_extended_kind_indices_are_8_through_15():
    indices = [KIND_ZERO, KIND_SUCC, KIND_NATLIT, KIND_NIL,
               KIND_CONS, KIND_EQ, KIND_FORALL, KIND_FIX]
    assert indices == [8, 9, 10, 11, 12, 13, 14, 15]


def test_extended_type_indices_within_16():
    for t in (TYPE_NAT, TYPE_LIST, TYPE_EQ, TYPE_PROP):
        assert 0 <= t < 16


def test_existing_kind_constants_unchanged():
    # mera_encoding re-exports the base kinds unchanged from encoding.py
    from src.qft_pcn.logic.encoding import KIND_PAD, KIND_VAR, KIND_BIN
    from src.qft_pcn.logic.mera_encoding import (
        KIND_PAD as M_PAD, KIND_VAR as M_VAR, KIND_BIN as M_BIN,
    )
    assert (M_PAD, M_VAR, M_BIN) == (KIND_PAD, KIND_VAR, KIND_BIN)
