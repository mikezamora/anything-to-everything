"""Tests for logic/encoding.py constants and metadata classes."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL, KIND_IF,
    KIND_BIN, KIND_CUTOFF,
    TYPE_NONE, TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI,
    TYPE_ARR_BB, TYPE_ARR_NESTED, TYPE_CUTOFF,
    BID_NONE, BID_0, BID_1, BID_2, BID_3, BID_4, BID_5, BID_6, BID_CUTOFF,
    VALUE_NONE, VALUE_FALSE, VALUE_TRUE,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ, VALUE_CUTOFF,
    INT_LIT_OFFSET, INT_LIT_MIN, INT_LIT_MAX,
    SPECIES, SPECIES_NAMES, SPECIES_DIMS, D_LOCAL,
    BinderHandle, EncodingMeta,
    EncodingError, EncodingTooLarge, TooManyBinders,
    IntLiteralOutOfRange, IllScopedVar, UnsupportedNode, DecodeError,
)


def test_kind_cutoff_and_basis():
    assert KIND_PAD == 0
    assert KIND_VAR == 1
    assert KIND_LAM == 2
    assert KIND_APP == 3
    assert KIND_INT == 4
    assert KIND_BOOL == 5
    assert KIND_IF == 6
    assert KIND_BIN == 7
    assert KIND_CUTOFF == 8


def test_type_cutoff_and_basis():
    assert TYPE_NONE == 0
    assert TYPE_INT == 1
    assert TYPE_BOOL == 2
    assert TYPE_ARR_II == 3
    assert TYPE_ARR_IB == 4
    assert TYPE_ARR_BI == 5
    assert TYPE_ARR_BB == 6
    assert TYPE_ARR_NESTED == 7
    assert TYPE_CUTOFF == 8


def test_bid_cutoff_and_basis():
    assert BID_NONE == 0
    assert BID_0 == 1
    assert BID_1 == 2
    assert BID_2 == 3
    assert BID_3 == 4
    assert BID_4 == 5
    assert BID_5 == 6
    assert BID_6 == 7
    assert BID_CUTOFF == 8


def test_value_cutoff_and_basis():
    assert VALUE_NONE == 0
    # int 0..15 are at value indices 7..15 via INT_LIT_OFFSET=7
    assert VALUE_FALSE == 0
    assert VALUE_TRUE == 1
    assert VALUE_PLUS == 2
    assert VALUE_MINUS == 3
    assert VALUE_TIMES == 4
    assert VALUE_LT == 5
    assert VALUE_EQ == 6
    # 7..15 used for int literals (offset 7 mapping [-7, 8] -> [0, 15])
    assert INT_LIT_OFFSET == 7
    assert INT_LIT_MIN == -7
    assert INT_LIT_MAX == 8
    assert VALUE_CUTOFF == 16


def test_species_metadata():
    assert SPECIES_NAMES == ("kind", "type", "bid", "value")
    assert SPECIES_DIMS == (8, 8, 8, 16)
    assert D_LOCAL == 8 * 8 * 8 * 16


def test_species_objects_are_field_species():
    from src.qft_pcn.qft.hamiltonian import FieldSpecies
    assert len(SPECIES) == 4
    for s in SPECIES:
        assert isinstance(s, FieldSpecies)
    assert [s.name for s in SPECIES] == list(SPECIES_NAMES)
    assert [s.cutoff for s in SPECIES] == list(SPECIES_DIMS)


def test_binder_handle_frozen_and_hashable():
    bh = BinderHandle(lam_site=3, depth_at_lam=2)
    assert bh.lam_site == 3
    assert bh.depth_at_lam == 2
    # Frozen -> hashable.
    s = {bh}
    assert bh in s


def test_encoding_meta_fields():
    meta = EncodingMeta(
        N=32,
        chi_max=16,
        field_dims={"kind": 8, "type": 8, "bid": 8, "value": 16},
        species=list(SPECIES),
        nested_type_index={},
        site_to_ast_path={},
        live_binders_per_bond=[],
    )
    assert meta.N == 32
    assert meta.chi_max == 16


def test_exception_types_inherit_from_encoding_error():
    for cls in (EncodingTooLarge, TooManyBinders, IntLiteralOutOfRange,
                IllScopedVar, UnsupportedNode, DecodeError):
        assert issubclass(cls, EncodingError)


def test_too_large_error_message():
    with pytest.raises(EncodingTooLarge, match="AST has 50 nodes but N=32"):
        raise EncodingTooLarge(n_nodes=50, N=32)


def test_too_many_binders_message():
    with pytest.raises(TooManyBinders, match="scope nesting 8 exceeds bid cutoff 8"):
        raise TooManyBinders(depth=8, cutoff=8)


def test_int_literal_out_of_range_message():
    with pytest.raises(IntLiteralOutOfRange, match=r"IntLit\(9\) outside"):
        raise IntLiteralOutOfRange(n=9)


def test_ill_scoped_var_message():
    with pytest.raises(IllScopedVar, match="Var\\('x'\\) not in lexical scope"):
        raise IllScopedVar(name="x")


def test_tobl_basis_mirrors_type():
    """Spec §4.2: tobl basis is 1:1 with type basis."""
    from src.qft_pcn.logic.encoding import (
        TOBL_NONE, TOBL_INT, TOBL_BOOL,
        TOBL_ARR_II, TOBL_ARR_IB, TOBL_ARR_BI, TOBL_ARR_BB,
        TOBL_ARR_NESTED, TOBL_CUTOFF,
    )
    assert TOBL_NONE == TYPE_NONE == 0
    assert TOBL_INT == TYPE_INT == 1
    assert TOBL_BOOL == TYPE_BOOL == 2
    assert TOBL_ARR_II == TYPE_ARR_II == 3
    assert TOBL_ARR_IB == TYPE_ARR_IB == 4
    assert TOBL_ARR_BI == TYPE_ARR_BI == 5
    assert TOBL_ARR_BB == TYPE_ARR_BB == 6
    assert TOBL_ARR_NESTED == TYPE_ARR_NESTED == 7
    assert TOBL_CUTOFF == TYPE_CUTOFF == 8


def test_species_list_has_tobl_as_5th():
    """Spec §4.1: SPECIES is 5 species, with tobl appended last."""
    from src.qft_pcn.logic.encoding import SPECIES, D_LOCAL
    names = [s.name for s in SPECIES]
    assert names == ["kind", "type", "bid", "value", "tobl"]
    assert SPECIES[-1].cutoff == 8
    # d_local = 8 * 8 * 8 * 16 * 8 = 65536.
    assert D_LOCAL == 65536


def test_species_dims_tuple():
    from src.qft_pcn.logic.encoding import SPECIES_DIMS, SPECIES_NAMES
    assert SPECIES_NAMES == ("kind", "type", "bid", "value", "tobl")
    assert SPECIES_DIMS == (8, 8, 8, 16, 8)
