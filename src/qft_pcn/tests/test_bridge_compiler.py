"""Tests for the constraint-term compiler (spec §4.3, §11.3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.bridge.dsl.expr_parser import parse_expr
from src.qft_pcn.bridge.dsl.compiler import compile_constraint
from src.qft_pcn.bridge.dsl.term import LocalTerm, TwoSiteTerm, FieldSpec
from src.qft_pcn.bridge.errors import (
    TermUnsupportedError, TypeError as BridgeTypeError,
)


def _canon_fields() -> list[FieldSpec]:
    return [
        FieldSpec(name="kind",  cutoff=8),
        FieldSpec(name="type",  cutoff=8),
        FieldSpec(name="bid",   cutoff=8),
        FieldSpec(name="value", cutoff=16),
    ]


def test_kind_equals_LAM_yields_local_projector():
    expr = parse_expr("kind == LAM")
    terms = compile_constraint(expr, kind="local", site=0,
                               fields=_canon_fields(), weight=1.0)
    assert len(terms) == 1
    t = terms[0]
    assert isinstance(t, LocalTerm)
    assert t.site == 0
    # Term is w * (I - P_LAM); trace(I-P) = 8192 - 8*8*16 = 8192 - 1024 = 7168
    assert np.allclose(t.operator.T.conj(), t.operator)
    assert np.isclose(np.trace(t.operator).real, 8192 - 1024)


def _small_fields() -> list[FieldSpec]:
    # d_local = 4*4 = 16, d_local^2 = 256 -> 256x256 complex matrix is feasible.
    return [FieldSpec("kind", 4), FieldSpec("type", 4)]


def test_two_site_type_equality_yields_two_site_term():
    expr = parse_expr("type(arg1) == type(arg2)")
    terms = compile_constraint(expr, kind="two_site", sites=(5, 7),
                               fields=_small_fields(), weight=1.0)
    assert len(terms) == 1
    t = terms[0]
    assert isinstance(t, TwoSiteTerm)
    assert t.sites == (5, 7)
    assert np.allclose(t.operator.T.conj(), t.operator)


def test_and_combines_to_term_list():
    expr = parse_expr("kind == LAM and type(arg1) == T_INT")
    terms = compile_constraint(expr, kind="local", site=3,
                               fields=_canon_fields(), weight=2.0)
    assert len(terms) == 2
    assert all(isinstance(t, LocalTerm) and t.site == 3 for t in terms)


def test_unsupported_pattern_raises():
    expr = parse_expr("kind(arg1) * value(arg1)")
    with pytest.raises(TermUnsupportedError):
        compile_constraint(expr, kind="local", site=0,
                           fields=_canon_fields(), weight=1.0)


def test_local_constraint_referencing_arg2_raises_type_error():
    expr = parse_expr("type(arg2) == T_INT")
    with pytest.raises(BridgeTypeError, match="arg2"):
        compile_constraint(expr, kind="local", site=0,
                           fields=_canon_fields(), weight=1.0)


def test_unknown_field_in_term_raises_type_error():
    expr = parse_expr("zzz == LAM")
    with pytest.raises(BridgeTypeError, match="zzz"):
        compile_constraint(expr, kind="local", site=0,
                           fields=_canon_fields(), weight=1.0)


def test_weight_scales_operator():
    e1 = parse_expr("kind == LAM")
    t_unit = compile_constraint(e1, kind="local", site=0,
                                fields=_canon_fields(), weight=1.0)[0]
    t_two  = compile_constraint(e1, kind="local", site=0,
                                fields=_canon_fields(), weight=2.0)[0]
    assert np.allclose(t_two.operator, 2.0 * t_unit.operator)


def test_true_lit_yields_no_terms():
    expr = parse_expr("true")
    terms = compile_constraint(expr, kind="local", site=0,
                               fields=_canon_fields(), weight=1.0)
    assert terms == []
