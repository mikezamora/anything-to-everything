"""Tests for the constraint-expression parser (spec §4.1, §4.2, §11.2)."""

from __future__ import annotations

import builtins
import pytest

from src.qft_pcn.bridge.dsl.expr_parser import parse_expr
from src.qft_pcn.bridge.dsl.expr_ast import Lit, Ident, Call, Unary, Binary
from src.qft_pcn.bridge.errors import BadTermError


def test_parse_simple_ident():
    assert parse_expr("KIND_LAM") == Ident(name="KIND_LAM")


def test_parse_int_literal():
    assert parse_expr("42") == Lit(value=42)


def test_parse_string_literal():
    assert parse_expr("'+'") == Lit(value="+")


def test_parse_call_one_arg():
    e = parse_expr("type(arg1)")
    assert e == Call(fn="type", args=[Ident(name="arg1")])


def test_parse_call_two_args():
    e = parse_expr("arr(T_INT, T_BOOL)")
    assert e == Call(fn="arr",
                     args=[Ident(name="T_INT"), Ident(name="T_BOOL")])


def test_parse_eq_with_calls():
    e = parse_expr("type(arg1) == type(arg2)")
    assert e == Binary(op="==",
                       lhs=Call(fn="type", args=[Ident("arg1")]),
                       rhs=Call(fn="type", args=[Ident("arg2")]))


def test_parse_kind_equals_lit():
    e = parse_expr("kind == LAM")
    assert e == Binary(op="==", lhs=Ident("kind"), rhs=Ident("LAM"))


def test_parse_and_left_assoc():
    e = parse_expr("a and b and c")
    assert e == Binary("and",
                       lhs=Binary("and", lhs=Ident("a"), rhs=Ident("b")),
                       rhs=Ident("c"))


def test_parse_or_lower_prec_than_and():
    e = parse_expr("a or b and c")
    assert e == Binary("or", lhs=Ident("a"),
                       rhs=Binary("and", lhs=Ident("b"), rhs=Ident("c")))


def test_parse_not_higher_prec_than_and():
    e = parse_expr("not a and b")
    assert e == Binary("and",
                       lhs=Unary(op="not", operand=Ident("a")),
                       rhs=Ident("b"))


def test_parse_arithmetic():
    e = parse_expr("n(arg1) + 1")
    assert e == Binary("+", lhs=Call("n", [Ident("arg1")]), rhs=Lit(1))


def test_parse_parens_override_precedence():
    e = parse_expr("(a or b) and c")
    assert e == Binary("and",
                       lhs=Binary("or", lhs=Ident("a"), rhs=Ident("b")),
                       rhs=Ident("c"))


def test_parse_cmp_non_associative():
    with pytest.raises(BadTermError):
        parse_expr("a == b == c")


def test_parse_bool_literals():
    assert parse_expr("true") == Lit(value=True)
    assert parse_expr("false") == Lit(value=False)


def test_parse_unary_minus():
    assert parse_expr("-3") == Unary(op="-", operand=Lit(value=3))


def test_parse_trailing_garbage_raises():
    with pytest.raises(BadTermError, match="unexpected"):
        parse_expr("a == b extra")


def test_parse_empty_raises():
    with pytest.raises(BadTermError):
        parse_expr("")


def test_parse_unclosed_paren_raises():
    with pytest.raises(BadTermError):
        parse_expr("(a == b")


def test_parser_does_not_call_eval(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("parser invoked eval/exec/compile!")
    monkeypatch.setattr(builtins, "eval", boom)
    monkeypatch.setattr(builtins, "exec", boom)
    monkeypatch.setattr(builtins, "compile", boom)
    for s in ["a", "type(arg1) == type(arg2)", "a and b", "n(x) + 1",
              "(a or b) and not c", "kind == '+'"]:
        parse_expr(s)


def test_bad_term_error_carries_position():
    try:
        parse_expr("a + + b")
    except BadTermError as e:
        assert "position" in str(e) or "pos" in e.details
    else:
        pytest.fail("expected BadTermError")
