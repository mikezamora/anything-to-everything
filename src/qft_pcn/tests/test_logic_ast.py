"""Tests for the AST data classes in logic/ast.py."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin,
    Ty, TInt, TBool, TArrow,
)


def test_var_construction():
    v = Var(name="x")
    assert v.name == "x"


def test_lam_construction_holds_param_and_body():
    lam = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    assert lam.param == "x"
    assert isinstance(lam.param_ty, TInt)
    assert isinstance(lam.body, Var)
    assert lam.body.name == "x"


def test_app_holds_fn_and_arg():
    app = App(fn=Var(name="f"), arg=IntLit(val=3))
    assert isinstance(app.fn, Var)
    assert app.arg.val == 3


def test_intlit_and_boollit():
    assert IntLit(val=5).val == 5
    assert BoolLit(val=True).val is True
    assert BoolLit(val=False).val is False


def test_if_holds_three_branches():
    e = If(cond=BoolLit(val=True), then_b=IntLit(val=1), else_b=IntLit(val=0))
    assert e.cond.val is True
    assert e.then_b.val == 1
    assert e.else_b.val == 0


def test_bin_holds_op_and_operands():
    b = Bin(op="+", lhs=IntLit(val=2), rhs=IntLit(val=3))
    assert b.op == "+"
    assert b.lhs.val == 2 and b.rhs.val == 3


def test_bin_op_must_be_supported():
    # Out-of-grammar op should be a runtime error from __post_init__.
    with pytest.raises(ValueError, match="op must be one of"):
        Bin(op="**", lhs=IntLit(val=1), rhs=IntLit(val=1))


def test_arrow_type_construction():
    t = TArrow(src=TInt(), dst=TBool())
    assert isinstance(t.src, TInt)
    assert isinstance(t.dst, TBool)


def test_ty_equality_by_structure():
    assert TInt() == TInt()
    assert TBool() == TBool()
    assert TArrow(src=TInt(), dst=TInt()) == TArrow(src=TInt(), dst=TInt())
    assert TArrow(src=TInt(), dst=TInt()) != TArrow(src=TBool(), dst=TInt())


def test_node_equality_by_structure():
    # Structural equality on nodes (modulo names, which DO matter for raw AST).
    assert Var(name="x") == Var(name="x")
    assert Var(name="x") != Var(name="y")
    assert IntLit(val=3) == IntLit(val=3)
    assert IntLit(val=3) != IntLit(val=4)
