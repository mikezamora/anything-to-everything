"""Tests for the extended-calculus AST nodes (spec §3.1, §6.3)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import (
    Node, Var, Zero, Succ, NatLit, Nil, Cons, Eq, Forall, Fix,
    Ty, TInt, TNat, TList, TEq, TProp,
)


def test_zero_is_a_node():
    assert isinstance(Zero(), Node)


def test_succ_holds_one_child():
    s = Succ(arg=Zero())
    assert isinstance(s.arg, Zero)


def test_natlit_holds_value():
    assert NatLit(val=3).val == 3


def test_natlit_rejects_negative():
    with pytest.raises(ValueError, match="NatLit value must be >= 0"):
        NatLit(val=-1)


def test_nil_is_a_node():
    assert isinstance(Nil(), Node)


def test_cons_holds_head_and_tail():
    c = Cons(head=NatLit(val=1), tail=Nil())
    assert c.head.val == 1
    assert isinstance(c.tail, Nil)


def test_eq_holds_two_sides():
    e = Eq(lhs=Var(name="x"), rhs=Var(name="x"))
    assert e.lhs.name == "x" and e.rhs.name == "x"


def test_forall_is_a_binder():
    f = Forall(param="n", param_ty=TNat(), body=Var(name="n"))
    assert f.param == "n"
    assert isinstance(f.param_ty, TNat)
    assert isinstance(f.body, Var)


def test_fix_is_a_binder():
    fx = Fix(param="f", param_ty=TArrow_or_skip(), body=Var(name="f"))
    assert fx.param == "f"


def TArrow_or_skip():
    from src.qft_pcn.logic.ast import TArrow
    return TArrow(src=TNat(), dst=TNat())


def test_extended_types_construct():
    assert isinstance(TNat(), Ty)
    assert isinstance(TList(elem=TNat()), Ty)
    assert isinstance(TEq(lhs_ty=TNat()), Ty)
    assert isinstance(TProp(), Ty)


def test_extended_types_equal_by_structure():
    assert TNat() == TNat()
    assert TList(elem=TNat()) == TList(elem=TNat())
    assert TList(elem=TNat()) != TList(elem=TInt())
