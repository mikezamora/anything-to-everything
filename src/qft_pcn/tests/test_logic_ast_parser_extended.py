"""Tests for the I-Task-10 parser surface extensions.

Covers parser productions for `forall`, `Eq`, `Nat`, `add`, `Zero`, `Succ`,
`NatLit`. The AST classes already exist; only the parser tokens / productions
are new. See memory/i-task-10-blockers.md (blocker #2).
"""
from __future__ import annotations

from src.qft_pcn.logic.ast import (
    App,
    Eq,
    Forall,
    NatLit,
    Succ,
    TNat,
    Var,
    Zero,
    parse,
)


def test_parse_zero_literal():
    ast = parse(r"Zero")
    assert isinstance(ast, Zero)


def test_parse_nat_literal_parens():
    ast = parse(r"NatLit(3)")
    assert isinstance(ast, NatLit) and ast.val == 3


def test_parse_nat_literal_juxt():
    ast = parse(r"NatLit 3")
    assert isinstance(ast, NatLit) and ast.val == 3


def test_parse_succ():
    ast = parse(r"Succ Zero")
    assert isinstance(ast, Succ)
    assert isinstance(ast.arg, Zero)


def test_parse_add_two_natlits():
    """add chosen as a free identifier: encoded as App(App(Var('add'), lhs), rhs)."""
    ast = parse(r"add (NatLit 2) (NatLit 3)")
    assert isinstance(ast, App)
    assert isinstance(ast.fn, App)
    assert isinstance(ast.fn.fn, Var) and ast.fn.fn.name == "add"
    assert isinstance(ast.fn.arg, NatLit) and ast.fn.arg.val == 2
    assert isinstance(ast.arg, NatLit) and ast.arg.val == 3


def test_parse_eq():
    ast = parse(r"Eq Zero Zero")
    assert isinstance(ast, Eq)
    assert isinstance(ast.lhs, Zero) and isinstance(ast.rhs, Zero)


def test_parse_forall_simple():
    ast = parse(r"forall x:Nat. x")
    assert isinstance(ast, Forall)
    assert ast.param == "x"
    assert isinstance(ast.param_ty, TNat)
    assert isinstance(ast.body, Var) and ast.body.name == "x"


def test_parse_forall_with_eq():
    ast = parse(r"forall x:Nat. Eq x x")
    assert isinstance(ast, Forall)
    assert isinstance(ast.body, Eq)


def test_parse_add_zero_theorem():
    """The first I-Task-10 demo theorem: forall x:Nat. Eq (add x Zero) x."""
    ast = parse(r"forall x:Nat. Eq (add x Zero) x")
    assert isinstance(ast, Forall)
    assert ast.param == "x"
    assert isinstance(ast.param_ty, TNat)
    assert isinstance(ast.body, Eq)
    # lhs = add x Zero  -> App(App(Var('add'), Var('x')), Zero())
    lhs = ast.body.lhs
    assert isinstance(lhs, App)
    assert isinstance(lhs.fn, App)
    assert isinstance(lhs.fn.fn, Var) and lhs.fn.fn.name == "add"
    assert isinstance(lhs.fn.arg, Var) and lhs.fn.arg.name == "x"
    assert isinstance(lhs.arg, Zero)
    # rhs = x
    assert isinstance(ast.body.rhs, Var) and ast.body.rhs.name == "x"


def test_parse_commutativity_theorem():
    """The second I-Task-10 demo: nested forall, commutativity of add."""
    ast = parse(r"forall x:Nat. forall y:Nat. Eq (add x y) (add y x)")
    assert isinstance(ast, Forall)
    assert ast.param == "x"
    inner = ast.body
    assert isinstance(inner, Forall) and inner.param == "y"
    assert isinstance(inner.body, Eq)
