"""Tests for logic/_resolve.py (lexical binder resolution)."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._resolve import resolve_binders, ResolvedRef
from src.qft_pcn.logic.encoding import IllScopedVar, TooManyBinders


def _refs(ast):
    out = []
    resolve_binders(ast, on_var=lambda var, ref: out.append((var.name, ref)))
    return out


def test_resolve_identity_lambda():
    ast = parse(r"\x:Int. x")
    refs = _refs(ast)
    # One Var, depth-from-innermost = 0 (its binder is the only one).
    assert len(refs) == 1
    name, ref = refs[0]
    assert name == "x"
    assert isinstance(ref, ResolvedRef)
    assert ref.depth_from_innermost == 0


def test_resolve_outer_binder_used_inside_inner_scope():
    ast = parse(r"\x:Int. \y:Int. x")
    refs = _refs(ast)
    # The Var(x) is inside an inner Lam(y), so x is one step out -> depth 1.
    assert len(refs) == 1
    name, ref = refs[0]
    assert name == "x"
    assert ref.depth_from_innermost == 1


def test_resolve_shadowing():
    ast = parse(r"\x:Int. \x:Int. x")
    # Inner x shadows outer.
    refs = _refs(ast)
    assert len(refs) == 1
    _, ref = refs[0]
    assert ref.depth_from_innermost == 0


def test_resolve_unbound_raises():
    ast = parse("x")
    with pytest.raises(IllScopedVar, match="Var\\('x'\\)"):
        resolve_binders(ast, on_var=lambda v, r: None)


def test_resolve_too_many_binders_raises():
    src = (r"\a:Int. \b:Int. \c:Int. \d:Int. \e:Int. \f:Int. \g:Int. "
           r"\h:Int. a")
    ast = parse(src)
    # 8 nested lambdas, but BID_CUTOFF - 1 = 7 allowed.
    with pytest.raises(TooManyBinders):
        resolve_binders(ast, on_var=lambda v, r: None)


def test_resolve_multi_var_through_bin_app_if():
    """All non-Lam/Var traversal arms (Bin, App, If, IntLit, BoolLit)
    fire correctly, with the callback invoked once per Var."""
    src = (r"\x:Int. \y:Int. "
           r"if x < y then x + y else (\f:Int->Int. f y) (\z:Int. z * x)")
    ast = parse(src)
    refs = _refs(ast)
    # Expected Vars in pre-order: x (in `x<y`), y, x (in `x+y`), y,
    #   f, y, z, x.
    names = [name for name, _ in refs]
    assert names == ["x", "y", "x", "y", "f", "y", "z", "x"], names
    # x in the `x<y` is inside both Lam_x and Lam_y -> depth_from_innermost=1.
    assert refs[0][1].depth_from_innermost == 1
    # The z inside `\z. z * x` sees Lam_z as innermost (depth 0); x is 2
    # levels out (Lam_z then Lam_f) -> depth 2.
    assert refs[6][1].depth_from_innermost == 0   # z is innermost
    assert refs[7][1].depth_from_innermost == 2   # x is 2 levels out
