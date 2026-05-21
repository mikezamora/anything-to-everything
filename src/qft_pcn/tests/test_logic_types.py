from __future__ import annotations

from src.qft_pcn.logic.ast import parse, TInt, TBool, TArrow
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._types import compute_site_types, ty_to_tag
from src.qft_pcn.logic.encoding import (
    TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_BB, TYPE_ARR_NESTED, TYPE_NONE,
    KIND_VAR,
)


def test_ty_to_tag_simple():
    assert ty_to_tag(TInt())[0] == TYPE_INT
    assert ty_to_tag(TBool())[0] == TYPE_BOOL
    assert ty_to_tag(TArrow(src=TInt(), dst=TInt()))[0] == TYPE_ARR_II
    assert ty_to_tag(TArrow(src=TBool(), dst=TBool()))[0] == TYPE_ARR_BB


def test_ty_to_tag_nested():
    # (Int -> Int) -> Int
    t = TArrow(src=TArrow(src=TInt(), dst=TInt()), dst=TInt())
    tag, nested = ty_to_tag(t)
    assert tag == TYPE_ARR_NESTED
    assert nested == t


def test_compute_p1_types():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    # site 0: Lam : Int -> Int
    assert types[0] == TYPE_ARR_II
    # site 1: Var(x) : Int
    assert types[1] == TYPE_INT
    # PAD sites have TYPE_NONE
    assert types[2] == TYPE_NONE
    assert types[3] == TYPE_NONE


def test_compute_p2_types():
    ast = parse(r"(\x:Int. x + 1)(2)")
    sites = serialize_preorder(ast, N=8)
    types = compute_site_types(ast, sites)
    # 0: App : Int
    assert types[0] == TYPE_INT
    # 1: Lam : Int -> Int
    assert types[1] == TYPE_ARR_II
    # 2: Bin + : Int
    assert types[2] == TYPE_INT
    # 3: Var(x) : Int
    assert types[3] == TYPE_INT
    # 4: IntLit 1 : Int
    assert types[4] == TYPE_INT
    # 5: IntLit 2 : Int
    assert types[5] == TYPE_INT


def test_compute_compare_op_returns_bool():
    ast = parse(r"\x:Int. x < 5")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    # 0: Lam Int -> Bool
    from src.qft_pcn.logic.encoding import TYPE_ARR_IB
    assert types[0] == TYPE_ARR_IB
    # 1: Bin < : Bool
    assert types[1] == TYPE_BOOL
