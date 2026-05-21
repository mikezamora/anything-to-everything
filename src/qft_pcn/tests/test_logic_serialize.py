"""Tests for logic/_serialize.py."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import (
    NodeOccupancy, serialize_preorder, BinderRef, VarRef, count_nodes,
)
from src.qft_pcn.logic.encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN, EncodingTooLarge,
)


def test_serialize_var():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    assert len(sites) == 4
    assert sites[0].kind == KIND_LAM
    assert isinstance(sites[0].binder_ref, BinderRef)
    assert sites[1].kind == KIND_VAR
    assert isinstance(sites[1].var_ref, VarRef)
    # Var refers to the Lam at site 0.
    assert sites[1].var_ref.binder_site == 0
    # Tail is PAD.
    assert sites[2].kind == KIND_PAD
    assert sites[3].kind == KIND_PAD


def test_serialize_application():
    ast = parse(r"\f:Int->Int. \x:Int. f x")
    sites = serialize_preorder(ast, N=8)
    # \f. \x. App(Var f, Var x)
    assert sites[0].kind == KIND_LAM      # outer
    assert sites[1].kind == KIND_LAM      # inner
    assert sites[2].kind == KIND_APP
    assert sites[3].kind == KIND_VAR      # f
    assert sites[4].kind == KIND_VAR      # x
    assert sites[5].kind == KIND_PAD


def test_serialize_p5_layout():
    ast = parse(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    sites = serialize_preorder(ast, N=32)
    # APP, LAM x, APP, LAM y, BIN +, VAR x, VAR y, INT 3, INT 4
    expected_kinds = [
        KIND_APP, KIND_LAM, KIND_APP, KIND_LAM, KIND_BIN,
        KIND_VAR, KIND_VAR, KIND_INT, KIND_INT,
    ]
    for i, k in enumerate(expected_kinds):
        assert sites[i].kind == k, f"site {i}: expected {k}, got {sites[i].kind}"
    # Tail must be PAD.
    for i in range(len(expected_kinds), 32):
        assert sites[i].kind == KIND_PAD


def test_serialize_too_large_raises():
    ast = parse(r"\x:Int. x + x + x + x + x")
    # Bin(+, Bin(+, ..., x), x) — at least 11 nodes (Lam + Bin*4 + Var*5).
    with pytest.raises(EncodingTooLarge):
        serialize_preorder(ast, N=5)


def test_count_nodes():
    assert count_nodes(parse(r"\x:Int. x")) == 2
    assert count_nodes(parse(r"(\x:Int. x + 1)(2)")) == 6
    # App, Lam, Bin, Var, IntLit, IntLit -> 6


def test_var_ref_includes_depth():
    ast = parse(r"\x:Int. \y:Int. x")
    sites = serialize_preorder(ast, N=8)
    # The Var(x) is at site 2. It refers to the outer Lam at site 0 and
    # the depth-from-innermost at the use site is 1 (one Lam, y, between).
    assert sites[2].kind == KIND_VAR
    assert sites[2].var_ref.binder_site == 0
    assert sites[2].var_ref.depth_from_innermost == 1
