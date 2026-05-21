"""Tests for logic/_typing_extension.py (obligation computation)."""

from __future__ import annotations

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._typing_extension import compute_tobl_tags
from src.qft_pcn.logic.encoding import (
    TOBL_NONE, TOBL_INT, TOBL_BOOL, TOBL_ARR_II, TOBL_ARR_BB,
)


def test_root_has_no_obligation():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    tags = compute_tobl_tags(ast, sites)
    # Root site (LAM): no obligation.
    assert tags[0] == TOBL_NONE
    # Var inside Lam: LAM expects body type = dst(LAM.type) = T_INT.
    assert tags[1] == TOBL_INT
    # PAD sites: no obligation.
    assert tags[2] == TOBL_NONE
    assert tags[3] == TOBL_NONE


def test_app_arg_has_fn_src_obligation():
    """(\\x:Int. x)(1): the arg '1' has obligation = fn.src = T_INT."""
    ast = parse(r"(\x:Int. x)(1)")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    # Layout: APP@0, LAM@1, VAR@2, INT@3 (the arg).
    assert tags[0] == TOBL_NONE         # APP root: no parent
    # fn (LAM at site 1) has NO obligation from APP — APP doesn't impose a
    # single type, only an arrow-dst constraint (handled by T-App-Arrow).
    assert tags[1] == TOBL_NONE
    # Body of LAM (VAR at site 2): obligation = dst(LAM.type) = T_INT.
    assert tags[2] == TOBL_INT
    # Arg (INT at site 3): obligation = src(fn.type) = T_INT.
    assert tags[3] == TOBL_INT


def test_if_branches_have_type_obligation():
    """if true then 1 else 2: cond is Bool-obligated, branches are
    obligated to the if's overall type (Int)."""
    ast = parse(r"if true then 1 else 2")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    # Layout: IF@0, BoolLit@1 (cond), IntLit@2 (then), IntLit@3 (else).
    assert tags[0] == TOBL_NONE   # IF root
    assert tags[1] == TOBL_BOOL   # cond -> Bool
    assert tags[2] == TOBL_INT    # then -> Int (= if's type)
    assert tags[3] == TOBL_INT    # else -> Int


def test_bin_operands_have_int_obligation():
    """1 + 2: both operands obligated to T_INT regardless of arith vs cmp."""
    ast = parse(r"1 + 2")
    sites = serialize_preorder(ast, N=4)
    tags = compute_tobl_tags(ast, sites)
    # Layout: BIN+@0, IntLit@1, IntLit@2.
    assert tags[0] == TOBL_NONE
    assert tags[1] == TOBL_INT   # lhs
    assert tags[2] == TOBL_INT   # rhs


def test_bin_cmp_operands_still_int():
    """x < 5: operands are Int (BIN-Cmp's operands are still Int)."""
    ast = parse(r"\x:Int. x < 5")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    # Layout: LAM@0, BIN<@1, VAR_x@2, IntLit@3.
    assert tags[0] == TOBL_NONE
    assert tags[1] == TOBL_BOOL  # LAM's body obligation = dst(Int->Bool) = Bool
    assert tags[2] == TOBL_INT   # BIN's lhs
    assert tags[3] == TOBL_INT   # BIN's rhs


def test_nested_lambda_obligations():
    """\\x:Int. \\y:Int. x + y:
       outer LAM has body obligation = dst(Int -> Int->Int) = T_ARR_II.
       inner LAM has body obligation = dst(Int->Int) = T_INT.
       BIN's operands obligation = T_INT each.
    """
    ast = parse(r"\x:Int. \y:Int. x + y")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    # Layout: LAM_x@0, LAM_y@1, BIN+@2, VAR_x@3, VAR_y@4.
    assert tags[0] == TOBL_NONE
    assert tags[1] == TOBL_ARR_II   # outer LAM expects body type = Int->Int
    assert tags[2] == TOBL_INT      # inner LAM expects body type = Int (sum)
    assert tags[3] == TOBL_INT      # BIN's lhs
    assert tags[4] == TOBL_INT      # BIN's rhs


def test_pad_sites_have_none_obligation():
    """Sites past the AST are PAD with TOBL_NONE."""
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=8)
    tags = compute_tobl_tags(ast, sites)
    for k in range(2, 8):
        assert tags[k] == TOBL_NONE
