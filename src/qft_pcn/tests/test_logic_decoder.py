from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import (
    parse, Var, Lam, App, IntLit, BoolLit, If, Bin, TInt, TBool, TArrow,
)
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.decoder import decode, DecodeResult


def test_decode_var_identity_lambda():
    state, meta = encode(parse(r"\x:Int. x"), N=8)
    res = decode(state, meta)
    assert isinstance(res, DecodeResult)
    assert res.residual_norm < 1e-10
    assert isinstance(res.ast, Lam)
    assert isinstance(res.ast.param_ty, TInt)
    assert isinstance(res.ast.body, Var)
    assert res.ast.body.name == res.ast.param


def test_decode_intlit():
    state, meta = encode(parse(r"\x:Int. 3"), N=8)
    res = decode(state, meta)
    assert res.residual_norm < 1e-10
    assert isinstance(res.ast, Lam)
    assert isinstance(res.ast.body, IntLit)
    assert res.ast.body.val == 3


def test_decode_boolean():
    state, meta = encode(parse(r"\x:Int. true"), N=8)
    res = decode(state, meta)
    assert isinstance(res.ast.body, BoolLit)
    assert res.ast.body.val is True


def test_decode_app():
    state, meta = encode(parse(r"\f:Int->Int. \x:Int. f x"), N=8)
    res = decode(state, meta)
    lam_f = res.ast
    assert isinstance(lam_f, Lam)
    lam_x = lam_f.body
    assert isinstance(lam_x, Lam)
    app = lam_x.body
    assert isinstance(app, App)
    assert isinstance(app.fn, Var) and app.fn.name == lam_f.param
    assert isinstance(app.arg, Var) and app.arg.name == lam_x.param


def test_decode_if():
    state, meta = encode(parse(r"\x:Int. if true then 1 else 2"), N=8)
    res = decode(state, meta)
    if_node = res.ast.body
    assert isinstance(if_node, If)
    assert if_node.cond.val is True
    assert if_node.then_b.val == 1
    assert if_node.else_b.val == 2


def test_decode_bin():
    state, meta = encode(parse(r"\x:Int. 1 + 2"), N=8)
    res = decode(state, meta)
    b = res.ast.body
    assert isinstance(b, Bin)
    assert b.op == "+"
    assert b.lhs.val == 1 and b.rhs.val == 2
