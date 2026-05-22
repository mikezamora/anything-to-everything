"""Tests for decode_mera (spec §7)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import parse, Lam, Var, App, IntLit, TInt
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera, DecodeResult


def test_decode_identity_lambda():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    res = decode_mera(state, meta)
    assert isinstance(res, DecodeResult)
    assert res.residual_norm < 1e-10
    assert isinstance(res.ast, Lam)
    assert isinstance(res.ast.body, Var)
    assert res.ast.body.name == res.ast.param


def test_decode_intlit():
    state, meta = encode_mera(parse(r"\x:Int. 3"))
    res = decode_mera(state, meta)
    assert isinstance(res.ast.body, IntLit)
    assert res.ast.body.val == 3


def test_decode_application():
    state, meta = encode_mera(parse(r"\f:Int->Int. \x:Int. f x"))
    res = decode_mera(state, meta)
    lam_f = res.ast
    assert isinstance(lam_f, Lam)
    app = lam_f.body.body
    assert isinstance(app, App)
