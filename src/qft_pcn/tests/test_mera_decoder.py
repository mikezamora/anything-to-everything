"""Tests for decode_mera (spec §7)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse, Lam, Var, App, IntLit, TInt
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera, DecodeResult, sample_mera


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


def test_sample_returns_n_results():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    out = sample_mera(state, meta, n_samples=3,
                      rng=np.random.default_rng(0))
    assert len(out) == 3
    for r in out:
        assert isinstance(r, DecodeResult)


def test_sample_product_state_is_deterministic():
    """For a concrete program (product state) every sample equals decode."""
    from src.qft_pcn.logic.decoder import ast_alpha_eq
    state, meta = encode_mera(parse(r"\x:Int. x + 1"))
    det = decode_mera(state, meta)
    rng = np.random.default_rng(7)
    for _ in range(4):
        s = sample_mera(state, meta, n_samples=1, rng=rng)[0]
        assert ast_alpha_eq(s.ast, det.ast)
