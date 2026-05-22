"""MERA encoder round-trip acceptance (spec §9.1, §9.7)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera
from src.qft_pcn.logic.decoder import ast_alpha_eq

# Extended-calculus programs use the parser; if the parser does not yet
# accept Succ/Cons/Eq syntax, build those ASTs directly (see P6-P8).
from src.qft_pcn.logic.ast import (
    Succ, Zero, Cons, NatLit, Nil, Lam, Eq, Var, TNat,
)

_STLC = {
    "P1": r"\x:Int. x",
    "P2": r"(\x:Int. x + 1)(2)",
    "P3": r"\f:Int->Int. \x:Int. f (f x)",
    "P4": r"if 1 < 2 then ((\x:Bool. x)(true)) else false",
    "P5": r"(\x:Int. (\y:Int. x + y)(3))(4)",
}


@pytest.mark.parametrize("name,src", list(_STLC.items()))
def test_roundtrip_stlc(name, src):
    expected = parse(src)
    state, meta = encode_mera(expected)
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10), f"{name} not unit norm"
    res = decode_mera(state, meta)
    assert res.residual_norm < 1e-10, f"{name} residual {res.residual_norm}"
    assert ast_alpha_eq(res.ast, expected), (
        f"{name}: {res.ast!r} != {expected!r}")


def test_roundtrip_p6_nat():
    expected = Succ(arg=Succ(arg=Zero()))
    state, meta = encode_mera(expected)
    res = decode_mera(state, meta)
    assert ast_alpha_eq(res.ast, expected)


def test_roundtrip_p7_list():
    expected = Cons(head=NatLit(val=1), tail=Cons(head=NatLit(val=2),
                                                  tail=Nil()))
    state, meta = encode_mera(expected)
    res = decode_mera(state, meta)
    assert ast_alpha_eq(res.ast, expected)


def test_roundtrip_p8_eq():
    expected = Lam(param="x", param_ty=TNat(),
                   body=Eq(lhs=Var(name="x"), rhs=Var(name="x")))
    state, meta = encode_mera(expected)
    res = decode_mera(state, meta)
    assert ast_alpha_eq(res.ast, expected)


def test_cross_substrate_anchor_p1_to_p5():
    """MERA and MPS encoders decode to the same AST (spec §9.7)."""
    from src.qft_pcn.logic.encoder import encode as encode_mps
    from src.qft_pcn.logic.decoder import decode as decode_mps
    for name, src in _STLC.items():
        p = parse(src)
        mera_ast = decode_mera(*encode_mera(p)).ast
        mps_state, mps_meta = encode_mps(p, N=32)
        mps_ast = decode_mps(mps_state, mps_meta).ast
        assert ast_alpha_eq(mera_ast, mps_ast), f"{name} substrate mismatch"
