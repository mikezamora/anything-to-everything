"""Tests for Forall / Fix binder decoding (Gap C, K-8 §10.10).

Until Gap C was fixed, decoder.parse_kind_stream raised
``DecodeError("...Forall/Fix binder decoding is Part-2 scope")`` on every
KIND_FORALL / KIND_FIX site, blocking register_lemma's decode_mera +
validate roundtrip for Forall-rooted proofs.

These tests pin the new behaviour: encode -> decode_mera round-trip
yields the expected binder node with the right param_ty and body.
"""
from __future__ import annotations

from src.qft_pcn.logic.ast import (
    parse, Forall, Fix, Var, TNat, Eq, Zero, Succ,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera


def test_decode_forall_identity_body():
    """forall x:Nat. x  round-trips through encode_mera + decode_mera."""
    src = parse(r"forall x:Nat. x")
    state, meta = encode_mera(src)
    res = decode_mera(state, meta)
    ast = res.ast
    assert isinstance(ast, Forall), f"expected Forall, got {type(ast).__name__}"
    assert isinstance(ast.param_ty, TNat)
    assert isinstance(ast.body, Var)
    # The decoder uses fresh names (_v0, _v1, ...); the body Var must
    # alpha-match the binder's param.
    assert ast.body.name == ast.param


def test_decode_forall_with_eq_body():
    """forall n:Nat. Eq(Succ(Zero), Succ(n)) — exercises a non-trivial body."""
    # Build the AST directly; the surface parser may not expose `Eq` as a
    # keyword form, so we construct it explicitly.
    body = Eq(lhs=Succ(arg=Zero()), rhs=Succ(arg=Var(name="n")))
    src = Forall(param="n", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    res = decode_mera(state, meta)
    ast = res.ast
    assert isinstance(ast, Forall)
    assert isinstance(ast.param_ty, TNat)
    # Body shape preserved: Eq(Succ(Zero), Succ(Var)).
    assert isinstance(ast.body, Eq)
    assert isinstance(ast.body.lhs, Succ)
    assert isinstance(ast.body.lhs.arg, Zero)
    assert isinstance(ast.body.rhs, Succ)
    assert isinstance(ast.body.rhs.arg, Var)
    # The Var in the body must reference the Forall's (fresh) param.
    assert ast.body.rhs.arg.name == ast.param


def test_decode_fix_nat_body():
    """fix f:Nat. f  round-trips and decodes to a Fix(param_ty=TNat)."""
    src = Fix(param="f", param_ty=TNat(), body=Var(name="f"))
    state, meta = encode_mera(src)
    res = decode_mera(state, meta)
    ast = res.ast
    assert isinstance(ast, Fix), f"expected Fix, got {type(ast).__name__}"
    assert isinstance(ast.param_ty, TNat)
    assert isinstance(ast.body, Var)
    assert ast.body.name == ast.param
