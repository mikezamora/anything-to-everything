"""Tests for Forall non-Nat param_ty round-trip (EXTENSIONS.md fix).

Pre-fix, ``decoder.parse_kind_stream``'s ``KIND_FORALL`` branch
unconditionally returned ``param_ty=TNat()`` because the site's flat type
tag is ``TYPE_PROP`` (Forall returns Prop, not its param's type). The
encoder already wrote ``ty_to_tag(param_ty)`` into the ``value`` species
in ``_mera_leaves.node_leaf_vectors`` — the decoder simply ignored it.

Post-fix, the decoder reads the ``value`` species (``vi``) and recovers
the param_ty. For non-flat param_ty (TList(elem=non-Nat), nested TArrow),
the encoder also records the full Ty in ``nested_type_index[site_idx]``
so the decoder reconstructs ``TList(elem=TBool())`` rather than
defaulting to ``TList(elem=TNat())``.
"""
from __future__ import annotations

from src.qft_pcn.logic.ast import (
    Forall, Var, Eq,
    TNat, TBool, TInt, TList,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera


def test_round_trip_forall_bool():
    """forall b:Bool. b — proposition over a Bool quantifier."""
    src = Forall(param="b", param_ty=TBool(), body=Var(name="b"))
    state, meta = encode_mera(src)
    res = decode_mera(state, meta)
    ast = res.ast
    assert isinstance(ast, Forall), (
        f"expected Forall, got {type(ast).__name__}"
    )
    assert isinstance(ast.param_ty, TBool), (
        f"expected param_ty=TBool, got {ast.param_ty!r}"
    )
    assert isinstance(ast.body, Var)
    assert ast.body.name == ast.param


def test_round_trip_forall_int():
    """forall x:Int. Eq x x — Int-quantified proposition."""
    body = Eq(lhs=Var(name="x"), rhs=Var(name="x"))
    src = Forall(param="x", param_ty=TInt(), body=body)
    state, meta = encode_mera(src)
    res = decode_mera(state, meta)
    ast = res.ast
    assert isinstance(ast, Forall)
    assert isinstance(ast.param_ty, TInt), (
        f"expected param_ty=TInt, got {ast.param_ty!r}"
    )
    assert isinstance(ast.body, Eq)


def test_round_trip_forall_list_nat():
    """forall xs:List Nat. Eq xs xs — no-regression on the legacy default.

    The decoder's legacy fallback was ``TList(elem=TNat())``; now it
    should be the genuine leaf-encoded value (still TList(elem=TNat())).
    """
    body = Eq(lhs=Var(name="xs"), rhs=Var(name="xs"))
    src = Forall(param="xs", param_ty=TList(elem=TNat()), body=body)
    state, meta = encode_mera(src)
    res = decode_mera(state, meta)
    ast = res.ast
    assert isinstance(ast, Forall)
    assert isinstance(ast.param_ty, TList)
    assert isinstance(ast.param_ty.elem, TNat), (
        f"expected elem=TNat, got {ast.param_ty.elem!r}"
    )


def test_round_trip_forall_list_bool():
    """forall xs:List Bool. Eq xs xs — the new capability.

    Pre-fix, the decoder would have returned ``TList(elem=TNat())`` and
    silently dropped the Bool elem. Post-fix, the elem is recovered via
    ``meta.nested_type_index[site_idx]``.
    """
    body = Eq(lhs=Var(name="xs"), rhs=Var(name="xs"))
    src = Forall(param="xs", param_ty=TList(elem=TBool()), body=body)
    state, meta = encode_mera(src)
    res = decode_mera(state, meta)
    ast = res.ast
    assert isinstance(ast, Forall)
    assert isinstance(ast.param_ty, TList)
    assert isinstance(ast.param_ty.elem, TBool), (
        f"expected elem=TBool, got {ast.param_ty.elem!r}"
    )


def test_existing_forall_nat_still_works():
    """No-regression pin on Gap C's canonical Nat-Forall round-trip."""
    src = Forall(param="x", param_ty=TNat(), body=Var(name="x"))
    state, meta = encode_mera(src)
    res = decode_mera(state, meta)
    ast = res.ast
    assert isinstance(ast, Forall)
    assert isinstance(ast.param_ty, TNat)
    assert isinstance(ast.body, Var)
    assert ast.body.name == ast.param
