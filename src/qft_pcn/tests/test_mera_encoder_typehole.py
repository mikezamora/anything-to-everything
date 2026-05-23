"""TypeHole encoding tests (M3 P5 acceptance).

The encoder must accept ``Lam.param_ty = TypeHole(candidates=...)`` and
encode the affected type leaves as a genuine candidate-tag superposition
(spec §1.1 / §5.3) — not a classical pick. These tests pin both the
"no longer raises" contract and the superposition shape.
"""
from __future__ import annotations

import numpy as np

from src.qft_pcn.logic.ast import (
    Lam, Var, TypeHole, TInt, TBool,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera, _leaf_marginal
from src.qft_pcn.logic.mera_encoding import (
    TYPE_INT, TYPE_BOOL,
)


def test_encoder_accepts_simple_typehole():
    """A Lam with a TypeHole param_ty encodes without exception."""
    sketch = Lam(
        param="x",
        param_ty=TypeHole(candidates=(TInt(), TBool())),
        body=Var(name="x"),
    )
    state, meta = encode_mera(sketch, n_nodes_max=8)
    assert meta.n_nodes >= 2          # Lam + Var
    assert len(meta.typehole_regions) == 1
    region = meta.typehole_regions[0]
    assert region["binder_node"] == 0
    assert TYPE_INT in region["candidate_tags"]
    assert TYPE_BOOL in region["candidate_tags"]


def test_encoder_typehole_is_superposition():
    """The Var-use type leaf carries non-trivial amplitude on BOTH
    candidate tags — a true superposition, not a first-candidate pick.
    """
    sketch = Lam(
        param="x",
        param_ty=TypeHole(candidates=(TInt(), TBool())),
        body=Var(name="x"),
    )
    state, meta = encode_mera(sketch, n_nodes_max=8)

    # The Var's `type` leaf reflects the bound variable's type, which
    # equals the candidate. Marginal must show weight on TINT and TBOOL.
    var_node_idx = 1     # pre-order: Lam=0, Var=1
    var_type_leaf = meta.layout.leaf_of(var_node_idx, "type")
    p = _leaf_marginal(state, var_type_leaf)
    assert p[TYPE_INT] > 0.1
    assert p[TYPE_BOOL] > 0.1
    # Total of the two candidate weights should dominate the leaf.
    assert p[TYPE_INT] + p[TYPE_BOOL] > 0.9


def test_decoder_handles_typehole_decode():
    """After encoding, decode_mera produces a Lam without crashing.

    The decoded param_ty is one of the candidates (argmax collapse). We
    accept either TInt or TBool — both are valid first-step relaxations
    before imaginary-time evolution biases the state toward one.
    """
    sketch = Lam(
        param="x",
        param_ty=TypeHole(candidates=(TInt(), TBool())),
        body=Var(name="x"),
    )
    state, meta = encode_mera(sketch, n_nodes_max=8)
    res = decode_mera(state, meta)
    assert res.ast is not None
    assert isinstance(res.ast, Lam)
    # The decoded param_ty should be either TInt or TBool (one of the
    # candidates); no exception decoding the type-leaf measurement.
    from src.qft_pcn.logic.ast import TInt as _TI, TBool as _TB
    assert isinstance(res.ast.param_ty, (_TI, _TB))
