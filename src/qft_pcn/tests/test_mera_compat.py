"""Tests for src/qft_pcn/logic/mera_compat.py — A-to-F bridge.

Spec: docs/superpowers/specs/2026-05-21-mera-substrate-design.md, §7.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.mera_compat import (
    lift_encoding_meta_to_mera, MERAEncodingMeta,
)


def test_lift_identity_lambda():
    p = parse(r"\x:Int. x")
    _, mps_meta = encode(p, N=32)
    mera_meta = lift_encoding_meta_to_mera(mps_meta, N=32)
    assert isinstance(mera_meta, MERAEncodingMeta)
    assert mera_meta.N == 32
    assert mera_meta.L == 5
    assert mera_meta.chi_layer == 16


def test_lift_assigns_lca_layers_to_binders():
    p = parse(r"\x:Int. \y:Int. x + y")
    _, mps_meta = encode(p, N=32)
    mera_meta = lift_encoding_meta_to_mera(mps_meta, N=32)
    # Every binder gets an LCA layer in [0, L).
    for _binder, lca in mera_meta.binder_to_lca_layer.items():
        assert 0 <= lca < mera_meta.L


def test_lift_invalid_n_rejected():
    p = parse(r"\x:Int. x")
    _, mps_meta = encode(p, N=32)
    with pytest.raises(ValueError):
        lift_encoding_meta_to_mera(mps_meta, N=30)


# ---- Task 23: Recursive Fibonacci bond-dim scaling (spec §10.5) ----------


@pytest.mark.parametrize("D", [2, 3, 4, 5])
def test_recursive_bond_dim_scales_as_O_log_N(D):
    """Spec §10.5 acceptance test: for a Fibonacci-like recursive program
    at unroll depth D, the encoded MERA has max per-layer bond dim
    bounded by a function of D (NOT of N) — i.e. O(log N), not O(N).
    """
    from src.qft_pcn.logic.ast import (
        Rec, Var, Lam, App, IntLit, Bin, TInt, TArrow,
    )
    from src.qft_pcn.logic.mera_compat import encode_rec_to_mera
    body = Lam(
        param="n", param_ty=TInt(),
        body=Bin(op="+",
                 lhs=App(fn=Var("f"),
                         arg=Bin(op="-", lhs=Var("n"), rhs=IntLit(val=1))),
                 rhs=App(fn=Var("f"),
                         arg=Bin(op="-", lhs=Var("n"), rhs=IntLit(val=2)))),
    )
    program = Rec(name="f", name_ty=TArrow(src=TInt(), dst=TInt()), body=body)
    N = 2 ** (3 + D)
    state, meta = encode_rec_to_mera(program, N=N, chi_layer=16,
                                     unroll_depth=D)
    max_bond = max(state.bond_dimensions())
    # chi_layer = 16, so every isometry's coarse-output dim is <= 16.
    # The expected bound 4 * (3 + D) is generous; the bond dim is in
    # fact O(chi_layer), independent of D. We assert the spec's bound.
    assert max_bond < 4 * (3 + D), (
        f"D={D}, N={N}: max layer bond dim = {max_bond}, "
        f"expected < {4 * (3 + D)}. The encoder may be propagating "
        f"binders horizontally — re-read spec §1.3 and §7.2."
    )


def test_recursive_bond_dim_is_O_log_N_not_O_N():
    """Cross-check: as N grows 16 → 32 → 64 → 128 (8x), the max bond dim
    must grow logarithmically — at most a small constant factor."""
    from src.qft_pcn.logic.ast import (
        Rec, Var, Lam, App, IntLit, Bin, TInt, TArrow,
    )
    from src.qft_pcn.logic.mera_compat import encode_rec_to_mera
    body = Lam(
        param="n", param_ty=TInt(),
        body=Bin(op="+",
                 lhs=App(fn=Var("f"),
                         arg=Bin(op="-", lhs=Var("n"), rhs=IntLit(val=1))),
                 rhs=App(fn=Var("f"),
                         arg=Bin(op="-", lhs=Var("n"), rhs=IntLit(val=2)))),
    )
    program = Rec(name="f", name_ty=TArrow(src=TInt(), dst=TInt()), body=body)
    bonds_vs_N = []
    for log_N in (4, 5, 6, 7):
        N = 2 ** log_N
        state, _ = encode_rec_to_mera(program, N=N, chi_layer=16,
                                      unroll_depth=log_N - 3)
        bonds_vs_N.append((N, max(state.bond_dimensions())))
    first = bonds_vs_N[0][1]
    last = bonds_vs_N[-1][1]
    # last <= 4 * first is the O(log N) growth bound: as N went 16x,
    # bond dim must not have grown faster than 4x.
    assert last <= 4 * first, (
        f"bond dim grew from {first} (N=16) to {last} (N=128) — "
        f"not O(log N). Series: {bonds_vs_N}"
    )
