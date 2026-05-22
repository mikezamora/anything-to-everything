"""Tests for hole-bearing MERA encoding (spec §5.3, §5.5)."""
from __future__ import annotations
import math
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, HoleVar
from src.qft_pcn.logic.mera_encoder import encode_mera


def _hole_program():
    """\\x:Int. \\y:Int. ?HOLE  with HoleVar over {x, y}."""
    h = HoleVar(candidates=["x", "y"])
    return Lam(param="x", param_ty=TInt(),
               body=Lam(param="y", param_ty=TInt(), body=h))


def test_hole_program_encodes_to_unit_norm():
    state, meta = encode_mera(_hole_program())
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10)


def test_hole_bid_leaf_is_superposed():
    """The hole's bid leaf has weight on more than one basis state."""
    state, meta = encode_mera(_hole_program())
    # The hole is AST node 2 (Lam_x@0, Lam_y@1, HOLE@2); its bid leaf is
    # 5*2 + 2 = 12.
    from src.qft_pcn.logic.mera_encoding import MERA_LEAF_DIM
    p = np.empty(MERA_LEAF_DIM)
    for b in range(MERA_LEAF_DIM):
        proj = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
        proj[b, b] = 1.0
        p[b] = float(np.real(state.local_expectation(12, proj)))
    nonzero = np.sum(p > 1e-9)
    assert nonzero >= 2, f"hole bid leaf not superposed: {p}"


def test_hole_program_is_not_a_product_state():
    """A genuine hole encoding has tree entanglement somewhere."""
    state, meta = encode_mera(_hole_program())
    # Some interior cut has positive entanglement entropy.
    max_S = max(state.entanglement_entropy(cut)
                for cut in range(1, state.N - 1))
    assert max_S > 1e-6, "hole program encoded as a product state"
