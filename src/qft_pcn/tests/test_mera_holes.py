"""Tests for hole-bearing MERA encoding (spec §5.3, §5.5)."""
from __future__ import annotations
import math
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, HoleVar
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.qft.mera import MERA


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
    """A genuine 2-candidate hole encoding carries tree entanglement.

    The §1.1 structural-marker test (spec §5.5 / §9.4): a hole is a genuine
    superposition over its candidates, so the encoded MERA state must NOT
    be a product state. With 2 candidates the hole is a 2-branch
    superposition; the interior cut that separates the branch-distinguishing
    leaves must carry exactly ln 2 of entanglement entropy.

    This pins the VALUE (not merely > 0): the negative control below
    (test_one_leaf_difference_superposition_has_zero_entropy) constructs a
    classical-lookup-shortcut-style one-leaf-difference superposition and
    shows it has S = 0 — so this assertion genuinely discriminates the
    principled binding-as-entanglement encoding from a shortcut.
    """
    state, meta = encode_mera(_hole_program())
    interior = [state.entanglement_entropy(cut)
                for cut in range(1, state.N - 1)]
    max_S = max(interior)
    # Positive (not a product state) AND pinned near ln 2: a genuine
    # 2-candidate hole branch superposition cuts to a rank-2 reduced
    # density with equal weights -> S = ln 2.
    assert max_S > 1e-6, "hole program encoded as a product state"
    assert abs(max_S - math.log(2)) < 1e-6, (
        f"max interior cut entropy {max_S} not near ln 2 = {math.log(2)}; "
        "the 2-candidate hole's tree entanglement is not the expected "
        "rank-2 equal-weight superposition")


def test_one_leaf_difference_superposition_has_zero_entropy():
    """NEGATIVE CONTROL for the §1.1 structural-marker test.

    A two-term superposition that differs on exactly ONE leaf is a genuine
    PRODUCT state — algebraically (|a>+|b>) (x) |rest>. This is the shape a
    classical-lookup *shortcut* would take if it tried to masquerade as a
    superposition: branching on a single leaf carries no tree entanglement.

    Its entanglement entropy must be 0 at every cut. This proves
    test_hole_program_is_not_a_product_state genuinely discriminates the
    principled encoding (S = ln 2) from the shortcut (S = 0) — which it
    could NOT do while _entropy_from_terms had the over-reporting bug
    (that bug returned ln 2 for this product state too).
    """
    d = 4

    def vec(i: int) -> np.ndarray:
        v = np.zeros(d, dtype=complex)
        v[i] = 1.0
        return v

    # |0000> + |1000>: differs only on leaf 0 -> genuine product state.
    base = [vec(0), vec(0), vec(0), vec(0)]
    other = [vec(1), vec(0), vec(0), vec(0)]
    shortcut = MERA.from_term_superposition(
        [(1.0, base), (1.0, other)]).normalize()
    for cut in range(shortcut.N - 1):
        S = shortcut.entanglement_entropy(cut)
        assert abs(S) < 1e-9, (
            f"one-leaf-difference superposition has S={S} at cut {cut}; "
            "a genuine product state must have S = 0 — the structural "
            "marker would not discriminate the shortcut otherwise")
