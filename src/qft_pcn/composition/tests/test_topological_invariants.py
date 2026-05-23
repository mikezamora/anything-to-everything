"""§12.2 Topological-invariant program-equivalence tests.

These tests pin down the spec contract: Wilson loops + Jones polynomial
on the MPS/MERA encoding form an operator-algebraic equivalence oracle
that is

  * identity-preserving (a program is equivalent to itself, bitwise),
  * alpha-invariant (``\\x:Int. x`` ~ ``\\y:Int. y``, the §1.1 binding-
    as-entanglement principle in action — a bound-variable rename leaves
    the bond entanglement untouched, hence every Wilson-loop expectation
    is identical), and
  * discriminating (distinct programs produce distinct signatures).

All tests run on real ``encode_mera`` output (no mocks, no AST hashes).
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.composition.topological_invariants import (
    Polynomial,
    TopologicalSignature,
    compute_jones_polynomial,
    compute_wilson_loops,
    programs_equivalent,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


def _encode(src: str):
    state, _ = encode_mera(parse(src))
    return state


# ---- §12.2 acceptance: identity --------------------------------------------


def test_identical_programs_have_same_invariants():
    """Encoding ``\\x:Int. x`` twice must yield bitwise-equal Wilson loops
    and a bitwise-equal Jones polynomial. This pins the determinism of
    ``encode_mera`` plus the determinism of the invariant computation."""
    s1 = _encode(r"\x:Int. x")
    s2 = _encode(r"\x:Int. x")

    w1 = compute_wilson_loops(s1)
    w2 = compute_wilson_loops(s2)
    assert set(w1.keys()) == set(w2.keys())
    for k in w1:
        assert abs(w1[k] - w2[k]) < 1e-12, (
            f"Wilson loop {k} differs across identical encodings: "
            f"{w1[k]} vs {w2[k]}"
        )

    j1 = compute_jones_polynomial(s1)
    j2 = compute_jones_polynomial(s2)
    assert j1.degree() == j2.degree()
    assert j1.approx_equal(j2, tol=1e-12)

    assert programs_equivalent(s1, s2)


# ---- §12.2 acceptance: alpha-equivalence -----------------------------------


def test_alpha_equivalent_programs_have_same_invariants():
    """``\\x:Int. x`` and ``\\y:Int. y`` are alpha-equivalent. Per §1.1
    the encoder canonicalizes binders so the two MERAs encode identical
    bond entanglement; therefore every Wilson loop and the Jones
    polynomial must match to floating-point precision.

    This is the §12.2 invariant property: alpha-equivalence falls out of
    the substrate (bond entanglement, not variable names) — we are *not*
    AST-normalizing here, we are measuring an operator expectation."""
    sx = _encode(r"\x:Int. x")
    sy = _encode(r"\y:Int. y")

    assert sx.N == sy.N, (
        "alpha-equivalent programs must encode to states of equal length"
    )

    wx = compute_wilson_loops(sx)
    wy = compute_wilson_loops(sy)
    assert set(wx.keys()) == set(wy.keys())
    for k in wx:
        assert abs(wx[k] - wy[k]) < 1e-10, (
            f"alpha-rename changed Wilson loop {k}: {wx[k]} vs {wy[k]}"
        )

    jx = compute_jones_polynomial(sx)
    jy = compute_jones_polynomial(sy)
    assert jx.approx_equal(jy, tol=1e-10), (
        "alpha-rename changed the Jones polynomial — §1.1 violated"
    )

    assert programs_equivalent(sx, sy), (
        "topological-signature oracle failed to recognize alpha-equivalence"
    )


# ---- §12.2 acceptance: discrimination --------------------------------------


def test_distinct_programs_have_different_invariants():
    """Two programs computing different functions must produce distinct
    topological signatures. We compare ``\\x:Int. x`` (identity on Int)
    against ``\\x:Int. \\y:Int. x`` (the K combinator at Int->Int->Int);
    the latter has a *different number of binders*, hence different
    bond-entanglement geometry, hence a different invariant.

    The discrimination is operator-algebraic: we do not inspect the AST,
    we read the difference off the Wilson-loop expectations / Jones
    polynomial."""
    s1 = _encode(r"\x:Int. x")
    s2 = _encode(r"\x:Int. \y:Int. x")

    sig1 = TopologicalSignature.of(s1)
    sig2 = TopologicalSignature.of(s2)

    # Either the leaf count differs (immediate distinction) or the
    # invariants themselves differ. Both routes are acceptable per §12.2;
    # both stem from the same source — distinct entanglement geometry.
    if sig1.n_leaves != sig2.n_leaves:
        assert not programs_equivalent(s1, s2)
        return

    # Same length but different invariants. At least one Wilson loop or
    # one Jones coefficient must differ above _SIGNATURE_TOL.
    w1 = dict(sig1.wilson)
    w2 = dict(sig2.wilson)
    diffs = [abs(w1[k] - w2[k]) for k in w1.keys() & w2.keys()]
    jones_diff = max(
        abs((sig1.jones.coeffs[k] if k < len(sig1.jones.coeffs) else 0)
            - (sig2.jones.coeffs[k] if k < len(sig2.jones.coeffs) else 0))
        for k in range(max(len(sig1.jones.coeffs), len(sig2.jones.coeffs)))
    )
    assert max(diffs + [jones_diff]) > 1e-6, (
        "distinct programs produced indistinguishable invariants — "
        "either the encoder collapsed them or the invariant is too coarse"
    )
    assert not programs_equivalent(s1, s2)


# ---- structural sanity: Jones polynomial shape -----------------------------


def test_jones_polynomial_has_unknot_normalization():
    """The Jones polynomial's constant term is the unknot value (= 1) by
    construction. This pins the normalization convention so signature
    comparisons across different ``encode_mera`` builds are stable."""
    s = _encode(r"\x:Int. x")
    j = compute_jones_polynomial(s)
    assert isinstance(j, Polynomial)
    assert abs(j.coeffs[0] - (1.0 + 0.0j)) < 1e-12
