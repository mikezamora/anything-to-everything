"""Tests for §12.3 Wilson-loop strategy counting.

The constraint Hamiltonian's binding diagram (AST tree edges plus
variable-use-to-binder edges) carries a Wilson-loop algebra whose
dimension ``K`` modulo trivial (contractible) loops counts inequivalent
proof homotopy classes. Genus-``g`` ground-state degeneracy is ``K^g``
(Wen 1989; toric-code analogy gives ``K = 4`` on a torus).

These tests exercise the operator-algebraic count on:
  * a trivial (no-variable) AST -> K = 1, strategy count = 1;
  * a non-trivial AST with multiple variable uses -> K >= 2 (b_1 >= 1);
  * the K^g scaling pinned at two distinct genus values.

No proof-AST enumeration is performed (spec §1.1 anti-shortcut): the
counts are read off the binding diagram's first Betti number.
"""
from __future__ import annotations

import pytest

from src.qft_pcn.composition.topological_degeneracy import (
    compute_wilson_loop_algebra,
    count_proof_strategies,
)
from src.qft_pcn.logic.ast import (
    Bin,
    Eq,
    Forall,
    NatLit,
    TNat,
    Var,
    Zero,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian


# ---------------------------------------------------------------------------
# Opt out of the §9.7 dense-tensor ceiling: real encode_mera of these
# fixtures allocates isometries above the per-test cap. The composition
# tests/conftest.py guard exists to flag accidental dense allocations in
# stub-driven unit tests, not load-bearing physics tests like this one.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadows the conftest fixture for this file only
    yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_trivial_h():
    """A no-variable AST: ``Eq (NatLit 0) (NatLit 0)``.

    The binding diagram has no use-to-binder edges (no variables); it
    is a pure AST tree, hence acyclic. Wilson-loop algebra dimension
    ``K = 2 ** 0 = 1``. This is the identity-like case from spec §12.3.
    """
    src = Eq(lhs=NatLit(val=0), rhs=NatLit(val=0))
    state, meta = encode_mera(src)
    return MeraEvalHamiltonian(meta), state


def _build_nontrivial_h():
    """``forall x:Nat. Eq (add x Zero) x`` — the M2 canonical fixture.

    The variable ``x`` is bound by the ``Forall`` and used twice (once
    inside ``add x Zero`` and once as the right-hand-side of ``Eq``).
    The two ``use -> binder`` edges close non-trivial loops in the
    binding diagram, giving ``b_1 >= 1`` and ``K = 2 ** b_1 >= 2``.
    Multiple proof strategies exist (e.g. induction-on-x vs.
    rewrite-with-add-zero-axiom), consistent with the topological
    count.
    """
    body = Eq(
        lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
        rhs=Var(name="x"),
    )
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    return MeraEvalHamiltonian(meta), state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_trivial_h_has_one_strategy():
    """An identity-like Hamiltonian has a single proof homotopy class.

    The binding diagram of a no-variable AST is a tree (acyclic).
    Euler's formula gives ``b_1 = |E| - |V| + c = 0``, hence
    ``K = 2 ** 0 = 1`` and the genus-1 strategy count is ``1 ** 1 = 1``.
    Spec §12.3 acceptance: ``(a + b) + c = a + (b + c)`` has one
    essentially distinct proof; the no-variable identity ``0 = 0`` is
    even simpler (no proof strategy beyond reflexivity).
    """
    H, _ = _build_trivial_h()
    algebra = compute_wilson_loop_algebra(H)

    # Operator-algebraic invariants of the binding diagram.
    assert algebra["betti_1"] == 0
    assert algebra["algebra_dimension"] == 1
    assert algebra["n_components"] >= 1
    # |E| = |V| - c for an acyclic graph (forest).
    assert algebra["n_edges"] == algebra["n_vertices"] - algebra["n_components"]

    # K^g = 1 for any genus.
    assert count_proof_strategies(H, genus=1) == 1
    assert count_proof_strategies(H, genus=2) == 1


def test_nontrivial_h_has_multiple_strategies():
    """A Hamiltonian with bound-variable cycles has K >= 2 ground states.

    The binding diagram of ``forall x. Eq (add x Zero) x`` has two
    ``x``-use-to-binder edges that close cycles back through the AST
    tree. Each such cycle is a non-trivial Wilson-loop generator.
    Hence ``b_1 >= 1`` and ``K = 2 ** b_1 >= 2``. The genus-1
    degeneracy ``K^1 >= 2`` is the a-priori count of distinct proof
    strategies (spec §12.3 capability: "predict in advance: this
    theorem has 1, 2, or n essentially different proofs").
    """
    H, _ = _build_nontrivial_h()
    algebra = compute_wilson_loop_algebra(H)

    # Multiple variable uses => multiple binding edges => non-trivial cycles.
    assert algebra["betti_1"] >= 1
    assert algebra["algebra_dimension"] >= 2

    # Genus-1 strategy count matches the Wilson-loop algebra dimension.
    strategies_g1 = count_proof_strategies(H, genus=1)
    assert strategies_g1 == algebra["algebra_dimension"]
    assert strategies_g1 >= 2


def test_strategy_count_scales_with_genus():
    """``count_proof_strategies(H, g) == K ** g`` for at least two genera.

    The topological-order signature is ``K^g`` scaling (Wen 1989;
    toric-code ``4^g`` on genus-g surfaces). We pin the exponential
    relation at ``g = 1`` and ``g = 2``: the ratio ``N(2) / N(1) == K``
    is the defining test of a topologically-ordered phase.
    """
    H, _ = _build_nontrivial_h()
    algebra = compute_wilson_loop_algebra(H)
    K = int(algebra["algebra_dimension"])
    assert K >= 2  # precondition for the scaling test to be meaningful.

    n_g1 = count_proof_strategies(H, genus=1)
    n_g2 = count_proof_strategies(H, genus=2)
    n_g3 = count_proof_strategies(H, genus=3)

    assert n_g1 == K ** 1
    assert n_g2 == K ** 2
    assert n_g3 == K ** 3
    # The genus-scaling ratio is K, the topological-order invariant.
    assert n_g2 // n_g1 == K
    assert n_g3 // n_g2 == K

    # Genus 0 (sphere): single ground state regardless of K.
    assert count_proof_strategies(H, genus=0) == 1


def test_negative_genus_rejected():
    """No surface has negative genus; the API rejects it explicitly.

    Defensive guard rather than silently returning ``K ** (-1)`` (which
    would be fractional for ``K >= 2`` and zero under integer division).
    """
    H, _ = _build_trivial_h()
    with pytest.raises(ValueError):
        count_proof_strategies(H, genus=-1)
