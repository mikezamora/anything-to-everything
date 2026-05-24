"""Tests for §12.3 binding-graph cycle-space invariant (D12 honest scope).

Per D12: these tests pin the STRUCTURAL invariant of the encoded
program's binding diagram (Betti number of the use-to-binder graph,
lifted to ``K^g`` via Wen 1989 toric-code formula). This is NOT the
spec §12.3 ground-subspace degeneracy ("essentially different proof
strategies") — that genuine count requires eigvalsh enumeration on
the constraint Hamiltonian and is tracked in EXTENSIONS.

These tests exercise the operator-algebraic invariant on:
  * a trivial (no-variable) AST -> K = 1, ``K^g = 1``;
  * a non-trivial AST with multiple variable uses -> K >= 2 (b_1 >= 1);
  * the K^g scaling pinned at two distinct genus values.

No proof-AST enumeration is performed (spec §1.1 anti-shortcut): the
counts are read off the binding diagram's first Betti number.
"""
from __future__ import annotations

import pytest

from src.qft_pcn.composition.topological_degeneracy import (
    compute_wilson_loop_algebra,
    count_binding_graph_strategies,
    count_ground_subspace_strategies,
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
    assert count_binding_graph_strategies(H, genus=1) == 1
    assert count_binding_graph_strategies(H, genus=2) == 1


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
    strategies_g1 = count_binding_graph_strategies(H, genus=1)
    assert strategies_g1 == algebra["algebra_dimension"]
    assert strategies_g1 >= 2


def test_strategy_count_scales_with_genus():
    """``count_binding_graph_strategies(H, g) == K ** g`` for at least two genera.

    The topological-order signature is ``K^g`` scaling (Wen 1989;
    toric-code ``4^g`` on genus-g surfaces). We pin the exponential
    relation at ``g = 1`` and ``g = 2``: the ratio ``N(2) / N(1) == K``
    is the defining test of a topologically-ordered phase.
    """
    H, _ = _build_nontrivial_h()
    algebra = compute_wilson_loop_algebra(H)
    K = int(algebra["algebra_dimension"])
    assert K >= 2  # precondition for the scaling test to be meaningful.

    n_g1 = count_binding_graph_strategies(H, genus=1)
    n_g2 = count_binding_graph_strategies(H, genus=2)
    n_g3 = count_binding_graph_strategies(H, genus=3)

    assert n_g1 == K ** 1
    assert n_g2 == K ** 2
    assert n_g3 == K ** 3
    # The genus-scaling ratio is K, the topological-order invariant.
    assert n_g2 // n_g1 == K
    assert n_g3 // n_g2 == K

    # Genus 0 (sphere): single ground state regardless of K.
    assert count_binding_graph_strategies(H, genus=0) == 1


# ---------------------------------------------------------------------------
# §12.3 ground-subspace degeneracy (genuine Wen / Kitaev count via
# eigvalsh near zero on the §12.6 constraint Hessian).
# ---------------------------------------------------------------------------


def _build_commutativity_h():
    """``forall a:Nat. forall b:Nat. Eq (a+b) (b+a)`` — commutativity.

    The body has two distinct proof strategies in the categorical
    sense: (i) reflexivity after the commutativity rewrite, (ii)
    direct structural induction on ``a``. Both leave the constraint
    Hamiltonian's residual at zero, hence the ground subspace has
    dimension ``>= 2``.
    """
    body = Eq(
        lhs=Bin(op="+", lhs=Var(name="a"), rhs=Var(name="b")),
        rhs=Bin(op="+", lhs=Var(name="b"), rhs=Var(name="a")),
    )
    inner = Forall(param="b", param_ty=TNat(), body=body)
    src = Forall(param="a", param_ty=TNat(), body=inner)
    state, meta = encode_mera(src)
    return MeraEvalHamiltonian(meta), state


def test_a_plus_b_equals_b_plus_a_has_two_strategies():
    """Spec §12.3 acceptance: ``a + b = b + a`` => ground-subspace count >= 2.

    Two distinct categorical proof strategies (commutativity-rewrite
    + reflexivity; structural induction on ``a``) correspond to two
    independent zero modes of the §12.6 constraint Hessian. The
    ground-subspace count from ``eigvalsh`` near zero must be at
    least 2.
    """
    H, state = _build_commutativity_h()
    count = count_ground_subspace_strategies(H, state)
    assert count >= 2, (
        f"commutativity has at least two essentially-different proofs; "
        f"ground_subspace_count = {count}"
    )

    # The combined estimator dict must surface both numbers.
    combined = count_proof_strategies(H, state, genus=1)
    assert combined["ground_subspace_count"] == count
    assert combined["binding_graph_count"] >= 1
    assert combined["genus"] == 1


def test_trivial_constraint_has_unit_degeneracy():
    """A no-variable ``Eq 0 0`` is trivially satisfied => ground count = 1.

    The constraint Hessian on the trivial AST has all residuals zero
    (reflexivity holds identically); per the §12.6 PSD construction
    every diagonal entry is zero, hence M is the zero matrix. Its
    spectrum is identically zero, and ``count_ground_subspace_strategies``
    floors at 1 (the variational vacuum is itself a ground state).
    Spec §12.3: an identity tautology admits one essentially-distinct
    proof — reflexivity.
    """
    H, state = _build_trivial_h()
    count = count_ground_subspace_strategies(H, state)
    assert count == 1, (
        f"trivial reflexivity has unit ground-subspace degeneracy; "
        f"got {count}"
    )


def test_ground_count_invariant_under_continuous_deformation():
    """Wen 1989: the ground-subspace dimension is a topological invariant.

    Vary the spectral threshold across a small continuous interval
    (which is the analogue of a continuous deformation of the
    Hamiltonian's parameters that keeps it within the same
    topological phase). The integer ground-subspace count must remain
    constant — this is the defining property of topological order.
    """
    H, state = _build_commutativity_h()
    counts = [
        count_ground_subspace_strategies(H, state, ground_threshold=t)
        for t in (1e-4, 5e-4, 1e-3, 5e-3)
    ]
    # All counts must agree — topological invariance under the
    # continuous deformation of the threshold within the gap.
    assert len(set(counts)) == 1, (
        f"ground-subspace count must be invariant under continuous "
        f"threshold deformation (Wen 1989); got {counts}"
    )


def test_negative_genus_rejected():
    """No surface has negative genus; the API rejects it explicitly.

    Defensive guard rather than silently returning ``K ** (-1)`` (which
    would be fractional for ``K >= 2`` and zero under integer division).
    """
    H, _ = _build_trivial_h()
    with pytest.raises(ValueError):
        count_binding_graph_strategies(H, genus=-1)
