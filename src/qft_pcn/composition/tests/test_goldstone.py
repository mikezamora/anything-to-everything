"""Acceptance tests for §12.6 Goldstone-mode missing-lemma diagnosis.

These tests exercise the real M2 substrate end-to-end:
  * encode_mera builds the MERA from a real AST
  * MeraEvalHamiltonian is the real evaluation Hamiltonian (spec §7)
  * mera_imaginary_evolve_state is the real imaginary-time descent
  * compute_near_null_subspace runs the real scipy.sparse.linalg.eigsh
    Lanczos solve on the constraint-Hessian matrix

No mocks, no stubs, no skips. The diagnostic is operator-algebraic:
its inputs are real term-energy expectations and real causal-cone leaf
footprints (entanglement-bond pattern, §1.1), not heuristic AST walks.
"""
from __future__ import annotations

import pytest

from src.qft_pcn.composition.goldstone import (
    CandidateMissingLemma,
    DEFAULT_GOLDSTONE_THRESHOLD,
    GoldstoneMode,
    LemmaDiagnostic,
    compute_near_null_subspace,
    diagnose_missing_lemma,
)
from src.qft_pcn.logic.ast import (
    Bin, Eq, Forall, TNat, Var, Zero, parse,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_evolution_logic import mera_imaginary_evolve_state


# ---------------------------------------------------------------------------
# Opt this file out of the composition/tests/conftest.py §9.7 dense-tensor
# memory ceiling. encode_mera allocates 16-dim leaves and isometries
# (4096-element pair matrices) by construction. Same opt-out used in
# test_cross_level_acceptance.py / test_hierarchical_proof_demo.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadows the conftest fixture for this file only
    yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _redex_program():
    """A small redex program ``2 + 3``. Initial <H> > 0; descent
    drives it to ~0."""
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    return state, meta, H


def _addzero_theorem():
    """``forall x:Nat. Eq (add x Zero) x`` -- substrate-supported
    composite that descends to <H>=0 under R-AddZero + R-Eq-Refl with
    Forall-protected leaves."""
    body = Eq(lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
              rhs=Var(name="x"))
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    return state, meta, H


# ---------------------------------------------------------------------------
# §12.6 acceptance
# ---------------------------------------------------------------------------


def test_solved_theorem_has_no_goldstones():
    """A fully relaxed theorem (residual driven to ~0 by imaginary-time
    descent) has its constraint Hamiltonian's spectrum lifted above
    ``threshold``: there is no broken-symmetry direction because every
    constraint is satisfied.

    The diagnostic surfaces no candidates -- ``has_missing_lemma`` is
    False -- because the support filter requires AT LEAST ONE term in
    the mode's support to carry residual above ``residual_floor``. A
    fully-relaxed theorem has every term residual at the descent floor.
    """
    state, meta, H = _addzero_theorem()
    protected = set(meta.forall_protected_leaves)
    _, final = mera_imaginary_evolve_state(
        state, H, dt=0.1, steps=300, chi_layer=16,
        frozen_leaves=protected,
    )
    # Sanity: descent actually drove the theorem to (near) the ground
    # state (this is the §10.10 / I-10 substrate proof; if it fails
    # here, the upstream substrate has regressed, not Goldstone).
    final_energy = float(H.total_energy(final))
    assert final_energy < 1e-3, (
        f"upstream substrate regression: addzero theorem did not "
        f"relax to ground state (residual={final_energy})"
    )

    diag = diagnose_missing_lemma(H, final, k=5)
    assert isinstance(diag, LemmaDiagnostic)
    # Spectrum is non-empty (the Hamiltonian has terms).
    assert len(diag.spectrum) > 0
    # No candidate missing lemma: the theorem is solved.
    assert not diag.has_missing_lemma, (
        f"fully relaxed theorem surfaced spurious missing-lemma "
        f"candidates: {diag.candidates}"
    )
    assert diag.candidates == ()
    assert diag.modes == ()


def test_unsolved_theorem_surfaces_goldstone_mode():
    """A partially-evolved (or un-evolved) theorem has unsatisfied
    constraints. The constraint-Hessian's near-null subspace then
    contains a Goldstone mode whose support points at the AST node
    where the missing reduction (or missing lemma) would apply.

    Concretely: take ``2 + 3`` BEFORE descent. The R-Arith term on the
    Bin node carries residual ~ initial overlap weight; this is the
    "broken symmetry direction" in the sense of §12.6 (the rewrite
    rule has not yet been applied), and the diagnostic should surface
    a candidate at that (rule, node) site.
    """
    state, _, H = _redex_program()
    # Pre-descent residual must be strictly positive (the redex is
    # present); otherwise the test fixture is degenerate.
    initial_energy = float(H.total_energy(state))
    assert initial_energy > 1e-3, (
        f"test fixture invariant: 2+3 must have non-zero initial "
        f"residual, got {initial_energy}"
    )

    diag = diagnose_missing_lemma(H, state, k=5)
    assert isinstance(diag, LemmaDiagnostic)
    assert diag.residual_energy == pytest.approx(initial_energy)
    assert diag.has_missing_lemma, (
        f"unsolved redex did NOT surface any Goldstone mode; "
        f"spectrum={diag.spectrum}, residual={initial_energy}"
    )
    assert len(diag.candidates) >= 1
    # Every candidate's eigenvalue must be below threshold (that's the
    # definition of a Goldstone mode in this implementation).
    for cand in diag.candidates:
        assert isinstance(cand, CandidateMissingLemma)
        assert cand.mode_eigenvalue <= DEFAULT_GOLDSTONE_THRESHOLD
        assert cand.confidence > 0.0
    # At least one mode object surfaced.
    assert len(diag.modes) >= 1
    for mode in diag.modes:
        assert isinstance(mode, GoldstoneMode)
        assert mode.eigenvalue <= DEFAULT_GOLDSTONE_THRESHOLD
        assert len(mode.support) >= 1
    # The 2+3 redex is a Bin node, so R-Arith is the rule the residual
    # is concentrated on. The dominant candidate (highest confidence)
    # should carry the R-Arith signature -- this is the operator-
    # algebraic identification of the rewrite that would close the
    # proof.
    top = diag.candidates[0]
    assert top.rule_id == "R-Arith", (
        f"top Goldstone candidate did not localize on R-Arith for the "
        f"2+3 redex; got {top}"
    )


def test_node_field_matches_redex_position():
    """The dominant Goldstone candidate must localize on the actual AST
    position of the redex, not just match the rule_id. For ``2 + 3``
    the redex is the Bin node at preorder index 0 (the root); the
    R-Arith term carrying the Goldstone amplitude must report
    ``node == 0`` so the diagnostic can be consumed as an
    (AST-position, rule) signature by downstream lemma-search code
    (spec §12.6 "localization on specific sites").
    """
    from src.qft_pcn.logic._serialize import serialize_preorder
    from src.qft_pcn.logic.encoding import KIND_BIN

    state, meta, H = _redex_program()
    # Confirm fixture invariant: the Bin redex sits at preorder index 0
    # for the program ``2 + 3``. If the encoder's preorder convention
    # ever shifts, this gives a clearer failure than a bare integer.
    sites = serialize_preorder(parse(r"2 + 3"), N=meta.n_nodes_max)
    bin_nodes = [i for i, s in enumerate(sites) if s.kind == KIND_BIN]
    assert bin_nodes, "fixture invariant: 2+3 must contain a Bin node"
    expected_bin_node = bin_nodes[0]

    diag = diagnose_missing_lemma(H, state, k=5)
    assert diag.has_missing_lemma, (
        f"unsolved redex did not surface any candidates; "
        f"spectrum={diag.spectrum}"
    )
    top = diag.candidates[0]
    assert top.rule_id == "R-Arith", (
        f"precondition: top candidate should be R-Arith, got {top.rule_id}"
    )
    assert top.node == expected_bin_node, (
        f"top Goldstone candidate did not localize on the Bin redex's "
        f"AST position; expected node={expected_bin_node}, got "
        f"node={top.node} (full candidate: {top})"
    )


def test_diagnostic_is_deterministic():
    """Same input -> byte-identical diagnostic. The Hessian matrix is
    a pure function of (H, state); eigsh / eigh on a fixed matrix are
    deterministic; the candidate ordering breaks ties on
    (confidence, rule_id, node). The full diagnostic dataclass is
    therefore reproducible.
    """
    state, _, H = _redex_program()
    diag_a = diagnose_missing_lemma(H, state, k=5)
    diag_b = diagnose_missing_lemma(H, state, k=5)

    assert diag_a.spectrum == diag_b.spectrum
    assert diag_a.residual_energy == diag_b.residual_energy
    assert diag_a.threshold == diag_b.threshold
    assert len(diag_a.candidates) == len(diag_b.candidates)
    for ca, cb in zip(diag_a.candidates, diag_b.candidates):
        assert ca.rule_id == cb.rule_id
        assert ca.node == cb.node
        assert ca.confidence == cb.confidence
        assert ca.mode_eigenvalue == cb.mode_eigenvalue
    assert len(diag_a.modes) == len(diag_b.modes)
    for ma, mb in zip(diag_a.modes, diag_b.modes):
        assert ma.eigenvalue == mb.eigenvalue
        assert ma.support == mb.support
        assert ma.term_indices == mb.term_indices
        assert ma.eigenvector == mb.eigenvector


# ---------------------------------------------------------------------------
# Defensive: parameter validation + edge cases
# ---------------------------------------------------------------------------


def test_compute_near_null_rejects_zero_k():
    state, _, H = _redex_program()
    with pytest.raises(ValueError, match="k"):
        compute_near_null_subspace(H, state, k=0)


def test_compute_near_null_returns_sorted_ascending():
    """eigsh / eigh may return eigenvalues in arbitrary order; the
    near-null subspace contract is that they come out ASCENDING so
    callers can read the Goldstone modes from index 0."""
    state, _, H = _redex_program()
    eigvals, _ = compute_near_null_subspace(H, state, k=5)
    assert len(eigvals) >= 1
    for i in range(len(eigvals) - 1):
        assert eigvals[i] <= eigvals[i + 1] + 1e-12
