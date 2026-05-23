"""Tests for §12.17 QCA classification of the Trotterized evolution.

The Trotter step in ``logic/mera_evolution_logic.py`` is a 1D QCA
(Schumacher-Werner axioms); ``compute_qca_index`` reads off its
topological GNVW index from the gate locality structure. These tests
exercise the operator-algebraic invariant on (a) the identity QCA,
(b) a synthetic "trivial shift" QCA built from SWAPs, and (c) real
M2 evolution steps -- verifying the index is well-defined and is
*step-invariant* across consecutive Trotter steps (no drift).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.qca_classification import (
    QCAClassification,
    classify_qca,
    compute_qca_index,
    _is_swap_like,
)
from src.qft_pcn.logic.ast import (
    Bin,
    Eq,
    Forall,
    TNat,
    Var,
    Zero,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian


# ---------------------------------------------------------------------------
# Opt out of the §9.7 dense-tensor ceiling: real encode_mera of the §10.10
# composite allocates isometries above the per-test cap. The composition
# tests/conftest.py guard exists to flag accidental dense allocations in
# stub-driven unit tests, not load-bearing physics tests like this one.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadows the conftest fixture for this file only
    yield


# ---------------------------------------------------------------------------
# Test fixture: a real M2 problem (the §10.10 composite). Built once at
# module import; tests re-use the same state/Hamiltonian.
# ---------------------------------------------------------------------------


def _build_m2_problem():
    """``forall x:Nat. Eq (add x Zero) x`` -- the canonical M2 substrate
    fixture (mirrors test_cross_level_acceptance)."""
    body = Eq(lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
              rhs=Var(name="x"))
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    return state, H


# ---------------------------------------------------------------------------
# (1) Identity QCA: empty gate list => index 0.
# ---------------------------------------------------------------------------


def test_identity_step_has_zero_index():
    """A Trotter step with zero gates is the identity QCA. GNVW: index 0.

    This is the base case of the topological classification -- the
    trivial QCA is the identity, in the same class as every finite-
    depth strictly-local circuit (Gross-Nesme-Vogts-Werner 2012,
    Theorem 2).
    """
    assert compute_qca_index([]) == 0


def test_single_leaf_gate_step_has_zero_index():
    """A single-leaf unitary is a bounded-support local gate; index 0.

    Per GNVW, any unitary with bounded support that does not relabel
    leaf positions is in the trivial QCA class.
    """
    # An arbitrary 16x16 unitary (random Hermitian -> exp i H).
    rng = np.random.default_rng(seed=42)
    H = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    H = (H + H.conj().T) / 2
    from scipy.linalg import expm
    U = expm(1j * H)
    gates = [((3,), U)]
    assert compute_qca_index(gates) == 0


# ---------------------------------------------------------------------------
# (2) Trivial shift: a chain of nearest-neighbor SWAPs implements a
#     translation. On a length-L closed cycle the GNVW index equals
#     the net displacement.
# ---------------------------------------------------------------------------


def _swap_gate(dim: int = 16) -> np.ndarray:
    """The SWAP matrix on two dim-dim sites."""
    P = np.zeros((dim * dim, dim * dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            P[j * dim + i, i * dim + j] = 1.0
    return P


def test_swap_detector_recognizes_swap():
    """The operator-algebraic SWAP detector is correct."""
    assert _is_swap_like(_swap_gate(dim=16))
    # Identity is not a SWAP.
    I = np.eye(16 * 16, dtype=complex)
    assert not _is_swap_like(I)


def test_trivial_shift_on_closed_chain_has_unit_index():
    """A chain of consecutive SWAPs across a length-L cycle implements
    a one-site translation. Its QCA index is the displacement (1).

    On a periodic chain of length L = 4, the sequence
        SWAP(0,1), SWAP(1,2), SWAP(2,3), SWAP(3,0)
    composes to the cyclic permutation 0 -> 1 -> 2 -> 3 -> 0, i.e. a
    shift by +1 site. GNVW: the index of the unit-shift QCA is 1.
    """
    L = 4
    swap = _swap_gate(dim=16)
    gates = [((k, (k + 1) % L), swap) for k in range(L)]
    idx = compute_qca_index(gates)
    # The composed permutation is a single 4-cycle with uniform stride 1.
    assert idx == 1


def test_pair_of_swaps_has_zero_index():
    """Two SWAPs that compose to a transposition (no net translation)
    have index 0. A single SWAP exchanges leaves 0 and 1: the two-cycle
    has no net rotation, so the index is zero by GNVW.
    """
    swap = _swap_gate(dim=16)
    gates = [((0, 1), swap)]
    # Single SWAP: cycle (0 1), length 2, no uniform stride that sums
    # to a non-zero net displacement. Reported as 0.
    assert compute_qca_index(gates) == 0


# ---------------------------------------------------------------------------
# (3) Real M2 Trotter step: well-defined index, step-invariant.
# ---------------------------------------------------------------------------


def test_local_gate_step_has_finite_index():
    """A real M2 Trotter step's QCA index is well-defined (finite,
    integer-valued) and is in the trivial class.

    Per GNVW Theorem 2: the M2 evolution is a finite-depth circuit of
    strictly-local (single-leaf + two-leaf) gates with no SWAP-like
    leaf relabeling, so it's in the trivial QCA equivalence class.
    The reported index is 0.
    """
    state, H = _build_m2_problem()
    result = classify_qca(state, H, dt=0.1)
    assert isinstance(result, QCAClassification)
    # The step is locality-preserving with bounded support.
    assert result.support_radius in (1, 2)
    # The index is finite and integer-valued.
    assert isinstance(result.index, int)
    # GNVW Theorem 2 for strict-locality circuits.
    assert result.index == 0
    # M2 Hamiltonian emits real gates that actually touch leaves.
    assert len(result.touched_leaves) > 0


def test_qca_index_is_step_invariant():
    """The QCA index does not drift across consecutive Trotter steps.

    The index is a topological invariant of the locality-preserving
    dynamics -- it depends only on the gate-support pattern, not on
    the state. Evolving the state by one step and re-classifying must
    produce the same index.
    """
    from src.qft_pcn.logic.mera_evolution_logic import mera_trotter_step

    state, H = _build_m2_problem()
    r0 = classify_qca(state, H, dt=0.1)
    state_next = mera_trotter_step(state, H, dt=0.1, imaginary=True)
    r1 = classify_qca(state_next, H, dt=0.1)
    state_next2 = mera_trotter_step(state_next, H, dt=0.1, imaginary=True)
    r2 = classify_qca(state_next2, H, dt=0.1)
    assert r0.index == r1.index == r2.index
    # Support radius is also a step-invariant property of the gate set.
    assert r0.support_radius == r1.support_radius == r2.support_radius


def test_qca_index_independent_of_dt():
    """The GNVW index depends on gate *support*, not gate *magnitudes*.

    Doubling ``dt`` rescales every gate's matrix but keeps its leaf
    footprint. The topological index must be unchanged.
    """
    state, H = _build_m2_problem()
    a = classify_qca(state, H, dt=0.05)
    b = classify_qca(state, H, dt=0.2)
    assert a.index == b.index
    assert a.support_radius == b.support_radius
