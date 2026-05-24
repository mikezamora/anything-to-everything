"""§12.9 Self-modification via meta-Hamiltonian — acceptance tests.

These tests pin the operator-algebraic contract on the real S1
:class:`MPO` substrate (no mocks, no AST manipulation, no parsed rule
identifiers — the meta-flow is reshape + linear algebra on the operator
tensor network only).

Spec §12.9 acceptance bullets (mapped to tests):

  * encode→decode is invertible (round-trip preserves the lower-level H);
  * imag-time meta-evolution preserves the **Hermitian** subspace of the
    decoded H (the meta-H's kernel by construction);
  * meta-energy monotonically decreases under imag-time evolution
    (impossibility-free meta-relaxation);
  * a decoded H is anomaly-extractable in the §12.1 sense — its
    operator-algebraic Hermiticity anomaly is exactly zero whenever the
    lower-level H was Hermitian on input.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.meta_hamiltonian import (
    DEFAULT_META_ANOMALY_FLOOR,
    decode_meta_state_to_hamiltonian,
    encode_hamiltonian_as_meta_state,
    hermiticity_anomaly,
    hermiticity_meta_hamiltonian,
    meta_energy,
    meta_evolve,
)
from src.qft_pcn.qft.mpo import MPO
from src.qft_pcn.qft.mps import MPS


# ---------------------------------------------------------------------------
# Test fixtures: small Hermitian Hs built from the real S1 MPO substrate
# ---------------------------------------------------------------------------


def _pauli_z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _pauli_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def _hermitian_local_h(N: int = 4, d: int = 2) -> MPO:
    """``H = Σ_k Z_k + 0.3 * X_k`` on N sites — bond-dim-2 Schollwock MPO.

    Real-symmetric local blocks (Hermitian on d=2), so the meta-flow's
    kernel contains this H exactly.
    """
    z = _pauli_z()
    x = _pauli_x()
    terms: list[tuple[int, np.ndarray]] = []
    for k in range(N):
        terms.append((k, z + 0.3 * x))
    return MPO.from_hamiltonian_sum(terms, N=N, d=d)


def _nonhermitian_perturbed_meta_state(N: int = 4, d: int = 2):
    """Encode a Hermitian H, then inject a small antisymmetric component
    into the meta-MPS to drive ``meta_energy > 0``."""
    H = _hermitian_local_h(N=N, d=d)
    meta = encode_hamiltonian_as_meta_state(H)
    # Antisymmetric perturbation in vec-space: pick a bulk site, add a
    # term to the (s_out=0, s_in=1) vec-component on a fresh "channel"
    # of the operator bond. Concretely, mutate the d²-leg directly with
    # an antisymmetric matrix's vectorization.
    perturb = 0.05 * (np.array([[0, 1], [-1, 0]], dtype=complex)).reshape(d * d)
    # Apply to site 1 (bulk site): add perturb broadcast along bonds.
    site_idx = 1
    t = meta.mps.tensors[site_idx]
    chi_l, d2, chi_r = t.shape
    # Add perturb to a SPECIFIC bond cell (the "I" cell at [0, :, 0]
    # carries the bulk identity for the Schollwock construction).
    t_new = t.copy()
    t_new[0, :, 0] = t_new[0, :, 0] + perturb
    new_tensors = list(meta.mps.tensors)
    new_tensors[site_idx] = t_new
    from src.qft_pcn.composition.meta_hamiltonian import MetaState
    return MetaState(mps=MPS(tensors=new_tensors), d_state=d)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_encode_decode_hamiltonian_round_trip():
    """Encoding then decoding an MPO returns the same operator (tensor-
    wise equality, since both transforms are pure reshapes)."""
    H = _hermitian_local_h(N=4, d=2)
    meta = encode_hamiltonian_as_meta_state(H)
    H_back = decode_meta_state_to_hamiltonian(meta)
    assert H_back.N == H.N
    assert H_back.d == H.d
    assert H_back.bond_dimensions() == H.bond_dimensions()
    for W_orig, W_back in zip(H.tensors, H_back.tensors):
        assert W_orig.shape == W_back.shape
        np.testing.assert_allclose(W_back, W_orig, atol=1e-14, rtol=0)
    # And the dense materializations agree — the operator is preserved.
    np.testing.assert_allclose(
        H_back.to_dense(), H.to_dense(), atol=1e-12, rtol=0
    )


def test_meta_evolution_preserves_hermiticity():
    """A Hermitian H encoded as a meta-state and evolved under the
    Hermiticity-projection meta-H remains in the meta-kernel: the
    decoded H is still Hermitian (anomaly = 0).
    """
    H = _hermitian_local_h(N=4, d=2)
    # Confirm input is Hermitian (sanity).
    H_dense = H.to_dense()
    np.testing.assert_allclose(H_dense, H_dense.conj().T, atol=1e-12, rtol=0)
    initial_anomaly = hermiticity_anomaly(H)
    assert initial_anomaly < DEFAULT_META_ANOMALY_FLOOR

    meta = encode_hamiltonian_as_meta_state(H)
    meta_evolved = meta_evolve(meta, dt=0.05, steps=10)
    H_evolved = decode_meta_state_to_hamiltonian(meta_evolved)
    # Decoded H is still Hermitian (full-dense check).
    H_evolved_dense = H_evolved.to_dense()
    np.testing.assert_allclose(
        H_evolved_dense, H_evolved_dense.conj().T, atol=1e-10, rtol=0
    )
    # Per-tensor non-Hermiticity anomaly still vanishes.
    assert hermiticity_anomaly(H_evolved) < DEFAULT_META_ANOMALY_FLOOR


def test_meta_evolution_reduces_meta_energy():
    """Imag-time meta-evolution monotonically reduces meta-energy.

    Starting from a meta-state that has been perturbed off the
    Hermiticity-kernel, evolving under ``h_meta = (I-S)†(I-S)`` strictly
    decreases the meta-energy. (Equality only at the kernel.)
    """
    meta_perturbed = _nonhermitian_perturbed_meta_state(N=4, d=2)
    e0 = meta_energy(meta_perturbed)
    assert e0 > 0.0, "test fixture should start off the meta-kernel"

    meta_after = meta_evolve(meta_perturbed, dt=0.1, steps=20)
    e1 = meta_energy(meta_after)

    # Imag-time relaxation: meta-energy strictly decreased.
    assert e1 < e0
    # And approaches zero (the meta-kernel) — strict for this PSD h_meta.
    assert e1 < 0.5 * e0


def test_anomaly_extraction_from_meta_state():
    """The Hermiticity anomaly extracted from the decoded H is a real,
    operator-algebraic quantity (§12.1 contract on the §12.9 output).

    Concretely: starting from a non-Hermitian-perturbed meta-state, the
    decoded H has positive anomaly; after meta-evolution the anomaly
    drops to zero (the meta-flow eliminates the obstruction without
    touching the rest of the operator structure).
    """
    meta_perturbed = _nonhermitian_perturbed_meta_state(N=4, d=2)
    H_dirty = decode_meta_state_to_hamiltonian(meta_perturbed)
    a_dirty = hermiticity_anomaly(H_dirty)
    assert a_dirty > DEFAULT_META_ANOMALY_FLOOR

    meta_clean = meta_evolve(meta_perturbed, dt=0.1, steps=40)
    H_clean = decode_meta_state_to_hamiltonian(meta_clean)
    a_clean = hermiticity_anomaly(H_clean)
    # Anomaly cleared — the §12.9 flow has restored §12.1 spectral
    # consistency.
    assert a_clean < DEFAULT_META_ANOMALY_FLOOR
    # And the meta-step is real, not a mock: the dense decoded H is now
    # Hermitian within tight tolerance.
    H_clean_dense = H_clean.to_dense()
    np.testing.assert_allclose(
        H_clean_dense, H_clean_dense.conj().T, atol=1e-7, rtol=0
    )


def test_meta_hamiltonian_is_hermitian_psd():
    """``h_meta`` itself is Hermitian and PSD by construction — verified
    on the actual matrix returned by :func:`hermiticity_meta_hamiltonian`.
    """
    h = hermiticity_meta_hamiltonian(d=2)
    np.testing.assert_allclose(h, h.conj().T, atol=1e-14, rtol=0)
    eigs = np.linalg.eigvalsh(h)
    assert eigs.min() >= -1e-12
    # Kernel dimension = d(d+1)/2 = 3 (symmetric 2x2 subspace).
    n_kernel = int(np.sum(eigs < 1e-10))
    assert n_kernel == 3


@pytest.mark.xfail(
    reason=(
        "real-symmetric projector; complex Hermitian projector deferred — "
        "see EXTENSIONS.md §12.9 complex-block entry (antilinear "
        "Choi-Jamiolkowski projector needed for iσ_y-type terms)."
    ),
    strict=True,
)
def test_complex_hermitian_h_with_pauli_y_term():
    """Scope-limitation pin: a Hermitian H with a complex Pauli-Y term
    is NOT preserved by the real-symmetric meta-projector.

    ``σ_y = [[0,-i],[i,0]]`` is Hermitian (``σ_y = σ_y†``) but
    *antisymmetric* (``σ_y = -σ_yᵀ``). The current
    ``(I-S)†(I-S)`` meta-H projects onto the symmetric-matrix subspace
    in vec-space, which has σ_y in its *cokernel* — meta-evolution
    drives σ_y → 0 even though it is Hermitian. The correct (antilinear
    Choi-Jamiołkowski) projector would preserve σ_y because it is in
    the Hermitian subspace.

    Acceptance criterion: after meta-evolution, the σ_y component of
    the decoded H should still be present. Under the current projector
    it is erased — xfail strict. When the antilinear CJ projector lands
    (EXTENSIONS.md §12.9), this test should pass.
    """
    d = 2
    N = 4
    pauli_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    pauli_y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    h_local = pauli_z + 0.5 * pauli_y
    # h_local is Hermitian (σ_y is Hermitian) but NOT symmetric.
    np.testing.assert_allclose(h_local, h_local.conj().T, atol=1e-14, rtol=0)
    assert not np.allclose(h_local, h_local.T)

    terms = [(k, h_local) for k in range(N)]
    H = MPO.from_hamiltonian_sum(terms, N=N, d=d)

    meta = encode_hamiltonian_as_meta_state(H)
    meta_evolved = meta_evolve(meta, dt=0.05, steps=20)
    H_evolved = decode_meta_state_to_hamiltonian(meta_evolved)

    # Probe σ_y content of the decoded H by inner-product of each
    # nontrivial bond-cell block with σ_y. The encoded H has σ_y
    # coefficient 0.5 per site; under the symmetric-only projector
    # that content is driven to ~0.
    sy_content = []
    for W in H_evolved.tensors:
        chi_l, _, _, chi_r = W.shape
        for il in range(chi_l):
            for ir in range(chi_r):
                block = W[il, :, :, ir]
                if np.linalg.norm(block) > 1e-6:
                    overlap = np.abs(np.trace(pauli_y.conj().T @ block)) / 2.0
                    sy_content.append(overlap)
    max_sy = max(sy_content) if sy_content else 0.0
    # Antilinear projector would preserve σ_y → max_sy ≈ 0.5.
    # Current symmetric projector erases it → max_sy ≈ 0.
    assert max_sy > 0.4, (
        f"σ_y Hermitian content was erased by symmetric-only projector "
        f"(max_sy={max_sy:.3e}); needs antilinear CJ projector "
        f"(EXTENSIONS.md §12.9)."
    )
