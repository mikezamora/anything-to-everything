"""Tests for the Matrix Product Operator substrate.

These exercise the six MPO surfaces against dense reference computations
on small systems (N = 3..6, d = 2). They are the substrate acceptance
gate for §12.9 (self-modification meta-Hamiltonian).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.qft.mpo import MPO
from src.qft_pcn.qft.mps import MPS


# ---- helpers ----------------------------------------------------------------

def _pauli() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def _dense_kron(ops_per_site: list[np.ndarray]) -> np.ndarray:
    """Tensor product in left-to-right site order (site 0 outer)."""
    acc = np.array([[1.0 + 0.0j]])
    for op in ops_per_site:
        acc = np.kron(acc, op)
    return acc


def _mps_to_dense_vector(mps: MPS) -> np.ndarray:
    """Materialize the full state vector for tests on small systems."""
    # Contract left-to-right; carries (chi_l, dim_so_far).
    v = np.ones((1, 1), dtype=complex)
    for A in mps.tensors:
        chi_l, d, chi_r = A.shape
        # v: (1, prev_dim) but indexed by chi_l (which equals 1 only for the
        # very first site if previously contracted). General: v[chi_l, dim].
        new = np.einsum('cD,csr->Dsr', v, A)
        D, s, r = new.shape
        v = new.reshape(D * s, r).T  # (chi_r, D*s)
    # Right boundary chi_r = 1.
    return v[0]


# ---- tests -----------------------------------------------------------------

def test_product_mpo_from_local_ops_identity_action():
    """Identity MPO action leaves the state vector untouched."""
    N, d = 4, 2
    rng = np.random.default_rng(7)
    sites = [rng.standard_normal(d) + 1j * rng.standard_normal(d)
             for _ in range(N)]
    psi = MPS.from_product(sites)

    O_id = MPO.identity(N, d)
    assert O_id.bond_dimensions() == [1, 1, 1]
    # Shape check.
    for k in range(N):
        assert O_id.tensors[k].shape == (1, d, d, 1)

    psi_out = O_id.apply_to_mps(psi)
    v_in = _mps_to_dense_vector(psi)
    v_out = _mps_to_dense_vector(psi_out)
    assert np.allclose(v_in, v_out, atol=1e-12)


def test_local_z_mpo_expectation_on_z_eigenstate():
    """Pauli-Z at site 0 has expectation +1 on |0> and -1 on |1>."""
    _, _, _, Z = _pauli()
    I = np.eye(2, dtype=complex)
    N, d = 3, 2

    # |0> at site 0 -> +1
    ket0 = np.array([1, 0], dtype=complex)
    ket1 = np.array([0, 1], dtype=complex)

    psi_plus = MPS.from_product([ket0, ket0, ket0])
    O = MPO.from_local_operators([Z, I, I])
    assert np.isclose(O.expectation(psi_plus), 1.0 + 0.0j, atol=1e-12)

    psi_minus = MPS.from_product([ket1, ket0, ket0])
    assert np.isclose(O.expectation(psi_minus), -1.0 + 0.0j, atol=1e-12)

    # Sanity: Z at a different site.
    O_site1 = MPO.from_local_operators([I, Z, I])
    psi_mid = MPS.from_product([ket0, ket1, ket0])
    assert np.isclose(O_site1.expectation(psi_mid), -1.0 + 0.0j, atol=1e-12)


def test_hamiltonian_sum_mpo_matches_dense_sum():
    """`H = sum_i h_i` via MPO matches the dense sum on a random state."""
    N, d = 5, 2
    _, X, _, Z = _pauli()
    I = np.eye(d, dtype=complex)

    # Random per-site local terms (mixture of X and Z, varying amplitudes).
    rng = np.random.default_rng(42)
    coeffs = rng.standard_normal((N, 2))
    local_terms: list[tuple[int, np.ndarray]] = []
    h_per_site: list[np.ndarray] = []
    for k in range(N):
        h_k = coeffs[k, 0] * X + coeffs[k, 1] * Z
        local_terms.append((k, h_k))
        h_per_site.append(h_k)

    mpo = MPO.from_hamiltonian_sum(local_terms, N=N, d=d)
    assert mpo.bond_dimensions() == [2, 2, 2, 2]

    # Dense reference H = sum_k I ⊗ ... ⊗ h_k ⊗ ... ⊗ I.
    H_dense = np.zeros((d**N, d**N), dtype=complex)
    for k in range(N):
        ops = [I] * N
        ops[k] = h_per_site[k]
        H_dense = H_dense + _dense_kron(ops)

    # Verify MPO.to_dense reproduces the dense reference.
    H_from_mpo = mpo.to_dense()
    assert H_from_mpo.shape == (d**N, d**N)
    assert np.allclose(H_from_mpo, H_dense, atol=1e-10), (
        f"max abs diff = {np.max(np.abs(H_from_mpo - H_dense))}"
    )

    # Verify expectation against random product state.
    sites = [rng.standard_normal(d) + 1j * rng.standard_normal(d)
             for _ in range(N)]
    psi = MPS.from_product(sites)
    psi.normalize()

    exp_mpo = mpo.expectation(psi)
    v = _mps_to_dense_vector(psi)
    exp_dense = v.conj() @ H_dense @ v
    assert np.isclose(exp_mpo, exp_dense, atol=1e-10), (
        f"MPO={exp_mpo}, dense={exp_dense}"
    )


def test_mpo_compose_matches_dense_matmul():
    """MPO composition equals dense operator product."""
    N, d = 4, 2
    I, X, _, Z = _pauli()

    # A = X on site 1; B = Z on site 1 + Z on site 2 (a sum-MPO).
    A = MPO.from_local_operators([I, X, I, I])

    B = MPO.from_hamiltonian_sum(
        [(1, Z), (2, Z)], N=N, d=d
    )

    AB = A.compose(B)
    BA = B.compose(A)

    A_dense = A.to_dense()
    B_dense = B.to_dense()

    assert np.allclose(AB.to_dense(), A_dense @ B_dense, atol=1e-10)
    assert np.allclose(BA.to_dense(), B_dense @ A_dense, atol=1e-10)

    # Sanity: action on a state matches dense action.
    rng = np.random.default_rng(3)
    sites = [rng.standard_normal(d) + 1j * rng.standard_normal(d)
             for _ in range(N)]
    psi = MPS.from_product(sites)
    psi.normalize()
    v = _mps_to_dense_vector(psi)

    exp_AB = AB.expectation(psi)
    exp_dense = v.conj() @ (A_dense @ B_dense) @ v
    assert np.isclose(exp_AB, exp_dense, atol=1e-10)


def test_mpo_apply_then_renormalize_is_unit_norm():
    """Applying an MPO and renormalizing yields a unit-norm MPS (sanity)."""
    N, d = 4, 2
    _, X, _, Z = _pauli()
    I = np.eye(d, dtype=complex)

    # A non-trivial Hermitian sum-of-locals MPO.
    H = MPO.from_hamiltonian_sum(
        [(0, X), (1, Z), (2, X), (3, Z)], N=N, d=d
    )

    rng = np.random.default_rng(11)
    sites = [rng.standard_normal(d) + 1j * rng.standard_normal(d)
             for _ in range(N)]
    psi = MPS.from_product(sites)
    psi.normalize()

    # Action grows operator-bond * state-bond = 2 * 1 = 2 internally.
    phi = H.apply_to_mps(psi)
    assert phi.N == N
    # Bond dimensions grew at interior cuts (operator chi was 2 there).
    assert phi.bond_dimensions() == [2, 2, 2]

    # Renormalize and check unit norm.
    phi.normalize()
    assert np.isclose(phi.norm_sq(), 1.0, atol=1e-12)

    # And the inner product <psi|H|psi> is real (H is Hermitian).
    exp = H.expectation(psi)
    assert np.isclose(exp.imag, 0.0, atol=1e-12), f"imag part {exp.imag}"


def test_mpo_identity_compose_is_identity():
    """Composing with the identity MPO leaves an MPO unchanged (action)."""
    N, d = 3, 2
    _, X, _, Z = _pauli()

    H = MPO.from_hamiltonian_sum(
        [(0, X), (1, Z), (2, X)], N=N, d=d
    )
    I_mpo = MPO.identity(N, d)

    H1 = H.compose(I_mpo)
    H2 = I_mpo.compose(H)

    H_dense = H.to_dense()
    assert np.allclose(H1.to_dense(), H_dense, atol=1e-12)
    assert np.allclose(H2.to_dense(), H_dense, atol=1e-12)


def test_mpo_validation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        MPO(tensors=[np.zeros((2, 2, 2), dtype=complex)])
    with pytest.raises(ValueError):
        MPO(tensors=[np.zeros((1, 2, 3, 1), dtype=complex)])
    with pytest.raises(ValueError):
        # left boundary not 1
        MPO(tensors=[np.zeros((2, 2, 2, 1), dtype=complex)])
    with pytest.raises(ValueError):
        # bond mismatch between sites
        MPO(tensors=[
            np.zeros((1, 2, 2, 3), dtype=complex),
            np.zeros((2, 2, 2, 1), dtype=complex),
        ])
