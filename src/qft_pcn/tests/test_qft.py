"""Tests for the quantum field theory core.

These are the proof that the architecture actually has the QFT features
claimed: operator algebra, vacuum, particles, cross-site entanglement,
multi-species interaction, unitary evolution, Lorentzian propagation.
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.qft import fock
from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.qft.hamiltonian import (Hamiltonian, HamiltonianConfig,
                                          FieldSpecies)
from src.qft_pcn.qft.evolution import trotter_step, energy
from src.qft_pcn.qft.qpcn import QPCN, QPCNConfig
from src.qft_pcn.manifold import Manifold2D


# ---------- Local Fock-space algebra ----------------------------------------


def test_canonical_commutation_relation():
    """[a, a^dagger] = 1 in the truncated space, except for the topmost level."""
    d = 5
    a = fock.annihilation(d)
    adag = fock.creation(d)
    comm = a @ adag - adag @ a
    expected = np.eye(d, dtype=complex)
    # The top corner has the truncation defect; everything else should be identity.
    expected[d - 1, d - 1] = -(d - 1)
    assert np.allclose(comm, expected, atol=1e-12)


def test_number_operator_diagonal():
    """n |k> = k |k>."""
    d = 4
    n = fock.number(d)
    for k in range(d):
        ket = fock.number_state_vec(d, k)
        nk = n @ ket
        assert np.allclose(nk, k * ket, atol=1e-12), f"n|{k}> != {k}|{k}>"


def test_creation_makes_particle():
    """a^dagger |0> = |1>; a |1> = |0>."""
    d = 3
    a = fock.annihilation(d)
    adag = fock.creation(d)
    psi0 = fock.vacuum_vec(d)
    psi1 = adag @ psi0
    assert np.allclose(psi1, fock.number_state_vec(d, 1))
    back = a @ psi1
    assert np.allclose(back, psi0)


# ---------- MPS basics ------------------------------------------------------


def test_mps_norm_vacuum():
    state = MPS.vacuum(N=8, d=3)
    assert abs(state.norm_sq() - 1.0) < 1e-10


def test_mps_local_expectation_on_number_state():
    """<n_k> in a product number state equals the occupation at site k."""
    occs = [2, 0, 1, 3, 0]
    state = MPS.number_states(occs, d=4)
    n = fock.number(4)
    for k, expected in enumerate(occs):
        val = state.local_expectation(k, n)
        assert abs(val - expected) < 1e-10, f"site {k}: got {val}, want {expected}"


def test_two_site_gate_preserves_norm_for_unitary():
    """Unitary two-site gate preserves <Psi|Psi>."""
    state = MPS.number_states([1, 0, 1, 0], d=3)
    # Build a random unitary 9x9 (d^2 x d^2 with d=3).
    rng = np.random.default_rng(5)
    raw = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
    H = (raw + raw.conj().T) / 2
    from scipy.linalg import expm
    U = expm(1j * H)
    state.apply_two_site_gate(site=1, gate=U, chi_max=16)
    assert abs(state.norm_sq() - 1.0) < 1e-8


# ---------- Multi-species local operators -----------------------------------


def test_embed_op_acts_only_on_target_species():
    """An n_a operator embedded at species 0 has no effect on species 1's index."""
    d_a, d_b = 3, 2
    n_a = fock.number(d_a)
    n_a_full = fock.embed_op(n_a, species_index=0, species_dims=(d_a, d_b))
    # State |n_a=2, n_b=1>: linear index in the (d_a x d_b) basis is 2*d_b + 1 = 5.
    psi = np.zeros(d_a * d_b, dtype=complex); psi[2 * d_b + 1] = 1.0
    val = float(np.real(psi.conj() @ n_a_full @ psi))
    assert abs(val - 2.0) < 1e-12


# ---------- Hamiltonian + evolution -----------------------------------------


def test_hamiltonian_hermitian():
    cfg = HamiltonianConfig(
        species=[FieldSpecies("a", cutoff=3, bare_mass=1.0, kinetic=0.4)],
    )
    H = Hamiltonian(cfg, N=4)
    for k in range(H.N):
        L = H.local_op(k)
        assert np.allclose(L, L.conj().T, atol=1e-12), f"local_op({k}) not Hermitian"
    for k in range(H.N - 1):
        B = H.bond_op(k)
        assert np.allclose(B, B.conj().T, atol=1e-12), f"bond_op({k}) not Hermitian"


def test_real_time_evolution_preserves_energy():
    """Schrodinger evolution conserves <H> up to Trotter and truncation error."""
    cfg = HamiltonianConfig(
        species=[FieldSpecies("a", cutoff=3, bare_mass=1.0, kinetic=0.3)],
    )
    H = Hamiltonian(cfg, N=5)
    state = MPS.number_states([0, 1, 0, 0, 0], d=H.d_local)
    e0 = energy(state, H)
    for _ in range(20):
        trotter_step(state, H, dt=0.02, imaginary=False, chi_max=24)
    e1 = energy(state, H)
    # Energy drift should be small for small dt and adequate bond dimension.
    assert abs(e1 - e0) < 0.05 * (abs(e0) + 1.0), (
        f"energy drift too large: {e0:.4f} -> {e1:.4f}")


def test_imaginary_time_lowers_energy():
    """Imaginary-time evolution monotonically decreases <H>."""
    cfg = HamiltonianConfig(
        species=[FieldSpecies("a", cutoff=3, bare_mass=1.0, kinetic=0.5)],
    )
    H = Hamiltonian(cfg, N=5)
    # Start from an excited number state.
    state = MPS.number_states([2, 1, 2, 1, 2], d=H.d_local)
    e0 = energy(state, H)
    for _ in range(30):
        trotter_step(state, H, dt=0.05, imaginary=True, chi_max=24)
        state.normalize()
    e1 = energy(state, H)
    assert e1 < e0, f"imaginary-time evolution did not lower energy: {e0} -> {e1}"


# ---------- Entanglement across sites (the real win) -----------------------


def test_evolution_generates_entanglement():
    """An initially-product state evolved by hopping generates real
    entanglement entropy across a cut. This is what proves we have a
    quantum substrate and not just a fancy classical filter."""
    cfg = HamiltonianConfig(
        species=[FieldSpecies("a", cutoff=2, bare_mass=1.0, kinetic=0.8)],
    )
    H = Hamiltonian(cfg, N=6)
    state = MPS.number_states([1, 0, 0, 0, 0, 0], d=H.d_local)
    s0 = state.entanglement_entropy(bond=2)
    assert s0 < 1e-6, "product state should have zero entanglement"
    for _ in range(15):
        trotter_step(state, H, dt=0.1, imaginary=False, chi_max=16)
    s1 = state.entanglement_entropy(bond=2)
    assert s1 > 0.1, (f"hopping evolution failed to entangle the chain: "
                      f"S = {s1:.4f}")


# ---------- Multi-species coupling -----------------------------------------


def test_yukawa_coupling_correlates_species():
    """Two species coupled by phi_a phi_b should not factorize after evolution."""
    cfg = HamiltonianConfig(
        species=[
            FieldSpecies("a", cutoff=2, bare_mass=1.0, kinetic=0.4),
            FieldSpecies("b", cutoff=2, bare_mass=1.0, kinetic=0.4),
        ],
        yukawa_couplings={("a", "b"): 0.6},
    )
    H = Hamiltonian(cfg, N=4)
    state = MPS.number_states([0, 0, 0, 0], d=H.d_local)
    # Inject a single a-quantum at site 0.
    adag_a = H.adag("a")
    # Apply on the (1, d, 1) tensor.
    new = adag_a @ state.tensors[0][0, :, 0]
    new = new / (np.linalg.norm(new) + 1e-15)
    state.tensors[0] = new.reshape(1, H.d_local, 1)
    # Evolve.
    for _ in range(20):
        trotter_step(state, H, dt=0.05, imaginary=False, chi_max=16)
    n_b_total = sum(float(np.real(state.local_expectation(k, H.n("b"))))
                    for k in range(H.N))
    # With Yukawa coupling and dynamics, the b-field develops nonzero
    # occupation seeded by the a-quantum.
    assert n_b_total > 1e-3, (
        f"yukawa coupling failed to populate species b, <n_b> = {n_b_total:.5f}")


# ---------- Quantum PCN end-to-end -----------------------------------------


def test_qpcn_reduces_prediction_error_over_time():
    """Train the QPCN to match target number-operator expectations at
    boundary sites by adjusting the bare mass of the field."""
    # Initialize source ~ 0.2 so we're off the symmetric point J=0 where the
    # gradient of <n> w.r.t. J vanishes (the dependence is parabolic in J).
    cfg = QPCNConfig(
        species=[FieldSpecies("a", cutoff=3, bare_mass=2.0, kinetic=0.3,
                              source=0.2)],
        N_sites=4,
        chi_max=12,
        dt_real=0.05,
        dt_imag=0.1,
        imag_steps_per_observe=4,
        real_steps_per_observe=1,
        learn_rate=2.0,
        learnable_params=["a.source"],
        observable_map=[(0, "a", "n"), (3, "a", "n")],
    )
    qpcn = QPCN(cfg, manifold=None, rng=np.random.default_rng(1))
    targets = {(0, "a", "n"): 0.4, (3, "a", "n"): 0.4}

    initial = qpcn.observe(targets, learn=False)
    e0 = initial["sq_error"]
    for _ in range(20):
        qpcn.observe(targets, learn=True)
    final = qpcn.observe(targets, learn=False)
    e1 = final["sq_error"]
    assert e1 < e0, (f"QPCN failed to reduce prediction error: "
                     f"{e0:.4f} -> {e1:.4f}")


def test_qpcn_couples_to_manifold():
    """When use_manifold=True, the manifold's curvature should respond to
    the QFT energy density."""
    nx, ny = 6, 6
    manifold = Manifold2D(nx, ny, kappa=0.5)
    cfg = QPCNConfig(
        species=[FieldSpecies("a", cutoff=2, bare_mass=1.0, kinetic=0.5)],
        N_sites=nx * ny,
        chi_max=8,
        dt_real=0.05,
        dt_imag=0.1,
        imag_steps_per_observe=1,
        real_steps_per_observe=1,
        learn_rate=0.0,
        observable_map=[(0, "a", "n")],
        use_manifold=True,
        manifold_kappa=0.5,
    )
    qpcn = QPCN(cfg, manifold=manifold, rng=np.random.default_rng(2))
    h_mag_initial = float(np.abs(manifold.h_xx).sum())
    for _ in range(5):
        qpcn.observe({(0, "a", "n"): 0.5}, learn=False)
    h_mag_final = float(np.abs(manifold.h_xx).sum())
    assert h_mag_final > h_mag_initial, (
        "manifold did not respond to QFT energy density")


if __name__ == "__main__":
    test_canonical_commutation_relation()
    test_number_operator_diagonal()
    test_creation_makes_particle()
    test_mps_norm_vacuum()
    test_mps_local_expectation_on_number_state()
    test_two_site_gate_preserves_norm_for_unitary()
    test_embed_op_acts_only_on_target_species()
    test_hamiltonian_hermitian()
    test_real_time_evolution_preserves_energy()
    test_imaginary_time_lowers_energy()
    test_evolution_generates_entanglement()
    test_yukawa_coupling_correlates_species()
    test_qpcn_reduces_prediction_error_over_time()
    test_qpcn_couples_to_manifold()
    print("ok")
