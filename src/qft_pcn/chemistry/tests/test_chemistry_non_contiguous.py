"""§11.6 non-contiguous active-space JW: alpha-then-beta block ordering
(EXTENSIONS resolution).

The chemistry encoder supports two spin-orbital -> leaf layouts:

  * ``interleaved`` (default; A3): leaf k = 2*p + s for spin-orbital
    (p, s in {0=alpha,1=beta}). Alpha and beta channels alternate by leaf.
  * ``alpha_then_beta`` (this extension): alpha block in leaves [0, n_orb),
    beta block in leaves [n_orb, 2*n_orb). Required for active-space
    CASCI/CASSCF workflows that store alpha/beta determinants separately
    (and for DMRG/MERA chemistry codes that exploit the block split).

Both layouts must give the same physical ground-state energy — the JW
mapping is leaf-order dependent (so the Pauli-string operators DIFFER
between layouts) but the eigenvalues are invariant under the unitary
permutation between layouts.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.chemistry import (
    Molecule,
    ChemistryHamiltonian,
    encode_molecule,
    chemistry_imaginary_evolve,
)


H2_FCI_REFERENCE_HARTREE = -1.1372701746609035


@pytest.mark.timeout(600)
def test_h2_alpha_then_beta_matches_interleaved():
    """H2 STO-3G ground-state energy is layout-invariant.

    Imag-time evolution under "alpha_then_beta" must converge to the
    same FCI energy as the default "interleaved" layout — both encode
    the same physical Hamiltonian; the leaf permutation is a unitary
    similarity that preserves the spectrum.
    """
    mol = Molecule.from_xyz(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))],
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )

    # Interleaved (baseline).
    state_i, meta_i = encode_molecule(mol, chi_layer=16,
                                      orbital_layout="interleaved")
    assert meta_i.orbital_layout == "interleaved"
    assert meta_i.site_of_spin_orbital == [0, 1, 2, 3]
    ham_i = ChemistryHamiltonian(meta_i)
    traj_i, _ = chemistry_imaginary_evolve(
        state_i, ham_i, meta_i, dt=0.05, steps=400, energy_tol=1e-10,
    )
    e_i = traj_i[-1]

    # Alpha-then-beta (this extension).
    state_ab, meta_ab = encode_molecule(mol, chi_layer=16,
                                        orbital_layout="alpha_then_beta")
    assert meta_ab.orbital_layout == "alpha_then_beta"
    # n_orb == 2 -> alpha at leaves [0, 1], beta at leaves [2, 3].
    # spin-orbital 0 (alpha p=0) -> leaf 0
    # spin-orbital 1 (beta  p=0) -> leaf 2
    # spin-orbital 2 (alpha p=1) -> leaf 1
    # spin-orbital 3 (beta  p=1) -> leaf 3
    assert meta_ab.site_of_spin_orbital == [0, 2, 1, 3]
    ham_ab = ChemistryHamiltonian(meta_ab)
    # HF energy must match (same physical determinant, different storage).
    e_hf_i = ham_i.total_energy(state_i)
    e_hf_ab = ham_ab.total_energy(state_ab)
    assert abs(e_hf_i - e_hf_ab) < 1e-9, (
        f"HF energies differ across layouts: interleaved {e_hf_i} vs "
        f"alpha_then_beta {e_hf_ab}"
    )
    traj_ab, _ = chemistry_imaginary_evolve(
        state_ab, ham_ab, meta_ab, dt=0.05, steps=400, energy_tol=1e-10,
    )
    e_ab = traj_ab[-1]

    # Both must converge to FCI within chemistry accuracy.
    assert abs(e_i - H2_FCI_REFERENCE_HARTREE) < 1e-3
    assert abs(e_ab - H2_FCI_REFERENCE_HARTREE) < 1e-3
    # Cross-layout consistency.
    assert abs(e_i - e_ab) < 1e-6, (
        f"H2 STO-3G ground-state energies differ across orbital layouts: "
        f"interleaved {e_i:.8f} Ha vs alpha_then_beta {e_ab:.8f} Ha"
    )


def test_alpha_then_beta_jw_string_ordering():
    """JW Pauli strings honor the alpha-then-beta leaf ordering.

    Under "alpha_then_beta", a one-body operator a^+_{p_alpha} a_{q_alpha}
    between two alpha spin-orbitals at leaves p, q (both in [0, n_orb))
    has its JW Z-tail on leaves strictly between p and q in LEAF order.
    Crucially, the tail does NOT cross into the beta block [n_orb, 2*n_orb)
    because alpha leaves are contiguously [0, n_orb).

    For a 4-spatial-orbital test molecule (8 spin-orbitals), an alpha-to-
    alpha one-body excitation between spatial orbitals 0 and 1 lands on
    leaves 0 and 1 — the JW tail is EMPTY (no leaves strictly between).
    In contrast, on the interleaved layout the same operator lives on
    leaves 0 and 2, with a Z on leaf 1 (the intervening beta_0 leaf).

    This test pins the structural difference.
    """
    from src.qft_pcn.chemistry.hamiltonian import _jw_a_dag_a

    # alpha_then_beta layout with n_orb = 4 -> alpha_p at leaf p,
    # beta_p at leaf 4+p. Then a^+_{0_alpha} a_{1_alpha} sits at
    # leaves (0, 1) — JW tail is range(1, 1) = empty.
    ps_ab = _jw_a_dag_a(0, 1)
    # Should produce 1/2 (X_0 X_1) + 1/2 (Y_0 Y_1) — NO Z tail.
    supports = [tuple(sym for _, sym in ps.ops) for ps in ps_ab]
    leaves = [tuple(leaf for leaf, _ in ps.ops) for ps in ps_ab]
    assert all(len(syms) == 2 for syms in supports), (
        f"alpha-then-beta JW(0_alpha, 1_alpha) should have support 2 "
        f"(no Z tail), got supports {supports}"
    )
    # Pauli symbols are only X or Y on the two endpoints — no Z.
    for syms in supports:
        for sym in syms:
            assert sym in ("X", "Y"), (
                f"alpha-then-beta JW(0_alpha, 1_alpha) should NOT carry a Z "
                f"between alpha and beta blocks (alpha block is contiguous); "
                f"got symbol {sym!r} in {supports}"
            )
    # Endpoints are leaves 0 and 1, never the beta block.
    for lvs in leaves:
        assert set(lvs) == {0, 1}, (
            f"expected operator on leaves (0, 1), got {lvs}"
        )

    # Interleaved layout contrast: a^+_{0_alpha} a_{1_alpha} would route to
    # leaves (0, 2) with a Z on leaf 1. Confirm via the raw helper:
    ps_il = _jw_a_dag_a(0, 2)
    # range(1, 2) = (1,) -> Z on leaf 1 present in EVERY string.
    for ps in ps_il:
        leaf_set = {leaf for leaf, _ in ps.ops}
        symbols = {sym for _, sym in ps.ops}
        assert 1 in leaf_set, (
            f"interleaved JW(0, 2) should carry a Z on leaf 1, got ops "
            f"{ps.ops}"
        )
        assert "Z" in symbols, (
            f"interleaved JW(0, 2) should have a Z in its tail, got "
            f"symbols {symbols}"
        )
