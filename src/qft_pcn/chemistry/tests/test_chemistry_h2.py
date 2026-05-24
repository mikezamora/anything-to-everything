"""H2 STO-3G ground-state energy via the §11.6 chemistry encoder + driver.

Reference: PySCF FCI on H2 at equilibrium bond length (0.7414 Angstrom)
in STO-3G basis gives E_FCI ~= -1.13727 Ha. We require convergence
within chemistry accuracy (1e-3 Ha) starting from the RHF product state.

The test exercises the full §11.6 substrate path:
  - PySCF integral builder (real ab-initio MO integrals)
  - encode_molecule -> product MERA at the HF determinant
  - ChemistryHamiltonian with JW Pauli-string total_energy
  - chemistry_imaginary_evolve ground-state driver
  - branch-superposition MERA reconstruction at the final state
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
def test_h2_sto3g_ground_state_converges_to_fci():
    mol = Molecule.from_xyz(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))],
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
    )
    state, meta = encode_molecule(mol, chi_layer=16)
    assert meta.n_orb == 2
    assert meta.n_spin_orbitals == 4
    assert meta.n_leaves == 4
    ham = ChemistryHamiltonian(meta)

    # HF product-state energy: should match PySCF's RHF SCF energy to
    # well below 1e-3 Ha — both are evaluating the same determinant via
    # the same MO integrals.
    e_hf_substrate = ham.total_energy(state)
    e_hf_pyscf = meta.integrals.hf_energy
    assert abs(e_hf_substrate - e_hf_pyscf) < 1e-6, (
        f"HF product-state substrate energy {e_hf_substrate} vs PySCF "
        f"RHF reference {e_hf_pyscf}"
    )

    trajectory, final_state = chemistry_imaginary_evolve(
        state, ham, meta, dt=0.05, steps=400, energy_tol=1e-10,
    )
    # Energy decreases monotonically in CI imag-time (within Trotter noise).
    for i in range(1, len(trajectory)):
        assert trajectory[i] <= trajectory[i - 1] + 1e-8, (
            f"non-monotone descent at step {i}: "
            f"{trajectory[i-1]} -> {trajectory[i]}"
        )
    e_final = trajectory[-1]
    err = abs(e_final - H2_FCI_REFERENCE_HARTREE)
    # Chemistry accuracy is 1e-3 Ha (~ 0.6 kcal/mol).
    assert err < 1e-3, (
        f"H2 STO-3G ground state {e_final:.6f} Ha did not converge "
        f"to FCI reference {H2_FCI_REFERENCE_HARTREE:.6f} Ha "
        f"(|delta| = {err:.2e})"
    )
    # Substrate (factored Pauli-string) energy at the final MERA must
    # agree with the CI-basis energy — both compute the same physical
    # expectation value.
    e_final_substrate = ham.total_energy(final_state)
    assert abs(e_final_substrate - e_final) < 1e-3, (
        f"substrate energy {e_final_substrate} disagrees with CI-basis "
        f"energy {e_final} at the same MERA state"
    )
