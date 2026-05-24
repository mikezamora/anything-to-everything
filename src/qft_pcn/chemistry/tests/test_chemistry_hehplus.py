"""HeH+ STO-3G ground-state energy via the §11.6 chemistry encoder.

HeH+ is the second canonical small-molecule benchmark (one more
electron than H2 minus the charge): 2 electrons across 2 spatial
orbitals — same 4-spin-orbital active space but a different nuclear
configuration. The FCI reference is computed at test time via PySCF
on the same MO integrals the substrate uses, eliminating any
table-lookup risk and confirming the substrate <-> ab-initio
correspondence.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.chemistry import (
    Molecule,
    ChemistryHamiltonian,
    build_integrals,
    encode_molecule,
    chemistry_imaginary_evolve,
)


def _fci_reference(mol: Molecule) -> float:
    """Run PySCF FCI on the same molecule for the reference energy."""
    from pyscf import gto, scf, fci
    atom_str = "; ".join(
        f"{a.symbol} {a.xyz[0]} {a.xyz[1]} {a.xyz[2]}" for a in mol.atoms
    )
    py_mol = gto.M(atom=atom_str, basis=mol.basis, charge=mol.charge,
                   spin=mol.spin, unit=mol.unit, verbose=0)
    mf = scf.RHF(py_mol).run(verbose=0)
    cisolver = fci.FCI(mf)
    e_fci, _ = cisolver.kernel()
    return float(e_fci)


@pytest.mark.timeout(600)
def test_hehplus_sto3g_ground_state_converges_to_fci():
    mol = Molecule.from_xyz(
        [("He", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7743))],
        basis="sto-3g",
        charge=1,
        spin=0,
        unit="Angstrom",
    )
    e_fci_ref = _fci_reference(mol)

    state, meta = encode_molecule(mol, chi_layer=16)
    assert meta.n_orb == 2
    assert meta.n_spin_orbitals == 4
    ham = ChemistryHamiltonian(meta)

    e_hf_substrate = ham.total_energy(state)
    e_hf_pyscf = meta.integrals.hf_energy
    assert abs(e_hf_substrate - e_hf_pyscf) < 1e-6

    trajectory, final_state = chemistry_imaginary_evolve(
        state, ham, meta, dt=0.05, steps=400, energy_tol=1e-10,
    )
    e_final = trajectory[-1]
    err = abs(e_final - e_fci_ref)
    assert err < 1e-3, (
        f"HeH+ STO-3G ground state {e_final:.6f} Ha did not converge "
        f"to FCI reference {e_fci_ref:.6f} Ha "
        f"(|delta| = {err:.2e})"
    )
