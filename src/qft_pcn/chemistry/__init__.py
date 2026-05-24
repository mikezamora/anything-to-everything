"""Quantum chemistry encoder + Hamiltonian + ground-state evolution
(spec §11.6, first-tier non-PL target domain).

Each spin-orbital is a MERA leaf (d_local=2: |0> = empty, |1> = occupied).
The molecular electronic Hamiltonian is decomposed into Jordan-Wigner
Pauli strings with real coefficients sourced from PySCF ab-initio
one- and two-electron MO integrals (no dense Fock-space Hamiltonian:
§1.3 — every Hamiltonian term is a factored product of single-leaf
Pauli operators).

Ground-state search uses imaginary-time evolution in a CI-determinant
branch superposition representation:

  - the state is a MERA from_term_superposition over C(n_e, n_so)
    Slater determinants;
  - per Trotter step, branch coefficients evolve by first-order
    imaginary-time power iteration with H matrix elements computed
    one-at-a-time via Slater-Condon rules from MO integrals (factored —
    no dense 4^N Hamiltonian materialized);
  - the MERA is rebuilt from the updated branch amplitudes via
    ``MERA.from_term_superposition``, preserving §1.1 entanglement
    semantics (each determinant is one branch direction).

See:
  - molecule.py — :class:`Molecule` dataclass + AB initio integral builder.
  - encoder.py — :func:`encode_molecule` -> (MERA, ChemEncodingMeta).
  - hamiltonian.py — :class:`ChemistryHamiltonian` (Pauli-string,
    factored-expectation total_energy; CI matrix builder via Slater-Condon).
  - evolve.py — :func:`chemistry_imaginary_evolve` ground-state driver.
"""
from __future__ import annotations

from .molecule import Molecule, MolecularIntegrals, build_integrals
from .encoder import ChemEncodingMeta, encode_molecule
from .hamiltonian import ChemistryHamiltonian, PauliString
from .evolve import chemistry_imaginary_evolve

__all__ = [
    "Molecule",
    "MolecularIntegrals",
    "build_integrals",
    "ChemEncodingMeta",
    "encode_molecule",
    "ChemistryHamiltonian",
    "PauliString",
    "chemistry_imaginary_evolve",
]
