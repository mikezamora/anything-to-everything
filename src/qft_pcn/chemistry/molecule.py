"""Molecule dataclass and ab-initio integral extraction (spec §11.6).

The Molecule object describes the chemical system; build_integrals runs
PySCF's RHF SCF, then transforms the one- and two-electron integrals
to the canonical MO basis. The MO integrals (h1[p,q] and eri[p,q,r,s])
plus the nuclear-repulsion constant are the *factored* inputs from
which :class:`ChemistryHamiltonian` builds Jordan-Wigner Pauli strings;
no dense Fock-space Hamiltonian is ever materialized (§1.3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Atom:
    symbol: str
    xyz: tuple[float, float, float]


@dataclass(frozen=True)
class Molecule:
    """Chemical system description for §11.6 quantum-chemistry runs.

    atoms:    list of (element symbol, (x, y, z)) tuples in Angstroms.
    basis:    Gaussian basis string ('sto-3g', '6-31g', ...).
    charge:   net charge (e units, e.g. 0 for H2, +1 for HeH+).
    spin:     2S (number of unpaired electrons; RHF assumes 0).
    unit:     'Angstrom' or 'Bohr' for atom coordinates (default Angstrom).
    """
    atoms: tuple[Atom, ...]
    basis: str = "sto-3g"
    charge: int = 0
    spin: int = 0
    unit: str = "Angstrom"

    @classmethod
    def from_xyz(cls, lines: Sequence[tuple[str, tuple[float, float, float]]],
                 basis: str = "sto-3g", charge: int = 0, spin: int = 0,
                 unit: str = "Angstrom") -> "Molecule":
        atoms = tuple(Atom(symbol=s, xyz=tuple(map(float, xyz)))
                      for s, xyz in lines)
        return cls(atoms=atoms, basis=basis, charge=charge, spin=spin,
                   unit=unit)

    def n_electrons(self) -> int:
        """Total electron count (sum of nuclear charges minus net charge)."""
        z = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
             "O": 8, "F": 9, "Ne": 10}
        n = sum(z[a.symbol] for a in self.atoms) - self.charge
        return int(n)


@dataclass
class MolecularIntegrals:
    """Mean-field SCF + MO-basis ab-initio integrals (PySCF output).

    n_orb:            number of spatial (MO) orbitals.
    n_electrons:      total electron count of the neutral system minus charge.
    n_alpha, n_beta:  alpha/beta electron counts (RHF: n/2 each for even n).
    h1[p, q]:         one-body MO integrals <p|h|q> (kinetic + nuclear-elec).
    eri[p, q, r, s]:  two-body MO integrals (pq|rs) in chemist's notation.
    nuc_repulsion:    classical nuclear-nuclear Coulomb constant.
    hf_energy:        RHF SCF energy (sanity reference).
    """
    n_orb: int
    n_electrons: int
    n_alpha: int
    n_beta: int
    h1: np.ndarray
    eri: np.ndarray
    nuc_repulsion: float
    hf_energy: float = field(default=0.0)


def build_integrals(mol: Molecule) -> MolecularIntegrals:
    """Run PySCF RHF on the molecule and extract MO-basis integrals.

    The returned :class:`MolecularIntegrals` are *factored* inputs to
    :func:`ChemistryHamiltonian.from_integrals`: every JW Pauli string
    in the chemistry Hamiltonian carries a coefficient that is a linear
    combination of h1[p,q] and eri[p,q,r,s] values. No dense Fock-space
    operator is ever assembled (§1.3 — chemistry Hamiltonian as a
    factored MERA operator).
    """
    try:
        from pyscf import gto, scf, ao2mo
    except ImportError as exc:
        raise RuntimeError(
            "pyscf is required for src.qft_pcn.chemistry; install "
            "via `uv sync` after ensuring pyproject.toml lists pyscf as "
            "a dependency."
        ) from exc

    atom_str = "; ".join(
        f"{a.symbol} {a.xyz[0]} {a.xyz[1]} {a.xyz[2]}" for a in mol.atoms
    )
    py_mol = gto.M(atom=atom_str, basis=mol.basis, charge=mol.charge,
                   spin=mol.spin, unit=mol.unit, verbose=0)
    mf = scf.RHF(py_mol).run(verbose=0)
    n_orb = int(mf.mo_coeff.shape[1])
    # One-body MO integrals: h1_mo = C^T h_AO C
    h_core = mf.get_hcore()
    h1 = mf.mo_coeff.T @ h_core @ mf.mo_coeff
    # Two-body MO integrals in chemist's notation (pq|rs), full 4D form.
    eri = ao2mo.kernel(py_mol, mf.mo_coeff, compact=False)
    eri = eri.reshape(n_orb, n_orb, n_orb, n_orb)
    nuc = float(py_mol.energy_nuc())
    n_e = int(py_mol.nelectron)
    n_alpha = int((n_e + mol.spin) // 2)
    n_beta = n_e - n_alpha
    return MolecularIntegrals(
        n_orb=n_orb,
        n_electrons=n_e,
        n_alpha=n_alpha,
        n_beta=n_beta,
        h1=np.asarray(h1, dtype=float),
        eri=np.asarray(eri, dtype=float),
        nuc_repulsion=nuc,
        hf_energy=float(mf.e_tot),
    )
