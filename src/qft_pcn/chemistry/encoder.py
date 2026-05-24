"""Encode a molecule's electronic structure as a MERA state (spec §11.6).

Convention (spin-orbital ordering):
  - Each spatial orbital p contributes TWO spin-orbitals at site indices
    2*p (alpha) and 2*p + 1 (beta). The full ordering is
    [alpha_0, beta_0, alpha_1, beta_1, ..., alpha_{n_orb-1}, beta_{n_orb-1}].
  - Local Hilbert space per site: d_local = 2, |0> = unoccupied,
    |1> = occupied (qubit encoding for the Jordan-Wigner mapping in
    :mod:`hamiltonian`).
  - The MERA layout requires N a power of 2; ghost spin-orbitals (always
    empty) pad up to the next power of two and contribute 0 to every
    Pauli string in :class:`ChemistryHamiltonian` (their h1/eri rows are
    zero by construction).

The initial state is the Restricted Hartree-Fock single determinant
(lowest n_alpha + n_beta spin-orbitals occupied), encoded as a product
MERA via :meth:`MERA.from_product`. Imaginary-time evolution
(:func:`chemistry_imaginary_evolve`) then promotes this product state
into a genuine MERA superposition over the CI determinant basis,
recovering correlation energy.

Returns ``(MERA, ChemEncodingMeta)`` analogous to
``encode_mera(ast)`` in :mod:`qft_pcn.logic.mera_encoder` — non-PL
encoders for §11.6 follow the same shape so downstream tooling
(decoders, viz panels) can be domain-polymorphic in the future.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.qft_pcn.qft.mera import MERA

from .molecule import Molecule, MolecularIntegrals, build_integrals

# Local Hilbert space per spin-orbital: |0> (empty) or |1> (occupied).
CHEM_LEAF_DIM = 2

# Spin-orbital -> leaf layout strategies (§11.6 + EXTENSIONS).
#  "interleaved":     alpha_0, beta_0, alpha_1, beta_1, ... (default; A3).
#  "alpha_then_beta": alpha_0..alpha_{n-1}, beta_0..beta_{n-1} (block ordering
#                    used by some DMRG/MERA chemistry codes, and required for
#                    CASCI/CASSCF non-contiguous active-space workflows).
OrbitalLayout = Literal["interleaved", "alpha_then_beta"]


@dataclass
class ChemEncodingMeta:
    """Layout + bookkeeping for a molecular MERA encoding (spec §11.6).

    n_spin_orbitals:   physical spin-orbital count (= 2 * n_orb).
    n_leaves:          MERA leaf count (padded power of two >= n_spin_orbitals).
    leaf_dim:          per-leaf local Hilbert dimension (= CHEM_LEAF_DIM = 2).
    L:                 MERA layer count (log2(n_leaves)).
    site_of_spin_orbital:  list mapping (spatial orbital p, spin s) -> leaf idx.
                       Specifically, spin-orbital index 2p (alpha) or
                       2p+1 (beta) is stored at MERA leaf
                       ``site_of_spin_orbital[2p + s]``.
    is_ghost:          per-leaf flag; True iff the leaf is a power-of-two
                       padding site (no spatial orbital backs it).
    integrals:         the source :class:`MolecularIntegrals` (h1, eri,
                       nuc_repulsion, hf_energy, n_alpha/beta/electrons).
    """
    n_spin_orbitals: int
    n_leaves: int
    leaf_dim: int
    L: int
    site_of_spin_orbital: list[int]
    is_ghost: list[bool]
    integrals: MolecularIntegrals
    chi_layer: int = 16
    orbital_layout: OrbitalLayout = "interleaved"

    @property
    def n_orb(self) -> int:
        return self.integrals.n_orb

    def hf_occupation(self) -> list[int]:
        """Return the RHF occupation list, length n_leaves, entries 0/1.

        The lowest ``n_alpha`` alpha and ``n_beta`` beta spin-orbitals are
        occupied; remaining (and ghost) leaves are empty. This is the
        single-determinant starting point for imaginary-time evolution.
        """
        occ = [0] * self.n_leaves
        n_orb = self.n_orb
        for p in range(n_orb):
            alpha_leaf = self.site_of_spin_orbital[2 * p]
            beta_leaf = self.site_of_spin_orbital[2 * p + 1]
            if p < self.integrals.n_alpha:
                occ[alpha_leaf] = 1
            if p < self.integrals.n_beta:
                occ[beta_leaf] = 1
        return occ


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def encode_molecule(mol: Molecule | None = None,
                    integrals: MolecularIntegrals | None = None,
                    chi_layer: int = 16,
                    orbital_layout: OrbitalLayout = "interleaved",
                    ) -> tuple[MERA, ChemEncodingMeta]:
    """Encode a molecule as a product MERA in the Hartree-Fock state.

    Either ``mol`` (which triggers a PySCF RHF SCF run via
    :func:`build_integrals`) or ``integrals`` (pre-computed) must be
    supplied. The returned state is the RHF single determinant on the
    interleaved alpha/beta spin-orbital layout, ready for
    :func:`chemistry_imaginary_evolve` to promote into a correlated
    ground state.

    Raises ValueError if both arguments are None.
    """
    if integrals is None:
        if mol is None:
            raise ValueError("encode_molecule requires `mol` or `integrals`")
        integrals = build_integrals(mol)
    n_orb = integrals.n_orb
    n_spin = 2 * n_orb
    n_leaves = _next_pow2(max(n_spin, 2))
    if orbital_layout == "interleaved":
        # Spin-orbital index 2*p + s -> leaf (2*p + s); contiguous block
        # of length n_spin then ghost padding.
        site_of_spin_orbital = list(range(n_spin))
    elif orbital_layout == "alpha_then_beta":
        # Alpha block first (s=0): leaves [0..n_orb).
        # Beta block second (s=1): leaves [n_orb..2*n_orb).
        # Spin-orbital index 2*p + 0 -> leaf p; 2*p + 1 -> leaf (n_orb + p).
        site_of_spin_orbital = [0] * n_spin
        for p in range(n_orb):
            site_of_spin_orbital[2 * p] = p
            site_of_spin_orbital[2 * p + 1] = n_orb + p
    else:
        raise ValueError(
            f"orbital_layout must be 'interleaved' or 'alpha_then_beta', "
            f"got {orbital_layout!r}"
        )
    # Ghost flag is per leaf: any leaf NOT used by some spin-orbital is a ghost.
    used_leaves = set(site_of_spin_orbital)
    is_ghost = [leaf not in used_leaves for leaf in range(n_leaves)]
    L = int(round(np.log2(n_leaves)))

    meta = ChemEncodingMeta(
        n_spin_orbitals=n_spin,
        n_leaves=n_leaves,
        leaf_dim=CHEM_LEAF_DIM,
        L=L,
        site_of_spin_orbital=site_of_spin_orbital,
        is_ghost=is_ghost,
        integrals=integrals,
        chi_layer=chi_layer,
        orbital_layout=orbital_layout,
    )
    # Build the HF product MERA: per-leaf state |0> or |1>.
    occ = meta.hf_occupation()
    leaf_vecs: list[np.ndarray] = []
    for n in occ:
        v = np.zeros(CHEM_LEAF_DIM, dtype=complex)
        v[n] = 1.0
        leaf_vecs.append(v)
    state = MERA.from_product(leaf_vecs, chi_layer=chi_layer)
    state.normalize()
    return state, meta
