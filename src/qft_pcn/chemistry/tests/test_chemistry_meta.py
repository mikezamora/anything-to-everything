"""Encoder-meta structural invariants for the §11.6 chemistry layout.

These checks pin down the encoder/meta contract:
  - leaf count is the next power of two >= 2 * n_orb,
  - every spatial orbital maps to exactly two spin-orbital leaves
    (alpha at 2*p, beta at 2*p + 1),
  - the HF occupation list matches PySCF's n_alpha / n_beta,
  - ghost leaves are flagged (when n_so is not already a power of two).

No imaginary-time evolution is run here; this test exists so that the
"encoder produces a valid meta with the right shape" claim of §11.6 is
backed by a fast unit test independent of the (slower) ground-state
acceptance tests.
"""
from __future__ import annotations

import pytest

from src.qft_pcn.chemistry import Molecule, encode_molecule


@pytest.mark.timeout(60)
def test_h2_meta_shape_matches_orbital_count():
    mol = Molecule.from_xyz(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))],
        basis="sto-3g",
    )
    state, meta = encode_molecule(mol, chi_layer=16)
    # H2 STO-3G has 2 spatial orbitals => 4 spin-orbital leaves.
    assert meta.n_orb == 2
    assert meta.n_spin_orbitals == 4
    assert meta.n_leaves == 4
    assert meta.L == 2
    assert meta.leaf_dim == 2
    assert all(not g for g in meta.is_ghost), (
        "H2 STO-3G should not need ghost padding (4 is a power of two)"
    )
    # HF occupation: 1 alpha + 1 beta electron in the lowest spatial orbital.
    occ = meta.hf_occupation()
    assert sum(occ) == 2
    assert occ[meta.site_of_spin_orbital[0]] == 1   # alpha-0 occupied
    assert occ[meta.site_of_spin_orbital[1]] == 1   # beta-0 occupied
    assert occ[meta.site_of_spin_orbital[2]] == 0   # alpha-1 empty (virtual)
    assert occ[meta.site_of_spin_orbital[3]] == 0   # beta-1 empty (virtual)


@pytest.mark.timeout(60)
def test_hehplus_meta_shape():
    mol = Molecule.from_xyz(
        [("He", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7743))],
        basis="sto-3g",
        charge=1,
    )
    state, meta = encode_molecule(mol)
    assert meta.n_orb == 2
    assert meta.n_spin_orbitals == 4
    assert meta.n_leaves == 4
    assert meta.integrals.n_electrons == 2
    assert meta.integrals.n_alpha == 1
    assert meta.integrals.n_beta == 1
    occ = meta.hf_occupation()
    assert sum(occ) == 2
