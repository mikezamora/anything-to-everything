"""Local Fock space and operators at a single site.

Truncated bosonic Fock space H_d = span{|0>, |1>, ..., |d-1>}. Provides
creation a†, annihilation a, number n = a† a, position-like
phi = (a + a†)/sqrt(2), and momentum-like pi = i(a† - a)/sqrt(2).

Multi-species sites have local Hilbert space H = H_a ⊗ H_b ⊗ ... and
operators are built by tensor product with identities for the other
species. The `embed_op` helper handles this.
"""

from __future__ import annotations

import numpy as np


def annihilation(d: int) -> np.ndarray:
    """a |n> = sqrt(n) |n-1>."""
    a = np.zeros((d, d), dtype=complex)
    for n in range(1, d):
        a[n - 1, n] = np.sqrt(n)
    return a


def creation(d: int) -> np.ndarray:
    """a† |n> = sqrt(n+1) |n+1>."""
    return annihilation(d).conj().T


def number(d: int) -> np.ndarray:
    """n = a† a, diagonal in the Fock basis."""
    return np.diag(np.arange(d, dtype=complex))


def phi_op(d: int) -> np.ndarray:
    """phi = (a + a†) / sqrt(2)."""
    a = annihilation(d)
    return (a + a.conj().T) / np.sqrt(2)


def pi_op(d: int) -> np.ndarray:
    """pi = i (a† - a) / sqrt(2)."""
    a = annihilation(d)
    return 1j * (a.conj().T - a) / np.sqrt(2)


def vacuum_vec(d: int) -> np.ndarray:
    v = np.zeros(d, dtype=complex)
    v[0] = 1.0
    return v


def number_state_vec(d: int, n: int) -> np.ndarray:
    if not 0 <= n < d:
        raise ValueError(f"n={n} out of range for cutoff d={d}")
    v = np.zeros(d, dtype=complex)
    v[n] = 1.0
    return v


def identity(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex)


def embed_op(op: np.ndarray, species_index: int,
             species_dims: tuple[int, ...]) -> np.ndarray:
    """Embed a single-species operator into a multi-species local Hilbert space.

    species_dims = (d_a, d_b, ...) gives the per-species cutoffs. The
    result acts as op on `species_index` and as identity on the rest.
    Local-site basis ordering: leftmost species index changes slowest, so
    |n_a, n_b> has linear index n_a * d_b + n_b.
    """
    parts = []
    for i, d in enumerate(species_dims):
        parts.append(op if i == species_index else np.eye(d, dtype=complex))
    result = parts[0]
    for p in parts[1:]:
        result = np.kron(result, p)
    return result


def two_site_op(op_left: np.ndarray, op_right: np.ndarray) -> np.ndarray:
    """Build the two-site operator op_left ⊗ op_right.

    Output basis ordering matches np.kron: site_left's index is the outer
    (slow-changing) one.
    """
    return np.kron(op_left, op_right)
