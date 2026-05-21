"""Hierarchical TEBD evolution for the MERA substrate.

Per spec §5.7 and §6.4, layer-0 carries the dynamics: a second-order
Suzuki-Trotter step applies half-step / full-step / half-step gates to
the layer-0 intra- and inter-pair disentanglers. Higher-layer tensors
are NOT optimized during evolution in sub-project F's encoder-time-only
scope; the deferred upper-layer adaptation introduces a known
approximation error (spec §6.4 line 533).

Mirrors qft/evolution.py's interface (trotter_step, evolve, energy) so
sub-projects A-E can be retargeted from MPS to MERA by changing the
import.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .mera import MERA
from .hamiltonian import Hamiltonian


def _full_bond_op(H: Hamiltonian, k: int) -> np.ndarray:
    """Two-site Hamiltonian on bond (k, k+1) with local terms folded in.

    Identical convention to qft/evolution._full_bond_op so the per-bond
    gates match the MPS substrate.
    """
    d = H.d_local
    h_bond = H.bond_op(k)
    h_left = H.local_op(k)
    h_right = H.local_op(k + 1)
    I = H.local_identity()
    half_left = 0.5 * np.kron(h_left, I)
    half_right = 0.5 * np.kron(I, h_right)
    if k == 0:
        half_left = np.kron(h_left, I)
    if k == H.N - 2:
        half_right = np.kron(I, h_right)
    return h_bond + half_left + half_right


def trotter_step(state: MERA, H: Hamiltonian, dt: float,
                 imaginary: bool = False, chi_max: int = 16,
                 eps: float = 1e-12) -> float:
    """One second-order Suzuki-Trotter step on the MERA substrate.

    Each layer-0 bond gate is absorbed into the corresponding intra-pair
    (even leaf) or inter-pair (odd leaf) disentangler via
    MERA.apply_two_site_gate. Returns the sum of per-gate truncation
    errors.
    """
    factor = -dt if imaginary else -1j * dt
    err_total = 0.0
    half_gates: list[np.ndarray] = []
    full_gates: list[np.ndarray] = []
    for k in range(H.N - 1):
        hb = _full_bond_op(H, k)
        half_gates.append(expm(0.5 * factor * hb))
        full_gates.append(expm(factor * hb))

    # Sweep order: odd bonds (intra-pair, even leaves) half-step, even
    # bonds (inter-pair, odd leaves) full-step, odd bonds half-step again.
    odd_bonds = list(range(0, H.N - 1, 2))
    even_bonds = list(range(1, H.N - 1, 2))

    for k in odd_bonds:
        err_total += state.apply_two_site_gate(k, half_gates[k],
                                               chi_max=chi_max, eps=eps)
    for k in even_bonds:
        err_total += state.apply_two_site_gate(k, full_gates[k],
                                               chi_max=chi_max, eps=eps)
    for k in odd_bonds:
        err_total += state.apply_two_site_gate(k, half_gates[k],
                                               chi_max=chi_max, eps=eps)
    return err_total


def evolve(state: MERA, H: Hamiltonian, dt: float, steps: int,
           imaginary: bool = False, chi_max: int = 16,
           normalize_every: int = 1) -> None:
    """Multi-step evolution. Imaginary-time normalizes every
    `normalize_every` steps.
    """
    for s in range(steps):
        trotter_step(state, H, dt, imaginary=imaginary, chi_max=chi_max)
        if imaginary and (s + 1) % normalize_every == 0:
            state.normalize()


def energy(state: MERA, H: Hamiltonian) -> float:
    """<psi|H|psi> via single-leaf and two-leaf expectations.

    Equivalent to qft/evolution.energy for the MERA substrate.
    """
    e = 0.0 + 0.0j
    for k in range(H.N):
        e += state.local_expectation(k, H.local_op(k))
    for k in range(H.N - 1):
        e += state.two_site_expectation(k, H.bond_op(k))
    return float(np.real(e))
