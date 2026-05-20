"""Time evolution of an MPS under a local Hamiltonian.

We support two evolution modes:

  - Real time (Lorentzian, unitary): |Psi(t+dt)> = exp(-i H dt) |Psi(t)>.
    This is the Schrodinger/QFT dynamics: oscillations, propagating
    excitations, light-cone-like spreading of correlations.

  - Imaginary time (Euclidean): |Psi(tau+dtau)> = exp(-H dtau) |Psi(tau)>
    followed by renormalization. Drives the state to the ground state of H
    (lowest energy). Used as variational relaxation when fitting the
    Hamiltonian to observations.

Both are built from second-order Suzuki-Trotter splitting:
  exp(-i H dt) ~ exp(-i H_odd dt/2) exp(-i H_even dt) exp(-i H_odd dt/2)
  where H_odd  = sum_{k odd} (H_bond(k) + 1/2 H_local(k) + 1/2 H_local(k+1))
        H_even = sum_{k even} (H_bond(k) + 1/2 H_local(k) + 1/2 H_local(k+1))

For a chain of N sites with bond decomposition, this is the standard TEBD
algorithm. Local terms are distributed across the bond gates they touch.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .mps import MPS
from .hamiltonian import Hamiltonian


def _full_bond_op(H: Hamiltonian, k: int) -> np.ndarray:
    """Two-site Hamiltonian acting on bond (k, k+1), with local terms folded in.

    Each local term H_local(k) gets split between the bonds (k-1, k) and
    (k, k+1) so the full Trotter step recovers the right Hamiltonian.
    """
    d = H.d_local
    h_bond = H.bond_op(k)
    h_left = H.local_op(k)
    h_right = H.local_op(k + 1)
    I = H.local_identity()
    # Half-weight contributions from the two adjacent sites.
    half_left = 0.5 * np.kron(h_left, I)
    half_right = 0.5 * np.kron(I, h_right)
    # Sites at the ends of the chain only have one bond, so they should be
    # weighted fully there. The caller compensates for the endpoints.
    if k == 0:
        half_left = np.kron(h_left, I)
    if k == H.N - 2:
        half_right = np.kron(I, h_right)
    return h_bond + half_left + half_right


def trotter_step(state: MPS, H: Hamiltonian, dt: float,
                 imaginary: bool = False, chi_max: int = 32,
                 eps: float = 1e-10) -> float:
    """One second-order Trotter step. Returns total truncation error.

    For imaginary time evolution, the caller is responsible for
    renormalizing the state after a (sequence of) step(s).
    """
    factor = -dt if imaginary else -1j * dt
    err_total = 0.0
    d = H.d_local

    # Build per-bond half-step and full-step gates once.
    half_gates: list[np.ndarray] = []
    full_gates: list[np.ndarray] = []
    for k in range(H.N - 1):
        hb = _full_bond_op(H, k)
        half_gates.append(expm(0.5 * factor * hb))
        full_gates.append(expm(factor * hb))

    # Sweep order: odd bonds first (half step), even bonds (full step),
    # odd bonds again (half step). Second-order Suzuki-Trotter.
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


def evolve(state: MPS, H: Hamiltonian, dt: float, steps: int,
           imaginary: bool = False, chi_max: int = 32,
           normalize_every: int = 1) -> None:
    """Multi-step evolution. Renormalizes every `normalize_every` steps in
    imaginary time mode."""
    for s in range(steps):
        trotter_step(state, H, dt, imaginary=imaginary, chi_max=chi_max)
        if imaginary and (s + 1) % normalize_every == 0:
            state.normalize()


def energy(state: MPS, H: Hamiltonian) -> float:
    """<Psi|H|Psi>. Real because H is Hermitian; the .real here is a safe cast."""
    e = 0.0 + 0.0j
    for k in range(H.N):
        e += state.local_expectation(k, H.local_op(k))
    for k in range(H.N - 1):
        e += state.two_site_expectation(k, H.bond_op(k))
    return float(np.real(e))
