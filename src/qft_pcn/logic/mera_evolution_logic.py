"""MERA imaginary-time evolution (spec §7.5).

mera_trotter_step applies each Hamiltonian term's factored transition
gate(s) to the MERA state; mera_imaginary_evolve repeats it and returns
the energy trajectory. Reduction is observed by decoding the relaxed
state — no classical rewrite anywhere (spec §1.6).

A product (concrete) MERA's causal-cone isometries are built from its
leaf vectors; a bare apply_local_gate would leave those isometries stale
and orthogonal to the mutated leaf. So a Trotter step extracts the
per-leaf vectors, applies the factored single-leaf transition gates to
them, renormalizes each leaf, and rebuilds the product MERA via
MERA.from_product so the causal-cone isometries stay consistent. The
transition gates are single-leaf (16x16) — well within the
no-dense-operator budget (spec §1.3) — and the tree-truncation cap
chi_layer is preserved across the rebuild.
"""
from __future__ import annotations

import numpy as np

from src.qft_pcn.qft.mera import MERA


def _leaf_vectors(state: MERA) -> list[np.ndarray]:
    return [state.leaves[k][0, :, 0].astype(complex).copy()
            for k in range(state.N)]


def mera_trotter_step(state: MERA, ham, dt: float,
                      imaginary: bool = True,
                      chi_layer: int | None = None) -> MERA:
    """One Trotter step: apply each term's factored transition gates to a
    copy of `state`'s leaf vectors, rebuild a consistent product MERA,
    and return it (the input is not mutated).

    For imaginary time each term's gate is exp(-dt.lambda.H_term) in
    factored form. Single-leaf gates act on the leaf vectors directly;
    the rebuilt MERA's isometries match the new leaves. chi_layer caps
    the rebuilt tree's layer bond dimension.
    """
    if chi_layer is None:
        chi_layer = state.layer_dims[-1] if state.layer_dims else 16
    vecs = _leaf_vectors(state)
    for term in ham.terms:
        gates = ham.term_gates(state, term, dt, imaginary)
        for leaves, gate in gates:
            if len(leaves) == 1:
                vecs[leaves[0]] = gate @ vecs[leaves[0]]
            elif len(leaves) == 2:
                # Two-leaf gate: act on the kron of the two leaf vectors,
                # then re-split by SVD (rank-1 keeps the product form).
                d = vecs[leaves[0]].shape[0]
                joint = np.kron(vecs[leaves[0]], vecs[leaves[1]])
                joint = gate @ joint
                m = joint.reshape(d, d)
                u, s, vh = np.linalg.svd(m)
                vecs[leaves[0]] = u[:, 0] * np.sqrt(s[0])
                vecs[leaves[1]] = vh[0, :] * np.sqrt(s[0])
            else:
                raise ValueError(
                    f"gate spans {len(leaves)} leaves; only 1 or 2 "
                    f"supported by the MERA gate primitives")
    # Renormalize each leaf vector and rebuild a consistent product MERA.
    for k in range(len(vecs)):
        nrm = np.linalg.norm(vecs[k])
        if nrm > 1e-30:
            vecs[k] = vecs[k] / nrm
    out = MERA.from_product(vecs, chi_layer=chi_layer)
    out.normalize()
    return out


def mera_imaginary_evolve(state: MERA, ham, dt: float, steps: int,
                          chi_layer: int | None = None) -> list[float]:
    """Repeat mera_trotter_step `steps` times in imaginary time. Returns
    the energy trajectory [<H>_0, <H>_1, ..., <H>_steps]. Energy decreases
    monotonically (architecture §13.1).

    NOTE: mera_trotter_step returns a fresh MERA each step; this driver
    threads it. Callers that need the final relaxed state should use
    mera_imaginary_evolve_state.
    """
    traj, _ = mera_imaginary_evolve_state(state, ham, dt, steps, chi_layer)
    return traj


def mera_imaginary_evolve_state(state: MERA, ham, dt: float, steps: int,
                                chi_layer: int | None = None):
    """Like mera_imaginary_evolve but also returns the final relaxed
    MERA state. Returns (trajectory, final_state)."""
    cur = state.copy()
    traj = [ham.total_energy(cur)]
    for _ in range(steps):
        cur = mera_trotter_step(cur, ham, dt, imaginary=True,
                                chi_layer=chi_layer)
        traj.append(ham.total_energy(cur))
    return traj, cur
