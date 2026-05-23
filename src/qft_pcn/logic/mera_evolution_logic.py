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
    # Collect all per-term gates BEFORE mutating leaf vectors, then group
    # by target leaf-tuple. Across a Trotter step many terms target the
    # SAME single leaf (e.g. the diagonal damping and the reduction
    # transition for the kind/value leaves of the same node). Sequentially
    # applying N single-leaf factors as `G_N @ ... @ G_1 @ v` is identical
    # to applying the precomputed matrix product `(G_N @ ... @ G_1)` once
    # — matrix multiplication is associative and there is no operator at
    # the same index between two same-leaf factors. For two-leaf gates the
    # same reasoning applies on a fixed ordered leaf-pair. Gates targeting
    # disjoint leaves commute trivially. Grouping collapses the work and
    # (importantly) keeps the MERA mutation-version stable across the step
    # so any downstream bra-descend cache amortizes within the step.
    single_combined: dict[int, np.ndarray] = {}
    single_order: list[int] = []
    pair_combined: dict[tuple[int, int], np.ndarray] = {}
    pair_order: list[tuple[int, int]] = []
    for term in ham.terms:
        gates = ham.term_gates(state, term, dt, imaginary)
        for leaves, gate in gates:
            if len(leaves) == 1:
                k = leaves[0]
                prev = single_combined.get(k)
                if prev is None:
                    single_combined[k] = gate
                    single_order.append(k)
                else:
                    # Original loop applies the LATER gate after the earlier
                    # one: v <- gate_later @ (gate_earlier @ v). The
                    # equivalent composite is gate_later @ gate_earlier.
                    single_combined[k] = gate @ prev
            elif len(leaves) == 2:
                key = (leaves[0], leaves[1])
                prev = pair_combined.get(key)
                if prev is None:
                    pair_combined[key] = gate
                    pair_order.append(key)
                else:
                    pair_combined[key] = gate @ prev
            else:
                raise ValueError(
                    f"gate spans {len(leaves)} leaves; only 1 or 2 "
                    f"supported by the MERA gate primitives")
    # Apply grouped single-leaf gates: one matmul per touched leaf.
    for k in single_order:
        vecs[k] = single_combined[k] @ vecs[k]
    # Apply grouped two-leaf gates: one kron+SVD per touched ordered pair.
    for key in pair_order:
        gate = pair_combined[key]
        l0, l1 = key
        d = vecs[l0].shape[0]
        joint = np.kron(vecs[l0], vecs[l1])
        joint = gate @ joint
        m = joint.reshape(d, d)
        u, s, vh = np.linalg.svd(m)
        vecs[l0] = u[:, 0] * np.sqrt(s[0])
        vecs[l1] = vh[0, :] * np.sqrt(s[0])
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
