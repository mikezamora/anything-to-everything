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
                      chi_layer: int | None = None,
                      cache: dict | None = None) -> MERA:
    """One Trotter step: apply each term's factored transition gates to a
    copy of `state`'s leaf vectors, rebuild a consistent product MERA,
    and return it (the input is not mutated).

    For imaginary time each term's gate is exp(-dt.lambda.H_term) in
    factored form. Single-leaf gates act on the leaf vectors directly;
    the rebuilt MERA's isometries match the new leaves. chi_layer caps
    the rebuilt tree's layer bond dimension.

    Optional ``cache`` is a mutable dict threaded by the evolution driver
    across Trotter steps within one anneal phase. It carries:
      * ``"inactive"``: set[id(term)] of terms whose previous call to
        ``term_gates`` returned [] (no gates emitted);
      * ``"changed_leaves"``: frozenset[int] of leaf indices the previous
        step's gates wrote to.
    A term in ``inactive`` whose ``term.term_affected_leaves`` is disjoint
    from ``changed_leaves`` cannot have become active (its read/write
    footprint did not move) and is skipped — eliminating the per-step
    argmax + guard scan on the ~352-of-360 P3 terms that stay inactive.
    A miss (the cache lacks the necessary key, or the Hamiltonian has no
    ``term_affected_leaves`` accessor) falls back to the live re-check;
    correctness is unchanged, only wall-clock varies.
    """
    if chi_layer is None:
        chi_layer = state.layer_dims[-1] if state.layer_dims else 16
    vecs = _leaf_vectors(state)
    if cache is None:
        cache = {}
    prev_inactive: set = cache.get("inactive", set())
    prev_changed: frozenset = cache.get("changed_leaves", frozenset())
    affected_fn = getattr(ham, "term_affected_leaves", None)
    new_inactive: set = set()
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
        # Redex-presence cache: a term that emitted no gates last step is
        # skipped IFF none of its read/write footprint leaves changed.
        if affected_fn is not None and id(term) in prev_inactive:
            footprint = affected_fn(term)
            if footprint.isdisjoint(prev_changed):
                new_inactive.add(id(term))
                continue
        gates = ham.term_gates(state, term, dt, imaginary)
        if not gates and affected_fn is not None:
            new_inactive.add(id(term))
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
    # Record what changed this step so the next step's redex-presence
    # filter can skip terms whose footprint did not move. Single-leaf
    # gates touch one leaf each; two-leaf gates touch their ordered pair.
    changed: set[int] = set()
    for k in single_order:
        changed.add(k)
    for (l0, l1) in pair_order:
        changed.add(l0)
        changed.add(l1)
    cache["inactive"] = new_inactive
    cache["changed_leaves"] = frozenset(changed)
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
    # Threaded redex-presence cache (see mera_trotter_step docstring). The
    # cache survives across steps within this one phase; phase boundaries
    # (warmup -> main -> fine) use a fresh driver call and a fresh cache,
    # which is correct because the Hamiltonian itself differs.
    cache: dict = {}
    for _ in range(steps):
        cur = mera_trotter_step(cur, ham, dt, imaginary=True,
                                chi_layer=chi_layer, cache=cache)
        traj.append(ham.total_energy(cur))
    return traj, cur
