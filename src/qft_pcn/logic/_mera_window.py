"""k-adjacent-leaf expectation on a MERA (spec §8.2).

mera_window_expectation is the dense form, used only for k<=2 to
cross-check F's two_site_expectation. M2's Hamiltonian terms call the
factored form (Task 9) — never the dense form for k>2, since a dense
(16**k, 16**k) operator is intractable for large k.
"""
from __future__ import annotations

import numpy as np

from src.qft_pcn.qft.mera import MERA


def mera_window_expectation(state: MERA, leaf0: int, k: int,
                            op: np.ndarray) -> complex:
    """<state | O | state> for O on the k adjacent leaves [leaf0, leaf0+k).

    op has shape (d**k, d**k) with d = state.d_local. For k==1 this
    delegates to MERA.local_expectation; for k==2 to
    MERA.two_site_expectation. k>2 dense is rejected — use the factored
    form (mera_window_expectation_factored).
    """
    d = state.d_local
    if k == 1:
        if op.shape != (d, d):
            raise ValueError(f"k=1 op shape {op.shape}, expected ({d},{d})")
        return state.local_expectation(leaf0, op)
    if k == 2:
        if op.shape != (d * d, d * d):
            raise ValueError(
                f"k=2 op shape {op.shape}, expected ({d*d},{d*d})")
        return state.two_site_expectation(leaf0, op)
    raise ValueError(
        f"dense mera_window_expectation supports k in {{1,2}}; for k={k} "
        f"use mera_window_expectation_factored")


def mera_window_expectation_factored(
    state: MERA, leaf_ops: dict[int, np.ndarray]) -> complex:
    """<state | O | state> where O = prod over listed leaves of a per-leaf
    (d, d) operator (unlisted leaves: identity).

    No (d**k, d**k) tensor is formed. Each per-leaf operator is applied to
    the state by MERA.apply_local_gate on a copy, then the inner product
    with the original gives the expectation — because the listed operators
    act on distinct leaves and therefore commute, the product operator's
    expectation equals < state | (prod gates) | state >.

    Apply each gate to the ket copy, then return <state | ket_copy>.

    Product-MERA fast path: when `state` is a plain product MERA (identity
    disentanglers, no term superposition) the double-network contraction
    telescopes exactly — every isometry W satisfies W^dag W = I and the
    unlisted leaves' overlaps reduce to <v_k|v_k>. So the expectation is
    just prod_k <v_k|O_k|v_k> over the listed leaves times the product
    norm of the rest. This is the SAME identity `MERA.inner` already uses
    for its product fast path; computing it directly here avoids the full
    isometry-tree `state.copy()` and the `_same_tree` tensor comparison
    that dominate imaginary-time evolution (spec §8.2 — physics unchanged,
    addressing only). The general-tree contraction below is the fallback.
    """
    d = state.d_local
    # Drop identity operators.
    active = {leaf: op for leaf, op in leaf_ops.items()
              if not _is_identity(op, d)}
    if not active:
        return complex(state.norm_sq())
    for leaf, op in active.items():
        if op.shape != (d, d):
            raise ValueError(
                f"leaf {leaf} op shape {op.shape}, expected ({d},{d})")
    # Product-MERA fast path: prod_k <v_k|O_k|v_k> on the listed leaves
    # times <v_j|v_j> on the rest (telescoping identity, see docstring).
    if state._superposition_terms is None and state._is_product():
        val = complex(1.0)
        for k in range(state.N):
            v = state.leaves[k][0, :, 0]
            op = active.get(k)
            if op is None:
                val *= complex(v.conj() @ v)
            else:
                val *= complex(v.conj() @ (op @ v))
        return val
    ket = state.copy()
    for leaf, op in active.items():
        ket.apply_local_gate(leaf, op)
    return state.inner(ket)


def _is_identity(op: np.ndarray, d: int) -> bool:
    if op.shape != (d, d):
        return False
    return np.allclose(op, np.eye(d, dtype=op.dtype), atol=1e-12)
