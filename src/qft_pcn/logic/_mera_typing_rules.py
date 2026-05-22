"""Shared factored-operator helpers for the MERA Hamiltonians (spec §5.2).

Every Hamiltonian term is a dict[leaf_index -> (16,16) matrix], identity
on unlisted leaves, evaluated by mera_window_expectation_factored. These
helpers build the 16-dim per-leaf projectors and address leaves through
the M1 layout. No 16**k dense operator is ever formed (spec §1.3).
"""
from __future__ import annotations

import numpy as np

from .mera_encoding import MERA_LEAF_DIM


def leaf_proj(idx: int) -> np.ndarray:
    """|idx><idx| on a 16-dim leaf."""
    if not 0 <= idx < MERA_LEAF_DIM:
        raise ValueError(f"leaf basis index {idx} out of [0, {MERA_LEAF_DIM})")
    p = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
    p[idx, idx] = 1.0
    return p


def leaf_proj_one_minus(idx: int) -> np.ndarray:
    """I - |idx><idx|."""
    return np.eye(MERA_LEAF_DIM, dtype=complex) - leaf_proj(idx)


def leaf_proj_set(idxs) -> np.ndarray:
    """Sum of |i><i| for i in idxs."""
    out = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
    for i in idxs:
        if not 0 <= i < MERA_LEAF_DIM:
            raise ValueError(f"leaf basis index {i} out of range")
        out[i, i] = 1.0
    return out


def leaf_identity() -> np.ndarray:
    return np.eye(MERA_LEAF_DIM, dtype=complex)


def node_leaf(meta, node: int, species: str) -> int:
    """Absolute leaf index of `species` of AST `node`, via the M1 layout."""
    return meta.layout.leaf_of(node, species)


def mutate_leaf(state, leaf: int, old_idx: int, new_idx: int):
    """Test helper: swap basis slices `old_idx` and `new_idx` of one leaf
    of a product (concrete) MERA state, returning a mutated copy. Used by
    the per-rule isolation tests to surgically inject a typing violation.

    A product MERA's isometries are built from its leaf vectors; a bare
    apply_local_gate would leave those isometries stale and orthogonal to
    the mutated leaf. So this helper extracts the per-leaf vectors,
    applies the basis swap to the target leaf, and rebuilds the product
    MERA via MERA.from_product so the causal-cone isometries match.
    """
    from src.qft_pcn.qft.mera import MERA
    leaf_vecs = [state.leaves[k][0, :, 0].astype(complex).copy()
                 for k in range(state.N)]
    g = np.eye(MERA_LEAF_DIM, dtype=complex)
    g[old_idx, old_idx] = 0.0
    g[new_idx, new_idx] = 0.0
    g[new_idx, old_idx] = 1.0
    g[old_idx, new_idx] = 1.0
    leaf_vecs[leaf] = g @ leaf_vecs[leaf]
    out = MERA.from_product(leaf_vecs, chi_layer=state.layer_dims[-1])
    out.normalize()
    return out
