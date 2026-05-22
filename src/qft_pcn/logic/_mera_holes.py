"""Hole-bearing MERA encoding: genuine tree entanglement (spec §5.3).

A ``HoleVar(candidates=[...])`` use site's ``bid`` leaf carries an
equal-amplitude superposition over the candidate binders' ``bid`` indices,
entangled *through the MERA tree* with those binders. This is the §1.1
soul on the tree: variable binding is genuine entanglement, never a
classical lookup. A hole-bearing program does NOT encode to a product
state.

Construction (spec §5.3, the "analytic" branch realized exactly).
A hole with ``k`` candidates defines a rank-``k`` superposition

    psi = (1/sqrt k) sum_j  | branch_j >

where ``branch_j`` is the concrete encoding of the program *committed to
candidate j*: the hole's ``bid`` leaf holds candidate j's depth-relative
bid index, AND candidate j's binder node carries a witness mark (a
distinct nonzero index on the binder's otherwise-unused ``value`` leaf)
recording "this binder is the one the hole resolved to". Branches j and
j' therefore differ on three leaves — the hole's ``bid`` leaf and the two
binders' witness leaves — so the state is *genuinely entangled*: across
any cut separating the hole's ``bid`` leaf from a candidate binder's
leaves the entropy is strictly positive (the §9.4 structural marker).

Why a witness mark and not the bare bid superposition. If only the hole's
``bid`` leaf differed across branches the state would be a product state
with one superposed leaf (zero tree entanglement) — the classical-lookup
shortcut §1.1 forbids. The witness mark correlates the hole's choice with
the *identity of the binder*, exactly the correlation the architecture
demands; it is the analytic analog of the plan's CNOT-like gate (which
"increments the target leaf to mark the correlation"), placed on an
unused binder leaf so it never corrupts the binder's kind/type/bid.

The rank-``k`` state is handed to ``MERA.from_term_superposition``, which
builds the tree exactly: identity disentanglers, isometries whose bond
dimension grows to span the ``k`` branch directions (the entanglement is
isometry-carried), and a rank-``k`` top tensor. No large dense tensor is
formed — every per-leaf vector is 16-dimensional and only ``k`` branches
are tracked.
"""
from __future__ import annotations

import numpy as np

from .mera_encoding import MERA_LEAF_DIM, SPECIES_LEAF_OFFSET
from src.qft_pcn.qft.mera import MERA


# Witness marks live on a candidate binder's `value` leaf. Binder nodes
# (Lam/Forall/Fix) carry no value, so their `value` leaf is index 0 in
# every concrete branch; a distinct high index per candidate rank is a
# faithful, collision-free "this binder was chosen" witness.
_WITNESS_BASE = MERA_LEAF_DIM - 1   # 15: top of the leaf basis, unused by
#                                     any kind/type/bid/value/tobl tag.


def witness_index(candidate_rank: int) -> int:
    """The witness basis index marking candidate ``candidate_rank``'s
    binder in the branch where the hole resolves to it. Distinct per rank,
    high in the leaf basis so it never collides with a real species tag.
    """
    idx = _WITNESS_BASE - candidate_rank
    if idx <= 0:
        raise ValueError(
            f"too many candidates ({candidate_rank + 1}) for the leaf "
            f"basis (dim {MERA_LEAF_DIM}); witness slots exhausted")
    return idx


def superposed_bid_vector(candidate_bid_values: list[int]) -> np.ndarray:
    """Equal-amplitude superposition ``(1/sqrt k) sum_j |v_j>`` over the
    candidate binder bid indices — the hole's ``bid`` leaf marginal
    direction (spec §5.3). Used for the leaf vector of the hole node so a
    per-leaf measurement sees the superposition.
    """
    if not candidate_bid_values:
        raise ValueError("a hole must have at least one candidate")
    v = np.zeros(MERA_LEAF_DIM, dtype=complex)
    amp = 1.0 / np.sqrt(len(candidate_bid_values))
    for bid in candidate_bid_values:
        if not 0 <= bid < MERA_LEAF_DIM:
            raise ValueError(
                f"candidate bid {bid} out of leaf range "
                f"[0, {MERA_LEAF_DIM})")
        v[bid] += amp
    return v


def _one_hot(index: int) -> np.ndarray:
    v = np.zeros(MERA_LEAF_DIM, dtype=complex)
    v[index] = 1.0
    return v


def build_hole_terms(
    base_leaf_vectors: list[np.ndarray],
    holes: list[dict],
) -> list[tuple[complex, list[np.ndarray]]]:
    """Build the rank-``k`` branch decomposition for a hole-bearing program.

    ``base_leaf_vectors`` is the concrete (hole-free) leaf-vector list — the
    one ``from_product`` would consume — with every hole's leaves already in
    their *first-candidate* concrete form (the per-branch overrides below
    replace them).

    ``holes`` is one dict per hole, each with:
        ``hole_bid_leaf``        — leaf index of the hole's ``bid`` species
        ``cand_bid_values``      — depth-relative bid index per candidate
        ``cand_witness_leaves``  — the ``value``-leaf index of each
                                   candidate binder node

    Returns the list of ``(coeff, per-leaf vectors)`` terms. With a single
    hole of ``k`` candidates there are ``k`` equal-amplitude branches; with
    multiple holes the branches are the Cartesian product (each hole is
    resolved independently, the architecture's structural superposition).

    Each branch starts from ``base_leaf_vectors`` and, per hole, overrides
    the hole's ``bid`` leaf with that branch's candidate bid index and the
    chosen binder's witness leaf with ``witness_index(rank)``.
    """
    # Cartesian product of per-hole candidate choices.
    branches: list[list[int]] = [[]]
    for hole in holes:
        k = len(hole["cand_bid_values"])
        branches = [b + [r] for b in branches for r in range(k)]
    n_branches = len(branches)
    amp = 1.0 / np.sqrt(n_branches)
    terms: list[tuple[complex, list[np.ndarray]]] = []
    for choice in branches:
        leaves = [v.copy() for v in base_leaf_vectors]
        for hole, rank in zip(holes, choice):
            bid_leaf = hole["hole_bid_leaf"]
            leaves[bid_leaf] = _one_hot(hole["cand_bid_values"][rank])
            witness_leaf = hole["cand_witness_leaves"][rank]
            leaves[witness_leaf] = _one_hot(witness_index(rank))
        terms.append((complex(amp), leaves))
    return terms


def encode_hole_state(
    base_leaf_vectors: list[np.ndarray],
    holes: list[dict],
    chi_layer: int = 16,
) -> MERA:
    """Encode a hole-bearing program into a genuinely tree-entangled MERA.

    Builds the rank-``k`` branch decomposition (``build_hole_terms``) and
    hands it to ``MERA.from_term_superposition``. The returned state is
    unit-norm after ``normalize`` and is NOT a product state — the §1.1
    binding-as-entanglement principle realized on the MERA tree.
    """
    terms = build_hole_terms(base_leaf_vectors, holes)
    state = MERA.from_term_superposition(terms, chi_layer=chi_layer)
    return state
