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

from .mera_encoding import MERA_LEAF_DIM, SPECIES_LEAF_OFFSET, LEAVES_PER_NODE
from .encoding import KIND_PAD as _ENC_KIND_PAD, BID_0
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


# ---- structural-hole superposition (spec §5.3, the M3 rank-k state) ------


def _pad_leaf_vec() -> np.ndarray:
    """Shape-pad leaf vector: kind = KIND_PAD, others zero-padded.

    Used to fill the trailing slots of a hole region for branches whose
    candidate sub-tree has fewer than n_max nodes. This is what makes
    branches differ on the `kind` leaves — KIND_PAD vs a real kind —
    so the superposition is genuinely shape-varying (spec §5.3).
    """
    v = np.zeros(MERA_LEAF_DIM, dtype=complex)
    v[_ENC_KIND_PAD] = 1.0
    return v


def structural_hole_branches(region, sketch_scope, layout):
    """Build the k per-branch leaf-vector lists for one structural hole.

    Each branch j is the concrete leaf assignment of the hole region
    committed to candidate j (spec §5.3):

      1. Candidate j is serialized pre-order to n_j nodes; node m writes
         its five 16-dim one-hot leaf vectors into region node slot m
         (reuses ``node_leaf_vectors``).
      2. Region node slots ``[n_j, n_max)`` are filled with PAD-leaf
         vectors so branches differ on the ``kind`` leaves: ``KIND_PAD``
         vs a real kind. This shape-pad mechanism is what makes branches
         structurally distinct.
      3. Each ``Var`` inside candidate j has its ``bid`` leaf set to the
         depth-relative bid index of the binder it references in
         ``sketch_scope`` (the lexical binders in scope at the hole
         position).
      4. Branch j writes a distinct witness index on the region root
         node's ``value`` leaf so the k branch directions are mutually
         orthogonal — the analytic analog of M1's CNOT-like witness
         mark. Without this, branches with shape-pad-only differences
         could collapse to a product state.

    Parameters
    ----------
    region : HoleRegion
        ``node_start`` is the expanded-layout pre-order index of the
        region's root; ``n_max`` is the max candidate node count;
        ``candidate_branches`` is the list of candidate sub-tree ASTs.
    sketch_scope : list
        The list of in-scope binders at the hole position, innermost
        last. Used to resolve each candidate ``Var`` name to its
        depth-relative bid index.
    layout : MeraLayout
        The expanded layout (already sized for the region's n_max slots).

    Returns
    -------
    list[dict[int, np.ndarray]]
        One dict per branch ``j``; each maps absolute leaf index ->
        leaf-vector overrides for the 5*n_max region leaves. The caller
        merges these with the concrete-node leaf vectors to form the
        per-branch full leaf list passed to ``MERA.from_term_superposition``.
    """
    # Import inside the function to avoid a circular import with
    # _mera_leaves (which itself imports from this module's neighbors).
    from ._mera_leaves import node_leaf_vectors
    from ._serialize import serialize_preorder
    from ._types import compute_site_types
    from ._typing_extension import compute_tobl_tags
    from .ast import Lam

    k = len(region.candidate_branches)
    if k == 0:
        raise ValueError("structural hole has no candidates")
    if k > MERA_LEAF_DIM - 1:
        raise ValueError(
            f"too many structural candidates ({k}); witness slots "
            f"available: {MERA_LEAF_DIM - 1}")

    branch_overrides: list[dict[int, np.ndarray]] = []
    pad_vec = _pad_leaf_vec()

    for j, candidate in enumerate(region.candidate_branches):
        # Wrap the candidate in the sketch's binder stack so the M1
        # front-half (resolve_binders + serialize_preorder +
        # compute_site_types) resolves each Var to the correct binder
        # WITHOUT us re-deriving bid index math. We then slice off the
        # wrapper-binder sites to get exactly the candidate's per-node
        # leaf vectors.
        wrapped = candidate
        # sketch_scope is innermost-last; wrap from innermost outward.
        for binder in reversed(sketch_scope):
            if not isinstance(binder, Lam):
                raise ValueError(
                    f"sketch_scope must contain Lam binders only; "
                    f"got {type(binder).__name__}")
            wrapped = Lam(param=binder.param,
                          param_ty=binder.param_ty,
                          body=wrapped)
        # Serialize the closed sub-program.
        n_scope = len(sketch_scope)
        n_cand_nodes = 0
        # Count just the candidate nodes (wrapped has n_scope extra Lams).
        from ._serialize import count_nodes as _count_nodes
        n_cand_nodes = _count_nodes(wrapped) - n_scope
        if n_cand_nodes > region.n_max:
            raise ValueError(
                f"candidate {j} has {n_cand_nodes} nodes, exceeds "
                f"n_max={region.n_max}")
        sites = serialize_preorder(wrapped, N=n_scope + region.n_max)
        type_tags = compute_site_types(wrapped, sites)
        compute_tobl_tags(wrapped, sites)
        # The candidate's sites begin at index n_scope (the wrappers are
        # the first n_scope pre-order sites). The next n_cand_nodes sites
        # are the candidate's nodes; the remainder are PAD (from
        # serialize_preorder's tail-padding) which we override with
        # shape-pad vectors below.
        overrides: dict[int, np.ndarray] = {}
        for m in range(region.n_max):
            cand_site_idx = n_scope + m
            region_node_idx = region.node_start + m
            base_leaf = LEAVES_PER_NODE * region_node_idx
            if m < n_cand_nodes:
                five = node_leaf_vectors(sites[cand_site_idx],
                                         type_tags[cand_site_idx])
                for s_off in range(LEAVES_PER_NODE):
                    overrides[base_leaf + s_off] = five[s_off]
            else:
                # Shape-pad: branches differ on the kind leaf here.
                for s_off in range(LEAVES_PER_NODE):
                    overrides[base_leaf + s_off] = pad_vec.copy()
        # Witness mark on the region root node's `value` leaf. Use a
        # distinct high basis index per branch j so branches with only
        # shape-pad differences are still mutually orthogonal.
        value_off = SPECIES_LEAF_OFFSET["value"]
        root_value_leaf = LEAVES_PER_NODE * region.node_start + value_off
        witness_vec = np.zeros(MERA_LEAF_DIM, dtype=complex)
        # j+1 keeps index 0 unused (concrete branches' default "no value"
        # state) and stays clear of any kind/type/bid/tobl tag because
        # this is the `value` species and the region root in any branch
        # has no semantically-meaningful value tag of its own.
        witness_idx = witness_index(j)
        if witness_idx >= MERA_LEAF_DIM:
            raise ValueError(
                f"witness index {witness_idx} out of leaf range")
        witness_vec[witness_idx] = 1.0
        overrides[root_value_leaf] = witness_vec
        branch_overrides.append(overrides)
    return branch_overrides
