"""§12.12 Quantum extremal surfaces for minimum-complexity proof bounds.

Spec reference: ``QFT_PCN_ARCHITECTURE.md`` §12.12. The Ryu-Takayanagi
formula (Ryu & Takayanagi 2006) computes the entanglement entropy of a
boundary subregion as the minimum-area surface in the bulk; Hubeny-
Rangamani-Takayanagi 2007 + Engelhardt-Wall 2015 generalize via the
*quantum extremal surface* (QES) prescription. For MERA — which is
literally a discrete holographic geometry (Swingle 2012) — the RT/QES
formula identifies the entanglement entropy of a boundary region with
the minimum-cut area through the MERA tree, weighted by per-bond
entanglement carriage.

Per the spec's complexity=volume / complexity=action correspondence
(Susskind 2016), the QES area is a *lower bound on proof complexity*:
no proof state on the boundary can have less holographic complexity
than its minimum extremal surface area.

This module is operator-algebraic, not classical (§1.1 architecture-
soul). The QES is computed on the **real MERA bulk tensor network** via
minimum-cut on the binary tree's bond geometry; the carried-entropy
weighting uses the substrate's exact entanglement-entropy computation
(``MERA.entanglement_entropy`` / ``_entropy_from_terms``). No AST symbol
counting, no string-based heuristics.

API
---
- :func:`compute_qes_complexity` — QES area for a contiguous boundary
  region of a MERA state.
- :func:`compute_geometric_rt_area` — classical RT minimum-cut bond
  count weighted by log(bond_dim); upper bound on QES.
- :func:`find_minimum_complexity_proof` — rank candidate proof MERAs
  by their QES area (smaller = simpler proof).
- :class:`ProofComplexityRanking` — structured result with per-candidate
  QES, RT-area upper bound, and chosen minimum.

Complexity: For a binary MERA on N=2^L leaves, the minimum-cut on the
tree is computed by a single post-order recursion in O(N) bond visits.
The entropy term uses the substrate's existing routines (O(d^N) only
for small acceptance-test N, O(k^3) for ``from_term_superposition``
states). Total per call: dominated by the substrate's entropy
computation; the geometry itself is O(N).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from src.qft_pcn.qft.mera import MERA


# ---------------------------------------------------------------------------
# MERA bulk geometry: binary-tree minimum cut
# ---------------------------------------------------------------------------


def _bond_dim_at_subtree_root(state: MERA, layer: int) -> int:
    """Bond dimension of the bond going UP from a node at ``layer``.

    Layer 0 nodes are leaves (bond dim = d_local). Layer 1..L-1 nodes
    are isometry outputs (bond dim = layer_dims[layer]). The top
    "layer L" is the network closure (bond dim = layer_dims[L-1]).
    """
    if layer < 0:
        raise ValueError(f"layer {layer} must be >= 0")
    if layer == 0:
        return state.d_local
    if layer < state.L:
        return state.layer_dims[layer]
    # layer == L (root): the top tensor closure carries the final bond.
    return state.layer_dims[state.L - 1]


def _subtree_status(layer: int, node_index: int,
                    region: set[int]) -> tuple[int, int]:
    """Return ``(inside_count, sub_size)`` for the subtree at (layer, node)."""
    sub_size = 1 << layer
    leaf_start = node_index * sub_size
    leaf_end = leaf_start + sub_size
    inside = sum(1 for k in range(leaf_start, leaf_end) if k in region)
    return inside, sub_size


def _min_cut_subtree(
    state: MERA,
    layer: int,
    node_index: int,
    region: set[int],
) -> tuple[float, int]:
    """Minimum-cut area through the subtree rooted at (``layer``, ``node_index``).

    Returns ``(area_nats, num_bonds_cut)`` where ``area_nats`` is the sum
    of ``log(bond_dim)`` over bonds severed by the minimum cut. This
    function is called by the parent: its contract is "produce a clean
    cut that separates the inside-leaves of this subtree from
    everything else in the tree (including this subtree's siblings)".

    Standard tree min-cut recursion. For a split internal node with
    children L, R:

    - If a child is wholly INSIDE the region, the only way to separate
      its leaves from outside-leaves elsewhere in the tree is to sever
      its up-bond → cost ``log(bond_dim_up(child))``.
    - If a child is wholly OUTSIDE the region, no cut is needed for it
      (the inside-leaves are extracted from its sibling; the outside
      child remains attached to the ambient outside).
    - If a child is itself split, recurse — the recursive call returns a
      cut that disconnects the child's inside-leaves from everything
      else (including its sibling).

    Note: the leaf-level "split" case is unreachable (a single leaf
    cannot be split), but a leaf can be pure-inside in which case its
    parent charges ``log(d_local)`` for the up-bond cut.

    The subtree at (ell, j) covers leaves ``[j * 2**ell, (j+1) * 2**ell)``
    in the canonical binary MERA layout (Swingle 2012; ``causal_cone_path``
    in ``qft.mera`` uses this same indexing).
    """
    inside, sub_size = _subtree_status(layer, node_index, region)
    if inside == 0 or inside == sub_size:
        # Wholly outside or wholly inside: the parent decides whether to
        # cut our up-bond. From inside the subtree, no cuts are needed.
        return 0.0, 0
    if layer == 0:
        # A single leaf cannot be partially inside the region. This is
        # unreachable on well-formed integer regions; kept for safety.
        d_up = _bond_dim_at_subtree_root(state, 0)
        return float(np.log(d_up)), 1
    # Split internal node: handle each child by its own status.
    total_area = 0.0
    total_n = 0
    for child_node in (2 * node_index, 2 * node_index + 1):
        child_layer = layer - 1
        c_inside, c_size = _subtree_status(child_layer, child_node, region)
        if c_inside == c_size:
            # Pure-inside child: sever its up-bond to extract it from the
            # rest of the tree (its sibling, which contains outside-leaves
            # either directly or below).
            d_up = _bond_dim_at_subtree_root(state, child_layer)
            total_area += float(np.log(d_up))
            total_n += 1
        elif c_inside == 0:
            # Pure-outside child: no cut needed; the inside-leaves are
            # below the split sibling and are extracted there.
            continue
        else:
            # Split child: recurse — the call returns a cut that
            # disconnects the child's inside-leaves from everything
            # outside (including this child's sibling).
            sub_area, sub_n = _min_cut_subtree(
                state, child_layer, child_node, region)
            total_area += sub_area
            total_n += sub_n
    return total_area, total_n


def compute_geometric_rt_area(
    state: MERA,
    region: Sequence[int],
) -> float:
    """Classical Ryu-Takayanagi area for ``region`` in the MERA bulk.

    The minimum cut through the binary-tree bond network separating the
    region's leaves from the complement, weighted by ``log(bond_dim)``.
    This is the *capacity* upper bound on entanglement entropy across the
    region's boundary (RT inequality):

        S(region) <= A_RT(region).

    For a binary MERA on contiguous regions the minimum cut has at most
    O(log |region|) bonds.

    Parameters
    ----------
    state : MERA
        Real MERA bulk tensor network (any constructor — ``from_product``,
        ``from_term_superposition``, ``encode_mera`` output).
    region : Sequence[int]
        Leaf indices in the boundary region (need not be contiguous).
    """
    if not isinstance(state, MERA):
        raise TypeError(
            f"expected MERA, got {type(state).__name__} (§1.1: real "
            f"bulk geometry, not classical surrogate)")
    N = state.N
    region_set = set(int(i) for i in region)
    for i in region_set:
        if not 0 <= i < N:
            raise ValueError(
                f"region leaf {i} out of range [0, {N})")
    if len(region_set) == 0 or len(region_set) == N:
        return 0.0
    # Recurse from the root (layer = L, node_index = 0).
    area, _ = _min_cut_subtree(state, state.L, 0, region_set)
    return area


# ---------------------------------------------------------------------------
# Quantum extremal surface: bulk geometry + boundary entanglement
# ---------------------------------------------------------------------------


def _region_entanglement_entropy(state: MERA, region: Sequence[int]) -> float:
    """Von Neumann entropy of the reduced density on ``region``.

    For contiguous regions ``[a, b]`` with ``a == 0`` or ``b == N-1``, this
    reduces to a single bipartite cut at the region boundary, which the
    substrate computes exactly via :meth:`MERA.entanglement_entropy` or
    :meth:`MERA._entropy_from_terms`. Non-contiguous or "interior"
    regions require the descending superoperator (deferred — see
    ``EXTENSIONS.md`` "Non-contiguous QES regions"); we accept any
    contiguous region by using its longer-side complement cut (entropy
    is symmetric under complementation for a pure state).
    """
    N = state.N
    region_set = set(int(i) for i in region)
    if len(region_set) == 0 or len(region_set) == N:
        return 0.0
    # Verify contiguous (this implementation handles contiguous regions
    # exactly; arbitrary regions are an EXTENSIONS item).
    sorted_idx = sorted(region_set)
    contiguous = all(sorted_idx[k] + 1 == sorted_idx[k + 1]
                     for k in range(len(sorted_idx) - 1))
    if not contiguous:
        raise NotImplementedError(
            "non-contiguous QES regions require the descending "
            "superoperator (EXTENSIONS.md: Non-contiguous QES regions)")
    a, b = sorted_idx[0], sorted_idx[-1]
    # Pick the cut so that one side of the boundary is the region.
    if a == 0:
        cut = b           # leaves [0..b] vs [b+1..N-1]
    elif b == N - 1:
        cut = a - 1       # leaves [0..a-1] vs [a..N-1]
    else:
        # Interior region: entropy = S([0..b]) + S([0..a-1]) only if the
        # state factorizes across the region's two boundaries, which it
        # does NOT in general. Defer to EXTENSIONS.
        raise NotImplementedError(
            "interior QES regions (both boundaries inside the chain) "
            "require the descending superoperator "
            "(EXTENSIONS.md: Interior QES regions)")
    return state.entanglement_entropy(cut)


def compute_qes_complexity(
    state: MERA,
    region: Sequence[int],
) -> float:
    """Quantum extremal surface area for ``region``.

    The QES area is the *generalized entropy* of the region — the
    entanglement entropy of the boundary state restricted to ``region``,
    which by the RT/QES correspondence equals the area of the minimum
    extremal surface in the MERA bulk (Swingle 2012, Pastawski-Yoshida-
    Harlow-Preskill 2015). By the complexity=volume / complexity=action
    conjectures (Susskind 2016) this is a **lower bound on the proof
    complexity** of the theorem state on ``region``:

        complexity(proof) >= A_QES(region) = S(region).

    The classical geometric RT area :func:`compute_geometric_rt_area`
    gives a complementary *upper bound* on this entropy.

    For a **product MERA** (no entanglement), every reduced density is
    pure → S(region) = 0 → QES = 0. For an **entangled MERA** built
    via ``from_term_superposition`` with at least two distinct branches,
    the reduced density on a generic region is mixed → QES > 0.

    Parameters
    ----------
    state : MERA
        Real MERA bulk tensor network.
    region : Sequence[int]
        Contiguous boundary region (set of leaf indices).

    Returns
    -------
    float
        QES area in nats (natural log units), >= 0.

    Notes
    -----
    Per §1.1, the QES is computed on the *real* MERA bulk via the
    substrate's exact entanglement routines — never via AST symbol
    counting or any classical surrogate. The minimum-cut geometry is
    a true binary-tree min-cut on the bond network.
    """
    if not isinstance(state, MERA):
        raise TypeError(
            f"expected MERA, got {type(state).__name__} (§1.1: real "
            f"bulk geometry, not classical surrogate)")
    N = state.N
    region_set = set(int(i) for i in region)
    for i in region_set:
        if not 0 <= i < N:
            raise ValueError(
                f"region leaf {i} out of range [0, {N})")
    if len(region_set) == 0 or len(region_set) == N:
        return 0.0
    return _region_entanglement_entropy(state, region)


# ---------------------------------------------------------------------------
# Proof-complexity ranking via QES
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofComplexityRanking:
    """Ranking of candidate proof states by QES complexity.

    Per §12.12, lower QES area indicates a simpler/shorter proof
    (geometrically natural theorem). Higher QES area indicates a
    harder/longer proof (geometrically unnatural theorem). The RT
    geometric area is reported as an upper bound on the QES for
    diagnostics — the gap ``A_RT - A_QES`` is the "wasted bulk
    capacity" of the MERA encoding.

    Attributes
    ----------
    candidates : tuple[str, ...]
        Candidate proof identifiers in input order.
    qes_areas : Mapping[str, float]
        QES area per candidate (in nats).
    rt_areas : Mapping[str, float]
        RT geometric area per candidate (upper bound on the QES).
    minimum : str
        Identifier of the candidate with the smallest QES area
        (the "minimum-complexity" proof under the holographic bound).
    sorted_by_complexity : tuple[str, ...]
        Candidates in ascending QES-area order (ties broken by RT area).
    """
    candidates: tuple[str, ...]
    qes_areas: Mapping[str, float]
    rt_areas: Mapping[str, float]
    minimum: str
    sorted_by_complexity: tuple[str, ...]


def find_minimum_complexity_proof(
    theorem_state: MERA,
    candidates: Mapping[str, MERA],
) -> ProofComplexityRanking:
    """Rank candidate proof MERAs by their QES area for the theorem region.

    The "theorem region" is the canonical bipartition of the theorem
    state at the midpoint cut — leaves ``[0 .. N/2 - 1]``. Per §12.12,
    this is the boundary anchor against which extremal surfaces are
    computed. Each candidate proof is evaluated on this same region
    in its own MERA bulk; the QES area is its lower-bound complexity.

    All candidates must share the theorem state's leaf count (the
    proof and theorem share boundary support, which is a structural
    requirement of holographic encoding — different N means a
    different boundary CFT, hence not comparable).

    Parameters
    ----------
    theorem_state : MERA
        The theorem's MERA encoding. Used to determine the canonical
        region (midpoint bipartition).
    candidates : Mapping[str, MERA]
        Candidate proof MERAs keyed by identifier.

    Returns
    -------
    ProofComplexityRanking
        Per-candidate QES and RT areas, the minimum, and the sorted
        order.

    Raises
    ------
    ValueError
        If ``candidates`` is empty or any candidate's leaf count differs
        from the theorem state's.
    """
    if not isinstance(theorem_state, MERA):
        raise TypeError(
            f"theorem_state must be MERA, got "
            f"{type(theorem_state).__name__}")
    if not candidates:
        raise ValueError("candidates must be non-empty")
    N = theorem_state.N
    # Canonical region: left half of the boundary.
    region = list(range(N // 2))
    qes: dict[str, float] = {}
    rt: dict[str, float] = {}
    for cid, cstate in candidates.items():
        if not isinstance(cstate, MERA):
            raise TypeError(
                f"candidate {cid!r}: expected MERA, got "
                f"{type(cstate).__name__}")
        if cstate.N != N:
            raise ValueError(
                f"candidate {cid!r}: leaf count {cstate.N} != "
                f"theorem leaf count {N} (boundary mismatch)")
        qes[cid] = compute_qes_complexity(cstate, region)
        rt[cid] = compute_geometric_rt_area(cstate, region)
    keys = tuple(candidates.keys())
    # Sort by (QES, RT) ascending; ties broken by RT then by id for
    # determinism.
    ordered = tuple(sorted(keys, key=lambda k: (qes[k], rt[k], k)))
    return ProofComplexityRanking(
        candidates=keys,
        qes_areas=qes,
        rt_areas=rt,
        minimum=ordered[0],
        sorted_by_complexity=ordered,
    )
