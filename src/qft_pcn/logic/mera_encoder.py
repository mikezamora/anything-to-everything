"""MERA-native logic encoder (spec §6).

Concrete (hole-free) programs encode to a product MERA via
MERA.from_product. Hole-bearing programs (Part 2) start from the product
MERA and apply entangling gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ast import Node, HoleVar
from ._serialize import serialize_preorder, NodeOccupancy
from ._types import compute_site_types
from ._typing_extension import compute_tobl_tags
from ._mera_layout import compute_layout, MeraLayout
from ._mera_leaves import node_leaf_vectors
from ._mera_holes import encode_hole_state
from .mera_encoding import (
    MERA_LEAF_DIM, KIND_LAM, KIND_FORALL, KIND_FIX,
)
from .encoding import KIND_PAD as _ENC_KIND_PAD, KIND_VAR, BID_0, TYPE_ARR_NESTED
from src.qft_pcn.qft.mera import MERA


@dataclass
class MeraEncodingMeta:
    n_nodes: int
    n_leaves: int
    L: int
    leaf_dim: int
    species_of_leaf: list[str]
    node_of_leaf: list[int]
    site_to_ast_path: dict[int, tuple[int, ...]]
    binder_leaves: dict[int, int]      # binder AST node -> its bid leaf
    use_to_binder: dict[int, int]      # use's bid leaf -> binder's bid leaf
    nested_type_index: dict[int, "object"] = field(default_factory=dict)
    layout: MeraLayout = field(repr=False, default=None)
    children_of_node: dict[int, list[int]] = field(default_factory=dict)
    n_nodes_max: int = 0      # encoder node budget (R-Fix unfold guard)


def _binder_kinds() -> set[int]:
    """Kind indices that introduce a binder (Lam, Forall, Fix)."""
    return {KIND_LAM, KIND_FORALL, KIND_FIX}


def _pad_leaf_vector() -> np.ndarray:
    v = np.zeros(MERA_LEAF_DIM, dtype=complex)
    v[_ENC_KIND_PAD] = 1.0    # PAD basis state, index 0
    return v


def _has_holes(ast: Node) -> bool:
    """True if the AST contains a HoleVar (Part-2 path)."""
    found = [False]

    def _walk(n: Node) -> None:
        if isinstance(n, HoleVar):
            found[0] = True
            return
        for attr in ("body", "fn", "arg", "lhs", "rhs", "cond",
                      "then_b", "else_b", "head", "tail"):
            child = getattr(n, attr, None)
            if isinstance(child, Node):
                _walk(child)
    _walk(ast)
    return found[0]


def _hole_nodes(sites: list[NodeOccupancy]) -> list[int]:
    """Indices of nodes whose var_ref carries HoleVar candidates."""
    out: list[int] = []
    for node_idx, occ in enumerate(sites):
        if (occ.kind == KIND_VAR and occ.var_ref is not None
                and occ.var_ref.candidates):
            out.append(node_idx)
    return out


def encode_mera(ast: Node, n_nodes_max: int = 32,
                chi_layer: int = 16) -> tuple[MERA, MeraEncodingMeta]:
    """Encode an AST into a unit-norm MERA (spec §6).

    Concrete (hole-free) programs encode to a product MERA. Hole-bearing
    programs (HoleVar use sites) start from a product MERA with the hole's
    bid leaf in a superposition over candidate binder bid indices, then
    apply CNOT-like entangling gates so the encoded state carries genuine
    tree entanglement between the hole and the candidate binders (spec
    §5.3 — the §1.1 binding-as-entanglement principle on the MERA tree).
    """
    # Front-half: reuse the MPS encoder pipeline.
    sites = serialize_preorder(ast, N=n_nodes_max)
    type_tags = compute_site_types(ast, sites)
    compute_tobl_tags(ast, sites)   # writes occ.tobl_tag in place
    n_nodes = sum(1 for occ in sites if occ.kind != _ENC_KIND_PAD)

    layout = compute_layout(n_nodes)
    hole_nodes = _hole_nodes(sites)

    # Build the 5N + pad concrete leaf vectors, node-major. A hole node's
    # leaves are filled with the concrete form for its first candidate;
    # the per-branch overrides in the hole path replace the bid leaf and
    # the chosen binders' witness leaves (spec §5.3).
    leaf_vectors: list[np.ndarray] = []
    for node_idx in range(n_nodes):
        five = node_leaf_vectors(sites[node_idx], type_tags[node_idx])
        leaf_vectors.extend(five)
    while len(leaf_vectors) < layout.n_leaves:
        leaf_vectors.append(_pad_leaf_vector())

    if hole_nodes:
        # Hole path (spec §5.3): build the rank-k branch superposition and
        # encode it as a genuinely tree-entangled MERA. Each hole resolves
        # to one of its candidate binders; the hole's bid leaf and the
        # chosen binder's witness leaf are entangled through the tree.
        holes: list[dict] = []
        for node_idx in hole_nodes:
            var_ref = sites[node_idx].var_ref
            holes.append({
                "hole_bid_leaf": layout.leaf_of(node_idx, "bid"),
                "cand_bid_values": [BID_0 + depth
                                    for _, depth in var_ref.candidates],
                "cand_witness_leaves": [
                    layout.leaf_of(binder_site, "value")
                    for binder_site, _ in var_ref.candidates],
            })
        state = encode_hole_state(leaf_vectors, holes, chi_layer=chi_layer)
    else:
        # Concrete path: product MERA, no tree entanglement needed.
        state = MERA.from_product(leaf_vectors, chi_layer=chi_layer)

    state.normalize()

    # Binder / use bookkeeping for the meta.
    binder_kinds = _binder_kinds()
    binder_leaves: dict[int, int] = {}
    use_to_binder: dict[int, int] = {}
    for node_idx in range(n_nodes):
        occ = sites[node_idx]
        if occ.kind in binder_kinds:
            binder_leaves[node_idx] = layout.leaf_of(node_idx, "bid")
        if occ.var_ref is not None:
            use_leaf = layout.leaf_of(node_idx, "bid")
            binder_node = occ.var_ref.binder_site
            use_to_binder[use_leaf] = layout.leaf_of(binder_node, "bid")

    # Nested-arrow type side table, keyed by node index — exactly what the
    # MPS meta.nested_type_index carries, so the shared structural parse
    # can recover a Lam's higher-order param type.
    nested_type_index: dict[int, object] = {}
    for node_idx in range(n_nodes):
        if type_tags[node_idx] == TYPE_ARR_NESTED and sites[node_idx].ty is not None:
            nested_type_index[node_idx] = sites[node_idx].ty

    # children_of_node: parent AST node index -> child node indices.
    # A node c is a child of node p iff c's ast_path == p's ast_path + (j,)
    # for some j (one level deeper, directly under p). This is layout
    # metadata from the pre-order walk (spec §5.6 (b)) — NOT an AST walk.
    children_of_node: dict[int, list[int]] = {}
    path_to_node: dict[tuple, int] = {}
    for node_idx in range(n_nodes):
        path_to_node[sites[node_idx].ast_path] = node_idx
    for node_idx in range(n_nodes):
        children_of_node[node_idx] = []
    for node_idx in range(n_nodes):
        path = sites[node_idx].ast_path
        if len(path) >= 1:
            parent_path = path[:-1]
            parent = path_to_node.get(parent_path)
            if parent is not None and parent != node_idx:
                children_of_node[parent].append(node_idx)
    for node_idx in children_of_node:
        children_of_node[node_idx].sort()

    meta = MeraEncodingMeta(
        n_nodes=n_nodes, n_leaves=layout.n_leaves, L=layout.L,
        leaf_dim=MERA_LEAF_DIM,
        species_of_leaf=layout.species_of_leaf,
        node_of_leaf=layout.node_of_leaf,
        site_to_ast_path={k: occ.ast_path for k, occ in enumerate(sites)
                          if occ.kind != _ENC_KIND_PAD},
        binder_leaves=binder_leaves,
        use_to_binder=use_to_binder,
        nested_type_index=nested_type_index,
        layout=layout,
        children_of_node=children_of_node,
        n_nodes_max=n_nodes_max,
    )
    return state, meta
