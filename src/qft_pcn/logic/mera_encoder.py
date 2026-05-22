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
from .mera_encoding import MERA_LEAF_DIM, KIND_LAM, KIND_FORALL, KIND_FIX
from .encoding import KIND_PAD as _ENC_KIND_PAD
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
    layout: MeraLayout = field(repr=False, default=None)


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


def encode_mera(ast: Node, n_nodes_max: int = 32,
                chi_layer: int = 16) -> tuple[MERA, MeraEncodingMeta]:
    """Encode an AST into a unit-norm MERA (spec §6).

    Concrete path only in this task; if the AST has holes, raise
    NotImplementedError (Part 2 supplies the hole path).
    """
    if _has_holes(ast):
        raise NotImplementedError(
            "hole-bearing encoding is implemented in Part 2 (Task 10)")

    # Front-half: reuse the MPS encoder pipeline.
    sites = serialize_preorder(ast, N=n_nodes_max)
    type_tags = compute_site_types(ast, sites)
    compute_tobl_tags(ast, sites)   # writes occ.tobl_tag in place
    n_nodes = sum(1 for occ in sites if occ.kind != _ENC_KIND_PAD)

    layout = compute_layout(n_nodes)

    # Build the 5N + pad leaf vectors, node-major.
    leaf_vectors: list[np.ndarray] = []
    for node_idx in range(n_nodes):
        five = node_leaf_vectors(sites[node_idx], type_tags[node_idx])
        leaf_vectors.extend(five)
    while len(leaf_vectors) < layout.n_leaves:
        leaf_vectors.append(_pad_leaf_vector())

    # Concrete program -> product MERA.
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

    meta = MeraEncodingMeta(
        n_nodes=n_nodes, n_leaves=layout.n_leaves, L=layout.L,
        leaf_dim=MERA_LEAF_DIM,
        species_of_leaf=layout.species_of_leaf,
        node_of_leaf=layout.node_of_leaf,
        site_to_ast_path={k: occ.ast_path for k, occ in enumerate(sites)
                          if occ.kind != _ENC_KIND_PAD},
        binder_leaves=binder_leaves,
        use_to_binder=use_to_binder,
        layout=layout,
    )
    return state, meta
