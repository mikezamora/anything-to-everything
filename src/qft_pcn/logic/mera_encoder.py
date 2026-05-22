"""MERA-native logic encoder (spec §6).

Concrete (hole-free) programs encode to a product MERA via
MERA.from_product. Hole-bearing programs (Part 2) start from the product
MERA and apply entangling gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ast import (
    Node, HoleVar, Var, Lam, App, If, Bin, IntLit, Forall, Fix,
    Succ, Cons, Eq,
)
from ._serialize import serialize_preorder, NodeOccupancy, count_nodes
from ._types import compute_site_types
from ._typing_extension import compute_tobl_tags
from ._mera_layout import compute_layout, MeraLayout
from ._mera_leaves import node_leaf_vectors
from ._mera_holes import encode_hole_state, structural_hole_branches
from .mera_encoding import (
    MERA_LEAF_DIM, KIND_LAM, KIND_FORALL, KIND_FIX, LEAVES_PER_NODE,
)
from .encoding import (
    KIND_PAD as _ENC_KIND_PAD, KIND_VAR, BID_0, TYPE_ARR_NESTED,
    EncodingTooLarge,
)
from .mera_synthesis.encode_ext import _expand_structural_holes
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
    hole_regions: list = field(default_factory=list)   # M3 structural-hole layout


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


def _has_structural_hole(ast: Node) -> bool:
    found = [False]
    def _walk(n: Node) -> None:
        if found[0]:
            return
        if isinstance(n, HoleVar) and n.candidate_kind() == "structural":
            found[0] = True
            return
        for attr in ("body", "fn", "arg", "lhs", "rhs", "cond",
                     "then_b", "else_b", "head", "tail"):
            child = getattr(n, attr, None)
            if isinstance(child, Node):
                _walk(child)
    _walk(ast)
    return found[0]


def _walk_structural_holes(ast: Node):
    """Walk pre-order; for each structural HoleVar yield
    (preorder_index_in_original, sketch_scope_list, hole_node).

    sketch_scope is the list of Lam binders in scope at the hole position,
    innermost last.
    """
    counter = [0]
    stack: list[Lam] = []
    out: list[tuple[int, list, HoleVar]] = []

    def _walk(n: Node) -> None:
        idx = counter[0]
        counter[0] += 1
        if isinstance(n, HoleVar):
            if n.candidate_kind() == "structural":
                out.append((idx, list(stack), n))
            return
        if isinstance(n, Lam):
            stack.append(n)
            _walk(n.body)
            stack.pop()
            return
        if isinstance(n, (Forall, Fix)):
            # Treat as binders for scope tracking, but only Lam is used in
            # sketch_scope for now (Var resolves via param name match).
            stack.append(n)   # type: ignore[arg-type]
            _walk(n.body)
            stack.pop()
            return
        for attr in ("fn", "arg", "lhs", "rhs", "cond",
                     "then_b", "else_b", "head", "tail"):
            child = getattr(n, attr, None)
            if isinstance(child, Node):
                _walk(child)
    _walk(ast)
    return out


def _substitute_structural_holes(ast: Node) -> Node:
    """Return a copy of `ast` with each structural HoleVar replaced by
    ``IntLit(0)``. The substitution is purely so the M1 front-half
    (resolve_binders, serialize_preorder, compute_site_types) can run on
    a fully concrete AST — the hole positions are recovered separately
    by ``_walk_structural_holes``.
    """
    if isinstance(ast, HoleVar) and ast.candidate_kind() == "structural":
        return IntLit(val=0)
    if isinstance(ast, Lam):
        return Lam(param=ast.param, param_ty=ast.param_ty,
                   body=_substitute_structural_holes(ast.body))
    if isinstance(ast, Forall):
        return Forall(param=ast.param, param_ty=ast.param_ty,
                      body=_substitute_structural_holes(ast.body))
    if isinstance(ast, Fix):
        return Fix(param=ast.param, param_ty=ast.param_ty,
                   body=_substitute_structural_holes(ast.body))
    if isinstance(ast, App):
        return App(fn=_substitute_structural_holes(ast.fn),
                   arg=_substitute_structural_holes(ast.arg))
    if isinstance(ast, If):
        return If(cond=_substitute_structural_holes(ast.cond),
                  then_b=_substitute_structural_holes(ast.then_b),
                  else_b=_substitute_structural_holes(ast.else_b))
    if isinstance(ast, Bin):
        return Bin(op=ast.op,
                   lhs=_substitute_structural_holes(ast.lhs),
                   rhs=_substitute_structural_holes(ast.rhs))
    if isinstance(ast, Succ):
        return Succ(arg=_substitute_structural_holes(ast.arg))
    if isinstance(ast, Cons):
        return Cons(head=_substitute_structural_holes(ast.head),
                    tail=_substitute_structural_holes(ast.tail))
    if isinstance(ast, Eq):
        return Eq(lhs=_substitute_structural_holes(ast.lhs),
                  rhs=_substitute_structural_holes(ast.rhs))
    return ast


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
    # Structural-hole path (M3, spec §5.3): if any HoleVar carries Node
    # candidates, the front-half cannot resolve the candidates as names
    # (the M1 path is for var-holes only). Route to the dedicated
    # rank-k structural superposition encoder, which uses the M1
    # front-half on a structurally-substituted AST and folds the
    # candidate sub-trees into one MERA state via from_term_superposition.
    if _has_structural_hole(ast):
        return _encode_with_structural_holes(
            ast, n_nodes_max=n_nodes_max, chi_layer=chi_layer)

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


# --------------------------------------------------------------------------
# M3 structural-hole encoding (spec §5.3)
# --------------------------------------------------------------------------


def _encode_with_structural_holes(ast: Node, n_nodes_max: int,
                                  chi_layer: int) -> tuple[MERA, MeraEncodingMeta]:
    """Encode an AST containing structural HoleVars (Node candidates).

    Builds ONE rank-k MERA state where the k candidate sub-trees become
    k mutually-orthogonal branch directions of a single quantum state
    (spec §5.3, principle 6). No Python loop over candidates outside the
    MERA state — branches are handed to ``MERA.from_term_superposition``.
    """
    # 1) Identify structural holes in the original AST (preorder index +
    #    in-scope binders), then substitute each with IntLit(0) so the M1
    #    front-half can serialize the resulting concrete program.
    hole_walk = _walk_structural_holes(ast)
    substituted = _substitute_structural_holes(ast)
    sub_sites = serialize_preorder(substituted, N=n_nodes_max)
    sub_type_tags = compute_site_types(substituted, sub_sites)
    compute_tobl_tags(substituted, sub_sites)
    n_sub = sum(1 for occ in sub_sites if occ.kind != _ENC_KIND_PAD)

    # 2) Pair each structural hole with its substituted-AST site index.
    #    Both walks use the same canonical pre-order, so the hole's
    #    original preorder index equals its position in the substituted
    #    pre-order (HoleVar->IntLit is a leaf-for-leaf swap).
    hole_sub_indices: list[int] = [idx for idx, _scope, _h in hole_walk]
    hole_scopes: list[list] = [scope for _idx, scope, _h in hole_walk]
    hole_nodes_obj: list[HoleVar] = [h for _idx, _scope, h in hole_walk]
    # Match each substituted-site index to a HoleRegion (computed on the
    # original AST in the same preorder).
    _skel, regions_in = _expand_structural_holes(ast)
    if len(regions_in) != len(hole_walk):
        raise RuntimeError(
            "structural-hole preorder mismatch between scope walk and "
            "_expand_structural_holes")

    # 3) Compute the expanded layout. Each substituted-hole site (1 slot)
    #    expands to n_max slots; later sites shift right by (n_max - 1).
    #    Build: sub_idx -> expanded_slot for concrete sites; expanded
    #    region_node_start per hole.
    expanded_slot_of_sub: dict[int, int] = {}
    region_node_starts: list[int] = []
    shift = 0
    hole_set = set(hole_sub_indices)
    sub_idx_to_region: dict[int, int] = {
        sub_idx: r for r, sub_idx in enumerate(hole_sub_indices)
    }
    n_total = 0
    # Walk in substituted preorder order so the shift accumulates left-to-right.
    for sub_idx in range(n_sub):
        if sub_idx in hole_set:
            r = sub_idx_to_region[sub_idx]
            region_node_starts.append(sub_idx + shift)
            n_total += regions_in[r].n_max
            shift += regions_in[r].n_max - 1
        else:
            expanded_slot_of_sub[sub_idx] = sub_idx + shift
            n_total += 1

    if n_total < 1:
        raise EncodingTooLarge(n_nodes=n_total, N=n_nodes_max)
    layout = compute_layout(n_total)
    n_leaves = layout.n_leaves

    # Guard against runaway leaf counts / candidate counts. chi_layer is
    # the per-layer bond dim cap; rank-k <= chi_layer is the principal
    # constraint for from_term_superposition.
    total_k = 1
    for r in regions_in:
        total_k *= max(1, len(r.candidate_branches))
    if total_k > chi_layer:
        raise EncodingTooLarge(n_nodes=n_total, N=n_nodes_max)

    # 4) Build the base per-leaf vectors (concrete-node leaves repeated
    #    across all branches; hole-region leaves overridden per branch).
    base_leaves: list[np.ndarray] = [_pad_leaf_vector() for _ in range(n_leaves)]
    for sub_idx in range(n_sub):
        if sub_idx in hole_set:
            continue
        new_idx = expanded_slot_of_sub[sub_idx]
        five = node_leaf_vectors(sub_sites[sub_idx], sub_type_tags[sub_idx])
        for s_off in range(LEAVES_PER_NODE):
            base_leaves[LEAVES_PER_NODE * new_idx + s_off] = five[s_off]

    # Build the regions list with expanded node_starts (what meta.hole_regions
    # exposes to callers / tests).
    hole_regions: list = []
    for r, region_in in enumerate(regions_in):
        from .mera_synthesis.encode_ext import HoleRegion
        hole_regions.append(HoleRegion(
            node_start=region_node_starts[r],
            n_max=region_in.n_max,
            candidate_branches=region_in.candidate_branches,
        ))

    # 5) Per-region per-branch overrides.
    per_region_branches: list[list[dict[int, np.ndarray]]] = []
    for r, region in enumerate(hole_regions):
        branches = structural_hole_branches(
            region, hole_scopes[r], layout)
        # Sanity: each branch covers exactly the 5*n_max region leaves.
        expected = LEAVES_PER_NODE * region.n_max
        for b in branches:
            if len(b) != expected:
                raise RuntimeError(
                    f"structural_hole_branches returned {len(b)} overrides; "
                    f"expected {expected}")
        per_region_branches.append(branches)

    # 6) Cartesian product across structural holes (each hole is resolved
    #    independently — the architecture's structural superposition).
    #    For each combined branch, assemble the full per-leaf vector list.
    branch_choices: list[list[int]] = [[]]
    for branches in per_region_branches:
        branch_choices = [c + [j] for c in branch_choices
                          for j in range(len(branches))]
    K = len(branch_choices)
    amp = 1.0 / np.sqrt(K)
    terms: list[tuple[complex, list[np.ndarray]]] = []
    for choice in branch_choices:
        leaves = [v.copy() for v in base_leaves]
        for r, j in enumerate(choice):
            for leaf_idx, vec in per_region_branches[r][j].items():
                leaves[leaf_idx] = vec
        terms.append((complex(amp), leaves))

    state = MERA.from_term_superposition(terms, chi_layer=chi_layer)
    state.normalize()

    # 7) Meta. We populate the same bookkeeping as the M1 path, computed
    #    from the substituted-AST sites and the expanded slot mapping for
    #    binder / use leaves. Hole-region leaves do not participate.
    binder_kinds = _binder_kinds()
    binder_leaves: dict[int, int] = {}
    use_to_binder: dict[int, int] = {}
    site_to_ast_path: dict[int, tuple] = {}
    for sub_idx in range(n_sub):
        if sub_idx in hole_set:
            continue
        new_idx = expanded_slot_of_sub[sub_idx]
        occ = sub_sites[sub_idx]
        site_to_ast_path[new_idx] = occ.ast_path
        if occ.kind in binder_kinds:
            binder_leaves[new_idx] = layout.leaf_of(new_idx, "bid")
        if occ.var_ref is not None:
            use_leaf = layout.leaf_of(new_idx, "bid")
            binder_sub = occ.var_ref.binder_site
            # The binder is always a concrete node (not a hole) — map it.
            binder_new = expanded_slot_of_sub.get(binder_sub)
            if binder_new is not None:
                use_to_binder[use_leaf] = layout.leaf_of(binder_new, "bid")

    nested_type_index: dict[int, object] = {}
    for sub_idx in range(n_sub):
        if sub_idx in hole_set:
            continue
        new_idx = expanded_slot_of_sub[sub_idx]
        if (sub_type_tags[sub_idx] == TYPE_ARR_NESTED
                and sub_sites[sub_idx].ty is not None):
            nested_type_index[new_idx] = sub_sites[sub_idx].ty

    # children_of_node by expanded-slot ast_path (concrete nodes only).
    children_of_node: dict[int, list[int]] = {i: [] for i in range(n_total)}
    path_to_node: dict[tuple, int] = {}
    for new_idx, path in site_to_ast_path.items():
        path_to_node[path] = new_idx
    for new_idx, path in site_to_ast_path.items():
        if len(path) >= 1:
            parent = path_to_node.get(path[:-1])
            if parent is not None and parent != new_idx:
                children_of_node[parent].append(new_idx)
    for k in children_of_node:
        children_of_node[k].sort()

    meta = MeraEncodingMeta(
        n_nodes=n_total, n_leaves=layout.n_leaves, L=layout.L,
        leaf_dim=MERA_LEAF_DIM,
        species_of_leaf=layout.species_of_leaf,
        node_of_leaf=layout.node_of_leaf,
        site_to_ast_path=site_to_ast_path,
        binder_leaves=binder_leaves,
        use_to_binder=use_to_binder,
        nested_type_index=nested_type_index,
        layout=layout,
        children_of_node=children_of_node,
        n_nodes_max=n_nodes_max,
        hole_regions=hole_regions,
    )
    return state, meta
