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
from .mera_synthesis.encode_ext import _expand_structural_holes, Bundle
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
    witness_node_ranges: list = field(default_factory=list)  # M3 synthesis: per-example witness node-index ranges (tuple of ints)


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
    # Bundle path (M3, Task 7): a forest of sketch + witness sub-trees
    # is encoded as one MERA state via per-child serialization. The
    # Bundle node itself is never written; its children populate the
    # virtual pre-order site list.
    if isinstance(ast, Bundle):
        return _encode_bundle(ast, n_nodes_max=n_nodes_max,
                              chi_layer=chi_layer)

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


# --------------------------------------------------------------------------
# M3 Task 7: Bundle (multi-root forest) encoding
# --------------------------------------------------------------------------


def _serialize_child(child: Node, n_nodes_max: int):
    """Serialize one Bundle child to (sites, type_tags, n_nodes)."""
    sites = serialize_preorder(child, N=n_nodes_max)
    type_tags = compute_site_types(child, sites)
    compute_tobl_tags(child, sites)
    n_nodes = sum(1 for occ in sites if occ.kind != _ENC_KIND_PAD)
    return sites[:n_nodes], type_tags[:n_nodes], n_nodes


def _structural_segment_data(ast: Node, n_nodes_max: int):
    """Compute the structural-hole expansion data for one AST segment.

    Returns a dict with:
      - n_total: total expanded slot count for this segment
      - sub_sites, sub_type_tags, n_sub
      - expanded_slot_of_sub: dict[sub_idx -> expanded slot (segment-local)]
      - hole_set: set of sub_idx that are holes
      - region_node_starts: list of expanded-slot starts per hole (segment-local)
      - regions_in: list[HoleRegion] (the un-expanded input regions)
      - hole_scopes: list of lex-scope binder lists per hole

    Mirrors steps 1-3 of _encode_with_structural_holes, but does not build
    leaves / layout / state. Pure layout bookkeeping.
    """
    hole_walk = _walk_structural_holes(ast)
    substituted = _substitute_structural_holes(ast)
    sub_sites = serialize_preorder(substituted, N=n_nodes_max)
    sub_type_tags = compute_site_types(substituted, sub_sites)
    compute_tobl_tags(substituted, sub_sites)
    n_sub = sum(1 for occ in sub_sites if occ.kind != _ENC_KIND_PAD)

    hole_sub_indices = [idx for idx, _scope, _h in hole_walk]
    hole_scopes = [scope for _idx, scope, _h in hole_walk]
    _skel, regions_in = _expand_structural_holes(ast)
    if len(regions_in) != len(hole_walk):
        raise RuntimeError(
            "structural-hole preorder mismatch between scope walk and "
            "_expand_structural_holes")

    expanded_slot_of_sub: dict[int, int] = {}
    region_node_starts: list[int] = []
    shift = 0
    hole_set = set(hole_sub_indices)
    sub_idx_to_region = {sub_idx: r for r, sub_idx in enumerate(hole_sub_indices)}
    n_total = 0
    for sub_idx in range(n_sub):
        if sub_idx in hole_set:
            r = sub_idx_to_region[sub_idx]
            region_node_starts.append(sub_idx + shift)
            n_total += regions_in[r].n_max
            shift += regions_in[r].n_max - 1
        else:
            expanded_slot_of_sub[sub_idx] = sub_idx + shift
            n_total += 1

    return {
        "n_total": n_total,
        "sub_sites": sub_sites,
        "sub_type_tags": sub_type_tags,
        "n_sub": n_sub,
        "expanded_slot_of_sub": expanded_slot_of_sub,
        "hole_set": hole_set,
        "region_node_starts": region_node_starts,
        "regions_in": regions_in,
        "hole_scopes": hole_scopes,
    }


def _encode_bundle(bundle: Bundle, n_nodes_max: int,
                   chi_layer: int) -> tuple[MERA, MeraEncodingMeta]:
    """Encode a Bundle of children as one MERA state.

    Each child is serialized independently in canonical pre-order; the
    resulting NodeOccupancy lists are concatenated to one virtual
    pre-order whose total length sizes the MERA layout. Per-child
    binder/var_ref site indices are rebased to the concatenated index
    space (M1 var_refs are local to each child's serialization). The
    witness sub-trees occupy children[1:]; their node-index ranges are
    recorded in meta.witness_node_ranges so the synthesis Hamiltonian
    can locate witness roots and exclude witness nodes from H_size.
    """
    if len(bundle.children) == 0:
        raise ValueError("Bundle has no children")

    # Detect a structural-hole sketch in children[0]: it cannot be passed
    # through the M1 var-hole serializer (which trips on HoleVar inside
    # `_emit`). For that case we route children[0] through the structural
    # segment helper (mirroring _encode_with_structural_holes) and pack
    # the witness children (children[1..]) as concrete segments after it.
    sketch_is_structural = _has_structural_hole(bundle.children[0])

    # 1) Serialize each child (per its hole kind); rebase var_ref.binder_site
    #    to the unified pre-order. site_to_path is per-child; we prefix each
    #    child's ast_path with (child_index,) so paths are globally unique.
    #    For the structural sketch we capture the segment data dict and
    #    reconstruct an aligned `sites_segment` / `type_tags_segment` over
    #    the expanded slots (holes -> placeholder NodeOccupancy with kind
    #    KIND_PAD so the meta machinery skips them).
    from ._serialize import VarRef, NodeOccupancy
    per_child_sites: list[list] = []
    per_child_tags: list[list] = []
    child_offsets: list[int] = []
    sketch_structural_data: dict | None = None
    running = 0
    for ci, child in enumerate(bundle.children):
        if ci == 0 and sketch_is_structural:
            seg = _structural_segment_data(child, n_nodes_max)
            sketch_structural_data = seg
            n_nodes = seg["n_total"]
            # Build a sites/tags list of length n_nodes for the segment.
            # Concrete sub_idx -> its expanded slot carries the original
            # NodeOccupancy (binder_site offset still segment-local = 0 here,
            # since this is child 0). Hole slots get PAD placeholders.
            sites = [NodeOccupancy(kind=_ENC_KIND_PAD) for _ in range(n_nodes)]
            tags = [0 for _ in range(n_nodes)]
            for sub_idx, new_idx in seg["expanded_slot_of_sub"].items():
                occ = seg["sub_sites"][sub_idx]
                # Prefix ast_path with child_index for global uniqueness.
                occ.ast_path = (ci,) + occ.ast_path
                sites[new_idx] = occ
                tags[new_idx] = seg["sub_type_tags"][sub_idx]
        else:
            sites, tags, n_nodes = _serialize_child(child, n_nodes_max)
            # Rebase binder_site indices for VarRefs inside this child.
            for occ in sites:
                if occ.var_ref is not None:
                    vr = occ.var_ref
                    occ.var_ref = VarRef(
                        binder_site=vr.binder_site + running,
                        depth_from_innermost=vr.depth_from_innermost,
                        candidates=[(bs + running, d) for bs, d in vr.candidates],
                    )
                # Prefix ast_path with child_index for global uniqueness.
                occ.ast_path = (ci,) + occ.ast_path
        per_child_sites.append(sites)
        per_child_tags.append(tags)
        child_offsets.append(running)
        running += n_nodes

    n_total = running
    if n_total < 1:
        raise EncodingTooLarge(n_nodes=n_total, N=n_nodes_max)

    # If the sketch is structural, rebase its concrete-node var_refs into
    # the unified pre-order using child0's expanded slots, then for ci>=1
    # children, additionally rebase using child_offsets[ci].
    if sketch_is_structural:
        # child 0 binder_sites are sub_idx (in sub_sites coordinates).
        # Map them to expanded segment-local slots.
        seg = sketch_structural_data
        eslot = seg["expanded_slot_of_sub"]
        for new_idx in eslot.values():
            occ = per_child_sites[0][new_idx]
            if occ.var_ref is not None:
                vr = occ.var_ref
                bs = eslot.get(vr.binder_site, vr.binder_site)
                occ.var_ref = VarRef(
                    binder_site=bs,
                    depth_from_innermost=vr.depth_from_innermost,
                    candidates=[(eslot.get(b, b), d) for b, d in vr.candidates],
                )
        # Children 1+ var_refs need offsetting by child_offsets[ci].
        for ci in range(1, len(bundle.children)):
            off = child_offsets[ci]
            for occ in per_child_sites[ci]:
                if occ.var_ref is not None:
                    vr = occ.var_ref
                    occ.var_ref = VarRef(
                        binder_site=vr.binder_site + off,
                        depth_from_innermost=vr.depth_from_innermost,
                        candidates=[(bs + off, d) for bs, d in vr.candidates],
                    )

    # 2) Concatenate.
    sites_all: list = []
    type_tags_all: list = []
    for s, t in zip(per_child_sites, per_child_tags):
        sites_all.extend(s)
        type_tags_all.extend(t)

    layout = compute_layout(n_total)

    # 3) Leaf vectors (concrete + PAD padding). For structural-sketch case,
    #    hole-region slots are left as PAD here; per-branch overrides will
    #    fill them.
    # For structural-sketch bundles, child 0's segment extent is
    # [0, child_offsets[1]) (when there is at least one witness) or
    # [0, n_total) (when only the sketch is present). Within that range,
    # PAD-marked occupancies are hole-region placeholders; they get PAD
    # leaves here and are overridden per-branch below.
    if sketch_is_structural:
        sketch_seg_end = child_offsets[1] if len(child_offsets) > 1 else n_total
    else:
        sketch_seg_end = 0
    leaf_vectors: list[np.ndarray] = []
    for node_idx in range(n_total):
        occ = sites_all[node_idx]
        if (sketch_is_structural and node_idx < sketch_seg_end
                and occ.kind == _ENC_KIND_PAD):
            five = [_pad_leaf_vector() for _ in range(LEAVES_PER_NODE)]
        else:
            five = node_leaf_vectors(occ, type_tags_all[node_idx])
        leaf_vectors.extend(five)
    while len(leaf_vectors) < layout.n_leaves:
        leaf_vectors.append(_pad_leaf_vector())

    # 4) State assembly.
    if sketch_is_structural:
        # Promote regions to absolute (segment-local already == absolute for
        # child 0 since it starts at offset 0).
        seg = sketch_structural_data
        regions_in = seg["regions_in"]
        hole_scopes = seg["hole_scopes"]
        region_node_starts = seg["region_node_starts"]
        hole_regions_out: list = []
        for r, region_in in enumerate(regions_in):
            from .mera_synthesis.encode_ext import HoleRegion
            hole_regions_out.append(HoleRegion(
                node_start=region_node_starts[r],
                n_max=region_in.n_max,
                candidate_branches=region_in.candidate_branches,
            ))
        # Per-region per-branch overrides.
        per_region_branches: list[list[dict[int, np.ndarray]]] = []
        for r, region in enumerate(hole_regions_out):
            branches = structural_hole_branches(
                region, hole_scopes[r], layout)
            per_region_branches.append(branches)
        # Cartesian product across structural holes.
        branch_choices: list[list[int]] = [[]]
        for branches in per_region_branches:
            branch_choices = [c + [j] for c in branch_choices
                              for j in range(len(branches))]
        K = len(branch_choices)
        amp = 1.0 / np.sqrt(K)
        terms: list[tuple[complex, list[np.ndarray]]] = []
        for choice in branch_choices:
            leaves = [v.copy() for v in leaf_vectors]
            for r, j in enumerate(choice):
                for leaf_idx, vec in per_region_branches[r][j].items():
                    leaves[leaf_idx] = vec
            terms.append((complex(amp), leaves))
        state = MERA.from_term_superposition(terms, chi_layer=chi_layer)
    else:
        hole_nodes = _hole_nodes(sites_all)
        if hole_nodes:
            holes: list[dict] = []
            for node_idx in hole_nodes:
                var_ref = sites_all[node_idx].var_ref
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
            state = MERA.from_product(leaf_vectors, chi_layer=chi_layer)
    state.normalize()

    # 5) Meta bookkeeping.
    binder_kinds = _binder_kinds()
    binder_leaves: dict[int, int] = {}
    use_to_binder: dict[int, int] = {}
    for node_idx in range(n_total):
        occ = sites_all[node_idx]
        if occ.kind in binder_kinds:
            binder_leaves[node_idx] = layout.leaf_of(node_idx, "bid")
        if occ.var_ref is not None:
            use_leaf = layout.leaf_of(node_idx, "bid")
            use_to_binder[use_leaf] = layout.leaf_of(
                occ.var_ref.binder_site, "bid")

    nested_type_index: dict[int, object] = {}
    for node_idx in range(n_total):
        if (type_tags_all[node_idx] == TYPE_ARR_NESTED
                and sites_all[node_idx].ty is not None):
            nested_type_index[node_idx] = sites_all[node_idx].ty

    children_of_node: dict[int, list[int]] = {i: [] for i in range(n_total)}
    path_to_node: dict[tuple, int] = {}
    for node_idx in range(n_total):
        path_to_node[sites_all[node_idx].ast_path] = node_idx
    for node_idx in range(n_total):
        path = sites_all[node_idx].ast_path
        if len(path) >= 1:
            parent = path_to_node.get(path[:-1])
            if parent is not None and parent != node_idx:
                children_of_node[parent].append(node_idx)
    for k in children_of_node:
        children_of_node[k].sort()

    # Witness node ranges: children[1..]; each is a tuple of the node
    # indices that child contributed in the unified pre-order.
    witness_node_ranges: list[tuple[int, ...]] = []
    for ci in range(1, len(bundle.children)):
        start = child_offsets[ci]
        end = child_offsets[ci + 1] if ci + 1 < len(child_offsets) else n_total
        witness_node_ranges.append(tuple(range(start, end)))

    hole_regions_meta = (
        hole_regions_out if sketch_is_structural else []
    )
    meta = MeraEncodingMeta(
        n_nodes=n_total, n_leaves=layout.n_leaves, L=layout.L,
        leaf_dim=MERA_LEAF_DIM,
        species_of_leaf=layout.species_of_leaf,
        node_of_leaf=layout.node_of_leaf,
        site_to_ast_path={k: sites_all[k].ast_path for k in range(n_total)
                          if sites_all[k].kind != _ENC_KIND_PAD},
        binder_leaves=binder_leaves,
        use_to_binder=use_to_binder,
        nested_type_index=nested_type_index,
        layout=layout,
        children_of_node=children_of_node,
        n_nodes_max=n_nodes_max,
        witness_node_ranges=witness_node_ranges,
        hole_regions=hole_regions_meta,
    )
    return state, meta
