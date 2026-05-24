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
    Succ, Cons, Eq, Ty, TypeHole,
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
    forall_protected_leaves: set[int] = field(default_factory=set)  # I-Task-10 blocker #5: leaves frozen during evolution to preserve universal quantification (Forall's own bid leaf + all 5 species leaves of every bound Var use)
    typehole_regions: list = field(default_factory=list)   # M3 P5: per-TypeHole entries: dict(lam_node, candidate_tags, affected_leaves)

    def project_to_leaves(self, lo: int, hi: int) -> "MeraEncodingMeta":
        """Return a sub-meta restricted to leaves ``[lo, hi)`` (D39).

        Used by :meth:`LemmaLibraryAdapter._save_consolidated` to build
        an encoding meta that matches the sub-piece's
        :class:`MeraTensorBundle` (``bundle.n_leaves = hi - lo``) instead
        of inheriting the parent's full-AST meta wholesale. Without this
        projection, ``lemma.encoding_meta.n_leaves`` exceeds
        ``bundle.n_leaves``: leaf-indexed downstream consumers
        (``decode_mera`` reading ``n_nodes`` and walking
        ``LEAVES_PER_NODE * n_nodes`` leaves; ``mine_subtrees`` iterating
        ``node_of_leaf``) IndexError or rebuild a phantom AST.

        Invariants:
        * ``hi - lo`` is a power of two (sub-piece blocks are
          ``2^d``-wide per ``_node_aligned_intervals``; the
          ``node_of_leaf`` array can carry ``-1`` padding for leaves
          not assigned to a real AST node, so widths need not be a
          multiple of ``LEAVES_PER_NODE``).
        * Returned ``n_leaves = hi - lo``; ``n_nodes`` is the count of
          DISTINCT non-padding node ids in
          ``node_of_leaf[lo:hi]`` (matches the §4.4 "touched" node set
          that the miner uses).
        * ``site_to_ast_path`` keys (which are leaf indices in this
          codebase) are re-indexed to ``key - lo``; entries outside
          ``[lo, hi)`` are dropped.
        * ``binder_leaves`` / ``use_to_binder`` keep only entries whose
          *values* (and ``use_to_binder`` keys) lie in ``[lo, hi)``,
          re-indexed by ``-lo``. Binders whose body extends outside the
          sub-piece are dropped — the sub-piece is a self-contained
          density on the projected leaves.
        * ``forall_protected_leaves`` is restricted and re-indexed.

        Note: ``layout``, ``children_of_node``, ``nested_type_index``,
        ``hole_regions``, ``witness_node_ranges``, ``typehole_regions``
        are NOT projected — they reference the parent's node graph and
        have no meaningful sub-piece restriction. They are zeroed/empty
        on the projected meta. Downstream consumers that depend on these
        for sub-pieces will need a richer projection; the §10.9
        consolidation path uses only the leaf-aligned fields.
        """
        if lo < 0 or hi < lo:
            raise ValueError(
                f"project_to_leaves: bad interval ({lo}, {hi})")
        if hi > self.n_leaves:
            raise ValueError(
                f"project_to_leaves: hi={hi} exceeds parent n_leaves="
                f"{self.n_leaves}")
        new_n_leaves = hi - lo
        if new_n_leaves <= 0 or (new_n_leaves & (new_n_leaves - 1)) != 0:
            raise ValueError(
                f"project_to_leaves: interval width {new_n_leaves} must "
                f"be a positive power of two (sub-piece blocks are "
                f"2^d-wide per spec §4.4)")
        new_species = list(self.species_of_leaf[lo:hi])
        # node_of_leaf for the sub-piece: re-index touched nodes to a
        # contiguous 0..k-1 range, preserving the relative order of their
        # first appearance in [lo, hi). Padding leaves (-1) stay -1.
        parent_nol = self.node_of_leaf
        touched_in_order: list[int] = []
        seen: set[int] = set()
        for i in range(lo, hi):
            nd = parent_nol[i] if i < len(parent_nol) else -1
            if nd != -1 and nd not in seen:
                seen.add(nd)
                touched_in_order.append(nd)
        node_remap = {nd: k for k, nd in enumerate(touched_in_order)}
        new_n_nodes = len(touched_in_order)
        new_node_of_leaf = []
        for i in range(lo, hi):
            nd = parent_nol[i] if i < len(parent_nol) else -1
            new_node_of_leaf.append(node_remap[nd] if nd != -1 else -1)
        # site_to_ast_path: keys are leaf indices.
        new_site_to_ast = {
            k - lo: v for k, v in self.site_to_ast_path.items()
            if lo <= k < hi
        }
        # binder_leaves: AST-node -> leaf. Keep entries whose leaf is
        # in range; AST-node keys are preserved (they identify the parent
        # AST node, not a leaf), values are re-indexed.
        new_binder_leaves = {
            ast_node: leaf - lo
            for ast_node, leaf in self.binder_leaves.items()
            if lo <= leaf < hi
        }
        # use_to_binder: leaf -> leaf. Keep entries whose BOTH leaves are
        # in range (a use whose binder is outside the sub-piece is dangling
        # in the projection — drop it; the sub-piece density does not
        # carry that variable binding).
        new_use_to_binder = {
            use - lo: binder - lo
            for use, binder in self.use_to_binder.items()
            if lo <= use < hi and lo <= binder < hi
        }
        new_protected = {
            leaf - lo for leaf in self.forall_protected_leaves
            if lo <= leaf < hi
        }
        return MeraEncodingMeta(
            n_nodes=new_n_nodes,
            n_leaves=new_n_leaves,
            L=self.L,
            leaf_dim=self.leaf_dim,
            species_of_leaf=new_species,
            node_of_leaf=new_node_of_leaf,
            site_to_ast_path=new_site_to_ast,
            binder_leaves=new_binder_leaves,
            use_to_binder=new_use_to_binder,
            nested_type_index={},
            layout=None,
            children_of_node={},
            n_nodes_max=0,
            hole_regions=[],
            witness_node_ranges=[],
            forall_protected_leaves=new_protected,
            typehole_regions=[],
        )


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


def _has_type_hole(ast: Node) -> bool:
    """True if any Lam/Forall/Fix.param_ty (or nested TArrow) is a TypeHole."""

    def _ty_has_hole(ty) -> bool:
        if isinstance(ty, TypeHole):
            return True
        from .ast import TArrow
        if isinstance(ty, TArrow):
            return _ty_has_hole(ty.src) or _ty_has_hole(ty.dst)
        return False

    found = [False]

    def _walk(n: Node) -> None:
        if found[0]:
            return
        if isinstance(n, (Lam, Forall, Fix)):
            if _ty_has_hole(n.param_ty):
                found[0] = True
                return
        for attr in ("body", "fn", "arg", "lhs", "rhs", "cond",
                     "then_b", "else_b", "head", "tail"):
            child = getattr(n, attr, None)
            if isinstance(child, Node):
                _walk(child)
    _walk(ast)
    return found[0]


def _substitute_one_typehole(ty, replacement: Ty):
    """Replace the FIRST TypeHole found in ty (DFS) with ``replacement``.

    Returns ``(new_ty, replaced)``. ``replaced`` is True iff a hole was
    found and substituted. Used to enumerate per-Lam typehole candidates
    one binder at a time.
    """
    from .ast import TArrow
    if isinstance(ty, TypeHole):
        return replacement, True
    if isinstance(ty, TArrow):
        new_src, ok = _substitute_one_typehole(ty.src, replacement)
        if ok:
            return TArrow(src=new_src, dst=ty.dst), True
        new_dst, ok = _substitute_one_typehole(ty.dst, replacement)
        if ok:
            return TArrow(src=ty.src, dst=new_dst), True
    return ty, False


def _substitute_typehole_at(ast: Node, target_id: int,
                             candidate: Ty) -> Node:
    """Return a copy of ``ast`` in which the Lam/Forall/Fix whose
    ``id() == target_id`` has its first TypeHole replaced by
    ``candidate``. Non-target binders are left untouched.
    """
    def go(n: Node) -> Node:
        if isinstance(n, Lam):
            new_ty = n.param_ty
            if id(n) == target_id:
                new_ty, _ = _substitute_one_typehole(new_ty, candidate)
            return Lam(param=n.param, param_ty=new_ty, body=go(n.body))
        if isinstance(n, Forall):
            new_ty = n.param_ty
            if id(n) == target_id:
                new_ty, _ = _substitute_one_typehole(new_ty, candidate)
            return Forall(param=n.param, param_ty=new_ty, body=go(n.body))
        if isinstance(n, Fix):
            new_ty = n.param_ty
            if id(n) == target_id:
                new_ty, _ = _substitute_one_typehole(new_ty, candidate)
            return Fix(param=n.param, param_ty=new_ty, body=go(n.body))
        if isinstance(n, App):
            return App(fn=go(n.fn), arg=go(n.arg))
        if isinstance(n, If):
            return If(cond=go(n.cond), then_b=go(n.then_b),
                      else_b=go(n.else_b))
        if isinstance(n, Bin):
            return Bin(op=n.op, lhs=go(n.lhs), rhs=go(n.rhs))
        if isinstance(n, Succ):
            return Succ(arg=go(n.arg))
        if isinstance(n, Cons):
            return Cons(head=go(n.head), tail=go(n.tail))
        if isinstance(n, Eq):
            return Eq(lhs=go(n.lhs), rhs=go(n.rhs))
        return n
    return go(ast)


def _collect_type_hole_binders(ast: Node) -> list[tuple[int, TypeHole]]:
    """Pre-order walk; return one (id(binder_node), TypeHole) entry for
    each Lam/Forall/Fix whose param_ty contains a TypeHole.

    Only the OUTER TypeHole is returned per binder; nested TypeHoles
    inside the same param_ty are handled by per-candidate recursion
    when the encoder substitutes them.
    """
    from .ast import TArrow
    out: list[tuple[int, TypeHole]] = []

    def first_hole(ty) -> "TypeHole | None":
        if isinstance(ty, TypeHole):
            return ty
        if isinstance(ty, TArrow):
            h = first_hole(ty.src)
            if h is not None:
                return h
            return first_hole(ty.dst)
        return None

    def go(n: Node) -> None:
        if isinstance(n, (Lam, Forall, Fix)):
            h = first_hole(n.param_ty)
            if h is not None:
                out.append((id(n), h))
            go(n.body)
            return
        for attr in ("fn", "arg", "lhs", "rhs", "cond",
                     "then_b", "else_b", "head", "tail"):
            child = getattr(n, attr, None)
            if isinstance(child, Node):
                go(child)
    go(ast)
    return out


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

    # TypeHole path (M3 P5, spec §5.3): if any Lam/Forall/Fix.param_ty
    # carries a TypeHole, lift the affected type leaves to a genuine
    # equal-amplitude superposition over the candidate type tags via
    # MERA.from_term_superposition. Per §1.1, the candidates are real
    # tensor-network amplitudes on the type leaves, not a classical
    # iteration. (Composition with structural HoleVar is a follow-up;
    # P5 has TypeHole only, so this branch is correct for it.)
    if _has_type_hole(ast):
        return _encode_with_type_holes(
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
    # Extended (Forall/Fix non-flat param_ty, Nil/Cons non-Nat elem): also
    # record the binder's full param_ty / list-elem Ty so the decoder
    # round-trips `forall xs:List Bool` etc. rather than collapsing to TNat.
    from ._mera_leaves import nested_binder_ty as _nested_binder_ty
    nested_type_index: dict[int, object] = {}
    for node_idx in range(n_nodes):
        if type_tags[node_idx] == TYPE_ARR_NESTED and sites[node_idx].ty is not None:
            nested_type_index[node_idx] = sites[node_idx].ty
        else:
            extra = _nested_binder_ty(sites[node_idx])
            if extra is not None:
                nested_type_index[node_idx] = extra

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

    # Forall-protected leaves (I-Task-10 blocker #5): universal quantification
    # in the tensor-network substrate is genuine inertia, not classical
    # iteration. Each Forall binder's own bid leaf and every bound Var use's
    # 5 species leaves are flagged here; the synthesis driver forwards this
    # set as frozen_leaves= to imaginary-time evolution (spec §5.2a / §8.6
    # operator-algebraic restriction). Reductions like R-AddZero may read the
    # bound Var's leaf indices for promotion targets, but never write them.
    forall_protected_leaves = _collect_forall_protected_leaves(
        sites, n_nodes, layout)

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
        forall_protected_leaves=forall_protected_leaves,
    )
    return state, meta


def _collect_forall_protected_leaves(
        sites: list, n_nodes: int, layout: MeraLayout) -> set[int]:
    """Collect leaves that must remain bitwise unchanged through evolution
    so universal quantification is genuine tensor-network inertia
    (I-Task-10 blocker #5, §1.1 binding-as-entanglement).

    For each Forall binder site:
      - Add the Forall's own bid leaf.
      - For every Var site whose ``var_ref.binder_site`` points to that
        Forall, add all 5 species leaves (kind, type, bid, value, tobl).

    Encodings without any Forall return an empty set, preserving prior
    behavior bitwise (default frozen_leaves=None at the evolution layer).
    """
    protected: set[int] = set()
    forall_sites: set[int] = {
        i for i in range(n_nodes) if sites[i].kind == KIND_FORALL
    }
    if not forall_sites:
        return protected
    # Forall binder's own bid leaf.
    for fs in forall_sites:
        protected.add(layout.leaf_of(fs, "bid"))
    # All 5 species leaves of every bound Var use.
    for i in range(n_nodes):
        occ = sites[i]
        if occ.var_ref is None:
            continue
        if occ.var_ref.binder_site in forall_sites:
            for species in ("kind", "type", "bid", "value", "tobl"):
                protected.add(layout.leaf_of(i, species))
    return protected


# --------------------------------------------------------------------------
# M3 P5: TypeHole encoding (spec §5.3)
# --------------------------------------------------------------------------


def _concrete_leaves_and_sites(
        ast: Node, n_nodes_max: int
) -> tuple[list[np.ndarray], list, list[int], int, MeraLayout]:
    """Run the concrete pipeline (serialize, types, tobl, leaves) on a
    TypeHole-free AST and return ``(leaf_vectors, sites, type_tags,
    n_nodes, layout)``. Layout is sized to ``n_nodes`` (TypeHole branches
    share the SAME layout since the AST structure is identical across
    candidate substitutions).
    """
    sites = serialize_preorder(ast, N=n_nodes_max)
    type_tags = compute_site_types(ast, sites)
    compute_tobl_tags(ast, sites)
    n_nodes = sum(1 for occ in sites if occ.kind != _ENC_KIND_PAD)
    layout = compute_layout(n_nodes)
    leaf_vectors: list[np.ndarray] = []
    for node_idx in range(n_nodes):
        five = node_leaf_vectors(sites[node_idx], type_tags[node_idx])
        leaf_vectors.extend(five)
    while len(leaf_vectors) < layout.n_leaves:
        leaf_vectors.append(_pad_leaf_vector())
    return leaf_vectors, sites, type_tags, n_nodes, layout


def _encode_with_type_holes(ast: Node, n_nodes_max: int,
                             chi_layer: int) -> tuple[MERA, MeraEncodingMeta]:
    """Encode an AST whose Lam/Forall/Fix.param_ty contains TypeHoles
    via an equal-amplitude superposition over the Cartesian product of
    per-binder candidate substitutions.

    For each combined candidate choice the AST is concretely substituted
    and run through the M1 pipeline; the resulting per-leaf vector lists
    are handed to ``MERA.from_term_superposition`` so the affected type
    leaves carry a genuine multi-tag superposition (spec §1.1, §5.3).
    """
    holes = _collect_type_hole_binders(ast)
    if not holes:
        # Shouldn't happen given the caller's guard, but be defensive.
        return encode_mera(ast, n_nodes_max=n_nodes_max,
                           chi_layer=chi_layer)

    # Cartesian product over per-binder candidates.
    candidate_lists: list[list[Ty]] = [list(h.candidates) for _, h in holes]
    # Rank-K guard: each TypeHole has <=4 flat candidates, so the
    # product is bounded in practice (P5: K=2). Respect the structural
    # encoder's chi_layer cap analogously.
    total_k = 1
    for cands in candidate_lists:
        total_k *= max(1, len(cands))
    if total_k > max(chi_layer, 4):
        raise EncodingTooLarge(n_nodes=0, N=n_nodes_max)

    # Enumerate combined choices (one index per binder).
    choices: list[list[int]] = [[]]
    for cands in candidate_lists:
        choices = [c + [j] for c in choices for j in range(len(cands))]

    # For each combined choice, substitute candidates into the AST and
    # run the concrete pipeline.
    base_leaves_per_choice: list[list[np.ndarray]] = []
    layout_ref: MeraLayout | None = None
    sites_ref: list | None = None
    type_tags_ref: list[int] | None = None
    n_nodes_ref: int | None = None
    for choice in choices:
        concretized = ast
        for hi, ((binder_id, _h), j) in enumerate(zip(holes, choice)):
            concretized = _substitute_typehole_at(
                concretized, binder_id, candidate_lists[hi][j])
        leaves, sites, type_tags, n_nodes, layout = (
            _concrete_leaves_and_sites(concretized, n_nodes_max))
        if layout_ref is None:
            layout_ref = layout
            sites_ref = sites
            type_tags_ref = type_tags
            n_nodes_ref = n_nodes
        else:
            if n_nodes != n_nodes_ref:
                raise RuntimeError(
                    "TypeHole substitution changed AST node count "
                    f"({n_nodes} vs {n_nodes_ref}); structural invariance "
                    "violated")
        base_leaves_per_choice.append(leaves)

    layout = layout_ref
    sites = sites_ref
    type_tags = type_tags_ref
    n_nodes = n_nodes_ref

    K = len(choices)
    amp = 1.0 / np.sqrt(K)
    terms: list[tuple[complex, list[np.ndarray]]] = [
        (complex(amp), leaves) for leaves in base_leaves_per_choice
    ]
    state = MERA.from_term_superposition(terms, chi_layer=chi_layer)
    state.normalize()

    # Meta bookkeeping using the FIRST candidate's sites (canonical).
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

    from ._mera_leaves import nested_binder_ty as _nested_binder_ty
    nested_type_index: dict[int, object] = {}
    for node_idx in range(n_nodes):
        if (type_tags[node_idx] == TYPE_ARR_NESTED
                and sites[node_idx].ty is not None):
            nested_type_index[node_idx] = sites[node_idx].ty
        else:
            extra = _nested_binder_ty(sites[node_idx])
            if extra is not None:
                nested_type_index[node_idx] = extra

    children_of_node: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    path_to_node: dict[tuple, int] = {}
    for node_idx in range(n_nodes):
        path_to_node[sites[node_idx].ast_path] = node_idx
    for node_idx in range(n_nodes):
        path = sites[node_idx].ast_path
        if len(path) >= 1:
            parent = path_to_node.get(path[:-1])
            if parent is not None and parent != node_idx:
                children_of_node[parent].append(node_idx)
    for k in children_of_node:
        children_of_node[k].sort()

    forall_protected_leaves = _collect_forall_protected_leaves(
        sites, n_nodes, layout)

    # typehole_regions: per-binder record with the binder's site index in
    # the canonical pre-order, the candidate tags, and the type leaf the
    # superposition primarily lives on (the Lam's `type` leaf — which
    # carries TArrow(candidate, body_ty) and varies across candidates).
    typehole_regions: list[dict] = []
    # Map id(binder_node) -> site index using the canonical (first-choice)
    # sites' binder_ref.
    id_to_site: dict[int, int] = {}
    for node_idx in range(n_nodes):
        br = sites[node_idx].binder_ref
        if br is not None:
            id_to_site[id(br.lam_node)] = node_idx
    # Note: id_to_site uses concretized-AST binder identities (per choice
    # 0). The ORIGINAL TypeHole-bearing AST has DIFFERENT id()s, so we
    # use the canonical preorder position via the index of each hole in
    # the binder-walk: the canonical AST's binders appear in the same
    # order (substitution preserves structure).
    canonical_binder_sites = [
        node_idx for node_idx in range(n_nodes)
        if sites[node_idx].binder_ref is not None
    ]
    # The original AST's binders appear in the same pre-order; find
    # which of them have TypeHoles to align with ``holes``.
    original_binder_id_order = _binder_id_preorder(ast)
    original_with_holes = [bid for bid, _h in holes]
    hole_canonical_sites: list[int] = []
    cursor_holes = 0
    for bid in original_binder_id_order:
        if cursor_holes >= len(original_with_holes):
            break
        if bid == original_with_holes[cursor_holes]:
            # This binder has a TypeHole; the matching canonical site is
            # the SAME-ordered binder in the substituted AST.
            # Position in original_binder_id_order = position in
            # canonical_binder_sites.
            position = original_binder_id_order.index(bid)
            hole_canonical_sites.append(canonical_binder_sites[position])
            cursor_holes += 1

    for (binder_id, hole), site_idx in zip(holes, hole_canonical_sites):
        cand_tags: list[int] = []
        for c in hole.candidates:
            from ._types import ty_to_tag as _t2t
            tag, _ = _t2t(c)
            cand_tags.append(tag)
        typehole_regions.append({
            "binder_node": site_idx,
            "type_leaf": layout.leaf_of(site_idx, "type"),
            "value_leaf": layout.leaf_of(site_idx, "value"),
            "candidate_tags": tuple(cand_tags),
        })

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
        forall_protected_leaves=forall_protected_leaves,
        typehole_regions=typehole_regions,
    )
    return state, meta


def _binder_id_preorder(ast: Node) -> list[int]:
    """Pre-order list of id() of every Lam/Forall/Fix encountered."""
    out: list[int] = []

    def go(n: Node) -> None:
        if isinstance(n, (Lam, Forall, Fix)):
            out.append(id(n))
            go(n.body)
            return
        for attr in ("fn", "arg", "lhs", "rhs", "cond",
                     "then_b", "else_b", "head", "tail"):
            child = getattr(n, attr, None)
            if isinstance(child, Node):
                go(child)
    go(ast)
    return out


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

    from ._mera_leaves import nested_binder_ty as _nested_binder_ty
    nested_type_index: dict[int, object] = {}
    for sub_idx in range(n_sub):
        if sub_idx in hole_set:
            continue
        new_idx = expanded_slot_of_sub[sub_idx]
        if (sub_type_tags[sub_idx] == TYPE_ARR_NESTED
                and sub_sites[sub_idx].ty is not None):
            nested_type_index[new_idx] = sub_sites[sub_idx].ty
        else:
            extra = _nested_binder_ty(sub_sites[sub_idx])
            if extra is not None:
                nested_type_index[new_idx] = extra

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

    # Forall-protected leaves (I-Task-10 blocker #5): same rule as the
    # concrete path, walking the structural-sketch's substituted sites by
    # their expanded-slot mapping. Hole-region slots themselves cannot
    # introduce Forall binders (HoleVar->IntLit substitution erases them
    # from candidates only, never the surrounding sketch).
    forall_protected_leaves: set[int] = set()
    forall_sites_str: set[int] = set()
    for sub_idx in range(n_sub):
        if sub_idx in hole_set:
            continue
        new_idx = expanded_slot_of_sub[sub_idx]
        if sub_sites[sub_idx].kind == KIND_FORALL:
            forall_sites_str.add(new_idx)
            forall_protected_leaves.add(layout.leaf_of(new_idx, "bid"))
    if forall_sites_str:
        for sub_idx in range(n_sub):
            if sub_idx in hole_set:
                continue
            new_idx = expanded_slot_of_sub[sub_idx]
            occ = sub_sites[sub_idx]
            if occ.var_ref is None:
                continue
            binder_new = expanded_slot_of_sub.get(occ.var_ref.binder_site)
            if binder_new in forall_sites_str:
                for species in ("kind", "type", "bid", "value", "tobl"):
                    forall_protected_leaves.add(
                        layout.leaf_of(new_idx, species))

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
        forall_protected_leaves=forall_protected_leaves,
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
        # Children 1+ var_refs were ALREADY rebased by `running` (== the same
        # value as child_offsets[ci] at that iteration) in the first pass
        # above (lines ~1077-1086). A second rebase here would double-offset
        # witness binder_sites past meta.n_nodes (e.g. P4: witness Var x had
        # binder_site=0 -> first pass -> 9 (== child_offsets[1]) -> second
        # pass -> 18, while meta.n_nodes=17, producing IndexError in
        # downstream leaf addressing). Drop the redundant second pass.
        # Child 0's structural-segment rebase (sub_idx -> expanded slot) is
        # handled separately above; this block was only ever needed for ci>=1,
        # and the first pass already covers that case.
        # (Bug fix: d580815 review — durable replacement for the localized
        # `_binder_node_of` >= n_nodes filter in mera_typing_hamiltonian.)

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

    from ._mera_leaves import nested_binder_ty as _nested_binder_ty
    nested_type_index: dict[int, object] = {}
    for node_idx in range(n_total):
        if (type_tags_all[node_idx] == TYPE_ARR_NESTED
                and sites_all[node_idx].ty is not None):
            nested_type_index[node_idx] = sites_all[node_idx].ty
        else:
            extra = _nested_binder_ty(sites_all[node_idx])
            if extra is not None:
                nested_type_index[node_idx] = extra

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

    # Forall-protected leaves (I-Task-10 blocker #5): scan the concatenated
    # sites for KIND_FORALL binders; freeze the Forall's bid leaf + the 5
    # species leaves of every Var whose var_ref.binder_site points at one
    # (binder_site indices are already rebased into the unified pre-order).
    forall_protected_leaves: set[int] = set()
    forall_sites_b: set[int] = {
        i for i in range(n_total)
        if sites_all[i].kind == KIND_FORALL
    }
    if forall_sites_b:
        for fs in forall_sites_b:
            forall_protected_leaves.add(layout.leaf_of(fs, "bid"))
        for i in range(n_total):
            occ = sites_all[i]
            if occ.var_ref is None:
                continue
            if occ.var_ref.binder_site in forall_sites_b:
                for species in ("kind", "type", "bid", "value", "tobl"):
                    forall_protected_leaves.add(layout.leaf_of(i, species))

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
        forall_protected_leaves=forall_protected_leaves,
    )
    return state, meta
