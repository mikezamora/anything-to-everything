"""Structural-hole encoder extensions (spec §5.6).

_expand_structural_holes computes, per structural HoleVar, the hole
region's pre-order position and n_max (max candidate node count), plus
the per-branch leaf-assignment specs. Pure computation; no tensors.

The pre-order walk's child-attribute order mirrors `_serialize.py`'s
canonical walk (Lam.body, App.fn/arg, Bin.lhs/rhs, If.cond/then_b/else_b,
Cons.head/tail, Succ.arg, Eq.lhs/rhs, Forall/Fix.body). Flattened across
node kinds the visitation order is:
    body, fn, arg, lhs, rhs, cond, then_b, else_b, head, tail
which matches the source-of-truth serializer exactly (each node kind
only has one of these subsets and the within-kind order matches).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ast import Node, HoleVar, Var, Lam, App, Forall, Fix, IntLit


# Child attribute names to walk, in the canonical pre-order matching
# src/qft_pcn/logic/_serialize.py. Listing them all in one tuple is safe
# because no AST node defines more than one disjoint subset of these
# attributes, and the within-kind order matches the serializer.
_CHILD_ATTRS = (
    "body",     # Lam / Forall / Fix
    "fn",       # App
    "arg",      # App / Succ
    "lhs",      # Bin / Eq
    "rhs",      # Bin / Eq
    "cond",     # If
    "then_b",   # If
    "else_b",   # If
    "head",     # Cons
    "tail",     # Cons
)


@dataclass
class HoleRegion:
    """One structural hole's leaf region (spec §5.2).

    node_start:         pre-order node index where the hole region begins.
    n_max:              max candidate sub-tree node count; region spans
                        5*n_max leaves.
    candidate_branches: list of candidate sub-tree ASTs, one per branch
                        direction of the rank-k superposition. The encoder
                        (Task 4) turns each into a leaf assignment.
    """
    node_start: int
    n_max: int
    candidate_branches: list


def _count_nodes(ast: Node) -> int:
    """Total number of AST nodes (pre-order, no PAD) in `ast`.

    Matches the M1 serializer's node count — `len(serialize_preorder(ast))`
    minus PAD entries — which is what Task 4 uses to size each candidate's
    n_j (one node => five leaf slots in the structural-hole region).
    """
    n = 1
    for attr in _CHILD_ATTRS:
        child = getattr(ast, attr, None)
        if isinstance(child, Node):
            n += _count_nodes(child)
    return n


def _structural_holes_in_preorder(ast: Node):
    """Yield (preorder_index, HoleVar) for every structural hole."""
    counter = [0]

    def walk(n: Node):
        idx = counter[0]
        counter[0] += 1
        if isinstance(n, HoleVar) and n.candidate_kind() == "structural":
            yield idx, n
        for attr in _CHILD_ATTRS:
            child = getattr(n, attr, None)
            if isinstance(child, Node):
                yield from walk(child)
    yield from walk(ast)


def _expand_structural_holes(ast: Node):
    """Return (skeleton, hole_regions).

    `skeleton` is `ast` itself (the encoder consumes the regions to size
    the layout; no AST rewrite is needed at this stage). `hole_regions`
    is a list of HoleRegion, one per structural HoleVar.
    """
    regions: list[HoleRegion] = []
    for idx, hole in _structural_holes_in_preorder(ast):
        branches = list(hole.candidates)
        n_max = max((_count_nodes(c) for c in branches), default=1)
        regions.append(HoleRegion(
            node_start=idx,
            n_max=n_max,
            candidate_branches=branches,
        ))
    return ast, regions


@dataclass
class RefVar(Node):
    """Encoder-internal node: references a previously-encoded sub-tree
    (spec §4.2). Carries KIND_VAR semantics; its bid leaf is intended to
    share entanglement with the referenced sub-tree's bid leaf via the
    tree's binder channel — NOT a classical copy.

    For Task 7 the RefVar appears in witness sub-trees, where it stands
    in for the sketch's function position. The current Bundle encoder
    rewrites it to a Var bound to the sketch's outermost binder (spec
    §4.2: witnesses share the sketch through the tree); a follow-on task
    will replace this with the genuine bid-leaf entanglement gate.

    `target` is the AST node index of the referenced sub-tree root
    (0 = sketch root) — recorded so the witness-region bookkeeping in
    meta can reconstruct the share-relation.
    """
    target: int = 0


@dataclass
class Bundle(Node):
    """Encoder-internal container: a forest of K+1 children sitting on
    one MERA tree.

    children[0] is the sketch; children[1..K] are the witness sub-trees
    (one per IOExample). The encoder serializes each child independently
    in canonical pre-order, concatenates the resulting NodeOccupancy
    lists into one virtual pre-order, and emits a single MERA state
    sized by the total node count.

    M1's `_serialize.py` only walks a single rooted AST; rather than
    forking the serializer (and risking M1 surface drift), the encoder
    dispatches on Bundle and re-uses M1's per-child machinery — the
    Bundle node itself is never written into the leaf array.
    """
    children: tuple = ()


def _ref_var_to_sketch_var(node: Node, sketch_param: str | None) -> Node:
    """Rewrite RefVar -> Var(sketch_param) and recurse into the witness AST.

    The sketch param is the outermost Lam's `param` in the sketch. If the
    sketch root is not a Lam, RefVar becomes IntLit(0) (placeholder —
    Task 7's test gate is total_energy(state) returning a float, not the
    semantic share). The bid-leaf entanglement that spec §4.2 prescribes
    will be wired in a follow-on task on the same plan.
    """
    if isinstance(node, RefVar):
        if sketch_param is not None:
            return Var(name=sketch_param)
        return IntLit(val=0)
    if isinstance(node, Lam):
        return Lam(param=node.param, param_ty=node.param_ty,
                   body=_ref_var_to_sketch_var(node.body, sketch_param))
    if isinstance(node, App):
        return App(fn=_ref_var_to_sketch_var(node.fn, sketch_param),
                   arg=_ref_var_to_sketch_var(node.arg, sketch_param))
    if isinstance(node, Forall):
        return Forall(param=node.param, param_ty=node.param_ty,
                      body=_ref_var_to_sketch_var(node.body, sketch_param))
    if isinstance(node, Fix):
        return Fix(param=node.param, param_ty=node.param_ty,
                   body=_ref_var_to_sketch_var(node.body, sketch_param))
    return node


def _witness_augmented_ast(sketch: Node, examples: tuple) -> Node:
    """Return a Bundle whose children are the sketch followed by one
    witness sub-tree per example (spec §4.2).

    Witness i is `App(...App(RefVar(0), in_0)..., in_{n-1})`. RefVar
    references the sketch root — the witness's function position shares
    the sketch through the MERA tree (genuine entanglement, not a
    classical copy: spec §4.2, principle §1.1). For the current encoder
    that share is approximated by rewriting RefVar to a Var bound to
    the sketch's outermost lambda; the bid-leaf gate that gives it
    quantum semantics is a follow-on task on this plan.

    Container choice: a Bundle node (encoder-internal) — M1's serializer
    handles only single-rooted ASTs, and adding a forest-mode there
    would touch M1 surface unnecessarily. The Bundle dispatch lives in
    `mera_encoder.encode_mera`.
    """
    # The witness wraps the sketch in App-applications of the example
    # inputs. To make the witness encodable as a closed term, RefVar is
    # rewritten to a Var bound by the sketch's outermost Lam param.
    sketch_param: str | None = None
    if isinstance(sketch, Lam):
        sketch_param = sketch.param

    children: list[Node] = [sketch]
    for ex in examples:
        # Build witness: ((RefVar(0) in_0) in_1) ... in_{n-1}
        wit: Node = RefVar(target=0)
        for inp in ex.inputs:
            wit = App(fn=wit, arg=inp)
        # Rewrite RefVar inside the witness so the serializer can handle it.
        wit = _ref_var_to_sketch_var(wit, sketch_param)
        # If the rewrite turned RefVar into a Var referencing the sketch
        # param, wrap the whole witness in the sketch's outer Lam so the
        # Var has a binder. The bound body is the witness application.
        if (isinstance(sketch, Lam) and sketch_param is not None):
            wit = Lam(param=sketch.param, param_ty=sketch.param_ty,
                      body=wit)
        children.append(wit)
    return Bundle(children=tuple(children))
