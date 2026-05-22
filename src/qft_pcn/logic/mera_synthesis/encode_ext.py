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

from ..ast import Node, HoleVar


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
