"""Lexical scope resolution: every Var is paired with its binder.

Walks the AST tracking a stack of in-scope binders. For each Var encountered,
computes:
  - the binding Lam node (via reference)
  - the lexical depth of the binder from the use site (0 = innermost)

Raises IllScopedVar for unbound references and TooManyBinders if scope
nesting exceeds MAX_BINDER_DEPTH.

HoleVar nodes resolve every candidate name, yielding a list of ResolvedRef.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .ast import (Node, Var, Lam, App, IntLit, BoolLit, If, Bin, HoleVar,
                  Zero, Succ, NatLit, Nil, Cons, Eq)
from .encoding import (IllScopedVar, TooManyBinders, UnsupportedNode,
                       MAX_BINDER_DEPTH)


@dataclass
class ResolvedRef:
    """Annotation attached to a Var by the resolver."""
    binder: Lam                       # the binding Lam node
    depth_from_innermost: int         # 0 = use is directly inside its binder


def resolve_binders(
    root: Node,
    on_var: Callable[["Node", list[ResolvedRef]], None],
) -> None:
    """Walk root pre-order. Call on_var(node, refs) for each Var/HoleVar.

    For a plain Var, refs is a singleton list. For HoleVar(candidates=[...]),
    refs is one ResolvedRef per candidate.

    The walk also maintains a stack of in-scope binders; visiting a Lam
    pushes it before descending into the body and pops on the way out.
    """
    stack: list[Lam] = []

    def _walk(node: Node) -> None:
        if isinstance(node, Var):
            for offset, lam in enumerate(reversed(stack)):
                if lam.param == node.name:
                    on_var(node, [ResolvedRef(binder=lam,
                                              depth_from_innermost=offset)])
                    return
            raise IllScopedVar(name=node.name)
        if isinstance(node, HoleVar):
            refs: list[ResolvedRef] = []
            for cand in node.candidates:
                found = False
                for offset, lam in enumerate(reversed(stack)):
                    if lam.param == cand:
                        refs.append(ResolvedRef(
                            binder=lam, depth_from_innermost=offset))
                        found = True
                        break
                if not found:
                    raise IllScopedVar(name=cand)
            on_var(node, refs)
            return
        if isinstance(node, Lam):
            if len(stack) >= MAX_BINDER_DEPTH:
                raise TooManyBinders(depth=len(stack) + 1,
                                     cutoff=MAX_BINDER_DEPTH)
            stack.append(node)
            _walk(node.body)
            stack.pop()
            return
        if isinstance(node, App):
            _walk(node.fn)
            _walk(node.arg)
            return
        if isinstance(node, If):
            _walk(node.cond)
            _walk(node.then_b)
            _walk(node.else_b)
            return
        if isinstance(node, Bin):
            _walk(node.lhs)
            _walk(node.rhs)
            return
        if isinstance(node, (IntLit, BoolLit, Zero, NatLit, Nil)):
            return
        # --- extended-calculus structural nodes ---
        if isinstance(node, Succ):
            _walk(node.arg)
            return
        if isinstance(node, Cons):
            _walk(node.head)
            _walk(node.tail)
            return
        if isinstance(node, Eq):
            _walk(node.lhs)
            _walk(node.rhs)
            return
        # Defensive: only the supported node types reach here.
        raise UnsupportedNode(node_type=type(node).__name__)

    _walk(root)
