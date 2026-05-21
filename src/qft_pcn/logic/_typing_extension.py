"""Encoder extension for sub-project B (typing Hamiltonian).

Computes, for each site of a pre-order-serialized AST, the parent-imposed
type obligation (`tobl_tag`). The Hamiltonian's T-Obligation rule then
projects, site-locally, onto the subspace where `type(k)` != `tobl(k)` when
`tobl(k)` is not TOBL_NONE.

See docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md, §5.1.

The obligation rule table:
  parent     | child position    | tobl(child_root)
  -----------|-------------------|-------------------------------
  (root)     | -                 | TOBL_NONE
  LAM        | body              | dst(LAM.type) as flat tag
  APP        | fn                | TOBL_NONE
  APP        | arg               | src(fn.type) as flat tag
  IF         | cond              | TOBL_BOOL
  IF         | then              | type(IF)
  IF         | else              | type(IF)
  BIN        | lhs               | TOBL_INT
  BIN        | rhs               | TOBL_INT
"""

from __future__ import annotations

from typing import Optional

from .ast import Node, Var, Lam, App, IntLit, BoolLit, If, Bin, HoleVar, Ty, TArrow
from ._serialize import NodeOccupancy
from ._types import ty_to_tag, _compute_ast_type
from .encoding import (
    KIND_PAD,
    TOBL_NONE, TOBL_INT, TOBL_BOOL, TOBL_ARR_NESTED,
    TYPE_ARR_NESTED,
)


def compute_tobl_tags(root: Node,
                      sites: list[NodeOccupancy]
                      ) -> list[int]:
    """For each site, compute the flat tobl tag.

    Returns a list of length N (= len(sites)). Sites with no parent
    obligation (root, PAD sites) get TOBL_NONE. The full Ty for a TOBL_ARR_NESTED
    obligation is stored on the corresponding NodeOccupancy.nested_tobl_ty
    attribute.
    """
    N = len(sites)
    tags: list[int] = [TOBL_NONE] * N

    # Walk the AST top-down with the parent-imposed obligation. Track the
    # site_iter cursor as we emit obligations in pre-order, matching
    # serialize_preorder's traversal exactly.

    site_iter = [0]   # mutable cursor over sites

    def _emit(node: Node, obligation: int,
              obligation_ty: Optional[Ty], env: list[tuple[str, Ty]]) -> None:
        idx = site_iter[0]
        tags[idx] = obligation
        sites[idx].tobl_tag = obligation
        if obligation == TOBL_ARR_NESTED and obligation_ty is not None:
            sites[idx].nested_tobl_ty = obligation_ty
        site_iter[0] += 1

        if isinstance(node, (Var, IntLit, BoolLit, HoleVar)):
            return

        if isinstance(node, Lam):
            # body's obligation = dst(LAM.type) = dst(TArrow(param_ty, body_type))
            #                   = body_type
            body_ty = _compute_ast_type(node.body,
                                        env + [(node.param, node.param_ty)])
            body_tag, body_nested = ty_to_tag(body_ty)
            _emit(node.body, body_tag,
                  body_nested if body_tag == TOBL_ARR_NESTED else None,
                  env + [(node.param, node.param_ty)])
            return

        if isinstance(node, App):
            # fn: no obligation imposed by App's local type (only an
            # arrow-dst constraint, handled by T-App-Arrow).
            _emit(node.fn, TOBL_NONE, None, env)
            # arg: obligation = src(fn.type).
            fn_ty = _compute_ast_type(node.fn, env)
            if isinstance(fn_ty, TArrow):
                src_tag, src_nested = ty_to_tag(fn_ty.src)
                _emit(node.arg, src_tag,
                      src_nested if src_tag == TOBL_ARR_NESTED else None, env)
            else:
                # ill-typed input: fn is not an arrow. We can't propagate
                # an obligation for arg in this case. Use TOBL_NONE; the
                # T-App-Arrow rule will fire at the parent APP regardless.
                _emit(node.arg, TOBL_NONE, None, env)
            return

        if isinstance(node, If):
            # cond: Bool.
            _emit(node.cond, TOBL_BOOL, None, env)
            # then/else: same as IF's overall type, which equals type(then).
            then_ty = _compute_ast_type(node.then_b, env)
            then_tag, then_nested = ty_to_tag(then_ty)
            _emit(node.then_b, then_tag,
                  then_nested if then_tag == TOBL_ARR_NESTED else None, env)
            _emit(node.else_b, then_tag,
                  then_nested if then_tag == TOBL_ARR_NESTED else None, env)
            return

        if isinstance(node, Bin):
            # Both operands: TOBL_INT (true for both arith and cmp ops).
            _emit(node.lhs, TOBL_INT, None, env)
            _emit(node.rhs, TOBL_INT, None, env)
            return

        raise TypeError(f"unsupported AST node in tobl walk: "
                        f"{type(node).__name__}")

    _emit(root, TOBL_NONE, None, [])
    # Remaining (PAD) sites: TOBL_NONE (already initialized).
    return tags
