"""Synthesis problem validation (spec §6.1).

Pure structural checks raised as SynthesisProblemError. The runner calls
this before encoding so malformed problems fail fast with a domain error
instead of an encoder traceback.
"""
from __future__ import annotations

from .ast import (
    Node, HoleVar, TypeHole, Var, Lam, App, Bin, If, Forall, Fix,
    Cons, Succ, Eq,
)
from .mera_synthesis.errors import SynthesisProblemError


_CHILD_ATTRS = (
    "body", "fn", "arg", "lhs", "rhs", "cond", "then_b", "else_b",
    "head", "tail",
)


def _iter_children(n: Node):
    for attr in _CHILD_ATTRS:
        child = getattr(n, attr, None)
        if isinstance(child, Node):
            yield child


def _has_any_hole(n: Node) -> bool:
    if isinstance(n, HoleVar):
        return True
    # TypeHole appears in Ty positions, not Node positions; check Lam param_ty.
    if isinstance(n, (Lam, Forall, Fix)) and isinstance(n.param_ty, TypeHole):
        return True
    for c in _iter_children(n):
        if _has_any_hole(c):
            return True
    return False


def _check_scope(n: Node, scope: tuple) -> None:
    """Walk the AST checking that every Var (and structural HoleVar string
    candidate) is in scope. Binders extend scope for their body.
    """
    if isinstance(n, Var):
        if n.name not in scope:
            raise SynthesisProblemError(
                f"Var {n.name!r} not in scope (scope={list(scope)})")
        return
    if isinstance(n, HoleVar):
        kind = n.candidate_kind()
        if kind == "var":
            # var-hole: candidate strings must all be in scope (empty tuple
            # means "any in-scope binder" which is fine).
            for c in n.candidates:
                if not isinstance(c, str):
                    continue
                if c not in scope:
                    raise SynthesisProblemError(
                        f"HoleVar var-candidate {c!r} not in scope "
                        f"(scope={list(scope)})")
        else:
            # structural: each candidate sub-tree must be well-scoped in
            # the current scope.
            for c in n.candidates:
                if isinstance(c, Node):
                    _check_scope(c, scope)
        return
    if isinstance(n, (Lam, Forall, Fix)):
        new_scope = scope + (n.param,)
        # The body sees the new binder.
        body = getattr(n, "body", None)
        if isinstance(body, Node):
            _check_scope(body, new_scope)
        return
    for c in _iter_children(n):
        _check_scope(c, scope)


def validate_problem(problem) -> None:
    """Validate a SynthesisProblem (spec §6.1). Raises SynthesisProblemError.

    Checks:
      * at least one hole (HoleVar or TypeHole) in the sketch;
      * structural HoleVar candidates are well-scoped at their hole site;
      * non-hole Vars are bound by an enclosing Lam/Forall/Fix.
    """
    sketch = problem.sketch
    if sketch is None:
        raise SynthesisProblemError("problem.sketch is None")
    if not _has_any_hole(sketch):
        raise SynthesisProblemError(
            "sketch contains no holes; synthesis requires at least one "
            "HoleVar or TypeHole (spec §6.1)")
    _check_scope(sketch, scope=())
