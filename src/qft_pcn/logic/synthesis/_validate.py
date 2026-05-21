"""Validation of SynthesisProblem invariants (spec §3.1).

The runner calls validate_problem() before any encoding work; violations
are reported as SynthesisProblemError.
"""

from __future__ import annotations

from src.qft_pcn.logic.ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin, HoleVar, TypeHole,
)
from .errors import SynthesisProblemError


def _walk_collect_holes_and_check_scopes(
    node: Node,
    scope: tuple = (),
) -> tuple:
    """Recursively visit `node`; return (n_var_holes, n_type_holes).

    Raises SynthesisProblemError on a HoleVar whose candidate is not in scope.
    """
    nvh, nth = 0, 0
    if isinstance(node, HoleVar):
        nvh += 1
        for cand in node.candidates:
            if cand not in scope:
                raise SynthesisProblemError(
                    f"HoleVar candidate {cand!r} not in scope "
                    f"(in-scope binders: {tuple(scope)!r})"
                )
        if isinstance(node.target_type, TypeHole):
            nth += 1
    elif isinstance(node, Var):
        pass
    elif isinstance(node, (IntLit, BoolLit)):
        pass
    elif isinstance(node, Lam):
        if isinstance(node.param_ty, TypeHole):
            nth += 1
        child_scope = scope + (node.param,)
        a, b = _walk_collect_holes_and_check_scopes(node.body, child_scope)
        nvh += a
        nth += b
    elif isinstance(node, App):
        for child in (node.fn, node.arg):
            a, b = _walk_collect_holes_and_check_scopes(child, scope)
            nvh += a; nth += b
    elif isinstance(node, If):
        for child in (node.cond, node.then_b, node.else_b):
            a, b = _walk_collect_holes_and_check_scopes(child, scope)
            nvh += a; nth += b
    elif isinstance(node, Bin):
        for child in (node.lhs, node.rhs):
            a, b = _walk_collect_holes_and_check_scopes(child, scope)
            nvh += a; nth += b
    else:
        raise SynthesisProblemError(
            f"unknown AST node in sketch: {type(node).__name__}"
        )
    return nvh, nth


def _is_literal(node: Node) -> bool:
    return isinstance(node, (IntLit, BoolLit))


def validate_problem(problem) -> None:
    """Raise SynthesisProblemError if `problem` violates spec §3.1.

    Checks:
        - knobs (N, chi_max, n_samples, anneal_*) are positive.
        - sketch contains at least one HoleVar or TypeHole.
        - HoleVar candidates are in lexical scope at their position.
        - IOExample inputs and output are literals (IntLit / BoolLit).
    """
    if problem.N < 1:
        raise SynthesisProblemError(f"N must be >= 1, got {problem.N}")
    if problem.chi_max < 1:
        raise SynthesisProblemError(
            f"chi_max must be >= 1, got {problem.chi_max}")
    if problem.n_samples < 1:
        raise SynthesisProblemError(
            f"n_samples must be >= 1, got {problem.n_samples}")
    if problem.anneal_steps < 1:
        raise SynthesisProblemError(
            f"anneal_steps must be >= 1, got {problem.anneal_steps}")
    if problem.anneal_dt <= 0:
        raise SynthesisProblemError(
            f"anneal_dt must be > 0, got {problem.anneal_dt}")

    nvh, nth = _walk_collect_holes_and_check_scopes(problem.sketch, ())
    if nvh + nth == 0:
        raise SynthesisProblemError(
            "SynthesisProblem must contain at least one hole "
            "(HoleVar or TypeHole). Got a fully-concrete sketch."
        )

    for i, ex in enumerate(problem.examples):
        for j, inp in enumerate(ex.inputs):
            if not _is_literal(inp):
                raise SynthesisProblemError(
                    f"IOExample[{i}].inputs[{j}] is not a literal: "
                    f"{type(inp).__name__}"
                )
        if not _is_literal(ex.output):
            raise SynthesisProblemError(
                f"IOExample[{i}].output is not a literal: "
                f"{type(ex.output).__name__}"
            )
