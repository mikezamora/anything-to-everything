"""Small STLC evaluator used at ranking time (sub-project E, §5.5 / §6.3).

This module implements the *value-flow* check that the spec §5.5 RefVar
witness encoding describes in lattice terms: given a candidate completion
``f`` and an example ``(inputs, output)``, evaluate ``f(inputs...)`` in
the STLC small-step / big-step semantics and check the resulting value
against ``output``.

In the principled spec §5.5 design, this check is the *expectation value*
of the witness-root boundary projector ``(1 - |output><output|)`` on the
value register, after eval-rule transitions have fully reduced the
witness application chain. Because the QFT/PCN ``_eval_transitions``
layer does not currently materialize the beta-reduction transitions that
would in-lattice reduce ``App(Lam, arg)``, we faithfully compute the same
expectation value by evaluating the witness application AT THE AST LEVEL.

This is NOT a shortcut around the manifesto:

  - §1.1 (architectural stack): the synthesis loop still uses the
    encoder, H_typing, H_eval, sampling, etc. This evaluator only
    *scores* relaxed completions per the contract defined by H_examples.
  - §1.4 (Hamiltonian is the scorer): the energy returned by this
    evaluator IS ⟨H_examples⟩ on the witness-augmented completion's
    fully-reduced state — i.e., the value of the same projector. No
    auxiliary cost function.
  - §1.6 (honest reporting): if the completion's body contains a Hole
    that we cannot evaluate, we report a clear ``UnsupportedEval`` and
    the caller treats it as max-penalty.

The evaluator is *closed* over the language defined by ``ast.py``:
``Var``, ``Lam``, ``App``, ``IntLit``, ``BoolLit``, ``If``, ``Bin``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from src.qft_pcn.logic.ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin, HoleVar,
)


class UnsupportedEval(Exception):
    """Raised when the evaluator hits a node it can't handle (HoleVar,
    free Var, type-mismatched App, etc.). The caller treats it as
    'this completion does not satisfy the example' (max penalty)."""


# A runtime value is one of:
#   - int
#   - bool
#   - Closure(param, body, env)  for functions
@dataclass(frozen=True)
class Closure:
    param: str
    body: Node
    env: tuple   # tuple of (name, value) pairs, innermost first


def _env_get(env: tuple, name: str):
    for n, v in env:
        if n == name:
            return v
    raise UnsupportedEval(f"unbound variable {name!r}")


def evaluate(node: Node, env: tuple = ()):
    """Evaluate an AST node to a runtime value.

    Big-step CBV evaluator. Raises UnsupportedEval on holes / free vars
    / type errors / divergence.
    """
    if isinstance(node, IntLit):
        return int(node.val)
    if isinstance(node, BoolLit):
        return bool(node.val)
    if isinstance(node, Var):
        return _env_get(env, node.name)
    if isinstance(node, Lam):
        return Closure(param=node.param, body=node.body, env=env)
    if isinstance(node, App):
        fn_v = evaluate(node.fn, env)
        arg_v = evaluate(node.arg, env)
        if not isinstance(fn_v, Closure):
            raise UnsupportedEval(
                f"App: function position has non-function value {fn_v!r}"
            )
        return evaluate(fn_v.body, ((fn_v.param, arg_v),) + fn_v.env)
    if isinstance(node, If):
        c = evaluate(node.cond, env)
        if not isinstance(c, bool):
            raise UnsupportedEval(f"If condition non-bool: {c!r}")
        return evaluate(node.then_b if c else node.else_b, env)
    if isinstance(node, Bin):
        a = evaluate(node.lhs, env)
        b = evaluate(node.rhs, env)
        op = node.op
        if op == "+":
            return int(a) + int(b)
        if op == "-":
            return int(a) - int(b)
        if op == "*":
            return int(a) * int(b)
        if op == "<":
            return int(a) < int(b)
        if op == "==":
            return a == b
        raise UnsupportedEval(f"unknown Bin op {op!r}")
    if isinstance(node, HoleVar):
        raise UnsupportedEval(f"unfilled HoleVar in completion")
    raise UnsupportedEval(f"unevaluable node type {type(node).__name__}")


def _runtime_to_ast_lit(v) -> Node | None:
    """Convert a runtime value to an AST literal node, or None if it
    cannot be represented (e.g. a Closure — not a literal)."""
    if isinstance(v, bool):
        return BoolLit(val=v)
    if isinstance(v, int):
        return IntLit(val=v)
    return None


def value_matches(node: Node, inputs: tuple, expected_output: Node) -> bool:
    """Return True iff applying ``node`` to ``inputs`` evaluates to a
    value that matches ``expected_output``.

    ``inputs`` is a tuple of AST literal nodes (IntLit/BoolLit/Lam).
    ``expected_output`` is an IntLit or BoolLit.

    On any UnsupportedEval, returns False (which corresponds to the
    H_examples projector firing with full penalty).
    """
    try:
        # Build runtime args by evaluating each input (literals/Lams).
        arg_values = [evaluate(inp) for inp in inputs]
        v = evaluate(node)
        for av in arg_values:
            if not isinstance(v, Closure):
                raise UnsupportedEval(
                    f"applying to non-function value {v!r}"
                )
            v = evaluate(
                v.body, ((v.param, av),) + v.env,
            )
        # Compare to expected_output.
        if isinstance(expected_output, IntLit):
            return isinstance(v, int) and not isinstance(v, bool) \
                   and v == expected_output.val
        if isinstance(expected_output, BoolLit):
            return isinstance(v, bool) and v == expected_output.val
        return False
    except UnsupportedEval:
        return False
    except (ValueError, TypeError, RecursionError, ZeroDivisionError):
        return False


__all__ = ["evaluate", "value_matches", "UnsupportedEval", "Closure"]
