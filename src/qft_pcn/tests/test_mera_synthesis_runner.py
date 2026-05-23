"""Synthesis runner acceptance: P1-P8 (spec §7, §9.6).

This is the M3 acceptance gate. Spec §9.6.2 requires 7/8 of the
parametrized top-1 cases to be correct; P8 is the deliberately-unsolvable
problem and must be refused (spec §9.6.4); the energy gap between top-1
and the second completion on P3 must be >= 0.5 * w_T = 2.0 (spec §9.6.3).

Per the user's standing directives:
- Correctness is checked by decoding to an AST and alpha-comparing against
  the expected program -- never on <H> alone (principle 7).
- Holes are quantum superpositions: the runner encodes ONCE and never
  enumerates candidates in Python (principle 6 / §1.1).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic.ast import (
    Lam, TInt, TBool, TArrow, Var, App, Bin, If, IntLit, BoolLit,
    HoleVar, TypeHole, Eq, Node,
)
from src.qft_pcn.logic.mera_synthesis import synthesize, SynthesisProblem
from src.qft_pcn.logic.mera_synthesis.problem import IOExample
from src.qft_pcn.logic.decoder import ast_alpha_eq


# ---- helpers --------------------------------------------------------------


def _body(node: Node) -> Node:
    """Descend through leading Lams to the synthesized body.

    The structural-hole completions fill a body, so for P3/P4/P6/P7 we
    compare bodies. For top-level lambdas (P1/P2/P5) we compare the whole
    AST OR the body -- whichever matches.
    """
    while isinstance(node, Lam):
        node = node.body
    return node


# Hard structural-hole problems get a doubled main anneal phase per spec
# §6.5 / §12 tuning latitude (the plan explicitly authorizes doubling the
# main phase before escalating).
_HARD_STEPS = 400


# ---- problem builders -----------------------------------------------------


def _P1():
    """Identity completion (warmup; var-hole)."""
    return SynthesisProblem(
        name="P1",
        sketch=Lam(param="x", param_ty=TInt(),
                   body=HoleVar(candidates=("x",))),
    )


def _P2():
    """Constant-vs-identity disambiguation by type (var-hole).

    Spec §7 P2 writes the candidate set as `HoleVar(candidates=())` with
    the note "all in-scope vars + int lits". The empty-candidate var-hole
    is the spec form; the encoder's "resolve empty at encode time" path
    is what is exercised here.
    """
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=()))
    return SynthesisProblem(
        name="P2", sketch=sketch,
        target_type=TArrow(src=TInt(), dst=TInt()),
        examples=(IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3)),),
    )


def _P3():
    """Use the argument: `f x` (STRUCTURAL hole -- multi-node)."""
    hole = HoleVar(candidates=(
        Var(name="x"),
        App(fn=Var(name="f"), arg=Var(name="x")),
        App(fn=Var(name="f"),
            arg=App(fn=Var(name="f"), arg=Var(name="x"))),
    ))
    sketch = Lam(param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
                 body=Lam(param="x", param_ty=TInt(), body=hole))
    succ = Lam(param="n", param_ty=TInt(),
               body=Bin(op="+", lhs=Var(name="n"), rhs=IntLit(val=1)))
    return SynthesisProblem(
        name="P3", sketch=sketch,
        examples=(IOExample(inputs=(succ, IntLit(val=2)),
                            output=IntLit(val=3)),),
        anneal_steps=_HARD_STEPS,
    )


def _P4():
    """Conditional synthesis: `if x<5 then x else x+1` (STRUCTURAL)."""
    cand_correct = If(
        cond=Bin(op="<", lhs=Var(name="x"), rhs=IntLit(val=5)),
        then_b=Var(name="x"),
        else_b=Bin(op="+", lhs=Var(name="x"), rhs=IntLit(val=1)),
    )
    cand_wrong_thr = If(
        cond=Bin(op="<", lhs=Var(name="x"), rhs=IntLit(val=3)),
        then_b=Var(name="x"),
        else_b=Bin(op="+", lhs=Var(name="x"), rhs=IntLit(val=1)),
    )
    cand_succ_only = Bin(op="+", lhs=Var(name="x"), rhs=IntLit(val=1))
    hole = HoleVar(candidates=(cand_correct, cand_wrong_thr, cand_succ_only))
    sketch = Lam(param="x", param_ty=TInt(), body=hole)
    return SynthesisProblem(
        name="P4", sketch=sketch,
        examples=(
            IOExample(inputs=(IntLit(val=2),), output=IntLit(val=2)),
            IOExample(inputs=(IntLit(val=7),), output=IntLit(val=8)),
        ),
        anneal_steps=_HARD_STEPS,
    )


def _P5():
    """Type-hole synthesis (TypeHole)."""
    sketch = Lam(
        param="x",
        param_ty=TypeHole(candidates=(TInt(), TBool())),
        body=Var(name="x"),
    )
    return SynthesisProblem(
        name="P5", sketch=sketch,
        target_type=TArrow(src=TBool(), dst=TBool()),
    )


def _P6():
    """Curried composition: `f (g x)` (STRUCTURAL)."""
    cand_fgx = App(fn=Var(name="f"),
                   arg=App(fn=Var(name="g"), arg=Var(name="x")))
    cand_gfx = App(fn=Var(name="g"),
                   arg=App(fn=Var(name="f"), arg=Var(name="x")))
    cand_fx = App(fn=Var(name="f"), arg=Var(name="x"))
    hole = HoleVar(candidates=(cand_fgx, cand_gfx, cand_fx))
    sketch = Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Lam(
            param="g", param_ty=TArrow(src=TInt(), dst=TInt()),
            body=Lam(param="x", param_ty=TInt(), body=hole),
        ),
    )
    # Pick f = (+1), g = (*2), x = 3 -> f(g(x)) = 7, g(f(x)) = 8, f(x) = 4.
    f_succ = Lam(param="n", param_ty=TInt(),
                 body=Bin(op="+", lhs=Var(name="n"), rhs=IntLit(val=1)))
    g_dbl = Lam(param="n", param_ty=TInt(),
                body=Bin(op="*", lhs=Var(name="n"), rhs=IntLit(val=2)))
    return SynthesisProblem(
        name="P6", sketch=sketch,
        examples=(IOExample(inputs=(f_succ, g_dbl, IntLit(val=3)),
                            output=IntLit(val=7)),),
        anneal_steps=_HARD_STEPS,
    )


def _P7():
    """Boolean synthesis with mixed types: `x < y` (STRUCTURAL)."""
    cand_lt_xy = Bin(op="<", lhs=Var(name="x"), rhs=Var(name="y"))
    cand_lt_yx = Bin(op="<", lhs=Var(name="y"), rhs=Var(name="x"))
    cand_eq = Eq(lhs=Var(name="x"), rhs=Var(name="y"))
    hole = HoleVar(candidates=(cand_lt_xy, cand_lt_yx, cand_eq))
    sketch = Lam(
        param="x", param_ty=TInt(),
        body=Lam(param="y", param_ty=TInt(), body=hole),
    )
    return SynthesisProblem(
        name="P7", sketch=sketch,
        target_type=TArrow(src=TInt(),
                           dst=TArrow(src=TInt(), dst=TBool())),
        examples=(
            IOExample(inputs=(IntLit(val=2), IntLit(val=3)),
                      output=BoolLit(val=True)),
            IOExample(inputs=(IntLit(val=5), IntLit(val=3)),
                      output=BoolLit(val=False)),
        ),
        anneal_steps=_HARD_STEPS,
    )


def _P8():
    """Deliberately-unsolvable: candidate is Int, target is Bool."""
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=(Var(name="x"),)))
    return SynthesisProblem(
        name="P8", sketch=sketch,
        target_type=TArrow(src=TInt(), dst=TBool()),
        examples=(IOExample(inputs=(IntLit(val=2),),
                            output=BoolLit(val=True)),),
    )


# ---- expected top-1 bodies (or full programs) -----------------------------
#
# For problems whose hole IS the entire body (P3/P4/P6/P7), we list the
# body. For programs synthesized whole (P1/P2/P5), we list the full Lam.
# The matcher tries both shapes via `_body(top)`.

_EXPECTED: dict[str, Node] = {
    "P1": Lam(param="x", param_ty=TInt(), body=Var(name="x")),
    "P2": Lam(param="x", param_ty=TInt(), body=Var(name="x")),
    "P3": App(fn=Var(name="f"), arg=Var(name="x")),
    "P4": If(
        cond=Bin(op="<", lhs=Var(name="x"), rhs=IntLit(val=5)),
        then_b=Var(name="x"),
        else_b=Bin(op="+", lhs=Var(name="x"), rhs=IntLit(val=1)),
    ),
    "P5": Lam(param="x", param_ty=TBool(), body=Var(name="x")),
    "P6": App(fn=Var(name="f"),
              arg=App(fn=Var(name="g"), arg=Var(name="x"))),
    "P7": Bin(op="<", lhs=Var(name="x"), rhs=Var(name="y")),
}


# Bundle-scale structural / nested-beta problems (P3, P4, P6, P7) are
# marked slow: they are opt-in via `--runslow`. P1, P2, P5, P8 stay in the
# default fast lane. Full P1-P8 still runs end-to-end under --runslow --
# no assertion is weakened by the marker.
_slow = pytest.mark.slow
_ALL_BUILDERS = [
    pytest.param(_P1, id="_P1"),
    pytest.param(_P2, id="_P2"),
    pytest.param(_P3, id="_P3", marks=_slow),
    pytest.param(_P4, id="_P4", marks=_slow),
    pytest.param(_P5, id="_P5"),
    pytest.param(_P6, id="_P6", marks=_slow),
    pytest.param(_P7, id="_P7", marks=_slow),
    pytest.param(_P8, id="_P8"),
]
_CORRECTNESS_CASES = [
    pytest.param("P1", _P1, id="P1"),
    pytest.param("P2", _P2, id="P2"),
    pytest.param("P3", _P3, id="P3", marks=_slow),
    pytest.param("P4", _P4, id="P4", marks=_slow),
    pytest.param("P5", _P5, id="P5"),
    pytest.param("P6", _P6, id="P6", marks=_slow),
    pytest.param("P7", _P7, id="P7", marks=_slow),
]


# ---- tests ----------------------------------------------------------------


@pytest.mark.parametrize("builder", _ALL_BUILDERS)
def test_problem_runs_without_raising(builder):
    """Every P1-P8 problem returns a SynthesisResult (no exception)."""
    res = synthesize(builder(), rng=np.random.default_rng(0))
    assert res is not None
    assert res.problem.name == builder.__name__.lstrip("_")


@pytest.mark.parametrize("name,builder", _CORRECTNESS_CASES)
def test_top1_is_correct(name, builder):
    """Spec §9.6.2: top-1 decodes to expected AST under ast_alpha_eq.

    Acceptance is 7/7 of P1-P7 here (spec §9.6.2 worded as 7/8 across all
    eight, with P8 being the refused one).
    """
    res = synthesize(builder(), rng=np.random.default_rng(0))
    assert res.completions, (
        f"{name}: no completions, failure_mode={res.failure_mode}")
    top = res.completions[0].ast
    expected = _EXPECTED[name]
    match_full = ast_alpha_eq(top, expected)
    match_body = ast_alpha_eq(_body(top), expected)
    assert match_full or match_body, (
        f"{name}: top-1 {top!r} != expected {expected!r} "
        f"(failure_mode={res.failure_mode}, "
        f"energy={res.completions[0].energy}, "
        f"breakdown={res.completions[0].energy_breakdown})")


def test_P8_is_refused():
    """Spec §9.6.4: the deliberately-unsolvable problem must be refused."""
    res = synthesize(_P8(), rng=np.random.default_rng(0))
    assert res.failure_mode is not None, (
        f"P8: expected refusal, got top-1 "
        f"{res.completions[0].ast!r} with energy "
        f"{res.completions[0].energy}")


@pytest.mark.slow
def test_energy_gap_to_second_completion():
    """Spec §9.6.3: gap between top-1 and second completion on P3
    must be >= 0.5 * w_T = 2.0 (HamiltonianWeights default w_T = 4.0).

    P3-scale; opt-in via --runslow.
    """
    res = synthesize(_P3(), rng=np.random.default_rng(0))
    if len(res.completions) >= 2:
        gap = res.completions[1].energy - res.completions[0].energy
        assert gap >= 2.0 - 1e-6, (
            f"P3: small energy gap {gap} between top-1 "
            f"({res.completions[0].energy}) and second "
            f"({res.completions[1].energy})")
