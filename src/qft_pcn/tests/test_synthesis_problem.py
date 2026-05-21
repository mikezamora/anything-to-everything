"""Tests for synthesis/problem.py + synthesis/_validate.py."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import (
    Var, Lam, App, IntLit, BoolLit, HoleVar, TypeHole,
    TInt, TBool, TArrow,
)
from src.qft_pcn.logic.synthesis.problem import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
)
from src.qft_pcn.logic.synthesis.errors import (
    SynthesisError, SynthesisProblemError, SynthesisRuntimeError,
)
from src.qft_pcn.logic.synthesis._validate import validate_problem


def test_io_example_construction():
    ex = IOExample(inputs=(IntLit(val=2),), output=IntLit(val=3))
    assert ex.inputs == (IntLit(val=2),)
    assert ex.output == IntLit(val=3)


def test_synthesis_problem_minimal():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar(candidates=["x"]))
    p = SynthesisProblem(sketch=sketch, name="P1")
    assert p.name == "P1"
    assert p.target_type is None
    assert p.examples == ()
    assert p.N == 32
    assert p.chi_max == 32
    assert p.n_samples == 64
    assert p.anneal_steps == 200
    assert p.anneal_dt == 0.05


def test_synthesis_problem_with_examples_and_target():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar(candidates=["x"]))
    p = SynthesisProblem(
        sketch=sketch,
        target_type=TArrow(src=TInt(), dst=TInt()),
        examples=(IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3)),),
        name="P2",
    )
    assert p.target_type == TArrow(src=TInt(), dst=TInt())
    assert len(p.examples) == 1


def test_hamiltonian_weights_defaults():
    w = HamiltonianWeights()
    assert w.w_typing == 4.0
    assert w.w_eval == 2.0
    assert w.w_examples == 3.0
    assert w.w_target_type == 2.0
    assert w.w_size == 0.1


def test_hamiltonian_weights_override():
    w = HamiltonianWeights(w_typing=8.0, w_size=0.0)
    assert w.w_typing == 8.0
    assert w.w_size == 0.0
    assert w.w_eval == 2.0


def test_completion_construction():
    c = Completion(
        ast=IntLit(val=3),
        energy=0.5,
        energy_breakdown={"typing": 0.0, "size": 0.5},
        diagnostics={},
        multiplicity=10,
    )
    assert c.energy == 0.5
    assert c.multiplicity == 10
    assert c.energy_breakdown["typing"] == 0.0


def test_synthesis_result_construction():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar(candidates=["x"]))
    p = SynthesisProblem(sketch=sketch, name="dummy")
    r = SynthesisResult(
        problem=p,
        completions=[],
        n_unique=0,
        n_samples_drawn=64,
        n_samples_decoded_ok=0,
        final_state_energy=42.0,
        wall_time_seconds=1.23,
        chi_observed_max=16,
        failure_mode="no_valid_completion",
    )
    assert r.failure_mode == "no_valid_completion"
    assert r.completions == []


# ---- error hierarchy + validation ----------------------------------------


def test_error_hierarchy():
    assert issubclass(SynthesisProblemError, SynthesisError)
    assert issubclass(SynthesisRuntimeError, SynthesisError)


def test_validate_rejects_no_holes():
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    p = SynthesisProblem(sketch=sketch)
    with pytest.raises(SynthesisProblemError, match="at least one hole"):
        validate_problem(p)


def test_validate_accepts_holevar_sketch():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar(candidates=["x"]))
    p = SynthesisProblem(sketch=sketch)
    validate_problem(p)


def test_validate_accepts_typehole_sketch():
    sketch = Lam(param="x",
                 param_ty=TypeHole(candidates=(TInt(), TBool())),
                 body=Var(name="x"))
    p = SynthesisProblem(sketch=sketch)
    validate_problem(p)


def test_validate_rejects_non_literal_io_example():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar(candidates=["x"]))
    bad_ex = IOExample(inputs=(Var(name="x"),), output=IntLit(val=1))
    p = SynthesisProblem(sketch=sketch, examples=(bad_ex,))
    with pytest.raises(SynthesisProblemError, match="literal"):
        validate_problem(p)


def test_validate_rejects_holevar_with_undef_candidate():
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=["z"]))
    p = SynthesisProblem(sketch=sketch)
    with pytest.raises(SynthesisProblemError, match="not in scope"):
        validate_problem(p)


def test_validate_rejects_invalid_knobs():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar(candidates=["x"]))
    with pytest.raises(SynthesisProblemError, match="N must"):
        validate_problem(SynthesisProblem(sketch=sketch, N=0))
    with pytest.raises(SynthesisProblemError, match="anneal_dt"):
        validate_problem(SynthesisProblem(sketch=sketch, anneal_dt=0.0))


def test_synthesize_function_importable():
    """synthesize should be exportable from the synthesis package."""
    from src.qft_pcn.logic.synthesis import synthesize
    assert callable(synthesize)
