"""Tests for synthesis problem data types (spec §6.1)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import Lam, TInt, Var, HoleVar
from src.qft_pcn.logic.mera_synthesis.problem import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
)
from src.qft_pcn.logic.mera_synthesis.errors import (
    SynthesisProblemError, SynthesisRuntimeError,
)


def test_io_example_holds_inputs_and_output():
    from src.qft_pcn.logic.ast import IntLit
    ex = IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3))
    assert ex.inputs[0].val == 3


def test_problem_defaults():
    hole = HoleVar(candidates=("x",))
    p = SynthesisProblem(sketch=Lam(param="x", param_ty=TInt(), body=hole))
    assert p.chi_layer == 32
    assert p.n_samples == 64
    assert p.target_type is None


def test_hamiltonian_weights_defaults():
    w = HamiltonianWeights()
    assert (w.w_T, w.w_E, w.w_X, w.w_Y, w.w_S) == (4.0, 2.0, 3.0, 2.0, 0.1)


def test_errors_are_distinct_classes():
    assert issubclass(SynthesisProblemError, Exception)
    assert issubclass(SynthesisRuntimeError, Exception)
    assert SynthesisProblemError is not SynthesisRuntimeError
