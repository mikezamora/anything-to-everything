"""Tests for relax_program (I-Task-10 driver, spec §8.12).

The relax_program driver encodes an AST directly, composes
H_typing + H_eval (+ promoted lemma constraints), and runs imag-time
evolution to residual < eps or until max_trotter_steps is consumed.
Distinct from the synthesis runner: no sketch holes, no witness
augmentation, no completion ranking.
"""
from __future__ import annotations

from src.qft_pcn.logic.ast import IntLit, Bin
from src.qft_pcn.logic.mera_synthesis import relax_program, RelaxResult


def test_relax_program_returns_named_result_for_closed_arith():
    """Closed arith relaxation returns a RelaxResult with sane fields.

    The driver must exercise REAL evolution: encode, compose, evolve.
    Residual must descend below eps within the budget.
    """
    src = Bin(op="+", lhs=IntLit(2), rhs=IntLit(3))
    res = relax_program(src, constraints=(), eps=1e-3,
                        max_trotter_steps=200, dt=0.05, chi_layer=16)
    assert isinstance(res, RelaxResult)
    assert res.state is not None
    assert res.meta.n_nodes >= 3
    assert res.residual < 1e-3
    assert 1 <= res.trotter_steps <= 200
    assert res.converged is True


def test_relax_program_stops_when_residual_below_eps():
    """Early termination: the driver does not consume the full budget
    once residual < eps. Verifies the per-chunk gate actually fires.
    """
    src = Bin(op="+", lhs=IntLit(1), rhs=IntLit(1))
    res = relax_program(src, constraints=(), eps=1e-2,
                        max_trotter_steps=500, dt=0.05, chi_layer=16)
    assert res.trotter_steps < 500
    assert res.residual < 1e-2
    assert res.converged is True


def test_relax_program_runs_full_budget_when_eps_unreachable():
    """eps=0 forces full-budget consumption; converged is False."""
    src = Bin(op="+", lhs=IntLit(2), rhs=IntLit(3))
    res = relax_program(src, constraints=(), eps=0.0,
                        max_trotter_steps=40, dt=0.05, chi_layer=16)
    assert res.trotter_steps == 40
    assert res.converged is False
