"""Tests for RunResult composition-layer fields (EXTENSIONS.md #1).

K-5's ``integrate_child`` and the dispatcher's ``run_child`` rely on
``RunResult`` carrying ``meta`` / ``ground_state`` / ``solved_ast`` /
``hamiltonian`` / ``trotter_steps``. These fields are additive with safe
defaults; the bridge's MPS path populates ``ground_state``,
``hamiltonian``, and ``trotter_steps`` (the rest stay ``None`` because
the MPS path doesn't produce a ``MeraEncodingMeta`` or a decoded AST --
that's the MERA-runner's job, see EXTENSIONS.md #1).
"""

from __future__ import annotations

from src.qft_pcn.bridge.runtime import run_problem
from src.qft_pcn.bridge.runtime.hamiltonian import BridgeHamiltonian
from src.qft_pcn.bridge.runtime.result import (
    ConvergenceHistorySummary, RunResult,
)
from src.qft_pcn.qft.mps import MPS


MINIMAL = {
    "fields": [{"name": "x", "cutoff": 4}],
    "sites": 2,
    "constraints": [],
    "observables": [{"site": 0, "field": "x", "op": "n"}],
    "search": {"method": "imag_time", "steps": 7, "chi_max": 4},
}


def test_run_result_defaults_keep_legacy_callers_working():
    """Constructing RunResult without the new fields must still succeed."""
    r = RunResult(
        observables=[], energy=0.0, energy_per_term=[],
        truncation_error_sum=0.0, final_bond_dimensions=[], converged=False,
        convergence_history=ConvergenceHistorySummary(),
    )
    assert r.meta is None
    assert r.ground_state is None
    assert r.solved_ast is None
    assert r.hamiltonian is None
    assert r.trotter_steps == 0


def test_run_result_accepts_and_preserves_composition_fields():
    """Caller-provided meta/ground_state/solved_ast round-trip on the dataclass."""
    sentinel_meta = object()
    sentinel_state = object()
    sentinel_ast = object()
    sentinel_H = object()
    r = RunResult(
        observables=[], energy=0.0, energy_per_term=[],
        truncation_error_sum=0.0, final_bond_dimensions=[], converged=True,
        convergence_history=ConvergenceHistorySummary(),
        meta=sentinel_meta, ground_state=sentinel_state,
        solved_ast=sentinel_ast, hamiltonian=sentinel_H, trotter_steps=12,
    )
    assert r.meta is sentinel_meta
    assert r.ground_state is sentinel_state
    assert r.solved_ast is sentinel_ast
    assert r.hamiltonian is sentinel_H
    assert r.trotter_steps == 12


def test_run_problem_populates_ground_state_and_hamiltonian():
    """End-to-end: run_problem must publish the live MPS + BridgeHamiltonian."""
    res = run_problem(MINIMAL)
    # MPS path produces no MeraEncodingMeta / decoded AST.
    assert res.meta is None
    assert res.solved_ast is None
    # But the relaxed state and composed Hamiltonian must come through.
    assert isinstance(res.ground_state, MPS)
    assert res.ground_state.N == 2
    assert isinstance(res.hamiltonian, BridgeHamiltonian)
    assert res.hamiltonian.N == 2
    assert res.trotter_steps == 7


def test_run_problem_trotter_steps_matches_search_steps():
    """trotter_steps reflects the configured search.steps, not a hard-coded zero."""
    dsl = dict(MINIMAL)
    dsl["search"] = {"method": "imag_time", "steps": 3, "chi_max": 4}
    res = run_problem(dsl)
    assert res.trotter_steps == 3
    # to_dict surface also publishes it for provenance.
    assert res.to_dict()["trotter_steps"] == 3


def test_dispatcher_getattr_path_recovers_real_values():
    """Dispatcher uses ``getattr(run_result, "ground_state", None)`` etc.

    With the new fields, those getattrs return real values, not None.
    Mirrors the access pattern in ``composition.dispatcher.run_child``.
    """
    res = run_problem(MINIMAL)
    assert getattr(res, "ground_state", None) is not None
    assert getattr(res, "meta", "missing") is None  # really None, not absent
    assert getattr(res, "solved_ast", "missing") is None
    assert getattr(res, "hamiltonian", None) is not None
    assert getattr(res, "trotter_steps", -1) == 7
