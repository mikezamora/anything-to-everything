"""Tests for run_problem / RunResult (spec §5, §11.4)."""

from __future__ import annotations

import math

from src.qft_pcn.bridge.runtime import run_problem, diagnose_problem
from src.qft_pcn.bridge.runtime.result import RunResult, ObservableValue


MINIMAL = {
    "fields": [{"name": "x", "cutoff": 4}],
    "sites": 2,
    "constraints": [],
    "observables": [{"site": 0, "field": "x", "op": "n"}],
    "search": {"method": "imag_time", "steps": 5, "chi_max": 4},
}


def test_minimal_run_returns_run_result():
    res = run_problem(MINIMAL)
    assert isinstance(res, RunResult)
    assert len(res.observables) == 1
    ov = res.observables[0]
    assert isinstance(ov, ObservableValue)
    assert ov.site == 0 and ov.field == "x" and ov.op == "n"
    assert math.isfinite(ov.value)
    assert abs(ov.imag_part) < 1e-9


def test_boundary_clamp_persists_after_evolution():
    dsl = {
        "fields": [{"name": "x", "cutoff": 4}],
        "sites": 2,
        "constraints": [],
        "boundary": {"0": {"x": 2}},
        "observables": [{"site": 0, "field": "x", "op": "n"}],
        "search": {"method": "imag_time", "steps": 20, "chi_max": 4},
    }
    res = run_problem(dsl)
    assert abs(res.observables[0].value - 2.0) < 1e-6


def test_energy_per_term_sum_matches_total():
    dsl = {
        "fields": [
            {"name": "kind", "cutoff": 4},
            {"name": "value", "cutoff": 4},
        ],
        "sites": 2,
        "constraints": [
            {"kind": "local", "site": 0, "term": "kind == 1", "weight": 1.0},
            {"kind": "local", "site": 1, "term": "kind == 1", "weight": 1.0},
        ],
        "observables": [{"site": 0, "field": "kind", "op": "n"}],
        "search": {"method": "imag_time", "steps": 5, "chi_max": 4},
    }
    res = run_problem(dsl)
    assert abs(sum(res.energy_per_term) - res.energy) < 1e-8


def test_converged_flag_for_simple_problem():
    dsl = {
        "fields": [{"name": "x", "cutoff": 4}],
        "sites": 2,
        "constraints": [
            {"kind": "local", "site": 0, "term": "x == 0", "weight": 5.0},
        ],
        "observables": [{"site": 0, "field": "x", "op": "n"}],
        "search": {"method": "imag_time", "steps": 50, "chi_max": 4,
                   "dt": 0.05},
    }
    res = run_problem(dsl)
    assert res.converged is True


def test_diagnose_returns_none_debugger_report_when_d_absent():
    res = diagnose_problem(MINIMAL)
    assert res.debugger_report is None
    assert res.result.observables[0].field == "x"
