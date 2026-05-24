"""D31 (DEVIATIONS.md): RunResult.spectral_gap default + dispatcher
_timeout_result fallback must be ``math.nan`` (the D23 unavailability
sentinel), not ``0.0``.

The audit-pass3 §6-§9 finding: a finite ``0.0`` default collides with
the D23 contract that "unavailable" must be NaN so the §6.3 gate's
``math.isnan(...)`` branch can refuse with a CLEAR reason rather than
the misleading "near-degenerate" one a finite-zero would trigger.
"""
from __future__ import annotations

import math

from src.qft_pcn.bridge.runtime.result import (
    ConvergenceHistorySummary, RunResult,
)
from src.qft_pcn.composition.dispatcher import _timeout_result
from src.qft_pcn.composition.goal_graph import SubGoal


def _bare_runresult() -> RunResult:
    """Construct a RunResult with the minimum positional args (no
    spectral_gap kwarg). The D31 contract: a caller that omits the
    spectral_gap MUST land on NaN, not 0.0."""
    return RunResult(
        observables=[],
        energy=0.0,
        energy_per_term=[],
        truncation_error_sum=0.0,
        final_bond_dimensions=[],
        converged=True,
        convergence_history=ConvergenceHistorySummary(),
    )


def test_runresult_default_spectral_gap_is_nan():
    """The dataclass default for spectral_gap is math.nan (D31).

    Any caller that constructs a RunResult without populating the
    field hits the D23 "unavailable" sentinel and the §6.3 gate's
    NaN branch fires with a truthful refusal reason -- NOT the
    misleading "near-degenerate" one a finite 0.0 would have
    produced. ``to_dict()`` must propagate the NaN unchanged so the
    integrator reading ``run_diagnostic['spectral_gap']`` sees it.
    """
    rr = _bare_runresult()
    assert math.isnan(rr.spectral_gap), (
        f"RunResult.spectral_gap default must be NaN (D31), got "
        f"{rr.spectral_gap!r}")
    diag = rr.to_dict()
    assert math.isnan(diag["spectral_gap"]), (
        "to_dict() must surface the NaN default; got "
        f"{diag['spectral_gap']!r}")


def test_timeout_result_spectral_gap_is_nan():
    """``_timeout_result``'s run_diagnostic carries spectral_gap=NaN.

    Pre-D31, ``run_diagnostic={}`` caused
    ``_spectral_gap.get('spectral_gap', 0.0)`` to return ``0.0`` and the
    integrator refused with the misleading "near-degenerate" reason.
    The D31 fix surfaces NaN explicitly so the §6.3 gate's
    ``math.isnan(...)`` branch fires and the refusal reason names the
    truthful structural cause (timeout → no gap measured).
    """
    sg = SubGoal(
        goal_id="g-timeout",
        dsl_spec={},
        goal_prop="p",
        boundary={},
        parent_leaves=(),
    )
    res = _timeout_result(sg)
    assert "spectral_gap" in res.run_diagnostic, (
        "timeout result must populate spectral_gap key (D31)")
    gap = res.run_diagnostic["spectral_gap"]
    assert math.isnan(gap), (
        f"timeout result spectral_gap must be NaN (D31), got {gap!r}")
    assert res.error == "timeout"
    assert res.converged is False
