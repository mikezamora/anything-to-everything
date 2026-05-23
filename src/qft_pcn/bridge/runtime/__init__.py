"""Bridge runtime: run_problem, diagnose_problem (spec §5).

The runtime is transport-independent; it accepts a Python dict (or JSON
string) DSL and returns a RunResult / RunDiagnostic dataclass.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dsl.pipeline import compile_dsl, CompiledDsl
from ..dsl.term import LocalTerm, TwoSiteTerm
from .hamiltonian import BridgeHamiltonian
from .initial_state import build_initial_state
from .clamp import Clamp
from .evolution import evolve_with_clamps
from .observables import build_observable_op
from .result import (
    ObservableValue, ConvergenceHistorySummary, RunResult, RunDiagnostic,
)
from ..errors import (
    UnsupportedMethodError, NumericFailureError, InternalError,
)
from src.qft_pcn.qft.evolution import energy as _energy


def _clamps_from_boundary(cd: CompiledDsl) -> list[Clamp]:
    out: list[Clamp] = []
    for site, fmap in cd.boundary.items():
        for fname, basis in fmap.items():
            out.append(Clamp(site=site, field=fname, basis_index=basis))
    return out


def _energy_per_term(state, terms: list) -> list[float]:
    vals = []
    for t in terms:
        if isinstance(t, LocalTerm):
            v = state.local_expectation(t.site, t.operator)
        elif isinstance(t, TwoSiteTerm):
            a, b = t.sites
            k = min(a, b)
            op = t.operator
            if b < a:
                # operator is in (b, a) order; trotter pre-aggregation
                # uses (k, k+1) order, so swap.
                d = int(round(np.sqrt(op.shape[0])))
                op = op.reshape(d, d, d, d).transpose(1, 0, 3, 2).reshape(
                    d * d, d * d)
            v = state.two_site_expectation(k, op)
        else:
            v = 0.0 + 0.0j
        vals.append(float(np.real(v)))
    return vals


def _converged(history: list[float], *, tol: float = 1e-6,
               window: int = 5) -> bool:
    if len(history) < window + 1:
        return False
    tail = history[-(window + 1):]
    deltas = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    monotonic = all(d <= 1e-8 for d in deltas)
    settled = all(abs(d) < tol for d in deltas)
    return monotonic and settled


def _bond_dimensions(state) -> list[int]:
    if hasattr(state, "bond_dimensions"):
        try:
            return list(state.bond_dimensions())
        except Exception:
            pass
    # Fallback: read from tensors directly.
    return [t.shape[2] for t in state.tensors[:-1]] if state.N > 1 else []


def run_problem(dsl: Any) -> RunResult:
    """Validate, compile, evolve, measure (spec §5.1)."""
    cd = compile_dsl(dsl)
    if cd.search.get("method", "imag_time") != "imag_time":
        raise UnsupportedMethodError(
            message=f"method {cd.search.get('method')!r} not supported",
            details={"method": cd.search.get("method")},
        )
    state = build_initial_state(fields=cd.fields, sites=cd.sites,
                                boundary=cd.boundary)
    H = BridgeHamiltonian(fields=cd.fields, sites=cd.sites, terms=cd.terms)
    clamps = _clamps_from_boundary(cd)
    history = evolve_with_clamps(
        state, H,
        dt=float(cd.search.get("dt", 0.05)),
        steps=int(cd.search["steps"]),
        chi_max=int(cd.search["chi_max"]),
        clamps=clamps, fields=cd.fields,
    )
    obs: list[ObservableValue] = []
    for o in cd.observables:
        op = build_observable_op(cd.fields, o["field"], o["op"])
        v = state.local_expectation(o["site"], op)
        if not (np.isfinite(v.real) and np.isfinite(v.imag)):
            raise NumericFailureError(
                message=f"observable at site {o['site']} returned non-finite",
                details={"site": o["site"], "field": o["field"], "op": o["op"]},
            )
        obs.append(ObservableValue(site=o["site"], field=o["field"],
                                    op=o["op"], value=float(v.real),
                                    imag_part=float(v.imag)))
    E = float(_energy(state, H))
    e_per_term = _energy_per_term(state, cd.terms)
    return RunResult(
        observables=obs,
        energy=E,
        energy_per_term=e_per_term,
        truncation_error_sum=float(sum(history.trunc_error_per_step)),
        final_bond_dimensions=_bond_dimensions(state),
        converged=_converged(history.energy_per_step),
        convergence_history=ConvergenceHistorySummary(
            energy_per_step=list(history.energy_per_step)
        ),
        # EXTENSIONS.md #1: surface the live state + Hamiltonian for the
        # composition layer's ``run_child`` -> ``integrate_child`` path.
        # The MPS bridge path doesn't produce a ``MeraEncodingMeta`` or a
        # decoded AST, so ``meta`` / ``solved_ast`` stay ``None`` here; a
        # future MERA-based runner can populate them. ``ground_state`` is
        # the final relaxed MPS, ``hamiltonian`` is the composed
        # ``BridgeHamiltonian`` under which ``energy`` was measured, and
        # ``trotter_steps`` is the imaginary-time step count actually run.
        ground_state=state,
        hamiltonian=H,
        trotter_steps=int(cd.search["steps"]),
    )


def diagnose_problem(dsl: Any) -> RunDiagnostic:
    """Same as run_problem plus sub-project D's diagnostic report (spec §5.3).

    Until D lands, debugger_report is None.
    """
    result = run_problem(dsl)
    try:
        from src.qft_pcn.logic.debugger import diagnose as _d_diagnose  # type: ignore  # noqa: F401
    except ImportError:
        return RunDiagnostic(result=result, debugger_report=None)
    return RunDiagnostic(result=result, debugger_report=None)


__all__ = [
    "run_problem", "diagnose_problem",
    "Clamp", "BridgeHamiltonian", "build_initial_state",
    "evolve_with_clamps", "build_observable_op",
    "RunResult", "RunDiagnostic", "ObservableValue",
]
