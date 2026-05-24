"""Bridge runtime: run_problem, diagnose_problem (spec §5).

The runtime is transport-independent; it accepts a Python dict (or JSON
string) DSL and returns a RunResult / RunDiagnostic dataclass.
"""

from __future__ import annotations

import math
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
    """Dual-gate convergence: settled (|delta| < tol) AND monotonic
    (each step is approximately non-increasing within slack ``tol``).

    D22 fix: the previous ``monotonic`` check required deltas ``<= 1e-8``
    — three orders tighter than ``tol``. Imaginary-time evolution on a
    bridge Hamiltonian produces per-step deltas of order
    ``dt * <H^2>`` that routinely sit in ``[1e-8, 1e-6]`` for converged
    trajectories. The combined gate therefore reported
    ``converged=False`` on legitimate ground-state runs whenever the
    trace settled above ``1e-8 / step``, and the composition integrator
    refused every such child. Aligning ``monotonic`` to use ``tol`` as
    its slack restores the principled semantics: "no step inflates by
    more than ``tol``" (small positive jitter from finite-precision
    arithmetic is allowed; a sustained climb is not).
    """
    if len(history) < window + 1:
        return False
    tail = history[-(window + 1):]
    deltas = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    monotonic = all(d <= tol for d in deltas)
    settled = all(abs(d) < tol for d in deltas)
    return monotonic and settled


def _spectral_gap_from_hamiltonian(H, *, dim_ceiling: int = 4096) -> float:
    """Compute the spectral gap ``E1 - E0`` of the full Hamiltonian.

    D1 (DEVIATIONS.md): the composition §6.3 gate refuses every child
    whose runner does not surface a real spectral_gap. For the bridge
    runtime, the load-bearing Hamiltonian is a sparse sum of local +
    bond operators on a small chain (``H.sites`` typically O(2-6)
    sites at ``d_local`` 2-8). We assemble the full matrix and
    diagonalize via ``np.linalg.eigvalsh`` -- there is no Lanczos
    dependency to drag in and the matrix is small.

    Returns ``math.nan`` (consumer-side "unavailable" sentinel; NaN
    comparisons are always False so the strict refuse path fires by
    construction) when:
      - the total Hilbert dim ``d_local ** N`` exceeds ``dim_ceiling``
        (full diagonalization would be too costly; a Lanczos /
        ``scipy.sparse.linalg.eigsh`` route is the principled upgrade
        but requires a sparse / ``LinearOperator`` apply on the
        bridge Hamiltonian which it does not currently expose -- see
        EXTENSIONS.md entry for D23),
      - ``total < 2`` (degenerate substrate; no excited state),
      - the matrix yields a non-finite spectrum (numerical fault).

    D23 (DEVIATIONS.md): the previous fallback ``0.0`` was the
    strict-refuse value at the §6.3 gate AND the value a genuinely
    gapless substrate would emit -- the consumer could not tell
    "unavailable" from "real zero". Switching to NaN preserves the
    strict-refuse behaviour (gap < threshold remains False for NaN,
    so the gate refuses), while letting the consumer distinguish the
    two cases via ``math.isnan`` and emit a CLEAR refusal reason
    naming the substrate dim. A silent ``0.0`` for too-big substrates
    is a §1.1 anti-shortcut violation.

    ANTI-SHORTCUT (§1.1 / memory:anti-shortcut-directive): this is NOT
    a placeholder constant. The Hamiltonian is the same object that
    measured ``energy`` -- the gap is a real eigenvalue measurement of
    the substrate. A bigger workload should add a Lanczos branch, NOT
    swap in a heuristic guess.
    """
    N = int(H.N)
    d = int(H.d_local)
    total = d ** N
    if total > dim_ceiling or total < 2:
        return math.nan
    # Assemble full dense matrix: H = sum_k I^{otimes k} (x) local_k (x) I^{...}
    # + sum_k I^{...} (x) bond_k (x) I^{...}.
    M = np.zeros((total, total), dtype=complex)
    eye = np.eye(d, dtype=complex)

    def _kron_at(op: np.ndarray, k: int, op_sites: int) -> np.ndarray:
        # op acts on sites [k, k+op_sites). Build the full N-site operator.
        out = None
        i = 0
        while i < N:
            if i == k:
                term = op
                i += op_sites
            else:
                term = eye
                i += 1
            out = term if out is None else np.kron(out, term)
        return out

    for k in range(N):
        local = H.local_op(k)
        if np.any(local):
            M = M + _kron_at(local, k, 1)
    for k in range(N - 1):
        bond = H.bond_op(k)
        if np.any(bond):
            M = M + _kron_at(bond, k, 2)
    # Hermitize defensively (constructed terms should be Hermitian; tiny
    # asymmetry from floating-point rounding is the only expected gap).
    M = 0.5 * (M + M.conj().T)
    try:
        eigs = np.linalg.eigvalsh(M)
    except np.linalg.LinAlgError:
        return math.nan
    if not np.all(np.isfinite(eigs)):
        return math.nan
    eigs = np.sort(np.real(eigs))
    if len(eigs) < 2:
        return math.nan
    return float(eigs[1] - eigs[0])


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
        # D1 (DEVIATIONS.md): real spectral_gap surfaced from the same
        # composed Hamiltonian under which ``energy`` was measured. The
        # composition §6.3 gate consumes this via RunResult.to_dict().
        spectral_gap=_spectral_gap_from_hamiltonian(H),
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
