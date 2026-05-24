"""Semidefinite Programming (SDP) solver shim.

Substrate task S2 — required prerequisite for §12.4 (conformal bootstrap for
type-only reasoning) in ``QFT_PCN_ARCHITECTURE.md``.

CFT-bootstrap-style reasoning formulates type-only program properties as
SDPs: PSD constraints encode unitarity (operator norms non-negative),
linear constraints encode crossing/parametricity, and a linear (or scalar)
objective extracts the quantitative bound. This module provides:

* :class:`SDPProblem`   — variables (PSD blocks) + linear constraints +
  linear objective.
* :class:`SDPSolution`  — optimal value, per-variable matrix solutions,
  and a :class:`SDPStatus` flag.
* :func:`solve_sdp`     — thin wrapper around CVXPY.
* :func:`psd_constraint_from_operator` — projects a Hermitian operator
  block (e.g. an operator slice from substrate data) into the CVXPY PSD
  constraint surface.

§1.6 note: SDPs are *numerical* optimizers — they are not themselves
operator-algebraic. What keeps this faithful to the architecture is that
the *problem encoding* must originate in substrate operator data
(:func:`psd_constraint_from_operator` is the supported channel for that).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

import cvxpy as cp


class SDPStatus(str, Enum):
    """Solver outcome, normalised across CVXPY status strings."""

    OPTIMAL = "OPTIMAL"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    INACCURATE = "INACCURATE"
    ERROR = "ERROR"


def _classify_status(cvxpy_status: str) -> SDPStatus:
    s = (cvxpy_status or "").lower()
    if s == "optimal":
        return SDPStatus.OPTIMAL
    if "infeasible" in s and "inaccurate" not in s:
        return SDPStatus.INFEASIBLE
    if "unbounded" in s and "inaccurate" not in s:
        return SDPStatus.UNBOUNDED
    if "inaccurate" in s:
        return SDPStatus.INACCURATE
    return SDPStatus.ERROR


# -- Constraint construction ------------------------------------------------


@dataclass(frozen=True)
class LinearConstraint:
    """A linear equality/inequality constraint over the SDP variables.

    The constraint is rendered by the user-supplied ``builder`` callable
    once CVXPY variables exist: ``builder(vars_dict) -> cvxpy.Constraint``.
    Keeping it lazy lets the same :class:`SDPProblem` be re-emitted into
    fresh CVXPY problem objects (useful for parameter sweeps).
    """

    builder: Any  # Callable[[Mapping[str, cp.Variable]], cp.Constraint]
    name: str = ""


@dataclass(frozen=True)
class PSDVariable:
    """A symmetric PSD matrix variable of shape ``(size, size)``."""

    name: str
    size: int

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"PSD variable {self.name!r} needs size >= 1")


@dataclass(frozen=True)
class SDPProblem:
    """An SDP in standard primal form.

    * ``variables``  — one or more PSD matrix variables.
    * ``constraints`` — lazy builders producing CVXPY constraints.
    * ``objective_builder`` — ``builder(vars_dict) -> cvxpy expression``.
    * ``minimize``  — if False, maximize.
    """

    variables: tuple[PSDVariable, ...]
    constraints: tuple[LinearConstraint, ...]
    objective_builder: Any  # Callable[[Mapping[str, cp.Variable]], cp expr]
    minimize: bool = True
    name: str = "sdp"

    def variable_names(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables)


@dataclass(frozen=True)
class SDPSolution:
    """Result of :func:`solve_sdp`."""

    status: SDPStatus
    optimal_value: float
    matrices: Mapping[str, np.ndarray] = field(default_factory=dict)
    raw_status: str = ""
    solver: str = ""


# -- Solver -----------------------------------------------------------------


def solve_sdp(
    problem: SDPProblem,
    *,
    solver: str | None = None,
    verbose: bool = False,
) -> SDPSolution:
    """Solve ``problem`` via CVXPY.

    On INFEASIBLE/UNBOUNDED problems CVXPY returns ``+inf``/``-inf`` as the
    objective value and ``None`` for variable values; the wrapper preserves
    the infinite value but emits an empty matrix dict.
    """

    cvx_vars: dict[str, cp.Variable] = {
        v.name: cp.Variable((v.size, v.size), symmetric=True)
        for v in problem.variables
    }

    cvx_constraints: list[Any] = [v >> 0 for v in cvx_vars.values()]
    for c in problem.constraints:
        cvx_constraints.append(c.builder(cvx_vars))

    expr = problem.objective_builder(cvx_vars)
    objective = cp.Minimize(expr) if problem.minimize else cp.Maximize(expr)

    cp_problem = cp.Problem(objective, cvx_constraints)

    solve_kwargs: dict[str, Any] = {"verbose": verbose}
    if solver is not None:
        solve_kwargs["solver"] = solver

    try:
        optimal_value = cp_problem.solve(**solve_kwargs)
    except cp.error.SolverError as exc:  # pragma: no cover - environment specific
        return SDPSolution(
            status=SDPStatus.ERROR,
            optimal_value=float("nan"),
            matrices={},
            raw_status=f"SolverError: {exc}",
            solver=str(solver or ""),
        )

    status = _classify_status(cp_problem.status)

    matrices: dict[str, np.ndarray] = {}
    if status in (SDPStatus.OPTIMAL, SDPStatus.INACCURATE):
        for name, var in cvx_vars.items():
            val = var.value
            if val is not None:
                matrices[name] = np.asarray(val)

    # CVXPY may return None when solver bails; normalise to nan.
    if optimal_value is None:
        opt_val = float("nan")
    else:
        opt_val = float(optimal_value)

    return SDPSolution(
        status=status,
        optimal_value=opt_val,
        matrices=matrices,
        raw_status=cp_problem.status,
        solver=str(solver or cp_problem.solver_stats.solver_name if cp_problem.solver_stats else ""),
    )


# -- Operator → PSD constraint helper --------------------------------------


def psd_constraint_from_operator(
    operator: np.ndarray,
    *,
    name: str = "operator",
    hermitize: bool = True,
    tol: float = 1e-10,
) -> tuple[PSDVariable, LinearConstraint]:
    """Encode a Hermitian operator block as a PSD-equality constraint.

    Given a Hermitian operator ``M`` (e.g. a substrate operator slice),
    build a PSD variable ``X`` of the same shape and a linear equality
    ``X == M_sym``, where ``M_sym = (M + M.conj().T)/2`` if
    ``hermitize`` is True. The natural use case is: "this substrate
    operator must be PSD to satisfy unitarity (§12.4)"; the SDP then
    declares the problem infeasible whenever ``M`` has a negative
    eigenvalue outside ``tol``.

    Returns the (variable, constraint) pair to add to an
    :class:`SDPProblem`.
    """

    op = np.asarray(operator)
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise ValueError(
            f"operator must be square 2D, got shape {op.shape}"
        )

    if hermitize:
        op_sym = 0.5 * (op + op.conj().T)
    else:
        herm_err = np.linalg.norm(op - op.conj().T)
        if herm_err > tol:
            raise ValueError(
                f"operator not Hermitian within tol={tol}: ||M-M*||={herm_err:.3e}"
            )
        op_sym = op

    if np.iscomplexobj(op_sym):
        max_imag = float(np.max(np.abs(op_sym.imag)))
        if max_imag > tol:
            raise ValueError(
                "complex-valued PSD constraints not yet supported by this "
                f"helper (max |Im|={max_imag:.3e}); pass a real symmetric "
                "operator or extend the helper."
            )
        op_sym = op_sym.real

    var = PSDVariable(name=name, size=op_sym.shape[0])
    target = np.asarray(op_sym, dtype=float)

    def _builder(vars_dict: Mapping[str, cp.Variable]) -> Any:
        return vars_dict[name] == target

    return var, LinearConstraint(builder=_builder, name=f"{name}==operator")


__all__ = [
    "SDPStatus",
    "PSDVariable",
    "LinearConstraint",
    "SDPProblem",
    "SDPSolution",
    "solve_sdp",
    "psd_constraint_from_operator",
]
