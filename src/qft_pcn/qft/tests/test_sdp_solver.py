"""Tests for the CVXPY-backed SDP solver shim (S2)."""

from __future__ import annotations

import numpy as np
import pytest

import cvxpy as cp

from src.qft_pcn.qft.sdp_solver import (
    LinearConstraint,
    PSDVariable,
    SDPProblem,
    SDPSolution,
    SDPStatus,
    psd_constraint_from_operator,
    solve_sdp,
)


def test_trivial_psd_min() -> None:
    """Minimize tr(X) over PSD X with X[0,0] = 1.

    The optimum is achieved by X = e0 e0^T (rank-1, trace 1).
    """

    var = PSDVariable(name="X", size=3)

    def _e00(vars_dict):
        return vars_dict["X"][0, 0] == 1.0

    def _objective(vars_dict):
        return cp.trace(vars_dict["X"])

    problem = SDPProblem(
        variables=(var,),
        constraints=(LinearConstraint(builder=_e00, name="X[0,0]=1"),),
        objective_builder=_objective,
        minimize=True,
        name="trivial_psd_min",
    )

    sol = solve_sdp(problem)

    assert sol.status == SDPStatus.OPTIMAL, sol.raw_status
    assert sol.optimal_value == pytest.approx(1.0, abs=1e-4)
    X = sol.matrices["X"]
    assert X.shape == (3, 3)
    assert X[0, 0] == pytest.approx(1.0, abs=1e-4)
    # Trace must equal the objective.
    assert float(np.trace(X)) == pytest.approx(1.0, abs=1e-4)
    # Eigenvalues must be non-negative (PSD).
    eigs = np.linalg.eigvalsh(X)
    assert (eigs >= -1e-7).all(), eigs


def test_infeasible_sdp_returns_infeasible() -> None:
    """X PSD + X[0,0] = -1 must be flagged INFEASIBLE.

    A PSD matrix has non-negative diagonal entries, so the constraint
    X[0,0] = -1 cannot be satisfied.
    """

    var = PSDVariable(name="X", size=2)

    def _neg_diag(vars_dict):
        return vars_dict["X"][0, 0] == -1.0

    def _obj(vars_dict):
        return cp.trace(vars_dict["X"])

    problem = SDPProblem(
        variables=(var,),
        constraints=(LinearConstraint(builder=_neg_diag, name="X[0,0]=-1"),),
        objective_builder=_obj,
        minimize=True,
        name="infeasible",
    )

    sol = solve_sdp(problem)

    assert sol.status == SDPStatus.INFEASIBLE, sol.raw_status
    # No PSD matrix solution should have been recovered.
    assert "X" not in sol.matrices


def test_max_eigenvalue_sdp() -> None:
    """λ_max(M) via SDP: minimize t s.t. t*I - M is PSD.

    Compare to numpy.linalg.eigh ground truth.
    """

    rng = np.random.default_rng(20260523)
    A = rng.normal(size=(5, 5))
    M = 0.5 * (A + A.T)  # symmetric

    eigs = np.linalg.eigvalsh(M)
    lam_max = float(eigs.max())

    # SDP encoding: variable Y = t*I - M must be PSD, where t is a free
    # scalar. We introduce Y as a PSDVariable and add equality
    # constraints Y == t*I - M for an auxiliary t. CVXPY can't represent
    # `t` as a free scalar through our shim's PSD-only variable surface,
    # so we instead use a 1×1 PSD variable T together with an unconstrained
    # offset: equivalently, we let the SDP solve over the lift
    #    Y >> 0, Y_ij == s*delta_ij - M_ij  with s free
    # which we encode by making s a degenerate "1×1 PSD" variable
    # (PSD ↔ s ≥ 0) and shifting M so that the optimum stays positive.
    #
    # To avoid the s ≥ 0 quirk, add a constant shift c > -eigs.min().
    c = float(max(0.0, -eigs.min() + 1.0))  # ensures s >= 0 at the optimum
    M_shifted = M + c * np.eye(5)
    # Now λ_max(M_shifted) = lam_max + c > 0.

    s_var = PSDVariable(name="s", size=1)  # s >= 0 (a 1×1 PSD scalar)
    Y_var = PSDVariable(name="Y", size=5)  # Y = s*I - M_shifted, PSD

    def _link(vars_dict):
        s = vars_dict["s"]
        Y = vars_dict["Y"]
        # s*I - M_shifted - Y == 0, elementwise.
        return Y + M_shifted == s * np.eye(5)

    def _obj(vars_dict):
        return vars_dict["s"][0, 0]

    problem = SDPProblem(
        variables=(s_var, Y_var),
        constraints=(LinearConstraint(builder=_link, name="Y=sI-M"),),
        objective_builder=_obj,
        minimize=True,
        name="max_eig_via_sdp",
    )

    sol = solve_sdp(problem)
    assert sol.status == SDPStatus.OPTIMAL, sol.raw_status

    lam_max_sdp_shifted = sol.optimal_value
    lam_max_sdp = lam_max_sdp_shifted - c

    assert lam_max_sdp == pytest.approx(lam_max, abs=1e-5)


def test_psd_constraint_from_operator_psd_input() -> None:
    """A genuinely PSD operator yields a feasible PSD equality constraint."""

    rng = np.random.default_rng(0)
    A = rng.normal(size=(4, 4))
    M = A @ A.T + 0.1 * np.eye(4)  # PSD by construction

    var, constraint = psd_constraint_from_operator(M, name="op")

    def _obj(vars_dict):
        return cp.trace(vars_dict["op"])

    problem = SDPProblem(
        variables=(var,),
        constraints=(constraint,),
        objective_builder=_obj,
        minimize=True,
        name="op_psd_feasible",
    )

    sol = solve_sdp(problem)
    assert sol.status == SDPStatus.OPTIMAL, sol.raw_status
    recovered = sol.matrices["op"]
    np.testing.assert_allclose(recovered, 0.5 * (M + M.T), atol=1e-5)


def test_psd_constraint_from_operator_indefinite_input_infeasible() -> None:
    """An indefinite operator forces the equality+PSD pair to be infeasible."""

    M = np.diag([1.0, -2.0, 3.0])  # indefinite: one negative eigenvalue

    var, constraint = psd_constraint_from_operator(M, name="op")

    def _obj(vars_dict):
        return cp.trace(vars_dict["op"])

    problem = SDPProblem(
        variables=(var,),
        constraints=(constraint,),
        objective_builder=_obj,
        minimize=True,
        name="op_psd_infeasible",
    )

    sol = solve_sdp(problem)
    assert sol.status == SDPStatus.INFEASIBLE, sol.raw_status


def test_psd_constraint_from_operator_rejects_nonsquare() -> None:
    with pytest.raises(ValueError):
        psd_constraint_from_operator(np.zeros((2, 3)), name="bad")
