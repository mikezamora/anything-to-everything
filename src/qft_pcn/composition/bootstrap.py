"""§12.4 Conformal bootstrap for type-only reasoning — partial.

Honesty scope (D13): this module currently delivers a *typing-
feasibility check* — an SDP wrapper around the §12.1 anomaly diagonals
that returns feasible for well-typed states and infeasible for
ill-typed states. It does **NOT** derive the full §12.4 spec
capability set: termination bounds, depth bounds, complexity bounds,
or parametricity bounds from the type signature alone. Those require
real OPE-like CFT-bootstrap crossing-equation constraints on the
typing Hamiltonian (substrate-wide future work; see EXTENSIONS.md
entry "§12.4 conformal bootstrap full bound capabilities").

What ships here is sound as a necessary-but-not-sufficient feasibility
witness: any program that fails the typing-feasibility SDP cannot
satisfy the full §12.4 bootstrap either; passing the SDP means the
§12.1-anomaly diagonals fit under the truncation gap, nothing more.

Physics origin (Polyakov 1974, Ferrara-Gatto-Grillo 1973, Rattazzi-
Rychkov-Tonni-Vichi 2008): the conformal bootstrap derives properties of
conformal field theories from pure consistency requirements — unitarity
(states have non-negative norm), crossing symmetry, OPE truncation —
without explicitly constructing the theory. Numerically it is formulated
as a semidefinite program: positive-semidefinite (PSD) constraints
encode unitarity, linear equality constraints encode crossing /
parametricity, a linear objective extracts a bound on a physical
observable. The optimum is a *provable* bound that any consistent CFT
must satisfy (Kos-Poland-Simmons-Duffin 2014's 3D Ising critical
exponents to six decimals are a famous instance).

QPCN realization (spec §12.4): the typing Hamiltonian's per-rule
leaf-projector structure is a *bootstrap system* in this exact sense.
For each one-node typing rule at each AST node, the obstruction
expectation

    o_i = <psi | P_kind^{(i)} . (I - P_required_type^{(i)}) | psi>

is the operator-algebraic measurement of "the rule fires on this node
but the type leaf is wrong" — the §12.1 ABJ-style obstruction surfaced
through a substrate window expectation (no AST is walked; §1.1). The
bootstrap SDP then asks: *is there a unitary OPE truncation under
which every obstruction stays below the truncation gap?* Concretely:

    minimize  trace(X)
    subject to  X >> 0   (unitarity)
                X[i, i] == o_i  (crossing: diagonals fixed by substrate)
                trace(X) <= tau (truncation gap)

For a well-typed program every ``o_i = 0``, so ``X = 0`` is feasible
with optimum 0 — the SDP is **feasible** and the dimension bound is
trivial. For an ill-typed program at least one ``o_i > 0``, so
``trace(X) >= sum o_i > tau`` and the SDP is **infeasible** — the
type-only bootstrap proves the program cannot have a consistent
implementation under any OPE truncation finer than ``tau``.

Also exposed: the *dimension bound* — when feasible, the maximum
eigenvalue of ``X`` (= max obstruction) is a provable upper bound on
the operator-dimension gap any concrete implementation must respect.
Well-typed: bound 0. Engineered substrate-level perturbation of one
type leaf: the bound matches the magnitude of that perturbation.

§1.1 anti-shortcut contract:
  * Constraints are read off the typing Hamiltonian's leaf-projector
    operator structure (the same ``_ONE_NODE_RULE_SIGNATURE`` table the
    §12.1 anomaly module uses) — NOT from AST symbol enumeration.
  * Diagonals ``o_i`` come from real ``mera_window_expectation_factored``
    calls on the substrate state — bond entanglement, not classical
    lookup.
  * The optimizer is the real S2 ``solve_sdp`` (CVXPY-backed). No mock.

§1.3: no dense ``16**k`` operator is ever constructed; every window
expectation routes through the factored primitive.

Cost: ``O(N_generators)`` substrate window calls + one SDP solve.
Generators = one-node typing rules × AST nodes; for STLC-class type
signatures this is small (tens to hundreds of variables of a *diagonal*
SDP), well within CVXPY/SCS reach.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cvxpy as cp

from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.qft.sdp_solver import (
    LinearConstraint,
    PSDVariable,
    SDPProblem,
    SDPSolution,
    SDPStatus,
    solve_sdp,
)
from src.qft_pcn.composition.anomaly import (
    SymmetryGenerator,
    compute_anomaly,
    extract_symmetries,
)
from src.qft_pcn.logic._mera_window import mera_window_expectation_factored
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# OPE truncation gap (the "tau" in the bootstrap SDP). Generators whose
# obstruction expectation is below this value are taken to be at the
# bootstrap floor — i.e., consistent with the truncated operator
# spectrum. Above this value the bootstrap declares the program
# inconsistent (infeasible SDP). The value matches the §12.1 anomaly
# floor so a single residual-zero convention propagates across the
# composition modules.
DEFAULT_TRUNCATION_GAP = 1e-6

# Floor for "the rule's kind projector actually fires on this node".
# Generators below this are inactive (the rule's hypothesis isn't
# satisfied on the node) and don't contribute a bootstrap constraint —
# their diagonal would be a vacuous 0. Matches the substrate's
# typing-Hamiltonian residual zero convention.
DEFAULT_KIND_ACTIVITY_FLOOR = 1e-9


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapResult:
    """Result of :func:`solve_bootstrap`.

    Fields:
      * ``status`` — the normalized SDP solver outcome
        (:class:`SDPStatus`). ``OPTIMAL`` and ``INACCURATE`` count as
        feasible; ``INFEASIBLE`` is the §12.4 impossibility certificate.
      * ``optimal_value`` — the optimum of ``trace(X)`` over the
        bootstrap region. For a well-typed program this is ~0 (well
        within the truncation gap). For an ill-typed program CVXPY
        returns ``+inf``.
      * ``dimension_bound`` — the largest eigenvalue of the recovered
        ``X`` matrix; a provable upper bound on the operator-dimension
        gap any concrete implementation must respect. ``+inf`` when
        infeasible, 0 when the only feasible solution is the zero
        matrix.
      * ``truncation_gap`` — the ``tau`` constant the problem was
        solved against; surfaced so downstream consumers can verify
        their numerical floor.
      * ``feasible`` — convenience boolean: True iff status is
        ``OPTIMAL`` or ``INACCURATE``.
    """
    status: SDPStatus
    optimal_value: float
    dimension_bound: float
    truncation_gap: float
    feasible: bool


@dataclass(frozen=True)
class BootstrapVerification:
    """Top-level §12.4 acceptance result.

    A well-typed substrate state produces ``well_typed=True`` and a
    feasible bootstrap with ``dimension_bound`` ~ 0. An ill-typed
    substrate state produces ``well_typed=False`` and an infeasible
    bootstrap. ``obstructions`` lists the per-generator obstruction
    expectations above the kind-activity floor (the inputs that drove
    the SDP outcome) so the caller can localize the violation.
    """
    well_typed: bool
    result: BootstrapResult
    obstructions: tuple[tuple[str, int, float], ...]


# ---------------------------------------------------------------------------
# Operator-derived obstruction extraction
# ---------------------------------------------------------------------------


def _obstruction_expectation(
    generator: SymmetryGenerator, state: MERA
) -> float:
    """``<P_kind . (I - P_required_type) . [P_value]> on state``.

    Operator-algebraic (§1.1): builds the window from the generator's
    *substrate-derived* leaf-projector triple — the same projectors the
    typing Hamiltonian rule body uses — and reads the expectation through
    :func:`mera_window_expectation_factored` (the §1.3 no-dense path).

    Reuses the §12.1 ABJ trace machinery: the bootstrap obstruction IS
    the ABJ anomaly trace for the rule's leaf-projector triple, surfaced
    here as the diagonal entry of the bootstrap SDP's ``X`` matrix.
    """
    return float(compute_anomaly(generator, state))


def _kind_expectation(generator: SymmetryGenerator, state: MERA) -> float:
    """``<P_kind>`` on state — the rule's hypothesis-firing probability.

    Used as the activity gate: generators below the kind-activity floor
    do not contribute a bootstrap constraint (the rule's antecedent
    isn't satisfied on the node). Operator-algebraic: pulls the kind
    projector from the generator's substrate-derived triple.
    """
    # First projector in the SymmetryGenerator triple is always P_kind by
    # construction in extract_symmetries (anomaly.py line 195).
    kind_leaf, kind_op = generator.projectors[0]
    ops = {kind_leaf: kind_op}
    return float(np.real(mera_window_expectation_factored(state, ops)))


# ---------------------------------------------------------------------------
# Bootstrap problem construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TypingConstraints:
    """Internal handle: the operator-derived inputs to the bootstrap SDP.

    Built by :func:`build_bootstrap_problem` from the (typing_H, state)
    pair. ``obstruction_diag`` is the vector ``o_i`` of obstruction
    expectations; ``generator_refs`` is the parallel list of
    (rule_id, node) labels for downstream localization. The SDP
    variable ``X`` has size ``len(obstruction_diag)``.
    """
    obstruction_diag: tuple[float, ...]
    generator_refs: tuple[tuple[str, int], ...]
    truncation_gap: float


def build_bootstrap_problem(
    typing_constraints: _TypingConstraints,
) -> SDPProblem:
    """Encode type-system constraints as a CFT bootstrap SDP.

    Constructs an :class:`SDPProblem` over a single PSD variable ``X``
    of size ``k`` (one slot per active generator) with three constraint
    classes mirroring the bootstrap structure:

      * **Unitarity**: ``X >> 0`` (added by the solver shim from the
        PSDVariable declaration).
      * **Crossing**: ``X[i, i] == o_i`` — the substrate obstruction
        expectations pin the diagonal. This is the operator-algebraic
        analogue of "the OPE coefficients are fixed by the operator
        product expansion"; here the substrate's factored window
        expectations are the OPE data.
      * **Truncation**: ``trace(X) <= tau`` — the bootstrap truncation
        gap, declaring that the sum of obstruction magnitudes is below
        the OPE truncation scale. For a well-typed program this is
        trivially satisfied (every ``o_i = 0``); for an ill-typed
        program the substrate-imposed diagonal forces a positive trace
        above ``tau``, rendering the problem **infeasible**.

    The objective is ``minimize trace(X)``: the optimum equals
    ``sum(o_i)`` when feasible, and recovers the dimension bound on the
    operator spectrum.

    For a degenerate input with zero active generators (no rule's kind
    projector fires anywhere on the substrate — pathological), we
    allocate a 1x1 trivial PSD variable so the SDP shim has something
    to solve over; the result is feasible at value 0 (no obstruction).
    """
    diag = list(typing_constraints.obstruction_diag)
    tau = float(typing_constraints.truncation_gap)
    if len(diag) == 0:
        # Trivial bootstrap: no active generator. Single 1x1 PSD scalar
        # with no diagonal constraint, trace bound 0. Feasible at 0.
        var = PSDVariable(name="X", size=1)

        def _trace_le_zero(vars_dict):
            return cp.trace(vars_dict["X"]) <= tau

        def _obj(vars_dict):
            return cp.trace(vars_dict["X"])

        return SDPProblem(
            variables=(var,),
            constraints=(LinearConstraint(
                builder=_trace_le_zero, name="trace<=tau"),),
            objective_builder=_obj,
            minimize=True,
            name="bootstrap_trivial",
        )

    k = len(diag)
    var = PSDVariable(name="X", size=k)

    def _make_diag_eq(i: int, value: float):
        def _builder(vars_dict):
            return vars_dict["X"][i, i] == value
        return LinearConstraint(
            builder=_builder, name=f"X[{i},{i}]=={value:.3e}")

    constraints: list[LinearConstraint] = [
        _make_diag_eq(i, diag[i]) for i in range(k)
    ]

    def _trace_bound(vars_dict):
        return cp.trace(vars_dict["X"]) <= tau

    constraints.append(LinearConstraint(
        builder=_trace_bound, name="trace<=tau"))

    def _objective(vars_dict):
        return cp.trace(vars_dict["X"])

    return SDPProblem(
        variables=(var,),
        constraints=tuple(constraints),
        objective_builder=_objective,
        minimize=True,
        name="bootstrap_type_only",
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def solve_bootstrap(
    problem: SDPProblem,
    *,
    truncation_gap: float = DEFAULT_TRUNCATION_GAP,
    solver: str | None = None,
) -> BootstrapResult:
    """Solve the bootstrap SDP via the S2 :func:`solve_sdp` wrapper.

    Returns a :class:`BootstrapResult` containing the solver status,
    the optimum trace, the recovered dimension bound (max eigenvalue
    of the optimal ``X``, ``+inf`` when infeasible), and the truncation
    gap used. Honors the ``truncation_gap`` argument for downstream
    surfacing — the actual gap is whatever was baked into ``problem``
    at build time; this field is informational.

    No mock optimizer (§1.1 anti-shortcut): the SDP is dispatched to
    the real CVXPY-backed solver.
    """
    solution: SDPSolution = solve_sdp(problem, solver=solver)
    feasible = solution.status in (SDPStatus.OPTIMAL, SDPStatus.INACCURATE)

    if feasible and "X" in solution.matrices:
        X = solution.matrices["X"]
        # Hermitize before eigendecomp — CVXPY's symmetric variable can
        # carry tiny asymmetry from the SCS interior solver.
        X_sym = 0.5 * (X + X.T)
        eigs = np.linalg.eigvalsh(X_sym)
        dim_bound = float(max(eigs.max(), 0.0))
    elif feasible:
        # Optimal but no matrix recovered (e.g. trivial 1x1 zero) — bound 0.
        dim_bound = 0.0
    else:
        dim_bound = float("inf")

    return BootstrapResult(
        status=solution.status,
        optimal_value=solution.optimal_value,
        dimension_bound=dim_bound,
        truncation_gap=truncation_gap,
        feasible=feasible,
    )


# ---------------------------------------------------------------------------
# Top-level §12.4 acceptance
# ---------------------------------------------------------------------------


def verify_typing_via_anomaly_sdp(
    state: MERA,
    typing_H: MeraTypingHamiltonian,
    *,
    truncation_gap: float = DEFAULT_TRUNCATION_GAP,
    kind_activity_floor: float = DEFAULT_KIND_ACTIVITY_FLOOR,
    solver: str | None = None,
) -> BootstrapVerification:
    """Typing-feasibility SDP wrapper around §12.1 anomaly diagonals.

    D13-honest naming: this is NOT the full §12.4 bootstrap. It is an
    SDP whose diagonal is fixed by the §12.1 anomaly trace; feasibility
    is exactly equivalent to "every anomaly diagonal fits under the
    truncation gap." That makes it a sound necessary-but-not-sufficient
    bound on the full §12.4 acceptance — termination, depth, complexity,
    and parametricity bounds are deferred to the substrate-wide
    crossing-equation OPE-truncated bootstrap (EXTENSIONS.md).

    Feasibility verdict:
      * Well-typed program (``H_typing |psi> = 0``) → bootstrap is
        **feasible**, dimension bound ~ 0.
      * Ill-typed program (``H_typing |psi> > 0`` at some rule) →
        bootstrap is **infeasible** (the §12.4 impossibility
        certificate, distinct from §12.1's anomaly trace which fires
        positive on the same substrate signature but is read off as a
        trace rather than an SDP infeasibility).

    Operator-algebraic pipeline (§1.1):
      1. Extract symmetry generators from ``typing_H`` via the §12.1
         ``extract_symmetries`` — the substrate-derived leaf-projector
         triples, NOT AST-walked.
      2. For each generator with ``<P_kind> > kind_activity_floor``
         (the rule's hypothesis fires on the node), read the
         obstruction expectation ``o = <P_kind . (I - P_type) . [P_value]>``
         via the factored window primitive.
      3. Build the bootstrap SDP via :func:`build_bootstrap_problem`.
      4. Solve via the real S2 solver :func:`solve_bootstrap`.
      5. Verdict: ``well_typed = feasible``.

    The ``obstructions`` field of the returned verification lists
    every active generator's obstruction so callers can localize the
    violation (same (rule_id, node) addressing the §12.1 anomaly
    certificate uses; the two diagnostics are complementary — §12.1
    surfaces the obstruction as a trace, §12.4 surfaces it as an SDP
    infeasibility witness).
    """
    generators = extract_symmetries(typing_H)
    diag: list[float] = []
    refs: list[tuple[str, int]] = []
    obstructions: list[tuple[str, int, float]] = []
    for G in generators:
        # Activity gate: only include rules whose hypothesis fires on
        # the node. Cheap pre-filter, keeps the SDP small.
        if _kind_expectation(G, state) < kind_activity_floor:
            continue
        # §12.1-style obstruction: <P_kind . (I - P_type) . [P_value]>.
        # Routed through compute_anomaly so the operator-algebraic
        # encoding-collision guard (anomaly.py:297) is preserved.
        o = _obstruction_expectation(G, state)
        diag.append(o)
        refs.append((G.rule_id, G.node))
        obstructions.append((G.rule_id, G.node, o))

    tc = _TypingConstraints(
        obstruction_diag=tuple(diag),
        generator_refs=tuple(refs),
        truncation_gap=truncation_gap,
    )
    problem = build_bootstrap_problem(tc)
    result = solve_bootstrap(
        problem, truncation_gap=truncation_gap, solver=solver)

    # Deterministic sort: descending obstruction, then by (rule, node).
    obstructions.sort(key=lambda r: (-r[2], r[0], r[1]))

    return BootstrapVerification(
        well_typed=result.feasible,
        result=result,
        obstructions=tuple(obstructions),
    )


__all__ = [
    "BootstrapResult",
    "BootstrapVerification",
    "DEFAULT_TRUNCATION_GAP",
    "DEFAULT_KIND_ACTIVITY_FLOOR",
    "build_bootstrap_problem",
    "solve_bootstrap",
    "verify_typing_via_anomaly_sdp",
]
