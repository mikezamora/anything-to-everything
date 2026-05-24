"""§13.8 Capability-growth-law empirical validation (A4).

Spec reference: ``QFT_PCN_ARCHITECTURE.md`` §13.8 (Conjecture: wake-sleep
capability satisfies ``dC/dt = alpha * (1 - C) - beta * C``).

This module fits the conjecture against an empirical capability trajectory
``C(t)`` produced by running real wake-sleep cycles on a corpus of theorems.
The capability ``C`` is the fraction of corpus problems the library can
"solve" -- where "solve" is the operator-algebraic check inherited from the
§10.9 abstraction-discovery substrate: a problem is solved if (a) the
library has a cached state with that source-id, OR (b) one of the problem's
mined subtree reduced densities is within trace-distance threshold of a
registered primitive's canonical density (the same matching logic
``wake_sleep._consolidate`` uses to subsume cached lemmas).

This is the §1.1 / §1.6 operator-algebraic capability check: matching is
on reduced densities under the trace-distance metric, never a classical
AST fingerprint or hash.

The closed-form solution of the ODE ``dC/dt = alpha*(1-C) - beta*C`` with
initial value ``C(0) = C_0`` is

    C(t) = alpha / (alpha + beta) * (1 - exp(-(alpha+beta) * t))
           + C_0 * exp(-(alpha + beta) * t)

so the fit reduces to a two-parameter nonlinear least squares on
``(alpha, beta)`` given measured ``(t, C(t))`` samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares

from src.qft_pcn.composition._abstraction_const import (
    DEFAULT_DISTANCE_THRESHOLD,
)
from src.qft_pcn.composition.abstraction import trace_distance
from src.qft_pcn.composition.subtree_miner import MineConfig, mine_subtrees
from src.qft_pcn.composition.wake_sleep import (
    Problem, WakeSleepConfig, wake_sleep_cycle,
)


__all__ = [
    "GrowthFit",
    "capability_curve",
    "measure_capability",
    "record_growth_trajectory",
    "fit_growth_law",
]


@dataclass(frozen=True)
class GrowthFit:
    """Result of fitting ``dC/dt = alpha*(1-C) - beta*C`` to a trajectory."""
    alpha: float
    beta: float
    ssr: float                # sum of squared residuals
    c_infinity: float         # alpha / (alpha + beta)
    c0: float                 # measured C(0) used as initial condition
    n_points: int             # number of (t, C) samples


# ---------------------------------------------------------------------------
# capability measurement
# ---------------------------------------------------------------------------


def _registered_primitives(library) -> list:
    """Return the library's registered primitives, regardless of backend.

    Both ``FakeLemmaLibrary`` (tests) and ``LemmaLibraryAdapter``
    (production) expose ``registered_primitives`` as a list attribute.
    A library without this attribute is treated as having no
    primitives (capability falls back to cached-id matching alone).
    """
    prims = getattr(library, "registered_primitives", None)
    if prims is None:
        return []
    return list(prims)


def _problem_is_solved(state, meta, source_id: str, library,
                       *, mine: MineConfig,
                       distance_threshold: float) -> bool:
    """Operator-algebraic check: does the library solve this problem?

    Two routes:

    (a) The library's cached_solutions already contains a state registered
        under this ``source_id`` (wake-phase solve already covered it).

    (b) One of the problem's mined subtree reduced densities is within
        trace-distance ``distance_threshold`` of a registered primitive's
        canonical density. This is the same subsume-matching logic used
        by :func:`wake_sleep._consolidate` (§10.9) to decide when a cached
        parent lemma can be replaced by a shorter proof that delegates to
        a discovered primitive: if the match holds, the primitive is a
        sub-lemma that "solves" the relevant substructure.
    """
    # Route (a): cache hit.
    try:
        cached = list(library.cached_solutions())
    except Exception:
        cached = []
    for _, _, cached_sid in cached:
        if cached_sid == source_id:
            return True

    # Route (b): subtree rho within distance of a registered primitive.
    prims = _registered_primitives(library)
    if not prims:
        return False
    try:
        cands = mine_subtrees(state, meta, source_id, mine)
    except Exception:
        return False
    for cand in cands:
        for prim in prims:
            if trace_distance(cand.rho, prim.rho_canonical) < distance_threshold:
                return True
    return False


def measure_capability(library, corpus: Sequence[tuple],
                       *,
                       mine: MineConfig | None = None,
                       distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
                       ) -> float:
    """Fraction of corpus problems the library can solve (snapshot).

    Parameters
    ----------
    library:
        A wake-sleep-compatible library (FakeLemmaLibrary or
        LemmaLibraryAdapter). Must expose ``cached_solutions()`` and
        ideally ``registered_primitives``.
    corpus:
        Iterable of ``(state, meta, source_id)`` triples in the same shape
        as ``build_induction_corpus`` / ``library.cached_solutions``.
    mine:
        Optional :class:`MineConfig`. Defaults to the standard config
        used by ``WakeSleepConfig``.
    distance_threshold:
        Trace-distance threshold for the route-(b) subsume match. Defaults
        to ``DEFAULT_DISTANCE_THRESHOLD`` (= 0.15) matching the §10.9
        consolidation threshold.

    Returns
    -------
    float
        ``C in [0, 1]``: ``solved_count / len(corpus)``. Empty corpus
        returns ``0.0``.
    """
    corpus = list(corpus)
    if not corpus:
        return 0.0
    if mine is None:
        mine = MineConfig()
    solved = 0
    for state, meta, sid in corpus:
        if _problem_is_solved(state, meta, sid, library,
                              mine=mine,
                              distance_threshold=distance_threshold):
            solved += 1
    return solved / len(corpus)


# ---------------------------------------------------------------------------
# trajectory recording
# ---------------------------------------------------------------------------


def record_growth_trajectory(library, corpus: Sequence[tuple], n_cycles: int,
                             *,
                             solve,
                             config: WakeSleepConfig | None = None,
                             ) -> list[tuple[int, float]]:
    """Run ``n_cycles`` wake-sleep cycles, returning ``(t, C(t))`` after each.

    The capability is also measured at ``t = 0`` (before any cycle runs),
    so the returned list has length ``n_cycles + 1``.

    Parameters
    ----------
    library:
        A wake-sleep-compatible library (will be MUTATED by the cycles).
    corpus:
        Iterable of ``(state, meta, source_id)`` problem triples. The same
        corpus is presented to every cycle (a stable target class, per
        the §13.8 conjecture's setup).
    n_cycles:
        Number of wake-sleep cycles to run. Must be >= 1.
    solve:
        ``SolveFn`` passed to :func:`wake_sleep_cycle`. Typically the
        stub solver from the composition test conftest.
    config:
        Optional :class:`WakeSleepConfig`; defaults to the standard one.

    Returns
    -------
    list[tuple[int, float]]
        ``[(0, C0), (1, C1), ..., (n_cycles, C_n)]``.
    """
    if n_cycles < 1:
        raise ValueError(f"n_cycles must be >= 1 (got {n_cycles!r})")
    corpus = list(corpus)
    if config is None:
        config = WakeSleepConfig()

    problems = [Problem(id=sid, hamiltonian_or_state=(state, meta))
                for state, meta, sid in corpus]

    trajectory: list[tuple[int, float]] = []
    c0 = measure_capability(library, corpus,
                            mine=config.mine,
                            distance_threshold=config.cluster.distance_threshold)
    trajectory.append((0, c0))

    for t in range(1, n_cycles + 1):
        wake_sleep_cycle(library, problems, solve, cycle_index=t - 1,
                         config=config)
        c_t = measure_capability(
            library, corpus,
            mine=config.mine,
            distance_threshold=config.cluster.distance_threshold,
        )
        trajectory.append((t, c_t))

    return trajectory


# ---------------------------------------------------------------------------
# ODE fit
# ---------------------------------------------------------------------------


def capability_curve(t: np.ndarray, alpha: float, beta: float,
                     c0: float) -> np.ndarray:
    """Closed-form solution of ``dC/dt = alpha*(1-C) - beta*C``.

        C(t) = alpha/(alpha+beta) * (1 - exp(-(alpha+beta)*t))
               + C_0 * exp(-(alpha+beta)*t)

    With ``alpha + beta -> 0`` the limit is ``C(t) = c0 + alpha * t``
    (linear regime); we handle that branch explicitly so the fitter does
    not divide by zero at the initial guess.
    """
    t = np.asarray(t, dtype=float)
    s = alpha + beta
    if abs(s) < 1e-12:
        return c0 + alpha * t
    c_inf = alpha / s
    return c_inf * (1.0 - np.exp(-s * t)) + c0 * np.exp(-s * t)


def fit_growth_law(trajectory: Sequence[tuple[float, float]],
                   *,
                   alpha0: float = 0.5,
                   beta0: float = 0.1,
                   ) -> GrowthFit:
    """Fit ``(alpha, beta)`` to a measured trajectory via scipy least_squares.

    The fit uses the closed-form ``capability_curve`` rather than
    integrating the ODE numerically: this is the exact analytic solution
    for constant ``(alpha, beta)``, so the residuals are clean.

    ``alpha`` and ``beta`` are constrained non-negative (physical: both
    are rates).

    Parameters
    ----------
    trajectory:
        Sequence of ``(t, C(t))`` samples. The ``C(0)`` value (smallest t)
        is used as the initial condition; the remaining points drive the
        fit.
    alpha0, beta0:
        Initial guess for the nonlinear solve. Defaults are tuned for
        typical wake-sleep dynamics on the §10.10 induction corpus.

    Returns
    -------
    GrowthFit
        Fitted parameters + sum-of-squared-residuals.
    """
    if len(trajectory) < 2:
        raise ValueError(
            f"need >= 2 trajectory points to fit (got {len(trajectory)})"
        )
    pts = sorted(trajectory, key=lambda tc: tc[0])
    t_arr = np.array([t for t, _ in pts], dtype=float)
    c_arr = np.array([c for _, c in pts], dtype=float)
    c0 = float(c_arr[0])

    def residuals(params):
        alpha, beta = params
        return capability_curve(t_arr, alpha, beta, c0) - c_arr

    # Bound both rates non-negative (with an upper bound of 1e3 to keep the
    # solver well-conditioned; rates above ~1 per cycle are unphysical).
    result = least_squares(
        residuals,
        x0=np.array([alpha0, beta0], dtype=float),
        bounds=([0.0, 0.0], [1.0e3, 1.0e3]),
        method="trf",
    )
    alpha_fit = float(result.x[0])
    beta_fit = float(result.x[1])
    ssr = float(np.sum(result.fun ** 2))
    c_inf = alpha_fit / (alpha_fit + beta_fit) if (alpha_fit + beta_fit) > 0 \
        else float("nan")
    return GrowthFit(
        alpha=alpha_fit,
        beta=beta_fit,
        ssr=ssr,
        c_infinity=c_inf,
        c0=c0,
        n_points=len(pts),
    )
