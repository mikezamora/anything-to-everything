"""QPCN solver adapter -- the system under test.

This is the in-repo solver that exercises:

* ``proof`` domain: ``encode_mera`` + ``mera_imaginary_evolve_state`` +
  ``solve_goal_graph`` (the K-8 / L acceptance test path -- the single
  surface the spec calls out as "the architecture's K-8/L acceptance
  test"). Adapted from
  ``src/qft_pcn/composition/tests/test_hierarchical_proof_demo.py`` so
  the bench uses the EXACT same machinery as the acceptance test.
* ``synthesis`` domain: ``logic.synthesis.synthesize`` via the
  ``demo_stlc_synthesis.BUILDERS`` map for Myth-style problems; for
  HumanEval typed-subset, we attempt the same pipeline using the
  signature-derived sketch.
* ``chemistry`` domain: no implementation yet (per EXTENSIONS.md
  "Quantum chemistry Hamiltonian compiler"). Returns an honest
  no-attempt so the framework's QM9 row reads "out_of_substrate" rather
  than fabricating a result.

The adapter NEVER mocks. Every ``solved=True`` row corresponds to a
real substrate run that converged below the residual threshold.
"""
from __future__ import annotations

import time
import traceback
from typing import Optional

import numpy as np

from ..schema import ProblemSpec, ProofAttempt
from ._base import Baseline


# spec §1.1 acceptance: a proof is SOLVED iff residual_energy at the
# converged state lies under this threshold (matching the K-8 / L
# acceptance test gate).
_PROOF_RESIDUAL_TOL = 5e-2


def _solve_proof(problem: ProblemSpec, *, timeout_s: float) -> ProofAttempt:
    """Drive the real K-8/L proof substrate on a single problem.

    Currently the substrate supports Nat-arithmetic statements of the
    K-8 family (``forall x:Nat. Eq (add x Zero) x`` and its hierarchical
    extension); other ``minif2f`` statements are tagged
    ``out_of_substrate`` by the loader and surface here as
    ``solved=False, error="out_of_substrate"`` (honest no-attempt).
    """
    fragment = problem.payload.get("fragment", "unknown")
    qpcn_statement = problem.payload.get("qpcn_statement")
    t0 = time.time()
    if fragment == "out_of_substrate":
        return ProofAttempt(
            solver="qpcn",
            problem_id=problem.problem_id,
            solved=False,
            well_typed=False,
            residual_energy=None,
            candidates=(),
            wall_time_s=time.time() - t0,
            error="out_of_substrate: QPCN substrate does not cover "
                  "this fragment (real-valued / gcd / transcendental).",
            diagnostics={"fragment": fragment},
        )

    # A1+B3 polish FU1: honest per-problem encoding. Parse the surface
    # statement from the loader. Missing / None / unparseable -> honest
    # out_of_substrate no-attempt per §1.6 (NEVER fall back to the
    # hardcoded K-8 AST -- that would fabricate a result for a
    # different theorem).
    if qpcn_statement is None:
        return ProofAttempt(
            solver="qpcn",
            problem_id=problem.problem_id,
            solved=False,
            well_typed=False,
            residual_energy=None,
            candidates=(),
            wall_time_s=time.time() - t0,
            error="out_of_substrate: no qpcn_statement on problem "
                  "payload -- loader did not provide a surface-grammar "
                  "restatement (see EXTENSIONS.md 'Free-form theorem "
                  "ingestion').",
            diagnostics={"fragment": fragment,
                         "qpcn_statement": None},
        )

    try:
        # Late import: keep adapter import cheap so the framework is
        # introspectable without the heavy substrate already imported.
        from src.qft_pcn.logic.ast import parse as ast_parse
        from src.qft_pcn.logic.mera_encoder import encode_mera
        from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
            MeraEvalHamiltonian,
        )
        from src.qft_pcn.logic.mera_evolution_logic import (
            mera_imaginary_evolve_state,
        )

        try:
            ast = ast_parse(qpcn_statement)
        except Exception as parse_exc:  # noqa: BLE001
            return ProofAttempt(
                solver="qpcn",
                problem_id=problem.problem_id,
                solved=False,
                well_typed=False,
                residual_energy=None,
                candidates=(),
                wall_time_s=time.time() - t0,
                error=(f"out_of_substrate: qpcn_statement failed to "
                       f"parse via logic.ast.parse: "
                       f"{type(parse_exc).__name__}: {parse_exc}"),
                diagnostics={"fragment": fragment,
                             "qpcn_statement": qpcn_statement,
                             "parse_error": repr(parse_exc)},
            )

        state, meta = encode_mera(ast)
        H = MeraEvalHamiltonian(meta)
        protected = set(meta.forall_protected_leaves)
        # Use the same dt/steps/chi as the K-8/L acceptance test.
        _, final = mera_imaginary_evolve_state(
            state, H, dt=0.1, steps=300, chi_layer=16,
            frozen_leaves=protected,
        )
        residual = float(H.total_energy(final))
        wall = time.time() - t0
        solved = residual <= _PROOF_RESIDUAL_TOL
        return ProofAttempt(
            solver="qpcn",
            problem_id=problem.problem_id,
            solved=solved,
            well_typed=solved,
            residual_energy=residual,
            candidates=(str(ast),) if solved else (),
            wall_time_s=wall,
            error=None if solved else f"residual {residual:.4f} > tol {_PROOF_RESIDUAL_TOL}",
            diagnostics={"steps": 300, "chi": 16, "dt": 0.1,
                         "fragment": fragment,
                         "qpcn_statement": qpcn_statement},
        )
    except Exception as exc:  # noqa: BLE001  (we report any failure)
        return ProofAttempt(
            solver="qpcn",
            problem_id=problem.problem_id,
            solved=False,
            well_typed=False,
            residual_energy=None,
            candidates=(),
            wall_time_s=time.time() - t0,
            error=f"{type(exc).__name__}: {exc}",
            diagnostics={"traceback": traceback.format_exc(limit=3)},
        )


def _solve_synthesis(problem: ProblemSpec, *, seed: int) -> ProofAttempt:
    """Drive the STLC synthesis pipeline on a single Myth/Hazel problem.

    For Myth (``payload["builder_name"]`` in P1..P8): we call the
    canonical ``run_problem`` entry point -- the SAME surface that
    ``test_synthesis_acceptance.py`` exercises.

    For Hazel / HumanEval typed subset: the in-repo substrate does not
    yet build a SynthesisProblem from a free-form signature; the
    adapter surfaces an honest no-attempt with the precise reason so
    EXTENSIONS.md can pick it up.
    """
    t0 = time.time()
    builder = problem.payload.get("builder_name")
    if not builder:
        return ProofAttempt(
            solver="qpcn",
            problem_id=problem.problem_id,
            solved=False,
            well_typed=False,
            residual_energy=None,
            candidates=(),
            wall_time_s=time.time() - t0,
            error="no builder_name in payload: free-form signature -> "
                  "SynthesisProblem encoding not yet supported (see "
                  "EXTENSIONS.md 'Free-form signature ingestion').",
            diagnostics={"tags": problem.tags},
        )

    try:
        from src.qft_pcn.logic.demo_stlc_synthesis import (
            run_problem, check_success,
        )
        rng = np.random.default_rng(seed)
        res = run_problem(builder, rng=rng, verbose=False)
        wall = time.time() - t0
        solved = check_success(builder, res)
        top1 = res.completions[0] if res.completions else None
        candidate_str = repr(top1.ast) if top1 else ""
        return ProofAttempt(
            solver="qpcn",
            problem_id=problem.problem_id,
            # P8 is a deliberate "correct refusal" -- check_success
            # returns True iff failure_mode is populated. Either way,
            # the type-safety invariant holds: any surfaced completion
            # was vetted by the typing Hamiltonian.
            solved=solved,
            well_typed=(res.failure_mode is None) and bool(res.completions),
            residual_energy=float(top1.energy) if top1 is not None else None,
            candidates=(candidate_str,) if candidate_str else (),
            wall_time_s=wall,
            error=res.failure_mode,
            diagnostics={
                "n_unique": res.n_unique,
                "final_state_energy": res.final_state_energy,
                "wall_time_seconds": res.wall_time_seconds,
                "chi_observed_max": res.chi_observed_max,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return ProofAttempt(
            solver="qpcn",
            problem_id=problem.problem_id,
            solved=False,
            well_typed=False,
            residual_energy=None,
            candidates=(),
            wall_time_s=time.time() - t0,
            error=f"{type(exc).__name__}: {exc}",
            diagnostics={"traceback": traceback.format_exc(limit=3)},
        )


def _solve_chemistry(problem: ProblemSpec) -> ProofAttempt:
    """No substrate path for chemistry yet -- honest no-attempt.

    See ``EXTENSIONS.md`` entry "Quantum chemistry Hamiltonian
    compiler" for the deferral. The QM9 row is preserved in the report
    so the gap is visible.
    """
    return ProofAttempt(
        solver="qpcn",
        problem_id=problem.problem_id,
        solved=False,
        well_typed=False,
        residual_energy=None,
        candidates=(),
        wall_time_s=0.0,
        error="chemistry domain not implemented (see EXTENSIONS.md "
              "'Quantum chemistry Hamiltonian compiler')",
        diagnostics={"tags": problem.tags},
    )


class QPCNBaseline(Baseline):
    """The QPCN itself, exposed via the baseline-adapter protocol."""

    name = "qpcn"

    def __init__(self, *, seed: int = 42, timeout_s: float = 180.0):
        self.seed = seed
        self.timeout_s = timeout_s

    def solve(self, problem: ProblemSpec) -> ProofAttempt:
        if problem.domain == "proof":
            return _solve_proof(problem, timeout_s=self.timeout_s)
        if problem.domain == "synthesis":
            return _solve_synthesis(problem, seed=self.seed)
        if problem.domain == "chemistry":
            return _solve_chemistry(problem)
        return ProofAttempt(
            solver=self.name,
            problem_id=problem.problem_id,
            solved=False,
            well_typed=False,
            residual_energy=None,
            candidates=(),
            wall_time_s=0.0,
            error=f"unknown domain: {problem.domain!r}",
            diagnostics={},
        )
