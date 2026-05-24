"""Shared data classes for the evaluation framework (spec §14).

Every benchmark module ingests upstream corpora into ``ProblemSpec``.
Every solver adapter (QPCN or baseline) emits ``ProofAttempt``. Metric
modules consume those. This is the *only* surface they share -- the
framework's internal lingua franca, kept minimal and frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ProblemSpec:
    """A single benchmark problem.

    ``domain``       - "proof" | "synthesis" | "chemistry".
    ``problem_id``   - corpus-unique identifier.
    ``statement``    - human-readable target (e.g. theorem statement,
                       function signature + I/O examples, molecule SMILES).
    ``payload``      - corpus-specific structured representation. For
                       miniF2F: Lean source. For HumanEval: prompt +
                       canonical solution + tests. For synthesis:
                       SynthesisProblem name or builder. For QM9:
                       atom list + bond list. Solver adapters decode
                       this; metric modules do not touch it.
    ``difficulty``   - numeric difficulty bucket [0.0, 1.0] used by
                       ``capability_curve``. Per spec §14.2: "capability
                       vs problem-difficulty curve".
    ``tags``         - set of corpus tags (e.g. {"typed_subset",
                       "negative_comparison"} for HumanEval typed subset).
    """

    domain: str
    problem_id: str
    statement: str
    payload: dict
    difficulty: float = 0.5
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProofAttempt:
    """A solver's response to one ``ProblemSpec``.

    ``solver``           - "qpcn" | "alphaproof" | "reprover" | "synquid"
                           | "gpt4" | "claude" | etc.
    ``problem_id``       - the ProblemSpec.problem_id this was for.
    ``solved``           - did the solver report a successful proof /
                           well-typed synthesis / correct prediction?
    ``well_typed``       - did the candidate type-check? This is what
                           ``type@k`` consumes. The QPCN should pin this
                           to True whenever ``solved`` is True (spec
                           §13.2 / §14.2 "type@1 = 1.0").
    ``residual_energy``  - substrate residual on the top candidate.
                           Zero = provably correct (§14.2). For non-QPCN
                           baselines this is None.
    ``candidates``       - top-k candidate strings (or AST reprs). For
                           ``pass@k`` evaluation: the metric checks
                           whether any of them satisfies the problem's
                           examples.
    ``wall_time_s``      - wall-clock budget consumed.
    ``error``            - structured error trace when the solver
                           refused / failed / timed out. None on success.
    ``diagnostics``      - free-form per-solver telemetry (e.g. proof
                           length, library size at solve time, MERA bond
                           reached, spectral_gap).
    """

    solver: str
    problem_id: str
    solved: bool
    well_typed: bool
    residual_energy: Optional[float]
    candidates: tuple[str, ...]
    wall_time_s: float
    error: Optional[str] = None
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate of one (solver, benchmark, configuration) run.

    Holds the raw per-problem attempts plus pre-computed metric values
    (filled in by the metric modules). The runner aggregates these into
    the report.
    """

    solver: str
    benchmark: str
    config_label: str
    attempts: tuple[ProofAttempt, ...]
    metrics: dict = field(default_factory=dict)
    seed: int = 0
