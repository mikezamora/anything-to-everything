"""QPCN evaluation framework (spec §14 + §14.1).

This package implements the *empirical apparatus* that turns the QPCN
substrate into a research contribution:

* ``benchmarks/`` - corpus ingestion (miniF2F, HumanEval typed subset,
  Myth, DreamCoder, QM9, Hazel).
* ``metrics/``    - ``pass@k``, ``type@k``, ``residual_energy@1``,
  ``capability_curve``.
* ``baselines/``  - AlphaProof, ReProver, Synquid adapter shims with
  loud-fail behaviour when the upstream binary is unavailable (tracked
  in ``EXTENSIONS.md`` per the no-placeholders directive).
* ``ablations/``  - A1-A8 configuration matrix per spec §14.4.
* ``stats/``      - bootstrap CI, Cohen's d / Cliff's delta,
  Holm-Bonferroni correction, spec-mandated protocol pipeline.
* ``runner.py``   - top-level orchestrator: load corpus, run QPCN +
  baselines, collect metrics, apply stats, write report.

The runner exercises the real QPCN substrate (encode_mera +
mera_imaginary_evolve_state + solve_goal_graph for proofs; the STLC
synthesis pipeline for code synthesis). Nothing is mocked.

Pre-registration (spec §14.7) lives alongside each campaign in
``reports/preregistration_<campaign>.md`` and is written BEFORE the
runner is invoked.
"""

__all__ = [
    "ProblemSpec",
    "ProofAttempt",
    "BenchmarkResult",
]

from .schema import ProblemSpec, ProofAttempt, BenchmarkResult
