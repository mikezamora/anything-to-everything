"""Evaluation metrics (spec §14.2).

Each metric is a pure function over ``BenchmarkResult`` (or its raw
``ProofAttempt`` list). They do NOT mutate inputs; the runner attaches
the computed value to ``BenchmarkResult.metrics``.
"""

from .pass_at_k import pass_at_k
from .type_at_k import type_at_k
from .residual_energy_at_1 import residual_energy_at_1
from .capability_curve import capability_curve

__all__ = ["pass_at_k", "type_at_k", "residual_energy_at_1", "capability_curve"]
