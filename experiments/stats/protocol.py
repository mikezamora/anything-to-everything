"""Spec-mandated statistical pipeline (spec §14.5).

Given a list of ``BenchmarkResult`` (one per (solver, benchmark,
configuration)), this module:

  1. Computes pass@k / type@k / residual_energy@1 / capability_curve
     for each result (delegating to ``metrics``).
  2. Bootstraps a 95% CI on each metric.
  3. For each pair (qpcn vs baseline), computes a paired t-test (via
     scipy.stats) and Cohen's d + Cliff's delta.
  4. Applies Holm-Bonferroni correction across all comparisons in a
     given benchmark.

The output is a structured ``StatisticalReport`` consumed by
``runner.write_report``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..metrics import pass_at_k, type_at_k, residual_energy_at_1
from ..schema import BenchmarkResult, ProofAttempt
from .bootstrap import BootstrapCI, bootstrap_ci
from .effect_size import cliffs_delta, cohens_d
from .multiple_comparisons import CorrectedPValue, holm_bonferroni


@dataclass(frozen=True)
class MetricsBlock:
    pass_at_1: float
    pass_at_k_val: float
    type_at_1: float
    residual_energy_mean: float
    residual_energy_ci: BootstrapCI
    pass_at_1_ci: BootstrapCI


@dataclass(frozen=True)
class PairwiseComparison:
    solver_a: str
    solver_b: str
    metric: str
    mean_a: float
    mean_b: float
    cohens_d: float
    cliffs_delta: float


@dataclass
class StatisticalReport:
    per_run: dict[str, MetricsBlock] = field(default_factory=dict)
    comparisons: list[PairwiseComparison] = field(default_factory=list)
    corrected_p: list[CorrectedPValue] = field(default_factory=list)


def _per_problem_solved(attempts: Sequence[ProofAttempt]) -> list[float]:
    return [1.0 if a.solved else 0.0 for a in attempts]


def run_statistical_protocol(
    results: Sequence[BenchmarkResult],
    *,
    k: int = 10,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 42,
) -> StatisticalReport:
    """Apply the spec §14.5 pipeline to a list of BenchmarkResults."""
    report = StatisticalReport()

    # ---- per-run metric blocks -----------------------------------------
    for r in results:
        atts = r.attempts
        p1 = pass_at_k(atts, k=1)
        pk = pass_at_k(atts, k=k)
        t1 = type_at_k(atts, k=1)
        rstats = residual_energy_at_1(atts)
        per_problem = _per_problem_solved(atts)
        p1_ci = bootstrap_ci(
            per_problem,
            statistic=lambda x: float(np.mean(x)),
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
        if rstats.samples:
            r_ci = bootstrap_ci(
                rstats.samples,
                statistic=lambda x: float(np.mean(x)),
                n_resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            )
        else:
            r_ci = bootstrap_ci(
                [], n_resamples=bootstrap_resamples, seed=bootstrap_seed,
            )
        block = MetricsBlock(
            pass_at_1=p1, pass_at_k_val=pk, type_at_1=t1,
            residual_energy_mean=rstats.mean,
            residual_energy_ci=r_ci, pass_at_1_ci=p1_ci,
        )
        key = f"{r.benchmark}/{r.solver}/{r.config_label}"
        report.per_run[key] = block

    # ---- pairwise comparisons (qpcn vs each baseline, per benchmark) ---
    # Group by benchmark.
    by_bench: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        by_bench.setdefault(r.benchmark, []).append(r)

    p_values: list[tuple[str, float]] = []
    for bench, runs in by_bench.items():
        # Identify the qpcn baseline (config_label=="BASELINE", or sole qpcn).
        qpcn_run = next(
            (r for r in runs if r.solver == "qpcn"),
            None,
        )
        if qpcn_run is None:
            continue
        a_solved = _per_problem_solved(qpcn_run.attempts)
        for other in runs:
            if other is qpcn_run:
                continue
            b_solved = _per_problem_solved(other.attempts)
            if not a_solved or not b_solved:
                continue
            d = cohens_d(a_solved, b_solved)
            delta = cliffs_delta(a_solved, b_solved)
            report.comparisons.append(PairwiseComparison(
                solver_a=qpcn_run.solver,
                solver_b=other.solver,
                metric=f"{bench}/solved",
                mean_a=float(np.mean(a_solved)) if a_solved else 0.0,
                mean_b=float(np.mean(b_solved)) if b_solved else 0.0,
                cohens_d=d,
                cliffs_delta=delta,
            ))
            # Paired-t p-value (only if sizes match).
            if len(a_solved) == len(b_solved) and len(a_solved) >= 2:
                from scipy import stats as scipy_stats
                t_stat, p_val = scipy_stats.ttest_rel(a_solved, b_solved)
                # Convert NaN (zero-variance) to 1.0 (no signal).
                if np.isnan(p_val):
                    p_val = 1.0
                p_values.append(
                    (f"{bench}/{qpcn_run.solver}-vs-{other.solver}", float(p_val)),
                )

    if p_values:
        report.corrected_p = holm_bonferroni(p_values)
    return report
