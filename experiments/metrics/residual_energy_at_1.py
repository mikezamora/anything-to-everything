"""residual_energy@1 (spec §14.2).

`residual_energy@1`: distribution of substrate residual energy on the
top-1 candidate. Zero = provably correct; nonzero = quantified
uncertainty. The runner reports mean + std + the full sample so the
report can compute bootstrap CIs.
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable, NamedTuple

from ..schema import ProofAttempt


class ResidualEnergyStats(NamedTuple):
    mean: float
    std: float
    median: float
    minimum: float
    maximum: float
    n: int
    samples: tuple[float, ...]


def residual_energy_at_1(attempts: Iterable[ProofAttempt]) -> ResidualEnergyStats:
    """Compute residual_energy@1 stats over a collection of attempts.

    Attempts where ``residual_energy is None`` (e.g. LLM baseline with
    no substrate residual) are excluded from the sample; the count is
    surfaced via ``ResidualEnergyStats.n`` so the caller can see how
    many real observations underpin the mean.
    """
    samples = tuple(
        float(a.residual_energy)
        for a in attempts
        if a.residual_energy is not None and math.isfinite(a.residual_energy)
    )
    if not samples:
        return ResidualEnergyStats(
            mean=float("nan"), std=float("nan"), median=float("nan"),
            minimum=float("nan"), maximum=float("nan"), n=0, samples=(),
        )
    return ResidualEnergyStats(
        mean=statistics.fmean(samples),
        std=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        median=statistics.median(samples),
        minimum=min(samples),
        maximum=max(samples),
        n=len(samples),
        samples=samples,
    )
