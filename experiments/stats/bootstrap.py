"""Bootstrap confidence intervals (spec §14.5).

Spec §14.5: "Confidence intervals: 95% CI by bootstrap on per-run
metrics". The implementation is a plain non-parametric percentile
bootstrap with a configurable resample count.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Sequence

import numpy as np


class BootstrapCI(NamedTuple):
    point: float
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    n_observations: int


def bootstrap_ci(
    samples: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> BootstrapCI:
    """Percentile bootstrap CI for an arbitrary statistic.

    Returns ``BootstrapCI(point, lower, upper, ...)`` with the point
    estimate computed on the full sample and the CI bounds computed
    from the resampled distribution.
    """
    arr = np.asarray(list(samples), dtype=float)
    if arr.size == 0:
        return BootstrapCI(
            point=float("nan"), lower=float("nan"), upper=float("nan"),
            confidence=confidence, n_resamples=n_resamples, n_observations=0,
        )
    rng = np.random.default_rng(seed)
    n = arr.size
    point = float(statistic(arr))
    # Vectorised resample: a (n_resamples, n) matrix of indices.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled = arr[idx]
    stat_per_resample = np.apply_along_axis(statistic, 1, resampled)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(stat_per_resample, alpha))
    upper = float(np.quantile(stat_per_resample, 1.0 - alpha))
    return BootstrapCI(
        point=point, lower=lower, upper=upper,
        confidence=confidence, n_resamples=n_resamples, n_observations=int(n),
    )
