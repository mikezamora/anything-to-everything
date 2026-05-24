"""Effect-size measures (spec §14.5).

Cohen's d (parametric) and Cliff's delta (non-parametric). The spec
calls out Cohen's d explicitly; Cliff's delta is the
distribution-free counterpart used when normality is suspect.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d on two independent samples.

    d = (mean(a) - mean(b)) / pooled_std

    Returns 0 on degenerate (zero-variance) inputs to avoid spurious
    "infinite effect size" reports.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    mx, my = float(x.mean()), float(y.mean())
    sx, sy = float(x.std(ddof=1)), float(y.std(ddof=1))
    pooled = np.sqrt(((x.size - 1) * sx ** 2 + (y.size - 1) * sy ** 2)
                     / (x.size + y.size - 2))
    if pooled == 0.0:
        return 0.0
    return (mx - my) / float(pooled)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta: probability of a > b minus probability of a < b.

    Range [-1, 1]. 0 means stochastically equal; +1 means a strictly
    dominates b; -1 the reverse. Robust to non-normal data.
    """
    x = np.asarray(list(a), dtype=float)
    y = np.asarray(list(b), dtype=float)
    if x.size == 0 or y.size == 0:
        return 0.0
    # Pairwise comparison; O(|x|*|y|) but the bench sizes are small.
    gt = float(np.sum(x[:, None] > y[None, :]))
    lt = float(np.sum(x[:, None] < y[None, :]))
    return (gt - lt) / (x.size * y.size)
