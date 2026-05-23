"""Clustering + canonical-form computation for abstraction discovery (spec §5)."""
from __future__ import annotations

import math

import numpy as np

from src.qft_pcn.composition._abstraction_const import DEFAULT_ALPHA


def _embed(rho: np.ndarray, dim: int) -> np.ndarray:
    """Embed a (d, d) density into a (dim, dim) space by zero-padding (spec §5.1)."""
    if rho.shape[0] == dim:
        return rho
    out = np.zeros((dim, dim), dtype=complex)
    d = rho.shape[0]
    out[:d, :d] = rho
    return out


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """D(rho1, rho2) = 0.5 * sum(|eigvals(rho1 - rho2)|)  (spec §5.1).

    Hermitian inputs; unequal dimensions are zero-padded to the larger space.
    """
    dim = max(rho1.shape[0], rho2.shape[0])
    diff = _embed(rho1, dim) - _embed(rho2, dim)
    # Hermitian difference -> eigvalsh is the correct, cheap Schatten-1 route.
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(diff))))


def k_min(n_candidates: int, alpha: float = DEFAULT_ALPHA) -> int:
    """Smallest cluster size unlikely to arise by chance (spec §5.3).

    Uniform null model: P(cluster of size k) ~ exp(-k log N). Solve
    exp(-k log N) <= alpha  ->  k >= -log(alpha) / log(N). Floored at 2.
    """
    n = max(n_candidates, 2)
    k = math.ceil(-math.log(alpha) / math.log(n))
    return max(k, 2)
