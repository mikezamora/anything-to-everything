"""Clustering + canonical-form computation for abstraction discovery (spec §5)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.qft_pcn.composition._abstraction_const import (
    DEFAULT_ALPHA, DEFAULT_DISTANCE_THRESHOLD,
)


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


# --- §5.2–5.3, §5.5: clustering, significance filter, provenance -------------
from src.qft_pcn.composition.subtree_miner import SubtreeCandidate, Fingerprint


@dataclass(frozen=True)
class ClusterConfig:
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    linkage: str = "average"          # "average" | "complete" | "single"


@dataclass
class Cluster:
    members: list[SubtreeCandidate]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class Provenance:
    source_ids: tuple[str, ...]
    occurrences: tuple[tuple[str, tuple[int, int]], ...]
    discovered_in_cycle: int
    use_log: list[str] = field(default_factory=list)


def _fp_adjacent(f1: Fingerprint, f2: Fingerprint, grid: float = 1e-3) -> bool:
    """Two fingerprints are comparison-eligible if same size and their rounded
    scalars differ by at most one grid step (spec §4.3)."""
    if f1.ast_size != f2.ast_size:
        return False
    if abs(f1.purity - f2.purity) > grid + 1e-12:
        return False
    return all(abs(a - b) <= grid + 1e-12
               for a, b in zip(f1.top_eigs, f2.top_eigs))


def _pair_distance(c1: SubtreeCandidate, c2: SubtreeCandidate) -> float:
    """Trace distance, gated by the fingerprint pre-filter (spec §4.3, §5.2)."""
    if not _fp_adjacent(c1.fingerprint, c2.fingerprint):
        return 1.0                    # never merged across fingerprint buckets
    return trace_distance(c1.rho, c2.rho)


def _linkage_distance(a: list[SubtreeCandidate], b: list[SubtreeCandidate],
                      dmat: dict[tuple[int, int], float],
                      idx: dict[int, int], how: str) -> float:
    vals = [dmat[(min(idx[id(x)], idx[id(y)]), max(idx[id(x)], idx[id(y)]))]
            for x in a for y in b]
    if how == "single":
        return min(vals)
    if how == "complete":
        return max(vals)
    return sum(vals) / len(vals)      # average


def cluster_candidates(candidates: list[SubtreeCandidate],
                       config: ClusterConfig = ClusterConfig()) -> list[Cluster]:
    """Hierarchical agglomerative clustering under trace distance (spec §5.2).

    Deterministic: candidates are first sorted by (source_id, leaf_interval);
    ties in the merge step break by that order.
    """
    ordered = sorted(candidates, key=lambda c: (c.source_id, c.leaf_interval))
    n = len(ordered)
    if n == 0:
        return []
    idx = {id(c): i for i, c in enumerate(ordered)}
    dmat: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dmat[(i, j)] = _pair_distance(ordered[i], ordered[j])

    groups: list[list[SubtreeCandidate]] = [[c] for c in ordered]
    while len(groups) > 1:
        best = None
        best_d = config.distance_threshold + 1e-12
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                d = _linkage_distance(groups[gi], groups[gj], dmat, idx,
                                      config.linkage)
                if d <= best_d - 1e-12 or (best is None and d <= best_d):
                    if best is None or d < best_d - 1e-12:
                        best, best_d = (gi, gj), d
        if best is None:
            break
        gi, gj = best
        groups[gi] = groups[gi] + groups[gj]
        del groups[gj]
    return [Cluster(members=g) for g in groups]


def significant_clusters(clusters: list[Cluster], n_candidates: int,
                         alpha: float = DEFAULT_ALPHA) -> list[Cluster]:
    """Keep clusters with size >= k_min(n_candidates, alpha) (spec §5.3)."""
    threshold = k_min(n_candidates, alpha)
    return [c for c in clusters if c.size >= threshold]
