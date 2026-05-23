"""Agglomerative clustering + significance filter (spec §5.2–5.3, §5.5; §9.3–9.4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.subtree_miner import SubtreeCandidate, fingerprint_of
from src.qft_pcn.composition.abstraction import (
    ClusterConfig, Cluster, cluster_candidates, significant_clusters, k_min,
)


def _cand(source_id: str, rho: np.ndarray, size: int = 4) -> SubtreeCandidate:
    return SubtreeCandidate(source_id, (0, 5 * size), size, rho,
                            fingerprint_of(rho, size))


def _pure(d: int, idx: int) -> np.ndarray:
    rho = np.zeros((d, d), dtype=complex)
    rho[idx, idx] = 1.0
    return rho


def test_two_well_separated_groups_yield_two_clusters():
    # group A: near |0>; group B: near |1>; inter-group distance ~ 1
    a = [_cand(f"a{i}", _pure(4, 0)) for i in range(4)]
    b = [_cand(f"b{i}", _pure(4, 1)) for i in range(3)]
    clusters = cluster_candidates(a + b, ClusterConfig(distance_threshold=0.15))
    assert len(clusters) == 2
    sizes = sorted(c.size for c in clusters)
    assert sizes == [3, 4]


def test_clustering_is_deterministic_under_permutation():
    rng = np.random.default_rng(2)
    cands = [_cand(f"c{i}", _pure(4, i % 2)) for i in range(8)]
    base = cluster_candidates(cands, ClusterConfig())
    shuffled = cands[::-1]
    perm = cluster_candidates(shuffled, ClusterConfig())
    assert sorted(c.size for c in base) == sorted(c.size for c in perm)


def test_significant_clusters_drops_small_ones():
    big = Cluster(members=[_cand(f"x{i}", _pure(4, 0)) for i in range(6)])
    tiny = Cluster(members=[_cand("y0", _pure(4, 1))])
    n = 200
    sig = significant_clusters([big, tiny], n_candidates=n)
    assert big in sig
    assert tiny not in sig
    assert all(c.size >= k_min(n) for c in sig)
