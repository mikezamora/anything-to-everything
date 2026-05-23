"""Clustering + canonical-form computation for abstraction discovery (spec §5)."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np

from src.qft_pcn.qft.mera import MERA

from src.qft_pcn.composition._abstraction_const import (
    DEFAULT_ALPHA, DEFAULT_CHI_CAP, DEFAULT_DISTANCE_THRESHOLD,
    DENSITY_HERMITICITY_TOL, REPURIFICATION_TAIL_TOL, RepurificationWarning,
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


# --- §5.4: canonical-form computation ----------------------------------------


@dataclass
class CanonicalPrimitive:
    rho_canonical: np.ndarray
    mera: MERA
    chi: int
    avg_trace_distance: float
    provenance: Provenance


def _repurify(rho: np.ndarray, chi_cap: int) -> tuple[MERA, int]:
    """Re-purify a density matrix into a bounded-bond MERA (spec §5.4 step 2).

    Eigendecompose rho = sum_j p_j |e_j><e_j|; keep the top chi_cap eigenpairs;
    build a purifying term superposition over the kept eigenvectors. The result
    is a bounded-bond MERA -- never a dense leaf-space tensor (§1.1).
    """
    eigvals, eigvecs = np.linalg.eigh(rho)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    keep = int(min(chi_cap, len(eigvals)))
    kept_p = np.clip(eigvals[:keep].real, 0.0, None)
    tail = float(np.sum(np.clip(eigvals[keep:].real, 0.0, None)))
    if tail > REPURIFICATION_TAIL_TOL:
        warnings.warn(
            f"re-purification truncated tail mass {tail:.3e}",
            RepurificationWarning, stacklevel=2,
        )
    total = float(kept_p.sum())
    if total <= 0.0:
        # degenerate (rho is the zero operator); fall back to |0>|0>
        kept_p = np.zeros_like(kept_p)
        kept_p[0] = 1.0
    else:
        kept_p = kept_p / total
    d = rho.shape[0]
    # Purifying term superposition: each kept eigenvector contributes a product
    # term sqrt(p_j) over two boundary leaves (system + purifying reference).
    # Two leaves -> N=2, every per-leaf object is d-dimensional (<= chi_cap**2).
    terms: list[tuple[complex, list[np.ndarray]]] = []
    for j in range(keep):
        vec = np.asarray(eigvecs[:, j], dtype=complex)
        leaf_ref = np.zeros(d, dtype=complex)
        leaf_ref[j % d] = 1.0
        terms.append((complex(np.sqrt(kept_p[j])), [vec, leaf_ref]))
    mera = MERA.from_term_superposition(terms).normalize()
    return mera, keep


def compute_canonical_form(cluster: Cluster,
                           chi_cap: int = DEFAULT_CHI_CAP,
                           cycle_index: int = 0) -> CanonicalPrimitive:
    """Cluster representative: operator-basis mean re-purified (spec §5.4).

    Operator-algebraic only (§1.5/§1.6): the canonical state is the embedded
    mean reduced density (Hermitian PSD trace-1), and the bounded-bond MERA
    is its eigendecomposition-based re-purification. No AST inspection.
    """
    members = cluster.members
    if not members:
        raise ValueError("cannot compute canonical form of an empty cluster")
    dim = max(m.rho.shape[0] for m in members)
    rho_canonical = sum(_embed(m.rho, dim) for m in members) / len(members)
    # Hermitize against round-off; validate the contract (§1.5).
    rho_canonical = 0.5 * (rho_canonical + rho_canonical.conj().T)
    herm_err = float(np.max(np.abs(rho_canonical - rho_canonical.conj().T)))
    assert herm_err <= DENSITY_HERMITICITY_TOL, (
        f"rho_canonical not Hermitian within tol: {herm_err:.3e}")
    tr = complex(np.trace(rho_canonical))
    assert abs(tr.real - 1.0) <= DENSITY_HERMITICITY_TOL and abs(tr.imag) <= DENSITY_HERMITICITY_TOL, (
        f"rho_canonical not unit-trace: tr={tr}")
    min_eig = float(np.linalg.eigvalsh(rho_canonical).min())
    assert min_eig >= -DENSITY_HERMITICITY_TOL, (
        f"rho_canonical not PSD: min_eig={min_eig:.3e}")
    avg_d = float(np.mean([trace_distance(rho_canonical, m.rho)
                           for m in members]))
    mera, chi = _repurify(rho_canonical, chi_cap)
    prov = Provenance(
        source_ids=tuple(dict.fromkeys(m.source_id for m in members)),
        occurrences=tuple((m.source_id, m.leaf_interval) for m in members),
        discovered_in_cycle=cycle_index,
    )
    return CanonicalPrimitive(rho_canonical=rho_canonical, mera=mera, chi=chi,
                              avg_trace_distance=avg_d, provenance=prov)
