"""Subtree mining for abstraction discovery (spec §4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.qft_pcn.composition._abstraction_const import (
    S_MIN, S_MAX, FP_DECIMALS, FP_RANK, DENSITY_HERMITICITY_TOL, MiningError,
)


@dataclass(frozen=True)
class MineConfig:
    s_min: int = S_MIN
    s_max: int = S_MAX
    fp_decimals: int = FP_DECIMALS
    fp_rank: int = FP_RANK


@dataclass(frozen=True)
class Fingerprint:
    purity: float
    top_eigs: tuple[float, ...]
    ast_size: int


@dataclass
class SubtreeCandidate:
    source_id: str
    leaf_interval: tuple[int, int]
    ast_size: int
    rho: np.ndarray
    fingerprint: Fingerprint


def _validate_density(rho: np.ndarray) -> None:
    """Raise MiningError unless rho is Hermitian, PSD, unit-trace (spec §4.2)."""
    tol = DENSITY_HERMITICITY_TOL
    if not np.allclose(rho, rho.conj().T, atol=tol):
        raise MiningError("reduced density is not Hermitian")
    if np.linalg.eigvalsh(rho).min() < -tol:
        raise MiningError("reduced density is not positive-semidefinite")
    if abs(np.trace(rho).real - 1.0) > 1e-6:
        raise MiningError("reduced density is not unit-trace")


def fingerprint_of(rho: np.ndarray, ast_size: int,
                   fp_decimals: int = FP_DECIMALS,
                   fp_rank: int = FP_RANK) -> Fingerprint:
    """Basis-invariant (purity, top-eigs, size) fingerprint (spec §4.3)."""
    # Tr(rho^2) — purity, basis-invariant.
    purity = round(float(np.trace(rho @ rho).real), fp_decimals)
    eigs = np.sort(np.linalg.eigvalsh(rho).real)[::-1]
    top = list(eigs[:fp_rank]) + [0.0] * max(0, fp_rank - len(eigs))
    top_eigs = tuple(round(float(e), fp_decimals) for e in top[:fp_rank])
    return Fingerprint(purity=purity, top_eigs=top_eigs, ast_size=ast_size)


def _node_aligned_intervals(meta) -> list[tuple[int, int, int]]:
    """Yield (leaf0, leaf_hi, ast_size) for every MERA-tree block whose AST
    size lies in [S_MIN, S_MAX]. A block is a width-2^d leaf interval aligned
    to a 2^d boundary (a sub-MERA-tree); ast_size counts only AST nodes that
    the block FULLY contains (every leaf of the node lies inside [lo,hi)).
    Partial-node leaves and PAD leaves are tolerated at the block edges — the
    boundary bond's reduced density is still well-defined regardless.
    """
    n_leaves = meta.n_leaves
    nol = meta.node_of_leaf
    # Pre-compute leaf membership per node so node-containment is O(1) per node.
    leaves_of: dict[int, list[int]] = {}
    for i, nd in enumerate(nol):
        if nd == -1:
            continue
        leaves_of.setdefault(nd, []).append(i)

    out: list[tuple[int, int, int]] = []
    d = 1
    while (1 << d) <= n_leaves:
        w = 1 << d
        for lo in range(0, n_leaves, w):
            hi = lo + w
            touched = {nol[i] for i in range(lo, hi)}
            touched.discard(-1)
            full = [nd for nd in touched
                    if leaves_of[nd] and leaves_of[nd][0] >= lo
                    and leaves_of[nd][-1] < hi]
            s = len(full)
            if S_MIN <= s <= S_MAX:
                out.append((lo, hi, s))
        d += 1
    return out


def _boundary_density(state, leaf0: int, leaf_hi: int) -> np.ndarray:
    """Reduced density matrix at the boundary bond of the [leaf0,leaf_hi) block.

    Ascends F's per-site layer densities (MERA._layer_density) to the layer at
    which the block is a single site; reads that site's reduced density. Never
    materializes the 16**s leaf space (spec §4.2, principle 4).
    """
    w = leaf_hi - leaf0
    d = w.bit_length() - 1            # block width 2^d
    rho_sites = state._layer_density(d)
    block_index = leaf0 >> d
    return np.asarray(rho_sites[block_index], dtype=complex)


def mine_subtrees(state, meta, source_id: str,
                  config: MineConfig = MineConfig()) -> list[SubtreeCandidate]:
    """Enumerate node-aligned sub-MERAs and extract boundary densities (spec §4.4)."""
    out: list[SubtreeCandidate] = []
    for lo, hi, s in _node_aligned_intervals(meta):
        rho = _boundary_density(state, lo, hi)
        _validate_density(rho)
        fp = fingerprint_of(rho, s, config.fp_decimals, config.fp_rank)
        out.append(SubtreeCandidate(source_id, (lo, hi), s, rho, fp))
    return out


def mine_corpus(solved: list[tuple[object, object, str]],
                config: MineConfig = MineConfig()) -> list[SubtreeCandidate]:
    """mine_subtrees across a corpus (spec §4.4)."""
    out: list[SubtreeCandidate] = []
    for state, meta, source_id in solved:
        out.extend(mine_subtrees(state, meta, source_id, config))
    return out


def bucket_by_fingerprint(
        candidates: list[SubtreeCandidate]
) -> dict[Fingerprint, list[SubtreeCandidate]]:
    """Group candidates by exact fingerprint (spec §4.3)."""
    buckets: dict[Fingerprint, list[SubtreeCandidate]] = {}
    for c in candidates:
        buckets.setdefault(c.fingerprint, []).append(c)
    return buckets
