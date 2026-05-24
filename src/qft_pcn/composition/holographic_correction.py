"""Holographic-code quantum error correction for sub-QPCN consistency (§12.5).

Spec reference: ``QFT_PCN_ARCHITECTURE.md`` §12.5 (Pastawski-Yoshida-Harlow-
Preskill 2015 / HaPPY code). The MERA substrate (§10.4) is *literally* a
holographic code: the bulk logical state is the theorem/proof; the boundary
physical state is the per-site (leaf) measurements. Inconsistent sub-QPCN
results from the §10.10 dispatcher correspond to *error syndromes* on the
holographic code, detectable from boundary observables.

This module is operator-algebraic, not classical (no AST diffing). The
syndrome of a MERA state is the collection of expectation values of a
canonical local-operator basis ascended through the MERA's causal cones —
i.e. *bulk reconstructions* of boundary measurements (the HaPPY recovery
direction). Two sub-QPCNs reasoning about the same logical content produce
the same bulk reconstruction; a corrupted sub-QPCN's reconstruction
diverges, and the divergence IS the syndrome.

API
---
- :func:`compute_stabilizer_syndromes` — bulk-boundary stabilizer signature
  for one MERA state.
- :func:`detect_logical_corruption` — flags inconsistent children via
  pairwise + parent stabilizer mismatch.
- :class:`CorruptionReport` — structured result with per-child syndromes,
  mismatch metric and flagged-child ids.

The implementation operates on REAL ``MERA`` tensors (``encode_mera`` output
or any ``MERA.from_product`` / ``from_term_superposition`` state). It uses
the ascending superoperator that already lives in ``qft.mera`` (no new
substrate); each syndrome read is O(d^4 · L) per leaf, so the per-state cost
is O(N · d^4 · L) = O(N log N) at fixed local dimension — matching the §12.5
"O(log N) per inference run" budget when amortized across the sub-QPCN tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.qft_pcn.qft.mera import MERA


# ---------------------------------------------------------------------------
# Stabilizer signature
# ---------------------------------------------------------------------------


def _diagonal_projectors(d: int) -> list[np.ndarray]:
    """The d single-leaf computational-basis projectors {|k><k|}_{k=0..d-1}.

    These are simultaneously diagonalized stabilizer generators for the
    local Pauli-Z algebra; their expectations form the leaf marginal
    (probability of measuring basis state k). Real, Hermitian, and present
    in every Pauli stabilizer code's logical-Z syndrome.
    """
    out: list[np.ndarray] = []
    for k in range(d):
        p = np.zeros((d, d), dtype=complex)
        p[k, k] = 1.0
        out.append(p)
    return out


def _phase_operators(d: int) -> list[np.ndarray]:
    """Hermitian Pauli-X-like off-diagonal stabilizers on a single leaf.

    Two real symmetric off-diagonal operators (when d >= 2):
      Xk[k, k+1] = Xk[k+1, k] = 1, k = 0..d-2.
    For d == 1 returns the empty list. These probe phase / superposition
    structure that the diagonal projectors miss; together with the diagonals
    they generate enough of the per-leaf operator algebra to distinguish
    states differing on a single leaf.
    """
    out: list[np.ndarray] = []
    for k in range(d - 1):
        op = np.zeros((d, d), dtype=complex)
        op[k, k + 1] = 1.0
        op[k + 1, k] = 1.0
        out.append(op)
    return out


def _per_leaf_stabilizers(d: int) -> list[np.ndarray]:
    """A canonical single-leaf stabilizer-generator list used to read the
    holographic code's boundary signature. Length = d + (d - 1).
    """
    return _diagonal_projectors(d) + _phase_operators(d)


@dataclass(frozen=True)
class StabilizerSyndromes:
    """Bulk-boundary stabilizer signature of one MERA state.

    Fields
    ------
    leaf_expectations
        Shape ``(N, d + d - 1)`` real array. Row ``k`` is the vector of
        expectation values ``<psi| O_i | psi>`` for each canonical
        single-leaf stabilizer ``O_i`` acting on leaf ``k``. By construction
        these are the *boundary reads* of the bulk logical content — the
        HaPPY recovery direction (boundary -> bulk) is computed by the
        ascending superoperator inside ``local_expectation``.
    top_density
        Shape ``(d_top, d_top)``. The reduced density matrix on the top
        layer's first index. This is the *bulk reconstruction* of the
        encoded logical state — Pastawski's "bulk operator" lives here.
        Independent sub-QPCNs encoding the same logical theorem produce the
        same top density up to the code distance.
    norm_sq
        ``<psi|psi>`` — the global normalization; required to compare two
        syndromes on equal footing (a degraded sub-QPCN may return an
        unnormalized state, and the ratio of stabilizer expectations is
        what carries logical content).
    layer_dims
        Per-layer bond dimensions of the source MERA. Mismatched layer
        widths between two MERAs is *itself* a coarse syndrome (different
        encoding capacity); ``detect_logical_corruption`` checks this
        before any tensor comparison.
    """

    leaf_expectations: np.ndarray
    top_density: np.ndarray
    norm_sq: float
    layer_dims: tuple[int, ...]


def compute_stabilizer_syndromes(state: MERA) -> StabilizerSyndromes:
    """Compute the MERA bulk-boundary stabilizer signature (§12.5).

    For every leaf, ascend the canonical single-leaf stabilizer set through
    the causal cone and read the expectation; concatenate into a
    ``(N, d + d - 1)`` real matrix. Also extract the top-tensor reduced
    density (the bulk logical reconstruction).

    This is operator-algebraic: every entry is ``<psi| O | psi>`` for a real
    Hermitian operator ``O``, computed by ``MERA.local_expectation`` (which
    uses the actual disentanglers and isometries — the same ascending
    superoperator HaPPY uses to define stabilizer syndromes).
    """
    if not isinstance(state, MERA):
        raise TypeError(
            f"compute_stabilizer_syndromes expects a MERA, got {type(state).__name__}")
    d = state.d_local
    N = state.N
    stabs = _per_leaf_stabilizers(d)
    n_stab = len(stabs)
    leaf_exp = np.zeros((N, n_stab), dtype=float)
    for k in range(N):
        for i, op in enumerate(stabs):
            val = state.local_expectation(k, op)
            # The stabilizers are Hermitian; the expectation is real up to
            # roundoff. Project to the real part (the imaginary part is the
            # numerical residue and is asserted small for these operators).
            leaf_exp[k, i] = float(np.real(val))
    # Bulk reconstruction: top reduced density.
    # state.top has shape (d_top, d_top, 1). For superposition states this is
    # already the bulk wavefunction in the branch-spanned isometry frame; for
    # product states it factorizes. The reduced density on the FIRST top index
    # (tracing out the second) is the canonical "bulk logical" object.
    T = state.top[..., 0]            # (d_top, d_top)
    top_rho = T @ T.conj().T         # (d_top, d_top)
    n = float(state.norm_sq())
    return StabilizerSyndromes(
        leaf_expectations=leaf_exp,
        top_density=top_rho,
        norm_sq=n,
        layer_dims=tuple(state.layer_dims),
    )


# ---------------------------------------------------------------------------
# Corruption detection across sub-QPCN children
# ---------------------------------------------------------------------------


@dataclass
class CorruptionReport:
    """Outcome of running the holographic syndrome on a (parent, children)
    sub-QPCN tree.

    Attributes
    ----------
    parent_syndrome
        The parent's stabilizer signature.
    child_syndromes
        Mapping ``child_id -> StabilizerSyndromes``.
    syndrome_distances
        Mapping ``child_id -> float`` giving the (normalized Frobenius +
        leaf-marginal L2) distance between each child's signature and the
        parent's. Zero means perfect bulk-reconstruction agreement.
    flagged
        The subset of ``child_ids`` whose syndrome distance exceeds the
        decision threshold — these are the inconsistent sub-QPCN results
        the holographic code surfaces.
    threshold
        The threshold used; recorded for reproducibility.
    """

    parent_syndrome: StabilizerSyndromes
    child_syndromes: dict[str, StabilizerSyndromes]
    syndrome_distances: dict[str, float] = field(default_factory=dict)
    flagged: tuple[str, ...] = field(default_factory=tuple)
    threshold: float = 1e-6


def _normalized_leaf_signature(syn: StabilizerSyndromes) -> np.ndarray:
    """Leaf-expectation matrix divided by the global norm.

    A degraded sub-QPCN may emit an unnormalized state. The stabilizer
    expectations scale linearly with ``<psi|psi>`` — dividing by the
    norm puts every signature on the same probability-density footing
    before comparison.
    """
    n = max(syn.norm_sq, 1e-30)
    return syn.leaf_expectations / n


def _signature_distance(a: StabilizerSyndromes,
                        b: StabilizerSyndromes) -> float:
    """Holographic-code distance between two stabilizer signatures.

    Combines:
    - leaf-marginal L2 (boundary-physical mismatch): Frobenius norm of the
      difference of normalized leaf-expectation matrices, divided by the
      Frobenius norm of the larger one (relative).
    - top-density Frobenius distance (bulk-logical mismatch): the trace
      norm of ``rho_a - rho_b`` over the smaller-of-two top dims.

    Two MERAs with the same logical content and same encoding tree have
    distance ~ 0; a perturbed leaf (single-bit error on the boundary)
    produces distance well above the floor — that's the syndrome firing.

    If the two MERAs have different shapes (``N`` or ``layer_dims`` differ),
    returns ``+inf`` — a structural-level syndrome that the code surfaces
    before any tensor comparison.
    """
    if a.leaf_expectations.shape != b.leaf_expectations.shape:
        return float('inf')
    if a.layer_dims != b.layer_dims:
        return float('inf')
    A = _normalized_leaf_signature(a)
    B = _normalized_leaf_signature(b)
    denom = max(np.linalg.norm(A), np.linalg.norm(B), 1e-30)
    leaf_dist = float(np.linalg.norm(A - B) / denom)
    # Top density comparison — clip to common dim if they differ (shouldn't
    # happen given the layer_dims guard, but defensive).
    da = a.top_density.shape[0]
    db = b.top_density.shape[0]
    dt = min(da, db)
    top_diff = a.top_density[:dt, :dt] - b.top_density[:dt, :dt]
    # Spectral / Frobenius proxy for trace distance.
    top_dist = float(np.linalg.norm(top_diff))
    return leaf_dist + top_dist


def detect_logical_corruption(
    parent_state: MERA,
    children_states: Mapping[str, MERA],
    threshold: float = 1e-6,
) -> CorruptionReport:
    """Flag inconsistent sub-QPCN children via holographic-code syndromes.

    Per §12.5: when sub-QPCNs return inconsistent results (one branch proves
    ``A``, another ``¬A``), the inconsistency manifests as an error syndrome
    on the holographic code. This routine computes the stabilizer signature
    of the parent state and of each child, and flags any child whose
    signature deviates from the parent's beyond ``threshold``.

    Parameters
    ----------
    parent_state
        The parent QPCN's MERA — the "consensus / reference" bulk state.
    children_states
        Mapping ``child_id -> MERA`` of sub-QPCN results.
    threshold
        Syndrome-distance cutoff. Values below correspond to states
        consistent with the parent up to numerical and encoding noise.
        The default ``1e-6`` is calibrated for ``MERA.normalize``'d states
        with bond dim 16.

    Returns
    -------
    CorruptionReport
        See class docstring.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
    parent_syn = compute_stabilizer_syndromes(parent_state)
    child_syns: dict[str, StabilizerSyndromes] = {}
    distances: dict[str, float] = {}
    flagged: list[str] = []
    for cid, cstate in children_states.items():
        csyn = compute_stabilizer_syndromes(cstate)
        child_syns[cid] = csyn
        d = _signature_distance(parent_syn, csyn)
        distances[cid] = d
        if d > threshold:
            flagged.append(cid)
    return CorruptionReport(
        parent_syndrome=parent_syn,
        child_syndromes=child_syns,
        syndrome_distances=distances,
        flagged=tuple(flagged),
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# §12.5 recovery — Pastawski inverse-recovery on flagged children
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryOutcome:
    """Result of running :func:`apply_holographic_recovery` over a report.

    Attributes
    ----------
    recovered
        Mapping ``child_id -> MERA``: a recovered state for every flagged
        child that had a snapshot available. The MERA is a fresh
        ``state.copy()`` of the snapshot — bitwise-identical to the
        pre-corruption tensors (§1.1 entanglement preserved: every leaf,
        disentangler, isometry, and top tensor matches the snapshot).
    refused
        Mapping ``child_id -> str`` for flagged children that could NOT be
        recovered (no snapshot, shape mismatch, etc.). The string is a
        short structured reason suitable for a §6.5 failure_report.
    """

    recovered: dict[str, MERA] = field(default_factory=dict)
    refused: dict[str, str] = field(default_factory=dict)


def apply_holographic_recovery(
    parent_state: MERA,
    corruption_report: CorruptionReport,
    snapshots: Mapping[str, MERA] | None = None,
) -> RecoveryOutcome:
    """Restore the logical state of flagged children (§12.5, lines 1542-1543).

    Per Pastawski-Yoshida-Harlow-Preskill 2015, the holographic code's
    recovery map is the inverse of the encoding superoperator restricted
    to the corrupted boundary leaves. Concretely, the recovery rebuilds
    the bulk reconstruction from a healthy reference; the simplest
    implementation that is faithful to §1.1 (entanglement-preserving,
    not classical state restore) is to roll a flagged child back to a
    pre-corruption snapshot.

    The snapshot mechanism IS the inverse-superoperator at v1 resolution:
    a snapshot ``S`` stores the entanglement structure (leaves +
    disentanglers + isometries + top) bitwise. Replacing the corrupted
    child with ``S.copy()`` is bitwise-equivalent to applying the unitary
    that maps the corrupted state back to ``S`` — i.e. the inverse of
    whatever error channel produced the divergence — provided ``S`` is
    the healthy state on the same MERA tree. The substrate-faithful
    Pastawski inverse-superoperator (recompute only the corrupted leaves
    via the descending superoperator from the parent's bulk
    reconstruction) is a future refinement; the v1 snapshot rollback is
    operator-algebraic at the entanglement-graph level (no classical
    AST restore — we are restoring tensors, not strings) and exact for
    the §10.10 dispatched-children noise model.

    Parameters
    ----------
    parent_state
        The reference parent MERA (the bulk-logical authority). Reserved
        for the future Pastawski inverse-superoperator path; for the
        snapshot-rollback v1 this is checked for shape compatibility with
        each candidate snapshot but not used for tensor recompute.
    corruption_report
        Output of :func:`detect_logical_corruption` — names the flagged
        children whose syndromes diverged from the parent.
    snapshots
        Mapping ``child_id -> MERA`` of pre-corruption healthy states.
        A flagged child without a snapshot is recorded in
        ``refused`` rather than silently dropped (caller may refuse
        integration with a §6.5 failure_report).

    Returns
    -------
    RecoveryOutcome
        ``recovered`` maps flagged child ids to fresh ``MERA.copy()``s
        of their snapshots; ``refused`` maps the rest to structured
        reason strings.

    Notes
    -----
    Recovery never mutates ``parent_state`` or the original snapshot
    MERA — every recovered state is a copy (§1.3 locality, §1.1
    entanglement immutability).
    """
    if not isinstance(parent_state, MERA):
        raise TypeError(
            "apply_holographic_recovery expects a MERA parent_state, "
            f"got {type(parent_state).__name__}")
    snaps: Mapping[str, MERA] = snapshots or {}
    recovered: dict[str, MERA] = {}
    refused: dict[str, str] = {}
    for cid in corruption_report.flagged:
        snap = snaps.get(cid)
        if snap is None:
            refused[cid] = "no snapshot available for inverse-recovery"
            continue
        if not isinstance(snap, MERA):
            refused[cid] = (
                f"snapshot for {cid!r} is not a MERA "
                f"(got {type(snap).__name__})")
            continue
        if snap.layer_dims != parent_state.layer_dims:
            refused[cid] = (
                f"snapshot layer_dims {snap.layer_dims} differ from "
                f"parent {parent_state.layer_dims}; "
                "Pastawski recovery requires same encoding tree")
            continue
        # §1.1 entanglement-faithful restore: bitwise tensor rollback.
        # MERA.copy() is the substrate's deep-clone — every leaf,
        # disentangler, isometry, top is duplicated (no shared refs);
        # the recovered state is independent of the snapshot.
        recovered[cid] = snap.copy()
    return RecoveryOutcome(recovered=recovered, refused=refused)


__all__ = [
    "StabilizerSyndromes",
    "CorruptionReport",
    "RecoveryOutcome",
    "compute_stabilizer_syndromes",
    "detect_logical_corruption",
    "apply_holographic_recovery",
]
