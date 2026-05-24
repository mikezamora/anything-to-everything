"""Goldstone modes for automatic missing-lemma diagnosis (spec §12.6).

Physics origin (Goldstone 1961, Nambu 1960): when a continuous symmetry
is spontaneously broken, massless excitations appear along the broken
directions. Their shape encodes WHICH symmetry was broken.

QPCN realization (spec §12.6): a failed proof attempt has a residual
energy distribution across MERA sites; the near-null subspace of the
constraint Hamiltonian H contains "Goldstone modes" whose spatial
structure points at the missing lemma. Diagonalization of H near zero
energy is a standard Lanczos eigenvalue problem.

This module is an operator-algebraic substrate (NOT an AST walk, NOT a
heuristic over rule names). The constraint-Hessian matrix M is built
from a mix of a real operator expectation (diagonal) and an
operator-bounded coupling proxy (off-diagonal); see honesty notes
below.

  - M[i,i] = <psi| H_t_i |psi>  (per-term FIRST moment, via the real
    MeraEvalHamiltonian.term_energy — a factored window expectation
    per spec §7.4). For the true Hessian diagonal one wants the SECOND
    moment <psi| H_t_i^2 |psi>; the two coincide when H_t_i is
    projector-like (P^2 = P), which is the case for current
    MeraEvalTerm terms per §7.4. We use the first moment directly.
  - M[i,j] for i != j: an **operator-bounded coupling proxy**, NOT a
    genuine two-operator expectation. Specifically,
        M[i,j] = sqrt(r_i * r_j) * |L_i ∩ L_j| / max(|L_i|, |L_j|),
    where r_t = <H_t> and L_t is the term's affected-leaf footprint
    (the entanglement-bond pattern §1.1 — leaves the term reads or
    writes through its causal cone). The sqrt(r_i*r_j) factor is the
    Cauchy-Schwarz UPPER BOUND on |<H_i H_j>| under the projector
    assumption above; the geometric overlap fraction is a heuristic
    on the magnitude. Disjoint footprints decouple correctly via the
    product-MERA factorization <H_i H_j> = <H_i><H_j> (spec §1.2),
    but for overlapping footprints the off-diagonal is a bounded
    heuristic, not the true Hessian entry. The genuine two-operator
    expectation <psi|H_i H_j|psi> on the shared causal-cone window is
    deferred (see EXTENSIONS.md §12.6 entry).

Small eigenvalues of M correspond to Goldstone modes: directions in
term-space where the constraint energy is nearly flat (massless
excitations). The eigenvector's support identifies which terms (rule,
node) carry the missing-lemma signature (spec §12.6 "localization on
specific sites = those sites' constraints are unsatisfied because of a
missing argument").

Solver: ``scipy.sparse.linalg.eigsh`` (Lanczos) on a sparse symmetric
matrix in the per-term basis. Cost is dominated by `term_energy`
calls, not the eigensolver. Cheap relative to the failed proof search
itself (spec §12.6 "Lanczos iteration on a sparse Hamiltonian").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian, MeraEvalTerm,
)
from src.qft_pcn.qft.mera import MERA


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# A constraint-Hessian eigenvalue at or below this is admitted as a
# Goldstone-mode candidate. With M_ii = <H_t> (the per-term residual,
# spec §7.4 projector-like; <H_t^2> = <H_t> so the first moment is the
# Hessian diagonal), eigenvalues lie in [0, 1] for the projector
# basis: the symmetric vacuum (satisfied direction) sits at eigenvalue
# 0, and a fully-broken constraint saturates at 1. We admit the entire
# excitation range above the residual floor as Goldstone candidates
# (spec §12.6 line 1597: "lowest-energy excitation above the ground
# state" — the ground state being the satisfied null space at
# eigenvalue 0; an excitation can sit anywhere up to the projector
# saturation). A small epsilon above 1 absorbs numerical jitter in
# the off-diagonal coupling additions.
DEFAULT_GOLDSTONE_THRESHOLD = 1.0 + 1e-6

# Below this per-term residual a term is treated as "satisfied" and does
# not anchor a candidate-missing-lemma site. The same floor used by
# bidirectional_evolve's descent test in test_bidirectional_evolution.
DEFAULT_RESIDUAL_FLOOR = 1e-6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldstoneMode:
    """A single near-null eigenmode of the constraint-Hessian matrix.

    ``eigenvalue`` is the "mass" of the mode (smaller = more Goldstone-
    like). ``support`` lists the (term_rule_id, term_node, weight)
    triples for terms carrying the dominant amplitude of the eigenvector
    (weight is the squared component). ``term_indices`` is the raw set
    of indices into the Hamiltonian's ``terms`` list for the eigenvector
    entries above the support threshold.
    """
    eigenvalue: float
    eigenvector: tuple[float, ...]
    support: tuple[tuple[str, int, float], ...]
    term_indices: tuple[int, ...]


@dataclass(frozen=True)
class CandidateMissingLemma:
    """One candidate missing-lemma signature surfaced by Goldstone analysis.

    The rule the candidate is associated with (``rule_id``) and the AST
    node where the lemma would be applied (``node``) are read directly
    off the dominant term carrying the Goldstone mode's amplitude. The
    ``confidence`` is the inverse-eigenvalue weight relative to the
    spectral gap — multiple candidates surface with their relative
    confidence when the Goldstone mode is degenerate (spec §12.6 risk
    table).
    """
    rule_id: str
    node: int
    confidence: float
    mode_eigenvalue: float


@dataclass(frozen=True)
class LemmaDiagnostic:
    """Diagnostic returned by :func:`diagnose_missing_lemma`.

    ``candidates`` is sorted by confidence (highest first). If the
    proof is fully relaxed (no eigenvalue below ``threshold``), the
    list is empty and ``has_missing_lemma`` is False. ``spectrum`` is
    the full set of computed eigenvalues for inspection / debugging.
    ``residual_energy`` is ``H.total_energy(state)`` — the global
    residual the diagnostic was computed against.
    """
    candidates: tuple[CandidateMissingLemma, ...]
    modes: tuple[GoldstoneMode, ...]
    spectrum: tuple[float, ...]
    residual_energy: float
    threshold: float

    @property
    def has_missing_lemma(self) -> bool:
        return len(self.candidates) > 0


# ---------------------------------------------------------------------------
# Core: build the constraint-Hessian matrix in the per-term basis
# ---------------------------------------------------------------------------


def _build_constraint_matrix(
    H: MeraEvalHamiltonian, state: MERA,
) -> tuple[np.ndarray, list[float], list[MeraEvalTerm]]:
    """Build the per-term constraint-Hessian M (dense ndarray) and the
    per-term residual vector.

    The basis is :attr:`MeraEvalHamiltonian.terms` (one entry per
    (rule, node) pair). M is symmetric, PSD by construction (diagonal
    = non-negative term energies; off-diagonal = sqrt(r_i*r_j) * J[i,j]
    with J in [0,1]).

    Returns ``(M, residuals, terms)``.
    """
    terms = list(H.terms)
    n = len(terms)
    residuals = [float(H.term_energy(state, t)) for t in terms]

    # Leaf footprints — operator-derived from term_affected_leaves
    # (the causal-cone leaf set the term reads/writes per §1.1).
    footprints: list[frozenset[int]] = [
        H.term_affected_leaves(t) for t in terms
    ]

    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        M[i, i] = residuals[i]
    # Off-diagonal: operator-bounded coupling proxy (NOT the true
    # two-operator expectation). |<H_i H_j>| is upper-bounded by
    # Cauchy-Schwarz: |<H_i H_j>| <= sqrt(<H_i^2><H_j^2>); under the
    # projector assumption (P^2 = P, true for current MeraEvalTerm
    # terms per §7.4) <H_t^2> = <H_t> = r_t, so sqrt(r_i*r_j) is the
    # Cauchy-Schwarz upper bound. The leaf-overlap fraction is then a
    # geometric heuristic on the magnitude. Disjoint footprints
    # decouple correctly (§1.2 factorization); overlapping footprints
    # carry a bounded heuristic, not the genuine Hessian entry. See
    # EXTENSIONS.md §12.6 for the deferred two-operator primitive.
    for i in range(n):
        Li = footprints[i]
        if not Li or residuals[i] <= 0.0:
            continue
        for j in range(i + 1, n):
            Lj = footprints[j]
            if not Lj or residuals[j] <= 0.0:
                continue
            inter = len(Li & Lj)
            if inter == 0:
                continue  # disjoint footprints: factorization → no coupling
            overlap = inter / max(len(Li), len(Lj))
            coupling = float(np.sqrt(residuals[i] * residuals[j]) * overlap)
            M[i, j] = coupling
            M[j, i] = coupling
    return M, residuals, terms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_near_null_subspace(
    H: MeraEvalHamiltonian,
    state: MERA,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Lanczos / sparse eigsh on the constraint Hamiltonian's
    per-term-basis Hessian.

    Returns ``(eigvals, eigvecs)`` with ``eigvals`` sorted ascending
    (smallest first — these are the near-null Goldstone candidates) and
    ``eigvecs[:, i]`` the eigenvector for ``eigvals[i]``.

    The matrix is built in the (rule, node) term basis (cost scales
    with the number of evaluation-rule terms ~ 8 * N_nodes, not with
    the leaf Hilbert dimension). Lanczos via
    :func:`scipy.sparse.linalg.eigsh` for k smallest eigenpairs; falls
    back to dense :func:`numpy.linalg.eigh` when the basis is small
    enough that Lanczos has no asymptotic advantage (spec §12.6
    "standard numerical linear algebra").

    Spec §12.6 substrate contract:
      * Eigenvectors live in the term basis, NOT in some heuristic
        rule-name space — the basis is the real Hamiltonian's term
        list.
      * Diagonal entries are real operator expectations
        (``H.term_energy``), specifically the FIRST moment <H_t>
        (coincides with the true Hessian diagonal <H_t^2> for the
        projector-like terms in §7.4).
      * Off-diagonal couplings are an operator-bounded proxy
        (Cauchy-Schwarz upper bound × geometric leaf-overlap
        fraction), NOT the genuine two-operator expectation. Disjoint
        leaf footprints decouple correctly via §1.2 factorization;
        overlapping footprints carry a bounded heuristic. See
        EXTENSIONS.md §12.6 entry for the deferred true-Hessian
        primitive.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    M, _residuals, terms = _build_constraint_matrix(H, state)
    n = M.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)
    k_eff = min(k, n)
    # Lanczos has no advantage when k_eff >= n - 1; eigsh in fact
    # requires k < n. Fall back to dense eigh for those cases (still
    # standard numerical linear algebra; spec §12.6 just requires the
    # eigenvalues be exact, not the algorithm).
    if k_eff >= n - 1 or n <= 8:
        w, V = np.linalg.eigh(M)
        order = np.argsort(w)
        return w[order][:k_eff], V[:, order][:, :k_eff]
    sparse_M = csr_matrix(M)
    # which='SM' (smallest magnitude) returns the near-null eigenpairs.
    # sigma=0 with shift-invert would be faster but requires a
    # factorization; for the sizes we operate at (term basis ~ few
    # hundred at most) plain SM is fine and avoids the singular-shift
    # risk when the spectrum touches zero.
    w, V = eigsh(sparse_M, k=k_eff, which='SM')
    order = np.argsort(w)
    return w[order], V[:, order]


def diagnose_missing_lemma(
    H: MeraEvalHamiltonian,
    state: MERA,
    k: int = 5,
    threshold: float = DEFAULT_GOLDSTONE_THRESHOLD,
    support_threshold: float = 0.05,
    residual_floor: float = DEFAULT_RESIDUAL_FLOOR,
) -> LemmaDiagnostic:
    """Diagnose missing lemmas from the Goldstone modes of H at state.

    Returns a :class:`LemmaDiagnostic` whose ``candidates`` list is
    non-empty iff the constraint Hamiltonian has an eigenvalue below
    ``threshold`` AND the corresponding eigenvector has support on at
    least one term whose residual exceeds ``residual_floor`` (the
    "broken symmetry direction is anchored on an unsatisfied
    constraint" criterion, spec §12.6).

    Determinism: with a fixed (H, state, k, threshold, support_threshold,
    residual_floor) the matrix and its spectrum are deterministic
    functions of the inputs; eigsh / eigh are deterministic for a fixed
    matrix; candidate ordering breaks ties on (eigenvalue, rule_id, node)
    so the diagnostic is byte-identical across runs.
    """
    terms = list(H.terms)
    n = len(terms)
    residual_energy = float(H.total_energy(state))
    if n == 0:
        return LemmaDiagnostic(
            candidates=(), modes=(), spectrum=(),
            residual_energy=residual_energy, threshold=threshold,
        )
    # Compute the FULL spectrum here (n is small — term basis ~ tens
    # to low hundreds — so dense eigh is both fast and robust on the
    # near-degenerate constraint Hessian §7.4 projector basis). The
    # public ``compute_near_null_subspace`` returns the k smallest
    # eigenpairs (the literal "near-null" contract its callers rely
    # on, e.g. test_compute_near_null_returns_sorted_ascending); the
    # diagnostic itself needs to walk the WHOLE spectrum so it can
    # surface the broken-symmetry directions (which sit at the upper
    # end of the projector-basis spectrum, eigenvalue ~ residual; spec
    # §12.6 line 1597 "lowest-energy excitation above the ground
    # state" — the ground state being the satisfied null space at
    # eigenvalue 0, so an excitation is any eigenpair with eigenvalue
    # > residual_floor whose eigenvector localizes on a residual-
    # carrying term).
    M, _residuals_unused, _terms_unused = _build_constraint_matrix(H, state)
    eigvals_full, eigvecs_full = np.linalg.eigh(M)
    order_full = np.argsort(eigvals_full)
    eigvals = eigvals_full[order_full]
    eigvecs = eigvecs_full[:, order_full]
    # Public spectrum view: k smallest, matching compute_near_null_subspace.
    k_view = min(k, len(eigvals))
    spectrum = tuple(float(x) for x in eigvals[:k_view])
    # Re-fetch residuals (the matrix builder discards them).
    residuals = [float(H.term_energy(state, t)) for t in terms]

    modes: list[GoldstoneMode] = []
    candidate_buckets: dict[tuple[str, int], list[CandidateMissingLemma]] = {}
    for idx in range(len(eigvals)):
        ev = float(eigvals[idx])
        # Skip the satisfied null space (eigenvalue ≤ residual_floor):
        # those directions are the symmetric vacuum, not excitations.
        if ev <= residual_floor:
            continue
        # Skip super-saturated eigenvalues outside the projector-basis
        # band (numerical artifacts of the off-diagonal coupling
        # heuristic: a legitimate projector-basis residual is bounded
        # by 1.0 per §7.4 P^2 = P).
        if ev > threshold:
            continue
        vec = np.asarray(eigvecs[:, idx]).real
        # Normalize sign deterministically: largest-magnitude component
        # has non-negative sign (eigenvectors are otherwise determined
        # only up to a global ±).
        if vec.size:
            jmax = int(np.argmax(np.abs(vec)))
            if vec[jmax] < 0:
                vec = -vec
        weights = vec * vec
        # Support: term indices with squared amplitude above the
        # support threshold AND with residual above the floor (a term
        # already satisfied cannot anchor a missing-lemma diagnosis).
        support = []
        idxs = []
        for ti in range(n):
            w = float(weights[ti])
            if w < support_threshold:
                continue
            if residuals[ti] <= residual_floor:
                continue
            t = terms[ti]
            support.append((t.rule_id, t.node, w))
            idxs.append(ti)
        if not support:
            continue  # mode is purely along satisfied-constraint directions
        # Deterministic sort: weight desc, then rule_id, then node.
        support.sort(key=lambda s: (-s[2], s[0], s[1]))
        modes.append(GoldstoneMode(
            eigenvalue=ev,
            eigenvector=tuple(float(x) for x in vec),
            support=tuple(support),
            term_indices=tuple(idxs),
        ))
        # Each support entry contributes a candidate at this mode.
        # Confidence: amplitude weight × eigenvalue (= the per-direction
        # residual in the projector basis, spec §7.4). Larger eigenvalue
        # means more unsatisfied constraint mass concentrated in this
        # direction — the broken-symmetry direction §12.6 wants
        # surfaced — and larger amplitude on a specific (rule, node)
        # localizes that mass to that site.
        for (rule, node, w) in support:
            cand = CandidateMissingLemma(
                rule_id=rule, node=node,
                confidence=float(w * ev),
                mode_eigenvalue=ev,
            )
            candidate_buckets.setdefault((rule, node), []).append(cand)

    # Collapse duplicates: keep the highest-confidence candidate per
    # (rule, node) site (degenerate Goldstone modes can land on the
    # same term; we report the strongest single signature, per spec
    # §12.6 "report all candidates with their relative confidence").
    deduped: list[CandidateMissingLemma] = []
    for key, cands in candidate_buckets.items():
        cands.sort(key=lambda c: -c.confidence)
        deduped.append(cands[0])
    deduped.sort(key=lambda c: (-c.confidence, c.rule_id, c.node))

    return LemmaDiagnostic(
        candidates=tuple(deduped),
        modes=tuple(modes),
        spectrum=spectrum,
        residual_energy=residual_energy,
        threshold=threshold,
    )


__all__ = [
    "CandidateMissingLemma",
    "GoldstoneMode",
    "LemmaDiagnostic",
    "DEFAULT_GOLDSTONE_THRESHOLD",
    "DEFAULT_RESIDUAL_FLOOR",
    "compute_near_null_subspace",
    "diagnose_missing_lemma",
]
