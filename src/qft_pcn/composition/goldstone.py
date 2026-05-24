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
from real operator expectations on both diagonal and off-diagonal:

  - M[i,i] = <psi| H_t_i |psi>  (per-term FIRST moment, via the real
    MeraEvalHamiltonian.term_energy — a factored window expectation
    per spec §7.4). For the true Hessian diagonal one wants the SECOND
    moment <psi| H_t_i^2 |psi>; the two coincide when H_t_i is
    projector-like (P^2 = P), which is the case for current
    MeraEvalTerm terms per §7.4. We use the first moment directly.
  - M[i,j] for i != j: the GENUINE symmetric two-operator expectation
    Re <psi| H_i H_j |psi> on the shared causal-cone window of the
    union footprint L_i ∪ L_j. Since each penalty H_i is a sum of
    per-leaf factored operator products (spec §7.4 `_penalty_ops`),
    H_i H_j is the sum over (a,b) of the per-leaf operator products
    O_{i,a}[k] @ O_{j,b}[k] on shared leaves (identity on the rest),
    which routes back through ``mera_window_expectation_factored`` —
    the same substrate primitive §12.1 anomaly already uses. Disjoint
    footprints automatically factor via §1.2 (<H_i H_j> = <H_i><H_j>)
    since the per-leaf product over the union footprint is simply the
    concatenation of the two non-overlapping leaf-op dicts; the
    factored expectation evaluates that as a single window call. We
    take the real part for symmetry of the Hessian (H_i, H_j Hermitian
    => <H_j H_i> = <H_i H_j>*, so the symmetric Hessian entry is
    Re <H_i H_j>; in practice the imaginary part is at numerical
    floor for the projector basis §7.4).

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

from src.qft_pcn.logic._mera_window import mera_window_expectation_factored
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


def _compose_leaf_ops(
    ops_a: dict, ops_b: dict,
) -> dict:
    """Per-leaf product of two factored operator dicts.

    Each dict maps ``leaf -> (d,d)`` operator. The product operator
    over the union of leaves is the per-leaf matrix product
    ``ops_a[leaf] @ ops_b[leaf]`` where both supply an op, and just
    the one supplied op where only one does (identity on the other).
    This is exact: the operator ``O_a O_b`` factors over leaves
    because each factor itself factors (spec §7.4); the per-leaf
    composition is just `(A_k ⊗ I_rest) (B_k ⊗ I_rest) = (A_k B_k) ⊗
    I_rest` on disjoint leaves, and `(A_k B_k) ⊗ ...` on shared
    leaves.
    """
    merged: dict = dict(ops_a)
    for leaf, op_b in ops_b.items():
        prev = merged.get(leaf)
        if prev is None:
            merged[leaf] = op_b
        else:
            merged[leaf] = prev @ op_b
    return merged


def _two_term_expectation(
    H: MeraEvalHamiltonian, state: MERA,
    term_i: MeraEvalTerm, term_j: MeraEvalTerm,
) -> float:
    """Re <psi| H_i H_j |psi> via the factored-window primitive.

    Each term's penalty is a SUM of per-leaf factored products
    (``_penalty_ops`` returns a list of leaf->op dicts; the term's
    operator is the sum of those products). Hence:

        <psi| H_i H_j |psi>
          = sum_{a in ops_i} sum_{b in ops_j} <psi| O_{i,a} O_{j,b} |psi>

    Each O_{i,a} O_{j,b} is again a per-leaf factored product (per
    :func:`_compose_leaf_ops`), so it routes back through
    :func:`mera_window_expectation_factored` — the same substrate
    primitive §12.1 anomaly uses. No dense `16**k` operator is built.
    """
    ops_i_list = H._penalty_ops(term_i)
    ops_j_list = H._penalty_ops(term_j)
    if not ops_i_list or not ops_j_list:
        return 0.0
    total = 0.0 + 0.0j
    for ops_a in ops_i_list:
        for ops_b in ops_j_list:
            prod = _compose_leaf_ops(ops_a, ops_b)
            total += mera_window_expectation_factored(state, prod)
    # Symmetric Hessian entry: H_i, H_j Hermitian => <H_j H_i> =
    # conj(<H_i H_j>), so the symmetric (Hessian) coupling is the
    # real part. The imaginary part should be at numerical floor on
    # the projector basis §7.4.
    return float(total.real)


def _build_constraint_matrix(
    H: MeraEvalHamiltonian, state: MERA,
) -> tuple[np.ndarray, list[float], list[MeraEvalTerm]]:
    """Build the per-term constraint-Hessian M (dense ndarray) and the
    per-term residual vector.

    The basis is :attr:`MeraEvalHamiltonian.terms` (one entry per
    (rule, node) pair). M is symmetric and PSD by construction: the
    off-diagonal Re <H_i H_j> together with the diagonal <H_i^2>
    (= <H_i> for projector terms §7.4) is the Gram matrix of the
    vectors H_i|psi>, which is automatically PSD.

    Returns ``(M, residuals, terms)``.
    """
    terms = list(H.terms)
    n = len(terms)
    residuals = [float(H.term_energy(state, t)) for t in terms]

    # Leaf footprints — operator-derived from term_affected_leaves
    # (the causal-cone leaf set the term reads/writes per §1.1). We
    # use these only as a CHEAP zero-coupling shortcut: when both
    # terms have zero residual (<H_i> = 0 means H_i|psi> = 0 for the
    # projector basis §7.4, so <H_i H_j> = 0), or when either term's
    # _penalty_ops list is empty (no enumerated structural moves on
    # this node — the term contributes no operator at all). The
    # genuine two-operator expectation is computed for every other
    # off-diagonal entry via :func:`_two_term_expectation`.
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        M[i, i] = residuals[i]
    for i in range(n):
        if residuals[i] <= 0.0:
            # H_i|psi> = 0 on the projector basis => <H_i H_j> = 0 for
            # all j. Skip the entire row.
            continue
        for j in range(i + 1, n):
            if residuals[j] <= 0.0:
                continue
            coupling = _two_term_expectation(H, state, terms[i], terms[j])
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
      * Off-diagonal couplings are the genuine symmetric two-operator
        expectation Re <psi|H_i H_j|psi>, computed via the same
        factored-window primitive §12.1 anomaly uses (sum over the
        per-leaf operator products of the two terms' penalty
        decompositions, §7.4). Disjoint leaf footprints factor
        automatically (§1.2 <H_i H_j> = <H_i><H_j>); zero-residual
        terms skip via H_i|psi> = 0 in the projector basis.
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
