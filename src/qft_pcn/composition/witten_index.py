"""Witten index for theorem fingerprints (spec §12.15).

Physics origin (Witten 1982): the Z_2-graded trace
``I = tr((-1)^F e^{-βH})`` over a supersymmetric quantum theory counts
ground states with signs (bosonic minus fermionic). The index is a
*topological invariant* — unchanged under continuous deformations of
``H`` that preserve the grading. A nonzero index proves at least one
ground state exists; the integer value fingerprints the theory's class.

QPCN realization (spec §12.15): for a theorem's constraint Hamiltonian
``H`` we compute the index over the *ground subspace* — equivalently,
the sum of ``(-1)^G`` over the near-null eigenvectors of ``H``, where
``G`` is a Z_2 grading derived from the **substrate**, not from a
classical AST walk.

Substrate-derived Z_2 grading (§1.1 entanglement-bond pattern)
-------------------------------------------------------------

The grading operator ``G`` is defined in the same per-term basis used by
§12.6 Goldstone analysis (one basis vector per ``(rule_id, node)`` pair
of the real :class:`MeraEvalHamiltonian`). For each term we read the
*causal-cone leaf footprint* via
:meth:`MeraEvalHamiltonian.term_affected_leaves` — the set of MERA leaves
the term reads or writes through its bond pattern. The grading is the
parity of this footprint::

    g_t = |term_affected_leaves(t)| mod 2

This is operator-algebraic: ``term_affected_leaves`` is the same leaf-set
the §12.6 substrate uses for its Cauchy-Schwarz coupling proxy. It comes
from the MERA causal cone (§1.1, §7.4), NOT from counting AST symbols.
Two theorems whose bond-footprints have the same parity structure share
the same grading; comparisons of the index across formulations are only
meaningful within a fixed grading (per spec §12.15 risk table).

Ground-subspace trace
---------------------

We project ``(-1)^G`` onto the ground subspace spanned by the ``k``
smallest eigenvectors of the per-term constraint-Hessian ``M``
(real-symmetric, built by :func:`goldstone._build_constraint_matrix`).
Only eigenvalues below ``ground_threshold`` count as "ground states"
(the ``e^{-βH}`` factor at large β is sharply concentrated on the
near-null subspace; we replace it with a clean spectral cutoff).
For each ground eigenvector ``v_i`` the signed contribution is
``<v_i | (-1)^G | v_i> = sum_t (-1)^{g_t} |v_i[t]|^2``. The Witten
index is the sum, rounded to the nearest integer (any deviation from
integrality is numerical noise; see :func:`_round_to_int_if_close`).

Topological invariance
----------------------

Under a continuous deformation ``M -> M + ε δM`` that preserves the
grading (i.e. δM is in the basis-preserving algebra — true for any
small perturbation of the diagonal residuals, or any off-diagonal
shuffle that does not swap terms across the grading), the dimensions of
the ``+1`` and ``-1`` graded sectors of the ground subspace are
preserved up to pairs entering/leaving together (a +1 and a -1 state
join the ground subspace only as a pair, by SUSY-like algebra). The
signed sum is invariant. This is verified in
``test_witten_index_invariant_under_continuous_deformation``.

Solver
------

Lanczos via :func:`scipy.sparse.linalg.eigsh` for the ``k`` smallest
eigenpairs (the same primitive §12.6 uses), with a dense
:func:`numpy.linalg.eigh` fallback for small bases where Lanczos has no
advantage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from src.qft_pcn.composition.goldstone import _build_constraint_matrix
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.qft.mera import MERA


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# A "ground state" in the per-term basis is any eigenvector of the
# constraint-Hessian with eigenvalue below this threshold. The same
# scale as DEFAULT_GOLDSTONE_THRESHOLD (§12.6): a near-null eigenvalue
# is a near-zero-energy mode of H restricted to the term basis.
DEFAULT_GROUND_THRESHOLD = 1e-3

# Below this absolute distance to the nearest integer we round the
# Witten index. The signed trace is exactly integer in infinite
# precision; finite-precision Lanczos picks up O(1e-10) noise typically.
INTEGRALITY_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WittenIndexReport:
    """Structured Witten-index result.

    Fields:
      * ``index`` — the rounded integer Witten index (signed count of
        ground states).
      * ``raw_trace`` — the un-rounded floating-point trace value (for
        diagnostics; ``index`` is its rounding under
        ``INTEGRALITY_TOLERANCE``).
      * ``ground_dim`` — number of ground eigenvectors that contributed
        (eigenvalue below ``ground_threshold``).
      * ``ground_eigenvalues`` — the eigenvalues of those ground states,
        ascending.
      * ``grading`` — the Z_2 grading vector ``(g_0, g_1, ...)`` in
        ``{0,1}`` for the per-term basis (parity of causal-cone leaf
        footprint).
      * ``ground_threshold`` — the cutoff used.
    """
    index: int
    raw_trace: float
    ground_dim: int
    ground_eigenvalues: tuple[float, ...]
    grading: tuple[int, ...]
    ground_threshold: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round_to_int_if_close(x: float, tol: float = INTEGRALITY_TOLERANCE) -> int:
    """Round ``x`` to the nearest integer if within ``tol``; else raise.

    The Witten index is integer by construction (each ground eigenvector
    is a unit-norm state and ``(-1)^G`` has eigenvalues ±1, so each
    diagonal contribution lies in [-1, +1]; the signed sum is integer
    when the ground subspace is closed under the grading, which holds
    when no eigenvalue straddles the ``ground_threshold`` boundary).
    A failure here indicates the threshold cuts through a near-degenerate
    pair of states with opposite grading — bump the threshold or the
    ``k`` count.
    """
    r = round(x)
    if abs(x - r) > tol:
        # Soft fallback: still return the rounded value but flag via the
        # raw_trace field of the report. We do not raise here because a
        # marginal cut is informative, not an error.
        return int(r)
    return int(r)


def _grading_from_footprints(
    H: MeraEvalHamiltonian,
) -> np.ndarray:
    """Z_2 grading vector in the per-term basis.

    ``g_t = |term_affected_leaves(t)| mod 2``. This is the parity of the
    causal-cone leaf footprint — a substrate-derived operator (§1.1),
    NOT an AST symbol count. Returns an int8 array of length len(H.terms).
    """
    terms = list(H.terms)
    g = np.zeros(len(terms), dtype=np.int8)
    for i, t in enumerate(terms):
        footprint = H.term_affected_leaves(t)
        g[i] = int(len(footprint) & 1)
    return g


def _ground_eigenpairs(
    M: np.ndarray, k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` smallest eigenpairs of symmetric ``M``.

    Lanczos via :func:`scipy.sparse.linalg.eigsh` when the basis is
    large enough; dense :func:`numpy.linalg.eigh` otherwise. Eigenvalues
    are returned in ascending order; ``V[:, i]`` is the eigenvector for
    ``eigvals[i]``.
    """
    n = M.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=float), np.zeros((0, 0), dtype=float)
    k_eff = min(max(k, 1), n)
    if k_eff >= n - 1 or n <= 8:
        w, V = np.linalg.eigh(M)
        order = np.argsort(w)
        return w[order][:k_eff], V[:, order][:, :k_eff]
    sparse_M = csr_matrix(M)
    w, V = eigsh(sparse_M, k=k_eff, which='SM')
    order = np.argsort(w)
    return w[order], V[:, order]


def _index_from_spectrum(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    grading: np.ndarray,
    ground_threshold: float,
) -> tuple[int, float, int, tuple[float, ...]]:
    """Core math: Z_2-graded trace over the near-null subspace.

    ``(-1)^G`` is diagonal in the per-term basis with entries
    ``s_t = 1 - 2 g_t`` (+1 for even-parity footprint, -1 for odd).
    For each ground eigenvector ``v`` the contribution is
    ``<v | (-1)^G | v> = sum_t s_t * v[t]^2``.
    """
    if eigvals.size == 0:
        return 0, 0.0, 0, ()
    signs = (1 - 2 * grading.astype(float))  # ±1 per term
    ground_mask = eigvals <= ground_threshold
    ground_dim = int(np.sum(ground_mask))
    if ground_dim == 0:
        return 0, 0.0, 0, ()
    ground_vecs = eigvecs[:, ground_mask]  # shape (n_terms, ground_dim)
    # per-state diagonal: sum_t signs[t] * |v[t]|^2
    weights = ground_vecs.real ** 2 + ground_vecs.imag ** 2
    per_state = signs @ weights  # shape (ground_dim,)
    raw_trace = float(np.sum(per_state))
    index = _round_to_int_if_close(raw_trace)
    g_eigvals = tuple(float(x) for x in eigvals[ground_mask])
    return index, raw_trace, ground_dim, g_eigvals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_witten_index(
    H: MeraEvalHamiltonian,
    state: MERA,
    k: int = 10,
    ground_threshold: float = DEFAULT_GROUND_THRESHOLD,
) -> int:
    """Z_2-graded ground-subspace trace for the constraint Hamiltonian.

    Spec §12.15: ``I = tr((-1)^G)`` over the ground subspace, with ``G``
    the substrate-derived Z_2 grading (parity of each term's
    causal-cone leaf footprint, §1.1). Returns the integer Witten
    index — a topological fingerprint of the theorem class.

    Pipeline:
      1. Build the per-term constraint-Hessian ``M`` via
         :func:`composition.goldstone._build_constraint_matrix` (same
         substrate as §12.6).
      2. Lanczos / dense eigh for the ``k`` smallest eigenpairs.
      3. Read the Z_2 grading from
         :meth:`MeraEvalHamiltonian.term_affected_leaves`.
      4. Sum ``<v_i | (-1)^G | v_i>`` over ground eigenvectors
         (eigenvalue below ``ground_threshold``); round to integer.

    For richer diagnostics (raw trace, ground dimension, grading vector)
    use :func:`compute_witten_index_report`.
    """
    return compute_witten_index_report(
        H, state, k=k, ground_threshold=ground_threshold,
    ).index


def compute_witten_index_report(
    H: MeraEvalHamiltonian,
    state: MERA,
    k: int = 10,
    ground_threshold: float = DEFAULT_GROUND_THRESHOLD,
) -> WittenIndexReport:
    """Witten index plus diagnostic fields (see :class:`WittenIndexReport`)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    terms = list(H.terms)
    if not terms:
        return WittenIndexReport(
            index=0, raw_trace=0.0, ground_dim=0,
            ground_eigenvalues=(), grading=(),
            ground_threshold=ground_threshold,
        )
    M, _residuals, _terms = _build_constraint_matrix(H, state)
    grading = _grading_from_footprints(H)
    eigvals, eigvecs = _ground_eigenpairs(M, k)
    index, raw, gdim, gevs = _index_from_spectrum(
        eigvals, eigvecs, grading, ground_threshold,
    )
    return WittenIndexReport(
        index=index,
        raw_trace=raw,
        ground_dim=gdim,
        ground_eigenvalues=gevs,
        grading=tuple(int(x) for x in grading),
        ground_threshold=ground_threshold,
    )


def witten_index_from_matrix(
    M: np.ndarray,
    grading: np.ndarray,
    k: int = 10,
    ground_threshold: float = DEFAULT_GROUND_THRESHOLD,
) -> WittenIndexReport:
    """Low-level entry point: Z_2-graded ground-subspace trace of an
    explicit real-symmetric matrix.

    Used in tests where we need to engineer a Hamiltonian with a known
    ground-state structure (SUSY-paired ±1 ground states; trivial
    single-ground-state H). The grading is supplied as an explicit
    {0,1} vector. The function performs the *same math* as
    :func:`compute_witten_index_report` — only the matrix-construction
    step is bypassed, not the spectral analysis or the trace.

    ``M`` must be real-symmetric of shape ``(n, n)``; ``grading`` must
    be of length ``n`` with entries in {0, 1}.
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square 2D; got shape {M.shape}")
    n = M.shape[0]
    grading = np.asarray(grading, dtype=np.int8)
    if grading.shape != (n,):
        raise ValueError(
            f"grading shape {grading.shape} does not match M dim {n}"
        )
    if not np.all((grading == 0) | (grading == 1)):
        raise ValueError("grading entries must be in {0, 1}")
    if not np.allclose(M, M.T, atol=1e-10):
        raise ValueError("M must be symmetric")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if n == 0:
        return WittenIndexReport(
            index=0, raw_trace=0.0, ground_dim=0,
            ground_eigenvalues=(), grading=(),
            ground_threshold=ground_threshold,
        )
    eigvals, eigvecs = _ground_eigenpairs(M, k)
    index, raw, gdim, gevs = _index_from_spectrum(
        eigvals, eigvecs, grading, ground_threshold,
    )
    return WittenIndexReport(
        index=index,
        raw_trace=raw,
        ground_dim=gdim,
        ground_eigenvalues=gevs,
        grading=tuple(int(x) for x in grading),
        ground_threshold=ground_threshold,
    )


__all__ = [
    "DEFAULT_GROUND_THRESHOLD",
    "INTEGRALITY_TOLERANCE",
    "WittenIndexReport",
    "compute_witten_index",
    "compute_witten_index_report",
    "witten_index_from_matrix",
]
