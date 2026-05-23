"""Acceptance tests for §12.15 Witten-index theorem fingerprints.

These tests exercise the real spectral pipeline:
  * ``_ground_eigenpairs`` runs the real :func:`numpy.linalg.eigh` (or
    :func:`scipy.sparse.linalg.eigsh` for larger bases) on a
    real-symmetric matrix.
  * The Z_2 grading is operator-derived (causal-cone leaf footprint
    parity for the §12.6 substrate-based path; explicit {0,1} vectors
    for the engineered-matrix path).
  * No mocks, no stubs, no skips.

The engineered-matrix tests (``witten_index_from_matrix``) verify the
core trace math against analytically-known ground-state structures
(trivial / SUSY-paired). The substrate test
(``compute_witten_index``) confirms the same pipeline runs end-to-end
on a real :class:`MeraEvalHamiltonian` built from a real AST.

Continuous-deformation invariance is checked by perturbing the
engineered matrix with a small grading-preserving δM and confirming
the integer index is unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.witten_index import (
    DEFAULT_GROUND_THRESHOLD,
    WittenIndexReport,
    compute_witten_index,
    compute_witten_index_report,
    witten_index_from_matrix,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian


# ---------------------------------------------------------------------------
# Opt this file out of the §9.7 dense-tensor memory ceiling (same as
# test_goldstone.py: encode_mera allocates 16-dim leaves by construction).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_large_dense():
    yield


# ---------------------------------------------------------------------------
# §12.15 acceptance — engineered matrices
# ---------------------------------------------------------------------------


def test_trivial_h_index_is_one():
    """Single ground state with grading +1 (even leaf-footprint parity)
    → Witten index = +1.

    Engineered H: a diagonal matrix with one zero entry (the ground
    state) and the rest above the ground threshold. The ground basis
    vector is e_0 with grading 0 (even), so ``(-1)^G`` acts as +1 on
    it.
    """
    n = 6
    M = np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    grading = np.array([0, 0, 0, 0, 0, 0], dtype=np.int8)  # all even
    report = witten_index_from_matrix(
        M, grading, k=4,
        ground_threshold=DEFAULT_GROUND_THRESHOLD,
    )
    assert isinstance(report, WittenIndexReport)
    assert report.ground_dim == 1, (
        f"expected exactly one ground state, got {report.ground_dim}"
    )
    assert report.index == 1, (
        f"expected I=+1 for trivial H with even-parity ground state, "
        f"got {report.index} (raw_trace={report.raw_trace})"
    )


def test_trivial_h_odd_grading_index_is_minus_one():
    """Single ground state with grading -1 (odd leaf-footprint parity)
    → Witten index = -1.

    Complement of the previous test: the same trivial diagonal H but
    with the ground basis vector graded as odd. ``(-1)^G`` acts as -1
    on it, giving a signed count of -1. Confirms the grading actually
    enters the trace with the correct sign.
    """
    n = 6
    M = np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    grading = np.array([1, 0, 0, 0, 0, 0], dtype=np.int8)  # ground is odd
    report = witten_index_from_matrix(
        M, grading, k=4,
        ground_threshold=DEFAULT_GROUND_THRESHOLD,
    )
    assert report.ground_dim == 1
    assert report.index == -1, (
        f"expected I=-1 for trivial H with odd-parity ground state, "
        f"got {report.index} (raw_trace={report.raw_trace})"
    )


def test_supersymmetric_pair_cancels():
    """Engineered SUSY-like H with paired ±1 ground states → index = 0.

    Two ground basis vectors at eigenvalue 0, one graded even, one
    graded odd. ``(-1)^G`` gives +1 on the first and -1 on the second;
    they cancel. This is the canonical SUSY scenario where the index
    proves nothing about ground-state existence (it could still be
    zero with multiple ground states, just paired).
    """
    n = 6
    # Two ground states at eigvals 0, four lifted states at 1..4.
    M = np.diag([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    grading = np.array([0, 1, 0, 0, 0, 0], dtype=np.int8)
    report = witten_index_from_matrix(
        M, grading, k=4,
        ground_threshold=DEFAULT_GROUND_THRESHOLD,
    )
    assert report.ground_dim == 2, (
        f"expected two ground states (SUSY pair), got {report.ground_dim}"
    )
    assert report.index == 0, (
        f"expected I=0 from ±1 cancellation, got {report.index} "
        f"(raw_trace={report.raw_trace})"
    )


def test_witten_index_invariant_under_continuous_deformation():
    """Perturb H by a small grading-preserving δM; index unchanged.

    Topological invariance of the Witten index (spec §12.15): under a
    continuous deformation of H that does not push eigenvalues across
    the ground-threshold boundary, the signed ground-state count is
    preserved.

    Construction: take a 6-dim H with two ground states (one even, one
    odd) and four lifted states (mixed parity). Reference index = 0
    from the SUSY-pair structure. Add ε * random_symmetric_perturbation
    (||δM|| ~ 1e-2) and confirm:
      * the ground subspace dimension is preserved (eigenvalues stay
        below threshold),
      * the index is still 0.

    Then take a single-ground-state H (index = +1) and perturb it the
    same way — index still +1. Demonstrates invariance for both the
    cancellation and the non-trivial case.
    """
    rng = np.random.default_rng(seed=20260523)

    # --- Case A: SUSY pair, index = 0 -----------------------------------
    n = 6
    base_diag = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    M0 = np.diag(base_diag)
    grading = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    base_report = witten_index_from_matrix(
        M0, grading, k=4, ground_threshold=DEFAULT_GROUND_THRESHOLD,
    )
    assert base_report.index == 0
    base_ground_dim = base_report.ground_dim

    # δM: small symmetric perturbation. The grading is fixed in the
    # term basis (it does not depend on M), so any matrix perturbation
    # is automatically "grading-preserving" in the sense relevant to
    # §12.15 (the operator (-1)^G is unchanged). Topological
    # invariance then reduces to: ground subspace dimension and signed
    # composition stay constant under small symmetric perturbations
    # whose eigenvalues do not cross the ground threshold.
    epsilon = 1e-3  # well below the gap (≈ 1.0)
    P = rng.standard_normal((n, n))
    deltaM = epsilon * (P + P.T) * 0.5
    perturbed_report = witten_index_from_matrix(
        M0 + deltaM, grading, k=4,
        ground_threshold=DEFAULT_GROUND_THRESHOLD,
    )
    # Ground dim preserved (eigenvalues should not cross threshold for
    # ε << gap; the unperturbed gap is 1.0, ε = 1e-3).
    assert perturbed_report.ground_dim == base_ground_dim, (
        f"ground dimension changed under perturbation: "
        f"{base_ground_dim} -> {perturbed_report.ground_dim} "
        f"(eigvals = {perturbed_report.ground_eigenvalues})"
    )
    assert perturbed_report.index == 0, (
        f"Witten index changed under perturbation: "
        f"0 -> {perturbed_report.index} "
        f"(raw_trace={perturbed_report.raw_trace})"
    )

    # --- Case B: single ground state, index = +1 ------------------------
    M1 = np.diag([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    grading_b = np.array([0, 1, 0, 1, 0, 1], dtype=np.int8)
    base_b = witten_index_from_matrix(
        M1, grading_b, k=4, ground_threshold=DEFAULT_GROUND_THRESHOLD,
    )
    assert base_b.index == 1 and base_b.ground_dim == 1
    Q = rng.standard_normal((n, n))
    deltaM_b = epsilon * (Q + Q.T) * 0.5
    perturbed_b = witten_index_from_matrix(
        M1 + deltaM_b, grading_b, k=4,
        ground_threshold=DEFAULT_GROUND_THRESHOLD,
    )
    assert perturbed_b.ground_dim == 1, (
        f"single-ground-state dim changed: 1 -> {perturbed_b.ground_dim}"
    )
    assert perturbed_b.index == 1, (
        f"non-trivial Witten index changed under perturbation: "
        f"1 -> {perturbed_b.index}"
    )


# ---------------------------------------------------------------------------
# §12.15 acceptance — real substrate (MeraEvalHamiltonian)
# ---------------------------------------------------------------------------


def test_substrate_witten_index_is_integer():
    """End-to-end pipeline on a real MeraEvalHamiltonian.

    Builds a real AST (``2 + 3``), encodes it via the real
    ``encode_mera``, constructs the real ``MeraEvalHamiltonian``, and
    runs :func:`compute_witten_index_report`. Confirms:
      * the grading vector is derived from real ``term_affected_leaves``
        sets (length matches H.terms count),
      * the index is an int,
      * the raw trace is close to the integer index (integrality
        sanity).

    This is the substrate-derived path (§1.1 entanglement-bond pattern,
    not AST symbol counting). It does NOT pin a specific integer
    value — the integer depends on the encoder's leaf layout, which
    is not load-bearing for this spec — but it does confirm the
    pipeline produces a well-defined integer fingerprint.
    """
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    report = compute_witten_index_report(H, state, k=8)
    assert isinstance(report, WittenIndexReport)
    assert isinstance(report.index, int)
    assert len(report.grading) == len(H.terms), (
        f"grading dim {len(report.grading)} != #terms {len(H.terms)}"
    )
    # Grading entries are all in {0, 1} — confirms substrate-derived
    # parity (not stray real values).
    assert all(g in (0, 1) for g in report.grading)
    # Integrality: raw trace is within INTEGRALITY_TOLERANCE of int.
    assert abs(report.raw_trace - report.index) < 1e-3, (
        f"raw_trace {report.raw_trace} not close to index {report.index}"
    )


def test_substrate_compute_witten_index_returns_int():
    """``compute_witten_index`` returns a plain int, matching the
    detailed report's ``index`` field. Confirms the convenience wrapper
    is consistent with the report-returning path."""
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    idx = compute_witten_index(H, state, k=8)
    report = compute_witten_index_report(H, state, k=8)
    assert isinstance(idx, int)
    assert idx == report.index
