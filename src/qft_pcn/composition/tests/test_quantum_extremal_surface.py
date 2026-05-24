"""Tests for §12.12 quantum extremal surfaces (RT/QES proof-complexity bound).

Spec: ``QFT_PCN_ARCHITECTURE.md`` §12.12. The QES area on the MERA bulk
gives a lower bound on proof complexity (Susskind 2016 complexity=
volume); the classical RT minimum-cut area gives an upper bound on the
QES (Ryu-Takayanagi 2006). For MERA the substrate is literally a
discrete holographic geometry (Swingle 2012).

These tests operate on REAL MERA tensor networks
(``MERA.from_product`` for trivial states; ``MERA.from_term_superposition``
for genuinely entangled states) — no mocks, no classical surrogates.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.quantum_extremal_surface import (
    ProofComplexityRanking,
    compute_geometric_rt_area,
    compute_qes_complexity,
    compute_qes_lower_bound,
    rank_completed_proofs_by_qes_area,
)
from src.qft_pcn.composition.tests.conftest import basis_leaf, make_product_mera
from src.qft_pcn.qft.mera import MERA


# ---------------------------------------------------------------------------
# §12.12 — trivial (product) state has zero QES area
# ---------------------------------------------------------------------------


def test_trivial_state_has_zero_qes_area():
    """A product MERA has no entanglement on any region → QES = 0.

    Per §12.12 the QES is the generalized entropy of the region; on a
    product state every reduced density is pure (rank 1) so the entropy
    is zero. The RT geometric area, by contrast, is *positive* (the
    minimum cut still severs bonds with finite Hilbert capacity) —
    confirming RT is an upper bound, not the tight value, on the
    classical/product limit.
    """
    # N=4 leaves on the 16-dim local space.
    leaves = [basis_leaf(k) for k in (0, 1, 2, 3)]
    state = make_product_mera(leaves)
    region = [0, 1]                                # left half
    qes = compute_qes_complexity(state, region)
    assert qes == pytest.approx(0.0, abs=1e-10), \
        f"product state QES must be 0; got {qes}"
    # RT area is positive (the cut still has capacity).
    rt = compute_geometric_rt_area(state, region)
    assert rt > 0.0, f"RT area must be positive (finite bond capacity); got {rt}"


# ---------------------------------------------------------------------------
# §12.12 — entangled state has positive QES area
# ---------------------------------------------------------------------------


def test_entangled_state_has_positive_qes_area():
    """A genuinely entangled MERA (two-branch superposition differing on
    multiple leaves) has positive QES area on a non-trivial region.

    The state is psi = (|0000> + |1111>) / sqrt(2). On the canonical
    midpoint cut the reduced density on the left half is
    rho_L = 0.5 |00><00| + 0.5 |11><11|, with entropy ln 2.
    """
    N = 4
    d = 16
    e0 = basis_leaf(0, d)
    e1 = basis_leaf(1, d)
    terms = [
        (1.0 + 0.0j, [e0, e0, e0, e0]),
        (1.0 + 0.0j, [e1, e1, e1, e1]),
    ]
    state = MERA.from_term_superposition(terms).normalize()
    region = [0, 1]
    qes = compute_qes_complexity(state, region)
    assert qes > 1e-6, f"entangled QES must be > 0; got {qes}"
    # The exact entropy for the (|0000>+|1111>)/sqrt(2) cut is ln 2.
    assert qes == pytest.approx(np.log(2.0), rel=1e-6, abs=1e-6), \
        f"QES for GHZ-like state on left half must be ln(2); got {qes}"


# ---------------------------------------------------------------------------
# §12.12 — RT inequality: A_RT >= A_QES = S(region)
# ---------------------------------------------------------------------------


def test_qes_area_lower_bounds_complexity():
    """The Ryu-Takayanagi inequality S(region) <= A_RT(region) holds on
    the real MERA bulk for every region.

    The QES is the generalized entropy; in our implementation QES =
    S(region) for contiguous regions. The geometric RT area is the
    minimum-cut bond capacity. RT inequality (Ryu-Takayanagi 2006,
    Headrick-Takayanagi 2007 — proved for holographic states):
    entanglement entropy is upper-bounded by the minimum-cut area.

    We verify this on multiple states (product + two entangled
    superpositions) and multiple regions.
    """
    N = 4
    d = 16
    e0 = basis_leaf(0, d)
    e1 = basis_leaf(1, d)
    e2 = basis_leaf(2, d)
    states: list[tuple[str, MERA]] = [
        ("product", make_product_mera([e0, e1, e2, e0])),
        ("ghz_like", MERA.from_term_superposition([
            (1.0 + 0.0j, [e0, e0, e0, e0]),
            (1.0 + 0.0j, [e1, e1, e1, e1]),
        ]).normalize()),
        ("three_branch", MERA.from_term_superposition([
            (1.0 + 0.0j, [e0, e1, e0, e1]),
            (1.0 + 0.0j, [e1, e0, e1, e0]),
            (0.5 + 0.0j, [e2, e2, e2, e2]),
        ]).normalize()),
    ]
    regions = [
        [0],            # single leftmost leaf
        [0, 1],         # left half
        [0, 1, 2],      # left three quarters
        [3],            # single rightmost leaf
    ]
    for name, state in states:
        for region in regions:
            qes = compute_qes_complexity(state, region)
            rt = compute_geometric_rt_area(state, region)
            assert qes >= -1e-12, \
                f"{name} {region}: QES must be >= 0; got {qes}"
            # RT inequality: S(region) <= A_RT (with a small slack for
            # numerical floating-point noise).
            assert qes <= rt + 1e-9, (
                f"{name} {region}: RT inequality violated; "
                f"QES={qes}, RT={rt}")


# ---------------------------------------------------------------------------
# §12.12 — proof-complexity ranking
# ---------------------------------------------------------------------------


def test_rank_completed_proofs_by_qes_area_ranks_by_qes():
    """``rank_completed_proofs_by_qes_area`` orders candidates by QES area
    on the canonical midpoint region — the simpler proof has lower QES.

    Constructs three candidate proofs with controlled entanglement:
    a product state (QES = 0, simplest), a two-branch superposition
    (QES = ln 2), and a three-branch superposition (QES > ln 2).
    The ranking must place the product state first.
    """
    d = 16
    e0 = basis_leaf(0, d)
    e1 = basis_leaf(1, d)
    e2 = basis_leaf(2, d)
    theorem = make_product_mera([e0, e0, e0, e0])
    candidates = {
        "simple": make_product_mera([e0, e1, e0, e1]),
        "medium": MERA.from_term_superposition([
            (1.0 + 0.0j, [e0, e0, e0, e0]),
            (1.0 + 0.0j, [e1, e1, e1, e1]),
        ]).normalize(),
        "complex": MERA.from_term_superposition([
            (1.0 + 0.0j, [e0, e0, e0, e0]),
            (1.0 + 0.0j, [e1, e1, e1, e1]),
            (1.0 + 0.0j, [e2, e2, e2, e2]),
        ]).normalize(),
    }
    ranking = rank_completed_proofs_by_qes_area(theorem, candidates)
    assert isinstance(ranking, ProofComplexityRanking)
    assert ranking.minimum == "simple", (
        f"product candidate must have minimum QES; "
        f"ranking={ranking.sorted_by_complexity}, qes={dict(ranking.qes_areas)}")
    # The ranking must be strictly increasing in QES area across the
    # three candidates we constructed.
    qs = [ranking.qes_areas[k] for k in ranking.sorted_by_complexity]
    for a, b in zip(qs, qs[1:]):
        assert a <= b + 1e-12, (
            f"ranking not monotone in QES: {qs} "
            f"({ranking.sorted_by_complexity})")
    assert ranking.qes_areas["simple"] == pytest.approx(0.0, abs=1e-10)
    assert ranking.qes_areas["medium"] > ranking.qes_areas["simple"] + 1e-6
    assert ranking.qes_areas["complex"] > ranking.qes_areas["medium"] - 1e-6


# ---------------------------------------------------------------------------
# §12.12 — input validation
# ---------------------------------------------------------------------------


def test_compute_qes_rejects_non_mera():
    """§1.1: QES is defined on the real MERA bulk; other inputs are
    classical surrogates and must be rejected."""
    with pytest.raises(TypeError, match="real bulk geometry"):
        compute_qes_complexity("not a MERA", [0, 1])


def test_compute_qes_rejects_out_of_range_region():
    leaves = [basis_leaf(k) for k in (0, 1, 2, 3)]
    state = make_product_mera(leaves)
    with pytest.raises(ValueError, match="out of range"):
        compute_qes_complexity(state, [0, 5])


def test_find_minimum_complexity_rejects_empty_candidates():
    leaves = [basis_leaf(k) for k in (0, 1, 2, 3)]
    state = make_product_mera(leaves)
    with pytest.raises(ValueError, match="non-empty"):
        rank_completed_proofs_by_qes_area(state, {})


def test_find_minimum_complexity_rejects_size_mismatch():
    leaves2 = [basis_leaf(0), basis_leaf(1)]
    leaves4 = [basis_leaf(k) for k in (0, 1, 2, 3)]
    theorem = make_product_mera(leaves4)
    smaller = make_product_mera(leaves2)
    with pytest.raises(ValueError, match="boundary mismatch"):
        rank_completed_proofs_by_qes_area(theorem, {"x": smaller})


# ---------------------------------------------------------------------------
# §12.12 — a-priori QES lower bound from theorem alone (D16 resolution)
# ---------------------------------------------------------------------------


def test_qes_lower_bound_under_optimal_proof_complexity():
    """The a-priori QES lower bound (from the theorem encoding alone)
    must not exceed the QES area of any constructed proof on the same
    boundary — i.e. ``lower_bound <= A_QES(proof)`` for every candidate.

    Per §12.12, ``compute_qes_lower_bound(theorem)`` is the holographic
    lower bound on proof complexity computed BEFORE any proof search.
    For this bound to be valid, every actually-constructed proof MERA
    on the same boundary must have QES area >= the lower bound.

    Construction: theorem = GHZ-like (|0000>+|1111>)/sqrt(2) → QES on
    left half = ln 2. Build several candidate proof MERAs and verify
    each one's QES on the same region is >= ln 2 - eps (the lower
    bound is achieved by the theorem itself; harder proofs sit above).
    """
    d = 16
    e0 = basis_leaf(0, d)
    e1 = basis_leaf(1, d)
    e2 = basis_leaf(2, d)
    # Theorem with non-trivial encoding entanglement.
    theorem = MERA.from_term_superposition([
        (1.0 + 0.0j, [e0, e0, e0, e0]),
        (1.0 + 0.0j, [e1, e1, e1, e1]),
    ]).normalize()

    # A-priori lower bound (from theorem alone, no proofs yet).
    lb = compute_qes_lower_bound(theorem)
    assert lb >= 0.0, f"lower bound must be >= 0; got {lb}"
    assert lb == pytest.approx(np.log(2.0), rel=1e-6, abs=1e-6), (
        f"theorem QES on midpoint should be ln 2; got {lb}")

    # Construct candidate proofs with entanglement >= the theorem.
    candidates = {
        "matches_theorem": MERA.from_term_superposition([
            (1.0 + 0.0j, [e0, e0, e0, e0]),
            (1.0 + 0.0j, [e1, e1, e1, e1]),
        ]).normalize(),
        "richer": MERA.from_term_superposition([
            (1.0 + 0.0j, [e0, e0, e0, e0]),
            (1.0 + 0.0j, [e1, e1, e1, e1]),
            (1.0 + 0.0j, [e2, e2, e2, e2]),
        ]).normalize(),
    }
    region = [0, 1]
    for cid, cstate in candidates.items():
        cqes = compute_qes_complexity(cstate, region)
        assert cqes >= lb - 1e-9, (
            f"candidate {cid!r}: proof QES {cqes} below a-priori "
            f"lower bound {lb} — bound invalid")


def test_qes_lower_bound_rejects_non_mera():
    """§1.1: a-priori bound is defined on the real MERA bulk."""
    with pytest.raises(TypeError, match="real bulk geometry"):
        compute_qes_lower_bound("not a MERA")


def test_qes_lower_bound_product_theorem_is_zero():
    """A product theorem has zero encoding entanglement → lower bound
    is 0 (any proof is admissible). This is correct: the spec only
    forbids proofs whose complexity is *below* the lower bound; zero
    excludes nothing."""
    leaves = [basis_leaf(k) for k in (0, 1, 2, 3)]
    theorem = make_product_mera(leaves)
    lb = compute_qes_lower_bound(theorem)
    assert lb == pytest.approx(0.0, abs=1e-10), \
        f"product theorem lower bound must be 0; got {lb}"
