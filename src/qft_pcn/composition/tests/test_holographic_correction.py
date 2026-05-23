"""Tests for §12.5 holographic-code QEC on sub-QPCN consistency.

Spec: ``QFT_PCN_ARCHITECTURE.md`` §12.5. The MERA substrate is literally a
holographic code (Pastawski-Yoshida-Harlow-Preskill 2015). Inconsistent
sub-QPCN children produce stabilizer-syndrome mismatches against the parent.

These tests use REAL ``encode_mera`` output and the substrate's ascending
superoperator — no mock MERAs, no classical AST comparison.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.holographic_correction import (
    CorruptionReport,
    StabilizerSyndromes,
    compute_stabilizer_syndromes,
    detect_logical_corruption,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


# ---------------------------------------------------------------------------
# §12.5 — consistent children produce zero syndrome
# ---------------------------------------------------------------------------


def test_compute_stabilizer_syndromes_shape_and_content():
    """Basic sanity: the syndrome has the documented shape and is real."""
    state, _ = encode_mera(parse(r"\x:Int. x"))
    syn = compute_stabilizer_syndromes(state)
    assert isinstance(syn, StabilizerSyndromes)
    N = state.N
    d = state.d_local
    assert syn.leaf_expectations.shape == (N, d + d - 1)
    # All entries are real (the stabilizers are Hermitian).
    assert np.all(np.isfinite(syn.leaf_expectations))
    # Leaf marginals (the diagonal-projector block) sum to 1 per leaf for a
    # normalized state — that's the probabilistic content of the boundary.
    diag_block = syn.leaf_expectations[:, :d]
    sums = diag_block.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), sums
    # Top density is Hermitian PSD-like.
    rho = syn.top_density
    assert rho.shape[0] == rho.shape[1]
    assert np.allclose(rho, rho.conj().T, atol=1e-10)


def test_consistent_children_have_zero_syndrome():
    """Coherent sub-QPCNs produce matching stabilizers.

    Three children are independently re-encoded from the same source as the
    parent. Their holographic-code syndromes must match the parent's within
    floating-point tolerance — distance is zero (well below the default
    threshold), and ``flagged`` is empty.
    """
    src = r"\x:Int. \y:Int. x + y"
    parent, _ = encode_mera(parse(src))
    # Independent encodings of the same source: same MERA tree, no
    # corruption injected. Use distinct calls so they're separate objects.
    children = {
        "c0": encode_mera(parse(src))[0],
        "c1": encode_mera(parse(src))[0],
        "c2": encode_mera(parse(src))[0],
    }
    report = detect_logical_corruption(parent, children)
    assert isinstance(report, CorruptionReport)
    assert set(report.child_syndromes.keys()) == {"c0", "c1", "c2"}
    for cid, d in report.syndrome_distances.items():
        assert d < 1e-8, f"child {cid} unexpectedly diverged: distance={d}"
    assert report.flagged == ()


# ---------------------------------------------------------------------------
# §12.5 — corrupted child surfaces syndrome
# ---------------------------------------------------------------------------


def test_corrupted_child_surfaces_syndrome():
    """Inject a real perturbation; detect_logical_corruption flags it.

    The corruption is a single-leaf Pauli-like operator (the discrete
    analog of an X-error on a code site) applied to one leaf of the child.
    Per §12.5 the holographic code must surface this as a syndrome — the
    flagged set contains exactly the corrupted child.
    """
    src = r"\x:Int. x + 1"
    parent, _ = encode_mera(parse(src))
    # Three coherent children and one corrupted child.
    clean = encode_mera(parse(src))[0]
    corrupted = encode_mera(parse(src))[0]
    # Apply a non-identity leaf rotation: cyclic permutation of basis
    # vectors at leaf 0. This is a Pauli-X^k style boundary error — exactly
    # the noise model §12.5 protects against. It does NOT touch the global
    # tree structure, so the syndrome must come from the BULK reconstruction
    # via the ascending superoperator, not from any structural diff.
    d = corrupted.d_local
    perm = np.zeros((d, d), dtype=complex)
    for k in range(d):
        perm[(k + 1) % d, k] = 1.0
    corrupted.apply_local_gate(0, perm)
    children = {
        "ok_a": clean,
        "ok_b": encode_mera(parse(src))[0],
        "corrupt": corrupted,
    }
    report = detect_logical_corruption(parent, children, threshold=1e-4)
    # The clean children pass.
    assert "ok_a" not in report.flagged
    assert "ok_b" not in report.flagged
    # The corrupted child is flagged.
    assert "corrupt" in report.flagged
    # And its distance dominates the clean children's distances.
    assert report.syndrome_distances["corrupt"] > 1e-3
    assert report.syndrome_distances["corrupt"] > 100 * max(
        report.syndrome_distances["ok_a"],
        report.syndrome_distances["ok_b"],
        1e-15,
    )


def test_structural_mismatch_is_infinite_syndrome():
    """A child whose MERA shape differs from the parent triggers the
    structural-level syndrome before any tensor comparison.

    Two programs with different node counts (hence different leaf counts
    after the encoder pads) produce MERAs that the holographic code cannot
    align — the syndrome is +inf and the child is flagged.
    """
    parent, _ = encode_mera(parse(r"\x:Int. x"))
    # A larger program → more leaves / different layer_dims.
    big, _ = encode_mera(parse(r"\x:Int. \y:Int. \z:Int. x + y + z"))
    if big.N == parent.N and big.layer_dims == parent.layer_dims:
        pytest.skip("encoder padded both programs to the same shape")
    report = detect_logical_corruption(parent, {"shape_diff": big})
    assert "shape_diff" in report.flagged
    assert report.syndrome_distances["shape_diff"] == float('inf')


# ---------------------------------------------------------------------------
# §1.1 — entanglement preservation: correction must not violate
# Forall-protected leaves.
# ---------------------------------------------------------------------------


def test_section_1_1_entanglement_preservation():
    """Syndrome computation is read-only — Forall-protected leaves and
    every other tensor are untouched.

    Per §1.1, the encoder marks certain leaves as Forall-protected (binding
    sites + species leaves of bound Var uses) and those leaves carry the
    universal-quantification entanglement. The holographic-code syndrome
    measurement MUST be a passive observation; it cannot mutate the state
    (no decohering measurement collapse) or the §1.1 guarantee is broken.

    Verification: capture every leaf, disentangler, isometry and the top
    tensor before computing the syndrome; assert byte-identical after. This
    covers both ``compute_stabilizer_syndromes`` and the full
    ``detect_logical_corruption`` pipeline.
    """
    from src.qft_pcn.logic.ast import (
        Forall, Fix, Eq, NatLit, Var, Bin, TNat, TProp,
    )
    n = Var(name="n")
    ast = Forall(
        param="n", param_ty=TNat(),
        body=Fix(param="ind", param_ty=TProp(),
                 body=Eq(lhs=Bin(op="+", lhs=n, rhs=NatLit(val=0)), rhs=n)),
    )
    state, meta = encode_mera(ast)
    # Sanity: encoder did mark Forall-protected leaves.
    assert len(meta.forall_protected_leaves) > 0, \
        "Forall AST must produce protected leaves"

    # Snapshot every tensor.
    leaves_before = [s.copy() for s in state.leaves]
    diss_before = [[u.copy() for u in layer] for layer in state.disentanglers]
    inter_before = [[u.copy() for u in layer]
                    for layer in state.inter_disentanglers]
    iso_before = [[w.copy() for w in layer] for layer in state.isometries]
    top_before = state.top.copy()
    mv_before = state._mutation_version

    syn = compute_stabilizer_syndromes(state)
    assert isinstance(syn, StabilizerSyndromes)

    # Also run the full corruption-detect pipeline (two coherent children).
    child = encode_mera(ast)[0]
    detect_logical_corruption(state, {"c": child})

    # No leaf, disentangler, isometry, or top was modified.
    for k, before in enumerate(leaves_before):
        assert np.array_equal(state.leaves[k], before), \
            f"leaf {k} mutated by syndrome read"
    for ell, layer in enumerate(diss_before):
        for j, u in enumerate(layer):
            assert np.array_equal(state.disentanglers[ell][j], u), \
                f"disentangler ({ell},{j}) mutated"
    for ell, layer in enumerate(inter_before):
        for j, u in enumerate(layer):
            assert np.array_equal(state.inter_disentanglers[ell][j], u), \
                f"inter-disentangler ({ell},{j}) mutated"
    for ell, layer in enumerate(iso_before):
        for j, w in enumerate(layer):
            assert np.array_equal(state.isometries[ell][j], w), \
                f"isometry ({ell},{j}) mutated"
    assert np.array_equal(state.top, top_before)
    # The mutation_version field is the in-substrate liveness check for
    # the §1.1 entanglement integrity. A passive measurement must NOT
    # bump it.
    assert state._mutation_version == mv_before, \
        "syndrome read bumped mutation_version — would invalidate §1.1 "\
        "entanglement caches even though no tensor changed"


# ---------------------------------------------------------------------------
# Edge cases / parameter validation
# ---------------------------------------------------------------------------


def test_threshold_negative_rejected():
    parent, _ = encode_mera(parse(r"\x:Int. x"))
    with pytest.raises(ValueError):
        detect_logical_corruption(parent, {}, threshold=-1.0)


def test_compute_rejects_non_mera():
    with pytest.raises(TypeError):
        compute_stabilizer_syndromes("not a mera")  # type: ignore[arg-type]


def test_empty_children_yields_empty_report():
    parent, _ = encode_mera(parse(r"\x:Int. x"))
    rep = detect_logical_corruption(parent, {})
    assert rep.flagged == ()
    assert rep.child_syndromes == {}
    assert rep.syndrome_distances == {}
