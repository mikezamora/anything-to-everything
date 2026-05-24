"""Tests for src/qft_pcn/qft/mera.py — binary 1D MERA substrate.

Spec: docs/superpowers/specs/2026-05-21-mera-substrate-design.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.qft.mera import (
    MERAError, InvalidLayerCount, CausalConeViolation,
    LayerDimMismatch, IsometryViolation, UnitaryViolation,
)


def test_mera_exceptions_inherit_from_mera_error():
    for cls in (InvalidLayerCount, CausalConeViolation,
                LayerDimMismatch, IsometryViolation, UnitaryViolation):
        assert issubclass(cls, MERAError)


def test_invalid_layer_count_message():
    with pytest.raises(InvalidLayerCount, match="N=6 is not a positive power of 2"):
        raise InvalidLayerCount(N=6)


def test_layer_dim_mismatch_message():
    with pytest.raises(LayerDimMismatch, match="layer 2: expected dim 16, got 8"):
        raise LayerDimMismatch(layer=2, expected=16, got=8)


from src.qft_pcn.qft.mera import layer_dims, causal_cone_path


def test_layer_dims_capped_at_chi_layer():
    # d_local=4, L=3, chi_layer=8:
    # dims[0] = 4
    # dims[1] = min(8, 4*4=16) = 8
    # dims[2] = min(8, 8*8=64) = 8
    assert layer_dims(d_local=4, L=3, chi_layer=8) == [4, 8, 8]


def test_layer_dims_for_d_local_8192_L_5_chi_16():
    # The A-encoder case.
    dims = layer_dims(d_local=8192, L=5, chi_layer=16)
    assert dims == [8192, 16, 16, 16, 16]


def test_layer_dims_below_chi_layer():
    # Small d_local, no cap pressure at the second layer.
    assert layer_dims(d_local=2, L=4, chi_layer=16) == [2, 4, 16, 16]


def test_causal_cone_path_at_leaf_0():
    # leaf=0, L=5 -> ascends through positions 0, 0, 0, 0, 0
    assert causal_cone_path(0, 5) == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]


def test_causal_cone_path_at_arbitrary_leaf():
    # leaf=11, L=4 -> 11, 5, 2, 1
    assert causal_cone_path(11, 4) == [(0, 11), (1, 5), (2, 2), (3, 1)]


from src.qft_pcn.qft.mera import MERATensor


def test_mera_tensor_construction():
    arr = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
    t = MERATensor(kind="disentangler", layer=1, position=3, array=arr)
    assert t.kind == "disentangler"
    assert t.layer == 1
    assert t.position == 3
    assert t.shape == (2, 2, 2, 2)


def test_mera_tensor_isometry_shape():
    w = np.zeros((4, 2, 2), dtype=complex)
    w[0, 0, 0] = 1.0
    t = MERATensor(kind="isometry", layer=2, position=0, array=w)
    assert t.shape == (4, 2, 2)


def test_mera_tensor_invalid_kind():
    with pytest.raises(ValueError, match="kind"):
        MERATensor(kind="bogus", layer=0, position=0,
                   array=np.zeros((1, 4, 1), dtype=complex))


from src.qft_pcn.qft.mera import MERA


def test_mera_n_l_d_local_properties():
    # Build a minimal valid MERA by hand for shape testing.
    leaves = [np.zeros((1, 4, 1), dtype=complex) for _ in range(4)]
    for s in leaves:
        s[0, 0, 0] = 1.0
    # L=2 layers since N=4=2^2
    dis_0 = [np.eye(16, dtype=complex).reshape(4, 4, 4, 4) for _ in range(2)]
    inter_0 = [np.eye(16, dtype=complex).reshape(4, 4, 4, 4)]
    iso_0 = [np.zeros((4, 4, 4), dtype=complex) for _ in range(2)]
    for w in iso_0:
        for k in range(4):
            w[k, k // 4, k % 4] = 1.0
    dis_1 = [np.eye(16, dtype=complex).reshape(4, 4, 4, 4)]
    inter_1 = []
    iso_1 = [np.zeros((4, 4, 4), dtype=complex)]
    for k in range(4):
        iso_1[0][k, k // 4, k % 4] = 1.0
    top = np.zeros((4, 4, 1), dtype=complex)
    top[0, 0, 0] = 1.0
    m = MERA(
        leaves=leaves,
        disentanglers=[dis_0, dis_1],
        inter_disentanglers=[inter_0, inter_1],
        isometries=[iso_0, iso_1],
        top=top,
        layer_dims=[4, 4],
    )
    assert m.N == 4
    assert m.L == 2
    assert m.d_local == 4


def test_mera_rejects_non_power_of_2_N():
    with pytest.raises(InvalidLayerCount):
        MERA(
            leaves=[np.zeros((1, 4, 1), dtype=complex) for _ in range(3)],
            disentanglers=[], inter_disentanglers=[], isometries=[],
            top=np.array([[[1.0]]], dtype=complex), layer_dims=[4],
        )


def test_vacuum_has_correct_layer_structure():
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    assert m.N == 8
    assert m.L == 3
    assert m.d_local == 4
    assert m.layer_dims == [4, 4, 4]
    assert len(m.disentanglers[0]) == 4
    assert len(m.isometries[0]) == 4
    assert len(m.disentanglers[1]) == 2
    assert len(m.isometries[1]) == 2
    assert len(m.disentanglers[2]) == 1
    assert len(m.isometries[2]) == 1
    # inter-pair counts: 3, 1, 0
    assert len(m.inter_disentanglers[0]) == 3
    assert len(m.inter_disentanglers[1]) == 1
    assert len(m.inter_disentanglers[2]) == 0


def test_vacuum_top_tensor_is_unit_norm():
    m = MERA.vacuum(N=8, d_local=4)
    assert abs(np.linalg.norm(m.top) - 1.0) < 1e-12


def test_vacuum_initial_isometries_are_isometric():
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    for ell in range(m.L):
        for j, w in enumerate(m.isometries[ell]):
            d_up = w.shape[0]
            mat = w.reshape(d_up, -1)
            prod = mat @ mat.conj().T
            assert np.allclose(prod, np.eye(d_up), atol=1e-10), \
                f"isometry ({ell}, {j}) not isometric"


def test_vacuum_initial_disentanglers_are_unitary():
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    for ell in range(m.L):
        for j, u in enumerate(m.disentanglers[ell]):
            d = u.shape[0]
            mat = u.reshape(d * d, d * d)
            assert np.allclose(mat @ mat.conj().T, np.eye(d * d),
                               atol=1e-10), f"disentangler ({ell}, {j}) not unitary"


def test_vacuum_invalid_N_raises():
    with pytest.raises(InvalidLayerCount):
        MERA.vacuum(N=6, d_local=4)


def test_from_product_matches_vacuum_when_all_zero():
    from src.qft_pcn.qft.fock import vacuum_vec
    states = [vacuum_vec(4) for _ in range(8)]
    m = MERA.from_product(states, chi_layer=4)
    v = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    # Same leaves, identical disentanglers/isometries.
    for k in range(8):
        assert np.allclose(m.leaves[k], v.leaves[k])


def test_number_states_constructs_correct_leaves():
    m = MERA.number_states([1, 0, 2, 0, 0, 0, 0, 0], d=4)
    # leaf 0 = |1>, leaf 2 = |2>.
    assert m.leaves[0][0, 1, 0] == 1.0
    assert m.leaves[2][0, 2, 0] == 1.0
    # Others = |0>.
    assert m.leaves[1][0, 0, 0] == 1.0


def test_vacuum_norm_sq_is_one():
    m = MERA.vacuum(N=8, d_local=4)
    assert abs(m.norm_sq() - 1.0) < 1e-10


def test_number_state_norm_sq_is_one():
    m = MERA.number_states([2, 0, 1, 3, 0, 0, 1, 0], d=4)
    assert abs(m.norm_sq() - 1.0) < 1e-10


def test_norm_sq_for_n_eq_2():
    # Minimal MERA: N=2, L=1.
    m = MERA.vacuum(N=2, d_local=4, chi_layer=4)
    assert abs(m.norm_sq() - 1.0) < 1e-10


def test_normalize_makes_norm_sq_one():
    m = MERA.number_states([2, 0, 1, 0, 0, 0, 0, 0], d=4)
    # Manually scale up so norm > 1.
    m.leaves[0] = 3.0 * m.leaves[0]
    assert m.norm_sq() > 1.0
    m.normalize()
    assert abs(m.norm_sq() - 1.0) < 1e-10


def test_inner_self_equals_norm_sq():
    m = MERA.vacuum(N=8, d_local=4)
    assert abs(m.inner(m) - m.norm_sq()) < 1e-10


def test_inner_orthogonal_number_states():
    a = MERA.number_states([1, 0, 0, 0, 0, 0, 0, 0], d=3)
    b = MERA.number_states([0, 1, 0, 0, 0, 0, 0, 0], d=3)
    assert abs(a.inner(b)) < 1e-10
    assert abs(b.inner(a)) < 1e-10


def test_inner_normalized_self_is_one():
    m = MERA.number_states([2, 1, 0, 1, 0, 0, 0, 0], d=3).normalize()
    assert abs(m.inner(m) - 1.0) < 1e-10


# ---- Task 8: apply_local_gate ---------------------------------------------


def test_apply_local_gate_modifies_only_target_leaf():
    m = MERA.vacuum(N=8, d_local=4)
    snap = [s.copy() for s in m.leaves]
    gate = np.eye(4, dtype=complex)
    gate[[0, 1]] = gate[[1, 0]]
    m.apply_local_gate(3, gate)
    for k in range(8):
        if k == 3:
            continue
        assert np.allclose(m.leaves[k], snap[k]), f"leaf {k} mutated"
    assert m.leaves[3][0, 1, 0] == 1.0
    assert m.leaves[3][0, 0, 0] == 0.0


def test_apply_local_gate_invalid_leaf():
    m = MERA.vacuum(N=8, d_local=4)
    with pytest.raises(IndexError):
        m.apply_local_gate(8, np.eye(4))


def test_apply_local_gate_invalid_shape():
    m = MERA.vacuum(N=8, d_local=4)
    with pytest.raises(ValueError):
        m.apply_local_gate(0, np.eye(3))


# ---- Task 9: _ascend_one_layer --------------------------------------------


def test_ascend_identity_stays_identity():
    """Identity ascended through any layer is identity at the next."""
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    for ell in range(m.L - 1):
        d_ell = m.layer_dims[ell]
        op = np.eye(d_ell, dtype=complex)
        op_up = m._ascend_one_layer(op, ell, pos=0)
        d_up = m.layer_dims[ell + 1]
        assert op_up.shape == (d_up, d_up)
        assert np.allclose(op_up, np.eye(d_up), atol=1e-10), \
            f"layer {ell}: identity did not ascend to identity"


def test_ascend_identity_at_odd_position():
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    d0 = m.layer_dims[0]
    op = np.eye(d0, dtype=complex)
    op_up = m._ascend_one_layer(op, ell=0, pos=1)
    d_up = m.layer_dims[1]
    assert np.allclose(op_up, np.eye(d_up), atol=1e-10)


# ---- Task 10: local_expectation -------------------------------------------


def test_local_expectation_number_op_on_number_state():
    from src.qft_pcn.qft.fock import number
    occ = [2, 0, 1, 3, 0, 0, 1, 0]
    d = 4
    m = MERA.number_states(occ, d=d)
    n_op = number(d)
    for leaf in range(8):
        e = m.local_expectation(leaf, n_op).real
        assert abs(e - occ[leaf]) < 1e-10, \
            f"leaf {leaf}: {e} vs {occ[leaf]}"


def test_local_expectation_identity_is_norm():
    m = MERA.number_states([1, 2, 0, 3, 0, 0, 1, 0], d=4)
    I = np.eye(4, dtype=complex)
    for leaf in range(8):
        e = m.local_expectation(leaf, I).real
        assert abs(e - 1.0) < 1e-10


# ---- Task 11: two_site_expectation intra-pair -----------------------------


def test_two_site_expectation_intra_pair_identity_is_one():
    m = MERA.vacuum(N=8, d_local=4)
    d = 4
    identity_op = np.eye(d * d, dtype=complex)
    # Leaf 0 is even, so (0, 1) is intra-pair.
    e = m.two_site_expectation(0, identity_op).real
    assert abs(e - 1.0) < 1e-10


def test_two_site_expectation_intra_pair_n0_otimes_n1_on_number_state():
    from src.qft_pcn.qft.fock import number
    d = 4
    m = MERA.number_states([2, 3, 0, 0, 0, 0, 0, 0], d=d)
    n = number(d)
    op = np.kron(n, n)
    e = m.two_site_expectation(0, op).real
    assert abs(e - 2.0 * 3.0) < 1e-10


# ---- Task 12: two_site_expectation inter-pair -----------------------------


def test_two_site_expectation_inter_pair_identity_is_one():
    m = MERA.vacuum(N=8, d_local=4)
    d = 4
    identity_op = np.eye(d * d, dtype=complex)
    e = m.two_site_expectation(1, identity_op).real
    assert abs(e - 1.0) < 1e-10


def test_two_site_expectation_inter_pair_number_op():
    from src.qft_pcn.qft.fock import number
    d = 4
    m = MERA.number_states([0, 2, 3, 0, 0, 0, 0, 0], d=d)
    n = number(d)
    op = np.kron(n, n)
    e = m.two_site_expectation(1, op).real
    assert abs(e - 2.0 * 3.0) < 1e-10


# ---- Task 13: bond_dimensions ---------------------------------------------


def test_bond_dimensions_match_layer_dims_for_vacuum():
    m = MERA.vacuum(N=16, d_local=4, chi_layer=4)
    bd = m.bond_dimensions()
    assert len(bd) == m.L
    # Each entry = max output dim of isometries at layer ℓ.
    # For a uniform chi_layer = d_local = 4 setup, the entries equal
    # layer_dims[ℓ+1] for ℓ < L-1, and layer_dims[-1] for ℓ = L-1.
    for ell in range(m.L - 1):
        assert bd[ell] == m.layer_dims[ell + 1]
    assert bd[-1] == m.layer_dims[-1]


def test_bond_dimensions_with_d_local_larger_than_chi_layer():
    m = MERA.vacuum(N=8, d_local=64, chi_layer=8)
    bd = m.bond_dimensions()
    # All isometry first-index sizes are capped at chi_layer = 8.
    for v in bd:
        assert v == 8


# ---- Task 14: entanglement_entropy product-state fast path ----------------


def test_entropy_zero_for_product_state():
    m = MERA.number_states([1, 0, 1, 0, 0, 1, 0, 1], d=4)
    for cut in range(7):
        S = m.entanglement_entropy(cut)
        assert abs(S) < 1e-9, f"cut {cut}: S={S}"


def test_entropy_zero_for_vacuum():
    m = MERA.vacuum(N=8, d_local=4)
    for cut in range(7):
        assert abs(m.entanglement_entropy(cut)) < 1e-9


def test_entropy_invalid_cut():
    m = MERA.vacuum(N=8, d_local=4)
    with pytest.raises(ValueError):
        m.entanglement_entropy(-1)
    with pytest.raises(ValueError):
        m.entanglement_entropy(7)


# ---- Task 15: layer_metric ------------------------------------------------


def test_layer_metric_default_is_identity():
    m = MERA.vacuum(N=8, d_local=4, chi_layer=4)
    for ell in range(m.L):
        g = m.layer_metric(ell)
        d_l = m.layer_dims[ell]
        assert g.shape == (d_l, d_l)
        assert np.allclose(g, np.eye(d_l))


def test_layer_metric_invalid_layer():
    m = MERA.vacuum(N=8, d_local=4)
    with pytest.raises(IndexError):
        m.layer_metric(10)


# ---- Task 16: apply_two_site_gate intra-pair ------------------------------


def test_apply_intra_pair_unitary_gate_structural():
    """Apply a unitary at leaves (0, 1) → structural verification.

    Note on norm preservation: the current MERA representation stores BOTH
    the leaf state vectors AND the top tensor; for a state built via
    from_product these encode the same wavefunction redundantly, and the
    network's <psi|psi> via inner(self, self) = leaf-product overlap with
    descent-from-top. apply_two_site_gate per spec §5.6/§6.4 (line 533)
    "defers" the upper-layer adaptation, so the top tensor becomes stale.
    Exact norm preservation under arbitrary unitary gates would require
    re-deriving the top (O(N) cost violating the causal-cone bound). The
    spec acknowledges this as encoder-time approximation. We verify the
    structural truth: SVD truncation reports zero loss for a unitary gate
    (the gate IS absorbed losslessly into the layer-0 disentangler).
    """
    m = MERA.vacuum(N=8, d_local=2, chi_layer=16)
    d = 2
    rng = np.random.default_rng(0)
    A = rng.standard_normal((d * d, d * d)) + 1j * rng.standard_normal((d * d, d * d))
    U, _ = np.linalg.qr(A)
    err = m.apply_two_site_gate(leaf=0, gate=U, chi_max=16)
    # The gate's SVD has 4 singular values all of magnitude 1 (it IS unitary).
    assert err < 1e-10
    # Structural: disentanglers[0][0] is now equal to U @ I = U (since the
    # vacuum's disentangler was the identity).
    u_new = m.disentanglers[0][0].reshape(d * d, d * d)
    assert np.allclose(u_new, U, atol=1e-10)


def test_apply_intra_pair_modifies_only_layer_0():
    m = MERA.vacuum(N=8, d_local=2, chi_layer=16)
    snap_dis = {(ell, j): m.disentanglers[ell][j].copy()
                for ell in range(1, m.L)
                for j in range(len(m.disentanglers[ell]))}
    snap_iso = {(ell, j): m.isometries[ell][j].copy()
                for ell in range(m.L)
                for j in range(len(m.isometries[ell]))}
    d = 2
    rng = np.random.default_rng(1)
    A = rng.standard_normal((d * d, d * d)) + 1j * rng.standard_normal((d * d, d * d))
    U, _ = np.linalg.qr(A)
    m.apply_two_site_gate(leaf=0, gate=U, chi_max=16)
    for k, arr in snap_dis.items():
        assert np.allclose(m.disentanglers[k[0]][k[1]], arr), \
            f"layer-{k[0]} disentangler {k[1]} mutated"
    for k, arr in snap_iso.items():
        assert np.allclose(m.isometries[k[0]][k[1]], arr), \
            f"layer-{k[0]} isometry {k[1]} mutated"


# ---- Task 17: apply_two_site_gate inter-pair + causal-cone test ----------


def test_apply_inter_pair_unitary_structural():
    """Apply a unitary at leaves (1, 2) — inter-pair boundary. Structural
    verification only (see note on test_apply_intra_pair_unitary_gate_structural
    for why exact norm preservation requires deferred top adaptation).
    """
    m = MERA.vacuum(N=8, d_local=2, chi_layer=16)
    d = 2
    rng = np.random.default_rng(2)
    A = rng.standard_normal((d * d, d * d)) + 1j * rng.standard_normal((d * d, d * d))
    U, _ = np.linalg.qr(A)
    err = m.apply_two_site_gate(leaf=1, gate=U, chi_max=16)
    assert err < 1e-10
    # Structural: inter_disentanglers[0][0] is now U.
    u_new = m.inter_disentanglers[0][0].reshape(d * d, d * d)
    assert np.allclose(u_new, U, atol=1e-10)


def test_apply_inter_pair_modifies_only_inter_disentangler_at_layer_0():
    m = MERA.vacuum(N=8, d_local=2, chi_layer=16)
    snap_intra = {(ell, j): m.disentanglers[ell][j].copy()
                  for ell in range(m.L)
                  for j in range(len(m.disentanglers[ell]))}
    snap_inter_higher = {(ell, j): m.inter_disentanglers[ell][j].copy()
                         for ell in range(1, m.L)
                         for j in range(len(m.inter_disentanglers[ell]))}
    d = 2
    rng = np.random.default_rng(3)
    A = rng.standard_normal((d * d, d * d)) + 1j * rng.standard_normal((d * d, d * d))
    U, _ = np.linalg.qr(A)
    m.apply_two_site_gate(leaf=1, gate=U, chi_max=16)
    for k, arr in snap_intra.items():
        assert np.allclose(m.disentanglers[k[0]][k[1]], arr), \
            f"intra disentangler ({k[0]}, {k[1]}) mutated"
    for k, arr in snap_inter_higher.items():
        assert np.allclose(m.inter_disentanglers[k[0]][k[1]], arr), \
            f"layer-{k[0]} inter disentangler mutated"


def test_local_expectation_causal_cone_O_log_N():
    """Spec §1.2: local_expectation touches only O(log N) tensors."""
    m = MERA.vacuum(N=64, d_local=4, chi_layer=4)
    calls = [0]
    orig = m._ascend_one_layer

    def counting_ascend(op, ell, pos):
        calls[0] += 1
        return orig(op, ell, pos)

    m._ascend_one_layer = counting_ascend
    from src.qft_pcn.qft.fock import number
    _ = m.local_expectation(leaf=17, op=number(4))
    assert calls[0] == m.L - 1, \
        f"local_expectation made {calls[0]} ascent calls; expected {m.L - 1}"


# ---- D5: inter-pair fold guard -------------------------------------------


def test_ascend_one_layer_handles_non_identity_inter_pair():
    """D5 fix: after :meth:`apply_two_site_gate` writes a non-identity
    inter-pair disentangler at layer 0, :meth:`local_expectation` and
    :meth:`two_site_expectation` must include the inter-pair causal cone
    in the result. Pre-fix, the single-pair ascending superoperator
    silently dropped this contribution and the expectation was biased.

    We verify the fold by:
    (a) constructing two MERAs with the same leaf state but different
        layer-0 inter-pair disentanglers (identity vs SWAP);
    (b) asserting that a non-symmetric local operator at the affected
        leaves observes a value-DIFFERENCE — i.e. the inter-pair
        disentangler is no longer dropped from the cone.
    Pre-fix, both expectations would equal the identity-disentangler
    value (silent drop). Post-fix, the SWAP'd MERA reads the swapped
    leaf state.
    """
    from src.qft_pcn.qft.mera import MERA
    # Two distinct leaf states so swap is observable.
    psi_a = np.array([1.0, 0.0], dtype=complex)         # |0>
    psi_b = np.array([0.0, 1.0], dtype=complex)         # |1>
    # 4-leaf product |0> |1> |0> |1>; leaves (1, 2) are inter-pair.
    m_id = MERA.from_product([psi_a, psi_b, psi_a, psi_b], chi_layer=4)
    m_swap = MERA.from_product([psi_a, psi_b, psi_a, psi_b], chi_layer=4)
    # SWAP gate at leaves (1, 2) — inter-pair, odd leaf.
    SWAP = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
    err = m_swap.apply_two_site_gate(leaf=1, gate=SWAP, chi_max=4)
    assert err < 1e-10, "SWAP is unitary; truncation should be ~0"
    # The inter-pair disentangler at layer 0, slot 0 (couples leaves
    # 1 and 2) is now non-identity.
    assert m_swap._layer0_any_nontrivial(), \
        "test precondition: SWAP should have made layer-0 disentanglers " \
        "non-identity"
    # n = diag(0, 1) projector on |1>. After SWAP on (1, 2):
    #   |0> |1> |0> |1>  -->  |0> |0> |1> |1>
    # Pre-swap <n_2> = 0; post-swap <n_2> = 1.
    n = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    v_id = m_id.local_expectation(leaf=2, op=n)
    v_swap = m_swap.local_expectation(leaf=2, op=n)
    assert abs(v_id.real) < 1e-8, f"identity-MERA <n_2> = {v_id}, expected 0"
    assert abs(v_swap.real - 1.0) < 1e-8, (
        f"SWAP'd MERA <n_2> = {v_swap}, expected 1 — D5 fold-fallback "
        "failed; inter-pair disentangler was dropped from the cone"
    )
    # Two-site expectation across the inter-pair boundary should also
    # be honest. n (x) n projector reads |1,1>. Pre-swap on (1, 2):
    # state is |1, 0>, <n(x)n> = 0. Post-swap: |0, 1>, also 0. Use
    # (n (x) I) instead: pre-swap <n_1> = 1, post-swap <n_1> = 0.
    n_op_left = np.kron(n, np.eye(2, dtype=complex))    # (4, 4)
    w_id = m_id.two_site_expectation(leaf=1, op=n_op_left)
    w_swap = m_swap.two_site_expectation(leaf=1, op=n_op_left)
    assert abs(w_id.real - 1.0) < 1e-8, \
        f"identity-MERA <n_1> via two-site = {w_id}, expected 1"
    assert abs(w_swap.real) < 1e-8, (
        f"SWAP'd MERA <n_1> via two-site = {w_swap}, expected 0 — "
        "D5 fold-fallback failed on intra-pair branch"
    )


# ---- D25: intra-pair fold guard (sister bug to D5) -----------------------


def test_two_site_expectation_handles_non_identity_intra():
    """D25 fix: ``two_site_expectation``'s odd-leaf branch silently dropped
    the layer-0 INTRA-pair disentanglers ``disentanglers[0][j_inter]`` and
    ``disentanglers[0][j_inter+1]`` from its 4-site fold (sister bug to
    D5, which only covered inter-pair). After this fix, when ANY layer-0
    disentangler is non-identity the call routes through
    :meth:`_two_site_expectation_via_materialize`, which folds intra +
    inter into a leaf-basis state and reads the operator honestly.

    Construction:
    (a) leaves |0,1,0,1>, 4-leaf MERA;
    (b) apply CNOT_{0,1} (an EVEN-leaf gate -> writes a non-identity
        INTRA disentangler at layer-0 slot 0; INTER stays identity);
    (c) read an ODD-leaf two-site observable at leaves (1, 2). Pre-fix,
        the odd-leaf branch ran the 4-site fold that ignored
        ``disentanglers[0][0]`` and would return the unperturbed value;
        post-fix the materialize fold sees the CNOT and returns the
        true (perturbed) expectation.

    Numeric check: with the initial product state |0,1,0,1>, applying
    CNOT on (0, 1) flips leaf 1 conditionally on leaf 0. Leaf 0 = |0>
    so CNOT acts as identity on this state — the state is unchanged.
    To make the bug observable we instead start in a state where leaf 0
    is in |+>: leaves |+,0,0,1>. CNOT_{0,1} on |+,0> produces the Bell
    pair (|00>+|11>)/sqrt2 on (0, 1). Then ``two_site_expectation`` at
    leaves (1, 2) of (n (x) I) — i.e. <n_1> — should be 1/2 (because
    leaf 1 is now in the marginal mixed state diag(1/2, 1/2)). Pre-fix
    the odd-leaf branch would have returned 0 (using the unperturbed
    leaf-1 = |0>, hence <n_1> = 0).
    """
    from src.qft_pcn.qft.mera import MERA
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    psi_1 = np.array([0.0, 1.0], dtype=complex)
    psi_plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    # 4-leaf product |+, 0, 0, 1>; pair 0 = (leaves 0, 1) is intra.
    m_cnot = MERA.from_product([psi_plus, psi_0, psi_0, psi_1],
                               chi_layer=4)
    # CNOT on (leaf 0, leaf 1) — INTRA-pair, even leaf. Writes a
    # non-identity disentanglers[0][0]; inter_disentanglers[0][*] stay
    # identity.
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)
    err = m_cnot.apply_two_site_gate(leaf=0, gate=CNOT, chi_max=4)
    assert err < 1e-10, "CNOT is unitary; truncation should be ~0"
    # Precondition: layer-0 INTRA disentangler is non-identity, but
    # INTER is still identity — exactly the regime D5 missed.
    any_intra_nontrivial = any(
        not np.allclose(u, np.eye(u.shape[0]))
        for u in m_cnot.disentanglers[0]
    )
    all_inter_identity = all(
        np.allclose(u, np.eye(u.shape[0]))
        for u in m_cnot.inter_disentanglers[0]
    )
    assert any_intra_nontrivial, (
        "test precondition: CNOT should write a non-identity intra "
        "disentangler at layer 0"
    )
    assert all_inter_identity, (
        "test precondition: even-leaf CNOT should NOT touch inter "
        "disentanglers — this is the D25 regime"
    )
    assert m_cnot._layer0_any_nontrivial(), \
        "guard must trigger on the intra-only nontrivial regime"
    # n = diag(0, 1) acting on leaf 1 via two-site op (n (x) I) at
    # leaves (1, 2) — this is the ODD-leaf branch that previously
    # dropped disentanglers[0][0] from its 4-site fold.
    n = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    n_op_left = np.kron(n, np.eye(2, dtype=complex))    # (4, 4)
    val = m_cnot.two_site_expectation(leaf=1, op=n_op_left)
    # Expected: after CNOT |+, 0> = (|00>+|11>)/sqrt2, leaf-1 marginal
    # is diag(1/2, 1/2) so <n_1> = 1/2. Pre-fix would return 0
    # (silent intra drop -> unperturbed leaf 1 = |0>).
    assert abs(val.real - 0.5) < 1e-8, (
        f"D25 fix failed: two_site_expectation(leaf=1, n(x)I) = {val}, "
        "expected 0.5; layer-0 INTRA disentangler still being dropped "
        "from the odd-leaf 4-site fold"
    )


# ---- Task 18: General entanglement entropy via materialization -----------


def test_entropy_of_bell_pair_at_leaves_0_and_1():
    """Bell pair on (0, 1) via Hadamard + CNOT; cut after leaf 0 → ln 2."""
    m = MERA.from_product(
        [np.array([1.0, 0.0]) for _ in range(8)],
        chi_layer=4,
    )
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    m.apply_local_gate(0, H)
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]], dtype=complex)
    m.apply_two_site_gate(0, CNOT, chi_max=4)
    S = m.entanglement_entropy(0)
    assert abs(S - np.log(2)) < 1e-6, f"S={S}, expected ln 2"


# ---- _entropy_from_terms regression: branch-superposition cut entropy -----


def _basis_vec(i: int, d: int = 2) -> np.ndarray:
    v = np.zeros(d, dtype=complex)
    v[i] = 1.0
    return v


def test_term_superposition_one_leaf_difference_is_product():
    """A 2-term superposition differing on exactly ONE leaf is a genuine
    product state: |0000> + |1000> = (|0>+|1>) (x) |0> (x) |0> (x) |0>.

    Every cut must have S = 0. Before the _entropy_from_terms fix the
    buggy K = cc.T*GR.T*GL form wrongly reported ln(2) at every cut.
    """
    z, o = _basis_vec(0), _basis_vec(1)
    t1 = (1.0, [z, z, z, z])
    t2 = (1.0, [o, z, z, z])
    m = MERA.from_term_superposition([t1, t2]).normalize()
    for cut in range(m.N - 1):
        S = m.entanglement_entropy(cut)
        assert abs(S) < 1e-9, f"cut {cut}: S={S}, expected 0 (product state)"


def test_term_superposition_two_leaf_difference_is_entangled():
    """A 2-term superposition differing on TWO leaves on the same side of
    a cut is genuinely entangled across that cut.

    |0000> + |1100>: leaves 0,1 differ. The cut after leaf 1 separates
    {0,1} from {2,3}; both branches agree on the right, so S(cut=1) = 0,
    but the cut after leaf 0 splits the differing leaves and yields ln 2.
    """
    z, o = _basis_vec(0), _basis_vec(1)
    t1 = (1.0, [z, z, z, z])
    t2 = (1.0, [o, o, z, z])
    m = MERA.from_term_superposition([t1, t2]).normalize()
    S0 = m.entanglement_entropy(0)
    assert abs(S0 - np.log(2)) < 1e-9, f"cut 0: S={S0}, expected ln 2"
    # Cuts 1 and 2: both branches' product on each side is consistent
    # (the two differing leaves sit together on the left), so S = 0.
    for cut in (1, 2):
        S = m.entanglement_entropy(cut)
        assert abs(S) < 1e-9, f"cut {cut}: S={S}, expected 0"


def test_term_superposition_is_not_reported_as_product():
    """_is_product() must not claim a 2-term entangled superposition is a
    product state (its disentanglers are identity; entanglement is
    isometry-carried)."""
    z, o = _basis_vec(0), _basis_vec(1)
    m = MERA.from_term_superposition(
        [(1.0, [z, z, z, z]), (1.0, [o, o, z, z])])
    assert not m._is_product()
    # A single-term superposition IS a product state.
    m1 = MERA.from_term_superposition([(1.0, [z, z, z, z])])
    assert m1._is_product()


# ---- Task 19: mera_evolution TEBD + energy --------------------------------


def test_mera_trotter_step_runs_without_error():
    """Sanity: trotter_step applies its layer-0 gates and reports finite err."""
    from src.qft_pcn.qft.hamiltonian import (
        FieldSpecies, HamiltonianConfig, Hamiltonian)
    from src.qft_pcn.qft.mera_evolution import trotter_step as mera_trotter
    species = [FieldSpecies(name="a", cutoff=4, bare_mass=1.0, kinetic=0.5)]
    cfg = HamiltonianConfig(species=species)
    H = Hamiltonian(cfg, N=8)
    # Use a chi_layer that makes upper-layer isos non-truncating in the
    # ascent path (dims = [4, 16, 16] with d=4).
    m = MERA.vacuum(N=8, d_local=4, chi_layer=16)
    err = mera_trotter(m, H, dt=0.01, imaginary=False, chi_max=16)
    # Truncation error is reported as a finite, non-negative float.
    assert err >= 0.0
    assert np.isfinite(err)


def test_mera_energy_matches_local_plus_bond_sum_for_vacuum():
    """Sanity: energy of the vacuum equals the sum of <local_op> + <bond_op>."""
    from src.qft_pcn.qft.hamiltonian import (
        FieldSpecies, HamiltonianConfig, Hamiltonian)
    from src.qft_pcn.qft.mera_evolution import energy as mera_energy
    species = [FieldSpecies(name="a", cutoff=4, bare_mass=1.0, kinetic=0.5)]
    cfg = HamiltonianConfig(species=species)
    H = Hamiltonian(cfg, N=8)
    m = MERA.vacuum(N=8, d_local=4, chi_layer=16)
    e = mera_energy(m, H)
    # Manual sum to verify.
    e_check = 0.0 + 0.0j
    for k in range(H.N):
        e_check += m.local_expectation(k, H.local_op(k))
    for k in range(H.N - 1):
        e_check += m.two_site_expectation(k, H.bond_op(k))
    assert abs(e - float(np.real(e_check))) < 1e-10


# ---- Task 20: public re-exports ------------------------------------------


def test_public_mera_imports():
    from src.qft_pcn.qft import MERA as MERA_pub, MERATensor as MERATensor_pub
    from src.qft_pcn.qft import mera_trotter_step, mera_evolve, mera_energy
    # Smoke: identifiers exist and constructors work.
    m = MERA_pub.vacuum(N=4, d_local=2, chi_layer=4)
    assert abs(m.norm_sq() - 1.0) < 1e-10
    assert callable(mera_trotter_step)
    assert callable(mera_evolve)
    assert callable(mera_energy)


def test_local_expectation_matches_mps_for_product():
    from src.qft_pcn.qft.mps import MPS
    from src.qft_pcn.qft.fock import number
    occ = [2, 0, 1, 3, 0, 0, 1, 0]
    d = 4
    mera_state = MERA.number_states(occ, d=d)
    mps_state = MPS.number_states(occ, d=d)
    n_op = number(d)
    for leaf in range(8):
        e_mera = mera_state.local_expectation(leaf, n_op).real
        e_mps = mps_state.local_expectation(leaf, n_op).real
        assert abs(e_mera - e_mps) < 1e-10, \
            f"leaf {leaf}: mera {e_mera} vs mps {e_mps}"
