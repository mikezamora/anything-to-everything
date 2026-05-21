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
