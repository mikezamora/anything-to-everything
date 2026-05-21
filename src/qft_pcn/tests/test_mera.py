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
