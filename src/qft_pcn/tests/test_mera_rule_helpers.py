"""Tests for the shared MERA Hamiltonian rule helpers (spec §5.2)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic._mera_typing_rules import (
    leaf_proj, leaf_proj_one_minus, leaf_proj_set, node_leaf,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


def test_leaf_proj_is_16x16_one_hot():
    p = leaf_proj(5)
    assert p.shape == (16, 16)
    assert p[5, 5] == 1.0
    assert np.count_nonzero(p) == 1


def test_leaf_proj_one_minus_is_complement():
    p = leaf_proj(3)
    q = leaf_proj_one_minus(3)
    assert np.allclose(p + q, np.eye(16))


def test_leaf_proj_set_sums_basis_projectors():
    p = leaf_proj_set([1, 2, 4])
    assert p[1, 1] == 1.0 and p[2, 2] == 1.0 and p[4, 4] == 1.0
    assert np.count_nonzero(p) == 3


def test_node_leaf_addresses_via_layout():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    # node 0, species "kind" -> leaf 0; node 1, "bid" -> leaf 7.
    assert node_leaf(meta, 0, "kind") == 0
    assert node_leaf(meta, 1, "bid") == 7
    assert node_leaf(meta, 1, "type") == 6
