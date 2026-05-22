"""Tests for per-leaf basis-vector construction (spec §6.2 step 4)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._types import compute_site_types
from src.qft_pcn.logic._mera_leaves import node_leaf_vectors
from src.qft_pcn.logic.mera_encoding import (
    MERA_LEAF_DIM, KIND_VAR, KIND_LAM,
)
from src.qft_pcn.logic.encoding import KIND_PAD


def _occ_and_types(src):
    ast = parse(src)
    sites = serialize_preorder(ast, N=8)
    types = compute_site_types(ast, sites)
    return sites, types


def test_returns_five_vectors_each_16dim():
    sites, types = _occ_and_types(r"\x:Int. x")
    vecs = node_leaf_vectors(sites[0], types[0])
    assert len(vecs) == 5
    for v in vecs:
        assert v.shape == (MERA_LEAF_DIM,)
        assert np.isclose(np.linalg.norm(v), 1.0)


def test_kind_leaf_is_one_hot_at_kind_index():
    sites, types = _occ_and_types(r"\x:Int. x")
    # site 0 is the Lam.
    vecs = node_leaf_vectors(sites[0], types[0])
    kind_vec = vecs[0]
    assert np.argmax(np.abs(kind_vec)) == KIND_LAM


def test_pad_node_is_all_pad_basis():
    sites, types = _occ_and_types(r"\x:Int. x")
    # site 2+ are PAD (only 2 AST nodes).
    vecs = node_leaf_vectors(sites[2], types[2])
    assert np.argmax(np.abs(vecs[0])) == KIND_PAD


def test_var_leaf_records_bid():
    sites, types = _occ_and_types(r"\x:Int. x")
    # site 1 is Var(x).
    vecs = node_leaf_vectors(sites[1], types[1])
    assert np.argmax(np.abs(vecs[0])) == KIND_VAR
    # bid leaf (index 2) is one-hot at the var's bid value (non-PAD).
    bid_vec = vecs[2]
    assert np.isclose(np.linalg.norm(bid_vec), 1.0)
