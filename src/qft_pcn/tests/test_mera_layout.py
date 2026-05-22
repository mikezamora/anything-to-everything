"""Tests for the node-major species-leaf layout (spec §4.3)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._mera_layout import compute_layout, MeraLayout


def _layout(src, n_max=32):
    sites = serialize_preorder(parse(src), N=n_max)
    n_nodes = sum(1 for s in sites if s.kind != 0)  # non-PAD count
    return compute_layout(n_nodes)


def test_layout_leaf_count_is_5n_padded_to_power_of_2():
    lay = compute_layout(n_nodes=2)        # 2 nodes -> 10 leaves -> pad to 16
    assert lay.n_leaves == 16
    assert lay.L == 4                       # log2(16)


def test_layout_pads_up():
    lay = compute_layout(n_nodes=7)        # 35 leaves -> pad to 64
    assert lay.n_leaves == 64
    assert lay.L == 6


def test_species_of_leaf_is_node_major():
    lay = compute_layout(n_nodes=2)
    # node 0: leaves 0-4, node 1: leaves 5-9, rest PAD
    assert lay.species_of_leaf[:10] == [
        "kind", "type", "bid", "value", "tobl",
        "kind", "type", "bid", "value", "tobl",
    ]
    assert all(s == "PAD" for s in lay.species_of_leaf[10:])


def test_node_of_leaf():
    lay = compute_layout(n_nodes=2)
    assert lay.node_of_leaf[:10] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert all(n == -1 for n in lay.node_of_leaf[10:])


def test_species_leaf_helpers():
    lay = compute_layout(n_nodes=3)
    # the bid leaf of node 2 is 5*2 + 2 = 12
    assert lay.leaf_of(node=2, species="bid") == 12
    assert lay.leaf_of(node=0, species="kind") == 0


def test_one_node_pads_to_8():
    lay = compute_layout(n_nodes=1)        # 5 leaves -> pad to 8
    assert lay.n_leaves == 8
    assert lay.L == 3
