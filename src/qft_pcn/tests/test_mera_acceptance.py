"""MERA-native encoder acceptance suite (spec §9.2, §9.3, §9.5)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


def test_layout_is_node_major_5n_padded():
    """spec §9.2."""
    state, meta = encode_mera(parse(r"\x:Int. x"))   # 2 nodes
    assert meta.n_nodes == 2
    assert meta.n_leaves == 16
    assert meta.species_of_leaf[:10] == [
        "kind", "type", "bid", "value", "tobl",
        "kind", "type", "bid", "value", "tobl",
    ]
    assert all(s == "PAD" for s in meta.species_of_leaf[10:])


def test_alpha_renaming_identical_state():
    """spec §9.3: \\x.x and \\y.y encode to the same MERA state."""
    s1, _ = encode_mera(parse(r"\x:Int. x"))
    s2, _ = encode_mera(parse(r"\y:Int. y"))
    overlap = abs(s1.inner(s2)) ** 2
    assert overlap > 1.0 - 1e-10, f"overlap {overlap}"


def test_pad_leaves_are_vacuum():
    """spec §9.5: PAD leaves have zero amplitude on non-PAD basis states."""
    from src.qft_pcn.logic.mera_encoding import MERA_LEAF_DIM, KIND_PAD
    state, meta = encode_mera(parse(r"\x:Int. x"))
    proj = np.eye(MERA_LEAF_DIM, dtype=complex)
    proj[KIND_PAD, KIND_PAD] = 0.0       # project away from PAD
    for leaf in range(meta.n_leaves):
        if meta.species_of_leaf[leaf] == "PAD":
            val = state.local_expectation(leaf, proj)
            assert abs(val) < 1e-10, f"PAD leaf {leaf} not vacuum: {val}"
