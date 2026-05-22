"""Tests for encode_mera on concrete (hole-free) programs (spec §6)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera, MeraEncodingMeta
from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.logic.encoding import (
    EncodingTooLarge, IllScopedVar, IntLiteralOutOfRange,
)


def test_encode_returns_mera_and_meta():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    assert isinstance(state, MERA)
    assert isinstance(meta, MeraEncodingMeta)


def test_encoded_state_is_unit_norm():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10)


def test_meta_leaf_count_is_5n_padded():
    state, meta = encode_mera(parse(r"\x:Int. x"))   # 2 nodes
    assert meta.n_nodes == 2
    assert meta.n_leaves == 16        # 10 -> pad 16
    assert state.N == 16


def test_meta_records_binder_and_use_leaves():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    # The Lam (node 0) is a binder; Var (node 1) uses it.
    assert 0 in meta.binder_leaves
    # use_to_binder maps the Var's bid leaf to the Lam's bid leaf.
    assert len(meta.use_to_binder) >= 1


def test_p5_unit_norm():
    state, meta = encode_mera(parse(r"(\x:Int. (\y:Int. x + y)(3))(4)"))
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10)


def test_encoding_too_large_raises():
    with pytest.raises(EncodingTooLarge):
        encode_mera(parse(r"\x:Int. x + x + x"), n_nodes_max=3)


def test_ill_scoped_var_raises():
    with pytest.raises(IllScopedVar):
        encode_mera(parse("undefined_name"))


def test_int_literal_out_of_range_raises():
    with pytest.raises(IntLiteralOutOfRange):
        encode_mera(parse(r"\x:Int. x + 99"))
