"""Tests for mera_window_expectation (spec §8.2, §9.8)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic._mera_window import mera_window_expectation
from src.qft_pcn.logic.mera_encoding import MERA_LEAF_DIM


def test_window_k1_matches_local_expectation():
    """k=1 window equals MERA.local_expectation."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    # Number operator on leaf 0.
    op = np.diag(np.arange(d)).astype(complex)
    via_window = mera_window_expectation(state, leaf0=0, k=1, op=op)
    via_local = state.local_expectation(0, op)
    assert np.isclose(via_window, via_local, atol=1e-10)


def test_window_k2_matches_two_site_expectation():
    """k=2 window equals MERA.two_site_expectation (spec §9.8)."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    rng = np.random.default_rng(0)
    # A random Hermitian two-leaf operator.
    m = rng.standard_normal((d * d, d * d)) + 1j * rng.standard_normal((d * d, d * d))
    op = m + m.conj().T
    via_window = mera_window_expectation(state, leaf0=0, k=2, op=op)
    via_two_site = state.two_site_expectation(0, op)
    assert np.isclose(via_window, via_two_site, atol=1e-10)


from src.qft_pcn.logic._mera_window import mera_window_expectation_factored


def test_factored_single_leaf_matches_local():
    """A factored op on one leaf equals local_expectation."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    op = np.diag(np.arange(d)).astype(complex)
    via_factored = mera_window_expectation_factored(state, {0: op})
    via_local = state.local_expectation(0, op)
    assert np.isclose(via_factored, via_local, atol=1e-10)


def test_factored_two_separable_leaves_matches_two_site():
    """A factored op O_a (x) O_b on leaves 0,1 equals two_site_expectation
    of the kron product."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    rng = np.random.default_rng(1)
    a = rng.standard_normal((d, d)); a = a + a.T
    b = rng.standard_normal((d, d)); b = b + b.T
    op_a = a.astype(complex)
    op_b = b.astype(complex)
    via_factored = mera_window_expectation_factored(state, {0: op_a, 1: op_b})
    via_two_site = state.two_site_expectation(0, np.kron(op_a, op_b))
    assert np.isclose(via_factored, via_two_site, atol=1e-10)


def test_factored_identity_leaves_skipped():
    """Listing an identity op for a leaf equals not listing it."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    op = np.diag(np.arange(d)).astype(complex)
    I = np.eye(d, dtype=complex)
    with_id = mera_window_expectation_factored(state, {0: op, 1: I})
    without = mera_window_expectation_factored(state, {0: op})
    assert np.isclose(with_id, without, atol=1e-10)
