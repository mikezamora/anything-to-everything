"""Tests for the STLC per-node typing rules (spec §5.3, B §3.1-§3.3)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm,
)
from src.qft_pcn.logic._mera_typing_rules import mutate_leaf
from src.qft_pcn.logic.mera_encoding import TYPE_INT, TYPE_BOOL, KIND_BIN


def _term_energy(src, rule, node):
    state, meta = encode_mera(parse(src))
    H = MeraTypingHamiltonian(meta)
    return H.term_energy(state, MeraTypingTerm(rule, node, 1)), state, meta


def test_lit_int_zero_on_well_typed():
    e, _, _ = _term_energy(r"\x:Int. 3", "T-Lit-Int", 1)
    assert abs(e) < 1e-9


def test_lit_int_fires_on_mistyped_intlit():
    state, meta = encode_mera(parse(r"\x:Int. 3"))
    type_leaf = meta.layout.leaf_of(1, "type")
    state = mutate_leaf(state, type_leaf, TYPE_INT, TYPE_BOOL)
    H = MeraTypingHamiltonian(meta)
    e = H.term_energy(state, MeraTypingTerm("T-Lit-Int", 1, 1))
    assert e > 0.99


def test_lit_bool_zero_on_well_typed():
    e, _, _ = _term_energy(r"\x:Int. true", "T-Lit-Bool", 1)
    assert abs(e) < 1e-9


def test_lit_bool_fires_on_mistyped_boollit():
    state, meta = encode_mera(parse(r"\x:Int. true"))
    type_leaf = meta.layout.leaf_of(1, "type")
    state = mutate_leaf(state, type_leaf, TYPE_BOOL, TYPE_INT)
    H = MeraTypingHamiltonian(meta)
    e = H.term_energy(state, MeraTypingTerm("T-Lit-Bool", 1, 1))
    assert e > 0.99


def test_bin_arith_zero_on_well_typed():
    state, meta = encode_mera(parse(r"\x:Int. 1 + 2"))
    H = MeraTypingHamiltonian(meta)
    total = sum(H.term_energy(state, MeraTypingTerm("T-Bin-Arith", n, 1))
                for n in range(meta.n_nodes))
    assert abs(total) < 1e-9


def test_bin_arith_fires_on_mistyped_bin():
    state, meta = encode_mera(parse(r"\x:Int. 1 + 2"))
    bin_node = None
    for n in range(meta.n_nodes):
        kind_leaf = meta.layout.leaf_of(n, "kind")
        proj = np.zeros((16, 16), dtype=complex)
        proj[KIND_BIN, KIND_BIN] = 1.0
        if np.real(state.local_expectation(kind_leaf, proj)) > 0.5:
            bin_node = n
            break
    assert bin_node is not None
    type_leaf = meta.layout.leaf_of(bin_node, "type")
    state = mutate_leaf(state, type_leaf, TYPE_INT, TYPE_BOOL)
    H = MeraTypingHamiltonian(meta)
    e = H.term_energy(state, MeraTypingTerm("T-Bin-Arith", bin_node, 1))
    assert e > 0.99
