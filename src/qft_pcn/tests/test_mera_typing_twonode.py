"""Tests for the STLC two-node typing rules (spec §5.4, §5.5)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm,
)
from src.qft_pcn.logic._mera_typing_rules import mutate_leaf
from src.qft_pcn.logic.mera_encoding import TYPE_INT, TYPE_BOOL


def _all_rule_energy(state, meta, rule):
    H = MeraTypingHamiltonian(meta)
    return sum(H.term_energy(state, MeraTypingTerm(rule, n, 2))
               for n in range(meta.n_nodes))


def test_t_var_zero_on_well_typed():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    assert abs(_all_rule_energy(state, meta, "T-Var")) < 1e-9


def test_t_var_fires_when_var_type_mismatches_binder():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    var_type_leaf = meta.layout.leaf_of(1, "type")
    state = mutate_leaf(state, var_type_leaf, TYPE_INT, TYPE_BOOL)
    assert _all_rule_energy(state, meta, "T-Var") > 0.99


def test_t_abs_zero_on_well_typed():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    assert abs(_all_rule_energy(state, meta, "T-Abs")) < 1e-9


def test_t_app_arrow_zero_on_well_typed():
    state, meta = encode_mera(parse(r"(\x:Int. x)(1)"))
    assert abs(_all_rule_energy(state, meta, "T-App-Arrow")) < 1e-9


def test_t_obligation_zero_on_well_typed():
    state, meta = encode_mera(parse(r"\x:Int. x + 1"))
    assert abs(_all_rule_energy(state, meta, "T-Obligation")) < 1e-9


def test_t_obligation_fires_on_bad_operand():
    state, meta = encode_mera(parse(r"\x:Int. x + true"))
    assert _all_rule_energy(state, meta, "T-Obligation") > 0.5
