"""Tests for the MeraTypingHamiltonian skeleton (spec §5.1, §5.3)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm, MeraTermNotFound,
)


def test_constructs_from_meta_only():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    assert isinstance(H.terms, list)
    assert len(H.terms) > 0


def test_terms_are_mera_typing_terms():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    for t in H.terms:
        assert isinstance(t, MeraTypingTerm)
        assert isinstance(t.rule_id, str)
        assert 0 <= t.node < meta.n_nodes
        assert t.arity in (1, 2)


def test_all_sixteen_rule_names_present():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    names = {t.rule_id for t in H.terms}
    expected = {
        "T-Lit-Int", "T-Lit-Bool", "T-Bin-Arith", "T-Bin-Cmp",
        "T-Var", "T-Abs", "T-App-Arrow", "T-Obligation",
        "T-Zero", "T-Succ", "T-NatLit", "T-Nil", "T-Cons",
        "T-Eq", "T-Forall", "T-Fix",
    }
    assert expected <= names


def test_term_energy_unknown_term_raises():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    bogus = MeraTypingTerm(rule_id="T-Nonsense", node=0, arity=1)
    with pytest.raises(MeraTermNotFound):
        state, _ = encode_mera(parse(r"\x:Int. x"))
        H.term_energy(state, bogus)


def test_total_energy_returns_float():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    e = H.total_energy(state)
    assert isinstance(e, float)


def test_residuals_keyed_by_rule_and_node():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    r = H.residuals(state)
    for key, val in r.items():
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(key[0], str) and isinstance(key[1], int)
        assert val >= -1e-12


def test_structural_no_ast_attributes():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    for attr in dir(H):
        if attr.startswith("_"):
            continue
        low = attr.lower()
        for bad in ("ast", "tree", "node_obj", "walk"):
            assert bad not in low, f"H.{attr} suggests AST dependency"
