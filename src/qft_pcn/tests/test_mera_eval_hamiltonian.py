"""Tests for MeraEvalHamiltonian redex penalties (spec §7.1, §9.3)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian, MeraEvalTerm, MeraEvalTermNotFound,
)


def test_constructs_from_meta():
    _, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    assert len(H.terms) > 0


def test_rule_names_present():
    _, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    names = {t.rule_id for t in H.terms}
    assert {"R-Beta", "R-Arith", "R-Cmp", "R-If"} <= names


def test_normal_form_zero_energy():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraEvalHamiltonian(meta)
    assert abs(H.total_energy(state)) < 1e-9


def test_unreduced_arith_positive_energy():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    assert H.total_energy(state) > 0.5


def test_unreduced_beta_positive_energy():
    state, meta = encode_mera(parse(r"(\x:Int. x)(1)"))
    H = MeraEvalHamiltonian(meta)
    res = H.residuals(state)
    assert any(k[0] == "R-Beta" and v > 0.5 for k, v in res.items())


def test_unknown_term_raises():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    with pytest.raises(MeraEvalTermNotFound):
        H.term_energy(state, MeraEvalTerm("R-Nonsense", 0, 2))


def test_residuals_keyed_by_rule_node():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    for key, val in H.residuals(state).items():
        assert isinstance(key[0], str) and isinstance(key[1], int)
        assert val >= -1e-12
