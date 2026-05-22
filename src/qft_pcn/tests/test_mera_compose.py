"""Tests for compose_mera_hamiltonians (spec §9.5)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_compose import (
    compose_mera_hamiltonians, IncompatibleHamiltonians,
)


def test_total_energy_is_sum():
    state, meta = encode_mera(parse(r"2 + 3"))
    H_t = MeraTypingHamiltonian(meta)
    H_e = MeraEvalHamiltonian(meta)
    H = compose_mera_hamiltonians(H_t, H_e)
    assert abs(H.total_energy(state)
               - (H_t.total_energy(state) + H_e.total_energy(state))) < 1e-9


def test_terms_concatenated():
    _, meta = encode_mera(parse(r"2 + 3"))
    H_t = MeraTypingHamiltonian(meta)
    H_e = MeraEvalHamiltonian(meta)
    H = compose_mera_hamiltonians(H_t, H_e)
    assert len(H.terms) == len(H_t.terms) + len(H_e.terms)


def test_incompatible_metas_raise():
    _, m1 = encode_mera(parse(r"\x:Int. x"))
    _, m2 = encode_mera(parse(r"(\x:Int. x + 1)(2)"))
    H1 = MeraTypingHamiltonian(m1)
    H2 = MeraEvalHamiltonian(m2)
    with pytest.raises(IncompatibleHamiltonians):
        compose_mera_hamiltonians(H1, H2)
