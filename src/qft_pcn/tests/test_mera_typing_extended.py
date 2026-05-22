"""Tests for the extended-calculus typing rules (spec §6)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import (
    parse, Succ, Zero, Cons, NatLit, Nil, Lam, Eq, Var, Forall, Fix, TNat,
    BoolLit,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm,
)


def _typing_energy(ast):
    state, meta = encode_mera(ast)
    return MeraTypingHamiltonian(meta).total_energy(state)


def test_succ_succ_zero_well_typed():
    assert abs(_typing_energy(Succ(arg=Succ(arg=Zero())))) < 1e-9


def test_cons_list_well_typed():
    ast = Cons(head=NatLit(val=1),
               tail=Cons(head=NatLit(val=2), tail=Nil()))
    assert abs(_typing_energy(ast)) < 1e-9


def test_eq_reflexivity_well_typed():
    ast = Lam(param="x", param_ty=TNat(),
              body=Eq(lhs=Var(name="x"), rhs=Var(name="x")))
    assert abs(_typing_energy(ast)) < 1e-9


def test_forall_well_typed():
    ast = Forall(param="n", param_ty=TNat(),
                 body=Eq(lhs=Var(name="n"), rhs=Var(name="n")))
    assert abs(_typing_energy(ast)) < 1e-9


def test_succ_of_bool_ill_typed():
    state, meta = encode_mera(Succ(arg=BoolLit(val=True)))
    H = MeraTypingHamiltonian(meta)
    res = H.residuals(state)
    fired = [k for k, v in res.items() if v > 0.5]
    assert any(k[0] == "T-Succ" for k in fired)


def test_eq_type_mismatch_ill_typed():
    state, meta = encode_mera(Eq(lhs=NatLit(val=1), rhs=BoolLit(val=False)))
    H = MeraTypingHamiltonian(meta)
    res = H.residuals(state)
    assert any(k[0] == "T-Eq" and v > 0.5 for k, v in res.items())
