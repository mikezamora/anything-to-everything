"""Acceptance suite for MeraTypingHamiltonian (spec §9.1, §9.2, §9.6)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import (
    parse, Succ, Zero, Cons, NatLit, Nil, Lam, Eq, Var, Forall, Fix,
    TNat, TArrow, BoolLit,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, MeraTypingTerm,
)
from src.qft_pcn.logic._mera_typing_rules import mutate_leaf
from src.qft_pcn.logic.mera_encoding import TYPE_INT, TYPE_BOOL

_WELL_TYPED = {
    "P1": parse(r"\x:Int. x"),
    "P2": parse(r"(\x:Int. x + 1)(2)"),
    "P3": parse(r"\f:Int->Int. \x:Int. f (f x)"),
    "P4": parse(r"if 1 < 2 then ((\x:Bool. x)(true)) else false"),
    "P5": parse(r"(\x:Int. (\y:Int. x + y)(3))(4)"),
    "P6": Succ(arg=Succ(arg=Zero())),
    "P7": Cons(head=NatLit(val=1),
               tail=Cons(head=NatLit(val=2), tail=Nil())),
    "P8": Lam(param="x", param_ty=TNat(),
              body=Eq(lhs=Var(name="x"), rhs=Var(name="x"))),
    "P9": Forall(param="n", param_ty=TNat(),
                 body=Eq(lhs=Var(name="n"), rhs=Var(name="n"))),
    "P10": Fix(param="f", param_ty=TArrow(src=TNat(), dst=TNat()),
               body=Lam(param="x", param_ty=TNat(), body=Var(name="x"))),
}


@pytest.mark.parametrize("name", list(_WELL_TYPED))
def test_well_typed_zero_energy(name):
    state, meta = encode_mera(_WELL_TYPED[name])
    H = MeraTypingHamiltonian(meta)
    e = H.total_energy(state)
    assert abs(e) < 1e-9, f"{name}: <H_typing> = {e}; {H.residuals(state)}"


def test_IT1_obligation_violation():
    state, meta = encode_mera(parse(r"\x:Int. x + true"))
    H = MeraTypingHamiltonian(meta)
    assert H.total_energy(state) > 0.5
    big = [k for k, v in H.residuals(state).items() if v > 0.5]
    assert len(big) == 1 and big[0][0] == "T-Obligation"


def test_IT2_succ_violation():
    state, meta = encode_mera(Succ(arg=BoolLit(val=True)))
    H = MeraTypingHamiltonian(meta)
    assert H.total_energy(state) > 0.5
    assert any(k[0] == "T-Succ"
               for k, v in H.residuals(state).items() if v > 0.5)


def test_IT4_eq_violation():
    state, meta = encode_mera(Eq(lhs=NatLit(val=1), rhs=BoolLit(val=False)))
    H = MeraTypingHamiltonian(meta)
    assert any(k[0] == "T-Eq"
               for k, v in H.residuals(state).items() if v > 0.5)


def test_IT5_surgical_lit_violation():
    state, meta = encode_mera(parse(r"\x:Int. 3"))
    type_leaf = meta.layout.leaf_of(1, "type")
    state = mutate_leaf(state, type_leaf, TYPE_INT, TYPE_BOOL)
    H = MeraTypingHamiltonian(meta)
    big = [k for k, v in H.residuals(state).items() if v > 0.5]
    assert ("T-Lit-Int", 1) in big


def test_residuals_sum_to_total():
    state, meta = encode_mera(parse(r"\x:Int. x + true"))
    H = MeraTypingHamiltonian(meta)
    assert abs(H.total_energy(state)
               - sum(H.residuals(state).values())) < 1e-9


def test_structural_same_instance_two_programs():
    s1, m1 = encode_mera(parse(r"\x:Int. x"))
    s2, m2 = encode_mera(parse(r"\y:Bool. y"))
    H1 = MeraTypingHamiltonian(m1)
    assert abs(H1.total_energy(s1)) < 1e-9
    H2 = MeraTypingHamiltonian(m2)
    assert abs(H2.total_energy(s2)) < 1e-9
