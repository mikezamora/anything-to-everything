"""Tests for the synthesis Hamiltonian blocks (spec §4.2-§4.5, §9.5)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, Var, IntLit, HoleVar
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_synthesis.problem import (
    SynthesisProblem, IOExample, HamiltonianWeights,
)
from src.qft_pcn.logic.mera_synthesis.hamiltonian import (
    build_size_terms, build_target_type_terms, build_example_terms,
    compile_mera_synthesis_hamiltonian,
)
from src.qft_pcn.logic.mera_debugger import NamedMeraTerm


def _concrete():
    return Lam(param="x", param_ty=TInt(), body=Var(name="x"))


def test_size_terms_are_named_mera_terms():
    state, meta = encode_mera(_concrete())
    terms = build_size_terms(meta, w_S=0.1)
    assert terms, "no size terms built"
    for t in terms:
        assert isinstance(t, NamedMeraTerm)
        assert t.rule_class == "S-Size"


def test_size_penalty_counts_non_pad_nodes():
    state, meta = encode_mera(_concrete())
    terms = build_size_terms(meta, w_S=0.1)
    total = sum(t.expectation(state) for t in terms)
    # 2 non-PAD nodes -> 2 * 0.1 = 0.2 (PAD projector complement).
    assert abs(total - 0.2) < 1e-6


def test_target_type_term_zero_when_type_matches():
    from src.qft_pcn.logic.ast import TArrow
    state, meta = encode_mera(_concrete())
    tt = TArrow(src=TInt(), dst=TInt())
    terms = build_target_type_terms(meta, tt, w_Y=2.0)
    total = sum(t.expectation(state) for t in terms)
    assert abs(total) < 1e-6, total


def test_target_type_term_nonzero_when_type_mismatches():
    from src.qft_pcn.logic.ast import TBool
    state, meta = encode_mera(_concrete())
    terms = build_target_type_terms(meta, TBool(), w_Y=2.0)
    total = sum(t.expectation(state) for t in terms)
    assert total > 1e-6, total


def test_compose_yields_total_energy():
    hole = HoleVar(candidates=("x",))
    problem = SynthesisProblem(
        sketch=Lam(param="x", param_ty=TInt(), body=hole),
        examples=(IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3)),),
    )
    # encode the witness-augmented sketch (Task 7 builds it).
    from src.qft_pcn.logic.mera_synthesis.encode_ext import (
        _witness_augmented_ast,
    )
    aug = _witness_augmented_ast(problem.sketch, problem.examples)
    state, meta = encode_mera(aug, chi_layer=32)
    H = compile_mera_synthesis_hamiltonian(meta, problem, HamiltonianWeights())
    e = H.total_energy(state)
    assert isinstance(e, float)
