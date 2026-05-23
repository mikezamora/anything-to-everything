"""Tests for validated registration (spec §4.5, acceptance §8.4)."""
from __future__ import annotations
from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, register_lemma, DerivationMetadata, RegistrationResult,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def _deriv(residual):
    return DerivationMetadata(
        hamiltonian_id="h", residual_energy=residual, energy_gap=0.5,
        trotter_steps=40, assumptions=(), lemma_deps=(),
        conditional=False, source_run_id="r")


def test_register_accepts_valid_low_residual(tmp_path):
    lib = LemmaLibrary(tmp_path)
    state, meta = encode_mera(parse(r"\x:Int. x"))
    res = register_lemma(lib, state, meta, hamiltonian=None,
                         derivation=_deriv(1e-12), eps_register=1e-8)
    assert res.accepted and res.reason == "ok"
    assert res.lemma_id in lib.all_ids()


def test_register_rejects_high_residual(tmp_path):
    lib = LemmaLibrary(tmp_path)
    state, meta = encode_mera(parse(r"\x:Int. x"))
    res = register_lemma(lib, state, meta, hamiltonian=None,
                         derivation=_deriv(1.0), eps_register=1e-8)
    assert not res.accepted and res.reason == "residual_too_high"
    assert res.lemma_id is None
    assert len(lib.all_ids()) == 0
    assert (tmp_path / "near_misses.log").exists()


def test_register_rejects_invalid_ast(tmp_path):
    # An ill-typed encoded state: build one whose decoded AST fails the
    # classical type-checker. If no such helper exists, monkeypatch the
    # validator to fail; the test asserts the reason prefix only.
    import src.qft_pcn.composition.lemma_library as L
    lib = LemmaLibrary(tmp_path)
    state, meta = encode_mera(parse(r"\x:Int. x"))
    orig = L._validate_decoded
    L._validate_decoded = lambda ast, ham: (False, "synthetic type error")
    try:
        res = register_lemma(lib, state, meta, hamiltonian=None,
                             derivation=_deriv(1e-12), eps_register=1e-8)
    finally:
        L._validate_decoded = orig
    assert not res.accepted
    assert res.reason.startswith("validation_failed")
    assert len(lib.all_ids()) == 0
