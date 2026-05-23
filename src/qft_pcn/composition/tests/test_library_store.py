"""Tests for LemmaLibrary store + indexing (spec §4, acceptance §8.1,3,5)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, Lemma, DerivationMetadata, MeraTensorBundle,
    bundle_from_mera, structural_fingerprint,
)
from src.qft_pcn.composition.errors import LemmaHashCollision, LemmaNotFound
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def _lemma(prop_type, steps, leaves=8):
    state, meta = encode_mera(parse(r"\x:Int. x"))
    bundle = bundle_from_mera(state)
    deriv = DerivationMetadata(
        hamiltonian_id="h", residual_energy=1e-12, energy_gap=0.5,
        trotter_steps=steps, assumptions=(), lemma_deps=(),
        conditional=False, source_run_id="r")
    lid = f"{prop_type}:{steps}:{leaves}"
    return Lemma(lemma_id=lid, proposition_type=prop_type,
                 mera_tensors=bundle, encoding_meta=meta,
                 derivation=deriv, fingerprint=structural_fingerprint(state)), state


def test_save_load_round_trip(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, _ = _lemma("A", 10)
    lib.save(lem)
    back = lib.load(lem.lemma_id)
    assert back.proposition_type == "A"
    assert back.fingerprint.shape == lem.fingerprint.shape


def test_materialize_is_faithful(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, state = _lemma("A", 10)
    lib.save(lem)
    rebuilt = lib.materialize(lem.lemma_id)
    assert abs(state.inner(rebuilt)) ** 2 > 1 - 1e-10


def test_find_by_type(tmp_path):
    lib = LemmaLibrary(tmp_path)
    a1, _ = _lemma("A", 10); a2, _ = _lemma("A", 20); b1, _ = _lemma("B", 5)
    for l in (a1, a2, b1):
        lib.save(l)
    assert {l.lemma_id for l in lib.find_by_type("A")} == {a1.lemma_id, a2.lemma_id}
    assert {l.lemma_id for l in lib.find_by_type("B")} == {b1.lemma_id}


def test_cheapest_for_type_prefers_fewer_leaves(tmp_path):
    lib = LemmaLibrary(tmp_path)
    big, _ = _lemma("A", 5, leaves=16); small, _ = _lemma("A", 99, leaves=8)
    lib.save(big); lib.save(small)
    assert lib.cheapest_for_type("A").lemma_id == small.lemma_id


def test_find_similar_ranks_by_distance(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, state = _lemma("A", 10)
    lib.save(lem)
    hits = lib.find_similar(structural_fingerprint(state), max_distance=1e-6)
    assert hits and hits[0][0].lemma_id == lem.lemma_id


def test_append_only_resave_is_noop(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, _ = _lemma("A", 10)
    lib.save(lem); lib.save(lem)   # no raise
    assert len(lib.all_ids()) == 1


def test_hash_collision_raises(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, _ = _lemma("A", 10)
    lib.save(lem)
    clash = Lemma(lemma_id=lem.lemma_id, proposition_type="DIFFERENT",
                  mera_tensors=lem.mera_tensors, encoding_meta=lem.encoding_meta,
                  derivation=lem.derivation, fingerprint=lem.fingerprint)
    with pytest.raises(LemmaHashCollision):
        lib.save(clash)


def test_load_unknown_raises(tmp_path):
    with pytest.raises(LemmaNotFound):
        LemmaLibrary(tmp_path).load("nope")
