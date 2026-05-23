"""Tests for the structural fingerprint (spec §4.3)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.composition.lemma_library import (
    structural_fingerprint, fingerprint_distance, FINGERPRINT_DIM,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def test_fingerprint_has_fixed_dim():
    state, _ = encode_mera(parse(r"\x:Int. x"))
    fp = structural_fingerprint(state)
    assert fp.shape == (FINGERPRINT_DIM,)


def test_identical_states_have_zero_distance():
    s1, _ = encode_mera(parse(r"\x:Int. x"))
    s2, _ = encode_mera(parse(r"\y:Int. y"))   # alpha-equivalent
    d = fingerprint_distance(structural_fingerprint(s1),
                             structural_fingerprint(s2))
    assert d < 1e-8


def test_different_states_have_positive_distance():
    s1, _ = encode_mera(parse(r"\x:Int. x"))
    s2, _ = encode_mera(parse(r"(\x:Int. x + 1)(2)"))
    d = fingerprint_distance(structural_fingerprint(s1),
                             structural_fingerprint(s2))
    assert d > 1e-6
