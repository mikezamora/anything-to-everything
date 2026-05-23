"""Tests for bundle<->MERA round-trip and .npz I/O (spec §3.2, §4.1)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.composition.lemma_library import (
    MeraTensorBundle, bundle_from_mera, mera_from_bundle,
    save_bundle_npz, load_bundle_npz,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def _state():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    return state, meta


def test_bundle_from_mera_then_back_is_faithful():
    state, _ = _state()
    bundle = bundle_from_mera(state)
    rebuilt = mera_from_bundle(bundle)
    ov = abs(state.inner(rebuilt)) ** 2
    assert ov > 1 - 1e-10


def test_npz_round_trip(tmp_path):
    state, _ = _state()
    bundle = bundle_from_mera(state)
    path = tmp_path / "b.npz"
    save_bundle_npz(bundle, path)
    loaded = load_bundle_npz(path)
    rebuilt = mera_from_bundle(loaded)
    assert abs(state.inner(rebuilt)) ** 2 > 1 - 1e-10
