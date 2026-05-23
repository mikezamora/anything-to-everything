"""Tests for bond-dimension compression (spec §4.1, acceptance §8.2)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.composition.lemma_library import (
    bundle_from_mera, compress_bundle,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def _energy(bundle):
    # placeholder energy probe: norm of leaf vectors; replaced by the real
    # H_L expectation once M2's MeraTypingHamiltonian is wired in Task 7.
    return float(sum(np.linalg.norm(v) for v in bundle.leaf_vectors))


def test_compression_preserves_energy_within_tol():
    state, _ = encode_mera(parse(r"\x:Int. x"))
    bundle = bundle_from_mera(state)
    eps = 1e-9
    compressed = compress_bundle(bundle, energy_fn=_energy, eps_compress=eps)
    assert abs(_energy(compressed) - _energy(bundle)) < eps


def test_compression_does_not_grow_storage(tmp_path):
    from src.qft_pcn.composition.lemma_library import save_bundle_npz
    state, _ = encode_mera(parse(r"\x:Int. x"))
    bundle = bundle_from_mera(state)
    compressed = compress_bundle(bundle, energy_fn=_energy, eps_compress=1e-9)
    p_raw, p_cmp = tmp_path / "raw.npz", tmp_path / "cmp.npz"
    save_bundle_npz(bundle, p_raw)
    save_bundle_npz(compressed, p_cmp)
    assert p_cmp.stat().st_size <= p_raw.stat().st_size
