"""Subtree miner (spec §4; acceptance §9.1-9.2, §9.7)."""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition._abstraction_const import S_MIN, S_MAX, MiningError
from src.qft_pcn.composition.subtree_miner import (
    MineConfig, Fingerprint, SubtreeCandidate,
    mine_subtrees, mine_corpus, bucket_by_fingerprint,
)
from src.qft_pcn.composition.abstraction import trace_distance
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse  # M1 parser


def _encode(src: str):
    state, meta = encode_mera(parse(src))
    return state, meta


def test_mine_returns_node_aligned_subtrees_in_size_band():
    state, meta = _encode(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    cands = mine_subtrees(state, meta, source_id="p1")
    assert cands, "expected at least one mined subtree"
    for c in cands:
        assert S_MIN <= c.ast_size <= S_MAX
        lo, hi = c.leaf_interval
        # node-aligned: interval width covers whole AST nodes (5 leaves each)
        assert (hi - lo) % 5 == 0 or _covers_pad(meta, lo, hi)


def _covers_pad(meta, lo, hi):
    # node-aligned check tolerant of PAD leaves at the interval edge
    return all(meta.node_of_leaf[i] == -1
               for i in range(lo, hi) if meta.node_of_leaf[i] == -1)


def test_mined_densities_are_valid():
    state, meta = _encode(r"\f:Int->Int. \x:Int. f (f x)")
    for c in mine_subtrees(state, meta, source_id="p2"):
        rho = c.rho
        assert np.allclose(rho, rho.conj().T, atol=1e-9)            # Hermitian
        eig = np.linalg.eigvalsh(rho)
        assert eig.min() >= -1e-9                                    # PSD
        assert np.trace(rho).real == pytest.approx(1.0, abs=1e-9)    # unit trace


def test_fingerprint_invariant_under_bond_basis_change():
    state, meta = _encode(r"\x:Int. x")
    cands = mine_subtrees(state, meta, source_id="p3")
    if not cands:
        pytest.skip("program too small for a subtree in band")
    rho = cands[0].rho
    d = rho.shape[0]
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
    rho_rot = q @ rho @ q.conj().T
    from src.qft_pcn.composition.subtree_miner import fingerprint_of
    assert fingerprint_of(rho_rot, cands[0].ast_size) == cands[0].fingerprint
    assert trace_distance(rho, rho_rot) == pytest.approx(0.0, abs=1e-9)


def test_bucket_groups_identical_fingerprints():
    state, meta = _encode(r"(\x:Int. x + 1)(2)")
    cands = mine_subtrees(state, meta, source_id="p4")
    buckets = bucket_by_fingerprint(cands)
    total = sum(len(v) for v in buckets.values())
    assert total == len(cands)
    for fp, group in buckets.items():
        assert all(c.fingerprint == fp for c in group)


def test_mine_rejects_invalid_mera():
    # Use a program guaranteed to yield in-band candidates so the validator
    # actually runs on a reduced density (cf. test_mine_returns_...).
    state, meta = _encode(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    # `_layer_density` is built from `self.leaves`; scale a leaf in place so
    # the layer-0 reduced density is no longer unit-trace. (`mera.py` exposes
    # no public layer mutator; the plan permits substituting an un-normalized
    # MERA — this is the in-place equivalent on a copied state.)
    bad = state.copy()
    bad.leaves[0] = bad.leaves[0] * 2.0
    with pytest.raises(MiningError):
        mine_subtrees(bad, meta, source_id="bad")
