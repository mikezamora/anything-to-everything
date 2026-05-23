"""Tests for the lemma record dataclasses (spec §3.2)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.composition.lemma_library import (
    Lemma, DerivationMetadata, MeraTensorBundle,
)


def _bundle():
    return MeraTensorBundle(
        n_leaves=8, leaf_dim=16, n_layers=3,
        leaf_vectors=[np.zeros(16) for _ in range(8)],
        disentanglers=[], isometries=[],
    )


def _deriv():
    return DerivationMetadata(
        hamiltonian_id="h0", residual_energy=1e-12, energy_gap=0.5,
        trotter_steps=40, assumptions=(), lemma_deps=(),
        conditional=False, source_run_id="run0",
    )


def test_derivation_metadata_is_frozen():
    d = _deriv()
    with pytest.raises(Exception):
        d.residual_energy = 2.0


def test_conditional_inferred_false_when_no_assumptions():
    assert _deriv().conditional is False


def test_lemma_holds_all_fields(tmp_path):
    from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta
    meta = MeraEncodingMeta(
        n_nodes=1, n_leaves=8, L=3, leaf_dim=16,
        species_of_leaf=["kind", "type", "bid", "value", "tobl",
                         "PAD", "PAD", "PAD"],
        node_of_leaf=[0, 0, 0, 0, 0, -1, -1, -1],
        site_to_ast_path={0: ()}, binder_leaves={}, use_to_binder={},
    )
    lem = Lemma(
        lemma_id="abc", proposition_type="forall x:Nat. Eq x x",
        mera_tensors=_bundle(), encoding_meta=meta,
        derivation=_deriv(), fingerprint=np.zeros(32),
    )
    assert lem.lemma_id == "abc"
    assert lem.fingerprint.shape == (32,)
    assert lem.encoding_meta.n_leaves == 8
