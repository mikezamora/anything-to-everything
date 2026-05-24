"""D3 regression: ``LemmaLibrary.re_evaluate_provisional`` must walk
provisional (``derivation.conditional=True``) lemmas, recompute the
residual against a caller-supplied ``energy_fn``, and either promote
(new residual <= gate) or drop (new residual > ceiling). Implements
spec §8 / §6.3 lazy re-evaluation -- the orchestrator-side hook closes
the gap audited as DEVIATIONS.md D3.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.lemma_library import (
    DerivationMetadata,
    Lemma,
    LemmaLibrary,
    MeraTensorBundle,
    structural_fingerprint,
)
from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta


@pytest.fixture(autouse=True)
def _no_dense_cap():  # provisional re-eval uses tiny hand-built bundles
    yield


def _tiny_meta() -> MeraEncodingMeta:
    return MeraEncodingMeta(
        n_nodes=1,
        n_leaves=2,
        L=1,
        leaf_dim=2,
        species_of_leaf=["kind", "value"],
        node_of_leaf=[0, 0],
        site_to_ast_path={0: (0,)},
        binder_leaves={},
        use_to_binder={},
        nested_type_index={},
        children_of_node={0: []},
        n_nodes_max=1,
        hole_regions=[],
        witness_node_ranges=[],
        forall_protected_leaves=set(),
        typehole_regions=[],
    )


def _tiny_bundle() -> MeraTensorBundle:
    """A minimal serializable bundle. The provisional re-eval API only
    reads ``derivation.conditional`` + the residual it computes from the
    supplied energy_fn -- it does NOT re-materialize the MERA, so a
    structurally-trivial bundle is sufficient for these tests."""
    leaf = np.zeros((1, 2, 1), dtype=float)
    leaf[0, 0, 0] = 1.0
    # n_layers=1 -> intra_counts = [N//2] = [1], inter_counts = [0].
    iso = np.eye(2).reshape(2, 1, 2).astype(float)  # (chi_out=2, in1=1, in2=2)
    disent = np.eye(4).reshape(2, 2, 2, 2).astype(float)
    top = np.array([[1.0], [0.0]])
    return MeraTensorBundle(
        n_leaves=2, leaf_dim=2, n_layers=1,
        leaf_vectors=[leaf, leaf.copy()],
        disentanglers=[disent],
        isometries=[iso],
        inter_disentanglers=[],
        top=top,
        layer_dims=(2, 2),
    )


def _make_lemma(lemma_id: str, residual: float, conditional: bool) -> Lemma:
    bundle = _tiny_bundle()
    return Lemma(
        lemma_id=lemma_id,
        proposition_type=f"prop_{lemma_id}",
        mera_tensors=bundle,
        encoding_meta=_tiny_meta(),
        derivation=DerivationMetadata(
            hamiltonian_id=f"H_{lemma_id}",
            residual_energy=residual,
            energy_gap=1.0,
            trotter_steps=0,
            assumptions=(),
            lemma_deps=(),
            conditional=conditional,
            source_run_id=f"run_{lemma_id}",
        ),
        fingerprint=np.zeros(32),
    )


def test_re_evaluate_provisional_drops_failed_lemmas(tmp_path):
    """A provisional lemma whose re-checked residual exceeds the
    CONJECTURE_CEILING must be evicted from the library: the cached
    conjecture is no longer plausible against the refreshed Hamiltonian,
    so clamping it onto a parent would corrupt downstream physics."""
    lib = LemmaLibrary(tmp_path)
    lem = _make_lemma("prov_bad", residual=5e-4, conditional=True)
    lib.save(lem)
    assert "prov_bad" in lib.all_ids()

    # energy_fn returns a residual ABOVE ceiling: must drop.
    def energy_fn(_lemma) -> float:
        return 1e-2  # > ceiling=1e-3

    summary = lib.re_evaluate_provisional(energy_fn)
    assert summary["dropped"] == ["prov_bad"]
    assert summary["promoted"] == []
    assert "prov_bad" not in lib.all_ids()
    # On-disk file is gone too (not just the manifest entry).
    assert not lib._path("prov_bad").exists()


def test_re_evaluate_provisional_promotes_valid(tmp_path):
    """A provisional lemma whose re-checked residual falls below the
    residual_gate must be promoted to ``conditional=False`` (no longer a
    conjecture). The lemma stays in the library; its
    ``derivation.conditional`` flips and the recorded residual updates."""
    lib = LemmaLibrary(tmp_path)
    lem = _make_lemma("prov_good", residual=5e-4, conditional=True)
    lib.save(lem)

    # energy_fn returns a residual BELOW gate: must promote.
    def energy_fn(_lemma) -> float:
        return 1e-9  # <= gate=1e-6

    summary = lib.re_evaluate_provisional(energy_fn)
    assert summary["promoted"] == ["prov_good"]
    assert summary["dropped"] == []

    reloaded = lib.load("prov_good")
    assert reloaded.derivation.conditional is False, (
        "promoted lemma must have conditional=False"
    )
    assert reloaded.derivation.residual_energy == pytest.approx(1e-9)


def test_re_evaluate_provisional_leaves_non_provisional_alone(tmp_path):
    """Non-provisional (``conditional=False``) lemmas must not be touched
    even if energy_fn would return a high residual -- the §8 hook
    ONLY rescans conjectures, not ground states."""
    lib = LemmaLibrary(tmp_path)
    lem = _make_lemma("ground", residual=0.0, conditional=False)
    lib.save(lem)

    def energy_fn(_lemma) -> float:
        return 1.0  # would drop if it were provisional

    summary = lib.re_evaluate_provisional(energy_fn)
    assert summary == {
        "promoted": [], "dropped": [], "kept_provisional": [], "skipped": [],
    }
    assert "ground" in lib.all_ids()


def test_re_evaluate_provisional_mid_band_kept(tmp_path):
    """A provisional lemma whose re-checked residual sits between the
    gate and the ceiling stays provisional (still a conjecture) but its
    residual is refreshed -- a partial re-eval, not a full promotion."""
    lib = LemmaLibrary(tmp_path)
    lem = _make_lemma("prov_mid", residual=5e-4, conditional=True)
    lib.save(lem)

    def energy_fn(_lemma) -> float:
        return 2e-4  # between gate=1e-6 and ceiling=1e-3

    summary = lib.re_evaluate_provisional(energy_fn)
    assert summary["kept_provisional"] == ["prov_mid"]

    reloaded = lib.load("prov_mid")
    assert reloaded.derivation.conditional is True
    assert reloaded.derivation.residual_energy == pytest.approx(2e-4)


def test_re_evaluate_provisional_resolver_returns_none_skips(tmp_path):
    """If ``energy_fn`` returns ``None`` for a lemma (caller has no
    resolver for its hamiltonian_id), the lemma is left untouched and
    reported under ``skipped``. The hook must not erase entries the
    caller cannot re-evaluate."""
    lib = LemmaLibrary(tmp_path)
    lem = _make_lemma("prov_unknown", residual=5e-4, conditional=True)
    lib.save(lem)

    def energy_fn(_lemma):
        return None

    summary = lib.re_evaluate_provisional(energy_fn)
    assert summary["skipped"] == ["prov_unknown"]
    assert "prov_unknown" in lib.all_ids()
    reloaded = lib.load("prov_unknown")
    assert reloaded.derivation.conditional is True
    assert reloaded.derivation.residual_energy == pytest.approx(5e-4)
