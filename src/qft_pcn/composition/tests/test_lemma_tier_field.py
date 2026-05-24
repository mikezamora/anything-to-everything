"""D19 EXTENSIONS resolution: tests for the first-class ``Lemma.tier``
field that replaces the prior ``LemmaLibraryAdapter._tiers`` sidecar dict.

The two cases pinned here:

1. ``test_lemma_tier_round_trips_through_save_load`` — building a Lemma
   with ``tier="primitive"`` and saving/loading it through
   :class:`LemmaLibrary` preserves the tier. Pre-D19 the tier lived in
   an adapter-side dict and was lost on every restart.
2. ``test_adapter_tier_of_callable_reads_lemma_tier`` — the adapter's
   :meth:`tier_of` now reads :attr:`Lemma.tier` from the persisted
   record, NOT a sidecar map. Registering a primitive (tier
   ``"primitive"``) and a solved triple (tier ``"dynamic"``) and probing
   :meth:`tier_of` returns the values stamped at save time.
"""
from __future__ import annotations

import numpy as np

from src.qft_pcn.composition.lemma_library import (
    DerivationMetadata, Lemma, LemmaLibrary, bundle_from_mera,
    structural_fingerprint,
)
from src.qft_pcn.composition.lemma_library_adapter import LemmaLibraryAdapter
from src.qft_pcn.composition.abstraction import (
    CanonicalPrimitive, Provenance,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


def _build_lemma(tier: str) -> Lemma:
    """Build a real Lemma over a product MERA with the requested tier."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    bundle = bundle_from_mera(state)
    fp = structural_fingerprint(state)
    deriv = DerivationMetadata(
        hamiltonian_id="test-d19",
        residual_energy=0.0,
        energy_gap=1.0,
        trotter_steps=0,
        assumptions=(),
        lemma_deps=(),
        conditional=False,
        source_run_id="test-d19",
    )
    return Lemma(
        lemma_id=f"test-d19:{tier}",
        proposition_type="Int -> Int",
        mera_tensors=bundle,
        encoding_meta=meta,
        derivation=deriv,
        fingerprint=fp,
        tier=tier,
    )


def test_lemma_tier_round_trips_through_save_load(tmp_path):
    """tier="primitive" survives save+load through LemmaLibrary's npz."""
    lib = LemmaLibrary(tmp_path)
    lem = _build_lemma(tier="primitive")
    lib.save(lem)

    loaded = lib.load(lem.lemma_id)
    assert loaded.tier == "primitive", (
        "Lemma.tier must round-trip through LemmaLibrary.save/load; "
        f"got {loaded.tier!r}")


def test_lemma_tier_default_is_dynamic(tmp_path):
    """Building a Lemma without a tier defaults to 'dynamic' (matching
    the spec §3.3 default ring for wake-phase entries)."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    bundle = bundle_from_mera(state)
    fp = structural_fingerprint(state)
    deriv = DerivationMetadata(
        hamiltonian_id="t",
        residual_energy=0.0,
        energy_gap=1.0,
        trotter_steps=0,
        assumptions=(),
        lemma_deps=(),
        conditional=False,
        source_run_id="t",
    )
    lem = Lemma(
        lemma_id="t:default",
        proposition_type="Int -> Int",
        mera_tensors=bundle,
        encoding_meta=meta,
        derivation=deriv,
        fingerprint=fp,
    )
    assert lem.tier == "dynamic"


def test_adapter_tier_of_callable_reads_lemma_tier(tmp_path):
    """The adapter's tier_of() must read Lemma.tier directly (no sidecar).

    Registering a CanonicalPrimitive stamps tier="primitive"; registering
    a (state, meta, source_id) triple stamps tier="dynamic". After save,
    LemmaLibraryAdapter.tier_of returns the persisted field — even
    against a fresh adapter that never saw the original register() call.
    """
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)

    # (a) solved triple — should be tier "dynamic"
    state, meta = encode_mera(parse(r"\x:Int. x"))
    solved_id = adapter.register((state, meta, "wake-source"))
    assert solved_id is not None

    # (b) primitive — should be tier "primitive"
    prim_state, _ = encode_mera(parse(r"\x:Int. 0"))
    rho = np.array([[1.0 + 0j]])
    prov = Provenance(
        source_ids=("p0",),
        occurrences=(("p0", (0, 1)),),
        discovered_in_cycle=0,
    )
    primitive = CanonicalPrimitive(
        rho_canonical=rho, mera=prim_state, chi=1,
        avg_trace_distance=0.0, provenance=prov,
    )
    prim_id = adapter.register(primitive)
    assert prim_id is not None

    # The sidecar dict no longer exists; this would AttributeError if
    # someone reintroduces it without updating tier_of.
    assert not hasattr(adapter, "_tiers"), (
        "_tiers sidecar must be removed (D19 EXTENSIONS resolution); "
        "tier_of should consult Lemma.tier on the persisted record.")

    # Read via the adapter API.
    assert adapter.tier_of(solved_id) == "dynamic"
    assert adapter.tier_of(prim_id) == "primitive"

    # Cross-process simulation: a brand-new adapter (no in-memory state)
    # must still report the correct tier because it lives on the .npz.
    fresh = LemmaLibraryAdapter(LemmaLibrary(tmp_path))
    assert fresh.tier_of(solved_id) == "dynamic"
    assert fresh.tier_of(prim_id) == "primitive"


def test_adapter_tier_of_callable_overrides_lemma_tier(tmp_path):
    """When a tier_of_callable is supplied at adapter construction it
    takes precedence over the persisted Lemma.tier — same precedence the
    sidecar map had pre-D19."""
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(
        lib, tier_of_callable=lambda lid: "core")
    state, meta = encode_mera(parse(r"\x:Int. x"))
    sid = adapter.register((state, meta, "src"))
    assert sid is not None
    # Persisted tier is "dynamic"; the callable override wins.
    assert lib.load(sid).tier == "dynamic"
    assert adapter.tier_of(sid) == "core"
