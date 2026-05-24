"""Tests for LemmaLibraryAdapter: spec §7 contract bridged to the real
append-only LemmaLibrary (file-backed)."""
from __future__ import annotations

import numpy as np

from src.qft_pcn.composition.lemma_library import LemmaLibrary
from src.qft_pcn.composition.lemma_library_adapter import LemmaLibraryAdapter
from src.qft_pcn.composition.abstraction import (
    CanonicalPrimitive, Provenance,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


_DISTINCT_PROGS = [
    r"\x:Int. x",
    r"\x:Int. 0",
    r"\x:Int. 1",
    r"\x:Int. 2",
    r"\x:Int. x + 1",
]


def _solved(source_id: str, variant: int = 0):
    """Build a (state, meta, source_id) triple via the real encoder.
    ``variant`` selects a distinct identity AST so content-addressed
    storage doesn't merge multiple registrations into one lemma."""
    state, meta = encode_mera(parse(_DISTINCT_PROGS[variant % len(_DISTINCT_PROGS)]))
    return (state, meta, source_id)


def _primitive(source_ids=("s0",), cycle=0):
    """Build a CanonicalPrimitive over a real product MERA. The canonical
    rho is irrelevant to adapter persistence; only the .mera field is read."""
    state, _ = encode_mera(parse(r"\x:Int. x"))
    # rho_canonical placeholder: a 1x1 unit-trace density satisfies
    # downstream assertions; the adapter only reads .mera and .provenance.
    rho = np.array([[1.0 + 0j]])
    prov = Provenance(
        source_ids=tuple(source_ids),
        occurrences=tuple((sid, (0, 1)) for sid in source_ids),
        discovered_in_cycle=cycle,
    )
    return CanonicalPrimitive(
        rho_canonical=rho, mera=state, chi=1,
        avg_trace_distance=0.0, provenance=prov,
    )


def test_adapter_register_persists_via_save(tmp_path):
    """register((state, meta, sid)) flows through register_lemma+save;
    register(primitive) flows through _save_primitive+save."""
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)

    lemma_id = adapter.register(_solved("alpha"))
    assert lemma_id is not None, "solved triple must register"
    assert lemma_id in lib.all_ids()

    prim_id = adapter.register(_primitive(source_ids=("alpha", "beta")))
    assert prim_id is not None
    assert prim_id in lib.all_ids()
    # Primitive proposition_type tag survives the round-trip.
    back = lib.load(prim_id)
    assert back.proposition_type.startswith("primitive:abstract:")


def test_adapter_cached_solutions_iterates_all_ids(tmp_path):
    """cached_solutions yields (state, meta, lemma_id) for every non-pruned
    lemma, matching mine_corpus's expected shape."""
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)
    ids = [adapter.register(_solved(f"s{i}", variant=i)) for i in range(3)]
    assert all(i is not None for i in ids)

    solutions = adapter.cached_solutions()
    assert len(solutions) == 3
    seen_ids = {sid for _, _, sid in solutions}
    assert seen_ids == set(ids)
    # Each tuple's first element is a MERA-like object (has .N, .L).
    for state, meta, _ in solutions:
        assert hasattr(state, "N") and hasattr(state, "L")
        assert meta is not None


def test_adapter_replace_appends_new_save_and_prunes_old(tmp_path):
    """replace persists a fresh lemma (append-only) and prunes the old id
    from the working set."""
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)
    old_id = adapter.register(_primitive(source_ids=("s0",), cycle=0))
    assert old_id is not None
    n_before = len(lib.all_ids())

    new_prim = _primitive(source_ids=("s0", "s1"), cycle=1)
    new_id = adapter.replace(old_id, new_prim)

    assert new_id != old_id
    assert new_id in lib.all_ids()
    assert old_id in lib.all_ids(), "store is append-only; old stays on disk"
    assert adapter.replacements[old_id] == new_id
    assert old_id in adapter.pruned
    assert len(lib.all_ids()) == n_before + 1
    # The replacement is logged in provenance.use_log.
    assert any(entry.startswith("replace:") for entry in new_prim.provenance.use_log)


def test_adapter_prune_records_ids_and_skips_core(tmp_path):
    """prune returns count, skips ids already pruned, skips core-tier ids,
    and removes them from subsequent cached_solutions walks."""
    lib = LemmaLibrary(tmp_path)
    # tier_of_callable forces id 's_core' (we cannot pre-know the real id) to
    # be core; we use a registered solved lemma as the core target.
    core_id_holder: dict[str, str] = {}

    def tier_of(lid: str) -> str:
        return "core" if lid == core_id_holder.get("core") else "dynamic"

    adapter = LemmaLibraryAdapter(lib, tier_of_callable=tier_of)
    a = adapter.register(_solved("a", variant=0))
    b = adapter.register(_solved("b", variant=1))
    c = adapter.register(_solved("c", variant=2))
    assert None not in (a, b, c)
    core_id_holder["core"] = c

    n = adapter.prune([a, b, c])
    assert n == 2, "core entry not pruned, two dynamic entries pruned"
    # Re-prune is idempotent (returns 0 new).
    n2 = adapter.prune([a, b])
    assert n2 == 0

    surviving = {sid for _, _, sid in adapter.cached_solutions()}
    assert surviving == {c}, "only the core entry survives the prune"


def test_cached_solutions_filters_out_primitives(tmp_path):
    """D26: cached_solutions must NOT surface lemmas registered as
    primitives (tier "primitive" / proposition_type "primitive:*"). Without
    this filter, ``_consolidate`` mines the primitive's own MERA, exact-
    matches it against itself, and replaces it with a stub copy of itself
    in the SAME cycle that promoted it."""
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)

    sol_id = adapter.register(_solved("alpha", variant=0))
    prim_id = adapter.register(_primitive(source_ids=("alpha",), cycle=0))
    assert sol_id is not None and prim_id is not None
    assert sol_id != prim_id
    # Both ids are on-disk.
    assert {sol_id, prim_id}.issubset(set(lib.all_ids()))

    surfaced = {sid for _, _, sid in adapter.cached_solutions()}
    assert prim_id not in surfaced, (
        "cached_solutions must not surface primitive-tier lemmas (D26)")
    assert sol_id in surfaced, \
        "concrete solved-triple lemmas must still surface"
    # Adapter tier records the primitive distinctly from "dynamic".
    assert adapter.tier_of(prim_id) == "primitive"


def test_save_primitive_robust_to_byte_identical_re_promotion(tmp_path):
    """D27: ``_save_primitive`` plumbs ``source_run_id`` through
    ``_content_id`` so two cycles that re-promote a byte-identical
    primitive across DIFFERENT cycle indices produce DIFFERENT lemma_ids
    and do not trip :class:`LemmaHashCollision`. (Within the same cycle,
    re-saving the same bundle is an idempotent no-op via the manifest
    dedup path in ``LemmaLibrary.save``.)"""
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)

    # Two primitives with identical source_ids but different cycle indices;
    # the underlying MERA + bundle are byte-identical.
    p0 = _primitive(source_ids=("alpha",), cycle=0)
    p1 = _primitive(source_ids=("alpha",), cycle=1)

    id0 = adapter.register(p0)
    id1 = adapter.register(p1)   # must NOT raise LemmaHashCollision
    assert id0 is not None and id1 is not None
    # Different source_run_ids ("cycle-0" vs "cycle-1") namespace the hash.
    assert id0 != id1, (
        "byte-identical primitives across cycles must namespace by "
        "source_run_id in _content_id (D27)")


def test_cheapest_for_type_skips_primitives(tmp_path):
    """D28: ``cheapest_for_type`` must skip primitive lemmas. Primitives
    persist with ``MeraEncodingMeta(n_nodes=0, ...)`` -> ``n_leaves_L`` of
    0, which would otherwise rank them ahead of every concrete lemma of
    the same proposition_type. Primitives are tensor-only per spec §5.4;
    concrete-AST candidate selection must consider only concrete lemmas."""
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)

    # Register a primitive; its proposition_type is "primitive:abstract:alpha".
    prim_id = adapter.register(_primitive(source_ids=("alpha",), cycle=0))
    assert prim_id is not None
    prim_prop_type = lib.load(prim_id).proposition_type
    assert prim_prop_type.startswith("primitive:")

    # No concrete lemma carries that proposition_type -> cheapest_for_type
    # must return None (the primitive must NOT be returned).
    assert lib.cheapest_for_type(prim_prop_type) is None, (
        "cheapest_for_type must skip primitive lemmas (D28)")
