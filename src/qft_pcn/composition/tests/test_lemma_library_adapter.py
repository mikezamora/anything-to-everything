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


def test_save_primitive_refuses_when_avg_trace_distance_exceeds_eps_register(
        tmp_path):
    """D32: ``_save_primitive`` must enforce the spec §4.5 residual gate.
    A CanonicalPrimitive whose ``avg_trace_distance > eps_register`` must
    NOT land on disk (would otherwise leak into ``find_similar`` /
    ``find_by_goal_id``).

    Acceptance:
    * ``adapter.register(primitive)`` returns ``None`` (mirror
      ``register_lemma``'s ``residual_too_high`` rejection).
    * ``library.all_ids()`` is empty afterwards (no stale primitive).
    * A near-miss line is appended to ``near_misses.log``.
    * Calling ``_save_primitive`` directly raises
      :class:`PrimitiveResidualExceedsGate`.
    """
    from src.qft_pcn.composition.lemma_library_adapter import (
        PrimitiveResidualExceedsGate, _primitive_deriv,
    )

    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)  # eps_register default 1e-8

    # Construct a noisy primitive — avg_trace_distance well above eps_register.
    state, _ = encode_mera(parse(r"\x:Int. x"))
    rho = np.array([[1.0 + 0j]])
    prov = Provenance(
        source_ids=("noisy_alpha", "noisy_beta"),
        occurrences=(("noisy_alpha", (0, 1)), ("noisy_beta", (0, 1))),
        discovered_in_cycle=0,
    )
    noisy = CanonicalPrimitive(
        rho_canonical=rho, mera=state, chi=1,
        avg_trace_distance=1e-3,  # >> eps_register=1e-8
        provenance=prov,
    )

    # register(...) must return None (refusal), not propagate.
    result = adapter.register(noisy)
    assert result is None, (
        "register() must refuse a primitive whose avg_trace_distance "
        "exceeds eps_register (D32)")
    assert list(lib.all_ids()) == [], (
        "no primitive must land on disk when the residual gate refuses (D32)")

    # Near-miss log records the rejection.
    near_log = tmp_path / "near_misses.log"
    assert near_log.exists(), "near_misses.log must be created on refusal"
    log_text = near_log.read_text()
    assert "primitive_residual_too_high" in log_text
    assert "noisy_alpha" in log_text

    # Direct call to _save_primitive raises the typed exception.
    deriv = _primitive_deriv(noisy)
    try:
        adapter._save_primitive(noisy, deriv)
    except PrimitiveResidualExceedsGate as exc:
        assert "eps_register" in str(exc)
    else:
        raise AssertionError(
            "_save_primitive must raise PrimitiveResidualExceedsGate when "
            "avg_trace_distance > eps_register")

    # Sanity: a primitive AT the gate (avg_trace_distance == 0.0) still
    # persists fine — the gate is strictly ``>``.
    clean = CanonicalPrimitive(
        rho_canonical=rho, mera=state, chi=1,
        avg_trace_distance=0.0, provenance=prov,
    )
    clean_id = adapter.register(clean)
    assert clean_id is not None
    assert clean_id in lib.all_ids()


# --- D38 / D39 -----------------------------------------------------------

def _parent_lemma_for_consolidation(adapter, lib, source_id: str = "parent0"):
    r"""Register a real solved-problem lemma whose encoding has enough
    leaves that a node-aligned sub-interval is a strict subset.
    ``\x. \y. x`` encodes to two binders (Lam) -> several nodes;
    encode_mera yields a meta with n_leaves >= 10. Returns
    ``(parent_id, parent_lemma)``."""
    state, meta = encode_mera(parse(r"\x:Int. \y:Int. x"))
    pid = adapter.register((state, meta, source_id))
    assert pid is not None, "parent solved-triple must register"
    return pid, lib.load(pid)


def _consolidation_primitive(parent_lemma, sub_n_leaves: int,
                             source_id: str = "child0",
                             parent_sid_for_log: str = "parent0"):
    """Build a CanonicalPrimitive whose ``mera.N == sub_n_leaves`` and whose
    provenance carries a single occurrence on ``parent_sid_for_log`` with
    interval ``(0, sub_n_leaves)`` -- the node-aligned sub-piece on the
    parent's leftmost leaves. The MERA itself is a vacuum product MERA of
    width ``sub_n_leaves`` (independent of the parent's tensors; D39 only
    requires the bundle/meta shape to match, not the parent's amplitudes)."""
    from src.qft_pcn.qft.mera import MERA
    leaf = np.zeros(parent_lemma.encoding_meta.leaf_dim, dtype=complex)
    leaf[0] = 1.0
    mera = MERA.from_product([leaf] * sub_n_leaves)
    rho = np.array([[1.0 + 0j]])
    prov = Provenance(
        source_ids=(source_id,),
        occurrences=((parent_sid_for_log, (0, sub_n_leaves)),),
        discovered_in_cycle=1,
    )
    return CanonicalPrimitive(
        rho_canonical=rho, mera=mera, chi=1,
        avg_trace_distance=0.0, provenance=prov,
    )


def test_save_consolidated_projects_encoding_meta(tmp_path):
    """D39: ``_save_consolidated`` must persist L' with an encoding_meta
    whose ``n_leaves`` matches the sub-piece's ``bundle.n_leaves``
    (= primitive.mera.N), NOT inherit the parent's full-AST n_leaves.

    Without this, downstream consumers indexing leaves by
    ``encoding_meta.n_leaves`` IndexError past ``bundle.n_leaves``, and
    ``decode_mera`` reconstructs a phantom AST from leaves that don't
    exist in L'.
    """
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)
    parent_id, parent_lemma = _parent_lemma_for_consolidation(adapter, lib)
    parent_n_leaves = parent_lemma.encoding_meta.n_leaves
    # Sub-piece: a power-of-2 leftmost interval on the parent's leaves
    # (``_node_aligned_intervals`` enumerates widths 2, 4, 8, ...; the
    # MERA binary ascend requires a power-of-two width).
    sub_n = 4
    assert parent_n_leaves > sub_n, (
        f"parent must have strictly more leaves than the sub-piece for "
        f"D39 to be observable; got parent={parent_n_leaves} sub={sub_n}")
    prim = _consolidation_primitive(parent_lemma, sub_n,
                                    parent_sid_for_log=parent_id)
    new_id = adapter.replace(parent_id, prim)
    L_prime = lib.load(new_id)
    # Core D39 invariant: bundle and meta agree on leaf count.
    assert L_prime.encoding_meta.n_leaves == sub_n, (
        f"D39: L'.encoding_meta.n_leaves must equal sub-piece N={sub_n}, "
        f"got {L_prime.encoding_meta.n_leaves} (parent had "
        f"{parent_n_leaves})")
    assert L_prime.encoding_meta.n_leaves != parent_n_leaves, (
        "D39: L'.encoding_meta.n_leaves must NOT inherit the parent's "
        "full-AST leaf count")
    # Derived: n_nodes scales with sub-piece, species_of_leaf restricted.
    assert L_prime.encoding_meta.n_nodes == len(set(parent_lemma.encoding_meta.node_of_leaf[:sub_n]) - {-1})
    assert len(L_prime.encoding_meta.species_of_leaf) == sub_n
    # Bundle/meta consistency: indexing leaves up to meta.n_leaves stays
    # within the bundle's leaf array.
    bundle = L_prime.mera_tensors
    assert bundle.n_leaves == L_prime.encoding_meta.n_leaves, (
        f"bundle.n_leaves ({bundle.n_leaves}) must equal "
        f"meta.n_leaves ({L_prime.encoding_meta.n_leaves})")


def test_save_consolidated_decode_reconstructs_sub_piece_ast(tmp_path):
    """D39: with the projected meta, leaf-indexed consumers of L' see
    only the sub-piece's leaves, not phantom parent indices. We check
    every structural invariant a downstream decoder/iterator relies on:

    * ``meta.n_leaves == bundle.n_leaves`` (no IndexError when iterating
      leaves up to ``meta.n_leaves``).
    * ``len(meta.node_of_leaf) == bundle.n_leaves`` and
      ``len(meta.species_of_leaf) == bundle.n_leaves`` (both are
      leaf-indexed by every consumer).
    * Every leaf referenced in ``meta.site_to_ast_path`` /
      ``meta.binder_leaves`` / ``meta.use_to_binder`` lies in
      ``[0, bundle.n_leaves)`` (no parent-AST phantom leaves leak
      through).
    * ``meta.n_nodes`` equals the count of distinct touched nodes in
      the projected ``node_of_leaf`` (a phantom-AST decoder reading
      this scalar walks the correct number of nodes).

    Pre-D39 fix: the inherited meta carries the parent's full-AST
    ``n_leaves``, ``site_to_ast_path`` keyed on parent leaf indices >=
    bundle.n_leaves, and parent ``n_nodes``. Any leaf-indexed consumer
    (decode_mera, mine_subtrees) IndexErrors or rebuilds a phantom AST
    from indices that no longer exist. Post-fix: every invariant
    above holds.
    """
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)
    parent_id, parent_lemma = _parent_lemma_for_consolidation(adapter, lib)
    sub_n = 4
    prim = _consolidation_primitive(parent_lemma, sub_n,
                                    parent_sid_for_log=parent_id)
    new_id = adapter.replace(parent_id, prim)
    L_prime = lib.load(new_id)
    bundle = L_prime.mera_tensors
    meta = L_prime.encoding_meta
    assert meta.n_leaves == bundle.n_leaves == sub_n
    assert len(meta.node_of_leaf) == sub_n, (
        f"D39: node_of_leaf must have one entry per sub-piece leaf; "
        f"got {len(meta.node_of_leaf)} entries for {sub_n} leaves")
    assert len(meta.species_of_leaf) == sub_n
    # No parent-AST phantom leaves leak through into the projected meta.
    for leaf in meta.site_to_ast_path.keys():
        assert 0 <= leaf < sub_n, (
            f"D39: site_to_ast_path leaf {leaf} out of sub-piece range "
            f"[0, {sub_n}); parent leaf would have leaked through")
    for _ast_node, leaf in meta.binder_leaves.items():
        assert 0 <= leaf < sub_n
    for use, binder in meta.use_to_binder.items():
        assert 0 <= use < sub_n and 0 <= binder < sub_n
    # n_nodes is the count of distinct touched parent nodes (after
    # remap to a contiguous 0..k-1 range).
    expected_n_nodes = len(
        set(parent_lemma.encoding_meta.node_of_leaf[:sub_n]) - {-1})
    assert meta.n_nodes == expected_n_nodes


def test_cheapest_for_type_skips_pruned_parents(tmp_path):
    """D38: the adapter's ``cheapest_for_type`` must filter out pruned
    parent lemmas. After ``replace(parent_id, L')``, ``parent_id`` is in
    ``adapter.pruned`` (the store is append-only, so the parent stays on
    disk). Without the filter, the library-level ``cheapest_for_type``
    iterates the manifest and sees BOTH the parent and L' under the same
    proposition_type; sort-key ties (after D39's projected meta) fall to
    the lemma_id content-hash tie-break, which is non-deterministic and
    may return the parent instead of L'.

    With the D38 fix, the adapter's ``cheapest_for_type`` excludes the
    parent and returns L'.
    """
    lib = LemmaLibrary(tmp_path)
    adapter = LemmaLibraryAdapter(lib)
    parent_id, parent_lemma = _parent_lemma_for_consolidation(adapter, lib)
    parent_prop_type = parent_lemma.proposition_type
    sub_n = 4
    prim = _consolidation_primitive(parent_lemma, sub_n,
                                    parent_sid_for_log=parent_id)
    new_id = adapter.replace(parent_id, prim)
    # The parent and L' both carry parent_prop_type and both live on disk.
    assert parent_id in lib.all_ids()
    assert new_id in lib.all_ids()
    assert lib.load(new_id).proposition_type == parent_prop_type
    assert parent_id in adapter.pruned
    # Adapter-level cheapest_for_type filters the pruned parent out.
    result = adapter.cheapest_for_type(parent_prop_type)
    assert result is not None
    assert result.lemma_id == new_id, (
        f"D38: adapter.cheapest_for_type must return L' (not pruned parent); "
        f"got {result.lemma_id}, expected {new_id} (parent was {parent_id})")
    # Sanity: the underlying library-level cheapest_for_type would not
    # filter the parent (it scans manifest directly and has no view of
    # adapter.pruned). Without the adapter override the test was
    # non-deterministic.
