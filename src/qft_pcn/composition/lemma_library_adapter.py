"""Adapter bridging the spec §7 ``register/cached_solutions/tier_of/replace/prune``
contract to the real append-only ``LemmaLibrary`` surface (``save/load/
materialize/all_ids/...``).

The wake-sleep orchestrator (composition.wake_sleep) is written against the
spec §7 contract. The real :class:`LemmaLibrary` has a different surface
because it is content-addressed and append-only on disk. This adapter is the
load-bearing bridge so the orchestrator can run against the production library
without inventing methods on it (per memory/no-placeholders.md).

Two kinds of entries flow through ``register``:

* A solved-problem tuple ``(state, meta, source_id)`` deposited by the wake
  phase of one cycle. It is wrapped as a :class:`Lemma` (with a minimal
  :class:`DerivationMetadata`) and persisted via
  :func:`register_lemma` so future cycles can mine it from
  :meth:`cached_solutions`.
* A :class:`CanonicalPrimitive` produced by the abstract phase. Its
  :attr:`mera` is wrapped the same way; the proposition type is taken from
  the canonical AST when available, else the primitive's source-id signature.

Replacement and pruning are tracked in two adapter-local sets because the
on-disk store is append-only by contract (spec §4): ``prune`` records the
ids to ignore on subsequent ``cached_solutions`` walks; ``replace`` saves
the new lemma and registers the (old_id -> new_id) pointer in
``replacements``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace as replace_dataclass
from typing import Callable

from src.qft_pcn.composition._abstraction_const import LibraryContractError
from src.qft_pcn.composition.abstraction import CanonicalPrimitive
from src.qft_pcn.composition.lemma_library import (
    DerivationMetadata, Lemma, LemmaLibrary, MeraTensorBundle,
    _content_id, bundle_from_mera, register_lemma, structural_fingerprint,
)
from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta


# D32: spec §4.5 step 1 residual gate. ``register_lemma`` enforces this
# for solved-problem triples (residual_energy >= eps_register =>
# RegistrationResult(False, ..., "residual_too_high")). _save_primitive
# bypasses register_lemma (primitives are tensor-only, no AST-level
# encoding_meta), so it MUST enforce the gate itself or noisy clusters
# (avg_trace_distance > eps_register) silently persist and leak into
# find_similar / find_by_goal_id.
_EPS_REGISTER_PRIMITIVE: float = 1e-8


class PrimitiveResidualExceedsGate(LibraryContractError):
    """Raised when :meth:`LemmaLibraryAdapter._save_primitive` is asked
    to persist a CanonicalPrimitive whose ``avg_trace_distance`` exceeds
    the spec §4.5 residual gate (``eps_register``). Mirrors the
    ``residual_too_high`` rejection in :func:`register_lemma`."""


def _primitive_deriv(primitive: CanonicalPrimitive) -> DerivationMetadata:
    """Build a :class:`DerivationMetadata` from a CanonicalPrimitive's
    provenance. ``residual_energy`` is the primitive's average trace-distance
    to its cluster members (it acts as the "residual" of the abstraction step,
    per spec §5.4); ``hamiltonian_id`` is a deterministic tag built from the
    source-id tuple.
    """
    prov = primitive.provenance
    src_tag = "+".join(prov.source_ids) if prov.source_ids else "abstract"
    return DerivationMetadata(
        hamiltonian_id=f"abstract:{src_tag}",
        residual_energy=float(primitive.avg_trace_distance),
        energy_gap=0.0,
        trotter_steps=0,
        assumptions=(),
        lemma_deps=tuple(prov.source_ids),
        conditional=False,
        source_run_id=f"cycle-{prov.discovered_in_cycle}",
    )


def _consolidated_deriv(primitive: CanonicalPrimitive,
                        parent_lemma) -> DerivationMetadata:
    """Build a :class:`DerivationMetadata` for a consolidation replacement
    lemma ``L'`` (D35). ``L'`` inherits the parent lemma's
    ``proposition_type`` (it proves the same proposition) and records
    the parent's ``source_run_id`` in :attr:`lemma_deps` together with
    the promoted-primitive source ids; ``hamiltonian_id`` is tagged
    ``consolidated:`` to distinguish it from both wake-phase and
    abstract-phase write paths in audit traces.
    """
    prov = primitive.provenance
    src_tag = "+".join(prov.source_ids) if prov.source_ids else "abstract"
    # Preserve the parent's source_run_id so find_by_goal_id can still
    # reach L' under the original goal id (see D32 follow-on).
    parent_run_id = parent_lemma.derivation.source_run_id
    parent_id = parent_lemma.lemma_id
    return DerivationMetadata(
        hamiltonian_id=f"consolidated:{src_tag}",
        residual_energy=float(primitive.avg_trace_distance),
        energy_gap=0.0,
        trotter_steps=parent_lemma.derivation.trotter_steps,
        assumptions=parent_lemma.derivation.assumptions,
        lemma_deps=tuple(prov.source_ids) + (parent_id,),
        conditional=parent_lemma.derivation.conditional,
        source_run_id=parent_run_id,
    )


def _solved_deriv(source_id: str) -> DerivationMetadata:
    """Build a :class:`DerivationMetadata` for a wake-phase solved problem.
    The orchestrator gates registration on ``residual < eps``, so the
    candidate has already passed the residual filter; we record the
    accepted residual as 0.0 and tag the run.
    """
    return DerivationMetadata(
        hamiltonian_id=f"wake:{source_id}",
        residual_energy=0.0,
        energy_gap=0.0,
        trotter_steps=0,
        assumptions=(),
        lemma_deps=(),
        conditional=False,
        source_run_id=source_id,
    )


@dataclass
class LemmaLibraryAdapter:
    """Bridge from spec §7 contract to the real :class:`LemmaLibrary`.

    Parameters
    ----------
    library:
        The underlying file-backed :class:`LemmaLibrary` instance.
    tier_of_callable:
        Optional callable ``(lemma_id) -> tier``. When provided, it takes
        precedence over both the explicit ``_tiers`` map and any heuristic.
        When ``None``, :meth:`tier_of` consults the explicit ``_tiers`` map
        (populated by :meth:`register` / :meth:`replace`) and falls back to
        ``"dynamic"`` for unknown ids. Note: ``use_log`` is *not* a tier
        signal — it carries provenance plus ``"replace:{old_id}"`` markers
        for subsumed primitives, and conflating non-empty ``use_log`` with
        "core" would misclassify replaced/subsumed entries (a §3.3
        violation). A proper tier field on :class:`Lemma` is deferred; see
        ``EXTENSIONS.md``.

    Attributes
    ----------
    replacements:
        Mapping ``{old_id: new_id}`` populated by :meth:`replace`. The store
        is append-only on disk, so this in-memory pointer is the only record
        of replacement.
    pruned:
        Set of lemma ids that have been pruned from the working corpus.
        :meth:`cached_solutions` skips them.
    """

    library: LemmaLibrary
    tier_of_callable: Callable[[str], str] | None = None
    replacements: dict[str, str] = field(default_factory=dict)
    pruned: set[str] = field(default_factory=set)
    # Adapter-side tier map: populated as the orchestrator registers entries.
    # Core entries are immune to pruning per spec §3.3.
    _tiers: dict[str, str] = field(default_factory=dict)
    # D32: spec §4.5 step 1 residual gate for primitive persistence.
    # ``_save_primitive`` rejects CanonicalPrimitive with
    # ``avg_trace_distance > eps_register``. Matches the default in
    # :func:`register_lemma` (1e-8).
    eps_register: float = _EPS_REGISTER_PRIMITIVE

    def register(self, entry) -> str | None:
        """Register a solved-problem tuple or a :class:`CanonicalPrimitive`.

        Returns the assigned ``lemma_id`` on success, ``None`` on rejection
        by :func:`register_lemma` (e.g. residual gate). The orchestrator
        treats ``None`` as a no-op.
        """
        if isinstance(entry, CanonicalPrimitive):
            deriv = _primitive_deriv(entry)
            try:
                lemma_id = self._save_primitive(entry, deriv)
            except PrimitiveResidualExceedsGate:
                # D32: spec §4.5 gate rejection. Mirror register_lemma's
                # ``RegistrationResult(accepted=False, ...)`` contract by
                # returning None (no-op) instead of propagating; the near-
                # miss is already logged inside _save_primitive.
                return None
            # Tag the primitive so cached_solutions filters it out (D26).
            # Tier is "primitive" — distinct from "dynamic" so the
            # consolidation walk can skip it without touching "core" semantics.
            self._tiers[lemma_id] = "primitive"
            return lemma_id

        # Solved-problem triple.
        if not (isinstance(entry, tuple) and len(entry) == 3):
            raise LibraryContractError(
                "LemmaLibraryAdapter.register expects a CanonicalPrimitive "
                f"or a (state, meta, source_id) triple; got {type(entry)!r}")
        state, meta, source_id = entry
        deriv = _solved_deriv(source_id)
        result = register_lemma(
            self.library, state, meta, hamiltonian=None, derivation=deriv)
        if not result.accepted:
            return None
        self._tiers[result.lemma_id] = "dynamic"
        return result.lemma_id

    def _save_primitive(self, primitive: CanonicalPrimitive,
                        deriv: DerivationMetadata) -> str:
        """Persist a CanonicalPrimitive's MERA as a Lemma. We bypass
        :func:`register_lemma` for primitives because they have no AST-level
        :class:`MeraEncodingMeta` (the abstract phase synthesises tensors,
        not source-level ASTs).

        D32: enforce the spec §4.5 step 1 residual gate inline. A
        CanonicalPrimitive whose ``avg_trace_distance > eps_register``
        is a too-noisy cluster representative; persisting it would leak a
        stale primitive into :meth:`find_by_goal_id` / :meth:`find_similar`
        (``cheapest_for_type`` filters by ``primitive:`` prefix, but the
        fingerprint queries do NOT). Mirrors :func:`register_lemma`'s
        ``residual_too_high`` rejection (which logs to ``near_misses.log``).
        """
        avg_td = float(primitive.avg_trace_distance)
        if avg_td > self.eps_register:
            near_log = self.library.root / "near_misses.log"
            with near_log.open("a") as fh:
                fh.write(
                    f"primitive_residual_too_high {avg_td} "
                    f"> eps_register={self.eps_register} "
                    f"source_ids={list(primitive.provenance.source_ids)}\n")
            raise PrimitiveResidualExceedsGate(
                f"CanonicalPrimitive.avg_trace_distance={avg_td} exceeds "
                f"eps_register={self.eps_register}; refusing to persist "
                f"(spec §4.5 residual gate). source_ids="
                f"{list(primitive.provenance.source_ids)}")
        bundle: MeraTensorBundle = bundle_from_mera(primitive.mera)
        fp = structural_fingerprint(primitive.mera)
        prop_type = f"primitive:{deriv.hamiltonian_id}"
        # Deterministic id from the provenance fingerprint. D27: plumb
        # ``source_run_id`` into the content hash so two cycles that re-promote
        # a byte-identical primitive (same tensors, same prop_type) namespace
        # by discovery cycle and do NOT trip LemmaHashCollision in
        # :meth:`library.save`. ``_novel`` in :func:`wake_sleep_loop` shields
        # most call paths, but a direct caller of :func:`wake_sleep_cycle`
        # across cycles would otherwise hit the collision.
        lemma_id = _content_id(bundle, prop_type,
                               source_run_id=deriv.source_run_id)
        # Primitives lack a real encoding_meta; build a minimal placeholder
        # carrying enough shape info for downstream consumers. This is the
        # ONE place where a stub encoding_meta is unavoidable: primitives are
        # tensor-only (spec §5.4), they have no AST.
        n_leaves = primitive.mera.N
        meta = MeraEncodingMeta(
            n_nodes=0,
            n_leaves=n_leaves,
            L=primitive.mera.L,
            leaf_dim=primitive.mera.d_local,
            species_of_leaf=["pad"] * n_leaves,
            node_of_leaf=[0] * n_leaves,
            site_to_ast_path={},
            binder_leaves={},
            use_to_binder={},
        )
        lemma = Lemma(lemma_id=lemma_id, proposition_type=prop_type,
                      mera_tensors=bundle, encoding_meta=meta,
                      derivation=deriv, fingerprint=fp)
        self.library.save(lemma)
        return lemma_id

    def cached_solutions(self) -> list[tuple[object, object, str]]:
        """Iterate every non-pruned lemma in the underlying library, yielding
        ``(materialized_mera, encoding_meta, lemma_id)`` triples — the
        shape :func:`mine_corpus` consumes.

        Primitives (adapter tier in ``{"primitive", "induction"}`` or
        ``proposition_type`` starting with ``"primitive:"``) are filtered
        OUT (D26): they are tensor-only abstractions per spec §5.4 with no
        AST-level :class:`MeraEncodingMeta`. Surfacing them in the
        consolidation walk caused :func:`_consolidate` to mine a
        primitive's own MERA, exact-match its own canonical density, and
        self-replace the primitive with a stub copy of itself. The
        wake-sleep consolidation loop is for PARENT lemmas re-derived
        through new primitives, never primitives themselves.

        Consolidated lemmas (adapter tier ``"consolidated"``, D35) are
        NOT filtered: they are concrete proofs of the parent lemma's
        proposition that delegate bulk substructure to a promoted
        primitive. §10.9 iterative-compression requires them to surface
        in subsequent consolidation passes so depth > 1 chains stack.
        """
        out: list[tuple[object, object, str]] = []
        for lemma_id in self.library.all_ids():
            if lemma_id in self.pruned:
                continue
            if self._tiers.get(lemma_id) in {"primitive", "induction"}:
                continue
            lemma = self.library.load(lemma_id)
            if lemma.proposition_type.startswith("primitive:"):
                continue
            state = self.library.materialize(lemma_id)
            out.append((state, lemma.encoding_meta, lemma_id))
        return out

    def tier_of(self, lemma_id: str) -> str:
        """Return ``"core"`` or ``"dynamic"`` for a lemma id. Override via
        the constructor's ``tier_of_callable`` if a richer tiering policy
        is needed."""
        if self.tier_of_callable is not None:
            return self.tier_of_callable(lemma_id)
        return self._tiers.get(lemma_id, "dynamic")

    def replace(self, old_id: str, new_primitive: CanonicalPrimitive) -> str:
        """Replace ``old_id`` with a consolidation replacement ``L'``:
        persist L' (the store is append-only, so the old one stays on
        disk), record the redirect in ``replacements`` and mark
        ``old_id`` pruned so :meth:`cached_solutions` no longer surfaces
        it. Returns the new lemma id.

        D35 fix: when ``old_id`` is a CONCRETE lemma (carries an
        AST-level :class:`MeraEncodingMeta` and a non-``"primitive:"``
        :attr:`proposition_type`), L' is saved as a CONSOLIDATED lemma
        that INHERITS the parent's ``proposition_type`` and
        ``encoding_meta``. This makes L' reachable by:

        * the §8 cache layer (:meth:`LemmaLibrary.cheapest_for_type`
          filters out ``"primitive:"`` prefixes only), and
        * future consolidation passes (:meth:`cached_solutions` filters
          out adapter tiers ``{"primitive", "induction"}`` and the
          ``"primitive:"`` proposition_type prefix; ``"consolidated"``
          carries neither).

        §10.9 iterative-compression intent restored: consolidations
        stack to depth > 1 because L' itself is eligible as a parent in
        a subsequent cycle.

        Fallback: if ``old_id`` is itself a primitive (no AST-level
        parent metadata to inherit — e.g. the unit-test path that calls
        :meth:`replace` directly on a registered primitive), L' is
        saved via the primitive path. The §10.9 iterative-compression
        chain does not apply to tensor-only primitives.
        """
        # Log the replacement in provenance (mutating use_log is fine; it is
        # a list and not part of the frozen primitive's identity).
        new_primitive.provenance.use_log.append(f"replace:{old_id}")
        # D37: also stamp the consolidation cycle in use_log so the
        # discovery timestamp survives the source_run_id inheritance below
        # (the cycle would otherwise be lost when ``_consolidated_deriv``
        # overrides ``source_run_id`` with the parent's goal_id).
        new_primitive.provenance.use_log.append(
            "consolidated_in_cycle:"
            f"cycle-{new_primitive.provenance.discovered_in_cycle}")

        parent_is_primitive_tier = (
            self._tiers.get(old_id) in {"primitive", "induction"}
        )
        if not parent_is_primitive_tier:
            try:
                parent_lemma = self.library.load(old_id)
            except Exception:
                parent_lemma = None
            if parent_lemma is not None and \
                    not parent_lemma.proposition_type.startswith("primitive:"):
                deriv = _consolidated_deriv(new_primitive, parent_lemma)
                # D37: tag the inherited source_run_id in provenance so
                # the cache-by-goal-id inheritance is auditable from the
                # primitive's use_log alone (the derivation's
                # source_run_id field is the load-bearing one for
                # find_by_goal_id; this entry mirrors it for debug).
                new_primitive.provenance.use_log.append(
                    f"inherited_source_run_id:{deriv.source_run_id}")
                new_id = self._save_consolidated(
                    new_primitive, deriv, parent_lemma)
                self.replacements[old_id] = new_id
                self.pruned.add(old_id)
                # L' carries the parent's proposition_type and is tagged
                # "consolidated" — NOT "primitive". Both §8 cache filters
                # (cheapest_for_type's "primitive:" prefix skip) and the
                # consolidation walk (cached_solutions's tier-set skip)
                # treat consolidated lemmas as concrete: they are
                # reachable to future cache hits AND eligible as parents
                # in further consolidation (D35; depth > 1 restored).
                self._tiers[new_id] = "consolidated"
                return new_id

        # Fallback: no concrete parent metadata to inherit. D37: even on
        # this primitive-only path, inherit the parent primitive's
        # ``source_run_id`` if one is recorded -- otherwise the cache-by-
        # goal-id reverse index loses every chain-of-consolidations
        # entry. Falls back to ``_primitive_deriv``'s ``cycle-{N}`` only
        # when the parent carries no source_run_id (e.g. a synthetic
        # test that registers a primitive directly without going through
        # the wake/promote pipeline).
        deriv = _primitive_deriv(new_primitive)
        try:
            parent_lemma = self.library.load(old_id)
            parent_run_id = parent_lemma.derivation.source_run_id
        except Exception:
            parent_run_id = None
        if parent_run_id:
            new_primitive.provenance.use_log.append(
                f"inherited_source_run_id:{parent_run_id}")
            deriv = replace_dataclass(deriv, source_run_id=parent_run_id)
        new_id = self._save_primitive(new_primitive, deriv)
        self.replacements[old_id] = new_id
        self.pruned.add(old_id)
        self._tiers[new_id] = "primitive"
        return new_id

    def _save_consolidated(self, primitive: CanonicalPrimitive,
                           deriv: DerivationMetadata,
                           parent_lemma: Lemma) -> str:
        """Persist a consolidation replacement L' under the PARENT
        lemma's ``proposition_type`` (D35) with a PROJECTED
        ``encoding_meta`` (D39) restricted to the matched sub-piece's
        leaves.

        L' is the §10.9 "shorter solution": same proposition as the
        parent, but its MERA delegates the bulk to the newly-promoted
        primitive recorded in :attr:`primitive.provenance.use_log` via
        ``uses_primitive:{src}`` markers.

        D39: ``bundle = bundle_from_mera(primitive.mera)`` carries
        ``bundle.n_leaves = primitive.mera.N`` (sub-piece width). The
        encoding_meta MUST agree on leaf count or downstream consumers
        (``decode_mera``, ``mine_subtrees``, ``cached_solutions``)
        IndexError or reconstruct a phantom AST. We synthesise the
        projected meta via :meth:`MeraEncodingMeta.project_to_leaves`
        using the matched sub-piece's leaf interval from
        :attr:`primitive.provenance.occurrences`. The §10.9 wake-sleep
        consolidator builds replacement primitives from a single-member
        cluster (``compute_canonical_form(Cluster(members=[matched_cand]),
        ...)``), so the occurrence list has exactly one entry whose
        ``leaf_interval`` width equals ``primitive.mera.N``. We pick
        that occurrence (defensively: width-match) and project the
        parent's meta to ``[lo, hi)``.

        The projected ``n_leaves`` is smaller than the parent's, so
        ``_n_leaves_L = 5 * n_nodes`` is smaller for L' than for the
        parent. This makes L' rank STRICTLY ABOVE the parent in
        :meth:`LemmaLibrary.cheapest_for_type` (D38: combined with the
        adapter-level pruned-filter override of ``cheapest_for_type``).
        """
        bundle: MeraTensorBundle = bundle_from_mera(primitive.mera)
        fp = structural_fingerprint(primitive.mera)
        prop_type = parent_lemma.proposition_type
        # D39: find the matched sub-piece's leaf interval on the parent.
        # ``compute_canonical_form`` records every cluster-member's
        # ``leaf_interval`` in ``provenance.occurrences``; the
        # consolidation path uses a single-member cluster, so the list
        # has exactly one entry. We defensively match by width to handle
        # any future multi-member consolidations.
        sub_piece_N = int(primitive.mera.N)
        leaf_interval = None
        for _src, interval in primitive.provenance.occurrences:
            lo, hi = int(interval[0]), int(interval[1])
            if hi - lo == sub_piece_N and lo >= 0 and \
                    hi <= parent_lemma.encoding_meta.n_leaves:
                leaf_interval = (lo, hi)
                break
        if leaf_interval is None:
            # No occurrence aligns with the sub-piece on the parent — the
            # primitive was promoted from a different source. Fall back
            # to a leaf-only stub meta whose scalar shape
            # (n_leaves, n_nodes=0, leaf_dim, L) matches the bundle but
            # carries no AST structure. This preserves bundle/meta
            # consistency at the cost of dropping the AST projection;
            # ``_n_leaves_L`` then falls back to its content-hash tail
            # parse on the lemma_id (a known limitation, noted in
            # EXTENSIONS for richer cross-parent consolidations).
            n_leaves = sub_piece_N
            projected_meta = MeraEncodingMeta(
                n_nodes=0,
                n_leaves=n_leaves,
                L=primitive.mera.L,
                leaf_dim=parent_lemma.encoding_meta.leaf_dim,
                species_of_leaf=["pad"] * n_leaves,
                node_of_leaf=[-1] * n_leaves,
                site_to_ast_path={},
                binder_leaves={},
                use_to_binder={},
            )
        else:
            projected_meta = parent_lemma.encoding_meta.project_to_leaves(
                leaf_interval[0], leaf_interval[1])
        # Namespace the content hash by parent source_run_id so
        # consolidating the same parent twice (or two distinct parents
        # to byte-identical L') do not collide in the manifest.
        lemma_id = _content_id(bundle, prop_type,
                               source_run_id=deriv.source_run_id)
        lemma = Lemma(lemma_id=lemma_id, proposition_type=prop_type,
                      mera_tensors=bundle,
                      encoding_meta=projected_meta,
                      derivation=deriv, fingerprint=fp)
        self.library.save(lemma)
        return lemma_id

    def cheapest_for_type(self, proposition_type: str):
        """Adapter-level :meth:`LemmaLibrary.cheapest_for_type` that
        filters out pruned ids (D38).

        The underlying :meth:`LemmaLibrary.cheapest_for_type` iterates
        ``self._manifest`` directly (the store is append-only by spec
        §4); it cannot know about the adapter-side ``self.pruned`` set
        that :meth:`replace` populates. Without this override, a
        subsumed parent stays in the candidate list next to its
        replacement L', and the sort tie-break by ``lemma_id`` (a
        content-addressed sha-ish) becomes non-deterministic between
        the two.

        With D39's projected ``encoding_meta`` on L', the primary sort
        key ``_n_leaves_L`` is STRICTLY SMALLER for L' than for the
        parent — but we still filter the parent out so the manifest
        reflects the §10.9 "subsume parent with shorter solution"
        contract regardless of the parent's ``n_leaves_L`` value (e.g.
        a same-width parent left in via a degenerate projection).
        """
        cands = [
            (m["n_leaves_L"], m["trotter_steps"], lid)
            for lid, m in self.library._manifest.items()
            if m["proposition_type"] == proposition_type
            and not m["proposition_type"].startswith("primitive:")
            and lid not in self.pruned
        ]
        if not cands:
            return None
        cands.sort()
        return self.library.load(cands[0][2])

    def prune(self, lemma_ids) -> int:
        """Mark ``lemma_ids`` as pruned. Core-tier entries are skipped (spec
        §3.3). Returns the number of ids actually pruned (excluding core
        and ids already in :attr:`pruned`)."""
        if isinstance(lemma_ids, str):
            lemma_ids = (lemma_ids,)
        n = 0
        for lid in lemma_ids:
            if self.tier_of(lid) == "core":
                continue
            if lid in self.pruned:
                continue
            self.pruned.add(lid)
            n += 1
        return n

    def has_induction_primitive(self) -> bool:
        """Convenience predicate mirroring :class:`FakeLemmaLibrary` so the
        step-counting solver in tests can probe a real adapter too."""
        # A primitive lemma_id carries proposition_type starting with
        # "primitive:" (see :meth:`_save_primitive`).
        for lid in self.library.all_ids():
            if lid in self.pruned:
                continue
            lemma = self.library.load(lid)
            if lemma.proposition_type.startswith("primitive:"):
                return True
        return False
