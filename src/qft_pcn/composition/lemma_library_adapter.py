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

from dataclasses import dataclass, field
from typing import Callable

from src.qft_pcn.composition._abstraction_const import LibraryContractError
from src.qft_pcn.composition.abstraction import CanonicalPrimitive
from src.qft_pcn.composition.lemma_library import (
    DerivationMetadata, Lemma, LemmaLibrary, MeraTensorBundle,
    _content_id, bundle_from_mera, register_lemma, structural_fingerprint,
)
from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta


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

    def register(self, entry) -> str | None:
        """Register a solved-problem tuple or a :class:`CanonicalPrimitive`.

        Returns the assigned ``lemma_id`` on success, ``None`` on rejection
        by :func:`register_lemma` (e.g. residual gate). The orchestrator
        treats ``None`` as a no-op.
        """
        if isinstance(entry, CanonicalPrimitive):
            deriv = _primitive_deriv(entry)
            lemma_id = self._save_primitive(entry, deriv)
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
        not source-level ASTs)."""
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
        """Replace ``old_id`` with a new primitive: persist the new lemma
        (the store is append-only, so the old one stays on disk), record
        the redirect in ``replacements`` and mark the old id pruned so
        :meth:`cached_solutions` no longer surfaces it. Returns the new
        lemma id."""
        deriv = _primitive_deriv(new_primitive)
        # Log the replacement in provenance (mutating use_log is fine; it is
        # a list and not part of the frozen primitive's identity).
        new_primitive.provenance.use_log.append(f"replace:{old_id}")
        new_id = self._save_primitive(new_primitive, deriv)
        self.replacements[old_id] = new_id
        self.pruned.add(old_id)
        # The replacement is itself a CanonicalPrimitive (the §10.9 step (1)
        # synthesises L' as a new primitive), so it carries the same
        # "primitive" tier semantics: tensor-only, must be filtered out of
        # cached_solutions to avoid self-replacement (D26).
        self._tiers[new_id] = "primitive"
        return new_id

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
