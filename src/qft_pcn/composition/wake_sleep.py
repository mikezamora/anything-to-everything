"""Wake-sleep cycle orchestration for abstraction discovery (spec §6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.qft_pcn.composition._abstraction_const import (
    DEFAULT_ALPHA, DEFAULT_CHI_CAP, N_QUIESCENT, LibraryContractError,
)
from src.qft_pcn.composition.subtree_miner import (
    MineConfig, mine_corpus, mine_subtrees,
)
from src.qft_pcn.composition.abstraction import (
    ClusterConfig, Cluster, CanonicalPrimitive, Provenance,
    cluster_candidates, significant_clusters, compute_canonical_form,
    trace_distance,
)


@dataclass
class Problem:
    id: str
    hamiltonian_or_state: Any


@dataclass
class WakeSleepConfig:
    mine: MineConfig = field(default_factory=MineConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    alpha: float = DEFAULT_ALPHA
    chi_cap: int = DEFAULT_CHI_CAP
    residual_eps: float = 1e-6
    n_quiescent: int = N_QUIESCENT


@dataclass
class CycleReport:
    cycle_index: int
    n_solved: int
    n_candidates: int
    n_clusters: int
    n_significant: int
    promoted: list[CanonicalPrimitive]
    n_consolidated: int
    n_pruned: int


# Spec §7 contract: the orchestrator does NOT depend on the raw LemmaLibrary
# surface (save/load/materialize/all_ids/...) -- it depends on an adapter
# (see :class:`LemmaLibraryAdapter`) that bridges the contract below to the
# real library. The wake_sleep_cycle accepts any object that satisfies it,
# so the FakeLemmaLibrary test double and the production adapter both fit.
_REQUIRED_LIBRARY_METHODS = ("register", "cached_solutions", "tier_of",
                             "replace", "prune")


def _check_library(library) -> None:
    for name in _REQUIRED_LIBRARY_METHODS:
        if not callable(getattr(library, name, None)):
            raise LibraryContractError(
                f"LemmaLibrary is missing required method '{name}' (spec §7). "
                f"Wrap a raw LemmaLibrary in LemmaLibraryAdapter "
                f"(src.qft_pcn.composition.lemma_library_adapter) to satisfy "
                f"this contract against the production store.")


SolveFn = Callable[[Problem, Any], tuple[Any, Any, float]]


def wake_sleep_cycle(library, problem_batch: list[Problem], solve: SolveFn,
                     cycle_index: int,
                     config: WakeSleepConfig | None = None,
                     validate: Callable[[CanonicalPrimitive], bool] | None = None,
                     ) -> CycleReport:
    """Run ONE wake-sleep cycle (architecture §10.9 pseudocode; spec §6.1)."""
    if config is None:
        config = WakeSleepConfig()
    _check_library(library)

    # WAKE -------------------------------------------------------------------
    solved: list[tuple[Any, Any, str]] = []
    for problem in problem_batch:
        state, meta, residual = solve(problem, library)
        if residual < config.residual_eps:
            solved.append((state, meta, problem.id))
            library.register((state, meta, problem.id))
    # mine over the whole cached corpus, not just this batch
    corpus = list(library.cached_solutions())

    # DREAM ------------------------------------------------------------------
    candidates = mine_corpus(corpus, config.mine)

    # CLUSTER ----------------------------------------------------------------
    clusters = cluster_candidates(candidates, config.cluster)
    sig = significant_clusters(clusters, len(candidates), config.alpha)

    # ABSTRACT ---------------------------------------------------------------
    promoted: list[CanonicalPrimitive] = []
    for cluster in sig:
        primitive = compute_canonical_form(cluster, config.chi_cap, cycle_index)
        if validate is not None and not validate(primitive):
            continue
        library.register(primitive)
        promoted.append(primitive)

    # CONSOLIDATE ------------------------------------------------------------
    n_consolidated, n_pruned = _consolidate(library, promoted, config)

    return CycleReport(
        cycle_index=cycle_index,
        n_solved=len(solved),
        n_candidates=len(candidates),
        n_clusters=len(clusters),
        n_significant=len(sig),
        promoted=promoted,
        n_consolidated=n_consolidated,
        n_pruned=n_pruned,
    )


def _consolidate(library, promoted: list[CanonicalPrimitive],
                 config: WakeSleepConfig) -> tuple[int, int]:
    """Re-derive cached solutions with the new primitives; prune redundancies
    (spec §6.2, §10.9). Never mutates a cached MERA in place; never prunes
    a ``"core"`` entry. Returns (n_consolidated, n_pruned).

    For each cached lemma ``sid`` whose mined sub-MERA boundary density
    falls within the cluster distance threshold of a newly-promoted
    primitive's canonical density:

    1.  Synthesise a REPLACEMENT primitive ``L'`` for ``sid``. ``L'`` is a
        :class:`CanonicalPrimitive` whose canonical density is the matched
        sub-piece's RDM (the part of the parent the new primitive
        subsumes) and whose provenance carries two ``use_log`` markers:

        - ``derived_from:{sid}`` — the parent lemma this replacement
          stands in for.
        - ``uses_primitive:{prim_source_ids}`` — the newly-promoted
          primitive that ``L'`` invokes as a sub-lemma (shorter overall
          MPS: ``L'`` only needs to encode the sub-piece, P handles the
          bulk substructure).

    2.  Persist ``L'`` via :meth:`library.replace`. This is the atomic
        save-then-prune: the library saves the new lemma FIRST, then
        marks ``sid`` pruned. If saving raises, ``sid`` is left in
        place (no orphan empty-library transition; D11 fix).

    3.  If :meth:`library.replace` is unavailable for the underlying
        store (no spec contract violation -- ``_check_library`` already
        verified its presence), the fallback ``library.prune`` runs
        only AFTER ``L'`` has been registered separately.

    Spec §10.9 lines 877-882: "FOR each |Ψ_i⟩ in library: IF
    can_be_expressed_using_new_primitives: replace with shorter
    solution." Prior to D11 this step silently destroyed cached lemmas
    without registering replacements.
    """
    if not promoted:
        return 0, 0
    n_consolidated = 0
    n_pruned = 0
    for state, meta, sid in library.cached_solutions():
        # Core-tier cached entries are skipped from consolidation per
        # spec §3.3: they are the foundational lemmas and must not be
        # subsumed by dynamic-ring discoveries.
        if library.tier_of(sid) == "core":
            continue
        # D26 belt-and-suspenders: primitives must never appear in
        # cached_solutions (the adapter filters them out), but if a future
        # library implementation surfaces them anyway, skip — a primitive
        # cannot subsume itself. The adapter tier "primitive"/"induction"
        # is the canonical signal here.
        if library.tier_of(sid) in {"primitive", "induction"}:
            continue
        cands = mine_subtrees(state, meta, sid, config.mine)
        matched_cand = None
        matched_prim = None
        for cand in cands:
            for prim in promoted:
                if trace_distance(cand.rho, prim.rho_canonical) \
                        < config.cluster.distance_threshold:
                    matched_cand = cand
                    matched_prim = prim
                    break
            if matched_cand is not None:
                break
        if matched_cand is None:
            continue

        # ----- §10.9 step (1): synthesise replacement L' ------------------
        # Single-member cluster of the matched sub-piece; re-purify into a
        # bounded-bond canonical primitive carrying the subsume + uses
        # provenance markers. The replacement's MERA is the re-purified
        # cand.rho (the sub-piece) -- strictly shorter than the parent's
        # full state because the bulk substructure is now delegated to
        # matched_prim as a sub-lemma.
        replacement = compute_canonical_form(
            Cluster(members=[matched_cand]), config.chi_cap,
            matched_prim.provenance.discovered_in_cycle,
        )
        # Tag the replacement with the parent it derives from AND the
        # primitive it invokes as a sub-lemma. These markers make the
        # save-then-prune transition auditable from provenance alone.
        replacement.provenance.use_log.append(f"derived_from:{sid}")
        for src in matched_prim.provenance.source_ids:
            replacement.provenance.use_log.append(f"uses_primitive:{src}")

        # ----- §10.9 step (2): atomic save-then-prune via library.replace -
        # ``library.replace`` is part of the spec §7 contract (already
        # verified by ``_check_library``). It is required to be atomic:
        # the new lemma must be persisted BEFORE the old one is marked
        # pruned, so a failure in save leaves the old id intact.
        try:
            library.replace(sid, replacement)
        except Exception:
            # The save failed; do NOT prune ``sid``. Re-raise so the
            # caller sees the consolidation failure rather than silently
            # losing data.
            raise

        # Audit trail on the parent primitive: record that it now
        # subsumes a piece of ``sid``. Kept for backward-compat with the
        # prior dual-sense ``use_log`` semantics (discovery provenance
        # + subsumed-source-ids).
        matched_prim.provenance.use_log.append(sid)
        n_consolidated += 1
        n_pruned += 1
    return n_consolidated, n_pruned


def wake_sleep_loop(library, problem_batches: list[list[Problem]],
                    solve: SolveFn,
                    config: WakeSleepConfig | None = None,
                    validate: Callable[[CanonicalPrimitive], bool] | None = None,
                    ) -> list[CycleReport]:
    """Run cycles until n_quiescent consecutive cycles promote nothing, or the
    batches are exhausted (spec §6.1).

    Across cycles in the same loop, primitives whose canonical density matches
    an already-promoted one (within the cluster-config distance threshold)
    are NOT re-promoted -- a re-discovery of the same abstraction is not a
    NEW abstraction. This is what makes the quiescence terminator meaningful:
    once the discoverable structure has been captured, subsequent cycles over
    the same cached corpus yield no new primitives.
    """
    if config is None:
        config = WakeSleepConfig()
    reports: list[CycleReport] = []
    quiescent = 0
    seen: list[CanonicalPrimitive] = []

    def _novel(prim: CanonicalPrimitive) -> bool:
        # Operator-algebraic novelty check (§1.5/§1.6): compare canonical
        # densities under the trace-distance metric used for clustering.
        for prior in seen:
            if trace_distance(prim.rho_canonical, prior.rho_canonical) \
                    < config.cluster.distance_threshold:
                return False
        return True

    def _validate(prim: CanonicalPrimitive) -> bool:
        if not _novel(prim):
            return False
        if validate is not None and not validate(prim):
            return False
        return True

    for cycle_index, batch in enumerate(problem_batches):
        report = wake_sleep_cycle(library, batch, solve, cycle_index,
                                  config, _validate)
        reports.append(report)
        seen.extend(report.promoted)
        if report.promoted:
            quiescent = 0
        else:
            quiescent += 1
            if quiescent >= config.n_quiescent:
                break
    return reports
