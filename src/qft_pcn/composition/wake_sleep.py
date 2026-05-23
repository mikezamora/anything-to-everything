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
    ClusterConfig, CanonicalPrimitive, cluster_candidates,
    significant_clusters, compute_canonical_form, trace_distance,
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
    (spec §6.2). Never mutates a cached MERA in place; never prunes a
    ``"core"`` entry. Returns (n_consolidated, n_pruned).

    A cached lemma is marked stale (collected for pruning) when at least one
    of its mined sub-MERA boundary densities falls within the cluster
    distance threshold of a newly-promoted primitive's canonical density --
    i.e. the new primitive subsumes a piece of that cached solution. The
    actual deletion is delegated to ``library.prune(...)`` so the library
    can honour append-only / core-immunity guarantees.
    """
    if not promoted:
        return 0, 0
    n_consolidated = 0
    stale_dynamic: list[str] = []
    for state, meta, sid in library.cached_solutions():
        # Core-tier cached entries are skipped from consolidation per
        # spec §3.3: they are the foundational lemmas and must not be
        # subsumed by dynamic-ring discoveries.
        if library.tier_of(sid) == "core":
            continue
        cands = mine_subtrees(state, meta, sid, config.mine)
        consolidated_this_sid = False
        for cand in cands:
            for prim in promoted:
                if trace_distance(cand.rho, prim.rho_canonical) \
                        < config.cluster.distance_threshold:
                    prim.provenance.use_log.append(sid)
                    n_consolidated += 1
                    consolidated_this_sid = True
                    break
            if consolidated_this_sid:
                break
        if consolidated_this_sid:
            stale_dynamic.append(sid)
    n_pruned = library.prune(stale_dynamic) if stale_dynamic else 0
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
