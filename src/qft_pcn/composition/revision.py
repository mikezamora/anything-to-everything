"""Revision: LLM-guided alternative decomposition on failure (spec §10).

When a sub-goal fails, ask G's LLM frontend for a different proof
strategy; cache failed decompositions by goal_id so a known-dead
decomposition is never re-proposed.
"""
from __future__ import annotations

from typing import Protocol

from .goal_graph import SubGoal, Node, make_sub_goal


class LLMReviser(Protocol):
    """Duck-typed contract for LLM-backed alternative decomposition.

    `suggest_decomposition` returns an ordered list of alternative
    decomposition dicts (each `{"goal_prop": str, "boundary": dict}` per
    spec §6.5). The orchestrator (Task 7) consults the LLM-backed reviser
    when present; otherwise falls back to `HeuristicReviser`.
    """

    def suggest_decomposition(
        self, goal_prop: str, boundary: dict
    ) -> list[dict]: ...

# Fixed catalogue of structural permutations the heuristic reviser cycles
# through. Each entry rewrites the failed goal into a distinct decomposition.
_HEURISTIC_CATALOGUE = (
    "swap_induction_variable",
    "split_conjunction_other_way",
    "strengthen_induction_hypothesis",
)


class FailedDecompositionCache:
    """Records decompositions known to fail, keyed by goal_id (spec §10)."""

    def __init__(self) -> None:
        self._failed: dict[str, set[tuple[str, ...]]] = {}

    def mark_failed(self, goal_id: str, decomposition: list[SubGoal]) -> None:
        ids = tuple(sg.goal_id for sg in decomposition)
        self._failed.setdefault(goal_id, set()).add(ids)

    def is_failed(self, goal_id: str, decomposition: list[SubGoal]) -> bool:
        ids = tuple(sg.goal_id for sg in decomposition)
        return ids in self._failed.get(goal_id, set())


class HeuristicReviser:
    """Deterministic offline fallback: permute the decomposition from a
    fixed catalogue (spec §10)."""

    def __init__(self) -> None:
        self._cursor: dict[str, int] = {}

    def decompose(self, node: Node) -> list[SubGoal]:
        gid = node.goal.goal_id
        idx = self._cursor.get(gid, 0)
        strategy = _HEURISTIC_CATALOGUE[idx % len(_HEURISTIC_CATALOGUE)]
        self._cursor[gid] = idx + 1
        spec = dict(node.goal.dsl_spec)
        spec["revision_strategy"] = strategy
        # Revision proposes a single replacement sub-goal; inherit the
        # parent's leaf footprint so the integrator clamps onto the same
        # window the original decomposition targeted.
        return [make_sub_goal(spec, goal_prop=f"{node.goal.goal_prop}::{strategy}",
                              boundary=node.goal.boundary,
                              parent_leaves=node.goal.parent_leaves)]


def revise(node: Node, *, llm: LLMReviser | None = None,
           reviser: HeuristicReviser | None = None,
           cache: FailedDecompositionCache | None = None) -> list[SubGoal]:
    """Return an alternative decomposition for a PENDING_REVISION node.

    Prefers G's LLM frontend when provided; falls back to the heuristic
    reviser. Never re-proposes a cached failed decomposition.

    Note: for cursor progression across calls, the caller should keep the
    same `HeuristicReviser` instance alive across invocations (the Task 7
    orchestrator already does so).
    """
    reviser = reviser or HeuristicReviser()
    cache = cache or FailedDecompositionCache()

    if llm is not None:
        suggestion = llm.suggest_decomposition(node.goal.goal_prop,
                                               node.goal.boundary)
        # LLM revisions land on a single contiguous slot per sibling --
        # the LLM does not yet publish an explicit footprint, so we fall
        # back to a single-leaf placeholder per sibling index. A richer
        # LLM contract is its own follow-on (decomposer J-Task).
        alt = [make_sub_goal(item["dsl_spec"], goal_prop=item["goal_prop"],
                             boundary=node.goal.boundary,
                             parent_leaves=(i,))
               for i, item in enumerate(suggestion)]
        if not cache.is_failed(node.goal.goal_id, alt):
            return alt

    # heuristic fallback; skip known-dead decompositions
    for _ in range(len(_HEURISTIC_CATALOGUE)):
        alt = reviser.decompose(node)
        if not cache.is_failed(node.goal.goal_id, alt):
            return alt
    return alt  # exhausted: caller (Task 7) turns this into RevisionExhausted
