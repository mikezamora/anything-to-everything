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
# through. Each entry rewrites the failed goal into a SUBSTRATE-distinct
# decomposition (different parent_leaves footprint, different sub-goal count,
# or different boundary content) -- NOT just a metadata tag. A pure-tag
# rewrite (D8) leaves the downstream compiler with identical compilable
# work, so retries fail identically; the FailedDecompositionCache cannot
# guard. The current entries vary:
#   * swap_induction_variable  -> single child, same footprint, perturbed
#                                  boundary (records an explicit induction
#                                  axis flip the compiler reads).
#   * split_conjunction_other_way -> TWO children splitting the parent_leaves
#                                    window in half (requires >= 2 leaves).
#   * strengthen_induction_hypothesis -> single child, footprint widened by
#                                        one fresh leaf (requires room to
#                                        grow).
# `HeuristicReviser.decompose` skips entries whose substrate precondition
# does not hold for the failing node and signals exhaustion by returning
# an empty list when nothing is feasible.
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

    def decompose(self, node: Node,
                  *, host_n_leaves: int | None = None) -> list[SubGoal]:
        """Cycle through the catalogue, returning the first SUBSTRATE-feasible
        permutation for this node. Returns ``[]`` when no remaining entry can
        produce a substrate-different decomposition (caller treats this as
        ``RevisionExhausted``).

        D24 fix: ``host_n_leaves`` (caps the maximum admissible leaf
        index, exclusive) is threaded through so
        ``strengthen_induction_hypothesis`` cannot append a leaf past
        the host MERA's footprint. When the cap is reached the strategy
        returns ``None`` here, the catalogue cursor advances, and
        exhaustion surfaces honestly as an empty list — rather than
        burning a revision attempt on a decomposition that
        ``integrate_child`` raises ``ValueError`` on (past D24, the
        exception leaked past ``RevisionExhausted``).
        """
        gid = node.goal.goal_id
        idx = self._cursor.get(gid, 0)
        for offset in range(len(_HEURISTIC_CATALOGUE)):
            strategy = _HEURISTIC_CATALOGUE[
                (idx + offset) % len(_HEURISTIC_CATALOGUE)
            ]
            alt = self._build(node, strategy, host_n_leaves=host_n_leaves)
            if alt is not None:
                self._cursor[gid] = idx + offset + 1
                return alt
        # No remaining catalogue entry is feasible -- exhausted.
        self._cursor[gid] = idx + len(_HEURISTIC_CATALOGUE)
        return []

    def _build(self, node: Node, strategy: str,
               *, host_n_leaves: int | None = None) -> list[SubGoal] | None:
        """Materialise ``strategy`` as a substrate-different decomposition.

        Returns ``None`` when the strategy's substrate precondition fails
        for this node (e.g., split needs >= 2 parent leaves). The cursor in
        ``decompose`` then advances to the next catalogue entry.
        """
        base_prop = node.goal.goal_prop
        leaves = node.goal.parent_leaves

        if strategy == "swap_induction_variable":
            # Single child, same footprint, but boundary records the
            # explicit axis flip so the compiler sees a different problem.
            spec = dict(node.goal.dsl_spec)
            spec["induction_axis"] = "flipped"
            boundary = dict(node.goal.boundary)
            boundary["induction_axis"] = "flipped"
            return [make_sub_goal(
                spec,
                goal_prop=f"{base_prop}::swap_induction_variable",
                boundary=boundary,
                parent_leaves=leaves,
            )]

        if strategy == "split_conjunction_other_way":
            # Two children covering disjoint halves of the parent footprint.
            # Substrate precondition: need at least 2 leaves to split.
            if len(leaves) < 2:
                return None
            mid = len(leaves) // 2
            left, right = leaves[:mid], leaves[mid:]
            spec = dict(node.goal.dsl_spec)
            return [
                make_sub_goal(
                    spec,
                    goal_prop=f"{base_prop}::split_lhs",
                    boundary=node.goal.boundary,
                    parent_leaves=left,
                ),
                make_sub_goal(
                    spec,
                    goal_prop=f"{base_prop}::split_rhs",
                    boundary=node.goal.boundary,
                    parent_leaves=right,
                ),
            ]

        if strategy == "strengthen_induction_hypothesis":
            # Single child whose footprint is widened by one fresh leaf
            # (next index after the current max). Substrate precondition:
            # the existing footprint must be non-empty (we need a max to
            # extend from). D24: refuse the strategy entirely when the
            # extended index would land *past* the host MERA's
            # ``n_leaves`` footprint — otherwise ``integrate_child``
            # raises ``ValueError`` downstream, leaking past
            # ``RevisionExhausted`` and burning a revision attempt on an
            # unrecoverable footprint. Returning ``None`` here lets the
            # catalogue cursor advance to the next strategy and (if all
            # exhausted) surfaces as a clean empty list to the caller.
            if not leaves:
                return None
            new_idx = max(leaves) + 1
            if host_n_leaves is not None and new_idx >= int(host_n_leaves):
                return None
            extended = tuple(leaves) + (new_idx,)
            spec = dict(node.goal.dsl_spec)
            return [make_sub_goal(
                spec,
                goal_prop=f"{base_prop}::strengthen_induction_hypothesis",
                boundary=node.goal.boundary,
                parent_leaves=extended,
            )]

        # Unknown strategy name -- treated as not feasible.
        return None


def revise(node: Node, *, llm: LLMReviser | None = None,
           reviser: HeuristicReviser | None = None,
           cache: FailedDecompositionCache | None = None,
           host_n_leaves: int | None = None) -> list[SubGoal]:
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

    # heuristic fallback; skip known-dead decompositions and substrate-
    # infeasible catalogue entries. ``decompose`` returns ``[]`` once the
    # cursor has cycled the full catalogue without finding a feasible
    # entry; the caller (Task 7) lifts that empty result into
    # ``RevisionExhausted``.
    alt: list[SubGoal] = []
    for _ in range(len(_HEURISTIC_CATALOGUE)):
        alt = reviser.decompose(node, host_n_leaves=host_n_leaves)
        if not alt:
            return []  # exhausted -- no remaining feasible variation
        if not cache.is_failed(node.goal.goal_id, alt):
            return alt
    return alt
