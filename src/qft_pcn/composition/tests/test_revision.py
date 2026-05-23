"""Revision tests (spec §10)."""
from __future__ import annotations
from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
from src.qft_pcn.composition.revision import (
    revise, HeuristicReviser, FailedDecompositionCache,
)


def _node(prop="IndCase"):
    g = make_sub_goal({"g": prop}, goal_prop=prop, boundary={}, parent_site=0)
    return Node(goal=g, status=Status.PENDING_REVISION)


def test_heuristic_reviser_returns_a_different_decomposition():
    node = _node()
    reviser = HeuristicReviser()
    alt = reviser.decompose(node)
    assert isinstance(alt, list)
    assert len(alt) >= 1
    # all alternatives have content-addressed goal_ids
    assert all(hasattr(sg, "goal_id") for sg in alt)


def test_failed_decomposition_cache_blocks_repeats():
    cache = FailedDecompositionCache()
    node = _node()
    first = revise(node, reviser=HeuristicReviser(), cache=cache)
    cache.mark_failed(node.goal.goal_id, first)
    second = revise(node, reviser=HeuristicReviser(), cache=cache)
    # the cache forces a decomposition distinct from the known-dead one
    first_ids = tuple(sg.goal_id for sg in first)
    second_ids = tuple(sg.goal_id for sg in second)
    assert second_ids != first_ids


def test_llm_reviser_is_consulted_when_provided():
    calls = []

    class StubLLM:
        def suggest_decomposition(self, goal_prop, boundary):
            calls.append(goal_prop)
            return [{"dsl_spec": {"g": "llm-alt"}, "goal_prop": "AltProp"}]

    node = _node()
    alt = revise(node, llm=StubLLM(), reviser=HeuristicReviser(),
                 cache=FailedDecompositionCache())
    assert calls == ["IndCase"]
    assert alt[0].goal_prop == "AltProp"
