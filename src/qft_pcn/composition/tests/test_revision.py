"""Revision tests (spec §10)."""
from __future__ import annotations
from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
from src.qft_pcn.composition.revision import (
    revise, HeuristicReviser, FailedDecompositionCache,
)


def _node(prop="IndCase"):
    g = make_sub_goal({"g": prop}, goal_prop=prop, boundary={}, parent_leaves=(0,))
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


def test_llm_reviser_branch_fires_when_supplied_via_orchestrator():
    """D4: ``solve_goal_graph`` must forward ``llm_reviser`` so the
    principled LLM branch in ``revision.revise`` is reachable.

    We exercise ``revise`` with the same call shape ``solve_goal_graph``
    uses (``llm=llm_reviser``, plus reviser+cache), and assert the stub
    LLM's ``suggest_decomposition`` is consulted. The orchestrator-level
    wiring is checked by inspecting the function signature for the new
    kwarg -- a full end-to-end orchestrator run requires a parent MERA
    workspace (out of scope for this unit test).
    """
    import inspect
    from src.qft_pcn.composition.orchestrator import solve_goal_graph
    assert "llm_reviser" in inspect.signature(solve_goal_graph).parameters

    calls = []

    class StubLLM:
        def suggest_decomposition(self, goal_prop, boundary):
            calls.append((goal_prop, dict(boundary)))
            return [
                {"dsl_spec": {"g": "llm-a"}, "goal_prop": "AltA"},
                {"dsl_spec": {"g": "llm-b"}, "goal_prop": "AltB"},
            ]

    node = _node()
    stub = StubLLM()
    alt = revise(node, llm=stub, reviser=HeuristicReviser(),
                 cache=FailedDecompositionCache())
    assert calls, "stub LLMReviser.suggest_decomposition was never called"
    assert [sg.goal_prop for sg in alt] == ["AltA", "AltB"]


def test_heuristic_reviser_returns_substrate_different_decompositions():
    """D8: successive ``HeuristicReviser.decompose`` calls must differ in
    actual substrate -- parent_leaves footprint, sub-goal count, or
    boundary content -- NOT just in a metadata ``revision_strategy`` tag.
    Pure-tag variation is dead work: the downstream compiler is free to
    ignore the tag, so retries fail identically.
    """
    # Wide enough footprint that both the split (>=2 leaves) and the
    # strengthen (non-empty) strategies are substrate-feasible.
    g = make_sub_goal({"g": "P"}, goal_prop="P", boundary={},
                      parent_leaves=(0, 1))
    node = Node(goal=g, status=Status.PENDING_REVISION)
    reviser = HeuristicReviser()

    first = reviser.decompose(node)
    second = reviser.decompose(node)
    third = reviser.decompose(node)
    fourth = reviser.decompose(node)

    # Each non-empty decomposition is substrate-different from the next.
    def substrate(alt):
        return tuple(
            (sg.parent_leaves, sg.goal_prop, tuple(sorted(sg.boundary.items())))
            for sg in alt
        )

    assert first and second and third, (
        "first three catalogue entries must all be feasible for a "
        "2-leaf node (swap / split / strengthen)"
    )
    assert substrate(first) != substrate(second)
    assert substrate(second) != substrate(third)
    assert substrate(first) != substrate(third)

    # Specifically: the split entry must yield TWO sub-goals (sub-goal
    # count is a substrate-level difference, not a tag), and the
    # strengthen entry must yield a parent_leaves tuple LONGER than the
    # parent's (footprint widening is a substrate-level difference).
    counts = {len(first), len(second), len(third)}
    assert 2 in counts, "split_conjunction_other_way must emit two sub-goals"
    widened = [alt for alt in (first, second, third)
               if alt and len(alt) == 1 and len(alt[0].parent_leaves) > 2]
    assert widened, (
        "strengthen_induction_hypothesis must widen parent_leaves beyond "
        "the parent footprint"
    )

    # After the catalogue is exhausted for this node, decompose returns
    # ``[]`` rather than re-emitting identical work.
    assert fourth == []


def test_heuristic_reviser_signals_exhausted_when_no_variation_possible():
    """D8: if the parent has no substrate room to vary (empty
    ``parent_leaves``), the catalogue's split/strengthen entries are
    infeasible, and the reviser exhausts after the single feasible entry
    (swap_induction_variable) rather than looping on identical work.

    NB: ``make_sub_goal`` permits an empty ``parent_leaves`` tuple, which
    is the substrate-empty case the spec explicitly tolerates for
    boundary-only revisions.
    """
    g = make_sub_goal({"g": "P"}, goal_prop="P", boundary={},
                      parent_leaves=())
    node = Node(goal=g, status=Status.PENDING_REVISION)
    reviser = HeuristicReviser()
    first = reviser.decompose(node)
    # swap_induction_variable is feasible even on empty footprint
    # (single child, perturbed boundary).
    assert len(first) == 1
    assert first[0].boundary.get("induction_axis") == "flipped"
    # split needs >= 2 leaves; strengthen needs non-empty -- both
    # infeasible, so the second call must signal exhaustion ([]).
    second = reviser.decompose(node)
    assert second == []
