"""Goal-graph data-structure tests (spec §4.1)."""
from __future__ import annotations
from src.qft_pcn.composition.goal_graph import SubGoal, Node, Status, make_sub_goal


def _spec(name):
    return {"goal": name, "fields": [], "hamiltonian": []}


def test_status_has_the_six_members():
    names = {s.name for s in Status}
    assert names == {
        "PENDING", "ACTIVE", "SOLVED", "FAILED", "CYCLE", "PENDING_REVISION"
    }


def test_sub_goal_is_frozen():
    g = make_sub_goal(_spec("a"), goal_prop="Nat", boundary={}, parent_site=0)
    import dataclasses
    assert dataclasses.is_dataclass(g)
    try:
        g.goal_prop = "Bool"
        assert False, "SubGoal must be frozen"
    except dataclasses.FrozenInstanceError:
        pass


def test_goal_id_is_content_addressed():
    g1 = make_sub_goal(_spec("a"), goal_prop="Nat", boundary={}, parent_site=0)
    g2 = make_sub_goal(_spec("a"), goal_prop="Nat", boundary={}, parent_site=7)
    g3 = make_sub_goal(_spec("b"), goal_prop="Nat", boundary={}, parent_site=0)
    # same (goal_prop, dsl_spec) -> same goal_id; parent_site does NOT affect it
    assert g1.goal_id == g2.goal_id
    # different dsl_spec -> different goal_id
    assert g1.goal_id != g3.goal_id


def test_goal_id_is_stable_across_dict_key_order():
    a = make_sub_goal({"x": 1, "y": 2}, goal_prop="P", boundary={}, parent_site=0)
    b = make_sub_goal({"y": 2, "x": 1}, goal_prop="P", boundary={}, parent_site=0)
    assert a.goal_id == b.goal_id


def test_node_defaults():
    g = make_sub_goal(_spec("a"), goal_prop="Nat", boundary={}, parent_site=0)
    n = Node(goal=g, status=Status.PENDING)
    assert n.children == []
    assert n.result is None
    assert n.revision_attempts == 0
    assert n.parent is None


def test_node_add_child_sets_back_edge():
    g0 = make_sub_goal(_spec("root"), goal_prop="P", boundary={}, parent_site=None)
    g1 = make_sub_goal(_spec("child"), goal_prop="Q", boundary={}, parent_site=3)
    root = Node(goal=g0, status=Status.PENDING)
    child = Node(goal=g1, status=Status.PENDING)
    root.add_child(child)
    assert root.children == [child]
    assert child.parent is root
