"""Goal-graph data-structure tests (spec §4.1)."""
from __future__ import annotations
from src.qft_pcn.composition.goal_graph import SubGoal, Node, Status, make_sub_goal
from src.qft_pcn.composition.goal_graph import (
    build_goal_graph, detect_cycle, assert_acyclic, compute_free_energy,
    extract_proof_tree, ProofTree,
)
from src.qft_pcn.composition.errors import GoalGraphError


class StubDecomposer:
    """Maps a goal_prop to a fixed list of child (spec, prop) pairs."""
    def __init__(self, table):
        self._table = table  # dict: goal_prop -> list[(dsl_spec, goal_prop)]

    def decompose(self, node):
        from src.qft_pcn.composition.goal_graph import make_sub_goal
        out = []
        for i, (spec, prop) in enumerate(self._table.get(node.goal.goal_prop, [])):
            out.append(make_sub_goal(spec, goal_prop=prop, boundary={}, parent_site=i))
        return out


def test_build_goal_graph_three_levels():
    table = {
        "Thm":   [({"g": "ind"}, "IndCase")],
        "IndCase": [({"g": "L1a"}, "LemmaA"), ({"g": "L1b"}, "LemmaB")],
        "LemmaA": [({"g": "axA"}, "AxiomA")],
        "LemmaB": [({"g": "axB"}, "AxiomB")],
    }
    root = build_goal_graph({"g": "root"}, root_prop="Thm",
                            decomposer=StubDecomposer(table))
    # lazy: only the root expanded so far
    assert root.goal.goal_prop == "Thm"
    assert len(root.children) == 1
    # depth reached after full expansion
    from src.qft_pcn.composition.goal_graph import expand_fully
    expand_fully(root, StubDecomposer(table))
    assert root.children[0].goal.goal_prop == "IndCase"
    assert {c.goal.goal_prop for c in root.children[0].children} == {"LemmaA", "LemmaB"}
    assert_acyclic(root)  # does not raise


def test_detect_cycle_on_ancestor_goal_id():
    g = make_sub_goal({"g": "a"}, goal_prop="A", boundary={}, parent_site=0)
    node = Node(goal=g, status=Status.PENDING)
    assert detect_cycle(node, frozenset({g.goal_id})) is True
    assert detect_cycle(node, frozenset()) is False


def test_shared_goal_id_in_independent_siblings_is_not_a_cycle():
    # the SAME goal_id under two independent sibling subtrees -- a shared lemma
    g = make_sub_goal({"g": "shared"}, goal_prop="Shared", boundary={}, parent_site=0)
    left = Node(goal=g, status=Status.PENDING)
    right = Node(goal=g, status=Status.PENDING)
    # left's path visited set does not contain right's path -> no cycle
    assert detect_cycle(right, frozenset()) is False


def test_assert_acyclic_raises_on_back_edge():
    g0 = make_sub_goal({"g": "a"}, goal_prop="A", boundary={}, parent_site=None)
    n0 = Node(goal=g0, status=Status.PENDING)
    n0.children.append(n0)  # deliberate back-edge
    import pytest
    with pytest.raises(GoalGraphError):
        assert_acyclic(n0)


def test_compute_free_energy_sums_residuals():
    g0 = make_sub_goal({"g": "r"}, goal_prop="R", boundary={}, parent_site=None)
    g1 = make_sub_goal({"g": "c"}, goal_prop="C", boundary={}, parent_site=0)
    root = Node(goal=g0, status=Status.SOLVED)
    child = Node(goal=g1, status=Status.SOLVED)
    root.add_child(child)

    class _R:  # minimal ChildResult-shaped stub
        def __init__(self, r): self.residual_energy = r
    root.result = _R(0.0)
    child.result = _R(2e-7)
    f = compute_free_energy(root)
    # accuracy term = sum of residuals; complexity term >= 0
    assert f >= 2e-7


def test_extract_proof_tree_mirrors_solved_nodes():
    g0 = make_sub_goal({"g": "r"}, goal_prop="R", boundary={}, parent_site=None)
    g1 = make_sub_goal({"g": "c"}, goal_prop="C", boundary={}, parent_site=0)
    root = Node(goal=g0, status=Status.SOLVED)
    child = Node(goal=g1, status=Status.SOLVED)
    root.add_child(child)

    class _R:
        def __init__(self, r, ast): self.residual_energy = r; self.solved_ast = ast
    root.result = _R(0.0, "ast_root")
    child.result = _R(1e-7, "ast_child")
    tree = extract_proof_tree(root)
    assert isinstance(tree, ProofTree)
    assert tree.root.goal_prop == "R"
    assert tree.root.children[0].goal_prop == "C"
    assert tree.root.children[0].solved_ast == "ast_child"
    assert abs(tree.total_residual - 1e-7) < 1e-12


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


def test_subgoal_is_hashable():
    """SubGoal is frozen but contains dict fields -- the auto __hash__ would
    raise TypeError. An explicit __hash__ keyed on goal_id (the content
    address of goal_prop+dsl_spec) makes SubGoal usable in sets / dicts
    for cycle detection and cross-sibling lemma sharing."""
    g1 = make_sub_goal(_spec("a"), goal_prop="P", boundary={"x": 1}, parent_site=0)
    # hash() must not raise
    h1 = hash(g1)
    # two SubGoals with the same content (-> same goal_id) hash-equal and
    # compare-equal regardless of parent_site / boundary aliasing.
    g2 = make_sub_goal(_spec("a"), goal_prop="P", boundary={"x": 2}, parent_site=7)
    assert g1.goal_id == g2.goal_id
    assert hash(g1) == hash(g2) == h1
    assert g1 == g2
    # set / dict admission
    s = {g1, g2}
    assert len(s) == 1


def test_add_child_rejects_reparent():
    """Silent reparenting is a graph bug: surfaces as GoalGraphError."""
    import pytest as _pytest
    g_p1 = make_sub_goal(_spec("p1"), goal_prop="P", boundary={}, parent_site=None)
    g_p2 = make_sub_goal(_spec("p2"), goal_prop="Q", boundary={}, parent_site=None)
    g_c = make_sub_goal(_spec("c"), goal_prop="R", boundary={}, parent_site=0)
    parent1 = Node(goal=g_p1, status=Status.PENDING)
    parent2 = Node(goal=g_p2, status=Status.PENDING)
    child = Node(goal=g_c, status=Status.PENDING)
    parent1.add_child(child)
    assert child.parent is parent1
    with _pytest.raises(GoalGraphError):
        parent2.add_child(child)
    # child still belongs to parent1; parent2 did not pick it up
    assert child.parent is parent1
    assert child not in parent2.children
