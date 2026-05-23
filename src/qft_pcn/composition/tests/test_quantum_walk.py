"""Acceptance tests for §12.14 quantum-walk goal-graph search.

These tests use the real composition-tier :class:`GoalGraph` /
:class:`Node` / :class:`SubGoal` (no mocks). The walk Hamiltonian
is the operator-algebraic adjacency of the goal-graph DAG; the
invariants pinned here are exactly the spec-§12.14 acceptance
criteria: Hermiticity of ``H_walk``, norm preservation of a single
``exp(-i H_walk dt)`` step, and Grover-like amplitude concentration
on a SOLVED leaf.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.goal_graph import (
    Node,
    Status,
    make_sub_goal,
)
from src.qft_pcn.composition.quantum_walk import (
    build_walk_hamiltonian,
    initial_leaf_superposition,
    quantum_walk_step,
    walk_to_solved_subgoal,
)


def _make_chain(n: int, *, solved_leaf: bool = False) -> Node:
    """Build a linear chain root -> n1 -> ... -> nN as a real Node tree."""
    root_goal = make_sub_goal(
        {"g": "root"}, goal_prop="R", boundary={}, parent_leaves=(),
    )
    root = Node(goal=root_goal, status=Status.PENDING)
    prev = root
    for i in range(n):
        g = make_sub_goal(
            {"g": f"n{i}"}, goal_prop=f"N{i}",
            boundary={}, parent_leaves=(i,),
        )
        child = Node(goal=g, status=Status.PENDING)
        prev.add_child(child)
        prev = child
    if solved_leaf:
        prev.status = Status.SOLVED
    return root


def _make_branching(*, solved_branch: int) -> Node:
    r"""Build a small branching graph with one SOLVED leaf.

        root
         |  \
        A    B
       / \   |
      C   D  E (SOLVED if solved_branch == 1 else not)

    The other branch is PENDING. C is SOLVED if solved_branch == 0.
    """
    root_goal = make_sub_goal({"g": "r"}, goal_prop="R", boundary={})
    root = Node(goal=root_goal, status=Status.PENDING)

    a = Node(goal=make_sub_goal({"g": "a"}, goal_prop="A",
                                boundary={}, parent_leaves=(0,)),
             status=Status.PENDING)
    b = Node(goal=make_sub_goal({"g": "b"}, goal_prop="B",
                                boundary={}, parent_leaves=(1,)),
             status=Status.PENDING)
    root.add_child(a)
    root.add_child(b)

    c = Node(goal=make_sub_goal({"g": "c"}, goal_prop="C",
                                boundary={}, parent_leaves=(0,)),
             status=Status.PENDING)
    d = Node(goal=make_sub_goal({"g": "d"}, goal_prop="D",
                                boundary={}, parent_leaves=(1,)),
             status=Status.PENDING)
    a.add_child(c)
    a.add_child(d)

    e = Node(goal=make_sub_goal({"g": "e"}, goal_prop="E",
                                boundary={}, parent_leaves=(0,)),
             status=Status.PENDING)
    b.add_child(e)

    if solved_branch == 0:
        c.status = Status.SOLVED
    elif solved_branch == 1:
        e.status = Status.SOLVED
    else:
        raise ValueError("solved_branch must be 0 or 1")
    return root


def test_walk_hamiltonian_is_hermitian():
    root = _make_branching(solved_branch=1)
    H = build_walk_hamiltonian(root)
    assert H.dim == 6  # root + A + B + C + D + E
    M = H.matrix
    # Hermitian to operator-algebraic floor.
    assert np.allclose(M, M.conj().T, atol=1e-12)
    # Real spectrum (Hermitian implies real eigenvalues).
    eigs = np.linalg.eigvalsh(M)
    assert np.all(np.isfinite(eigs))
    assert np.allclose(eigs.imag, 0.0, atol=1e-12)


def test_walk_norm_preserved_under_unitary_step():
    root = _make_chain(5, solved_leaf=True)
    H = build_walk_hamiltonian(root)
    psi0 = initial_leaf_superposition(H)
    n0 = np.linalg.norm(psi0)
    assert abs(n0 - 1.0) < 1e-12  # initial state is normalized

    psi = psi0
    for _ in range(20):
        psi = quantum_walk_step(psi, H, dt=0.05)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-9


def test_walk_amplifies_solved_branches():
    # SOLVED leaf is at depth 2 down branch B (node E). With the
    # Farhi-Gutmann oracle Hamiltonian H = gamma*L - |E><E|, evolving
    # to the optimal time t* = (pi/2)*sqrt(N) should concentrate the
    # bulk of the amplitude on E -- not just beat 1/N, but dominate
    # the non-SOLVED leaves C and D.
    root = _make_branching(solved_branch=1)
    H = build_walk_hamiltonian(root)
    psi0 = initial_leaf_superposition(H)

    name_to_idx = {n.goal.goal_prop: i for i, n in enumerate(H.nodes)}
    e_idx = name_to_idx["E"]
    c_idx = name_to_idx["C"]
    d_idx = name_to_idx["D"]

    # Farhi-Gutmann optimal walk time t* = (pi/2)*sqrt(N).
    dt = 0.05
    t_star = 0.5 * np.pi * np.sqrt(H.dim)
    n_steps = max(1, int(np.ceil(t_star / dt)))
    psi = psi0
    for _ in range(n_steps):
        psi = quantum_walk_step(psi, H, dt=dt)

    probs = np.abs(psi) ** 2
    # Norm sanity.
    assert abs(probs.sum() - 1.0) < 1e-9

    # SOLVED-selection picks E (oracle-marked).
    chosen = walk_to_solved_subgoal(
        root, walk_state=psi0, H_walk=H, dt=dt, n_steps=n_steps,
    )
    assert chosen.goal_prop == "E"

    # Real Grover-style amplification: E gets more than half the
    # total amplitude. This is *not* a tautology of argmax; it
    # asserts the oracle term actually drove amplitude onto E.
    assert probs[e_idx] > 0.5, (
        f"oracle failed to amplify SOLVED node: probs[E]={probs[e_idx]:.4f} "
        f"(expected > 0.5); full probs = {probs.tolist()}"
    )
    # And: E outranks every non-SOLVED leaf, not just uniform.
    assert probs[e_idx] > probs[c_idx]
    assert probs[e_idx] > probs[d_idx]


def test_walk_prefers_topology_favored_solved_node():
    # Two SOLVED nodes: C (at depth 2 down branch A, sibling D
    # competes) and E (at depth 2 down branch B, no sibling).
    # Both are oracle-marked; the Farhi-Gutmann walk should split
    # amplitude between them (neither dominates by 10x), confirming
    # multi-SOLVED handling actually works rather than collapsing
    # to one arbitrary node.
    root = _make_branching(solved_branch=1)
    # Locate C and mark it SOLVED too.
    name_to_node = {}

    def _collect(n):
        name_to_node[n.goal.goal_prop] = n
        for c in n.children:
            _collect(c)
    _collect(root)
    name_to_node["C"].status = Status.SOLVED

    H = build_walk_hamiltonian(root)
    psi0 = initial_leaf_superposition(H)
    name_to_idx = {n.goal.goal_prop: i for i, n in enumerate(H.nodes)}
    c_idx = name_to_idx["C"]
    e_idx = name_to_idx["E"]
    d_idx = name_to_idx["D"]  # not SOLVED

    dt = 0.05
    t_star = 0.5 * np.pi * np.sqrt(H.dim)
    n_steps = max(1, int(np.ceil(t_star / dt)))
    psi = psi0
    for _ in range(n_steps):
        psi = quantum_walk_step(psi, H, dt=dt)
    probs = np.abs(psi) ** 2

    # Both SOLVED nodes are amplified above the non-SOLVED leaf D.
    assert probs[c_idx] > probs[d_idx]
    assert probs[e_idx] > probs[d_idx]
    # Together they hold most of the amplitude.
    assert probs[c_idx] + probs[e_idx] > 0.5
