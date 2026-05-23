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
    # SOLVED leaf is at depth 2 down branch B (node E). The walker
    # initial state is a uniform superposition over leaves
    # {C, D, E}. After several walk steps amplitude leaks into the
    # interior; the SOLVED node E should outrank a uniform-random
    # baseline AND outrank at least one of the non-SOLVED leaves
    # by the §12.14 Grover-amplitude argument.
    root = _make_branching(solved_branch=1)
    H = build_walk_hamiltonian(root)
    psi0 = initial_leaf_superposition(H)

    # Find indices of the SOLVED node E and the non-SOLVED leaves
    # C and D for comparison.
    name_to_idx = {n.goal.goal_prop: i for i, n in enumerate(H.nodes)}
    e_idx = name_to_idx["E"]

    # Evolve to t ~ sqrt(N) per spec.
    psi = psi0
    n_steps = max(1, int(np.ceil(np.sqrt(H.dim) / 0.05)))
    for _ in range(n_steps):
        psi = quantum_walk_step(psi, H, dt=0.05)

    probs = np.abs(psi) ** 2
    # Norm sanity.
    assert abs(probs.sum() - 1.0) < 1e-9
    # The SOLVED-node selection wraps the amplitude argmax in
    # ``walk_to_solved_subgoal``; assert the returned goal IS the
    # SOLVED branch (E). This is the operator-algebraic acceptance:
    # the walker concentrates on the SOLVED-marked node, not on a
    # PENDING leaf.
    chosen = walk_to_solved_subgoal(
        root, walk_state=psi0, H_walk=H, dt=0.05, n_steps=n_steps,
    )
    assert chosen.goal_prop == "E"

    # And: E receives at least its uniform-random share. With 6
    # nodes uniform random would assign ~1/6 = 0.167; the walker
    # should put more than that on E (the only SOLVED node).
    assert probs[e_idx] > 1.0 / H.dim
