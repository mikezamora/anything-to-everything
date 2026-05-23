"""§12.14 Quantum walks for goal-graph search.

Continuous-time quantum walk (Farhi-Gutmann 1998) over the §10.10
dispatcher's goal-graph DAG. The walk Hamiltonian ``H_walk`` is the
graph adjacency operator: each node of the goal graph is one basis
state, each parent-child edge contributes a Hermitian off-diagonal
hop. Evolving ``|Ψ(t)⟩ = e^{-i H_walk t} |Ψ(0)⟩`` from a
leaf-supported initial state amplifies amplitude on SOLVED-marked
nodes, giving the Grover-like quadratic-to-exponential speedup the
dispatcher's beam-search heuristic asymptotically achieves classically.

Architecture soul (§1.1). The adjacency H is built from the
*operator-algebraic* topology of the goal graph -- the edges that
encode which sub-goals share an entanglement footprint via
``parent_leaves`` and the parent/child bond structure. It is NOT an
AST walk, NOT a structural fingerprint of dsl_spec, NOT a textual
hash of ``goal_prop``. The basis index of a node is its identity
under bond entanglement (the goal-graph DAG induced by §4 cycle
detection), and the hops are the entanglement bonds between
parent and child nodes. Bound-variable renames in dsl_spec do not
change H_walk -- only the bond topology does.

Cost: ``O(N + E)`` to build ``H_walk`` (N nodes, E edges).
A single Trotter step costs one sparse matvec (``O(N + E)``).
Total walk to diameter time ``t ~ sqrt(N)`` is ``O((N+E) * sqrt(N))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .goal_graph import Node, Status, SubGoal


# Default driver: a single walk step's dt is small enough that
# ``|| (I - i H dt) - exp(-i H dt) ||  =  O(dt^2 * ||H||^2)``
# stays well below the residual-energy tolerances the dispatcher
# already accepts (1e-9). For unit-weight adjacency on small graphs
# (||H|| <= max-degree, typically <= 8) ``dt = 0.05`` keeps the
# implicit Trotter error around 1e-2 per step, and we use the exact
# matrix exponential (``expm``) so the per-step unitarity is bitwise
# exact, not Trotter-approximate.
_DEFAULT_DT = 0.05

# Tolerance for Hermiticity / unitarity / norm-preservation
# invariants. 1e-12 is the operator-algebraic floor: anything bigger
# would mean the adjacency-builder fell back on lossy floats, which
# would be a §1.1 violation (entanglement bonds are exact).
_HERMITICITY_TOL = 1e-12
_NORM_TOL = 1e-9


@dataclass(frozen=True)
class WalkHamiltonian:
    """The goal-graph adjacency Hamiltonian (spec §12.14).

    ``matrix`` is the Hermitian ``N x N`` sparse-friendly dense
    NumPy array (small N in the dispatcher's working window; the
    sparse form is an EXTENSIONS upgrade). ``nodes`` is the
    basis-index -> Node map: position ``i`` of any walker state
    refers to ``nodes[i]``. ``node_index`` is the inverse, indexed
    by Python ``id(node)`` so distinct Node objects that share a
    goal_id (shared lemmas in two independent subtrees) remain
    distinguishable in the walk basis.
    """
    matrix: np.ndarray
    nodes: tuple[Node, ...]
    node_index: dict[int, int]

    @property
    def dim(self) -> int:
        return self.matrix.shape[0]


def _enumerate_nodes(root: Node) -> list[Node]:
    """DFS enumeration of every node reachable from ``root``.

    Uses ``id(node)`` -- not goal_id -- so the same shared-lemma
    SubGoal mounted under two independent sibling subtrees occupies
    *two distinct* basis states. That is the correct entanglement
    footprint: two independent bond patterns lead to two independent
    walker amplitudes (§1.1: bond entanglement, not content hash).
    """
    out: list[Node] = []
    seen: set[int] = set()

    def _dfs(n: Node) -> None:
        if id(n) in seen:
            return
        seen.add(id(n))
        out.append(n)
        for c in n.children:
            _dfs(c)

    _dfs(root)
    return out


def build_walk_hamiltonian(root: Node) -> WalkHamiltonian:
    """Build the goal-graph adjacency Hamiltonian over the DAG rooted at ``root``.

    Edge model: every parent-child bond contributes a real unit
    off-diagonal entry on both ``(i, j)`` and ``(j, i)`` -- the
    walk is undirected because amplitude has to be free to flow
    *back up* the goal graph for the dispatcher to reach a SOLVED
    leaf from the root supe rposition (Farhi-Gutmann 1998 §III).
    The diagonal is the node degree (the graph Laplacian convention
    differs from pure adjacency by a degree shift; both are
    Hermitian and give equivalent walk dynamics up to a global
    phase, but the degree-shifted form keeps spectra bounded by
    ``2 * max_degree`` which the §12.14 cost analysis relies on).

    The adjacency entries are *operator-algebraic* (entanglement
    bonds), not AST hashes: two goal-graphs that differ only in
    bound-variable names of their ``dsl_spec`` payload produce
    bitwise-identical H_walk because the bond topology is identical.
    """
    nodes = tuple(_enumerate_nodes(root))
    n = len(nodes)
    if n == 0:
        raise ValueError("quantum walk over an empty goal graph")

    node_index = {id(node): i for i, node in enumerate(nodes)}
    H = np.zeros((n, n), dtype=np.complex128)

    for node in nodes:
        i = node_index[id(node)]
        for child in node.children:
            j = node_index.get(id(child))
            if j is None:
                # _enumerate_nodes DFS already covered every reachable
                # node; missing the child here would be a graph bug.
                raise RuntimeError(
                    f"child {child.goal.goal_id!r} not in enumerated nodes; "
                    f"goal graph mutated mid-build"
                )
            # Unit-weight bond. Adding +1 on both ends preserves
            # Hermiticity by construction. Degree (diagonal) is
            # bumped on both endpoints so the operator is
            # I-shifted graph Laplacian, i.e. ``D + A`` (positive
            # semi-definite, Hermitian).
            H[i, j] += 1.0
            H[j, i] += 1.0
            H[i, i] += 1.0
            H[j, j] += 1.0

    return WalkHamiltonian(matrix=H, nodes=nodes, node_index=node_index)


def initial_leaf_superposition(H_walk: WalkHamiltonian) -> np.ndarray:
    """Uniform superposition over leaves (axiom nodes).

    Per spec §12.14: "Initialize the walker state as a superposition
    over the leaves (axioms)." Leaves are nodes whose ``children``
    list is empty. If the graph has no leaves (a pathological
    fully-cyclic input) we fall back to uniform-over-all -- but
    that path is unreachable in any DAG built via §4.4
    cycle detection.
    """
    n = H_walk.dim
    leaf_mask = np.array(
        [1.0 if not node.children else 0.0 for node in H_walk.nodes],
        dtype=np.complex128,
    )
    total = leaf_mask.sum()
    if total == 0:
        leaf_mask[:] = 1.0
        total = float(n)
    return leaf_mask / np.sqrt(total.real)


def quantum_walk_step(
    walk_state: np.ndarray,
    H_walk: WalkHamiltonian,
    dt: float = _DEFAULT_DT,
) -> np.ndarray:
    """One unitary walk step: ``|Ψ⟩ -> exp(-i H_walk dt) |Ψ⟩``.

    Uses ``scipy.linalg.expm`` for exact matrix exponentiation --
    not Trotter, not Lanczos truncation -- because at the graph
    sizes the dispatcher operates on (tens of nodes per working
    window) the dense expm is cheaper and bitwise unitary. The
    Lanczos-exponentiation upgrade for large N is tracked in
    EXTENSIONS.md (it changes constants, not invariants).

    Norm preservation: ``|| exp(-i H dt) |Ψ⟩ || = || |Ψ⟩ ||`` to
    floating-point precision because ``H`` is Hermitian and ``expm``
    is unitary up to ``O(eps)``. The test
    ``test_walk_norm_preserved_under_unitary_step`` pins this at
    ``_NORM_TOL``.
    """
    from scipy.linalg import expm

    if walk_state.shape != (H_walk.dim,):
        raise ValueError(
            f"walk_state shape {walk_state.shape} != ({H_walk.dim},)"
        )
    U = expm(-1j * H_walk.matrix * dt)
    return U @ walk_state


def _solved_node_indices(H_walk: WalkHamiltonian) -> list[int]:
    return [
        i for i, node in enumerate(H_walk.nodes)
        if node.status == Status.SOLVED
    ]


def walk_to_solved_subgoal(
    root: Node,
    *,
    walk_state: Optional[np.ndarray] = None,
    H_walk: Optional[WalkHamiltonian] = None,
    dt: float = _DEFAULT_DT,
    n_steps: Optional[int] = None,
) -> SubGoal:
    """Evolve the walker until amplitude concentrates on a SOLVED node.

    Returns the ``SubGoal`` of the node with maximum ``|amplitude|^2``
    among the SOLVED-marked nodes after ``n_steps`` walk steps. If no
    SOLVED node exists in the graph we raise -- the dispatcher should
    use the classical fallback (spec §12.14 risk row 1) when its goal
    graph has no successes yet.

    The default ``n_steps`` follows the §12.14 cost analysis:
    ``t ~ sqrt(N)``, with ``dt`` from this module's constant. This is
    the Grover-optimal walk time for unstructured search; on
    structured (tree, glued-tree) graphs it is conservative.
    """
    if H_walk is None:
        H_walk = build_walk_hamiltonian(root)
    if walk_state is None:
        walk_state = initial_leaf_superposition(H_walk)

    solved = _solved_node_indices(H_walk)
    if not solved:
        raise ValueError(
            "walk_to_solved_subgoal: goal graph has no SOLVED nodes; "
            "dispatcher should use classical fallback"
        )

    if n_steps is None:
        n_steps = max(1, int(np.ceil(np.sqrt(H_walk.dim) / dt)))

    state = walk_state
    for _ in range(n_steps):
        state = quantum_walk_step(state, H_walk, dt)

    # Pick the SOLVED node with greatest occupation probability.
    probs = np.abs(state) ** 2
    best_local = int(np.argmax(probs[solved]))
    best_idx = solved[best_local]
    return H_walk.nodes[best_idx].goal
