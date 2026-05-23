"""§12.14 Quantum walks for goal-graph search.

Continuous-time quantum walk (Farhi-Gutmann 1998) over the §10.10
dispatcher's goal-graph DAG. The search Hamiltonian is

    H = gamma * L  -  sum_{w in SOLVED} |w><w|

where ``L = D - A`` is the graph Laplacian (degree minus adjacency)
and the projector term marks SOLVED nodes as the oracle. Evolving
``|Psi(t)> = exp(-i H t) |Psi(0)>`` from a uniform leaf superposition
concentrates amplitude on the oracle-marked (SOLVED) nodes -- the
Farhi-Gutmann analogue of Grover amplification on a graph.

Architecture soul (§1.1). The Laplacian L is built from the
*operator-algebraic* topology of the goal graph -- the entanglement
bonds between parent and child nodes. It is NOT an AST walk, NOT a
structural fingerprint of dsl_spec, NOT a textual hash of
``goal_prop``. The basis index of a node is its identity under bond
entanglement (the goal-graph DAG induced by §4 cycle detection), and
the off-diagonal hops are the bonds. The oracle term ``|w><w|`` marks
SOLVED nodes by index, not by content hash. Bound-variable renames in
dsl_spec do not change H -- only the bond topology + Status do.

Cost: ``O(N + E)`` to build H (N nodes, E edges). A single Trotter
step costs one sparse matvec (``O(N + E)``). The Farhi-Gutmann
optimal evolution time is ``t ~ (pi/2) * sqrt(N)``; total walk cost
is ``O((N+E) * sqrt(N))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .goal_graph import Node, Status, SubGoal


# Default driver: a single walk step's dt is small enough that the
# exact ``expm`` is unitary to floating-point precision. For
# small graphs (||H|| <= max-degree + 1, typically <= 10)
# ``dt = 0.05`` keeps spectrum*dt well below the aliasing limit.
_DEFAULT_DT = 0.05

# Tolerance for Hermiticity / unitarity / norm-preservation
# invariants. 1e-12 is the operator-algebraic floor: anything bigger
# would mean the Laplacian-builder fell back on lossy floats, which
# would be a §1.1 violation (entanglement bonds are exact).
_HERMITICITY_TOL = 1e-12
_NORM_TOL = 1e-9


@dataclass(frozen=True)
class WalkHamiltonian:
    """The goal-graph search Hamiltonian (spec §12.14).

    ``matrix`` is the Hermitian ``N x N`` dense NumPy array
    (small N in the dispatcher's working window; the sparse form is
    an EXTENSIONS upgrade). ``nodes`` is the basis-index -> Node map:
    position ``i`` of any walker state refers to ``nodes[i]``.
    ``node_index`` is the inverse, indexed by Python ``id(node)`` so
    distinct Node objects that share a goal_id (shared lemmas in two
    independent subtrees) remain distinguishable in the walk basis.
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


def _build_laplacian(
    nodes: tuple[Node, ...],
    node_index: dict[int, int],
) -> np.ndarray:
    """Graph Laplacian L = D - A over the (undirected) goal-graph DAG.

    Every parent-child bond contributes ``-1`` on both ``(i, j)`` and
    ``(j, i)`` (the off-diagonal hop is symmetric so amplitude can
    flow back up the goal graph -- Farhi-Gutmann 1998 §III), and
    ``+1`` on both diagonal entries (degree count). The result is
    positive semi-definite, Hermitian, and ``L |1> = 0`` (uniform
    vector is the zero-mode of any graph Laplacian).
    """
    n = len(nodes)
    L = np.zeros((n, n), dtype=np.complex128)
    for node in nodes:
        i = node_index[id(node)]
        for child in node.children:
            j = node_index.get(id(child))
            if j is None:
                raise RuntimeError(
                    f"child {child.goal.goal_id!r} not in enumerated nodes; "
                    f"goal graph mutated mid-build"
                )
            # L = D - A: off-diagonal is negative, diagonal counts degree.
            L[i, j] -= 1.0
            L[j, i] -= 1.0
            L[i, i] += 1.0
            L[j, j] += 1.0
    return L


def build_walk_hamiltonian(
    root: Node,
    *,
    gamma: Optional[float] = None,
) -> WalkHamiltonian:
    """Build the Farhi-Gutmann oracle search Hamiltonian (spec §12.14).

        H = gamma * L  -  sum_{w in SOLVED} |w><w|

    where ``L = D - A`` is the graph Laplacian and the projector term
    marks SOLVED nodes as the oracle. ``gamma`` controls the
    Laplacian/oracle balance; we default to ``gamma = 1.0`` so the
    Laplacian hop-scale matches the oracle projector scale
    (``||L|| ~ max_degree ~ O(1)`` on the dispatcher's sparse goal
    graphs, and ``|| -|w><w| || = 1`` exactly). The asymptotic
    Farhi-Gutmann optimum on dense regular graphs is ``gamma = 1/N``,
    which is what an EXTENSIONS-tracked spectral-gap auto-tuner would
    pick once N grows past the small-graph regime. If no node is
    SOLVED the Hamiltonian degenerates to a pure free walk
    (``gamma * L``); the dispatcher should not invoke the search
    routine in that case (see ``walk_to_solved_subgoal``).

    The bond entries are *operator-algebraic* (entanglement bonds),
    not AST hashes: two goal-graphs that differ only in
    bound-variable names of their ``dsl_spec`` payload produce
    bitwise-identical H because the bond topology + node Status are
    identical.
    """
    nodes = tuple(_enumerate_nodes(root))
    n = len(nodes)
    if n == 0:
        raise ValueError("quantum walk over an empty goal graph")

    node_index = {id(node): i for i, node in enumerate(nodes)}

    if gamma is None:
        gamma = 1.0

    L = _build_laplacian(nodes, node_index)
    H = gamma * L

    # Oracle: subtract a rank-one projector |w><w| for each SOLVED node.
    # This drives the energy of SOLVED nodes below the Laplacian band,
    # producing Farhi-Gutmann amplitude concentration under exp(-iHt).
    for i, node in enumerate(nodes):
        if node.status == Status.SOLVED:
            H[i, i] -= 1.0

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
        dtype=np.float64,
    )
    total = leaf_mask.sum()
    if total == 0:
        leaf_mask[:] = 1.0
        total = float(n)
    return (leaf_mask / np.sqrt(total)).astype(np.complex128)


def quantum_walk_step(
    walk_state: np.ndarray,
    H_walk: WalkHamiltonian,
    dt: float = _DEFAULT_DT,
) -> np.ndarray:
    """One unitary walk step: ``|Psi> -> exp(-i H_walk dt) |Psi>``.

    Uses ``scipy.linalg.expm`` for exact matrix exponentiation --
    not Trotter, not Lanczos truncation -- because at the graph
    sizes the dispatcher operates on (tens of nodes per working
    window) the dense expm is cheaper and bitwise unitary. The
    Lanczos-exponentiation upgrade for large N is tracked in
    EXTENSIONS.md (it changes constants, not invariants).

    Norm preservation: ``|| exp(-i H dt) |Psi> || = || |Psi> ||`` to
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

    The default ``n_steps`` follows the Farhi-Gutmann optimal time
    ``t ~ (pi/2) * sqrt(N)`` for oracle-walk amplitude concentration,
    with ``dt`` from this module's constant.
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
        # Farhi-Gutmann optimal walk time t* = (pi/2) sqrt(N).
        t_star = 0.5 * np.pi * np.sqrt(H_walk.dim)
        n_steps = max(1, int(np.ceil(t_star / dt)))

    state = walk_state
    for _ in range(n_steps):
        state = quantum_walk_step(state, H_walk, dt)

    # Pick the SOLVED node with greatest occupation probability.
    probs = np.abs(state) ** 2
    best_local = int(np.argmax(probs[solved]))
    best_idx = solved[best_local]
    return H_walk.nodes[best_idx].goal
