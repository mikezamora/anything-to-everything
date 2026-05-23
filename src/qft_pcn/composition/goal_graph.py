"""Goal graph: the DAG of sub-goals (spec §4).

A parent QPCN's target goal is the root; sub-goals are children. Built
top-down, consumed bottom-up. goal_id is a content hash of
(goal_prop, dsl_spec) -- this is what makes cycle detection sound and
cross-sibling lemma sharing real.
"""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .errors import GoalGraphError


def _content_hash(goal_prop: str, dsl_spec: dict) -> str:
    """Stable content hash; dict key order does not matter."""
    canonical = json.dumps(
        {"goal_prop": goal_prop, "dsl_spec": dsl_spec},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class SubGoal:
    """A DSL spec plus its expected proposition type -- the unit a child
    QPCN run proves (spec §4.1).

    ``parent_leaves`` is the explicit host-leaf footprint the child lemma
    will occupy on its parent's MERA (spec §5.2a). It is a tuple of host
    leaf indices, not a single int, so non-contiguous / species-permuted
    layouts are first-class -- the integrator does not extend or guess
    the window. An empty tuple is the root-goal sentinel (no parent).
    """
    goal_id: str
    dsl_spec: dict
    goal_prop: str
    boundary: dict
    parent_leaves: tuple[int, ...] = ()

    # The dataclass is frozen but holds mutable (unhashable) dict fields, so
    # the auto-generated __hash__ would raise TypeError on use. goal_id IS
    # the content address of (goal_prop, dsl_spec), so equality and hashing
    # on goal_id alone are sound -- and downstream cycle detection / lemma
    # sharing needs SubGoal to live in sets and dicts.
    def __hash__(self) -> int:
        return hash(self.goal_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubGoal):
            return NotImplemented
        return self.goal_id == other.goal_id


def make_sub_goal(dsl_spec: dict, *, goal_prop: str, boundary: dict,
                  parent_leaves: tuple[int, ...] = ()) -> SubGoal:
    """Construct a SubGoal with a content-addressed goal_id.

    ``parent_leaves`` is the explicit host-leaf tuple the child lemma
    occupies. Callers with only a contiguous base+count should use
    :func:`make_contiguous_sub_goal`.

    Invariants on ``parent_leaves`` (raised loud, not silently coerced --
    caller-discipline failures must surface, per ``no-placeholders``):
      * Every element is a NON-NEGATIVE int.
      * Every element is UNIQUE (no duplicates).
    The element ORDER is significant (it is the entanglement-footprint
    order on the parent MERA, §1.1) and is preserved verbatim. The
    range check against ``parent_meta.n_nodes * LEAVES_PER_NODE`` is
    deferred to :func:`result_integrator.integrate_child`, where the
    parent meta is in scope.
    """
    normalized = tuple(int(i) for i in parent_leaves)
    if any(i < 0 for i in normalized):
        raise ValueError(
            f"SubGoal.parent_leaves must be non-negative ints; "
            f"got {normalized!r}"
        )
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            f"SubGoal.parent_leaves must contain unique leaf indices; "
            f"got {normalized!r}"
        )
    return SubGoal(
        goal_id=_content_hash(goal_prop, dsl_spec),
        dsl_spec=dsl_spec,
        goal_prop=goal_prop,
        boundary=dict(boundary),
        parent_leaves=normalized,
    )


def make_contiguous_sub_goal(dsl_spec: dict, *, goal_prop: str, boundary: dict,
                             base: int, n_leaves: int) -> SubGoal:
    """Convenience constructor for the contiguous-window case.

    Builds ``parent_leaves = (base, base+1, ..., base+n_leaves-1)``. This
    is the principled isomorphic-decomposer shape; non-isomorphic or
    species-permuted decomposers should call :func:`make_sub_goal`
    directly with the explicit tuple.
    """
    return make_sub_goal(
        dsl_spec, goal_prop=goal_prop, boundary=boundary,
        parent_leaves=tuple(range(int(base), int(base) + int(n_leaves))),
    )


class Status(enum.Enum):
    PENDING = "pending"                     # created, not yet dispatched
    ACTIVE = "active"                       # a child QPCN run is in flight
    SOLVED = "solved"                       # true ground state returned; integrated
    FAILED = "failed"                       # did not converge; revision exhausted
    CYCLE = "cycle"                         # already an ancestor under proof
    PENDING_REVISION = "pending_revision"   # child failed; awaiting alternative


@dataclass
class Node:
    goal: SubGoal
    status: Status
    children: list["Node"] = field(default_factory=list)
    result: Any = None                      # ChildResult once SOLVED/FAILED
    revision_attempts: int = 0
    parent: "Node | None" = None
    quarantined: bool = False               # spec §6.6

    def add_child(self, child: "Node") -> None:
        # Silent reparenting is a graph bug: a node should be added once,
        # to one parent. Surface the violation via GoalGraphError so it
        # cannot be lost in a stack trace.
        if child.parent is not None and child.parent is not self:
            raise GoalGraphError(
                f"reparent of node {child.goal.goal_id!r} attempted"
            )
        child.parent = self
        self.children.append(child)


# --- construction (spec §4.2) ----------------------------------------------

COMPLEXITY_WEIGHT = 1e-9   # complexity term coefficient in F_hierarchy (spec §7)


def build_goal_graph(root_spec: dict, root_prop: str, decomposer) -> Node:
    """Build the root and lazily expand only the root's immediate children.

    Children deeper than level 1 are populated on demand by expand_node /
    expand_fully -- the graph can be combinatorially large (spec §4.2).
    """
    root_goal = make_sub_goal(root_spec, goal_prop=root_prop,
                              boundary={}, parent_leaves=())
    root = Node(goal=root_goal, status=Status.PENDING)
    expand_node(root, decomposer)
    return root


def expand_node(node: Node, decomposer) -> None:
    """Populate node.children once, from the decomposer (idempotent)."""
    if node.children:
        return
    for sub in decomposer.decompose(node):
        node.add_child(Node(goal=sub, status=Status.PENDING))


def expand_fully(node: Node, decomposer) -> None:
    """Eagerly expand the whole subtree -- for tests / small graphs only."""
    expand_node(node, decomposer)
    for child in node.children:
        expand_fully(child, decomposer)


# --- cycle detection (spec §4.4) -------------------------------------------

def detect_cycle(node: Node, visited: "frozenset[str]") -> bool:
    """True iff this node's goal is already on the active path from the root.

    visited is PER ACTIVE PATH, not global: the same goal_id in two
    independent sibling subtrees is a shared lemma, not a cycle.
    """
    return node.goal.goal_id in visited


def assert_acyclic(root: Node) -> None:
    """DFS guard: a back-edge that slipped past detect_cycle is a bug."""
    on_path: set[int] = set()

    def _dfs(n: Node) -> None:
        if id(n) in on_path:
            raise GoalGraphError(f"back-edge at goal {n.goal.goal_id!r}")
        on_path.add(id(n))
        for c in n.children:
            _dfs(c)
        on_path.discard(id(n))

    _dfs(root)


# --- free energy (spec §7) -------------------------------------------------

def compute_free_energy(root: Node) -> float:
    """F_hierarchy = accuracy (sum of residual energies) + complexity.

    The same F = accuracy + complexity a single predictive-coding layer
    minimizes, instantiated at whole-QPCN scale (spec §7, §10.10).
    """
    accuracy = 0.0
    n_nodes = 0

    def _walk(n: Node) -> None:
        nonlocal accuracy, n_nodes
        n_nodes += 1
        res = getattr(n.result, "residual_energy", None)
        if res is not None and res != float("inf"):
            accuracy += res
        for c in n.children:
            _walk(c)

    _walk(root)
    complexity = COMPLEXITY_WEIGHT * n_nodes
    return accuracy + complexity


# --- proof tree (spec §4.6) ------------------------------------------------

@dataclass(frozen=True)
class ProofTreeNode:
    goal_prop: str
    solved_ast: Any
    residual_energy: float
    children: tuple["ProofTreeNode", ...]


@dataclass(frozen=True)
class ProofTree:
    root: ProofTreeNode
    total_residual: float


def extract_proof_tree(root: Node) -> ProofTree:
    """Walk SOLVED nodes into a verified proof tree (spec §4.6)."""
    total = 0.0

    def _build(n: Node) -> ProofTreeNode:
        nonlocal total
        res = getattr(n.result, "residual_energy", 0.0)
        ast = getattr(n.result, "solved_ast", None)
        total += res
        return ProofTreeNode(
            goal_prop=n.goal.goal_prop,
            solved_ast=ast,
            residual_energy=res,
            children=tuple(_build(c) for c in n.children
                           if c.status == Status.SOLVED),
        )

    tree_root = _build(root)
    return ProofTree(root=tree_root, total_residual=total)
