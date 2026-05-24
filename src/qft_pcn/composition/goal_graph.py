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
        # Internal nodes carry a synthetic _JointResult whose residual is
        # already the sum of their children's residuals (orchestrator
        # §6.3 joint). Counting both would double-count and break the
        # §9.5 monotonicity invariant whenever the tree has depth > 1.
        # Leaves are the source of truth -- only their residuals add.
        if not n.children:
            res = getattr(n.result, "residual_energy", None)
            if res is not None and res != float("inf"):
                accuracy += res
        for c in n.children:
            _walk(c)

    _walk(root)
    complexity = COMPLEXITY_WEIGHT * n_nodes
    return accuracy + complexity


# --- §9.5 monotonicity enforcement -----------------------------------------

# Default tolerance for F_hierarchy step-to-step comparison. The spec calls
# the increase "a bug in the integrator"; a strictly positive epsilon absorbs
# IEEE-754 noise on the per-step accuracy sum without papering over a real
# inflation.
F_MONOTONICITY_TOL = 1e-6


class MonotonicityViolation(GoalGraphError):
    """Raised when F_hierarchy strictly increases between two integration
    steps beyond ``F_MONOTONICITY_TOL`` (spec §9.5, §13.5). A non-monotone
    step is a bug in the integrator — see also DEVIATIONS.md D6.
    """


def make_monotonicity_tracker(
    *,
    strict: bool = True,
    tol: float = F_MONOTONICITY_TOL,
    sink=None,
):
    """Return an ``on_step``-shaped callable that enforces §9.5 monotonicity.

    The returned callable accepts a single ``F`` argument (the current
    ``compute_free_energy(root)`` value). On every invocation after the
    first, it raises :class:`MonotonicityViolation` if
    ``F_new > F_prev + tol``.

    Parameters
    ----------
    strict:
        Default ``True`` (assertion fires on violation). Production callers
        that want the orchestrator to keep running through a non-monotone
        step (e.g. for diagnostics replay) can pass ``strict=False`` to
        downgrade to a record-only walker. The default is ON so silent
        inflations cannot pass through tests.
    tol:
        Absolute tolerance; defaults to :data:`F_MONOTONICITY_TOL`.
    sink:
        Optional list-like ``append`` target — every observed ``F`` is
        recorded, preserving the historical "callers verify by inspecting
        the recorded values" contract from :func:`orchestrator.solve_goal_graph`.
    """
    history: list[float] = []

    def _track(F: float) -> None:
        if sink is not None:
            sink.append(F)
        history.append(float(F))
        if len(history) >= 2:
            prev, curr = history[-2], history[-1]
            if curr > prev + tol:
                if strict:
                    raise MonotonicityViolation(
                        f"F_hierarchy increased: step {len(history) - 2} "
                        f"-> {len(history) - 1}: {prev!r} -> {curr!r} "
                        f"(tol={tol!r}; §9.5 violation)"
                    )
                # strict=False: record only; caller assumes the diagnostic
                # responsibility documented above.
    return _track


# --- §6.6 quarantine-aware completion --------------------------------------

def all_solved(node: Node) -> bool:
    """True iff every non-quarantined child of ``node`` is SOLVED.

    Spec §6.6: "quarantine the failed sub-graph and continue exploring
    alternatives in parallel". A quarantined child is an out-of-band
    failure; it must not block the parent's completion when live siblings
    cover the proof. The predicate uses the same ``not c.quarantined``
    filter that the orchestrator's ``ready`` schedule uses, so the two
    surfaces honour quarantine consistently (DEVIATIONS.md D9).

    Returns ``False`` if ``node`` has no live children at all — a parent
    whose every child is quarantined has no covering proof and must
    revise. This matches the orchestrator's PENDING_REVISION fallback.
    """
    live = [c for c in node.children if not c.quarantined]
    if not live:
        return False
    return all(c.status == Status.SOLVED for c in live)


# --- proof tree (spec §4.6) ------------------------------------------------

@dataclass(frozen=True)
class ProofTreeNode:
    """A node in the verified proof tree (spec §4.6).

    ``bond_entanglement`` is the von-Neumann entropy across a canonical
    mid-network cut of the node's converged substrate state — a real
    operator-algebraic measurement of how much entanglement the proof
    step deposited at its bonds (spec §12.16 path fitness signal). The
    default ``0.0`` covers nodes built without a substrate ground state
    (test fixtures, synthetic joints), preserving backward compatibility.
    """
    goal_prop: str
    solved_ast: Any
    residual_energy: float
    children: tuple["ProofTreeNode", ...]
    bond_entanglement: float = 0.0


@dataclass(frozen=True)
class ProofTree:
    root: ProofTreeNode
    total_residual: float


def _bond_entanglement_of(result: Any) -> float:
    """Schmidt-spectrum entanglement entropy across a canonical mid-network
    cut of the node's converged ground state (spec §12.16).

    Reads ``result.ground_state`` (a MERA whose ``entanglement_entropy(cut)``
    surfaces the §5.8 substrate measurement) and evaluates at the mid-network
    cut ``cut = N // 2 - 1`` (centered, in-range for N >= 2). Returns ``0.0``
    when no substrate state is present (synthetic ``_JointResult`` for
    internal nodes; non-MERA backends; or a node whose runner did not return
    a ground_state). The fallback is silent on type — a ground_state without
    ``entanglement_entropy`` simply contributes zero, matching the §1.1
    architecture-soul invariant that bond-entanglement is a measurement, not
    a fabrication. The fallback is NOT a §1.1 shortcut: the field exists to
    carry real Schmidt-spectrum data when present.
    """
    gs = getattr(result, "ground_state", None)
    if gs is None:
        return 0.0
    fn = getattr(gs, "entanglement_entropy", None)
    if fn is None:
        return 0.0
    N = getattr(gs, "N", None)
    if not isinstance(N, int) or N < 2:
        return 0.0
    # Canonical mid-network cut: cut after leaf (N//2 - 1) splits the chain
    # roughly in half. entanglement_entropy requires 0 <= cut < N - 1.
    cut = max(0, min(N - 2, N // 2 - 1))
    # D10 fix (memory/no-placeholders loud-fail): narrow the except to
    # the documented `entanglement_entropy` contract failures (out-of-range
    # cut → IndexError, ill-shaped state → ValueError). A genuine substrate
    # fault (e.g. NotImplementedError, NumericalInstability,
    # broken-invariant AssertionError) is a §1.1 anti-shortcut violation
    # when silently swallowed: surface it so the proof-extraction caller
    # sees the substrate fault instead of an invisible zero in the §12.16
    # path-fitness signal.
    try:
        return float(fn(cut))
    except (IndexError, ValueError):
        # Documented contract failure for an out-of-range / ill-shaped
        # cut — the action term legitimately degrades to zero (no
        # Schmidt-spectrum data available at this slice). All other
        # exceptions propagate.
        return 0.0


def extract_proof_tree(root: Node) -> ProofTree:
    """Walk SOLVED nodes into a verified proof tree (spec §4.6).

    Each leaf node carries ``bond_entanglement`` derived from its converged
    substrate ground state via :func:`_bond_entanglement_of` (the §12.16
    Schmidt-spectrum path-fitness signal). Internal nodes (whose
    ``result`` is a synthetic ``_JointResult`` without a ground_state)
    contribute zero — they are joins, not substrate measurements.
    """
    total = 0.0

    def _build(n: Node) -> ProofTreeNode:
        nonlocal total
        res = getattr(n.result, "residual_energy", 0.0)
        ast = getattr(n.result, "solved_ast", None)
        ent = _bond_entanglement_of(n.result)
        total += res
        return ProofTreeNode(
            goal_prop=n.goal.goal_prop,
            solved_ast=ast,
            residual_energy=res,
            children=tuple(_build(c) for c in n.children
                           if c.status == Status.SOLVED),
            bond_entanglement=ent,
        )

    tree_root = _build(root)
    return ProofTree(root=tree_root, total_residual=total)
