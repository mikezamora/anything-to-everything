"""Orchestrator: the free-energy-minimizing search loop (spec §5.3, §6.5, §7).

Ties the composition modules together into a single search loop:

    expand -> cycle-check -> dispatch siblings in parallel
            -> gated integration -> revise on failure -> recompute F_hierarchy

The schedule orders the frontier with a structural proxy (boundary size +
child count); a real expected-ΔF estimator is a follow-on per spec §5.3.
Plain BFS is the easy fallback only.

Spec invariants enforced (per Task 7 in
``docs/superpowers/plans/2026-05-22-cross-level-passing-plan.md``):

* parallel sibling dispatch (§5.2) via :mod:`composition.dispatcher`,
* cycle detection per active path (§4.4) via :func:`goal_graph.detect_cycle`,
* strict residual / spectral-gap gate at integration (§6.3) via
  :func:`result_integrator.integrate_child` -- no fabricated proofs,
* ``MAX_REVISIONS`` then ``FAILED`` (§6.5) -- a sub-goal that exhausts its
  revisions raises :class:`errors.RevisionExhausted` internally; the root
  wrapper catches it and returns a structured ``failure_report`` rather than
  propagating an exception to callers (§6.5),
* monotone ``F_hierarchy`` along the schedule (§9.5) -- the caller observes
  every step via the optional ``on_step`` callback,
* structured failure report on root failure (§6.5).

This module ONLY orchestrates: it calls operator-algebraic surfaces (dispatch,
integrate, revise) and never performs AST splice itself (§1.5 / §1.6 hard
line).
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import RevisionExhausted
from .goal_graph import (
    Node,
    ProofTree,
    Status,
    assert_acyclic,
    build_goal_graph,
    compute_free_energy,
    detect_cycle,
    expand_node,
    extract_proof_tree,
)

MAX_REVISIONS = 3   # spec §6.5


@dataclass(frozen=True)
class SolveResult:
    """The outcome of :func:`solve_goal_graph`.

    Exactly one of ``proof_tree`` / ``failure_report`` is non-None. The
    invariant is enforced by construction in :func:`solve_goal_graph`.
    """
    solved: bool
    proof_tree: "ProofTree | None"
    failure_report: dict | None
    final_free_energy: float


class _JointResult:
    """ChildResult-shaped namespace for the synthetic 'joint of children'
    result attached to internal nodes whose children all SOLVED.

    Only the attributes that downstream consumers (``compute_free_energy``,
    ``extract_proof_tree``) read from ``ChildResult`` are populated; the
    rest are deliberately absent so a stray reader gets ``AttributeError``
    instead of a silently-wrong default.
    """

    __slots__ = ("residual_energy", "solved_ast")

    def __init__(self, *, residual_energy: float, solved_ast):
        self.residual_energy = residual_energy
        self.solved_ast = solved_ast


def _frontier_priority(node: Node) -> float:
    """Structural fan-out proxy for frontier ordering (spec §5.3).

    This is NOT an expected-ΔF estimate -- it returns ``len(boundary) +
    len(children)``, a deterministic structural fan-out signal that orders
    higher-coupled, more-expanded nodes first. A real expected-ΔF estimator
    (snapshot goal_graph, hypothetically decompose, compare
    :func:`compute_free_energy`) is a follow-on per spec §5.3. Plain BFS is
    the EASY fallback -- this proxy is the principled-but-cheap interim.
    """
    return float(len(node.goal.boundary) + len(node.children))


def solve_goal_graph(
    root_spec: dict,
    *,
    root_prop: str,
    decomposer,
    backend,
    lemma_library,
    parent_state,
    parent_meta,
    runner=None,
    timeout_s: float,
    on_step=None,
) -> SolveResult:
    """Drive the goal graph to a verified proof tree or a structured failure
    report. The schedule orders the frontier with a structural fan-out proxy
    (spec §7); a real expected-ΔF estimator is a follow-on per spec §5.3.

    Parameters
    ----------
    root_spec, root_prop:
        Root SubGoal spec and proposition type. The orchestrator builds the
        goal graph from these via :func:`goal_graph.build_goal_graph`.
    decomposer:
        Anything with ``decompose(node) -> list[SubGoal]`` -- consulted on
        every expansion, including post-revision re-expansion.
    backend:
        A :class:`dispatcher.DispatchBackend` (e.g.
        :class:`dispatcher.ThreadPoolBackend`) used for parallel sibling
        dispatch.
    lemma_library:
        A :class:`lemma_library.LemmaLibrary` used by the integrator to
        register / clamp converged children. A test double honouring the
        ``register_lemma`` + ``Promoter`` surfaces is acceptable.
    runner:
        Optional child-run callable; defaults to :func:`dispatcher.run_child`.
    timeout_s:
        Per-sibling-batch hard timeout (spec §5.4). A straggler resolves to
        a non-converged ChildResult inside ``dispatch_siblings``.
    on_step:
        Optional callback invoked with the current ``F_hierarchy`` after
        every integration step. Callers verify the §9.5 monotonicity
        invariant by inspecting the recorded values.
    parent_state, parent_meta:
        The parent QPCN's MERA state + encoding meta -- REQUIRED. The
        integrator clamps each converged child's lemma onto
        ``parent_state`` at the host-leaf window read directly from
        ``node.goal.parent_leaves`` and ``parent_meta.species_of_leaf``
        (spec §5.2a). The §1.1 architecture-soul binding IS this
        entanglement clamp; the orchestrator must drive it, not skip it
        -- a missing parent workspace raises ``TypeError`` rather than
        silently degrading to lemma-registration-only (anti-shortcut:
        no graceful skip; the clamp must fire).

    Returns
    -------
    SolveResult
        ``solved=True`` plus the extracted :class:`ProofTree` on success, or
        ``solved=False`` plus a structured ``failure_report`` (spec §6.5) on
        failure. Never returns a fabricated proof.
    """
    # Late imports keep the orchestrator's import-time cost off the hot path
    # of pure goal-graph callers.
    from .dispatcher import dispatch_siblings, run_child
    from .result_integrator import integrate_child
    from .revision import FailedDecompositionCache, HeuristicReviser, revise

    # §1.1 binding = entanglement clamp. The orchestrator is a solver over
    # an existing workspace; without a parent MERA the integrator cannot
    # fire ``Promoter.apply_init_clamp`` and the lemma is only logically
    # registered, not entanglement-clamped (a classical-lookup proof, not
    # a §1.1 proof). Refuse the call rather than silently degrade.
    if parent_state is None or parent_meta is None:
        raise TypeError(
            "solve_goal_graph requires a parent workspace: "
            "parent_state and parent_meta must both be non-None "
            "(§6.1 / §8.6 clamp is the orchestrator's load-bearing step)"
        )

    runner = runner or run_child
    root = build_goal_graph(root_spec, root_prop, decomposer)
    reviser, cache = HeuristicReviser(), FailedDecompositionCache()

    def _solve(node: Node, visited: "frozenset[str]") -> bool:
        # Spec §4.4: cycle is per active path; the same goal_id in two
        # independent sibling subtrees is a shared lemma, not a cycle.
        if detect_cycle(node, visited):
            node.status = Status.CYCLE
            return False
        visited = visited | {node.goal.goal_id}
        expand_node(node, decomposer)

        # MAX_REVISIONS + 1: one initial attempt plus MAX_REVISIONS retries.
        for _ in range(MAX_REVISIONS + 1):
            if node.children:
                # Spec §5.3: structural fan-out proxy orders the frontier;
                # see _frontier_priority. BFS is the easy fallback only.
                ready = sorted(
                    [c for c in node.children if not c.quarantined],
                    key=_frontier_priority,
                    reverse=True,
                )
                # Siblings that are themselves leaf goals batch together
                # for parallel dispatch (spec §5.2); internal-node children
                # recurse depth-first.
                leaf_siblings = [
                    c for c in ready
                    if not c.children and c.status == Status.PENDING
                ]
                if leaf_siblings:
                    # dispatch_siblings sets ``node.result`` on each input
                    # node directly; the returned results list is in
                    # completion order, so we integrate by walking the
                    # input list and reading each child's stored result.
                    dispatch_siblings(
                        leaf_siblings, backend,
                        runner=runner, timeout_s=timeout_s,
                    )
                    for child in leaf_siblings:
                        # integrate_child mutates child.status to SOLVED/FAILED.
                        integrate_child(
                            parent_state, parent_meta, child, child.result,
                            lemma_library,
                        )
                for c in ready:
                    if c.children or c.status == Status.PENDING:
                        ok = _solve(c, visited)
                        if not ok:
                            # Spec §6.6: quarantine a FAILED sub-graph so
                            # the rest of the proof can keep progressing.
                            c.quarantined = True
                all_solved = all(
                    c.status == Status.SOLVED for c in node.children
                )
                if all_solved:
                    node.status = Status.SOLVED
                    # The parent's residual is the joint of its children;
                    # an attribute-only namespace keeps the shape compatible
                    # with ChildResult-consuming utilities (compute_free_energy,
                    # extract_proof_tree) without dragging in a heavier type.
                    # Every SOLVED child has a result attached (invariant of
                    # integrate_child); a strict assertion is preferred over
                    # a silent fallback (anti-shortcut).
                    for c in node.children:
                        assert c.result is not None, (
                            f"SOLVED child {c.goal.goal_id} has no result"
                        )
                    node.result = _JointResult(
                        residual_energy=sum(
                            c.result.residual_energy for c in node.children
                        ),
                        solved_ast=tuple(
                            c.result.solved_ast for c in node.children
                        ),
                    )
                else:
                    node.status = Status.PENDING_REVISION
            else:
                # Leaf goal: dispatch a single child run.
                dispatch_siblings(
                    [node], backend, runner=runner, timeout_s=timeout_s,
                )
                # integrate_child mutates node.status to SOLVED/FAILED.
                integrate_child(
                    parent_state, parent_meta, node, node.result,
                    lemma_library,
                )

            if on_step is not None:
                on_step(compute_free_energy(root))
            if node.status == Status.SOLVED:
                return True

            # PENDING_REVISION (or FAILED at a leaf): try an alternative
            # decomposition. MAX_REVISIONS bounds the retries (spec §6.5).
            node.revision_attempts += 1
            if node.revision_attempts > MAX_REVISIONS:
                node.status = Status.FAILED
                # Spec §6.5: surface revision exhaustion as a typed error;
                # the outer wrapper catches it into a structured failure
                # report. Silent looping is forbidden (memory: anti-shortcut).
                raise RevisionExhausted(
                    goal_id=node.goal.goal_id,
                    attempts=node.revision_attempts,
                )
            alt = revise(node, reviser=reviser, cache=cache)
            cache.mark_failed(
                node.goal.goal_id,
                [c.goal for c in node.children],
            )
            # Detach the failed children before reparenting the alternative
            # decomposition's nodes (add_child raises GoalGraphError on a
            # silent reparent -- spec §4 hardening).
            for c in node.children:
                c.parent = None
            node.children = []
            for sg in alt:
                node.add_child(Node(goal=sg, status=Status.PENDING))

        # Loop fell through without solving and without raising. The
        # MAX_REVISIONS guard inside the loop body already handled
        # exhaustion (it raises RevisionExhausted), so this path is
        # unreachable: the for-range is MAX_REVISIONS + 1 >= 1.
        assert False, "unreachable: MAX_REVISIONS + 1 >= 1"

    try:
        solved = _solve(root, frozenset())
    except RevisionExhausted as exc:
        # Spec §6.5: revision exhausted at (or under) the root -- surface
        # as a structured failure_report, never a fabricated proof.
        f_final = compute_free_energy(root)
        return SolveResult(
            solved=False,
            proof_tree=None,
            failure_report={
                "root_status": root.status.value,
                "free_energy": f_final,
                "exhausted_goal_id": exc.goal_id,
                "revision_attempts": exc.attempts,
                "message": "the parent's plan was flawed; revision exhausted",
            },
            final_free_energy=f_final,
        )
    f_final = compute_free_energy(root)
    if solved:
        # A back-edge that slipped past detect_cycle is a graph bug; the
        # assertion turns it into a typed GoalGraphError rather than a
        # silent corrupt proof tree.
        assert_acyclic(root)
        return SolveResult(
            solved=True,
            proof_tree=extract_proof_tree(root),
            failure_report=None,
            final_free_energy=f_final,
        )
    return SolveResult(
        solved=False,
        proof_tree=None,
        failure_report={
            "root_status": root.status.value,
            "free_energy": f_final,
            "message": "root did not solve; no revision path remained",
        },
        final_free_energy=f_final,
    )
