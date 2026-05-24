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
from .worldline_pi import ProofRanking, bayesian_rank_proofs

MAX_REVISIONS = 3   # spec §6.5


@dataclass(frozen=True)
class SolveResult:
    """The outcome of :func:`solve_goal_graph`.

    Exactly one of ``proof_tree`` / ``failure_report`` is non-None. The
    invariant is enforced by construction in :func:`solve_goal_graph`.

    ``ranked_proofs`` is the §12.16 Bayesian-ranked list of candidate
    proofs (Boltzmann-weighted by ``exp(-S/T)``). When the orchestrator
    is called with ``n_top_k == 1`` (default) the list contains the
    single solved :class:`ProofTree` (weight ``1.0``) if any; on failure
    it is empty. ``n_top_k > 1`` is plumbed at the API surface; producing
    multiple distinct candidate proofs from a single solve requires
    additional revision-tracking and is deferred (see EXTENSIONS.md A.3).
    """
    solved: bool
    proof_tree: "ProofTree | None"
    failure_report: dict | None
    final_free_energy: float
    ranked_proofs: tuple[ProofRanking, ...] = ()


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
    """Precision-weighted ΔF estimate for frontier ordering (spec §5.3).

    D7 (DEVIATIONS.md): the previous body was a structural fan-out proxy
    (``len(boundary) + len(children)``) -- spec §5.3 mandates a precision-
    weighted estimate. This implementation composes the two operator-
    algebraic signals already attached to a Node when its result is
    available:

      precision := 1 / (1 + residual / RESIDUAL_SCALE)
                  -- the §6.2 precision weight, high for low-residual
                  children that carry near-ground-state messages.
      coupling  := 1 + bond_entanglement
                  -- the §12.16 Schmidt-spectrum signal across a canonical
                  mid-network cut of the substrate state. High entanglement
                  marks a node whose proof step deposited information at
                  its bonds -- the §5.3 ΔF-favoured frontier candidate.

      priority := precision * coupling + structural_fanout_tiebreak

    A node without a result (un-dispatched leaf / freshly-expanded
    internal) has no substrate measurement to feed the precision factor;
    we fall back to a DSL-derived structural fan-out signal (boundary +
    children count) scaled DOWN by 1e-3 so that any node with real
    measurements outranks the no-measurement fallback. This preserves the
    spec §5.3 contract that precision-weighted frontiers run before
    unmeasured ones, while keeping the schedule deterministic across the
    pre-dispatch warm-up.

    ANTI-SHORTCUT (§1.1 / memory:anti-shortcut-directive): bond_
    entanglement here is the entropy across the substrate state's
    canonical bond cut (via ``_bond_entanglement_of``) -- the SAME
    measurement used by :func:`compute_free_energy` and the §12.16 path-
    fitness signal. It is NOT a classical fan-out look-up.

    Full snapshot-and-compare ΔF estimator (clone goal_graph, virtually
    decompose, diff ``compute_free_energy``) is the heavier upgrade;
    that path is recorded in EXTENSIONS.md as a refinement, but the
    precision-weighted signal already matches the §5.3 ordering
    contract on every node that has been dispatched at least once.
    """
    from .goal_graph import _bond_entanglement_of
    from .result_integrator import RESIDUAL_SCALE
    res_obj = getattr(node, "result", None)
    if res_obj is not None and hasattr(res_obj, "residual_energy"):
        residual = float(getattr(res_obj, "residual_energy", float("inf")))
        if residual == float("inf"):
            precision = 0.0
        else:
            precision = 1.0 / (1.0 + residual / RESIDUAL_SCALE)
        coupling = 1.0 + _bond_entanglement_of(res_obj)
        # Tiny structural tiebreak so two equally-precise nodes still
        # admit a deterministic order; never large enough to dominate
        # the precision signal.
        tiebreak = 1e-6 * (len(node.goal.boundary) + len(node.children))
        return precision * coupling + tiebreak
    # No substrate measurement yet -- fall back to a scaled-DOWN
    # structural fan-out so any measured node outranks this node.
    return 1e-3 * (len(node.goal.boundary) + len(node.children))


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
    n_top_k: int = 1,
    ranking_temperature: float = 1.0,
    provisional_energy_fn=None,
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
    n_top_k:
        Number of candidate proofs to surface via Bayesian ranking
        (spec §12.16). The default ``1`` preserves prior behaviour:
        ``ranked_proofs`` carries the single solved proof at weight 1.0.
        ``n_top_k > 1`` is plumbed at the API surface but currently still
        yields the single solved tree — multi-candidate generation via
        revision-tracking is deferred (EXTENSIONS.md A.3 remainder).
    ranking_temperature:
        Boltzmann temperature ``T`` forwarded to
        :func:`composition.worldline_pi.bayesian_rank_proofs`. Only takes
        effect when more than one candidate is surfaced.
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
    if n_top_k < 1:
        raise ValueError(
            f"n_top_k must be >= 1 (the §12.16 top-k surface ranks at least "
            f"one proof); got n_top_k={n_top_k!r}"
        )

    runner = runner or run_child
    root = build_goal_graph(root_spec, root_prop, decomposer)
    reviser, cache = HeuristicReviser(), FailedDecompositionCache()

    # Spec §8 / §6.3 lazy re-evaluation: provisional (conditional=True)
    # lemmas registered by prior runs are re-checked against the caller-
    # supplied ``provisional_energy_fn`` BEFORE the solve loop begins, so
    # downstream goals never clamp a stale conjecture. The hook is a
    # no-op when no resolver is supplied (the caller is opting out of
    # cross-run re-eval); when supplied, promote/drop decisions are made
    # by :meth:`LemmaLibrary.re_evaluate_provisional`.
    if provisional_energy_fn is not None and hasattr(
        lemma_library, "re_evaluate_provisional"
    ):
        lemma_library.re_evaluate_provisional(provisional_energy_fn)

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
                # Eagerly expand each PENDING child one level so we can
                # classify it correctly: a sub-goal that decomposes further
                # is an INTERNAL node and must recurse, even though it has
                # no children at the moment of classification (build_goal_graph
                # only expands the root one level; deeper levels are lazy
                # per §4.2). Without this pre-expansion, an internal sub-
                # lemma like L1 (which decomposes to [A1]) is misclassified
                # as a leaf and dispatched directly -- collapsing the §10.11
                # multi-level hierarchy in extract_proof_tree.
                for c in ready:
                    if c.status == Status.PENDING and not c.children:
                        expand_node(c, decomposer)
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
                        # §12.5 / §10.10: route children through the
                        # holographic-code corruption detector against the
                        # parent's bulk reconstruction. parent_state may be
                        # None for non-MERA workspaces — dispatcher skips
                        # the QEC pass in that case.
                        parent_state=parent_state,
                        # D2 (§8): plumb library for pre-dispatch cache
                        # lookup; siblings re-proving the same sub-goal
                        # short-circuit to a synthesized SOLVED result.
                        lemma_library=lemma_library,
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
                    parent_state=parent_state,
                    # D2 (§8): cache lookup for the single-leaf path too.
                    lemma_library=lemma_library,
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
        proof_tree = extract_proof_tree(root)
        # §12.16 Bayesian ranking surface. With a single candidate the
        # ranking is trivially (proof_tree, weight=1.0) regardless of T.
        # Multi-candidate generation (n_top_k > 1) requires revision-
        # tracking and is deferred per EXTENSIONS.md A.3; we still
        # populate ``ranked_proofs`` so downstream callers can rely on
        # the API surface unconditionally.
        candidates = [proof_tree]
        ranked = tuple(
            bayesian_rank_proofs(candidates, T=ranking_temperature)
        )
        return SolveResult(
            solved=True,
            proof_tree=proof_tree,
            failure_report=None,
            final_free_energy=f_final,
            ranked_proofs=ranked,
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
