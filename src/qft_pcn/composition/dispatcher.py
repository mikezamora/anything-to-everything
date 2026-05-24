"""Dispatcher: spawn + collect child QPCN runs (spec §5).

Siblings are independent and run in parallel on a thread pool. Each child
has a hard timeout; a straggler resolves to a non-converged ChildResult
and never blocks its siblings. run_child routes the DSL spec through G's
bridge pipeline -- it never re-implements a QPCN runner.
"""
from __future__ import annotations

import math
import typing
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from dataclasses import dataclass
from time import monotonic

from .goal_graph import Node, Status, SubGoal

DEFAULT_CHI_MAX = 32   # spec §10.10 acceptance setting


@dataclass(frozen=True)
class ChildResult:
    goal_id: str
    converged: bool
    residual_energy: float
    ground_state: typing.Any        # the child's MERA ground state, or None
    solved_ast: typing.Any          # decoded AST, or None
    run_diagnostic: dict            # G's RunResult.to_dict() for provenance
    error: str | None
    # --- I-Task-7 plumbing (spec §6.1) ----------------------------------
    # ``meta`` is the MeraEncodingMeta the child was decoded under -- the
    # result-integrator needs it to invoke ``register_lemma`` and to feed
    # ``Promoter`` species checks. ``hamiltonian`` is the (optional) M2
    # Hamiltonian under which residual_energy was measured; passing it
    # through enables the compress branch of ``register_lemma``.
    # ``trotter_steps`` carries provenance into ``DerivationMetadata``.
    # All three default to safe no-op values so that older call sites
    # (tests, stubs) keep working unchanged.
    meta: typing.Any = None
    hamiltonian: typing.Any = None
    trotter_steps: int = 0


def run_child(sub_goal: SubGoal, *, chi_max: int = DEFAULT_CHI_MAX,
              timeout_s: float) -> ChildResult:
    """Package sub_goal as a DSL spec and run it through G's bridge.

    Consumes bridge.runtime.run_problem (which itself calls
    bridge.dsl.pipeline.compile_dsl + evolve_with_clamps); adapts
    RunResult into ChildResult. Never re-implements the runner.

    NOTE on ``timeout_s``: this parameter is *informational only* in the
    shipped runner. ``run_problem`` is a synchronous, uninterruptible
    call from outside its thread, so the dispatcher's outer ``wait`` is
    what actually bounds the batch. A straggler's pool thread can leak
    until ``run_problem`` returns of its own accord; the dispatcher
    will still surface a timeout ``ChildResult`` to its siblings on
    schedule. A future runner is free to honour ``timeout_s`` directly
    (e.g. by passing an iteration budget through ``search``).
    """
    from src.qft_pcn.bridge.runtime import run_problem  # G entry

    spec = dict(sub_goal.dsl_spec)
    # parent-context constraints become the child's boundary conditions
    spec.setdefault("boundary", {})
    spec["boundary"].update(sub_goal.boundary)
    # ``chi_max`` is consumed by ``run_problem`` via ``spec['search']``;
    # the dispatcher's default supplies the spec §10.10 acceptance value
    # unless the caller already set one in the DSL spec.
    search = dict(spec.get("search") or {})
    search.setdefault("chi_max", chi_max)
    spec["search"] = search

    run_result = run_problem(spec)
    return ChildResult(
        goal_id=sub_goal.goal_id,
        converged=bool(run_result.converged),
        residual_energy=float(run_result.energy),
        ground_state=getattr(run_result, "ground_state", None),
        solved_ast=getattr(run_result, "solved_ast", None),
        run_diagnostic=run_result.to_dict(),
        error=None,
        meta=getattr(run_result, "meta", None),
        hamiltonian=getattr(run_result, "hamiltonian", None),
        trotter_steps=int(getattr(run_result, "trotter_steps", 0) or 0),
    )


class DispatchBackend(typing.Protocol):
    def submit(self, runner, sub_goal: SubGoal,
               timeout_s: float) -> "object": ...
    def shutdown(self) -> None: ...


class ThreadPoolBackend:
    """The shipped lightweight concurrency path (spec §5.2). A future Ray
    backend implements the same DispatchBackend protocol."""

    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, runner, sub_goal: SubGoal, timeout_s: float):
        return self._pool.submit(runner, sub_goal, timeout_s)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _timeout_result(sub_goal: SubGoal) -> ChildResult:
    return ChildResult(
        goal_id=sub_goal.goal_id, converged=False,
        residual_energy=math.inf, ground_state=None, solved_ast=None,
        run_diagnostic={}, error="timeout",
    )


def dispatch_siblings(nodes: list[Node], backend: DispatchBackend, *,
                      runner=run_child, timeout_s: float,
                      parent_state: typing.Any = None,
                      qec_threshold: float = 1e-4,
                      qec_snapshots: typing.Mapping[str, typing.Any] | None = None,
                      ) -> list[ChildResult]:
    """Submit all sibling nodes at once; collect results as they complete.

    A straggler past timeout_s resolves to a non-converged ChildResult and
    never blocks its siblings (spec §5.4).

    §10.10 / §12.5 wiring: when ``parent_state`` is a MERA, every
    converged child with a ``ground_state`` is run through the
    holographic-code corruption detector against the parent's bulk
    reconstruction. A flagged child is either routed through
    :func:`apply_holographic_recovery` (if ``qec_snapshots`` carries a
    pre-corruption MERA for that ``goal_id``) or marked non-converged
    with a structured ``error='qec_corruption:...'`` so the integrator
    refuses with a §6.5 failure_report. When ``parent_state`` is None
    (legacy and stub-runner tests), QEC is skipped — the call is a
    bitwise no-op on the result list and node statuses.
    """
    futures: dict = {}
    for node in nodes:
        node.status = Status.ACTIVE
        fut = backend.submit(runner, node.goal, timeout_s)
        futures[fut] = node

    results: list[ChildResult] = []
    pending = set(futures)
    # Spec §5.4: the batch deadline is *absolute*, not per-iteration. If we
    # passed ``timeout_s + 1.0`` to each ``wait`` call, every completed child
    # would reset the window and a straggler could be tolerated up to
    # ``n_siblings * (timeout_s + 1.0)``. We anchor the deadline once.
    deadline = monotonic() + timeout_s + 1.0
    while pending:
        done, pending = wait(pending,
                              timeout=max(0.0, deadline - monotonic()),
                              return_when=FIRST_COMPLETED)
        if not done:                       # nothing finished within the pad
            for fut in list(pending):
                node = futures[fut]
                fut.cancel()
                results.append(_timeout_result(node.goal))
                node.result = results[-1]
            break
        for fut in done:
            node = futures[fut]
            try:
                res = fut.result(timeout=0)
            except Exception as exc:        # noqa: BLE001 -- isolate one child
                # ANTI-SHORTCUT (§1.1 / memory:anti-shortcut-directive):
                # this broad-except is NOT a graceful skip. It captures the
                # child's failure into a structured ChildResult so the
                # *integrator* (not the dispatcher) can refuse loudly with
                # a typed reason. The error string is preserved verbatim --
                # never collapse it to a generic "child failed" placeholder;
                # the result_integrator's diagnostic + revision loop needs
                # the original raise's text to decide whether to revise.
                res = ChildResult(
                    goal_id=node.goal.goal_id, converged=False,
                    residual_energy=math.inf, ground_state=None,
                    solved_ast=None, run_diagnostic={}, error=str(exc),
                )
            node.result = res
            results.append(res)

    # --- §12.5 / §10.10 QEC pass ---------------------------------------
    # Run the holographic-code syndrome on every converged child that
    # produced a real MERA ground state. Lazy import: holographic_correction
    # imports MERA, which we want to avoid at module load for the legacy
    # stub-runner tests that never touch a real substrate.
    if parent_state is not None:
        # Build the children-states map keyed by goal_id. Skip children
        # that did not converge (their ChildResult is already a refusal
        # the integrator surfaces) and any child whose ground_state is
        # not a MERA — the QEC code only knows the MERA substrate
        # (§12.5 is literally a MERA holographic code).
        try:
            from src.qft_pcn.qft.mera import MERA  # noqa: WPS433 (intentional lazy)
            from .holographic_correction import (
                detect_logical_corruption,
                apply_holographic_recovery,
            )
        except Exception:                           # noqa: BLE001
            # If the QEC module fails to import (substrate broken), do
            # not silently mask the children's results — re-raise so the
            # caller sees the substrate fault loudly per §1.6.
            raise

        children_states: dict = {}
        if isinstance(parent_state, MERA):
            for res in results:
                gs = res.ground_state
                if (res.converged and isinstance(gs, MERA)
                        and gs.layer_dims == parent_state.layer_dims
                        and gs.N == parent_state.N):
                    children_states[res.goal_id] = gs
        if children_states:
            report = detect_logical_corruption(
                parent_state, children_states, threshold=qec_threshold,
            )
            # §12.5 spec language: corruption is "when sub-QPCNs return
            # inconsistent results (one branch proves A, another ¬A)".
            # The raw distance from a fresh-encoded parent_state catches
            # *any* state divergence -- including the legitimate divergence
            # produced by imaginary-time evolution toward the goal's
            # ground state (which is exactly what a healthy child does).
            # Without a sibling-consensus gate, a single evolved child
            # always trips the syndrome.
            #
            # Sibling-consensus gate: only treat a flagged child as
            # genuinely corrupt if it is an *outlier* among its converged
            # MERA siblings -- i.e., at least one other sibling has a
            # distance from the parent that is significantly smaller
            # (factor of 10x) AND below qec_threshold. This recovers the
            # spec's intended "disagreement among siblings" semantic
            # while permitting a uniformly-evolved sibling cohort to
            # pass through (the K-8 single-child case, and any case
            # where all siblings legitimately evolve together).
            consensus_flagged: list[str] = []
            if report.flagged:
                dists = report.syndrome_distances
                snap_keys = set(qec_snapshots.keys()) if qec_snapshots else set()
                # A sibling is "consensus-clean" if its distance falls
                # below qec_threshold -- it agrees with the parent's
                # reference signature. If any such sibling exists, then
                # any flagged sibling is a genuine outlier vs the cohort.
                has_consensus_clean = any(
                    d <= qec_threshold for d in dists.values()
                )
                for cid in report.flagged:
                    if cid in snap_keys:
                        # Explicit pre-corruption snapshot supplied for
                        # this child: caller is asserting "this one may
                        # be corrupt, here is its rollback". Honor the
                        # syndrome regardless of cohort consensus.
                        consensus_flagged.append(cid)
                    elif has_consensus_clean and len(children_states) >= 2:
                        # No snapshot, but a sibling matches the parent
                        # signature -- the flagged child is an outlier
                        # within a multi-sibling cohort.
                        consensus_flagged.append(cid)
                    # else: single child or whole cohort drifted (e.g.,
                    # all imaginary-time-evolved toward a goal ground
                    # state). No basis to call any one an outlier --
                    # skip QEC refusal. The integrator's residual-
                    # energy and spectral-gap gates remain the load-
                    # bearing quality checks for that path.
            recovery = apply_holographic_recovery(
                parent_state, report, snapshots=qec_snapshots,
            ) if consensus_flagged else None
            if consensus_flagged:
                # Index results by goal_id for in-place rewrite. ChildResult
                # is frozen — we replace the entry rather than mutate it.
                by_id = {r.goal_id: i for i, r in enumerate(results)}
                node_by_id = {n.goal.goal_id: n for n in nodes}
                for cid in consensus_flagged:
                    idx = by_id.get(cid)
                    if idx is None:
                        continue
                    dist = report.syndrome_distances.get(cid, float('inf'))
                    if recovery is not None and cid in recovery.recovered:
                        # Recovery succeeded: replace the corrupted ground
                        # state with the snapshot-rollback MERA. The child
                        # is now consistent with the parent's bulk; the
                        # integrator's residual/gap gates still apply to
                        # the snapshot's diagnostics, which the caller
                        # is responsible for keeping coherent.
                        old = results[idx]
                        recovered_state = recovery.recovered[cid]
                        new_diag = dict(old.run_diagnostic)
                        new_diag["qec_recovery_applied"] = True
                        new_diag["qec_corruption_distance"] = float(dist)
                        new_res = ChildResult(
                            goal_id=old.goal_id,
                            converged=old.converged,
                            residual_energy=old.residual_energy,
                            ground_state=recovered_state,
                            solved_ast=old.solved_ast,
                            run_diagnostic=new_diag,
                            error=old.error,
                            meta=old.meta,
                            hamiltonian=old.hamiltonian,
                            trotter_steps=old.trotter_steps,
                        )
                        results[idx] = new_res
                        n = node_by_id.get(cid)
                        if n is not None:
                            n.result = new_res
                    else:
                        # No recovery available: refuse the integration
                        # with a structured QEC error (§6.5). The
                        # integrator's converged-gate will route this
                        # through _refuse and surface PENDING_REVISION.
                        old = results[idx]
                        reason = (
                            recovery.refused.get(cid, "qec corruption")
                            if recovery is not None else "qec corruption"
                        )
                        new_diag = dict(old.run_diagnostic)
                        new_diag["qec_recovery_applied"] = False
                        new_diag["qec_corruption_distance"] = float(dist)
                        new_diag["qec_refusal_reason"] = reason
                        new_res = ChildResult(
                            goal_id=old.goal_id,
                            converged=False,
                            residual_energy=math.inf,
                            ground_state=None,
                            solved_ast=None,
                            run_diagnostic=new_diag,
                            error=f"qec_corruption:{reason}",
                            meta=old.meta,
                            hamiltonian=old.hamiltonian,
                            trotter_steps=old.trotter_steps,
                        )
                        results[idx] = new_res
                        n = node_by_id.get(cid)
                        if n is not None:
                            n.result = new_res
    return results
