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
                      runner=run_child, timeout_s: float) -> list[ChildResult]:
    """Submit all sibling nodes at once; collect results as they complete.

    A straggler past timeout_s resolves to a non-converged ChildResult and
    never blocks its siblings (spec §5.4).
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
                res = ChildResult(
                    goal_id=node.goal.goal_id, converged=False,
                    residual_energy=math.inf, ground_state=None,
                    solved_ast=None, run_diagnostic={}, error=str(exc),
                )
            node.result = res
            results.append(res)
    return results
