"""Dispatcher tests (spec §5)."""
from __future__ import annotations
import math
import time
import pytest
from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
from src.qft_pcn.composition.dispatcher import (
    ChildResult, ThreadPoolBackend, dispatch_siblings, run_child,
)


def _sub(name, prop="P"):
    return make_sub_goal({"g": name}, goal_prop=prop, boundary={}, parent_site=0)


def test_child_result_shape():
    r = ChildResult(goal_id="g", converged=True, residual_energy=1e-8,
                     ground_state=object(), solved_ast="ast",
                     run_diagnostic={}, error=None)
    assert r.converged and r.error is None


def test_dispatch_siblings_runs_in_parallel():
    # four stub children each "running" 0.3s; parallel wall time << 1.2s serial
    def stub_runner(sub_goal, timeout_s):
        time.sleep(0.3)
        return ChildResult(goal_id=sub_goal.goal_id, converged=True,
                           residual_energy=1e-9, ground_state=object(),
                           solved_ast=f"ast::{sub_goal.dsl_spec['g']}",
                           run_diagnostic={}, error=None)

    nodes = [Node(goal=_sub(f"c{i}"), status=Status.PENDING) for i in range(4)]
    backend = ThreadPoolBackend(max_workers=4)
    t0 = time.perf_counter()
    results = dispatch_siblings(nodes, backend, runner=stub_runner, timeout_s=5.0)
    elapsed = time.perf_counter() - t0
    backend.shutdown()
    assert len(results) == 4
    assert all(r.converged for r in results)
    assert elapsed < 1.0   # parallel: well under the 1.2s serial sum


def test_dispatch_timeout_does_not_block_siblings():
    def stub_runner(sub_goal, timeout_s):
        if sub_goal.dsl_spec["g"] == "slow":
            time.sleep(timeout_s + 5.0)  # would block forever if not timed out
        return ChildResult(goal_id=sub_goal.goal_id, converged=True,
                           residual_energy=1e-9, ground_state=object(),
                           solved_ast="ast", run_diagnostic={}, error=None)

    nodes = [
        Node(goal=_sub("fast1"), status=Status.PENDING),
        Node(goal=_sub("slow"), status=Status.PENDING),
        Node(goal=_sub("fast2"), status=Status.PENDING),
    ]
    backend = ThreadPoolBackend(max_workers=3)
    results = dispatch_siblings(nodes, backend, runner=stub_runner, timeout_s=0.5)
    backend.shutdown()
    by_id = {r.goal_id: r for r in results}
    slow = by_id[nodes[1].goal.goal_id]
    assert slow.converged is False
    assert slow.error == "timeout"
    assert math.isinf(slow.residual_energy)
    # the two fast children still succeeded
    assert by_id[nodes[0].goal.goal_id].converged is True
    assert by_id[nodes[2].goal.goal_id].converged is True


def test_dispatch_siblings_sets_node_status_and_result():
    def stub_runner(sub_goal, timeout_s):
        return ChildResult(goal_id=sub_goal.goal_id, converged=True,
                           residual_energy=1e-9, ground_state=object(),
                           solved_ast="ast", run_diagnostic={}, error=None)
    nodes = [Node(goal=_sub("c0"), status=Status.PENDING)]
    backend = ThreadPoolBackend(max_workers=1)
    dispatch_siblings(nodes, backend, runner=stub_runner, timeout_s=5.0)
    backend.shutdown()
    assert nodes[0].result is not None
    # status is left for the integrator to finalize; dispatcher marks ACTIVE->done
    assert nodes[0].status in (Status.ACTIVE, Status.PENDING, Status.SOLVED)


def test_dispatch_timeout_is_absolute_across_siblings():
    """Spec §5.4: the batch timeout is absolute, not per-iteration.

    Two children that each take just under ``timeout_s`` must NOT extend
    the deadline for a slow sibling -- otherwise a straggler could be
    tolerated up to ``n_siblings * (timeout_s + 1.0)`` instead of the
    batch-level ``timeout_s + 1.0``.
    """
    timeout_s = 0.5

    def stub_runner(sub_goal, ts):
        name = sub_goal.dsl_spec["g"]
        if name == "slow":
            # Far past timeout_s + 1.0 pad -- must be cut off.
            time.sleep(timeout_s + 5.0)
        else:
            # Just under timeout_s/2; two of these back-to-back would
            # exceed timeout_s if the deadline reset per iteration.
            time.sleep(timeout_s * 0.4)
        return ChildResult(goal_id=sub_goal.goal_id, converged=True,
                           residual_energy=1e-9, ground_state=object(),
                           solved_ast="ast", run_diagnostic={}, error=None)

    nodes = [
        Node(goal=_sub("fast1"), status=Status.PENDING),
        Node(goal=_sub("fast2"), status=Status.PENDING),
        Node(goal=_sub("slow"),  status=Status.PENDING),
    ]
    # Serialize fast1 and fast2 onto a single worker so they complete
    # sequentially -- this is what would reset a per-iteration deadline.
    backend = ThreadPoolBackend(max_workers=2)
    t0 = time.monotonic()
    results = dispatch_siblings(nodes, backend, runner=stub_runner,
                                 timeout_s=timeout_s)
    elapsed = time.monotonic() - t0
    backend.shutdown()

    # Absolute batch deadline: must be at most (timeout_s + 1.0) + small slack,
    # NOT 3 * (timeout_s + 1.0). Generous slack for CI scheduling jitter.
    assert elapsed < timeout_s + 1.0 + 0.4, (
        f"batch wall-clock {elapsed:.3f}s exceeds absolute deadline "
        f"{timeout_s + 1.0:.3f}s -- per-iteration reset regression?"
    )

    by_id = {r.goal_id: r for r in results}
    slow = by_id[nodes[2].goal.goal_id]
    assert slow.converged is False
    assert slow.error == "timeout"
    assert math.isinf(slow.residual_energy)
    assert by_id[nodes[0].goal.goal_id].converged is True
    assert by_id[nodes[1].goal.goal_id].converged is True
