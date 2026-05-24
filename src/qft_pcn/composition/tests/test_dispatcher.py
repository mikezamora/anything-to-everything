"""Dispatcher tests (spec §5)."""
from __future__ import annotations
import math
import time
import pytest
from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
from src.qft_pcn.composition.dispatcher import (
    ChildResult, ThreadPoolBackend, dispatch_siblings, run_child,
)
from src.qft_pcn.bridge.runtime.hamiltonian import BridgeHamiltonian


def _sub(name, prop="P"):
    return make_sub_goal({"g": name}, goal_prop=prop, boundary={}, parent_leaves=(0,))


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


@pytest.fixture
def _no_large_dense():
    """Override conftest's §9.7 MERA-memory guard for the real-bridge test.

    The bridge MPS path legitimately allocates 16x16+ matrices for
    imaginary-time evolution (cutoff=4, 2 sites -> d=16 product space);
    that ceiling is a MERA-side invariant and does not apply here.
    """
    yield


def test_run_child_invokes_real_bridge_pipeline(_no_large_dense):
    """run_child must go through bridge.runtime.run_problem end-to-end.

    Locks in the import-correctness invariant the stub-runner tests above
    cannot exercise: any future regression that breaks the live bridge
    wiring (e.g. another missing symbol) will fail here loudly.
    """
    # A simple boundary-pinned problem the bridge converges quickly;
    # mirrors the converged-flag fixture in test_bridge_runtime_run.py.
    dsl_spec = {
        "fields": [{"name": "x", "cutoff": 4}],
        "sites": 2,
        "constraints": [
            {"kind": "local", "site": 0, "term": "x == 0", "weight": 5.0},
        ],
        "observables": [{"site": 0, "field": "x", "op": "n"}],
        "search": {"method": "imag_time", "steps": 50, "chi_max": 4,
                   "dt": 0.05},
    }
    sub_goal = make_sub_goal(dsl_spec, goal_prop="X_pinned",
                              boundary={}, parent_leaves=(0,))

    result = run_child(sub_goal, timeout_s=60.0)

    assert isinstance(result, ChildResult)
    assert result.error is None
    # Real bridge pipeline -- not a stub.
    assert result.converged is True
    # RunResult.ground_state must propagate through (EXTENSIONS#1).
    assert result.ground_state is not None
    # MPS path: hamiltonian + trotter_steps populated; meta / solved_ast None.
    assert isinstance(result.hamiltonian, BridgeHamiltonian)
    assert result.trotter_steps == 50
    # run_diagnostic carries the RunResult.to_dict() payload.
    assert result.run_diagnostic["converged"] is True
    assert "energy" in result.run_diagnostic


# ---------------------------------------------------------------------------
# §12.5 / §10.10 — dispatcher invokes QEC on children (A.2)
# ---------------------------------------------------------------------------


def test_dispatcher_calls_qec_on_children():
    """Dispatch two sub-QPCNs with one corrupted; verify dispatcher
    routes the corrupted child through QEC.

    Setup: stub runners return real MERA ground_states (one clean,
    one boundary-perturbed). With parent_state provided, the
    dispatcher's post-pass QEC must flag the corrupted child and
    rewrite its ChildResult to non-converged with a
    ``qec_corruption:`` error (refusal path, since no snapshot is
    supplied to the dispatcher).
    """
    import numpy as np
    from src.qft_pcn.logic.ast import parse as ast_parse
    from src.qft_pcn.logic.mera_encoder import encode_mera as enc

    src = r"\x:Int. x + 1"
    parent_state, _ = enc(ast_parse(src))
    clean_state = enc(ast_parse(src))[0]
    corrupted_state = enc(ast_parse(src))[0]
    d = corrupted_state.d_local
    g = np.zeros((d * d, d * d), dtype=complex)
    for k in range(d * d):
        g[(k + 1) % (d * d), k] = 1.0
    corrupted_state.apply_two_site_gate(0, g)

    states_by_name = {"clean": clean_state, "corrupt": corrupted_state}

    def stub_runner(sub_goal, timeout_s):
        name = sub_goal.dsl_spec["g"]
        return ChildResult(
            goal_id=sub_goal.goal_id, converged=True, residual_energy=1e-9,
            ground_state=states_by_name[name], solved_ast="ast",
            run_diagnostic={"spectral_gap": 1.0}, error=None,
        )

    nodes = [
        Node(goal=_sub("clean"), status=Status.PENDING),
        Node(goal=_sub("corrupt"), status=Status.PENDING),
    ]
    backend = ThreadPoolBackend(max_workers=2)
    results = dispatch_siblings(
        nodes, backend, runner=stub_runner, timeout_s=5.0,
        parent_state=parent_state, qec_threshold=1e-4,
    )
    backend.shutdown()

    by_id = {r.goal_id: r for r in results}
    clean_res = by_id[nodes[0].goal.goal_id]
    corrupt_res = by_id[nodes[1].goal.goal_id]

    # Clean child passes through unchanged.
    assert clean_res.converged is True
    assert clean_res.error is None

    # Corrupted child: dispatcher flipped it to non-converged with a
    # structured QEC error so the integrator's converged-gate refuses.
    assert corrupt_res.converged is False
    assert corrupt_res.error is not None
    assert corrupt_res.error.startswith("qec_corruption:")
    assert corrupt_res.ground_state is None
    assert "qec_corruption_distance" in corrupt_res.run_diagnostic
    assert corrupt_res.run_diagnostic["qec_corruption_distance"] > 1e-4
    assert corrupt_res.run_diagnostic["qec_recovery_applied"] is False


def test_dispatcher_qec_skipped_without_parent_state():
    """Legacy path: parent_state=None -> QEC is a no-op.

    Pins backward compatibility for stub-runner suites and any caller
    that drives the dispatcher without a real MERA workspace.
    """
    def stub_runner(sub_goal, timeout_s):
        return ChildResult(
            goal_id=sub_goal.goal_id, converged=True, residual_energy=1e-9,
            ground_state=object(), solved_ast="ast",
            run_diagnostic={}, error=None,
        )

    nodes = [Node(goal=_sub("c0"), status=Status.PENDING)]
    backend = ThreadPoolBackend(max_workers=1)
    results = dispatch_siblings(
        nodes, backend, runner=stub_runner, timeout_s=5.0,
        # parent_state intentionally omitted
    )
    backend.shutdown()
    assert results[0].converged is True
    assert results[0].error is None
    assert "qec_corruption_distance" not in results[0].run_diagnostic


def test_dispatcher_qec_recovery_with_snapshot():
    """Dispatcher applies snapshot-rollback recovery when a snapshot
    is supplied for a flagged child — the result's ground_state is
    replaced with the recovered (snapshot) MERA and converged stays True.
    """
    import numpy as np
    from src.qft_pcn.logic.ast import parse as ast_parse
    from src.qft_pcn.logic.mera_encoder import encode_mera as enc

    src = r"\x:Int. x"
    parent_state, _ = enc(ast_parse(src))
    snapshot = enc(ast_parse(src))[0]
    corrupted_state = enc(ast_parse(src))[0]
    d = corrupted_state.d_local
    g = np.zeros((d * d, d * d), dtype=complex)
    for k in range(d * d):
        g[(k + 1) % (d * d), k] = 1.0
    corrupted_state.apply_two_site_gate(0, g)

    sg = _sub("corrupt")
    cid = sg.goal_id

    def stub_runner(sub_goal, timeout_s):
        return ChildResult(
            goal_id=sub_goal.goal_id, converged=True, residual_energy=1e-9,
            ground_state=corrupted_state, solved_ast="ast",
            run_diagnostic={"spectral_gap": 1.0}, error=None,
        )

    nodes = [Node(goal=sg, status=Status.PENDING)]
    backend = ThreadPoolBackend(max_workers=1)
    results = dispatch_siblings(
        nodes, backend, runner=stub_runner, timeout_s=5.0,
        parent_state=parent_state, qec_threshold=1e-4,
        qec_snapshots={cid: snapshot},
    )
    backend.shutdown()

    out = results[0]
    assert out.converged is True
    assert out.error is None
    assert out.run_diagnostic["qec_recovery_applied"] is True
    assert out.run_diagnostic["qec_corruption_distance"] > 1e-4
    # ground_state is now the snapshot (bitwise leaves equal).
    assert out.ground_state is not None
    for k, leaf in enumerate(snapshot.leaves):
        assert np.array_equal(out.ground_state.leaves[k], leaf)


# ---------------------------------------------------------------------------
# D1 / D2 (DEVIATIONS.md): real spectral_gap surfaced + cache-hit skip
# ---------------------------------------------------------------------------


def test_dispatcher_emits_spectral_gap_for_real_run(_no_large_dense):
    """D1 (DEVIATIONS.md): RunResult.to_dict must carry a real spectral_gap.

    The previous integrator default was 0.0 so every production child
    was refused by the §6.3 gate. After D1, a real bridge run surfaces
    ``spectral_gap`` from the composed Hamiltonian's eigenvalue spread,
    and the value must be strictly positive for the boundary-pinned
    test problem (the pin produces a clear lowest-eigenstate gap).
    """
    # Pin BOTH sites so the Hamiltonian has a non-degenerate ground state
    # (|0,0>). A single-site pin leaves the other site's eigenstates
    # degenerate (4-fold) and yields a gap of exactly 0.
    dsl_spec = {
        "fields": [{"name": "x", "cutoff": 4}],
        "sites": 2,
        "constraints": [
            {"kind": "local", "site": 0, "term": "x == 0", "weight": 5.0},
            {"kind": "local", "site": 1, "term": "x == 0", "weight": 5.0},
        ],
        "observables": [{"site": 0, "field": "x", "op": "n"}],
        "search": {"method": "imag_time", "steps": 50, "chi_max": 4,
                   "dt": 0.05},
    }
    sub_goal = make_sub_goal(dsl_spec, goal_prop="X_pinned",
                              boundary={}, parent_leaves=(0,))

    result = run_child(sub_goal, timeout_s=60.0)

    assert "spectral_gap" in result.run_diagnostic
    gap = float(result.run_diagnostic["spectral_gap"])
    assert gap > 0.0, (
        f"D1 contract broken: expected positive spectral_gap from the "
        f"boundary-pinned Hamiltonian; got {gap}"
    )
    # Smallest excitation = flipping one pinned site to a non-zero level
    # costs weight 5.0; the gap equals that.
    assert gap == pytest.approx(5.0, rel=1e-6)


def test_dispatcher_cache_hit_skips_redispatch():
    """D2 (§8): a sibling whose goal_id matches a cached lemma's
    source_run_id must short-circuit to a synthesized SOLVED ChildResult
    WITHOUT invoking the runner.
    """
    from src.qft_pcn.composition.dispatcher import _cached_child_from_lemma  # noqa: F401
    sg = _sub("cached-sibling", prop="P")

    # Build a fake library that returns a lemma for this goal_id.
    class _FakeLemma:
        lemma_id = "lem_cached"
        class encoding_meta: pass  # noqa: E701
        class derivation:
            residual_energy = 1e-10
            energy_gap = 1.0
            trotter_steps = 7
            source_run_id = sg.goal_id
        # Minimal mera_tensors stand-in; the synth path calls
        # mera_from_bundle which needs a bundle. We bypass by patching
        # _cached_child_from_lemma's dependency via a stub library that
        # the dispatcher routes through.
        mera_tensors = None

    class _StubLibWithGoalId:
        def __init__(self, lemma):
            self.lemma = lemma
        def find_by_goal_id(self, goal_id):
            return self.lemma if goal_id == sg.goal_id else None

    # Patch mera_from_bundle for the synth so we don't need a real bundle.
    import src.qft_pcn.composition.lemma_library as L
    real_mfb = L.mera_from_bundle
    L.mera_from_bundle = lambda b: "stub_state"  # sentinel ground_state

    called = {"n": 0}
    def never_run(sub_goal, timeout_s):
        called["n"] += 1
        raise AssertionError(
            "runner must NOT be invoked when goal_id hits the cache")

    try:
        lib = _StubLibWithGoalId(_FakeLemma())
        nodes = [Node(goal=sg, status=Status.PENDING)]
        backend = ThreadPoolBackend(max_workers=1)
        results = dispatch_siblings(
            nodes, backend, runner=never_run, timeout_s=5.0,
            lemma_library=lib,
        )
        backend.shutdown()
    finally:
        L.mera_from_bundle = real_mfb

    assert called["n"] == 0, "D2 cache-hit failed: runner was invoked"
    assert len(results) == 1
    assert results[0].converged is True
    assert results[0].run_diagnostic.get("cache_hit") is True
    assert results[0].run_diagnostic.get("cached_lemma_id") == "lem_cached"
    # Gap propagates from the cached derivation for the integrator gate.
    assert results[0].run_diagnostic["spectral_gap"] == 1.0
