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
    import src.qft_pcn.logic.mera_decoder as MD
    real_mfb = L.mera_from_bundle
    L.mera_from_bundle = lambda b: "stub_state"  # sentinel ground_state
    # D36: dispatcher now decodes the cached MERA to populate solved_ast.
    # This test uses a sentinel state, so patch decode_mera through to a
    # sentinel DecodeResult (the runner-invocation invariant is what is
    # under test here; a full decode_mera round-trip is exercised by
    # test_cached_child_decodes_ast below).
    real_decode = MD.decode_mera
    class _SentinelDecode:
        ast = "sentinel-ast"
        residual_norm = 0.0
    MD.decode_mera = lambda state, meta: _SentinelDecode()

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
        MD.decode_mera = real_decode

    assert called["n"] == 0, "D2 cache-hit failed: runner was invoked"
    assert len(results) == 1
    assert results[0].converged is True
    assert results[0].run_diagnostic.get("cache_hit") is True
    assert results[0].run_diagnostic.get("cached_lemma_id") == "lem_cached"
    # Gap propagates from the cached derivation for the integrator gate.
    assert results[0].run_diagnostic["spectral_gap"] == 1.0
    # D36: cache-hit MUST decode and populate solved_ast (no None pollution
    # of §10.11 ProofTreeNode leaves).
    assert results[0].solved_ast == "sentinel-ast"


def test_cached_child_records_compress_skipped_or_hamiltonian():
    """D34 (DEVIATIONS.md): the cache-hit ChildResult must EITHER carry a
    real Hamiltonian (so register_lemma's compress branch can run) OR
    stamp ``compress_skipped=True`` so the integrator deliberately skips
    the compress step instead of silently feeding ``hamiltonian=None``.

    Today the LemmaLibrary persists only bundle + meta + derivation
    (EXTENSIONS.md "cache-hit hamiltonian trade-off"), so the implemented
    branch is the explicit compress-skipped marker. The flag must
    appear on BOTH the dataclass field and inside ``run_diagnostic`` so
    downstream provenance reports see it.
    """
    from src.qft_pcn.composition.dispatcher import _cached_child_from_lemma
    sg = _sub("d34-cached-sibling", prop="P")

    class _FakeLemma:
        lemma_id = "lem_d34"
        class encoding_meta: pass  # noqa: E701
        class derivation:
            residual_energy = 1e-12
            energy_gap = 0.7
            trotter_steps = 11
            source_run_id = sg.goal_id
        mera_tensors = None

    import src.qft_pcn.composition.lemma_library as L
    import src.qft_pcn.logic.mera_decoder as MD
    real_mfb = L.mera_from_bundle
    real_decode = MD.decode_mera
    L.mera_from_bundle = lambda b: "stub_state"
    MD.decode_mera = lambda state, meta: type("D", (), {"ast": None})()
    try:
        node = Node(goal=sg, status=Status.PENDING)
        cr = _cached_child_from_lemma(node, _FakeLemma())
    finally:
        L.mera_from_bundle = real_mfb
        MD.decode_mera = real_decode

    # D34 acceptance: explicit marker + reason on both surfaces.
    has_ham = cr.hamiltonian is not None
    has_flag = getattr(cr, "compress_skipped", False) is True
    assert has_ham or has_flag, (
        "D34: cache-hit ChildResult must surface either a Hamiltonian "
        "or compress_skipped=True; got hamiltonian=None and no flag")
    if not has_ham:
        assert has_flag, "compress_skipped flag missing"
        assert cr.run_diagnostic.get("compress_skipped") is True, (
            "D34: compress_skipped must also appear in run_diagnostic "
            "for downstream provenance")
        assert "compress_skipped_reason" in cr.run_diagnostic, (
            "D34: compress_skipped_reason must name the §8 trade")


# ---------------------------------------------------------------------------
# D23 (DEVIATIONS.md): above-ceiling substrate must NOT silently emit 0.0
# ---------------------------------------------------------------------------


def test_spectral_gap_above_ceiling_routes_to_lanczos_or_explicit_refuse():
    """D23 (DEVIATIONS.md): when ``d_local ** N > dim_ceiling`` the bridge
    dense-diag path cannot materialise the Hamiltonian. The pre-D23 behaviour
    returned ``0.0`` -- bitwise identical to a genuinely gapless substrate
    and to the integrator's strict-refuse default, so an M3-scale child
    (d_local=8, N>=5 -> dim=32768) was silently refused at the §6.3 gate
    with the misleading "near-degenerate" reason. §1.1 anti-shortcut
    violation.

    Acceptance: the producer surfaces a NaN sentinel (option (a): explicit
    "unavailable", chosen over option (b) Lanczos because
    :class:`BridgeHamiltonian` does not currently expose a sparse /
    ``LinearOperator`` apply; EXTENSIONS.md records the Lanczos route).
    The :func:`integrate_child` consumer detects NaN and refuses with a
    CLEAR reason that names the substrate dim instead of
    "near-degenerate".
    """
    import math as _math
    from src.qft_pcn.bridge.runtime import _spectral_gap_from_hamiltonian
    from src.qft_pcn.bridge.runtime.hamiltonian import BridgeHamiltonian
    from src.qft_pcn.bridge.dsl.term import FieldSpec
    from src.qft_pcn.composition.result_integrator import (
        integrate_child, IntegrationOutcome,
    )
    from src.qft_pcn.composition.dispatcher import ChildResult
    from src.qft_pcn.composition.goal_graph import (
        make_sub_goal, Node, Status,
    )

    # --- producer side: dim above ceiling must emit NaN, NOT 0.0 ---------
    # Construct a real bridge Hamiltonian whose ``d_local ** N`` exceeds the
    # default ``dim_ceiling=4096``. cutoff=4, sites=7 -> dim 4**7 = 16384.
    fields = [FieldSpec(name="x", cutoff=4)]
    H_above = BridgeHamiltonian(fields=fields, sites=7, terms=[])
    assert int(H_above.d_local) ** int(H_above.N) > 4096, (
        "test premise broken: substrate must be above the dense-diag ceiling")
    gap_above = _spectral_gap_from_hamiltonian(H_above)
    assert _math.isnan(gap_above), (
        f"D23 contract broken: above-ceiling substrate must surface NaN "
        f"(sentinel) not a silent numeric -- got {gap_above!r}")

    # And the under-ceiling path still returns a finite gap (no regression).
    H_below = BridgeHamiltonian(fields=fields, sites=2, terms=[])
    gap_below = _spectral_gap_from_hamiltonian(H_below)
    assert not _math.isnan(gap_below), (
        "D23 regression: under-ceiling substrate must still compute a "
        f"finite gap; got {gap_below!r}")

    # --- consumer side: NaN must refuse with a CLEAR reason --------------
    # Stand up a minimal Node and a synthesised converged ChildResult whose
    # run_diagnostic carries the NaN. The integrator must refuse with a
    # reason that names the unavailability, NOT "near-degenerate".
    sg = make_sub_goal({"g": "above_ceiling"}, goal_prop="P", boundary={},
                       parent_leaves=(0,))
    node = Node(goal=sg, status=Status.PENDING)
    cr = ChildResult(
        goal_id=sg.goal_id,
        converged=True,
        residual_energy=1e-9,
        ground_state=object(),
        solved_ast=None,
        run_diagnostic={"spectral_gap": _math.nan},
        error=None,
        meta=object(),
        hamiltonian=H_above,
        trotter_steps=0,
    )
    outcome: IntegrationOutcome = integrate_child(
        parent_state=None, parent_meta=None, node=node,
        child_result=cr, lemma_library=None,
    )
    assert outcome.integrated is False
    assert "spectral_gap unavailable" in outcome.reason, (
        f"D23 contract: refusal reason must name the unavailability "
        f"explicitly, not be the misleading 'near-degenerate'; "
        f"got: {outcome.reason!r}")
    # Substrate dim surfaced in the message so the operator can act.
    assert "16384" in outcome.reason, (
        f"D23 contract: refusal reason should name the substrate dim "
        f"(16384 for cutoff=4 N=7); got: {outcome.reason!r}")
    assert node.status == Status.FAILED


# ---------------------------------------------------------------------------
# D36 (DEVIATIONS.md): cache-hit MUST decode solved_ast from the persisted
# MERA + encoding_meta, not pollute §10.11 ProofTreeNode leaves with None.
# ---------------------------------------------------------------------------


def test_cached_child_decodes_ast(tmp_path, _no_large_dense):
    """D36: ``_cached_child_from_lemma`` must invoke ``decode_mera`` on
    the restored MERA so the synthesised ``ChildResult.solved_ast`` is
    populated, and so ``extract_proof_tree`` carries the decoded AST
    into the §10.11 ``ProofTreeNode`` for cache-hit leaves.

    End-to-end real-substrate test: encode a Forall AST, save it as a
    lemma keyed on a specific ``source_run_id``, dispatch a sibling
    whose ``goal_id`` matches that source_run_id (cache hit fires), and
    verify both the direct synth result, the dispatch_siblings result,
    AND the extracted proof tree leaf carry the decoded AST.
    """
    from src.qft_pcn.composition.dispatcher import (
        _cached_child_from_lemma,
        dispatch_siblings as _dispatch,
    )
    from src.qft_pcn.composition.lemma_library import (
        DerivationMetadata,
        Lemma,
        LemmaLibrary,
        bundle_from_mera,
        structural_fingerprint,
    )
    from src.qft_pcn.composition.goal_graph import (
        extract_proof_tree,
        ProofTree,
    )
    from src.qft_pcn.logic.ast import (
        Bin,
        Eq,
        Forall,
        TNat,
        Var,
        Zero,
    )
    from src.qft_pcn.logic.mera_encoder import encode_mera

    # 1. Real AST -> real MERA + meta through the production encoder.
    body = Eq(
        lhs=Bin(op="add", a=Var(name="x"), b=Zero()),
        rhs=Var(name="x"),
    )
    ast = Forall(var="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(ast)

    # 2. Save a Lemma keyed by source_run_id == the sibling's goal_id.
    sg = make_sub_goal(
        {"g": "d36-cache-hit"},
        goal_prop="forall x:Nat. Eq (add x Zero) x",
        boundary={}, parent_leaves=(0,),
    )
    cached_goal_id = sg.goal_id
    bundle = bundle_from_mera(state)
    fp = structural_fingerprint(state)
    deriv = DerivationMetadata(
        hamiltonian_id="d36-test",
        residual_energy=1e-12,
        energy_gap=1.0,
        trotter_steps=0,
        assumptions=(),
        lemma_deps=(),
        conditional=False,
        source_run_id=cached_goal_id,
    )
    lemma = Lemma(
        lemma_id="d36:forall_eq_addzero",
        proposition_type="forall x:Nat. Eq (add x Zero) x",
        mera_tensors=bundle,
        encoding_meta=meta,
        derivation=deriv,
        fingerprint=fp,
    )
    lib = LemmaLibrary(tmp_path)
    lib.save(lemma)

    # 3. Direct synth check: cache-hit path under test in isolation.
    node_direct = Node(goal=sg, status=Status.PENDING)
    res_direct = _cached_child_from_lemma(
        node_direct, lib.find_by_goal_id(cached_goal_id),
    )
    assert res_direct.converged is True
    assert res_direct.run_diagnostic.get("cache_hit") is True
    assert res_direct.solved_ast is not None, (
        "D36: _cached_child_from_lemma must decode the cached MERA and "
        "stamp solved_ast -- got None, which pollutes §10.11 proof "
        "tree leaves on every cache hit."
    )

    # 4. End-to-end via dispatch_siblings: cache hit short-circuits the
    # runner AND the synthesised ChildResult carries solved_ast.
    nodes = [Node(goal=sg, status=Status.PENDING)]

    def never_run(sub_goal, timeout_s):
        raise AssertionError("runner must NOT fire on D36 cache hit")

    backend = ThreadPoolBackend(max_workers=1)
    try:
        results = _dispatch(
            nodes, backend, runner=never_run, timeout_s=5.0,
            lemma_library=lib,
        )
    finally:
        backend.shutdown()
    assert len(results) == 1
    assert results[0].converged is True
    assert results[0].solved_ast is not None, (
        "D36: dispatch_siblings cache-hit path lost solved_ast"
    )

    # 5. extract_proof_tree: the §10.11 ProofTreeNode leaf must carry
    # the decoded AST (not None). The cache-hit path leaves the node
    # in ACTIVE state with a populated result; promote to SOLVED so
    # the extractor walks it as a real leaf (mirrors what the
    # orchestrator does post-integrate_child for a converged hit).
    root = nodes[0]
    root.status = Status.SOLVED
    tree = extract_proof_tree(root)
    assert isinstance(tree, ProofTree)
    assert tree.root.solved_ast is not None, (
        "D36: extract_proof_tree carried solved_ast=None into the "
        "§10.11 ProofTreeNode for a cache-hit leaf"
    )


def test_cached_child_raises_on_decode_failure():
    """D36: a cached entry whose ``encoding_meta`` cannot be decoded is
    a cache-integrity fault -- surface ``CacheDecodeError`` loudly per
    §1.1, never silently keep ``solved_ast=None``.
    """
    from src.qft_pcn.composition.dispatcher import (
        CacheDecodeError,
        _cached_child_from_lemma,
    )
    import src.qft_pcn.composition.lemma_library as L
    import src.qft_pcn.logic.mera_decoder as MD

    sg = _sub("d36-decode-fail", prop="P")

    class _FakeLemma:
        lemma_id = "lem_corrupt"
        class encoding_meta: pass  # noqa: E701
        class derivation:
            residual_energy = 1e-10
            energy_gap = 1.0
            trotter_steps = 0
        mera_tensors = None

    real_mfb = L.mera_from_bundle
    real_decode = MD.decode_mera
    L.mera_from_bundle = lambda b: "stub_state"

    def _boom(state, meta):
        raise RuntimeError("bundle/meta disagree about n_nodes")
    MD.decode_mera = _boom
    try:
        node = Node(goal=sg, status=Status.PENDING)
        with pytest.raises(CacheDecodeError) as excinfo:
            _cached_child_from_lemma(node, _FakeLemma())
        assert excinfo.value.lemma_id == "lem_corrupt"
        assert excinfo.value.goal_id == sg.goal_id
        assert isinstance(excinfo.value.cause, RuntimeError)
    finally:
        L.mera_from_bundle = real_mfb
        MD.decode_mera = real_decode
