"""Orchestrator tests (Task 7, spec §5.3, §6.5, §7, §9.5).

End-to-end of the free-energy-minimizing search loop with a stub runner +
real LemmaLibrary + real MERA states. The real-QPCN acceptance is the
Task 8 §10.10 inductive theorem.

K-5 review tightened ``integrate_child`` to use the real
:func:`lemma_library.register_lemma` + :class:`promoter.Promoter`
surfaces, so the orchestrator's child runs must return ChildResults with
real ``meta`` and ``ground_state`` -- the same pattern the K-5 result-
integrator tests use.
"""
from __future__ import annotations

import math

import pytest

from src.qft_pcn.composition.dispatcher import ChildResult, ThreadPoolBackend
from src.qft_pcn.composition.errors import RevisionExhausted
from src.qft_pcn.composition.goal_graph import (
    Status,
    make_contiguous_sub_goal,
    make_sub_goal,
)
from src.qft_pcn.composition.lemma_library import LemmaLibrary
from src.qft_pcn.composition.orchestrator import (
    MAX_REVISIONS,
    SolveResult,
    solve_goal_graph,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


# ---------------------------------------------------------------------------
# Opt this file out of the conftest §9.7 dense-tensor ceiling: the real
# encode_mera that backs every fixture allocates 4096-element pair matrices
# by construction (same pattern as test_cross_level_acceptance.py and the
# new test_orchestrator_parent_workspace.py). Anti-shortcut: we do NOT
# raise the ceiling for the whole composition module.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadows the conftest fixture for this file only
    yield


# ---------------------------------------------------------------------------
# Fixtures: real LemmaLibrary + real MERA state so integrate_child's
# register_lemma + Promoter machinery is genuinely exercised.
# ---------------------------------------------------------------------------


@pytest.fixture
def lib(tmp_path):
    """A real, file-backed LemmaLibrary in tmp_path."""
    return LemmaLibrary(tmp_path)


@pytest.fixture
def child_state_meta():
    """A real MERA ground state + encoding meta for child runs."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    return state, meta


@pytest.fixture
def parent_state_meta():
    """A real parent MERA + meta whose species_of_leaf pattern matches the
    child fixture at indices ``[0, n_leaves)`` -- the natural case for an
    integrator clamping a child into an isomorphic parent region."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    return state, meta


# ---------------------------------------------------------------------------
# Stub decomposer (same shape as the goal_graph test stub).
# ---------------------------------------------------------------------------


class StubDecomposer:
    """Maps a goal_prop to a fixed list of child (spec, prop) pairs."""

    def __init__(self, table):
        self._table = table

    def decompose(self, node):
        out = []
        for i, (spec, prop) in enumerate(
            self._table.get(node.goal.goal_prop, [])
        ):
            # Sibling lemmas in this stub share the same isomorphic
            # 16-leaf window as the child fixture's encoding -- overlap
            # is acceptable for the test (clamps are idempotent on
            # identical child states).
            out.append(make_contiguous_sub_goal(
                spec, goal_prop=prop, boundary={},
                base=0, n_leaves=16,
            ))
        return out


# ---------------------------------------------------------------------------
# Test 1: three-level end-to-end -- proof tree + monotone F_hierarchy.
# ---------------------------------------------------------------------------


def test_solve_goal_graph_three_levels_with_stub_runner(
    lib, child_state_meta, parent_state_meta,
):
    """End-to-end of the orchestrator with a stub runner + real lemma lib.

    Spec invariants exercised:
        * parallel sibling dispatch on the two leaf goals (§5.2),
        * gated integration via the real register_lemma + Promoter (§6.3),
        * monotone F_hierarchy across the run (§9.5),
        * proof tree mirrors the SOLVED structure (§4.6).
    """
    cstate, cmeta = child_state_meta
    pstate, pmeta = parent_state_meta
    # IndCase is the internal node (it decomposes into LemmaA + LemmaB);
    # LemmaA and LemmaB are leaves (no entry in the table -> empty
    # decomposition -> handled by the leaf branch of _solve).
    table = {
        "Thm":     [({"g": "ind"}, "IndCase")],
        "IndCase": [({"g": "L1a"}, "LemmaA"), ({"g": "L1b"}, "LemmaB")],
    }

    def stub_runner(sub_goal, timeout_s):
        # Each child returns a real MERA ground state + meta so the
        # integrator's register_lemma + Promoter pipeline accepts it.
        return ChildResult(
            goal_id=sub_goal.goal_id,
            converged=True,
            residual_energy=1e-9,
            ground_state=cstate,
            solved_ast=f"ast::{sub_goal.goal_prop}",
            run_diagnostic={"spectral_gap": 1.0},
            error=None,
            meta=cmeta,
            hamiltonian=None,
            trotter_steps=0,
        )

    free_energies: list[float] = []
    result = solve_goal_graph(
        {"g": "root"}, root_prop="Thm",
        decomposer=StubDecomposer(table),
        backend=ThreadPoolBackend(max_workers=4),
        lemma_library=lib,
        runner=stub_runner, timeout_s=5.0,
        on_step=free_energies.append,
        parent_state=pstate, parent_meta=pmeta,
    )
    assert isinstance(result, SolveResult)
    assert result.solved is True
    # F_hierarchy monotone non-increasing across the run (spec §9.5).
    assert all(
        free_energies[i] >= free_energies[i + 1] - 1e-12
        for i in range(len(free_energies) - 1)
    )
    assert result.proof_tree is not None
    assert result.proof_tree.root.goal_prop == "Thm"
    assert result.failure_report is None


# ---------------------------------------------------------------------------
# Test 2: root failure returns a structured report -- never a fabricated proof.
# ---------------------------------------------------------------------------


def test_solve_goal_graph_root_failure_returns_structured_report(
    lib, parent_state_meta,
):
    """A non-converging runner exhausts revisions -- the orchestrator must
    surface a structured failure_report (spec §6.5), not a fabricated proof
    tree."""
    def failing_runner(sub_goal, timeout_s):
        return ChildResult(
            goal_id=sub_goal.goal_id,
            converged=False,
            residual_energy=float("inf"),
            ground_state=None,
            solved_ast=None,
            run_diagnostic={},
            error="diverged",
            meta=None,
            hamiltonian=None,
            trotter_steps=0,
        )

    pstate, pmeta = parent_state_meta
    result = solve_goal_graph(
        {"g": "root"}, root_prop="Leaf",
        decomposer=StubDecomposer({}),   # no decomposition -> leaf goal
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=lib,
        runner=failing_runner, timeout_s=2.0,
        parent_state=pstate, parent_meta=pmeta,
    )
    assert isinstance(result, SolveResult)
    assert result.solved is False
    assert result.proof_tree is None
    assert result.failure_report is not None      # structured, not fabricated
    # The report carries the exhaustion diagnostics so the failure is
    # diagnosable from logs alone (spec §6.5).
    assert result.failure_report["revision_attempts"] == MAX_REVISIONS + 1
    assert result.failure_report["root_status"] in {
        Status.FAILED.value, Status.PENDING_REVISION.value,
    }


# ---------------------------------------------------------------------------
# Test 3: RevisionExhausted carries the failing goal_id + attempt count.
#
# The orchestrator catches RevisionExhausted internally at the root, but
# the failure_report MUST preserve the exhaustion metadata (spec §6.5,
# K Task 6 deferral note in revision.py).
# ---------------------------------------------------------------------------


def test_root_failure_report_carries_revision_exhaustion_metadata(
    lib, parent_state_meta,
):
    """The structured report exposes the typed-error fields so callers can
    branch on revision exhaustion specifically."""
    def failing_runner(sub_goal, timeout_s):
        return ChildResult(
            goal_id=sub_goal.goal_id, converged=False,
            residual_energy=float("inf"), ground_state=None,
            solved_ast=None, run_diagnostic={}, error="diverged",
            meta=None, hamiltonian=None, trotter_steps=0,
        )

    pstate, pmeta = parent_state_meta
    result = solve_goal_graph(
        {"g": "root"}, root_prop="Leaf",
        decomposer=StubDecomposer({}),
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=lib,
        runner=failing_runner, timeout_s=2.0,
        parent_state=pstate, parent_meta=pmeta,
    )
    assert result.solved is False
    report = result.failure_report
    assert report is not None
    # The exhausted goal_id is carried so a caller can locate the failing
    # sub-tree in the parent's bookkeeping.
    assert "exhausted_goal_id" in report
    assert isinstance(report["exhausted_goal_id"], str)
    # MAX_REVISIONS retries + 1 initial attempt = MAX_REVISIONS + 1.
    assert report["revision_attempts"] == MAX_REVISIONS + 1


# ---------------------------------------------------------------------------
# Test 4: a RevisionExhausted under a non-root subtree still surfaces as a
# root-level structured report. The orchestrator must never silently loop
# (anti-shortcut: silent infinite retry is the §6.5 antipattern).
# ---------------------------------------------------------------------------


def test_subtree_revision_exhaustion_bubbles_into_root_failure_report(
    lib, child_state_meta, parent_state_meta,
):
    cstate, cmeta = child_state_meta
    pstate, pmeta = parent_state_meta

    def failing_runner(sub_goal, timeout_s):
        # Every child diverges -- a sub-goal under the root will exhaust
        # its revision budget; that exhaustion bubbles up as a typed error
        # that the root wrapper catches into the structured report.
        return ChildResult(
            goal_id=sub_goal.goal_id, converged=False,
            residual_energy=float("inf"), ground_state=None,
            solved_ast=None, run_diagnostic={}, error="diverged",
            meta=None, hamiltonian=None, trotter_steps=0,
        )

    table = {"Thm": [({"g": "child"}, "ChildLeaf")]}
    result = solve_goal_graph(
        {"g": "root"}, root_prop="Thm",
        decomposer=StubDecomposer(table),
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=lib,
        runner=failing_runner, timeout_s=2.0,
        parent_state=pstate, parent_meta=pmeta,
    )
    assert result.solved is False
    assert result.failure_report is not None
    # The orchestrator never raised to the caller -- exhaustion is captured.
    assert "exhausted_goal_id" in result.failure_report


# ---------------------------------------------------------------------------
# Test 5: RevisionExhausted is a real, typed error -- the orchestrator
# imports it from composition.errors (not a string sentinel). This anchors
# the §6.5 typed-error contract.
# ---------------------------------------------------------------------------


def test_revision_exhausted_is_a_typed_error_from_composition_errors():
    e = RevisionExhausted(goal_id="g_test", attempts=4)
    assert e.goal_id == "g_test"
    assert e.attempts == 4
    assert isinstance(e, Exception)


# ---------------------------------------------------------------------------
# §12.16 / A.3 top-k ranking API surface.
# ---------------------------------------------------------------------------


def _stub_runner_for(cstate, cmeta):
    def stub_runner(sub_goal, timeout_s):
        return ChildResult(
            goal_id=sub_goal.goal_id,
            converged=True,
            residual_energy=1e-9,
            ground_state=cstate,
            solved_ast=f"ast::{sub_goal.goal_prop}",
            run_diagnostic={"spectral_gap": 1.0},
            error=None,
            meta=cmeta,
            hamiltonian=None,
            trotter_steps=0,
        )
    return stub_runner


def test_orchestrator_top_k_default_is_single_proof(
    lib, child_state_meta, parent_state_meta,
):
    """Default ``n_top_k=1``: the orchestrator returns the solved proof
    tree and ``ranked_proofs`` is the single-element ranking with weight
    1.0. No behaviour change vs prior callers (spec §12.16 / A.3
    plumbing)."""
    cstate, cmeta = child_state_meta
    pstate, pmeta = parent_state_meta
    table = {
        "Thm":     [({"g": "ind"}, "IndCase")],
        "IndCase": [({"g": "L1a"}, "LemmaA"), ({"g": "L1b"}, "LemmaB")],
    }
    result = solve_goal_graph(
        {"g": "root"}, root_prop="Thm",
        decomposer=StubDecomposer(table),
        backend=ThreadPoolBackend(max_workers=2),
        lemma_library=lib,
        runner=_stub_runner_for(cstate, cmeta), timeout_s=5.0,
        parent_state=pstate, parent_meta=pmeta,
    )
    assert result.solved is True
    assert result.proof_tree is not None
    assert len(result.ranked_proofs) == 1
    r0 = result.ranked_proofs[0]
    assert r0.tree is result.proof_tree
    assert r0.weight == 1.0


def test_orchestrator_top_k_when_multiple_candidates_ranks_via_worldline_pi(
    lib, child_state_meta, parent_state_meta,
):
    """``n_top_k > 1`` is plumbed at the API surface. Multi-candidate
    generation requires revision-tracking and is deferred per
    EXTENSIONS.md A.3; the surface still returns a sorted ``ranked_proofs``
    list (currently size 1) so downstream callers depend only on the
    invariant ``ranked_proofs`` is non-empty + sorted by weight descending
    on success."""
    cstate, cmeta = child_state_meta
    pstate, pmeta = parent_state_meta
    table = {"Thm": [({"g": "ind"}, "LemmaA")]}
    result = solve_goal_graph(
        {"g": "root"}, root_prop="Thm",
        decomposer=StubDecomposer(table),
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=lib,
        runner=_stub_runner_for(cstate, cmeta), timeout_s=5.0,
        parent_state=pstate, parent_meta=pmeta,
        n_top_k=3,
        ranking_temperature=0.5,
    )
    assert result.solved is True
    assert len(result.ranked_proofs) >= 1
    # Sorted by weight descending.
    weights = [r.weight for r in result.ranked_proofs]
    assert weights == sorted(weights, reverse=True)
    # And every ranking is computed via bayesian_rank_proofs (action +
    # weight populated, weight in [0, 1], action is the §12.16 action).
    from src.qft_pcn.composition.worldline_pi import (
        ProofRanking, compute_action,
    )
    for r in result.ranked_proofs:
        assert isinstance(r, ProofRanking)
        assert 0.0 <= r.weight <= 1.0
        assert math.isclose(r.action, compute_action(r.tree),
                            rel_tol=1e-9, abs_tol=1e-12)


def test_orchestrator_rejects_n_top_k_below_one(
    lib, child_state_meta, parent_state_meta,
):
    """``n_top_k < 1`` is nonsense (the top-k surface ranks at least one
    proof); the API surfaces a ValueError instead of silently coercing."""
    cstate, cmeta = child_state_meta
    pstate, pmeta = parent_state_meta
    with pytest.raises(ValueError):
        solve_goal_graph(
            {"g": "root"}, root_prop="Thm",
            decomposer=StubDecomposer({"Thm": []}),
            backend=ThreadPoolBackend(max_workers=1),
            lemma_library=lib,
            runner=_stub_runner_for(cstate, cmeta), timeout_s=2.0,
            parent_state=pstate, parent_meta=pmeta,
            n_top_k=0,
        )


# ---------------------------------------------------------------------------
# D7 (DEVIATIONS.md): _frontier_priority uses precision-weighted signal
# ---------------------------------------------------------------------------


def test_frontier_priority_uses_precision_weighted_dF():
    """D7: _frontier_priority must order nodes by precision-weighted ΔF,
    not by structural fan-out alone.

    Setup: two nodes with IDENTICAL structural fan-out (same boundary
    size and child count), but different attached results -- one
    low-residual (high precision), one high-residual (low precision).
    The low-residual node MUST sort higher than the high-residual
    node under the precision-weighted ordering. Under the old structural
    proxy the two would tie.
    """
    from src.qft_pcn.composition.orchestrator import _frontier_priority
    from src.qft_pcn.composition.dispatcher import ChildResult
    from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status

    sg_a = make_sub_goal({"g": "A"}, goal_prop="P", boundary={"b": 0},
                         parent_leaves=(0,))
    sg_b = make_sub_goal({"g": "B"}, goal_prop="P", boundary={"b": 0},
                         parent_leaves=(0,))
    node_low_res = Node(goal=sg_a, status=Status.ACTIVE)
    node_high_res = Node(goal=sg_b, status=Status.ACTIVE)

    # Attach matching-shape results with very different residuals. The
    # synthesized run_diagnostic is irrelevant -- _frontier_priority
    # reads residual_energy directly.
    node_low_res.result = ChildResult(
        goal_id=sg_a.goal_id, converged=True, residual_energy=1e-10,
        ground_state=None, solved_ast=None,
        run_diagnostic={"spectral_gap": 1.0}, error=None,
    )
    node_high_res.result = ChildResult(
        goal_id=sg_b.goal_id, converged=True, residual_energy=1e-2,
        ground_state=None, solved_ast=None,
        run_diagnostic={"spectral_gap": 1.0}, error=None,
    )

    p_low = _frontier_priority(node_low_res)
    p_high = _frontier_priority(node_high_res)
    assert p_low > p_high, (
        f"D7 contract broken: precision-weighted priority should rank "
        f"the low-residual node higher; got p_low={p_low}, p_high={p_high}"
    )


def test_frontier_priority_measured_outranks_unmeasured():
    """D7: a node WITH a substrate measurement must outrank a node
    without one (no-measurement fallback is scaled down by 1e-3 so it
    cannot dominate any precision-weighted signal).
    """
    from src.qft_pcn.composition.orchestrator import _frontier_priority
    from src.qft_pcn.composition.dispatcher import ChildResult
    from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status

    sg_m = make_sub_goal({"g": "M"}, goal_prop="P", boundary={"b": 0},
                         parent_leaves=(0,))
    # Unmeasured node with a very large structural fan-out (huge
    # boundary). Even so, the measured node must outrank it.
    sg_u = make_sub_goal(
        {"g": "U"}, goal_prop="P",
        boundary={f"b{i}": 0 for i in range(100)}, parent_leaves=(0,),
    )
    node_measured = Node(goal=sg_m, status=Status.ACTIVE)
    node_unmeasured = Node(goal=sg_u, status=Status.ACTIVE)

    # Modest residual: precision is small but nonzero.
    node_measured.result = ChildResult(
        goal_id=sg_m.goal_id, converged=True, residual_energy=1e-4,
        ground_state=None, solved_ast=None,
        run_diagnostic={"spectral_gap": 1.0}, error=None,
    )
    p_m = _frontier_priority(node_measured)
    p_u = _frontier_priority(node_unmeasured)
    assert p_m > p_u, (
        f"D7 contract broken: measured node must outrank unmeasured; "
        f"got p_measured={p_m}, p_unmeasured={p_u}"
    )
