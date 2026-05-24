"""A4 ablation: substrate switch for §14.4 row A4 / §10.9.

E26 (EXTENSIONS.md): ``solve_goal_graph(freeze_library=True)`` is the
real substrate switch -- the orchestrator snapshots the
LemmaLibrary at entry and rolls back every novel lemma in a
``finally`` guard, so the library exits in its initial state. The A4
ablation row in ``ABLATION_CONFIGS`` is flipped ``wired=True`` and the
runner produces genuine A4 attempts via ``QPCNBaseline(
freeze_library=True)``.

ANTI-SHORTCUT (§1.1 / memory:anti-shortcut-directive): the freeze
substrate switch does NOT disable the in-loop ``register_lemma`` +
``Promoter.apply_init_clamp`` -- the clamp IS the §1.1 entanglement
binding and skipping it would decay the solver to classical lookup.
Only the PERSISTED side-effect is rolled back at solve exit; the
in-solve clamp still fires.
"""
from __future__ import annotations

import pytest

from experiments.ablations.ablation_runner import (
    ABLATION_CONFIGS,
    _apply_ablation_to_attempt,
    run_ablation_matrix,
)
from experiments.baselines.qpcn import QPCNBaseline
from experiments.benchmarks import load_minif2f
from src.qft_pcn.composition.dispatcher import ChildResult, ThreadPoolBackend
from src.qft_pcn.composition.goal_graph import make_contiguous_sub_goal
from src.qft_pcn.composition.lemma_library import LemmaLibrary
from src.qft_pcn.composition.orchestrator import (
    SolveResult,
    solve_goal_graph,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


# ---------------------------------------------------------------------------
# Opt out of the conftest §9.7 dense-tensor ceiling: ``encode_mera`` on the
# fixture statement allocates 4096-element pair matrices by construction
# (same pattern as test_orchestrator.py).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadows the conftest ceiling for this file only
    yield


# ---------------------------------------------------------------------------
# Fixtures: real LemmaLibrary + real MERA states so the orchestrator's
# register_lemma + Promoter pipeline is genuinely exercised.
# ---------------------------------------------------------------------------


@pytest.fixture
def lib(tmp_path):
    return LemmaLibrary(tmp_path)


@pytest.fixture
def states():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    return state, meta


class _StubDecomposer:
    """One internal-level decomposer: Thm -> two leaf siblings.

    Mirrors the StubDecomposer used by test_orchestrator.py so the
    register_lemma + Promoter pipeline runs end-to-end.
    """

    def __init__(self, table):
        self._table = table

    def decompose(self, node):
        out = []
        for spec, prop in self._table.get(node.goal.goal_prop, []):
            out.append(make_contiguous_sub_goal(
                spec, goal_prop=prop, boundary={},
                base=0, n_leaves=16,
            ))
        return out


def _converging_runner(cstate, cmeta):
    def runner(sub_goal, timeout_s):
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
    return runner


# ---------------------------------------------------------------------------
# Test 1: A4 freeze post-condition -- library size unchanged across the
# solve. This is the §14.4 / §10.9 substrate post-condition: "no new
# lemmas land in the library during the solve".
# ---------------------------------------------------------------------------


def test_a4_freezes_lemma_library(lib, states):
    """``solve_goal_graph(freeze_library=True)`` rolls back every lemma
    registered during the solve, so ``library.all_ids()`` is unchanged
    across the call. The same solve with ``freeze_library=False``
    (BASELINE) DOES grow the library -- the two regimes must differ on
    the library post-condition for the ablation to mean anything.
    """
    state, meta = states
    table = {
        "Thm": [({"g": "L1a"}, "LemmaA"), ({"g": "L1b"}, "LemmaB")],
    }

    initial_ids_snapshot = set(lib.all_ids())
    initial_size = len(initial_ids_snapshot)

    # --- BASELINE: library grows (sanity check that the freeze is doing
    # real work, not no-op behaviour) -------------------------------------
    baseline_result = solve_goal_graph(
        {"g": "root"}, root_prop="Thm",
        decomposer=_StubDecomposer(table),
        backend=ThreadPoolBackend(max_workers=2),
        lemma_library=lib,
        runner=_converging_runner(state, meta), timeout_s=5.0,
        parent_state=state, parent_meta=meta,
        freeze_library=False,
    )
    assert isinstance(baseline_result, SolveResult)
    assert baseline_result.solved is True
    baseline_size = len(lib.all_ids())
    assert baseline_size > initial_size, (
        f"BASELINE solve should register at least one lemma "
        f"(initial={initial_size}, after={baseline_size})"
    )

    # --- A4 FREEZE: library size is EXACTLY the post-BASELINE size; no
    # additional novel lemmas land. The post-condition is computed
    # against the pre-A4 snapshot (after BASELINE registered its lemmas)
    # so any lemma registered DURING this A4 solve must be rolled back.
    pre_a4_ids = set(lib.all_ids())
    a4_result = solve_goal_graph(
        {"g": "root_a4"}, root_prop="Thm",
        decomposer=_StubDecomposer(table),
        backend=ThreadPoolBackend(max_workers=2),
        lemma_library=lib,
        runner=_converging_runner(state, meta), timeout_s=5.0,
        parent_state=state, parent_meta=meta,
        freeze_library=True,
    )
    assert isinstance(a4_result, SolveResult)
    post_a4_ids = set(lib.all_ids())
    assert post_a4_ids == pre_a4_ids, (
        f"freeze_library=True must roll back every novel lemma; "
        f"diff={post_a4_ids - pre_a4_ids}"
    )


# ---------------------------------------------------------------------------
# Test 2: a problem the initial library + in-solve clamp CAN handle still
# solves under A4. The freeze is a POST-SOLVE rollback, not a refusal of
# the in-loop clamp -- so any problem that the BASELINE solves also
# solves under A4 (with the same residual structure). This is the §1.1
# "binding = entanglement clamp, not classical lookup" guarantee.
# ---------------------------------------------------------------------------


def test_a4_solves_problems_solvable_from_initial_library(lib, states):
    """A converging substrate-supported theorem solved by BASELINE also
    solves under A4 freeze. The freeze only ROLLS BACK persisted
    lemmas; the in-solve clamp + decomposition still fires.
    """
    state, meta = states
    # A single-level decomposition (Thm -> two leaves) is the same shape
    # used by the existing orchestrator acceptance tests; both BASELINE
    # and A4 should converge identically because the freeze is a
    # post-solve effect.
    table = {
        "Thm": [({"g": "L1a"}, "LemmaA"), ({"g": "L1b"}, "LemmaB")],
    }

    result = solve_goal_graph(
        {"g": "root_a4_solve"}, root_prop="Thm",
        decomposer=_StubDecomposer(table),
        backend=ThreadPoolBackend(max_workers=2),
        lemma_library=lib,
        runner=_converging_runner(state, meta), timeout_s=5.0,
        parent_state=state, parent_meta=meta,
        freeze_library=True,
    )
    assert isinstance(result, SolveResult)
    assert result.solved is True, (
        f"A4 freeze should not prevent the solve from converging on a "
        f"substrate-supported theorem; got failure_report="
        f"{result.failure_report!r}"
    )
    assert result.proof_tree is not None
    assert result.proof_tree.root.goal_prop == "Thm"
    # Post-condition: the library is in its initial state (the solve
    # registered children mid-flight, but the freeze guard rolled them
    # back). This is the same invariant as test_a4_freezes_lemma_library
    # but exercised on the SOLVE-SUCCEEDS path.
    assert lib.all_ids() == [], (
        "freeze_library=True must leave an initially-empty library empty "
        f"after the solve; got {lib.all_ids()!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: the ablation matrix exposes A4 as wired and tags the
# freeze_library diagnostic on every A4 attempt.
# ---------------------------------------------------------------------------


def test_a4_config_is_wired_and_tagged():
    """The A4 row of ABLATION_CONFIGS is now ``wired=True`` and the
    ``_apply_ablation_to_attempt`` helper tags ``freeze_library=True``
    on the resulting diagnostics.
    """
    a4 = next(cfg for cfg in ABLATION_CONFIGS if cfg.label == "A4")
    assert a4.wired is True


def test_a4_ablation_matrix_row_carries_freeze_diagnostic():
    """End-to-end: ``run_ablation_matrix`` produces an A4 row whose
    attempts carry ``freeze_library=True`` in their diagnostics. The
    runner builds a ``QPCNBaseline(freeze_library=True)`` for the A4
    row, so the wiring is observable from the BenchmarkResult alone.
    """
    problems = load_minif2f(limit=1)
    rows = run_ablation_matrix(
        problems,
        benchmark_label="minif2f",
        solver_factory=lambda: QPCNBaseline(),
    )
    a4_row = next(r for r in rows if r.config_label == "A4")
    assert len(a4_row.attempts) == 1
    for attempt in a4_row.attempts:
        assert attempt.diagnostics.get("freeze_library") is True
        # Wired row must NOT carry the not_yet_wired marker -- that
        # would be an honesty regression (§1.6).
        assert attempt.diagnostics.get("not_yet_wired") is not True
