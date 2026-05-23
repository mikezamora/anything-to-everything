"""Orchestrator parent-workspace tests (spec §6.1 / §8.6, §1.1 binding).

The orchestrator must OWN (accept) a parent MERA + meta and drive
``Promoter.apply_init_clamp`` on every solved sub-goal -- the §1.1
architecture-soul binding IS the entanglement clamp into the parent
network. These tests pin that the clamp ACTUALLY FIRES through the
orchestrator entry point (not only through the integrator unit tests).

Resolves EXTENSIONS.md "Missing dependency: orchestrator does not own
a parent MERA" (§6.1 / §8.6).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.qft_pcn.composition.dispatcher import ChildResult, ThreadPoolBackend
from src.qft_pcn.composition.goal_graph import make_sub_goal
from src.qft_pcn.composition.lemma_library import LemmaLibrary
from src.qft_pcn.composition.orchestrator import (
    SolveResult,
    solve_goal_graph,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


# ---------------------------------------------------------------------------
# Opt this file out of the conftest §9.7 dense-tensor ceiling. The real
# encode_mera allocates 16-dim leaves + 16-up isometries (4096-element pair
# matrices) by construction. (Same pattern as test_cross_level_acceptance.py.)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadows the conftest fixture for this file only
    yield


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def lib(tmp_path):
    return LemmaLibrary(tmp_path)


@pytest.fixture
def child_state_meta():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    return state, meta


@pytest.fixture
def parent_state_meta():
    """A parent MERA isomorphic to the child fixture so the species
    pattern matches at the clamped window."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    return state, meta


def _make_single_child_decomposer(n_leaves: int):
    """Decomposes the root ``Thm`` into ONE leaf child whose
    ``parent_leaves`` is the leading host-leaf window of width
    ``n_leaves``. Other propositions decompose to nothing.
    """
    window = tuple(range(0, n_leaves))

    class _SingleChildDecomposer:
        def decompose(self, node):
            if node.goal.goal_prop != "Thm":
                return []
            return [make_sub_goal(
                {"g": "child"}, goal_prop="ChildLeaf", boundary={},
                parent_leaves=window,
            )]
    return _SingleChildDecomposer()


def _make_runner(cstate, cmeta, *, residual_energy: float = 0.0):
    def runner(sub_goal, timeout_s):
        return ChildResult(
            goal_id=sub_goal.goal_id,
            converged=True,
            residual_energy=residual_energy,
            ground_state=cstate,
            solved_ast=f"ast::{sub_goal.goal_prop}",
            run_diagnostic={"spectral_gap": 1.0},
            error=None,
            meta=cmeta,
            hamiltonian=None,
            trotter_steps=0,
        )
    return runner


def _scramble_child_leaves(cstate_real, cmeta):
    """Return a deep-copied child MERA whose non-PAD leaves are the |1>
    basis vector -- a value clearly distinct from the parent fixture's
    leaves (which encode the identity AST). A successful clamp will
    overwrite the parent's window with these basis vectors, producing a
    bitwise observable change.
    """
    cstate = copy.deepcopy(cstate_real)
    for i in range(cmeta.n_leaves):
        if cmeta.species_of_leaf[i] == "PAD":
            continue
        d = cstate.leaves[i].shape[1]
        new_leaf = np.zeros((1, d, 1), dtype=complex)
        new_leaf[0, 1, 0] = 1.0
        cstate.leaves[i] = new_leaf
    return cstate


# ---------------------------------------------------------------------------
# Test 1: the clamp ACTUALLY FIRES on the parent state (bitwise change).
# ---------------------------------------------------------------------------


def test_orchestrator_clamps_lemma_into_parent_mera(
    lib, child_state_meta, parent_state_meta,
):
    """A SubGoal whose ground-state DIFFERS from the parent's leaves
    (at the clamped window) must mutate ``parent_state.leaves`` bitwise
    through the orchestrator entry point. Proves
    ``Promoter.apply_init_clamp`` fires (§1.1 / §6.1 binding).
    """
    cstate_real, cmeta = child_state_meta
    pstate, pmeta = parent_state_meta

    cstate = _scramble_child_leaves(cstate_real, cmeta)
    pre_leaves = [np.array(l, copy=True) for l in pstate.leaves]

    runner = _make_runner(cstate, cmeta, residual_energy=0.0)
    result = solve_goal_graph(
        {"g": "root"}, root_prop="Thm",
        decomposer=_make_single_child_decomposer(cmeta.n_leaves),
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=lib,
        runner=runner, timeout_s=5.0,
        parent_state=pstate, parent_meta=pmeta,
    )

    assert isinstance(result, SolveResult)
    assert result.solved is True, (
        f"orchestrator did not solve; failure_report={result.failure_report}"
    )

    # The clamp window is [0, cmeta.n_leaves). At least one non-PAD leaf
    # in that range must differ bitwise from pre-state; otherwise
    # apply_init_clamp was silently skipped (the §1.1 binding failure
    # EXTENSIONS.md warned about). At residual=0 the strength is 1.0
    # which short-circuits to a direct copy, so the post-clamp leaf
    # equals the cached child leaf exactly.
    changed_any = False
    for i in range(cmeta.n_leaves):
        if cmeta.species_of_leaf[i] == "PAD":
            continue
        post = np.asarray(pstate.leaves[i])
        if not np.array_equal(post, pre_leaves[i]):
            changed_any = True
            assert np.array_equal(post, cstate.leaves[i]), (
                f"leaf {i}: post-clamp value differs from cached child "
                f"-- clamp wrote something other than the lemma tensor"
            )
    assert changed_any, (
        "Promoter.apply_init_clamp did not fire: parent leaves are "
        "bitwise unchanged through the orchestrator entry point. "
        "This is the §1.1 / §6.1 binding failure."
    )


# ---------------------------------------------------------------------------
# Test 2: refusing the call without a parent workspace (no graceful skip).
# ---------------------------------------------------------------------------


def test_orchestrator_refuses_without_parent_workspace(lib):
    """``parent_state=None`` / ``parent_meta=None`` must raise; the
    orchestrator never falls back to lemma-registration-only (anti-
    shortcut: no graceful skip)."""
    def runner(sub_goal, timeout_s):  # pragma: no cover -- never reached
        raise AssertionError("runner should not be invoked")

    class _NoDecomposer:
        def decompose(self, node):
            return []

    with pytest.raises(TypeError, match="parent workspace"):
        solve_goal_graph(
            {"g": "root"}, root_prop="Leaf",
            decomposer=_NoDecomposer(),
            backend=ThreadPoolBackend(max_workers=1),
            lemma_library=lib,
            runner=runner, timeout_s=1.0,
            parent_state=None, parent_meta=None,
        )


# ---------------------------------------------------------------------------
# Test 3: clamp is local -- leaves OUTSIDE the parent_leaves window stay
# bitwise unchanged (spec §1.3 factored op).
# ---------------------------------------------------------------------------


def test_orchestrator_preserves_unclamped_leaves(
    lib, child_state_meta, parent_state_meta,
):
    """Every host leaf OUTSIDE the parent_leaves tuple
    window must be bitwise identical to its pre-clamp value -- the
    §1.3 factored-op locality invariant."""
    cstate_real, cmeta = child_state_meta
    pstate, pmeta = parent_state_meta

    cstate = _scramble_child_leaves(cstate_real, cmeta)

    # Append a sentinel leaf to the parent that the clamp must NOT touch
    # (it's outside the window [0, cmeta.n_leaves)). With the parent and
    # child fixtures sharing n_leaves, the natural "outside" set on the
    # parent is empty; this sentinel makes the non-overlap check
    # non-vacuous.
    d = pstate.leaves[0].shape[1]
    sentinel = np.zeros((1, d, 1), dtype=complex)
    sentinel[0, 2, 0] = 1.0
    sentinel_idx = len(pstate.leaves)
    pstate.leaves.append(np.array(sentinel, copy=True))
    sentinel_pre = np.array(sentinel, copy=True)

    runner = _make_runner(cstate, cmeta, residual_energy=0.0)
    result = solve_goal_graph(
        {"g": "root"}, root_prop="Thm",
        decomposer=_make_single_child_decomposer(cmeta.n_leaves),
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=lib,
        runner=runner, timeout_s=5.0,
        parent_state=pstate, parent_meta=pmeta,
    )

    assert result.solved is True

    post_sentinel = np.asarray(pstate.leaves[sentinel_idx])
    assert np.array_equal(post_sentinel, sentinel_pre), (
        "clamp touched a host leaf outside the parent_leaves window -- "
        "§1.3 factored-op locality invariant violated."
    )
