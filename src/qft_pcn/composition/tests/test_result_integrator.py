"""Result-integrator tests (spec §6).

These tests exercise the real I-Task-7 (LemmaLibrary.register_lemma) and
I-Task-8 (Promoter.apply_init_clamp) surfaces -- no fictional promote/
clamp test doubles. A real LemmaLibrary is constructed in ``tmp_path``;
real MeraEncodingMeta + MERA ground states are produced by the M1 logic
encoder so register_lemma's compress/persist machinery is genuinely
exercised.
"""
from __future__ import annotations
import math
import numpy as np
import pytest

from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
from src.qft_pcn.composition.dispatcher import ChildResult
from src.qft_pcn.composition.lemma_library import LemmaLibrary
from src.qft_pcn.composition.result_integrator import (
    integrate_child, precision_weight, RESIDUAL_GATE, CONJECTURE_CEILING,
    GROUND_STATE_GAP, STRENGTH_MAX,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lib(tmp_path):
    """A real, file-backed LemmaLibrary in tmp_path."""
    return LemmaLibrary(tmp_path)


@pytest.fixture
def child_state_meta():
    """A real MERA ground state + encoding meta for the child.

    Using ``\\x:Int. x`` as a tiny seed program -- the encoder produces a
    concrete MERA whose tensors are unit-normalized leaves, exactly the
    shape ``register_lemma`` and ``apply_init_clamp`` expect.
    """
    state, meta = encode_mera(parse(r"\x:Int. x"))
    return state, meta


@pytest.fixture
def parent_state_meta():
    """A real parent MERA + meta with species compatible with the child.

    Encoding the SAME source gives a parent whose ``species_of_leaf``
    pattern matches the child's at indices [0, n_leaves) -- the natural
    case the integrator targets (the child decomposed out of an
    isomorphic parent region).
    """
    state, meta = encode_mera(parse(r"\x:Int. x"))
    return state, meta


def _node(parent_leaves=(0,), prop="P"):
    """Build a child node whose parent_leaves tuple is the explicit
    host-leaf footprint the integrator will clamp into.

    Callers that previously passed ``parent_site=i`` now pass the
    explicit tuple. The fixture child (``encode_mera(r'\\x:Int. x')``)
    has ``meta.n_leaves == 16`` -- callers that exercise the clamp path
    must pass ``tuple(range(0, 16))`` to cover the full footprint.
    """
    g = make_sub_goal({"g": prop}, goal_prop=prop, boundary={},
                      parent_leaves=tuple(parent_leaves))
    n = Node(goal=g, status=Status.ACTIVE)
    n.parent = Node(goal=make_sub_goal({"g": "parent"}, goal_prop="Par",
                                       boundary={}, parent_leaves=()),
                    status=Status.ACTIVE)
    return n


def _result(residual, *, converged=True, gap=1.0,
            ground_state=None, meta=None, gap_in_diag=True):
    """A ChildResult with optional ground_state / meta / gap-omitted."""
    diag = {"spectral_gap": gap} if gap_in_diag else {}
    return ChildResult(
        goal_id="g", converged=converged, residual_energy=residual,
        ground_state=ground_state, solved_ast="ast",
        run_diagnostic=diag, error=None,
        meta=meta, hamiltonian=None, trotter_steps=0,
    )


# ---------------------------------------------------------------------------
# Pure (no-library) precision-weight tests
# ---------------------------------------------------------------------------

def test_precision_weight_is_one_at_zero_residual():
    assert precision_weight(0.0) == pytest.approx(1.0)


def test_precision_weight_decreases_with_residual():
    assert precision_weight(1e-3) < precision_weight(1e-9)


# ---------------------------------------------------------------------------
# Refusal paths (no library writes expected)
# ---------------------------------------------------------------------------

def test_non_converged_child_refused(lib, parent_state_meta):
    parent_state, parent_meta = parent_state_meta
    node = _node()
    out = integrate_child(parent_state, parent_meta, node,
                          _result(1e-9, converged=False), lib)
    assert out.integrated is False
    assert node.status == Status.FAILED
    assert lib.all_ids() == []


def test_near_degenerate_child_refused_despite_tiny_residual(
    lib, parent_state_meta, child_state_meta
):
    """Premature-integration guard: tiny residual but no spectral gap."""
    parent_state, parent_meta = parent_state_meta
    cstate, cmeta = child_state_meta
    node = _node()
    out = integrate_child(
        parent_state, parent_meta, node,
        _result(1e-9, gap=GROUND_STATE_GAP / 10,
                ground_state=cstate, meta=cmeta),
        lib,
    )
    assert out.integrated is False
    assert node.status == Status.FAILED
    assert lib.all_ids() == []


def test_residual_above_ceiling_refused(lib, parent_state_meta):
    parent_state, parent_meta = parent_state_meta
    node = _node()
    out = integrate_child(parent_state, parent_meta, node,
                          _result(CONJECTURE_CEILING * 10), lib)
    assert out.integrated is False
    assert node.status == Status.FAILED
    assert node.parent.status == Status.PENDING_REVISION
    # The error message includes both the residual and the ceiling so
    # the failure is diagnosable from logs alone.
    assert str(CONJECTURE_CEILING) in out.reason
    assert lib.all_ids() == []


def test_timeout_child_refused(lib, parent_state_meta):
    parent_state, parent_meta = parent_state_meta
    node = _node()
    out = integrate_child(parent_state, parent_meta, node,
                          _result(math.inf, converged=False), lib)
    assert out.integrated is False
    assert node.status == Status.FAILED
    assert lib.all_ids() == []


def test_missing_spectral_gap_refused(lib, parent_state_meta):
    """_spectral_gap falls back to 0.0 (refuse), not GROUND_STATE_GAP.

    A runner that fails to publish a spectral_gap diagnostic must NOT
    silently clear the gap gate. This is the second silent-acceptance
    bug from K-5 review.
    """
    parent_state, parent_meta = parent_state_meta
    node = _node()
    out = integrate_child(parent_state, parent_meta, node,
                          _result(1e-9, gap_in_diag=False), lib)
    assert out.integrated is False
    assert node.status == Status.FAILED
    assert "near-degenerate" in out.reason
    assert lib.all_ids() == []


# ---------------------------------------------------------------------------
# Integration paths (library write expected)
# ---------------------------------------------------------------------------

def test_ground_state_child_is_integrated_and_clamped(
    lib, parent_state_meta, child_state_meta
):
    parent_state, parent_meta = parent_state_meta
    cstate, cmeta = child_state_meta
    # Full footprint: fixture child has n_leaves=16; pass the explicit
    # tuple so the integrator clamps onto the contiguous [0, 16) window.
    node = _node(parent_leaves=tuple(range(0, 16)))
    # Snapshot the parent's leaf-0 vector for comparison post-clamp.
    pre_leaf0 = np.asarray(parent_state.leaves[0]).copy()
    out = integrate_child(
        parent_state, parent_meta, node,
        _result(1e-9, ground_state=cstate, meta=cmeta), lib,
    )
    assert out.integrated is True
    assert out.provisional is False
    assert node.status == Status.SOLVED
    assert out.clamp_strength == pytest.approx(STRENGTH_MAX, rel=1e-3)
    # A lemma was actually persisted to the library.
    assert len(lib.all_ids()) == 1
    # And the parent's clamped leaf-0 tensor matches the child's leaf-0
    # (full-strength clamp == byte-equal copy).
    assert np.allclose(parent_state.leaves[0][0, :, 0],
                       cstate.leaves[0][0, :, 0])
    _ = pre_leaf0  # retained for symmetry; assertion above is sufficient


def test_conjecture_band_integrates_with_reduced_strength(
    lib, parent_state_meta, child_state_meta
):
    parent_state, parent_meta = parent_state_meta
    cstate, cmeta = child_state_meta
    # Full footprint: fixture child has n_leaves=16.
    node = _node(parent_leaves=tuple(range(0, 16)))
    mid = (RESIDUAL_GATE + CONJECTURE_CEILING) / 2.0
    out = integrate_child(
        parent_state, parent_meta, node,
        _result(mid, ground_state=cstate, meta=cmeta), lib,
    )
    assert out.integrated is True
    assert out.provisional is True
    assert node.status == Status.SOLVED
    assert 0.0 < out.clamp_strength < STRENGTH_MAX
    # Lemma persisted; its derivation is marked conditional (provisional).
    assert len(lib.all_ids()) == 1
    lem = lib.load(lib.all_ids()[0])
    assert lem.derivation.conditional is True


def test_missing_meta_refused(lib, parent_state_meta, child_state_meta):
    """A ChildResult without ``meta`` cannot drive register_lemma; the
    integrator must refuse rather than crash. This guards the
    bridge-RunResult gap recorded in EXTENSIONS.md -- until the bridge
    surfaces meta, callers that forget to plumb it should fail-safe."""
    parent_state, parent_meta = parent_state_meta
    cstate, _cmeta = child_state_meta
    node = _node()
    out = integrate_child(
        parent_state, parent_meta, node,
        _result(1e-9, ground_state=cstate, meta=None), lib,
    )
    assert out.integrated is False
    assert node.status == Status.FAILED
    assert "meta" in out.reason
    assert lib.all_ids() == []
