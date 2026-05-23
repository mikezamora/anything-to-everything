"""Result-integrator tests (spec §6)."""
from __future__ import annotations
import math
import pytest
from src.qft_pcn.composition.goal_graph import make_sub_goal, Node, Status
from src.qft_pcn.composition.dispatcher import ChildResult
from src.qft_pcn.composition.result_integrator import (
    integrate_child, precision_weight, RESIDUAL_GATE, CONJECTURE_CEILING,
    GROUND_STATE_GAP, STRENGTH_MAX,
)


class FakeLemmaLibrary:
    """Test double for sub-project I. Records promote + clamp calls."""
    def __init__(self):
        self.promoted = []
        self.clamps = []

    def promote(self, ground_state, solved_ast, goal_id):
        self.promoted.append(goal_id)
        return {"lemma_id": goal_id, "state": ground_state}

    def clamp(self, parent_state, parent_site, lemma, *, strength):
        self.clamps.append((parent_site, strength))
        return parent_state  # the clamped MERA (mutated copy in real I)


def _node(prop="P", site=3):
    g = make_sub_goal({"g": prop}, goal_prop=prop, boundary={}, parent_site=site)
    n = Node(goal=g, status=Status.ACTIVE)
    n.parent = Node(goal=make_sub_goal({"g": "parent"}, goal_prop="Par",
                                       boundary={}, parent_site=None),
                    status=Status.ACTIVE)
    return n


def _result(residual, *, converged=True, gap=1.0):
    return ChildResult(goal_id="g", converged=converged,
                       residual_energy=residual, ground_state=object(),
                       solved_ast="ast", run_diagnostic={"spectral_gap": gap},
                       error=None)


def test_precision_weight_is_one_at_zero_residual():
    assert precision_weight(0.0) == pytest.approx(1.0)


def test_precision_weight_decreases_with_residual():
    assert precision_weight(1e-3) < precision_weight(1e-9)


def test_ground_state_child_is_integrated_and_clamped():
    lib = FakeLemmaLibrary()
    node = _node(site=5)
    out = integrate_child("parent_mera", {}, node, _result(1e-8), lib)
    assert node.status == Status.SOLVED
    assert out.provisional is False
    assert lib.promoted == ["g"]
    assert lib.clamps[0][0] == 5                       # clamped at parent_site
    assert lib.clamps[0][1] == pytest.approx(STRENGTH_MAX, rel=1e-3)


def test_conjecture_band_integrates_with_reduced_strength():
    lib = FakeLemmaLibrary()
    node = _node()
    mid = (RESIDUAL_GATE + CONJECTURE_CEILING) / 2.0
    out = integrate_child("parent_mera", {}, node, _result(mid), lib)
    assert node.status == Status.SOLVED
    assert out.provisional is True
    assert lib.clamps[0][1] < STRENGTH_MAX             # reduced clamp strength


def test_residual_above_ceiling_refused():
    lib = FakeLemmaLibrary()
    node = _node()
    out = integrate_child("parent_mera", {}, node, _result(CONJECTURE_CEILING * 10),
                          lib)
    assert node.status == Status.FAILED
    assert node.parent.status == Status.PENDING_REVISION
    assert lib.clamps == []                            # no clamp applied


def test_non_converged_child_refused():
    lib = FakeLemmaLibrary()
    node = _node()
    out = integrate_child("parent_mera", {}, node,
                          _result(1e-9, converged=False), lib)
    assert node.status == Status.FAILED
    assert lib.clamps == []


def test_near_degenerate_child_refused_despite_tiny_residual():
    # premature-integration guard: tiny residual but no spectral gap
    lib = FakeLemmaLibrary()
    node = _node()
    out = integrate_child("parent_mera", {}, node,
                          _result(1e-9, gap=GROUND_STATE_GAP / 10), lib)
    assert node.status == Status.FAILED
    assert lib.clamps == []


def test_timeout_child_refused():
    lib = FakeLemmaLibrary()
    node = _node()
    out = integrate_child("parent_mera", {}, node,
                          _result(math.inf, converged=False), lib)
    assert node.status == Status.FAILED
    assert lib.clamps == []
