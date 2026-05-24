"""Tests for the metric functions."""
from __future__ import annotations

import math

import pytest

from experiments.metrics import (
    capability_curve, pass_at_k, residual_energy_at_1, type_at_k,
)
from experiments.metrics.capability_curve import capability_vs_difficulty
from experiments.schema import ProblemSpec, ProofAttempt


def _att(pid, *, solved=True, well_typed=True, residual=0.0, candidates=("x",)):
    return ProofAttempt(
        solver="qpcn", problem_id=pid, solved=solved,
        well_typed=well_typed, residual_energy=residual,
        candidates=candidates, wall_time_s=0.1,
    )


def test_pass_at_k_simple_semantic():
    atts = [_att("p1"), _att("p2", solved=False),
            _att("p3"), _att("p4", solved=False)]
    assert pass_at_k(atts, k=1) == 0.5
    assert pass_at_k(atts, k=10) == 0.5


def test_pass_at_k_unbiased_estimator_matches_humaneval_paper():
    # n=10, c=5, k=1 -> 1 - C(5,1)/C(10,1) = 1 - 0.5 = 0.5
    atts = [ProofAttempt(
        solver="x", problem_id="p1", solved=True, well_typed=True,
        residual_energy=0.0, candidates=("a",), wall_time_s=0.0,
        diagnostics={"n_samples": 10, "n_passing": 5},
    )]
    assert pass_at_k(atts, k=1, unbiased=True) == pytest.approx(0.5)


def test_pass_at_k_rejects_zero_k():
    with pytest.raises(ValueError):
        pass_at_k([], k=0)


def test_type_at_k_all_typed_gives_1():
    atts = [_att("p1"), _att("p2"), _att("p3")]
    assert type_at_k(atts, k=1) == 1.0


def test_type_at_k_mixed():
    atts = [_att("p1", well_typed=True),
            _att("p2", well_typed=False),
            _att("p3", well_typed=True),
            _att("p4", well_typed=False)]
    assert type_at_k(atts, k=1) == 0.5


def test_residual_energy_at_1_excludes_none():
    atts = [_att("p1", residual=0.0),
            _att("p2", residual=0.1),
            ProofAttempt(solver="x", problem_id="p3", solved=False,
                         well_typed=False, residual_energy=None,
                         candidates=(), wall_time_s=0.0)]
    s = residual_energy_at_1(atts)
    assert s.n == 2
    assert s.mean == pytest.approx(0.05)
    assert math.isfinite(s.std)


def test_capability_curve_per_cycle():
    cycle0 = [_att("a", solved=False), _att("b", solved=False)]
    cycle1 = [_att("a", solved=True), _att("b", solved=False)]
    cycle2 = [_att("a", solved=True), _att("b", solved=True)]
    curve = capability_curve([cycle0, cycle1, cycle2])
    assert curve == (0.0, 0.5, 1.0)


def test_capability_vs_difficulty_bucketing():
    problems = [
        ProblemSpec(domain="synthesis", problem_id="easy",
                    statement="", payload={}, difficulty=0.1),
        ProblemSpec(domain="synthesis", problem_id="hard",
                    statement="", payload={}, difficulty=0.9),
    ]
    atts = [_att("easy", solved=True), _att("hard", solved=False)]
    points = capability_vs_difficulty(problems, atts, n_buckets=2)
    # first bucket (difficulty < 0.5) should be 1.0; second 0.0.
    assert points[0].pass_rate == 1.0
    assert points[1].pass_rate == 0.0
