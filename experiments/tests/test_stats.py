"""Tests for the statistical-protocol modules."""
from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.stats.bootstrap import bootstrap_ci
from experiments.stats.effect_size import cliffs_delta, cohens_d
from experiments.stats.multiple_comparisons import (
    bonferroni, holm_bonferroni,
)


def test_bootstrap_ci_contains_mean_of_constant_sample():
    ci = bootstrap_ci([0.5] * 20, n_resamples=200, seed=0)
    assert ci.point == 0.5
    assert ci.lower == 0.5
    assert ci.upper == 0.5


def test_bootstrap_ci_handles_empty():
    ci = bootstrap_ci([], n_resamples=10)
    assert math.isnan(ci.point)
    assert ci.n_observations == 0


def test_bootstrap_ci_brackets_sample_mean():
    rng = np.random.default_rng(7)
    sample = rng.normal(loc=2.0, scale=1.0, size=200).tolist()
    sample_mean = float(np.mean(sample))
    ci = bootstrap_ci(sample, n_resamples=500, seed=7)
    # Percentile bootstrap CI brackets the OBSERVED sample mean (point
    # estimate); the population mean need not lie inside on a given draw.
    assert ci.point == pytest.approx(sample_mean)
    assert ci.lower < sample_mean < ci.upper


def test_cohens_d_zero_variance_returns_zero():
    assert cohens_d([1, 1, 1], [1, 1, 1]) == 0.0


def test_cohens_d_positive_when_a_greater():
    d = cohens_d([2, 2, 3, 3], [0, 0, 1, 1])
    assert d > 0.5


def test_cliffs_delta_extreme():
    assert cliffs_delta([5, 5, 5], [0, 0, 0]) == 1.0
    assert cliffs_delta([0, 0, 0], [5, 5, 5]) == -1.0


def test_holm_bonferroni_monotone():
    p = [("a", 0.001), ("b", 0.01), ("c", 0.05)]
    out = holm_bonferroni(p, alpha=0.05)
    adjusted = sorted((cp.label, cp.adjusted_p) for cp in out)
    # all adjusted >= raw
    for cp in out:
        raw = dict(p)[cp.label]
        assert cp.adjusted_p >= raw - 1e-12


def test_bonferroni_caps_at_one():
    out = bonferroni([("a", 0.9), ("b", 0.9), ("c", 0.9)])
    for cp in out:
        assert cp.adjusted_p <= 1.0
