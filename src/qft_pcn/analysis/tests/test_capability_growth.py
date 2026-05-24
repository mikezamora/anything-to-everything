"""Tests for §13.8 capability-growth-law fitter (A4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.analysis.capability_growth import (
    GrowthFit,
    capability_curve,
    fit_growth_law,
    measure_capability,
    record_growth_trajectory,
)
from src.qft_pcn.composition.tests.conftest import (
    FakeLemmaLibrary, build_induction_corpus, make_stub_solver,
)


# ---------------------------------------------------------------------------
# closed-form sanity
# ---------------------------------------------------------------------------


def test_capability_curve_asymptote():
    """C(t->inf) -> alpha/(alpha+beta) (the §13.8 asymptotic capability)."""
    alpha, beta, c0 = 0.6, 0.2, 0.0
    t = np.array([0.0, 1e6])
    c = capability_curve(t, alpha, beta, c0)
    assert c[0] == pytest.approx(c0)
    assert c[-1] == pytest.approx(alpha / (alpha + beta), rel=1e-6)


def test_capability_curve_zero_rate_limit():
    """alpha + beta -> 0 reduces to the linear regime C(t) = c0 + alpha*t."""
    alpha, beta, c0 = 0.0, 0.0, 0.3
    t = np.array([0.0, 1.0, 5.0])
    c = capability_curve(t, alpha, beta, c0)
    np.testing.assert_allclose(c, np.full_like(t, c0))


# ---------------------------------------------------------------------------
# synthetic-trajectory recovery
# ---------------------------------------------------------------------------


def _synthetic_trajectory(alpha: float, beta: float, c0: float,
                          ts: np.ndarray) -> list[tuple[float, float]]:
    cs = capability_curve(ts, alpha, beta, c0)
    return list(zip(ts.tolist(), cs.tolist()))


@pytest.mark.parametrize(
    "alpha,beta,c0",
    [(0.50, 0.10, 0.00),
     (0.30, 0.20, 0.05),
     (0.80, 0.40, 0.10)],
)
def test_fit_recovers_known_alpha_beta(alpha, beta, c0):
    """The fitter recovers (alpha, beta) within 5% on a clean synthetic
    trajectory of the closed-form curve (spec A4 acceptance criterion).
    """
    ts = np.linspace(0.0, 10.0, 21)
    traj = _synthetic_trajectory(alpha, beta, c0, ts)
    fit = fit_growth_law(traj)
    assert fit.alpha == pytest.approx(alpha, rel=0.05, abs=1e-3)
    assert fit.beta == pytest.approx(beta, rel=0.05, abs=1e-3)
    # Closed-form trajectory has zero residual at convergence.
    assert fit.ssr < 1e-10
    assert fit.c0 == pytest.approx(c0)
    assert fit.c_infinity == pytest.approx(alpha / (alpha + beta), rel=1e-6)
    assert fit.n_points == len(ts)


def test_fit_rejects_singleton_trajectory():
    with pytest.raises(ValueError):
        fit_growth_law([(0.0, 0.0)])


# ---------------------------------------------------------------------------
# real wake-sleep trajectory smoke (operator-algebraic, §1.1 binding)
# ---------------------------------------------------------------------------


def test_measure_capability_empty_library_is_zero():
    corpus = build_induction_corpus()
    lib = FakeLemmaLibrary()
    assert measure_capability(lib, corpus) == 0.0


def test_capability_grows_after_one_cycle():
    """After one wake-sleep cycle the library should solve every problem
    in the corpus (either via cache-hit on the wake-phase register or via
    the discovered induction primitive's subsume match)."""
    corpus = build_induction_corpus()
    lib = FakeLemmaLibrary()
    traj = record_growth_trajectory(
        lib, corpus, n_cycles=1, solve=make_stub_solver({})
    )
    assert traj[0] == (0, 0.0)
    assert traj[1][0] == 1
    # Wake phase registered each (state, meta, sid) under its own id, so
    # capability after cycle 0 is 1.0 (every problem id is cached).
    assert traj[1][1] == pytest.approx(1.0)


def test_real_short_trajectory_fits_growth_law():
    """End-to-end: a 5-cycle wake-sleep trajectory + scipy fit returns a
    finite (alpha, beta) and non-negative ssr. This is the integration
    test that the §13.8 conjecture pipeline runs against the real §10.9
    substrate (operator-algebraic matching, no classical lookup).
    """
    corpus = build_induction_corpus()
    lib = FakeLemmaLibrary()
    traj = record_growth_trajectory(
        lib, corpus, n_cycles=5, solve=make_stub_solver({})
    )
    assert len(traj) == 6
    fit = fit_growth_law(traj)
    assert isinstance(fit, GrowthFit)
    assert np.isfinite(fit.alpha) and fit.alpha >= 0.0
    assert np.isfinite(fit.beta) and fit.beta >= 0.0
    assert fit.ssr >= 0.0
