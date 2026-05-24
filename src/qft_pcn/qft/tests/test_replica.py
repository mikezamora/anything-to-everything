"""Tests for the replica analytic-continuation tooling (substrate S3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.qft_pcn.qft.replica import (
    analytic_continuation_at_zero,
    compute_log_z_from_replicas,
    compute_zn_for_ensemble,
)


# --------------------------------------------------------------------------
# 1. Known-log recovery: engineered <Z^n> = exp(n * c) => <log Z> = c.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("c", [-0.5, -0.3, 0.0, 0.3])
def test_analytic_continuation_recovers_known_log(c: float) -> None:
    """For a delta-ensemble at Z = exp(c), <Z^n> = exp(n c) and
    <log Z> = c.  Replica continuation through the (Z^n - 1)/n
    auxiliary should recover c.

    Restricted to |c| <~ 1 — the standard replica-trick regime.
    Large |c| pushes <Z^n> into the rapidly-growing tail where
    polynomial extrapolation from positive-integer n to n=0 is
    fundamentally ill-conditioned (this is the well-known
    'analytic continuation in n is delicate' caveat from §12.7).
    """
    n_grid = [1, 2, 3, 4, 5, 6]
    zn = [math.exp(c * n) for n in n_grid]

    estimate = analytic_continuation_at_zero(zn, n_grid)

    assert estimate == pytest.approx(c, abs=5e-3), (
        f"replica continuation gave {estimate}, expected {c}"
    )


def test_compute_log_z_from_replicas_matches_direct_call() -> None:
    c = 0.25
    n_grid = [1, 2, 3, 4, 5, 6]
    zn = {n: math.exp(c * n) for n in n_grid}
    assert compute_log_z_from_replicas(zn) == pytest.approx(c, abs=5e-3)


def test_replica_regime_limitation_documented() -> None:
    """Document the well-known limitation: polynomial extrapolation from
    positive-integer n to n=0 is well-conditioned only when <Z^n> stays
    moderate over the sample grid.  For |log Z| large enough that
    <Z^n> ~ exp(n |log Z|) explodes by the largest n_grid, the
    continuation error grows uncontrolled.  This mirrors the §12.7 risk
    note ('analytic continuation n -> 0 not well-defined for the
    specific ensemble' — addressed by restricting to the canonical
    replica-trick regime here).
    """
    # c = 1.0 with grid 1..6 -> <Z^6> = e^6 ~ 403; severely
    # ill-conditioned.  We assert only that the result is *finite*
    # (no NaN/inf) — accuracy is not promised in this regime.
    c = 1.0
    n_grid = [1, 2, 3, 4, 5, 6]
    zn = [math.exp(c * n) for n in n_grid]
    estimate = analytic_continuation_at_zero(zn, n_grid)
    assert math.isfinite(estimate)
    # And the answer is *not* expected to be close to c — that's the
    # point.  We pin the qualitative direction (wrong sign / large bias)
    # to make sure we notice if some accidental fix masks the issue.
    assert abs(estimate - c) > 0.5


# --------------------------------------------------------------------------
# 2. Polynomial interpolation is exact for polynomials of matching degree.
# --------------------------------------------------------------------------


def test_polynomial_interp_handles_finite_grid() -> None:
    """For a valid replica generating function P(n) = <Z^n> with
    P(0) = 1, the continuation should recover the linear coefficient
    exactly (up to floating-point noise) when the sample grid is at
    least the polynomial degree + 1.

    P(n) = 1 + 3 n + 2 n^2 - 0.5 n^3 + 0.1 n^4
    P(0) = 1, P'(0) = 3 => <log Z> = 3.
    """
    coeffs_low_to_high = [1.0, 3.0, 2.0, -0.5, 0.1]
    n_grid = [1, 2, 3, 4, 5]

    def P(n: int) -> float:
        return sum(c * (n ** k) for k, c in enumerate(coeffs_low_to_high))

    zn = [P(n) for n in n_grid]

    estimate = analytic_continuation_at_zero(zn, n_grid)
    assert estimate == pytest.approx(3.0, abs=1e-6)


def test_polynomial_interp_with_numpy_path() -> None:
    """Force the numpy.polyfit fallback (use_sympy=False) and check the
    same recovery as the sympy path."""
    # P(n) = 1 - 2 n + 0.5 n^2; valid replica form (P(0)=1).
    coeffs_low_to_high = [1.0, -2.0, 0.5]
    n_grid = [1, 2, 3]
    P = lambda n: sum(c * (n ** k) for k, c in enumerate(coeffs_low_to_high))
    zn = [P(n) for n in n_grid]
    estimate = analytic_continuation_at_zero(zn, n_grid, use_sympy=False)
    assert estimate == pytest.approx(-2.0, abs=1e-9)


# --------------------------------------------------------------------------
# 3. Z = 0 is handled (returns -inf rather than NaN-crashing).
# --------------------------------------------------------------------------


def test_zero_partition_function_handled() -> None:
    """If some n sample is exactly zero, <log Z> diverges to -inf.
    The primitive must return -inf, not NaN or a misleading float."""
    n_grid = [1, 2, 3, 4]
    zn = [1.0, 0.0, 0.5, 0.25]  # Z=0 contamination at n=2

    result = analytic_continuation_at_zero(zn, n_grid)
    assert result == -math.inf


# --------------------------------------------------------------------------
# 4. Validation.
# --------------------------------------------------------------------------


def test_grid_validation_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        analytic_continuation_at_zero([1.0, 2.0], [0, 1])


def test_grid_validation_rejects_short_grid() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        analytic_continuation_at_zero([1.0], [1])


def test_grid_validation_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="distinct"):
        analytic_continuation_at_zero([1.0, 1.0], [2, 2])


# --------------------------------------------------------------------------
# 5. Ensemble driver: average <Z^n> across instances, then continue.
# --------------------------------------------------------------------------


def test_compute_zn_for_ensemble_matches_manual_average() -> None:
    # An ensemble of three deterministic Z values.
    zs = [1.5, 2.0, 0.8]
    n_values = [1, 2, 3, 4]
    result = compute_zn_for_ensemble(zs, n_values)

    for n in n_values:
        expected = float(np.mean([z ** n for z in zs]))
        assert result[n] == pytest.approx(expected, abs=1e-12)


def test_ensemble_with_callables() -> None:
    # Each instance is a callable returning Z; check the driver invokes
    # it exactly once and feeds the result into the average.
    call_counts = {"a": 0, "b": 0}

    def make(key: str, z: float):
        def _f() -> float:
            call_counts[key] += 1
            return z
        return _f

    ens = [make("a", 1.0), make("b", 3.0)]
    result = compute_zn_for_ensemble(ens, [1, 2])
    assert result[1] == pytest.approx(2.0)
    assert result[2] == pytest.approx((1.0 + 9.0) / 2)
    assert call_counts == {"a": 1, "b": 1}


def test_ensemble_replica_pipeline_recovers_log_z() -> None:
    """End-to-end: build an ensemble of three Z's, run the driver,
    feed into the continuation.  Compare against the direct empirical
    average of log Z (which the replica method asymptotically targets;
    for finite ensembles the two agree because <log Z> is just the
    polynomial linear coefficient of the empirical Z^n moments)."""
    zs = [1.2, 1.0, 0.8]  # |log Z_i| < 0.25, well within well-conditioned regime
    n_values = [1, 2, 3, 4, 5, 6]

    zn = compute_zn_for_ensemble(zs, n_values)
    log_z_est = compute_log_z_from_replicas(zn)

    # The continuation extracts the linear-in-n coefficient of
    # (1/|ens|) sum_i Z_i^n = (1/|ens|) sum_i exp(n log Z_i)
    # which at n=0 differentiates to (1/|ens|) sum_i log Z_i.
    direct = float(np.mean([math.log(z) for z in zs]))
    assert log_z_est == pytest.approx(direct, abs=5e-3)


def test_ensemble_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        compute_zn_for_ensemble([], [1, 2])
