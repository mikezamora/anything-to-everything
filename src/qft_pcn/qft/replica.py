"""Replica-method analytic-continuation tooling (substrate task S3).

Implements the numerical/symbolic primitives required by §12.7 of
``QFT_PCN_ARCHITECTURE.md`` — the replica trick

    <log Z> = lim_{n -> 0} (<Z^n> - 1) / n

We sample ``<Z^n>`` at positive integer ``n``, fit a low-order polynomial
in ``n`` (using the ansatz ``<Z^n> = 1 + n c1 + n^2 c2 + ...``), and read
off ``c1 = <log Z>`` as the linear coefficient of the Taylor expansion at
``n = 0``.

Scope notes (honest)
--------------------
* This module supplies only the *analytic-continuation* primitive plus a
  thin ensemble driver that averages ``Z^n`` across a user-supplied
  problem ensemble.  The ensemble must hand us partition functions ``Z``
  that are themselves *operator-derived* (e.g. ``Z = tr exp(-beta H)``
  computed on a substrate operator).  We do not synthesise ``Z`` here;
  see §1.6 anti-shortcut directive.
* Polynomial interpolation is exact for polynomial ``<Z^n>(n)`` of
  degree ``<= len(n_grid) - 1``.  For genuinely non-polynomial behaviour
  the continuation is the standard truncated-Taylor approximation —
  callers should pass a wide enough ``n_grid`` and inspect the fit
  residuals.
* Sympy is used opportunistically for exact rational interpolation when
  the inputs are exact (Python ``Fraction`` / ``int``); otherwise we
  fall back to ``numpy.polyfit`` over floats.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Optional

import numpy as np

try:  # sympy is in the resolved environment (transitive dep)
    import sympy as _sp

    _HAVE_SYMPY = True
except ImportError:  # pragma: no cover - sympy is in pyproject
    _HAVE_SYMPY = False


__all__ = [
    "analytic_continuation_at_zero",
    "compute_zn_for_ensemble",
    "compute_log_z_from_replicas",
]


def _validate_grid(n_grid: Sequence[int], zn_values: Sequence[float]) -> None:
    if len(n_grid) != len(zn_values):
        raise ValueError(
            f"n_grid length {len(n_grid)} != zn_values length {len(zn_values)}"
        )
    if len(n_grid) < 2:
        raise ValueError(
            "need at least 2 sample points to extract the linear "
            "coefficient at n=0"
        )
    if any(int(n) <= 0 for n in n_grid):
        raise ValueError(
            "replica n_grid must contain positive integers (n >= 1); the "
            "n -> 0 limit is reached via continuation, never sampled"
        )
    if len(set(int(n) for n in n_grid)) != len(n_grid):
        raise ValueError("n_grid must have distinct entries")


def analytic_continuation_at_zero(
    zn_values: Sequence[float],
    n_grid: Sequence[int],
    *,
    use_sympy: Optional[bool] = None,
) -> float:
    """Recover ``<log Z>`` from samples ``<Z^n>`` at integer ``n``.

    We construct the auxiliary samples ``g_i = (<Z^n_i> - 1) / n_i`` and
    interpolate them with a polynomial of degree ``len(n_grid) - 1``,
    then evaluate at ``n = 0``.  Justification: writing
    ``<Z^n> = exp(n F(n))`` with ``F(0) = <log Z>``, the quantity
    ``(<Z^n> - 1)/n = F(n) + O(n)`` is smooth at ``n = 0`` and limits
    to ``<log Z>``.  Interpolating this auxiliary function — rather
    than ``<Z^n>`` directly — is the standard replica-trick
    construction and is dramatically better-conditioned because it
    removes the ``n -> 0`` cancellation.

    Parameters
    ----------
    zn_values
        Samples of ``<Z^n>`` (real, finite).
    n_grid
        Positive integer replica counts at which the samples were taken.
    use_sympy
        ``True`` to force exact symbolic interpolation (Lagrange via
        sympy), ``False`` to force float ``numpy.polyfit``.  Default
        ``None`` picks sympy when it is importable and the residual of
        the numpy fit looks suspicious.

    Returns
    -------
    float
        Estimate of ``<log Z>``.

    Raises
    ------
    ValueError
        If the grid is malformed (see :func:`_validate_grid`).
    """
    _validate_grid(n_grid, zn_values)

    n_arr = np.asarray([int(n) for n in n_grid], dtype=float)
    z_arr = np.asarray(zn_values, dtype=float)

    # Z = 0 in the ensemble would give <Z^n> = 0 for all n => the
    # auxiliary g(n) = (0 - 1)/n = -1/n diverges as n -> 0, so
    # <log Z> = -inf.  Detect this honestly rather than emit NaN.
    if np.any(z_arr == 0.0):
        return -math.inf

    if not np.all(np.isfinite(z_arr)):
        raise ValueError("zn_values contains non-finite entries")

    # Build the auxiliary samples g_i = (Z^n_i - 1) / n_i.
    g_arr = (z_arr - 1.0) / n_arr

    degree = len(n_arr) - 1

    prefer_sympy = use_sympy if use_sympy is not None else _HAVE_SYMPY

    if prefer_sympy and _HAVE_SYMPY:
        n_sym = _sp.symbols("n")
        try:
            pts = [
                (int(n_arr[i]), _sp.Float(float(g_arr[i])))
                for i in range(len(n_arr))
            ]
            poly = _sp.interpolate(pts, n_sym)
            val0 = float(poly.subs(n_sym, 0))
            return val0
        except Exception:
            # fall through to numpy
            pass

    # numpy.polyfit returns highest-order coefficient first; the
    # constant term (value at n=0) is the last entry.
    coeffs = np.polyfit(n_arr, g_arr, degree)
    return float(coeffs[degree])


def compute_zn_for_ensemble(
    problem_ensemble: Iterable[Callable[[], float] | float],
    n_values: Sequence[int],
) -> dict[int, float]:
    """Compute ``<Z^n>`` for each ``n`` over an iterable problem ensemble.

    Each element of ``problem_ensemble`` is either

    * a callable ``() -> float`` that returns the partition function
      ``Z`` of one ensemble instance (preferred — keeps the operator
      computation lazy), or
    * a precomputed ``float`` ``Z``.

    The ensemble is materialised once (so each instance contributes to
    every ``n``), giving the empirical average
    ``<Z^n> ~ (1/|ens|) sum_i Z_i^n``.

    Parameters
    ----------
    problem_ensemble
        Iterable of ``Z`` values or zero-argument callables returning
        ``Z``.  Must be non-empty.
    n_values
        Positive integer replica counts.

    Returns
    -------
    dict[int, float]
        Mapping ``n -> <Z^n>``.
    """
    if any(int(n) <= 0 for n in n_values):
        raise ValueError("n_values must be positive integers")

    z_samples: list[float] = []
    for entry in problem_ensemble:
        if callable(entry):
            z = float(entry())
        else:
            z = float(entry)
        z_samples.append(z)

    if not z_samples:
        raise ValueError("problem_ensemble is empty")

    z_arr = np.asarray(z_samples, dtype=float)
    return {int(n): float(np.mean(z_arr ** int(n))) for n in n_values}


def compute_log_z_from_replicas(
    zn_values: dict[int, float],
    *,
    use_sympy: Optional[bool] = None,
) -> float:
    """Convenience wrapper: dict ``{n: <Z^n>}`` -> ``<log Z>``."""
    if not zn_values:
        raise ValueError("zn_values is empty")
    items = sorted(zn_values.items())
    n_grid = [n for n, _ in items]
    z_vals = [z for _, z in items]
    return analytic_continuation_at_zero(z_vals, n_grid, use_sympy=use_sympy)
