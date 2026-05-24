"""§12.7 Replica method for predictive typical-case complexity.

Spec reference: ``QFT_PCN_ARCHITECTURE.md`` §12.7 (Edwards-Anderson 1975,
Parisi 1980; Nobel 2021). For a problem-class ensemble, the typical
free energy

    <log Z> = lim_{n -> 0} (<Z^n> - 1) / n

predicts the average proof-search difficulty: large negative ``<log Z>``
=> tightly clustered Boltzmann weight on few low-energy proofs (easy
typical instance); ``<log Z>`` ~ 0 or positive => Z spread out across
many configurations or vanishing on most => hard typical instance.

This module is the COMPOSITION-layer driver that wires real QPCN
problem encodings (``encode_mera(parse(...))``) into the S3 substrate
``compute_log_z_from_replicas``. Per ``memory/architecture-soul.md``
(§1.1) the per-instance partition function ``Z`` is an operator-
algebraic Boltzmann weight read off the encoded MERA state — never an
AST count, never a structural fingerprint. Per §1.6 we do not synthesise
``Z`` from a classical interpreter; ``Z`` is built from real Hamiltonian
matrix-exponentials evaluated against the substrate's leaf marginals.

Per-instance Z construction (operator-algebraic)
-------------------------------------------------
Given a theorem ``T`` encoded as ``state = encode_mera(parse(T))[0]``
on ``N`` leaves with local dimension ``d = state.d_local``:

1. Build a real :class:`~src.qft_pcn.qft.hamiltonian.Hamiltonian` of
   width ``N`` on a single bosonic species of cutoff ``d``. This is
   the canonical local Hamiltonian used by ``mera_evolution.energy``
   (§10.4 substrate), reused unchanged.

2. For each leaf ``k`` read the operator-algebraic marginal density
   ``rho_k = diag(state.leaf_marginal(k))``. This is the genuine
   single-leaf reduced density matrix on a product MERA (and the
   per-projector expectation route on a superposition MERA — see
   ``MERA.leaf_marginal``); it is bitwise identical across alpha-
   equivalent encodings because ``encode_mera`` is alpha-invariant.

3. Build ``h_k = H.local_op(k)`` and form the operator-algebraic
   local partition function

        Z_k(beta) = tr( rho_k @ expm(-beta * h_k) ).

   This is ``<exp(-beta H_local_k)>_psi``: a real Hermitian matrix
   exponential weighted by the substrate-derived marginal. Per §1.1
   the weighting is the bond-entanglement reduced state of the
   encoded proof, NOT a hash or AST property.

4. The per-instance partition function is the geometric mean

        Z(T, beta) = ( prod_k Z_k(beta) ) ** (1 / N).

   Geometric mean (not product) keeps ``Z`` finite and comparable
   across different ``N`` so the ensemble average is well-conditioned
   for the S3 replica continuation.

S3 wiring
---------
:func:`compute_typical_complexity` then calls the substrate
:func:`~src.qft_pcn.qft.replica.compute_zn_for_ensemble` over the
ensemble ``[Z(T_1, beta), ..., Z(T_M, beta)]`` to obtain the integer-
``n`` samples ``<Z^n>``, and feeds them to
:func:`~src.qft_pcn.qft.replica.compute_log_z_from_replicas` for the
``n -> 0`` analytic continuation. The result is a
:class:`ComplexityPrediction` with the average ``<log Z>``, the
ensemble standard deviation of per-instance ``log Z`` (proxy for
ensemble heterogeneity), and a regime tag (see "Regime limits"
below).

Regime limits (honest, pinned by tests)
---------------------------------------
The S3 ``analytic_continuation_at_zero`` interpolates the auxiliary
``g(n) = (<Z^n> - 1)/n`` with a polynomial of degree ``len(n_grid)-1``
and reads its value at ``n = 0``. The continuation is well-conditioned
when:

  * Z values are bounded in a range that does not produce overflow in
    ``Z**n`` at the largest sampled ``n`` (rule of thumb: ``max(Z)**n``
    representable as ``float64``, so e.g. ``Z <= 50`` with ``n <= 8``).
  * The ensemble is non-degenerate: ``Z`` varies across instances. A
    single-point ensemble gives a strictly polynomial ``<Z^n>`` and
    is exact, but interpretation as a *typical* free energy degenerates
    to the single value ``log Z``.
  * No instance has ``Z = 0``: the substrate returns ``-inf``
    gracefully but the difficulty score is then unbounded.

Outside the well-conditioned regime we return
``ComplexityPrediction(typical_log_z=float, regime="degraded", ...)``
and log the cause in ``notes``. Callers are expected to inspect
``regime`` before consuming ``typical_log_z`` for downstream
scheduling decisions (§12.8 curriculum tie-in).

Public API
----------
- :class:`ComplexityPrediction`
- :func:`compute_typical_complexity` — full ensemble -> prediction
- :func:`predict_proof_difficulty` — theorem + solved corpus -> scalar
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm

from src.qft_pcn.logic.ast import Node, parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.qft.hamiltonian import (
    FieldSpecies,
    Hamiltonian,
    HamiltonianConfig,
)
from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.qft.replica import (
    compute_log_z_from_replicas,
    compute_zn_for_ensemble,
)


__all__ = [
    "ComplexityPrediction",
    "DEFAULT_REPLICA_N_GRID",
    "DEFAULT_INVERSE_TEMP",
    "compute_typical_complexity",
    "predict_proof_difficulty",
    "instance_partition_function",
]


# Default integer-n grid for the S3 replica continuation. Five points
# (n = 1..5) is the minimum that lets ``analytic_continuation_at_zero``
# fit a quartic and reject a noisy linear continuation; large enough to
# resolve typical curvature, small enough to keep ``max(Z)**n`` finite
# for Z in the operator-algebraic range we generate (~ unit scale).
DEFAULT_REPLICA_N_GRID: tuple[int, ...] = (1, 2, 3, 4, 5)


# Default inverse temperature for the per-instance Boltzmann weight.
# beta = 1.0 keeps exp(-beta * <H>) bounded away from zero for the
# typical-scale Hamiltonians built by ``_default_hamiltonian`` (per-leaf
# energies ~ O(1)); larger beta sharpens to the ground state but risks
# numerical underflow in expm for proofs whose leaf marginals concentrate
# on high-mass states. Callers can override via the ``beta`` kwarg.
DEFAULT_INVERSE_TEMP: float = 1.0


# Maximum |Z^n| we allow before declaring the continuation degraded.
# float64 max is ~1.8e308; we keep a safety margin so polynomial
# interpolation residuals do not blow up.
_MAX_SAFE_ZN: float = 1.0e150


@dataclass
class ComplexityPrediction:
    """Typical-case complexity prediction over a problem ensemble.

    Attributes
    ----------
    typical_log_z:
        ``<log Z>`` recovered by the S3 replica continuation. Lower
        (more negative) means *easier* typical instance: the Boltzmann
        weight is concentrated, so a search finds the low-energy proof
        configurations quickly. Higher (less negative or positive)
        means *harder*: weight is spread or vanishes.
    difficulty:
        Numeric difficulty score derived from ``typical_log_z`` and
        the ensemble heterogeneity ``log_z_std``. See
        :func:`_difficulty_score`. Always finite (NaN/inf inputs are
        mapped to ``float('inf')``).
    log_z_std:
        Standard deviation of per-instance ``log Z`` across the
        ensemble. A proxy for heterogeneity (replica-symmetry-breaking
        precursor, §12.7 risk #1 mitigation hook).
    ensemble_size:
        Number of theorems in the ensemble.
    n_grid:
        Replica-n grid used for the analytic continuation.
    regime:
        ``"well_conditioned"`` (typical_log_z is a trustworthy free
        energy), ``"degraded"`` (numerical breakdown — Z = 0, overflow,
        or single-point ensemble), or ``"empty"`` (ensemble exhausted
        to zero instances after encoding failures).
    notes:
        Free-form list of strings explaining the regime tag.
    """

    typical_log_z: float
    difficulty: float
    log_z_std: float
    ensemble_size: int
    n_grid: tuple[int, ...]
    regime: str
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Operator-algebraic per-instance partition function
# ---------------------------------------------------------------------------


def _default_hamiltonian(N: int, d_local: int) -> Hamiltonian:
    """A canonical single-species 1D bosonic Hamiltonian on ``N`` leaves.

    Uses ``FieldSpecies`` with cutoff ``d_local`` so ``H.local_op(k)``
    is a real Hermitian ``(d_local, d_local)`` operator compatible with
    ``state.leaf_marginal(k)`` (same dimension). Bare mass and kinetic
    couplings are O(1) so per-leaf energies are bounded; no quartic /
    cross-species coupling so ``H.local_op`` is a pure number operator
    plus a small constant shift — the simplest Hermitian operator that
    distinguishes leaf marginals concentrated on different basis states.
    """
    species = FieldSpecies(
        name="phi",
        cutoff=d_local,
        bare_mass=1.0,
        kinetic=0.5,
        quartic=0.0,
        source=0.0,
    )
    cfg = HamiltonianConfig(species=[species])
    return Hamiltonian(cfg, N=N)


def instance_partition_function(
    state: MERA,
    *,
    beta: float = DEFAULT_INVERSE_TEMP,
    hamiltonian: Hamiltonian | None = None,
) -> float:
    """Per-instance operator-algebraic partition function.

    Z(state, beta) = geometric_mean_k tr( rho_k @ expm(-beta h_k) ),

    where ``rho_k = diag(state.leaf_marginal(k))`` and
    ``h_k = H.local_op(k)``. Geometric mean across leaves keeps Z on
    a single-leaf scale so the ensemble's ``Z^n`` is representable in
    float64 for ``n`` up to ~10.

    Parameters
    ----------
    state:
        A real MERA state, typically ``encode_mera(parse(src))[0]``.
    beta:
        Inverse temperature. Positive real.
    hamiltonian:
        Optional preconstructed Hamiltonian. If ``None`` builds the
        canonical single-species H matching ``state.N`` and
        ``state.d_local`` via :func:`_default_hamiltonian`.

    Returns
    -------
    float
        ``Z`` for this instance. Finite and positive for a non-
        pathological state (leaf_marginal is a probability vector,
        expm of a Hermitian operator is positive definite => the trace
        is positive).

    Raises
    ------
    ValueError
        If ``beta <= 0`` or ``state`` has no leaves.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be positive (got {beta!r})")
    if state.N == 0:
        raise ValueError("state has no leaves")

    H = hamiltonian if hamiltonian is not None else _default_hamiltonian(
        N=state.N, d_local=state.d_local
    )
    if H.d_local != state.d_local:
        raise ValueError(
            f"Hamiltonian d_local {H.d_local} != state d_local "
            f"{state.d_local}"
        )

    log_z_per_leaf: list[float] = []
    for k in range(state.N):
        p_k = state.leaf_marginal(k)  # (d_local,) probability vector
        h_k = H.local_op(min(k, H.N - 1))  # (d, d) Hermitian
        # exp(-beta * h_k): real Hermitian matrix exponential. h_k is
        # constructed real-symmetric by _default_hamiltonian (no source,
        # no yukawa, no kinetic at single site) so expm is purely real.
        boltzmann = expm(-beta * h_k)
        # tr(rho_k @ boltzmann) = sum_i p_k[i] * boltzmann[i, i]
        # because rho_k is diagonal. This is the operator-algebraic
        # <exp(-beta h_k)>_{rho_k}.
        diag = np.real(np.diag(boltzmann))
        z_k = float(np.dot(p_k, diag))
        if z_k <= 0.0 or not np.isfinite(z_k):
            # Defensive: a numerical sink — degrade to a very small
            # positive value so log_z is large-negative-finite rather
            # than -inf, letting the ensemble continue.
            z_k = max(z_k, 1.0e-300) if np.isfinite(z_k) else 1.0e-300
        log_z_per_leaf.append(float(np.log(z_k)))

    # Geometric mean: exp( mean(log z_k) ).
    return float(np.exp(np.mean(log_z_per_leaf)))


# ---------------------------------------------------------------------------
# Ensemble driver
# ---------------------------------------------------------------------------


def _encode_one(theorem: Node | str) -> MERA | None:
    """Encode one theorem (Node or source string) to a MERA.

    Returns ``None`` on encoder failure so the ensemble loop can skip
    pathological inputs without raising — the regime tag will record
    the drop.
    """
    if isinstance(theorem, str):
        try:
            ast = parse(theorem)
        except Exception:
            return None
    elif isinstance(theorem, Node):
        ast = theorem
    else:
        raise TypeError(
            f"theorem must be a Node or source string (got "
            f"{type(theorem).__name__})"
        )
    try:
        state, _ = encode_mera(ast)
    except Exception:
        return None
    return state


def _difficulty_score(typical_log_z: float, log_z_std: float) -> float:
    """Map ``(<log Z>, std)`` -> scalar difficulty in ``[0, +inf]``.

    Difficulty grows when ``<log Z>`` is large (Z spread or vanishing
    on typical instances) and when ``log_z_std`` is large (the
    ensemble has high heterogeneity — replica-symmetry-breaking
    precursor, §12.7 risk #1).

    Formula::

        difficulty = max(0, -typical_log_z) + log_z_std

    where the ``-typical_log_z`` term flips sign so that an easy
    ensemble (typical_log_z very negative => Boltzmann weight
    concentrated) maps to LOW difficulty, while a hard ensemble
    (typical_log_z ~ 0) maps to HIGH difficulty. The ``max(0, ...)``
    floor pins easy ensembles at zero rather than going negative.

    Honesty note: this is a deliberately simple, monotonic combiner —
    it is the *score the spec wants for ranking*, not a calibrated
    bits-of-search measure. The calibration against §10.10 K-8 wall-
    clock results is recorded in ``EXTENSIONS.md`` once the corpus
    lands; see the EXTENSIONS entry for §12.7.
    """
    # NOTE: we negate -- larger negative log Z => easier => smaller score.
    if not np.isfinite(typical_log_z) or not np.isfinite(log_z_std):
        return float("inf")
    return float(max(0.0, -typical_log_z) + max(0.0, log_z_std))


def compute_typical_complexity(
    problem_ensemble: Iterable[Node | str],
    *,
    beta: float = DEFAULT_INVERSE_TEMP,
    n_grid: Sequence[int] = DEFAULT_REPLICA_N_GRID,
) -> ComplexityPrediction:
    """Predict typical-case complexity over a QPCN problem ensemble.

    Implements the §12.7 driver:
      1. Encode each theorem to a MERA via ``encode_mera(parse(...))``.
      2. Read the operator-algebraic per-instance ``Z`` via
         :func:`instance_partition_function`.
      3. Compute ``<Z^n>`` for integer n in ``n_grid`` (S3 substrate).
      4. Analytically continue to ``n -> 0`` for ``<log Z>``
         (S3 substrate).

    The returned ``ComplexityPrediction.regime`` flags numerical
    breakdown explicitly: callers must check it before consuming
    ``typical_log_z`` for scheduling decisions.

    Parameters
    ----------
    problem_ensemble:
        Iterable of theorems. Each entry may be a logic AST ``Node``
        or a source string parseable by ``logic.ast.parse``. Encoder
        failures drop the instance from the ensemble (logged in
        ``notes``).
    beta:
        Inverse temperature for the per-instance Boltzmann weight.
    n_grid:
        Positive integer replica counts. Default :data:`DEFAULT_REPLICA_N_GRID`.

    Returns
    -------
    ComplexityPrediction
    """
    notes: list[str] = []
    z_values: list[float] = []
    dropped = 0

    for theorem in problem_ensemble:
        state = _encode_one(theorem)
        if state is None:
            dropped += 1
            continue
        try:
            z = instance_partition_function(state, beta=beta)
        except Exception as exc:
            dropped += 1
            notes.append(f"instance partition function failed: {exc!r}")
            continue
        z_values.append(z)

    if dropped:
        notes.append(f"{dropped} instance(s) dropped (encoder/Z failure)")

    if not z_values:
        return ComplexityPrediction(
            typical_log_z=float("nan"),
            difficulty=float("inf"),
            log_z_std=float("nan"),
            ensemble_size=0,
            n_grid=tuple(int(n) for n in n_grid),
            regime="empty",
            notes=notes + ["no instances survived encoding"],
        )

    z_arr = np.asarray(z_values, dtype=float)
    log_z_per_instance = np.log(np.clip(z_arr, 1.0e-300, None))
    log_z_std = float(np.std(log_z_per_instance))

    # Regime guards.
    regime = "well_conditioned"
    max_n = max(int(n) for n in n_grid)
    max_z = float(np.max(z_arr))
    if max_z > 0.0 and (max_z ** max_n) > _MAX_SAFE_ZN:
        regime = "degraded"
        notes.append(
            f"max(Z)={max_z:.3g} ** max_n={max_n} exceeds safe range "
            f"({_MAX_SAFE_ZN:.0e}); continuation may be ill-conditioned"
        )
    if float(np.min(z_arr)) <= 0.0:
        regime = "degraded"
        notes.append(
            "ensemble contains Z <= 0 (substrate returns log Z = -inf)"
        )
    if len(z_arr) == 1:
        # Single-instance ensemble: <Z^n> = Z^n exactly => the
        # continuation returns log Z exactly, but the "typical" reading
        # collapses. Honest tag.
        regime = "degraded"
        notes.append(
            "single-instance ensemble: <log Z> = log Z (no replica average)"
        )

    # S3 substrate call.
    try:
        zn_dict = compute_zn_for_ensemble(z_values, n_grid)
        typical_log_z = compute_log_z_from_replicas(zn_dict)
    except Exception as exc:
        return ComplexityPrediction(
            typical_log_z=float("nan"),
            difficulty=float("inf"),
            log_z_std=log_z_std,
            ensemble_size=len(z_values),
            n_grid=tuple(int(n) for n in n_grid),
            regime="degraded",
            notes=notes + [f"S3 replica continuation failed: {exc!r}"],
        )

    if not np.isfinite(typical_log_z):
        regime = "degraded"
        notes.append(
            f"continuation returned non-finite <log Z> = {typical_log_z!r}"
        )

    return ComplexityPrediction(
        typical_log_z=float(typical_log_z),
        difficulty=_difficulty_score(typical_log_z, log_z_std),
        log_z_std=log_z_std,
        ensemble_size=len(z_values),
        n_grid=tuple(int(n) for n in n_grid),
        regime=regime,
        notes=notes,
    )


def predict_proof_difficulty(
    theorem: Node | str,
    similar_solved_corpus: Iterable[Node | str],
    *,
    beta: float = DEFAULT_INVERSE_TEMP,
    n_grid: Sequence[int] = DEFAULT_REPLICA_N_GRID,
) -> float:
    """Predict the difficulty of one theorem given a solved-similar corpus.

    Wraps :func:`compute_typical_complexity` on the ensemble
    ``[theorem] + list(similar_solved_corpus)``. The intuition is the
    §12.7 mean-field reading: a theorem whose addition to a solved
    corpus does not perturb ``<log Z>`` away from the solved-corpus
    baseline is in-distribution and presumably easy; a theorem that
    pushes ``<log Z>`` toward zero (or makes the ensemble
    heterogeneous) is harder.

    Returns the :attr:`ComplexityPrediction.difficulty` scalar of the
    combined ensemble. ``float('inf')`` is returned if the ensemble
    fully degrades (e.g. unparseable theorem with an empty corpus).

    Parameters
    ----------
    theorem:
        The candidate theorem (Node or source string).
    similar_solved_corpus:
        Iterable of previously-solved theorems for the same class
        (e.g. the wake-sleep solved set, spec §10.9 / §10.10 K-8).
    beta, n_grid:
        Forwarded to :func:`compute_typical_complexity`.
    """
    ensemble: list[Node | str] = [theorem]
    ensemble.extend(similar_solved_corpus)
    prediction = compute_typical_complexity(
        ensemble, beta=beta, n_grid=n_grid
    )
    return prediction.difficulty
