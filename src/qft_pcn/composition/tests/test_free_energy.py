"""Tests for the §13.6 substrate ΔF estimator (E1).

The estimator backs the orchestrator's lemma-selection scoring (spec §5.3)
with a real operator-algebraic free-energy reading: ``F = ⟨H⟩ + T · S(ρ_cut)``
computed from tensor-network contractions on the live substrate. The
tests below pin three load-bearing invariants:

1. ``free_energy`` is real-valued and finite on a non-trivial MERA / MPS
   substrate with a real Hamiltonian.
2. ``delta_free_energy`` is *negative* for a state transformation that
   demonstrably compresses the substrate -- the spec §5.3 "favoured
   frontier candidate" signal. Exercised through the §10.10 R-AddZero
   pipeline: ``encode_mera(2 + 0)`` → imaginary-time descent → the
   reduced state has strictly lower ⟨H⟩ than the encoded redex.
3. ``delta_free_energy`` is ≈ 0 for an idempotent / no-op transformation:
   feeding the SAME substrate state in as both ``before`` and ``after``
   recovers ΔF = 0 exactly (modulo IEEE-754 noise).

Anti-shortcut (§1.1 / memory:anti-shortcut-directive): every test reads
the energy via the Hamiltonian's ``total_energy`` / ``local_op`` +
``bond_op`` surface and the entropy via the substrate's
``entanglement_entropy(cut)`` surface. No state is fabricated; no value
is cached by lemma id; no heuristic depth-cost stand-in is accepted.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.qft_pcn.composition.free_energy import (
    FreeEnergyError,
    delta_free_energy,
    free_energy,
)
from src.qft_pcn.logic.ast import Bin, IntLit, Zero
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_evolution_logic import mera_imaginary_evolve_state
from src.qft_pcn.qft.evolution import trotter_step
from src.qft_pcn.qft.hamiltonian import (
    FieldSpecies,
    Hamiltonian,
    HamiltonianConfig,
)
from src.qft_pcn.qft.mps import MPS


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadows the conftest §9.7 dense-tensor ceiling
    """Opt this file out of the §9.7 dense-tensor ceiling: ``encode_mera``
    of the R-AddZero redex allocates 16-dim leaves and isometries that
    exceed the conftest cap by construction (same opt-out as
    ``test_bidirectional_evolution.py``).
    """
    yield


# ---------------------------------------------------------------------------
# Invariant 1: real-valued, finite F on a non-trivial substrate.
# ---------------------------------------------------------------------------


def _two_site_mps_hamiltonian():
    """A small two-site MPS Hamiltonian with non-trivial expectation.

    Single bosonic species, cutoff 3, mass 1.0, kinetic 0.3 -- enough
    structure for both ⟨H⟩ and S(ρ_cut) to be non-zero on a number-state
    input under a few Trotter sweeps (the same configuration used by
    ``test_qft.test_imaginary_time_lowers_energy``).
    """
    cfg = HamiltonianConfig(
        species=[FieldSpecies("a", cutoff=3, bare_mass=1.0, kinetic=0.3)],
    )
    return Hamiltonian(cfg, N=4)


def test_free_energy_real_and_finite_on_mps():
    """F on a non-trivial MPS state is a real, finite float.

    The MPS path exercises the ``local_op`` / ``bond_op`` fallback in
    ``_energy_expectation`` (no ``total_energy`` surface), and the
    standard MPS Schmidt entropy across a mid-network cut.
    """
    H = _two_site_mps_hamiltonian()
    # Excited number state with population on multiple sites -- ⟨H⟩
    # is non-zero by construction (the mass term sums the n_k weights).
    state = MPS.number_states([1, 0, 1, 0], d=H.d_local)
    F = free_energy(state, H, temperature=1.0)
    assert isinstance(F, float)
    assert math.isfinite(F), f"F = {F} must be finite on a number state"
    # ⟨H⟩ >= 0 for this Hamiltonian (mass + kinetic, no source) and
    # entropy >= 0 always -- F is non-negative on a number-state input.
    assert F >= 0.0, f"F = {F} must be non-negative on a number-state input"


def test_free_energy_real_and_finite_on_mera():
    """F on an encoded R-AddZero MERA state is real and finite.

    Exercises the ``total_energy`` surface (MERA-shaped Hamiltonian) plus
    the MERA's ``entanglement_entropy`` Schmidt-spectrum reading.
    """
    state, meta = encode_mera(Bin(op="+", lhs=IntLit(val=2), rhs=Zero()))
    H = MeraEvalHamiltonian(meta)
    F = free_energy(state, H, temperature=1.0)
    assert isinstance(F, float)
    assert math.isfinite(F), f"F = {F} must be finite on the redex state"


# ---------------------------------------------------------------------------
# Invariant 2: ΔF < 0 for a state transformation that compresses F
# (the §10.10 R-AddZero redex collapsed by imaginary-time descent).
# ---------------------------------------------------------------------------


def test_delta_free_energy_negative_for_compressing_lemma():
    """R-AddZero on ``2 + 0`` demonstrably compresses the substrate F.

    Construction:
      * ``before`` = the encoded redex ``2 + 0`` (carries non-zero
        R-AddZero penalty mass on the Bin node).
      * ``after`` = the substrate state after a short imaginary-time
        descent under :class:`MeraEvalHamiltonian` (the spec §13.1
        gradient flow on ⟨H⟩). The R-AddZero penalty term relaxes;
        ⟨H⟩ strictly drops.

    The Hamiltonian is held FIXED across the two states -- spec §13.6
    is a comparison of two ρ under one H. Both states are read from
    the live substrate (no cached value).
    """
    state_before, meta = encode_mera(
        Bin(op="+", lhs=IntLit(val=2), rhs=Zero())
    )
    H = MeraEvalHamiltonian(meta)
    # A short descent -- enough to relax the R-AddZero term but not so
    # long that we are observing post-collapse oscillations.
    _, state_after = mera_imaginary_evolve_state(
        state_before, H, dt=0.2, steps=20, chi_layer=16,
    )

    # Sanity: the descent did lower ⟨H⟩ (otherwise this fixture is not
    # exercising a real compression and the ΔF sign would be meaningless).
    e_before = float(H.total_energy(state_before))
    e_after = float(H.total_energy(state_after))
    assert e_after < e_before - 1e-6, (
        f"fixture broken: imaginary-time descent did not lower ⟨H⟩: "
        f"{e_before} -> {e_after}"
    )

    df = delta_free_energy(state_before, state_after, H, temperature=1.0)
    assert df < 0.0, (
        f"ΔF must be negative for a compressing lemma; got {df} "
        f"(⟨H⟩: {e_before} -> {e_after}; spec §5.3 favoured-frontier "
        f"contract violated)"
    )


# ---------------------------------------------------------------------------
# Invariant 3: ΔF ≈ 0 for a no-op transformation.
# ---------------------------------------------------------------------------


def test_delta_free_energy_zero_for_noop_lemma():
    """ΔF(state, state, H) is exactly zero (modulo IEEE noise).

    An idempotent / no-op lemma application leaves the substrate state
    untouched; the spec §5.3 contract requires the schedule key to be
    zero in that case. The estimator MUST NOT inject a bias that
    distinguishes a no-op from absence-of-information.
    """
    H = _two_site_mps_hamiltonian()
    state = MPS.number_states([1, 0, 1, 0], d=H.d_local)
    df = delta_free_energy(state, state, H, temperature=1.0)
    # Exactly zero is the ideal; IEEE-754 noise on the entropy SVD can
    # produce a tiny non-zero residual at the eps level.
    assert abs(df) < 1e-10, (
        f"ΔF for a no-op must vanish; got {df}. The §5.3 schedule key "
        f"would then prefer a no-op over an absent measurement, which "
        f"violates the favoured-frontier contract."
    )


def test_delta_free_energy_zero_for_noop_mera():
    """The no-op invariant holds on the MERA path too.

    Cross-checks the MERA ``total_energy`` + ``entanglement_entropy``
    surfaces (vs the MPS ``local_op``/``bond_op`` + ``entanglement_entropy``
    surfaces in the previous test) yield the same ΔF = 0 invariant.
    """
    state, meta = encode_mera(Bin(op="+", lhs=IntLit(val=2), rhs=Zero()))
    H = MeraEvalHamiltonian(meta)
    df = delta_free_energy(state, state, H, temperature=1.0)
    assert abs(df) < 1e-10, (
        f"ΔF for a MERA no-op must vanish; got {df}"
    )


# ---------------------------------------------------------------------------
# Anti-shortcut guards: the estimator refuses to fabricate a value.
# ---------------------------------------------------------------------------


def test_free_energy_raises_on_missing_state():
    """``state=None`` must raise -- a synthetic _JointResult internal-node
    join is the spec's "no substrate measurement" case; the §1.1
    anti-shortcut directive forbids silently zeroing F in that case.
    """
    H = _two_site_mps_hamiltonian()
    with pytest.raises(FreeEnergyError, match="real substrate state"):
        free_energy(None, H)


def test_free_energy_raises_on_missing_hamiltonian():
    """``hamiltonian=None`` must raise -- the §13.6 variational principle
    is defined relative to an operator H. A missing H is not a "weak"
    F reading, it is no F at all.
    """
    H = _two_site_mps_hamiltonian()
    state = MPS.number_states([1, 0, 1, 0], d=H.d_local)
    with pytest.raises(FreeEnergyError, match="requires a Hamiltonian"):
        free_energy(state, None)


def test_free_energy_raises_on_negative_temperature():
    """Negative T is not in the §13.6 domain; refuse rather than ship a
    silently-inverted F-sign convention.
    """
    H = _two_site_mps_hamiltonian()
    state = MPS.number_states([1, 0, 1, 0], d=H.d_local)
    with pytest.raises(FreeEnergyError, match="non-negative"):
        free_energy(state, H, temperature=-0.5)


def test_free_energy_temperature_scaling_is_real():
    """The T dependence comes from the entropy term ALONE.

    F(T=T0) − F(T=0) = T0 · S(ρ_cut). Pinning this identity proves the
    T-scaling channel is the operator-algebraic entropy (not a stray
    classical multiplier on the energy expectation).
    """
    H = _two_site_mps_hamiltonian()
    # Build a state with non-zero bond entanglement: a hopping Trotter
    # sweep generates real Schmidt entropy across the mid-network cut.
    state = MPS.number_states([1, 0, 1, 0], d=H.d_local)
    for _ in range(4):
        trotter_step(state, H, dt=0.1, imaginary=False, chi_max=16)
    state.normalize()

    F0 = free_energy(state, H, temperature=0.0)
    F1 = free_energy(state, H, temperature=1.0)
    F2 = free_energy(state, H, temperature=2.0)

    # F(T) = ⟨H⟩ + T · S  ⇒  F(T) − F(0) = T · S  is linear in T.
    s_from_1 = F1 - F0
    s_from_2 = F2 - F0
    assert s_from_1 >= 0.0, (
        f"entropy contribution must be non-negative; got {s_from_1}"
    )
    assert abs(s_from_2 - 2.0 * s_from_1) < 1e-9, (
        f"T-scaling not linear: F(2)-F(0)={s_from_2}, "
        f"2*(F(1)-F(0))={2.0 * s_from_1}; the entropy channel is the "
        f"sole T-dependent contribution"
    )
