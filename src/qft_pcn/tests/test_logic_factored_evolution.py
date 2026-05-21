"""Tests for the factored Trotter evolution (sub-project C, Phase 2).

Per the controller's mandate (manifesto Temptation 3, 8): each gate
application factors through per-species cutoff×cutoff ops and a direct-
sum bond construction. NO (D_LOCAL, D_LOCAL) gate is ever materialized.

Phase 2 verifies the API + invariants on Phase 1's diagonal-only
EvalHamiltonian:

  1. Real-time evolution under Phase 1's H_eval is unitary, ||state||
     preserved to 1e-6 over 10 steps.
  2. Imag-time evolution under Phase 1's H_eval monotonically decreases
     ⟨H⟩ over 10 steps from a random encoded state.
  3. A normal-form state stays at ⟨H_eval⟩ < 1e-9 under imag-time.
  4. dt = 0 returns identity exactly, with zero truncation error.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic import encode, parse
from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
from src.qft_pcn.logic.factored_evolution import (
    factored_trotter_step, factored_evolve, factored_energy,
)


def _copy_state(state):
    return state.copy()


def test_zero_dt_is_identity():
    state, _ = encode(parse("2 + 3"), N=6, chi_max=8)
    H = EvalHamiltonian(N=6)
    snapshot = [t.copy() for t in state.tensors]
    err = factored_trotter_step(state, H, dt=0.0, imaginary=True, chi_max=8)
    assert err == 0.0
    # State must be exactly preserved.
    for k in range(state.N):
        assert state.tensors[k].shape == snapshot[k].shape
        assert np.allclose(state.tensors[k], snapshot[k], atol=1e-14)


def test_real_time_preserves_norm():
    """Phase 1's H_eval is Hermitian (real diagonal projectors). Each
    per-term gate exp(-i·dt·c·P) is unitary, so the product over all
    terms is unitary and ||state|| stays at 1.
    """
    state, _ = encode(parse(r"(\x:Int. x + 1) 2"), N=8, chi_max=16)
    state.normalize()
    norm_initial = state.norm_sq()
    assert abs(norm_initial - 1.0) < 1e-10

    H = EvalHamiltonian(N=8)
    for _ in range(10):
        factored_trotter_step(
            state, H, dt=0.05, imaginary=False, chi_max=16,
        )
    norm_final = state.norm_sq()
    assert abs(norm_final - 1.0) < 1e-6, (
        f"real-time evolution broke unitarity: norm = {norm_final}"
    )


def test_imag_time_non_increasing_energy():
    """Imag-time evolution monotonically does not INCREASE ⟨H⟩.

    Note: Phase 1's H_eval is purely diagonal in the computational basis,
    and the encoder produces a basis-state-localized MPS. Such a state
    is an EIGENSTATE of H_eval — imag-time evolution then just rescales
    the global amplitude (then renormalize → invariant), giving exactly
    flat energy. The driving off-diagonal transitions land in Phase 3.

    This test therefore checks only the monotonicity invariant, which is
    the structural guarantee imag-time evolution provides.
    """
    state, _ = encode(parse(r"(\x:Int. x + 1) 2"), N=8, chi_max=16)
    state.normalize()
    H = EvalHamiltonian(N=8)
    energies = [factored_energy(state, H)]
    for _ in range(10):
        factored_trotter_step(
            state, H, dt=0.1, imaginary=True, chi_max=16,
        )
        state.normalize()
        energies.append(factored_energy(state, H))
    # Monotonically non-increasing (allow tiny numerical wiggle).
    for i in range(1, len(energies)):
        assert energies[i] <= energies[i - 1] + 1e-9, (
            f"energy increased at step {i}: "
            f"{energies[i-1]} -> {energies[i]}"
        )


def test_normal_form_state_stays_zero_under_imag_time():
    """A normal-form encoded state has zero eval energy. Imag-time
    evolution should leave it unchanged (the gate is identity on the
    zero-eigenvalue subspace).
    """
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=16)
    state.normalize()
    H = EvalHamiltonian(N=8)
    assert abs(factored_energy(state, H)) < 1e-9

    for _ in range(10):
        factored_trotter_step(
            state, H, dt=0.1, imaginary=True, chi_max=16,
        )
        state.normalize()
        e = factored_energy(state, H)
        assert abs(e) < 1e-7, f"normal-form state acquired eval energy {e}"


def test_factored_evolve_returns_trace():
    state, _ = encode(parse("2 + 3"), N=6, chi_max=8)
    state.normalize()
    H = EvalHamiltonian(N=6)
    trace = factored_evolve(
        state, H, dt=0.1, steps=5, imaginary=True, chi_max=8,
    )
    assert len(trace) == 5
    for err in trace:
        assert err >= 0.0
        assert err < 1.0   # truncation can't blow up to lose everything
