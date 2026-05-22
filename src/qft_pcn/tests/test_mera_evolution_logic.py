"""Tests for MERA imaginary-time evolution (spec §7.5)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_compose import compose_mera_hamiltonians
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_trotter_step, mera_imaginary_evolve,
)


def test_trotter_step_preserves_norm():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    state = mera_trotter_step(state, H, dt=0.1, imaginary=True)
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-8)


def test_imaginary_evolve_returns_energy_trajectory():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    traj = mera_imaginary_evolve(state, H, dt=0.1, steps=20)
    assert len(traj) == 21


def test_imaginary_evolve_monotone_decrease():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    traj = mera_imaginary_evolve(state, H, dt=0.3, steps=30)
    for i in range(len(traj) - 1):
        assert traj[i + 1] <= traj[i] + 1e-6, (
            f"energy rose at step {i}: {traj[i]} -> {traj[i+1]}")
    assert traj[-1] < traj[0] * 0.5


def test_evolve_accepts_composed_hamiltonian():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = compose_mera_hamiltonians(MeraTypingHamiltonian(meta),
                                  MeraEvalHamiltonian(meta))
    traj = mera_imaginary_evolve(state, H, dt=0.3, steps=10)
    assert traj[-1] <= traj[0] + 1e-6
