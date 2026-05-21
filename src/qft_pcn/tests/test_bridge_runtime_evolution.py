"""Tests for the runtime evolution wrapper (spec §5.5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.bridge.dsl.term import FieldSpec, LocalTerm, TwoSiteTerm
from src.qft_pcn.bridge.runtime.hamiltonian import BridgeHamiltonian
from src.qft_pcn.bridge.runtime.evolution import evolve_with_clamps
from src.qft_pcn.bridge.runtime.clamp import Clamp
from src.qft_pcn.bridge.runtime.initial_state import build_initial_state
from src.qft_pcn.qft.evolution import energy


def _two_field_setup():
    return [FieldSpec("kind", 4), FieldSpec("value", 4)]   # d_local = 16


def test_bridge_hamiltonian_local_op_is_hermitian():
    fields = _two_field_setup()
    op = np.zeros((16, 16), dtype=complex)
    op[3, 3] = 1.0
    op[7, 7] = 0.5
    H = BridgeHamiltonian(
        fields=fields, sites=3,
        terms=[LocalTerm(site=1, operator=op)],
    )
    L = H.local_op(1)
    assert L.shape == (16, 16)
    assert np.allclose(L, L.T.conj())


def test_bridge_hamiltonian_zero_op_when_site_unconstrained():
    fields = _two_field_setup()
    H = BridgeHamiltonian(fields=fields, sites=3, terms=[])
    assert np.allclose(H.local_op(2), 0.0)


def test_evolution_lowers_energy_toward_clamp():
    """Single-site local term penalizing kind != 1 at site 0; starting
    from a superposition, imag-time evolution should drive toward kind=1
    and lower energy."""
    fields = _two_field_setup()
    d_local = 16
    P_kind1 = np.zeros((d_local, d_local), dtype=complex)
    for i in range(4, 8):
        P_kind1[i, i] = 1.0
    op = np.eye(d_local, dtype=complex) - P_kind1
    H = BridgeHamiltonian(fields=fields, sites=2,
                          terms=[LocalTerm(site=0, operator=op)])
    # Start with equal superposition over the first 8 basis states (kind=0,1).
    from src.qft_pcn.qft.mps import MPS
    v = np.zeros(d_local, dtype=complex)
    v[:8] = 1.0 / np.sqrt(8)
    state = MPS.from_product([v, np.eye(d_local)[0].astype(complex)])
    E0 = energy(state, H)
    history = evolve_with_clamps(state, H, dt=0.1, steps=20,
                                 chi_max=8, clamps=[], fields=fields)
    E_final = history.energy_per_step[-1]
    assert E_final < E0 - 0.1


def test_clamp_persists_through_evolution():
    fields = _two_field_setup()
    H = BridgeHamiltonian(fields=fields, sites=2, terms=[])
    state = build_initial_state(fields=fields, sites=2,
                                boundary={0: {"kind": 2}})
    clamps = [Clamp(site=0, field="kind", basis_index=2)]
    evolve_with_clamps(state, H, dt=0.05, steps=10, chi_max=8,
                       clamps=clamps, fields=fields)
    # kind=2 means site 0 should remain on basis-index 8 (kind=2, value=0).
    P = np.zeros((16, 16), dtype=complex)
    P[8, 8] = 1.0
    val = state.local_expectation(0, P)
    assert val.real > 1.0 - 1e-6
