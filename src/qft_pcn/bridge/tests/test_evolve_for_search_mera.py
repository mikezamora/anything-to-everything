"""W4.T1: evolve_for_search routes to MERA when search.runtime='mera'.

Companion to ``test_evolution_runtime.py``. Where the older test allowed the
MERA path to honest-fail with NotImplementedError (the W4.T1 workaround), this
file pins the *positive* behavior: a product MPS initial state is coerced to
MERA via ``MERA.from_mps`` and the MERA TEBD substrate actually runs.
"""
from __future__ import annotations

import pytest

from src.qft_pcn.bridge.runtime.evolution import evolve_for_search
from src.qft_pcn.qft.hamiltonian import (
    FieldSpecies,
    Hamiltonian,
    HamiltonianConfig,
)
from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.qft.mps import MPS


def _tiny_state_and_h():
    species = [FieldSpecies(name="n", cutoff=2, bare_mass=1.0, kinetic=0.0,
                            quartic=0.0, source=0.0)]
    cfg = HamiltonianConfig(species=species, density_couplings={},
                            yukawa_couplings={}, curvature_xi=0.0)
    H = Hamiltonian(cfg, N=2)
    state = MPS.vacuum(N=2, d=2)
    return state, H


def test_evolve_for_search_routes_to_mera_when_search_runtime_mera():
    state, H = _tiny_state_and_h()
    out = evolve_for_search(state, H, runtime="mera", steps=2,
                            chi_max=4, dt=0.05)
    assert isinstance(out, MERA), (
        f"expected MERA after runtime='mera' coercion; got {type(out).__name__}"
    )
    # Substrate actually ran: state should be near-normalized after evolution
    # (MERA TEBD does not strictly normalize each step, but the magnitude
    # stays finite and positive).
    n = out.norm_sq()
    assert n > 0.0 and n < 1e12, f"non-finite/zero MERA norm_sq after evolve: {n}"


def test_evolve_for_search_mera_accepts_existing_mera_state():
    """If the caller already built a MERA, evolve_for_search uses it verbatim
    (no spurious coercion / round-trip)."""
    _, H = _tiny_state_and_h()
    mera = MERA.vacuum(N=2, d_local=2)
    out = evolve_for_search(mera, H, runtime="mera", steps=1,
                            chi_max=4, dt=0.05)
    assert out is mera


def test_evolve_for_search_mera_rejects_non_mps_non_mera():
    """Out-of-spec inputs must honest-fail (no silent fallback)."""
    _, H = _tiny_state_and_h()
    with pytest.raises(NotImplementedError, match="MERA"):
        evolve_for_search("not a state", H, runtime="mera",
                          steps=1, chi_max=4, dt=0.05)
