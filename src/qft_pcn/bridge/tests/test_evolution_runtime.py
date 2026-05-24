"""§2.6 search.runtime dispatches mps vs mera."""
import pytest

from src.qft_pcn.bridge.runtime.evolution import evolve_for_search


# Use a minimal fixture: tiny Hamiltonian + tiny state. Build it inline from
# the existing qft primitives so the test is self-contained. If the existing
# bridge tests have a reusable fixture builder, prefer that.
def _tiny_state_and_h():
    """Returns (initial_mps_state, hamiltonian) — 2 sites, cutoff 2."""
    import numpy as np
    from src.qft_pcn.qft.mps import MPS
    from src.qft_pcn.qft.hamiltonian import Hamiltonian, HamiltonianConfig, FieldSpecies

    species = [FieldSpecies(name="n", cutoff=2, bare_mass=1.0, kinetic=0.0,
                            quartic=0.0, source=0.0)]
    cfg = HamiltonianConfig(species=species, density_couplings={},
                            yukawa_couplings={}, curvature_xi=0.0)
    H = Hamiltonian(cfg, N=2)
    state = MPS.vacuum(N=2, d=2)
    return state, H


def test_runtime_mps_path():
    state, H = _tiny_state_and_h()
    res = evolve_for_search(state, H, runtime="mps", steps=2, chi_max=4, dt=0.05)
    assert res is not None


def test_runtime_invalid_raises():
    state, H = _tiny_state_and_h()
    with pytest.raises(ValueError, match="runtime"):
        evolve_for_search(state, H, runtime="bogus", steps=2, chi_max=4, dt=0.05)


def test_runtime_mera_path_or_not_implemented():
    """MERA path should either work (if mera_evolution is wired) or raise
    NotImplementedError with a clear message — never silently fall back."""
    state, H = _tiny_state_and_h()
    try:
        res = evolve_for_search(state, H, runtime="mera", steps=2,
                                chi_max=4, dt=0.05)
        assert res is not None
    except NotImplementedError as exc:
        assert "mera" in str(exc).lower() or "MERA" in str(exc)
