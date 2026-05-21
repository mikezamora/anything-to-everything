"""evolve_with_clamps — imag-time evolution with per-step boundary projection.

Reuses qft/evolution.py:trotter_step as-is; the only addition is the
post-step clamp loop and an energy-per-step trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hamiltonian import BridgeHamiltonian
from .clamp import Clamp, project_site
from ..dsl.term import FieldSpec
from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.qft.evolution import trotter_step, energy


@dataclass
class ConvergenceHistory:
    energy_per_step: list[float] = field(default_factory=list)
    trunc_error_per_step: list[float] = field(default_factory=list)


def evolve_with_clamps(state: MPS, H: BridgeHamiltonian, *,
                       dt: float, steps: int, chi_max: int,
                       clamps: list[Clamp], fields: list[FieldSpec]
                       ) -> ConvergenceHistory:
    hist = ConvergenceHistory()
    for _ in range(steps):
        if H.N >= 2:
            err = trotter_step(state, H, dt, imaginary=True, chi_max=chi_max)
        else:
            # Single-site chains have no bonds; apply local exp(-dt H) directly.
            err = _single_site_local_step(state, H, dt)
        state.normalize()
        for c in clamps:
            project_site(state, c, fields=fields)
            state.normalize()
        hist.energy_per_step.append(float(energy(state, H)))
        hist.trunc_error_per_step.append(float(err if err is not None else 0.0))
    return hist


def _single_site_local_step(state: MPS, H: BridgeHamiltonian,
                            dt: float) -> float:
    """Imag-time step for an N=1 chain: apply exp(-dt * H_local(0))."""
    from scipy.linalg import expm
    import numpy as np
    h = H.local_op(0)
    gate = expm(-dt * h)
    state.apply_local_gate(0, gate.astype(np.complex128))
    return 0.0
