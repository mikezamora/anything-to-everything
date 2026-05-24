"""Bridge evolution utilities.

Two public entry points:

  evolve_with_clamps — imag-time evolution with per-step boundary projection.
      Reuses qft/evolution.py:trotter_step as-is; the only addition is the
      post-step clamp loop and an energy-per-step trace.

  evolve_for_search — §2.6 routing wrapper that dispatches search.runtime
      'mps' | 'mera' to the appropriate TEBD substrate.
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


def evolve_for_search(state, hamiltonian, *, runtime: str, steps: int,
                      chi_max: int, dt: float, imaginary: bool = True):
    """Route TEBD to the flat-MPS or MERA substrate based on §2.6 runtime.

    Parameters
    ----------
    state:
        MPS (for runtime='mps') or MERA (for runtime='mera') initial state.
        Evolved in-place; the same object is returned so callers can compute
        observables without keeping a separate reference.
    hamiltonian:
        A Hamiltonian compatible with the chosen substrate.
    runtime:
        'mps' — flat-MPS TEBD via qft.evolution.evolve.
        'mera' — hierarchical TEBD via qft.mera_evolution.evolve.
    steps, chi_max, dt, imaginary:
        Forwarded verbatim to the substrate evolve() call.

    Returns
    -------
    The evolved state (same object as `state`).

    Raises
    ------
    ValueError  if runtime is not 'mps' or 'mera'.
    NotImplementedError  if runtime='mera' but state is not a MERA instance
        (the bridge does not coerce MPS to MERA; see EXTENSIONS.md).
    """
    if runtime == "mps":
        from src.qft_pcn.qft.evolution import evolve
        evolve(state, hamiltonian, dt=dt, steps=steps,
               imaginary=imaginary, chi_max=chi_max)
        return state
    if runtime == "mera":
        from src.qft_pcn.qft.mera import MERA
        if not isinstance(state, MERA):
            raise NotImplementedError(
                "MERA evolution requires a MERA state; received "
                f"{type(state).__name__!r}. The bridge does not coerce MPS to "
                "MERA automatically; see EXTENSIONS.md (bridge MERA routing)."
            )
        from src.qft_pcn.qft.mera_evolution import evolve as mera_evolve
        mera_evolve(state, hamiltonian, dt=dt, steps=steps,
                    imaginary=imaginary, chi_max=chi_max)
        return state
    raise ValueError(
        f"unknown search.runtime: {runtime!r} (must be 'mps' or 'mera')"
    )


def _single_site_local_step(state: MPS, H: BridgeHamiltonian,
                            dt: float) -> float:
    """Imag-time step for an N=1 chain: apply exp(-dt * H_local(0))."""
    from scipy.linalg import expm
    import numpy as np
    h = H.local_op(0)
    gate = expm(-dt * h)
    state.apply_local_gate(0, gate.astype(np.complex128))
    return 0.0
