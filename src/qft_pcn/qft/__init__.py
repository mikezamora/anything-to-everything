"""Quantum field theory core for the PCN-QFT hybrid.

Public surface:
    fock           - local Fock-space operators (a, a^dagger, n, phi, pi)
    MPS            - Matrix Product State representation
    Hamiltonian    - multi-species 1D Hamiltonian with curvature coupling
    HamiltonianConfig, FieldSpecies
    trotter_step, evolve, energy
    QPCN           - quantum predictive coder on an MPS substrate
    QPCNConfig
"""

from . import fock
from .mps import MPS
from .hamiltonian import Hamiltonian, HamiltonianConfig, FieldSpecies
from .evolution import trotter_step, evolve, energy
from .qpcn import QPCN, QPCNConfig

__all__ = [
    "fock",
    "MPS",
    "Hamiltonian",
    "HamiltonianConfig",
    "FieldSpecies",
    "trotter_step",
    "evolve",
    "energy",
    "QPCN",
    "QPCNConfig",
]
