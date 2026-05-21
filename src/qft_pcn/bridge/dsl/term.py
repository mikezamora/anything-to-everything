"""Term data classes for the bridge compiler.

A LocalTerm is a (site, operator) pair where `operator` acts on the local
d_local Hilbert space (a d_local x d_local Hermitian matrix). A TwoSiteTerm
is a (sites, operator) pair where `operator` acts on d_local^2.

These wrappers exist so the bridge can collect terms produced by B/C's
factory functions and pass them to a Hamiltonian constructor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    name: str
    cutoff: int


@dataclass
class LocalTerm:
    site: int
    operator: np.ndarray   # (d_local, d_local), Hermitian


@dataclass
class TwoSiteTerm:
    sites: tuple[int, int]
    operator: np.ndarray   # (d_local^2, d_local^2), Hermitian
