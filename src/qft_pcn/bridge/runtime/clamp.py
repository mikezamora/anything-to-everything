"""Boundary-site projection (spec §5.4).

A Clamp pins one site's named field to a fixed basis index. After each
evolution step the runtime projects every clamped site back onto the
sub-Hilbert-space where that field has that value (and renormalizes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsl.term import FieldSpec
from src.qft_pcn.qft.mps import MPS


@dataclass(frozen=True)
class Clamp:
    site: int
    field: str
    basis_index: int


def _field_projector(fields: list[FieldSpec], field_name: str,
                     basis_index: int) -> np.ndarray:
    names = [f.name for f in fields]
    idx = names.index(field_name)
    diag = np.array([1.0])
    for i, f in enumerate(fields):
        if i == idx:
            d = np.zeros(f.cutoff)
            d[basis_index] = 1.0
            diag = np.kron(diag, d)
        else:
            diag = np.kron(diag, np.ones(f.cutoff))
    return np.diag(diag).astype(complex)


def project_site(state: MPS, clamp: Clamp, *,
                 fields: list[FieldSpec]) -> None:
    """Apply the projector for `clamp` to `state` in-place. Caller is
    responsible for re-normalizing after."""
    P = _field_projector(fields, clamp.field, clamp.basis_index)
    state.apply_local_gate(clamp.site, P)
