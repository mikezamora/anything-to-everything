"""Stub term factories used until sub-project B/C land.

Supports the patterns in spec §3.9 plus a handful of obvious extensions.
Any unsupported shape raises TermUnsupportedError.

When sub-project B/C land, the bridge's compiler should call B/C's named
factories instead of these stubs; this file is retained as a fallback.
"""

from __future__ import annotations

import numpy as np

from .term import LocalTerm, TwoSiteTerm, FieldSpec
from ..errors import TermUnsupportedError


def _species_dims(fields: list[FieldSpec]) -> tuple[int, ...]:
    return tuple(f.cutoff for f in fields)


def _embed_diag(field_idx: int, diag_per_field: np.ndarray,
                dims: tuple[int, ...]) -> np.ndarray:
    """Build a diagonal d_local x d_local operator whose `field_idx`'th
    species contributes `diag_per_field` and other species contribute
    identity. Leftmost species changes slowest (matches qft/fock.embed_op).
    """
    d_local = int(np.prod(dims))
    diag = np.array([1.0])
    for i, d in enumerate(dims):
        if i == field_idx:
            diag = np.kron(diag, diag_per_field)
        else:
            diag = np.kron(diag, np.ones(d))
    assert diag.shape == (d_local,)
    return np.diag(diag).astype(complex)


def projector_on_field_value(fields: list[FieldSpec], field_name: str,
                              basis_index: int) -> np.ndarray:
    """One-site projector onto |field_name = basis_index> in the full
    d_local Hilbert space."""
    names = [f.name for f in fields]
    if field_name not in names:
        raise TermUnsupportedError(
            message=f"field {field_name!r} not declared",
            details={"known": names},
        )
    idx = names.index(field_name)
    cutoff = fields[idx].cutoff
    if not 0 <= basis_index < cutoff:
        raise TermUnsupportedError(
            message=f"basis index {basis_index} out of [0, {cutoff-1}] "
                    f"for field {field_name!r}",
            details={"field": field_name, "basis": basis_index,
                     "cutoff": cutoff},
        )
    diag = np.zeros(cutoff)
    diag[basis_index] = 1.0
    return _embed_diag(idx, diag, _species_dims(fields))


def term_field_equals_constant(fields: list[FieldSpec], site: int,
                               field_name: str, basis_index: int,
                               weight: float) -> LocalTerm:
    """w * (I - P_{field=basis}) at `site`."""
    dims = _species_dims(fields)
    d_local = int(np.prod(dims))
    P = projector_on_field_value(fields, field_name, basis_index)
    op = weight * (np.eye(d_local, dtype=complex) - P)
    return LocalTerm(site=site, operator=op)


def _projector_diag(fields: list[FieldSpec], field_name: str,
                    basis_index: int) -> np.ndarray:
    """Diagonal entries of the one-site projector onto |field=basis>."""
    names = [f.name for f in fields]
    idx = names.index(field_name)
    cutoff = fields[idx].cutoff
    if not 0 <= basis_index < cutoff:
        raise TermUnsupportedError(
            message=f"basis index {basis_index} out of [0, {cutoff-1}] "
                    f"for field {field_name!r}",
            details={"field": field_name, "basis": basis_index,
                     "cutoff": cutoff},
        )
    diag = np.array([1.0])
    for i, f in enumerate(fields):
        if i == idx:
            d = np.zeros(f.cutoff)
            d[basis_index] = 1.0
            diag = np.kron(diag, d)
        else:
            diag = np.kron(diag, np.ones(f.cutoff))
    return diag


def term_field_equality_two_site(fields: list[FieldSpec],
                                 sites: tuple[int, int],
                                 field_name: str,
                                 weight: float) -> TwoSiteTerm:
    """w * (I - sum_t P_t @ A x P_t @ B): penalize disagreement on `field_name`.

    Built via diagonal arithmetic to avoid materializing a d_local^2 x d_local^2
    dense matrix for large d_local.
    """
    names = [f.name for f in fields]
    if field_name not in names:
        raise TermUnsupportedError(
            message=f"field {field_name!r} not declared",
            details={"known": names},
        )
    idx = names.index(field_name)
    cutoff = fields[idx].cutoff
    dims = _species_dims(fields)
    d_local = int(np.prod(dims))
    agree_diag = np.zeros(d_local * d_local, dtype=float)
    for t in range(cutoff):
        pd = _projector_diag(fields, field_name, t)
        # Kron of two diagonals is the outer product of the diagonals.
        agree_diag += np.kron(pd, pd)
    op_diag = weight * (1.0 - agree_diag)
    op = np.diag(op_diag.astype(complex))
    return TwoSiteTerm(sites=sites, operator=op)
