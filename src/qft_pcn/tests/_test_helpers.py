"""Test-only helpers for typing-Hamiltonian acceptance tests.

`mutate_local_register` swaps two basis-state slices on one species'
register at one MPS site, simulating a surgical typing-rule violation.
Vectorized via reshape + slice assignment — no 65K-iteration Python loop.
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.logic.encoding import (
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
)


_SPECIES_INDEX = {"kind": 0, "type": 1, "bid": 2, "value": 3, "tobl": 4}
_SPECIES_DIMS = (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF)


def mutate_local_register(state: MPS, site: int, register: str,
                          old: int, new: int) -> MPS:
    """Swap basis slices `old` and `new` on `register` at `site`.

    Equivalent to a local-unitary swap gate on the chosen register;
    preserves norm. Returns the SAME MPS (mutation is in place).
    """
    if register not in _SPECIES_INDEX:
        raise ValueError(f"unknown register {register!r}")
    sp_idx = _SPECIES_INDEX[register]
    sp_dim = _SPECIES_DIMS[sp_idx]
    if not (0 <= old < sp_dim and 0 <= new < sp_dim):
        raise ValueError(f"old/new out of range for {register}")
    t = state.tensors[site]
    chi_l, _, chi_r = t.shape
    A = t.reshape(chi_l, *_SPECIES_DIMS, chi_r).copy()
    target = sp_idx + 1
    A_moved = np.moveaxis(A, target, 1)
    tmp = A_moved[:, old, ...].copy()
    A_moved[:, old, ...] = A_moved[:, new, ...]
    A_moved[:, new, ...] = tmp
    A_back = np.moveaxis(A_moved, 1, target)
    state.tensors[site] = A_back.reshape(chi_l, t.shape[1], chi_r)
    return state
