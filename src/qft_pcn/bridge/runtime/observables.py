"""Construct local observable operators (n, phi, pi, a, adag, identity) on
the canonical d_local Hilbert space, embedded on the named field species.
"""

from __future__ import annotations

import numpy as np

from ..dsl.term import FieldSpec
from src.qft_pcn.qft.fock import (
    annihilation, creation, number, phi_op, pi_op, identity, embed_op,
)


_OPS = {
    "a":        annihilation,
    "adag":     creation,
    "n":        number,
    "phi":      phi_op,
    "pi":       pi_op,
    "identity": identity,
}


def build_observable_op(fields: list[FieldSpec], field_name: str,
                        op: str) -> np.ndarray:
    names = [f.name for f in fields]
    if field_name not in names:
        raise KeyError(f"unknown observable field {field_name!r}")
    if op not in _OPS:
        raise KeyError(f"unknown observable op {op!r}")
    idx = names.index(field_name)
    cutoff = fields[idx].cutoff
    op_local = _OPS[op](cutoff)
    dims = tuple(f.cutoff for f in fields)
    return embed_op(op_local, idx, dims)
