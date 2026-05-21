"""Build the initial MPS from boundary clamps (spec §5.1 step 4)."""

from __future__ import annotations

import numpy as np

from ..dsl.term import FieldSpec
from src.qft_pcn.qft.mps import MPS


def build_initial_state(*, fields: list[FieldSpec], sites: int,
                        boundary: dict[int, dict[str, int]]) -> MPS:
    """Return a product MPS where each clamped site holds its clamped basis
    state on the named field (and vacuum on the others); unclamped sites are
    full vacuum (basis index 0)."""
    d_local = 1
    for f in fields:
        d_local *= f.cutoff
    cutoffs = [f.cutoff for f in fields]
    names = [f.name for f in fields]

    site_vecs: list[np.ndarray] = []
    for s in range(sites):
        per_species = [0] * len(fields)
        for fname, basis in (boundary.get(s) or {}).items():
            j = names.index(fname)
            per_species[j] = int(basis)
        # Row-major composite index: leftmost species changes slowest.
        idx = 0
        for j, val in enumerate(per_species):
            idx = idx * cutoffs[j] + val
        v = np.zeros(d_local, dtype=complex)
        v[idx] = 1.0
        site_vecs.append(v)

    return MPS.from_product(site_vecs)
