"""BridgeHamiltonian: thin wrapper exposing local_op(k)/bond_op(k) from
a list of LocalTerm / TwoSiteTerm objects.

Compatible with qft/evolution.py:trotter_step's expectations: only
local_op(k), bond_op(k), local_identity(), N, and d_local are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np

from ..dsl.term import FieldSpec, LocalTerm, TwoSiteTerm


@dataclass
class BridgeHamiltonian:
    fields: list[FieldSpec]
    sites: int
    terms: list = _field(default_factory=list)

    def __post_init__(self) -> None:
        self.N = self.sites
        self.d_local = 1
        for f in self.fields:
            self.d_local *= f.cutoff
        self._local: dict[int, np.ndarray] = {}
        self._bond: dict[int, np.ndarray] = {}
        for t in self.terms:
            if isinstance(t, LocalTerm):
                if t.site not in self._local:
                    self._local[t.site] = np.zeros(
                        (self.d_local, self.d_local), dtype=complex)
                self._local[t.site] = self._local[t.site] + t.operator
            elif isinstance(t, TwoSiteTerm):
                a, b = t.sites
                k = min(a, b)
                d2 = self.d_local * self.d_local
                if k not in self._bond:
                    self._bond[k] = np.zeros((d2, d2), dtype=complex)
                # If sites are reversed (b < a), the operator is in
                # (a, b) order; swap into (k, k+1) order via SWAP.
                if a < b:
                    op = t.operator
                else:
                    op = _swap_two_site(t.operator, self.d_local)
                self._bond[k] = self._bond[k] + op
            else:
                raise TypeError(f"unexpected term type {type(t).__name__}")

    def local_identity(self) -> np.ndarray:
        return np.eye(self.d_local, dtype=complex)

    def local_op(self, k: int) -> np.ndarray:
        if k in self._local:
            return self._local[k]
        return np.zeros((self.d_local, self.d_local), dtype=complex)

    def bond_op(self, k: int) -> np.ndarray:
        d2 = self.d_local * self.d_local
        if k in self._bond:
            return self._bond[k]
        return np.zeros((d2, d2), dtype=complex)


def _swap_two_site(op: np.ndarray, d: int) -> np.ndarray:
    """Given a (d^2, d^2) operator acting on (B, A), return the operator
    in (A, B) site order."""
    # Reshape to (a, b, a', b') and transpose to (b, a, b', a').
    t = op.reshape(d, d, d, d).transpose(1, 0, 3, 2)
    return t.reshape(d * d, d * d)
