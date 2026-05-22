"""compose_mera_hamiltonians: operator-sum of MERA Hamiltonians at the
expectation level (spec §9.5). Mirrors compose.py for the MERA substrate.
"""
from __future__ import annotations

from src.qft_pcn.qft.mera import MERA


class IncompatibleHamiltonians(Exception):
    """Inputs disagree on n_leaves / n_nodes."""


class ComposedMeraHamiltonian:
    """Operator-sum wrapper. .terms is the concatenation; .term_energy
    routes by term ownership; .total_energy sums the sub-Hamiltonians;
    .term_gates dispatches to the owning sub-Hamiltonian (spec §7.5).
    """

    def __init__(self, *hamiltonians):
        if not hamiltonians:
            raise ValueError("compose_mera_hamiltonians requires >=1 input")
        ref = hamiltonians[0].meta
        for h in hamiltonians[1:]:
            if (h.meta.n_leaves != ref.n_leaves
                    or h.meta.n_nodes != ref.n_nodes):
                raise IncompatibleHamiltonians(
                    f"meta mismatch: n_leaves {ref.n_leaves} vs "
                    f"{h.meta.n_leaves}, n_nodes {ref.n_nodes} vs "
                    f"{h.meta.n_nodes}")
        self._hams = tuple(hamiltonians)
        self.meta = ref
        self._owner = {}
        merged = []
        for h in self._hams:
            for t in h.terms:
                merged.append(t)
                self._owner[id(t)] = h
        self.terms = merged

    def _owner_of(self, term):
        owner = self._owner.get(id(term))
        if owner is not None:
            return owner
        for h in self._hams:
            if term in getattr(h, "_terms_set", ()):
                return h
        raise KeyError(term)

    def term_energy(self, state: MERA, term) -> float:
        return self._owner_of(term).term_energy(state, term)

    def total_energy(self, state: MERA) -> float:
        return sum(h.total_energy(state) for h in self._hams)

    def residuals(self, state: MERA) -> dict:
        out = {}
        for h in self._hams:
            for key, val in h.residuals(state).items():
                if key in out:
                    out[(type(h).__name__, *key)] = val
                else:
                    out[key] = val
        return out

    def term_gates(self, state: MERA, term, dt: float,
                   imaginary: bool = True):
        return self._owner_of(term).term_gates(state, term, dt, imaginary)


def compose_mera_hamiltonians(*hamiltonians):
    """Compose two or more MERA sum-of-terms Hamiltonians into one.

    The result's total_energy equals the sum of the inputs' total_energy
    on every state. All inputs must share n_leaves and n_nodes.
    """
    return ComposedMeraHamiltonian(*hamiltonians)
