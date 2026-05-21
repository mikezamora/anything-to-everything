"""compose_hamiltonians: structural sum at the expectation level.

Per controller resolution #4: H_total.terms = concatenation, .term_energy
dispatches to the originating sub-Hamiltonian by term identity,
.total_energy = sum of the two. This is operator-sum at the expectation
level, not a dense addition — both H_typing and H_eval factor through
factored expectations and never materialize d_local^2 operators.
"""

from __future__ import annotations

from src.qft_pcn.qft.mps import MPS


class IncompatibleHamiltonians(Exception):
    """compose_hamiltonians() received Hamiltonians with mismatching N."""


class ComposedHamiltonian:
    """Operator-sum wrapper. .terms is the concatenation; .term_energy
    routes by identity to the originating sub-Hamiltonian; .total_energy
    sums the two .total_energy calls.
    """

    def __init__(self, *hamiltonians):
        if not hamiltonians:
            raise ValueError("compose_hamiltonians requires >=1 input")
        N = hamiltonians[0].N
        for h in hamiltonians[1:]:
            if h.N != N:
                raise IncompatibleHamiltonians(
                    f"N mismatch: {N} vs {h.N}")
        self._hams = tuple(hamiltonians)
        self.N = N
        # Map id(term) -> sub-Hamiltonian.
        self._owner = {}
        merged = []
        for h in self._hams:
            for t in h.terms:
                merged.append(t)
                self._owner[id(t)] = h
        self.terms = merged

    def term_energy(self, state: MPS, term, envs=None) -> float:
        owner = self._owner.get(id(term))
        if owner is None:
            # Fall back to membership search in each sub-Hamiltonian
            # (covers freshly-constructed equivalent terms).
            for h in self._hams:
                if term in getattr(h, "_terms_set", ()):
                    return h.term_energy(state, term, envs=envs)
            raise KeyError(term)
        return owner.term_energy(state, term, envs=envs)

    def total_energy(self, state: MPS) -> float:
        return sum(h.total_energy(state) for h in self._hams)

    def residuals(self, state: MPS) -> dict:
        out = {}
        for h in self._hams:
            r = h.residuals(state)
            for key, val in r.items():
                if key in out:
                    # Disambiguate by prepending the Hamiltonian's class
                    # name (rare in practice; B and C use distinct rule
                    # names so there are no collisions).
                    out[(type(h).__name__, *key) if isinstance(key, tuple)
                        else (type(h).__name__, key)] = val
                else:
                    out[key] = val
        return out


def compose_hamiltonians(*hamiltonians):
    """Compose two or more sum-of-terms Hamiltonians into one.

    The result's `total_energy` equals the sum of the inputs'
    `total_energy` on every state (operator addition at the expectation
    level). `term_energy` dispatches by term identity.

    All inputs must share the same N. They need not share rule names.
    """
    return ComposedHamiltonian(*hamiltonians)
