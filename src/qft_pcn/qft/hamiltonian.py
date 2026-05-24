"""Local Hamiltonian construction for a 1D chain of multi-species bosonic
sites.

The Hamiltonian is built as a sum of one-site and two-site terms:

    H = sum_k  H1_k(theta)  +  sum_k  H2_{k,k+1}(theta)

Supported terms:

    Free part (one-site):
        omega_k * n_k        oscillator frequency  (mass)
        eta_k * phi_k^2       additional scalar mass term

    Hopping / kinetic (two-site, like a discretized gradient):
        -t_k (a_k^\dagger a_{k+1} + h.c.)    boson hopping
        -kappa_k (phi_k - phi_{k+1})^2        scalar kinetic gradient

    Within-site multi-species interactions:
        g_AB_k * n_A_k * n_B_k                density-density coupling
        lambda_AB_k * phi_A_k * phi_B_k       Yukawa-like field coupling

    Curvature coupling:
        omega_k modulated by local Ricci scalar R(x_k):  omega_k -> omega_0 (1 + xi * R(x_k))

Hamiltonians here are stored as a list of one-site operators (shape (d, d))
and two-site operators (shape (d^2, d^2)), one per bond. This is the
"site/bond decomposition" used by TEBD evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np

from .fock import (annihilation, creation, number, phi_op, identity,
                   embed_op, two_site_op)


@dataclass
class FieldSpecies:
    """A single quantum field species in the multi-field system."""

    name: str
    cutoff: int                   # local Fock-space cutoff d
    bare_mass: float = 1.0        # base omega
    kinetic: float = 0.5          # hopping amplitude t
    quartic: float = 0.0          # phi^4 self-interaction
    source: float = 0.0           # coherent source J (linear in phi)


@dataclass
class HamiltonianConfig:
    """Configuration for the multi-species 1D Hamiltonian."""

    species: list[FieldSpecies]
    # Pair interactions keyed by alphabetized species-name pair.
    density_couplings: dict[tuple[str, str], float] = field(default_factory=dict)
    yukawa_couplings: dict[tuple[str, str], float] = field(default_factory=dict)
    # Curvature coupling strength xi: omega -> omega (1 + xi * R(x)).
    curvature_xi: float = 0.0


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


class Hamiltonian:
    """Decomposed Hamiltonian for TEBD on N sites.

    The per-site Hilbert space has dimension d_local = prod_i species[i].cutoff,
    with species_dims = tuple(s.cutoff for s in species).

    `local_op(k)` returns the one-site operator at site k.
    `bond_op(k)` returns the two-site operator on bond (k, k+1).
    Both may depend on per-site curvature values supplied at construction.
    """

    def __init__(self, cfg: HamiltonianConfig, N: int,
                 curvature: np.ndarray | None = None):
        self.cfg = cfg
        self.N = N
        self.species = cfg.species
        self.species_dims = tuple(s.cutoff for s in cfg.species)
        self.d_local = int(np.prod(self.species_dims))
        self.curvature = (curvature if curvature is not None
                          else np.zeros(N))

        # Precompute the per-species single-site operators in the *local* basis.
        self._a = {}
        self._adag = {}
        self._n = {}
        self._phi = {}
        for i, s in enumerate(cfg.species):
            a_i = annihilation(s.cutoff)
            self._a[s.name] = embed_op(a_i, i, self.species_dims)
            self._adag[s.name] = embed_op(creation(s.cutoff), i,
                                          self.species_dims)
            self._n[s.name] = embed_op(number(s.cutoff), i,
                                       self.species_dims)
            self._phi[s.name] = embed_op(phi_op(s.cutoff), i,
                                         self.species_dims)

        # Parallel metadata enumeration of every active term contributing to
        # local_op(site) and bond_op(site). Mirrors the structure of
        # `_assemble_local` / `_assemble_bond` exactly; the op-construction
        # logic above/below is untouched. Each entry is a dict:
        #   {kind, species, site, coeff}
        # where `kind` is one of:
        #   'mass'     omega_k * n_k                 (one-site, single species)
        #   'quartic'  s.quartic * n^2               (one-site, single species)
        #   'source'   s.source * phi                (one-site, single species)
        #   'density'  g_ab * n_a * n_b              (one-site, pair species)
        #   'yukawa'   lambda_ab * phi_a * phi_b     (one-site, pair species)
        #   'kinetic'  -t (a†_k a_{k+1} + h.c.)      (two-site, single species)
        #   'curvature' xi * R(x_k) * omega_k * n_k  (one-site, single species)
        #                                            (zero entries omitted)
        # `site` is an int for one-site terms and a (k, k+1) tuple for
        # bond/kinetic terms. `species` is a str for single-species terms
        # and a (a, b) tuple for cross-species terms.
        terms: list[dict] = []
        for k in range(N):
            R = float(self.curvature[k]) if k < len(self.curvature) else 0.0
            mass_scale = 1.0 + cfg.curvature_xi * R
            for s in cfg.species:
                terms.append({"kind": "mass", "species": s.name,
                              "site": k, "coeff": float(s.bare_mass * mass_scale)})
                if cfg.curvature_xi != 0.0 and R != 0.0:
                    terms.append({"kind": "curvature", "species": s.name,
                                  "site": k,
                                  "coeff": float(s.bare_mass
                                                 * cfg.curvature_xi * R)})
                if s.source != 0.0:
                    terms.append({"kind": "source", "species": s.name,
                                  "site": k, "coeff": float(s.source)})
                if s.quartic != 0.0:
                    terms.append({"kind": "quartic", "species": s.name,
                                  "site": k, "coeff": float(s.quartic)})
            for pair, g in cfg.density_couplings.items():
                a, b = pair
                terms.append({"kind": "density", "species": (a, b),
                              "site": k, "coeff": float(g)})
            for pair, lam in cfg.yukawa_couplings.items():
                a, b = pair
                terms.append({"kind": "yukawa", "species": (a, b),
                              "site": k, "coeff": float(lam)})
        for k in range(N - 1):
            for s in cfg.species:
                if s.kinetic == 0.0:
                    continue
                terms.append({"kind": "kinetic", "species": s.name,
                              "site": (k, k + 1),
                              "coeff": float(-s.kinetic)})
        self.terms: list[dict] = terms

    # ---- operator accessors -----------------------------------------------

    def a(self, species: str) -> np.ndarray:
        return self._a[species]

    def adag(self, species: str) -> np.ndarray:
        return self._adag[species]

    def n(self, species: str) -> np.ndarray:
        return self._n[species]

    def phi(self, species: str) -> np.ndarray:
        return self._phi[species]

    def local_identity(self) -> np.ndarray:
        return np.eye(self.d_local, dtype=complex)

    # ---- term construction ------------------------------------------------

    def local_op(self, site: int) -> np.ndarray:
        """Sum of all one-site terms at this site."""
        h = np.zeros((self.d_local, self.d_local), dtype=complex)
        R = self.curvature[site] if site < self.N else 0.0
        mass_scale = 1.0 + self.cfg.curvature_xi * float(R)
        for s in self.cfg.species:
            omega = s.bare_mass * mass_scale
            h = h + omega * self._n[s.name]
            if s.source != 0.0:
                # Coherent source: shifts the vacuum to a non-zero <phi>,
                # and via <n> = J^2/(2 omega^2) creates particle population
                # without re-evolution being necessary at every site.
                h = h + s.source * self._phi[s.name]
            if s.quartic != 0.0:
                n_s = self._n[s.name]
                h = h + s.quartic * (n_s @ n_s)
        # Within-site cross-species interactions.
        for pair, g in self.cfg.density_couplings.items():
            a, b = pair
            h = h + g * (self._n[a] @ self._n[b])
        for pair, lam in self.cfg.yukawa_couplings.items():
            a, b = pair
            h = h + lam * (self._phi[a] @ self._phi[b])
        return h

    def bond_op(self, site: int) -> np.ndarray:
        """Two-site operator on bond (site, site+1).

        Includes per-species hopping; cross-species hopping is not used.
        Returns shape (d_local * d_local, d_local * d_local).
        """
        d = self.d_local
        h = np.zeros((d * d, d * d), dtype=complex)
        for s in self.cfg.species:
            if s.kinetic == 0.0:
                continue
            adag_a = two_site_op(self._adag[s.name], self._a[s.name])
            a_adag = two_site_op(self._a[s.name], self._adag[s.name])
            h = h - s.kinetic * (adag_a + a_adag)
        return h

    # ---- learnable parameter slots ----------------------------------------

    def update_param(self, name: str, value: float) -> None:
        """Set a named Hamiltonian parameter.

        Supported names:
            "<species>.mass"     : species.bare_mass
            "<species>.kinetic"  : species.kinetic
            "<species>.quartic"  : species.quartic
            "density.<a>.<b>"    : density-coupling g_ab
            "yukawa.<a>.<b>"     : yukawa coupling lambda_ab
            "curvature_xi"       : curvature_xi
        """
        if name == "curvature_xi":
            self.cfg.curvature_xi = float(value); return
        if name.startswith("density."):
            _, a, b = name.split(".")
            self.cfg.density_couplings[_pair_key(a, b)] = float(value); return
        if name.startswith("yukawa."):
            _, a, b = name.split(".")
            self.cfg.yukawa_couplings[_pair_key(a, b)] = float(value); return
        if "." in name:
            species_name, attr = name.split(".")
            attr_canonical = "bare_mass" if attr == "mass" else attr
            for s in self.cfg.species:
                if s.name == species_name:
                    setattr(s, attr_canonical, float(value))
                    return
        raise KeyError(f"unknown Hamiltonian parameter: {name}")

    def get_param(self, name: str) -> float:
        if name == "curvature_xi":
            return self.cfg.curvature_xi
        if name.startswith("density."):
            _, a, b = name.split(".")
            return self.cfg.density_couplings.get(_pair_key(a, b), 0.0)
        if name.startswith("yukawa."):
            _, a, b = name.split(".")
            return self.cfg.yukawa_couplings.get(_pair_key(a, b), 0.0)
        if "." in name:
            species_name, attr = name.split(".")
            attr_canonical = "bare_mass" if attr == "mass" else attr
            for s in self.cfg.species:
                if s.name == species_name:
                    return getattr(s, attr_canonical)
        raise KeyError(name)
