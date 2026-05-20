"""Quantum predictive coder: PCN on an MPS substrate.

This is the fully-quantum version of the PCN architecture. The "belief"
is a many-body quantum state |Psi> stored as an MPS. The "generative
model" is the Hamiltonian H[theta] whose Schrodinger dynamics define how
beliefs evolve. Observations enter as target expectation values of local
operators (e.g., number operators) at boundary sites; prediction errors
drive gradient updates on H's parameters.

What lives where, compared to the classical PCN:

    classical PCN          quantum PCN
    ---------------------  -----------------------------------------
    Phi(x)  (real array)   |Psi>  (MPS state, complex amplitudes)
    g_l(Phi)               H[theta] generating Schrodinger evolution
    prediction error       (target <O> - <Psi|O|Psi>)
    free-energy descent    imag-time relaxation + grad on theta
    metric g_{mu nu}       per-site curvature -> local omega (mass)
    stress-energy T        <Psi|T_local|Psi>  feeds back into geometry

Multi-field sits inside the local Hilbert space at each site as a tensor
product over species. Genuine inter-site entanglement is carried by the
MPS bond dimension. Vacuum is |0,0,...,0>, particles are excitations
above it created by a^\dagger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .mps import MPS
from .hamiltonian import Hamiltonian, HamiltonianConfig, FieldSpecies
from .evolution import trotter_step, energy
from ..manifold import Manifold2D


@dataclass
class QPCNConfig:
    """Configuration for the quantum predictive coder."""

    species: list[FieldSpecies]
    N_sites: int
    chi_max: int = 16
    dt_real: float = 0.05         # real-time step (Lorentzian)
    dt_imag: float = 0.05         # imaginary-time step (relaxation)
    imag_steps_per_observe: int = 2
    real_steps_per_observe: int = 1
    learn_rate: float = 0.05
    # Parameters that are tunable by gradient descent. Names follow the
    # convention in Hamiltonian.update_param.
    learnable_params: list[str] = field(default_factory=list)
    # Mapping from observed channel name to (site, operator-name on that species).
    observable_map: list[tuple[int, str, str]] = field(default_factory=list)
    # Optional curvature coupling.
    use_manifold: bool = False
    manifold_kappa: float = 0.02      # how strongly the QFT energy feeds the metric


class QPCN:
    """Quantum predictive coder.

    The flow per `observe`:
        1. Imaginary-time relaxation (drives Psi toward H's ground manifold).
        2. Real-time Schrodinger evolution (Lorentzian propagation).
        3. Compute predicted expectations <Psi|O_i|Psi> at observable sites.
        4. Compute prediction errors err_i = obs_i - <O_i>.
        5. Finite-difference gradient of |err|^2 w.r.t. each learnable param,
           applied as one SGD step.
        6. If use_manifold: update the classical metric from the energy
           density expectation along the snake path.
    """

    def __init__(self, cfg: QPCNConfig, manifold: Manifold2D | None = None,
                 rng: np.random.Generator | None = None):
        self.cfg = cfg
        rng = rng if rng is not None else np.random.default_rng(0)
        # Map snake path to manifold coordinates if a manifold is supplied.
        self.manifold = manifold
        self.snake_coords = self._snake_coords(cfg.N_sites,
                                               manifold.nx if manifold else None,
                                               manifold.ny if manifold else None)
        curvature = self._sample_curvature()
        hcfg = HamiltonianConfig(species=cfg.species)
        self.H = Hamiltonian(hcfg, N=cfg.N_sites, curvature=curvature)
        self.state = MPS.vacuum(cfg.N_sites, self.H.d_local)
        # Seed a tiny non-vacuum perturbation so dynamics aren't a no-op.
        for k in range(cfg.N_sites):
            # Mix a sprinkle of the |1,0,..> state in.
            local = self.state.tensors[k][0, :, 0].copy()
            local += 0.01 * (rng.standard_normal(self.H.d_local)
                             + 1j * rng.standard_normal(self.H.d_local))
            local /= np.linalg.norm(local)
            self.state.tensors[k] = local.reshape(1, self.H.d_local, 1)
        self._step = 0

    # ---- snake-path helpers ------------------------------------------------

    @staticmethod
    def _snake_coords(N: int, nx: int | None, ny: int | None
                      ) -> list[tuple[int, int]] | None:
        if nx is None or ny is None:
            return None
        coords: list[tuple[int, int]] = []
        for r in range(nx):
            row = range(ny) if r % 2 == 0 else range(ny - 1, -1, -1)
            for c in row:
                coords.append((r, c))
        # Truncate or repeat as needed to match N_sites.
        if len(coords) >= N:
            return coords[:N]
        # If N > grid size, tile by wrapping.
        out: list[tuple[int, int]] = []
        while len(out) < N:
            out.extend(coords)
        return out[:N]

    def _sample_curvature(self) -> np.ndarray:
        if self.manifold is None or self.snake_coords is None:
            return np.zeros(self.cfg.N_sites)
        R = self.manifold.ricci_scalar()
        return np.array([float(R[r, c]) for (r, c) in self.snake_coords])

    # ---- observation interface --------------------------------------------

    def _build_observable(self, site: int, species: str,
                          op_name: str) -> np.ndarray:
        ops = {
            "n": self.H.n(species),
            "phi": self.H.phi(species),
        }
        if op_name not in ops:
            raise KeyError(f"unknown observable '{op_name}', "
                           f"available: {list(ops)}")
        return ops[op_name]

    def predict(self) -> dict[tuple[int, str, str], float]:
        """Current predictions <Psi|O|Psi> for every observable in the map."""
        out = {}
        for (site, species, op_name) in self.cfg.observable_map:
            op = self._build_observable(site, species, op_name)
            out[(site, species, op_name)] = float(np.real(
                self.state.local_expectation(site, op)))
        return out

    # ---- one PCN step ------------------------------------------------------

    def observe(self, targets: dict[tuple[int, str, str], float],
                learn: bool = True) -> dict:
        """One predictive-coding update against a dict of target expectations.

        Keys of `targets` should be a subset of cfg.observable_map entries.
        Missing entries are ignored.
        """
        # 1. Refresh the curvature snapshot from the (possibly evolving) manifold.
        if self.manifold is not None and self.cfg.use_manifold:
            self.H.curvature = self._sample_curvature()

        # 2. Hybrid imaginary-then-real time evolution.
        for _ in range(self.cfg.imag_steps_per_observe):
            trotter_step(self.state, self.H, self.cfg.dt_imag,
                         imaginary=True, chi_max=self.cfg.chi_max)
            self.state.normalize()
        for _ in range(self.cfg.real_steps_per_observe):
            trotter_step(self.state, self.H, self.cfg.dt_real,
                         imaginary=False, chi_max=self.cfg.chi_max)

        # 3. Compute predictions and prediction errors.
        preds = self.predict()
        errors = {k: targets[k] - preds[k]
                  for k in preds if k in targets}
        sq_err = float(sum(e * e for e in errors.values()))

        # 4. Gradient descent on learnable parameters (finite difference).
        if learn and self.cfg.learnable_params and errors:
            self._grad_step(targets)

        # 5. Manifold coupling: stress-energy of the QFT sources curvature.
        if self.manifold is not None and self.cfg.use_manifold:
            self._update_manifold()

        self._step += 1
        E = energy(self.state, self.H)
        return {
            "step": self._step,
            "energy": E,
            "sq_error": sq_err,
            "predictions": preds,
            "errors": errors,
            "max_bond": max(self.state.bond_dimensions()),
            "entropy_mid": self.state.entanglement_entropy(
                self.cfg.N_sites // 2),
        }

    # ---- gradient on Hamiltonian parameters --------------------------------

    def _objective(self, targets: dict[tuple[int, str, str], float]) -> float:
        preds = self.predict()
        return float(sum((targets[k] - preds[k]) ** 2
                         for k in preds if k in targets))

    def _grad_step(self, targets: dict[tuple[int, str, str], float]) -> None:
        """Central-difference gradient on H's learnable parameters.

        For each parameter p we measure how the *next-step* state responds:
        propagate one short imag-then-real micro-step at p +/- eps from a
        cloned state, score the resulting predictions, and step against
        the finite-difference gradient. This captures the state's
        derivative w.r.t. p, not just the objective at the current state
        (which would be exactly zero — predictions don't depend on H
        without re-evolution).
        """
        eps = 0.05  # large enough to capture parabolic dependence near zero
        micro_dt_real = self.cfg.dt_real
        micro_dt_imag = self.cfg.dt_imag

        def score_after_perturbation(name: str, delta: float) -> float:
            p0 = self.H.get_param(name)
            self.H.update_param(name, p0 + delta)
            # Clone the current state cheaply and propagate one micro step.
            probe = self.state.copy()
            trotter_step(probe, self.H, micro_dt_imag, imaginary=True,
                         chi_max=self.cfg.chi_max)
            probe.normalize()
            trotter_step(probe, self.H, micro_dt_real, imaginary=False,
                         chi_max=self.cfg.chi_max)
            # Compute objective on the probe state.
            s = 0.0
            for (site, species, op_name), target in targets.items():
                op = self._build_observable(site, species, op_name)
                pred = float(np.real(probe.local_expectation(site, op)))
                s += (target - pred) ** 2
            # Restore.
            self.H.update_param(name, p0)
            return s

        for name in self.cfg.learnable_params:
            f_plus = score_after_perturbation(name, +eps)
            f_minus = score_after_perturbation(name, -eps)
            grad = (f_plus - f_minus) / (2 * eps)
            p0 = self.H.get_param(name)
            new_p = float(np.clip(p0 - self.cfg.learn_rate * grad, -5.0, 5.0))
            self.H.update_param(name, new_p)

    # ---- manifold coupling -------------------------------------------------

    def _update_manifold(self) -> None:
        """Project QFT energy density onto the 2D manifold and source curvature.

        For each snake-path site k at coordinate (r, c), compute the local
        energy density e_k = <Psi|H_local(k)|Psi>. Build a 2D scalar field
        from these values and call the manifold to deform its metric.
        """
        if self.manifold is None or self.snake_coords is None:
            return
        nx, ny = self.manifold.nx, self.manifold.ny
        e_field = np.zeros((1, nx, ny))
        for k, (r, c) in enumerate(self.snake_coords[:self.cfg.N_sites]):
            e_k = float(np.real(
                self.state.local_expectation(k, self.H.local_op(k))))
            e_field[0, r, c] += e_k * self.cfg.manifold_kappa
        # Use the manifold's own stress-energy machinery on a field whose
        # gradient is the local energy variation.
        self.manifold.update_metric(e_field, dt=1.0)
