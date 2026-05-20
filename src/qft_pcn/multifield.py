"""Multi-field PCN-QFT: several field types sharing one manifold.

Each "field type" (think electron, photon, Higgs — or in cortical terms,
shape, motion, color) is its own QFTPCNNetwork. All field types live on the
SAME Manifold2D, so they share the substrate geometry. Their prediction-
error stress-energies all source the single metric, so curvature reflects
joint informativeness across modalities.

Interaction Lagrangian (Yukawa-like pairwise coupling):

    L_int = sum_{i<j} g_{ij}(x) Phi_i(x) Phi_j(x)

with learnable coupling fields g_{ij}. Variational gradient gives a
cross-field message added to each Phi's update:

    dPhi_i/dt  +=  - g_{ij} * Phi_j(x)         (for every coupled j)

The coupling constants themselves are learned by gradient descent on the
total free energy. A coupling that doesn't reduce free energy decays to
zero; pairs of fields that explain each other's residuals grow theirs.

This is the structural piece of QFT that "slots so nicely into PCN":
interactions are *local in space, polynomial in the fields*, and the
coupling strengths are themselves dynamical degrees of freedom that the
learner discovers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import numpy as np

from .layer import QFTPCNLayer, LayerConfig
from .manifold import Manifold2D


@dataclass
class MultiFieldConfig:
    field_names: list[str]
    layer_configs: dict[str, LayerConfig]
    # Initial coupling constants between pairs of fields. Pairs not present
    # default to zero coupling (free fields).
    coupling: dict[tuple[str, str], float] = field(default_factory=dict)
    learn_coupling: bool = True
    coupling_lr: float = 0.005
    coupling_max: float = 1.0
    dt: float = 0.4
    kappa_R: float = 0.02
    metric_update_every: int = 1


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


class MultiFieldNetwork:
    """Coupled PCN sub-networks on a shared manifold.

    Each "field" here is a single-layer PCN — the architecture composes
    fine with the hierarchical version too (just make each value of
    `self.fields` a list of layers), but a 1-layer-per-field version is the
    cleanest way to demonstrate the coupling.
    """

    def __init__(self, nx: int, ny: int, cfg: MultiFieldConfig,
                 rng: np.random.Generator | None = None):
        if not cfg.field_names:
            raise ValueError("at least one field is required")
        rng = rng if rng is not None else np.random.default_rng(0)
        self.cfg = cfg
        self.manifold = Manifold2D(nx, ny)
        self.fields: dict[str, QFTPCNLayer] = {
            name: QFTPCNLayer(self.manifold, cfg.layer_configs[name], rng=rng)
            for name in cfg.field_names
        }
        # Symmetric coupling table: g_{ij} == g_{ji}, stored once per pair.
        self.couplings: dict[tuple[str, str], float] = {}
        for a, b in combinations(cfg.field_names, 2):
            self.couplings[_pair_key(a, b)] = float(
                cfg.coupling.get(_pair_key(a, b),
                                 cfg.coupling.get((a, b), 0.0)))
        self._step = 0

    # ---- one step on observations for all fields ---------------------------

    def step(self, observations: dict[str, np.ndarray], learn: bool = True
             ) -> dict:
        """One coupled-PDE step.

        `observations` maps field name -> (C, Nx, Ny) input array. Each
        field's bottom-of-stack sees its own observation; cross-field
        information is exchanged via the interaction Lagrangian.
        """
        dt = self.cfg.dt

        # 1. Errors: each field compares its own prediction against its obs.
        for name, layer in self.fields.items():
            layer.update_error(observations[name], dt)

        # 2. Beliefs: own bottom-up drive + interaction with coupled fields.
        for name, layer in self.fields.items():
            # Build the interaction "top-down" message.
            interaction = np.zeros_like(layer.phi.values)
            for other_name, other_layer in self.fields.items():
                if other_name == name:
                    continue
                g = self.couplings[_pair_key(name, other_name)]
                if g == 0.0:
                    continue
                # Variational grad of L_int = g Phi_i Phi_j wrt Phi_i is g Phi_j.
                # Channel broadcasting: take the mean over channels of the
                # other field (so different-channel-count fields interoperate)
                # and broadcast across this field's channels.
                phi_j = other_layer.phi.values.mean(axis=0, keepdims=True)
                interaction += -g * np.broadcast_to(phi_j, layer.phi.values.shape)
            layer.update_phi(interaction, dt)

        # 3. Precision: each field independently.
        for layer in self.fields.values():
            layer.update_precision(dt)

        # 4. Metric: sourced by aggregate stress-energy of ALL fields' errors.
        if self._step % self.cfg.metric_update_every == 0:
            all_e = np.concatenate(
                [l.error.values for l in self.fields.values()], axis=0)
            self.manifold.update_metric(all_e, dt)

        # 5. Learning: generative maps and the coupling constants.
        if learn:
            for name, layer in self.fields.items():
                layer.learn_kernel(observations[name])
            if self.cfg.learn_coupling:
                self._learn_couplings()

        self._step += 1
        return self._diagnostics(observations)

    # ---- coupling learning -------------------------------------------------

    def _learn_couplings(self) -> None:
        """Gradient descent on coupling constants.

        dF/dg_{ij} ~ <Phi_i, Phi_j> averaged over the manifold (this is the
        functional derivative of the interaction term integrated against
        sqrt|g|). Negative inner product means the coupling helps reduce F.
        """
        sqrt_g = self.manifold.sqrt_det_g()
        for pair, g in list(self.couplings.items()):
            a, b = pair
            phi_a = self.fields[a].phi.values.mean(axis=0)
            phi_b = self.fields[b].phi.values.mean(axis=0)
            grad_g = float((phi_a * phi_b * sqrt_g).sum() / sqrt_g.sum())
            new_g = g - self.cfg.coupling_lr * grad_g
            new_g = float(np.clip(new_g, -self.cfg.coupling_max,
                                  self.cfg.coupling_max))
            self.couplings[pair] = new_g

    # ---- diagnostics --------------------------------------------------------

    def _diagnostics(self, observations: dict[str, np.ndarray]) -> dict:
        per_field_F = {}
        for name, layer in self.fields.items():
            per_field_F[name] = layer.free_energy(observations[name],
                                                  self.cfg.kappa_R)
        # Interaction contribution to F.
        sqrt_g = self.manifold.sqrt_det_g()
        interaction_F = 0.0
        for pair, g in self.couplings.items():
            a, b = pair
            phi_a = self.fields[a].phi.values.mean(axis=0)
            phi_b = self.fields[b].phi.values.mean(axis=0)
            interaction_F += g * float((phi_a * phi_b * sqrt_g).sum())
        return {
            "step": self._step,
            "per_field_F": per_field_F,
            "interaction_F": interaction_F,
            "total_F": sum(per_field_F.values()) + interaction_F,
            "couplings": dict(self.couplings),
            "mean_abs_curvature": float(
                np.abs(self.manifold.ricci_scalar()).mean()),
        }

    def free_energy(self, observations: dict[str, np.ndarray]) -> float:
        return self._diagnostics(observations)["total_F"]
