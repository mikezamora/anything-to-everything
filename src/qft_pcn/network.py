"""Multi-layer PCN-QFT network.

Layers are coupled bottom-up by prediction errors and top-down by generative
predictions. All layers share a single manifold instance: the metric is the
common substrate, and stress-energy from every layer's error field sources
its curvature. This realises the QFT analogy where multiple fields couple to
the same dynamical geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .layer import QFTPCNLayer, LayerConfig
from .manifold import Manifold2D


@dataclass
class NetworkConfig:
    layers: list[LayerConfig] = field(default_factory=list)
    dt: float = 0.5
    kappa_R: float = 0.01           # Ricci-scalar coupling in free energy
    metric_update_every: int = 1    # update metric every k field steps


class QFTPCNNetwork:
    def __init__(self, nx: int, ny: int, cfg: NetworkConfig,
                 rng: np.random.Generator | None = None):
        if not cfg.layers:
            raise ValueError("NetworkConfig.layers must be non-empty")
        rng = rng if rng is not None else np.random.default_rng(0)
        self.cfg = cfg
        self.manifold = Manifold2D(nx, ny)
        self.layers: list[QFTPCNLayer] = [
            QFTPCNLayer(self.manifold, lc, rng=rng) for lc in cfg.layers
        ]
        self._step = 0

    # ---- one PDE step on a single observation ------------------------------

    def step(self, observation: np.ndarray, learn: bool = True) -> dict:
        """Advance the coupled fields by one timestep on the given observation.

        `observation` shape must match the bottom layer's (C0, Nx, Ny). The
        function returns a small diagnostics dict (per-layer free energy,
        total free energy, max |E|).
        """
        dt = self.cfg.dt
        # 1. Errors propagate up: each layer compares its downward prediction
        #    against the layer below.
        below = observation
        for layer in self.layers:
            layer.update_error(below, dt)
            below = layer.phi.values

        # 2. Beliefs update: bottom-up drive from own error, top-down from
        #    next layer's error (via its kernel, applied to its precision*E).
        for i, layer in enumerate(self.layers):
            if i + 1 < len(self.layers):
                upper = self.layers[i + 1]
                pi_up = upper.precision.pi[None, :, :]
                # The top-down message is the residual that the upper layer
                # is currently predicting (i.e., its prediction at this level).
                # Channel counts must match for top-down injection; otherwise
                # we project via a mean over upper channels.
                td_prediction = upper.predict_below()
                if td_prediction.shape[0] == layer.phi.channels:
                    td = (td_prediction - layer.phi.values) * pi_up
                else:
                    td_mean = td_prediction.mean(axis=0, keepdims=True)
                    td = (td_mean - layer.phi.values.mean(axis=0,
                                                          keepdims=True)
                          ) * pi_up
                    td = np.broadcast_to(td, layer.phi.values.shape).copy()
            else:
                td = None
            below_i = observation if i == 0 else self.layers[i - 1].phi.values
            layer.update_phi(td, dt)

        # 3. Precision adapts to recent error magnitude.
        for layer in self.layers:
            layer.update_precision(dt)

        # 4. Metric perturbed by aggregate stress-energy of all error fields.
        if self._step % self.cfg.metric_update_every == 0:
            total_e = np.concatenate([l.error.values for l in self.layers],
                                     axis=0)
            self.manifold.update_metric(total_e, dt)

        # 5. Optional learning step on the generative kernels.
        if learn:
            below = observation
            for layer in self.layers:
                layer.learn_kernel(below)
                below = layer.phi.values

        self._step += 1
        return self._diagnostics(observation)

    # ---- diagnostics --------------------------------------------------------

    def _diagnostics(self, observation: np.ndarray) -> dict:
        per_layer_F = []
        below = observation
        for layer in self.layers:
            per_layer_F.append(layer.free_energy(below, self.cfg.kappa_R))
            below = layer.phi.values
        total_F = sum(per_layer_F)
        max_e = max(float(np.abs(l.error.values).max()) for l in self.layers)
        mean_curv = float(np.abs(self.manifold.ricci_scalar()).mean())
        return {
            "step": self._step,
            "free_energy": total_F,
            "per_layer_F": per_layer_F,
            "max_error": max_e,
            "mean_abs_curvature": mean_curv,
        }

    def free_energy(self, observation: np.ndarray) -> float:
        return self._diagnostics(observation)["free_energy"]
