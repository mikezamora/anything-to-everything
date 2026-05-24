"""Single hierarchical layer of the PCN-QFT hybrid.

A layer owns three coupled fields (Phi, E, Pi) and a *generative map* g_l
that projects its representation Phi_l down to a prediction of Phi_{l-1}.
The map is pluggable: the default is a learnable 3x3 convolution; a
variational quantum circuit (see `quantum.py`) can be dropped in via
`LayerConfig.gen_map` to make this layer's generative model live on a
quantum substrate.

The Lagrangian implemented per layer:

    L = 1/2 Pi |E|^2 - 1/2 log Pi  + kappa_R R(g)

with E = Phi_below - g_l(Phi_l) and R(g) the manifold's Ricci scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import numpy as np

from .fields import Field, PrecisionField
from .manifold import Manifold2D


@runtime_checkable
class GenerativeMap(Protocol):
    """Pluggable downward predictor.

    Implementations must be operating on (C, Nx, Ny) arrays end-to-end so the
    layer doesn't care whether the map is a classical convolution, a quantum
    circuit applied per-patch, or anything else.
    """

    def forward(self, phi: np.ndarray) -> np.ndarray: ...
    def grad_input(self, phi: np.ndarray, upstream: np.ndarray
                   ) -> np.ndarray: ...
    def update_params(self, phi: np.ndarray, upstream: np.ndarray,
                      lr: float) -> None: ...


def _conv2d_periodic(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Per-channel 3x3 convolution with periodic boundaries.

    x shape: (C, Nx, Ny); kernel shape: (C, 3, 3). Each channel uses its own
    kernel — no cross-channel mixing.
    """
    out = np.zeros_like(x)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            w = kernel[:, dx + 1, dy + 1][:, None, None]
            out += w * np.roll(np.roll(x, dx, axis=-2), dy, axis=-1)
    return out


class ClassicalConvMap:
    """Default generative map: per-channel learnable 3x3 conv + bias.

    forward(phi)            = conv(phi, k) + b
    grad_input(phi, u)      = conv(u, flip(k))    (transpose-convolution)
    update_params(phi, u)   = SGD on k, b minimizing -<u, conv(phi, k) + b>
    """

    def __init__(self, channels: int, rng: np.random.Generator):
        k = np.zeros((channels, 3, 3))
        k[:, 1, 1] = 1.0
        self.kernel = k + 0.02 * rng.standard_normal(k.shape)
        self.bias = np.zeros((channels, 1, 1))

    def forward(self, phi: np.ndarray) -> np.ndarray:
        return _conv2d_periodic(phi, self.kernel) + self.bias

    def grad_input(self, phi: np.ndarray, upstream: np.ndarray) -> np.ndarray:
        # Transpose conv: flip kernel along both spatial axes.
        flipped = self.kernel[:, ::-1, ::-1].copy()
        return _conv2d_periodic(upstream, flipped)

    def update_params(self, phi: np.ndarray, upstream: np.ndarray,
                      lr: float) -> None:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                shifted = np.roll(np.roll(phi, dx, axis=-2), dy, axis=-1)
                grad_w = -(upstream * shifted).mean(axis=(-1, -2))
                grad_w = np.clip(grad_w, -1.0, 1.0)
                self.kernel[:, dx + 1, dy + 1] -= lr * grad_w
        grad_b = -upstream.mean(axis=(-1, -2), keepdims=True)
        self.bias -= lr * np.clip(grad_b, -1.0, 1.0)
        self.kernel *= (1.0 - 1e-3)


@dataclass
class LayerConfig:
    channels: int = 1
    diffusion: float = 0.05      # D_Phi: belief-diffusion coefficient
    error_decay: float = 1.0     # lambda_E: error attractor = residual / lambda
    alpha: float = 0.05          # precision learning rate
    learn_rate: float = 0.01     # SGD rate for the generative map
    pi_min: float = 0.1          # floor on precision (avoid -log Pi blowup)
    pi_max: float = 10.0         # cap on precision (avoid runaway gain)
    phi_clip: float = 5.0        # safety clip on belief magnitude
    learn_every: int = 1         # apply gen_map update every N field steps
    gen_map: GenerativeMap | None = None  # if None, ClassicalConvMap is used


class QFTPCNLayer:
    """One level of the predictive hierarchy."""

    def __init__(self, manifold: Manifold2D, cfg: LayerConfig,
                 rng: np.random.Generator | None = None):
        self.manifold = manifold
        self.cfg = cfg
        rng = rng if rng is not None else np.random.default_rng(0)
        c, nx, ny = cfg.channels, manifold.nx, manifold.ny
        self.phi = Field.randn(c, nx, ny, scale=0.05, rng=rng)
        self.error = Field.zeros(c, nx, ny)
        self.precision = PrecisionField.ones(nx, ny)
        self.gen_map: GenerativeMap = (cfg.gen_map if cfg.gen_map is not None
                                       else ClassicalConvMap(c, rng))
        self._learn_tick = 0

    # ---- generative model --------------------------------------------------

    def predict_below(self) -> np.ndarray:
        """Downward prediction g_l(Phi_l) of the level below."""
        return self.gen_map.forward(self.phi.values)

    # ---- field updates -----------------------------------------------------

    def update_error(self, phi_below: np.ndarray, dt: float) -> None:
        """dE/dt = (phi_below - g(phi)) - lambda_E * E.

        Precision is applied where E couples back into other fields (belief
        drive, free energy) — not inside E's own dynamics, otherwise high
        precision drives larger E which drives higher precision (runaway).
        """
        residual = phi_below - self.predict_below()
        self.error.values += dt * (residual
                                   - self.cfg.error_decay
                                   * self.error.values)

    def update_phi(self, top_down: np.ndarray | None, dt: float) -> None:
        """Update Phi_l: backward drive through g_l from the precision-weighted
        error, plus geometric diffusion and an optional top-down message.
        """
        pi = self.precision.pi[None, :, :]
        drive = self.gen_map.grad_input(self.phi.values,
                                        pi * self.error.values)
        lb = self.manifold.laplace_beltrami(self.phi.values)
        dphi = drive + self.cfg.diffusion * lb
        if top_down is not None:
            dphi += top_down
        self.phi.values += dt * dphi
        np.clip(self.phi.values, -self.cfg.phi_clip, self.cfg.phi_clip,
                out=self.phi.values)

    def update_precision(self, dt: float) -> None:
        """dPi/dt = -dF/dPi at fixed point Pi = 1/E^2, applied in log space."""
        e2 = (self.error.values ** 2).mean(axis=0)
        pi = self.precision.pi
        d_log = self.cfg.alpha * (1.0 - pi * e2)
        self.precision.log_pi += dt * np.clip(d_log, -0.5, 0.5)
        log_min = np.log(self.cfg.pi_min)
        log_max = np.log(self.cfg.pi_max)
        np.clip(self.precision.log_pi, log_min, log_max,
                out=self.precision.log_pi)

    def learn_kernel(self, phi_below: np.ndarray) -> None:
        """One SGD-equivalent step on the generative map's parameters.

        Skipped according to cfg.learn_every so expensive quantum gradient
        evaluations don't run on every PDE step.
        """
        self._learn_tick += 1
        if self._learn_tick % self.cfg.learn_every != 0:
            return
        pi = self.precision.pi[None, :, :]
        residual = phi_below - self.predict_below()
        upstream = pi * residual
        self.gen_map.update_params(self.phi.values, upstream,
                                   self.cfg.learn_rate)

    # ---- diagnostics --------------------------------------------------------

    def free_energy(self, phi_below: np.ndarray, kappa_R: float = 0.01
                    ) -> float:
        pi = self.precision.pi
        e = phi_below - self.predict_below()
        e2 = (e * e).mean(axis=0)
        ricci = self.manifold.ricci_scalar()
        sqrt_g = self.manifold.sqrt_det_g()
        density = 0.5 * pi * e2 - 0.5 * np.log(pi) + kappa_R * ricci
        return float((density * sqrt_g).sum())

    def kl_divergence(self, phi_below: np.ndarray) -> float:
        """KL contribution of this layer's belief against its prior.

        The full per-layer free energy density (see ``free_energy``) is

            f = 1/2 * Pi * |E|^2 - 1/2 * log Pi + kappa_R * R

        The first two terms are the entropy-like view of the KL between
        the precision-weighted Gaussian belief and the unit-precision
        Gaussian prior (a constant offset of ``+1/2`` per pixel is dropped,
        leaving a non-negative log-likelihood gap up to that shift). The
        third term, ``kappa_R * R``, is the geometric *regulariser* over
        the manifold and is NOT part of the belief/prior KL — it is the
        curvature-coupling penalty added on top in the layer's free
        energy. ``kl_divergence`` therefore returns only the first two
        terms, integrated against ``sqrt(det g)``:

            KL = sum_x [ 1/2 * Pi(x) * |E(x)|^2 - 1/2 * log Pi(x) ] * sqrt_g(x)

        with ``E = phi_below - g(phi)`` matching ``free_energy``.
        """
        pi = self.precision.pi
        e = phi_below - self.predict_below()
        e2 = (e * e).mean(axis=0)
        sqrt_g = self.manifold.sqrt_det_g()
        density = -0.5 * np.log(pi) + 0.5 * pi * e2
        return float((density * sqrt_g).sum())
