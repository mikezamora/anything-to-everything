"""Single hierarchical layer of the PCN-QFT hybrid.

A layer owns three coupled fields (Phi, E, Pi) and a generative map g_l that
projects its representation Phi_l down to a prediction of Phi_{l-1}. The
generative map here is a small learnable convolution + bias — small enough
to keep things transparent, expressive enough to do nontrivial prediction.

The Lagrangian implemented per layer:

    L = 1/2 Pi |E|^2 - 1/2 log Pi  + kappa_R R(g)

with E = Phi_below - g_l(Phi_l) and R(g) the manifold's Ricci scalar.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .fields import Field, PrecisionField
from .manifold import Manifold2D


def _conv2d_periodic(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Per-channel 3x3 convolution with periodic boundaries.

    x shape: (C, Nx, Ny); kernel shape: (C, 3, 3). Each channel uses its own
    kernel — no cross-channel mixing. Cross-channel mixing is left to the
    diffusive Laplace-Beltrami step, keeping g_l explicitly per-mode.
    """
    out = np.zeros_like(x)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            w = kernel[:, dx + 1, dy + 1][:, None, None]
            out += w * np.roll(np.roll(x, dx, axis=-2), dy, axis=-1)
    return out


@dataclass
class LayerConfig:
    channels: int = 1
    diffusion: float = 0.05      # D_Phi: belief-diffusion coefficient
    error_decay: float = 1.0     # lambda_E: error attractor = residual / lambda
    alpha: float = 0.05          # precision learning rate
    learn_rate: float = 0.01     # SGD rate for the generative kernel
    pi_min: float = 0.1          # floor on precision (avoid -log Pi blowup)
    pi_max: float = 10.0         # cap on precision (avoid runaway gain)
    phi_clip: float = 5.0        # safety clip on belief magnitude


class QFTPCNLayer:
    """One level of the predictive hierarchy.

    The error field lives at the *interface below* the layer: E_l measures the
    mismatch between Phi_{l-1} (what the layer below currently represents) and
    g_l(Phi_l) (this layer's downward prediction). Phi_0 is the observed
    input and is not owned by any layer — it is provided externally to the
    bottom layer.
    """

    def __init__(self, manifold: Manifold2D, cfg: LayerConfig,
                 rng: np.random.Generator | None = None):
        self.manifold = manifold
        self.cfg = cfg
        rng = rng if rng is not None else np.random.default_rng(0)
        c, nx, ny = cfg.channels, manifold.nx, manifold.ny
        self.phi = Field.randn(c, nx, ny, scale=0.05, rng=rng)
        self.error = Field.zeros(c, nx, ny)
        self.precision = PrecisionField.ones(nx, ny)
        # Generative kernel g_l: small 3x3 Gaussian-ish init biased to identity
        # so the layer initially predicts its own representation.
        k = np.zeros((c, 3, 3))
        k[:, 1, 1] = 1.0
        self.kernel = k + 0.02 * rng.standard_normal(k.shape)
        self.bias = np.zeros((c, 1, 1))

    # ---- generative model --------------------------------------------------

    def predict_below(self) -> np.ndarray:
        """Downward prediction g_l(Phi_l) of the level below."""
        return _conv2d_periodic(self.phi.values, self.kernel) + self.bias

    # ---- field updates -----------------------------------------------------

    def update_error(self, phi_below: np.ndarray, dt: float) -> None:
        """Update E_l toward the raw residual phi_below - g(phi).

        dE/dt = (phi_below - g(phi)) - lambda_E * E.

        Precision is applied where E couples back into other fields (belief
        drive, free energy) — not inside E's own dynamics, otherwise high
        precision drives larger E which drives higher precision (runaway).
        This is the classical Friston/Bogacz precision-weighted PCN form.
        """
        residual = phi_below - self.predict_below()
        self.error.values += dt * (residual
                                   - self.cfg.error_decay
                                   * self.error.values)

    def update_phi(self, top_down: np.ndarray | None, dt: float) -> None:
        """Update Phi_l: diffusion on the manifold + downward error pressure
        + optional top-down message from layer above.

        The variational gradient of 1/2 Pi |E|^2 wrt Phi_l (through g_l) is
        -Pi * E pulled back through the kernel. With a per-channel 3x3
        convolution this is the same kernel applied to (Pi * E).
        """
        pi = self.precision.pi[None, :, :]
        drive = _conv2d_periodic(pi * self.error.values, self.kernel)
        lb = self.manifold.laplace_beltrami(self.phi.values)
        dphi = drive + self.cfg.diffusion * lb
        if top_down is not None:
            dphi += top_down
        self.phi.values += dt * dphi
        np.clip(self.phi.values, -self.cfg.phi_clip, self.cfg.phi_clip,
                out=self.phi.values)

    def update_precision(self, dt: float) -> None:
        """Precision tracks inverse error variance via gradient flow on F.

        With F = 1/2 Pi E^2 - 1/2 log Pi, descent gives
            dPi/dt = -dF/dPi = 1/(2 Pi) - 1/2 E^2,
        whose fixed point is Pi = 1/E^2. We apply the equivalent log-space
        update for numerical stability and floor/cap Pi so the -log Pi term
        in F can't diverge when E happens to be zero.
        """
        e2 = (self.error.values ** 2).mean(axis=0)
        pi = self.precision.pi
        d_log = self.cfg.alpha * (1.0 - pi * e2)
        self.precision.log_pi += dt * np.clip(d_log, -0.5, 0.5)
        log_min = np.log(self.cfg.pi_min)
        log_max = np.log(self.cfg.pi_max)
        np.clip(self.precision.log_pi, log_min, log_max,
                out=self.precision.log_pi)

    def learn_kernel(self, phi_below: np.ndarray) -> None:
        """SGD step on the generative kernel: minimize 1/2 Pi |E|^2 wrt kernel.

        Derivative of the error w.r.t. each kernel weight w_{c,dx,dy} is
        -shift(phi_l[c], -dx, -dy). We mean over manifold sites (not sum)
        so the learning rate is grid-size independent.
        """
        pi = self.precision.pi[None, :, :]
        residual = phi_below - self.predict_below()
        weighted = pi * residual
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                shifted = np.roll(np.roll(self.phi.values, dx, axis=-2),
                                  dy, axis=-1)
                grad_w = -(weighted * shifted).mean(axis=(-1, -2))
                # Clip the per-element gradient to bound updates when Pi is
                # near its cap and residuals are still large.
                grad_w = np.clip(grad_w, -1.0, 1.0)
                self.kernel[:, dx + 1, dy + 1] -= self.cfg.learn_rate * grad_w
        grad_b = -weighted.mean(axis=(-1, -2), keepdims=True)
        self.bias -= self.cfg.learn_rate * np.clip(grad_b, -1.0, 1.0)
        self.kernel *= (1.0 - 1e-3)

    # ---- diagnostics --------------------------------------------------------

    def free_energy(self, phi_below: np.ndarray, kappa_R: float = 0.01
                    ) -> float:
        """Per-layer variational free energy contribution.

        F = sum_x [ 1/2 Pi E^2 - 1/2 log Pi + kappa_R R(g) ] sqrt|g|.
        """
        pi = self.precision.pi
        # Re-derive E directly so this is a clean snapshot, not a stale field.
        e = phi_below - self.predict_below()
        e2 = (e * e).mean(axis=0)
        ricci = self.manifold.ricci_scalar()
        sqrt_g = self.manifold.sqrt_det_g()
        density = 0.5 * pi * e2 - 0.5 * np.log(pi) + kappa_R * ricci
        return float((density * sqrt_g).sum())
