"""Dynamic 2D Riemannian manifold for the PCN-QFT hybrid.

The metric is g_{mu nu} = eta_{mu nu} + h_{mu nu}[E], where the perturbation
h is sourced by the stress-energy of the prediction-error field. The
Laplace-Beltrami operator built from this metric replaces the flat Laplacian
when diffusing beliefs across the substrate.

Discretization: central finite differences on a unit-spaced grid with
periodic boundaries. Periodicity keeps the operators translation-invariant
and avoids boundary artifacts in the prototype.
"""

from __future__ import annotations

import numpy as np


def _grad(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference gradient with periodic boundaries.

    Works for arrays of shape (..., Nx, Ny). Returns (d/dx, d/dy)."""
    dx = 0.5 * (np.roll(field, -1, axis=-2) - np.roll(field, 1, axis=-2))
    dy = 0.5 * (np.roll(field, -1, axis=-1) - np.roll(field, 1, axis=-1))
    return dx, dy


def _div(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Divergence of a 2D vector field with periodic boundaries."""
    dvx = 0.5 * (np.roll(vx, -1, axis=-2) - np.roll(vx, 1, axis=-2))
    dvy = 0.5 * (np.roll(vy, -1, axis=-1) - np.roll(vy, 1, axis=-1))
    return dvx + dvy


class Manifold2D:
    """A discretized 2D manifold with a dynamic metric tensor.

    The metric perturbation `h` is a symmetric 2x2 tensor stored as three
    independent components (h_xx, h_xy, h_yy) at every grid point. The metric
    itself is g = I + h, kept symmetric and bounded so that g stays
    positive-definite (det g > 0).
    """

    def __init__(self, nx: int, ny: int, kappa: float = 0.05,
                 metric_decay: float = 0.02, h_clip: float = 0.6):
        self.nx = nx
        self.ny = ny
        self.kappa = kappa          # coupling: how strongly errors warp space
        self.metric_decay = metric_decay  # h relaxes back to 0 at this rate
        self.h_clip = h_clip        # bound on metric perturbation magnitude
        # Metric perturbation components.
        self.h_xx = np.zeros((nx, ny))
        self.h_xy = np.zeros((nx, ny))
        self.h_yy = np.zeros((nx, ny))

    # ---- metric accessors ---------------------------------------------------

    def g(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (g_xx, g_xy, g_yy)."""
        return 1.0 + self.h_xx, self.h_xy, 1.0 + self.h_yy

    def det_g(self) -> np.ndarray:
        gxx, gxy, gyy = self.g()
        return np.maximum(gxx * gyy - gxy * gxy, 1e-6)

    def g_inv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inverse metric (g^xx, g^xy, g^yy)."""
        gxx, gxy, gyy = self.g()
        det = self.det_g()
        return gyy / det, -gxy / det, gxx / det

    def sqrt_det_g(self) -> np.ndarray:
        return np.sqrt(self.det_g())

    # ---- differential operators --------------------------------------------

    def laplace_beltrami(self, phi: np.ndarray) -> np.ndarray:
        """Laplace-Beltrami operator on a (possibly multi-channel) field.

        Delta phi = (1/sqrt|g|) d_mu (sqrt|g| g^{mu nu} d_nu phi).

        For multi-channel input phi of shape (C, Nx, Ny), the operator is
        applied per channel — channels are independent scalar fields here.
        """
        gxx_inv, gxy_inv, gyy_inv = self.g_inv()
        sqrt_g = self.sqrt_det_g()

        dx, dy = _grad(phi)
        # Contract with inverse metric: V^mu = g^{mu nu} d_nu phi.
        vx = gxx_inv * dx + gxy_inv * dy
        vy = gxy_inv * dx + gyy_inv * dy
        # Multiply by volume element before taking divergence.
        wx = sqrt_g * vx
        wy = sqrt_g * vy
        return _div(wx, wy) / sqrt_g

    def stress_energy(self, e_field: np.ndarray) -> tuple[np.ndarray,
                                                          np.ndarray,
                                                          np.ndarray]:
        """Stress-energy components (T_xx, T_xy, T_yy) of a scalar field E.

        T_{mu nu} = d_mu E d_nu E - 1/2 g_{mu nu} g^{ab} d_a E d_b E.

        For multi-channel E (shape (C, Nx, Ny)), components are summed over
        channels — each channel contributes additively, as independent scalar
        excitations of the same substrate.
        """
        dx, dy = _grad(e_field)
        gxx_inv, gxy_inv, gyy_inv = self.g_inv()
        # |dE|^2 in the metric.
        de2 = gxx_inv * dx * dx + 2.0 * gxy_inv * dx * dy + gyy_inv * dy * dy
        if e_field.ndim == 3:
            t_xx = (dx * dx).sum(axis=0)
            t_xy = (dx * dy).sum(axis=0)
            t_yy = (dy * dy).sum(axis=0)
            de2_sum = de2.sum(axis=0)
        else:
            t_xx = dx * dx
            t_xy = dx * dy
            t_yy = dy * dy
            de2_sum = de2
        gxx, gxy, gyy = self.g()
        t_xx -= 0.5 * gxx * de2_sum
        t_xy -= 0.5 * gxy * de2_sum
        t_yy -= 0.5 * gyy * de2_sum
        return t_xx, t_xy, t_yy

    def ricci_scalar(self) -> np.ndarray:
        """Linearized Ricci scalar curvature.

        Exact R for a 2D metric is messy; for small perturbations h we use
        the linearized expression
            R ~ d_x d_y h_xy - 1/2 (d_y^2 h_xx + d_x^2 h_yy)
        scaled by 2 so it matches conventional sign for a sphere-like bump.
        This is sufficient as a regularizer in the free-energy functional.
        """
        # Second derivatives via repeated central difference.
        d2h_xx_yy = (np.roll(self.h_xx, -1, axis=-1)
                     - 2.0 * self.h_xx
                     + np.roll(self.h_xx, 1, axis=-1))
        d2h_yy_xx = (np.roll(self.h_yy, -1, axis=-2)
                     - 2.0 * self.h_yy
                     + np.roll(self.h_yy, 1, axis=-2))
        # Mixed derivative d_x d_y h_xy.
        d_x_hxy = 0.5 * (np.roll(self.h_xy, -1, axis=-2)
                         - np.roll(self.h_xy, 1, axis=-2))
        d2h_xy_mixed = 0.5 * (np.roll(d_x_hxy, -1, axis=-1)
                              - np.roll(d_x_hxy, 1, axis=-1))
        return 2.0 * d2h_xy_mixed - (d2h_xx_yy + d2h_yy_xx)

    # ---- dynamics ----------------------------------------------------------

    def update_metric(self, e_field: np.ndarray, dt: float) -> None:
        """Evolve metric perturbation toward error stress-energy.

        dh/dt = kappa * T[E] - decay * h, then clip to keep |h| < h_clip so
        that det g stays away from zero.
        """
        t_xx, t_xy, t_yy = self.stress_energy(e_field)
        self.h_xx += dt * (self.kappa * t_xx - self.metric_decay * self.h_xx)
        self.h_xy += dt * (self.kappa * t_xy - self.metric_decay * self.h_xy)
        self.h_yy += dt * (self.kappa * t_yy - self.metric_decay * self.h_yy)
        np.clip(self.h_xx, -self.h_clip, self.h_clip, out=self.h_xx)
        np.clip(self.h_xy, -self.h_clip, self.h_clip, out=self.h_xy)
        np.clip(self.h_yy, -self.h_clip, self.h_clip, out=self.h_yy)

    def reset(self) -> None:
        self.h_xx.fill(0.0)
        self.h_xy.fill(0.0)
        self.h_yy.fill(0.0)
