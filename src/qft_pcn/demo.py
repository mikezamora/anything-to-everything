"""Headless demo: drive a QFTPCNNetwork on a synthetic 2D pattern and show

  - free energy decreasing as the network settles,
  - manifold curvature concentrating where the input is most informative,
  - precision rising in low-error regions (gauge-like specialization).

Run:  python -m src.qft_pcn.demo
"""

from __future__ import annotations

import numpy as np

from .layer import LayerConfig
from .network import NetworkConfig, QFTPCNNetwork


def make_observation(nx: int, ny: int, t: float) -> np.ndarray:
    """A slowly drifting Gaussian blob — the network must track it."""
    xs = np.arange(nx)[:, None] - nx / 2
    ys = np.arange(ny)[None, :] - ny / 2
    cx = 4.0 * np.cos(0.05 * t)
    cy = 4.0 * np.sin(0.05 * t)
    blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / 12.0)
    return blob[None, :, :]  # (1, Nx, Ny)


def run(steps: int = 200, nx: int = 24, ny: int = 24,
        verbose: bool = True) -> list[dict]:
    cfg = NetworkConfig(
        layers=[
            LayerConfig(channels=1, diffusion=0.10, learn_rate=0.03),
            LayerConfig(channels=1, diffusion=0.05, learn_rate=0.01),
        ],
        dt=0.4,
        kappa_R=0.02,
    )
    net = QFTPCNNetwork(nx, ny, cfg, rng=np.random.default_rng(42))

    history: list[dict] = []
    for t in range(steps):
        obs = make_observation(nx, ny, t)
        info = net.step(obs, learn=True)
        history.append(info)
        if verbose and (t < 5 or t % 25 == 0 or t == steps - 1):
            print(f"step={t:4d}  F={info['free_energy']:+.3e}  "
                  f"|E|_max={info['max_error']:.3f}  "
                  f"<|R|>={info['mean_abs_curvature']:.3e}")

    if verbose:
        # Where in the manifold did precision concentrate?
        pi = net.layers[0].precision.pi
        peak = np.unravel_index(int(pi.argmax()), pi.shape)
        print(f"\nBottom-layer precision peak at {peak}, value={pi.max():.3f}")
        h_xx_rms = float(np.sqrt((net.manifold.h_xx ** 2).mean()))
        print(f"Metric perturbation RMS (h_xx): {h_xx_rms:.4f}")
        print(f"sqrt|g| range: [{net.manifold.sqrt_det_g().min():.3f}, "
              f"{net.manifold.sqrt_det_g().max():.3f}]")

    return history


if __name__ == "__main__":
    run()
