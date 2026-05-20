"""Sanity tests for the PCN-QFT prototype.

These are not learning-quality benchmarks — they verify that:
  - operators on the manifold reduce to the expected flat-space behavior
    when the metric perturbation is zero,
  - the stress-energy / curvature update keeps the metric well-conditioned,
  - the variational free energy actually decreases on a stationary input.
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.manifold import Manifold2D
from src.qft_pcn.layer import LayerConfig
from src.qft_pcn.network import NetworkConfig, QFTPCNNetwork


def test_flat_laplace_beltrami_matches_flat_laplacian():
    m = Manifold2D(16, 16)
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((1, 16, 16))
    lb = m.laplace_beltrami(phi)
    # Central-difference taken twice produces a 2-spaced Laplacian stencil:
    #   (phi(x+2) - 2 phi(x) + phi(x-2)) / 4 + (same in y).
    expected = 0.25 * (
        np.roll(phi, -2, axis=-2) + np.roll(phi, 2, axis=-2)
        + np.roll(phi, -2, axis=-1) + np.roll(phi, 2, axis=-1)
        - 4.0 * phi)
    assert np.allclose(lb, expected, atol=1e-10)


def test_metric_stays_positive_under_update():
    m = Manifold2D(20, 20, kappa=0.5)
    rng = np.random.default_rng(1)
    # Drive the metric with random fields for many steps.
    for _ in range(200):
        e = rng.standard_normal((2, 20, 20))
        m.update_metric(e, dt=0.5)
    det = m.det_g()
    assert (det > 0).all(), "metric lost positive-definiteness"
    assert np.isfinite(det).all()


def test_free_energy_decreases_on_stationary_input():
    cfg = NetworkConfig(
        layers=[LayerConfig(channels=1, learn_rate=0.04)],
        dt=0.4,
    )
    net = QFTPCNNetwork(16, 16, cfg, rng=np.random.default_rng(7))
    xs = np.arange(16)[:, None] - 8
    ys = np.arange(16)[None, :] - 8
    obs = np.exp(-(xs ** 2 + ys ** 2) / 8.0)[None, :, :]

    initial_F = net.free_energy(obs)
    for _ in range(80):
        net.step(obs, learn=True)
    final_F = net.free_energy(obs)
    assert final_F < initial_F, (
        f"free energy did not decrease: {initial_F:.3e} -> {final_F:.3e}")


def test_curvature_concentrates_near_input_features():
    cfg = NetworkConfig(
        layers=[LayerConfig(channels=1, learn_rate=0.03)],
        dt=0.4,
    )
    net = QFTPCNNetwork(20, 20, cfg, rng=np.random.default_rng(3))
    xs = np.arange(20)[:, None] - 10
    ys = np.arange(20)[None, :] - 10
    obs = np.exp(-(xs ** 2 + ys ** 2) / 6.0)[None, :, :]

    for _ in range(120):
        net.step(obs, learn=True)
    # Sum of squared metric perturbation should be larger near the blob
    # (the input gradient is highest on its flank) than at the far corner.
    h_mag = (net.manifold.h_xx ** 2 + net.manifold.h_xy ** 2
             + net.manifold.h_yy ** 2)
    center_mass = h_mag[6:14, 6:14].mean()
    corner_mass = h_mag[:4, :4].mean()
    assert center_mass > corner_mass, (
        f"curvature did not concentrate near input: "
        f"center={center_mass:.3e}, corner={corner_mass:.3e}")


if __name__ == "__main__":
    test_flat_laplace_beltrami_matches_flat_laplacian()
    test_metric_stays_positive_under_update()
    test_free_energy_decreases_on_stationary_input()
    test_curvature_concentrates_near_input_features()
    print("ok")
