"""Tests for the quantum generative map.

Verify that the variational circuit acts as a sensible nonlinear function,
that parameter-shift gradients agree with finite differences, and that
plugging a QuantumConvMap into a layer keeps the PCN dynamics stable.
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.quantum import QuantumGenerativeMap, QuantumConvMap
from src.qft_pcn.layer import LayerConfig
from src.qft_pcn.network import NetworkConfig, QFTPCNNetwork


def test_forward_returns_values_in_unit_range():
    rng = np.random.default_rng(0)
    qmap = QuantumGenerativeMap(n_qubits=4, n_layers=2, rng=rng)
    for _ in range(5):
        x = rng.standard_normal(4) * 0.5
        y = qmap.forward(x)
        assert y.shape == (4,)
        assert (np.abs(y) <= 1.0 + 1e-9).all(), (
            "Pauli-Z expectations must lie in [-1, 1]")


def test_parameter_shift_matches_finite_difference():
    """Parameter-shift gradient should match a small finite-difference check.

    Both compute df/dtheta exactly in the noiseless simulator regime, so
    they should agree to numerical precision."""
    rng = np.random.default_rng(2)
    qmap = QuantumGenerativeMap(n_qubits=3, n_layers=1, rng=rng)
    x = rng.standard_normal(3) * 0.3
    upstream = rng.standard_normal(3)

    g_ps = qmap.parameter_shift_grad(x, upstream)

    eps = 1e-4
    g_fd = np.zeros_like(qmap.theta)
    for idx in np.ndindex(*qmap.theta.shape):
        theta_plus = qmap.theta.copy(); theta_plus[idx] += eps
        theta_minus = qmap.theta.copy(); theta_minus[idx] -= eps
        fp = qmap._expectations(x, theta_plus)
        fm = qmap._expectations(x, theta_minus)
        g_fd[idx] = float(np.dot(upstream, (fp - fm) / (2 * eps)))

    assert np.allclose(g_ps, g_fd, atol=1e-3), (
        f"parameter-shift disagrees with finite difference: "
        f"max diff = {np.abs(g_ps - g_fd).max():.4e}")


def test_quantum_layer_runs_and_stays_finite():
    rng = np.random.default_rng(7)
    q_map = QuantumConvMap(channels=1, patch=2, n_layers=1, rng=rng)
    cfg = NetworkConfig(
        layers=[LayerConfig(channels=1, learn_rate=0.05, learn_every=5,
                            gen_map=q_map)],
        dt=0.4,
    )
    net = QFTPCNNetwork(8, 8, cfg, rng=rng)
    xs = np.arange(8)[:, None] - 4
    ys = np.arange(8)[None, :] - 4
    obs = np.exp(-(xs ** 2 + ys ** 2) / 3.0)[None]
    for _ in range(8):
        info = net.step(obs, learn=True)
        assert np.isfinite(info["free_energy"]), (
            "quantum layer produced non-finite free energy")


if __name__ == "__main__":
    test_forward_returns_values_in_unit_range()
    test_parameter_shift_matches_finite_difference()
    test_quantum_layer_runs_and_stays_finite()
    print("ok")
