"""Per-layer KL divergence: substrate-side regression test for
QFTPCNLayer.kl_divergence.

Asserts the method returns a finite float for a 1-channel zero
observation on a small layer (the same setup snapshot_pcn_dynamics
uses for the bottom layer).
"""

import math

import numpy as np

from src.qft_pcn.layer import LayerConfig, QFTPCNLayer
from src.qft_pcn.manifold import Manifold2D


def test_kl_divergence_finite_for_zero_observation():
    manifold = Manifold2D(nx=8, ny=8)
    layer = QFTPCNLayer(manifold, LayerConfig(channels=1),
                        rng=np.random.default_rng(0))
    zero_obs = np.zeros((1, manifold.nx, manifold.ny))
    kl = layer.kl_divergence(zero_obs)
    assert isinstance(kl, float)
    assert math.isfinite(kl)


def test_kl_divergence_omits_curvature_term():
    """KL must not include the kappa_R * R regulariser: free_energy at
    kappa_R = 0 equals KL up to the geometric integral measure."""
    manifold = Manifold2D(nx=6, ny=6)
    layer = QFTPCNLayer(manifold, LayerConfig(channels=1),
                        rng=np.random.default_rng(1))
    obs = 0.1 * np.random.default_rng(2).standard_normal(
        (1, manifold.nx, manifold.ny))
    kl = layer.kl_divergence(obs)
    f_no_curv = layer.free_energy(obs, kappa_R=0.0)
    assert math.isclose(kl, f_no_curv, rel_tol=1e-9, abs_tol=1e-9)
