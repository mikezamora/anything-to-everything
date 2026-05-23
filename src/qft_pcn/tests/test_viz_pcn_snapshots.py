"""PCN-side snapshot extractor tests.

Builds a real two-layer QFTPCNNetwork (no substrate mocks) and verifies
snapshot_pcn_fields / snapshot_pcn_dynamics / snapshot_pcn_coupling return
the documented shape. Defensive paths covered via a NetworkConfig that
strips optional attributes.
"""

import numpy as np
import pytest

from src.qft_pcn.layer import LayerConfig
from src.qft_pcn.network import NetworkConfig, QFTPCNNetwork
from src.qft_pcn.viz.snapshots import (
    snapshot_pcn_fields,
    snapshot_pcn_dynamics,
    snapshot_pcn_coupling,
)


@pytest.fixture
def net():
    rng = np.random.default_rng(0)
    cfg = NetworkConfig(layers=[LayerConfig(channels=1),
                                LayerConfig(channels=1)])
    n = QFTPCNNetwork(8, 8, cfg, rng=rng)
    obs = np.zeros((1, 8, 8))
    n.step(obs, learn=True)
    return n


def test_pcn_fields_shape(net):
    s = snapshot_pcn_fields(net)
    assert isinstance(s["layers"], list) and len(s["layers"]) == 2
    for layer in s["layers"]:
        assert "phi" in layer and "E" in layer and "Pi" in layer
        assert layer["channels"] == 1


def test_pcn_dynamics_shape(net):
    s = snapshot_pcn_dynamics(net)
    assert isinstance(s["total_free_energy"], float)
    assert isinstance(s["per_layer_free_energy"], list)
    assert len(s["per_layer_free_energy"]) == 2


def test_pcn_coupling_shape(net):
    s = snapshot_pcn_coupling(net)
    # Stress-energy + curvature aggregates from layer 0's error field.
    assert isinstance(s["mean_abs_stress_energy"], float)
    assert isinstance(s["mean_abs_ricci"], float)
    assert isinstance(s["kappa_R"], float)


def test_extractors_defensive_with_missing_attrs():
    # A bare object without manifold/layers must not raise.
    class Empty: pass
    assert snapshot_pcn_fields(Empty()) == {"layers": [], "step": None}
    assert snapshot_pcn_dynamics(Empty())["total_free_energy"] is None
    assert snapshot_pcn_coupling(Empty())["kappa_R"] is None


from src.qft_pcn.viz.runs import RunSpec, run_simulation


def test_run_simulation_emits_pcn_layer_states():
    spec = RunSpec(layers=["pcn-fields", "pcn-dynamics", "pcn-coupling"],
                   steps=2, grid=8)
    frames = list(run_simulation(spec))
    assert len(frames) == 2
    for key in ("pcn-fields", "pcn-dynamics", "pcn-coupling"):
        assert key in frames[0].layer_states
