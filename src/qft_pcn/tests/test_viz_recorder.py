"""Tests for the non-invasive viz instrumentation (Task 1)."""

from __future__ import annotations

import numpy as np

from src.qft_pcn.viz.schema import Frame
from src.qft_pcn.viz.recorder import Recorder
from src.qft_pcn.viz.snapshots import snapshot_network
from src.qft_pcn.network import QFTPCNNetwork, NetworkConfig
from src.qft_pcn.layer import LayerConfig


# ---- 1.1 Frame schema --------------------------------------------------------

def test_frame_json_roundtrip():
    frame = Frame(step=3, layer_states={"manifold": {"free_energy": 1.5}})
    blob = frame.to_json()
    restored = Frame.from_json(blob)
    assert restored.step == 3
    assert restored.layer_states["manifold"]["free_energy"] == 1.5


def test_frame_serializes_numpy_arrays():
    import json
    frame = Frame(step=0, layer_states={"mps": {"bonds": np.array([1, 4, 4, 1])}})
    blob = frame.to_json()
    parsed = json.loads(blob)
    assert parsed["layer_states"]["mps"]["bonds"] == [1, 4, 4, 1]


def test_frame_serializes_numpy_scalars_and_complex():
    import json
    frame = Frame(step=0, layer_states={"x": {
        "i": np.int64(7),
        "f": np.float64(2.5),
        "c": complex(1.0, -2.0),
    }})
    parsed = json.loads(frame.to_json())
    layer = parsed["layer_states"]["x"]
    assert layer["i"] == 7
    assert layer["f"] == 2.5
    assert layer["c"] == {"re": 1.0, "im": -2.0}


# ---- 1.5 snapshot extractors -------------------------------------------------

def test_snapshot_network_is_readonly():
    net = QFTPCNNetwork(8, 8, NetworkConfig(layers=[LayerConfig(channels=1)]))
    obs = np.zeros((1, 8, 8))
    net.step(obs)
    before = net.manifold.h_xx.copy()
    snap = snapshot_network(net)
    assert "fields" in snap and "metric_h" in snap
    np.testing.assert_array_equal(net.manifold.h_xx, before)


def test_snapshot_network_contents():
    net = QFTPCNNetwork(8, 8, NetworkConfig(layers=[LayerConfig(channels=1)]))
    obs = np.zeros((1, 8, 8))
    net.step(obs)
    snap = snapshot_network(net)
    assert "ricci" in snap
    assert "free_energy" in snap
    assert isinstance(snap["fields"], list)
    assert len(snap["fields"]) == 1
    layer0 = snap["fields"][0]
    assert "phi" in layer0 and "E" in layer0 and "Pi" in layer0


def test_snapshot_network_downsamples_large_grid():
    net = QFTPCNNetwork(64, 64, NetworkConfig(layers=[LayerConfig(channels=1)]))
    snap = snapshot_network(net)
    h = np.asarray(snap["metric_h"]["h_xx"])
    assert max(h.shape) <= 32


# ---- 1.8 recorder ------------------------------------------------------------

def test_recorder_captures_frames_without_changing_sim():
    net = QFTPCNNetwork(8, 8, NetworkConfig(layers=[LayerConfig(channels=1)]))
    obs = np.zeros((1, 8, 8))
    rec = Recorder()
    for _ in range(5):
        net.step(obs)
        rec.capture_network(net)
    assert len(rec.frames) == 5
    assert rec.frames[-1].step == 4
    # Round-trip the first frame.
    rec.frames[0].to_json()


def test_recorder_clear_and_to_json_list():
    net = QFTPCNNetwork(8, 8, NetworkConfig(layers=[LayerConfig(channels=1)]))
    obs = np.zeros((1, 8, 8))
    rec = Recorder()
    net.step(obs)
    rec.capture_network(net)
    blob = rec.to_json_list()
    import json
    assert isinstance(json.loads(blob), list)
    rec.clear()
    assert rec.frames == []


def test_recorder_generic_capture_merges_layers():
    net = QFTPCNNetwork(8, 8, NetworkConfig(layers=[LayerConfig(channels=1)]))
    obs = np.zeros((1, 8, 8))
    net.step(obs)
    rec = Recorder()
    from src.qft_pcn.viz.snapshots import snapshot_network
    rec.capture(network=snapshot_network(net), manifold={"free_energy": 0.0})
    assert len(rec.frames) == 1
    assert "network" in rec.frames[0].layer_states
    assert "manifold" in rec.frames[0].layer_states
