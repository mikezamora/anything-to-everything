"""Tests for the non-invasive viz instrumentation (Task 1)."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.viz.schema import Frame
from src.qft_pcn.viz.recorder import Recorder
from src.qft_pcn.viz.snapshots import (
    snapshot_network, snapshot_qpcn, snapshot_mps, snapshot_multifield,
    snapshot_hamiltonian, snapshot_mera, snapshot_vqc, snapshot_logic,
)
from src.qft_pcn.network import QFTPCNNetwork, NetworkConfig
from src.qft_pcn.layer import LayerConfig
from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.qft.qpcn import QPCN, QPCNConfig
from src.qft_pcn.qft.hamiltonian import (
    FieldSpecies, Hamiltonian, HamiltonianConfig,
)
from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.multifield import MultiFieldNetwork, MultiFieldConfig
from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian

# quantum.py depends on qiskit, an optional heavy dependency. Skip the VQC
# extractor tests (only) when it is unavailable rather than failing collection.
try:
    from src.qft_pcn.quantum import QuantumGenerativeMap, QuantumConvMap
    _HAS_QISKIT = True
except ImportError:  # pragma: no cover - depends on environment
    _HAS_QISKIT = False


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
    assert "mean_abs_ricci" in snap
    assert isinstance(snap["mean_abs_ricci"], float)
    assert isinstance(snap["fields"], list)
    assert len(snap["fields"]) == 1
    layer0 = snap["fields"][0]
    assert "phi" in layer0 and "E" in layer0 and "Pi" in layer0


def test_snapshot_network_downsamples_large_grid():
    net = QFTPCNNetwork(64, 64, NetworkConfig(layers=[LayerConfig(channels=1)]))
    snap = snapshot_network(net)
    h = np.asarray(snap["metric_h"]["h_xx"])
    assert max(h.shape) <= 32


def test_snapshot_mps_contents_and_readonly():
    mps = MPS.vacuum(4, 2)
    before = [t.copy() for t in mps.tensors]
    snap = snapshot_mps(mps)
    assert "bond_dims" in snap and "entropies" in snap
    assert snap["n_sites"] == 4
    assert snap["d_local"] == 2
    # 4 sites -> 5 bonds (incl. boundary) and 3 internal cut entropies.
    assert isinstance(snap["bond_dims"], list) and len(snap["bond_dims"]) >= 1
    assert isinstance(snap["entropies"], list) and len(snap["entropies"]) == 3
    # Vacuum is a product state: zero entanglement at every cut.
    for s in snap["entropies"]:
        assert s is not None and abs(s) < 1e-9
    # MPS unchanged.
    assert mps.N == 4
    for t, b in zip(mps.tensors, before):
        np.testing.assert_array_equal(t, b)


def test_snapshot_qpcn_contents():
    species = [FieldSpecies(name="phi", cutoff=2)]
    cfg = QPCNConfig(species=species, N_sites=3, chi_max=4,
                     learnable_params=["phi.mass"])
    q = QPCN(cfg)
    snap = snapshot_qpcn(q)
    for key in ("bond_dims", "entropies", "occupations",
                "energy", "pred_errors", "params"):
        assert key in snap
    # params must map the learnable param name to its real Hamiltonian value.
    assert snap["params"] == {"phi.mass": q.H.get_param("phi.mass")}
    assert isinstance(snap["params"]["phi.mass"], float)
    assert isinstance(snap["bond_dims"], list)
    assert isinstance(snap["entropies"], list) and len(snap["entropies"]) == 2
    assert isinstance(snap["occupations"], list) and len(snap["occupations"]) == 3


def test_snapshot_qpcn_after_observe():
    species = [FieldSpecies(name="phi", cutoff=2)]
    cfg = QPCNConfig(species=species, N_sites=3, chi_max=4,
                     learnable_params=["phi.mass"],
                     observable_map=[(0, "phi", "n")])
    q = QPCN(cfg)
    q.observe({(0, "phi", "n"): 0.5})
    snap = snapshot_qpcn(q)
    assert snap["step"] == 1
    assert snap["params"]["phi.mass"] is not None


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


# ---- 1.5b previously-untested extractors ------------------------------------

def test_snapshot_multifield_contents():
    cfg = MultiFieldConfig(
        field_names=["a", "b"],
        layer_configs={"a": LayerConfig(channels=1),
                       "b": LayerConfig(channels=1)},
        coupling={("a", "b"): 0.1},
    )
    mf = MultiFieldNetwork(8, 8, cfg)
    snap = snapshot_multifield(mf)
    assert set(snap["fields"]) == {"a", "b"}
    for name in ("a", "b"):
        f = snap["fields"][name]
        assert f["phi"] is not None and f["E"] is not None
        assert f["Pi"] is not None
    assert snap["couplings"] is not None
    assert snap["couplings"]["a|b"] == 0.1
    # mean_abs_coupling is the cheap scalar series consumed by the Manim scene.
    assert "mean_abs_coupling" in snap
    assert isinstance(snap["mean_abs_coupling"], float)
    assert snap["mean_abs_coupling"] == pytest.approx(0.1)
    assert snap["step"] == 0


def test_snapshot_hamiltonian_contents():
    cfg = HamiltonianConfig(species=[FieldSpecies(name="phi", cutoff=2)])
    H = Hamiltonian(cfg, N=3)
    snap = snapshot_hamiltonian(H)
    assert snap["n_sites"] == 3
    assert snap["d_local"] == 2
    assert snap["species_dims"] == [2]
    assert snap["species"] == ["phi"]
    assert snap["curvature"] is not None
    assert len(snap["curvature"]) == 3


def test_snapshot_mera_contents():
    m = MERA.vacuum(4, 2)
    snap = snapshot_mera(m)
    assert snap["n_leaves"] == 4
    assert isinstance(snap["layer_dims"], list)
    assert isinstance(snap["bond_dims"], list)
    # 4 leaves -> 3 internal cuts (0-indexed 0..2), none out of range.
    assert isinstance(snap["entropies"], list) and len(snap["entropies"]) == 3
    # Vacuum product MERA: zero entanglement at every cut.
    for s in snap["entropies"]:
        assert s is not None and abs(s) < 1e-9
    # iso_residuals: one float per MERA layer; vacuum isometries are exactly
    # canonical, so each layer's mean residual should be ~0.
    assert "iso_residuals" in snap
    assert isinstance(snap["iso_residuals"], list)
    assert len(snap["iso_residuals"]) == len(snap["layer_dims"])
    for r in snap["iso_residuals"]:
        assert isinstance(r, float)
        assert r < 1e-9


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
def test_snapshot_vqc_contents():
    vqc = QuantumGenerativeMap(n_qubits=2, n_layers=2)
    snap = snapshot_vqc(vqc)
    # theta is the real attribute name (was wrongly `params`).
    assert snap["theta"] is not None
    assert np.asarray(snap["theta"]).shape == (2, 2, 2)
    assert snap["n_qubits"] == 2
    assert snap["n_layers"] == 2
    assert snap["input_scale"] == 1.0


@pytest.mark.skipif(not _HAS_QISKIT, reason="qiskit not installed")
def test_snapshot_vqc_convmap_reads_inner():
    conv = QuantumConvMap(channels=1, patch=2, n_layers=1)
    snap = snapshot_vqc(conv)
    # QuantumConvMap delegates angles to the inner .qmap.
    assert snap["theta"] is not None
    assert snap["n_qubits"] == 4
    assert snap["bias"] is not None


def test_snapshot_logic_contents():
    enc = EvalHamiltonian(N=4)
    snap = snapshot_logic(enc)
    # Real attribute is `N` (was wrongly `n_sites`); `terms` exists.
    assert snap["n_sites"] == 4
    # 5 rules * (N - 1) terms.
    assert snap["term_count"] == 5 * (4 - 1)
    assert snap["lambda_beta"] is not None
    assert snap["lambda_arith"] is not None
    assert snap["lambda_if"] is not None
    # `terms` exposes the real AST/rule-term list, one dict per EvalTerm.
    assert "terms" in snap
    terms = snap["terms"]
    assert isinstance(terms, list)
    assert len(terms) == snap["term_count"]
    for t in terms:
        assert set(t) == {"rule_id", "site", "arity"}
        assert isinstance(t["rule_id"], str) and t["rule_id"]
        assert isinstance(t["site"], int) and 0 <= t["site"] < snap["n_sites"]
        assert isinstance(t["arity"], int) and t["arity"] == 2
