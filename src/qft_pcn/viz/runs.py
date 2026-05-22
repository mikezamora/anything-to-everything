"""Run registry and simulation drivers for the viz server.

A `RunSpec` captures what to simulate (which substrate layers, how many steps,
grid size, seed). `run_simulation` builds the requested substrate(s), steps
them, and yields one `Frame` per step via the Task-1 `Recorder`. `RunRegistry`
maps an opaque run id to its spec so the WebSocket endpoint can replay it.

Every run is kept deliberately small and fast — substrates are constructed
with tiny dimensions so a full run streams in well under a second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from ..layer import LayerConfig
from ..network import NetworkConfig, QFTPCNNetwork
from ..qft.hamiltonian import FieldSpecies
from ..qft.mera import MERA
from ..qft.qpcn import QPCN, QPCNConfig
from .recorder import Recorder
from .schema import Frame

# Defaults applied by RunSpec.from_dict when a key is absent.
_DEFAULT_STEPS = 20
_DEFAULT_GRID = 12


@dataclass
class RunSpec:
    """A request to simulate one or more substrate layers."""

    layers: list[str]
    steps: int = _DEFAULT_STEPS
    grid: int = _DEFAULT_GRID
    seed: int | None = None
    params: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "RunSpec":
        """Build a `RunSpec` from POSTed JSON, applying defaults.

        Unknown keys are ignored. `layers` defaults to ``["manifold"]`` so a
        bare ``{}`` still yields a valid (if minimal) run.
        """
        d = d or {}
        layers = list(d.get("layers") or ["manifold"])
        steps = int(d.get("steps", _DEFAULT_STEPS))
        grid = int(d.get("grid", _DEFAULT_GRID))
        seed = d.get("seed")
        seed = int(seed) if seed is not None else None
        params = dict(d.get("params") or {})
        return cls(layers=layers, steps=steps, grid=grid, seed=seed,
                   params=params)


class RunRegistry:
    """In-memory map from run id (uuid4 hex) to its `RunSpec`."""

    def __init__(self) -> None:
        self._runs: dict[str, RunSpec] = {}

    def add(self, run_id: str, spec: RunSpec) -> None:
        """Register `spec` under `run_id`."""
        self._runs[run_id] = spec

    def get(self, run_id: str) -> RunSpec | None:
        """Return the spec for `run_id`, or None if it was never registered."""
        return self._runs.get(run_id)


# ---- substrate builders ------------------------------------------------------

# Substrate dimensions are clamped to keep every run cheap regardless of the
# grid the caller requests.
_MAX_GRID = 16
_QPCN_SITES = 4
_QPCN_CHI = 8
_MERA_LEAVES = 4


def _build_network(spec: RunSpec) -> QFTPCNNetwork:
    """Build a small two-layer `QFTPCNNetwork` for the manifold/multifield."""
    n = min(spec.grid, _MAX_GRID)
    rng = np.random.default_rng(0 if spec.seed is None else spec.seed)
    cfg = NetworkConfig(layers=[LayerConfig(channels=1),
                                LayerConfig(channels=1)])
    return QFTPCNNetwork(n, n, cfg, rng=rng)


def _build_qpcn(spec: RunSpec) -> QPCN:
    """Build a small single-species `QPCN` (MPS substrate)."""
    rng = np.random.default_rng(0 if spec.seed is None else spec.seed)
    species = [FieldSpecies(name="A", cutoff=2, bare_mass=1.0, kinetic=0.5)]
    cfg = QPCNConfig(
        species=species,
        N_sites=_QPCN_SITES,
        chi_max=_QPCN_CHI,
        learnable_params=["A.mass"],
        observable_map=[(0, "A", "n")],
    )
    return QPCN(cfg, rng=rng)


def _build_mera() -> MERA:
    """Build a small vacuum `MERA` on a power-of-two leaf count."""
    return MERA.vacuum(_MERA_LEAVES, d_local=2, chi_layer=4)


# ---- simulation driver -------------------------------------------------------

def run_simulation(spec: RunSpec) -> Iterator[Frame]:
    """Run the requested substrate(s) and yield one `Frame` per step.

    Layers are grouped by substrate: `manifold`/`multifield` share a
    `QFTPCNNetwork`; `mps`/`qpcn`/`hamiltonian` share a `QPCN`; `mera` runs
    standalone. Every active substrate contributes its snapshot to a single
    Frame per step via the generic `Recorder.capture`, so one stream can carry
    several layers at once. `vqc`/`logic` are accepted but not simulated here.
    """
    requested = set(spec.layers)
    recorder = Recorder()

    want_network = bool(requested & {"manifold", "multifield"})
    want_qpcn = bool(requested & {"mps", "qpcn", "hamiltonian"})
    want_mera = "mera" in requested

    # Fall back to the manifold substrate if nothing recognised was asked for,
    # so a stream always yields content rather than silently producing zero
    # frames.
    if not (want_network or want_qpcn or want_mera):
        want_network = True

    net = _build_network(spec) if want_network else None
    qpcn = _build_qpcn(spec) if want_qpcn else None
    mera = _build_mera() if want_mera else None

    observation = None
    if net is not None:
        c = net.layers[0].phi.channels
        observation = np.zeros((c, net.manifold.nx, net.manifold.ny))

    qpcn_targets = {(0, "A", "n"): 0.25} if qpcn is not None else {}

    from . import snapshots

    for _ in range(spec.steps):
        snaps: dict[str, dict] = {}

        if net is not None:
            net.step(observation, learn=True)
            net_snap = snapshots.snapshot_network(net)
            if "manifold" in requested:
                snaps["manifold"] = net_snap
            if "multifield" in requested:
                # No MultiFieldNetwork here; expose the multi-layer field
                # data from the same network snapshot under the multifield
                # key so the layer still receives content.
                snaps["multifield"] = net_snap

        if qpcn is not None:
            qpcn.observe(qpcn_targets, learn=True)
            q_snap = snapshots.snapshot_qpcn(qpcn)
            if "qpcn" in requested:
                snaps["qpcn"] = q_snap
            if "mps" in requested:
                snaps["mps"] = snapshots.snapshot_mps(qpcn.state)
            if "hamiltonian" in requested:
                snaps["hamiltonian"] = snapshots.snapshot_hamiltonian(qpcn.H)

        if mera is not None:
            # MERA has no time dynamics here; re-snapshot the static tree so
            # the layer still receives a Frame on every step.
            snaps["mera"] = snapshots.snapshot_mera(mera)

        yield recorder.capture(**snaps)
