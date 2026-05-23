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
from ..multifield import MultiFieldConfig, MultiFieldNetwork
from ..network import NetworkConfig, QFTPCNNetwork
from ..qft.hamiltonian import FieldSpecies
from ..qft.mera import MERA
from ..qft.qpcn import QPCN, QPCNConfig
from . import snapshots
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
        steps = min(int(d.get("steps", _DEFAULT_STEPS)), _MAX_STEPS)
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
# grid the caller requests. _MAX_STEPS bounds run length so a single request
# cannot exhaust server resources.
_MAX_GRID = 16
_MAX_STEPS = 500
_QPCN_SITES = 4
_QPCN_CHI = 8
_MERA_LEAVES = 4
_LOGIC_SITES = 6
_LOGIC_CHI = 8
_LOGIC_DEFAULT_EXPR = "2 + 3"
_VQC_QUBITS = 3
_VQC_LAYERS = 2


def _build_network(spec: RunSpec) -> QFTPCNNetwork:
    """Build a two-layer `QFTPCNNetwork`.

    Honours `spec.params["manifold"]["source"]` in {"flat", "hot-spot",
    "two-source"} — the source pattern is *applied* in `run_simulation`
    via the `observation` array. This builder only constructs the net.
    """
    n = min(spec.grid, _MAX_GRID)
    rng = np.random.default_rng(0 if spec.seed is None else spec.seed)
    cfg = NetworkConfig(layers=[LayerConfig(channels=1),
                                LayerConfig(channels=1)])
    return QFTPCNNetwork(n, n, cfg, rng=rng)


def _build_qpcn(spec: RunSpec) -> QPCN:
    """Build a single-species `QPCN` (MPS substrate).

    Honours `spec.params["qpcn"]` keys: `mass`, `kinetic`, `chi_max`,
    `target_n0` (the *observation* target is consumed in `run_simulation`,
    not here), `species` (list of species names — defaults to ["A"]).
    """
    rng = np.random.default_rng(0 if spec.seed is None else spec.seed)
    p = dict(spec.params.get("qpcn") or {})
    mass = float(p.get("mass", 1.0))
    kinetic = float(p.get("kinetic", 0.5))
    chi_max = min(int(p.get("chi_max", _QPCN_CHI)), _QPCN_CHI * 2)
    names = list(p.get("species") or ["A"])
    species = [FieldSpecies(name=n, cutoff=2, bare_mass=mass, kinetic=kinetic)
               for n in names]
    cfg = QPCNConfig(
        species=species,
        N_sites=_QPCN_SITES,
        chi_max=chi_max,
        learnable_params=[f"{names[0]}.mass"],
        observable_map=[(0, names[0], "n")],
    )
    return QPCN(cfg, rng=rng)


def _build_mera(spec: RunSpec) -> MERA:
    """Build a vacuum `MERA`. Honours `spec.params["mera"]`."""
    p = dict(spec.params.get("mera") or {})
    leaves = int(p.get("leaves", _MERA_LEAVES))
    chi = int(p.get("chi_layer", 4))
    if leaves not in (2, 4, 8):
        leaves = _MERA_LEAVES
    return MERA.vacuum(leaves, d_local=2, chi_layer=chi)


def _build_vqc(spec: RunSpec):
    """Build a real `QuantumGenerativeMap`. Lazy-imports qiskit so the viz
    server doesn't pull it in unless a `vqc` run is requested."""
    try:
        from ..quantum import QuantumGenerativeMap
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "the 'vqc' layer requires qiskit; install the 'viz' extra "
            "(pip install -e '.[viz]') to enable it"
        ) from exc
    p = dict(spec.params.get("vqc") or {})
    n_qubits = int(p.get("n_qubits", _VQC_QUBITS))
    n_layers = int(p.get("n_layers", _VQC_LAYERS))
    rng = np.random.default_rng(0 if spec.seed is None else spec.seed)
    return QuantumGenerativeMap(n_qubits=n_qubits, n_layers=n_layers, rng=rng)


def _build_logic(spec: RunSpec):
    """Build a real `EvalHamiltonian` + a logic-encoded MPS for relaxation.

    Honours `spec.params["logic"]` keys: `N` (sites), `chi_max`, `expr`
    (lambda-source program string; defaults to a small arithmetic example).
    Returns ``(H, state, chi_max)``.
    """
    from ..logic.ast import parse
    from ..logic.encoder import encode
    from ..logic.evaluation_hamiltonian import EvalHamiltonian
    p = dict(spec.params.get("logic") or {})
    N = int(p.get("N", _LOGIC_SITES))
    chi_max = int(p.get("chi_max", _LOGIC_CHI))
    expr = str(p.get("expr", _LOGIC_DEFAULT_EXPR))
    state, _meta = encode(parse(expr), N=N, chi_max=chi_max)
    state.normalize()
    H = EvalHamiltonian(N=N)
    return H, state, chi_max


def _build_multifield(spec: RunSpec) -> MultiFieldNetwork:
    """Build a two-field `MultiFieldNetwork` on a shared manifold.

    Honours `spec.params["multifield"]`: `learn_coupling` (bool),
    `initial_coupling` (float in [-1, 1]).
    """
    n = min(spec.grid, _MAX_GRID)
    rng = np.random.default_rng(0 if spec.seed is None else spec.seed)
    p = dict(spec.params.get("multifield") or {})
    learn = bool(p.get("learn_coupling", True))
    g0 = float(p.get("initial_coupling", 0.0))
    cfg = MultiFieldConfig(
        field_names=["a", "b"],
        layer_configs={"a": LayerConfig(channels=1),
                       "b": LayerConfig(channels=1)},
        coupling={("a", "b"): g0},
        learn_coupling=learn,
    )
    return MultiFieldNetwork(n, n, cfg, rng=rng)


# ---- simulation driver -------------------------------------------------------

def run_simulation(spec: RunSpec) -> Iterator[Frame]:
    """Run the requested substrate(s) and yield one `Frame` per step.

    Layers are grouped by substrate: `manifold` runs on a `QFTPCNNetwork`;
    `multifield` runs on its own real `MultiFieldNetwork`; `mps`/`qpcn`/
    `hamiltonian` share a `QPCN`; `mera` runs standalone. The `vqc` and
    `logic` layers are accepted in `spec.layers` (so subscribers receive
    their key in the WS handshake) but are *not* simulated here — they
    remain fixture-only pending EXTENSIONS.md follow-up. Every active
    substrate contributes its snapshot to a single Frame per step via the
    generic `Recorder.capture`, so one stream can carry several layers at
    once.
    """
    requested = set(spec.layers)
    recorder = Recorder()

    want_network = "manifold" in requested
    want_multifield = "multifield" in requested
    want_qpcn = bool(requested & {"mps", "qpcn", "hamiltonian"})
    want_mera = "mera" in requested
    want_logic = "logic" in requested
    want_vqc = "vqc" in requested

    # Fall back to the manifold substrate if nothing recognised was asked for,
    # so a stream always yields content rather than silently producing zero
    # frames.
    if not (want_network or want_multifield or want_qpcn or want_mera
            or want_logic or want_vqc):
        want_network = True

    net = _build_network(spec) if want_network else None
    multifield = _build_multifield(spec) if want_multifield else None
    qpcn = _build_qpcn(spec) if want_qpcn else None
    mera = _build_mera(spec) if want_mera else None
    logic_H = logic_state = None
    logic_chi = _LOGIC_CHI
    if want_logic:
        logic_H, logic_state, logic_chi = _build_logic(spec)
    vqc = _build_vqc(spec) if want_vqc else None
    vqc_x = vqc_target = None
    vqc_lr = 0.2
    if vqc is not None:
        _vqc_rng = np.random.default_rng(0 if spec.seed is None else spec.seed)
        vqc_x = _vqc_rng.standard_normal(vqc.n_qubits)
        vqc_target = np.full(vqc.n_qubits, 0.5)

    observation = None
    if net is not None:
        c = net.layers[0].phi.channels
        nx, ny = net.manifold.nx, net.manifold.ny
        observation = np.zeros((c, nx, ny))
        source = ((spec.params.get("manifold") or {}).get("source")
                  or "flat")
        if source != "flat":
            xs = np.linspace(-1.0, 1.0, nx)[:, None]
            ys = np.linspace(-1.0, 1.0, ny)[None, :]
            if source == "hot-spot":
                observation[0] = np.exp(-((xs ** 2 + ys ** 2) / 0.1))
            elif source == "two-source":
                observation[0] = (
                    np.exp(-(((xs - 0.4) ** 2 + ys ** 2) / 0.08))
                    + np.exp(-(((xs + 0.4) ** 2 + ys ** 2) / 0.08))
                )

    mf_observations = None
    if multifield is not None:
        # Each field's bottom layer sees a (C, Nx, Ny) observation; a shared
        # zero field is sufficient to drive the coupled dynamics for viz.
        mf_observations = {
            name: np.zeros(
                (layer.phi.channels,
                 multifield.manifold.nx, multifield.manifold.ny))
            for name, layer in multifield.fields.items()
        }

    qpcn_targets = {}
    if qpcn is not None:
        tgt = (spec.params.get("qpcn") or {}).get("target_n0", 0.25)
        if tgt is not None:
            qpcn_targets = {(0, qpcn.cfg.species[0].name, "n"): float(tgt)}

    # Recorder retains every captured Frame in `recorder.frames` for callers
    # that consume the generator's side effects. Note `/export` does NOT reuse
    # this sequence: it re-derives frames by re-running this (deterministic,
    # seeded) simulation from scratch -- an intentional trade-off, since runs
    # are tiny and re-running avoids threading the recorded state across the
    # request boundary.
    for _ in range(spec.steps):
        snaps: dict[str, dict] = {}

        if net is not None:
            net.step(observation, learn=True)
            snaps["manifold"] = snapshots.snapshot_network(net)

        if multifield is not None:
            multifield.step(mf_observations, learn=True)
            snaps["multifield"] = snapshots.snapshot_multifield(multifield)

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

        if vqc is not None:
            pred = vqc.forward(vqc_x)
            upstream = pred - vqc_target
            grad = vqc.parameter_shift_grad(vqc_x, upstream)
            vqc.theta = vqc.theta - vqc_lr * np.clip(grad, -1.0, 1.0)
            snaps["vqc"] = snapshots.snapshot_vqc(vqc)

        if logic_H is not None and logic_state is not None:
            from ..logic.factored_evolution import factored_trotter_step
            factored_trotter_step(logic_state, logic_H, dt=0.1,
                                  imaginary=True, chi_max=logic_chi)
            logic_state.normalize()
            snaps["logic"] = snapshots.snapshot_logic(logic_H, logic_state)

        yield recorder.capture(**snaps)
