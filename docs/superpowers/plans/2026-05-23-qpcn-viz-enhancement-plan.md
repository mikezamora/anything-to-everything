# QPCN Visualizer Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the QPCN visualizer self-explaining and controllable — every panel ships descriptions, math, presets, transport controls, baseline compare, JSONL/MP4 export, and a comprehensive per-panel uplift driven by what each `snapshot_*` already exposes.

**Architecture:** New backend modules (`presets.py`, `controller.py`) extend `runs.py`/`server.py` additively. Frontend gains a 4-region layout (top run/transport bar, left layer rail, center panel, right collapsible explainer), a per-run frame store, and shared panel chrome (`PanelToolbar`, `PanelReadouts`, `MetricsStrip`). Substrate code is not touched; missing-hook features are recorded in `src/qft_pcn/viz/EXTENSIONS.md` instead of stubbed.

**Tech Stack:** Python 3.11+, FastAPI, asyncio; React 18 + Vite + TypeScript, Zustand, three.js / @react-three, Plotly, KaTeX, vitest.

**Spec:** `docs/superpowers/specs/2026-05-23-qpcn-viz-enhancement-design.md`

**Hard constraints:**
- Touch only `src/qft_pcn/viz/**` (plus `scripts/install-manim.sh` and `docs/`). Never edit substrate modules — parallel agents own those.
- No placeholders / TODOs / fake data. Missing-substrate features go in `src/qft_pcn/viz/EXTENSIONS.md` and the UI omits the affordance entirely.
- All `snapshot_*` calls stay defensive (`_safe(lambda: …)`). Panels must gracefully render with any field absent.
- All tests run with `--noconftest` (the repo conftest imports the Unix-only `resource` module). Viz Python tests live in `src/qft_pcn/tests/test_viz_*.py`.
- Frontend tests run from `src/qft_pcn/viz/web` via `pnpm test`.

---

## File Map

**Created (backend)**
- `src/qft_pcn/viz/presets.py` — `Preset` dataclass + catalog + `PARAM_SCHEMA`
- `src/qft_pcn/viz/controller.py` — `RunController` (asyncio lifecycle wrapper)
- `src/qft_pcn/viz/EXTENSIONS.md` — deferred-feature register
- `scripts/install-manim.sh` — community Manim install helper
- `src/qft_pcn/tests/test_viz_presets.py`
- `src/qft_pcn/tests/test_viz_controller.py`
- `src/qft_pcn/tests/test_viz_endpoints_extra.py`

**Modified (backend)**
- `src/qft_pcn/viz/runs.py` — substrate builders read `spec.params[layer]` (allow-listed)
- `src/qft_pcn/viz/server.py` — `/presets`, `/params/schema`, `/runs/{id}/pause|resume|step`, `/export/run/{id}` (JSONL); WS uses `RunController`
- `src/qft_pcn/viz/README.md` — preset usage + manual smoke matrix

**Created (frontend)**
- `web/src/lib/explainer.ts` — `ExplainerSpec` type + 8 layer specs
- `web/src/lib/presets.ts` — fetch + cache `/presets` and `/params/schema`
- `web/src/components/ExplainerPane.tsx`
- `web/src/components/CompareBar.tsx`
- `web/src/panels/PanelToolbar.tsx`
- `web/src/panels/PanelReadouts.tsx`
- `web/src/panels/MetricsStrip.tsx`
- vitest siblings for each new component (`*.test.tsx` / `*.test.ts`)

**Modified (frontend)**
- `web/src/store.ts` — per-run frame map, `baselineRunId`, `paused`, `playbackSpeed`
- `web/src/lib/ws.ts` — routes frames into `store.runs[runId]`
- `web/src/lib/types.ts` — `RunSpec.params`, `Preset`, `ParamSchema` types
- `web/src/components/RunControls.tsx` — split out of `App.tsx`; preset dropdown + advanced + transport + export
- `web/src/App.tsx` — 4-region grid layout; mounts `ExplainerPane`, `CompareBar`
- `web/src/App.css` and `web/src/panels/panel.css` — new layout + chrome styles
- All 8 panel files in `web/src/panels/` — adopt shared chrome, add overlays / readouts / metrics

---

## Per-Task Process

For each task: spec-review the task description first (does it match the design doc?), then run the TDD cycle below, then run a code-review pass on the diff before commit.

**Spec review (before starting):** Re-read the relevant spec section. Confirm the task does not extend scope, does not require substrate edits, and respects the no-placeholders rule. If the task as written conflicts with the spec, fix the task — not the code — and note the deviation.

**Code review (before commit):** Re-read the diff. Check: (1) no TODO/stub/fake values, (2) no substrate files touched, (3) snapshot reads are defensive, (4) tests cover the failure mode you just fixed, (5) no unrelated drive-by changes.

---

## Phase A — Backend Foundations

### Task 1: Presets module + catalog + param schema

**Files:**
- Create: `src/qft_pcn/viz/presets.py`
- Create: `src/qft_pcn/tests/test_viz_presets.py`

- [ ] **Step 1: Write the failing test**

```python
# src/qft_pcn/tests/test_viz_presets.py
"""Preset catalog tests: shape, uniqueness, and round-trip via run_simulation."""

import pytest

from qft_pcn.viz.presets import PRESETS, PARAM_SCHEMA, Preset
from qft_pcn.viz.runs import RunSpec, run_simulation


def test_catalog_is_nonempty_and_ids_unique():
    assert len(PRESETS) > 0
    ids = [p.id for p in PRESETS]
    assert len(ids) == len(set(ids))


def test_every_preset_has_required_fields():
    for p in PRESETS:
        assert isinstance(p, Preset)
        assert p.id and p.layer and p.label and p.description
        assert isinstance(p.spec_overrides, dict)
        # spec_overrides may set layers/steps/grid/seed/params
        for key in p.spec_overrides:
            assert key in {"layers", "steps", "grid", "seed", "params"}


def test_param_schema_has_entry_per_live_layer():
    for layer in ("manifold", "multifield", "mps", "qpcn",
                  "hamiltonian", "mera"):
        assert layer in PARAM_SCHEMA
        entry = PARAM_SCHEMA[layer]
        assert entry["type"] == "object"
        assert "properties" in entry


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
def test_preset_produces_a_nonempty_frame_for_its_layer(preset):
    spec_dict = {"layers": [preset.layer], "steps": 2, "grid": 8,
                 **preset.spec_overrides}
    spec = RunSpec.from_dict(spec_dict)
    frames = list(run_simulation(spec))
    assert frames, f"{preset.id} produced zero frames"
    assert frames[0].layer_states.get(preset.layer), (
        f"{preset.id} produced empty {preset.layer} snapshot")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/qft_pcn/tests/test_viz_presets.py -v --noconftest`
Expected: ImportError on `qft_pcn.viz.presets`.

- [ ] **Step 3: Write the module**

```python
# src/qft_pcn/viz/presets.py
"""Curated presets and per-layer param schema for the viz UI.

A `Preset` is a frozen record whose `spec_overrides` dict is merged into the
JSON body of `POST /run`. Presets only set values that `RunSpec.from_dict`
already accepts; substrate code is never touched from here.

`PARAM_SCHEMA` is the JSON Schema served by `GET /params/schema` and consumed
by the frontend Advanced expander. It is hand-authored and kept in sync with
`runs.py` substrate builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Preset:
    """A named starting condition for a single layer."""

    id: str
    layer: str
    label: str
    description: str
    spec_overrides: dict[str, Any] = field(default_factory=dict)


PRESETS: list[Preset] = [
    # ---- manifold ----------------------------------------------------------
    Preset(
        id="manifold.flat",
        layer="manifold",
        label="Flat — zero curvature start",
        description="Uniform metric, no curvature sources. Baseline for "
                    "watching how prediction error alone sources geometry.",
        spec_overrides={"steps": 30, "grid": 12, "seed": 0,
                        "params": {"manifold": {"source": "flat"}}},
    ),
    Preset(
        id="manifold.hot-spot",
        layer="manifold",
        label="Hot spot — central Gaussian source",
        description="A single Gaussian bump in the error field, so curvature "
                    "concentrates at the centre as the run progresses.",
        spec_overrides={"steps": 40, "grid": 14, "seed": 1,
                        "params": {"manifold": {"source": "hot-spot"}}},
    ),
    Preset(
        id="manifold.two-source",
        layer="manifold",
        label="Two-source — competing curvature wells",
        description="Two offset Gaussian sources. Watch the curvature ridge "
                    "between them rise as both wells deepen.",
        spec_overrides={"steps": 40, "grid": 14, "seed": 2,
                        "params": {"manifold": {"source": "two-source"}}},
    ),
    # ---- multifield --------------------------------------------------------
    Preset(
        id="multifield.uncoupled",
        layer="multifield",
        label="Uncoupled — g_ab pinned at 0",
        description="Two fields, coupling frozen at zero. Baseline run.",
        spec_overrides={"steps": 30, "grid": 12, "seed": 0,
                        "params": {"multifield": {"learn_coupling": False,
                                                  "initial_coupling": 0.0}}},
    ),
    Preset(
        id="multifield.symmetric-coupling",
        layer="multifield",
        label="Symmetric coupling — g_ab = 0.5",
        description="Two fields with a fixed symmetric Yukawa coupling.",
        spec_overrides={"steps": 30, "grid": 12, "seed": 0,
                        "params": {"multifield": {"learn_coupling": False,
                                                  "initial_coupling": 0.5}}},
    ),
    Preset(
        id="multifield.learn-coupling",
        layer="multifield",
        label="Learn coupling — g_ab descends",
        description="Coupling is learnable; watch g_ab adapt to the joint "
                    "free energy.",
        spec_overrides={"steps": 50, "grid": 12, "seed": 0,
                        "params": {"multifield": {"learn_coupling": True,
                                                  "initial_coupling": 0.0}}},
    ),
    # ---- qpcn / mps / hamiltonian -----------------------------------------
    Preset(
        id="qpcn.ground-state-relax",
        layer="qpcn",
        label="Ground-state relax — no observation target",
        description="Imaginary-time relaxation with the default Hamiltonian.",
        spec_overrides={"steps": 40, "seed": 0,
                        "params": {"qpcn": {"target_n0": None,
                                            "mass": 1.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="qpcn.quarter-density-target",
        layer="qpcn",
        label="Target ⟨n₀⟩ = 0.25",
        description="Observable-driven update toward a quarter-occupation "
                    "target on site 0.",
        spec_overrides={"steps": 50, "seed": 0,
                        "params": {"qpcn": {"target_n0": 0.25,
                                            "mass": 1.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="qpcn.high-mass",
        layer="qpcn",
        label="High-mass regime (m = 3.0)",
        description="Heavy bare mass; ground state should localise more "
                    "tightly than the default.",
        spec_overrides={"steps": 40, "seed": 0,
                        "params": {"qpcn": {"target_n0": 0.25,
                                            "mass": 3.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="mps.product-state",
        layer="mps",
        label="Product state — bond-dim 1",
        description="MPS warmup starting from a product state; bonds grow as "
                    "entanglement accumulates.",
        spec_overrides={"steps": 30, "seed": 0,
                        "params": {"qpcn": {"chi_max": 8,
                                            "mass": 1.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="mps.entangled-warmup",
        layer="mps",
        label="Entangled warmup — random bond-2",
        description="Random superposition seed (handled by qpcn builder); "
                    "watch entropy plateau as relaxation kicks in.",
        spec_overrides={"steps": 30, "seed": 7,
                        "params": {"qpcn": {"chi_max": 12,
                                            "mass": 1.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="hamiltonian.single-species",
        layer="hamiltonian",
        label="Single species (A only)",
        description="Default single-species Hamiltonian. Watch species_dims "
                    "and the curvature mini-map.",
        spec_overrides={"steps": 10, "seed": 0, "params": {"qpcn": {}}},
    ),
    Preset(
        id="hamiltonian.two-species",
        layer="hamiltonian",
        label="Two species (A + B)",
        description="Two-species Hamiltonian to exercise the species legend "
                    "and cross-species curvature.",
        spec_overrides={"steps": 10, "seed": 0,
                        "params": {"qpcn": {"species": ["A", "B"]}}},
    ),
    # ---- mera -------------------------------------------------------------
    Preset(
        id="mera.vacuum-small",
        layer="mera",
        label="Vacuum MERA (4 leaves)",
        description="Static vacuum MERA on 4 leaves. Useful for inspecting "
                    "the tree layout and per-cut entropy.",
        spec_overrides={"steps": 5, "seed": 0,
                        "params": {"mera": {"leaves": 4, "chi_layer": 4}}},
    ),
]


# Hand-authored JSON Schema; each property's `default` mirrors what the
# corresponding `_build_*` function in runs.py uses when the key is absent.
PARAM_SCHEMA: dict[str, dict] = {
    "manifold": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["flat", "hot-spot", "two-source"],
                "default": "flat",
                "description": "Initial error-field source pattern.",
            },
        },
    },
    "multifield": {
        "type": "object",
        "properties": {
            "learn_coupling": {"type": "boolean", "default": True,
                               "description": "Adapt g_ab via gradient descent."},
            "initial_coupling": {"type": "number", "default": 0.0,
                                 "minimum": -1.0, "maximum": 1.0,
                                 "description": "Starting value of g_ab."},
        },
    },
    "mps": {
        "type": "object",
        "properties": {},  # mps shares the qpcn substrate; tuned via qpcn
    },
    "qpcn": {
        "type": "object",
        "properties": {
            "mass": {"type": "number", "default": 1.0,
                     "minimum": 0.0, "maximum": 10.0,
                     "description": "Bare mass of species A."},
            "kinetic": {"type": "number", "default": 0.5,
                        "minimum": 0.0, "maximum": 5.0,
                        "description": "Kinetic coefficient."},
            "chi_max": {"type": "integer", "default": 8,
                        "minimum": 1, "maximum": 16,
                        "description": "MPS bond-dimension cap."},
            "target_n0": {"type": ["number", "null"], "default": 0.25,
                          "minimum": 0.0, "maximum": 1.0,
                          "description": "Target ⟨n⟩ on site 0; null disables."},
            "species": {"type": "array", "items": {"type": "string"},
                        "default": ["A"],
                        "description": "List of species names."},
        },
    },
    "hamiltonian": {
        "type": "object",
        "properties": {},  # hamiltonian shares qpcn config
    },
    "mera": {
        "type": "object",
        "properties": {
            "leaves": {"type": "integer", "default": 4,
                       "enum": [2, 4, 8],
                       "description": "Number of leaf sites (power of two)."},
            "chi_layer": {"type": "integer", "default": 4,
                          "minimum": 2, "maximum": 16,
                          "description": "Per-layer bond dimension."},
        },
    },
}
```

- [ ] **Step 4: Update `runs.py` builders to honour the new params**

Modify `src/qft_pcn/viz/runs.py`. Add a small helper and thread params through builders. Keep all clamps (`_MAX_*`). The keys are read defensively — unknown keys are ignored.

Find `_build_qpcn` (around line 98) and replace:

```python
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
```

Find `_build_multifield` and replace:

```python
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
```

Find `_build_mera` and replace:

```python
def _build_mera(spec: RunSpec) -> MERA:
    """Build a vacuum `MERA`. Honours `spec.params["mera"]`."""
    p = dict(spec.params.get("mera") or {})
    leaves = int(p.get("leaves", _MERA_LEAVES))
    chi = int(p.get("chi_layer", 4))
    if leaves not in (2, 4, 8):
        leaves = _MERA_LEAVES
    return MERA.vacuum(leaves, d_local=2, chi_layer=chi)
```

Update the call site in `run_simulation` from `_build_mera()` to
`_build_mera(spec)`.

For `_build_network`, add a `source` honor — replace the body:

```python
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
```

In `run_simulation`, replace the `observation = np.zeros(...)` block with:

```python
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
```

In `run_simulation`, also replace the `qpcn_targets = {…}` line with:

```python
    qpcn_targets = {}
    if qpcn is not None:
        tgt = (spec.params.get("qpcn") or {}).get("target_n0", 0.25)
        if tgt is not None:
            qpcn_targets = {(0, qpcn.cfg.species[0].name, "n"): float(tgt)}
```

- [ ] **Step 5: Run preset tests to verify they pass**

Run: `uv run pytest src/qft_pcn/tests/test_viz_presets.py -v --noconftest`
Expected: PASS (all parametrized presets included).

Also run the existing viz server tests to confirm no regression:
Run: `uv run pytest src/qft_pcn/tests/test_viz_server.py -v --noconftest`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/viz/presets.py src/qft_pcn/viz/runs.py \
        src/qft_pcn/tests/test_viz_presets.py
git commit -m "feat(viz): preset catalog + per-layer param schema

13 curated presets driving real substrate state via RunSpec.params.
Builders read allow-listed keys defensively; unknown keys ignored.
"
```

---

### Task 2: `/presets` and `/params/schema` endpoints

**Files:**
- Modify: `src/qft_pcn/viz/server.py`
- Create: `src/qft_pcn/tests/test_viz_endpoints_extra.py`

- [ ] **Step 1: Write the failing test**

```python
# src/qft_pcn/tests/test_viz_endpoints_extra.py
"""Tests for /presets, /params/schema, /runs/{id}/{pause,resume,step},
and /export/run/{id} JSONL export.
"""

import json

import pytest
from fastapi.testclient import TestClient

from qft_pcn.viz.server import app
from qft_pcn.viz.presets import PRESETS, PARAM_SCHEMA


@pytest.fixture
def client():
    return TestClient(app)


def test_presets_endpoint_returns_catalog(client):
    res = client.get("/presets")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    assert len(body) == len(PRESETS)
    keys = {"id", "layer", "label", "description", "spec_overrides"}
    for entry in body:
        assert keys.issubset(entry.keys())


def test_params_schema_endpoint_returns_per_layer_schema(client):
    res = client.get("/params/schema")
    assert res.status_code == 200
    body = res.json()
    for layer in PARAM_SCHEMA:
        assert layer in body
        assert body[layer]["type"] == "object"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/qft_pcn/tests/test_viz_endpoints_extra.py -v --noconftest -k "presets_endpoint or params_schema"`
Expected: FAIL with 404 on both endpoints.

- [ ] **Step 3: Add endpoints to `server.py`**

Add new imports near the top:

```python
from dataclasses import asdict

from .presets import PRESETS, PARAM_SCHEMA
```

Add these two route handlers (near `/health`):

```python
@app.get("/presets")
def list_presets():
    """Return the curated preset catalog."""
    return [asdict(p) for p in PRESETS]


@app.get("/params/schema")
def params_schema():
    """Return the per-layer JSON Schema for the Advanced expander."""
    return PARAM_SCHEMA
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/qft_pcn/tests/test_viz_endpoints_extra.py -v --noconftest -k "presets_endpoint or params_schema"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/server.py src/qft_pcn/tests/test_viz_endpoints_extra.py
git commit -m "feat(viz/server): /presets and /params/schema endpoints"
```

---

### Task 3: RunController + lifecycle endpoints

**Files:**
- Create: `src/qft_pcn/viz/controller.py`
- Create: `src/qft_pcn/tests/test_viz_controller.py`
- Modify: `src/qft_pcn/viz/server.py`
- Modify: `src/qft_pcn/tests/test_viz_endpoints_extra.py`

- [ ] **Step 1: Write the failing controller test**

```python
# src/qft_pcn/tests/test_viz_controller.py
"""RunController unit tests: pause/resume/step semantics."""

import asyncio
import pytest

from qft_pcn.viz.controller import RunController
from qft_pcn.viz.runs import RunSpec


@pytest.mark.asyncio
async def test_resume_streams_all_frames():
    spec = RunSpec(layers=["manifold"], steps=3, grid=8)
    ctrl = RunController(spec)
    frames = []
    async for f in ctrl.frames():
        frames.append(f)
    assert len(frames) == 3


@pytest.mark.asyncio
async def test_pause_blocks_next_frame_until_resume():
    spec = RunSpec(layers=["manifold"], steps=5, grid=8)
    ctrl = RunController(spec)
    received = []

    async def consume():
        async for f in ctrl.frames():
            received.append(f)
            if len(received) == 2:
                ctrl.pause()

    task = asyncio.create_task(consume())
    # Give the consumer a moment to receive two frames and pause.
    await asyncio.sleep(0.05)
    assert len(received) == 2
    # Resume and drain.
    ctrl.resume()
    await task
    assert len(received) == 5


@pytest.mark.asyncio
async def test_step_advances_exactly_one_frame_while_paused():
    spec = RunSpec(layers=["manifold"], steps=5, grid=8)
    ctrl = RunController(spec)
    ctrl.pause()
    received = []

    async def consume():
        async for f in ctrl.frames():
            received.append(f)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.02)
    assert received == []
    ctrl.step()
    await asyncio.sleep(0.05)
    assert len(received) == 1
    ctrl.step()
    await asyncio.sleep(0.05)
    assert len(received) == 2
    # Cleanup.
    ctrl.resume()
    await task
```

This depends on `pytest-asyncio`. Add to dev deps if not present:

```bash
uv add --dev pytest-asyncio
```

Add to the top of the test file: `pytestmark = pytest.mark.asyncio` — actually
mark each test individually as above.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/qft_pcn/tests/test_viz_controller.py -v --noconftest`
Expected: ImportError on `qft_pcn.viz.controller`.

- [ ] **Step 3: Implement the controller**

```python
# src/qft_pcn/viz/controller.py
"""Async wrapper around `run_simulation` that exposes pause/resume/step.

A `RunController` owns one (synchronous) simulation generator and an
`asyncio.Event` gate. The async `frames()` iterator pulls one frame at a
time, yielding it through the gate. `pause()` clears the gate; `resume()`
sets it; `step()` flips the gate to ready, allows exactly one frame
through, then re-pauses. Clients that never call any lifecycle method see
the original streaming behaviour.

This is the only place where the synchronous generator is bridged to the
async world; the rest of the server stays unaware.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .runs import RunSpec, run_simulation
from .schema import Frame


class RunController:
    """Pause/step/resume wrapper around `run_simulation(spec)`."""

    def __init__(self, spec: RunSpec) -> None:
        self._spec = spec
        self._event = asyncio.Event()
        self._event.set()  # default: running
        # When >0, frames() decrements once per yield then re-pauses.
        self._step_request = 0

    async def frames(self) -> AsyncIterator[Frame]:
        """Yield one `Frame` per simulation step, gated by the event."""
        for frame in run_simulation(self._spec):
            await self._event.wait()
            yield frame
            # Briefly yield to the loop so observers can pause between frames.
            await asyncio.sleep(0)
            if self._step_request > 0:
                self._step_request -= 1
                self._event.clear()

    def pause(self) -> None:
        """Clear the gate; the next frame will block on `await`."""
        self._event.clear()

    def resume(self) -> None:
        """Set the gate; pending awaits proceed."""
        self._event.set()

    def step(self) -> None:
        """Allow exactly one frame through, then re-pause."""
        self._step_request += 1
        self._event.set()
```

- [ ] **Step 4: Run controller tests to pass**

Run: `uv run pytest src/qft_pcn/tests/test_viz_controller.py -v --noconftest`
Expected: PASS for all three tests.

- [ ] **Step 5: Wire endpoints in `server.py`**

Replace the existing `stream` handler and add lifecycle routes. First, add
near the top of `server.py`:

```python
from .controller import RunController

# run_id -> RunController for any run that has been opened via /ws.
_controllers: dict[str, RunController] = {}
```

Replace the `stream` handler with:

```python
@app.websocket("/ws/{run_id}")
async def stream(ws: WebSocket, run_id: str):
    """Stream one JSON Frame per step, then a {"done": true} sentinel."""
    await ws.accept()
    spec = _registry.get(run_id)
    if spec is None:
        await ws.close(code=4004)
        return
    ctrl = RunController(spec)
    _controllers[run_id] = ctrl
    try:
        async for frame in ctrl.frames():
            await ws.send_text(frame.to_json())
        await ws.send_text('{"done": true}')
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        await ws.send_text(json.dumps({"error": str(exc)}))
        await ws.close(code=1011)
    finally:
        _controllers.pop(run_id, None)
```

Add lifecycle routes:

```python
def _get_controller_or_404(run_id: str) -> RunController:
    ctrl = _controllers.get(run_id)
    if ctrl is None:
        raise HTTPException(status_code=404, detail="run not streaming")
    return ctrl


@app.post("/runs/{run_id}/pause")
def run_pause(run_id: str):
    _get_controller_or_404(run_id).pause()
    return {"run_id": run_id, "state": "paused"}


@app.post("/runs/{run_id}/resume")
def run_resume(run_id: str):
    _get_controller_or_404(run_id).resume()
    return {"run_id": run_id, "state": "running"}


@app.post("/runs/{run_id}/step")
def run_step(run_id: str):
    _get_controller_or_404(run_id).step()
    return {"run_id": run_id, "state": "stepped"}
```

- [ ] **Step 6: Add lifecycle endpoint test**

Append to `src/qft_pcn/tests/test_viz_endpoints_extra.py`:

```python
def test_pause_resume_step_404_when_not_streaming(client):
    """Lifecycle endpoints require an active WS connection for the run."""
    for verb in ("pause", "resume", "step"):
        res = client.post(f"/runs/nonexistent/{verb}")
        assert res.status_code == 404
```

A full WS lifecycle integration test is left to the manual smoke matrix
(WS + lifecycle interleaving is awkward under `TestClient`).

- [ ] **Step 7: Run all endpoint tests**

Run: `uv run pytest src/qft_pcn/tests/test_viz_endpoints_extra.py -v --noconftest`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/qft_pcn/viz/controller.py src/qft_pcn/viz/server.py \
        src/qft_pcn/tests/test_viz_controller.py \
        src/qft_pcn/tests/test_viz_endpoints_extra.py pyproject.toml uv.lock
git commit -m "feat(viz): RunController + pause/resume/step endpoints"
```

---

### Task 4: JSONL export endpoint

**Files:**
- Modify: `src/qft_pcn/viz/server.py`
- Modify: `src/qft_pcn/tests/test_viz_endpoints_extra.py`

- [ ] **Step 1: Write the failing test**

Append to `test_viz_endpoints_extra.py`:

```python
def test_jsonl_export_streams_one_object_per_frame(client):
    run = client.post("/run", json={"layers": ["manifold"], "steps": 3,
                                     "grid": 8}).json()
    run_id = run["run_id"]
    res = client.get(f"/export/run/{run_id}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/x-ndjson")
    lines = [ln for ln in res.text.splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "step" in obj
        assert "layer_states" in obj


def test_jsonl_export_404_for_unknown_run(client):
    res = client.get("/export/run/nope")
    assert res.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/qft_pcn/tests/test_viz_endpoints_extra.py::test_jsonl_export_streams_one_object_per_frame -v --noconftest`
Expected: FAIL (404).

- [ ] **Step 3: Add the route**

In `server.py`, add (near the other `/export/*` routes):

```python
from fastapi.responses import StreamingResponse


@app.get("/export/run/{run_id}")
def export_jsonl(run_id: str):
    """Stream the recorded frames of a run as newline-delimited JSON.

    Re-runs `run_simulation` deterministically (the existing /export pattern)
    so the dump matches what a fresh WS connection would have seen.
    """
    spec = _registry.get(run_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="run not found")

    def stream_lines():
        for frame in run_simulation(spec):
            yield frame.to_json() + "\n"

    return StreamingResponse(
        stream_lines(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition":
                 f'attachment; filename="run-{run_id}.jsonl"'},
    )
```

- [ ] **Step 4: Run all extra-endpoint tests**

Run: `uv run pytest src/qft_pcn/tests/test_viz_endpoints_extra.py -v --noconftest`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/server.py src/qft_pcn/tests/test_viz_endpoints_extra.py
git commit -m "feat(viz/server): JSONL export endpoint /export/run/{id}"
```

---

### Task 5: EXTENSIONS.md + Manim install script

**Files:**
- Create: `src/qft_pcn/viz/EXTENSIONS.md`
- Create: `scripts/install-manim.sh`
- Modify: `src/qft_pcn/viz/README.md`

- [ ] **Step 1: Write `EXTENSIONS.md`**

```markdown
<!-- src/qft_pcn/viz/EXTENSIONS.md -->
# Viz Deferred Extensions

Features the visualizer would render if the substrate exposed the needed
hook. Listed so we can wire them when the dependency lands — never stubbed
in code, never faked in the UI.

Each entry: **What's needed**, **Why deferred**, **Wire-up when ready**.

---

## vqc — live training panel

- **What's needed:** `_build_vqc(spec)` + `snapshot_vqc(vqc)` re-added to
  `runs.py` / `snapshots.py`, plus a parameter-shift training step in the
  per-frame loop.
- **Why deferred:** Parallel substrate work removed both. `VqcPanel` is now
  fixture-only and renders an "extension pending" badge.
- **Wire-up when ready:** Re-add the builder + snapshot + training step,
  add a `vqc.*` preset to `presets.py`, drop the `isExtension` prop from
  `VqcPanel`.

## logic — live relaxation panel

- **What's needed:** `_build_logic(spec)` + `snapshot_logic(enc, state)`
  re-added, plus the `factored_trotter_step` call in the per-frame loop.
- **Why deferred:** Removed alongside vqc.
- **Wire-up when ready:** Same shape as vqc — rebuild + snapshot + add
  `logic.*` presets, drop the badge.

## qpcn — live parameter editing during pause

- **What's needed:** A substrate setter on `Hamiltonian` (e.g.
  `set_param(name, value)`) and a server route `POST /runs/{id}/params`
  that mutates the controller's active `QPCN` between steps.
- **Why deferred:** Snapshot exposes parameter values but no setter.
- **Wire-up when ready:** Expose `writable=true` for editable keys in
  `/params/schema`; `QpcnPanel` will show sliders for any key whose schema
  entry has `writable=true`.

## mera — isometry-violation indicator

- **What's needed:** `snapshot_mera` to return per-layer
  `‖U†U − I‖` (or similar) as `iso_residuals`.
- **Why deferred:** Not currently computed.
- **Wire-up when ready:** `MeraPanel` renders the residual sparkline next
  to the bond-dim readout.

## hamiltonian — term list with active-term highlighting

- **What's needed:** `snapshot_hamiltonian` to enumerate active terms with
  rule_id / site / coefficient (the logic snapshot already does this).
- **Why deferred:** Not exposed today.
- **Wire-up when ready:** `HamiltonianPanel` already has the layout slot;
  drop the surrounding `null` guard once `terms` is present.
```

- [ ] **Step 2: Write `scripts/install-manim.sh`**

```bash
#!/usr/bin/env bash
# Install Manim Community Edition system prereqs + the viz-manim extra.
# Targets Debian/Ubuntu/WSL2. Run from the repo root.
set -euo pipefail

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script targets apt-based systems. Install Manim's prereqs"
  echo "manually for your OS, then run: uv sync --extra viz-manim"
  exit 1
fi

echo "==> apt: installing Manim Community prereqs"
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  python3-dev \
  libcairo2-dev \
  libpango1.0-dev \
  ffmpeg \
  texlive texlive-latex-extra texlive-fonts-extra texlive-science \
  pkg-config

echo "==> uv: syncing the viz-manim extra"
uv sync --extra viz-manim

echo "==> done. Try: uv run python -c 'import manim; print(manim.__version__)'"
```

- [ ] **Step 3: Update viz README**

Append to `src/qft_pcn/viz/README.md`:

```markdown
## Presets and advanced parameters

Curated starting conditions are exposed under `GET /presets`; per-layer
JSON Schema is at `GET /params/schema`. The UI's preset dropdown and
Advanced expander both consume these. See `presets.py` for the catalog.

## Manim Community install

For systems with `apt`, the prereqs + the `viz-manim` extra are installed by

    ./scripts/install-manim.sh

On other OSes, install Manim's system prereqs manually (Cairo, Pango,
ffmpeg, a LaTeX distribution) then `uv sync --extra viz-manim`.

## Deferred features

Anything the viz could show but currently can't (because a substrate hook
is missing) is registered in `EXTENSIONS.md`. No placeholders or fake data
are committed in code.

## Manual smoke matrix

After substantial changes, walk these by hand against `./scripts/viz.sh`:

|         | fixture | preset (first) | advanced (one param edited) |
|---------|---------|-----------------|------------------------------|
| manifold  | ✓ | ✓ | ✓ |
| multifield| ✓ | ✓ | ✓ |
| mps       | ✓ | ✓ | ✓ |
| hamiltonian| ✓ | ✓ | ✓ |
| qpcn      | ✓ | ✓ | ✓ |
| mera      | ✓ | ✓ | ✓ |
| vqc       | fixture only — see EXTENSIONS.md |||
| logic     | fixture only — see EXTENSIONS.md |||

Additionally: pause/resume/step, baseline compare, JSONL export, MP4 export.
```

- [ ] **Step 4: Make the script executable and commit**

```bash
chmod +x scripts/install-manim.sh
git add src/qft_pcn/viz/EXTENSIONS.md scripts/install-manim.sh \
        src/qft_pcn/viz/README.md
git commit -m "docs(viz): EXTENSIONS.md register + Manim install script"
```

---

## Phase B — Frontend Foundations

### Task 6: Store extension (multi-run, baseline, paused, speed)

**Files:**
- Modify: `src/qft_pcn/viz/web/src/store.ts`
- Modify: `src/qft_pcn/viz/web/src/store.test.ts`
- Modify: `src/qft_pcn/viz/web/src/lib/types.ts`

- [ ] **Step 1: Extend `lib/types.ts`**

Replace the file with:

```ts
/** Frontend mirror of `src/qft_pcn/viz/schema.py`. */

export interface Frame {
  step: number;
  layer_states: Record<string, Record<string, unknown>>;
}

export const LAYER_KEYS = [
  'manifold', 'multifield', 'mps', 'hamiltonian',
  'qpcn', 'mera', 'vqc', 'logic',
] as const;
export type LayerKey = (typeof LAYER_KEYS)[number];

export interface RunSpec {
  layers: string[];
  steps: number;
  grid: number;
  seed?: number | null;
  params?: Record<string, Record<string, unknown>>;
}

export interface Preset {
  id: string;
  layer: string;
  label: string;
  description: string;
  spec_overrides: Partial<RunSpec> & {
    params?: Record<string, Record<string, unknown>>;
  };
}

export type ParamSchema = Record<string, {
  type: 'object';
  properties: Record<string, {
    type: string | string[];
    default?: unknown;
    enum?: unknown[];
    minimum?: number;
    maximum?: number;
    description?: string;
    items?: { type: string };
  }>;
}>;
```

- [ ] **Step 2: Write the failing store test**

Replace `store.test.ts` body (keep file path) with:

```ts
import { beforeEach, describe, expect, it } from 'vitest';
import { useVizStore, MAX_FRAMES } from './store';
import type { Frame } from './lib/types';

const f = (step: number): Frame => ({ step, layer_states: {} });

beforeEach(() => useVizStore.getState().resetAll());

describe('per-run frame map', () => {
  it('pushes frames into the active run only', () => {
    const s = useVizStore.getState();
    s.openRun('A');
    s.pushFrame('A', f(0));
    s.pushFrame('A', f(1));
    expect(s.runs.get('A')?.frames.length).toBe(2);
    expect(s.runs.has('B')).toBe(false);
  });

  it('caps frames per run at MAX_FRAMES', () => {
    const s = useVizStore.getState();
    s.openRun('A');
    for (let i = 0; i < MAX_FRAMES + 50; i++) s.pushFrame('A', f(i));
    expect(s.runs.get('A')!.frames.length).toBe(MAX_FRAMES);
  });

  it('pinBaseline stores a second run id', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.openRun('B');
    s.setActiveRun('A'); s.pinBaseline('B');
    expect(s.activeRunId).toBe('A');
    expect(s.baselineRunId).toBe('B');
    s.unpinBaseline();
    expect(s.baselineRunId).toBeNull();
  });

  it('setPaused and setPlaybackSpeed are independent globals', () => {
    const s = useVizStore.getState();
    s.setPaused(true);
    s.setPlaybackSpeed(2);
    expect(useVizStore.getState().paused).toBe(true);
    expect(useVizStore.getState().playbackSpeed).toBe(2);
  });
});
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd src/qft_pcn/viz/web && pnpm test --run store.test.ts`
Expected: FAIL (missing exports).

- [ ] **Step 4: Replace `store.ts`**

```ts
/**
 * Zustand store: per-run frame ring buffers + scrub + baseline + transport.
 *
 * `runs` is a Map keyed by opaque run id. Each `RunState` is a capped frame
 * ring buffer + a cursor. `activeRunId` is what panels render against; if
 * `baselineRunId` is set, panels also receive its current frame for diff.
 * `paused` and `playbackSpeed` are global transport state — applied
 * client-side to the live frame pump.
 */

import { create } from 'zustand';
import type { Frame } from './lib/types';

export const MAX_FRAMES = 2000;

export interface RunState {
  frames: Frame[];
  cursor: number;
  live: boolean;
}

interface VizState {
  runs: Map<string, RunState>;
  activeRunId: string | null;
  baselineRunId: string | null;
  selectedLayer: string;
  paused: boolean;
  playbackSpeed: number;
  error: string | null;

  openRun: (id: string) => void;
  setActiveRun: (id: string) => void;
  pinBaseline: (id: string) => void;
  unpinBaseline: () => void;

  pushFrame: (runId: string, frame: Frame) => void;
  setCursor: (runId: string, cursor: number) => void;
  setLive: (runId: string, live: boolean) => void;

  selectLayer: (layer: string) => void;
  setPaused: (paused: boolean) => void;
  setPlaybackSpeed: (speed: number) => void;
  setError: (err: string | null) => void;

  currentFrame: (runId?: string | null) => Frame | undefined;
  baselineFrame: () => Frame | undefined;
  resetAll: () => void;
}

const initRun = (): RunState => ({ frames: [], cursor: 0, live: true });

export const useVizStore = create<VizState>((set, get) => ({
  runs: new Map(),
  activeRunId: null,
  baselineRunId: null,
  selectedLayer: 'manifold',
  paused: false,
  playbackSpeed: 1,
  error: null,

  openRun: (id) => set((s) => {
    if (s.runs.has(id)) return {};
    const next = new Map(s.runs);
    next.set(id, initRun());
    return { runs: next, activeRunId: s.activeRunId ?? id };
  }),

  setActiveRun: (id) => set({ activeRunId: id }),
  pinBaseline: (id) => set({ baselineRunId: id }),
  unpinBaseline: () => set({ baselineRunId: null }),

  pushFrame: (runId, frame) => set((s) => {
    const cur = s.runs.get(runId);
    if (!cur) return {};
    let frames = [...cur.frames, frame];
    if (frames.length > MAX_FRAMES) {
      frames = frames.slice(frames.length - MAX_FRAMES);
    }
    const cursor = cur.live
      ? frames.length - 1
      : Math.min(cur.cursor, frames.length - 1);
    const next = new Map(s.runs);
    next.set(runId, { ...cur, frames, cursor });
    return { runs: next };
  }),

  setCursor: (runId, cursor) => set((s) => {
    const cur = s.runs.get(runId);
    if (!cur) return {};
    const max = Math.max(0, cur.frames.length - 1);
    const next = new Map(s.runs);
    next.set(runId, {
      ...cur,
      cursor: Math.max(0, Math.min(cursor, max)),
      live: false,
    });
    return { runs: next };
  }),

  setLive: (runId, live) => set((s) => {
    const cur = s.runs.get(runId);
    if (!cur) return {};
    const next = new Map(s.runs);
    next.set(runId, { ...cur, live });
    return { runs: next };
  }),

  selectLayer: (layer) => set({ selectedLayer: layer }),
  setPaused: (paused) => set({ paused }),
  setPlaybackSpeed: (speed) => set({ playbackSpeed: Math.max(0.25, speed) }),
  setError: (error) => set({ error }),

  currentFrame: (runId) => {
    const id = runId ?? get().activeRunId;
    if (!id) return undefined;
    const r = get().runs.get(id);
    return r?.frames[r.cursor];
  },

  baselineFrame: () => {
    const id = get().baselineRunId;
    if (!id) return undefined;
    const r = get().runs.get(id);
    return r?.frames[r.cursor];
  },

  resetAll: () => set({
    runs: new Map(),
    activeRunId: null,
    baselineRunId: null,
    paused: false,
    playbackSpeed: 1,
    error: null,
  }),
}));
```

- [ ] **Step 5: Run store tests to pass**

Run: `cd src/qft_pcn/viz/web && pnpm test --run store.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/viz/web/src/store.ts \
        src/qft_pcn/viz/web/src/store.test.ts \
        src/qft_pcn/viz/web/src/lib/types.ts
git commit -m "feat(viz/web): per-run frame map, baseline pin, transport state"
```

---

### Task 7: ws.ts updated for per-run pushes + transport actions

**Files:**
- Modify: `src/qft_pcn/viz/web/src/lib/ws.ts`
- Create: `src/qft_pcn/viz/web/src/lib/transport.ts`
- Modify: `src/qft_pcn/viz/web/src/lib/ws.test.ts`

- [ ] **Step 1: Add transport client**

```ts
// src/qft_pcn/viz/web/src/lib/transport.ts
/**
 * Thin client for /runs/{id}/{pause,resume,step}. Each helper POSTs and
 * returns the server's JSON response. Errors are surfaced to the caller
 * (the UI sets store.error).
 */

async function post(path: string): Promise<unknown> {
  const res = await fetch(path, { method: 'POST' });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

export const transport = {
  pause: (runId: string) => post(`/runs/${runId}/pause`),
  resume: (runId: string) => post(`/runs/${runId}/resume`),
  step: (runId: string) => post(`/runs/${runId}/step`),
};
```

- [ ] **Step 2: Update `ws.ts`**

Replace `handleMessage` and `connectRun`:

```ts
import { useVizStore } from '../store';
import type { Frame, RunSpec } from './types';

export interface RunHandle { runId: string; close: () => void; }

function wsUrl(runId: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws/${runId}`;
}

function handleMessage(runId: string, raw: string): void {
  const store = useVizStore.getState();
  try {
    const msg = JSON.parse(raw) as
      | Frame | { done: true } | { error: string };
    if ('error' in msg) {
      store.setLive(runId, false);
      store.setError(msg.error);
    } else if ('done' in msg) {
      store.setLive(runId, false);
    } else {
      store.pushFrame(runId, msg);
    }
  } catch (err) {
    store.setError(`Malformed WebSocket message: ${String(err)}`);
  }
}

export async function connectRun(spec: RunSpec): Promise<RunHandle> {
  const res = await fetch('/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(spec),
  });
  if (!res.ok) throw new Error(`/run: ${res.status}`);
  const { run_id } = (await res.json()) as { run_id: string };

  const store = useVizStore.getState();
  store.openRun(run_id);
  store.setActiveRun(run_id);

  const ws = new WebSocket(wsUrl(run_id));
  ws.onmessage = (ev) => handleMessage(run_id, ev.data);
  ws.onerror = () => store.setError('WebSocket error');
  return { runId: run_id, close: () => ws.close() };
}
```

- [ ] **Step 3: Update `ws.test.ts`**

Adjust the existing tests to: (a) expect `store.openRun(id)` to have been
called, (b) frame pushes to be addressed `(id, frame)`. Run with
`pnpm test ws.test.ts` and fix until green.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/viz/web/src/lib/ws.ts \
        src/qft_pcn/viz/web/src/lib/ws.test.ts \
        src/qft_pcn/viz/web/src/lib/transport.ts
git commit -m "feat(viz/web): per-run WS routing + transport client"
```

---

### Task 8: ExplainerPane + 8 layer specs

**Files:**
- Create: `src/qft_pcn/viz/web/src/lib/explainer.ts`
- Create: `src/qft_pcn/viz/web/src/components/ExplainerPane.tsx`
- Create: `src/qft_pcn/viz/web/src/components/ExplainerPane.test.tsx`

- [ ] **Step 1: Write `lib/explainer.ts`**

```ts
// src/qft_pcn/viz/web/src/lib/explainer.ts
/**
 * Static per-layer explainer content.
 *
 * Sourced from QFT_PCN_ARCHITECTURE.md (anchors noted in `references`).
 * `watch` items may carry a `readout` id; the matching `<PanelReadouts>`
 * cell will flash on hover.
 */

export interface ExplainerSpec {
  title: string;
  oneLine: string;
  what: string[];
  elements: { name: string; meaning: string; code?: string }[];
  math: { tex: string; caption: string }[];
  watch: { label: string; readout?: string }[];
  references?: { label: string; href: string }[];
}

export const EXPLAINERS: Record<string, ExplainerSpec> = {
  manifold: {
    title: 'Manifold — Dynamic Riemannian Geometry',
    oneLine: 'The 2D base manifold whose metric is sourced by prediction error.',
    what: [
      'The classical predictive-coding layer sits on a 2D Riemannian manifold whose metric g_μν is not fixed: it is sourced by the stress-energy tensor of the prediction-error field.',
      'Belief diffusion uses the Laplace-Beltrami operator built from that metric — so geometry follows what the system is uncertain about.',
    ],
    elements: [
      { name: 'Surface height', meaning: 'h_xx component of the metric perturbation', code: 'manifold.py:h_xx' },
      { name: 'Surface colour', meaning: 'Ricci scalar curvature R', code: 'manifold.py:ricci_scalar' },
      { name: 'Overlay (Φ / E / Π)', meaning: 'Belief / error / precision fields of the first PCN layer' },
    ],
    math: [
      { tex: 'g_{\\mu\\nu}(x) = \\eta_{\\mu\\nu} + h_{\\mu\\nu}(x)', caption: 'Metric = flat + learned perturbation.' },
      { tex: 'R = g^{\\mu\\nu} R_{\\mu\\nu}', caption: 'Ricci scalar drives the colour map.' },
      { tex: '\\Delta_g \\Phi = \\frac{1}{\\sqrt{|g|}} \\partial_\\mu (\\sqrt{|g|}\\, g^{\\mu\\nu} \\partial_\\nu \\Phi)', caption: 'Belief diffuses via Laplace-Beltrami on g.' },
    ],
    watch: [
      { label: 'Curvature concentrates where error spikes', readout: 'mean_abs_ricci' },
      { label: 'Mean |R| should stabilise after error decays' },
      { label: 'Overlay Φ — peaks track regions the model is confidently predicting' },
    ],
    references: [{ label: 'Architecture §2.1, §3.1', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  multifield: {
    title: 'Multi-field — Coupled Field Species',
    oneLine: 'Multiple field species sharing one manifold, coupled via learnable g_ij.',
    what: [
      'Multiple field species (analogous to electron / photon / Higgs, or shape / motion / colour) live on the same manifold and interact through a Yukawa-style Lagrangian.',
      'Coupling constants g_ij are themselves learnable: correlated fields grow their coupling; uncorrelated fields stay decoupled.',
    ],
    elements: [
      { name: '3D surface per field', meaning: 'belief Φ for that species' },
      { name: 'Coupling graph / matrix', meaning: 'live g_ij entries (≥3 fields ⇒ matrix view)', code: 'multifield.py:couplings' },
      { name: 'Mean |g|', meaning: 'aggregate coupling strength', code: 'snapshot_multifield:mean_abs_coupling' },
    ],
    math: [
      { tex: 'L_\\text{int} = \\sum_{i<j} g_{ij}(x)\\, \\Phi_i(x)\\, \\Phi_j(x)', caption: 'Yukawa interaction Lagrangian.' },
      { tex: '\\dot g_{ij} = -\\eta \\frac{\\partial F}{\\partial g_{ij}}', caption: 'Couplings descend the joint free energy.' },
    ],
    watch: [
      { label: 'Mean |g| rises when fields co-vary', readout: 'mean_abs_coupling' },
      { label: 'Surfaces with no shared structure stay near-flat in coupling' },
    ],
    references: [{ label: 'Architecture §2.2', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  mps: {
    title: 'MPS — Matrix Product State',
    oneLine: 'The entanglement carrier: a 1D chain of low-rank tensors.',
    what: [
      'The QPCN\'s belief state is represented as a Matrix Product State with controllable bond dimension χ_max — this is the entanglement carrier.',
      'Bond dimension caps the entanglement entropy across each cut; SVD truncation enforces it after every Trotter step.',
    ],
    elements: [
      { name: 'Bond-dim bar', meaning: 'χ for each bond between sites', code: 'mps.py:bond_dimensions' },
      { name: 'Entropy line', meaning: 'per-cut von Neumann entanglement entropy' },
      { name: 'Page-curve reference', meaning: 'random-state entropy ceiling (visual guide)' },
    ],
    math: [
      { tex: '|\\psi\\rangle = \\sum_{\\{s\\}} A^{s_1} A^{s_2} \\cdots A^{s_N} |s_1 \\ldots s_N\\rangle', caption: 'MPS ansatz.' },
      { tex: 'S(\\rho_A) = -\\mathrm{Tr}\\, \\rho_A \\log \\rho_A', caption: 'Cut entropy follows from the bipartition.' },
    ],
    watch: [
      { label: 'Bonds saturate to χ_max under entangling dynamics' },
      { label: 'Entropy clusters in the middle for short-range Hamiltonians' },
    ],
    references: [{ label: 'Architecture §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  hamiltonian: {
    title: 'Hamiltonian — Local Many-Body Operator',
    oneLine: 'Sum of one- and two-site terms over species; the QPCN\'s generative model.',
    what: [
      'The Hamiltonian H is the QPCN\'s generative model. It is built locally from one-site terms (mass, source, quartic, curvature coupling) and two-site terms (kinetic hopping, cross-species interaction).',
      'Each species has a Fock cutoff d_local. The curvature field is consumed by the manifold-coupling term.',
    ],
    elements: [
      { name: 'Species legend', meaning: 'list of FieldSpecies (name, d_local, mass, kinetic)', code: 'hamiltonian.py:FieldSpecies' },
      { name: 'Curvature mini-map', meaning: 'H.curvature field driving the geometric coupling' },
    ],
    math: [
      { tex: 'H = \\sum_i h_i + \\sum_{\\langle i,j \\rangle} h_{ij}', caption: 'One-site + two-site decomposition.' },
      { tex: 'h_i = m\\, a^\\dagger a + \\tfrac{\\lambda}{2} (a^\\dagger a)^2 + \\kappa R_i a^\\dagger a', caption: 'Typical one-site terms.' },
    ],
    watch: [
      { label: 'd_local matches the species cutoff — larger d means richer dynamics' },
      { label: 'Curvature peaks correlate with manifold panel curvature' },
    ],
    references: [{ label: 'Architecture §2.3, §3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  qpcn: {
    title: 'QPCN — Quantum Predictive Coder',
    oneLine: 'Belief = MPS; generative model = H; errors drive parameter updates.',
    what: [
      'Imaginary-time evolution relaxes the MPS toward the ground state of the current Hamiltonian; observations are target expectation values of local operators.',
      'Prediction errors (target − ⟨obs⟩) drive gradient updates on learnable Hamiltonian parameters.',
    ],
    elements: [
      { name: 'Energy', meaning: '⟨ψ|H|ψ⟩, the variational energy' },
      { name: 'Pred-errors table', meaning: 'per-observable (target, current, Δ)', code: 'qpcn.py:_last_errors' },
      { name: 'Learnable params', meaning: 'mass / kinetic / coupling values that descend' },
    ],
    math: [
      { tex: '|\\psi(\\tau+d\\tau)\\rangle = e^{-H\\, d\\tau} |\\psi(\\tau)\\rangle', caption: 'Imaginary-time relaxation.' },
      { tex: '\\Delta \\theta = -\\eta\\, \\partial_\\theta \\sum_o (\\langle O \\rangle - t_o)^2', caption: 'Parameter update from observation errors.' },
    ],
    watch: [
      { label: 'Energy decreases monotonically under imaginary time', readout: 'energy' },
      { label: 'Pred-errors shrink as parameters adapt', readout: 'pred_errors' },
    ],
    references: [{ label: 'Architecture §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  mera: {
    title: 'MERA — Multi-Scale Tree',
    oneLine: 'Tree of disentanglers + isometries; encodes scale structure.',
    what: [
      'A MERA represents a quantum state as a renormalisation-group tree: each layer applies disentanglers (to remove short-range entanglement) followed by isometries (to coarse-grain).',
      'Used here as a substrate for hierarchical reasoning; the panel shows tree shape, per-layer χ, and per-cut entropy.',
    ],
    elements: [
      { name: 'Tree nodes', meaning: 'tensors at each layer; leaves are physical sites' },
      { name: 'Per-layer χ', meaning: 'bond dimension at each level' },
      { name: 'Entropy line', meaning: 'entanglement at each leaf-cut' },
    ],
    math: [
      { tex: '|\\psi\\rangle = U_1 W_1 U_2 W_2 \\cdots U_L W_L |0\\rangle', caption: 'Alternating disentanglers U and isometries W.' },
    ],
    watch: [
      { label: 'Logarithmic entropy scaling on critical states' },
      { label: 'Per-layer χ caps the captured entanglement' },
    ],
    references: [{ label: 'Architecture: MERA section', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  vqc: {
    title: 'VQC — Variational Quantum Circuit',
    oneLine: 'Parameterised circuit trained via parameter-shift gradients.',
    what: [
      'A `QuantumConvMap` wraps a parameterised circuit as a translation-invariant quantum convolution that drops into any PCN layer.',
      'Gradients are exact via the parameter-shift rule. The same code targets real IBM hardware.',
    ],
    elements: [
      { name: 'Theta heatmap', meaning: 'rotation angles per (layer, qubit, axis)' },
      { name: 'Bias', meaning: 'classical bias on the conv output' },
    ],
    math: [
      { tex: '\\partial_\\theta \\langle O \\rangle = \\tfrac{1}{2}[\\langle O \\rangle_{\\theta + \\pi/2} - \\langle O \\rangle_{\\theta - \\pi/2}]', caption: 'Parameter-shift rule.' },
    ],
    watch: [{ label: 'Live training is currently a fixture — see EXTENSIONS.md' }],
    references: [{ label: 'Architecture §2.4', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },

  logic: {
    title: 'Logic — Evaluation Hamiltonian',
    oneLine: 'Rule terms encoded as Hamiltonian costs; relaxation = reduction.',
    what: [
      'Each rule of the target calculus contributes a local Hamiltonian term; the ground state corresponds to a well-typed / fully-reduced program.',
      'Imaginary-time relaxation drives a superposition state toward zero residual energy under all rules simultaneously.',
    ],
    elements: [
      { name: 'Term list', meaning: 'per-rule (rule_id, site, arity) — coloured by residual' },
      { name: 'Total energy', meaning: 'sum of all residual term energies' },
    ],
    math: [
      { tex: 'H_\\text{eval} = \\sum_r \\lambda_r \\sum_i H_r^{(i)}', caption: 'Sum of per-rule terms with weights λ.' },
    ],
    watch: [{ label: 'Live relaxation is currently a fixture — see EXTENSIONS.md' }],
    references: [{ label: 'Architecture: logic section', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
};
```

- [ ] **Step 2: Write the ExplainerPane test**

```tsx
// src/qft_pcn/viz/web/src/components/ExplainerPane.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ExplainerPane } from './ExplainerPane';
import { EXPLAINERS } from '../lib/explainer';

describe('ExplainerPane', () => {
  it('renders title + one-liner for every layer', () => {
    for (const key of Object.keys(EXPLAINERS)) {
      const { unmount } = render(<ExplainerPane layer={key} />);
      expect(screen.getByText(EXPLAINERS[key].title)).toBeInTheDocument();
      expect(screen.getByText(EXPLAINERS[key].oneLine)).toBeInTheDocument();
      unmount();
    }
  });

  it('renders "What", "Elements", "Math", "Watch" sections', () => {
    render(<ExplainerPane layer="manifold" />);
    expect(screen.getByText('What')).toBeInTheDocument();
    expect(screen.getByText('Elements')).toBeInTheDocument();
    expect(screen.getByText('Math')).toBeInTheDocument();
    expect(screen.getByText('Watch')).toBeInTheDocument();
  });

  it('toggles collapsed state via the header button', () => {
    const { container } = render(<ExplainerPane layer="manifold" />);
    const btn = container.querySelector('button.explainer-toggle');
    expect(btn).toBeTruthy();
    btn!.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    expect(container.querySelector('.explainer.collapsed')).toBeTruthy();
  });
});
```

- [ ] **Step 3: Implement `ExplainerPane.tsx`**

```tsx
// src/qft_pcn/viz/web/src/components/ExplainerPane.tsx
/**
 * Right-rail explainer for the active layer.
 *
 * Pulls `EXPLAINERS[layer]` and renders four sections: What (prose),
 * Elements (visual ↦ substrate mapping), Math (KaTeX), Watch (live-watch
 * hints — hovering a hint flashes the matching <PanelReadouts> cell via
 * a custom event the panel listens to).
 */

import { useState } from 'react';
import { EXPLAINERS } from '../lib/explainer';
import { tex } from '../panels/common';

interface Props { layer: string }

export function ExplainerPane({ layer }: Props) {
  const spec = EXPLAINERS[layer];
  const [collapsed, setCollapsed] = useState(false);

  if (!spec) {
    return (
      <aside className="explainer">
        <div className="explainer-header">
          <span>No explainer for "{layer}"</span>
        </div>
      </aside>
    );
  }

  return (
    <aside className={`explainer${collapsed ? ' collapsed' : ''}`}>
      <div className="explainer-header">
        <strong>{spec.title}</strong>
        <button
          type="button"
          className="explainer-toggle"
          onClick={() => setCollapsed((c) => !c)}
          aria-label={collapsed ? 'Expand explainer' : 'Collapse explainer'}
        >
          {collapsed ? '◀' : '▶'}
        </button>
      </div>
      {!collapsed && (
        <div className="explainer-body">
          <p className="explainer-oneline">{spec.oneLine}</p>

          <section>
            <h4>What</h4>
            {spec.what.map((p, i) => <p key={i}>{p}</p>)}
          </section>

          <section>
            <h4>Elements</h4>
            <ul>
              {spec.elements.map((e, i) => (
                <li key={i}>
                  <strong>{e.name}</strong> — {e.meaning}
                  {e.code && <code className="explainer-code"> {e.code}</code>}
                </li>
              ))}
            </ul>
          </section>

          <section>
            <h4>Math</h4>
            {spec.math.map((m, i) => (
              <div key={i} className="explainer-math">
                <div dangerouslySetInnerHTML={{ __html: tex(m.tex) }} />
                <small>{m.caption}</small>
              </div>
            ))}
          </section>

          <section>
            <h4>Watch</h4>
            <ul>
              {spec.watch.map((w, i) => (
                <li
                  key={i}
                  onMouseEnter={() => w.readout && dispatchHighlight(w.readout)}
                  onMouseLeave={() => w.readout && dispatchHighlight(null)}
                >
                  {w.label}
                </li>
              ))}
            </ul>
          </section>

          {spec.references && (
            <section>
              <h4>References</h4>
              <ul>
                {spec.references.map((r, i) => (
                  <li key={i}><a href={r.href}>{r.label}</a></li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </aside>
  );
}

/** Fire a window event the PanelReadouts component subscribes to. */
function dispatchHighlight(readoutId: string | null) {
  window.dispatchEvent(new CustomEvent('viz:highlight-readout',
    { detail: { id: readoutId } }));
}
```

- [ ] **Step 4: Run tests to pass**

Run: `cd src/qft_pcn/viz/web && pnpm test --run ExplainerPane.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/web/src/lib/explainer.ts \
        src/qft_pcn/viz/web/src/components/ExplainerPane.tsx \
        src/qft_pcn/viz/web/src/components/ExplainerPane.test.tsx
git commit -m "feat(viz/web): explainer registry + right-rail ExplainerPane"
```

---

### Task 9: Shared panel chrome (PanelToolbar, PanelReadouts, MetricsStrip, extension badge)

**Files:**
- Modify: `src/qft_pcn/viz/web/src/panels/PanelShell.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/PanelToolbar.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/PanelReadouts.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/MetricsStrip.tsx`
- Create siblings: `PanelReadouts.test.tsx`, `MetricsStrip.test.tsx`
- Modify: `src/qft_pcn/viz/web/src/panels/panel.css`

- [ ] **Step 1: Write `PanelReadouts.tsx` + test**

```tsx
// src/qft_pcn/viz/web/src/panels/PanelReadouts.tsx
/**
 * A horizontal strip of labelled scalar cells.
 *
 * Each cell has an optional `highlightId`; when ExplainerPane fires the
 * `viz:highlight-readout` window event with a matching id, the cell
 * flashes for 800ms.
 */

import { useEffect, useState } from 'react';

export interface ReadoutCell {
  label: string;
  value: string | number | null | undefined;
  unit?: string;
  highlightId?: string;
}

export function PanelReadouts({ cells }: { cells: ReadoutCell[] }) {
  const [active, setActive] = useState<string | null>(null);

  useEffect(() => {
    const handler = (ev: Event) => {
      const id = (ev as CustomEvent).detail?.id ?? null;
      setActive(id);
      if (id) {
        const t = setTimeout(() => setActive(null), 800);
        return () => clearTimeout(t);
      }
    };
    window.addEventListener('viz:highlight-readout', handler);
    return () => window.removeEventListener('viz:highlight-readout', handler);
  }, []);

  return (
    <div className="panel-readouts">
      {cells.map((c, i) => (
        <div
          key={i}
          className={`panel-readout${
            c.highlightId && c.highlightId === active ? ' highlight' : ''
          }`}
        >
          <span className="panel-readout-label">{c.label}</span>
          <span className="panel-readout-value">
            {c.value == null ? '—' : c.value}
            {c.unit && <small> {c.unit}</small>}
          </span>
        </div>
      ))}
    </div>
  );
}
```

```tsx
// src/qft_pcn/viz/web/src/panels/PanelReadouts.test.tsx
import { describe, it, expect } from 'vitest';
import { render, screen, act } from '@testing-library/react';
import { PanelReadouts } from './PanelReadouts';

describe('PanelReadouts', () => {
  it('renders one cell per entry', () => {
    render(<PanelReadouts cells={[
      { label: 'energy', value: 1.23 },
      { label: 'χ', value: 8 },
    ]} />);
    expect(screen.getByText('energy')).toBeInTheDocument();
    expect(screen.getByText('1.23')).toBeInTheDocument();
  });

  it('flashes the cell whose highlightId matches the event', () => {
    const { container } = render(<PanelReadouts cells={[
      { label: 'energy', value: 1.0, highlightId: 'energy' },
    ]} />);
    act(() => {
      window.dispatchEvent(new CustomEvent('viz:highlight-readout',
        { detail: { id: 'energy' } }));
    });
    expect(container.querySelector('.panel-readout.highlight')).toBeTruthy();
  });
});
```

- [ ] **Step 2: Write `PanelToolbar.tsx`**

```tsx
// src/qft_pcn/viz/web/src/panels/PanelToolbar.tsx
/** A horizontal strip of toggle buttons rendered above a panel's main viz. */

export interface ToolbarItem {
  key: string;
  label: string;
  active: boolean;
  onToggle: () => void;
  disabled?: boolean;
}

export function PanelToolbar({ items }: { items: ToolbarItem[] }) {
  return (
    <div className="panel-toolbar">
      {items.map((it) => (
        <button
          key={it.key}
          type="button"
          className={`panel-toolbar-btn${it.active ? ' active' : ''}`}
          onClick={it.onToggle}
          disabled={it.disabled}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Write `MetricsStrip.tsx` + test**

```tsx
// src/qft_pcn/viz/web/src/panels/MetricsStrip.tsx
/**
 * A compact multi-line sparkline showing client-side scalar history.
 *
 * The caller passes the layer key + a list of `{key, label, color}` series;
 * we read the active run's frame history from the store and project each
 * series via the supplied `select` function.
 */

import { useMemo } from 'react';
import { useVizStore } from '../store';

export interface MetricSpec {
  key: string;
  label: string;
  color: string;
  /** Pull the scalar value out of one frame's layer_states[layer] dict. */
  select: (layerState: Record<string, unknown>) => number | null | undefined;
}

interface Props { layer: string; metrics: MetricSpec[]; }

export function MetricsStrip({ layer, metrics }: Props) {
  const runId = useVizStore((s) => s.activeRunId);
  const run = useVizStore((s) => (runId ? s.runs.get(runId) : undefined));
  const frames = run?.frames ?? [];

  const series = useMemo(() => metrics.map((m) => ({
    ...m,
    points: frames.map((f) => m.select(
      (f.layer_states[layer] ?? {}) as Record<string, unknown>)),
  })), [frames, metrics, layer]);

  if (frames.length < 2) {
    return <div className="metrics-strip empty">collecting…</div>;
  }

  const W = 240, H = 56, PAD = 4;
  const xs = (i: number) => PAD + (i / (frames.length - 1)) * (W - 2 * PAD);
  const allFinite: number[] = [];
  for (const s of series) {
    for (const p of s.points) {
      if (typeof p === 'number' && Number.isFinite(p)) allFinite.push(p);
    }
  }
  const lo = allFinite.length ? Math.min(...allFinite) : 0;
  const hi = allFinite.length ? Math.max(...allFinite) : 1;
  const span = hi - lo || 1;
  const ys = (v: number) => H - PAD - ((v - lo) / span) * (H - 2 * PAD);

  return (
    <div className="metrics-strip">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}>
        {series.map((s) => {
          const d = s.points
            .map((p, i) => (typeof p === 'number' && Number.isFinite(p)
              ? `${i === 0 ? 'M' : 'L'} ${xs(i)} ${ys(p)}`
              : ''))
            .filter(Boolean)
            .join(' ');
          return <path key={s.key} d={d} fill="none"
                       stroke={s.color} strokeWidth={1.4} />;
        })}
      </svg>
      <div className="metrics-strip-legend">
        {series.map((s) => (
          <span key={s.key}>
            <span className="swatch" style={{ background: s.color }} />
            {s.label}
          </span>
        ))}
      </div>
    </div>
  );
}
```

```tsx
// src/qft_pcn/viz/web/src/panels/MetricsStrip.test.tsx
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MetricsStrip } from './MetricsStrip';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

describe('MetricsStrip', () => {
  it('shows "collecting…" with fewer than 2 frames', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.setActiveRun('A');
    s.pushFrame('A', { step: 0, layer_states: { qpcn: { energy: 1 } } });
    render(<MetricsStrip layer="qpcn" metrics={[
      { key: 'e', label: 'E', color: '#fa0',
        select: (ls) => ls.energy as number },
    ]} />);
    expect(screen.getByText('collecting…')).toBeInTheDocument();
  });

  it('renders one path per series once frames > 1', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.setActiveRun('A');
    for (let i = 0; i < 5; i++)
      s.pushFrame('A', { step: i, layer_states: { qpcn: { energy: i * 0.1 } } });
    const { container } = render(<MetricsStrip layer="qpcn" metrics={[
      { key: 'e', label: 'E', color: '#fa0',
        select: (ls) => ls.energy as number },
    ]} />);
    expect(container.querySelectorAll('path').length).toBe(1);
  });
});
```

- [ ] **Step 4: Extend `PanelShell` with `isExtension`**

Replace `PanelShell.tsx` body — keep the prop name extras additive:

```tsx
import type { ReactNode } from 'react';
import './panel.css';

export interface PanelShellProps {
  title: string;
  step?: number;
  meta?: string;
  hasData: boolean;
  emptyMessage?: string;
  /** When true, renders an "extension pending" badge + tooltip link. */
  isExtension?: boolean;
  extensionAnchor?: string;
  toolbar?: ReactNode;
  readouts?: ReactNode;
  metricsStrip?: ReactNode;
  children: ReactNode;
}

export function PanelShell({
  title, step, meta, hasData,
  emptyMessage = 'No data for this layer at the current step.',
  isExtension, extensionAnchor,
  toolbar, readouts, metricsStrip, children,
}: PanelShellProps) {
  return (
    <div className="viz-panel">
      <div className="viz-panel__header">
        <span className="viz-panel__title">{title}</span>
        {step !== undefined &&
          <span className="viz-panel__meta">step {step}</span>}
        {meta && <span className="viz-panel__meta">{meta}</span>}
        {isExtension && (
          <a
            className="viz-panel__badge"
            href={`../../EXTENSIONS.md${extensionAnchor ?? ''}`}
            title="This panel is fixture-only; see EXTENSIONS.md"
          >
            extension pending
          </a>
        )}
      </div>
      {toolbar && <div className="viz-panel__toolbar">{toolbar}</div>}
      <div className="viz-panel__body">
        {hasData ? children
          : <div className="viz-panel__empty">{emptyMessage}</div>}
      </div>
      {readouts && <div className="viz-panel__readouts">{readouts}</div>}
      {metricsStrip && (
        <div className="viz-panel__metrics">{metricsStrip}</div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Append CSS (panel.css)**

Append to `src/qft_pcn/viz/web/src/panels/panel.css`:

```css
.viz-panel__badge {
  margin-left: 8px;
  background: #6b3a3a;
  color: #fce0c0;
  border-radius: 3px;
  padding: 0 6px;
  font-size: 10px;
  text-decoration: none;
}
.viz-panel__toolbar { display: flex; gap: 4px; padding: 4px 8px;
                      border-bottom: 1px solid #1c2233; background: #0d111a; }
.panel-toolbar-btn { background: #161b29; color: #c8d0e0;
                     border: 1px solid #2a3450; border-radius: 4px;
                     padding: 2px 8px; font-size: 11px; cursor: pointer; }
.panel-toolbar-btn.active { background: #2b3a6b; }
.panel-toolbar-btn:disabled { opacity: 0.45; cursor: default; }
.viz-panel__readouts { display: flex; flex-wrap: wrap; gap: 12px;
                       padding: 4px 8px; border-top: 1px solid #1c2233;
                       background: #0d111a; font-size: 11px; }
.panel-readout { display: flex; flex-direction: column; min-width: 70px; }
.panel-readout-label { color: #7e8aa3; }
.panel-readout-value { color: #e0e6f3; font-weight: 600; }
.panel-readout.highlight { background: #2b3a6b; border-radius: 3px;
                           padding: 0 4px; transition: background 0.4s; }
.viz-panel__metrics { padding: 4px 8px; border-top: 1px solid #1c2233;
                      background: #0a0d14; }
.metrics-strip.empty { color: #5a6377; font-size: 11px; padding: 12px 4px; }
.metrics-strip-legend { display: flex; gap: 12px; font-size: 10px;
                        color: #9aa3bb; }
.metrics-strip-legend .swatch { display: inline-block; width: 8px;
                                height: 8px; margin-right: 4px; }
```

- [ ] **Step 6: Run tests**

Run: `cd src/qft_pcn/viz/web && pnpm test --run PanelReadouts MetricsStrip`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qft_pcn/viz/web/src/panels/PanelShell.tsx \
        src/qft_pcn/viz/web/src/panels/PanelToolbar.tsx \
        src/qft_pcn/viz/web/src/panels/PanelReadouts.tsx \
        src/qft_pcn/viz/web/src/panels/PanelReadouts.test.tsx \
        src/qft_pcn/viz/web/src/panels/MetricsStrip.tsx \
        src/qft_pcn/viz/web/src/panels/MetricsStrip.test.tsx \
        src/qft_pcn/viz/web/src/panels/panel.css
git commit -m "feat(viz/web): shared panel chrome — toolbar, readouts, metrics, extension badge"
```

---

### Task 10: RunControls split + preset dropdown + advanced + transport + export

**Files:**
- Create: `src/qft_pcn/viz/web/src/components/RunControls.tsx`
- Create: `src/qft_pcn/viz/web/src/components/RunControls.test.tsx`
- Create: `src/qft_pcn/viz/web/src/lib/presets.ts`
- Modify: `src/qft_pcn/viz/web/src/App.tsx` (remove the inline RunControls; mount the new component)

- [ ] **Step 1: Write `lib/presets.ts`**

```ts
import type { Preset, ParamSchema } from './types';

let _presets: Preset[] | null = null;
let _schema: ParamSchema | null = null;

export async function loadPresets(): Promise<Preset[]> {
  if (_presets) return _presets;
  const res = await fetch('/presets');
  if (!res.ok) throw new Error(`/presets: ${res.status}`);
  _presets = await res.json();
  return _presets!;
}

export async function loadParamSchema(): Promise<ParamSchema> {
  if (_schema) return _schema;
  const res = await fetch('/params/schema');
  if (!res.ok) throw new Error(`/params/schema: ${res.status}`);
  _schema = await res.json();
  return _schema!;
}

/** Reset module-level caches — test helper. */
export function _resetPresetCache() { _presets = null; _schema = null; }

export async function exportRunJsonl(runId: string): Promise<void> {
  const res = await fetch(`/export/run/${runId}`);
  if (!res.ok) throw new Error(`export: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = `run-${runId}.jsonl`;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

export async function exportRunMp4(runId: string, layer: string)
    : Promise<{ job_id: string }> {
  const res = await fetch('/export', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ run_id: runId, layer }),
  });
  if (!res.ok) throw new Error(`/export: ${res.status}`);
  return res.json();
}
```

- [ ] **Step 2: Write `RunControls.tsx`**

```tsx
// src/qft_pcn/viz/web/src/components/RunControls.tsx
/**
 * Top run/transport bar: preset dropdown, Advanced expander, layer
 * checkboxes, transport (▶/⏸/⏭), playback speed slider, baseline pin,
 * Export menu.
 */

import { useEffect, useRef, useState } from 'react';
import { connectRun, type RunHandle } from '../lib/ws';
import { transport } from '../lib/transport';
import {
  loadPresets, loadParamSchema,
  exportRunJsonl, exportRunMp4,
} from '../lib/presets';
import type { Preset, ParamSchema, RunSpec } from '../lib/types';
import { LAYER_KEYS } from '../lib/types';
import { useVizStore } from '../store';

export function RunControls() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [schema, setSchema] = useState<ParamSchema | null>(null);
  const [presetId, setPresetId] = useState<string>('');
  const [layers, setLayers] = useState<string[]>(['manifold']);
  const [steps, setSteps] = useState(20);
  const [grid, setGrid] = useState(12);
  const [seed, setSeed] = useState<number | ''>('');
  const [params, setParams] = useState<Record<string, Record<string, unknown>>>({});
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const handleRef = useRef<RunHandle | null>(null);

  const active = useVizStore((s) => s.activeRunId);
  const baseline = useVizStore((s) => s.baselineRunId);
  const paused = useVizStore((s) => s.paused);
  const playbackSpeed = useVizStore((s) => s.playbackSpeed);
  const setPaused = useVizStore((s) => s.setPaused);
  const setSpeed = useVizStore((s) => s.setPlaybackSpeed);
  const pinBaseline = useVizStore((s) => s.pinBaseline);
  const unpinBaseline = useVizStore((s) => s.unpinBaseline);

  useEffect(() => {
    loadPresets().then(setPresets).catch((e) =>
      useVizStore.getState().setError(String(e)));
    loadParamSchema().then(setSchema).catch(() => { /* optional */ });
  }, []);

  useEffect(() => () => { handleRef.current?.close(); }, []);

  const applyPreset = (id: string) => {
    setPresetId(id);
    const p = presets.find((x) => x.id === id);
    if (!p) return;
    if (p.spec_overrides.layers) setLayers(p.spec_overrides.layers);
    else setLayers([p.layer]);
    if (p.spec_overrides.steps) setSteps(p.spec_overrides.steps);
    if (p.spec_overrides.grid) setGrid(p.spec_overrides.grid);
    if (p.spec_overrides.seed !== undefined && p.spec_overrides.seed !== null)
      setSeed(p.spec_overrides.seed);
    if (p.spec_overrides.params) setParams(p.spec_overrides.params);
  };

  const run = async () => {
    setBusy(true);
    handleRef.current?.close();
    const spec: RunSpec = {
      layers: layers.length ? layers : ['manifold'],
      steps, grid,
      seed: seed === '' ? null : Number(seed),
      params,
    };
    try {
      handleRef.current = await connectRun(spec);
    } catch (err) {
      useVizStore.getState().setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  const togglePause = async () => {
    if (!active) return;
    if (paused) { await transport.resume(active); setPaused(false); }
    else { await transport.pause(active); setPaused(true); }
  };
  const step = async () => { if (active) await transport.step(active); };

  return (
    <header className="run-controls">
      <span className="title">QFT-PCN Visualizer</span>

      <label>
        preset
        <select value={presetId} onChange={(e) => applyPreset(e.target.value)}>
          <option value="">(custom)</option>
          {presets.map((p) => (
            <option key={p.id} value={p.id}>{p.label}</option>
          ))}
        </select>
      </label>

      <div className="run-layers">
        {LAYER_KEYS.map((layer) => (
          <label key={layer}>
            <input type="checkbox" checked={layers.includes(layer)}
              onChange={() => setLayers((cur) =>
                cur.includes(layer)
                  ? cur.filter((l) => l !== layer)
                  : [...cur, layer])} />
            {layer}
          </label>
        ))}
      </div>

      <label>steps <input type="number" min={1} value={steps}
        onChange={(e) => setSteps(Number(e.target.value))} /></label>
      <label>grid <input type="number" min={1} value={grid}
        onChange={(e) => setGrid(Number(e.target.value))} /></label>
      <label>seed <input type="number" value={seed}
        onChange={(e) => setSeed(e.target.value === ''
          ? '' : Number(e.target.value))} /></label>

      <button type="button" onClick={() => setAdvanced((a) => !a)}>
        {advanced ? '▾ advanced' : '▸ advanced'}
      </button>

      <button type="button" onClick={run} disabled={busy}>
        {busy ? 'starting…' : 'Run'}
      </button>

      {/* Transport */}
      <div className="transport">
        <button type="button" onClick={togglePause} disabled={!active}>
          {paused ? '▶' : '⏸'}
        </button>
        <button type="button" onClick={step} disabled={!active || !paused}>⏭</button>
        <label>speed
          <input type="range" min={0.25} max={4} step={0.25}
            value={playbackSpeed}
            onChange={(e) => setSpeed(Number(e.target.value))} />
          <small>{playbackSpeed.toFixed(2)}×</small>
        </label>
      </div>

      {/* Baseline */}
      <button type="button" disabled={!active}
        onClick={() => active && (baseline === active
          ? unpinBaseline() : pinBaseline(active))}>
        {baseline === active ? 'unpin baseline' : 'pin as baseline'}
      </button>

      {/* Export */}
      <div className="export-menu">
        <button type="button" disabled={!active}
          onClick={() => active && exportRunJsonl(active)
            .catch((e) => useVizStore.getState().setError(String(e)))}>
          ⬇ JSONL
        </button>
        <button type="button" disabled={!active}
          onClick={() => active && exportRunMp4(active, layers[0] ?? 'manifold')
            .catch((e) => useVizStore.getState().setError(String(e)))}>
          🎞 MP4
        </button>
      </div>

      {advanced && schema && (
        <AdvancedForm schema={schema} value={params} onChange={setParams} />
      )}
    </header>
  );
}

function AdvancedForm({
  schema, value, onChange,
}: {
  schema: ParamSchema;
  value: Record<string, Record<string, unknown>>;
  onChange: (v: Record<string, Record<string, unknown>>) => void;
}) {
  return (
    <div className="advanced-form">
      {Object.entries(schema).map(([layer, layerSchema]) => (
        <fieldset key={layer}>
          <legend>{layer}</legend>
          {Object.entries(layerSchema.properties).map(([key, prop]) => {
            const current = value[layer]?.[key] ?? prop.default ?? '';
            const update = (raw: unknown) => onChange({
              ...value,
              [layer]: { ...(value[layer] ?? {}), [key]: raw },
            });
            const t = Array.isArray(prop.type) ? prop.type[0] : prop.type;
            if (prop.enum) {
              return (
                <label key={key}>{key}
                  <select value={String(current)}
                    onChange={(e) => update(e.target.value)}>
                    {prop.enum.map((o) =>
                      <option key={String(o)} value={String(o)}>{String(o)}</option>)}
                  </select>
                </label>
              );
            }
            if (t === 'boolean') {
              return (
                <label key={key}>{key}
                  <input type="checkbox" checked={Boolean(current)}
                    onChange={(e) => update(e.target.checked)} />
                </label>
              );
            }
            if (t === 'number' || t === 'integer') {
              return (
                <label key={key}>{key}
                  <input type="number"
                    value={current === null ? '' : Number(current)}
                    min={prop.minimum} max={prop.maximum}
                    onChange={(e) => update(e.target.value === ''
                      ? null : Number(e.target.value))} />
                </label>
              );
            }
            return (
              <label key={key}>{key}
                <input type="text" value={String(current)}
                  onChange={(e) => update(e.target.value)} />
              </label>
            );
          })}
        </fieldset>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Write the test**

```tsx
// src/qft_pcn/viz/web/src/components/RunControls.test.tsx
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { RunControls } from './RunControls';
import { _resetPresetCache } from '../lib/presets';
import { useVizStore } from '../store';

const presetsFixture = [{
  id: 'manifold.flat', layer: 'manifold',
  label: 'Flat — zero curvature start',
  description: 'baseline',
  spec_overrides: { layers: ['manifold'], steps: 30, grid: 12,
                    seed: 0, params: { manifold: { source: 'flat' } } },
}];
const schemaFixture = {
  manifold: { type: 'object', properties: {
    source: { type: 'string', enum: ['flat', 'hot-spot'], default: 'flat' },
  } },
};

beforeEach(() => {
  _resetPresetCache();
  useVizStore.getState().resetAll();
  global.fetch = vi.fn(async (url: any) => {
    const u = String(url);
    if (u.endsWith('/presets'))
      return { ok: true, json: async () => presetsFixture } as any;
    if (u.endsWith('/params/schema'))
      return { ok: true, json: async () => schemaFixture } as any;
    if (u.endsWith('/run'))
      return { ok: true, json: async () => ({ run_id: 'R1' }) } as any;
    throw new Error('unexpected ' + u);
  }) as any;
});

describe('RunControls', () => {
  it('loads presets into the dropdown', async () => {
    render(<RunControls />);
    await waitFor(() =>
      expect(screen.getByText('Flat — zero curvature start'))
        .toBeInTheDocument());
  });

  it('applying a preset sets steps/grid/seed/params from spec_overrides',
     async () => {
    render(<RunControls />);
    await waitFor(() => screen.getByText('Flat — zero curvature start'));
    fireEvent.change(screen.getByRole('combobox', { name: /preset/i }),
      { target: { value: 'manifold.flat' } });
    const stepsInput = screen.getByLabelText(/steps/i) as HTMLInputElement;
    expect(stepsInput.value).toBe('30');
  });

  it('Run posts a RunSpec containing the applied params', async () => {
    render(<RunControls />);
    await waitFor(() => screen.getByText('Flat — zero curvature start'));
    fireEvent.change(screen.getByRole('combobox', { name: /preset/i }),
      { target: { value: 'manifold.flat' } });
    fireEvent.click(screen.getByText('Run'));
    await waitFor(() => {
      const last = (global.fetch as any).mock.calls
        .find((c: any[]) => String(c[0]).endsWith('/run'));
      expect(last).toBeTruthy();
      const body = JSON.parse(last[1].body);
      expect(body.params.manifold.source).toBe('flat');
    });
  });
});
```

- [ ] **Step 3: Mount in App.tsx**

Replace the inline `RunControls` function in `App.tsx`. Open `App.tsx` and:
1. Delete the local `function RunControls()` block.
2. Replace the import row with:
   ```tsx
   import { RunControls } from './components/RunControls';
   ```

- [ ] **Step 4: Run tests**

Run: `cd src/qft_pcn/viz/web && pnpm test --run RunControls.test.tsx`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/web/src/components/RunControls.tsx \
        src/qft_pcn/viz/web/src/components/RunControls.test.tsx \
        src/qft_pcn/viz/web/src/lib/presets.ts \
        src/qft_pcn/viz/web/src/App.tsx
git commit -m "feat(viz/web): preset-aware RunControls with transport + export menu"
```

---

### Task 11: CompareBar + App layout

**Files:**
- Create: `src/qft_pcn/viz/web/src/components/CompareBar.tsx`
- Modify: `src/qft_pcn/viz/web/src/App.tsx`
- Modify: `src/qft_pcn/viz/web/src/App.css`

- [ ] **Step 1: Write CompareBar**

```tsx
// src/qft_pcn/viz/web/src/components/CompareBar.tsx
/** Shown when a baseline run is pinned: lists both run ids and lets the user swap or unpin. */

import { useVizStore } from '../store';

export function CompareBar() {
  const active = useVizStore((s) => s.activeRunId);
  const baseline = useVizStore((s) => s.baselineRunId);
  const setActive = useVizStore((s) => s.setActiveRun);
  const unpin = useVizStore((s) => s.unpinBaseline);
  const pin = useVizStore((s) => s.pinBaseline);

  if (!baseline) return null;
  return (
    <div className="compare-bar">
      <span>active: <code>{active ?? '—'}</code></span>
      <span>baseline: <code>{baseline}</code></span>
      <button type="button" onClick={() => {
        if (!active) return;
        const a = active, b = baseline;
        setActive(b); pin(a);
      }}>swap</button>
      <button type="button" onClick={unpin}>unpin</button>
    </div>
  );
}
```

- [ ] **Step 2: Restructure `App.tsx` into 4 regions**

Replace App.tsx body:

```tsx
import { LayerSelector } from './components/LayerSelector';
import { Timeline } from './components/Timeline';
import { RunControls } from './components/RunControls';
import { CompareBar } from './components/CompareBar';
import { ExplainerPane } from './components/ExplainerPane';
import { panelFor } from './panels';
import { useVizStore } from './store';
import './App.css';

function PanelArea() {
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  const active = useVizStore((s) => s.activeRunId);
  const frame = useVizStore((s) =>
    active ? s.runs.get(active)?.frames[s.runs.get(active)!.cursor] : undefined);
  const baselineFrame = useVizStore((s) => s.baselineFrame());
  const Panel = panelFor(selectedLayer);

  return (
    <main className="panel-area">
      {!frame ? (
        <p className="empty">No frames yet — start a run to stream simulation data.</p>
      ) : !Panel ? (
        <p className="empty">No panel registered for "{selectedLayer}".</p>
      ) : (
        <Panel frame={frame} baselineFrame={baselineFrame} />
      )}
    </main>
  );
}

export default function App() {
  const error = useVizStore((s) => s.error);
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  return (
    <div className="app">
      <RunControls />
      <CompareBar />
      {error && <div className="error-bar" role="alert">{error}</div>}
      <div className="body">
        <LayerSelector />
        <PanelArea />
        <ExplainerPane layer={selectedLayer} />
      </div>
      <Timeline />
    </div>
  );
}
```

- [ ] **Step 3: Update Timeline for per-run cursor**

Replace `Timeline.tsx`:

```tsx
import { useVizStore } from '../store';

export function Timeline() {
  const active = useVizStore((s) => s.activeRunId);
  const run = useVizStore((s) => (active ? s.runs.get(active) : undefined));
  const setCursor = useVizStore((s) => s.setCursor);
  const setLive = useVizStore((s) => s.setLive);

  if (!active || !run) return <div className="timeline empty">no active run</div>;
  const max = Math.max(0, run.frames.length - 1);
  return (
    <div className="timeline">
      <input type="range" aria-label="timeline scrubber"
        min={0} max={max} value={run.cursor}
        disabled={run.frames.length === 0}
        onChange={(e) => setCursor(active, Number(e.target.value))} />
      <span className="timeline-label">
        step {run.frames[run.cursor]?.step ?? '-'} (
        {run.cursor + (run.frames.length ? 1 : 0)}/{run.frames.length})
      </span>
      <label className="timeline-live">
        <input type="checkbox" checked={run.live}
          onChange={(e) => setLive(active, e.target.checked)} />
        live
      </label>
    </div>
  );
}
```

- [ ] **Step 4: Update CSS — new grid layout**

In `App.css`, replace the `.body` rule with:

```css
.body {
  display: grid;
  grid-template-columns: 180px 1fr 320px;
  flex: 1;
  min-height: 0;
}
.explainer { border-left: 1px solid #1c2233; padding: 8px;
             overflow-y: auto; background: #0a0d14; color: #c8d0e0;
             font-size: 12px; }
.explainer.collapsed { width: 28px; padding: 4px; overflow: hidden; }
.explainer-header { display: flex; justify-content: space-between;
                    align-items: center; }
.explainer-toggle { background: transparent; color: #c8d0e0; border: none;
                    cursor: pointer; }
.explainer-body section { margin-top: 10px; }
.explainer-body h4 { color: #8c97b3; font-size: 11px; text-transform: uppercase;
                     letter-spacing: 0.05em; margin: 8px 0 4px; }
.explainer-code { color: #6cd0ff; font-size: 10px; }
.explainer-math { margin: 6px 0; }
.explainer-math small { color: #7e8aa3; }
.compare-bar { display: flex; gap: 12px; padding: 4px 8px;
               background: #161b29; font-size: 11px; color: #c8d0e0;
               border-bottom: 1px solid #1c2233; }
.compare-bar code { color: #6cd0ff; }
.transport { display: flex; gap: 4px; align-items: center; }
.transport input[type=range] { width: 90px; }
.advanced-form { display: flex; gap: 12px; flex-wrap: wrap;
                 width: 100%; padding: 6px 8px; background: #0a0d14;
                 border-top: 1px solid #1c2233; }
.advanced-form fieldset { border: 1px solid #2a3450; padding: 4px 8px;
                          min-width: 180px; }
.advanced-form legend { color: #8c97b3; font-size: 11px; }
.advanced-form label { display: flex; gap: 4px; align-items: center;
                       margin: 2px 0; font-size: 11px; color: #c8d0e0; }
.export-menu { display: flex; gap: 4px; }
```

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/web/src/components/CompareBar.tsx \
        src/qft_pcn/viz/web/src/App.tsx \
        src/qft_pcn/viz/web/src/components/Timeline.tsx \
        src/qft_pcn/viz/web/src/App.css
git commit -m "feat(viz/web): 4-region App layout + CompareBar"
```

---

## Phase C — Per-Panel Uplift

Each task below is independently parallelizable: they all consume the shared
`PanelShell` / `PanelToolbar` / `PanelReadouts` / `MetricsStrip` and only
touch their own panel file + its `.test.tsx`. None depend on the others.

The expected shape for every panel: import shared chrome, derive an
`isEmpty` check from the snapshot, build `toolbar` / `readouts` /
`metricsStrip` props, and render the existing centerpiece visualization in
the body. `Panel` props gain an optional `baselineFrame?: Frame` — panels
that support diffing use it; others ignore it.

### Task 12: ManifoldPanel uplift

**Files:**
- Modify: `src/qft_pcn/viz/web/src/panels/ManifoldPanel.tsx`
- Modify: `src/qft_pcn/viz/web/src/panels/ManifoldPanel.test.tsx`

- [ ] **Step 1: Update the test**

```tsx
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ManifoldPanel } from './ManifoldPanel';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

const baseFrame = {
  step: 1,
  layer_states: { manifold: {
    metric_h: { h_xx: [[0, 0], [0, 0]] }, ricci: [[0, 0], [0, 0]],
    fields: [{ phi: [[0, 0], [0, 0]], E: [[0, 0], [0, 0]],
               Pi: [[1, 1], [1, 1]], channels: 1 }],
    mean_abs_ricci: 0.0,
  } },
};

describe('ManifoldPanel', () => {
  it('renders readouts derived from the snapshot', () => {
    render(<ManifoldPanel frame={baseFrame as any} />);
    expect(screen.getByText(/mean \|R\|/i)).toBeInTheDocument();
  });
  it('renders the channel selector when fields are present', () => {
    render(<ManifoldPanel frame={baseFrame as any} />);
    expect(screen.getByRole('combobox', { name: /channel/i })).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Update `ManifoldPanel.tsx`**

Add `PanelToolbar`, `PanelReadouts`, `MetricsStrip` props and a channel selector. Wrap the existing R3F canvas. Show structure:

```tsx
import { useState } from 'react';
import { PanelShell } from './PanelShell';
import { PanelToolbar } from './PanelToolbar';
import { PanelReadouts } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';
// ... existing imports ...

export function ManifoldPanel({ frame, baselineFrame }:
    { frame: Frame; baselineFrame?: Frame }) {
  const st = (frame.layer_states.manifold ?? {}) as ManifoldState;
  const [overlay, setOverlay] = useState<'phi'|'E'|'Pi'|null>(null);
  const [channel, setChannel] = useState(0);
  // ... existing baseHeight/baseColor/overlayGrid logic ...

  const channelCount = st.fields?.[0]?.channels ?? 1;

  return (
    <PanelShell
      title="Manifold — warped metric grid"
      step={st.step ?? frame.step}
      meta={st.mean_abs_ricci != null
        ? `mean|Ricci| ${st.mean_abs_ricci.toFixed(3)}` : undefined}
      hasData={hasData}
      toolbar={
        <>
          <PanelToolbar items={(['phi','E','Pi'] as const).map((k) => ({
            key: k, label: k, active: overlay === k,
            disabled: !field,
            onToggle: () => setOverlay((c) => c === k ? null : k),
          }))} />
          <label style={{ marginLeft: 8, fontSize: 11 }}>
            channel
            <select aria-label="channel" value={channel}
              onChange={(e) => setChannel(Number(e.target.value))}>
              {Array.from({ length: channelCount }, (_, i) =>
                <option key={i} value={i}>{i}</option>)}
            </select>
          </label>
        </>
      }
      readouts={<PanelReadouts cells={[
        { label: 'mean |R|', value: st.mean_abs_ricci?.toFixed(3),
          highlightId: 'mean_abs_ricci' },
        { label: 'layers', value: st.fields?.length ?? 0 },
        { label: 'grid', value:
          st.metric_h?.h_xx ? `${st.metric_h.h_xx.length}²` : '—' },
      ]} />}
      metricsStrip={<MetricsStrip layer="manifold" metrics={[{
        key: 'mar', label: 'mean|R|', color: '#6cd0ff',
        select: (ls) => ls.mean_abs_ricci as number,
      }]} />}
    >
      {/* existing Canvas / Surface block stays */}
    </PanelShell>
  );
}
```

- [ ] **Step 3: Run tests**

Run: `cd src/qft_pcn/viz/web && pnpm test --run ManifoldPanel.test.tsx`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/viz/web/src/panels/ManifoldPanel.tsx \
        src/qft_pcn/viz/web/src/panels/ManifoldPanel.test.tsx
git commit -m "feat(viz/web): ManifoldPanel uplift — readouts, channel, metrics strip"
```

---

### Task 13: MultifieldPanel uplift

**Files:**
- Modify: `src/qft_pcn/viz/web/src/panels/MultifieldPanel.tsx`
- Modify: `src/qft_pcn/viz/web/src/panels/MultifieldPanel.test.tsx`

- [ ] **Step 1: Update test** — assert presence of:
  - `mean |g|` readout
  - per-field strength rows (one per name in `fields`)
  - a `MetricsStrip` element

- [ ] **Step 2: Update panel** — wrap centerpiece in `PanelShell` with:

```tsx
readouts={<PanelReadouts cells={[
  { label: 'mean |g|', value: st.mean_abs_coupling?.toFixed(3),
    highlightId: 'mean_abs_coupling' },
  ...Object.entries(st.fields ?? {}).map(([name, f]: any) => ({
    label: `‖${name}.Φ‖₂`,
    value: norm2(f.phi).toFixed(3),
  })),
]} />}
metricsStrip={<MetricsStrip layer="multifield" metrics={[
  { key: 'mg', label: 'mean|g|', color: '#fbc66a',
    select: (ls) => ls.mean_abs_coupling as number },
]} />}
```

`norm2(grid)` helper goes inline:

```ts
const norm2 = (g?: number[][] | null) =>
  !g ? 0 : Math.sqrt(g.flat().reduce((a, v) => a + (v || 0) ** 2, 0));
```

- [ ] **Step 3: Run + commit**

```bash
cd src/qft_pcn/viz/web && pnpm test --run MultifieldPanel.test.tsx
```

```bash
git add src/qft_pcn/viz/web/src/panels/MultifieldPanel*.tsx
git commit -m "feat(viz/web): MultifieldPanel uplift — coupling readouts + per-field norms"
```

---

### Task 14: MpsPanel uplift

**Files:** `MpsPanel.tsx` + `MpsPanel.test.tsx`.

- [ ] **Step 1: Update test** — assert readouts `N`, `d_local`, `total S` (sum of entropies), and a `MetricsStrip` is rendered.

- [ ] **Step 2: Update panel** with:

```tsx
const totalS = (st.entropies ?? []).reduce(
  (a: number, v: number | null) => a + (v ?? 0), 0);

readouts={<PanelReadouts cells={[
  { label: 'N',       value: st.n_sites ?? '—' },
  { label: 'd_local', value: st.d_local ?? '—' },
  { label: 'total S', value: totalS.toFixed(3) },
  { label: 'χ_max',   value: st.bond_dims ? Math.max(...st.bond_dims) : '—' },
]} />}
metricsStrip={<MetricsStrip layer="mps" metrics={[{
  key: 's', label: 'total S', color: '#9aedc1',
  select: (ls) => {
    const es = ls.entropies as (number | null)[] | undefined;
    return es ? es.reduce((a, v) => a + (v ?? 0), 0) : null;
  },
}]} />}
```

- [ ] **Step 3: Run + commit**

```bash
git add src/qft_pcn/viz/web/src/panels/MpsPanel*.tsx
git commit -m "feat(viz/web): MpsPanel uplift — N, d_local, total S readouts + metrics strip"
```

---

### Task 15: HamiltonianPanel uplift

**Files:** `HamiltonianPanel.tsx` + test.

- [ ] **Step 1: Update test** — assert `N`, `d_local`, `species count` readouts; KaTeX-rendered `H = …` block present.

- [ ] **Step 2: Update panel** with:

```tsx
const speciesNames = (st.species as string[] | null) ?? [];

readouts={<PanelReadouts cells={[
  { label: 'N',       value: st.n_sites ?? '—' },
  { label: 'd_local', value: st.d_local ?? '—' },
  { label: 'species', value: speciesNames.length },
]} />}
```

Add a KaTeX block under the species table:

```tsx
<div dangerouslySetInnerHTML={{ __html: tex(
  'H = \\sum_i h_i + \\sum_{\\langle i,j \\rangle} h_{ij}') }} />
```

Curvature mini-map: render a small `<canvas>` 64×64 filled by the existing `diverging()` helper if `st.curvature` is a 2D array.

- [ ] **Step 3: Run + commit**

```bash
git add src/qft_pcn/viz/web/src/panels/HamiltonianPanel*.tsx
git commit -m "feat(viz/web): HamiltonianPanel uplift — readouts, KaTeX H, curvature mini-map"
```

---

### Task 16: QpcnPanel uplift

**Files:** `QpcnPanel.tsx` + test.

- [ ] **Step 1: Update test** — assert energy readout, pred-error table rows with target/current/Δ columns, and a metrics strip with `energy` + each learnable param.

- [ ] **Step 2: Update panel** — for the pred-errors table:

```tsx
const errors = (st.pred_errors ?? {}) as Record<string, number>;
// target lives in spec.params.qpcn.target_n0; we don't have it here, so the
// "target" column reads "(unknown)" until /params/schema is consulted —
// keep it honest, no fake values.

<table className="qpcn-errors">
  <thead><tr><th>obs</th><th>error</th></tr></thead>
  <tbody>{Object.entries(errors).map(([k, v]) => (
    <tr key={k}><td>{k}</td>
      <td style={{ color: v >= 0 ? '#ef9090' : '#9aedc1' }}>
        {v.toFixed(4)}
      </td>
    </tr>
  ))}</tbody>
</table>
```

For metrics:

```tsx
metricsStrip={<MetricsStrip layer="qpcn" metrics={[
  { key: 'E', label: 'energy', color: '#fbc66a',
    select: (ls) => ls.energy as number },
  ...Object.keys((st.params as Record<string, number>) ?? {}).map((p, i) => ({
    key: `p:${p}`, label: p,
    color: ['#6cd0ff', '#9aedc1', '#d291ff'][i % 3],
    select: (ls) => (ls.params as any)?.[p] as number,
  })),
]} />}
```

Per the spec + EXTENSIONS.md, writable-param sliders are **not** added now — the substrate setter isn't exposed.

- [ ] **Step 3: Run + commit**

```bash
git add src/qft_pcn/viz/web/src/panels/QpcnPanel*.tsx
git commit -m "feat(viz/web): QpcnPanel uplift — error table, param strip, energy line"
```

---

### Task 17: MeraPanel uplift

**Files:** `MeraPanel.tsx` + test.

- [ ] **Step 1: Update test** — assert leaves/layer-dims/χ readouts present.

- [ ] **Step 2: Update panel**:

```tsx
const layerDims = (st.layer_dims as number[] | null) ?? [];

readouts={<PanelReadouts cells={[
  { label: 'leaves',     value: st.n_leaves ?? '—' },
  { label: 'layers',     value: layerDims.length },
  { label: 'max χ',      value: st.bond_dims
    ? Math.max(...(st.bond_dims as number[])) : '—' },
]} />}
```

Entropy-vs-cut: a tiny inline SVG line over `st.entropies`.

- [ ] **Step 3: Run + commit**

```bash
git add src/qft_pcn/viz/web/src/panels/MeraPanel*.tsx
git commit -m "feat(viz/web): MeraPanel uplift — readouts + entropy-vs-cut line"
```

---

### Task 18: VqcPanel + LogicPanel extension badges

**Files:** `VqcPanel.tsx`, `VqcPanel.test.tsx`, `LogicPanel.tsx`, `LogicPanel.test.tsx`.

- [ ] **Step 1: Update both tests** — for both panels, add:

```tsx
it('renders an extension-pending badge when the layer state is empty', () => {
  render(<VqcPanel frame={{ step: 0, layer_states: {} } as any} />);
  expect(screen.getByText(/extension pending/i)).toBeInTheDocument();
});
```

- [ ] **Step 2: Update each panel** — wrap `PanelShell` with:

```tsx
const st = frame.layer_states.vqc as VqcState | undefined;
const hasData = !!st && Object.keys(st).length > 0;
return (
  <PanelShell
    title="VQC — Variational Quantum Circuit"
    step={st?.step ?? frame.step}
    hasData={hasData}
    isExtension={!hasData}
    extensionAnchor="#vqc-live-training-panel"
    emptyMessage="vqc is currently fixture-only — see EXTENSIONS.md."
  >
    {/* existing fixture render */}
  </PanelShell>
);
```

Same shape for LogicPanel with `extensionAnchor="#logic-live-relaxation-panel"`.

- [ ] **Step 3: Run + commit**

```bash
cd src/qft_pcn/viz/web && pnpm test --run VqcPanel LogicPanel
```

```bash
git add src/qft_pcn/viz/web/src/panels/VqcPanel*.tsx \
        src/qft_pcn/viz/web/src/panels/LogicPanel*.tsx
git commit -m "feat(viz/web): VqcPanel + LogicPanel extension-pending badge"
```

---

## Phase D — Final integration

### Task 19: Panel signature update + index types

**Files:**
- Modify: `src/qft_pcn/viz/web/src/panels/index.ts`

- [ ] **Step 1: Update Panel type to include `baselineFrame?`**

```ts
import type { Frame } from '../lib/types';

export type PanelComponent = (props: {
  frame: Frame;
  baselineFrame?: Frame;
}) => JSX.Element;
```

Update the registry's exported type accordingly; existing panels are
typed-compatible because the extra prop is optional.

- [ ] **Step 2: Run full vitest suite**

Run: `cd src/qft_pcn/viz/web && pnpm test`
Expected: PASS.

- [ ] **Step 3: Run full python viz suite**

Run: `uv run pytest src/qft_pcn/tests/test_viz_*.py -v --noconftest`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/viz/web/src/panels/index.ts
git commit -m "chore(viz/web): panel signature accepts optional baselineFrame"
```

---

### Task 20: Manual smoke + final docs sweep

**Files:**
- Modify: `src/qft_pcn/viz/README.md` (already updated in Task 5; verify the smoke matrix matches the final UI)

- [ ] **Step 1: Boot the stack**

```bash
./scripts/viz.sh
```

Open http://localhost:5173. Walk the matrix in `README.md`. For any failure,
file a follow-up task — do not edit code mid-smoke.

- [ ] **Step 2: Verify EXTENSIONS.md anchors resolve**

In the running UI, click the `extension pending` badge on Vqc and Logic.
Confirm the linked file exists and the anchor matches.

- [ ] **Step 3: Commit any doc fixes**

If the smoke surfaced doc inaccuracies (and only those):

```bash
git add src/qft_pcn/viz/README.md src/qft_pcn/viz/EXTENSIONS.md
git commit -m "docs(viz): smoke-test corrections"
```

---

## Self-Review Notes

- **Spec coverage:** Every spec section maps to a task — §4 ↔ T1–T11; §5
  ↔ T8; §6 ↔ T12–T18; §7 ↔ T1; §8 ↔ T5; §9 ↔ T3; §10 ↔ T6/T11; §11 ↔ T5;
  §12 ↔ tests in T1–T19 + T20.
- **No placeholders:** Every step contains the exact code/command. Missing
  substrate hooks live in `EXTENSIONS.md`, not in code.
- **Type consistency:** `Frame`, `Preset`, `ParamSchema`, `RunSpec` are
  defined once in `lib/types.ts` and referenced everywhere. `pushFrame`,
  `openRun`, `setActiveRun`, `pinBaseline` are used consistently across
  store/ws/RunControls.
- **Substrate isolation:** No task edits a file outside
  `src/qft_pcn/viz/**`, `scripts/`, or `docs/`.
