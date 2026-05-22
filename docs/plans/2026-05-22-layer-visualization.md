# QFT-PCN Layer Visualization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers-extended-cc:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive browser visualizer for every substrate layer of the QFT-PCN architecture, fed by a live FastAPI+WebSocket stream, plus a Manim MP4 export pipeline.

**Architecture:** A new `src/qft_pcn/viz/` package. Non-invasive instrumentation: free extractor functions read public attributes of existing substrate objects (`QFTPCNNetwork`, `QPCN`, `MERA`, etc.) — no existing file is modified, so simulation behavior is provably unchanged. A `Recorder` collects per-step `Frame`s; a FastAPI app runs simulations and streams `Frame` JSON over a WebSocket. A React+Vite SPA renders one best-fit panel per layer (Three.js / D3 / Plotly). The same recorded `Frame` sequence drives Manim scenes for MP4 export.

**Tech Stack:** Python 3.11, FastAPI, uvicorn, websockets, NumPy (existing), Manim (optional group); React 18, Vite, TypeScript, Zustand, three / @react-three/fiber, d3, plotly.js, KaTeX, Vitest, pnpm.

**Conventions:**
- Package is imported as `src.qft_pcn.viz` (tests use `from src.qft_pcn... import`).
- Run tests from repo root: `python -m pytest src/qft_pcn/tests/ -v`.
- Frontend lives in `src/qft_pcn/viz/web/`; commands run from that dir.
- Commit after every passing step group. Use `feat(viz): ...` / `test(viz): ...`.

---

## Task 1: Viz instrumentation — schema, snapshots, recorder

**Files:**
- Create: `src/qft_pcn/viz/__init__.py`
- Create: `src/qft_pcn/viz/schema.py`
- Create: `src/qft_pcn/viz/snapshots.py`
- Create: `src/qft_pcn/viz/recorder.py`
- Test: `src/qft_pcn/tests/test_viz_recorder.py`

### Step 1.1 — Write failing test for the Frame schema

In `test_viz_recorder.py`:

```python
"""Tests for src/qft_pcn/viz instrumentation."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.viz.schema import Frame


def test_frame_json_roundtrip():
    f = Frame(step=3, layer_states={"manifold": {"free_energy": 1.5}})
    blob = f.to_json()
    back = Frame.from_json(blob)
    assert back.step == 3
    assert back.layer_states["manifold"]["free_energy"] == 1.5


def test_frame_serializes_numpy_arrays():
    f = Frame(step=0, layer_states={"mps": {"bond_dims": np.array([1, 4, 4, 1])}})
    blob = f.to_json()
    assert "1" in blob and "4" in blob  # arrays became JSON lists
```

### Step 1.2 — Run, verify failure

Run: `python -m pytest src/qft_pcn/tests/test_viz_recorder.py -v`
Expected: FAIL — `ModuleNotFoundError: src.qft_pcn.viz.schema`.

### Step 1.3 — Implement `schema.py`

`Frame` is a dataclass with `step: int` and `layer_states: dict[str, dict]`.
`to_json()` uses a custom encoder that converts `np.ndarray` → `.tolist()`,
`np.integer/np.floating` → Python scalar, and `complex` → `{"re":..,"im":..}`.
`from_json()` is `json.loads` into the dataclass (arrays stay as plain lists —
the browser wants lists anyway). Add `LAYER_KEYS` constant listing the eight
layer names: `manifold, multifield, mps, hamiltonian, qpcn, mera, vqc, logic`.

Create `viz/__init__.py` (empty, or re-export `Frame`, `Recorder`).

### Step 1.4 — Run schema tests, verify pass

Run: `python -m pytest src/qft_pcn/tests/test_viz_recorder.py -v`
Expected: PASS (2 passed).

### Step 1.5 — Write failing test for snapshot extractors

Append to `test_viz_recorder.py`:

```python
def test_snapshot_network_is_readonly():
    from src.qft_pcn.network import QFTPCNNetwork, NetworkConfig
    from src.qft_pcn.layer import LayerConfig
    from src.qft_pcn.viz.snapshots import snapshot_network
    net = QFTPCNNetwork(8, 8, NetworkConfig(layers=[LayerConfig(channels=1)]))
    obs = np.zeros((1, 8, 8))
    net.step(obs)
    before = net.manifold.h.copy()
    snap = snapshot_network(net)
    assert "fields" in snap and "metric_h" in snap
    np.testing.assert_array_equal(net.manifold.h, before)  # unchanged
```

(Check `LayerConfig`'s real field names in `src/qft_pcn/layer.py` and the
manifold's perturbation attribute name in `src/qft_pcn/manifold.py` before
finalizing — adjust `LayerConfig(...)` and `net.manifold.h` accordingly.)

### Step 1.6 — Run, verify failure; then implement `snapshots.py`

Implement free functions, each taking a live object and returning a plain
`dict` of JSON-ready values (lists/floats), reading only public attributes:

- `snapshot_network(net) -> dict` — `metric_h`, `ricci`, per-layer `phi/E/Pi`
  (downsample grids to ≤32×32), `free_energy`.
- `snapshot_multifield(mf) -> dict` — per-species surfaces, `couplings` as a
  list of `{i, j, g}`.
- `snapshot_qpcn(q) -> dict` — `energy`, `pred_errors`, `params`,
  `bond_dims` (`q.state.bond_dimensions()`), `entropies`, `occupations`.
- `snapshot_mps(mps) -> dict` — bond dims, per-bond entanglement entropy.
- `snapshot_hamiltonian(H) -> dict` — per-term energies (use existing
  `local_op`/`bond_op` expectations).
- `snapshot_mera(m) -> dict`, `snapshot_vqc(vqc) -> dict`,
  `snapshot_logic(enc) -> dict` — nodes/edges/site-map per design Section B.

Each extractor is defensive: wrap attribute access so a missing optional piece
yields `None` rather than raising. Keep them pure reads — never call mutating
methods.

### Step 1.7 — Run snapshot test, verify pass

### Step 1.8 — Write failing test for `Recorder`

```python
def test_recorder_captures_frames_without_changing_sim():
    from src.qft_pcn.network import QFTPCNNetwork, NetworkConfig
    from src.qft_pcn.layer import LayerConfig
    from src.qft_pcn.viz.recorder import Recorder
    cfg = NetworkConfig(layers=[LayerConfig(channels=1)])
    net = QFTPCNNetwork(8, 8, cfg)
    obs = np.zeros((1, 8, 8))
    rec = Recorder()
    for _ in range(5):
        net.step(obs)
        rec.capture_network(net)
    assert len(rec.frames) == 5
    assert rec.frames[-1].step == 4
    # Frames are JSON-serializable
    rec.frames[0].to_json()
```

### Step 1.9 — Implement `recorder.py`

`Recorder` holds `frames: list[Frame]` and an internal step counter.
Methods `capture_network(net)`, `capture_qpcn(q)`, `capture_mera(m)`, etc.,
each calls the matching `snapshot_*` and appends a `Frame` with the right
`layer_states` key. A generic `capture(**layer_snaps)` merges multiple layers
into one `Frame` (for runs that exercise several substrates at once).
Add `clear()` and `to_json_list() -> str`.

### Step 1.10 — Run all viz recorder tests, verify pass

Run: `python -m pytest src/qft_pcn/tests/test_viz_recorder.py -v`
Expected: PASS (all).

### Step 1.11 — Add dependencies & commit

In `pyproject.toml`, add an optional group:

```toml
[project.optional-dependencies]
viz = ["fastapi>=0.110", "uvicorn[standard]>=0.29", "websockets>=12"]
viz-manim = ["manim>=0.18"]
```

Run: `git add src/qft_pcn/viz/ src/qft_pcn/tests/test_viz_recorder.py pyproject.toml`
Commit: `feat(viz): non-invasive Frame recorder + per-layer snapshot extractors`

---

## Task 2: FastAPI + WebSocket server

**Blocked by:** Task 1.

**Files:**
- Create: `src/qft_pcn/viz/runs.py` — run registry + simulation drivers
- Create: `src/qft_pcn/viz/server.py` — FastAPI app
- Test: `src/qft_pcn/tests/test_viz_server.py`

### Step 2.1 — Write failing test for `/run` and `/health`

```python
"""Tests for the viz FastAPI server."""
from fastapi.testclient import TestClient
from src.qft_pcn.viz.server import app


def test_health():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"


def test_run_returns_run_id():
    c = TestClient(app)
    r = c.post("/run", json={"layers": ["manifold"], "steps": 5, "grid": 8})
    assert r.status_code == 200
    assert "run_id" in r.json()
```

### Step 2.2 — Run, verify failure (no `server` module)

### Step 2.3 — Implement `runs.py`

A `RunSpec` dataclass (`layers`, `steps`, `grid`, optional `seed`, params).
`run_simulation(spec) -> Iterator[Frame]`: a generator that builds the
requested substrate(s), steps them `spec.steps` times, and `yield`s a `Frame`
per step via the `Recorder` snapshot functions from Task 1. Branch on which
layers are requested (`manifold` → `QFTPCNNetwork`; `multifield` →
`MultiFieldNetwork` from `src/qft_pcn/multifield.py`, so `snapshot_multifield`
reads a real coupled-field substrate; `mps/qpcn/hamiltonian` → `QPCN`;
`mera`/`vqc`/`logic` → their constructors). Keep grids
small (≤16) and steps modest by default for responsiveness.
A `RunRegistry` dict maps `run_id` (uuid4 hex) → `RunSpec`.

### Step 2.4 — Implement `server.py`

```python
"""FastAPI app streaming QFT-PCN layer state to the browser."""
from __future__ import annotations
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .runs import RunSpec, RunRegistry, run_simulation

app = FastAPI(title="QFT-PCN Visualizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
_registry = RunRegistry()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run")
def start_run(spec: dict):
    run_id = uuid.uuid4().hex
    _registry.add(run_id, RunSpec.from_dict(spec))
    return {"run_id": run_id}


@app.websocket("/ws/{run_id}")
async def stream(ws: WebSocket, run_id: str):
    await ws.accept()
    spec = _registry.get(run_id)
    if spec is None:
        await ws.close(code=4004)
        return
    try:
        for frame in run_simulation(spec):
            await ws.send_text(frame.to_json())
        await ws.send_text('{"done": true}')
    except WebSocketDisconnect:
        pass
```

### Step 2.5 — Add WebSocket frame-format test

```python
def test_ws_streams_frames():
    c = TestClient(app)
    run_id = c.post("/run", json={"layers": ["manifold"],
                                  "steps": 3, "grid": 8}).json()["run_id"]
    with c.websocket_connect(f"/ws/{run_id}") as ws:
        first = ws.receive_json()
        assert first["step"] == 0
        assert "manifold" in first["layer_states"]
```

### Step 2.6 — Run all server tests, verify pass

Run: `python -m pytest src/qft_pcn/tests/test_viz_server.py -v`
Expected: PASS.

### Step 2.7 — Add `/export` stub endpoint

Add `POST /export` accepting `{run_id}` and returning `{job_id, status:"queued"}`.
The real render is wired in Task 5; for now register the job in an in-memory
dict with status `queued`. Add `GET /export/{job_id}` returning the job status.
Add a test asserting `/export` returns a `job_id`.

### Step 2.8 — Commit

Run: `git add src/qft_pcn/viz/runs.py src/qft_pcn/viz/server.py src/qft_pcn/tests/test_viz_server.py`
Commit: `feat(viz): FastAPI run registry, WebSocket frame stream, export stub`

---

## Task 3: React + Vite app shell

**Blocked by:** Task 1 (needs the `Frame` shape).

**Files:**
- Create: `src/qft_pcn/viz/web/` — Vite project (`package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`)
- Create: `src/qft_pcn/viz/web/src/main.tsx`, `App.tsx`
- Create: `src/qft_pcn/viz/web/src/lib/types.ts` — `Frame` TS type mirroring `schema.py`
- Create: `src/qft_pcn/viz/web/src/lib/ws.ts` — WebSocket client
- Create: `src/qft_pcn/viz/web/src/store.ts` — Zustand store
- Create: `src/qft_pcn/viz/web/src/components/LayerSelector.tsx`, `Timeline.tsx`
- Test: `src/qft_pcn/viz/web/src/store.test.ts`, `lib/ws.test.ts`

### Step 3.1 — Scaffold the Vite project

From `src/qft_pcn/viz/`:

```bash
pnpm create vite@latest web --template react-ts
cd web
pnpm install
pnpm add zustand three @react-three/fiber @react-three/drei d3 plotly.js-dist-min katex
pnpm add -D vitest @testing-library/react @testing-library/jest-dom jsdom @types/three @types/d3
```

In `vite.config.ts` add a dev proxy: `/run` and `/ws` → `http://localhost:8000`,
and a `test` block (`environment: 'jsdom'`, `globals: true`).

### Step 3.2 — Write failing store test

```typescript
import { describe, it, expect } from 'vitest';
import { useVizStore } from './store';

describe('viz store', () => {
  it('appends frames and scrubs', () => {
    const s = useVizStore.getState();
    s.reset();
    s.pushFrame({ step: 0, layer_states: {} });
    s.pushFrame({ step: 1, layer_states: {} });
    expect(useVizStore.getState().frames.length).toBe(2);
    useVizStore.getState().setCursor(0);
    expect(useVizStore.getState().currentFrame()?.step).toBe(0);
  });
});
```

### Step 3.3 — Run, verify failure

Run (from `web/`): `pnpm vitest run src/store.test.ts`
Expected: FAIL — no `store` module.

### Step 3.4 — Implement `store.ts`

Zustand store: `frames: Frame[]` (ring buffer, cap ~2000), `cursor: number`,
`live: boolean`, `selectedLayer: string`. Actions: `pushFrame`, `setCursor`,
`reset`, `selectLayer`, `currentFrame()` (returns `frames[cursor]`). When
`live`, `pushFrame` advances `cursor` to the new tail.

### Step 3.5 — Run store test, verify pass

### Step 3.6 — Implement `lib/types.ts` and `lib/ws.ts`

`types.ts`: `Frame` interface mirroring `schema.py` `LAYER_KEYS`.
`ws.ts`: `connectRun(spec)` → `POST /run`, then opens `ws://.../ws/{run_id}`,
parses each message, calls `pushFrame`; on `{done:true}` sets `live=false`.
Add a `ws.test.ts` that mocks `WebSocket` and asserts a parsed frame reaches
the store.

### Step 3.7 — Implement `App.tsx`, `LayerSelector`, `Timeline`

`App` lays out: a left `LayerSelector` (the eight layers), a top run-control
bar (`POST /run` form), a central panel area (renders the panel for
`selectedLayer` — Task 4 fills these in; for now a placeholder), and a bottom
`Timeline` scrubber bound to `cursor`. Dark theme; load KaTeX CSS globally.

### Step 3.8 — Run all frontend tests, verify pass

Run (from `web/`): `pnpm vitest run`
Expected: PASS.

### Step 3.9 — Commit

Run: `git add src/qft_pcn/viz/web/` (ensure `web/node_modules` is gitignored)
Commit: `feat(viz): React+Vite app shell — WS client, store, layer selector`

---

## Task 4: Per-layer visualization panels

**Blocked by:** Task 2 and Task 3.

**Files (one component + one smoke test each):**
- `web/src/panels/ManifoldPanel.tsx` — Three.js warped grid + Phi/E/Pi surfaces
- `web/src/panels/MultifieldPanel.tsx` — Three.js surfaces + D3 coupling graph
- `web/src/panels/MpsPanel.tsx` — D3 tensor-network diagram + Plotly entropy curve
- `web/src/panels/HamiltonianPanel.tsx` — D3 term-energy heatmap
- `web/src/panels/QpcnPanel.tsx` — Plotly streaming charts
- `web/src/panels/MeraPanel.tsx` — Three.js Poincaré-disk tree
- `web/src/panels/VqcPanel.tsx` — D3 circuit + Three.js Bloch spheres
- `web/src/panels/LogicPanel.tsx` — D3 AST tree + binder-entanglement arcs
- `web/src/panels/__fixtures__/frames.ts` — sample `Frame`s for tests
- `web/src/panels/*.test.tsx` — smoke-mount per panel

### Step 4.0 — Create fixtures and a panel registry

`__fixtures__/frames.ts` exports one realistic `Frame` per layer (hand-written
JSON matching `schema.py`). Add `panels/index.ts` mapping layer name → panel
component, consumed by `App.tsx`.

### Step 4.1–4.8 — For EACH panel, repeat this micro-cycle:

1. **Write failing smoke test** (`<Panel>.test.tsx`):
   ```tsx
   import { render } from '@testing-library/react';
   import { ManifoldPanel } from './ManifoldPanel';
   import { manifoldFrame } from './__fixtures__/frames';
   it('mounts with a frame', () => {
     render(<ManifoldPanel frame={manifoldFrame} />);
   });
   ```
2. **Run, verify failure** — `pnpm vitest run src/panels/ManifoldPanel.test.tsx`.
3. **Implement the panel** per design Section B (see spec table below). Each
   panel takes a single `frame: Frame` prop and reads only its own
   `layer_states[key]`. Resize-aware; dark background; KaTeX labels.
4. **Run, verify pass.**
5. **Commit** — `feat(viz): <layer> panel`.

**Panel spec (design Section B):**

| Panel | Library | What it draws |
|---|---|---|
| Manifold | three / r3f | Grid mesh, vertex z + color from `metric_h`/`ricci`; toggleable Phi/E/Pi height-surfaces; Plotly free-energy inset |
| Multifield | three + d3 | Per-species surfaces; D3 force graph, edge width/color = `g_ij` |
| MPS | d3 + plotly | Tensor-node chain, bond width ∝ `bond_dims`; entropy curve with `log χ` ceiling; Fock-occupation columns |
| Hamiltonian | d3 | Term matrix heatmap colored by per-term energy |
| QPCN | plotly | Streaming energy descent, per-observable error, param trajectories |
| MERA | three / r3f | Binary tree on Poincaré disk; disentangler vs isometry glyphs; edge width = bond dim |
| VQC | d3 + three | Circuit diagram with live angles; Bloch spheres + Z-readout |
| Logic | d3 + three | AST tree over MPS site chain; binder arcs, opacity ∝ entanglement |

Note: r3f WebGL does not render under jsdom — smoke tests assert the component
mounts without throwing (wrap WebGL-only subtrees so they no-op when
`canvas` is unavailable, or mock `@react-three/fiber`'s `Canvas`).

### Step 4.9 — Wire panels into `App.tsx`, manual end-to-end check

Start backend (`uvicorn src.qft_pcn.viz.server:app --reload`) and frontend
(`pnpm dev`); start a run covering all layers; confirm each panel renders and
the timeline scrubs. Commit: `feat(viz): wire all panels into app shell`.

---

## Task 5: Manim export pipeline

**Blocked by:** Task 1 (needs `Frame`).

**Files:**
- Create: `src/qft_pcn/viz/manim/__init__.py`
- Create: `src/qft_pcn/viz/manim/scenes.py` — one `Scene` subclass per layer
- Create: `src/qft_pcn/viz/manim/render.py` — frame-sequence → MP4 driver
- Modify: `src/qft_pcn/viz/server.py` — wire `/export` to a background render job
- Test: `src/qft_pcn/tests/test_viz_manim.py`

### Step 5.1 — Write failing test for the render driver

```python
import pytest
manim = pytest.importorskip("manim")
from src.qft_pcn.viz.manim.render import render_layer


def test_render_produces_mp4(tmp_path):
    frames = [{"step": i, "layer_states": {"qpcn": {"energy": 1.0 - i * 0.1}}}
              for i in range(3)]
    out = render_layer("qpcn", frames, tmp_path, quality="low")
    assert out.exists() and out.suffix == ".mp4"
```

### Step 5.2 — Run, verify failure

### Step 5.3 — Implement `scenes.py` and `render.py`

`scenes.py`: a `LayerScene` base that accepts a frame list; subclasses
`QpcnScene`, `ManifoldScene`, etc., animating the recorded data (start with
`QpcnScene` plotting the energy curve — it is the simplest and proves the
pipeline). `render.py::render_layer(layer, frames, out_dir, quality)` selects
the scene class, configures Manim for low quality in tests, renders, and
returns the MP4 `Path`.

### Step 5.4 — Run, verify pass

Run: `python -m pytest src/qft_pcn/tests/test_viz_manim.py -v`
Expected: PASS if `manim` installed, else SKIPPED.

### Step 5.5 — Wire `/export` to a background job

In `server.py`, make `/export` capture the run's recorded frames and schedule
`render_layer` via FastAPI `BackgroundTasks`, updating the job dict to
`running` → `done` (with file path) or `error`. `GET /export/{job_id}` returns
status; add `GET /export/{job_id}/download` serving the MP4 via `FileResponse`.

### Step 5.6 — Add a server test for the export flow

Mark it `importorskip("manim")`; assert a job reaches `done` and the file
exists.

### Step 5.7 — Commit

Run: `git add src/qft_pcn/viz/manim/ src/qft_pcn/viz/server.py src/qft_pcn/tests/test_viz_manim.py`
Commit: `feat(viz): Manim export pipeline + /export background render job`

### Step 5.8 — Add a `viz` launch script and README

Create `src/qft_pcn/viz/README.md` documenting:
`pip install -e .[viz]`, `uvicorn src.qft_pcn.viz.server:app`, `pnpm dev`,
and optional `pip install -e .[viz-manim]`. Add a `scripts/viz` (or
`Makefile` target) that launches backend + frontend together.
Commit: `docs(viz): launch script and usage README`

---

## Final verification

- `python -m pytest src/qft_pcn/tests/ -v` — all suites green (existing + viz).
- `cd src/qft_pcn/viz/web && pnpm vitest run` — frontend green.
- Manual: run the full demo, scrub every panel, export one Manim MP4.
- Confirm no existing `src/qft_pcn/` file outside `viz/` was modified.
