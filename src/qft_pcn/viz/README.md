# QFT-PCN Visualizer

A FastAPI backend that streams live QFT-PCN substrate state to a React + Vite
frontend over WebSockets, plus an optional Manim export pipeline that renders a
recorded run to an MP4.

## Architecture

- `schema.py` — `Frame` dataclass (`{step, layer_states}`) + JSON (de)serialization.
- `snapshots.py` — per-substrate `snapshot_*` extractors (NumPy → JSON-ready dicts).
- `recorder.py` — `Recorder` accumulating `Frame`s as a simulation runs.
- `runs.py` — `RunSpec` / `RunRegistry` / `run_simulation` (builds tiny
  substrates and yields one `Frame` per step).
- `server.py` — FastAPI app: `/run`, `/ws/{run_id}`, `/export`.

### Live vs. fixture-only layers

`run_simulation` streams **live** data for 6 layers: `manifold`, `multifield`,
`mps`, `hamiltonian`, `qpcn`, and `mera`. The `vqc` and `logic` panels are
currently **fixture/demo-only** — their substrates are not yet wired into
`run_simulation`, so those panels only render against test fixtures, never
live simulation data. Wiring them up is future work (the plan deliberately
scoped them as minimal/optional).
- `manim/` — `render_layer` + per-layer `Scene` subclasses (optional).
- `web/` — React + Vite frontend.

## Backend

Install the FastAPI dependencies and run the server:

```bash
uv sync --extra viz
uv run uvicorn src.qft_pcn.viz.server:app --reload
```

The backend listens on http://localhost:8000 (`/health` is a liveness probe).

## Frontend

```bash
cd src/qft_pcn/viz/web
pnpm install
pnpm dev
```

The Vite dev server runs on http://localhost:5173 and connects to the backend
WebSocket.

## Manim export (optional)

Rendering a recorded run to an MP4 needs Manim:

```bash
uv sync --extra viz-manim
```

`/export` is an API-only endpoint (there is no UI button) — trigger it with a
`POST /export` request.

With Manim installed, `POST /export {"run_id": ..., "layer": "qpcn"}` schedules
a background render job:

- `GET /export/{job_id}` — poll job status (`queued` → `running` → `done`/`error`).
- `GET /export/{job_id}/download` — download the MP4 once the job is `done`.

If Manim is not installed the server still starts and every non-export endpoint
works; an export job simply finishes with status `error`.

## Launch both at once

`scripts/viz.sh` starts the uvicorn backend and the `pnpm dev` frontend
together (bash / WSL):

```bash
./scripts/viz.sh
```

## Tests

The repo `conftest.py` imports the Unix-only `resource` module, so run the viz
tests with `--noconftest`:

```bash
uv run pytest src/qft_pcn/tests/test_viz_server.py -v --noconftest
uv run pytest src/qft_pcn/tests/test_viz_manim.py -v --noconftest
```

The Manim render tests skip automatically when Manim is not installed.
