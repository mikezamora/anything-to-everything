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

## PCN-side smoke

| layer | fixture | preset | advanced |
|---|---|---|---|
| pcn-fields    | ✓ | ✓ | n/a |
| pcn-dynamics  | ✓ | ✓ | n/a |
| pcn-coupling  | ✓ | ✓ | n/a |

## Narrative sections

- Click each section header (QFT Substrate / PCN Substrate / QPCN Fusion);
  confirm the matching `Intro*` panel renders with the section's tagline +
  layer summaries.
- Collapse / expand each section using the ▾/▸ toggle.

## DSL route

- Switch routes via the top-bar `Viz / DSL` toggle.
- Confirm the Ollama model picker populates from `ollama list`.
- Chat mode: enter a prompt → DSL appears in the editor → click Run in the
  right pane → frames stream → click Verbalize → assistant turn appears.
- Stepped mode: toggle "stepped-flow mode" → use the three numbered
  buttons in order; confirm each is gated on the previous step.
- Import: paste a DSL JSON into the editor → click Run.
- Export: pin a run as baseline, click "Export DSL" in the right pane,
  confirm a `.dsl.json` downloads with the expected structure.


## Running with Ollama on a Windows host (WSL viz server)

Ollama listens on the Windows host; WSL's `localhost` does NOT reach it. Use
the WSL→Windows gateway IP. The helper script computes it for you:

    OLLAMA_HOST=$(./scripts/ollama-host.sh) ./scripts/viz.sh

Or set it permanently in your shell rc:

    export OLLAMA_HOST=http://$(ip route show | awk '/^default/ {print $3}'):11434

When the DSL route's model picker shows `(no models — is Ollama running?)`,
the most likely cause is that `OLLAMA_HOST` is not set and the server defaulted
to `http://localhost:11434` which never reaches the Windows host.
