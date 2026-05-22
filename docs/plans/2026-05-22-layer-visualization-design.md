# QFT-PCN Layer Visualization — Design

**Date:** 2026-05-22
**Status:** Validated design, ready for implementation planning
**Branch:** `claude/qft-pcn-hybrid-architecture-ihCIR`

## Goal

An interactive, browser-based visualizer for the QFT-PCN hybrid architecture
(`src/qft_pcn/`). Every substrate layer is rendered with the representation
that best matches its mathematics — a 3D field where a field is the right
object, a tensor-network diagram where a tensor network is, a hyperbolic tree
for MERA, and so on. A Manim export pipeline produces presentation-quality
MP4s from the same recorded data.

## Decisions (from brainstorming)

- **Viewing experience:** interactive web app (rotate/zoom/scrub), not pre-baked
  video. Manim's *aesthetic* (dark background, LaTeX labels) is borrowed.
- **Data flow:** live FastAPI + WebSocket server streams simulation state to
  the browser as a run evolves.
- **Scope:** all substrate layers, plus any other useful surface (Hamiltonian
  term view).
- **Frontend approach:** Approach C — React + Vite app with the best-fit tool
  per panel (Three.js / D3 / Plotly), **plus** a Manim MP4 export pipeline.

## Section A — Architecture, instrumentation & data schema

New backend package `src/qft_pcn/viz/`:

```
qft_pcn simulation (Python, unchanged behavior)
   │   Recorder hook captures read-only snapshots per step
   ▼
FastAPI app  (viz/server.py)
   ├─ POST /run      start a simulation run (which layers, params)
   ├─ WS   /ws       stream Frame JSON, one per step
   └─ POST /export   enqueue a Manim render job → job id
   ▼
React + Vite SPA  (viz/web/)
   ├─ WebSocket client → Zustand store (ring buffer + timeline scrubber)
   ├─ Layer selector
   └─ Panels: Three.js / D3 / Plotly per layer
```

**Instrumentation — non-invasive and optional.** Each substrate object gets a
small read-only `snapshot()` method; a `Recorder` collects them. The simulation
runs identically whether or not a recorder is attached (`recorder=None` is a
no-op). Files: `viz/recorder.py`, `viz/schema.py`.

**Frame schema** (one per simulation step; layers populated only if active):

```
Frame { step, layer_states: {
  manifold:   { metric_h, ricci, fields:{phi,E,Pi}, free_energy }
  multifield: { species:[surfaces], couplings:{(i,j):g} }
  mps:        { bond_dims[], entropies[], occupations[][] }
  hamiltonian:{ term_energies{} }
  qpcn:       { energy, pred_errors[], params{} }
  mera:       { nodes[], edges[], bond_dims[], depth }
  vqc:        { circuit_params{}, z_expectations[] }
  logic:      { ast_nodes[], site_map, binder_arcs[] }
}}
```

The same `Frame` stream feeds both the live browser and the Manim exporter.

## Section B — Per-layer visualization specs

1. **Classical PCN + manifold — Three.js.** 3D grid mesh warped by metric
   perturbation `h_μν` (color = Ricci scalar `R`); `Phi/E/Pi` as toggleable 3D
   height-surfaces; free-energy descent as a Plotly inset.
2. **Multi-field coupling — Three.js + D3.** Per-species surfaces on the shared
   manifold; D3 force-directed graph with edge thickness/color = coupling `g_ij`,
   animating as correlated fields pull together.
3. **Quantum substrate (MPS) — D3 + Plotly + Three.js.** D3 tensor-network
   diagram, bond thickness ∝ live `χ`; Plotly entanglement-entropy curve with
   `log χ` ceiling; per-site Fock occupation as Three.js columns.
4. **Hamiltonian — D3.** Heatmap of local + bond operator terms colored by
   per-term energy; doubles as the constraint-debugger view.
5. **QPCN training — Plotly.** Streaming charts: energy descent, prediction
   error per observable, parameter trajectories.
6. **MERA — Three.js.** Hierarchical binary tree on a Poincaré disk; disentanglers
   vs isometries as distinct glyphs; depth = RG layer; edge thickness = bond dim.
7. **Qiskit VQC — D3 + Three.js.** D3 circuit diagram with live angles + Three.js
   Bloch spheres per qubit with Z-expectation readout.
8. **Logic / AST encoding — D3 + Three.js.** D3 AST tree above the MPS site
   chain; arcs from variable-use sites to binder sites, opacity ∝ bond
   entanglement ("binding = entanglement").

Styling: dark background, KaTeX math labels across all panels.

## Section C — Manim export, testing & build

**Manim export.** `viz/manim/` holds one `Scene` subclass per layer, each
consuming a recorded `Frame` sequence and rendering an MP4. `POST /export`
enqueues a FastAPI background render task, returns a job id; the frontend polls
and offers download. Manim is an optional dependency group (`.[viz-manim]`).

**Testing.**
- Python: pytest for `recorder`/`schema` (snapshot determinism, JSON round-trip,
  "recorder absent ⇒ identical output"); FastAPI `TestClient` for `/run`,
  `/export`; WebSocket frame-format test.
- Frontend: Vitest for WS client + store; smoke-mount test per panel with a
  fixture `Frame`.
- Manim: one fast low-res 2-frame smoke render per scene in CI.

**Build / run.**
- Backend: `uvicorn qft_pcn.viz.server:app`; deps `fastapi`, `uvicorn`,
  `websockets` (+ optional `manim`).
- Frontend: Vite + `pnpm`, dev server proxied to backend.
- Single `viz` script launches both.

**YAGNI:** no auth, no database/persistence (runs ephemeral), no Playwright e2e
initially. Manim export is built but optional.
