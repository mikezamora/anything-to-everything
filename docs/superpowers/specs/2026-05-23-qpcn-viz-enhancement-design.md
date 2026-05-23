# QPCN Visualizer Enhancement — Design

**Status:** Draft (brainstorming approved 2026-05-23)
**Scope:** `src/qft_pcn/viz/**` only. No edits to substrate modules.
**Branch:** `claude/qft-pcn-hybrid-architecture-ihCIR`

## 1. Goal

The QPCN visualizer today renders eight substrate layers but provides almost
no context: there are no descriptions explaining what each layer *is*, the
starting conditions are limited to `layers + steps + grid`, and the only
runtime control is a single "Run" button followed by a passive timeline.

This design enhances the viz so that:

1. **Every panel explains itself.** A side-by-side explainer pane describes
   what the layer is in the QPCN, what each visual element represents, the
   math behind it, and what to watch for as the simulation runs — content
   sourced from `QFT_PCN_ARCHITECTURE.md`.
2. **Starting conditions are first-class.** Curated presets per layer give a
   one-click path to an interesting initial state; an Advanced expander
   exposes every accepted substrate parameter.
3. **General controls match the dynamism of the system.** Play/pause/step,
   playback speed, baseline-comparison, timeline scrubbing, and export
   (JSONL + Manim MP4) are all available from the UI.
4. **Per-panel visualizations surface every signal already in the snapshot.**
   Legends, axis labels, live time-series of key scalars, readouts of active
   parameters, and overlay toggles for unused snapshot fields.

## 2. Non-goals

- Substrate changes. Parallel agents are enhancing the QPCN; this work
  touches `src/qft_pcn/viz/**` and the FastAPI server only.
- New snapshot fields. The spec consumes what `snapshot_*` already exposes;
  any visualization that would require a new substrate hook is deferred to
  `EXTENSIONS.md` rather than stubbed.
- A redesigned schema. `Frame` / `LAYER_KEYS` / `snapshot_*` are stable.
- LLM/DSL integration in the UI.

## 3. Constraints

- **No placeholders.** If a feature depends on a substrate hook that doesn't
  exist yet, the UI omits the affordance entirely (no fake data, no TODO
  comments) and records the gap in `src/qft_pcn/viz/EXTENSIONS.md`.
- **Defensive snapshots only.** The existing `_safe(lambda: …)` pattern is
  preserved so substrate-shape drift from parallel work breaks at most one
  readout cell, never a panel.
- **Determinism.** All presets are seeded; re-running the same preset
  produces an identical frame sequence (`/export` already depends on this).
- **Run size clamps stay.** `_MAX_GRID`, `_MAX_STEPS`, `_QPCN_SITES`,
  `_QPCN_CHI`, `_MERA_LEAVES` continue to bound resource use.

## 4. Architecture

The viz becomes a four-region layout: top run/transport bar, left layer
rail (unchanged), center panel area, right collapsible explainer pane. The
bottom timeline scrubber remains.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Run controls │ Preset ▾ │ Advanced ▸ │ ▶ ⏸ ⏭ │ speed ▬─── │ Export │
├──────┬───────────────────────────────────────────────────┬──────────┤
│      │                                                   │ Manifold │
│ Layers│         [active panel: viz + readouts]            │ ──────── │
│      │                                                   │ What…    │
│      │                                                   │ Math…    │
│      │                                                   │ Watch…   │
├──────┴───────────────────────────────────────────────────┴──────────┤
│  ◀ ─────────────────●─────────────────── ▶   step 42 / 200   ☑ live │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.1 Backend additions (`src/qft_pcn/viz/`)

- `presets.py` — `Preset` dataclass + module-level catalog. Each preset is a
  frozen record `(id, layer, label, description, spec_overrides)`. Presets
  drive **real substrate state** via `spec_overrides` merged into
  `RunSpec.params`; none invent data.
- `runs.py` extension — `RunSpec.params` already exists as a dict; substrate
  builders begin honouring per-layer keys (e.g. `params["qpcn"]["mass"]`)
  with explicit allow-lists and the existing clamps. Unknown keys are
  ignored. No call sites change for callers that don't pass `params`.
- `controller.py` — `RunController` wraps `run_simulation` with an
  `asyncio.Event` (set = running, clear = paused) and a step counter.
  `/runs/{id}/pause`, `/resume`, `/step` flip these. The WS loop awaits the
  event each iteration. Clients that never call lifecycle endpoints see
  today's behaviour.
- `server.py` endpoints (additive):
  - `GET /presets` — preset catalog
  - `GET /params/schema` — per-layer JSON Schema describing accepted params
    (used by the Advanced expander). Authored by hand in `presets.py` and
    kept in sync manually; substrate modules are not edited from this work.
  - `POST /runs/{id}/pause`, `/resume`, `/step` — lifecycle
  - `GET /export/run/{id}` — JSONL dump of recorded frames
  - The existing `POST /export` MP4 endpoint becomes UI-callable
- `EXTENSIONS.md` — deferred-feature register (Section 8)

### 4.2 Frontend additions (`src/qft_pcn/viz/web/src/`)

- `lib/explainer.ts` — typed registry mapping `layer_key → ExplainerSpec`.
  Static content, no network call:

  ```ts
  interface ExplainerSpec {
    title: string;
    oneLine: string;
    what: string[];
    elements: { name: string; meaning: string; code?: string }[];
    math: { tex: string; caption: string }[];
    watch: { label: string; readout?: string }[];
    references?: { label: string; href: string }[];
  }
  ```

- `lib/presets.ts` — fetches `/presets` once, caches in the zustand store.
- `components/ExplainerPane.tsx` — collapsible right rail, KaTeX-rendered
  math, hover on a `watch` item highlights the matching `PanelReadouts`
  cell so prose links to the live number.
- `components/RunControls.tsx` — replaces the inline controls. Adds:
  preset dropdown, Advanced expander (renders the `/params/schema` form),
  transport buttons, playback-speed slider, "Pin as baseline" toggle, an
  Export menu (JSONL / MP4).
- `components/CompareBar.tsx` — shown when a baseline is pinned; lists both
  run ids with swap/unpin actions.
- `store.ts` — `runs: Map<runId, RunState>` (frames + cursor per run),
  `activeRunId`, `baselineRunId`, `playbackSpeed`, `paused`.
- `panels/PanelToolbar.tsx` — shared overlay-toggle strip per panel.
- `panels/PanelReadouts.tsx` — shared `{label, value, unit?, highlightId?}`
  cell grid.
- `panels/MetricsStrip.tsx` — shared mini-time-series strip backed by a
  per-layer client-side scalar buffer (capped at `MAX_FRAMES`).

### 4.3 Data flow

```
preset/advanced form ── POST /run ──▶ RunRegistry ─┐
                                                   │
        WS /ws/{run_id} ◀── RunController.frames ──┤
                                                   │
        store.runs[id].push(frame) ────────────────┘
              │
              ├──▶ active panel (frame)
              ├──▶ baseline panel (if pinned)  → delta render
              └──▶ MetricsStrip scalar buffer  → mini time-series
```

## 5. Explainer content (per layer)

Every layer ships an `ExplainerSpec` sourced from `QFT_PCN_ARCHITECTURE.md`:

- **manifold** → §2.1, §3.1 (dynamic Riemannian metric; Laplace-Beltrami)
- **multifield** → §2.2 (Yukawa-style coupling, learnable `g_ij`)
- **mps** → §2.3 (entanglement carrier, bond dimension)
- **hamiltonian** → §2.3, §3.x (one/two-site terms, species)
- **qpcn** → §2.3 (belief = MPS, generative model = H, errors drive params)
- **mera** → MERA section (tree of isometries / disentanglers)
- **vqc** → §2.4 (parameterized circuit, parameter-shift gradients)
- **logic** → logic-as-Hamiltonian sections (`EvalHamiltonian` term layout)

For each, the spec authors: `what` (2-3 short paragraphs), `elements`
(table mapping every on-screen feature to its substrate meaning), `math`
(2-4 KaTeX formulas with one-line captions), `watch` (3-5 bullet hints
keyed to live readouts).

## 6. Per-panel uplift

Each panel keeps its current centerpiece visualization and adds: a toolbar
of overlay toggles, a readouts strip, and (where there's a useful scalar)
a metrics strip. Specifics, driven by what each `snapshot_*` already
exposes:

| Panel | Current centerpiece | Added |
|---|---|---|
| manifold | warped grid + phi/E/Pi overlays | axis labels, Ricci legend, readouts (mean|Ricci|, max curvature, layer count), metrics strip on `mean_abs_ricci`, channel selector |
| multifield | 3D surfaces + coupling graph | per-field active toggle, coupling-matrix heatmap when ≥3 fields, readouts (mean_abs_coupling, per-field ‖Φ‖₂ and ‖E‖₂), metrics strip per coupling |
| mps | bond-dim bars + entropy | local-dim & N readout, page-curve reference overlay, hover-to-highlight bond cut, metrics strip on total entropy |
| hamiltonian | term/param view | per-species occupation legend, `curvature` mini-map (snapshot field currently unused), KaTeX of H with present terms highlighted |
| qpcn | energy + bond dims | pred-errors table with target/current/Δ colouring, parameter sliders **only if** `/params/schema` reports them writable, metrics strip on energy + learnable params |
| mera | tree | per-layer bond-dim readout, entropy-vs-cut line, click-to-highlight subtree |
| vqc | theta heatmap + bias (fixture) | extension badge until substrate returns to `runs.py` |
| logic | term list + residuals (fixture) | extension badge until substrate returns to `runs.py` |

Per-panel test files gain "renders with extension badge when snapshot is
empty" cases for vqc/logic.

## 7. Presets

Initial catalog (real substrate state only; none invent data):

- `manifold.flat`, `manifold.hot-spot`, `manifold.two-source`
- `multifield.uncoupled`, `multifield.symmetric-coupling`,
  `multifield.learn-coupling`
- `qpcn.ground-state-relax`, `qpcn.quarter-density-target`,
  `qpcn.high-mass`
- `mps.product-state`, `mps.entangled-warmup`
- `hamiltonian.single-species`, `hamiltonian.two-species`
- `mera.vacuum-small`

vqc/logic presets are intentionally absent — they are recorded in
`EXTENSIONS.md` and surface once the parallel substrate work re-adds
`_build_vqc` / `_build_logic` to `runs.py`.

## 8. EXTENSIONS.md (deferred-feature register)

A new `src/qft_pcn/viz/EXTENSIONS.md` documents every viz affordance that a
substrate hook does not currently support. Each entry has three lines:
**what's needed**, **why it's deferred**, **what to wire when the hook
lands**. The spec ships with these entries pre-populated:

- **vqc live panel** — `runs.py` no longer builds a `QuantumGenerativeMap`
  and `snapshots.py` no longer exports `snapshot_vqc`. Panel stays
  fixture-only with an extension badge. Re-enable when those return.
- **logic live panel** — same shape: `_build_logic` and `snapshot_logic`
  removed. Re-enable when restored.
- **qpcn live parameter editing during pause** — needs a substrate setter
  the snapshot extractor does not expose. Sliders render only if
  `/params/schema` flags them writable.
- **MERA isometry-violation indicator** — not in `snapshot_mera`. Omitted
  from panel until exposed.
- **hamiltonian term list with active-term highlighting** —
  `snapshot_hamiltonian` does not list terms. Panel renders what is
  present and records the term-list need.

`PanelShell` gains an `isExtension?: boolean` prop that renders a small
"extension pending" badge linking to the matching anchor in
`EXTENSIONS.md`.

## 9. Run lifecycle

`RunController` wraps the existing `run_simulation` generator:

```python
class RunController:
    def __init__(self, spec: RunSpec):
        self._spec = spec
        self._event = asyncio.Event(); self._event.set()
        self._step_request = 0
    async def frames(self) -> AsyncIterator[Frame]:
        for frame in run_simulation(self._spec):
            await self._event.wait()
            yield frame
            if self._step_request > 0:
                self._step_request -= 1
                self._event.clear()
    def pause(self):  self._event.clear()
    def resume(self): self._event.set()
    def step(self):   self._step_request += 1; self._event.set()
```

The WS handler awaits each frame from `controller.frames()`. Clients that
never call `/pause` are unaffected.

## 10. Compare two runs

`store.runs` is keyed by run id. The active panel always renders the active
frame; if a baseline is pinned, it also receives `baselineFrame`. Each
panel's render decides how to diff:

- Heatmaps (manifold, multifield, hamiltonian curvature): subtract baseline
  pixel-wise, render in the diverging ramp.
- Scalar readouts: show `current (Δ vs baseline)` with sign-coloured Δ.
- Metrics strips: overlay both runs as separate-coloured lines.
- 3D structural views (MERA tree): side-by-side miniatures rather than a
  diff render.

## 11. Manim community + export

`pyproject.toml` already has the `viz-manim` extra. The spec adds:

- `scripts/install-manim.sh` — installs system prereqs (cairo, pango,
  ffmpeg, latex packages) and runs `uv sync --extra viz-manim`.
- README instructions on running it.
- An Export menu in `RunControls` that calls `POST /export` and polls
  `/export/{job_id}`; the download button appears when state = `done`.
- JSONL export is a separate menu entry that hits `GET /export/run/{id}`.

## 12. Testing

**Python (`src/qft_pcn/tests/`):**

- Extend `test_viz_server.py`:
  - `GET /presets` returns the catalog
  - `GET /params/schema` validates against a known shape
  - `POST /runs/{id}/pause` then `/step` advances exactly one frame
  - `GET /export/run/{id}` returns valid JSONL
- New `test_presets.py`: round-trip every preset through `run_simulation`
  and assert the first frame's `layer_states[preset.layer]` is non-empty.

**TypeScript (vitest, `web/src/`):**

- `ExplainerPane` renders all 8 layer specs (KaTeX strings non-empty).
- `RunControls` POSTs the right `params` payload per preset.
- `PanelReadouts` highlights the right cell on `watch`-item hover.
- Compare mode swaps baseline/active correctly and renders deltas without
  crashing on missing fields.
- Each panel's `.test.tsx` gets a "renders with extension badge when
  snapshot is empty" case for vqc and logic.

**Manual smoke (recorded in `viz/README.md`):**

8 layers × {fixture, preset, advanced-params} matrix, plus
pause/resume/step, baseline compare, JSONL export, MP4 export.

## 13. Open questions

None blocking. Implementation plan will decide ordering of frontend vs
backend work — likely backend additions first (preset catalog, lifecycle,
JSONL export), then `ExplainerPane` + `RunControls`, then per-panel uplift
in parallel by panel.
