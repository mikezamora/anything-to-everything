# PCN Panels + Narrative Arc + DSL Pipeline — Design

**Status:** Draft (brainstorming approved 2026-05-23).
**Scope:** `src/qft_pcn/viz/**` only. No substrate edits.
**Branch:** `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Companion to:** `2026-05-23-qpcn-viz-enhancement-design.md` (already implemented).

## 1. Goal

The visualizer currently presents the QFT side of the QPCN in depth (MPS, MERA, VQC, Hamiltonian, QPCN, Logic) but treats the PCN side as a single rolled-up "Manifold" panel. There is no narrative arc explaining how QFT and PCN fit together, and no surface for creating, importing, exporting, or running the QPCN DSL (the JSON spec the LLM emits per `QFT_PCN_ARCHITECTURE.md` §1.3).

This design adds three new subsystems:

1. **PCN substrate panels.** Three new layer keys (`pcn-fields`, `pcn-dynamics`, `pcn-coupling`) bring the PCN side to QFT-side fidelity.
2. **Narrative LayerSelector.** The left rail groups layers into three collapsible sections (QFT Substrate / PCN Substrate / QPCN Fusion). Section headers expand inline intro panels explaining the section's role.
3. **DSL editor + Ollama LLM pipeline.** A new top-level `/dsl` route with a Monaco editor, an Ollama-driven chat pane, a stepped-flow debug bar, and import/export — wiring the full LLM → DSL → QPCN → DSL → LLM round trip from the architecture's §1.3.

## 2. Non-goals

- Substrate changes. Parallel agents own the substrate; this work touches viz only.
- New snapshot fields from substrate hooks that don't exist. Anything that needs a new substrate API is recorded in `src/qft_pcn/viz/EXTENSIONS.md`.
- Replacing the existing audit-driven viz work. The 4-region layout, presets, compare-mode, RunController lifecycle, JSONL/MP4 export stay as-is.
- Lossless DSL round-tripping. Export-current-run-as-DSL is best-effort and documented.
- LLM retry-on-failure loops. v1 surfaces the LLM's raw output to the user when DSL validation fails; the user fixes it manually.

## 3. Constraints

- **No placeholders.** Missing substrate hooks go in `src/qft_pcn/viz/EXTENSIONS.md`; the UI omits the affordance.
- **Defensive snapshot reads.** Every `snapshot_pcn_*` uses the existing `_safe(lambda: …)` pattern.
- **Substrate isolation.** Touch only `src/qft_pcn/viz/**` and `src/qft_pcn/tests/test_viz_*.py`.
- **Determinism.** A given DSL + seed produces an identical frame sequence (re-uses the existing `RunSpec` determinism).
- **LLM validation.** Every model output is JSON-Schema validated before being accepted into the editor. Failed validation returns `{error, raw, validation_errors}` to the chat pane.
- **Run-size clamps stay.** `_MAX_GRID`, `_MAX_STEPS`, `_QPCN_*`, `_MERA_*`, `_LOGIC_*` continue to bound resource use. DSL-driven runs are clamped the same way.

## 4. Architecture

### 4.1 PCN substrate panels

Three new layer keys, all reading the existing `QFTPCNNetwork` and `MultiFieldNetwork` public attributes.

**`pcn-fields`** — hierarchical Φ / E / Π stack
- Renders one mini-3D-surface per PCN layer (`net.layers[0..L-1]`), stacked vertically.
- Per-layer toggle selects which field (Φ, E, Π) is rendered as the surface; channel selector multiplexes channels.
- Click a stack entry to expand it to a full panel-width surface for detailed inspection.
- Readouts: layer depth L, total ‖Φ‖₂, total ‖E‖₂, mean precision Π.

**`pcn-dynamics`** — free energy + learning trajectories
- Free energy `F` decomposed (where possible) into the accuracy term `½ Π E²` and the complexity term `−½ log Π` — both are integrals over the manifold per layer.
- Time-series strip per layer (one line per PCN layer) of: F-contribution, mean ‖E‖₂, mean Π.
- Readouts: total F (sum over layers), per-layer F contributions.
- Damping / learning-rate readouts where the layer config exposes them.

**`pcn-coupling`** — QFT ↔ PCN bridge
- Left ↔ right bridge diagram. Left node = PCN, right node = QFT.
- Top arrow: PCN error-field stress-energy → QFT manifold metric (`g_μν` source).
- Bottom arrow: QFT operator expectations ⟨Ô⟩ → PCN observation targets.
- Arrow widths animate with live magnitudes (mean |stress-energy| up, mean |expectation| down).
- Readouts: coupling strength `κ` (manifold-coupling Hamiltonian term), mean stress-energy, mean operator expectation.

Snapshot extractors `snapshot_pcn_fields`, `snapshot_pcn_dynamics`, `snapshot_pcn_coupling` live in `snapshots.py` and are added to `LAYER_KEYS` in `schema.py`. `run_simulation` adds them to `want_*` dispatch — each runs against the existing `QFTPCNNetwork` and adds the corresponding key to that step's `layer_states`. No new substrate built.

### 4.2 Narrative LayerSelector + intro panels

`SectionedLayerSelector` replaces the flat list. Three sections:

```
▾ QFT Substrate      [click header → IntroQftPanel]
    mps · mera · vqc · hamiltonian
▾ PCN Substrate      [click header → IntroPcnPanel]
    pcn-fields · pcn-dynamics · multifield
▾ QPCN Fusion        [click header → IntroQpcnPanel]
    pcn-coupling · qpcn · manifold · logic
```

Section headers are clickable. Clicking a header selects a pseudo-layer (`intro-qft`, `intro-pcn`, `intro-qpcn`); `panelFor` routes to the corresponding `IntroPanel` which reads static content from `lib/sections.ts`:

```ts
interface SectionSpec {
  key: 'qft' | 'pcn' | 'qpcn';
  title: string;             // "QFT Substrate"
  tagline: string;           // one-line section summary
  story: string[];           // 3-4 paragraphs from QFT_PCN_ARCHITECTURE.md
  layerSummaries: { layer: string; oneLine: string }[];
  references: { label: string; href: string }[];
}
```

`panels/index.ts` maps `intro-qft` / `intro-pcn` / `intro-qpcn` to their components alongside the 11 real layers.

### 4.3 DSL editor + Ollama LLM pipeline

A new top-level route. `App.tsx` wraps the body in a `RouteSwitcher` exposing two routes: `viz` (the existing 4-region viz) and `dsl` (the new DSL surface).

#### DSL route layout

```
┌─────────────────────────────────────────────────────────────┐
│  [Viz | DSL]   model: gemma4:31b ▾   ⛬ stepped-flow mode    │
├─────────────────────┬───────────────────┬───────────────────┤
│ Chat pane           │ Monaco DSL editor │ Run output pane   │
│ ─────────────────── │ ───────────────── │ ───────────────── │
│ user> simulate ...  │ {                 │ status: running   │
│ assistant> DSL ⬇    │   "fields": [...] │ step 12/30        │
│ assistant> result   │   "hamiltonian":  │ observables:      │
│                     │     { terms:...}  │   ⟨n_0⟩ = 0.243   │
│                     │   "observables":  │                   │
│                     │     [...]         │                   │
│                     │ }                 │                   │
│ [Send] [Import]     │ [Validate] [Run]  │ [Export DSL]      │
└─────────────────────┴───────────────────┴───────────────────┘
```

A stepped-flow bar overlays the same surface when "stepped-flow mode" is on: three explicit buttons (Generate DSL → Run DSL → Verbalize result) with the intermediate artifact shown clearly between steps.

#### DSL schema (`dsl-schema.json`)

Authored per `QFT_PCN_ARCHITECTURE.md` §1.3, in JSON Schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["fields", "hamiltonian", "observables"],
  "properties": {
    "fields": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "cutoff"],
        "properties": {
          "name": { "type": "string" },
          "cutoff": { "type": "integer", "minimum": 1, "maximum": 8 },
          "bare_mass": { "type": "number", "default": 1.0 },
          "kinetic": { "type": "number", "default": 0.5 }
        }
      }
    },
    "hamiltonian": {
      "type": "object",
      "properties": {
        "terms": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["kind"],
            "properties": {
              "kind": {
                "type": "string",
                "enum": ["mass", "kinetic", "quartic", "yukawa",
                         "density", "curvature_coupling"]
              },
              "site": { "type": "integer" },
              "species": { "type": "string" },
              "coefficient": { "type": "number" }
            }
          }
        }
      }
    },
    "observables": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["operator"],
        "properties": {
          "operator": { "type": "string", "enum": ["n", "phi", "phi2"] },
          "site": { "type": "integer" },
          "species": { "type": "string" },
          "target": { "type": ["number", "null"] }
        }
      }
    },
    "run": {
      "type": "object",
      "properties": {
        "steps": { "type": "integer", "default": 30, "maximum": 500 },
        "chi_max": { "type": "integer", "default": 8, "maximum": 16 },
        "seed": { "type": ["integer", "null"], "default": null }
      }
    }
  }
}
```

#### DSL → RunSpec translation (`dsl.py`)

```python
def dsl_to_runspec(dsl: dict) -> RunSpec:
    """Translate a validated DSL to a RunSpec the existing simulator can consume."""

def runspec_to_dsl(spec: RunSpec) -> dict:
    """Best-effort reverse — generates a DSL whose dsl_to_runspec round-trips back
    to a RunSpec equivalent to `spec` for the fields present. Documented as
    lossy because RunSpec.params is a flat dict and the DSL has structured
    sub-objects."""
```

`dsl.py` also exposes:
- `validate(dsl: dict) -> list[str]` — returns validation errors (empty list = valid)
- `load_schema() -> dict` — reads `dsl-schema.json` from disk
- `EXAMPLES: list[dict]` — three canonical examples used in the LLM system prompt few-shot

#### Ollama wrapper (`llm.py`)

```python
def list_models() -> list[dict]
    # Calls `ollama list` (or http://localhost:11434/api/tags).
    # Frontend default-picks the first model whose name starts with "gemma";
    # else the first model in the list; else null (user must pick manually).

def generate_dsl(prompt: str, model: str, *, schema: dict,
                 examples: list[dict]) -> dict
    # System prompt: "You emit QPCN DSL conforming to <schema>. Examples: ..."
    # Validates output against schema. On failure returns {error, raw, validation_errors}.
    # Strips deepseek-r1 <think>...</think> wrappers if present.

def verbalize(observations: dict, model: str, original_prompt: str) -> str
    # Asks the LLM to explain the observation results in plain language given
    # the user's original question.
```

#### New server endpoints (`server.py`)

- `GET /dsl/schema` → the JSON Schema
- `GET /dsl/models` → Ollama models list (404 if Ollama unreachable, 503 surfaced)
- `POST /dsl/translate` body `{prompt, model}` → `{dsl}` or `{error, raw, validation_errors}`
- `POST /dsl/run` body `{dsl}` → validates, translates to `RunSpec`, registers a run, returns `{run_id}`
- `POST /dsl/verbalize` body `{run_id, model, original_prompt}` → `{text}`
- `GET /dsl/export/{run_id}` → reverse-DSL JSON for the registered run

## 5. Data flow

### 5.1 PCN snapshots

```
QFTPCNNetwork ──┬─▶ snapshot_pcn_fields(net)    ─▶ layer_states["pcn-fields"]
                ├─▶ snapshot_pcn_dynamics(net)  ─▶ layer_states["pcn-dynamics"]
                └─▶ snapshot_pcn_coupling(net,  ─▶ layer_states["pcn-coupling"]
                                          qpcn?)
```

### 5.2 DSL pipeline

```
chat prompt
   │
   ▼
POST /dsl/translate (model=gemma4:31b)
   │  Ollama emits + viz validates against dsl-schema.json
   ▼
Monaco DSL editor (user can edit)
   │
   ▼ click "Run DSL"
POST /dsl/run
   │  dsl_to_runspec → RunSpec → _registry.add(run_id, spec)
   ▼
existing WS /ws/{run_id} (panels stream as normal)
   │
   ▼  user clicks "Verbalize result"
POST /dsl/verbalize {run_id, model, original_prompt}
   │  Ollama produces natural-language summary
   ▼
chat pane appends assistant turn
```

## 6. File map

### Created (backend)
- `src/qft_pcn/viz/dsl.py`
- `src/qft_pcn/viz/dsl-schema.json`
- `src/qft_pcn/viz/llm.py`
- `src/qft_pcn/tests/test_viz_pcn_snapshots.py`
- `src/qft_pcn/tests/test_viz_dsl.py`
- `src/qft_pcn/tests/test_viz_llm.py`

### Modified (backend)
- `src/qft_pcn/viz/snapshots.py` (add 3 `snapshot_pcn_*` extractors)
- `src/qft_pcn/viz/runs.py` (add `pcn-fields`, `pcn-dynamics`, `pcn-coupling` dispatch)
- `src/qft_pcn/viz/schema.py` (`LAYER_KEYS` adds 3 keys)
- `src/qft_pcn/viz/server.py` (add 6 `/dsl/*` endpoints)
- `src/qft_pcn/viz/presets.py` (PCN presets + DSL param schema entry)
- `src/qft_pcn/viz/EXTENSIONS.md` (3 new deferred items)
- `src/qft_pcn/tests/test_viz_endpoints_extra.py` (DSL endpoint tests)
- `pyproject.toml` (`jsonschema`, `httpx` to `viz` extra)

### Created (frontend)
- `web/src/lib/sections.ts`
- `web/src/lib/dsl.ts`
- `web/src/lib/llm.ts`
- `web/src/panels/PcnFieldsPanel.tsx` + `.test.tsx`
- `web/src/panels/PcnDynamicsPanel.tsx` + `.test.tsx`
- `web/src/panels/PcnCouplingPanel.tsx` + `.test.tsx`
- `web/src/panels/IntroQftPanel.tsx`, `IntroPcnPanel.tsx`, `IntroQpcnPanel.tsx`
- `web/src/panels/IntroPanel.test.tsx`
- `web/src/components/SectionedLayerSelector.tsx` + `.test.tsx`
- `web/src/components/RouteSwitcher.tsx` + `.test.tsx`
- `web/src/routes/DslRoute.tsx`
- `web/src/routes/dsl/ChatPane.tsx` + `.test.tsx`
- `web/src/routes/dsl/DslEditor.tsx` + `.test.tsx`
- `web/src/routes/dsl/RunOutputPane.tsx`
- `web/src/routes/dsl/SteppedFlowBar.tsx` + `.test.tsx`

### Modified (frontend)
- `web/src/App.tsx` (mount `RouteSwitcher`; route between viz body and DSL route)
- `web/src/lib/types.ts` (`LAYER_KEYS` adds 3 keys; new `DslSpec`, `LlmModel`, `ChatTurn`, `Section`, `Route` types)
- `web/src/panels/index.ts` (register PcnFields/PcnDynamics/PcnCoupling/IntroQft/IntroPcn/IntroQpcn)
- `web/src/lib/explainer.ts` (ExplainerSpecs for the 3 new PCN layers)
- `web/src/App.css` (section rail styles; DSL route layout)
- `web/src/store.ts` (add `route: 'viz' | 'dsl'`, `chat: ChatTurn[]`, `dslText: string`, `llmModel: string | null`, `steppedMode: boolean`)
- `web/package.json` (add `@monaco-editor/react`, `monaco-editor`, `ajv`, `ajv-formats`)
- `web/vite.config.ts` (proxy `/dsl` to backend; already proxies via `/dsl` -> backend port 8000 entry to be added)

## 7. EXTENSIONS.md additions

Three new entries appended to `src/qft_pcn/viz/EXTENSIONS.md`:

- **Per-PCN-layer KL divergence** — `PcnDynamicsPanel`'s F decomposition needs a per-layer `kl_divergence()` substrate method. Until it lands the panel renders aggregate F plus per-layer prediction-error norm.
- **Lossless DSL ↔ RunSpec round-trip** — `runspec_to_dsl` is best-effort because `RunSpec.params` is flat. To make round-trip exact, `RunSpec` would need a structured `dsl` companion field. Deferred.
- **LLM retry / chain-of-thought streaming** — `/dsl/translate` is a single shot; if validation fails the user repairs manually. A retry loop with the validation error fed back into the LLM is a natural follow-up but not in v1.

## 8. Decomposition for parallel implementation

Five workstreams, isolated to non-overlapping files so they parallelize cleanly:

| WS | Files (owns) | Depends on |
|---|---|---|
| **A** PCN snapshots + runs wiring | `snapshots.py` (3 new fns), `runs.py` (dispatch), `schema.py` (LAYER_KEYS), `test_viz_pcn_snapshots.py` | — |
| **B** PCN panels (Fields / Dynamics / Coupling) + explainer entries | `panels/Pcn*Panel.tsx`, tests, `lib/explainer.ts` adds | A (snapshot keys present) |
| **C** Sectioned LayerSelector + Intro panels | `components/SectionedLayerSelector.tsx` + test, `panels/Intro*Panel.tsx` + test, `lib/sections.ts`, `panels/index.ts` adds | — |
| **D** DSL backend (schema + translator + LLM + endpoints) | `dsl.py`, `dsl-schema.json`, `llm.py`, `server.py` adds, `pyproject.toml` extras, `test_viz_dsl.py`, `test_viz_llm.py`, `test_viz_endpoints_extra.py` adds | — |
| **E** DSL frontend route (Monaco + chat + stepped) | `routes/DslRoute.tsx` + sub-files + tests, `lib/dsl.ts`, `lib/llm.ts`, `components/RouteSwitcher.tsx` + test, `App.tsx` mount, `store.ts` adds, `package.json`, `vite.config.ts` | D (endpoint shape known) |

A → B and D → E are soft dependencies (B's tests need A's keys; E's client matches D's contract). C is fully independent. The plan will sequence A and D first, then B, C, E in parallel.

## 9. Testing

### Python
- `test_viz_pcn_snapshots.py` — each `snapshot_pcn_*` round-trips a real `QFTPCNNetwork`; asserts expected keys + defensive behaviour with missing optional attrs.
- `test_viz_dsl.py` — schema validation accepts good DSL, rejects malformed; `dsl_to_runspec` produces a valid `RunSpec`; round-trip DSL → RunSpec → DSL preserves the field/observable structure.
- `test_viz_llm.py` — mocks `httpx` against Ollama; `generate_dsl` returns parsed+validated JSON; `verbalize` strips `<think>…</think>` wrappers from reasoning-model output.
- `test_viz_endpoints_extra.py` — appends tests for the 6 new `/dsl/*` endpoints (mocking the Ollama HTTP calls).

### TypeScript (vitest)
- `PcnFieldsPanel.test.tsx`, `PcnDynamicsPanel.test.tsx`, `PcnCouplingPanel.test.tsx` — each renders with a 2-layer fixture; readouts present.
- `IntroPanel.test.tsx` — each of three intro panels renders title + body from `lib/sections.ts`.
- `SectionedLayerSelector.test.tsx` — sections collapse/expand; click on a layer selects it; click on a section header selects the intro pseudo-layer.
- `RouteSwitcher.test.tsx` — toggling routes preserves frame buffer in the store.
- `DslEditor.test.tsx` — Monaco mounts; validation errors appear inline.
- `ChatPane.test.tsx` — turns accumulate; model picker calls `/dsl/models`; Generate populates editor; Verbalize appends assistant turn.
- `SteppedFlowBar.test.tsx` — each button gated on previous step's success.

### Manual smoke
Appended to `viz/README.md`: walk the three sections + intro panels; run the DSL editor through a chat-mode and stepped-mode round-trip with at least one model from `ollama list`.

## 10. Risks and open questions

- **Monaco bundle size.** Monaco adds ~2 MB to the Vite build. Acceptable for a research tool; document the build-size growth in the README.
- **Ollama availability at runtime.** When Ollama is not running, every `/dsl/*` route except `/dsl/schema` returns 503. The chat pane handles this with a clear "Ollama unreachable — start it with `ollama serve`" banner.
- **Reasoning model formatting.** `deepseek-r1-32b` wraps output in `<think>` blocks. `llm.generate_dsl` strips them; `verbalize` keeps the post-think portion only.
- **DSL evolution.** The schema is v1 and will likely change as the substrate adds new Hamiltonian term types. The schema is versioned via a `$id`; the editor will refuse a DSL with a mismatched `$id` and direct the user to update.

## 11. Open questions

None blocking. The implementation plan will sequence A and D first, then B / C / E parallelizable.
