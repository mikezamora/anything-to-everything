# Visualizer Learning Documentation Expansion — Design

**Status:** Draft (brainstorming approved 2026-05-23).
**Scope:** `src/qft_pcn/viz/**` only. No substrate edits.
**Branch:** `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Companion to:** `2026-05-23-qpcn-viz-enhancement-design.md` + `2026-05-23-pcn-narrative-dsl-design.md` (both implemented).

## 1. Goal

The visualizer ships a working surface for every QPCN substrate layer, but the documentation density is too low for a software engineer with hobby-level QFT and PCN knowledge to deeply understand:

- what each panel's visualization actually *represents* in the underlying theory,
- how the live signals will evolve as the system trains, and
- how the active participants of each equation map onto the live numbers on screen.

This design adds a comprehensive learning surface to the viz so a reader can build their understanding from a software-engineer baseline up to fluency in the QPCN's QFT-side and PCN-side machinery and their fusion, with every equation showing who the active participants are.

Concretely it adds:

1. **Tabbed `ExplainerPane`** — the right rail grows from a stacked-section pane into a five-tab pane (Overview / Math / Worked Example / Training Dynamics / What to Watch) per layer.
2. **`/learn` route** — a textbook-style route with a contents tree on the left and full-article body on the right. Articles include Orientation, four Foundations chapters (vectors/tensors, Hilbert/operators, variational free energy, Riemannian geometry), per-panel deep dives, and a DSL walkthrough.
3. **`/training` route** — a single end-to-end narrative explaining how the whole QPCN trains across one run, annotated with the live-frame numbers of a chosen preset.
4. **Live-frame callouts** — a small `<FrameInterpreter>` near each panel's centerpiece that reads the current snapshot fields and renders 1–2 sentences interpreting them in the panel's theory.
5. **Annotated-equation system** — a shared `lib/equations.ts` registry. Each equation carries a per-symbol role map (input / parameter-learnable / output / state) with a stable colour vocabulary reused across every panel and article so visual familiarity builds across the docs.
6. **Bug-fix bundle** — fix the silent DSL-route Ollama failure: `ChatPane`'s model picker self-diagnoses with status-aware placeholder options + inline error span; `App.tsx`'s error banner moves outside the viz-only route so DSL/Learn/Training also surface errors; `viz/README.md` documents the WSL-to-Windows `OLLAMA_HOST` gateway IP idiom; `scripts/ollama-host.sh` prints the right URL.

## 2. Non-goals

- Substrate changes. Parallel agents own the substrate; this work touches viz only.
- New snapshot fields. The `<FrameInterpreter>` consumes only fields the snapshot extractors already expose; gaps go in `viz/EXTENSIONS.md`.
- Replacing the existing 4-region viz layout. Routes Viz / DSL stay as-is; Learn / Training are additive.
- Auto-generating articles from substrate code. Articles are hand-authored against `QFT_PCN_ARCHITECTURE.md` and the substrate files for accuracy, NOT scraped or templated.
- LLM-emitted documentation. No model output is treated as canonical.

## 3. Constraints

- **No placeholders.** Missing substrate hooks → `viz/EXTENSIONS.md`. Missing interpreters → render nothing for that frame.
- **Defensive snapshot reads.** Every `<FrameInterpreter>` per-panel function uses the existing `_safe`-style guard pattern (null-check then fall back to no-render).
- **Stable role vocabulary.** Once defined in `lib/equations.ts`, the role taxonomy `input | param-learn | param-const | output | state | observable` does not change. Articles and explainer tabs reuse the same colours.
- **Mathematical accuracy is binding.** Every annotated equation must match either an equation in `QFT_PCN_ARCHITECTURE.md` (with section cite) or the substrate code (with file:line cite). No invented or paraphrased physics.
- **Software-engineer baseline.** Foundations chapters teach from "you know what a 2D array is" up. QFT and PCN are taught from first principles each, then composed.
- **Live-frame callouts honest about uncertainty.** When the interpretation depends on a sign convention or a chosen preset, say so. No oracle-style fortune-telling.

## 4. Architecture

### 4.1 Tabbed Explainer Pane

`ExplainerPane.tsx` is rewritten to render five tabs:

```
┌─ Manifold ──────────────────────┐
│ [Overview*][Math][WE][Dyn][Watch]│
├─────────────────────────────────┤
│ <tab body>                       │
│                                  │
│                                  │
└─────────────────────────────────┘
```

Tab content comes from a widened `ExplainerSpec`:

```ts
interface ExplainerSpec {
  title: string;
  oneLine: string;
  tabs: {
    overview: { what: string[]; elements: { name; meaning; code? }[] };
    math: { equationIds: string[] };  // refs into lib/equations.ts
    workedExample: WorkedExample;
    trainingDynamics: {
      updateRule: string;             // equation id
      expect: string[];               // bullets
      pathologies: { signal: string; cause: string }[];
    };
    watch: { label: string; readout?: string }[];
  };
  references?: { label: string; href: string }[];
}

interface WorkedExample {
  title: string;
  setup: string;                       // 1-2 sentence scenario
  steps: { description: string; equationId?: string; result: string }[];
  takeaway: string;
}
```

The pane shows the tab buttons and the body for the active tab. State is per-layer (so a reader's tab choice persists when they switch layers and back). Hover on a `watch` item still fires `viz:highlight-readout` as today.

### 4.2 Learn Route

A new top-level route added to `RouteSwitcher`. Two-column layout: left `LearnContents` (contents tree, current article highlighted, sticky on scroll) + right `LearnArticle` (article body).

```
┌─ [Viz] [DSL] [Learn*] [Training] ──────────────────────────┐
├─────────────────┬──────────────────────────────────────────┤
│ §0 Orientation  │ ## §3.1 Manifold — dynamic Riemannian    │
│ §1 Foundations  │ ## geometry as a substrate               │
│   1.1 Vectors   │                                          │
│   1.2 Hilbert   │ A 2D Riemannian manifold whose metric    │
│   1.3 Var. FE   │ g_μν is NOT fixed: it is sourced by …    │
│   1.4 Riemann   │                                          │
│ §2 QFT side     │ <AnnotatedEquation id="metric-perturb"/> │
│   2.1 MPS       │                                          │
│   2.2 MERA      │ ### Worked example                       │
│   …             │   Setup: 4×4 grid with one hot spot…     │
│ §3 PCN side     │   Step 1: …                              │
│   3.1 Manifold *│                                          │
│   3.2 PCN-Field │ ### Training dynamics                    │
│ §4 Fusion       │   You should see mean|R| rise then…      │
│ §5 DSL          │                                          │
└─────────────────┴──────────────────────────────────────────┘
```

Each article is a TSX file exporting an `ArticleSpec`:

```ts
interface ArticleSpec {
  id: string;            // 'fusion-manifold'
  title: string;
  sectionPath: string[]; // ['§3 PCN side', '3.1 Manifold']
  prerequisites?: string[]; // ids of articles a reader should consume first
  sections: ArticleSection[];
  citations: { label: string; href: string }[];
}

type ArticleSection =
  | { kind: 'prose'; body: string[] /* paragraphs */ }
  | { kind: 'equation'; equationId: string; caption?: string }
  | { kind: 'workedExample'; example: WorkedExample }
  | { kind: 'trainingDynamics'; updateRuleId: string; expect: string[]; pathologies: Pathology[] }
  | { kind: 'miniViz'; layer: string; fixtureFrameId: string } // cheap panels only
  | { kind: 'citation'; label: string; href: string }
  | { kind: 'callout'; severity: 'note' | 'warn'; body: string };
```

`LearnArticle.tsx` walks the sections list and dispatches per `kind`. `LearnContents.tsx` builds the tree from the static articles registry.

Articles to ship (19 total):
- `orientation`
- `foundations-vectors-tensors`, `foundations-hilbert-operators`, `foundations-variational-fe`, `foundations-riemannian`
- `qft-mps`, `qft-mera`, `qft-hamiltonian`, `qft-vqc`
- `pcn-fields`, `pcn-dynamics`, `pcn-multifield`
- `fusion-manifold`, `fusion-pcn-coupling`, `fusion-qpcn`, `fusion-logic`, `fusion-mera-relax`, `fusion-bridge`
- `dsl-walkthrough`

### 4.3 Training Route

A new top-level route with a single long-form article walking through the QPCN's end-to-end training dynamics, annotated with live-frame numbers from `qpcn.quarter-density-target` preset. Same `<LearnArticle>` renderer; one `ArticleSpec` driving it.

Outline:
1. The variational free-energy descent (the system-level objective).
2. Substrate construction from the DSL (`dsl_to_runspec` → builders).
3. The per-frame loop: observe → cascade up → stress-energy sources the metric → metric reshapes belief diffusion → bottom-up signal reaches the QPCN → imag-time evolution updates the MPS → prediction errors → Hamiltonian parameter descent → next frame.
4. What to expect over a 30-step run: which signals decay, which oscillate, which converge.
5. Failure modes: stalling, divergence, mode collapse, gauge-fixing drift.

### 4.4 Live-frame callouts

A `<FrameInterpreter layer="…" />` mounts near each panel's centerpiece. It subscribes to the active run's current frame and calls a per-panel function from `lib/interpreters.ts`:

```ts
interface InterpretationOutput {
  text: string;          // 1-2 sentence interpretation
  citation?: string;     // article id or equation id to deep-link to
}

type Interpreter = (layerState: Record<string, unknown>, step: number) =>
  InterpretationOutput | null;

export const INTERPRETERS: Record<string, Interpreter> = {
  manifold: (st, step) => {
    const mar = st.mean_abs_ricci as number | undefined;
    if (mar == null) return null;
    if (mar > 0.4) return { text: `Step ${step}: mean|R| = ${mar.toFixed(3)} — curvature is concentrating, likely tracking an error spike.`, citation: 'fusion-manifold' };
    if (mar < 0.02) return { text: `Step ${step}: mean|R| ≈ 0 — geometry has nearly flattened, the error field is no longer sourcing curvature.`, citation: 'fusion-manifold' };
    return { text: `Step ${step}: mean|R| = ${mar.toFixed(3)} — moderate curvature, system is mid-relaxation.`, citation: 'fusion-manifold' };
  },
  // … 12 more
};
```

The renderer shows the interpretation as a small floating box near the centerpiece. When `citation` is present, the box has a "→ open article" link that navigates to `/learn#<citation>` (or scrolls within the active explainer tab).

### 4.5 Annotated-equation system

A new `lib/equations.ts`:

```ts
type Role = 'input' | 'param-learn' | 'param-const' | 'output' | 'state' | 'observable';

interface EquationSpec {
  id: string;                         // 'free-energy-functional'
  tex: string;                        // KaTeX
  roleMap: Record<string, Role>;      // { '\\Pi': 'param-learn', 'E': 'state', 'F': 'output' }
  gloss: string;                      // natural-language one-liner
  symbolGlosses: { symbol: string; gloss: string; role: Role }[];
  sourceCitation: string;             // 'arch §3.1' or 'src/qft_pcn/layer.py:192'
}

export const EQUATIONS: Record<string, EquationSpec> = { … };
```

The colour palette is fixed:
- input → cool blue (`#6cd0ff`)
- param-learn → amber (`#fbc66a`)
- param-const → slate (`#7e8aa3`)
- output → green (`#9aedc1`)
- state → violet (`#d291ff`)
- observable → coral (`#ef9090`)

`<AnnotatedEquation id="..." />`:
- Renders the equation in KaTeX.
- Below: a per-symbol glossary line where each symbol is rendered with its role colour as a swatch dot.
- Below that: the natural-language `gloss`.
- Citation in small grey text at the bottom-right.

`<AnnotatedEquation>` is the single renderer; explainer tabs and Learn articles both consume it by id. No drift possible.

### 4.6 Bug-fix bundle

**ChatPane self-diagnosis** — `routes/dsl/ChatPane.tsx`:

```tsx
const [loadStatus, setLoadStatus] = useState<'loading' | 'ok' | 'error'>('loading');
const [loadError, setLoadError] = useState<string>('');
useEffect(() => {
  loadModels()
    .then((ms) => { setModels(ms); setLoadStatus('ok'); if (!model) setModel(defaultModel(ms)); })
    .catch((e) => { setLoadStatus('error'); setLoadError(String(e)); setError(String(e)); });
}, []);

// in render:
<select value={model ?? ''} onChange={(e) => setModel(e.target.value || null)}>
  {loadStatus === 'loading' && <option value="">(loading models…)</option>}
  {loadStatus === 'error' && <option value="">(no models — is Ollama running?)</option>}
  {loadStatus === 'ok' && models.length === 0 && <option value="">(no models installed)</option>}
  {loadStatus === 'ok' && <option value="">(pick a model)</option>}
  {models.map((m) => <option key={m.name} value={m.name}>{m.name}</option>)}
</select>
{loadStatus === 'error' && <div className="chat-error">{loadError}</div>}
```

**Error banner moved** — `App.tsx`: the `{error && <div className="error-bar" ...>}` block moves OUT of the `{route === 'viz' ? ... : <DslRoute />}` ternary so it renders for every route.

**WSL note + helper** — `src/qft_pcn/viz/README.md` gains:

```markdown
## Running with Ollama on a Windows host (WSL viz server)

Ollama listens on the Windows host; WSL's `localhost` does NOT reach it. Use the
WSL→Windows gateway IP:

    OLLAMA_HOST=$(./scripts/ollama-host.sh) ./scripts/viz.sh

Or set it permanently in your shell rc:

    export OLLAMA_HOST=http://$(ip route show | awk '/^default/ {print $3}'):11434
```

`scripts/ollama-host.sh`:

```bash
#!/usr/bin/env bash
# Print the URL the WSL viz server should use to reach a Windows-hosted Ollama.
gw=$(ip route show | awk '/^default/ {print $3}')
echo "http://${gw:-localhost}:11434"
```

## 5. Data flow

### Static (build-time)
- `lib/equations.ts` is import-only; consumed by `<AnnotatedEquation>`.
- `lib/explainer.ts` carries tab content per layer; consumed by `ExplainerPane`.
- `routes/learn/articles/*.tsx` each export an `ArticleSpec`; aggregated in `routes/learn/articles/index.ts` for the route's contents tree.

### Live (per-frame)
- `<FrameInterpreter layer="X" />` subscribes via `useVizStore` to the active run's current frame.
- On each frame change, calls `INTERPRETERS[X](frame.layer_states[X], frame.step)`.
- If non-null, renders the interpretation as a small overlay box on the panel.

### Routing
- `RouteSwitcher` widens to four buttons (Viz | DSL | Learn | Training).
- `store.route: 'viz' | 'dsl' | 'learn' | 'training'`.
- `App.tsx` route ternary widens to a switch.
- Error banner mounted OUTSIDE the switch so it persists across all routes.

## 6. File layout

### Created (frontend)
- `web/src/lib/equations.ts`
- `web/src/lib/interpreters.ts`
- `web/src/lib/article-types.ts`
- `web/src/components/AnnotatedEquation.tsx` + `.test.tsx`
- `web/src/components/FrameInterpreter.tsx` + `.test.tsx`
- `web/src/routes/LearnRoute.tsx`
- `web/src/routes/learn/LearnContents.tsx`
- `web/src/routes/learn/LearnArticle.tsx` + `.test.tsx`
- `web/src/routes/learn/articles/orientation.tsx`
- `web/src/routes/learn/articles/foundations-vectors-tensors.tsx`
- `web/src/routes/learn/articles/foundations-hilbert-operators.tsx`
- `web/src/routes/learn/articles/foundations-variational-fe.tsx`
- `web/src/routes/learn/articles/foundations-riemannian.tsx`
- `web/src/routes/learn/articles/qft-mps.tsx`, `qft-mera.tsx`, `qft-hamiltonian.tsx`, `qft-vqc.tsx`
- `web/src/routes/learn/articles/pcn-fields.tsx`, `pcn-dynamics.tsx`, `pcn-multifield.tsx`
- `web/src/routes/learn/articles/fusion-manifold.tsx`, `fusion-pcn-coupling.tsx`, `fusion-qpcn.tsx`, `fusion-logic.tsx`, `fusion-mera-relax.tsx`, `fusion-bridge.tsx`
- `web/src/routes/learn/articles/dsl-walkthrough.tsx`
- `web/src/routes/learn/articles/index.ts`
- `web/src/routes/TrainingRoute.tsx` + `.test.tsx`
- `web/src/routes/LearnRoute.test.tsx`
- `scripts/ollama-host.sh`

### Modified (frontend)
- `web/src/lib/explainer.ts` (widen to tab-shaped ExplainerSpec)
- `web/src/components/ExplainerPane.tsx` + `.test.tsx` (tab UI)
- `web/src/components/RouteSwitcher.tsx` + `.test.tsx` (Learn + Training buttons)
- `web/src/lib/types.ts` (`Route` widens)
- `web/src/store.ts` (no real change; Route widens via types)
- `web/src/App.tsx` (route switch widens; error banner moves out)
- `web/src/App.css` (tab styles, learn-route layout, training-route layout, chat-error style, annotated-equation glossary swatches, FrameInterpreter overlay)
- `web/src/routes/dsl/ChatPane.tsx` (load-status + status-aware select + chat-error span)
- `web/src/routes/dsl/ChatPane.test.tsx` (new loading/error path tests)
- All 13 panel files (mount `<FrameInterpreter layer="…" />` near each centerpiece)

### Modified (backend / docs)
- `src/qft_pcn/viz/README.md` (Ollama-from-WSL note)
- `src/qft_pcn/viz/EXTENSIONS.md` (3 new deferred entries)

## 7. EXTENSIONS.md additions

Three new entries appended:

- **Step-replay annotation for the §10.10 induction-theorem demo** — the mera_relax panel would benefit from a frame-by-frame "this leaf moved because of rule R-AddZero at site 4" overlay. Snapshot already emits per-term residuals; the wire-up is straightforward but the annotation logic is its own design problem. Deferred.
- **Live mini-viz embedded in heavy-panel Learn articles** — R3F (manifold, multifield, MERA) and Three.js (manifold) panels are too expensive to mount multiple times in a scrolling article. v1 embeds them as static SVG snapshots from fixture frames. Live-mini-viz for those panels deferred until a lightweight "shrunken-render" mode exists.
- **Cross-panel concept search** — a future "where else does the Hamiltonian appear?" full-text search across articles + tabs would require an index over the article TOC + tab content. Deferred.

## 8. Decomposition for parallel implementation

Six workstreams, isolated to non-overlapping file sets where possible:

| WS | Files (owns) | Depends on |
|---|---|---|
| **A. Foundations layer** | `lib/equations.ts`, `lib/article-types.ts`, `lib/interpreters.ts`, `AnnotatedEquation.tsx` + test, `FrameInterpreter.tsx` + test | — |
| **B. Bug-fix bundle** | `ChatPane.tsx` (load-status), `ChatPane.test.tsx` (loading/error path tests), `App.tsx` (error-banner move), `App.css` (chat-error style), `viz/README.md` Ollama section, `scripts/ollama-host.sh` | — |
| **C. Explainer tab refactor** | `lib/explainer.ts` (widen), `ExplainerPane.tsx` + test (tab UI), `App.css` (tab styles). Seed all 13 layer specs with the existing content in the Overview tab; subsequent tasks fill the other 4 tabs per layer. | A |
| **D. Per-panel interpreters + FrameInterpreter mounts** | `lib/interpreters.ts` (13 functions), all 13 panel files (mount `<FrameInterpreter layer="…" />`) | A |
| **E. Learn route shell + Foundations + Orientation articles** | `LearnRoute.tsx`, `LearnContents.tsx`, `LearnArticle.tsx` + test, `articles/index.ts`, 5 articles (orientation + 4 foundations), `RouteSwitcher` (add Learn button), `Route` type widen, `App.tsx` route switch + error-banner move, `App.css` learn-route styles | A |
| **F. Learn route per-panel articles + Training route** | 13 per-panel articles + dsl-walkthrough + `TrainingRoute.tsx` + test + `RouteSwitcher` (add Training button) + App.tsx route switch (Training branch) + App.css training-route styles | E (route shell), A (annotated equations) |

A and B run in parallel from start. C, D, E start once A is committed. F starts once E's route shell lands. Within F, the 13 per-panel articles are fully independent and parallelize freely (different files).

The implementation plan will:
1. Land A + B in parallel.
2. Land C, D, E in parallel.
3. Land F's articles in parallel (multiple subagent waves until all 14 articles + Training route are in).
4. Final smoke + README + EXTENSIONS update.

Per-task TDD where applicable. Articles get a "renders without crashing for every registered article" sweep test + a per-article assertion that the article exports an `ArticleSpec` and that every `equationId` it references exists in `EQUATIONS`.

## 9. Testing

### TypeScript / vitest
- `AnnotatedEquation.test.tsx` — renders KaTeX, glossary swatches coloured per role, citation footer.
- `FrameInterpreter.test.tsx` — for each panel in `INTERPRETERS`, the interpreter is a function; for one panel a fixture frame produces non-null output.
- `ExplainerPane.test.tsx` — tab buttons render; clicking a tab swaps the body; the per-layer tab state persists across layer switches.
- `LearnRoute.test.tsx` — contents tree renders; clicking an entry selects the article; the right pane renders the article body.
- `LearnArticle.test.tsx` — for every registered article, renders without crashing; for one article (`fusion-manifold`), specific sections render in order.
- `TrainingRoute.test.tsx` — mounts; key section headings present.
- `RouteSwitcher.test.tsx` — extended: 4 routes; clicking Learn/Training sets store.
- `ChatPane.test.tsx` — extended: loading state shows `(loading models…)`; error state shows `(no models — is Ollama running?)` + chat-error span; success path unchanged.
- App.tsx — implicit coverage via routing tests; banner-on-DSL-route assertion via a Learn-route mount that simulates an error.
- Article-coverage sweep test: every `EXPLAINERS[layer]` whose tab references an `equationId` has a matching entry in `EQUATIONS`.

### No new pytest tests
The backend is untouched except for the README + helper script.

### Manual smoke (appended to viz/README.md)
- Route switching: Viz / DSL / Learn / Training each render their content.
- Explainer tabs: each of 13 layers, each of 5 tabs, renders content; tab persists when switching layers.
- Learn route: contents tree shows all 19 articles; each article renders.
- Training route: the end-to-end article renders.
- Live-frame callout: start a manifold run; confirm the FrameInterpreter overlay updates as frames stream.
- Ollama-from-WSL: `OLLAMA_HOST=$(./scripts/ollama-host.sh) ./scripts/viz.sh` then DSL route → model picker populates with `ollama list` contents.
- Ollama-down: stop Ollama (or `OLLAMA_HOST=http://0.0.0.0:1`) → DSL route picker shows `(no models — is Ollama running?)` and the error banner appears.

## 10. Risks and open questions

- **Authoring scale.** 19 long-form articles + 13×5 tab cells is the biggest content chunk this viz has absorbed. Plan staggers the priority articles (qpcn, manifold, pcn-coupling, mera_relax, fusion-overview) first.
- **Mathematical accuracy.** Every equation cites either `QFT_PCN_ARCHITECTURE.md` or a substrate file. Each article task includes a "cross-check equations against the cited source" step.
- **Bundle size.** Learn + Training routes add ~80–120 KB of TSX content (mostly text). Negligible next to Monaco's ~6 MB already in the bundle.
- **Live-frame callouts perf.** Subscribes to every frame; runs a small synchronous function. The panel-mounting overhead is bounded by a single store hook + one paragraph render — cheap.
- **Per-layer tab state.** Stored in component state, not the Zustand store. If a reader switches layer, comes back, their tab choice resets. Acceptable v1 trade-off.

## 11. Open questions

None blocking. The plan will produce 22-ish tasks (foundations + bug-fix + explainer refactor + interpreters + learn shell + 19 articles + training route + final smoke) sequenced per the dependency graph.
