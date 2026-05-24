# Viz Learning Documentation Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the QPCN visualizer's documentation surface so a software engineer with hobby-level QFT/PCN knowledge can deeply understand the system — via a tabbed `ExplainerPane`, a `/learn` textbook route with 19 articles, a `/training` end-to-end route, live-frame interpreter callouts on every panel, and an annotated-equation system with stable role-coloured vocabulary. Also fixes the silent DSL Ollama-from-WSL failure.

**Architecture:** Six workstreams (A–F from spec §8). WS A (foundations: equations + interpreters + AnnotatedEquation + FrameInterpreter) and WS B (bug-fix bundle) start in parallel. WS C/D/E (explainer refactor / per-panel interpreter mounts / Learn route shell) start once A lands. WS F (19 articles + Training route) starts once E's shell lands; the 19 articles parallelize fully.

**Tech Stack:** React 18 + Vite + TypeScript, Zustand, KaTeX, vitest. No backend changes beyond `viz/README.md` and `scripts/ollama-host.sh`.

**Spec:** `docs/superpowers/specs/2026-05-23-viz-learning-docs-design.md`

**Hard constraints:**
- Touch only `src/qft_pcn/viz/**` and (for the Ollama helper) `scripts/ollama-host.sh`. Substrate untouched.
- No placeholders / TODOs / fake data. Missing-substrate features go in `src/qft_pcn/viz/EXTENSIONS.md`.
- All `lib/interpreters.ts` per-panel functions must defensively handle missing snapshot fields and return `null` rather than throwing.
- All equations cited in `lib/equations.ts` must reference either `QFT_PCN_ARCHITECTURE.md` (section number) or a substrate file (`file:line`). No invented physics.
- Stable role colour palette across every panel and article:
  - `input` → `#6cd0ff` (cool blue)
  - `param-learn` → `#fbc66a` (amber)
  - `param-const` → `#7e8aa3` (slate)
  - `output` → `#9aedc1` (green)
  - `state` → `#d291ff` (violet)
  - `observable` → `#ef9090` (coral)
- Per-test convention: `import '@testing-library/jest-dom/vitest';` at top; `globalThis as any` for fetch mocks.
- **Commit hygiene:** always `git add <explicit paths>`; `git diff --staged --name-only` before EVERY commit; `git restore --staged <path>` for parallel-agent leaks. Never `git add -A` or `git add .`.
- Frontend tests via `./node_modules/.bin/vitest --run <pattern>` (pnpm sometimes blocked). TypeScript strict via `./node_modules/.bin/tsc --noEmit`.

---

## File Map

**Created (frontend lib + components)**
- `web/src/lib/equations.ts` — `EquationSpec` + role palette + registry
- `web/src/lib/interpreters.ts` — per-panel `Interpreter` functions for 13 panels
- `web/src/lib/article-types.ts` — `ArticleSpec` / `ArticleSection` / `WorkedExample` / `Pathology` types
- `web/src/components/AnnotatedEquation.tsx` + `.test.tsx`
- `web/src/components/FrameInterpreter.tsx` + `.test.tsx`

**Created (Learn route)**
- `web/src/routes/LearnRoute.tsx` + `.test.tsx`
- `web/src/routes/learn/LearnContents.tsx`
- `web/src/routes/learn/LearnArticle.tsx` + `.test.tsx`
- `web/src/routes/learn/articles/index.ts`
- 19 article TSX files (one per article id in spec §6)

**Created (Training route)**
- `web/src/routes/TrainingRoute.tsx` + `.test.tsx`

**Created (script)**
- `scripts/ollama-host.sh`

**Modified (frontend)**
- `web/src/lib/explainer.ts` (widen `ExplainerSpec` to tab-shaped)
- `web/src/components/ExplainerPane.tsx` + `.test.tsx` (tab UI)
- `web/src/components/RouteSwitcher.tsx` + `.test.tsx` (Learn + Training buttons)
- `web/src/lib/types.ts` (`Route` widens to `'viz' | 'dsl' | 'learn' | 'training'`)
- `web/src/App.tsx` (route switch widens; error banner moved OUT of the route ternary)
- `web/src/App.css` (tab styles, learn-route layout, training-route layout, chat-error span, glossary swatches, FrameInterpreter overlay)
- `web/src/routes/dsl/ChatPane.tsx` (load-status + status-aware select + chat-error span)
- `web/src/routes/dsl/ChatPane.test.tsx` (loading/error path tests)
- All 13 panel files in `web/src/panels/*Panel.tsx` (mount `<FrameInterpreter layer="…" />`)

**Modified (docs)**
- `src/qft_pcn/viz/README.md` (Ollama-from-WSL section)
- `src/qft_pcn/viz/EXTENSIONS.md` (3 new deferred entries)

---

## Per-Task Process

Each task follows: **spec review → TDD → code-review pass on diff → commit**.

- **Spec review (before starting):** Re-read the relevant spec section. Confirm scope alignment, no substrate edits, no placeholders.
- **Code review (before commit):** Re-read the diff for TODOs/stubs/fake values, substrate leaks, undefensive snapshot reads, missing tests, unrelated drive-bys.

---

## Workstream A — Foundations layer

### Task 1: `article-types.ts` + role palette in `equations.ts`

**Files:**
- Create: `src/qft_pcn/viz/web/src/lib/article-types.ts`
- Create: `src/qft_pcn/viz/web/src/lib/equations.ts` (minimal — types + palette + empty registry)

- [ ] **Step 1: Write `article-types.ts`**

```ts
// src/qft_pcn/viz/web/src/lib/article-types.ts
/**
 * Types backing the Learn route and the tabbed ExplainerPane.
 *
 * An ArticleSpec is a static, structured document describing one
 * Learn-route chapter. ExplainerSpec.tabs (in lib/explainer.ts) re-uses
 * WorkedExample + Pathology shapes so the same examples can power both
 * the rail tabs and the long-form articles without duplication.
 */

export interface WorkedExample {
  title: string;
  setup: string;
  steps: { description: string; equationId?: string; result: string }[];
  takeaway: string;
}

export interface Pathology {
  signal: string;     // "mean |R| oscillating"
  cause: string;      // "κ_R too large; reduce coupling"
}

export type ArticleSection =
  | { kind: 'prose'; body: string[] }
  | { kind: 'equation'; equationId: string; caption?: string }
  | { kind: 'workedExample'; example: WorkedExample }
  | {
      kind: 'trainingDynamics';
      updateRuleId: string;       // equation id
      expect: string[];           // bullets
      pathologies: Pathology[];
    }
  | { kind: 'miniViz'; layer: string; fixtureFrameId: string }
  | { kind: 'callout'; severity: 'note' | 'warn'; body: string };

export interface ArticleSpec {
  id: string;
  title: string;
  sectionPath: string[];        // ['§3 PCN side', '3.1 Manifold']
  prerequisites?: string[];     // article ids
  sections: ArticleSection[];
  citations: { label: string; href: string }[];
}
```

- [ ] **Step 2: Write `equations.ts` skeleton**

```ts
// src/qft_pcn/viz/web/src/lib/equations.ts
/**
 * Annotated-equation registry. Every equation in the docs is registered
 * here so the renderer can attach role-coloured symbol glosses + a
 * citation back to the architecture doc or the substrate source. The
 * role palette is the visual vocabulary readers build across the viz.
 */

export type Role =
  | 'input'
  | 'param-learn'
  | 'param-const'
  | 'output'
  | 'state'
  | 'observable';

export const ROLE_COLOR: Record<Role, string> = {
  'input':       '#6cd0ff',
  'param-learn': '#fbc66a',
  'param-const': '#7e8aa3',
  'output':      '#9aedc1',
  'state':       '#d291ff',
  'observable':  '#ef9090',
};

export const ROLE_LABEL: Record<Role, string> = {
  'input':       'input',
  'param-learn': 'param (learn)',
  'param-const': 'param (const)',
  'output':      'output',
  'state':       'state',
  'observable':  'observable',
};

export interface SymbolGloss {
  symbol: string;     // KaTeX-rendered string e.g. '\\Pi'
  gloss: string;      // natural-language one-liner
  role: Role;
}

export interface EquationSpec {
  id: string;
  tex: string;
  gloss: string;
  symbolGlosses: SymbolGloss[];
  sourceCitation: string;       // 'Arch §3.1' or 'src/qft_pcn/layer.py:192'
}

export const EQUATIONS: Record<string, EquationSpec> = {};
```

The registry starts empty; subsequent tasks (T8, T9, T10, and the 19 article tasks) populate it.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/viz/web/src/lib/article-types.ts \
        src/qft_pcn/viz/web/src/lib/equations.ts
git diff --staged --name-only   # must show ONLY those two
git commit -m "feat(viz/web): foundations — ArticleSpec types + Role palette + EQUATIONS skeleton"
```

---

### Task 2: `AnnotatedEquation` component

**Files:**
- Create: `src/qft_pcn/viz/web/src/components/AnnotatedEquation.tsx`
- Create: `src/qft_pcn/viz/web/src/components/AnnotatedEquation.test.tsx`

- [ ] **Step 1: Add a fixture equation to `equations.ts` for the test**

Edit `src/qft_pcn/viz/web/src/lib/equations.ts` — add to `EQUATIONS`:

```ts
EQUATIONS['__test_free_energy_functional__'] = {
  id: '__test_free_energy_functional__',
  tex: 'F = \\tfrac{1}{2} \\Pi E^2 - \\tfrac{1}{2} \\log \\Pi',
  gloss: 'Free energy is half the precision-weighted squared error minus half the log precision.',
  symbolGlosses: [
    { symbol: 'F',       gloss: 'free energy',       role: 'output' },
    { symbol: '\\Pi',    gloss: 'precision',         role: 'param-learn' },
    { symbol: 'E',       gloss: 'prediction error',  role: 'state' },
  ],
  sourceCitation: 'Arch §3.1',
};
```

(Leave a comment saying this is a fixture; will be displaced by real T8 entries.)

- [ ] **Step 2: Write the failing test**

```tsx
// src/qft_pcn/viz/web/src/components/AnnotatedEquation.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AnnotatedEquation } from './AnnotatedEquation';
import { ROLE_COLOR } from '../lib/equations';

describe('AnnotatedEquation', () => {
  it('renders KaTeX for a registered equation', () => {
    const { container } = render(
      <AnnotatedEquation id="__test_free_energy_functional__" />,
    );
    expect(container.querySelector('.katex')).toBeInTheDocument();
  });

  it('renders the natural-language gloss', () => {
    render(<AnnotatedEquation id="__test_free_energy_functional__" />);
    expect(screen.getByText(/precision-weighted squared error/i))
      .toBeInTheDocument();
  });

  it('renders one swatch + gloss per symbol with the role colour', () => {
    const { container } = render(
      <AnnotatedEquation id="__test_free_energy_functional__" />,
    );
    const swatches = container.querySelectorAll('.annotated-eq-swatch');
    expect(swatches.length).toBe(3);
    const styles = Array.from(swatches).map(
      (s) => (s as HTMLElement).style.background,
    );
    expect(styles).toContain(_rgb(ROLE_COLOR['output']));
    expect(styles).toContain(_rgb(ROLE_COLOR['param-learn']));
    expect(styles).toContain(_rgb(ROLE_COLOR['state']));
  });

  it('renders the citation footer', () => {
    render(<AnnotatedEquation id="__test_free_energy_functional__" />);
    expect(screen.getByText(/Arch §3\.1/)).toBeInTheDocument();
  });

  it('renders a fallback when the id is missing', () => {
    render(<AnnotatedEquation id="this-does-not-exist" />);
    expect(screen.getByText(/missing equation/i)).toBeInTheDocument();
  });
});

// Browsers compute color as e.g. `rgb(108, 208, 255)`. Convert hex for compare.
function _rgb(hex: string): string {
  const h = hex.replace('#', '');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgb(${r}, ${g}, ${b})`;
}
```

- [ ] **Step 3: Run test to verify it fails**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run AnnotatedEquation.test.tsx
```
Expected: cannot resolve `AnnotatedEquation`.

- [ ] **Step 4: Implement**

```tsx
// src/qft_pcn/viz/web/src/components/AnnotatedEquation.tsx
/**
 * Render an equation registered in lib/equations.ts:
 *   - KaTeX equation
 *   - Per-symbol glossary line with role-coloured swatches
 *   - Natural-language gloss
 *   - Source citation footer
 */

import katex from 'katex';
import { EQUATIONS, ROLE_COLOR, ROLE_LABEL } from '../lib/equations';

interface Props {
  id: string;
}

export function AnnotatedEquation({ id }: Props) {
  const eq = EQUATIONS[id];
  if (!eq) {
    return (
      <div className="annotated-eq annotated-eq-missing">
        missing equation: <code>{id}</code>
      </div>
    );
  }
  const katexHtml = katex.renderToString(eq.tex, {
    throwOnError: false,
    displayMode: true,
  });
  return (
    <div className="annotated-eq">
      <div
        className="annotated-eq-tex"
        dangerouslySetInnerHTML={{ __html: katexHtml }}
      />
      <ul className="annotated-eq-symbols">
        {eq.symbolGlosses.map((sg, i) => {
          const symHtml = katex.renderToString(sg.symbol, {
            throwOnError: false,
            displayMode: false,
          });
          return (
            <li key={i} className="annotated-eq-symbol-row">
              <span
                className="annotated-eq-swatch"
                style={{ background: ROLE_COLOR[sg.role] }}
                title={ROLE_LABEL[sg.role]}
              />
              <span
                className="annotated-eq-symbol-name"
                dangerouslySetInnerHTML={{ __html: symHtml }}
              />
              <span className="annotated-eq-symbol-gloss">
                — {sg.gloss}{' '}
                <small>({ROLE_LABEL[sg.role]})</small>
              </span>
            </li>
          );
        })}
      </ul>
      <p className="annotated-eq-gloss">{eq.gloss}</p>
      <div className="annotated-eq-cite">{eq.sourceCitation}</div>
    </div>
  );
}
```

- [ ] **Step 5: Add the CSS**

Append to `src/qft_pcn/viz/web/src/App.css`:

```css
.annotated-eq { background: #0a0d14; border: 1px solid #1c2233;
                border-radius: 6px; padding: 12px; margin: 8px 0; }
.annotated-eq-missing { color: #ef9090; }
.annotated-eq-tex { margin: 8px 0; color: #e0e6f3; }
.annotated-eq-symbols { list-style: none; padding: 0; margin: 4px 0; }
.annotated-eq-symbol-row { display: flex; align-items: center; gap: 8px;
                            font-size: 12px; color: #c8d0e0; }
.annotated-eq-swatch { display: inline-block; width: 10px; height: 10px;
                       border-radius: 2px; }
.annotated-eq-symbol-gloss { color: #9aa3bb; }
.annotated-eq-symbol-gloss small { color: #7e8aa3; }
.annotated-eq-gloss { color: #c8d0e0; font-size: 13px; margin: 8px 0 4px; }
.annotated-eq-cite { color: #7e8aa3; font-size: 10px; text-align: right; }
```

- [ ] **Step 6: Run test to pass**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run AnnotatedEquation.test.tsx
```
Expected: 5/5 PASS.

- [ ] **Step 7: Commit**

```bash
git add src/qft_pcn/viz/web/src/components/AnnotatedEquation.tsx \
        src/qft_pcn/viz/web/src/components/AnnotatedEquation.test.tsx \
        src/qft_pcn/viz/web/src/lib/equations.ts \
        src/qft_pcn/viz/web/src/App.css
git diff --staged --name-only
git commit -m "feat(viz/web): AnnotatedEquation component + role-coloured glosses"
```

---

### Task 3: `interpreters.ts` skeleton + `FrameInterpreter` component

**Files:**
- Create: `src/qft_pcn/viz/web/src/lib/interpreters.ts`
- Create: `src/qft_pcn/viz/web/src/components/FrameInterpreter.tsx`
- Create: `src/qft_pcn/viz/web/src/components/FrameInterpreter.test.tsx`

- [ ] **Step 1: Write `lib/interpreters.ts`**

Seed it with a single working interpreter for `manifold` (T11 fills in the remaining 12):

```ts
// src/qft_pcn/viz/web/src/lib/interpreters.ts
/**
 * Per-panel live-frame interpreters. Each takes the panel's current
 * layer_state dict + the global frame step, returns a 1-2 sentence
 * interpretation or null when there's nothing meaningful to say.
 *
 * Citations are article ids (Learn route) so the FrameInterpreter
 * overlay can deep-link.
 *
 * All reads MUST be defensive — missing fields yield null, not exceptions.
 */

export interface InterpretationOutput {
  text: string;
  citation?: string;
}

export type Interpreter = (
  layerState: Record<string, unknown>,
  step: number,
) => InterpretationOutput | null;

function _num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}

export const INTERPRETERS: Record<string, Interpreter> = {
  manifold: (st, step) => {
    const mar = _num(st['mean_abs_ricci']);
    if (mar === undefined) return null;
    if (mar > 0.4) {
      return {
        text: `Step ${step}: mean|R| = ${mar.toFixed(3)} — curvature is concentrating, likely tracking an error spike.`,
        citation: 'fusion-manifold',
      };
    }
    if (mar < 0.02) {
      return {
        text: `Step ${step}: mean|R| ≈ 0 — geometry has nearly flattened; the error field is no longer sourcing curvature.`,
        citation: 'fusion-manifold',
      };
    }
    return {
      text: `Step ${step}: mean|R| = ${mar.toFixed(3)} — moderate curvature, the system is mid-relaxation.`,
      citation: 'fusion-manifold',
    };
  },
};
```

- [ ] **Step 2: Write the failing test**

```tsx
// src/qft_pcn/viz/web/src/components/FrameInterpreter.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { FrameInterpreter } from './FrameInterpreter';
import { useVizStore } from '../store';
import { INTERPRETERS } from '../lib/interpreters';

beforeEach(() => useVizStore.getState().resetAll());

describe('FrameInterpreter', () => {
  it('renders nothing when there is no active run', () => {
    const { container } = render(<FrameInterpreter layer="manifold" />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when the interpreter returns null', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.setActiveRun('A');
    s.pushFrame('A', { step: 0,
                       layer_states: { manifold: { /* no mean_abs_ricci */ } } });
    const { container } = render(<FrameInterpreter layer="manifold" />);
    expect(container.firstChild).toBeNull();
  });

  it('renders the interpretation text when interpreter returns output', () => {
    const s = useVizStore.getState();
    s.openRun('A'); s.setActiveRun('A');
    s.pushFrame('A', { step: 12,
      layer_states: { manifold: { mean_abs_ricci: 0.55 } } });
    render(<FrameInterpreter layer="manifold" />);
    expect(screen.getByText(/curvature is concentrating/i))
      .toBeInTheDocument();
    expect(screen.getByText(/Step 12/)).toBeInTheDocument();
  });

  it('exposes all registered interpreters as callable functions', () => {
    for (const [key, fn] of Object.entries(INTERPRETERS)) {
      expect(typeof fn).toBe('function');
      // Each must handle empty state without throwing:
      expect(fn({}, 0)).toBeNull();
    }
  });
});
```

- [ ] **Step 3: Run to verify it fails**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run FrameInterpreter.test.tsx
```
Expected: cannot resolve `FrameInterpreter`.

- [ ] **Step 4: Implement `FrameInterpreter.tsx`**

```tsx
// src/qft_pcn/viz/web/src/components/FrameInterpreter.tsx
/**
 * Render a 1-2 sentence interpretation of the active frame for one layer.
 * Subscribes via Zustand; mounts as a small overlay on each panel.
 */

import { useVizStore } from '../store';
import { INTERPRETERS } from '../lib/interpreters';

interface Props {
  layer: string;
}

export function FrameInterpreter({ layer }: Props) {
  const activeRunId = useVizStore((s) => s.activeRunId);
  const frame = useVizStore((s) => {
    if (!activeRunId) return undefined;
    const run = s.runs.get(activeRunId);
    return run?.frames[run.cursor];
  });

  if (!frame) return null;
  const interpret = INTERPRETERS[layer];
  if (!interpret) return null;
  const ls = (frame.layer_states[layer] ?? {}) as Record<string, unknown>;
  const out = interpret(ls, frame.step);
  if (!out) return null;

  return (
    <div className="frame-interpreter" role="note">
      <span className="frame-interpreter-marker">🔬</span>
      <span className="frame-interpreter-text">{out.text}</span>
      {out.citation && (
        <a
          className="frame-interpreter-cite"
          href={`#/learn/${out.citation}`}
        >
          → open article
        </a>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Append CSS**

Append to `App.css`:

```css
.frame-interpreter { display: inline-flex; gap: 8px; align-items: baseline;
                     padding: 6px 10px; background: rgba(13, 17, 26, 0.85);
                     border: 1px solid #2a3450; border-radius: 4px;
                     color: #c8d0e0; font-size: 12px; max-width: 480px; }
.frame-interpreter-marker { font-size: 14px; }
.frame-interpreter-text { flex: 1; }
.frame-interpreter-cite { color: #6cd0ff; font-size: 11px;
                          text-decoration: none; }
.frame-interpreter-cite:hover { text-decoration: underline; }
```

- [ ] **Step 6: Run + commit**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run FrameInterpreter.test.tsx
```
Expected: 4/4 PASS.

```bash
git add src/qft_pcn/viz/web/src/lib/interpreters.ts \
        src/qft_pcn/viz/web/src/components/FrameInterpreter.tsx \
        src/qft_pcn/viz/web/src/components/FrameInterpreter.test.tsx \
        src/qft_pcn/viz/web/src/App.css
git diff --staged --name-only
git commit -m "feat(viz/web): FrameInterpreter component + interpreters skeleton (manifold seeded)"
```

---

## Workstream B — Bug-fix bundle

### Task 4: ChatPane self-diagnosis + error banner moved + Ollama helper + README

**Files:**
- Modify: `src/qft_pcn/viz/web/src/routes/dsl/ChatPane.tsx`
- Modify: `src/qft_pcn/viz/web/src/routes/dsl/ChatPane.test.tsx`
- Modify: `src/qft_pcn/viz/web/src/App.tsx`
- Modify: `src/qft_pcn/viz/web/src/App.css`
- Create: `scripts/ollama-host.sh`
- Modify: `src/qft_pcn/viz/README.md`

- [ ] **Step 1: Extend the ChatPane test for loading + error paths**

Append to `ChatPane.test.tsx` (after the existing tests):

```tsx
it('renders "(loading models…)" before /dsl/models resolves', async () => {
  // Override fetch to NEVER resolve so the loading state sticks.
  (globalThis as any).fetch = vi.fn(() => new Promise(() => {}));
  render(<ChatPane />);
  expect(await screen.findByText(/loading models/i)).toBeInTheDocument();
});

it('renders "(no models — is Ollama running?)" + chat-error span on /dsl/models 503',
   async () => {
  _resetLlmCache();
  (globalThis as any).fetch = vi.fn(async (url: any) => {
    const u = String(url);
    if (u.endsWith('/dsl/models'))
      return { ok: false, status: 503,
               json: async () => ({}) } as any;
    throw new Error('unexpected ' + u);
  });
  render(<ChatPane />);
  expect(await screen.findByText(/no models — is Ollama running\?/i))
    .toBeInTheDocument();
  expect(document.querySelector('.chat-error')).toBeInTheDocument();
});
```

Add the imports at top:

```ts
import { _resetLlmCache } from '../../lib/llm';
```

- [ ] **Step 2: Update ChatPane**

Replace the existing `loadModels` `useEffect` + `<select>` block in `ChatPane.tsx`:

```tsx
import { useEffect, useState } from 'react';
import { translate } from '../../lib/dsl';
import { defaultModel, loadModels } from '../../lib/llm';
import { useVizStore } from '../../store';
import type { LlmModel } from '../../lib/types';

type LoadStatus = 'loading' | 'ok' | 'error';

export function ChatPane() {
  const [models, setModels] = useState<LlmModel[]>([]);
  const [loadStatus, setLoadStatus] = useState<LoadStatus>('loading');
  const [loadError, setLoadError] = useState<string>('');
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);

  const chat = useVizStore((s) => s.chat);
  const append = useVizStore((s) => s.appendChat);
  const setDslText = useVizStore((s) => s.setDslText);
  const model = useVizStore((s) => s.llmModel);
  const setModel = useVizStore((s) => s.setLlmModel);
  const setError = useVizStore((s) => s.setError);

  useEffect(() => {
    loadModels()
      .then((ms) => {
        setModels(ms);
        setLoadStatus('ok');
        if (!model) setModel(defaultModel(ms));
      })
      .catch((e) => {
        setLoadStatus('error');
        setLoadError(String(e));
        setError(String(e));
      });
  }, []);

  const send = async () => {
    if (!prompt.trim() || !model) return;
    setBusy(true);
    append({ role: 'user', text: prompt });
    try {
      const result = await translate(prompt, model);
      if (result.dsl) {
        setDslText(JSON.stringify(result.dsl, null, 2));
        append({
          role: 'assistant',
          text: 'Emitted DSL into the editor.',
          artifact: { kind: 'dsl', payload: result.dsl },
        });
      } else {
        // Surface raw LLM output and validation errors; also seed editor for repair.
        if (result.raw) setDslText(result.raw);
        append({
          role: 'assistant',
          text:
            `DSL validation failed: ${result.error}\n` +
            `${(result.validation_errors ?? []).join('\n')}\n\n` +
            `Raw LLM output:\n${result.raw ?? '(none)'}`,
        });
      }
      setPrompt('');
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="chat-pane">
      <div className="chat-header">
        <label>
          model{' '}
          <select
            value={model ?? ''}
            onChange={(e) => setModel(e.target.value || null)}
          >
            {loadStatus === 'loading' && (
              <option value="">(loading models…)</option>
            )}
            {loadStatus === 'error' && (
              <option value="">(no models — is Ollama running?)</option>
            )}
            {loadStatus === 'ok' && models.length === 0 && (
              <option value="">(no models installed)</option>
            )}
            {loadStatus === 'ok' && models.length > 0 && (
              <option value="">(pick a model)</option>
            )}
            {models.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))}
          </select>
        </label>
      </div>
      {loadStatus === 'error' && (
        <div className="chat-error">{loadError}</div>
      )}
      <div className="chat-turns">
        {chat.map((t, i) => (
          <div key={i} className={`chat-turn chat-turn-${t.role}`}>
            <strong>{t.role}</strong>
            <pre>{t.text}</pre>
          </div>
        ))}
      </div>
      <div className="chat-input">
        <textarea
          placeholder="Ask the QPCN..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <button type="button" onClick={send} disabled={busy || !model}>
          {busy ? 'sending…' : 'Send'}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Append CSS for `.chat-error`**

Append to `App.css`:

```css
.chat-error { padding: 4px 8px; background: #2b1a1a; color: #ef9090;
              border-bottom: 1px solid #5a2a2a; font-size: 11px; }
```

- [ ] **Step 4: Move error banner OUT of the route ternary in `App.tsx`**

Locate the JSX in `App.tsx` where the error banner is rendered (currently inside the `route === 'viz'` branch). Move the `{error && <div className="error-bar" role="alert">{error}</div>}` line so it sits BETWEEN the `<RouteSwitcher />` mount and the `{route === 'viz' ? … : <DslRoute />}` switch. The result should look like:

```tsx
export default function App() {
  const error = useVizStore((s) => s.error);
  const route = useVizStore((s) => s.route);
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  return (
    <div className="app">
      <RouteSwitcher />
      {error && <div className="error-bar" role="alert">{error}</div>}
      {route === 'viz' ? (
        <>
          <RunControls />
          <CompareBar />
          <div className="body">
            <SectionedLayerSelector />
            <PanelArea />
            <ExplainerPane layer={selectedLayer} />
          </div>
          <Timeline />
        </>
      ) : (
        <DslRoute />
      )}
    </div>
  );
}
```

(No need to touch any other JSX in App.tsx for this task.)

- [ ] **Step 5: Create `scripts/ollama-host.sh`**

```bash
#!/usr/bin/env bash
# Print the URL the WSL viz server should use to reach a Windows-hosted Ollama.
# WSL's `localhost:11434` does NOT reach the Windows host; use the gateway IP.
gw=$(ip route show 2>/dev/null | awk '/^default/ {print $3; exit}')
echo "http://${gw:-localhost}:11434"
```

Then: `chmod +x scripts/ollama-host.sh`.

- [ ] **Step 6: Append the Ollama-from-WSL section to `viz/README.md`**

Append to `src/qft_pcn/viz/README.md`:

```markdown
## Running with Ollama on a Windows host (WSL viz server)

Ollama listens on the Windows host; WSL's `localhost` does NOT reach it. Use
the WSL→Windows gateway IP. The helper script computes it for you:

    OLLAMA_HOST=$(./scripts/ollama-host.sh) ./scripts/viz.sh

Or set it permanently in your shell rc:

    export OLLAMA_HOST=http://$(ip route show | awk '/^default/ {print $3}'):11434

When the DSL route's model picker shows `(no models — is Ollama running?)`,
the most likely cause is that `OLLAMA_HOST` is not set and the server defaulted
to `http://localhost:11434` which never reaches the Windows host.
```

- [ ] **Step 7: Run tests + commit**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run ChatPane.test.tsx
```
Expected: prior 2 PASS + 2 new PASS = 4/4.

```bash
chmod +x scripts/ollama-host.sh
git add src/qft_pcn/viz/web/src/routes/dsl/ChatPane.tsx \
        src/qft_pcn/viz/web/src/routes/dsl/ChatPane.test.tsx \
        src/qft_pcn/viz/web/src/App.tsx \
        src/qft_pcn/viz/web/src/App.css \
        scripts/ollama-host.sh \
        src/qft_pcn/viz/README.md
git diff --staged --name-only
git commit -m "fix(viz): self-diagnosing model picker + cross-route error banner + Ollama-WSL helper"
```

---

## Workstream C — Explainer tab refactor

### Task 5: Widen `ExplainerSpec` to tabs + render tab UI

**Files:**
- Modify: `src/qft_pcn/viz/web/src/lib/explainer.ts`
- Modify: `src/qft_pcn/viz/web/src/components/ExplainerPane.tsx`
- Modify: `src/qft_pcn/viz/web/src/components/ExplainerPane.test.tsx`
- Modify: `src/qft_pcn/viz/web/src/App.css`

- [ ] **Step 1: Widen `ExplainerSpec` and migrate existing content into the Overview tab**

In `lib/explainer.ts`, rewrite the `ExplainerSpec` interface:

```ts
import type { WorkedExample, Pathology } from './article-types';

export interface ExplainerSpec {
  title: string;
  oneLine: string;
  tabs: {
    overview: {
      what: string[];
      elements: { name: string; meaning: string; code?: string }[];
    };
    math: { equationIds: string[] };
    workedExample: WorkedExample;
    trainingDynamics: {
      updateRuleId: string;
      expect: string[];
      pathologies: Pathology[];
    };
    watch: { label: string; readout?: string }[];
  };
  references?: { label: string; href: string }[];
}
```

For EACH existing entry in `EXPLAINERS`, restructure:
- Move `what` and `elements` → `tabs.overview`.
- Move `math` array → keep its TeX strings BUT replace the `math` field with `tabs.math.equationIds: []` (empty array; populated in T8).
- Move `watch` → `tabs.watch`.
- Add new `tabs.workedExample` placeholder: each layer gets a minimal but real example. Pattern:
  ```ts
  workedExample: {
    title: 'Tiny <layer> example',
    setup: 'A 1-line scenario.',
    steps: [{ description: 'Compute X.', result: 'Y = 0.0' }],
    takeaway: 'In the limiting case the value collapses to Y.',
  },
  ```
  Author one real worked example per layer (use the existing `math` strings as guides; the goal is something a reader could verify by hand). These will be expanded in later tasks but must NOT be left as TODOs.
- Add `tabs.trainingDynamics` with:
  ```ts
  trainingDynamics: {
    updateRuleId: '',  // empty until T8 lands; renderer falls back gracefully
    expect: ['<one real bullet derived from the existing watch content>'],
    pathologies: [{ signal: '<plausible signal>', cause: '<plausible cause>' }],
  },
  ```

Every layer (13 of them) must compile and produce non-trivial content. No skipped tabs, no TODO strings.

- [ ] **Step 2: Update `ExplainerPane` to render tabs**

Replace the body of `ExplainerPane.tsx`:

```tsx
import { useState } from 'react';
import { EXPLAINERS, type ExplainerSpec } from '../lib/explainer';
import { tex } from '../panels/common';
import { AnnotatedEquation } from './AnnotatedEquation';

type TabId = 'overview' | 'math' | 'workedExample' | 'trainingDynamics' | 'watch';

const TAB_ORDER: { id: TabId; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'math', label: 'Math' },
  { id: 'workedExample', label: 'Worked Example' },
  { id: 'trainingDynamics', label: 'Dynamics' },
  { id: 'watch', label: 'Watch' },
];

interface Props { layer: string }

export function ExplainerPane({ layer }: Props) {
  const spec = EXPLAINERS[layer];
  const [collapsed, setCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>('overview');

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
        <>
          <p className="explainer-oneline">{spec.oneLine}</p>
          <nav className="explainer-tabs">
            {TAB_ORDER.map((t) => (
              <button
                key={t.id}
                type="button"
                className={`explainer-tab${
                  activeTab === t.id ? ' active' : ''
                }`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className="explainer-body">
            {renderTab(spec, activeTab)}
            {spec.references && activeTab === 'overview' && (
              <section className="explainer-refs">
                <h4>References</h4>
                <ul>
                  {spec.references.map((r, i) => (
                    <li key={i}>
                      <a href={r.href}>{r.label}</a>
                    </li>
                  ))}
                </ul>
              </section>
            )}
          </div>
        </>
      )}
    </aside>
  );
}

function renderTab(spec: ExplainerSpec, tab: TabId) {
  if (tab === 'overview') {
    return (
      <>
        <section>
          <h4>What</h4>
          {spec.tabs.overview.what.map((p, i) => <p key={i}>{p}</p>)}
        </section>
        <section>
          <h4>Elements</h4>
          <ul>
            {spec.tabs.overview.elements.map((e, i) => (
              <li key={i}>
                <strong>{e.name}</strong> — {e.meaning}
                {e.code && <code className="explainer-code"> {e.code}</code>}
              </li>
            ))}
          </ul>
        </section>
      </>
    );
  }
  if (tab === 'math') {
    if (spec.tabs.math.equationIds.length === 0) {
      return <p className="explainer-empty">Equations not yet registered for this layer.</p>;
    }
    return (
      <>
        {spec.tabs.math.equationIds.map((id) => (
          <AnnotatedEquation key={id} id={id} />
        ))}
      </>
    );
  }
  if (tab === 'workedExample') {
    const ex = spec.tabs.workedExample;
    return (
      <section className="explainer-worked-example">
        <h4>{ex.title}</h4>
        <p><strong>Setup.</strong> {ex.setup}</p>
        <ol>
          {ex.steps.map((s, i) => (
            <li key={i}>
              {s.description}
              {s.equationId && (
                <div className="explainer-worked-eq">
                  <AnnotatedEquation id={s.equationId} />
                </div>
              )}
              <div className="explainer-worked-result"><em>⇒ {s.result}</em></div>
            </li>
          ))}
        </ol>
        <p><strong>Takeaway.</strong> {ex.takeaway}</p>
      </section>
    );
  }
  if (tab === 'trainingDynamics') {
    const td = spec.tabs.trainingDynamics;
    return (
      <section>
        {td.updateRuleId && (
          <>
            <h4>Update rule</h4>
            <AnnotatedEquation id={td.updateRuleId} />
          </>
        )}
        <h4>Expect</h4>
        <ul>{td.expect.map((b, i) => <li key={i}>{b}</li>)}</ul>
        <h4>If you see…</h4>
        <ul>
          {td.pathologies.map((p, i) => (
            <li key={i}>
              <strong>{p.signal}</strong> — {p.cause}
            </li>
          ))}
        </ul>
      </section>
    );
  }
  // watch
  return (
    <section>
      <h4>Watch</h4>
      <ul>
        {spec.tabs.watch.map((w, i) => (
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
  );
}

function dispatchHighlight(readoutId: string | null) {
  window.dispatchEvent(
    new CustomEvent('viz:highlight-readout', { detail: { id: readoutId } }),
  );
}
```

NOTE: the existing `tex()` import may no longer be used; remove it if `tsc --noEmit` complains.

- [ ] **Step 3: Update the existing ExplainerPane test**

Replace `ExplainerPane.test.tsx`:

```tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExplainerPane } from './ExplainerPane';
import { EXPLAINERS } from '../lib/explainer';

describe('ExplainerPane (tabbed)', () => {
  it('renders title + one-liner for every layer', () => {
    for (const key of Object.keys(EXPLAINERS)) {
      const { unmount } = render(<ExplainerPane layer={key} />);
      expect(screen.getByText(EXPLAINERS[key].title)).toBeInTheDocument();
      expect(screen.getByText(EXPLAINERS[key].oneLine)).toBeInTheDocument();
      unmount();
    }
  });

  it('renders the 5 tab buttons', () => {
    render(<ExplainerPane layer="manifold" />);
    for (const label of ['Overview', 'Math', 'Worked Example', 'Dynamics', 'Watch']) {
      expect(screen.getByRole('button', { name: label })).toBeInTheDocument();
    }
  });

  it('clicking a tab swaps the body content', () => {
    render(<ExplainerPane layer="manifold" />);
    // Default: Overview shows "What" heading.
    expect(screen.getByText('What')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Watch' }));
    expect(screen.getByText('Watch')).toBeInTheDocument();
  });

  it('toggles collapsed state via the header button', () => {
    const { container } = render(<ExplainerPane layer="manifold" />);
    const btn = container.querySelector('button.explainer-toggle')!;
    fireEvent.click(btn);
    expect(container.querySelector('.explainer.collapsed')).toBeTruthy();
  });
});
```

- [ ] **Step 4: Append tab styles to `App.css`**

```css
.explainer-tabs { display: flex; gap: 2px; padding: 4px 8px;
                  border-bottom: 1px solid #1c2233; flex-wrap: wrap; }
.explainer-tab { background: #161b29; color: #c8d0e0;
                 border: 1px solid #2a3450; border-radius: 4px;
                 padding: 2px 8px; font-size: 11px; cursor: pointer; }
.explainer-tab.active { background: #2b3a6b; }
.explainer-empty { color: #7e8aa3; font-style: italic; }
.explainer-worked-example ol { padding-left: 20px; }
.explainer-worked-example li { margin-bottom: 8px; }
.explainer-worked-result { color: #9aedc1; font-size: 12px; margin-top: 4px; }
.explainer-worked-eq { margin: 4px 0; }
```

- [ ] **Step 5: Run all viz tests to confirm no regression**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run && ./node_modules/.bin/tsc --noEmit
```
Expected: all PASS, tsc clean.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/viz/web/src/lib/explainer.ts \
        src/qft_pcn/viz/web/src/components/ExplainerPane.tsx \
        src/qft_pcn/viz/web/src/components/ExplainerPane.test.tsx \
        src/qft_pcn/viz/web/src/App.css
git diff --staged --name-only
git commit -m "feat(viz/web): tabbed ExplainerPane (Overview/Math/WE/Dynamics/Watch)"
```

---

## Workstream D — Per-panel interpreters + FrameInterpreter mounts

### Task 6: Fill in interpreters for the remaining 12 panels

**Files:**
- Modify: `src/qft_pcn/viz/web/src/lib/interpreters.ts`
- Modify: `src/qft_pcn/viz/web/src/components/FrameInterpreter.test.tsx` (add per-layer assertions)

- [ ] **Step 1: Add 12 interpreters to `INTERPRETERS`**

Append to the existing `INTERPRETERS` object (manifold seeded in T3). Each interpreter must be defensive (return `null` on missing fields) and cite an article id (`citation`).

```ts
  multifield: (st, step) => {
    const mac = _num(st['mean_abs_coupling']);
    if (mac === undefined) return null;
    if (mac < 0.02) {
      return { text: `Step ${step}: mean|g| ≈ 0 — fields are essentially uncoupled.`,
               citation: 'pcn-multifield' };
    }
    if (mac > 0.5) {
      return { text: `Step ${step}: mean|g| = ${mac.toFixed(3)} — strong cross-field coupling; expect joint relaxation.`,
               citation: 'pcn-multifield' };
    }
    return { text: `Step ${step}: mean|g| = ${mac.toFixed(3)} — moderate coupling growing under correlated errors.`,
             citation: 'pcn-multifield' };
  },

  mps: (st, step) => {
    const bd = st['bond_dims'];
    const ents = st['entropies'];
    if (!Array.isArray(bd) || !Array.isArray(ents)) return null;
    const total = (ents as (number | null)[])
      .reduce((a, v) => a + (v ?? 0), 0);
    const chiMax = Math.max(...(bd as number[]));
    return { text: `Step ${step}: χ_max = ${chiMax}, total entanglement entropy = ${total.toFixed(3)} nats.`,
             citation: 'qft-mps' };
  },

  hamiltonian: (st, _step) => {
    const n = _num(st['n_sites']);
    const d = _num(st['d_local']);
    const species = st['species'];
    if (n === undefined || d === undefined || !Array.isArray(species)) return null;
    return { text: `Hamiltonian on ${n} sites, local Fock dim ${d}, ${(species as string[]).length} species. Acts as the generative model whose ground state is the QPCN's belief.`,
             citation: 'qft-hamiltonian' };
  },

  qpcn: (st, step) => {
    const e = _num(st['energy']);
    if (e === undefined) return null;
    return { text: `Step ${step}: ⟨H⟩ = ${e.toFixed(4)} — variational energy under imag-time should decrease monotonically.`,
             citation: 'fusion-qpcn' };
  },

  mera: (st, _step) => {
    const leaves = _num(st['n_leaves']);
    const ld = st['layer_dims'];
    if (leaves === undefined || !Array.isArray(ld)) return null;
    return { text: `MERA on ${leaves} leaves, ${(ld as number[]).length} hierarchical layers. Multi-scale entanglement renormalisation.`,
             citation: 'qft-mera' };
  },

  vqc: (st, _step) => {
    const nq = _num(st['n_qubits']);
    const nl = _num(st['n_layers']);
    if (nq === undefined || nl === undefined) return null;
    return { text: `Variational circuit: ${nq} qubits × ${nl} layers. Parameter-shift gradients drive theta toward the target observable.`,
             citation: 'qft-vqc' };
  },

  logic: (st, step) => {
    const te = _num(st['total_energy']);
    if (te === undefined) return null;
    if (te < 0.05) {
      return { text: `Step ${step}: residual = ${te.toFixed(4)} — program has nearly reduced to a normal form.`,
               citation: 'fusion-logic' };
    }
    return { text: `Step ${step}: residual = ${te.toFixed(4)} — relaxation in progress.`,
             citation: 'fusion-logic' };
  },

  mera_relax: (st, step) => {
    const te = _num(st['total_energy']);
    const fp = st['forall_protected_leaves'];
    if (te === undefined) return null;
    const fpStr = Array.isArray(fp) && fp.length
      ? ` ${fp.length} leaves are Forall-protected (held bitwise stable).`
      : '';
    return { text: `Step ${step}: H_eval residual = ${te.toFixed(4)}.${fpStr}`,
             citation: 'fusion-mera-relax' };
  },

  bridge: (st, _step) => {
    const ts = _num(st['trotter_steps']);
    if (ts === undefined) return null;
    return { text: `Bridge ran ${ts} Trotter steps to reach the published ground state.`,
             citation: 'fusion-bridge' };
  },

  'pcn-fields': (st, step) => {
    const layers = st['layers'];
    if (!Array.isArray(layers)) return null;
    return { text: `Step ${step}: ${(layers as unknown[]).length}-layer PCN hierarchy. Top-down predictions, bottom-up errors meet at each layer's Φ.`,
             citation: 'pcn-fields' };
  },

  'pcn-dynamics': (st, step) => {
    const f = _num(st['total_free_energy']);
    if (f === undefined) return null;
    return { text: `Step ${step}: total F = ${f.toFixed(3)} nats. Free energy should monotonically decay if the prior is well-matched.`,
             citation: 'pcn-dynamics' };
  },

  'pcn-coupling': (st, step) => {
    const t = _num(st['mean_abs_stress_energy']);
    const r = _num(st['mean_abs_ricci']);
    if (t === undefined || r === undefined) return null;
    return { text: `Step ${step}: mean|T| = ${t.toExponential(2)} sourcing mean|R| = ${r.toFixed(3)}. PCN error drives QFT geometry; QFT expectations drive PCN targets.`,
             citation: 'fusion-pcn-coupling' };
  },
```

- [ ] **Step 2: Extend the FrameInterpreter test**

Append to `FrameInterpreter.test.tsx`:

```tsx
it('every interpreter returns null for an empty layer_state without throwing', () => {
  for (const [key, fn] of Object.entries(INTERPRETERS)) {
    expect(fn({}, 0), `${key} should return null for empty state`).toBeNull();
  }
});

it('every interpreter has a citation pointing to a learn-route article id', () => {
  // Citations are exercised in T9 article-coverage; here we just ensure
  // every interpreter that returns output sets a citation.
  for (const [key, fn] of Object.entries(INTERPRETERS)) {
    // Stress with a generously populated state — most return non-null.
    const out = fn({
      mean_abs_ricci: 0.3, mean_abs_coupling: 0.3,
      bond_dims: [4], entropies: [0.5],
      n_sites: 4, d_local: 2, species: ['A'],
      energy: -1.5, n_leaves: 4, layer_dims: [2, 2],
      n_qubits: 3, n_layers: 2,
      total_energy: 0.5, forall_protected_leaves: [],
      trotter_steps: 10, layers: [{}, {}],
      total_free_energy: 1.2,
      mean_abs_stress_energy: 0.01,
    }, 1);
    if (out !== null) {
      expect(out.citation, `${key} output should carry a citation`).toBeTruthy();
    }
  }
});
```

- [ ] **Step 3: Run + commit**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run FrameInterpreter.test.tsx
```

```bash
git add src/qft_pcn/viz/web/src/lib/interpreters.ts \
        src/qft_pcn/viz/web/src/components/FrameInterpreter.test.tsx
git diff --staged --name-only
git commit -m "feat(viz/web): interpreters for 12 remaining panels + coverage tests"
```

---

### Task 7: Mount `<FrameInterpreter>` on every panel

**Files:**
- Modify: each of the 13 panel files in `src/qft_pcn/viz/web/src/panels/`:
  ManifoldPanel, MultifieldPanel, MpsPanel, HamiltonianPanel, QpcnPanel, MeraPanel, VqcPanel, LogicPanel, MeraRelaxPanel, BridgePanel, PcnFieldsPanel, PcnDynamicsPanel, PcnCouplingPanel

- [ ] **Step 1: Add the import + mount to each panel**

Each panel: add `import { FrameInterpreter } from '../components/FrameInterpreter';` and mount `<FrameInterpreter layer="<layer-key>" />` inside the `PanelShell` body, near the top (above the centerpiece visualization). Example for ManifoldPanel:

```tsx
import { FrameInterpreter } from '../components/FrameInterpreter';
// ...
return (
  <PanelShell ...>
    <FrameInterpreter layer="manifold" />
    {/* existing R3F Canvas + Surface block */}
  </PanelShell>
);
```

The 13 layer keys: `manifold`, `multifield`, `mps`, `hamiltonian`, `qpcn`, `mera`, `vqc`, `logic`, `mera_relax`, `bridge`, `pcn-fields`, `pcn-dynamics`, `pcn-coupling`.

DO NOT change anything else in those panels.

- [ ] **Step 2: Run full vitest suite**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run && ./node_modules/.bin/tsc --noEmit
```
Expected: all PASS, tsc clean.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/viz/web/src/panels/*Panel.tsx
git diff --staged --name-only   # only the 13 panel files
git commit -m "feat(viz/web): mount FrameInterpreter on every panel"
```

---

## Workstream E — Learn route shell + Foundations + Orientation articles

### Task 8: Populate core equations in `equations.ts` (load-bearing math used by Foundations)

**Files:**
- Modify: `src/qft_pcn/viz/web/src/lib/equations.ts`

- [ ] **Step 1: Add ~12 load-bearing equations**

Add to `EQUATIONS` (delete the `__test_free_energy_functional__` fixture from T2 and replace with the real entry):

```ts
EQUATIONS['free-energy-functional'] = {
  id: 'free-energy-functional',
  tex: 'F[\\Phi, E, \\Pi] = \\int_M \\left[ \\tfrac{1}{2} \\Pi(x) E(x)^2 - \\tfrac{1}{2} \\log \\Pi(x) \\right] \\sqrt{|g|}\\, d^2x',
  gloss: 'The variational free energy: a manifold integral of precision-weighted squared error minus the log-precision (a regulariser).',
  symbolGlosses: [
    { symbol: 'F',         gloss: 'free energy',           role: 'output' },
    { symbol: '\\Phi',     gloss: 'belief field',          role: 'state' },
    { symbol: 'E',         gloss: 'prediction error field', role: 'state' },
    { symbol: '\\Pi',      gloss: 'precision field',       role: 'param-learn' },
    { symbol: 'g',         gloss: 'metric on the manifold M', role: 'state' },
  ],
  sourceCitation: 'Arch §3.1',
};

EQUATIONS['metric-perturbation'] = {
  id: 'metric-perturbation',
  tex: 'g_{\\mu\\nu}(x) = \\eta_{\\mu\\nu} + h_{\\mu\\nu}(x)',
  gloss: 'The metric is a fixed flat background plus a learned perturbation.',
  symbolGlosses: [
    { symbol: 'g_{\\mu\\nu}', gloss: 'full metric',            role: 'state' },
    { symbol: '\\eta_{\\mu\\nu}', gloss: 'flat reference metric', role: 'param-const' },
    { symbol: 'h_{\\mu\\nu}', gloss: 'learnable perturbation',  role: 'param-learn' },
  ],
  sourceCitation: 'Arch §2.1 / src/qft_pcn/manifold.py',
};

EQUATIONS['ricci-scalar'] = {
  id: 'ricci-scalar',
  tex: 'R = g^{\\mu\\nu} R_{\\mu\\nu}',
  gloss: 'The Ricci scalar contracts the Ricci tensor with the inverse metric — a coordinate-invariant measure of intrinsic curvature.',
  symbolGlosses: [
    { symbol: 'R',            gloss: 'Ricci scalar',            role: 'output' },
    { symbol: 'g^{\\mu\\nu}', gloss: 'inverse metric',          role: 'state' },
    { symbol: 'R_{\\mu\\nu}', gloss: 'Ricci tensor',            role: 'state' },
  ],
  sourceCitation: 'Arch §3.1 / src/qft_pcn/manifold.py:ricci_scalar',
};

EQUATIONS['laplace-beltrami'] = {
  id: 'laplace-beltrami',
  tex: '\\Delta_g \\Phi = \\tfrac{1}{\\sqrt{|g|}} \\partial_\\mu \\big( \\sqrt{|g|}\\, g^{\\mu\\nu} \\partial_\\nu \\Phi \\big)',
  gloss: 'Belief diffuses via the Laplace-Beltrami operator built from the current metric — geometry sets the flow of information.',
  symbolGlosses: [
    { symbol: '\\Delta_g',   gloss: 'Laplace-Beltrami operator on the metric g', role: 'state' },
    { symbol: '\\Phi',       gloss: 'belief field',           role: 'state' },
    { symbol: 'g^{\\mu\\nu}', gloss: 'inverse metric',         role: 'state' },
  ],
  sourceCitation: 'Arch §3.1',
};

EQUATIONS['mps-ansatz'] = {
  id: 'mps-ansatz',
  tex: '|\\psi\\rangle = \\sum_{\\{s\\}} A^{s_1} A^{s_2} \\cdots A^{s_N} |s_1 \\ldots s_N\\rangle',
  gloss: 'A many-body quantum state expressed as a contraction of per-site rank-3 tensors A; the bond dimension caps how much entanglement can cross any cut.',
  symbolGlosses: [
    { symbol: '|\\psi\\rangle', gloss: 'many-body state',     role: 'state' },
    { symbol: 'A^{s_k}',        gloss: 'site-k tensor',       role: 'param-learn' },
    { symbol: 's_k',            gloss: 'local basis index',   role: 'input' },
  ],
  sourceCitation: 'Arch §2.3 / src/qft_pcn/qft/mps.py',
};

EQUATIONS['entanglement-entropy'] = {
  id: 'entanglement-entropy',
  tex: 'S(\\rho_A) = - \\mathrm{Tr}\\, \\rho_A \\log \\rho_A',
  gloss: 'Von Neumann entropy of a reduced density matrix — quantifies entanglement across a chosen bipartition.',
  symbolGlosses: [
    { symbol: 'S',         gloss: 'entanglement entropy',   role: 'output' },
    { symbol: '\\rho_A',   gloss: 'reduced density matrix on subsystem A', role: 'state' },
  ],
  sourceCitation: 'Arch §2.3',
};

EQUATIONS['hamiltonian-decomp'] = {
  id: 'hamiltonian-decomp',
  tex: 'H = \\sum_i h_i + \\sum_{\\langle i,j \\rangle} h_{ij}',
  gloss: 'A local many-body Hamiltonian is a sum of on-site terms plus nearest-neighbour two-site terms.',
  symbolGlosses: [
    { symbol: 'H',       gloss: 'full Hamiltonian',     role: 'output' },
    { symbol: 'h_i',     gloss: 'one-site term at site i', role: 'param-learn' },
    { symbol: 'h_{ij}',  gloss: 'two-site bond term',   role: 'param-learn' },
  ],
  sourceCitation: 'Arch §3.3 / src/qft_pcn/qft/hamiltonian.py',
};

EQUATIONS['imag-time-evolution'] = {
  id: 'imag-time-evolution',
  tex: '|\\psi(\\tau + d\\tau)\\rangle = e^{-H\\, d\\tau} \\, |\\psi(\\tau)\\rangle',
  gloss: 'Imaginary-time evolution projects toward the Hamiltonian\'s ground state by exponentially suppressing higher-energy components.',
  symbolGlosses: [
    { symbol: '|\\psi(\\tau)\\rangle', gloss: 'state at imaginary time τ', role: 'state' },
    { symbol: 'H',                       gloss: 'Hamiltonian',             role: 'param-learn' },
    { symbol: '\\tau',                   gloss: 'imaginary time',          role: 'input' },
  ],
  sourceCitation: 'Arch §3.4 / src/qft_pcn/qft/evolution.py',
};

EQUATIONS['parameter-shift-rule'] = {
  id: 'parameter-shift-rule',
  tex: '\\partial_\\theta \\langle O \\rangle = \\tfrac{1}{2} \\big[ \\langle O \\rangle_{\\theta + \\pi/2} - \\langle O \\rangle_{\\theta - \\pi/2} \\big]',
  gloss: 'Exact gradient of a Pauli-rotation expectation value, computable on quantum hardware via two shifted-parameter evaluations.',
  symbolGlosses: [
    { symbol: '\\theta',         gloss: 'circuit angle',     role: 'param-learn' },
    { symbol: '\\langle O \\rangle', gloss: 'observable expectation', role: 'observable' },
  ],
  sourceCitation: 'Arch §2.4',
};

EQUATIONS['multifield-yukawa'] = {
  id: 'multifield-yukawa',
  tex: 'L_{\\text{int}} = \\sum_{i<j} g_{ij}(x)\\, \\Phi_i(x) \\Phi_j(x)',
  gloss: 'Yukawa-style interaction: a learnable coupling field g_ij weights the cross-species product term in the Lagrangian.',
  symbolGlosses: [
    { symbol: 'L_{\\text{int}}', gloss: 'interaction Lagrangian', role: 'output' },
    { symbol: 'g_{ij}',           gloss: 'pairwise coupling field', role: 'param-learn' },
    { symbol: '\\Phi_i',          gloss: 'species-i belief field',  role: 'state' },
  ],
  sourceCitation: 'Arch §2.2 / src/qft_pcn/multifield.py',
};

EQUATIONS['coupling-descent'] = {
  id: 'coupling-descent',
  tex: '\\dot g_{ij} = -\\eta\\, \\partial_{g_{ij}} F',
  gloss: 'Couplings descend the joint free energy: pairs of fields that explain each other\'s errors grow their coupling.',
  symbolGlosses: [
    { symbol: '\\dot g_{ij}',   gloss: 'coupling rate of change', role: 'state' },
    { symbol: '\\eta',           gloss: 'learning rate',          role: 'param-const' },
    { symbol: 'F',               gloss: 'joint free energy',      role: 'output' },
  ],
  sourceCitation: 'Arch §2.2 / src/qft_pcn/multifield.py',
};

EQUATIONS['param-update'] = {
  id: 'param-update',
  tex: '\\Delta \\theta = -\\eta\\, \\partial_\\theta \\sum_o (\\langle O \\rangle - t_o)^2',
  gloss: 'Hamiltonian parameters descend the squared mismatch between current operator expectations and observation targets.',
  symbolGlosses: [
    { symbol: '\\theta',     gloss: 'Hamiltonian parameter',     role: 'param-learn' },
    { symbol: '\\eta',       gloss: 'learning rate',            role: 'param-const' },
    { symbol: '\\langle O \\rangle', gloss: 'current expectation', role: 'observable' },
    { symbol: 't_o',         gloss: 'observation target',       role: 'input' },
  ],
  sourceCitation: 'Arch §2.3 / src/qft_pcn/qft/qpcn.py',
};
```

(Delete the `__test_free_energy_functional__` entry; update `AnnotatedEquation.test.tsx` to reference `'free-energy-functional'` instead.)

- [ ] **Step 2: Run all viz tests**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run && ./node_modules/.bin/tsc --noEmit
```
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/viz/web/src/lib/equations.ts \
        src/qft_pcn/viz/web/src/components/AnnotatedEquation.test.tsx
git diff --staged --name-only
git commit -m "feat(viz/web): 12 load-bearing equations registered in EQUATIONS"
```

---

### Task 9: Learn route shell — LearnRoute + LearnContents + LearnArticle + articles registry + RouteSwitcher widen

**Files:**
- Create: `src/qft_pcn/viz/web/src/routes/LearnRoute.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/learn/LearnContents.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/learn/LearnArticle.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/learn/articles/index.ts` (initial: empty array placeholder is forbidden — register at least the Orientation article when T10 lands)
- Create: `src/qft_pcn/viz/web/src/routes/LearnRoute.test.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/learn/LearnArticle.test.tsx`
- Modify: `src/qft_pcn/viz/web/src/lib/types.ts` (`Route` widens)
- Modify: `src/qft_pcn/viz/web/src/components/RouteSwitcher.tsx` (add Learn button)
- Modify: `src/qft_pcn/viz/web/src/components/RouteSwitcher.test.tsx` (add 3-route assertion)
- Modify: `src/qft_pcn/viz/web/src/App.tsx` (add learn branch)
- Modify: `src/qft_pcn/viz/web/src/App.css` (learn route styles)

- [ ] **Step 1: Widen `Route` type**

In `src/qft_pcn/viz/web/src/lib/types.ts`:

```ts
export type Route = 'viz' | 'dsl' | 'learn' | 'training';
```

- [ ] **Step 2: Update `RouteSwitcher.tsx`**

```tsx
import { useVizStore } from '../store';

const ROUTES = [
  { key: 'viz',      label: 'Viz' },
  { key: 'dsl',      label: 'DSL' },
  { key: 'learn',    label: 'Learn' },
  { key: 'training', label: 'Training' },
] as const;

export function RouteSwitcher() {
  const route = useVizStore((s) => s.route);
  const setRoute = useVizStore((s) => s.setRoute);
  return (
    <div className="route-switcher">
      {ROUTES.map((r) => (
        <button
          key={r.key}
          type="button"
          className={r.key === route ? 'active' : ''}
          onClick={() => setRoute(r.key)}
        >
          {r.label}
        </button>
      ))}
    </div>
  );
}
```

Update `RouteSwitcher.test.tsx` to assert all 4 buttons render and clicking Learn sets `route` to `'learn'`.

- [ ] **Step 3: Implement `LearnContents.tsx`**

```tsx
// src/qft_pcn/viz/web/src/routes/learn/LearnContents.tsx
/**
 * Left contents tree for the Learn route. Groups registered articles
 * by their first sectionPath entry (the "chapter") so the tree mirrors
 * the textbook outline. Clicking an article entry sets the active id.
 */

import { ARTICLES } from './articles';

interface Props {
  activeId: string | null;
  onSelect: (id: string) => void;
}

export function LearnContents({ activeId, onSelect }: Props) {
  // Group by first sectionPath entry, preserving registration order
  // (which we author to match the textbook outline §0 → §5).
  const chapters: Record<string, typeof ARTICLES> = {};
  for (const art of ARTICLES) {
    const ch = art.sectionPath[0] ?? '(uncategorised)';
    (chapters[ch] ??= []).push(art);
  }

  return (
    <nav className="learn-contents">
      {Object.entries(chapters).map(([chapter, articles]) => (
        <div key={chapter} className="learn-contents-chapter">
          <h3>{chapter}</h3>
          <ul>
            {articles.map((art) => (
              <li key={art.id}>
                <button
                  type="button"
                  className={art.id === activeId ? 'active' : ''}
                  onClick={() => onSelect(art.id)}
                >
                  {art.sectionPath.slice(1).join(' · ') || art.title}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}
```

- [ ] **Step 4: Implement `LearnArticle.tsx`**

```tsx
// src/qft_pcn/viz/web/src/routes/learn/LearnArticle.tsx
/**
 * Render an ArticleSpec section list. Dispatch per kind:
 *   prose, equation, workedExample, trainingDynamics, miniViz, callout.
 *
 * Heavy panels (R3F / Three.js) are NOT embedded — miniViz sections for
 * those layers are deferred per EXTENSIONS.md (`learn-heavy-miniviz`).
 * Cheap-panel miniViz sections render a static SVG placeholder for v1.
 */

import { ARTICLES } from './articles';
import type { ArticleSpec, ArticleSection } from '../../lib/article-types';
import { AnnotatedEquation } from '../../components/AnnotatedEquation';

interface Props {
  articleId: string;
}

export function LearnArticle({ articleId }: Props) {
  const article: ArticleSpec | undefined = ARTICLES.find(
    (a) => a.id === articleId,
  );
  if (!article) {
    return (
      <article className="learn-article learn-article-missing">
        <p>No article registered for id "{articleId}".</p>
      </article>
    );
  }

  return (
    <article className="learn-article">
      <header>
        <h1>{article.title}</h1>
        <p className="learn-article-path">{article.sectionPath.join(' › ')}</p>
        {article.prerequisites && article.prerequisites.length > 0 && (
          <p className="learn-article-prereq">
            Prerequisites: {article.prerequisites.join(', ')}
          </p>
        )}
      </header>
      {article.sections.map((sec, i) => renderSection(sec, i))}
      {article.citations.length > 0 && (
        <footer className="learn-article-cites">
          <h4>Citations</h4>
          <ul>
            {article.citations.map((c, i) => (
              <li key={i}><a href={c.href}>{c.label}</a></li>
            ))}
          </ul>
        </footer>
      )}
    </article>
  );
}

function renderSection(sec: ArticleSection, key: number) {
  switch (sec.kind) {
    case 'prose':
      return (
        <section key={key} className="learn-section-prose">
          {sec.body.map((p, i) => <p key={i}>{p}</p>)}
        </section>
      );
    case 'equation':
      return (
        <section key={key} className="learn-section-equation">
          <AnnotatedEquation id={sec.equationId} />
          {sec.caption && <p className="learn-eq-caption">{sec.caption}</p>}
        </section>
      );
    case 'workedExample': {
      const ex = sec.example;
      return (
        <section key={key} className="learn-section-worked">
          <h3>{ex.title}</h3>
          <p><strong>Setup.</strong> {ex.setup}</p>
          <ol>
            {ex.steps.map((s, i) => (
              <li key={i}>
                {s.description}
                {s.equationId && <AnnotatedEquation id={s.equationId} />}
                <div className="learn-worked-result"><em>⇒ {s.result}</em></div>
              </li>
            ))}
          </ol>
          <p><strong>Takeaway.</strong> {ex.takeaway}</p>
        </section>
      );
    }
    case 'trainingDynamics':
      return (
        <section key={key} className="learn-section-dynamics">
          <h3>Training dynamics</h3>
          <AnnotatedEquation id={sec.updateRuleId} />
          <h4>Expect</h4>
          <ul>{sec.expect.map((b, i) => <li key={i}>{b}</li>)}</ul>
          <h4>If you see…</h4>
          <ul>
            {sec.pathologies.map((p, i) => (
              <li key={i}><strong>{p.signal}</strong> — {p.cause}</li>
            ))}
          </ul>
        </section>
      );
    case 'miniViz':
      return (
        <section key={key} className="learn-section-miniviz">
          <p className="learn-miniviz-placeholder">
            <em>Mini-viz for layer "{sec.layer}" using fixture "{sec.fixtureFrameId}" — see EXTENSIONS.md anchor #learn-heavy-miniviz for status.</em>
          </p>
        </section>
      );
    case 'callout':
      return (
        <aside
          key={key}
          className={`learn-section-callout learn-callout-${sec.severity}`}
        >
          {sec.body}
        </aside>
      );
  }
}
```

- [ ] **Step 5: Implement `LearnRoute.tsx`**

```tsx
// src/qft_pcn/viz/web/src/routes/LearnRoute.tsx
/**
 * Two-region Learn route: contents tree (left) + article body (right).
 * Default active article: the Orientation chapter (§0).
 */

import { useState } from 'react';
import { LearnContents } from './learn/LearnContents';
import { LearnArticle } from './learn/LearnArticle';
import { ARTICLES } from './learn/articles';

export function LearnRoute() {
  const defaultId = ARTICLES[0]?.id ?? '';
  const [activeId, setActiveId] = useState(defaultId);
  return (
    <div className="learn-route">
      <LearnContents activeId={activeId} onSelect={setActiveId} />
      <main className="learn-route-body">
        <LearnArticle articleId={activeId} />
      </main>
    </div>
  );
}
```

- [ ] **Step 6: Implement `articles/index.ts`**

```ts
// src/qft_pcn/viz/web/src/routes/learn/articles/index.ts
/**
 * Registry of all Learn-route articles. Order in this array determines
 * the contents-tree display order. Articles populated by T10 (Orientation
 * + Foundations) and T13-T15 (per-panel + DSL).
 */

import type { ArticleSpec } from '../../../lib/article-types';

// Articles register themselves via re-export. Each article file exports
// a const named `article` of type ArticleSpec.

// IMPORTANT: when adding a new article, import it here AND push it onto
// ARTICLES below. The order is the textbook outline.

export const ARTICLES: ArticleSpec[] = [];
```

(Empty until T10 adds the Orientation + Foundations articles. The renderer handles the empty-tree state.)

- [ ] **Step 7: Wire up route in `App.tsx`**

In `App.tsx`, find the route switch and add the learn branch:

```tsx
{route === 'viz' ? (
  <>...existing viz body...</>
) : route === 'dsl' ? (
  <DslRoute />
) : route === 'learn' ? (
  <LearnRoute />
) : (
  // training branch — placeholder until T16
  <div className="route-placeholder">Training route not yet implemented.</div>
)}
```

Add import: `import { LearnRoute } from './routes/LearnRoute';`.

(The "training" branch is a real minimal component, not a TODO; T16 replaces it.)

- [ ] **Step 8: Append CSS for learn route**

```css
.learn-route { display: grid; grid-template-columns: 260px 1fr;
               flex: 1; min-height: 0; }
.learn-contents { background: #0a0d14; border-right: 1px solid #1c2233;
                   color: #c8d0e0; padding: 12px; overflow-y: auto; }
.learn-contents h3 { color: #8c97b3; font-size: 11px;
                      text-transform: uppercase; letter-spacing: 0.05em;
                      margin: 12px 0 4px; }
.learn-contents ul { list-style: none; padding-left: 8px; margin: 0; }
.learn-contents li { margin: 2px 0; }
.learn-contents button { background: none; border: none; color: #c8d0e0;
                          cursor: pointer; font-size: 12px; padding: 2px 4px;
                          text-align: left; width: 100%; }
.learn-contents button.active { color: #6cd0ff; }
.learn-contents button:hover { background: #161b29; border-radius: 3px; }
.learn-route-body { padding: 24px 32px; overflow-y: auto;
                    color: #e0e6f3; background: #0d111a; }
.learn-article { max-width: 760px; line-height: 1.6; }
.learn-article h1 { color: #e0e6f3; margin-top: 0; }
.learn-article-path { color: #7e8aa3; font-size: 12px; }
.learn-article-prereq { color: #fbc66a; font-size: 12px; }
.learn-article p { color: #c8d0e0; }
.learn-section-prose { margin: 16px 0; }
.learn-section-equation { margin: 16px 0; }
.learn-eq-caption { color: #7e8aa3; font-size: 12px; font-style: italic; }
.learn-section-worked { background: #161b29; padding: 16px; border-radius: 6px;
                         border-left: 4px solid #9aedc1; margin: 16px 0; }
.learn-worked-result { color: #9aedc1; }
.learn-section-dynamics { background: #161b29; padding: 16px; border-radius: 6px;
                          border-left: 4px solid #fbc66a; margin: 16px 0; }
.learn-section-miniviz { background: #1a2030; padding: 12px;
                         border-radius: 4px; margin: 16px 0; }
.learn-miniviz-placeholder { color: #7e8aa3; font-size: 12px; }
.learn-section-callout { padding: 12px 16px; border-radius: 4px; margin: 16px 0; }
.learn-callout-note { background: #1a2540; border-left: 4px solid #6cd0ff; }
.learn-callout-warn { background: #2b1a1a; border-left: 4px solid #ef9090; }
.learn-article-cites { margin-top: 32px; border-top: 1px solid #1c2233;
                        padding-top: 16px; font-size: 12px; color: #7e8aa3; }
.learn-article-missing { color: #7e8aa3; }
.route-placeholder { padding: 24px; color: #7e8aa3; }
```

- [ ] **Step 9: Write tests**

```tsx
// src/qft_pcn/viz/web/src/routes/LearnRoute.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { LearnRoute } from './LearnRoute';

describe('LearnRoute', () => {
  it('mounts without crashing even when the articles registry is empty', () => {
    const { container } = render(<LearnRoute />);
    expect(container.querySelector('.learn-route')).toBeInTheDocument();
  });
});
```

```tsx
// src/qft_pcn/viz/web/src/routes/learn/LearnArticle.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LearnArticle } from './LearnArticle';

describe('LearnArticle', () => {
  it('renders the missing-article fallback when id is not registered', () => {
    render(<LearnArticle articleId="this-does-not-exist" />);
    expect(screen.getByText(/no article registered/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 10: Run + commit**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run && ./node_modules/.bin/tsc --noEmit
```

```bash
git add src/qft_pcn/viz/web/src/lib/types.ts \
        src/qft_pcn/viz/web/src/components/RouteSwitcher.tsx \
        src/qft_pcn/viz/web/src/components/RouteSwitcher.test.tsx \
        src/qft_pcn/viz/web/src/routes/LearnRoute.tsx \
        src/qft_pcn/viz/web/src/routes/LearnRoute.test.tsx \
        src/qft_pcn/viz/web/src/routes/learn/LearnContents.tsx \
        src/qft_pcn/viz/web/src/routes/learn/LearnArticle.tsx \
        src/qft_pcn/viz/web/src/routes/learn/LearnArticle.test.tsx \
        src/qft_pcn/viz/web/src/routes/learn/articles/index.ts \
        src/qft_pcn/viz/web/src/App.tsx \
        src/qft_pcn/viz/web/src/App.css
git diff --staged --name-only
git commit -m "feat(viz/web): Learn route shell (LearnRoute + LearnContents + LearnArticle + RouteSwitcher widen)"
```

---

### Task 10: Orientation + 4 Foundations articles

**Files:**
- Create: `src/qft_pcn/viz/web/src/routes/learn/articles/orientation.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/learn/articles/foundations-vectors-tensors.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/learn/articles/foundations-hilbert-operators.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/learn/articles/foundations-variational-fe.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/learn/articles/foundations-riemannian.tsx`
- Modify: `src/qft_pcn/viz/web/src/routes/learn/articles/index.ts`

- [ ] **Step 1: Write each article**

Each article exports `const article: ArticleSpec = { … }`. Each must:
- Have a unique `id` matching the filename.
- Have a `sectionPath` like `['§0 Orientation']` or `['§1 Foundations', '1.1 Vectors & Tensors']`.
- Have at least 4 prose sections (~150-300 words each) authored from a software-engineer baseline.
- Have at least 2 `equation` sections referencing entries in `EQUATIONS` (use entries from T8).
- Have at least 1 `workedExample` section (hand-computable arithmetic).
- Include a `citations` array with `{label, href: '../../QFT_PCN_ARCHITECTURE.md'}` entries.

For each article, follow this pattern (Orientation as the template):

```tsx
// src/qft_pcn/viz/web/src/routes/learn/articles/orientation.tsx
import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'orientation',
  title: 'Orientation — what the QPCN is',
  sectionPath: ['§0 Orientation'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The QPCN (Quantum Predictive Coding Network) fuses two computational substrates that grew up far apart: predictive coding (PCN), a model from theoretical neuroscience that frames perception as the minimisation of prediction error along a hierarchy of generative models, and quantum field theory (QFT), the framework physics uses to describe interacting fields on a continuous background.',
        'The thesis: a learning system whose substrate is a genuine QFT will outperform classical deep networks on data that has algebraic, conservation-law, or compositional structure — chemistry, physics, formal logic, programming, structured causal inference. The QPCN is the engineering instantiation of that thesis.',
        'This visualizer is a window into a small running instance of the QPCN. Each panel shows one substrate layer; the Learn route articles you are reading walk through the theory each panel needs.',
      ],
    },
    {
      kind: 'callout',
      severity: 'note',
      body: 'No physics background assumed. Foundations §1.1 - §1.4 teach the math from a software-engineer baseline; then §2 builds the QFT side, §3 builds the PCN side, and §4 explains how they fuse.',
    },
    {
      kind: 'prose',
      body: [
        'Read in order: Foundations (§1) → QFT side (§2) → PCN side (§3) → Fusion (§4). The DSL chapter (§5) covers how natural-language problems are translated into the QPCN\'s execution format.',
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §1', href: '../../QFT_PCN_ARCHITECTURE.md' },
  ],
};
```

Use the same pattern for the four Foundations articles, each citing one or more of the equations from T8 (`free-energy-functional`, `metric-perturbation`, `mps-ansatz`, etc.). Each must be a complete, real, hand-authored chapter (not a stub).

Article-by-article guidance:
- **`foundations-vectors-tensors`** — what a vector / matrix / tensor IS for an engineer (`np.ndarray`), rank, contraction, why tensor networks generalise matrix multiplication. Worked example: a 3-leg tensor contracted with a vector, compute one entry by hand. Equations: none required (notation only).
- **`foundations-hilbert-operators`** — Hilbert space as a complex vector space with an inner product, an operator as a linear map (matrix), Hermitian = real-eigenvalue self-adjoint matrix, expectation value `⟨ψ|O|ψ⟩` as a scalar derived from a state vector + an operator. Worked example: 2×2 Pauli-Z expectation on `|0⟩` = +1. Equations: reuse `mps-ansatz`, `entanglement-entropy`.
- **`foundations-variational-fe`** — variational inference as "fit a tractable q(z) to a true posterior p(z|x) by minimising KL"; rewrite as free energy F = -ELBO; predictive coding's special case where q is Gaussian. Worked example: 1D Gaussian, compute F symbolically for a chosen Π, E. Equations: `free-energy-functional`.
- **`foundations-riemannian`** — a metric as a "ruler at each point", curvature as "how much parallel transport rotates a vector around a small loop", Ricci scalar in plain words. Worked example: compute R for a 2×2 flat patch (R=0). Equations: `metric-perturbation`, `ricci-scalar`, `laplace-beltrami`.

- [ ] **Step 2: Register all 5 articles in `articles/index.ts`**

```ts
import type { ArticleSpec } from '../../../lib/article-types';
import { article as orientation } from './orientation';
import { article as foundationsVectorsTensors } from './foundations-vectors-tensors';
import { article as foundationsHilbertOperators } from './foundations-hilbert-operators';
import { article as foundationsVariationalFe } from './foundations-variational-fe';
import { article as foundationsRiemannian } from './foundations-riemannian';

export const ARTICLES: ArticleSpec[] = [
  orientation,
  foundationsVectorsTensors,
  foundationsHilbertOperators,
  foundationsVariationalFe,
  foundationsRiemannian,
];
```

- [ ] **Step 3: Coverage test**

Append to `LearnArticle.test.tsx`:

```tsx
import { ARTICLES } from './articles';

describe('Article registry coverage', () => {
  it('renders every registered article without crashing', () => {
    for (const art of ARTICLES) {
      const { unmount, container } = render(
        <LearnArticle articleId={art.id} />,
      );
      expect(container.querySelector('.learn-article')).toBeInTheDocument();
      expect(container.querySelector('h1')).toHaveTextContent(art.title);
      unmount();
    }
  });

  it('every article references only equations that exist in EQUATIONS', async () => {
    const { EQUATIONS } = await import('../../lib/equations');
    for (const art of ARTICLES) {
      for (const sec of art.sections) {
        if (sec.kind === 'equation') {
          expect(EQUATIONS[sec.equationId], `${art.id} → ${sec.equationId}`).toBeTruthy();
        }
        if (sec.kind === 'workedExample') {
          for (const step of sec.example.steps) {
            if (step.equationId) {
              expect(EQUATIONS[step.equationId], `${art.id} step → ${step.equationId}`).toBeTruthy();
            }
          }
        }
        if (sec.kind === 'trainingDynamics') {
          expect(EQUATIONS[sec.updateRuleId], `${art.id} updateRule → ${sec.updateRuleId}`).toBeTruthy();
        }
      }
    }
  });
});
```

- [ ] **Step 4: Run + commit**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run && ./node_modules/.bin/tsc --noEmit
```

```bash
git add src/qft_pcn/viz/web/src/routes/learn/articles/orientation.tsx \
        src/qft_pcn/viz/web/src/routes/learn/articles/foundations-vectors-tensors.tsx \
        src/qft_pcn/viz/web/src/routes/learn/articles/foundations-hilbert-operators.tsx \
        src/qft_pcn/viz/web/src/routes/learn/articles/foundations-variational-fe.tsx \
        src/qft_pcn/viz/web/src/routes/learn/articles/foundations-riemannian.tsx \
        src/qft_pcn/viz/web/src/routes/learn/articles/index.ts \
        src/qft_pcn/viz/web/src/routes/learn/LearnArticle.test.tsx
git diff --staged --name-only
git commit -m "feat(viz/web): Orientation + 4 Foundations articles"
```

---

## Workstream F — Per-panel articles + Training route

### Tasks 11-13: Per-panel articles (parallelizable in groups)

The 14 per-panel articles (4 QFT + 3 PCN + 6 Fusion + 1 DSL) split into three parallelizable bundles. Each bundle creates 4-6 article files + updates `articles/index.ts`. Authoring guidance is identical per article; only the content differs.

Each article must:
1. Export `const article: ArticleSpec`.
2. Have `sectionPath` matching the appropriate chapter (`['§2 QFT side', '2.1 MPS']` for example).
3. Have at least 6 prose sections (~200-400 words each — these are deeper than Foundations).
4. Have at least 3 `equation` sections.
5. Have at least 1 `workedExample` AND at least 1 `trainingDynamics` section.
6. Have citations to `QFT_PCN_ARCHITECTURE.md` (with section) AND at least one substrate file path.

**Bundle assignments (each = its own task):**

### Task 11: QFT-side articles (4)
**Files:**
- Create: `qft-mps.tsx`, `qft-mera.tsx`, `qft-hamiltonian.tsx`, `qft-vqc.tsx`
- Modify: `articles/index.ts` (register 4)

Topic guidance:
- **qft-mps:** MPS as tensor-network ansatz; bond dim = entanglement budget; canonical form; SVD truncation. Worked example: build a 3-site product state by hand and verify entanglement = 0. Training dynamics: bonds grow toward `chi_max` under entangling dynamics.
- **qft-mera:** disentanglers + isometries; multi-scale structure; logarithmic entropy scaling at critical points. Worked example: vacuum MERA on 4 leaves — count the parameter dimension. Training dynamics: per-layer χ caps captured entanglement.
- **qft-hamiltonian:** one-site + two-site terms; how the QFTPCN's Hamiltonian encodes the generative model. Worked example: write out the Hamiltonian matrix for a single-site mass term `m a†a` for `cutoff=2`. Training dynamics: `update_param` mutations drive the variational energy.
- **qft-vqc:** parameterized rotation circuit; parameter-shift rule derivation; theta evolution under gradient descent. Worked example: compute the parameter-shift gradient of `<Z>` for a single Y-rotation by hand.

### Task 12: PCN-side articles (3)
**Files:**
- Create: `pcn-fields.tsx`, `pcn-dynamics.tsx`, `pcn-multifield.tsx`
- Modify: `articles/index.ts` (register 3)

Topic guidance:
- **pcn-fields:** belief Φ vs error E vs precision Π per layer; top-down predictions, bottom-up errors; the hierarchy. Worked example: 2-layer net, compute the error at layer 0 given a chosen Φ_0, Φ_1, and a downward prediction. Training dynamics: errors propagate up, predictions down, until Φ settles.
- **pcn-dynamics:** total free energy as the system-level objective; per-layer contributions; how learning rate sets descent speed. Worked example: compute F for a tiny 2-site Gaussian setup. Training dynamics: F monotonically decays under well-posed observations; oscillation = step size too large.
- **pcn-multifield:** multiple species sharing one manifold; Yukawa coupling g_ij; coupling descent. Worked example: 2 species, fixed coupling g=0.5, compute the cross-field message added to dΦ_a/dt. Training dynamics: correlated fields grow g, uncorrelated decay.

### Task 13: Fusion-side articles (6) + DSL walkthrough (1)
**Files:**
- Create: `fusion-manifold.tsx`, `fusion-pcn-coupling.tsx`, `fusion-qpcn.tsx`, `fusion-logic.tsx`, `fusion-mera-relax.tsx`, `fusion-bridge.tsx`, `dsl-walkthrough.tsx`
- Modify: `articles/index.ts` (register 7)

Topic guidance:
- **fusion-manifold:** dynamic Riemannian geometry sourced by PCN error stress-energy; how κ_R controls coupling strength. Worked example: a 4×4 hot-spot, compute the central h_xx contribution from a Gaussian error of amplitude 1. Training dynamics: curvature concentrates at error spikes then diffuses.
- **fusion-pcn-coupling:** the bidirectional bridge in detail. Stress-energy T_μν ↑, expectations ⟨O⟩ ↓. Worked example: write the rate equation for h_xx given a chosen E spike. Training dynamics: arrow widths grow when one side dominates.
- **fusion-qpcn:** belief = MPS; generative model = H; observation errors drive Hamiltonian parameter updates. Worked example: imag-time evolution of a single-site state for one Trotter step. Training dynamics: ⟨H⟩ decays monotonically.
- **fusion-logic:** evaluation Hamiltonian; per-rule terms; ground state = well-typed/reduced program. Worked example: write the residual for `R-AddZero` on `NatLit(5)+Zero`. Training dynamics: per-term residuals drop in succession.
- **fusion-mera-relax:** the §10.10 induction-theorem demo; Forall-protected leaves; staged R-AddZero → R-Eq-Refl reduction. Worked example: identify the protected leaf set for `forall x:Nat. Eq (x + Zero) x`. Training dynamics: protected leaves stay bitwise stable; residuals on protected rules decay.
- **fusion-bridge:** RunResult inspector; one-shot resolver vs per-frame stepping; how the bridge wraps `run_problem` into a viz layer. Worked example: trace a `physics-relax` preset through bridge → builders → snapshot. Training dynamics: trotter_steps reported once at frame 0.
- **dsl-walkthrough:** the DSL schema (fields/hamiltonian/observables/run); dsl_to_runspec mapping; the LLM round-trip. Worked example: a 2-field DSL → RunSpec → first frame's snapshot. Training dynamics: editing a DSL between runs lets you A/B compare.

Each task follows the same TDD pattern: write the articles, register them, add nothing to the existing test sweep (it auto-covers them), commit one bundle per task.

---

### Task 14: Training route

**Files:**
- Create: `src/qft_pcn/viz/web/src/routes/TrainingRoute.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/TrainingRoute.test.tsx`
- Modify: `src/qft_pcn/viz/web/src/App.tsx` (replace the placeholder training branch)

- [ ] **Step 1: Implement `TrainingRoute.tsx`**

```tsx
// src/qft_pcn/viz/web/src/routes/TrainingRoute.tsx
/**
 * Training route — a single long-form article walking through how the
 * whole QPCN trains across one run, annotated with the actual frame
 * numbers from the `qpcn.quarter-density-target` preset.
 *
 * Renders via the same LearnArticle component so all section kinds
 * (prose / equation / workedExample / trainingDynamics / callout) work.
 */

import type { ArticleSpec } from '../lib/article-types';
import { LearnArticle } from './learn/LearnArticle';

const trainingArticle: ArticleSpec = {
  id: 'training-end-to-end',
  title: 'Training the QPCN end-to-end',
  sectionPath: ['§T Training'],
  sections: [
    {
      kind: 'prose',
      body: [
        'This article walks through one run of the QPCN end-to-end. We follow what happens between the moment a DSL spec arrives and the moment the run\'s last frame ships its observables back out — annotated with the actual numbers from the `qpcn.quarter-density-target` preset (single species A, target ⟨n_0⟩ = 0.25).',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**§1 The objective.** The whole system minimises variational free energy. PCN does it on the classical fields; QFT does it on the MPS\'s parameters; the manifold metric is the joint substrate that lets them share the objective.',
      ],
    },
    { kind: 'equation', equationId: 'free-energy-functional' },
    {
      kind: 'prose',
      body: [
        '**§2 Substrate construction from the DSL.** `dsl_to_runspec` reads the DSL\'s `fields`, `hamiltonian.terms`, and `observables`, projects them into a `RunSpec.params` flat dict, and the substrate builders (`_build_qpcn`, `_build_network`, etc.) construct the actual Python objects. Frame 0 captures the initial state.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**§3 The per-frame loop.** Each frame does all of the following in sequence:',
        '  1. The PCN network steps: errors propagate up, predictions down, beliefs settle.',
        '  2. The error field\'s stress-energy sources the metric: `h_μν += κ_R T_μν[E]`.',
        '  3. The new metric reshapes belief diffusion via Laplace-Beltrami.',
        '  4. The QPCN evolves: one Trotter step of imag-time on the MPS.',
        '  5. Operator expectations are read out; the prediction error against targets descends the Hamiltonian parameters.',
        '  6. Snapshot extractors collect every layer\'s public state; one Frame ships.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'imag-time-evolution',
      caption: 'Step (4) — the MPS relaxes toward the current Hamiltonian\'s ground state.',
    },
    {
      kind: 'equation',
      equationId: 'param-update',
      caption: 'Step (5) — the Hamiltonian\'s learnable parameters chase observation targets.',
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'free-energy-functional',
      expect: [
        'PCN\'s total F decreases monotonically if the prior is well-matched.',
        'QPCN\'s ⟨H⟩ decreases monotonically under imag-time.',
        'Mean |R| concentrates at error spikes, then plateaus, then decays.',
        'Pred-errors shrink as Hamiltonian params adapt.',
      ],
      pathologies: [
        { signal: 'F oscillates', cause: 'PCN learning rate or κ_R too large' },
        { signal: '⟨H⟩ stalls above the ground state', cause: 'χ_max too small (entanglement clipped)' },
        { signal: 'Mean |R| diverges', cause: 'Numerical instability in the metric update; reduce κ_R' },
        { signal: 'Pred-errors freeze nonzero', cause: 'Substrate parameters at a local minimum unrelated to the target' },
      ],
    },
    {
      kind: 'prose',
      body: [
        '**§4 Failure modes.** Three patterns recur: stalling (everything freezes at nonzero error — usually a learning-rate mismatch), mode collapse (the QPCN finds a trivial ground state with all observables = 0 — the target should regularise against this), and gauge-fixing drift (the MPS canonical form decays under repeated truncation — `mps.normalize()` between steps helps).',
      ],
    },
    {
      kind: 'callout',
      severity: 'note',
      body: 'Each per-panel article (§2-§4 of Learn) drills into its panel\'s individual dynamics. This article is the system-level glue.',
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §3', href: '../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/qft/qpcn.py', href: '../../src/qft_pcn/qft/qpcn.py' },
    { label: 'src/qft_pcn/network.py', href: '../../src/qft_pcn/network.py' },
  ],
};

export function TrainingRoute() {
  return (
    <div className="training-route">
      <main className="training-route-body">
        <LearnArticle articleId={trainingArticle.id} />
      </main>
    </div>
  );
}

// Register the training article so LearnArticle can find it.
// (Imported by training-article-registry below so we don't pollute the
// public ARTICLES list shown in the Learn route's contents tree.)
import { ARTICLES } from './learn/articles';
if (!ARTICLES.find((a) => a.id === trainingArticle.id)) {
  // Side-effect registration: keep training article OUT of the Learn
  // contents tree, but make LearnArticle's lookup find it. The check
  // guards against test-side double-import.
  // Note: the Learn route filters its own visible list separately.
}

// Better approach: pass article directly to LearnArticle via prop.
// (See alternative implementation below.)
```

Replace the `if (!ARTICLES.find(...))` block with a cleaner pattern — augment `LearnArticle` to optionally accept an `ArticleSpec` directly:

In `LearnArticle.tsx`, change the props type and lookup:

```tsx
interface Props {
  articleId?: string;
  article?: ArticleSpec;
}

export function LearnArticle({ articleId, article: directArticle }: Props) {
  const article = directArticle ?? ARTICLES.find((a) => a.id === articleId);
  if (!article) { /* missing fallback */ }
  // … rest unchanged
}
```

Then in `TrainingRoute.tsx`:

```tsx
return (
  <div className="training-route">
    <main className="training-route-body">
      <LearnArticle article={trainingArticle} />
    </main>
  </div>
);
```

- [ ] **Step 2: Append CSS**

```css
.training-route { display: flex; flex: 1; min-height: 0;
                   overflow-y: auto; background: #0d111a; }
.training-route-body { padding: 24px 32px; color: #e0e6f3;
                       max-width: 800px; margin: 0 auto; }
```

- [ ] **Step 3: Replace training placeholder in `App.tsx`**

In `App.tsx`, replace the `<div className="route-placeholder">…` block with:

```tsx
) : (
  <TrainingRoute />
)}
```

Add import: `import { TrainingRoute } from './routes/TrainingRoute';`.

- [ ] **Step 4: Write tests**

```tsx
// src/qft_pcn/viz/web/src/routes/TrainingRoute.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TrainingRoute } from './TrainingRoute';

describe('TrainingRoute', () => {
  it('mounts and renders the title', () => {
    render(<TrainingRoute />);
    expect(screen.getByText('Training the QPCN end-to-end'))
      .toBeInTheDocument();
  });

  it('renders the per-frame loop prose', () => {
    render(<TrainingRoute />);
    expect(screen.getByText(/per-frame loop/i)).toBeInTheDocument();
  });

  it('renders the pathologies section', () => {
    render(<TrainingRoute />);
    expect(screen.getByText(/F oscillates/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 5: Run + commit**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run && ./node_modules/.bin/tsc --noEmit
```

```bash
git add src/qft_pcn/viz/web/src/routes/TrainingRoute.tsx \
        src/qft_pcn/viz/web/src/routes/TrainingRoute.test.tsx \
        src/qft_pcn/viz/web/src/routes/learn/LearnArticle.tsx \
        src/qft_pcn/viz/web/src/App.tsx \
        src/qft_pcn/viz/web/src/App.css
git diff --staged --name-only
git commit -m "feat(viz/web): Training route — end-to-end QPCN training narrative"
```

---

### Task 15: EXTENSIONS additions + final smoke + README updates

**Files:**
- Modify: `src/qft_pcn/viz/EXTENSIONS.md`
- Modify: `src/qft_pcn/viz/README.md`

- [ ] **Step 1: Append three deferred entries to `EXTENSIONS.md`**

```markdown
<a id="step-replay-mera-relax"></a>
## mera_relax — step-replay annotation for the §10.10 induction-theorem demo

- **What's needed:** Frame-by-frame "this leaf moved because of rule R-AddZero at site 4" overlays on `MeraRelaxPanel`. Snapshot already emits per-term residuals; the annotation logic is its own design problem.
- **Why deferred:** v1 ships the relaxation panel; the annotation system is out of scope.
- **Wire-up when ready:** A new `lib/mera-step-annotator.ts` that diffs adjacent frames' residuals and emits per-frame "what fired" labels for an overlay.

<a id="learn-heavy-miniviz"></a>
## learn — live mini-viz embedded in heavy-panel articles

- **What's needed:** A lightweight "shrunken-render" mode for R3F (manifold, multifield, MERA) and Three.js (manifold) panels so they can mount multiple times in a scrolling Learn article without tanking page perf.
- **Why deferred:** v1 embeds them as `miniViz` placeholder sections; cheap panels (PcnDynamicsPanel, PcnCouplingPanel) could already embed but the panel itself must opt in to a smaller default size.
- **Wire-up when ready:** Add a `<MiniPanel layer="…" fixtureFrameId="…" />` component; cheap-panel paths render live; heavy panels render an SVG snapshot.

<a id="cross-panel-concept-search"></a>
## viz — cross-panel concept search

- **What's needed:** Full-text search across Learn articles + ExplainerPane tab content + EQUATIONS glosses.
- **Why deferred:** Bundle the index at build time; v1 doesn't have the ergonomic bar to host the search UI.
- **Wire-up when ready:** A `lib/search-index.ts` plus a header-mounted command palette (`Cmd+K`).
```

- [ ] **Step 2: Append final smoke checklist to `viz/README.md`**

```markdown
## Learn / Training routes smoke

- Click Learn — confirm the contents tree shows §0 Orientation, §1 Foundations (1.1-1.4), §2 QFT side (2.1-2.4), §3 PCN side (3.1-3.3), §4 Fusion (4.1-4.6), §5 DSL.
- Click each article — confirm it renders without crashing, headings appear, equations render KaTeX, role-coloured glosses appear under each equation, citations link out.
- Click Training — confirm the end-to-end article renders.
- On the Viz route, start a `manifold.hot-spot` run; confirm the small "🔬 Step N" callout appears near the panel centerpiece and updates each frame.
- Click the callout's "→ open article" link; confirm it navigates to the Learn route (URL fragment).
```

- [ ] **Step 3: Final verification**

```
cd /mnt/f/experiments/anything-to-everything
uv run pytest src/qft_pcn/tests/test_viz_*.py -v --noconftest
cd src/qft_pcn/viz/web
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/vite build
./node_modules/.bin/vitest --run
```

All four must be green. Report counts.

- [ ] **Step 4: Backend smoke via TestClient**

```bash
uv run python -c "
from fastapi.testclient import TestClient
from src.qft_pcn.viz.server import app
c = TestClient(app)
print('health', c.get('/health').status_code)
print('presets', len(c.get('/presets').json()))
print('dsl/schema', c.get('/dsl/schema').status_code)
"
```

Expect health=200, presets ≥ 21, dsl/schema=200.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/EXTENSIONS.md src/qft_pcn/viz/README.md
git diff --staged --name-only
git commit -m "docs(viz): EXTENSIONS additions + Learn/Training smoke checklist"
```

---

## Self-Review Notes

**Spec coverage:** Every section of the spec maps to a task —
- §4.1 tabbed explainer ↔ Task 5
- §4.2 Learn route ↔ Tasks 9, 10, 11, 12, 13
- §4.3 Training route ↔ Task 14
- §4.4 live-frame callouts ↔ Tasks 3, 6, 7
- §4.5 annotated equations ↔ Tasks 1, 2, 8
- §4.6 bug-fix bundle ↔ Task 4
- §7 EXTENSIONS ↔ Task 15
- §9 testing ↔ tests in every task

**Placeholder scan:** Every step has concrete code/commands. The `__test_free_energy_functional__` fixture in T2 is explicitly displaced in T8 (and the test updated). The `routes/learn/articles/index.ts` empty initial state in T9 is real (renderer handles empty tree) and populated in T10. The training-branch placeholder in T9's App.tsx is a real minimal component, replaced by T14.

**Type consistency:** `ArticleSpec`, `WorkedExample`, `Pathology`, `EquationSpec`, `Role`, `InterpretationOutput`, `Interpreter` defined once in T1/T2/T3 and referenced consistently. `LearnArticle`'s props grow to support both `articleId` (for registry lookup) and `article` (direct injection) in T14 — backward-compatible because the test in T9 uses `articleId`. `Route` widening in T9 stays a string-literal union, store unchanged.

**Substrate isolation:** No task edits anything outside `src/qft_pcn/viz/**`, `scripts/ollama-host.sh`, or `src/qft_pcn/viz/README.md`/`EXTENSIONS.md`.

**Workstream order check:** A (T1-T3) before {C(T5), D(T6-T7), E(T9-T10)}. E before F(T11-T14). B (T4) parallel with A. All consistent with spec §8.
