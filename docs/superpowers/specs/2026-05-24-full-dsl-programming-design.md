# Full §9.2 DSL + Programming-in-DSL Support — Design

**Date:** 2026-05-24
**Branch:** `claude/qft-pcn-hybrid-architecture-ihCIR`
**Scope:** Greenfield rewrite of the visualizer's DSL pipeline so the LLM frontend emits the architecture's §9.2 constraint-expression DSL and the QPCN consumes it as a program-synthesis substrate. Replaces the current physics-tuning DSL.
**Authority:** All schema, flow, and naming decisions trace to specific sections of `QFT_PCN_ARCHITECTURE.md`, cited inline.

---

## 1. Scope & Flow

### 1.1 What we're building

Full §9.2 / §15.3 DSL realization. Replaces today's physics-tuning DSL with the architecture's constraint-expression DSL — turning the visualizer's LLM pipeline into a program-synthesis frontend.

### 1.2 Operational flow (§9.3 verbatim, §10.5 bridge as the physical seam)

```
User (NL)
   ↓
LLM frontend  (viz/llm.py)
   ↓  emits §9.2 DSL spec  {fields, sites, boundary, constraints, observables, search}
   ↓
QPCN bridge  (src/qft_pcn/bridge/  — already exists per §10.5/§17)
   ↓
Hamiltonian compiler: H = Σ wᵢ Pᵢ from constraints  (each Pᵢ ⪰ 0, §13.2)
   ↓
Substrate run, routed by search.runtime:
     'mps'   → flat TEBD (qft/evolution.py)
     'mera'  → hierarchical TEBD (qft/mera_evolution.py, §10.4)
   ↓
Measurement → {observable_values, residual_constraint_energies}   ← §9.3 return shape
   ↓
LLM frontend verbalizes (residual = confidence; §15.12)
   ↓
NL answer
```

### 1.3 Architecture-pinned commitments

1. **Constraints ARE the intent** (§8/§10.3). No `intent` field.
2. **`search.runtime: 'mps' | 'mera'`** — explicit user choice. No auto-routing.
3. **No intermediate "result DSL"** — §9.3 return is the raw dict. The verbalizer reads observables + residuals directly.
4. **`use_lemma` is a day-one constraint kind** (§10.8). The composition layer (`src/qft_pcn/composition/`) already exists in the repo.
5. **Constraint kinds from §15.4** the canonical example uses: `well_typed_subtree`, `example`, `vocabulary`, plus the generic §9.2 `local` / `two_site` predicate kinds.
6. **§15 `length` synthesis IS the canonical acceptance test and the first preset** — 12 sites, χ_max=32, dt=0.05, 100 steps, expected ⟨H⟩≈0.0008, midpoint S≈2.37 bits.
7. **MERA is the default for recursive synthesis** — §10.4 gives bond dimension O(log N) vs MPS O(N).

### 1.4 What changes vs. current viz state

- Greenfield rewrite of `viz/dsl.py` + `viz/dsl-schema.json` (no backward compat for the constraint-DSL path).
- `viz/llm.py` system prompt rewritten around §9.2 schema + §15 as few-shot.
- Wire to existing `src/qft_pcn/bridge/` (do not reimplement the bridge).
- Predicates compiled via Pythonic-restricted AST (`ast.parse` + AllowedNodeVisitor + helper allowlist).
- Visualizer panels retained; new presets added; physics presets kept for per-layer pedagogy (see §3.2 for endpoint split).

---

## 2. DSL Schema & Predicate Language

### 2.1 Top-level schema (§9.2 + §15.3 grounded)

```json
{
  "version": "1",
  "fields":      [ <FieldSpec>, ... ],
  "sites":       <int>,
  "boundary":    { "<site_idx>": { "<field_name>": <value>, ... }, ... },
  "constraints": [ <Constraint>, ... ],
  "observables": [ <Observable>, ... ],
  "search":      <SearchSpec>
}
```

All keys required. Unknown keys reject at parse.

### 2.2 `FieldSpec`

```json
{ "name": "node_kind", "cutoff": 12 }
```

- `name`: string, snake_case, unique within `fields[]`.
- `cutoff`: int ≥ 2, per-species local Fock truncation `d` (§3.3.1).
- Per-site local Hilbert dimension = ∏ field.cutoff (§3.3.2). Validator rejects if product > `D_LOCAL=65536`.

§15.3 canonical: `node_kind:12, type:8, binder_id:16, value:8` → 12·8·16·8 = 12288 ≤ 65536 ✓.

### 2.3 `boundary`

String-keyed dict of site index → partial field assignment. Compiler clamps via strong projectors (§15.5). Values are concrete primitive labels (looked up against the encoder's vocabulary) or numeric literals matched against the `value` field.

### 2.4 `Constraint` kinds (closed enumeration)

Every constraint carries `weight: float > 0` (§9.2).

**(a) `local`** — predicate over one site (§9.2).
```json
{ "kind": "local", "site": 5, "term": "<predicate>", "weight": 1.0 }
```

**(b) `two_site`** — predicate over a pair of sites (§9.2).
```json
{ "kind": "two_site", "sites": [5, 7], "term": "<predicate>", "weight": 1.0 }
```

**(c) `well_typed_subtree`** — §15.3/§15.4. At AST root `root`, all §10.2 typing-rule projectors fire (T-Var, T-App, T-Abs).
```json
{ "kind": "well_typed_subtree", "root": 0, "weight": 10.0 }
```

**(d) `example`** — §15.3/§15.4. Input→output evaluation constraint instantiating the §10.3 evaluation Hamiltonian on auxiliary sites.
```json
{ "kind": "example", "input": "[]", "output": "0", "weight": 5.0 }
```

**(e) `vocabulary`** — §15.3. Restricts `node_kind` projectors at unclamped sites to listed primitives.
```json
{ "kind": "vocabulary", "primitives": ["Match", "Cons", "Nil", "Succ", "Zero", "Var", "App"] }
```

**(f) `use_lemma`** — §10.8. Day-one because `composition/lemma_library_adapter.py` and `composition/promoter.py` already exist.
```json
{ "kind": "use_lemma", "lemma_id": "length-base-case", "sites": [3, 4], "weight": 8.0 }
```

Compiler resolves `lemma_id` via the library adapter, clamps the listed sites to the cached MPS sub-state (or equivalently adds projector `−W|Ψ_L⟩⟨Ψ_L|` per §10.8).

### 2.5 `Observable`

§9.2 + §15.7:
```json
{ "site": 12, "field": "node_kind", "op": "argmax" }
```

`op` ∈ `{"argmax", "n", "phi", "expectation"}`. `argmax` returns the dominant basis label (AST decode §15.7). `n` returns ⟨n⟩. `phi` returns ⟨φ⟩. `expectation` requires `op_name` extension for arbitrary local ops.

### 2.6 `SearchSpec`

```json
{ "runtime": "mera", "steps": 100, "chi_max": 32, "dt": 0.05 }
```

- `runtime`: `'mps' | 'mera'` — explicit, no auto-routing.
- `steps`, `chi_max`, `dt`: all required.
- Optional: `dt_real`, `real_steps_per_observe` for §4.7.5 real-time mixing. Default omits (pure imaginary-time relaxation per §15.6).

### 2.7 Predicate language (`local` / `two_site` terms)

Pythonic-restricted, compiled via `ast.parse(term, mode='eval')` + AllowedNodeVisitor.

**Allowed AST nodes:** `Expression`, `BoolOp` (And/Or), `UnaryOp` (Not), `Compare` (==, !=, <, <=, >, >=, in), `BinOp` (Add, Sub, Mult), `Name`, `Constant`, `Call` (allowlisted helpers only), `Attribute` (only on bound site objects), `Subscript`, `Tuple`, `List`.

**Banned:** `Lambda`, `Import`, `ListComp`/`GeneratorExp`/`DictComp`, `IfExp`, assignment forms, `FormattedValue`, attribute access on disallowed names.

**Bound names — `local` (single site):**
- `expr` — alias for site's `node_kind` field value as a label string
- `type(expr)`, `value(expr)`, `binder(expr)` — field projectors
- Any field name from the spec, by bare identifier (`node_kind`, `type`, …)

**Bound names — `two_site` (sites `[i, j]`):**
- `arg1`, `arg2` — aliases for the two sites
- Same helpers applied to `arg1` / `arg2`

**Helper allowlist** (fixed, compiler imports nothing else):
- `type(x)`, `value(x)`, `binder(x)`, `kind(x)` — field projectors
- `int`, `str`, `bool`, `len` — built-in coercions/length

**Compilation:** each predicate becomes a positive-semidefinite projector `P = I − Π_satisfying` on the relevant 1- or 2-site local Hilbert space. The compiler enumerates basis labels satisfying the predicate (≤ ∏cutoff per site) and builds `Π_satisfying` as a sum of rank-1 projectors. Hermiticity verified at compile time (§15.4).

§9.2 examples re-validated:
- `expr_node == '+'` ✓
- `type(expr) == int` ✓
- `type(arg1) == type(arg2)` ✓

### 2.8 Return shape (§9.3 verbatim, plus diagnostics for the visualizer)

```json
{
  "observable_values":            { "<obs_index>": <value>, ... },
  "residual_constraint_energies": { "<constraint_index>": <float>, ... },
  "diagnostics": {
    "final_energy": <float>,
    "step_history": [ { "step": int, "tau": float, "energy": float, "max_bond": int, "midpoint_entropy": float }, ... ],
    "converged": <bool>
  }
}
```

`diagnostics` powers the §15.6 step-table view. `observable_values` + `residual_constraint_energies` are what the LLM consumes per §9.3.

### 2.9 Lemma-library wiring

- **Read path:** spec includes `use_lemma` → bridge resolves via `composition/lemma_library_adapter.py` → compiler installs projector before TEBD.
- **Write path:** after a successful run (residual < ε_register from §10.8), bridge surfaces the resulting state for registration. Visualizer offers a "register lemma" action on the run-complete panel; user supplies label + lemma_id, type signature is auto-derived from the decoded AST at observable sites.
- **No automatic registration** in v1 — keeps the library curated, mitigates cache pollution (§10.8 risks).

---

## 3. File Layout, Workstreams, Presets

### 3.1 File layout

**Visualizer side** — full rewrite of the DSL surface:
```
src/qft_pcn/viz/
├── dsl.py                           # NEW: §9.2 schema parser + validator
├── dsl_schema.json                  # NEW: JSON Schema for §2.1-§2.6
├── dsl_predicates.py                # NEW: AST visitor + helper allowlist + projector compiler (§2.7)
├── llm.py                           # REWRITE: system prompt rebuilt around §9.2 + §15 few-shot
├── llm_examples/
│   ├── length_synthesis.json        # NEW: §15.3 canonical
│   ├── peano_zero_axiom.json        # NEW: §10.8 lemma-promotion demo
│   └── stlc_id_function.json        # NEW: minimal STLC sanity
├── presets.py                       # UPDATE: register programming presets, retire/relocate physics-tuning
└── web/src/
    ├── lib/dsl-client.ts            # UPDATE: typed client for new schema
    ├── panels/DslEditorPanel.tsx    # UPDATE: Monaco wired to new JSON Schema
    ├── panels/LemmaLibraryPanel.tsx # NEW: read/register lemmas via bridge
    └── lib/run-decoder.ts           # NEW: parse §2.8 → AST decode for §15.8 view
```

**Bridge side** — additive, bridge layer already exists per §10.5/§17:
```
src/qft_pcn/bridge/
├── dsl.py                           # UPDATE: accept new schema on new endpoint
├── runtime.py                       # UPDATE: route on search.runtime, wire use_lemma → composition/
├── hamiltonian_compiler.py          # NEW: §10.2 typing rules + §10.3 eval Ham + six constraint-kind compilers
└── tests/
    ├── test_dsl_v1.py               # NEW: schema/predicate compiler unit tests
    ├── test_length_synthesis.py     # NEW: §15 end-to-end acceptance
    └── test_use_lemma.py            # NEW: §10.8 acceptance (Peano zero-axiom reuse)
```

**Composition layer** — already exists, no rewrites. Exercised via `use_lemma`.

### 3.2 Versioning at the bridge seam

- `POST /dsl/v1/run` — new, accepts §2.1 schema, returns §2.8 shape.
- `POST /dsl/legacy/run` — old physics DSL, kept temporarily for layer-exploration presets. Marked deprecated; removed once physics presets are ported or retired.

If a hard cut is preferred even for physics presets, drop `/dsl/legacy/run` and re-implement physics-tuning presets as canned `/run` calls bypassing the DSL. Either way the constraint-DSL pipeline is the only DSL the LLM emits.

### 3.3 Workstream decomposition (one-task-per-subagent batches)

**Layer A — foundations (parallel, no inter-deps):**
- **W1** — `viz/dsl.py` + `dsl_schema.json` + parser unit tests
- **W2** — `viz/dsl_predicates.py` + AST visitor + helper allowlist + projector unit tests
- **W3** — `bridge/hamiltonian_compiler.py` — typing rules (§10.2), eval Ham (§10.3), six constraint-kind compilers (§2.4), Hermiticity check (§15.4)

**Layer B — wiring (parallel, depends on A):**
- **W4** — `bridge/dsl.py` + `bridge/runtime.py` new endpoint, runtime routing (mps/mera), `use_lemma` → composition adapter
- **W5** — `viz/llm.py` rewrite + three `llm_examples/*.json` + Ollama prompt regression test

**Layer C — UI + acceptance (parallel, depends on B):**
- **W6a** — `viz/web/.../DslEditorPanel.tsx` Monaco rewire to new schema, `LemmaLibraryPanel.tsx`, `run-decoder.ts`, §15.8 AST decode view
- **W6b** — `bridge/tests/test_length_synthesis.py` (§15 acceptance), `test_use_lemma.py` (§10.8 acceptance), `test_dsl_v1.py` (parser/predicate edge cases)

Each subagent gets full Section 2 + relevant §X.Y architecture references, the anti-shortcut directive, no-placeholders directive, and explicit file ownership boundaries to prevent commit-race bleed.

### 3.4 Presets (day-one)

1. **`length-synthesis`** (§15 canonical) — 12 sites, MERA, χ_max=32, dt=0.05, 100 steps. Demonstrates typed synthesis, recursive AST via MERA, §15.6 monotone descent, §15.7 binder-id entanglement readout.
2. **`peano-zero-axiom`** (§10.8) — two-stage: stage 1 registers `∀x. x+0=x`; stage 2 proves `∀x. (x+0)+0=x` using `use_lemma` with fewer Trotter steps. Headline visualization: step-count delta.
3. **`stlc-id`** — minimal STLC sanity check, 4 sites, MPS. Smoke test.

Physics presets retained for per-layer pedagogy via `/dsl/legacy/run` or canned non-DSL calls per 3.2.

### 3.5 Acceptance gates

- §15 `length` synthesis preset converges to ⟨H⟩ < 0.01 within 100 steps on MERA; decoded AST type-checks classically (§15.9); all three examples (§15.11) pass.
- `use_lemma` preset shows ≥3× Trotter-step reduction on stage 2 vs stage 1 (§10.8 acceptance).
- Predicate compiler rejects every banned AST node in §2.7 with a clear error.
- All Layer B endpoints round-trip the §2.8 return shape; LLM verbalize consumes it directly with no intermediate transformation.

### 3.6 Out of scope (recorded in EXTENSIONS.md per no-placeholders directive)

- §12 extensions (anomaly, topological invariants, bootstrap, holographic codes, etc.) — orthogonal; tracked separately.
- Dependent types, effects, concurrency (§16.1, §16.3).
- Lemma auto-registration without curator review (§10.8 risk mitigation).
- §10.10 cross-level message passing surfaced as a programming-DSL feature — server-side support exists; UI exposure is a follow-on spec.
