# Spec: LLM Bridge / DSL Runtime for the QPCN

**Document type**: Implementation specification (sub-project G of the §10 roadmap).
**Date**: 2026-05-21.
**Author**: Brainstorming session, this branch.
**Status**: Approved design, ready for implementation plan.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Architecture doc**: `QFT_PCN_ARCHITECTURE.md` (root). This spec realizes §10.5.
**Acceptance owner**: human review of the end-to-end demo and protocol tests passing.

---

## 0. How to read this spec

This document is the contract for one sub-project. It exists because the §10 roadmap is decomposed into seven sub-projects (A–G); this is **sub-project G: LLM bridge / DSL runtime**. The other six are:

- A — AST ↔ MPS encoder/decoder (spec: `docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md`, largely implemented in `src/qft_pcn/logic/`).
- B — typing-rule Hamiltonian compiler (spec: `docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md`, *in flight*).
- C — evaluation Hamiltonian (spec: `docs/superpowers/specs/2026-05-21-evaluation-hamiltonian-design.md`, *in flight*).
- D — constraint debugger (spec: `docs/superpowers/specs/2026-05-21-constraint-debugger-design.md`, *in flight*).
- E — STLC synthesis demo (spec: `docs/superpowers/specs/2026-05-21-stlc-synthesis-design.md`, *in flight*).
- F — MERA upgrade (deferred; not required for G).

G is the **last** sub-project. After G, the QPCN system is end-to-end functional: NL goes in via an LLM, structured DSL goes through the bridge, the QPCN solves, observables come back, the LLM verbalizes. This sub-project is the only place LLM-SDK code lives in the entire repository.

Every section below is part of the contract. If something is missing that you need to decide while implementing, **stop and ask**. Do not fill in by guessing — see §1.

---

## 1. Driving principles (non-negotiable)

The architecture document's §1.3 thesis is that **the LLM and the QPCN are two systems with disjoint competences glued together by a small typed JSON DSL**. The point of the bridge is to make that gluing as thin and rigid as possible. A subagent will be tempted to make the bridge "helpful" — to do business logic on either side, to massage natural language into half-DSL, to retry, to fix up unconverged ground states with heuristics, to invent fields the LLM "obviously meant". Every one of those impulses is wrong. They re-couple the two systems, and the entire architectural advantage evaporates.

The following principles are non-negotiable. **None of them may be traded away for implementation simplicity.** If you find yourself tempted to violate one, stop and ask the human.

### 1.1 The DSL is the entire contract

The DSL (§3) is *the* interface between the LLM and the QPCN. The QPCN never sees natural language; the LLM never sees an MPS, a Hamiltonian, a bond dimension, or any tensor. The bridge does **not** translate prose into DSL — that is the LLM's job. The bridge does **not** verbalize DSL outputs into prose — that is the LLM's job. The bridge transports JSON in, executes the QPCN, transports JSON (and structured diagnostics) out.

The "easy shortcut" — letting the bridge accept a `"description": "find a function of type int → int that doubles its input"` field and silently `eval()` it into DSL terms via an in-process LLM call — is **rejected**. That would put LLM dependencies in the QPCN core's blast radius and destroy the contract.

### 1.2 The bridge is THIN

The bridge contains:

- A JSON schema validator (§3.7).
- A constraint-expression parser (§4.2) and term compiler (§4.3) that emits Hamiltonian terms via sub-project B/C's public APIs.
- A runner that calls sub-project A's `encode`/`decode`, builds the Hamiltonian via B/C, runs `evolve` from `qft/evolution.py`, measures observables, and returns the result.
- An optional D diagnostic call for `problem.diagnose`.
- A stdio JSON-RPC transport.

The bridge does **not** contain:

- Custom MPS code.
- Custom Hamiltonian-term math (it only *composes* terms produced by B and C).
- Custom observable-measurement code (it calls `MPS.local_expectation` and friends).
- Retry / fallback / repair logic. If imaginary-time evolution does not converge, the bridge returns the unconverged state's observables and the unconverged diagnostic. The LLM decides what to do.
- A "smart" mode where the bridge guesses fields, sites, or constraints. Missing required fields → 400-style error.

The "easy shortcut" — reimplementing a small Hamiltonian compiler inside the bridge so the bridge is "self-contained" — is **rejected**. The whole point of B's existence is to be the canonical Hamiltonian compiler; if G duplicates it, the two will drift, and the LLM will be debugging an off-by-one in the wrong file.

### 1.3 The QPCN core knows nothing of natural language

There is **no** LLM SDK import (`anthropic`, `openai`, etc.) anywhere under `src/qft_pcn/` except inside `src/qft_pcn/bridge/`. The classical PCN, multi-field, quantum, qft, and logic packages remain language-model-free. The dependency graph runs strictly:

```
src/qft_pcn/bridge/  ──►  src/qft_pcn/logic/  ──►  src/qft_pcn/qft/
                  └────►  src/qft_pcn/qft/
```

Nothing under `logic/` or `qft/` may `import qft_pcn.bridge`. An import-graph test (§7.6) verifies this.

The "easy shortcut" — putting "intelligent verbalization" of diagnostics under `logic/debugger.py` because "the diagnostic is for the LLM anyway" — is **rejected**. The debugger emits structured records; verbalization happens in the LLM, period.

### 1.4 Errors are structured, never strings-only

Every failure mode the bridge can produce has a stable **error code** (§10), a human-readable **message**, and (where applicable) a machine-readable **details** payload. The LLM consumes codes and details, not free-form prose. This is what makes the LLM's error-handling reliable across model upgrades.

The "easy shortcut" — `raise ValueError(f"the spec is bad, somewhere")` — is **rejected**. Every error in the bridge must be a `BridgeError` subclass with a code from the table in §10.

### 1.5 Synchronous JSON-RPC over stdio is the transport

The architecture doc allows FastAPI or stdio JSON-RPC; we pick **stdio JSON-RPC** because:

- It runs without a network port → trivially sandboxable.
- It matches how MCP servers and many LLM-tool harnesses already work, so an LLM can drive it via subprocess.
- It removes an entire dependency (FastAPI, uvicorn, asyncio plumbing).
- Local-demo throughput is fine — a single QPCN run takes seconds; transport overhead is negligible.

A future sub-project can add HTTP if needed; the runtime API (§5) is transport-agnostic so adding HTTP is purely additive.

The "easy shortcut" — making the protocol asynchronous "to be ready for streaming" — is **rejected**. The QPCN's `evolve` is synchronous and synchronous is the right shape for this contract. We will add streaming when there is a real need.

### 1.6 No LLM is required to test the bridge

Every test in §7 runs without making a single external API call. The end-to-end demo (§8) ships with a `MockLLM` that emits canned DSL specs. The "real LLM" path is a thin shim in `bridge/llm.py` that calls the Anthropic SDK (via the `claude-api` skill's patterns) — it exists so a human can drive the demo from a prompt, but it is not on the critical path of any test.

The "easy shortcut" — making one of the acceptance tests depend on a live `claude.com` API call — is **rejected**. Tests are deterministic and offline.

### 1.7 Reuse, do not reinvent

`src/qft_pcn/logic/` and `src/qft_pcn/qft/` are the canonical machinery:

- `encode(ast, N, chi_max)` and `decode(state, meta)` from sub-project A.
- B's `compile_typing_hamiltonian(meta, ...) -> Hamiltonian` (or whatever B's spec defines).
- C's `compile_eval_hamiltonian(meta, ...) -> Hamiltonian` (or whatever C's spec defines).
- D's `diagnose(state, meta, H, terms) -> DiagnosticReport`.
- `evolve(state, H, dt, steps, imaginary, chi_max)` from `qft/evolution.py`.
- `MPS.local_expectation(site, op)` from `qft/mps.py`.
- `Hamiltonian.local_op(k)`, `Hamiltonian.bond_op(k)` from `qft/hamiltonian.py`.

The bridge **adapts** these APIs into the JSON-shaped surface. If B's or C's public API is not stable yet (they are in flight), the bridge depends on the named entry points; when B/C land, the bridge's wiring is the only place that needs touching.

The "easy shortcut" — inlining a "simple" Hamiltonian compiler that ignores B's spec because "B isn't ready" — is **rejected**. The bridge stays stubbed against the named entry points; sub-project G can land with B/C entry points marked `TODO(B)`/`TODO(C)` and a small adapter that raises `BridgeError(code="constraint_kind_unsupported")` for any constraint not in a known subset. The structural and protocol-level tests still pass; the end-to-end demo waits on B/C.

### 1.8 The constraint-expression DSL is parsed, not eval'd

`constraints[i].term` is a string like `"type(arg1) == type(arg2)"`. It is parsed into an AST by the bridge's expression parser (§4.2) and then compiled into a Hamiltonian term via B's term factories. The bridge does **not** call `eval()`, `exec()`, `compile(...)`, `ast.literal_eval`, `pickle.loads`, or `subprocess` on any user-supplied string. The parser accepts a closed grammar (§4.1) with no escape hatches.

The "easy shortcut" — `eval(constraint.term, {"type": type_fn, ...})` to "save writing a parser" — is **categorically rejected**. The bridge is the trust boundary between an LLM (which we treat as untrusted in this respect) and the QPCN; arbitrary code execution from DSL input is a security incident.

---

## 2. Scope

### 2.1 In scope (this sub-project)

- A JSON schema for the DSL (§3), validated on every incoming request.
- A constraint-expression parser (§4.1, §4.2) and term compiler (§4.3) that maps DSL constraints to Hamiltonian terms via B's and C's APIs.
- A synchronous `run_problem(dsl) -> Result` and `diagnose_problem(dsl) -> Result + Diagnostic` runtime API (§5).
- A stdio JSON-RPC server (§6) exposing `problem.run`, `problem.diagnose`, `dsl.validate`, `health.check`.
- A small Python builder library (`bridge/templates.py`) for constructing DSL specs from typed templates (§7).
- An end-to-end demo (§8) where a `MockLLM` emits a DSL spec, the bridge runs the QPCN, and the result is rendered. Plus a thin (non-test-path) shim for invoking a real LLM via the Anthropic SDK.
- A structured error model (§10) with stable codes.
- An import-graph test enforcing §1.3.

### 2.2 Out of scope (deferred to other sub-projects or future work)

- FastAPI / HTTP transport (additive future work).
- Streaming partial results (additive future work; the QPCN doesn't naturally stream).
- Authentication / multi-tenant isolation (the demo is single-user, single-process, local-only).
- A LLM-side prompt-engineering harness (the LLM template is part of the demo, not part of the bridge contract).
- Persistent state across calls (each `problem.run` is independent; the bridge holds no session state besides the cached schema and the JSON-RPC dispatcher).
- Caching of Hamiltonians between calls (a future optimization; the runtime recomputes from scratch on every call).
- Sub-project B's, C's, or D's internal logic. The bridge consumes their public APIs only.

### 2.3 Will not do, even if asked later

- Inline a copy of the Hamiltonian compiler inside the bridge.
- Add `eval()` / `exec()` / shell-out behaviour for any user-supplied string.
- Add an "LLM auto-fix" mode where the bridge calls an LLM to repair a malformed DSL spec before validating it.
- Hide bond-dimension / MPS / Hamiltonian terminology from the bridge's *internal* logs (the bridge logs everything; only the JSON-RPC response surface is LLM-shaped).
- Accept natural-language fields anywhere in the DSL.

---

## 3. The DSL schema

The DSL is a single JSON document with six top-level fields. All are required except `boundary` (optional) and `search` (defaults). The JSON schema in `src/qft_pcn/bridge/dsl_schema.json` is the canonical source; this section is the human-readable form.

### 3.1 Top-level shape

```json
{
  "fields":      [ <FieldSpec>, ... ],
  "sites":       <int, 1..256>,
  "constraints": [ <ConstraintSpec>, ... ],
  "boundary":    { "<site>": <BoundarySpec>, ... },
  "observables": [ <ObservableSpec>, ... ],
  "search":      <SearchSpec>
}
```

### 3.2 `FieldSpec`

```json
{ "name": "expr",  "cutoff": 16 }
```

| key | type | constraint |
|---|---|---|
| `name`   | string | matches `^[a-z][a-z0-9_]{0,31}$` |
| `cutoff` | int    | `2 ≤ cutoff ≤ 32` |

The `fields` list must contain at least one field. Duplicate names are rejected.

**Compatibility with sub-project A**: when the DSL is going to consume an AST-encoded MPS (the typical case for §10.7 synthesis), `fields` must be the canonical four `kind/type/bid/value` species with the cutoffs `8/8/8/16` from spec A §4. For other problem types (chemistry, lattice models) `fields` may be any list. Sub-project A produces these names automatically via `EncodingMeta.species`; the LLM-side templates (§7) hard-code the canonical four for the synthesis use case.

### 3.3 `sites`

Positive integer in `[1, 256]`. This is the MPS length `N`. Larger values are rejected for safety — `sites=256` already implies a worst-case ~8 GB local-tensor space at `d_local=8192`, so the bridge surfaces this limit early. Future work may raise the cap once B/C's terms are validated for larger lattices.

### 3.4 `ConstraintSpec`

Two kinds, distinguished by `kind`:

```json
{ "kind": "local",    "site":  5,        "term": "<expr>", "weight": 1.0 }
{ "kind": "two_site", "sites": [5, 7],   "term": "<expr>", "weight": 1.0 }
```

| key      | type      | constraint |
|---|---|---|
| `kind`   | string    | one of `"local"`, `"two_site"` |
| `site`   | int       | required for `local`; in `[0, sites-1]` |
| `sites`  | [int,int] | required for `two_site`; both in `[0, sites-1]`; ordered, distinct |
| `term`   | string    | parses successfully under §4.1 grammar |
| `weight` | float     | finite, ≥ 0. Default 1.0 if omitted. |

For `two_site`, the two sites need not be adjacent — but B's compiler is responsible for actually placing the term (it may require adjacency; in that case the bridge surfaces B's "non-adjacent two_site constraint" error rather than papering over it).

### 3.5 `BoundarySpec`

```json
{ "5":  { "value": 3.0 },
  "12": { "expr":  "result" } }
```

Keys are stringified site indices (JSON has no integer keys). Each value is an object mapping `field_name → fixed_value`. Fixed values are one of:

- A **number** (int or float): interpreted as a value-register basis index for the named field. The bridge sets that site's state to be a definite eigenstate on that field. Float values are rejected for cutoff-bounded fields unless they round-trip to an int.
- A **string** containing a basis-name literal recognized by the field's vocabulary (e.g. `"V_TRUE"`, `"T_INT"`, `"KIND_LAM"`). The bridge resolves these via `bridge/vocab.py` which mirrors the constants from `logic/encoding.py`. For non-canonical (chemistry, lattice) fields the LLM passes integers.

A site listed in `boundary` is "clamped" — see §5.4 for how clamping interacts with imaginary-time evolution.

### 3.6 `ObservableSpec`

```json
{ "site": 12, "field": "value", "op": "n" }
```

| key     | type   | constraint |
|---|---|---|
| `site`  | int    | in `[0, sites-1]` |
| `field` | string | one of the `fields[i].name` values |
| `op`    | string | one of `"n"`, `"phi"`, `"pi"`, `"a"`, `"adag"`, `"identity"` |

Returned as a real number for `n`, `phi`, `pi` (which are Hermitian on the appropriate species), and as a complex number (split into `re`/`im`) for `a`, `adag`. The bridge always returns full Hermiticity diagnostics: `{"site": 12, "field": "value", "op": "n", "value": 2.347, "imag_part": 1.2e-13}` so the LLM can sanity-check.

### 3.7 `SearchSpec`

```json
{ "method": "imag_time", "steps": 50, "chi_max": 32, "dt": 0.05 }
```

| key       | type | constraint |
|---|---|---|
| `method`  | string | `"imag_time"` (only one supported in this sub-project) |
| `steps`   | int    | `1 ≤ steps ≤ 1000` |
| `chi_max` | int    | `2 ≤ chi_max ≤ 64` |
| `dt`      | float  | `0 < dt ≤ 0.5`; defaults to `0.05` |

Defaults if `search` is omitted: `{"method": "imag_time", "steps": 50, "chi_max": 32, "dt": 0.05}`.

Other methods (`real_time`, `vqe`) are reserved for future expansion. The bridge rejects them with `BRIDGE_E_UNSUPPORTED_METHOD`.

### 3.8 Schema validation

The validation pipeline:

1. JSON parse → produces a Python `dict`. Parse errors → `BRIDGE_E_BAD_JSON`.
2. JSON Schema validation against `dsl_schema.json` (we use the `jsonschema` package, a single small dependency). Failures → `BRIDGE_E_BAD_SCHEMA` with the JSON Pointer to the offending field.
3. Cross-field validation (e.g. `sites` referenced in `constraints` are < `sites`; field names in `observables` exist in `fields`). Failures → `BRIDGE_E_BAD_REFERENCE` with the JSON Pointer.
4. Constraint-expression parsing for every `constraints[i].term`. Failures → `BRIDGE_E_BAD_TERM` with the offending site/index and parser error location.

All four phases run on `dsl.validate` (which returns a structured result without executing); only phases 1–4 succeeding triggers actual Hamiltonian compilation in `problem.run`.

### 3.9 The full DSL example (canonical §10.7 STLC synthesis)

```json
{
  "fields": [
    { "name": "kind",  "cutoff": 8  },
    { "name": "type",  "cutoff": 8  },
    { "name": "bid",   "cutoff": 8  },
    { "name": "value", "cutoff": 16 }
  ],
  "sites": 32,
  "constraints": [
    { "kind": "local",     "site": 0,  "term": "kind == LAM",            "weight": 1.0 },
    { "kind": "local",     "site": 1,  "term": "kind == VAR",            "weight": 1.0 },
    { "kind": "two_site",  "sites": [0, 1],
      "term": "type(arg1) == arr(T_INT, T_INT) and type(arg2) == T_INT", "weight": 1.0 }
  ],
  "boundary": {
    "0":  { "kind": "KIND_LAM" },
    "1":  { "kind": "KIND_VAR" }
  },
  "observables": [
    { "site": 1, "field": "kind", "op": "n" }
  ],
  "search": { "method": "imag_time", "steps": 50, "chi_max": 32 }
}
```

This is what the LLM produces; the bridge ingests, compiles, runs, returns observables.

---

## 4. Constraint expression language

This is the hardest part of G. The DSL embeds a small expression sub-language for constraint terms. It is parsed, type-checked, and compiled to Hamiltonian terms.

### 4.1 Grammar

The expression grammar:

```
expr     ::= or_expr
or_expr  ::= and_expr ("or" and_expr)*                       (left-assoc, lowest prec)
and_expr ::= not_expr ("and" not_expr)*                      (left-assoc)
not_expr ::= "not" not_expr | cmp_expr
cmp_expr ::= add_expr (cmp_op add_expr)?                     (non-associative; one cmp max)
cmp_op   ::= "==" | "!=" | "<" | "<=" | ">" | ">="
add_expr ::= mul_expr (("+" | "-") mul_expr)*                (left-assoc)
mul_expr ::= unary_expr ("*" unary_expr)*                    (left-assoc)
unary_expr ::= "-" unary_expr | call_expr
call_expr ::= atom ("(" arg_list? ")")?                      (one call max; no chained calls)
arg_list ::= expr ("," expr)*
atom     ::= IDENT | INT | STRING | "(" expr ")"
```

Reserved identifiers (constants and the bound-argument vocabulary):

- **Boolean literals**: `true`, `false`.
- **Field-basis constants**: any constant defined in `bridge/vocab.py`. For the canonical four-field species these are the names from `logic/encoding.py`: `KIND_PAD`, …, `KIND_BIN`; `T_NONE`, …, `T_ARR_NESTED`; `B_NONE`, …, `B_6`; `V_NONE`, `V_FALSE`, `V_TRUE`, `V_PLUS`, …, `V_EQ`, and integer literals via `V_INT(n)` calls. For chemistry / other fields the LLM may pass raw ints.
- **Built-in functions**: `type(arg)`, `kind(arg)`, `bid(arg)`, `value(arg)`, `n(arg)`, `phi(arg)`, `pi(arg)`, `arr(τ_src, τ_dst)`.
- **Site-bound arguments**: `arg1`, `arg2` (refer to the constraint's site arguments). For `kind=local`, only `arg1` is in scope (and refers to "this site"); aliasing `self` is also accepted.

Strings in atoms (`"+"`, `"result"`, etc.) are reserved for B/C — the bridge passes them through to B as identifier-like tokens; B's grammar may use them for specific term factories (e.g. `kind == '+'` for selecting a BIN op).

### 4.2 The parser

`bridge/dsl/expr_parser.py` implements a recursive-descent parser with the grammar above. It produces an `Expr` AST whose node types live in `bridge/dsl/expr_ast.py`:

```python
@dataclass class Lit:    value: int | bool | str
@dataclass class Ident:  name: str
@dataclass class Call:   fn: str; args: list[Expr]
@dataclass class Unary:  op: str; operand: Expr
@dataclass class Binary: op: str; lhs: Expr; rhs: Expr
```

Parser errors carry `(position, expected, found)` for the LLM to fix the spec. They subclass `BridgeError(code=BRIDGE_E_BAD_TERM)`.

This parser is **structurally similar** to (and shares no code with) `logic/ast.py`'s parser. Reuse is *not* possible: the surface languages and target ASTs are different. We accept the duplication — it's <200 lines.

### 4.3 The term compiler

`bridge/dsl/compiler.py` walks the `Expr` AST and emits Hamiltonian terms via B's term factories. The mapping is:

| `Expr` shape | Hamiltonian term |
|---|---|
| `Binary("==", Call("kind", arg), Lit(C))` | one-site projector: `+w * (1 - P_kind=C)` at `arg`'s site |
| `Binary("==", Call("type", arg), Lit(T))` | one-site projector: `+w * (1 - P_type=T)` at `arg`'s site |
| `Binary("==", Call("type", arg1), Call("type", arg2))` | two-site term: `+w * (1 - Σ_t P_type=t@arg1 ⊗ P_type=t@arg2)` |
| `Binary("==", Call("kind", arg), Lit_string("+"))` | one-site: `+w * (1 - P_kind=BIN ∧ value=V_PLUS)` |
| `Call("n", arg)` (in a `local` constraint context with no comparator) | one-site: `+w * n_field @ arg`'s site |
| `Binary("and", e1, e2)` | sum: `compile(e1) + compile(e2)` |
| `Binary("or", e1, e2)` | product-like: `+w * (1 - P_e1) * (1 - P_e2)` (penalize the AND of negations) |
| `Unary("not", e)` | `+w * (1 - compile_projector(e))` for projector-shaped `e`; rejected otherwise |
| `Lit(true)` | no term (vacuously satisfied) |
| `Lit(false)` | `+w * I` (always-violated; useful for testing) |

**Crucially**, the compiler does **not** synthesize Hamiltonian operators directly. It calls B's named factory functions:

```python
# B's public API (sub-project B spec, when written):
typing.term_kind_equals(meta, site, kind_value, weight) -> LocalTerm
typing.term_type_equals_constant(meta, site, type_value, weight) -> LocalTerm
typing.term_type_equality(meta, sites, weight) -> TwoSiteTerm
typing.term_arr_destructure(meta, site, src_ty, dst_ty, weight) -> LocalTerm
# C's public API (sub-project C spec, when written):
eval_.term_bin_op(meta, site, op_value, weight) -> LocalTerm
eval_.term_density(meta, site, field_name, weight) -> LocalTerm
```

Each `LocalTerm` / `TwoSiteTerm` is a thin wrapper around the operator matrix that B's `Hamiltonian.add_term` (or the equivalent) consumes. The compiler collects them and constructs the full Hamiltonian via B's `Hamiltonian.from_terms(terms, meta)` factory.

**When B/C aren't ready yet**: the compiler's term-factory calls are routed through `bridge/dsl/compiler_stub.py` that raises `BridgeError(code=BRIDGE_E_TERM_UNSUPPORTED, details={"term": ...})` for any pattern not in a hard-coded fallback set. The fallback set covers the §3.9 example and the four §7 acceptance tests, so end-to-end demos can be written and tested independently of B/C's landing. The compiler swaps to B/C's real factories the moment B/C lands; the stub is removed in that landing commit.

### 4.4 Type and arity checking

Before emitting any term the compiler verifies:

- `arg1`/`arg2` references match the constraint's `kind` (e.g. `local` constraints may not reference `arg2`).
- Built-in functions have the right arity (`type` takes 1 arg; `arr` takes 2).
- Comparators apply to compatible operand types (e.g. `kind(arg1) == kind(arg2)` is fine; `kind(arg1) + 1` is rejected as `BRIDGE_E_TYPE_ERROR`).
- Field references match a name in `fields`.

Failures here are `BridgeError(code=BRIDGE_E_TYPE_ERROR)` with `details.term`, `details.constraint_index`, `details.kind`.

---

## 5. The runtime API

`src/qft_pcn/bridge/runtime.py` exposes the transport-independent runner.

### 5.1 `run_problem`

```python
def run_problem(dsl: dict) -> RunResult:
    """Validate the DSL, compile a Hamiltonian, evolve, measure observables.

    Returns a structured RunResult (§5.2). Raises BridgeError subclasses
    for any validation or compilation failure (see §10).
    """
```

The function does, in order:

1. Validate (§3.8). Raises on failure.
2. Compile constraint terms (§4) into a list of `Term` objects via B/C's factories. Raises on failure.
3. Construct a Hamiltonian: `H = sum(t.weight * t.operator for t in terms)`, expressed via B's `Hamiltonian.from_terms(terms, meta_or_species_list, N=sites)`. The `species` list is built from `dsl.fields` (or A's canonical `SPECIES` if the names match).
4. Construct an initial MPS:
   - If `boundary` is empty: `MPS.vacuum(N, d_local)` — the all-PAD / vacuum state.
   - If `boundary` is non-empty: construct a product state where the boundary sites hold the clamped basis state on the named field and `PAD`/vacuum on the others; all other sites hold the all-vacuum local state.
5. Run `evolve(state, H, dt=search.dt, steps=search.steps, imaginary=True, chi_max=search.chi_max, normalize_every=1)`.
6. Re-clamp boundary sites after every step to maintain the constraint (§5.4); the bridge does this by projecting the local site tensor onto the clamped basis state at the end of each `trotter_step` (this requires a small wrapper around `evolve`, see §5.5).
7. Compute each observable via `MPS.local_expectation(site, embed(op_for_field, ...))`.
8. Return:
   ```json
   {
     "observables": [
       { "site": 12, "field": "value", "op": "n",
         "value": 2.347, "imag_part": 1.2e-13 }
     ],
     "energy": 0.034,
     "energy_per_term": [...],
     "truncation_error_sum": 1.4e-7,
     "final_bond_dimensions": [...],
     "converged": true,
     "convergence_history": {
       "energy_per_step": [0.81, 0.41, 0.21, ..., 0.034]
     }
   }
   ```

`converged` is `true` iff `|E_{step} - E_{step-1}| < 1e-6` for the final 5 steps and energy is monotonically non-increasing across the run.

### 5.2 `RunResult` and `RunDiagnostic`

```python
@dataclass
class RunResult:
    observables: list[ObservableValue]
    energy: float
    energy_per_term: list[float]     # one entry per constraint, in DSL order
    truncation_error_sum: float
    final_bond_dimensions: list[int]
    converged: bool
    convergence_history: ConvergenceHistory

@dataclass
class ObservableValue:
    site: int
    field: str
    op: str
    value: float            # real part
    imag_part: float        # imaginary part; should be ~0 for Hermitian ops

@dataclass
class ConvergenceHistory:
    energy_per_step: list[float]
    # bond_dim_per_step and trunc_error_per_step omitted to keep payloads
    # compact; available on diagnose_problem.

@dataclass
class RunDiagnostic:
    result: RunResult
    debugger_report: dict | None     # filled by D's diagnose() when available;
                                     # None if D's API isn't wired or the run
                                     # converged below the threshold
```

`debugger_report`'s shape is owned by sub-project D. The bridge passes D's report through verbatim — it does not interpret or reformat it. Until D lands, this field is `None` and `diagnose_problem` behaves identically to `run_problem` plus a `null` debugger report.

### 5.3 `diagnose_problem`

```python
def diagnose_problem(dsl: dict) -> RunDiagnostic:
    """Same as run_problem, but also runs D's diagnose() on the final state.

    Returns RunResult with debugger_report populated by D. Raises BridgeError
    on validation/compilation failures, identical to run_problem.
    """
```

D's API (when written) is expected to be:

```python
# Sub-project D, src/qft_pcn/logic/debugger.py
def diagnose(state: MPS, meta: EncodingMeta | None,
             H: Hamiltonian, terms: list[Term]) -> DiagnosticReport: ...
```

`meta` is `None` for non-AST problems (chemistry, etc.). The bridge passes whatever `meta` it constructed in §5.1.4; if the DSL fields don't match A's canonical SPECIES, `meta` is `None` and D is expected to produce a degraded but still-structured report.

### 5.4 Boundary clamping during evolution

A site listed in `boundary` is constrained to remain on a fixed basis state on the named field. The bridge implements this by:

1. Building an initial state where that site's local tensor is a rank-1 product on the named field × vacuum on the others.
2. After each Trotter step, re-projecting that site's tensor onto the same product structure (and re-normalizing). This is a "post-selection" treatment — equivalent to adding an infinite-weight Hamiltonian term and is simpler / more numerically stable.

This is the bridge's job; B/C are not asked to express boundary clamps via terms. The implementation lives in `bridge/runtime/clamp.py`.

### 5.5 The evolution wrapper

`evolve` from `qft/evolution.py` is the canonical evolver; the bridge calls it in a tight loop, projecting after each step:

```python
def _evolve_with_clamps(state: MPS, H: Hamiltonian, *, dt: float, steps: int,
                        chi_max: int, clamps: list[Clamp]
                        ) -> ConvergenceHistory:
    history = ConvergenceHistory(energy_per_step=[])
    for k in range(steps):
        trotter_step(state, H, dt, imaginary=True, chi_max=chi_max)
        state.normalize()
        for c in clamps:
            _project_site(state, c.site, c.field, c.basis_index)
            state.normalize()
        history.energy_per_step.append(energy(state, H))
    return history
```

This wrapper is what makes the bridge non-trivial; everything else is plumbing.

---

## 6. The protocol layer (stdio JSON-RPC)

`src/qft_pcn/bridge/api.py` implements a synchronous JSON-RPC 2.0 server over stdin/stdout. Each line on stdin is one request; each line on stdout is one response. (`Content-Length`-framed messages are accepted as an optional mode for tools that prefer LSP-style framing — `--framing=lsp` CLI flag.)

### 6.1 Supported methods

| method            | params                                | result |
|---|---|---|
| `problem.run`     | `{ "dsl": <DSL document> }`           | `RunResult` (§5.2) serialized as JSON |
| `problem.diagnose`| `{ "dsl": <DSL document> }`           | `RunDiagnostic` (§5.2) serialized as JSON |
| `dsl.validate`    | `{ "dsl": <DSL document> }`           | `{ "valid": bool, "errors": [<BridgeError>] }` |
| `health.check`    | `{}`                                  | `{ "ok": true, "version": "<git-sha>", "subprojects": { "A": "ok", "B": "stub", ... } }` |

`subprojects` in `health.check` reports which downstream packages have landed and which the bridge is stubbing for. This is for human ops, not the LLM.

### 6.2 Request / response shapes

Standard JSON-RPC 2.0:

```json
// Request
{ "jsonrpc": "2.0", "id": 1, "method": "problem.run", "params": { "dsl": {...} } }
// Success response
{ "jsonrpc": "2.0", "id": 1, "result": {...} }
// Error response
{ "jsonrpc": "2.0", "id": 1, "error":
  { "code": -32001, "message": "BRIDGE_E_BAD_SCHEMA: ...",
    "data": { "code": "BRIDGE_E_BAD_SCHEMA", "details": {...} } }
}
```

JSON-RPC error codes:

- `-32700` parse error (invalid JSON)
- `-32600` invalid request (not JSON-RPC 2.0)
- `-32601` method not found
- `-32602` invalid params
- `-32603` internal error
- `-32001` through `-32099` reserved for `BridgeError` subclasses; `error.data.code` carries the typed BridgeError code (§10).

### 6.3 Server lifecycle

```
$ python -m qft_pcn.bridge        # stdio mode (default)
$ python -m qft_pcn.bridge --framing=lsp
$ python -m qft_pcn.bridge --once "<JSON request>"   # one-shot CLI mode for scripting
```

The `--once` mode runs a single request from a CLI string and prints the response to stdout. This is what the end-to-end demo uses (§8) and what most ad-hoc invocations want.

A single request is processed synchronously; the next line on stdin is not read until the response has been written. There is no concurrency by design — the QPCN is not thread-safe and the bridge is the single critical section.

### 6.4 Logging

All logs go to **stderr**, never stdout (stdout is the protocol channel). Default log level is `INFO`; `--verbose` flips to `DEBUG`. The bridge logs:

- Every incoming request id and method.
- Every BridgeError with its code and the JSON Pointer / index that triggered it.
- Convergence history at DEBUG.

Logs are plain text; if the operator wants structured logs they can pipe `--log-format=json` (additive; not required for the acceptance tests).

---

## 7. The LLM-side helpers

`src/qft_pcn/bridge/templates.py` exposes a small Python library that an LLM (or a script pretending to be one) can use to construct DSL specs from typed templates. **This file does not import any LLM SDK** — it's pure JSON construction. The "real LLM" calls live in `bridge/llm.py` (§8).

### 7.1 The `Problem` builder

```python
@dataclass
class Problem:
    fields: list[FieldSpec]
    sites: int
    constraints: list[ConstraintSpec] = field(default_factory=list)
    boundary: dict[int, dict[str, Any]] = field(default_factory=dict)
    observables: list[ObservableSpec] = field(default_factory=list)
    search: SearchSpec = field(default_factory=SearchSpec.default)

    def to_dsl(self) -> dict: ...
    def validate(self) -> None: ...   # locally validate before sending
```

### 7.2 Named factories

For the common §10.7 use cases:

```python
def stlc_synthesis(*, sites: int = 32,
                   holes: list[HoleSpec],
                   constraints: list[ConstraintSpec],
                   examples: list[InputOutputExample] | None = None,
                   ) -> Problem: ...

def stlc_type_check(*, program: Node) -> Problem:
    """Build a DSL spec that asks the QPCN to verify well-typedness.
    Uses sub-project A's encode() to produce boundary clamps for every
    AST node and asks for the total energy (zero iff well-typed)."""
    ...
```

`stlc_synthesis` constructs:

- The canonical four `kind/type/bid/value` fields.
- `sites` clamped at each known position; holes left free.
- Default typing-rule constraints (the ones B will provide once landed); until then, a hard-coded subset matching §3.9.
- Observables on each hole site (the `kind` and `value` registers).

`stlc_type_check` constructs:

- The canonical four fields with all sites clamped to the encoded AST.
- All typing-rule constraints active.
- One observable: total energy (achieved by adding an observable at every site of `n` on each field and summing — but for type-checking the LLM cares about *total* energy, which is in the `RunResult` directly).

These factories are the **only** code in the bridge that knows about the §10.7 problem; they exist so the demo (§8) and the tests are not littered with raw JSON.

### 7.3 The "mock LLM"

For tests and demos, `bridge/llm.py` ships a `MockLLM` class:

```python
class MockLLM:
    """A canned 'LLM' that returns hardcoded DSL specs for known prompts."""
    def __init__(self, responses: dict[str, dict]): ...
    def emit_dsl(self, prompt: str) -> dict: ...
    def verbalize(self, prompt: str, result: dict) -> str: ...
```

Tests use this. The real-LLM shim (next section) is structurally identical so the test path matches the demo path 1:1.

### 7.4 The "real LLM" shim

`bridge/llm.py` also exposes:

```python
class AnthropicLLM:
    """Adapter calling the Anthropic SDK to produce a DSL spec from a prompt.

    Uses prompt caching for the static schema-shaped portion of the prompt
    (see claude-api skill patterns). Returns a parsed JSON dict.

    Requires ANTHROPIC_API_KEY; not used by any test.
    """
    def emit_dsl(self, prompt: str) -> dict: ...
    def verbalize(self, prompt: str, result: dict) -> str: ...
```

This is the **only** place an LLM SDK is imported in the entire repository. The import is guarded so the bridge package still imports cleanly without the SDK installed (the test suite passes without it).

---

## 8. End-to-end demo

`src/qft_pcn/bridge/demo.py` is a single script that:

1. Defines a hardcoded "user prompt" string ("find a function of type Int → Int that doubles its input").
2. Instantiates a `MockLLM` configured to return the §3.9 DSL spec for that prompt.
3. Calls `mock_llm.emit_dsl(prompt)` to get the DSL.
4. Calls `runtime.run_problem(dsl)` to get the result.
5. Calls `mock_llm.verbalize(prompt, result)` to get a string.
6. Prints a five-section transcript: prompt, DSL, observables table, energy, verbalization.

The same `demo.py` accepts a `--llm=anthropic` flag that swaps `MockLLM` for `AnthropicLLM` — the demo is then a real end-to-end test against the Anthropic API. The default and the CI path is `--llm=mock`.

A second variant `demo_protocol.py` does the same flow but **via the JSON-RPC server**: it spawns `python -m qft_pcn.bridge --framing=lsp` as a subprocess, sends the DSL as a `problem.run` request, reads the result, and feeds it to `MockLLM.verbalize`. This exercises the full transport.

---

## 9. File layout

```
src/qft_pcn/bridge/
├── __init__.py                     # public re-exports: run_problem,
│                                   #   diagnose_problem, validate_dsl,
│                                   #   Problem, MockLLM, AnthropicLLM,
│                                   #   BridgeError + subclasses
├── __main__.py                     # CLI entry point: `python -m qft_pcn.bridge`
├── api.py                          # stdio JSON-RPC server (§6)
├── runtime.py                      # run_problem, diagnose_problem (§5)
├── runtime/
│   └── clamp.py                    # boundary-site projection (§5.4)
├── dsl/
│   ├── __init__.py
│   ├── schema.py                   # JSON Schema construction + jsonschema validate (§3)
│   ├── dsl_schema.json             # the canonical schema document (build artifact of schema.py)
│   ├── cross_validate.py           # cross-field validation (§3.8 phase 3)
│   ├── expr_ast.py                 # Expr AST node types (§4.2)
│   ├── expr_parser.py              # constraint-expression parser (§4.1)
│   ├── compiler.py                 # Expr -> Hamiltonian terms via B/C (§4.3)
│   ├── compiler_stub.py            # fallback term factories until B/C land (§4.3)
│   └── vocab.py                    # name <-> integer constants for canonical fields (§3.5)
├── templates.py                    # Problem builder + stlc_synthesis / stlc_type_check (§7)
├── llm.py                          # MockLLM + AnthropicLLM (§7.3, §7.4)
├── errors.py                       # BridgeError hierarchy + codes (§10)
├── demo.py                         # end-to-end demo (§8)
└── demo_protocol.py                # demo via subprocess JSON-RPC (§8)
src/qft_pcn/tests/
├── test_bridge_schema.py           # §3 schema and cross-validation tests
├── test_bridge_expr_parser.py      # §4.1/4.2 parser tests
├── test_bridge_compiler.py         # §4.3 compiler tests (using stub factories)
├── test_bridge_runtime.py          # §5 run_problem / diagnose_problem tests
├── test_bridge_protocol.py         # §6 JSON-RPC tests (in-proc and subprocess)
├── test_bridge_templates.py        # §7 builder tests
├── test_bridge_dependency_graph.py # §1.3 import-graph test
└── test_bridge_demo.py             # end-to-end demo test (§8 mock path)
```

The bridge adds **one** new third-party dependency: `jsonschema` (small, pure Python, MIT-licensed). The `anthropic` SDK is *optional*: `bridge/llm.py::AnthropicLLM` imports it lazily and raises `ImportError` with a helpful message if missing. The test suite does not require `anthropic`.

No changes to existing files except:

- `pyproject.toml`: add `jsonschema` to required deps.
- `src/qft_pcn/__init__.py`: re-export the bridge package top-level names so callers can `from qft_pcn import run_problem`.

---

## 10. Error model

Every error from the bridge is a subclass of `BridgeError` with a stable string code, a human message, and an optional `details` dict.

```python
class BridgeError(Exception):
    code: str
    details: dict

# Concrete codes:
BRIDGE_E_BAD_JSON                # JSON parse failure
BRIDGE_E_BAD_SCHEMA              # JSON-Schema validation failure
BRIDGE_E_BAD_REFERENCE           # cross-field validation failure
BRIDGE_E_BAD_TERM                # constraint-expression parse failure
BRIDGE_E_TYPE_ERROR              # constraint-expression type-check failure
BRIDGE_E_TERM_UNSUPPORTED        # compiler stub: B/C don't yet support this term
BRIDGE_E_UNSUPPORTED_METHOD      # search.method not in {imag_time}
BRIDGE_E_SITES_OUT_OF_RANGE      # site index out of [0, sites-1]
BRIDGE_E_TOO_LARGE               # sites > 256
BRIDGE_E_CONSTRAINT_NOT_ADJACENT # two_site sites not adjacent (when B requires it)
BRIDGE_E_NUMERIC_FAILURE         # NaN/Inf in observable or energy
BRIDGE_E_INTERNAL                # unexpected exception (with stack trace in logs)
```

The JSON-RPC error response carries:

```json
{ "code": -32001,
  "message": "BRIDGE_E_BAD_SCHEMA: 'sites' must be a positive integer",
  "data": {
    "code": "BRIDGE_E_BAD_SCHEMA",
    "details": { "pointer": "/sites", "value": -3, "constraint": "minimum=1" }
  } }
```

The LLM consumes `data.code` and `data.details`. The `message` is for humans.

---

## 11. Tests

Located in `src/qft_pcn/tests/`. The bridge test suite is sized so a fresh `pytest src/qft_pcn/tests/test_bridge_*` runs in under 30 seconds.

### 11.1 Schema tests (`test_bridge_schema.py`)

- `test_valid_minimal_dsl_passes`: smallest valid DSL (one field, one site, no constraints, no boundary, one observable, default search) validates.
- `test_valid_canonical_stlc_dsl_passes`: the §3.9 example validates.
- `test_missing_required_field_raises_bad_schema`: each of `fields`, `sites`, `constraints`, `observables` triggers `BRIDGE_E_BAD_SCHEMA` with the right pointer.
- `test_field_name_duplicate_raises_bad_schema`.
- `test_site_out_of_range_in_constraint_raises_bad_reference`.
- `test_unknown_field_in_observable_raises_bad_reference`.
- `test_unsupported_search_method_raises`.

### 11.2 Expression parser tests (`test_bridge_expr_parser.py`)

- Round-trip parse → pretty for ~15 expression strings covering every grammar production in §4.1.
- `test_eval_is_not_invoked`: confirm the parser does not call `eval/exec/compile`. (Test by monkey-patching `builtins.eval` to raise.)
- Error-path tests: each grammar violation raises `BRIDGE_E_BAD_TERM` with the right position.
- `test_string_literal_carries_through`: `"kind == '+'"` produces an `Expr` whose Lit value is the string `"+"`.

### 11.3 Compiler tests (`test_bridge_compiler.py`)

Use stub factories (a tiny in-test implementation of B/C's named entry points that produces dense matrices via `qft/fock.py`).

- `test_kind_equals_constant_produces_local_projector`: term `"kind == LAM"` on `site=0` produces a 1-site Hermitian operator that's a projector onto the LAM subspace.
- `test_two_site_type_equality_produces_two_site_term`.
- `test_and_combines_two_terms`.
- `test_or_combines_via_projector_product`.
- `test_type_error_rejects_arithmetic_on_kind`.
- `test_unsupported_pattern_raises_term_unsupported`.

### 11.4 Runtime tests (`test_bridge_runtime.py`)

- `test_minimal_run_returns_observables`: a trivial DSL with one site, one observable, no constraints. After 1-step imaginary-time evolution, observables come back finite.
- `test_boundary_clamp_persists`: clamp site 0 to `KIND_LAM`; after 50 evolution steps, `<kind=LAM @ 0> ≈ 1.0`.
- `test_converged_flag`: a DSL whose ground state is reachable in 20 steps reports `converged=True`; a deliberately under-stepped DSL reports `converged=False`.
- `test_energy_per_term_matches_total`: the sum of `energy_per_term` equals `energy` to 1e-10.
- `test_observable_imag_part_is_negligible`: for Hermitian ops, `|imag_part| < 1e-10`.
- `test_diagnose_returns_debugger_report_or_none`: with D stubbed, `debugger_report is None`; with D present (via a stub fixture), it's a dict.

### 11.5 Protocol tests (`test_bridge_protocol.py`)

- `test_health_check_returns_ok`: spawn a subprocess running `python -m qft_pcn.bridge`, send a `health.check`, read the response.
- `test_dsl_validate_on_valid_spec_returns_valid_true`.
- `test_dsl_validate_on_invalid_spec_returns_errors_list`.
- `test_problem_run_round_trip_subprocess`: send §3.9 DSL via subprocess, parse the JSON response, assert shape.
- `test_protocol_handles_invalid_json`: send malformed JSON on stdin, expect a parse-error response.
- `test_protocol_method_not_found`: send `"method": "nonexistent"`, expect `-32601`.
- `test_protocol_request_id_echoed`: send `id=42`, expect `id=42` back.

### 11.6 Template tests (`test_bridge_templates.py`)

- `Problem().to_dsl()` produces JSON that validates against the schema.
- `stlc_synthesis(...)` builds a spec that round-trips through `dsl.validate`.
- `stlc_type_check(parse(P5))` produces a spec whose `boundary` clamps every AST site.

### 11.7 Import-graph dependency test (`test_bridge_dependency_graph.py`)

- `test_logic_does_not_import_bridge`: walk every `.py` file under `src/qft_pcn/logic/` and `src/qft_pcn/qft/`; assert none of them contain `from qft_pcn.bridge` or `import qft_pcn.bridge`. This enforces §1.3.
- `test_bridge_does_not_import_anthropic_at_module_level`: importing `qft_pcn.bridge` should not import the `anthropic` package transitively (verified by checking `sys.modules` after a fresh import).

### 11.8 End-to-end demo test (`test_bridge_demo.py`)

- `test_demo_mock_path_runs_to_completion`: run `bridge/demo.py` as a subprocess; assert exit 0, transcript contains the five expected sections, and the observables JSON in stdout parses.
- `test_demo_protocol_path_runs_to_completion`: run `bridge/demo_protocol.py` as a subprocess; same assertions.

The real-LLM (`AnthropicLLM`) path is **not** tested — it's a documented manual demo.

---

## 12. Acceptance criteria

The sub-project is complete when:

1. All tests in §11 pass on a fresh shell with `jsonschema` installed and `anthropic` *not* installed.
2. `python -m qft_pcn.bridge --once '{"jsonrpc":"2.0","id":1,"method":"health.check","params":{}}'` prints a valid JSON-RPC response with `"ok": true`.
3. `python -m qft_pcn.bridge.demo` (mock path) produces a five-section transcript ending in a non-empty `verbalization` line.
4. `python -m qft_pcn.bridge.demo_protocol` produces the same five-section transcript via subprocess JSON-RPC.
5. The import-graph test (§11.7) passes — no `bridge` import anywhere outside `bridge/`.
6. `BRIDGE_E_*` codes are stable: a test snapshots the set of code strings and fails on additions/removals (`test_bridge_error_codes_snapshot`).
7. All previously passing tests under `src/qft_pcn/tests/` still pass.
8. `pyproject.toml` has exactly one new required dep (`jsonschema`); `anthropic` is optional.
9. `src/qft_pcn/__init__.py` re-exports `run_problem`, `diagnose_problem`, `validate_dsl`, `Problem`, `MockLLM`, `BridgeError`.

No claim of completion is acceptable without these tests actually running green in a fresh shell. (See `superpowers:verification-before-completion`.)

---

## 13. Open questions

The following are **known unresolved** and will be settled by the time sub-project G lands:

1. **Should boundary clamping be a Hamiltonian term (B's job) or a runtime projection (G's job)?** This spec picks runtime projection (§5.4) for numerical stability and to avoid asking B for an "infinite-weight" term. If, during implementation, B's term-based approach turns out to be cleaner, swap to it — but document the swap in a follow-up.
2. **Should `dsl.validate` also dry-run the term compiler?** Currently it does (§3.8 phase 4). If compile is too expensive to run on every validate call, we may move phase 4 into a separate `dsl.compile_check` method.
3. **What happens when `search.steps` is large enough that intermediate energies are needed by the LLM mid-run?** Reserved for a future `problem.run_streaming` method; not addressed in this sub-project.

These are *expected* open questions, not gaps in the contract. Every section above is binding.

---

## 14. Where this fits in the larger arc

G is the last piece. After G:

- A user (or harness) sends a NL prompt to an LLM.
- The LLM emits a DSL spec via the `Problem` builder (§7) or by direct JSON construction.
- The LLM (or harness) invokes the bridge: `python -m qft_pcn.bridge.demo --prompt "..."` or via subprocess JSON-RPC.
- The bridge validates, compiles, evolves, measures, returns.
- The LLM verbalizes the result.

The full §10.7 publishable milestone (STLC synthesis with holes) becomes a `Problem.stlc_synthesis(...)` call from a thin LLM-facing harness. **The QPCN system is end-to-end functional.**

---

## 15. Glossary (local)

- **DSL** — the JSON document defined in §3; the LLM's output and the bridge's input.
- **Bridge** — this sub-project; the package `src/qft_pcn/bridge/`.
- **Runtime** — the in-process API (§5); transport-independent.
- **Protocol** — the stdio JSON-RPC layer (§6); transports requests to the runtime.
- **Constraint expression** — a string in `constraints[i].term`; parsed by §4.1's grammar, compiled by §4.3.
- **Term factory** — a named function exposed by sub-project B (or C) that produces a `Term` object the bridge consumes.
- **Clamp** — a runtime projection of an MPS site onto a fixed basis state, applied after each evolution step (§5.4).
- **MockLLM** — a deterministic stub LLM used in tests and the default demo.
- **AnthropicLLM** — the optional real-LLM shim (§7.4) gated on the `anthropic` SDK being installed.
- **Stub compiler** — `bridge/dsl/compiler_stub.py`, used until B/C land; produces terms for the §3.9 example only and raises `BRIDGE_E_TERM_UNSUPPORTED` for everything else.
