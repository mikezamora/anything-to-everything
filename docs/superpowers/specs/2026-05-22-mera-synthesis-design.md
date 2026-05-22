# Spec: MERA-Native Debugger + Synthesis (migration sub-project M3)

**Document type**: Implementation specification.
**Date**: 2026-05-22.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Realizes**: `QFT_PCN_ARCHITECTURE.md` §10.6 (constraint debugger) and §10.7 (the first publishable milestone — STLC synthesis with holes), retargeted onto the MERA substrate.
**Supersedes (on the MERA substrate)**: `docs/superpowers/specs/2026-05-21-constraint-debugger-design.md` (sub-project D) and `docs/superpowers/specs/2026-05-21-stlc-synthesis-design.md` (sub-project E). The MPS-era D and E modules (`src/qft_pcn/logic/debugger.py`, `src/qft_pcn/logic/synthesis/`) stay shipped and tested as the cross-check oracle until M3 lands; M3 does not modify them.
**Depends on**: M1 (`docs/superpowers/specs/2026-05-22-mera-native-encoder-design.md` — encoder/decoder/window) and M2 (typing/eval Hamiltonians on MERA, factored imaginary-time evolution).
**Status**: Approved design; ready for implementation plan.

---

## 0. How to read this spec

This specifies the **MERA-native constraint debugger and program synthesizer**, and the new piece that the MPS-era synthesis could not deliver: **structural superposition over candidate AST sub-trees**. It is "sub-projects D and E, redone on the MERA substrate, with holes that are genuinely multi-node."

Everything here is the contract. If something needed during implementation is missing, **stop and ask** — do not guess.

No time, duration, or effort estimates appear in this document. Sequencing and dependency order are specified; calendar/effort are not. Standing project directive.

---

## 1. Driving principles (non-negotiable)

Inherited project-wide (see `2026-05-22-mera-migration-design.md` §1 and M1 §1) and specialized here. A subagent implementing M3 will be tempted by shortcuts that *destroy the point* of the milestone. Each principle below names the shortcut and the principled alternative. If you find yourself reaching for the shortcut, **stop and ask the human**.

### 1.1 No time, effort, or duration estimates

This is a standing critical directive. The plan sequences tasks and pins dependencies; it never estimates calendar time, story points, or "how long" anything takes. **Shortcut rejected**: appending "this should take ~N hours" to any task. **Alternative**: state dependency order and acceptance gates only.

### 1.2 Binding is genuine entanglement, never a classical lookup

A hole's resolution — which binder a `HoleVar` refers to, which sub-tree shape fills a structural hole — is carried by entanglement on the MERA tree, realized by isometries and disentanglers. It is never a Python-side `dict` lookup, never a runtime branch over candidates. **Shortcut rejected**: representing a hole as a Python list the runner iterates. **Alternative**: the hole is a rank-`k` superposition state on the tree; imaginary-time evolution under the composed Hamiltonian collapses amplitude onto the correct branch. The structural-superposition marker test (§9) fails loudly — positive entanglement entropy is required — if a classical-lookup shortcut is taken.

### 1.3 No dense operator at scale — factored per-species on 16-dim leaves

Every MERA leaf is ≤16-dimensional (M1 §4). Synthesis Hamiltonian terms are factored per-species-leaf operators (M2 contract). No `16**k` dense operator is materialized for `k>2`. **Shortcut rejected**: building a dense `H` over a synthesis window "because it's small." **Alternative**: M3 calls `mera_window_expectation_factored` with per-leaf 16×16 operators; the dense `mera_window_expectation` is for `k≤2` cross-checks only.

### 1.4 OOM means optimize, not shrink

An out-of-memory or out-of-budget signal is an instruction to *optimize the contraction* (better einsum path, factored form, causal-cone restriction), never to reduce node count `N`, bond dimension `χ`, or the candidate set. **Shortcut rejected**: dropping a synthesis problem from P1–P8 or trimming its candidate set to dodge a memory wall. **Alternative**: factor harder.

### 1.5 `optimize='greedy'` on every einsum

Every `numpy.einsum` / `tensordot`-equivalent contraction in M3 passes `optimize='greedy'`. No exceptions.

### 1.6 Holes are quantum superpositions, not classical enumeration

This is the load-bearing principle of M3. The MPS-era E (`2026-05-21-stlc-synthesis-design.md` §1.3) already forbade classically enumerating *binder-name* candidates. M3 extends the prohibition to **structural** candidates: a `HoleVar` whose candidates are AST **sub-trees** (`App(Var,Var)`, `If(...)`, `Bin("+",Var,IntLit)`) is encoded as ONE genuine MERA superposition state spanning a multi-leaf region — never a Python `for` loop over candidate sub-trees, each encoded and evolved separately.

**Shortcut rejected**: "for each candidate sub-tree shape, encode a product state, evolve it, keep the lowest energy." This collapses synthesis to enumerate-and-filter, scales as the product of branching factors, and discards the architectural claim that *ground-state finding is synthesis*. **Alternative**: §5 — the distinct candidate sub-tree shapes become one rank-`k` superposition on the tree via the structural-superposition encoder extension; one evolution; one sampling pass.

### 1.7 Synthesis ranks by ⟨H⟩; success is verified by decoding

A completion's score is `⟨H_typing + H_eval + H_examples + H_target_type + H_size⟩` on the re-encoded concrete completion. There is no auxiliary heuristic scorer (LLM perplexity, edit distance, hand-tuned weights outside the Hamiltonian). **But** success of a synthesis problem is *not* declared on energy alone: the top-1 completion's decoded AST is checked for alpha-equivalence against the expected answer (§7). Energy ranks; decoding verifies. **Shortcut rejected**: declaring P-`i` solved because `⟨H⟩ < ε` without decoding and alpha-checking the AST. **Alternative**: §6.3 + §7 — rank by energy, then verify by structural decode.

### 1.8 The debugger is a generic reporter, not a Hamiltonian re-implementation

M3's `MeraDebugger` consumes named per-term operators (M2's `.terms`) and reports per-term residual energy. It does not re-derive typing/eval rules, does not peek inside the Hamiltonian to break out subterms, does not branch on `isinstance(H, MeraTypingHamiltonian)`. **Shortcut rejected**: inlining STLC rule knowledge into the debugger so it "knows" which AST site is at fault. **Alternative**: §3 — the debugger reuses D's generic `NamedHamiltonianTerm` Protocol, retargeted to MERA states; rule-specific knowledge lives only in the explanation registry the rule authors (M2) populate.

### 1.9 Reuse M1/M2; do not reinvent

M1's `encode_mera`/`decode_mera`/`sample_mera`/`mera_window_expectation_factored` and M2's `MeraTypingHamiltonian`/`MeraEvalHamiltonian`/`compose_mera_hamiltonians`/factored imaginary-time evolution are the only ways M3 touches those layers. **Shortcut rejected**: reimplementing a typing rule or a sampler inline because "the M1/M2 API doesn't quite expose what I need." **Alternative**: if a genuine API gap blocks M3, fix it in M1/M2 and amend that spec; otherwise use what is there.

### 1.10 Honest reporting; negative results are publishable

If a synthesis problem produces a top-1 that is ill-typed, the demo prints `top1 ill-typed; residual H_typing = X` rather than silently dropping it. P8 is *deliberately unsolvable* under its candidate set; success on P8 means the demo correctly refuses it. **Shortcut rejected**: quietly filtering ill-typed completions so the printed result looks better. **Alternative**: §7 — every problem reports its full energy distribution and failure mode.

---

## 2. What this supersedes, and where M3 sits

The revised Phase-1 chain (M1 §2):

| Piece | Scope | Depends on |
|---|---|---|
| M1 | MERA-native logic encoder | F |
| M2 | MERA-native typing + evaluation Hamiltonians + factored imaginary-time evolution | M1 |
| **M3** (this spec) | MERA-native debugger + synthesis; absorbs structural superposition over AST sub-trees | M2 |

M3 retargets sub-project D and sub-project E onto MERA **and** delivers the structural-superposition extension that the MPS-era E (`2026-05-21-stlc-synthesis-design.md`) could not: there, `HoleVar.candidates` was `list[str]` (binder names), so a hole occupied a single MPS site and could only be completed by a single `Var`. Synthesis problems whose answer is a multi-node sub-tree — `f x` = `App(Var,Var)`, `if x<5 then x else x+1` = `If(Bin("<",Var,IntLit),Var,Bin("+",Var,IntLit))` — were structurally unreachable. The MPS-era E acceptance was 4/8 for exactly this reason. M3 lifts the wall.

### 2.1 In scope

- `MeraDebugger` / a `diagnose` retarget — generic over any MERA Hamiltonian exposing M2's named-term contract; emits a JSON `DiagnosticReport`. Reuses D's generic `NamedHamiltonianTerm` Protocol design.
- The **structural-superposition encoder extension** (§5): `HoleVar.candidates` upgraded to admit `list[Node]` (AST sub-trees), not only `list[str]`. The MERA encoder allocates a multi-leaf-span rank-`k` superposition over the distinct candidate sub-tree shapes — one genuine entangled tree state, no Python enumeration. This is an extension to M1's encoder, specified exactly in §5.
- `TypeHole` superposition (a type-position hole) on the MERA `type` register, analogous to the structural `bid` superposition.
- The MERA synthesis runner (§6): encode sketch-with-holes → compose `H_typing + H_eval + H_examples + H_target_type + H_size` → factored imaginary-time evolve → `sample_mera` → rank by re-encoded ⟨H⟩.
- The publishable demo `demo_mera_stlc_synthesis.py` running the 8 synthesis problems P1–P8 (§7).
- An acceptance test suite (§9).

### 2.2 Out of scope

- The typing/eval rules themselves (M2).
- The factored imaginary-time evolution loop (M2 owns it; M3 calls it).
- Natural-language → `SynthesisProblem` conversion (sub-project G / a later migration piece).
- LLM-in-the-loop reranking.
- Lemma library / abstraction discovery (§10.8–10.11; later phases).

### 2.3 Will not do, even if asked later

- Replace the imaginary-time evolution with classical enumeration "for the small problems" — §1.6.
- Add an LLM proposal stage.
- Replace `sample_mera` with per-leaf marginal argmax — holes are entangled; independent per-leaf decisions collapse correlations and produce ill-formed programs.
- Add a "fallback enumerator" when imaginary-time fails — if it fails, M3 reports the failure (§1.10).
- Encode candidate sub-trees as a Python list the runner walks (§1.6).
- Modify `src/qft_pcn/qft/mps.py`, the MPS `logic/` modules, or the MPS-era `debugger.py` / `synthesis/`.

---

## 3. The MERA constraint debugger

### 3.1 Reused contract — `NamedHamiltonianTerm`

M3 reuses the generic Protocol from D's spec (`2026-05-21-constraint-debugger-design.md` §3) verbatim in intent, retargeted to MERA states. M2's Hamiltonians (`MeraTypingHamiltonian`, `MeraEvalHamiltonian`, and the composed object) expose `.terms` — an iterable of objects each satisfying:

```python
# src/qft_pcn/logic/mera_debugger.py

from typing import Protocol, runtime_checkable
from src.qft_pcn.qft.mera import MERA

@runtime_checkable
class NamedMeraTerm(Protocol):
    """One named, individually measurable MERA Hamiltonian term.

    Mirrors D's NamedHamiltonianTerm but its expectation() takes a MERA
    state and is computed via mera_window_expectation_factored — never a
    dense window operator (principle §1.3).

    Attributes:
        name:       stable id, e.g. "T-App@node_5".
        rule_class: reusable rule id, e.g. "T-App", "E-Beta".
        node:       primary AST node index the term diagnoses.
        leaves:     full tuple of MERA leaf indices the term touches.
    """
    name: str
    rule_class: str
    node: int
    leaves: tuple[int, ...]

    def expectation(self, state: MERA) -> float:
        """Re <state|H_term|state> as a Python float. Imaginary residual
        above 1e-10 raises ValueError (a non-Hermitian term is an M2 bug)."""
        ...
```

The debugger asserts protocol conformance at the entry of `diagnose` (`@runtime_checkable`) and names the offending term on failure. M2 owns the contract; M3 consumes it. No `isinstance` dispatch on Hamiltonian type inside the debugger (§1.8).

### 3.2 `diagnose`

```python
# src/qft_pcn/logic/mera_debugger.py

def diagnose(
    state: MERA,
    meta: MeraEncodingMeta,
    hamiltonian_terms: list[NamedMeraTerm],
    threshold: float = 1e-6,
    sort: str = "descending",
) -> DiagnosticReport:
    """Compute a structured diagnostic report from per-term residual
    energies on a MERA state.

    Read-only on all arguments. Behavioural contract identical to D's
    diagnose (2026-05-21-constraint-debugger-design.md §4):
      - total_energy is the sum of every term's expectation, before
        threshold filtering.
      - rule_violations holds terms with |<H_term>| >= threshold, ordered
        per `sort` ("descending" | "ascending" | "node").
      - a term whose expectation() raises is captured into report.errors,
        not propagated; other terms still evaluated.
      - ast_path lookup uses meta.site_to_ast_path keyed by the term's
        `node`; a missing key yields ast_path=None, lookup_failed=True
        (no raise).
    Raises TypeError (non-conforming term, naming the index + missing
    attribute) and ValueError (threshold < 0, bad sort).
    """
```

### 3.3 Report data shape

`DiagnosticReport`, `RuleViolation`, `TermEvaluationError` are reused from D's spec §5 with one rename: the AST-anchor field is `node: int` (a MERA AST node index) rather than `site: int` (an MPS site). All dataclasses are `frozen`, JSON-serializable (`to_dict`, `to_json`, `from_dict`), no numpy scalars, no enums, no tuple keys (§1.3 of D's spec — JSON-friendly). The MERA-leaf tuple is exposed as `leaves: list[int]` in `RuleViolation` (was `sites` in D).

```python
@dataclass(frozen=True)
class RuleViolation:
    name: str
    rule_class: str
    node: int                       # primary AST node index
    leaves: list[int]               # MERA leaves the term touches
    energy_contribution: float
    ast_path: list[int] | None
    lookup_failed: bool
    explanation: str
    context: dict
    def to_dict(self) -> dict: ...
```

`DiagnosticReport` carries `total_energy`, `threshold`, `n_terms_evaluated`, `n_violations`, `rule_violations`, `by_rule_class`, `errors` — identical semantics to D §5.1.

### 3.4 Explanation registry

Reused verbatim from D §6: a module-level `register_explanation(rule_class, template, context_extractor)` / `get_explanation` / `clear_explanations`. M2's typing/eval modules populate it at import time. M3 bundles the same minimal STLC seed templates D bundled (`T-Var`, `T-Abs`, `T-App`, `T-If`, `T-Bin`, `T-IntLit`, `T-BoolLit`) so the debugger acceptance tests run end-to-end even if M2's registration is incomplete; M2 overwrites the registry when it ships.

### 3.5 What M3 reuses vs. rewrites

The debugger is **D's design, retargeted**. The Protocol, the report dataclasses, the registry, the `diagnose` behavioural contract, the pretty-printer (`format_report`) are structurally identical to D. The only changes are: `MERA` instead of `MPS`; `node`/`leaves` instead of `site`/`sites`; expectations computed via `mera_window_expectation_factored` instead of `MPS.local_expectation`/`bond_op`. M3 does not invent a new debugger design. The shortcut of "write a fresh MERA-specific debugger" is rejected — D's generality is the §10.6 architectural claim and M3 preserves it.

---

## 4. The synthesis Hamiltonian on MERA

M3 does not invent typing/eval terms — those come from M2. M3 composes them and adds synthesis-specific constraint terms, exactly as the MPS-era E did (`2026-05-21-stlc-synthesis-design.md` §4), retargeted to the factored MERA window contract.

```
H_total = w_T · H_typing(meta)                 # M2: MeraTypingHamiltonian
        + w_E · H_eval(meta)                   # M2: MeraEvalHamiltonian
        + w_X · H_examples(meta, problem.examples)
        + w_Y · H_target_type(meta, problem.target_type)
        + w_S · H_size(meta)
```

Composition is via M2's `compose_mera_hamiltonians`, which produces a single object exposing `.terms`, `.term_energy`, `.residuals`, `.total_energy` (M2 contract — mirrors B/C). The synthesis-specific blocks are themselves lists of `NamedMeraTerm` so the debugger reports them per-block.

### 4.1 Recommended weights (defaults)

| weight | default | rationale |
|---|---:|---|
| `w_T` | 4.0 | type errors are the strongest signal |
| `w_E` | 2.0 | evaluation correctness, downstream of typing |
| `w_X` | 3.0 | per-example penalty; demo problems use 1–3 examples |
| `w_Y` | 2.0 | active only when `target_type` is set |
| `w_S` | 0.1 | gentle Occam preference; breaks ties between `f x` and `f (f x)` |

These are starting points, not test-set-tuned hyperparameters. The demo prints the per-block breakdown so a human sees which constraint pushed which way. If §9 fails because a weight is wrong, the fix lives here in this spec.

### 4.2 `H_examples` — witness regions on the leaf layout

For each `IOExample(inputs, output)`, the sketch's top-level expression `e` must satisfy `e(inputs...) ⇓ output`. M3 reuses the MPS-era E witness-region idea (`2026-05-21-stlc-synthesis-design.md` §4.2), retargeted to the node-major species-leaf layout:

- The encoder is called on a **witness-augmented AST** (`_witness_augmented_ast(sketch, examples)`) that appends, after the sketch's nodes, one `App(... App(RefVar(root), in_0) ..., in_{n-1})` sub-tree per example. `RefVar` is an encoder-internal node kind whose `bid` leaf is entangled with the sketch root's `bid` leaf — the witness *shares* its function sub-tree with the sketch through the tree, not by classical copy (§1.2; the shortcut of duplicating the encoding into the witness region is rejected).
- Each example contributes a one-node **boundary pin**: a factored term on the witness-root's `value` leaf, `λ·(I − |output⟩⟨output|)` with `λ = w_X`.
- `meta.witness_node_ranges: list[range]` flags the witness nodes so the decoder ignores them when reconstructing the synthesized AST.

Witness nodes are part of the same MERA tree; the Hamiltonian operates over all leaves. The witness term is a factored per-leaf operator — no dense window object (§1.3).

### 4.3 `H_target_type`

When `problem.target_type` is set, a one-node factored term pins the program root's `type` leaf:

```
H_target_type = w_Y · (I − |target_tag⟩⟨target_tag|)   on the type leaf of node 0
```

For a nested arrow type that does not fit a flat tag, the term pins the nested-type table entry (M1's extended `type` basis). For the §7 problems all target types fit the flat tag set.

### 4.4 `H_size`

A gentle Occam penalty: each non-PAD node contributes `w_S·⟨I − P_PAD⟩` on its `kind` leaf, summed over sketch nodes (witness nodes excluded — they are fixed by the problem). This breaks energy ties between competing well-typed completions in favour of the smaller one. `w_S = 0.1` is intentionally small so it never overwhelms a typing violation (`w_T = 4`).

### 4.5 Assembly

```python
# src/qft_pcn/logic/mera_synthesis/hamiltonian.py

def compile_mera_synthesis_hamiltonian(
    meta: MeraEncodingMeta,
    problem: SynthesisProblem,
    weights: HamiltonianWeights | None = None,
):
    """Compose H_typing (M2) + H_eval (M2) + H_examples + H_target_type +
    H_size into a single composed Hamiltonian via M2's
    compose_mera_hamiltonians. The result exposes .terms / .term_energy /
    .residuals / .total_energy (M2 contract). Every synthesis-specific
    block is a list of NamedMeraTerm so diagnose() reports it per block.
    """
```

---

## 5. Structural superposition over candidate AST sub-trees — the load-bearing extension

This is the new piece M3 delivers and the §10.7 milestone gap it closes. **Read this section in full before implementing anything.**

### 5.1 The problem

A `HoleVar` in the MPS-era E carried `candidates: tuple[str, ...]` — binder *names*. The hole occupied one MPS site; its only degree of freedom was the `bid` register. It could be completed by exactly one `Var(name)`. There was no way to express "this hole might be `App(Var,Var)`, or `Bin("+",Var,IntLit)`, or `If(...)`" — completions that are *multi-node sub-trees*. P3 (`f x`), P4 (`if x<5 then x else x+1`), P6 (`f (g x)`), P7 (`x < y`) all need multi-node completions and were therefore unreachable.

M3 fixes this. `HoleVar.candidates` is upgraded:

```python
@dataclass
class HoleVar(Node):
    """A hole. `candidates` may be:
      - tuple[str, ...]   : binder-name candidates (the old, single-Var case);
      - tuple[Node, ...]  : AST sub-tree candidates (the new structural case).
    The two forms are not mixed in one HoleVar. An empty tuple means
    "any in-scope binder" (resolved at encode time, single-Var form only).
    """
    candidates: tuple = ()
    target_type: Ty | None = None
    name: str = ""
```

A structural `HoleVar` whose candidates are sub-trees of varying shapes — `Var(...)`, `App(Var,Var)`, `Bin("+",Var,IntLit)`, `If(...)` — must encode to **one** MERA superposition state spanning the leaf region the hole occupies. Not a Python loop. Not `k` separately-encoded states. One genuine entangled tree state.

### 5.2 The leaf region a structural hole occupies

M1's encoder lays each AST node onto 5 consecutive species-leaves, node-major (M1 §4.3). A structural hole with `k` candidate sub-trees of (possibly different) node counts `n_1, …, n_k` is allocated a **hole region** of `n_max = max_j n_j` AST-node slots — `5·n_max` consecutive leaves — at the hole's pre-order position. The sketch's pre-order serialization reserves this region: nodes after the hole are shifted to start at `hole_position + n_max`.

`n_max` is bounded and small (the §7 problems have `n_max ≤ 4`: `If(Bin,Var,Bin)` is 4 nodes counting the `If` and its three children appropriately — see §7.1). The hole region is part of the single MERA tree (M1 §4.4); structural superposition is therefore representable — the candidate shapes correlate `kind` leaves across the region, and one tree can carry that correlation.

### 5.3 The structural-superposition state — exact construction

This generalizes M1's `from_term_superposition` rank-`k` branch construction (M1's `_mera_holes.py` docstring — the "analytic branch realized exactly"). M1's holes superpose over **leaf values** at a fixed shape; M3's structural holes superpose over **shapes** too.

A structural `HoleVar` with `k` candidate sub-trees `c_1, …, c_k` defines a rank-`k` superposition

```
|ψ_hole⟩ = (1/√k) Σ_j  |branch_j⟩
```

where `branch_j` is the concrete leaf assignment of the hole region **committed to candidate `c_j`**:

1. **Shape leaves.** Candidate `c_j` is serialized in pre-order to `n_j` nodes; node `m` of `c_j` (for `m < n_j`) writes its five species-leaf one-hot vectors (`kind`, `type`, `bid`, `value`, `tobl`) into hole-region node slot `m` exactly as M1's `node_leaf_vectors` does for a concrete node. The remaining `n_max − n_j` node slots of the region (the "shape-pad" slots for this branch) are filled with PAD-leaves. **Two branches `j ≠ j'` with different sub-tree shapes therefore differ on the `kind` leaves of the hole region** — `KIND_APP` vs `KIND_VAR` vs `KIND_BIN` vs `KIND_PAD` at the same leaf index. That difference is what makes the state genuinely structural-entangled, not a value-only superposition.

2. **`bid` resolution.** Where a candidate sub-tree contains a `Var`, its `bid` leaf holds the depth-relative bid index of the binder it references (resolved against the sketch's lexical scope at the hole position). Different candidates referencing different binders differ on those `bid` leaves — the M1 binding-as-entanglement mechanism, now inside a structural branch.

3. **Witness mark.** As in M1's `_mera_holes.py`: each branch `j` writes a distinct nonzero witness index on an otherwise-unused leaf (the hole-region root node's `value` leaf), recording "this branch is candidate `j`." Without the witness mark, two branches that happened to differ only on `bid` could degenerate toward a product state with a few superposed leaves; the witness mark guarantees the rank-`k` directions are mutually orthogonal and the entanglement is genuine across **any** cut separating the hole region from the rest of the tree.

4. **Tree construction.** The `k` branch leaf-assignments are handed to M1's `MERA.from_term_superposition` (the same primitive M1's value-superposition holes use). It builds the tree exactly: identity disentanglers where branches agree, isometries whose bond dimension grows to span the `k` branch directions over the leaves where branches differ, and a rank-`k` top tensor. **No large dense tensor is formed** — every per-leaf vector is 16-dimensional and only `k` branches are tracked. The isometry bond dimension over the hole region is bounded by `k` (≤ the candidate count, ≤ 8 for the §7 problems); `χ_layer` is raised to accommodate it (§5.6).

This is the load-bearing construction. It is a **genuine quantum superposition over distinct AST sub-tree shapes**, encoded once, with measurable entanglement entropy. There is **no Python enumeration** anywhere: the runner never iterates candidate sub-trees; the candidates exist only as the `k` branch directions inside one MERA state, and imaginary-time evolution under `H_total` redistributes amplitude across those directions.

### 5.4 Why this is not five trees, and why one tree suffices

A structural candidate differs from another in `kind`, `type`, and `bid` leaves jointly: `App(Var,Var)` and `Bin("+",Var,IntLit)` differ in the hole-region root's `kind`, in the children's `kind`, and in their `bid`/`value`. The hole region's leaves are entangled *with each other across species*. Five independent per-species trees factorize across species and cannot represent this (M1 §4.4, §5.4). The single MERA tree can, because the hole region's leaves are contiguous and node-major, so layer-0 disentanglers entangle `kind`/`type`/`bid` of the same node and low-layer isometries entangle across the region's nodes.

### 5.5 `TypeHole` superposition

A `TypeHole(candidates: tuple[Ty, ...])` is the type-position analog: a single-node hole whose `type` leaf carries `(1/√k) Σ_j |type tag of c_j⟩`, and whose resolution is correlated through the tree with every node whose typing computation flows through the hole (M1's type-channel mechanism). It is the §10.7 type-synthesis case (P5). The construction is M1's value-superposition hole specialized to the `type` register — no new machinery beyond M1; M3 only wires `TypeHole` into the encoder's hole detection.

### 5.6 Encoder change M3 requires of M1's encoder — exact statement

M3 extends M1's `encode_mera` with a **structural-hole branch**. The exact change:

- **M1's `_has_holes`** already detects `HoleVar`. M3 adds a classifier `_hole_kind(node) -> {"var", "structural", "type"}`: a `HoleVar` with `tuple[str,...]` candidates (or empty) is `"var"`; a `HoleVar` with `tuple[Node,...]` candidates is `"structural"`; a `TypeHole` is `"type"`.
- **Layout.** M1's `compute_layout` is given the **expanded node count**: each structural hole contributes `n_max` node slots instead of 1. M3 supplies a layout pre-pass `_expand_structural_holes(ast) -> (expanded_ast_skeleton, hole_regions)` that computes, per structural hole, its pre-order position and `n_max`. `hole_regions: list[HoleRegion]` where `HoleRegion` records `(node_start, n_max, candidate_branches)` and `candidate_branches` is the list of `k` per-branch leaf-assignment specs.
- **Leaf vectors.** M1's concrete path builds `node_leaf_vectors` per node. M3's structural path builds, per structural hole, the `k` branch leaf-assignments of §5.3 and threads them — alongside the concrete nodes' single assignments — into `MERA.from_term_superposition`. Concrete nodes contribute a single (degenerate rank-1) branch direction; structural-hole regions contribute `k`. M1's `from_term_superposition` already accepts a list of branches; M3 supplies a richer branch list. **This is the one encoder change.** M1's `from_product` concrete path and M1's value-superposition `_mera_holes` path are untouched and still selected for hole-free and var-hole programs respectively.
- **`χ_layer`.** Raised from M1's default 16 to **32** for synthesis, to carry the rank-`k` isometry bonds over hole regions. Programs that would exceed 32 raise `EncodingTooLarge` (M1's exception family, reused).
- **Meta.** `MeraEncodingMeta` gains `hole_regions: list[HoleRegion]` and `witness_node_ranges: list[range]`. The decoder and runner consume these. No other `MeraEncodingMeta` field changes meaning.

If during implementation M1's `from_term_superposition` turns out not to accept shape-varying branches (branches whose non-PAD leaf *count* differs), that is a genuine M1 gap: fix it in M1's `_mera_holes.py` / `mera.py` and amend M1's spec — do not work around it with a Python enumeration (§1.6, §1.9).

### 5.7 Decoder / `sample_mera` recovery of sub-tree structure

`sample_mera` (M1) draws conditional samples from the MERA. For a structural-hole state each sample collapses the hole region onto one branch `j`; the decoder's structural parse (M1's shared `parse_kind_stream`) reads the hole region's `kind` leaves and reconstructs the chosen sub-tree shape directly — the `KIND_PAD` shape-pad slots are dropped, the non-PAD slots parse as ordinary nodes. No M3-specific decoder logic is needed beyond skipping `witness_node_ranges`. The structural superposition is invisible to the decoder: it just sees, per sample, a concrete program.

### 5.8 The structural-superposition marker (acceptance, §9)

For a structural-hole sketch, the MERA's `entanglement_entropy` across a cut separating the hole region from the rest of the tree is strictly positive — and, because branches differ in *shape* (`kind` leaves), the entropy is strictly larger than it would be for a value-only (M1-style) superposition over the same candidate count. A classical-enumeration shortcut (the runner looping over candidate sub-trees, each a product state) would give zero entropy on every individual encoded state. The marker test (§9.4) fails loudly if the shortcut is taken. This is the §1.1-class principle made measurable for M3.

---

## 6. The synthesis runner

### 6.1 Data types

```python
# src/qft_pcn/logic/mera_synthesis/problem.py

@dataclass(frozen=True)
class IOExample:
    inputs: tuple[Node, ...]      # IntLit / BoolLit / NatLit literals
    output: Node

@dataclass(frozen=True)
class SynthesisProblem:
    sketch: Node                  # AST with HoleVar / TypeHole nodes
    target_type: Ty | None = None
    examples: tuple[IOExample, ...] = ()
    name: str = ""
    n_nodes_max: int = 32
    chi_layer: int = 32           # raised from M1's 16 for hole superpositions
    n_samples: int = 64
    anneal_steps: int = 200
    anneal_dt: float = 0.05

@dataclass(frozen=True)
class Completion:
    ast: Node
    energy: float
    energy_breakdown: dict[str, float]   # "typing"/"eval"/"examples"/
                                         # "target_type"/"size"
    diagnostics: dict                    # from diagnose()
    multiplicity: int

@dataclass(frozen=True)
class SynthesisResult:
    problem: SynthesisProblem
    completions: list[Completion]        # sorted ascending by energy
    n_unique: int
    n_samples_drawn: int
    n_samples_decoded_ok: int
    final_state_energy: float            # <H> on the relaxed state
    chi_observed_max: int
    failure_mode: str | None             # None | "no_valid_completion" |
                                         # "ambiguous_top1" |
                                         # "imag_time_did_not_converge"
```

Invariants on a well-formed `SynthesisProblem`: the sketch has at least one hole; every structural `HoleVar` lists `tuple[Node,...]` candidates whose sub-trees are well-scoped against the sketch's lexical context at the hole position; every `TypeHole` lists flat-tag `Ty` candidates; every non-hole `Var` is lexically scoped. Violations raise `SynthesisProblemError`.

### 6.2 `synthesize`

```python
# src/qft_pcn/logic/mera_synthesis/runner.py

def synthesize(problem: SynthesisProblem,
               rng: np.random.Generator | None = None,
               verbose: bool = False) -> SynthesisResult:
    """MERA-native synthesis pipeline:
      1. Validate the problem (§6.1 invariants).
      2. Build the witness-augmented AST (§4.2) and encode it with
         encode_mera — structural holes become rank-k superposition
         states (§5), TypeHoles become type-register superpositions.
      3. compile_mera_synthesis_hamiltonian (§4.5).
      4. Factored imaginary-time evolution (M2) with the §6.5 schedule.
      5. sample_mera (M1) for n_samples completions.
      6. Dedupe by alpha-equivalence; per unique completion re-encode the
         concrete AST and compute <H_total> + per-block energies.
      7. Sort ascending; classify failure_mode; assemble SynthesisResult.

    Raises SynthesisProblemError (malformed problem) and
    SynthesisRuntimeError (NaN energies, evolution divergence, zero
    decodable samples). A problem with no good completion is NOT raised —
    it is reported via failure_mode (§1.10).
    """
```

### 6.3 Sampling and ranking

- **Draw.** `sample_mera(state, meta, n_samples=problem.n_samples, rng=rng)` after evolution. Default `n_samples = 64`.
- **Dedupe.** Drop samples with decode residual > 1e-3. Group survivors by alpha-equivalence (M1's `ast_alpha_eq`); the group representative is the alpha-normalized form. `multiplicity` is the group's sample count.
- **Rank.** For each unique completion: re-encode the concrete AST (`encode_mera`, no holes → product MERA), recompile `H_total` on the new meta, compute `⟨H_total⟩` and per-block energies. Sort ascending by energy; ties within 1e-6 break by multiplicity (desc) then AST size (asc).
- **Verify (§1.7).** The top-1's decoded AST is the answer; the demo checks it against the expected program by alpha-equivalence. Energy ranks; decoding verifies. A completion is never declared correct on energy alone.

### 6.4 `failure_mode`

- `None`: top-1 energy < `tolerance_correct = 1e-3·(w_T+w_E+w_X+w_Y)` — the dominant constraints are satisfied (size penalty alone may remain nonzero).
- `"no_valid_completion"`: zero unique completions decoded.
- `"ambiguous_top1"`: top-1 and top-2 differ by < `1e-3` and top-1 is well-typed.
- `"imag_time_did_not_converge"`: `final_state_energy` above threshold after the full anneal; diagnose's per-term breakdown is still printed.

### 6.5 Annealing schedule

Three sequential factored-imaginary-time passes (M2 owns the evolution primitive):

```
phase 1 (warmup):  50 steps, dt = 0.1,  H_typing + H_eval only
phase 2 (main):   100 steps, dt = 0.05, all blocks at full weight
phase 3 (fine):    50 steps, dt = 0.01, all blocks
```

`problem.anneal_steps`/`anneal_dt` override the main-phase totals; warmup/fine scale proportionally with floors of 25/25. Every step normalizes; bond dimension is capped at `problem.chi_layer`. The warmup-before-full-H rationale is M2's: full weight from step 0 can lock the state into a typing-violating configuration before example constraints can bend it.

---

## 7. The 8 synthesis problems (acceptance test set)

`demo_mera_stlc_synthesis.py` runs these eight, escalating in difficulty. **P3, P4, P6, P7 are the multi-node-completion problems** that the structural superposition (§5) makes reachable — they were the unreachable half of the MPS-era 4/8 wall.

### P1 — Identity completion (warmup; var-hole)

```
sketch:   \x:Int. ?HOLE          HoleVar(candidates=("x",))   [var-hole]
expected top-1:  \x:Int. x
```

Single-`Var` completion; exercises the M1 var-hole path through the M3 runner.

### P2 — Constant-vs-identity disambiguation by type (var-hole)

```
sketch:   \x:Int. ?HOLE          HoleVar(candidates=())  -> all in-scope vars + int lits
target_type: Int -> Int
examples: IOExample((IntLit(3),), IntLit(3))
expected top-1:  \x:Int. x
```

### P3 — Use the argument: `f x` (STRUCTURAL hole — multi-node)

```
sketch:   \f:Int->Int. \x:Int. ?HOLE
HoleVar(candidates=( Var("x"),
                     App(Var("f"), Var("x")),
                     App(Var("f"), App(Var("f"), Var("x"))) ))
examples: IOExample((Lam("n",Int,Bin("+",Var("n"),IntLit(1))), IntLit(2)), IntLit(3))
expected top-1:  f x
```

The candidate set mixes a 1-node sub-tree (`Var`), a 2-node sub-tree (`App(Var,Var)`), and a 3-node sub-tree (`App(Var,App(Var,Var))`). `n_max = 3`. This is the canonical §10.7 example and the first problem **unreachable** to the MPS-era E.

### P4 — Conditional synthesis: `if x<5 then x else x+1` (STRUCTURAL hole)

```
sketch:   \x:Int. ?HOLE
HoleVar(candidates=( If(Bin("<",Var("x"),IntLit(5)), Var("x"),
                        Bin("+",Var("x"),IntLit(1))),
                     If(Bin("<",Var("x"),IntLit(3)), Var("x"),
                        Bin("+",Var("x"),IntLit(1))),
                     Bin("+",Var("x"),IntLit(1)) ))
examples: IOExample((IntLit(2),), IntLit(2))
          IOExample((IntLit(7),), IntLit(8))
expected top-1:  \x:Int. if (x < 5) then x else (x + 1)
```

`n_max` is the node count of the largest candidate `If(Bin,Var,Bin)`. Tests structural superposition over candidates of differing internal shape; the examples disambiguate the threshold branch.

### P5 — Type-hole synthesis (TypeHole)

```
sketch:   \x:?T. x               TypeHole(candidates=(TInt, TBool))
target_type: Bool -> Bool
expected top-1:  \x:Bool. x
```

Pure type-level synthesis; exercises §5.5.

### P6 — Curried composition: `f (g x)` (STRUCTURAL hole)

```
sketch:   \f:Int->Int. \g:Int->Int. \x:Int. ?HOLE
HoleVar(candidates=( App(Var("f"), App(Var("g"), Var("x"))),
                     App(Var("g"), App(Var("f"), Var("x"))),
                     App(Var("f"), Var("x")) ))
examples: IOExample over concrete f,g,x picked so f(g(x)) = expected, g(f(x)) != expected.
expected top-1:  f (g x)
```

Two-level composition; structural superposition over 3-node and 2-node `App` chains.

### P7 — Boolean synthesis with mixed types: `x < y` (STRUCTURAL hole)

```
sketch:   \x:Int. \y:Int. ?HOLE
HoleVar(candidates=( Bin("<",Var("x"),Var("y")),
                     Bin("<",Var("y"),Var("x")),
                     Eq(Var("x"),Var("y")) ))
target_type: Int -> Int -> Bool
examples: IOExample((IntLit(2),IntLit(3)), BoolLit(True))
          IOExample((IntLit(5),IntLit(3)), BoolLit(False))
expected top-1:  x < y
```

Structural superposition where candidates differ in the root node kind (`Bin` vs `Eq`); the target type and examples select.

### P8 — Negative case (deliberately unsolvable)

```
sketch:   \x:Int. ?HOLE          HoleVar(candidates=( Var("x"), ))   [var-hole]
target_type: Int -> Bool
examples: IOExample((IntLit(2),), BoolLit(True))
expected:  SynthesisResult.failure_mode != None;
           the demo prints diagnose()'s residual H_typing breakdown
           identifying the offending term.
```

The only candidate (`x`, an `Int`) cannot satisfy a `Bool` return type. P8 tests §1.10 — the demo must not lie about success.

### 7.1 Node / leaf budget

| problem | sketch nodes | hole `n_max` | witnesses | total nodes | leaves (×5, padded) | within `n_nodes_max=32`? |
|---:|---:|---:|---:|---:|---:|---:|
| P1 | 2 | 1 (var) | 0 | 2 | 16 | ✓ |
| P2 | 2 | 1 (var) | 3 | 5 | 32 | ✓ |
| P3 | 3 | 3 | 6 | 11 | 64 | ✓ |
| P4 | 1 | 4 | 8 | 12 | 64 | ✓ |
| P5 | 2 | — (type) | 0 | 2 | 16 | ✓ |
| P6 | 3 | 3 | 6 | 11 | 64 | ✓ |
| P7 | 2 | 3 | 8 | 12 | 64 | ✓ |
| P8 | 2 | 1 (var) | 3 | 5 | 32 | ✓ |

All within `n_nodes_max = 32`. The structural-hole `n_max` slots are counted into the sketch's node budget by `_expand_structural_holes` (§5.6).

---

## 8. File layout

```
src/qft_pcn/logic/
├── ast.py                              # MODIFY: HoleVar.candidates admits tuple[Node,...]
├── mera_encoder.py                     # MODIFY: structural-hole branch (§5.6)
├── _mera_holes.py                      # MODIFY: structural-superposition branch builder
├── mera_debugger.py                    # NEW: NamedMeraTerm protocol, diagnose,
│                                       #   DiagnosticReport/RuleViolation/TermEvaluationError,
│                                       #   explanation registry, format_report
└── mera_synthesis/
    ├── __init__.py                     # NEW: public re-exports
    ├── problem.py                      # NEW: SynthesisProblem, IOExample, Completion,
    │                                   #   SynthesisResult, HamiltonianWeights
    ├── encode_ext.py                   # NEW: _expand_structural_holes, HoleRegion,
    │                                   #   _witness_augmented_ast, RefVar
    ├── hamiltonian.py                  # NEW: compile_mera_synthesis_hamiltonian,
    │                                   #   H_examples / H_target_type / H_size builders
    ├── runner.py                       # NEW: synthesize()
    ├── ranking.py                      # NEW: dedupe, rerank, classify failure_mode
    └── errors.py                       # NEW: SynthesisProblemError, SynthesisRuntimeError
src/qft_pcn/logic/demo_mera_stlc_synthesis.py   # NEW: runs P1-P8, structured output
src/qft_pcn/tests/
├── test_mera_debugger.py               # NEW: §9.1 debugger acceptance
├── test_mera_structural_holes.py       # NEW: §9.2-9.4 structural-superposition
├── test_mera_synthesis_hamiltonian.py  # NEW: §9.5 H_examples/target_type/size
├── test_mera_synthesis_runner.py       # NEW: §9.6 P1-P8 integration
└── test_mera_synthesis_demo.py         # NEW: §9.7 demo smoke test
```

`ast.py`, `mera_encoder.py`, `_mera_holes.py` are modified additively (no behaviour change for hole-free / var-hole / type-hole programs that already work in M1). Everything else is new. `mps.py`, the MPS `logic/` modules, the MPS-era `debugger.py` and `synthesis/` are untouched.

Public re-exports (`src/qft_pcn/logic/__init__.py`, additive):

```python
from .mera_debugger import (
    NamedMeraTerm, DiagnosticReport, RuleViolation, TermEvaluationError,
    diagnose, format_report, register_explanation, get_explanation,
    clear_explanations,
)
from .mera_synthesis import (
    synthesize, SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights, SynthesisProblemError, SynthesisRuntimeError,
)
```

---

## 9. Acceptance tests

### 9.1 Debugger

`test_mera_debugger.py` mirrors D's §7 against MERA states and mock `NamedMeraTerm` instances: structured-report shape for an ill-typed program, JSON round-trip (`report == from_dict(to_dict(report))`), threshold filtering and the three sort modes, term-evaluation-error capture, protocol enforcement at entry (`TypeError` naming the offending index), empty-input zero report, partial-state (mid-evolution) report, `ast_path` lookup-failure path (`ast_path=None`, `lookup_failed=True`, no raise), explanation-registry idempotence/conflict/fallback/extractor, pretty-printer sanity. An end-to-end test against the real M2 `MeraTypingHamiltonian` runs if M2 is importable, else `pytest.skip` with a non-misleading reason.

### 9.2 Structural-hole encoding — unit norm and shape superposition

A structural-hole sketch (`\f:Int->Int. \x:Int. ?HOLE` with candidates `{Var("x"), App(Var("f"),Var("x"))}`) encodes to a unit-norm MERA. The hole region's root-node `kind` leaf has nonzero marginal weight on **at least two** distinct kind basis states (`KIND_VAR` and `KIND_APP`) — proving the superposition is over *shapes*, not just values.

### 9.3 Structural-hole decode — every branch is recoverable

For the same sketch, `sample_mera` drawn many times yields, among its decoded completions, every candidate sub-tree shape (the `KIND_PAD` shape-pad slots dropped). No sample decodes to an ill-formed program (decode residual ≤ 1e-3 for the kept samples).

### 9.4 Structural-superposition marker (the §1.1 / §1.6 proof)

For a structural-hole sketch, `entanglement_entropy` across a cut separating the hole region from the rest of the tree is strictly positive and strictly greater than the entropy of an M1-style value-only superposition over the same candidate count (because branches differ in `kind` leaves, not only `bid`). A classical-enumeration encoding (each candidate a separate product state) gives zero entropy — this test fails loudly if the §1.6 shortcut is taken.

### 9.5 Synthesis Hamiltonian blocks

Unit tests for `H_examples` (a witness-root `value`-leaf pin costs +`w_X` when the value is wrong, 0 when right), `H_target_type` (root `type`-leaf pin), `H_size` (per-node `kind`-leaf PAD penalty). Each block's terms satisfy `NamedMeraTerm`; each `expectation` is computed via `mera_window_expectation_factored` — a test asserts no dense `16**k` array (`k>2`) is allocated (the `conftest.py` memory ceiling, reused from M1).

### 9.6 Synthesis runner — P1–P8 (the milestone acceptance)

`test_mera_synthesis_runner.py` runs each of P1–P8 via `synthesize()`:

1. All eight run without raising.
2. **P1, P2, P3, P4, P5, P6, P7 (7 of 8) produce a top-1 completion alpha-equivalent to the expected answer** (§1.7 — verified by decoding, not energy alone). 7/8 is the acceptance floor; 8/8 with P8 also correct (i.e. correctly refused) is the target. **P3, P4, P6, P7 passing is the proof the structural superposition lifted the §10.7 milestone past the MPS-era 4/8 wall.**
3. For each of P1–P7, the second-ranked completion has strictly higher energy than the top-1, gap ≥ `0.5·w_T = 2.0`. On a smaller gap the demo prints `WARNING: P_i has small energy gap` with the distribution.
4. P8 produces `failure_mode != None`; the demo prints `diagnose()`'s residual `H_typing` breakdown identifying the offending term.

### 9.7 Demo smoke test

`test_mera_synthesis_demo.py` runs `demo_mera_stlc_synthesis.py` end-to-end without raising; the printed output for each problem includes problem name, expected top-1, actual top-1, energy, energy_breakdown, multiplicity, n_unique, n_samples, failure_mode.

### 9.8 No regression

All M1 and M2 tests, all F MERA tests, and all shipped MPS-stack tests still pass. No M3 test materializes a tensor larger than `16**2` densely.

---

## 10. Error model

`src/qft_pcn/logic/mera_synthesis/errors.py`:

```python
class SynthesisError(Exception): ...
class SynthesisProblemError(SynthesisError):
    """Malformed SynthesisProblem (§6.1 invariants)."""
class SynthesisRuntimeError(SynthesisError):
    """Evolution diverged / NaN energies / zero decodable samples."""
```

The runner does **not** raise on a problem with no good completion — that is `failure_mode`, not an exception (§1.10). Raises are reserved for malformed input and infrastructure failure. The encoder reuses M1's `EncodingTooLarge` / `TooManyBinders` / `IntLiteralOutOfRange` / `IllScopedVar` / `UnsupportedNode` family; M3 adds no encoder exceptions.

The debugger raises `TypeError` (non-conforming term) and `ValueError` (bad threshold/sort) exactly as D §9; term-evaluation failures are captured into `report.errors`, not raised.

---

## 11. Acceptance criteria

M3 is complete when:

1. The debugger acceptance suite (§9.1) passes; the end-to-end test passes or skips cleanly.
2. The structural-hole encoding tests (§9.2, §9.3) pass.
3. The structural-superposition marker (§9.4) passes — proving §1.1 and §1.6 are realized: candidate sub-trees are a genuine MERA superposition with measurable entanglement entropy, not a Python enumeration.
4. The synthesis-Hamiltonian-block tests (§9.5) pass with the factored-window contract (no dense `16**k`, `k>2`).
5. The runner suite (§9.6) passes: **7 of 8 of P1–P8 produce a correct top-1**, P8 correctly refused. P3/P4/P6/P7 (the multi-node completions) are among the passing problems.
6. The demo smoke test (§9.7) passes.
7. No regression in M1, M2, F, or the MPS stack (§9.8).
8. `from qft_pcn.logic import synthesize, diagnose` works.
9. Every completion claim is backed by fresh pytest output (`superpowers:verification-before-completion`).

---

## 12. Open questions

**None that block implementation.** Two design points are noted for transparency, each with a locked default:

1. **`from_term_superposition` and shape-varying branches.** §5.6's default assumes M1's `from_term_superposition` accepts branches whose non-PAD leaf count differs (the shape-pad slots make every branch the same *leaf* count, all 16-dimensional, so the primitive sees uniform-length branches that merely differ in which leaves are PAD). If, while implementing, the primitive cannot handle the rank growth this implies, the resolution is to **extend M1's primitive and amend M1's spec** — not to fall back to per-candidate enumeration (§1.6, §1.9).
2. **Witness sharing via `RefVar`** vs. classical copy of the sketch sub-tree into the witness region. **Default: `RefVar` with its `bid` leaf entangled to the sketch root** (§4.2). Classical copy doubles leaf usage and breaks the architectural claim that the witness *shares* its function sub-tree. If `RefVar` proves impractical, escalate; do not silently swap to classical copy.

If, while implementing, a third real ambiguity surfaces that this document does not resolve — **stop and ask the human.** Do not paper over it.

---

## 13. Contract M3 exposes downstream

- `synthesize(problem: SynthesisProblem) -> SynthesisResult` — the publishable synthesis API; a later LLM-bridge migration piece consumes it as a black box.
- `diagnose(state, meta, terms) -> DiagnosticReport` — the generic constraint debugger; any MERA Hamiltonian exposing `NamedMeraTerm` instances can be diagnosed.
- `HoleVar.candidates` admitting `tuple[Node, ...]` — the structural-superposition AST surface; any future synthesis client builds sketches against it.
- The 8 P1–P8 problems and their printed demo output are the paper's headline figure for the §10.7 milestone.

---

## 14. Glossary (M3-local)

- **Structural hole** — a `HoleVar` whose candidates are AST sub-trees (`tuple[Node,...]`); completable by a multi-node sub-tree.
- **Var hole** — a `HoleVar` whose candidates are binder names (`tuple[str,...]`); completable by a single `Var`. The M1 case.
- **Type hole** — a `TypeHole`; a type-position hole, superposed on the `type` register.
- **Hole region** — the `5·n_max` contiguous leaves a structural hole occupies, `n_max` = max candidate node count.
- **Branch** — one of the `k` directions of a structural hole's rank-`k` superposition; the leaf assignment committing the hole to one candidate sub-tree.
- **Shape-pad slot** — a hole-region node slot filled with PAD-leaves in a branch whose candidate has fewer than `n_max` nodes.
- **Witness region / witness mark** — the leaves encoding example-application sub-trees (`H_examples`) / the distinct nonzero index on a branch's hole-region root `value` leaf recording which candidate the branch is.
- **Structural-superposition marker** — the §9.4 test: positive, shape-distinguishing tree entanglement entropy across a hole-region cut.
