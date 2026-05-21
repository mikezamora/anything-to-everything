# Spec: STLC Synthesis Milestone Demo (Sub-Project E)

**Document type**: Implementation specification (sub-project E of the §10 roadmap).
**Date**: 2026-05-21.
**Author**: Design session, this branch.
**Status**: Approved design, ready for implementation plan.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Architecture doc**: `QFT_PCN_ARCHITECTURE.md` (root). This spec realizes §10.7.
**Acceptance owner**: human review of the synthesis test suite passing and the demo printout matching expectations.

---

## 0. Document purpose

This document is the contract for **sub-project E: the STLC synthesis milestone demo**. It is the publishable milestone of the QPCN logic layer. The §10 roadmap was decomposed into seven sub-projects (A = AST↔MPS encoder, B = typing-rule Hamiltonian, C = evaluation Hamiltonian, D = constraint debugger, **E = this**, F = MERA, G = LLM bridge). E composes A.encode + B.H_typing + C.H_eval + D.diagnose into a working program-synthesis system.

> **Take a simply-typed lambda-calculus program with holes (`?`) at arbitrary positions in either expressions or types. Produce a distribution over valid completions, ranked by the QPCN's energy.**

This is the result the project was built to produce. LLMs are demonstrably weak at type-directed program synthesis (they hallucinate ill-typed completions); tensor-network methods have **never** been tried at this task. With bond dimension `χ = 32` and ~30 AST nodes, this is well within reach of the architectural stack we built. The publishability target is a workshop paper at NeurIPS or ICML on *Tensor-Network Predictive Coding for Program Synthesis*. Even a partial or negative result is publishable — the method is new.

Every section below is part of the contract. If something is missing here that you need to decide while implementing, **stop and ask**. Do not fill in by guessing — see §1.

---

## 1. Driving principles (non-negotiable)

The synthesis demo is the architectural payoff. A subagent will be tempted to take shortcuts that destroy the *point* of the demo. The principles below are non-negotiable. If you find yourself tempted to violate one, stop and ask the human.

### 1.1 End-to-end uses the architectural stack — no shortcuts

The synthesis loop is exactly:

```
encode(program_with_holes)                    # sub-project A
   ──► MPS state on a fixed-N lattice with holes as bid/type superpositions
   ──► H = H_typing + H_eval + H_synthesis    # sub-projects B + C + E's terms
   ──► imag-time evolve (trotter_step with imaginary=True) for T steps
   ──► sample(state, meta, n_samples=K)       # sub-project A
   ──► dedupe completions, rank by ⟨H⟩ per completion
   ──► report top-N with residual diagnostics  # sub-project D
```

The "easy shortcut" — replacing any of these stages with a classical enumerator (e.g. exhaustive search over candidate completions with a type-checker filter) — is rejected. The whole point of E is to demonstrate that the QFT/PCN substrate produces synthesis answers *as a side effect of ground-state finding*. If we wanted enumerate-and-filter we wouldn't have built any of A–D. Negative results from the principled approach are publishable; classical fallbacks are not.

### 1.2 No LLM in the loop

E is the bold core demonstration: the QPCN does program synthesis **without** an LLM. Sub-project G adds an LLM frontend later. E proves the core works standalone.

The "easy shortcut" — calling an LLM to propose candidates and then scoring them with the QPCN — is rejected. That collapses to "LLM does the work, QPCN reranks" which is not a new result and does not test the architectural claim. E must show the QPCN producing completions de novo from the encoded constraint structure.

### 1.3 Holes are quantum superpositions, not enumeration placeholders

A `HoleVar` (variable-position hole) is encoded by sub-project A as an equal-amplitude superposition over candidate binders on the `bid` register (§5.4 of A's spec, the "Bell-pair / superposition extension" paragraph). A `TypeHole` (type-position hole, introduced by this sub-project) is encoded as an equal-amplitude superposition over candidate type tags on the `type` register. Both are **quantum** superpositions on the MPS, not classical enumerations the runtime walks.

The "easy shortcut" — enumerating candidate completions classically, encoding each as a product state, evolving each, and picking the best — is rejected. It loses the architectural claim ("ground-state finding *is* synthesis") and scales as the product of branching factors, not as the bond dimension. The Hamiltonian must drive the *superposed* state toward the correct completion via imag-time relaxation.

### 1.4 Ranking is by ⟨H⟩ on relaxed completions, not heuristics

A completion's score is `⟨H_typing + H_eval + H_synthesis⟩` after the system has been allowed to relax (imag-time evolved into the appropriate variational subspace). Lower energy = better completion. There is **no** auxiliary cost function (LLM perplexity, edit distance, hand-tuned weight). The Hamiltonian *is* the scoring function. Tie-breaking ties at energy 0 falls back to ⟨H_size⟩ (a small program-size penalty, §4.4), which is itself part of the Hamiltonian.

The "easy shortcut" — building a custom scorer on top of decoded ASTs — is rejected. If the energy ranking is wrong, the bug is in H_typing/H_eval/H_synthesis, not in a missing post-processor.

### 1.5 The sampler is the principled MPS sampler

We use sub-project A's `sample(state, meta, n_samples, rng)` — the standard left-to-right MPS conditional sampler from A.§6.3. We do not write a custom AST traversal that samples sites independently; we do not write a beam-search decoder.

The "easy shortcut" — independent per-site argmax on local marginals — is rejected. Holes are entangled with their constraint context; independent decisions collapse correlations and produce ill-formed programs. (This is exactly what A.§1.5 protects against.)

### 1.6 Honest reporting; negative results are publishable

If the system handles 30-node STLC programs with hole completions but fails at 50-node ones, the demo prints that. If a synthesis problem produces a top-1 completion that is ill-typed because H_typing wasn't strong enough, the demo prints `top1 ill-typed; H_typing weight may be too low` rather than silently filtering. The acceptance criteria in §10 specify the threshold for "majority of cases" — but the demo *reports* all of them, including the failures.

The "easy shortcut" — quietly dropping ill-typed completions during ranking so the printed result looks better than it is — is rejected.

### 1.7 Reuse A/B/C/D; do not reinvent

A's `encode`/`decode`/`sample`, B's `compile_typing_hamiltonian`, C's `compile_eval_hamiltonian`, D's `diagnose` are the only ways E touches those layers. If a real bug or missing feature in those layers blocks E, fix it in that layer and update its spec; otherwise, use what's there. New code in E goes only under `src/qft_pcn/logic/synthesis/` and `src/qft_pcn/logic/demo_stlc_synthesis.py`.

The "easy shortcut" — reimplementing typing rules inline in the demo because the H_typing API "doesn't quite expose what I need" — is rejected. If H_typing's interface is wrong for synthesis, that is a B-spec amendment, not an E-local hack.

---

## 2. Scope

### 2.1 In scope (this sub-project)

- A formal definition of the **synthesis problem** as a data type: a hole-bearing AST + optional constraints.
- A small extension to A's `ast.py`: a `TypeHole` node and a `HoleVar` constructor with full candidate-list support (A already added the data class minimally; E exercises it).
- A small extension to A's encoder: encoding `TypeHole` as a superposition on the `type` register (analogous to `HoleVar` on `bid`).
- The **synthesis Hamiltonian** `H_synthesis`: a composition of B's `H_typing`, C's `H_eval`, and E-specific constraint terms (example-based constraints, optional target-type pin, size penalty).
- An **annealing schedule** for imag-time evolution: warmup, main pass, optional fine sweep.
- A **sampling/ranking protocol**: draw K samples from the relaxed state, dedupe, compute ⟨H⟩ per unique completion, sort ascending.
- A `synthesize(problem) -> SynthesisResult` API.
- A **demo script** `demo_stlc_synthesis.py` that runs ~8 synthesis problems of escalating difficulty and prints structured output.
- A test suite covering: each demo problem's top-1 correctness, energy strictly ordering correct < wrong, hole encoding round-trip with mock constraint, and synthesis-API error paths.

### 2.2 Out of scope (deferred)

- Recursion / `let rec` / `fix` (deferred to F — MERA — sub-project).
- Polymorphism / Hindley-Milner inference (out of scope entirely; we stay in STLC).
- Hole positions inside *types* deeper than top-level type variables (i.e., `?` inside an arrow type `?->Int` is supported, but `(? -> ?) -> Int` with both holes resolved jointly is the *limit*; nested arrow synthesis with arbitrary depth is deferred).
- Natural-language → DSL conversion (sub-project G).
- LLM-in-the-loop reranking (sub-project G).
- Wall-clock benchmarking or comparison to baseline systems for paper figures (a follow-up evaluation pass after this sub-project lands).

### 2.3 Will not do, even if asked later

- Replace the imag-time evolution with classical enumeration even "for the small problems" — see §1.1.
- Add an LLM proposal stage — see §1.2.
- Replace `sample()` with per-site marginal argmax — see §1.5.
- Add a "fallback enumerator" when imag-time fails — see §1.1, §1.6. If it fails, we report the failure.

---

## 3. The synthesis problem formalization

### 3.1 Data types

```python
# src/qft_pcn/logic/synthesis/problem.py

from dataclasses import dataclass, field
from src.qft_pcn.logic.ast import Node, Ty

@dataclass(frozen=True)
class IOExample:
    """An example-based constraint: applying the synthesized function to
    `inputs` (a tuple of Node literals) should produce `output` (a Node literal).

    The encoder turns this into a witness sub-AST + a pinned-value boundary
    constraint that contributes to H_synthesis. See §4.3.
    """
    inputs: tuple[Node, ...]   # IntLit/BoolLit only
    output: Node               # IntLit/BoolLit only


@dataclass(frozen=True)
class SynthesisProblem:
    """The contract: solve me."""
    sketch: Node                   # AST with HoleVar / TypeHole nodes
    target_type: Ty | None = None  # optional: pin the whole program's type
    examples: tuple[IOExample, ...] = ()
    name: str = ""                 # for reporting
    # Annealing knobs (defaults are good for §10's test set):
    N: int = 32
    chi_max: int = 32              # raised from A's 16 to accommodate hole superpositions
    n_samples: int = 64
    anneal_steps: int = 200
    anneal_dt: float = 0.05
```

Mandatory invariants on a well-formed `SynthesisProblem`:

- The sketch has **at least one hole** (`HoleVar` or `TypeHole`); otherwise `synthesize` returns an empty completion list (no synthesis needed).
- Each `HoleVar` carries a `candidates: list[str]` (may be empty meaning "any in-scope binder") and an optional `target_type: Ty` constraint.
- Each `TypeHole` carries a `candidates: list[Ty]` over the flat tag set (`TInt`, `TBool`, `TArrow(TInt,TInt)`, etc.). It may not list `TYPE_ARR_NESTED` candidates — synthesis stays within the flat type lattice (§2.2).
- Every `Var` not under a hole is a normal lexically-scoped reference; the sketch is well-scoped except at hole sites.

### 3.2 Output types

```python
@dataclass(frozen=True)
class Completion:
    ast: Node                       # the filled-in AST
    energy: float                   # <H_typing + H_eval + H_synthesis>
    energy_breakdown: dict[str, float]  # per-term: "typing", "eval", "examples",
                                         # "target_type", "size"
    diagnostics: dict               # from D.diagnose: residual constraint violations
    multiplicity: int               # how many of the K samples produced this AST


@dataclass(frozen=True)
class SynthesisResult:
    problem: SynthesisProblem
    completions: list[Completion]   # sorted ascending by energy
    n_unique: int
    n_samples_drawn: int
    n_samples_decoded_ok: int       # samples that survived alpha-eq deduplication
                                    # (others were parse-rejected — see §6)
    final_state_energy: float       # ⟨H⟩ on the relaxed state itself (pre-sampling)
    wall_time_seconds: float
    chi_observed_max: int           # max bond dim during evolution
    failure_mode: str | None        # None if a top-1 was found; else "no_valid_completion"
                                    # or "ambiguous_top1" or "imag_time_did_not_converge"
```

### 3.3 The synthesize API

```python
# src/qft_pcn/logic/synthesis/runner.py

def synthesize(problem: SynthesisProblem,
               rng: np.random.Generator | None = None,
               verbose: bool = False) -> SynthesisResult:
    """Run the synthesis pipeline.

    Pipeline:
        1. Validate the problem (§3.1 invariants).
        2. Build the encoder-extended state from `problem.sketch` with holes
           as superpositions (A's encode, with E's TypeHole extension).
        3. Compile H_typing (B) + H_eval (C) + H_synthesis (this sub-project).
        4. Imag-time evolve for `anneal_steps` steps with `anneal_dt`.
        5. Draw `n_samples` samples via A's `sample()`.
        6. Dedupe by alpha-equivalence; per unique completion compute
           ⟨H_total⟩ on its product-state encoding.
        7. Sort ascending; classify failure_mode; assemble SynthesisResult.

    Raises:
        SynthesisProblemError: malformed problem (§3.1 invariants violated).
        SynthesisRuntimeError: evolution diverged or sampling produced no
                               decode-able completions in K tries.
    """
```

---

## 4. The synthesis Hamiltonian

E does not invent typing/eval terms — those come from B and C. E composes them and adds **synthesis-specific** constraint terms.

```
H_total = w_T · H_typing(meta)                 # from sub-project B
        + w_E · H_eval(meta)                   # from sub-project C
        + w_X · H_examples(meta, problem.examples)
        + w_Y · H_target_type(meta, problem.target_type)
        + w_S · H_size(meta)
```

All `w_*` are positive scalars; defaults below.

### 4.1 Recommended weights (defaults)

| weight | default | rationale |
|---|---:|---|
| `w_T` | 4.0 | type errors are the strongest signal — drown out everything else |
| `w_E` | 2.0 | evaluation correctness matters but is downstream of typing |
| `w_X` | 3.0 | per-example penalty; the demo problems use 1–3 examples |
| `w_Y` | 2.0 | only active when `target_type` is set |
| `w_S` | 0.1 | gentle Occam preference; small literal penalty per non-PAD site |

These are **not** hyperparameters tuned for the test set; they are the starting point. The demo prints the per-term energy breakdown so the human can see which constraint pushed which way. If the test suite (§7) fails because some weight is wrong, that is a bug — but the fix lives here in this spec.

### 4.2 `H_examples(meta, examples)`

For each `IOExample(inputs, output)`, the sketch's top-level expression `e` must satisfy `e(inputs[0], inputs[1], ...) ⇓ output`. We encode this as follows:

1. The encoder produces the MPS for `problem.sketch` as usual.
2. For each example, we *augment* the lattice with a **witness application**: conceptually a fresh sub-tree `App(App(... App(e, in_0)..., in_{n-1}), in_n)` whose root is pinned to the value `output`. This witness lives on sites `[n_sketch, n_sketch + n_witness)` — sub-project E reserves a "witness region" at the end of the lattice.
3. The witness terms add to `H_synthesis`:
   - structural Hamiltonian terms forcing the witness region to encode the application chain (these are local, fixed by the example);
   - a **boundary pin** on the witness-root site: a one-site term `(1 - |output⟩⟨output|) on value` that costs +1 when the witness root's value register is not the expected output.

This is the same idea as A.§5.4's "boundary" support and §9.2's `boundary` map. The witness pin is a local projector; H_examples is a sum of such projectors over examples.

**Concrete construction.** Let `e_site` = site 0 (the sketch root). For each example `IOExample(inputs=(i_0, ..., i_{n-1}), output=o)`:

- Append witness sites starting at `wit_start`, encoding the AST `App(... App(App(REF(e_site), i_0), i_1) ..., i_{n-1})`. The witness sub-tree is fully product-state (no holes).
- Add a two-site Hamiltonian term coupling `e_site` to the leftmost witness `App.fn` site: a delta on `kind`/`type`/`bid` enforcing that the witness's reference subtree is alpha-equivalent to the sketch's root. (In practice, we use a **shared-state coupling**: the encoder writes the same MPS tensor at the witness's reference position as at `e_site`, and a two-site swap-test-style projector verifies equality.)
- Add a one-site boundary term on `wit_root` site (the outermost App's value register): `λ · (I - |o⟩⟨o|)` on `value`, with `λ = w_X`.

**Lattice layout.** For `n_sketch` sketch sites + `Σ n_witness_i` witness sites:

```
sites: [ sketch | witness_1 | witness_2 | ... | witness_K | PAD ]
       0       n_sketch     ...                 ...        N-1
```

The witness regions are *part of the same MPS*; the Hamiltonian operates over the whole lattice. The decoder ignores witness sites when reconstructing the synthesized AST (a `meta.witness_regions: list[range]` flag identifies them).

**Number of witnesses.** Each example contributes `1 + 2·n_inputs` sites (one outer App per input, plus the literal input nodes, plus the output is pinned not stored). For up to 3 examples with up to 2 inputs each, witness regions consume up to 15 sites. Hence the default `N = 32` allows ~17 sketch sites — sufficient for the §7 demo problems.

### 4.3 `H_target_type(meta, target_type)`

When `problem.target_type` is set, the program's root expression must have that type. This is a **one-site** term on site 0 (the root):

```
H_target_type = w_Y · (I - |target_tag⟩⟨target_tag|)  on the type register at site 0
```

If `target_type` is a nested arrow (would map to `TYPE_ARR_NESTED`), we recursively pin the corresponding `nested_type_index` entry rather than the on-lattice tag — the encoder's nested-type table is the source of truth (A.§4.2). For the §7 problems all target types fit the flat tag set.

### 4.4 `H_size(meta)`

A gentle Occam penalty: each non-PAD site contributes `w_S · ⟨I - P_PAD⟩` where `P_PAD` is the projector onto `KIND_PAD` on the `kind` register. Sum over all sketch sites (witnesses excluded — they are determined by the problem, not by the synthesis).

```
H_size = w_S · Σ_{i in sketch_range} (I - P_PAD^kind)_i
```

This makes simpler completions energetically preferred among completions that satisfy typing + evaluation + examples. **Without** `H_size`, completions like `f x` and `f (f x)` and `f (f (f x))` would tie at energy 0 on H_typing + H_eval + H_examples (all are well-typed `Int` programs satisfying the examples in many test cases); `H_size` breaks the tie in favor of the shorter one.

The size penalty is intentionally small (`w_S = 0.1`); it must not overwhelm a typing violation. Sanity check: a typing violation costs `w_T = 4`, a size of one extra node costs `0.1`; the inequality `4 > 0.1 · max_extra_nodes` (max_extra_nodes ≤ 30) gives `4 < 3`, which is **wrong**. So `w_S` is intentionally small; in fact the inequality is `w_T > w_S · max_program_difference` and `max_program_difference` is typically O(5) between competing completions. With `w_T = 4`, `w_S = 0.1`: `4 > 0.1 · 5 = 0.5` — comfortable.

### 4.5 Hamiltonian assembly

```python
# src/qft_pcn/logic/synthesis/hamiltonian.py

def compile_synthesis_hamiltonian(
    meta: EncodingMeta,
    problem: SynthesisProblem,
    weights: HamiltonianWeights | None = None,
) -> Hamiltonian:
    """Compose H_typing + H_eval + H_examples + H_target_type + H_size into a
    single Hamiltonian object suitable for trotter_step / evolve.

    Internally:
        - calls B.compile_typing_hamiltonian(meta) -> Hamiltonian_typing
        - calls C.compile_eval_hamiltonian(meta) -> Hamiltonian_eval
        - builds local/two-site terms for the three E-specific blocks
        - sums via Hamiltonian.__add__ (sub-project B added this; if missing,
          E adds it as a small extension to qft/hamiltonian.py)
    """
```

`Hamiltonian.__add__` may need to be added during E if B did not need it. The contract: term-wise sum where `(H_a + H_b).local_op(site) = H_a.local_op(site) + H_b.local_op(site)` and similarly for `bond_op`. **B's spec is responsible for adding it** if it is not already present; E checks for it during implementation and escalates if missing.

### 4.6 Per-term diagnostics

After evolution, the demo emits one row per Hamiltonian *block* (typing / eval / examples / target_type / size) with `⟨H_block⟩`. This is sub-project D's job; E calls `D.diagnose(state, meta, weighted_hamiltonians)` where `weighted_hamiltonians` is a dict mapping block name to the weighted `Hamiltonian` object.

---

## 5. Encoder extensions for synthesis

### 5.1 `TypeHole` AST node

Add to `src/qft_pcn/logic/ast.py`:

```python
@dataclass
class TypeHole(Ty):
    """A type-position hole: 'this type is unknown but must come from one of
    the candidates'.

    Encoded by sub-project E's encoder extension as an equal-amplitude
    superposition over the candidate tags on the `type` register at every
    site whose typing computation flows through this hole.
    """
    candidates: tuple[Ty, ...]   # each must be a non-arrow or a flat-tag arrow
    name: str = ""               # optional label for diagnostics
```

### 5.2 `HoleVar` extension

`HoleVar` already exists in A's `ast.py` minimally. Extend with:

```python
@dataclass
class HoleVar(Node):
    candidates: tuple[str, ...] = ()    # binder names; empty = "any in-scope"
    target_type: Ty | None = None       # optional: expected type
    name: str = ""                      # for diagnostics
```

If `candidates` is empty, the encoder resolves it to "every binder in lexical scope at this AST position" *at encode time*.

### 5.3 Encoder extension for `TypeHole`

The base encoder (A) computes the `type` register at each site by bottom-up walk (A.§5.3). When the type computation hits a `TypeHole`:

1. The encoder records the hole's site and candidate set.
2. At every site whose type was computed from a `TypeHole` path, instead of writing a definite `T_*` tag the encoder writes the equal-amplitude **superposition vector** on the `type` register: `|ψ_type⟩ = (1/√k) Σ_i |T_{c_i}⟩` for `k = len(candidates)`.
3. The bond going right from the type-hole site carries the type-hole's resolution as a *bond channel* on the `type` register, analogous to `bid`'s live-binder channels. Sites downstream that depend on this type use the channel to copy the resolved type into their own local state.
4. The hole site itself records `meta.type_holes: dict[int, TypeHoleHandle]` for E's downstream consumers.

This is structurally identical to the `bid` superposition path in A.§5.4 (the "Bell-pair / superposition extension" paragraph). The implementation reuses A's channel-construction machinery; the only addition is a separate per-bond channel set for `type`.

Hence E's encoder is *not* a rewrite of A's encoder; it is a few extra cases inside `_resolve.py`/`_tensors.py` to detect `TypeHole`/`HoleVar(candidates=…)` nodes and route them through the existing superposition branch.

### 5.4 `chi_max` requirements

A's default `chi_max=16` covers 6 live binders + no-info. E's holes add channels on the `type` register; with up to 4 candidate types per hole and up to 2 holes per problem, the additional channel count is bounded. **E uses `chi_max = 32` as default**, which gives 4× headroom over A's worst case. Programs that would exceed this raise `EncodingTooLarge` as in A.

### 5.5 Witness-region encoding

Witness sites are appended *after* the sketch sites; the encoder is called with an "augmented AST" that the runner constructs: `_witness_augmented_ast(sketch, examples)`. The augmentation:

```python
def _witness_augmented_ast(sketch: Node, examples: tuple[IOExample, ...]) -> Node:
    """Return an AST where the original sketch sits at the head and each
    example contributes a `(WitnessApp e in_0 in_1 ...)` subtree appended
    in pre-order after the sketch's last site.

    The witness applications share their `fn` subtree with the sketch via
    the encoder's binder-channel mechanism — the witness's outermost App.fn
    site is a `RefVar` (a new node kind reserved for this purpose) whose
    bid is wired to the sketch's site 0.
    """
```

`RefVar(target_site: int)` is a small encoder-internal node kind that does not show up in user-facing ASTs. It carries `KIND_VAR` with a bid wired to the sketch root (treated as a binder for this purpose). The encoder handles it transparently.

---

## 6. Sampling and ranking protocol

### 6.1 Draw

After imag-time evolution, call `sample(state, meta, n_samples=problem.n_samples, rng=rng)`. The default `n_samples = 64` was chosen empirically:

- For a unique-answer problem, 64 samples is overkill but ensures the top-1 dominates.
- For a problem with K plausible completions of similar energy, 64 samples gives ~64/K samples per completion — enough for stable energy estimates.

### 6.2 Dedupe

Each `DecodeResult` from `sample()` carries an `ast` (the sampler's output) and a `residual_norm`. We:

1. Drop samples with `residual_norm > 1e-3` (they hit a state outside the decoder's expected basis — likely an under-converged region).
2. Group surviving samples by `ast_alpha_eq` from A.§6.2. The group's representative is the canonical alpha-renamed form (`ast_canonical(node)`, a new helper in A's decoder that picks the smallest-name normalization).
3. The group's `multiplicity` is the sample count; the group's `ast` is the canonical form.

Samples that parse-reject (a `DecodeError` during the AST recovery) are counted but excluded. If *no* sample decodes, `failure_mode = "no_valid_completion"` and `completions = []`.

### 6.3 Rank

For each unique completion `c`:

1. Re-encode the *concrete* completion: `state_c, meta_c = encode(c.ast)` (no holes).
2. Re-compile the Hamiltonian on `meta_c` (the lattice may differ slightly in PAD positions; the witness region is reattached identically because `examples` are fixed).
3. Compute `⟨H_total⟩` and per-block energies via `Hamiltonian.expectation(state_c)`. This is sub-project D's job — D wraps `evolution.energy` for the block-wise breakdown.
4. Sort completions ascending by `energy`. Ties (within `1e-6`) break by `multiplicity` (descending), then by AST size (ascending).

### 6.4 Classify failure_mode

- `failure_mode = None`: the top-1 has energy < `tolerance_correct = 1e-3 · (w_T + w_E + w_X + w_Y)` — i.e. the dominant constraints are satisfied. (Size penalty alone is allowed to remain nonzero.)
- `failure_mode = "no_valid_completion"`: zero unique completions decoded.
- `failure_mode = "ambiguous_top1"`: top-1 and top-2 differ by less than `tolerance_ambiguous = 1e-3` *and* the top-1 is well-typed. The demo prints both and asks the human to disambiguate (in practice for §7 this won't happen).
- `failure_mode = "imag_time_did_not_converge"`: `final_state_energy > some_threshold` even after the anneal — the H couldn't be driven low. Sub-project D's diagnostics are still printed.

### 6.5 Annealing schedule

The default annealing schedule:

```
phase 1 (warmup):   50 steps, dt = 0.1, only H_typing + H_eval active
phase 2 (main):    100 steps, dt = 0.05, all blocks active at full weight
phase 3 (fine):     50 steps, dt = 0.01, all blocks active
```

The schedule is implemented in `synthesize` as three sequential `evolve()` calls with different Hamiltonians (phase 1 builds a partial H without H_examples/H_target_type/H_size; phases 2 and 3 use the full H). The rationale: jumping straight to the full H with all weights can lock the state into a bad typing-violating configuration before example constraints get a chance to bend it.

The `problem.anneal_steps` and `problem.anneal_dt` knobs override the *main* phase totals (warmup and fine scale proportionally, with floors of 25 and 25). For the §7 problems the defaults are used unchanged.

Imag-time evolution normalizes every step (passing `normalize_every=1` to `evolve`). Bond dimension is capped at `problem.chi_max`.

---

## 7. The §10 demo problems (acceptance test set)

The eight problems in `demo_stlc_synthesis.py`, of escalating difficulty:

### P1. Identity completion (warmup)

```
sketch:    \x:Int. ?HOLE
target:    \x:Int. x
expected top-1 energy: ~0 (H_size dominates the residual)
```

Hole candidates: `["x"]` (the only in-scope binder). Trivially correct, tests the encoder's superposition branch.

### P2. Constant-vs-identity disambiguation by type

```
sketch:    \x:Int. ?HOLE
target_type: Int -> Int
examples:  IOExample((IntLit(3),), IntLit(3))
expected top-1: \x:Int. x        (energy ≈ 0)
expected top-2: \x:Int. 0..7     (constants, energy > 0 because example fails)
```

Here `HoleVar` has *no* candidates listed → encoder fills in all in-scope variables and integer literals. The example pins the output.

### P3. Use both arguments

```
sketch:    \f:Int->Int. \x:Int. ?HOLE
expected top-1: f x                  (energy ≈ 0)
expected top-2: f (f x)              (slightly higher: H_size penalty)
expected top-3: x                    (higher: doesn't use f → no example, but H_size lower
                                       — handled by adding an example so f matters)
examples:  IOExample((Lam("x",Int,Bin("+",Var("x"),IntLit(1))), IntLit(2)), IntLit(3))
           # i.e. give it (succ, 2) and expect 3 → forces f x
expected top-1 after examples: f x
```

This is the canonical example from the architecture doc (§10.7 intro example).

### P4. Conditional synthesis

```
sketch:    \x:Int. if (x < ?HOLE_INT) then x else (x + ?HOLE_INT)
examples:  IOExample((IntLit(2),), IntLit(2))     # x < 5 → x, returns 2
           IOExample((IntLit(7),), IntLit(8))     # x ≥ 5, returns x+1
expected top-1: \x:Int. if (x < 5) then x else (x + 1)    (energy ≈ 0)
```

Two type-anchored int-literal holes; the examples disambiguate. Tests `H_examples` against multiple inputs.

### P5. Type-hole synthesis

```
sketch:    \x:?T. x
target_type: Bool -> Bool
expected top-1: \x:Bool. x   (TypeHole resolves to TBool)
```

Pure type-level synthesis — exercises §5.3's `TypeHole` encoder extension.

### P6. Curried completion

```
sketch:    \f:Int->Int. \g:Int->Int. \x:Int. ?HOLE
examples:  IOExample((..., ..., IntLit(2)), IntLit(5))
           # with concrete f = (\n. n+1), g = (\n. n*2), x = 2 → expected output 5
           # which is g(f(x)) = (2+1)*2 = 6? actually wrong; let me pick
           # f(g(x)) = f(4) = 5 ✓
expected top-1: f (g x)         (energy ≈ 0)
expected top-2: g (f x)         (would give 6 ≠ 5, higher energy)
```

Tests two-level function composition synthesis. Multiple plausible completions, examples select.

### P7. Boolean synthesis with mixed types

```
sketch:    \x:Int. \y:Int. ?HOLE
target_type: Int -> Int -> Bool
examples:  IOExample((IntLit(2), IntLit(3)), BoolLit(True))     # 2 < 3
           IOExample((IntLit(5), IntLit(3)), BoolLit(False))    # 5 < 3 → false
expected top-1: x < y    (energy ≈ 0)
expected alt:   x == y   (would give false, false — fails example 1)
```

### P8. Negative case (intentionally unsolvable as stated)

```
sketch:    \x:Int. ?HOLE
target_type: Int -> Bool        # impossible with this sketch's hole candidates
examples:  IOExample((IntLit(2),), BoolLit(True))
                                # but the hole candidates are limited to:
                                # candidates=["x"]  (only x in scope, x:Int, can't make Bool)
expected: SynthesisResult with failure_mode = "no_valid_completion" OR
          top-1 with energy > tolerance_correct, and D's diagnostics report
          residual H_typing > 0 with the offending term identified.
```

This problem is **deliberately unsolvable** under the given hole candidates. P8 tests §1.6 (honest reporting): the demo must not lie about success.

### 7.1 Difficulty / sites budget

| problem | sketch sites | witnesses | total | within N=32? |
|---:|---:|---:|---:|---:|
| P1 |  2 |  0 |  2 | ✓ |
| P2 |  2 |  3 |  5 | ✓ |
| P3 |  4 |  6 | 10 | ✓ |
| P4 |  7 |  8 | 15 | ✓ |
| P5 |  2 |  0 |  2 | ✓ |
| P6 |  5 |  6 | 11 | ✓ |
| P7 |  5 |  8 | 13 | ✓ |
| P8 |  2 |  3 |  5 | ✓ |

All fit `N = 32` with comfortable PAD margins.

---

## 8. File layout

```
src/qft_pcn/logic/
├── ast.py                          # MODIFIED: TypeHole, HoleVar.candidates upgrade
├── synthesis/
│   ├── __init__.py                 # public re-exports
│   ├── problem.py                  # SynthesisProblem, IOExample, Completion,
│   │                               #   SynthesisResult, HamiltonianWeights
│   ├── encode_ext.py               # encoder extensions: TypeHole superposition,
│   │                               #   witness-augmented AST builder
│   ├── hamiltonian.py              # compile_synthesis_hamiltonian()
│   │                               #   - H_examples, H_target_type, H_size builders
│   │                               #   - composition with B's H_typing, C's H_eval
│   ├── runner.py                   # synthesize() — the main entry point
│   ├── ranking.py                  # dedupe, rerank, classify failure_mode
│   └── errors.py                   # SynthesisProblemError, SynthesisRuntimeError
├── demo_stlc_synthesis.py          # runs P1–P8, prints structured output

src/qft_pcn/tests/
├── test_synthesis_problem.py       # problem validation, IOExample, invariants
├── test_synthesis_encoder_ext.py   # TypeHole encoder, witness-AST builder
├── test_synthesis_hamiltonian.py   # H_examples, H_target_type, H_size unit tests
├── test_synthesis_runner.py        # synthesize() integration tests on P1–P8
└── test_synthesis_demo.py          # smoke test: the demo script runs without error
```

Public exports from `src/qft_pcn/logic/synthesis/__init__.py`:

```python
from .problem import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
)
from .runner import synthesize
from .errors import SynthesisProblemError, SynthesisRuntimeError
```

Re-exported from `src/qft_pcn/logic/__init__.py`:

```python
from .synthesis import (
    synthesize, SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights, SynthesisProblemError, SynthesisRuntimeError,
)
from .ast import TypeHole  # newly added
```

No changes to existing files **except**:

- `src/qft_pcn/logic/ast.py`: add `TypeHole`, upgrade `HoleVar` per §5.1/5.2.
- `src/qft_pcn/logic/_tensors.py` / `src/qft_pcn/logic/_resolve.py` / `src/qft_pcn/logic/encoder.py`: extend with the type-hole superposition branch (analogous to existing bid-hole branch).
- `src/qft_pcn/logic/encoding.py`: add `type_holes: dict[int, TypeHoleHandle]` to `EncodingMeta`, add `TypeHoleHandle` dataclass.
- `src/qft_pcn/logic/decoder.py`: add `ast_canonical(node)` helper for dedup (alpha-normalized representative).
- `src/qft_pcn/qft/hamiltonian.py`: add `Hamiltonian.__add__` and `Hamiltonian.expectation` if not already present (B may have added them).

---

## 9. Error model

`src/qft_pcn/logic/synthesis/errors.py`:

```python
from src.qft_pcn.logic.encoding import EncodingError

class SynthesisError(Exception):
    """Base class for synthesis-runner errors."""


class SynthesisProblemError(SynthesisError):
    """The SynthesisProblem violates §3.1 invariants (no holes, malformed
    examples, etc.)."""


class SynthesisRuntimeError(SynthesisError):
    """Evolution diverged or produced no decodable samples — distinct from
    a problem that 'just has no good completion' which is reported as
    failure_mode in SynthesisResult, not raised."""
```

The synthesis runner does **not** raise on a problem with no good completion — that is reported via `SynthesisResult.failure_mode`. Raises are reserved for runtime/infrastructure failures (malformed input, NaN energies, evolution divergence to infinity).

`SynthesisRuntimeError` examples:
- imag-time evolution produced NaN energies (likely a bug in H assembly)
- all 64 samples failed to decode (zero useful output — should never happen with a well-formed problem)
- bond dimension exceeded `chi_max` and truncation produced an unphysical state (norm collapse < 0.5)

---

## 10. Acceptance criteria

The sub-project is complete when **all** of the following hold:

1. All §7 problems (P1–P8) run via `synthesize()` without raising.
2. **P1, P2, P3, P4, P5, P6, P7 (7 of 8) produce a top-1 completion that is alpha-equivalent to the expected answer.** This is the "majority of cases" success criterion from the user's brief. (Note: P8 is the *intentionally unsolvable* problem; success on P8 means correctly classifying as `failure_mode != None`.)
3. For each of P1–P7, the second-ranked completion has strictly higher energy than the top-1, with a gap of at least `0.5 * w_T = 2.0` (i.e. the wrong completion incurs at least half a typing-violation worth of penalty). If this fails on any problem, the demo prints the energy distribution and a `WARNING: P_i has small energy gap` line.
4. P8 produces `failure_mode != None` and the demo prints the residual H_typing breakdown from D.
5. The demo script `demo_stlc_synthesis.py` runs end-to-end in under 5 minutes on a developer laptop (commodity CPU, no GPU); each problem completes in under 60 seconds.
6. All unit tests in `tests/test_synthesis_*` pass.
7. All previously passing tests in `src/qft_pcn/tests/` still pass.
8. The `TypeHole` encoder extension is verified: a small synthetic test puts a `TypeHole` in a 2-node sketch and checks that the encoded state has nonzero entanglement entropy on the `type` register bond (analogous to A.§7.4's superposition probe).
9. Per-block energy diagnostics are emitted by D for every problem; the breakdown sums (within 1e-6) to `final_state_energy`.
10. The demo's printed output for each problem includes: problem name, expected top-1, actual top-1, energy, energy_breakdown, multiplicity, n_unique, n_samples, failure_mode, wall_time.

No claim of completion is acceptable without these tests running green in a fresh shell. (See `superpowers:verification-before-completion`.)

---

## 11. Open questions

**None that block implementation.** Two genuine open design points are noted for transparency; both have a documented default that is locked unless the implementer finds it wrong while testing:

1. **Witness-region encoding via `RefVar`** vs. classical copying of the sketch sub-tree into the witness region. **Default: `RefVar` channel approach (§5.5).** The classical copy doubles lattice usage and breaks the architectural claim that the witness *shares* its `e` subtree with the sketch. If `RefVar` proves impractical during implementation, escalate; do not silently swap to classical copy.

2. **Annealing schedule constants** (50/100/50 step counts; dt = 0.1/0.05/0.01). **Default: the schedule in §6.5.** If P3 or P4 fails to converge with the default schedule, the implementer may double the main phase before escalating; further tuning is a §11 amendment.

If you, while implementing, find a third real ambiguity that this document does not resolve — **stop and ask the human**. Do not paper over it.

---

## 12. Contract for sub-project G (LLM bridge)

G consumes E's synthesis API as a black box. G's responsibilities:

- Parse a natural-language request like "give me a function that takes an Int and returns an Int twice as large, with hole at the function body" into a `SynthesisProblem`.
- Call `synthesize(problem)`.
- Verbalize the `SynthesisResult` back to the user.

G does **not** touch E's internals (encoder extensions, Hamiltonian assembly, evolution loop). The contract is:

```python
# Public API E exports:
synthesize(problem: SynthesisProblem, rng=None, verbose=False) -> SynthesisResult

# Public types E exports:
SynthesisProblem, IOExample, Completion, SynthesisResult, HamiltonianWeights,
SynthesisProblemError, SynthesisRuntimeError

# Public AST types E adds for sketch construction:
TypeHole (in ast.py), HoleVar (already in ast.py)
```

G must accept that E may *fail to find a completion* (P8 case) and verbalize the diagnostics. G must not interpret a `failure_mode != None` as "try a different sampler" or "increase n_samples"; that's E's job to tune.

If during G's implementation a real API gap is found (E exposes too much or too little), the resolution is to amend this spec — not to bypass it.

---

## 13. Glossary (local)

- **Sketch** — an STLC AST with one or more holes; the input to a synthesis problem.
- **Hole** — a `HoleVar` (variable-position) or `TypeHole` (type-position) marker in a sketch.
- **Candidates** — the finite set of admissible fillings for a hole, attached to the hole node.
- **Completion** — a fully concrete AST produced by filling every hole with one of its candidates.
- **Witness region** — sites in the MPS lattice that encode the example-application sub-trees used by `H_examples`.
- **RefVar** — encoder-internal node kind referencing a previously-encoded sub-tree; used in witness regions to point at the sketch root without duplicating the encoding.
- **Annealing schedule** — the three-phase imag-time evolution: warmup (typing+eval only), main (all blocks), fine (lower dt).
- **Failure mode** — a string label on `SynthesisResult` describing why no clean top-1 was identified, or `None` for success.
- **Block** — a logical group of Hamiltonian terms (typing, eval, examples, target_type, size) reported separately in diagnostics.

---

## 14. Where this fits in the larger arc

This is the publishable milestone. After E lands:

- F (MERA) adds recursion to the language, lifting the §2.2 restriction.
- G (LLM bridge) wraps E in natural-language IO.
- The paper writes itself off the §7 problems' printed output + D's diagnostics + a comparison study against a baseline (e.g. a typed-Holes synthesizer like Hoogle+ or `agda2hs`).

Failing to realize §1.1–§1.6 here will make the paper unwriteable — the architectural claim collapses to "we have a small constraint solver" and the QFT/PCN substrate is irrelevant. That is the practical reason §1.1–§1.6 are non-negotiable.
