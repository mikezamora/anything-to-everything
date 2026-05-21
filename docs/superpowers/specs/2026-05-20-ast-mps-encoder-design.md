# Spec: AST ↔ MPS Encoder/Decoder for the QPCN Logic Layer

**Document type**: Implementation specification (sub-project A of the §10 roadmap).
**Date**: 2026-05-20.
**Author**: Brainstorming session, this branch.
**Status**: Approved design, ready for implementation plan.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Architecture doc**: `QFT_PCN_ARCHITECTURE.md` (root). This spec realizes §10.1.
**Acceptance owner**: human review of the round-trip test suite passing.

---

## 0. How to read this spec

This document is the contract for one sub-project. It exists because the §10 roadmap is too large for a single design cycle, so it was decomposed into seven sub-projects (A–G); this is **sub-project A: AST ↔ MPS encoder/decoder**. The others (B = typing-rule Hamiltonian compiler, C = evaluation Hamiltonian, D = constraint debugger, E = STLC synthesis demo, F = MERA, G = LLM bridge) will get their own specs.

Every section below is part of the contract. If something is missing here that you need to decide while implementing, **stop and ask**. Do not fill in by guessing — see §1.

---

## 1. Driving principles (non-negotiable)

The whole reason this project exists is that the team did not stop digging when the connection between fields (quantum many-body physics, predictive coding, type theory, programming-language semantics) looked too complex to be real. The architecture document's central thesis is that **variable binding in code is literally the same mathematical object as entanglement in quantum mechanics** (§8.1 of `QFT_PCN_ARCHITECTURE.md`), and the whole technical agenda hinges on realizing that correspondence operationally — not gesturing at it.

A subagent reading this spec will be tempted to take shortcuts that destroy the correspondence. The following principles are non-negotiable. **None of them may be traded away for implementation simplicity.** If you find yourself tempted to violate one, stop and ask the human.

### 1.1 Binder ↔ use coupling is genuine entanglement, not a copied field

A `Var(x)` whose lexical binder is some `Lam` site does **not** know which binder it refers to via a separately set `binder_id` integer field on its own site. The reference is realized by the joint quantum state on the `bid` register: the (Lam, Var) pair occupies an entangled bipartite state on `H_bid ⊗ H_bid`, and the MPS bond running between them carries this entanglement. Measurement of the Var's `bid` register yields the binder id; the *same* measurement on the Lam yields the *same* id; in superposition (when the AST has holes) the joint state is a Bell-like sum over candidate binders.

The "easy shortcut" — pre-computing the binder id during pre-order walk and writing it as a definite local state at the Var site — looks indistinguishable from the principled scheme for a concrete program. **It is not the same scheme.** It cannot represent unresolved-reference superpositions, which §10.7 (the publishable milestone) requires. It also makes the entanglement-entropy test (§7.4 of this spec) trivially fail, which is the test we will use to *verify* the principled scheme was actually implemented.

**Rule**: the encoder constructs MPS bonds whose dimension on the `bid` register grows with the number of live binders crossing that bond. Bond tensors are not classical correlations; they are full quantum-correlated tensors with the structure described in §5.4.

### 1.2 PAD is a vacuum state, not a sentinel

Sites beyond the last AST node hold the `PAD` state on every field species. PAD is the lowest-energy state, ignored by every Hamiltonian term we will build in sub-projects B/C, contributing zero to any expectation value we will care about. The reason this matters: the Hamiltonian is a *structural* object on a fixed-shape lattice (this matches the QFT mental model — the Lagrangian is a property of spacetime, not of the field configuration). One Hamiltonian operates on every AST of size ≤ N.

The "easy shortcut" — sizing the MPS per-program — is rejected. It breaks §10.7's hole-completion semantics (a hole changes the AST size and hence the Hamiltonian), and it forces every downstream piece to be recompiled per program.

### 1.3 Reuse the existing QFT machinery; do not reinvent

`src/qft_pcn/qft/fock.py`, `mps.py`, `hamiltonian.py`, `evolution.py` are already built and tested (39 passing tests). The encoder must consume them as-is. New code goes only under `src/qft_pcn/logic/`. If you discover an actual bug or missing feature in the QFT core, fix it in `qft/`; otherwise, *use what's there*.

The "easy shortcut" — bypassing the multi-species `FieldSpecies`/`HamiltonianConfig` plumbing and inlining a custom local-Hilbert-space construction — is rejected. The multi-species machinery is the *reason* we can have four field registers per site; opting out of it forces every downstream piece to re-derive the basis ordering, embedding, and operator construction.

### 1.4 The encoder is total on well-scoped ASTs; it is not best-effort

A well-scoped AST that fits within N and the binder-cutoff produces an MPS whose `decode()` returns the *same* AST. Round-trip is a structural equality, not "structurally similar". When the AST exceeds limits the encoder raises (§9), it never silently truncates or drops nodes.

The "easy shortcut" — letting the encoder fall back to some lossy representation when scope nesting is high — is rejected. Either you encoded the program faithfully or you raised.

### 1.5 The decoder is honest about uncertainty

For a product (concrete) input, `decode()` is deterministic and exact. For an entangled (e.g. hole-bearing) input, `sample()` produces one completion per call, drawn from the true MPS measurement distribution — not from the argmax of marginals.

The "easy shortcut" — sampling each site independently using its local reduced density matrix — is rejected. It ignores correlations across sites, which is the entire point of an MPS. The correct sweep collapses each site sequentially using the conditioned MPS, the standard left-to-right MPS sampling algorithm.

### 1.6 Cross-checks are mandatory

The encoder's analytic MPS construction (§5.4) and gate-based construction (§5.5) compute the same state two different ways. The test suite verifies they agree (§7.3). If they don't agree, that is an unresolved bug, not a numerical curiosity; do not lower the tolerance to make the test pass.

---

## 2. Scope

### 2.1 In scope (this sub-project)

- AST data classes for an STLC surface extended with `Int`, `Bool`, `if/then/else`, and the operators `+ - * < ==`.
- A canonical pre-order encoder `encode(ast) -> MPS` plus encoding metadata.
- A deterministic decoder `decode(mps) -> AST` and a stochastic `sample(mps, n) -> [AST]`.
- A parser + pretty-printer for human-readable lambda terms (test affordance only).
- A test suite covering five programs of escalating complexity, plus structural property tests (entanglement entropy, alpha-renaming invariance, norm preservation, gate-based cross-check).

### 2.2 Out of scope (deferred to other sub-projects)

- Typing rules and a Hamiltonian compiler (sub-project B / §10.2).
- Evaluation / beta-reduction Hamiltonian (sub-project C / §10.3).
- Constraint debugger (sub-project D / §10.6).
- The STLC synthesis demo (sub-project E / §10.7).
- MERA upgrade (sub-project F / §10.4).
- LLM bridge / DSL runtime (sub-project G / §10.5).
- Recursion / `let rec` / `fix` (forwarded to sub-project F when MERA arrives).
- Polymorphism, type inference, type holes (sub-project E may introduce holes; this sub-project supports them at the encoding level via §5.4's superposition mechanism, but does not introduce any typing logic).

### 2.3 Will not do, even if asked later

- Drop the multi-species local Hilbert space in favor of a single integer-token field.
- Use a single classical `binder_id` integer at the Var site as a stand-in for entanglement.
- Re-implement MPS / Fock primitives.

---

## 3. Surface language

The encoder/decoder must round-trip every well-scoped, well-typed (per STLC) AST in the following grammar. Type annotations on lambda parameters are required (we are *not* doing inference here).

```
e ::= x                            -- variable (identifier)
    | n                            -- int literal (n ∈ ℤ, range constrained, see §4.5)
    | b                            -- bool literal (true | false)
    | λx:τ. e                      -- lambda abstraction
    | e e                          -- application
    | if e then e else e           -- conditional
    | e ⊕ e                        -- binary op, ⊕ ∈ {+, -, *, <, ==}

τ ::= Int | Bool | τ → τ           -- types
```

Identifiers: `[a-zA-Z_][a-zA-Z0-9_]*`. Shadowing is allowed; the encoder resolves each `Var(name)` to its nearest enclosing binder by ordinary lexical scope.

Out-of-grammar elements (let bindings, tuples, recursion, type variables, holes) are explicitly not supported by sub-project A. They are not parse errors — they are not in the grammar.

---

## 4. Field species and cutoffs

The local Hilbert space per site is the tensor product of four registers ("field species" in the existing `FieldSpecies` sense from `src/qft_pcn/qft/hamiltonian.py`). The basis ordering inside each register follows the existing `embed_op` convention (leftmost species index changes slowest).

### 4.1 `kind` — AST node kind

| dim | basis name | meaning |
|---:|---|---|
| 0 | `PAD`  | empty / vacuum site |
| 1 | `VAR`  | variable reference |
| 2 | `LAM`  | lambda abstraction (a binder) |
| 3 | `APP`  | function application |
| 4 | `INT`  | int literal |
| 5 | `BOOL` | bool literal |
| 6 | `IF`   | if-then-else |
| 7 | `BIN`  | binary operator (op identity carried in `value`) |

`cutoff = 8`. PAD = `|0⟩` so that "vacuum" (`MPS.vacuum`) is the all-PAD lattice on this register.

### 4.2 `type` — typing tag

Types are recursive, but we live in a finite-cutoff Hilbert space. We use a **flat enumeration of common types plus an overflow slot**:

| dim | basis name | meaning |
|---:|---|---|
| 0 | `T_NONE`        | no type (PAD sites; non-typeable structural sites) |
| 1 | `T_INT`         | `Int` |
| 2 | `T_BOOL`        | `Bool` |
| 3 | `T_ARR_II`      | `Int → Int` |
| 4 | `T_ARR_IB`      | `Int → Bool` |
| 5 | `T_ARR_BI`      | `Bool → Int` |
| 6 | `T_ARR_BB`      | `Bool → Bool` |
| 7 | `T_ARR_NESTED`  | any higher-order arrow type (`Int → (Int → Int)`, `(Int → Int) → Int`, etc.) |

`cutoff = 8`.

For `T_ARR_NESTED` the full type tree is stored in `EncodingMeta.nested_type_index[site] -> Ty` (out-of-band table) and recovered by the decoder via that table. This is a known compromise: a finite-cutoff register cannot represent the unbounded type lattice. The compromise is local to this sub-project. Sub-project B may later introduce additional arrow-shape tags (`T_ARR_III`, etc.) if real programs hit `T_ARR_NESTED` often; for now, type bookkeeping is the encoder's responsibility, not the Hamiltonian's.

**Note for sub-project B**: typing-rule Hamiltonian terms in B will operate only on the *flat* tags `T_NONE … T_ARR_BB`. Encountering `T_ARR_NESTED` at a constraint site will be a B-time decision (likely: treat as "type unknown, allow anything"). That decision belongs in B's spec.

### 4.3 `bid` — binder identifier

| dim | basis name | meaning |
|---:|---|---|
| 0   | `B_NONE` | no binder reference (non-Var, non-Lam) |
| 1–7 | `B_0`, `B_1`, …, `B_6` | the k-th nested binder in lexical scope |

`cutoff = 8`. A program with more than 7 simultaneously live binders at any AST position raises `TooManyBinders` (§9).

`B_0` is the innermost binder at any position, `B_1` next, etc. This is a **lexical-depth indexing** scheme (essentially De Bruijn indices, but with the index being the position from the innermost enclosing binder). The encoder allocates `bid` per-position, not globally:

> When the encoder is *inside* a binder, that binder is `B_0`; binders enclosing it are `B_1`, `B_2`, …. When the encoder leaves a binder's scope, the binder's `bid` is released for reuse at the next sibling subtree.

This scheme guarantees alpha-equivalent terms (e.g. `λx.x` and `λy.y`) produce **identical** MPS states up to a local unitary on the `bid` register. See §7.5 for the test.

### 4.4 `value` — literal payload + operator id for BIN

This register is overloaded by `kind`:

| `kind` | meaning of `value` basis |
|---|---|
| `PAD`        | `V_PAD` (= 0) only |
| `VAR`, `LAM`, `APP`, `IF` | `V_NONE` (= 0) only |
| `INT`        | the integer literal mapped to `[0, 15]` via the offset in §4.5 |
| `BOOL`       | `V_FALSE` (= 0), `V_TRUE` (= 1) |
| `BIN`        | `V_PLUS` (= 2), `V_MINUS` (= 3), `V_TIMES` (= 4), `V_LT` (= 5), `V_EQ` (= 6) |

`cutoff = 16`. Indices 7..15 are reserved (unused in sub-project A, available for sub-project C to encode intermediate evaluation results).

### 4.5 Integer literal range

Integers are encoded in the range `[-7, 8]` via offset `value = lit + 7` (so `-7 → 0`, …, `8 → 15`). Out-of-range literals raise `IntLiteralOutOfRange` (§9). This is a real limitation; sub-project E may need to widen `value`'s cutoff to support larger demo programs. Until then, the demos in this spec use literals in `[0, 5]`.

### 4.6 Local dimension

Per site: `d_local = 8 × 8 × 8 × 16 = 8192`.

Note: this is large but acceptable — the encoder constructs explicit MPS tensors of shape `(χ_left, 8192, χ_right)`. With `χ_max = 16` (§5.7), a per-site tensor is at most `16 × 8192 × 16 ≈ 2.1 M complex entries ≈ 33 MB`. With `N = 32` sites, the whole MPS is ≈ 1 GB worst case. **In practice we never densify the full local register**: encoder writes are sparse, and we use a sparse construction path (§5.6). Tests verify this with a benchmark on `N = 32`, `chi_max = 16` that completes in under 5 s on a developer laptop.

### 4.7 `FieldSpecies` mapping

For consumption by sub-projects B/C/E that will build Hamiltonians via `HamiltonianConfig`:

```python
SPECIES = [
    FieldSpecies(name="kind",  cutoff=8,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="type",  cutoff=8,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="bid",   cutoff=8,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="value", cutoff=16, bare_mass=0.0, kinetic=0.0),
]
```

Sub-project A does *not* construct any Hamiltonian — it only ensures the species list is correct so that B/C can pick it up.

---

## 5. Encoder

### 5.1 API

```python
# src/qft_pcn/logic/encoder.py

@dataclass
class EncodingMeta:
    N: int                              # lattice length
    chi_max: int                        # encoder bond cap
    field_dims: dict[str, int]          # {"kind": 8, "type": 8, "bid": 8, "value": 16}
    species: list[FieldSpecies]         # the SPECIES list above
    # Side tables for information that doesn't fit in the cutoff lattice:
    nested_type_index: dict[int, Ty]    # site -> full Ty when type tag is T_ARR_NESTED
    site_to_ast_path: dict[int, tuple[int, ...]]  # site -> path in original AST (for debugging only)
    # Live-binder bookkeeping per bond, exposed so sub-projects B/C/D and tests
    # can reason about the channel structure:
    live_binders_per_bond: list[list[BinderHandle]]
    # BinderHandle is a small dataclass identifying a binder uniquely:
    #   @dataclass(frozen=True)
    #   class BinderHandle:
    #       lam_site: int        # site of the binder
    #       depth_at_lam: int    # lexical depth when the Lam was introduced
    # Channel ordering at each bond matches the basis ordering of the bond
    # tensor on the bid register (see §5.4).


def encode(ast: Node, N: int = 32, chi_max: int = 16) -> tuple[MPS, EncodingMeta]:
    """Encode an AST into an MPS of length N with bond cap chi_max.

    Returns (state, meta). The state is normalized (||state|| = 1).
    Pre-order walks the AST; sites beyond the AST are PAD.

    Raises:
        EncodingTooLarge: AST has more nodes than N can hold.
        TooManyBinders: scope nesting at some point exceeds bid cutoff - 1.
        IntLiteralOutOfRange: an IntLit value is outside [-7, 8].
        IllScopedVar: a Var reference is not in lexical scope.
        UnsupportedNode: AST contains a node type not in §3's grammar.
    """
```

### 5.2 Pre-order walk and arity

The walk visits the AST in canonical order, writing one site per node:

| node | site write | child order in next sites |
|---|---|---|
| `Var(x)` | `(VAR, type_of_var, bid_of_binder, V_NONE)` | (leaf) |
| `Lam(x, ty_x, body)` | `(LAM, type_of_lam, bid_for_x, V_NONE)` | `[body]` |
| `App(fn, arg)` | `(APP, type_of_app, B_NONE, V_NONE)` | `[fn, arg]` |
| `IntLit(n)` | `(INT, T_INT, B_NONE, n + 7)` | (leaf) |
| `BoolLit(b)` | `(BOOL, T_BOOL, B_NONE, V_TRUE / V_FALSE)` | (leaf) |
| `If(c, t, e)` | `(IF, type_of_if, B_NONE, V_NONE)` | `[c, t, e]` |
| `Bin(op, lhs, rhs)` | `(BIN, type_of_bin, B_NONE, op_code)` | `[lhs, rhs]` |

The arity is a direct function of `kind`: `PAD/VAR/INT/BOOL → 0`, `LAM → 1`, `APP/BIN → 2`, `IF → 3`. Knowing the kind, the decoder can recover children from subsequent sites by recursion (§6.2).

### 5.3 Type computation

Each site's `type` register holds the type of the *expression rooted at that site*. The encoder computes types bottom-up:

- `IntLit → T_INT`
- `BoolLit → T_BOOL`
- `Var(x) → type of x's binder's param_ty` (looked up via lexical scope)
- `Lam(x:τ, body) → T_ARR(τ, type(body))` mapped to a flat tag or `T_ARR_NESTED`
- `App(f, a) → range of f's type` (assumes well-typed input)
- `If(c, t, e) → type(t)` (also = type(e) if well-typed; encoder does not check)
- `Bin(op, l, r) → T_INT` for `+ - *`, `T_BOOL` for `< ==`

Sub-project A does *not* perform type checking. The encoder trusts its input is well-typed (this is consistent with §3's grammar specification — type checking is sub-project B's job). If type computation hits an arrow type that doesn't fit `T_ARR_II..T_ARR_BB`, that site's type tag is `T_ARR_NESTED` and the full `Ty` is stored in `meta.nested_type_index[site]`.

### 5.4 The principled binder ↔ use coupling (analytic MPS construction)

This is the architectural soul of the project. It is **not** to be replaced with a classical-correlation shortcut.

**Setup.** Let `K = {(lam_site, var_sites)}` be the set of binder-to-uses mappings recovered from lexical scope. For a single (lam, var) pair, define the bipartite state on `H_bid ⊗ H_bid`:

```
|Φ⟩_{lam,var} = |b⟩_lam ⊗ |b⟩_var
```

where `b` is the binder's id (always `B_0` from the *binder's own* perspective — see §4.3). The MPS bond between the two sites must support this correlation.

**General case (multiple live binders crossing a bond).** Consider a bond between sites `i` and `i+1`. Let `L_i ⊂ K` be the set of *live binders* at that bond: binders that have been declared at some `lam_site ≤ i` and have at least one variable use at some `var_site ≥ i+1`. Then the bond's `bid`-register dimension must be at least `2^{|L_i|}` if we want to support superpositions over arbitrary subsets — but we don't. We only need to support **definite assignments** to each live binder during encoding plus the *option* to lift to superposition.

The encoder constructs the MPS analytically using the following bond structure on the `bid` register only (other registers are product):

> Bond `i` between sites `i` and `i+1` carries a "live-binder register" of dimension `|L_i| + 1`. The extra `+1` is the "no-information" channel (zero or padding). The bond basis is `{|no_info⟩, |L_i[0]⟩, |L_i[1]⟩, …}` — one basis state per live binder plus a placeholder.

The per-site tensor on the `bid` register is constructed as follows:

1. **Initialization.** At the binder's own site `lam_site`, the local `bid` value is `B_0` (relative to that binder's own scope). The bond going right from `lam_site` carries the binder forward in the `|L⟩` channel corresponding to this binder.
2. **Pass-through sites** (sites in between binder and uses where the binder is in scope but neither bound nor used): the `bid` register is `B_NONE` locally, and the left-bond / right-bond carry the live-binder channels through as the identity in the live-binder subspace.
3. **Use sites** (`Var(x)` sites): the local `bid` value is read off the live-binder channel for `x`'s binder. The bond tensor "couples" the local `bid` register to the channel.

**Concretely** (this is the construction every test must verify), at any site `i`:

```
The bid-only sub-tensor T_bid[i] has shape (|L_{i-1}| + 1, 8, |L_i| + 1).
```

- For a non-binder, non-Var site whose only role is to pass binders through:
  `T_bid[i][ℓ_in, B_NONE, ℓ_out] = δ_{ℓ_in, ℓ_out}` if `ℓ_in` corresponds to a live binder in `L_i`, else 0. All other local-register entries are 0.

- For a `Lam` site introducing binder `b` (which becomes channel `c` on the right bond, where `c` is the index `|L_i|` assigned to `b` in the live-binder list — see ordering below):
  `T_bid[lam_site][ℓ_in, B_0, ℓ_out] = δ_{ℓ_in, ℓ_out}` for pre-existing channels, and `T_bid[lam_site][no_info_in, B_0, c] = 1` for the new binder's channel. (The lam's *own* `bid` value is `B_0` because from the lam's perspective the binder it introduces is the innermost.)

- For a `Var(x)` site referring to binder living in channel `c` of `L_{i-1}`:
  `T_bid[var_site][c, B_k, ℓ_out] = δ_{ℓ_out, ℓ_out_target}`
  where `B_k` is the `bid` index reflecting `x`'s lexical depth at the use site (not at the binder!), and `ℓ_out_target` removes channel `c` from the live-binder list if this is the *last* use of binder `b`; otherwise it passes through.

**Live-binder ordering, channel assignment, compaction.** At each bond `i`, the bond's `bid`-register basis is
```
{ |no_info⟩ } ∪ { |b⟩ : b ∈ L_i (live binders at bond i) }
```
with the live binders listed in **declaration order** (the order in which they entered scope during the pre-order walk). Bond dimension on the `bid` register equals `|L_i| + 1`. When a binder leaves scope (after the last `Var` referring to it has been processed), it is removed from `L_i` for *subsequent* bonds, and the basis is **compacted**: remaining binders shift down to fill the gap. Channel indices are therefore **local to each bond**, not global. The encoder maintains `L_i` as an ordered list and emits, per bond, the explicit list of `(binder_handle, channel_index)` mappings in `EncodingMeta` (sub-projects B/C/D need this to interpret the bond tensors). Two adjacent bonds typically differ by at most one entry: either a new binder is added at a `Lam` site, or a binder is removed at the site immediately after its last `Var` use.

**Bell-pair / superposition extension.** When the AST has a hole (e.g. `Var(?)`) whose binder is unknown but constrained to a candidate set `{b_1, …, b_k}`, the use site's local-`bid` state is the equal-amplitude superposition over the candidates' `B_k` indices, and the corresponding bond tensor is the standard sum-of-projections that produces a Bell-like joint state with the candidate binders. Sub-project A does not exercise this path; the algorithm is documented here so that sub-project E can implement holes without redesigning the encoder.

### 5.5 The gate-based construction (for cross-check)

An equivalent state can be constructed by:

1. Starting from the all-PAD vacuum MPS (`MPS.vacuum(N, d_local)`).
2. Applying single-site gates that "write" the local state (kind, type, value) at each non-PAD site. These are projector-like gates of the form `|target⟩⟨0|` on the relevant register, padded by identity on the rest. (In practice they are constructed as outer products and applied via `MPS.apply_local_gate`.)
3. For each (lam, var) pair: applying a sequence of **CNOT-like two-site gates** on the `bid` register that propagate the binder's `bid` value rightward along the chain to the var site. The CNOT is between consecutive sites and is conditioned on the live-binder channel; it is constructed as an explicit `8 × 8` operator on `H_bid` and embedded into the local `d_local × d_local` Hilbert space via `embed_op` from `fock.py`, then into the two-site `d_local^2 × d_local^2` operator via `two_site_op`. Applied via `MPS.apply_two_site_gate(site, gate, chi_max)`.

This is slower than the analytic construction and requires repeated SVD-truncation steps, but it is **algorithmically independent** of the analytic path: bugs that exist in §5.4 will not be replicated in §5.5 and vice versa. The cross-check test (§7.3) compares the two states by fidelity `|⟨ψ_analytic | ψ_gate⟩|^2 > 1 - 1e-8`.

### 5.6 Sparse construction note

The `d_local = 8192` figure (§4.6) makes naive tensor allocation expensive. The encoder constructs each site tensor in factored form on the four registers:

```
T[i] = T_kind[i] ⊗ T_type[i] ⊗ T_bid[i] ⊗ T_value[i]
```

where each per-register tensor has its own bond structure. For `kind`, `type`, `value` the bond is dimension 1 (product on those registers); for `bid` it carries the live-binder channels (§5.4). The factored construction is then converted to the unified `(χ_left, 8192, χ_right)` form before being placed into the `MPS.tensors` list.

Concretely, the unified tensor is:

```
T[i][ℓ_in, s, ℓ_out] = T_kind[i][s_kind] * T_type[i][s_type]
                      * T_bid[i][ℓ_in, s_bid, ℓ_out]
                      * T_value[i][s_value]
```

where `s = (s_kind, s_type, s_bid, s_value)` is the multi-index of the local 8192-dim state. Because the kind/type/value parts are product (single nonzero entry each), most of the 8192 entries are zero — exactly one slice has a nonzero contribution. The encoder builds this slice directly without materializing the full 8192-entry vector.

### 5.7 Bond dimension cap

`chi_max = 16` covers:

- up to 6 simultaneously live binders (each consuming one channel) + the no-info channel,
- additional capacity for hole superpositions (sub-project E will exercise this).

Programs that would exceed this raise `TooManyBinders`. (The architecturally clean response is to raise the cap; the test suite asserts §7's programs stay within 16.)

### 5.8 Normalization

The encoded MPS is normalized to unit norm via `MPS.normalize()` before being returned.

---

## 6. Decoder

### 6.1 API

```python
# src/qft_pcn/logic/decoder.py

@dataclass
class DecodeResult:
    ast: Node
    residual_norm: float        # 0.0 for a clean product input; > 0 if MPS state has
                                # any leakage outside the basis the decoder expected

def decode(state: MPS, meta: EncodingMeta) -> DecodeResult:
    """Deterministic maximum-likelihood decode.

    For each site, takes the argmax over local-register basis states. Suitable
    only for product (zero-entanglement on kind/type/value) inputs. For
    superposed inputs use `sample`.

    Raises:
        DecodeError: if the recovered structure does not parse as a valid AST
                     (e.g. arity mismatch, dangling PAD, etc.).
    """

def sample(state: MPS, meta: EncodingMeta, n_samples: int = 1,
           rng: np.random.Generator | None = None) -> list[DecodeResult]:
    """Sample `n_samples` ASTs from the MPS distribution.

    Uses standard left-to-right MPS conditional sampling: at each site, build
    the local reduced state *conditioned* on prior measurements, draw from
    the resulting categorical, project, advance. This is the principled
    sampler (see Schollwock 2011 §7.3); independent per-site sampling is
    explicitly NOT used (see §1.5).
    """
```

### 6.2 Decode algorithm (deterministic)

1. For each site `k = 0, 1, …, N-1`, compute the local marginal probability over the `(kind, type, bid, value)` basis by contracting the MPS environment outside that site. Concretely: bring the MPS into left-canonical form up to site `k`, then take the squared singular values of the bond. (Or, equivalently, use `MPS.local_expectation` against rank-1 projectors — slower but identical math.)
2. Argmax to obtain `kind_k`. Then conditional on `kind_k`, argmax `type`, `bid`, `value` from the appropriate sub-distributions.
3. Use `kind_k` to drive a recursive structural parse of the site stream into an AST, with PAD sites terminating subtrees that didn't need them. The structural parser is:
   ```
   parse(stream) =
       if stream empty: error
       site = stream.pop_front()
       case site.kind of:
           PAD  -> error  (caller asked for a node, got PAD)
           VAR  -> Var(name = name_for_bid(site.bid, current_scope), ty = site.type)
           LAM  -> let body = parse(stream)
                   in Lam(param = fresh_name(), param_ty = site.type.src, body = body)
           APP  -> let fn = parse(stream)
                       arg = parse(stream)
                   in App(fn, arg)
           INT  -> IntLit(value = site.value - 7)
           BOOL -> BoolLit(value = site.value == V_TRUE)
           IF   -> let c = parse(stream)
                       t = parse(stream)
                       e = parse(stream)
                   in If(c, t, e)
           BIN  -> let l = parse(stream)
                       r = parse(stream)
                   in Bin(op = op_for_value(site.value), l, r)
   ```
4. After parsing the top-level expression, remaining sites must all be `PAD` (or `DecodeError`).

**Name re-generation.** The decoder does not preserve the original identifier names — it has no way to. Instead it generates names `_v0, _v1, …` per binder during the parse and uses the `bid` register to wire `Var` uses to the right `Lam` (via a stack of in-scope `(bid_at_lam, generated_name)` pairs).

This means structural equality between input and output ASTs (§7.2) is **modulo alpha-renaming**. The test suite uses an alpha-equivalence comparator (`ast_alpha_eq(a, b)`) for the round-trip assertions.

### 6.3 Sampling algorithm (`sample`)

Standard MPS left-to-right sampling, exactly as in Stoudenmire & White (PRA 2010) / Schollwock (2011 review):

```
ψ = state.copy()
for k = 0 .. N - 1:
    # Bring ψ into mixed canonical form with the orthogonality center at site k.
    canonicalize_around(ψ, k)
    # The site k tensor is now isometric on the left bond; the on-site
    # reduced state is given by p(s) = ||A[k][:, s, :]||^2.
    local_marginal = sum over (left_bond, right_bond) of |A[k][ℓ, s, r]|^2
    s_k ~ Categorical(local_marginal)
    # Project ψ onto the measured outcome.
    A[k] = A[k][:, s_k:s_k+1, :] / sqrt(local_marginal[s_k])
# After the sweep, ψ is fully projected; read off the basis labels.
return decode_from_basis_labels(s_0, s_1, …, s_{N-1})
```

This is the *only* sampler we use; per-site independent sampling against local reduced density matrices is not used (see §1.5).

### 6.4 Residual norm

`residual_norm` reports the leakage of the input MPS outside the decoder's expected basis. For a perfectly-encoded product state it is 0. For a slightly noisy state (e.g. after evolution) it reports how far off we are. The decoder does *not* error on small residuals; it returns the best-effort AST and lets the caller decide. The acceptance tests assert residual < 1e-10 for products of `encode`.

---

## 7. Tests

Located in `src/qft_pcn/tests/test_logic_encoder.py`. Uses the existing pytest convention from the other `test_*.py` files in that directory.

### 7.1 The five round-trip programs

Programs are given in concrete syntax; the test parses them with the included parser, encodes, decodes, and asserts alpha-equivalence:

```
P1.  λx:Int. x
P2.  (λx:Int. x + 1)(2)
P3.  λf:Int→Int. λx:Int. f (f x)
P4.  if (1 < 2) then ((λx:Bool. x)(true)) else false
P5.  (λx:Int. (λy:Int. x + y)(3))(4)
```

For each Pi:

```python
def test_roundtrip_P{i}():
    p = parse(P_i_source)
    state, meta = encode(p, N=32, chi_max=16)
    assert abs(state.norm_sq() - 1.0) < 1e-10
    result = decode(state, meta)
    assert result.residual_norm < 1e-10
    assert ast_alpha_eq(result.ast, p)
```

### 7.2 Alpha-equivalence helper

```python
def ast_alpha_eq(a: Node, b: Node, env: dict | None = None) -> bool: ...
```

Walks both ASTs in parallel; maintains a bijection between bound names; literal values, op codes, types, and structural shape must match exactly.

### 7.3 Analytic vs gate-based construction cross-check

```python
def test_analytic_matches_gate_construction():
    p = parse(P5_source)                      # the most binder-heavy program
    state_a, meta = encode(p, N=32, chi_max=16)
    state_g, _   = encode(p, N=32, chi_max=16, _construction="gate")
    # encode() takes a private kwarg routing to the gate-based path
    overlap = mps_inner_product(state_a, state_g)
    fidelity = abs(overlap) ** 2
    assert fidelity > 1.0 - 1e-8
```

The `_construction="gate"` kwarg is private (underscored, not in any public API), used only by tests. The two implementations live in:

- `src/qft_pcn/logic/encoder.py` — analytic (default).
- `src/qft_pcn/logic/_gate_construction.py` — gate-based (test-only).

`mps_inner_product(a, b)` is a small helper computing `⟨a|b⟩` by sweeping environment tensors; either implemented here as a private function in the test module or added to `MPS` as a method (decision: implemented as a method `MPS.inner(other)` — sub-project B will also need it).

### 7.4 Binder bonds carry information capacity — bond-dimension structural check

This is the test that **proves the binding-is-entanglement *mechanism* is realized in code** and not just claimed in prose. For a concrete (no-hole) program the encoded MPS happens to be a *product* state on the `bid` register (both endpoints of every binder→use pair are in definite `B_k` eigenstates), so the *entanglement entropy* across binder-spanning bonds is 0 in this case. What proves the mechanism is the bond *dimension*: with the principled channel construction the bond dim on `bid` equals `|L_i| + 1`; with the shortcut (classical `bid` field at use sites, no bond coupling) the bond dim on `bid` is 1 even though `|L_i| > 0`.

```python
def test_binder_bonds_have_channel_dimension():
    """Every bond crossed by k live binders must have at least k+1 channels."""
    p = parse(P5_source)
    state, meta = encode(p, N=32, chi_max=16)
    # meta exposes per-bond live-binder counts; the encoder MUST have built
    # the bid-register bond at dimension >= |L_i| + 1.
    for i in range(len(meta.live_binders_per_bond)):
        n_live = len(meta.live_binders_per_bond[i])
        bond_dim_total = state.bond_dimensions()[i]
        # Total bond dim is at least the bid sub-bond dim, which is at least
        # n_live + 1 by construction (see §5.4). If a subagent collapsed the
        # bid bond to dimension 1, this fails with a message naming the
        # violation.
        assert bond_dim_total >= n_live + 1, (
            f"bond {i} has dim {bond_dim_total} but {n_live} binders are live "
            f"across it; the encoder collapsed the channel structure — "
            f"this is the §1.1 shortcut. Re-read the spec.")
```

In addition to the dimension check, we run a **superposition probe** that *does* test entanglement entropy by manually putting one Var into superposition over two candidate binders (this exercises the §5.4 hole path even though sub-project A doesn't expose holes at the parser level):

```python
def test_superposition_var_produces_entropy():
    """Manually constructed hole: Var with bid in superposition over two binders.
    
    Verifies the encoder's superposition branch in §5.4 is implemented (even
    though it's not parser-exposed in sub-project A). The bond across the
    span between the candidate binders and the use site must carry > 0
    entanglement entropy after encoding.
    """
    p = parse("\\x:Int. \\y:Int. ?HOLE")
    # The encoder's superposition path takes a Hole node with a candidate
    # binder list:
    hole = HoleVar(candidates=["x", "y"])     # x and y are both candidate binders
    p_with_hole = substitute_hole(p, hole)
    state, meta = encode(p_with_hole, N=32, chi_max=16)
    var_site = locate_hole_site(meta)
    bond = var_site - 1
    S = state.entanglement_entropy(bond)
    # Two equal-amplitude candidates -> entropy = log(2) on bid register.
    assert abs(S - math.log(2)) < 1e-6, (
        f"superposed-hole bond entropy = {S}, expected ~ln 2; "
        f"the §5.4 superposition branch is not entangling — "
        f"hole completions in sub-project E will be broken")
```

`HoleVar` and `substitute_hole` are minimal additions to `ast.py` to support this test — they will be needed by sub-project E anyway and adding them here avoids a future refactor. The test message in both cases is deliberate: it names the spec section being violated so a future subagent gets pointed at the right place.

### 7.5 Alpha-renaming invariance

```python
def test_alpha_renaming_is_local_unitary_on_bid():
    p1 = parse("\\x:Int. x")
    p2 = parse("\\y:Int. y")
    s1, m1 = encode(p1, N=32)
    s2, m2 = encode(p2, N=32)
    # The two states differ only by local unitaries (no unitary at all, in fact;
    # the bid scheme is depth-relative so they're identical):
    overlap = abs(s1.inner(s2)) ** 2
    assert overlap > 1.0 - 1e-10
```

### 7.6 PAD vacuum property

```python
def test_pad_sites_are_vacuum():
    p = parse(P1_source)
    state, meta = encode(p, N=32)
    # AST has 2 nodes (Lam + Var), so sites 2..31 must be PAD on every register.
    # For each register, the local expectation of the projector onto the non-PAD
    # subspace must be ~0.
    for site in range(2, 32):
        kind_proj_non_pad = build_register_projector("kind", exclude={"PAD"})
        # Embed into the full local Hilbert space.
        kind_op_full = embed_op(kind_proj_non_pad, 0, meta.species_dims)
        e = state.local_expectation(site, kind_op_full)
        assert abs(e) < 1e-10
```

### 7.7 Norm preservation, encoding size limits, error paths

- `test_norm_is_unity`: every successful encode returns a unit-norm MPS.
- `test_encoding_too_large_raises`: an AST with 33 nodes encoded into N=32 raises `EncodingTooLarge`.
- `test_too_many_binders_raises`: 8-deep nested lambdas raise `TooManyBinders`.
- `test_int_literal_range`: `IntLit(9)` raises `IntLiteralOutOfRange`.
- `test_ill_scoped_var`: `Var("undefined")` raises `IllScopedVar`.
- `test_unsupported_node`: a node type not in §3's grammar raises `UnsupportedNode`.

### 7.8 Performance budget

```python
@pytest.mark.timeout(5)
def test_encode_decode_performance_budget():
    # Encode and decode P5 100 times in under 5 seconds total.
    p = parse(P5_source)
    for _ in range(100):
        state, meta = encode(p, N=32, chi_max=16)
        decode(state, meta)
```

If this test fails, the sparse construction (§5.6) was not implemented properly. Do not relax the budget without a written justification reviewed by the human.

---

## 8. File layout

```
src/qft_pcn/logic/
├── __init__.py                  # public re-exports: encode, decode, sample,
│                                #   Node, Ty, EncodingMeta, exception types
├── ast.py                       # Node hierarchy, Ty hierarchy, parser, pretty-printer
├── encoding.py                  # constants: KIND_*, TYPE_*, BID_*, VALUE_*,
│                                #   SPECIES, EncodingMeta dataclass
├── encoder.py                   # encode(), _resolve_binders(), _analytic_build()
├── decoder.py                   # decode(), sample(), ast_alpha_eq()
└── _gate_construction.py        # _gate_build() — test-only equivalent of encoder
src/qft_pcn/tests/
└── test_logic_encoder.py        # all tests from §7
```

Public exports from `src/qft_pcn/logic/__init__.py`:

```python
from .ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin,
    HoleVar, substitute_hole,
    Ty, TInt, TBool, TArrow,
    parse, pretty,
)
from .encoding import (
    SPECIES, EncodingMeta,
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL, KIND_IF, KIND_BIN,
    TYPE_NONE, TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI,
    TYPE_ARR_BB, TYPE_ARR_NESTED,
    BID_NONE, BID_0, BID_1, BID_2, BID_3, BID_4, BID_5, BID_6,
)
from .encoder import encode
from .decoder import decode, sample, DecodeResult, ast_alpha_eq
```

No changes to any existing file *except*:

- `src/qft_pcn/qft/mps.py`: add `def inner(self, other: "MPS") -> complex` for the cross-check tests. Add a unit test for it in `test_qft.py`.
- `src/qft_pcn/__init__.py`: re-export the `logic.*` public names so callers can `from qft_pcn import encode`.

---

## 9. Error model

Custom exceptions live in `src/qft_pcn/logic/encoding.py`:

```python
class EncodingError(Exception):
    """Base class for encoder/decoder errors."""

class EncodingTooLarge(EncodingError):
    def __init__(self, n_nodes: int, N: int):
        self.n_nodes, self.N = n_nodes, N
        super().__init__(f"AST has {n_nodes} nodes but N={N}")

class TooManyBinders(EncodingError):
    def __init__(self, depth: int, cutoff: int):
        super().__init__(f"scope nesting {depth} exceeds bid cutoff {cutoff}")

class IntLiteralOutOfRange(EncodingError):
    def __init__(self, n: int):
        super().__init__(f"IntLit({n}) outside [-7, 8]")

class IllScopedVar(EncodingError):
    def __init__(self, name: str):
        super().__init__(f"Var({name!r}) not in lexical scope")

class UnsupportedNode(EncodingError):
    def __init__(self, node_type: str):
        super().__init__(f"node type {node_type} not in supported grammar")

class DecodeError(EncodingError): ...
```

Encoder errors are raised eagerly during the pre-order walk, with line/path info from `meta.site_to_ast_path` in the message where applicable.

---

## 10. Acceptance criteria

The sub-project is complete when:

1. The five round-trip tests in §7.1 pass.
2. The cross-check test in §7.3 passes (fidelity > 1 - 1e-8).
3. The bond-dimension test in §7.4 (`test_binder_bonds_have_channel_dimension`) passes.
4. The superposition-probe test in §7.4 (`test_superposition_var_produces_entropy`) passes (S ≈ ln 2).
5. The alpha-renaming test in §7.5 passes (overlap > 1 - 1e-10).
6. The PAD-vacuum test in §7.6 passes.
7. All five error-path tests in §7.7 raise the right exception.
8. The performance budget test in §7.8 passes.
9. All previously passing tests in `src/qft_pcn/tests/` still pass.
10. `MPS.inner` is implemented and unit-tested.
11. `HoleVar` and `substitute_hole` are implemented (minimal — sub-project E will extend them).

No claim of completion is acceptable without these tests actually running green in a fresh shell. (See `superpowers:verification-before-completion`.)

---

## 11. Open questions

**None.** Every design choice above is locked. If you, while implementing, find a real ambiguity that this document does not resolve — stop and ask the human. Do not paper over it.

---

## 12. Where this fits in the larger arc

This sub-project produces the data structure that sub-projects B–E will operate on. The contract for them is:

- B (typing-rule Hamiltonian compiler) consumes `EncodingMeta.species`, builds local and two-site Hamiltonian terms acting on `kind`/`type`/`bid` registers, such that `⟨H_typing⟩ = 0` exactly when the AST is well-typed.
- C (evaluation Hamiltonian) adds terms acting on `kind`/`value` such that the ground state of `H_typing + H_eval` is a well-typed *and* fully-reduced program.
- D (debugger) reads per-term `⟨H_term⟩` and emits structured error reports.
- E (STLC synthesis demo) takes an AST with holes (Var sites whose `bid` is in superposition, or Type sites in superposition over `T_*`), encodes via this sub-project's `encode()` extended with the §5.4 superposition path, runs imaginary-time evolution under `H_typing + H_eval`, samples completions via `sample()`.

Failing to realize §1.1 here will silently break E — the holes will collapse to whatever the shortcut wrote. That is the practical reason §1.1 is non-negotiable.

---

## 13. Glossary (local)

- **Site / lattice position** — one of the N positions in the MPS, indexed 0 … N-1.
- **Register / field species** — one of the four tensor factors of the local Hilbert space (`kind`, `type`, `bid`, `value`).
- **Bond** — the connection between adjacent sites in the MPS, with its own dimension `χ`.
- **Live binders at a bond** — the set of binders declared on the left side of the bond that have at least one variable use on the right side.
- **Channel** — one basis state of the `bid` bond register; carries one live binder's identity through the bond.
- **PAD site** — a site in the all-PAD local state; the encoder writes these to fill the lattice beyond the AST.
- **Analytic construction** — building the MPS site tensors in closed form (§5.4).
- **Gate construction** — building the MPS state by applying a sequence of unitary gates to the vacuum (§5.5).
